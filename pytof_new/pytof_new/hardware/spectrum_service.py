"""Persistent hardware-owner service for a real Spectrum digitizer.

One instance lives in a dedicated QThread and serialises all driver calls.
"""

from __future__ import annotations

import logging
from enum import Enum

from PySide6 import QtCore

from pytof_new.exceptions import DigitizerError
from pytof_new.hardware.spectrum_digitizer import SpectrumDigitizer
from pytof_new.hardware.spectrum_driver import SpectrumDriverApi
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionRequest,
    SpectrumAcquisitionResult,
    SpectrumHardwareInfo,
)

LOGGER = logging.getLogger(__name__)


class ServiceState(str, Enum):
    IDLE = "idle"
    CONNECTED = "connected"
    CONFIGURED = "configured"
    PREPARED = "prepared"
    ARMED = "armed"
    ACQUIRING = "acquiring"
    ERROR = "error"


class SpectrumAcquisitionService(QtCore.QObject):
    """Own one Spectrum digitizer handle; process jobs on its owner thread."""

    state_changed = QtCore.Signal(str)
    hardware_info_ready = QtCore.Signal(object)
    configured = QtCore.Signal()
    prepared = QtCore.Signal(object)  # SpectrumAcquisitionPlan
    armed = QtCore.Signal()
    result_ready = QtCore.Signal(object)  # SpectrumAcquisitionResult
    error_occurred = QtCore.Signal(str)
    log_message = QtCore.Signal(str)

    def __init__(self, api: SpectrumDriverApi | None = None, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._digitizer = SpectrumDigitizer(api=api or SpectrumDriverApi())
        self._state: ServiceState = ServiceState.IDLE
        self._abort_requested = False

    @property
    def state(self) -> ServiceState:
        return self._state

    def _set_state(self, value: ServiceState) -> None:
        self._state = value
        self.state_changed.emit(value.value)

    @QtCore.Slot()
    def connect_card(self) -> None:
        """Open the digitizer and discover hardware info once."""
        if self._state != ServiceState.IDLE:
            self.error_occurred.emit(f"connect_card cannot be called in state {self._state.value}")
            return
        try:
            self._digitizer.connect()
            info = self._digitizer.hardware_info or SpectrumHardwareInfo()
            self._set_state(ServiceState.CONNECTED)
            self.hardware_info_ready.emit(info)
            self.log_message.emit("Spectrum digitizer connected")
        except Exception as exc:
            self._set_state(ServiceState.ERROR)
            self.error_occurred.emit(str(exc))
            LOGGER.exception("Spectrum connect failed")

    @QtCore.Slot(object)
    def configure(self, request: SpectrumAcquisitionRequest) -> None:
        """Validate and apply an acquisition configuration."""
        if self._state not in (ServiceState.CONNECTED, ServiceState.CONFIGURED):
            self.error_occurred.emit(f"configure cannot be called in state {self._state.value}")
            return
        try:
            self._digitizer.configure_request(request)
            self._set_state(ServiceState.CONFIGURED)
            self.configured.emit()
            self.log_message.emit("Spectrum digitizer configured")
        except Exception as exc:
            self._set_state(ServiceState.ERROR)
            self.error_occurred.emit(str(exc))
            LOGGER.exception("Spectrum configure failed")

    @QtCore.Slot()
    def acquire(self) -> None:
        """Run one finite acquisition.

        Blocks on the service thread during WAITREADY.  Call ``abort()`` from
        any thread to cancel.
        """
        if self._state != ServiceState.CONFIGURED:
            self.error_occurred.emit(f"acquire cannot be called in state {self._state.value}")
            return
        self._abort_requested = False
        self._set_state(ServiceState.ACQUIRING)
        self.log_message.emit("Spectrum acquisition started")
        try:
            result = self._digitizer.acquire_configured()
            if self._abort_requested:
                self.log_message.emit("Spectrum acquisition aborted")
                self._cleanup_after_acquire()
                return
            self.log_message.emit("Spectrum acquisition completed")
            self._set_state(ServiceState.CONFIGURED)
            self.result_ready.emit(result)
        except Exception as exc:
            LOGGER.exception("Spectrum acquire failed")
            self._cleanup_after_acquire()
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def prepare(self) -> None:
        """Define the DMA buffer for one configured finite acquisition."""
        if self._state != ServiceState.CONFIGURED:
            self.error_occurred.emit(f"prepare cannot be called in state {self._state.value}")
            return
        try:
            plan = self._digitizer.prepare_configured_acquisition()
            self._set_state(ServiceState.PREPARED)
            self.prepared.emit(plan)
            self.log_message.emit("Spectrum acquisition prepared")
        except Exception as exc:
            LOGGER.exception("Spectrum prepare failed")
            self._cleanup_after_acquire()
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def start_prepared(self) -> None:
        """Start Spectrum trigger/DMA engines without waiting for completion."""
        if self._state != ServiceState.PREPARED:
            self.error_occurred.emit(f"start_prepared cannot be called in state {self._state.value}")
            return
        self._abort_requested = False
        try:
            self._digitizer.start_prepared_acquisition()
            self._set_state(ServiceState.ARMED)
            self.armed.emit()
            self.log_message.emit("Spectrum acquisition armed and waiting for triggers")
        except Exception as exc:
            LOGGER.exception("Spectrum start prepared failed")
            self._cleanup_after_acquire()
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def wait_result(self) -> None:
        """Wait for a previously started acquisition and emit its result."""
        if self._state != ServiceState.ARMED:
            self.error_occurred.emit(f"wait_result cannot be called in state {self._state.value}")
            return
        self._set_state(ServiceState.ACQUIRING)
        try:
            result = self._digitizer.wait_for_prepared_result()
            if self._abort_requested:
                self.log_message.emit("Spectrum acquisition aborted")
                self._cleanup_after_acquire()
                return
            self._digitizer.stop()
            self._set_state(ServiceState.CONFIGURED)
            self.log_message.emit("Spectrum acquisition completed")
            self.result_ready.emit(result)
        except Exception as exc:
            LOGGER.exception("Spectrum wait result failed")
            self._cleanup_after_acquire()
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def abort(self) -> None:
        """Request cancellation of a running acquisition.

        This is safe to call from any thread.  The stop command is issued
        on the current thread; WAITREADY on the service thread returns
        promptly with an error.
        """
        if self._state not in (ServiceState.PREPARED, ServiceState.ARMED, ServiceState.ACQUIRING):
            return
        self._abort_requested = True
        try:
            self._digitizer.stop()
            self.log_message.emit("Acquisition abort requested")
        except Exception:
            LOGGER.exception("abort stop failed")

    def recover(self) -> bool:
        """Attempt to recover from an error state without closing the card.

        Recovery ladder:
          1. Read/clear driver error text (best-effort)
          2. Issue card stop
          3. Return to CONNECTED (or ERROR if card still unreachable)
          4. Caller should re-apply configuration via ``configure()``.

        Returns True if the card was successfully stopped.
        """
        if self._digitizer.handle is None:
            self._set_state(ServiceState.ERROR)
            return False
        try:
            self._digitizer.api.read_error_text(self._digitizer.handle)
        except Exception:
            pass
        try:
            self._digitizer.stop()
        except Exception:
            try:
                self._digitizer.close()
            except Exception:
                pass
            self._digitizer = SpectrumDigitizer(api=self._digitizer.api)
            self._set_state(ServiceState.IDLE)
            self.log_message.emit("Card reset needed — handle closed after recovery failure")
            return False
        self._set_state(ServiceState.CONNECTED)
        self.log_message.emit("Card recovered and returned to CONNECTED state")
        return True

    @QtCore.Slot()
    def disconnect_card(self) -> None:
        """Close the digitizer and return to idle."""
        if self._state in (ServiceState.IDLE, ServiceState.ERROR):
            self._digitizer.close()
            self._set_state(ServiceState.IDLE)
            return
        try:
            self._digitizer.close()
        except Exception as exc:
            LOGGER.exception("Spectrum disconnect failed")
            self.error_occurred.emit(str(exc))
        self._set_state(ServiceState.IDLE)
        self.log_message.emit("Spectrum digitizer disconnected")

    def _cleanup_after_acquire(self) -> None:
        if self._digitizer.handle is not None:
            try:
                self._digitizer.stop()
            except Exception:
                pass
        self._set_state(ServiceState.CONFIGURED if self._digitizer.acquisition_plan is not None else ServiceState.CONNECTED)
