"""Persistent hardware-owner service for a real BME delay generator.

One instance should live in a dedicated ``QThread``.  All BME DLL calls are
exposed as Qt slots so production code can invoke them through queued signal
connections or queued ``QMetaObject.invokeMethod`` calls.
"""

from __future__ import annotations

import logging
from enum import Enum

from PySide6 import QtCore

from pytof_new.config.models import BMEConfig
from pytof_new.hardware.bme_delay_generator import BMEDelayGenerator
from pytof_new.hardware.bme_driver import BMEDriverApi

LOGGER = logging.getLogger(__name__)


class BMEServiceState(str, Enum):
    """User-visible BME service state."""

    IDLE = "idle"
    CONNECTED = "connected"
    CONFIGURED = "configured"
    ARMED = "armed"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class BMEDelayGeneratorService(QtCore.QObject):
    """Own one BME delay-generator session on its QObject thread."""

    state_changed = QtCore.Signal(str)
    hardware_info_ready = QtCore.Signal(object)
    configured = QtCore.Signal()
    armed = QtCore.Signal(int)
    started = QtCore.Signal()
    stopped = QtCore.Signal()
    trigger_count_ready = QtCore.Signal(int)
    status_ready = QtCore.Signal(int)
    error_occurred = QtCore.Signal(str)
    log_message = QtCore.Signal(str)

    def __init__(
        self,
        api: BMEDriverApi | None = None,
        delay_generator: BMEDelayGenerator | None = None,
        card_index: int = 0,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._delay = delay_generator or BMEDelayGenerator(api=api or BMEDriverApi(), card_index=card_index)
        self._state = BMEServiceState.IDLE

    @property
    def state(self) -> BMEServiceState:
        """Return the current service state."""
        return self._state

    def _set_state(self, value: BMEServiceState) -> None:
        self._state = value
        self.state_changed.emit(value.value)

    @QtCore.Slot()
    def connect_card(self) -> None:
        """Connect and initialize the BME card without generating pulses."""
        if self._state != BMEServiceState.IDLE:
            self.error_occurred.emit(f"connect_card cannot be called in state {self._state.value}")
            return
        try:
            self._delay.connect()
            self._set_state(BMEServiceState.CONNECTED)
            self.hardware_info_ready.emit(self._delay.info)
            self.log_message.emit("BME delay generator connected")
        except Exception as exc:
            LOGGER.exception("BME connect failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot(object)
    def configure(self, config: BMEConfig) -> None:
        """Validate and apply BME timing configuration when supported."""
        if self._state not in (BMEServiceState.CONNECTED, BMEServiceState.CONFIGURED):
            self.error_occurred.emit(f"configure cannot be called in state {self._state.value}")
            return
        try:
            self._delay.configure(config)
            self._set_state(BMEServiceState.CONFIGURED)
            self.configured.emit()
            self.log_message.emit("BME delay generator configured")
        except Exception as exc:
            LOGGER.exception("BME configure failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot(int)
    def arm(self, expected_trigger_count: int) -> None:
        """Reset the event counter for a finite trigger sequence."""
        if self._state not in (BMEServiceState.CONNECTED, BMEServiceState.CONFIGURED, BMEServiceState.ARMED):
            self.error_occurred.emit(f"arm cannot be called in state {self._state.value}")
            return
        try:
            self._delay.arm(expected_trigger_count)
            self._set_state(BMEServiceState.ARMED)
            self.armed.emit(int(expected_trigger_count))
            self.log_message.emit(f"BME armed for {expected_trigger_count:,} trigger events")
        except Exception as exc:
            LOGGER.exception("BME arm failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def start(self) -> None:
        """Start physical BME output generation when implemented."""
        if self._state != BMEServiceState.ARMED:
            self.error_occurred.emit(f"start cannot be called in state {self._state.value}")
            return
        try:
            self._delay.start()
            self._set_state(BMEServiceState.RUNNING)
            self.started.emit()
            self.log_message.emit("BME trigger generation started")
        except Exception as exc:
            LOGGER.exception("BME start failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def stop(self) -> None:
        """Gracefully stop/deactivate the BME card."""
        if self._state in (BMEServiceState.IDLE, BMEServiceState.ERROR):
            return
        previous = self._state
        self._set_state(BMEServiceState.STOPPING)
        try:
            self._delay.stop()
            self._set_state(BMEServiceState.CONFIGURED if previous in (BMEServiceState.CONFIGURED, BMEServiceState.ARMED, BMEServiceState.RUNNING) else BMEServiceState.CONNECTED)
            self.stopped.emit()
            self.log_message.emit("BME delay generator stopped")
        except Exception as exc:
            LOGGER.exception("BME stop failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def emergency_stop(self) -> None:
        """Immediately deactivate BME outputs, logging any failure."""
        try:
            self._delay.emergency_stop()
            if self._state not in (BMEServiceState.IDLE, BMEServiceState.ERROR):
                self._set_state(BMEServiceState.CONNECTED)
            self.stopped.emit()
            self.log_message.emit("BME emergency stop requested")
        except Exception as exc:
            LOGGER.exception("BME emergency stop failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def read_trigger_count(self) -> None:
        """Read and emit the BME event counter."""
        if self._state == BMEServiceState.IDLE:
            self.error_occurred.emit("read_trigger_count cannot be called while idle")
            return
        try:
            self.trigger_count_ready.emit(self._delay.read_trigger_count())
        except Exception as exc:
            LOGGER.exception("BME trigger-count read failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def read_status(self) -> None:
        """Read and emit the raw BME status register."""
        if self._state == BMEServiceState.IDLE:
            self.error_occurred.emit("read_status cannot be called while idle")
            return
        try:
            self.status_ready.emit(self._delay.read_status())
        except Exception as exc:
            LOGGER.exception("BME status read failed")
            self._set_state(BMEServiceState.ERROR)
            self.error_occurred.emit(str(exc))

    @QtCore.Slot()
    def disconnect_card(self) -> None:
        """Close the BME session and return to idle."""
        try:
            self._delay.close()
        except Exception as exc:
            LOGGER.exception("BME disconnect failed")
            self.error_occurred.emit(str(exc))
        self._set_state(BMEServiceState.IDLE)
        self.log_message.emit("BME delay generator disconnected")
