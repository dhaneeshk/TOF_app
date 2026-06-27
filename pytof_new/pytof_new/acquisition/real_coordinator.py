"""Coordinate one finite real Spectrum+BME hardware batch.

The coordinator talks to the persistent hardware services only through Qt
signals/slots.  That preserves the service ownership model: Spectrum and BME
DLL calls still execute on their respective service threads when the signals are
connected with queued delivery in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging

from PySide6 import QtCore

from pytof_new.acquisition.trigger_counts import required_bme_triggers_for_plan
from pytof_new.hardware.spectrum_models import SpectrumAcquisitionPlan, SpectrumAcquisitionResult

LOGGER = logging.getLogger(__name__)


class RealBatchCoordinatorState(str, Enum):
    """Coordinator state for one finite synchronized hardware batch."""

    IDLE = "idle"
    STOPPING_BME = "stopping_bme"
    ARMING_BME = "arming_bme"
    PREPARING_SPECTRUM = "preparing_spectrum"
    STARTING_SPECTRUM = "starting_spectrum"
    STARTING_BME = "starting_bme"
    WAITING_SPECTRUM = "waiting_spectrum"
    READING_BME_COUNT = "reading_bme_count"
    READING_BME_STATUS = "reading_bme_status"
    FINAL_STOPPING_BME = "final_stopping_bme"
    FINISHED = "finished"
    ERROR = "error"


@dataclass(frozen=True)
class CoordinatedBatchMetadata:
    """Synchronization metadata produced after one coordinated batch."""

    expected_bme_trigger_count: int
    actual_bme_trigger_count: int
    bme_status: int
    expected_spectrum_records: int
    actual_spectrum_records: int


class RealBatchCoordinator(QtCore.QObject):
    """Signal-driven state machine for one finite Spectrum+BME batch."""

    state_changed = QtCore.Signal(str)
    result_ready = QtCore.Signal(object)  # SpectrumAcquisitionResult with BME metadata
    error_occurred = QtCore.Signal(str)
    log_message = QtCore.Signal(str)
    finished = QtCore.Signal()

    _request_bme_stop = QtCore.Signal()
    _request_bme_arm = QtCore.Signal(int)
    _request_bme_start = QtCore.Signal()
    _request_bme_read_count = QtCore.Signal()
    _request_bme_read_status = QtCore.Signal()
    _request_bme_emergency_stop = QtCore.Signal()
    _request_spectrum_prepare = QtCore.Signal()
    _request_spectrum_start = QtCore.Signal()
    _request_spectrum_wait = QtCore.Signal()
    _request_spectrum_abort = QtCore.Signal()

    def __init__(self, spectrum_service: QtCore.QObject, bme_service: QtCore.QObject, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._spectrum = spectrum_service
        self._bme = bme_service
        self._state = RealBatchCoordinatorState.IDLE
        self._expected_bme_count: int | None = None
        self._spectrum_result: SpectrumAcquisitionResult | None = None
        self._bme_count: int | None = None
        self._bme_status: int | None = None
        self._connect_services()

    @property
    def state(self) -> RealBatchCoordinatorState:
        """Return current coordinator state."""
        return self._state

    @QtCore.Slot(int)
    def start_batch(self, expected_bme_trigger_count: int) -> None:
        """Start one finite coordinated batch."""
        if self._state not in (RealBatchCoordinatorState.IDLE, RealBatchCoordinatorState.FINISHED):
            self._fail(f"start_batch cannot be called in state {self._state.value}")
            return
        if expected_bme_trigger_count <= 0:
            self._fail("expected_bme_trigger_count must be positive")
            return
        self._expected_bme_count = int(expected_bme_trigger_count)
        self._spectrum_result = None
        self._bme_count = None
        self._bme_status = None
        self._set_state(RealBatchCoordinatorState.STOPPING_BME)
        self.log_message.emit("Ensuring BME is inactive before coordinated batch")
        self._request_bme_stop.emit()

    @QtCore.Slot(object)
    def start_plan_batch(self, plan: SpectrumAcquisitionPlan) -> None:
        """Start one batch using the BME trigger count derived from a Spectrum plan."""
        self.start_batch(required_bme_triggers_for_plan(plan))

    @QtCore.Slot()
    def request_stop(self) -> None:
        """Abort both hardware services for errors or user stop."""
        if self._state in (RealBatchCoordinatorState.IDLE, RealBatchCoordinatorState.FINISHED):
            return
        self._request_spectrum_abort.emit()
        self._request_bme_emergency_stop.emit()
        self._set_state(RealBatchCoordinatorState.ERROR)
        self.finished.emit()

    def _connect_services(self) -> None:
        self._request_bme_stop.connect(self._bme.stop)
        self._request_bme_arm.connect(self._bme.arm)
        self._request_bme_start.connect(self._bme.start)
        self._request_bme_read_count.connect(self._bme.read_trigger_count)
        self._request_bme_read_status.connect(self._bme.read_status)
        self._request_bme_emergency_stop.connect(self._bme.emergency_stop)
        self._request_spectrum_prepare.connect(self._spectrum.prepare)
        self._request_spectrum_start.connect(self._spectrum.start_prepared)
        self._request_spectrum_wait.connect(self._spectrum.wait_result)
        self._request_spectrum_abort.connect(self._spectrum.abort)

        self._bme.stopped.connect(self._on_bme_stopped)
        self._bme.armed.connect(self._on_bme_armed)
        self._bme.started.connect(self._on_bme_started)
        self._bme.trigger_count_ready.connect(self._on_bme_trigger_count)
        self._bme.status_ready.connect(self._on_bme_status)
        self._bme.error_occurred.connect(self._on_bme_error)

        self._spectrum.prepared.connect(self._on_spectrum_prepared)
        self._spectrum.armed.connect(self._on_spectrum_armed)
        self._spectrum.result_ready.connect(self._on_spectrum_result)
        self._spectrum.error_occurred.connect(self._on_spectrum_error)

    def _set_state(self, value: RealBatchCoordinatorState) -> None:
        self._state = value
        self.state_changed.emit(value.value)

    @QtCore.Slot()
    def _on_bme_stopped(self) -> None:
        if self._state == RealBatchCoordinatorState.STOPPING_BME:
            self._set_state(RealBatchCoordinatorState.ARMING_BME)
            self._request_bme_arm.emit(int(self._expected_bme_count or 0))
        elif self._state == RealBatchCoordinatorState.FINAL_STOPPING_BME:
            self._complete_or_fail()

    @QtCore.Slot(int)
    def _on_bme_armed(self, _expected_count: int) -> None:
        if self._state != RealBatchCoordinatorState.ARMING_BME:
            return
        self._set_state(RealBatchCoordinatorState.PREPARING_SPECTRUM)
        self._request_spectrum_prepare.emit()

    @QtCore.Slot(object)
    def _on_spectrum_prepared(self, _plan: SpectrumAcquisitionPlan) -> None:
        if self._state != RealBatchCoordinatorState.PREPARING_SPECTRUM:
            return
        self._set_state(RealBatchCoordinatorState.STARTING_SPECTRUM)
        self._request_spectrum_start.emit()

    @QtCore.Slot()
    def _on_spectrum_armed(self) -> None:
        if self._state != RealBatchCoordinatorState.STARTING_SPECTRUM:
            return
        self._set_state(RealBatchCoordinatorState.STARTING_BME)
        self.log_message.emit("Spectrum is armed; starting BME trigger sequence")
        self._request_bme_start.emit()

    @QtCore.Slot()
    def _on_bme_started(self) -> None:
        if self._state != RealBatchCoordinatorState.STARTING_BME:
            return
        self._set_state(RealBatchCoordinatorState.WAITING_SPECTRUM)
        self._request_spectrum_wait.emit()

    @QtCore.Slot(object)
    def _on_spectrum_result(self, result: SpectrumAcquisitionResult) -> None:
        if self._state != RealBatchCoordinatorState.WAITING_SPECTRUM:
            return
        self._spectrum_result = result
        self._set_state(RealBatchCoordinatorState.READING_BME_COUNT)
        self._request_bme_read_count.emit()

    @QtCore.Slot(int)
    def _on_bme_trigger_count(self, count: int) -> None:
        if self._state != RealBatchCoordinatorState.READING_BME_COUNT:
            return
        self._bme_count = int(count)
        self._set_state(RealBatchCoordinatorState.READING_BME_STATUS)
        self._request_bme_read_status.emit()

    @QtCore.Slot(int)
    def _on_bme_status(self, status: int) -> None:
        if self._state != RealBatchCoordinatorState.READING_BME_STATUS:
            return
        self._bme_status = int(status)
        self._set_state(RealBatchCoordinatorState.FINAL_STOPPING_BME)
        self._request_bme_stop.emit()

    @QtCore.Slot(str)
    def _on_bme_error(self, message: str) -> None:
        self._request_spectrum_abort.emit()
        self._fail(f"BME service error: {message}")

    @QtCore.Slot(str)
    def _on_spectrum_error(self, message: str) -> None:
        self._request_bme_emergency_stop.emit()
        self._fail(f"Spectrum service error: {message}")

    def _complete_or_fail(self) -> None:
        if self._spectrum_result is None or self._expected_bme_count is None or self._bme_count is None or self._bme_status is None:
            self._fail("coordinated batch finished with incomplete metadata")
            return
        expected_records = int(self._spectrum_result.plan.output_shape[0])
        actual_records = int(self._spectrum_result.data.shape[0])
        if self._bme_count != self._expected_bme_count:
            self._fail(f"BME trigger count mismatch: expected {self._expected_bme_count}, got {self._bme_count}")
            return
        if actual_records != expected_records:
            self._fail(f"Spectrum record count mismatch: expected {expected_records}, got {actual_records}")
            return

        metadata = CoordinatedBatchMetadata(
            expected_bme_trigger_count=self._expected_bme_count,
            actual_bme_trigger_count=self._bme_count,
            bme_status=self._bme_status,
            expected_spectrum_records=expected_records,
            actual_spectrum_records=actual_records,
        )
        result_metadata = dict(self._spectrum_result.metadata)
        result_metadata.update(
            {
                "bme_expected_trigger_count": metadata.expected_bme_trigger_count,
                "bme_actual_trigger_count": metadata.actual_bme_trigger_count,
                "bme_status": metadata.bme_status,
                "spectrum_expected_records": metadata.expected_spectrum_records,
                "spectrum_actual_records": metadata.actual_spectrum_records,
            }
        )
        result = SpectrumAcquisitionResult(data=self._spectrum_result.data, plan=self._spectrum_result.plan, metadata=result_metadata)
        self._set_state(RealBatchCoordinatorState.FINISHED)
        self.result_ready.emit(result)
        self.finished.emit()

    def _fail(self, message: str) -> None:
        LOGGER.error(message)
        self._set_state(RealBatchCoordinatorState.ERROR)
        self.error_occurred.emit(message)
        self.finished.emit()
