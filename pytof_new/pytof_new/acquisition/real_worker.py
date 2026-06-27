"""Event-driven acquisition workers for the real Spectrum path.

These workers use a persistent ``SpectrumAcquisitionService`` instead of
creating an ``AcquisitionController`` directly.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any

import numpy as np
from PySide6 import QtCore

from pytof_new.acquisition.real_coordinator import RealBatchCoordinator, RealBatchCoordinatorState
from pytof_new.acquisition.trigger_counts import required_bme_triggers_for_request
from pytof_new.config.models import RunConfig
from pytof_new.hardware.acquisition_planner import AcquisitionRunPlan
from pytof_new.hardware.spectrum_converter import spectrum_result_to_batch
from pytof_new.hardware.spectrum_models import SpectrumAcquisitionResult
from pytof_new.hardware.spectrum_service import ServiceState, SpectrumAcquisitionService
from pytof_new.processing.filtering import smooth_savgol
from pytof_new.processing.jitter import analyze_jitter, time_aligned_average
from pytof_new.processing.pipeline import process_batch
from pytof_new.processing.quality import RejectionSummary

LOGGER = logging.getLogger(__name__)


class RealAcquisitionWorker(QtCore.QObject):
    """Acquire one or more batches through a persistent Spectrum service."""

    state_changed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    batch_ready = QtCore.Signal(object, object)
    warning = QtCore.Signal(str)
    error = QtCore.Signal(str)
    log_message = QtCore.Signal(str)
    finished = QtCore.Signal()
    _request_acquire = QtCore.Signal()

    def __init__(
        self,
        service: SpectrumAcquisitionService,
        config: RunConfig,
        run_plan: AcquisitionRunPlan | None = None,
        coordinator: RealBatchCoordinator | None = None,
        continuous: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._config = config
        self._run_plan = run_plan
        self._coordinator = coordinator
        self._continuous = continuous
        self._current_request = run_plan.primary_request if run_plan is not None else None
        self._stop_requested = False
        self._finished = False
        self._batch_count = 0
        self._total_segments = 0
        self._pending_sum: np.ndarray | None = None
        self._pending_count = 0
        self._latest_processed: Any = None
        self._last_display_emit = 0.0

        if self._coordinator is None:
            self._request_acquire.connect(self._service.acquire)
            self._service.result_ready.connect(self._on_result)
            self._service.error_occurred.connect(self._on_service_error)
        else:
            self._request_acquire.connect(self._request_coordinated_acquire)
            self._coordinator.result_ready.connect(self._on_result)
            self._coordinator.error_occurred.connect(self._on_service_error)
            self._coordinator.log_message.connect(self.log_message)

    @QtCore.Slot()
    def run(self) -> None:
        """Start the first acquisition.  Called when the worker thread starts."""
        self._batch_count = 0
        self._total_segments = 0
        self._pending_sum = None
        self._pending_count = 0
        self._latest_processed = None
        self._last_display_emit = 0.0
        self._request_next()

    @QtCore.Slot()
    def request_stop(self) -> None:
        """Stop after the current batch completes."""
        self._stop_requested = True
        if self._coordinator is not None:
            self._coordinator.request_stop()
            if self._coordinator.state not in (
                RealBatchCoordinatorState.STOPPING_BME,
                RealBatchCoordinatorState.ARMING_BME,
                RealBatchCoordinatorState.PREPARING_SPECTRUM,
                RealBatchCoordinatorState.STARTING_SPECTRUM,
                RealBatchCoordinatorState.STARTING_BME,
                RealBatchCoordinatorState.WAITING_SPECTRUM,
                RealBatchCoordinatorState.READING_BME_COUNT,
                RealBatchCoordinatorState.READING_BME_STATUS,
                RealBatchCoordinatorState.FINAL_STOPPING_BME,
            ):
                self._finish()
            return
        self._service.abort()
        if self._service.state != ServiceState.ACQUIRING:
            self._finish()

    @QtCore.Slot(object)
    def _on_result(self, result: SpectrumAcquisitionResult) -> None:
        if self._stop_requested:
            self._finish()
            return
        try:
            plan = result.plan
            request = plan.request
            batch = spectrum_result_to_batch(
                result,
                sample_rate_hz=request.sample_rate_hz,
                pretrigger_samples=request.pretrigger_samples,
                first_trigger_index=self._total_segments,
            )
            processed = process_batch(batch, self._config.digitizer, self._config.processing)
            self._latest_processed = processed
            self._batch_count += 1
            batch_segments = batch.raw_adc.shape[0]
            self._total_segments += batch_segments

            batch_sum = processed.accepted_baseline_corrected_segments.sum(axis=0, dtype=np.float64)
            if self._pending_sum is None:
                self._pending_sum = np.zeros_like(batch_sum, dtype=np.float64)
            self._pending_sum += batch_sum
            self._pending_count += processed.accepted_count
            self.progress.emit(self._total_segments if not plan.is_fpga_sum else self._total_segments * request.averages_per_segment, 0)

            import time
            now = time.monotonic()
            target = self._run_plan.requested_display_interval_s if self._run_plan is not None else None
            should_emit = not self._continuous or target is None or self._last_display_emit == 0.0 or now - self._last_display_emit >= target
            if should_emit:
                self.batch_ready.emit(batch, _display_processed(self._latest_processed, self._pending_sum, self._pending_count))
                self._pending_sum = None
                self._pending_count = 0
                self._last_display_emit = now

            if not self._continuous:
                if self._run_plan is not None and self._run_plan.total_requested_shots is not None:
                    target = self._run_plan.total_requested_shots
                    if self._total_segments < target:
                        remaining = target - self._total_segments
                        if self._run_plan.final_batch_request is not None and remaining == self._run_plan.final_batch_shots:
                            self._current_request = self._run_plan.final_batch_request
                            self._service.configure(self._run_plan.final_batch_request)
                        self._request_next()
                    else:
                        self._finish()
                else:
                    self._finish()
            else:
                QtCore.QTimer.singleShot(0, self._request_next)
        except Exception as exc:
            LOGGER.exception("Real acquisition batch processing failed")
            self.error.emit(f"Batch processing error: {exc}")
            self._finish()

    @QtCore.Slot(str)
    def _on_service_error(self, message: str) -> None:
        self._stop_requested = True
        self.error.emit(f"Spectrum service error: {message}")
        self._finish()

    def _request_next(self) -> None:
        if self._stop_requested:
            self._finish()
            return
        self._request_acquire.emit()

    @QtCore.Slot()
    def _request_coordinated_acquire(self) -> None:
        if self._coordinator is None:
            return
        if self._current_request is None:
            self.error.emit("Coordinated real acquisition requires an armed acquisition plan")
            self._finish()
            return
        self._coordinator.start_batch(required_bme_triggers_for_request(self._current_request))

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if self._coordinator is None:
                self._service.result_ready.disconnect(self._on_result)
            else:
                self._coordinator.result_ready.disconnect(self._on_result)
        except (RuntimeError, TypeError):
            pass
        try:
            if self._coordinator is None:
                self._service.error_occurred.disconnect(self._on_service_error)
            else:
                self._coordinator.error_occurred.disconnect(self._on_service_error)
        except (RuntimeError, TypeError):
            pass
        if self._coordinator is not None:
            try:
                self._coordinator.log_message.disconnect(self.log_message)
            except (RuntimeError, TypeError):
                pass
        self.finished.emit()


class RealShotAnalysisWorker(QtCore.QObject):
    """Acquire N finite records via the Spectrum service and estimate jitter."""

    progress = QtCore.Signal(int, int)
    result_ready = QtCore.Signal(object)
    display_ready = QtCore.Signal(object, object, object, int)
    warning = QtCore.Signal(str)
    error = QtCore.Signal(str)
    log_message = QtCore.Signal(str)
    finished = QtCore.Signal()
    _request_acquire = QtCore.Signal()

    def __init__(
        self,
        service: SpectrumAcquisitionService,
        config: RunConfig,
        record_count: int,
        delay_s: float,
        max_lag_s: float,
        min_mz: float,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._config = config
        self._record_count = record_count
        self._delay_s = delay_s
        self._max_lag_s = max_lag_s
        self._min_mz = min_mz
        self._stop_requested = False
        self._finished = False

        self._traces: list[np.ndarray] = []
        self._latest_processed: Any = None
        self._current_index = 0

        self._request_acquire.connect(self._service.acquire)
        self._service.result_ready.connect(self._on_result)
        self._service.error_occurred.connect(self._on_service_error)

    @QtCore.Slot()
    def run(self) -> None:
        """Start collecting records."""
        self._traces.clear()
        self._latest_processed = None
        self._current_index = 0
        self._request_next()

    @QtCore.Slot()
    def request_stop(self) -> None:
        """Stop after the current record."""
        self._stop_requested = True
        self._service.abort()
        if self._service.state != ServiceState.ACQUIRING:
            self._finish()

    @QtCore.Slot(object)
    def _on_result(self, result: SpectrumAcquisitionResult) -> None:
        if self._stop_requested:
            self._finish()
            return
        try:
            plan = result.plan
            request = plan.request
            batch = spectrum_result_to_batch(
                result,
                sample_rate_hz=request.sample_rate_hz,
                pretrigger_samples=request.pretrigger_samples,
                first_trigger_index=0,
            )
            processed = process_batch(batch, self._config.digitizer, self._config.processing)
            self._latest_processed = processed
            self._traces.append(processed.unfiltered_average_trace.astype(np.float32, copy=True))
            self._current_index += 1
            self.progress.emit(self._current_index, self._record_count)

            if self._current_index >= self._record_count:
                self._analyze_and_finish()
            elif self._delay_s > 0:
                QtCore.QTimer.singleShot(int(self._delay_s * 1000), self._request_next)
            else:
                self._request_next()
        except Exception as exc:
            LOGGER.exception("Shot analysis record failed")
            self.error.emit(f"Shot analysis record error: {exc}")
            self._finish()

    @QtCore.Slot(str)
    def _on_service_error(self, message: str) -> None:
        self._stop_requested = True
        self.error.emit(f"Spectrum service error: {message}")
        self._finish()

    def _analyze_and_finish(self) -> None:
        if len(self._traces) < 2:
            self.warning.emit("Shot analysis needs at least two acquired records")
            self._finish()
            return
        try:
            trace_array = np.vstack(self._traces)
            if self._latest_processed is None or self._latest_processed.mass_axis is None:
                raise ValueError("mass calibration is required for m/z-gated shot analysis")
            analysis_mask = self._latest_processed.mass_axis >= self._min_mz
            if np.count_nonzero(analysis_mask) < 3:
                raise ValueError(f"shot analysis m/z gate leaves too few samples at m/z >= {self._min_mz:g}")
            result = analyze_jitter(trace_array[:, analysis_mask], self._config.digitizer.sample_rate_hz, self._max_lag_s)
            aligned_average = time_aligned_average(trace_array, result.shifts_s, self._config.digitizer.sample_rate_hz)
            self.display_ready.emit(
                self._latest_processed,
                aligned_average,
                trace_array[-1].astype(np.float32, copy=True),
                len(self._traces),
            )
            self.result_ready.emit(result)
        except Exception as exc:
            LOGGER.exception("Shot analysis computation failed")
            self.error.emit(f"Shot analysis computation error: {exc}")
        self._finish()

    def _request_next(self) -> None:
        if self._stop_requested:
            self._finish()
            return
        self._request_acquire.emit()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._service.result_ready.disconnect(self._on_result)
        except (RuntimeError, TypeError):
            pass
        try:
            self._service.error_occurred.disconnect(self._on_service_error)
        except (RuntimeError, TypeError):
            pass
        self.finished.emit()


def _physical_shots_for_batch(output_records: int, averages_per_segment: int, is_fpga_sum: bool) -> int:
    if is_fpga_sum:
        return output_records * averages_per_segment
    return output_records


def _display_processed(processed: Any, pending_sum: np.ndarray, pending_count: int) -> Any:
    """Build a display result from accumulated pending data."""
    unfiltered_average = (pending_sum / pending_count).astype(np.float32)
    average = unfiltered_average
    if processed.smoothing_enabled:
        average = smooth_savgol(unfiltered_average, processed.smoothing_window)
    segments = unfiltered_average[np.newaxis, :]
    return replace(
        processed,
        accepted_baseline_corrected_segments=segments,
        average_trace=average,
        unfiltered_average_trace=unfiltered_average,
        rejection_summary=RejectionSummary(
            accepted_mask=np.ones(pending_count, dtype=bool),
            clipping_rejection_count=0,
            baseline_noise_rejection_count=0,
        ),
    )
