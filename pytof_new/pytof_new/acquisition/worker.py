"""Qt acquisition worker for GUI-driven mock acquisition."""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from typing import Any

import numpy as np
from PySide6 import QtCore


from pytof_new.acquisition.controller import AcquisitionController
from pytof_new.config.models import RunConfig
from pytof_new.hardware.mock_delay_generator import MockDelayGenerator
from pytof_new.hardware.mock_digitizer import MockDigitizer, MockDigitizerProfile
from pytof_new.processing.filtering import smooth_savgol
from pytof_new.processing.jitter import analyze_jitter, time_aligned_average
from pytof_new.processing.pipeline import process_batch
from pytof_new.processing.quality import RejectionSummary

LOGGER = logging.getLogger(__name__)


class AcquisitionWorker(QtCore.QObject):
    """Run mock acquisition in a dedicated Qt thread."""

    state_changed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    batch_ready = QtCore.Signal(object, object)
    warning = QtCore.Signal(str)
    error = QtCore.Signal(str)
    log_message = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, config: RunConfig, continuous: bool = False, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.continuous = continuous
        self._stop_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        """Acquire one or more mock batches and emit processed results."""
        profile = MockDigitizerProfile(
            peaks=MockDigitizerProfile().peaks if self.config.mock_spectra.ion_peaks_enabled else (),
            timing_jitter_s=self.config.mock_spectra.timing_jitter_s,
            resolving_power=self.config.mock_spectra.resolving_power,
            noise_rms_v=self.config.mock_spectra.noise_rms_v,
            ringing_amplitude_v=(
                self.config.mock_spectra.ringing_amplitude_v if self.config.mock_spectra.ringing_enabled else 0.0
            ),
            ringing_frequency_hz=self.config.mock_spectra.ringing_frequency_hz,
            ringing_decay_s=self.config.mock_spectra.ringing_decay_s,
            ringing_phase_rad=self.config.mock_spectra.ringing_phase_rad,
            ringing_follows_timing_jitter=self.config.mock_spectra.ringing_follows_timing_jitter,
            random_seed=None,
        )
        controller = AcquisitionController(MockDigitizer(profile), MockDelayGenerator())
        try:
            self.config.validate()
            controller.begin_acquisition(self.config)
            self.state_changed.emit(controller.state.name)
            self.log_message.emit("Acquisition started")
            last_display_emit = 0.0
            display_interval_s = 0.1
            pending_sum: np.ndarray | None = None
            pending_count = 0
            latest_processed = None

            while not self._stop_requested:
                batch = controller.read_batch()
                self.state_changed.emit(controller.state.name)
                processed = process_batch(batch, self.config.digitizer, self.config.processing)
                latest_processed = processed
                batch_sum = processed.accepted_baseline_corrected_segments.sum(axis=0, dtype=np.float64)
                if pending_sum is None:
                    pending_sum = np.zeros_like(batch_sum, dtype=np.float64)
                pending_sum += batch_sum
                pending_count += processed.accepted_count
                dropped = int(batch.metadata.get("dropped_triggers", 0))
                self.progress.emit(controller.total_segments, dropped)
                now = time.monotonic()
                if now - last_display_emit >= display_interval_s or not self.continuous:
                    self.batch_ready.emit(batch, _display_processed(latest_processed, pending_sum, pending_count))
                    pending_sum = None
                    pending_count = 0
                    last_display_emit = now
                if not self.continuous:
                    break
                time.sleep(self.config.bme.repetition_period_s * self.config.digitizer.hardware_averages_per_record)
            if pending_sum is not None and latest_processed is not None and pending_count > 0:
                self.batch_ready.emit(batch, _display_processed(latest_processed, pending_sum, pending_count))
        except Exception as exc:
            LOGGER.exception("GUI acquisition failed")
            self.error.emit(f"Acquisition stopped due to error: {exc}")
        finally:
            try:
                controller.end_acquisition()
                controller.disconnect_hardware()
                self.state_changed.emit(controller.state.name)
            except Exception as exc:
                self.warning.emit(f"safe shutdown warning: {exc}")
            self.log_message.emit("Acquisition stopped safely")
            self.finished.emit()

    @QtCore.Slot()
    def request_stop(self) -> None:
        """Ask the worker to stop after the current blocking batch completes."""
        self._stop_requested = True


class ShotAnalysisWorker(QtCore.QObject):
    """Acquire delayed single-record spectra and estimate timing jitter."""

    progress = QtCore.Signal(int, int)
    result_ready = QtCore.Signal(object)
    display_ready = QtCore.Signal(object, object, object, int)
    warning = QtCore.Signal(str)
    error = QtCore.Signal(str)
    log_message = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        config: RunConfig,
        record_count: int,
        delay_s: float,
        max_lag_s: float,
        min_mz: float,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.record_count = record_count
        self.delay_s = delay_s
        self.max_lag_s = max_lag_s
        self.min_mz = min_mz
        self._stop_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        """Acquire records with a delay and emit a cross-correlation jitter summary."""
        profile = MockDigitizerProfile(
            peaks=MockDigitizerProfile().peaks if self.config.mock_spectra.ion_peaks_enabled else (),
            timing_jitter_s=self.config.mock_spectra.timing_jitter_s,
            resolving_power=self.config.mock_spectra.resolving_power,
            noise_rms_v=self.config.mock_spectra.noise_rms_v,
            ringing_amplitude_v=(
                self.config.mock_spectra.ringing_amplitude_v if self.config.mock_spectra.ringing_enabled else 0.0
            ),
            ringing_frequency_hz=self.config.mock_spectra.ringing_frequency_hz,
            ringing_decay_s=self.config.mock_spectra.ringing_decay_s,
            ringing_phase_rad=self.config.mock_spectra.ringing_phase_rad,
            ringing_follows_timing_jitter=self.config.mock_spectra.ringing_follows_timing_jitter,
            random_seed=None,
        )
        controller = AcquisitionController(MockDigitizer(profile), MockDelayGenerator())
        traces: list[np.ndarray] = []
        latest_processed = None
        try:
            self.config.validate()
            controller.begin_acquisition(self.config)
            self.log_message.emit("Shot analysis started")
            for index in range(self.record_count):
                if self._stop_requested:
                    break
                batch = controller.read_batch()
                processed = process_batch(batch, self.config.digitizer, self.config.processing)
                latest_processed = processed
                traces.append(processed.unfiltered_average_trace.astype(np.float32, copy=True))
                self.progress.emit(index + 1, self.record_count)
                if index < self.record_count - 1 and not self._stop_requested:
                    time.sleep(self.delay_s)
            if len(traces) >= 2:
                trace_array = np.vstack(traces)
                if latest_processed is None or latest_processed.mass_axis is None:
                    raise ValueError("mass calibration is required for m/z-gated shot analysis")
                analysis_mask = latest_processed.mass_axis >= self.min_mz
                if np.count_nonzero(analysis_mask) < 3:
                    raise ValueError(f"shot analysis m/z gate leaves too few samples at m/z >= {self.min_mz:g}")
                result = analyze_jitter(trace_array[:, analysis_mask], self.config.digitizer.sample_rate_hz, self.max_lag_s)
                aligned_average = time_aligned_average(trace_array, result.shifts_s, self.config.digitizer.sample_rate_hz)
                self.display_ready.emit(latest_processed, aligned_average, trace_array[-1].astype(np.float32, copy=True), len(traces))
                self.result_ready.emit(result)
            else:
                self.warning.emit("Shot analysis needs at least two acquired records")
        except Exception as exc:
            LOGGER.exception("shot analysis failed")
            self.error.emit(f"Shot analysis failed: {exc}")
        finally:
            try:
                controller.end_acquisition()
                controller.disconnect_hardware()
            except Exception as exc:
                self.warning.emit(f"safe shutdown warning: {exc}")
            self.log_message.emit("Shot analysis stopped safely")
            self.finished.emit()

    @QtCore.Slot()
    def request_stop(self) -> None:
        """Ask shot analysis to stop after the current record."""
        self._stop_requested = True


class QtLogHandler(logging.Handler):
    """Forward Python logging records into a Qt signal or callable."""

    def __init__(self, sink: Any) -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a formatted log record."""
        message = self.format(record)
        try:
            if hasattr(self.sink, "emit"):
                self.sink.emit(message)
            else:
                self.sink(message)
        except RuntimeError:
            # The GUI signal source may already be destroyed during app shutdown.
            pass


def _display_processed(processed: object, pending_sum: np.ndarray, pending_count: int) -> object:
    """Build a display result that represents all records since the last plot update."""
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
