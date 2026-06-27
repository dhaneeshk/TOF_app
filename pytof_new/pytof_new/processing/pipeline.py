"""Composable processing pipeline for acquired batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytof_new.acquisition.models import AcquisitionBatch
from pytof_new.config.models import DigitizerConfig, ProcessingConfig
from pytof_new.processing.averaging import average_segments
from pytof_new.processing.baseline import baseline_values, subtract_pretrigger_baseline
from pytof_new.processing.conversion import adc_counts_to_voltage, create_tof_axis, tof_to_mass
from pytof_new.processing.filtering import high_pass_filter, low_pass_filter, smooth_savgol
from pytof_new.processing.peaks import PeakResults, find_trace_peaks
from pytof_new.processing.quality import RejectionSummary, ShotQualityMetrics, calculate_shot_quality, reject_shots
from pytof_new.processing.reference import load_reference_trace, subtract_reference


@dataclass(frozen=True)
class ProcessedBatch:
    """Processed representation of one acquired batch."""

    voltage_segments: np.ndarray
    baseline_corrected_segments: np.ndarray
    accepted_baseline_corrected_segments: np.ndarray
    baseline_values: np.ndarray
    quality_metrics: ShotQualityMetrics
    rejection_summary: RejectionSummary
    record_mode: str
    hardware_averages_per_record: int
    smoothing_enabled: bool
    smoothing_window: int
    average_trace: np.ndarray
    unfiltered_average_trace: np.ndarray
    tof_axis: np.ndarray
    mass_axis: np.ndarray | None
    peaks: PeakResults | None

    @property
    def accepted_count(self) -> int:
        """Number of shots accepted into the average."""
        return self.rejection_summary.accepted_count

    @property
    def rejected_count(self) -> int:
        """Number of shots rejected from the average."""
        return self.rejection_summary.rejected_count


def process_batch(batch: AcquisitionBatch, digitizer: DigitizerConfig, processing: ProcessingConfig) -> ProcessedBatch:
    """Process a raw acquisition batch into display/storage products."""
    voltage = adc_counts_to_voltage(batch.raw_adc, digitizer.input_range_v, processing.adc_full_scale_counts)
    voltage = voltage * np.float32(processing.detector_polarity)
    corrected = voltage
    values = np.zeros(voltage.shape[0], dtype=np.float32)
    if processing.subtract_baseline:
        values = baseline_values(voltage, processing.baseline_start, processing.baseline_stop, method=processing.baseline_method).astype(np.float32)
        corrected = subtract_pretrigger_baseline(
            voltage,
            processing.baseline_start,
            processing.baseline_stop,
            method=processing.baseline_method,
        )
    if processing.high_pass_enabled:
        corrected = high_pass_filter(
            corrected,
            digitizer.sample_rate_hz,
            processing.high_pass_cutoff_hz,
            order=processing.filter_order,
        )
    if processing.low_pass_enabled:
        corrected = low_pass_filter(
            corrected,
            digitizer.sample_rate_hz,
            processing.low_pass_cutoff_hz,
            order=processing.filter_order,
        )
    if processing.reference_subtraction_enabled:
        if processing.reference_path is None:
            raise ValueError("reference_path is required when reference subtraction is enabled")
        corrected = subtract_reference(corrected, load_reference_trace(processing.reference_path))
    if processing.absolute_signal_enabled:
        corrected = np.abs(corrected).astype(np.float32, copy=False)
    quality = calculate_shot_quality(
        corrected,
        processing.baseline_start,
        processing.baseline_stop,
        digitizer.input_range_v,
        processing.clipping_margin_fraction,
    )
    if batch.record_mode == "hardware_average" and processing.rejection_enabled:
        raise ValueError("shot rejection requires raw_segments mode; current data are hardware-averaged records")
    rejection_enabled = processing.rejection_enabled and batch.record_mode == "raw_segments"
    rejection = reject_shots(
        quality,
        rejection_enabled=rejection_enabled,
        reject_clipped=processing.reject_clipped,
        maximum_baseline_rms_v=processing.maximum_baseline_rms_v,
    )
    if rejection.accepted_count == 0:
        raise ValueError("all shots rejected")
    accepted_corrected = corrected[rejection.accepted_mask]
    average = average_segments(accepted_corrected)
    unfiltered_average = average.copy()
    if processing.smoothing_enabled:
        average = smooth_savgol(average, processing.smoothing_window)
    tof_axis = create_tof_axis(
        batch.raw_adc.shape[1],
        batch.sample_rate_hz,
        batch.pretrigger_samples,
        processing.time_zero_offset_s,
    )
    mass_axis = None
    if processing.mass_calibration_enabled and processing.mass_calibration is not None:
        mass_axis = tof_to_mass(tof_axis, processing.mass_calibration)
    peaks = None
    if processing.peak_finding_enabled:
        peaks = find_trace_peaks(mass_axis if mass_axis is not None else tof_axis, average)
    return ProcessedBatch(
        voltage_segments=voltage,
        baseline_corrected_segments=corrected,
        accepted_baseline_corrected_segments=accepted_corrected,
        baseline_values=values,
        quality_metrics=quality,
        rejection_summary=rejection,
        record_mode=batch.record_mode,
        hardware_averages_per_record=batch.hardware_averages_per_record,
        smoothing_enabled=processing.smoothing_enabled,
        smoothing_window=processing.smoothing_window,
        average_trace=average,
        unfiltered_average_trace=unfiltered_average,
        tof_axis=tof_axis,
        mass_axis=mass_axis,
        peaks=peaks,
    )
