import numpy as np
import pytest

from pytof_new.acquisition.models import AcquisitionBatch
from pytof_new.acquisition.worker import _display_processed
from pytof_new.config.models import DigitizerConfig, ProcessingConfig, RunConfig
from pytof_new.processing.averaging import average_segments
from pytof_new.processing.baseline import baseline_values, subtract_pretrigger_baseline
from pytof_new.processing.calibration import fit_mass_calibration, mass_to_tof_ns
from pytof_new.processing.conversion import adc_counts_to_voltage, create_tof_axis
from pytof_new.processing.filtering import smooth_savgol
from pytof_new.processing.jitter import analyze_jitter, time_aligned_average
from pytof_new.processing.pipeline import process_batch
from pytof_new.processing.peaks import fit_peak_window
from pytof_new.processing.quality import calculate_shot_quality, reject_shots
from pytof_new.processing.reference import subtract_reference
from pytof_new.storage.hdf5_writer import save_reference_spectrum


def test_adc_counts_to_voltage() -> None:
    raw = np.array([-32767, 0, 32767], dtype=np.int16)
    volts = adc_counts_to_voltage(raw, input_range_v=0.5, adc_full_scale_counts=32767)
    np.testing.assert_allclose(volts, [-0.5, 0.0, 0.5], rtol=1e-6)


def test_pretrigger_baseline_subtraction() -> None:
    data = np.array([[2.0, 2.0, 4.0], [1.0, 3.0, 6.0]], dtype=np.float32)
    corrected = subtract_pretrigger_baseline(data, 0, 2)
    np.testing.assert_allclose(corrected, [[0.0, 0.0, 2.0], [-1.0, 1.0, 4.0]])


def test_pretrigger_median_baseline_subtraction() -> None:
    data = np.array([[1.0, 1.0, 100.0, 5.0], [2.0, 4.0, 6.0, 10.0]], dtype=np.float32)
    values = baseline_values(data, 0, 3, method="median")
    corrected = subtract_pretrigger_baseline(data, 0, 3, method="median")
    np.testing.assert_allclose(values, [1.0, 4.0])
    np.testing.assert_allclose(corrected, [[0.0, 0.0, 99.0, 4.0], [-2.0, 0.0, 2.0, 6.0]])


def test_fit_peak_window_recovers_gaussian_center_and_fwhm() -> None:
    axis = np.linspace(8.0, 12.0, 401, dtype=np.float32)
    center = 10.25
    sigma = 0.18
    trace = 0.03 + 1.4 * np.exp(-0.5 * ((axis - center) / sigma) ** 2)
    result = fit_peak_window(axis, trace.astype(np.float32))
    assert result.fitted
    assert result.center == pytest.approx(center, abs=1e-3)
    assert result.fwhm == pytest.approx(2.354820045 * sigma, rel=1e-2)


def test_analyze_jitter_estimates_cross_correlation_rms() -> None:
    sample_rate_hz = 1.0e9
    x = np.arange(256, dtype=np.float64)
    center = 110.0
    sigma = 8.0
    shifts_samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    traces = np.vstack([np.exp(-0.5 * ((x - center - shift) / sigma) ** 2) for shift in shifts_samples])
    result = analyze_jitter(traces, sample_rate_hz, max_lag_s=10e-9)
    assert result.record_count == 5
    assert result.rms_jitter_s * sample_rate_hz == pytest.approx(np.std(shifts_samples), rel=0.15)
    assert result.uncertainty_s > 0.0


def test_time_aligned_average_sharpens_shifted_traces() -> None:
    sample_rate_hz = 1.0e9
    x = np.arange(256, dtype=np.float64)
    center = 110.0
    sigma = 6.0
    shifts_samples = np.array([-2.0, 0.0, 2.0])
    traces = np.vstack([np.exp(-0.5 * ((x - center - shift) / sigma) ** 2) for shift in shifts_samples])
    result = analyze_jitter(traces, sample_rate_hz, max_lag_s=10e-9)
    aligned = time_aligned_average(traces, result.shifts_s, sample_rate_hz)
    unaligned = np.mean(traces, axis=0)
    assert float(np.nanmax(aligned)) > float(np.max(unaligned))


def test_time_aligned_average_suppresses_misaligned_ghost_peak() -> None:
    sample_rate_hz = 1.0e9
    x = np.arange(256, dtype=np.float64)
    center = 110.0
    sigma = 4.0
    good = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    ghost = np.exp(-0.5 * ((x - center - 20.0) / sigma) ** 2)
    traces = np.vstack([good for _ in range(19)] + [ghost])
    aligned = time_aligned_average(traces, np.zeros(traces.shape[0]) / sample_rate_hz, sample_rate_hz)
    assert aligned[110] > 0.95
    assert aligned[130] < 0.02


def test_fit_mass_calibration_recovers_quadratic_coefficients() -> None:
    tof_ns = np.array([1000.0, 2000.0, 3500.0, 5000.0])
    coefficients = (3.5e-6, 0.0129, 6.27)
    mz = coefficients[0] * tof_ns**2 + coefficients[1] * tof_ns + coefficients[2]
    fitted = fit_mass_calibration(tof_ns, mz)
    np.testing.assert_allclose(fitted, coefficients, rtol=1e-10, atol=1e-10)


def test_mass_to_tof_ns_inverts_current_quadratic() -> None:
    tof_ns = np.array([1000.0, 2000.0, 3500.0, 5000.0])
    coefficients = (3.5e-6, 0.0129, 6.27)
    mz = coefficients[0] * tof_ns**2 + coefficients[1] * tof_ns + coefficients[2]
    np.testing.assert_allclose(mass_to_tof_ns(mz, coefficients), tof_ns, rtol=1e-10, atol=1e-10)


def test_shot_quality_metrics_and_clipping() -> None:
    data = np.array([[0.0, 0.0, 0.1, -0.1], [0.0, 0.2, 0.49, -0.1]], dtype=np.float32)
    metrics = calculate_shot_quality(data, 0, 2, input_range_v=0.5, clipping_margin_fraction=0.05)
    np.testing.assert_allclose(metrics.baseline_mean, [0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(metrics.baseline_rms, [0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(metrics.peak_to_peak, [0.2, 0.59], atol=1e-6)
    assert metrics.near_clipped.tolist() == [False, True]


def test_reject_shots_by_clipping_and_baseline_rms() -> None:
    data = np.array([[0.0, 0.0, 0.1], [0.0, 0.2, 0.1], [0.0, 0.0, 0.49]], dtype=np.float32)
    metrics = calculate_shot_quality(data, 0, 2, input_range_v=0.5, clipping_margin_fraction=0.05)
    summary = reject_shots(metrics, rejection_enabled=True, reject_clipped=True, maximum_baseline_rms_v=0.05)
    assert summary.accepted_mask.tolist() == [True, False, False]
    assert summary.accepted_count == 1
    assert summary.rejected_count == 2
    assert summary.clipping_rejection_count == 1
    assert summary.baseline_noise_rejection_count == 1


def test_average_segments() -> None:
    data = np.array([[1, 3], [3, 5]], dtype=np.float32)
    np.testing.assert_allclose(average_segments(data), [2, 4])


def test_create_tof_axis() -> None:
    axis = create_tof_axis(segment_samples=4, sample_rate_hz=2.0, pretrigger_samples=1)
    np.testing.assert_allclose(axis, [-0.5, 0.0, 0.5, 1.0])


def test_process_batch_returns_expected_shapes() -> None:
    raw = np.ones((3, 64), dtype=np.int16) * 100
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=3, pretrigger_samples=2)
    processing = ProcessingConfig(baseline_start=0, baseline_stop=2)
    processed = process_batch(batch, digitizer, processing)
    assert processed.voltage_segments.shape == (3, 64)
    assert processed.average_trace.shape == (64,)
    assert processed.tof_axis.shape == (64,)
    assert processed.accepted_count == 3
    assert processed.rejected_count == 0
    assert processed.record_mode == "hardware_average"


def test_process_batch_applies_baseline_subtraction() -> None:
    raw = np.array([[1000, 1000, 1000, 2000]], dtype=np.int16)
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=1, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(baseline_start=0, baseline_stop=3, baseline_method="median")
    processed = process_batch(batch, digitizer, processing)
    expected_peak = 1000 / 32767
    np.testing.assert_allclose(processed.baseline_values, [1000 / 32767], rtol=1e-6)
    np.testing.assert_allclose(processed.average_trace, [0.0, 0.0, 0.0, expected_peak], rtol=1e-6)


def test_process_batch_absolute_signal_makes_negative_polarity_positive() -> None:
    raw = np.array([[0, 0, 0, 2000]], dtype=np.int16)
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=1, pretrigger_samples=2, input_range_v=1.0)
    negative = process_batch(
        batch,
        digitizer,
        ProcessingConfig(baseline_start=0, baseline_stop=2, detector_polarity=-1),
    )
    absolute = process_batch(
        batch,
        digitizer,
        ProcessingConfig(baseline_start=0, baseline_stop=2, detector_polarity=-1, absolute_signal_enabled=True),
    )
    assert negative.average_trace[3] < 0
    assert absolute.average_trace[3] > 0
    np.testing.assert_allclose(absolute.average_trace, np.abs(negative.average_trace), rtol=1e-6)


def test_process_batch_smooths_average_only() -> None:
    raw = np.zeros((2, 64), dtype=np.int16)
    raw[:, 30] = 3000
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=2, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(
        baseline_start=0,
        baseline_stop=2,
        smoothing_enabled=True,
        smoothing_window=11,
    )
    processed = process_batch(batch, digitizer, processing)
    np.testing.assert_allclose(processed.unfiltered_average_trace, raw[0] / 32767, rtol=1e-6)
    np.testing.assert_allclose(processed.average_trace, smooth_savgol(processed.unfiltered_average_trace, 11), rtol=1e-6)
    assert processed.average_trace[30] < processed.unfiltered_average_trace[30]


def test_display_processed_preserves_smoothing_after_aggregation() -> None:
    raw = np.zeros((2, 64), dtype=np.int16)
    raw[:, 30] = 3000
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=2, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(
        baseline_start=0,
        baseline_stop=2,
        smoothing_enabled=True,
        smoothing_window=11,
    )
    processed = process_batch(batch, digitizer, processing)
    pending_sum = processed.accepted_baseline_corrected_segments.sum(axis=0, dtype=np.float64)
    display = _display_processed(processed, pending_sum, processed.accepted_count)
    np.testing.assert_allclose(display.unfiltered_average_trace, processed.unfiltered_average_trace, rtol=1e-6)
    np.testing.assert_allclose(display.average_trace, processed.average_trace, rtol=1e-6)
    assert display.average_trace[30] < display.unfiltered_average_trace[30]


def test_process_batch_low_pass_filters_records_before_averaging() -> None:
    sample_rate_hz = 1000.0
    samples = 512
    time = np.arange(samples) / sample_rate_hz
    low_frequency = np.sin(2.0 * np.pi * 10.0 * time)
    high_frequency = 0.5 * np.sin(2.0 * np.pi * 200.0 * time)
    raw = np.rint((low_frequency + high_frequency) * 10000).astype(np.int16)[np.newaxis, :]
    batch = AcquisitionBatch(raw, None, sample_rate_hz=sample_rate_hz, pretrigger_samples=0, first_trigger_index=0)
    digitizer = DigitizerConfig(sample_rate_hz=sample_rate_hz, segment_samples=samples, number_of_segments=1, pretrigger_samples=0, input_range_v=1.0)
    unfiltered = process_batch(batch, digitizer, ProcessingConfig(baseline_start=0, baseline_stop=1, subtract_baseline=False))
    filtered = process_batch(
        batch,
        digitizer,
        ProcessingConfig(
            baseline_start=0,
            baseline_stop=1,
            subtract_baseline=False,
            low_pass_enabled=True,
            low_pass_cutoff_hz=50.0,
        ),
    )
    high_reference = high_frequency.astype(np.float32) * (10000 / 32767)
    unfiltered_error = float(np.std(unfiltered.average_trace - low_frequency * (10000 / 32767)))
    filtered_error = float(np.std(filtered.average_trace - low_frequency * (10000 / 32767)))
    assert filtered_error < unfiltered_error * 0.3
    assert filtered_error < float(np.std(high_reference)) * 0.3


def test_process_batch_high_pass_filters_records_before_averaging() -> None:
    sample_rate_hz = 1000.0
    samples = 512
    time = np.arange(samples) / sample_rate_hz
    low_frequency = 0.5 * np.sin(2.0 * np.pi * 5.0 * time)
    high_frequency = np.sin(2.0 * np.pi * 150.0 * time)
    raw = np.rint((low_frequency + high_frequency) * 10000).astype(np.int16)[np.newaxis, :]
    batch = AcquisitionBatch(raw, None, sample_rate_hz=sample_rate_hz, pretrigger_samples=0, first_trigger_index=0)
    digitizer = DigitizerConfig(sample_rate_hz=sample_rate_hz, segment_samples=samples, number_of_segments=1, pretrigger_samples=0, input_range_v=1.0)
    unfiltered = process_batch(batch, digitizer, ProcessingConfig(baseline_start=0, baseline_stop=1, subtract_baseline=False))
    filtered = process_batch(
        batch,
        digitizer,
        ProcessingConfig(
            baseline_start=0,
            baseline_stop=1,
            subtract_baseline=False,
            high_pass_enabled=True,
            high_pass_cutoff_hz=50.0,
        ),
    )
    unfiltered_error = float(np.std(unfiltered.average_trace - high_frequency * (10000 / 32767)))
    filtered_error = float(np.std(filtered.average_trace - high_frequency * (10000 / 32767)))
    assert filtered_error < unfiltered_error * 0.35


def test_process_batch_subtracts_reference_before_averaging(tmp_path) -> None:
    reference_path = tmp_path / "reference.h5"
    reference = np.linspace(0.0, 0.01, 64, dtype=np.float32)
    save_reference_spectrum(reference_path, np.arange(64, dtype=np.float32), reference, 5, RunConfig())
    signal = np.zeros(64, dtype=np.float32)
    signal[20] = 0.05
    raw = np.rint((signal + reference) * 32767).astype(np.int16)[np.newaxis, :]
    batch = AcquisitionBatch(raw, None, sample_rate_hz=1.0e9, pretrigger_samples=0, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=1, pretrigger_samples=0, input_range_v=1.0)
    processing = ProcessingConfig(
        baseline_start=0,
        baseline_stop=1,
        subtract_baseline=False,
        reference_subtraction_enabled=True,
        reference_path=reference_path,
    )
    processed = process_batch(batch, digitizer, processing)
    np.testing.assert_allclose(processed.average_trace, signal, atol=1 / 32767)


def test_reference_subtraction_rejects_shape_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError, match="reference trace length"):
        subtract_reference(np.zeros((1, 64), dtype=np.float32), np.zeros(32, dtype=np.float32))


def test_process_batch_averages_accepted_shots_only() -> None:
    raw = np.zeros((3, 64), dtype=np.int16)
    raw[0, 10] = 1000
    raw[1, 10] = 2000
    raw[2, 10] = 32767
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0, record_mode="raw_segments")
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=3, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(
        baseline_start=0,
        baseline_stop=2,
        rejection_enabled=True,
        reject_clipped=True,
        clipping_margin_fraction=0.01,
    )
    processed = process_batch(batch, digitizer, processing)
    expected = ((1000 / 32767) + (2000 / 32767)) / 2
    assert processed.accepted_count == 2
    assert processed.rejected_count == 1
    np.testing.assert_allclose(processed.average_trace[10], expected, rtol=1e-5)


def test_process_batch_raises_when_all_shots_rejected() -> None:
    raw = np.ones((2, 64), dtype=np.int16) * 32767
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0, record_mode="raw_segments")
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=2, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(
        baseline_start=0,
        baseline_stop=2,
        subtract_baseline=False,
        rejection_enabled=True,
        reject_clipped=True,
        clipping_margin_fraction=0.01,
    )
    import pytest

    with pytest.raises(ValueError, match="all shots rejected"):
        process_batch(batch, digitizer, processing)


def test_rejection_is_invalid_for_hardware_averaged_records() -> None:
    raw = np.zeros((2, 64), dtype=np.int16)
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=2, pretrigger_samples=2)
    processing = ProcessingConfig(baseline_start=0, baseline_stop=2, rejection_enabled=True)
    import pytest

    with pytest.raises(ValueError, match="hardware-averaged records"):
        process_batch(batch, digitizer, processing)
