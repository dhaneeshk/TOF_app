"""Tests for the automatic acquisition run planner."""

import pytest

from pytof_new.config.models import AcquisitionPriority, AcquisitionWorkflow
from pytof_new.hardware.acquisition_planner import (
    MAX_MEMORY_FRACTION,
    actual_posttrigger_window_s,
    compute_segment_samples,
    mz_at_window_end,
    plan_acquisition,
)
from pytof_new.hardware.spectrum_models import SpectrumAcquisitionMode, SpectrumHardwareInfo, SpectrumTriggerSource


_CAL = (3.5e-6, 0.0129, 6.27)
_HW = SpectrumHardwareInfo(serial_number=12345, max_adc_value=127, average_16bit_supported=True)


def test_mz_at_window_end_with_calibration() -> None:
    mz = mz_at_window_end(50.0, 32, 1.25e9, _CAL)
    assert mz is not None
    assert mz == pytest.approx(3.5e-6 * 50000**2 + 0.0129 * 50000 + 6.27, rel=1e-4)


def test_compute_segment_samples_keeps_window_or_longer() -> None:
    segment = compute_segment_samples(50.0, 1.25e9, 32, SpectrumAcquisitionMode.RAW_MULTI)
    assert segment == 62560
    assert actual_posttrigger_window_s(segment, 32, 1.25e9) >= 50e-6


def test_live_averaged_balanced_uses_32bit_block_average() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        acquisition_priority=AcquisitionPriority.BALANCED,
        tof_window_us=10.0,
        target_display_interval_s=0.5,
        averages_per_segment=100,
        hardware_info=_HW,
        repetition_period_s=111e-6,
        calibration=_CAL,
    )
    assert plan.workflow == AcquisitionWorkflow.LIVE_AVERAGED
    assert plan.continuous is True
    assert plan.primary_request.mode == SpectrumAcquisitionMode.AVERAGE_32BIT
    assert plan.primary_request.sample_rate_hz == 625e6
    assert plan.physical_shots_per_batch == plan.primary_request.averages_per_segment * plan.primary_request.number_of_segments
    assert plan.onboard_memory_fraction <= MAX_MEMORY_FRACTION
    assert any("Approximate maximum m/z" in line for line in plan.summary_lines)
    assert any("Time resolution: 1.600 ns/sample" in line for line in plan.summary_lines)
    assert any("including 32 pretrigger" in line for line in plan.summary_lines)
    assert any("Hardware batch interval" in line for line in plan.summary_lines)
    assert any("Target display interval" in line for line in plan.summary_lines)
    assert not any("Final partial batch" in line for line in plan.summary_lines)


def test_live_raw_keeps_hardware_batch_separate_from_display_interval() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_RAW,
        acquisition_priority=AcquisitionPriority.FAST_UPDATES,
        tof_window_us=50.0,
        target_display_interval_s=0.5,
        hardware_info=_HW,
        repetition_period_s=111e-6,
    )
    assert plan.primary_request.mode == SpectrumAcquisitionMode.RAW_MULTI
    assert plan.primary_request.number_of_segments <= 500
    assert plan.estimated_display_interval_s >= plan.estimated_batch_acquisition_s
    assert plan.total_requested_shots is None


def test_finite_shot_analysis_splits_total_shots() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.FINITE_SHOT_ANALYSIS,
        acquisition_priority=AcquisitionPriority.BALANCED,
        tof_window_us=20.0,
        total_shots=1200,
        manual_raw_shots_per_batch=500,
        advanced_mode=True,
        hardware_info=_HW,
        repetition_period_s=111e-6,
        timeout_s=2.0,
    )
    assert plan.primary_request.mode == SpectrumAcquisitionMode.RAW_MULTI
    assert plan.primary_request.number_of_segments == 500
    assert plan.full_batch_count == 2
    assert plan.final_batch_shots == 200
    assert plan.final_batch_request is not None
    assert plan.final_batch_request.number_of_segments == 200


def test_final_partial_none_when_total_exact() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.FINITE_SHOT_ANALYSIS,
        total_shots=1000,
        manual_raw_shots_per_batch=500,
        advanced_mode=True,
        hardware_info=_HW,
        repetition_period_s=111e-6,
        timeout_s=2.0,
    )
    assert plan.full_batch_count == 2
    assert plan.final_batch_shots == 0
    assert plan.final_batch_request is None


def test_basic_finite_shot_analysis_uses_largest_safe_raw_batch() -> None:
    hw = SpectrumHardwareInfo(onboard_memory_bytes=1024 * 1024)
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.FINITE_SHOT_ANALYSIS,
        total_shots=1200,
        advanced_mode=False,
        manual_sample_rate_hz=156.25e6,
        manual_segment_samples=1024,
        hardware_info=hw,
        repetition_period_s=100e-6,
    )
    assert plan.primary_request.number_of_segments == 512
    assert plan.full_batch_count == 2
    assert plan.final_batch_shots == 176
    assert plan.onboard_memory_fraction == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("priority", "expected_rate"),
    [
        (AcquisitionPriority.FAST_UPDATES, 312.5e6),
        (AcquisitionPriority.BALANCED, 625e6),
        (AcquisitionPriority.HIGHEST_TIME_RESOLUTION, 1.25e9),
    ],
)
def test_priority_selects_expected_live_raw_rate(priority, expected_rate) -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_RAW,
        acquisition_priority=priority,
        tof_window_us=10.0,
        target_display_interval_s=0.1,
        hardware_info=_HW,
        repetition_period_s=111e-6,
    )
    assert plan.primary_request.sample_rate_hz == expected_rate


def test_32bit_to_16bit_fallback_when_window_exceeds_32bit_limit() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        acquisition_priority=AcquisitionPriority.HIGHEST_TIME_RESOLUTION,
        tof_window_us=56.0,
        target_display_interval_s=0.5,
        averages_per_segment=100,
        hardware_info=_HW,
        repetition_period_s=120e-6,
    )
    assert plan.primary_request.mode == SpectrumAcquisitionMode.AVERAGE_16BIT


def test_manual_16bit_accumulator_is_available_for_default_target_card() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        acquisition_priority=AcquisitionPriority.BALANCED,
        tof_window_us=10.0,
        target_display_interval_s=0.5,
        averages_per_segment=100,
        advanced_mode=True,
        manual_accumulator_mode="16bit",
        repetition_period_s=111e-6,
        timeout_s=10.0,
    )
    assert plan.primary_request.mode == SpectrumAcquisitionMode.AVERAGE_16BIT


def test_best_signal_to_noise_prefers_lower_rate_32bit_over_high_rate_16bit() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        acquisition_priority=AcquisitionPriority.BEST_SIGNAL_TO_NOISE,
        tof_window_us=56.0,
        target_display_interval_s=0.5,
        averages_per_segment=100,
        hardware_info=_HW,
        repetition_period_s=120e-6,
    )
    assert plan.primary_request.mode == SpectrumAcquisitionMode.AVERAGE_32BIT
    assert plan.primary_request.sample_rate_hz < 1.25e9


def test_basic_live_averaged_priority_controls_shots_per_fpga_sum() -> None:
    fast = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        acquisition_priority=AcquisitionPriority.FAST_UPDATES,
        tof_window_us=10.0,
        averages_per_segment=100,
        hardware_info=_HW,
        repetition_period_s=111e-6,
    )
    balanced = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        acquisition_priority=AcquisitionPriority.BALANCED,
        tof_window_us=10.0,
        averages_per_segment=100,
        hardware_info=_HW,
        repetition_period_s=111e-6,
    )
    best_snr = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        acquisition_priority=AcquisitionPriority.BEST_SIGNAL_TO_NOISE,
        tof_window_us=10.0,
        averages_per_segment=100,
        hardware_info=_HW,
        repetition_period_s=111e-6,
    )
    assert fast.primary_request.averages_per_segment == 25
    assert balanced.primary_request.averages_per_segment == 100
    assert best_snr.primary_request.averages_per_segment > balanced.primary_request.averages_per_segment


def test_memory_boundary_at_50_percent_is_accepted() -> None:
    segment = 1024
    records = (1024 * 1024) // segment
    hw = SpectrumHardwareInfo(onboard_memory_bytes=2 * 1024 * 1024)
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_RAW,
        tof_window_us=1.0,
        advanced_mode=True,
        manual_sample_rate_hz=156.25e6,
        manual_segment_samples=segment,
        manual_raw_shots_per_batch=records,
        hardware_info=hw,
        repetition_period_s=100e-6,
        timeout_s=200.0,
    )
    assert plan.onboard_memory_fraction == pytest.approx(0.5)


def test_advanced_memory_above_50_percent_rejected() -> None:
    hw = SpectrumHardwareInfo(onboard_memory_bytes=2 * 1024 * 1024)
    with pytest.raises(ValueError, match="Maximum raw shots"):
        plan_acquisition(
            acquisition_workflow=AcquisitionWorkflow.LIVE_RAW,
            advanced_mode=True,
            manual_sample_rate_hz=156.25e6,
            manual_segment_samples=1024,
            manual_raw_shots_per_batch=2000,
            hardware_info=hw,
            repetition_period_s=100e-6,
            timeout_s=200.0,
        )


def test_block_average_software_trigger_rejected() -> None:
    with pytest.raises(ValueError, match="software trigger"):
        plan_acquisition(
            acquisition_workflow=AcquisitionWorkflow.LIVE_AVERAGED,
            trigger_source=SpectrumTriggerSource.SOFTWARE,
            hardware_info=_HW,
            repetition_period_s=111e-6,
        )


def test_raw_software_trigger_allowed_for_advanced_testing() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_RAW,
        advanced_mode=True,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
        hardware_info=_HW,
        repetition_period_s=111e-6,
    )
    assert plan.primary_request.trigger_source == SpectrumTriggerSource.SOFTWARE


def test_timeout_is_derived_from_expected_trigger_duration() -> None:
    plan = plan_acquisition(
        acquisition_workflow=AcquisitionWorkflow.LIVE_RAW,
        manual_raw_shots_per_batch=1000,
        advanced_mode=False,
        timeout_s=2.0,
        hardware_info=_HW,
        repetition_period_s=0.01,
    )
    assert plan.primary_request.timeout_s == pytest.approx(13.5)


def test_advanced_timeout_too_short_rejected() -> None:
    with pytest.raises(ValueError, match="minimum required timeout"):
        plan_acquisition(
            acquisition_workflow=AcquisitionWorkflow.LIVE_RAW,
            manual_raw_shots_per_batch=1000,
            advanced_mode=True,
            timeout_s=1.0,
            hardware_info=_HW,
            repetition_period_s=0.01,
        )
