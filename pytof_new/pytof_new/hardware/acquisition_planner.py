"""Automatic finite-batch acquisition planner for the Spectrum digitizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pytof_new.config.models import AcquisitionPriority, AcquisitionWorkflow
from pytof_new.hardware.spectrum_limits import (
    AVG16_MAX_AVERAGES,
    AVG16_MAX_SEGMENT_SAMPLES,
    AVG16_MIN_AVERAGES,
    AVG16_MIN_SEGMENT_SAMPLES,
    AVG16_SEGMENT_STEP,
    AVG32_MAX_AVERAGES,
    AVG32_MAX_SEGMENT_SAMPLES,
    AVG32_MIN_AVERAGES,
    AVG32_MIN_SEGMENT_SAMPLES,
    AVG32_SEGMENT_STEP,
    RAW_MIN_SEGMENT_SAMPLES,
    RAW_SEGMENT_STEP,
    default_m4i2210_info,
    plan_spectrum_acquisition,
)
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionRequest,
    SpectrumHardwareInfo,
    SpectrumTriggerSource,
)
from pytof_new.processing.conversion import tof_to_mass


MAX_MEMORY_FRACTION = 0.50
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_PRETRIGGER_SAMPLES = 32
DEFAULT_TOF_WINDOW_US = 50.0
DEFAULT_TARGET_DISPLAY_INTERVAL_S = 0.5
DEFAULT_TOTAL_SHOTS = 1000
DEFAULT_AVERAGES_PER_SUM = 100
DEFAULT_RAW_SHOTS_PER_BATCH = 500
SAMPLE_RATES_HZ = (1.25e9, 625e6, 312.5e6, 156.25e6)


@dataclass(frozen=True)
class AcquisitionRunPlan:
    """Complete user-requested acquisition run plan.

    ``primary_request`` is one finite Spectrum acquisition. Continuous runs
    repeat it until stopped. Finite runs repeat it for ``full_batch_count`` and
    optionally acquire ``final_batch_request`` once.
    """

    workflow: AcquisitionWorkflow
    priority: AcquisitionPriority
    primary_request: SpectrumAcquisitionRequest
    final_batch_request: SpectrumAcquisitionRequest | None
    continuous: bool
    total_requested_shots: int | None
    physical_shots_per_batch: int
    output_records_per_batch: int
    full_batch_count: int | None
    final_batch_shots: int | None
    requested_display_interval_s: float | None
    estimated_batch_acquisition_s: float
    estimated_batch_cycle_s: float
    estimated_display_interval_s: float | None
    transfer_bytes_per_batch: int
    onboard_memory_bytes: int
    onboard_memory_fraction: float
    actual_posttrigger_window_s: float
    warnings: tuple[str, ...]
    summary_lines: tuple[str, ...]

    @property
    def request(self) -> SpectrumAcquisitionRequest:
        """Compatibility alias for older code that expected a single request."""
        return self.primary_request

    @property
    def plan_lines(self) -> tuple[str, ...]:
        """Compatibility alias for older GUI code."""
        return self.summary_lines


AcquisitionPlanResult = AcquisitionRunPlan


def mz_at_window_end(
    tof_window_us: float,
    pretrigger_samples: int,
    sample_rate_hz: float,
    calibration: tuple[float, float, float] | None,
) -> float | None:
    """Return approximate m/z at the requested post-trigger TOF window end."""
    if calibration is None:
        return None
    post_samples = max(1, int(math.ceil(tof_window_us * 1e-6 * sample_rate_hz)))
    tof_end_s = post_samples / sample_rate_hz
    return float(tof_to_mass(np.array([tof_end_s]), calibration)[0])


def compute_segment_samples(
    tof_window_us: float,
    sample_rate_hz: float,
    pretrigger_samples: int,
    mode: SpectrumAcquisitionMode,
) -> int:
    """Round requested TOF window + pretrigger up to a valid segment size."""
    posttrigger_samples = int(math.ceil(tof_window_us * 1e-6 * sample_rate_hz))
    return compute_segment_samples_from_posttrigger(posttrigger_samples, pretrigger_samples, mode)


def compute_segment_samples_from_posttrigger(
    posttrigger_samples: int,
    pretrigger_samples: int,
    mode: SpectrumAcquisitionMode,
) -> int:
    """Round explicit posttrigger samples + pretrigger to a valid segment size."""
    minimum, step, _maximum = _segment_limits(mode)
    return max(minimum, _round_up(pretrigger_samples + posttrigger_samples, step))


def actual_posttrigger_window_s(segment_samples: int, pretrigger_samples: int, sample_rate_hz: float) -> float:
    """Return the actual recorded post-trigger duration."""
    return (segment_samples - pretrigger_samples) / sample_rate_hz


def plan_acquisition(
    acquisition_workflow: AcquisitionWorkflow | str = AcquisitionWorkflow.LIVE_AVERAGED,
    acquisition_priority: AcquisitionPriority | str = AcquisitionPriority.BALANCED,
    tof_window_us: float = DEFAULT_TOF_WINDOW_US,
    target_display_interval_s: float = DEFAULT_TARGET_DISPLAY_INTERVAL_S,
    total_shots: int = DEFAULT_TOTAL_SHOTS,
    pretrigger_samples: int = DEFAULT_PRETRIGGER_SAMPLES,
    averages_per_segment: int = DEFAULT_AVERAGES_PER_SUM,
    hardware_info: SpectrumHardwareInfo | None = None,
    repetition_period_s: float = 111e-6,
    calibration: tuple[float, float, float] | None = None,
    advanced_mode: bool = False,
    manual_sample_rate_hz: float | None = None,
    manual_accumulator_mode: str = "automatic",
    manual_fpga_sums_per_batch: int | None = None,
    manual_raw_shots_per_batch: int | None = None,
    manual_segment_samples: int | None = None,
    input_range_v: float = 0.5,
    coupling: str = "dc",
    bandwidth_limit_enabled: bool = False,
    trigger_source: str | SpectrumTriggerSource = SpectrumTriggerSource.EXTERNAL0,
    trigger_edge: str = "rising",
    trigger_level_v: float = 1.5,
    trigger_termination_ohm: int = 50,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    config_mode: str | None = None,
) -> AcquisitionRunPlan:
    """Plan a complete run made of one or more finite hardware batches."""
    info = hardware_info or default_m4i2210_info()
    workflow = _workflow(acquisition_workflow, config_mode)
    priority = _priority(acquisition_priority)
    if repetition_period_s <= 0:
        raise ValueError("repetition_period_s must be positive")
    if tof_window_us <= 0:
        raise ValueError("tof_window_us must be positive")

    trigger = _trigger(trigger_source)
    if workflow == AcquisitionWorkflow.LIVE_AVERAGED:
        return _plan_live_averaged(
            priority=priority,
            tof_window_us=tof_window_us,
            target_display_interval_s=target_display_interval_s,
            pretrigger_samples=pretrigger_samples,
            averages_per_segment=averages_per_segment,
            hardware_info=info,
            repetition_period_s=repetition_period_s,
            calibration=calibration,
            advanced_mode=advanced_mode,
            manual_sample_rate_hz=manual_sample_rate_hz,
            manual_accumulator_mode=manual_accumulator_mode,
            manual_fpga_sums_per_batch=manual_fpga_sums_per_batch,
            manual_segment_samples=manual_segment_samples,
            input_range_v=input_range_v,
            coupling=coupling,
            bandwidth_limit_enabled=bandwidth_limit_enabled,
            trigger_source=trigger,
            trigger_edge=trigger_edge,
            trigger_level_v=trigger_level_v,
            trigger_termination_ohm=trigger_termination_ohm,
            timeout_s=timeout_s,
        )
    if workflow == AcquisitionWorkflow.LIVE_RAW:
        return _plan_raw(
            workflow=workflow,
            priority=priority,
            tof_window_us=tof_window_us,
            target_display_interval_s=target_display_interval_s,
            total_shots=None,
            pretrigger_samples=pretrigger_samples,
            hardware_info=info,
            repetition_period_s=repetition_period_s,
            calibration=calibration,
            advanced_mode=advanced_mode,
            manual_sample_rate_hz=manual_sample_rate_hz,
            manual_raw_shots_per_batch=manual_raw_shots_per_batch,
            manual_segment_samples=manual_segment_samples,
            input_range_v=input_range_v,
            coupling=coupling,
            bandwidth_limit_enabled=bandwidth_limit_enabled,
            trigger_source=trigger,
            trigger_edge=trigger_edge,
            trigger_level_v=trigger_level_v,
            trigger_termination_ohm=trigger_termination_ohm,
            timeout_s=timeout_s,
        )
    if workflow == AcquisitionWorkflow.FINITE_SHOT_ANALYSIS:
        return _plan_raw(
            workflow=workflow,
            priority=priority,
            tof_window_us=tof_window_us,
            target_display_interval_s=None,
            total_shots=total_shots,
            pretrigger_samples=pretrigger_samples,
            hardware_info=info,
            repetition_period_s=repetition_period_s,
            calibration=calibration,
            advanced_mode=advanced_mode,
            manual_sample_rate_hz=manual_sample_rate_hz,
            manual_raw_shots_per_batch=manual_raw_shots_per_batch,
            manual_segment_samples=manual_segment_samples,
            input_range_v=input_range_v,
            coupling=coupling,
            bandwidth_limit_enabled=bandwidth_limit_enabled,
            trigger_source=trigger,
            trigger_edge=trigger_edge,
            trigger_level_v=trigger_level_v,
            trigger_termination_ohm=trigger_termination_ohm,
            timeout_s=timeout_s,
        )
    raise ValueError(f"unsupported acquisition workflow: {workflow}")


def plan_continuous_block_average(*args, **kwargs) -> AcquisitionRunPlan:
    """Compatibility wrapper for live averaged planning."""
    kwargs["acquisition_workflow"] = AcquisitionWorkflow.LIVE_AVERAGED
    if "target_update_interval_s" in kwargs and "target_display_interval_s" not in kwargs:
        kwargs["target_display_interval_s"] = kwargs.pop("target_update_interval_s")
    if "manual_n" in kwargs and "averages_per_segment" not in kwargs:
        kwargs["averages_per_segment"] = kwargs.pop("manual_n")
    if "manual_k" in kwargs and "manual_fpga_sums_per_batch" not in kwargs:
        kwargs["manual_fpga_sums_per_batch"] = kwargs.pop("manual_k")
    if "manual_rate_hz" in kwargs and "manual_sample_rate_hz" not in kwargs:
        kwargs["manual_sample_rate_hz"] = kwargs.pop("manual_rate_hz")
    return plan_acquisition(*args, **kwargs)


def plan_continuous_raw(*args, **kwargs) -> AcquisitionRunPlan:
    """Compatibility wrapper for live raw planning."""
    kwargs["acquisition_workflow"] = AcquisitionWorkflow.LIVE_RAW
    if "target_update_interval_s" in kwargs and "target_display_interval_s" not in kwargs:
        kwargs["target_display_interval_s"] = kwargs.pop("target_update_interval_s")
    if "manual_shots_per_batch" in kwargs and "manual_raw_shots_per_batch" not in kwargs:
        kwargs["manual_raw_shots_per_batch"] = kwargs.pop("manual_shots_per_batch")
    if "manual_rate_hz" in kwargs and "manual_sample_rate_hz" not in kwargs:
        kwargs["manual_sample_rate_hz"] = kwargs.pop("manual_rate_hz")
    return plan_acquisition(*args, **kwargs)


def plan_finite_raw_shot_analysis(*args, **kwargs) -> AcquisitionRunPlan:
    """Compatibility wrapper for finite raw shot-analysis planning."""
    kwargs["acquisition_workflow"] = AcquisitionWorkflow.FINITE_SHOT_ANALYSIS
    if "finite_shot_count" in kwargs and "total_shots" not in kwargs:
        kwargs["total_shots"] = kwargs.pop("finite_shot_count")
    if "sample_rate_hz" in kwargs and "manual_sample_rate_hz" not in kwargs:
        kwargs["manual_sample_rate_hz"] = kwargs.pop("sample_rate_hz")
    return plan_acquisition(*args, **kwargs)


def format_existing_plan(request, plan, hardware_info: SpectrumHardwareInfo) -> list[str]:
    """Format an advanced request/validated plan for GUI logging."""
    bytes_mb = plan.transfer_bytes / 1e6
    lines = [f"Spectrum {_model_display_name(hardware_info)}"]
    lines.append(f"Hardware mode: {_mode_display(request.mode)}")
    lines.append(f"Sample rate: {_rate_label(request.sample_rate_hz)}")
    lines.append(f"Segment size: {request.segment_samples:,} samples")
    lines.append(f"Output records per hardware batch: {request.number_of_segments:,}")
    if plan.is_fpga_sum:
        lines.append(f"Shots per FPGA sum: {request.averages_per_segment:,}")
    lines.append(f"Transfer per batch: {bytes_mb:.2f} MB")
    if request.trigger_source == SpectrumTriggerSource.EXTERNAL0:
        lines.append(f"Trigger: External 0, {request.trigger_edge} edge")
        lines.append("External trigger required")
    return lines


def _plan_live_averaged(**kwargs) -> AcquisitionRunPlan:
    priority: AcquisitionPriority = kwargs["priority"]
    info: SpectrumHardwareInfo = kwargs["hardware_info"]
    advanced = kwargs["advanced_mode"]
    warnings: list[str] = []
    trigger: SpectrumTriggerSource = kwargs["trigger_source"]
    if trigger == SpectrumTriggerSource.SOFTWARE:
        raise ValueError("Block Averaging requires an external or channel trigger; software trigger is not supported")

    candidate_rates = _candidate_rates(priority, kwargs["manual_sample_rate_hz"])
    candidate_rates = [rate for rate in candidate_rates if rate <= info.max_sample_rate_hz]
    if not candidate_rates:
        raise ValueError("No sample rate is compatible with the discovered hardware maximum")

    modes = _averaged_mode_order(priority, kwargs["manual_accumulator_mode"])
    errors: list[str] = []
    for mode, rate in _averaged_candidates(priority, candidate_rates, modes):
        try:
            return _try_live_averaged_candidate(mode, rate, warnings=list(warnings), **kwargs)
        except ValueError as exc:
            errors.append(str(exc))
            if advanced:
                break
    raise ValueError(errors[-1] if errors else "No valid live averaged acquisition plan found")


def _try_live_averaged_candidate(
    mode: SpectrumAcquisitionMode,
    rate_hz: float,
    warnings: list[str],
    **kwargs,
) -> AcquisitionRunPlan:
    info: SpectrumHardwareInfo = kwargs["hardware_info"]
    advanced = kwargs["advanced_mode"]
    segment_samples = _segment_samples_for_request(
        kwargs["tof_window_us"], rate_hz, kwargs["pretrigger_samples"], mode, kwargs["manual_segment_samples"]
    )
    _validate_record_timing(segment_samples, kwargs["pretrigger_samples"], rate_hz, kwargs["repetition_period_s"])
    minimum, _step, maximum = _segment_limits(mode)
    if segment_samples < minimum or (maximum is not None and segment_samples > maximum):
        raise ValueError(f"{_mode_display(mode)} segment size {segment_samples:,} is outside [{minimum:,}, {maximum:,}]")
    if mode == SpectrumAcquisitionMode.AVERAGE_16BIT and not info.average_16bit_supported:
        raise ValueError("16-bit Block Averaging is not available on this card")

    min_n, max_n = _average_limits(mode)
    n = int(kwargs["averages_per_segment"])
    if not advanced:
        planned_n = _basic_averages_per_segment(
            priority=kwargs["priority"],
            configured_n=n,
            mode=mode,
            target_display_interval_s=kwargs["target_display_interval_s"],
            repetition_period_s=kwargs["repetition_period_s"],
        )
        if planned_n != n:
            n = planned_n
        if n < min_n:
            warnings.append(f"Shots per FPGA sum increased from {n} to {min_n} for {_mode_display(mode)}.")
            n = min_n
        if n > max_n:
            warnings.append(f"Shots per FPGA sum reduced from {n:,} to {max_n:,} for {_mode_display(mode)}.")
            n = max_n
    elif n < min_n or n > max_n:
        raise ValueError(f"Shots per FPGA sum for {_mode_display(mode)} must be in [{min_n:,}, {max_n:,}]")

    bytes_per_sample = _bytes_per_sample(mode)
    max_records = _max_records_by_memory(info, segment_samples, bytes_per_sample)
    if max_records < 1:
        raise ValueError(_one_record_memory_error(info, segment_samples, bytes_per_sample))
    target_s = kwargs["target_display_interval_s"]
    derived_k = max(1, int(round(target_s / (n * kwargs["repetition_period_s"]))))
    k = int(kwargs["manual_fpga_sums_per_batch"] or derived_k)
    if k < 1:
        raise ValueError("FPGA sums per hardware batch must be positive")
    if k > max_records:
        if advanced:
            requested_bytes = k * segment_samples * bytes_per_sample
            raise ValueError(
                f"Requested batch uses {_human_bytes(requested_bytes)}, exceeding the {_human_bytes(_max_batch_bytes(info))} safety limit. "
                f"Maximum FPGA sums per batch for this configuration: {max_records:,}."
            )
        warnings.append(f"FPGA sums per hardware batch reduced from {k:,} to {max_records:,} to stay within 50% onboard memory.")
        k = max_records

    physical_shots = n * k
    acq_s = physical_shots * kwargs["repetition_period_s"]
    timeout_s = _timeout(kwargs["timeout_s"], acq_s, advanced)
    request = _request(
        mode=mode,
        rate_hz=rate_hz,
        segment_samples=segment_samples,
        number_of_segments=k,
        averages_per_segment=n,
        timeout_s=timeout_s,
        context=kwargs,
    )
    validated = plan_spectrum_acquisition(request, info)
    transfer_bytes = validated.transfer_bytes
    memory_fraction = transfer_bytes / info.onboard_memory_bytes
    _validate_memory_fraction(memory_fraction, info)
    actual_window_s = actual_posttrigger_window_s(segment_samples, kwargs["pretrigger_samples"], rate_hz)
    display_s = max(target_s, acq_s)
    lines = _summary_lines(
        workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        request=request,
        total_shots=None,
        full_batches=None,
        final_shots=None,
        physical_shots=physical_shots,
        output_records=k,
        acq_s=acq_s,
        display_s=display_s,
        transfer_bytes=transfer_bytes,
        info=info,
        actual_window_s=actual_window_s,
        requested_display_s=target_s,
        calibration=kwargs["calibration"],
        tof_window_us=kwargs["tof_window_us"],
        warnings=warnings,
    )
    return AcquisitionRunPlan(
        workflow=AcquisitionWorkflow.LIVE_AVERAGED,
        priority=kwargs["priority"],
        primary_request=request,
        final_batch_request=None,
        continuous=True,
        total_requested_shots=None,
        physical_shots_per_batch=physical_shots,
        output_records_per_batch=k,
        full_batch_count=None,
        final_batch_shots=None,
        requested_display_interval_s=target_s,
        estimated_batch_acquisition_s=acq_s,
        estimated_batch_cycle_s=acq_s,
        estimated_display_interval_s=display_s,
        transfer_bytes_per_batch=transfer_bytes,
        onboard_memory_bytes=info.onboard_memory_bytes,
        onboard_memory_fraction=memory_fraction,
        actual_posttrigger_window_s=actual_window_s,
        warnings=tuple(warnings),
        summary_lines=tuple(lines),
    )


def _plan_raw(**kwargs) -> AcquisitionRunPlan:
    workflow: AcquisitionWorkflow = kwargs["workflow"]
    priority: AcquisitionPriority = kwargs["priority"]
    info: SpectrumHardwareInfo = kwargs["hardware_info"]
    advanced = kwargs["advanced_mode"]
    warnings: list[str] = []
    candidate_rates = _candidate_rates(priority, kwargs["manual_sample_rate_hz"])
    candidate_rates = [rate for rate in candidate_rates if rate <= info.max_sample_rate_hz]
    errors: list[str] = []
    for rate_hz in candidate_rates:
        try:
            return _try_raw_candidate(rate_hz, warnings=list(warnings), **kwargs)
        except ValueError as exc:
            errors.append(str(exc))
            if advanced:
                break
    raise ValueError(errors[-1] if errors else f"No valid {workflow.value} acquisition plan found")


def _try_raw_candidate(rate_hz: float, warnings: list[str], **kwargs) -> AcquisitionRunPlan:
    workflow: AcquisitionWorkflow = kwargs["workflow"]
    info: SpectrumHardwareInfo = kwargs["hardware_info"]
    advanced = kwargs["advanced_mode"]
    segment_samples = _segment_samples_for_request(
        kwargs["tof_window_us"], rate_hz, kwargs["pretrigger_samples"], SpectrumAcquisitionMode.RAW_MULTI, kwargs["manual_segment_samples"]
    )
    _validate_record_timing(segment_samples, kwargs["pretrigger_samples"], rate_hz, kwargs["repetition_period_s"])
    max_records = _max_records_by_memory(info, segment_samples, 1)
    if max_records < 1:
        raise ValueError(_one_record_memory_error(info, segment_samples, 1))

    if workflow == AcquisitionWorkflow.LIVE_RAW:
        derived = max(1, int(round(kwargs["target_display_interval_s"] / kwargs["repetition_period_s"])))
        # Keep raw hardware batches smaller than display cadence to reduce transfer latency.
        preferred = min(derived, DEFAULT_RAW_SHOTS_PER_BATCH)
        requested_batch = int(kwargs["manual_raw_shots_per_batch"] or preferred)
        total_shots = None
        requested_display = kwargs["target_display_interval_s"]
    else:
        total_shots = int(kwargs["total_shots"])
        if kwargs["manual_raw_shots_per_batch"] is not None:
            requested_batch = int(kwargs["manual_raw_shots_per_batch"])
        elif advanced:
            requested_batch = min(DEFAULT_RAW_SHOTS_PER_BATCH, total_shots)
        else:
            requested_batch = min(max_records, total_shots)
        requested_display = None
    if requested_batch < 1:
        raise ValueError("Raw shots per hardware batch must be positive")
    if requested_batch > max_records:
        if advanced:
            requested_bytes = requested_batch * segment_samples
            raise ValueError(
                f"Requested batch uses {_human_bytes(requested_bytes)}, exceeding the {_human_bytes(_max_batch_bytes(info))} safety limit. "
                f"Maximum raw shots per hardware batch for this configuration: {max_records:,}."
            )
        warnings.append(f"Raw shots per hardware batch reduced from {requested_batch:,} to {max_records:,} to stay within 50% onboard memory.")
        requested_batch = max_records

    if total_shots is None:
        full_batches = None
        final_shots = None
        final_request = None
        primary_shots = requested_batch
    else:
        primary_shots = min(requested_batch, total_shots)
        full_batches = total_shots // primary_shots
        final_shots = total_shots % primary_shots
        if final_shots == 0:
            final_request = None
        else:
            final_request = _request(
                mode=SpectrumAcquisitionMode.RAW_MULTI,
                rate_hz=rate_hz,
                segment_samples=segment_samples,
                    number_of_segments=final_shots,
                    averages_per_segment=1,
                    timeout_s=_timeout(kwargs["timeout_s"], final_shots * kwargs["repetition_period_s"], advanced),
                    context=kwargs,
            )
            plan_spectrum_acquisition(final_request, info)

    acq_s = primary_shots * kwargs["repetition_period_s"]
    timeout_s = _timeout(kwargs["timeout_s"], acq_s, advanced)
    request = _request(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        rate_hz=rate_hz,
        segment_samples=segment_samples,
        number_of_segments=primary_shots,
        averages_per_segment=1,
        timeout_s=timeout_s,
        context=kwargs,
    )
    validated = plan_spectrum_acquisition(request, info)
    transfer_bytes = validated.transfer_bytes
    memory_fraction = transfer_bytes / info.onboard_memory_bytes
    _validate_memory_fraction(memory_fraction, info)
    actual_window_s = actual_posttrigger_window_s(segment_samples, kwargs["pretrigger_samples"], rate_hz)
    display_s = max(requested_display, acq_s) if requested_display is not None else None
    lines = _summary_lines(
        workflow=workflow,
        request=request,
        total_shots=total_shots,
        full_batches=full_batches,
        final_shots=final_shots,
        physical_shots=primary_shots,
        output_records=primary_shots,
        acq_s=acq_s,
        display_s=display_s,
        transfer_bytes=transfer_bytes,
        info=info,
        actual_window_s=actual_window_s,
        requested_display_s=requested_display,
        calibration=kwargs["calibration"],
        tof_window_us=kwargs["tof_window_us"],
        warnings=warnings,
    )
    return AcquisitionRunPlan(
        workflow=workflow,
        priority=kwargs["priority"],
        primary_request=request,
        final_batch_request=final_request if total_shots is not None else None,
        continuous=workflow == AcquisitionWorkflow.LIVE_RAW,
        total_requested_shots=total_shots,
        physical_shots_per_batch=primary_shots,
        output_records_per_batch=primary_shots,
        full_batch_count=full_batches,
        final_batch_shots=final_shots,
        requested_display_interval_s=requested_display,
        estimated_batch_acquisition_s=acq_s,
        estimated_batch_cycle_s=acq_s,
        estimated_display_interval_s=display_s,
        transfer_bytes_per_batch=transfer_bytes,
        onboard_memory_bytes=info.onboard_memory_bytes,
        onboard_memory_fraction=memory_fraction,
        actual_posttrigger_window_s=actual_window_s,
        warnings=tuple(warnings),
        summary_lines=tuple(lines),
    )


def _request(
    mode: SpectrumAcquisitionMode,
    rate_hz: float,
    segment_samples: int,
    number_of_segments: int,
    averages_per_segment: int,
    timeout_s: float,
    context: dict,
) -> SpectrumAcquisitionRequest:
    return SpectrumAcquisitionRequest(
        mode=mode,
        sample_rate_hz=rate_hz,
        segment_samples=segment_samples,
        pretrigger_samples=context["pretrigger_samples"],
        number_of_segments=number_of_segments,
        averages_per_segment=averages_per_segment,
        trigger_source=context["trigger_source"],
        input_range_v=context["input_range_v"],
        trigger_level_v=context["trigger_level_v"],
        timeout_s=timeout_s,
        coupling=context["coupling"],
        bandwidth_limit_enabled=context["bandwidth_limit_enabled"],
        trigger_edge=context["trigger_edge"],
        trigger_termination_ohm=context["trigger_termination_ohm"],
    )


def _summary_lines(
    workflow: AcquisitionWorkflow,
    request: SpectrumAcquisitionRequest,
    total_shots: int | None,
    full_batches: int | None,
    final_shots: int | None,
    physical_shots: int,
    output_records: int,
    acq_s: float,
    display_s: float | None,
    transfer_bytes: int,
    info: SpectrumHardwareInfo,
    actual_window_s: float,
    requested_display_s: float | None,
    calibration: tuple[float, float, float] | None,
    tof_window_us: float,
    warnings: list[str],
) -> list[str]:
    lines = [f"Workflow: {_workflow_display(workflow)}"]
    lines.append(f"Hardware mode: {_mode_display(request.mode)}")
    if total_shots is not None:
        lines.append(f"Total requested shots: {total_shots:,}")
        lines.append(f"Shots per hardware batch: {physical_shots:,}")
        lines.append(f"Full batches: {full_batches:,}")
        lines.append(f"Final partial batch: {final_shots:,}" if final_shots else "Final partial batch: none")
    lines.append(f"Sample rate: {_rate_label(request.sample_rate_hz)}")
    lines.append(f"Time resolution: {_time_resolution_label(request.sample_rate_hz)}")
    lines.append(f"Segment size: {request.segment_samples:,} samples total, including {request.pretrigger_samples:,} pretrigger")
    lines.append(f"Requested post-trigger TOF window: {tof_window_us:.3g} us")
    lines.append(f"Actual recorded post-trigger window: {actual_window_s * 1e6:.3f} us")
    if calibration is not None:
        mz = mz_at_window_end(actual_window_s * 1e6, request.pretrigger_samples, request.sample_rate_hz, calibration)
        if mz is not None:
            lines.append(f"Approximate maximum m/z: {mz:.1f}")
    if request.mode in (SpectrumAcquisitionMode.AVERAGE_32BIT, SpectrumAcquisitionMode.AVERAGE_16BIT):
        lines.append(f"Shots per FPGA sum: {request.averages_per_segment:,}")
        lines.append(f"FPGA sums per hardware batch: {request.number_of_segments:,}")
    lines.append(f"Physical triggers per batch: {physical_shots:,}")
    lines.append(f"Output records per hardware batch: {output_records:,}")
    lines.append(f"Hardware batch interval: {acq_s:.3f} s")
    if requested_display_s is not None:
        lines.append(f"Target display interval: {requested_display_s:.3f} s")
    if display_s is not None:
        lines.append(f"Expected display update: approximately {display_s:.3f} s")
    lines.append(f"Transfer per batch: {_human_bytes(transfer_bytes)}")
    lines.append(_memory_line(transfer_bytes, info))
    lines.append(f"Timeout: {request.timeout_s:.3f} s")
    if request.trigger_source == SpectrumTriggerSource.EXTERNAL0:
        lines.append(f"Trigger: External 0, {request.trigger_edge} edge")
        lines.append("External trigger required")
    else:
        lines.append(f"Trigger: {request.trigger_source.value}, {request.trigger_edge} edge")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"WARNING: {warning}" for warning in warnings)
    return lines


def _workflow(value: AcquisitionWorkflow | str, config_mode: str | None) -> AcquisitionWorkflow:
    if isinstance(value, AcquisitionWorkflow):
        return value
    if value == "continuous":
        return AcquisitionWorkflow.LIVE_AVERAGED if config_mode != "raw_segments" else AcquisitionWorkflow.LIVE_RAW
    if value == "shot_analysis":
        return AcquisitionWorkflow.FINITE_SHOT_ANALYSIS
    return AcquisitionWorkflow(value)


def _priority(value: AcquisitionPriority | str) -> AcquisitionPriority:
    if isinstance(value, AcquisitionPriority):
        return value
    old = {
        "fast_display": AcquisitionPriority.FAST_UPDATES,
        "high_time_resolution": AcquisitionPriority.HIGHEST_TIME_RESOLUTION,
        "high_sensitivity": AcquisitionPriority.BEST_SIGNAL_TO_NOISE,
        "continuous": AcquisitionPriority.BALANCED,
    }
    return old.get(value, AcquisitionPriority(value))


def _trigger(value: str | SpectrumTriggerSource) -> SpectrumTriggerSource:
    if isinstance(value, SpectrumTriggerSource):
        return value
    mapping = {
        "external0": SpectrumTriggerSource.EXTERNAL0,
        "software": SpectrumTriggerSource.SOFTWARE,
        "channel0": SpectrumTriggerSource.CHANNEL0,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported trigger source: {value}") from exc


def _candidate_rates(priority: AcquisitionPriority, manual_rate: float | None) -> list[float]:
    if manual_rate is not None:
        return [manual_rate]
    if priority == AcquisitionPriority.FAST_UPDATES:
        return [312.5e6, 625e6, 156.25e6, 1.25e9]
    if priority == AcquisitionPriority.HIGHEST_TIME_RESOLUTION:
        return [1.25e9, 625e6, 312.5e6, 156.25e6]
    if priority == AcquisitionPriority.BEST_SIGNAL_TO_NOISE:
        return [625e6, 312.5e6, 156.25e6, 1.25e9]
    return [625e6, 312.5e6, 1.25e9, 156.25e6]


def _averaged_mode_order(priority: AcquisitionPriority, manual: str) -> list[SpectrumAcquisitionMode]:
    if manual == "32bit":
        return [SpectrumAcquisitionMode.AVERAGE_32BIT]
    if manual == "16bit":
        return [SpectrumAcquisitionMode.AVERAGE_16BIT]
    if priority == AcquisitionPriority.BEST_SIGNAL_TO_NOISE:
        return [SpectrumAcquisitionMode.AVERAGE_32BIT, SpectrumAcquisitionMode.AVERAGE_16BIT]
    return [SpectrumAcquisitionMode.AVERAGE_32BIT, SpectrumAcquisitionMode.AVERAGE_16BIT]


def _averaged_candidates(priority, rates, modes):
    if priority == AcquisitionPriority.BEST_SIGNAL_TO_NOISE and len(modes) > 1:
        for mode in modes:
            for rate in rates:
                yield mode, rate
        return
    for rate in rates:
        for mode in modes:
            yield mode, rate


def _segment_samples_for_request(tof_window_us, rate_hz, pretrigger_samples, mode, manual_segment_samples):
    if manual_segment_samples is not None:
        return int(manual_segment_samples)
    return compute_segment_samples(tof_window_us, rate_hz, pretrigger_samples, mode)


def _segment_limits(mode: SpectrumAcquisitionMode) -> tuple[int, int, int | None]:
    if mode == SpectrumAcquisitionMode.RAW_MULTI:
        return RAW_MIN_SEGMENT_SAMPLES, RAW_SEGMENT_STEP, None
    if mode == SpectrumAcquisitionMode.AVERAGE_32BIT:
        return AVG32_MIN_SEGMENT_SAMPLES, AVG32_SEGMENT_STEP, AVG32_MAX_SEGMENT_SAMPLES
    if mode == SpectrumAcquisitionMode.AVERAGE_16BIT:
        return AVG16_MIN_SEGMENT_SAMPLES, AVG16_SEGMENT_STEP, AVG16_MAX_SEGMENT_SAMPLES
    raise ValueError(f"unsupported mode: {mode}")


def _average_limits(mode: SpectrumAcquisitionMode) -> tuple[int, int]:
    if mode == SpectrumAcquisitionMode.AVERAGE_32BIT:
        return AVG32_MIN_AVERAGES, AVG32_MAX_AVERAGES
    if mode == SpectrumAcquisitionMode.AVERAGE_16BIT:
        return AVG16_MIN_AVERAGES, AVG16_MAX_AVERAGES
    return 1, 1


def _basic_averages_per_segment(
    priority: AcquisitionPriority,
    configured_n: int,
    mode: SpectrumAcquisitionMode,
    target_display_interval_s: float,
    repetition_period_s: float,
) -> int:
    """Choose Basic-mode FPGA averaging count from user planning priority."""
    if priority == AcquisitionPriority.FAST_UPDATES:
        target_n = 25
    elif priority == AcquisitionPriority.BEST_SIGNAL_TO_NOISE:
        useful_n = max(2, int(round(target_display_interval_s / repetition_period_s / 2.0)))
        target_n = max(configured_n, useful_n)
    else:
        target_n = configured_n
    minimum, maximum = _average_limits(mode)
    return max(minimum, min(maximum, target_n))


def _bytes_per_sample(mode: SpectrumAcquisitionMode) -> int:
    if mode == SpectrumAcquisitionMode.AVERAGE_32BIT:
        return 4
    if mode == SpectrumAcquisitionMode.AVERAGE_16BIT:
        return 2
    return 1


def _max_batch_bytes(info: SpectrumHardwareInfo) -> int:
    return math.floor(info.onboard_memory_bytes * MAX_MEMORY_FRACTION)


def _max_records_by_memory(info, segment_samples, bytes_per_sample):
    bytes_per_record = segment_samples * bytes_per_sample
    return _max_batch_bytes(info) // bytes_per_record


def _validate_memory_fraction(fraction: float, info: SpectrumHardwareInfo) -> None:
    if fraction > MAX_MEMORY_FRACTION:
        raise ValueError(f"planned batch uses {fraction * 100:.2f}% of onboard memory, exceeding the 50% safety limit")


def _validate_record_timing(segment_samples, pretrigger_samples, rate_hz, repetition_period_s) -> None:
    record_s = segment_samples / rate_hz
    if record_s >= repetition_period_s:
        raise ValueError(
            f"Record duration {record_s * 1e6:.3f} us reaches or exceeds BME repetition period {repetition_period_s * 1e6:.3f} us"
        )
    if pretrigger_samples >= segment_samples:
        raise ValueError("pretrigger_samples must be smaller than segment_samples")


def _timeout(requested_timeout_s: float, expected_acq_s: float, advanced: bool) -> float:
    minimum = expected_acq_s * 1.25 + 1.0
    if advanced and requested_timeout_s < minimum:
        raise ValueError(f"Timeout {requested_timeout_s:.3f} s is too short; minimum required timeout is {minimum:.3f} s")
    return max(requested_timeout_s, minimum)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _model_display_name(info: SpectrumHardwareInfo) -> str:
    return str(info.metadata.get("model_name", info.model_name))


def _human_bytes(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.2f} MB"


def _memory_line(transfer_bytes: int, info: SpectrumHardwareInfo) -> str:
    return (
        f"Onboard memory used: {_human_bytes(transfer_bytes)} / {_human_bytes(info.onboard_memory_bytes)} "
        f"({transfer_bytes / info.onboard_memory_bytes * 100:.2f}%)"
    )


def _one_record_memory_error(info, segment_samples, bytes_per_sample) -> str:
    return (
        f"One output record uses {_human_bytes(segment_samples * bytes_per_sample)}, exceeding the "
        f"{_human_bytes(_max_batch_bytes(info))} safety limit. Reduce TOF window or sample rate."
    )


def _mode_display(mode: SpectrumAcquisitionMode) -> str:
    if mode == SpectrumAcquisitionMode.RAW_MULTI:
        return "Standard Multiple"
    if mode == SpectrumAcquisitionMode.AVERAGE_32BIT:
        return "32-bit Block Average"
    if mode == SpectrumAcquisitionMode.AVERAGE_16BIT:
        return "16-bit Block Average"
    return str(mode)


def _workflow_display(workflow: AcquisitionWorkflow) -> str:
    if workflow == AcquisitionWorkflow.LIVE_AVERAGED:
        return "Live Averaged Spectrum"
    if workflow == AcquisitionWorkflow.LIVE_RAW:
        return "Live Raw-Shot Spectrum"
    return "Finite Shot Analysis"


def _rate_label(rate_hz: float) -> str:
    if rate_hz >= 1e9:
        return f"{rate_hz / 1e9:g} GS/s"
    return f"{rate_hz / 1e6:g} MS/s"


def _time_resolution_label(rate_hz: float) -> str:
    return f"{1e9 / rate_hz:.3f} ns/sample"
