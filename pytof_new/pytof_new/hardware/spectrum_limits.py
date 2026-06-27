"""Validation and planning for Spectrum M4i.2210-x8 acquisitions."""

from __future__ import annotations

import numpy as np

from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionPlan,
    SpectrumAcquisitionRequest,
    SpectrumHardwareInfo,
    SpectrumTriggerSource,
)


M4I2210_MAX_SAMPLE_RATE_HZ = 1.25e9
M4I2210_ONBOARD_MEMORY_BYTES = 4096 * 1024 * 1024

AVG32_MIN_SEGMENT_SAMPLES = 64
AVG32_SEGMENT_STEP = 32
AVG32_MAX_SEGMENT_SAMPLES = 65_536
AVG32_MIN_AVERAGES = 2
AVG32_MAX_AVERAGES = 16_777_216

AVG16_MIN_SEGMENT_SAMPLES = 128
AVG16_SEGMENT_STEP = 64
AVG16_MAX_SEGMENT_SAMPLES = 131_072
AVG16_MIN_AVERAGES = 2
AVG16_MAX_AVERAGES = 256

RAW_MIN_SEGMENT_SAMPLES = 64
RAW_SEGMENT_STEP = 32


def default_m4i2210_info() -> SpectrumHardwareInfo:
    """Return default limits for the installed target card model."""
    return SpectrumHardwareInfo(
        max_sample_rate_hz=M4I2210_MAX_SAMPLE_RATE_HZ,
        onboard_memory_bytes=M4I2210_ONBOARD_MEMORY_BYTES,
        average_16bit_supported=True,
    )


def plan_spectrum_acquisition(
    request: SpectrumAcquisitionRequest,
    hardware: SpectrumHardwareInfo | None = None,
) -> SpectrumAcquisitionPlan:
    """Validate a finite request and return expected transfer details."""
    info = hardware or default_m4i2210_info()
    _validate_common(request, info)

    if request.mode == SpectrumAcquisitionMode.RAW_MULTI:
        dtype = np.dtype(np.int8)
        shots_per_segment = 1
        is_fpga_sum = False
        _validate_segment("raw multiple", request.segment_samples, RAW_MIN_SEGMENT_SAMPLES, RAW_SEGMENT_STEP)
        if request.averages_per_segment != 1:
            raise ValueError("raw multiple mode requires averages_per_segment = 1")
    elif request.mode == SpectrumAcquisitionMode.AVERAGE_32BIT:
        dtype = np.dtype(np.int32)
        shots_per_segment = request.averages_per_segment
        is_fpga_sum = True
        _validate_block_average_trigger(request)
        _validate_segment(
            "32-bit block average",
            request.segment_samples,
            AVG32_MIN_SEGMENT_SAMPLES,
            AVG32_SEGMENT_STEP,
            AVG32_MAX_SEGMENT_SAMPLES,
        )
        _validate_range("32-bit block averages", request.averages_per_segment, AVG32_MIN_AVERAGES, AVG32_MAX_AVERAGES)
    elif request.mode == SpectrumAcquisitionMode.AVERAGE_16BIT:
        if not info.average_16bit_supported:
            raise ValueError("16-bit block average mode is not available on this Spectrum API/card")
        dtype = np.dtype(np.int16)
        shots_per_segment = request.averages_per_segment
        is_fpga_sum = True
        _validate_block_average_trigger(request)
        _validate_segment(
            "16-bit block average",
            request.segment_samples,
            AVG16_MIN_SEGMENT_SAMPLES,
            AVG16_SEGMENT_STEP,
            AVG16_MAX_SEGMENT_SAMPLES,
        )
        _validate_range("16-bit block averages", request.averages_per_segment, AVG16_MIN_AVERAGES, AVG16_MAX_AVERAGES)
    else:
        raise ValueError(f"unsupported Spectrum acquisition mode: {request.mode}")

    output_shape = (request.number_of_segments, request.segment_samples)
    transfer_bytes = int(np.prod(output_shape) * dtype.itemsize)
    if transfer_bytes > info.onboard_memory_bytes:
        raise ValueError(
            f"planned transfer requires {transfer_bytes} bytes, exceeding onboard memory "
            f"({info.onboard_memory_bytes} bytes)"
        )

    return SpectrumAcquisitionPlan(
        request=request,
        dtype=dtype,
        output_shape=output_shape,
        transfer_bytes=transfer_bytes,
        physical_shots_per_output_segment=shots_per_segment,
        is_fpga_sum=is_fpga_sum,
        metadata={"total_physical_shots": request.number_of_segments * shots_per_segment},
    )


def _validate_common(request: SpectrumAcquisitionRequest, hardware: SpectrumHardwareInfo) -> None:
    if request.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if request.sample_rate_hz > hardware.max_sample_rate_hz:
        raise ValueError("sample_rate_hz exceeds hardware maximum")
    if request.input_range_v <= 0:
        raise ValueError("input_range_v must be positive")
    if request.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if request.number_of_segments <= 0:
        raise ValueError("number_of_segments must be positive")
    if request.pretrigger_samples < 0:
        raise ValueError("pretrigger_samples must be non-negative")
    if request.pretrigger_samples >= request.segment_samples:
        raise ValueError("pretrigger_samples must be smaller than segment_samples")
    if request.trigger_level_v <= -10 or request.trigger_level_v >= 10:
        raise ValueError("trigger_level_v out of reasonable range")
    if request.coupling not in ("dc", "ac"):
        raise ValueError("coupling must be 'dc' or 'ac'")
    if request.trigger_edge not in ("rising", "falling"):
        raise ValueError("trigger_edge must be 'rising' or 'falling'")
    if request.trigger_termination_ohm not in (50, 1000):
        raise ValueError("trigger_termination_ohm must be 50 or 1000")


def _validate_block_average_trigger(request: SpectrumAcquisitionRequest) -> None:
    if request.trigger_source == SpectrumTriggerSource.SOFTWARE:
        raise ValueError("Spectrum block averaging requires external or channel trigger; software trigger is not supported")


def _validate_segment(
    label: str,
    value: int,
    minimum: int,
    step: int,
    maximum: int | None = None,
) -> None:
    if value < minimum:
        raise ValueError(f"{label} segment_samples must be at least {minimum}")
    if value % step != 0:
        raise ValueError(f"{label} segment_samples must be a multiple of {step}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} segment_samples must be at most {maximum}")


def _validate_range(label: str, value: int, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
