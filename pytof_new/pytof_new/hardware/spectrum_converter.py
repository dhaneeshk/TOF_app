"""Convert SpectrumAcquisitionResult to AcquisitionBatch for the processing pipeline."""

from __future__ import annotations

import time

import numpy as np

from pytof_new.acquisition.models import AcquisitionBatch
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionResult,
)


def spectrum_result_to_batch(
    result: SpectrumAcquisitionResult,
    sample_rate_hz: float,
    pretrigger_samples: int,
    first_trigger_index: int = 0,
) -> AcquisitionBatch:
    """Wrap a Spectrum result into an AcquisitionBatch.

    For ``RAW_MULTI`` the native int8 data is passed through unchanged.
    For FPGA-block-average modes (``AVERAGE_32BIT``, ``AVERAGE_16BIT``) the
    integer sum data is normalised to per-shot float32 ADC counts by dividing
    by ``physical_shots_per_output_segment``.
    """
    plan = result.plan
    metadata = dict(result.metadata)
    metadata["total_physical_shots"] = plan.metadata.get("total_physical_shots", 0)
    metadata["is_fpga_sum"] = plan.is_fpga_sum

    if not plan.is_fpga_sum:
        data = result.data  # native int8
        record_mode = "raw_segments"
        hardware_averages = 1
    else:
        shots = float(plan.physical_shots_per_output_segment)
        if shots <= 0:
            raise ValueError("physical_shots_per_output_segment must be positive for FPGA-sum modes")
        data = (result.data.astype(np.float32, copy=False) / shots).astype(np.float32)
        record_mode = "fpga_average"
        hardware_averages = plan.physical_shots_per_output_segment

    if data.ndim == 1:
        data = data[np.newaxis, :]

    timestamps = None
    if plan.request is not None:
        segs = data.shape[0]
        timestamps = np.arange(segs, dtype=np.float64) * 1.0 + time.time()

    return AcquisitionBatch(
        raw_adc=data,
        timestamps=timestamps,
        sample_rate_hz=sample_rate_hz,
        pretrigger_samples=pretrigger_samples,
        first_trigger_index=first_trigger_index,
        record_mode=record_mode,
        hardware_averages_per_record=hardware_averages,
        metadata=metadata,
    )
