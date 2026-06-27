"""Trigger-count helpers for coordinated real hardware batches."""

from __future__ import annotations

from pytof_new.hardware.spectrum_models import SpectrumAcquisitionMode, SpectrumAcquisitionPlan, SpectrumAcquisitionRequest


def required_bme_triggers_for_request(request: SpectrumAcquisitionRequest) -> int:
    """Return accepted BME trigger events required by one Spectrum request.

    This assumes the Spectrum-trigger BME output channel is configured to emit
    exactly one pulse for each accepted BME trigger event.
    """
    if request.number_of_segments <= 0:
        raise ValueError("number_of_segments must be positive")
    if request.averages_per_segment <= 0:
        raise ValueError("averages_per_segment must be positive")
    if request.mode == SpectrumAcquisitionMode.RAW_MULTI:
        return int(request.number_of_segments)
    if request.mode in (SpectrumAcquisitionMode.AVERAGE_32BIT, SpectrumAcquisitionMode.AVERAGE_16BIT):
        return int(request.number_of_segments * request.averages_per_segment)
    raise ValueError(f"unsupported Spectrum acquisition mode: {request.mode}")


def required_bme_triggers_for_plan(plan: SpectrumAcquisitionPlan) -> int:
    """Return required BME trigger events for a validated Spectrum plan."""
    return required_bme_triggers_for_request(plan.request)
