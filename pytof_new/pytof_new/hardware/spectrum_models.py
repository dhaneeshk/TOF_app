"""Spectrum digitizer acquisition models.

These models are intentionally import-safe: they do not import the Spectrum
driver wrapper or load vendor DLLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SpectrumAcquisitionMode(str, Enum):
    """Supported finite Spectrum acquisition modes."""

    RAW_MULTI = "raw_multi"
    AVERAGE_32BIT = "average_32bit"
    AVERAGE_16BIT = "average_16bit"


class SpectrumTriggerSource(str, Enum):
    """Trigger sources used by the real Spectrum digitizer path."""

    EXTERNAL0 = "external0"
    SOFTWARE = "software"
    CHANNEL0 = "channel0"


@dataclass(frozen=True)
class SpectrumHardwareInfo:
    """Discovered hardware capabilities relevant to planning acquisitions."""

    model_name: str = "M4i.2210-x8"
    serial_number: int | None = None
    max_sample_rate_hz: float = 1.25e9
    onboard_memory_bytes: int = 4096 * 1024 * 1024
    adc_bits: int = 8
    channel_count: int = 1
    max_adc_value: int = 127
    average_16bit_supported: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpectrumAcquisitionRequest:
    """A finite acquisition requested from the Spectrum card."""

    mode: SpectrumAcquisitionMode
    sample_rate_hz: float
    segment_samples: int
    pretrigger_samples: int
    number_of_segments: int = 1
    averages_per_segment: int = 1
    trigger_source: SpectrumTriggerSource = SpectrumTriggerSource.EXTERNAL0
    input_range_v: float = 0.5
    trigger_level_v: float = 1.5
    timeout_s: float = 5.0
    coupling: str = "dc"
    bandwidth_limit_enabled: bool = False
    trigger_edge: str = "rising"
    trigger_termination_ohm: int = 50


@dataclass(frozen=True)
class SpectrumAcquisitionPlan:
    """Validated acquisition shape and transfer details."""

    request: SpectrumAcquisitionRequest
    dtype: np.dtype
    output_shape: tuple[int, int]
    transfer_bytes: int
    physical_shots_per_output_segment: int
    is_fpga_sum: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpectrumAcquisitionResult:
    """Finite data returned by the Spectrum hardware owner."""

    data: np.ndarray
    plan: SpectrumAcquisitionPlan
    metadata: dict[str, Any] = field(default_factory=dict)
