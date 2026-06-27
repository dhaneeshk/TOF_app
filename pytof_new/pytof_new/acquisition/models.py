"""Acquisition data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AcquisitionBatch:
    """A batch of triggered time-of-flight waveform segments."""

    raw_adc: np.ndarray
    timestamps: np.ndarray | None
    sample_rate_hz: float
    pretrigger_samples: int
    first_trigger_index: int
    record_mode: str = "hardware_average"
    hardware_averages_per_record: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate batch array shape."""
        if self.raw_adc.ndim != 2:
            raise ValueError("raw_adc must have shape (segments, samples)")
        if self.timestamps is not None and len(self.timestamps) != self.raw_adc.shape[0]:
            raise ValueError("timestamps length must match number of segments")
