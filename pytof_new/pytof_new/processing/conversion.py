"""ADC and axis conversion functions."""

from __future__ import annotations

import numpy as np


def adc_counts_to_voltage(raw_adc: np.ndarray, input_range_v: float, adc_full_scale_counts: int) -> np.ndarray:
    """Convert ADC counts to volts using a symmetric full-scale input range."""
    if adc_full_scale_counts <= 0:
        raise ValueError("adc_full_scale_counts must be positive")
    return raw_adc.astype(np.float32, copy=False) * np.float32(input_range_v / adc_full_scale_counts)


def create_tof_axis(segment_samples: int, sample_rate_hz: float, pretrigger_samples: int, time_zero_offset_s: float = 0.0) -> np.ndarray:
    """Create a time-of-flight axis in seconds."""
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    indices = np.arange(segment_samples, dtype=np.float64) - float(pretrigger_samples)
    return indices / sample_rate_hz - time_zero_offset_s


def tof_to_mass(tof_s: np.ndarray, coefficients: tuple[float, float, float]) -> np.ndarray:
    """Convert TOF seconds to m/z using a quadratic calibration in nanoseconds."""
    a, b, c = coefficients
    tof_ns = tof_s * 1e9
    return (a * tof_ns * tof_ns) + (b * tof_ns) + c
