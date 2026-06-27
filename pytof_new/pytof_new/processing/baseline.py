"""Baseline correction functions."""

from __future__ import annotations

import numpy as np


def baseline_values(data: np.ndarray, start: int, stop: int, method: str = "mean") -> np.ndarray:
    """Return per-waveform baseline values over [start:stop]."""
    _validate_baseline_interval(data, start, stop)
    region = data[..., start:stop]
    if method == "mean":
        return region.mean(axis=-1)
    if method == "median":
        return np.median(region, axis=-1)
    raise ValueError("baseline method must be 'mean' or 'median'")


def subtract_pretrigger_baseline(data: np.ndarray, start: int, stop: int, method: str = "mean") -> np.ndarray:
    """Subtract a per-waveform mean or median baseline over [start:stop]."""
    baseline = baseline_values(data, start, stop, method=method)
    return data - np.expand_dims(baseline, axis=-1)


def _validate_baseline_interval(data: np.ndarray, start: int, stop: int) -> None:
    """Validate a baseline interval against the waveform length."""
    if start < 0 or stop <= start:
        raise ValueError("baseline interval is invalid")
    if stop > data.shape[-1]:
        raise ValueError("baseline stop exceeds waveform length")


def subtract_background(data: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Subtract a background waveform or batch from data."""
    return data - background
