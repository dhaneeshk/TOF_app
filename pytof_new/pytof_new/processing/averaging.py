"""Waveform averaging functions."""

from __future__ import annotations

import numpy as np


def average_segments(data: np.ndarray) -> np.ndarray:
    """Average all segments in a batch."""
    if data.ndim != 2:
        raise ValueError("data must have shape (segments, samples)")
    return data.mean(axis=0, dtype=np.float32)


def accumulate_average(current_average: np.ndarray | None, current_count: int, new_segments: np.ndarray) -> tuple[np.ndarray, int]:
    """Update a running average without retaining all prior segments."""
    batch_average = average_segments(new_segments)
    batch_count = new_segments.shape[0]
    if current_average is None or current_count == 0:
        return batch_average, batch_count
    total_count = current_count + batch_count
    updated = ((current_average * current_count) + (batch_average * batch_count)) / total_count
    return updated.astype(np.float32, copy=False), total_count
