"""Optional signal filtering functions."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt


def butterworth_filter(
    data: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    kind: str,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass or high-pass filter."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    nyquist_hz = sample_rate_hz / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist_hz:
        raise ValueError("filter cutoff must be positive and below Nyquist")
    if kind not in {"lowpass", "highpass"}:
        raise ValueError("filter kind must be 'lowpass' or 'highpass'")
    if order <= 0:
        raise ValueError("filter order must be positive")
    sos = butter(order, cutoff_hz, btype=kind, fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)


def low_pass_filter(data: np.ndarray, sample_rate_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase low-pass filter along the last axis."""
    return butterworth_filter(data, sample_rate_hz, cutoff_hz, "lowpass", order=order)


def high_pass_filter(data: np.ndarray, sample_rate_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase high-pass filter along the last axis."""
    return butterworth_filter(data, sample_rate_hz, cutoff_hz, "highpass", order=order)


def smooth_savgol(data: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along the last axis."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and >= 3")
    if polyorder >= window:
        raise ValueError("polyorder must be smaller than window")
    return savgol_filter(data, window_length=window, polyorder=polyorder, axis=-1).astype(np.float32, copy=False)
