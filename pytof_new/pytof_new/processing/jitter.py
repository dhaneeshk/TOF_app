"""Timing-jitter estimation from repeated single-record spectra."""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np


@dataclass(frozen=True)
class JitterResult:
    """Summary of relative timing shifts estimated by cross-correlation."""

    shifts_s: np.ndarray
    mean_shift_s: float
    rms_jitter_s: float
    uncertainty_s: float
    record_count: int
    max_lag_s: float


def analyze_jitter(traces: np.ndarray, sample_rate_hz: float, max_lag_s: float) -> JitterResult:
    """Estimate record-to-record timing jitter using normalized cross-correlation."""
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError("traces must be a 2D array")
    if traces.shape[0] < 2:
        raise ValueError("at least two records are required")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if max_lag_s <= 0:
        raise ValueError("max_lag_s must be positive")
    max_lag_samples = max(1, int(round(max_lag_s * sample_rate_hz)))
    if max_lag_samples >= traces.shape[1]:
        raise ValueError("max_lag_s is too large for the trace length")

    reference = np.mean(traces, axis=0)
    shifts = np.asarray([_estimate_shift(trace, reference, max_lag_samples) / sample_rate_hz for trace in traces], dtype=np.float64)
    centered = shifts - np.mean(shifts)
    rms = float(np.sqrt(np.mean(centered**2)))
    uncertainty = rms / math.sqrt(2.0 * (traces.shape[0] - 1)) if traces.shape[0] > 1 else math.nan
    return JitterResult(
        shifts_s=shifts,
        mean_shift_s=float(np.mean(shifts)),
        rms_jitter_s=rms,
        uncertainty_s=uncertainty,
        record_count=int(traces.shape[0]),
        max_lag_s=max_lag_samples / sample_rate_hz,
    )


def time_aligned_average(traces: np.ndarray, shifts_s: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Return the average trace after compensating cross-correlation shifts."""
    traces = np.asarray(traces, dtype=np.float64)
    shifts_s = np.asarray(shifts_s, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError("traces must be a 2D array")
    if shifts_s.shape != (traces.shape[0],):
        raise ValueError("shifts_s must have one value per trace")
    sample_axis = np.arange(traces.shape[1], dtype=np.float64)
    aligned = np.empty_like(traces, dtype=np.float64)
    for index, (trace, shift_s) in enumerate(zip(traces, shifts_s, strict=True)):
        shift_samples = shift_s * sample_rate_hz
        aligned[index] = np.interp(sample_axis - shift_samples, sample_axis, trace, left=np.nan, right=np.nan)
    return _robust_mean(aligned).astype(np.float32)


def _robust_mean(aligned: np.ndarray) -> np.ndarray:
    """Mean with pointwise MAD clipping to suppress misaligned ghost peaks."""
    median = np.nanmedian(aligned, axis=0)
    absolute_deviation = np.abs(aligned - median)
    mad = np.nanmedian(absolute_deviation, axis=0)
    # 1.4826 converts MAD to a Gaussian sigma estimate. Fall back to plain mean
    # where the local distribution is too narrow to estimate a useful scale.
    sigma = 1.4826 * mad
    mask = np.isfinite(aligned)
    clip_mask = sigma > 0
    mask[:, clip_mask] &= absolute_deviation[:, clip_mask] <= 6.0 * sigma[clip_mask]
    zero_scale_mask = ~clip_mask
    if np.any(zero_scale_mask):
        tolerance = np.maximum(np.abs(median[zero_scale_mask]) * 1e-6, 1e-12)
        mask[:, zero_scale_mask] &= absolute_deviation[:, zero_scale_mask] <= tolerance
    clipped = np.where(mask, aligned, np.nan)
    mean = np.nanmean(clipped, axis=0)
    fallback = np.nanmean(aligned, axis=0)
    return np.where(np.isfinite(mean), mean, fallback)


def _estimate_shift(trace: np.ndarray, reference: np.ndarray, max_lag_samples: int) -> float:
    trace = trace - np.mean(trace)
    reference = reference - np.mean(reference)
    correlations = np.empty(2 * max_lag_samples + 1, dtype=np.float64)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.float64)
    for index, lag in enumerate(range(-max_lag_samples, max_lag_samples + 1)):
        if lag < 0:
            x = trace[-lag:]
            y = reference[:lag]
        elif lag > 0:
            x = trace[:-lag]
            y = reference[lag:]
        else:
            x = trace
            y = reference
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        correlations[index] = 0.0 if denominator == 0 else float(np.dot(x, y) / denominator)
    peak_index = int(np.argmax(correlations))
    peak_lag = float(lags[peak_index])
    if 0 < peak_index < correlations.size - 1:
        y0, y1, y2 = correlations[peak_index - 1], correlations[peak_index], correlations[peak_index + 1]
        denominator = y0 - 2.0 * y1 + y2
        if denominator != 0:
            peak_lag += 0.5 * (y0 - y2) / denominator
    return peak_lag
