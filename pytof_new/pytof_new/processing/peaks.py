"""Peak finding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PeakResults:
    """Detected peak positions and heights."""

    indices: np.ndarray
    positions: np.ndarray
    heights: np.ndarray


@dataclass(frozen=True)
class PeakFitResult:
    """Fitted peak center and width for a selected trace window."""

    center: float
    fwhm: float | None
    amplitude: float
    offset: float
    fitted: bool


def find_trace_peaks(axis: np.ndarray, trace: np.ndarray, prominence: float | None = None) -> PeakResults:
    """Find peaks in a one-dimensional trace."""
    indices, properties = find_peaks(trace, prominence=prominence)
    heights = properties.get("peak_heights", trace[indices])
    return PeakResults(indices=indices, positions=axis[indices], heights=heights)


def fit_peak_window(axis: np.ndarray, trace: np.ndarray) -> PeakFitResult:
    """Fit a Gaussian peak in a selected axis window."""
    if axis.ndim != 1 or trace.ndim != 1 or axis.size != trace.size:
        raise ValueError("axis and trace must be one-dimensional arrays with matching length")
    finite = np.isfinite(axis) & np.isfinite(trace)
    x = axis[finite].astype(np.float64, copy=False)
    y = trace[finite].astype(np.float64, copy=False)
    if x.size < 5:
        raise ValueError("at least five finite points are required to fit a peak")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    offset0 = float(np.median(y))
    deviations = y - offset0
    peak_index = int(np.argmax(np.abs(deviations)))
    amplitude0 = float(deviations[peak_index])
    if amplitude0 == 0.0:
        amplitude0 = float(y[peak_index] - np.min(y)) or 1e-12
    center0 = float(x[peak_index])
    window = float(x[-1] - x[0])
    sigma0 = max(window / 6.0, _minimum_sigma(x))
    lower = [float(np.min(y) - 2.0 * np.ptp(y)), x[0], _minimum_sigma(x), float(np.min(y) - 2.0 * np.ptp(y))]
    upper = [float(np.max(y) + 2.0 * np.ptp(y)), x[-1], max(window, _minimum_sigma(x)), float(np.max(y) + 2.0 * np.ptp(y))]
    try:
        params, _covariance = curve_fit(
            _gaussian_with_offset,
            x,
            y,
            p0=[amplitude0, center0, sigma0, offset0],
            bounds=(lower, upper),
            maxfev=5000,
        )
        amplitude, center, sigma, offset = [float(value) for value in params]
        return PeakFitResult(
            center=center,
            fwhm=2.354820045 * abs(sigma),
            amplitude=amplitude,
            offset=offset,
            fitted=True,
        )
    except Exception:
        fwhm = _estimate_fwhm(x, y, peak_index, offset0)
        return PeakFitResult(center=center0, fwhm=fwhm, amplitude=amplitude0, offset=offset0, fitted=False)


def _gaussian_with_offset(x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float) -> np.ndarray:
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _minimum_sigma(axis: np.ndarray) -> float:
    if axis.size < 2:
        return 1e-12
    spacing = np.diff(axis)
    positive = spacing[spacing > 0]
    if positive.size == 0:
        return 1e-12
    return float(np.min(positive) / 2.0)


def _estimate_fwhm(axis: np.ndarray, trace: np.ndarray, peak_index: int, offset: float) -> float | None:
    amplitude = trace[peak_index] - offset
    if amplitude == 0.0:
        return None
    half_height = offset + amplitude / 2.0
    if amplitude > 0:
        above = trace >= half_height
    else:
        above = trace <= half_height
    indices = np.flatnonzero(above)
    if indices.size < 2 or peak_index not in indices:
        return None
    contiguous = indices[(indices >= indices[0]) & (indices <= indices[-1])]
    left_candidates = contiguous[contiguous <= peak_index]
    right_candidates = contiguous[contiguous >= peak_index]
    if left_candidates.size == 0 or right_candidates.size == 0:
        return None
    left = int(left_candidates[0])
    right = int(right_candidates[-1])
    if right <= left:
        return None
    return float(axis[right] - axis[left])
