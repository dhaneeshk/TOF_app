"""Shot quality metrics and transparent rejection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShotQualityMetrics:
    """Per-shot quality metrics for waveform batches.

    Arrays have shape ``(segments,)`` and voltage units except for booleans.
    """

    baseline_mean: np.ndarray
    baseline_rms: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    peak_to_peak: np.ndarray
    integrated_absolute_signal: np.ndarray
    near_clipped: np.ndarray


@dataclass(frozen=True)
class RejectionSummary:
    """Transparent shot rejection result."""

    accepted_mask: np.ndarray
    clipping_rejection_count: int
    baseline_noise_rejection_count: int

    @property
    def accepted_count(self) -> int:
        """Number of accepted shots."""
        return int(np.count_nonzero(self.accepted_mask))

    @property
    def rejected_count(self) -> int:
        """Number of rejected shots."""
        return int(self.accepted_mask.size - self.accepted_count)


def calculate_shot_quality(
    data: np.ndarray,
    baseline_start: int,
    baseline_stop: int,
    input_range_v: float,
    clipping_margin_fraction: float,
) -> ShotQualityMetrics:
    """Calculate lightweight per-shot quality metrics from voltage waveforms."""
    if data.ndim != 2:
        raise ValueError("data must have shape (segments, samples)")
    if baseline_start < 0 or baseline_stop <= baseline_start:
        raise ValueError("baseline interval is invalid")
    if baseline_stop > data.shape[-1]:
        raise ValueError("baseline stop exceeds waveform length")
    if input_range_v <= 0:
        raise ValueError("input_range_v must be positive")
    if not 0.0 <= clipping_margin_fraction < 1.0:
        raise ValueError("clipping_margin_fraction must be in [0, 1)")

    baseline_region = data[:, baseline_start:baseline_stop]
    baseline_mean = baseline_region.mean(axis=1, dtype=np.float64).astype(np.float32)
    baseline_centered = baseline_region - baseline_mean[:, np.newaxis]
    baseline_rms = np.sqrt(np.mean(baseline_centered * baseline_centered, axis=1, dtype=np.float64)).astype(np.float32)
    minimum = data.min(axis=1)
    maximum = data.max(axis=1)
    peak_to_peak = maximum - minimum
    integrated_absolute_signal = np.sum(np.abs(data), axis=1, dtype=np.float64).astype(np.float32)
    clip_threshold = input_range_v * (1.0 - clipping_margin_fraction)
    near_clipped = (maximum >= clip_threshold) | (minimum <= -clip_threshold)
    return ShotQualityMetrics(
        baseline_mean=baseline_mean,
        baseline_rms=baseline_rms,
        minimum=minimum,
        maximum=maximum,
        peak_to_peak=peak_to_peak,
        integrated_absolute_signal=integrated_absolute_signal,
        near_clipped=near_clipped,
    )


def reject_shots(
    metrics: ShotQualityMetrics,
    rejection_enabled: bool,
    reject_clipped: bool,
    maximum_baseline_rms_v: float | None,
) -> RejectionSummary:
    """Return an accepted-shot mask and rejection counts."""
    accepted = np.ones(metrics.baseline_rms.shape, dtype=bool)
    clipping_mask = np.zeros_like(accepted)
    baseline_noise_mask = np.zeros_like(accepted)
    if rejection_enabled:
        if reject_clipped:
            clipping_mask = metrics.near_clipped
            accepted &= ~clipping_mask
        if maximum_baseline_rms_v is not None:
            if maximum_baseline_rms_v < 0:
                raise ValueError("maximum_baseline_rms_v must be non-negative")
            baseline_noise_mask = metrics.baseline_rms > maximum_baseline_rms_v
            accepted &= ~baseline_noise_mask
    return RejectionSummary(
        accepted_mask=accepted,
        clipping_rejection_count=int(np.count_nonzero(clipping_mask)),
        baseline_noise_rejection_count=int(np.count_nonzero(baseline_noise_mask)),
    )
