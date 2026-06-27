"""Mock digitizer that generates synthetic TOF traces."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from pytof_new.config.models import DigitizerConfig
from pytof_new.exceptions import AcquisitionTimeoutError, DigitizerError
from pytof_new.hardware.digitizer_base import DigitizerBase

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticPeak:
    """Synthetic Gaussian peak definition."""

    position_s: float
    amplitude_v: float


@dataclass(frozen=True)
class MockDigitizerProfile:
    """Parameters controlling synthetic waveform generation."""

    peaks: tuple[SyntheticPeak, ...] = (
        SyntheticPeak(position_s=8.0e-6, amplitude_v=0.08),
        SyntheticPeak(position_s=18.0e-6, amplitude_v=0.18),
        SyntheticPeak(position_s=37.0e-6, amplitude_v=0.10),
    )
    detector_polarity: int = 1
    noise_rms_v: float = 0.003
    baseline_offset_v: float = 0.005
    baseline_drift_v: float = 0.002
    pickup_amplitude_v: float = 0.0
    pickup_decay_s: float = 0.8e-6
    ringing_amplitude_v: float = 0.0
    ringing_frequency_hz: float = 18e6
    ringing_decay_s: float = 2.5e-6
    ringing_phase_rad: float = 0.0
    ringing_follows_timing_jitter: bool = False
    timing_jitter_s: float = 0.0
    resolving_power: float = 1000.0
    dropped_trigger_probability: float = 0.0
    saturation_v: float | None = None
    repetition_shift_s_per_khz: float = 2e-9
    random_seed: int | None = 1
    adc_full_scale_counts: int = 32767
    extra_metadata: dict[str, str] = field(default_factory=dict)


class MockDigitizer(DigitizerBase):
    """Stateful mock Spectrum-like digitizer."""

    def __init__(self, profile: MockDigitizerProfile | None = None) -> None:
        self.profile = profile or MockDigitizerProfile()
        self.connected = False
        self.configured = False
        self.armed = False
        self.config: DigitizerConfig | None = None
        self.trigger_index = 0
        self.dropped_triggers = 0
        self._rng = np.random.default_rng(self.profile.random_seed)

    def connect(self) -> None:
        """Connect the mock digitizer."""
        self.connected = True
        LOGGER.info("mock digitizer connected")

    def configure(self, config: DigitizerConfig) -> None:
        """Configure synthetic acquisition shape and timing."""
        if not self.connected:
            raise DigitizerError("digitizer is not connected")
        config.validate()
        self.config = config
        self.configured = True
        LOGGER.info("mock digitizer configured")

    def arm(self) -> None:
        """Arm mock acquisition."""
        if not self.configured:
            raise DigitizerError("digitizer is not configured")
        self.armed = True
        LOGGER.info("mock digitizer armed")

    def read_batch(self) -> np.ndarray:
        """Return a synthetic raw ADC batch."""
        if not self.armed or self.config is None:
            raise DigitizerError("digitizer is not armed")
        config = self.config
        if self.profile.dropped_trigger_probability >= 1.0:
            raise AcquisitionTimeoutError("all mock triggers were dropped")

        traces = np.empty((config.number_of_segments, config.segment_samples), dtype=np.int16)
        for index in range(config.number_of_segments):
            if self._rng.random() < self.profile.dropped_trigger_probability:
                self.dropped_triggers += 1
                traces[index] = 0
                continue
            trace_v = self._make_trace_volts(config)
            traces[index] = self._volts_to_adc(trace_v, config.input_range_v)
            self.trigger_index += 1
        return traces

    def stop(self) -> None:
        """Stop mock acquisition."""
        if not self.armed:
            return
        self.armed = False
        LOGGER.info("mock digitizer stopped")

    def close(self) -> None:
        """Close the mock digitizer."""
        self.stop()
        self.connected = False
        self.configured = False
        LOGGER.info("mock digitizer closed")

    def _make_trace_volts(self, config: DigitizerConfig) -> np.ndarray:
        times = (np.arange(config.segment_samples, dtype=np.float64) - config.pretrigger_samples) / config.sample_rate_hz
        rep_khz = 1.0 / max(config.timeout_s, 1e-12) * 1e-3
        rep_shift = rep_khz * self.profile.repetition_shift_s_per_khz
        trace = np.full(config.segment_samples, self.profile.baseline_offset_v, dtype=np.float64)
        trace += np.linspace(0.0, self.profile.baseline_drift_v, config.segment_samples)
        internal_average_count = config.hardware_averages_per_record if config.record_mode == "hardware_average" else 1
        peak_trace = np.zeros(config.segment_samples, dtype=np.float64)
        jittered_ringing_trace = np.zeros(config.segment_samples, dtype=np.float64)
        for _ in range(internal_average_count):
            jitter = self._rng.normal(0.0, self.profile.timing_jitter_s)
            for peak in self.profile.peaks:
                center = peak.position_s + jitter + rep_shift
                width = _peak_sigma_from_resolving_power(peak.position_s, self.profile.resolving_power)
                peak_trace += self.profile.detector_polarity * peak.amplitude_v * np.exp(-0.5 * ((times - center) / width) ** 2)
            if self.profile.ringing_amplitude_v > 0.0 and self.profile.ringing_follows_timing_jitter:
                jittered_ringing_trace += _ringing_waveform(times, self.profile, jitter)
        trace += peak_trace / internal_average_count
        if self.profile.ringing_amplitude_v > 0.0 and self.profile.ringing_follows_timing_jitter:
            trace += jittered_ringing_trace / internal_average_count
        post_trigger = np.maximum(times, 0.0)
        if self.profile.pickup_amplitude_v > 0.0:
            trace += self.profile.pickup_amplitude_v * np.exp(-post_trigger / self.profile.pickup_decay_s) * (times >= 0.0)
        if self.profile.ringing_amplitude_v > 0.0 and not self.profile.ringing_follows_timing_jitter:
            trace += _ringing_waveform(times, self.profile, 0.0)
        noise_scale = 1.0
        if config.record_mode == "hardware_average":
            noise_scale = 1.0 / np.sqrt(config.hardware_averages_per_record)
        trace += self._rng.normal(0.0, self.profile.noise_rms_v * noise_scale, config.segment_samples)
        if self.profile.saturation_v is not None:
            trace = np.clip(trace, -self.profile.saturation_v, self.profile.saturation_v)
        return trace.astype(np.float32)

    def _volts_to_adc(self, trace_v: np.ndarray, input_range_v: float) -> np.ndarray:
        counts = trace_v / input_range_v * self.profile.adc_full_scale_counts
        counts = np.clip(counts, -self.profile.adc_full_scale_counts, self.profile.adc_full_scale_counts)
        return np.rint(counts).astype(np.int16)


def _peak_sigma_from_resolving_power(tof_s: float, resolving_power: float) -> float:
    """Convert resolving power m/dm into Gaussian sigma in TOF seconds."""
    if resolving_power <= 0:
        raise ValueError("resolving_power must be positive")
    tof_fwhm_s = tof_s / (2.0 * resolving_power)
    return max(tof_fwhm_s / 2.354820045, 1e-12)


def _ringing_waveform(times: np.ndarray, profile: MockDigitizerProfile, timing_shift_s: float) -> np.ndarray:
    shifted_time = times - timing_shift_s
    active = shifted_time >= 0.0
    post_trigger = np.maximum(shifted_time, 0.0)
    return (
        profile.ringing_amplitude_v
        * np.cos(2.0 * np.pi * profile.ringing_frequency_hz * post_trigger + profile.ringing_phase_rad)
        * np.exp(-post_trigger / profile.ringing_decay_s)
        * active
    )
