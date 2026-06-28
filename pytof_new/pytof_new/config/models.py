"""Immutable configuration models for TOF acquisition runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class AcquisitionWorkflow(str, Enum):
    """User-facing acquisition workflow."""

    LIVE_AVERAGED = "live_averaged"
    LIVE_RAW = "live_raw"
    FINITE_SHOT_ANALYSIS = "finite_shot_analysis"


class AcquisitionPriority(str, Enum):
    """Planning preference used by Basic mode."""

    FAST_UPDATES = "fast_updates"
    BALANCED = "balanced"
    HIGHEST_TIME_RESOLUTION = "highest_time_resolution"
    BEST_SIGNAL_TO_NOISE = "best_signal_to_noise"


@dataclass(frozen=True)
class DigitizerConfig:
    """Spectrum digitizer acquisition settings."""

    sample_rate_hz: float = 1.25e9
    input_range_v: float = 0.5
    coupling: str = "dc"
    bandwidth_limit_enabled: bool = False
    trigger_source: str = "external0"
    trigger_level_v: float = 1.5
    trigger_edge: str = "rising"
    trigger_termination_ohm: int = 50
    pretrigger_samples: int = 32
    segment_samples: int = 65536
    number_of_segments: int = 1
    record_mode: Literal["hardware_average", "raw_segments"] = "hardware_average"
    hardware_averages_per_record: int = 100
    timeout_s: float = 5.0
    # Planner / basic-mode fields
    tof_window_us: float = 50.0
    acquisition_workflow: AcquisitionWorkflow = AcquisitionWorkflow.LIVE_AVERAGED
    acquisition_priority: AcquisitionPriority = AcquisitionPriority.BALANCED
    target_update_interval_s: float = 0.5
    total_shots: int = 1000
    advanced_mode: bool = False
    accumulator_mode: Literal["automatic", "32bit", "16bit"] = "automatic"
    fpga_sums_per_batch: int = 1
    raw_shots_per_batch: int = 500
    override_segment_samples: bool = False

    def validate(self) -> None:
        """Validate settings before hardware calls are made."""
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.input_range_v <= 0:
            raise ValueError("input_range_v must be positive")
        if self.coupling.lower() not in {"dc", "ac"}:
            raise ValueError("coupling must be 'dc' or 'ac'")
        if self.trigger_source.lower() not in {"external0", "software", "channel0"}:
            raise ValueError("unsupported trigger_source")
        if self.trigger_edge.lower() not in {"rising", "falling"}:
            raise ValueError("trigger_edge must be 'rising' or 'falling'")
        if self.trigger_termination_ohm not in {50, 1000}:
            raise ValueError("trigger_termination_ohm must be 50 or 1000")
        if self.pretrigger_samples < 0:
            raise ValueError("pretrigger_samples must be non-negative")
        if self.segment_samples <= 0:
            raise ValueError("segment_samples must be positive")
        if self.segment_samples < 64:
            raise ValueError("segment_samples must be at least 64")
        if self.segment_samples % 32 != 0:
            raise ValueError("segment_samples must be a multiple of 32")
        if self.pretrigger_samples >= self.segment_samples:
            raise ValueError("pretrigger_samples must be smaller than segment_samples")
        if self.number_of_segments <= 0:
            raise ValueError("number_of_segments must be positive")
        if self.record_mode not in {"hardware_average", "raw_segments"}:
            raise ValueError("record_mode must be 'hardware_average' or 'raw_segments'")
        if self.hardware_averages_per_record <= 0:
            raise ValueError("hardware_averages_per_record must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.acquisition_workflow not in set(AcquisitionWorkflow):
            raise ValueError("acquisition_workflow is invalid")
        if self.acquisition_priority not in set(AcquisitionPriority):
            raise ValueError("acquisition_priority is invalid")
        if self.target_update_interval_s <= 0:
            raise ValueError("target_update_interval_s must be positive")
        if self.total_shots <= 0:
            raise ValueError("total_shots must be positive")
        if self.accumulator_mode not in {"automatic", "32bit", "16bit"}:
            raise ValueError("accumulator_mode must be 'automatic', '32bit', or '16bit'")
        if self.fpga_sums_per_batch <= 0:
            raise ValueError("fpga_sums_per_batch must be positive")
        if self.raw_shots_per_batch <= 0:
            raise ValueError("raw_shots_per_batch must be positive")


@dataclass(frozen=True)
class BMEConfig:
    """BME delay generator settings."""

    advanced_mode: bool = False
    tof_window_s: float = 50e-6
    extraction_region_fill_time_s: float = 55e-6
    repetition_period_s: float = 105e-6
    digitizer_trigger_delay_s: float = 0.0
    push_trigger_delay_s: float = 0.0
    pull_trigger_delay_s: float = 0.0
    digitizer_trigger_width_s: float = 50e-6
    push_trigger_width_s: float = 50e-6
    pull_trigger_width_s: float = 50e-6
    digitizer_channel: str = "A"
    push_channel: str = "C"
    pull_channel: str = "F"
    enabled_output_roles: tuple[str, ...] = ("digitizer", "push", "pull")
    digitizer_polarity_positive: bool = True
    push_polarity_positive: bool = True
    pull_polarity_positive: bool = False
    trigger_termination_ohm: int = 50
    go_signal: Literal["local_primary", "master_primary"] = "local_primary"

    def validate(self) -> None:
        """Validate delay generator settings."""
        if self.tof_window_s <= 0:
            raise ValueError("tof_window_s must be positive")
        if self.extraction_region_fill_time_s <= 0:
            raise ValueError("extraction_region_fill_time_s must be positive")
        if self.repetition_period_s <= 0:
            raise ValueError("repetition_period_s must be positive")
        if self.repetition_period_s <= self.tof_window_s:
            raise ValueError("repetition_period_s must be greater than TOF window")
        delays = (self.digitizer_trigger_delay_s, self.push_trigger_delay_s, self.pull_trigger_delay_s)
        widths = (self.digitizer_trigger_width_s, self.push_trigger_width_s, self.pull_trigger_width_s)
        if any(delay < 0 for delay in delays):
            raise ValueError("delays must be non-negative")
        if any(delay >= self.tof_window_s for delay in delays):
            raise ValueError("delays must be less than TOF window")
        if any(width <= 0 for width in widths):
            raise ValueError("pulse widths must be positive")
        if any(width >= self.repetition_period_s for width in widths):
            raise ValueError("pulse widths must be less than repetition period")
        if any(delay + width >= self.repetition_period_s for delay, width in zip(delays, widths)):
            raise ValueError("pulse delay plus width must be less than repetition period")
        channels = [self.digitizer_channel.upper(), self.push_channel.upper(), self.pull_channel.upper()]
        valid_channels = {"A", "B", "C", "D", "E", "F"}
        if any(channel not in valid_channels for channel in channels):
            raise ValueError("BME channels must be one of A, B, C, D, E, F")
        if len(set(channels)) != len(channels):
            raise ValueError("BME output channels must be unique")
        valid_roles = {"digitizer", "push", "pull"}
        roles = tuple(role.lower() for role in self.enabled_output_roles)
        if any(role not in valid_roles for role in roles):
            raise ValueError("enabled_output_roles must contain only digitizer, push, or pull")
        if len(set(roles)) != len(roles):
            raise ValueError("enabled_output_roles must be unique")
        if self.trigger_termination_ohm not in {50, 1000}:
            raise ValueError("trigger_termination_ohm must be 50 or 1000")
        if self.go_signal not in {"local_primary", "master_primary"}:
            raise ValueError("go_signal must be 'local_primary' or 'master_primary'")


@dataclass(frozen=True)
class MockSpectraConfig:
    """Mock-only synthetic spectrum settings."""

    ion_peaks_enabled: bool = True
    timing_jitter_s: float = 0.0
    resolving_power: float = 1000.0
    noise_rms_v: float = 0.003
    ringing_enabled: bool = False
    ringing_amplitude_v: float = 0.0
    ringing_frequency_hz: float = 18e6
    ringing_decay_s: float = 2.5e-6
    ringing_phase_rad: float = 0.0
    ringing_follows_timing_jitter: bool = False

    def validate(self) -> None:
        """Validate mock spectrum settings."""
        if self.timing_jitter_s < 0:
            raise ValueError("timing_jitter_s must be non-negative")
        if self.resolving_power <= 0:
            raise ValueError("resolving_power must be positive")
        if self.noise_rms_v < 0:
            raise ValueError("noise_rms_v must be non-negative")
        if self.ringing_amplitude_v < 0:
            raise ValueError("ringing_amplitude_v must be non-negative")
        if self.ringing_enabled and self.ringing_frequency_hz <= 0:
            raise ValueError("ringing_frequency_hz must be positive when ringing is enabled")
        if self.ringing_enabled and self.ringing_decay_s <= 0:
            raise ValueError("ringing_decay_s must be positive when ringing is enabled")


@dataclass(frozen=True)
class ProcessingConfig:
    """Processing settings applied to acquired raw waveforms."""

    detector_polarity: int = 1
    adc_full_scale_counts: int = 32767
    baseline_start: int = 0
    baseline_stop: int = 32
    subtract_baseline: bool = True
    baseline_method: Literal["mean", "median"] = "median"
    rejection_enabled: bool = False
    reject_clipped: bool = True
    clipping_margin_fraction: float = 0.02
    maximum_baseline_rms_v: float | None = None
    low_pass_enabled: bool = False
    low_pass_cutoff_hz: float = 50e6
    high_pass_enabled: bool = False
    high_pass_cutoff_hz: float = 0.1e6
    filter_order: int = 4
    reference_subtraction_enabled: bool = False
    reference_path: Path | None = None
    absolute_signal_enabled: bool = False
    smoothing_enabled: bool = False
    smoothing_window: int = 11
    time_zero_offset_s: float = 0.0
    mass_calibration_enabled: bool = False
    mass_calibration: tuple[float, float, float] | None = None
    peak_finding_enabled: bool = False

    def validate(self) -> None:
        """Validate processing settings."""
        if self.detector_polarity not in {-1, 1}:
            raise ValueError("detector_polarity must be -1 or 1")
        if self.adc_full_scale_counts <= 0:
            raise ValueError("adc_full_scale_counts must be positive")
        if self.baseline_start < 0 or self.baseline_stop <= self.baseline_start:
            raise ValueError("baseline interval is invalid")
        if self.baseline_method not in {"mean", "median"}:
            raise ValueError("baseline_method must be 'mean' or 'median'")
        if not 0.0 <= self.clipping_margin_fraction < 1.0:
            raise ValueError("clipping_margin_fraction must be in [0, 1)")
        if self.maximum_baseline_rms_v is not None and self.maximum_baseline_rms_v < 0:
            raise ValueError("maximum_baseline_rms_v must be non-negative")
        if self.low_pass_cutoff_hz <= 0:
            raise ValueError("low_pass_cutoff_hz must be positive")
        if self.high_pass_cutoff_hz <= 0:
            raise ValueError("high_pass_cutoff_hz must be positive")
        if self.low_pass_enabled and self.high_pass_enabled and self.high_pass_cutoff_hz >= self.low_pass_cutoff_hz:
            raise ValueError("high_pass_cutoff_hz must be lower than low_pass_cutoff_hz")
        if self.filter_order <= 0:
            raise ValueError("filter_order must be positive")
        if self.reference_subtraction_enabled and self.reference_path is None:
            raise ValueError("reference_path is required when reference subtraction is enabled")
        if self.smoothing_window < 3 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be odd and >= 3")


@dataclass(frozen=True)
class StorageConfig:
    """Storage settings for one run."""

    output_path: Path = Path("mock_run.pytof")
    molecule: str = "mock"
    surface: str = ""
    q1: str = ""
    q2: str = ""
    uv: str = ""
    notes: str = ""
    save_raw_segments: bool = True
    save_processed: bool = True
    compression: str | None = "gzip"


@dataclass(frozen=True)
class RunConfig:
    """Complete immutable configuration snapshot for one acquisition run."""

    digitizer: DigitizerConfig = field(default_factory=DigitizerConfig)
    bme: BMEConfig = field(default_factory=BMEConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    mock_spectra: MockSpectraConfig = field(default_factory=MockSpectraConfig)

    def validate(self) -> None:
        """Validate all run settings."""
        self.digitizer.validate()
        self.bme.validate()
        self.processing.validate()
        self.mock_spectra.validate()


def to_plain_dict(value: Any) -> Any:
    """Convert dataclasses and paths to JSON/HDF5-friendly containers."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    return value
