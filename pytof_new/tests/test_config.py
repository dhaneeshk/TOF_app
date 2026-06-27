from pathlib import Path

import pytest

from pytof_new.config.models import BMEConfig, DigitizerConfig, MockSpectraConfig, RunConfig, StorageConfig, to_plain_dict
from pytof_new.config.legacy_ini import PyTOFIniSettings, load_pytof_ini, save_pytof_ini


def test_default_run_config_validates() -> None:
    config = RunConfig(storage=StorageConfig(output_path=Path("out.h5")))
    config.validate()
    assert config.digitizer.record_mode == "hardware_average"
    assert config.digitizer.number_of_segments == 1
    assert config.digitizer.hardware_averages_per_record == 100


def test_digitizer_rejects_bad_segment_shape_settings() -> None:
    config = DigitizerConfig(pretrigger_samples=100, segment_samples=96)
    with pytest.raises(ValueError, match="pretrigger"):
        config.validate()


def test_digitizer_rejects_unaligned_segment_size() -> None:
    config = DigitizerConfig(segment_samples=100)
    with pytest.raises(ValueError, match="multiple of 32"):
        config.validate()


def test_digitizer_rejects_too_small_segment_size() -> None:
    config = DigitizerConfig(segment_samples=32)
    with pytest.raises(ValueError, match="at least 64"):
        config.validate()


def test_bme_config_defaults_match_basic_push_pull_timing() -> None:
    config = BMEConfig()
    config.validate()
    assert config.extraction_region_fill_time_s == pytest.approx(55e-6)
    assert config.repetition_period_s == pytest.approx(105e-6)
    assert config.digitizer_channel == "A"
    assert config.push_channel == "C"
    assert config.pull_channel == "F"
    assert config.digitizer_polarity_positive is True
    assert config.push_polarity_positive is True
    assert config.pull_polarity_positive is False


def test_bme_config_rejects_invalid_push_pull_timing_and_channels() -> None:
    with pytest.raises(ValueError, match="unique"):
        BMEConfig(digitizer_channel="A", push_channel="A").validate()
    with pytest.raises(ValueError, match="less than TOF"):
        BMEConfig(push_trigger_delay_s=50e-6).validate()
    with pytest.raises(ValueError, match="delay plus width"):
        BMEConfig(push_trigger_delay_s=49e-6, push_trigger_width_s=56e-6).validate()


def test_processing_rejects_invalid_filter_settings() -> None:
    from pytof_new.config.models import ProcessingConfig

    with pytest.raises(ValueError, match="low_pass_cutoff_hz"):
        ProcessingConfig(low_pass_cutoff_hz=0.0).validate()
    with pytest.raises(ValueError, match="high_pass_cutoff_hz"):
        ProcessingConfig(high_pass_cutoff_hz=0.0).validate()
    with pytest.raises(ValueError, match="lower"):
        ProcessingConfig(low_pass_enabled=True, low_pass_cutoff_hz=1e6, high_pass_enabled=True, high_pass_cutoff_hz=2e6).validate()
    with pytest.raises(ValueError, match="reference_path"):
        ProcessingConfig(reference_subtraction_enabled=True).validate()


def test_to_plain_dict_converts_paths() -> None:
    data = to_plain_dict(StorageConfig(output_path=Path("abc.h5")))
    assert data["output_path"] == "abc.h5"


def test_mock_spectra_rejects_invalid_ringing_settings() -> None:
    with pytest.raises(ValueError, match="noise"):
        MockSpectraConfig(noise_rms_v=-0.001).validate()
    with pytest.raises(ValueError, match="amplitude"):
        MockSpectraConfig(ringing_amplitude_v=-1.0).validate()
    with pytest.raises(ValueError, match="frequency"):
        MockSpectraConfig(ringing_enabled=True, ringing_frequency_hz=0.0).validate()
    with pytest.raises(ValueError, match="decay"):
        MockSpectraConfig(ringing_enabled=True, ringing_decay_s=0.0).validate()


def test_pytof_ini_round_trip(tmp_path) -> None:
    path = tmp_path / "PyTOF.ini"
    settings = PyTOFIniSettings(
        save_dir=Path(r"Z:\DATA\pyTOF"),
        n_average=3000,
        positive_calibration=(1.0, 2.0, 3.0),
        negative_calibration=(4.0, 5.0, 6.0),
    )
    save_pytof_ini(path, settings)
    loaded = load_pytof_ini(path)
    assert loaded == settings
