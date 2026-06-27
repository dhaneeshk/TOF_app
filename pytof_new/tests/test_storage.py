import h5py
import numpy as np
import pytest

from pytof_new.acquisition.controller import AcquisitionController
from pytof_new.config.models import DigitizerConfig, ProcessingConfig, RunConfig, StorageConfig
from pytof_new.hardware.mock_delay_generator import MockDelayGenerator
from pytof_new.hardware.mock_digitizer import MockDigitizer
from pytof_new.processing.pipeline import process_batch
from pytof_new.storage.hdf5_writer import HDF5RunWriter, load_reference_spectrum, save_reference_spectrum
from pytof_new.storage.pytof_reader import load_pytof_spectrum
from pytof_new.storage.pytof_writer import save_pytof_spectrum


def test_hdf5_writer_saves_raw_processed_and_metadata(tmp_path) -> None:
    output = tmp_path / "run.h5"
    config = RunConfig(
        digitizer=DigitizerConfig(number_of_segments=4, segment_samples=512, pretrigger_samples=32),
        storage=StorageConfig(output_path=output),
    )
    controller = AcquisitionController(MockDigitizer(), MockDelayGenerator())
    controller.connect_hardware()
    batch = controller.acquire_batch(config)
    processed = process_batch(batch, config.digitizer, config.processing)
    with HDF5RunWriter(output, config) as writer:
        writer.append_raw_batch(batch)
        writer.write_processed(processed)
    controller.disconnect_hardware()

    with h5py.File(output, "r") as handle:
        run = handle["run_0001"]
        assert run["raw/adc_segments"].shape == (4, 512)
        assert run["processed/average_trace"].shape == (512,)
        assert run["processed"].attrs["record_mode"] == "hardware_average"
        assert run["processed"].attrs["accepted_record_count"] == 4
        assert run["processed"].attrs["accepted_shot_count"] == 4
        assert run["processed"].attrs["rejected_shot_count"] == 0
        assert "run_config_json" in run["metadata"].attrs
        assert run["metadata"].attrs["record_count"] == 4
        assert run["metadata"].attrs["trigger_count"] == 4 * config.digitizer.hardware_averages_per_record
        assert run["metadata"].attrs["record_mode"] == "hardware_average"


def test_pytof_writer_saves_header_and_mass_axis(tmp_path) -> None:
    output = tmp_path / "Cr6_Ag111_20260623_120000.pytof"
    config = RunConfig(
        processing=ProcessingConfig(detector_polarity=1, mass_calibration_enabled=True, mass_calibration=(3.5e-6, 0.0129, 6.27)),
        storage=StorageConfig(
            output_path=output,
            molecule="Cr6",
            surface="Ag111",
            q1="380",
            q2="764",
            uv="6.5",
            notes="note 1\nnote 2",
        ),
    )
    axis_us = np.array([1.0, 2.0], dtype=np.float32)
    trace = np.array([10.0, 20.0], dtype=np.float32)
    save_pytof_spectrum(output, axis_us, trace, config, axis_mode="TOF")
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("## pyTOF data saved on ")
    assert lines[1] == "## Cr6_Ag111_"
    assert lines[2] == "## Pos MODE"
    assert lines[3] == "## Calib POS:"
    assert lines[4].startswith("## POS_A:")
    assert lines[7] == "## Q1:380"
    assert lines[8] == "## Q2:764"
    assert lines[9] == "## UV:6.5"
    assert lines[10] == "## note 1"
    assert lines[11] == "## note 2"
    assert lines[12] == "##"
    assert lines[13] == "##"
    assert len(lines[14].split()) == 2
    first_mz = float(lines[14].split()[0])
    assert first_mz == pytest.approx(config.processing.mass_calibration[0] * 1000.0**2 + config.processing.mass_calibration[1] * 1000.0 + config.processing.mass_calibration[2])


def test_pytof_reader_loads_header_label_and_data(tmp_path) -> None:
    output = tmp_path / "saved.pytof"
    output.write_text(
        "\n".join(
            [
                "## pyTOF data saved on 2026-06-24 12:00:00",
                "## Cr6_Ag111_",
                "## Pos MODE",
                "## Calib POS:",
                "## POS_A:1",
                "## POS_B:2",
                "## POS_C:3",
                "## Q1:380",
                "## Q2:764",
                "## UV:6.5",
                "##",
                "##",
                "##",
                "##",
                "10.0  1.5",
                "20.0  2.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spectrum = load_pytof_spectrum(output)
    assert spectrum.label == "Cr6_Ag111"
    np.testing.assert_allclose(spectrum.mass_axis, [10.0, 20.0])
    np.testing.assert_allclose(spectrum.trace, [1.5, 2.5])


def test_reference_spectrum_round_trip(tmp_path) -> None:
    output = tmp_path / "reference.h5"
    config = RunConfig(storage=StorageConfig(output_path=output))
    axis = np.arange(64, dtype=np.float32)
    trace = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    save_reference_spectrum(output, axis, trace, 12, config)
    reference = load_reference_spectrum(output)
    np.testing.assert_allclose(reference.axis, axis)
    np.testing.assert_allclose(reference.trace, trace)
    assert reference.record_count == 12
    assert "digitizer" in reference.run_config_json
