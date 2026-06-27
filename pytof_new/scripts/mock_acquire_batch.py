"""Acquire a finite batch using mock hardware and save HDF5 output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytof_new.acquisition.controller import AcquisitionController
from pytof_new.config.models import BMEConfig, DigitizerConfig, MockSpectraConfig, ProcessingConfig, RunConfig, StorageConfig
from pytof_new.hardware.mock_delay_generator import MockDelayGenerator
from pytof_new.hardware.mock_digitizer import MockDigitizer, MockDigitizerProfile
from pytof_new.logging_config import configure_logging
from pytof_new.processing.pipeline import process_batch
from pytof_new.storage.hdf5_writer import HDF5RunWriter


def build_config(output_path: Path, averages_per_record: int, timing_jitter_ns: float, resolving_power: float) -> RunConfig:
    """Create the default mock run configuration."""
    digitizer = DigitizerConfig(
        number_of_segments=1,
        hardware_averages_per_record=averages_per_record,
        segment_samples=65536,
        pretrigger_samples=32,
    )
    bme = BMEConfig()
    processing = ProcessingConfig(baseline_start=0, baseline_stop=32, subtract_baseline=True)
    storage = StorageConfig(output_path=output_path, molecule="mock", notes="Mock acquisition")
    mock_spectra = MockSpectraConfig(timing_jitter_s=timing_jitter_ns * 1e-9, resolving_power=resolving_power)
    return RunConfig(digitizer=digitizer, bme=bme, processing=processing, storage=storage, mock_spectra=mock_spectra)


def main() -> int:
    """Run the mock acquisition command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("mock_run.h5"), help="HDF5 output path")
    parser.add_argument("--averages-per-record", type=int, default=100, help="internal mock spectra averaged into one transferred record")
    parser.add_argument("--segments", type=int, default=None, help="deprecated alias for --averages-per-record")
    parser.add_argument("--timing-jitter-ns", type=float, default=0.0, help="RMS timing jitter applied before internal averaging")
    parser.add_argument("--resolving-power", type=float, default=1000.0, help="mock resolving power m/dm")
    args = parser.parse_args()

    configure_logging()
    averages_per_record = args.segments if args.segments is not None else args.averages_per_record
    config = build_config(args.output, averages_per_record, args.timing_jitter_ns, args.resolving_power)
    digitizer = MockDigitizer(
        MockDigitizerProfile(
            random_seed=42,
            timing_jitter_s=config.mock_spectra.timing_jitter_s,
            resolving_power=config.mock_spectra.resolving_power,
        )
    )
    delay = MockDelayGenerator()
    controller = AcquisitionController(digitizer, delay)

    controller.connect_hardware()
    batch = controller.acquire_batch(config)
    processed = process_batch(batch, config.digitizer, config.processing)
    with HDF5RunWriter(args.output, config) as writer:
        writer.append_raw_batch(batch)
        writer.write_processed(processed)
    controller.disconnect_hardware()

    print(
        f"saved {batch.raw_adc.shape[0]} averaged record x {batch.raw_adc.shape[1]} samples "
        f"({averages_per_record} internal spectra averaged) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
