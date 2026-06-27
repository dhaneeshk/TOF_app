"""HDF5 storage for acquisition runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pytof_new.acquisition.models import AcquisitionBatch
from pytof_new.config.models import RunConfig, to_plain_dict
from pytof_new.processing.pipeline import ProcessedBatch
from pytof_new.storage.metadata import utc_now_iso


@dataclass(frozen=True)
class ReferenceSpectrum:
    """Saved blank/reference spectrum for background subtraction."""

    axis: np.ndarray
    trace: np.ndarray
    record_count: int
    run_config_json: str


class HDF5RunWriter:
    """Incremental HDF5 writer for one acquisition run."""

    def __init__(self, output_path: Path, config: RunConfig, run_name: str = "run_0001") -> None:
        self.output_path = Path(output_path)
        self.config = config
        self.run_name = run_name
        self._file: h5py.File | None = None
        self._run: h5py.Group | None = None
        self._raw_dataset: h5py.Dataset | None = None
        self._timestamps: h5py.Dataset | None = None
        self._raw_count = 0

    def __enter__(self) -> "HDF5RunWriter":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the file and create metadata groups."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.output_path, "w")
        self._run = self._file.create_group(self.run_name)
        self._run.create_group("raw")
        self._run.create_group("processed")
        metadata = self._run.create_group("metadata")
        metadata.attrs["acquisition_start"] = utc_now_iso()
        metadata.attrs["run_config_json"] = json.dumps(to_plain_dict(self.config), sort_keys=True)
        metadata.attrs["molecule"] = self.config.storage.molecule
        metadata.attrs["surface"] = self.config.storage.surface
        metadata.attrs["q1"] = self.config.storage.q1
        metadata.attrs["q2"] = self.config.storage.q2
        metadata.attrs["uv"] = self.config.storage.uv
        metadata.attrs["notes"] = self.config.storage.notes

    def append_raw_batch(self, batch: AcquisitionBatch) -> None:
        """Append raw ADC segments and optional timestamps."""
        if self._run is None:
            raise RuntimeError("writer is not open")
        if not self.config.storage.save_raw_segments:
            return
        raw_group = self._run["raw"]
        segments, samples = batch.raw_adc.shape
        if self._raw_dataset is None:
            self._raw_dataset = raw_group.create_dataset(
                "adc_segments",
                shape=(0, samples),
                maxshape=(None, samples),
                chunks=(max(1, min(segments, 128)), samples),
                dtype=batch.raw_adc.dtype,
                compression=self.config.storage.compression,
            )
        self._raw_dataset.resize((self._raw_count + segments, samples))
        self._raw_dataset[self._raw_count : self._raw_count + segments] = batch.raw_adc
        if batch.timestamps is not None:
            if self._timestamps is None:
                self._timestamps = raw_group.create_dataset(
                    "timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(max(1, min(segments, 1024)),),
                    dtype=batch.timestamps.dtype,
                    compression=self.config.storage.compression,
                )
            self._timestamps.resize((self._raw_count + segments,))
            self._timestamps[self._raw_count : self._raw_count + segments] = batch.timestamps
        self._raw_count += segments

    def write_processed(self, processed: ProcessedBatch) -> None:
        """Write processed results for the run."""
        if self._run is None:
            raise RuntimeError("writer is not open")
        if not self.config.storage.save_processed:
            return
        group = self._run["processed"]
        _replace_dataset(group, "average_trace", processed.average_trace)
        _replace_dataset(group, "baseline_corrected_trace", processed.baseline_corrected_segments.mean(axis=0, dtype=np.float32))
        _replace_dataset(group, "tof_axis", processed.tof_axis)
        group.attrs["record_mode"] = processed.record_mode
        group.attrs["hardware_averages_per_record"] = processed.hardware_averages_per_record
        group.attrs["accepted_record_count"] = processed.accepted_count
        group.attrs["rejected_record_count"] = processed.rejected_count
        group.attrs["accepted_shot_count"] = processed.accepted_count
        group.attrs["rejected_shot_count"] = processed.rejected_count
        group.attrs["clipping_rejection_count"] = processed.rejection_summary.clipping_rejection_count
        group.attrs["baseline_noise_rejection_count"] = processed.rejection_summary.baseline_noise_rejection_count
        if processed.mass_axis is not None:
            _replace_dataset(group, "mass_axis", processed.mass_axis)
        if processed.peaks is not None:
            peak_group = group.require_group("detected_peaks")
            _replace_dataset(peak_group, "indices", processed.peaks.indices)
            _replace_dataset(peak_group, "positions", processed.peaks.positions)
            _replace_dataset(peak_group, "heights", processed.peaks.heights)

    def close(self) -> None:
        """Close the HDF5 file after writing final metadata."""
        if self._run is not None:
            self._run["metadata"].attrs["acquisition_end"] = utc_now_iso()
            self._run["metadata"].attrs["record_count"] = self._raw_count
            self._run["metadata"].attrs["trigger_count"] = self._raw_count * self.config.digitizer.hardware_averages_per_record
            self._run["metadata"].attrs["record_mode"] = self.config.digitizer.record_mode
            self._run["metadata"].attrs["hardware_averages_per_record"] = self.config.digitizer.hardware_averages_per_record
        if self._file is not None:
            self._file.close()
        self._file = None
        self._run = None


def _replace_dataset(group: h5py.Group, name: str, data: Any) -> None:
    array = np.asarray(data)
    if name in group:
        dataset = group[name]
        if dataset.shape == array.shape and dataset.dtype == array.dtype:
            dataset[...] = array
            return
        del group[name]
    group.create_dataset(name, data=array)


def save_cumulative_spectrum(output_path: Path, axis: np.ndarray, trace: np.ndarray, record_count: int, config: RunConfig) -> None:
    """Save a manually requested cumulative spectrum to HDF5."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        group = handle.create_group("cumulative_spectrum")
        group.create_dataset("axis", data=axis)
        group.create_dataset("average_trace", data=trace)
        group.attrs["saved_at"] = utc_now_iso()
        group.attrs["record_count"] = record_count
        group.attrs["record_mode"] = config.digitizer.record_mode
        group.attrs["hardware_averages_per_record"] = config.digitizer.hardware_averages_per_record
        group.attrs["run_config_json"] = json.dumps(to_plain_dict(config), sort_keys=True)


def save_reference_spectrum(output_path: Path, axis: np.ndarray, trace: np.ndarray, record_count: int, config: RunConfig) -> None:
    """Save a blank/reference spectrum for later subtraction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        group = handle.create_group("reference")
        group.create_dataset("axis", data=axis)
        group.create_dataset("trace", data=trace)
        group.attrs["created_at"] = utc_now_iso()
        group.attrs["record_count"] = record_count
        group.attrs["record_mode"] = config.digitizer.record_mode
        group.attrs["hardware_averages_per_record"] = config.digitizer.hardware_averages_per_record
        group.attrs["run_config_json"] = json.dumps(to_plain_dict(config), sort_keys=True)


def load_reference_spectrum(input_path: Path) -> ReferenceSpectrum:
    """Load a blank/reference spectrum saved by save_reference_spectrum."""
    with h5py.File(input_path, "r") as handle:
        group = handle["reference"]
        return ReferenceSpectrum(
            axis=group["axis"][...],
            trace=group["trace"][...].astype(np.float32, copy=False),
            record_count=int(group.attrs["record_count"]),
            run_config_json=str(group.attrs["run_config_json"]),
        )
