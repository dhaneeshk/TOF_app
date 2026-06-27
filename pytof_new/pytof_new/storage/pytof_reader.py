"""Read legacy text .pytof spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PyTOFSpectrum:
    """A loaded .pytof spectrum stored as m/z and intensity arrays."""

    path: Path
    label: str
    mass_axis: np.ndarray
    trace: np.ndarray
    header: tuple[str, ...]


def load_pytof_spectrum(path: Path) -> PyTOFSpectrum:
    """Load m/z and intensity columns from a .pytof text spectrum."""
    path = Path(path)
    header: list[str] = []
    mass_values: list[float] = []
    trace_values: list[float] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("##"):
            header.append(line)
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid .pytof data line {line_number}: {raw_line}")
        try:
            mass_values.append(float(parts[0]))
            trace_values.append(float(parts[1]))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric .pytof data line {line_number}: {raw_line}") from exc
    if not mass_values:
        raise ValueError(f"No spectrum data found in {path}")
    return PyTOFSpectrum(
        path=path,
        label=_label_from_header(header, path),
        mass_axis=np.asarray(mass_values, dtype=np.float64),
        trace=np.asarray(trace_values, dtype=np.float32),
        header=tuple(header),
    )


def _label_from_header(header: list[str], path: Path) -> str:
    if len(header) >= 2:
        label = header[1].removeprefix("##").strip().strip("_")
        if label:
            return label
    return path.stem
