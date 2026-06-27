"""Reference spectrum subtraction helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_reference_trace(path: Path) -> np.ndarray:
    """Load a saved reference trace from an HDF5 reference file."""
    with h5py.File(path, "r") as handle:
        return handle["reference/trace"][...].astype(np.float32, copy=False)


def subtract_reference(data: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Subtract a reference trace from every record in data."""
    if data.shape[-1] != reference.shape[-1]:
        raise ValueError("reference trace length does not match acquired record length")
    return data - reference
