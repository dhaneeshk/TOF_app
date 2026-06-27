"""Mass calibration helpers."""

from __future__ import annotations

import numpy as np

from pytof_new.processing.conversion import tof_to_mass


def apply_mass_calibration(tof_s: np.ndarray, coefficients: tuple[float, float, float]) -> np.ndarray:
    """Apply a quadratic TOF-to-mass calibration."""
    return tof_to_mass(tof_s, coefficients)


def fit_mass_calibration(tof_ns: np.ndarray, mz: np.ndarray) -> tuple[float, float, float]:
    """Fit m/z = A*t_ns^2 + B*t_ns + C from at least three calibration points."""
    tof_ns = np.asarray(tof_ns, dtype=np.float64)
    mz = np.asarray(mz, dtype=np.float64)
    finite = np.isfinite(tof_ns) & np.isfinite(mz)
    tof_ns = tof_ns[finite]
    mz = mz[finite]
    if tof_ns.size < 3:
        raise ValueError("at least three calibration points are required")
    coefficients = np.polyfit(tof_ns, mz, deg=2)
    return tuple(float(value) for value in coefficients)


def mass_to_tof_ns(mz: np.ndarray, coefficients: tuple[float, float, float]) -> np.ndarray:
    """Invert m/z = A*t_ns^2 + B*t_ns + C and return positive TOF in ns."""
    mz = np.asarray(mz, dtype=np.float64)
    a, b, c = coefficients
    if abs(a) < 1e-30:
        if b == 0:
            raise ValueError("cannot invert mass calibration with zero A and B")
        return (mz - c) / b
    discriminant = (b * b) - (4.0 * a * (c - mz))
    if np.any(discriminant < 0):
        raise ValueError("mass value is outside invertible calibration range")
    sqrt_discriminant = np.sqrt(discriminant)
    roots = np.stack(((-b + sqrt_discriminant) / (2.0 * a), (-b - sqrt_discriminant) / (2.0 * a)))
    positive = np.where(roots >= 0.0, roots, np.nan)
    if np.any(np.all(np.isnan(positive), axis=0)):
        raise ValueError("mass calibration inversion produced no positive TOF root")
    return np.nanmin(positive, axis=0)
