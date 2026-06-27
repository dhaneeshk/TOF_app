"""Text .pytof cumulative spectrum export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pytof_new.config.models import RunConfig
from pytof_new.processing.conversion import tof_to_mass
from pytof_new.storage.metadata import utc_now_iso


def save_pytof_spectrum(output_path: Path, axis: np.ndarray, trace: np.ndarray, config: RunConfig, axis_mode: str = "TOF") -> None:
    """Save a cumulative spectrum in the legacy text .pytof format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mass_axis = _mass_axis(axis, config, axis_mode)
    polarity = "POS" if config.processing.detector_polarity == 1 else "NEG"
    a, b, c = config.processing.mass_calibration or (0.0, 0.0, 0.0)
    lines = [
        f"## pyTOF data saved on {utc_now_iso().replace('T', ' ').replace('Z', '')}",
        f"## {_sample_label(config.storage.molecule, config.storage.surface)}",
        f"## {'Pos' if polarity == 'POS' else 'Neg'} MODE",
        f"## Calib {polarity}:",
        f"## {polarity}_A:{a}",
        f"## {polarity}_B:{b}",
        f"## {polarity}_C:{c}",
        f"## Q1:{config.storage.q1}",
        f"## Q2:{config.storage.q2}",
        f"## UV:{config.storage.uv}",
    ]
    notes = config.storage.notes.splitlines()[:4]
    lines.extend(f"## {note}" if note else "##" for note in notes)
    lines.extend("##" for _ in range(4 - len(notes)))
    lines.extend(_bme_header_lines(config))
    lines.extend(f"{float(x)}  {float(y)}" for x, y in zip(mass_axis, trace, strict=True))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mass_axis(axis: np.ndarray, config: RunConfig, axis_mode: str) -> np.ndarray:
    if config.processing.mass_calibration is None:
        raise ValueError("mass calibration constants are required for .pytof export")
    if axis_mode == "Mass":
        return axis
    return tof_to_mass(axis * 1e-6, config.processing.mass_calibration)


def _sample_label(molecule: str, surface: str) -> str:
    molecule = molecule.strip() or "pytof"
    surface = surface.strip()
    if surface:
        return f"{molecule}_{surface}_"
    return f"{molecule}_"


def _bme_header_lines(config: RunConfig) -> list[str]:
    bme = config.bme
    return [
        f"## BME_MODE:{'ADVANCED' if bme.advanced_mode else 'BASIC'}",
        f"## BME_TOF_WINDOW_US:{bme.tof_window_s * 1e6:.9g}",
        f"## BME_EXTRACTION_FILL_US:{bme.extraction_region_fill_time_s * 1e6:.9g}",
        f"## BME_REPETITION_US:{bme.repetition_period_s * 1e6:.9g}",
        f"## BME_CHANNELS:DIG={bme.digitizer_channel},PUSH={bme.push_channel},PULL={bme.pull_channel}",
        f"## BME_POLARITIES:DIG={'POS' if bme.digitizer_polarity_positive else 'NEG'},PUSH={'POS' if bme.push_polarity_positive else 'NEG'},PULL={'POS' if bme.pull_polarity_positive else 'NEG'}",
        f"## BME_DELAYS_US:DIG={bme.digitizer_trigger_delay_s * 1e6:.9g},PUSH={bme.push_trigger_delay_s * 1e6:.9g},PULL={bme.pull_trigger_delay_s * 1e6:.9g}",
        f"## BME_WIDTHS_US:DIG={bme.digitizer_trigger_width_s * 1e6:.9g},PUSH={bme.push_trigger_width_s * 1e6:.9g},PULL={bme.pull_trigger_width_s * 1e6:.9g}",
        f"## BME_TERMINATION_OHM:{bme.trigger_termination_ohm}",
    ]
