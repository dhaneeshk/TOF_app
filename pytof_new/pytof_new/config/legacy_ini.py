"""Compatibility helpers for the legacy PyTOF.ini format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pytof_new.storage.metadata import utc_now_iso


@dataclass(frozen=True)
class PyTOFIniSettings:
    """Settings stored in the legacy PyTOF.ini file."""

    save_dir: Path
    n_average: int
    positive_calibration: tuple[float, float, float]
    negative_calibration: tuple[float, float, float]


def load_pytof_ini(path: Path) -> PyTOFIniSettings:
    """Load legacy PyTOF.ini key/value settings."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    required = ["SaveDir", "N_Average", "POS_A", "POS_B", "POS_C", "NEG_A", "NEG_B", "NEG_C"]
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"PyTOF.ini is missing required keys: {', '.join(missing)}")
    return PyTOFIniSettings(
        save_dir=Path(values["SaveDir"]),
        n_average=int(values["N_Average"]),
        positive_calibration=(float(values["POS_A"]), float(values["POS_B"]), float(values["POS_C"])),
        negative_calibration=(float(values["NEG_A"]), float(values["NEG_B"]), float(values["NEG_C"])),
    )


def save_pytof_ini(path: Path, settings: PyTOFIniSettings) -> None:
    """Write settings using the legacy PyTOF.ini key names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos_a, pos_b, pos_c = settings.positive_calibration
    neg_a, neg_b, neg_c = settings.negative_calibration
    timestamp = utc_now_iso().replace("T", " ").replace("Z", "")
    text = (
        f"## pyTOF.ini file saved by software on {timestamp}\n\n"
        f"SaveDir = {settings.save_dir}\n"
        f"N_Average = {settings.n_average}\n"
        f"POS_A = {pos_a}\n"
        f"POS_B = {pos_b}\n"
        f"POS_C = {pos_c}\n"
        f"NEG_A = {neg_a}\n"
        f"NEG_B = {neg_b}\n"
        f"NEG_C = {neg_c}\n\n"
        "## End Automated pyTOF.ini writer\n"
    )
    path.write_text(text, encoding="utf-8")
