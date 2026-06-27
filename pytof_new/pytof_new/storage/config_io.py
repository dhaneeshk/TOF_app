"""JSON configuration I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pytof_new.config.models import to_plain_dict


def save_config_json(path: Path, config: Any) -> None:
    """Save a dataclass configuration as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_dict(config), indent=2, sort_keys=True), encoding="utf-8")


def load_config_json(path: Path) -> dict[str, Any]:
    """Load a JSON configuration dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))
