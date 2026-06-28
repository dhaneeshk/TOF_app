from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_spectrum_bme.py"
    spec = importlib.util.spec_from_file_location("probe_spectrum_bme", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_pickup_events_finds_expected_edges() -> None:
    probe = _load_probe_module()
    sample_rate_hz = 10_000_000.0
    pretrigger = 100
    trace = np.zeros(400, dtype=np.float64)
    for _label, time_us in probe.DEFAULT_EVENTS_US:
        index = int(round(pretrigger + time_us * 1e-6 * sample_rate_hz))
        trace[index:] += 20.0

    detections = probe.detect_pickup_events(
        trace,
        sample_rate_hz=sample_rate_hz,
        pretrigger_samples=pretrigger,
        window_us=0.2,
        min_sigma=4.0,
        min_abs=5.0,
    )

    assert all(item["detected"] for item in detections)
    assert [item["label"] for item in detections] == [label for label, _time_us in probe.DEFAULT_EVENTS_US]


def test_detect_pickup_events_warns_on_flat_trace() -> None:
    probe = _load_probe_module()
    detections = probe.detect_pickup_events(
        np.zeros(400, dtype=np.float64),
        sample_rate_hz=10_000_000.0,
        pretrigger_samples=100,
        window_us=0.2,
        min_sigma=4.0,
        min_abs=5.0,
    )

    assert not any(item["detected"] for item in detections)


def test_expected_events_are_derived_from_bme_config() -> None:
    probe = _load_probe_module()
    from pytof_new.config.models import BMEConfig

    config = BMEConfig(
        repetition_period_s=111e-6,
        digitizer_trigger_delay_s=1e-6,
        digitizer_trigger_width_s=3e-6,
        push_trigger_delay_s=5e-6,
        push_trigger_width_s=7e-6,
        pull_trigger_delay_s=10e-6,
        pull_trigger_width_s=11e-6,
    )

    events = probe.expected_events_us(config)
    assert [label for label, _time in events] == ["A start", "A end", "C start", "F start", "C end", "F end"]
    assert [time for _label, time in events] == pytest.approx([1.0, 4.0, 5.0, 10.0, 12.0, 21.0])


def test_detect_pickup_events_for_records_summarizes_detection_fraction() -> None:
    probe = _load_probe_module()
    sample_rate_hz = 10_000_000.0
    pretrigger = 100
    trace = np.zeros(400, dtype=np.float64)
    for _label, time_us in probe.DEFAULT_EVENTS_US:
        index = int(round(pretrigger + time_us * 1e-6 * sample_rate_hz))
        trace[index:] += 20.0
    data = np.vstack([trace, np.zeros_like(trace)])

    summary = probe.detect_pickup_events_for_records(
        data,
        sample_rate_hz=sample_rate_hz,
        pretrigger_samples=pretrigger,
        window_us=0.2,
        min_sigma=4.0,
        min_abs=5.0,
    )

    assert all(item["record_count"] == 2 for item in summary)
    assert all(item["detected_count"] == 1 for item in summary)
    assert all(item["detection_fraction"] == 0.5 for item in summary)
