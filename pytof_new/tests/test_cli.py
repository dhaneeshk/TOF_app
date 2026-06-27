from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from pytof_new import cli


def test_diagnose_bme_safe_mode_does_not_activate_outputs(monkeypatch, capsys) -> None:
    fake_delay = _FakeBMEDelay()
    _patch_bme_cli(monkeypatch, fake_delay)
    monkeypatch.delenv("PYTOF_RUN_HARDWARE_TESTS", raising=False)

    cli._run_diagnose_bme(
        argparse.Namespace(
            dll=None,
            card_index=0,
            pulse_test=False,
            pulse_count=1,
            repetition_us=111.0,
            tof_window_us=50.0,
            digitizer_channel="A",
            push_channel="C",
            pull_channel="F",
            digitizer_polarity="pos",
            push_polarity="pos",
            pull_polarity="neg",
            settle_ms=0.0,
        )
    )

    output = capsys.readouterr().out
    assert "Safe diagnostics only" in output
    assert fake_delay.calls == ["connect", "read_status", "read_trigger_count", "close"]


def test_diagnose_bme_pulse_test_requires_hardware_gate(monkeypatch, capsys) -> None:
    monkeypatch.delenv("PYTOF_RUN_HARDWARE_TESTS", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli._run_diagnose_bme(
            argparse.Namespace(
                dll=None,
                card_index=0,
                    pulse_test=True,
                    pulse_count=1,
                    repetition_us=111.0,
                    tof_window_us=50.0,
                    digitizer_channel="A",
                    push_channel="C",
                    pull_channel="F",
                    digitizer_polarity="pos",
                    push_polarity="pos",
                    pull_polarity="neg",
                    settle_ms=0.0,
                )
        )

    assert exc.value.code == 1
    assert "BME pulse diagnostics are disabled" in capsys.readouterr().err


def test_diagnose_bme_pulse_test_activates_only_when_gated(monkeypatch, capsys) -> None:
    fake_delay = _FakeBMEDelay()
    fake_api = _patch_bme_cli(monkeypatch, fake_delay)
    monkeypatch.setenv("PYTOF_RUN_HARDWARE_TESTS", "1")
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli._run_diagnose_bme(
        argparse.Namespace(
            dll="DelayGenerator.dll",
            card_index=0,
            pulse_test=True,
            pulse_count=3,
            repetition_us=222.0,
            tof_window_us=60.0,
            digitizer_channel="B",
            push_channel="D",
            pull_channel="E",
            digitizer_polarity="neg",
            push_polarity="pos",
            pull_polarity="neg",
            settle_ms=0.0,
        )
    )

    output = capsys.readouterr().out
    assert "Planned pulse table" in output
    assert "Digitizer: channel B, NEG" in output
    assert fake_api.dll_path == "DelayGenerator.dll"
    assert fake_delay.card_index == 0
    assert fake_delay.config.repetition_period_s == pytest.approx(222e-6)
    assert fake_delay.config.tof_window_s == pytest.approx(60e-6)
    assert fake_delay.config.digitizer_channel == "B"
    assert fake_delay.config.push_channel == "D"
    assert fake_delay.config.pull_channel == "E"
    assert fake_delay.config.digitizer_polarity_positive is False
    assert fake_delay.calls == [
        "connect",
        "read_status",
        "read_trigger_count",
        "configure",
        "arm:3",
        "start",
        "read_status",
        "read_trigger_count",
        "stop",
        "close",
    ]


def test_diagnose_bme_pulse_test_rejects_invalid_timing(monkeypatch, capsys) -> None:
    fake_delay = _FakeBMEDelay()
    _patch_bme_cli(monkeypatch, fake_delay)
    monkeypatch.setenv("PYTOF_RUN_HARDWARE_TESTS", "1")

    with pytest.raises(SystemExit) as exc:
        cli._run_diagnose_bme(
            argparse.Namespace(
                dll=None,
                card_index=0,
                pulse_test=True,
                pulse_count=1,
                repetition_us=40.0,
                tof_window_us=50.0,
                digitizer_channel="A",
                push_channel="C",
                pull_channel="F",
                digitizer_polarity="pos",
                push_polarity="pos",
                pull_polarity="neg",
                settle_ms=0.0,
            )
        )

    assert exc.value.code == 1
    assert "Invalid BME pulse-test configuration" in capsys.readouterr().err
    assert "start" not in fake_delay.calls


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pulse_count", 0, "--pulse-count must be positive"),
        ("repetition_us", 0.0, "--repetition-us must be positive"),
        ("tof_window_us", 0.0, "--tof-window-us must be positive"),
        ("settle_ms", -1.0, "--settle-ms must be non-negative"),
    ],
)
def test_diagnose_bme_validates_arguments(monkeypatch, capsys, field: str, value: object, message: str) -> None:
    monkeypatch.delenv("PYTOF_RUN_HARDWARE_TESTS", raising=False)
    args = argparse.Namespace(
        dll=None,
        card_index=0,
        pulse_test=False,
        pulse_count=1,
        repetition_us=111.0,
        tof_window_us=50.0,
        digitizer_channel="A",
        push_channel="C",
        pull_channel="F",
        digitizer_polarity="pos",
        push_polarity="pos",
        pull_polarity="neg",
        settle_ms=0.0,
    )
    setattr(args, field, value)

    with pytest.raises(SystemExit) as exc:
        cli._run_diagnose_bme(args)

    assert exc.value.code == 1
    assert message in capsys.readouterr().err


def _patch_bme_cli(monkeypatch, fake_delay: "_FakeBMEDelay") -> "_FakeBMEApi":
    from pytof_new.hardware import bme_delay_generator, bme_driver

    fake_api = _FakeBMEApi()

    def make_api(*, dll_path=None):
        fake_api.dll_path = dll_path
        return fake_api

    def make_delay(*, api, card_index=0):
        fake_delay.api = api
        fake_delay.card_index = card_index
        return fake_delay

    monkeypatch.setattr(bme_driver, "BMEDriverApi", make_api)
    monkeypatch.setattr(bme_delay_generator, "BMEDelayGenerator", make_delay)

    return fake_api


class _FakeBMEApi:
    def __init__(self, dll_path=None) -> None:
        self.dll_path = dll_path


class _FakeBMEDelay:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.info = SimpleNamespace(product=8, slot=2, master=True, index=0, detected_count=1, detect_error=0)
        self.config = None
        self.api = None
        self.card_index = None

    def connect(self) -> None:
        self.calls.append("connect")

    def read_status(self) -> int:
        self.calls.append("read_status")
        return 0x44

    def read_trigger_count(self) -> int:
        self.calls.append("read_trigger_count")
        return 3

    def configure(self, config) -> None:
        self.calls.append("configure")
        self.config = config

    def arm(self, count: int) -> None:
        self.calls.append(f"arm:{count}")

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def close(self) -> None:
        self.calls.append("close")
