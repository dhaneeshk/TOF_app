#!/usr/bin/env python3
"""Safe BME SG08p hardware probe.

Run on the control machine that has ``DelayGenerator.dll`` and the BME card.

Safe commands do not activate outputs. Physical output generation requires both
``--pulse-test`` and ``PYTOF_RUN_HARDWARE_TESTS=1``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytof_new.config.models import BMEConfig
from pytof_new.hardware.bme_delay_generator import BMEDelayGenerator
from pytof_new.hardware.bme_driver import BMEDriverApi


CHANNELS = ("A", "B", "C", "D", "E", "F")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe BME SG08p delay generator safely")
    parser.add_argument("--dll", type=Path, default=None, help="Path to DelayGenerator.dll")
    parser.add_argument("--card-index", type=int, default=0, help="BME card index, default 0")
    parser.add_argument("--info", action="store_true", help="Connect, report identity/status/counter, then close")
    parser.add_argument("--configure", action="store_true", help="Configure A/C/F outputs while inactive, then close")
    parser.add_argument("--arm", action="store_true", help="Configure and arm finite trigger count without activation")
    parser.add_argument("--pulse-test", action="store_true", help="Activate outputs; requires PYTOF_RUN_HARDWARE_TESTS=1")
    parser.add_argument("--pulse-count", type=int, default=1, help="Finite trigger count, default 1")
    parser.add_argument("--tof-window-us", type=float, default=50.0, help="Pulse width and TOF window, default 50 us")
    parser.add_argument("--fill-us", type=float, default=55.0, help="Extraction region fill time, default 55 us")
    parser.add_argument("--repetition-us", type=float, default=None, help="Override repetition period in us")
    parser.add_argument("--digitizer-channel", choices=CHANNELS, default="A")
    parser.add_argument("--push-channel", choices=CHANNELS, default="C")
    parser.add_argument("--pull-channel", choices=CHANNELS, default="F")
    parser.add_argument("--digitizer-width-us", type=float, default=None, help="Default follows --tof-window-us")
    parser.add_argument("--push-width-us", type=float, default=None, help="Default follows --tof-window-us")
    parser.add_argument("--pull-width-us", type=float, default=None, help="Default follows --tof-window-us")
    parser.add_argument("--digitizer-delay-us", type=float, default=0.0)
    parser.add_argument("--push-delay-us", type=float, default=0.0)
    parser.add_argument("--pull-delay-us", type=float, default=0.0)
    parser.add_argument("--digitizer-polarity", choices=("pos", "neg"), default="pos")
    parser.add_argument("--push-polarity", choices=("pos", "neg"), default="pos")
    parser.add_argument("--pull-polarity", choices=("pos", "neg"), default="neg")
    parser.add_argument("--enabled-channels", default="A", help="Comma-separated output channels to enable; default A for first safe pulse test")
    parser.add_argument("--termination", type=int, choices=(50, 1000), default=50, help="BME trigger/output termination setting passed to the driver")
    parser.add_argument("--go-signal", choices=("local-primary", "master-primary"), default="local-primary", help="Channel GoSignal source")
    parser.add_argument("--settle-ms", type=float, default=100.0, help="Extra wait after expected pulse duration before final readback")
    parser.add_argument("--timeout-s", type=float, default=None, help="Pulse-test timeout; default derives from count and repetition")
    args = parser.parse_args()

    if not any((args.info, args.configure, args.arm, args.pulse_test)):
        args.info = True
    if args.pulse_test and not _hardware_tests_enabled():
        print("ERROR: --pulse-test requires PYTOF_RUN_HARDWARE_TESTS=1", file=sys.stderr)
        return 1
    if args.pulse_count <= 0:
        print("ERROR: --pulse-count must be positive", file=sys.stderr)
        return 1
    if args.settle_ms < 0:
        print("ERROR: --settle-ms must be non-negative", file=sys.stderr)
        return 1

    try:
        config = _config_from_args(args)
        config.validate()
    except ValueError as exc:
        print(f"ERROR: invalid BME timing: {exc}", file=sys.stderr)
        return 1

    delay = BMEDelayGenerator(api=BMEDriverApi(dll_path=args.dll), card_index=args.card_index)
    try:
        print("Connecting to BME...")
        delay.connect()
        _print_identity(delay)
        print(f"Initial status: {delay.read_status()}")
        print(f"Initial trigger counter: {delay.read_trigger_count()}")

        if args.configure or args.arm or args.pulse_test:
            print("\nConfiguring BME while inactive...")
            _print_pulse_table(config, args.pulse_count)
            delay.configure(config)
            print(f"Status after configure: {delay.read_status()}")
            print(f"Trigger counter after configure: {delay.read_trigger_count()}")

        if args.arm or args.pulse_test:
            print(f"\nArming BME for {args.pulse_count} accepted trigger events (no activation yet)...")
            delay.arm(args.pulse_count)
            print(f"Status after arm: {delay.read_status()}")
            print(f"Trigger counter after arm: {delay.read_trigger_count()}")

        if args.pulse_test:
            print("\nActivating BME outputs now")
            delay.start()
            reached = _wait_for_trigger_count(delay, args.pulse_count, config.repetition_period_s, args.settle_ms / 1000.0, args.timeout_s)
            print(f"Status after activation: {delay.read_status()}")
            actual_count = delay.read_trigger_count()
            print(f"Trigger counter after activation: {actual_count}")
            delay.stop()
            print("BME outputs deactivated")
            if not reached or actual_count != args.pulse_count:
                print(f"ERROR: BME trigger counter mismatch: expected {args.pulse_count}, got {actual_count}", file=sys.stderr)
                return 1
        else:
            print("\nSafe probe complete. Outputs were not activated.")
        return 0
    finally:
        print("Closing BME...")
        try:
            delay.emergency_stop()
        finally:
            delay.close()


def _hardware_tests_enabled() -> bool:
    return os.environ.get("PYTOF_RUN_HARDWARE_TESTS", "").strip().lower() in {"1", "true", "yes"}


def _config_from_args(args: argparse.Namespace) -> BMEConfig:
    tof_window_us = float(args.tof_window_us)
    repetition_us = float(args.repetition_us) if args.repetition_us is not None else tof_window_us + float(args.fill_us)
    enabled_roles = _enabled_roles_from_channels(args.enabled_channels, args)
    return BMEConfig(
        advanced_mode=True,
        tof_window_s=tof_window_us * 1e-6,
        extraction_region_fill_time_s=(repetition_us - tof_window_us) * 1e-6,
        repetition_period_s=repetition_us * 1e-6,
        digitizer_trigger_delay_s=args.digitizer_delay_us * 1e-6,
        push_trigger_delay_s=args.push_delay_us * 1e-6,
        pull_trigger_delay_s=args.pull_delay_us * 1e-6,
        digitizer_trigger_width_s=(args.digitizer_width_us if args.digitizer_width_us is not None else tof_window_us) * 1e-6,
        push_trigger_width_s=(args.push_width_us if args.push_width_us is not None else tof_window_us) * 1e-6,
        pull_trigger_width_s=(args.pull_width_us if args.pull_width_us is not None else tof_window_us) * 1e-6,
        digitizer_channel=args.digitizer_channel,
        push_channel=args.push_channel,
        pull_channel=args.pull_channel,
        enabled_output_roles=enabled_roles,
        digitizer_polarity_positive=args.digitizer_polarity == "pos",
        push_polarity_positive=args.push_polarity == "pos",
        pull_polarity_positive=args.pull_polarity == "pos",
        trigger_termination_ohm=args.termination,
        go_signal=args.go_signal.replace("-", "_"),
    )


def _enabled_roles_from_channels(value: str, args: argparse.Namespace) -> tuple[str, ...]:
    requested = {part.strip().upper() for part in value.split(",") if part.strip()}
    if not requested:
        raise ValueError("--enabled-channels must name at least one channel")
    channel_to_role = {
        args.digitizer_channel.upper(): "digitizer",
        args.push_channel.upper(): "push",
        args.pull_channel.upper(): "pull",
    }
    unknown = sorted(channel for channel in requested if channel not in channel_to_role)
    if unknown:
        raise ValueError(f"enabled channel(s) {', '.join(unknown)} are not configured as digitizer/PUSH/PULL outputs")
    return tuple(channel_to_role[channel] for channel in sorted(requested))


def _wait_for_trigger_count(delay: BMEDelayGenerator, expected: int, repetition_s: float, settle_s: float, timeout_s: float | None) -> bool:
    expected_duration_s = max(0.0, expected * repetition_s)
    deadline = time.monotonic() + (timeout_s if timeout_s is not None else max(1.0, expected_duration_s + settle_s + 0.5))
    while time.monotonic() < deadline:
        if delay.read_trigger_count() >= expected:
            time.sleep(settle_s)
            return True
        time.sleep(min(0.05, max(0.001, repetition_s)))
    return delay.read_trigger_count() >= expected


def _print_identity(delay: BMEDelayGenerator) -> None:
    info = delay.info
    if info is None:
        print("WARNING: connected without identity information")
        return
    print("BME identity:")
    print(f"  Product: {info.product}")
    print(f"  Slot: {info.slot}")
    print(f"  Master: {info.master}")
    print(f"  Index: {info.index}")
    print(f"  Detected cards: {info.detected_count}")
    print(f"  Detect error: {info.detect_error}")


def _print_pulse_table(config: BMEConfig, pulse_count: int) -> None:
    print("Planned BME pulses:")
    print(f"  Count: {pulse_count}")
    print(f"  Repetition: {config.repetition_period_s * 1e6:.9g} us")
    print(f"  Enabled roles: {', '.join(config.enabled_output_roles)}")
    print(f"  GoSignal: {config.go_signal}")
    rows = (
        ("Digitizer", config.digitizer_channel, config.digitizer_polarity_positive, config.digitizer_trigger_delay_s, config.digitizer_trigger_width_s),
        ("PUSH", config.push_channel, config.push_polarity_positive, config.push_trigger_delay_s, config.push_trigger_width_s),
        ("PULL", config.pull_channel, config.pull_polarity_positive, config.pull_trigger_delay_s, config.pull_trigger_width_s),
    )
    for label, channel, positive, delay_s, width_s in rows:
        polarity = "POS" if positive else "NEG"
        print(f"  {label}: channel {channel}, {polarity}, delay {delay_s * 1e6:.9g} us, width {width_s * 1e6:.9g} us")


if __name__ == "__main__":
    raise SystemExit(main())
