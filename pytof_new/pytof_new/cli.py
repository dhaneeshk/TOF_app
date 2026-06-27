"""CLI entry points for pytof_new.

Hardware diagnostics are gated by ``PYTOF_RUN_HARDWARE_TESTS=1`` so they
never run by accident on a system without a real Spectrum card.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from pytof_new.config.models import BMEConfig


def _hardware_tests_enabled() -> bool:
    return os.environ.get("PYTOF_RUN_HARDWARE_TESTS", "").strip() in ("1", "true", "yes")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pytof-new", description="TOF-MS acquisition application")
    sub = parser.add_subparsers(dest="command")

    diag = sub.add_parser("diagnose", help="Run Spectrum hardware diagnostics (requires PYTOF_RUN_HARDWARE_TESTS=1)")
    diag.add_argument("--device", default="/dev/spcm0", help="Spectrum device path (default /dev/spcm0)")
    diag.add_argument("--segments", type=int, default=1024, help="Segment samples (default 1024)")
    diag.add_argument("--pretrigger", type=int, default=32, help="Pretrigger samples (default 32)")
    diag.add_argument("--mode", choices=["raw_multi", "average_32bit"], default="raw_multi", help="Acquisition mode (default raw_multi)")

    bme_diag = sub.add_parser("diagnose-bme", help="Run safe BME diagnostics; pulse tests require --pulse-test and PYTOF_RUN_HARDWARE_TESTS=1")
    bme_diag.add_argument("--dll", type=Path, default=None, help="Path to DelayGenerator.dll (default: search standard locations)")
    bme_diag.add_argument("--card-index", type=int, default=0, help="BME card index (default 0)")
    bme_diag.add_argument("--pulse-test", action="store_true", help="Activate BME outputs for a finite pulse test; requires PYTOF_RUN_HARDWARE_TESTS=1")
    bme_diag.add_argument("--pulse-count", type=int, default=1, help="Pulse-test trigger count (default 1)")
    bme_diag.add_argument("--repetition-us", type=float, default=111.0, help="Pulse-test repetition period in microseconds (default 111)")
    bme_diag.add_argument("--tof-window-us", type=float, default=50.0, help="Pulse-test TOF window and default output width in microseconds (default 50)")
    bme_diag.add_argument("--digitizer-channel", choices=["A", "B", "C", "D", "E", "F"], default="A", help="BME digitizer trigger channel (default A)")
    bme_diag.add_argument("--push-channel", choices=["A", "B", "C", "D", "E", "F"], default="C", help="BME PUSH trigger channel (default C)")
    bme_diag.add_argument("--pull-channel", choices=["A", "B", "C", "D", "E", "F"], default="F", help="BME PULL trigger channel (default F)")
    bme_diag.add_argument("--digitizer-polarity", choices=["pos", "neg"], default="pos", help="Digitizer pulse polarity (default pos)")
    bme_diag.add_argument("--push-polarity", choices=["pos", "neg"], default="pos", help="PUSH pulse polarity (default pos)")
    bme_diag.add_argument("--pull-polarity", choices=["pos", "neg"], default="neg", help="PULL pulse polarity (default neg)")
    bme_diag.add_argument("--settle-ms", type=float, default=100.0, help="Delay after activation before readback in milliseconds (default 100)")

    args = parser.parse_args()
    if args.command == "diagnose":
        _run_diagnose(args)
    elif args.command == "diagnose-bme":
        _run_diagnose_bme(args)
    else:
        parser.print_help()


def _run_diagnose(args: argparse.Namespace) -> None:
    if not _hardware_tests_enabled():
        print("Hardware diagnostics are disabled. Set PYTOF_RUN_HARDWARE_TESTS=1 to enable.", file=sys.stderr)
        sys.exit(1)

    from pytof_new.hardware.spectrum_digitizer import SpectrumDigitizer
    from pytof_new.hardware.spectrum_driver import SpectrumDriverApi
    from pytof_new.hardware.spectrum_models import (
        SpectrumAcquisitionMode,
        SpectrumAcquisitionRequest,
        SpectrumTriggerSource,
    )

    mode_map = {"raw_multi": SpectrumAcquisitionMode.RAW_MULTI, "average_32bit": SpectrumAcquisitionMode.AVERAGE_32BIT}

    digitizer = SpectrumDigitizer(device=args.device)
    print(f"Connecting to {args.device}...")
    digitizer.connect()
    info = digitizer.hardware_info
    print(f"  Card serial: {info.serial_number}")
    print(f"  Max ADC value: {info.max_adc_value}")
    print(f"  16-bit average supported: {info.average_16bit_supported}")
    print(f"  Metadata: {info.metadata}")

    request = SpectrumAcquisitionRequest(
        mode=mode_map[args.mode],
        sample_rate_hz=1.25e9,
        segment_samples=args.segments,
        pretrigger_samples=args.pretrigger,
        number_of_segments=1,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
        input_range_v=0.5,
    )
    print(f"\nConfiguring {args.mode} with {args.segments} samples...")
    plan = digitizer.configure_request(request)
    print(f"  Output shape: {plan.output_shape}")
    print(f"  Transfer bytes: {plan.transfer_bytes}")
    print(f"  FPGA sum: {plan.is_fpga_sum}")
    print(f"  Metadata: {plan.metadata}")

    print("\nAcquiring one batch (software trigger, WAITREADY)...")
    result = digitizer.acquire_configured()
    print(f"  Result data shape: {result.data.shape}")
    print(f"  Result dtype: {result.data.dtype}")
    print(f"  Data range: [{result.data.min()}, {result.data.max()}]")

    print("\nDisconnecting...")
    digitizer.close()
    print("Done.")


def _run_diagnose_bme(args: argparse.Namespace) -> None:
    """Run BME diagnostics without activating outputs unless explicitly gated."""
    if args.pulse_test and not _hardware_tests_enabled():
        print("BME pulse diagnostics are disabled. Set PYTOF_RUN_HARDWARE_TESTS=1 and pass --pulse-test to enable outputs.", file=sys.stderr)
        sys.exit(1)
    if args.pulse_count <= 0:
        print("--pulse-count must be positive", file=sys.stderr)
        sys.exit(1)
    if args.repetition_us <= 0:
        print("--repetition-us must be positive", file=sys.stderr)
        sys.exit(1)
    if args.tof_window_us <= 0:
        print("--tof-window-us must be positive", file=sys.stderr)
        sys.exit(1)
    if args.settle_ms < 0:
        print("--settle-ms must be non-negative", file=sys.stderr)
        sys.exit(1)

    from pytof_new.hardware.bme_delay_generator import BMEDelayGenerator
    from pytof_new.hardware.bme_driver import BMEDriverApi

    api = BMEDriverApi(dll_path=args.dll)
    delay = BMEDelayGenerator(api=api, card_index=args.card_index)
    print("Connecting to BME delay generator...")
    try:
        delay.connect()
        info = delay.info
        if info is None:
            raise RuntimeError("BME connected without identity information")
        print(f"  Product: {info.product}")
        print(f"  Slot: {info.slot}")
        print(f"  Master: {info.master}")
        print(f"  Index: {info.index}")
        print(f"  Detected cards: {info.detected_count}")
        print(f"  Detect error: {info.detect_error}")
        print(f"  Status: {delay.read_status()}")
        print(f"  Trigger counter: {delay.read_trigger_count()}")

        if args.pulse_test:
            repetition_s = args.repetition_us * 1e-6
            tof_window_s = args.tof_window_us * 1e-6
            config = BMEConfig(
                advanced_mode=True,
                tof_window_s=tof_window_s,
                extraction_region_fill_time_s=repetition_s - tof_window_s,
                repetition_period_s=repetition_s,
                digitizer_trigger_width_s=tof_window_s,
                push_trigger_width_s=tof_window_s,
                pull_trigger_width_s=tof_window_s,
                digitizer_channel=args.digitizer_channel,
                push_channel=args.push_channel,
                pull_channel=args.pull_channel,
                digitizer_polarity_positive=args.digitizer_polarity == "pos",
                push_polarity_positive=args.push_polarity == "pos",
                pull_polarity_positive=args.pull_polarity == "pos",
            )
            try:
                config.validate()
            except ValueError as exc:
                print(f"Invalid BME pulse-test configuration: {exc}", file=sys.stderr)
                sys.exit(1)
            print("\nConfiguring finite BME pulse test...")
            _print_bme_pulse_table(config, args.pulse_count)
            delay.configure(config)
            delay.arm(args.pulse_count)
            print(f"  Armed trigger count: {args.pulse_count}")
            print("  Activating BME outputs now")
            delay.start()
            time.sleep(args.settle_ms / 1000.0)
            print(f"  Status after activation: {delay.read_status()}")
            print(f"  Trigger counter after activation: {delay.read_trigger_count()}")
            delay.stop()
            print("  BME outputs deactivated")
        else:
            print("\nSafe diagnostics only. Outputs were not activated. Pass --pulse-test with PYTOF_RUN_HARDWARE_TESTS=1 for a finite output test.")
    finally:
        print("\nDisconnecting BME...")
        delay.close()
        print("Done.")


def _print_bme_pulse_table(config: BMEConfig, pulse_count: int) -> None:
    rows = (
        ("Digitizer", config.digitizer_channel, config.digitizer_polarity_positive, config.digitizer_trigger_delay_s, config.digitizer_trigger_width_s),
        ("PUSH", config.push_channel, config.push_polarity_positive, config.push_trigger_delay_s, config.push_trigger_width_s),
        ("PULL", config.pull_channel, config.pull_polarity_positive, config.pull_trigger_delay_s, config.pull_trigger_width_s),
    )
    print("  Planned pulse table:")
    print(f"    Count: {pulse_count}")
    print(f"    Repetition: {config.repetition_period_s * 1e6:.9g} us")
    for label, channel, positive, delay_s, width_s in rows:
        polarity = "POS" if positive else "NEG"
        print(f"    {label}: channel {channel}, {polarity}, delay {delay_s * 1e6:.9g} us, width {width_s * 1e6:.9g} us")


if __name__ == "__main__":
    main()
