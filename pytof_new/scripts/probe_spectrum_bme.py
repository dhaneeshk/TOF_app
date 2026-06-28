#!/usr/bin/env python3
"""Combined Spectrum+BME synchronization and pickup-timing probe.

This script is intended for the control machine with both hardware devices.
It first verifies that BME connect/configure/arm does not trigger Spectrum, then
activates BME and checks for pickup/ringing events at expected pulse edges.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytof_new.config.models import BMEConfig
from pytof_new.hardware.bme_delay_generator import BMEDelayGenerator
from pytof_new.hardware.bme_driver import BMEDriverApi
from pytof_new.hardware.spectrum_digitizer import SpectrumDigitizer
from pytof_new.hardware.spectrum_driver import SpectrumDriverError
from pytof_new.hardware.spectrum_models import SpectrumAcquisitionMode, SpectrumAcquisitionRequest, SpectrumTriggerSource


DEFAULT_EVENTS_US = (
    ("A start", 0.0),
    ("C start", 5.0),
    ("A end", 7.0),
    ("F start", 10.0),
    ("C end", 12.0),
    ("F end", 17.0),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe coordinated Spectrum+BME triggering")
    parser.add_argument("--device", default="/dev/spcm0", help="Spectrum device path")
    parser.add_argument("--dll", type=Path, default=None, help="Path to DelayGenerator.dll")
    parser.add_argument("--card-index", type=int, default=0, help="BME card index")
    parser.add_argument("--mode", choices=("raw_multi", "average_32bit", "average_16bit", "all"), default="raw_multi")
    parser.add_argument("--segments", type=int, default=1, help="Spectrum output records per batch")
    parser.add_argument("--segment-samples", type=int, default=65536)
    parser.add_argument("--pretrigger", type=int, default=4096)
    parser.add_argument("--sample-rate", type=float, default=1.25e9)
    parser.add_argument("--input-range-v", type=float, default=0.5)
    parser.add_argument("--trigger-level-v", type=float, default=1.5)
    parser.add_argument("--safety-timeout-s", type=float, default=0.5)
    parser.add_argument("--acquire-timeout-s", type=float, default=5.0)
    parser.add_argument("--averages", type=int, default=32)
    parser.add_argument("--pulse-count", type=int, default=None, help="Defaults to Spectrum physical trigger count; override requires --allow-trigger-count-mismatch")
    parser.add_argument("--allow-trigger-count-mismatch", action="store_true", help="Allow --pulse-count to differ from Spectrum-required triggers")
    parser.add_argument("--tof-window-us", type=float, default=25.0)
    parser.add_argument("--repetition-us", type=float, default=111.0)
    parser.add_argument("--edge-width-us", type=float, default=7.0)
    parser.add_argument("--push-delay-us", type=float, default=5.0)
    parser.add_argument("--pull-delay-us", type=float, default=10.0)
    parser.add_argument("--window-us", type=float, default=0.4, help="Pickup detection window half-width")
    parser.add_argument("--min-sigma", type=float, default=6.0, help="Derivative threshold in robust sigma")
    parser.add_argument("--min-abs", type=float, default=2.0, help="Minimum absolute derivative threshold")
    parser.add_argument("--output-prefix", default="spectrum_bme_probe")
    parser.add_argument("--skip-safety", action="store_true", help="Skip no-early-pulse safety phase")
    args = parser.parse_args()

    if not _hardware_tests_enabled():
        print("ERROR: this combined hardware probe requires PYTOF_RUN_HARDWARE_TESTS=1", file=sys.stderr)
        return 1
    if args.pretrigger >= args.segment_samples:
        print("ERROR: --pretrigger must be smaller than --segment-samples", file=sys.stderr)
        return 1
    if args.segments <= 0 or args.averages <= 0:
        print("ERROR: --segments and --averages must be positive", file=sys.stderr)
        return 1

    modes = ["raw_multi", "average_32bit", "average_16bit"] if args.mode == "all" else [args.mode]
    spectrum = SpectrumDigitizer(device=args.device)
    bme = BMEDelayGenerator(api=BMEDriverApi(dll_path=args.dll), card_index=args.card_index)
    try:
        spectrum.connect()
        print(f"Connected Spectrum: {spectrum.hardware_info}")

        failures = 0
        for mode_name in modes:
            if mode_name == "average_16bit" and spectrum.hardware_info is not None and not spectrum.hardware_info.average_16bit_supported:
                print("\n=== average_16bit: SKIPPED (not reported supported) ===")
                continue
            print(f"\n=== Mode: {mode_name} ===")
            request = _request_for_mode(args, mode_name)
            physical_triggers = request.number_of_segments * (request.averages_per_segment if mode_name != "raw_multi" else 1)
            if args.pulse_count is not None and args.pulse_count != physical_triggers and not args.allow_trigger_count_mismatch:
                print(
                    f"ERROR: --pulse-count {args.pulse_count} differs from Spectrum-required triggers {physical_triggers}. "
                    "Use --allow-trigger-count-mismatch only for deliberate mismatch testing.",
                    file=sys.stderr,
                )
                failures += 1
                continue
            pulse_count = args.pulse_count if args.pulse_count is not None else physical_triggers
            bme_config = _bme_config(args)
            try:
                bme_config.validate()
                _validate_repetition_vs_record(request, bme_config)
            except ValueError as exc:
                print(f"ERROR: invalid combined probe configuration: {exc}", file=sys.stderr)
                failures += 1
                continue
            events_us = expected_events_us(bme_config)

            if not args.skip_safety:
                if not _run_no_early_pulse_check(spectrum, bme, request, bme_config, pulse_count, args.safety_timeout_s):
                    failures += 1
                    continue

            result = _run_activation_acquisition(spectrum, bme, request, bme_config, pulse_count, args.acquire_timeout_s)
            if result is None:
                failures += 1
                continue
            trace = _first_trace(result.data)
            prefix = Path(f"{args.output_prefix}_{mode_name}")
            _save_result(prefix, result.data, request, bme_config, pulse_count, bme.read_trigger_count(), bme.read_status())
            detections = detect_pickup_events_for_records(
                result.data,
                sample_rate_hz=request.sample_rate_hz,
                pretrigger_samples=request.pretrigger_samples,
                events_us=events_us,
                window_us=args.window_us,
                min_sigma=args.min_sigma,
                min_abs=args.min_abs,
            )
            _print_detections(detections)
            if not all(item["detection_fraction"] > 0 for item in detections):
                print("WARNING: not all expected pickup events were detected automatically. Inspect saved trace files.")
        return 1 if failures else 0
    finally:
        try:
            bme.emergency_stop()
        finally:
            bme.close()
            spectrum.close()


def _hardware_tests_enabled() -> bool:
    return os.environ.get("PYTOF_RUN_HARDWARE_TESTS", "").strip().lower() in {"1", "true", "yes"}


def _request_for_mode(args: argparse.Namespace, mode_name: str) -> SpectrumAcquisitionRequest:
    mode = {
        "raw_multi": SpectrumAcquisitionMode.RAW_MULTI,
        "average_32bit": SpectrumAcquisitionMode.AVERAGE_32BIT,
        "average_16bit": SpectrumAcquisitionMode.AVERAGE_16BIT,
    }[mode_name]
    return SpectrumAcquisitionRequest(
        mode=mode,
        sample_rate_hz=args.sample_rate,
        segment_samples=args.segment_samples,
        pretrigger_samples=args.pretrigger,
        number_of_segments=args.segments,
        averages_per_segment=args.averages if mode != SpectrumAcquisitionMode.RAW_MULTI else 1,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
        input_range_v=args.input_range_v,
        trigger_level_v=args.trigger_level_v,
        timeout_s=args.acquire_timeout_s,
    )


def _bme_config(args: argparse.Namespace) -> BMEConfig:
    return BMEConfig(
        advanced_mode=True,
        tof_window_s=args.tof_window_us * 1e-6,
        extraction_region_fill_time_s=(args.repetition_us - args.tof_window_us) * 1e-6,
        repetition_period_s=args.repetition_us * 1e-6,
        digitizer_trigger_delay_s=0.0,
        push_trigger_delay_s=args.push_delay_us * 1e-6,
        pull_trigger_delay_s=args.pull_delay_us * 1e-6,
        digitizer_trigger_width_s=args.edge_width_us * 1e-6,
        push_trigger_width_s=args.edge_width_us * 1e-6,
        pull_trigger_width_s=args.edge_width_us * 1e-6,
        digitizer_channel="A",
        push_channel="C",
        pull_channel="F",
        digitizer_polarity_positive=True,
        push_polarity_positive=True,
        pull_polarity_positive=False,
        trigger_termination_ohm=50,
    )


def _validate_repetition_vs_record(request: SpectrumAcquisitionRequest, config: BMEConfig) -> None:
    record_duration_s = request.segment_samples / request.sample_rate_hz
    rearm_margin_s = (80 + request.pretrigger_samples) / request.sample_rate_hz
    minimum_s = record_duration_s + rearm_margin_s
    if config.repetition_period_s <= minimum_s:
        raise ValueError(
            "BME repetition period must exceed Spectrum record duration plus rearm margin "
            f"({config.repetition_period_s * 1e6:.3f} us <= {minimum_s * 1e6:.3f} us)"
        )


def expected_events_us(config: BMEConfig) -> tuple[tuple[str, float], ...]:
    events = (
        (f"{config.digitizer_channel} start", config.digitizer_trigger_delay_s * 1e6),
        (f"{config.digitizer_channel} end", (config.digitizer_trigger_delay_s + config.digitizer_trigger_width_s) * 1e6),
        (f"{config.push_channel} start", config.push_trigger_delay_s * 1e6),
        (f"{config.push_channel} end", (config.push_trigger_delay_s + config.push_trigger_width_s) * 1e6),
        (f"{config.pull_channel} start", config.pull_trigger_delay_s * 1e6),
        (f"{config.pull_channel} end", (config.pull_trigger_delay_s + config.pull_trigger_width_s) * 1e6),
    )
    return tuple(sorted(events, key=lambda item: item[1]))


def _run_no_early_pulse_check(
    spectrum: SpectrumDigitizer,
    bme: BMEDelayGenerator,
    request: SpectrumAcquisitionRequest,
    config: BMEConfig,
    pulse_count: int,
    timeout_s: float,
) -> bool:
    print("Safety phase: Spectrum waits while BME connect/configure/arm occurs; expected result is timeout.")
    safety_request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=request.sample_rate_hz,
        segment_samples=request.segment_samples,
        pretrigger_samples=request.pretrigger_samples,
        number_of_segments=1,
        averages_per_segment=1,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
        input_range_v=request.input_range_v,
        trigger_level_v=request.trigger_level_v,
        timeout_s=timeout_s,
    )
    spectrum.configure_request(safety_request)
    spectrum.prepare_configured_acquisition()
    try:
        spectrum.start_prepared_acquisition()
        if not bme.connected:
            print("Connecting BME while Spectrum is already waiting...")
            bme.connect()
            print(f"Connected BME: {bme.info}")
        bme.stop()
        bme.configure(config)
        bme.arm(pulse_count)
        try:
            result = spectrum.wait_for_prepared_result()
        except SpectrumDriverError as exc:
            if _is_timeout(spectrum, exc):
                print("PASS: Spectrum timed out before BME activation; no early pulse detected.")
                return True
            print(f"FAIL: Spectrum wait failed with non-timeout error: {exc}")
            return False
        print(f"FAIL: Spectrum acquired data before BME activation: shape={result.data.shape}")
        return False
    finally:
        _safe_stop_bme_then_spectrum(bme, spectrum)


def _run_activation_acquisition(
    spectrum: SpectrumDigitizer,
    bme: BMEDelayGenerator,
    request: SpectrumAcquisitionRequest,
    config: BMEConfig,
    pulse_count: int,
    timeout_s: float,
):
    print("Activation phase: Spectrum armed first, then BME activated.")
    acquire_request = SpectrumAcquisitionRequest(**{**request.__dict__, "timeout_s": timeout_s})
    spectrum.configure_request(acquire_request)
    if not bme.connected:
        bme.connect()
        print(f"Connected BME: {bme.info}")
    bme.stop()
    bme.configure(config)
    bme.arm(pulse_count)
    print("Planned BME events:")
    for label, time_us in expected_events_us(config):
        print(f"  {label}: {time_us:.9g} us")
    spectrum.prepare_configured_acquisition()
    try:
        spectrum.start_prepared_acquisition()
        bme.start()
        result = spectrum.wait_for_prepared_result()
        actual_count = bme.read_trigger_count()
        status = bme.read_status()
        print(f"PASS: Spectrum acquired data: shape={result.data.shape} dtype={result.data.dtype}")
        print(f"BME trigger counter: expected {pulse_count}, actual {actual_count}, status {status}")
        if actual_count != pulse_count:
            print("FAIL: BME counter does not match expected pulse count")
            return None
        return result
    except SpectrumDriverError as exc:
        if _is_timeout(spectrum, exc):
            print("FAIL: Spectrum timed out after BME activation")
            return None
        raise
    finally:
        _safe_stop_bme_then_spectrum(bme, spectrum)


def _safe_stop_bme_then_spectrum(bme: BMEDelayGenerator, spectrum: SpectrumDigitizer) -> None:
    errors = []
    try:
        bme.stop()
    except Exception as exc:
        errors.append(f"BME stop failed: {exc}")
    try:
        spectrum.stop()
    except Exception as exc:
        errors.append(f"Spectrum stop failed: {exc}")
    for message in errors:
        print(f"WARNING: {message}", file=sys.stderr)


def _is_timeout(spectrum: SpectrumDigitizer, exc: SpectrumDriverError) -> bool:
    try:
        return int(exc.code) == int(getattr(spectrum.api.module, "ERR_TIMEOUT"))
    except Exception:
        return "timeout" in str(exc).lower()


def _first_trace(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 1:
        return array.astype(np.float64)
    return array[0].astype(np.float64)


def detect_pickup_events(
    trace: np.ndarray,
    *,
    sample_rate_hz: float,
    pretrigger_samples: int,
    events_us=DEFAULT_EVENTS_US,
    window_us: float = 0.4,
    min_sigma: float = 6.0,
    min_abs: float = 2.0,
) -> list[dict[str, object]]:
    """Detect pickup events by looking for large local derivatives."""
    trace = np.asarray(trace, dtype=np.float64)
    derivative = np.abs(np.diff(trace, prepend=trace[0]))
    baseline_stop = max(2, min(pretrigger_samples, derivative.size // 4 if derivative.size >= 8 else derivative.size))
    baseline = derivative[:baseline_stop]
    median = float(np.median(baseline)) if baseline.size else 0.0
    mad = float(np.median(np.abs(baseline - median))) if baseline.size else 0.0
    robust_sigma = 1.4826 * mad if mad > 0 else float(np.std(baseline)) if baseline.size else 0.0
    threshold = max(float(min_abs), median + float(min_sigma) * robust_sigma)
    half_window = max(1, int(round(window_us * 1e-6 * sample_rate_hz)))
    results = []
    for label, time_us in events_us:
        center = int(round(pretrigger_samples + time_us * 1e-6 * sample_rate_hz))
        start = max(0, center - half_window)
        stop = min(derivative.size, center + half_window + 1)
        local = derivative[start:stop]
        if local.size == 0:
            peak_value = 0.0
            peak_index = center
        else:
            rel = int(np.argmax(local))
            peak_index = start + rel
            peak_value = float(local[rel])
        results.append(
            {
                "label": label,
                "expected_us": float(time_us),
                "detected": bool(peak_value >= threshold),
                "peak_derivative": peak_value,
                "threshold": threshold,
                "peak_time_us": (peak_index - pretrigger_samples) / sample_rate_hz * 1e6,
            }
        )
    return results


def detect_pickup_events_for_records(
    data: np.ndarray,
    *,
    sample_rate_hz: float,
    pretrigger_samples: int,
    events_us=DEFAULT_EVENTS_US,
    window_us: float = 0.4,
    min_sigma: float = 6.0,
    min_abs: float = 2.0,
) -> list[dict[str, object]]:
    array = np.asarray(data)
    records = array.reshape((1, array.shape[0])) if array.ndim == 1 else array.reshape((-1, array.shape[-1]))
    per_record = [
        detect_pickup_events(
            record,
            sample_rate_hz=sample_rate_hz,
            pretrigger_samples=pretrigger_samples,
            events_us=events_us,
            window_us=window_us,
            min_sigma=min_sigma,
            min_abs=min_abs,
        )
        for record in records
    ]
    summary = []
    for event_index, (label, expected_us) in enumerate(events_us):
        items = [record[event_index] for record in per_record]
        detected = [item for item in items if item["detected"]]
        peak_times = [float(item["peak_time_us"]) for item in detected]
        summary.append(
            {
                "label": label,
                "expected_us": float(expected_us),
                "record_count": len(items),
                "detected_count": len(detected),
                "detection_fraction": len(detected) / len(items) if items else 0.0,
                "mean_peak_time_us": float(np.mean(peak_times)) if peak_times else float("nan"),
                "std_peak_time_us": float(np.std(peak_times)) if peak_times else float("nan"),
                "max_peak_derivative": max(float(item["peak_derivative"]) for item in items) if items else 0.0,
                "threshold": max(float(item["threshold"]) for item in items) if items else 0.0,
            }
        )
    return summary


def _print_detections(detections: list[dict[str, object]]) -> None:
    print("Pickup detection (derivative-based, inspect trace if WARN):")
    for item in detections:
        status = "PASS" if item.get("detection_fraction", 0.0) > 0 else "WARN"
        print(
            f"  {status}: {item['label']} expected {item['expected_us']:.3g} us, "
            f"detected {item.get('detected_count', 0)}/{item.get('record_count', 1)} records, "
            f"mean peak {item.get('mean_peak_time_us', float('nan')):.3g} us, "
            f"std {item.get('std_peak_time_us', float('nan')):.3g} us, "
            f"max derivative {item.get('max_peak_derivative', 0.0):.4g}, threshold {item['threshold']:.4g}"
        )


def _save_trace(prefix: Path, trace: np.ndarray, sample_rate_hz: float, pretrigger_samples: int) -> None:
    np.save(prefix.with_suffix(".npy"), trace)
    time_us = (np.arange(trace.size) - pretrigger_samples) / sample_rate_hz * 1e6
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_us", "adc"])
        writer.writerows(zip(time_us, trace, strict=True))
    print(f"Saved trace: {prefix.with_suffix('.npy')} and {prefix.with_suffix('.csv')}")


def _save_result(
    prefix: Path,
    data: np.ndarray,
    request: SpectrumAcquisitionRequest,
    config: BMEConfig,
    expected_count: int,
    actual_count: int,
    bme_status: int,
) -> None:
    trace = _first_trace(data)
    _save_trace(prefix, trace, request.sample_rate_hz, request.pretrigger_samples)
    np.savez_compressed(prefix.with_suffix(".npz"), data=data)
    metadata = {
        "spectrum_request": {key: str(value) for key, value in request.__dict__.items()},
        "bme_config": {key: str(value) for key, value in config.__dict__.items()},
        "bme_expected_trigger_count": expected_count,
        "bme_actual_trigger_count": actual_count,
        "bme_status": bme_status,
        "expected_events_us": list(expected_events_us(config)),
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved full result: {prefix.with_suffix('.npz')} and metadata {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    raise SystemExit(main())
