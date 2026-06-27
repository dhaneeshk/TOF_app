"""Lazy low-level adapter for the BME DelayGenerator DLL.

This module is import-safe on machines without BME hardware or vendor DLLs.
The DLL is loaded only when a real driver operation is requested.  Function
signatures here are flat exports verified from ``DG_DLL_1.h``; structure-based
APIs that require ``DG_Data.h`` are intentionally not bound.
"""

from __future__ import annotations

from ctypes import CDLL, POINTER, byref, c_double, c_int, c_long, c_ulong
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pytof_new.exceptions import DelayGeneratorError
from pytof_new.hardware.bme_constants import BME_SUCCESS, ERROR_DESCRIPTIONS, REQUIRED_FLAT_EXPORTS


_BOOL = c_int
_LONG = c_long
_ULONG = c_ulong
_DOUBLE = c_double


@dataclass(frozen=True)
class BMEPciDelayGeneratorInfo:
    """Detected PCI BME delay-generator identity."""

    product: int
    slot: int
    master: bool
    index: int


class BMEDriverError(DelayGeneratorError):
    """Raised when the BME driver reports an error or cannot be used."""

    def __init__(self, function: str, code: int, detail: str = "") -> None:
        self.function = function
        self.code = int(code)
        self.detail = detail or ERROR_DESCRIPTIONS.get(self.code, "unknown BME driver error")
        super().__init__(f"BME driver call failed during {function} with code {self.code}: {self.detail}")


class BMEDriverApi:
    """Small wrapper around flat DelayGenerator.dll exports.

    ``dll`` may be a fake object for unit tests.  When omitted, the vendor DLL
    is loaded lazily from ``dll_path`` or from normal Windows DLL search paths.
    """

    REQUIRED_SYMBOLS: tuple[str, ...] = REQUIRED_FLAT_EXPORTS

    def __init__(self, dll: Any | None = None, dll_path: str | Path | None = None) -> None:
        self._dll = dll
        self.dll_path = Path(dll_path) if dll_path is not None else None
        self._bound = False

    @property
    def loaded(self) -> bool:
        """Return True once a real or fake DLL object is present."""
        return self._dll is not None

    @property
    def dll(self) -> Any:
        """Return the loaded DLL, loading and binding it on first use."""
        if self._dll is None:
            path = self._resolve_dll_path()
            self._verify_dll_architecture(path)
            try:
                self._dll = CDLL(str(path))
            except OSError as exc:
                raise DelayGeneratorError(f"could not load BME DelayGenerator.dll from {path}: {exc}") from exc
        self.bind_required_functions()
        return self._dll

    def bind_required_functions(self) -> None:
        """Bind all required flat exports with prototypes from DG_DLL_1.h."""
        if self._bound:
            return
        dll = self._dll
        if dll is None:
            return
        for name in self.REQUIRED_SYMBOLS:
            self.require_symbol(name)

        self._prototype("Reserve_DG_Data", _LONG, [_LONG])
        self._prototype("Release_DG_Data", _LONG, [])
        self._prototype("DetectPciDelayGenerators", _LONG, [POINTER(_LONG)])
        self._prototype("GetPciDelayGenerator", _LONG, [POINTER(_LONG), POINTER(_LONG), POINTER(_BOOL), _LONG])
        self._prototype("Initialize_DG_BME", _LONG, [_LONG, _LONG, _LONG])
        self._prototype(
            "Set_TriggerParameters",
            _LONG,
            [_BOOL, _DOUBLE, _DOUBLE, _ULONG, _ULONG, _BOOL, _BOOL, _BOOL, _BOOL, _BOOL, _BOOL, _BOOL, _BOOL, _LONG],
        )
        self._prototype(
            "Set_G08_TriggerParameters",
            _LONG,
            [_BOOL, _DOUBLE, _DOUBLE, _BOOL, _BOOL, _DOUBLE, _DOUBLE, _ULONG, _ULONG, _LONG],
        )
        self._prototype("Set_G08_ClockParameters", _LONG, [_BOOL, _ULONG, _ULONG, _ULONG, _ULONG, _LONG])
        self._prototype(
            "Set_G08_Delay",
            _LONG,
            [_ULONG, _DOUBLE, _DOUBLE, _ULONG, _ULONG, _ULONG, _BOOL, _BOOL, _BOOL, _BOOL, _BOOL, _LONG],
        )
        self._prototype("Activate_DG_BME", _LONG, [_LONG])
        self._prototype("Deactivate_DG_BME", _LONG, [_LONG])
        self._prototype("ResetEventCounter", _LONG, [_LONG])
        self._prototype("ResetOutputModuloCounters", _LONG, [_LONG])
        self._prototype("ReadTriggerCounter", _ULONG, [_LONG])
        self._prototype("Read_DG_Status", _ULONG, [_LONG])
        self._bound = True

    def require_symbol(self, name: str) -> Any:
        """Return a DLL symbol or raise a clear delay-generator error."""
        dll = self._dll
        if dll is None:
            dll = self.dll
        try:
            return getattr(dll, name)
        except AttributeError as exc:
            raise DelayGeneratorError(f"BME DelayGenerator.dll symbol is not available: {name}") from exc

    def reserve_data(self, number_of_delay_generators: int) -> None:
        self.check(self.dll.Reserve_DG_Data(_LONG(number_of_delay_generators)), "Reserve_DG_Data")

    def release_data(self) -> None:
        self.check(self.dll.Release_DG_Data(), "Release_DG_Data")

    def detect_pci_delay_generators(self) -> tuple[int, int]:
        error = _LONG(0)
        count = int(self.dll.DetectPciDelayGenerators(byref(error)))
        return count, int(error.value)

    def get_pci_delay_generator(self, index: int) -> BMEPciDelayGeneratorInfo:
        product = _LONG(0)
        slot = _LONG(0)
        master = _BOOL(0)
        code = self.dll.GetPciDelayGenerator(byref(product), byref(slot), byref(master), _LONG(index))
        self.check(code, "GetPciDelayGenerator")
        return BMEPciDelayGeneratorInfo(product=int(product.value), slot=int(slot.value), master=bool(master.value), index=int(index))

    def initialize(self, slot: int, product: int, index: int) -> None:
        code = self.dll.Initialize_DG_BME(_LONG(slot), _LONG(product), _LONG(index))
        self.check(code, "Initialize_DG_BME")

    def set_trigger_parameters(
        self,
        *,
        trigger_terminate: bool,
        internal_clock_us: float,
        trigger_level_v: float,
        preset_value: int,
        gate_divider: int,
        positive_gate: bool,
        internal_trigger: bool,
        internal_arm: bool,
        software_trigger: bool,
        rising_edge: bool,
        stop_on_preset: bool,
        reset_when_done: bool,
        trigger_enable: bool,
        index: int,
    ) -> None:
        code = self.dll.Set_TriggerParameters(
            _BOOL(trigger_terminate),
            _DOUBLE(internal_clock_us),
            _DOUBLE(trigger_level_v),
            _ULONG(preset_value),
            _ULONG(gate_divider),
            _BOOL(positive_gate),
            _BOOL(internal_trigger),
            _BOOL(internal_arm),
            _BOOL(software_trigger),
            _BOOL(rising_edge),
            _BOOL(stop_on_preset),
            _BOOL(reset_when_done),
            _BOOL(trigger_enable),
            _LONG(index),
        )
        self.check(code, "Set_TriggerParameters")

    def set_g08_trigger_parameters(
        self,
        *,
        gate_terminate: bool,
        gate_level_v: float,
        gate_delay_us: float,
        ignore_gate: bool,
        synchronize_gate: bool,
        force_trigger_us: float,
        step_back_time_us: float,
        burst_counter: int,
        ms_bus: int,
        index: int,
    ) -> None:
        code = self.dll.Set_G08_TriggerParameters(
            _BOOL(gate_terminate),
            _DOUBLE(gate_level_v),
            _DOUBLE(gate_delay_us),
            _BOOL(ignore_gate),
            _BOOL(synchronize_gate),
            _DOUBLE(force_trigger_us),
            _DOUBLE(step_back_time_us),
            _ULONG(burst_counter),
            _ULONG(ms_bus),
            _LONG(index),
        )
        self.check(code, "Set_G08_TriggerParameters")

    def set_g08_clock_parameters(
        self,
        *,
        clock_enable: bool,
        oscillator_divider: int,
        trigger_divider: int,
        trigger_multiplier: int,
        clock_source: int,
        index: int,
    ) -> None:
        code = self.dll.Set_G08_ClockParameters(
            _BOOL(clock_enable),
            _ULONG(oscillator_divider),
            _ULONG(trigger_divider),
            _ULONG(trigger_multiplier),
            _ULONG(clock_source),
            _LONG(index),
        )
        self.check(code, "Set_G08_ClockParameters")

    def set_g08_delay(
        self,
        *,
        channel: int,
        fire_first_us: float,
        pulse_width_us: float,
        output_modulo: int,
        output_offset: int,
        go_signal: int,
        positive: bool,
        terminate: bool,
        disconnect: bool,
        onto_ms_bus: bool,
        input_positive: bool,
        index: int,
    ) -> None:
        code = self.dll.Set_G08_Delay(
            _ULONG(channel),
            _DOUBLE(fire_first_us),
            _DOUBLE(pulse_width_us),
            _ULONG(output_modulo),
            _ULONG(output_offset),
            _ULONG(go_signal),
            _BOOL(positive),
            _BOOL(terminate),
            _BOOL(disconnect),
            _BOOL(onto_ms_bus),
            _BOOL(input_positive),
            _LONG(index),
        )
        self.check(code, "Set_G08_Delay")

    def activate(self, index: int) -> None:
        self.check(self.dll.Activate_DG_BME(_LONG(index)), "Activate_DG_BME")

    def deactivate(self, index: int) -> None:
        self.check(self.dll.Deactivate_DG_BME(_LONG(index)), "Deactivate_DG_BME")

    def reset_event_counter(self, index: int) -> None:
        self.check(self.dll.ResetEventCounter(_LONG(index)), "ResetEventCounter")

    def reset_output_modulo_counters(self, index: int) -> None:
        self.check(self.dll.ResetOutputModuloCounters(_LONG(index)), "ResetOutputModuloCounters")

    def read_trigger_counter(self, index: int) -> int:
        return int(self.dll.ReadTriggerCounter(_LONG(index)))

    def read_status(self, index: int) -> int:
        return int(self.dll.Read_DG_Status(_LONG(index)))

    def check(self, code: int, function: str) -> None:
        """Raise a typed exception for nonzero BME return codes."""
        error_code = int(code)
        if error_code == BME_SUCCESS:
            return
        raise BMEDriverError(function, error_code)

    def _prototype(self, name: str, restype: Any, argtypes: list[Any]) -> None:
        symbol = self.require_symbol(name)
        try:
            symbol.restype = restype
            symbol.argtypes = argtypes
        except AttributeError:
            # Plain Python fakes do not expose ctypes prototype attributes.
            return

    def _resolve_dll_path(self) -> Path:
        if self.dll_path is not None:
            path = self.dll_path
            if not path.exists():
                raise DelayGeneratorError(f"BME DelayGenerator.dll was not found: {path}")
            return path
        for path in _default_dll_candidates():
            if path.exists():
                return path
        raise DelayGeneratorError("BME DelayGenerator.dll was not found")

    def _verify_dll_architecture(self, path: Path) -> None:
        dll_bits = _pe_architecture_bits(path)
        python_bits = _python_bits()
        if dll_bits is None:
            raise DelayGeneratorError(f"could not determine BME DLL architecture: {path}")
        if dll_bits != python_bits:
            raise DelayGeneratorError(f"BME DLL architecture mismatch: DLL is {dll_bits}-bit, Python is {python_bits}-bit")


def _default_dll_candidates() -> tuple[Path, ...]:
    package_root = Path(__file__).resolve().parents[2]
    project_root = package_root.parent
    return (
        package_root / "DelayGenerator.dll",
        project_root / "DelayGenerator.dll",
        Path.cwd() / "DelayGenerator.dll",
        Path("DelayGenerator.dll"),
    )


def _python_bits() -> int:
    import struct

    return struct.calcsize("P") * 8


def _pe_architecture_bits(path: Path) -> int | None:
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    pe_offset = int.from_bytes(data[0x3C:0x40], "little")
    if pe_offset + 6 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        return None
    machine = int.from_bytes(data[pe_offset + 4 : pe_offset + 6], "little")
    if machine == 0x8664:
        return 64
    if machine == 0x014C:
        return 32
    return None
