import ctypes
import sys

import pytest

from pytof_new.exceptions import DelayGeneratorError
from pytof_new.hardware.bme_driver import BMEDriverApi, BMEDriverError, BMEPciDelayGeneratorInfo


def test_bme_driver_api_is_lazy_until_first_use() -> None:
    api = BMEDriverApi(dll_path="does-not-exist.dll")

    assert api.loaded is False


def test_missing_dll_reports_clear_error(tmp_path) -> None:
    api = BMEDriverApi(dll_path=tmp_path / "DelayGenerator.dll")

    with pytest.raises(DelayGeneratorError, match="not found"):
        api.reserve_data(1)


def test_missing_symbol_reports_clear_error() -> None:
    fake = _FakeBMEDll()
    delattr(fake, "Set_G08_Delay")
    api = BMEDriverApi(fake)

    with pytest.raises(DelayGeneratorError, match="Set_G08_Delay"):
        api.bind_required_functions()


def test_bind_required_functions_sets_ctypes_prototypes() -> None:
    fake = _FakeBMEDll()
    api = BMEDriverApi(fake)

    api.bind_required_functions()

    assert fake.Set_G08_Delay.restype is ctypes.c_long
    assert fake.Set_G08_Delay.argtypes == [
        ctypes.c_ulong,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    assert fake.ReadTriggerCounter.restype is ctypes.c_ulong
    assert fake.ResetOutputModuloCounters.argtypes == [ctypes.c_long]
    assert fake.Read_DG_Status.argtypes == [ctypes.c_long]


def test_discovery_and_initialization_wrappers() -> None:
    fake = _FakeBMEDll()
    api = BMEDriverApi(fake)

    api.reserve_data(1)
    count, error = api.detect_pci_delay_generators()
    info = api.get_pci_delay_generator(0)
    api.initialize(info.slot, info.product, info.index)

    assert count == 1
    assert error == 0
    assert info == BMEPciDelayGeneratorInfo(product=47, slot=5, master=True, index=0)
    assert fake.calls[:4] == [
        ("Reserve_DG_Data", 1),
        ("DetectPciDelayGenerators",),
        ("GetPciDelayGenerator", 0),
        ("Initialize_DG_BME", 5, 47, 0),
    ]


def test_set_trigger_parameters_forwards_typed_values() -> None:
    fake = _FakeBMEDll()
    api = BMEDriverApi(fake)

    api.set_trigger_parameters(
        trigger_terminate=True,
        internal_clock_us=111.0,
        trigger_level_v=1.25,
        preset_value=4500,
        gate_divider=1,
        positive_gate=True,
        internal_trigger=True,
        internal_arm=False,
        software_trigger=False,
        rising_edge=True,
        stop_on_preset=True,
        reset_when_done=True,
        trigger_enable=True,
        index=0,
    )

    name, args = fake.calls[-1]
    assert name == "Set_TriggerParameters"
    assert args == (1, 111.0, 1.25, 4500, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0)


def test_set_g08_delay_forwards_typed_values() -> None:
    fake = _FakeBMEDll()
    api = BMEDriverApi(fake)

    api.set_g08_delay(
        channel=1,
        fire_first_us=0.5,
        pulse_width_us=56.0,
        output_modulo=1,
        output_offset=0,
        go_signal=0x10,
        positive=True,
        terminate=False,
        disconnect=False,
        onto_ms_bus=False,
        input_positive=True,
        index=0,
    )

    name, args = fake.calls[-1]
    assert name == "Set_G08_Delay"
    assert args == (1, 0.5, 56.0, 1, 0, 0x10, 1, 0, 0, 0, 1, 0)


def test_counter_status_and_deactivate_wrappers() -> None:
    fake = _FakeBMEDll()
    api = BMEDriverApi(fake)

    api.deactivate(0)
    api.reset_event_counter(0)
    api.reset_output_modulo_counters(0)
    trigger_count = api.read_trigger_counter(0)
    status = api.read_status(0)
    api.release_data()

    assert trigger_count == 12
    assert status == 0x40
    assert fake.calls[-6:] == [
        ("Deactivate_DG_BME", 0),
        ("ResetEventCounter", 0),
        ("ResetOutputModuloCounters", 0),
        ("ReadTriggerCounter", 0),
        ("Read_DG_Status", 0),
        ("Release_DG_Data",),
    ]


def test_nonzero_return_code_raises_typed_error() -> None:
    fake = _FakeBMEDll(error_code=11)
    api = BMEDriverApi(fake)

    with pytest.raises(BMEDriverError, match="PCI/PLX") as excinfo:
        api.reserve_data(1)

    assert excinfo.value.function == "Reserve_DG_Data"
    assert excinfo.value.code == 11


def test_architecture_mismatch_is_reported(tmp_path) -> None:
    dll_path = tmp_path / "DelayGenerator.dll"
    _write_minimal_pe(dll_path, machine=0x014C if sys.maxsize > 2**32 else 0x8664)
    api = BMEDriverApi(dll_path=dll_path)

    with pytest.raises(DelayGeneratorError, match="architecture mismatch"):
        api.reserve_data(1)


class _Func:
    def __init__(self, owner: "_FakeBMEDll", name: str, return_value: int = 0) -> None:
        self.owner = owner
        self.name = name
        self.return_value = return_value
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.owner._call(self.name, *args)


class _FakeBMEDll:
    def __init__(self, error_code: int = 0) -> None:
        self.error_code = error_code
        self.calls: list[tuple] = []
        for name in BMEDriverApi.REQUIRED_SYMBOLS:
            setattr(self, name, _Func(self, name))

    def _call(self, name: str, *args):
        if name == "Release_DG_Data":
            self.calls.append((name,))
            return self.error_code
        if name == "Reserve_DG_Data":
            self.calls.append((name, _as_int(args[0])))
            return self.error_code
        if name == "DetectPciDelayGenerators":
            args[0]._obj.value = 0
            self.calls.append((name,))
            return 1
        if name == "GetPciDelayGenerator":
            args[0]._obj.value = 47
            args[1]._obj.value = 5
            args[2]._obj.value = 1
            self.calls.append((name, _as_int(args[3])))
            return self.error_code
        if name == "Initialize_DG_BME":
            self.calls.append((name, _as_int(args[0]), _as_int(args[1]), _as_int(args[2])))
            return self.error_code
        if name in {"Set_TriggerParameters", "Set_G08_TriggerParameters", "Set_G08_ClockParameters", "Set_G08_Delay"}:
            self.calls.append((name, tuple(_as_int_or_float(arg) for arg in args)))
            return self.error_code
        if name in {"Activate_DG_BME", "Deactivate_DG_BME", "ResetEventCounter", "ResetOutputModuloCounters"}:
            self.calls.append((name, _as_int(args[0])))
            return self.error_code
        if name == "ReadTriggerCounter":
            self.calls.append((name, _as_int(args[0])))
            return 12
        if name == "Read_DG_Status":
            self.calls.append((name, _as_int(args[0])))
            return 0x40
        raise AssertionError(f"unexpected fake call: {name}")


def _as_int(value) -> int:
    return int(value.value if hasattr(value, "value") else value)


def _as_int_or_float(value):
    raw = value.value if hasattr(value, "value") else value
    if isinstance(raw, float):
        return float(raw)
    return int(raw)


def _write_minimal_pe(path, machine: int) -> None:
    data = bytearray(512)
    data[0:2] = b"MZ"
    data[0x3C:0x40] = (0x80).to_bytes(4, "little")
    data[0x80:0x84] = b"PE\0\0"
    data[0x84:0x86] = machine.to_bytes(2, "little")
    path.write_bytes(data)
