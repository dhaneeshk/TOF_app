import importlib
import sys

import pytest

from pytof_new.exceptions import DigitizerError
from pytof_new.hardware.spectrum_driver import SpectrumDriverApi, SpectrumDriverError


def test_driver_api_is_lazy_until_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_import(name: str):
        calls.append(name)
        raise AssertionError("unexpected import")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    api = SpectrumDriverApi()

    assert api.loaded is False
    assert calls == []


def test_missing_symbol_reports_clear_error() -> None:
    api = SpectrumDriverApi(_FakeSpectrumModule())

    with pytest.raises(DigitizerError, match="NOT_A_SYMBOL"):
        api.require_symbol("NOT_A_SYMBOL")


def test_get_set_and_command_call_fake_driver() -> None:
    fake = _FakeSpectrumModule()
    api = SpectrumDriverApi(fake)
    handle = api.open()

    api.set_i32(handle, fake.SPC_CARDMODE, 22)
    api.set_i64(handle, fake.SPC_MEMSIZE, 2048)
    api.command(handle, fake.M2CMD_CARD_START)

    assert api.get_i32(handle, fake.SPC_CARDMODE) == 22
    assert api.get_i64(handle, fake.SPC_MEMSIZE) == 2048
    assert ("set_i32", handle, fake.SPC_M2CMD, fake.M2CMD_CARD_START) in fake.calls


def test_def_transfer_forwards_arguments() -> None:
    fake = _FakeSpectrumModule()
    api = SpectrumDriverApi(fake)
    handle = api.open()
    buffer = object()

    api.def_transfer(handle, fake.SPCM_BUF_DATA, fake.SPCM_DIR_CARDTOPC, 0, buffer, 5, 64)

    assert fake.calls[-1] == ("def_transfer", handle, fake.SPCM_BUF_DATA, fake.SPCM_DIR_CARDTOPC, 0, buffer, 5, 64)


def test_driver_error_includes_error_text() -> None:
    fake = _FakeSpectrumModule(error_code=123, error_text=b"bad register")
    api = SpectrumDriverApi(fake)
    handle = api.open()

    with pytest.raises(SpectrumDriverError, match="bad register") as excinfo:
        api.set_i32(handle, fake.SPC_CARDMODE, 1)

    assert excinfo.value.operation == "set_i32"
    assert excinfo.value.code == 123


def test_import_does_not_load_pyspcm() -> None:
    sys.modules.pop("pyspcm", None)

    import pytof_new.hardware.spectrum_driver  # noqa: F401

    assert "pyspcm" not in sys.modules


class _FakeSpectrumModule:
    ERR_OK = 0
    ERRORTEXTLEN = 1024
    SPC_CARDMODE = 10
    SPC_MEMSIZE = 11
    SPC_M2CMD = 12
    M2CMD_CARD_START = 0x1
    SPCM_BUF_DATA = 1000
    SPCM_DIR_CARDTOPC = 1

    def __init__(self, error_code: int = 0, error_text: bytes = b"") -> None:
        self.error_code = error_code
        self.error_text = error_text
        self.handle = object()
        self.i32: dict[int, int] = {}
        self.i64: dict[int, int] = {}
        self.calls: list[tuple] = []

    def spcm_hOpen(self, _device):
        self.calls.append(("open",))
        return self.handle

    def spcm_vClose(self, handle):
        self.calls.append(("close", handle))

    def spcm_dwGetParam_i32(self, handle, register, value_ptr):
        self.calls.append(("get_i32", handle, register))
        value_ptr._obj.value = self.i32.get(register, 0)
        return self.ERR_OK

    def spcm_dwGetParam_i64(self, handle, register, value_ptr):
        self.calls.append(("get_i64", handle, register))
        value_ptr._obj.value = self.i64.get(register, 0)
        return self.ERR_OK

    def spcm_dwSetParam_i32(self, handle, register, value):
        self.calls.append(("set_i32", handle, register, value))
        if self.error_code:
            return self.error_code
        self.i32[register] = value
        return self.ERR_OK

    def spcm_dwSetParam_i64(self, handle, register, value):
        self.calls.append(("set_i64", handle, register, value))
        if self.error_code:
            return self.error_code
        self.i64[register] = value
        return self.ERR_OK

    def spcm_dwDefTransfer_i64(self, handle, buffer_kind, direction, notify_size, buffer, offset, length):
        self.calls.append(("def_transfer", handle, buffer_kind, direction, notify_size, buffer, offset, length))
        if self.error_code:
            return self.error_code
        return self.ERR_OK

    def spcm_dwGetErrorInfo_i32(self, _handle, _reg, _value, text_buffer):
        text_buffer.value = self.error_text
        return self.ERR_OK
