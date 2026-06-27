"""Tests for SpectrumAcquisitionService using a fake driver module."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore

from pytof_new.hardware.spectrum_driver import SpectrumDriverApi
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionRequest,
    SpectrumHardwareInfo,
    SpectrumTriggerSource,
)
from pytof_new.hardware.spectrum_service import SpectrumAcquisitionService, ServiceState


class _FakeSpectrumModule:
    """Minimal fake driver that lets SpectrumDigitizer connect, configure, and transfer."""

    ERR_OK = 0
    ERRORTEXTLEN = 1024

    def __init__(self) -> None:
        self.handle = object()
        self.i32: dict[int, int] = {}
        self.i64: dict[int, int] = {}
        self.calls: list[tuple] = []
        self.next_transfer_data: np.ndarray | None = None
        self.transfer_buffer = None
        self.transfer_length = 0
        # Required symbols
        self.CHANNEL0 = 1
        self.SPC_CHENABLE = 10
        self.SPC_AMP0 = 11
        self.SPC_CARDMODE = 12
        self.SPC_MEMSIZE = 13
        self.SPC_PRETRIGGER = 14
        self.SPC_POSTTRIGGER = 15
        self.SPC_SEGMENTSIZE = 16
        self.SPC_AVERAGES = 17
        self.SPC_CLOCKMODE = 18
        self.SPC_SAMPLERATE = 19
        self.SPC_CLOCKOUT = 20
        self.SPC_TRIG_ORMASK = 21
        self.SPC_TRIG_EXT0_MODE = 22
        self.SPC_TRIG_EXT0_LEVEL0 = 23
        self.SPC_TRIG_CH_ORMASK0 = 24
        self.SPC_TIMEOUT = 25
        self.SPC_REC_STD_MULTI = 1000
        self.SPC_REC_STD_AVERAGE = 1001
        self.SPC_CM_INTPLL = 2000
        self.SPC_TMASK_SOFTWARE = 3000
        self.SPC_TMASK_EXT0 = 3001
        self.SPC_TM_POS = 3002
        self.SPC_TMASK0_CH0 = 3003
        self.SPCM_BUF_DATA = 4000
        self.SPCM_DIR_CARDTOPC = 4001
        self.SPC_M2CMD = 5000
        self.M2CMD_CARD_START = 0x1
        self.M2CMD_CARD_ENABLETRIGGER = 0x2
        self.M2CMD_DATA_STARTDMA = 0x4
        self.M2CMD_CARD_WAITREADY = 0x8
        self.M2CMD_DATA_WAITDMA = 0x10
        self.M2CMD_CARD_STOP = 0x20
        self.M2CMD_CARD_DISABLETRIGGER = 0x40
        self.M2CMD_DATA_STOPDMA = 0x80
        # Optional discovery symbols
        self.SPC_PCITYP = 1
        self.SPC_PCISERIALNO = 2
        self.SPC_FNCTYPE = 3
        self.SPC_MIINST_MAXADCVALUE = 4

    def spcm_hOpen(self, _device):
        self.calls.append(("open",))
        return self.handle

    def spcm_vClose(self, handle):
        self.calls.append(("close", handle))

    def spcm_dwGetParam_i32(self, handle, register, value_ptr):
        self.calls.append(("get_i32", handle, register))
        value_ptr._obj.value = self.i32.get(register, 0)
        return self.ERR_OK

    def spcm_dwSetParam_i32(self, handle, register, value):
        self.calls.append(("set_i32", handle, register, value))
        if register == self.SPC_M2CMD:
            self.calls.append(("command", handle, value))
            if value & self.M2CMD_CARD_WAITREADY:
                self._fill_transfer_buffer()
        self.i32[register] = value
        return self.ERR_OK

    def spcm_dwSetParam_i64(self, handle, register, value):
        self.calls.append(("set_i64", handle, register, value))
        self.i64[register] = value
        return self.ERR_OK

    def spcm_dwDefTransfer_i64(self, handle, buffer_kind, direction, notify_size, buffer, offset, length):
        self.calls.append(("def_transfer", handle, buffer_kind, direction, notify_size, offset, length))
        self.transfer_buffer = buffer
        self.transfer_length = length
        return self.ERR_OK

    def spcm_dwGetErrorInfo_i32(self, _handle, _reg, _value, text_buffer):
        text_buffer.value = b""
        return self.ERR_OK

    def _fill_transfer_buffer(self) -> None:
        if self.transfer_buffer is None or self.next_transfer_data is None:
            return
        payload = self.next_transfer_data.tobytes()
        if len(payload) != self.transfer_length:
            raise AssertionError(f"fake payload {len(payload)} != expected {self.transfer_length}")
        self.transfer_buffer.raw = payload


@pytest.fixture()
def fake():
    return _FakeSpectrumModule()


@pytest.fixture()
def service(fake) -> SpectrumAcquisitionService:
    svc = SpectrumAcquisitionService(api=SpectrumDriverApi(fake))
    yield svc


def test_service_starts_idle(service: SpectrumAcquisitionService) -> None:
    assert service.state == ServiceState.IDLE


def test_connect_transitions_to_connected(service: SpectrumAcquisitionService, fake) -> None:
    fake.i32[fake.SPC_PCISERIALNO] = 12345
    results: list[str] = []
    info_results: list[object] = []
    service.state_changed.connect(results.append)
    service.hardware_info_ready.connect(info_results.append)

    _run_on_thread(service.connect_card)

    assert "connected" in results
    assert len(info_results) == 1
    assert isinstance(info_results[0], SpectrumHardwareInfo)
    assert service.state == ServiceState.CONNECTED


def test_configure_transitions_to_configured(service: SpectrumAcquisitionService) -> None:
    _run_on_thread(service.connect_card)
    results: list[str] = []
    configured_called: list[bool] = []
    service.state_changed.connect(results.append)
    service.configured.connect(lambda: configured_called.append(True))

    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    _run_on_thread(lambda: service.configure(request))

    assert "configured" in results
    assert configured_called
    assert service.state == ServiceState.CONFIGURED


def test_acquire_emits_result_ready(service: SpectrumAcquisitionService, fake) -> None:
    fake.i32[fake.SPC_PCISERIALNO] = 999
    fake.next_transfer_data = np.arange(1024, dtype=np.int8).reshape(1, 1024)
    _run_on_thread(service.connect_card)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    _run_on_thread(lambda: service.configure(request))

    results: list[object] = []
    service.result_ready.connect(results.append)
    _run_on_thread(service.acquire)

    assert len(results) == 1
    result = results[0]
    assert result.data.shape == (1, 1024)
    assert result.data.dtype == np.dtype(np.int8)
    assert service.state == ServiceState.CONFIGURED


def test_acquire_32bit_average_emits_fpga_result(service: SpectrumAcquisitionService, fake) -> None:
    fake.next_transfer_data = (np.arange(64, dtype=np.int32) * 10).reshape(1, 64)
    _run_on_thread(service.connect_card)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=64,
        pretrigger_samples=32,
        averages_per_segment=10,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )
    _run_on_thread(lambda: service.configure(request))

    results: list[object] = []
    service.result_ready.connect(results.append)
    _run_on_thread(service.acquire)

    assert len(results) == 1
    assert results[0].plan.is_fpga_sum is True
    assert results[0].plan.physical_shots_per_output_segment == 10


def test_split_acquisition_emits_prepared_armed_and_result(service: SpectrumAcquisitionService, fake) -> None:
    fake.next_transfer_data = np.arange(1024, dtype=np.int8).reshape(1, 1024)
    _run_on_thread(service.connect_card)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    _run_on_thread(lambda: service.configure(request))
    prepared: list[object] = []
    armed: list[bool] = []
    results: list[object] = []
    states: list[str] = []
    service.prepared.connect(prepared.append)
    service.armed.connect(lambda: armed.append(True))
    service.result_ready.connect(results.append)
    service.state_changed.connect(states.append)

    _run_on_thread(service.prepare)
    assert service.state == ServiceState.PREPARED
    assert len(prepared) == 1
    _run_on_thread(service.start_prepared)
    assert service.state == ServiceState.ARMED
    assert armed == [True]
    start_call = ("command", fake.handle, fake.M2CMD_CARD_START | fake.M2CMD_CARD_ENABLETRIGGER | fake.M2CMD_DATA_STARTDMA)
    wait_call = ("command", fake.handle, fake.M2CMD_CARD_WAITREADY | fake.M2CMD_DATA_WAITDMA)
    assert start_call in fake.calls
    assert wait_call not in fake.calls
    _run_on_thread(service.wait_result)

    assert service.state == ServiceState.CONFIGURED
    assert len(results) == 1
    assert results[0].data.shape == (1, 1024)
    assert "prepared" in states
    assert "armed" in states
    assert fake.calls.index(wait_call) > fake.calls.index(start_call)


def test_split_invalid_state_transitions_emit_errors(service: SpectrumAcquisitionService) -> None:
    errors: list[str] = []
    service.error_occurred.connect(errors.append)

    _run_on_thread(service.prepare)
    _run_on_thread(service.start_prepared)
    _run_on_thread(service.wait_result)

    assert "prepare cannot be called" in errors[0]
    assert "start_prepared cannot be called" in errors[1]
    assert "wait_result cannot be called" in errors[2]


def test_abort_cancels_acquisition(service: SpectrumAcquisitionService) -> None:
    _run_on_thread(service.connect_card)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    _run_on_thread(lambda: service.configure(request))

    errors: list[str] = []
    service.error_occurred.connect(errors.append)
    service.abort()
    _run_on_thread(service.acquire)

    assert service.state in (ServiceState.CONFIGURED, ServiceState.ERROR)


def test_acquire_without_config_emits_error(service: SpectrumAcquisitionService) -> None:
    _run_on_thread(service.connect_card)
    errors: list[str] = []
    service.error_occurred.connect(errors.append)

    _run_on_thread(service.acquire)

    assert len(errors) >= 1
    assert "acquire cannot be called" in errors[0]


def test_disconnect_returns_to_idle(service: SpectrumAcquisitionService) -> None:
    _run_on_thread(service.connect_card)
    _run_on_thread(service.disconnect_card)

    assert service.state == ServiceState.IDLE


def _run_on_thread(fn):
    """Execute a callable synchronously on the calling thread.

    The service's slots are designed to be callable from any thread for
    testing.  In production they should be invoked via queued signals.
    """
    fn()
