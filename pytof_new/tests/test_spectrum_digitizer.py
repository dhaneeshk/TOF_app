import sys

import numpy as np
import pytest

from pytof_new.exceptions import DigitizerError
from pytof_new.hardware.spectrum_digitizer import SpectrumDigitizer
from pytof_new.hardware.spectrum_driver import SpectrumDriverApi
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionRequest,
    SpectrumTriggerSource,
)


def test_import_does_not_load_pyspcm() -> None:
    sys.modules.pop("pyspcm", None)

    import pytof_new.hardware.spectrum_digitizer  # noqa: F401

    assert "pyspcm" not in sys.modules


def test_connect_opens_once_and_discovers_info() -> None:
    fake = _FakeSpectrumModule()
    fake.i32[fake.SPC_PCISERIALNO] = 12345
    fake.i32[fake.SPC_MIINST_MAXADCVALUE] = 127
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))

    digitizer.connect()
    digitizer.connect()

    assert digitizer.connected is True
    assert digitizer.hardware_info is not None
    assert digitizer.hardware_info.serial_number == 12345
    assert digitizer.hardware_info.max_adc_value == 127
    assert fake.calls.count(("open",)) == 1


def test_close_closes_once() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))

    digitizer.connect()
    handle = digitizer.handle
    digitizer.close()
    digitizer.close()

    assert digitizer.connected is False
    assert fake.calls.count(("close", handle)) == 1


def test_optional_missing_symbols_do_not_break_connect() -> None:
    fake = _FakeSpectrumModule(include_optional_registers=False)
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))

    digitizer.connect()

    assert digitizer.hardware_info is not None
    assert digitizer.hardware_info.serial_number is None
    assert digitizer.hardware_info.metadata == {}


def test_detects_16bit_average_symbol() -> None:
    fake = _FakeSpectrumModule(average_16bit_supported=True)
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))

    digitizer.connect()

    assert digitizer.hardware_info is not None
    assert digitizer.hardware_info.average_16bit_supported is True


def test_acquisition_methods_remain_explicitly_unimplemented() -> None:
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(_FakeSpectrumModule()))

    with pytest.raises(DigitizerError, match="configuration"):
        digitizer.configure(None)  # type: ignore[arg-type]
    with pytest.raises(DigitizerError, match="arm/start"):
        digitizer.arm()
    with pytest.raises(DigitizerError, match="readout"):
        digitizer.read_batch()


def test_configure_request_writes_raw_multi_registers() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    handle = digitizer.handle
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=4,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )

    plan = digitizer.configure_request(request)

    assert digitizer.acquisition_plan is plan
    assert fake.i32[fake.SPC_CARDMODE] == fake.SPC_REC_STD_MULTI
    assert fake.i64[fake.SPC_MEMSIZE] == 4096
    assert fake.i64[fake.SPC_POSTTRIGGER] == 992
    assert fake.i64[fake.SPC_SEGMENTSIZE] == 1024
    assert fake.i32[fake.SPC_TRIG_ORMASK] == fake.SPC_TMASK_SOFTWARE
    assert fake.calls.count(("open",)) == 1
    assert ("set_i32", handle, fake.SPC_CHENABLE, fake.CHANNEL0) in fake.calls


def test_configure_request_writes_32bit_average_registers() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=2048,
        pretrigger_samples=32,
        number_of_segments=2,
        averages_per_segment=10,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
        trigger_level_v=1.5,
    )

    plan = digitizer.configure_request(request)

    assert plan.is_fpga_sum is True
    assert fake.i32[fake.SPC_CARDMODE] == fake.SPC_REC_STD_AVERAGE
    assert fake.i32[fake.SPC_AVERAGES] == 10
    assert fake.i32[fake.SPC_TRIG_ORMASK] == fake.SPC_TMASK_EXT0
    assert fake.i32[fake.SPC_TRIG_EXT0_MODE] == fake.SPC_TM_POS
    assert fake.i32[fake.SPC_TRIG_EXT0_LEVEL0] == 1500


def test_block_average_software_trigger_rejected_before_writes() -> None:
    """Ensure configure_request validates trigger before touching card registers."""
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    writes_before = len([call for call in fake.calls if call[0].startswith("set_")])
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=2048,
        pretrigger_samples=32,
        averages_per_segment=2,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )

    with pytest.raises(ValueError, match="software trigger"):
        digitizer.configure_request(request)

    writes_after = len([call for call in fake.calls if call[0].startswith("set_")])
    assert writes_after == writes_before


def test_configure_request_writes_16bit_average_registers() -> None:
    fake = _FakeSpectrumModule(average_16bit_supported=True)
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_16BIT,
        sample_rate_hz=1.25e9,
        segment_samples=2048,
        pretrigger_samples=32,
        number_of_segments=2,
        averages_per_segment=10,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
        trigger_level_v=1.5,
        timeout_s=3.0,
    )

    plan = digitizer.configure_request(request)

    assert plan.is_fpga_sum is True
    assert fake.i32[fake.SPC_CARDMODE] == fake.SPC_REC_STD_AVERAGE_16BIT
    assert fake.i32[fake.SPC_AVERAGES] == 10
    assert fake.i32[fake.SPC_TRIG_ORMASK] == fake.SPC_TMASK_EXT0
    assert fake.i32[fake.SPC_TRIG_EXT0_MODE] == fake.SPC_TM_POS
    assert fake.i32[fake.SPC_TRIG_EXT0_LEVEL0] == 1500
    assert fake.i32[fake.SPC_TIMEOUT] == 3000


def test_missing_required_configuration_symbol_is_clear_error() -> None:
    fake = _FakeSpectrumModule()
    del fake.SPC_REC_STD_MULTI
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
    )

    with pytest.raises(DigitizerError, match="SPC_REC_STD_MULTI"):
        digitizer.configure_request(request)


def test_reconfigure_reuses_same_open_handle() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
    )

    digitizer.configure_request(request)
    digitizer.configure_request(request)

    assert fake.calls.count(("open",)) == 1


def test_acquire_configured_raw_multi_returns_native_shape() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=8 * 32,
        pretrigger_samples=32,
        number_of_segments=2,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    digitizer.configure_request(request)
    fake.next_transfer_data = np.arange(512, dtype=np.int8).reshape(2, 256)

    result = digitizer.acquire_configured()

    assert result.data.dtype == np.dtype(np.int8)
    assert result.data.shape == (2, 256)
    np.testing.assert_array_equal(result.data, fake.next_transfer_data)
    assert ("def_transfer", digitizer.handle, fake.SPCM_BUF_DATA, fake.SPCM_DIR_CARDTOPC, 0, 0, 512) in fake.calls
    assert ("command", digitizer.handle, fake.M2CMD_CARD_START | fake.M2CMD_CARD_ENABLETRIGGER | fake.M2CMD_DATA_STARTDMA) in fake.calls
    assert ("command", digitizer.handle, fake.M2CMD_CARD_WAITREADY | fake.M2CMD_DATA_WAITDMA) in fake.calls
    assert ("command", digitizer.handle, fake.M2CMD_CARD_STOP | fake.M2CMD_CARD_DISABLETRIGGER | fake.M2CMD_DATA_STOPDMA) in fake.calls


def test_acquire_configured_32bit_average_returns_sum_dtype() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=64,
        pretrigger_samples=32,
        averages_per_segment=4,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )
    digitizer.configure_request(request)
    fake.next_transfer_data = np.arange(64, dtype=np.int32).reshape(1, 64)

    result = digitizer.acquire_configured()

    assert result.data.dtype == np.dtype(np.int32)
    assert result.data.shape == (1, 64)
    assert result.plan.is_fpga_sum is True
    assert result.metadata["total_physical_shots"] == 4
    np.testing.assert_array_equal(result.data, fake.next_transfer_data)


def test_acquire_configured_16bit_average_returns_sum_dtype() -> None:
    fake = _FakeSpectrumModule(average_16bit_supported=True)
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_16BIT,
        sample_rate_hz=1.25e9,
        segment_samples=128,
        pretrigger_samples=32,
        averages_per_segment=4,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )
    digitizer.configure_request(request)
    fake.next_transfer_data = np.arange(128, dtype=np.int16).reshape(1, 128)

    result = digitizer.acquire_configured()

    assert result.data.dtype == np.dtype(np.int16)
    assert result.data.shape == (1, 128)
    assert result.plan.is_fpga_sum is True
    assert result.metadata["total_physical_shots"] == 4
    np.testing.assert_array_equal(result.data, fake.next_transfer_data)


def test_split_acquisition_prepares_starts_then_waits_in_order() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=256,
        pretrigger_samples=32,
        number_of_segments=2,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    digitizer.configure_request(request)
    fake.next_transfer_data = np.arange(512, dtype=np.int8).reshape(2, 256)

    plan = digitizer.prepare_configured_acquisition()
    calls_after_prepare = list(fake.calls)
    digitizer.start_prepared_acquisition()
    calls_after_start = list(fake.calls)
    result = digitizer.wait_for_prepared_result()
    digitizer.stop()

    assert plan.output_shape == (2, 256)
    assert result.data.shape == (2, 256)
    def_index = calls_after_prepare.index(("def_transfer", digitizer.handle, fake.SPCM_BUF_DATA, fake.SPCM_DIR_CARDTOPC, 0, 0, 512))
    start_call = ("command", digitizer.handle, fake.M2CMD_CARD_START | fake.M2CMD_CARD_ENABLETRIGGER | fake.M2CMD_DATA_STARTDMA)
    wait_call = ("command", digitizer.handle, fake.M2CMD_CARD_WAITREADY | fake.M2CMD_DATA_WAITDMA)
    assert start_call not in calls_after_prepare
    assert wait_call not in calls_after_prepare
    assert start_call in calls_after_start
    assert wait_call not in calls_after_start
    assert fake.calls.index(start_call) > def_index
    assert fake.calls.index(wait_call) > fake.calls.index(start_call)


def test_split_start_requires_prepare() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=256,
        pretrigger_samples=32,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    digitizer.configure_request(request)

    with pytest.raises(DigitizerError, match="not prepared"):
        digitizer.start_prepared_acquisition()


def test_acquire_configured_requires_configuration() -> None:
    fake = _FakeSpectrumModule()
    digitizer = SpectrumDigitizer(api=SpectrumDriverApi(fake))
    digitizer.connect()

    with pytest.raises(DigitizerError, match="not configured"):
        digitizer.acquire_configured()


class _FakeSpectrumModule:
    ERR_OK = 0
    ERRORTEXTLEN = 1024

    def __init__(self, include_optional_registers: bool = True, average_16bit_supported: bool = False) -> None:
        self.handle = object()
        self.i32: dict[int, int] = {}
        self.i64: dict[int, int] = {}
        self.calls: list[tuple] = []
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
        self.next_transfer_data: np.ndarray | None = None
        self.transfer_buffer = None
        self.transfer_length = 0
        if include_optional_registers:
            self.SPC_PCITYP = 1
            self.SPC_PCISERIALNO = 2
            self.SPC_FNCTYPE = 3
            self.SPC_MIINST_MAXADCVALUE = 4
        if average_16bit_supported:
            self.SPC_REC_STD_AVERAGE_16BIT = 100

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
            raise AssertionError("fake transfer data length does not match configured transfer length")
        self.transfer_buffer.raw = payload
