"""Tests for RealAcquisitionWorker and RealShotAnalysisWorker with fake driver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtCore

from pytof_new.acquisition.real_worker import RealAcquisitionWorker, RealShotAnalysisWorker
from pytof_new.config.models import DigitizerConfig, ProcessingConfig, RunConfig
from pytof_new.hardware.acquisition_planner import plan_acquisition
from pytof_new.hardware.spectrum_driver import SpectrumDriverApi
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionResult,
    SpectrumAcquisitionRequest,
    SpectrumTriggerSource,
)
from pytof_new.hardware.spectrum_service import SpectrumAcquisitionService


class _FakeSpectrumModule:
    """Minimal fake driver for worker tests."""

    ERR_OK = 0
    ERRORTEXTLEN = 1024

    def __init__(self) -> None:
        self.handle = object()
        self.i32: dict[int, int] = {}
        self.i64: dict[int, int] = {}
        self.next_transfer_data: np.ndarray | None = None
        self.transfer_buffer = None
        self.transfer_length = 0
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
        self.SPC_PCITYP = 1
        self.SPC_PCISERIALNO = 2
        self.SPC_FNCTYPE = 3
        self.SPC_MIINST_MAXADCVALUE = 4

    def spcm_hOpen(self, _device):
        return self.handle

    def spcm_vClose(self, _handle):
        pass

    def spcm_dwGetParam_i32(self, handle, register, value_ptr):
        value_ptr._obj.value = self.i32.get(register, 0)
        return self.ERR_OK

    def spcm_dwSetParam_i32(self, handle, register, value):
        if register == self.SPC_M2CMD and value & self.M2CMD_CARD_WAITREADY:
            self._fill_transfer_buffer()
        self.i32[register] = value
        return self.ERR_OK

    def spcm_dwSetParam_i64(self, handle, register, value):
        self.i64[register] = value
        return self.ERR_OK

    def spcm_dwDefTransfer_i64(self, handle, buffer_kind, direction, notify_size, buffer, offset, length):
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


class _FakeCoordinator(QtCore.QObject):
    result_ready = QtCore.Signal(object)
    error_occurred = QtCore.Signal(str)
    log_message = QtCore.Signal(str)

    def __init__(self, result: SpectrumAcquisitionResult) -> None:
        super().__init__()
        self.result = result
        self.start_counts: list[int] = []

    @QtCore.Slot(int)
    def start_batch(self, expected_count: int) -> None:
        self.start_counts.append(expected_count)
        self.result_ready.emit(self.result)

    @QtCore.Slot()
    def request_stop(self) -> None:
        self.error_occurred.emit("stopped")


@pytest.fixture()
def fake():
    return _FakeSpectrumModule()


@pytest.fixture()
def config():
    return RunConfig(
        digitizer=DigitizerConfig(sample_rate_hz=1.25e9, segment_samples=1024, pretrigger_samples=32, number_of_segments=1),
        processing=ProcessingConfig(subtract_baseline=False, detector_polarity=1, adc_full_scale_counts=127),
    )


def _prepare_service(fake, config: RunConfig) -> SpectrumAcquisitionService:
    """Return a service in CONFIGURED state with a fake-driver backend."""
    api = SpectrumDriverApi(fake)
    svc = SpectrumAcquisitionService(api=api)
    svc.connect_card()
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=config.digitizer.sample_rate_hz,
        segment_samples=config.digitizer.segment_samples,
        pretrigger_samples=config.digitizer.pretrigger_samples,
        number_of_segments=config.digitizer.number_of_segments,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )
    svc.configure(request)
    return svc


def test_real_worker_finite_acquire_one_batch(fake, config) -> None:
    fake.next_transfer_data = np.ones((1, 1024), dtype=np.int8) * 50
    svc = _prepare_service(fake, config)
    worker = RealAcquisitionWorker(svc, config, continuous=False)

    batches: list[object] = []
    worker.batch_ready.connect(lambda b, p: batches.append(b))
    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert len(batches) == 1
    assert finished == [True]


def test_real_worker_uses_coordinator_for_real_batch(fake, config) -> None:
    svc = _prepare_service(fake, config)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=config.digitizer.sample_rate_hz,
        segment_samples=config.digitizer.segment_samples,
        pretrigger_samples=config.digitizer.pretrigger_samples,
        number_of_segments=1,
        averages_per_segment=8,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )
    svc.configure(request)
    run_plan = plan_acquisition(
        acquisition_workflow="live_averaged",
        tof_window_us=10.0,
        total_shots=8,
        pretrigger_samples=config.digitizer.pretrigger_samples,
        averages_per_segment=8,
        advanced_mode=True,
        manual_sample_rate_hz=config.digitizer.sample_rate_hz,
        manual_accumulator_mode="32bit",
        manual_fpga_sums_per_batch=1,
        manual_segment_samples=config.digitizer.segment_samples,
        trigger_source="external0",
    )
    result = SpectrumAcquisitionResult(
        data=np.ones((1, config.digitizer.segment_samples), dtype=np.int32) * 400,
        plan=svc._digitizer.acquisition_plan,
        metadata={"bme_actual_trigger_count": 8},
    )
    coordinator = _FakeCoordinator(result)
    worker = RealAcquisitionWorker(svc, config, run_plan, coordinator=coordinator, continuous=False)

    batches: list[object] = []
    worker.batch_ready.connect(lambda b, p: batches.append(b))
    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert coordinator.start_counts == [8]
    assert len(batches) == 1
    assert finished == [True]


def test_real_worker_continuous_acquires_two_then_stop(fake, config) -> None:
    fake.next_transfer_data = np.ones((1, 1024), dtype=np.int8) * 50
    svc = _prepare_service(fake, config)
    worker = RealAcquisitionWorker(svc, config, continuous=True)

    batches: list[object] = []
    worker.batch_ready.connect(lambda b, p: batches.append(b))
    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))

    worker.run()
    assert len(batches) == 1  # first batch is emitted immediately
    assert not finished  # continuous means worker is still waiting

    fake.next_transfer_data = np.ones((1, 1024), dtype=np.int8) * 60
    worker._request_next()
    assert len(batches) == 2

    worker.request_stop()
    # stop triggers abort; next acquire will not start
    assert finished == [True]


def test_real_worker_stop_before_acquire(fake, config) -> None:
    svc = _prepare_service(fake, config)
    worker = RealAcquisitionWorker(svc, config, continuous=True)

    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))

    worker.request_stop()
    worker.run()

    assert finished == [True]


def test_shot_analysis_acquires_n_records(fake, config) -> None:
    samples = config.digitizer.segment_samples
    fake.next_transfer_data = np.ones((1, samples), dtype=np.int8) * 20
    svc = _prepare_service(fake, config)
    config_with_cal = replace(config, processing=replace(config.processing, mass_calibration_enabled=True, mass_calibration=(3.5e-6, 0.0129, 6.27)))

    worker = RealShotAnalysisWorker(
        svc, config_with_cal, record_count=3, delay_s=0.0, max_lag_s=1e-7, min_mz=0.0,
    )

    results: list[object] = []
    worker.result_ready.connect(results.append)
    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert len(results) == 1
    assert finished == [True]


def test_shot_analysis_can_use_coordinator(fake, config) -> None:
    samples = config.digitizer.segment_samples
    svc = _prepare_service(fake, config)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=config.digitizer.sample_rate_hz,
        segment_samples=samples,
        pretrigger_samples=config.digitizer.pretrigger_samples,
        number_of_segments=1,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )
    svc.configure(request)
    result = SpectrumAcquisitionResult(
        data=np.ones((1, samples), dtype=np.int8) * 20,
        plan=svc._digitizer.acquisition_plan,
        metadata={"bme_actual_trigger_count": 1},
    )
    coordinator = _FakeCoordinator(result)
    config_with_cal = replace(config, processing=replace(config.processing, mass_calibration_enabled=True, mass_calibration=(3.5e-6, 0.0129, 6.27)))
    worker = RealShotAnalysisWorker(
        svc,
        config_with_cal,
        record_count=3,
        delay_s=0.0,
        max_lag_s=1e-7,
        min_mz=0.0,
        coordinator=coordinator,
        request=request,
    )

    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))
    worker.run()

    assert coordinator.start_counts == [1, 1, 1]
    assert finished == [True]
