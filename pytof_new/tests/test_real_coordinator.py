import numpy as np
from PySide6 import QtCore

from pytof_new.acquisition.real_coordinator import RealBatchCoordinator, RealBatchCoordinatorState
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionPlan,
    SpectrumAcquisitionRequest,
    SpectrumAcquisitionResult,
    SpectrumTriggerSource,
)


def test_coordinator_runs_safe_batch_order_and_emits_metadata() -> None:
    order: list[str] = []
    result = _spectrum_result(records=2)
    spectrum = _FakeSpectrumService(order, result=result)
    bme = _FakeBMEService(order, trigger_count=4500, status=0x44)
    coordinator = RealBatchCoordinator(spectrum, bme)
    results: list[object] = []
    errors: list[str] = []
    coordinator.result_ready.connect(results.append)
    coordinator.error_occurred.connect(errors.append)

    coordinator.start_batch(4500)

    assert errors == []
    assert coordinator.state == RealBatchCoordinatorState.FINISHED
    assert order == [
        "bme.stop",
        "bme.arm:4500",
        "spectrum.prepare",
        "spectrum.start_prepared",
        "bme.start",
        "spectrum.wait_result",
        "bme.read_trigger_count",
        "bme.read_status",
        "bme.stop",
    ]
    assert len(results) == 1
    metadata = results[0].metadata
    assert metadata["bme_expected_trigger_count"] == 4500
    assert metadata["bme_actual_trigger_count"] == 4500
    assert metadata["bme_status"] == 0x44
    assert metadata["spectrum_expected_records"] == 2
    assert metadata["spectrum_actual_records"] == 2


def test_coordinator_can_derive_bme_count_from_spectrum_plan() -> None:
    order: list[str] = []
    result = _spectrum_result(records=3, averages_per_segment=20, mode=SpectrumAcquisitionMode.AVERAGE_32BIT)
    coordinator = RealBatchCoordinator(_FakeSpectrumService(order, result=result), _FakeBMEService(order, trigger_count=60))
    results: list[object] = []
    coordinator.result_ready.connect(results.append)

    coordinator.start_plan_batch(result.plan)

    assert coordinator.state == RealBatchCoordinatorState.FINISHED
    assert order[1] == "bme.arm:60"
    assert results[0].metadata["bme_expected_trigger_count"] == 60


def test_coordinator_reports_bme_count_mismatch() -> None:
    order: list[str] = []
    coordinator = RealBatchCoordinator(_FakeSpectrumService(order, result=_spectrum_result(records=2)), _FakeBMEService(order, trigger_count=4499))
    errors: list[str] = []
    results: list[object] = []
    coordinator.error_occurred.connect(errors.append)
    coordinator.result_ready.connect(results.append)

    coordinator.start_batch(4500)

    assert results == []
    assert coordinator.state == RealBatchCoordinatorState.ERROR
    assert errors and "BME trigger count mismatch" in errors[-1]


def test_coordinator_reports_spectrum_record_mismatch() -> None:
    order: list[str] = []
    coordinator = RealBatchCoordinator(
        _FakeSpectrumService(order, result=_spectrum_result(records=1, expected_records=2)),
        _FakeBMEService(order, trigger_count=4500),
    )
    errors: list[str] = []
    coordinator.error_occurred.connect(errors.append)

    coordinator.start_batch(4500)

    assert coordinator.state == RealBatchCoordinatorState.ERROR
    assert errors and "Spectrum record count mismatch" in errors[-1]


def test_bme_error_aborts_spectrum() -> None:
    order: list[str] = []
    spectrum = _FakeSpectrumService(order, result=_spectrum_result(records=2))
    bme = _FakeBMEService(order, trigger_count=4500, fail_start=True)
    coordinator = RealBatchCoordinator(spectrum, bme)
    errors: list[str] = []
    coordinator.error_occurred.connect(errors.append)

    coordinator.start_batch(4500)

    assert coordinator.state == RealBatchCoordinatorState.ERROR
    assert "spectrum.abort" in order
    assert errors and "BME service error" in errors[-1]


def test_spectrum_error_emergency_stops_bme() -> None:
    order: list[str] = []
    spectrum = _FakeSpectrumService(order, result=_spectrum_result(records=2), fail_wait=True)
    bme = _FakeBMEService(order, trigger_count=4500)
    coordinator = RealBatchCoordinator(spectrum, bme)
    errors: list[str] = []
    coordinator.error_occurred.connect(errors.append)

    coordinator.start_batch(4500)

    assert coordinator.state == RealBatchCoordinatorState.ERROR
    assert "bme.emergency_stop" in order
    assert errors and "Spectrum service error" in errors[-1]


def test_coordinator_rejects_invalid_trigger_count() -> None:
    order: list[str] = []
    coordinator = RealBatchCoordinator(_FakeSpectrumService(order, result=_spectrum_result(records=1)), _FakeBMEService(order))
    errors: list[str] = []
    coordinator.error_occurred.connect(errors.append)

    coordinator.start_batch(0)

    assert coordinator.state == RealBatchCoordinatorState.ERROR
    assert errors and "positive" in errors[-1]
    assert order == []


class _FakeSpectrumService(QtCore.QObject):
    prepared = QtCore.Signal(object)
    armed = QtCore.Signal()
    result_ready = QtCore.Signal(object)
    error_occurred = QtCore.Signal(str)

    def __init__(self, order: list[str], *, result: SpectrumAcquisitionResult, fail_wait: bool = False) -> None:
        super().__init__()
        self.order = order
        self.result = result
        self.fail_wait = fail_wait

    @QtCore.Slot()
    def prepare(self) -> None:
        self.order.append("spectrum.prepare")
        self.prepared.emit(self.result.plan)

    @QtCore.Slot()
    def start_prepared(self) -> None:
        self.order.append("spectrum.start_prepared")
        self.armed.emit()

    @QtCore.Slot()
    def wait_result(self) -> None:
        self.order.append("spectrum.wait_result")
        if self.fail_wait:
            self.error_occurred.emit("wait failed")
            return
        self.result_ready.emit(self.result)

    @QtCore.Slot()
    def abort(self) -> None:
        self.order.append("spectrum.abort")


class _FakeBMEService(QtCore.QObject):
    stopped = QtCore.Signal()
    armed = QtCore.Signal(int)
    started = QtCore.Signal()
    trigger_count_ready = QtCore.Signal(int)
    status_ready = QtCore.Signal(int)
    error_occurred = QtCore.Signal(str)

    def __init__(self, order: list[str], *, trigger_count: int = 1, status: int = 0x40, fail_start: bool = False) -> None:
        super().__init__()
        self.order = order
        self.trigger_count = trigger_count
        self.status = status
        self.fail_start = fail_start

    @QtCore.Slot()
    def stop(self) -> None:
        self.order.append("bme.stop")
        self.stopped.emit()

    @QtCore.Slot(int)
    def arm(self, expected_count: int) -> None:
        self.order.append(f"bme.arm:{expected_count}")
        self.armed.emit(expected_count)

    @QtCore.Slot()
    def start(self) -> None:
        self.order.append("bme.start")
        if self.fail_start:
            self.error_occurred.emit("start failed")
            return
        self.started.emit()

    @QtCore.Slot()
    def read_trigger_count(self) -> None:
        self.order.append("bme.read_trigger_count")
        self.trigger_count_ready.emit(self.trigger_count)

    @QtCore.Slot()
    def read_status(self) -> None:
        self.order.append("bme.read_status")
        self.status_ready.emit(self.status)

    @QtCore.Slot()
    def emergency_stop(self) -> None:
        self.order.append("bme.emergency_stop")


def _spectrum_result(
    *,
    records: int,
    expected_records: int | None = None,
    averages_per_segment: int = 1,
    mode: SpectrumAcquisitionMode = SpectrumAcquisitionMode.RAW_MULTI,
) -> SpectrumAcquisitionResult:
    expected = expected_records if expected_records is not None else records
    request = SpectrumAcquisitionRequest(
        mode=mode,
        sample_rate_hz=1.25e9,
        segment_samples=256,
        pretrigger_samples=32,
        number_of_segments=expected,
        averages_per_segment=averages_per_segment,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )
    plan = SpectrumAcquisitionPlan(
        request=request,
        dtype=np.dtype(np.int8),
        output_shape=(expected, 256),
        transfer_bytes=expected * 256,
        physical_shots_per_output_segment=averages_per_segment,
        is_fpga_sum=mode != SpectrumAcquisitionMode.RAW_MULTI,
        metadata={"total_physical_shots": expected * averages_per_segment},
    )
    return SpectrumAcquisitionResult(data=np.zeros((records, 256), dtype=np.int8), plan=plan, metadata=dict(plan.metadata))
