import numpy as np
import pytest

from pytof_new.acquisition.models import AcquisitionBatch
from pytof_new.hardware.spectrum_converter import spectrum_result_to_batch
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionPlan,
    SpectrumAcquisitionRequest,
    SpectrumAcquisitionResult,
    SpectrumTriggerSource,
)


def _make_fake_result(
    mode: SpectrumAcquisitionMode,
    data: np.ndarray,
    shots_per_segment: int,
    segments: int,
    is_fpga: bool,
) -> SpectrumAcquisitionResult:
    samples = data.shape[1]
    request = SpectrumAcquisitionRequest(
        mode=mode,
        sample_rate_hz=1.25e9,
        segment_samples=samples,
        pretrigger_samples=32,
        number_of_segments=segments,
        averages_per_segment=shots_per_segment if is_fpga else 1,
    )
    plan = SpectrumAcquisitionPlan(
        request=request,
        dtype=data.dtype,
        output_shape=data.shape,
        transfer_bytes=int(np.prod(data.shape) * data.dtype.itemsize),
        physical_shots_per_output_segment=shots_per_segment,
        is_fpga_sum=is_fpga,
        metadata={"total_physical_shots": segments * shots_per_segment},
    )
    return SpectrumAcquisitionResult(data=data, plan=plan, metadata=dict(plan.metadata))


def test_raw_multi_passthrough() -> None:
    raw = np.arange(256, dtype=np.int8).reshape(2, 128)
    result = _make_fake_result(SpectrumAcquisitionMode.RAW_MULTI, raw, shots_per_segment=1, segments=2, is_fpga=False)

    batch = spectrum_result_to_batch(result, sample_rate_hz=1.25e9, pretrigger_samples=0)

    assert batch.raw_adc.dtype == np.dtype(np.int8)
    assert batch.raw_adc.shape == (2, 128)
    assert batch.record_mode == "raw_segments"
    assert batch.hardware_averages_per_record == 1
    assert batch.metadata["total_physical_shots"] == 2
    assert batch.metadata["is_fpga_sum"] is False
    np.testing.assert_array_equal(batch.raw_adc, raw)


def test_32bit_fpga_average_normalises_by_shots() -> None:
    shots = 10
    segments = 3
    native_data = (np.arange(192, dtype=np.int32).reshape(segments, 64)) * shots
    result = _make_fake_result(
        SpectrumAcquisitionMode.AVERAGE_32BIT, native_data, shots_per_segment=shots, segments=segments, is_fpga=True
    )

    batch = spectrum_result_to_batch(result, sample_rate_hz=1.25e9, pretrigger_samples=32)

    assert batch.raw_adc.dtype == np.dtype(np.float32)
    assert batch.raw_adc.shape == (segments, 64)
    assert batch.record_mode == "fpga_average"
    assert batch.hardware_averages_per_record == shots
    assert batch.metadata["total_physical_shots"] == segments * shots
    assert batch.metadata["is_fpga_sum"] is True
    expected = np.arange(192, dtype=np.float32).reshape(segments, 64)
    np.testing.assert_allclose(batch.raw_adc, expected, rtol=1e-6)


def test_16bit_fpga_average_normalises_by_shots() -> None:
    shots = 20
    segments = 1
    native_data = (np.arange(64, dtype=np.int16) * shots).reshape(segments, 64)
    result = _make_fake_result(
        SpectrumAcquisitionMode.AVERAGE_16BIT, native_data, shots_per_segment=shots, segments=segments, is_fpga=True
    )

    batch = spectrum_result_to_batch(result, sample_rate_hz=1.25e9, pretrigger_samples=32)

    assert batch.raw_adc.dtype == np.dtype(np.float32)
    assert batch.record_mode == "fpga_average"
    assert batch.hardware_averages_per_record == shots
    expected = np.arange(64, dtype=np.float32).reshape(1, 64)
    np.testing.assert_allclose(batch.raw_adc, expected, rtol=1e-6)


def test_converter_rejects_zero_shots_in_fpga_mode() -> None:
    data = np.zeros((1, 16), dtype=np.int32)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=16,
        pretrigger_samples=0,
        averages_per_segment=0,
    )
    plan = SpectrumAcquisitionPlan(
        request=request,
        dtype=np.dtype(np.int32),
        output_shape=(1, 16),
        transfer_bytes=64,
        physical_shots_per_output_segment=0,
        is_fpga_sum=True,
        metadata={"total_physical_shots": 0},
    )
    result = SpectrumAcquisitionResult(data=data, plan=plan, metadata={})

    with pytest.raises(ValueError, match="physical_shots_per_output_segment must be positive"):
        spectrum_result_to_batch(result, sample_rate_hz=1.25e9, pretrigger_samples=0)


def test_1d_array_is_promoted_to_2d() -> None:
    raw = np.arange(128, dtype=np.int8)
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=128,
        pretrigger_samples=0,
        number_of_segments=1,
        averages_per_segment=1,
    )
    plan = SpectrumAcquisitionPlan(
        request=request,
        dtype=np.dtype(np.int8),
        output_shape=(128,),
        transfer_bytes=128,
        physical_shots_per_output_segment=1,
        is_fpga_sum=False,
    )
    result = SpectrumAcquisitionResult(data=raw, plan=plan, metadata={})

    batch = spectrum_result_to_batch(result, sample_rate_hz=1.25e9, pretrigger_samples=0)

    assert batch.raw_adc.ndim == 2
    assert batch.raw_adc.shape == (1, 128)


def test_first_trigger_index_and_sample_rate_carried_through() -> None:
    raw = np.ones((2, 64), dtype=np.int8)
    result = _make_fake_result(SpectrumAcquisitionMode.RAW_MULTI, raw, shots_per_segment=1, segments=2, is_fpga=False)

    batch = spectrum_result_to_batch(
        result, sample_rate_hz=500e6, pretrigger_samples=16, first_trigger_index=42
    )

    assert batch.sample_rate_hz == 500e6
    assert batch.pretrigger_samples == 16
    assert batch.first_trigger_index == 42
