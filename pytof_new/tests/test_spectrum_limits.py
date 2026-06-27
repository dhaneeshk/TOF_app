import numpy as np
import pytest

from pytof_new.hardware.spectrum_limits import plan_spectrum_acquisition
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionRequest,
    SpectrumHardwareInfo,
    SpectrumTriggerSource,
)


def test_raw_multi_plan_uses_native_8bit_shape() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=7,
        averages_per_segment=1,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )

    plan = plan_spectrum_acquisition(request)

    assert plan.dtype == np.dtype(np.int8)
    assert plan.output_shape == (7, 1024)
    assert plan.transfer_bytes == 7 * 1024
    assert plan.is_fpga_sum is False
    assert plan.metadata["total_physical_shots"] == 7


def test_average_32bit_plan_tracks_fpga_sum_metadata() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=65_536,
        pretrigger_samples=32,
        number_of_segments=3,
        averages_per_segment=100,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )

    plan = plan_spectrum_acquisition(request)

    assert plan.dtype == np.dtype(np.int32)
    assert plan.output_shape == (3, 65_536)
    assert plan.transfer_bytes == 3 * 65_536 * 4
    assert plan.is_fpga_sum is True
    assert plan.physical_shots_per_output_segment == 100
    assert plan.metadata["total_physical_shots"] == 300


def test_block_average_rejects_software_trigger() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        averages_per_segment=2,
        trigger_source=SpectrumTriggerSource.SOFTWARE,
    )

    with pytest.raises(ValueError, match="software trigger"):
        plan_spectrum_acquisition(request)


def test_average_16bit_requires_detected_support() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_16BIT,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        averages_per_segment=2,
    )
    hardware = SpectrumHardwareInfo(average_16bit_supported=False)

    with pytest.raises(ValueError, match="16-bit block average"):
        plan_spectrum_acquisition(request, hardware)


def test_average_16bit_plan_when_support_is_detected() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_16BIT,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        averages_per_segment=2,
    )
    hardware = SpectrumHardwareInfo(average_16bit_supported=True)

    plan = plan_spectrum_acquisition(request, hardware)

    assert plan.dtype == np.dtype(np.int16)
    assert plan.output_shape == (1, 1024)
    assert plan.transfer_bytes == 2048


def test_plan_rejects_onboard_memory_overflow() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=2,
    )
    hardware = SpectrumHardwareInfo(onboard_memory_bytes=1024)

    with pytest.raises(ValueError, match="exceeding onboard memory"):
        plan_spectrum_acquisition(request, hardware)
