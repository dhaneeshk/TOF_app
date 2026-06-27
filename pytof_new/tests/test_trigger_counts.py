import pytest

from pytof_new.acquisition.trigger_counts import required_bme_triggers_for_plan, required_bme_triggers_for_request
from pytof_new.hardware.spectrum_limits import plan_spectrum_acquisition
from pytof_new.hardware.spectrum_models import SpectrumAcquisitionMode, SpectrumAcquisitionRequest, SpectrumHardwareInfo, SpectrumTriggerSource


def test_raw_multi_bme_triggers_equal_raw_segments() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=37,
        averages_per_segment=1,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )

    assert required_bme_triggers_for_request(request) == 37


def test_32bit_block_average_bme_triggers_equal_n_times_k() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=45,
        averages_per_segment=100,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )

    assert required_bme_triggers_for_request(request) == 4500


def test_16bit_block_average_bme_triggers_equal_n_times_k() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_16BIT,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=12,
        averages_per_segment=8,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )

    assert required_bme_triggers_for_request(request) == 96


def test_final_partial_raw_batch_uses_actual_final_request_count() -> None:
    final_request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=13,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )

    assert required_bme_triggers_for_request(final_request) == 13


def test_plan_helper_uses_validated_plan_request() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.AVERAGE_32BIT,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=3,
        averages_per_segment=20,
        trigger_source=SpectrumTriggerSource.EXTERNAL0,
    )
    plan = plan_spectrum_acquisition(request, SpectrumHardwareInfo(average_16bit_supported=True))

    assert required_bme_triggers_for_plan(plan) == 60
    assert plan.metadata["total_physical_shots"] == 60


def test_invalid_request_counts_are_rejected() -> None:
    request = SpectrumAcquisitionRequest(
        mode=SpectrumAcquisitionMode.RAW_MULTI,
        sample_rate_hz=1.25e9,
        segment_samples=1024,
        pretrigger_samples=32,
        number_of_segments=0,
    )

    with pytest.raises(ValueError, match="number_of_segments"):
        required_bme_triggers_for_request(request)
