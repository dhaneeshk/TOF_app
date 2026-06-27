from pytof_new.acquisition.controller import AcquisitionController
from pytof_new.acquisition.state import AcquisitionState, AcquisitionStateMachine
from pytof_new.config.models import DigitizerConfig, RunConfig
from pytof_new.exceptions import InvalidStateTransitionError
from pytof_new.hardware.mock_delay_generator import MockDelayGenerator
from pytof_new.hardware.mock_digitizer import MockDigitizer, MockDigitizerProfile, SyntheticPeak


def test_state_machine_rejects_invalid_transition() -> None:
    machine = AcquisitionStateMachine()
    try:
        machine.transition_to(AcquisitionState.ACQUIRING)
    except InvalidStateTransitionError:
        pass
    else:
        raise AssertionError("invalid transition was accepted")


def test_mock_controller_acquires_batch_and_stops() -> None:
    config = RunConfig(digitizer=DigitizerConfig(number_of_segments=5, segment_samples=1024, pretrigger_samples=32))
    controller = AcquisitionController(MockDigitizer(), MockDelayGenerator())
    controller.connect_hardware()
    batch = controller.acquire_batch(config)
    assert batch.raw_adc.shape == (5, 1024)
    assert controller.state == AcquisitionState.CONFIGURED
    controller.disconnect_hardware()
    assert controller.state == AcquisitionState.DISCONNECTED


def test_mock_resolving_power_controls_peak_width() -> None:
    narrow = _mock_record(resolving_power=5000.0)
    broad = _mock_record(resolving_power=500.0)
    assert _half_max_width_samples(broad) > _half_max_width_samples(narrow)


def test_mock_default_t0_artifact_is_flat() -> None:
    trace = _artifact_record(MockDigitizerProfile(peaks=(), noise_rms_v=0.0, baseline_offset_v=0.0, baseline_drift_v=0.0))
    assert trace.tolist() == [0] * len(trace)


def test_mock_enabled_ringing_adds_phased_t0_artifact() -> None:
    phase_zero = _artifact_record(
        MockDigitizerProfile(
            peaks=(),
            noise_rms_v=0.0,
            baseline_offset_v=0.0,
            baseline_drift_v=0.0,
            ringing_amplitude_v=0.1,
            ringing_frequency_hz=10e6,
            ringing_decay_s=1e-6,
            ringing_phase_rad=0.0,
        )
    )
    phase_quarter = _artifact_record(
        MockDigitizerProfile(
            peaks=(),
            noise_rms_v=0.0,
            baseline_offset_v=0.0,
            baseline_drift_v=0.0,
            ringing_amplitude_v=0.1,
            ringing_frequency_hz=10e6,
            ringing_decay_s=1e-6,
            ringing_phase_rad=1.5707963267948966,
        )
    )
    assert phase_zero[0] > 0
    assert abs(int(phase_quarter[0])) < abs(int(phase_zero[0]))


def test_mock_ringing_can_follow_timing_jitter() -> None:
    fixed = _artifact_record(
        MockDigitizerProfile(
            peaks=(),
            noise_rms_v=0.0,
            baseline_offset_v=0.0,
            baseline_drift_v=0.0,
            ringing_amplitude_v=0.1,
            ringing_frequency_hz=20e6,
            ringing_decay_s=1e-6,
            timing_jitter_s=10e-9,
            random_seed=1,
        )
    )
    jittered = _artifact_record(
        MockDigitizerProfile(
            peaks=(),
            noise_rms_v=0.0,
            baseline_offset_v=0.0,
            baseline_drift_v=0.0,
            ringing_amplitude_v=0.1,
            ringing_frequency_hz=20e6,
            ringing_decay_s=1e-6,
            timing_jitter_s=10e-9,
            ringing_follows_timing_jitter=True,
            random_seed=1,
        )
    )
    assert fixed.tolist() != jittered.tolist()


def _mock_record(resolving_power: float):
    profile = MockDigitizerProfile(
        peaks=(SyntheticPeak(position_s=10.0e-6, amplitude_v=0.2),),
        noise_rms_v=0.0,
        baseline_offset_v=0.0,
        baseline_drift_v=0.0,
        pickup_amplitude_v=0.0,
        ringing_amplitude_v=0.0,
        timing_jitter_s=0.0,
        resolving_power=resolving_power,
        random_seed=1,
    )
    digitizer = MockDigitizer(profile)
    config = DigitizerConfig(
        sample_rate_hz=1.0e9,
        input_range_v=1.0,
        segment_samples=32768,
        pretrigger_samples=0,
        number_of_segments=1,
        hardware_averages_per_record=1,
    )
    digitizer.connect()
    digitizer.configure(config)
    digitizer.arm()
    return digitizer.read_batch()[0]


def _artifact_record(profile: MockDigitizerProfile):
    digitizer = MockDigitizer(profile)
    config = DigitizerConfig(
        sample_rate_hz=1.0e9,
        input_range_v=1.0,
        segment_samples=128,
        pretrigger_samples=0,
        number_of_segments=1,
        hardware_averages_per_record=1,
    )
    digitizer.connect()
    digitizer.configure(config)
    digitizer.arm()
    return digitizer.read_batch()[0]


def _half_max_width_samples(trace):
    half_max = trace.max() / 2
    indices = (trace >= half_max).nonzero()[0]
    return int(indices[-1] - indices[0] + 1)
