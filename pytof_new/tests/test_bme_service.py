import pytest

from pytof_new.config.models import BMEConfig
from pytof_new.exceptions import DelayGeneratorError
from pytof_new.hardware.bme_delay_generator import BMEConnectionInfo
from pytof_new.hardware.bme_service import BMEDelayGeneratorService, BMEServiceState


def test_service_starts_idle() -> None:
    service = BMEDelayGeneratorService(delay_generator=_FakeDelayGenerator())

    assert service.state == BMEServiceState.IDLE


def test_connect_transitions_to_connected_and_emits_info() -> None:
    fake = _FakeDelayGenerator()
    service = BMEDelayGeneratorService(delay_generator=fake)
    states: list[str] = []
    infos: list[object] = []
    service.state_changed.connect(states.append)
    service.hardware_info_ready.connect(infos.append)

    service.connect_card()

    assert service.state == BMEServiceState.CONNECTED
    assert states == ["connected"]
    assert infos == [fake.info]
    assert fake.calls == ["connect"]


def test_connect_failure_transitions_to_error() -> None:
    service = BMEDelayGeneratorService(delay_generator=_FakeDelayGenerator(fail_on="connect"))
    errors: list[str] = []
    service.error_occurred.connect(errors.append)

    service.connect_card()

    assert service.state == BMEServiceState.ERROR
    assert errors and "connect failed" in errors[0]


def test_configure_transitions_to_configured() -> None:
    fake = _FakeDelayGenerator()
    service = BMEDelayGeneratorService(delay_generator=fake)
    configured: list[bool] = []
    service.configured.connect(lambda: configured.append(True))
    service.connect_card()

    config = BMEConfig()
    service.configure(config)

    assert service.state == BMEServiceState.CONFIGURED
    assert configured == [True]
    assert fake.config is config


def test_configure_wrong_state_emits_error() -> None:
    service = BMEDelayGeneratorService(delay_generator=_FakeDelayGenerator())
    errors: list[str] = []
    service.error_occurred.connect(errors.append)

    service.configure(BMEConfig())

    assert service.state == BMEServiceState.IDLE
    assert "configure cannot be called" in errors[0]


def test_arm_resets_count_and_emits_expected_count() -> None:
    fake = _FakeDelayGenerator()
    service = BMEDelayGeneratorService(delay_generator=fake)
    armed: list[int] = []
    service.armed.connect(armed.append)
    service.connect_card()

    service.arm(4500)

    assert service.state == BMEServiceState.ARMED
    assert fake.expected_trigger_count == 4500
    assert armed == [4500]


def test_start_failure_transitions_to_error() -> None:
    fake = _FakeDelayGenerator(fail_on="start")
    service = BMEDelayGeneratorService(delay_generator=fake)
    errors: list[str] = []
    service.error_occurred.connect(errors.append)
    service.connect_card()
    service.arm(1)

    service.start()

    assert service.state == BMEServiceState.ERROR
    assert errors and "start failed" in errors[-1]


def test_start_success_transitions_to_running() -> None:
    fake = _FakeDelayGenerator()
    service = BMEDelayGeneratorService(delay_generator=fake)
    started: list[bool] = []
    service.started.connect(lambda: started.append(True))
    service.connect_card()
    service.arm(1)

    service.start()

    assert service.state == BMEServiceState.RUNNING
    assert fake.running is True
    assert started == [True]


def test_stop_deactivates_and_returns_to_configured_state() -> None:
    fake = _FakeDelayGenerator()
    service = BMEDelayGeneratorService(delay_generator=fake)
    stopped: list[bool] = []
    service.stopped.connect(lambda: stopped.append(True))
    service.connect_card()
    service.arm(1)
    service.start()

    service.stop()

    assert service.state == BMEServiceState.CONFIGURED
    assert fake.running is False
    assert stopped == [True]


def test_emergency_stop_uses_emergency_path() -> None:
    fake = _FakeDelayGenerator()
    service = BMEDelayGeneratorService(delay_generator=fake)
    service.connect_card()
    service.arm(1)

    service.emergency_stop()

    assert "emergency_stop" in fake.calls
    assert service.state == BMEServiceState.CONNECTED


def test_read_trigger_count_and_status_emit_values() -> None:
    fake = _FakeDelayGenerator(trigger_count=17, status=0x55)
    service = BMEDelayGeneratorService(delay_generator=fake)
    counts: list[int] = []
    statuses: list[int] = []
    service.trigger_count_ready.connect(counts.append)
    service.status_ready.connect(statuses.append)
    service.connect_card()

    service.read_trigger_count()
    service.read_status()

    assert counts == [17]
    assert statuses == [0x55]


def test_disconnect_closes_and_returns_idle() -> None:
    fake = _FakeDelayGenerator()
    service = BMEDelayGeneratorService(delay_generator=fake)
    service.connect_card()

    service.disconnect_card()

    assert service.state == BMEServiceState.IDLE
    assert fake.closed is True


class _FakeDelayGenerator:
    def __init__(self, *, fail_on: str | None = None, trigger_count: int = 12, status: int = 0x40) -> None:
        self.fail_on = fail_on
        self.trigger_count = trigger_count
        self.status = status
        self.info = BMEConnectionInfo(product=47, slot=5, master=True, index=0, detected_count=1, detect_error=0)
        self.config = None
        self.expected_trigger_count = None
        self.running = False
        self.closed = False
        self.calls: list[str] = []

    def connect(self) -> None:
        self.calls.append("connect")
        self._maybe_fail("connect")

    def configure(self, config: BMEConfig) -> None:
        self.calls.append("configure")
        self._maybe_fail("configure")
        self.config = config

    def arm(self, expected_trigger_count: int) -> None:
        self.calls.append("arm")
        self._maybe_fail("arm")
        self.expected_trigger_count = expected_trigger_count

    def start(self) -> None:
        self.calls.append("start")
        self._maybe_fail("start")
        self.running = True

    def stop(self) -> None:
        self.calls.append("stop")
        self._maybe_fail("stop")
        self.running = False

    def emergency_stop(self) -> None:
        self.calls.append("emergency_stop")
        self.running = False

    def read_trigger_count(self) -> int:
        self.calls.append("read_trigger_count")
        return self.trigger_count

    def read_status(self) -> int:
        self.calls.append("read_status")
        return self.status

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True

    def _maybe_fail(self, operation: str) -> None:
        if self.fail_on == operation:
            raise DelayGeneratorError(f"{operation} failed")
