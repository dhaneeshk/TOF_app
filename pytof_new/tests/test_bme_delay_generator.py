import pytest

from pytof_new.config.models import BMEConfig
from pytof_new.exceptions import DelayGeneratorError
from pytof_new.hardware.bme_delay_generator import BMEDelayGenerator, BMEDelayGeneratorState
from pytof_new.hardware.bme_driver import BMEPciDelayGeneratorInfo


def test_connect_initializes_and_deactivates_without_activation() -> None:
    api = _FakeBMEApi()
    bme = BMEDelayGenerator(api=api)

    bme.connect()

    assert bme.connected is True
    assert bme.state == BMEDelayGeneratorState.CONNECTED
    assert bme.info is not None
    assert bme.info.product == 47
    assert bme.info.slot == 5
    assert bme.info.master is True
    assert api.calls == [
        ("detect_pci_delay_generators",),
        ("reserve_data", 1),
        ("get_pci_delay_generator", 0),
        ("initialize", 5, 47, 0),
        ("deactivate", 0),
    ]
    assert "activate" not in [call[0] for call in api.calls]


def test_connect_is_idempotent() -> None:
    api = _FakeBMEApi()
    bme = BMEDelayGenerator(api=api)

    bme.connect()
    bme.connect()

    assert api.calls.count(("detect_pci_delay_generators",)) == 1


def test_connect_fails_when_no_cards_detected() -> None:
    api = _FakeBMEApi(count=0, detect_error=13)
    bme = BMEDelayGenerator(api=api)

    with pytest.raises(DelayGeneratorError, match="No BME PCI"):
        bme.connect()

    assert bme.connected is False
    assert bme.state == BMEDelayGeneratorState.ERROR
    assert api.calls == [("detect_pci_delay_generators",)]


def test_connect_fails_for_out_of_range_card_index() -> None:
    api = _FakeBMEApi(count=1)
    bme = BMEDelayGenerator(api=api, card_index=2)

    with pytest.raises(DelayGeneratorError, match="outside detected range"):
        bme.connect()

    assert api.calls == [("detect_pci_delay_generators",)]


def test_connect_releases_after_initialize_failure() -> None:
    api = _FakeBMEApi(fail_on="initialize")
    bme = BMEDelayGenerator(api=api)

    with pytest.raises(DelayGeneratorError, match="initialize failed"):
        bme.connect()

    assert ("release_data",) in api.calls
    assert bme.connected is False
    assert bme.state == BMEDelayGeneratorState.ERROR


def test_close_deactivates_and_releases_once_then_idempotent() -> None:
    api = _FakeBMEApi()
    bme = BMEDelayGenerator(api=api)

    bme.connect()
    bme.close()
    bme.close()

    assert api.calls.count(("deactivate", 0)) == 2  # connect safety deactivate + close deactivate
    assert api.calls.count(("release_data",)) == 1
    assert bme.connected is False
    assert bme.state == BMEDelayGeneratorState.DISCONNECTED


def test_configure_programs_clock_trigger_and_all_channels_without_activation() -> None:
    api = _FakeBMEApi()
    bme = BMEDelayGenerator(api=api)
    bme.connect()

    bme.configure(BMEConfig())

    assert bme.state == BMEDelayGeneratorState.CONFIGURED
    assert ("set_g08_clock_parameters", True, 16, 1, 1, 1, 0) in api.calls
    assert ("set_g08_trigger_parameters", False, 0.0, -1.0, True, False, -1.0, -1.0, 0, 0, 0) in api.calls
    delay_calls = [call for call in api.calls if call[0] == "set_g08_delay"]
    assert len(delay_calls) == 6
    assert delay_calls[0] == ("set_g08_delay", 2, 0.0, 50.0, 1, 0, 0x1, True, True, False, False, True, 0)
    assert delay_calls[1][6] == 0
    assert delay_calls[2] == ("set_g08_delay", 4, 0.0, 50.0, 1, 0, 0x1, True, True, False, False, True, 0)
    assert delay_calls[5] == ("set_g08_delay", 7, 0.0, 50.0, 1, 0, 0x1, False, True, False, False, True, 0)
    assert "activate" not in [call[0] for call in api.calls]


def test_configure_rejects_duplicate_or_unknown_channels_before_constants() -> None:
    api = _FakeBMEApi()
    bme = BMEDelayGenerator(api=api)
    bme.connect()

    with pytest.raises(DelayGeneratorError, match="unique"):
        bme.configure(BMEConfig(digitizer_channel="A", push_channel="A"))
    with pytest.raises(DelayGeneratorError, match="one of A"):
        bme.configure(BMEConfig(digitizer_channel="Z"))


def test_arm_resets_counter_and_read_counter_status() -> None:
    api = _FakeBMEApi(trigger_count=7, status=0x44)
    bme = BMEDelayGenerator(api=api)
    bme.connect()
    bme.configure(BMEConfig())

    bme.arm(4500)

    assert bme.expected_trigger_count == 4500
    assert bme.state == BMEDelayGeneratorState.ARMED
    assert bme.read_trigger_count() == 7
    assert bme.read_status() == 0x44
    assert ("set_trigger_parameters", True, 105.0, 0.0, 4500, 1, True, True, False, False, True, True, True, True, 0) in api.calls
    assert ("reset_event_counter", 0) in api.calls
    assert ("reset_output_modulo_counters", 0) in api.calls


def test_arm_rejects_invalid_count() -> None:
    bme = BMEDelayGenerator(api=_FakeBMEApi())
    bme.connect()
    bme.configure(BMEConfig())

    with pytest.raises(DelayGeneratorError, match="positive"):
        bme.arm(0)


def test_start_activates_only_after_arm() -> None:
    api = _FakeBMEApi()
    bme = BMEDelayGenerator(api=api)
    bme.connect()
    bme.configure(BMEConfig())
    bme.arm(1)

    with pytest.raises(DelayGeneratorError, match="deferred"):
        bme.enable_outputs()
    bme.start()

    assert bme.state == BMEDelayGeneratorState.RUNNING
    assert ("activate", 0) in api.calls


def test_start_requires_arm() -> None:
    bme = BMEDelayGenerator(api=_FakeBMEApi())
    bme.connect()
    bme.configure(BMEConfig())

    with pytest.raises(DelayGeneratorError, match="armed"):
        bme.start()


def test_stop_and_emergency_stop_deactivate_when_connected() -> None:
    api = _FakeBMEApi()
    bme = BMEDelayGenerator(api=api)
    bme.connect()

    bme.stop()
    bme.emergency_stop()

    assert api.calls.count(("deactivate", 0)) == 3


class _FakeBMEApi:
    def __init__(
        self,
        *,
        count: int = 1,
        detect_error: int = 0,
        fail_on: str | None = None,
        trigger_count: int = 12,
        status: int = 0x40,
        product: int = 47,
    ) -> None:
        self.count = count
        self.detect_error = detect_error
        self.fail_on = fail_on
        self.trigger_count = trigger_count
        self.status = status
        self.product = product
        self.calls: list[tuple] = []

    def detect_pci_delay_generators(self) -> tuple[int, int]:
        self.calls.append(("detect_pci_delay_generators",))
        return self.count, self.detect_error

    def reserve_data(self, count: int) -> None:
        self.calls.append(("reserve_data", count))
        self._maybe_fail("reserve_data")

    def get_pci_delay_generator(self, index: int) -> BMEPciDelayGeneratorInfo:
        self.calls.append(("get_pci_delay_generator", index))
        self._maybe_fail("get_pci_delay_generator")
        return BMEPciDelayGeneratorInfo(product=self.product, slot=5, master=True, index=index)

    def initialize(self, slot: int, product: int, index: int) -> None:
        self.calls.append(("initialize", slot, product, index))
        self._maybe_fail("initialize")

    def deactivate(self, index: int) -> None:
        self.calls.append(("deactivate", index))
        self._maybe_fail("deactivate")

    def release_data(self) -> None:
        self.calls.append(("release_data",))
        self._maybe_fail("release_data")

    def reset_event_counter(self, index: int) -> None:
        self.calls.append(("reset_event_counter", index))
        self._maybe_fail("reset_event_counter")

    def reset_output_modulo_counters(self, index: int) -> None:
        self.calls.append(("reset_output_modulo_counters", index))
        self._maybe_fail("reset_output_modulo_counters")

    def set_g08_clock_parameters(self, *, clock_enable, oscillator_divider, trigger_divider, trigger_multiplier, clock_source, index) -> None:
        self.calls.append(("set_g08_clock_parameters", clock_enable, oscillator_divider, trigger_divider, trigger_multiplier, clock_source, index))
        self._maybe_fail("set_g08_clock_parameters")

    def set_g08_trigger_parameters(
        self,
        *,
        gate_terminate,
        gate_level_v,
        gate_delay_us,
        ignore_gate,
        synchronize_gate,
        force_trigger_us,
        step_back_time_us,
        burst_counter,
        ms_bus,
        index,
    ) -> None:
        self.calls.append(("set_g08_trigger_parameters", gate_terminate, gate_level_v, gate_delay_us, ignore_gate, synchronize_gate, force_trigger_us, step_back_time_us, burst_counter, ms_bus, index))
        self._maybe_fail("set_g08_trigger_parameters")

    def set_g08_delay(
        self,
        *,
        channel,
        fire_first_us,
        pulse_width_us,
        output_modulo,
        output_offset,
        go_signal,
        positive,
        terminate,
        disconnect,
        onto_ms_bus,
        input_positive,
        index,
    ) -> None:
        self.calls.append(("set_g08_delay", channel, fire_first_us, pulse_width_us, output_modulo, output_offset, go_signal, positive, terminate, disconnect, onto_ms_bus, input_positive, index))
        self._maybe_fail("set_g08_delay")

    def set_trigger_parameters(
        self,
        *,
        trigger_terminate,
        internal_clock_us,
        trigger_level_v,
        preset_value,
        gate_divider,
        positive_gate,
        internal_trigger,
        internal_arm,
        software_trigger,
        rising_edge,
        stop_on_preset,
        reset_when_done,
        trigger_enable,
        index,
    ) -> None:
        self.calls.append(("set_trigger_parameters", trigger_terminate, internal_clock_us, trigger_level_v, preset_value, gate_divider, positive_gate, internal_trigger, internal_arm, software_trigger, rising_edge, stop_on_preset, reset_when_done, trigger_enable, index))
        self._maybe_fail("set_trigger_parameters")

    def activate(self, index: int) -> None:
        self.calls.append(("activate", index))
        self._maybe_fail("activate")

    def read_trigger_counter(self, index: int) -> int:
        self.calls.append(("read_trigger_counter", index))
        return self.trigger_count

    def read_status(self, index: int) -> int:
        self.calls.append(("read_status", index))
        return self.status

    def _maybe_fail(self, operation: str) -> None:
        if self.fail_on == operation:
            raise DelayGeneratorError(f"{operation} failed")
