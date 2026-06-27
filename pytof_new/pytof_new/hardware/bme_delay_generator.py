"""Persistent BME SG08p delay-generator session.

The implementation is import-safe: it does not load ``DelayGenerator.dll`` at
module import time.  This step owns safe session lifecycle operations only
(discovery, initialization, deactivation, counters, status, cleanup).  Physical
output activation remains disabled until the required numeric SG08p channel and
clock constants have been verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging

from pytof_new.config.models import BMEConfig
from pytof_new.exceptions import DelayGeneratorError
from pytof_new.hardware.bme_constants import (
    CRYSTAL_OSCILLATOR,
    DELAY_CHANNEL_IDS,
    ORDINARY_TOF_GO_SIGNAL,
    SUPPORTED_SG08P_PRODUCTS,
)
from pytof_new.hardware.bme_driver import BMEDriverApi, BMEPciDelayGeneratorInfo
from pytof_new.hardware.delay_generator_base import DelayGeneratorBase

LOGGER = logging.getLogger(__name__)


class BMEDelayGeneratorState(str, Enum):
    """Internal lifecycle state for one persistent BME session."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    CONFIGURED = "configured"
    ARMED = "armed"
    RUNNING = "running"
    ERROR = "error"


@dataclass(frozen=True)
class BMEConnectionInfo:
    """Connected BME card identity discovered through the DLL."""

    product: int
    slot: int
    master: bool
    index: int
    detected_count: int
    detect_error: int


class BMEDelayGenerator(DelayGeneratorBase):
    """Safe real BME delay-generator session.

    The object owns one logical BME card index.  ``connect`` initializes the card
    and immediately deactivates it so no trigger output is produced merely by
    connecting.  Channel programming and physical start are deliberately blocked
    until verified ``DG_Data.h`` numeric constants are available.
    """

    def __init__(self, api: BMEDriverApi | None = None, card_index: int = 0) -> None:
        self.api = api or BMEDriverApi()
        self.card_index = int(card_index)
        self.state = BMEDelayGeneratorState.DISCONNECTED
        self.info: BMEConnectionInfo | None = None
        self.config: BMEConfig | None = None
        self.expected_trigger_count: int | None = None
        self._reserved = False
        self._initialized = False

    @property
    def connected(self) -> bool:
        """Return whether a card has been initialized in this session."""
        return self._initialized and self.info is not None

    def connect(self) -> None:
        """Discover, reserve, initialize, and deactivate one PCI BME card."""
        if self.connected:
            return
        try:
            count, detect_error = self.api.detect_pci_delay_generators()
            if count <= 0:
                raise DelayGeneratorError(f"No BME PCI delay generators detected (driver error {detect_error})")
            if self.card_index < 0 or self.card_index >= count:
                raise DelayGeneratorError(f"BME card index {self.card_index} is outside detected range 0..{count - 1}")

            self.api.reserve_data(count)
            self._reserved = True
            card = self.api.get_pci_delay_generator(self.card_index)
            if card.product not in SUPPORTED_SG08P_PRODUCTS:
                raise DelayGeneratorError(f"Unsupported BME product code {card.product}; expected SG08p product")
            self.api.initialize(card.slot, card.product, card.index)
            self._initialized = True
            self.info = _connection_info(card, count=count, detect_error=detect_error)
            self.api.deactivate(card.index)
            self.state = BMEDelayGeneratorState.CONNECTED
            LOGGER.info("BME delay generator connected: %s", self.info)
        except Exception:
            self.state = BMEDelayGeneratorState.ERROR
            self._cleanup_after_failed_connect()
            raise

    def configure(self, config: BMEConfig) -> None:
        """Program SG08p clock and output channels while outputs are inactive."""
        if not self.connected:
            raise DelayGeneratorError("BME delay generator is not connected")
        try:
            config.validate()
        except ValueError as exc:
            raise DelayGeneratorError(str(exc)) from exc
        self._validate_current_basic_channels(config)
        self._validate_active_windows(config)

        self.api.deactivate(self.card_index)
        self.api.set_g08_clock_parameters(
            clock_enable=True,
            oscillator_divider=16,
            trigger_divider=1,
            trigger_multiplier=1,
            clock_source=CRYSTAL_OSCILLATOR,
            index=self.card_index,
        )
        self.api.set_g08_trigger_parameters(
            gate_terminate=False,
            gate_level_v=0.0,
            gate_delay_us=-1.0,
            ignore_gate=True,
            synchronize_gate=False,
            force_trigger_us=-1.0,
            step_back_time_us=-1.0,
            burst_counter=0,
            ms_bus=0,
            index=self.card_index,
        )
        for channel in ("A", "B", "C", "D", "E", "F"):
            self._program_channel(config, channel)
        self.config = config
        self.state = BMEDelayGeneratorState.CONFIGURED

    def arm(self, expected_trigger_count: int) -> None:
        """Prepare the event counter for a finite trigger sequence."""
        if not self.connected:
            raise DelayGeneratorError("BME delay generator is not connected")
        if self.config is None:
            raise DelayGeneratorError("BME delay generator is not configured")
        if expected_trigger_count <= 0:
            raise DelayGeneratorError("expected_trigger_count must be positive")
        self.expected_trigger_count = int(expected_trigger_count)
        self.api.set_trigger_parameters(
            trigger_terminate=self.config.trigger_termination_ohm == 50,
            internal_clock_us=self.config.repetition_period_s * 1e6,
            trigger_level_v=0.0,
            preset_value=self.expected_trigger_count,
            gate_divider=1,
            positive_gate=True,
            internal_trigger=True,
            internal_arm=False,
            software_trigger=False,
            rising_edge=True,
            stop_on_preset=True,
            reset_when_done=True,
            trigger_enable=True,
            index=self.card_index,
        )
        self.api.reset_event_counter(self.card_index)
        self.api.reset_output_modulo_counters(self.card_index)
        self.state = BMEDelayGeneratorState.ARMED

    def enable_outputs(self) -> None:
        """Compatibility method; physical outputs remain blocked for safety."""
        raise DelayGeneratorError("BME output activation is deferred until SG08p channel constants are verified")

    def disable_outputs(self) -> None:
        """Force outputs inactive when connected; safe and idempotent."""
        if not self.connected:
            return
        self.api.deactivate(self.card_index)
        if self.state != BMEDelayGeneratorState.ERROR:
            self.state = BMEDelayGeneratorState.CONNECTED if self.config is None else BMEDelayGeneratorState.CONFIGURED

    def start(self) -> None:
        """Activate the BME after the Spectrum card has been armed."""
        if self.state != BMEDelayGeneratorState.ARMED:
            raise DelayGeneratorError("BME delay generator must be armed before start")
        self.api.activate(self.card_index)
        self.state = BMEDelayGeneratorState.RUNNING

    def stop(self) -> None:
        """Gracefully stop by deactivating the card."""
        self.disable_outputs()

    def emergency_stop(self) -> None:
        """Immediately deactivate the card, logging failures."""
        if not self.connected:
            return
        try:
            self.api.deactivate(self.card_index)
        except Exception:
            LOGGER.exception("BME emergency deactivate failed")
            self.state = BMEDelayGeneratorState.ERROR

    def read_trigger_count(self) -> int:
        """Read the BME event counter."""
        if not self.connected:
            raise DelayGeneratorError("BME delay generator is not connected")
        return self.api.read_trigger_counter(self.card_index)

    def read_status(self) -> int:
        """Read the raw BME status register value."""
        if not self.connected:
            raise DelayGeneratorError("BME delay generator is not connected")
        return self.api.read_status(self.card_index)

    def close(self) -> None:
        """Deactivate and release DLL resources; safe after partial failures."""
        errors: list[BaseException] = []
        if self._initialized:
            try:
                self.api.deactivate(self.card_index)
            except Exception as exc:
                LOGGER.exception("BME deactivate during close failed")
                errors.append(exc)
        if self._reserved:
            try:
                self.api.release_data()
            except Exception as exc:
                LOGGER.exception("BME release during close failed")
                errors.append(exc)
        self._clear_state()
        if errors:
            self.state = BMEDelayGeneratorState.ERROR

    def _cleanup_after_failed_connect(self) -> None:
        try:
            self.close()
        except Exception:
            LOGGER.exception("BME cleanup after failed connect failed")
        self._clear_state()
        self.state = BMEDelayGeneratorState.ERROR

    def _clear_state(self) -> None:
        self.info = None
        self.config = None
        self.expected_trigger_count = None
        self._initialized = False
        self._reserved = False
        self.state = BMEDelayGeneratorState.DISCONNECTED

    def _validate_current_basic_channels(self, config: BMEConfig) -> None:
        valid = {"A", "B", "C", "D", "E", "F"}
        channels = [config.digitizer_channel, config.push_channel, config.pull_channel]
        normalized = [channel.upper() for channel in channels]
        if any(channel not in valid for channel in normalized):
            raise DelayGeneratorError("BME channels must be one of A, B, C, D, E, F")
        if len(set(normalized)) != len(normalized):
            raise DelayGeneratorError("BME configured channels must be unique")

    def _validate_active_windows(self, config: BMEConfig) -> None:
        for label, delay_s, width_s in (
            ("digitizer", config.digitizer_trigger_delay_s, config.digitizer_trigger_width_s),
            ("PUSH", config.push_trigger_delay_s, config.push_trigger_width_s),
            ("PULL", config.pull_trigger_delay_s, config.pull_trigger_width_s),
        ):
            if delay_s >= config.tof_window_s:
                raise DelayGeneratorError(f"{label} trigger delay must be less than TOF window")
            if delay_s + width_s >= config.repetition_period_s:
                raise DelayGeneratorError(f"{label} trigger delay plus width must be less than repetition period")

    def _program_channel(self, config: BMEConfig, channel: str) -> None:
        channel = channel.upper()
        if channel == config.digitizer_channel.upper():
            enabled = True
            delay_s = config.digitizer_trigger_delay_s
            width_s = config.digitizer_trigger_width_s
            positive = config.digitizer_polarity_positive
        elif channel == config.push_channel.upper():
            enabled = True
            delay_s = config.push_trigger_delay_s
            width_s = config.push_trigger_width_s
            positive = config.push_polarity_positive
        elif channel == config.pull_channel.upper():
            enabled = True
            delay_s = config.pull_trigger_delay_s
            width_s = config.pull_trigger_width_s
            positive = config.pull_polarity_positive
        else:
            enabled = False
            delay_s = 0.0
            width_s = 0.0
            positive = True
        self.api.set_g08_delay(
            channel=DELAY_CHANNEL_IDS[channel],
            fire_first_us=delay_s * 1e6,
            pulse_width_us=width_s * 1e6,
            output_modulo=1,
            output_offset=0,
            go_signal=ORDINARY_TOF_GO_SIGNAL if enabled else 0,
            positive=positive,
            terminate=config.trigger_termination_ohm == 50,
            disconnect=False,
            onto_ms_bus=False,
            input_positive=True,
            index=self.card_index,
        )


def _connection_info(card: BMEPciDelayGeneratorInfo, *, count: int, detect_error: int) -> BMEConnectionInfo:
    return BMEConnectionInfo(
        product=card.product,
        slot=card.slot,
        master=card.master,
        index=card.index,
        detected_count=int(count),
        detect_error=int(detect_error),
    )
