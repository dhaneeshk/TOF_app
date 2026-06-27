"""High-level acquisition controller."""

from __future__ import annotations

import logging
import time

import numpy as np

from pytof_new.acquisition.models import AcquisitionBatch
from pytof_new.acquisition.state import AcquisitionState, AcquisitionStateMachine
from pytof_new.config.models import RunConfig
from pytof_new.hardware.delay_generator_base import DelayGeneratorBase
from pytof_new.hardware.digitizer_base import DigitizerBase

LOGGER = logging.getLogger(__name__)


class AcquisitionController:
    """Coordinate digitizer and delay generator through high-level commands."""

    def __init__(self, digitizer: DigitizerBase, delay_generator: DelayGeneratorBase) -> None:
        self.digitizer = digitizer
        self.delay_generator = delay_generator
        self.state_machine = AcquisitionStateMachine()
        self.config: RunConfig | None = None
        self.total_segments = 0
        self.dropped_triggers = 0

    @property
    def state(self) -> AcquisitionState:
        """Return current acquisition state."""
        return self.state_machine.state

    def connect_hardware(self) -> None:
        """Connect both hardware devices."""
        self.digitizer.connect()
        self.delay_generator.connect()
        self.state_machine.transition_to(AcquisitionState.CONNECTED)
        LOGGER.info("hardware connected")

    def configure_run(self, config: RunConfig) -> None:
        """Validate and configure both devices."""
        config.validate()
        self.digitizer.configure(config.digitizer)
        self.delay_generator.configure(config.bme)
        self.delay_generator.disable_outputs()
        self.config = config
        self.state_machine.transition_to(AcquisitionState.CONFIGURED)
        LOGGER.info("run configured")

    def arm(self) -> None:
        """Arm the digitizer while BME outputs remain disabled."""
        self.digitizer.arm()
        self.state_machine.transition_to(AcquisitionState.ARMED)
        LOGGER.info("digitizer armed")

    def start_acquisition(self, config: RunConfig | None = None) -> AcquisitionBatch:
        """Acquire one configured batch using the safe acquisition order."""
        try:
            self.begin_acquisition(config)
            return self.read_batch()
        finally:
            self.end_acquisition()

    def acquire_batch(self, config: RunConfig | None = None) -> AcquisitionBatch:
        """Acquire one batch."""
        return self.start_acquisition(config)

    def begin_acquisition(self, config: RunConfig | None = None) -> None:
        """Configure, arm, enable outputs, and start triggering once for a run."""
        if config is not None:
            if self.state == AcquisitionState.DISCONNECTED:
                self.connect_hardware()
            self.configure_run(config)
        if self.config is None:
            raise RuntimeError("run is not configured")
        if self.state == AcquisitionState.CONFIGURED:
            self.arm()
        try:
            self.delay_generator.enable_outputs()
            self.delay_generator.start()
            self.state_machine.transition_to(AcquisitionState.ACQUIRING)
            LOGGER.info("acquisition started")
        except Exception:
            self.state_machine.force_error()
            LOGGER.exception("acquisition start failed")
            raise

    def read_batch(self) -> AcquisitionBatch:
        """Read one batch while acquisition is already running."""
        if self.config is None:
            raise RuntimeError("run is not configured")
        if self.state != AcquisitionState.ACQUIRING:
            raise RuntimeError("acquisition is not running")
        try:
            raw_adc = self.digitizer.read_batch().copy()
            timestamps = np.arange(raw_adc.shape[0], dtype=np.float64) * self.config.bme.repetition_period_s + time.time()
            batch = AcquisitionBatch(
                raw_adc=raw_adc,
                timestamps=timestamps,
                sample_rate_hz=self.config.digitizer.sample_rate_hz,
                pretrigger_samples=self.config.digitizer.pretrigger_samples,
                first_trigger_index=self.total_segments,
                record_mode=self.config.digitizer.record_mode,
                hardware_averages_per_record=self.config.digitizer.hardware_averages_per_record,
                metadata={"dropped_triggers": getattr(self.digitizer, "dropped_triggers", 0)},
            )
            self.total_segments += raw_adc.shape[0]
            self.dropped_triggers = int(getattr(self.digitizer, "dropped_triggers", 0))
            return batch
        except Exception:
            self.state_machine.force_error()
            LOGGER.exception("batch read failed")
            raise

    def end_acquisition(self) -> None:
        """Stop an active acquisition run safely."""
        self.stop_acquisition()

    def stop_acquisition(self) -> None:
        """Stop devices safely and return to CONFIGURED when possible."""
        previous = self.state
        if previous not in {AcquisitionState.DISCONNECTED, AcquisitionState.ERROR}:
            try:
                if previous == AcquisitionState.ACQUIRING:
                    self.state_machine.transition_to(AcquisitionState.STOPPING)
                self.delay_generator.stop()
                self.delay_generator.disable_outputs()
                self.digitizer.stop()
                if self.state == AcquisitionState.STOPPING:
                    self.state_machine.transition_to(AcquisitionState.CONFIGURED)
            except Exception:
                self.state_machine.force_error()
                LOGGER.exception("safe stop failed")
                raise

    def disconnect_hardware(self) -> None:
        """Stop and close both devices."""
        try:
            self.stop_acquisition()
        finally:
            self.delay_generator.close()
            self.digitizer.close()
            if self.state == AcquisitionState.ERROR:
                self.state_machine.transition_to(AcquisitionState.DISCONNECTED)
            elif self.state != AcquisitionState.DISCONNECTED:
                while self.state != AcquisitionState.DISCONNECTED:
                    if self.state == AcquisitionState.CONFIGURED:
                        self.state_machine.transition_to(AcquisitionState.CONNECTED)
                    elif self.state == AcquisitionState.CONNECTED:
                        self.state_machine.transition_to(AcquisitionState.DISCONNECTED)
                    else:
                        self.state_machine.force_error()
                        self.state_machine.transition_to(AcquisitionState.DISCONNECTED)
            LOGGER.info("hardware disconnected")
