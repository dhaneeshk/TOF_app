"""Mock BME delay generator for simulation and tests."""

from __future__ import annotations

import logging

from pytof_new.config.models import BMEConfig
from pytof_new.exceptions import DelayGeneratorError
from pytof_new.hardware.delay_generator_base import DelayGeneratorBase

LOGGER = logging.getLogger(__name__)


class MockDelayGenerator(DelayGeneratorBase):
    """Stateful simulated delay generator."""

    def __init__(self) -> None:
        self.connected = False
        self.configured = False
        self.outputs_enabled = False
        self.running = False
        self.config: BMEConfig | None = None

    def connect(self) -> None:
        """Connect the mock delay generator."""
        self.connected = True
        LOGGER.info("mock delay generator connected")

    def configure(self, config: BMEConfig) -> None:
        """Apply mock delay settings."""
        if not self.connected:
            raise DelayGeneratorError("delay generator is not connected")
        config.validate()
        self.config = config
        self.configured = True
        LOGGER.info("mock delay generator configured")

    def enable_outputs(self) -> None:
        """Enable mock outputs."""
        if not self.configured:
            raise DelayGeneratorError("delay generator is not configured")
        self.outputs_enabled = True
        LOGGER.info("mock delay generator outputs enabled")

    def disable_outputs(self) -> None:
        """Disable mock outputs."""
        if not self.outputs_enabled:
            return
        self.outputs_enabled = False
        LOGGER.info("mock delay generator outputs disabled")

    def start(self) -> None:
        """Start mock triggering."""
        if not self.outputs_enabled:
            raise DelayGeneratorError("delay generator outputs are not enabled")
        self.running = True
        LOGGER.info("mock delay generator started")

    def stop(self) -> None:
        """Stop mock triggering."""
        if not self.running:
            return
        self.running = False
        LOGGER.info("mock delay generator stopped")

    def close(self) -> None:
        """Close the mock delay generator."""
        self.stop()
        self.disable_outputs()
        self.connected = False
        self.configured = False
        LOGGER.info("mock delay generator closed")
