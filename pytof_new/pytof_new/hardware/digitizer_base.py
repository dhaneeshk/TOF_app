"""Abstract interface for digitizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pytof_new.config.models import DigitizerConfig


class DigitizerBase(ABC):
    """High-level digitizer interface used by acquisition controllers."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the digitizer."""

    @abstractmethod
    def configure(self, config: DigitizerConfig) -> None:
        """Apply acquisition configuration."""

    @abstractmethod
    def arm(self) -> None:
        """Arm the digitizer and wait for triggers."""

    @abstractmethod
    def read_batch(self) -> np.ndarray:
        """Read one batch with shape (segments, samples)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition and DMA activity."""

    @abstractmethod
    def close(self) -> None:
        """Close all digitizer resources."""
