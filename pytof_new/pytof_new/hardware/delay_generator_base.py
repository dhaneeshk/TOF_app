"""Abstract interface for delay generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pytof_new.config.models import BMEConfig


class DelayGeneratorBase(ABC):
    """High-level delay-generator interface used by acquisition controllers."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the delay generator."""

    @abstractmethod
    def configure(self, config: BMEConfig) -> None:
        """Apply delay and pulse configuration."""

    @abstractmethod
    def enable_outputs(self) -> None:
        """Enable pulse outputs."""

    @abstractmethod
    def disable_outputs(self) -> None:
        """Disable pulse outputs."""

    @abstractmethod
    def start(self) -> None:
        """Start pulse generation."""

    @abstractmethod
    def stop(self) -> None:
        """Stop pulse generation."""

    @abstractmethod
    def close(self) -> None:
        """Close all delay-generator resources."""
