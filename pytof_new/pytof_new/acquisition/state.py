"""Explicit acquisition state machine."""

from __future__ import annotations

from enum import Enum, auto

from pytof_new.exceptions import InvalidStateTransitionError


class AcquisitionState(Enum):
    """Allowed high-level acquisition states."""

    DISCONNECTED = auto()
    CONNECTED = auto()
    CONFIGURED = auto()
    ARMED = auto()
    ACQUIRING = auto()
    STOPPING = auto()
    ERROR = auto()


_ALLOWED_TRANSITIONS: dict[AcquisitionState, set[AcquisitionState]] = {
    AcquisitionState.DISCONNECTED: {AcquisitionState.CONNECTED, AcquisitionState.ERROR},
    AcquisitionState.CONNECTED: {AcquisitionState.CONFIGURED, AcquisitionState.DISCONNECTED, AcquisitionState.ERROR},
    AcquisitionState.CONFIGURED: {AcquisitionState.ARMED, AcquisitionState.CONNECTED, AcquisitionState.ERROR},
    AcquisitionState.ARMED: {AcquisitionState.ACQUIRING, AcquisitionState.CONFIGURED, AcquisitionState.ERROR},
    AcquisitionState.ACQUIRING: {AcquisitionState.STOPPING, AcquisitionState.ERROR},
    AcquisitionState.STOPPING: {AcquisitionState.CONFIGURED, AcquisitionState.DISCONNECTED, AcquisitionState.ERROR},
    AcquisitionState.ERROR: {AcquisitionState.DISCONNECTED, AcquisitionState.CONNECTED},
}


class AcquisitionStateMachine:
    """Validate and store acquisition state transitions."""

    def __init__(self) -> None:
        self._state = AcquisitionState.DISCONNECTED

    @property
    def state(self) -> AcquisitionState:
        """Return the current state."""
        return self._state

    def transition_to(self, new_state: AcquisitionState) -> None:
        """Move to a new state if the transition is allowed."""
        if new_state == self._state:
            return
        if new_state not in _ALLOWED_TRANSITIONS[self._state]:
            raise InvalidStateTransitionError(f"invalid transition {self._state.name} -> {new_state.name}")
        self._state = new_state

    def force_error(self) -> None:
        """Move to ERROR regardless of current state."""
        self._state = AcquisitionState.ERROR
