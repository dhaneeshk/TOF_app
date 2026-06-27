"""Application-specific exceptions."""


class HardwareError(RuntimeError):
    """Base class for hardware-related failures."""


class DigitizerError(HardwareError):
    """Raised when digitizer operations fail."""


class DelayGeneratorError(HardwareError):
    """Raised when delay-generator operations fail."""


class AcquisitionTimeoutError(HardwareError):
    """Raised when acquisition times out before data is available."""


class InvalidStateTransitionError(RuntimeError):
    """Raised when an acquisition state transition is not allowed."""
