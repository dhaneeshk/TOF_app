"""Verified constants and flat export names for the BME SG08p driver.

Only values verified from available source material are defined here.  Numeric
constants that live only in the unavailable ``DG_Data.h`` are deliberately not
guessed; call ``require_constant`` to fail with a clear message if later code
tries to use one before it has been verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pytof_new.exceptions import DelayGeneratorError


BME_SUCCESS = 0

# Product codes verified from the BME manual's DG_Data.h constants table.
BME_SG08P1 = 44
BME_SG08P2 = 45
BME_SG08P3 = 46
BME_SG08P4 = 47
SUPPORTED_SG08P_PRODUCTS = frozenset({BME_SG08P1, BME_SG08P2, BME_SG08P3, BME_SG08P4})

# Initialization / clock-source constants verified from the manual.
MASTER_MODULE = 1
SLAVE_MODULE = 2
CRYSTAL_OSCILLATOR = 1
TRIGGER_INPUT = 2
TRIGGER_AND_OSCILLATOR = 3
MASTER_SLAVE_BUS = 4

# Output-level constants verified from the manual.
TTL_VOLTAGE_LEVEL = 1
NIM_VOLTAGE_LEVEL = 2
ECL_VOLTAGE_LEVEL = 3

# Channel identifiers verified from the manual.
DELAY_CHANNEL_T0 = 1
DELAY_CHANNEL_A = 2
DELAY_CHANNEL_B = 3
DELAY_CHANNEL_C = 4
DELAY_CHANNEL_D = 5
DELAY_CHANNEL_E = 6
DELAY_CHANNEL_F = 7
DELAY_CHANNEL_IDS: dict[str, int] = {
    "A": DELAY_CHANNEL_A,
    "B": DELAY_CHANNEL_B,
    "C": DELAY_CHANNEL_C,
    "D": DELAY_CHANNEL_D,
    "E": DELAY_CHANNEL_E,
    "F": DELAY_CHANNEL_F,
}

# Channel GoSignal bits verified from the manual.
LOCAL_PRIMARY = 0x1
LOCAL_SECONDARY = 0x2
LOCAL_FORCE = 0x4
RESYNCHRONIZE = 0x8
MASTER_PRIMARY = 0x10
MASTER_SECONDARY = 0x20
MASTER_FORCE = 0x40
SYSTEM_CLOCK = 0x80
DELAY_CLOCK = 0x100
INHIBIT_LOCAL = 0x200
START_LOCAL = 0x400
START_BUS = 0x800
STEP_BACK_LOCAL = 0x1000
STEP_BACK_BUS = 0x2000
RUN_CIRCLE = 0x4000
SYNCH_RELOAD = 0x8000
ENABLE_FROM_E = 0x10000
ENABLE_FROM_F = 0x20000
ENABLE_FROM_BUS = 0x40000

# SG08p ordinary TOF defaults.  LocalPrimary is the chosen first production
# path; legacy pyTOF used MasterPrimary with MS-bus routing.
ORDINARY_TOF_GO_SIGNAL = LOCAL_PRIMARY

REQUIRED_FLAT_EXPORTS: tuple[str, ...] = (
    "Reserve_DG_Data",
    "Release_DG_Data",
    "DetectPciDelayGenerators",
    "GetPciDelayGenerator",
    "Initialize_DG_BME",
    "Set_TriggerParameters",
    "Set_G08_TriggerParameters",
    "Set_G08_ClockParameters",
    "Set_G08_Delay",
    "Activate_DG_BME",
    "Deactivate_DG_BME",
    "ResetEventCounter",
    "ResetOutputModuloCounters",
    "ReadTriggerCounter",
    "Read_DG_Status",
)

DEFERRED_STRUCTURE_EXPORTS: tuple[str, ...] = (
    "Set_DG_BME",
    "SetDelayGenerator",
    "ResetRegisters",
    "ResetControl",
    "SaveState",
    "ReadState",
    "SaveParameters",
    "ReadParameters",
)

ERROR_DESCRIPTIONS: dict[int, str] = {
    0: "success",
    2: "delay-generator index out of range",
    3: "invalid negative active delay",
    4: "internal and software trigger are both enabled",
    5: "delay too long",
    6: "invalid output level or card open failed",
    7: "invalid clock source",
    11: "PCI/PLX command failure",
    13: "control-line or card-connection problem",
}


class BMEChannel(str, Enum):
    """Human channel labels supported by SG08p hardware."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


@dataclass(frozen=True)
class DeferredBMEConstant:
    """A named vendor constant that still needs numeric verification."""

    name: str
    required_for: str
    source_needed: str = "DG_Data.h or another authoritative numeric source"


DEFERRED_CONSTANTS: dict[str, DeferredBMEConstant] = {
    "MasterSlavePrimary": DeferredBMEConstant("MasterSlavePrimary", "routing the primary trigger on the master/slave bus"),
}


def require_constant(name: str) -> int:
    """Return a verified numeric constant or fail with the missing source."""
    deferred = DEFERRED_CONSTANTS.get(name)
    if deferred is None:
        raise DelayGeneratorError(f"BME constant is not defined or verified: {name}")
    raise DelayGeneratorError(
        f"BME constant {deferred.name} is required for {deferred.required_for}, "
        f"but its numeric value has not been verified from {deferred.source_needed}."
    )
