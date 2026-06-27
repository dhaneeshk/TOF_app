import pytest

from pytof_new.exceptions import DelayGeneratorError
from pytof_new.hardware.bme_constants import (
    BME_SUCCESS,
    DEFERRED_CONSTANTS,
    DEFERRED_STRUCTURE_EXPORTS,
    ERROR_DESCRIPTIONS,
    CRYSTAL_OSCILLATOR,
    DELAY_CHANNEL_IDS,
    LOCAL_PRIMARY,
    MASTER_PRIMARY,
    ORDINARY_TOF_GO_SIGNAL,
    REQUIRED_FLAT_EXPORTS,
    SUPPORTED_SG08P_PRODUCTS,
    BMEChannel,
    require_constant,
)


def test_required_flat_exports_are_the_initial_bound_surface() -> None:
    assert "Set_G08_Delay" in REQUIRED_FLAT_EXPORTS
    assert "Set_TriggerParameters" in REQUIRED_FLAT_EXPORTS
    assert "ResetOutputModuloCounters" in REQUIRED_FLAT_EXPORTS
    assert "Set_DG_BME" not in REQUIRED_FLAT_EXPORTS
    assert "ResetRegisters" not in REQUIRED_FLAT_EXPORTS


def test_structure_exports_are_explicitly_deferred() -> None:
    assert "Set_DG_BME" in DEFERRED_STRUCTURE_EXPORTS
    assert "SetDelayGenerator" in DEFERRED_STRUCTURE_EXPORTS
    assert "ResetControl" in DEFERRED_STRUCTURE_EXPORTS


def test_verified_error_codes_include_success_and_common_failures() -> None:
    assert BME_SUCCESS == 0
    assert ERROR_DESCRIPTIONS[0] == "success"
    assert "index" in ERROR_DESCRIPTIONS[2]
    assert "PCI" in ERROR_DESCRIPTIONS[11]


def test_channel_labels_are_supported_without_numeric_ids() -> None:
    assert tuple(channel.value for channel in BMEChannel) == ("A", "B", "C", "D", "E", "F")


def test_manual_verified_constants_are_available() -> None:
    assert SUPPORTED_SG08P_PRODUCTS == frozenset({44, 45, 46, 47})
    assert CRYSTAL_OSCILLATOR == 1
    assert DELAY_CHANNEL_IDS == {"A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "F": 7}
    assert LOCAL_PRIMARY == 0x1
    assert MASTER_PRIMARY == 0x10
    assert ORDINARY_TOF_GO_SIGNAL == LOCAL_PRIMARY


def test_unverified_numeric_constants_fail_clearly() -> None:
    assert "MasterSlavePrimary" in DEFERRED_CONSTANTS

    with pytest.raises(DelayGeneratorError, match="MasterSlavePrimary") as excinfo:
        require_constant("MasterSlavePrimary")

    assert "numeric value has not been verified" in str(excinfo.value)


def test_unknown_constant_fails_clearly() -> None:
    with pytest.raises(DelayGeneratorError, match="not defined or verified"):
        require_constant("NOT_A_BME_CONSTANT")
