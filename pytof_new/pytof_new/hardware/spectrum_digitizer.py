"""Spectrum digitizer session skeleton.

This module intentionally does not import vendor DLL wrappers at import time.
"""

from __future__ import annotations

from ctypes import addressof, c_char, create_string_buffer, sizeof

import numpy as np

from pytof_new.config.models import DigitizerConfig
from pytof_new.exceptions import DigitizerError
from pytof_new.hardware.digitizer_base import DigitizerBase
from pytof_new.hardware.spectrum_driver import SpectrumDriverApi
from pytof_new.hardware.spectrum_limits import plan_spectrum_acquisition
from pytof_new.hardware.spectrum_models import (
    SpectrumAcquisitionMode,
    SpectrumAcquisitionPlan,
    SpectrumAcquisitionRequest,
    SpectrumAcquisitionResult,
    SpectrumHardwareInfo,
    SpectrumTriggerSource,
)


class SpectrumDigitizer(DigitizerBase):
    """Persistent Spectrum card session.

    The acquisition/configuration path is deliberately not implemented yet;
    this slice only owns the card handle and discovers safe card metadata.
    """

    def __init__(self, device: str = "/dev/spcm0", api: SpectrumDriverApi | None = None) -> None:
        self.device = device
        self.api = api or SpectrumDriverApi()
        self.handle: object | None = None
        self.hardware_info: SpectrumHardwareInfo | None = None
        self.acquisition_plan: SpectrumAcquisitionPlan | None = None
        self._dma_buffer: object | None = None

    def connect(self) -> None:
        """Open the Spectrum card once and discover basic card information."""
        if self.handle is not None:
            return
        self.handle = self.api.open(self.device)
        try:
            self.hardware_info = self._discover_hardware_info()
        except Exception:
            handle = self.handle
            self.handle = None
            self.hardware_info = None
            self.api.close(handle)
            raise

    def configure(self, config: DigitizerConfig) -> None:
        raise DigitizerError("Spectrum acquisition configuration is not implemented yet")

    def configure_request(self, request: SpectrumAcquisitionRequest) -> SpectrumAcquisitionPlan:
        """Apply a validated finite Spectrum acquisition request to the card."""
        if self.handle is None:
            raise DigitizerError("Spectrum digitizer is not connected")
        hardware_info = self.hardware_info or self._discover_hardware_info()
        plan = plan_spectrum_acquisition(request, hardware_info)

        self._set_i32_symbol("SPC_CHENABLE", self.api.require_symbol("CHANNEL0"))
        self._set_i32_symbol("SPC_AMP0", int(round(request.input_range_v * 1000.0)))
        self._apply_input_settings(request)
        self._set_i32_symbol("SPC_CARDMODE", self._card_mode_value(request.mode))
        self._set_i64_symbol("SPC_MEMSIZE", request.segment_samples * request.number_of_segments)
        self._set_i64_symbol("SPC_POSTTRIGGER", request.segment_samples - request.pretrigger_samples)
        self._set_i64_symbol("SPC_SEGMENTSIZE", request.segment_samples)
        if plan.is_fpga_sum:
            self._set_i32_symbol("SPC_AVERAGES", request.averages_per_segment)
        self._set_i32_symbol("SPC_CLOCKMODE", self.api.require_symbol("SPC_CM_INTPLL"))
        self._set_i64_symbol("SPC_SAMPLERATE", int(request.sample_rate_hz))
        self._set_i32_symbol("SPC_CLOCKOUT", 0)
        self._set_i32_if_symbol(("SPC_TIMEOUT",), int(round(request.timeout_s * 1000.0)))
        self._apply_trigger(request)

        self.acquisition_plan = plan
        return plan

    def arm(self) -> None:
        raise DigitizerError("Spectrum acquisition arm/start is not implemented yet")

    def read_batch(self) -> np.ndarray:
        raise DigitizerError("Spectrum acquisition readout is not implemented yet")

    def acquire_configured(self) -> SpectrumAcquisitionResult:
        """Acquire one finite batch using the current Spectrum configuration."""
        self.prepare_configured_acquisition()
        try:
            self.start_prepared_acquisition()
            return self.wait_for_prepared_result()
        finally:
            self.stop()

    def prepare_configured_acquisition(self) -> SpectrumAcquisitionPlan:
        """Allocate and define the DMA buffer for the current finite plan."""
        if self.handle is None:
            raise DigitizerError("Spectrum digitizer is not connected")
        if self.acquisition_plan is None:
            raise DigitizerError("Spectrum acquisition is not configured")

        plan = self.acquisition_plan
        buffer = self._allocate_page_aligned_buffer(plan.transfer_bytes)
        self._dma_buffer = buffer
        self.api.def_transfer(
            self.handle,
            self.api.require_symbol("SPCM_BUF_DATA"),
            self.api.require_symbol("SPCM_DIR_CARDTOPC"),
            0,
            buffer,
            0,
            plan.transfer_bytes,
        )
        return plan

    def start_prepared_acquisition(self) -> None:
        """Start the card, DMA engine, and trigger engine without waiting."""
        if self.handle is None:
            raise DigitizerError("Spectrum digitizer is not connected")
        if self.acquisition_plan is None or self._dma_buffer is None:
            raise DigitizerError("Spectrum acquisition is not prepared")
        self.api.command(
            self.handle,
            self.api.require_symbol("M2CMD_CARD_START")
            | self.api.require_symbol("M2CMD_CARD_ENABLETRIGGER")
            | self.api.require_symbol("M2CMD_DATA_STARTDMA"),
        )

    def wait_for_prepared_result(self) -> SpectrumAcquisitionResult:
        """Wait for the prepared acquisition and return copied data."""
        if self.handle is None:
            raise DigitizerError("Spectrum digitizer is not connected")
        if self.acquisition_plan is None or self._dma_buffer is None:
            raise DigitizerError("Spectrum acquisition is not prepared")
        plan = self.acquisition_plan
        buffer = self._dma_buffer
        self.api.command(
            self.handle,
            self.api.require_symbol("M2CMD_CARD_WAITREADY") | self.api.require_symbol("M2CMD_DATA_WAITDMA"),
        )
        flat = np.frombuffer(buffer, dtype=plan.dtype, count=int(np.prod(plan.output_shape)))
        data = flat.reshape(plan.output_shape).copy()
        return SpectrumAcquisitionResult(data=data, plan=plan, metadata=dict(plan.metadata))

    def stop(self) -> None:
        """Stop card, trigger, and DMA activity when possible."""
        if self.handle is None:
            return None
        try:
            self.api.command(
                self.handle,
                self.api.require_symbol("M2CMD_CARD_STOP")
                | self.api.require_symbol("M2CMD_CARD_DISABLETRIGGER")
                | self.api.require_symbol("M2CMD_DATA_STOPDMA"),
            )
        except DigitizerError:
            raise
        self._dma_buffer = None
        return None

    def close(self) -> None:
        """Close the persistent Spectrum card handle if it is open."""
        if self.handle is None:
            return
        handle = self.handle
        self.handle = None
        self.api.close(handle)

    @property
    def connected(self) -> bool:
        """Return whether the card handle is currently open."""
        return self.handle is not None

    def _discover_hardware_info(self) -> SpectrumHardwareInfo:
        if self.handle is None:
            raise DigitizerError("Spectrum digitizer is not connected")

        metadata: dict[str, object] = {}
        card_type = self._optional_i32("SPC_PCITYP")
        serial = self._optional_i32("SPC_PCISERIALNO")
        function_type = self._optional_i32("SPC_FNCTYPE")
        max_adc = self._optional_i32("SPC_MIINST_MAXADCVALUE")
        if card_type is not None:
            metadata["card_type"] = card_type
        if function_type is not None:
            metadata["function_type"] = function_type

        return SpectrumHardwareInfo(
            serial_number=serial,
            max_adc_value=max_adc if max_adc is not None else SpectrumHardwareInfo().max_adc_value,
            average_16bit_supported=self._detect_average_16bit_support(),
            metadata=metadata,
        )

    def _optional_i32(self, symbol_name: str) -> int | None:
        if self.handle is None or not self.api.has_symbol(symbol_name):
            return None
        return self.api.get_i32(self.handle, self.api.require_symbol(symbol_name))

    def _detect_average_16bit_support(self) -> bool:
        """Try available 16-bit average mode symbols by writing CARDMODE."""
        mode16 = None
        for symbol in ("SPC_REC_STD_AVERAGE_16BIT", "SPC_REC_STD_AVERAGE16", "SPC_REC_STD_AVERAGE_16"):
            if self.api.has_symbol(symbol):
                mode16 = self.api.require_symbol(symbol)
                break
        if mode16 is None:
            return False
        if self.handle is None:
            return False
        try:
            cardmode_reg = self.api.require_symbol("SPC_CARDMODE")
            saved = self.api.get_i32(self.handle, cardmode_reg)
            self.api.set_i32(self.handle, cardmode_reg, mode16)
            readback = self.api.get_i32(self.handle, cardmode_reg)
            self.api.set_i32(self.handle, cardmode_reg, saved)
            return readback == mode16
        except DigitizerError:
            return False

    def _card_mode_value(self, mode: SpectrumAcquisitionMode) -> int:
        if mode == SpectrumAcquisitionMode.RAW_MULTI:
            return self.api.require_symbol("SPC_REC_STD_MULTI")
        if mode == SpectrumAcquisitionMode.AVERAGE_32BIT:
            return self.api.require_symbol("SPC_REC_STD_AVERAGE")
        if mode == SpectrumAcquisitionMode.AVERAGE_16BIT:
            for symbol in ("SPC_REC_STD_AVERAGE_16BIT", "SPC_REC_STD_AVERAGE16", "SPC_REC_STD_AVERAGE_16"):
                if self.api.has_symbol(symbol):
                    return self.api.require_symbol(symbol)
        raise DigitizerError(f"unsupported Spectrum acquisition mode: {mode}")

    def _apply_trigger(self, request: SpectrumAcquisitionRequest) -> None:
        if request.trigger_source == SpectrumTriggerSource.SOFTWARE:
            self._set_i32_symbol("SPC_TRIG_ORMASK", self.api.require_symbol("SPC_TMASK_SOFTWARE"))
        elif request.trigger_source == SpectrumTriggerSource.EXTERNAL0:
            self._set_i32_symbol("SPC_TRIG_ORMASK", self.api.require_symbol("SPC_TMASK_EXT0"))
            edge_symbol = "SPC_TM_NEG" if request.trigger_edge == "falling" else "SPC_TM_POS"
            self._set_i32_symbol("SPC_TRIG_EXT0_MODE", self.api.require_symbol(edge_symbol))
            self._set_i32_symbol("SPC_TRIG_EXT0_LEVEL0", int(round(request.trigger_level_v * 1000.0)))
            self._apply_trigger_termination(request)
        elif request.trigger_source == SpectrumTriggerSource.CHANNEL0:
            self._set_i32_symbol("SPC_TRIG_CH_ORMASK0", self.api.require_symbol("SPC_TMASK0_CH0"))

    def _apply_input_settings(self, request: SpectrumAcquisitionRequest) -> None:
        """Apply optional channel input settings when symbols exist in the SDK."""
        coupling_value = 0 if request.coupling == "dc" else 1
        self._set_i32_if_symbol(("SPC_ACDC0", "SPC_COUPLING0"), coupling_value)
        bandwidth_value = 1 if request.bandwidth_limit_enabled else 0
        self._set_i32_if_symbol(("SPC_BWLIMIT0", "SPC_FILTER0", "SPC_BW_LIMIT0"), bandwidth_value)

    def _apply_trigger_termination(self, request: SpectrumAcquisitionRequest) -> None:
        """Apply optional external-trigger termination when symbols exist."""
        value = 1 if request.trigger_termination_ohm == 50 else 0
        self._set_i32_if_symbol(("SPC_TRIG_EXT0_TERM", "SPC_TRIG_EXT0_TERMINATION"), value)

    def _set_i32_if_symbol(self, symbol_names: tuple[str, ...], value: int) -> None:
        for register_name in symbol_names:
            if self.api.has_symbol(register_name):
                self._set_i32_symbol(register_name, value)
                return

    def _set_i32_symbol(self, register_name: str, value: int) -> None:
        if self.handle is None:
            raise DigitizerError("Spectrum digitizer is not connected")
        self.api.set_i32(self.handle, self.api.require_symbol(register_name), value)

    def _set_i64_symbol(self, register_name: str, value: int) -> None:
        if self.handle is None:
            raise DigitizerError("Spectrum digitizer is not connected")
        self.api.set_i64(self.handle, self.api.require_symbol(register_name), value)

    @staticmethod
    def _allocate_page_aligned_buffer(n_bytes: int) -> c_char:
        """Allocate a ctypes buffer aligned to a 4096-byte page boundary for DMA."""
        alignment = 4096
        mask = alignment - 1
        raw = (c_char * (n_bytes + mask))()
        offset = (alignment - (addressof(raw) & mask)) % alignment
        return (c_char * n_bytes).from_buffer(raw, offset)
