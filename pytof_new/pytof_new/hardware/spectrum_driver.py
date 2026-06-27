"""Lazy low-level wrapper for the Spectrum driver API.

This module must remain import-safe. It does not import ``pyspcm`` until a
real driver call is requested.
"""

from __future__ import annotations

from ctypes import byref, create_string_buffer, c_int32, c_int64
import importlib
from types import ModuleType
from typing import Any

from pytof_new.exceptions import DigitizerError


class SpectrumDriverError(DigitizerError):
    """Raised when the Spectrum driver reports an error."""

    def __init__(self, operation: str, code: int, driver_text: str = "") -> None:
        self.operation = operation
        self.code = code
        self.driver_text = driver_text
        suffix = f": {driver_text}" if driver_text else ""
        super().__init__(f"Spectrum driver call failed during {operation} with code {code}{suffix}")


class SpectrumDriverApi:
    """Small adapter around pyspcm functions with typed error handling."""

    def __init__(self, module: ModuleType | Any | None = None) -> None:
        self._module = module

    @property
    def loaded(self) -> bool:
        """Return True once the vendor wrapper module has been loaded."""
        return self._module is not None

    @property
    def module(self) -> ModuleType | Any:
        """Return the loaded pyspcm module, importing it on first use."""
        if self._module is None:
            self._module = importlib.import_module("pyspcm")
        return self._module

    def require_symbol(self, name: str) -> Any:
        """Return a Spectrum symbol or fail with a clear driver error."""
        try:
            return getattr(self.module, name)
        except AttributeError as exc:
            raise DigitizerError(f"Spectrum API symbol is not available: {name}") from exc

    def has_symbol(self, name: str) -> bool:
        """Return whether a Spectrum symbol exists in the loaded API."""
        return hasattr(self.module, name)

    def open(self, device: str = "/dev/spcm0") -> Any:
        """Open a Spectrum card handle."""
        handle = self.require_symbol("spcm_hOpen")(create_string_buffer(device.encode("ascii")))
        if handle is None:
            raise DigitizerError(f"could not open Spectrum device {device}")
        return handle

    def close(self, handle: Any) -> None:
        """Close a Spectrum card handle."""
        self.require_symbol("spcm_vClose")(handle)

    def get_i32(self, handle: Any, register: int) -> int:
        """Read a 32-bit Spectrum register."""
        value = c_int32(0)
        code = self.require_symbol("spcm_dwGetParam_i32")(handle, register, byref(value))
        self.check(code, "get_i32", handle)
        return int(value.value)

    def get_i64(self, handle: Any, register: int) -> int:
        """Read a 64-bit Spectrum register."""
        value = c_int64(0)
        code = self.require_symbol("spcm_dwGetParam_i64")(handle, register, byref(value))
        self.check(code, "get_i64", handle)
        return int(value.value)

    def set_i32(self, handle: Any, register: int, value: int) -> None:
        """Write a 32-bit Spectrum register."""
        code = self.require_symbol("spcm_dwSetParam_i32")(handle, register, int(value))
        self.check(code, "set_i32", handle)

    def set_i64(self, handle: Any, register: int, value: int) -> None:
        """Write a 64-bit Spectrum register."""
        code = self.require_symbol("spcm_dwSetParam_i64")(handle, register, int(value))
        self.check(code, "set_i64", handle)

    def command(self, handle: Any, command_value: int) -> None:
        """Write a command value to SPC_M2CMD."""
        self.set_i32(handle, self.require_symbol("SPC_M2CMD"), command_value)

    def def_transfer(
        self,
        handle: Any,
        buffer_kind: int,
        direction: int,
        notify_size: int,
        buffer: Any,
        offset: int,
        length: int,
    ) -> None:
        """Define a DMA transfer buffer."""
        code = self.require_symbol("spcm_dwDefTransfer_i64")(
            handle,
            buffer_kind,
            direction,
            notify_size,
            buffer,
            offset,
            length,
        )
        self.check(code, "def_transfer", handle)

    def read_error_text(self, handle: Any) -> str:
        """Read and clear detailed Spectrum driver error text."""
        error_text_len = int(getattr(self.module, "ERRORTEXTLEN", 1024))
        buffer = create_string_buffer(error_text_len)
        try:
            self.require_symbol("spcm_dwGetErrorInfo_i32")(handle, None, None, buffer)
        except Exception:
            return ""
        return buffer.value.decode("utf-8", errors="replace")

    def check(self, code: int, operation: str, handle: Any | None = None) -> None:
        """Raise a typed exception when a Spectrum call returns an error."""
        error_code = int(code)
        ok_code = int(getattr(self.module, "ERR_OK", 0))
        if error_code == ok_code:
            return
        driver_text = self.read_error_text(handle) if handle is not None else ""
        raise SpectrumDriverError(operation, error_code, driver_text)
