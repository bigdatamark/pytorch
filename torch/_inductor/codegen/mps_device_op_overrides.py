# [MODULE DOC] Defines MPSDeviceOpOverrides (codegen snippets for MPS device guard, set_device, kernel driver include,
# kernel handle type). Import-time side effect: registers an instance under the device name "mps".
from __future__ import annotations

from .common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    # [CLASS DOC] Device op overrides for the MPS backend: single-device, so guards and set-device
    # are no-ops in generated code. Stateless.
    # MPS is single-device, so a non-zero index is unexpected. device_idx may be a
    # runtime expr (str) under compile-on-one-rank, but MPS never receives that (it has
    # no current_device_idx_expr override), so guard only the int case.
    def device_guard(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns generated-code text for a no-op context manager (MPS has one device). Raises AssertionError if an integer device_idx is non-zero.
        if isinstance(device_idx, int) and device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns generated-code text for a no-op statement. Raises AssertionError if an integer device_idx is non-zero.
        if isinstance(device_idx, int) and device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "pass  # MPS set device"

    def kernel_driver(self) -> str:
        # [FUNCTION DOC] Returns C++ preamble text that includes the Metal shader library header.
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        # [FUNCTION DOC] Returns the C++ type name used for kernel handles on MPS (MTLFunction_t).
        return "MTLFunction_t"


register_device_op_overrides("mps", MPSDeviceOpOverrides())
