# [MODULE DOC] Registers the CPU DeviceOpOverrides, supplying trivial code snippets (no streams, no device
# switching) used by Inductor code generation when targeting cpu.
# Import-time side effect: registers the overrides under the name cpu.
from __future__ import annotations

from textwrap import dedent

from .common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class CpuDeviceOpOverrides(DeviceOpOverrides):
    # [CLASS DOC] DeviceOpOverrides for cpu; every method returns a string of source code to be emitted by codegen.
    def import_get_raw_stream_as(self, name: str) -> str:
        # [FUNCTION DOC] Return Python source defining get_raw_stream that ignores its argument and returns 0 (name arg unused).
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def cpp_kernel_type(self) -> str:
        # [FUNCTION DOC] Return the C++ kernel handle type string for cpu kernels.
        return "void*"

    def set_device(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Return a statement that does nothing; there is no device to set on cpu.
        return "pass"

    def synchronize(self) -> str:
        # [FUNCTION DOC] Return a statement that does nothing; cpu needs no synchronization.
        return "pass"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Return an expression string for a null context manager used as a device guard on cpu.
        return "torch._ops.contextlib.nullcontext()"


register_device_op_overrides("cpu", CpuDeviceOpOverrides())
