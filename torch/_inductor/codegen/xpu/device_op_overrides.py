# [MODULE DOC] XPU implementation of the inductor DeviceOpOverrides: strings of Python and C++ code that wrapper
# codegen emits for XPU devices (streams, device guards, kernel headers). Import-time side effect: registers an
# instance under the device name xpu.
from __future__ import annotations

from ..common import (
    DeviceIdx,
    DeviceOpOverrides,
    register_device_op_overrides,
    TritonScratchWorkspace,
)


class XPUDeviceOpOverrides(DeviceOpOverrides):
    # [CLASS DOC] Supplies XPU-specific code snippets (as strings) for wrapper codegen. Stateless; each method returns text
    # for Python wrapper code or AOTInductor C++ code.
    def import_get_raw_stream_as(self, name: str) -> str:
        # [FUNCTION DOC] Returns an import statement text that binds the XPU current-raw-stream getter to the given name.
        return f"from torch._C import _xpu_getCurrentRawStream as {name}"

    def set_device(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns code text that sets the current XPU device to the given index.
        return f"torch.xpu.set_device({device_idx})"

    def synchronize(self) -> str:
        # [FUNCTION DOC] Returns code text that synchronizes the current XPU device.
        return "torch.xpu.synchronize()"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns code text that creates an XPU device guard context for the given index.
        return f"torch.xpu._DeviceGuard({device_idx})"

    def current_device_idx_expr(self) -> str:
        # [FUNCTION DOC] Returns code text evaluating to the current XPU device index.
        return "torch.xpu.current_device()"

    def current_stream(self) -> str:
        # [FUNCTION DOC] Returns code text evaluating to the current XPU stream.
        return "torch.xpu.current_stream()"

    def cpp_device_guard(self) -> str:
        # [FUNCTION DOC] Returns the C++ device guard type name (generic at::DeviceGuard).
        return "at::DeviceGuard"

    def cpp_aoti_device_guard(self) -> str:
        # [FUNCTION DOC] Returns the AOTInductor C++ XPU device guard type name.
        return "AOTIXpuGuard"

    def cpp_stream_guard(self) -> str:
        # [FUNCTION DOC] Returns the C++ XPU stream guard type name.
        return "at::xpu::XPUStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        # [FUNCTION DOC] Returns the AOTInductor C++ XPU stream guard type name.
        return "AOTIXpuStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        # [FUNCTION DOC] Returns the C++ function name used to wrap an external stream as an XPU stream.
        return "at::xpu::getStreamFromExternal"

    def kernel_header(self) -> str:
        # [FUNCTION DOC] Returns C++ source text including the SYCL runtime wrapper header for AOTInductor kernels.
        source_codes = """
        #include <torch/csrc/inductor/aoti_runtime/sycl_runtime_wrappers.h>
        """
        return source_codes

    def kernel_driver(self) -> str:
        # [FUNCTION DOC] Returns an empty string; XPU needs no kernel driver code.
        return ""

    def cpp_stream_type(self) -> str:
        # [FUNCTION DOC] Returns the C++ type name for a stream (pointer to a SYCL queue).
        return "sycl::queue*"

    def aoti_get_stream(self) -> str:
        # [FUNCTION DOC] Returns the name of the AOTI C API function that fetches the current XPU stream.
        return "aoti_torch_get_current_xpu_stream"

    def cpp_kernel_type(self) -> str:
        # [FUNCTION DOC] Returns the C++ type name for a loaded kernel (unique pointer to a SYCL kernel).
        return "std::unique_ptr<sycl::kernel>"

    def cpp_device_ptr(self) -> str:
        # [FUNCTION DOC] Returns the C++ type name for device pointers (void pointer).
        return "void *"

    def cpp_scratch(
        self, idx: int, workspace: TritonScratchWorkspace, prefix: str | None = None
    ) -> tuple[list[str], str] | None:
        # [FUNCTION DOC] Returns a pair: a list with one C++ declaration of a null global scratch pointer named with idx, and
        # that variable name. The workspace and prefix arguments are ignored (never returns None).
        return [f"void *global_scratch_{idx} = 0;"], f"global_scratch_{idx}"


register_device_op_overrides("xpu", XPUDeviceOpOverrides())
