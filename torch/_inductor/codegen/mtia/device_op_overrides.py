# [MODULE DOC] Defines MTIADeviceOpOverrides (codegen snippets for MTIA device/stream operations).
# Import-time side effect: registers an instance under the device name "mtia".
from __future__ import annotations

from ..common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class MTIADeviceOpOverrides(DeviceOpOverrides):
    # [CLASS DOC] Device op overrides for MTIA: each method returns generated-code text that calls the torch.mtia API. Stateless.
    def import_get_raw_stream_as(self, name: str) -> str:
        # [FUNCTION DOC] Returns import statement text binding the MTIA current-raw-stream getter to the given name.
        return f"from torch._C import _mtia_getCurrentRawStream as {name}"

    def set_device(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns generated-code text that sets the current MTIA device to device_idx.
        return f"torch.mtia.set_device({device_idx})"

    def synchronize(self) -> str:
        # [FUNCTION DOC] Returns generated-code text that synchronizes the MTIA device.
        return "torch.mtia.synchronize()"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns generated-code text creating an MTIA device context manager for device_idx.
        return f"torch.mtia.device({device_idx})"

    def current_stream(self) -> str:
        # [FUNCTION DOC] Returns generated-code text that fetches the current MTIA stream.
        return "torch.mtia.current_stream()"


register_device_op_overrides("mtia", MTIADeviceOpOverrides())
