# [MODULE DOC] Per-device context registry for lazy tensors: DeviceContext objects are created on demand
# and cached in a lock-protected class-level mapping keyed by device string.
import threading

import torch._C._lazy


class DeviceContext:
    # [CLASS DOC] Holds the lazy device name and owns the class-level mapping of device string to context
    # (_CONTEXTS) with its lock (_CONTEXTS_LOCK).
    _CONTEXTS: dict[str, "DeviceContext"] = {}
    _CONTEXTS_LOCK = threading.Lock()

    def __init__(self, device: str) -> None:
        # [FUNCTION DOC] Stores the device string.
        self.device = device


def get_device_context(device: str | None = None) -> DeviceContext:
    # [FUNCTION DOC] Returns the singleton DeviceContext for a device (string-converted; None means the lazy default
    # device type). Creates and registers it under the lock if missing.
    if device is None:
        device = torch._C._lazy._get_default_device_type()
    else:
        device = str(device)
    with DeviceContext._CONTEXTS_LOCK:
        devctx = DeviceContext._CONTEXTS.get(device, None)
        if devctx is None:
            devctx = DeviceContext(device)
            DeviceContext._CONTEXTS[device] = devctx
        return devctx
