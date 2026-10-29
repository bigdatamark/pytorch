# [MODULE DOC] Dynamo integration for the openreg backend: a trivial graph compiler backend
# named "openreg" and a DeviceInterface for the device, both registered at import time
# (register_interface_for_device and register_backend). Requires the native _C module.
import logging

import torch
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.device_interface import (
    DeviceInterface,
    register_interface_for_device,
)

import torch_openreg._C  # type: ignore[misc]


log = logging.getLogger(__name__)


def openreg(gm, example_inputs):
    # [FUNCTION DOC] Dynamo backend compiler: validates the graph module, then returns it
    # uncompiled. Lints the graph, raises an error if any placeholder's fake tensor lives on a
    # device other than openreg or cpu, removes dead code, recompiles the module, logs the
    # generated source at debug level, and returns the module's forward callable.
    # example_inputs is unused. Mutates gm in place.
    gm.graph.lint()

    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake = node.meta["val"]
            if isinstance(fake, torch.Tensor):
                if fake.device.type not in ("openreg", "cpu"):
                    raise AssertionError(
                        f"Unexpected device {fake.device} in openreg backend"
                    )

    gm.graph.eliminate_dead_code()
    gm.recompile()

    code = gm.graph.python_code("self")
    log.debug("Compiled graph source:\n%s", code.src)

    return gm.forward


class OpenRegInterface(DeviceInterface):
    # [CLASS DOC] Dynamo DeviceInterface for the openreg device. Stateless; all methods are
    # static and delegate device selection/queries to the native _C module. Event and Stream
    # are stubs that cannot be instantiated; Worker offers device selection only.
    class Event:
        # [CLASS DOC] HELPER: Placeholder event type; instantiation is not supported.
        def __new__(cls, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises NotImplementedError.
            raise NotImplementedError

    class Stream:
        # [CLASS DOC] HELPER: Placeholder stream type; instantiation is not supported.
        def __new__(cls, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises NotImplementedError.
            raise NotImplementedError

    class Worker:
        # [CLASS DOC] HELPER: Minimal worker namespace exposing device get/set helpers.
        @staticmethod
        def set_device(device):
            # [FUNCTION DOC] HELPER: Makes the given device index current via the native module.
            torch_openreg._C._set_device(device)

        @staticmethod
        def current_device():
            # [FUNCTION DOC] HELPER: Returns the current device index from the native module.
            return torch_openreg._C._get_device()

        @staticmethod
        def get_device_properties(device=None):
            # [FUNCTION DOC] HELPER: Not supported; always raises NotImplementedError.
            raise NotImplementedError

    @staticmethod
    def current_device():
        # [FUNCTION DOC] Returns the current openreg device index from the native module.
        return torch_openreg._C._get_device()

    @staticmethod
    def set_device(device):
        # [FUNCTION DOC] Makes the given device index current. Side effect: changes the
        # native current device.
        torch_openreg._C._set_device(device)

    @staticmethod
    def exchange_device(device):
        # [FUNCTION DOC] Sets the current device and returns the previously current index.
        return torch_openreg._C._exchangeDevice(device)

    @staticmethod
    def maybe_exchange_device(device):
        # [FUNCTION DOC] Like exchange_device but a negative index means "no change": returns
        # -1 without touching the device. Otherwise returns the previous device index.
        if device < 0:
            return -1
        return torch_openreg._C._exchangeDevice(device)

    @staticmethod
    def device_count():
        # [FUNCTION DOC] Returns the number of openreg devices reported by the native module.
        return torch_openreg._C._get_device_count()

    @staticmethod
    def is_available():
        # [FUNCTION DOC] Always reports the device as available.
        return True

    @staticmethod
    def synchronize(device=None):
        # [FUNCTION DOC] No-op; the argument is ignored.
        pass

    @staticmethod
    def get_raw_stream(device_idx):
        # [FUNCTION DOC] Returns a fixed raw stream handle of zero for any device index.
        return 0


register_interface_for_device("openreg", OpenRegInterface)

register_backend(compiler_fn=openreg, name="openreg")
