# [MODULE DOC] HELPER: Test-only extension backend for Inductor built on the Triton codegen.
# Defines a Python wrapper codegen, a scheduling class delegating to the Triton scheduling,
# and a device-op-overrides class with do-nothing/empty-string behavior. No import-time effects.
from torch._inductor.codegen import triton, wrapper
from torch._inductor.codegen.common import DeviceOpOverrides
from torch._inductor.scheduler import BaseScheduling


class ExtensionWrapperCodegen(wrapper.PythonWrapperCodegen):
    # [CLASS DOC] HELPER: Plain subclass of the Python wrapper codegen; adds no behavior.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the parent constructor unchanged.
        super().__init__(*args, **kwargs)


class ExtensionScheduling(BaseScheduling):
    # [CLASS DOC] HELPER: Scheduling backend that wraps a Triton scheduling object (kept in the
    # _triton_scheduling attribute) and delegates grouping, node codegen and flush to it. Fusion
    # is always permitted and template codegen is a no-op.
    def __init__(self, scheduler):
        # [FUNCTION DOC] HELPER: Calls the parent constructor and builds the wrapped Triton
        # scheduling object for the given scheduler.
        super().__init__(scheduler)
        self._triton_scheduling = triton.TritonScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        # [FUNCTION DOC] HELPER: Fusion hook; always answers true regardless of the two nodes.
        return True

    def can_fuse_horizontal(self, node1, node2):
        # [FUNCTION DOC] HELPER: Fusion hook; always answers true regardless of the two nodes.
        return True

    def group_fn(self, sizes):
        # [FUNCTION DOC] HELPER: Returns the result of the wrapped Triton scheduling's own
        # grouping function for the given sizes.
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(self, template_node, epilogue_nodes):
        # [FUNCTION DOC] HELPER: Template codegen is not supported; does nothing and returns None.
        pass

    def codegen_node(self, node):
        # [FUNCTION DOC] HELPER: Generates code for one scheduler node by delegating to the
        # wrapped Triton scheduling object.
        self._triton_scheduling.codegen_node(node)

    def codegen_sync(self):
        # [FUNCTION DOC] HELPER: No-op synchronization hook.
        pass

    def flush(self):
        # [FUNCTION DOC] HELPER: Flushes pending generated code by delegating to the wrapped
        # Triton scheduling object.
        self._triton_scheduling.flush()


class CPUDeviceOpOverrides(DeviceOpOverrides):
    # [CLASS DOC] HELPER: Device op overrides for the fake CPU-like extension device. Every hook
    # produces an empty result so that no real device code is emitted.
    def import_get_raw_stream_as(self, name: str) -> str:
        # [FUNCTION DOC] HELPER: Returns source text for a stub raw-stream getter with the given
        # name that takes one argument and does nothing (it returns no stream).
        return f"def {name}(name): None\n"

    def set_device(self, device_idx: int) -> str:
        # [FUNCTION DOC] HELPER: Returns an empty string; no set-device code is generated.
        return ""

    def synchronize(self) -> None:
        # [FUNCTION DOC] HELPER: No-op; generates nothing and returns None.
        pass

    def device_guard(self, device_idx: int) -> str:
        # [FUNCTION DOC] HELPER: Returns an empty string; no device guard code is generated.
        return ""
