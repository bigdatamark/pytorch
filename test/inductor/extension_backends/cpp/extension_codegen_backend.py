# [MODULE DOC] HELPER: Test-only extension backend for Inductor that reuses the C++ CPU
# codegen. Provides a Python wrapper codegen, a C++ wrapper codegen and a scheduling class that
# treat the fake "extension_device" device as an ordinary CPU device. No import-time
# side effects.
from torch._inductor import ir
from torch._inductor.codegen import cpp, cpp_wrapper_cpu, wrapper
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V


class ExtensionWrapperCodegen(wrapper.PythonWrapperCodegen):
    # [CLASS DOC] HELPER: Python wrapper codegen for the extension device. Behaves like the
    # standard Python wrapper, but remaps kernel calls on "extension_device" to the CPU device.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the parent constructor unchanged.
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ):
        # [FUNCTION DOC] HELPER: Factory used by Inductor to build the wrapper. Ignores every
        # argument (subgraph flag, name, parent wrapper, partition signatures) and returns a new
        # default-constructed ExtensionWrapperCodegen.
        return ExtensionWrapperCodegen()

    def _generate_kernel_call_helper(
        self, kernel_name, call_args, *, device=None, **kwargs
    ):
        # [FUNCTION DOC] HELPER: Emits a kernel call via the parent implementation. If no device
        # is given, uses the current graph device (raises if none is set). A device of type
        # "extension_device" is replaced by the CPU device before delegating; nothing is returned.
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "extension_device":
            import torch

            device = torch.device("cpu")
        super()._generate_kernel_call_helper(
            kernel_name, call_args, device=device, **kwargs
        )


class ExtensionCppWrapperCodegen(cpp_wrapper_cpu.CppWrapperCpu):
    # [CLASS DOC] HELPER: C++ wrapper codegen (based on the CPU C++ wrapper) for the extension
    # device; it maps "extension_device" to the CPU when looking up device include paths.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the parent constructor unchanged.
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ):
        # [FUNCTION DOC] HELPER: Factory used by Inductor. Ignores all arguments and returns a
        # new default-constructed ExtensionCppWrapperCodegen.
        return ExtensionCppWrapperCodegen()

    @staticmethod
    def get_device_include_path_jit(device: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the JIT include path text from the CPU C++ wrapper,
        # asking for the CPU when the device string is "extension_device", else for the
        # device given.
        if device == "extension_device":
            return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path_jit("cpu")
        return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path_jit(device)

    @staticmethod
    def get_device_include_path_aot(device: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the ahead-of-time (AOT) include path text from the
        # CPU C++ wrapper, asking for the CPU when the device is "extension_device", else
        # for the device given.
        if device == "extension_device":
            return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path_aot("cpu")
        return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path_aot(device)


class ExtensionScheduling(BaseScheduling):
    # [CLASS DOC] HELPER: Scheduling backend for the extension device. Wraps a CPU C++
    # scheduling object (held in the _scheduling attribute) and delegates node codegen and flush
    # to it; it always allows vertical and horizontal fusion and generates no template code.
    def __init__(self, scheduler):
        # [FUNCTION DOC] HELPER: Calls the parent constructor and creates the wrapped CPU C++
        # scheduling object for the given scheduler.
        super().__init__(scheduler)
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        # [FUNCTION DOC] HELPER: Fusion hook; always answers true regardless of the two nodes.
        return True

    def can_fuse_horizontal(self, node1, node2):
        # [FUNCTION DOC] HELPER: Fusion hook; always answers true regardless of the two nodes.
        return True

    def group_fn(self, sizes):
        # [FUNCTION DOC] HELPER: Normalizes loop sizes: returns a tuple with one tuple per entry
        # of sizes, where each size expression is simplified by the graph's size variables.
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_template(self, template_node, epilogue_nodes):
        # [FUNCTION DOC] HELPER: Template codegen is not supported; does nothing and returns None.
        pass

    def codegen_node(self, node):
        # [FUNCTION DOC] HELPER: Generates code for one scheduler node by delegating to the
        # wrapped CPU C++ scheduling object.
        self._scheduling.codegen_node(node)

    def codegen_sync(self):
        # [FUNCTION DOC] HELPER: No-op synchronization hook.
        pass

    def flush(self):
        # [FUNCTION DOC] HELPER: Flushes pending generated code by delegating to the wrapped CPU
        # C++ scheduling object.
        self._scheduling.flush()
