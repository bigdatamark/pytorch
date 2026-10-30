# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that an out-of-tree device backend can be registered with inductor using
# Triton scheduling. Imports the example extension backend (device interface, codegen classes
# and a custom triton heuristics module) from the extension_backends directory, with a
# fallback import path for fbcode. Defines requires_triton_backend (skip decorator), a hash
# mock helper, and TritonExtensionBackendTests. Imports triton only when a triton CPU or GPU
# backend exists, and sets TRITON_DEVICE_TYPE to cpu or GPU_TYPE accordingly. The main block
# runs tests only when a CPU is available and the platform is not macOS.
import functools
import random
import string
import sys
import unittest
from pathlib import Path
from typing import Any

import torch
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config


try:
    from extension_backends.triton.device_interface import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:device_interface
        DeviceInterface,
    )
    from extension_backends.triton.extension_codegen_backend import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_codegen_backend
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
    from extension_backends.triton.extension_triton_heuristics import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_triton_heuristics
        EXTENSION_TRITON_META_FIELD,
    )
except ImportError:
    from .extension_backends.triton.device_interface import DeviceInterface
    from .extension_backends.triton.extension_codegen_backend import (
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
    from .extension_backends.triton.extension_triton_heuristics import (
        EXTENSION_TRITON_META_FIELD,
    )

import torch._inductor.lowering as inductor_lowering
from torch._C import FileCheck
from torch._dynamo import device_interface
from torch._inductor import codegen, ir, metrics
from torch._inductor.codegen import common
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    IndentedBuffer,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.utils import get_triton_code, run_and_get_triton_code
from torch.testing._internal.common_utils import IS_FBCODE, IS_MACOS
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU_AND_TRITON,
    TRITON_HAS_CPU,
)


try:
    from .test_extension_backend import BaseExtensionBackendTests
except ImportError:
    from test_extension_backend import BaseExtensionBackendTests

if TRITON_HAS_CPU or HAS_GPU_AND_TRITON:
    import triton
    import triton.language as tl

    if TRITON_HAS_CPU:
        TRITON_DEVICE_TYPE = "cpu"
    else:
        TRITON_DEVICE_TYPE = GPU_TYPE

requires_triton_backend = unittest.skipUnless(
    HAS_GPU_AND_TRITON or TRITON_HAS_CPU, "Requires Triton backend."
)


def mock_triton_hash_with_backend(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the triton hash function that ignores its
    # arguments and returns a random 64 character string of uppercase letters and digits, so
    # compilation can proceed without a real triton backend.
    # Generate a random string of length 64. Used to mock the triton_hash_with_backend function
    # since we don't have a triton backend
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=64))


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class TritonExtensionBackendTests(BaseExtensionBackendTests):
    # [CLASS DOC] Tests registering a privateuseone device with a Triton-based scheduling and
    # wrapper codegen, and a custom triton heuristics module. Skipped in fbcode. Inherits the
    # extension module setup from BaseExtensionBackendTests; class setup/teardown save and
    # restore inductor global backend registrations and lowerings.
    """
    Test creating a backend for inductor with Triton scheduling.
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture. Calls the parent setup, skips the whole class when
        # cpp_wrapper is enabled (CppWrapperCpu has no triton CPU support), re-initializes the
        # default backend registration, and starts a patch that snapshots the lowerings dict,
        # stored on the class so teardown can stop it.
        super().setUpClass()
        if config.cpp_wrapper:
            raise unittest.SkipTest(
                "Not possible to fix until CppWrapperCpu supports triton for CPU"
            )

        # Store the default backends and reset later
        common.init_backend_registration()

        default_backend_patch = unittest.mock.patch.dict(inductor_lowering.lowerings)
        default_backend_patch.start()
        cls._default_backend_patch = default_backend_patch

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class fixture. Calls the parent teardown and stops the lowerings
        # patch started in setUpClass, restoring the default lowerings.
        super().tearDownClass()

        # Restore the default backend.
        cls._default_backend_patch.stop()

    def test_open_device_registration(self):
        # [FUNCTION DOC] Registers the extension module as the privateuseone device module,
        # plus its scheduling/wrapper codegen, device op overrides and device interface, and
        # asserts each lookup returns the registered object. Then compiles a small function
        # (sin plus min) on a privateuseone tensor with the triton hash mocked, and uses
        # FileCheck to confirm the generated code imports triton, has a jit kernel, uses the
        # math sin, and names the privateuseone device.
        torch._register_device_module("privateuseone", self.module)
        register_backend_for_device(
            "privateuseone", ExtensionScheduling, ExtensionWrapperCodegen
        )
        register_device_op_overrides("privateuseone", CPUDeviceOpOverrides())
        device_interface.register_interface_for_device("privateuseone", DeviceInterface)

        self.assertEqual(
            get_scheduling_for_device("privateuseone"), ExtensionScheduling
        )
        self.assertEqual(
            get_wrapper_codegen_for_device("privateuseone"), ExtensionWrapperCodegen
        )
        self.assertEqual(
            device_interface.get_interface_for_device("privateuseone"), DeviceInterface
        )

        device = torch.device("privateuseone")
        x = torch.empty(2, 16).fill_(1).to(device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns elementwise sine of x plus
            # the minimum of x.
            return torch.sin(x) + x.min()

        metrics.reset()
        opt_fn = torch.compile(foo)

        # Since we don't have a triton backend, we need to mock the triton_hash_with_backend
        # function
        with unittest.mock.patch(
            "torch.utils._triton.triton_hash_with_backend",
            new=mock_triton_hash_with_backend,
        ):
            code = get_triton_code(opt_fn, x)

        FileCheck().check("import triton").check("@triton.jit").check(
            "tl_math.sin"
        ).check("device_str='privateuseone'").run(code)

    def _register_custom_backend_with_heuristics(self, device):
        # [FUNCTION DOC] HELPER: Registers for the given device a Triton scheduling and Python
        # wrapper codegen that use a custom kernel class pulling in the extension heuristics
        # module. Adds the extension_backends/triton directory to sys.path if missing (so the
        # autotuner class can be unpickled from the compile subprocess), with a cleanup that
        # removes it afterwards.
        path_to_ext_heuristics = str(
            Path(__file__).parent / "extension_backends" / "triton"
        )
        # Add the path to sys.path in the parent process so that the
        # ExtensionCachingAutotuner class (defined in extension_triton_heuristics)
        # can be resolved when the compiled kernel is unpickled from the
        # compile subprocess back into the parent process.
        if path_to_ext_heuristics not in sys.path:
            sys.path.append(path_to_ext_heuristics)
            self.addCleanup(sys.path.remove, path_to_ext_heuristics)

        class ExtensionTritonKernel(codegen.triton.TritonKernel):
            # [CLASS DOC] HELPER: TritonKernel subclass that makes generated code import the
            # extension triton heuristics module as the heuristics module and adds the
            # extension marker field (set to true) to the common triton metadata.
            @classmethod
            @functools.lru_cache(None)
            def gen_common_triton_imports(cls) -> str:
                # [FUNCTION DOC] HELPER: Cached class method returning the default triton
                # import block with an extra line importing extension_triton_heuristics under
                # the name triton_heuristics.
                default_imports = super().gen_common_triton_imports()
                custom_imports = IndentedBuffer()
                custom_imports.splice(default_imports)

                custom_imports.splice("""
                    import extension_triton_heuristics as triton_heuristics
                """)
                return custom_imports

            @classmethod
            def triton_meta_common(cls) -> dict[str, Any]:
                # [FUNCTION DOC] HELPER: Returns the parent common triton metadata mapping
                # with the EXTENSION_TRITON_META_FIELD key set to true.
                triton_meta = super().triton_meta_common()
                triton_meta[EXTENSION_TRITON_META_FIELD] = True
                return triton_meta

        class ExtensionTritonScheduling(codegen.triton.TritonScheduling):
            # [CLASS DOC] HELPER: TritonScheduling subclass whose kernel type is the extension
            # kernel.
            kernel_type = ExtensionTritonKernel

        class ExtensionPythonWrapperCodegen(PythonWrapperCodegen):
            # [CLASS DOC] HELPER: PythonWrapperCodegen subclass that reports the extension
            # kernel as the triton kernel class used for triton info and builds itself for the
            # top-level graph.
            @classmethod
            def _get_triton_info_kernel_cls(cls) -> type[codegen.triton.TritonKernel]:
                # [FUNCTION DOC] HELPER: Class method returning the extension kernel class.
                return ExtensionTritonKernel

            @staticmethod
            def create(
                is_subgraph: bool,
                subgraph_name: str | None,
                parent_wrapper: PythonWrapperCodegen | None,
                partition_signatures: ir.GraphPartitionSignature | None = None,
            ):
                # [FUNCTION DOC] HELPER: Factory for wrapper codegen. For a subgraph, raises
                # AssertionError if the subgraph name or parent wrapper is missing, and
                # delegates to the base PythonWrapperCodegen factory; otherwise returns a new
                # extension wrapper instance.
                if is_subgraph:
                    if subgraph_name is None:
                        raise AssertionError
                    if parent_wrapper is None:
                        raise AssertionError
                    return PythonWrapperCodegen.create(
                        subgraph_name, parent_wrapper, partition_signatures
                    )
                return ExtensionPythonWrapperCodegen()

        register_backend_for_device(
            device, ExtensionTritonScheduling, ExtensionPythonWrapperCodegen
        )

    @requires_triton_backend
    def test_codegen_with_custom_heuristics_module(self):
        # [FUNCTION DOC] Needs a triton backend. Registers the custom heuristics backend for
        # the triton device type, compiles an elementwise add on GPU_TYPE tensors, and uses
        # FileCheck on the generated code to confirm it imports the extension heuristics
        # module, includes the extension meta field, and contains a jit kernel.
        self._register_custom_backend_with_heuristics(TRITON_DEVICE_TYPE)

        def add(x, y):
            # [FUNCTION DOC] HELPER: Function under test; returns the sum of x and y.
            return x + y

        x = torch.zeros((32,), device=GPU_TYPE)
        y = x
        compiled_add = torch.compile(add)

        code = run_and_get_triton_code(compiled_add, x, y)
        FileCheck().check("import extension_triton_heuristics").check(
            f"{EXTENSION_TRITON_META_FIELD}"
        ).check("@triton.jit").run(code)

    @requires_triton_backend
    def test_codegen_with_custom_heuristics_module_udtk(self):
        # [FUNCTION DOC] Needs a triton backend. Registers the custom heuristics backend, then
        # compiles a function that launches a user-defined triton kernel, and uses FileCheck
        # to confirm the generated code imports the extension heuristics module and has a jit
        # kernel.
        self._register_custom_backend_with_heuristics(TRITON_DEVICE_TYPE)

        # [FUNCTION DOC] HELPER: User-defined triton add kernel (doc placed above the
        # decorator). Each program loads a BLOCK_SIZE chunk of both inputs under a bounds
        # mask, adds them and stores into the output pointer.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocates an output like x, launches the triton add
            # kernel over a grid sized by cdiv of the element count and BLOCK_SIZE 16, and
            # returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Grid callback returning a one-element launch grid
                # holding the number of blocks needed to cover n_elements at the meta
                # BLOCK_SIZE.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        args = [torch.randn(32, device=GPU_TYPE) for _ in range(2)]
        code = run_and_get_triton_code(torch.compile(add), *args)

        FileCheck().check("import extension_triton_heuristics").check(
            "@triton.jit"
        ).run(code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU and not IS_MACOS:
        run_tests()
