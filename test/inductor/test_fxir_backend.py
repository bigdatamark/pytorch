# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the FX IR inductor backend (the wrapper code generator that emits a
# torch.fx graph module instead of Python or C++ wrapper source). Defines a base test class that
# swaps the device backend registration for the GPU device to the Triton scheduler plus the FX
# wrapper code generator, compiles small functions, captures the produced FX graph modules and
# inspects their nodes (Triton kernel calls, allocations, views, frees, fallbacks, subgraphs,
# symbolic shapes). Import time: imports the control flow test models (with a fallback import
# path), imports Triton only when a GPU exists, and defines a shared inductor config used by the
# tests that disables runtime asserts and compile threads.
"""
Test the FX IR backend.
"""

import itertools
import operator
import unittest
from collections.abc import Callable
from unittest.mock import MagicMock

import sympy

import torch
import torch._inductor.codegen.common as common
import torch.utils._pytree as pytree
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.utils import same
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_mutation,
)
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, UnbackedSymbolDefsLine
from torch._inductor.codegen.wrapper_fxir import (
    FxConverter,
    replace_floor_div,
    WrapperFxCodegen,
)
from torch._inductor.exc import InductorError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch.export import Dim
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    patch_custom_fallback_pass,
    requires_gpu,
)
from torch.utils._sympy.functions import FloorDiv


try:
    from .test_control_flow import CondModels, SwitchModels
except ImportError:
    from test_control_flow import (
        CondModels,  # @manual=fbcode//caffe2/test/inductor:control_flow-library
        SwitchModels,  # @manual=fbcode//caffe2/test/inductor:control_flow-library
    )

if HAS_GPU:
    import triton
    import triton.language as tl

    from torch.testing._internal.triton_utils import add_kernel_2d_autotuned

test_config = {
    "compile_threads": 1,
    "alignment_asserts": False,
    "size_asserts": False,
    "scalar_asserts": False,
    "nan_asserts": False,
}


@requires_gpu()
@config.patch(test_config)
@instantiate_parametrized_tests
class FxirTestCase(InductorTestCase):
    # [CLASS DOC] Main FX backend test case; requires a GPU, patches inductor config with the
    # shared test config, and expands parametrized tests. The class setup registers the FX wrapper
    # codegen for the GPU device (remembering the default registration) and the teardown restores
    # it. Provides helpers to count node targets, capture the generated graph modules and compile-
    # and-check a function against eager; all test methods use them.
    device = GPU_TYPE

    def _count_ops(self, gm: torch.fx.GraphModule, target: Callable) -> int:
        # [FUNCTION DOC] HELPER: Returns how many call_function nodes in the graph module's graph
        # have exactly the given target.
        return len(gm.graph.find_nodes(op="call_function", target=target))

    def _run_and_capture_graphs(self, opt, args) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Runs the given compiled callable on the arguments while
        # temporarily replacing the FX converter generate method with a wrapper that records every
        # graph module it returns. Returns the list of recorded graph modules (one per generated
        # wrapper, e.g. forward and backward). Despite the annotation it returns a list.
        gms = []

        orig_generate = FxConverter.generate

        def generate(self) -> torch.fx.GraphModule:
            # [FUNCTION DOC] HELPER: Calls the original converter generate method, appends the
            # resulting graph module to the enclosing captured list, and returns it.
            nonlocal gms
            gm = orig_generate(self)
            gms.append(gm)
            return gm

        with unittest.mock.patch.object(
            torch._inductor.codegen.wrapper_fxir.FxConverter, "generate", generate
        ):
            opt(*args)

        return gms

    def _compile_and_check(
        self,
        func,
        args,
        expected_num_triton_kernels: int = 1,
        metadata_only: bool = False,
        compile_kwargs: dict | None = None,
    ):
        # [FUNCTION DOC] HELPER: Compiles func with torch.compile (extra keyword arguments
        # optional), captures the generated FX graphs by running on args, asserts the total number
        # of Triton kernel wrapper nodes across the graphs equals the expected count, then runs
        # compiled and eager again and asserts the results match with the dynamo same helper. With
        # metadata_only, both results are replaced by zero tensors first so only shapes, dtypes
        # and structure are compared. Returns the list of captured graph modules.
        if compile_kwargs is None:
            compile_kwargs = {}

        opt = torch.compile(func, **compile_kwargs)

        # Get the FX graph from the backend.
        gms = self._run_and_capture_graphs(opt, args)

        # Check the code for triton kernels.
        num_kernels = sum(
            self._count_ops(gm, triton_kernel_wrapper_mutation) for gm in gms
        )
        self.assertEqual(num_kernels, expected_num_triton_kernels)

        # Check accuracy.
        result = opt(*args)
        ref = func(*args)
        if metadata_only:
            # When we only want to check metadata, fill in zeros for tensor data.
            ref, result = tuple(
                pytree.tree_map(torch.zeros_like, x) for x in (ref, result)
            )

        self.assertTrue(same(ref, result))

        return gms

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class-level setup: calls the parent setup, initializes backend
        # registration, saves the default device codegen for the class device in a class
        # attribute, and registers the Triton scheduler with the FX wrapper codegen for that
        # device (global side effect until teardown).
        super().setUpClass()

        # Register the FX backend, storing the default for later.
        common.init_backend_registration()
        cls._default_backend = common.device_codegens[cls.device]
        common.register_backend_for_device(
            cls.device, TritonScheduling, WrapperFxCodegen
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class-level teardown: calls the parent teardown and restores the saved
        # default backend for the device in the global device codegen registry.
        super().tearDownClass()

        # Restore the default backend.
        common.device_codegens[cls.device] = cls._default_backend

    def test_basic(self):
        # [FUNCTION DOC] Compiles an addition of two random tensors and checks one Triton kernel
        # and correct results.
        args = [torch.randn(8, device=self.device) for _ in range(2)]
        self._compile_and_check(torch.add, args)

    def test_standard_kernel_omits_empty_launch_kwargs(self):
        # [FUNCTION DOC] Compiles an addition and asserts the single Triton kernel wrapper node
        # does not carry a launch_kwargs keyword, since it is omitted when empty to stay
        # compatible with downstream implementations.
        args = [torch.randn(8, device=self.device) for _ in range(2)]
        (gm,) = self._compile_and_check(torch.add, args)
        (triton_node,) = gm.graph.find_nodes(
            op="call_function", target=triton_kernel_wrapper_mutation
        )

        # launch_kwargs is only needed when user Triton backend options must be
        # replayed. Omitting the empty case keeps standard FXIR HOP calls
        # compatible with downstream py_impls that have fixed keyword-only
        # signatures matching the original HOP payload.
        self.assertNotIn("launch_kwargs", triton_node.kwargs)

    def test_device_type(self):
        # [FUNCTION DOC] Passes inputs on an indexed device and checks that the single
        # empty_strided allocation node uses a device with no index but the same device type.
        """
        Test that we allocate on a device type instead of a specific index.
        """
        # Pass in a tensor on an indexed device.
        device_runtime = getattr(torch, self.device)
        indexed_device = torch.device(self.device, device_runtime.current_device())
        args = [torch.randn(8, device=indexed_device) for _ in range(2)]
        (gm,) = self._compile_and_check(torch.add, args)
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )

        # Check that the device of the output allocation is not indexed.
        output_device = torch.device(empty_strided.kwargs["device"])
        self.assertIs(output_device.index, None)
        self.assertEqual(output_device.type, indexed_device.type)

    def test_multiple_kernels(self):
        # [FUNCTION DOC] Compiles a function summing two inputs separately and adding the sums;
        # expects two Triton kernels.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x plus the sum of y.
            return x.sum() + y.sum()

        args = [torch.randn(length, device=self.device) for length in [517, 1029]]
        self._compile_and_check(foo, args, expected_num_triton_kernels=2)

    def test_free(self):
        # [FUNCTION DOC] Compiles a program with three inputs where intermediate buffers die
        # early; expects three kernels, and asserts the generated code text contains at least one
        # statement that frees a buffer by assigning none.
        """
        Test a program that frees a buffer which is no longer in use.
        """

        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Computes sum of x plus y, then returns the sum of z plus the
            # sum of that result.
            w = x.sum() + y
            return z.sum() + w.sum()

        args = [torch.randn(length, device=self.device) for length in [517, 1029, 123]]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=3)

        # Check the generated code for frees.
        num_frees = gm.code.count("= None")
        self.assertGreater(num_frees, 0)

    def test_extern(self):
        # [FUNCTION DOC] Compiles a matmul plus a sum; expects one Triton kernel and exactly one
        # call to the addmm out variant extern kernel in the graph.
        """
        Test a program that calls an extern kernel.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of x and y plus the sum of y.
            return x @ y + y.sum()

        args = [
            torch.randn(size, device=self.device) for size in [(129, 129), (129, 1)]
        ]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Check for the extern kernel
        num_extern = self._count_ops(gm, torch.ops.aten.addmm.out)
        self.assertEqual(num_extern, 1)

    def test_fallback(self):
        # [FUNCTION DOC] Compiles a program with random batch tensors and an addbmm call, checking
        # metadata only because values are random. Expects two Triton kernels and exactly two
        # fallback calls total (randint low out plus addbmm default).
        """
        Test a program that calls aten fallbacks.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Creates two random batch tensors on the device and returns
            # addbmm of x with them.
            batch1 = torch.randn(2, 3, 5, device=self.device)
            batch2 = torch.randn(2, 5, 4, device=self.device)
            return torch.addbmm(x, batch1, batch2)

        args = (torch.randn(3, 4, device=self.device),)

        # Since the program has a random output, just check metadata.
        # Don't check for an exact value.
        (gm,) = self._compile_and_check(
            foo, args, expected_num_triton_kernels=2, metadata_only=True
        )

        # Check for the fallback kernel.
        num_fallback = self._count_ops(
            gm, torch.ops.aten.randint.low_out
        ) + self._count_ops(gm, torch.ops.aten.addbmm.default)
        self.assertEqual(num_fallback, 2)

    def test_cat_inputs(self):
        # [FUNCTION DOC] Compiles concatenation of two inputs plus one; expects one Triton kernel.
        """
        Test concatenation of graph inputs.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Concatenates x and y and adds 1.
            return torch.cat((x, y)) + 1

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        self._compile_and_check(foo, args, expected_num_triton_kernels=1)

    def test_cat_views(self):
        # [FUNCTION DOC] Compiles a program where two kernels write into slices of the same
        # concatenated buffer. Expects two kernels and exactly two as_strided view nodes, one of
        # which has a nonzero offset.
        """
        Test concatenation with multiple kernels writing to the same buffer.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns a (x minus 2), b (sum of y kept as a row) and c
            # (clone of their concatenation).
            a = x - 2
            b = y.sum(0, keepdim=True)
            c = torch.cat((a, b)).clone()
            return a, b, c

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=2)

        def get_offset(node: torch.fx.Node) -> int:
            # [FUNCTION DOC] HELPER: Returns the integer storage offset of an as_strided node from
            # its fourth positional argument; raises AssertionError if that offset is not an int.
            (input_, shape, stride, offset) = node.args
            if not isinstance(offset, int):
                raise AssertionError
            return offset

        # Check for 2 views, one of which is offset.
        as_strided_nodes = list(
            gm.graph.find_nodes(op="call_function", target=torch.as_strided)
        )
        self.assertEqual(len(as_strided_nodes), 2)
        num_offset_views = sum(get_offset(node) > 0 for node in as_strided_nodes)
        self.assertEqual(num_offset_views, 1)

    def test_cat_to_alloc(self):
        # [FUNCTION DOC] Compiles x plus a concatenation of two aranges, which inductor turns into
        # a single allocation. Expects one kernel and exactly one empty_strided allocation.
        """
        Test concatenation that's optimized out to an allocation.
        """
        length = 8

        def foo(x):
            # [FUNCTION DOC] HELPER: Builds two aranges of half length and returns x plus their
            # concatenation.
            y, z = tuple(
                torch.arange(length // 2, device=self.device) for _ in range(2)
            )
            return x + torch.cat((y, z))

        args = [torch.randn(length, device=self.device)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Expect a single allocation, even though eager mode would use 2.
        num_allocs = self._count_ops(gm, torch.empty_strided)
        self.assertEqual(num_allocs, 1)

    def test_cat_reinterpret_view(self):
        # [FUNCTION DOC] Compiles x plus the concatenation of two random tensors, checking
        # metadata only. Expects three kernels and two as_strided nodes (the reinterpreted views).
        """
        Test torch.cat using ReinterpretView.
        """
        length = 8

        def foo(x):
            # [FUNCTION DOC] HELPER: Creates two random half-length tensors and returns x plus
            # their concatenation.
            y, z = tuple(torch.randn(length // 2, device=self.device) for _ in range(2))
            return x + torch.cat((y, z))

        args = [torch.randn(length, device=self.device)]

        # Since this test generates random numbers, check metadata only.
        (gm,) = self._compile_and_check(
            foo, args, expected_num_triton_kernels=3, metadata_only=True
        )

        # Check for as_strided. We map ReinterpretView to this.
        num_as_strided = self._count_ops(gm, torch.as_strided)
        self.assertEqual(num_as_strided, 2)

    def test_reshape_output(self):
        # [FUNCTION DOC] Compiles an addition followed by a reshape to a flat output, expecting
        # one kernel and one as_strided node (the reinterpreted view of the output).
        """
        Test reshaping the output, which maps to a ReinterpretView.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Adds x and y then reshapes the sum to length eight.
            return torch.reshape(x + y, (8,))

        args = [torch.randn((2, 4), device=self.device) for _ in range(2)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Check for as_strided. We map ReinterpretView to this.
        num_as_strided = self._count_ops(gm, torch.as_strided)
        self.assertEqual(num_as_strided, 1)

    def test_reinterpret_view_floordiv_offset_dynamic(self):
        # [FUNCTION DOC] With dynamic shapes, compiles a narrow-and-contiguous program whose view
        # offset involves floor division of a symbolic size. For each as_strided node whose offset
        # argument is an FX node with a recorded value, asserts that value is not a symbolic float
        # (must be a symbolic int).
        """
        Test that ReinterpretView with a FloorDiv offset emits valid FX IR
        with SymInt metadata when dynamic shapes are enabled.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Narrows x along dim 0 starting at half the size with half the
            # size, makes it contiguous and adds 1.
            n = x.shape[0]
            return x.narrow(0, n // 2, n // 2).contiguous() + 1

        args = [torch.randn(16, 4, device=self.device)]
        (gm,) = self._compile_and_check(
            foo, args, compile_kwargs={"dynamic": True}, metadata_only=True
        )

        as_strided_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.as_strided
        )
        for node in as_strided_nodes:
            offset = node.args[3] if len(node.args) > 3 else None
            if offset is not None and isinstance(offset, torch.fx.Node):
                val = offset.meta.get("val")
                if val is not None:
                    self.assertNotIsInstance(
                        val,
                        torch.SymFloat,
                        "ReinterpretView offset should be SymInt, not SymFloat",
                    )

    def test_reshape_fallback(self):
        # [FUNCTION DOC] Uses the custom fallback pass patch so that every node falls back to
        # aten, compiles a reshape, expects zero Triton kernels and asserts the graph contains
        # exactly one aten reshape default call.
        """
        Test falling back to aten.reshape. This uses a custom pass to enable more fallbacks.
        """

        def always_fallback(node: torch.fx.Node) -> bool:
            # [FUNCTION DOC] HELPER: Fallback predicate that returns true for every node.
            return True

        def foo(x):
            # [FUNCTION DOC] HELPER: Reshapes x to two by five.
            return x.reshape((2, 5))

        args = (torch.randn(10, device=self.device),)
        with patch_custom_fallback_pass(always_fallback):
            (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=0)

        # Check for the reshape.
        (reshape_node,) = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.reshape.default
        )

    def test_extern_multi_output(self):
        # [FUNCTION DOC] Compiles a topk program with two graph outputs. Expects two kernels, two
        # getitem nodes extracting the extern kernel outputs, and an output node with two entries.
        """
        Test an extern kernel with multiple outputs.
        Also test a graph with multiple outputs.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns topk values plus 1 and topk indices times 2, for k
            # equal to two.
            top, idx = torch.topk(x, 2)
            return top + 1, idx * 2

        args = [torch.randn(8, device=self.device)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=2)

        # Check for multiple kernel outputs via getitems.
        num_getitems = self._count_ops(gm, operator.getitem)
        self.assertEqual(num_getitems, 2)

        # Check for multiple graph outputs.
        output_node = gm.graph.find_nodes(op="output")[0]
        self.assertEqual(len(output_node.args[0]), 2)

    def test_duplicate_input(self):
        # [FUNCTION DOC] Compiles an addition where the same tensor is passed twice; expects one
        # kernel and the graph module to have a single placeholder.
        """
        Test duplicated inputs. This will collapse into a single input in the GM.
        """

        args = [torch.randn(4, device=self.device)] * 2
        (gm,) = self._compile_and_check(torch.add, args, expected_num_triton_kernels=1)

        num_placeholders = len(gm.graph.find_nodes(op="placeholder"))
        self.assertEqual(num_placeholders, 1)

    def test_backward(self):
        # [FUNCTION DOC] Compiles a function doing a matmul with global parameters, a bias and a
        # binary cross entropy loss, then calls backward, returning the parameter grads. Expects
        # four kernels in total and exactly two captured graphs (forward and backward); the
        # unpacked graphs are not inspected further.
        """
        Test a program with a backward pass.
        """

        x = torch.ones(5, device=self.device)  # input tensor
        y = torch.zeros(3, device=self.device)  # expected output
        w = torch.randn(5, 3, requires_grad=True, device=self.device)
        b = torch.randn(3, requires_grad=True, device=self.device)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Computes the logistic loss for x times global weight plus
            # global bias against target y, runs backward and returns the weight and bias
            # gradients.
            z = torch.matmul(x, w) + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
            return w.grad, b.grad

        # Expect separate forward and backward graphs.
        (forward_gm, backward_gm) = self._compile_and_check(
            foo, (x, y), expected_num_triton_kernels=4
        )

    def test_custom_compiler(self):
        # [FUNCTION DOC] Registers (via a patched device codegen mapping) a wrapper codegen
        # subclass whose compile step returns a function adding 1 to every output of the graph
        # module. Compiles an addition, then asserts the result differs from eager but equals
        # eager after subtracting the offset.
        """
        Test a derived backend with a custom compiler.
        """
        offset = 1

        class CustomWrapperCodegen(WrapperFxCodegen):
            # [CLASS DOC] HELPER: Wrapper codegen derived from the FX wrapper codegen with a
            # custom compile_graph hook that shifts all outputs.
            def compile_graph(self, gm):
                # [FUNCTION DOC] HELPER: Overrides graph compilation: returns a function running
                # the graph module and adding 1 to every output leaf.
                def compiled_fn(*args):
                    # [FUNCTION DOC] HELPER: Runs the graph module on the arguments and returns
                    # the outputs with 1 added to each leaf.
                    # Adds an offset to the program's outputs.
                    outputs = gm(*args)
                    return pytree.tree_map(lambda x: x + 1, outputs)

                return compiled_fn

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        custom_backend = common.DeviceCodegen(
            TritonScheduling, CustomWrapperCodegen, None
        )
        with unittest.mock.patch.dict(
            common.device_codegens, {self.device: custom_backend}
        ):
            func = torch.add
            opt = torch.compile(func)
            result = opt(*args)

        # Check the output is offset from eager mode.
        ref = func(*args)
        self.assertFalse(same(result, ref))
        self.assertNotEqual(offset, 0)
        self.assertTrue(same(result - offset, ref))

    def test_dynamic_shapes_and_strides(self):
        # [FUNCTION DOC] Compiles an addition of strided views with dynamic shapes. Asserts the
        # single empty_strided allocation has the same number of symbolic dims as the input, the
        # last stride is one, and every symbolic size or leading stride is a symbolic int with a
        # matching placeholder node in the graph.
        """
        Test a graph with dynamic shapes and strides.
        """

        static_dims = (8, 8)

        def get_input():
            # [FUNCTION DOC] HELPER: Creates a larger random tensor and returns a strided view of
            # it with the static dims but the stride of the full tensor.
            full_size = (16, 8)
            full = torch.randn(full_size, device=self.device)
            view = torch.as_strided(full, static_dims, full.stride())
            return view

        func = torch.add
        args = [get_input() for _ in range(2)]
        (gm,) = self._compile_and_check(func, args, compile_kwargs={"dynamic": True})

        # Check for a symbolic output shape.
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )
        example_tensor = empty_strided.meta["val"]
        symbolic_dims = example_tensor.shape
        self.assertEqual(len(symbolic_dims), len(static_dims))

        # Check for symbolic output strides.
        (stride, one) = example_tensor.stride()
        self.assertEqual(one, sympy.S.One)

        # Find the size symbols, and check for a corresponding placeholders defining them.
        for symbol in itertools.chain(symbolic_dims, [stride]):
            self.assertTrue(isinstance(symbol, torch.SymInt))
            (placeholder,) = [
                node
                for node in gm.graph.find_nodes(op="placeholder")
                if node.name == str(symbol)
            ]
            self.assertEqual(placeholder.meta["val"], symbol)

    @parametrize(
        "shape",
        [
            (20,),
            (50, 30),
            (50, 30, 40),
        ],
    )
    @torch._inductor.config.patch(
        {
            "pad_dynamic_shapes": True,
            "comprehensive_padding": True,
            "padding_alignment_bytes": 32,
            "pad_outputs": True,
        }
    )
    def test_dynamic_shapes_with_padding(self, shape):
        # [FUNCTION DOC] Parametrized by input shape (1D to 3D); config patched to pad dynamic
        # shapes and outputs with 32 byte alignment. Compiles an addition of padded-stride inputs
        # with dynamic shapes and asserts the strides of the allocated output match the expected
        # padded strides computed from the symbolic dims, rounding each stride up to a multiple of
        # 32 bytes worth of elements.
        """
        Test a graph with dynamic shapes with padding.
        """

        def get_input(shape):
            # [FUNCTION DOC] HELPER: Allocates a random tensor whose last dimension is rounded up
            # to a multiple of eight and returns a view with the requested shape but the padded
            # strides.
            pad_size = list(shape)
            pad_size[-1] = ((shape[-1] + 7) // 8) * 8
            pad = torch.randn(pad_size, dtype=torch.float32, device=self.device)
            view = torch.as_strided(pad, shape, pad.stride())
            return view

        args = [get_input(shape) for _ in range(2)]
        (gm,) = self._compile_and_check(
            torch.add, args, compile_kwargs={"dynamic": True}
        )

        # Check for a symbolic output shape.
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )
        example_tensor = empty_strided.meta["val"]
        symbolic_dims = example_tensor.shape
        symbolic_strides = example_tensor.stride()

        align_elems = 32 // args[0].dtype.itemsize
        expected_strides = [1 for _ in range(len(shape))]
        for i in range(len(shape) - 1, 0, -1):
            expected_strides[i - 1] = align_elems * (
                ((expected_strides[i] * symbolic_dims[i]) + align_elems - 1)
                // align_elems
            )
        for i, j in zip(symbolic_strides, expected_strides):
            self.assertEqual(i, j)

    def test_dynamic_shapes_precomputed_size(self):
        # [FUNCTION DOC] Compiles an addition of broadcast shaped inputs with dynamic shapes and
        # asserts the single Triton kernel node receives a precomputed size argument named ks0 in
        # its keyword arguments.
        """
        Test dynamic shapes where a kernel's size arg is precomputed.
        """
        func = torch.add
        args = [
            torch.randn(shape, device=self.device) for shape in [(7, 12, 9), (7, 1, 1)]
        ]
        (gm,) = self._compile_and_check(func, args, compile_kwargs={"dynamic": True})

        # Check for the precomputed size arg.
        (triton_node,) = gm.graph.find_nodes(
            op="call_function", target=triton_kernel_wrapper_mutation
        )
        self.assertIn("ks0", triton_node.kwargs["kwargs"])

    def test_dynamic_launch_grid_calc(self):
        # [FUNCTION DOC] Compiles an addition with dynamic shapes and checks the Triton kernel
        # node has a grid argument and xnumel and XBLOCK kernel arguments, and that the first grid
        # entry equals the ceiling of xnumel divided by XBLOCK while the other two entries are
        # one.
        """
        Test the dynamic launch grid calculation.
        """

        func = torch.add
        args = [torch.randn(shape, device=self.device) for shape in [(7, 12), (7, 1)]]
        (gm,) = self._compile_and_check(func, args, compile_kwargs={"dynamic": True})

        # Check for the precomputed size arg.
        (triton_node,) = gm.graph.find_nodes(
            op="call_function", target=triton_kernel_wrapper_mutation
        )
        self.assertIn("grid", triton_node.kwargs)
        self.assertIn("xnumel", triton_node.kwargs["kwargs"])
        self.assertIn("XBLOCK", triton_node.kwargs["kwargs"])
        grid = triton_node.kwargs["grid"][0]
        xnumel = triton_node.kwargs["kwargs"]["xnumel"].meta["val"]
        xblock = triton_node.kwargs["kwargs"]["XBLOCK"]
        self.assertEqual(grid[0].meta["val"], -(-xnumel // xblock))
        self.assertEqual(grid[1], 1)
        self.assertEqual(grid[2], 1)

    @config.patch({"trace.enabled": True})
    @unittest.mock.patch("torch._inductor.debug.DebugFormatter.output_code")
    def test_debug(self, mock_output_code):
        # [FUNCTION DOC] With trace enabled and the debug formatter output_code method mocked,
        # compiles a subtraction, asserts output_code was called exactly once, reads the file it
        # was given and asserts it mentions the Triton kernel wrapper mutation call.
        # Compile in debug mode.
        args = [torch.randn(11, device=self.device) for _ in range(2)]
        self._compile_and_check(torch.sub, args)

        # Check the output code for a Triton kernel call.
        mock_output_code.assert_called_once()
        (output_filename,) = mock_output_code.call_args.args
        with open(output_filename) as f:
            output_code = f.read()
        self.assertIn("triton_kernel_wrapper_mutation", output_code)

    @parametrize(
        "const",
        (1, 1.5),
    )
    def test_export_const_placeholder(self, const):
        # [FUNCTION DOC] Parametrized by an int or float constant. Exports a subtraction module
        # with a tensor and a constant input, compiles the exported module with inductor, captures
        # the single generated graph and asserts its result matches eager.
        """
        Test that we can compile a graph coming from torch.export with a constant input.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Simple module whose forward subtracts the second input from the
            # first.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x minus y.
                return x - y

        args = (torch.randn(8, device=self.device), const)
        mod = TestModule()
        export_gm = torch.export.export(mod, args).module()

        def compile_module(*inps):
            # [FUNCTION DOC] HELPER: Compiles the exported module with inductor for the given
            # inputs; the compiled result is discarded.
            torch._inductor.compile(export_gm, inps)

        (inductor_gm,) = self._run_and_capture_graphs(compile_module, args)
        result = inductor_gm(*args)
        ref = mod(*args)

        self.assertTrue(same(ref, result))

    def test_scatter_fallback_scalar_src(self):
        # [FUNCTION DOC] Under the deterministic guard, compiles a scatter whose source is a
        # Python float scalar, with an index tensor defined outside the function. Expects the
        # graph to contain exactly one scatter value in-place fallback call.
        """
        Test a special case where ScatterFallback takes a scalar 'src' argument.
        """

        def foo(input_):
            # [FUNCTION DOC] HELPER: Scatters the scalar 1.5 into the input along dim 0 using the
            # enclosing index tensor.
            dim = 0
            src = 1.5
            return torch.ops.aten.scatter(input_, dim, index, src)

        length = 8
        index = torch.randint(length, (length,), device=self.device)
        input_ = torch.randn(length, device=self.device)
        with DeterministicGuard(True):
            (gm,) = self._compile_and_check(
                foo,
                (input_,),
            )

        # Check for the fallback op.
        num_fallback = self._count_ops(gm, torch.ops.aten.scatter_.value)
        self.assertEqual(num_fallback, 1)

    @config.patch("partitioned_scatter_enabled", False)
    def test_index_put_fallback(self):
        # [FUNCTION DOC] With partitioned scatter disabled and the deterministic guard on,
        # compiles index_put with accumulate. Expects one kernel and exactly one in-place
        # index_put default call in the graph.
        """
        Test the deterministic fallback for index_put.
        """
        length = 8
        out, values = [torch.randn(length, device=self.device) for _ in range(2)]
        indices = (torch.randint(length, (length,), device=self.device),)
        accumulate = True
        with DeterministicGuard(True):
            (gm,) = self._compile_and_check(
                torch.index_put,
                (out, indices, values, accumulate),
                expected_num_triton_kernels=1,
            )

        # Check for the fallback op.
        self.assertEqual(self._count_ops(gm, torch.ops.aten.index_put_.default), 1)

    def test_scatter_reduce_fallback(self):
        # [FUNCTION DOC] Compiles an in-place scatter_reduce with amax reduction (include_self
        # off) followed by an add. Expects two kernels and exactly one call of the fallback op.
        """
        Test the customized wrapper codegen for ScatterFallback ops.
        """
        fallback_op = torch.ops.aten.scatter_reduce_.two

        def foo(out, index, src):
            # [FUNCTION DOC] HELPER: Applies the scatter reduce fallback op along dim 0 and
            # returns the result plus 1.
            dim = 0
            out = fallback_op(out, dim, index, src, reduce="amax", include_self=False)
            return out + 1

        length = 8
        out, src = [torch.randn(length, device=self.device) for _ in range(2)]
        index = torch.randint(length, (length,), device=self.device)
        (gm,) = self._compile_and_check(
            foo, (out, index, src), expected_num_triton_kernels=2
        )

        # Check for the fallback.
        self.assertEqual(self._count_ops(gm, fallback_op), 1)

    @parametrize("pred", (False, True))
    def test_cond_subgraph(self, pred: bool):
        # [FUNCTION DOC] Parametrized by predicate value. Compiles a conditional choosing cos or
        # sin plus one and expects three kernels. Checks the last graph has two get_attr nodes,
        # each referring to a GraphModule attribute (the branch subgraphs).
        """
        Test a model with subgraphs.
        """

        def foo(pred, x):
            # [FUNCTION DOC] HELPER: Applies cond on pred between cos and sin over x and adds 1.
            return torch.cond(pred, torch.cos, torch.sin, [x]) + 1

        x = torch.randn((2, 3), device=self.device)
        pred_tensor = torch.tensor([pred], device=self.device)
        gm = self._compile_and_check(
            foo, [pred_tensor, x], expected_num_triton_kernels=3
        )[-1]

        # Check for subgraphs.
        subgm_getattrs = list(gm.graph.find_nodes(op="get_attr"))
        self.assertEqual(len(subgm_getattrs), 2)
        for subgm_getattr in subgm_getattrs:
            target = subgm_getattr.name
            self.assertTrue(isinstance(getattr(gm, target), torch.fx.GraphModule))

    @parametrize("idx", (0, 1, 2))
    def test_switch_subgraph(self, idx: int):
        # [FUNCTION DOC] Parametrized by branch index (0 to 2). Compiles the simple switch model
        # from the control flow tests with an index tensor and expects four kernels. Checks the
        # last graph has one switch higher-order call and three get_attr nodes that are
        # GraphModule subgraphs.
        x = torch.randn((2, 3), device=self.device)
        idx_tensor = torch.tensor(idx, device=self.device)
        model = SwitchModels.Simple()
        gm = self._compile_and_check(
            model, [idx_tensor, x], expected_num_triton_kernels=4
        )[-1]

        # The FX graph should call torch.ops.higher_order.switch (not cond).
        switch_nodes = list(
            gm.graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.switch
            )
        )
        self.assertEqual(len(switch_nodes), 1)

        # Each branch should be a subgraph GraphModule attached as an attribute.
        subgm_getattrs = list(gm.graph.find_nodes(op="get_attr"))
        self.assertEqual(len(subgm_getattrs), 3)
        for subgm_getattr in subgm_getattrs:
            target = subgm_getattr.name
            self.assertTrue(isinstance(getattr(gm, target), torch.fx.GraphModule))

    @parametrize("pred", (False, True))
    def test_cond_no_operands(self, pred: bool):
        # [FUNCTION DOC] Parametrized by predicate. Compiles a cond whose branches take no
        # operands and expects two kernels.
        """
        Test torch.cond when the subgraphs take no inputs.
        """

        length = 8

        def true_fn():
            # [FUNCTION DOC] HELPER: Returns a zero tensor of the enclosing length on the device.
            return torch.zeros(length, device=self.device)

        def false_fn():
            # [FUNCTION DOC] HELPER: Returns the true branch result plus 5.
            return true_fn() + 5

        def foo(pred):
            # [FUNCTION DOC] HELPER: Calls cond with the predicate and the two operand-free
            # branches.
            return torch.cond(pred, true_fn, false_fn, ())

        pred_tensor = torch.tensor([pred], device=self.device)
        self._compile_and_check(foo, [pred_tensor], expected_num_triton_kernels=2)

    def test_cpp_raises(self):
        # [FUNCTION DOC] Registers the C++ scheduler with the FX wrapper for the CPU device and
        # compiles a CPU function; asserts that BackendCompilerFailed is raised with a message
        # mentioning Triton, since C++ kernels are not supported yet.
        """
        Test the C++ CPU backend. C++ kernels are not yet supported, so for now check
        that we get the expected exception.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y times 5.
            return x + y * 5

        device = torch.device("cpu")
        args = [torch.randn(5, device=device) for _ in range(2)]

        cpp_backend = common.DeviceCodegen(CppScheduling, WrapperFxCodegen, None)
        with (
            unittest.mock.patch.dict(
                common.device_codegens, {device.type: cpp_backend}
            ),
            self.assertRaisesRegex(BackendCompilerFailed, "Triton"),
        ):
            self._compile_and_check(foo, args)

    @parametrize("enable_tuning", (False, True))
    @parametrize("use_dynamic_shapes", (False, True))
    def test_autotune(self, use_dynamic_shapes: bool, enable_tuning: bool):
        # [FUNCTION DOC] Parametrized by dynamic shapes and by whether autotuning happens at
        # compile time. Patches the caching autotuner run method with a recording wrapper and sets
        # the autotune-at-compile-time config, asserts the tuner was not called before compiling,
        # compiles a multiplication of two inputs and asserts the tuner was called exactly when
        # compile-time tuning is on. Then asserts that, for dynamic shapes, the shape entries of
        # the single empty_strided node are FX nodes.
        orig_run = torch._inductor.runtime.triton_heuristics.CachingAutotuner.run
        called = False

        def run(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Sets the enclosing called flag and delegates to the original
            # autotuner run method.
            nonlocal called
            called = True
            return orig_run(*args, **kwargs)

        args = [torch.randn(8, device=self.device) for _ in range(2)]

        with (
            config.patch("triton.autotune_at_compile_time", enable_tuning),
            unittest.mock.patch.object(
                torch._inductor.runtime.triton_heuristics.CachingAutotuner, "run", run
            ),
        ):
            # Compile and check that the tuner was called.
            self.assertFalse(called)
            (gm,) = self._compile_and_check(
                torch.mul, args, compile_kwargs={"dynamic": use_dynamic_shapes}
            )
            self.assertEqual(called, enable_tuning)

        # Check for a symbolic output shape.
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )
        (shape, stride) = empty_strided.args
        if use_dynamic_shapes:
            self.assertEqual(type(shape[0]), torch.fx.Node)

    def test_custom_triton(self):
        # [FUNCTION DOC] Compiles a user-defined Triton add kernel launched through a Python
        # wrapper with a grid function; expects the default of one Triton kernel node and matching
        # results.
        # [FUNCTION DOC] HELPER: Triton kernel adding two input vectors elementwise over blocks of
        # the given size and storing the masked result to the output pointer.
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
            # [FUNCTION DOC] HELPER: Allocates an output like x and launches the add kernel over
            # it with block size 16, returning the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Launch grid: returns a one element tuple with the ceiling
                # division of the element count by the chosen block size.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        args = [torch.randn(32, device=self.device) for _ in range(2)]
        self._compile_and_check(add, args)

    def test_output_slice_view(self):
        # [FUNCTION DOC] Compiles a function returning a strided, transposed and squeezed view of
        # a four dimensional input; expects no Triton kernels and correct results.
        """
        Test when the output is a view of the input.
        The sliced strides create a TensorBox in the output IR.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Slices x, transposes it, slices again and squeezes dim 0,
            # producing a view.
            return x[0:2:2].T[3:].squeeze(0)

        args = [torch.rand([4, 4, 4, 4], device=self.device)]
        self._compile_and_check(foo, args, expected_num_triton_kernels=0)

    def test_fallback_tuple_constant_arg(self):
        # [FUNCTION DOC] Compiles a permute on a complex64 input, which forces a fallback op.
        # Expects no kernels, exactly one aten permute default call, and that its permutation
        # argument is a list of length three equal to the original tuple values (not flattened).
        """
        Test a fallback op with tuple constant argument.
        Check that tuple arguments are not flattened during codegen.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Permutes the dims of x to the order zero, two, one.
            # permute with a tuple argument
            return torch.permute(x, (0, 2, 1))

        # Use complex64 to force permute to become a fallback op
        args = [torch.randn(2, 3, 4, dtype=torch.complex64, device=self.device)]

        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=0)

        # Check for the fallback kernel with permute
        num_fallback = self._count_ops(gm, torch.ops.aten.permute.default)
        self.assertEqual(num_fallback, 1)

        # Verify the permute node has the correct tuple argument
        permute_node = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.permute.default
                )
            )
        )

        # The second argument should be the permutation (0, 2, 1)
        # Check that it's not flattened
        perm_arg = permute_node.args[1]
        self.assertIsInstance(
            perm_arg, list, "Permutation argument should not be flattened"
        )
        self.assertEqual(len(perm_arg), 3)
        self.assertEqual(tuple(perm_arg), (0, 2, 1))


@instantiate_parametrized_tests
class AOTFxirTestCase(InductorTestCase):
    # [CLASS DOC] Test case for the FX wrapper in the ahead-of-time compile path: models are
    # exported and compiled with the fx_wrapper option, and the resulting graph modules are
    # compared against eager and inspected. Uses the GPU device and expands parametrized tests.
    device = GPU_TYPE

    def check(
        self, model, inp, dynamic_shapes=None, strict=False
    ) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Exports the model (with optional dynamic shapes and strict flag),
        # AOT compiles the exported module with the fx_wrapper option plus the shared test config,
        # all with gradients disabled. Flattens the example inputs, asserts the compiled graph
        # module output equals eager, and asserts every call_function node other than the Triton
        # kernel wrapper has recorded value metadata. Returns the compiled graph module.
        with torch.no_grad():
            ep = torch.export.export(
                model, inp, dynamic_shapes=dynamic_shapes, strict=strict
            )
            gm = torch._inductor.aot_compile(
                ep.module(), inp, options={"fx_wrapper": True, **test_config}
            )
            # Flatten args for fx_wrapper gm
            flat_args, _ = pytree.tree_flatten(inp)
            self.assertTrue(same(model(*inp), gm(*flat_args)))

            for node in gm.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target != triton_kernel_wrapper_mutation
                ):
                    self.assertTrue(node.meta.get("val", None) is not None)

            return gm

    def test_aoti_fx_add(self):
        # [FUNCTION DOC] Checks an elementwise addition model on the AOT FX path.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
        self.check(M(), inp)

    @requires_gpu()
    def test_aoti_fx_parallel_compile_reloads_triton_kernel(self):
        # [FUNCTION DOC] Restarts the compile workers, then with subprocess workers, four compile
        # threads and a fresh cache, waits for the pool and asserts a process pool is used. AOT
        # compiles an addition model and asserts the graph has exactly one Triton kernel wrapper
        # node whose kernel can be retrieved from the kernel side table with a non-null function,
        # and that results match eager. Always shuts the compile workers down afterwards.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        shutdown_compile_workers()
        try:
            with config.patch(worker_start_method="subprocess", compile_threads=4):
                with fresh_cache():
                    AsyncCompile.wait_pool_ready()
                    self.assertTrue(AsyncCompile.use_process_pool())

                    inp = (
                        torch.ones(3, device=self.device),
                        torch.ones(3, device=self.device),
                    )
                    with torch.no_grad():
                        ep = torch.export.export(M(), inp)
                        gm = torch._inductor.aot_compile(
                            ep.module(),
                            inp,
                            options={
                                "fx_wrapper": True,
                                **test_config,
                                "compile_threads": 4,
                            },
                        )

                    triton_nodes = [
                        node
                        for node in gm.graph.nodes
                        if (
                            node.op == "call_function"
                            and node.target == triton_kernel_wrapper_mutation
                        )
                    ]
                    self.assertEqual(len(triton_nodes), 1)
                    kernel = kernel_side_table.get_kernel(
                        triton_nodes[0].kwargs["kernel_idx"]
                    )
                    self.assertIsNotNone(kernel.fn)

                    flat_args, _ = pytree.tree_flatten(inp)
                    self.assertTrue(same(M().to(self.device)(*inp), gm(*flat_args)))
        finally:
            shutdown_compile_workers()

    def test_aoti_fx_const(self):
        # [FUNCTION DOC] Checks a model that adds a parameter, a plain tensor attribute and a
        # freshly created scalar tensor to its inputs, exercising constants on the AOT FX path.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding a device, a ones parameter and a ones tensor
            # constant.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the device, creates a parameter of ones and a
                # plain tensor of ones on it.
                super().__init__()
                self.device = device
                self.a = torch.nn.Parameter(torch.ones(3, device=self.device))
                self.b = torch.ones(3, device=self.device)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y plus the parameter plus the constant
                # plus a new scalar tensor with value three.
                return x + y + self.a + self.b + torch.tensor(3, device=self.device)

        inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
        self.check(M(self.device), inp)

    def test_aoti_fx_linear(self):
        # [FUNCTION DOC] Checks a model with a single linear layer applied to a three by three
        # input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a three-to-three linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                return self.linear(x)

        inp = (torch.ones(3, 3, device=self.device),)
        self.check(M().to(self.device), inp)

    def test_aoti_fx_dynamic(self):
        # [FUNCTION DOC] Checks the addition model with both inputs exported with a dynamic first
        # dimension.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding its two inputs.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent initializer only.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
        self.check(
            M().to(device=self.device),
            inp,
            dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}),
        )

    def test_custom_triton_autotune_dynamic(self):
        # [FUNCTION DOC] Exports (strict mode) a model launching an autotuned 2D Triton add kernel
        # with a dynamic first dimension bounded between one and ten for each input, and checks it
        # on the AOT FX path.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that launches the autotuned 2D add kernel over zeros-like
            # output.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Allocates a zero output like x, launches the autotuned 2D
                # kernel with a grid built from the output sizes, and returns the output.
                output = torch.zeros_like(x)
                x_elements = output.size()[0]
                y_elements = output.size()[1]

                def grid(meta):
                    # [FUNCTION DOC] HELPER: Launch grid returning the block-count ceiling
                    # divisions for both dimensions using the chosen block sizes.
                    return (
                        triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                        triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                    )

                add_kernel_2d_autotuned[grid](x, y, output, x_elements, y_elements)

                return output

        num_dims = 2
        dims = [10] * num_dims
        x = torch.randn(*dims, device=self.device)
        y = torch.randn(*dims, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=10)
        dim0_y = Dim("dim0_y", min=1, max=10)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check(
            Model().to(device=self.device),
            (x, y),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )

    def test_custom_backend(self):
        # [FUNCTION DOC] Registers (through a patched device codegen mapping) a Triton scheduler
        # with the Python wrapper and a custom FX wrapper codegen subclass. Asserts the subclass
        # hook is not called before compiling an add-one model through the AOT FX path and was
        # called afterwards.
        """
        Test registering a custom FX backend.
        """
        called = False

        class CustomWrapperCodegen(WrapperFxCodegen):
            # [CLASS DOC] HELPER: FX wrapper codegen subclass that records in the enclosing flag
            # that its compile hook ran.
            def compile_graph(self, gm):
                # [FUNCTION DOC] HELPER: Sets the enclosing called flag and delegates to the
                # parent compile_graph implementation, returning its result.
                """
                Simply records whether this override was called.
                """
                nonlocal called
                called = True
                return super().compile_graph(gm)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding one to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus 1.
                return x + 1

        # Register a custom FX backend.
        custom_backend = common.DeviceCodegen(
            TritonScheduling,
            PythonWrapperCodegen,
            fx_wrapper_codegen=CustomWrapperCodegen,
        )
        with unittest.mock.patch.dict(
            common.device_codegens, {self.device: custom_backend}
        ):
            # The backend should not have been called yet.
            self.assertFalse(called)

            inp = (torch.randn(8, device=self.device),)
            self.check(M().to(self.device), inp)

        # Now the backend should have been called.
        self.assertTrue(called)

    @parametrize(
        "expr",
        [
            (2 * Dim("x") + 1),
            (Dim("x", min=3) - 3),
        ],
    )
    def test_dynamic_input_expr(self, expr: sympy.Expr):
        # [FUNCTION DOC] Parametrized by a dimension expression (twice a symbol plus one, or a
        # symbol with a minimum of 3 minus 3). Exports a model with a dynamic first dimension
        # expressed as that expression, compiles on the AOT FX path and asserts the graph has
        # exactly one aten sym_size call.
        """
        Test dynamic shapes with a nontrivial input expression.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model flattening the input and adding its second dimension size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reshapes x to a vector of length rows times columns and
                # adds the column count.
                return x.reshape(x.shape[0] * x.shape[1]) + x.shape[1]

        dynamic_shapes = {"x": {0: expr}}
        inp = (torch.randn((5, 4), device=self.device),)
        gm = self.check(M().to(self.device), inp, dynamic_shapes=dynamic_shapes)

        # Check for dynamic size ops.
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.sym_size.int
                )
            ),
            1,
        )

    @parametrize("pred", (False, True))
    def test_cond_multi_inputs_and_outputs(self, pred):
        # [FUNCTION DOC] Parametrized by predicate (the parameter is then overwritten by a one-
        # element true tensor). Checks a cond model with two inputs and two outputs on the AOT FX
        # path and asserts the generated code matches an expected text with a cond call between
        # true and false subgraphs, two getitem outputs and a list return.
        """
        Test torch.cond and check the output graphs.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying cond to two tensors with two outputs per branch.
            def forward(self, pred, x, y):
                # [FUNCTION DOC] HELPER: Defines two branches and returns the cond result for the
                # predicate and the two tensors.
                def true_fn(x, y):
                    # [FUNCTION DOC] HELPER: True branch returning tanh of x and relu of y.
                    return torch.tanh(x), torch.relu(y)

                def false_fn(x, y):
                    # [FUNCTION DOC] HELPER: False branch returning the true branch outputs each
                    # divided by two.
                    return tuple(t / 2 for t in true_fn(x, y))

                return torch.cond(pred, true_fn, false_fn, (x, y))

        pred = torch.tensor([True], device=self.device)
        (x, y) = [torch.randn(8, device=self.device) for _ in range(2)]
        gm = self.check(M(), (pred, x, y))

        # Check the graph.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    false_graph_0 = self.false_graph_0
    true_graph_0 = self.true_graph_0
    cond = torch.ops.higher_order.cond(arg0_1, true_graph_0, false_graph_0, (arg1_1, arg2_1));  arg0_1 = true_graph_0 = false_graph_0 = arg1_1 = arg2_1 = None
    buf1 = cond[0]
    buf2 = cond[1];  cond = None
    return [buf1, buf2]""",
        )

    def test_dims_dynamic_outer_static_padded_inner(self):
        # [FUNCTION DOC] Checks an addition on strided inputs whose last dimension is padded to
        # twice its size, with dynamic outer dimensions and a static innermost dimension.
        """
        Test padding on inner dimensions, with dynamic outer dimensions.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        def get_input_padded_inner(shape):
            # [FUNCTION DOC] HELPER: Allocates a random tensor with the last dimension doubled and
            # returns a view with the given shape and the strides of the larger tensor.
            full_shape = shape[:-1] + (shape[-1] * 2,)
            full = torch.randn(full_shape, dtype=torch.float32, device=self.device)
            view = torch.as_strided(full, shape, full.stride())
            return view

        shape = (4, 4, 4)
        args = tuple(get_input_padded_inner(shape) for _ in range(2))
        self.check(
            M(),
            args,
            dynamic_shapes=({0: Dim.DYNAMIC, 1: Dim.DYNAMIC, 2: Dim.STATIC},) * 2,
        )

    @parametrize("length", (4, 8))
    def test_cond_dynamic_shape_pred_scalar_closure(self, length: int):
        # [FUNCTION DOC] Parametrized by length. Checks a cond whose predicate depends on a
        # dynamic size (first dimension of x above five) and whose branches use a scalar size
        # computed outside the branches, with both inputs having a dynamic first dimension.
        """
        Test cond using a predicate computed from dynamic shapes.
        Also test a dynamic scalar computed outside the branches.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model using a cond with a shape-dependent predicate and a
            # closed-over size.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Flattens x, captures the first dimension of y, and
                # returns the cond over the flattened tensor.
                z = x.reshape(-1)
                a = y.shape[0]

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch adding the captured size to its input.
                    return x + a

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch returning the true branch result divided
                    # by two.
                    return true_fn(x) / 2

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (z,))

        (x, y) = [
            torch.randn(shape, device=self.device)
            for shape in [(length // 2,) * 2, (length,)]
        ]
        dynamic_shapes = {
            "x": {0: Dim.DYNAMIC},
            "y": {0: Dim.DYNAMIC},
        }
        self.check(M(), (x, y), dynamic_shapes=dynamic_shapes)

    def test_dynamic_scalar_output(self):
        # [FUNCTION DOC] Checks a model that returns a Python scalar computed from a dynamic
        # dimension.
        """
        Test an output scalar from dynamic shapes.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning three times the first dimension size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the first dimension size of x times 3.
                return x.shape[0] * 3

        x = torch.randn(7, device=self.device)
        self.check(M(), (x,), dynamic_shapes=({0: Dim.DYNAMIC},))

    @parametrize("dynamic", (False, True))
    @parametrize("input_", (1.5, 2, False))
    def test_item(self, input_, dynamic: bool):
        # [FUNCTION DOC] Parametrized by an input scalar value (float, int or bool) and by whether
        # the dimension is dynamic (twice a symbol with minimum one). Checks a model that calls
        # item on an element of a ten element CPU tensor.
        """
        Test calling Tensor.item.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the item of the second element of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the Python scalar of x at index one.
                return x[1].item()

        x = torch.tensor((input_,) * 10)
        d = Dim("s0", min=1)
        dynamic_shapes = ({0: 2 * d},) if dynamic else None
        self.check(M(), (x,), dynamic_shapes=dynamic_shapes)

    @parametrize("pred", (False, True))
    def test_mismatched_branch_dynamic(self, pred: bool):
        # [FUNCTION DOC] Parametrized by predicate. Builds inputs offset so the predicate is
        # truthful (a one-element predicate tensor and three random tensors) and exports the
        # mismatched-output-size cond model from the control flow tests with dynamic first
        # dimensions for three of the inputs; checks it on the AOT FX path.
        """
        Test cond branches with mismatched dynamic shapes.
        """

        # Apply an offset to guarantee the truith of the predicate.
        pred_offset = 1 if pred else -1

        inputs = [
            torch.tensor([pred], device=self.device),
        ] + [torch.randn(10, 20, device=self.device) + pred_offset for _ in range(3)]
        dim0_a = Dim("s0", min=4, max=1024)
        dim0_b = Dim("s1", min=4, max=1024)
        dynamic_shapes = {
            "p": {},
            "x": {0: dim0_a, 1: None},
            "y": {0: dim0_b, 1: None},
            "z": {0: dim0_a, 1: None},
        }

        self.check(
            CondModels.MismatchedOutputSize(),
            tuple(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_const_folded_subgraph(self):
        # [FUNCTION DOC] Builds a parent graph whose only node calls an exported submodule
        # (producing a random tensor plus one that can be const-folded) and takes its item through
        # getitem. AOT compiles it with the fx_wrapper option while a hook on the node metadata
        # setter in the FX converter module is replaced with a checker, which records a call and
        # asserts the partial it receives has a non-null fake_mode keyword. Asserts the hook was
        # called and the output shape is five by ten.
        """
        If a graph only contains a call_module node to a subgraph,
        where the subgraph can be const-folded away,
        validate the fake mode used in FXConverter generation is not None.
        """
        device = self.device
        shape = (5, 10)

        class Submodule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model without inputs returning a random tensor of the enclosing
            # shape plus one.
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns random values of the enclosing shape on the
                # enclosing device plus 1.
                return torch.randn(*shape, device=device) + 1

        # Create a parent graph with this module as a subgraph and output
        ep = torch.export.export(Submodule(), ())
        parent_graph = torch.fx.Graph()
        call_mod = parent_graph.call_module("sub", args=())
        get_item = parent_graph.call_function(
            operator.getitem, args=(call_mod, slice(None))
        )
        parent_graph.output((get_item,))
        parent = torch.fx.GraphModule({"sub": ep.module()}, parent_graph)

        # Verify FXConverter.generate uses non-null fake mode
        # Intercept _set_node_metadata_hook to ensure fake_mode is not None
        orig_set_hook = torch._inductor.codegen.wrapper_fxir._set_node_metadata_hook
        called = False

        def mock_set_hook(gm: torch.fx.GraphModule, fn):
            # [FUNCTION DOC] HELPER: Replacement for the metadata hook setter: marks the enclosing
            # flag called, asserts the function it receives has a fake_mode keyword that is not
            # none, then calls the original setter and returns its result.
            nonlocal called
            called = True
            # Please update this check if `fake_mode` is
            # no longer used in FXConverter call to _node_metadata_hook
            self.assertTrue("fake_mode" in fn.keywords)
            self.assertIsNotNone(fn.keywords["fake_mode"])
            return orig_set_hook(gm, fn)

        self.assertFalse(called)
        with unittest.mock.patch.object(
            torch._inductor.codegen.wrapper_fxir,
            "_set_node_metadata_hook",
            mock_set_hook,
        ):
            args = ()
            compiled = torch._inductor.aot_compile(
                parent, args, options={"fx_wrapper": True}
            )
            self.assertTrue(called)

            compiled_out = compiled(*args)
            self.assertEqual(compiled_out.shape, shape)

    def test_reshape_dynamic_ph(self):
        # [FUNCTION DOC] Checks, using automatic dynamic dimensions, a model that reshapes a
        # tensor to a shape passed as a list input (symbolic ints become placeholders) and adds 2.
        """
        Test dynamic scalars using SymInts placeholder
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model reshaping x to a given shape and adding 2.
            def forward(self, x, shape):
                # [FUNCTION DOC] HELPER: Returns x reshaped to the given shape plus 2.
                return torch.reshape(x, shape) + 2

        ds = {
            "x": (torch.export.Dim.AUTO, torch.export.Dim.AUTO),
            "shape": [torch.export.Dim.AUTO, torch.export.Dim.AUTO],
        }
        args = (torch.randn((12, 14), device=self.device), [6, 28])
        self.check(TestModule(), args, ds)

    def test_reshape_dynamic_tmd(self):
        # [FUNCTION DOC] Checks a model reshaping to a shape computed from the input dimensions
        # (half the first, double the second) with automatic dynamic dimensions.
        """
        Test dynamic reshape using shape dependent information
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model reshaping x using shape-derived sizes and adding 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a new shape from half the first dim and double the
                # second dim, reshapes x and adds 2.
                new_shape = [x.shape[0] // 2, x.shape[1] * 2]
                return torch.reshape(x, new_shape) + 2

        ds = {
            "x": (torch.export.Dim.AUTO, torch.export.Dim.AUTO),
        }
        args = (torch.randn((12, 14), device=self.device),)
        self.check(TestModule(), args, ds)

    def test_extern_kernel_irnode_kwargs(self):
        # [FUNCTION DOC] Checks a model calling segment_reduce (sum) with the offsets passed as a
        # keyword tensor argument, ensuring IR nodes in extern kernel kwargs are materialized.
        """
        Test that IR nodes passed as kwargs to extern kernels are properly materialized.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying segment_reduce with sum.
            def forward(self, data, offsets):
                # [FUNCTION DOC] HELPER: Returns the sum segment reduction of data using the
                # offsets keyword.
                return torch.segment_reduce(data, "sum", offsets=offsets)

        length = 10
        data = torch.randn(length, device=self.device)
        offsets = torch.tensor([0, 3, 7, length], dtype=torch.int64, device=self.device)

        self.check(TestModule(), (data, offsets))

    def test_compound_symint_graph_input(self):
        # [FUNCTION DOC] Exports a cond whose branches close over twice a size plus one, with
        # dynamic first dimensions. After checking on the AOT FX path, collects the two branch
        # subgraph modules and asserts each has exactly one symbolic int placeholder whose
        # expression is a linear form (two times a single symbol plus one), i.e. the compound
        # expression is bound to one placeholder.
        """A compound symbolic graph input binds to a single placeholder.

        The branches close over 2 * y.shape[0] + 1, so the lifted argument
        reaches the converter as a sympy.Add rather than a single Symbol.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a cond whose branches capture a compound symbolic
            # size.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes twice the size of y plus one and returns the
                # cond over x on its first dimension size.
                a = 2 * y.shape[0] + 1

                def true_fn(t):
                    # [FUNCTION DOC] HELPER: True branch adding the captured value to t.
                    return t + a

                # Identical branches would share one Triton kernel between
                # two separately converted subgraphs, which FX conversion
                # cannot resolve.
                def false_fn(t):
                    # [FUNCTION DOC] HELPER: False branch adding the captured value plus 1 to t.
                    return t + a + 1

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        inp = tuple(torch.randn(numel, device=self.device) for numel in (8, 4))
        gm = self.check(M(), inp, dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}))

        # The lifted argument is a placeholder of the branch subgraphs, not of
        # the parent graph. Check each branch on its own: one symbolic
        # placeholder holding the whole expression, so neither branch fell back
        # to taking the constituent symbol instead.
        branches = [
            submod
            for name, submod in gm.named_modules()
            if name and isinstance(submod, torch.fx.GraphModule)
        ]
        self.assertEqual(len(branches), 2, "expected the two cond subgraphs")
        for branch in branches:
            symbolic = [
                node.meta["val"]
                for node in branch.graph.find_nodes(op="placeholder")
                if isinstance(node.meta.get("val"), torch.SymInt)
            ]
            self.assertEqual(len(symbolic), 1)
            expr = symbolic[0].node.expr
            (sym,) = expr.free_symbols
            self.assertEqual(expr, 2 * sym + 1)

    def test_nonlinear_compound_input_rejected(self):
        # [FUNCTION DOC] Compiles a cond model whose branches capture the square of a size and
        # expects InductorError with a message containing "Cannot solve input expression", since
        # the solver cannot invert a nonlinear expression.
        """A compound input the solver cannot invert fails with a clear error.

        y.shape[0] * y.shape[0] lifts as s**2, which the solver does not invert
        to recover s.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose cond branches capture the square of a dynamic size.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes the square of y's first dimension size and
                # returns the cond over x.
                a = y.shape[0] * y.shape[0]

                def true_fn(t):
                    # [FUNCTION DOC] HELPER: True branch adding the captured square to t.
                    return t + a

                def false_fn(t):
                    # [FUNCTION DOC] HELPER: False branch adding the captured square plus 1 to t.
                    return t + a + 1

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        inp = tuple(torch.randn(numel, device=self.device) for numel in (8, 4))
        with self.assertRaisesRegex(InductorError, "Cannot solve input expression"):
            self.check(M(), inp, dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}))

    def test_underdetermined_compound_input_rejected(self):
        # [FUNCTION DOC] Compiles a cond model whose branches capture the sum of two different
        # dynamic sizes and expects InductorError with a message about leaving symbols undefined,
        # since one value cannot define both.
        """A compound input holding two unknowns fails with a clear error.

        The branches close over y.shape[0] + z.shape[0] but take neither
        tensor, so one bound value would have to determine both symbols.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose cond branches capture the sum of two dynamic sizes.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Computes the sum of the first dimension sizes of y and z
                # and returns the cond over x.
                a = y.shape[0] + z.shape[0]

                def true_fn(t):
                    # [FUNCTION DOC] HELPER: True branch adding the captured sum to t.
                    return t + a

                def false_fn(t):
                    # [FUNCTION DOC] HELPER: False branch adding the captured sum plus 1 to t.
                    return t + a + 1

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        inp = tuple(torch.randn(numel, device=self.device) for numel in (8, 4, 6))
        with self.assertRaisesRegex(InductorError, "leaves these symbols undefined"):
            self.check(
                M(),
                inp,
                dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}),
            )


class TestReplaceFloorDiv(InductorTestCase):
    # [CLASS DOC] Tests for the sympy helper converting floor of a quotient into the FloorDiv
    # function. Each test builds an expression, rewrites it and uses a shared check that verifies
    # equivalence.
    """
    Tests for floor -> FloorDiv conversion.
    """

    def _check(self, expr: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] HELPER: Rewrite-and-verify helper. Asserts the input has at least one
        # floor, applies the floor to FloorDiv replacement, asserts no floor remains and the
        # number of FloorDiv terms grew by at most the number of floors, expands FloorDiv back
        # into floor and asserts the simplified results are equal to the original. Returns the
        # rewritten expression.
        # Check that we started with floor's.
        num_floors = expr.count(sympy.floor)
        self.assertGreater(num_floors, 0)

        replaced = replace_floor_div(expr)

        # Check that all floor's were replaced.
        # We should have no more new FloorDiv's than floor's in the original expression,
        # although we can have less due to simplification.
        self.assertEqual(replaced.count(sympy.floor), 0)
        self.assertLessEqual(
            replaced.count(FloorDiv) - expr.count(FloorDiv), num_floors
        )

        def expand_floor_div(
            numerator: sympy.Expr, denominator: sympy.Expr
        ) -> sympy.Expr:
            # [FUNCTION DOC] HELPER: Returns the floor of the numerator divided by the
            # denominator.
            return sympy.floor(numerator / denominator)

        # Expand FloorDiv back into floor and check for equality.
        self.assertEqual(
            *[
                sympy.simplify(e.replace(FloorDiv, expand_floor_div))
                for e in (replaced, expr)
            ]
        )

        return replaced

    def test_rewrite_floor_div_mul_pow(self):
        # [FUNCTION DOC] Checks floor of x over y (a product with a power) is rewritten to a
        # single FloorDiv whose arguments are x and y.
        x, y = sympy.symbols("x y")
        expr = sympy.floor(x / y)
        self.assertEqual(expr.count(FloorDiv), 0)
        self.assertEqual(expr.count(sympy.core.mul.Mul), 1)
        self.assertEqual(expr.count(sympy.Pow), 1)

        rewritten = self._check(expr)
        self.assertTrue(isinstance(rewritten, FloorDiv))
        self.assertEqual(rewritten.args, (x, y))

    def test_rewrite_floor_div_mul_rational(self):
        # [FUNCTION DOC] Checks floor of x over 5 (product with a rational) is rewritten to a
        # FloorDiv with arguments x and 5.
        x = sympy.Symbol("x")
        expr = sympy.floor(x / 5)
        self.assertEqual(expr.count(FloorDiv), 0)
        self.assertEqual(expr.count(sympy.core.mul.Mul), 1)
        self.assertEqual(expr.count(sympy.Rational), 1)

        rewritten = self._check(expr)
        self.assertTrue(isinstance(rewritten, FloorDiv))
        self.assertEqual(rewritten.args, (x, 5))

    def test_no_rewrite_div(self):
        # [FUNCTION DOC] Checks an expression without floor (plain division) is returned unchanged
        # by the replacement.
        x, y = sympy.symbols("x y")
        expr = x / y
        self.assertEqual(expr.count(FloorDiv), 0)

        rewritten = replace_floor_div(expr)
        self.assertEqual(rewritten, expr)

    def test_rewrite_floor_div_nested(self):
        # [FUNCTION DOC] Checks a nested floor expression is rewritten and the result contains two
        # FloorDiv terms.
        x, y = sympy.symbols("x y")
        expr = sympy.floor((sympy.floor(x / 5) + 1) / y)
        self.assertEqual(expr.count(FloorDiv), 0)

        rewritten = self._check(expr)
        self.assertEqual(rewritten.count(FloorDiv), 2)

    def test_rewrite_floor_div_rational_const(self):
        # [FUNCTION DOC] Checks an unevaluated floor of the constant one fifth is rewritten to the
        # constant zero.
        expr = sympy.floor(sympy.S.One / 5, evaluate=False)
        self.assertEqual(expr.count(FloorDiv), 0)
        self.assertEqual(expr.count(sympy.Mul), 0)
        self.assertEqual(expr.count(sympy.Rational), 1)

        # Expression evaluates to a compile time constant
        rewritten = self._check(expr)
        self.assertEqual(rewritten, sympy.S.Zero)

    def test_no_distribute_mul_floordiv(self):
        # [FUNCTION DOC] Checks that multiplying by two a floor of x over two keeps one product
        # and one FloorDiv after rewrite (no distribution of the multiplication).
        """
        Test that multiplication doesn't distribute with floor division.
        """
        x = sympy.Symbol("x")
        expr = 2 * sympy.floor(x / 2)
        rewritten = self._check(expr)
        self.assertEqual(rewritten.count(sympy.Mul), 1)
        self.assertEqual(rewritten.count(FloorDiv), 1)

    def test_rational_multi_pows(self):
        # [FUNCTION DOC] Checks a floor of a product of a rational and powers of y and z produces
        # exactly one FloorDiv after rewrite.
        """
        Test an expression with a rational and multiple pows.
        """
        x, y, z = sympy.symbols("x y z")
        expr = sympy.floor((x / 5) * (y**2) * (z**3))
        mul = expr.args[0]
        self.assertTrue(isinstance(mul, sympy.Mul))
        self.assertTrue(isinstance(mul.args[0], sympy.Rational))
        self.assertEqual(expr.count(sympy.Pow), 2)
        rewritten = self._check(expr)
        self.assertEqual(rewritten.count(FloorDiv), 1)

    def test_variable_exp(self):
        # [FUNCTION DOC] Checks floor of two to the minus x (positive symbol) is rewritten to a
        # FloorDiv with numerator one and denominator two to the x.
        """
        Test pow when the exponent is a variable.
        """
        x = sympy.Symbol("x", positive=True)
        expr = sympy.floor(2**-x)
        replaced = self._check(expr)

        # Check that x went to the denominator.
        self.assertEqual(replaced.args, (1, 2**x))

    def test_launch_grid_dynamic_padding(self):
        # [FUNCTION DOC] Checks a nested expression of floor with FloorDiv terms of products (as
        # from padded dynamic launch grids) is rewritten and equivalent to the original.
        """
        Test a complex launch grid expression arising from padding with dynamic shapes.
        """
        x, y = sympy.symbols("x y")
        expr = sympy.floor(-FloorDiv(x * y, 2) / FloorDiv(-x * y, 131070))
        self._check(expr)


class TestUnbackedSymbolDefs(InductorTestCase):
    # [CLASS DOC] Tests for the FX converter method that generates definitions for unbacked
    # symbols, using mock objects.
    """Tests for FxConverter._generate_unbacked_symbol_defs."""

    def test_empty_bindings_returns_before_buffer_lookup(self):
        # [FUNCTION DOC] Uses mocks for the converter and an unbacked symbol definitions line with
        # an output name but no bindings and an empty buffer-to-node mapping. Asserts that calling
        # the generation method returns none instead of attempting the buffer lookup that would
        # raise KeyError.
        # A line with no unbacked symbols must return before the output-buffer
        # lookup: such kernels are not recorded in buffer_to_node, so the lookup
        # would KeyError. Fails without the early-return guard, passes with it.
        conv = MagicMock(spec=FxConverter)
        conv.gm = MagicMock()
        conv.buffer_to_node = {}
        line = MagicMock(spec=UnbackedSymbolDefsLine)
        # output_name is set so that, absent the guard, the buffer_to_node lookup
        # is actually reached and raises KeyError (its documented failure mode).
        line.output_name = "buf0"
        line.unbacked_bindings = {}
        self.assertIsNone(FxConverter._generate_unbacked_symbol_defs(conv, line))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
