# Owner(s): ["oncall: pt2"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Test suite for AOT Autograd / aot_function / aot_module / aot_export APIs in
# functorch. Covers make_fx tracing, mutation and aliasing handling, input/output metadata,
# tensor subclasses, dynamic shapes, partitioners, autograd cache, saved-tensor hooks, and
# OpInfo/ModuleInfo based sweeps. Import time: tries to import torchvision and networkx,
# emitting a warning and setting a USE_* flag when missing; the module bottom runs the tests
# through run_tests.
import copy
import gc
import itertools
import operator
import unittest
import warnings
import weakref
from collections.abc import Callable
from contextlib import ContextDecorator, ExitStack, nullcontext
from functools import partial, wraps
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from common_utils import (
    capture_codegen_source,
    decorate,
    decorateForModules,
    saved_tensors_hooks_to_gm,
    skip,
    xfail,
)

import torch
import torch._dynamo as torchdynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
from functorch import grad, jacrev, make_fx, vjp, vmap
from functorch.compile import (
    aot_function,
    aot_module,
    aot_module_simplified,
    compiled_function,
    compiled_module,
    default_decompositions,
    default_partition,
    get_aot_compilation_context,
    make_boxed_compiler,
    make_boxed_func,
    memory_efficient_fusion,
    min_cut_rematerialization_partition,
    nnc_jit,
    nop,
)
from functorch.experimental import control_flow
from torch._decomp import decomposition_table
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._functorch.aot_autograd import (
    _aot_export_function,
    aot_export_joint_simple,
    aot_export_module,
    SerializableAOTDispatchCompiler,
)
from torch._functorch.partitioners import (
    _extract_fwd_bwd_modules,
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
)
from torch._higher_order_ops.out_dtype import out_dtype
from torch._inductor.codecache import compiled_fx_graph_hash
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch._inductor.output_code import MockFXGraphCacheOutput
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensorMode
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, ShapeEnv
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.utils.rnn import PackedSequence
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    skipOps,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    compare_equal_outs_and_grads,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_MACOS,
    IS_WINDOWS,
    IS_X86,
    outs_and_grads,
    parametrize,
    run_tests,
    TEST_MKL,
    TestCase,
    xfail_inherited_tests,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_tensor import ConstantExtraMetadataTensor
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.optests import (
    _test_aot_autograd_forwards_backwards_helper,
    aot_autograd_check,
)
from torch.testing._internal.subclasses import WrapperSubclass
from torch.testing._internal.two_tensor import TwoTensor, TwoTensorMode
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)


USE_TORCHVISION = False
try:
    import torchvision

    USE_TORCHVISION = True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )

USE_NETWORKX = False
try:
    import networkx  # noqa: F401

    USE_NETWORKX = True
except ImportError:
    warnings.warn("Some tests use networkx but it was not installed", UserWarning)

# NB: numpy is a testing dependency!


def amax_to_scale(
    amax: torch.Tensor,
    float8_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
):
    # [FUNCTION DOC] HELPER: Compute a float8 scaling factor from an absolute-max tensor.
    # Converts amax to float64, clamps it to at least a tiny epsilon, divides the largest
    # representable value of the given float8 dtype by it, and returns the result as float32.
    # The round_scales_to_power_of_2 argument is accepted but ignored.
    amax = amax.to(torch.float64)
    res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=1e-12)
    res = res.to(torch.float32)
    return res


# Must be at module level to use fx.wrap
@torch.fx.wrap
def _pack_fp8_with_scale_wrap(x):
    # [FUNCTION DOC] HELPER: fx-wrapped (opaque to tracing) packing hook for saved-tensor tests.
    # Non floating point inputs are returned unchanged; otherwise computes a scale from the
    # tensor's max absolute value, scales it in float32, casts to float8_e5m2 and returns a
    # tuple of original dtype, scale and the float8 tensor.
    if not x.dtype.is_floating_point:
        return x

    amax = torch.max(torch.abs(x))
    scale = amax_to_scale(amax, torch.float8_e5m2)
    x_scaled = x.to(torch.float32) * scale
    x_fp8 = x_scaled.to(torch.float8_e5m2)
    return x.dtype, scale, x_fp8


@torch.fx.wrap
def _unpack_fp8_with_scale_wrap(x):
    # [FUNCTION DOC] HELPER: fx-wrapped inverse of the scale-based packing hook. A plain tensor
    # is returned unchanged; otherwise the packed tuple is unpacked, the float8 data is cast to
    # float32, divided by the scale and cast back to the original dtype.
    if isinstance(x, torch.Tensor):
        return x

    dtype, scale, x_fp8 = x
    y = x_fp8.to(torch.float32) / scale
    return y.to(dtype)


@torch.fx.wrap
def _pack_fp8_wrap(x):
    # [FUNCTION DOC] HELPER: fx-wrapped packing hook that casts floating point tensors to
    # float8_e5m2 and returns a tuple of original dtype and the cast tensor; other tensors
    # pass through. When the input is a tensor subclass (compilation time) it raises
    # AssertionError unless the saved-tensor hook context holds non-None forward graph,
    # backward graph and node.
    if not x.dtype.is_floating_point:
        return x

    if type(x) is not torch.Tensor:
        # Check only during compilation
        # Test calls hooks to get reference output
        ctx = torch._functorch._aot_autograd.graph_compile._get_saved_tensor_hook_context()
        if ctx["_fw_graph"] is None:
            raise AssertionError("Expected _fw_graph to not be None")
        if ctx["_bw_graph"] is None:
            raise AssertionError("Expected _bw_graph to not be None")
        if ctx["_node"] is None:
            raise AssertionError("Expected _node to not be None")

    return (x.dtype, x.to(torch.float8_e5m2))


@torch.fx.wrap
def _unpack_fp8_wrap(x):
    # [FUNCTION DOC] HELPER: fx-wrapped inverse of the float8 packing hook. A plain tensor is
    # returned unchanged; otherwise unpacks dtype and tensor and, if the stored tensor is a
    # subclass, raises AssertionError unless the saved-tensor hook context has forward graph,
    # backward graph and node set. Returns the tensor cast back to its dtype.
    if isinstance(x, torch.Tensor):
        return x

    dtype, tensor = x
    if type(tensor) is not torch.Tensor:
        # Check only during compilation
        # Test calls hooks to get reference output
        ctx = torch._functorch._aot_autograd.graph_compile._get_saved_tensor_hook_context()
        if ctx["_fw_graph"] is None:
            raise AssertionError("Expected _fw_graph to not be None")
        if ctx["_bw_graph"] is None:
            raise AssertionError("Expected _bw_graph to not be None")
        if ctx["_node"] is None:
            raise AssertionError("Expected _node to not be None")
    return tensor.to(dtype)


def pack_fp8(x):
    # [FUNCTION DOC] HELPER: Pack hook entry point that delegates to the fx-wrapped float8
    # packing function.
    return _pack_fp8_wrap(x)


def unpack_fp8(packed):
    # [FUNCTION DOC] HELPER: Unpack hook entry point that delegates to the fx-wrapped float8
    # unpacking function.
    return _unpack_fp8_wrap(packed)


def pack_fp8_with_scale(x):
    # [FUNCTION DOC] HELPER: Pack hook entry point that delegates to the fx-wrapped scale-based
    # float8 packing function.
    return _pack_fp8_with_scale_wrap(x)


def unpack_fp8_with_scale(packed):
    # [FUNCTION DOC] HELPER: Unpack hook entry point that delegates to the fx-wrapped
    # scale-based float8 unpacking function.
    return _unpack_fp8_with_scale_wrap(packed)


class AOTTestCase(TestCase):
    # [CLASS DOC] Base TestCase for the AOT tests in this file; adds a metadata
    # comparison assertion.
    def assertTensorMetadataEqual(self, actual, expected):
        # [FUNCTION DOC] HELPER: Assert that two tensors have equal shape, strides and storage
        # offset. If the expected tensor is a traceable wrapper subclass, also require the
        # actual one to be one, require equal flattened attribute names, and recursively compare
        # the metadata of each inner tensor attribute. Raises assertion failures on mismatch.
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())

        if is_traceable_wrapper_subclass(expected):
            self.assertTrue(is_traceable_wrapper_subclass(actual))
            expected_attrs, _ = expected.__tensor_flatten__()
            actual_attrs, _ = actual.__tensor_flatten__()
            self.assertEqual(actual_attrs, expected_attrs)
            for attr in expected_attrs:
                expected_inner = getattr(expected, attr)
                actual_inner = getattr(actual, attr)
                if isinstance(expected_inner, torch.Tensor):
                    self.assertTensorMetadataEqual(actual_inner, expected_inner)


class TestPythonKey(AOTTestCase):
    # [CLASS DOC] Tests that make_fx and nnc_jit (functorch python-key tracing) produce
    # callables matching eager results, parameterized per device by instantiation.
    def test_make_fx(self, device):
        # [FUNCTION DOC] Trace a sine function with make_fx and check the traced module matches
        # eager on a fresh input.
        def f(x):
            # [FUNCTION DOC] HELPER: Elementwise sine, the function being traced.
            return torch.sin(x)

        inp = torch.randn(3)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_grad(self, device):
        # [FUNCTION DOC] Trace the gradient of a sum-of-sine function with make_fx and check the
        # traced result equals the grad function on new input.
        def f(x):
            # [FUNCTION DOC] HELPER: Sum of sines, differentiated by the test.
            return torch.sin(x).sum()

        inp = torch.randn(3)
        f = grad(f)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_scalar_device(self, device):
        # [FUNCTION DOC] Trace an addition of a device tensor and a CPU scalar tensor with
        # make_fx and check the traced module reproduces eager output.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add two inputs.
            return a + b

        inps = [torch.randn(3, device=device), torch.tensor(5)]
        fx_f = make_fx(f)(*inps)
        self.assertEqual(fx_f(*inps), f(*inps))

    def test_make_fx_vmap(self, device):
        # [FUNCTION DOC] Trace a vmapped sine function with make_fx and compare with the vmapped
        # function on a new batch.
        def f(x):
            # [FUNCTION DOC] HELPER: Elementwise sine mapped over a batch.
            return torch.sin(x)

        inp = torch.randn(5, 3)
        f = vmap(f)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(5, 3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_jacrev(self, device):
        # [FUNCTION DOC] Trace a double jacrev of a sin-sum function with make_fx and compare
        # against the eager transform on new input.
        def f(x):
            # [FUNCTION DOC] HELPER: Sum of sines whose second derivative is traced.
            return x.sin().sum()

        inp = torch.randn(3)
        f = jacrev(jacrev(f))
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_vjp(self, device):
        # [FUNCTION DOC] Trace the vjp function returned by vjp of a sin-sum function with
        # make_fx and compare outputs on a new cotangent.
        def f(x):
            # [FUNCTION DOC] HELPER: Sum of sines used to build the vjp.
            return torch.sin(x).sum()

        primals = torch.randn(3)
        _, vjp_fn = vjp(f, primals)
        cotangent = torch.randn(())
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    def test_make_fx_functionalize(self, device):
        # [FUNCTION DOC] Check make_fx over a functionalized symbolic trace: the symbolic graph
        # contains the in-place relu method, and after make_fx of the functionalized module the
        # graph contains the aten relu op. Regression test for issue 84570.
        from functorch.experimental import functionalize

        def fn(a):
            # [FUNCTION DOC] HELPER: Multiply by two then apply in-place relu.
            a = a * 2
            a.relu_()
            return a

        a = torch.randn(3, device=device)
        symbolic_gm = torch.fx.symbolic_trace(fn)
        includes_method_relu_ = any(
            str(n.target) == "relu_" for n in symbolic_gm.graph.nodes
        )
        self.assertTrue(includes_method_relu_)
        # Also verifies fix for https://github.com/pytorch/pytorch/issues/84570
        gm = make_fx(functionalize(symbolic_gm))(a)
        includes_aten_relu = any(
            n.target == torch.ops.aten.relu.default for n in gm.graph.nodes
        )
        self.assertTrue(includes_aten_relu)

    def test_make_fx_no_decompose(self, device):
        # [FUNCTION DOC] Skipped immediately (returns through skipTest citing a recursion
        # error); the code after the skip is unreachable and would check that tanh_backward
        # appears only without a decomposition table.
        # FIXME
        return self.skipTest("error: maximum recursion reached")

        def f(x):
            # [FUNCTION DOC] HELPER: Sum of tanh, unreachable body helper.
            return torch.tanh(x).sum()

        fx_f = make_fx(grad(f))(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}

        self.assertEqual(torch.ops.aten.tanh_backward in ops, True)

        fx_f = make_fx(grad(f), decomposition_table)(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}
        self.assertEqual(torch.ops.aten.tanh_backward in ops, False)

    def test_nnc_jit(self, device):
        # [FUNCTION DOC] Check nnc_jit of a sine function matches eager on a vector input.
        def f(x):
            # [FUNCTION DOC] HELPER: Elementwise sine.
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(3)
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_scalar(self, device):
        # [FUNCTION DOC] Check nnc_jit works on a zero-dimensional tensor input.
        def f(x):
            # [FUNCTION DOC] HELPER: Elementwise sine.
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(())
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_pytrees(self, device):
        # [FUNCTION DOC] Check nnc_jit handles a list of tensors as input and output.
        def f(x):
            # [FUNCTION DOC] HELPER: Return a list holding the sine of the first element.
            return [torch.sin(x[0])]

        jit_f = nnc_jit(f)

        inp = [torch.randn(3)]
        self.assertEqual(jit_f(inp), f(inp))

    def test_external_calls(self, device):
        # [FUNCTION DOC] Check nnc_jit of a matrix-vector product (external call) matches eager.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Matrix-vector product.
            return torch.mv(a, b)

        jit_f = nnc_jit(f)
        inp = [torch.randn(3, 3), torch.randn(3)]
        self.assertEqual(jit_f(*inp), f(*inp))

    def test_nnc_passthrough(self, device):
        # [FUNCTION DOC] Check nnc_jit with outputs that pass through an input unchanged, and
        # with a dict input that is mutated in place and returned; results must match eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Return the sum and the second input unchanged.
            return x + y, y

        inp = (torch.randn(3), torch.randn(3))
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

        def f(x):
            # [FUNCTION DOC] HELPER: Double the entry with key a in the dict and return the dict
            # (mutates the input).
            x["a"] = x["a"] * 2
            return x

        inp = ({"a": torch.randn(3), "b": torch.randn(3)},)
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_backward_trace(self, device):
        # [FUNCTION DOC] Skipped without torchvision. Despite the name no tracing happens: it
        # runs a resnet18 backward inside a helper function and checks the parameter gradients
        # equal those from a second plain backward on the same input.
        mod = torchvision.models.resnet18()

        def f(x):
            # [FUNCTION DOC] HELPER: Run the model, backpropagate the summed output and return
            # all parameter gradients.
            out = mod(x)
            out.sum().backward()
            return [a.grad for a in mod.parameters()]

        inp = torch.randn(3, 3, 250, 250, requires_grad=True)
        grads = f(inp)

        mod.zero_grad()
        mod(inp).sum().backward()
        grads2 = [a.grad for a in mod.parameters()]
        self.assertEqual(grads, grads2)


def get_base(t):
    # [FUNCTION DOC] HELPER: Return the base tensor of a view, or the tensor itself when it is
    # not a view.
    return t._base if t._is_view() else t


def is_in_base(t, maybe_tensors):
    # [FUNCTION DOC] HELPER: Return True if the base of the first tensor is the base of any
    # tensor in the given sequence (non-tensors ignored), otherwise False.
    t_base = get_base(t)
    for maybe_tensor in maybe_tensors:
        if isinstance(maybe_tensor, torch.Tensor):
            if t_base is get_base(maybe_tensor):
                return True
    return False


def skipIfDynamoInput(reason):
    # [FUNCTION DOC] HELPER: Decorator factory for tests: when the test instance is a
    # TestAOTAutogradWithDynamo, the wrapped test is skipped with a message including the
    # reason; otherwise the test runs normally.
    """
    Skip TestAOTAutograd if running with dynamo input
    """

    def decorator(func):
        # [FUNCTION DOC] HELPER: Wrap a test function with the dynamo-skipping wrapper.
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Skip the test with the stated reason under the dynamo
            # variant, otherwise call the wrapped test (its return value is discarded).
            if isinstance(self, TestAOTAutogradWithDynamo):
                self.skipTest(
                    f"Skipping {self._testMethodName} in TestAOTAutogradWithDynamo because {reason}"
                )
            else:
                func(self, *args, **kwargs)

        return wrapper

    return decorator


class TestAOTAutograd(AOTTestCase):
    # [CLASS DOC] Main AOT autograd test class. Provides the verify_aot_autograd harness that
    # compares eager and aot_function/aot_module results and gradients, then many tests of
    # mutation, aliasing, views, subclasses and graph structure, some with expected graph text.
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: list[Callable | None],
        decompositions: dict | None,
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        # [FUNCTION DOC] HELPER: Compile f with aot_function (or aot_module when f is an
        # nn.Module) using a forward compiler that stores the traced graph in the given
        # one-element list, a no-op backward compiler, the given decompositions, input mutation
        # retention flag and dynamic flag. Returns the compiled callable.
        """
        Runs aot_autograd with the specified settings on f.
        """
        if isinstance(f, nn.Module):
            compiled_f = aot_module(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        else:
            compiled_f = aot_function(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        return compiled_f

    # test_mutation will:
    # - Ensure that inputs are non-leaves, so our graphs can mutate them
    # - try to mutate outputs of the graph (to ensure that autograd meta is set properly on outputs)
    @patch("functorch.compile.config.debug_assert", True)
    def verify_aot_autograd(
        self,
        f,
        inp_: Callable | list[Any],
        *,
        test_mutation: bool = False,
        keep_inp_mutations: bool = False,
        decompositions: dict | None = None,
        dynamic: bool = False,
        # Only active when inp_ is Callable.
        # TODO: probably consolidate all tests to make inp a Callable.
        make_inputs_subclasses: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Core harness. For each setting of keep_input_mutations (only
        # True when keep_inp_mutations is set, else both) it builds reference and test inputs
        # (cloning, preserving duplicate inputs, making them non-leaves when test_mutation),
        # compiles f, runs outputs and grads for both and checks equal values, grads,
        # requires_grad, leaf-ness, view-of-input relationships, and optionally that outputs can
        # be mutated. In the cache subclass it also reruns the compiled function. Returns the
        # captured forward graph.
        def make_inputs(inp_):
            # [FUNCTION DOC] HELPER: Build inputs for a run. If the argument is callable, call
            # it for a pair of function inputs and graph inputs (under TwoTensorMode when
            # subclass inputs requested). Otherwise clone tensors (detached, same requires_grad,
            # reusing clones for duplicates) and, when test_mutation, make graph inputs by
            # adding one so they are non-leaves. Returns the pair.
            # Some tests pass in a callable for inp, to generate the inputs
            # (useful if we want to generate complicated aliasing inputs)
            if isinstance(inp_, Callable):
                inp_callable = inp_
                # The callable should return a tuple of f_inputs, f_graph_inputs
                # (The idea is that we might want to compile a function with the graph inputs,
                # but test autograd backprop all the way through the actual inputs)
                with TwoTensorMode() if make_inputs_subclasses else nullcontext():
                    inp, graph_inps = inp_callable()
            else:
                inp = []
                # Our input clones need to mimic when inputs are duplicates of one another
                dupes_map = {}
                for i, x in enumerate(inp_):
                    if x in dupes_map:
                        x_dupe_idx = dupes_map[x]
                        inp.append(inp[x_dupe_idx])
                    else:
                        dupes_map[x] = i
                        if not isinstance(x, torch.Tensor):
                            x_copy = x
                        else:
                            x_copy = x.detach().clone().requires_grad_(x.requires_grad)
                            if x.requires_grad and not x.is_leaf:
                                x_copy = x_copy.clone()

                        inp.append(x_copy)

                if test_mutation:
                    # For graphs where we mutate inputs, need our test to make sure inputs aren't leaves
                    graph_inps = [x.add(1) for x in inp]
                else:
                    graph_inps = inp

            return inp, graph_inps

        def check_results(
            ref_results,
            test_results,
            ref_graph_inps,
            test_graph_inps,
            ref_inp,
            test_inp,
        ):
            # [FUNCTION DOC] HELPER: Compare reference and compiled results: grads equal; for
            # each output tensor compare requires_grad, leaf-ness, whether it aliases a graph
            # input or another output, and values; when test_mutation also add and subtract 2 in
            # place on both to verify autograd metadata. Finally compare requires_grad and
            # values of all inputs.
            ref_out, ref_grad = ref_results
            test_out, test_grad = test_results
            self.assertEqual(ref_grad, test_grad)
            if isinstance(ref_out, torch.Tensor):
                self.assertTrue(isinstance(test_out, torch.Tensor))
                ref_out, test_out = [ref_out], [test_out]
            for ref_o, test_o in zip(ref_out, test_out):
                if isinstance(ref_o, torch.Tensor):
                    self.assertEqual(ref_o.requires_grad, test_o.requires_grad)
                    self.assertEqual(ref_o.is_leaf, test_o.is_leaf)
                    ref_is_view_of_non_interm = is_in_base(
                        ref_o, ref_graph_inps
                    ) or is_in_base(ref_o, ref_out)
                    test_is_view_of_non_interm = is_in_base(
                        test_o, test_graph_inps
                    ) or is_in_base(test_o, test_out)
                    self.assertEqual(
                        ref_is_view_of_non_interm, test_is_view_of_non_interm
                    )
                    self.assertEqual(ref_o, test_o)
                    if test_mutation:
                        # This tests that autograd meta is set properly on the output we can
                        # mutate it.
                        ref_o.add_(2)
                        test_o.add_(2)
                        self.assertEqual(ref_o, test_o)
                        # Reverse the modification
                        ref_o.sub_(2)
                        test_o.sub_(2)
                        self.assertEqual(ref_o, test_o)
            for ref_i, test_i in zip(ref_inp, test_inp):
                if isinstance(ref_i, torch.Tensor):
                    self.assertEqual(ref_i.requires_grad, test_i.requires_grad)
                self.assertEqual(ref_i, test_i)

        for keep_input_mutations in [True] if keep_inp_mutations else [True, False]:
            inp, graph_inps = make_inputs(inp_)
            test_inp, test_graph_inps = make_inputs(inp_)
            fw_graph_cell = [None]
            compiled_f = self.run_autograd(
                f, fw_graph_cell, decompositions, keep_input_mutations, dynamic
            )
            ref_results = outs_and_grads(f, graph_inps, inp)
            test_results = outs_and_grads(compiled_f, test_graph_inps, test_inp)

            check_results(
                ref_results, test_results, graph_inps, test_graph_inps, inp, test_inp
            )
            if isinstance(self, TestAOTAutogradWithCache):
                # When testing with cache, run compiled_f a second time
                cached_inp, cached_graph_inps = make_inputs(inp_)
                cached_results = outs_and_grads(
                    compiled_f, cached_graph_inps, cached_inp
                )
                check_results(
                    ref_results,
                    cached_results,
                    graph_inps,
                    cached_graph_inps,
                    inp,
                    cached_inp,
                )

        return fw_graph_cell[0]

    def test_non_tensor_and_none_inputs(self):
        # [FUNCTION DOC] Check aot_autograd with non tensor and None inputs, with and
        # without requires_grad.
        # int, None, Tensor
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Return first times third argument.
            return a * c

        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=True)]
        self.verify_aot_autograd(f, inp)
        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=False)]
        self.verify_aot_autograd(f, inp)

    def test_single_output(self):
        # [FUNCTION DOC] Check a single output function with grad-requiring and
        # non-requiring inputs.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add two inputs.
            return a + b

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output(self):
        # [FUNCTION DOC] Check a function returning a tuple of two outputs.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return the sum and the difference.
            return a + b, a - b

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output_list(self):
        # [FUNCTION DOC] Check a function returning a list of two outputs.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return a list with the sum and the difference.
            return [a + b, a - b]

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    # Test for bug occurring at the intersection of fake tensors & functionalization.
    def test_squeeze_mutation(self):
        # [FUNCTION DOC] Test for a bug at the intersection of fake tensors and
        # functionalization: mutating a squeezed clone, with dynamic shapes enabled.
        def f(a):
            # [FUNCTION DOC] HELPER: Squeeze a clone of the input, add one in place, return
            # input plus the result.
            b = a.clone().squeeze(-1)
            b.add_(1.0)
            return a + b

        inp = [torch.randn(3, 1, requires_grad=True)]
        self.verify_aot_autograd(f, inp, dynamic=True)
        inp = [torch.randn(3, 1, requires_grad=False)]
        self.verify_aot_autograd(f, inp, dynamic=True)

    def test_to_dense_strided_tensor(self):
        # [FUNCTION DOC] Check to_dense on a strided tensor in several variants (masked_grad,
        # dtype conversions) with dynamic shapes.
        def f(a):
            # [FUNCTION DOC] HELPER: Return a tuple of to_dense results with default, masked
            # grad, float32 and float64 options.
            return (
                a.to_dense(),
                a.to_dense(masked_grad=True),
                a.to_dense(dtype=torch.float32),
                a.to_dense(dtype=torch.float64),
            )

        inp = [torch.randn(3, 4)]
        self.verify_aot_autograd(f, inp, dynamic=True)

    def test_complex_linear(self):
        # [FUNCTION DOC] Regression test for issue 93424: aot_autograd on a complex64 linear
        # layer module with a sum of absolute values output.
        # https://github.com/pytorch/pytorch/issues/93424
        inp = [torch.randn(1, 10, 10, dtype=torch.complex64)]

        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single complex64 Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module and its complex linear layer.
                super().__init__()
                self.linear = nn.Linear(10, 10, dtype=torch.complex64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the linear layer, sum and take absolute value.
                return self.linear(x).sum().abs()

        self.verify_aot_autograd(F(), inp)

    def test_embedding_bag_view_dynamic(self):
        # [FUNCTION DOC] Check an EmbeddingBag with sparse gradients and a view on its output,
        # in static and dynamic modes. Verifies wrapping a storage-less sparse tensor in a
        # functional wrapper works in backward.
        # Backwards pass tries to wrap a sparse tensor in a FunctionalTensorWrapper;
        # test that this works even though the sparse tensor has no storage.

        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a sparse EmbeddingBag.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module and its embedding bag layer.
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(100, 8, sparse=True)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Embed indices with offsets and flatten the result.
                return self.emb(x, y).view(-1)

        x = torch.arange(3)
        y = torch.arange(3)
        self.verify_aot_autograd(F(), [x, y], dynamic=False)
        self.verify_aot_autograd(F(), [x, y], dynamic=True)

    def test_input_mutation_simple(self):
        # [FUNCTION DOC] Check a simple input mutation (in-place multiply) with and without
        # requires_grad, and assert the forward graph text: a clone, a multiply, another
        # multiply, returning the mutated value plus output.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply the input by two in place and return it
            # times three.
            a.mul_(2)
            return a * 3

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        # Things to note:
        # - the extra clone is because we need to pass the pre-mutated input to grad(),
        #   but autograd operates above functionalization so we need to manually clone.
        #   Hopefully backends can optimize this easily.
        # - The extra return arg is because the compiled forward returns (mutated inputs + outputs)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, 3)
    return (mul, mul_1)""",
        )

    def test_input_mutation_set__input_mutation(self):
        # [FUNCTION DOC] Check an input mutated through set_ under no_grad (with input mutations
        # kept) for requires_grad and non-requiring inputs.
        def f(a):
            # [FUNCTION DOC] HELPER: Under no_grad set the input's storage to an arange matrix,
            # return input times that matrix.
            b = torch.arange(9, dtype=a.dtype).reshape(3, 3)
            with torch.no_grad():
                a.set_(b)
            return a * b

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)

    def test_set__steals_view_chain(self):
        # [FUNCTION DOC] Check set_ stealing the view chain of its source: the destination,
        # after being set to a view slice, is mutated and multiplied, with input mutations kept.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Set a scaled copy to a view slice of another scaled copy,
            # mutate it in place, return the product with the slice.
            a_ = a.mul(2)
            b_ = b.mul(2)
            b_slice = b_[1].view(3, 3)
            # a_clone should inherit the view chain from b_slice
            a_.set_(b_slice)
            # Also mutates b_,
            a_.view(-1).mul_(2)
            return a_ * b_slice

        inp = [
            torch.ones(3, 3, requires_grad=False),
            torch.zeros(3, 9, requires_grad=False),
        ]
        self.verify_aot_autograd(f, inp, keep_inp_mutations=True)

    def _compile_autocast(self, device, *, forward_autocast):
        # [FUNCTION DOC] HELPER: Define a custom library op with CompositeExplicitAutograd,
        # and autocast kernels (CPU and CUDA) that add one, wrap it in an autograd Function
        # and compile a function applying it with autocast disabled using the aot_eager
        # backend. Runs with autocast enabled or not, returns the output and the gradient
        # w.r.t. the input.
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x) -> Tensor")
            m.impl("foo", torch.clone, "CompositeExplicitAutograd")

            def autocast(x):
                # [FUNCTION DOC] HELPER: Autocast kernel for the custom op; adds one.
                return x + 1

            m.impl("foo", autocast, "AutocastCPU")
            m.impl("foo", autocast, "AutocastCUDA")

            foo = torch.ops.mylib.foo.default

            class Foo(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function around the custom op whose backward
                # multiplies grad by the op applied to the saved input.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Save the input and call the custom op.
                    ctx.save_for_backward(x)
                    return foo(x)

                @staticmethod
                def backward(ctx, grad):
                    # [FUNCTION DOC] HELPER: Return grad times the op applied to the
                    # saved input.
                    (x,) = ctx.saved_tensors
                    return grad * foo(x)

            def fn(x):
                # [FUNCTION DOC] HELPER: Apply the Function with autocast disabled.
                with torch.amp.autocast(device, enabled=False):
                    return Foo.apply(x)

            x = torch.tensor(0.0, device=device, requires_grad=True)
            if forward_autocast:
                with (
                    torch.amp.autocast(device),
                    torch._dynamo.config.patch(recompile_limit=999),
                ):
                    out = torch.compile(fn, fullgraph=True, backend="aot_eager")(x)
            else:
                with torch._dynamo.config.patch(recompile_limit=999):
                    out = torch.compile(fn, fullgraph=True, backend="aot_eager")(x)
            (grad,) = torch.autograd.grad(out, x)
            return out, grad

    @torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    def test_backward_pass_autocast_on(self):
        # [FUNCTION DOC] With backward autocast set same as forward, the autocast kernel (adds
        # one) runs in backward as well: expects output zeros and gradient ones, on CPU and CUDA
        # if available.
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            out, grad = self._compile_autocast(device, forward_autocast=True)
            self.assertEqual(out, torch.zeros_like(out))
            self.assertEqual(grad, torch.ones_like(grad))

    @torch._functorch.config.patch(backward_pass_autocast="off")
    def test_backward_pass_autocast_off(self):
        # [FUNCTION DOC] With backward autocast off, expects output zeros and gradient zeros on
        # available devices.
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            out, grad = self._compile_autocast(device, forward_autocast=True)
            self.assertEqual(out, torch.zeros_like(out))
            self.assertEqual(grad, torch.zeros_like(grad))

    @torch._functorch.config.patch(backward_pass_autocast="off")
    def test_backward_pass_autocast_custom(self):
        # [FUNCTION DOC] With a custom backward autocast list and no forward autocast, expects
        # output zeros and gradient ones on available devices.
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            with torch._functorch.config.patch(
                backward_pass_autocast=[{"device_type": device}]
            ):
                out, grad = self._compile_autocast(device, forward_autocast=False)
                self.assertEqual(out, torch.zeros_like(out))
                self.assertEqual(grad, torch.ones_like(grad))

    @skipIfDynamoInput(
        "Test doesn't make sense with dynamo, which changes order of mutations"
    )
    def test_set__and_data_mutation_good(self):
        # [FUNCTION DOC] Skipped under the dynamo variant. Checks set_ followed by data mutation
        # of the source (allowed) with input mutations kept, and asserts the forward graph text.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Under no_grad set a to b then double b in place; return a
            # plus b.
            # The data mutation happens *after* the set_(). This is ok (see the graph below)
            with torch.no_grad():
                a.set_(b)
                b.mul_(2)
            return a + b

        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        inp = [
            torch.ones(3, 3, requires_grad=False),
            torch.zeros(3, 3, requires_grad=False),
        ]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        # Important things to note:
        # - "return a.set_(b)" desugars into "return b"
        # - Both a and b are recorded as experiencing mutations,
        #   which is why we see "b_updated" (output of the mul) twice in the graph outputs.
        #   a is recorded as both a data mutation and a metadata mutation (due to set_ swapping its storage).
        # - the runtime epilogue for a is "a.set_(mul)"
        # - the runtime epilogue for b is "b.copy_(mul)"
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_2, 2)
    add = torch.ops.aten.add.Tensor(mul, mul)
    set_ = torch.ops.aten.set_.source_Tensor(primals_1, mul);  primals_1 = set_ = None
    copy_ = torch.ops.aten.copy_.default(primals_2, mul);  primals_2 = mul = copy_ = None
    return (add,)""",
        )

    # This is a (hopefully) extremely rare case that is difficult to handle,
    # so we ban it.
    # https://github.com/pytorch/pytorch/issues/126236
    # https://github.com/pytorch/pytorch/pull/126113
    @xfailIfTorchDynamo
    def test_set__and_data_mutation_bad(self):
        # [FUNCTION DOC] Expected-failure under dynamo. Setting an input to a new tensor and
        # then mutating a view made before the set_ is banned; asserts a RuntimeError about
        # mutating tensors with frozen storage.
        def f(a):
            # [FUNCTION DOC] HELPER: Take a view, set the input to a fresh tensor, mutate the
            # stale view under no_grad, return input plus tensor.
            a_view = a.view(-1)
            tmp = torch.ones(3, 3, requires_grad=True)
            # Now, any mutations on either tmp
            # will be tracked as graph input mutations.
            with torch.no_grad():
                a.set_(tmp)
                # BAD: a_view is now detached from every graph input,
                # so we won't recognize that this caused an input mutation!
                a_view.mul_(2)
            return a + tmp

        inp = [torch.ones(3, 3, requires_grad=True)]
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    @skipIfDynamoInput(
        "Test doesn't make sense with dynamo, which changes order of mutations"
    )
    def test_set__not_allowed(self):
        # [FUNCTION DOC] Skipped under the dynamo variant. Mutating an input after set_ is not
        # supported; expects AssertionError mentioning other mutations that cannot be handled.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Set a to b under no_grad, then multiply a in place, return
            # a plus b.
            with torch.no_grad():
                a.set_(b)
            # Mutating a will change a's grad_fn, which requires us to replay the mutation outside of the graph.
            # We currently ban this today, when the input also received a set_() input mutation.
            a.mul_(2)
            return a + b

        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        with self.assertRaisesRegex(
            AssertionError, "but the input has other mutations that we cannot"
        ):
            self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    def test_input_mutation_set__nop(self):
        # [FUNCTION DOC] Check two set_ calls that undo each other result in no input mutation;
        # asserts the graph has no set_ ops and only one output.
        def f(a):
            # [FUNCTION DOC] HELPER: Set the input to an arange and back to its alias under
            # no_grad, return the sum with the reshaped arange.
            b = torch.arange(9, dtype=a.dtype)
            a_old = torch.ops.aten.alias.default(a)
            with torch.no_grad():
                a.set_(b)
                a.set_(a_old)
            return a + b.reshape(3, 3)

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        # Things to note:
        # - There are no set_() calls in the graph (we functionalize a.set_(b) into "b")
        # - There is only **1** graph output. We properly realized that the two set_() calls
        #   undo each other, and so effectively no inputs are mutated.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    arange = torch.ops.aten.arange.default(9, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    alias = torch.ops.aten.alias.default(primals_1);  primals_1 = None
    view = torch.ops.aten.view.default(arange, [3, 3]);  arange = None
    add = torch.ops.aten.add.Tensor(alias, view);  alias = view = None
    return (add,)""",
        )

    def test_input_mutation_simple_with_none_and_nontensor(self):
        # [FUNCTION DOC] Check aot_function with None and non-tensor arguments gives the same
        # result as eager for inputs with and without requires_grad.
        # Tensor, None, int
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Return first times third argument.
            return a * c

        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = [torch.ones(3, 3, requires_grad=req_grad), None, 3]
            out_ref = f(*inp)
            out_test = f_compiled(*inp)
            self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/93363
    def test_mutates_input_noncontiguous(self):
        # [FUNCTION DOC] Regression test for issue 93363: mutating a noncontiguous view input
        # inside the compiled function must update the underlying base tensor like eager.
        def f(a):
            # [FUNCTION DOC] HELPER: Add one in place to the input and return an empty tuple.
            a.add_(1)
            return ()

        f_compiled = aot_function(f, nop)
        ref = torch.ones(4, requires_grad=True) + 0
        ref_view = ref[0::2]

        test = torch.ones(4, requires_grad=True) + 0
        test_view = test[0::2]

        out_ref = f(ref_view)  # noqa: F841
        out_test = f_compiled(test_view)  # noqa: F841
        self.assertEqual(ref, test)

    def test_input_mutation_modifies_autograd_meta_of_aliases(self):
        # [FUNCTION DOC] Check that an input mutation updates autograd metadata of aliases of
        # the input (version counter and grad_fn class) and gradients match eager.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply input by two in place and return detached input
            # plus one.
            a.mul_(2)
            out = a + 1
            return out.detach()

        x_ref = torch.ones(3, 3, requires_grad=True).clone()
        x_ref_view = x_ref.view(3, 3)

        x_test = torch.ones(3, 3, requires_grad=True).clone()
        x_test_view = x_test.view(3, 3)

        f_compiled = aot_function(f, nop, keep_inference_input_mutations=True)
        f(x_ref)
        f_compiled(x_test)
        # f will mutate aliases of the input, including its autograd metadata!
        # y.grad_fn is AsStridedBackward
        self.assertEqual(x_ref_view, x_test_view)
        self.assertEqual(x_ref_view._version, x_test_view._version)
        self.assertEqual(x_ref_view.grad_fn.__class__, x_test_view.grad_fn.__class__)
        # Test the actual gradients are correct
        (x_ref * x_ref_view).sum().backward()
        (x_test * x_test_view).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)
        self.assertEqual(x_ref_view.grad, x_test_view.grad)

    def test_nested_subclasses(self):
        # [FUNCTION DOC] Compile a sin-cos function with the aot_eager backend on a doubly
        # nested TwoTensor input; checks the output is nested TwoTensor all the way down and
        # that backward produces a nested TwoTensor gradient on the input.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Sine then cosine, compiled with the aot_eager backend.
            return x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)
        aaaa = TwoTensor(aa, aa2)
        out = f(aaaa)
        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, TwoTensor))
        self.assertTrue(isinstance(out.b, TwoTensor))
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        self.assertTrue(isinstance(out.b.b, torch.Tensor))

        out.sum().backward()
        self.assertTrue(isinstance(aaaa.grad, TwoTensor))
        self.assertTrue(isinstance(aaaa.grad.a, TwoTensor))
        self.assertTrue(isinstance(aaaa.grad.b, TwoTensor))

    def test_nested_subclasses_non_nested_grad(self):
        # [FUNCTION DOC] Backward with a gradient subclass structure that differs from what was
        # guessed at compile time (a flat TwoTensor input combined in eager with a nested
        # tensor); expects RuntimeError saying the subclass metadata was guessed incorrectly.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Sine then cosine, compiled with the aot_eager backend.
            return x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        new_aa = TwoTensor(a3, a4)
        aa = TwoTensor(a, a2)

        aa2 = aa.detach().clone().requires_grad_()
        aaaa = TwoTensor(aa, aa2)
        out = f(new_aa)
        new_out = out + aaaa
        with self.assertRaisesRegex(
            RuntimeError,
            """
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
""",
        ):
            new_out.sum().backward()

    def test_nested_subclasses_non_homogenous(self):
        # [FUNCTION DOC] Compile a function over a ConstantExtraMetadataTensor wrapping a
        # TwoTensor; checks output subclass structure, closeness to eager output, and that
        # gradients keep the wrapper subclass structure and match eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiply the wrapper's constant attribute, its inner
            # element and the sine-cosine of the input.
            x_elem = x.elem
            x_metadata = x.constant_attribute
            return x_metadata * x_elem * x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)
        custom_aa = ConstantExtraMetadataTensor(aa)
        custom_aa.constant_attribute = 6
        custom_aa2 = ConstantExtraMetadataTensor(aa2)
        custom_aa2.constant_attribute = 6

        out_eager = f(custom_aa)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_aa2)

        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.b, ConstantExtraMetadataTensor))
        self.assertTrue(torch.allclose(out_eager, out))

        out_eager.sum().backward()
        out.sum().backward()

        self.assertTrue(torch.allclose(custom_aa.grad, custom_aa2.grad))
        self.assertTrue(isinstance(custom_aa2.grad, TwoTensor))
        self.assertTrue(isinstance(custom_aa2.grad.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(custom_aa2.grad.b, ConstantExtraMetadataTensor))

    def test_subclasses_mixed(self):
        # [FUNCTION DOC] Compile a function mixing a metadata wrapper input and a TwoTensor
        # input; checks result matches eager and output is a TwoTensor whose parts are metadata
        # wrapper tensors.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Build a TwoTensor from two products involving the constant
            # attribute of x and the parts of y.
            x_metadata = x.constant_attribute
            out_a = x_metadata * x * y.a
            out_b = x * y.a * y.b
            return TwoTensor(out_a, out_b)

        a = torch.ones(4, requires_grad=False)
        a2 = a.clone()
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 5
        custom_a2 = ConstantExtraMetadataTensor(a2)
        custom_a2.constant_attribute = 5

        b = torch.ones(4, requires_grad=False)
        b2 = b.clone()
        b3 = b.clone()
        b4 = b.clone()
        bb = TwoTensor(b, b2)
        bb2 = TwoTensor(b3, b4)

        out_eager = f(custom_a, bb)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_a2, bb2)

        self.assertTrue(torch.allclose(out_eager, out))
        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.b, ConstantExtraMetadataTensor))

    def test_subclasses_mixed_mode(self):
        # [FUNCTION DOC] Run a sine-cosine function under a dispatch mode that wraps outputs in
        # a metadata subclass (unless already one); checks the compiled output is a wrapper
        # around a TwoTensor, matches eager, and gradients keep that structure.
        def f(x):
            # [FUNCTION DOC] HELPER: Sine then cosine.
            return x.sin().cos()

        class AddConstantMetadataMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that runs each op and, if the argument types do
            # not include the metadata subclass, wraps the result in the subclass with constant
            # attribute 5.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Run the op; wrap the result in the metadata subclass
                # with constant attribute 5 when that subclass is absent from the argument
                # types; return the result.
                out = func(*args, **(kwargs or {}))
                if ConstantExtraMetadataTensor not in types:
                    out = ConstantExtraMetadataTensor(out)
                    out.constant_attribute = 5
                return out

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)

        with AddConstantMetadataMode():
            out_eager = f(aa)

        compiled_f = torch.compile(f, backend="aot_eager")

        with AddConstantMetadataMode():
            out = compiled_f(aa2)

        self.assertTrue(isinstance(out, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.elem, TwoTensor))
        self.assertTrue(torch.allclose(out_eager, out))

        out_eager.sum().backward()
        out.sum().backward()

        self.assertTrue(torch.allclose(aa.grad, aa2.grad))
        self.assertTrue(isinstance(aa2.grad, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(aa2.grad.elem, TwoTensor))

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    def test_custom_tensor_metadata(self):
        # [FUNCTION DOC] Not run on Windows. Compile a function on a doubly nested metadata
        # wrapper tensor (different constants at each level) and check outputs match eager and
        # gradients keep the nested wrapper structure.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiply the input by its element, the element's element
            # and the element's constant attribute.
            x_elem = x.elem
            x_elem_elem = x_elem.elem
            x_elem_metadata = x_elem.constant_attribute
            return x * x_elem * x_elem_elem * x_elem_metadata

        a = torch.ones(4, requires_grad=True)
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 6
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        custom_aa.constant_attribute = 4

        custom_aa_compile = custom_aa.detach().clone().requires_grad_()
        custom_aa_compile.elem.constant_attribute = 6
        out_eager = f(custom_aa)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_aa_compile)

        self.assertTrue(torch.allclose(out_eager, out))

        out.sum().backward()

        self.assertTrue(isinstance(custom_aa_compile.grad, ConstantExtraMetadataTensor))
        self.assertTrue(
            isinstance(custom_aa_compile.grad.elem, ConstantExtraMetadataTensor)
        )

    def test_nested_subclasses_complicated_inps(self):
        # [FUNCTION DOC] Compile a function with nested TwoTensor inputs plus a plain tensor;
        # check output structure, and that gradients of the nested inputs are nested TwoTensors
        # matching eager values.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Mix of sum of nested tensors, sum of plain parts, and
            # sine-cosine of the first input.
            temp = x + y
            temp_plain = x.a + y.b
            res = temp.sum() + temp_plain.sum()
            return x.sin().cos() + res

        x = torch.ones(4, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        xx = TwoTensor(x, x2)
        xx2 = xx.detach().clone().requires_grad_()

        x_nested = TwoTensor(xx, xx2)
        x_nested_compile = x_nested.detach().clone().requires_grad_()

        y_nested = x_nested.detach().clone().requires_grad_()
        y_nested_compile = y_nested.detach().clone().requires_grad_()

        z = x.detach().clone().requires_grad_()
        z_compile = z.detach().clone().requires_grad_()

        out_eager = f(x_nested, y_nested, z)
        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(x_nested_compile, y_nested_compile, z_compile)
        self.assertTrue(torch.allclose(out_eager, out))

        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, TwoTensor))
        self.assertTrue(isinstance(out.b, TwoTensor))
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        self.assertTrue(isinstance(out.b.b, torch.Tensor))

        out.sum().backward()
        out_eager.sum().backward()

        self.assertTrue(isinstance(x_nested_compile.grad, TwoTensor))
        self.assertTrue(isinstance(x_nested_compile.grad.a, TwoTensor))
        self.assertTrue(isinstance(x_nested_compile.grad.b, TwoTensor))

        self.assertTrue(isinstance(y_nested_compile.grad, TwoTensor))
        self.assertTrue(isinstance(y_nested_compile.grad.a, TwoTensor))
        self.assertTrue(isinstance(y_nested_compile.grad.b, TwoTensor))

        self.assertTrue(torch.allclose(x_nested_compile.grad.a.a, x_nested.grad.a.a))
        self.assertTrue(torch.allclose(x_nested_compile.grad.a.b, x_nested.grad.a.b))
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.a, y_nested.grad.a.a))
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.b, y_nested.grad.a.b))

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    def test_nested_subclasses_complicated_inps_mixed(self):
        # [FUNCTION DOC] Not run on Windows. Compile a function combining a nested TwoTensor
        # input with a doubly nested metadata wrapper input; checks output and both gradients
        # match eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Product of the wrapper tensor, its element, its element's
            # element and constant attribute, plus x.
            y_elem = y.elem
            y_elem_elem = y_elem.elem
            y_elem_metadata = y_elem.constant_attribute
            return y * y_elem * y_elem_elem * y_elem_metadata + x

        x = torch.ones(4, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        xx = TwoTensor(x, x2)
        xx2 = xx.detach().clone().requires_grad_()

        x_nested = TwoTensor(xx, xx2)
        x_nested_compile = x_nested.detach().clone().requires_grad_()

        a = torch.ones(4, requires_grad=True)
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 6
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        custom_aa.constant_attribute = 4

        custom_aa_compile = custom_aa.detach().clone().requires_grad_()
        custom_aa_compile.constant_attribute = 4
        custom_aa_compile.elem.constant_attribute = 6

        compiled_f = torch.compile(f, backend="aot_eager")
        out_eager = f(x_nested, custom_aa)
        out = compiled_f(x_nested_compile, custom_aa_compile)
        self.assertTrue(torch.allclose(out_eager, out))

        out.sum().backward()
        out_eager.sum().backward()

        self.assertTrue(torch.allclose(x_nested_compile.grad, x_nested.grad))
        self.assertTrue(torch.allclose(custom_aa_compile.grad, custom_aa.grad))

    def test_composite_impl_compile(self):
        # [FUNCTION DOC] Check aot_autograd on a module holding a Linear layer (composite op):
        # asserts the forward graph in training mode (transpose and addmm, saving tensors for
        # backward) and again under inference mode (no saved tensors).
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module and its linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Apply the linear layer.
                return self.linear(a)

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(Foo(), inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    t = torch.ops.aten.t.default(primals_1);  primals_1 = None
    addmm = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = None
    return (addmm, primals_3, t)""",
        )

        with torch.inference_mode():
            fw_graph = self.verify_aot_autograd(Foo(), inp, test_mutation=True)
            inp = [torch.ones(3, 3, requires_grad=False)]
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    addmm = torch.ops.aten.addmm.default(arg1_1, arg2_1, t);  arg1_1 = arg2_1 = t = None
    return (addmm,)""",
            )

    def test_outputs_are_aliased(self):
        # [FUNCTION DOC] Check outputs that alias each other (an output and a view of it) keep
        # their aliasing after compilation: mutating one output affects the other the same way
        # as in eager, with and without requires_grad.
        # Tensor, None, int
        def f(a):
            # [FUNCTION DOC] HELPER: Return a product and a flattened view of it.
            b = a.mul(2)
            c = b.view(-1)
            return b, c

        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = torch.ones(3, requires_grad=req_grad)
            out_ref = f(inp)
            out_test = f_compiled(inp)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])
            # Try mutating one of the outputs, which is aliased.
            out_ref[0].mul_(3)
            out_test[0].mul_(3)
            # Assert that the aliasing relationship was preserved
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])

    def test_input_mutation_is_output(self):
        # [FUNCTION DOC] Check an input mutation whose result is also returned; asserts the
        # graph clones the input, multiplies, and returns the value twice.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply the input by two in place and return it.
            a.mul_(2)
            return a

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    return (mul, mul)""",
        )

    def test_input_mutation_multiple(self):
        # [FUNCTION DOC] Check mutating two of three inputs; asserts the graph clones the
        # mutated inputs and returns their updated values before the output.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Double a and c in place, return a plus b plus c.
            a.mul_(2)
            c.mul_(2)
            return a + b + c

        def create_inp(req_grad):
            # [FUNCTION DOC] HELPER: Build three ones matrices with the requested requires_grad.
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None
    add = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
    add_1 = torch.ops.aten.add.Tensor(add, mul_1);  add = None
    return (mul, mul_1, add_1)""",
        )

    def test_input_mutation_return(self):
        # [FUNCTION DOC] Check sin with an out argument writing into a second input, with input
        # mutation kept; asserts a copy_ into the out argument.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Compute sine of a into b and return the result.
            return torch.sin(a, out=b)

        inp = [torch.randn(3, 3), torch.ones(3, 3)]

        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    copy_ = torch.ops.aten.copy_.default(arg1_1, sin);  arg1_1 = sin = None
    return (copy_,)""",
        )

    def test_input_mutation_metadata(self):
        # [FUNCTION DOC] Check a metadata-only input mutation (in-place transpose) with both
        # requires_grad settings.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Transpose a in place and return the sum.
            a.transpose_(1, 0)
            return a + b

        def create_inp(req_grad):
            # [FUNCTION DOC] HELPER: Build two ones matrices with the requested requires_grad.
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

    @parametrize("backend", ["aot_eager", "inductor"])
    @parametrize("view_replay_for_aliased_outputs", [False, True])
    @parametrize("dynamic_shapes", [False, True])
    def test_alias_of_intermediate_detach(
        self, backend, view_replay_for_aliased_outputs, dynamic_shapes
    ):
        # [FUNCTION DOC] Parameterized over backend, view replay config and dynamic shapes.
        # Compiles a function returning a detached transposed intermediate and the non-detached
        # transposed view; checks values, requires_grad flags, that both outputs share storage
        # and the second is a view, and that gradients match eager.
        with patch(
            "torch._functorch.config.view_replay_for_aliased_outputs",
            view_replay_for_aliased_outputs,
        ):

            def fn(x):
                # [FUNCTION DOC] HELPER: Add one, transpose, return the detached view and
                # the view.
                x = x + 1
                a = x.transpose(0, 1)
                return a.detach(), a

            def inp_fn():
                # [FUNCTION DOC] HELPER: Make a ones matrix requiring grad, marking both dims
                # dynamic when dynamic shapes are enabled.
                t = torch.ones(3, 3, requires_grad=True)
                if dynamic_shapes:
                    torch._dynamo.mark_dynamic(t, 0)
                    torch._dynamo.mark_dynamic(t, 1)
                return t

            x_ref = inp_fn()
            y_ref = fn(x_ref)

            x = inp_fn()
            y = torch.compile(fn, backend=backend, fullgraph=True)(x)
            self.assertEqual(y_ref, y)
            y0, y1 = y
            self.assertFalse(y0.requires_grad)
            self.assertTrue(y1.requires_grad)
            # Check that detach and diff view points to the same intermediate tensor storage
            self.assertEqual(y0.data_ptr(), y1.data_ptr())
            self.assertTrue(y1._is_view())

            sum(y_ref).sum().backward()
            sum(y).sum().backward()
            self.assertEqual(x_ref.grad, x.grad)

    def test_input_mutation_storage_resize_up(self):
        # [FUNCTION DOC] Check storage resize up on an input starting with empty storage
        # followed by a copy into it; asserts the graph contains a resize op and a copy into
        # the input.
        def f(a):
            # [FUNCTION DOC] HELPER: Resize the input's storage to 32 bytes, fill it with ones
            # under no_grad, return it plus one.
            torch.ops.inductor.resize_storage_bytes_(a, 32)
            # float32, 4 bytes per element, 32 bytes == 8 elements
            with torch.no_grad():
                a.copy_(torch.ones(8))
            return a + 1

        inp = torch.zeros(8, requires_grad=True)
        # Input starts with zero-size-storage
        inp.untyped_storage().resize_(0)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)
        # Final functionalized graph has two mutation ops:
        # (1) a resize_() to resize input tensor up
        # (2) a copy_() to fill in the resized input with valid data
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1):
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 32);  resize_storage_bytes_ = None
    ones = torch.ops.aten.ones.default([8], device = device(type='cpu'), pin_memory = False)
    copy = torch.ops.aten.copy.default(primals_1, ones);  ones = None
    add = torch.ops.aten.add.Tensor(copy, 1)
    copy_ = torch.ops.aten.copy_.default(primals_1, copy);  primals_1 = copy = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_storage_resize_down(self):
        # [FUNCTION DOC] Check storage resize to zero of an input after use; asserts the graph
        # has only the resize op, with no data mutation retained.
        def f(a):
            # [FUNCTION DOC] HELPER: Take the sine, resize the input's storage to zero bytes,
            # return the sine.
            out = a.sin()
            torch.ops.inductor.resize_storage_bytes_(a, 0)
            return out

        inp = torch.zeros(8, requires_grad=True)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)
        # Final functionalized graph has one mutation ops:
        # (1) a resize_() to resize input tensor down
        # Even though there was technically a "data mutation" on the input (from a.copy_()),
        # We don't include it in the graph since the final input size has zero storage
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1):
    sin = torch.ops.aten.sin.default(primals_1)
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 0);  resize_storage_bytes_ = None
    return (sin, primals_1)""",
        )

    #     def test_input_mutation_storage_resize_up_down(self):
    #         def f(a):
    #             torch.ops.inductor.resize_storage_bytes_(a, 32)
    #             # float32, 4 bytes per element, 32 bytes == 8 elements
    #             with torch.no_grad():
    #                 a.copy_(torch.ones(8))
    #             out = a.sin()
    #             torch.ops.inductor.resize_storage_bytes_(a, 0)
    #             return out

    #         inp = torch.zeros(8, requires_grad=True)
    #         # Input starts with zero-size-storage
    #         inp.untyped_storage().resize_(0)

    #         fw_graph_cell = [None]
    #         compiled_f = aot_function(
    #             f,
    #             fw_compiler=make_boxed_compiler(
    #                 partial(extract_graph, graph_cell=fw_graph_cell)
    #             ),
    #             bw_compiler=nop,
    #             decompositions={},
    #             keep_inference_input_mutations=True,
    #             dynamic=False,
    #         )
    #         out = compiled_f(inp)
    #         # Final graph has two interesting properties:
    #         # (1) no resizes in the functional graph, since the two resizes cancel out
    #         #     and the final size is zero
    #         # (2) no copy_ in the functional graph, even though we copied data into the input,
    #         #     because the input has no storage at the end of graph execution (so no data to copy)
    #         self.assertExpectedInline(
    #             fw_graph_cell[0].code.strip(),
    #             """\
    # def forward(self, primals_1):
    #     ones = torch.ops.aten.ones.default([8], device = device(type='cpu'), pin_memory = False)
    #     copy = torch.ops.aten.copy.default(primals_1, ones);  primals_1 = ones = None
    #     sin = torch.ops.aten.sin.default(copy)
    #     return [sin, copy]""",
    #         )

    # skipped after confirming with @yf225 and @bdhirsh
    @unittest.skipIf(
        True,
        "using set_ unsafely and PT2 FSDP2 no longer uses set_ as used in this test",
    )
    def test_input_mutation_storage_resize_down_and_set_(self):
        # [FUNCTION DOC] Always skipped (skip decorator with a true condition). Mimics FSDP:
        # a custom autograd Function sets a dummy parameter to an all-gathered tensor, then
        # the parameter's storage is resized to zero; asserts graph text with cat, sin,
        # resize and set_.
        # Meant to mimic ppFSDP
        class TracableCreateParameter(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function that sets the placeholder to the given
            # tensor (raises AssertionError if that tensor requires grad); backward routes
            # gradient to the placeholder.
            @staticmethod
            def forward(ctx, tensor, placeholder):
                # [FUNCTION DOC] HELPER: Raise AssertionError if the tensor requires grad, else
                # set the placeholder to it and return the placeholder.
                if tensor.requires_grad:
                    raise AssertionError("Expected tensor to not require grad")
                return placeholder.set_(tensor)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return no gradient for the tensor and pass the gradient
                # to the placeholder.
                return None, grad  # grad flows to placeholder

        def f(dummy_param, param_shard):
            # [FUNCTION DOC] HELPER: Concatenate the shard with itself under no_grad, apply the
            # Function to the dummy param, take its sine, resize its storage to zero, return
            # the sine.
            # simulate allgather
            with torch.no_grad():
                allgather_param = torch.cat([param_shard, param_shard])
            # simulate propagating grad state through dummy param, using data of allgather param
            dummy_param_with_grad_state = TracableCreateParameter.apply(  # noqa: F841
                allgather_param, dummy_param
            )
            out = dummy_param.sin()
            # Resize out dummy param, which now has the allgather data
            torch.ops.inductor.resize_storage_bytes_(dummy_param, 0)
            return out

        # Simulates the local shard of our param
        param_shard = torch.zeros(8, requires_grad=True)
        # The dummy, zero-sized allgathered param that autograd will actually compute gradients on
        dummy_param = torch.zeros(16, requires_grad=True)
        dummy_param.untyped_storage().resize_(0)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(dummy_param, param_shard)
        # Important stuff to point out:
        # (1) We save cat for backward (input to the sin()).
        #     While the original code was dummy_param.sin(),
        #     dummy_param actually contains the `cat` tensor due to the set_() call
        # (2) We emit a cat.resize_storage_(0) in the graph.
        #     After the set_(), cat is the actually data of dummy_param, which is what we call resize_() on
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    cat = torch.ops.aten.cat.default([primals_2, primals_2]);  primals_2 = None
    sin = torch.ops.aten.sin.default(cat)
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(cat, 0);  resize_storage_bytes_ = None
    set_ = torch.ops.aten.set_.source_Tensor(primals_1, cat);  primals_1 = set_ = None
    return (sin, cat)""",
        )

    def test_input_mutation_storage_resize_before_set_(self):
        # [FUNCTION DOC] Check that resizing an input's storage to zero and then setting it to
        # another tensor compiles and runs without error.
        def f(a):
            # [FUNCTION DOC] HELPER: Under no_grad resize the input's storage to zero then set
            # it to a new ones tensor.
            with torch.no_grad():
                torch.ops.inductor.resize_storage_bytes_(a, 0)
                a.set_(torch.ones(2))

        inp = torch.zeros(8, requires_grad=True)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)

    # def test_input_mutation_storage_resize_not_supported(self):
    #     def f(a):
    #         a.mul_(2)
    #         torch.ops.inductor.resize_storage_bytes_(a, 0)
    #         return a

    #     inp = torch.zeros(8, requires_grad=True)

    #     with self.assertRaisesRegex(
    #         AssertionError, "the input has other mutations that we cannot"
    #     ):
    #         compiled_f = aot_function(
    #             f,
    #             fw_compiler=nop,
    #             bw_compiler=nop,
    #             decompositions={},
    #             keep_inference_input_mutations=True,
    #             dynamic=False,
    #         )
    #         out = compiled_f(inp)

    def test_input_output_aliase_custom_autograd_function(self):
        # [FUNCTION DOC] Check an autograd Function that returns its input unchanged
        # (input-output alias) works through aot_autograd.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function returning its input; backward halves
            # the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input unchanged.
                return x

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Return half of the incoming gradient.
                return gx * 0.5

        def f(x):
            # [FUNCTION DOC] HELPER: Apply the custom Function.
            return Foo.apply(x)

        inp = [torch.ones(2, 2, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)

    def test_input_mutation_requires_grad_detach(self):
        # [FUNCTION DOC] Check mutation of an input through a detached view (does not take
        # part in autograd) still compiles and matches eager, with and without making the
        # input a non-leaf.
        # Here, "a" requires grad, and gets mutated, so we append a copy_() to the end of the graph.
        # Its mutation doesn't take part in autograd though, because we mutated a detach'd view.
        # Need to make sure that this copy_() doesn't error, and doesn't participate in autograd either.
        def f(a):
            # [FUNCTION DOC] HELPER: Double a detached alias of the input in place and return
            # the input plus three.
            a.detach().mul_(2)
            return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)
        inp = [torch.ones(4, requires_grad=True)]
        # test_mutation=True will first do some compute on inp, so it is no longer an autograd leaf
        # by the time it becomes a graph input. Good to test both cases.
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_hidden_from_autograd_aliasing(self):
        # [FUNCTION DOC] Check a mutation under no_grad through a view of the input is kept
        # inside the graph; asserts the graph ends with a copy into the input.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply a view of the input by two under no_grad and
            # return the input plus one.
            a_alias = a.view(-1)
            with torch.no_grad():
                a_alias.mul_(2)
            return a + 1

        inp = [torch.ones(4, requires_grad=True)]
        # The important bit: we detected that the input mutation is safe
        # to include **inside** the graph, since it was under no_grad
        # (so all we need to do is use mark_dirty() on the input to bump the VC)
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    view = torch.ops.aten.view.default(primals_1, [-1])
    mul = torch.ops.aten.mul.Tensor(view, 2);  view = None
    view_1 = torch.ops.aten.view.default(mul, [4]);  mul = None
    add = torch.ops.aten.add.Tensor(view_1, 1)
    copy_ = torch.ops.aten.copy_.default(primals_1, view_1);  primals_1 = view_1 = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad(self):
        # [FUNCTION DOC] Check an input mutation under no_grad is kept inside the training graph
        # (copy into input) even though the input requires grad.
        def f(a):
            # [FUNCTION DOC] HELPER: Double the input under no_grad and return it plus three.
            with torch.no_grad():
                a.mul_(2)
            return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        # Even though the input requires_grad, we expect the keep the input mutation in the graph
        # (Even though this is a training graph!)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 3)
    copy_ = torch.ops.aten.copy_.default(primals_1, mul);  primals_1 = mul = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad_inference_graph(self):
        # [FUNCTION DOC] Same as the no_grad mutation test but the whole function body,
        # including the returned value, runs under no_grad; asserts the same copy into
        # the input.
        def f(a):
            # [FUNCTION DOC] HELPER: Under no_grad double the input and return it plus three.
            with torch.no_grad():
                a.mul_(2)
                return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        # Even though the input requires_grad, we expect the keep the input mutation in the graph
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 3)
    copy_ = torch.ops.aten.copy_.default(arg0_1, mul);  arg0_1 = mul = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad_detach_mixed(self):
        # [FUNCTION DOC] Check an input mutated three ways (through a detached alias, normally,
        # and under no_grad); only the normal mutation should take part in autograd.
        # Perform a mix of mutations on a:
        # 1 normal, 1 in no_grad, 1 on a detach'd tensor.
        # Only the first should participate in gradient computation.
        def f(a):
            # [FUNCTION DOC] HELPER: Mutate the input via detached alias, normally, and under
            # no_grad, then return it plus five.
            a.detach().mul_(2)
            a.mul_(3)
            with torch.no_grad():
                a.mul_(4)
            return a + 5

        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_metadata2(self):
        # [FUNCTION DOC] Check a metadata mutation (transpose) followed by a data mutation of
        # an input.
        def f(a):
            # [FUNCTION DOC] HELPER: Transpose in place, double in place, return plus one.
            a.transpose_(1, 0)
            a.mul_(2)
            return a + 1

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_batchnorm(self):
        # [FUNCTION DOC] Check batch norm with mutated running statistics under inductor-like
        # decompositions (legit functional batch norm plus its backward); the mutated inputs are
        # also saved for backward.
        def f(inpt, weight, bias, running_mean, running_var):
            # [FUNCTION DOC] HELPER: Call the legit native batch norm in training mode with
            # momentum 0.5.
            # This is additionally a good test, because the input tensors that we mutate
            # are *also* saved for backwards.
            # This tests that what we save for the backward is actually cloned inputs,
            # and not the original inputs that got mutated.
            return torch._native_batch_norm_legit(
                inpt, weight, bias, running_mean, running_var, True, 0.5, 1e-5
            )

        def create_inp(req_grad):
            # [FUNCTION DOC] HELPER: Build five inputs: a 4D activation, weight and bias with
            # the requested requires_grad, and running mean and variance.
            return [
                torch.ones(2, 5, 5, 5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5),
                torch.ones(5),
            ]

        from torch._decomp import get_decompositions

        # This simulates what inductor does (running the fw + bw decompositions)
        decompositions = get_decompositions(
            [
                torch.ops.aten._native_batch_norm_legit_functional,
                torch.ops.aten.native_batch_norm_backward,
            ]
        )
        self.verify_aot_autograd(
            f, create_inp(True), test_mutation=True, decompositions=decompositions
        )
        self.verify_aot_autograd(
            f, create_inp(False), test_mutation=True, decompositions=decompositions
        )

    def test_batchnorm_inference(self):
        # [FUNCTION DOC] Compile a BatchNorm2d module in eval mode under no_grad with input
        # mutations kept and assert no copy_ appears in the forward graph.
        inp = [
            torch.ones(2, 5, 5, 5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5),
            torch.ones(5),
        ]

        m = torch.nn.BatchNorm2d(4, 4)
        m.eval()
        fw_graph_cell = [None]
        inp = torch.ones(4, 4, 4, 4)
        fw_graph_cell = [None]
        compiled_m = aot_module(
            m,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp = torch.ones(4, 4, 4, 4)
        with torch.no_grad():
            compiled_m(inp)
        # expectation: there are no copy_() calls in the decomposed batch norm when running under training=False (eval mode)
        code = fw_graph_cell[0].code.strip()
        self.assertTrue("copy_" not in str(code))

    def test_input_output_view_simple(self):
        # [FUNCTION DOC] Check an output that is a view of an input; asserts the compiled graph
        # only contains the view op.
        def f(a):
            # [FUNCTION DOC] HELPER: Return a flattened view of the input.
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # Outputs that alias inputs are pulled out of the graph entirely, so we don't compile anything here
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    view = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
    return (view,)""",
        )

    def test_input_output_view_mutate_multiple(self):
        # [FUNCTION DOC] Check inputs mutated while others are returned as views; asserts the
        # graph returns the two mutated values then the view outputs.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Double a, triple c in place, return views of b and c.
            a.mul_(2)
            c.mul_(3)
            return b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            # [FUNCTION DOC] HELPER: Build three non-leaf matrices with the
            # requested requires_grad.
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # The original function returned two outputs, both of which aliased inputs.
        # We expect two outputs in the functional graph, a_updated and c_updated.
        # The actual aliased outputs themselves aren't in the compiled forward graph;
        # Instead, they're generated outside of  the graph.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 3);  clone_1 = None
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    view_2 = torch.ops.aten.view.default(mul_1, [2, 2])
    return (mul, mul_1, view, view_2)""",
        )

    def test_input_output_view_metadata_mutate_multiple(self):
        # [FUNCTION DOC] Check metadata mutation plus data mutation on different inputs with all
        # outputs being views of inputs; only the data mutation of one input and a metadata
        # mutation appear in graph outputs.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Triple b in place, transpose c in place, return views of
            # all three inputs.
            b.mul_(3)
            c.t_()
            return a.view(2, 2), b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            # [FUNCTION DOC] HELPER: Build three non-leaf matrices with the
            # requested requires_grad.
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # Important thing to check here: of the three inputs:
        # Only the b.mul_(3) should show up in the graph (we functionalize it and return it).
        # Everything else that does not show up in the graph includes:
        # - The metadata mutation on c (we do it outside the graph)
        # - All 3 original fw outputs, which are aliases of inputs (we regenerate them outside of the graph)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_2);  primals_2 = None
    view = torch.ops.aten.view.default(primals_3, [2, 2]);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 3);  clone = None
    t = torch.ops.aten.t.default(view);  view = None
    view_1 = torch.ops.aten.view.default(primals_1, [2, 2]);  primals_1 = None
    view_3 = torch.ops.aten.view.default(t, [2, 2])
    view_4 = torch.ops.aten.view.default(mul, [2, 2])
    return (mul, t, view_1, view_4, view_3)""",
        )

    def test_input_mutation_and_output_view(self):
        # [FUNCTION DOC] Check a function that mutates its input and returns a view of it;
        # asserts the graph returns the updated value and a view.
        def f(a):
            # [FUNCTION DOC] HELPER: Add one in place and return a flattened view.
            a.add_(1)
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # Here, total # of outputs is 1 because:
        # - num_mutated_inps = 1 (a_updated)
        # - num_fw_outputs = 0 (the output is an alias of the input, so we move it outside the compiled fw)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    view_1 = torch.ops.aten.view.default(add, [-1])
    return (add, view_1)""",
        )

    def test_input_mutation_output_view_multiple(self):
        # [FUNCTION DOC] Check a mix of metadata mutation (transpose), data mutation, a diagonal
        # view output and plain outputs; asserts the resulting graph text.
        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Transpose b in place, add one to c in place, return d plus
            # one, b's diagonal and a plus c.
            b.transpose_(1, 0)
            c.add_(1)
            return d + 1, b.diagonal(), a + c

        def create_inp(req_grad):
            # [FUNCTION DOC] HELPER: Build four 2 by 2 non-leaf float inputs with the
            # requested requires_grad.
            return [
                torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
                .view(2, 2)
                .add(1),
                torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
                .view(2, 2)
                .add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    clone = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(primals_4, 1);  primals_4 = None
    diagonal = torch.ops.aten.diagonal.default(transpose)
    add_2 = torch.ops.aten.add.Tensor(primals_1, add);  primals_1 = None
    return (transpose, add, add_1, diagonal, add_2)""",
        )

    def test_output_aliases_intermediate_single(self):
        # [FUNCTION DOC] Check an output that is a view of an intermediate; asserts the compiled
        # graph returns the view directly.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply by three and return a flattened view.
            out = torch.mul(a, 3)
            return out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # In AOTAutograd, we are obligated to make the compiled forward directly return `out`,
        # and reconstruct `out.view(-1)` as a fresh output.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1]);  mul = None
    return (view,)""",
        )

    def test_output_aliases_input_multi_output_view_should_raise_autograd_error(self):
        # [FUNCTION DOC] Check that returning multiple outputs from a multi-output view (unbind)
        # of an input and then mutating them raises the same autograd RuntimeError as eager, in
        # three scenarios (mutating an output, mutating the input then reading grad_fn, mutating
        # a detached output).
        def f1(a):
            # [FUNCTION DOC] HELPER: Return the list of unbound rows.
            return list(a.unbind(0))

        f1_compiled = aot_function(f1, nop)

        inp1 = torch.ones(3, 3, requires_grad=True).clone()
        inp2 = torch.ones(3, 3, requires_grad=True).clone()
        inp3 = torch.ones(3, 3, requires_grad=True).clone()

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            out_test1 = f1_compiled(inp1)
            # This raises a runtime error from autograd in eager mode
            out_test1[0].mul_(2)

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            out_test2 = f1_compiled(inp2)
            inp2.mul_(2)
            # In eager mode, if we mutate a tensor, any multi-output-view aliases
            # get their grad_fn replaced with error nodes, so accessing grad_fn should error
            out_test2[0].grad_fn

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            f1_compiled(inp3)
            out_test1[0].detach().mul_(2)
            # The above case also applies to detached aliases (they turn the multi-output-view
            # alias's grad_fns into error nodes)
            out_test2[0].grad_fn

    def test_output_aliases_input_multi_output_view(self):
        # [FUNCTION DOC] Check outputs from a multi-output view: the backward nodes are the
        # compiled function backward (aliasing hidden from autograd), and gradients match eager.
        # A second function that also returns a plain view of the input falls back to
        # regenerating aliases, so unbind backward nodes appear, mutations propagate, and
        # gradients match.
        # All aliased outs are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f1(a):
            # [FUNCTION DOC] HELPER: Return the list of unbound rows.
            return list(a.unbind(0))

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)

        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # Several of the outputs are from multi-output views.
        # However: they are part of the same alias set as "a", and "a.view(out.shape)",
        # which are both user-visible.
        # AOTAutograd will not try to be smart here and hide the aliasing relationships from autograd.
        # Instead, it will perform its "output aliases input" logic, and regenerate all aliases.
        def f3(a):
            # [FUNCTION DOC] HELPER: Return the unbound rows plus a view of the input with the
            # same shape.
            return *list(a.unbind(0)), a.view(a.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)

        inp_ref_clone = inp_ref.clone()
        inp_clone = inp.clone()
        out_ref = f3(inp_ref_clone)
        out_test = f3_compiled(inp_clone)
        self.assertTrue(all("UnbindBackward" in str(o.grad_fn) for o in out_test[:3]))

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        # Also mutate the input, which should affect the aliased output.
        inp_ref_clone.view(-1).mul_(3)
        inp_clone.view(-1).mul_(3)
        # Do backward
        (inp_ref + out_ref[-1]).sum().backward()
        (inp + out_test[-1]).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_multi_output_view(self):
        # [FUNCTION DOC] Check aliasing among outputs that are views of a graph intermediate.
        # Four scenarios: all outputs from a multi-output view (aliasing hidden from autograd,
        # backward nodes are the compiled function backward); the same plus the intermediate
        # itself returned; the same plus a plain view of it; and a case with ordinary aliases
        # mixed in that triggers the intermediate-base path. Each compares mutation and
        # gradients with eager.
        # All aliased outs are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f1(a):
            # [FUNCTION DOC] HELPER: Multiply by three and return the list of unbound rows.
            out = torch.mul(a, 3)
            return list(out.unbind(0))

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)

        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # All aliased outs but one are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f2(a):
            # [FUNCTION DOC] HELPER: Multiply by three and return the unbound rows followed by
            # the intermediate itself.
            out = torch.mul(a, 3)
            return *list(out.unbind(0)), out

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f2_compiled = aot_function(f2, nop)

        out_ref = f2(inp_ref)
        out_test = f2_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # All aliased outs but one are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f3(a):
            # [FUNCTION DOC] HELPER: Multiply by three and return the unbound rows followed by a
            # same-shape view of the intermediate.
            out = torch.mul(a, 3)
            return *list(out.unbind(0)), out.view(out.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)

        out_ref = f3(inp_ref)
        out_test = f3_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # There are 5 outputs that all alias each other.
        # 3 of them come from multi-output views, but the other 3 are "ordinary" aliases.
        # Therefore, AOTAutograd will not attempt the multi-output-view optimization,
        # and apply the intermediate_base logic to all aliases.
        # (In theory we could probably get AOTAutograd to only apply the intermediate base
        # logic to the last 2 outputs and not the first 3. We should probably
        # just do the graph partitioning defined in this doc instead though).
        # https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit
        def f4(a):
            # [FUNCTION DOC] HELPER: Multiply by three and return the unbound rows, the
            # intermediate, and a same-shape view of it.
            out = torch.mul(a, 3)
            # also return the graph intermediate directly,
            # which will force AOTAutograd to do the "intermediate base" logic.
            # (Why? The user can mutate "out", which should change the autograd metadata
            #  of the other aliased outputs)
            return *list(out.unbind(0)), out, out.view(out.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f4_compiled = aot_function(f4, nop)

        out_ref = f4(inp_ref)
        out_test = f4_compiled(inp)
        # Mutate the last output of f4 (autograd will allow this, since it is not a multi-output view,
        # as long as *only* the non-multi-output views participate in the backward)
        # Note: We could probably try to hide **only** the multi-output views from autograd here
        # and only do the intermediate base logic for the last two aliases.
        # Longer term solution of graph partitioning is probably cleaner though (see the note).
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)

        out_ref_sum = out_ref[-1] + out_ref[-2]
        out_test_sum = out_test[-1] + out_test[-2]
        out_ref_sum.sum().backward()
        out_test_sum.sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_mutation_linear(self):
        # [FUNCTION DOC] Check that mutating an output that is a flattened view of an
        # intermediate works under inductor decompositions (unsafe view becomes view); mutated
        # results must match eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Add one and return a flattened view.
            return (x + 1).view(-1)

        inp = [torch.ones(3, 3, requires_grad=True)]
        # use inductor's decomps (which will e.g. turn _unsafe_view() into view())
        from torch._inductor.decomposition import decompositions

        f_compiled = aot_function(f, nop, decompositions=decompositions)

        out_ref = f(*inp)
        out_test = f_compiled(*inp)

        out_ref.mul_(2)
        out_test.mul_(2)
        self.assertEqual(out_ref, out_test)

    def test_output_aliases_intermediate_no_grad(self):
        # [FUNCTION DOC] Check an output aliasing an intermediate that does not require grad: no
        # intermediate base is added as a graph output; asserts the graph text.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return a flattened view of a product and the second input
            # plus one.
            out = torch.mul(a, 3)
            # First output is an alias of an intermediate that doesn't require grad
            return out.view(-1), b.add(1)

        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # important bit: we don't bother generating an intermediate base as an output in the graph,
        # because the intermediate base itself didn't require gradients.
        # (the only problematic case is when both the base and the aliasesed output require gradients).
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1]);  mul = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    return (view, add)""",
        )

    def test_output_aliases_intermediate_no_grad_view(self):
        # [FUNCTION DOC] Check a view of a differentiable intermediate created under no_grad: it
        # keeps requires_grad like eager but is not a differentiable output; the other output
        # participates in backward and gradients match eager.
        # View of a differentiable intermediate created under no_grad().
        # The view inherits requires_grad from the base but has no grad_fn,
        # so it should not be treated as a differentiable output.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply by three, take a flattened view under no_grad,
            # return the sum and that view.
            out = torch.mul(a, 3)
            with torch.no_grad():
                out_view = out.view(-1)
            return out.sum(), out_view

        inp = torch.ones(3, 3, requires_grad=True)
        compiled_f = aot_function(f, nop)

        ref_loss, ref_view = f(inp)
        test_loss, test_view = compiled_f(inp)
        # The differentiable output (sum) participates in backward.
        self.assertEqual(ref_loss, test_loss)
        self.assertTrue(test_loss.requires_grad)
        # The no_grad view still has requires_grad (matching eager) but
        # does not participate in backward.
        self.assertEqual(ref_view, test_view)
        self.assertTrue(test_view.requires_grad)

        ref_loss.backward()
        ref_grad = inp.grad.clone()
        inp.grad = None
        test_loss.backward()
        test_grad = inp.grad
        self.assertEqual(ref_grad, test_grad)

    def test_output_aliases_intermediate_returned_multiple_times(self):
        # [FUNCTION DOC] Check returning an intermediate, a view of it, and the intermediate
        # again, for inputs with and without requires_grad.
        def f(a):
            # [FUNCTION DOC] HELPER: Return the intermediate, a flattened view of it, and the
            # intermediate again.
            out = torch.mul(a, 3)
            out_view = out.view(-1)
            return out, out_view, out

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_multiple(self):
        # [FUNCTION DOC] Check two views of the same intermediate returned; asserts the graph
        # returns both views plus the intermediate as an extra output.
        def f(a):
            # [FUNCTION DOC] HELPER: Return two flattened views of a product.
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate these two output views in the epilogue.
            return out.view(-1), out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    view_1 = torch.ops.aten.view.default(mul, [-1])
    return (view, view_1, mul)""",
        )

    def test_output_aliases_intermediate_and_returned(self):
        # [FUNCTION DOC] Check returning a view of an intermediate and the intermediate itself;
        # asserts the graph returns the view and the intermediate.
        def f(a):
            # [FUNCTION DOC] HELPER: Return a flattened view of a product and the product.
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out.view(-1), out

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    return (view, mul)""",
        )

    def test_output_aliases_intermediate_and_returned_flipped(self):
        # [FUNCTION DOC] Same as the previous test with the output order flipped (intermediate
        # first, then its view); asserts the graph text.
        def f(a):
            # [FUNCTION DOC] HELPER: Return the product and a flattened view of it.
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out, out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    return (mul, view)""",
        )

    def test_output_aliases_intermediate_and_returned_different_grad(self):
        # [FUNCTION DOC] Check outputs: a view of an intermediate, the intermediate, and a
        # detached select of it; asserts the graph includes the view, select and detach.
        def f(a):
            # [FUNCTION DOC] HELPER: Return a flattened view, the product and the detached
            # first row.
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out.view(-1), out, out[0].detach()

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    select = torch.ops.aten.select.int(mul, 0, 0)
    detach = torch.ops.aten.detach.default(select);  select = None
    return (view, mul, detach)""",
        )

    def test_output_aliases_intermediate_inplace_view(self):
        # [FUNCTION DOC] Defines a function with an in-place view op on an intermediate but runs
        # no assertions (the verification call is commented out pending issue 90507), so
        # effectively a no-op test.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply by three, transpose in place and return;
            # never called.
            out = torch.mul(a, 3)
            out.t_()
            return out

        # TODO: fix this test.
        # See https://github.com/pytorch/pytorch/issues/90507
        # self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_inplace_view_with_detach(self):
        # [FUNCTION DOC] Check an intermediate transposed in place and then detached in
        # place: AOT autograd needs no alias handling; asserts the graph text with a
        # transpose and an add.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply by three, transpose in place, detach in place,
            # return it and the input plus one.
            out = torch.mul(a, 3)
            out.t_()
            out.detach_()
            # Thanks to the detach_() AOT Autograd doesn't need to do anything.
            # `out` will show up as having OutputType.non_alias,
            # and ._is_view() == False
            return out, a + 1

        inp = [torch.ones(2, 4, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3)
    t = torch.ops.aten.t.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(primals_1, 1);  primals_1 = None
    return (t, add)""",
        )

    def test_output_aliases_intermediate_inplace_view_and_view(self):
        # [FUNCTION DOC] Defines a function with views taken before and after an in-place
        # transpose of an intermediate but runs no assertions (verification commented out), so
        # effectively a no-op test.
        def f(a):
            # [FUNCTION DOC] HELPER: Return an unsqueezed view before the in-place transpose,
            # the intermediate, and an unsqueezed view after; never called.
            out = torch.mul(a, 3)
            out_view = out.unsqueeze(0)
            out.t_()
            out_view2 = out.unsqueeze(0)
            return out_view, out, out_view2

        inp = [torch.ones(2, 4, requires_grad=True)]  # noqa: F841

        # TODO: fix this test.
        # See <github issue link>
        # self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_multiple_mixed(self):
        # [FUNCTION DOC] Check outputs that are views of two different intermediates, one of
        # them used twice; asserts the graph text including the extra intermediate output.
        def f(a):
            # [FUNCTION DOC] HELPER: Return a flattened view of one product, a transposed second
            # product and a transposed first product.
            out1 = torch.mul(a, 3)
            out2 = torch.mul(a, 4)
            # AOTAutograd should manually generate these two output views in the epilogue.
            return out1.view(-1), out2.transpose(1, 0), out1.transpose(1, 0)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3)
    mul_1 = torch.ops.aten.mul.Tensor(primals_1, 4);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    transpose = torch.ops.aten.transpose.int(mul_1, 1, 0);  mul_1 = None
    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)
    return (view, transpose, transpose_1, mul)""",
        )

    def test_output_all_alias_types(self):
        # [FUNCTION DOC] Check all three kinds of alias metadata returned by the compiled
        # forward (output views of inputs, of intermediates, and input metadata mutations);
        # asserts the graph text.
        # There are 3 types of aliasing that require us to return metadata in the compiled fw:
        # (1) outputs that are views of inputs
        # (2) outputs that are views of intermediates
        # (3) inputs that get metadata mutations
        # test all 3 of them here
        def f(a):
            # [FUNCTION DOC] HELPER: Transpose the input in place, double it, and return squeeze
            # and transpose views of the result plus an unsqueezed view of the input.
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.squeeze(), tmp.transpose(1, 0), a.unsqueeze(0)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Build one ones tensor (cloned, with requested
            # requires_grad) and return it as both function and graph inputs.
            x = torch.ones(1, 2, 4, requires_grad=req_grad).clone()
            return [(x,), (x,)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # TODO: make this test run with dynamic shapes so it is more meaningful
        # metadata output order: (a_updated_meta, out1_meta, out2_meta, out3_meta)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    view = torch.ops.aten.view.default(primals_1, [1, 2, 4]);  primals_1 = None
    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None
    mul = torch.ops.aten.mul.Tensor(transpose, 2)
    squeeze = torch.ops.aten.squeeze.default(mul)
    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)
    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0)
    return (transpose, squeeze, transpose_1, unsqueeze, mul)""",
        )

    @parametrize("req_grad", [False, True])
    def test_subclass_metadata_mutation(self, req_grad):
        # [FUNCTION DOC] Parameterized by requires_grad. Verify a metadata mutation (in-place
        # transpose) followed by a data op with TwoTensor subclass inputs.
        def f(a):
            # [FUNCTION DOC] HELPER: Transpose in place, double, return the transposed result.
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.transpose(1, 0)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Build one cloned ones tensor with the requested
            # requires_grad and return it as both function and graph inputs.
            x = torch.ones(1, 2, 4, requires_grad=req_grad).clone()
            return [(x,), (x,)]

        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=req_grad),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

    def test_subclass_metadata_mutation_aot_function_metadata(self):
        # [FUNCTION DOC] Check aot_function with TwoTensor inputs created under TwoTensorMode
        # and an in-place transpose: compiled outputs, mutated inputs and their
        # shape/stride/offset metadata match eager.
        def f(a):
            # [FUNCTION DOC] HELPER: Transpose in place, double, return the transposed result.
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.transpose(1, 0)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            decompositions=None,
            keep_inference_input_mutations=True,
            dynamic=False,
        )

        with TwoTensorMode():
            ref_inp = torch.ones(1, 2, 4).clone()
        with TwoTensorMode():
            test_inp = torch.ones(1, 2, 4).clone()

        ref_out = f(ref_inp)
        test_out = compiled_f(test_inp)

        self.assertEqual(test_inp, ref_inp)
        self.assertEqual(test_out, ref_out)
        self.assertTensorMetadataEqual(test_inp, ref_inp)
        self.assertTensorMetadataEqual(test_out, ref_out)

    @parametrize("req_grad", [False, True])
    @skipIfDynamoInput("Dynamo fails to fakeify non-contiguous TwoTensor inputs")
    def test_subclass_metadata_mutation_noncontiguous_input(self, req_grad):
        # [FUNCTION DOC] Parameterized by requires_grad; skipped under the dynamo variant.
        # Same as the subclass metadata mutation test but the input is a non-contiguous
        # slice; checked through the harness and again directly via aot_function including
        # metadata equality.
        def f(a):
            # [FUNCTION DOC] HELPER: Transpose in place, double, return the transposed result.
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.transpose(1, 0)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Build a cloned ones tensor, take a strided column slice,
            # and return it as both function and graph inputs.
            x = torch.ones(2, 4, requires_grad=req_grad).clone()[:, ::2]
            return [(x,), (x,)]

        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=req_grad),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            decompositions=None,
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        with TwoTensorMode():
            ref_inp = torch.ones(2, 4, requires_grad=req_grad).clone()[:, ::2]
        with TwoTensorMode():
            test_inp = torch.ones(2, 4, requires_grad=req_grad).clone()[:, ::2]

        ref_out = f(ref_inp)
        test_out = compiled_f(test_inp)

        self.assertEqual(test_inp, ref_inp)
        self.assertEqual(test_out, ref_out)
        self.assertTensorMetadataEqual(test_inp, ref_inp)
        self.assertTensorMetadataEqual(test_out, ref_out)

    def test_input_data_and_metadata_mutation(self):
        # [FUNCTION DOC] Check an input that gets both a metadata mutation (in-place transpose)
        # and a data mutation of one row; asserts the graph text.
        def f(a):
            # [FUNCTION DOC] HELPER: Transpose in place, double the first row in place, return a
            # view with the same shape.
            a.t_()
            a[0].mul_(2)
            return a.view(a.shape)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    t = torch.ops.aten.t.default(clone)
    select = torch.ops.aten.select.int(t, 0, 0);  t = None
    mul = torch.ops.aten.mul.Tensor(select, 2);  select = None
    t_1 = torch.ops.aten.t.default(clone);  clone = None
    select_scatter = torch.ops.aten.select_scatter.default(t_1, mul, 0, 0);  t_1 = mul = None
    t_2 = torch.ops.aten.t.default(select_scatter);  select_scatter = None
    t_4 = torch.ops.aten.t.default(t_2)
    t_6 = torch.ops.aten.t.default(t_2);  t_2 = None
    view_1 = torch.ops.aten.view.default(t_6, [3, 3]);  t_6 = None
    return (t_4, view_1)""",
        )

    def test_view_and_inplace_view(self):
        # [FUNCTION DOC] Check a view of one input and an in-place metadata mutation of another;
        # asserts the graph returns the transposed input plus views.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Transpose a in place and return same-shape views of b
            # and a.
            a.t_()
            return b.view(b.shape), a.view(a.shape)

        def create_inp(req_grad):
            # [FUNCTION DOC] HELPER: Build two ones matrices with the requested requires_grad.
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    view = torch.ops.aten.view.default(arg1_1, [3, 3]);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(t, [3, 3])
    return (t, view, view_1)""",
        )

    def test_view_detach(self):
        # [FUNCTION DOC] Check an output that is a detached alias created before an in-place
        # multiply, together with the mutated input.
        def f(a):
            # [FUNCTION DOC] HELPER: Detach the input, double the input in place, return both.
            tmp = a.detach()
            a.mul_(2)
            return a, tmp

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_inplace_requires_grad_true(self):
        # [FUNCTION DOC] Check an input whose requires_grad is switched on in place inside the
        # function; asserts the graph simply multiplies both inputs.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Enable requires_grad on a in place and return products of
            # both inputs.
            a.requires_grad_(True)
            return a.mul(3), b.mul(4)

        inp = [
            # First inp doesn't require grad, but we switch it on
            torch.ones(3, 3, requires_grad=False),
            torch.ones(3, 3, requires_grad=True),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 4);  primals_2 = None
    return (mul, mul_1)""",
        )

    # This is a torture test:
    # a and b get turned into a synthetic base in the compiled graph
    # One gets a data mutation, the other gets a metadata mutation.
    # We need to make sure that the metadata mutation gets propagated
    # back to the original input.
    @skipIfDynamoInput("Dynamo removes runtime error")
    def test_input_data_and_metadata_mutation_aliases_other_input(self):
        # [FUNCTION DOC] Skipped under the dynamo variant. Torture test where two aliased inputs
        # get a data mutation and a metadata mutation; works for plain inputs, and with subclass
        # inputs expects a RuntimeError about aliased inputs mutated in the graph.
        # a and b are aliased
        def f(a, b):
            # [FUNCTION DOC] HELPER: Double a in place, transpose b in place, return
            # their product.
            a.mul_(2)
            b.t_()
            return a.mul(b)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Build a base tensor and two identical row views of a
            # non-leaf derived from it; returns the base as function input and the two views as
            # graph inputs.
            base = torch.ones(2, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[0]
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=True),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

    # https://github.com/pytorch/pytorch/issues/106456
    def test_input_mutation_noncontiguous(self):
        # [FUNCTION DOC] Regression test for issue 106456: mutating a non-contiguous view input,
        # verified for plain and subclass inputs and both requires_grad settings.
        def f(a):
            # [FUNCTION DOC] HELPER: Double the input in place and return it plus one.
            a.mul_(2)
            return a + 1

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Build a base tensor and a non-contiguous column view of a
            # non-leaf derived from it; returns the base and the view.
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            # create a non-contiguous view to pass as an input to the compiler
            inp = x[:, 0]
            return [base], [inp]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

    def test_backward_mutation_data(self):
        # [FUNCTION DOC] Check a custom autograd Function whose backward mutates a saved input;
        # allowed when that buffer does not require grad, and also verified when both inputs
        # require grad.
        class BwMutation(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function that clones in forward and, in backward,
            # doubles the saved input in place and returns a cloned gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input for backward and return a clone.
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Double the saved input in place and return a clone of
                # the gradient.
                (x,) = ctx.saved_tensors
                # bw mutation
                x.mul_(2)
                return grad_output.clone()

        def f(a, b):
            # [FUNCTION DOC] HELPER: Apply the mutating Function to b and multiply with a.
            out = BwMutation.apply(b)
            return a * out

        inp_no_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        # Mutation on buffer that does not require grad during the backward is allowed
        self.verify_aot_autograd(f, inp_no_grad, test_mutation=True)

        inp_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp_grad, test_mutation=True)

    def test_fw_bw_mutation_no_functionalization1(self):
        # [FUNCTION DOC] Use compiled_function with functionalization disabled over a Function
        # that mutates its input and an intermediate in forward and a saved tensor in backward.
        # Checks outputs and gradient match eager, the forward graph has two mutation ops and
        # the backward graph has one.
        class FwBwMutation(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function with foreach in-place multiplications on its
            # input, an intermediate, and the saved tensor in backward.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Mutate b and an intermediate in place, save the
                # intermediate, return its product with a.
                # input mutation
                torch._foreach_mul_([b], [2])
                x = b + 1
                # intermediate mutation
                torch._foreach_mul_([x], [3])
                ctx.save_for_backward(x)
                return x * a

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Mutate the saved tensor in place and return two
                # products of the gradient with it.
                (x,) = ctx.saved_tensors
                # bw mutation
                torch._foreach_mul_([x], [4])
                return grad_output * x, grad_output * x

        def f(a, b):
            # [FUNCTION DOC] HELPER: Apply the Function, then in-place sine and clone.
            return FwBwMutation.apply(a, b).sin_().clone()

        inps = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]
        inps_ref = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        fw_graph = [None]
        bw_graph = [None]

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler that stores the graph in a list cell and
            # returns it unchanged.
            fw_graph[0] = gm
            return gm

        def bw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backward compiler that stores the graph in a list cell and
            # returns it unchanged.
            bw_graph[0] = gm
            return gm

        compiled_f = compiled_function(
            f,
            fw_compiler,
            bw_compiler,
            dynamic=False,
            partition_fn=default_partition,
            keep_inference_input_mutations=True,
            disable_functionalization=True,
        )

        out_ref = f(*inps_ref)
        out = compiled_f(*inps)
        self.assertEqual(out, out_ref)

        out_ref.sum().backward()
        out.sum().backward()
        self.assertEqual(inps_ref[0].grad, inps[0].grad)

        # important bit: there are 2 mutations in the fw
        self.assertExpectedInline(
            fw_graph[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    _foreach_mul_ = torch.ops.aten._foreach_mul_.ScalarList([primals_2], [2]);  _foreach_mul_ = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    _foreach_mul__1 = torch.ops.aten._foreach_mul_.ScalarList([add], [3]);  _foreach_mul__1 = None
    mul = torch.ops.aten.mul.Tensor(add, primals_1);  primals_1 = None
    clone = torch.ops.aten.clone.default(mul)
    sin_ = torch.ops.aten.sin_.default(mul);  mul = None
    clone_1 = torch.ops.aten.clone.default(sin_);  sin_ = None
    return (clone_1, add, clone)""",
        )

        # important bit: there is 1 mutation in the bw
        self.assertExpectedInline(
            bw_graph[0].code.strip(),
            """\
def forward(self, add, clone, tangents_1):
    cos = torch.ops.aten.cos.default(clone);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, cos);  tangents_1 = cos = None
    _foreach_mul__2 = torch.ops.aten._foreach_mul_.ScalarList([add], [4]);  _foreach_mul__2 = None
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, add);  mul_1 = add = None
    return (mul_2, None)""",
        )

    def test_fw_bw_mutation_no_functionalization2(self):
        # [FUNCTION DOC] Like the previous test with a single-input Function: functionalization
        # disabled; checks outputs and gradient match eager and asserts the forward graph has
        # two mutations (the activation mutation is not moved to backward) and the backward
        # graph has none.
        class FwBwMutation(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function mutating its input and an intermediate in
            # forward and the saved tensor in backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Mutate the input and an intermediate in place, save the
                # intermediate and return it.
                # input mutation
                torch._foreach_mul_([x], [2])
                x = x + 1
                # intermediate mutation
                torch._foreach_mul_([x], [3])
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Mutate the saved tensor in place and return the
                # gradient times it.
                (x,) = ctx.saved_tensors
                # bw mutation
                torch._foreach_mul_([x], [4])
                return grad_output * x

        def f(a, b):
            # [FUNCTION DOC] HELPER: Apply the Function to b and multiply with a.
            out = FwBwMutation.apply(b)
            return out * a

        inps = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]
        inps_ref = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        fw_graph = [None]
        bw_graph = [None]

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler that stores the graph in a list cell and
            # returns it unchanged.
            fw_graph[0] = gm
            return gm

        def bw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backward compiler that stores the graph in a list cell and
            # returns it unchanged.
            bw_graph[0] = gm
            return gm

        compiled_f = compiled_function(
            f,
            fw_compiler,
            bw_compiler,
            dynamic=False,
            partition_fn=default_partition,
            keep_inference_input_mutations=True,
            disable_functionalization=True,
        )

        out_ref = f(*inps_ref)
        out = compiled_f(*inps)
        self.assertEqual(out, out_ref)

        out_ref.sum().backward()
        out.sum().backward()
        self.assertEqual(inps_ref[0].grad, inps[0].grad)

        # important bit: there are 2 mutations in the fw
        # (the mutation on an activation doesn't get moved to bw)
        self.assertExpectedInline(
            fw_graph[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    _foreach_mul_ = torch.ops.aten._foreach_mul_.ScalarList([primals_2], [2]);  _foreach_mul_ = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    _foreach_mul__1 = torch.ops.aten._foreach_mul_.ScalarList([add], [3]);  _foreach_mul__1 = None
    mul = torch.ops.aten.mul.Tensor(add, primals_1);  primals_1 = None
    return (mul, add)""",
        )

        self.assertExpectedInline(
            bw_graph[0].code.strip(),
            """\
def forward(self, add, tangents_1):
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, add);  tangents_1 = add = None
    return (mul_1, None)""",
        )

    # def test_backward_mutation_metadata(self):
    #     class BwMutation(torch.autograd.Function):
    #         @staticmethod
    #         def forward(ctx, a, b):
    #             ctx.save_for_backward(b)
    #             return a.clone(), b.clone()

    #         @staticmethod
    #         def backward(ctx, grad_a, grad_b):
    #             (b,) = ctx.saved_tensors
    #             # bw metadata mutation
    #             b.transpose_(1, 0)
    #             return grad_a.clone(), grad_b.clone()

    #     def f(a, b):
    #         a_, b_ = BwMutation.apply(a, b)
    #         out = a_ * b_
    #         return out

    #     inp_no_grad = [
    #         torch.ones(3, 3, requires_grad=True),
    #         torch.ones(3, 3, requires_grad=False),
    #     ]

    #     with self.assertRaisesRegex(
    #         AssertionError, "input that had its metadata mutated in the backward"
    #     ):
    #         self.verify_aot_autograd(f, inp_no_grad, test_mutation=True)

    def test_backward_mutation_on_grad_out(self):
        # [FUNCTION DOC] Check a Function whose backward mutates the incoming gradient; compiled
        # gradients for both inputs must equal eager.
        class BwMutation(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function cloning in forward and doubling the gradient
            # in place before returning a clone in backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone of the input.
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Double the gradient in place and return a clone.
                grad_output.mul_(2)
                return grad_output.clone()

        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiply inputs then apply the mutating Function.
            tmp = a * b
            out = BwMutation.apply(tmp)
            return out

        inp_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        inp_grad_ref = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]

        f_compiled = aot_function(f, nop)
        out = f_compiled(*inp_grad)
        out.mul(2).sum().backward()
        out_ref = f(*inp_grad_ref)
        out_ref.mul(2).sum().backward()
        self.assertEqual(inp_grad[0].grad, inp_grad_ref[0].grad)
        self.assertEqual(inp_grad[1].grad, inp_grad_ref[1].grad)

    def test_backward_mutation_forward_inputs(self):
        # [FUNCTION DOC] Register a custom op whose backward zeroes a forward input and whose
        # forward function also mutates another input. Checks forward mutations are applied in
        # the compiled run, backward mutation is applied after backward, and outputs, inputs and
        # grads match eager.
        @torch.library.custom_op("_test::_clone", mutates_args={})
        def f(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation returning a clone of the
            # first argument.
            return x.clone()

        def f_fake(x, x1):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like
            # the input.
            return torch.empty_like(x)

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward rule: zero the stored second input under no_grad
            # and return the doubled gradient and no gradient for the second input.
            with torch.no_grad():
                ctx.x1.zero_()
            return grad * 2, None

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Store both inputs on the context.
            (x, x1) = inputs
            ctx.x = x
            ctx.x1 = x1

        f.register_fake(f_fake)
        f.register_autograd(backward, setup_context=setup_context)

        def fn(x: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Multiply x2 by five in place and return the custom op
            # output plus x2.
            x2.mul_(5)
            return torch.ops._test._clone(x, x1) + x2

        inp_x, inp_x1, inp_x2 = (
            torch.randn(3, requires_grad=True),
            torch.randn(3, requires_grad=False),
            torch.randn(3, requires_grad=False),
        )

        ref_x, ref_x1, ref_x2 = inp_x.clone(), inp_x1.clone(), inp_x2.clone()
        ref_y = fn(ref_x, ref_x1, ref_x2)

        compiled_f = aot_function(fn, nop, keep_inference_input_mutations=True)

        x, x1, x2 = inp_x.clone(), inp_x1.clone(), inp_x2.clone()
        y = compiled_f(x, x1, x2)

        # Verify mutation in forward applied and mutation in backward is not in forward
        self.assertEqual(ref_x, x)
        self.assertEqual(ref_x1, x1)
        self.assertEqual(ref_x2, x2)
        self.assertEqual(ref_y, y)

        ref_y.sum().backward()
        y.sum().backward()

        # Verify mutations in backward applied
        self.assertEqual(ref_x, x)
        self.assertEqual(ref_x1, x1)
        self.assertEqual(ref_x2, x2)
        self.assertEqual(ref_y, y)

        self.assertEqual(ref_x.grad, x.grad)
        self.assertEqual(ref_x1.grad, x1.grad)
        self.assertEqual(ref_x2.grad, x2.grad)

    def test_backward_mutation_forward_inputs_create_graph(self):
        # [FUNCTION DOC] Like the previous test but with a custom op registered under a
        # different name, and double backward requested: computing gradients with create_graph
        # on the compiled function must raise RuntimeError that input mutations with
        # requires_grad in backward are unsupported.
        @torch.library.custom_op("_test::_clone_create_graph", mutates_args={})
        def f(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation returning a clone of the
            # first argument.
            return x.clone()

        def f_fake(x, x1):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like
            # the input.
            return torch.empty_like(x)

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward rule: zero the stored second input under no_grad
            # and return the doubled gradient and no gradient for the second input.
            with torch.no_grad():
                ctx.x1.zero_()
            return grad * 2, None

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Store both inputs on the context.
            (x, x1) = inputs
            ctx.x = x
            ctx.x1 = x1

        f.register_fake(f_fake)
        f.register_autograd(backward, setup_context=setup_context)

        def fn(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Call the custom op on the two inputs.
            return torch.ops._test._clone_create_graph(x, x1)

        inp_x, inp_x1 = (
            torch.randn(3, requires_grad=True),
            torch.randn(3, requires_grad=True),
        )

        ref_x, ref_x1 = inp_x.clone(), inp_x1.clone()
        ref_y = f(ref_x, ref_x1)
        ref_y.sum().backward()
        x, x1 = inp_x.clone(), inp_x1.clone()
        compiled_f = aot_function(fn, nop)
        y = compiled_f(x, x1)
        loss = y.sum()
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_autograd does not support input mutations with requires_grad in backward for create_graph=True",
        ):
            torch.autograd.grad(loss, inp_x, create_graph=True)
        # Not checking equality of ref and x as Exception is expected

    # Partially addresses https://github.com/pytorch/pytorch/issues/106457
    def test_input_mutation_false_aliasing(self):
        # [FUNCTION DOC] Partially addresses issue 106457. Check that two views of one storage
        # that do not actually overlap are treated as separate graph inputs (false aliasing
        # detected) under many layouts: contiguous halves, interleaved columns and several
        # strided layouts, and verify overlapping layouts take the synthetic base path.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Triple a and double b in place and return the sum of their
            # flattened clones.
            a.mul_(3)
            b.mul_(2)
            return a.clone().view(-1) + b.clone().view(-1)

        # No overlap, contiguous
        def inp_callable1(req_grad):
            # [FUNCTION DOC] HELPER: Make two non-overlapping contiguous row-halves of a
            # non-leaf and return the base and the two views.
            base = torch.ones(4, 4, requires_grad=req_grad)
            x = base.add(1)
            # create two views that share storage, but are actually non-overlapping
            a = x[0:2]
            b = x[2:4]
            return [base], [a, b]

        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable1, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable1, req_grad=True), test_mutation=True
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable1, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable1, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

        # Important characteristic: the graph takes in 2 inputs!
        # That shows that we didn't try to run our complicated synthetic base logic,
        # because we successfully detected false aliasing across the two inputs.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 3);  arg0_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    clone = torch.ops.aten.clone.default(mul)
    view = torch.ops.aten.view.default(clone, [-1]);  clone = None
    clone_1 = torch.ops.aten.clone.default(mul_1)
    view_1 = torch.ops.aten.view.default(clone_1, [-1]);  clone_1 = None
    add = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
    return (mul, mul_1, add)""",
        )

        # No overlap, non-contiguous: first tensor ends before second tensor start
        def inp_callable2(req_grad):
            # [FUNCTION DOC] HELPER: Make two strided views where the first ends before the
            # second starts; returns the base and the views.
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (8, 1), storage_offset=0)
            b = x.as_strided((4, 4), (8, 1), storage_offset=28)
            return [base], [a, b]

        # No overlap, non-contiguous: tensors are perfectly interleaved
        def inp_callable3(req_grad):
            # [FUNCTION DOC] HELPER: Make two interleaved column-slice views; returns the base
            # and the views.
            base = torch.ones(4, 4, requires_grad=req_grad)
            x = base.add(1)
            a = x[:, 0:2]
            b = x[:, 2:4]
            return [base], [a, b]

        # No overlap, non-contiguous
        def inp_callable4(req_grad):
            # [FUNCTION DOC] HELPER: Make two strided non-overlapping views at offsets 0 and 22;
            # returns the base and the views.
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=22)
            return [base], [a, b]

        # No overlap, non-contiguous
        def inp_callable5(req_grad):
            # [FUNCTION DOC] HELPER: Make two strided non-overlapping views at offsets 0 and 23;
            # returns the base and the views.
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=23)
            return [base], [a, b]

        # No overlap, non-contiguous
        def inp_callable6(req_grad):
            # [FUNCTION DOC] HELPER: Make a three-dimensional strided view ending at offset 195
            # and a slice starting at 196 of equal length; returns the base and the views.
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            # a's last element is at offset 195 (24 total elements)
            a = x.as_strided((2, 4, 3), (110, 24, 4), storage_offset=5)
            # b's first element is at offset 196: no overlap
            b = x[196 : 196 + a.numel()]
            return [base], [a, b]

        # overlap! non-contiguous
        def inp_callable_overlap1(req_grad):
            # [FUNCTION DOC] HELPER: Make two strided views at offsets 0 and 24 that overlap;
            # returns the base and the views.
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=24)
            return [base], [a, b]

        # overlap! non-contiguous
        def inp_callable_overlap2(req_grad):
            # [FUNCTION DOC] HELPER: Make two strided views at offsets 0 and 25 that overlap;
            # returns the base and the views.
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=25)
            return [base], [a, b]

        # overlap! non-contiguous
        def inp_callable_overlap3(req_grad):
            # [FUNCTION DOC] HELPER: Make a three-dimensional strided view ending at offset 195
            # and a slice starting at 195 of equal length so the two overlap; returns the base
            # and the views.
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            # a's last element is at offset 195 (24 total elements)
            a = x.as_strided((2, 4, 3), (110, 24, 4), storage_offset=5)
            # b's first element is at offset 195: overlap!
            b = x[195 : 195 + a.numel()]
            return [base], [a, b]

        fw_graph2 = self.verify_aot_autograd(
            f, partial(inp_callable2, req_grad=False), test_mutation=True
        )
        fw_graph3 = self.verify_aot_autograd(
            f, partial(inp_callable3, req_grad=False), test_mutation=True
        )
        fw_graph4 = self.verify_aot_autograd(
            f, partial(inp_callable4, req_grad=False), test_mutation=True
        )
        fw_graph5 = self.verify_aot_autograd(
            f, partial(inp_callable5, req_grad=False), test_mutation=True
        )
        fw_graph6 = self.verify_aot_autograd(
            f, partial(inp_callable6, req_grad=False), test_mutation=True
        )

        fw_graph_overlap1 = self.verify_aot_autograd(
            f, partial(inp_callable_overlap2, req_grad=False), test_mutation=True
        )
        fw_graph_overlap2 = self.verify_aot_autograd(
            f, partial(inp_callable_overlap1, req_grad=False), test_mutation=True
        )

        # All non-overlap graphs should be the same since we detected false aliasing
        self.assertEqual(str(fw_graph.code), str(fw_graph2.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph3.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph4.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph5.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph6.code))

        # All overlap graphs should be the same since we detected real aliasing
        self.assertNotEqual(str(fw_graph.code), str(fw_graph_overlap1.code))
        self.assertNotEqual(str(fw_graph.code), str(fw_graph_overlap2.code))
        self.assertTrue("as_strided_scatter" in str(fw_graph_overlap1.code))
        self.assertTrue("as_strided_scatter" in str(fw_graph_overlap2.code))

    def _check_merge_view_inputs_error(self, fn, make_inputs, expected_error):
        # [FUNCTION DOC] HELPER: Reset dynamo, compile the function with aot_eager on freshly
        # made inputs and expect compilation to fail; fails the test if no error, otherwise
        # compares the inner exception message with the expected inline text.
        """Compile fn with aot_eager, assert the error message from merge_view_inputs."""
        torch._dynamo.reset()
        try:
            torch.compile(fn, backend="aot_eager")(*make_inputs())
            self.fail("Expected BackendCompilerFailed")
        except torch._dynamo.exc.BackendCompilerFailed as e:
            self.assertExpectedInline(str(e.inner_exception), expected_error)

    def test_merge_view_inputs_error_non_differentiable_views(self):
        # [FUNCTION DOC] Training mode: two inputs share storage, one a differentiable view and
        # one a set_ alias with no base, and the first is mutated; expects the error saying non
        # differentiable view input mutations are unsupported because the inputs are not
        # differentiable views of each other.
        # Training mode (requires_grad) reaches site-1 in merge_view_inputs.
        def make_inputs():
            # [FUNCTION DOC] HELPER: Build a large grad-requiring tensor, a view of it, and an
            # empty tensor set to its storage; return them with the set-alias first.
            big = torch.randn(10, requires_grad=True)
            a = big[0:5]  # _base = big
            b = torch.empty(5)
            b.set_(big.untyped_storage(), 0, (5,), (1,))  # _base=None
            return [b, a]

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Double a in place and return a plus b.
            a.mul_(2)
            return a + b

        self._check_merge_view_inputs_error(
            fn,
            make_inputs,
            """aot_autograd() does not yet handle non-differentiable view input mutations. input 0 (a) and input 1 (b) share storage but are not differentiable views of each other.""",
        )

    def test_merge_view_inputs_error_different_bases(self):
        # [FUNCTION DOC] Inference mode: two views whose bases are different tensor objects over
        # the same storage with a mutated input; expects the error about aliased inputs having
        # different base tensors.
        # Inference mode (no requires_grad) skips site-1, reaching site-2.
        def make_inputs():
            # [FUNCTION DOC] HELPER: Make two tensors sharing storage via set_ and return a
            # half-size view of each.
            x = torch.randn(10)
            y = torch.randn(10)
            y.set_(x.untyped_storage(), 0, (10,), (1,))
            v1 = x[0:5]  # _base = x
            v2 = y[0:5]  # _base = y (different object, same storage)
            return [v1, v2]

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Double a in place and return a plus b.
            a.mul_(2)
            return a + b

        self._check_merge_view_inputs_error(
            fn,
            make_inputs,
            """aot_autograd() does not yet handle non-differentiable view input mutations. Aliased inputs share storage but have different autograd ._base tensors: input 0 (a) and input 1 (b) have ._base fields that point to different tensors.""",
        )

    def test_merge_view_inputs_error_mixed_base_states(self):
        # [FUNCTION DOC] Inference mode: aliased inputs where one has a base and the other has
        # none; expects the error about mixed autograd base states, naming both inputs.
        # Inference mode (no requires_grad) skips site-1, reaching site-3.
        def make_inputs():
            # [FUNCTION DOC] HELPER: Make a view with a base and an empty tensor set to the same
            # storage (no base); returns both.
            x = torch.randn(10)
            v1 = x[0:5]  # _base = x
            v2 = torch.empty(5)
            v2.set_(x.untyped_storage(), 0, (5,), (1,))  # _base=None
            return [v1, v2]

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Double a in place and return a plus b.
            a.mul_(2)
            return a + b

        self._check_merge_view_inputs_error(
            fn,
            make_inputs,
            """aot_autograd() does not yet handle non-differentiable view input mutations. Aliased inputs share storage but have mixed autograd ._base states: ['input 0 (a)'] have ._base set, while ['input 1 (b)'] have ._base=None (and are not the synthetic base).""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_mem_leak_from_save_for_bw(self):
        # [FUNCTION DOC] CUDA only. Regression for issue 94990: tensors saved for backward are
        # detached to avoid a reference cycle; after running the compiled forward the allocated
        # CUDA memory must equal the amount before.
        # See a full diagnosis at this issue: https://github.com/pytorch/pytorch/issues/94990
        # Note [Detaching saved tensors in AOTAutograd]
        # This program creates a ref-cycle. Long term, we should fix this ref cycle
        # (since it can arise, naturally albeit rarely, from uses of autograd.Function).
        # But AOTAutograd makes it more likely to show up from tracing user programs,
        # so we deal with it by manually detaching the tensors that we save for backward.
        # This is completely wrong and would give wrong results if we were to do double backward.
        # Fortunately today, double backward is explicitly banned in AOTAutograd.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add input to itself, split into two column halves,
            # unsqueeze the second and multiply by b; returns the second half and the product.
            add = a + a
            split = torch.functional.split(add, [4, 4], dim=1)
            getitem_2 = split[1]
            unsqueeze = getitem_2.unsqueeze(-1)
            mul = unsqueeze * b
            return (getitem_2, mul)

        f_compiled = aot_function(f, nop)
        inps = [
            torch.ones(8, 8, device="cuda", requires_grad=True),
            torch.ones(1, 4, 1, device="cuda", requires_grad=True),
        ]
        mem_before = torch.cuda.memory_allocated()
        f_compiled(*inps)
        mem_after = torch.cuda.memory_allocated()
        self.assertTrue(mem_after == mem_before)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_save_input_view_for_bw_does_not_leak_memory(self):
        # [FUNCTION DOC] CUDA only. Checks that saving a view input for backward does not leak:
        # records saved views and bases via hooks, verifies the saved view and base are the
        # original input objects (not the output or its grad_fn), that memory is held while the
        # output lives, and fully freed (all weak references dead) after deleting the output.
        def f(x):
            # [FUNCTION DOC] HELPER: Sum of squares of the input.
            return (x * x).sum()

        f_compiled = aot_function(f, nop)

        def run_once(check_saved_view):
            # [FUNCTION DOC] HELPER: Create a large grad-requiring base, a non-leaf scaled copy
            # and a view of its first half, with weak references; run the compiled function on
            # the view, optionally under saved-tensor hooks that record weak references to
            # saved views and their bases. Returns the output, tensors, weak references and
            # recorded lists.
            base = torch.randn(1024, 1024, device="cuda", requires_grad=True)
            non_leaf_base = base * 2
            input_view = non_leaf_base[:512]
            input_view_ref = weakref.ref(input_view)
            non_leaf_base_ref = weakref.ref(non_leaf_base)
            saved_view_refs = []
            saved_base_refs = []

            def pack_hook(t):
                # [FUNCTION DOC] HELPER: Saved tensor pack hook that records weak references to
                # a saved tensor and its base if it is a view, and returns the tensor unchanged.
                if t._is_view():
                    saved_view_refs.append(weakref.ref(t))
                    saved_base_refs.append(weakref.ref(t._base))
                return t

            if check_saved_view:
                with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
                    out = f_compiled(input_view)
            else:
                out = f_compiled(input_view)

            return (
                out,
                input_view,
                non_leaf_base,
                base,
                input_view_ref,
                non_leaf_base_ref,
                saved_view_refs,
                saved_base_refs,
            )

        warmup = run_once(check_saved_view=False)
        del warmup
        gc.collect()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        (
            out,
            input_view,
            non_leaf_base,
            base,
            input_view_ref,
            non_leaf_base_ref,
            saved_view_refs,
            saved_base_refs,
        ) = run_once(check_saved_view=True)
        out_ref = weakref.ref(out)
        self.assertGreater(torch.cuda.memory_allocated(), mem_before)

        self.assertEqual(len(saved_view_refs), 1)
        saved_view = saved_view_refs[0]()
        saved_base = saved_base_refs[0]()
        self.assertIs(saved_view, input_view)
        self.assertIs(saved_base, non_leaf_base)
        self.assertIsNot(saved_base, out)
        self.assertIsNot(saved_base.grad_fn, out.grad_fn)

        saved_view = saved_base = None
        del input_view, non_leaf_base, base
        gc.collect()
        self.assertIsNotNone(out_ref())
        self.assertIsNotNone(input_view_ref())
        self.assertIsNotNone(non_leaf_base_ref())

        del out
        gc.collect()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        self.assertEqual(mem_after, mem_before)
        self.assertIsNone(out_ref())
        self.assertIsNone(input_view_ref())
        self.assertIsNone(non_leaf_base_ref())

    def test_output_aliases_multiple_inputs_get_correct_one(self):
        # [FUNCTION DOC] Check outputs that are views of two aliased inputs with different
        # shapes each regenerate from the correct input; verified with and without subclass
        # inputs and both requires_grad settings.
        # a and b are aliased, but have different shapes
        # The first output should view off the first input, the 2nd output should view off the 2nd input
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return same-shape views of both inputs.
            return a.view(a.shape), b.view(b.shape)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a base and two aliased non-leaf views (flattened and
            # first row); return the base and the views.
            base = torch.ones(2, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.mul(2)
            inp1 = x.view(-1)
            inp2 = x[0]
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

    def test_input_mutation_aliases_other_input(self):
        # [FUNCTION DOC] Check mutation of one of two aliased inputs: the graph takes a
        # synthetic base, clones it, and regenerates views after the mutation; asserts the
        # graph text.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add one to a in place and return a plus b.
            a.add_(1)
            return a + b

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a base and two identical row views of a non-leaf;
            # return the base and the views.
            base = torch.ones(4, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[0]
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # Important parts of the graph:
        # - the compiled graph takes in a base, and we generate a and b (the views) off of the base
        # - clone() is still in the graph, because we need to call grad() on the original (non-mutated) inputs
        # - We re-generate the views *after* the clone, to preserve view relationships.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = add = None
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)
    add_1 = torch.ops.aten.add.Tensor(as_strided_2, as_strided_5);  as_strided_2 = as_strided_5 = None
    return (as_strided_scatter, add_1)""",
        )

    def test_input_mutation_aliases_other_input2(self):
        # [FUNCTION DOC] Same as the previous test but one aliased input is the whole base
        # itself; asserts the graph text.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add one to a in place and return a plus b.
            a.add_(1)
            return a + b

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a base, a row view and the whole non-leaf; return the
            # base and the pair.
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            inp1 = x[0]
            # Here, one of the aliased inputs is the base itself
            inp2 = x
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = add = None
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [2, 2], [2, 1], 0)
    add_1 = torch.ops.aten.add.Tensor(as_strided_2, as_strided_5);  as_strided_2 = as_strided_5 = None
    return (as_strided_scatter, add_1)""",
        )

    def test_input_mutation_aliases_and_output_alias(self):
        # [FUNCTION DOC] Check output that is a view of an aliased input that was updated by a
        # mutation through the other alias; the view is generated from the updated base. Asserts
        # the graph text.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add one to a in place and return a same-shape view of b.
            # Here, we need to take care:that because and b are aliased
            # since a and b are aliased, we generate a view off of "updated b"
            a.add_(1)
            return b.view(b.shape)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a base and two flattened views of a non-leaf; return
            # the base and the views.
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return [base], [x.view(-1), x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None
    as_strided_9 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    view_1 = torch.ops.aten.view.default(as_strided_9, [4]);  as_strided_9 = None
    return (as_strided_scatter, view_1)""",
        )

    def test_input_aliased_with_mutation_output_alias(self):
        # [FUNCTION DOC] Check two aliased inputs with a mutation through one while a separate
        # third input stays unaliased and an output views the mutated alias; asserts the graph
        # (two compiled inputs instead of three).
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Double c in place and return b plus one and a flattened
            # view of c.
            # a and c alias
            c.mul_(2)
            # The main thing we're testing here is that
            # (1) We need to reconstruct c.view(-1) from the 3rd input to the forward
            # (2) But we need to be careful to do this *before* converting aliased inputs into synthetic bases.
            #     The original fw takes in 3 args, but the compiled fw takes in only 2 args.
            return b.add(1), c.view(-1)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make two bases, derive non-leaf tensors, return the bases
            # and the arguments (two aliased flat views and one separate tensor).
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            x = base1.add(1)
            y = base2.add(1)
            return [base1, base2], [x.view(-1), y, x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    as_strided_7 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    view_1 = torch.ops.aten.view.default(as_strided_7, [-1]);  as_strided_7 = None
    return (as_strided_scatter, add, view_1)""",
        )

    def test_input_aliased_with_mutation_nontensor_interleaved(self):
        # [FUNCTION DOC] Check aliased tensor inputs with a non-tensor scalar argument
        # interleaved between them.
        def f(a, scale, b):
            # [FUNCTION DOC] HELPER: Double b in place and return a plus the scalar.
            b.mul_(2)
            return a + scale

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a base and arguments consisting of a flat view, the
            # integer three and another flat view.
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return [base], [x.view(-1), 3, x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )

    def test_input_aliased_with_mutation_multiple_nontensors(self):
        # [FUNCTION DOC] Check aliased tensor inputs with several non-tensor arguments
        # interleaved with a separate tensor.
        def f(a, scale, b, offset, c):
            # [FUNCTION DOC] HELPER: Double c in place and return a plus scale plus offset.
            c.mul_(2)
            return a + scale + offset

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make two bases and arguments of aliased flat views, scalars
            # and a separate tensor.
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            standalone = torch.ones(4, requires_grad=req_grad)
            y = standalone.add(1)
            return [base, standalone], [x.view(-1), 3, y, 5, x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )

    def test_input_metadata_mutation_aliases(self):
        # [FUNCTION DOC] Check two aliased inputs where one gets only a metadata mutation
        # (transpose): no synthetic base is built and the graph keeps two inputs. Asserts
        # graph text.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Transpose a in place and return a plus b.
            # a and b alias, and we do a metadata mutation on a
            # Since we're not mutating data, then b isn't affected at all.
            # We expect aot autograd to not bother with constructing a synthetic base.
            a.t_()
            return a + b

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a base and two flat views of a non-leaf.
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return [base], [x.view(-1), x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # Expectation: fwd() takes in 2 args, and we don't construct a synthetic base.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    t = torch.ops.aten.t.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(t, primals_2);  t = primals_2 = None
    return (add,)""",
        )

    def test_input_mutation_aliases_and_none_require_gradients(self):
        # [FUNCTION DOC] Check aliased inputs that do not require grad (no base attribute) with
        # mutation: plain inputs work; with subclass inputs expects RuntimeError that subclasses
        # are unsupported; with requires_grad on the third input asserts the graph text with
        # as_strided scatter.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Double a in place and return b plus one and c plus one.
            # a and b alias, but neither require gradients (so they don't have a _base)
            # aot autograd should construct the synthetic base from `torch.Tensor(a.storage())`
            a.mul_(2)
            return b + 1, c + 1

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a non-grad base, a separate input with the requested
            # requires_grad, and two flat views of the derived tensor plus the separate input.
            base = torch.ones(2, 2)
            c_arg = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return [base, c_arg], [x.view(-1), x.view(-1), c_arg]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )

        with self.assertRaisesRegex(
            RuntimeError, "is a tensor subclass. This is not supported today"
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    as_strided = torch.ops.aten.as_strided.default(primals_1, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided, 2);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(primals_1, mul, [4], [1], 0);  primals_1 = mul = None
    as_strided_3 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided_3, 1);  as_strided_3 = None
    add_1 = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    return (as_strided_scatter, add, add_1)""",
        )

    @skipIfDynamoInput("Fails with dynamo")
    def test_input_mutation_aliases_bases_out_of_order(self):
        # [FUNCTION DOC] Skipped under the dynamo variant. Two pairs of aliased inputs where
        # only one pair is mutated (data mutation plus metadata mutation); plain inputs work,
        # subclass inputs raise RuntimeError about aliased mutated inputs, and the grad case
        # asserts the three-input graph text.
        # This tests our calling convention: if b and d are aliased, then the outer calling convention
        # that we send to the compiled forward becomes:
        # (b_d_base, a, c)
        # Importantly, even though a and c alias in our test, neither inputs are mutated,
        # So we don't need to do the base construction / deconstruction
        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Add one to b in place, unsqueeze d in place, return the sum
            # of a, c, d and a flattened view of b.
            b.add_(1)
            d.unsqueeze_(0)
            return a + c + d, b.view(-1)

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make two bases and four flat views where a and c alias one
            # base and b and d alias the other.
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            x1 = base1.add(1)
            x2 = base2.add(1)
            # a and c alias, b and d alias
            return [base1, base2], [x1.view(-1), x2.view(-1), x1.view(-1), x2.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # 3 graph inputs: (b_d_base, a, c)
        # 2 returns: (b_updated, a+c+d)
        # (there are 2 original fw outs, but one is a view of b so it's not part of the graph)
        # (there are also 2 input mutations, but one is a metadata-only mutation so the compiled forward doesn't return it)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    unsqueeze = torch.ops.aten.unsqueeze.default(as_strided_5, 0);  as_strided_5 = None
    add_1 = torch.ops.aten.add.Tensor(primals_2, primals_3);  primals_2 = primals_3 = None
    add_2 = torch.ops.aten.add.Tensor(add_1, unsqueeze);  add_1 = None
    as_strided_14 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    view_2 = torch.ops.aten.view.default(as_strided_14, [-1]);  as_strided_14 = None
    return (as_strided_scatter, add_2, view_2, unsqueeze)""",
        )

    def test_input_mutation_unsqueeze_view_of_base_input(self):
        # [FUNCTION DOC] Regression test: one input is the base itself and another is an
        # unsqueezed view of it; returning the base must be treated as an alias of an input in
        # the synthetic base calling convention.
        # Regression test: when one input is the base tensor (._base is None)
        # and another is an unsqueeze view of it, returning the base as output
        # must be treated as alias_of_input (not is_input) in the synthetic
        # base calling convention.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add one to b in place and return a.
            b.add_(1)
            return a

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make a base and arguments consisting of its non-leaf
            # derived tensor and an unsqueezed view of it.
            base = torch.ones(4, requires_grad=req_grad)
            x = base.add(1)
            return [base], [x, x.unsqueeze(0)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_synthetic_base_base_attribute_is_none(self):
        # [FUNCTION DOC] CUDA only. Aliased CUDA inputs (detached slices so no input has a
        # base attribute) with a mutation, to check the synthetic base logic handles missing
        # base attributes.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Add one to a in place and return a plus b.
            a.add_(1)
            return a + b

        def inp_callable():
            # [FUNCTION DOC] HELPER: Make a CUDA base and two detached row slices; returns the
            # base and the slices (also creates an unused grad tensor).
            base = torch.ones(4, 4, device="cuda")
            # detach() so that none of the inputs have a ._base attribute.
            a = base[0].detach()
            b = base[1].detach()
            base2 = torch.ones(2, 2, requires_grad=True)  # noqa: F841
            return [base], [a, b]

        self.verify_aot_autograd(f, inp_callable, test_mutation=True)

    def test_input_mutation_alias_everything(self):
        # [FUNCTION DOC] Large combined test: an aliased and mutated input pair, a metadata
        # mutation on another input, and outputs aliasing an intermediate and each other;
        # asserts the graph text with synthetic base and a single real output.
        # Mondo test that tests a combination of:
        # input is mutated, that aliases another input (so we make a synthetic base)
        # an output is an alias of another output
        # an output is an alias of an intermediate
        # a and c are aliased
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Double c in place, transpose b in place, form a sum of a
            # and c, return a flattened view, a transposed b and an unsqueezed view.
            c.mul_(2)  # mutates c
            b.t_()  # metadata mutate b
            tmp = a + c
            out1 = tmp.view(-1)
            out2 = b.t()
            out3 = out1.unsqueeze(0)
            # out1 and out3 are aliases of an intermediate, and alias each other!
            # out2 aliases an input, so we don't return it
            return out1, out2, out3

        def inp_callable(req_grad):
            # [FUNCTION DOC] HELPER: Make two bases and arguments where a and c are flat views
            # of one non-leaf and b is the other non-leaf.
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            base1_ = base1.add(1)
            base2_ = base2.add(1)
            a = base1_.view(-1)
            b = base2_
            c = base1_.view(-1)
            return [base1, base2], [a, b, c]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # Expected:
        # - 2 inputs in the forward: synthetic_base_a_c, b
        # - 1 output in the forward: "tmp"
        #   out2 is an alias of an input, and will be generated off of b outside of the compiled fn
        #   out1 and out3 are aliases of tmp, that we generate outside of the compiled function
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    t = torch.ops.aten.t.default(view);  view = None
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided_5, as_strided_2);  as_strided_5 = as_strided_2 = None
    view_1 = torch.ops.aten.view.default(add, [-1])
    t_1 = torch.ops.aten.t.default(t)
    unsqueeze = torch.ops.aten.unsqueeze.default(view_1, 0)
    return (as_strided_scatter, t, view_1, t_1, unsqueeze, add)""",
        )

    def test_dynamic_shape_output_not_in_bw_graph(self):
        # [FUNCTION DOC] Check that when a compiled function returns a symbolic size as an
        # output (dynamic shapes), the backward graph takes only one gradient input; asserts the
        # graph text.
        def f(x):
            # [FUNCTION DOC] HELPER: Return a list of the input plus one and its size.
            return [x + 1, x.shape[0]]

        inp = torch.ones(5, requires_grad=True)
        bw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            decompositions={},
            keep_inference_input_mutations=False,
            dynamic=True,
        )
        out = compiled_f(inp)
        out[0].sum().backward()
        # The important bit: the forward fn returns 2 outputs,
        # but one of them is a symint so we should only see
        # 1 grad_output as an input to the backward graph.
        # (Otherwise, autograd will plumb a None as the value of the grad_output,
        # which causes inductor to complain).
        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, tangents_1):
    return (tangents_1,)""",
        )

    def test_no_grad_input_output(self):
        # [FUNCTION DOC] Check functions with outputs from two inputs across all
        # requires_grad combinations.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return cosines of both inputs and their product.
            return a.cos(), b.cos(), a * b

        inp_thunks = [
            lambda: torch.randn(5, requires_grad=True),
            lambda: torch.randn(5, requires_grad=False),
        ]
        for inps in itertools.product(inp_thunks, repeat=2):
            inps = [i() for i in inps]
            self.verify_aot_autograd(f, inps)

    def test_some_output_requires_grad_input_doesnt(self):
        # [FUNCTION DOC] Check an output that is a view of an input with requires_grad turned on
        # in place for a non-requiring input.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Flatten a, enable requires_grad on the view and return it.
            a_view = a.view(-1)
            a_view.requires_grad_(True)
            return a_view

        inp = [torch.randn(3, 3), torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

    def test_some_outputs_dont_require_grad_view(self):
        # [FUNCTION DOC] Check returning a detached view of an input together with a
        # differentiable input.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return a detached alias of a and b.
            return a.detach(), b

        inp = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp)

    def test_some_outputs_dont_require_grad_non_view(self):
        # [FUNCTION DOC] Check returning a detached computed value together with a
        # differentiable input.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return the detached result of a plus one and b.
            return a.add(1).detach(), b

        inp = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp)

    def test_inner_grad(self):
        # [FUNCTION DOC] Check a function that internally calls autograd grad on a
        # computed value.
        def foo(x):
            # [FUNCTION DOC] HELPER: Exponentiate and return the gradient of the result w.r.t.
            # the input.
            y = torch.exp(x)
            z = torch.autograd.grad(y, x)
            return z

        inps = [torch.randn((), requires_grad=True)]
        self.verify_aot_autograd(foo, inps)

    def test_grad_context(self):
        # [FUNCTION DOC] Check the compiler only runs for backward when grad mode matters: under
        # disabled grad mode the second compiler (the graph size recorder) never runs; with grad
        # enabled it is not run until backward, then records a graph larger than two nodes.
        def foo(x):
            # [FUNCTION DOC] HELPER: Double the input.
            return x * 2

        inps = [torch.randn((), requires_grad=True)]
        graph_size = None

        def get_graph_size(fx_g, _):
            # [FUNCTION DOC] HELPER: Compiler recording the node count of the graph in an
            # enclosing variable and returning the graph unchanged.
            nonlocal graph_size
            graph_size = len(fx_g.graph.nodes)
            return fx_g

        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(False):
            f(*inps)
        self.assertIsNone(graph_size)

        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(True):
            out = f(*inps)
            self.assertIsNone(graph_size)
            out.sum().backward()
            self.assertTrue(graph_size > 2)

    def test_output_dict(self):
        # [FUNCTION DOC] Check functions returning dicts: same tensor under two keys, a dict
        # mixing input and computation, and a dict input iterated and scaled.
        def f(x):
            # [FUNCTION DOC] HELPER: Return a dict mapping two keys to the same input.
            return {"a": x, "b": x}

        inp = [torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Return a dict with the first input and the sum of
            # both inputs.
            return {"a": x, "b": y + x}

        inp = [torch.randn(3, requires_grad=True), torch.randn(3)]
        self.verify_aot_autograd(f, inp)

        def f(x):
            # [FUNCTION DOC] HELPER: Return a dict with each value of the input dict doubled.
            new_d = {}
            for k in x:
                new_d[k] = x[k] * 2
            return new_d

        a = torch.randn(3, requires_grad=True)
        b = torch.randn(3, requires_grad=True)

        def inp_callable():
            # [FUNCTION DOC] HELPER: Build one dict of the two tensors and return it as both
            # function and graph inputs.
            inps = [{"a": a, "b": b}]
            return inps, inps

        self.verify_aot_autograd(f, inp_callable)

    def test_module(self):
        # [FUNCTION DOC] Check compiled_module on a Linear plus ReLU model with no-op compilers:
        # output and parameter gradients match eager.
        mod = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        compiled_mod = compiled_module(mod, nop, nop)
        inp = torch.randn(32, 32)
        ref_out = mod(inp)
        ref_out.sum().backward()
        ref_grads = sorted([(name, p.grad) for name, p in mod.named_parameters()])
        out = compiled_mod(inp)
        out.sum().backward()
        grads = sorted([(name, p.grad) for name, p in mod.named_parameters()])
        self.assertEqual((out, grads), (ref_out, ref_grads))

    def test_batchnorm(self):
        # [FUNCTION DOC] Check compiled_module on a BatchNorm2d with no-op compilers runs
        # forward and backward without error.
        mod = compiled_module(nn.BatchNorm2d(4), nop, nop)
        x = torch.ones(1, 4, 2, 2)
        mod(x).sum().backward()

    def test_list_codegen(self):
        # [FUNCTION DOC] Check a compiler that returns a boxed-call function (takes a single
        # list of inputs) works for forward and backward of a three-input function.
        def list_nop(f, _):
            # [FUNCTION DOC] HELPER: Compiler returning a wrapper flagged as a boxed call that
            # unpacks the input list into the graph module call.
            def g(inps):
                # [FUNCTION DOC] HELPER: Call the graph module with the unpacked input list.
                return f(*inps)

            g._boxed_call = True
            return g

        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Product of sine of a, cosine of b and sine of c.
            return a.sin() * b.cos() * c.sin()

        f = aot_function(f, list_nop)
        inp = [torch.randn(5, requires_grad=True) for _ in range(3)]
        f(*inp).sum().backward()

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    def test_compilation_context(self, counter):
        # [FUNCTION DOC] Check the AOT compilation context reported to compilers: records the
        # context name and node count for each compile (forward, inference, backward), with the
        # global counter patched to a fresh counter; asserts the recorded list.
        def f(x):
            # [FUNCTION DOC] HELPER: Sine applied twice.
            return x.sin().sin()

        count = []

        def compiler(fx_g, _):
            # [FUNCTION DOC] HELPER: Compiler that appends the current compilation context name
            # and graph node count to a list and returns the graph.
            context = get_aot_compilation_context()
            count.append((context[0], len(fx_g.graph.nodes)))
            return fx_g

        f = aot_function(f, compiler)
        out = f(torch.randn(5, requires_grad=True))
        f = aot_function(f, compiler)
        f(torch.randn(5))
        out.sum().backward()
        self.assertExpectedInline(
            str(count),
            """[(['0_forward'], 4), (['1_inference'], 4), (['0_backward'], 8)]""",
        )

    def test_dupe_arg(self):
        # [FUNCTION DOC] Check calling with the same tensor passed for two arguments (duplicate
        # arguments) matches eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Add the two inputs.
            return x + y

        x = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(f, [x, x])

    def test_dupe_arg_torture(self):
        # [FUNCTION DOC] Check duplicate arguments where the function applies a metadata
        # mutation to each (transpose and unsqueeze in place) and adds them.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Transpose x in place, unsqueeze y in place, return
            # their sum.
            x.t_()
            y.unsqueeze_(0)
            return x + y

        x = torch.randn(3, 3, requires_grad=True).clone()
        self.verify_aot_autograd(f, [x, x])

    # See https://github.com/pytorch/pytorch/issues/100224
    def test_dupe_arg_returned_as_output(self):
        # [FUNCTION DOC] Regression test for issue 100224: a duplicated argument that is
        # mutated through one name and returned through the other; result and mutated input
        # must match eager.
        def f(a, b, a_):
            # [FUNCTION DOC] HELPER: Add one to the first element of a in place and return the
            # third argument.
            a[0].add_(1)
            return a_

        f_compiled = aot_function(f, nop)
        a = torch.ones(2)
        b = torch.ones(2)
        out_ref = f(a, b, a)

        a2 = torch.ones(2)
        b2 = torch.ones(2)
        out_test = f_compiled(a2, b2, a2)

        self.assertEqual(out_ref, out_test)
        self.assertEqual(a, a2)

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe_left_bias(self, counter):
        # [FUNCTION DOC] Patched counter and debug assert. Checks that when only the first of
        # two duplicate arguments gets a metadata mutation, the deduplicating strategy is
        # chosen; calling the module compiled for duplicates with distinct tensors raises
        # AssertionError saying the duplicate assumption was violated at runtime.
        # This test checks that, just because only the first
        # argument did a metadata mutation, we still correctly
        # switch to strategy 2 (deduplicate)
        # See: https://github.com/pytorch/pytorch/pull/89896#discussion_r1036224447
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward transposes x in place and returns a tuple
            # of the sum.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Transpose x in place and return a one-element tuple
                # with x plus y.
                x.t_()
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(F(), [x, x])

        fxx = aot_module_simplified(F(), (x, x), nop)
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxx(x, y),
            """At compilation time, graph 2 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",
        )

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe(self, counter):
        # [FUNCTION DOC] Run the invalid-duplicate check with real tensors.
        self._test_invalid_dupe(counter, fake=False)

    # See Note: Dynamo recompilation guarding invalid grad for why this test exists
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe_fake(self, counter):
        # [FUNCTION DOC] Run the invalid-duplicate check with fake tensors.
        self._test_invalid_dupe(counter, fake=True)

    def _test_invalid_dupe(self, counter, fake):
        # [FUNCTION DOC] HELPER: Compile a module that unsqueezes both inputs in place with
        # distinct inputs, run it with duplicates (allowed), recompile for duplicate inputs
        # (real or fake tensors per the fake flag), run with duplicates, then confirm calling
        # with distinct tensors raises AssertionError about the broken duplicate assumption.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that unsqueezes both inputs in place and returns a
            # tuple of their sum.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Unsqueeze both inputs in place and return a one-element
                # tuple with their sum.
                x.unsqueeze_(0)
                y.unsqueeze_(0)
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()

        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)

            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)

        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)

        fxy(x, y)
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()
        fxy(x, x)  # is ok!

        if fake:
            fxx = aot_module_simplified(F(), (fake_x, fake_x), nop)
        else:
            fxx = aot_module_simplified(F(), (x, x), nop)

        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()
        fxx(x, x)
        # Note This should not raise! Once we have guards in place here,
        # we will have this working correctly, as it should recompile.
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxx(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",
        )

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_requires_grad(self, counter):
        # [FUNCTION DOC] Run the invalid-requires-grad check with real tensors.
        self._test_invalid_requires_grad(counter, fake=False)

    # See Note: Dynamo recompilation guarding invalid grad for why this test exists
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_requires_grad_fake(self, counter):
        # [FUNCTION DOC] Run the invalid-requires-grad check with fake tensors.
        self._test_invalid_requires_grad(counter, fake=True)

    def _test_invalid_requires_grad(self, counter, fake):
        # [FUNCTION DOC] HELPER: Compile a module for inputs that require grad, compare outputs
        # and grads with eager also for a non-requiring second input, then compile for a
        # non-requiring second input and confirm calling it with a grad-requiring input raises
        # AssertionError about the requires-grad assumption.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that returns a tuple with the sum of its inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Return a one-element tuple with the sum of both inputs.
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        z = torch.randn(3, 3, requires_grad=False)

        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)

            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)
            fake_z = fake_mode.from_tensor(z)

        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)

        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))

        if fake:
            fxz = aot_module_simplified(F(), (fake_x, fake_z), nop)
        else:
            fxz = aot_module_simplified(F(), (x, z), nop)

        compare_equal_outs_and_grads(self, F(), fxz, (x, z))

        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxz(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would not require grad, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",
        )

    def test_custom_autograd(self):
        # [FUNCTION DOC] Check a custom autograd Function whose backward adds one to the
        # gradient works through the harness.
        class CustomFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function cloning in forward with a backward adding
            # one to the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone of the input.
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient plus one.
                return grad_output + 1

        def f(x):
            # [FUNCTION DOC] HELPER: Apply the custom Function.
            return CustomFn.apply(x)

        self.verify_aot_autograd(f, [torch.randn(3)])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_autocast_disable_guard(self):
        # [FUNCTION DOC] CUDA only. Within the disable-autocast guard a CUDA matrix
        # multiplication keeps dtype float32.
        with torch._C._DisableAutocast():
            x = torch.rand([4, 4]).cuda()
            y = x @ x
            self.assertEqual(y.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_nonidempotent_amp(self):
        # [FUNCTION DOC] CUDA only. Check an einsum plus log softmax with float32 and float16
        # inputs under autocast, without and with requires_grad, via the harness.
        def f(self_s_emb, add_3):
            # [FUNCTION DOC] HELPER: Einsum of the two inputs, log softmax, returned as a
            # one-element tuple.
            einsum_2 = torch.functional.einsum("ah,th->t", self_s_emb, add_3)
            log_softmax_2 = einsum_2.log_softmax(-1)
            return (log_softmax_2,)

        args = [
            torch.rand((1, 256), dtype=torch.float32, device="cuda"),
            torch.rand((30, 256), dtype=torch.float16, device="cuda"),
        ]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

        args = [e.requires_grad_(True) for e in args]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "CUDNN is unavailable")
    def test_batch_norm_amp(self):
        # [FUNCTION DOC] CUDA and cudnn required. Traces a half precision batch norm with
        # float32 parameters through make_fx using the cudnn batch norm decomposition
        # (miopen on ROCm) and checks outputs are close to the direct call; raises
        # AssertionError on mismatch.
        device = "cuda"
        input_dtype = torch.float16
        param_dtype = torch.float32
        weight, bias = (
            torch.ones(64, device=device, dtype=param_dtype, requires_grad=True)
            for _ in range(2)
        )
        running_mean, running_var = (
            torch.ones(64, device=device, dtype=param_dtype) for _ in range(2)
        )

        def bn(x):
            # [FUNCTION DOC] HELPER: Call the vendor batch norm kernel on x with the surrounding
            # weight, bias and running stats in training mode.
            fn = (
                torch.ops.aten.cudnn_batch_norm
                if torch.version.hip is None
                else torch.ops.aten.miopen_batch_norm
            )
            return fn(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                False,
                0.1,
                1e-05,
            )

        inp = torch.ones(
            torch.Size([16, 64, 112, 112]), dtype=input_dtype, device=device
        )

        ref = bn(inp)
        cudnn_batch_norm_decomp = torch._decomp.get_decompositions(
            {torch.ops.aten.cudnn_batch_norm}
        )
        aot_fn = make_fx(bn, decomposition_table=cudnn_batch_norm_decomp)(inp)
        res = aot_fn(inp)
        for a, b in zip(ref, res):
            if not torch.allclose(a, b):
                raise AssertionError(f"Tensors not allclose: {a} vs {b}")

    def test_output_op_depending_on_symint(self):
        # [FUNCTION DOC] End-to-end check that an expand whose size depends on a symbolic int
        # compiles with dynamic shapes and the min cut partitioner (regression for a proxy
        # association bug); output must match eager.
        """
        It won't be obvious from reading this test what it's testing for.  We should probably make it into a more
        focused unit test.

        An issue with the following program was the expand op would end up depending on a symint whose proxy was
        incorrectly associated with one of the grad tensors rather than input tensors.  It broke partitioner logic
        and the net result was aot_function failed to produce a function and threw an exception instead.
        """
        inp = torch.randn(5, requires_grad=True)

        def f(x):
            # [FUNCTION DOC] HELPER: Expand the input to its own shape.
            return x.expand(x.shape)

        # TODO(whc) make this work (test setup is wrong somehow)
        # joint_forward_backward = create_joint_forward_backward(f)
        # out = f(inp)
        # joint_inputs =  ([inp], [out.detach().contiguous()])
        # fx_g = make_fx(joint_forward_backward)(*joint_inputs)
        # TODO: assert outputs of fwd graph trace to correct symint

        # e2e test that fails without symint clone fix
        af = aot_function(
            f,
            nop,
            partition_fn=partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
            dynamic=True,
        )
        out = af(inp)
        self.assertEqual(out, f(inp))

    def test_inference_mode(self):
        # [FUNCTION DOC] Check an aot_module of a Linear layer gives the same result as eager
        # under inference mode.
        m = torch.nn.Linear(4, 4)
        inp = torch.randn(4, 4)

        aot_mod = aot_module(m, fw_compiler=nop)

        with torch.inference_mode():
            out_ref = m(inp)
            out_test = aot_mod(inp)
        self.assertEqual(out_ref, out_test)

    def test_default_partitioner_saves_symints_not_tensors_for_bw(self):
        # [FUNCTION DOC] Check the default partitioner saves symbolic sizes rather than the
        # input tensor for backward (saving the input would fail because it is mutated through
        # an alias); compares output and gradient with eager under dynamic shapes.
        """
        In this test, the important thing is that primals_1 is **only** needed in the backward
        in order to grab its sizes.
        We need to assert that what we save for the backward are the tensor's sizes, and not the tensor itself.

        The way this test is set up, it will actually fail if we try to save the input tensor for backward.
        Why?
        b.masked_fill_(c, 0) has a backward that requires knowing a's sizes
        b.masked_fill_(c, 0) **also** mutates a (because b and a are aliased)
        The autograd engine yells at us if we save "a" for backward, and then try to mutate it.
        """

        def f(a):
            # [FUNCTION DOC] HELPER: Take the first row, build a boolean mask of ones, fill the
            # row with zero in place and return the result.
            b = a[0]
            c = torch.ones_like(b, dtype=torch.bool)
            d = b.masked_fill_(c, 0)
            return d

        compiled_f = aot_function(f, nop, dynamic=True)
        inp_ref = torch.ones(2, 2, requires_grad=True)
        inp_test = torch.ones(2, 2, requires_grad=True)

        out_ref = f(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_copied_in_graph(self):
        # [FUNCTION DOC] Check a module with a mutated buffer and two parameters under
        # aot_module with input mutations kept; asserts the forward graph includes a copy back
        # into the buffer, and that outputs, parameter grads and input grads match eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer, and two scalar parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create a zero buffer and two zero parameters.
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.w1 = torch.nn.Parameter(torch.zeros(1))
                self.w2 = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add one to the buffer in place and return the sum of
                # the product of parameters and input plus the buffer sum.
                self.buf.add_(1)
                return (self.w1 * x * self.w2).sum() + self.buf.sum()

        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)

        fw_graph_cell = [None]
        compiled_f = aot_module(
            model_for_compile,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp_ref = torch.ones(1, requires_grad=True)
        inp_test = torch.ones(1, requires_grad=True)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    add = torch.ops.aten.add.Tensor(primals_3, 1)
    mul = torch.ops.aten.mul.Tensor(primals_1, primals_4)
    mul_1 = torch.ops.aten.mul.Tensor(mul, primals_2)
    sum_1 = torch.ops.aten.sum.default(mul_1);  mul_1 = None
    sum_2 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    copy_ = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
    return (add_1, primals_1, primals_2, primals_4, mul)""",
        )

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        eager_grads = [p.grad for _, p in model_for_eager.named_parameters()]
        compile_grads = [p.grad for _, p in model_for_compile.named_parameters()]

        self.assertEqual(eager_grads, compile_grads)
        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_copied_in_graph_with_different_shapes(self):
        # [FUNCTION DOC] Same buffer mutation check with different shapes (matrix parameter and
        # square buffer); asserts the graph and compares outputs and gradients with eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer and a matrix parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the buffer and the parameter.
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(4, 4))
                self.w = torch.nn.Parameter(
                    torch.Tensor([[4, 5], [1, 2], [6, 7], [8, 9]])
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add one to the buffer in place and return the sum of
                # the matrix product plus the buffer sum.
                self.buf.add_(1)
                return (self.w @ x).sum() + self.buf.sum()

        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)

        fw_graph_cell = [None]
        compiled_f = aot_module(
            model_for_compile,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp_ref = torch.ones(2, 4, requires_grad=True)
        inp_test = torch.ones(2, 4, requires_grad=True)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    add = torch.ops.aten.add.Tensor(primals_2, 1)
    mm = torch.ops.aten.mm.default(primals_1, primals_3)
    sum_1 = torch.ops.aten.sum.default(mm);  mm = None
    sum_2 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    copy_ = torch.ops.aten.copy_.default(primals_2, add);  primals_2 = add = copy_ = None
    return (add_1, primals_1, primals_3)""",
        )
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        eager_grads = [p.grad for _, p in model_for_eager.named_parameters()]
        compile_grads = [p.grad for _, p in model_for_compile.named_parameters()]

        self.assertEqual(eager_grads, compile_grads)

        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_batch_norm(self):
        # [FUNCTION DOC] Check a BatchNorm1d module with input mutation kept: asserts forward
        # graph (functional batch norm plus copies of the updated statistics and counter) and
        # backward graph, and compares outputs and grads with eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a BatchNorm1d layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the batch norm layer.
                super().__init__()
                self.m = torch.nn.BatchNorm1d(100)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the batch norm layer.
                return self.m(x)

        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_f = aot_module(
            model_for_compile,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=bw_graph_cell)
            ),
            keep_inference_input_mutations=True,
        )
        inp_ref = torch.ones(20, 100, requires_grad=True)
        inp_test = torch.ones(20, 100, requires_grad=True)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_5, 1)
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(primals_6, primals_1, primals_2, primals_3, primals_4, True, 0.1, 1e-05);  primals_2 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    copy_ = torch.ops.aten.copy_.default(primals_3, getitem_3);  primals_3 = copy_ = None
    copy__1 = torch.ops.aten.copy_.default(primals_4, getitem_4);  primals_4 = copy__1 = None
    copy__2 = torch.ops.aten.copy_.default(primals_5, add);  primals_5 = add = copy__2 = None
    return (getitem, primals_1, primals_6, getitem_1, getitem_2, getitem_3, getitem_4)""",
        )

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        eager_grads = [p.grad for _, p in model_for_eager.named_parameters()]
        compile_grads = [p.grad for _, p in model_for_compile.named_parameters()]
        self.assertEqual(eager_grads, compile_grads)

        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_6, getitem_1, getitem_2, getitem_3, getitem_4, tangents_1):
    native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(tangents_1, primals_6, primals_1, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  tangents_1 = primals_6 = primals_1 = getitem_3 = getitem_4 = getitem_1 = getitem_2 = None
    getitem_5 = native_batch_norm_backward[0]
    getitem_6 = native_batch_norm_backward[1]
    getitem_7 = native_batch_norm_backward[2];  native_batch_norm_backward = None
    return (getitem_6, getitem_7, None, None, None, getitem_5)""",
        )

        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_new_inp_requires_grad_now(self):
        # [FUNCTION DOC] Check an in-place add on an input that does not require grad with a
        # second input that does: forward graph clones instead of copying back; backward returns
        # no gradient for the first and the incoming gradient for the second.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Add y to x in place and return x.
            return x.add_(y)

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=bw_graph_cell)
            ),
            keep_inference_input_mutations=True,
        )

        inp_ref = (
            torch.ones(20, 100, requires_grad=False),
            torch.ones(20, 100, requires_grad=True),
        )
        inp_test = (
            torch.ones(20, 100, requires_grad=False),
            torch.ones(20, 100, requires_grad=True),
        )

        out_ref = f(*inp_ref)
        out_test = compiled_f(*inp_test)

        # There is no copy_ method
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(clone, primals_2);  clone = primals_2 = None
    return (add, add)""",
        )

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, tangents_1):
    return (None, tangents_1)""",
        )

    def test_real_weights_in_symbolic_mode(self):
        # [FUNCTION DOC] Check make_fx in symbolic mode with real weights allowed: traced and
        # functionalized graphs match eager, have one placeholder (no parameter lifting), and
        # tracing with non-fake inputs not allowed raises an error about converting tensors to
        # fake tensors.
        from functorch.experimental import functionalize

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the linear layer.
                x = self.linear(x)
                return x

        m = M().eval()

        inp = torch.randn(2, 5)

        gm = make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)
        self.assertEqual(gm(torch.ones(2, 5)), m(torch.ones(2, 5)))

        gm_functionalized = make_fx(
            functionalize(
                gm,
            ),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(inp)
        self.assertEqual(gm_functionalized(torch.ones(2, 5)), m(torch.ones(2, 5)))

        inp_count = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                inp_count += 1

        # No more param lifting
        self.assertEqual(inp_count, 1)

        inp_count = 0
        for node in gm_functionalized.graph.nodes:
            if node.op == "placeholder":
                inp_count += 1

        # No more param lifting
        self.assertEqual(inp_count, 1)

        with self.assertRaisesRegex(
            Exception, "Please convert all Tensors to FakeTensors"
        ):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=False)(
                torch.randn(2, 5)
            )

    def test_real_weights_in_symbolic_mode_with_inplace_ops(self):
        # [FUNCTION DOC] Check tracing in symbolic mode with real weights where a buffer
        # is mutated in place and resized; expects an error mentioning that metadata
        # cannot be changed.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the buffer.
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add three to the buffer in place, resize the result,
                # raise AssertionError if its shape differs from the buffer shape, return the
                # sum of the input and the buffer.
                y = self.buffer.add_(3)
                y.resize_([20])
                if y.shape != self.buffer.shape:
                    raise AssertionError(
                        f"Expected y.shape {y.shape} == buffer.shape {self.buffer.shape}"
                    )
                return x.sum() + self.buffer.sum()

        m = M().eval()
        inp = torch.randn(2, 5)
        # inplace mutation on attr is not allowed
        with self.assertRaisesRegex(Exception, "Can't call metadata"):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)

    def _compile_and_erase_bases(self, *output_view_indices):
        # [FUNCTION DOC] HELPER: Return a decorator that compiles a function with
        # aot_function using a forward compiler that wraps the outputs at the given positions
        # in a tensor subclass without base or view function, forcing view reconstruction
        # without view replay.
        # Overrides _base and _view_func tensor attributes, so as to avoid the view-replay
        # execution path when reconstructing views.
        class NoViewReplayTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass reporting no base and no view
            # function; dispatch simply forwards to the default behavior.
            @staticmethod
            def __new__(cls, tensor):
                # [FUNCTION DOC] HELPER: Create a wrapper subclass instance matching the
                # tensor's shape, strides, offset, device, layout, requires_grad and dtype.
                kwargs = {
                    "strides": tensor.stride(),
                    "storage_offset": tensor.storage_offset(),
                    "device": tensor.device,
                    "layout": tensor.layout,
                    "requires_grad": tensor.requires_grad,
                    "dtype": tensor.dtype,
                }
                return torch.Tensor._make_wrapper_subclass(cls, tensor.shape, **kwargs)

            @property
            def _base(self):
                # [FUNCTION DOC] HELPER: Always report no base.
                return None

            @property
            def _view_func(self):
                # [FUNCTION DOC] HELPER: Always report no view function.
                return None

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Forward the operation to the default dispatch
                # implementation with empty kwargs if none given.
                if kwargs is None:
                    kwargs = {}
                return super().__torch_dispatch__(func, types, args, kwargs)

        # Wraps the outputs that are views of the FX graph 'g' with NoViewReplayTensor,
        # since they are the only ones that will get reconstructed.
        def wrapper(g, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Run the graph and wrap the outputs at the chosen positions
            # in the no-view-replay subclass.
            outs = list(g(*args, **kwargs))
            for i in output_view_indices:
                outs[i] = NoViewReplayTensor(outs[i])
            return tuple(outs)

        return lambda f: aot_function(f, fw_compiler=lambda g, _: partial(wrapper, g))

    def test_output_aliases_input_view_meta_replay(self):
        # [FUNCTION DOC] Check an output that is a view of an input, rebuilt without view
        # replay, still gets a view backward node.
        @self._compile_and_erase_bases(0)
        def f(a):
            # [FUNCTION DOC] HELPER: Return a flattened view of the input.
            return a.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out = f(inp)

        self.assertIsNotNone(out.grad_fn)
        self.assertExpectedInline(
            str(out.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_output_aliases_intermediate_view_meta_replay(self):
        # [FUNCTION DOC] Check two outputs that are views of an intermediate, rebuilt without
        # view replay, both have a view backward node.
        @self._compile_and_erase_bases(0, 1)
        def f(a):
            # [FUNCTION DOC] HELPER: Clone the input and return two flattened views of
            # the clone.
            b = a.clone()
            return b.view(-1), b.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp)

        self.assertIsNotNone(out1.grad_fn)
        self.assertExpectedInline(
            str(out1.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_output_aliases_output_view_meta_replay(self):
        # [FUNCTION DOC] Check an output that is a view of another output, rebuilt without view
        # replay: storages are shared and the view has a view backward node.
        @self._compile_and_erase_bases(1)
        def f(a):
            # [FUNCTION DOC] HELPER: Return an intermediate and a flattened view of it.
            b = a.add(10)
            return b, b.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp)

        self.assertEqual(out1.untyped_storage(), out2.untyped_storage())
        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    @patch("torch._dynamo.config.assume_static_by_default", False)
    def test_dynamic_output_aliases_input_view_meta_replay(self):
        # [FUNCTION DOC] With dynamic shapes off by default patched to false and inductor
        # backend: views with a symbolic size must be rebuilt through as_strided (first output)
        # while a static view uses a view backward node.
        # - torch.compile: using it so we can have a SymInt in the FX graph.
        # - Compiling with inductor, so that tensor._base isn't tracked.
        #
        # This should force the use of as_strided in the view reconstruction path.
        # The first 2 view-replay paths won't be taken because:
        #   - target_functional_tensor will be symbolic (_functionalize_is_symbolic call)
        #   - tensor._base will be None
        @torch.compile(backend="inductor")
        def f(a, sz):
            # [FUNCTION DOC] HELPER: Return a view with the given size and a flattened view.
            return a.view(sz), a.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp, (4,))

        self.assertIsNotNone(out1.grad_fn)
        self.assertExpectedInline(
            str(out1.grad_fn.__class__), """<class 'AsStridedBackward0'>"""
        )

        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_duplicated_arguments_on_tensor_overlap(self):
        # [FUNCTION DOC] Regression check that duplicated size arguments with overlapping tensor
        # storages compile with dynamic shapes without error.
        # Test whether we correctly handle duplicated arguments when changing the
        # parameters, so that we take the base tensor as argument.
        #
        # - t0 and t1 must have storage overlap: triggers the target execution flow.
        # - s0 and s1 must be equal: triggers the error in the target execution flow.

        @torch.compile(dynamic=True)
        def foo(t0, t1, s0, s1):
            # [FUNCTION DOC] HELPER: Add each scalar to its tensor in place and return both.
            return t0.add_(s0), t1.add_(s1)

        tensor = torch.rand(10)
        foo(tensor, tensor[1:-1], 2, 2)

    @parametrize("use_autograd", [False, True])
    def test_mark_outputs_dynamic(self, use_autograd: bool):
        # [FUNCTION DOC] Parameterized by whether autograd is used. Resets counters and dynamo,
        # compiles two aot_eager functions, then checks outputs of a static-shape call do not
        # carry the propagated dynamic indices marker while outputs of a later call with new
        # (dynamic) sizes do; two AOT compilations are counted each time.
        counters.clear()
        torch._dynamo.reset()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of the two inputs.
            return torch.matmul(x, y)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn2(z):
            # [FUNCTION DOC] HELPER: Double the input.
            return z * 2

        # 1. static
        x = torch.randn(10, 10, requires_grad=use_autograd)
        y = torch.randn(10, 10, requires_grad=use_autograd)
        out = fn(x, y)
        self.assertFalse(hasattr(out, "_dynamo_propagated_dynamic_indices"))
        out2 = fn2(out)
        self.assertFalse(hasattr(out2, "_dynamo_propagated_dynamic_indices"))
        self.assertEqual(counters["aot_autograd"]["total"], 2)
        counters.clear()

        # 2. dynamic
        x = torch.randn(20, 20)
        y = torch.randn(20, 20)
        out = fn(x, y)
        self.assertTrue(hasattr(out, "_dynamo_propagated_dynamic_indices"))
        out2 = fn2(out)
        self.assertTrue(hasattr(out2, "_dynamo_propagated_dynamic_indices"))
        self.assertEqual(counters["aot_autograd"]["total"], 2)
        counters.clear()
        torch._dynamo.reset()

    def test_mark_activations_dynamic(self):
        # [FUNCTION DOC] Checks activations saved for backward are marked dynamic only when the
        # compile was dynamic: uses saved-tensor hooks whose pack and unpack check for the
        # propagated dynamic marker; static first call expects no marker, a second call with new
        # sizes expects the marker; one AOT compile each.
        counters.clear()
        torch._dynamo.reset()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Chain of four matrix products with the second input.
            out = torch.matmul(x, y)
            out2 = torch.matmul(out, y)
            out3 = torch.matmul(out2, y)
            return torch.matmul(out3, y)

        def make_assert_pack(dynamic):
            # [FUNCTION DOC] HELPER: Return a saved-tensor pack hook that raises AssertionError
            # unless the presence of the dynamic marker on the activation equals the expected
            # flag, and returns the activation.
            def pack(activation):
                # [FUNCTION DOC] HELPER: Raise AssertionError unless the marker presence equals
                # the expected flag; return the activation.
                if hasattr(activation, "_dynamo_propagated_dynamic_indices") != dynamic:
                    raise AssertionError(
                        f"Expected hasattr(..., '_dynamo_propagated_dynamic_indices') to be {dynamic}"
                    )
                return activation

            return pack

        def make_assert_unpack(dynamic):
            # [FUNCTION DOC] HELPER: Return a saved-tensor unpack hook that raises
            # AssertionError unless the presence of the dynamic marker on the activation equals
            # the expected flag, and returns the activation.
            def unpack(activation):
                # [FUNCTION DOC] HELPER: Raise AssertionError unless the marker presence equals
                # the expected flag; return the activation.
                if hasattr(activation, "_dynamo_propagated_dynamic_indices") != dynamic:
                    raise AssertionError(
                        f"Expected hasattr(..., '_dynamo_propagated_dynamic_indices') to be {dynamic}"
                    )
                return activation

            return unpack

        # 1. static
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(False), make_assert_unpack(False)
        ):
            fn(x, y)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()

        # 2. dynamic
        x = torch.randn(20, 20, requires_grad=True)
        y = torch.randn(20, 20, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(True), make_assert_unpack(True)
        ):
            fn(x, y)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()
        torch._dynamo.reset()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @torch._functorch.config.patch(saved_tensors_hooks_filtering_mode="no_static")
    @torch._functorch.config.patch(recompute_views=True)
    def test_saved_tensors_hooks_mutations_raise(self):
        # [FUNCTION DOC] CUDA only, with no_static hook filtering and view recomputation
        # configs. Saved tensor hooks that mutate their input in place under compile with
        # dynamic inputs must raise AssertionError that mutations in saved tensor hooks are not
        # allowed (inner compiler exception is re-raised).
        ctx = torch.autograd.graph.saved_tensors_hooks
        device = "cuda"

        class SAF(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function that returns its input and saves it, with a
            # backward adding the saved input to the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it.
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Return the gradient plus the saved input.
                (saved_x,) = ctx.saved_tensors
                return gx + saved_x

        def mutate(x):
            # [FUNCTION DOC] HELPER: Double the tensor in place and return it.
            return x.mul_(2)

        def fn(x):
            # [FUNCTION DOC] HELPER: Double the input and apply the saving Function.
            x = 2 * x
            x = SAF.apply(x)
            return x

        def inp_fn():
            # [FUNCTION DOC] HELPER: Make a CUDA ones matrix requiring grad with both dimensions
            # marked dynamic.
            x = torch.ones(2, 3, device=device, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return x

        with self.assertRaisesRegex(
            AssertionError, "Saved tensors hooks with inputs mutations are not allowed"
        ):
            try:
                with ctx(*saved_tensors_hooks_to_gm(mutate, mutate, None, None)):
                    x = inp_fn()
                    y = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
                    y.sum().backward()
            except torch._dynamo.exc.BackendCompilerFailed as e:
                raise e.inner_exception from e

    def test_mark_activations_dynamic_with_nested(self):
        # [FUNCTION DOC] Like the activation marking test but with a jagged nested tensor input
        # whose flattened tensors shift output offsets; checks hook-observed activations are
        # unmarked for static sizes and marked for new dynamic sizes.
        # The flattened tensors of the nested tensor aren't
        # marked as activations, but they add some offset
        # to the fw_outs. This test ensures that we handle
        # that offset properly.
        counters.clear()
        torch._dynamo.reset()

        def make_assert_pack(dynamic):
            # [FUNCTION DOC] HELPER: Return a saved-tensor pack hook that raises AssertionError
            # unless presence of the dynamic marker equals the expected flag.
            def pack(activation):
                # [FUNCTION DOC] HELPER: Raise AssertionError unless the marker presence equals
                # the expected flag; return the activation.
                if hasattr(activation, "_dynamo_propagated_dynamic_indices") != dynamic:
                    raise AssertionError(
                        f"Expected hasattr(..., '_dynamo_propagated_dynamic_indices') to be {dynamic}"
                    )
                return activation

            return pack

        def make_assert_unpack(dynamic):
            # [FUNCTION DOC] HELPER: Return a saved-tensor unpack hook that raises
            # AssertionError unless presence of the dynamic marker equals the expected flag.
            def unpack(activation):
                # [FUNCTION DOC] HELPER: Raise AssertionError unless the marker presence equals
                # the expected flag; return the activation.
                if hasattr(activation, "_dynamo_propagated_dynamic_indices") != dynamic:
                    raise AssertionError(
                        f"Expected hasattr(..., '_dynamo_propagated_dynamic_indices') to be {dynamic}"
                    )
                return activation

            return unpack

        # 1. static
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y, nt):
            # [FUNCTION DOC] HELPER: Sum of a matrix product plus a clone of the nested tensor.
            out = torch.matmul(x, y)
            return out.sum() + nt.clone()

        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(False), make_assert_unpack(False)
        ):
            fn(x, y, nt)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()

        # 2. dynamic
        x = torch.randn(20, 20, requires_grad=True)
        y = torch.randn(20, 20, requires_grad=True)
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(True), make_assert_unpack(True)
        ):
            fn(x, y, nt)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()
        torch._dynamo.reset()

    def test_compute_outer_size_and_stride_bug(self):
        # [FUNCTION DOC] Builds a subclass creation metadata object bypassing its constructor,
        # with two plain inner tensor entries, dynamic outer size (unknown) and partially
        # unknown stride, then checks computing the outer size and stride from a packed list of
        # tensors and symbolic ints returns size (4, 6) and stride (6, 1).
        import dataclasses

        from torch._functorch._aot_autograd.schemas import (
            PlainTensorMeta,
            SubclassCreationMeta,
        )

        # Simulate a 2D TwoTensor(4, 6), contiguous, with dynamic shapes.
        # After make_runtime_safe(), outer_size=(None, None), outer_stride=(None, 1).
        # The packed symints in all_args at curr_start_idx are: [4, 6, 6]
        # (2 size symints for the two None entries, 1 stride symint for the first None).
        meta = object.__new__(SubclassCreationMeta)
        for f in dataclasses.fields(SubclassCreationMeta):
            setattr(meta, f.name, f.default)
        meta.flat_tensor_start_idx = 0
        meta.arg_count = 2
        meta.included_subclass_symints = True
        meta.attrs = {
            "a": PlainTensorMeta(unwrapped_idx=0),
            "b": PlainTensorMeta(unwrapped_idx=1),
        }
        meta.outer_size = (None, None)
        meta.outer_stride = (None, 1)
        meta.meta = None
        meta.original_subclass = None
        meta.original_subclass_type = TwoTensor
        meta.memory_format = None

        # all_args: [tensor_a, tensor_b, 4, 6, 6]
        # curr_start_idx = 2 (after the two inner tensors)
        a = torch.randn(4, 6)
        b = torch.randn(4, 6)
        all_args = [a, b, 4, 6, 6]

        outer_size, outer_stride = meta.compute_outer_size_and_stride(
            all_args, curr_start_idx=2
        )
        self.assertEqual(outer_size, (4, 6))
        self.assertEqual(outer_stride, (6, 1))

    def test_subclass_dynamic_stride(self):
        # [FUNCTION DOC] Compile a sine function over a TwoTensor with dynamic shapes on the
        # aot_eager backend and check the output stride equals the input stride.
        @torch.compile(backend="aot_eager", dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Sine of the input.
            return x.sin()

        a = torch.randn(4, 6)
        b = torch.randn(4, 6)
        inp = TwoTensor(a, b)
        out = f(inp)
        self.assertEqual(out.stride(), inp.stride())

    def _make_model_and_input(self, hidden=1024, vocab=4096, seq_len=128):
        # [FUNCTION DOC] HELPER: Build a small bias-free multi-layer perceptron producing
        # vocabulary logits on CUDA, a grad-requiring random input and random integer labels;
        # returns all three.
        """Build a small model that produces non-scalar output (like an LM head)."""
        model = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, vocab, bias=False),
        ).cuda()
        x = torch.randn(seq_len, hidden, device="cuda", requires_grad=True)
        labels = torch.randint(0, vocab, (seq_len,), device="cuda")
        return model, x, labels

    def _make_refcount_probe(self):
        # [FUNCTION DOC] HELPER: Create an autograd Function that receives gradients in a boxed
        # list, pops the gradient, keeps only a weak reference, deletes it, and records in a
        # result dictionary whether the weak reference is dead (no extra references held).
        # Returns the Function class and the dictionary.
        """Create a boxed-grads probe that checks the framework holds no extra refs.

        The probe uses boxed_grads_call=True so PyNode::apply moves grads
        into a mutable list (same mechanism as CompiledFunction). After
        removing the grad from the list, a weakref must become dead — proving
        the framework released all its refs.

        Returns (ProbeClass, result_box). After backward,
        result_box["ref_dead"] is True if no extra refs were held."""
        result_box = {}

        class RefcountProbe(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function with boxed gradient calls that clones its
            # input and records in the result dictionary whether the popped gradient was freed.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, pred):
                # [FUNCTION DOC] HELPER: Return a clone of the prediction.
                return pred.clone()

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Pop the gradient from the list, clone it for return,
                # weakly reference and delete the original, record whether the reference is
                # dead, return the clone.
                grad = grads.pop(0)
                grad_for_return = grad.clone()
                ref = weakref.ref(grad)
                del grad
                result_box["ref_dead"] = ref() is None
                return grad_for_return

        return RefcountProbe, result_box

    def _assert_no_extra_refs(self, result_box):
        # [FUNCTION DOC] HELPER: Assert the result dictionary has the freed flag and that it is
        # true, with a message that the framework holds extra references to the tangent.
        """Assert the framework holds no extra refs to the grad tensor."""
        self.assertIn("ref_dead", result_box)
        self.assertTrue(
            result_box["ref_dead"],
            "Framework holds extra refs to tangent: weakref still alive after "
            "removing all user-visible references",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_tangent_freed_compiled_model_and_loss(self):
        # [FUNCTION DOC] CUDA only. Compiled model and compiled loss with a probe function
        # between them; after backward checks the tangent passed to the model backward was freed
        # by the boxed calling convention.
        """Scenario: compiled model + compiled loss (torchtitan simple_fsdp pattern).

        Mirrors torchtitan's simple_fsdp training loop:
          pred = compiled_model(input)   # torch.compile(model)
          loss = compiled_loss(pred, labels)  # torch.compile(loss_fn)
          loss.backward()

        The tangent for model backward is created internally by loss backward.
        No user variable holds it — only the C++ pyInputs tuple on the stack.
        The boxed calling convention alone frees it."""
        model, x, labels = self._make_model_and_input()
        compiled_model = torch.compile(model, backend="inductor")

        def loss_fn(pred, labels):
            # [FUNCTION DOC] HELPER: Cross entropy of float-cast predictions and labels.
            return torch.nn.functional.cross_entropy(pred.float(), labels)

        compiled_loss = torch.compile(loss_fn, backend="inductor")

        Probe, refcount_box = self._make_refcount_probe()
        pred = compiled_model(x)
        pred = Probe.apply(pred)
        loss = compiled_loss(pred, labels)
        del pred
        loss.backward()

        self._assert_no_extra_refs(refcount_box)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_tangent_freed_compiled_model_eager_loss(self):
        # [FUNCTION DOC] CUDA only. Compiled model with an eager cross entropy loss; the tangent
        # must be freed during backward.
        """Scenario: compiled model + eager loss (common training pattern).

        Similar to torchtitan but loss is not compiled. The tangent is
        created by eager cross_entropy backward. Boxed convention frees it."""
        model, x, labels = self._make_model_and_input()
        compiled_model = torch.compile(model, backend="inductor")

        Probe, refcount_box = self._make_refcount_probe()
        pred = compiled_model(x)
        pred = Probe.apply(pred)
        loss = torch.nn.functional.cross_entropy(pred.float(), labels)
        del pred
        loss.backward()

        self._assert_no_extra_refs(refcount_box)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_tangent_user_provided_via_pop(self):
        # [FUNCTION DOC] CUDA only. Compiled model and loss where the loss is held only in a
        # list and popped when calling backward; the tangent must still be freed.
        """Scenario: user creates tangent but releases it before backward runs.

        The user wraps the tangent in a list and calls out.backward(l.pop()).
        After pop(), no Python variable holds the tangent — same as torchtitan.
        The boxed calling convention should free it."""
        model, x, _labels = self._make_model_and_input()
        compiled_model = torch.compile(model, backend="inductor")

        def loss_fn(pred, labels):
            # [FUNCTION DOC] HELPER: Cross entropy of float-cast predictions and labels.
            return torch.nn.functional.cross_entropy(pred.float(), labels)

        compiled_loss = torch.compile(loss_fn, backend="inductor")
        _, _, labels = self._make_model_and_input()

        Probe, refcount_box = self._make_refcount_probe()
        pred = compiled_model(x)
        pred = Probe.apply(pred)
        loss = compiled_loss(pred, labels)
        del pred

        # Wrap loss in a list and use pop() — no user variable holds loss
        # after this. The tangent created by loss backward has no user ref.
        loss_list = [loss]
        del loss
        loss_list.pop().backward()

        self._assert_no_extra_refs(refcount_box)


def extract_graph(fx_g, _, graph_cell):
    # [FUNCTION DOC] HELPER: Compiler that stores the graph into the first slot of a list cell
    # and returns it unchanged.
    graph_cell[0] = fx_g
    return fx_g


def get_ins_outs(fx_g):
    # [FUNCTION DOC] HELPER: Collect the placeholder nodes and the output node arguments of a
    # graph; returns a pair of the inputs (list) and outputs (tuple).
    ins = []
    outs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            ins.append(n)
        elif n.op == "output":
            outs = tuple(n.args[0])
    return ins, outs


def get_num_ins_outs(fx_g):
    # [FUNCTION DOC] HELPER: Return the counts of graph inputs and outputs as a tuple.
    return tuple(len(i) for i in get_ins_outs(fx_g))


def get_fw_bw_graph(
    f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False
):
    # [FUNCTION DOC] HELPER: Compile a function with aot_function using graph-capturing
    # compilers, a given partitioner (default min cut), default decompositions and dynamic flag,
    # run it on the inputs, backpropagate the sum, and return the forward and backward graphs.
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    aot_function(
        f,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
        bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
        partition_fn=partitioner,
        decompositions=default_decompositions,
        dynamic=dynamic,
    )(*inps).sum().backward()
    return (fw_graph_cell[0], bw_graph_cell[0])


class TestMod(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with one parameter of two ones that calls the supplied function
    # with that parameter and the module inputs.
    def __init__(self, fn):
        # [FUNCTION DOC] HELPER: Store the function and create the two-element parameter.
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2, requires_grad=True))
        self.fn = fn

    def forward(self, *args):
        # [FUNCTION DOC] HELPER: Call the stored function with the parameter followed by
        # the inputs.
        return self.fn(self.p, *args)


class TestAOTExport(AOTTestCase):
    # [CLASS DOC] Tests of aot_export_module and pre-dispatch export (functionalization graphs
    # with expected code).
    def setUp(self):
        # [FUNCTION DOC] HELPER: Run the base setup then reset dynamo state.
        super().setUp()
        torch._dynamo.reset()

    def test_aot_export_ban_dropout_mut_pre_dispatch(self):
        # [FUNCTION DOC] Check that exporting with pre_dispatch raises RuntimeError about frozen
        # storage when mutating the output of a dropout op, that without pre-dispatch the
        # mutation is recorded as a user input mutation with expected graph, and that
        # aot_function with dynamic shapes yields the same graph.
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Apply dropout (evaluation mode) then add one in place and
            # return the result in a tuple.
            y = torch.ops.aten.dropout.default(x, 0.1, train=False)
            y.add_(1)
            return (y,)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)

        gm, graph_sig = aot_export_module(
            mod, [inp], trace_joint=False, pre_dispatch=False
        )
        self.assertEqual(graph_sig.user_inputs_to_mutate, {"add": "arg1_1"})
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg1_1, 1);  arg1_1 = None
    return (add, add)""",
        )

        fw_graph_cell = [None]
        bw_graph_cell = [None]

        aot_function(
            fn,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=default_partition,
            decompositions=default_decompositions,
            dynamic=True,
        )(*inp)
        fw_graph = fw_graph_cell[0]

        self.assertExpectedInline(
            str(fw_graph.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg1_1, 1);  arg1_1 = None
    return (add, add)""",
        )

    def test_aot_export_predispatch_func_simple(self):
        # [FUNCTION DOC] Check pre-dispatch export of a function with an in-place add under
        # no_grad; asserts the graph includes the grad-mode toggles.
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Add two to the input, add two in place under no_grad,
            # return input times two plus that.
            y = x + 2
            with torch.no_grad():
                y.add_(2)
            return (x * 2 + y,)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        with torch.no_grad():
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg1_1, 2)
    _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
    add_1 = torch.ops.aten.add.Tensor(add, 2);  add = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    mul = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    add_2 = torch.ops.aten.add.Tensor(mul, add_1);  mul = add_1 = None
    return (add_2,)""",
        )

    def test_aot_export_predispatch_func_composite_implicit(self):
        # [FUNCTION DOC] Check pre-dispatch export keeps a composite implicit matmul traced
        # under enable_grad followed by an in-place add under no_grad; asserts graph text with
        # grad-mode toggles.
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Matrix product under enable_grad, in-place add of two,
            # return the sum of input sum and result sum.
            with torch.enable_grad():
                y = x @ x
            y.add_(2)
            return (x.sum() + y.sum(),)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        with torch.no_grad():
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _set_grad_enabled = torch._C._set_grad_enabled(True);  _set_grad_enabled = None
    matmul = torch.ops.aten.matmul.default(arg1_1, arg1_1)
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    add = torch.ops.aten.add.Tensor(matmul, 2);  matmul = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    sum_2 = torch.ops.aten.sum.default(add);  add = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_1,)""",
        )

    def test_aot_export_predispatch_composite_implicit_inplace(self):
        # [FUNCTION DOC] Check pre-dispatch export of the in-place absolute op on a clone of the
        # input becomes a clone and a non in-place abs.
        def fn(x, p):
            # [FUNCTION DOC] HELPER: Apply the in-place absolute op to a clone of the
            # first argument.
            return (torch.ops.aten.absolute_.default(x.clone()),)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    abs_1 = torch.ops.aten.abs.default(clone);  clone = None
    return (abs_1,)""",
        )

    def test_aot_export_predispatch_composite_implicit_linear(self):
        # [FUNCTION DOC] Check pre-dispatch export keeps the linear op as a single
        # composite operator.
        class MM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the linear layer in a tuple.
                return (self.linear(x),)

        mod = MM()
        inp = torch.randn(2, 2)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    linear = torch.ops.aten.linear.default(arg2_1, arg0_1, arg1_1);  arg2_1 = arg0_1 = arg1_1 = None
    return (linear,)""",
        )

    @unittest.expectedFailure
    def test_aot_export_predispatch_outdtype(self):
        # [FUNCTION DOC] Expected failure. Export with pre-dispatch of a module using the
        # output-dtype higher-order op on an int8 matrix product after an in-place add; the
        # expected graph text corresponds to a different program, so the assertion cannot match.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an integer weight tensor.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Store the weight.
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add two, add five in place, return the int32 output
                # dtype matrix product with the weight.
                y = x + 2
                y.add_(5)
                return (
                    out_dtype(torch.ops.aten.mm.default, torch.int32, y, self.weight),
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        mod = M(weight)
        inp = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _set_grad_enabled = torch._C._set_grad_enabled(True);  _set_grad_enabled = None
    mm = torch.ops.aten.mm.default(arg1_1, arg1_1)
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    add = torch.ops.aten.add.Tensor(mm, 2);  mm = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    sum_2 = torch.ops.aten.sum.default(add);  add = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_1,)""",
        )

    def test_aot_export_predispatch_func_view(self):
        # [FUNCTION DOC] Check pre-dispatch export of an in-place add on a matrix product
        # followed by a view; asserts graph text.
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Matrix product, in-place add of two, return the input sum
            # plus the sum of a reshaped view.
            y = x @ x
            y.add_(2)
            return (x.sum() + y.view(1, 4).sum(),)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    matmul = torch.ops.aten.matmul.default(arg1_1, arg1_1)
    add = torch.ops.aten.add.Tensor(matmul, 2);  matmul = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(add, [1, 4]);  add = None
    sum_2 = torch.ops.aten.sum.default(view_1);  view_1 = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_1,)""",
        )

    def test_aot_export_predispatch_buffer_mutation_metadata(self):
        # [FUNCTION DOC] Check pre-dispatch export with a buffer mutated in place: the buffer
        # becomes an input and output of the graph; checks running the exported module twice
        # matches eager and the signature records the buffer.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a zero buffer mutated in place by forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the zero buffer.
                super().__init__()
                self.foo = torch.nn.Buffer(torch.zeros(2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add four to the buffer in place and return the input
                # sum plus the buffer sum.
                self.foo.add_(4)
                return (x.sum() + self.foo.sum(),)

        inp = torch.randn(2, 2)

        gm, graph_sig = aot_export_module(
            Foo(), [inp], trace_joint=False, pre_dispatch=True
        )
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 4);  arg0_1 = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    sum_2 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add, add_1)""",
        )
        eager_mod = Foo()
        output_1, output_2 = gm(torch.zeros(2, 2), inp)
        eager_output = eager_mod(inp)
        self.assertTrue(torch.allclose(output_2, eager_output[0]))

        _, output_2 = gm(output_1, inp)
        eager_output = eager_mod(inp)
        self.assertTrue(torch.allclose(output_2, eager_output[0]))
        self.assertTrue("foo" in graph_sig.buffers)
        self.assertTrue(graph_sig.inputs_to_buffers["arg0_1"] == "foo")

    def test_aot_export_predispatch_with_autograd_op(self):
        # [FUNCTION DOC] Check pre-dispatch export of operations under enable_grad within a
        # no_grad outer context; asserts grad-mode toggles in the exported graph.
        def foo(p, x):
            # [FUNCTION DOC] HELPER: Under enable_grad compute two in-place adds and return
            # cosine plus sine.
            with torch.enable_grad():
                y = x + 5
                y.add_(5)
                y.add_(7)
                return (x.cos() + y.sin(),)

        inp = torch.randn(2, 2)
        mod = TestMod(foo)

        with torch.no_grad():
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _set_grad_enabled = torch._C._set_grad_enabled(True);  _set_grad_enabled = None
    add = torch.ops.aten.add.Tensor(arg1_1, 5)
    add_1 = torch.ops.aten.add.Tensor(add, 5);  add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, 7);  add_1 = None
    cos = torch.ops.aten.cos.default(arg1_1);  arg1_1 = None
    sin = torch.ops.aten.sin.default(add_2);  add_2 = None
    add_3 = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    return (add_3,)""",
        )

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torchdynamo.is_dynamo_supported(), "TorchDynamo is not supported"
    )
    def test_aot_export_predispatch_with_cond_nested(self):
        # [FUNCTION DOC] Not on Windows; requires dynamo support. Check pre-dispatch export of
        # nested conditionals with in-place ops in branches; asserts the main graph and nested
        # branch graph code.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with nested conditional branches.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Conditional on the input sum, with nested functions for
                # the branches, returns two shifted copies of the result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: sine, in-place add, then a
                    # nested conditional.
                    y = x.sin()
                    y.add_(5)

                    def true_true_fn(x):
                        # [FUNCTION DOC] HELPER: Nested true branch: sine, in-place add of
                        # seven, sine.
                        y = x.sin()
                        y.add_(7)
                        return y.sin()

                    def true_false_fn(x):
                        # [FUNCTION DOC] HELPER: Nested false branch: cosine.
                        return x.cos()

                    return torch.cond(
                        y.cos().sum() > 5, true_true_fn, true_false_fn, [y.cos()]
                    )

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: cosine, in-place add of six, sine.
                    z = x.cos()
                    z.add_(6)
                    return z.sin()

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x])
                return (a + 3, a + 4)

        inp = torch.randn(2, 2)
        gm, _ = aot_export_module(M(), [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1,));  gt = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    add = torch.ops.aten.add.Tensor(getitem, 3)
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    return (add, add_1)""",
        )

        self.assertExpectedInline(
            str(gm.true_graph_0.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(sin, 5);  sin = None
    cos = torch.ops.aten.cos.default(add)
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    gt = torch.ops.aten.gt.Scalar(sum_1, 5);  sum_1 = None
    cos_1 = torch.ops.aten.cos.default(add);  add = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (cos_1,));  gt = true_graph_0 = false_graph_0 = cos_1 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
        )

        self.assertExpectedInline(
            str(gm.true_graph_0.true_graph_0.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(sin, 7);  sin = None
    sin_1 = torch.ops.aten.sin.default(add);  add = None
    return (sin_1,)""",
        )

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torchdynamo.is_dynamo_supported(), "TorchDynamo is not supported"
    )
    def test_aot_export_predispatch_map_1(self):
        # [FUNCTION DOC] Not on Windows; requires dynamo support. Pre-dispatch export of a
        # conditional whose branches include a map higher-order op with in-place work; asserts
        # the readable normalized graph code.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with conditional branches that use map.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Conditional on the input sum choosing between two
                # branch functions; returns two shifted copies.
                def true_fn(x, r):
                    # [FUNCTION DOC] HELPER: True branch: sine, in-place add, cosine plus the
                    # sum of the second argument.
                    y = x.sin()
                    y.add_(5)
                    return y.cos() + r.sum()

                def false_fn(x, r):
                    # [FUNCTION DOC] HELPER: False branch: cosine plus two mapped sums.
                    z = x.cos()

                    def f(x, y):
                        # [FUNCTION DOC] HELPER: Map body: cosine, in-place add of five, plus
                        # the second argument.
                        a = x.cos()
                        a.add_(5)
                        return a + y

                    return (
                        z
                        + control_flow.map(f, z, r).sum()
                        + control_flow.map(f, z, r).sum()
                    )

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x, y])
                return (a + 3, a + 4)

        inps = [torch.randn(2, 2), torch.ones(2)]
        gm, _ = aot_export_module(M(), inps, trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(False, expanded_def=True)),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[2, 2]",  # PlainAOTInput(idx=0)
        arg1_1: "f32[2]",  # PlainAOTInput(idx=1)
    ):
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1, arg1_1));  gt = true_graph_0 = false_graph_0 = arg0_1 = arg1_1 = None
        getitem: "f32[2, 2]" = cond[0];  cond = None

        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem, 3)
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
        return (
            add,  # PlainAOTOutput(idx=0)
            add_1,  # PlainAOTOutput(idx=1)
        )

    class true_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2]"):
            sin: "f32[2, 2]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None

            add: "f32[2, 2]" = torch.ops.aten.add.Tensor(sin, 5);  sin = None

            cos: "f32[2, 2]" = torch.ops.aten.cos.default(add);  add = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None

            add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
            return (add_1,)

    class false_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2]"):
            cos: "f32[2, 2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

            body_graph_0 = self.body_graph_0
            map_impl = torch.ops.higher_order.map_impl(body_graph_0, [cos], [arg1_1]);  body_graph_0 = None
            getitem: "f32[2, 2]" = map_impl[0];  map_impl = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None

            add: "f32[2, 2]" = torch.ops.aten.add.Tensor(cos, sum_1);  sum_1 = None

            body_graph_1 = self.body_graph_1
            map_impl_1 = torch.ops.higher_order.map_impl(body_graph_1, [cos], [arg1_1]);  body_graph_1 = cos = arg1_1 = None
            getitem_1: "f32[2, 2]" = map_impl_1[0];  map_impl_1 = None

            sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None

            add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
            return (add_1,)

        class body_graph_0(torch.nn.Module):
            def forward(self, arg0_1: "f32[2]", arg1_1: "f32[2]"):
                cos: "f32[2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

                add: "f32[2]" = torch.ops.aten.add.Tensor(cos, 5);  cos = None

                add_1: "f32[2]" = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
                return (add_1,)

        class body_graph_1(torch.nn.Module):
            def forward(self, arg0_1: "f32[2]", arg1_1: "f32[2]"):
                cos: "f32[2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

                add: "f32[2]" = torch.ops.aten.add.Tensor(cos, 5);  cos = None

                add_1: "f32[2]" = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
                return (add_1,)
""",
        )

    def test_aot_export_predispatch_map_2(self):
        # [FUNCTION DOC] Pre-dispatch export of a module that maps a function (with an
        # in-place add inside the body) over a cosine; asserts the readable graph with a map
        # body subgraph.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a map over a body function.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Take cosine, map a body over it with the second input,
                # return the sum plus the cosine.
                z = x.cos()

                def f(x, y):
                    # [FUNCTION DOC] HELPER: Map body: cosine, in-place add of five, plus the
                    # second argument.
                    a = x.cos()
                    a.add_(5)
                    return a + y

                return (z + control_flow.map(f, z, y).sum(),)

        inps = [torch.randn(2, 2), torch.ones(2)]
        gm, _ = aot_export_module(M(), inps, trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(False, expanded_def=True)),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[2, 2]",  # PlainAOTInput(idx=0)
        arg1_1: "f32[2]",  # PlainAOTInput(idx=1)
    ):
        cos: "f32[2, 2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

        body_graph_0 = self.body_graph_0
        map_impl = torch.ops.higher_order.map_impl(body_graph_0, [cos], [arg1_1]);  body_graph_0 = arg1_1 = None
        getitem: "f32[2, 2]" = map_impl[0];  map_impl = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
        return (
            add,  # PlainAOTOutput(idx=0)
        )

    class body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "f32[2]"):
            cos: "f32[2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

            add: "f32[2]" = torch.ops.aten.add.Tensor(cos, 5);  cos = None

            add_1: "f32[2]" = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
            return (add_1,)
""",
        )

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torchdynamo.is_dynamo_supported(), "TorchDynamo is not supported"
    )
    def test_aot_export_predispatch_with_cond(self):
        # [FUNCTION DOC] Not on Windows; requires dynamo support. Pre-dispatch export of a
        # conditional whose true branch calls linear with a random weight and in-place add;
        # asserts main graph and true-branch graph code.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conditional on the input sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Conditional between two branch functions, returns two
                # shifted copies of the result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: sine, linear with a random weight,
                    # in-place add of five, cosine.
                    y = x.sin()
                    z = torch.ops.aten.linear.default(y, torch.randn(2, 2))
                    z.add_(5)
                    return z.cos()

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: cosine, in-place add of six, sine.
                    z = x.cos()
                    z.add_(6)
                    return z.sin()

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x])
                return (a + 3, a + 4)

        inp = torch.randn(2, 2)
        gm, _ = aot_export_module(M(), [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1,));  gt = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    add = torch.ops.aten.add.Tensor(getitem, 3)
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    return (add, add_1)""",
        )
        self.assertExpectedInline(
            str(gm.true_graph_0.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    randn = torch.ops.aten.randn.default([2, 2], device = device(type='cpu'), pin_memory = False)
    linear = torch.ops.aten.linear.default(sin, randn);  sin = randn = None
    add = torch.ops.aten.add.Tensor(linear, 5);  linear = None
    cos = torch.ops.aten.cos.default(add);  add = None
    return (cos,)""",
        )

    def test_aot_export_predispatch_conv_and_bn(self):
        # [FUNCTION DOC] Check pre-dispatch export of a training mode conv followed by batch
        # norm: batch norm statistics and counter are functionalized as inputs and outputs;
        # asserts graph text.
        class ConvBatchnorm(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution and a batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the convolution and batch norm layers.
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply convolution then batch norm and return in
                # a tuple.
                x = self.conv(x)
                x = self.bn(x)
                return (x,)

        mod = ConvBatchnorm()
        mod.train()
        inp = torch.randn(1, 1, 3, 3)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    conv2d = torch.ops.aten.conv2d.default(arg7_1, arg0_1, arg1_1);  arg7_1 = arg0_1 = arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  conv2d = arg2_1 = arg3_1 = arg4_1 = arg5_1 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    return (getitem_3, getitem_4, add, getitem)""",
        )

    def test_aot_export_predispatch_reshape(self):
        # [FUNCTION DOC] Check pre-dispatch export turns reshape into a view op before a sum.
        class Reshape(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes its input to a square and sums it.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reshape to four by four and return the sum in a tuple.
                y = x.reshape(4, 4)
                return (y.sum(),)

        mod = Reshape()
        inp = torch.randn(2, 8)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    view = torch.ops.aten.view.default(arg0_1, [4, 4]);  arg0_1 = None
    sum_1 = torch.ops.aten.sum.default(view);  view = None
    return (sum_1,)""",
        )

    def test_aot_export_predispatch_contiguous(self):
        # [FUNCTION DOC] Check pre-dispatch export drops a contiguous call on an already
        # contiguous input.
        class Cont(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that calls the contiguous op and sums.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply contiguous and return the sum in a tuple.
                y = torch.ops.aten.contiguous.default(x)
                return (y.sum(),)

        mod = Cont()
        inp = torch.randn(2, 8)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    return (sum_1,)""",
        )

    def test_aot_export_module_joint(self):
        # [FUNCTION DOC] Check joint (forward and backward) export of a conv, batch norm and
        # relu module with the loss index set: asserts the printed graph, signature fields
        # (parameters, buffers, user inputs, mapping from inputs and gradients) and then the
        # inference-only export graph with input mutation outputs.
        class ConvBatchnormRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with convolution, batch norm and relu producing a loss
            # and a detached output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the convolution and batch norm layers.
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply convolution, batch norm and relu; return the
                # summed loss and the detached activation.
                x = self.conv(x)
                x = self.bn(x)
                user_out = torch.nn.functional.relu(x)
                loss = user_out.sum()
                return loss, user_out.detach()

        mod = ConvBatchnormRelu()
        mod.train()
        inp = torch.randn(1, 1, 3, 3)
        mod(inp)
        fx_g, signature = aot_export_module(
            mod, [inp], trace_joint=True, output_loss_index=0
        )
        # Some important characteristics of the exported graph below:
        # 8 arguments: 2 params from conv, 2 params from batchnorm, 2 buffers from 1 batchnorm, 1 user input
        # 9 outputs: 3 mutated buffers (from batchnorm), 2 user outputs and 4 gradients (since there were 4 parameters)
        for node in fx_g.graph.nodes:
            node.meta.pop("stack_trace", None)
        self.assertExpectedInline(
            fx_g.print_readable(print_output=False, expanded_def=True),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[3, 1, 1, 1]",
        arg1_1: "f32[3]",
        arg2_1: "f32[3]",
        arg3_1: "f32[3]",
        arg4_1: "f32[3]",
        arg5_1: "f32[3]",
        arg6_1: "i64[]",
        arg7_1: "f32[1, 1, 3, 3]",
    ):
        # No stacktrace found for following nodes
        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(arg7_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1_1 = None
        add: "i64[]" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  arg3_1 = arg4_1 = arg5_1 = None
        getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
        getitem_1: "f32[3]" = _native_batch_norm_legit_functional[1]
        getitem_2: "f32[3]" = _native_batch_norm_legit_functional[2]
        getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
        getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
        relu: "f32[1, 3, 3, 3]" = torch.ops.aten.relu.default(getitem);  getitem = None
        alias: "f32[1, 3, 3, 3]" = torch.ops.aten.alias.default(relu);  alias = None
        alias_1: "f32[1, 3, 3, 3]" = torch.ops.aten.alias.default(relu)
        sum_1: "f32[]" = torch.ops.aten.sum.default(relu)
        alias_2: "f32[1, 3, 3, 3]" = torch.ops.aten.alias.default(relu);  relu = None
        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)
        expand: "f32[1, 3, 3, 3]" = torch.ops.aten.expand.default(ones_like, [1, 3, 3, 3]);  ones_like = None
        alias_3: "f32[1, 3, 3, 3]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        threshold_backward: "f32[1, 3, 3, 3]" = torch.ops.aten.threshold_backward.default(expand, alias_3, 0);  expand = alias_3 = None
        native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(threshold_backward, convolution, arg2_1, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  threshold_backward = convolution = arg2_1 = getitem_1 = getitem_2 = None
        getitem_5: "f32[1, 3, 3, 3]" = native_batch_norm_backward[0]
        getitem_6: "f32[3]" = native_batch_norm_backward[1]
        getitem_7: "f32[3]" = native_batch_norm_backward[2];  native_batch_norm_backward = None
        convolution_backward = torch.ops.aten.convolution_backward.default(getitem_5, arg7_1, arg0_1, [3], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  getitem_5 = arg7_1 = arg0_1 = None
        getitem_8 = convolution_backward[0];  getitem_8 = None
        getitem_9: "f32[3, 1, 1, 1]" = convolution_backward[1]
        getitem_10: "f32[3]" = convolution_backward[2];  convolution_backward = None
        return (getitem_3, getitem_4, add, sum_1, alias_2, getitem_9, getitem_10, getitem_6, getitem_7)
""",
        )

        self.assertExpectedInline(
            str(signature.parameters),
            """['conv.weight', 'conv.bias', 'bn.weight', 'bn.bias']""",
        )
        self.assertExpectedInline(
            str(signature.buffers),
            """['bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked']""",
        )
        self.assertExpectedInline(str(signature.user_inputs), """['arg7_1']""")
        self.assertExpectedInline(
            str(signature.inputs_to_parameters),
            """{'arg0_1': 'conv.weight', 'arg1_1': 'conv.bias', 'arg2_1': 'bn.weight', 'arg3_1': 'bn.bias'}""",
        )
        self.assertExpectedInline(
            str(signature.inputs_to_buffers),
            """{'arg4_1': 'bn.running_mean', 'arg5_1': 'bn.running_var', 'arg6_1': 'bn.num_batches_tracked'}""",
        )
        self.assertExpectedInline(
            str(signature.buffers_to_mutate),
            """{'getitem_3': 'bn.running_mean', 'getitem_4': 'bn.running_var', 'add': 'bn.num_batches_tracked'}""",
        )
        self.assertExpectedInline(
            str(signature.backward_signature.gradients_to_parameters),
            """{'getitem_9': 'conv.weight', 'getitem_10': 'conv.bias', 'getitem_6': 'bn.weight', 'getitem_7': 'bn.bias'}""",
        )
        self.assertExpectedInline(
            str(signature.backward_signature.gradients_to_user_inputs), """{}"""
        )
        self.assertExpectedInline(
            str(signature.backward_signature.loss_output), """getitem_3"""
        )

        # Also check the inference graph
        # Main important thing here is that there are 5 total outputs: 3 total mutated buffers (from batchnorm), 2 user outputs.
        fx_g_inference, signature_inference = aot_export_module(
            mod, [inp], trace_joint=False
        )
        for node in fx_g_inference.graph.nodes:
            node.meta.pop("stack_trace", None)
        self.assertExpectedInline(
            fx_g_inference.print_readable(print_output=False, expanded_def=True),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[3, 1, 1, 1]",  # PlainAOTInput(idx=0)
        arg1_1: "f32[3]",  # PlainAOTInput(idx=1)
        arg2_1: "f32[3]",  # PlainAOTInput(idx=2)
        arg3_1: "f32[3]",  # PlainAOTInput(idx=3)
        arg4_1: "f32[3]",  # PlainAOTInput(idx=4)
        arg5_1: "f32[3]",  # PlainAOTInput(idx=5)
        arg6_1: "i64[]",  # PlainAOTInput(idx=6)
        arg7_1: "f32[1, 1, 3, 3]",  # PlainAOTInput(idx=7)
    ):
        # No stacktrace found for following nodes
        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(arg7_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg7_1 = arg0_1 = arg1_1 = None
        add: "i64[]" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  convolution = arg2_1 = arg3_1 = arg4_1 = arg5_1 = None
        getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
        getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
        getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
        relu: "f32[1, 3, 3, 3]" = torch.ops.aten.relu.default(getitem);  getitem = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(relu)
        detach: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu);  relu = None
        return (
            getitem_3,  # InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=4))
            getitem_4,  # InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=5))
            add,  # InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=6))
            sum_1,  # PlainAOTOutput(idx=0)
            detach,  # PlainAOTOutput(idx=1)
        )
""",
        )
        # Some important characteristics of the exported graph below:
        # 8 arguments: 2 params from conv, 2 params from batchnorm, 2 buffers from 1 batchnorm, 1 user input
        # 9 outputs: 2 mutated buffers (from batchnorm), 2 user outputs and 4 gradients (since there were 4 parameters)

    def test_aot_export_simplified_basic(self):
        # [FUNCTION DOC] Check aot_export_joint_simple forward-only export matches eager without
        # calling convention changes, and that joint export partitioned with default partition
        # gives forward outputs matching eager and backward gradients matching autograd.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Return the product of the inputs and the second input times
            # its detached self.
            return x * y, y * y.detach()

        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)

        f_graph_fw = aot_export_joint_simple(f, [x, y], trace_joint=False)
        out_ref = f(x, y)
        # No calling convention changes necessary to invoke the traced graph
        out_test = f_graph_fw(x, y)
        self.assertEqual(out_ref, out_test)

        # Now test the backward
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)
        x3 = x.detach().clone().requires_grad_(True)
        y3 = y.detach().clone().requires_grad_(True)
        f_graph_joint = aot_export_joint_simple(f, [x, y], trace_joint=True)
        num_fw_outputs = 2
        fw_g, bw_g = default_partition(
            f_graph_joint, [x, y], num_fwd_outputs=num_fw_outputs
        )
        out_ref2 = f(x2, y2)
        fw_outs = fw_g(x3, y3)
        out_test2, activations = fw_outs[:num_fw_outputs], fw_outs[num_fw_outputs:]
        self.assertEqual(out_ref2, out_test2)

        # Test running the traced backward graph with a mocked-up grad_output
        grad_outs = [torch.ones_like(x) for x in out_ref2]
        grads_ref = torch.autograd.grad(out_ref2, [x2, y2], grad_outputs=grad_outs)
        grads_test = bw_g(*activations, *grad_outs)
        for g_ref, g_test in zip(grads_ref, grads_test):
            self.assertEqual(g_ref, g_test)

    def test_aot_export_metadata_mutation_banned(self):
        # [FUNCTION DOC] Check that exporting a function that transposes an input in place
        # raises RuntimeError about a metadata mutation (only the first export call is reached).
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Transpose the input in place and return it doubled in
            # a tuple.
            x.t_()
            return (x * 2,)

        mod = TestMod(fn)
        inp = torch.randn(2, 4)
        with self.assertRaisesRegex(
            RuntimeError, "Found an input that received a metadata mutation"
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            aot_export_module(mod, [inp], trace_joint=False)

    def test_aot_export_forward_mutation_no_buffer_mut(self):
        # [FUNCTION DOC] Check export (forward only) of a module that mutates a user input in
        # place but no buffer: the mutated user input is reported in the signature and returned.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the buffer.
                super().__init__()
                self.buffer1 = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add four to the input in place and return the cosine
                # sum plus the buffer sum.
                x.add_(4)
                return (x.cos().sum() + self.buffer1.sum(),)

        mod = M()
        inp = torch.ones(6, 4)
        gm, sig = aot_export_module(mod, [inp], trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg1_1, 4);  arg1_1 = None
    cos = torch.ops.aten.cos.default(add)
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    sum_2 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add, add_1)""",
        )
        self.assertEqual(sig.user_inputs_to_mutate, {"add": "arg1_1"})

    def test_aot_export_forward_mutation_multiple_mut(self):
        # [FUNCTION DOC] Check export (forward only) with mutation of both a user input and a
        # buffer: the signature reports both mutations.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the buffer.
                super().__init__()
                self.buffer1 = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Add four to y in place, add five to the buffer in
                # place, return a combined sum and the buffer sum.
                y.add_(4)
                self.buffer1.add_(5)
                return (
                    x.cos().sum() + y.sin().sum(),
                    self.buffer1.sum(),
                )

        mod = M()
        inp = [torch.ones(6, 4), torch.zeros(6, 4)]
        gm, sig = aot_export_module(mod, inp, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    add = torch.ops.aten.add.Tensor(arg2_1, 4);  arg2_1 = None
    add_1 = torch.ops.aten.add.Tensor(arg0_1, 5);  arg0_1 = None
    cos = torch.ops.aten.cos.default(arg1_1);  arg1_1 = None
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    sin = torch.ops.aten.sin.default(add)
    sum_2 = torch.ops.aten.sum.default(sin);  sin = None
    add_2 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    sum_3 = torch.ops.aten.sum.default(add_1)
    return (add_1, add, add_2, sum_3)""",
        )
        self.assertEqual(sig.user_inputs_to_mutate, {"add": "arg2_1"})
        self.assertEqual(sig.buffers_to_mutate, {"add_1": "buffer1"})

    def test_aot_export_input_mutation_on_input_requiring_grad_banned(self):
        # [FUNCTION DOC] Check forward-only export of a module that adds in place to an input
        # requiring grad; the graph returns the updated value twice.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds four in place to its input and returns it.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add four in place and return the input in a tuple.
                x.add_(4)
                return (x,)

        mod = M()
        inp = torch.randn(2, requires_grad=True)
        gm, _ = aot_export_module(mod, [inp], trace_joint=False)
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 4), kwargs = {})
    return (add, add)""",
        )

    def test_aot_export_input_mutation_on_parameter_banned(self):
        # [FUNCTION DOC] Check mutation of a parameter in exports: the simple joint export
        # raises RuntimeError for forward only and for joint, while module export allows it and
        # returns the mutated parameter.
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Double the parameter in place and return parameter
            # plus input.
            p.mul_(2)
            return (p + x,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_export_joint_simple does not support input mutations. ViewAndMutationMeta",
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
        with self.assertRaisesRegex(
            RuntimeError,
            "Found a graph input that requires gradients, and received a mutation",
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False)
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg1_1), kwargs = {})
    return (mul, add)""",
        )

    def test_aot_export_synthetic_bases_banned(self):
        # [FUNCTION DOC] Check that exporting with aliased inputs where one is mutated raises
        # RuntimeError about aliased mutated inputs.
        def fn(p, x, y):
            # [FUNCTION DOC] HELPER: Double x in place and return x plus y.
            x.mul_(2)
            return (x + y,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        inp2 = inp.view(-1)
        with self.assertRaisesRegex(
            RuntimeError, "Encountered aliased inputs that are mutated"
        ):
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=True)
            aot_export_module(mod, [inp, inp2], trace_joint=False)

    def test_aot_export_input_dupes_banned(self):
        # [FUNCTION DOC] Check that exporting with the same tensor passed twice and one mutated
        # raises RuntimeError about duplicated mutated inputs.
        def fn(p, x, y):
            # [FUNCTION DOC] HELPER: Double x in place and return x plus y.
            x.mul_(2)
            return (x + y,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(
            RuntimeError, "Encountered duplicated inputs that are mutated in the graph"
        ):
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=True)
            aot_export_module(mod, [inp, inp], trace_joint=False)

    def test_aot_export_multiple_outputs_require_grad_banned(self):
        # [FUNCTION DOC] Check that joint export with a forward output requiring grad other than
        # the loss raises RuntimeError.
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Return a product and its sum.
            out = p * x
            return out, out.sum()

        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(
            RuntimeError,
            "Found an output of the forward that requires gradients, that was not",
        ):
            aot_export_module(mod, [inp], trace_joint=True, output_loss_index=1)

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torch._dynamo.is_dynamo_supported(), "Cond needs dynamo to run"
    )
    def test_aot_export_with_torch_cond(self):
        # [FUNCTION DOC] Not on Windows; requires dynamo support. Export of a module with a
        # conditional (no pre-dispatch); asserts the main graph and both branch graphs.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conditional on the input sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Conditional between two branch functions, returns two
                # shifted copies of the result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: temporary shift with in-place add,
                    # returns the cosine of the input.
                    y = x + 4
                    y.add_(5)
                    return x.cos()

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: temporary shift with in-place add,
                    # returns the sine of the input.
                    y = x + 5
                    y.add_(6)
                    return x.sin()

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x])
                return (a + 3, a + 4)

        inp = torch.randn(3, 4)
        gm, _ = aot_export_module(M(), (inp,), trace_joint=False)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1,));  gt = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    add = torch.ops.aten.add.Tensor(getitem, 3)
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    return (add, add_1)""",
        )

        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    return (cos,)""",
        )

        self.assertExpectedInline(
            gm.false_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    return (sin,)""",
        )

    def test_aot_export_simplified_pytrees_banned(self):
        # [FUNCTION DOC] Check that the simple joint export rejects pytree (list) inputs
        # with RuntimeError.
        def fn(inps):
            # [FUNCTION DOC] HELPER: Add the two elements of the input list in a tuple.
            return (inps[0] + inps[1],)

        inp1 = torch.randn(2)
        inp2 = torch.randn(2)
        inps = [inp1, inp2]
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_export_joint_simple requires individual inputs not to be pytrees",
        ):
            aot_export_joint_simple(fn, [inps], trace_joint=False)
            aot_export_joint_simple(fn, [inps], trace_joint=True)

    def test_aot_export_functionalized_rng_banned(self):
        # [FUNCTION DOC] Check that export under functionalized RNG ops raises RuntimeError that
        # it is unsupported.
        def fn(p, x):
            # [FUNCTION DOC] HELPER: Add parameter and input in a tuple.
            return (p + x,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        with (
            patch("functorch.compile.config.functionalize_rng_ops", True),
            self.assertRaisesRegex(
                RuntimeError,
                "Functionalized RNG is not currently supported in the aot_export",
            ),
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            aot_export_module(mod, [inp], trace_joint=False)

    def test_aot_export_unbacked_arg(self):
        # [FUNCTION DOC] Check export with dynamic shapes of a module using a data dependent
        # size from item; asserts the graph includes local scalar dense and a full op.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no inputs creating a full tensor sized by an
            # extracted scalar.
            def forward(self):
                # [FUNCTION DOC] HELPER: Create a scalar, extract its Python value, return a
                # zero tensor of that size in a tuple.
                full = torch.full((), 11)
                i0 = full.item()
                return (torch.full((i0,), 0),)

        gm, _ = aot_export_module(
            mod=M(), args=(), trace_joint=False, dynamic_shapes=True
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self):
    full = torch.ops.aten.full.default([], 11, device = device(type='cpu'), pin_memory = False)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(full);  full = None
    full_1 = torch.ops.aten.full.default([_local_scalar_dense], 0, device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return (full_1,)""",
        )

    def test_aot_export_input_mutation(self):
        # [FUNCTION DOC] Check _aot_export_function (joint with tangents) on a function that
        # mutates a buffer-like input in place and multiplies it with a grad-requiring input,
        # keeping input mutations; asserts the graph copies the update back into the buffer and
        # returns output and gradient.
        def f(x, buf):
            # [FUNCTION DOC] HELPER: Add one to the buffer in place and return buffer times x.
            buf.add_(1)
            return buf * x

        x = torch.randn(2, requires_grad=True)
        buf = torch.zeros(2, requires_grad=False)

        gm, _, _, _ = _aot_export_function(
            f,
            (x, buf),
            decompositions={},
            num_params_buffers=1,
            no_tangents=False,
            pre_dispatch=False,
            dynamic_shapes=False,
            keep_input_mutations=True,
            kwargs={},
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, primals, tangents):
    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    add = torch.ops.aten.add.Tensor(primals_2, 1)
    mul = torch.ops.aten.mul.Tensor(add, primals_1);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, add);  tangents_1 = None
    copy_ = torch.ops.aten.copy_.default(primals_2, add);  primals_2 = add = copy_ = None
    return pytree.tree_unflatten([mul, mul_1, None], self._out_spec)""",
        )


class TestPartitioning(AOTTestCase):
    # [CLASS DOC] Tests of the partitioners (default and min cut rematerialization) and the
    # graph and calling-convention invariants they must keep, using helper graph extractors.
    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_recompute_partitioning(self):
        # [FUNCTION DOC] Skipped without networkx. Compile a double sine plus input with the min
        # cut rematerialization partition and identity compilers; outputs and gradients must be
        # close to eager, otherwise AssertionError.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Sine of sine of a plus b.
            return torch.sin(torch.sin(a)) + b

        # Reference calculation
        ref_a = torch.rand(10, 10, requires_grad=True)
        ref_b = torch.rand(10, 10, requires_grad=True)
        ref = fn(ref_a, ref_b)
        ref.sum().backward()

        # Compiled function calculation
        res_a = ref_a.detach().clone().requires_grad_(True)
        res_b = ref_b.detach().clone().requires_grad_(True)

        def compile_fn(x, _):
            # [FUNCTION DOC] HELPER: Identity compiler that returns the graph unchanged.
            return x

        compiled_fn = compiled_function(
            fn, compile_fn, compile_fn, min_cut_rematerialization_partition
        )
        res = compiled_fn(res_a, res_b)
        res.sum().backward()
        if not torch.allclose(ref, res, atol=1e-3, rtol=1e-3):
            raise AssertionError("ref and res not allclose")
        if not torch.allclose(ref_a.grad, res_a.grad, atol=1e-3, rtol=1e-3):
            raise AssertionError("ref_a.grad and res_a.grad not allclose")
        if not torch.allclose(ref_b.grad, res_b.grad, atol=1e-3, rtol=1e-3):
            raise AssertionError("ref_b.grad and res_b.grad not allclose")

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_partitioned_backward_nodes_are_backward_tagged(self):
        # [FUNCTION DOC] Skipped without networkx. For the default partition and for a min cut
        # partition forced to recompute sine and cosine nodes, checks every call or
        # get-attribute node in the backward graph is tagged as backward in its metadata,
        # including recomputed nodes that originated in the forward.
        from torch._functorch.partitioners import CheckpointPolicy

        def run(partition_fn):
            # [FUNCTION DOC] HELPER: Compile a sin-cos-sum function with the given partition
            # function, run forward and backward, capturing the backward graph; asserts a graph
            # was captured and all its call or attribute nodes carry the backward tag; returns
            # the graph.
            bw_gm = None

            def fw_compiler(gm, _):
                # [FUNCTION DOC] HELPER: Forward compiler returning the graph forward as a
                # boxed function.
                return make_boxed_func(gm.forward)

            def bw_compiler(gm, _):
                # [FUNCTION DOC] HELPER: Backward compiler that stores the graph in an enclosing
                # variable and returns its forward as a boxed function.
                nonlocal bw_gm
                bw_gm = gm
                return make_boxed_func(gm.forward)

            def fn(x):
                # [FUNCTION DOC] HELPER: Sum of cosine of sine of the input.
                y = torch.sin(x)
                return torch.cos(y).sum()

            x = torch.randn(4, 4, requires_grad=True)
            compiled_fn = aot_function(
                fn,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=partition_fn,
            )

            compiled_fn(x).backward()

            if bw_gm is None:
                raise AssertionError("backward graph was not captured")

            backward_nodes = [
                node
                for node in bw_gm.graph.nodes
                if node.op in ("call_function", "get_attr")
            ]
            self.assertTrue(backward_nodes)
            self.assertTrue(
                all(
                    node.meta.get("autograd_backward", False) for node in backward_nodes
                )
            )
            return bw_gm

        run(default_partition)

        def min_cut_with_forced_recompute(joint_gm, joint_inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Partition function that marks sine and cosine nodes of
            # the joint graph as must-recompute, then delegates to the min cut
            # rematerialization partition.
            for node in joint_gm.graph.nodes:
                if node.op == "call_function" and node.target in {
                    torch.ops.aten.sin.default,
                    torch.ops.aten.cos.default,
                }:
                    node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
            return min_cut_rematerialization_partition(joint_gm, joint_inputs, **kwargs)

        bw_gm = run(min_cut_with_forced_recompute)
        forward_origin_recomputed_nodes = [
            node
            for node in bw_gm.graph.nodes
            if node.op == "call_function"
            and node.target
            in {
                torch.ops.aten.sin.default,
                torch.ops.aten.cos.default,
            }
            and node.meta.get("partitioner_tag") == "is_forward"
        ]
        self.assertTrue(forward_origin_recomputed_nodes)
        self.assertTrue(
            all(
                node.meta.get("autograd_backward", False)
                for node in forward_origin_recomputed_nodes
            )
        )

    def test_meta_tensor_inplace_op(self):
        # [FUNCTION DOC] Check that every non-output node in the forward graph of a Linear plus
        # GELU module has tensor metadata recorded, even with in-place ops during tracing.
        # Following module results in inplace ops while tracing. The test checks
        # that the meta tensor information is stored for inplace ops.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a large linear layer's parameters that applies
            # linear and GELU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create random weight and bias parameters.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(3072, 768, requires_grad=True)
                )
                self.bias = torch.nn.Parameter(torch.randn(3072, requires_grad=True))

            def forward(self, add_4):
                # [FUNCTION DOC] HELPER: Apply linear then GELU.
                linear_4 = torch.nn.functional.linear(
                    add_4, self.weight, bias=self.bias
                )
                gelu = torch.nn.functional.gelu(linear_4)
                return gelu

        def check_meta_tensor(fx_g, _):
            # [FUNCTION DOC] HELPER: Forward compiler that raises AssertionError if any
            # non-output node lacks tensor metadata; returns the graph.
            for node in fx_g.graph.nodes:
                if node.op != "output":
                    if "tensor_meta" not in node.meta:
                        raise AssertionError(
                            f"Expected 'tensor_meta' in node.meta for {node}"
                        )
            return fx_g

        inp0 = torch.randn(16, 128, 768, requires_grad=True)
        inputs = [
            inp0,
        ]
        mod = MockModule().to(device="cpu")
        aot_mod = aot_module(mod, fw_compiler=check_meta_tensor)
        aot_mod(*inputs)

    def test_default_partitioner_getitem(self):
        # [FUNCTION DOC] Check the default partitioner with layer norm: forward has 3 inputs and
        # 6 outputs, backward 6 inputs and 3 outputs.
        mod = nn.LayerNorm([10])

        def f(x, mod_weight, mod_bias):
            # [FUNCTION DOC] HELPER: Layer norm of x over the last dimension with the given
            # weight and bias.
            return torch.nn.functional.layer_norm(
                x, [10], mod_weight, mod_bias, eps=1e-6
            )

        fw_graph, bw_graph = get_fw_bw_graph(
            f,
            [torch.randn(3, 10, requires_grad=True), mod.weight, mod.bias],
            partitioner=default_partition,
        )
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 6))
        self.assertEqual(get_num_ins_outs(bw_graph), (6, 3))

    def test_extract_fwd_bwd_modules_omit_aot_autograd_runtime(self):
        # [FUNCTION DOC] Check splitting a backward graph into input-gradient and
        # weight-gradient graphs: partitions a matmul sum joint graph, renames tangent
        # placeholders so they are not treated as tangents, extracts the first output subgraph,
        # picks saved values, and builds two modules with the runtime omitted; checks no
        # tangent placeholders remain and that the two modules together reproduce the full
        # backward gradients.
        """Test splitting a backward graph into dI/dW subgraphs using
        omit_aot_autograd_runtime and ignore_must_be_in_fw_bw."""
        from torch._functorch.partitioners import is_sym_node

        def f(x, w):
            # [FUNCTION DOC] HELPER: Sum of the matrix product in a tuple.
            return ((x @ w).sum(),)

        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4, requires_grad=True)
        joint = aot_export_joint_simple(f, [x, w], trace_joint=True)
        fw_g, bw_g = default_partition(joint, [x, w], num_fwd_outputs=1)

        # Run reference full backward
        fw_outs = fw_g(x.detach(), w.detach())
        activations = fw_outs[1:]
        grad_out = torch.ones_like(fw_outs[0])
        full_bw = bw_g(*activations, grad_out)

        # Re-partition bw_g into dI (grad_x) and dW (grad_w),
        # mimicking the split_dI_dW pattern.
        # NOTE: placeholder names are load-bearing for the partitioner —
        # _is_primal / _is_tangent classify nodes by checking substrings of
        # node.target (e.g. "tangents" in target → is_tangent). We must create
        # new placeholder nodes (not just rename) so the target is also updated.
        bw_gm = copy.deepcopy(bw_g)
        for p in list(bw_gm.graph.find_nodes(op="placeholder")):
            if p.name.startswith("tangent"):
                with bw_gm.graph.inserting_before(p):
                    new_p = bw_gm.graph.placeholder(f"not_tngnt{p.name[8:]}")
                    new_p.meta.update(p.meta)
                    p.replace_all_uses_with(new_p)
                    bw_gm.graph.erase_node(p)
        bw_gm.recompile()

        num_di = 1  # first output = grad_x
        di_outs, dw_outs, di_descs, _ = _extract_fwd_bwd_outputs(
            bw_gm, num_fwd_outputs=num_di
        )
        args = list(bw_gm.graph.find_nodes(op="placeholder"))
        di_graph = _extract_graph_with_inputs_outputs(
            bw_gm.graph,
            args,
            di_outs,
            di_descs,
            "forward",
            ignore_must_be_in_fw_bw=True,
        )
        di_names = {n.name for n in di_graph.nodes if n.op != "output"}

        saved_values = [
            n
            for n in bw_gm.graph.nodes
            if n.name in di_names
            and not is_sym_node(n)
            and "tensor_meta" in n.meta
            and any(n2.name not in di_names for n2 in n.users)
        ]

        di_mod, dw_mod = _extract_fwd_bwd_modules(
            bw_gm,
            saved_values,
            saved_sym_nodes=[],
            num_fwd_outputs=num_di,
            ignore_must_be_in_fw_bw=True,
            omit_aot_autograd_runtime=True,
        )

        # Verify graph structure
        def ph_names(gm):
            # [FUNCTION DOC] HELPER: List of the placeholder node names of a graph module.
            return [n.name for n in gm.graph.find_nodes(op="placeholder")]

        self.assertFalse(any(n.startswith("tangent") for n in ph_names(di_mod)))
        self.assertFalse(any(n.startswith("tangent") for n in ph_names(dw_mod)))

        # Verify numerics: dI + dW matches full backward
        bw_inputs = [*activations, grad_out]
        di_outs = di_mod(*bw_inputs)
        grad_x = di_outs[:num_di]
        dw_inputs = di_outs[num_di:]
        grad_w = dw_mod(*dw_inputs)
        if not isinstance(grad_w, (list, tuple)):
            grad_w = [grad_w]
        self.assertEqual(grad_x[0], full_bw[0])
        self.assertEqual(grad_w[0], full_bw[1])

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_raise_getitems(self):
        # [FUNCTION DOC] Skipped without networkx. Checks that min cut partitioning moves
        # getitem nodes after split so that the backward graph matches the expected text.
        def f(x):
            # [FUNCTION DOC] HELPER: Split into halves and add sine of the first to cosine of
            # the second.
            y = torch.split(x, x.size(0) // 2, dim=0)
            a = y[0].sin()
            b = y[1].cos()
            return a + b

        _, bw_graph = get_fw_bw_graph(f, [torch.randn(4, 4, requires_grad=True)])

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, primals_1, tangents_1):
    split = torch.ops.aten.split.Tensor(primals_1, 2);  primals_1 = None
    getitem_1 = split[1]
    getitem = split[0];  split = None
    sin_1 = torch.ops.aten.sin.default(getitem_1);  getitem_1 = None
    neg = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, neg);  neg = None
    cos_1 = torch.ops.aten.cos.default(getitem);  getitem = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, cos_1);  tangents_1 = cos_1 = None
    cat = torch.ops.aten.cat.default([mul_1, mul]);  mul_1 = mul = None
    return (cat,)""",
        )

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_custom_partitioner_fn(self):
        # [FUNCTION DOC] Skipped without networkx. A user-defined partitioner object (subclass
        # of the custom partitioner base) is used by the helper; checks it was called and the
        # forward and backward graph text.
        class MyCustomPartitionerFn(CustomPartitionerFn):
            # [CLASS DOC] HELPER: Custom partitioner that records being called and delegates to
            # the min cut rematerialization partition.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the base and clear the called flag.
                super().__init__()
                self.called = False

            def __call__(self, gm, joint_inputs, **kwargs):
                # [FUNCTION DOC] HELPER: Set the called flag and return the min cut partition of
                # the graph.
                self.called = True
                return min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)

            def uuid(self):
                # [FUNCTION DOC] HELPER: Return no identifier.
                return None

        def f(x):
            # [FUNCTION DOC] HELPER: Cosine applied twice.
            return x.cos().cos()

        inp = [torch.randn((4, 4), requires_grad=True)]
        custom_partitioner_fn = MyCustomPartitionerFn()
        fw_graph, bw_graph = get_fw_bw_graph(f, inp, partitioner=custom_partitioner_fn)
        self.assertTrue(custom_partitioner_fn.called)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    cos = torch.ops.aten.cos.default(primals_1)
    cos_1 = torch.ops.aten.cos.default(cos);  cos = None
    return (cos_1, primals_1)""",
        )
        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, primals_1, tangents_1):
    cos = torch.ops.aten.cos.default(primals_1)
    sin = torch.ops.aten.sin.default(cos);  cos = None
    neg = torch.ops.aten.neg.default(sin);  sin = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, neg);  tangents_1 = neg = None
    sin_1 = torch.ops.aten.sin.default(primals_1);  primals_1 = None
    neg_1 = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, neg_1);  mul = neg_1 = None
    return (mul_1,)""",
        )

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_save_shape(self):
        # [FUNCTION DOC] Skipped without networkx. Check min cut partitioner saves symbolic
        # sizes instead of larger tensors for a sum under dynamic shapes, and for a
        # concatenation function with size manipulations saves only symbolic ints beyond the
        # first output.
        def f(x):
            # [FUNCTION DOC] HELPER: Sum over dimension one.
            s = x.sum(dim=1)
            return s

        inp = [torch.ones([10, 10], requires_grad=True)]
        fw_graph, bw_graph = get_fw_bw_graph(f, inp, dynamic=True)
        _, fw_output = get_ins_outs(fw_graph)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))
        self.assertEqual(str(fw_output[0]), "sum_1")
        # make sure we don't do the suboptimal thing of saving the bigger primals input to sum,
        # rather than saving the sizes of the primals input for use in backward expand
        self.assertEqual(str(fw_output[1]), "sym_size_int")
        self.assertEqual(str(fw_output[2]), "sym_size_int_1")

        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
        ]

        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Concatenate an expanded a (sized from sizes of b and c)
            # with b and c.
            # tried to test what happens if we save a size tuple in the graph;
            # turns out we never will due to how we trace, but this is probably
            # still a good test case for various size manipulations
            sb = torch.ops.aten.sym_size(b)
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            return torch.cat([a.expand(a_sz), b, c])

        fw_graph, bw_graph = get_fw_bw_graph(f, inp, dynamic=True)
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 4))
        self.assertEqual(get_num_ins_outs(bw_graph), (4, 3))
        _, outs = get_ins_outs(fw_graph)
        self.assertTrue(all(is_sym_node(n) for n in outs[1:]))

    def test_default_partitioner_output_tensor_shape_tensor(self):
        # [FUNCTION DOC] Check the default partitioner with outputs that include a size
        # (symbolic ints) and input views, under dynamic shapes: counts of graph inputs and
        # outputs, which forward outputs are symbolic nodes, compiled output equality to eager,
        # and that the size output is not returned as a size object.
        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
            torch.randn((10, 1), requires_grad=True),
        ]

        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Build a concatenation with sizes of the inputs, two matrix
            # products, and return the concatenation, a size, an input and the product.
            # Try to force symints intermixed with outputs in the function's returns
            sb = b.size()
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            cat = torch.cat([a.expand(a_sz), b, c])
            mm = torch.mm(cat, d)
            mm2 = torch.mm(
                mm, a.view(mm.size(1), a.size(0))
            )  # this saves 4 new ints for backward. why?
            # and what do i have to do to make it save a tensor for backward?
            return cat, sb, c, mm2

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=default_partition,
            decompositions=default_decompositions,
            dynamic=True,
        )(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]

        # in the fwd graph, 13 outs because:
        # - 5 original outputs (sb is a tuple, gets expanded to 2 symints)
        # - 8 saved outputs for backward: 5 tensors, 3 symints
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 13))
        # in the bwd graph, 10 inputs (grad outs) because:
        # - The fwd graph had 13 outputs
        # - 1 was a view of an input, which gets regenerated outside of the graph
        #   and doesn't participate in the backward
        # - 2 user outs were symints (b.size()), which don't get tangents in the backward
        self.assertEqual(get_num_ins_outs(bw_graph), (10, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            #
            # TODO(whc)- are the saved-tensors/saved-symints correct here?
            # i just made the test pass based on what default partition did
            # Of the 5 original forward outputs, the 4th (c) is an input,
            # which won't show up in the compiled forward graph
            [False, True, True, False, False] + [False] * 4 + [True] * 4,
            [is_sym_node(n) for n in fw_graph_out_nodes],
        )

        real_outs = f(*inp)
        self.assertEqual(compiled_outs, real_outs)
        self.assertTrue(isinstance(real_outs[1], torch.Size))

        # TODO(whc) we should learn to return torch.Sizes
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_output_tensor_shape_tensor(self):
        # [FUNCTION DOC] Skipped without networkx. Same as the previous test with the min cut
        # partitioner; expected counts differ (12 forward outputs and 9 backward inputs).
        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
            torch.randn((10, 1), requires_grad=True),
        ]

        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Build a concatenation with sizes of the inputs, two matrix
            # products, and return the concatenation, a size, an input and the product.
            # Try to force symints intermixed with outputs in the function's returns
            sb = b.size()
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            cat = torch.cat([a.expand(a_sz), b, c])
            mm = torch.mm(cat, d)
            mm2 = torch.mm(
                mm, a.view(mm.size(1), a.size(0))
            )  # this saves 4 new ints for backward. why?
            # and what do i have to do to make it save a tensor for backward?
            return cat, sb, c, mm2

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=min_cut_rematerialization_partition,
            decompositions=default_decompositions,
            dynamic=True,
        )(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]

        self.assertEqual(get_num_ins_outs(fw_graph), (4, 12))
        self.assertEqual(get_num_ins_outs(bw_graph), (9, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            # then 4 tensors (transposes of matrices used for mm) are saved
            # finally 3 symints are saved
            [False, True, True, False, False] + [False] * 4 + [True] * 3,
            [is_sym_node(n) for n in fw_graph_out_nodes],
        )

        real_outs = f(*inp)
        self.assertEqual(compiled_outs, real_outs)
        self.assertTrue(isinstance(real_outs[1], torch.Size))

        # TODO(whc) we should learn to return torch.Sizes
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner(self):
        # [FUNCTION DOC] Skipped without networkx. Check input and output counts of forward and
        # backward graphs for a triple cosine and a sum-then-double-cosine function under the
        # min cut partitioner.
        def f(x):
            # [FUNCTION DOC] HELPER: Cosine applied three times.
            return x.cos().cos().cos()

        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)])
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 1))

        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Sum four inputs then apply cosine twice.
            x = a + b + c + d
            return x.cos().cos()

        fw_graph, bw_graph = get_fw_bw_graph(
            f, [torch.randn(3, requires_grad=True) for _ in range(4)]
        )
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 4))

    def test_contiguous(self):
        # [FUNCTION DOC] Simulates a transpose followed by a view happening in the backward
        # pass; computes a gradient through the compiled function and expects no error.
        # The test simulates the condition where transpose followed by view
        # happens in the backward pass.
        # https://discuss.pytorch.org/t/error-on-transpose-and-view/434
        def f(x):
            # [FUNCTION DOC] HELPER: View as two by three and transpose.
            return x.view(2, 3).t()

        inp = torch.randn(6, requires_grad=True)
        out = aot_function(f, nop)(inp)
        torch.autograd.grad(out, inp, torch.randn(3, 2))

    def test_preserve_random(self):
        # [FUNCTION DOC] Check random number generation (dropout) under aot_function with the
        # same seed gives the same result as eager; raises AssertionError otherwise.
        def fn(x):
            # [FUNCTION DOC] HELPER: Dropout plus the input.
            return torch.nn.functional.dropout(x, 0.5) + x

        x = torch.randn(4)

        torch.manual_seed(0)
        ref = fn(x)

        torch.manual_seed(0)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        if not torch.allclose(ref, res):
            raise AssertionError("ref and res not allclose")

    # https://github.com/pytorch/pytorch/issues/110666
    def test_generate_gives_inference_graph(self):
        # [FUNCTION DOC] Regression for issue 110666: a function that multiplies under no_grad
        # yields an inference graph even when the input requires grad.
        # We expect this to give an inference graph
        def generate(x):
            # [FUNCTION DOC] HELPER: Square the input under no_grad.
            with torch.no_grad():
                return torch.mul(x, x)

        inference_graph_cell = [None]
        inference_compiler = make_boxed_compiler(
            partial(extract_graph, graph_cell=inference_graph_cell)
        )
        aot_fn = aot_function(generate, nop, inference_compiler=inference_compiler)
        # Even though x requires grad, we should still get an inference graph
        x = torch.randn(4, requires_grad=True)
        aot_fn(x)
        self.assertTrue(inference_graph_cell[0] is not None)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_autocast(self):
        # [FUNCTION DOC] Requires CUDA and torchvision. Runs resnet18 through memory efficient
        # fusion in training mode under autocast and backpropagates.
        mod = torchvision.models.resnet18().cuda()
        mod.train()

        x = torch.randn(16, 3, 32, 32, device="cuda")
        aot_mod = memory_efficient_fusion(mod)

        # Ensure that AOT Autograd works with AMP
        with torch.cuda.amp.autocast(True):
            res = aot_mod(x)
        res.sum().backward()

    def test_quantize_activation_duplicate_nodes(self):
        # [FUNCTION DOC] Checks activation quantization graph passes handle duplicate output
        # nodes: builds a forward graph with a node repeated in the outputs and runs the forward
        # quantization pass, then verifies quantized outputs are named by position, share a
        # scale node, and the backward pass handles matching placeholders.
        """Test both quantize_activation_fw and quantize_activation_bw handle duplicate nodes correctly"""
        import torch.fx as fx
        from torch._functorch.partitioners import (
            quantize_activation_bw,
            quantize_activation_fw,
        )
        from torch._subclasses.fake_tensor import extract_tensor_metadata

        # Mock the inductor config
        with patch.dict(
            "torch._inductor.config.post_grad_fusion_options",
            {
                "activation_quantization_aten_pass": {
                    "allowed_dtypes": "torch.bfloat16",
                    "size_in_mb": 1,
                    "use_scaling": True,
                    "exclude_primals": False,
                    "skip_dynamo_guards": True,
                    "quantize_dynamic_shape": False,
                    "quant_type": "torch.float16",  # float8_e5m2 must be GPU
                }
            },
        ):
            # Test Forward Graph with duplicate nodes
            fwd_graph = fx.Graph()

            # Create input nodes
            x = fwd_graph.placeholder("x")
            x.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            x.meta["tensor_meta"] = extract_tensor_metadata(x.meta["val"])

            y = fwd_graph.placeholder("y")
            y.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            y.meta["tensor_meta"] = extract_tensor_metadata(y.meta["val"])

            # Create a computation node that will be duplicated in outputs
            mul_node = fwd_graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
            mul_node.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            mul_node.meta["tensor_meta"] = extract_tensor_metadata(mul_node.meta["val"])
            mul_node.meta["saved_for_quantization"] = True

            # Create another node
            add_node = fwd_graph.call_function(torch.ops.aten.add.Tensor, (x, y))
            add_node.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            add_node.meta["tensor_meta"] = extract_tensor_metadata(add_node.meta["val"])

            # Create output with DUPLICATE nodes - mul_node appears at positions 0 and 2
            fwd_graph.output((mul_node, add_node, mul_node))

            # Test the forward quantization function
            quantize_activation_fw(fwd_graph)

            # Get the forward output node
            fwd_output_node = fwd_graph.find_nodes(op="output")[0]
            fwd_output_args = fwd_output_node.args[0]

            # Verify forward graph has the correct structure
            self.assertGreaterEqual(
                len(fwd_output_args), 3, "Should have at least the original 3 outputs"
            )

            # Check that positions 0 and 2 reuse the same quantized node
            pos_0_node = fwd_output_args[0]
            pos_2_node = fwd_output_args[2]

            # Both should be quantized nodes
            self.assertTrue(
                pos_0_node.name.startswith("fp8_quant_"),
                lambda msg: f"{msg}\nPosition 0 should be quantized node, got: {pos_0_node.name}",
            )
            self.assertTrue(
                pos_2_node.name.startswith("fp8_quant_"),
                lambda msg: f"{msg}\nPosition 2 should be quantized node, got: {pos_2_node.name}",
            )

            # The shared quantized node should have the first occurrence position in its name
            self.assertIn(
                "_pos_0",
                pos_0_node.name,
                lambda msg: f"{msg}\nShared quantized node should have '_pos_0' in name: {pos_0_node.name}",
            )
            self.assertIn(
                "_pos_2",
                pos_2_node.name,
                lambda msg: f"{msg}\nShared quantized node should have '_pos_2' in name: {pos_2_node.name}",
            )
            # Find scale nodes in the forward output
            fwd_scale_nodes = [
                node for node in fwd_output_args if "fp8_scale_" in node.name
            ]
            self.assertEqual(
                len(fwd_scale_nodes),
                2,
                "Should have exactly 2 scale node (shared for both quantized instances)",
            )

            # Test Backward Graph with duplicate nodes
            bwd_graph = fx.Graph()

            # Create backward placeholders corresponding to forward outputs
            quant_input1 = bwd_graph.placeholder("fp8_quant_pos_0_mul_tensor")
            quant_input1.meta["val"] = torch.randn(100, 100, dtype=torch.float16)
            quant_input1.meta["tensor_meta"] = extract_tensor_metadata(
                quant_input1.meta["val"]
            )
            quant_input1.meta["saved_for_quantization"] = True
            quant_input1.meta["dequant_type"] = torch.bfloat16

            add_input = bwd_graph.placeholder("add")
            add_input.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            add_input.meta["tensor_meta"] = extract_tensor_metadata(
                add_input.meta["val"]
            )

            quant_input2 = bwd_graph.placeholder("fp8_quant_pos_2_mul_tensor")
            quant_input2.meta["val"] = torch.randn(100, 100, dtype=torch.float16)
            quant_input2.meta["tensor_meta"] = extract_tensor_metadata(
                quant_input2.meta["val"]
            )
            quant_input2.meta["saved_for_quantization"] = True
            quant_input2.meta["dequant_type"] = torch.bfloat16

            # Add scale node (would come from forward)
            scale_input = bwd_graph.placeholder("fp8_scale_pos_0_mul_tensor")
            scale_input.meta["val"] = torch.randn(100, 100, dtype=torch.float32)
            scale_input.meta["tensor_meta"] = extract_tensor_metadata(
                scale_input.meta["val"]
            )

            scale_input2 = bwd_graph.placeholder("fp8_scale_pos_2_mul_tensor")
            scale_input2.meta["val"] = torch.randn(100, 100, dtype=torch.float32)
            scale_input2.meta["tensor_meta"] = extract_tensor_metadata(
                scale_input.meta["val"]
            )
            # Create some backward computation using both quantized inputs
            grad_output1 = bwd_graph.placeholder("tangents_1")
            grad_output1.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            grad_output1.meta["tensor_meta"] = extract_tensor_metadata(
                grad_output1.meta["val"]
            )

            grad_output2 = bwd_graph.placeholder("tangents_2")
            grad_output2.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            grad_output2.meta["tensor_meta"] = extract_tensor_metadata(
                grad_output2.meta["val"]
            )

            # Create backward operations using the quantized inputs
            mul_bwd1 = bwd_graph.call_function(
                torch.ops.aten.mul.Tensor, (quant_input1, grad_output1)
            )
            mul_bwd1.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            mul_bwd1.meta["tensor_meta"] = extract_tensor_metadata(mul_bwd1.meta["val"])

            mul_bwd2 = bwd_graph.call_function(
                torch.ops.aten.mul.Tensor, (quant_input2, grad_output2)
            )
            mul_bwd2.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            mul_bwd2.meta["tensor_meta"] = extract_tensor_metadata(mul_bwd2.meta["val"])

            # Create output
            bwd_graph.output((mul_bwd1, mul_bwd2))

            # Test the backward quantization function
            quantize_activation_bw(bwd_graph)

            # Verify backward graph processing
            bwd_placeholders = list(bwd_graph.find_nodes(op="placeholder"))
            quantized_placeholders = [
                p for p in bwd_placeholders if "fp8_quant_" in p.name
            ]
            scale_placeholders = [p for p in bwd_placeholders if "fp8_scale_" in p.name]

            # Should have processed the quantized placeholders
            self.assertGreater(
                len(quantized_placeholders), 0, "Should have quantized placeholders"
            )
            self.assertGreater(
                len(scale_placeholders), 0, "Should have scale placeholders"
            )

            # Check that dequantization operations were added
            dequant_operations = [
                node
                for node in bwd_graph.nodes
                if node.op == "call_function"
                and "convert_element_type" in str(node.target)
            ]

            # Should have dequantization operations for each quantized input that was processed
            self.assertGreater(
                len(dequant_operations),
                0,
                "Should have dequantization operations in backward graph",
            )

            # Verify the backward graph users were properly updated
            for quant_placeholder in quantized_placeholders:
                # The quantized placeholder should not be directly used in final operations
                # (it should be replaced by dequantized versions)
                direct_users = [
                    user
                    for user in quant_placeholder.users
                    if user.op == "call_function" and "mul" in str(user.target)
                ]
                # Direct usage should be minimal (only for dequantization chain)
                self.assertLessEqual(
                    len(direct_users),
                    1,
                    lambda msg: f"{msg}\nQuantized placeholder {quant_placeholder.name} should have minimal direct users",
                )

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_unbounded_error_message(self):
        # [FUNCTION DOC] Skipped without networkx. Tests the helper that finds a path of
        # infinite capacity edges in a flow graph and returns it as a list of triples (from
        # node, to node, reason). Checks a three-edge chain is found with its reasons, a finite
        # capacity edge breaks the path (returns none), and a data dependency edge reason is
        # included in a four-edge path.
        """Test that NetworkXUnbounded errors produce user-friendly error messages."""
        import math

        import networkx as nx

        from torch._functorch.partitioners import _find_infinite_capacity_path

        # Test 1: Verify _find_infinite_capacity_path finds a path with edge reasons
        nx_graph = nx.DiGraph()
        # Create a simple graph with an infinite capacity path and reasons
        nx_graph.add_edge(
            "source", "node1_in", capacity=math.inf, reason="cannot recompute: banned"
        )
        nx_graph.add_edge(
            "node1_in",
            "node1_out",
            capacity=math.inf,
            reason="cannot save: non-tensor output",
        )
        nx_graph.add_edge(
            "node1_out",
            "sink",
            capacity=math.inf,
            reason="must be computed in backward: required for gradient",
        )

        path = _find_infinite_capacity_path(nx_graph)
        self.assertIsNotNone(path)
        # path is a list of (from_node, to_node, reason) tuples
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0][0], "source")  # first edge starts from source
        self.assertEqual(path[0][1], "node1_in")
        self.assertIn("cannot recompute", path[0][2])
        self.assertEqual(path[-1][1], "sink")  # last edge ends at sink
        self.assertIn("must be computed in backward", path[-1][2])

        # Test 2: Verify path not found when there's no infinite capacity path
        nx_graph2 = nx.DiGraph()
        nx_graph2.add_edge(
            "source", "node1_in", capacity=math.inf, reason="cannot recompute: banned"
        )
        nx_graph2.add_edge(
            "node1_in", "node1_out", capacity=1.0
        )  # Finite capacity breaks the chain
        nx_graph2.add_edge(
            "node1_out",
            "sink",
            capacity=math.inf,
            reason="must be computed in backward",
        )

        path2 = _find_infinite_capacity_path(nx_graph2)
        self.assertIsNone(path2)

        # Test 3: Verify data dependency edges have reasons
        nx_graph3 = nx.DiGraph()
        nx_graph3.add_edge(
            "source", "node1_in", capacity=math.inf, reason="cannot recompute: primal"
        )
        nx_graph3.add_edge(
            "node1_in", "node1_out", capacity=math.inf, reason="cannot save: view op"
        )
        nx_graph3.add_edge(
            "node1_out", "node2_in", capacity=math.inf, reason="data dependency"
        )
        nx_graph3.add_edge(
            "node2_in",
            "sink",
            capacity=math.inf,
            reason="must be computed in backward: required for gradient",
        )

        path3 = _find_infinite_capacity_path(nx_graph3)
        self.assertIsNotNone(path3)
        self.assertEqual(len(path3), 4)
        # Check that data dependency edge is captured
        data_dep_edge = next(
            (e for e in path3 if e[0] == "node1_out" and e[1] == "node2_in"), None
        )
        self.assertIsNotNone(data_dep_edge)
        self.assertEqual(data_dep_edge[2], "data dependency")

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_getitem_of_banned_multi_output(self):
        # [FUNCTION DOC] Skipped without networkx. Regression test that a getitem from a
        # must-save multi-output node (variance and mean op) returned directly as a gradient
        # output does not make the min cut flow problem unbounded; uses selective checkpointing
        # and a custom Function under the aot_eager decomp partition backend.
        """Test that getitem from a banned multi-output node doesn't cause NetworkXUnbounded.

        Uses selective checkpoint to mark var_mean as MUST_SAVE (banning recomputation),
        with a custom autograd.Function that returns the mean (getitem from var_mean)
        directly as the gradient output.

        This test is doing weird things, because ordinarily the returned grad_output will
        always be a function of the input tangent, preventing it from being returned directly
        as a forward output. It is still important to test this case because it can actually
        happen if you use subclasses.
        """
        from torch.utils.checkpoint import (
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        class GetitemAsGradOutput(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function that computes variance and mean of a stack
            # of the input, saves the mean, and in backward returns the saved mean directly as
            # the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Compute variance and mean over a stacked input, save
                # the mean, return the variance.
                var, mean = torch.var_mean(torch.stack([x, x]), dim=0)
                ctx.save_for_backward(mean)
                return var

            @staticmethod
            def backward(ctx, grad_var):
                # [FUNCTION DOC] HELPER: Return the saved mean as the gradient, ignoring the
                # incoming gradient.
                return ctx.saved_tensors[0]

        context_fn = partial(
            create_selective_checkpoint_contexts,
            [torch.ops.aten.var_mean.correction],
        )

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Run the custom Function under non-reentrant checkpointing
            # with the selective context.
            return checkpoint(
                GetitemAsGradOutput.apply,
                x,
                use_reentrant=False,
                context_fn=context_fn,
            )

        x = torch.randn(4, requires_grad=True)
        fn(x).sum().backward()

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_multi_output_must_save_budget(self):
        # [FUNCTION DOC] Skipped without networkx. With a low fractional activation memory
        # budget and a joint pass marking the top-k op as must-save, the min cut partition must
        # not crash, must honor must-save (the op does not appear in the backward graph), and
        # gradients must match eager.
        """MUST_SAVE on a multi-output (tuple) producer under a fractional
        activation_memory_budget must not crash and must be honored.

        With a low budget, choose_saved_values_set can reach the branch where
        there is nothing left for the knapsack to trade off (every banned node
        is must-save). That branch used to return inputs + must_save_nodes, which
        leaked the non-saveable tuple producer node into saved_values and tripped
        a downstream "expected all ... to be Tensors" assertion. It must instead
        return a real min-cut, saving the producer's getitem tensors.
        """
        import torch._functorch.config as functorch_config
        from torch.utils.checkpoint import CheckpointPolicy

        producer = torch.ops.aten.topk.default

        def f(x):
            # [FUNCTION DOC] HELPER: Top-k of the sine of the input, return the sum of cosine of
            # values times sine of float indices.
            vals, idx = torch.topk(x.sin(), k=4, dim=1)
            return (vals.cos() * idx.float().sin()).sum()

        def mark_producer_must_save(gm, joint_inputs):
            # [FUNCTION DOC] HELPER: Joint custom pass marking nodes calling the top-k op as
            # must-save; returns the graph module.
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target is producer:
                    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            return gm

        bw_graph = {}

        def bw_compiler(gm, _):
            # [FUNCTION DOC] HELPER: Backward compiler that stores the graph in a dictionary and
            # returns it.
            bw_graph["gm"] = gm
            return gm

        compiled = aot_function(
            f,
            fw_compiler=lambda gm, _: gm,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        x = torch.randn(64, 64, requires_grad=True)
        with functorch_config.patch(
            activation_memory_budget=0.1,
            joint_custom_pass=mark_producer_must_save,
        ):
            compiled(x).backward()

        x_ref = x.detach().clone().requires_grad_()
        f(x_ref).backward()
        self.assertEqual(x.grad, x_ref.grad)

        # MUST_SAVE on the producer is honored: it is saved, not recomputed,
        # so the producer op does not appear in the backward graph.
        self.assertNotIn("topk", bw_graph["gm"].code)

    def test_disable_functionalization_ignores_effect_token_metadata(self):
        # [FUNCTION DOC] Check disabling functionalization with a function taking a list input
        # and calling matrix inverse does not break on effect token metadata: output and
        # gradient equal eager.
        def fn(args):
            # [FUNCTION DOC] HELPER: Unpack the single element of the list and return its
            # matrix inverse.
            (x,) = args
            return torch.linalg.inv(x)

        compiled_fn = compiled_function(
            fn,
            nop,
            nop,
            partition_fn=default_partition,
            keep_inference_input_mutations=True,
            disable_functionalization=True,
        )

        x = torch.tensor(
            [[2.0, 0.1, -0.2], [0.3, 1.7, 0.4], [-0.1, 0.2, 2.1]]
        ).requires_grad_()
        eager_x = x.detach().clone().requires_grad_()
        compiled_x = x.detach().clone().requires_grad_()

        eager_out = fn([eager_x])
        (eager_grad,) = torch.autograd.grad(eager_out.sum(), eager_x)

        compiled_out = compiled_fn([compiled_x])
        (compiled_grad,) = torch.autograd.grad(compiled_out.sum(), compiled_x)

        self.assertEqual(compiled_out, eager_out)
        self.assertEqual(compiled_grad, eager_grad)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_force_save_effectful_ops(self):
        # [FUNCTION DOC] CUDA only. Registers a custom op as ordered effectful, captures the
        # inductor graph via a custom backend, applies the force-save pass for effectful ops,
        # and checks the one effect node has all its getitem users (two) marked must-save; the
        # effect registration is destroyed afterwards.
        """Test that effectful op outputs are saved, not recomputed.

        This test traces a function with a with_effects node and verifies
        that the getitem outputs are marked MUST_SAVE.
        """
        from torch._functorch.partitioners import (
            CheckpointPolicy,
            force_save_effectful_ops,
        )
        from torch._higher_order_ops.effects import _register_effectful_op, with_effects
        from torch._library.effects import EffectType

        @torch.library.custom_op("test::effectful_op", mutates_args=())
        def effectful_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op that doubles its input.
            return x * 2

        @effectful_op.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of the custom op returning an empty
            # tensor like the input.
            return torch.empty_like(x)

        handle = _register_effectful_op(effectful_op, EffectType.ORDERED)

        try:
            gm = None

            def graph_capture_backend(graph_module, example_inputs):
                # [FUNCTION DOC] HELPER: Backend that runs inductor compile with an inner
                # compile wrapper recording the graph module in an enclosing variable.
                """Custom backend that captures the graph before passing to inductor."""
                nonlocal gm

                from torch._inductor.compile_fx import compile_fx, compile_fx_inner

                def log_and_compile(graph_module, example_inputs, **kwargs):
                    # [FUNCTION DOC] HELPER: Record the graph module then delegate to the
                    # inductor inner compile.
                    nonlocal gm
                    gm = graph_module
                    return compile_fx_inner(graph_module, example_inputs, **kwargs)

                return compile_fx(
                    graph_module, example_inputs, inner_compile=log_and_compile
                )

            @torch.compile(backend=graph_capture_backend)
            def fn(x, weight):
                # [FUNCTION DOC] HELPER: Apply the effectful op to x and return the sum times
                # the weight.
                a = torch.ops.test.effectful_op(x)
                return a.sum() * weight

            x = torch.randn(4, 4)
            weight = torch.randn(4, 4)
            fn(x, weight)

            force_save_effectful_ops(gm)

            with_effects_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "call_function" and n.target == with_effects
            ]
            self.assertEqual(
                len(with_effects_nodes), 1, "should have one with_effects node"
            )

            getitem_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target == operator.getitem
                and n.args[0] == with_effects_nodes[0]
            ]

            def is_must_save(node):
                # [FUNCTION DOC] HELPER: True if the node's recompute policy equals must-save.
                return node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE

            must_save_count = sum(1 for n in getitem_nodes if is_must_save(n))
            self.assertEqual(
                must_save_count,
                2,
                lambda msg: f"{msg}\n2 items should be MUST_SAVE, got {must_save_count}",
            )
            self.assertEqual(
                len(getitem_nodes),
                must_save_count,
                "all getitem nodes should be MUST_SAVE",
            )
        finally:
            handle.destroy()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_force_save_effectful_ops_nested_tuple(self):
        # [FUNCTION DOC] CUDA only. Like the previous test with an effectful op returning two
        # tensors: all three tensor getitems of the effect node (token excluded as per the
        # count) are marked must-save after the pass.
        """Test that effectful ops returning tuples have all tensor outputs marked MUST_SAVE.

        This test creates a custom op that returns a tuple, registers it as effectful,
        and verifies that all tensor getitems are marked MUST_SAVE after tracing.
        """
        from torch._functorch.partitioners import (
            CheckpointPolicy,
            force_save_effectful_ops,
        )
        from torch._higher_order_ops.effects import _register_effectful_op, with_effects
        from torch._library.effects import EffectType

        @torch.library.custom_op("test::effectful_tuple_op", mutates_args=())
        def effectful_tuple_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Custom op returning a pair, the input doubled and the input
            # plus one.
            return x * 2, x + 1

        @effectful_tuple_op.register_fake
        def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Fake implementation returning two empty tensors like
            # the input.
            return torch.empty_like(x), torch.empty_like(x)

        handle = _register_effectful_op(effectful_tuple_op, EffectType.ORDERED)

        try:
            gm = None

            def graph_capture_backend(graph_module, example_inputs):
                # [FUNCTION DOC] HELPER: Backend that runs inductor compile with an inner
                # compile wrapper recording the graph module in an enclosing variable.
                """Custom backend that captures the graph before passing to inductor."""
                nonlocal gm

                from torch._inductor.compile_fx import compile_fx, compile_fx_inner

                def log_and_compile(graph_module, example_inputs, **kwargs):
                    # [FUNCTION DOC] HELPER: Record the graph module then delegate to the
                    # inductor inner compile.
                    nonlocal gm
                    gm = graph_module
                    return compile_fx_inner(graph_module, example_inputs, **kwargs)

                return compile_fx(
                    graph_module, example_inputs, inner_compile=log_and_compile
                )

            @torch.compile(backend=graph_capture_backend)
            def fn(x):
                # [FUNCTION DOC] HELPER: Call the effectful tuple op and return the sum of
                # both outputs.
                a, b = torch.ops.test.effectful_tuple_op(x)
                return a.sum() + b.sum()

            x = torch.randn(4, 4)
            fn(x)

            with_effects_node = None
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target == with_effects:
                    with_effects_node = node
                    break
            self.assertIsNotNone(with_effects_node, "should have a with_effects node")

            getitem_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target == operator.getitem
                and n.args[0] == with_effects_node
            ]

            force_save_effectful_ops(gm)

            def is_must_save(node):
                # [FUNCTION DOC] HELPER: True if the node's recompute policy equals must-save.
                return node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE

            tensor_getitem_count = 0
            must_save_count = 0
            for node in getitem_nodes:
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    tensor_getitem_count += 1
                    if is_must_save(node):
                        must_save_count += 1

            self.assertEqual(
                tensor_getitem_count,
                3,
                lambda msg: f"{msg}\nexpected 3 getitems, got {tensor_getitem_count}",
            )
            self.assertEqual(
                must_save_count,
                tensor_getitem_count,
                lambda msg: f"{msg}\nall {tensor_getitem_count} tensor getitems should be MUST_SAVE, got {must_save_count}",
            )
        finally:
            handle.destroy()

    def test_static_input_indices_with_effect_tokens(self):
        # [FUNCTION DOC] Check that static input indices in graph metadata are shifted by the
        # number of effect tokens when tokens are prepended to arguments: builds empty metadata
        # with one token and static indices one and two, runs the token handling function, and
        # verifies arguments grew by the token count and indices were offset.
        """Test that static_input_indices are correctly offset when effect tokens are prepended."""
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._functorch._aot_autograd.graph_capture_wrappers import (
            handle_effect_tokens_fn,
        )
        from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        @torch.library.custom_op("test::effectful_static_idx_op", mutates_args=())
        def effectful_static_idx_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op returning a clone of its input.
            return x.clone()

        @effectful_static_idx_op.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like
            # the input.
            return torch.empty_like(x)

        handle = _register_effectful_op(effectful_static_idx_op, EffectType.ORDERED)

        try:
            meta = ViewAndMutationMeta(
                input_info=[],
                output_info=[],
                num_intermediate_bases=0,
                keep_input_mutations=False,
                traced_tangents=[],
                subclass_inp_meta=[],
                subclass_fw_graph_out_meta=[],
                subclass_tangent_meta=[],
                traced_tangents_descs=[],
            )
            meta.tokens = {EffectType.ORDERED: torch.tensor([])}
            # Simulate: params at indices 1, 2 before token insertion
            meta.static_input_indices = [1, 2]

            args = [torch.randn(4, 3), torch.randn(2, 3), torch.randn(2)]
            args_descs = [PlainAOTInput(i) for i in range(3)]

            _, new_args, _ = handle_effect_tokens_fn(
                lambda *a: (torch.randn(1), []),
                args,
                args_descs,
                meta=meta,
                trace_joint=False,
            )

            num_tokens = len(meta.tokens)
            self.assertGreater(num_tokens, 0)
            # After prepending tokens, args grow and indices must be offset
            self.assertEqual(len(new_args), len(args) + num_tokens)
            self.assertEqual(
                meta.static_input_indices,
                [1 + num_tokens, 2 + num_tokens],
            )
        finally:
            handle.destroy()

    def _make_effectful_op(self, name):
        # [FUNCTION DOC] HELPER: Define a custom op with the given name in the test namespace
        # that clones its input, register a fake implementation, and return the op.
        @torch.library.custom_op(f"test::{name}", mutates_args=())
        def op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op that clones its input (the op name is supplied by
            # the helper).
            return x.clone()

        @op.register_fake
        def _(x):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like
            # the input.
            return torch.empty_like(x)

        return op

    def _make_wrapper_via_post_compile(self, compiled_fn, num_tokens):
        # [FUNCTION DOC] HELPER: Wrap the compiled function using the effect token runtime
        # wrapper with a mock metadata holding the given number of tokens and an empty config;
        # returns the wrapped function.
        from torch._functorch._aot_autograd.runtime_wrappers import EffectTokensWrapper

        mock_meta = SimpleNamespace(tokens=dict.fromkeys(range(num_tokens)))
        mock_aot_config = SimpleNamespace()
        return EffectTokensWrapper().post_compile(
            compiled_fn, mock_aot_config, runtime_metadata=mock_meta
        )

    def test_effect_tokens_codegen_single(self):
        # [FUNCTION DOC] Check the effect token wrapper source is generated exactly once for one
        # effectful op (captured codegen) and that the function still returns the input; the
        # source mentions none and slicing off the first output.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        op = self._make_effectful_op("codegen_single_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:
            with capture_codegen_source("effect_tokens_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Call the effectful op on x.
                    return torch.ops.test.codegen_single_effect(x)

                x = torch.randn(4)
                out = f(x)

            self.assertEqual(out, x)
            self.assertEqual(
                len(captured),
                1,
                "Expected effect_tokens_wrapper codegen artifact to be emitted",
            )
            source = captured[0]
            self.assertIn("None", source)
            self.assertIn("outs[1:]", source)
        finally:
            handle.destroy()

    def test_effect_tokens_no_codegen_when_zero(self):
        # [FUNCTION DOC] Check no wrapper codegen artifact is emitted when the function has no
        # effect tokens.
        with capture_codegen_source("effect_tokens_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4)
            out = f(x)

        self.assertEqual(out, x * 2)
        self.assertEqual(
            len(captured),
            0,
            "No codegen artifact should be emitted when there are no effect tokens",
        )

    def test_effect_tokens_codegen_correctness(self):
        # [FUNCTION DOC] Check correctness of the compiled function with an effectful op
        # followed by an add.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        op = self._make_effectful_op("codegen_correctness_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply the effectful op then add one.
                y = torch.ops.test.codegen_correctness_effect(x)
                return y + 1

            x = torch.randn(3, 3)
            out = f(x)
            self.assertEqual(out, x + 1)
        finally:
            handle.destroy()

    def test_effect_tokens_codegen_with_mutation(self):
        # [FUNCTION DOC] Check effect tokens with an input mutation: result is the original
        # value and the input is incremented.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        op = self._make_effectful_op("codegen_mutation_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply the effectful op, add one to the input in place,
                # return the op result.
                y = torch.ops.test.codegen_mutation_effect(x)
                x.add_(1)
                return y

            x = torch.randn(4)
            x_ref = x.clone()
            out = f(x)

            self.assertEqual(out, x_ref)
            self.assertEqual(x, x_ref + 1)
        finally:
            handle.destroy()

    def test_effect_tokens_codegen_multiple_outputs(self):
        # [FUNCTION DOC] Check effect tokens with several outputs (op result, double
        # and increment).
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        op = self._make_effectful_op("codegen_multi_out_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Return the effectful op result, input doubled and input
                # plus one.
                y = torch.ops.test.codegen_multi_out_effect(x)
                return y, x * 2, x + 1

            x = torch.randn(4)
            out1, out2, out3 = f(x)

            self.assertEqual(out1, x)
            self.assertEqual(out2, x * 2)
            self.assertEqual(out3, x + 1)
        finally:
            handle.destroy()

    def test_effect_tokens_codegen_training_path(self):
        # [FUNCTION DOC] Check effect tokens on a training path with a registered autograd rule;
        # a compiled function applying the effectful op and doubling; after forward and backward
        # the output equals double the input and the gradient is two everywhere.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        op = self._make_effectful_op("codegen_training_effect")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward returning the gradient unchanged.
            return grad

        op.register_autograd(backward, setup_context=setup_context)
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply the effectful op and double the result.
                y = torch.ops.test.codegen_training_effect(x)
                return y * 2

            x = torch.randn(3, requires_grad=True)
            out = f(x)
            out.sum().backward()

            self.assertEqual(out, x * 2)
            self.assertEqual(x.grad, torch.full((3,), 2.0))
        finally:
            handle.destroy()

    def test_effect_tokens_codegen_multiple_tokens(self):
        # [FUNCTION DOC] Unit test of the effect token wrapper with two tokens: the wrapper
        # prepends two empty slots to the argument list, clears the caller's list, calls the
        # wrapped function once, and returns the outputs with the first two removed.
        call_log = []

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function that logs a copy of its argument
            # list and returns a list of indices.
            call_log.append(list(args))
            return list(range(len(args)))

        wrapper = self._make_wrapper_via_post_compile(mock_compiled_fn, num_tokens=2)

        args = [10, 20, 30]
        result = wrapper(args)
        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0], [None, None, 10, 20, 30])
        self.assertEqual(args, [])
        self.assertEqual(list(result), [2, 3, 4])

    def test_effect_tokens_codegen_none_output(self):
        # [FUNCTION DOC] Check the effect token wrapper returns none when the wrapped function
        # returns none.
        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function that returns none.
            return None

        wrapper = self._make_wrapper_via_post_compile(mock_compiled_fn, num_tokens=2)

        result = wrapper([10, 20])
        self.assertIsNone(result)

    # --- FunctionalizedRngRuntimeWrapper codegen tests ---

    def _make_rng_wrapper_via_post_compile(
        self, compiled_fn, *, return_new_outs=True, num_forward_returns=1
    ):
        # [FUNCTION DOC] HELPER: Wrap a compiled function with the functionalized RNG runtime
        # wrapper using mock metadata (RNG functionalized, one offset output) and the given
        # return_new_outs flag and forward return count; returns the wrapped function.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            FunctionalizedRngRuntimeWrapper,
        )

        mock_meta = SimpleNamespace(
            is_rng_op_functionalized=True,
            num_forward_returns=num_forward_returns,
            num_outputs_rng_offset=1,
        )
        mock_aot_config = SimpleNamespace()
        return FunctionalizedRngRuntimeWrapper(
            return_new_outs=return_new_outs
        ).post_compile(compiled_fn, mock_aot_config, runtime_metadata=mock_meta)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_functionalized_rng_codegen_emitted(self):
        # [FUNCTION DOC] CUDA only. With functionalized RNG ops, compiling a random function
        # emits exactly one RNG wrapper codegen artifact that mentions getting the RNG state and
        # setting the offset.
        with torch._functorch.config.patch(functionalize_rng_ops=True):
            with capture_codegen_source("functionalized_rng_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Add a random tensor like x to x.
                    return torch.rand_like(x) + x

                x = torch.randn(4, device="cuda")
                f(x)

        self.assertEqual(
            len(captured),
            1,
            "Expected functionalized_rng_wrapper codegen artifact",
        )
        source = captured[0]
        self.assertIn("_get_rng_state_", source)
        self.assertIn("_set_offset_", source)

    def test_functionalized_rng_no_codegen(self):
        # [FUNCTION DOC] Check no RNG wrapper codegen is emitted when RNG ops are
        # not functionalized.
        with capture_codegen_source("functionalized_rng_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            f(torch.randn(4))

        self.assertEqual(
            len(captured),
            0,
            "No codegen should be emitted when RNG is not functionalized",
        )

    def test_functionalized_rng_no_codegen_returns_same_fn(self):
        # [FUNCTION DOC] Check the RNG wrapper returns the original compiled function unchanged
        # when metadata says RNG is not functionalized.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            FunctionalizedRngRuntimeWrapper,
        )

        sentinel = lambda args: args  # noqa: E731
        mock_meta = SimpleNamespace(
            is_rng_op_functionalized=False,
            num_forward_returns=1,
        )
        result = FunctionalizedRngRuntimeWrapper().post_compile(
            sentinel, SimpleNamespace(), runtime_metadata=mock_meta
        )
        self.assertIs(result, sentinel)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_functionalized_rng_codegen_correctness(self):
        # [FUNCTION DOC] CUDA only. Functionalized RNG: a compiled random function returns
        # uniform values in the unit interval with correct shape and device.
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Random tensor like the input.
                return torch.rand_like(x)

            x = torch.randn(8, device="cuda")
            out = f(x)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.device, x.device)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_functionalized_rng_codegen_multi_output(self):
        # [FUNCTION DOC] CUDA only. Functionalized RNG with two outputs: the second equals
        # double the input and the first has the right shape.
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Return input plus noise and input doubled.
                noise = torch.rand_like(x)
                return x + noise, x * 2

            x = torch.randn(4, device="cuda")
            out1, out2 = f(x)

        self.assertEqual(out2, x * 2)
        self.assertEqual(out1.shape, x.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_functionalized_rng_codegen_advances_state(self):
        # [FUNCTION DOC] CUDA only. Functionalized RNG: successive calls advance the RNG state
        # so the outputs differ.
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Random tensor like the input.
                return torch.rand_like(x)

            x = torch.randn(100, device="cuda")
            out1 = f(x)
            out2 = f(x)

        self.assertFalse(
            torch.allclose(out1, out2),
            "Successive calls should produce different random outputs",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_functionalized_rng_codegen_source_structure(self):
        # [FUNCTION DOC] CUDA only. Checks the generated RNG wrapper source defines the wrapper
        # function and uses extend and output slicing.
        with torch._functorch.config.patch(functionalize_rng_ops=True):
            with capture_codegen_source("functionalized_rng_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Random tensor like the input.
                    return torch.rand_like(x)

                f(torch.randn(4, device="cuda"))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _functionalized_rng_wrapper", source)
        self.assertIn("extend", source)
        self.assertIn("outs[", source)

    def test_functionalized_rng_codegen_unit_return_new_outs(self):
        # [FUNCTION DOC] Unit test of the RNG wrapper with returning new outputs: mocks the CUDA
        # RNG state helper so the wrapper gets a fake seed and offset, runs with two inputs, and
        # checks the new offset output is passed to the offset setter and removed from the
        # returned list.
        set_offset_log = []

        def mock_get_rng_state():
            # [FUNCTION DOC] HELPER: Mock returning a fake seed and offset pair.
            return ("seed", "offset")

        def mock_set_offset(val):
            # [FUNCTION DOC] HELPER: Mock offset setter that appends the value to a log list.
            set_offset_log.append(val)

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function returning a labeled output for each
            # input argument.
            return [f"out_{i}" for i in range(len(args))]

        from unittest.mock import patch as mock_patch

        with (
            mock_patch(
                "torch._prims_common.CUDARngStateHelper.get_torch_state_as_tuple",
                mock_get_rng_state,
            ),
            mock_patch(
                "torch._prims_common.CUDARngStateHelper.set_new_offset",
                mock_set_offset,
            ),
        ):
            wrapper = self._make_rng_wrapper_via_post_compile(
                mock_compiled_fn, return_new_outs=True, num_forward_returns=2
            )
            result = wrapper(["a", "b"])

        self.assertEqual(set_offset_log, ["out_2"])
        self.assertEqual(result, ["out_0", "out_1", "out_3"])

    def test_functionalized_rng_codegen_unit_no_return_new_outs(self):
        # [FUNCTION DOC] Same unit test with returning new outputs disabled: the offset output
        # is still passed to the setter but all outputs are returned.
        set_offset_log = []

        def mock_get_rng_state():
            # [FUNCTION DOC] HELPER: Mock returning a fake seed and offset pair.
            return ("seed", "offset")

        def mock_set_offset(val):
            # [FUNCTION DOC] HELPER: Mock offset setter that appends the value to a log list.
            set_offset_log.append(val)

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function returning a labeled output for each
            # input argument.
            return [f"out_{i}" for i in range(len(args))]

        from unittest.mock import patch as mock_patch

        with (
            mock_patch(
                "torch._prims_common.CUDARngStateHelper.get_torch_state_as_tuple",
                mock_get_rng_state,
            ),
            mock_patch(
                "torch._prims_common.CUDARngStateHelper.set_new_offset",
                mock_set_offset,
            ),
        ):
            wrapper = self._make_rng_wrapper_via_post_compile(
                mock_compiled_fn, return_new_outs=False, num_forward_returns=2
            )
            result = wrapper(["a", "b"])

        self.assertEqual(set_offset_log, ["out_2"])
        self.assertEqual(result, ["out_0", "out_1", "out_2", "out_3"])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_functionalized_rng_codegen_training(self):
        # [FUNCTION DOC] CUDA only. Functionalized RNG in training: forward and backward run and
        # gradient has input shape.
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Random tensor like the input times the input.
                return torch.rand_like(x) * x

            x = torch.randn(4, device="cuda", requires_grad=True)
            out = f(x)
            out.sum().backward()

            self.assertEqual(out.shape, x.shape)
            self.assertIsNotNone(x.grad)
            self.assertEqual(x.grad.shape, x.shape)

    # --- Backward prologue codegen tests ---

    def test_backward_prologue_codegen_emitted(self):
        # [FUNCTION DOC] Check that backward compilation emits exactly one backward prologue
        # codegen artifact that defines the prologue function and refers to the functorch
        # interpreter stack.
        with capture_codegen_source("backward_prologue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            out = f(x)
            out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _backward_prologue(", source)
        self.assertIn("torch._C._functorch.peek_interpreter_stack()", source)

    def test_backward_prologue_no_codegen_for_inference(self):
        # [FUNCTION DOC] Check no backward prologue codegen for inference-only compilation.
        with capture_codegen_source("backward_prologue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            f(torch.randn(4))

        self.assertEqual(len(captured), 0)

    def test_backward_prologue_baked_arity(self):
        # [FUNCTION DOC] Check the generated prologue bakes in the expected number of backward
        # arguments (three outputs).
        with capture_codegen_source("backward_prologue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Return input plus one, doubled input and input
                # minus one.
                return x + 1, x * 2, x - 1

            x = torch.randn(4, requires_grad=True)
            out1, out2, out3 = f(x)
            (out1 + out2 + out3).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("if len(flat_args) != 3:", source)

    def test_backward_prologue_tangent_filtering(self):
        # [FUNCTION DOC] Check the prologue filters tangents of outputs that are not
        # differentiable (an aliased view output), so only the first gradient is used.
        with capture_codegen_source("backward_prologue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Return the doubled input and a flattened view of it.
                return x * 2, x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)
            out1.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_bw_grads = [flat_args[0]]", source)

    def test_backward_prologue_correctness_multi_output(self):
        # [FUNCTION DOC] Correctness of backward through two outputs with two inputs: gradients
        # are the other input plus one.
        @torch.compile(backend="aot_eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Return product and sum of the inputs.
            return x * y, x + y

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        o1, o2 = f(x, y)
        (o1.sum() + o2.sum()).backward()
        self.assertEqual(x.grad, y + 1)
        self.assertEqual(y.grad, x + 1)

    def test_backward_prologue_correctness_with_alias(self):
        # [FUNCTION DOC] Correctness of backward when one output aliases the input: gradient of
        # the first output is two everywhere.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Return the doubled input and a flattened view of the input.
            return x * 2, x.view(-1)

        x = torch.randn(2, 3, requires_grad=True)
        out1, out2 = f(x)
        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((2, 3), 2.0))

    @xfailIfTorchDynamo
    def test_backward_prologue_correctness_with_mutation(self):
        # [FUNCTION DOC] Expected failure under dynamo. Backward through aot_function with an
        # input mutated in the forward: gradient of the other input is two.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(x, y):
            # [FUNCTION DOC] HELPER: Add one to x in place and return y doubled.
            x.add_(1)
            return y * 2

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        compiled_f = aot_function(f, nop)
        out = compiled_f(x, y)
        out.sum().backward()
        self.assertEqual(y.grad, torch.full((4,), 2.0))

    def test_backward_prologue_no_deterministic_check_when_true(self):
        # [FUNCTION DOC] Under deterministic algorithms mode, the generated backward
        # prologue does not contain a check for deterministic mode; the previous setting is
        # restored afterwards.
        prev = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with capture_codegen_source("backward_prologue") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Double the input.
                    return x * 2

                x = torch.randn(4, requires_grad=True)
                out = f(x)
                out.sum().backward()
        finally:
            torch.use_deterministic_algorithms(prev)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("are_deterministic_algorithms_enabled", source)

    def test_backward_prologue_elides_tokens_when_zero(self):
        # [FUNCTION DOC] Check the generated prologue omits token handling when there are no
        # effect tokens.
        with capture_codegen_source("backward_prologue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("[None]", source)

    def test_backward_prologue_deterministic_check_when_false(self):
        # [FUNCTION DOC] Check the generated prologue contains a deterministic algorithms check
        # when the mode is off.
        with capture_codegen_source("backward_prologue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("are_deterministic_algorithms_enabled", source)

    def test_backward_prologue_correctness_with_subclass(self):
        # [FUNCTION DOC] Correctness of compiled backward with a TwoTensor subclass input
        # against eager, comparing gradients of both components.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            # [FUNCTION DOC] HELPER: Double the input.
            return x * 2

        a_ref = torch.randn(4, requires_grad=True)
        b_ref = torch.randn(4, requires_grad=True)
        tt_ref = TwoTensor(a_ref, b_ref)
        out_ref = f(tt_ref)
        out_ref.sum().backward()

        a = a_ref.clone().detach().requires_grad_(True)
        b = b_ref.clone().detach().requires_grad_(True)
        tt = TwoTensor(a, b)
        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(tt)
        out.sum().backward()

        self.assertEqual(tt.grad.a, tt_ref.grad.a)
        self.assertEqual(tt.grad.b, tt_ref.grad.b)

    def test_backward_prologue_includes_tokens_when_nonzero(self):
        # [FUNCTION DOC] Check the backward prologue includes the token placeholder when the
        # backward graph contains an effectful op: forward op with a registered autograd rule
        # that calls another effectful op in backward.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        bwd_op = self._make_effectful_op("bw_prologue_token_bwd")
        fwd_op = self._make_effectful_op("bw_prologue_token_fwd")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward calling the effectful backward op on the gradient.
            return torch.ops.test.bw_prologue_token_bwd(grad)

        fwd_op.register_autograd(backward, setup_context=setup_context)
        h1 = _register_effectful_op(fwd_op, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op, EffectType.ORDERED)
        try:
            with capture_codegen_source("backward_prologue") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Apply the effectful forward op and double
                    # the result.
                    return torch.ops.test.bw_prologue_token_fwd(x) * 2

                x = torch.randn(4, requires_grad=True)
                out = f(x)
                out.sum().backward()

            self.assertEqual(len(captured), 1)
            source = captured[0]
            self.assertIn("[None]", source)
        finally:
            h1.destroy()
            h2.destroy()

    def test_backward_prologue_create_graph_mutation_codegen(self):
        # [FUNCTION DOC] Check the backward prologue code asks for create_graph handling when a
        # custom op's backward mutates a forward input; the op has a fake and autograd rule.
        @torch.library.custom_op("test::_bw_prologue_clone_mutate", mutates_args={})
        def clone_op(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op returning a clone of the first argument.
            return x.clone()

        @clone_op.register_fake
        def _(x, x1):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like
            # the input.
            return torch.empty_like(x)

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward rule: zero the stored second input under no_grad
            # and return the doubled gradient and no gradient for the second input.
            with torch.no_grad():
                ctx.x1.zero_()
            return grad * 2, None

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Store the second input on the context.
            (x, x1) = inputs
            ctx.x1 = x1

        clone_op.register_autograd(backward, setup_context=setup_context)

        with capture_codegen_source("backward_prologue") as captured:

            def fn(x, x1):
                # [FUNCTION DOC] HELPER: Call the custom op on both inputs.
                return torch.ops.test._bw_prologue_clone_mutate(x, x1)

            x = torch.randn(3, requires_grad=True)
            x1 = torch.randn(3, requires_grad=True)
            compiled_f = aot_function(fn, nop)
            out = compiled_f(x, x1)
            out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("create_graph=True", source)

    def test_backward_prologue_create_graph_mutation_raises(self):
        # [FUNCTION DOC] Check that double backward (create_graph) through a backward that
        # mutates a forward input raises RuntimeError that such mutations are unsupported.
        @torch.library.custom_op("test::_bw_prologue_clone_mutate2", mutates_args={})
        def clone_op(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op returning a clone of the first argument.
            return x.clone()

        @clone_op.register_fake
        def _(x, x1):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like
            # the input.
            return torch.empty_like(x)

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward rule: zero the stored second input under no_grad
            # and return the doubled gradient and no gradient for the second input.
            with torch.no_grad():
                ctx.x1.zero_()
            return grad * 2, None

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Store the second input on the context.
            (x, x1) = inputs
            ctx.x1 = x1

        clone_op.register_autograd(backward, setup_context=setup_context)

        def fn(x, x1):
            # [FUNCTION DOC] HELPER: Call the custom op on both inputs.
            return torch.ops.test._bw_prologue_clone_mutate2(x, x1)

        x = torch.randn(3, requires_grad=True)
        x1 = torch.randn(3, requires_grad=True)
        compiled_f = aot_function(fn, nop)
        out = compiled_f(x, x1)
        loss = out.sum()
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_autograd does not support input mutations with "
            "requires_grad in backward for create_graph=True",
        ):
            torch.autograd.grad(loss, x, create_graph=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_prologue_rng_codegen(self):
        # [FUNCTION DOC] CUDA only. With functionalized RNG, the generated backward prologue
        # refers to the RNG state getter.
        with torch._functorch.config.patch(functionalize_rng_ops=True):
            with capture_codegen_source("backward_prologue") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Random tensor like the input plus the input.
                    return torch.rand_like(x) + x

                x = torch.randn(4, device="cuda", requires_grad=True)
                out = f(x)
                out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_get_rng_state_", source)

    # --- CompiledFunction.forward codegen tests ---

    def test_compiled_forward_codegen_emitted(self):
        # [FUNCTION DOC] Check that the generated compiled forward source is emitted once and
        # handles tuple and list forward outputs.
        with capture_codegen_source("compiled_function_forward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _compiled_forward(", source)
        self.assertIn("if isinstance(fw_outs, tuple):", source)
        self.assertIn("elif not isinstance(fw_outs, list):", source)

    def test_compiled_forward_elides_backward_state(self):
        # [FUNCTION DOC] Check the generated forward source omits backward state handling when
        # not needed.
        with capture_codegen_source("compiled_function_forward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("BackwardState", source)

    def test_compiled_forward_elides_amp(self):
        # [FUNCTION DOC] Check the generated forward source omits the autocast disabling guard
        # when autocast is inactive.
        with capture_codegen_source("compiled_function_forward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("DisableAutocast", source)

    def test_compiled_forward_includes_amp_when_active(self):
        # [FUNCTION DOC] Check the generated forward source includes the autocast disabling
        # guard when compiled under CPU autocast.
        with capture_codegen_source("compiled_function_forward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            with torch.autocast("cpu"):
                x = torch.randn(4, requires_grad=True)
                f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_DisableAutocast_", source)

    def test_compiled_forward_no_codegen_for_inference(self):
        # [FUNCTION DOC] Check no forward codegen artifact for inference-only compilation.
        with capture_codegen_source("compiled_function_forward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            f(torch.randn(4))

        self.assertEqual(len(captured), 0)

    def test_compiled_forward_elides_rng_when_off(self):
        # [FUNCTION DOC] Check the generated forward source omits RNG state advancing when no
        # RNG states are needed.
        with capture_codegen_source("compiled_function_forward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("_rng_add_(ctx, args)", source)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_compiled_forward_rng_codegen(self):
        # [FUNCTION DOC] CUDA only. With checkpointed random ops (graph safe RNG states), the
        # generated forward source includes the RNG advance call.
        # _rng_add_ is emitted when num_graphsafe_rng_states > 0, which
        # requires recomputable RNG ops (e.g. from activation checkpointing).
        from torch.utils.checkpoint import checkpoint

        with capture_codegen_source("compiled_function_forward") as captured:

            def gn(x):
                # [FUNCTION DOC] HELPER: Random tensor like the input times the input.
                return torch.rand_like(x) * x

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply non-reentrant checkpointing to the function.
                return checkpoint(gn, x, use_reentrant=False)

            x = torch.randn(4, device="cuda", requires_grad=True)
            out = f(x)
            out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_rng_add_(ctx, args)", source)

    def test_compiled_forward_correctness(self):
        # [FUNCTION DOC] Correctness of compiled forward with two outputs: gradients are the
        # other input plus one.
        @torch.compile(backend="aot_eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Return product and sum of the inputs.
            return x * y, x + y

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        o1, o2 = f(x, y)
        (o1.sum() + o2.sum()).backward()
        self.assertEqual(x.grad, y + 1)
        self.assertEqual(y.grad, x + 1)

    # --- CompiledFunction.backward codegen tests ---

    def test_compiled_backward_codegen_emitted(self):
        # [FUNCTION DOC] Check the backward codegen artifact is emitted once and defines the
        # compiled backward function.
        with capture_codegen_source("compiled_function_backward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _compiled_backward(", source)

    def test_compiled_backward_no_codegen_for_inference(self):
        # [FUNCTION DOC] Check no backward codegen for inference-only compilation.
        with capture_codegen_source("compiled_function_backward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            f(torch.randn(4))

        self.assertEqual(len(captured), 0)

    def test_compiled_backward_elides_rng(self):
        # [FUNCTION DOC] Check the generated backward source omits RNG advancing when unused.
        with capture_codegen_source("compiled_function_backward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("_rng_add_(_ctx_", source)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_compiled_backward_rng_codegen(self):
        # [FUNCTION DOC] CUDA only. With checkpointed random ops, the generated backward source
        # includes the RNG advance call.
        # _rng_add_ is emitted when num_graphsafe_rng_states > 0, which
        # requires recomputable RNG ops (e.g. from activation checkpointing),
        # not functionalize_rng_ops (a separate mechanism using seed/offset).
        from torch.utils.checkpoint import checkpoint

        with capture_codegen_source("compiled_function_backward") as captured:

            def gn(x):
                # [FUNCTION DOC] HELPER: Random tensor like the input times the input.
                return torch.rand_like(x) * x

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply non-reentrant checkpointing to the function.
                return checkpoint(gn, x, use_reentrant=False)

            x = torch.randn(4, device="cuda", requires_grad=True)
            out = f(x)
            out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_rng_add_(_ctx_", source)

    def test_compiled_backward_elides_vc_check(self):
        # [FUNCTION DOC] Check the generated backward source omits the version counter check
        # helper when not needed.
        with capture_codegen_source("compiled_function_backward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("_tensors_no_vc_check", source)

    def test_compiled_backward_correctness(self):
        # [FUNCTION DOC] Correctness of the compiled backward: gradients of x times y plus one
        # are y and x.
        @torch.compile(backend="aot_eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Product plus one.
            return x * y + 1

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        out = f(x, y)
        out.sum().backward()
        self.assertEqual(x.grad, y)
        self.assertEqual(y.grad, x)

    def test_compiled_backward_vc_check_codegen(self):
        # [FUNCTION DOC] Check the version counter check helper appears in the generated
        # backward source when a custom Function saves its output for backward, and that the
        # gradient is twice the input.
        class VcCheckFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function with functional style forward doubling the
            # input; stores the output on the context in setup, and backward multiplies the
            # gradient by it.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Store the output on the context.
                ctx.out = output

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient times the stored output.
                return grad * ctx.out

        with capture_codegen_source("compiled_function_backward") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply the custom Function.
                return VcCheckFn.apply(x)

            x = torch.randn(4, requires_grad=True)
            out = f(x)
            out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_tensors_no_vc_check", source)
        self.assertEqual(x.grad, 2 * x)

    def test_compiled_backward_double_backward(self):
        # [FUNCTION DOC] Check double backward: create_graph gradient has the right value
        # and a grad_fn, but backward through it raises RuntimeError that double backward
        # is unsupported.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Square the input.
            return x * x

        x = torch.randn(4, requires_grad=True)
        out = f(x)
        (grad_x,) = torch.autograd.grad(out.sum(), x, create_graph=True)
        self.assertEqual(grad_x, 2 * x)
        self.assertTrue(grad_x.requires_grad)
        self.assertIsNotNone(grad_x.grad_fn)
        with self.assertRaisesRegex(
            RuntimeError, "does not currently support double backward"
        ):
            grad_x.sum().backward()

    def test_compiled_backward_multiple_outputs(self):
        # [FUNCTION DOC] Check backward through three outputs sums gradients correctly.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input doubled, squared and plus one.
            return x * 2, x * x, x + 1

        x = torch.randn(4, requires_grad=True)
        a, b, c = f(x)
        (a + b + c).sum().backward()
        self.assertEqual(x.grad, 2 + 2 * x + 1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_compiled_backward_rng_correctness(self):
        # [FUNCTION DOC] CUDA only. With functionalized RNG, the gradient through random add is
        # all ones.
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Random tensor like the input plus the input.
                return torch.rand_like(x) + x

            x = torch.randn(4, device="cuda", requires_grad=True)
            out = f(x)
            out.sum().backward()
            self.assertEqual(x.grad, torch.ones_like(x))

    def test_compiled_backward_dynamic_shapes(self):
        # [FUNCTION DOC] Check dynamic shape compilation across several input sizes gives
        # correct gradients.
        @torch.compile(backend="aot_eager", dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Square the input.
            return x * x

        for n in [4, 8, 16]:
            x = torch.randn(n, requires_grad=True)
            f(x).sum().backward()
            self.assertEqual(x.grad, 2 * x)

    def test_compiled_backward_rejects_non_list_args(self):
        # [FUNCTION DOC] Check the generated compiled backward function rejects arguments that
        # are not a single mutable list: a tuple of one tensor and a tuple of two both raise
        # AssertionError mentioning a single mutable list.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_compiled_backward,
        )

        bwd_fn = _codegen_compiled_backward(
            num_rng=0, num_tensors_no_vc_check=None, inputs_require_grad=False
        )

        def noop(*a, **kw):
            # [FUNCTION DOC] HELPER: Do nothing and return none.
            return None

        with self.assertRaisesRegex(AssertionError, "single mutable list"):
            bwd_fn(
                (torch.tensor(1.0),),
                None,
                noop,
                noop,
                noop,
                noop,
                noop,
            )
        with self.assertRaisesRegex(AssertionError, "single mutable list"):
            bwd_fn(
                (torch.tensor(1.0), torch.tensor(2.0)),
                None,
                noop,
                noop,
                noop,
                noop,
                noop,
            )

    # --- Backward epilogue codegen tests ---

    def test_backward_epilogue_codegen_emitted(self):
        # [FUNCTION DOC] Check the backward epilogue codegen artifact is emitted once, defines
        # the epilogue function and converts the output to a tuple.
        with capture_codegen_source("backward_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _backward_epilogue(", source)
        self.assertIn("tuple(out)", source)

    def test_backward_epilogue_no_codegen_for_inference(self):
        # [FUNCTION DOC] Check no backward epilogue codegen for inference-only compilation.
        with capture_codegen_source("backward_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            f(torch.randn(4))

        self.assertEqual(len(captured), 0)

    def test_backward_epilogue_elides_rng_when_off(self):
        # [FUNCTION DOC] Check the generated epilogue omits setting the RNG offset when
        # not needed.
        with capture_codegen_source("backward_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("_set_offset_", source)

    def test_backward_epilogue_elides_tokens_when_zero(self):
        # [FUNCTION DOC] Check the generated epilogue omits token stripping slices when no
        # tokens exist.
        with capture_codegen_source("backward_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("[:-", source)

    def test_backward_epilogue_correctness(self):
        # [FUNCTION DOC] Correctness of the epilogue path: gradients of product plus one.
        @torch.compile(backend="aot_eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Product plus one.
            return x * y + 1

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        out = f(x, y)
        out.sum().backward()
        self.assertEqual(x.grad, y)
        self.assertEqual(y.grad, x)

    def test_backward_epilogue_correctness_multi_output(self):
        # [FUNCTION DOC] Correctness with two outputs: gradient is five everywhere.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input times two and times three.
            return x * 2, x * 3

        x = torch.randn(4, requires_grad=True)
        a, b = f(x)
        (a.sum() + b.sum()).backward()
        self.assertEqual(x.grad, torch.full((4,), 5.0))

    def test_backward_epilogue_correctness_with_subclass(self):
        # [FUNCTION DOC] Correctness of compiled backward with a TwoTensor subclass input
        # compared against eager.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            # [FUNCTION DOC] HELPER: Double the input.
            return x * 2

        a_ref = torch.randn(4, requires_grad=True)
        b_ref = torch.randn(4, requires_grad=True)
        tt_ref = TwoTensor(a_ref, b_ref)
        out_ref = f(tt_ref)
        out_ref.sum().backward()

        a = a_ref.clone().detach().requires_grad_(True)
        b = b_ref.clone().detach().requires_grad_(True)
        tt = TwoTensor(a, b)
        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(tt)
        out.sum().backward()

        self.assertEqual(tt.grad.a, tt_ref.grad.a)
        self.assertEqual(tt.grad.b, tt_ref.grad.b)

    def test_backward_epilogue_subclass_emits_wrap(self):
        # [FUNCTION DOC] Check the epilogue codegen wraps gradients back into subclasses when
        # the input is a TwoTensor.
        from torch.testing._internal.two_tensor import TwoTensor

        with capture_codegen_source("backward_epilogue") as captured:

            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            a = torch.randn(4, requires_grad=True)
            b = torch.randn(4, requires_grad=True)
            tt = TwoTensor(a, b)
            compiled_f = torch.compile(f, backend="aot_eager")
            out = compiled_f(tt)
            out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_wrap_", source)

    def test_backward_epilogue_includes_tokens_when_nonzero(self):
        # [FUNCTION DOC] Check the epilogue strips backward effect tokens (slicing off the end)
        # when the backward graph has effectful ops.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        bwd_op = self._make_effectful_op("bw_epilogue_token_bwd")
        fwd_op = self._make_effectful_op("bw_epilogue_token_fwd")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward calling the effectful backward op on the gradient.
            return torch.ops.test.bw_epilogue_token_bwd(grad)

        fwd_op.register_autograd(backward, setup_context=setup_context)
        h1 = _register_effectful_op(fwd_op, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op, EffectType.ORDERED)
        try:
            with capture_codegen_source("backward_epilogue") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Apply the effectful forward op and double
                    # the result.
                    return torch.ops.test.bw_epilogue_token_fwd(x) * 2

                x = torch.randn(4, requires_grad=True)
                out = f(x)
                out.sum().backward()

            self.assertEqual(len(captured), 1)
            source = captured[0]
            self.assertIn("[:-", source)
        finally:
            h1.destroy()
            h2.destroy()

    def test_backward_epilogue_tokens_correctness(self):
        # [FUNCTION DOC] Correctness with backward effect tokens: gradient is two.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        bwd_op = self._make_effectful_op("bw_epilogue_tok_corr_bwd")
        fwd_op = self._make_effectful_op("bw_epilogue_tok_corr_fwd")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward calling the effectful backward op on the gradient.
            return torch.ops.test.bw_epilogue_tok_corr_bwd(grad)

        fwd_op.register_autograd(backward, setup_context=setup_context)
        h1 = _register_effectful_op(fwd_op, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply the effectful forward op and double the result.
                return torch.ops.test.bw_epilogue_tok_corr_fwd(x) * 2

            x = torch.randn(4, requires_grad=True)
            out = f(x)
            out.sum().backward()

            self.assertEqual(x.grad, torch.full((4,), 2.0))
        finally:
            h1.destroy()
            h2.destroy()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_epilogue_rng_codegen(self):
        # [FUNCTION DOC] CUDA only. With functionalized RNG the generated epilogue sets the
        # RNG offset.
        with torch._functorch.config.patch(functionalize_rng_ops=True):
            with capture_codegen_source("backward_epilogue") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Random tensor like the input plus the input.
                    return torch.rand_like(x) + x

                x = torch.randn(4, device="cuda", requires_grad=True)
                out = f(x)
                out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_set_offset_", source)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_epilogue_rng_correctness(self):
        # [FUNCTION DOC] CUDA only. Gradient with functionalized RNG compiled matches eager
        # gradient when the same seed is used.
        def f(x):
            # [FUNCTION DOC] HELPER: Random tensor like the input plus double the input.
            return torch.rand_like(x) + x * 2

        x_ref = torch.randn(4, device="cuda", requires_grad=True)
        torch.manual_seed(42)
        out_ref = f(x_ref)
        out_ref.sum().backward()

        x = x_ref.clone().detach().requires_grad_(True)
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f_compiled(x):
                # [FUNCTION DOC] HELPER: Random tensor like the input plus double the input.
                return torch.rand_like(x) + x * 2

            torch.manual_seed(42)
            out = f_compiled(x)
            out.sum().backward()

        self.assertEqual(x.grad, x_ref.grad)

    def test_backward_epilogue_create_graph(self):
        # [FUNCTION DOC] Check double backward construction (create_graph) works in the epilogue
        # path and gives twice the input.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Square the input.
            return x * x

        x = torch.randn(4, requires_grad=True)
        out = f(x)
        (grad_x,) = torch.autograd.grad(out.sum(), x, create_graph=True)
        self.assertEqual(grad_x, 2 * x)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_epilogue_tokens_and_rng_codegen(self):
        # [FUNCTION DOC] CUDA only. Check the generated epilogue slices tokens before setting
        # the RNG offset (token position precedes the offset position) when both tokens and
        # functionalized RNG are present.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        bwd_op = self._make_effectful_op("bw_epi_tok_rng_bwd")
        fwd_op = self._make_effectful_op("bw_epi_tok_rng_fwd")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward calling the effectful backward op on the gradient.
            return torch.ops.test.bw_epi_tok_rng_bwd(grad)

        fwd_op.register_autograd(backward, setup_context=setup_context)
        h1 = _register_effectful_op(fwd_op, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op, EffectType.ORDERED)
        try:
            with torch._functorch.config.patch(functionalize_rng_ops=True):
                with capture_codegen_source("backward_epilogue") as captured:

                    @torch.compile(backend="aot_eager")
                    def f(x):
                        # [FUNCTION DOC] HELPER: Apply the effectful forward op, double, add a
                        # random tensor like the input.
                        return torch.ops.test.bw_epi_tok_rng_fwd(
                            x
                        ) * 2 + torch.rand_like(x)

                    x = torch.randn(4, device="cuda", requires_grad=True)
                    f(x).sum().backward()

            self.assertEqual(len(captured), 1)
            source = captured[0]
            self.assertIn("[:-", source)
            self.assertIn("_set_offset_", source)
            token_pos = source.index("[:-")
            rng_pos = source.index("_set_offset_")
            self.assertLess(token_pos, rng_pos)
        finally:
            h1.destroy()
            h2.destroy()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_epilogue_tokens_and_rng_correctness(self):
        # [FUNCTION DOC] CUDA only. Correctness with both tokens and functionalized RNG:
        # gradient is two on CUDA.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        bwd_op = self._make_effectful_op("bw_epi_tokrng_c_bwd")
        fwd_op = self._make_effectful_op("bw_epi_tokrng_c_fwd")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward calling the effectful backward op on the gradient.
            return torch.ops.test.bw_epi_tokrng_c_bwd(grad)

        fwd_op.register_autograd(backward, setup_context=setup_context)
        h1 = _register_effectful_op(fwd_op, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op, EffectType.ORDERED)
        try:
            with torch._functorch.config.patch(functionalize_rng_ops=True):

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Apply the effectful forward op, double, add a
                    # random tensor like the input.
                    return torch.ops.test.bw_epi_tokrng_c_fwd(x) * 2 + torch.rand_like(
                        x
                    )

                x = torch.randn(4, device="cuda", requires_grad=True)
                f(x).sum().backward()

            self.assertEqual(x.grad, torch.full((4,), 2.0, device="cuda"))
        finally:
            h1.destroy()
            h2.destroy()

    def test_backward_epilogue_subclass_and_tokens_codegen(self):
        # [FUNCTION DOC] Check the epilogue codegen slices tokens and wraps subclasses when a
        # TwoTensor input is combined with an effectful op.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType
        from torch.testing._internal.two_tensor import TwoTensor

        bwd_op = self._make_effectful_op("bw_epi_sub_tok_bwd")
        fwd_op = self._make_effectful_op("bw_epi_sub_tok_fwd")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward calling the effectful backward op on the gradient.
            return torch.ops.test.bw_epi_sub_tok_bwd(grad)

        fwd_op.register_autograd(backward, setup_context=setup_context)
        h1 = _register_effectful_op(fwd_op, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op, EffectType.ORDERED)
        try:
            with capture_codegen_source("backward_epilogue") as captured:

                def f(x):
                    # [FUNCTION DOC] HELPER: Apply the effectful forward op and double
                    # the result.
                    return torch.ops.test.bw_epi_sub_tok_fwd(x) * 2

                a = torch.randn(4, requires_grad=True)
                b = torch.randn(4, requires_grad=True)
                tt = TwoTensor(a, b)
                compiled_f = torch.compile(f, backend="aot_eager")
                out = compiled_f(tt)
                out.sum().backward()

            self.assertEqual(len(captured), 1)
            source = captured[0]
            self.assertIn("[:-", source)
            self.assertIn("_wrap_", source)
        finally:
            h1.destroy()
            h2.destroy()

    @unittest.skip(
        "TwoTensor + effectful tokens backward interaction not yet supported"
    )
    def test_backward_epilogue_subclass_and_tokens_correctness(self):
        # [FUNCTION DOC] Skipped (unsupported). TwoTensor with effect tokens in backward: both
        # components should get gradient two.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType
        from torch.testing._internal.two_tensor import TwoTensor

        bwd_op = self._make_effectful_op("bw_epi_sub_tok_c_bwd")
        fwd_op = self._make_effectful_op("bw_epi_sub_tok_c_fwd")

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup that does nothing.
            pass

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward calling the effectful backward op on the gradient.
            return torch.ops.test.bw_epi_sub_tok_c_bwd(grad)

        fwd_op.register_autograd(backward, setup_context=setup_context)
        h1 = _register_effectful_op(fwd_op, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op, EffectType.ORDERED)
        try:

            def f(x):
                # [FUNCTION DOC] HELPER: Apply the effectful forward op and double the result.
                return torch.ops.test.bw_epi_sub_tok_c_fwd(x) * 2

            a = torch.randn(4, requires_grad=True)
            b = torch.randn(4, requires_grad=True)
            tt = TwoTensor(a, b)
            compiled_f = torch.compile(f, backend="aot_eager")
            out = compiled_f(tt)
            out.sum().backward()

            self.assertEqual(a.grad, torch.full((4,), 2.0))
            self.assertEqual(b.grad, torch.full((4,), 2.0))
        finally:
            h1.destroy()
            h2.destroy()

    def test_backward_epilogue_multi_token_codegen(self):
        # [FUNCTION DOC] Check the generated epilogue slices a single trailing token slot when
        # two effectful ops each have effectful backwards.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        bwd_op1 = self._make_effectful_op("bw_epi_mt_bwd1")
        fwd_op1 = self._make_effectful_op("bw_epi_mt_fwd1")
        bwd_op2 = self._make_effectful_op("bw_epi_mt_bwd2")
        fwd_op2 = self._make_effectful_op("bw_epi_mt_fwd2")

        def setup_context1(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup for the first op that does nothing.
            pass

        def backward1(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward of the first op calling its effectful backward op.
            return torch.ops.test.bw_epi_mt_bwd1(grad)

        def setup_context2(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup for the second op that does nothing.
            pass

        def backward2(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward of the second op calling its effectful
            # backward op.
            return torch.ops.test.bw_epi_mt_bwd2(grad)

        fwd_op1.register_autograd(backward1, setup_context=setup_context1)
        fwd_op2.register_autograd(backward2, setup_context=setup_context2)
        h1 = _register_effectful_op(fwd_op1, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op1, EffectType.ORDERED)
        h3 = _register_effectful_op(fwd_op2, EffectType.ORDERED)
        h4 = _register_effectful_op(bwd_op2, EffectType.ORDERED)
        try:
            with capture_codegen_source("backward_epilogue") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Apply the first effectful op, then the second,
                    # and double.
                    y = torch.ops.test.bw_epi_mt_fwd1(x)
                    return torch.ops.test.bw_epi_mt_fwd2(y) * 2

                x = torch.randn(4, requires_grad=True)
                f(x).sum().backward()

            self.assertEqual(len(captured), 1)
            source = captured[0]
            self.assertIn("[:-1]", source)
        finally:
            h1.destroy()
            h2.destroy()
            h3.destroy()
            h4.destroy()

    def test_backward_epilogue_multi_token_correctness(self):
        # [FUNCTION DOC] Correctness with two effectful ops in forward and backward: gradient
        # is two.
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        bwd_op1 = self._make_effectful_op("bw_epi_mtc_bwd1")
        fwd_op1 = self._make_effectful_op("bw_epi_mtc_fwd1")
        bwd_op2 = self._make_effectful_op("bw_epi_mtc_bwd2")
        fwd_op2 = self._make_effectful_op("bw_epi_mtc_fwd2")

        def setup_context1(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup for the first op that does nothing.
            pass

        def backward1(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward of the first op calling its effectful backward op.
            return torch.ops.test.bw_epi_mtc_bwd1(grad)

        def setup_context2(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup for the second op that does nothing.
            pass

        def backward2(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward of the second op calling its effectful
            # backward op.
            return torch.ops.test.bw_epi_mtc_bwd2(grad)

        fwd_op1.register_autograd(backward1, setup_context=setup_context1)
        fwd_op2.register_autograd(backward2, setup_context=setup_context2)
        h1 = _register_effectful_op(fwd_op1, EffectType.ORDERED)
        h2 = _register_effectful_op(bwd_op1, EffectType.ORDERED)
        h3 = _register_effectful_op(fwd_op2, EffectType.ORDERED)
        h4 = _register_effectful_op(bwd_op2, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Apply the first effectful op, then the second,
                # and double.
                y = torch.ops.test.bw_epi_mtc_fwd1(x)
                return torch.ops.test.bw_epi_mtc_fwd2(y) * 2

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

            self.assertEqual(x.grad, torch.full((4,), 2.0))
        finally:
            h1.destroy()
            h2.destroy()
            h3.destroy()
            h4.destroy()

    def test_backward_epilogue_subclass_and_rng_codegen(self):
        # [FUNCTION DOC] Call the backward epilogue code generator directly with mock metadata
        # having functionalized RNG and no tokens; the artifact sets the offset, wraps with the
        # given callable, and has no token slicing.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=0,
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
        )

        def dummy_wrap(out):
            # [FUNCTION DOC] HELPER: Identity wrapper returning its input.
            return out

        with capture_codegen_source("backward_epilogue") as captured:
            _codegen_backward_epilogue(
                fw_metadata, SimpleNamespace(grad_input_metas=[]), dummy_wrap
            )

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_set_offset_", source)
        self.assertIn("_wrap_", source)
        self.assertNotIn("[:-", source)

    def test_backward_epilogue_subclass_tokens_and_rng_codegen(self):
        # [FUNCTION DOC] Call the epilogue code generator with two tokens and functionalized
        # RNG: the source slices two tokens first, then sets the RNG offset, then wraps.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=2,
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
        )

        def dummy_wrap(out):
            # [FUNCTION DOC] HELPER: Identity wrapper returning its input.
            return out

        with capture_codegen_source("backward_epilogue") as captured:
            _codegen_backward_epilogue(
                fw_metadata, SimpleNamespace(grad_input_metas=[]), dummy_wrap
            )

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("[:-2]", source)
        self.assertIn("_set_offset_", source)
        self.assertIn("_wrap_", source)
        token_pos = source.index("[:-2]")
        rng_pos = source.index("_set_offset_")
        wrap_pos = source.index("_wrap_")
        self.assertLess(token_pos, rng_pos)
        self.assertLess(rng_pos, wrap_pos)

    def test_backward_epilogue_subclass_fallback_codegen(self):
        # [FUNCTION DOC] Call the epilogue code generator without a custom wrapper (none) and
        # empty gradient metadata: the generated source calls the general subclass wrapping
        # function rather than a custom wrapper.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=0,
            is_rng_op_functionalized=False,
        )
        subclass_meta = SimpleNamespace(grad_input_metas=[])

        with capture_codegen_source("backward_epilogue") as captured:
            _codegen_backward_epilogue(fw_metadata, subclass_meta, None)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_wrap_subclasses_", source)
        self.assertNotIn("_wrap_(", source)

    def test_backward_epilogue_subclass_fallback_none_metas_raises(self):
        # [FUNCTION DOC] Check the epilogue code generator raises AssertionError when the
        # gradient input metadata is none and no custom wrapper is given.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=0,
            is_rng_op_functionalized=False,
        )
        subclass_meta = SimpleNamespace(grad_input_metas=None)

        with self.assertRaises(AssertionError):
            _codegen_backward_epilogue(fw_metadata, subclass_meta, None)

    def test_backward_epilogue_make_subclass_override_codegen(self):
        # [FUNCTION DOC] Call the epilogue code generator with a custom wrapper and empty
        # gradient metadata: the source references the override as well as the general
        # subclass wrapper.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=0,
            is_rng_op_functionalized=False,
        )
        subclass_meta = SimpleNamespace(grad_input_metas=[])

        def dummy_wrap(out):
            # [FUNCTION DOC] HELPER: Identity wrapper returning its input.
            return out

        with capture_codegen_source("backward_epilogue") as captured:
            _codegen_backward_epilogue(fw_metadata, subclass_meta, dummy_wrap)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("make_subclass_override", source)
        self.assertIn("_wrap_(", source)
        self.assertIn("_wrap_subclasses_(", source)

    def test_backward_epilogue_make_subclass_override_fast_path(self):
        # [FUNCTION DOC] Call the generated epilogue with a fast wrapper and no override: the
        # fast wrapper is called once with the output list and nothing else.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=0,
            is_rng_op_functionalized=False,
        )
        subclass_meta = SimpleNamespace(grad_input_metas=[])

        call_log = []

        def fast_wrap(out):
            # [FUNCTION DOC] HELPER: Wrapper that logs a fast call and returns its input.
            call_log.append("fast")
            return out

        fn = _codegen_backward_epilogue(fw_metadata, subclass_meta, fast_wrap)
        fn([1, 2, 3])
        self.assertEqual(call_log, ["fast"])

    def test_backward_epilogue_make_subclass_override_slow_path(self):
        # [FUNCTION DOC] Call the generated epilogue with an override function supplied: the
        # override path is taken and the fast wrapper is never called.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=0,
            is_rng_op_functionalized=False,
        )
        subclass_meta = SimpleNamespace(grad_input_metas=[])

        call_log = []

        def fast_wrap(out):
            # [FUNCTION DOC] HELPER: Wrapper that logs a fast call and returns its input.
            call_log.append("fast")
            return out

        def override(meta, is_runtime, args):
            # [FUNCTION DOC] HELPER: Override function that logs an override call and returns
            # the first argument.
            call_log.append("override")
            return args[0]

        fn = _codegen_backward_epilogue(fw_metadata, subclass_meta, fast_wrap)
        fn([], make_subclass_override=override)
        self.assertNotIn("fast", call_log)

    def test_backward_epilogue_no_subclass_no_override_param(self):
        # [FUNCTION DOC] Check the generated epilogue has no override parameter when there is no
        # subclass metadata.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _codegen_backward_epilogue,
        )

        fw_metadata = SimpleNamespace(
            num_backward_tokens=0,
            is_rng_op_functionalized=False,
        )

        with capture_codegen_source("backward_epilogue") as captured:
            _codegen_backward_epilogue(fw_metadata, None, None)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertNotIn("make_subclass_override", source)

    # --- Compiled autograd + backward epilogue integration tests ---

    def _run_with_compiled_autograd(self, fn, compiler_fn=None):
        # [FUNCTION DOC] HELPER: Run the function with compiled autograd enabled and
        # multithreading disabled after resetting dynamo; uses a default compiler that compiles
        # the backward graph with aot_eager, full graph and dynamic shapes unless one is given.
        # Returns the function result.
        from torch._dynamo import compiled_autograd

        if compiler_fn is None:

            def compiler_fn(gm):
                # [FUNCTION DOC] HELPER: Default compiler for compiled autograd: compile the
                # graph module with aot_eager.
                return torch.compile(
                    gm, backend="aot_eager", fullgraph=True, dynamic=True
                )

        torch._dynamo.reset()
        with (
            compiled_autograd._enable(compiler_fn),
            torch.autograd.set_multithreading_enabled(False),
        ):
            return fn()

    def test_backward_epilogue_compiled_autograd_basic(self):
        # [FUNCTION DOC] Check the backward result is the same under compiled autograd as
        # without it for a simple compiled function.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Double the input.
            return x * 2

        def run():
            # [FUNCTION DOC] HELPER: Run forward and backward on a fresh input and return
            # its gradient.
            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()
            return x.grad

        expected = run()
        torch._dynamo.reset()
        actual = self._run_with_compiled_autograd(run)
        self.assertEqual(actual, expected)

    def test_backward_epilogue_compiled_autograd_multi_output(self):
        # [FUNCTION DOC] Same compiled autograd comparison for a function with two outputs.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input doubled and tripled.
            return x * 2, x * 3

        def run():
            # [FUNCTION DOC] HELPER: Run forward and backward on both outputs of a fresh input
            # and return its gradient.
            x = torch.randn(4, requires_grad=True)
            a, b = f(x)
            (a.sum() + b.sum()).backward()
            return x.grad

        expected = run()
        torch._dynamo.reset()
        actual = self._run_with_compiled_autograd(run)
        self.assertEqual(actual, expected)

    def test_backward_epilogue_compiled_autograd_subclass(self):
        # [FUNCTION DOC] Same compiled autograd comparison for a TwoTensor subclass input,
        # comparing both component gradients.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            # [FUNCTION DOC] HELPER: Double the input.
            return x * 2

        compiled_f = torch.compile(f, backend="aot_eager")

        def run():
            # [FUNCTION DOC] HELPER: Run forward and backward on a TwoTensor and return the
            # gradients of both components.
            a = torch.randn(4, requires_grad=True)
            b = torch.randn(4, requires_grad=True)
            tt = TwoTensor(a, b)
            out = compiled_f(tt)
            out.sum().backward()
            return a.grad, b.grad

        expected = run()
        torch._dynamo.reset()
        actual = self._run_with_compiled_autograd(run)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])

    # --- AOTSyntheticBaseWrapper codegen tests ---

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_emitted(self):
        # [FUNCTION DOC] Expected failure under dynamo. Aliased inputs with a mutation make
        # aot_function emit the synthetic base wrapper codegen once, referring to the wrapper
        # function and the bases.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        with capture_codegen_source("synthetic_base_wrapper") as captured:

            def f(a, b):
                # [FUNCTION DOC] HELPER: Double a in place and return a plus one.
                a.mul_(2)
                return a + 1

            base = torch.randn(8)
            compiled_f = aot_function(f, nop)
            compiled_f(base[:6], base[2:])

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _synthetic_base_wrapper(", source)
        self.assertIn("_bases", source)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_correctness(self):
        # [FUNCTION DOC] Expected failure under dynamo. Correctness: after calling a function
        # that doubles a in place on two overlapping views, a equals double its old value.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(a, b):
            # [FUNCTION DOC] HELPER: Double a in place and return a plus b.
            a.mul_(2)
            return a + b

        base = torch.randn(8)
        a = base[:6]
        b = base[2:]
        a_ref = a.clone()
        compiled_f = aot_function(f, nop)
        compiled_f(a, b)
        self.assertEqual(a, a_ref * 2)

    def test_synthetic_base_no_codegen_when_not_needed(self):
        # [FUNCTION DOC] Check no synthetic base wrapper is generated for a simple function.
        with capture_codegen_source("synthetic_base_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Double the input.
                return x * 2

            f(torch.randn(4))

        self.assertEqual(len(captured), 0)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_metadata_mutation(self):
        # [FUNCTION DOC] Expected failure under dynamo. With a data mutation on one view and a
        # metadata mutation on an aliased view, the generated wrapper uses strided re-creation
        # and a metadata mutation helper.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(a, b):
            # [FUNCTION DOC] HELPER: Double a in place, transpose b in place, return a plus one.
            a.mul_(2)
            b.t_()
            return a + 1

        base = torch.randn(4, 4)
        a = base[:3, :]
        b = base[1:, :]

        with capture_codegen_source("synthetic_base_wrapper") as captured:
            compiled_f = aot_function(f, nop)
            compiled_f(a, b)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("as_strided_", source)
        self.assertIn("_meta_mut", source)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_multiple_base_groups(self):
        # [FUNCTION DOC] Expected failure under dynamo. Two groups of aliased views over two
        # bases: the generated source allocates a list sized for two bases.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        base1 = torch.randn(8)
        base2 = torch.randn(8)
        a1 = base1[:6]
        a2 = base1[2:]
        b1 = base2[:6]
        b2 = base2[2:]

        def f(a1, a2, b1, b2):
            # [FUNCTION DOC] HELPER: Double a1, triple b1 in place, return the sum of all four.
            a1.mul_(2)
            b1.mul_(3)
            return a1 + a2 + b1 + b2

        with capture_codegen_source("synthetic_base_wrapper") as captured:
            compiled_f = aot_function(f, nop)
            compiled_f(a1, a2, b1, b2)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("[None] * 2", source)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_with_non_aliased_args(self):
        # [FUNCTION DOC] Expected failure under dynamo. Aliased inputs plus an unaliased tensor:
        # mutation of the first view is applied correctly.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        base = torch.randn(8)
        a = base[:6]
        b = base[2:]
        c = torch.randn(6)

        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Double a in place and return the sum of the three.
            a.mul_(2)
            return a + b + c

        compiled_f = aot_function(f, nop)
        a_ref = a.clone()
        compiled_f(a, b, c)
        self.assertEqual(a, a_ref * 2)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_base_is_none(self):
        # [FUNCTION DOC] Expected failure under dynamo. Aliased inputs detached so neither has a
        # base: generated source creates an empty tensor and uses set_; mutated view equals old
        # value plus one.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(a, b):
            # [FUNCTION DOC] HELPER: Add one to a in place and return a plus b.
            a.add_(1)
            return a + b

        base = torch.randn(8)
        a = base[:6].detach()
        b = base[2:].detach()
        self.assertIsNone(a._base)
        self.assertIsNone(b._base)

        a_ref = a.clone()
        with capture_codegen_source("synthetic_base_wrapper") as captured:
            compiled_f = aot_function(f, nop)
            compiled_f(a, b)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("torch.empty((0,)", source)
        self.assertIn("set_", source)
        self.assertEqual(a, a_ref + 1)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_metadata_mutation_correctness(self):
        # [FUNCTION DOC] Expected failure under dynamo. Correctness of a data mutation plus
        # metadata mutation on aliased views: a doubled and b's shape transposed.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(a, b):
            # [FUNCTION DOC] HELPER: Double a in place, transpose b in place, return a plus one.
            a.mul_(2)
            b.t_()
            return a + 1

        base = torch.randn(4, 4)
        a = base[:3, :]
        b = base[1:, :]

        a_ref = a.clone()
        b_shape_before = b.shape

        compiled_f = aot_function(f, nop)
        compiled_f(a, b)

        self.assertEqual(a, a_ref * 2)
        self.assertEqual(b.shape, (b_shape_before[1], b_shape_before[0]))

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_multiple_groups_correctness(self):
        # [FUNCTION DOC] Expected failure under dynamo. Correctness with two groups of aliased
        # views: both mutated views updated.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        base1 = torch.randn(8)
        base2 = torch.randn(8)
        a1 = base1[:6]
        a2 = base1[2:]
        b1 = base2[:6]
        b2 = base2[2:]

        def f(a1, a2, b1, b2):
            # [FUNCTION DOC] HELPER: Double a1, triple b1 in place, return the sum of all four.
            a1.mul_(2)
            b1.mul_(3)
            return a1 + a2 + b1 + b2

        a1_ref = a1.clone()
        b1_ref = b1.clone()

        compiled_f = aot_function(f, nop)
        compiled_f(a1, a2, b1, b2)

        self.assertEqual(a1, a1_ref * 2)
        self.assertEqual(b1, b1_ref * 3)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_all_aliased_no_others(self):
        # [FUNCTION DOC] Expected failure under dynamo. When all inputs are aliased and none are
        # other arguments, the generated source concatenates the bases with an empty list;
        # mutated view is doubled.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(a, b):
            # [FUNCTION DOC] HELPER: Double a in place and return a plus b.
            a.mul_(2)
            return a + b

        base = torch.randn(8)
        a = base[:6]
        b = base[2:]

        a_ref = a.clone()

        with capture_codegen_source("synthetic_base_wrapper") as captured:
            compiled_f = aot_function(f, nop)
            compiled_f(a, b)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_bases + []", source)
        self.assertEqual(a, a_ref * 2)

    @xfailIfTorchDynamo
    def test_synthetic_base_codegen_non_aliased_before_aliased(self):
        # [FUNCTION DOC] Expected failure under dynamo. A non-aliased argument placed before two
        # aliased views: mutation applied to the view and the other argument unchanged.
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(c, a, b):
            # [FUNCTION DOC] HELPER: Double a in place and return the sum of the three.
            a.mul_(2)
            return a + b + c

        c = torch.randn(6)
        base = torch.randn(8)
        a = base[:6]
        b = base[2:]

        c_ref = c.clone()
        a_ref = a.clone()

        compiled_f = aot_function(f, nop)
        compiled_f(c, a, b)

        self.assertEqual(a, a_ref * 2)
        self.assertEqual(c, c_ref)

    def _compile_with_aot_stage2(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        arg: torch.Tensor,
        *,
        fw_compiler: Callable[[torch.fx.GraphModule, list[Any]], Callable[..., Any]],
        bw_compiler: Callable[[torch.fx.GraphModule, list[Any]], Callable[..., Any]]
        | None = None,
        inference_compiler: Callable[
            [torch.fx.GraphModule, list[Any]], Callable[..., Any]
        ]
        | None = None,
    ) -> tuple[Callable[..., Any], pytree.TreeSpec, Any]:
        # [FUNCTION DOC] HELPER: Run only the graph capture and compile stages for a
        # one-argument function using hand-built configuration (no compilers set on the config,
        # keep input mutations), fake mode, flattened function and the default partition with
        # the given compilers. Returns the compiled function, the output tree spec and the
        # configuration object as it was before stage two.
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._functorch._aot_autograd.frontend_utils import (
            construct_fake_mode,
            process_inputs,
        )
        from torch._functorch._aot_autograd.graph_compile import (
            aot_stage1_graph_capture,
            aot_stage2_compile,
        )
        from torch._functorch._aot_autograd.utils import create_tree_flattened_fn
        from torch._functorch.aot_autograd import AOTConfig, create_aot_state

        flat_args = [arg]
        flat_fn, out_spec = create_tree_flattened_fn(f, (arg,), {})
        aot_config = AOTConfig(
            fw_compiler=None,
            bw_compiler=None,
            inference_compiler=None,
            partition_fn=None,
            decompositions=None,
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=True,
            enable_log=False,
        )
        fake_mode, shape_env = construct_fake_mode(flat_args, aot_config)
        fake_flat_args, act_input_paths = process_inputs(
            flat_args, aot_config, fake_mode, shape_env
        )
        flat_args_descs = [PlainAOTInput(i) for i in range(len(fake_flat_args))]

        with ExitStack() as stack:
            aot_state = create_aot_state(
                stack,
                flat_fn,
                fake_flat_args,
                flat_args_descs,
                aot_config,
                fake_mode,
                shape_env,
            )
            aot_state.fw_metadata.act_input_paths = act_input_paths
            aot_config_before_stage2 = aot_state.aot_config
            aot_graph_capture = aot_stage1_graph_capture(aot_state, flat_fn)
            compiled_fn, _ = aot_stage2_compile(
                aot_state,
                aot_graph_capture,
                default_partition,
                fw_compiler,
                bw_compiler,
                inference_compiler,
            )

        return compiled_fn, out_spec, aot_config_before_stage2

    def test_aot_stage2_keeps_aot_config_immutable_for_lazy_backward(self):
        # [FUNCTION DOC] Check that stage two compilation does not alter the frozen
        # configuration (compilers and partition function stay none, assignment raises a frozen
        # instance error) and that the backward compiler runs lazily only on the first backward;
        # outputs and gradients match eager.
        import dataclasses

        bw_compile_calls = [0]

        def f(x):
            # [FUNCTION DOC] HELPER: Sum of sine of x times x.
            return (x.sin() * x).sum()

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler that returns the graph as a
            # boxed function.
            return make_boxed_func(gm)

        def bw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backward compiler that counts calls and returns the graph
            # as a boxed function.
            bw_compile_calls[0] += 1
            return make_boxed_func(gm)

        compiled_x = torch.randn(4, requires_grad=True)
        eager_x = compiled_x.detach().clone().requires_grad_()

        compiled_fn, out_spec, aot_config = self._compile_with_aot_stage2(
            f,
            compiled_x,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
        )

        self.assertIsNone(aot_config.partition_fn)
        self.assertIsNone(aot_config.fw_compiler)
        self.assertIsNone(aot_config.bw_compiler)
        self.assertIsNone(aot_config.inference_compiler)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            aot_config.partition_fn = default_partition

        compiled_out = out_spec.unflatten(compiled_fn([compiled_x]))
        eager_out = f(eager_x)
        self.assertEqual(bw_compile_calls[0], 0)
        compiled_out.backward()
        eager_out.backward()

        self.assertEqual(compiled_out, eager_out)
        self.assertEqual(compiled_x.grad, eager_x.grad)
        self.assertEqual(bw_compile_calls[0], 1)

    def test_aot_stage2_uses_explicit_inference_compiler(self):
        # [FUNCTION DOC] Check stage two uses the explicit inference compiler for a no-grad
        # function: forward compiler never called, inference compiler called once, result
        # matches eager.
        fw_compile_calls = [0]
        inference_compile_calls = [0]

        def f(x):
            # [FUNCTION DOC] HELPER: Cosine of the input.
            return x.cos()

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler that counts calls and returns the graph as
            # a boxed function.
            fw_compile_calls[0] += 1
            return make_boxed_func(gm)

        def inference_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Inference compiler that counts calls and returns the graph
            # as a boxed function.
            inference_compile_calls[0] += 1
            return make_boxed_func(gm)

        x = torch.randn(4)

        compiled_fn, out_spec, aot_config = self._compile_with_aot_stage2(
            f,
            x,
            fw_compiler=fw_compiler,
            inference_compiler=inference_compiler,
        )

        compiled_out = out_spec.unflatten(compiled_fn([x]))

        self.assertEqual(compiled_out, f(x))
        self.assertEqual(fw_compile_calls[0], 0)
        self.assertEqual(inference_compile_calls[0], 1)
        self.assertIsNone(aot_config.inference_compiler)

    def test_collect_metadata_subclass_fw_outs_follow_input_mutation_type(self):
        # [FUNCTION DOC] Check subclass forward output metadata with a TwoTensor input that is
        # mutated in place: with input mutations kept the metadata has one subclass entry
        # starting at flat index zero; with mutations out of graph there are two entries
        # starting at 0 and 2 and the mutated input runtime index is recorded.
        from torch._functorch._aot_autograd.collect_metadata_analysis import (
            run_functionalized_fw_and_collect_metadata,
        )
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

        def f(x):
            # [FUNCTION DOC] HELPER: Add one in place and return a list with the sine.
            x.add_(1)
            return [torch.sin(x)]

        fake_mode = FakeTensorMode()
        subclass_arg = TwoTensor(
            fake_mode.from_tensor(torch.ones(2)),
            fake_mode.from_tensor(torch.ones(2)),
        )

        keep_input_mutations_meta = run_functionalized_fw_and_collect_metadata(
            f,
            flat_args_descs=[PlainAOTInput(0)],
            keep_input_mutations=True,
            static_input_indices=[],
        )(subclass_arg)
        self.assertEqual(keep_input_mutations_meta.mutated_inp_runtime_indices, [])
        self.assertEqual(len(keep_input_mutations_meta.subclass_fw_graph_out_meta), 1)
        self.assertIsInstance(
            keep_input_mutations_meta.subclass_fw_graph_out_meta[0],
            SubclassCreationMeta,
        )
        self.assertEqual(
            keep_input_mutations_meta.subclass_fw_graph_out_meta[
                0
            ].flat_tensor_start_idx,
            0,
        )

        out_of_graph_mutation_meta = run_functionalized_fw_and_collect_metadata(
            f,
            flat_args_descs=[PlainAOTInput(0)],
            keep_input_mutations=False,
            static_input_indices=[],
        )(subclass_arg)
        self.assertEqual(out_of_graph_mutation_meta.mutated_inp_runtime_indices, [0])
        self.assertEqual(len(out_of_graph_mutation_meta.subclass_fw_graph_out_meta), 2)
        self.assertIsInstance(
            out_of_graph_mutation_meta.subclass_fw_graph_out_meta[0],
            SubclassCreationMeta,
        )
        self.assertIsInstance(
            out_of_graph_mutation_meta.subclass_fw_graph_out_meta[1],
            SubclassCreationMeta,
        )
        self.assertEqual(
            out_of_graph_mutation_meta.subclass_fw_graph_out_meta[
                0
            ].flat_tensor_start_idx,
            0,
        )
        self.assertEqual(
            out_of_graph_mutation_meta.subclass_fw_graph_out_meta[
                1
            ].flat_tensor_start_idx,
            2,
        )

    def test_collect_metadata_subclass_fw_outs_include_metadata_only_mutation(self):
        # [FUNCTION DOC] Check subclass forward output metadata when a plain input has only a
        # metadata mutation (transpose) and the function returns a TwoTensor: first entry is a
        # plain tensor metadata at unwrapped index zero, second a subclass creation entry
        # starting at flat index one.
        from torch._functorch._aot_autograd.collect_metadata_analysis import (
            run_functionalized_fw_and_collect_metadata,
        )
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._functorch._aot_autograd.schemas import (
            PlainTensorMeta,
            SubclassCreationMeta,
        )

        def f(x):
            # [FUNCTION DOC] HELPER: Transpose in place and return a list with a TwoTensor of
            # sine and cosine.
            x.transpose_(0, 1)
            return [TwoTensor(torch.sin(x), torch.cos(x))]

        fake_mode = FakeTensorMode()
        arg = fake_mode.from_tensor(torch.ones(2, 3))

        metadata = run_functionalized_fw_and_collect_metadata(
            f,
            flat_args_descs=[PlainAOTInput(0)],
            keep_input_mutations=True,
            static_input_indices=[],
        )(arg)

        self.assertEqual(metadata.mutated_inp_runtime_indices, [0])
        self.assertEqual(len(metadata.subclass_fw_graph_out_meta), 2)
        self.assertIsInstance(metadata.subclass_fw_graph_out_meta[0], PlainTensorMeta)
        self.assertEqual(metadata.subclass_fw_graph_out_meta[0].unwrapped_idx, 0)
        self.assertIsInstance(
            metadata.subclass_fw_graph_out_meta[1],
            SubclassCreationMeta,
        )
        self.assertEqual(
            metadata.subclass_fw_graph_out_meta[1].flat_tensor_start_idx,
            1,
        )

    def test_collect_metadata_subclass_fw_outs_include_intermediate_bases(self):
        # [FUNCTION DOC] Check subclass forward output metadata with intermediate bases: a
        # TwoTensor input yields two aliasing view outputs of one intermediate; one intermediate
        # base is recorded and all three output entries are plain tensor metadata at indices
        # zero, one, two.
        from torch._functorch._aot_autograd.collect_metadata_analysis import (
            run_functionalized_fw_and_collect_metadata,
        )
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._functorch._aot_autograd.schemas import PlainTensorMeta

        def f(x):
            # [FUNCTION DOC] HELPER: Sum the two components and return two flattened views of
            # the sum.
            y = x.a + x.b
            return [y.view(-1), y.view(-1)]

        fake_mode = FakeTensorMode()
        subclass_arg = TwoTensor(
            fake_mode.from_tensor(torch.ones(2, 3, requires_grad=True)),
            fake_mode.from_tensor(torch.ones(2, 3, requires_grad=True)),
        )

        metadata = run_functionalized_fw_and_collect_metadata(
            f,
            flat_args_descs=[PlainAOTInput(0)],
            keep_input_mutations=True,
            static_input_indices=[],
        )(subclass_arg)

        self.assertEqual(metadata.num_intermediate_bases, 1)
        self.assertEqual(len(metadata.subclass_fw_graph_out_meta), 3)
        self.assertTrue(
            all(
                isinstance(out_meta, PlainTensorMeta)
                for out_meta in metadata.subclass_fw_graph_out_meta
            )
        )
        self.assertEqual(
            [
                out_meta.unwrapped_idx
                for out_meta in metadata.subclass_fw_graph_out_meta
            ],
            [0, 1, 2],
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_register_hook_in_checkpoint_accumulated_grad(self):
        # [FUNCTION DOC] CUDA only. A hook registered on an intermediate inside a checkpointed
        # region halves its gradient; compiled with aot_eager the input gradient equals the
        # eager gradient.
        def body(y):
            # [FUNCTION DOC] HELPER: Register a gradient-halving hook on the input and return
            # a clone.
            y.register_hook(lambda grad: grad * 0.5)
            return y.clone()

        def fn(x):
            # [FUNCTION DOC] HELPER: Double the input, run the body with non-reentrant
            # checkpointing, return the sum with the doubled input.
            y = x * 2
            z = torch.utils.checkpoint.checkpoint(body, y, use_reentrant=False)
            return z + y

        x = torch.randn(4, device="cuda", requires_grad=True)
        out = fn(x)
        out.sum().backward()
        eager_grad = x.grad.clone()

        x2 = x.detach().clone().requires_grad_(True)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        out2 = opt_fn(x2)
        out2.sum().backward()
        self.assertEqual(x2.grad, eager_grad)

    def test_size_of_void_returning_nodes(self):
        # [FUNCTION DOC] Check the partitioner size helper returns zero for void-returning graph
        # nodes (assert ops, range constraint) which have no value metadata, instead of raising.
        # Void-returning ops have no `val` metadata. _size_of must return 0
        # for them instead of raising RuntimeError.
        from torch._functorch.partitioners import _size_of

        g = torch.fx.Graph()
        for target in (
            torch.ops.aten._assert_async.default,
            torch.ops.aten._assert_async.msg,
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten.sym_constrain_range.default,
            torch.ops.aten._assert_tensor_metadata.default,
        ):
            node = g.call_function(target, args=())
            self.assertEqual(_size_of(node), 0)

    def test_size_of_fake_script_object(self):
        # [FUNCTION DOC] Check the partitioner size helper with a fake script object value: by
        # default it raises RuntimeError saying the size cannot be computed; with the unsafe
        # config enabled it treats it as zero size.
        import torch._functorch.config as functorch_config
        from torch._functorch.partitioners import _size_of
        from torch._library.fake_class_registry import FakeScriptObject

        g = torch.fx.Graph()
        node = g.call_function(torch.ops.aten.abs.default, args=())
        node.meta["val"] = FakeScriptObject(None, "test_class", None)

        # A (Fake)ScriptObject may hold tensors internally, so by default
        # _size_of refuses to guess its memory footprint and raises.
        with self.assertRaisesRegex(RuntimeError, "Cannot compute the size"):
            _size_of(node)

        # The unsafe escape hatch makes _size_of assume such objects are
        # zero size, unblocking compilation at the cost of soundness.
        with functorch_config.patch(unsafe_treat_script_objects_as_zero_size=True):
            self.assertEqual(_size_of(node), 0)

    def _build_control_deps_graph(
        self,
        deps,
        additional_deps=(),
        subgraph_op=None,
        fwd_uses=None,
    ):
        # [FUNCTION DOC] HELPER: Build a small test graph with a control dependency node over
        # given dependencies. Dependencies are labeled forward (derived from a forward
        # placeholder) or backward (derived from a backward placeholder); extra ordering
        # dependencies, an optional operation inside the subgraph, and which dependency outputs
        # the forward result uses can be given. Returns the graph, forward inputs, outputs,
        # output descriptors and owning module.
        """Build a minimal control_deps test graph.

        Args:
            deps: list of ("fwd" | "bw") indicating validity of each dep
            additional_deps: list of ("fwd" | "bw") for ordering deps
            subgraph_op: if provided, the subgraph's first output is
                         op(dep_0) instead of None
            fwd_uses: which dep indices (0-based) to consume in the forward
                      output. Defaults to all fwd deps.
        Returns: (graph, inputs, outputs, outputs_descs, mod)
        """
        import torch.fx as fx
        from torch._functorch._aot_autograd.descriptors import DummyAOTOutput
        from torch._inductor.fx_passes.control_dependencies import (
            control_deps as cd_hop,
        )

        g = fx.Graph()
        mod = torch.nn.Module()
        t = torch.randn(4)

        fwd_in = g.placeholder("primals_0")
        fwd_in.meta = {"val": t}
        bw_in = g.placeholder("tangents_0")
        bw_in.meta = {"val": t}

        def _make_val(label):
            # [FUNCTION DOC] HELPER: Create a sine node of the forward placeholder or a cosine
            # node of the backward placeholder depending on the label and attach value metadata.
            if label == "fwd":
                n = g.call_function(torch.ops.aten.sin.default, (fwd_in,))
            else:
                n = g.call_function(torch.ops.aten.cos.default, (bw_in,))
            n.meta = {"val": t}
            return n

        dep_nodes = [_make_val(d) for d in deps]
        addl_nodes = [_make_val(d) for d in additional_deps]

        sg = fx.Graph()
        sg_phs = [sg.placeholder(f"d{i}") for i in range(len(deps))]
        if subgraph_op is not None:
            op_out = sg.call_function(subgraph_op, (sg_phs[0],))
            sg.output(tuple([op_out] + sg_phs))
        else:
            sg.output(tuple([None] + sg_phs))
        sg_name = "_test_cd_sg"
        setattr(mod, sg_name, fx.GraphModule(torch.nn.Module(), sg))
        g.owning_module = mod  # type: ignore[assignment]

        sg_node = g.get_attr(sg_name)
        sg_node.meta = {}

        n_outputs = 1 + len(deps)
        cd = g.call_function(
            cd_hop,
            args=(tuple(addl_nodes), sg_node, *dep_nodes),
        )
        cd.meta = {"val": tuple(t for _ in range(n_outputs))}

        for i in range(n_outputs):
            gi = g.call_function(operator.getitem, (cd, i))
            gi.meta = {"val": t}

        if fwd_uses is None:
            fwd_uses = [i for i, d in enumerate(deps) if d == "fwd"]
        fwd_getitems = []
        for gi_node in list(g.nodes):
            if (
                gi_node.op == "call_function"
                and gi_node.target is operator.getitem
                and gi_node.args[0] is cd
            ):
                idx = gi_node.args[1]
                if idx == 0 and subgraph_op is not None:
                    fwd_getitems.append(gi_node)
                elif idx - 1 in fwd_uses:
                    fwd_getitems.append(gi_node)

        if len(fwd_getitems) == 0:
            fwd_out = g.call_function(torch.ops.aten.sin.default, (fwd_in,))
        elif len(fwd_getitems) == 1:
            fwd_out = g.call_function(torch.ops.aten.relu.default, (fwd_getitems[0],))
        else:
            fwd_out = fwd_getitems[0]
            for gi in fwd_getitems[1:]:
                fwd_out = g.call_function(torch.ops.aten.add.Tensor, (fwd_out, gi))
        fwd_out.meta = {"val": t}
        g.output((fwd_out,))

        return (
            g,
            [fwd_in],
            [fwd_out],
            [DummyAOTOutput(0)],
            mod,
        )

    def _extract_and_check(self, graph, inputs, outputs, descs):
        # [FUNCTION DOC] HELPER: Extract a forward graph from the inputs and outputs while
        # ignoring must-be-in-forward-backward tags, lint it, and return the new graph with the
        # list of control dependency nodes and the list of getitem nodes.
        from torch._inductor.fx_passes.control_dependencies import (
            control_deps as cd_hop,
        )

        new_graph = _extract_graph_with_inputs_outputs(
            graph, inputs, outputs, descs, ignore_must_be_in_fw_bw=True
        )
        new_graph.lint()
        cd_nodes = [
            n for n in new_graph.nodes if n.op == "call_function" and n.target is cd_hop
        ]
        gi_nodes = [
            n
            for n in new_graph.nodes
            if n.op == "call_function" and n.target is operator.getitem
        ]
        return new_graph, cd_nodes, gi_nodes

    def test_extract_graph_control_deps_mixed_validity(self):
        # [FUNCTION DOC] Check extraction keeping only the forward dependency from mixed
        # dependencies: one control node with one dependency and one getitem with index one.
        g, ins, outs, descs, _ = self._build_control_deps_graph(deps=["fwd", "bw"])
        _, cd_nodes, gi_nodes = self._extract_and_check(g, ins, outs, descs)
        self.assertEqual(len(cd_nodes), 1)
        self.assertEqual(len(cd_nodes[0].args) - 2, 1)
        self.assertEqual(len(gi_nodes), 1)
        self.assertEqual(gi_nodes[0].args[1], 1)

    def test_extract_graph_control_deps_all_invalid(self):
        # [FUNCTION DOC] Check extraction when all dependencies are backward-only drops the
        # control dependency node.
        g, ins, outs, descs, _ = self._build_control_deps_graph(deps=["bw"])
        _, cd_nodes, _ = self._extract_and_check(g, ins, outs, descs)
        self.assertEqual(len(cd_nodes), 0)

    def test_extract_graph_control_deps_getitem_index_remap(self):
        # [FUNCTION DOC] Check getitem indices are remapped when the invalid dependency
        # comes first.
        g, ins, outs, descs, _ = self._build_control_deps_graph(deps=["bw", "fwd"])
        _, cd_nodes, gi_nodes = self._extract_and_check(g, ins, outs, descs)
        self.assertEqual(len(cd_nodes), 1)
        self.assertEqual(len(gi_nodes), 1)
        self.assertEqual(gi_nodes[0].args[1], 1)

    def test_extract_graph_control_deps_additional_deps_filtered(self):
        # [FUNCTION DOC] Check additional ordering dependencies of the wrong side are filtered,
        # leaving one ordering dependency and one data dependency.
        g, ins, outs, descs, _ = self._build_control_deps_graph(
            deps=["fwd", "bw"], additional_deps=["fwd"]
        )
        _, cd_nodes, _ = self._extract_and_check(g, ins, outs, descs)
        self.assertEqual(len(cd_nodes), 1)
        self.assertEqual(len(cd_nodes[0].args[0]), 1)
        self.assertEqual(len(cd_nodes[0].args) - 2, 1)

    def test_extract_graph_control_deps_preserves_subgraph_operation(self):
        # [FUNCTION DOC] Check the operation inside the control dependency subgraph is preserved
        # after extraction and the output indices of getitems are zero and one but not two.
        g, ins, outs, descs, mod = self._build_control_deps_graph(
            deps=["fwd", "bw"], subgraph_op=torch.ops.aten.abs.default
        )
        _, cd_nodes, gi_nodes = self._extract_and_check(g, ins, outs, descs)
        self.assertEqual(len(cd_nodes), 1)
        sg_mod = getattr(mod, cd_nodes[0].args[1].target)
        sg_ops = [n for n in sg_mod.graph.nodes if n.op == "call_function"]
        self.assertEqual(len(sg_ops), 1)
        self.assertEqual(sg_ops[0].target, torch.ops.aten.abs.default)
        gi_indices = sorted(n.args[1] for n in gi_nodes)
        self.assertIn(0, gi_indices)
        self.assertIn(1, gi_indices)
        self.assertNotIn(2, gi_indices)

    def _build_backward_extract_graph(self, dep_labels, cd_tag, subgraph_op):
        # [FUNCTION DOC] HELPER: Build a graph with a control dependency node over placeholders,
        # a subgraph applying the given op, a partitioner tag on the node, and an output that
        # uses only getitems for the placeholders labeled as backward inputs. Returns the graph,
        # backward inputs, outputs, descriptors and the module.
        """Build a graph for testing backward extraction of control_deps.

        Args:
            dep_labels: list of ("bw_input" | "not_input") per dep.
                "bw_input" deps become backward extraction inputs.
                "not_input" deps are placeholders NOT in the input list.
            cd_tag: partitioner_tag for the control_deps node.
            subgraph_op: op for the sync subgraph (makes _control_deps_has_sync_op True).
        Returns: (graph, bw_inputs, bw_output_node, descs, mod)
        """
        import torch.fx as fx
        from torch._functorch._aot_autograd.descriptors import DummyAOTOutput
        from torch._inductor.fx_passes.control_dependencies import (
            control_deps as cd_hop,
        )

        g = fx.Graph()
        mod = torch.nn.Module()
        t = torch.randn(4)

        dep_nodes = []
        bw_inputs = []
        for i, label in enumerate(dep_labels):
            ph = g.placeholder(f"dep_{i}")
            ph.meta = {"val": t}
            dep_nodes.append(ph)
            if label == "bw_input":
                bw_inputs.append(ph)

        sg = fx.Graph()
        sg_phs = [sg.placeholder(f"d{i}") for i in range(len(dep_labels))]
        op_out = sg.call_function(subgraph_op, (sg_phs[0],))
        sg.output(tuple([op_out] + sg_phs))
        sg_name = "_test_cd_sg"
        setattr(mod, sg_name, fx.GraphModule(torch.nn.Module(), sg))
        g.owning_module = mod  # type: ignore[assignment]

        sg_node = g.get_attr(sg_name)
        sg_node.meta = {}

        n_outputs = 1 + len(dep_labels)
        cd = g.call_function(cd_hop, args=((), sg_node, *dep_nodes))
        cd.meta = {
            "val": tuple(t for _ in range(n_outputs)),
            "partitioner_tag": cd_tag,
        }

        getitems = []
        for i in range(n_outputs):
            gi = g.call_function(operator.getitem, (cd, i))
            gi.meta = {"val": t}
            getitems.append(gi)

        # Build an output that uses only bw_input getitems (indices 1-based).
        bw_input_set = {id(n) for n in bw_inputs}
        used = [
            getitems[i + 1]
            for i, dep in enumerate(dep_nodes)
            if id(dep) in bw_input_set
        ]
        if not used:
            out = g.call_function(torch.ops.aten.sin.default, (bw_inputs[0],))
        elif len(used) == 1:
            out = g.call_function(torch.ops.aten.relu.default, (used[0],))
        else:
            out = used[0]
            for u in used[1:]:
                out = g.call_function(torch.ops.aten.add.Tensor, (out, u))
        out.meta = {"val": t}
        g.output((out,))

        return g, bw_inputs, [out], [DummyAOTOutput(0)], mod

    def test_backward_extract_eliminates_fwd_sync_fully_valid(self):
        # [FUNCTION DOC] Check backward extraction removes a forward-tagged synchronization
        # control dependency node entirely when all dependencies are backward inputs, leaving
        # two placeholders.
        """All deps are backward inputs -- forward sync control_deps
        should be eliminated entirely, getitems resolve to deps."""
        from torch._functorch._aot_autograd.streams import _SYNC_OPS
        from torch._inductor.fx_passes.control_dependencies import (
            control_deps as cd_hop,
        )

        g, ins, outs, descs, _ = self._build_backward_extract_graph(
            dep_labels=["bw_input", "bw_input"],
            cd_tag="is_forward",
            subgraph_op=_SYNC_OPS[0],
        )
        new_graph = _extract_graph_with_inputs_outputs(
            g, ins, outs, descs, subgraph="backward"
        )
        new_graph.lint()
        cd_nodes = [
            n for n in new_graph.nodes if n.op == "call_function" and n.target is cd_hop
        ]
        self.assertEqual(len(cd_nodes), 0)
        phs = [n for n in new_graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(phs), 2)
        out_node = next(n for n in new_graph.nodes if n.op == "output")
        self.assertIsNotNone(out_node)

    def test_backward_extract_eliminates_fwd_sync_mixed_validity(self):
        # [FUNCTION DOC] Check backward extraction removes a forward-tagged synchronization
        # control dependency node when only some dependencies are backward inputs, leaving
        # one placeholder.
        """Some deps are backward inputs, some are not -- forward sync
        control_deps eliminated, valid getitems resolve to deps."""
        from torch._functorch._aot_autograd.streams import _SYNC_OPS
        from torch._inductor.fx_passes.control_dependencies import (
            control_deps as cd_hop,
        )

        g, ins, outs, descs, _ = self._build_backward_extract_graph(
            dep_labels=["bw_input", "not_input"],
            cd_tag="is_forward",
            subgraph_op=_SYNC_OPS[0],
        )
        new_graph = _extract_graph_with_inputs_outputs(
            g, ins, outs, descs, subgraph="backward"
        )
        new_graph.lint()
        cd_nodes = [
            n for n in new_graph.nodes if n.op == "call_function" and n.target is cd_hop
        ]
        self.assertEqual(len(cd_nodes), 0)
        phs = [n for n in new_graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(phs), 1)

    def test_backward_extract_preserves_bwd_sync_control_deps(self):
        # [FUNCTION DOC] Check a backward-tagged synchronization control dependency node is
        # preserved in extraction and its subgraph keeps one operation.
        """Backward-tagged sync control_deps must stay in the backward graph."""
        from torch._functorch._aot_autograd.streams import _SYNC_OPS
        from torch._inductor.fx_passes.control_dependencies import (
            control_deps as cd_hop,
        )

        g, ins, outs, descs, _ = self._build_backward_extract_graph(
            dep_labels=["bw_input"],
            cd_tag="must_be_in_backward",
            subgraph_op=_SYNC_OPS[0],
        )
        new_graph = _extract_graph_with_inputs_outputs(
            g, ins, outs, descs, subgraph="backward"
        )
        new_graph.lint()
        cd_nodes = [
            n for n in new_graph.nodes if n.op == "call_function" and n.target is cd_hop
        ]
        self.assertEqual(len(cd_nodes), 1)
        sg_node = cd_nodes[0].args[1]
        sg_mod = getattr(
            g.owning_module,
            sg_node.target,  # type: ignore[union-attr]
        )
        sg_ops = [n for n in sg_mod.graph.nodes if n.op == "call_function"]
        self.assertEqual(len(sg_ops), 1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_control_deps_mixed_fwd_bw_deps_e2e(self):
        # [FUNCTION DOC] CUDA only. Compile with aot_eager a function using side streams, events
        # and stream waits; forward and backward must not crash when the stream wait collects
        # forward dependencies.
        """Forward compilation and backward must not crash when
        wait_stream's control_deps collects forward deps."""

        def fn(x, w):
            # [FUNCTION DOC] HELPER: Run a matrix product on a side stream, record and wait on
            # an event, return the product.
            s1 = torch.cuda.Stream()
            s1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s1):
                h = x @ w
            ev = torch.cuda.Event()
            ev.record(s1)
            ev.wait()
            return h

        w = torch.randn(64, 64, device="cuda", requires_grad=True)
        x = torch.randn(4, 64, device="cuda", requires_grad=True)
        compiled = torch.compile(fn, backend="aot_eager")
        out = compiled(x, w)
        out.sum().backward()


class TestAOTDispatch(AOTTestCase):
    # [CLASS DOC] Tests of AOT dispatch with tensor subclass inputs and outputs (TwoTensor).
    # Tests to add cases for (non-exhaustive list, mostly for my notes):
    # - subclass / mode introduced in the middle of the compiled fn
    # - various input mutation / intermediate base tests
    # - input mutation that changes a tensor into a subclass
    # - metadata mutation? (TBD)
    # - guard tests (fw guards *and* bw guards)
    # - subclass test involving _indices_of_inps_to_detach
    def test_aminmax_out_dtype_mismatch_errors(self):
        # [FUNCTION DOC] Check the aminmax op with out tensors of the wrong dtype raises
        # RuntimeError in eager and TorchRuntimeError under compile with the same message.
        def f(inp, out_min, out_max):
            # [FUNCTION DOC] HELPER: Call aminmax along the last dimension writing into two
            # output tensors.
            return torch.aminmax(inp, dim=-1, out=(out_min, out_max))

        inp = torch.rand(10, 10)
        out_min = torch.empty(10, dtype=torch.float64)
        out_max = torch.empty(10, dtype=torch.float64)

        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            f(inp, out_min, out_max)

        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "Expected out tensor to have dtype",
        ):
            compiled_f(inp, out_min, out_max)

    def test_aot_dispatch_simple(self):
        # [FUNCTION DOC] Check aot_function with one TwoTensor input and one plain input:
        # outputs and gradients match eager and the forward graph does subclass ops twice
        # for the subclass path and once for the plain path; the backward graph has four
        # dense outputs.
        # a is a subclass, b is not
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiply a by six, divide b by two, return the sum.
            aa = torch.mul(a, 6)
            bb = torch.div(b, 2)
            return aa + bb

        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3, requires_grad=True)

        a1_test = a1_ref.detach().clone().requires_grad_(True)
        a2_test = a2_ref.detach().clone().requires_grad_(True)
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.detach().clone().requires_grad_(True)

        fw_graph_cell = [None]
        bw_graph_cell = [None]

        compiled_f = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)

        # Output is a TwoTensor (check both inner tensors)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        out_ref.sum().backward()
        out_test.sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref.grad.a, a_test.grad.a)
        self.assertEqual(a_ref.grad.b, a_test.grad.b)
        self.assertEqual(b_ref.grad.a, b_test.grad.a)
        self.assertEqual(b_ref.grad.b, b_test.grad.b)

        # Important pieces of the graph:
        # - mul() and div() show up twice, because we called them on a TwoTensor
        # - add() shows up once, because we called it on a plain Tensor
        # - The user forward() fn returns 1 output (the result of add),
        #   while the graph itself returns two outputs (add, add_1)
        # - add, add_1 correspond to the two inner dense tensors that will be wrapped
        # - into a single TwoTensor output.
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    mul = torch.ops.aten.mul.Tensor(primals_1, 6);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 6);  primals_2 = None
    div = torch.ops.aten.div.Tensor(primals_3, 2);  primals_3 = None
    add = torch.ops.aten.add.Tensor(mul, div);  mul = None
    add_1 = torch.ops.aten.add.Tensor(mul_1, div);  mul_1 = div = None
    return (add, add_1)""",
        )

        # Important pieces of the graph:
        # - 4 total dense outputs.
        #   This corresponds to the fact that each user fwd input (a, b)
        #   will get a gradient that is a TwoTensor subclass,
        #   so (mul_2, mul_3) will be wrapped into a.grad
        #   and (div_1, div_2) will be wrapped into b.grad
        # - 4 total dense outputs,
        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, tangents_1, tangents_2):
    div_1 = torch.ops.aten.div.Tensor(tangents_1, 2)
    div_2 = torch.ops.aten.div.Tensor(tangents_2, 2)
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, 6);  tangents_1 = None
    mul_3 = torch.ops.aten.mul.Tensor(tangents_2, 6);  tangents_2 = None
    return (mul_2, mul_3, div_1, div_2)""",
        )

    def test_aot_dispatch_inference(self):
        # [FUNCTION DOC] Inference variant of the previous test (no gradients): outputs of the
        # subclass compiled function match eager.
        # a is a subclass, b is not
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiply a by six, divide b by two, return the sum.
            aa = torch.mul(a, 6)
            bb = torch.div(b, 2)
            return aa + bb

        a1_ref = torch.ones(3, 3)
        a2_ref = torch.ones(3, 3)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3)

        a1_test = a1_ref.clone()
        a2_test = a2_ref.clone()
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.clone()

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)

        # Output is a TwoTensor (check both inner tensors)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

    def test_aot_dispatch_incorrect_backward(self):
        # [FUNCTION DOC] Check that when the backward gradient for the second output is a
        # subclass but was assumed plain at compile time, backward raises RuntimeError about
        # guessing the subclass metadata incorrectly.
        # a is a subclass, b is not
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return a divided subclass value and a plain doubled value.
            aa = torch.mul(a, 2)
            bb = torch.add(b, 3)
            out_subclass = torch.div(aa, bb)
            out_reg = torch.add(b, b)
            # When creating the joint, we assume that the second grad_out
            # is not a subclass.
            # In the below test case though, we end up being wrong.
            # This would require re-tracing and recompiling the backward.
            return out_subclass, out_reg

        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3, requires_grad=True)

        a1_test = a1_ref.detach().clone().requires_grad_(True)
        a2_test = a2_ref.detach().clone().requires_grad_(True)
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.detach().clone().requires_grad_(True)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        # First out is a TwoTensor, second is an ordinary tensor
        self.assertEqual(out_ref[0].a, out_test[0].a)
        self.assertEqual(out_ref[0].b, out_test[0].b)
        self.assertEqual(out_ref[1], out_test[1])

        # We compiled our graph assuming type(grad_out[1]) is torch.Tensor,
        # but we were wrong: in the below tests, it is a subclass.
        # This will eventually require a repartition + recompile
        with self.assertRaisesRegex(
            RuntimeError,
            """
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
""",
        ):
            (out_test[0] + out_test[1]).sum().backward()

    def test_aot_dispatch_output_alias(self):
        # [FUNCTION DOC] Check a function returning a view of a TwoTensor input together with
        # a product of plain and subclass tensors: outputs and gradients (both TwoTensor)
        # match eager.
        # a is a tensor, b is a TwoTensor
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return a same-shape view of b and the product a times b.
            return b.view(b.shape), a * b

        b1_ref = torch.ones(3, 3, requires_grad=True)
        b2_ref = torch.ones(3, 3, requires_grad=True)
        b_ref = TwoTensor(b1_ref, b2_ref)
        a_ref = torch.ones(3, 3, requires_grad=True)

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test = TwoTensor(b1_test, b2_test)
        a_test = a_ref.detach().clone().requires_grad_(True)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref1, out_ref2 = f(a_ref, b_ref)
        out_test1, out_test2 = compiled_f(a_test, b_test)
        self.assertEqual(out_ref1, out_test1)
        self.assertEqual(out_ref2.a, out_test2.a)
        self.assertEqual(out_ref2.b, out_test2.b)

        (out_ref1 + out_ref2).sum().backward()
        (out_test1 + out_test2).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref.grad.a, a_test.grad.a)
        self.assertEqual(a_ref.grad.b, a_test.grad.b)
        self.assertEqual(b_ref.grad.a, b_test.grad.a)
        self.assertEqual(b_ref.grad.b, b_test.grad.b)

    @torch._functorch.config.patch(
        {
            "disable_guess_zero_tangent_for_mutated_input_subclass": True,
        }
    )
    def test_aot_dispatch_input_mutation(self):
        # [FUNCTION DOC] With the config for not guessing zero tangents for mutated input
        # subclasses, check input mutations of a TwoTensor and a plain tensor are visible after
        # the compiled call, metadata matches, and gradients match eager.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Double a and triple b in place and return their sum.
            a.mul_(2)
            b.mul_(3)
            return a + b

        b1_ref = torch.ones(3, 3, requires_grad=True)
        b2_ref = torch.ones(3, 3, requires_grad=True)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = torch.ones(3, 3, requires_grad=True)
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)
        self.assertTensorMetadataEqual(a_test, a_ref)
        self.assertTensorMetadataEqual(b_test, b_ref)
        self.assertTensorMetadataEqual(out_test, out_ref)

        # NOTE: we need to use b in our gradient compute. Otherwise we will need to recompile the backward.
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    def test_aot_dispatch_input_metadata_mutation(self):
        # [FUNCTION DOC] Check metadata-only input mutations (in-place transpose on a plain
        # tensor, unsqueeze on a TwoTensor) under aot_function: outputs match eager, mutated
        # inputs and their metadata match, and gradients of both inputs are TwoTensor values
        # equal to eager.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Transpose a in place, unsqueeze b in place, return
            # their sum.
            a.t_()
            b.unsqueeze_(0)
            return a + b

        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()
            .requires_grad_(True)
        )
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)
        self.assertTensorMetadataEqual(a_test, a_ref)
        self.assertTensorMetadataEqual(b_test, b_ref)
        self.assertTensorMetadataEqual(out_test, out_ref)

        # NOTE: we need to use b in our gradient compute. Otherwise we will need to recompile the backward.
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    # NB: Mixed data and metadata mutations still need tangent metadata support.
    @unittest.expectedFailure
    def test_aot_dispatch_input_data_and_metadata_mutation(self):
        # [FUNCTION DOC] Expected failure. Mixed metadata and data mutations on plain and
        # TwoTensor inputs still need tangent metadata support, so this test is known not
        # to pass.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Transpose a, unsqueeze b, then double a and triple b in
            # place and return their sum.
            a.t_()
            b.unsqueeze_(0)
            a.mul_(2)
            b.mul_(3)
            return a + b

        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()
            .requires_grad_(True)
        )
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        # NOTE: we need to use b in our gradient compute. Otherwise we will need to recompile the backward.
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    @torch._functorch.config.patch(
        {
            "disable_guess_zero_tangent_for_mutated_input_subclass": True,
        }
    )
    def test_aot_dispatch_input_mutation_and_output_alias(self):
        # [FUNCTION DOC] With the config disabling zero-tangent guessing for mutated subclass
        # inputs: check data mutation of inputs together with an output that is a view of a
        # mutated TwoTensor input; outputs, mutated inputs and gradients of the plain input
        # match eager.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Double a, triple b in place, return a view of b and the sum
            # of both.
            a.mul_(2)
            b.mul_(3)
            return b.view(b.shape), a + b

        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()
            .requires_grad_(True)
        )
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref1, out_ref2 = f(a_ref, b_ref)
        out_test1, out_test2 = compiled_f(a_test, b_test)
        self.assertEqual(out_ref1.a, out_test1.a)
        self.assertEqual(out_ref1.b, out_test1.b)
        self.assertEqual(out_ref2.a, out_test2.a)
        self.assertEqual(out_ref2.b, out_test2.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        (out_ref1 * out_ref2).sum().backward()
        (out_test1 * out_test2).sum().backward()
        # Both grad_inputs are TwoTensors
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)

    def test_aot_dispatch_output_requires_grad_in_no_grad(self):
        # [FUNCTION DOC] Check that outputs created with grad enabled inside a no_grad region
        # keep requires_grad like eager for inputs both requiring and not requiring grad, and
        # that gradients match when backward is run later.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the sine of x and the cosine of x computed with
            # grad enabled.
            out1 = x.sin()
            with torch.enable_grad():
                out2 = x.cos()
            return out1, out2

        inp_fns = [
            lambda: torch.ones(10, requires_grad=True),
            lambda: torch.ones(10, requires_grad=False),
        ]

        compiled_f = aot_function(fn, nop)
        for inp_fn in inp_fns:
            with torch.no_grad():
                ref_x = inp_fn()
                ref_out = fn(ref_x)
                x = inp_fn()
                out = compiled_f(x)
                for r, o in zip(ref_out, out):
                    self.assertEqual(r.requires_grad, o.requires_grad)
            if ref_x.requires_grad:
                with torch.enable_grad():
                    (ref_out[0] + ref_out[1]).sum().backward()
                    (out[0] + out[1]).sum().backward()
                    self.assertEqual(ref_x.grad, x.grad)
                    if not torch.allclose(ref_x.grad, x.grad, atol=1e-3, rtol=1e-3):
                        raise AssertionError("ref_x.grad and x.grad not allclose")

    def test_aot_dispatch_output_requires_grad_in_no_grad_views(self):
        # [FUNCTION DOC] Check view outputs keep requires_grad in no_grad mode: the inference
        # compiler graph is used and requires_grad flags of both outputs equal eager.
        # view-type ops preserve requires_grad even in no_grad.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return a flattened view and the sine of the input.
            return x.view(-1), x.sin()

        inference_graph_cell = [None]
        inference_compiler = make_boxed_compiler(
            partial(extract_graph, graph_cell=inference_graph_cell)
        )
        compiled_fn = aot_function(fn, nop, inference_compiler=inference_compiler)

        inp_x0 = torch.ones(2, 3, requires_grad=True)
        # Clone in no_grad will make requires_grad=False tensors, keep clone outside of no_grad
        ref_x0 = inp_x0.clone()
        x0 = inp_x0.clone()
        with torch.no_grad():
            ref_out1, ref_out2 = fn(ref_x0)

            out1, out2 = compiled_fn(x0)
            # Assert that we executed inference graph
            self.assertTrue(inference_graph_cell[0] is not None)

            self.assertEqual(ref_out1.requires_grad, out1.requires_grad)
            self.assertEqual(ref_out2.requires_grad, out2.requires_grad)


class GradsNoForceContiguousContextManager(ContextDecorator):
    # [CLASS DOC] Context manager and decorator that registers a temporary operator library with
    # an effectful pass-through op whose backward calls another effectful op that records the
    # memory format and stride of the gradient it receives. State: the library handle, a counter
    # per memory format (channels last, contiguous) and a list of tangent strides. Used to check
    # how gradient memory formats are predicted.
    def __enter__(self):
        # [FUNCTION DOC] HELPER: Create the temporary library; define the two ops, register CPU,
        # CUDA and meta implementations (the logging implementation counts the suggested memory
        # format of its input and records its stride), register a backward that calls the
        # logging op and returns a clone, mark both ops ordered effectful, reset counters;
        # returns the manager itself.
        # noqa: SCOPED_LIBRARY
        self.lib = torch.library.Library("_test_aotdispatch_lib", "FRAGMENT")
        self.d = {
            torch.channels_last: 0,
            torch.contiguous_format: 0,
        }
        self.tangent_strides = []

        self.lib.define("log_tangents_memory_format(Tensor x) -> Tensor")
        self.lib.define("log_tangents_memory_format_log(Tensor x) -> Tensor")

        def log_tangents_memory_format_impl(a):
            # [FUNCTION DOC] HELPER: Clone implementation of the op.
            return a.clone()

        def log_tangents_memory_format_meta(a):
            # [FUNCTION DOC] HELPER: Meta implementation cloning the input.
            return a.clone()

        def log_tangents_memory_format_log_impl(x):
            # [FUNCTION DOC] HELPER: Logging implementation: increment the counter for the
            # suggested memory format of the input, append its strides to the list, return
            # a clone.
            self.d[torch._prims_common.suggest_memory_format(x)] += 1
            self.tangent_strides.append(x.stride())
            return x.clone()

        def log_tangents_memory_format_log_meta(a):
            # [FUNCTION DOC] HELPER: Meta implementation cloning the input.
            return a.clone()

        for backend in ["CPU", "CUDA"]:
            self.lib.impl(
                "log_tangents_memory_format", log_tangents_memory_format_impl, backend
            )
            self.lib.impl(
                "log_tangents_memory_format_log",
                log_tangents_memory_format_log_impl,
                backend,
            )

        self.lib.impl(
            "log_tangents_memory_format", log_tangents_memory_format_meta, "Meta"
        )
        self.lib.impl(
            "log_tangents_memory_format_log",
            log_tangents_memory_format_log_meta,
            "Meta",
        )

        def log_tangents_memory_format_bwd(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward of the op: call the logging op on the gradient and
            # return a clone of it.
            torch.ops._test_aotdispatch_lib.log_tangents_memory_format_log(grad)
            return grad.clone()

        torch.library.register_autograd(
            "_test_aotdispatch_lib::log_tangents_memory_format",
            log_tangents_memory_format_bwd,
            lib=self.lib,
        )

        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

        _register_effectful_op(
            torch.ops._test_aotdispatch_lib.log_tangents_memory_format.default,
            _EffectType.ORDERED,
        )
        _register_effectful_op(
            torch.ops._test_aotdispatch_lib.log_tangents_memory_format_log.default,
            _EffectType.ORDERED,
        )

        return self

    def __exit__(self, type, value, tb):
        # [FUNCTION DOC] HELPER: Destroy the temporary library; never suppresses exceptions
        # (returns false).
        self.lib._destroy()
        return False

    def reset_counters(self):
        # [FUNCTION DOC] HELPER: Reset the per-format counters to zero.
        self.d = {
            torch.channels_last: 0,
            torch.contiguous_format: 0,
        }


class TestAOTModuleSimplified(AOTTestCase):
    # [CLASS DOC] Tests of aot_module_simplified and the tangent handling in module compilation.
    def test_aot_module_simplified(self):
        # [FUNCTION DOC] Compile a module with a linear layer through aot_module_simplified with
        # no-op compilers; outputs and input gradients are compared with eager and
        # AssertionError is raised if not close.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer from 20 to 30 features adding a
            # second input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Return a one-element tuple of the linear output plus y.
                return (self.linear(x) + y,)

        mod = MockModule()
        mod.zero_grad()

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]
        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]

        ref = mod(*inputs)
        ref[0].sum().backward()

        compiled_f = aot_module_simplified(mod, cloned_inputs, nop)
        mod.zero_grad()
        res = compiled_f(*cloned_inputs)
        res[0].sum().backward()

        if not torch.allclose(ref[0], res[0]):
            raise AssertionError("ref[0] and res[0] not allclose")
        if not torch.allclose(inputs[0].grad, cloned_inputs[0].grad):
            raise AssertionError(
                "inputs[0].grad and cloned_inputs[0].grad not allclose"
            )
        if not torch.allclose(inputs[1].grad, cloned_inputs[1].grad):
            raise AssertionError(
                "inputs[1].grad and cloned_inputs[1].grad not allclose"
            )

    def test_aot_module_simplified_dynamic(self):
        # [FUNCTION DOC] Same as the previous test with fake inputs under a symbolic shape
        # environment; also asserts the recorded shape guards are two equalities for the
        # static sizes.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer from 20 to 30 features adding a
            # second input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Return a one-element tuple of the linear output plus y.
                return (self.linear(x) + y,)

        mod = MockModule()

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)

        inputs = [x, y]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)

        ref = mod(*inputs)
        ref[0].sum().backward()

        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        res = compiled_f(*cloned_inputs)
        res[0].sum().backward()

        self.assertExpectedInline(
            shape_env.format_guards(),
            """\
 - Eq(s49, 20)
 - Eq(s70, 30)""",
        )

        if not torch.allclose(ref[0], res[0]):
            raise AssertionError("ref[0] and res[0] not allclose")
        if not torch.allclose(inputs[0].grad, cloned_inputs[0].grad):
            raise AssertionError(
                "inputs[0].grad and cloned_inputs[0].grad not allclose"
            )
        if not torch.allclose(inputs[1].grad, cloned_inputs[1].grad):
            raise AssertionError(
                "inputs[1].grad and cloned_inputs[1].grad not allclose"
            )

    # https://github.com/pytorch/pytorch/issues/105327
    def test_lift_fresh_copy_in_graph(self):
        # [FUNCTION DOC] Regression for issue 105327: a module building a constant tensor and
        # passing it through lift fresh copy compiles with fake inputs and gives the same result
        # as eager.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating a constant tensor, copying via lift fresh copy
            # and multiplying the input by it.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiply the input by a fresh copy of a constant
                # tensor; returns a one-element tuple.
                _tensor_constant0 = torch.tensor([1])
                lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(
                    _tensor_constant0
                )
                y = x.mul(lift_fresh_copy)
                return (y,)

        mod = MyMod()
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = torch.ones(4, requires_grad=True)
        inputs = [x]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)

        out_ref = mod(x)
        out_test = compiled_f(x)
        self.assertEqual(out_ref[0].detach(), out_test[0].detach())

    def test_inference_python_dispatcher(self):
        # [FUNCTION DOC] Check compiling a module with a bilinear upsample (extracted from a
        # unet) with fake inputs that do not require grad works under the python dispatcher.
        # Extracted from unet
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bilinear upsample layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the upsample layer.
                super().__init__()
                self.upsample = torch.nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the upsample in a tuple.
                return (self.upsample(x),)

        mod = MockModule()
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = torch.randn(2, 512, 40, 59)  # NB: must not require grad
        inputs = [x]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        aot_module_simplified(mod, fake_inputs, nop)

    def test_aot_module_simplified_preserves_stack_trace(self):
        # [FUNCTION DOC] Check stack traces recorded when tracing a module with the fx tracer
        # survive into the graphs given to forward and backward compilers: each call node must
        # have a stack trace mentioning this test file (AssertionError otherwise).
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying linear, add and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Linear of x, plus y, then relu, returned in a tuple.
                z = self.linear(x)
                z = z + y
                z = z.relu()
                return (z,)

        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(MockModule())
        mod = torch.fx.GraphModule(tracer.root, graph)

        for node in mod.graph.nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(node.stack_trace is not None)
            if "test_aotdispatch.py" not in node.stack_trace:
                raise AssertionError(
                    f"Expected 'test_aotdispatch.py' in stack trace, got: {node.stack_trace}"
                )

        def assert_compiler(gm: torch.fx.GraphModule, _):
            # [FUNCTION DOC] HELPER: Compiler asserting every non-placeholder, non-output node
            # has a stack trace containing this test file; returns the graph forward.
            for node in gm.graph.nodes:
                if node.op == "output" or node.op == "placeholder":
                    continue
                self.assertTrue(node.stack_trace is not None)
                if "test_aotdispatch.py" not in node.stack_trace:
                    raise AssertionError(
                        f"Expected 'test_aotdispatch.py' in stack trace, got: {node.stack_trace}"
                    )
            return gm.forward  # return a python callable

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]

        compiled_f = aot_module_simplified(
            mod, inputs, fw_compiler=assert_compiler, bw_compiler=assert_compiler
        )
        res = compiled_f(*inputs)
        res[0].sum().backward()

    def test_aot_joint_print_readable_marks_backward_stack_trace(self):
        # [FUNCTION DOC] Check printing the joint graph marks backward nodes with a comment
        # naming the forward node whose stack trace they come from: sets fake stack traces
        # on a few nodes and uses FileCheck on the readable joint graph captured by a
        # partition function.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module casting to float32 then summing the square.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Cast x to float32 and return the sum of squares in
                # a tuple.
                y = x.to(dtype=torch.float32)
                return ((y * y).sum(),)

        mod = torch.fx.symbolic_trace(MockModule())
        for node in mod.graph.nodes:
            if node.name == "to":
                node.stack_trace = (
                    '  File "/tmp/model.py", line 5, in forward\n'
                    "    y = x.to(dtype=torch.float32)\n"
                )
            elif node.name in {"mul", "sum_1"}:
                node.stack_trace = (
                    '  File "/tmp/model.py", line 6, in forward\n'
                    "    return ((y * y).sum(),)\n"
                )

        def compiler(gm: torch.fx.GraphModule, _):
            # [FUNCTION DOC] HELPER: Compiler returning the graph forward as a boxed function.
            return make_boxed_func(gm.forward)

        joint_graph: str | None = None

        def partition_fn(gm: torch.fx.GraphModule, inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Partition function that saves the readable text of the
            # joint graph in an enclosing variable and delegates to the default partition.
            nonlocal joint_graph
            joint_graph = gm.print_readable(print_output=False)
            return default_partition(gm, inputs, **kwargs)

        x = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        compiled_f = aot_module_simplified(
            mod,
            [x],
            fw_compiler=compiler,
            bw_compiler=compiler,
            partition_fn=partition_fn,
        )
        compiled_f(x)[0].backward()

        self.assertIsNotNone(joint_graph)
        fwd_to_comment = (
            "# File: /tmp/model.py:5 in forward, code: y = x.to(dtype=torch.float32)"
        )
        bw_to_comment = (
            "# Backward of forward node: File: /tmp/model.py:5 in forward, "
            "code: y = x.to(dtype=torch.float32)"
        )
        FileCheck().check(fwd_to_comment).check("_to_copy").check(bw_to_comment).check(
            "_to_copy_1"
        ).run(joint_graph)

    def test_aot_module_simplified_preserves_stack_trace_from_mutation(self):
        # [FUNCTION DOC] Check the stack trace of an input mutation copy node produced by
        # tracing a module that mutates a view of the input is preserved for the copy op;
        # compilers raise AssertionError if not.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module mutating a view of the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Double the first row of x in place and return x plus x
                # in a tuple.
                x_view = x[0]
                x_view.mul_(2)
                return (x + x,)

        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(MockModule())
        mod = torch.fx.GraphModule(tracer.root, graph)

        for node in mod.graph.nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(node.stack_trace is not None)
            if "test_aotdispatch.py" not in node.stack_trace:
                raise AssertionError("Expected 'test_aotdispatch.py' in stack trace")

        def assert_compiler(gm: torch.fx.GraphModule, _):
            # [FUNCTION DOC] HELPER: Compiler asserting a copy op exists and that every copy
            # op node has a stack trace mentioning the mutation source line; returns the
            # graph forward.
            if torch.ops.aten.copy_.default not in [x.target for x in gm.graph.nodes]:
                raise AssertionError("Expected aten.copy_.default in graph nodes")
            for node in gm.graph.nodes:
                if node.target == torch.ops.aten.copy_.default:
                    if "stack_trace" not in node.meta:
                        raise AssertionError("Expected 'stack_trace' in node.meta")
                    if "x_view.mul_(2)" not in node.meta["stack_trace"]:
                        raise AssertionError("Expected 'x_view.mul_(2)' in stack trace")
            return gm.forward  # return a python callable

        x = torch.randn(128, 20)
        inputs = [x]

        aot_module_simplified(
            mod,
            inputs,
            fw_compiler=assert_compiler,
            bw_compiler=assert_compiler,
            keep_inference_input_mutations=True,
        )

    def test_aot_module_simplified_fake_tensor_gm_raises(self):
        # [FUNCTION DOC] Check compiling a module that closes over a fake tensor raises
        # AssertionError with the message about an unexpected fake tensor.
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        real_x = torch.randn(4, requires_grad=True)
        fake_x = fake_mode.from_tensor(real_x)
        real_z = torch.randn(4)
        fake_z = fake_mode.from_tensor(real_z)

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a captured fake tensor to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add a captured fake tensor to x (the captured tensor
                # ends up as a constant in the traced graph).
                # Accessing a free variable fake tensor will look like a
                # constant to make_fx, and result in the tensor being traced
                # into the graph, which is an error condition.  Make sure we
                # report adequately in this case.
                return (x + fake_z,)

        with self.assertRaisesRegex(AssertionError, "Unexpected fake"):
            aot_module_simplified(MockModule(), (fake_x,), nop)

    def test_aot_test_subclasses_with_tensor_factories(self):
        # [FUNCTION DOC] Check a subclass whose factory functions create tensors compiles with
        # aot_eager and matches eager.
        from torch.testing._internal.common_subclass import SubclassWithTensorFactory

        inp = SubclassWithTensorFactory(torch.zeros(3, 5))

        def fn(x):
            # [FUNCTION DOC] HELPER: Double the input.
            return 2 * x

        ref_out = fn(inp)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(inp)
        self.assertEqual(ref_out, out)

    # Next several tests are related to issue:
    # https://github.com/pytorch/pytorch/issues/134644
    # AOTD tries to predict tangents for tracing ahead of time.
    # The first strategy was to coerce traced_tangents and runtime_tangents to be contiguous().
    # But for models working in channels_last memory format this will add additional contiguous() calls.
    # The fix is predicting tangents memory format to be similar to outputs memory format.
    # And coerce runtime tangents to that traced memory format.
    def test_grads_no_force_contiguous_dense(self):
        # [FUNCTION DOC] Check gradients are not forced contiguous for channels-last dense
        # inputs: compiles a conv module with a logged op under inductor and verifies exactly
        # one gradient seen in channels-last format and none contiguous.
        with GradsNoForceContiguousContextManager() as ctx:

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a convolution, mutating the second input and
                # returning conv output, its transpose, views of derived tensor and a doubled
                # contiguous input.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Create the convolution layer.
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 3, 3)

                def forward(self, x, y, cont_inp):
                    # [FUNCTION DOC] HELPER: Compute a derived tensor, double y in place, apply
                    # the convolution and the logging op, return five outputs.
                    z = y + 3
                    y.mul_(2)
                    r = self.conv(x)
                    r = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(r)
                    return (
                        r,
                        r.transpose(0, 1),
                        z.view(-1),
                        z.transpose(0, 1),
                        cont_inp * 2,
                    )

            m = M()
            m.to(memory_format=torch.channels_last)
            m.train()

            def dense_inps():
                # [FUNCTION DOC] HELPER: Build two channels-last grad-requiring inputs and one
                # contiguous input.
                return (
                    torch.randn(2, 3, 5, 5, requires_grad=True).to(
                        memory_format=torch.channels_last
                    ),
                    torch.randn(3, 2, 1, 1, requires_grad=True).to(
                        memory_format=torch.channels_last
                    ),
                    torch.randn(3, 2, 1, 1, requires_grad=True),
                )

            ref_inps = dense_inps()
            ref_outs = m(*ref_inps)
            ref_outs[0].sum().backward()

            ctx.reset_counters()
            inps = dense_inps()
            outs = torch.compile(m, backend="inductor", fullgraph=True)(*inps)
            outs[0].sum().backward()

            self.assertEqual(ctx.d[torch.channels_last], 1)
            self.assertEqual(ctx.d[torch.contiguous_format], 0)

    def test_grads_no_force_contiguous_subclass(self):
        # [FUNCTION DOC] Same check for a TwoTensor channels-last input: two gradients seen in
        # channels-last format and none contiguous.
        with GradsNoForceContiguousContextManager() as ctx:

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a convolution applying the logging op and
                # returning the result and y plus one.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Create the convolution layer.
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 3, 3)

                def forward(self, x, y):
                    # [FUNCTION DOC] HELPER: Apply the convolution and logging op, return it
                    # with y plus one.
                    r = self.conv(x)
                    r = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(r)
                    return r, y + 1

            m = M()
            m.to(memory_format=torch.channels_last)
            m.train()

            def inps_fn():
                # [FUNCTION DOC] HELPER: Build a TwoTensor of channels-last grad-requiring
                # tensors and a plain tensor.
                return (
                    TwoTensor(
                        torch.randn(2, 3, 5, 5, requires_grad=True).to(
                            memory_format=torch.channels_last
                        ),
                        torch.randn(2, 3, 5, 5, requires_grad=True).to(
                            memory_format=torch.channels_last
                        ),
                    ),
                    torch.randn(3, 2, requires_grad=True).clone(),
                )

            ref_outs = m(*inps_fn())
            ref_outs[0].sum().backward()

            ctx.reset_counters()
            mc = M()
            mc.to(memory_format=torch.channels_last)
            mc.train()
            outs = torch.compile(mc, backend="aot_eager", fullgraph=True)(*inps_fn())
            outs[0].sum().backward()

            self.assertEqual(ctx.d[torch.channels_last], 2)
            self.assertEqual(ctx.d[torch.contiguous_format], 0)

    def test_grads_no_force_contiguous_nested_subclass(self):
        # [FUNCTION DOC] Same check for doubly nested TwoTensor input: four gradients seen in
        # channels-last format and none contiguous.
        with GradsNoForceContiguousContextManager() as ctx:

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a convolution followed by the logging op.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Create the convolution layer.
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 3, 3)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Apply the convolution and the logging op and return
                    # the result.
                    r = self.conv(x)
                    r = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(r)
                    return r

            m = M()
            m.to(memory_format=torch.channels_last)
            m.train()

            def inps_fn(x):
                # [FUNCTION DOC] HELPER: Wrap four clones of a tensor in nested TwoTensor and
                # return as a one-element tuple.
                return (
                    TwoTensor(
                        TwoTensor(x.clone(), x.clone()), TwoTensor(x.clone(), x.clone())
                    ),
                )

            x = torch.randn(2, 3, 5, 5, requires_grad=True).to(
                memory_format=torch.channels_last
            )
            ref_inps = inps_fn(x)
            ref_outs = m(*ref_inps)
            ref_outs[0].sum().backward()

            ctx.reset_counters()

            mc = M()
            mc.to(memory_format=torch.channels_last)
            mc.train()

            x = torch.randn(2, 3, 5, 5, requires_grad=True).to(
                memory_format=torch.channels_last
            )
            inps = inps_fn(x)
            outs = torch.compile(mc, backend="aot_eager", fullgraph=True)(*inps)
            outs[0].sum().backward()
            self.assertEqual(ctx.d[torch.channels_last], 4)
            self.assertEqual(ctx.d[torch.contiguous_format], 0)

    def test_grads_no_force_contiguous_nested_tensor_tangent(self):
        # [FUNCTION DOC] Check gradient computation through a compiled clone of a jagged nested
        # tensor works (setting attributes on nested tensor tangents must not fail with an
        # attribute error).
        # NestedTensor setattr could fails with AttributeError for attr "_min_seqlen_tensor"
        # Adding test to verify that it is handled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Clone the input.
            return x.clone()

        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)

        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(nt)
        out_buffer = out.values()
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

    def test_wrong_guess_tangent_type(self):
        # [FUNCTION DOC] Check backward still works when the tangent type guessed at compile
        # time (TwoTensor from a TwoTensor input) is wrong by providing an explicit TwoTensor
        # gradient to backward on both eager and compiled outputs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Clone the input.
            return x.clone()

        ref_x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        ref_y = fn(ref_x)
        ref_y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        fn_comp = torch.compile(fn, fullgraph=True)

        x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y = fn_comp(x)
        y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        x2 = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y2 = fn_comp(x2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"""
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
Expected a .* tangent but got a plain Tensor.""",
        ):
            y2.backward(gradient=torch.randn(2, 3))

    def test_tangent_type_coercion(self):
        # [FUNCTION DOC] Check tangent type coercion in backward: a compiled clone of a
        # TwoTensor accepts a gradient that is a different subclass (wrapper subclass) and is
        # coerced, and a plain tensor input accepts a wrapper subclass gradient.
        def fn(x):
            # [FUNCTION DOC] HELPER: Clone the input.
            return x.clone()

        ref_y = fn(WrapperSubclass(torch.randn(2, 3, requires_grad=True)))
        ref_y.sum().backward()

        fn_comp = torch.compile(fn, fullgraph=True)

        x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y = fn_comp(x)
        y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        x2 = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y2 = fn_comp(x2)
        # Test coercion WrapperSubclass -> TwoTensor
        y2.backward(gradient=WrapperSubclass(torch.randn(2, 3)))

        y3 = torch.compile(fn, fullgraph=True)(torch.randn(2, 3, requires_grad=True))
        # Test coercion WrapperSubclass -> Tensor
        y3.backward(gradient=WrapperSubclass(torch.randn(2, 3)))

    @torch._inductor.config.patch({"freezing": True})
    def test_inductor_freezing_with_subclasses(self):
        # [FUNCTION DOC] With inductor freezing enabled, compile a module holding a TwoTensor
        # attribute and a plain tensor attribute under no_grad; expects no error.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a TwoTensor and a plain tensor, adding both to
            # a gathered input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the TwoTensor and plain tensor attributes.
                super().__init__()
                self.w = TwoTensor(torch.randn(3, 4), torch.randn(3, 4))
                self.wt = torch.randn(3, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Select rows of x in a permuted order and add
                # both attributes.
                return (
                    x.index_select(
                        dim=0, index=torch.tensor([0, 2, 1], dtype=torch.int64)
                    )
                    + self.w
                    + self.wt
                )

        m = M()
        inp = torch.randn(3, 4)
        with torch.no_grad():
            torch.compile(m, fullgraph=True)(inp)

    def test_rrelu(self):
        # [FUNCTION DOC] Check the randomized leaky relu in training mode compiles with inductor
        # in functional and in-place form without error.
        def fn(x):
            # [FUNCTION DOC] HELPER: Randomized relu in training mode.
            return torch.rrelu(x, training=True)

        def fn_(x):
            # [FUNCTION DOC] HELPER: Randomized relu in place in training mode, return
            # the input.
            torch.rrelu_(x, training=True)
            return x

        x = torch.randn(4, 4)
        torch.compile(fn, backend="inductor", fullgraph=True)(x)
        torch.compile(fn_, backend="inductor", fullgraph=True)(x)

    def test_layer_norm(self):
        # [FUNCTION DOC] Check layer norm over the last dimension compiled with aot_eager
        # exactly matches eager (zero tolerance).
        def fn(x):
            # [FUNCTION DOC] HELPER: Layer norm over the last dimension of size eight.
            return F.layer_norm(x, normalized_shape=(8,))

        x = torch.randn(2, 4, 8)
        eager = fn(x)
        aot_eager = torch.compile(backend="aot_eager")(fn)(x)
        self.assertEqual(eager, aot_eager, atol=0, rtol=0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_rms_norm(self):
        # [FUNCTION DOC] CUDA only. Check RMS norm over the last dimension compiled with
        # aot_eager exactly matches eager (only the CUDA version fails to decompose).
        # Only CUDA rms norm fails to be decomposed
        def fn(x):
            # [FUNCTION DOC] HELPER: RMS norm over the last dimension of size eight.
            return F.rms_norm(x, normalized_shape=(8,))

        x = torch.randn(2, 4, 8, device="cuda")
        eager = fn(x)
        aot_eager = torch.compile(backend="aot_eager")(fn)(x)
        self.assertEqual(eager, aot_eager, atol=0, rtol=0)

    def test_subclass_parameters(self):
        # [FUNCTION DOC] Check modules with nested subclass parameters: after unwrapping
        # subclass parameters, eager outputs and gradients are unchanged, and compiled outputs
        # and gradients match.
        class _M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a parameter that is a nested TwoTensor and adds
            # it to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the nested subclass parameter.
                super().__init__()
                self.p = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(torch.zeros(3, 4), torch.randn(3, 4)),
                        torch.ones(3, 4),
                    )
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add the parameter to the input.
                return x + self.p

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a plain parameter, a TwoTensor parameter and a
            # child module with a nested subclass parameter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the parameters and the child module.
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        torch.ones(3, 4),
                        TwoTensor(torch.randn(3, 4), torch.randn(3, 4)),
                    )
                )
                self._m = _M()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sum of the child module output, the input, twice the
                # plain parameter and the subclass parameter.
                return self._m(x) + x + 2 * self.p1 + self.p2

        m = M()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ref_out.sum().backward()
        m.zero_grad()

        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        unwrap_tensor_subclass_parameters(m)

        ref_x2 = ref_x.detach().clone()
        ref_out2 = m(ref_x2)
        self.assertEqual(ref_out2, ref_out)
        ref_out2.sum().backward()
        self.assertEqual(ref_x2.grad, ref_x.grad)
        m.zero_grad()

        x = ref_x.detach().clone()
        comp_fn = torch.compile(m, backend="aot_eager", fullgraph=True)
        out = comp_fn(x)
        self.assertEqual(ref_out, out)
        out.sum().backward()
        self.assertEqual(ref_x.grad, x.grad)

    def test_subclass_parameters_torture_case(self):
        # [FUNCTION DOC] Like the previous test for a deeply nested subclass parameter, using
        # only part of it in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a plain parameter and a deeply nested
            # TwoTensor parameter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the plain and deeply nested parameters.
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(
                            torch.ones(3, 4),
                            TwoTensor(torch.randn(3, 4), torch.randn(3, 4)),
                        ),
                        TwoTensor(
                            TwoTensor(torch.randn(3, 4), torch.randn(3, 4)),
                            TwoTensor(torch.ones(3, 4), torch.randn(3, 4)),
                        ),
                    )
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add the input, twice the plain parameter and the inner
                # component of the first part of the subclass parameter.
                return x + 2 * self.p1 + self.p2.a.b

        m = M()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ref_out.sum().backward()
        m.zero_grad()

        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        unwrap_tensor_subclass_parameters(m)

        ref_x2 = ref_x.detach().clone()
        ref_out2 = m(ref_x2)
        self.assertEqual(ref_out2, ref_out)
        ref_out2.sum().backward()
        self.assertEqual(ref_x2.grad, ref_x.grad)
        m.zero_grad()

        x = ref_x.detach().clone()
        comp_fn = torch.compile(m, backend="aot_eager", fullgraph=True)
        out = comp_fn(x)
        self.assertEqual(ref_out, out)
        out.sum().backward()
        self.assertEqual(ref_x.grad, x.grad)

    def test_rrelu_with_noise_mutation(self):
        # [FUNCTION DOC] Check randomized relu with a noise tensor in functional, mutating and
        # in-place forms compiles with inductor (bfloat16, negative inputs), the noise output is
        # updated (values below one) and backward works for the first two forms.
        def fn_functional(x):
            # [FUNCTION DOC] HELPER: Call the functional noise variant and return the result and
            # the noise output.
            noise = torch.ones_like(x)
            result, noise_out = torch.ops.aten.rrelu_with_noise_functional(
                x, noise, 0.2, 0.8, True
            )
            return result, noise_out

        def fn_mutation(x):
            # [FUNCTION DOC] HELPER: Call the noise op that mutates the noise tensor and return
            # the result and noise.
            noise = torch.ones_like(x)
            result = torch.ops.aten.rrelu_with_noise(x, noise, 0.2, 0.8, True)
            return result, noise

        def fn_inplace(x):
            # [FUNCTION DOC] HELPER: Call the in-place noise variant on x and return x and
            # the noise.
            noise = torch.ones_like(x, requires_grad=False)
            torch.ops.aten.rrelu_with_noise_(x, noise, 0.2, 0.8, True)
            return x, noise

        def _test_fn(fn, check_backward=True):
            # [FUNCTION DOC] HELPER: Run the function eagerly and compiled, check eager noise is
            # below one, optionally run backward on the compiled output, check compiled noise is
            # below one.
            x = -torch.abs(torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True))

            ref_y, ref_noise = fn(x)
            self.assertTrue(torch.all(ref_noise < torch.ones_like(ref_noise)).item())

            comp_y, comp_noise = torch.compile(fn, backend="inductor", fullgraph=True)(
                x
            )

            if check_backward:
                comp_y.sum().backward()
            self.assertTrue(torch.all(comp_noise < torch.ones_like(comp_noise)).item())

        _test_fn(fn_functional)
        _test_fn(fn_mutation)
        _test_fn(fn_inplace, check_backward=False)

    @parametrize("case", ["sum_expand", "computed_view_expand"])
    @patch("torch._functorch.config.guess_tangent_strides_as_outputs", True)
    def test_backward_with_expanded_computed_output(self, case):
        # [FUNCTION DOC] Parameterized by case. With tangent strides guessed as outputs, check
        # an output that is an expand (of a sum or of a computed slice) has zero stride, strides
        # and values match eager, and backward gradient matches.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return either the sum over rows expanded to the input shape
            # or the sine of the first row expanded; raises AssertionError for an unknown case.
            if case == "sum_expand":
                return x.sum(dim=0, keepdim=True).expand_as(x)
            elif case == "computed_view_expand":
                return x[:1].sin().expand_as(x)
            raise AssertionError(f"unexpected case: {case}")

        ref_x = torch.randn(4, 8, requires_grad=True)
        x = ref_x.detach().clone().requires_grad_(True)
        ref_grad = torch.randn(4, 8)
        grad = ref_grad.detach().clone()

        ref_y = fn(ref_x)
        y = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref_y, y)
        self.assertEqual(ref_y.stride(), y.stride())
        self.assertTrue(0 in y.stride())

        ref_y.backward(ref_grad)
        y.backward(grad)
        self.assertEqual(ref_x.grad, x.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @parametrize("dynamic_shapes", [True, False])
    @parametrize("test_subclasses", [True, False])
    @parametrize("device", ["cuda", "cpu"])
    @patch("torch._functorch.config.guess_tangent_strides_as_outputs", True)
    def test_noncontig_nonmemformat_tangents(
        self, dynamic_shapes, test_subclasses, device
    ):
        # [FUNCTION DOC] CUDA required. Parameterized by dynamic shapes, subclass inputs and
        # device. With tangent strides guessed as outputs, checks backward with tangents that
        # are all ones, zero-strided (memory overlap) and non-dense strided, for plain or
        # TwoTensor inputs; gradients match eager.
        B = 2
        T = 4
        E = 6

        def fn(x):
            # [FUNCTION DOC] HELPER: Add one then transpose dims one and two.
            x = x + 1
            return x.transpose(1, 2)

        def _inp_dense():
            # [FUNCTION DOC] HELPER: Make a random grad-requiring tensor, marking all dims
            # dynamic when dynamic shapes are on.
            t = torch.randn(B, T, E, device=device, requires_grad=True)
            if dynamic_shapes:
                for i in range(t.ndim):
                    torch._dynamo.mark_dynamic(t, i)
            return t

        def _inp_sc():
            # [FUNCTION DOC] HELPER: Make a TwoTensor of two such dense inputs.
            return TwoTensor(_inp_dense(), _inp_dense())

        _inp = _inp_dense if not test_subclasses else _inp_sc

        comp_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        def _tg3(y):
            # [FUNCTION DOC] HELPER: Make a random tensor twice the size, then view it with
            # doubled strides to get a non-dense tangent with the shape of y.
            t = torch.randn(
                2 * y.shape, dtype=y.dtype, layout=y.layout, device=y.device
            )
            return t.as_strided(y.shape, tuple(s * 2 for s in y.stride()))

        TEST_CASES = [
            (_inp, lambda y: torch.ones(y.shape, dtype=y.dtype, device=y.device)),
            # Memory overlap, dense tangent
            (
                _inp,
                lambda y: torch.tensor([1], dtype=y.dtype, device=y.device).as_strided(
                    y.shape, (0,) * y.ndim
                ),
            ),
            # No memory overlap, not-dense tangent
            (_inp, _tg3),
        ]

        for inp_fn, tg_fn in TEST_CASES:
            ref_x = inp_fn()
            x = ref_x.detach().clone().requires_grad_()

            ref_y = fn(ref_x)

            y = comp_fn(x)
            self.assertEqual(ref_y, y)

            ref_tg = (
                tg_fn(ref_y)
                if not test_subclasses
                else TwoTensor(tg_fn(ref_y), tg_fn(ref_y))
            )
            tg = ref_tg.clone()

            ref_y.backward(ref_tg)
            y.backward(tg)

            self.assertEqual(ref_x.grad, x.grad)

    @patch("torch._functorch.config.guess_tangent_strides_as_outputs", True)
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_flex_attn_noncontiguous_tangents(self):
        # [FUNCTION DOC] CUDA only. With tangent strides guessed as outputs and the logging
        # manager, flex attention output tangents through a logged op are received with the
        # stride pattern of the transposed layout rather than forced contiguous; one stride
        # tuple recorded.
        with GradsNoForceContiguousContextManager() as ctx:
            E = 16  # embedding dim
            H = 4  # number of heads

            @torch.compile(backend="aot_eager", fullgraph=True)
            def attn_fn(q, k, v):
                # [FUNCTION DOC] HELPER: Compiled flex attention followed by the logging op.
                y = flex_attention(query=q, key=k, value=v)
                y = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(y)
                return y

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Self attention module with a combined linear projection.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Create the projection layer.
                    super().__init__()
                    self.c_attn = torch.nn.Linear(E, 3 * E)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Project, split into q, k, v, reshape to heads and
                    # transpose, run the attention helper, restore the layout and return.
                    B, T, E = x.size()
                    q, k, v = self.c_attn(x).split(E, dim=2)
                    k = k.view(B, T, H, E // H).transpose(1, 2)  # (B, nh, T, hs)
                    q = q.view(B, T, H, E // H).transpose(1, 2)  # (B, nh, T, hs)
                    v = v.view(B, T, H, E // H).transpose(1, 2)  # (B, nh, T, hs)

                    y = attn_fn(q, k, v)

                    return y.transpose(1, 2).contiguous().view(B, T, E)

            m = M().cuda()
            B = 1
            T = 8

            def _inp():
                # [FUNCTION DOC] HELPER: Make a random CUDA input requiring grad.
                return torch.randn(B, T, E, requires_grad=True, device="cuda")

            x = _inp()
            y = m(x)
            y.backward(torch.ones_like(y).contiguous())

            self.assertEqual(1, len(ctx.tangent_strides))
            self.assertEqual((128, 4, 16, 1), ctx.tangent_strides[0])

    def _test_pack_hooks(
        self,
        fn,
        inp_fn,
        hooks,
        symbolic_tracing=True,
        pre_compile_fn=None,
        backend="aot_eager",
    ):
        # [FUNCTION DOC] HELPER: Run a function with saved tensor hooks in eager for reference
        # (all hooks as plain Python), then recompile with the same hooks (inline ones converted
        # to graph form when requested) and compare output and input gradients with tolerance.
        # Optional step before compile (for example clearing logs) and backend choice.
        ctx = torch.autograd.graph.saved_tensors_hooks
        torch._dynamo.reset()
        with ExitStack() as stack:
            # All hooks in eager to get ref
            for hook, _ in hooks:
                pack, unpack = hook
                stack.enter_context(ctx(pack, unpack))
            ref_x = inp_fn()

            def _f(t):
                # [FUNCTION DOC] HELPER: Detach and clone floating point tensors with
                # requires_grad, return others unchanged.
                if t.dtype.is_floating_point:
                    return t.detach().clone().requires_grad_()

                return t

            x = pytree.tree_map_only(torch.Tensor, _f, ref_x)

            ref_y = fn(*ref_x)
            ref_y.sum().backward()
        if pre_compile_fn:
            pre_compile_fn()

        with ExitStack() as stack:
            for hook, inline in hooks:
                pack, unpack = hook
                if inline:
                    if symbolic_tracing:
                        stack.enter_context(
                            ctx(
                                *saved_tensors_hooks_to_gm(
                                    pack,
                                    unpack,
                                    "pack_hash",
                                    "unpack_hash",
                                )
                            )
                        )
                    else:
                        stack.enter_context(
                            ctx(
                                *saved_tensors_hooks_to_gm(
                                    pack, unpack, "pack_hash", "unpack_hash"
                                )
                            )
                        )
                else:
                    stack.enter_context(ctx(pack, unpack))
            y = torch.compile(fn, backend=backend, fullgraph=True)(*x)
            y.sum().backward()
            self.assertEqual(ref_y, y, atol=1e-2, rtol=1e-2)
            ref_x_grad = pytree.tree_map_only(torch.Tensor, lambda t: t.grad, ref_x)
            x_grad = pytree.tree_map_only(torch.Tensor, lambda t: t.grad, x)
            self.assertEqual(ref_x_grad, x_grad, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @parametrize("saved_tensors_hooks_filtering_mode", ["donated", "no_static", "all"])
    def test_saved_tensors_hooks_base(self, saved_tensors_hooks_filtering_mode):
        # [FUNCTION DOC] CUDA and SM80 required. Parameterized by hook filtering mode. Run
        # several pack/unpack hook pairs (CPU offload, bfloat16, scaling, device and symbolic
        # size, float8, float8 with scale, Python-only) against two functions saving tensors for
        # backward including an integer tensor and a dynamic-size context.
        with patch(
            "torch._functorch.config.saved_tensors_hooks_filtering_mode",
            saved_tensors_hooks_filtering_mode,
        ):
            # y argument is expected to test saving of int tensor,
            # to check filtering functionality to not apply hooks for e.g. is_floating_point
            class SAF(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function saving two inputs and returning the
                # first; backward adds both saved tensors to the gradient.
                @staticmethod
                def forward(ctx, x, y):
                    # [FUNCTION DOC] HELPER: Save both inputs, return the first.
                    ctx.save_for_backward(x, y)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    # [FUNCTION DOC] HELPER: Add the saved tensors to the gradient; no gradient
                    # for the second input.
                    (saved_x, saved_y) = ctx.saved_tensors
                    return gx + saved_x + saved_y, None

            class AF(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function saving its input and its second
                # dimension size; backward adds saved input times the size.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Save the input and its size in dimension one;
                    # return the input.
                    ctx.save_for_backward(x)
                    ctx.d1 = x.size(1)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    # [FUNCTION DOC] HELPER: Add the saved input times the stored size to
                    # the gradient.
                    (saved_x,) = ctx.saved_tensors
                    d1 = ctx.d1
                    return gx + saved_x * d1

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Relu, add one, relu, double, then apply the
                # saving Function.
                x = x.relu()
                x = x + 1
                x = x.relu()
                x = 2 * x
                x = AF.apply(x)
                return x

            def simple_fn(x, y):
                # [FUNCTION DOC] HELPER: Add one, transpose, relu, transpose, then apply the
                # Function saving both inputs.
                x = x + 1
                x = x.t()
                x = x.relu()
                x = x.t()
                x = SAF.apply(x, y)
                return x

            device = torch.device("cuda:0")

            def inp_fn():
                # [FUNCTION DOC] HELPER: Make a grad-requiring CUDA tensor with dynamic dims and
                # an int64 zeros tensor.
                x = torch.ones(2, 2, device=device, requires_grad=True)
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                y = torch.zeros(2, 2, device=device, dtype=torch.int64)
                return x, y

            def pack_dev_sym_cpu(x):
                # [FUNCTION DOC] HELPER: Pack hook returning dtype, device, size in dimension
                # one and a CPU copy.
                return x.dtype, x.device, x.size(1), x.cpu()

            def unpack_dev_sym_cpu(packed):
                # [FUNCTION DOC] HELPER: Unpack hook restoring device and dtype from the
                # packed tuple.
                dtype, device, dim1, x = packed
                x = x.to(device=device)
                return x.to(dtype)

            def pack_tensor(x):
                # [FUNCTION DOC] HELPER: Pack hook returning device and a CPU copy.
                return x.device, x.cpu()

            def unpack_tensor(packed):
                # [FUNCTION DOC] HELPER: Unpack hook moving the CPU tensor back to its device.
                device, t_cpu = packed
                return t_cpu.to(device)

            def pack_bf16(x):
                # [FUNCTION DOC] HELPER: Pack hook returning dtype and a bfloat16 copy.
                return x.dtype, x.to(dtype=torch.bfloat16)

            def unpack_bf16(packed):
                # [FUNCTION DOC] HELPER: Unpack hook converting the stored tensor back to
                # its dtype.
                dtype, x = packed
                return x.to(dtype)

            def pack_mul2(x):
                # [FUNCTION DOC] HELPER: Pack hook returning dtype and the tensor doubled.
                return x.dtype, x * 2

            def unpack_mul2(x):
                # [FUNCTION DOC] HELPER: Unpack hook halving the stored tensor and converting
                # to dtype.
                dtype, x = x
                x = x / 2
                return x.to(dtype)

            def pack_wrapper_sc(x):
                # [FUNCTION DOC] HELPER: Pack hook wrapping in a wrapper subclass (unused:
                # subclass tests are disabled).
                return WrapperSubclass(x)

            def unpack_wrapper_sc(x):
                # [FUNCTION DOC] HELPER: Unpack hook returning the wrapped tensor.
                return x.a

            def pack_wrapper_two_tensor(x):
                # [FUNCTION DOC] HELPER: Pack hook wrapping in a TwoTensor (unused).
                return TwoTensor(x, x)

            def unpack_wrapper_two_tensor(x):
                # [FUNCTION DOC] HELPER: Unpack hook adding the two components (unused).
                return x.a + x.b

            def pack_mul2_eager(x):
                # [FUNCTION DOC] HELPER: Plain eager pack hook doubling the tensor.
                return x * 2

            def unpack_mul2_eager(x):
                # [FUNCTION DOC] HELPER: Plain eager unpack hook halving the tensor.
                return x / 2

            def pack_cpu(x):
                # [FUNCTION DOC] HELPER: Pack hook moving the tensor to the CPU.
                return x.to(device="cpu")

            def unpack_cpu(x):
                # [FUNCTION DOC] HELPER: Unpack hook moving the tensor back to the test device.
                return x.to(device=device)

            for test_fn in [simple_fn, fn]:
                self._test_pack_hooks(
                    test_fn,
                    inp_fn,
                    [((pack_cpu, unpack_cpu), True)],
                    symbolic_tracing=False,
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_bf16, unpack_bf16), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_mul2, unpack_mul2), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_tensor, unpack_tensor), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_dev_sym_cpu, unpack_dev_sym_cpu), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_mul2_eager, unpack_mul2_eager), False)]
                )
                self._test_pack_hooks(
                    test_fn,
                    inp_fn,
                    [((pack_fp8, unpack_fp8), True)],
                )
                self._test_pack_hooks(
                    test_fn,
                    inp_fn,
                    [((pack_fp8_with_scale, unpack_fp8_with_scale), True)],
                )
                # Disable testing of Subclasses for now
                # self._test_pack_hooks(test_fn, inp_fn, [(pack_wrapper_sc, unpack_wrapper_sc)])
                # self._test_pack_hooks(
                #     test_fn, inp_fn, [(pack_wrapper_two_tensor, unpack_wrapper_two_tensor)]
                # )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    def test_saved_tensors_hooks_params(self):
        # [FUNCTION DOC] CUDA and SM80 required. Check which saved tensors get hooks under each
        # filtering mode: a module with parameters and a float64 input saved by a Function runs
        # with a pack hook that logs shape and dtype through a custom op; donated mode skips
        # parameters and the input, no-static mode keeps the input, all mode logs everything.
        with torch.library._scoped_library("_test_aotdispatch_lib", "FRAGMENT") as lib:
            logged_shapes = []
            logged_dtypes = []
            lib.define("log(Tensor x) -> Tensor")

            def log_impl(x):
                # [FUNCTION DOC] HELPER: Log implementation recording shape and dtype and
                # returning a clone.
                logged_shapes.append(list(x.shape))
                logged_dtypes.append(x.dtype)
                return x.clone()

            def log_meta(x):
                # [FUNCTION DOC] HELPER: Meta implementation returning a clone.
                return x.clone()

            for backend in ["CPU", "CUDA"]:
                lib.impl(
                    "log",
                    log_impl,
                    backend,
                )
            lib.impl("log", log_meta, "Meta")

            def pack_fp8_with_scale_and_log(x):
                # [FUNCTION DOC] HELPER: Pack hook that logs through the custom op then packs
                # with scale to float8.
                torch.ops._test_aotdispatch_lib.log(x)
                return _pack_fp8_with_scale_wrap(x)

            def unpack_fp8_with_scale_and_log(packed):
                # [FUNCTION DOC] HELPER: Unpack hook delegating to the float8 scale
                # unpacking function.
                return _unpack_fp8_with_scale_wrap(packed)

            def m_inp_fn():
                # [FUNCTION DOC] HELPER: Make a float64 grad-requiring CUDA input with dynamic
                # dims, returned in a tuple.
                x = torch.ones(
                    2, 2, 2, device=device, dtype=torch.float64, requires_grad=True
                )
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                return (x,)

            class SAF0(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function saving the input; backward adds the
                # saved input to the gradient.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Save the input and return it.
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    # [FUNCTION DOC] HELPER: Add the saved input to the gradient.
                    (saved_x,) = ctx.saved_tensors
                    return gx + saved_x

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying the Function, casting to float32 and two
                # linear layers with relu.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Create two linear layers and relu.
                    super().__init__()
                    self.fc1 = nn.Linear(2, 2)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(2, 2)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Apply the Function, cast, linear, relu, linear.
                    x = SAF0.apply(x)
                    x = x.to(dtype=torch.float32)
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x

            def _reset_logged():
                # [FUNCTION DOC] HELPER: Clear the logged shapes and dtypes.
                logged_shapes.clear()
                logged_dtypes.clear()

            device = torch.device("cuda:0")
            m = M().to(device=device)

            def _test_m():
                # [FUNCTION DOC] HELPER: Run the pack hook comparison on the module with the
                # float8 logging hooks, clearing logs before compile.
                self._test_pack_hooks(
                    m,
                    m_inp_fn,
                    [
                        (
                            (
                                pack_fp8_with_scale_and_log,
                                unpack_fp8_with_scale_and_log,
                            ),
                            True,
                        )
                    ],
                    pre_compile_fn=_reset_logged,
                    backend="aot_eager",
                )

            with patch(
                "torch._functorch.config.saved_tensors_hooks_filtering_mode", "donated"
            ):
                _reset_logged()
                _test_m()
                # Check that hooks were not applied to Parameters
                # parameters excluded
                self.assertFalse([2, 2] in logged_shapes)
                self.assertTrue([2, 2, 2] in logged_shapes)
                # input excluded
                self.assertFalse(torch.float64 in logged_dtypes)

            with patch(
                "torch._functorch.config.saved_tensors_hooks_filtering_mode",
                "no_static",
            ):
                _reset_logged()
                _test_m()
                # Check that hooks were not applied to Parameters
                # parameters excluded
                self.assertFalse([2, 2] in logged_shapes)
                self.assertTrue([2, 2, 2] in logged_shapes)
                self.assertTrue(torch.float64 in logged_dtypes)

            with patch(
                "torch._functorch.config.saved_tensors_hooks_filtering_mode", "all"
            ):
                _reset_logged()
                _test_m()
                # Check that hooks were applied to all saved tensors
                self.assertTrue([2, 2] in logged_shapes)
                self.assertTrue([2, 2, 2] in logged_shapes)
                self.assertTrue(torch.float64 in logged_dtypes)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @torch._functorch.config.patch(saved_tensors_hooks_filtering_mode="all")
    def test_saved_tensors_hooks_recompile(self):
        # [FUNCTION DOC] CUDA and SM80 required, with hook filtering for all tensors. Checks
        # recompilation behavior when saved tensor hooks change between runs: with two different
        # plain hook pairs there is one compiled frame in total, while with the same pairs in
        # inline graph form there are three.
        ctx = torch.autograd.graph.saved_tensors_hooks

        def pack_bf16(x):
            # [FUNCTION DOC] HELPER: Pack hook converting to bfloat16.
            return x.to(dtype=torch.bfloat16)

        def unpack_bf16(x):
            # [FUNCTION DOC] HELPER: Unpack hook converting to float.
            return x.to(dtype=torch.float)

        def pack_mul2(x):
            # [FUNCTION DOC] HELPER: Pack hook doubling the tensor.
            return x * 2

        def unpack_mul2(x):
            # [FUNCTION DOC] HELPER: Unpack hook halving the tensor.
            return x / 2

        def _test(hooks, inline, expected_compile_count):
            # [FUNCTION DOC] HELPER: Compile a function once without hooks, then with each of
            # two hook pairs (inline graph form or plain), and assert the frame compile count
            # equals the expected count.
            class SAF(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function saving the input; backward adds the
                # saved input to the gradient.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Save the input and return it.
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    # [FUNCTION DOC] HELPER: Add the saved input to the gradient.
                    (saved_x,) = ctx.saved_tensors
                    return gx + saved_x

            class AF(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function saving the input and its size; backward
                # adds saved input times size.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Save the input and its size in dimension one;
                    # return the input.
                    ctx.save_for_backward(x)
                    ctx.d1 = x.size(1)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    # [FUNCTION DOC] HELPER: Add the saved input times the stored size to
                    # the gradient.
                    (saved_x,) = ctx.saved_tensors
                    d1 = ctx.d1
                    return gx + saved_x * d1

            def fn(x):
                # [FUNCTION DOC] HELPER: Relu, add one, double, then apply the saving Function.
                x = x.relu()
                x = x + 1
                x = 2 * x
                x = AF.apply(x)
                return x

            device = torch.device("cuda:0")

            def inp_fn():
                # [FUNCTION DOC] HELPER: Make a grad-requiring CUDA tensor with dynamic dims.
                x = torch.ones(2, 3, device=device, requires_grad=True)
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                return x

            from torch._dynamo.testing import CompileCounter

            cnt = CompileCounter()
            x = inp_fn()
            y = torch.compile(fn, backend=cnt, fullgraph=True)(x)
            y.sum().backward()

            def _test_with_hooks(hooks):
                # [FUNCTION DOC] HELPER: Enter the saved tensor hooks (inline graph form if
                # requested), compile and run forward and backward.
                with ExitStack() as stack:
                    pack, unpack = hooks
                    if inline:
                        stack.enter_context(
                            ctx(
                                *saved_tensors_hooks_to_gm(
                                    pack, unpack, "pack_hash", "unpack_hash"
                                )
                            )
                        )
                    else:
                        stack.enter_context(ctx(pack, unpack))

                    x = inp_fn()
                    y = torch.compile(fn, backend=cnt, fullgraph=True)(x)
                    y.sum().backward()

            _test_with_hooks(hooks[0])
            _test_with_hooks(hooks[1])
            self.assertEqual(cnt.frame_count, expected_compile_count)

        _test(
            ((pack_bf16, unpack_bf16), (pack_mul2, unpack_mul2)),
            inline=False,
            expected_compile_count=1,
        )
        _test(
            ((pack_bf16, unpack_bf16), (pack_mul2, unpack_mul2)),
            inline=True,
            expected_compile_count=3,
        )

    @torch._functorch.config.patch(donated_buffer=True)
    @torch._functorch.config.patch(saved_tensors_hooks_filtering_mode="no_static")
    def test_saved_tensors_hooks_donated_buffers(self):
        # [FUNCTION DOC] With donated buffers on and no_static hook filtering: first, without
        # hooks, compile a function returning its input and a relu of a Function output and
        # check the log reports no donated backward buffers (the saved input is also a user
        # output); second, with float8 pack and unpack hooks in graph form, check the log
        # reports two donated backward buffers.
        pack_gm, unpack_gm = saved_tensors_hooks_to_gm(
            pack_fp8,
            unpack_fp8,
            "pack_hash",
            "unpack_hash",
        )
        logger_name = "torch._functorch._aot_autograd.graph_compile"

        class SAF(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function saving the input; backward adds the saved
            # input to the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it.
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Add the saved input to the gradient.
                (saved_x,) = ctx.saved_tensors
                return gx + saved_x

        def fn(x):
            # [FUNCTION DOC] HELPER: Return the input and a relu of the output of the
            # saving Function.
            x0 = x
            x = SAF.apply(x)
            return x0, torch.nn.functional.relu(x)

        inp = torch.rand([3, 3], requires_grad=True)
        # 1. No donated buffers without hooks, as relu saves input which is also user output.
        with self.assertLogs(logger_name, level="INFO") as captured:
            out = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=False)(
                inp
            )
            out[1].sum().backward()
            expected_msg = "bw_donated_idxs=[]"

        FileCheck().check(expected_msg).run("\n".join(captured.output))

        # 2. Hooks applied for all saved, as we set saved_tensors_hooks_no_filtering=True
        # Results of the hooks become donated buffers.
        inp = torch.rand([3, 3], requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_gm, unpack_gm):
            with self.assertLogs(logger_name, level="INFO") as captured:
                out = torch.compile(
                    fn, backend="aot_eager", fullgraph=True, dynamic=False
                )(inp)
                out[1].sum().backward()
                expected_msg = "bw_donated_idxs=[0, 1]"

        FileCheck().check(expected_msg).run("\n".join(captured.output))


# entries in here don't work and need to be fixed.
# Each one of these is a bug (or needs to be investigated)
aot_autograd_failures = {
    # data-dependent control flow
    xfail("cov"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("tensor_split"),
    xfail("corrcoef"),
    skip("narrow"),
    xfail("istft"),
    xfail("linalg.eig"),
    skip("as_strided", "partial_views"),  # flaky
    # Given input size: (s0xs1x2). Calculated output size: ...
    skip("max_pool2d_with_indices_backward"),
    # Misc
    xfail("to_sparse"),
    xfail("corrcoef"),
    xfail("cov"),
    xfail("chalf"),  # RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'
    xfail("sparse.sampled_addmm"),
    xfail("sparse.mm", "reduce"),
    skip("nn.functional.binary_cross_entropy_with_logits"),  # seems to fail sometimes?
    skip("nn.functional.margin_ranking_loss"),  # seems flaky
    skip("linalg.lu_solve"),  # flaky
    decorate("matmul", decorator=unittest.skipIf(IS_ARM64, "flaky")),
    decorate("__rmatmul__", decorator=unittest.skipIf(IS_ARM64, "flaky")),
    # overrides atol=1e-4, rtol=1e-5 would do as well
    decorate(
        "svd_lowrank",
        decorator=toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-05)}),
    ),
    decorate(
        "linalg.householder_product",
        decorator=unittest.skipIf(IS_MACOS and IS_X86, "flaky"),
    ),
    decorate(
        "linalg.pinv",
        "singular",
        # This delta is coming entirely from the clone() on tangents
        # in AOTDispatcher to make them contiguous
        decorator=toleranceOverride({torch.float32: tol(atol=1e-02, rtol=1e-02)}),
    ),
    decorate(
        "cholesky_inverse",
        # Numerical differences due to tangent stride differences between
        # eager (.sum().backward() uses contiguous tangent) and compiled
        # (tangent strides match output strides). With ill-conditioned inputs,
        # matmul accumulates rounding errors differently for different layouts.
        decorator=toleranceOverride({torch.float32: tol(atol=3e02, rtol=2e-03)}),
    ),
    decorate(
        "nn.functional.interpolate",
        "bicubic",
        decorator=toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-05)}),
    ),
    # conv2d sometimes nondeterministic in this config?
    decorate("nn.functional.conv2d", decorator=unittest.skipIf(IS_ARM64, "flaky")),
}

if not TEST_MKL:
    aot_autograd_failures.update(
        {
            decorate(
                "matmul",
                decorator=toleranceOverride(
                    {torch.float32: tol(atol=6e-05, rtol=4e-06)}
                ),
            ),
            decorate(
                "__rmatmul__",
                decorator=toleranceOverride(
                    {torch.float32: tol(atol=6e-05, rtol=4e-06)}
                ),
            ),
        }
    )

symbolic_aot_autograd_failures = {
    xfail(
        "linalg.lstsq", ""
    ),  # aten.linalg_lstsq.default - couldn't find symbolic meta function/decomposition
    xfail(
        "linalg.lstsq", "grad_oriented"
    ),  # aten.linalg_lstsq.default - couldn't find symbolic meta funct...
    xfail(
        "linalg.lu_solve", ""
    ),  # aten.linalg_lu_solve.default - couldn't find symbolic meta function/deco...
    skip(
        "nn.functional.batch_norm", ""
    ),  # '0 is not tracked with proxy for <torch.fx.experimental.proxy_te..
    xfail(
        "nn.functional.ctc_loss", ""
    ),  # aten._ctc_loss.Tensor - couldn't find symbolic meta function/deco...
    xfail(
        "nn.functional.fractional_max_pool3d", ""
    ),  # rand() received an invalid combination of arguments - g...
    decorate(
        "linalg.householder_product",
        decorator=unittest.skipIf(IS_MACOS and IS_X86, "flaky"),
    ),
}


def _test_aot_autograd_helper(
    self,
    device,
    dtype,
    op,
    dynamic=False,
    disable_functionalization=False,
):
    # [FUNCTION DOC] HELPER: Shared body of the OpInfo autograd tests. Skips if the op does not
    # support autograd. For every sample input (with grad enabled) runs the aot autograd check
    # on the op (data specialization checks except for the unpool ops, optionally dynamic shapes
    # and disabled functionalization); a dynamic output shape error skips the test, and a data
    # dependent guard error skips only for the getitem op and is re-raised otherwise.
    if not op.supports_autograd:
        self.skipTest("Op does not support autograd")

    # aot_autograd_check is able to check data specialization by
    # randomizing the inputs. Here's a list of ops that really do not
    # like random inputs for which we want to disable that.
    cant_check_data_specialization = set(
        {
            "nn.functional.max_unpool1d",
            "nn.functional.max_unpool2d",
            "nn.functional.max_unpool3d",
        }
    )
    try_check_data_specialization = op.name not in cant_check_data_specialization

    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
    for sample_input in sample_inputs_itr:
        t_args = [sample_input.input] + list(sample_input.args)
        t_kwargs = sample_input.kwargs
        try:
            aot_autograd_check(
                op.op,
                t_args,
                t_kwargs,
                dynamic,
                self.assertRaisesRegex,
                self.assertEqual,
                check_gradients=True,
                try_check_data_specialization=try_check_data_specialization,
                skip_correctness_check=op.skip_correctness_check_compile_vs_eager,
                disable_functionalization=disable_functionalization,
            )
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")
        except GuardOnDataDependentSymNode:
            # Carveout for getitem; I don't want to xfail the entire test
            # because that will reject known to be good tests see
            # https://github.com/pytorch/pytorch/issues/94705
            if op.name == "__getitem__":
                self.skipTest("Dynamic output shape operation in trace")
            else:
                raise


def _test_aot_autograd_module_helper(
    self, device, dtype, training, module_info, *, dynamic=False
):
    # [FUNCTION DOC] HELPER: Shared body of the module autograd tests. For each module sample
    # input with a forward input (skipping packed sequence inputs), builds the module in the
    # given device, dtype and train mode, runs lazy modules once, defines a functional-call
    # wrapper taking parameters, buffers and tensor arguments, compiles it with aot_function
    # using no-op compilers and compares forward and backward results via the
    # forwards-backwards helper.
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(
        module_info, device=device, dtype=dtype, requires_grad=True, training=training
    )
    for module_input in module_inputs:
        if module_input.forward_input is None:
            continue

        args, kwargs = (
            module_input.constructor_input.args,
            module_input.constructor_input.kwargs,
        )
        m = module_cls(*args, **kwargs)
        m.to(device).to(dtype)
        m.train(training)

        # Lazy modules need to see an input first to initialize params.
        args, kwargs = (
            module_input.forward_input.args,
            module_input.forward_input.kwargs,
        )
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # PackedSequence is only used for RNNs. It might be possible to fake-ify if they're pytrees but
        # torchdynamo already doesn't support RNNs
        if any(tuple(isinstance(flat_arg, PackedSequence) for flat_arg in flat_args)):
            continue

        if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
            with torch.no_grad():
                m(*args, **kwargs)

        sentinel_val = -42
        is_tensor_spec = [
            sentinel_val if isinstance(arg, torch.Tensor) else arg for arg in flat_args
        ]
        args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

        def f(params_buffers_args):
            # [FUNCTION DOC] HELPER: Rebuild the flattened arguments by filling tensor slots
            # in order, unflatten, and call the module functionally with the given parameters
            # and buffers.
            named_params, named_buffers, args = params_buffers_args
            cur_flat_args = list(is_tensor_spec)
            args = iter(args)
            for idx, v in enumerate(cur_flat_args):
                if v == sentinel_val:
                    cur_flat_args[idx] = next(args)
            c_args, c_kwargs = pytree.tree_unflatten(cur_flat_args, args_spec)
            params_and_buffers = {**named_params, **named_buffers}
            return torch.func.functional_call(m, params_and_buffers, c_args, c_kwargs)

        named_params = dict(m.named_parameters(remove_duplicate=False))
        named_buffers = dict(m.named_buffers(remove_duplicate=False))
        num_params_buffers = len(named_params) + len(named_buffers)
        compiled_f = aot_function(
            f, nop, num_params_buffers=num_params_buffers, dynamic=dynamic
        )
        params_buffers_args = [named_params, named_buffers, args]
        _test_aot_autograd_forwards_backwards_helper(
            f,
            compiled_f,
            params_buffers_args,
            self.assertRaisesRegex,
            self.assertEqual,
            True,
        )


class TestEagerFusionOpInfo(AOTTestCase):
    # [CLASS DOC] OpInfo based tests running the exhaustive aot autograd check for each
    # operator in the operator and higher-order-op databases, with float dtype, in plain,
    # symbolic, and functionalization-disabled variants, and with known failures skipped
    # through the failure dictionaries.
    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @skipOps(aot_autograd_failures)
    def test_aot_autograd_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] Run the exhaustive aot autograd helper for the operator.
        _test_aot_autograd_helper(self, device, dtype, op)

    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @patch("functorch.compile.config.debug_assert", True)
    @skipOps(
        aot_autograd_failures | symbolic_aot_autograd_failures,
    )
    def test_aot_autograd_symbolic_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] Run the exhaustive helper for the operator with dynamic shapes (debug
        # assertions on).
        _test_aot_autograd_helper(self, device, dtype, op, dynamic=True)

    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @skipOps(
        aot_autograd_failures,
    )
    def test_aot_autograd_disable_functionalization_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] Run the exhaustive helper for the operator with
        # functionalization disabled.
        _test_aot_autograd_helper(
            self, device, dtype, op, disable_functionalization=True
        )

    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @patch("functorch.compile.config.debug_assert", True)
    @skipOps(
        aot_autograd_failures | symbolic_aot_autograd_failures,
    )
    def test_aot_autograd_disable_functionalization_symbolic_exhaustive(
        self, device, dtype, op
    ):
        # [FUNCTION DOC] Run the exhaustive helper for the operator with dynamic shapes and
        # functionalization disabled (debug assertions on).
        _test_aot_autograd_helper(
            self,
            device,
            dtype,
            op,
            dynamic=True,
            disable_functionalization=True,
        )


aot_autograd_module_failures = set(
    {
        torch.nn.CTCLoss,  # torch._subclasses.fake_tensor.DynamicOutputShapeException: aten._ctc_loss.default
        torch.nn.GaussianNLLLoss,  # RuntimeError: It appears that you're trying to get value out
        # of a tracing tensor with aten._local_scalar_dense.default -
        # erroring out! It's likely that this is caused by data-dependent
        # control flow or similar.
        torch.nn.MultiLabelMarginLoss,  # AssertionError: The values for attribute 'shape' do not match:
        # torch.Size([1]) != torch.Size([]). Outputs of the operator are different in
        # eager-mode PyTorch vs AOTAutograd. This means the operator will have incorrect
        # output underneath torch.compile. This could be because the operator's
        # implementation not traceable or that there is a bug in AOTAutograd.
        torch.nn.TransformerEncoder,  # DataDependentOutputException: aten.eq compares a mask input
        # to a causal mask tensor, to see if Boolean is_causal should be set
        # for TransformerEncoder layers, MHA and sdp custom kernels
        torch.nn.Transformer,  # DataDependentOutputException: aten.equal compares a mask input
        # to a causal mask tensor, to see if Boolean is_causal should be set
        # for TransformerEncoder layers, MHA and sdp custom kernels
        # (this bubbles up to Transformer)
    }
)

symbolic_aot_autograd_module_failures = {
    torch.nn.Transformer,  # DataDependentOutputException: aten.equal compares a mask input to a mask producing a bool
    torch.nn.TransformerEncoder,  # DataDependentOutputException: aten.equal compares a mask input to a mask producing a bool
    torch.nn.GaussianNLLLoss,  # NotImplementedError: local_scalar_dense/item NYI for torch.bool
    torch.nn.FractionalMaxPool3d,  # int() argument must be a string, a bytes-like object or a number, not 'SymFloat'
}


class TestEagerFusionModuleInfo(AOTTestCase):
    # [CLASS DOC] ModuleInfo based tests running the aot autograd check for each module in the
    # module database with float dtype, static and dynamic shapes, with known failing modules
    # marked as expected failures.
    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(unittest.expectedFailure, aot_autograd_module_failures)
    def test_aot_autograd_module_exhaustive(self, device, dtype, training, module_info):
        # [FUNCTION DOC] Run the module autograd helper for the module in the given
        # training mode.
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info)

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(
        unittest.expectedFailure,
        aot_autograd_module_failures | symbolic_aot_autograd_module_failures,
    )
    def test_aot_autograd_symbolic_module_exhaustive(
        self, device, dtype, training, module_info
    ):
        # [FUNCTION DOC] Run the module autograd helper with dynamic shapes.
        _test_aot_autograd_module_helper(
            self, device, dtype, training, module_info, dynamic=True
        )


instantiate_parametrized_tests(TestAOTAutograd)
instantiate_parametrized_tests(TestAOTModuleSimplified)
only_for = "cpu"
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)
instantiate_device_type_tests(TestEagerFusionModuleInfo, globals(), only_for=only_for)


@xfail_inherited_tests(
    [
        "test_set__and_data_mutation_bad",
    ]
)
class TestAOTAutogradWithDynamo(TestAOTAutograd):
    # [CLASS DOC] Runs every TestAOTAutograd test again, but first lowering the function through
    # dynamo to get a graph module, then compiling it with aot_module_simplified; graph text
    # assertions are disabled, and one test is expected to fail.
    """
    These are the same as TestAOTAutograd tests, but we run dynamo first to get a graph module.
    """

    def assertExpectedInline(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Override that ignores expected-text assertions because dynamo
        # produces different graphs.
        # These will have different outputs because dynamo returns a different graph module
        # But we don't really care about that assertion when testing with dynamo,
        # only that the outputs match, etc.
        pass

    def make_compiler(self, graph_cell):
        # [FUNCTION DOC] HELPER: Make a boxed compiler that stores the graph into the given
        # list cell.
        return make_boxed_compiler(partial(extract_graph, graph_cell=graph_cell))

    # Compiler to passes to dynamo
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: list[Callable | None],
        decompositions: dict | None,
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        # [FUNCTION DOC] HELPER: Override of the base compile helper: returns a function that
        # resets dynamo and runs the function through the compile entry point with a backend
        # that compiles the graph with aot_module_simplified (forward graph captured into the
        # cell, backward compiler discarded); backend failures are re-raised as the underlying
        # exception so assertion helpers work.
        """
        Runs dynamo and aot_autograd with the specified settings
        """

        def dynamo_compiler(gm, inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Backend that compiles the dynamo graph module with
            # aot_module_simplified using the test's compilers and settings.
            result = aot_module_simplified(
                gm,
                inputs,
                fw_compiler=self.make_compiler(fw_graph_cell),
                bw_compiler=self.make_compiler([None]),
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                # Dynamic is calculated from whether the inputs have fake tensors
            )
            return result

        def torch_compile_wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Reset dynamo, compile the function with the dynamo backend,
            # call it, and re-raise the inner exception of a backend failure.
            torch._dynamo.reset()
            fn = torch.compile(f, backend=dynamo_compiler)
            try:
                result = fn(*args, **kwargs)
            except torch._dynamo.exc.BackendCompilerFailed as e:
                # So that assertRaises works properly
                raise e.inner_exception from e
            return result

        return torch_compile_wrapper

    def test_inputs_overlapping_unsqueeze_with_mutation(self):
        # [FUNCTION DOC] Check dynamic compilation of a function mutating two unsqueezed views
        # of the same base matches eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Add one in place to both inputs and return the first.
            x.add_(1)
            y.add_(1)
            return x

        def run(f):
            # [FUNCTION DOC] HELPER: Call the given function with two unsqueezed views of a
            # ones base.
            base = torch.ones(10)
            inputs = [base.unsqueeze(0), base.unsqueeze(0)]
            return f(*inputs)

        optf = torch.compile(backend="aot_eager", dynamic=True)(f)

        out = run(f)
        optout = run(optf)

        self.assertEqual(out, optout)

    def test_inputs_overlapping_with_mutation_guard_base(self):
        # [FUNCTION DOC] Check dynamic compilation of a function mutating two identical
        # overlapping slices matches eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Add one in place to both inputs and return the first.
            x.add_(1)
            y.add_(1)
            return x

        def run(f):
            # [FUNCTION DOC] HELPER: Call the given function with two identical slices of a
            # ones base.
            base = torch.ones(10)
            inputs = [base[1:], base[1:]]
            return f(*inputs)

        optf = torch.compile(backend="aot_eager", dynamic=True)(f)

        out = run(f)
        optout = run(optf)

        self.assertEqual(out, optout)

    def test_lift_after_factory(self):
        # [FUNCTION DOC] Check a lift op after a random integer factory under the same seed
        # gives the same result compiled and eager.
        def f(low, high, size):
            # [FUNCTION DOC] HELPER: Make a random integer tensor and apply the lift op.
            y = torch.randint(low=low, high=high, size=size)
            return torch.ops.aten.lift.default(y)

        torch.manual_seed(0)
        out = f(0, 100, [2, 3])
        torch.manual_seed(0)
        opt_out = torch.compile(f, backend="aot_eager", fullgraph=True)(0, 100, [2, 3])

        self.assertEqual(out, opt_out)

    def test_mutations_in_bw_detached_from_tangent(self):
        # [FUNCTION DOC] Check an in-place mutation in backward on a tensor that is not tied to
        # the incoming gradient (a gradient-derived attachment is added) gives the same values
        # after forward and after backward as eager.
        class AF(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function keeping a tensor on the context; backward
            # adds a one-valued gradient-shaped attachment to it in place.
            @staticmethod
            def forward(ctx, dummy, inplace_tensor):
                # [FUNCTION DOC] HELPER: Keep the tensor on the context and return a clone of
                # the dummy.
                ctx.inplace_tensor = inplace_tensor
                return dummy.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Add ones shaped like the gradient to the kept tensor in
                # place; return the gradient and no gradients for the others.
                inplace_tensor = ctx.inplace_tensor
                gradient_attachment = grad_output * 0 + 1
                inplace_tensor.add_(1 * gradient_attachment)
                return grad_output, None, None

        def fn(dummy, inplace_tensor):
            # [FUNCTION DOC] HELPER: Apply the custom Function.
            return AF.apply(dummy, inplace_tensor)

        def _inps():
            # [FUNCTION DOC] HELPER: Make a grad-requiring dummy and a plain zeros tensor.
            dummy = torch.zeros((2,), requires_grad=True)
            inplace_tensor = torch.zeros((2,), requires_grad=False)
            return dummy, inplace_tensor

        inps = _inps()
        out = fn(*inps)
        ref_inps_after_fw = [x.clone().detach() for x in inps]
        out.sum().backward()
        ref_inps_after_bw = [x.clone().detach() for x in inps]

        inps = _inps()
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(*inps)
        inps_after_fw = [x.clone().detach() for x in inps]
        out.sum().backward()
        inps_after_bw = [x.clone().detach() for x in inps]

        self.assertEqual(ref_inps_after_fw, inps_after_fw)
        self.assertEqual(ref_inps_after_bw, inps_after_bw)

    def test_mutations_in_bw_requires_grad_input(self):
        # [FUNCTION DOC] Check mutating a forward input that requires grad during backward does
        # not trigger an in-place check error and results match eager.
        # Inplace mutation of a requires_grad=True forward input during the
        # backward pass should not trigger the check_inplace error
        class AF(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function saving a tensor; backward doubles it
            # in place.
            @staticmethod
            def forward(ctx, dummy, inplace_tensor):
                # [FUNCTION DOC] HELPER: Save the tensor and return a clone of the dummy.
                ctx.save_for_backward(inplace_tensor)
                return dummy.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Double the saved tensor in place; return the gradient
                # and no gradient for the other input.
                (inplace_tensor,) = ctx.saved_tensors
                inplace_tensor.mul_(2.0)
                return grad_output, None

        def fn(dummy, inplace_tensor):
            # [FUNCTION DOC] HELPER: Apply the custom Function.
            return AF.apply(dummy, inplace_tensor)

        def _inps():
            # [FUNCTION DOC] HELPER: Make a zeros dummy and a ones tensor, both requiring grad.
            dummy = torch.zeros((2,), requires_grad=True)
            # requires_grad=True is what triggered the bug
            inplace_tensor = torch.ones((2,), requires_grad=True)
            return dummy, inplace_tensor

        inps = _inps()
        out = fn(*inps)
        ref_inps_after_fw = [x.clone().detach() for x in inps]
        out.sum().backward()
        ref_inps_after_bw = [x.clone().detach() for x in inps]

        inps = _inps()
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(*inps)
        inps_after_fw = [x.clone().detach() for x in inps]
        out.sum().backward()
        inps_after_bw = [x.clone().detach() for x in inps]

        self.assertEqual(ref_inps_after_fw, inps_after_fw)
        self.assertEqual(ref_inps_after_bw, inps_after_bw)

    def test_mutation_of_input_in_fw_and_bw(self):
        # [FUNCTION DOC] Check mutations of an input in both forward and backward, for plain and
        # TwoTensor inputs, are applied at the right times after compiled forward and backward.
        class AF(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function that increments the kept tensor in place in
            # both forward and backward.
            @staticmethod
            def forward(ctx, dummy, inplace_tensor):
                # [FUNCTION DOC] HELPER: Increment the kept tensor in place, keep it, return a
                # clone of the dummy.
                inplace_tensor.add_(1)

                ctx.inplace_tensor = inplace_tensor
                return dummy.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Increment the kept tensor in place and return
                # the gradient.
                inplace_tensor = ctx.inplace_tensor
                inplace_tensor.add_(1)
                return grad_output, None, None

        def fn(dummy, inplace_tensor):
            # [FUNCTION DOC] HELPER: Apply the custom Function.
            return AF.apply(dummy, inplace_tensor)

        def inps():
            # [FUNCTION DOC] HELPER: Make a random grad-requiring dummy and a zeros tensor.
            dummy = torch.randn((2,), requires_grad=True)
            inplace_tensor = torch.zeros((2,), requires_grad=False)
            return dummy, inplace_tensor

        def sc_inps():
            # [FUNCTION DOC] HELPER: Make the same inputs as TwoTensor subclass values.
            dummy = TwoTensor(
                torch.randn((2,), requires_grad=True),
                torch.randn((2,), requires_grad=True),
            )
            inplace_tensor = TwoTensor(
                torch.zeros((2,), requires_grad=False),
                torch.zeros((2,), requires_grad=False),
            )
            return dummy, inplace_tensor

        for _inps in [inps, sc_inps]:
            dummy, inplace = _inps()
            y = fn(dummy, inplace)
            ref0 = inplace.clone().detach()
            y.sum().backward()
            ref = inplace.clone().detach()

            dummy, inplace = _inps()
            y = torch.compile(fn, backend="aot_eager", fullgraph=True)(dummy, inplace)
            self.assertEqual(ref0, inplace)
            y.sum().backward()
            self.assertEqual(ref, inplace)


class MockFXGraphCache:
    # [CLASS DOC] HELPER: In-memory stand-in for the inductor graph cache keyed by graph hash;
    # stores graph modules in a mapping and serves them back as cache output objects.
    """
    In memory version of FXGraphCache so we can isolate testing for FXGraphCache
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create an empty cache mapping.
        self.cache = {}

    def save(self, key, gm):
        # [FUNCTION DOC] HELPER: Store the graph module under the key.
        self.cache[key] = gm

    def load(self, gm, inputs):
        # [FUNCTION DOC] HELPER: Compute the hash key of the graph and inputs, insert the graph
        # if missing, and return the graph retrieved by key.
        key, _ = compiled_fx_graph_hash(gm, inputs, {}, [])
        if key not in self.cache:
            self.cache[key] = gm
        gm, _ = self.load_with_key(key, [], inputs, None, None, None, None, None)
        return gm

    def load_with_key(
        self,
        key,
        debug_lines,
        inputs,
        local,
        remote_cache,
        is_backward,
        constants,
        evaluate_guards,
    ):
        # [FUNCTION DOC] HELPER: Look up the key and wrap the graph as a boxed function inside a
        # mock cache output object with key, empty debug lines and zero time taken; returns that
        # (or none if missing) and an empty dictionary. Other arguments are ignored.
        gm = self.cache.get(key)
        if gm is not None:
            gm = make_boxed_func(gm)
            gm = MockFXGraphCacheOutput(gm)
            gm._fx_graph_cache_key = key  # (cache_key, debug lines)
            gm._fx_graph_cache_debug_lines = []
            gm._time_taken_ns = 0
        return gm, {}


# The following tests fail in strict caching mode (i.e. they bypass or
# cache miss instead of cache hitting). They will be fixed in the PRs above this.
FAILING_CACHE_TESTS = (
    # BypassAOTAutogradCache: unsupported nodes
    "test_backward_mutation_data",  # Custom Autograd Function
    # "test_backward_mutation_metadata",  # Custom Autograd Function
    "test_input_output_aliase_custom_autograd_function",
)


@xfail_inherited_tests(FAILING_CACHE_TESTS)
class TestAOTAutogradWithCache(TestAOTAutogradWithDynamo):
    # [CLASS DOC] Runs the dynamo-based AOT autograd tests with an in-memory inductor graph
    # cache and strict autograd cache mode (bypasses or misses fail); a few tests known to fail
    # are marked expected failures.
    """
    In memory version of FXGraphCache so we can isolate testing for FXGraphCache
    """

    def make_compiler(self, fw_graph_cell):
        # [FUNCTION DOC] HELPER: Make a compiler that loads the graph through the in-memory
        # cache, records the original graph in the cell, wraps it as a serializable compiler for
        # cache output.
        mock_inductor_cache = self.inductor_cache

        def compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Load the graph through the mock cache, store the graph in
            # the cell and return the cached result.
            nonlocal mock_inductor_cache, fw_graph_cell
            result = mock_inductor_cache.load(gm, example_inputs)
            fw_graph_cell[0] = gm
            return result

        compiler = SerializableAOTDispatchCompiler(MockFXGraphCacheOutput, compiler)
        return compiler

    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: list[Callable | None],
        decompositions: dict | None,
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        # [FUNCTION DOC] HELPER: Delegate to the parent implementation with the same arguments.
        return super().run_autograd(
            f,
            fw_graph_cell,
            decompositions,
            keep_input_mutations,
            dynamic,
        )

    @torch._functorch.config.patch(
        {
            "enable_autograd_cache": True,
            "strict_autograd_cache": True,
        }
    )
    @torch._inductor.config.patch("fx_graph_cache", True)
    def verify_aot_autograd(
        self,
        f,
        inp_: Callable | list[Any],
        *,
        test_mutation: bool = False,
        keep_inp_mutations: bool = False,
        decompositions: dict | None = None,
        dynamic: bool = False,
        # Only active when inp_ is Callable.
        # TODO: probably consolidate all tests to make inp a Callable.
        make_inputs_subclasses: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Override of the verification harness: with the autograd cache
        # and strict mode on and the inductor graph cache enabled, creates a fresh in-memory
        # cache, clears the autograd cache, patches graph cache loading to use the mock, and
        # delegates to the parent harness.
        self.inductor_cache = MockFXGraphCache()
        AOTAutogradCache.clear()
        with patch(
            "torch._inductor.codecache.FxGraphCache.load_with_key",
            new=self.inductor_cache.load_with_key,
        ):
            return super().verify_aot_autograd(
                f,
                inp_,
                test_mutation=test_mutation,
                keep_inp_mutations=keep_inp_mutations,
                decompositions=decompositions,
                dynamic=dynamic,
                make_inputs_subclasses=make_inputs_subclasses,
            )

    def test_input_mutation_false_aliasing(self):
        # [FUNCTION DOC] Disabled: skips immediately because the test fails in strict cache mode
        # and cannot be marked expected failure (it causes undefined behavior in address
        # sanitizer runs).
        # This test is disabled because it fails in strict cache mode
        # But also can't be xfailed because it causes undefined behavior for
        # ASAN
        self.skipTest("Skipping because it fails in strict cache mode")


if __name__ == "__main__":
    run_tests()
