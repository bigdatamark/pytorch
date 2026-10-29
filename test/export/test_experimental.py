# Owner(s): ["oncall: export"]
# flake8: noqa
# [MODULE DOC] Tests for experimental export features: forward-backward joint export, sticky
# export, dynamo graph capture for export (public and private variants), and BlockMask / mask_mod
# closure extraction into pytree leaves for flex attention.
# Contains a small pytree-flattenable context class used as a dynamo output type, a helper that
# registers BlockMask as a pytree node, and one test case class. Many tests need CUDA and are
# skipped otherwise.
# Import time effect: defines a module-level global list used by a side-effect test; no
# registration happens until a test runs.
import contextlib
import copy
import functools
import types
import unittest
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch._dynamo
from torch._dynamo.functional_export import (
    _dynamo_graph_capture_for_export,
    dynamo_graph_capture_for_export,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._export.utils import _compiling_state_context
from torch._functorch.aot_autograd import (
    aot_export_joint_with_descriptors,
    aot_export_module,
)
from torch._guards import tracing as torch_tracing, TracingContext
from torch.export import export
from torch.export.experimental import _export_forward_backward, _sticky_export
from torch.export.graph_signature import OutputKind
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import (
    IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED,
)
from torch.testing._internal.common_utils import TEST_CUDA
from torch.utils import _pytree as pytree


GLOBAL_LIST = []


def _register_blockmask_pytree():
    # [FUNCTION DOC] HELPER: Import BlockMask and the pytree supported-node table from the flex
    # attention module and, only if BlockMask is not already registered, register it as a pytree
    # node using its own flatten, unflatten and keyed flatten methods plus a serialized type
    # name. Idempotent; mutates global pytree registry.
    """Register BlockMask as a pytree node if not already registered."""
    from torch.nn.attention.flex_attention import BlockMask
    from torch.utils._pytree import register_pytree_node, SUPPORTED_NODES

    if BlockMask not in SUPPORTED_NODES:
        register_pytree_node(
            BlockMask,
            BlockMask._flatten,
            BlockMask._unflatten,
            flatten_with_keys_fn=BlockMask._flatten_with_keys,
            serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
        )


class GlobalContext:
    # [CLASS DOC] HELPER: Test fixture that acts as both a context manager and a
    # pytree-flattenable container. State is two dicts, one of summaries and one of tensors, both
    # initially empty. It provides flatten and unflatten methods suitable for registering as a
    # custom pytree node. Entering returns itself and exiting does nothing.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create empty summary and tensor dictionaries. KNOWN ISSUE: the
        # annotations name MetricValue and Tensor, which are not defined or imported in this
        # module; harmless only because annotations on attributes inside a function body are
        # never evaluated.
        self._summaries: dict[str, MetricValue] = {}
        self._tensors: dict[str, Tensor] = {}

    def __flatten__(self):
        # [FUNCTION DOC] HELPER: Flatten both dicts with pytree. Returns a pair: the leaves (all
        # summary leaves followed by all tensor leaves, as a tuple) and a context tuple holding
        # the two tree specs.
        """Flattens into (leaves, ctx)."""
        summary_leaves, summary_spec = pytree.tree_flatten(self._summaries)
        tensor_leaves, tensor_spec = pytree.tree_flatten(self._tensors)
        leaves = (*summary_leaves, *tensor_leaves)
        ctx = (summary_spec, tensor_spec)
        return leaves, ctx

    @classmethod
    def __unflatten__(cls, leaves, ctx: tuple[pytree.TreeSpec, pytree.TreeSpec]):
        # [FUNCTION DOC] HELPER: Class method inverse of flatten. Takes the leaves and the
        # context pair of specs, raises AssertionError if the number of leaves differs from the
        # combined leaf count of both specs, then creates a new instance and rebuilds its summary
        # dict from the first leaves and its tensor dict from the rest. Returns the new instance.
        """Reconstructs from (leaves, ctx)."""
        output = cls()
        summary_spec, tensor_spec = ctx
        if len(leaves) != summary_spec.num_leaves + tensor_spec.num_leaves:
            raise AssertionError(
                f"Expected {summary_spec.num_leaves + tensor_spec.num_leaves} leaves, "
                f"got {len(leaves)}"
            )
        output._summaries = pytree.tree_unflatten(
            leaves[: summary_spec.num_leaves], summary_spec
        )
        output._tensors = pytree.tree_unflatten(
            leaves[summary_spec.num_leaves :], tensor_spec
        )
        return output

    def __enter__(self) -> "GlobalContext":
        # [FUNCTION DOC] HELPER: Context manager entry; returns the same object.
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # [FUNCTION DOC] HELPER: Context manager exit; does nothing and returns None, so
        # exceptions are not suppressed.
        pass


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestExperiment(TestCase):
    # [CLASS DOC] Test case for experimental export APIs. Groups tests for joint forward/backward
    # export, sticky export wrapping, dynamo graph capture for export, flex attention BlockMask
    # closure extraction and tracing, and restoring state dict names. The whole class is skipped
    # when dynamo is unsupported.
    def test_joint_basic(self) -> None:
        # [FUNCTION DOC] Export a small linear plus cross entropy module strictly with the new
        # experimental tracer, convert it to a joint forward-backward program and compare the
        # generated code with an expected snapshot; then run decompositions and compare again
        # with a second snapshot (alias nodes renamed).
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module made of a three by three linear layer and a cross
            # entropy loss.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer and the loss module.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.loss = torch.nn.CrossEntropyLoss()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply softmax over dim 0 to the linear output and return
                # the cross entropy loss against a fixed one-hot target vector.
                return self.loss(
                    self.linear(x).softmax(dim=0), torch.tensor([1.0, 0.0, 0.0])
                )

        m = Module()
        example_inputs = (torch.randn(3),)
        m(*example_inputs)
        with torch._export.config.patch(use_new_tracer_experimental=True):
            ep = torch.export.export(m, example_inputs, strict=True)
        joint_ep = _export_forward_backward(ep)
        self.assertExpectedInline(
            str(joint_ep.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, c_lifted_tensor_0, x):
    view = torch.ops.aten.view.default(x, [1, 3]);  x = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, view, permute);  p_linear_bias = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [3]);  addmm = None
    _softmax = torch.ops.aten._softmax.default(view_1, 0, False);  view_1 = None
    alias_1 = torch.ops.aten.alias.default(_softmax)
    clone = torch.ops.aten.clone.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_3 = torch.ops.aten.alias.default(_log_softmax)
    mul = torch.ops.aten.mul.Tensor(_log_softmax, clone);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, clone);  expand = clone = None
    alias_4 = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    exp = torch.ops.aten.exp.default(alias_4);  alias_4 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_6 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_6);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_6, sum_3);  alias_6 = sum_3 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    view_2 = torch.ops.aten.view.default(sub_1, [1, 3]);  sub_1 = None
    permute_1 = torch.ops.aten.permute.default(view_2, [1, 0])
    mm = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
    sum_4 = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
    view_3 = torch.ops.aten.view.default(sum_4, [3]);  sum_4 = None
    permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return (div, permute_3, view_3)""",
        )
        ep = joint_ep.run_decompositions()
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, c_lifted_tensor_0, x):
    view = torch.ops.aten.view.default(x, [1, 3]);  x = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, view, permute);  p_linear_bias = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [3]);  addmm = None
    _softmax = torch.ops.aten._softmax.default(view_1, 0, False);  view_1 = None
    alias = torch.ops.aten.alias.default(_softmax)
    clone = torch.ops.aten.clone.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_1 = torch.ops.aten.alias.default(_log_softmax)
    mul = torch.ops.aten.mul.Tensor(_log_softmax, clone);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, clone);  expand = clone = None
    alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    exp = torch.ops.aten.exp.default(alias_2);  alias_2 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_3 = torch.ops.aten.alias.default(alias);  alias = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_3);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_3, sum_3);  alias_3 = sum_3 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    view_2 = torch.ops.aten.view.default(sub_1, [1, 3]);  sub_1 = None
    permute_1 = torch.ops.aten.permute.default(view_2, [1, 0])
    mm = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
    sum_4 = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
    view_3 = torch.ops.aten.view.default(sum_4, [3]);  sum_4 = None
    permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return (div, permute_3, view_3)""",
        )

    def _test_export_blockmask_with_mask_fn(self, make_mask_fn):
        # [FUNCTION DOC] HELPER: Shared body for block mask export tests. Registers BlockMask as
        # a pytree node, builds a model whose forward constructs a block mask from a mask
        # function produced by the given factory, captures it with the private dynamo export
        # capture function on CUDA input, and checks the captured graph returns the same tensor
        # and an equivalent mask_mod result as eager execution. Requires CUDA.
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that creates a mask function from a stored factory
            # and builds a 64 by 64 block mask from it.
            def __init__(self, mask_fn_factory):
                # [FUNCTION DOC] HELPER: Store the mask function factory.
                super().__init__()
                self.mask_fn_factory = mask_fn_factory

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Call the factory for a fresh mask function, build a
                # block mask with batch 1, heads 1 and length 64 for queries and keys on the
                # input device, and return the input together with the block mask.
                mask_fn = self.mask_fn_factory()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model(make_mask_fn)

        out_eager, mask_eager = module(x)

        compiled = _dynamo_graph_capture_for_export(module)(x)
        out_compiled, mask_compiled = compiled(x)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(
            mask_eager.mask_mod(1, 1, 64, 64),
            mask_compiled.mask_mod(1, 1, 64, 64),
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask(self):
        # [FUNCTION DOC] CUDA only. Export a model whose mask function is a nested closure
        # capturing an int, using the shared block mask test body.
        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory returning a mask function whose causal check is
            # shifted by a captured integer.
            res = 4

            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function that is true where the query index is at
                # least key index plus the captured constant.
                return q >= k + res

            return fn

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    def test_export_with_default_kwargs(self):
        # [FUNCTION DOC] Capture a module wrapper whose __call__ has a keyword-only parameter
        # with a default (and forwards other arguments to an inner linear layer) with the public
        # capture function. Checks captured and eager outputs agree, including the passed-through
        # default, and compares the generated code with a snapshot.
        class FunctionalWrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Local wrapper module overriding __call__ with a keyword-only
            # default context argument.
            """Wrapper with keyword-only argument in __call__."""

            def __init__(self, module):
                # [FUNCTION DOC] HELPER: Store the wrapped module.
                super().__init__()
                self._module = module

            def __call__(self, *args, _ctx=None, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped module with the positional and keyword
                # arguments and return a pair of its output and the keyword-only context value
                # (None by default).
                # _ctx is keyword-only with default None
                out = self._module(*args, **kwargs)
                return out, _ctx

        inner = torch.nn.Linear(4, 4)
        wrapper = FunctionalWrapper(inner)
        x = torch.randn(2, 4, requires_grad=True)

        gm = dynamo_graph_capture_for_export(wrapper)(x)
        compile_out, compile_ctx = gm(x)
        eager_out, eager_ctx = wrapper(x)
        self.assertEqual(compile_ctx, eager_ctx)
        self.assertEqual(compile_out, eager_out)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_self_modules_module_parameters_weight_ , L_self_modules_module_parameters_bias_ , L_args_0_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_self_modules_module_parameters_weight_ = L_self_modules_module_parameters_weight_
    l_self_modules_module_parameters_bias_ = L_self_modules_module_parameters_bias_
    l_args_0_ = L_args_0_
    out = torch._C._nn.linear(l_args_0_, l_self_modules_module_parameters_weight_, l_self_modules_module_parameters_bias_);  l_args_0_ = l_self_modules_module_parameters_weight_ = l_self_modules_module_parameters_bias_ = None
    return self._dynamo_bytecode_unflatten((out,), _fn_args)""",
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_mutated_closure(self):
        # [FUNCTION DOC] CUDA only. A mask function closure whose captured integer is rebound
        # after the function is defined must still be exported using the final value; uses the
        # shared block mask test body.
        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory that defines a mask function capturing a variable,
            # then reassigns the variable and returns the function.
            res = 1

            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function comparing the query index with key index
                # plus the captured variable.
                return q >= k + res

            res = 4  # mutation after function definition
            return fn

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_with_containers(self):
        # [FUNCTION DOC] CUDA only. Export a mask function whose closure captures a list and a
        # nested dictionary, via the shared body.
        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory creating a list of offsets and a nested config dict
            # and returning a mask function using them.
            offsets = [1, 2, 3]
            config = {"base": 4, "nested": {"scale": 2}}

            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function true where query index is at least key
                # index plus the base config value plus the sum of the offsets.
                return q >= k + config["base"] + sum(offsets)

            return fn

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_triple_nested(self):
        # [FUNCTION DOC] CUDA only. Export a mask function whose captured variables come from
        # three nested enclosing functions, via the shared body.
        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Outermost factory defining a captured variable and an inner
            # function, returning the result of calling the inner function.
            a = 1

            def level1():
                # [FUNCTION DOC] HELPER: Middle level defining a second captured variable and a
                # nested function; returns that function's result.
                b = 2

                def level2():
                    # [FUNCTION DOC] HELPER: Innermost enclosing level defining a third captured
                    # variable and returning the mask function.
                    c = 3

                    def fn(bx, h, q, k):
                        # [FUNCTION DOC] HELPER: Mask function using the variables from all three
                        # enclosing levels in its comparison.
                        return q >= k + a + b + c

                    return fn

                return level2()

            return level1()

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_self_recursive(self):
        # [FUNCTION DOC] CUDA only. A mask function that refers to itself through its closure
        # cannot be reconstructed; capturing a model that returns such a block mask must raise a
        # dynamo Unsupported error mentioning a nested function with non-constructible closure
        # in output.
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory returning a mask function that refers to itself.
            # Self-referential: fn captures itself through the closure
            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function that references itself in its body and
                # compares query index with key index plus four.
                _ = fn  # self-reference
                return q >= k + 4

            return fn

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module building a 64 by 64 block mask from the
            # self-referential mask function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Build the mask function, create a block mask and return
                # the input with it.
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_tensor(self):
        # [FUNCTION DOC] CUDA only. A mask function that captures itself and a tensor is not
        # constructible; capturing must raise the dynamo Unsupported error about
        # non-constructible closure in output.
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory creating a captured tensor and returning a mask
            # function that refers to itself and uses the tensor sum.
            tensor = torch.ones(2, 2)

            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function referencing itself and adding the captured
                # tensor sum to the comparison.
                _ = fn
                return q >= k + 4 + tensor.sum()

            return fn

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module building a 64 by 64 block mask from the
            # factory function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Build the mask function, create a block mask and return
                # the input with it.
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_unsupported_class_instance(self):
        # [FUNCTION DOC] CUDA only. A mask function capturing an instance of an arbitrary user
        # class is unsupported; capturing must raise the dynamo Unsupported error about
        # non-constructible closure in output.
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        class MaskConfig:
            # [CLASS DOC] HELPER: Plain user class holding an offset, used as an unsupported
            # closure value.
            def __init__(self, offset):
                # [FUNCTION DOC] HELPER: Store the offset.
                self.offset = offset

        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory creating a config object and returning a mask
            # function that uses its offset.
            cfg = MaskConfig(offset=5)

            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function comparing query index with key index plus
                # the config offset.
                return q >= k + cfg.offset

            return fn

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module building a 64 by 64 block mask from the
            # factory function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Build the mask function, create a block mask and return
                # the input with it.
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_mutually_recursive(self):
        # [FUNCTION DOC] CUDA only. Two mask functions that refer to each other cannot be
        # serialized; capturing must raise the dynamo Unsupported error about non-constructible
        # closure in output.
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory defining two mutually referencing functions and
            # returning the first.
            # Create mutually recursive closures: fn_a references fn_b, fn_b references fn_a
            # This is non-constructible because we cannot serialize mutually recursive closures
            def fn_a(b, h, q, k):
                # [FUNCTION DOC] HELPER: First mask function; refers to the second function and
                # compares query index with key index.
                _ = fn_b  # reference to fn_b
                return q >= k

            def fn_b(b, h, q, k):
                # [FUNCTION DOC] HELPER: Second mask function; refers to the first function and
                # compares query index with key index plus one.
                _ = fn_a  # reference to fn_a
                return q >= k + 1

            return fn_a

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module building a 64 by 64 block mask from the mutually
            # recursive function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Build the mask function, create a block mask and return
                # the input with it.
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    @unittest.skipUnless(
        IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip,
        "Requires CUDA with SM >= 8.0, Triton, and not ROCm",
    )
    def test_aot_export_flex_attention_callable_mask_mod(self):
        # [FUNCTION DOC] Runs only on supported flex attention CUDA platforms that are not ROCm.
        # Exports a flex attention model whose mask_mod is an instance of a callable class (with
        # its own equality and hash) with the AOT export module function, and checks the exported
        # graph output matches eager output in shape and within a small tolerance.
        """Test flex_attention AOT export with callable class as mask_mod.

        _MaskModWrapper must delegate __eq__ to callable objects for TreeSpec
        comparison in AOTAutograd's PytreeThunk.set() (utils.py:162).
        """
        from torch._functorch.aot_autograd import aot_export_module
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        _register_blockmask_pytree()

        class ComposedMaskMod:
            # [CLASS DOC] HELPER: Callable class combining several mask functions with logical
            # and; defines equality and hash by the tuple of functions.
            def __init__(self, *mask_fns):
                # [FUNCTION DOC] HELPER: Store the given mask functions as a tuple.
                self.mask_fns = mask_fns

            def __call__(self, b, h, q, k):
                # [FUNCTION DOC] HELPER: Start from True and combine with each stored mask
                # function applied to the same indices using bitwise and; return the result.
                result = True
                for fn in self.mask_fns:
                    result = result & fn(b, h, q, k)
                return result

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: return NotImplemented for other types,
                # otherwise compare the tuples of functions.
                if not isinstance(other, ComposedMaskMod):
                    return NotImplemented
                return self.mask_fns == other.mask_fns

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of the tuple of functions.
                return hash(self.mask_fns)

        def causal_mask(b, h, q, k):
            # [FUNCTION DOC] HELPER: Causal mask function: true where query index is at least
            # key index.
            return q >= k

        class FlexAttentionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Single layer attention model using flex attention with a causal
            # block mask built inside forward.
            def __init__(self, embed_dim: int, num_heads: int):
                # [FUNCTION DOC] HELPER: Store head count and head dimension and create query,
                # key and value linear projections.
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
                self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
                self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Project the input into query, key and value, reshape to
                # heads and transpose, build a causal block mask from the callable mask class,
                # run flex attention and return a one element tuple with the output reshaped back
                # to batch, length, embedding.
                B, L, D = x.shape
                q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
                k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
                v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

                mask_mod = ComposedMaskMod(causal_mask)
                block_mask = create_block_mask(
                    mask_mod, B=B, H=self.num_heads, Q_LEN=L, KV_LEN=L, device=x.device
                )
                out = flex_attention(q, k, v, block_mask=block_mask)
                return (out.transpose(1, 2).contiguous().view(B, L, D),)

        embed_dim, num_heads, seq_len = 64, 2, 128
        model = FlexAttentionModel(embed_dim, num_heads).cuda()
        x = torch.randn(1, seq_len, embed_dim, device="cuda")

        gm, signature = aot_export_module(model, [x], trace_joint=False)

        # aot_export_module flattens params/buffers into the graph signature
        params = [p for p in model.parameters()]
        out_eager = model(x)[0]
        out_export = gm(*params, x)[0]
        self.assertEqual(out_eager.shape, out_export.shape)
        self.assertTrue(torch.allclose(out_eager, out_export, atol=1e-5))

    @unittest.skipUnless(
        IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip,
        "Requires CUDA with SM >= 8.0, Triton, and not ROCm",
    )
    def test_aot_export_flex_attention_with_blockmask_placeholders(self):
        # [FUNCTION DOC] Runs only on supported flex attention CUDA platforms that are not ROCm.
        # Build a model holding a block mask attribute on the meta device, convert its parameters
        # to fake CUDA parameters under fake tensor mode, capture with the public function,
        # assert that the graph has placeholders for the block mask, and check the joint export
        # with descriptors returns a graph module.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        _register_blockmask_pytree()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a bias-free linear layer and a block mask
            # attribute created at construction time.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer and a causal block mask of
                # length 16 on CUDA.
                super().__init__()
                self.wq = torch.nn.Linear(64, 64, bias=False)
                self.block_mask = create_block_mask(
                    lambda b, h, q, kv: q >= kv,
                    B=None,
                    H=None,
                    Q_LEN=16,
                    KV_LEN=16,
                    device="cuda",
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Project the input, reshape to four heads and run flex
                # attention using the stored block mask; returns the sum of the result.
                q = self.wq(x).view(1, 16, 4, 16).transpose(1, 2)
                return flex_attention(q, q, q, block_mask=self.block_mask).sum()

        with torch.device("meta"):
            model = Model()

        fake_mode = FakeTensorMode()
        with fake_mode:
            for name, param in list(model.named_parameters()):
                parts = name.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                setattr(
                    mod,
                    parts[-1],
                    torch.nn.Parameter(
                        torch.empty(param.shape, dtype=param.dtype, device="cuda"),
                        requires_grad=param.requires_grad,
                    ),
                )
            x = torch.randn(1, 16, 64, device="cuda")

        gm = dynamo_graph_capture_for_export(model)(x)
        block_mask_placeholders = [
            node
            for node in gm.graph.nodes
            if node.op == "placeholder" and "block_mask" in node.name
        ]
        self.assertGreater(len(block_mask_placeholders), 0)

        with contextlib.ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(stack, gm, (x,))

        self.assertIsNotNone(joint_with_descriptors.graph_module)

    def test_joint_dynamic(self) -> None:
        # [FUNCTION DOC] Export a module with a parameter and a dynamically sized input (Dim
        # for dim 0) strictly, then convert to a forward-backward program; passes if no
        # exception is raised.
        from torch.export import Dim

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a parameter that adds it to a tensor of ones
            # sized by the input batch.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the parameter.
                super().__init__()
                self.y = torch.nn.Parameter(torch.randn(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Ignore the input values, build a tensor of ones with the
                # batch size of the input and three columns, and return the sum of the parameter
                # plus it.
                x = torch.ones(x.shape[0], 3)
                return (self.y + x).sum()

        m = Module()
        example_inputs = (torch.randn(3),)
        m(*example_inputs)
        ep = torch.export.export(
            m, example_inputs, dynamic_shapes={"x": {0: Dim("x0")}}, strict=True
        )
        _export_forward_backward(ep)

    def test_joint_cifar10_backwards(self) -> None:
        # [FUNCTION DOC] Export a small convolutional classifier (the tutorial CIFAR10 network
        # with a cross entropy loss) strictly and convert it to a forward-backward program;
        # passes if no exception is raised.
        import torch.nn as nn
        import torch.nn.functional as F

        # From Pytorch's CIFAR10 example:
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Local convolution and fully connected network with a cross
            # entropy loss on the labels.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create two convolutions, a pooling layer, three linear
                # layers and the loss.
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                self.loss = nn.CrossEntropyLoss()

            def forward(self, x, labels):
                # [FUNCTION DOC] HELPER: Run convolution, relu and pooling twice, flatten, three
                # linear layers with relu between them, and return the cross entropy loss against
                # the labels.
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)  # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return self.loss(x, labels)

        net = Net()
        x = torch.randn(4, 3, 32, 32)
        labels = torch.ones(4, dtype=torch.int64)
        inputs = (x, labels)

        ep = export(net, inputs, strict=True)
        ep = _export_forward_backward(ep)

    def test_joint_loss_index(self):
        # [FUNCTION DOC] For each possible loss index of a two output module, export strictly and
        # convert to forward-backward using that loss index; the output spec at that index must
        # be the loss output and no other output spec may be.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the loss and its detached negative or
            # the detached loss and the loss, depending on a constructor argument.
            def __init__(self, index):
                # [FUNCTION DOC] HELPER: Create a linear layer and store the index.
                super().__init__()
                self.l = torch.nn.Linear(4, 4)
                self.index = index

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sum the linear output; return the pair of that sum and
                # its negated detached copy if the index is zero, otherwise the pair of the
                # detached sum and the sum.
                x = self.l(x)
                x = x.sum()
                if self.index == 0:
                    return x, -x.detach()
                else:
                    return x.detach(), x

        inputs = (torch.randn(4, 4),)
        for i in [0, 1]:
            ep = export(Foo(i), inputs, strict=True)
            ep_joint = _export_forward_backward(ep, joint_loss_index=i)
            for j, spec in enumerate(ep_joint.graph_signature.output_specs):
                if i == j:
                    self.assertTrue(spec.kind == OutputKind.LOSS_OUTPUT)
                else:
                    self.assertTrue(spec.kind != OutputKind.LOSS_OUTPUT)

    def test_joint_buffer_input_mutations(self):
        # [FUNCTION DOC] Export a module that mutates both a user input and a buffer then convert
        # to forward-backward. Expect five output specs: the first a buffer mutation targeting
        # buf, the second a user input mutation targeting x, the third the loss output.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a linear layer, a buffer and a loss that
            # mutates both the input and the buffer in forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer, the buffer and the loss.
                super().__init__()
                self.l = torch.nn.Linear(4, 4)
                self.register_buffer("buf", torch.randn(4))
                self.loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, label):
                # [FUNCTION DOC] HELPER: Add the buffer to the input in place, apply the linear
                # layer, add two to the buffer in place and return the cross entropy loss.
                x.add_(self.buf)
                x = self.l(x)
                self.buf.add_(2.0)
                return self.loss(x, label)

        inputs = (
            torch.randn(4, 4),
            torch.randint(0, 4, (4,)),
        )
        ep = export(Foo(), inputs)
        ep_joint = _export_forward_backward(ep)
        self.assertEqual(len(ep_joint.graph_signature.output_specs), 5)
        self.assertEqual(
            ep_joint.graph_signature.output_specs[0].kind,
            OutputKind.BUFFER_MUTATION,
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[0].target,
            "buf",
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[1].kind,
            OutputKind.USER_INPUT_MUTATION,
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[1].target,
            "x",
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[2].kind,
            OutputKind.LOSS_OUTPUT,
        )

    def test_sticky_export(self):
        # [FUNCTION DOC] Wrap the forward method of a model with sticky export, call it through a
        # small pipeline object, then restore the original forward and call again; the two
        # results must be close.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a single linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the linear layer applied to the input.
                return self.linear(x)

        class Pipeline:
            # [CLASS DOC] HELPER: Local object that owns a model and forwards generate calls
            # to it.
            def __init__(self, model):
                # [FUNCTION DOC] HELPER: Store the model.
                self.model = model

            def generate(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the model with the given positional and keyword
                # arguments and return its result.
                return self.model(*args, **kwargs)

        inp = torch.randn(4, 4)

        p = Pipeline(Model())
        orig_forward = p.model.forward
        p.model.forward = _sticky_export(p.model.forward)
        res = p.generate(inp)

        p.model.forward = orig_forward
        res2 = p.generate(inp)
        self.assertTrue(torch.allclose(res, res2))

    def test_sticky_export_dynamic(self):
        # [FUNCTION DOC] Use sticky export with a callback that marks every dimension of every
        # tensor argument as automatic dynamic through a shape collection. Calls the pipeline
        # once on an input with batch four and compares the exported artifact code with a
        # snapshot showing only the linear branch.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that uses its linear layer when the batch is under
            # five, otherwise the sine.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the linear output when the first dimension is
                # less than five, otherwise the sine of the input.
                if x.shape[0] < 5:
                    return self.linear(x)
                return x.sin()

        class Pipeline:
            # [CLASS DOC] HELPER: Local object that owns a model and forwards generate calls
            # to it.
            def __init__(self, model):
                # [FUNCTION DOC] HELPER: Store the model.
                self.model = model

            def generate(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the model with the given arguments and return
                # its result.
                return self.model(*args, **kwargs)

        inp = torch.randn(4, 4)

        def callback(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Sticky export dynamic shapes callback. Flattens the call
            # arguments, and for every tensor builds an entry in a ShapesCollection that marks
            # every dimension as automatic. Returns the collection.
            # I think it is bit weird to use the forward arg name here, so
            # lets just use ShapeCollections

            flat_args, _ = torch.utils._pytree.tree_flatten((args, kwargs))
            collections = torch.export.ShapesCollection()
            for arg in flat_args:
                if isinstance(arg, torch.Tensor):
                    collections[arg] = {
                        i: torch.export.Dim.AUTO for i in range(len(arg.shape))
                    }
            return collections

        p = Pipeline(Model())
        p.model.forward = _sticky_export(
            p.model.forward, dynamic_shapes_callback=callback
        )
        _ = p.generate(inp)
        self.assertExpectedInline(
            str(p.model.forward._exported_artifact.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    linear_weight = self.linear.weight
    linear_bias = self.linear.bias
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    linear_default = torch.ops.aten.linear.default(x, linear_weight, linear_bias);  x = linear_weight = linear_bias = None
    return pytree.tree_unflatten((linear_default,), self._out_spec)""",
        )

    def test_sticky_export_nested_inp(self):
        # [FUNCTION DOC] Sticky export where the model forward takes a keyword-only list
        # argument; wrap, call through a pipeline with keyword tensors, restore and call again;
        # results must be close.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module applying a linear layer to each of two list
            # elements and adding them; forward takes keyword-only inputs.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, *, inputs):
                # [FUNCTION DOC] HELPER: Return the sum of the linear layer applied to the first
                # two items of the inputs.
                return self.linear(inputs[0]) + self.linear(inputs[1])

        class Pipeline:
            # [CLASS DOC] HELPER: Local object that packs two tensors into a list and calls its
            # model with the keyword argument.
            def __init__(self, model):
                # [FUNCTION DOC] HELPER: Store the model.
                self.model = model

            def generate(self, *, input_tensor, input_tensor2):
                # [FUNCTION DOC] HELPER: Build a list of the two keyword tensors and return the
                # model result for it.
                inputs = [input_tensor, input_tensor2]
                return self.model(inputs=inputs)

        inp = torch.randn(4, 4)
        inp2 = torch.randn(4, 4)

        p = Pipeline(Model())
        orig_forward = p.model.forward
        p.model.forward = _sticky_export(p.model.forward)
        res = p.generate(input_tensor=inp, input_tensor2=inp2)

        p.model.forward = orig_forward
        res2 = p.generate(input_tensor=inp, input_tensor2=inp2)
        self.assertTrue(torch.allclose(res, res2))

    def test_side_effect(self):
        # [FUNCTION DOC] With side effect replay turned off by config, capture a module that
        # appends its input to an outer list; the list must remain empty after capture.
        global_env = []

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that appends the input to an enclosing list and
            # returns the sine.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Append the input to the enclosing list and return the
                # sine of the input.
                global_env.append(x)
                return x.sin()

        with torch._dynamo.config.patch(replay_side_effects=False):
            _ = dynamo_graph_capture_for_export(Foo())(torch.randn(4, 4))
            self.assertEqual(len(global_env), 0)

    def test_export_add_in_out_info(self):
        # [FUNCTION DOC] Capture a module with a dict, a nested list and an int input whose
        # outputs include inputs, a copied and extended list, new dict entries and constant
        # lists; the captured module result must equal eager execution on a deep copy of
        # the inputs.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that multiplies values of the dict and list
            # inputs, mutates a copy of the list and returns several values.
            def forward(self, dct, lst, bleh):
                # [FUNCTION DOC] HELPER: Compute products of dict and list entries, extend a copy
                # of the list with its first element, and return a tuple of the first dict entry,
                # product y, the int input, the last item of the extended copy, product x and a
                # constant list.
                x = dct["a"] * lst[1][0]
                y = dct["b"] * lst[0]
                out_dict = {}
                # Mutate and get a new entry in there
                lst_copy = lst.copy()
                lst_copy.append(lst[0])
                out_dict["a"] = x
                out_dict["b"] = y
                return (
                    dct["a"],
                    out_dict["b"],
                    bleh,
                    lst_copy[-1],
                    out_dict["a"],
                    [5, 6],
                )

        dct = {"a": torch.randn(2, 3), "b": torch.randn(2, 3)}
        lst = [torch.randn(2, 3), [torch.randn(2, 3), torch.randn(2, 3)]]

        export_inputs = ((dct, lst, 56), {})
        eager_inputs = copy.deepcopy(export_inputs)

        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        graph_module = dynamo_graph_capture_for_export(Foo())(
            *export_inputs[0], **export_inputs[1]
        )

        res_export = graph_module(*export_inputs[0], **export_inputs[1])
        res_eager = Foo()(*eager_inputs[0], **eager_inputs[1])

        self.assertEqual(res_export, res_eager)

    def test_single_op(self):
        # [FUNCTION DOC] Capture a bare aten operator (addition, then range with an out tensor)
        # with the public capture function and compare the generated code against snapshots.
        # from torch._dynamo.functional_export import dynamo_graph_capture_for_export
        x, y = (torch.randn(2, 3), torch.randn(2, 3))
        graph = dynamo_graph_capture_for_export(torch.ops.aten.add.Tensor)(x, y)
        self.assertExpectedInline(
            graph.code.strip("\r\n "),
            """\
def forward(self, other):
    _fn_args = (self, other)
    self, L_self_ , L_other_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_self_ = L_self_
    l_other_ = L_other_
    add_tensor = torch.ops.aten.add.Tensor(self = l_self_, other = l_other_, alpha = 1);  l_self_ = l_other_ = None
    return self._dynamo_bytecode_unflatten((add_tensor,), _fn_args)""",
        )
        out = torch.empty(10)
        graph = dynamo_graph_capture_for_export(torch.ops.aten.range.out)(0, 9, out=out)
        self.assertExpectedInline(
            graph.code.strip("\r\n "),
            """\
def forward(self, start, end, out):
    _fn_args = (start, end, out)
    L_out_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_out_ = L_out_
    range_out = torch.ops.aten.range.out(start = 0, end = 9, step = 1, out = l_out_);  l_out_ = None
    return self._dynamo_bytecode_unflatten((range_out,), _fn_args)""",
        )

    def test_export_leaf(self):
        # [FUNCTION DOC] Capture a simple module and check the captured module output equals the
        # eager result.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the sine of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the sine of the input.
                return x.sin()

        export_inputs = ((torch.randn(4, 4),), {})
        eager_inputs = copy.deepcopy(export_inputs)

        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        graph_module = dynamo_graph_capture_for_export(Foo())(
            *export_inputs[0], **export_inputs[1]
        )

        res_export = graph_module(*export_inputs[0], **export_inputs[1])
        res_eager = Foo()(*eager_inputs[0], **eager_inputs[1])

        self.assertEqual(res_export, res_eager)

    def test_dynamo_graph_capture(self):
        # [FUNCTION DOC] Capture a module with dict, list and tensor inputs and structured
        # outputs; compare the captured module with eager on fresh inputs of the same structure.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with products of dict and list entries and
            # structured output.
            def forward(self, dct, lst, bleh):
                # [FUNCTION DOC] HELPER: Same computation as the add-in-out test: returns the
                # first dict entry, a product, the third input, the last item of an extended list
                # copy, another product and a constant list.
                x = dct["a"] * lst[1][0]
                y = dct["b"] * lst[0]
                out_dict = {}

                # Mutate and get a new entry in there
                lst_copy = lst.copy()
                lst_copy.append(lst[0])
                out_dict["a"] = x
                out_dict["b"] = y
                return (
                    dct["a"],
                    out_dict["b"],
                    bleh,
                    lst_copy[-1],
                    out_dict["a"],
                    [5, 6],
                )

        foo = Foo()

        def make_inputs():
            # [FUNCTION DOC] HELPER: Build fresh example inputs: a dict of two tensors, a list
            # holding a tensor and a one-element tuple, and a tensor.
            return (
                {"a": torch.randn(2, 3), "b": torch.randn(2, 3)},
                [torch.randn(2, 3), (torch.randn(2, 3),)],
                torch.randn(2, 3),
            )

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))

    def test_dynamo_graph_capture_with_call_override(self):
        # [FUNCTION DOC] Capture a module whose call method is overridden to forward to an inner
        # module; the captured module must agree with eager and have the same number of buffers
        # and parameters.
        class _InterestingModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Local wrapper module whose call method forwards to a
            # stored module.
            def __init__(self, module):
                # [FUNCTION DOC] HELPER: Store the wrapped module.
                super().__init__()
                self._module = module

            def __call__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped module with all arguments and return
                # its result.
                return self._module(*args, **kwargs)

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that adds one to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input plus one.
                return x + 1

        foo = _InterestingModule(MyModel())

        def make_inputs():
            # [FUNCTION DOC] HELPER: Build a one element tuple with a random tensor of two
            # by three.
            return (torch.randn(2, 3),)

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        self.assertEqual(len(list(gm.buffers())), len(list(foo.buffers())))
        self.assertEqual(len(list(gm.parameters())), len(list(foo.parameters())))

    def test_dynamo_graph_capture_custom_pytree_type(self):
        # [FUNCTION DOC] Register a dataclass as a pytree node, capture a module taking it,
        # compare generated code with a snapshot and check outputs against eager. The
        # registration is undone in a finally block.
        import torch.utils._pytree as pytree

        @dataclass
        class Bar:
            # [CLASS DOC] HELPER: Dataclass with two tensor fields.
            x: torch.Tensor
            y: torch.Tensor

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the two fields of the dataclass input.
            def forward(self, bar: Bar):
                # [FUNCTION DOC] HELPER: Return the sum of the two tensor fields.
                return bar.x + bar.y

        foo = Foo()

        def make_inputs():
            # [FUNCTION DOC] HELPER: Build a one element tuple holding the dataclass with two
            # random tensors.
            return (Bar(torch.randn(2, 3), torch.randn(2, 3)),)

        pytree.register_dataclass(Bar)
        try:
            trace_inputs = make_inputs()
            gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
            test_inputs = make_inputs()
            self.assertExpectedInline(
                gm.code.strip("\r\n "),
                """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_bar_x , L_bar_y , = self._dynamo_bytecode_flatten(*_fn_args)
    l_bar_x = L_bar_x
    l_bar_y = L_bar_y
    add = l_bar_x + l_bar_y;  l_bar_x = l_bar_y = None
    return self._dynamo_bytecode_unflatten((add,), _fn_args)""",
            )
            self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        finally:
            pytree._deregister_pytree_node(Bar)

    def test_dynamo_graph_capture_closure(self):
        # [FUNCTION DOC] Capture a module that uses an enclosing tensor and constant int, compare
        # generated code with a snapshot (constants inlined, tensor passed as extra input) and
        # check output against a fresh module.
        from torch.export import Dim

        N = 3
        outer = torch.randn(10, 32)

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that adds an enclosing tensor, slices, stacks
            # and reshapes.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add the enclosing tensor, drop the last row, stack the
                # result three times and reshape to a three dimensional tensor.
                z = x + outer
                y = z[:-1, :]  # [s0 - 1, 32]
                stacked = torch.stack([y] * N, dim=0)  # [N * (s0 - 1), 32]
                reshaped = stacked.reshape(-1, N, 32)  # [(s0 - 1), N, 32]
                return reshaped

        inps = (torch.randn(10, 32),)
        ep = dynamo_graph_capture_for_export(MyModel())(*inps)
        self.assertExpectedInline(
            ep.code.strip("\r\n "),
            """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_x_ , L_outer_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_x_ = L_x_
    l_outer_ = L_outer_
    z = l_x_ + l_outer_;  l_x_ = l_outer_ = None
    y = z[(slice(None, -1, None), slice(None, None, None))];  z = None
    stacked = torch.stack([y, y, y], dim = 0);  y = None
    reshaped = stacked.reshape(-1, 3, 32);  stacked = None
    return self._dynamo_bytecode_unflatten((reshaped,), _fn_args)""",
        )
        self.assertEqual(ep(*inps), MyModel()(*inps))

    def test_dynamo_graph_capture_full_tracing_context(self) -> None:
        # [FUNCTION DOC] Capture with an input whose dim 0 is marked dynamic; the captured module
        # matches eager on a different batch size and its metadata tracing context has a fake
        # mode and exactly one tensor context entry.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the first dimension size to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input plus its first dimension size.
                return x + x.shape[0]

        foo = Foo()

        def make_inputs(b: int):
            # [FUNCTION DOC] HELPER: Create a random tensor with the given batch size and three
            # columns, mark dim 0 dynamic, and return it in a tuple.
            ret = (torch.randn(b, 3),)
            torch._dynamo.mark_dynamic(ret[0], 0)
            return ret

        trace_inputs = make_inputs(2)
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs(3)
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        self.assertIsNotNone(gm.meta["tracing_context"].fake_mode)
        self.assertEqual(len(gm.meta["tracing_context"].tensor_to_context), 1)

    def test_dynamo_graph_capture_ctx_return(self):
        # [FUNCTION DOC] Register GlobalContext as a pytree node, capture a module that returns
        # such an object created in a with block, and compare flattened leaves of captured and
        # eager outputs. Deregisters the node in a finally block.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that returns a value and a context object holding
            # a tensor computed from the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Inside a context, compute the input plus one, store
                # input plus two in the context tensors under key six, and return both.
                with GlobalContext() as ctx:
                    z = x + 1
                    ctx._tensors["6"] = x + 2
                return z, ctx

        def make_inputs():
            # [FUNCTION DOC] HELPER: Build a one element tuple with a random tensor.
            return (torch.randn(2, 3),)

        try:
            pytree.register_pytree_node(
                GlobalContext,
                lambda x: x.__flatten__(),
                GlobalContext.__unflatten__,
            )
            mod = Module()

            gm = dynamo_graph_capture_for_export(mod)(*make_inputs())
            test_inputs = make_inputs()
            actual_outputs = pytree.tree_leaves(gm(*test_inputs))
            expected_outputs = pytree.tree_leaves(mod(*test_inputs))
            self.assertEqual(actual_outputs, expected_outputs)
        finally:
            pytree._deregister_pytree_node(GlobalContext)

    def test_dynamo_graph_capture_dict_keys_getitem(self):
        # [FUNCTION DOC] Capture a module whose forward pre-hook assigns into an attribute of a
        # local object; the code snapshot shows the hook's dead add and the module's
        # multiplication, and the captured module matches eager.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module doubling its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input times two.
                return x * 2

        foo = Module()

        class BlockMask:
            # [CLASS DOC] HELPER: Local stand-in class named like BlockMask holding a
            # data attribute.
            def __init__(self, d):
                # [FUNCTION DOC] HELPER: Store the given value.
                self.d = d

        block_mask = BlockMask(torch.randn(4))

        def pre_hook_function(m, input):
            # [FUNCTION DOC] HELPER: Forward pre-hook: set the data attribute of the local object
            # to the first input plus one and return the inputs unchanged.
            block_mask.d = input[0] + 1
            return input  # Return a tuple of modified inputs

        foo.register_forward_pre_hook(pre_hook_function)

        def make_inputs():
            # [FUNCTION DOC] HELPER: Build a one element tuple with a random tensor of
            # length four.
            return (torch.randn(4),)

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertExpectedInline(
            gm.code.strip("\r\n "),
            """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_args_0_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_args_0_ = L_args_0_
    add = l_args_0_ + 1;  add = None
    mul = l_args_0_ * 2;  l_args_0_ = None
    return self._dynamo_bytecode_unflatten((mul,), _fn_args)""",
        )
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))

    def test_dynamo_graph_capture_with_tensor_constant(self):
        # [FUNCTION DOC] Capture a module adding an enclosing tensor constant; the captured
        # module must match eager and have the same numbers of buffers and parameters.
        outer = torch.randn(2, 3)

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding an enclosing tensor to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input plus the enclosing tensor.
                z = x + outer
                return z

        foo = MyModel()

        def make_inputs():
            # [FUNCTION DOC] HELPER: Build a one element tuple with a random tensor of two
            # by three.
            return (torch.randn(2, 3),)

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        self.assertEqual(len(list(gm.buffers())), len(list(foo.buffers())))
        self.assertEqual(len(list(gm.parameters())), len(list(foo.parameters())))

    def test_dynamo_graph_capture_side_effects(self):
        # [FUNCTION DOC] Clear the module-level list, capture a plain function that appends to
        # it, with replay disabled. Checks exactly one warning about side effects while
        # compiling, that tracing and calling the captured module leave the list empty, and that
        # eager execution appends one entry.
        GLOBAL_LIST.clear()

        def foo(x):
            # [FUNCTION DOC] HELPER: Add one to the input, append the result to the module-level
            # list and return it.
            z = x + 1
            GLOBAL_LIST.append(z)
            return z

        def make_inputs():
            # [FUNCTION DOC] HELPER: Build a one element tuple with a random tensor of two
            # by three.
            return (torch.randn(2, 3),)

        trace_inputs = make_inputs()
        with (
            torch._dynamo.config.patch(replay_side_effects=False),
            warnings.catch_warnings(record=True) as w,
        ):
            gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
            cnt = 0
            for entry in w:
                if "While compiling, we found certain side effects happened" in str(
                    entry.message
                ):
                    cnt += 1
            self.assertEqual(cnt, 1)
        self.assertEqual(len(GLOBAL_LIST), 0)
        test_inputs = make_inputs()
        gm_results = gm(*test_inputs)
        self.assertEqual(len(GLOBAL_LIST), 0)
        self.assertEqual(gm_results, foo(*test_inputs))
        self.assertEqual(len(GLOBAL_LIST), 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_dynamo_graph_capture_fx_graph_annotate_overlap_pass(self):
        # [FUNCTION DOC] CUDA only. Capture a model composed of custom autograd function
        # applications, two of them inside fx annotations, with node metadata preservation
        # enabled, and check the captured module matches the model on fresh inputs. The
        # triple-quoted block in the body is an unused string literal documenting the expected
        # graph; nothing asserts on it.
        class DummyOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd function adding a scalar in forward and passing
            # the gradient unchanged in backward.
            @staticmethod
            def forward(ctx, x, scalar):
                # [FUNCTION DOC] HELPER: Save the input for backward and return it plus
                # the scalar.
                ctx.save_for_backward(x)
                return x + scalar

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Return the incoming gradient for the tensor and None for
                # the scalar.
                return grad_out, None

        def mock_fw_compute(x):
            # [FUNCTION DOC] HELPER: Apply the custom function with ten under a
            # compute annotation.
            with fx_traceback.annotate({"compute": 0}):
                return DummyOp.apply(x, 10)

        def mock_bw_comm(x):
            # [FUNCTION DOC] HELPER: Apply the custom function with twenty under a
            # communication annotation.
            with fx_traceback.annotate({"comm": 0}):
                return DummyOp.apply(x, 20)

        def mock_bw_compute(x):
            # [FUNCTION DOC] HELPER: Apply the custom function with thirty and no annotation.
            return DummyOp.apply(x, 30)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module chaining the helper functions on a forward input
            # and a backward input.
            def forward(self, fw_in, bw_in):
                # [FUNCTION DOC] HELPER: Apply forward compute to the first input, communication
                # then compute to the second, and return both results.
                fw_out = mock_fw_compute(fw_in)
                # bw_in blocks bw_out
                bw_in = mock_bw_comm(bw_in)
                bw_out = mock_bw_compute(bw_in)
                return fw_out, bw_out

        def input_fn():
            # [FUNCTION DOC] HELPER: Build a tuple of a random CUDA tensor that requires grad
            # followed by a random CUDA tensor.
            inputs = (torch.rand(2, 128, device="cuda", requires_grad=True),)
            grad_ins = (torch.rand(2, 128, device="cuda"),)
            return (
                *inputs,
                *grad_ins,
            )

        with torch.device("meta"):
            model = Model()

        import torch.fx.traceback as fx_traceback

        with fx_traceback.preserve_node_meta():
            gm = dynamo_graph_capture_for_export(model)(*input_fn())

        """
        def forward(self, args_0, args_1):
            _tree_leaf_0, _tree_leaf_1, _tree_leaf_2, = pytree.tree_leaves((self, args_0, args_1,))
            L_fw_in_ , L_bw_in_ , = self._in_shuffle_graph(_tree_leaf_0, _tree_leaf_1, _tree_leaf_2)
            l_fw_in_ = L_fw_in_
            l_bw_in_ = L_bw_in_
            fwd_body_0 = self.fwd_body_0
            bwd_body_0 = self.bwd_body_0
            fw_out = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_fw_in_, args_tensor_mask = [True, False], non_differentiable_idx = []);  fwd_body_0 = bwd_body_0 = l_fw_in_ = None
            bw_in = l_bw_in_ + 20;  l_bw_in_ = None
            bw_out = bw_in + 30;  bw_in = None
            return pytree.tree_unflatten(self._out_shuffle_graph(_tree_leaf_0, _tree_leaf_1, _tree_leaf_2, fw_out, bw_out), self._out_spec)
        """
        test_inputs = input_fn()
        self.assertEqual(gm(*test_inputs), model(*test_inputs))

    def test_dynamo_graph_capture_default_args(self):
        # [FUNCTION DOC] Capture a module whose forward has a default second argument and call
        # the captured module with the default; output must equal eager.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding a default valued second argument.
            def forward(self, x, y=1):
                # [FUNCTION DOC] HELPER: Return x plus y.
                return x + y

        m = Module()
        ep = dynamo_graph_capture_for_export(m)(torch.randn(2, 3))
        test_inputs = (torch.randn(2, 3),)
        self.assertEqual(ep(*test_inputs), m(*test_inputs))

    def test_restore_state_dict_basic(self):
        # [FUNCTION DOC] Capture a module with a linear layer and a buffer, restore the original
        # state dict names onto the captured module, and check parameter and buffer names equal
        # the original names and outputs equal eager.
        """Test basic state dict restoration with a simple model."""
        from torch.export import _restore_state_dict

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a linear layer and a buffer added to
            # its output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer and register a buffer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.register_buffer("buf", torch.randn(4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the linear output plus the buffer.
                return self.linear(x) + self.buf

        model = Model()
        gm = dynamo_graph_capture_for_export(model)(torch.randn(2, 4))

        # Before restoration, FQNs may be flattened
        state_dict_before = dict(gm.named_parameters())
        buffer_dict_before = dict(gm.named_buffers())

        _restore_state_dict(model, gm)

        # After restoration, FQNs should match original module
        state_dict_after = dict(gm.named_parameters())
        buffer_dict_after = dict(gm.named_buffers())

        # Check that the parameter names match the original
        original_param_names = set(dict(model.named_parameters()).keys())
        restored_param_names = set(state_dict_after.keys())
        self.assertEqual(original_param_names, restored_param_names)

        # Check that the buffer names match the original
        original_buffer_names = set(dict(model.named_buffers()).keys())
        restored_buffer_names = set(buffer_dict_after.keys())
        self.assertEqual(original_buffer_names, restored_buffer_names)

        # Verify the model still works correctly
        test_input = torch.randn(2, 4)
        expected = model(test_input)
        actual = gm(test_input)
        self.assertEqual(expected, actual)

    def test_restore_state_dict_nested_modules(self):
        # [FUNCTION DOC] Restore state dict names after capturing a model made of two submodules
        # each with a linear layer and buffer; parameter and buffer names must match the original
        # and outputs match eager.
        """Test state dict restoration with nested modules."""
        from torch.export import _restore_state_dict

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Local submodule with a linear layer and a scale buffer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer and register the scale buffer.
                super().__init__()
                self.fc = torch.nn.Linear(8, 8)
                self.register_buffer("scale", torch.tensor(2.0))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the linear output times the scale.
                return self.fc(x) * self.scale

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module summing the outputs of two submodules.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create two submodules.
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the sum of both submodule outputs.
                return self.sub1(x) + self.sub2(x)

        model = Model()
        gm = dynamo_graph_capture_for_export(model)(torch.randn(2, 8))

        _restore_state_dict(model, gm)

        # Check that nested FQNs are properly restored
        original_param_names = set(dict(model.named_parameters()).keys())
        restored_param_names = set(dict(gm.named_parameters()).keys())
        self.assertEqual(original_param_names, restored_param_names)

        # Check buffers too
        original_buffer_names = set(dict(model.named_buffers()).keys())
        restored_buffer_names = set(dict(gm.named_buffers()).keys())
        self.assertEqual(original_buffer_names, restored_buffer_names)

        # Verify functionality
        test_input = torch.randn(2, 8)
        expected = model(test_input)
        actual = gm(test_input)
        self.assertEqual(expected, actual)

    def test_restore_state_dict_with_bound_method(self):
        # [FUNCTION DOC] Capture the bound forward method of a model, restore state dict names
        # passing the bound method, and check parameter names match the original.
        """Test state dict restoration when passing a bound method."""
        from torch.export import _restore_state_dict

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with one linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the linear layer applied to the input.
                return self.linear(x)

        model = Model()
        gm = dynamo_graph_capture_for_export(model.forward)(torch.randn(2, 4))

        # Pass bound method instead of module
        _restore_state_dict(model.forward, gm)

        # Verify FQNs match
        original_param_names = set(dict(model.named_parameters()).keys())
        restored_param_names = set(dict(gm.named_parameters()).keys())
        self.assertEqual(original_param_names, restored_param_names)

    def test_restore_state_dict_type_error(self):
        # [FUNCTION DOC] Restoring state dict names with an argument that is neither a module nor
        # a bound method must raise TypeError.
        """Test that _restore_state_dict raises TypeError for invalid input."""
        from torch.export import _restore_state_dict

        gm = torch.fx.GraphModule({}, torch.fx.Graph())

        # Should raise TypeError when given a non-module, non-bound-method
        with self.assertRaises(TypeError):
            _restore_state_dict(lambda x: x, gm)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_aot_export_blockmask_with_new_closure(self):
        # [FUNCTION DOC] CUDA only. Capture a model returning a block mask built from a closure,
        # then export through the AOT joint export under a tracing context, compile state context
        # and fake mode; compare the graph code with a snapshot. Then compile the joint result
        # and check it is not None.
        import contextlib

        from torch._export.utils import _compiling_state_context
        from torch._functorch.aot_autograd import (
            aot_compile_joint_with_descriptors,
            aot_export_joint_with_descriptors,
        )
        from torch._guards import tracing as torch_tracing, TracingContext
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn():
            # [FUNCTION DOC] HELPER: Factory returning a mask function capturing an integer.
            res = 4

            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function true where query index is at least key
                # index plus the captured integer.
                return q >= k + res

            return fn

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module building a block mask from the factory
            # mask function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Build the mask function, create a 64 by 64 block mask,
                # return the input with it.
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        args = (torch.randn(2, 128, device="cuda"),)
        gm = dynamo_graph_capture_for_export(Model())(*args)

        fake_mode = gm.meta["fake_mode"]
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch_tracing(TracingContext(fake_mode)))
            stack.enter_context(_compiling_state_context())
            stack.enter_context(fake_mode)

            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack,
                gm,
                args=args,
                kwargs={},
                keep_inference_input_mutations=True,
            )
            self.assertExpectedInline(
                str(joint_with_descriptors.graph_module.code).strip(),
                """\
def forward(self, arg0_1):
    arange_2 = torch.ops.aten.arange.start(0, 64, device = device(type='cuda', index=0), pin_memory = False)
    arange_3 = torch.ops.aten.arange.start(0, 64, device = device(type='cuda', index=0), pin_memory = False)
    add = torch.ops.aten.add.Tensor(arange_3, 4);  arange_3 = None
    view = torch.ops.aten.view.default(arange_2, [64, 1]);  arange_2 = None
    ge = torch.ops.aten.ge.Tensor(view, add);  view = add = None
    unsqueeze = torch.ops.aten.unsqueeze.default(ge, 0);  ge = None
    expand = torch.ops.aten.expand.default(unsqueeze, [1, 64, 64]);  unsqueeze = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(expand, 0);  expand = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [1, 1, 64, 64]);  unsqueeze_1 = None
    constant_pad_nd = torch.ops.aten.constant_pad_nd.default(expand_1, [0, 64, 0, 64], 0.0);  expand_1 = None
    view_1 = torch.ops.aten.view.default(constant_pad_nd, [1, 1, 1, 128, 1, 128]);  constant_pad_nd = None
    permute = torch.ops.aten.permute.default(view_1, [0, 1, 2, 4, 3, 5]);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(permute, [-2, -1]);  permute = None
    eq = torch.ops.aten.eq.Scalar(sum_1, 16384)
    gt = torch.ops.aten.gt.Scalar(sum_1, 0)
    lt = torch.ops.aten.lt.Scalar(sum_1, 16384);  sum_1 = None
    bitwise_and = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None
    _to_copy = torch.ops.aten._to_copy.default(bitwise_and, dtype = torch.int8);  bitwise_and = None
    _to_copy_1 = torch.ops.aten._to_copy.default(eq, dtype = torch.int8);  eq = None
    _to_copy_2 = torch.ops.aten._to_copy.default(_to_copy, dtype = torch.int32);  _to_copy = None
    sum_2 = torch.ops.aten.sum.dim_IntList(_to_copy_2, [-1])
    sort = torch.ops.aten.sort.stable(_to_copy_2, stable = True, descending = True);  _to_copy_2 = None
    getitem_1 = sort[1];  sort = None
    _to_copy_3 = torch.ops.aten._to_copy.default(sum_2, dtype = torch.int32, memory_format = torch.contiguous_format);  sum_2 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(getitem_1, dtype = torch.int32, memory_format = torch.contiguous_format);  getitem_1 = None
    _to_copy_5 = torch.ops.aten._to_copy.default(_to_copy_1, dtype = torch.int32);  _to_copy_1 = None
    sum_3 = torch.ops.aten.sum.dim_IntList(_to_copy_5, [-1])
    sort_1 = torch.ops.aten.sort.stable(_to_copy_5, stable = True, descending = True);  _to_copy_5 = None
    getitem_3 = sort_1[1];  sort_1 = None
    _to_copy_6 = torch.ops.aten._to_copy.default(sum_3, dtype = torch.int32, memory_format = torch.contiguous_format);  sum_3 = None
    _to_copy_7 = torch.ops.aten._to_copy.default(getitem_3, dtype = torch.int32, memory_format = torch.contiguous_format);  getitem_3 = None
    new_zeros = torch.ops.aten.new_zeros.default(_to_copy_4, [1, 1, 1, 2], dtype = torch.int32, pin_memory = False)
    arange_4 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(arange_4, -1);  arange_4 = None
    arange_5 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(_to_copy_3, 3)
    lt_1 = torch.ops.aten.lt.Tensor(arange_5, unsqueeze_3);  arange_5 = unsqueeze_3 = None
    scalar_tensor = torch.ops.aten.scalar_tensor.default(1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0))
    where = torch.ops.aten.where.self(lt_1, _to_copy_4, scalar_tensor);  lt_1 = scalar_tensor = None
    new_ones = torch.ops.aten.new_ones.default(new_zeros, [1, 1], pin_memory = False)
    arange_6 = torch.ops.aten.arange.default(1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(arange_6, -1);  arange_6 = None
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    view_2 = torch.ops.aten.view.default(new_ones, [1, 1, 1, 1]);  new_ones = None
    arange_7 = torch.ops.aten.arange.default(1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    unsqueeze_6 = torch.ops.aten.unsqueeze.default(arange_7, -1);  arange_7 = None
    unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
    index_put = torch.ops.aten.index_put.default(new_zeros, [unsqueeze_8, unsqueeze_5, unsqueeze_2, where], view_2);  new_zeros = unsqueeze_8 = unsqueeze_5 = unsqueeze_2 = where = view_2 = None
    slice_3 = torch.ops.aten.slice.Tensor(index_put, 3, 0, 1);  index_put = None
    transpose_1 = torch.ops.aten.transpose.int(slice_3, -2, -1);  slice_3 = None
    sum_4 = torch.ops.aten.sum.dim_IntList(transpose_1, [-1])
    sort_2 = torch.ops.aten.sort.stable(transpose_1, stable = True, descending = True);  transpose_1 = None
    getitem_5 = sort_2[1];  sort_2 = None
    _to_copy_8 = torch.ops.aten._to_copy.default(sum_4, dtype = torch.int32, memory_format = torch.contiguous_format);  sum_4 = None
    _to_copy_9 = torch.ops.aten._to_copy.default(getitem_5, dtype = torch.int32, memory_format = torch.contiguous_format);  getitem_5 = None
    new_zeros_1 = torch.ops.aten.new_zeros.default(_to_copy_7, [1, 1, 1, 2], dtype = torch.int32, pin_memory = False)
    arange_8 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
    unsqueeze_9 = torch.ops.aten.unsqueeze.default(arange_8, -1);  arange_8 = None
    arange_9 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
    unsqueeze_10 = torch.ops.aten.unsqueeze.default(_to_copy_6, 3)
    lt_2 = torch.ops.aten.lt.Tensor(arange_9, unsqueeze_10);  arange_9 = unsqueeze_10 = None
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1 = torch.ops.aten.where.self(lt_2, _to_copy_7, scalar_tensor_1);  lt_2 = scalar_tensor_1 = None
    new_ones_1 = torch.ops.aten.new_ones.default(new_zeros_1, [1, 1], pin_memory = False)
    arange_10 = torch.ops.aten.arange.default(1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    unsqueeze_11 = torch.ops.aten.unsqueeze.default(arange_10, -1);  arange_10 = None
    unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, -1);  unsqueeze_11 = None
    view_3 = torch.ops.aten.view.default(new_ones_1, [1, 1, 1, 1]);  new_ones_1 = None
    arange_11 = torch.ops.aten.arange.default(1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    unsqueeze_13 = torch.ops.aten.unsqueeze.default(arange_11, -1);  arange_11 = None
    unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, -1);  unsqueeze_13 = None
    unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    index_put_1 = torch.ops.aten.index_put.default(new_zeros_1, [unsqueeze_15, unsqueeze_12, unsqueeze_9, where_1], view_3);  new_zeros_1 = unsqueeze_15 = unsqueeze_12 = unsqueeze_9 = where_1 = view_3 = None
    slice_6 = torch.ops.aten.slice.Tensor(index_put_1, 3, 0, 1);  index_put_1 = None
    transpose_3 = torch.ops.aten.transpose.int(slice_6, -2, -1);  slice_6 = None
    sum_5 = torch.ops.aten.sum.dim_IntList(transpose_3, [-1])
    sort_3 = torch.ops.aten.sort.stable(transpose_3, stable = True, descending = True);  transpose_3 = None
    getitem_7 = sort_3[1];  sort_3 = None
    _to_copy_10 = torch.ops.aten._to_copy.default(sum_5, dtype = torch.int32, memory_format = torch.contiguous_format);  sum_5 = None
    _to_copy_11 = torch.ops.aten._to_copy.default(getitem_7, dtype = torch.int32, memory_format = torch.contiguous_format);  getitem_7 = None
    return (arg0_1, _to_copy_3, _to_copy_4, _to_copy_6, _to_copy_7, _to_copy_8, _to_copy_9, _to_copy_10, _to_copy_11)""",
            )

            compiled_fn = aot_compile_joint_with_descriptors(joint_with_descriptors)
            # Shouldn't crash
            self.assertIsNotNone(compiled_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_aot_export_blockmask_closure_spec_mismatch(self):
        # [FUNCTION DOC] CUDA only. Flattening block masks: same closure code with same or
        # different captured values gives equal tree specs, while a mask function with different
        # code gives a different spec.
        """BlockMasks with same closure structure produce equal TreeSpecs.

        Closure values are extracted into pytree leaves, so two BlockMasks
        whose mask_mod closures have the same code + structure but different
        captured values have the same spec (values differ in the leaves, not
        the context).  BlockMasks with different closure *structure* (e.g.
        different code) must still produce different specs.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn(offset):
            # [FUNCTION DOC] HELPER: Factory returning a mask function that captures the
            # given offset.
            def fn(b, h, q, k):
                # [FUNCTION DOC] HELPER: Mask function true where query index is at least key
                # index plus the offset.
                return q >= k + offset

            return fn

        mask_a = create_block_mask(
            make_mask_fn(4), B=1, H=1, Q_LEN=64, KV_LEN=64, device="cuda"
        )
        mask_b = create_block_mask(
            make_mask_fn(8), B=1, H=1, Q_LEN=64, KV_LEN=64, device="cuda"
        )
        mask_a_same = create_block_mask(
            make_mask_fn(4), B=1, H=1, Q_LEN=64, KV_LEN=64, device="cuda"
        )

        _, spec_a = pytree.tree_flatten(mask_a)
        _, spec_b = pytree.tree_flatten(mask_b)
        _, spec_a_same = pytree.tree_flatten(mask_a_same)

        # Same closure code + same captured value -> same spec
        self.assertEqual(spec_a, spec_a_same)
        # Same closure code + different captured value -> same spec
        # (values are in the leaves, not the context)
        self.assertEqual(spec_a, spec_b)

        # Different closure *code* -> different spec
        def different_mask(b, h, q, k):
            # [FUNCTION DOC] HELPER: Different mask function: true where query index is strictly
            # greater than key index.
            return q > k

        mask_c = create_block_mask(
            different_mask, B=1, H=1, Q_LEN=64, KV_LEN=64, device="cuda"
        )
        _, spec_c = pytree.tree_flatten(mask_c)
        self.assertNotEqual(spec_a, spec_c)

    def _assert_blockmask_partial_replays_bound_tensors(self, make_mask_mod):
        # [FUNCTION DOC] HELPER: Shared check for partial mask_mod extraction. Builds block masks
        # whose mask_mod is a partial with bound tensors from the given factory, traces the
        # mask_mod with make_fx on the flattened leaves of the first mask, replays the traced
        # function with the leaves of a second mask, and asserts the replay equals the second
        # mask's eager result and differs from the first's.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.nn.attention.flex_attention import BlockMask, create_block_mask

        query_indices = torch.arange(8, dtype=torch.int32)[:, None]
        key_indices = torch.arange(8, dtype=torch.int32)[None, :]
        document_ids = torch.zeros(8, dtype=torch.int64)
        batch0_attn_regions = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
        batch1_attn_regions = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int32)

        def mask_rule(batch, head, query_idx, key_idx, attn_regions, document_ids):
            # [FUNCTION DOC] HELPER: Mask rule requiring causal order, equal attention regions
            # and equal document ids of query and key positions.
            return (
                (query_idx >= key_idx)
                & (attn_regions[query_idx] == attn_regions[key_idx])
                & (document_ids[query_idx] == document_ids[key_idx])
            )

        def build_block_mask(attn_regions):
            # [FUNCTION DOC] HELPER: Create a small 8 by 8 CPU block mask whose mask_mod binds
            # the given attention regions.
            return create_block_mask(
                make_mask_mod(mask_rule, attn_regions, document_ids),
                B=1,
                H=1,
                Q_LEN=8,
                KV_LEN=8,
                device="cpu",
                BLOCK_SIZE=4,
            )

        def trace_mask_mod(attn_regions):
            # [FUNCTION DOC] HELPER: Build a block mask, flatten it, and return a make_fx trace
            # of a function that rebuilds the block mask from the leaves and evaluates mask_mod
            # on the index grid.
            block_mask = build_block_mask(attn_regions)
            flat_leaves, spec = block_mask._flatten()
            return make_fx(
                lambda *flat_leaves: BlockMask._unflatten(flat_leaves, spec).mask_mod(
                    0, 0, query_indices, key_indices
                )
            )(*flat_leaves)

        traced_mask_mod = trace_mask_mod(batch0_attn_regions)
        replayed_batch1 = traced_mask_mod(
            *build_block_mask(batch1_attn_regions)._flatten()[0]
        )
        expected_batch0 = build_block_mask(batch0_attn_regions).mask_mod(
            0, 0, query_indices, key_indices
        )
        expected_batch1 = build_block_mask(batch1_attn_regions).mask_mod(
            0, 0, query_indices, key_indices
        )

        self.assertFalse(torch.equal(replayed_batch1, expected_batch0))
        self.assertTrue(torch.equal(replayed_batch1, expected_batch1))

    def test_blockmask_partial_extraction_replays_bound_tensors(self):
        # [FUNCTION DOC] Run the partial replay check with a mask_mod that is a single functools
        # partial binding two keyword tensors.
        self._assert_blockmask_partial_replays_bound_tensors(
            lambda mask_rule, attn_regions, document_ids: functools.partial(
                mask_rule,
                attn_regions=attn_regions,
                document_ids=document_ids,
            )
        )

    def test_blockmask_recursive_partial_extraction_replays_bound_tensors(self):
        # [FUNCTION DOC] Run the partial replay check with a mask_mod that is a nested
        # functools partial.
        self._assert_blockmask_partial_replays_bound_tensors(
            lambda mask_rule, attn_regions, document_ids: functools.partial(
                functools.partial(mask_rule, attn_regions=attn_regions),
                document_ids=document_ids,
            )
        )

    def test_mask_mod_wrapper_eq_for_partials(self):
        # [FUNCTION DOC] Mask mod wrappers built from extracted partials with the same function
        # but different bound tensors compare equal and have equal hashes, for both plain and
        # nested partials. Calling the wrapper raises RuntimeError saying the stripped callable
        # is not callable.
        from torch.nn.attention.flex_attention import (
            _extract_callable_pytree,
            _MaskModWrapper,
        )

        def mask_rule(batch, head, query_idx, key_idx, attn_regions, document_ids):
            # [FUNCTION DOC] HELPER: Mask rule requiring causal order, equal attention regions
            # and equal document ids.
            return (
                (query_idx >= key_idx)
                & (attn_regions[query_idx] == attn_regions[key_idx])
                & (document_ids[query_idx] == document_ids[key_idx])
            )

        document_ids = torch.zeros(8, dtype=torch.int64)
        batch0_attn_regions = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
        batch1_attn_regions = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int32)

        def wrap(mask_mod):
            # [FUNCTION DOC] HELPER: Extract the pytree leaves from a callable and wrap the
            # stripped callable and spec in a mask mod wrapper.
            _, spec, stripped = _extract_callable_pytree(mask_mod)
            return _MaskModWrapper(stripped, spec)

        partial_a = functools.partial(
            mask_rule,
            attn_regions=batch0_attn_regions,
            document_ids=document_ids,
        )
        partial_b = functools.partial(
            mask_rule,
            attn_regions=batch1_attn_regions,
            document_ids=document_ids,
        )
        recursive_partial_a = functools.partial(
            functools.partial(mask_rule, attn_regions=batch0_attn_regions),
            document_ids=document_ids,
        )
        recursive_partial_b = functools.partial(
            functools.partial(mask_rule, attn_regions=batch1_attn_regions),
            document_ids=document_ids,
        )

        wrapped_partial_a = wrap(partial_a)
        wrapped_partial_b = wrap(partial_b)
        wrapped_recursive_partial_a = wrap(recursive_partial_a)
        wrapped_recursive_partial_b = wrap(recursive_partial_b)

        self.assertEqual(wrapped_partial_a, wrapped_partial_b)
        self.assertEqual(hash(wrapped_partial_a), hash(wrapped_partial_b))
        self.assertEqual(wrapped_recursive_partial_a, wrapped_recursive_partial_b)
        self.assertEqual(
            hash(wrapped_recursive_partial_a), hash(wrapped_recursive_partial_b)
        )
        with self.assertRaisesRegex(RuntimeError, "stripped callable is not callable"):
            wrapped_partial_a(
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
            )

    def test_mask_mod_wrapper_eq_for_plain_functions(self):
        # [FUNCTION DOC] Two separately created but structurally identical plain mask functions
        # give wrappers that are equal and hash equally.
        from torch.nn.attention.flex_attention import (
            _extract_callable_pytree,
            _MaskModWrapper,
        )

        def make_mask_mod():
            # [FUNCTION DOC] HELPER: Factory returning a new causal mask function.
            def mask_mod(batch, head, query_idx, key_idx):
                # [FUNCTION DOC] HELPER: Causal mask ignoring batch and head.
                del batch, head
                return query_idx >= key_idx

            return mask_mod

        def wrap(mask_mod):
            # [FUNCTION DOC] HELPER: Extract the pytree leaves from a callable and wrap the
            # stripped callable and spec in a mask mod wrapper.
            _, spec, stripped = _extract_callable_pytree(mask_mod)
            return _MaskModWrapper(stripped, spec)

        wrapped_a = wrap(make_mask_mod())
        wrapped_b = wrap(make_mask_mod())

        self.assertEqual(wrapped_a, wrapped_b)
        self.assertEqual(hash(wrapped_a), hash(wrapped_b))

    def test_blockmask_self_referential_function_closure_extraction(self):
        # [FUNCTION DOC] Block masks made from mask functions that refer to a self-referential
        # helper function must have equal tree specs, and a block mask rebuilt from flattened
        # leaves must give the same mask_mod result as the original.
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_mod():
            # [FUNCTION DOC] HELPER: Factory returning a causal mask function that references a
            # self-returning helper.
            def helper():
                # [FUNCTION DOC] HELPER: Helper that returns itself.
                return helper

            def mask_mod(batch, head, query_idx, key_idx):
                # [FUNCTION DOC] HELPER: Causal mask that references the helper but ignores it.
                del batch, head
                _ = helper
                return query_idx >= key_idx

            return mask_mod

        mask_a = create_block_mask(
            make_mask_mod(), B=1, H=1, Q_LEN=8, KV_LEN=8, device="cpu", BLOCK_SIZE=4
        )
        mask_b = create_block_mask(
            make_mask_mod(), B=1, H=1, Q_LEN=8, KV_LEN=8, device="cpu", BLOCK_SIZE=4
        )

        leaves_a, spec_a = pytree.tree_flatten(mask_a)
        leaves_b, spec_b = pytree.tree_flatten(mask_b)

        self.assertEqual(spec_a, spec_b)

        restored = pytree.tree_unflatten(leaves_a, spec_a)
        self.assertTrue(
            torch.equal(
                restored.mask_mod(
                    0, 0, torch.arange(8)[:, None], torch.arange(8)[None, :]
                ),
                mask_a.mask_mod(
                    0, 0, torch.arange(8)[:, None], torch.arange(8)[None, :]
                ),
            )
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_blockmask_and_masks_closure_extraction(self):
        # [FUNCTION DOC] CUDA only. Combine two mask functions with and_masks where one captures
        # a CUDA tensor; flattening the block mask yields leaves of the regular tensor attributes
        # plus the captured tensor, all tensors, passes the dynamo input check, and unflattening
        # gives a callable mask_mod.
        """and_masks closure tensors are recursively extracted into pytree leaves.

        and_masks(fn1, fn2) returns a closure capturing a tuple of functions.
        _extract_callable_pytree must recursively process these functions
        (extracting their closure tensors) rather than emitting the functions
        themselves as leaves, since functions are not supported export input
        types.
        """
        from torch.nn.attention.flex_attention import (
            and_masks,
            BlockMask,
            create_block_mask,
        )

        _register_blockmask_pytree()

        def causal(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: Causal mask: true where query index is at least key index.
            return q_idx >= kv_idx

        offset = torch.tensor(3, device="cuda")

        def offset_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: Mask comparing the query index against key index plus the
            # captured tensor.
            return q_idx >= kv_idx + offset

        mask_mod = and_masks(causal, offset_mask)
        block_mask = create_block_mask(
            mask_mod, B=1, H=None, Q_LEN=128, KV_LEN=128, device="cuda"
        )

        leaves, spec = pytree.tree_flatten(block_mask)

        # Present BlockMask tensor attrs + offset extracted from
        # offset_mask's closure
        n_regular = sum(
            getattr(block_mask, attr) is not None for attr in BlockMask._TENSOR_ATTRS
        )
        self.assertEqual(len(leaves), n_regular + 1)
        self.assertTrue(all(isinstance(l, torch.Tensor) for l in leaves))
        self.assertIs(leaves[n_regular], offset)

        # Leaves must all pass check_user_input_output
        from torch._dynamo.eval_frame import check_user_input_output
        from torch._dynamo.exc import UserErrorType

        check_user_input_output(leaves, UserErrorType.INVALID_INPUT)

        # Round-trip: unflatten should reconstruct a working mask_mod
        restored = pytree.tree_unflatten(leaves, spec)
        self.assertTrue(callable(restored.mask_mod))

    def test_aot_export_closure_buffer_mutation(self):
        # [FUNCTION DOC] Buffer mutation through a closure: captures a wrapper module whose
        # function mutates a buffer through closure, verifies the dynamo capture mutates the
        # buffer, exports joint graphs through both the public and private capture functions and
        # checks their code is identical, then runs both joint graphs and compares outputs and
        # the mutated buffer with eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a zero buffer that accumulates the input sum
            # and returns the sine.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Register the buffer.
                super().__init__()
                self.register_buffer("buf", torch.zeros(()))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add the sum of the input to the buffer in place and
                # return the sine of the input.
                self.buf.add_(x.sum())
                return x.sin()

        def make_closure(mod):
            # [FUNCTION DOC] HELPER: Return a function that mutates the buffer of the given
            # module in place with the input sum and returns the sine.
            def fn(x):
                # [FUNCTION DOC] HELPER: Add the input sum to the module buffer in place and
                # return the sine of the input.
                mod._buffers["buf"].add_(x.sum())
                return x.sin()

            return fn

        class Wrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module sharing the parameter, buffer and submodule
            # dictionaries of another module and calling a stored function.
            def __init__(self, fn, mod):
                # [FUNCTION DOC] HELPER: Alias the dictionaries of the given module and store
                # the function.
                super().__init__()
                self._parameters = mod._parameters
                self._buffers = mod._buffers
                self._modules = mod._modules
                self._fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Call the stored function with the input.
                return self._fn(x)

        def run_export(capture_fn):
            # [FUNCTION DOC] HELPER: Create a module and wrapper, capture with the given capture
            # function, then run joint export with descriptors inside tracing, compile state and
            # fake mode contexts with input mutation kept and functionalization disabled. Returns
            # the joint graph module, the wrapper and the input.
            mod = Mod()
            wrapped = Wrapper(make_closure(mod), mod)
            x = torch.randn(4)
            gm = capture_fn(wrapped)(x)

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    torch_tracing(
                        gm.meta.get(
                            "tracing_context", TracingContext(gm.meta["fake_mode"])
                        )
                    )
                )
                stack.enter_context(_compiling_state_context())
                stack.enter_context(gm.meta["fake_mode"])

                jd = aot_export_joint_with_descriptors(
                    stack,
                    gm,
                    args=(x,),
                    kwargs={},
                    keep_inference_input_mutations=True,
                    disable_functionalization=True,
                )
            return jd.graph_module, wrapped, x

        # Verify Dynamo-captured graph mutates the buffer via closure
        mod = Mod()
        wrapped = Wrapper(make_closure(mod), mod)
        x = torch.randn(4)
        gm = dynamo_graph_capture_for_export(wrapped)(x)
        wrapped.buf.zero_()
        gm(x)
        self.assertEqual(wrapped.buf, x.sum())

        # Verify joint graphs from both APIs match
        joint_public, _, _ = run_export(dynamo_graph_capture_for_export)
        joint_private, _, _ = run_export(_dynamo_graph_capture_for_export)
        self.assertEqual(
            str(joint_public.code).strip(), str(joint_private.code).strip()
        )

        # Verify numerical correctness of both joint graphs against eager
        mod = Mod()
        x = torch.randn(4)
        eager_out = mod(x)
        eager_buf = mod.buf.clone()

        for label, joint_gm in [("public", joint_public), ("private", joint_private)]:
            buf_input = torch.zeros(())
            (exported_out,) = joint_gm(buf_input, x)
            self.assertEqual(
                exported_out,
                eager_out,
                msg=lambda msg: f"{msg}\n{label}: output mismatch",
            )
            self.assertEqual(
                buf_input,
                eager_buf,
                msg=lambda msg: f"{msg}\n{label}: buffer mutation mismatch",
            )


if __name__ == "__main__":
    run_tests()
