# Owner(s): ["module: functorch"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Tests for functorch vmap (torch.vmap): the public API behavior and error
# messages, the batching-rule/fallback machinery, per-operator exhaustive OpInfo-based vmap
# tests, vmap over autograd functions, random ops, chunk_vmap, and sdpa backends. Importing
# builds the platform-specific list of SDPA backends and defines device-generic test classes
# that are instantiated at the bottom of the file (run via run_tests).
import contextlib
import functools
import itertools
import os
import random
import types
import unittest
import warnings
from collections import namedtuple, OrderedDict

from common_utils import (
    check_vmap_fallback,
    compute_quantities_for_vmap_test,
    decorate,
    DisableVmapFallback,
    generate_vmap_inputs,
    get_fallback_and_vmap_exhaustive,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    opsToleranceOverride,
    skip,
    skipIf,
    tol1,
    xfail,
    xfailIf,
)
from functorch_additional_op_db import additional_op_db

import functorch
import torch
import torch.nn.functional as F
from functorch import grad, grad_and_value, jacfwd, jvp, vjp, vmap
from functorch.experimental import chunk_vmap
from torch import Tensor
from torch._C._functorch import reshape_dim_into, reshape_dim_outof
from torch._functorch.make_functional import functional_init_with_buffers
from torch._functorch.vmap import restore_vmap
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.autograd_function_db import autograd_function_db
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    TEST_CUDNN_VERSION,
    tf32_on_and_off,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    OpDTypes,
    ops,
    skipOps,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_MPS,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    unMarkDynamoStrictTest,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.utils import _pytree as pytree


def get_platform_specific_sdpa():
    # [FUNCTION DOC] Returns the list of scaled-dot-product-attention backends usable on this
    # platform: always the math backend, plus flash, memory-efficient and cuDNN attention when
    # the corresponding PLATFORM_SUPPORTS flags imported from common_cuda are true.
    ret = [SDPBackend.MATH]
    if PLATFORM_SUPPORTS_FLASH_ATTENTION:
        ret.append(SDPBackend.FLASH_ATTENTION)
    if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
        ret.append(SDPBackend.CUDNN_ATTENTION)
    return ret


PLATFORM_SPECIFIC_SDPA = get_platform_specific_sdpa()

FALLBACK_REGEX = "There is a performance drop"


class EnableVmapFallbackWarnings:
    # [CLASS DOC] Context manager that turns on the debug-only vmap fallback warnings in the C++
    # dispatcher while active and restores the previous enabled/disabled state on exit. State:
    # prev_state, set on entry. Used by tests that check the fallback warning is emitted.
    def __enter__(self):
        # [FUNCTION DOC] Saves the current fallback-warning enabled flag into prev_state, then
        # enables the debug-only display of vmap fallback warnings.
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        # [FUNCTION DOC] Restores the fallback-warning flag saved on entry. Ignores exception
        # arguments, so exceptions raised in the body are not suppressed.
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)


@markDynamoStrictTest
class TestVmapAPI(TestCase):
    # [CLASS DOC] Tests of the public vmap API surface: output/input validation and error
    # messages, in_dims/out_dims handling (negative, nested, pytree, None), nesting of vmaps,
    # and the fallback-warning behavior. Marked as a dynamo strict test class.
    def test_non_tensor_output_raises(self):
        # [FUNCTION DOC] Checks vmap raises a ValueError when the mapped function returns a
        # Python float, or a tuple containing a plain int, instead of tensors.
        with self.assertRaisesRegex(ValueError, "got type <class 'float'>"):
            vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            # [FUNCTION DOC] HELPER: Returns a tuple of the input tensor and the int 3, used to
            # trigger the non-tensor-output error.
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'>"):
            vmap(multiple_outputs)(torch.ones(3))

    def test_different_map_dim_size_raises(self):
        # [FUNCTION DOC] Checks that vmap raises a ValueError when mapped inputs differ in size
        # along the mapped dimension, for plain tensors, tuple inputs and dict inputs.
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = (
            "Expected all tensors to have the same size in the mapped dimension"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
                {"x": x, "y": y}
            )

    def test_func_with_no_inputs(self):
        # [FUNCTION DOC] Checks vmap raises a ValueError mentioning "got no inputs" when
        # the mapped function is called without any arguments, for functions with zero and
        # one parameters.
        expected_msg = "got no inputs"

        def foo():
            # [FUNCTION DOC] HELPER: Zero-argument function returning a fresh random tensor.
            return torch.randn(3)

        def bar(x):
            # [FUNCTION DOC] HELPER: One-argument function that ignores its input and returns a
            # random tensor.
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    def test_func_with_no_tensors(self):
        # [FUNCTION DOC] Checks vmap raises a ValueError requiring at least one Tensor when the
        # only input is a non-tensor and its in_dim is None.
        def foo(x):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns a random tensor.
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, "at least one Tensor"):
            vmap(foo, (None,))(1)

    def test_constant_function(self):
        # [FUNCTION DOC] Checks a function that returns a constant scalar tensor under vmap
        # yields a tensor of that constant repeated once per batch element.
        output = vmap(lambda x: torch.tensor(3.14))(torch.ones(3))
        self.assertEqual(output, torch.tensor([3.14, 3.14, 3.14]))

    def test_single_input(self):
        # [FUNCTION DOC] Checks vmap of an elementwise square equals squaring the whole input.
        x = torch.randn(2, 3)

        def square(x):
            # [FUNCTION DOC] HELPER: Elementwise square of its argument.
            return x * x

        output = vmap(square)(x)
        self.assertEqual(output, x * x)

    def test_multiple_inputs(self):
        # [FUNCTION DOC] Checks vmap of torch.mul over two same-shaped inputs equals the
        # direct product.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul)(x, y)
        self.assertEqual(output, x * y)

    def test_multiple_outputs(self):
        # [FUNCTION DOC] Checks vmap of a function returning two tensors returns a pair matching
        # the direct computation of the square and the cube.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the square and the cube of its argument as a tuple.
            return x * x, x * x * x

        x = torch.randn(3)
        outputs = vmap(foo)(x)
        self.assertEqual(outputs[0], x * x)
        self.assertEqual(outputs[1], x * x * x)

    def test_multiple_outputs2(self):
        # [FUNCTION DOC] Checks vmap accepts functions returning a tuple of two tensors, a list
        # of two tensors and a list of one tensor (nothing should raise).
        # This is the same thing as
        # def returns_tuple_of_tensors(x):
        #     return x, x
        def returns_tuple_of_tensors(x):
            # [FUNCTION DOC] HELPER: Returns the input twice in a tuple.
            return (x, x)

        def returns_list_of_two_tensors(x):
            # [FUNCTION DOC] HELPER: Returns the input twice in a list.
            return [x, x]

        def returns_list_of_one_tensor(x):
            # [FUNCTION DOC] HELPER: Returns the input wrapped in a one-element list.
            return [x]

        x = torch.randn(3)

        # should not throw
        vmap(returns_tuple_of_tensors)(x)
        vmap(returns_list_of_two_tensors)(x)
        vmap(returns_list_of_one_tensor)(x)

    def test_nested_with_same_map_dim(self):
        # [FUNCTION DOC] Checks that two and three levels of nested vmap over the same leading
        # dims give the same result as direct elementwise multiplication.
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)

        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)

    def test_nested_with_diag_embed(self):
        # [FUNCTION DOC] Checks nested vmap of diag_embed matches the direct call; diag_embed is
        # registered with conditional functionalization so it needs separate coverage.
        # diag_embed requires special testing because it is registered with conditional functionalization.
        x = torch.randn(3, 3, 5)
        output = vmap(vmap(torch.diag_embed))(x)
        self.assertEqual(output, torch.diag_embed(x))

    def test_nested_with_different_map_dim(self):
        # [FUNCTION DOC] Checks nested vmaps mapping over differently sized dims (two and three
        # levels) produce the outer-product style broadcast results with the expected shapes.
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        z = torch.randn(7, 3)
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

    def test_noop_in_inner_vmap(self):
        # [FUNCTION DOC] Checks an inner vmap whose function ignores the inner mapped input and
        # returns the outer one yields the outer values expanded along the inner batch size.
        x = torch.randn(3)
        y = torch.randn(5)
        output = vmap(lambda x: vmap(lambda y: x)(y))(x)
        self.assertEqual(output, x.view(3, 1).expand(3, 5))

    def test_checkpoint(self):
        # [FUNCTION DOC] Checks the gradient of a vmapped batched QR-based loss is identical
        # with and without non-reentrant activation checkpointing.
        A = torch.randn((3, 8, 8), dtype=torch.float64, requires_grad=True)

        def get_grad(checkpoint):
            # [FUNCTION DOC] HELPER: Clears the grad of A, computes the summed Q factor loss
            # through vmap of linalg.qr (optionally under checkpoint), runs backward and
            # returns A.grad.
            A.grad = None

            def get_loss(A):
                # [FUNCTION DOC] HELPER: Loss function: vmapped QR of A, summing the
                # orthogonal factors.
                ortho_A, _ = torch.func.vmap(torch.linalg.qr)(A)
                return torch.sum(ortho_A)

            if checkpoint:
                loss = torch.utils.checkpoint.checkpoint(
                    get_loss, A, use_reentrant=False
                )
            else:
                loss = get_loss(A)
            loss.backward()
            return A.grad

        expected = get_grad(checkpoint=False)
        result = get_grad(checkpoint=True)
        self.assertEqual(result, expected)

    def test_unsupported_op_err_msg(self):
        # [FUNCTION DOC] Checks the error for unsupported ops under vmap: an out= op raises a
        # RuntimeError that the fallback does not support out=/view ops, and torch.equal
        # (non-tensor return) raises a RuntimeError about missing batching rule.
        # Unsupported view op
        tensor = torch.randn(2, 3)
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        # TODO: find a view op
        # with self.assertRaisesRegex(RuntimeError, msg):
        #     vmap(torch.ravel)(tensor)

        def out_op(x, y):
            # [FUNCTION DOC] HELPER: Calls torch.abs with its second argument as the out tensor.
            return torch.abs(x, out=y)

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        # Don't support non-tensor returns. This is a limitation of vmap;
        # functions that don't return tensors must be special cased
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(torch.equal)(tensor, tensor)

    def test_nonzero_out_dims(self):
        # [FUNCTION DOC] Checks out_dims: results are permuted views sharing the input storage
        # for positive and negative out_dims, apply to all outputs, work with 64-dim tensors,
        # and with a non-identity function.
        # Basic test
        tensor = torch.randn(2, 3)
        result = vmap(lambda x: x, out_dims=1)(tensor)
        self.assertEqual(result, tensor.permute(1, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # Test that the batch dimension gets permuted to dim 2
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 0, 3))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # negative out_dim
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=-1)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 3, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # check that out_dims works on ALL outputs
        tensor = torch.randn(2, 3, 5, 7)
        other = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x, y: (x, y), out_dims=2)(tensor, other)
        self.assertEqual(
            result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3))
        )

        # use out_dims with the maximum vmap-able tensor dims (64 dims)
        ndims = 64
        shape = [2] + [1] * (ndims - 1)
        expected_shape = [1, 1, 2] + [1] * (ndims - 3)
        tensor = torch.randn(shape)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result.shape, expected_shape)

        # test something that is not the identity function
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns x, x times y, and x times y times y as a tuple.
            return x, x * y, x * y * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=1)(x, y)
        self.assertEqual(
            result,
            (
                x.permute(1, 0, 2),
                (x * y).permute(1, 0, 2),
                (x * y * y).permute(1, 0, 2),
            ),
        )

    def test_multiple_out_dims(self):
        # [FUNCTION DOC] Checks tuple out_dims assigns a separate output dimension to each
        # returned tensor, including negative values, for functions with two and four outputs.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input twice in a tuple.
            return x, x

        def bar(x, y):
            # [FUNCTION DOC] HELPER: Returns x three times followed by the product of x and y.
            return x, x, x, x * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=(0, 1))(x)
        self.assertEqual(result, (x, x.permute(1, 0, 2)))

        result = vmap(bar, out_dims=(-1, 0, 1, 2))(x, y)
        expected = (
            x.permute(1, 2, 0),
            x,
            x.permute(1, 0, 2),
            (x * y).permute(1, 2, 0),
        )
        self.assertEqual(result, expected)

    def test_nested_out_dims(self):
        # [FUNCTION DOC] Checks out_dims on nested vmaps (inner only, all levels, negative
        # values, and a non-identity function) give the expected output shapes and permutations.
        y = torch.randn(2, 3, 5, 7)

        # Inner vmap has non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y))(y)
        self.assertEqual(result.shape, (2, 5, 3, 7))
        self.assertEqual(result, y.permute(0, 2, 1, 3))

        # all vmaps have non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y), out_dims=1)(y)
        self.assertEqual(result.shape, (5, 2, 3, 7))
        self.assertEqual(result, y.permute(2, 0, 1, 3))

        # throwing in some negative out_dims
        result = vmap(lambda y: vmap(lambda x: x, out_dims=-1)(y), out_dims=-1)(y)
        self.assertEqual(result.shape, (5, 7, 3, 2))
        self.assertEqual(result, y.permute(2, 3, 1, 0))

        # testing fn that isn't the identity
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        result = vmap(lambda y: vmap(lambda x: x * y, out_dims=1)(x), out_dims=-1)(y)
        self.assertEqual(result.shape, (3, 2, 5))
        self.assertEqual(result, (y.view(5, 1, 3) * x).permute(2, 1, 0))

    def test_out_dims_edge_case(self):
        # [FUNCTION DOC] Checks a one-element tuple of out_dims is accepted for a function with
        # a single output and equals the integer form.
        def foo(x):
            # [FUNCTION DOC] HELPER: Identity function.
            return x

        # Test that we accept out_dims=(1,) for a function with one output.
        tensor = torch.randn(2, 3)
        expected = vmap(foo, out_dims=1)(tensor)
        result = vmap(foo, out_dims=(1,))(tensor)
        self.assertEqual(result, expected)

    def test_out_dims_none_tuple(self):
        # [FUNCTION DOC] Checks a None entry in an out_dims tuple passes through non-tensor
        # outputs unchanged, including when every output is None-mapped (with an in-place
        # side effect).
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input tensor and a string.
            return x, "hello world"

        tensor = torch.randn(2, 3)
        result = vmap(foo, out_dims=(0, None))(tensor)
        self.assertEqual(result[1], "hello world")
        self.assertEqual(result[0], tensor)

        def foo(x):
            # [FUNCTION DOC] HELPER: Adds one in place to its input and returns None plus
            # a string.
            x.add_(1)
            return None, "hello world"

        result = vmap(foo, out_dims=(None, None))(tensor)
        self.assertEqual(result, (None, "hello world"))

    def test_out_dims_none(self):
        # [FUNCTION DOC] Checks out_dims None raises a ValueError when a batched tensor is
        # returned, but allows returning a non-tensor value.
        def foo(x):
            # [FUNCTION DOC] HELPER: Identity function.
            return x

        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(
            ValueError, "can not return a BatchedTensor when out_dim is None"
        ):
            vmap(foo, out_dims=None)(tensor)

        def foo(x):
            # [FUNCTION DOC] HELPER: Adds one in place to its input and returns a string.
            x.add_(1)
            return "hello world"

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, "hello world")

    def test_out_dims_normal_tensor(self):
        # [FUNCTION DOC] Checks a function returning an unbatched tensor is expanded along the
        # batch dim by default, and returned as-is when out_dims is None.
        def foo(x):
            # [FUNCTION DOC] HELPER: Ignores its input and returns a three element
            # arange tensor.
            return torch.arange(3)

        tensor = torch.randn(2, 3)
        result = vmap(foo)(tensor)
        self.assertEqual(result.shape, [2, 3])

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, torch.arange(3))

    def test_pytree_returns(self):
        # [FUNCTION DOC] Checks vmap preserves nested tuple/list return structure and values.
        x = torch.randn(2, 3)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine in a nested structure of tuple and list.
            y = x.sin()
            return y, (y, y), [y, (y, y)]

        y0, (y1, y2), (y3, (y4, y5)) = vmap(f)(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y0, y1)
        self.assertEqual(y2, y1)
        self.assertEqual(y2, y3)
        self.assertEqual(y4, y3)
        self.assertEqual(y5, y4)

    def test_pytree_odict_returns(self):
        # [FUNCTION DOC] Checks vmap preserves OrderedDict return type and its values.
        x = torch.randn(2, 3)

        def f(t):
            # [FUNCTION DOC] HELPER: Returns an OrderedDict with the sine and cosine of
            # the input.
            y = t.sin()
            return OrderedDict([("sin", y), ("cos", t.cos())])

        out = vmap(f)(x)
        if not isinstance(out, OrderedDict):
            raise AssertionError(f"Expected OrderedDict, got {type(out)}")
        expected = f(x)
        self.assertEqual(out["sin"], expected["sin"])
        self.assertEqual(out["cos"], expected["cos"])

    def test_pytree_returns_outdims(self):
        # [FUNCTION DOC] Checks a nested out_dims pytree is applied per leaf of a nested
        # tuple output.
        x = torch.randn(2, 3)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine and a pair of copies of it.
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=(0, (0, 1)))(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, x.sin())
        self.assertEqual(y2, x.sin().t())

    def test_pytree_returns_broadcast_simple(self):
        # [FUNCTION DOC] Checks a single integer out_dims is broadcast to every leaf of a
        # nested output.
        x = torch.randn(2, 3)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine and a pair of copies of it.
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=1)(x)
        self.assertEqual(y0, x.sin().t())
        self.assertEqual(y1, y0)
        self.assertEqual(y2, y0)

    def test_pytree_returns_broadcast_nested(self):
        # [FUNCTION DOC] Checks a partial out_dims tuple is broadcast over the nested
        # output subtree.
        x = torch.randn(2, 3)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine and a pair of copies of it.
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=(0, 1))(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, y0.t())
        self.assertEqual(y2, y0.t())

    def test_out_dims_must_be_int_or_collection_of_int_err_msg(self):
        # [FUNCTION DOC] Checks out_dims given as a string, or a tuple containing a string,
        # raises a ValueError saying it must be an int, None or a collection of ints.
        msg = "must be an int, None or a python collection of ints"
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims="lol")(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=("lol",))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        # [FUNCTION DOC] Checks out_dims tuples of the wrong length (too many or too few for the
        # number of outputs) raise a ValueError saying the structure is not compatible.
        msg = "not compatible"
        x = torch.randn(2, 3, 5)

        # Too many out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0, 0, 0))(x)

        # Too few out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x), out_dims=(0,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0))(x)

    def test_out_dim_out_of_bounds_err_msg(self):
        # [FUNCTION DOC] Checks out_dims values outside the output rank (3 and -4 for a 3-d
        # result) raise an IndexError about the dimension being out of range.
        # TODO(rzou): This error message isn't that great. It comes straight
        # from maybe_wrap_dim. Consider doing a try-catch-(add some context) to
        # the error message in the future in C++
        msg = "Dimension out of range"
        x = torch.randn(2, 3, 5)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=3)(x)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=-4)(x)

    def test_non_zero_in_dims(self):
        # [FUNCTION DOC] Checks non-zero in_dims: the batch dim is moved to the front without a
        # copy, including for two inputs mapped along different dims.
        tensor = torch.randn(2, 3, 5)

        # Implicit out_dims = 0; vmap will move the batch dim to the front.
        output = vmap(lambda x: x, (1,))(tensor)
        self.assertEqual(output, tensor.permute(1, 0, 2))
        self.assertEqual(output.data_ptr(), tensor.data_ptr())

        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        output = vmap(torch.mul, (0, 1))(x, y)
        self.assertEqual(output, x * y.t())
        output = vmap(torch.mul, (1, 0))(x, y)
        self.assertEqual(output, x.t() * y)

    def test_none_in_dims(self):
        # [FUNCTION DOC] Checks a None in_dim leaves the tensor unmapped (broadcast) and that
        # None works for non-tensor arguments.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # None in_dim for a Tensor means we don't map over it
        output = vmap(torch.mul, (0, None))(x, y)
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        # None in_dim for non-tensor arguments
        output = vmap(torch.mul, (0, None))(x, 2)
        self.assertEqual(output, x * 2)

    def test_nested_non_default_in_dims(self):
        # [FUNCTION DOC] Checks three levels of nested vmap each with non-default in_dims.
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.mul), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    def test_nested_negative_in_dims(self):
        # [FUNCTION DOC] Checks negative in_dims map over the last dimension of each input.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul, (-1, -1))(x, y)
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(output, (x * y).permute(1, 0))

    def test_non_default_in_dims_out_dims(self):
        # [FUNCTION DOC] Checks combinations of in_dims and out_dims (equal and different)
        # for an identity and a scaling function, including nested vmaps; also that views
        # share storage.
        x = torch.randn(2, 3, 5)

        # Same in_dim as out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x)
        self.assertEqual(result.data_ptr(), x.data_ptr())

        # Different in_dim from out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, x.transpose(1, 2))
        self.assertEqual(result.data_ptr(), x.data_ptr())

        def foo(x):
            # [FUNCTION DOC] HELPER: Doubles its input.
            return x * 2

        # Same in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x * 2)

        # Different in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, (x * 2).transpose(1, 2))

        # Basic nested test.
        result = vmap(vmap(foo, 1, 1), 1, 1)(x)
        self.assertEqual(result, x * 2)

    def test_item_throws(self):
        # [FUNCTION DOC] Checks calling item on a batched tensor inside vmap raises
        # a RuntimeError.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the Python scalar of its tensor input.
            return x.item()

        with self.assertRaisesRegex(RuntimeError, r"item\(\) on a Tensor"):
            vmap(f)(torch.randn(3))

    def test_data_dependent_control_flow_throws(self):
        # [FUNCTION DOC] Checks Python branching on a batched tensor inside vmap raises a
        # RuntimeError about data-dependent control flow.
        def f(x):
            # [FUNCTION DOC] HELPER: Branches on the truthiness of its input.
            if x:
                return x
            return 0

        with self.assertRaisesRegex(RuntimeError, r"data-dependent control flow"):
            vmap(f)(torch.randn(3))

    def test_accepts_nested_inputs(self):
        # [FUNCTION DOC] Checks vmap accepts tuple, list and dict inputs, with default,
        # per-argument and fully structured in_dims, and deeper nested containers.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Single layer of nesting
        out = vmap(lambda z: z[0] + z[1])((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z[0] + z[1])([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, y])
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z["x"] + z["y"])({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=(0,))({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
            {"x": x, "y": y}
        )
        self.assertEqual(out, x + y)

        # Multiple layers of nesting
        out_fn = vmap(lambda z: z["x"][0] + z["x"][1][0] + z["y"][0] + z["y"][1])
        out = out_fn({"x": [x, (x,)], "y": [y, y]})
        self.assertEqual(out, x + x + y + y)

    def test_in_dims_wrong_type_err_msg(self):
        # [FUNCTION DOC] Checks in_dims given as a list, set or string raises a ValueError
        # saying it must be an int or a nested tuple; a plain tuple of ints is accepted.
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"expected `in_dims` to be int or a \(potentially nested\) tuple"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0}))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, "lol")(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=[0, 0])([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_not_enough_in_dims_err_msg(self):
        # [FUNCTION DOC] Checks in_dims whose structure does not match the inputs (too short,
        # too long, wrong nesting) raises a ValueError; the matching form is accepted.
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"in_dims is not compatible with the structure of `inputs`"

        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0,))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0, 0, 0))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0],))([x, y])
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        # [FUNCTION DOC] Checks an integer in_dim assigned to a non-tensor input raises a
        # ValueError; a None in_dim for the non-tensor is accepted.
        def foo(xy):
            # [FUNCTION DOC] HELPER: Unused; multiplies the two entries of a pair.
            # KNOWN ISSUE: never called.
            return xy[0] * xy[1]

        def bar(x, yz):
            # [FUNCTION DOC] HELPER: Unused; multiplies a tensor by the two entries of a pair.
            # KNOWN ISSUE: never called.
            return x * yz[0] * yz[1]

        x = torch.randn(2, 3)

        # the following are errors in jax (and will always be errors)
        msg = "Got in_dim=0 for an input but the input is of type"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, 1])
        # The following should not throw
        vmap(torch.sum, (0, None))(x, 0)

    def test_in_dim_not_in_tensor_err_msg(self):
        # [FUNCTION DOC] Checks an in_dim outside the rank of the tensor (scalar, -3 or 2 for
        # 2-d, or 3 in a list) raises a ValueError; valid dims 0 and 1 do not raise.
        def foo(x):
            # [FUNCTION DOC] HELPER: Elementwise square of its argument.
            return x * x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        msg = r"Got in_dim=-?\w for some input, but that input is a Tensor of dimensionality \w"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-3,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # the following should not throw
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        # [FUNCTION DOC] Checks that vmap over the test fallback op emits only the
        # experimental-API warning (one warning) when fallback warnings are not enabled.
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            torch.vmap(op)(x, y)
            # The single warning here is the "vmap is experimental"
            # warning, not a warning from the vmap fallback path.
            self.assertEqual(len(wa), 1)

    @skipIfTorchDynamo("Flaky test")
    @unittest.expectedFailure
    def test_fallback_warns_when_warnings_are_enabled(self):
        # [FUNCTION DOC] Checks that with fallback warnings enabled the fallback op emits an
        # extra warning matching FALLBACK_REGEX. Marked expectedFailure and skipped under dynamo
        # as flaky.
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                torch.vmap(op)(x, y)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        # [FUNCTION DOC] HELPER: Currently a no-op; returns immediately. The commented-out
        # remainder would have asserted the fallback warning was emitted. Arguments are unused.
        return
        # with warnings.catch_warnings(record=True) as wa:
        #     with EnableVmapFallbackWarnings():
        #         result = vmap(*vmap_args)(*inputs)
        #     self.assertEqual(len(wa), 2)
        #     self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        # [FUNCTION DOC] Checks the vmap fallback path rejects a zero-sized batch dimension:
        # with a batch of size zero (mapped on either argument or both, also with an extra dim)
        # vmap over the fallback test op raises a RuntimeError saying the fallback does not
        # support size-0 dims.
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        B0, B1 = 0, 3
        x = torch.randn(B0, 11)
        y = torch.randn(11)

        msg = "The fallback path does not support vmap over dims of size 0"

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

        x = torch.randn(B0, B1, 11)
        y = torch.randn(B1, 11)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

    def test_fallback_warning(self):
        # [FUNCTION DOC] Checks results of vmap over the fallback test op (non-default in_dims,
        # nested vmaps, and a 100x10x10 batch) equal calling the op directly on permuted inputs.
        # We use a dummy function _test_functorch_fallback
        # defined in prim_native_functions.cpp for this
        op = torch._test_functorch_fallback

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # nested vmap
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # big batch size (total 10000)
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    # TODO: No clue what is wrong here.
    @unittest.skip
    def test_fallback_masked_fill(self):
        # [FUNCTION DOC] Skipped test (unconditional skip decorator, reason unknown). It runs a
        # batched index_add with batch sizes 5 and 1237 and compares to a direct index_add on
        # shifted dim.
        # NB: One day we will implement a batching rule for masked_fill
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        def run_test(batch_size):
            # [FUNCTION DOC] HELPER: For the given batch size, builds a batched tensor, index
            # and values, checks vmap of index_add (map dims 0, none, none, 0) equals the direct
            # call with dim+1.
            B0 = batch_size
            x = torch.randn(B0, 7, 11, 13)
            dim = 0
            index = torch.tensor([0, 4, 2])
            values = torch.randn(B0, 3, 13)

            self._assert_uses_vmap_fallback(
                (torch.index_add, (0, None, None, 0)), (x, dim, index, values)
            )

            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 1, 13))
            self.assertEqual(result, expected)

        run_test(batch_size=5)
        run_test(batch_size=1237)

    def test_fallback_multiple_returns(self):
        # [FUNCTION DOC] Checks the fallback for an op with several returns (var_mean): single,
        # nested and triple-nested vmaps (batch up to 1237) match calling var_mean along the
        # last dim.
        # NB: One day we will implement a batching rule for torch.var_mean
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        B0, B1, B2 = 2, 3, 1237
        tensor = torch.randn(B0, 10)

        self._assert_uses_vmap_fallback((torch.var_mean,), (tensor,))

        # fallback correctness on torch.var_mean
        result = vmap(torch.var_mean)(tensor)
        expected = torch.var_mean(tensor, dim=1)
        self.assertEqual(result, expected)

        # nested vmap
        tensor = torch.randn(B0, B1, 10)
        result = vmap(vmap(torch.var_mean))(tensor)
        expected = torch.var_mean(tensor, dim=2)
        self.assertEqual(result, expected)

        # big batch size, nested vmap
        tensor = torch.randn(B0, B1, B2, 10)
        result = vmap(vmap(vmap(torch.var_mean)))(tensor)
        expected = torch.var_mean(tensor, dim=3)
        self.assertEqual(result, expected)

    def test_inplace_fallback_unary(self):
        # [FUNCTION DOC] Checks the in-place fallback for a unary in-place method (acos_): the
        # result is the very input tensor for plain vmaps (single, nested, large batch), a
        # non-zero out_dim gives a view whose base is the input, and values equal the
        # out-of-place acos.
        # Test the in-place fallback on an in-place method that takes no
        # additional Tensor arguments. This is the simplest case of the fallback.
        # NB: One day we will implement a batching rule for acos_.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.acos_
        B0, B1, B2 = 2, 3, 10000

        x = torch.randn(B0, 5)
        self._assert_uses_vmap_fallback((op,), (x,))

        # Single vmap
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op)(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Single vmap + different out_dim produces a view(!)
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op, out_dims=(1,))(x)
        self.assertTrue(result._base is x)
        self.assertEqual(result, x_orig.t().acos())

        # Nested vmap
        x_orig = torch.randn(B0, B1, 5)
        x = x_orig.clone()
        result = vmap(vmap(op))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Nested vmap, large batch size
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        result = vmap(vmap(vmap(op)))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

    def test_inplace_fallback_nary_same_levels(self):
        # [FUNCTION DOC] Checks the in-place fallback for a binary in-place method (atan2_)
        # where both operands have the same vmap levels (single, nested, 100x10x10 batch): the
        # first input is modified to equal the out-of-place result computed on moved dims.
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        # Single vmap
        B0 = 5
        x_orig = torch.randn(7, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, 7, 11)
        vmap(op, (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim(0, 2)))

        # Nested vmap
        B0, B1 = 5, 7
        x_orig = torch.randn(B1, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, B1, 11)
        vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim([0, 1], [2, 0])))

        # big batch size (total 10000)
        B0, B1, B2 = 100, 10, 10
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        y = torch.randn(B0, B1, B2)
        vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    # ("Fallback isInplaceVmapCompatible check is broken")
    @unittest.expectedFailure
    def test_inplace_fallback_nary_different_levels(self):
        # [FUNCTION DOC] Checks the in-place fallback when operand levels differ. Cases where
        # all levels of the right operand exist on the left should work; cases with extra levels
        # on the right should raise a RuntimeError saying the in-place vmap is not possible.
        # Marked expectedFailure because the fallback in-place compatibility check is broken.
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2
        B0, B1 = 2, 3

        x = torch.rand(B0, 7)
        y = torch.rand(7)
        self._assert_uses_vmap_fallback((op, (0, None)), (x, y))

        # op(left, right): All of the levels in right are found in left
        x_orig = torch.rand(B0, 7)
        x = x_orig.clone()
        y = torch.rand(7)
        vmap(op, in_dims=(0, None))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y))

        x_orig = torch.rand(B0, B1, 7)
        x = x_orig.clone()
        y = torch.rand(B0, 7)
        vmap(vmap(op, in_dims=(0, None)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, 1, 7)))

        # op(left, right): Some of the levels in right are not found in left
        msg = r"vmap: aten::atan2_\(self, \*extra_args\) is not possible"
        x = torch.rand(7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(7, B0)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 1))(x, y)

        x = torch.rand(B0, 7)
        y = torch.rand(B0, B1, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(None, 0)))(x, y)

    def test_backward_unsupported_interaction(self):
        # [FUNCTION DOC] Intended to check that calling backward inside vmap raises a
        # RuntimeError about a functorch transform.
        # KNOWN ISSUE: it unconditionally returns via skipTest right after defining the first
        # helper, so everything after that point is unreachable dead code.
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(5)
        grad = torch.randn_like(x)
        err_msg = r"backward\(\) called inside a functorch transform"

        def backward_on_vmapped_tensor(x):
            # [FUNCTION DOC] HELPER: Calls backward on the sum of its input.
            x.sum().backward()

        # FIXME
        return self.skipTest(
            "error: element 0 of tensors does not require grad and does not have a grad_fn"
        )
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_on_vmapped_tensor)(x)

        def backward_with_vmapped_grad(x, grad):
            # [FUNCTION DOC] HELPER: Calls backward on the first argument using the second as
            # the gradient.
            x.backward(grad)

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_with_vmapped_grad)(x, grad)

        def completely_unrelated_backward(y):
            # [FUNCTION DOC] HELPER: Calls backward on an unrelated captured tensor's sum,
            # returns input.
            x.sum().backward()
            return y

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)

    @unittest.expectedFailure
    def test_grad_unsupported_interaction(self):
        # [FUNCTION DOC] Checks autograd.grad called inside vmap, where either the output or the
        # input being differentiated is batched, raises a RuntimeError. Marked expectedFailure.
        input_tensor = torch.randn(3, requires_grad=True)
        err_msg = "autograd.grad.* called inside torch.vmap"

        captured = torch.randn(3, requires_grad=True)

        def output_to_grad_is_vmapped(input_tensor):
            # [FUNCTION DOC] HELPER: Differentiates a captured-tensor product against the
            # captured tensor and returns that gradient.
            output = (captured * input_tensor).sum()
            return torch.autograd.grad([output], [captured])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)

        output = (input_tensor**2).sum()

        def input_to_grad_is_vmapped(input_tensor):
            # [FUNCTION DOC] HELPER: Returns the gradient of the captured output with respect to
            # its input.
            return torch.autograd.grad([output], [input_tensor])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(input_to_grad_is_vmapped)(input_tensor)

    def test_batched_gradient_basic(self):
        # [FUNCTION DOC] Checks vmap over a vector-Jacobian product function with identity rows
        # as grad outputs gives the Jacobian, a diagonal matrix of y.
        N = 3
        x = torch.randn(N, requires_grad=True)
        y = torch.randn(N)

        def vjp_mul(v):
            # [FUNCTION DOC] HELPER: Vector-Jacobian product of x times y with respect to x for
            # vector v.
            return torch.autograd.grad([x * y], [x], grad_outputs=[v])[0]

        batched_v = torch.eye(N)
        jacobian = vmap(vjp_mul)(batched_v)
        self.assertEqual(jacobian, torch.diagflat(y))

    def test_functools_partial(self):
        # [FUNCTION DOC] Checks vmap works on a functools.partial callable.
        x = torch.randn(3)
        y = torch.randn(2, 3)
        result = vmap(functools.partial(torch.mul, x))(y)
        self.assertEqual(result, x * y)

    def test_nn_module(self):
        # [FUNCTION DOC] Checks vmap over a Linear module applied to a batch equals applying
        # it directly.
        tensor = torch.randn(2, 3)
        model = torch.nn.Linear(3, 3, bias=False)
        result = vmap(model)(tensor)
        self.assertEqual(result, model(tensor))

    def test_fallback_with_undefined_grad(self):
        # [FUNCTION DOC] Checks vmap over a conv2d vector-Jacobian function does not error even
        # though the convolution backward returns an undefined bias gradient. (The
        # fallback-usage assertion helper is a no-op, so only absence of an error is verified.)
        B0 = 7
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        weight = torch.randn(3, 3, 1, 1)
        v = torch.randn(B0, 2, 3, 4, 5)

        def get_vjp(v):
            # [FUNCTION DOC] HELPER: Returns the gradient of conv2d output with respect to x
            # given v.
            result = torch.nn.functional.conv2d(x, weight)
            (grad_x,) = torch.autograd.grad(result, x, v)
            return grad_x

        # Runs vmap(get_vjp)(v), which should not error out.
        # The backward formula for convolution returns an undefined
        # Tensor for grad_bias because the original bias does not exist.
        #
        # In the future we'll probably add a batching rule for convolution
        # backward. When this happens, we should modify this test to use a
        # different op (and/or create and use a dummy operator) to avoid bitrot.
        self._assert_uses_vmap_fallback([get_vjp], [v])

    def test_reshape_dim_into(self):
        # [FUNCTION DOC] Checks reshape_dim_into merges a source dim into a destination dim
        # (positive and negative indices) and matches movedim followed by reshape.
        x = torch.randn(2, 3, 5, 7)

        y = reshape_dim_into(0, 0, x)
        self.assertEqual(y, x.reshape(6, 5, 7))

        y = reshape_dim_into(0, 1, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        y = reshape_dim_into(0, 2, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        y = reshape_dim_into(1, 2, x)
        self.assertEqual(y, x.movedim(1, 2).reshape(2, 5, 3 * 7))

        y = reshape_dim_into(0, -2, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        y = reshape_dim_into(0, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        y = reshape_dim_into(-4, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

    def test_reshape_dim_outof(self):
        # [FUNCTION DOC] Checks reshape_dim_outof splits a dim into a given outer size
        # (positive and negative dims, non-contiguous input, and a zero-size dim) matching
        # reshape results.
        x = torch.randn(12, 12, 12).permute(2, 1, 0)

        y = reshape_dim_outof(0, 2, x)
        self.assertEqual(y, x.reshape(2, 6, 12, 12))

        y = reshape_dim_outof(1, 4, x)
        self.assertEqual(y, x.reshape(12, 4, 3, 12))

        y = reshape_dim_outof(2, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

        # Case: `0` sized dim.
        x = torch.randn(12, 12, 0)
        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y.shape, torch.Size((12, 12, 6, 0)))

    def test_batch_rule_does_not_need_to_handle_no_batched_input(self):
        # [FUNCTION DOC] Checks nested vmap where the inner batching rule gets no batched input
        # (the dot product uses only the unbatched-at-that-level tensor) gives the expected
        # broadcast sum.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Adds x to the dot product of y with a ones vector.
            res = torch.dot(y, torch.ones(2))
            return x + res

        x = torch.randn(7, 5)
        y = torch.randn(3, 2)
        out = vmap(vmap(f, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        expected = torch.mv(y, torch.ones(2)).view(3, 1, 1) + x
        self.assertEqual(out, expected)

    def test_decomposition_under_python_dispatcher(self):
        # [FUNCTION DOC] Checks vmap of square under the Python dispatcher with the
        # fallback disabled works, showing decompositions for batching are respected by
        # the Python dispatcher.
        # This test will raise an error if the vmap fallback gets invoked.
        # Here we test that decomps registered to FuncTorchBatchedDecomposition
        # are respected by the Python Dispatcher.
        t = torch.ones(3, 3) * 5
        with DisableVmapFallback():
            with torch._dispatch.python.enable_python_dispatcher():
                o = torch.vmap(torch.square)(t)
        self.assertEqual(o, torch.square(t))

    def _test_vmap_autocast(self, device):
        # [FUNCTION DOC] HELPER: Checks vmap interacts correctly with autocast on the given
        # device: autocast inside the function, as decorator, and outside the vmap, each
        # compared to the unvmapped result. Uses bfloat16 on cpu and float16 otherwise. Raises
        # AssertionError on dtype or value mismatches.
        if torch.device(device).type == "cpu":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        a_float32 = torch.rand(4, 2, 3, device=device)
        b_float32 = torch.rand(4, 3, 2, device=device)
        c_float32 = torch.rand(4, 2, 2, device=device)
        d_float32 = torch.rand(4, 3, 2, device=device)

        # Case 1, autocast inside vmapped function
        def func1(x, y, z, w):
            # [FUNCTION DOC] HELPER: Three matmuls inside an autocast region, checking
            # intermediates have the autocast dtype, then final matmul with a float32 cast
            # result outside the region.
            with torch.autocast(dtype=amp_dtype, device_type=device):
                e_float16 = torch.matmul(x, y)
                if e_float16.dtype != amp_dtype:
                    raise AssertionError(
                        f"Expected dtype {amp_dtype}, got {e_float16.dtype}"
                    )
                f_float16 = torch.matmul(z, e_float16)
                if f_float16.dtype != amp_dtype:
                    raise AssertionError(
                        f"Expected dtype {amp_dtype}, got {f_float16.dtype}"
                    )
            return torch.matmul(w, f_float16.float())

        expected = func1(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func1)(a_float32, b_float32, c_float32, d_float32)
        if not expected.allclose(out):
            raise AssertionError("Expected func1 output to be close to vmap output")

        # Case 2, autocast decorator inside vmapped function
        @torch.autocast(dtype=amp_dtype, device_type=device)
        def func2(x, y, z, w):
            # [FUNCTION DOC] HELPER: Autocast-decorated chain of three matmuls, checking
            # intermediate dtypes.
            e_float16 = torch.matmul(x, y)
            if e_float16.dtype != amp_dtype:
                raise AssertionError(
                    f"Expected dtype {amp_dtype}, got {e_float16.dtype}"
                )
            f_float16 = torch.matmul(z, e_float16)
            if f_float16.dtype != amp_dtype:
                raise AssertionError(
                    f"Expected dtype {amp_dtype}, got {f_float16.dtype}"
                )
            return torch.matmul(w, f_float16)

        expected = func2(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func2)(a_float32, b_float32, c_float32, d_float32)
        if not expected.allclose(out):
            raise AssertionError("Expected func2 output to be close to vmap output")

        # Case 3, autocast is outside vmapped function
        def func3(x, y, z, w):
            # [FUNCTION DOC] HELPER: Chain of three matmuls without its own autocast; dtype
            # checks only hold when called under an outer autocast context.
            e_float16 = torch.matmul(x, y)
            if e_float16.dtype != amp_dtype:
                raise AssertionError(
                    f"Expected dtype {amp_dtype}, got {e_float16.dtype}"
                )
            f_float16 = torch.matmul(z, e_float16)
            if f_float16.dtype != amp_dtype:
                raise AssertionError(
                    f"Expected dtype {amp_dtype}, got {f_float16.dtype}"
                )
            return torch.matmul(w, f_float16)

        with torch.autocast(dtype=amp_dtype, device_type=device):
            expected = func3(a_float32, b_float32, c_float32, d_float32)
            out = vmap(func3)(a_float32, b_float32, c_float32, d_float32)

        if not expected.allclose(out):
            raise AssertionError("Expected func3 output to be close to vmap output")

    def test_vmap_autocast_cpu(self):
        # [FUNCTION DOC] Runs the autocast check on cpu.
        self._test_vmap_autocast("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_vmap_autocast_cuda(self):
        # [FUNCTION DOC] Runs the autocast check on cuda; skipped when cuda is unavailable.
        self._test_vmap_autocast("cuda")

    @unittest.skipIf(not TEST_MPS, "MPS is unavailable")
    def test_vmap_autocast_mps(self):
        # [FUNCTION DOC] Runs the autocast check on mps; skipped when mps is unavailable.
        self._test_vmap_autocast("mps")

    def test_restore_vmap_pytree_input_output(self):
        # [FUNCTION DOC] Checks restore_vmap with pytree inputs and a dict output: results and
        # output dims equal vmap with matching in_dims and out_dims.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a dict with the sum of the pair in x and the
            # tensor y.
            output0 = x[0] + x[1]
            output1 = y
            return {"a": output0, "b": output1}

        B = 2
        x0 = torch.randn(B, 3)
        x1 = torch.randn(B)
        y = torch.randn(4, B)

        out, out_dims = restore_vmap(f, ((0, 0), 1), B, "error")((x0, x1), y)
        expected = vmap(f, in_dims=((0, 0), 1), out_dims={"a": 0, "b": 1})((x0, x1), y)
        self.assertEqual(out, expected)
        self.assertEqual(out_dims, {"a": 0, "b": 1})

    def test_restore_vmap_no_vmapped_inputs(self):
        # [FUNCTION DOC] Checks restore_vmap with no mapped inputs (all in_dims None, tensors
        # and an int): outputs equal the plain call and out dims are all None.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Returns x, the product y times z, and z.
            return x, y * z, z

        B = 2
        # Mix of tensor and non-tensor inputs
        x = torch.randn(3)
        y = torch.randn(4)
        z = 5
        out, out_dims = restore_vmap(f, (None, None, None), B, "error")(x, y, z)
        self.assertEqual(out, f(x, y, z))
        self.assertEqual(out_dims, (None, None, None))

    def test_restore_vmap_unexpanded_outputs(self):
        # [FUNCTION DOC] Checks restore_vmap with outputs not depending on the batched input
        # (mix of tensor and None outputs): results match calling with a None placeholder; out
        # dims all None.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns three times y, the sum of y and None.
            # Mix of tensor and non-tensor outputs
            return 3 * y, y.sum(), None

        B = 2
        x = torch.randn(B, 3)
        y = torch.randn(4)
        out, out_dims = restore_vmap(f, (0, None), B, "error")(x, y)
        self.assertEqual(out, f(None, y))
        self.assertEqual(out_dims, (None, None, None))

    def test_data_attribute(self):
        # [FUNCTION DOC] Checks reading or assigning the data attribute of a tensor inside vmap
        # raises a RuntimeError mentioning data access or mutation under vmap.
        def foo(x):
            # [FUNCTION DOC] HELPER: Reads the data attribute of its input and returns
            # the input.
            y = x.data  # noqa: F841
            return x

        with self.assertRaisesRegex(
            RuntimeError, "accessing `data` under vmap transform"
        ):
            torch.func.vmap(foo)(torch.randn(3, 3))

        def foo(x):
            # [FUNCTION DOC] HELPER: Assigns a new tensor to the data attribute of its input and
            # returns it.
            x.data = torch.ones(3, 3)
            return x

        with self.assertRaisesRegex(
            RuntimeError, "mutating directly with `.data` under vmap"
        ):
            torch.func.vmap(foo)(torch.randn(3, 3))

    def test_vmap_out_dims_negative_one_independent_output(self):
        # [FUNCTION DOC] Checks out_dims of minus one for outputs dependent on the input,
        # independent (tensor captured from outside) and independent scalar; the batch dim is
        # placed last.
        t = torch.randn(2, 3)
        t_scalar = torch.randn([])

        def f_dep(x):
            # [FUNCTION DOC] HELPER: Identity function.
            return x

        def f_ind(x):
            # [FUNCTION DOC] HELPER: Ignores its input and returns the captured tensor.
            return t

        def f_ind_scalar(x):
            # [FUNCTION DOC] HELPER: Ignores its input and returns the captured scalar tensor.
            return t_scalar

        res_dep_neg1 = vmap(f_dep, in_dims=0, out_dims=-1)(t)
        self.assertEqual(res_dep_neg1.shape, torch.Size([3, 2]))

        res_ind_neg1 = vmap(f_ind, in_dims=0, out_dims=-1)(torch.zeros(1))
        self.assertEqual(res_ind_neg1.shape, torch.Size([2, 3, 1]))

        res_ind_scalar_neg1 = vmap(f_ind_scalar, in_dims=0, out_dims=-1)(torch.zeros(5))
        self.assertEqual(res_ind_scalar_neg1.shape, torch.Size([5]))


def slice_inputs(inputs, bdims, i):
    # [FUNCTION DOC] HELPER: Selects index i along the batch dim for each input, except inputs
    # whose batch dim is None which are passed unchanged. Returns a tuple.
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)


def reference_vmap(op, inputs, in_dims=0, out_dims=0, return_nt=False):
    # [FUNCTION DOC] HELPER: Reference for vmap: slices each input per batch index, applies op,
    # and stacks results along out_dims (or builds nested tensors if return_nt). An int in_dims
    # or out_dims applies to all inputs/outputs. Handles single and tuple returns. Raises
    # AssertionError if batch sizes mismatch or results are not all tensors or tuples.
    if isinstance(in_dims, int):
        in_dims = (in_dims,) * len(inputs)
    bdim_sizes = [inp.size(dim) for inp, dim in zip(inputs, in_dims) if dim is not None]
    if not all(bdim_size == bdim_sizes[0] for bdim_size in bdim_sizes):
        raise AssertionError(f"Expected all bdim sizes to be equal, got {bdim_sizes}")
    bdim_size = bdim_sizes[0]
    results = tuple(op(*slice_inputs(inputs, in_dims, i)) for i in range(bdim_size))

    if len(results) == 0:
        raise AssertionError("Expected at least one result")
    op_has_single_return = not isinstance(results[0], tuple)
    if op_has_single_return:
        if not all(isinstance(result, torch.Tensor) for result in results):
            raise AssertionError(
                "Expected all results to be torch.Tensor for single-return op"
            )
        if isinstance(out_dims, int):
            out_dims = (out_dims,) * 1
        if return_nt:
            return torch.nested.nested_tensor(list(results))
        else:
            return torch.stack(results, dim=out_dims[0])

    if not all(isinstance(result, tuple) for result in results):
        raise AssertionError("Expected all results to be tuples")
    num_returns = len(results[0])
    if not all(len(result) == num_returns for result in results):
        raise AssertionError(f"Expected all results to have {num_returns} elements")
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    if return_nt:
        return tuple(
            torch.nested.nested_tensor(list(result_shards))
            for result_shards in zip(*results)
        )
    else:
        return tuple(
            torch.stack(result_shards, out_dim)
            for result_shards, out_dim in zip(zip(*results), out_dims)
        )


class TensorFactory:
    # [CLASS DOC] HELPER: Namespace of static factory functions producing random tensors of a
    # given size, device and dtype, used as input getters in vmap operator tests.
    @staticmethod
    def rand(size, device="cpu", dtype=torch.float):
        # [FUNCTION DOC] HELPER: Uniform random tensor in zero to one.
        return torch.rand(size, device=device, dtype=dtype)

    @staticmethod
    def randn(size, device="cpu", dtype=torch.float):
        # [FUNCTION DOC] HELPER: Standard normal random tensor.
        return torch.randn(size, device=device, dtype=dtype)

    @staticmethod
    def randp1(size, device="cpu", dtype=torch.float):
        # [FUNCTION DOC] HELPER: Uniform random tensor shifted into one to two.
        return torch.rand(size, device=device, dtype=dtype) + 1


# Tests vmap(op, in_dims, out_dims)(*inputs) by comparing the output to a
# (slow) sequential map+stack fallback.
#
# check_view: Test if the first returned output is a view of the first input
# check_propagates_grad: Test if the operation propagates gradients.


def _vmap_test(
    self,
    op,
    inputs,
    in_dims=0,
    out_dims=0,
    check_view=False,
    check_propagates_grad=True,
):
    # [FUNCTION DOC] HELPER: Core vmap check: runs vmap and a sequential reference, asserting
    # equality on the test case self. If check_view, asserts each output is a view of the first
    # input's base. If check_propagates_grad, reruns with a requires-grad copy of the first
    # input and asserts the first element of the result requires grad.
    result = vmap(op, in_dims, out_dims)(*inputs)
    are_nested = [t.is_nested for t in pytree.tree_leaves(result)]
    reference_result = reference_vmap(
        op, inputs, in_dims, out_dims, return_nt=any(are_nested)
    )
    self.assertEqual(result, reference_result)
    op_has_single_return = not isinstance(result, tuple)

    if check_view:
        result_as_tuple = (result,) if op_has_single_return else result
        for output in result_as_tuple:
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            self.assertTrue(
                output._base is input0_base,
                msg="result was not a view of the first input!",
            )

    if not check_propagates_grad:
        return
    # Assuming input[0] is a floating-point tensor. Check if the vmap
    # operation propagates the requires_grad flag to the zeroth output.
    # Some vmap operators are implemented in a way that assumes that
    # they are composite with respect to autograd. If the operator ever is
    # changed to not be composite with respect to autograd, then the
    # following check should fail.
    inputs_clone = list(inputs)
    inputs_clone[0] = inputs[0].clone().requires_grad_()
    result = vmap(op, in_dims, out_dims)(*inputs_clone)
    result_as_tuple = (result,) if op_has_single_return else result
    self.assertTrue(result[0].requires_grad)


def should_allow_vmap_fallback_usage(fn):
    # [FUNCTION DOC] HELPER: Returns whether the function carries the flag allowing vmap
    # fallback use.
    return getattr(fn, "_allow_vmap_fallback_usage", False)


def allowVmapFallbackUsage(fn):
    # [FUNCTION DOC] HELPER: Decorator marking a test function as allowed to use the vmap
    # fallback; sets the flag on the function and returns the same function.
    fn._allow_vmap_fallback_usage = True
    return fn


# All tests of TestVmapBase check that the slow vmap fallback is never invoked.
# This is so that we can incrementally add batching rules for operators to
# replace the slow vmap fallback path for said operators. To skip this check,
# please use the allowVmapFallbackUsage decorator.
#
# NB: Don't add tests to TestVmapBase directly, unless you want them to run
# on every subclass of TestVmapBase. Add them to e.g. TestVmapOperators.
#
# NB: TestVmapBase is a nested class. This prevents test runners from picking
# it up and running it.


class Namespace:
    # [CLASS DOC] HELPER: Container class holding TestVmapBase so test runners do not
    # collect it.
    class TestVmapBase(TestCase):
        # [CLASS DOC] Base test case for vmap operator tests. Each test method not marked as
        # allowed is wrapped so it runs with fallback warnings enabled. Subclasses add the
        # actual tests.
        def __init__(self, method_name="runTest"):
            # [FUNCTION DOC] Constructs the test case, then replaces the named test method by a
            # wrapped version unless it is flagged as allowing vmap fallback; does nothing if no
            # such method.
            super().__init__(method_name)

            test_method = getattr(self, method_name, None)
            if test_method is None:
                return

            if not should_allow_vmap_fallback_usage(test_method):
                setattr(
                    self,
                    method_name,
                    self._wrap_method_with_vmap_fallback_check(test_method),
                )

        def _wrap_method_with_vmap_fallback_check(self, method):
            # [FUNCTION DOC] HELPER: Returns a bound wrapper that runs the method with warnings
            # recorded and vmap fallback warnings enabled.
            # KNOWN ISSUE: the check that no fallback warning was emitted is commented out, so
            # the wrapper never fails on fallback use.
            # msg = (
            #     'Expected the test to not invoke the vmap fallback path, i.e., '
            #     'all of the operators being tested in this test should have batching '
            #     'rules implemented. If you are intentionally testing something to '
            #     'do with the fallback path, use allowVmapFallbackUsage. Otherwise, '
            #     'please make sure that batching rules are implemented for the '
            #     'operator(s) being tested.'
            # )

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Runs the wrapped method inside a warning-recording
                # context with always-filter and the fallback warning enabler, discarding
                # captured warnings.
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    with EnableVmapFallbackWarnings():
                        method(*args, **kwargs)
                    # for captured_warning in wa:
                    #     self.assertNotRegex(str(captured_warning.message), FALLBACK_REGEX, msg)

            return types.MethodType(wrapper, self)

        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            # [FUNCTION DOC] Checks that a test marked as allowed to use the fallback can call
            # vmap over var_mean.
            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean
            vmap(op_using_fallback)(torch.rand(3))

        @unittest.expectedFailure
        def test_vmap_fallback_check(self):
            # [FUNCTION DOC] Expected-failure test that a wrapped function using the fallback
            # raises AssertionError.
            # KNOWN ISSUE: the wrapper no longer asserts, so the final assertRaises fails.
            @self._wrap_method_with_vmap_fallback_check
            def no_fallback(self):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean

            @self._wrap_method_with_vmap_fallback_check
            def uses_fallback(self):
                # [FUNCTION DOC] HELPER: Runs vmap over var_mean on a small tensor.
                vmap(op_using_fallback)(torch.rand(3))

            no_fallback(self)

            with self.assertRaises(AssertionError):
                uses_fallback(self)


def _make_case(op, input_getter=TensorFactory.randn):
    # [FUNCTION DOC] HELPER: Returns a pair of the op and its input getter for
    # parametrized cases.
    return (op, input_getter)


@markDynamoStrictTest
class TestVmapOperators(Namespace.TestVmapBase):
    # [CLASS DOC] Operator-level vmap tests, run through the no-fallback wrapper of the
    # base class. Covers unary, binary, view, reduction, indexing, and other ops. Marked
    # as dynamo strict.
    def _vmap_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards to the module-level vmap test function using self as
        # the case.
        return _vmap_test(self, *args, **kwargs)

    def _vmap_view_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the vmap test with the view check enabled.
        self._vmap_test(*args, **kwargs, check_view=True)

    def _test_unary(self, op, getter, device, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs an op on a single input through a single vmap and a doubly
        # nested vmap, with combinations of in_dims and out_dims. The getter builds each input
        # on the device.
        test = functools.partial(self._vmap_test, *args, **kwargs)
        B0, B1 = 7, 11

        # Single vmap, various in_dims / out_dims
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(
            vmap(op, in_dims=2),
            [getter([2, 5, B0, B1, 3], device)],
            in_dims=2,
            out_dims=2,
        )

    @parametrize(
        "case",
        [
            (torch.abs, TensorFactory.randn),
            (torch.acos, TensorFactory.rand),
            (torch.asin, TensorFactory.rand),
            (torch.atan, TensorFactory.rand),
            (torch.ceil, TensorFactory.randn),
            (torch.cos, TensorFactory.rand),
            (torch.cosh, TensorFactory.rand),
            (torch.digamma, TensorFactory.rand),
            (torch.exp, TensorFactory.randn),
            (torch.expm1, TensorFactory.randn),
            (torch.floor, TensorFactory.randn),
            (torch.frac, TensorFactory.randn),
            (torch.lgamma, TensorFactory.rand),
            (torch.log, TensorFactory.randp1),
            (torch.log10, TensorFactory.randp1),
            (torch.log1p, TensorFactory.randp1),
            (torch.log2, TensorFactory.randp1),
            (torch.neg, TensorFactory.randn),
            (torch.reciprocal, TensorFactory.randp1),
            (torch.relu, TensorFactory.randn),
            (torch.round, TensorFactory.randn),
            (torch.rsqrt, TensorFactory.randp1),
            (torch.sigmoid, TensorFactory.randn),
            (torch.sign, TensorFactory.randn),
            (torch.sin, TensorFactory.rand),
            (torch.sinh, TensorFactory.rand),
            (torch.sqrt, TensorFactory.rand),
            (torch.tan, TensorFactory.rand),
            (torch.tanh, TensorFactory.rand),
            (torch.trunc, TensorFactory.randn),
        ],
        name_fn=lambda x: x[0].__name__,
    )
    def test_unary_pointwise(self, case):
        # [FUNCTION DOC] Parametrized over a table of unary pointwise torch functions and their
        # input generators. Runs the unary vmap checks on the function, then on its in-place
        # Tensor method variant (name plus underscore) with the grad propagation check disabled.
        op, getter = case
        self._test_unary(op, getter, "cpu")

        # test in-place
        method = getattr(Tensor, f"{op.__name__ + '_'}")
        self._test_unary(method, getter, "cpu", check_propagates_grad=False)

    def test_clone(self):
        # [FUNCTION DOC] Checks vmap of clone with default, preserve and contiguous memory
        # formats, that contiguous_format clones produce contiguous per-example results (single
        # and nested vmap), and that channels_last formats raise a RuntimeError saying only
        # preserve/contiguous supported.
        # Some basic tests
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, "cpu")
        self._test_unary(
            lambda x: x.clone(memory_format=torch.preserve_format),
            TensorFactory.randn,
            "cpu",
        )
        self._test_unary(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            TensorFactory.randn,
            "cpu",
        )

        # Test that the per-examples are contiguous when using torch.contiguous_format
        def clone_contiguous(x):
            # [FUNCTION DOC] HELPER: Clones its input using the contiguous memory format.
            return x.clone(memory_format=torch.contiguous_format)

        B0, B1 = 3, 5
        x = torch.randn(2, B0, 7)
        y = vmap(clone_contiguous, in_dims=1, out_dims=1)(x)
        self.assertTrue(y.movedim(1, 0).is_contiguous())
        self.assertTrue(y[:, 0, :].is_contiguous())

        x = torch.randn(2, B0, 7, B1)
        y = vmap(vmap(clone_contiguous, in_dims=2), in_dims=1)(x)
        self.assertTrue(y.is_contiguous())
        self.assertTrue(y[0][0].is_contiguous())

        msg = r"only supported with memory_format torch.preserve_format or torch.contiguous_format"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(
                torch.randn(B0)
            )

    def test_weird_matmul_case(self):
        # [FUNCTION DOC] Regression test (functorch issue 417): nested vmap of matmul with one
        # input unmapped at the outer level must not crash; the result is not checked.
        # Check that this doesn't crash.
        # https://github.com/pytorch/functorch/issues/417
        x = torch.randn(5, 2, 2, 2)
        y = torch.randn(5, 7, 2)

        vmap(vmap(torch.matmul, in_dims=(None, 0)))(x, y)

    @parametrize(
        "case",
        (
            (torch.clamp_min_, TensorFactory.randn),
            (torch.clamp_max_, TensorFactory.randn),
        ),
        name_fn=lambda x: x[0].__name__,
    )
    def test_clamp_inplace_variant(self, case):
        # [FUNCTION DOC] Parametrized over the in-place clamp_min_/clamp_max_ ops. Runs vmap
        # checks for tensor-tensor forms (various in_dims/out_dims, broadcasting, nested) and
        # the Python number overload, with grad propagation checking disabled.
        test = self._vmap_test

        def get_number(getter):
            # [FUNCTION DOC] HELPER: Returns a Python number drawn from the getter by sampling a
            # scalar tensor.
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0], device), getter([B0], device)),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([2, B0, 3], device), getter([2, B0, 3], device)),
            in_dims=(1, 1),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 2, 3], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 2, 3], device), getter([1, 1], device)),
            in_dims=(0, None),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            in_dims=(0, 0),
            check_propagates_grad=False,
        )

        # Nested vmap: op(Tensor, Tensor)
        test(
            vmap(op),
            (getter([B0, B1, 2, 3], device), getter([B0, B1, 1, 3], device)),
            check_propagates_grad=False,
        )

        # Python number overload: op(Tensor, Number)
        number = get_number(getter)
        self._test_unary(
            lambda t: op(t, number), getter, device, check_propagates_grad=False
        )

    @parametrize(
        "case",
        [
            subtest(_make_case(torch.clamp_min), name="clamp_min"),
            subtest(_make_case(torch.clamp_max), name="clamp_max"),
        ],
    )
    def test_clamp_variant(self, case):
        # [FUNCTION DOC] Parametrized over clamp_min and clamp_max. Runs vmap checks for
        # tensor-tensor forms (various in_dims/out_dims, broadcasting, nested) and the Python
        # number overload.
        test = self._vmap_test

        def get_number(getter):
            # [FUNCTION DOC] HELPER: Returns a Python number drawn from the getter by sampling a
            # scalar tensor.
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(None, 0))

        # Nested vmap: op(Tensor, Tensor)
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

        # Python number overload: op(Tensor, Number)
        number = get_number(getter)
        self._test_unary(lambda t: op(t, number), getter, device)

    def test_copy_(self):
        # [FUNCTION DOC] Checks vmap of the in-place copy_ writes the source into the
        # destination, also when the destination is batched with an unbatched source, and
        # raises a RuntimeError about in-place when the destination is unbatched but the
        # source is batched.
        x = torch.randn(3)
        y = torch.randn(3)
        vmap(Tensor.copy_)(x, y)
        self.assertEqual(x, y)

        x = torch.randn(3)
        y = torch.randn(3, 2)
        vmap(Tensor.copy_, in_dims=(1, None))(y, x)
        self.assertEqual(y, x.expand(2, 3).t())

        x = torch.randn(3)
        y = torch.randn(2, 3)
        with self.assertRaisesRegex(RuntimeError, "inplace"):
            vmap(Tensor.copy_, in_dims=(None, 0))(x, y)

    def test_silu_backward(self):
        # [FUNCTION DOC] Checks vmap over the aten silu_backward op with both batched, and with
        # either argument unbatched (including scalar), against the sequential reference.
        test = self._vmap_test
        device = "cpu"
        getter = TensorFactory.randp1
        B0 = 7
        op = torch.ops.aten.silu_backward

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([], device), getter([B0], device)), in_dims=(None, 0))
        test(op, (getter([2, B0], device), getter([2], device)), in_dims=(1, None))

    @parametrize(
        "case",
        [
            subtest(_make_case(torch.add), name="add"),
            subtest(_make_case(lambda x, y: x + y), name="add_dunder"),
            subtest(_make_case(torch.sub), name="sub"),
            subtest(_make_case(lambda x, y: x - y), name="sub_dunder"),
            subtest(_make_case(torch.mul), name="mul"),
            subtest(_make_case(lambda x, y: x * y), name="mul_dunder"),
            subtest(
                _make_case(torch.div, input_getter=TensorFactory.randp1), name="div"
            ),
            subtest(
                _make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
                name="div_dunder",
            ),
            subtest(
                _make_case(torch.pow, input_getter=TensorFactory.randp1), name="pow"
            ),
            subtest(
                _make_case(lambda x, y: x**y, input_getter=TensorFactory.randp1),
                name="pow_dunder",
            ),
        ],
    )
    def test_arithmetic(self, case):
        # [FUNCTION DOC] Parametrized over add, sub, mul, div, pow and their operator forms.
        # Checks vmap with tensor-tensor inputs of various in_dims/out_dims and nesting, Python
        # numbers on either side, and type promotion between float and double tensors. Returns
        # early when cuda is unavailable (the cross-device checks are commented out).
        test = self._vmap_test

        def get_number(getter):
            # [FUNCTION DOC] HELPER: Returns a Python number drawn from the getter by sampling a
            # scalar tensor.
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None))

        # Nested vmap: op(Tensor, Tensor)
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

        # Python number overload: op(Tensor, Number) (and vice-versa)
        number = get_number(getter)
        self._test_unary(lambda t: op(t, number), getter, device)
        number = get_number(getter)
        self._test_unary(lambda t: op(number, t), getter, device)

        # Type promotion: op(Logical Scalar Tensor, Logical Scalar Tensor)
        test(op, (getter([B0], device), getter([B0], device, dtype=torch.double)))
        test(op, (getter([B0], device, dtype=torch.double), getter([B0], device)))
        test(op, (getter([B0], device), getter([B0], device)))

        # Type promotion: op(Tensor, Logical Scalar Tensor) (and vice-versa)
        test(op, (getter([B0, 2], device), getter([B0], device, torch.double)))
        test(op, (getter([B0], device, torch.double), getter([B0, 2], device)))

        if not torch.cuda.is_available():
            return

        # TODO(rzou): fix the following
        # # Test cross-device scalars
        # number = get_number(getter)
        # self._test_unary(lambda t: op(t, number), getter, device='cuda')
        # self._test_unary(lambda t: op(number, t), getter, device='cuda')
        # self._test_unary(lambda t: op(t, torch.tensor(number)), getter, device='cuda')

    def test_as_strided(self):
        # [FUNCTION DOC] Checks the as_strided batching rule: for many contiguous and strided
        # inputs, vmap of as_strided (broadcast, transpose, select, diagonal, strided slice)
        # matches the equivalent view op and shares its base; nested vmap too. Also checks
        # RuntimeErrors for mismatched size/stride lengths and for strides that would index
        # memory outside the per-example tensor.
        def _test(sizes, strides, offset, tensor, lambd):
            # [FUNCTION DOC] HELPER: Compares vmap of as_strided with given sizes, strides and
            # offset against vmap of a reference view function, with the batch dim at 0 and then
            # at -1. Asserts equal bases and values.
            # bdim at dim 0 test
            result = vmap(lambda t: t.as_strided(sizes, strides, offset))(tensor)
            expected = vmap(lambd)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

            # bdim at dim -1 test
            tensor = tensor.movedim(0, -1)
            result = vmap(lambda t: t.as_strided(sizes, strides, offset), -1)(tensor)
            expected = vmap(lambd, -1)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

        # single vmap test
        B0 = 5
        # Each Tensor has shape [B0, 2, 3]; the expressions below
        # are just to get tensors of different strides that have shape [B0, 2, 3]
        tensors = [
            # contiguous
            torch.randn(B0, 2, 3),
            # non-contiguous
            torch.randn(B0, 3, 2).transpose(1, 2),
            torch.randn(3, 2, B0).movedim(-1, 0).transpose(1, 2),
            # non-zero storage offset
            torch.randn(2, B0, 2, 3)[1],
            torch.randn(2, 2, B0, 3)[1].movedim(1, 0),
            # non-contiguous strides, zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 0, :, 0],
            torch.randn(2, 4, B0, 3, 7).movedim(2, 0)[:, :, 0, :, 0],
            # non-contiguous strides, non-zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 2, :, 1],
            torch.randn(2, 4, 3, 7, B0).movedim(-1, 0)[:, :, 2, :, 1],
        ]

        for x in tensors:
            S0, S1 = x.stride()[1:]
            offset = x.storage_offset()

            # Broadcast
            _test(
                [5, 5, 2, 3], [0, 0, S0, S1], offset, x, lambda x: x.expand(5, 5, 2, 3)
            )
            # transpose
            _test([3, 2], [S1, S0], offset, x, lambda x: x.transpose(0, 1))
            # select
            _test([2], [S0], offset + S1, x, lambda x: x[:, 1])
            # diagonal
            _test([2], [S0 + S1], offset, x, lambda x: x.diagonal())
            # strided slice
            _test([2], [S1 * 2], offset, x, lambda x: x[0, ::2])

        # Nested vmap test
        B1 = 7
        x = torch.randn(B1, B0, 2, 3)
        S0, S1 = x.stride()[2:]
        result = vmap(
            vmap(lambda t: t.as_strided([5, 5, 2, 3], [0, 0, S0, S1])), in_dims=1
        )(x)
        expected = vmap(vmap(lambda t: t.expand(5, 5, 2, 3)), in_dims=1)(x)
        self.assertTrue(result._base is expected._base)
        self.assertEqual(result, expected)

        # Check that mal-formatted size/strides doesn't crash
        with self.assertRaisesRegex(
            RuntimeError, "size and stride must have the same length"
        ):
            x = torch.randn(B0, 2, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([1, 1, 1], [1, 1]))(x)

        # All the Sanity check #1{a,b,c} cases check that
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # doesn't index memory that is out of bounds of xs[i]. This condition
        # is important to the correctness of the as_strided batching rule
        # (see NOTE: [When will the as_strided_batching_rule fail?])

        # Sanity check #1a: The maximum indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is less than or equal to the maximum indexable location of xs[i].
        msg = "This is not supported inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3)
            vmap(lambda x: x.as_strided([3], [1], 1))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3, 5)
            vmap(lambda x: x.as_strided([4, 4], [4, 1], 0))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, B1, 3, 5)
            vmap(vmap(lambda x: x.as_strided([4, 4], [4, 1], 0)))(x)

        # Sanity check #1b: The min indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is greater than or equal to the min indexable location of xs[i].
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3)[1]
            vmap(lambda x: x.as_strided([3], [1], B0 * 3 - 1))(x)

        # Sanity check #1c:
        # xs[i] is a zero-dim tensor, but
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is not
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 0, 3)
            vmap(lambda x: x.as_strided([3], [1]))(x)

    def test_nll_loss(self):
        # [FUNCTION DOC] Checks vmap of nll_loss with mean, sum and none reduction, with batched
        # target and with unbatched target.
        test = self._vmap_test
        op = F.nll_loss
        B = 3

        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (B, 2))
        test(op, (y, t))
        test(functools.partial(op, reduction="sum"), (y, t))
        test(functools.partial(op, reduction="none"), (y, t))

        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (2,))
        test(op, (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="sum"), (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="none"), (y, t), in_dims=(0, None))

    def test_adaptive_avg_pool2d(self):
        # [FUNCTION DOC] Checks vmap of adaptive_avg_pool2d (output size 3x3) on a 5-d input
        # with the batch dim at 0, 1 and 4.
        test = self._vmap_test
        op = functools.partial(F.adaptive_avg_pool2d, output_size=(3, 3))

        x = torch.randn(3, 5, 7, 9, 11)
        test(op, (x,))
        test(op, (x,), in_dims=(1,))
        test(op, (x,), in_dims=(4,))

    def test_bmm(self):
        # [FUNCTION DOC] Checks vmap of bmm: shape mismatches raise RuntimeError; left, right
        # and both arguments batched (plain and nested) match the reference.
        op = torch.bmm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)),
            in_dims=(None, 0),
        )

    def test_cat(self):
        # [FUNCTION DOC] Checks vmap of cat with several input shapes including empty tensors,
        # unbatched args, negative dim, and nested vmaps.
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            # [FUNCTION DOC] HELPER: Returns a function that concatenates its tensor arguments
            # along the given dim, since vmap cannot take a list of tensors directly.
            def op(*tensors):
                # [FUNCTION DOC] HELPER: Concatenates the given tensors along the captured dim.
                return torch.cat(tensors, dim=dim)

            return op

        test(get_op(0), (torch.rand(B0, 2), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(B0, 0), torch.rand(B0, 0)))
        test(get_op(0), (torch.rand(2), torch.rand(B0, 0)), in_dims=(None, 0))
        test(
            get_op(1),
            (torch.rand(2, 5), torch.rand(B0, 0), torch.rand(2, 3)),
            in_dims=(None, 0, None),
        )
        test(get_op(1), (torch.rand(B0, 2, 3), torch.rand(B0, 0)))
        test(get_op(1), (torch.rand(B0, 2, 3, 4), torch.rand(0)), in_dims=(0, None))
        test(
            get_op(0),
            (torch.rand(0), torch.rand(B0, 2), torch.rand(B0, 0)),
            in_dims=(None, 0, 0),
        )
        test(get_op(0), (torch.rand(2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(3, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(17, 2), torch.rand(17, 3, B0)), in_dims=(None, 2))
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 3)),
            in_dims=(None, 0),
        )
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 3)),
            in_dims=(None, 0),
        )

    def test_unsafe_view(self):
        # [FUNCTION DOC] Checks the unsafe_view batching rule indirectly via vmap of the
        # gradient of a batched matmul-sum function, since unsafe_view is not exposed directly.
        # Unsafe view isn't exposed, so we get at it via
        # vmap(grad(matmul))
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B = 2
        x = torch.randn(B, 2, 3, 3)
        y = torch.randn(B, 3, 3)

        def baz(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the matrix product of x and y.
            return (x @ y).sum()

        test(functorch.grad(baz), (x, y))

    def test_conj(self):
        # [FUNCTION DOC] Checks vmap of conj for real and complex dtypes (single and nested,
        # various dims), and that conj of a real tensor returns the same storage.
        op = torch.conj

        def run_test(dtype):
            # [FUNCTION DOC] HELPER: Runs the conj vmap checks for the given dtype.
            def get(shape):
                # [FUNCTION DOC] HELPER: Returns a random tensor of the given shape and
                # captured dtype.
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11
            test = self._vmap_test

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        # correctness tests
        run_test(torch.float)
        run_test(torch.cfloat)

        # check that torch.conj on a non-complex tensor returns the same tensor
        real_tensor = torch.randn(3)
        result = vmap(op)(real_tensor)
        self.assertEqual(result.data_ptr(), real_tensor.data_ptr())

    def test_contiguous(self):
        # [FUNCTION DOC] Checks vmap of contiguous: unary checks, that already contiguous
        # per-example data returns the original tensor, and that channels_last formats raise a
        # RuntimeError (NYI).
        op = Tensor.contiguous

        self._test_unary(op, TensorFactory.randn, "cpu")

        # check that contiguous returns the original tensor if the per-examples
        # are already contiguous
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        x = x.movedim(0, 2)
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        self.assertTrue(result is x)

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last_3d))(tensor)

    def test_stride(self):
        # [FUNCTION DOC] Checks the per-example strides seen inside vmap equal the strides of an
        # unbatched tensor, for a contiguous input and one with the batch moved to the front.
        B0 = 3

        x = torch.randn(B0, 2, 5, 7)

        def foo(x):
            # [FUNCTION DOC] HELPER: Asserts the stride of its input matches the contiguous
            # 3-d strides.
            if x.stride() != (7 * 5, 7, 1):
                raise AssertionError(f"Expected stride (35, 7, 1), got {x.stride()}")
            return x

        vmap(foo)(x)

        x = torch.randn(2, B0, 5, 7).movedim(1, 0)

        def bar(x):
            # [FUNCTION DOC] HELPER: Asserts the stride of its input matches the expected
            # batch-adjusted strides.
            if x.stride() != (7 * 5 * B0, 7, 1):
                raise AssertionError(
                    f"Expected stride ({7 * 5 * B0}, 7, 1), got {x.stride()}"
                )
            return x

        vmap(bar)(x)

    def test_chunk(self):
        # [FUNCTION DOC] Checks vmap of chunk using view checks, with positive and negative dim,
        # non-zero in_dims and nested vmaps.
        test = self._vmap_view_test
        op = torch.chunk
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 15, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 9, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 4, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_clamp(self):
        # [FUNCTION DOC] Runs unary checks for clamp with min only, max only, both, clamp_min
        # and clamp_max.
        clamp_cases = (
            (lambda t: t.clamp(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp(max=0.5), TensorFactory.randn),
            (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn),
            (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp_max(max=0.5), TensorFactory.randn),
        )
        for op, getter in clamp_cases:
            self._test_unary(op, getter, "cpu")

    def test_comparison_ops(self):
        # [FUNCTION DOC] Checks vmap of comparison ops (function and operator forms) with
        # tensor-tensor inputs in various in_dims/out_dims, nested vmaps, and a Python number,
        # without grad checks.
        test = functools.partial(self._vmap_test, check_propagates_grad=False)

        getter = TensorFactory.randn
        B0, B1 = 7, 11

        ops = (
            torch.eq,
            lambda x, y: x == y,
            torch.gt,
            lambda x, y: x > y,
            torch.ge,
            lambda x, y: x >= y,
            torch.le,
            lambda x, y: x <= y,
            torch.lt,
            lambda x, y: x < y,
            torch.ne,
            lambda x, y: x != y,
        )

        for op in ops:
            # Single vmap: op(Tensor, Tensor)
            test(op, (getter([B0, 3]), getter([B0, 3])))
            test(op, (getter([B0]), getter([B0, 2, 3])))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0]), getter([2, 3])), in_dims=(0, None))
            test(op, (getter([2, 3]), getter([B0, 3])), in_dims=(0, None))

            # Nested vmap: op(Tensor, Tensor)
            test(vmap(op), (getter([B0, B1, 2, 3]), getter([B0, B1, 3])))
            test(
                vmap(op, in_dims=(None, 0)),
                (getter([B0, 2, 3]), getter([B1, 3])),
                in_dims=(0, None),
            )

            # test number as inputs
            number = getter([]).item()
            self._test_unary(
                lambda t: op(t, number), getter, "cpu", check_propagates_grad=False
            )

    def test_cross_batch_size_three(self):
        # [FUNCTION DOC] Checks cross with batch size 3 and no dim argument: the dim selection
        # must not pick the batch dim, for plain and nested vmap.
        # Let's test corner case when batch_size is 3 and cross' dim argument is not specified
        # According to the cross API, dim will be assigned to the first dim with value 3
        # In this test we ensure that found dim is not batch dim.
        op = torch.cross
        test = self._vmap_test
        B0 = B1 = 3
        test(op, (torch.rand(B0, 2, 3), torch.rand(B0, 2, 3)))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B0, B1, 2, 3), torch.rand(B0, B1, 2, 3)),
            in_dims=(None, 1),
        )

    def test_diagonal(self):
        # [FUNCTION DOC] Checks vmap of diagonal with several offsets and dims (view checks),
        # including negative dims and nested vmaps.
        tensor = torch.randn(3, 5, 7, 11, 13)
        test = self._vmap_view_test
        op = torch.diagonal
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        test(
            vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3),
            (tensor,),
            in_dims=1,
            out_dims=1,
        )

    def test_dot(self):
        # [FUNCTION DOC] Checks vmap of dot: shape mismatches raise RuntimeError; left, right
        # and both arguments batched, plain and nested.
        op = torch.dot
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_expand_as(self):
        # [FUNCTION DOC] Checks vmap of expand_as (view check) with batched self/other, one
        # unbatched, different batch dim placement, and nested vmaps with a lower-rank self.
        op = torch.Tensor.expand_as
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(
            vmap(op),
            (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(0, 1),
        )
        test(vmap(op), (torch.rand(B0, B1), torch.rand(B1, 2, 3, 5)), in_dims=(0, None))
        test(vmap(vmap(op)), (torch.rand(B0, B1, B2), torch.rand(B0, B1, B2, 2, 3, 5)))

    def test_fill_and_zero_inplace(self):
        # [FUNCTION DOC] Checks vmap of in-place fill_ (scalar, scalar tensor) and zero_ for
        # single and nested vmaps, plus fill_ with a batched value tensor; filling an unbatched
        # tensor from a batched value raises a RuntimeError.
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11
        ops = (
            lambda t: t.fill_(0.1),
            lambda t: t.fill_(torch.tensor(0.2)),
            lambda t: t.zero_(),
        )

        for op in ops:
            # Single vmap, various in_dims / out_dims
            test(op, [TensorFactory.randn([B0, 3])])
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2)
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [TensorFactory.randn([B0, B1])])
            test(vmap(op), [TensorFactory.randn([B1, 2, 5, B0, 3])], in_dims=2)
            test(
                vmap(op, in_dims=2),
                [TensorFactory.randn([2, 5, B0, B1, 3])],
                in_dims=2,
                out_dims=2,
            )

        # test when value is a batched tensor for fill_ operator
        B0, B1 = 3, 5
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])

        with self.assertRaisesRegex(RuntimeError, ""):
            # Runtime Error is thrown when the tensor being written to isn't being vmapped over
            vmap(Tensor.fill_, (None, 0))(
                TensorFactory.randn([B0, B1]), TensorFactory.randn([B0])
            )

    def _test_complex_views(self, op, dtypes):
        # [FUNCTION DOC] HELPER: Runs view-checked vmap tests of an op over complex inputs for
        # every given dtype (single and doubly nested vmaps, various in_dims/out_dims).
        test = self._vmap_view_test

        def run_test(op, dtype):
            # [FUNCTION DOC] HELPER: Runs the view-checked vmap cases of the op for one dtype.
            def get(shape):
                # [FUNCTION DOC] HELPER: Returns a random tensor of the given shape and
                # captured dtype.
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([3, B0])], in_dims=1)
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, 3, B0])], in_dims=4)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        for dtype in dtypes:
            run_test(op, dtype)

    def test_real(self):
        # [FUNCTION DOC] Checks vmap of torch.real on cfloat and cdouble inputs.
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    def test_imag(self):
        # [FUNCTION DOC] Checks vmap of torch.imag on cfloat and cdouble inputs.
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_real(self):
        # [FUNCTION DOC] Checks vmap of view_as_real on cfloat and cdouble inputs.
        self._test_complex_views(
            torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble]
        )

    def test_view_as_complex(self):
        # [FUNCTION DOC] Checks vmap of view_as_complex for float and double inputs: normal and
        # nested cases, batch dim next to or after the size-2 dim (when stride allows), and
        # RuntimeErrors for last dim stride not 1, missing dimensions, and last logical dim not
        # of size 2.
        def run_test(dtype):
            # [FUNCTION DOC] HELPER: Runs all view_as_complex vmap success and failure cases for
            # one dtype.
            def get(shape):
                # [FUNCTION DOC] HELPER: Returns a random tensor of the given shape and
                # captured dtype.
                return torch.randn(shape, dtype=dtype)

            op = torch.view_as_complex
            test = self._vmap_view_test
            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3, 2])])
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2)
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1, 2])])
            test(vmap(op), [get([B1, 2, 5, B0, 3, 2])], in_dims=2)
            test(
                vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])], in_dims=2, out_dims=2
            )

            # Interesting case #1: Batch dim directly before dim of size 2
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)

            # Interesting case #2: Batch dim at end of tensor, success cases
            # view_as_complex requires that the dim with size 2 have stride 1
            # in order for the view to function property
            test(op, [get([B0, 2]).transpose(0, 1)], in_dims=1)
            test(vmap(op, in_dims=1), [get([B0, B1, 2]).movedim(1, 2)])
            test(vmap(op, in_dims=2), [get([B0, 3, B1, 2]).movedim(2, 3)])

            # Interesting case #3: Batch dim at end of tensor, failure cases
            msg = "Tensor must have a last dimension with stride 1"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([2, B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op, in_dims=1), in_dims=1)(get([2, B0, B1]))

            # Invalid input: no dimension of size 2
            msg = "Input tensor must have one or more dimensions"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))

            # Invalid input: Batch dim has size 2, but the logical last dim does
            # not have size 2
            msg = "Tensor must have a last dimension of size 2"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([3, 2]))

        for dtype in [torch.float, torch.double]:
            run_test(dtype)

    def test_is_complex(self):
        # [FUNCTION DOC] Checks is_complex is queryable inside vmap: yields 1 for each complex
        # example and 0 for real.
        ctensor = torch.randn(3, dtype=torch.cfloat)
        tensor = torch.randn(3)

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns tensor one if its input is complex, otherwise
            # tensor zero.
            if x.is_complex():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(ctensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(tensor), torch.tensor([0, 0, 0]))

    def test_is_floating_point(self):
        # [FUNCTION DOC] Checks is_floating_point inside vmap: 1 for float examples, 0 for
        # integer ones.
        float_tensor = torch.tensor([1.0, 2.0, 3.0])
        long_tensor = torch.tensor([1, 2, 3])

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns tensor one if its input is floating point,
            # otherwise zero.
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    def test_is_contiguous(self):
        # [FUNCTION DOC] Checks is_contiguous inside vmap reports per-example contiguity for
        # various batch-dim placements, nested vmaps, empty tensors, and that querying with
        # channels_last formats raises a RuntimeError (NYI), eagerly and under compile. Skipped
        # on Windows.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns float one if the input is contiguous, otherwise
            # float zero.
            if x.is_contiguous():
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)

        B0, B1 = 3, 5

        # Single batch dim
        contig = torch.randn(B0, 2, 7)
        self.assertEqual(vmap(foo)(contig), torch.ones(B0))

        noncontig = torch.randn(2, B0, 7)
        self.assertEqual(vmap(foo, in_dims=1)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, B0, 7).movedim(1, 0)
        self.assertEqual(vmap(foo)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, 7, B0)
        self.assertEqual(vmap(foo, in_dims=2)(noncontig), torch.zeros(B0))

        # Multiple batch dims
        contig = torch.randn(B0, B1, 3)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3)
        self.assertEqual(vmap(vmap(foo), in_dims=1)(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3).movedim(0, 1)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        noncontig = torch.randn(B0, 3, B1)
        self.assertEqual(vmap(vmap(foo, in_dims=1))(noncontig), torch.zeros(B0, B1))

        # is_contiguous on empty tensor is True
        def bar(x):
            # [FUNCTION DOC] HELPER: Asserts its input is contiguous and returns it.
            if not x.is_contiguous():
                raise AssertionError("Expected tensor to be contiguous")
            return x

        vmap(bar)(torch.randn(B0, 0, 3))
        vmap(bar, in_dims=1)(torch.randn(0, B0, 3))
        vmap(bar)(torch.randn(B0, 0, 3).transpose(-1, -2))

        # is_contiguous with other memory formats
        def baz(x, memory_format):
            # [FUNCTION DOC] HELPER: Queries is_contiguous with the given memory format and
            # returns the input.
            x.is_contiguous(memory_format=memory_format)
            return x

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)

        for mf in (torch.channels_last, torch.channels_last_3d):

            @torch.compile(backend="eager", fullgraph=True)
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled (eager backend, full graph) function that
                # returns sin of the input if it is contiguous in the loop's memory format,
                # else cos.
                if x.is_contiguous(memory_format=mf):
                    return x.sin()
                return x.cos()

            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(f)(torch.randn(3, 3))

    def test_unsqueeze(self):
        # [FUNCTION DOC] Checks vmap of unsqueeze at the front and last dim (positive,
        # negative), with view checks, plus nested vmaps with canonical and shuffled batch
        # dim order.
        op = torch.unsqueeze
        test = self._vmap_view_test
        B0, B1 = 7, 11

        # unsqueeze dim 0
        test(op, (torch.rand(B0, 2, 5), 0), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 0), in_dims=(1, None))

        # unsqueeze last dim (positive)
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 2), in_dims=(1, None))

        # unsqueeze last dim (negative)
        test(op, (torch.rand(B0, 2, 5), -1), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), -1), in_dims=(1, None))

        # nested vmaps
        def unsqueeze_0(x):
            # [FUNCTION DOC] HELPER: Unsqueezes dim 0.
            return torch.unsqueeze(x, 0)

        def unsqueeze_last(x):
            # [FUNCTION DOC] HELPER: Unsqueezes the last dim.
            return torch.unsqueeze(x, -1)

        # bdims in canonical order
        test(vmap(unsqueeze_0), (torch.rand(B0, B1, 2),))
        test(vmap(unsqueeze_last), (torch.rand(B0, B1, 2),))

        # wild bdims
        test(vmap(unsqueeze_0), (torch.rand(B1, 2, B0),), in_dims=2)
        test(vmap(unsqueeze_0, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)
        test(vmap(unsqueeze_last), (torch.rand(B1, 2, B0),), in_dims=2)
        test(vmap(unsqueeze_last, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)

    def test_movedim(self):
        # [FUNCTION DOC] Checks vmap of movedim with int source/destination and with list
        # variants, view checked, plain and doubly nested.
        op = torch.movedim
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # movedim(tensor, int, int) variant
        test(op, (torch.rand(B0, 2, 5), 0, 1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 0, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), 0, 1),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), 0, 1),
            in_dims=(2, None, None),
        )

        # movedim(tensor, intlist, intlist) variant
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )

    def test_mm(self):
        # [FUNCTION DOC] Checks vmap of mm: shape mismatches raise a RuntimeError matching
        # "Shape mismatch"; left, right and both arguments batched, plain and nested.
        op = torch.mm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5, 2)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2)),
            in_dims=(None, 0),
        )

    def test_mv(self):
        # [FUNCTION DOC] Checks vmap of mv: shape mismatches raise a RuntimeError; left, right
        # and both arguments batched, plain and nested.
        op = torch.mv
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        test(
            vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0)
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_narrow(self):
        # [FUNCTION DOC] Checks vmap of narrow with view checks, at different in_dims and
        # nested vmaps.
        op = torch.narrow
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 2, B0, 5), 1, 0, 0),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3),
            in_dims=(2, None, None, None),
        )

    def test_new_empty(self):
        # [FUNCTION DOC] Checks vmap of new_empty gives results with the expected shape
        # (batch dim prepended) for single and nested vmaps; contents are not checked since
        # they are undefined.
        # Empty is non-deterministic so we just check that the shape of the
        # output tensor is what we expect and that the vmap fallback isn't used.
        op = Tensor.new_empty

        B0, B1 = 7, 11

        result = vmap(lambda x: op(x, [2, 3]))(torch.randn(B0))
        self.assertEqual(result.shape, [B0, 2, 3])

        result = vmap(lambda x: op(x, []))(torch.randn(B0))
        self.assertEqual(result.shape, [B0])

        result = vmap(vmap(lambda x: op(x, [2, 3])))(torch.randn(B0, B1))
        self.assertEqual(result.shape, [B0, B1, 2, 3])

    def test_new_empty_strided(self):
        # [FUNCTION DOC] Checks vmap of new_empty_strided yields the batch dim prepended and
        # the strides expected from the underlying storage size, for contiguous, expanded and
        # odd strides.
        # Empty is non-deterministic so we just check that the size and shape
        # of the output are what we expect and that the vmap fallback isn't used
        B0, B1 = 7, 11

        def _test_single_vmap(size, stride, B0):
            # [FUNCTION DOC] HELPER: Checks shape and strides of a single-level vmap
            # of new_empty_strided.
            x = torch.randn(B0)
            result = vmap(lambda x: x.new_empty_strided(size, stride))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0] + size)
            self.assertEqual(result.stride(), [S] + stride)

        def _test_double_vmap(size, stride, B0, B1):
            # [FUNCTION DOC] HELPER: Checks shape and strides for two-level vmaps, with batch
            # dims in both orders.
            x = torch.randn(B0, B1)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

            x = torch.randn(B1, B0)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(
                x
            )
            S = x.new_empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

        # contiguous case
        _test_single_vmap([2, 3, 5], [3 * 5, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [3 * 5, 5, 1], B0, B1)

        # expanded
        _test_single_vmap([2, 3, 5], [0, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [0, 5, 1], B0, B1)

        # some of these cases are pretty strange, just verifying that if
        # empty_strided allows them then BatchedTensor.new_empty_strided
        # can as well
        for shape in [[2, 3, 4], [0, 2, 0]]:
            for strides in [[12, 4, 1], [2, 4, 6], [0, 0, 0]]:
                _test_single_vmap(shape, strides, B0)
                _test_double_vmap(shape, strides, B0, B1)

    def test_new_zeros(self):
        # [FUNCTION DOC] Checks vmap of new_zeros with explicit sizes and an empty size, plain
        # and nested.
        op = Tensor.new_zeros
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11

        test(lambda x: op(x, 2, 3), (torch.rand(B0),))
        test(lambda x: op(x, []), (torch.rand(B0),))
        test(vmap(lambda x: op(x, 3, 5)), (torch.rand(B0, B1),))

    def test_select(self):
        # [FUNCTION DOC] Checks vmap of select with view checks, different in_dims, and
        # nested vmaps.
        op = torch.select
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 0, 0), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1), in_dims=(1, None, None))
        test(vmap(lambda t: op(t, 1, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )

    def test_roll_no_dims(self):
        # [FUNCTION DOC] Checks vmap of roll with only a shift (flattened roll), plain
        # and nested.
        op = torch.roll
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 3), in_dims=(1, None))
        test(vmap(lambda t: op(t, 3)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, 3), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )

    def test_stack(self):
        # [FUNCTION DOC] Checks vmap of stack with batched and unbatched inputs, dims 0 and -1,
        # and nested vmaps.
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            # [FUNCTION DOC] HELPER: Returns a function that stacks its tensor arguments along
            # the given dim.
            def op(*tensors):
                # [FUNCTION DOC] HELPER: Stacks the given tensors along the captured dim.
                return torch.stack(tensors, dim=dim)

            return op

        test(get_op(0), (torch.rand(B0, 3), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(3), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 2)),
            in_dims=(None, 0),
        )
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 2)),
            in_dims=(None, 0),
        )

    def test_slice(self):
        # [FUNCTION DOC] Checks vmap of slicing indexing (view checked) at different batch dim
        # placements, plain and nested.
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)
        test(
            vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2
        )
        test(
            vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2),
            (torch.rand(3, 5, B0, B1, B2),),
            in_dims=2,
        )

    @xfailIfTorchDynamo
    def test_squeeze(self):
        # [FUNCTION DOC] Checks vmap of squeeze, with and without dim (including negative dims),
        # on tensors with size-1 dims and batch size 1; an out-of-range dim must raise a
        # RuntimeError under vmap. Marked as expected failure under dynamo.
        def verify_behavior(op, min_ndim=1):
            # [FUNCTION DOC] HELPER: Runs view-checked vmap cases for a squeeze op; the
            # single-dim cases are only run when the op needs at most one dim after batching
            # (min_ndim argument).
            test = self._vmap_view_test
            B0, B1 = 1, 11
            # These tests cannot be used with an operator that requires more
            # than 1 dimension after batching.
            if min_ndim <= 1:
                test(op, (torch.rand(B0),))
                test(op, (torch.rand(B1),))
                test(vmap(op), (torch.rand(B0, B1, 1),))
                test(vmap(op), (torch.rand(B1, 1, B0),), in_dims=2)
            test(op, (torch.rand(B0, 3, 5),))
            test(op, (torch.rand(1, B0, 5),), in_dims=1)
            test(op, (torch.rand(B0, 0, 1, 5, 1),))
            test(op, (torch.rand(B0, 1, 1, 1, 1),))
            test(vmap(op), (torch.rand(B0, B1, 1, 3, 4),))
            test(vmap(op), (torch.rand(B1, 1, B0, 4, 5),), in_dims=2)

        verify_behavior(torch.squeeze)
        verify_behavior(lambda x: torch.squeeze(x, dim=0), min_ndim=1)
        verify_behavior(lambda x: torch.squeeze(x, dim=1), min_ndim=2)
        verify_behavior(lambda x: torch.squeeze(x, dim=-1), min_ndim=2)
        verify_behavior(lambda x: torch.squeeze(x, dim=-2), min_ndim=3)

        msg = ""
        try:
            torch.squeeze(torch.rand(10), dim=1)
        except IndexError as err:
            msg = str(err)
        with self.assertRaises(RuntimeError, msg=msg):
            vmap(lambda x: torch.squeeze(x, dim=1))(torch.rand(10))

    def _test_mean_sum_dim(self, op):
        # [FUNCTION DOC] HELPER: Checks vmap of a dim-reducing mean/sum style op (positive,
        # negative dims, nested, various in_dims/out_dims) against the reference.
        test = self._vmap_test
        B0, B1 = 5, 7

        # Single vmap, various in_dims / out_dims
        test(lambda x: op(x, 0), [torch.randn([B0])])
        test(lambda x: op(x, -1), [torch.randn([B0])])
        test(lambda x: op(x, 0), [torch.randn([B0, 3])])
        test(lambda x: op(x, -1), [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(lambda x: op(x, 2), [torch.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(lambda x: op(x, 0)), [torch.randn([B0, B1])])
        test(vmap(lambda x: op(x, -1)), [torch.randn([B0, B1])])
        test(vmap(lambda x: op(x, -2)), [torch.randn([B1, 2, 5, B0, 3])], in_dims=2)
        test(
            vmap(lambda x: op(x, 2), in_dims=2),
            [torch.randn([2, 5, B0, B1, 3])],
            in_dims=2,
            out_dims=2,
        )

    def test_sum_dim(self):
        # [FUNCTION DOC] Checks vmap of sum along a dim.
        self._test_mean_sum_dim(torch.sum)

    def test_mean_dim(self):
        # [FUNCTION DOC] Checks vmap of mean along a dim.
        self._test_mean_sum_dim(torch.mean)

    def test_argmax_dim(self):
        # [FUNCTION DOC] Checks vmap of argmax with no dim and with dims 0, -1, 2 against the
        # loop fallback computations and exhaustive vmap variants.
        def test(f, args):
            # [FUNCTION DOC] HELPER: For each pair of looped and vmapped outputs from the
            # exhaustive helper, asserts equality.
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(f, args, {}):
                self.assertEqual(loop_out, batched_out)

        B0 = 5
        test(lambda x: torch.argmax(x), [torch.randn(B0)])
        test(lambda x: torch.argmax(x), [torch.randn(B0, 2, 3)])
        test(lambda x: torch.argmax(x, 0), [torch.randn(B0, 2, 3)])
        test(lambda x: torch.argmax(x, -1), [torch.randn(B0, 2, 3)])
        test(lambda x: torch.argmax(x, 2), [torch.randn(B0, 2, 3)])

    def _test_sum_mean(self, op):
        # [FUNCTION DOC] HELPER: Checks vmap of a full reduction op (sum or mean over all
        # elements), single and nested.
        test = self._vmap_test
        B0, B1 = 5, 7

        # Single vmap, various in_dims / out_dims
        test(op, [torch.randn([B0])])
        test(op, [torch.randn([B0, 3])])
        test(op, [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(op, [torch.randn([2, 5, B0, 3])], in_dims=2)

        # Doubly nested vmap
        test(vmap(op), [torch.randn([B0, B1])])
        test(vmap(op), [torch.randn([B1, 2, 5, B0, 3])])
        test(vmap(op), [torch.randn([2, 5, B0, B1, 3])], in_dims=2)

    def test_sum(self):
        # [FUNCTION DOC] Checks vmap of sum over all elements.
        self._test_sum_mean(torch.sum)

    def test_mean(self):
        # [FUNCTION DOC] Checks vmap of mean over all elements.
        self._test_sum_mean(torch.mean)

    def test_repeat(self):
        # [FUNCTION DOC] Checks vmap of Tensor.repeat with a fixed repeat tuple at batch dim 0
        # and 1.
        test = self._vmap_test
        B0 = 7
        op = Tensor.repeat
        test(lambda x: op(x, (2, 3)), (torch.rand(B0, 1, 1),))
        test(lambda x: op(x, (2, 3)), (torch.rand(1, B0, 1),), in_dims=1)

    def test_repeat_interleave(self):
        # [FUNCTION DOC] Tests repeat_interleave under vmap (issue 135424): mapping over
        # repeats with a fixed output_size works for index and value forms; mapping only over
        # values with constant repeats works; omitting output_size when repeats is mapped
        # raises a RuntimeError.
        # https://github.com/pytorch/pytorch/issues/135424
        # Vmapping over the `repeats` tensor requires `output_size` because the
        # output length (the sum of `repeats`) is data-dependent.
        repeats = torch.tensor([[4, 0, 8], [2, 2, 8], [4, 2, 6], [2, 4, 6], [0, 4, 8]])
        output_size = 12  # every row sums to 12

        # repeat_interleave.Tensor: the base overload returning gather indices.
        def index_op(rep):
            # [FUNCTION DOC] HELPER: repeat_interleave of a repeats vector with the fixed
            # output size.
            return torch.repeat_interleave(rep, output_size=output_size)

        expected = torch.stack([index_op(r) for r in repeats])
        self.assertEqual(vmap(index_op)(repeats), expected)

        # repeat_interleave.self_Tensor: torch.repeat_interleave(input, repeats).
        def op(rep):
            # [FUNCTION DOC] HELPER: repeat_interleave of an arange of values using given
            # repeats and output size.
            values = torch.arange(1, rep.shape[0] + 1)
            return torch.repeat_interleave(values, rep, output_size=output_size)

        expected = torch.stack([op(r) for r in repeats])
        self.assertEqual(vmap(op)(repeats), expected)

        # Batching `values` while leaving `repeats` an unbatched constant keeps
        # the output length the same per sample, so `output_size` is optional.
        values = torch.randn(5, 3)
        const_repeats = torch.tensor([2, 1, 3])

        def op2(v):
            # [FUNCTION DOC] HELPER: repeat_interleave of the input with captured
            # constant repeats.
            return torch.repeat_interleave(v, const_repeats)

        expected = torch.stack([op2(v) for v in values])
        self.assertEqual(vmap(op2)(values), expected)

        # Omitting `output_size` while vmapping over `repeats` is data-dependent
        # and must raise a clear error.
        def bad(rep):
            # [FUNCTION DOC] HELPER: Like op but without output_size, expected to fail when
            # repeats is batched.
            values = torch.arange(1, rep.shape[0] + 1)
            return torch.repeat_interleave(values, rep)

        with self.assertRaisesRegex(RuntimeError, "output_size"):
            vmap(bad)(repeats)

    @skipIfTorchDynamo()
    def test_slogdet(self):
        # [FUNCTION DOC] Checks vmap of linalg.slogdet for various matrix shapes and batch dim
        # at the end. Skipped under dynamo.
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0 = 7
        op = torch.linalg.slogdet
        test(op, (torch.rand(B0, 1, 1),))
        test(op, (torch.rand(B0, 2, 2),))
        test(op, (torch.rand(B0, 3, 2, 2),))
        test(op, (torch.rand(3, 2, 2, B0),), in_dims=3)

    def test_reshape(self):
        # [FUNCTION DOC] Checks vmap of reshape: view-checked when the per-example reshape can
        # be a view, not view-checked when batch dim placement forces a copy; includes flatten
        # and nested vmaps.
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        op = torch.reshape
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None), check_view=True)
        test(
            op, (torch.rand(2, B0, 5), [1, 1, 10]), in_dims=(1, None), check_view=False
        )
        test(
            vmap(lambda t: t.reshape([-1])),
            (torch.rand(B0, B1, 2, 5),),
            check_view=True,
        )
        test(
            vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1),
            (torch.rand(3, B1, 2, B2, 5, B0),),
            in_dims=5,
            check_view=False,
        )

    def test_reshape_as(self):
        # [FUNCTION DOC] Checks vmap of reshape_as with batched/unbatched self and other, view
        # checks only where a view is possible, and nested vmaps.
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.reshape_as
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)), check_view=True)
        test(
            op,
            (torch.rand(2 * 5), torch.rand(B0, 2, 5)),
            in_dims=(None, 0),
            check_view=True,
        )
        test(
            op,
            (torch.rand(B0, 2 * 5), torch.rand(2, 5)),
            in_dims=(0, None),
            check_view=True,
        )

        test(
            op,
            (torch.rand(2, B0, 5), torch.rand(1, 1, 10)),
            in_dims=(1, None),
            check_view=False,
        )

        test(
            vmap(op),
            (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)),
            check_view=True,
        )
        test(
            vmap(vmap(op, in_dims=(2, None)), in_dims=(1, None)),
            (torch.rand(3, B1, 2, B2, 5, B0), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(5, 0),
            check_view=False,
        )

    def test_result_type(self):
        # [FUNCTION DOC] Checks vmap of result_type (wrapped to return a scalar tensor of that
        # dtype) for tensor pairs of different dtypes, tensor with Python int/float, and tensor
        # with 0-dim tensors, at several ranks and with a batched/unbatched mix; grad
        # propagation check disabled.
        def scalar_tensor_with_dtype(op):
            # [FUNCTION DOC] HELPER: Wraps a dtype-returning op so it returns a scalar ones
            # tensor of that dtype.
            def wrapped(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the wrapped op and returns a scalar ones tensor
                # of the resulting dtype.
                dtype = op(*args, **kwargs)
                return torch.ones([], dtype=dtype)

            return wrapped

        test = self._vmap_test
        op = scalar_tensor_with_dtype(torch.result_type)

        B0 = 2

        test(
            op,
            (torch.randn(B0), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        test(lambda x: op(x, 1), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0),), check_propagates_grad=False)

        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )

        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        test(lambda x: op(x, 1), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0, 2),), check_propagates_grad=False)

        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )

        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

    def test_tensor_split(self):
        # [FUNCTION DOC] Checks vmap of tensor_split with an int section count and with a list
        # of indices, at different dims, in_dims and nested vmaps (view-checked).
        test = self._vmap_view_test
        op = torch.tensor_split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.tensor_split(self, indices_or_sections: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 5, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 150, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.tensor_split(self, indices_or_sections: List[int], dim)
        test(
            op,
            (torch.rand(B0, 2, 1024), [50, 100, 378, 890], -1),
            in_dims=(0, None, None),
        )
        test(
            op,
            (torch.rand(2, B0, 1024), [50, 100, 212, 345, 0, 378, 890], 1),
            in_dims=(1, None, None),
        )
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [50, 100, 212, 345, 0, 378, 890], 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    @skipIfTorchDynamo("really slow")
    def test_split(self):
        # [FUNCTION DOC] Checks vmap of split with an int size and with a list of sizes at
        # different dims, in_dims and nested vmaps (view-checked). Skipped under dynamo as
        # really slow.
        test = self._vmap_view_test
        op = torch.split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 101, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 130, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.split(self, split_size: List[int], dim)
        test(op, (torch.rand(B0, 2, 1024), [1, 1020, 3], -1), in_dims=(0, None, None))
        test(
            op, (torch.rand(2, B0, 1024), [100] * 10 + [24], 1), in_dims=(1, None, None)
        )
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [256] * 3 + [255], 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, [4] * 8 + [8] * 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_trace(self):
        # [FUNCTION DOC] Checks vmap of trace, plain and nested, at different batch
        # dim placements.
        op = torch.trace
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_transpose(self):
        # [FUNCTION DOC] Checks vmap of transpose (view-checked) with positive and negative
        # dims, nested vmaps, and that for a scalar-per-example input the output is the same
        # tensor object.
        op = torch.transpose
        test = self._vmap_view_test

        B0, B1, B2 = 7, 11, 13
        test(lambda x: op(x, 0, 1), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, -1, -2), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, 3, 1), (torch.rand(B0, 2, 5, 4, 6),))
        test(lambda x: op(x, 1, 0), (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(lambda x: op(x, 0, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 5, B2),),
            in_dims=2,
        )

        # Special case: scalar tensor
        for dim1, dim2 in itertools.product([0, -1], [0, -1]):
            x = torch.rand(B0)
            result = vmap(lambda x: op(x, dim1, dim2))(x)
            self.assertTrue(result is x)

    def test_t(self):
        # [FUNCTION DOC] Checks vmap of torch.t (view-checked), plain and nested.
        op = torch.t
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_T_numpy(self):
        # [FUNCTION DOC] Checks vmap of the T property (numpy-style full
        # transpose), view-checked.
        def op(t):
            # [FUNCTION DOC] HELPER: Returns the T property of its input.
            return t.T

        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 3, 5),))
        test(op, (torch.rand(2, B0, 3, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(op), (torch.rand(B1, 2, B0, 3, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 3, B2, 5),), in_dims=2)

    def test_to(self):
        # [FUNCTION DOC] Checks vmap of tensor to with device string, dtype, other-tensor
        # argument (batched and unbatched), nested, and of the casting shortcuts double, float,
        # int, long.
        test = self._vmap_test
        B0, B1 = 7, 11

        test(lambda t: t.to("cpu"), (torch.rand(B0),))
        test(lambda t: t.to(torch.double), (torch.rand(B0),))
        test(
            lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64))
        )
        test(
            lambda t, o: t.to(o),
            (torch.rand(B0), torch.randn(B0, dtype=torch.float64)),
            in_dims=(0, None),
        )
        test(vmap(lambda t: t.to(torch.double)), (torch.rand(B0, B1, 3),))

        # also test some casting methods
        test(lambda t: t.double(), (torch.rand(B0),))
        test(lambda t: t.float(), (torch.rand(B0),))
        test(lambda t: t.int(), (torch.rand(B0),), check_propagates_grad=False)
        test(lambda t: t.long(), (torch.rand(B0),), check_propagates_grad=False)

    def test_unfold(self):
        # [FUNCTION DOC] Checks vmap of Tensor.unfold (view-checked) with different dims, sizes,
        # steps, in_dims and nested vmaps.
        op = torch.Tensor.unfold
        test = self._vmap_view_test
        B0, B1, B2 = 3, 2, 5

        test(op, (torch.rand(B0, 7, 11), 0, 2, 1), in_dims=(0, None, None, None))
        test(op, (torch.rand(7, B0, 11), 1, 4, 2), in_dims=(1, None, None, None))
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 7, B0, 11), 1, 5, 1),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4),
            in_dims=(2, None, None, None),
        )

    def test_unbind(self):
        # [FUNCTION DOC] Checks vmap of unbind (view-checked), including empty dims, different
        # dims, in_dims and nested vmaps.
        test = self._vmap_view_test
        op = torch.unbind
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 1024), -1), in_dims=(0, None))
        test(op, (torch.rand(B0, 2, 0),))
        test(op, (torch.rand(2, B0, 7), 0), in_dims=(1, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 1023, B0, 5), 1),
            in_dims=(2, None),
        )
        test(
            vmap(vmap(lambda t: op(t, dim=1), in_dims=2)),
            (torch.rand(B1, 2, B0, 32, B2),),
            in_dims=2,
        )

    def test_view(self):
        # [FUNCTION DOC] Checks vmap of Tensor.view: a view that would be incorrect with
        # the batch dim raises a RuntimeError; valid views (including flatten, nested)
        # match the reference.
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.view

        # We should error out if the view would produce an incorrect result
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, None))(torch.rand(2, B0, 5), [10])

        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None))
        test(op, (torch.rand(B0, 4, 5), [1, 2, 1, 10]), in_dims=(0, None))
        test(vmap(lambda t: t.view([-1])), (torch.rand(B0, B1, 2, 5, 3),))
        test(
            vmap(vmap(lambda t: t.reshape([-1])), in_dims=1),
            (torch.rand(B2, B0, B1, 3, 2, 5),),
            in_dims=1,
        )

    def test_view_as(self):
        # [FUNCTION DOC] Checks vmap of Tensor.view_as: an invalid view raises a RuntimeError;
        # batched or unbatched self/other, and nested vmaps match the reference.
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.view_as

        # We should error out if the view would produce an incorrect result
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, 0))(torch.rand(2, B0, 5), torch.rand(B0, 10))

        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)))
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0))
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None))

        test(op, (torch.rand(B0, 4, 5), torch.rand(2, 1, 1, 10)), in_dims=(0, None))

        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)))
        test(
            vmap(vmap(op, in_dims=(0, None)), in_dims=(0, None)),
            (torch.rand(B1, B2, B0, 3, 2, 5), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(2, 0),
        )

    def test_view_dtype(self):
        # [FUNCTION DOC] Checks vmap of the view dtype overload for uint8 and int32
        # reinterpretation, and that a zero-dim per-example float tensor viewed as byte
        # raises a RuntimeError.
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        op = torch.ops.aten.view.dtype

        test(op, (torch.rand(2, 3, 4), torch.uint8), in_dims=(1, None), out_dims=1)
        test(op, (torch.rand(5), torch.int32), in_dims=(0, None), out_dims=0)
        with self.assertRaisesRegex(
            RuntimeError, r"dim\(\) cannot be 0 to view Float as Byte"
        ):
            vmap(op, in_dims=(0, None))(torch.rand(6), torch.uint8)

    def test_conv2d(self):
        # [FUNCTION DOC] Checks vmap of conv1d, conv2d, conv3d with and without bias, default
        # and with groups/stride/padding/dilation, comparing loop and batched results from the
        # exhaustive helper for all batching combinations.
        conv_setups = [
            (torch.nn.Conv1d, torch.conv1d, [2, 4, 15]),
            (torch.nn.Conv2d, torch.conv2d, [2, 4, 15, 20]),
            (torch.nn.Conv3d, torch.conv3d, [2, 4, 15, 20, 25]),
            # (torch.nn.ConvTranspose2d, torch.conv_transpose2d, [2, 4, 15, 20])
        ]
        for conv_mod, conv_fn, inp_shape in conv_setups:
            mod = conv_mod(4, 8, kernel_size=3)
            arg_values = [torch.randn(inp_shape), mod.weight, mod.bias]
            kwarg_values = {}
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

            arg_values = [torch.randn(inp_shape), mod.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

            mod2 = conv_mod(
                4, 8, kernel_size=3, groups=2, stride=3, padding=1, dilation=2
            )
            arg_values = [torch.randn(inp_shape), mod2.weight, mod2.bias]
            kwarg_values = dict(groups=2, stride=3, padding=1, dilation=2)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

            arg_values = [torch.randn(inp_shape), mod2.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

    def test_one_hot(self):
        # [FUNCTION DOC] Checks vmap of one_hot for a scalar and a 3-d integer input against the
        # loop and exhaustive batching combinations.
        sample_inputs = [
            (torch.randint(0, 3, []), 3),
            (torch.randint(0, 3, [2, 3, 4]), 4),
        ]
        for args in sample_inputs:
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                F.one_hot, args, {}
            ):
                self.assertEqual(loop_out, batched_out)

    def test_conj_bit(self):
        # [FUNCTION DOC] Checks the conjugate bit is handled under vmap: inside the mapped
        # function the input is not conj, conj makes it conj, and the result equals conj of
        # the input.
        x = torch.tensor([1 + 1j, 2 + 1j])

        def foo(x):
            # [FUNCTION DOC] HELPER: Asserts input is not conj and its conj is, then returns
            # the conj.
            if x.is_conj():
                raise AssertionError("Expected x to not be conj")
            y = x.conj()
            if not y.is_conj():
                raise AssertionError("Expected y to be conj")
            return y

        res = vmap(foo)(x)
        self.assertEqual(res, x.conj())

    def test_mode_key(self):
        # [FUNCTION DOC] Checks random numbers under nested vmap with different randomness match
        # the unbatched random generation order given the same seed; also checks a plain randn
        # call outside vmap still produces a 0-dim tensor afterwards.
        def vmap_f(x):
            # [FUNCTION DOC] HELPER: Adds a scalar random normal value to its input.
            return x + torch.randn(())

        def naive_f(x, shape):
            # [FUNCTION DOC] HELPER: Adds a random normal tensor of the given shape to
            # its input.
            return x + torch.randn(shape)

        torch.manual_seed(0)
        out1 = vmap(vmap(vmap_f, randomness="different"), randomness="different")(
            torch.ones(2, 3)
        )

        torch.manual_seed(0)
        out2 = naive_f(torch.ones(2, 3), (2, 3))
        self.assertEqual(out1, out2)

        torch.manual_seed(0)
        out1 = vmap(vmap(vmap_f, randomness="different"), randomness="different")(
            torch.ones(2, 3, 4)
        )

        torch.manual_seed(0)
        out2 = naive_f(torch.ones(2, 3, 4), (2, 3, 1))
        self.assertEqual(out1, out2)

        self.assertTrue(torch.randn(()).dim() == 0)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    @parametrize("randomness", ["error", "same"])
    def test_chunk_vmap(self, in_dim, out_dim, randomness):
        # [FUNCTION DOC] Parametrized over in_dim, out_dim and randomness (error, same). Checks
        # chunk_vmap with various chunk counts equals plain vmap, restoring the RNG state before
        # each run.
        x = torch.randn(4, 5, 6)

        def f(x):
            # [FUNCTION DOC] HELPER: Sine of its input, plus uniform noise when randomness is
            # not error.
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        rs = torch.get_rng_state()
        expected = vmap(f, in_dims=in_dim, out_dims=out_dim, randomness=randomness)(x)

        for chunks in [1, 2, 3, 4, 7, 10, 16]:
            torch.set_rng_state(rs)
            output = chunk_vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunks=chunks,
            )(x)
            self.assertEqual(output, expected)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize(self, in_dim, out_dim, randomness):
        # [FUNCTION DOC] Parametrized over in_dim, out_dim and randomness. Checks vmap with
        # chunk_size (many values) equals unchunked vmap for five function shapes: tensor to
        # tensor, pair input, dict output, dict input with an unmapped entry, and dict input
        # with a non-tensor entry.
        x = torch.randn(4, 5, 6)
        y = torch.randn_like(x)

        # fn: Single Input/Single Output
        def f(x):
            # [FUNCTION DOC] HELPER: Single tensor in and out: sine, plus noise if randomness is
            # not error.
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        f_args = (x,)
        f_kwargs = {"in_dims": in_dim, "out_dims": out_dim, "randomness": randomness}

        # fn: Nested Input/Single Output
        def f1(pair):
            # [FUNCTION DOC] HELPER: Pair input: sine of the first plus cosine of the second,
            # plus noise unless randomness is error.
            x, y = pair
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return z

        f1_args = ((x, y),)
        f1_kwargs = {
            "in_dims": ((in_dim,) * 2,),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        # fn: Single Input/Nested Output
        def f2(x):
            # [FUNCTION DOC] HELPER: Returns a dict with the noisy sine and that plus two.
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return {"out": y, "out1": y + 2}

        f2_args = (x,)
        f2_kwargs = {"in_dims": in_dim, "out_dims": out_dim, "randomness": randomness}

        # fn: Nested Input/Nested Output (first tensor is not vmapped).
        def f3(inp_dict):
            # [FUNCTION DOC] HELPER: Dict input with an unmapped first tensor; returns a dict
            # with the result and a tuple of it and it plus one.
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return {"z": z, "tuple": (z, z + 1)}

        f3_args = (
            {
                "inp": x.index_select(in_dim, torch.tensor([0])).squeeze(in_dim),
                "inp1": y,
            },
        )
        f3_kwargs = {
            "in_dims": ({"inp": None, "inp1": in_dim},),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        # fn: Nested Input/Nested Output (first argument is not a Tensor).
        def f4(inp_dict):
            # [FUNCTION DOC] HELPER: Dict input whose first entry is a Python float; returns a
            # dict with the result and a tuple of it and it plus one.
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            z = x + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return {"z": z, "tuple": (z, z + 1)}

        f4_args = ({"inp": 2.0, "inp1": y},)
        f4_kwargs = {
            "in_dims": ({"inp": None, "inp1": in_dim},),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        fns_and_args = (
            (f, f_args, f_kwargs),
            (f1, f1_args, f1_kwargs),
            (f2, f2_args, f2_kwargs),
            (f3, f3_args, f3_kwargs),
            (f4, f4_args, f4_kwargs),
        )
        for fn, args, kwargs in fns_and_args:
            rs = torch.get_rng_state()
            expected_vmap = vmap(fn, **kwargs)(*args)
            for chunk_size in (1, 2, 3, 4, 7, 10, 16, 100):
                torch.set_rng_state(rs)
                output = vmap(fn, chunk_size=chunk_size, **kwargs)(*args)
                self.assertEqual(output, expected_vmap)

    @parametrize("in_dim", [0, 1])
    @parametrize("out_dim", [0, 1])
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize_error(self, in_dim, out_dim, randomness):
        # [FUNCTION DOC] Checks vmap with chunk_size of minus one or zero raises a ValueError
        # (must be None or positive), and an out_dims structure that does not match the outputs
        # raises a ValueError.
        x = torch.randn(4, 5, 6)

        def f(x):
            # [FUNCTION DOC] HELPER: Sine of its input, plus noise if randomness is not error.
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        # Incorrect `chunk_size`
        for chunk_size in (-1, 0):
            with self.assertRaisesRegex(
                ValueError, "vmap: chunk_size should be None or greater than 0."
            ):
                vmap(
                    f,
                    in_dims=in_dim,
                    out_dims=out_dim,
                    randomness=randomness,
                    chunk_size=chunk_size,
                )(x)

        # Incorrect `out_dims`
        msg = "out_dims is not compatible with the structure of `outputs`"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(
                f,
                in_dims=in_dim,
                out_dims=(out_dim, out_dim),
                randomness=randomness,
                chunk_size=2,
            )(x)

    @parametrize("in_dim", [0, 1])
    @parametrize("out_dim", [0, 1])
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize_composition(self, in_dim, out_dim, randomness):
        # [FUNCTION DOC] Checks two nested vmaps both with chunk_size equal the unchunked nested
        # vmap for tensor, pair, dict-output and dict-input functions, restoring the RNG state
        # before each run.
        x = torch.randn(4, 5, 6)
        y = torch.randn_like(x)

        # fn: Single Input/Single Output
        def f(x):
            # [FUNCTION DOC] HELPER: Sine of its input, plus noise if randomness is not error.
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        f_args = (x,)

        # fn: Nested Input/Single Output
        def f1(pair):
            # [FUNCTION DOC] HELPER: Pair input: sine plus cosine, plus noise if randomness is
            # not error.
            x, y = pair
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return z

        f1_args = ((x, y),)

        # fn: Single Input/Nested Output
        def f2(x):
            # [FUNCTION DOC] HELPER: Returns a dict with the noisy sine and that plus two.
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return {"out": y, "out1": y + 2}

        f2_args = (x,)

        # fn: Nested Input/Nested Output
        def f3(inp_dict):
            # [FUNCTION DOC] HELPER: Dict input; returns a dict with the result and a tuple of
            # it and it plus one.
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return {"z": z, "tuple": (z, z + 1)}

        f3_args = ({"inp": x, "inp1": y},)

        for fn, args in ((f, f_args), (f1, f1_args), (f2, f2_args), (f3, f3_args)):
            rs = torch.get_rng_state()
            expected = vmap(
                vmap(fn, in_dims=in_dim, out_dims=out_dim, randomness=randomness),
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
            )(*args)
            for chunk_size in (1, 2, 3, 4, 7, 10, 16, 100):
                torch.set_rng_state(rs)
                actual = vmap(
                    vmap(
                        fn,
                        in_dims=in_dim,
                        out_dims=out_dim,
                        randomness=randomness,
                        chunk_size=chunk_size,
                    ),
                    in_dims=in_dim,
                    out_dims=out_dim,
                    randomness=randomness,
                    chunk_size=chunk_size,
                )(*args)
                self.assertEqual(actual, expected)


instantiate_parametrized_tests(TestVmapOperators)


def construct_v(output, batch_size, contig=False):
    # [FUNCTION DOC] HELPER: Builds a random tensor of vectors to batch over: shape is the batch
    # size followed by the output shape (contiguous), or batch dim last then moved to front (a
    # non-contiguous layout). Same dtype and device as the output.
    if contig:
        return torch.randn(
            batch_size, *output.shape, dtype=output.dtype, device=output.device
        )
    result = torch.randn(
        *output.shape, batch_size, dtype=output.dtype, device=output.device
    )
    return result.movedim(-1, 0)


def as_tuple(x):
    # [FUNCTION DOC] HELPER: Normalizes a value to a tuple: tuples returned as is, lists
    # converted, any other value wrapped in a one-element tuple.
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def differentiable(args):
    # [FUNCTION DOC] HELPER: Returns a tuple of those entries of the (normalized) input that are
    # tensors requiring grad; non-tensors and non-grad tensors are dropped.
    return tuple(
        arg
        for arg in as_tuple(args)
        if isinstance(arg, torch.Tensor) and arg.requires_grad
    )


def _get_rand_no_zeros(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Random uniform tensor from the given arguments with every entry
    # clamped to at least 0.1 in place, so values are never near zero. The requires_grad keyword
    # (default false) is stripped for creation and applied to the clamped result.
    requires_grad = kwargs.get("requires_grad", False)
    kwargs_without_requires_grad = kwargs.copy()
    kwargs_without_requires_grad["requires_grad"] = False
    result = torch.rand(*args, **kwargs_without_requires_grad)
    return result.clamp_min_(0.1).requires_grad_(requires_grad)


@markDynamoStrictTest
class TestVmapBatchedGradient(Namespace.TestVmapBase):
    # [CLASS DOC] Tests that batched gradients computed with vmap over autograd.grad
    # (vector-Jacobian and vector-Hessian products) match a sequential loop, for many ops. Test
    # methods take a device (device-generic instantiation). Marked dynamo strict.
    def _vmap_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards to the module-level vmap test function using self as
        # the case.
        return _vmap_test(self, *args, **kwargs)

    # Tests batched gradient computation of outputs = op(*args, **kwargs)
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    def _batched_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        # [FUNCTION DOC] HELPER: Computes outputs of op, keeps the differentiable ones (after
        # the optional output processing function), and for contiguous and non-contiguous
        # batched vectors vmap-tests a vector-Jacobian product against the sequential reference.
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        for contig in [True, False]:
            batched_vectors = tuple(
                construct_v(out, batch_size, contig) for out in outputs
            )

            def vector_jacobian_product(*vectors):
                # [FUNCTION DOC] HELPER: Vector-Jacobian product: autograd.grad of the captured
                # outputs with respect to the differentiable args, using the given vectors; retains
                # the graph.
                return torch.autograd.grad(
                    outputs, differentiable(args), vectors, retain_graph=True
                )

            self._vmap_test(
                vector_jacobian_product, batched_vectors, check_propagates_grad=False
            )

    # Tests batched second grad computation of outputs = op(*args, **kwargs).
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    #
    # NB: we only test computing batched gradients in the second gradient
    # computation. One specific use case that does this is computing the hessian
    # matrix of a scalar-valued function; this is useful in Bayesian Logistic
    # Regression.
    # It might be useful to have a test that computes batched first gradients and
    # then uses those to compute batched second gradients in the future.
    def _batched_grad_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        # [FUNCTION DOC] HELPER: Like the first-order version but for second derivatives: takes
        # first grads with create_graph using ones, then vmap-tests batched vector-Hessian
        # products. Fails the test if no first grad is differentiable.
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        ones = tuple(torch.ones_like(out) for out in outputs)
        # Same thing as summing together all of the outputs and calling .backward()
        first_grads = torch.autograd.grad(
            outputs, differentiable(args), ones, create_graph=True
        )
        first_grads = differentiable(first_grads)
        self.assertNotEqual(
            len(first_grads), 0, "None of the first grads depend on the input!"
        )

        for contig in [True, False]:
            batched_vectors = tuple(
                construct_v(grad, batch_size, contig) for grad in first_grads
            )

            def vector_hessian_product(*vectors):
                # [FUNCTION DOC] HELPER: Vector-Hessian product: differentiates the captured first
                # grads with respect to the differentiable args with allow_unused, drops None
                # results and raises AssertionError if none remain.
                outputs = torch.autograd.grad(
                    first_grads,
                    differentiable(args),
                    vectors,
                    retain_graph=True,
                    allow_unused=True,
                )
                outputs = tuple(out for out in outputs if out is not None)
                if len(outputs) == 0:
                    raise AssertionError("Expected at least one non-None output")
                return outputs

            self._vmap_test(
                vector_hessian_product, batched_vectors, check_propagates_grad=False
            )

    def _test_arithmetic(self, op, device, test_grad_grad=True):
        # [FUNCTION DOC] HELPER: Batched-grad checks for a binary arithmetic op on
        # tensor-tensor, scalar-tensor and tensor-scalar inputs, plus second-order check when
        # test_grad_grad is true.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        y = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        scalar = 3.14
        self._batched_grad_test(op, (x, y))
        self._batched_grad_test(op, (scalar, y))
        self._batched_grad_test(op, (x, scalar))

        if test_grad_grad:
            self._batched_grad_grad_test(op, (x, y))

    def test_add(self, device):
        # [FUNCTION DOC] Batched grad check for add (function and operator form), first
        # order only.
        self._test_arithmetic(torch.add, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x + y, device, test_grad_grad=False)

    def test_sub(self, device):
        # [FUNCTION DOC] Batched grad check for sub (function and operator form), first
        # order only.
        self._test_arithmetic(torch.sub, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x - y, device, test_grad_grad=False)

    def test_mul(self, device):
        # [FUNCTION DOC] Batched grad check for mul (function and operator form), first and
        # second order.
        self._test_arithmetic(torch.mul, device)
        self._test_arithmetic(lambda x, y: x * y, device)

    def test_div(self, device):
        # [FUNCTION DOC] Batched grad check for div (function and operator form), first and
        # second order.
        self._test_arithmetic(torch.div, device)
        self._test_arithmetic(lambda x, y: x / y, device)

    def test_binary_cross_entropy(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for binary_cross_entropy
        # against a fixed random target, with sigmoid probabilities as input.
        x = F.sigmoid(torch.randn(3, 2, device=device, requires_grad=True))
        target = torch.rand(3, 2, device=device)

        op = functools.partial(F.binary_cross_entropy, target=target)

        self._batched_grad_test(op, (x,), {})
        self._batched_grad_grad_test(op, (x,), {})

    def test_log_softmax(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for log_softmax over the
        # last dim.
        op = functools.partial(torch.log_softmax, dim=-1)
        x = torch.randn(3, 2, device=device, requires_grad=True)

        self._batched_grad_test(op, (x,), {})
        self._batched_grad_grad_test(op, (x,), {})

    def test_expand(self, device):
        # [FUNCTION DOC] Batched first-order grad check for expand to a larger shape.
        x = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x):
            # [FUNCTION DOC] HELPER: Expands its input to shape five by five by two by three.
            return x.expand(5, 5, 2, 3)

        self._batched_grad_test(op, (x,))

    @allowVmapFallbackUsage
    def test_index(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for advanced indexing of a
        # squared input. Allowed to use the vmap fallback.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        index = torch.tensor([[0, 0], [1, 1]], device=device)

        def op(x):
            # [FUNCTION DOC] HELPER: Squares the input and indexes it with the captured
            # index tensor.
            y = x * x
            return y[index]

        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_lgamma(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for lgamma.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.lgamma, (x,))
        self._batched_grad_grad_test(Tensor.lgamma, (x,))

    def test_log(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for log, on inputs bounded
        # away from zero.
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log, (x,))
        self._batched_grad_grad_test(torch.log, (x,))

    def test_logsumexp(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for logsumexp over the
        # last dim.
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)

        def op(x):
            # [FUNCTION DOC] HELPER: logsumexp over the last dim.
            return torch.logsumexp(x, -1)

        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_log1p(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for log1p.
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log1p, (x,))
        self._batched_grad_grad_test(torch.log1p, (x,))

    @allowVmapFallbackUsage
    def test_max(self, device):
        # [FUNCTION DOC] Batched first-order grad check for max (full reduction);
        # fallback allowed.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.max, (x,))

    @allowVmapFallbackUsage
    def test_median(self, device):
        # [FUNCTION DOC] Batched first-order grad check for median (full reduction);
        # fallback allowed.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.median, (x,))

    @allowVmapFallbackUsage
    def test_min(self, device):
        # [FUNCTION DOC] Batched first-order grad check for min (full reduction);
        # fallback allowed.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.min, (x,))

    def test_permute(self, device):
        # [FUNCTION DOC] Batched first-order grad check for permute.
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            # [FUNCTION DOC] HELPER: Permutes dims to order two, zero, one.
            return x.permute(2, 0, 1)

        self._batched_grad_test(op, (x,))

    def test_reshape(self, device):
        # [FUNCTION DOC] Batched first-order grad check for reshape to a 2-d shape.
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            # [FUNCTION DOC] HELPER: Reshapes its input to six by five.
            return x.reshape([2 * 3, 5])

        self._batched_grad_test(op, (x,))

    def test_sigmoid(self, device):
        # [FUNCTION DOC] Batched first and second order grad check for sigmoid.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.sigmoid, (x,))
        self._batched_grad_grad_test(Tensor.sigmoid, (x,))

    def test_stack(self, device):
        # [FUNCTION DOC] Batched first-order grad check for stacking two tensors.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        y = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x, y):
            # [FUNCTION DOC] HELPER: Stacks the two inputs along a new first dim.
            return torch.stack([x, y])

        self._batched_grad_test(op, (x, y))

    def test_select(self, device):
        # [FUNCTION DOC] Batched first-order grad check for select via indexing and the select
        # method, including negative dim.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[1], (x,))
        self._batched_grad_test(lambda x: x.select(1, 2), (x,))
        self._batched_grad_test(lambda x: x.select(-1, 0), (x,))

    def test_slice(self, device):
        # [FUNCTION DOC] Batched first-order grad check for slicing in several forms.
        x = torch.randn(2, 3, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[0:1], (x,))
        self._batched_grad_test(lambda x: x[:, 1:3], (x,))
        self._batched_grad_test(lambda x: x[..., 1:3], (x,))

    def test_trace(self, device):
        # [FUNCTION DOC] Batched first-order grad check for trace, and that vmap of grad of a
        # function summing the gradient of trace (a second-order quantity) is all zeros.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(Tensor.trace, (x,))

        x = torch.randn(3, 2, 2, device=device)

        def sum_grad_trace(x):
            # [FUNCTION DOC] HELPER: Sums the functorch gradient of trace of its input.
            return grad(torch.trace)(x).sum()

        output = vmap(grad(sum_grad_trace))(x)
        self.assertEqual(output, torch.zeros_like(output))

    def test_where(self, device):
        # [FUNCTION DOC] Checks vmap of where with a condition tensor runs without error, and
        # that the single-argument where form raises a RuntimeError saying it cannot be vmapped.
        x = torch.randn(3, 2, device=device)
        y = torch.ones(3, 2, device=device)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Three-argument where selecting x where it is positive,
            # else y.
            return torch.where(x > 0, x, y)

        # Check that there is no runtime error, exactness tests are done with opinfo
        vmap(f)(x, y)

        x = torch.randint(0, 2, size=(4, 3), dtype=torch.float)

        def f(t):
            # [FUNCTION DOC] HELPER: Single-argument where (returns indices of non-zero
            # entries), whose output size is data dependent.
            return torch.where(t)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to vmap over aten::where"
        ):
            vmap(f)(x)

    def test_threshold(self, device):
        # [FUNCTION DOC] Batched first-order grad check for threshold with threshold 0.5 and
        # value 0.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: F.threshold(x, 0.5, 0.0), (x,))

    @parametrize("backend", PLATFORM_SPECIFIC_SDPA)
    def test_sdpa(self, device, backend):
        # [FUNCTION DOC] Parametrized over the available SDPA backends. Skipped on cpu. Checks
        # vmap of scaled_dot_product_attention for float16 inputs with various batched/unbatched
        # combinations of query, key, value, and a case with differing batch dims. The backward
        # test is disabled.
        if device == "cpu":
            raise unittest.SkipTest("This test is only for CUDA for now")

        def T(*args):
            # [FUNCTION DOC] HELPER: Random float16 tensor on the test device with the
            # given shape.
            return torch.randn(*args, dtype=torch.float16, device=device)

        backend_ctx = sdpa_kernel([backend])
        with backend_ctx:
            for batching in [
                (True, True, True),
                (True, False, False),
                (False, True, True),
            ]:
                size = [8, 4, 128, 64]
                if batching[0]:
                    query = T(3, *size)
                else:
                    query = T(*size)
                if batching[1]:
                    key = T(3, *size)
                else:
                    key = T(*size)
                if batching[2]:
                    value = T(3, *size)
                else:
                    value = T(*size)
                in_dims = tuple(0 if b else None for b in batching)
                attention = F.scaled_dot_product_attention

                self._vmap_test(
                    attention,
                    (query, key, value),
                    in_dims=in_dims,
                )
                # Backwards test doesn't work yet
                # self._batched_grad_test(
                #     lambda query, key, value: F.scaled_dot_product_attention(
                #         query, key, value
                #     ),
                #     (query, key, value),
                # )

            B = 4
            query = torch.rand(4, 32, B, 8, 128, dtype=torch.float16, device=device)
            key = torch.rand(4, B, 32, 8, 128, dtype=torch.float16, device=device)
            value = torch.rand(4, 32, 8, 128, dtype=torch.float16, device=device)
            self._vmap_test(
                F.scaled_dot_product_attention,
                (query, key, value),
                in_dims=(2, 1, None),
            )

    @parametrize("backend", PLATFORM_SPECIFIC_SDPA)
    def test_sdpa_unbatched_inputs(self, device, backend):
        # [FUNCTION DOC] Parametrized over SDPA backends. Skipped on cpu. Checks forward
        # results, backward gradients and functorch grad results of vmap over
        # scaled_dot_product_attention equal the direct computation on stacked inputs, under the
        # chosen backend.
        if device == "cpu":
            raise unittest.SkipTest("This test is only for CUDA for now")

        query = torch.randn(
            3, 4, 32, 64, dtype=torch.float16, device=device, requires_grad=True
        )
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn_like(query, requires_grad=True)

        def loss(q, k, v):
            # [FUNCTION DOC] HELPER: Sum of the scaled dot product attention output.
            return F.scaled_dot_product_attention(q, k, v).sum()

        loss_grad = grad(loss, argnums=(0, 1, 2))
        with sdpa_kernel([backend]):
            expected = F.scaled_dot_product_attention(query, key, value)
            actual = vmap(F.scaled_dot_product_attention)(query, key, value)
            grad_output = torch.randn_like(expected)
            expected_backward_grads = torch.autograd.grad(
                expected, (query, key, value), grad_output
            )
            actual_backward_grads = torch.autograd.grad(
                actual, (query, key, value), grad_output
            )
            expected_grads = loss_grad(query, key, value)
            actual_grads = vmap(loss_grad)(query, key, value)

        self.assertEqual(actual, expected)
        self.assertEqual(actual_backward_grads, expected_backward_grads)
        self.assertEqual(actual_grads, expected_grads)

    def test_sdpa_unbatched_inputs_cpu(self, device):
        # [FUNCTION DOC] CPU only (skipped on other devices). Same checks as the
        # unbatched-inputs test but using the flash attention backend on cpu, ignoring the
        # fallback performance warning.
        if device != "cpu":
            raise unittest.SkipTest("This test is only for CPU")

        query = torch.randn(3, 4, 32, 64, device=device, requires_grad=True)
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn_like(query, requires_grad=True)

        def loss(q, k, v):
            # [FUNCTION DOC] HELPER: Sum of the scaled dot product attention output.
            return F.scaled_dot_product_attention(q, k, v).sum()

        loss_grad = grad(loss, argnums=(0, 1, 2))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=FALLBACK_REGEX)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                expected = F.scaled_dot_product_attention(query, key, value)
                actual = vmap(F.scaled_dot_product_attention)(query, key, value)
                grad_output = torch.randn_like(expected)
                expected_backward_grads = torch.autograd.grad(
                    expected, (query, key, value), grad_output
                )
                actual_backward_grads = torch.autograd.grad(
                    actual, (query, key, value), grad_output
                )
                expected_grads = loss_grad(query, key, value)
                actual_grads = vmap(loss_grad)(query, key, value)

        self.assertEqual(actual, expected)
        self.assertEqual(actual_backward_grads, expected_backward_grads)
        self.assertEqual(actual_grads, expected_grads)

    @parametrize(
        "backend",
        [
            backend
            for backend in PLATFORM_SPECIFIC_SDPA
            if backend != SDPBackend.FLASH_ATTENTION
        ],
    )
    def test_sdpa_unbatched_inputs_with_mask(self, device, backend):
        # [FUNCTION DOC] Parametrized over SDPA backends other than flash attention. Skipped on
        # cpu. Checks vmap of scaled_dot_product_attention with a mask where the mask is also
        # mapped matches the direct call on the full-batched inputs.
        if device == "cpu":
            raise unittest.SkipTest("This test is only for CUDA for now")

        query = torch.randn(3, 4, 32, 64, dtype=torch.float16, device=device)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        attn_mask = torch.randn(3, 1, 32, 32, dtype=torch.float16, device=device)

        with sdpa_kernel([backend]):
            expected = F.scaled_dot_product_attention(query, key, value, attn_mask)
            actual = vmap(F.scaled_dot_product_attention)(
                query, key, value, attn_mask[:, 0]
            )

        self.assertEqual(actual, expected)

    @parametrize("backend", PLATFORM_SPECIFIC_SDPA)
    @parametrize("randomness", ["error", "same", "different"])
    def test_randomness(self, device, randomness, backend):
        # [FUNCTION DOC] Parametrized over backend and randomness mode. Skipped on cpu (and on
        # cuDNN 9.10 to 9.13 for the cuDNN backend with randomness different). Checks that
        # attention with zero dropout works in every mode, while nonzero dropout raises a
        # RuntimeError for mode error, and also for mode same except with the math backend, and
        # passes for different.
        if device == "cpu":
            raise unittest.SkipTest("This test is only for CUDA for now")

        # xfail for cuDNN version between 9.10 and 9.13
        if backend == SDPBackend.CUDNN_ATTENTION and randomness == "different":
            if 91100 <= TEST_CUDNN_VERSION <= 91300:
                raise unittest.SkipTest(
                    "xfail on cuDNN 9.10-9.13 with CUDNN backend and randomness='different'"
                )

        backend_ctx = sdpa_kernel([backend])
        with backend_ctx:
            B = 4
            query = torch.rand(B, 4, 32, 8, 128, dtype=torch.float16, device=device)
            key = torch.rand(B, 4, 32, 8, 128, dtype=torch.float16, device=device)
            value = torch.rand(B, 4, 32, 8, 128, dtype=torch.float16, device=device)

            def f(q, k, v, dropout):
                # [FUNCTION DOC] HELPER: Scaled dot product attention with the given
                # dropout probability.
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)

            # No matter the randomness mode, dropout=0.0 should pass
            vmap(
                functools.partial(f, dropout=0.0),
                in_dims=(0, 0, 0),
                randomness=randomness,
            )(query, key, value)

            fail_with_randomness = randomness == "error"
            if backend != SDPBackend.MATH:
                fail_with_randomness |= randomness == "same"
            context = (
                self.assertRaises(RuntimeError)
                # We currently don't support randomness == "same", and "error" should always error with randomness
                if fail_with_randomness
                else contextlib.nullcontext()
            )
            with context:
                vmap(
                    functools.partial(f, dropout=0.5),
                    in_dims=(0, 0, 0),
                    randomness=randomness,
                )(query, key, value)

    @allowVmapFallbackUsage
    def test_inplace_view(self, device):
        # [FUNCTION DOC] Batched first and second order grad check of a function that applies an
        # in-place cos to a view of a squared leaf. Fallback allowed.
        leaf = torch.randn(4, 5, requires_grad=True)

        def func(leaf):
            # [FUNCTION DOC] HELPER: Squares the leaf, takes its first row as a view, applies
            # cos in place on the view and returns it.
            # Make sure the function is non-trivially twice differentiable
            base = leaf * leaf
            view = base[0]
            view.cos_()
            return view

        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    @allowVmapFallbackUsage
    def test_inplace_manyview(self, device):
        # [FUNCTION DOC] Batched first and second order grad check with a chain of views
        # (transpose, select, diagonal, strided slice) of a squared leaf modified in place by
        # cos. Fallback allowed.
        leaf = torch.randn(4, 4, 5, requires_grad=True)

        def func(leaf):
            # [FUNCTION DOC] HELPER: Builds a chain of views of the squared leaf, applies cos in
            # place, returns it.
            # Make sure the function is non-trivially twice differentiable
            base = leaf * leaf
            view = base.transpose(0, 2)
            view = view[1]
            view = view.diagonal()
            view = view[::2]
            view.cos_()
            return view

        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    def test_diagonal(self, device):
        # [FUNCTION DOC] Batched first-order grad check for diagonal with offset and
        # negative dims.
        x = torch.randn(4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(1, 0, 1), (x,))

        x = torch.randn(3, 4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(0, -1, -2), (x,))

    @allowVmapFallbackUsage
    def test_unrelated_output(self, device):
        # [FUNCTION DOC] Checks vmap of a vector-Jacobian function whose output is unrelated to
        # the input yields zeros for each batch entry. Fallback allowed.
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            # [FUNCTION DOC] HELPER: Gradient of y with respect to x for vector v, or zeros if y
            # does not depend on x.
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))

    @allowVmapFallbackUsage
    def test_unrelated_output_multiple_grad(self, device):
        # [FUNCTION DOC] Same as the unrelated output test but first calls the vector-Jacobian
        # function once outside vmap on a single batch entry, to check repeated gradient calls
        # still work. Fallback allowed.
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            # [FUNCTION DOC] HELPER: Gradient of y with respect to x for vector v, or zeros if y
            # does not depend on x.
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        _ = vjp(gy[0])
        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))


def discover_variants(opinfo):
    # [FUNCTION DOC] HELPER: Given an OpInfo, returns a pair of lists: all out-of-place
    # callables (the op and each alias) and all in-place variants (of the op and its aliases)
    # that exist.
    aliases = []
    inplace_variants = []

    if opinfo.inplace_variant:
        inplace_variants.append(opinfo.inplace_variant)

    aliases.append(opinfo.op)
    for alias in opinfo.aliases:
        aliases.append(alias.op)
        if alias.inplace_variant:
            inplace_variants.append(alias.inplace_variant)
    return aliases, inplace_variants


# TODO: enable this when we get a bit closer to getting torch.vmap x torch.compile working.
# @markDynamoStrictTest
@unMarkDynamoStrictTest
class TestVmapOperatorsOpInfo(TestCase):
    # [CLASS DOC] Device and dtype generic vmap tests driven by the OpInfo database: each op is
    # run under vmap with many generated batching layouts and compared with a loop. Holds the
    # lists of expected failures and skips. The dynamo strict marking is intentionally disabled.
    def vmap_outplace_test(
        self,
        func,
        args,
        kwargs,
        in_dims,
        check_shape_only=False,
        postprocess_fn=None,
        out_dim=0,
    ):
        # [FUNCTION DOC] HELPER: For each (vmap, loop) result pair produced by the shared helper
        # for the function, optionally postprocesses both and asserts equality (or just equal
        # shapes when check_shape_only).
        for vmap_out, loop_out in compute_quantities_for_vmap_test(
            func, args, kwargs, in_dims, out_dim=out_dim
        ):
            if postprocess_fn is not None:
                loop_out = postprocess_fn(loop_out)
                vmap_out = postprocess_fn(vmap_out)
            if check_shape_only:
                self.assertEqual(vmap_out.shape, loop_out.shape)
                continue
            self.assertEqual(vmap_out, loop_out)

    def vmap_inplace_test(
        self, func, args, kwargs, in_dims, postprocess_fn=None, out_dim=0
    ):
        # [FUNCTION DOC] HELPER: As for the out-of-place test but for in-place ops, using
        # cloned inputs. Assumes the first argument is modified. If its in_dim is None,
        # asserts a RuntimeError is raised instead since vmap of an in-place op on an
        # unbatched tensor is impossible.
        # NB: This test assumes that the first argument is being modified.
        # This is OK because it's what every other OpInfo-based test assumes,
        # but it is going to need a more robust solution eventually.
        if in_dims[0] is None:
            # Check that we correctly raise an error when vmap is impossible
            # on the in-place operation
            with self.assertRaises(RuntimeError):
                for _ in compute_quantities_for_vmap_test(
                    func,
                    args,
                    kwargs,
                    in_dims,
                    out_dim=out_dim,
                    compute_loop_out=False,
                    clone_inputs=True,
                ):
                    pass
            return
        for vmap_out, loop_out in compute_quantities_for_vmap_test(
            func,
            args,
            kwargs,
            in_dims,
            clone_inputs=True,
            out_dim=out_dim,
        ):
            if postprocess_fn is not None:
                loop_out = postprocess_fn(loop_out)
                vmap_out = postprocess_fn(vmap_out)
            self.assertEqual(vmap_out, loop_out)

    def opinfo_vmap_test(
        self,
        device,
        dtype,
        op,
        check_has_batch_rule,
        skip_inplace=(),
        postprocess_fn=None,
    ):
        # [FUNCTION DOC] HELPER: Runs the OpInfo vmap check for an op and dtype on a device.
        # Defines an inner test that (1) vmaps each error input and expects an exception, (2)
        # iterates sample (or reference, for most ops) inputs, generating batching layouts and
        # checking out-of-place aliases and valid in-place variants against the loop. If
        # check_has_batch_rule, runs the inner test under the fallback-use check; otherwise runs
        # it directly. skip_inplace lists op names to skip in-place.
        def test():
            # [FUNCTION DOC] HELPER: Body of the OpInfo check: error-input checks, then
            # sample-input checks for all aliases and in-place variants. Uses the enclosing op,
            # device, dtype and options.
            # Error inputs check
            if op.error_inputs_func is not None:
                error_inputs = op.error_inputs(device)
                for error_input in error_inputs:
                    sample_input = error_input.sample_input
                    args = (sample_input.input,) + tuple(sample_input.args)
                    kwargs = sample_input.kwargs
                    for batched_args, in_dims, _ in generate_vmap_inputs(args, {}):
                        with self.assertRaises(Exception):
                            vmap(op, in_dims)(*batched_args, **kwargs)

            # Sample inputs check
            sample_inputs_op = {
                # Take too long with reference inputs
                "special.chebyshev_polynomial_t",
                "special.chebyshev_polynomial_u",
                "special.chebyshev_polynomial_v",
                "special.chebyshev_polynomial_w",
                "special.hermite_polynomial_he",
                "special.laguerre_polynomial_l",
                "special.legendre_polynomial_p",
                "special.shifted_chebyshev_polynomial_t",
                "special.shifted_chebyshev_polynomial_u",
                "special.shifted_chebyshev_polynomial_v",
                "special.shifted_chebyshev_polynomial_w",
            }
            if op.name in sample_inputs_op:
                sample_inputs_itr = op.sample_inputs(
                    device, dtype, requires_grad=False, use_subtests=True
                )
            else:
                sample_inputs_itr = op.reference_inputs(
                    device, dtype, requires_grad=False, use_subtests=True
                )
            aliases, inplace_aliases = discover_variants(op)
            check_shape_only = op.name in ("empty_like", "new_empty")
            for sample_input, subtest_ctx, skip_xfail_ctx in sample_inputs_itr:
                with subtest_ctx(self), skip_xfail_ctx(self):
                    args = (sample_input.input,) + sample_input.args
                    if not any(isinstance(arg, torch.Tensor) for arg in args):
                        # At least one tensor required for vmap.
                        continue
                    kwargs = sample_input.kwargs
                    is_batch_norm_and_training = is_batch_norm_training(op.name, kwargs)
                    out_dim = 0
                    if op.name == "NumpySplitCopyWithIntCustomOp":
                        # special case for this custom op
                        def sample_vmap_out_dim_numpy_split_copy_with_int(
                            x, splits, dim
                        ):
                            # [FUNCTION DOC] HELPER: For the custom numpy split op, returns an
                            # output-dim list of zeros, one per split output (splits count plus
                            # one), and None; the inputs themselves are ignored.
                            return [0 for _ in range(len(splits) + 1)], None

                        out_dim = sample_vmap_out_dim_numpy_split_copy_with_int(*args)
                    for batched_args, in_dims, _ in generate_vmap_inputs(
                        args, {}, is_batch_norm_and_training=is_batch_norm_and_training
                    ):
                        for func in aliases:
                            self.vmap_outplace_test(
                                func,
                                batched_args,
                                kwargs,
                                in_dims,
                                check_shape_only,
                                postprocess_fn,
                                out_dim=out_dim,
                            )
                        if op.name in skip_inplace:
                            continue
                        if not is_valid_inplace_sample_input(
                            sample_input, op, op.inplace_variant
                        ):
                            continue
                        for func in inplace_aliases:
                            self.vmap_inplace_test(
                                func, batched_args, kwargs, in_dims, postprocess_fn
                            )

        if check_has_batch_rule:
            check_vmap_fallback(self, test, op)
        else:
            test()

    vmap_fail = {
        # -------------------- ALLOWED FAILURES --------------------------------
        # These are things that we either cannot fix or are not actually problems
        xfail("resize_"),
        xfail("resize_as_"),
        xfail("to_sparse"),
        xfail("__getitem__"),  # dynamic mask
        xfail("index_put"),  # dynamic mask
        xfail(
            "nn.functional.dropout"
        ),  # works, can't check against for loop because of randomness inconsistency
        xfail("nn.functional.scaled_dot_product_attention"),  # randomness
        xfail("nn.functional.multi_head_attention_forward"),  # randomness
        xfail("masked_select"),  # dynamic op
        xfail("nonzero"),  # dynamic op
        xfail("unique", ""),  # dynamic op
        xfail("unique_consecutive", ""),  # dynamic op
        xfail("allclose"),  # returns a boolean
        xfail("uniform"),  # randomness is tested separately
        xfail("rand_like"),  # randomness is tested separately
        xfail("randint_like"),  # randomness is tested separately
        xfail("randn_like"),  # randomness is tested separately
        xfail("bernoulli", ""),  # randomness is tested separately
        xfail("normal", ""),  # randomness is tested separately
        xfail("normal", "number_mean"),  # randomness is tested separately
        xfail("multinomial", ""),  # randomness
        xfail("nn.functional.embedding", ""),  # we only support some cases
        xfail("nn.functional.rrelu"),  # randomness
        xfail("nn.functional.dropout2d", ""),  # randomness
        xfail("nn.functional.dropout3d", ""),  # randomness
        xfail("nn.functional.alpha_dropout", ""),  # randomness
        xfail("nn.functional.feature_alpha_dropout", "with_train"),  # randomness
        xfail("as_strided"),  # Our test runner can't handle this; manual test exists
        xfail("as_strided_copy"),
        xfail(
            "as_strided_scatter"
        ),  # no batching rule implemented, default doesn't work
        skip(
            "new_empty_strided"
        ),  # empty tensor data is garbage so it's hard to make comparisons with it
        xfail("nn.functional.fractional_max_pool3d"),  # randomness
        xfail("nn.functional.fractional_max_pool2d"),  # randomness
        xfail("pca_lowrank", ""),  # random operation
        xfail("svd_lowrank", ""),  # random operation
        xfail("sparse.sampled_addmm"),  # sparse
        xfail("sparse.mm", "reduce"),  # sparse
        xfail(
            "NumpyCubeNotComposableAutogradFunction"
        ),  # Not composable autograd.Function
        skip("_softmax_backward_data"),
        skip(
            "linalg.eigh", ""
        ),  # not always return the same result for the same input, see test_linalg_eigh for manual test
        # UnimplementedError: data-dependent operators cannot be vmapped
        xfail("NumpyNonzeroCustomOp"),
        xfail("NumpyNMSCustomOp"),
        # ----------------------------------------------------------------------
        # ---------------------------- BUGS ------------------------------------
        # entries in here don't work and need to be fixed.
        # Each one of these is a bug
        decorate("frexp", decorator=skipIfTorchDynamo()),
        xfail("clamp_min", ""),  # Exception not raised on error input
        xfail("clamp_max", ""),  # Exception not raised on error input
        xfail(
            "view_as_complex"
        ),  # RuntimeError: Tensor must have a last dimension with stride 1
        xfail("tensor_split"),  # data_ptr
        xfail(
            "histogramdd"
        ),  # expected Tensor as element 0 in argument 0, but got tuple
        xfail("nn.functional.gaussian_nll_loss"),  # data-dependent control flow error
        xfail(
            "nn.functional.embedding_bag"
        ),  # embedding renorm vmap inplace incompatible
        xfail("narrow"),  # Batching rule not implemented for aten::narrow.Tensor
        # required rank 4 tensor to use channels_last format
        xfail("bfloat16"),
        xfail("bool"),
        xfail("byte"),
        xfail("char"),
        xfail("double"),
        xfail("float"),
        xfail("half"),
        xfail("int"),
        xfail("long"),
        xfail("short"),
        xfail("cdouble"),
        xfail("cfloat"),
        xfail(
            "jiterator_binary", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "jiterator_binary_return_by_ref", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "jiterator_4inputs_with_extra_args", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "equal", ""
        ),  # TypeError: object of type 'bool' has no len(); likely testrunner problem
        xfail(
            "jiterator_unary", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "jiterator_2inputs_2outputs", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        # ---------------------------------------------------------------------
        # TypeError: expected Tensor as element 0 in argument 0, but got NotImplementedType
        xfail("__rsub__"),
        # RuntimeError: Batching rule not implemented for aten::moveaxis.int;
        # the fallback path doesn't work on out= or view ops.
        xfail("movedim"),
        # RuntimeError: NYI: querying is_contiguous inside of vmap for
        # memory_format other than torch.contiguous_format
        xfail("contiguous"),
        # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only supported
        # with memory_format torch.preserve_format or torch.contiguous_format (got ChannelsLast)
        xfail("clone"),
        # RuntimeError: When vmap-ing torch.nn.functional.one_hot,
        # please provide an explicit positive num_classes argument.
        xfail("nn.functional.one_hot"),
        # RuntimeError: aten::_flash_attention_forward hit the vmap fallback which is currently disabled
        xfail("torch.ops.aten._flash_attention_forward"),
    }

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(
        op_db + additional_op_db + autograd_function_db + custom_op_db,
        dtypes=OpDTypes.any_one,
    )
    @opsToleranceOverride(
        "TestVmapOperatorsOpInfo",
        "test_vmap_exhaustive",
        (
            tol1(
                "linalg.det",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            # The following is often flaky, but just on windows.
            # We should investigate if it's actually a problem or not.
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=1e-04, rtol=1e-02)},
                device_type="cuda",
            ),
        ),
    )
    @toleranceOverride(
        {
            torch.float32: tol(atol=1e-04, rtol=1e-04),
            torch.complex64: tol(atol=1e-04, rtol=1e-04),
        }
    )
    @skipOps(
        vmap_fail.union(
            {
                # RuntimeError: Batch norm got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("tril"),  # Exception not raised on error input
                xfail("triu"),  # Exception not raised on error input
                xfail("as_strided", "partial_views"),
                # RuntimeError: output with shape [4, 4] doesn't match the broadcast shape [1, 4, 4]
                xfail("addcdiv"),
                xfail("addcmul"),
                xfail("clamp"),
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                # TypeError: expected Tensor as element 0 in argument 0, but got float
                xfail("item"),
                # RuntimeError: required rank 4 tensor to use channels_last format
                xfailIf(
                    "to",
                    lambda sample: (
                        sample.kwargs["memory_format"] == torch.channels_last
                    ),
                ),
                # https://github.com/pytorch/pytorch/issues/164556
                skipIf("cholesky_solve", lambda *args: TEST_WITH_ROCM),
            }
        ),
    )
    def test_vmap_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] Main OpInfo-driven exhaustive vmap test, instantiated per op, device
        # and one dtype (tf32 disabled, per-op tolerance overrides, and the known failure and
        # skip table assembled from the vmap_fail set plus extra entries such as batch norm and
        # tril). Runs the shared check with in-place checks enabled for all ops and no
        # requirement that a batching rule exists.
        # needs to be fixed
        inplace_failure_list = ()
        self.opinfo_vmap_test(
            device,
            dtype,
            op,
            check_has_batch_rule=False,
            skip_inplace=inplace_failure_list,
        )

    @with_tf32_off
    @ops(
        op_db + additional_op_db + autograd_function_db + custom_op_db,
        dtypes=OpDTypes.any_one,
    )
    @opsToleranceOverride(
        "TestVmapOperatorsOpInfo",
        "test_op_has_batch_rule",
        (
            tol1(
                "linalg.det",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cuda",
            ),
        ),
    )
    @toleranceOverride(
        {
            torch.float32: tol(atol=1e-04, rtol=1e-04),
            torch.complex64: tol(atol=1e-04, rtol=1e-04),
        }
    )
    @skipOps(
        vmap_fail.union(
            {
                xfail("as_strided", "partial_views"),
                skip(
                    "to"
                ),  # RuntimeError: required rank 4 tensor to use channels_last format
                # Batch norm got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("histogram"),
                # `index_put` OpInfo in pytorch/pytorch has
                # masked index as input which is not supported
                xfail("index_put", ""),
                xfail("isin"),
                xfail("masked_scatter"),
                xfail("masked_select"),
                xfail("nanquantile"),
                xfail("ormqr"),
                xfail("put"),
                xfail("quantile"),
                xfail("renorm"),
                xfail("resize_as_"),
                xfail("take"),
                xfail("tensor_split"),
                xfail("to_sparse"),
                # TypeError: expected Tensor as element 0 in argument 0, but got float
                xfail("item"),
                xfail("tril"),  # Exception not raised on error input
                xfail("triu"),  # Exception not raised on error input
                xfail("__getitem__", ""),
                xfail(
                    "nn.functional.dropout"
                ),  # works, can't check against for loop because of randomness inconsistency
                xfail("nn.functional.scaled_dot_product_attention"),  # randomness
                xfail("nn.functional.multi_head_attention_forward"),  # randomness
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                xfail("resize_"),
                xfail("view_as_complex"),
                xfail("fft.ihfft2"),
                xfail("fft.ihfftn"),
                xfail("allclose"),
                xfail("argwhere"),
                xfail("unique_consecutive"),
                xfail("unique"),
                xfail("nn.functional.ctc_loss"),
                xfail("nn.functional.gaussian_nll_loss"),
                xfail("histc"),
                xfail("as_strided"),
                xfail("as_strided_copy"),
                xfail("istft"),
                xfail("nonzero"),
                xfail("nn.functional.fractional_max_pool2d"),
                xfail("stft"),
                xfail("isclose"),
                xfail("nn.functional.fractional_max_pool3d"),
                xfail("nn.functional.bilinear"),
                xfail("nn.functional.embedding_bag"),
                xfail("linalg.tensorsolve"),
                xfail("bernoulli", ""),
                xfail("nn.functional.feature_alpha_dropout", "with_train"),
                xfail("nn.functional.kl_div", ""),
                xfail("multinomial", ""),
                xfail("pca_lowrank", ""),
                xfail("normal", ""),
                xfail("nn.functional.dropout2d", ""),
                xfail("normal", "number_mean"),
                xfail("svd_lowrank", ""),
                xfail("diagflat", ""),
                xfail("special.log_ndtr"),
                xfail(
                    "narrow"
                ),  # Batching rule not implemented for aten::narrow.Tensor
                xfail("nn.functional.triplet_margin_loss", ""),
                xfail("nn.functional.pdist", ""),
                xfail("nn.functional.max_unpool1d", "grad"),
                xfail("nn.functional.multi_margin_loss", ""),
                xfail("nn.functional.multilabel_margin_loss", ""),
                xfail("nn.functional.max_unpool3d", "grad"),
                xfail("nn.functional.max_unpool2d", ""),
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail("nn.functional.margin_ranking_loss", ""),
                xfail("nn.functional.max_unpool1d", ""),
                xfail("nn.functional.soft_margin_loss", ""),
                xfail("nn.functional.max_unpool3d", ""),
                xfail("linalg.ldl_solve", "", device_type="cpu"),
                xfail("chalf", ""),
                xfail("clamp_max", ""),
                xfail("jiterator_binary_return_by_ref", device_type="cuda"),
                xfail("jiterator_unary", device_type="cuda"),
                xfail("jiterator_2inputs_2outputs", device_type="cuda"),
                xfail("special.airy_ai"),
                xfail("clamp_min", ""),
                xfail("sparse.sampled_addmm"),
                xfail("sparse.mm", "reduce"),
                xfail("special.chebyshev_polynomial_t"),
                xfail("special.chebyshev_polynomial_v"),
                xfail("special.chebyshev_polynomial_u"),
                xfail("special.chebyshev_polynomial_w"),
                xfail("special.shifted_chebyshev_polynomial_t"),
                xfail("special.shifted_chebyshev_polynomial_v"),
                xfail("special.shifted_chebyshev_polynomial_u"),
                xfail("special.shifted_chebyshev_polynomial_w"),
                xfail("_segment_reduce", "offsets"),
                xfail("index_reduce", "prod"),
                xfail("index_reduce", "mean"),
                xfail("index_reduce", "amin"),
                xfail("index_reduce", "amax"),
                xfail("special.laguerre_polynomial_l"),
                xfail("special.legendre_polynomial_p"),
                xfail("special.hermite_polynomial_h"),
                xfail("jiterator_binary", device_type="cuda"),
                xfail("jiterator_4inputs_with_extra_args", device_type="cuda"),
                xfail("_segment_reduce", "lengths"),
                xfail("lu_solve", ""),
                xfail("special.hermite_polynomial_he"),
                xfail("nn.functional.dropout3d", ""),
                xfail("as_strided_scatter", ""),
                xfail("equal", ""),
                xfail("linalg.lu", ""),
                skip("linalg.ldl_solve", ""),
                skip("_softmax_backward_data"),
                # One or more of the overload doesn't have a Batch rule.
                xfail("bincount"),
                xfail("torch.ops.aten._scaled_dot_product_flash_attention_for_cpu"),
            }
        ),
    )
    def test_op_has_batch_rule(self, device, dtype, op):
        # [FUNCTION DOC] OpInfo-driven test like the exhaustive one but each run is wrapped in
        # the check that the vmap fallback is not used, so every op listed must have a batching
        # rule. Has a larger expected-failure table for ops lacking rules, and skips in-place
        # checking for a list of ops.
        # needs to be fixed
        inplace_failures = (
            "addbmm",
            "addcdiv",
            "addcmul",
            "addmm",
            "addmv",
            "addr",
            "baddbmm",
            "clamp",
            "conj_physical",
            "cumprod",
            "cumsum",
            "floor_divide",
            "fmod",
            "heaviside",
            "hypot",
            "igamma",
            "igammac",
            "index_copy",
            "ldexp",
            "lerp",
            "neg",
            "nextafter",
            "polygamma",
            "pow",
            "remainder",
            "scatter",
            "square",
            "sub",
            "trunc",
            "xlogy",
        )
        self.opinfo_vmap_test(
            device, dtype, op, check_has_batch_rule=True, skip_inplace=inplace_failures
        )

    def test_linalg_svd(self, device):
        # [FUNCTION DOC] Checks vmap of linalg.svd through reconstruction: since the
        # factorization is not unique, the U, S, Vh outputs are recombined into the matrix and
        # compared with the loop result. Uses every OpInfo named linalg.svd with float dtype and
        # the no-fallback check.
        # linalg_svd returns a tuple of three tensors, (U, S, Vh).
        # Given the same input, it may return different tensors,
        # because svd isn't unique. To test that the svd is correct, we multiply
        # U @ diag(S) @ Vh and check that the output from vmap matches the
        # output from a for-loop.
        def compute_A(out):
            # [FUNCTION DOC] HELPER: Rebuilds the original matrices from SVD output (U times a
            # diagonal-padded S times Vh).
            U, S, Vh = out
            m = U.shape[-1]
            n = Vh.shape[-2]
            diag_S = S.new_zeros(*S.shape[:-1], m, n)
            diag_S.diagonal(offset=0, dim1=-2, dim2=-1).copy_(S)
            return U @ diag_S @ Vh

        opinfos = [op for op in op_db if op.name == "linalg.svd"]
        if len(opinfos) == 0:
            raise AssertionError("Expected at least one opinfo for linalg.svd")

        for op in opinfos:
            self.opinfo_vmap_test(
                device,
                torch.float,
                op,
                check_has_batch_rule=True,
                postprocess_fn=compute_A,
            )

    def test_linalg_eigh(self, device):
        # [FUNCTION DOC] Checks vmap of linalg.eigh through reconstruction (Q times diagonal L
        # times Q conjugate transpose) since eigenvectors are not unique. Uses every OpInfo
        # named linalg.eigh with float dtype and the no-fallback check; raises AssertionError if
        # none is found.
        # linalg_svd returns two tensors, (Q, L).
        # Given the same input, it may return different tensors,
        # because the eig decomposition isn't unique.
        # To test that eigh is correct, we multiply
        # Q @ diag(L) @ Qh and check that the output from vmap matches the
        # output from a for-loop.
        def compute_A(out):
            # [FUNCTION DOC] HELPER: Rebuilds the matrices from eigh output given as eigenvalues
            # and vectors.
            L, Q = out
            n = Q.shape[-1]
            diag_L = L.new_zeros(*L.shape[:-1], n, n)
            diag_L.diagonal(offset=0, dim1=-2, dim2=-1).copy_(L)
            Qh = Q.transpose(-2, -1).conj()
            return Q @ diag_L @ Qh

        opinfos = [op for op in op_db if op.name == "linalg.eigh"]
        if len(opinfos) == 0:
            raise AssertionError("Expected at least one opinfo for linalg.eigh")

        for op in opinfos:
            self.opinfo_vmap_test(
                device,
                torch.float,
                op,
                check_has_batch_rule=True,
                postprocess_fn=compute_A,
            )

    @skipIfTorchDynamo()
    def test_slogdet(self, device):
        # [FUNCTION DOC] Checks vmap of slogdet on a batch of matrices against the loop (no
        # OpInfo exists), wrapped in the fallback-use check. Skipped under dynamo.
        # There's no OpInfo for this
        def test():
            # [FUNCTION DOC] HELPER: Runs the out-of-place vmap comparison for slogdet on a
            # batch of two 5x5 matrices.
            B = 2
            x = torch.randn(B, 5, 5, device=device)
            self.vmap_outplace_test(torch.slogdet, (x,), {}, (0,))

        check_vmap_fallback(self, test, torch.slogdet)

    def test_index_fill(self, device):
        # [FUNCTION DOC] Checks vmap of index_fill (no OpInfo coverage) in eight scenarios with
        # different batched/unbatched self, index, and value (Python float or 0-dim tensor),
        # each wrapped in the fallback-use check.
        # There's no OpInfo for these tests

        B = 2

        def test1():
            # [FUNCTION DOC] HELPER: Scenario with a negative dim and only the index batched.
            # negative dim
            x = torch.randn(B, 5, 5, device=device)
            dim = -2
            index = torch.tensor([[2, 3], [0, 4]], device=device)
            value = 5.0
            self.vmap_outplace_test(
                torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
            )

        def test2():
            # [FUNCTION DOC] HELPER: Scenario with self and index batched, self rank one, index
            # rank one.
            # self batched, self logical rank 1, index logical rank 1
            x = torch.zeros(B, 3, device=device)
            dim = 0
            index = torch.tensor([[0], [1]], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (0, None, 0, None)
                )

        def test3():
            # [FUNCTION DOC] HELPER: Scenario with self and index batched, self rank one, index
            # rank zero.
            # self batched, self logical rank 1, index logical rank 0
            x = torch.zeros(B, 3, device=device)
            dim = 0
            index = torch.tensor([0, 1], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (0, None, 0, None)
                )

        def test4():
            # [FUNCTION DOC] HELPER: Scenario with scalar self unbatched and index batched
            # (rank one).
            # self not batched, self logical rank 0, index logical rank 1
            x = torch.zeros([], device=device)
            dim = 0
            index = torch.tensor([[0], [0]], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test5():
            # [FUNCTION DOC] HELPER: Scenario with scalar self unbatched and index batched
            # (rank zero).
            # self not batched, self logical rank 0, index logical rank 0
            x = torch.zeros([], device=device)
            dim = 0
            index = torch.tensor([0, 0], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test6():
            # [FUNCTION DOC] HELPER: Scenario with a vector self unbatched and index batched
            # (rank one).
            # self not batched, self logical rank 0, index logical rank 1
            x = torch.zeros(3, device=device)
            dim = 0
            index = torch.tensor([[0], [1]], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test7():
            # [FUNCTION DOC] HELPER: Scenario with a vector self unbatched and index batched
            # (rank zero).
            # self not batched, self logical rank 0, index logical rank 0
            x = torch.zeros(3, device=device)
            dim = 0
            index = torch.tensor([0, 1], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test8():
            # [FUNCTION DOC] HELPER: Scenario with a batched matrix self and a batched
            # rank-zero index.
            # self batched, self logical rank > 1, index logical rank 0
            x = torch.zeros(B, 3, 3, device=device)
            dim = 0
            index = torch.tensor([0, 1], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (0, None, 0, None)
                )

        for test in (test1, test2, test3, test4, test5, test6, test7, test8):
            check_vmap_fallback(self, test, torch.index_fill)

    def test_fill__Tensor(self, device):
        # [FUNCTION DOC] Checks vmap of the in-place fill_ with a tensor value (no OpInfo covers
        # it) for batched/unbatched self and value in several in_dims forms, under the
        # fallback-use check.
        # There's no OpInfo for fill_.Tensor, so here's an extra test for it.
        def test():
            # [FUNCTION DOC] HELPER: Runs four in-place fill_ vmap comparisons against the loop.
            B = 2
            args = (torch.randn(B, 3, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (0, 0))

            args = (torch.randn(3, B, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (-1, 0))

            args = (torch.randn(3, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (None, 0))

            args = (torch.randn(3, B, device=device), torch.randn([]))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (1, None))

        check_vmap_fallback(self, test, Tensor.fill_)

    @parametrize(
        "op,msg,extra_positional_args,extra_kwargs",
        [
            subtest(
                (
                    Tensor.scatter_add_,
                    "out-of-place operators instead of scatter_add_",
                    (),
                    {},
                ),
                name="scatter_add",
            ),
            subtest(
                (
                    Tensor.scatter_reduce_,
                    "out-of-place operators instead of scatter_reduce_",
                    ("sum",),
                    {"include_self": True},
                ),
                name="scatter_reduce",
            ),
        ],
    )
    def test_scatter_inplace_self_not_batched(
        self, device, op, msg, extra_positional_args, extra_kwargs
    ):
        # [FUNCTION DOC] Parametrized over in-place scatter_add_ and scatter_reduce_. Checks
        # vmap raises a RuntimeError recommending out-of-place operators when the destination
        # tensor is unbatched but the index or source is batched.
        x = torch.zeros(5, device=device)

        def call_op(self, dim, index, src):
            # [FUNCTION DOC] HELPER: Calls the parametrized in-place op with the given self,
            # dim, index, src plus the extra positional and keyword arguments of the case.
            return op(
                self,
                dim,
                index,
                src,
                *extra_positional_args,
                **extra_kwargs,
            )

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(call_op, in_dims=(None, None, None, 0))(
                x,
                0,
                torch.tensor([0, 1], device=device),
                torch.randn(2, 2, device=device),
            )

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(call_op, in_dims=(None, None, 0, None))(
                x,
                0,
                torch.tensor([[0, 1], [2, 3]], device=device),
                torch.randn(2, device=device),
            )

    def test_masked_fill__Tensor(self, device):
        # [FUNCTION DOC] Checks vmap of in-place masked_fill_ with a tensor value for
        # batched/unbatched self, mask and value combinations, under the fallback-use check; the
        # case with unbatched self and batched mask/value is expected to raise.
        def test():
            # [FUNCTION DOC] HELPER: Runs four in-place masked_fill_ vmap comparisons against
            # the loop.
            B = 2
            # Both self, mask, and value batched
            args = (
                torch.randn(B, 3, device=device),
                torch.randn(B, 3, device=device) > 0,
                torch.randn(B, device=device),
            )
            self.vmap_inplace_test(Tensor.masked_fill_, args, {}, (0, 0, 0))

            # self and mask batched, value not batched
            args = (
                torch.randn(B, 3, device=device),
                torch.randn(B, 3, device=device) > 0,
                torch.tensor(1.0, device=device),
            )
            self.vmap_inplace_test(Tensor.masked_fill_, args, {}, (0, 0, None))

            # self batched, mask and value not batched
            args = (
                torch.randn(B, 3, device=device),
                torch.randn(3, device=device) > 0,
                torch.tensor(1.0, device=device),
            )
            self.vmap_inplace_test(Tensor.masked_fill_, args, {}, (0, None, None))

            # self not batched (should error)
            args = (
                torch.randn(3, device=device),
                torch.randn(B, 3, device=device) > 0,
                torch.randn(B, device=device),
            )
            self.vmap_inplace_test(Tensor.masked_fill_, args, {}, (None, 0, 0))

        check_vmap_fallback(self, test, Tensor.masked_fill_)

    def test_masked_fill_Tensor_broadcast(self, device):
        # [FUNCTION DOC] Checks vmap of out-of-place masked_fill where mask rank differs from
        # self rank (self larger, mask larger, or mask unbatched) matches the loop.
        B = 2
        # self rank > mask rank, both batched
        self.vmap_outplace_test(
            torch.masked_fill,
            (
                torch.randn(B, 2, 3, device=device),
                torch.randn(B, 3, device=device) > 0,
                torch.randn(B, device=device),
            ),
            {},
            (0, 0, 0),
        )
        # mask rank > self rank, both batched
        self.vmap_outplace_test(
            torch.masked_fill,
            (
                torch.randn(B, 3, device=device),
                torch.randn(B, 2, 3, device=device) > 0,
                torch.randn(B, device=device),
            ),
            {},
            (0, 0, 0),
        )
        # self batched, mask unbatched with different rank
        self.vmap_outplace_test(
            torch.masked_fill,
            (
                torch.randn(B, 2, 3, device=device),
                torch.randn(3, device=device) > 0,
                torch.randn(B, device=device),
            ),
            {},
            (0, None, 0),
        )

    @tf32_on_and_off(0.005)
    def test_conv_double_backward(self, device):
        # [FUNCTION DOC] Checks vmap of the convolution double backward op against the loop over
        # all exhaustive batching combinations, with looser tolerance on cuda sm86. Runs with
        # tf32 on and off (decorator tolerance 0.005) and under the fallback-use check.
        images = torch.randn(2, 1, 5, 5, device=device)
        weight = torch.randn(2, 1, 2, 2, device=device)
        bias = torch.randn(2, device=device)
        ggI = torch.randn_like(images)
        ggW = torch.randn_like(weight)
        ggb = torch.randn_like(bias)
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        transposed = False
        output_padding = (0, 0)
        groups = 1
        output_mask = (True, True, True)
        gO = torch.randn_like(
            F.conv2d(images, weight, bias, stride, padding, dilation, groups)
        )

        args = (
            ggI,
            ggW,
            ggb,
            gO,
            weight,
            images,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
        op = torch.ops.aten._convolution_double_backward

        generator = get_fallback_and_vmap_exhaustive(op, args, {})
        is_cuda_sm86 = device.startswith("cuda") and torch.cuda.get_device_capability(
            0
        ) == (8, 6)
        atol, rtol = (1e-3, 1e-3) if is_cuda_sm86 else (1e-4, 1e-4)

        def test():
            # [FUNCTION DOC] HELPER: Asserts loop and batched outputs from the generator are
            # equal within tolerance.
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out, atol=atol, rtol=rtol)

        check_vmap_fallback(self, test, op)

    def test_isnan(self, device):
        # [FUNCTION DOC] Checks vmap of isnan on a 5-d tensor in which the positive entries are
        # set to NaN.
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = torch.isnan

        x = torch.randn(B, N, C, H, W)
        x[x > 0] = float("nan")
        test(self, op, (x,), in_dims=(0))

    def test_sum_scalar(self, device):
        # [FUNCTION DOC] Checks vmap of sum over a 1-element vector, with no dim, dim 0 and dim
        # minus one, returns the same values.
        x = torch.tensor([10.0], device=device)
        y = vmap(torch.sum)(x)
        self.assertEqual(y, x)

        y = vmap(lambda x: x.sum(0))(x)
        self.assertEqual(y, x)

        y = vmap(lambda x: x.sum(-1))(x)
        self.assertEqual(y, x)

    def test_isinf(self, device):
        # [FUNCTION DOC] Checks vmap of isinf on a 5-d tensor in which the positive entries are
        # set to infinity.
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = torch.isinf

        x = torch.randn(B, N, C, H, W)
        x[x > 0] = float("inf")
        test(self, op, (x,), in_dims=(0))

    def test_foo_like(self, device):
        # [FUNCTION DOC] Smoke test that vmap of ones_like and zeros_like runs on a 5-d input;
        # values not checked.
        # vfdev-5: Probably, we can remove this line. Flake8 reported as unused
        # test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        for op in [torch.ones_like, torch.zeros_like]:
            x = torch.randn(B, N, C, H, W)
            # todo(chilli): test these better
            # Not testing correctness, just that they run
            vmap(op, in_dims=(0,))(
                x,
            )

    def test_flatten(self, device):
        # [FUNCTION DOC] Checks vmap of flatten over dims one to two with the batch dim at 0.
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        op = torch.flatten

        x = torch.randn(2, 3, 4, 5)
        test(self, op, (x, 1, 2), in_dims=(0, None, None))

    def test_group_norm(self, device):
        # [FUNCTION DOC] Checks vmap of group_norm with unbatched weight and bias, and with
        # batched weight and bias.
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = F.group_norm

        x = torch.randn(B, N, C, H, W)
        weight = torch.randn(C)
        bias = torch.randn(C)
        test(self, op, (x, 3, weight, bias), in_dims=(0, None, None, None))

        x = torch.randn(B, N, C, H, W)
        weight = torch.randn(B, C)
        bias = torch.randn(B, C)
        test(self, op, (x, 4, weight, bias), in_dims=(0, None, 0, 0))

    def test_group_norm_layout_corruption(self, device):
        # [FUNCTION DOC] Regression test (issue 176432): vmap of a gradient through group_norm
        # gives the same results whether the batch dim is first or last in the cotangent.
        # Regression test for https://github.com/pytorch/pytorch/issues/176432
        def op_function(cotangent):
            # [FUNCTION DOC] HELPER: Gradient of group_norm of a fixed small input with respect
            # to that input, for the given cotangent.
            input = torch.tensor([[-6.517, -6.264]], device=device, requires_grad=True)
            result = F.group_norm(input, num_groups=1)
            return torch.autograd.grad(result, input, grad_outputs=cotangent)[0]

        cotangent = torch.tensor([[-0.0236, -0.1431]], device=device)
        result_in_dims_0 = vmap(op_function, in_dims=0)(
            cotangent.unsqueeze(0).expand(2, -1, -1)
        )
        result_in_dims_neg1 = vmap(op_function, in_dims=-1)(
            cotangent.unsqueeze(-1).expand(-1, -1, 2)
        )

        self.assertEqual(result_in_dims_0, result_in_dims_neg1)

    def test_index_put(self, device):
        # [FUNCTION DOC] Checks vmap of index_put in several forms: assignment with advanced
        # indexing at the innermost and middle dim, with slices, the functional in-place
        # index_put_, and boolean masks.
        def test(f, t, idx, values):
            # [FUNCTION DOC] HELPER: Applies f to the first batch entry as a reference then
            # asserts vmap with all batched, and with indices and/or values unbatched, gives the
            # same slice.
            base = f(t[0], idx[0], values[0])
            self.assertEqual(vmap(f, in_dims=(0, 0, 0))(t, idx, values)[0], base)
            self.assertEqual(
                vmap(f, in_dims=(0, None, None))(t, idx[0], values[0])[0], base
            )
            self.assertEqual(vmap(f, in_dims=(0, None, 0))(t, idx[0], values)[0], base)
            self.assertEqual(vmap(f, in_dims=(0, 0, None))(t, idx, values[0])[0], base)

        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Assigns z at positions y of x in place and returns x.
            x[y] = z
            return x

        x = torch.randn(3, 4, 5, device=device)
        y = torch.zeros((3, 2), device=device).long()
        z = torch.randn(3, 2, 5, device=device)
        test(f, x, y, z)

        # indexing innermost dim
        def f(t, idx, values):
            # [FUNCTION DOC] HELPER: Assigns values at indices on the second dim of t in place
            # and returns t.
            t[:, idx] = values
            return t

        t = torch.zeros((3, 2, 3))
        values = torch.ones((3, 1, 2))
        idx = torch.tensor([[1, 2]]).expand((3, 2))
        test(f, t, idx, values)

        # indexing middle dim
        def f(t, idx, values):
            # [FUNCTION DOC] HELPER: Assigns values at indices on the middle dim of t in place
            # and returns t.
            t[:, idx, :] = values
            return t

        t = torch.zeros((3, 2, 3, 3))
        values = torch.ones((3, 1, 2, 3))
        idx = torch.tensor([[0, 2]]).expand((3, 2))
        test(f, t, idx, values)

        # indexing with slices
        def f(t, values):
            # [FUNCTION DOC] HELPER: Assigns values to a slice of the second dim of t in place
            # and returns t.
            t[:, :2, :] = values
            return t

        base = f(t[0], values[0])
        self.assertEqual(vmap(f, in_dims=(0, 0))(t, values)[0], base)
        self.assertEqual(vmap(f, in_dims=(0, None))(t, values[0])[0], base)

        # index_put_
        tensor = torch.zeros(3, 3, 4)
        value = torch.ones(3, 2)
        idxs = (
            torch.tensor([[0], [1], [2]]),
            torch.tensor([[0]]),
            torch.tensor([1, 2]),
        )
        expected = torch.index_put_(tensor.clone(), idxs, value)

        def f(t, idx, v):
            # [FUNCTION DOC] HELPER: Calls index_put_ on t with the given indices and values and
            # returns t.
            torch.index_put_(t, idx, v)
            return t

        self.assertEqual(
            vmap(f, in_dims=(0, (None, None), 0))(tensor, idxs[1:], value), expected
        )
        self.assertEqual(
            vmap(f, in_dims=(0, (None, None), None))(tensor, idxs[1:], value[0]),
            expected,
        )

        # boolean mask
        B = 2
        x = torch.randn(1, 3, 3)
        gy = torch.randn(B, 1, 3, 3)

        def f(x, gy):
            # [FUNCTION DOC] HELPER: Builds a boolean mask of entries below a tiny threshold in
            # x and returns gy with the masked entries replaced by a zero scalar via the aten
            # index_put overload.
            mask = x < 1e-09
            zeros = torch.zeros([])
            index_put = torch.ops.aten.index_put.default(gy, [mask], zeros)
            return index_put

        self.vmap_outplace_test(f, (x, gy), {}, in_dims=(None, 0))

    @onlyCUDA
    @parametrize("inplace", [True, False])
    def test_0d_tensor_index_put(self, device, inplace):
        # [FUNCTION DOC] CUDA only. Parametrized on in-place versus out-of-place index_put.
        # Checks vmap over a cuda tensor with a CPU index and a 0-dim CPU value produces rows
        # with ones at the indexed positions.
        def f(t, idx, v):
            # [FUNCTION DOC] HELPER: Calls index_put_ or index_put (captured flag) on t with
            # indices and value.
            fn = torch.index_put_ if inplace else torch.index_put
            return fn(t, idx, v)

        N = 2
        t = torch.zeros((N, 5), device="cuda")
        idx = torch.tensor([1, 3])
        v = torch.tensor(1, dtype=t.dtype, device="cpu")

        expected = torch.tensor([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]], dtype=t.dtype)
        self.assertEqual(expected, vmap(f, in_dims=(0, None, None))(t, (idx,), v))

    @parametrize("training", [True, False])
    @parametrize("track_running_stats", [True, False])
    @parametrize("affine", [True, False])
    def test_batch_norm(self, device, affine, track_running_stats, training):
        # [FUNCTION DOC] Parametrized over affine, track_running_stats and training. Checks vmap
        # over an ensemble of batch norm layers (ensemble size 10) with batched weights, biases
        # and running stats against the loop; returns early for the invalid combination without
        # stats and not training.
        if not track_running_stats and not training:
            return

        test = functools.partial(_vmap_test, check_propagates_grad=False)
        BN = torch.nn.BatchNorm2d
        ensemble_size = 10
        hidden_dim = 3

        weights, buffers, _, _, _ = functional_init_with_buffers(BN, [ensemble_size])(
            hidden_dim, affine=affine, track_running_stats=track_running_stats
        )

        inputs = [torch.randn(ensemble_size, 32, hidden_dim, 16, 16, device=device)]
        in_dims = [0]

        def append(inp, in_dim):
            # [FUNCTION DOC] HELPER: Appends an input and its in_dim to the captured input and
            # in_dim lists.
            inputs.append(inp)
            in_dims.append(in_dim)

        if track_running_stats:
            running_mean, running_var, _ = buffers
            append(running_mean.to(device), 0)
            append(running_var.to(device), 0)
        else:
            append(None, None)
            append(None, None)

        if affine:
            weight, bias = weights
            append(weight.to(device), 0)
            append(bias.to(device), 0)
        else:
            append(None, None)
            append(None, None)

        append(training, None)

        def op(inp, running_mean, running_var, weight, bias, training):
            # [FUNCTION DOC] HELPER: Calls batch_norm with the given arguments; returns the
            # output with the running stats when stats are tracked, otherwise just the output.
            res = F.batch_norm(inp, running_mean, running_var, weight, bias, training)
            if track_running_stats:
                return res, running_mean, running_var
            return res

        test(self, op, tuple(inputs), in_dims=tuple(in_dims))

    def test_torch_return_types_returns(self, device):
        # [FUNCTION DOC] Checks vmap of min, max, topk and (unless unsupported on ROCm without
        # magma) linalg.eig preserves the torch.return_types result types.
        t = torch.randn(3, 2, 2, device=device)
        self.assertTrue(
            isinstance(vmap(torch.min, (0, None))(t, 0), torch.return_types.min)
        )
        self.assertTrue(
            isinstance(vmap(torch.max, (0, None))(t, 0), torch.return_types.max)
        )
        self.assertTrue(
            isinstance(
                vmap(torch.topk, (0, None, None))(t, 1, 0), torch.return_types.topk
            )
        )
        if not (TEST_WITH_ROCM and not torch.cuda.has_magma):
            self.assertTrue(
                isinstance(
                    vmap(torch.linalg.eig, (0))(t), torch.return_types.linalg_eig
                )
            )

    def test_namedtuple_returns(self, device):
        # [FUNCTION DOC] Checks vmap preserves a user-defined namedtuple return type.
        Point = namedtuple("Point", ["x", "y"])

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a Point namedtuple holding the two inputs.
            return Point(x=x, y=y)

        x = torch.randn(2, 5, device=device)
        y = torch.randn(2, 3, device=device)
        self.assertTrue(isinstance(vmap(f)(x, y), Point))

    def test_inplace_on_view(self, device):
        # [FUNCTION DOC] Checks vmap of a vector-Jacobian push function through an
        # in-place-heavy view chain matches the loop for many in_dims layouts; skips layouts
        # with unbatched cotangent due to a composite compliance issue.
        def func(leaf):
            # [FUNCTION DOC] HELPER: Squares the leaf, then applies in-place multiplication,
            # sine and cosine on views of the transposed result and returns the final view.
            base = leaf * leaf
            view = base.transpose(0, 1)
            view[2:4, 2:4] *= 2
            view[0:2, 0:2].diagonal().sin_()
            view = view[1:3, 1:3]
            view.cos_()
            return view

        def push_vjp(leaf, gout):
            # [FUNCTION DOC] HELPER: Vector-Jacobian product of the view function at the leaf
            # with the given output cotangent.
            _, vjp_fn = vjp(func, leaf)
            (result,) = vjp_fn(gout)
            return result

        leaf = torch.randn(4, 4, device=device)
        gout = torch.randn(2, 2, device=device)
        args = (leaf, gout)

        for (
            batched_args,
            in_dims,
            _,
        ) in generate_vmap_inputs(args, {}):
            if in_dims[1] is None:
                # triggers some composite compliance problem
                continue
            self.vmap_outplace_test(push_vjp, batched_args, {}, in_dims)

    def test_advanced_indexing(self, device):
        # [FUNCTION DOC] Checks vmap of advanced indexing along different axes for several input
        # ranks and index shapes against the loop over exhaustive batching combinations.
        def test(f, args):
            # [FUNCTION DOC] HELPER: Asserts loop and batched outputs from the exhaustive helper
            # are equal.
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(f, args, {}):
                self.assertEqual(loop_out, batched_out)

        def f(x, idx):
            # [FUNCTION DOC] HELPER: Indexes the second dim of x with idx.
            return x[:, idx]

        def f2(x, idx):
            # [FUNCTION DOC] HELPER: Indexes the first dim of x with idx.
            return x[idx, :]

        def f3(x, idx):
            # [FUNCTION DOC] HELPER: Indexes the third dim of x with idx.
            return x[:, :, idx]

        inps = (
            torch.randn(5, 5, 5, device=device),
            torch.randn(5, 5, 5, 5, device=device),
            torch.randn(5, 5, 5, 5, 5, device=device),
        )
        idxes = (
            torch.tensor([0, 1, 2], device=device),
            torch.tensor([0, 1, 2], device=device).reshape(3, 1),
            torch.tensor([0, 1, 2], device=device).reshape(3, 1, 1),
        )
        for inp, idx in itertools.product(inps, idxes):
            test(f, (inp, idx))
            test(f2, (inp, idx))
            test(f3, (inp, idx))

    def test_nested_advanced_indexing(self, device):
        # [FUNCTION DOC] Checks that advanced indexing inside nested vmaps (an indexing
        # function mapped over dim one inside an outer vmap over indices) equals a naive
        # stack-of-selects implementation.
        e = torch.rand(7, 4, device=device)
        idx = torch.tensor([0, 1], device=device).view(2, 1)

        # simple reference implementation for comparison
        def _fake_vmap(f, in_dims=0, out_dims=0):
            # [FUNCTION DOC] HELPER: Reference implementation of vmap that loops over the batch,
            # applies f to each slice and stacks the results along out_dims.
            def w(input):
                # [FUNCTION DOC] HELPER: Applies f on each slice of the input along in_dims and
                # stacks results.
                r = [f(input.select(in_dims, i)) for i in range(input.size(in_dims))]
                return torch.stack(r, out_dims)

            return w

        def with_vmap(_vmap):
            # [FUNCTION DOC] HELPER: Applies the given vmap-like transform to an index-using
            # function nested inside another transform, and returns the stacked result.
            def g(idx_):
                # [FUNCTION DOC] HELPER: For a given index tensor, maps over dim one of the
                # captured tensor, indexing it.
                def f(e_):
                    # [FUNCTION DOC] HELPER: Indexes its argument with the captured
                    # index tensor.
                    return e_[idx_]

                return _vmap(f, in_dims=1)(e)

            r = _vmap(g)(idx)
            return r

        a = with_vmap(vmap)
        b = with_vmap(_fake_vmap)
        self.assertEqual(a, b)

    @ops(
        filter(lambda op: "linalg" in op.name, op_db + additional_op_db),
        allowed_dtypes=(torch.float,),
    )
    @skipOps(
        {
            xfail("linalg.vector_norm"),  # can accept vector inputs
            xfail("linalg.norm"),  # can accept vector inputs
            xfail("linalg.norm", "subgradients_at_zero"),  # can accept vector inputs
            xfail("linalg.vander"),  # can accept vector inputs
            skip(
                "linalg.multi_dot"
            ),  # accepts list of tensor inputs, has its own special test
            xfail("linalg.vecdot"),
            # throws in vmap on CUDA
            # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
            # but it passes locally
            xfail("linalg.diagonal"),
            skip("linalg.matrix_norm", ""),
            skip("linalg.ldl_solve", ""),
        },
    )
    def test_vmap_linalg_failure_1D_input(self, device, dtype, op):
        # [FUNCTION DOC] OpInfo test over linalg ops (float): for each op with a 2-d sample,
        # passing a 1-d row directly must raise a RuntimeError about dimension, and the same op
        # under vmap with an expanded square input must also raise one. Has an expected-failure
        # and skip table.
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            if sample.input.dim() != 2 or sample.input.shape[0] == 0:
                continue
            test_input = sample.input[
                0
            ]  # using the sample input avoids numerical inconsistency issues
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                op(test_input, *sample.args, **sample.kwargs)

            def op_wrapper(inp):
                # [FUNCTION DOC] HELPER: Calls the op on an input with the sample's other
                # arguments.
                return op(inp, *sample.args, **sample.kwargs)

            # square inputs are more likely to pass linalg checks
            test_input = test_input.expand(test_input.shape[0], test_input.shape[0])
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                return vmap(op_wrapper)(test_input)

    def test_vmap_multi_dot_failure_1D_input(self):
        # [FUNCTION DOC] Checks multi_dot with a 1-d middle tensor raises a RuntimeError ("must
        # be 2D") both directly and under vmap.
        # special exception for first and last tensors so making giving 3 items avoids special cases
        inputs = (torch.randn(3, 3), torch.randn(3), torch.randn(3, 3))
        with self.assertRaisesRegex(RuntimeError, "tensor 1 must be 2D but got 1D"):
            torch.linalg.multi_dot(inputs)

        # square inputs are more likely to pass linalg checks
        inputs = tuple(i.expand(i.shape[0], i.shape[0]) for i in inputs)
        with self.assertRaisesRegex(RuntimeError, "tensor 1 must be 2D but got 1D"):
            return vmap(torch.linalg.multi_dot)(inputs)

    def test_vmap_escaped_error(self):
        # [FUNCTION DOC] Checks that using a tensor that escaped from a vmapped function
        # afterwards raises a RuntimeError mentioning escape, with the name of the C++
        # plumbing/batch rule reporting it, for out-of-place, in-place, loss, pooling,
        # reduction, boxed and no-return paths.
        escaped = None

        def f(x):
            # [FUNCTION DOC] HELPER: Stores its input in a nonlocal variable (escaping it) and
            # returns its square.
            nonlocal escaped
            escaped = x
            return x**2

        x = torch.randn([3, 3, 3, 3, 3])
        vmap(f)(x)

        common_message = (
            r"your tensor may have escaped from inside a function being vmapped.*{0}.*"
        )

        # Note: These are not a complete set of tests for all possible functions calling 'vmap_check_escaped'

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_plumbing")
        ):
            escaped.sin()

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_tensor_inputs_batch_rule")
        ):
            escaped.sin_()

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_inplace_plumbing")
        ):
            escaped.mul_(1)

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("binary_cross_entropy_plumbing")
        ):
            torch.nn.functional.binary_cross_entropy(escaped, torch.zeros([3, 3, 3, 3]))

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_existing_bdim_all_batch_rule")
        ):
            torch.nn.functional.adaptive_max_pool2d(escaped, output_size=(1, 1))

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_reduction_batch_rule")
        ):
            escaped.argmin()

        a = torch.zeros([4, 4, 4, 4])
        b = torch.zeros([4, 4, 4, 4], dtype=torch.long)
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_all_tensors_have_optional_bdim")
        ):
            torch.ops.aten.adaptive_max_pool2d_backward(escaped, a, b)

        vmap(f)(torch.tensor([[0, 0], [0, 0]], dtype=torch.int))
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_plumbing_no_returns")
        ):
            torch.ops.aten._linalg_check_errors(escaped, "linalg.inv", is_matrix=False)

    def test_vmap_with_anomaly_detection(self):
        # [FUNCTION DOC] Checks vmap of grad works with anomaly detection on and still reports
        # the NaN backward error for a function with a sqrt of negative values.
        with torch.autograd.set_detect_anomaly(True):
            x = torch.zeros(3) - 1

            def fn(x):
                # [FUNCTION DOC] HELPER: Sum of its input.
                return x.sum()

            per_sample_grad = vmap(grad(fn))(x)
            self.assertEqual(per_sample_grad, torch.ones_like(x))

            def bad_fn(x):
                # [FUNCTION DOC] HELPER: Sum of square roots of its input; produces NaN gradients
                # for negative inputs.
                return x.sqrt().sum()

            err_msg = "Function 'SqrtBackward0' returned nan values in its 0th output."
            with self.assertRaisesRegex(RuntimeError, err_msg):
                vmap(grad(bad_fn))(x)

    def test_searchsorted_bucketize(self, device):
        # [FUNCTION DOC] Checks vmap of searchsorted and bucketize with different sample
        # boundaries per batch entry (OpInfo reuses repeated samples), for scalar and vector
        # values, using the out-of-place helper.
        # OpInfo generates test with repeated samples in batch dim.
        # Thus we test explicitly with different samples across a batch.

        def test():
            # [FUNCTION DOC] HELPER: Runs the searchsorted and bucketize vmap comparisons.
            boundaries = torch.tensor(
                [[1, 4, 5, 7, 9], [1, 2, 6, 8, 10]], device=device
            )
            v = torch.tensor(3, device=device)
            self.vmap_outplace_test(torch.searchsorted, (boundaries, v), {}, (0, None))
            self.vmap_outplace_test(torch.bucketize, (v, boundaries), {}, (None, 0))
            boundaries = torch.tensor([[1, 4, 5, 7, 9], [1, 2, 4, 8, 9]], device=device)
            v = torch.tensor([3, 4], device=device)
            self.vmap_outplace_test(torch.searchsorted, (boundaries, v), {}, (0, 0))
            self.vmap_outplace_test(torch.bucketize, (v, boundaries), {}, (0, 0))

        test()


@markDynamoStrictTest
class TestRandomness(TestCase):
    # [CLASS DOC] Tests vmap randomness modes (error, same, different) for random ops: seeding,
    # generator state, batched/unbatched inputs and expected errors. Provides assertion helpers
    # on slice equality and error messages. Marked dynamo strict.
    def _reset_random(self, generator, orig_state, use_generator, seed):
        # [FUNCTION DOC] HELPER: Resets the random state, via the generator's saved state if a
        # generator is used, otherwise by manual seed, and returns the call result.
        return (
            generator.set_state(orig_state)
            if use_generator
            else torch.manual_seed(seed)
        )

    def _get_image(self, batched_input, batch_size, device):
        # [FUNCTION DOC] HELPER: Returns a tensor of ones shaped as an image batch of 3x3x14x14,
        # with the batch dim first, last, or absent depending on the string (first, last, none);
        # raises AssertionError otherwise.
        if batched_input == "first":
            return torch.ones([batch_size, 3, 3, 14, 14], device=device)
        if batched_input == "last":
            return torch.ones([3, 3, 14, 14, batch_size], device=device)
        if batched_input != "none":
            raise AssertionError(
                f"Expected batched_input to be 'none', got '{batched_input}'"
            )
        return torch.ones([3, 3, 14, 14], device=device)

    def _assert_all_slices_equal(self, tensor):
        # [FUNCTION DOC] HELPER: Asserts all slices along dim 0 equal the first slice.
        expected = tensor[0]
        self.assertTrue((tensor == expected).all())

    def _assert_all_slices_unique(self, tensor):
        # [FUNCTION DOC] HELPER: Asserts all slices along dim 0 are pairwise different,
        # comparing all pairs with nested vmaps and requiring the off-diagonal equality matrix
        # to be all false.
        B0 = tensor.shape[0]
        slices_equal = vmap(vmap(lambda x, y: (x == y).all(), (0, None)), (None, 0))(
            tensor, tensor
        )
        if slices_equal.shape != (B0, B0):
            raise AssertionError(
                f"Expected shape ({B0}, {B0}), got {slices_equal.shape}"
            )
        slices_equal.diagonal().zero_()
        self.assertEqual(slices_equal, torch.zeros_like(slices_equal))

    def _assert_throws_in_error_mode(self, fn, args, in_dims):
        # [FUNCTION DOC] HELPER: Asserts vmap in randomness error mode raises a RuntimeError
        # saying a random operation was called in error mode.
        with self.assertRaisesRegex(
            RuntimeError, r"called random operation while in randomness error mode"
        ):
            vmap(fn, in_dims=in_dims, randomness="error")(*args)

    def _assert_throws_in_different_mode_inplace(self, fn, args, in_dims):
        # [FUNCTION DOC] HELPER: Asserts vmap in different mode raises a RuntimeError about
        # in-place randomness on an unbatched tensor.
        with self.assertRaisesRegex(
            RuntimeError, r"different inplace randomness on an unbatched tensor"
        ):
            vmap(fn, in_dims=in_dims, randomness="different")(*args)

    def _assert_throws_in_same_mode_batched(self, fn, args, in_dims):
        # [FUNCTION DOC] HELPER: Asserts vmap in same mode raises a RuntimeError saying same
        # randomness with a batched tensor input is not supported.
        with self.assertRaisesRegex(
            RuntimeError,
            r"Vmap does not currently support same randomness with a batched tensor input",
        ):
            vmap(fn, in_dims=in_dims, randomness="same")(*args)

    def _in_dims(self, *batched_strings):
        # [FUNCTION DOC] HELPER: Converts placement strings (first, last, none) for the leading
        # arguments into a tuple of in_dims (zero, minus one, None), then appends zero for the
        # extra always-batched dummy argument. Raises AssertionError for any other string.
        def get_in_dim(batched_string):
            # [FUNCTION DOC] HELPER: Maps a single placement string to an in_dim; raises
            # AssertionError if invalid.
            if batched_string == "first":
                return 0
            if batched_string == "last":
                return -1
            if batched_string != "none":
                raise AssertionError(
                    f"Expected batched_string to be 'none', got '{batched_string}'"
                )
            return None

        batched_strings = batched_strings + (
            "first",
        )  # for the always batched as first dim dummy argument
        return tuple(get_in_dim(batched_string) for batched_string in batched_strings)

    @parametrize("randomness", ["same", "different", "error"])
    @parametrize("use_generator", [True, False])
    def test_factory_ops(self, device, randomness, use_generator):
        # [FUNCTION DOC] Parametrized over randomness mode and generator use. Checks random
        # factory functions (randn, rand, randint with and without low, normal) under vmap:
        # error mode raises; different mode equals one call with the batch dim prepended and
        # slices unique; same mode gives equal slices equal to one unbatched call. Seeds or
        # restores generator state before each run.
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = (
            {"device": device, "generator": generator}
            if use_generator
            else {"device": device}
        )
        ops = [
            lambda _, shape: torch.randn(shape, **kwargs),
            lambda _, shape: torch.rand(shape, **kwargs),
            lambda _, shape: torch.randint(100, shape, **kwargs),
            lambda _, shape: torch.randint(5, 100, shape, **kwargs),
            lambda _, shape: torch.normal(0.0, 1.0, shape, **kwargs),
        ]
        B0 = 4
        shape = (3, 3)
        seed = 1234567

        for op in ops:
            passed = torch.randn(B0, device=device)
            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, shape), in_dims=(0, None)
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=(0, None), randomness=randomness)(
                passed, shape
            )

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                expected = op(passed, [B0, *shape])
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                expected = op(passed, shape)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("randomness", ["same", "different", "error"])
    @parametrize("use_generator", [True, False])
    def test_randperm(self, device, randomness, use_generator):
        # [FUNCTION DOC] Parametrized over randomness mode and generator use. Checks randperm
        # under vmap (special case since it takes no batch size): error mode raises; different
        # mode matches a sequence of unbatched calls (or only uniqueness on cuda under dynamo);
        # same mode gives equal slices.
        # needs a special case because randperm doesn't take a batch size
        B0 = 4
        seed = 1234567
        passed = torch.randn(B0, device=device)

        torch.manual_seed(seed)
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()

        kwargs = (
            {"device": device, "generator": generator}
            if use_generator
            else {"device": device}
        )

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(lambda _: torch.randperm(10, **kwargs), randomness=randomness)(
                    passed
                )
            return

        vmap_result = vmap(
            lambda _: torch.randperm(10, **kwargs), randomness=randomness
        )(passed)
        generator = generator.set_state(orig_state)
        torch.manual_seed(seed)
        if randomness == "different":
            for i in range(B0):
                expected = torch.randperm(10, **kwargs)
                # RNG differs between eager and via dynamo trace on CUDA
                if TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda":
                    self._assert_all_slices_unique(vmap_result)
                else:
                    self.assertEqual(vmap_result[i], expected)
        else:
            expected = torch.randperm(10, **kwargs)
            # RNG differs between eager and via dynamo trace on CUDA
            if TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda":
                self._assert_all_slices_equal(vmap_result)
            else:
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_dropout(self, device, randomness, batched_input):
        # [FUNCTION DOC] Parametrized over randomness and where the batch is. Checks dropout
        # under vmap: error mode raises; otherwise the drop rate estimate is between 0.25 and
        # 0.75, slices differ in different mode and are identical in same mode.
        def op(t, ignored):
            # [FUNCTION DOC] HELPER: Training dropout applied to a tensor of ones shaped like
            # the first argument; the second argument is ignored (it provides an always-batched
            # dummy input).
            return torch.nn.functional.dropout(torch.ones_like(t), training=True)

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # Check that the randomness is within bounds...
        # ideally this is close to 0.5
        p_estimate = vmap_result.mean() / 2
        self.assertTrue(p_estimate < 0.75)
        self.assertTrue(p_estimate > 0.25)

        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        if randomness != "same":
            raise AssertionError(
                f"Expected randomness to be 'same', got '{randomness}'"
            )
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_alpha_dropout(self, device, randomness, batched_input):
        # [FUNCTION DOC] Parametrized over randomness and batch placement. Checks alpha_dropout
        # under vmap: error mode raises; different mode gives unique slices; same mode identical
        # slices. No value check.
        def op(t, ignored):
            # [FUNCTION DOC] HELPER: Training alpha dropout on a tensor of ones shaped like the
            # first argument; the second argument is ignored.
            return torch.nn.functional.alpha_dropout(torch.ones_like(t), training=True)

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        # I have no clue how to actually test correctness of alpha dropout because the docs
        # seem wrong: https://github.com/pytorch/pytorch/issues/74004
        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )
        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        if randomness != "same":
            raise AssertionError(
                f"Expected randomness to be 'same', got '{randomness}'"
            )
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("dim", [2, 3])
    def test_feature_dropout(self, device, randomness, batched_input, dim):
        # [FUNCTION DOC] Parametrized over randomness, batch placement and dim (2 or 3). Checks
        # dropout2d or dropout3d under vmap: error mode raises; the output has the feature
        # pattern (each plane all zero or all scaled), slices are unique in different mode and
        # equal in same mode.
        def op(t, ignored):
            # [FUNCTION DOC] HELPER: Training 2d or 3d feature dropout (by captured dim) on ones
            # shaped like the first argument; the second argument is ignored.
            f = (
                torch.nn.functional.dropout2d
                if dim == 2
                else torch.nn.functional.dropout3d
            )
            return f(torch.ones_like(t), training=True)

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        if dim == 3:
            unsqueeze_dim = -2 if batched_input == "last" else -1
            passed = passed.unsqueeze(unsqueeze_dim)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # Check the "feature" pattern
        dims = [-1, -2] if dim == 2 else [-1, -2, -3]
        planes_numel = (
            2
            * vmap_result.numel()
            / (vmap_result.shape[0] * vmap_result.shape[1] * vmap_result.shape[2])
        )
        planes = vmap_result.sum(dims)
        result = (planes == 0) ^ (planes == planes_numel)
        self.assertEqual(result, torch.ones_like(result, dtype=torch.bool))

        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        if randomness != "same":
            raise AssertionError(
                f"Expected randomness to be 'same', got '{randomness}'"
            )
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_feature_alpha_dropout(self, device, randomness, batched_input):
        # [FUNCTION DOC] Parametrized over randomness and batch placement. Checks
        # feature_alpha_dropout under vmap: error mode raises; each plane takes only the min or
        # max value; slices unique in different mode and equal in same mode.
        def op(t, ignored):
            # [FUNCTION DOC] HELPER: Training feature alpha dropout on ones shaped like the
            # first argument; the second argument is ignored.
            return torch.nn.functional.feature_alpha_dropout(
                torch.ones_like(t), training=True
            )

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        unsqueeze_dim = -2 if batched_input == "last" else -1
        passed = passed.unsqueeze(unsqueeze_dim)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # I have no clue how to actually test correctness of alpha dropout because the docs
        # seem wrong: https://github.com/pytorch/pytorch/issues/74004

        # Check the "feature" pattern
        dims = [-1, -2, -3]
        planes = vmap_result.sum(dims)
        max_elt = planes.max()
        min_elt = planes.min()
        result = (planes == min_elt) ^ (planes == max_elt)
        self.assertEqual(result, torch.ones_like(result, dtype=torch.bool))

        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        if randomness != "same":
            raise AssertionError(
                f"Expected randomness to be 'same', got '{randomness}'"
            )
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_like_functions(self, device, randomness, batched_input):
        # [FUNCTION DOC] Parametrized over randomness and batch placement. Checks randint_like,
        # rand_like and randn_like under vmap: error mode raises; different mode matches the
        # like call on the batched tensor (value check skipped on cuda under dynamo) with unique
        # slices; same mode gives equal slices matching one unbatched call.
        seed = 1234567
        supported_ops = [
            lambda t, _: torch.randint_like(t, 20),
            lambda t, _: torch.randint_like(t, 0, 20),
            lambda t, _: torch.rand_like(t),
            lambda t, _: torch.randn_like(t),
        ]
        B0 = 4

        for op in supported_ops:
            always_batched = torch.randn(B0)
            passed = self._get_image(batched_input, B0, device)
            in_dims = self._in_dims(batched_input)

            if randomness == "error":
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"called random operation while in randomness error mode",
                ):
                    vmap(op, in_dims=in_dims, randomness=randomness)(
                        passed, always_batched
                    )
                return

            torch.manual_seed(seed)
            vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
                passed, always_batched
            )

            torch.manual_seed(seed)

            if batched_input == "last":
                passed = passed.movedim(-1, 0)
            if randomness == "different":
                if batched_input == "none":
                    passed = passed.expand(B0, *passed.shape)
                expected = op(passed, 0)

                self._assert_all_slices_unique(vmap_result)
                # RNG differs between eager and via dynamo trace on CUDA
                if not (TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda"):
                    self.assertEqual(expected, vmap_result)
                return

            if randomness != "same":
                raise AssertionError(
                    f"Expected randomness to be 'same', got '{randomness}'"
                )
            if batched_input != "none":
                passed = passed[0]
            expected = op(passed, 0)
            self._assert_all_slices_equal(vmap_result)
            # RNG differs between eager and via dynamo trace on CUDA
            if not (TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda"):
                for i in range(B0):
                    self.assertEqual(expected, vmap_result[i])

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_random_unary_inplace(
        self, device, use_generator, randomness, batched_input
    ):
        # [FUNCTION DOC] Parametrized over generator use, randomness and batch placement. Checks
        # in-place random ops (random_, normal_, bernoulli_, cauchy_, exponential_, geometric_,
        # log_normal_, uniform_) under vmap: error mode raises; different mode on an unbatched
        # tensor raises; otherwise results match the same operation on cloned inputs (batched
        # copy or single unbatched call) with unique or equal slices.
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        ops = [
            lambda t, _: t.random_(**kwargs),
            lambda t, _: t.random_(100, **kwargs),
            lambda t, _: t.random_(-5, 100, **kwargs),
            lambda t, _: t.normal_(**kwargs),
            lambda t, _: t.bernoulli_(**kwargs),
            lambda t, _: t.cauchy_(**kwargs),
            lambda t, _: t.exponential_(**kwargs),
            lambda t, _: t.geometric_(0.5, **kwargs),
            lambda t, _: t.log_normal_(**kwargs),
            lambda t, _: t.uniform_(**kwargs),
        ]
        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input)

        for op in ops:
            # because of in place updates, clone inputs
            always_batched = torch.randn(B0, device=device)
            passed = self._get_image(batched_input, B0, device)
            passed_expected = passed.clone()

            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return
            if randomness == "different" and batched_input == "none":
                self._assert_throws_in_different_mode_inplace(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                passed, always_batched
            )

            if batched_input == "last":
                passed_expected = passed_expected.movedim(-1, 0)
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                expected = op(passed_expected, always_batched)
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                if batched_input != "none":
                    passed_expected = passed_expected[
                        0
                    ].clone()  # bug in pytorch, normal_ on views doesn't work
                expected = op(passed_expected, always_batched)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("batched_probability", ["first", "last", "none"])
    def test_bernoulli_in_place(
        self, device, use_generator, randomness, batched_input, batched_probability
    ):
        # [FUNCTION DOC] Parametrized over generator use, randomness, input placement and
        # probability placement. Checks in-place bernoulli with a probability tensor under vmap:
        # error mode raises; same mode with a batched probability raises; unbatched input with
        # batched probability raises a RuntimeError about extra args being larger than self;
        # different mode with unbatched input raises; otherwise results match the same op on the
        # equivalent unbatched/stacked tensors.
        B0 = 4
        seed = 1234567
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        in_dims = self._in_dims(batched_input, batched_probability)

        def op(t, p, ignored):
            # [FUNCTION DOC] HELPER: In-place bernoulli of the first argument with probability
            # tensor second argument (third argument is an ignored always-batched dummy), using
            # the captured generator options.
            return t.bernoulli_(p, **kwargs)

        # because of in place updates, clone inputs
        always_batched = torch.randn(B0, device=device)
        input = self._get_image(batched_input, B0, device)
        input_expected = input.clone()
        probability = self._get_image(batched_probability, B0, device) - 0.5

        if randomness == "error":
            self._assert_throws_in_error_mode(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return
        if randomness == "same" and batched_probability != "none":
            self._assert_throws_in_same_mode_batched(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return
        if batched_input == "none" and batched_probability != "none":
            regex = r"there exists a Tensor `other` in extra_args that has more elements than `self`"
            with self.assertRaisesRegex(RuntimeError, regex):
                vmap(op, in_dims=in_dims, randomness=randomness)(
                    input, probability, always_batched
                )
            return
        if randomness == "different" and batched_input == "none":
            self._assert_throws_in_different_mode_inplace(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return

        self._reset_random(generator, orig_state, use_generator, seed)
        vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
            input, probability, always_batched
        )

        self._reset_random(generator, orig_state, use_generator, seed)
        if batched_input == "last":
            input_expected = input_expected.movedim(-1, 0)
        if batched_probability == "last":
            probability = probability.movedim(-1, 0)
        if randomness == "different":
            expected = op(input_expected, probability, always_batched)
            self._assert_all_slices_unique(vmap_result)
            self.assertEqual(vmap_result, expected)
        else:
            if batched_input != "none":
                input_expected = input_expected[0]
            expected = op(input_expected, probability, always_batched)
            self._assert_all_slices_equal(vmap_result)
            for i in range(B0):
                self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("batched_other", ["first", "last", "none"])
    def test_random_binary_out_of_place(
        self, device, use_generator, randomness, batched_input, batched_other
    ):
        # [FUNCTION DOC] Parametrized over generator use, randomness, input and other
        # placement. Checks out-of-place normal (tensor mean and std) and binomial under
        # vmap: error mode raises; same mode with any batched input raises; different mode
        # matches the op applied to stacked inputs with unique slices; same mode with both
        # unbatched gives equal slices.
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        ops = [
            lambda t, o, _: torch.normal(t, o, **kwargs),
            lambda t, o, _: torch.binomial(t, (o - 0.5), **kwargs),
        ]

        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input, batched_other)

        for op in ops:
            always_batched = torch.randn(B0, device=device)
            input = self._get_image(batched_input, B0, device)
            other = self._get_image(batched_other, B0, device)

            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (input, other, always_batched), in_dims=in_dims
                )
                return
            if randomness == "same" and (
                batched_input != "none" or batched_other != "none"
            ):
                self._assert_throws_in_same_mode_batched(
                    op, (input, other, always_batched), in_dims=in_dims
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                input, other, always_batched
            )

            if batched_input == "last":
                input = input.movedim(-1, 0)
            if batched_other == "last":
                other = other.movedim(-1, 0)

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                if batched_input == "none":
                    input = input.expand(B0, *input.shape)
                expected = op(input, other, always_batched)
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                if batched_input != "none" or batched_other != "none":
                    raise AssertionError(
                        f"Expected batched_input='none' and batched_other='none', "
                        f"got batched_input='{batched_input}' and "
                        f"batched_other='{batched_other}'"
                    )
                expected = op(input, other, always_batched)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_random_unary_out_of_place(
        self, device, use_generator, randomness, batched_input
    ):
        # [FUNCTION DOC] Parametrized over generator use, randomness and input placement. Checks
        # out-of-place random ops with a tensor input (normal with tensor mean or std, bernoulli
        # forms, standard gamma, dirichlet, poisson) under vmap: error raises; same mode with
        # batched input raises; different mode matches the stacked application with unique
        # slices; same mode with unbatched input gives equal slices.
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        ops = [
            lambda t, _: torch.normal(0.0, torch.abs(t), **kwargs),
            lambda t, _: torch.normal(t, 1.0, **kwargs),
            lambda t, _: torch.bernoulli(t - 0.5, **kwargs),
            lambda t, _: torch.bernoulli(t, 0.5, **kwargs),
            lambda t, _: torch._standard_gamma(t, **kwargs),
            lambda t, _: torch._sample_dirichlet(t, **kwargs),
            lambda t, _: torch.poisson(t, **kwargs),
        ]

        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input)

        for op in ops:
            always_batched = torch.randn(B0, device=device)
            passed = self._get_image(batched_input, B0, device)
            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return
            if randomness == "same" and batched_input != "none":
                self._assert_throws_in_same_mode_batched(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                passed, always_batched
            )

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                if batched_input == "none":
                    passed = passed.expand(B0, *passed.shape)
                if batched_input == "last":
                    passed = passed.movedim(-1, 0)
                expected = op(passed, always_batched)
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                expected = op(passed, always_batched)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_call", [True, False])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_multinomial(
        self, device, use_generator, randomness, batched_call, batched_input
    ):
        # [FUNCTION DOC] Parametrized over generator use, randomness, whether the call itself is
        # batched and input placement. Checks multinomial (ten samples) under vmap: error
        # raises; same mode with batched input raises; different mode matches a flattened
        # batched call reshaped; same mode with unbatched input gives equal slices.
        def flatten_input(input, batch_call, batch_location):
            # [FUNCTION DOC] HELPER: Flattens the image tensor to the rank multinomial expects
            # (batch, batch dim, categories) depending on whether the call is batched and where
            # the batch dim lies; raises AssertionError if the resulting rank is not the
            # expected one.
            if batch_call and batch_location != "none":
                final_size = 3  # [B0, B, N]
            elif not batch_call and batch_location == "none":
                final_size = 1  # [N]
            else:
                final_size = 2  # [B0, N] or [B, N]

            start_idx = final_size - 1
            end_idx = -1
            if batch_location == "last":
                start_idx -= 1
                end_idx -= (
                    1  # gets to correct final size because using negative indices
                )

            ret = input.flatten(start_idx, end_idx)
            if ret.dim() != final_size:
                raise AssertionError(f"Expected dim {final_size}, got {ret.dim()}")
            return ret

        def op(input, _):
            # [FUNCTION DOC] HELPER: Draws ten multinomial samples from the first argument
            # with the captured generator options; the second argument is an ignored
            # always-batched dummy.
            return torch.multinomial(input, 10, **kwargs)

        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}

        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input)

        always_batched = torch.randn(B0, device=device)
        passed = self._get_image(batched_input, B0, device)
        passed = flatten_input(passed, batched_call, batched_input)
        if randomness == "error":
            self._assert_throws_in_error_mode(
                op, (passed, always_batched), in_dims=in_dims
            )
            return
        if randomness == "same" and batched_input != "none":
            self._assert_throws_in_same_mode_batched(
                op, (passed, always_batched), in_dims=in_dims
            )
            return

        generator = self._reset_random(generator, orig_state, use_generator, seed)
        vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
            passed, always_batched
        )

        generator = self._reset_random(generator, orig_state, use_generator, seed)

        if randomness == "different":
            if batched_input == "none":
                passed = passed.expand(B0, *passed.shape)
            if batched_input == "last":
                passed = passed.movedim(-1, 0)
            orig_passed_size = passed.shape[:2] if batched_call else passed.shape[:1]
            passed = passed.flatten(0, 1) if batched_call else passed
            expected = op(passed, always_batched)
            expected = expected.reshape(*orig_passed_size, 10)
            self._assert_all_slices_unique(vmap_result)
            self.assertEqual(vmap_result, expected)
        else:
            expected = op(passed, always_batched)
            self._assert_all_slices_equal(vmap_result)
            for i in range(B0):
                self.assertEqual(vmap_result[i], expected)

    def test_unsupported_random(self, device):
        # [FUNCTION DOC] Checks vmap with same randomness rejects random ops that use out
        # variants (randn with out, normal with out) with a RuntimeError, and rrelu with a
        # RuntimeError saying it is not yet supported.
        x = torch.randn(3, device=device)
        y = x.abs()
        z = x.abs()
        with self.assertRaisesRegex(RuntimeError, "calling out variants"):

            def f(x):
                # [FUNCTION DOC] HELPER: Calls randn with an out tensor, which is unsupported
                # under vmap randomness.
                return torch.randn(3, device=device, out=y)

            vmap(f, randomness="same")(x)
        with self.assertRaisesRegex(RuntimeError, "calling out variants"):

            def f(x0, x1):
                # [FUNCTION DOC] HELPER: Calls normal with an out tensor, which is unsupported
                # under vmap randomness.
                return torch.normal(x, y, out=x)

            vmap(f, randomness="same")(z, z)
        with self.assertRaisesRegex(RuntimeError, "do not yet support"):

            def f(z):
                # [FUNCTION DOC] HELPER: Calls rrelu on a captured tensor, which is not
                # supported under vmap randomness.
                return torch.rrelu(x)

            vmap(f, randomness="same")(z)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    def test_chunk_vmap(self, in_dim, out_dim):
        # [FUNCTION DOC] Parametrized over in_dim and out_dim. Checks chunk_vmap with different
        # randomness and many chunk counts produces unique slices (the random numbers differ
        # across batch entries).
        randomness = "different"

        x = torch.randn(4, 5, 6)

        def f(x):
            # [FUNCTION DOC] HELPER: Sine of the input plus uniform noise.
            y = x.sin() + torch.rand_like(x)
            return y

        for chunks in [1, 2, 3, 4, 7, 10, 16]:
            output = chunk_vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunks=chunks,
            )(x)
            self._assert_all_slices_unique(output)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    def test_vmap_chunksize(self, in_dim, out_dim):
        # [FUNCTION DOC] Parametrized over in_dim and out_dim. Checks vmap with chunk_size and
        # different randomness over many chunk sizes yields unique slices.
        randomness = "different"

        x = torch.randn(4, 5, 6)

        def f(x):
            # [FUNCTION DOC] HELPER: Sine of the input plus uniform noise.
            y = x.sin() + torch.rand_like(x)
            return y

        for chunk_size in [1, 2, 3, 4, 7, 10, 16, 100]:
            output = vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunk_size=chunk_size,
            )(x)
            self._assert_all_slices_unique(output)

    def test_jacfwd_with_random(self):
        # [FUNCTION DOC] Checks jacfwd respects the randomness argument: default raises a
        # RuntimeError for a random op, and same and different modes run without error.
        # checks on behavior are above, this just checks that jacfwd respects
        # the randomness param

        x = torch.rand(3, 4)
        with self.assertRaisesRegex(
            RuntimeError, r"called random operation while in randomness error mode"
        ):
            jacfwd(torch.bernoulli)(x)

        # x isn't batched so use bernoulli since it doesn't do inplace randomness
        jacfwd(torch.bernoulli, randomness="same")(x)
        jacfwd(torch.bernoulli, randomness="different")(x)

    @parametrize("randomness", ["error", "same", "different"])
    def test_dropout_unbatched(self, device, randomness):
        # [FUNCTION DOC] Parametrized over randomness. Regression test for issue 92283: dropout
        # applied to an unbatched tensor inside vmap over another input raises a RuntimeError in
        # error mode and must not raise in same or different modes.
        x = torch.randn(3, device=device)
        y = torch.randn(1, 3, device=device)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds x to the mean over dim one of dropout of y with
            # probability 0.5.
            # output from dropout should be a Tensor[B, 1, 3] (B=3)
            return x + torch.nn.functional.dropout(y, p=0.5).mean(1)

        # We just verify that this doesn't raise an error for
        # `same` and `different` randomness.
        # Ref: https://github.com/pytorch/pytorch/issues/92283
        context = (
            self.assertRaises(RuntimeError)
            if randomness == "error"
            else contextlib.nullcontext()
        )
        with context:
            vmap(fn, in_dims=(0, None), randomness=randomness)(x, y)


@markDynamoStrictTest
class TestTransformFailure(TestCase):
    # [CLASS DOC] Tests that functorch transforms (vmap, grad, grad_and_value, vjp, jvp, jacrev,
    # jacfwd) fail with a clear error when applied to a function using a plain
    # autograd.Function. Marked dynamo strict.
    @skipIfTorchDynamo()
    @parametrize(
        "transform",
        ["vmap", "grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd"],
    )
    def test_fails_with_autograd_function(self, device, transform):
        # [FUNCTION DOC] Parametrized over the transform name. Skipped under dynamo. Builds an
        # identity autograd.Function, applies the chosen transform to a function calling it, and
        # asserts a RuntimeError mentioning autograd.Function; skips on cpu for grad and vmap in
        # certain dynamo CI environments where unexpected successes occur.
        failed_build_envs = ("linux-focal-py3.8-clang10", "linux-focal-py3.11-clang10")
        if (
            device == "cpu"
            and transform in ["grad", "vmap"]
            and TEST_WITH_TORCHDYNAMO
            and os.getenv("BUILD_ENVIRONMENT", "") in failed_build_envs
        ):
            raise unittest.SkipTest(
                "Unexpected successes on focal with dynamo,"
                + " see https://github.com/pytorch/pytorch/issues/107173"
            )

        class Test(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity autograd Function whose backward passes the gradient
            # through unchanged.
            @staticmethod
            def forward(_, input):
                # [FUNCTION DOC] HELPER: Returns the input unchanged (the first argument is the
                # ignored context).
                return input

            @staticmethod
            def backward(_, grad_input):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged (the first
                # argument is the ignored context).
                return grad_input

        transform = getattr(functorch, transform)

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the identity autograd Function to its input.
            return Test.apply(x)

        if transform in (grad, grad_and_value):
            input = torch.tensor(4.0)
        else:
            input = torch.randn(5)

        if transform is vjp:
            transform = functools.partial(transform, f)
        elif transform is jvp:
            input = (input,)
            transform = functools.partial(transform, f, input)
        else:
            transform = transform(f)

        with self.assertRaisesRegex(RuntimeError, "autograd.Function"):
            transform(input)


@markDynamoStrictTest
class TestVmapDeviceType(Namespace.TestVmapBase):
    # [CLASS DOC] Device-specific vmap tests for internal aten ops (_is_all_true, _is_any_true,
    # the tensor check op) run through the no-fallback base class. Marked dynamo strict.
    def _vmap_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards to the module-level vmap test function using self as
        # the case.
        return _vmap_test(self, *args, **kwargs)

    def test__is_all_true(self, device):
        # [FUNCTION DOC] Checks vmap of the all-true reduction returns a plain scalar boolean
        # tensor (not a batched tensor) matching the expected value for several inputs, under
        # the fallback-use check.
        def test():
            # [FUNCTION DOC] HELPER: Runs the all-true vmap cases on random data with
            # comparisons made all or partly true or false.
            def f(x, *, expected_result):
                # [FUNCTION DOC] HELPER: Calls the all-true op, asserts the result is not a
                # batched tensor, is scalar-shaped and has the expected value, and returns it.
                result = torch.ops.aten._is_all_true(x)
                self.assertFalse(torch._C._functorch.is_batchedtensor(result))
                self.assertEqual(result.shape, torch.Size([]))
                self.assertEqual(result.item(), expected_result)
                return result

            x = torch.rand(10, device=device)
            vmap(f)(x >= 0, expected_result=True)
            vmap(f)(x < 0, expected_result=False)

            x[random.choice(range(10))] *= -1
            vmap(f)(x >= 0, expected_result=False)
            vmap(f)(x < 0, expected_result=False)

            x = -torch.rand(10, device=device)
            vmap(f)(x > 0, expected_result=False)
            vmap(f)(x <= 0, expected_result=True)

        check_vmap_fallback(self, test, torch._is_all_true)

    def test__is_any_true(self, device):
        # [FUNCTION DOC] Checks vmap of the any-true reduction returns a plain scalar boolean
        # tensor with the expected value for all-false, one-true and strided slices, under the
        # fallback-use check.
        def test():
            # [FUNCTION DOC] HELPER: Runs the any-true vmap cases.
            def f(x, *, expected_result):
                # [FUNCTION DOC] HELPER: Calls the any-true op, asserts the result is not a
                # batched tensor, is scalar-shaped and has the expected value, and returns it.
                result = torch.ops.aten._is_any_true(x)
                self.assertFalse(torch._C._functorch.is_batchedtensor(result))
                self.assertEqual(result.shape, torch.Size([]))
                self.assertEqual(result.item(), expected_result)
                return result

            x = torch.zeros(10, device=device, dtype=torch.bool)
            vmap(f)(x > 0, expected_result=False)

            x[5] = True
            vmap(f)(x > 0, expected_result=True)
            vmap(f)(x[1::2], expected_result=True)
            vmap(f)(x[0::2], expected_result=False)

        check_vmap_fallback(self, test, torch._is_any_true)

    def test_check_tensor(self, device):
        # [FUNCTION DOC] Checks the test-only tensor check op under vmap and nested vmap for
        # many shapes: passes when all entries satisfy the condition, raises a RuntimeError
        # with the check message when all or exactly one entry fails. Run under the
        # fallback-use check.
        def test():
            # [FUNCTION DOC] HELPER: Runs all tensor-check vmap cases over the list of
            # test sizes.
            test_sizes = [
                (1,),
                (10,),
                (1, 1),
                (1, 10),
                (10, 1),
                (10, 10),
                (1, 1, 1),
                (10, 1, 1),
                (1, 10, 1),
                (10, 10, 10),
            ]

            def check_gte_0(t):
                # [FUNCTION DOC] HELPER: Runs the tensor check op on a boolean tensor of entries
                # that are non-negative.
                return torch._test_check_tensor(t >= 0)

            error_message = "Test message for TORCH_CHECK_TENSOR_ALL"

            for size in test_sizes:
                t_all_gte_0 = torch.rand(size, device=device)
                t_all_lt_0 = t_all_gte_0 - 1

                vmap(check_gte_0)(t_all_gte_0)

                if len(size) >= 2:
                    vmap(vmap(check_gte_0))(t_all_gte_0)

                with self.assertRaisesRegex(RuntimeError, error_message):
                    vmap(check_gte_0)(t_all_lt_0)

                if len(size) >= 2:
                    with self.assertRaisesRegex(RuntimeError, error_message):
                        vmap(vmap(check_gte_0))(t_all_lt_0)

                if t_all_gte_0.numel() > 1:
                    t_all_gte_0_but_one = t_all_gte_0.clone()
                    idx = (random.choice(range(dim_size)) for dim_size in size)
                    t_all_gte_0_but_one[(..., *idx)] = -1

                    with self.assertRaisesRegex(RuntimeError, error_message):
                        vmap(check_gte_0)(t_all_gte_0_but_one)

                    if len(size) >= 2:
                        with self.assertRaisesRegex(RuntimeError, error_message):
                            vmap(vmap(check_gte_0))(t_all_gte_0_but_one)

        check_vmap_fallback(self, test, torch._test_check_tensor)


@markDynamoStrictTest
class TestVmapNestedTensor(Namespace.TestVmapBase):
    # [CLASS DOC] vmap tests on nested tensors, using the no-fallback base class: fallback with
    # nested inputs, dense/nested mixing, cat, and error cases (non-zero dims, multiple vmap
    # levels). Includes helpers to build nested tensors. Marked dynamo strict.
    def _vmap_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards to the module-level vmap test function using self as
        # the case.
        return _vmap_test(self, *args, **kwargs)

    # dims should be something like [5, None, 10], with None indicating that a
    # random ragged structure should be used
    def _create_nt(self, dims, device):
        # [FUNCTION DOC] HELPER: Builds a nested tensor of dims[0] components on the device;
        # dims after the first give each component's size, with None meaning a random size from
        # two to nine (ragged).
        sizes = [
            [
                d if d is not None else torch.randint(2, 10, size=(1,)).item()
                for d in dims[1:]
            ]
            for d in range(dims[0])
        ]
        return torch.nested.nested_tensor(
            [torch.randn(*size) for size in sizes], device=device
        )

    # Creates an NT matching another NT's number of components and
    # shape / ragged structure for all dims specified to be -1.
    def _nt_from_similar(self, other, dims):
        # [FUNCTION DOC] HELPER: Builds a random nested tensor with the same number of
        # components as other. Each dim given as minus one copies the matching size of each
        # component of other; other values are used directly. Raises AssertionError if the rank
        # or batch size does not match.
        if len(dims) != other.dim():
            raise AssertionError(
                f"Expected len(dims) == other.dim(), got {len(dims)} != {other.dim()}"
            )
        if dims[0] != -1 and dims[0] != other.size(0):
            raise AssertionError(
                f"Expected dims[0] to be -1 or {other.size(0)}, got {dims[0]}"
            )

        ret_sizes = []
        for t in other.unbind():
            other_size = t.shape
            ret_size = []
            for i, d in enumerate(dims[1:]):
                if d == -1:
                    ret_size.append(other_size[i])
                else:
                    ret_size.append(d)
            ret_sizes.append(ret_size)

        return torch.nested.nested_tensor(
            [torch.randn(*size) for size in ret_sizes], device=other.device
        )

    @allowVmapFallbackUsage
    def test_fallback_unary(self, device):
        # [FUNCTION DOC] Checks the vmap fallback on a unary pointwise function of a nested
        # tensor. Fallback allowed.
        def f(x):
            # [FUNCTION DOC] HELPER: Scales the sine of x and adds a constant.
            return x.sin() * 5.0 + 4.0

        nt = self._create_nt([4, None, 3], device=device)
        self._vmap_test(f, (nt,))

    @allowVmapFallbackUsage
    def test_fallback_binary(self, device):
        # [FUNCTION DOC] Checks the vmap fallback on matrix product of two nested tensors.
        # Fallback allowed.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of the two arguments.
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = self._create_nt([5, 3, None], device=device)
        self._vmap_test(f, (x, y))

    @allowVmapFallbackUsage
    def test_fallback_binary_nt_and_unbatched_dense(self, device):
        # [FUNCTION DOC] Checks the fallback for a nested tensor times an unbatched dense
        # matrix. Fallback allowed.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of the two arguments.
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(3, 4, device=device)
        self._vmap_test(f, (x, y), in_dims=(0, None))

    @allowVmapFallbackUsage
    def test_fallback_binary_nt_and_batched_dense(self, device):
        # [FUNCTION DOC] Checks the fallback for a nested tensor times a batched dense tensor.
        # Fallback allowed.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of the two arguments.
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(5, 3, 4, device=device)
        self._vmap_test(f, (x, y))

    def test_nt_acts_as_dense_in_vmap(self, device):
        # [FUNCTION DOC] Checks that inside vmap a nested tensor input appears as a dense
        # (non-nested) tensor.
        def f(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError if the input is nested, otherwise
            # returns it.
            if x.is_nested:
                raise AssertionError("Expected x to not be nested inside vmap")
            return x

        x = self._create_nt([5, None, 3], device=device)
        self._vmap_test(f, (x,))

    def test_cat_batching_rule(self, device):
        # [FUNCTION DOC] Checks the cat batching rule for nested tensors, concatenating along
        # dim 0 and 1 with differing ragged structure but same other dims, and same structure
        # but differing other dims.
        def f(x, y, dim):
            # [FUNCTION DOC] HELPER: Concatenates the two tensors along the given dim.
            return torch.cat([x, y], dim=dim)

        # Different nested structure, same other dims
        x = self._create_nt([3, None, 2], device=device)
        y = self._create_nt([3, None, 2], device=device)
        self._vmap_test(functools.partial(f, dim=0), (x, y))

        x = self._create_nt([3, 2, None], device=device)
        y = self._create_nt([3, 2, None], device=device)
        self._vmap_test(functools.partial(f, dim=1), (x, y))

        # Same nested structure, different other dims
        x = self._create_nt([3, 2, None], device=device)
        y = self._nt_from_similar(x, [-1, 4, -1])
        self._vmap_test(functools.partial(f, dim=0), (x, y))

        x = self._create_nt([3, None, 2], device=device)
        y = self._nt_from_similar(x, [-1, -1, 4])
        self._vmap_test(functools.partial(f, dim=1), (x, y))

    # .shape calls don't work on NTs
    # TODO: Fix this somehow?
    @unittest.expectedFailure
    def test_shape_call(self, device):
        # [FUNCTION DOC] Expected-failure test that reading the shape of a nested tensor inside
        # vmap works.
        # KNOWN ISSUE: shape calls fail on nested tensors, and the nested tensor helper is
        # called without the required device argument, so the failure may come from that
        # TypeError instead.
        def f(x):
            # [FUNCTION DOC] HELPER: Reads the first entry of the shape of its input and returns
            # the input.
            x.shape[0]
            return x

        x = self._create_nt([3, None, 2])
        self._vmap_test(f, (x,))

    def test_nt_with_nonzero_in_dim_raises(self, device):
        # [FUNCTION DOC] Checks vmap over dim 2 of a nested tensor raises a RuntimeError saying
        # only dim 0 is allowed.
        def f(x):
            # [FUNCTION DOC] HELPER: Identity function.
            return x

        x = self._create_nt([3, None, 2], device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Nested tensors can only be vmapped over dim=0"
        ):
            vmap(f, in_dims=2)(x)

    def test_nt_with_nonzero_out_dim_raises(self, device):
        # [FUNCTION DOC] Checks vmap with out_dims 2 on a nested tensor raises a RuntimeError
        # saying only dim 0 is allowed.
        def f(x):
            # [FUNCTION DOC] HELPER: Identity function.
            return x

        x = self._create_nt([3, None, 2], device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Nested tensors can only be vmapped over dim=0"
        ):
            vmap(f, out_dims=2)(x)

    def test_fallback_with_nt_and_batched_dense_with_nonzero_bdim_raises(self, device):
        # [FUNCTION DOC] Checks the fallback with a nested tensor and a dense tensor batched at
        # a dim other than 0 raises a RuntimeError about mixed nested and non-nested arguments
        # requiring bdim zero.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of the two arguments.
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(3, 5, 4, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            "Fallback not supported for mixed nested / non-nested arguments without bdim=0",
        ):
            vmap(f, in_dims=(0, 1))(x, y)

    def test_multilevel_vmap_raises(self, device):
        # [FUNCTION DOC] Checks two and three levels of nested vmap on a nested tensor raise a
        # RuntimeError saying only one level of vmap is supported.
        def f(x):
            # [FUNCTION DOC] HELPER: Scales the sine of x and adds a constant.
            return x.sin() * 4.0 + 3.0

        x = self._create_nt([2, 2, 2, None], device=device)

        with self.assertRaisesRegex(
            RuntimeError, "Only one level of vmap is supported"
        ):
            vmap(vmap(f))(x)

        with self.assertRaisesRegex(
            RuntimeError, "Only one level of vmap is supported"
        ):
            vmap(vmap(vmap(f)))(x)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestVmapOperatorsOpInfo, globals(), only_for=only_for)

instantiate_device_type_tests(
    TestVmapBatchedGradient,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestTransformFailure, globals(), only_for=only_for)
instantiate_device_type_tests(TestRandomness, globals(), only_for=only_for)
instantiate_device_type_tests(TestVmapDeviceType, globals(), only_for=only_for)
instantiate_device_type_tests(TestVmapNestedTensor, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
