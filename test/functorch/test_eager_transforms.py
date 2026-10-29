# Owner(s): ["module: functorch"]
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Tests for the functorch eager-mode function transforms: grad, grad_and_value, vjp,
# jvp, jacrev, jacfwd, hessian, linearize, functionalize, make_functional, functional_call,
# stack_module_state, ensembling and make_fx, plus the internal argnums slicing helper.
# Test classes are decorated for strict Dynamo handling, most are device-generic (instantiated per
# device type near the end of the file) and some are parametrized. Import time side effects: tries
# to import torchvision and emits a UserWarning if it is missing, setting the module flag
# USE_TORCHVISION accordingly.
import copy
import math
import os
import subprocess
import sys
import unittest
import warnings
from functools import partial, wraps

# NB: numpy is a testing dependency!
import numpy as np
from common_utils import expectedFailureIf

import functorch
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from functorch import (
    combine_state_for_ensemble,
    grad,
    grad_and_value,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    make_functional,
    make_functional_with_buffers,
    make_fx,
    vjp,
    vmap,
)
from functorch.experimental import functionalize, replace_all_batch_norm_modules_
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._dynamo import allow_in_graph
from torch._functorch.eager_transforms import _slice_argnums
from torch._functorch.make_functional import (
    functional_init,
    functional_init_with_buffers,
)
from torch._functorch.utils import enable_single_level_autograd_function
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.func import functional_call, linearize, stack_module_state
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    SM70OrLater,
    TEST_CUDA,
    tf32_on_and_off,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import get_all_fp_dtypes
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_WINDOWS,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_CUDA_MEM_LEAK_CHECK,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


USE_TORCHVISION = False
try:
    import torchvision  # noqa: F401

    USE_TORCHVISION = True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )


def _assert_functorch_wrapper_repr(test_case, actual, op_list):
    # [FUNCTION DOC] HELPER: Checks that the printed representation of a tensor (the string
    # actual) contains exactly the functorch wrappers implied by op_list, a list of transforms
    # (grad or vmap) ordered from outermost-first creation level.
    # It counts the GradTrackingTensor and BatchedTensor markers against the number of grad and
    # vmap entries, asserts no FunctionalTensor appears and that the text mentions a Tensor. Then
    # it builds the expected wrapper tokens with level numbers and batch dims, requires the text
    # to start with the last (innermost) token, and requires all tokens to appear in reverse
    # order. Returns nothing; failures go through the test case assertions.
    expected_grad_wrappers = sum(op is grad for op in op_list)
    expected_batched_wrappers = sum(op is vmap for op in op_list)

    test_case.assertEqual(actual.count("GradTrackingTensor("), expected_grad_wrappers)
    test_case.assertEqual(actual.count("BatchedTensor("), expected_batched_wrappers)
    test_case.assertNotIn("FunctionalTensor(", actual)
    test_case.assertTrue("Tensor(" in actual or "tensor(" in actual, actual)

    wrapper_tokens = []
    bdim = expected_batched_wrappers
    for level, op in enumerate(op_list):
        if op is grad:
            wrapper_tokens.append(f"GradTrackingTensor(lvl={level + 1}, value=")
        elif op is vmap:
            bdim -= 1
            wrapper_tokens.append(f"BatchedTensor(lvl={level + 1}, bdim={bdim}, value=")

    if not wrapper_tokens:
        return

    test_case.assertTrue(actual.startswith(wrapper_tokens[-1]), actual)
    search_from = 0
    for token in reversed(wrapper_tokens):
        pos = actual.find(token, search_from)
        test_case.assertNotEqual(pos, -1, actual)
        search_from = pos + len(token)


def _assert_leaf_tensor_repr(test_case, actual):
    # [FUNCTION DOC] HELPER: Asserts that the string actual, the repr of a tensor, is that of a
    # plain leaf tensor: no GradTrackingTensor, BatchedTensor or FunctionalTensor wrapper text and
    # it starts with Tensor( or tensor(. Uses test_case assertions and returns nothing.
    test_case.assertNotIn("GradTrackingTensor(", actual)
    test_case.assertNotIn("BatchedTensor(", actual)
    test_case.assertNotIn("FunctionalTensor(", actual)
    test_case.assertTrue(actual.startswith(("Tensor(", "tensor(")), actual)


# TestCase for _slice_argnums, an important helper function


class VmapTearDownMixin:
    # [CLASS DOC] HELPER: Mixin for test classes that use the vmap nesting counters directly. Its
    # only member is a tearDown hook that cleans up leaked functorch interpreter state.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Does nothing unless running under Dynamo (TEST_WITH_TORCHDYNAMO).
        # Otherwise it pops every vmap interpreter at the top of the functorch interpreter stack
        # by decrementing the vmap nesting, stopping at the first non-vmap interpreter, and emits
        # a warning if anything was popped. This keeps a failed test from leaving stale nesting
        # that breaks later tests.
        # Ensure that in the case of a test failure, the next test won't fail
        # because of a previous call to _vmap_increment_nesting that wasn't undone
        # i.e. test_vmap_free_tensor fails when PYTORCH_TEST_WITH_DYNAMO=1
        # and the call to increment nesting is not undone
        if not TEST_WITH_TORCHDYNAMO:
            return

        warn = False
        while ci := torch._C._functorch.peek_interpreter_stack():
            if ci.key() == torch._C._functorch.TransformType.Vmap:
                warn = True
                torch._C._functorch._vmap_decrement_nesting()
            else:
                break

        if warn:
            msg = (
                "Interpreter stack is not empty. Test should have called "
                "'torch._C._functorch._vmap_decrement_nesting()'"
            )
            warnings.warn(msg)


@markDynamoStrictTest
class TestSliceArgnums(TestCase):
    # [CLASS DOC] Tests for the internal _slice_argnums helper that selects positional arguments
    # by an int or tuple of ints (negative allowed), covering validation errors, ordering and
    # pytree arguments.
    def test_invalid_argnum_type(self):
        # [FUNCTION DOC] Checks that _slice_argnums raises RuntimeError for argnums that are not
        # an int or tuple (float, list), that tuple elements must be int, and that nested tuples
        # are rejected with a must-be-int message.
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _slice_argnums(args, 0.0)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _slice_argnums(args, [0])
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _slice_argnums(args, (0.0,))

        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _slice_argnums(args, ((0, 1), 2))

    def test_out_of_bounds_argnum_values(self):
        # [FUNCTION DOC] Checks that _slice_argnums raises RuntimeError mentioning positional
        # inputs when an int or tuple argnum is out of range, including negative values below
        # minus the argument count.
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, 1)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, -2)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, (-2,))

    def test_not_enough_argnums(self):
        # [FUNCTION DOC] Checks that an empty tuple of argnums raises RuntimeError saying it must
        # be non-empty.
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            _slice_argnums(args, ())

    def test_duplicate_argnums(self):
        # [FUNCTION DOC] Checks that repeated argnums, including the same index given as positive
        # and negative, raise RuntimeError saying they must be unique.
        x = torch.randn(3)
        args = (x, x)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _slice_argnums(args, (0, 0))
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _slice_argnums(args, (0, -2))

    def test_flat_args_with_positive_int_argnum(self):
        # [FUNCTION DOC] Checks that an int argnum selecting from flat float arguments returns a
        # one-element tuple holding that argument, for the first and last positive index.
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        res = _slice_argnums(args, 0)
        self.assertEqual(res, (0.1,))

        res = _slice_argnums(args, 4)
        self.assertEqual(res, (4.1,))

    def test_flat_args_with_negative_int_argnum(self):
        # [FUNCTION DOC] Checks that negative int argnums index from the end of the arguments and
        # return a one-element tuple, for minus one and minus the argument count.
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        res = _slice_argnums(args, -1)
        self.assertEqual(res, (4.1,))

        res = _slice_argnums(args, -5)
        self.assertEqual(res, (0.1,))

    def test_flat_args_with_tuple_argnum(self):
        # [FUNCTION DOC] Checks that a tuple of argnums returns the selected arguments as a tuple
        # in the given order, including selecting all of them and mixing positive with
        # negative indices.
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        res = _slice_argnums(args, (0, 1, 2, 3, 4))
        self.assertEqual(res, args)

        res = _slice_argnums(args, (0, -3))
        self.assertEqual(res, (0.1, 2.1))

    def test_pytree_args(self):
        # [FUNCTION DOC] Checks that arguments that are pytrees (tuples, lists) are returned whole
        # as single entries when selected by int, one-element tuple, negative or mixed argnums.
        args = ((0.1, 1.1), 2.0, [3.1])

        res = _slice_argnums(args, 0)
        self.assertEqual(res, args[0:1])

        res = _slice_argnums(args, (0,))
        self.assertEqual(res, args[0:1])

        res = _slice_argnums(args, -1)
        self.assertEqual(res, args[-1:])

        res = _slice_argnums(args, (0, -2))
        self.assertEqual(res, args[0:2])

    def test_argnums_reorders(self):
        # [FUNCTION DOC] Checks that the result order follows the argnums order, so selecting
        # indices one then zero yields the arguments swapped.
        args = ((0.1, 1.1, 2.1), 3.1, 4.1)

        res = _slice_argnums(args, (1, 0))
        self.assertEqual(res, (args[1], args[0]))


def _get_weights_and_functional_call(net, mechanism):
    # [FUNCTION DOC] HELPER: Returns a pair (function, weights) for module net. With mechanism
    # make_functional it returns the result of make_functional on the module. Otherwise it
    # requires the mechanism to be functional_call (raises AssertionError if not) and returns a
    # wrapper taking weights and data plus the dict of the module named parameters, so both
    # mechanisms share one calling convention.
    if mechanism == "make_functional":
        return make_functional(net)
    else:
        if mechanism != "functional_call":
            raise AssertionError(f"Expected 'functional_call', got {mechanism!r}")
        # this makes it so the function from make_functional and this call have the same signature

        def net_func(weights, data):
            # [FUNCTION DOC] HELPER: Calls the module via functional_call with the given weights
            # dict and the data as the single positional input, returning its output.
            return functional_call(net, weights, (data,))

        return net_func, dict(net.named_parameters())


def _get_weights_and_functional_call_with_buffers(net, mechanism):
    # [FUNCTION DOC] HELPER: Like the weights helper but also handles buffers. With mechanism
    # make_functional it returns the result of make_functional_with_buffers; otherwise it requires
    # functional_call (raises AssertionError if not) and returns a wrapper function plus dicts of
    # named parameters and named buffers.
    if mechanism == "make_functional":
        return make_functional_with_buffers(net)
    else:
        if mechanism != "functional_call":
            raise AssertionError(f"Expected 'functional_call', got {mechanism!r}")

        # this makes it so the function from make_functional and this call have the same signature
        def net_func(weights, buffers, data):
            # [FUNCTION DOC] HELPER: Calls the module via functional_call using the weights and
            # buffers as the parameter and buffer dicts and the data as the single positional
            # input, returning its output.
            return functional_call(net, (weights, buffers), (data,))

        return net_func, dict(net.named_parameters()), dict(net.named_buffers())


@markDynamoStrictTest
class TestGradTransform(TestCase):
    # [CLASS DOC] Device-generic tests for the grad transform: primitive and composite functions,
    # tensor attribute access inside grad, in-place ops and views, escaped wrappers, composition
    # with vjp and autograd, argnums handling, pytree inputs, has_aux and unrelated outputs.
    def test_primitive(self, device):
        # [FUNCTION DOC] Checks grad of sin on a scalar equals cos.
        x = torch.randn([], device=device)
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_composite_simple(self, device):
        # [FUNCTION DOC] Checks grad of a function that flattens and sums a 3-d tensor is
        # all ones.
        x = torch.randn(2, 3, 4, device=device)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

    def test_fn_with_kwargs(self, device):
        # [FUNCTION DOC] Checks grad works when the function takes a keyword argument, giving the
        # same result as passing it positionally.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the elementwise product of its two inputs.
            return (x * y).sum()

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected = grad(foo)(x, y)
        result = grad(foo)(x, y=y)
        self.assertEqual(result, expected)

    def test_composite_complicated(self, device):
        # [FUNCTION DOC] Checks grad of a vector-matrix product summed to a scalar against the
        # gradient computed by regular autograd.
        x = torch.randn(3, device=device)
        y = torch.randn(3, 5, device=device)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Matrix-multiplies x and y and returns the sum.
            result = x @ y
            return result.sum()

        result = grad(foo)(x, y)

        x.requires_grad_()
        out = foo(x, y)
        (expected,) = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_composite_two_ops(self, device):
        # [FUNCTION DOC] Checks grad of a cross entropy loss with respect to logits matches
        # regular autograd, with integer class targets passed as a second argument.
        N, C = 2, 5
        y = torch.randn(N, C, device=device)
        targets = torch.randint(0, C, (N,), device=device)

        def foo(y, targets):
            # [FUNCTION DOC] HELPER: Returns the cross entropy of y against the targets.
            return F.cross_entropy(y, targets)

        result = grad(foo)(y, targets)

        y.requires_grad_()
        (expected,) = torch.autograd.grad(foo(y, targets), y)

        self.assertEqual(result, expected)

    def _test_attributes(self, get_attr_lambda, device):
        # [FUNCTION DOC] HELPER: Shared body for attribute tests. Computes the attribute with the
        # callable get_attr_lambda on a double precision tensor, then runs grad on a function that
        # asserts the same attribute computed on the wrapped tensor is equal, so wrappers must
        # report the same metadata.
        x = torch.randn(2, 3, 5, dtype=torch.double, device=device)
        expected = get_attr_lambda(x)

        def foo(x):
            # [FUNCTION DOC] HELPER: Asserts the attribute of the wrapped input equals the
            # expected value captured from the outer scope, and returns the input sum.
            self.assertEqual(get_attr_lambda(x), expected)
            return x.sum()

        grad(foo)(x)

    def test_shape(self, device):
        # [FUNCTION DOC] Checks that the shape attribute is preserved inside grad.
        self._test_attributes(lambda x: x.shape, device)

    def test_dtype(self, device):
        # [FUNCTION DOC] Checks that the dtype attribute is preserved inside grad.
        self._test_attributes(lambda x: x.dtype, device)

    def test_is_cuda(self, device):
        # [FUNCTION DOC] Checks that the is_cuda attribute is preserved inside grad.
        self._test_attributes(lambda x: x.is_cuda, device)

    def test_numel(self, device):
        # [FUNCTION DOC] Checks that numel is preserved inside grad.
        self._test_attributes(lambda x: x.numel(), device)

    def test_layout_sparse(self, device):
        # [FUNCTION DOC] Checks that a sparse COO input keeps its sparse layout inside grad (the
        # wrapper reports sparse layout) and that the returned gradient is also sparse.
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]], device=device)
        values = torch.tensor([3.0, 4.0, 5.0], device=device)
        sparse_x = torch.sparse_coo_tensor(indices, values, (2, 3), device=device)

        # Verify the input is sparse
        self.assertEqual(sparse_x.layout, torch.sparse_coo)

        def foo(x):
            # [FUNCTION DOC] HELPER: Asserts the wrapped input reports sparse COO layout and
            # returns the sum of the values of the coalesced tensor.
            # assert GradTrackingTensor still reports sparse layout
            self.assertEqual(x.layout, torch.sparse_coo)
            return x.coalesce()._values().sum()

        result = grad(foo)(sparse_x)

        # The gradient should also be sparse
        self.assertEqual(result.layout, torch.sparse_coo)

    def test_inplace(self, device):
        # [FUNCTION DOC] Checks grad through an in-place sin on a clone gives cosine.
        x = torch.randn([], device=device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Clones the input and applies in-place sin, returning it.
            return x.clone().sin_()

        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_inplace_on_view(self, device):
        # [FUNCTION DOC] Checks grad when an in-place op is applied to a view (first element) of a
        # clone and the base is summed; compares against regular autograd.
        x = torch.randn(3, device=device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Clones the input, applies in-place sin to its first element
            # view, returns the sum of the clone.
            y = x.clone()
            y0 = y[0]
            y0.sin_()
            return y.sum()

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        (expected,) = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_view_base(self, device):
        # [FUNCTION DOC] Checks grad when an in-place op is applied to the base after a view was
        # taken and the view is returned; compares against regular autograd.
        x = torch.randn(3, device=device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Clones the input, takes a view of the first element, applies
            # in-place sin to the whole clone and returns the view.
            y = x.clone()
            y0 = y[0]
            y.sin_()
            return y0

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        (expected,) = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_captures(self, device):
        # [FUNCTION DOC] Checks that mutating a captured (closed over) tensor in place with a
        # value from inside grad raises RuntimeError saying a captured Tensor cannot be mutated.
        # Skipped under Dynamo.
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        captured = torch.randn(3, device=device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Copies the input into the captured tensor in place and
            # returns the sum of their product.
            captured.copy_(x)
            return (x * captured).sum()

        with self.assertRaisesRegex(RuntimeError, "mutate a captured Tensor"):
            grad(foo)(x)

    def test_nesting_simple(self, device):
        # [FUNCTION DOC] Checks that nested grad of sin gives minus sin.
        x = torch.randn([], device=device)
        result = grad(grad(torch.sin))(x)
        self.assertEqual(result, -torch.sin(x))

    @skipIfTorchDynamo("Ref: https://github.com/pytorch/pytorch/issues/103613")
    def test_escaped_wrappers_are_marked_as_dead(self, device):
        # [FUNCTION DOC] Checks that a tensor that escapes grad (stored in a list) is marked dead,
        # reporting a level of minus one. Skipped under Dynamo (issue 103613).
        x = torch.randn([], device=device)
        escaped = []

        def foo(x):
            # [FUNCTION DOC] HELPER: Computes sin of the input, appends it to the outer escaped
            # list and returns it.
            y = x.sin()
            escaped.append(y)
            return y

        grad(foo)(x)
        self.assertEqual(torch._C._functorch.dlevel(escaped[0]), -1)

    @skipIfTorchDynamo("Ref: https://github.com/pytorch/pytorch/issues/103613")
    def test_escaped_wrappers_are_ignored(self, device):
        # [FUNCTION DOC] Checks that an escaped wrapper is ignored afterwards: operations on it
        # behave as at level zero and give the plain result. Skipped under Dynamo (issue 103613).
        x = torch.randn([], device=device)
        escaped = []

        def foo(x):
            # [FUNCTION DOC] HELPER: Computes sin of the input, appends it to the outer escaped
            # list and returns it.
            y = x.sin()
            escaped.append(y)
            return y

        grad(foo)(x)

        something = escaped[0].sum()
        self.assertEqual(torch._C._functorch.dlevel(something), 0)
        self.assertEqual(something, x.sin().sum())

    def test_manual_seed_inside_grad(self, device):
        # [FUNCTION DOC] Checks that seeding the RNG inside a function run under grad works,
        # comparing against regular autograd under a frozen RNG state.
        x = torch.randn([], device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Seeds the RNG with zero and returns the input times a random
            # tensor of the same shape.
            torch.manual_seed(0)
            return x * torch.randn_like(x)

        with freeze_rng_state():
            result = grad(f)(x)
            x.requires_grad_()
            (expected,) = torch.autograd.grad(f(x), x)
            self.assertEqual(result, expected)

    def test_vjp(self, device):
        # [FUNCTION DOC] Checks the vjp function returns the sine output and a function mapping
        # a cotangent to cotangent times cosine of the input.
        x = torch.randn([], device=device)
        out, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(out, x.sin())

        v = torch.randn([], device=device)
        (result,) = vjp_fn(v)
        self.assertEqual(result, v * x.cos())

    def test_vjp_two_outputs(self, device):
        # [FUNCTION DOC] Checks vjp works for a function returning the same tensor twice, by
        # calling the returned vjp function with the result as cotangents (no value assertion).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input twice as a tuple.
            return x, x

        result, vjp_fn = vjp(f, torch.tensor(1.0))
        vjp_fn(result)

    def test_conj_bit(self):
        # [FUNCTION DOC] Checks the conjugate bit is handled: inside grad the input is not conj,
        # its conjugate is, and the gradient of abs of a complex scalar equals its sign.
        x = torch.tensor(1 + 1j)

        def foo(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError if the input is conj or its conjugate
            # is not conj; returns the absolute value of the conjugate.
            if x.is_conj():
                raise AssertionError("Expected x to not be conj")
            y = x.conj()
            if not y.is_conj():
                raise AssertionError("Expected y to be conj")
            return y.abs()

        res = grad(foo)(x)
        with torch.no_grad():
            self.assertEqual(res, torch.ones_like(res) * torch.sgn(x))

    def test_composed_with_autograd(self, device):
        # [FUNCTION DOC] Checks grad output can be differentiated further with regular autograd,
        # giving minus sin.
        x = torch.randn([], requires_grad=True, device=device)

        y = grad(torch.sin)(x)
        (result,) = torch.autograd.grad(y, x)
        self.assertEqual(result, -x.sin())

    def test_grad_of_vjp_composition(self, device):
        # [FUNCTION DOC] Checks grad of a function that calls vjp and returns the vjp output
        # gives cosine of x.
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Runs vjp of sin at x, then returns the grad of the vjp
            # function applied to y.
            out, vjp_fn = vjp(torch.sin, x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_vjp_of_grad_composition(self, device):
        # [FUNCTION DOC] Checks vjp of grad of sin applied to a cotangent equals minus the
        # cotangent times sine of x.
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Runs vjp of grad of sin at x and returns the vjp function
            # applied to y.
            out, vjp_fn = vjp(grad(torch.sin), x)
            return vjp_fn(y)[0]

        result = foo(x, y)
        expected = -y * x.sin()
        self.assertEqual(result, expected)

    def test_grad_of_vjp_of_grad_composition(self, device):
        # [FUNCTION DOC] Checks a three-level composition (grad of vjp of grad of negative cosine)
        # gives cosine of x.
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Runs vjp of the grad of negative cosine at x, then returns
            # grad of the vjp function applied to y.
            df, vjp_fn = vjp(grad(lambda x: -torch.cos(x)), x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_views(self, device):
        # [FUNCTION DOC] Checks that grad inside a function of regular-autograd inputs works with
        # view ops, then differentiates the result with autograd to get minus sin of each input.
        x = torch.randn([], requires_grad=True, device=device)
        y = torch.randn([], requires_grad=True, device=device)

        def silly_sin(x):
            # [FUNCTION DOC] HELPER: Views the input as a scalar and applies sin.
            x = x.view([])
            x = x.sin()
            return x

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Sums the grad of silly_sin at x and the cosine of y.
            z1 = grad(silly_sin)(x)
            z2 = torch.cos(y)
            return z1 + z2

        result = foo(x, y)
        grads = torch.autograd.grad(result, [x, y])
        self.assertEqual(grads[0], -x.sin())
        self.assertEqual(grads[1], -y.sin())

    def test_view_inplace_simple(self, device):
        # [FUNCTION DOC] Checks grad through an in-place sin applied to a scalar view of a clone
        # gives cosine.
        def foo(x):
            # [FUNCTION DOC] HELPER: Clones the input, applies in-place sin on a scalar view and
            # returns the clone.
            x = x.clone()
            x.view([]).sin_()
            return x

        x = torch.randn([], requires_grad=True, device=device)
        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_invalid_argnums(self, device):
        # [FUNCTION DOC] Checks grad rejects invalid argnums with RuntimeError: out of range ints,
        # a list, non-int entries, and duplicate indices (including via negative index), matching
        # expected message fragments.
        x = torch.randn([])
        y = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "but only"):
            grad(torch.mul, argnums=-3)(x, y)
        with self.assertRaisesRegex(RuntimeError, "but only"):
            grad(torch.mul, argnums=2)(x, y)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            grad(torch.mul, argnums=[0])(x, y)
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            grad(torch.mul, argnums=("0",))(x, y)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            grad(torch.mul, argnums=(0, 0))(x, y)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            grad(torch.mul, argnums=(0, -2))(x, y)

    def test_argnums(self, device):
        # [FUNCTION DOC] Checks grad with int and tuple argnums on a product returns the other
        # operand's value (single or tuple results).
        x = torch.randn([])
        y = torch.randn([])
        gx = grad(torch.mul, argnums=0)(x, y)
        self.assertEqual(gx, y)

        gy = grad(torch.mul, argnums=1)(x, y)
        self.assertEqual(gy, x)

        (gx,) = grad(torch.mul, argnums=(0,))(x, y)
        self.assertEqual(gx, y)

        gx, gy = grad(torch.mul, argnums=(0, 1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_out_of_order_argnums(self, device):
        # [FUNCTION DOC] Checks that tuple argnums in reversed order return gradients in
        # that order.
        x = torch.randn([])
        y = torch.randn([])
        gy, gx = grad(torch.mul, argnums=(1, 0))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_negative_argnums(self, device):
        # [FUNCTION DOC] Checks negative int and tuple argnums select the same arguments as their
        # positive equivalents.
        x = torch.randn([])
        y = torch.randn([])
        gx = grad(torch.mul, argnums=-2)(x, y)
        self.assertEqual(gx, y)

        gy = grad(torch.mul, argnums=-1)(x, y)
        self.assertEqual(gy, x)

        (gx,) = grad(torch.mul, argnums=(-2,))(x, y)
        self.assertEqual(gx, y)

        gx, gy = grad(torch.mul, argnums=(-2, -1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_grad_pytree_inputs(self, device):
        # [FUNCTION DOC] Checks grad with a pytree input: default argnums returns the gradient
        # pytree of the first argument, a one-element tuple returns it in a tuple, and both
        # argnums return a pair containing the dict gradient with expected coefficients.
        x = torch.randn([], device=device)

        def f(a, b):
            # [FUNCTION DOC] HELPER: Unpacks a pair from the first argument and returns one times
            # the first plus two times the second plus three times the value under the foo key of
            # the second argument.
            x, y = a
            return 1 * x + 2 * y + 3 * b["foo"]

        args = ((x, x), {"foo": x})

        gx, gy = grad(f)(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))

        ((gx, gy),) = grad(f, argnums=(0,))(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))

        (gx, gy), gz = grad(f, argnums=(0, 1))(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))
        self.assertEqual(gz["foo"], torch.tensor(3.0, device=device))

    def test_grad_aux_tensor(self, device):
        # [FUNCTION DOC] Checks grad with has_aux: a function returning a non-tuple or a tuple of
        # the wrong length raises RuntimeError about the output needing to be a tuple; a valid
        # function returns the gradient and the auxiliary output.
        x = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            r"grad_and_value\(f\)\(\*args\): output of function f should be a tuple",
        ):
            grad(lambda t: [t, t], has_aux=True)(x)

        with self.assertRaisesRegex(
            RuntimeError,
            r"grad_and_value\(f\)\(\*args\): output of function f should be a tuple",
        ):
            grad(lambda t: (t, t + 2, t + 3), has_aux=True)(x)

        def f(t):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input and the cosine as the
            # auxiliary value.
            y = t.sin()
            return y.sum(), t.cos()

        out, aux = grad(f, has_aux=True)(x)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.cos())

    def test_grad_aux_pytree(self, device):
        # [FUNCTION DOC] Checks that with has_aux the aux output may be a pytree of tensors, and
        # that non-tensor aux values (int, float, string, or lists containing them) raise
        # RuntimeError about unsupported types.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input plus an aux dict holding
            # cosine and a list with tangent.
            y = x.sin()
            return y.sum(), {"a": x.cos(), "b": [x.tan()]}

        x = torch.randn(3, device=device)

        out, aux = grad(f, has_aux=True)(x)
        _, expected_aux = f(x)
        self.assertEqual(aux, expected_aux)
        self.assertEqual(out, x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = grad(lambda x: (x.sum(), aux), has_aux=True)(x)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = grad(lambda x: (x.sum(), [x, aux]), has_aux=True)(x)

    def test_zero_grad(self, device):
        # [FUNCTION DOC] Checks that grad over a dict input returns zero for entries the function
        # does not use and nonzero for used ones.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the squares of the dict entry a.
            return (x["a"] ** 2.0).sum()

        inps = {
            "a": torch.randn(10, device=device) + 3,
            "b": torch.randn(10, device=device),
        }
        grads = grad(f)(inps)
        self.assertNotEqual(grads["a"].sum(), 0.0)
        self.assertEqual(grads["b"].sum(), 0.0)

    def test_unrelated_grad(self, device):
        # [FUNCTION DOC] Checks grad of a function returning an unrelated captured tensor gives
        # zeros like the input.
        x = torch.tensor(1.0, device=device)
        y = torch.tensor(2.0, device=device)

        def unrelated(x):
            # [FUNCTION DOC] HELPER: Ignores its input and returns the captured tensor y.
            return y

        result = grad(unrelated)(x)
        self.assertEqual(result, torch.zeros_like(x))

    def test_unrelated_vjp(self, device):
        # [FUNCTION DOC] Checks vjp of a function returning an unrelated captured tensor gives a
        # tuple with zeros like the input.
        x = torch.tensor(1.0, device=device)
        y = torch.tensor(2.0, device=device)
        v = torch.tensor(1.0, device=device)

        def unrelated(x):
            # [FUNCTION DOC] HELPER: Ignores its input and returns the captured tensor y.
            return y

        out, vjp_fn = vjp(unrelated, x)
        result = vjp_fn(v)
        expected = (torch.zeros_like(x),)
        self.assertEqual(result, expected)

    def test_unrelated_vjp_multiple_inputs_outputs(self, device):
        # [FUNCTION DOC] Checks vjp with two inputs and three outputs where outputs are unrelated
        # captured tensors or one input: gradient for the unused input is zero and for the
        # passed-through input is ones.
        w = torch.tensor(3.0, device=device)
        x = torch.tensor(4.0, device=device)
        y = torch.tensor(2.0, device=device)
        v = torch.tensor(1.0, device=device)

        def unrelated(w, x):
            # [FUNCTION DOC] HELPER: Returns the captured tensor twice and the second input.
            return y, y, x

        out, vjp_fn = vjp(unrelated, w, x)
        result = vjp_fn((v, v, v))
        expected = (torch.zeros_like(x), torch.ones_like(x))
        self.assertEqual(result, expected)

    # TODO: https://github.com/pytorch/functorch/issues/12
    @onlyCPU
    def test_unrelated_hessian(self, device):
        # [FUNCTION DOC] CPU only (see functorch issue 12). Checks jacrev of jacrev of a linear
        # function with a captured matrix, an unrelated hessian, gives a zero tensor of shape
        # output by input by input.
        N = 5
        M = 3
        W = torch.randn(N, M, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the captured matrix by the input vector.
            return W @ x

        x = torch.randn(M)
        result = jacrev(jacrev(f))(x)
        expected = torch.zeros(N, M, M, device=device)
        self.assertEqual(result, expected)

    def test_vjp_pytree_input(self, device):
        # [FUNCTION DOC] Checks vjp with a nested tuple input: output is the product of two
        # entries, and the vjp function returns a one-element tuple mirroring the input structure,
        # with the cotangent-weighted gradients and a plain zero for the unused leaf.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the product of the first element and the first
            # element of the nested second element of its tuple input.
            return x[0] * x[1][0]

        x = torch.randn([], device=device)
        v = torch.randn([], device=device)
        out, vjp_fn = vjp(f, (x, (x, x)))
        self.assertEqual(out, x * x)
        result = vjp_fn(v)
        self.assertEqual(result, ((x * v, (x * v, 0.0)),))

    def test_vjp_pytree_output(self, device):
        # [FUNCTION DOC] Checks vjp with a function returning a nested tuple output: the vjp
        # function accepts a matching nested tuple of cotangents and returns the sum of them for
        # the single input.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input and a tuple of the input twice.
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        (result,) = vjp_fn((v1, (v2, v3)))
        self.assertEqual(result, v1 + v2 + v3)

    def test_vjp_outputs_can_any_pytree(self, device):
        # [FUNCTION DOC] Checks which output types vjp accepts. None and empty tuple outputs raise
        # RuntimeError about non-empty output; scalars, bool, float and string outputs raise
        # RuntimeError about returning only tensors. List, dict and deeply nested list/tuple/dict
        # outputs work, the output keeps its structure, and the vjp function accepts cotangents of
        # the same structure and returns a tensor.
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"vjp\(f, \*primals\): Expected f to be a function that has non-empty output",
            ):
                _, vjp_fn = vjp(lambda _: output, x)
                vjp_fn(t)

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"vjp\(f, \*primals\): expected f\(\*primals\) to return only tensors",
            ):
                _, vjp_fn = vjp(lambda _: output, x)
                vjp_fn(t)

        # Check list output
        output, vjp_fn = vjp(lambda x: [x, x.sum()], x)
        (vjp_out,) = vjp_fn([t, t.sum()])
        if not isinstance(output, list) or len(output) != 2:
            raise AssertionError(f"Expected list of length 2, got {type(output)}")
        if not isinstance(vjp_out, torch.Tensor):
            raise AssertionError(f"Expected Tensor, got {type(vjp_out)}")

        # Check dict output
        output, vjp_fn = vjp(lambda x: {"x": x, "xsum": x.sum()}, x)
        (vjp_out,) = vjp_fn({"x": t, "xsum": t.sum()})
        if not isinstance(output, dict) or len(output) != 2 or "xsum" not in output:
            raise AssertionError(f"Expected dict with 'xsum', got {output}")
        if not isinstance(vjp_out, torch.Tensor):
            raise AssertionError(f"Expected Tensor, got {type(vjp_out)}")

        def composite_output(x):
            # [FUNCTION DOC] HELPER: Returns a list holding a tuple of the sum and a dict
            # containing the input and a list of the input and the sum.
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        output, vjp_fn = vjp(composite_output, x)
        (vjp_out,) = vjp_fn(
            [
                (t.sum(), {"a": t, "out": [t, t.sum()]}),
            ]
        )
        if not isinstance(output, list):
            raise AssertionError(f"Expected list, got {type(output)}")
        if not isinstance(output[0], tuple) or not isinstance(output[0][1], dict):
            raise AssertionError(f"Expected tuple with dict, got {output[0]}")
        if not isinstance(vjp_out, torch.Tensor):
            raise AssertionError(f"Expected Tensor, got {type(vjp_out)}")

    def test_vjp_pytree_error(self, device):
        # [FUNCTION DOC] Checks that passing the vjp function a cotangent with a different pytree
        # structure than the output raises RuntimeError saying a pytree structure was expected.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input and a tuple of the input twice.
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        with self.assertRaisesRegex(RuntimeError, "Expected pytree structure"):
            (result,) = vjp_fn(((v1, (v2, v3)),))

    def test_vjp_aux_tensor(self, device):
        # [FUNCTION DOC] Checks vjp with has_aux: a function returning a non-tuple or wrong length
        # tuple raises RuntimeError about the output needing to be a tuple; a valid function
        # returns output, vjp function and aux, and the vjp function gives cotangent times cosine.
        x = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError, r"vjp\(f, \*primals\): output of function f should be a tuple"
        ):
            vjp(lambda t: [t, t], x, has_aux=True)

        with self.assertRaisesRegex(
            RuntimeError, r"vjp\(f, \*primals\): output of function f should be a tuple"
        ):
            vjp(lambda t: (t, t + 2, t + 3), x, has_aux=True)

        def f(t):
            # [FUNCTION DOC] HELPER: Returns sin of the input as output and cosine as the
            # auxiliary value.
            y = t.sin()
            return y, t.cos()

        out, vjp_fn, aux = vjp(f, x, has_aux=True)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.sin())

        v = torch.randn(3, device=device)
        (grad_x,) = vjp_fn(v)
        self.assertEqual(grad_x, v * x.cos())

    def test_vjp_aux_pytree(self, device):
        # [FUNCTION DOC] Checks vjp with a pytree (dict containing list) aux output returns it
        # unchanged, gradients are correct, and non-tensor aux values (int, float, string, or
        # lists containing them) raise RuntimeError about unsupported types.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns sin of the input and an aux dict with cosine and a
            # list holding tangent.
            y = x.sin()
            return y, {"a": x.cos(), "b": [x.tan()]}

        x = torch.randn(3, device=device)

        out, vjp_fn, aux = vjp(f, x, has_aux=True)
        expected_out, expected_aux = f(x)
        self.assertEqual(out, expected_out)
        self.assertEqual(aux, expected_aux)

        v = torch.randn(3, device=device)
        (grad_x,) = vjp_fn(v)
        self.assertEqual(grad_x, v * x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = vjp(lambda x: (x, aux), x, has_aux=True)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = vjp(lambda x: (x, [x, aux]), x, has_aux=True)

    def test_functional_init(self, device):
        # [FUNCTION DOC] Checks that functional_init, given a module class, a batch size and
        # device, builds batch-many stacked weights; the resulting functional model runs under
        # vmap on a batch of inputs without error (no output assertion).
        class MLPClassifier(nn.Module):
            # [CLASS DOC] HELPER: Small two-layer perceptron classifier (linear, relu, linear, log
            # softmax) from 2 input features to n_classes, with hidden width hidden_dim.
            def __init__(self, hidden_dim=32, n_classes=2):
                # [FUNCTION DOC] HELPER: Stores hidden_dim and n_classes and creates two linear
                # layers, the first from 2 to hidden_dim and the second from hidden_dim
                # to n_classes.
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first linear layer, relu, the second linear
                # layer and log softmax over the last dim; returns the result.
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, fn, _ = functional_init(MLPClassifier, (B,), device=device)(32, 2)
        inputs = torch.randn(B, 7, 2, device=device)
        vmap(fn)(weights, (inputs,))

    def test_functional_init_with_buffers(self, device):
        # [FUNCTION DOC] Checks that functional_init_with_buffers (batch size given as a list)
        # builds stacked weights and buffers for a model containing batch norm, and that the
        # functional model runs under vmap with weights, buffers and inputs (no output assertion).
        class MLPClassifier(nn.Module):
            # [CLASS DOC] HELPER: Perceptron classifier like the plain one but with a
            # one-dimensional batch norm layer (affine) between the first relu and the second
            # linear layer, so it has buffers.
            def __init__(self, hidden_dim=32, n_classes=2):
                # [FUNCTION DOC] HELPER: Stores sizes and creates the first linear layer, a batch
                # norm over the hidden width, and the output linear layer.
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.bn = nn.BatchNorm1d(self.hidden_dim, affine=True)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, relu, batch norm, linear and log softmax
                # over the last dim; returns the result.
                x = self.fc1(x)
                x = F.relu(x)
                x = self.bn(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, buffers, fn, _, _ = functional_init_with_buffers(
            MLPClassifier, [B], device=device
        )(32, 2)
        inputs = torch.randn(B, 7, 2, device=device)
        vmap(fn)(weights, buffers, (inputs,))

    def test_advanced_indexing(self, device):
        # [FUNCTION DOC] Checks grad works with boolean-mask indexed assignment: a function that
        # mutates an unrelated tensor with a false mask returns the input and gets gradient one; a
        # function that zeros the positive entries of a clone and sums gives gradient one at
        # non-positive entries and zero elsewhere.
        def f(value):
            # [FUNCTION DOC] HELPER: Assigns zero into an all-ones scalar through an all-false
            # mask, then returns the input unchanged.
            log_prob = torch.ones((), device=device)
            val = torch.zeros(()) > 0
            log_prob[val] = 0
            return value

        result = grad(f)(torch.randn((), device=device))
        self.assertEqual(result, torch.ones_like(result))

        def f2(value):
            # [FUNCTION DOC] HELPER: Clones the input, zeros entries greater than zero via a mask
            # assignment and returns the sum.
            value = value.clone()
            value[value > 0] = 0
            return value.sum()

        x = torch.randn(100, device=device)
        result = grad(f2)(x)
        self.assertEqual(result, (x <= 0).type_as(x))

    def test_tensor_ctor_inside_grad(self, device):
        # [FUNCTION DOC] Checks that creating a new tensor with the torch.tensor constructor
        # inside a function under grad works (runs without assertion on the result).
        def foo(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by a freshly created scalar tensor
            # holding two.
            return x * torch.tensor(2.0, device=device)

        x = torch.tensor(3.14, device=device)
        functorch.grad(foo)(x)

    @parametrize(
        "op_list_data",
        [
            subtest(
                (
                    [
                        vmap,
                    ],
                    [(4, 2), (64, 3, 32, 32)],
                ),
                name="vmap",
            ),
            subtest(([vmap, vmap], [(4, 3, 2), (64, 3, 32, 32)]), name="vmap_vmap"),
            subtest(
                (
                    [
                        grad,
                    ],
                    [(0,), [], (4, 2), (64, 3, 32, 32)],
                ),
                name="grad",
            ),
            subtest(
                (
                    [grad, grad],
                    [
                        [],
                    ],
                ),
                name="grad_grad",
            ),
            subtest(([vmap, grad], [(4, 2)]), name="vmap_grad"),
        ],
    )
    def test_tensor_print(self, device, op_list_data):
        # [FUNCTION DOC] Parametrized over combinations of vmap and grad transforms and input
        # shapes. For every floating dtype and shape it builds the nested transform (vmap with
        # increasing in_dims) around a function that records the repr of the tensor it receives in
        # an outer variable, runs it, and validates the recorded string via the wrapper
        # repr helper.
        op_list, shapes = op_list_data

        for dt in get_all_fp_dtypes():
            data = [torch.randn(s, dtype=dt, device=device) for s in shapes]

            for x in data:
                buf = None

                def foo(t):
                    # [FUNCTION DOC] HELPER: Stores the repr of its argument in the enclosing
                    # variable and returns the mean.
                    nonlocal buf
                    buf = repr(t)
                    return t.mean()

                fn = foo
                bdim = 0
                for op in reversed(op_list):
                    if op is vmap:
                        fn = op(fn, in_dims=bdim)
                        bdim += 1
                    else:
                        fn = op(fn)

                fn(x)
                self.assertIsNotNone(buf)
                _assert_functorch_wrapper_repr(self, buf, op_list)

    def test_print_captured_tensor_inside_transform(self, device):
        # [FUNCTION DOC] Checks the repr of a captured outer tensor printed inside vjp: under
        # Dynamo it must look like a plain leaf tensor, otherwise it must equal the repr
        # taken outside.
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        out = None

        def f(y):
            # [FUNCTION DOC] HELPER: Stores the repr of the captured tensor in the enclosing
            # variable and returns its argument.
            nonlocal out
            out = repr(x)
            return y

        vjp(f, torch.randn(4, device=device))
        self.assertIsNotNone(out)
        if TEST_WITH_TORCHDYNAMO:
            _assert_leaf_tensor_repr(self, out)
        else:
            self.assertEqual(out, repr(x))

    def test_no_grad_outside(self, device):
        # [FUNCTION DOC] Checks grad called inside a no_grad block still computes the derivative
        # (cosine) but the result does not require grad.
        x = torch.randn([], device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(torch.sin)(x)
        self.assertEqual(y, x.cos())
        self.assertFalse(y.requires_grad)

    def test_no_grad_inside(self, device):
        # [FUNCTION DOC] Checks no_grad inside the differentiated function: the shifted quantity
        # is treated as constant, giving first derivative two x, second derivative two, and a
        # double derivative via regular autograd of two.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes a square under no_grad as a shift and returns the
            # square minus that shift.
            with torch.no_grad():
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device)
        y = grad(f)(x)
        self.assertEqual(y, 2 * x)
        y = grad(grad(f))(x)
        self.assertEqual(y, 2)

        x = torch.randn([], device=device, requires_grad=True)
        y = grad(f)(x)
        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 2)

    def test_no_grad_mixed(self, device):
        # [FUNCTION DOC] Checks no_grad inside the function combined with an outer no_grad: the
        # gradient is still two x but does not require grad.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes a square under no_grad as a shift and returns the
            # square minus that shift.
            with torch.no_grad():
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(f)(x)

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)

    def test_no_grad_nested_simple(self, device):
        # [FUNCTION DOC] Checks a nested grad evaluated under no_grad is treated as a constant:
        # outer grad gives three x squared and regular autograd of that gives six x.
        def h(x):
            # [FUNCTION DOC] HELPER: Computes the grad of a quartic under no_grad as a shift and
            # returns the cube minus the shift.
            with torch.no_grad():
                shift = grad(lambda x: 0.25 * x**4)(x)
            return x**3 - shift

        x = torch.tensor(1.5, device=device, requires_grad=True)
        y = grad(h)(x)
        self.assertEqual(y, 3 * x**2)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 6 * x)

    def test_no_grad_nested_complicated(self, device):
        # [FUNCTION DOC] Checks a combination of nested grad calls, some under no_grad, so that
        # only the cube contributes: gradient is six x and its autograd derivative is six.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the cube minus a no_grad copy of the cube.
            with torch.no_grad():
                shift = x**3
            return x**3 - shift

        def g(x):
            # [FUNCTION DOC] HELPER: Returns the grad of the function above at x minus the same
            # grad computed under no_grad.
            r1 = grad(f)(x)
            with torch.no_grad():
                shift = grad(f)(x)
            return r1 - shift

        x = torch.randn([], requires_grad=True, device=device)
        y = grad(g)(x)
        # The only differential part of g is x ** 3
        self.assertEqual(y, 6 * x)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 6)

    def test_no_grad_value(self, device):
        # [FUNCTION DOC] Checks the value returned by grad_and_value under no_grad is treated as
        # constant: outer grad gives three x squared and autograd of that gives six x.
        def h(x):
            # [FUNCTION DOC] HELPER: Computes the value of a cube via grad_and_value under no_grad
            # and returns the cube minus that value.
            with torch.no_grad():
                gvalue, value = grad_and_value(lambda x: x**3)(x)
            return x**3 - value

        x = torch.tensor(1.6, device=device, requires_grad=True)
        y = grad(h)(x)
        self.assertEqual(y, 3 * x**2)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 6 * x)

    def test_no_grad_outside_vjp(self, device):
        # [FUNCTION DOC] Checks vjp called and applied under no_grad: gradient is two x and
        # neither the gradient nor the output requires grad.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the square of the input.
            return x**2

        x = torch.tensor(2.0, requires_grad=True, device=device)
        with torch.no_grad():
            out, vjp_fn = vjp(h, x)
            (y,) = vjp_fn(torch.tensor(1.0, device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(out.requires_grad)

    def test_no_grad_outside_vjp_fn(self, device):
        # [FUNCTION DOC] Checks that when only the vjp function call is under no_grad, the
        # gradient does not require grad but the output does, and autograd can still differentiate
        # the output.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the square of the input.
            return x**2

        x = torch.tensor(3.14, requires_grad=True, device=device)
        out, vjp_fn = vjp(h, x)
        with torch.no_grad():
            (y,) = vjp_fn(torch.tensor(1.0, device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)
        self.assertTrue(out.requires_grad)

        (z,) = torch.autograd.grad(out, x)
        self.assertEqual(z, 2 * x)

    def test_no_grad_outside_vjp_only(self, device):
        # [FUNCTION DOC] Checks that when only the vjp call itself is under no_grad, the output
        # does not require grad while the gradient computed later does (noted in the test as odd),
        # and autograd gives two.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the square of the input.
            return x**2

        x = torch.tensor(3.14, requires_grad=True, device=device)
        with torch.no_grad():
            out, vjp_fn = vjp(h, x)
        (y,) = vjp_fn(torch.tensor(1.0, device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(out.requires_grad)

        # This one is a little weird...
        self.assertTrue(y.requires_grad)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 2)

    @skipIfTorchDynamo("internal API test")
    def test_pop_dynamic_layer_stack_to_depth_single(self, device):
        # [FUNCTION DOC] Internal API test (skipped under Dynamo). Increments grad nesting, checks
        # the dynamic layer stack depth is one, pops back to depth zero and checks the depth
        # is zero.
        ft = torch._C._functorch
        ft._grad_increment_nesting()
        self.assertEqual(ft.get_dynamic_layer_stack_depth(), 1)
        ft.pop_dynamic_layer_stack_and_undo_to_depth(0)
        self.assertEqual(ft.get_dynamic_layer_stack_depth(), 0)

    @skipIfTorchDynamo("internal API test")
    def test_pop_dynamic_layer_stack_to_depth_mixed(self, device):
        # [FUNCTION DOC] Internal API test (skipped under Dynamo). Pushes vmap, grad and jvp
        # layers (depth three), pops down to depth two removing exactly one layer, then pops to
        # zero. Does not decrement nesting explicitly, relying on the pop function to undo layers.
        ft = torch._C._functorch
        ft._vmap_increment_nesting(3, "error")
        ft._grad_increment_nesting()
        ft._jvp_increment_nesting()
        self.assertEqual(ft.get_dynamic_layer_stack_depth(), 3)
        # Pop only jvp — must remove exactly one layer
        ft.pop_dynamic_layer_stack_and_undo_to_depth(2)
        self.assertEqual(ft.get_dynamic_layer_stack_depth(), 2)
        # Pop remaining
        ft.pop_dynamic_layer_stack_and_undo_to_depth(0)
        self.assertEqual(ft.get_dynamic_layer_stack_depth(), 0)

    def test_inference_mode_outside_grad(self, device):
        # [FUNCTION DOC] Checks grad run inside inference mode gives two x for a sum of squares.
        x = torch.randn(3, device=device)
        with torch.inference_mode():
            y = grad(lambda x: (x**2).sum())(x)
        self.assertEqual(y, 2 * x)

    def test_inference_mode_nograd_outside_grad(self, device):
        # [FUNCTION DOC] Checks grad run inside inference mode combined with no_grad gives two x
        # for a sum of squares.
        x = torch.randn(3, device=device)
        with torch.inference_mode():
            with torch.no_grad():
                y = grad(lambda x: (x**2).sum())(x)
        self.assertEqual(y, 2 * x)

    def test_inference_mode_outside_vjp(self, device):
        # [FUNCTION DOC] Checks vjp and its vjp function used inside inference mode give two x.
        x = torch.randn(3, device=device)
        with torch.inference_mode():
            out, vjp_fn = vjp(lambda x: (x**2).sum(), x)
            (y,) = vjp_fn(torch.tensor(1.0, device=device))
        self.assertEqual(y, 2 * x)

    def test_inference_mode_outside_jvp(self, device):
        # [FUNCTION DOC] Checks jvp inside inference mode returns the directional derivative, the
        # sum of two x times the tangent.
        x = torch.randn(3, device=device)
        t = torch.ones(3, device=device)
        with torch.inference_mode():
            _, y = jvp(lambda x: (x**2).sum(), (x,), (t,))
        self.assertEqual(y, (2 * x * t).sum())

    def test_inference_mode_outside_jacrev(self, device):
        # [FUNCTION DOC] Checks jacrev of an elementwise square inside inference mode gives the
        # diagonal matrix of two x.
        x = torch.randn(3, device=device)
        with torch.inference_mode():
            y = jacrev(lambda x: x**2)(x)
        self.assertEqual(y, torch.diag(2 * x))

    def test_inference_mode_outside_vmap_grad(self, device):
        # [FUNCTION DOC] Checks vmap of grad inside inference mode gives two x for each row.
        xs = torch.randn(5, 3, device=device)
        with torch.inference_mode():
            ys = vmap(grad(lambda x: (x**2).sum()))(xs)
        self.assertEqual(ys, 2 * xs)

    def test_inference_mode_outside_grad_vmap(self, device):
        # [FUNCTION DOC] Checks grad over a function that internally uses vmap, inside inference
        # mode, gives two x.
        x = torch.randn(3, device=device)
        with torch.inference_mode():
            y = grad(lambda x: vmap(lambda x: (x**2).sum())(x).sum())(x)
        self.assertEqual(y, 2 * x)

    def test_inference_mode_nested_grad(self, device):
        # [FUNCTION DOC] Checks nested grad of a cube inside inference mode gives six x.
        x = torch.randn([], device=device)
        with torch.inference_mode():
            y = grad(grad(lambda x: x**3))(x)
        self.assertEqual(y, 6 * x)

    def test_inference_mode_jacrev_grad(self, device):
        # [FUNCTION DOC] Checks jacrev of grad (a hessian) of a sum of cubes inside inference mode
        # gives the diagonal matrix of six x.
        x = torch.randn(3, device=device)
        with torch.inference_mode():
            H = jacrev(grad(lambda x: (x**3).sum()))(x)
        self.assertEqual(H, torch.diag(6 * x))

    def test_inference_mode_inside_grad(self, device):
        # [FUNCTION DOC] Checks that using inference mode inside the differentiated function,
        # while also in an outer inference mode, treats the computed value as constant so the
        # gradient is ones.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes a square in inference mode and returns the input
            # minus it.
            with torch.inference_mode():
                c = x**2
            return x - c

        x = torch.randn(3, device=device)
        with torch.inference_mode():
            y = grad(lambda x: f(x).sum())(x)
        self.assertEqual(y, torch.ones_like(x))

    def test_inference_mode_restored(self, device):
        # [FUNCTION DOC] Checks inference mode state is reported as disabled before, enabled
        # inside, still enabled after running grad inside it, and disabled after leaving
        # the block.
        self.assertTrue(not torch.is_inference_mode_enabled())
        with torch.inference_mode():
            self.assertTrue(torch.is_inference_mode_enabled())
            grad(lambda x: (x**2).sum())(torch.randn(3, device=device))
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertTrue(not torch.is_inference_mode_enabled())


@markDynamoStrictTest
class TestAutogradFunction(TestCase):
    # [CLASS DOC] Device-generic tests for functorch interaction with custom
    # torch.autograd.Function: dead wrapper unwrapping, materialize grads, functions that return
    # their input, needs_input_grad and once_differentiable.
    @skipIfTorchDynamo("internal API test")
    def test_unwrap_dead_wrappers(self, device):
        # [FUNCTION DOC] Internal API test (skipped under Dynamo) of the utility that unwraps dead
        # grad wrappers in an argument tuple. Verifies the same tuple object is returned for empty
        # tuples, non-tensor tuples and tuples with only live tensors, and a new tuple with only
        # the dead wrapper replaced by the underlying tensor (other items identical objects) for a
        # dead wrapper at each of three positions.
        ft = torch._C._functorch
        unwrap_dead_wrappers = torch._functorch.utils.unwrap_dead_wrappers

        def make_dead_wrapper(tensor):
            # [FUNCTION DOC] HELPER: Creates a grad level, wraps the tensor for it, leaves the
            # level (always, via cleanup) so the wrapper is dead, asserts it is reported dead and
            # returns it.
            level = ft._grad_increment_nesting()
            try:
                wrapped = ft._wrap_for_grad(tensor, level)
            finally:
                ft._grad_decrement_nesting()
            self.assertTrue(ft.is_dead_tensor_wrapper(wrapped))
            return wrapped

        empty = ()
        self.assertIs(unwrap_dead_wrappers(empty), empty)

        non_tensors = (None, 1, "arg")
        self.assertIs(unwrap_dead_wrappers(non_tensors), non_tensors)

        live_tensor = torch.randn(2, device=device)
        mixed_live = ("arg", live_tensor, None)
        self.assertIs(unwrap_dead_wrappers(mixed_live), mixed_live)

        for dead_idx in range(3):
            live_before = torch.randn(2, device=device)
            live_after = torch.randn(2, device=device)
            base = torch.randn(2, device=device)
            dead = make_dead_wrapper(base)
            args = [live_before, "arg", live_after]
            args[dead_idx] = dead
            args = tuple(args)

            result = unwrap_dead_wrappers(args)
            self.assertIsNot(result, args)
            self.assertEqual(result[dead_idx], base)
            self.assertFalse(ft.is_dead_tensor_wrapper(result[dead_idx]))
            for idx in range(3):
                if idx != dead_idx:
                    self.assertIs(result[idx], args[idx])

    def test_set_materialize_grads(self, device):
        # [FUNCTION DOC] Checks that a custom Function that disables materialization of grads in
        # its setup receives a real gradient for the output that is used and None for the unused
        # one, under grad and nested grad.
        class A(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd Function that returns both inputs, turns off
            # gradient materialization in setup_context, and whose backward asserts the first
            # gradient is present and the second is None.
            @staticmethod
            def forward(x, y):
                # [FUNCTION DOC] HELPER: Returns both inputs unchanged.
                return x, y

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Disables materialization of undefined gradients on
                # the context.
                ctx.set_materialize_grads(False)

            @staticmethod
            def backward(ctx, gx, gy):
                # [FUNCTION DOC] HELPER: Asserts via the enclosing test that the first gradient is
                # not None and the second is None, then returns both.
                self.assertIsNotNone(gx)
                self.assertIsNone(gy)
                return gx, gy

        def f(y, x):
            # [FUNCTION DOC] HELPER: Applies the custom function to the two inputs (note argument
            # order is swapped relative to grad's differentiation arg) and returns the square of
            # the first result.
            x, y = A.apply(x, y)
            return x**2

        x = torch.tensor(2.0, device=device)
        y = torch.tensor(3.0, device=device)
        # grad differentiates w.r.t. arg 0 by default
        grad(f)(y, x)
        grad(grad(f))(y, x)

    @parametrize("inner_requires_grad", [True, False])
    @parametrize("save_for", ["jvp", "vjp"])
    @parametrize("save_tensors", ["input", "output", "neither"])
    @parametrize("mark_dirty", [True, False])
    def test_function_returns_input(
        self, device, inner_requires_grad, save_for, save_tensors, mark_dirty
    ):
        # [FUNCTION DOC] Parametrized over whether the inner input requires grad, whether tensors
        # are saved for forward or backward, which tensor is saved (input, output or neither) and
        # whether the input is marked dirty. Checks that a custom Function returning its input
        # as-is raises RuntimeError ("An input that has been returned as-is") under grad and jvp
        # when it saves the input or output without marking it dirty, and otherwise works; also
        # checks direct application and forward-mode dual inputs, and that the output is the same
        # object when marked dirty (skipping one dual case tracked by issue 97827).
        class A(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose forward returns its input unchanged. Its
            # setup_context saves the input, output or nothing for either forward or backward,
            # optionally marking the input dirty, as chosen by the enclosing test parameters;
            # backward returns the output gradient and jvp returns a modified (dirty) or
            # viewed tangent.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Returns the input as-is.
                return x

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Saves the input, output or nothing, using the forward or
                # backward saving method and optionally marking the input dirty, per the enclosing
                # test parameters.
                if save_for == "jvp":
                    save_fn = ctx.save_for_forward
                else:
                    save_fn = ctx.save_for_backward

                if mark_dirty:
                    ctx.mark_dirty(inputs[0])

                if save_tensors == "input":
                    save_fn(inputs[0])
                elif save_tensors == "output":
                    save_fn(output)
                elif save_tensors == "neither":
                    pass

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns the output gradient unchanged.
                return grad_output

            @staticmethod
            def jvp(ctx, x_t):
                # [FUNCTION DOC] HELPER: Returns the tangent: with dirty marking an in-place add
                # of zero on it, otherwise a view of it.
                # NB: the logic to check ctx.save_for_forward happens
                #     before we reach this!
                if mark_dirty:
                    ret = x_t.add_(0)
                else:
                    ret = x_t.view_as(x_t)
                return ret

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the custom function to a clone of the input.
            return A.apply(x.clone())

        err_msg = "An input that has been returned as-is"

        a = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad)
        a_t = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad)
        if save_tensors in ("input", "output") and not mark_dirty:
            with self.assertRaisesRegex(RuntimeError, err_msg):
                grad(fn)(a)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                jvp(fn, (a,), (a_t,))
        else:
            grad(fn)(a)
            jvp(fn, (a,), (a_t,))

        a = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad).clone()
        a_t = torch.tensor(
            2.0, device=device, requires_grad=inner_requires_grad
        ).clone()

        if save_tensors in ("input", "output") and not mark_dirty:
            with self.assertRaisesRegex(RuntimeError, err_msg):
                A.apply(a)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                with fwAD.dual_level():
                    A.apply(fwAD.make_dual(a, a_t))
        else:
            b = A.apply(a)
            if mark_dirty:
                self.assertTrue(a is b)
            if not (
                mark_dirty and save_for == "vjp" and save_tensors in ("input", "output")
            ):
                # TODO(soulitzer): https://github.com/pytorch/pytorch/issues/97827
                with fwAD.dual_level():
                    a_dual = fwAD.make_dual(a, a_t)
                    b_dual = A.apply(a_dual)
                if mark_dirty:
                    self.assertTrue(a_dual is b_dual)

    def test_needs_input_grads(self, device):
        # [FUNCTION DOC] Checks that backward of a custom Function sees needs_input_grad true for
        # the first input and false for the second under grad and nested grad.
        class A(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function computing the product of two inputs, with an
            # empty context setup; its backward asserts needs_input_grad is true for the first and
            # false for the second input and returns None for both.
            @staticmethod
            def forward(x, y):
                # [FUNCTION DOC] HELPER: Returns the product of the inputs.
                return x * y

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                return

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Asserts via the enclosing test the needs_input_grad flags
                # and returns None for both gradients.
                self.assertTrue(ctx.needs_input_grad[0])
                self.assertFalse(ctx.needs_input_grad[1])
                return None, None

        x = torch.tensor(2.0, device=device)
        y = torch.tensor(3.0, device=device)
        # grad differentiates w.r.t. arg 0 by default
        grad(A.apply)(x, y)
        grad(grad(A.apply))(x, y)

    def _get_NumpyCubeNotComposable(self):
        # [FUNCTION DOC] HELPER: Defines and returns a custom Function class that cubes a tensor
        # using numpy (not composable) and returns the numpy array as a second output.
        class NumpyCubeNotComposable(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function cubing its input via numpy, returning the cubed
            # tensor and the numpy array. Its backward is marked once_differentiable and computes
            # three times the square of the saved numpy input.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Converts the input to numpy on the CPU and returns the
                # cube as a tensor on the original device plus the numpy array.
                input_np = input.cpu().numpy()
                return torch.tensor(input_np**3, device=input.device), input_np

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Stores the numpy array output and the input device on
                # the context.
                ctx.input_np = output[1]
                ctx.device = inputs[0].device

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad_output, grad_saved):
                # [FUNCTION DOC] HELPER: Returns three times the square of the saved numpy array
                # as a tensor on the saved device; ignores the incoming gradients.
                result_np = 3 * (ctx.input_np**2)
                return torch.tensor(result_np, device=ctx.device)

        return NumpyCubeNotComposable

    def test_once_differentiable_autograd_vjp(self, device):
        # [FUNCTION DOC] Checks that using regular autograd on the vjp function result of a
        # function using a once_differentiable custom Function raises RuntimeError saying it was
        # marked with once_differentiable when calling backward on the gradient.
        NumpyCubeNotComposable = self._get_NumpyCubeNotComposable()

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the not-composable cube function and returns the
            # first output.
            y, _ = NumpyCubeNotComposable.apply(x)
            return y

        # regular autograd x vjp
        x = torch.randn([], requires_grad=True, device=device)
        grad_y = torch.randn_like(x, requires_grad=True)
        _, vjp_fn = vjp(f, x)
        (gx,) = vjp_fn(grad_y)

        with self.assertRaisesRegex(RuntimeError, "marked with @once_differentiable"):
            gx.backward()

    # TODO: support torch.autograd.function.once_differentiable
    # (or, if impossible, figure out how to raise a nice error)
    # https://github.com/pytorch/pytorch/issues/90224
    @unittest.expectedFailure
    def test_once_differentiable_grad_vjp(self, device):
        # [FUNCTION DOC] Expected failure (unittest) with TODO linked to issue 90224:
        # once_differentiable is not supported for grad of vjp. KNOWN ISSUE: the helper below
        # calls a name f that is not defined in this test (flagged by the linter as undefined), so
        # the failure is expected regardless of the feature.
        # grad x vjp
        x = torch.randn([], device=device)
        grad_y = torch.randn_like(x)

        def h(x, grad_y):
            # [FUNCTION DOC] HELPER: Runs vjp of the undefined name f at x and returns the vjp
            # function applied to grad_y.
            _, vjp_fn = vjp(f, x)  # noqa: F821
            (gx,) = vjp_fn(grad_y)
            return gx

        grad(h, argnums=(0, 1))(x, grad_y)

    def test_grad_fn_name(self, device):
        # [FUNCTION DOC] Checks that a custom Function applied under grad produces an output whose
        # grad_fn class name is the Function name plus the suffix GeneratedBackward; the name is
        # recorded by the inner function into an outer list.
        names = []

        class FooBar(torch.autograd.Function):
            # [CLASS DOC] HELPER: Minimal custom Function that clones its input and passes the
            # gradient straight through.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Returns a clone of the input.
                return x.clone()

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                return

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns the output gradient unchanged.
                return grad_output

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the custom function, appends its grad_fn class name
            # to the outer names list and returns the result.
            y = FooBar.apply(x)
            names.append(type(y.grad_fn).__name__)
            return y

        x = torch.tensor(1.0)
        grad(f)(x)
        self.assertEqual(names, ["FooBarGeneratedBackward"])


@markDynamoStrictTest
class TestAutogradFunctionVmapAPI(TestCase):
    # [CLASS DOC] Device-generic tests for the vmap staticmethod protocol of custom autograd
    # Functions: missing or conflicting vmap support errors, the info object and in_dims
    # arguments, empty layers, None out dims and return validation.
    def test_no_vmap_staticmethod_and_no_generate_vmap_rule(self, device):
        # [FUNCTION DOC] Checks that vmap over a custom Function with neither a vmap staticmethod
        # nor generate_vmap_rule raises RuntimeError saying it does not have vmap support. KNOWN
        # ISSUE: the forward calls an undefined name to_numpy, but it is never reached since the
        # error is raised first.
        class NumpyCube(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function that cubes via numpy (helper name undefined
            # here), with no vmap rule; its backward always raises.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Intended to return the cube and derivative of the input
                # via numpy; refers to an undefined helper so it would fail if called (it is never
                # reached in the test).
                input_np = to_numpy(input)  # noqa: F821
                dinput = torch.tensor(3 * input_np**2, device=input.device)
                return torch.tensor(input_np**3, device=input.device), dinput

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Saves the inputs and the second output for backward.
                ctx.save_for_backward(inputs, output[1])

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("foobar")

        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "does not have vmap support"):
            vmap(NumpyCube.apply)(x)

    def test_has_vmap_staticmethod_and_has_generate_vmap_rule(self, device):
        # [FUNCTION DOC] Checks that a custom Function defining both a vmap staticmethod and the
        # generate_vmap_rule flag raises RuntimeError mentioning both when vmapped. KNOWN ISSUE:
        # forward refers to an undefined name to_numpy but is not reached.
        class NumpyCube(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function with generate_vmap_rule enabled and also a vmap
            # staticmethod, which is an invalid combination.
            generate_vmap_rule = True

            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Intended to return the cube and derivative of the input
                # via an undefined numpy helper; never reached.
                input_np = to_numpy(input)  # noqa: F821
                dinput = torch.tensor(3 * input_np**2, device=input.device)
                return torch.tensor(input_np**3, device=input.device), dinput

            @staticmethod
            def setup_context(ctx, outputs, input):
                # [FUNCTION DOC] HELPER: Saves the input and the second output for backward
                # (parameter names are swapped relative to the usual order).
                ctx.save_for_backward(input, outputs[1])

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("foobar")

            @staticmethod
            def vmap(infos, in_dims, x):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("foobar")

        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "generate_vmap_rule=True and"):
            vmap(NumpyCube.apply)(x)

    def test_info_object(self, device):
        # [FUNCTION DOC] Checks that the info object passed to a custom vmap staticmethod has the
        # batch size (ten) and the randomness mode, for each of the three modes error, different
        # and same.
        batch_size = 10

        class Id(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose vmap staticmethod asserts the batch size
            # and randomness on the info object against the enclosing test values and returns the
            # input with its first in_dim.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Does nothing (returns None).
                pass

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Asserts via the enclosing test that info batch size and
                # randomness equal the expected values, then returns the input and its
                # single in_dim.
                self.assertEqual(info.batch_size, batch_size)
                self.assertEqual(info.randomness, randomness)
                return input, in_dims[0]

        x = torch.randn(batch_size, 3, device=device)

        for randomness in ("error", "different", "same"):
            vmap(Id.apply, randomness=randomness)(x)

    def test_in_dims_single_input(self, device):
        # [FUNCTION DOC] Checks that for a single input the vmap staticmethod receives in_dims as
        # a one-element tuple, whether the caller gives an int or a one-element tuple.
        class Id(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose vmap staticmethod asserts the in_dims is a
            # tuple holding one and returns the input with its in_dim.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Asserts in_dims equals a one-element tuple holding one,
                # and returns the input with its in_dim.
                self.assertEqual(in_dims, (1,))
                return input, in_dims[0]

        B = 10
        x = torch.randn(3, B, device=device)
        vmap(Id.apply, in_dims=1)(x)
        vmap(Id.apply, in_dims=(1,))(x)

    def test_in_dims_multiple_inputs(self, device):
        # [FUNCTION DOC] Checks that with two inputs where the second is a list, the vmap
        # staticmethod receives in_dims as a tuple whose first entry is zero and whose second
        # entry is a list of two zeros.
        class Id(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function with two inputs whose vmap staticmethod checks
            # the in_dims structure and returns the inputs with it.
            @staticmethod
            def forward(x, y):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, x, y):
                # [FUNCTION DOC] HELPER: Asserts in_dims is a tuple with zero and a list, then
                # returns the inputs and in_dims.
                self.assertEqual(in_dims, (0, [0, 0]))
                self.assertTrue(isinstance(in_dims, tuple))
                self.assertTrue(isinstance(in_dims[1], list))
                return (x, y), in_dims

        x = torch.randn(2, device=device)
        vmap(Id.apply)(x, [x, x])

    def test_skips_empty_layer(self, device):
        # [FUNCTION DOC] Checks that a custom Function applied to a constant not batched by vmap
        # does not invoke the vmap staticmethod (which would raise), i.e. empty vmap layers
        # are skipped.
        class Id(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function that returns its input and whose vmap
            # staticmethod raises if called.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns the input.
                return input

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Raises RuntimeError saying it should not be called.
                raise RuntimeError("expected to not be called")

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the custom function to an unbatched constant and
            # returns the input times one.
            y = torch.tensor(1.0)
            y = Id.apply(y)
            return x * 1

        x = torch.randn(2, 3)
        vmap(f)(x)

    def test_none_returns(self, device):
        # [FUNCTION DOC] Checks that a custom vmap staticmethod may return None as the out dims,
        # meaning the output is unbatched or all outputs are, for a single output and for a tuple
        # output; the batched result is expanded to the batch and equals zeros.
        class Zeros(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function producing zeros of the input shape, whose vmap
            # staticmethod asserts in_dims is zero and returns zeros of the per-sample shape with
            # None as out dims.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns zeros with the shape and device of the input.
                return torch.zeros(input.shape, device=input.device)

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Raises AssertionError unless in_dims is a one-element
                # tuple holding zero; returns per-sample zeros and None.
                if in_dims != (0,):
                    raise AssertionError(f"Expected in_dims == (0,), got {in_dims}")
                return torch.zeros(input.shape[1:], device=input.device), None

        B = 2
        x = torch.randn(B, 3)
        y = vmap(Zeros.apply)(x)
        self.assertEqual(y, torch.zeros_like(x))

        class TwoZeros(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function returning the same zeros tensor twice; its vmap
            # staticmethod returns the zero tuple with None out dims.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns a tuple of the same zeros tensor twice.
                r = torch.zeros(input.shape, device=input.device)
                return r, r

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Raises AssertionError unless in_dims is a one-element
                # tuple holding zero; returns a tuple of two per-sample zeros tensors and None.
                if in_dims != (0,):
                    raise AssertionError(f"Expected in_dims == (0,), got {in_dims}")
                r = torch.zeros(input.shape[1:], device=input.device)
                return (r, r), None

        B = 2
        x = torch.randn(B, 3)
        result = vmap(TwoZeros.apply)(x)

        self.assertTrue(isinstance(result, tuple))
        y, z = result
        self.assertEqual(y, torch.zeros_like(x))
        self.assertEqual(z, torch.zeros_like(x))

    def test_should_have_two_returns(self, device):
        # [FUNCTION DOC] Checks that the vmap staticmethod must return a pair (output, out dims):
        # returning only an output, and also returning a four-element tuple, are both expected to
        # raise RuntimeError saying it must have two returns. KNOWN ISSUE: the second check
        # applies vmap to the first class (Zeros) again instead of TwoZeros, so the four-return
        # class is never exercised.
        class Zeros(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose vmap staticmethod wrongly returns a single
            # value rather than a pair.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns zeros with the shape and device of the input.
                r = torch.zeros(input.shape, device=input.device)
                return r

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Returns per-sample zeros only (invalid, no out dims).
                r = torch.zeros(input.shape[1:], device=input.device)
                return r

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "to have two returns"):
            vmap(Zeros.apply)(x)

        class TwoZeros(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose vmap staticmethod wrongly returns four
            # values; never actually applied in the test.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns a tuple of the same zeros tensor twice.
                r = torch.zeros(input.shape, device=input.device)
                return r, r

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Returns per-sample zeros twice plus two integer zeros
                # (four values, invalid).
                r = torch.zeros(input.shape[1:], device=input.device)
                return r, r, 0, 0

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "to have two returns"):
            vmap(Zeros.apply)(x)

    def test_incompatible_out_dims_error_msg(self, device):
        # [FUNCTION DOC] Checks that a vmap staticmethod whose out dims structure does not match
        # the output structure raises RuntimeError saying an incompatible value was returned, for
        # a tensor output with a tuple out dims and a list output with a tuple out dims.
        class Zeros(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function returning a tensor, whose vmap staticmethod
            # returns a tensor with a tuple of None as out dims (mismatched structure).
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns zeros with the shape and device of the input.
                r = torch.zeros(input.shape, device=input.device)
                return r

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Returns per-sample zeros and a tuple holding None as out
                # dims (mismatched).
                r = torch.zeros(input.shape[1:], device=input.device)
                return r, (None,)

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "returned an incompatible"):
            vmap(Zeros.apply)(x)

        class Zeros(torch.autograd.Function):
            # [CLASS DOC] HELPER: Redefinition for the second check: a Function returning a list,
            # whose vmap staticmethod returns a list with a tuple of None as out
            # dims (mismatched).
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns a list holding zeros of the input shape.
                r = torch.zeros(input.shape, device=input.device)
                return [r]

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                # [FUNCTION DOC] HELPER: Returns a list of per-sample zeros and a tuple holding
                # None as out dims (mismatched).
                r = torch.zeros(input.shape[1:], device=input.device)
                return [r], (None,)

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "returned an incompatible"):
            vmap(Zeros.apply)(x)

    def test_kwarg_only_tensors(self, device):
        # [FUNCTION DOC] Checks that defining a custom Function with a keyword-only tensor
        # argument and a vmap rule and then vmapping it raises NotImplementedError saying
        # kwarg-only Tensor args are unsupported. The class definition and vmap call are both
        # inside the expected-error block.
        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            class MyClass(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom Function adding a positional input and a keyword-only
                # input, with a vmap staticmethod returning their sum and an out dim of zero.
                @staticmethod
                def forward(x, *, y):
                    # [FUNCTION DOC] HELPER: Returns the sum of x and the keyword-only y.
                    return x + y

                @staticmethod
                def setup_context(ctx, inputs, output):
                    # [FUNCTION DOC] HELPER: Does nothing.
                    pass

                @staticmethod
                def vmap(info, in_dims, x, *, y):
                    # [FUNCTION DOC] HELPER: Asserts in_dims is a one-element tuple holding zero
                    # and returns the sum and out dim zero.
                    if in_dims != (0,):
                        raise AssertionError(f"Expected in_dims == (0,), got {in_dims}")
                    return x + y, 0

            x = torch.randn(3)
            y = torch.randn(3)

            vmap(MyClass.apply)(x, y=y)


@markDynamoStrictTest
class TestVmapOfGrad(TestCase):
    # [CLASS DOC] Device-generic tests for vmap of grad: per-sample gradients with views and
    # in-place ops, new_zeros and new_empty materialization, embedding network gradients, and log
    # softmax vjp.
    def test_per_sample_grads_inplace_view(self, device):
        # [FUNCTION DOC] Checks per-sample gradients of a loss that uses an in-place squeeze on a
        # matrix product: vmap of grad with the weight bound via partial matches a loop of
        # per-sample grads stacked (absolute tolerance zero, relative tolerance 5e-4).
        def compute_loss(weight, x, t):
            # [FUNCTION DOC] HELPER: Multiplies x by weight, squeezes the first dim in place and
            # returns the sum of the difference with the target.
            x = x.mm(weight)
            y = x.squeeze_(0)
            return (y - t).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 1, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_new_zeros_materializes_tensor(self, device):
        # [FUNCTION DOC] Checks that new_zeros called on a batched tensor inside vmap of grad
        # yields a tensor that can be filled by copy, with the gradient equal to ones.
        N = 3
        C = 5

        def foo(y, x):
            # [FUNCTION DOC] HELPER: Makes a zero vector via new_zeros on x, copies y into it in
            # place, returns the sum.
            result = x.new_zeros((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_new_empty_materializes_tensor(self, device):
        # [FUNCTION DOC] Same as the new_zeros test but using new_empty: the gradient from vmap of
        # grad is all ones.
        N = 3
        C = 5

        def foo(y, x):
            # [FUNCTION DOC] HELPER: Makes an uninitialized vector via new_empty on x, copies y
            # into it in place, returns the sum.
            result = x.new_empty((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_per_sample_grads_simple(self, device):
        # [FUNCTION DOC] Checks per-sample gradients of a squared error of a linear model via vmap
        # of grad against a loop of grads stacked (zero absolute and 5e-4 relative tolerance).
        def compute_loss(weight, x, t):
            # [FUNCTION DOC] HELPER: Returns the sum of the squared difference between x times
            # weight and the target.
            y = x @ weight
            return ((y - t) ** 2).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def _compare_expected_and_result(self, expected, result, mechanism):
        # [FUNCTION DOC] HELPER: Compares expected per-sample gradients with batched results. For
        # make_functional it stacks the per-sample tuples element-wise and compares each element;
        # otherwise it requires functional_call (raises AssertionError if not), rebuilds a dict of
        # stacked tensors per key and compares per key. Tolerances are zero absolute and
        # 1.5e-3 relative.
        if mechanism == "make_functional":
            expected = zip(*expected)
            expected = tuple(torch.stack(shards) for shards in expected)
            for r, e in zip(result, expected):
                self.assertEqual(r, e, atol=0, rtol=1.5e-3)
        else:
            if mechanism != "functional_call":
                raise AssertionError(
                    f"Expected mechanism 'functional_call', got '{mechanism}'"
                )
            expected = {
                k: tuple(d[k] for d in expected) for k, v in expected[0].items()
            }
            expected = {k: torch.stack(shards) for k, shards in expected.items()}
            for key in result:
                self.assertEqual(result[key], expected[key], atol=0, rtol=1.5e-3)

    @tf32_on_and_off(0.005)
    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_per_sample_grads_embeddingnet(self, device, mechanism):
        # [FUNCTION DOC] Parametrized over make_functional and functional_call and run with TF32
        # on and off. Builds an embedding-based network, computes per-sample gradients through
        # vmap of grad and compares with a loop over 64 samples via the comparison helper.
        class SampleNet(nn.Module):
            # [CLASS DOC] HELPER: Small network with an embedding of the vocabulary to 16 dims,
            # followed by transposing, a mean over the last dim, a linear layer, relu and a linear
            # layer to 2 outputs.
            def __init__(self, vocab_size: int):
                # [FUNCTION DOC] HELPER: Creates the embedding and two linear layers for the given
                # vocabulary size.
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 16)
                self.fc1 = nn.Linear(16, 16)
                self.fc2 = nn.Linear(16, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Embeds the input, swaps the last two dims, averages over
                # the last dim, then applies linear, relu and linear; returns the result.
                x = self.emb(x)
                x = torch.transpose(x, -1, -2)
                x = torch.mean(x, -1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return x

            def name(self):
                # [FUNCTION DOC] HELPER: Returns the string SampleNet.
                return "SampleNet"

        # Create our inputs...
        vocab_size = 1000
        batch_shape = [64]
        words_per_sentence = 5
        data = torch.randint(
            0, vocab_size, (*batch_shape, words_per_sentence), device=device
        )
        targets = torch.randint(0, 1, (*batch_shape,), device=device)

        # Construct our module
        net = SampleNet(vocab_size).to(device=device)
        criterion = nn.CrossEntropyLoss()

        net_func, weights = _get_weights_and_functional_call(net, mechanism)

        def compute_loss(weights, data, target):
            # [FUNCTION DOC] HELPER: Runs the functional network with the weights and data and
            # returns the cross entropy against the target.
            output = net_func(weights, data)
            result = criterion(output, target)
            return result

        expected = [grad(compute_loss)(weights, data[i], targets[i]) for i in range(64)]
        result = vmap(partial(grad(compute_loss), weights))(data, targets)
        self._compare_expected_and_result(expected, result, mechanism)

    def test_log_softmax(self, device):
        # [FUNCTION DOC] Checks vjp of log softmax inside vmap (input batched, cotangent not)
        # equals the gradient obtained by regular autograd backward with the cotangent expanded to
        # the input shape.
        x = torch.randn(3, 5, device=device)
        v = torch.randn(5, device=device)

        def foo(x, v):
            # [FUNCTION DOC] HELPER: Runs vjp of log softmax over the last dim at x and returns
            # the gradient for the cotangent v.
            _, vjp_fn = vjp(partial(torch.log_softmax, dim=-1), x)
            return vjp_fn(v)[0]

        result = vmap(foo, (0, None))(x, v)

        v = v.expand_as(x)
        x.requires_grad_()
        output = torch.log_softmax(x, dim=-1)
        output.backward(v)
        self.assertEqual(result, x.grad)


jacrev_and_jacfwd = parametrize(
    "jacapi", [subtest(jacrev, name="jacrev"), subtest(jacfwd, name="jacfwd")]
)

FIXME_jacrev_only = parametrize("jacapi", [subtest(jacrev, name="jacrev")])


@markDynamoStrictTest
class TestJac(VmapTearDownMixin, TestCase):
    # [CLASS DOC] Device-generic tests for jacrev and jacfwd (parametrized by the decorator over
    # both APIs unless noted): simple and non-flat inputs, vmap and nesting, argnums, pytree
    # inputs and outputs, has_aux, and later sections. Uses the mixin to clean up leaked vmap
    # nesting under Dynamo.
    @jacrev_and_jacfwd
    def test_simple(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of sin on a vector equals the diagonal matrix
        # of cosine.
        x = torch.randn(3, device=device)
        y = jacapi(torch.sin)(x)
        expected = torch.diagflat(x.cos())
        if not torch.allclose(y, expected):
            raise AssertionError("torch.allclose failed: y and expected differ")

    @jacrev_and_jacfwd
    def test_simple_not_flat(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of sin for a 2 by 3 input has shape 2,3,2,3 and
        # equals the reshaped diagonal of cosine.
        x = torch.randn(2, 3, device=device)
        y = jacapi(torch.sin)(x)
        expected = torch.diagflat(x.view(-1).cos())
        expected = expected.view(2, 3, 2, 3)
        if not torch.allclose(y, expected):
            raise AssertionError("torch.allclose failed: y and expected differ")

    @jacrev_and_jacfwd
    def test_take(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of an index-select (take) function equals the one
        # from the torch.autograd.functional Jacobian. KNOWN ISSUE: it always calls jacrev rather
        # than the jacapi parameter, so the jacfwd variant tests jacrev too.
        x = torch.rand(5)

        def func(x):
            # [FUNCTION DOC] HELPER: Takes elements of x at an index tensor of three ones and
            # returns them.
            y = torch.ones(3, dtype=torch.long)
            z = torch.take(x, y)
            return z

        self.assertEqual(jacrev(func)(x), torch.autograd.functional.jacobian(func, x))

    @jacrev_and_jacfwd
    def test_diff_numel(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of a function mapping a 2 by 4 input to a 3 by 1
        # output (a slice of the first row) has shape 3,1,2,4 with ones at the three expected
        # positions and zeros elsewhere.
        x = torch.randn(2, 4, device=device)

        # Tensor[2, 4] -> Tensor[3, 1]
        def f(x):
            # [FUNCTION DOC] HELPER: Returns elements one onward of the first row, with a trailing
            # dim added.
            return x[0, 1:].unsqueeze(-1)

        y = jacapi(f)(x)
        self.assertEqual(y.shape, (3, 1, 2, 4))

        expected = x.new_zeros(3, 1, 2, 4)
        expected[0, 0, 0, 1] = 1
        expected[1, 0, 0, 2] = 1
        expected[2, 0, 0, 3] = 1
        self.assertEqual(y, expected)

    @jacrev_and_jacfwd
    def test_vmap_on_jac_simple(self, device, jacapi):
        # [FUNCTION DOC] Checks vmap over the Jacobian of sin gives a stack of per-row diagonal
        # matrices of cosine.
        x = torch.randn(2, 3, device=device)
        y = vmap(jacapi(torch.sin))(x)
        expected = torch.stack([torch.diagflat(x[i].cos()) for i in range(2)])
        if not torch.allclose(y, expected):
            raise AssertionError("torch.allclose failed: y and expected differ")

    @jacrev_and_jacfwd
    def test_nested_jac_simple(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of the Jacobian (nested) of a scalar function, sum of
        # sin, is the diagonal matrix of minus sin.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input.
            return x.sin().sum()

        x = torch.randn(3, device=device)
        y = jacapi(jacapi(foo))(x)
        expected = torch.diagflat(-x.sin())
        if not torch.allclose(y, expected):
            raise AssertionError("torch.allclose failed: y and expected differ")

    @jacrev_and_jacfwd
    def test_multiple_args(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of an elementwise product with respect to the second
        # argument (argnums one) is the diagonal matrix of the first argument.
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=1)(x, y)
        expected = torch.diagflat(x)
        if not torch.allclose(z, expected):
            raise AssertionError("torch.allclose failed: z and expected differ")

    @jacrev_and_jacfwd
    def test_multiple_outputs_multiple_argnums(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of a function with two outputs and argnums of both
        # inputs is a tuple of tuples of diagonal matrices holding the expected linear
        # coefficients per output and input.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a tuple of two linear combinations of x and y with
            # coefficients two,three and four,five.
            return 2 * x + 3 * y, 4 * x + 5 * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f, argnums=(0, 1))(x, y)
        expected_out0_x = torch.diagflat(torch.full_like(x, 2))
        expected_out0_y = torch.diagflat(torch.full_like(y, 3))
        expected_out1_x = torch.diagflat(torch.full_like(x, 4))
        expected_out1_y = torch.diagflat(torch.full_like(y, 5))

        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertEqual(len(z[0]), 2)
        self.assertTrue(isinstance(z[0], tuple))
        self.assertEqual(z[0][0], expected_out0_x)
        self.assertEqual(z[0][1], expected_out0_y)
        self.assertEqual(z[1][0], expected_out1_x)
        self.assertEqual(z[1][1], expected_out1_y)

    @jacrev_and_jacfwd
    def test_multiple_outputs_single_argnums(self, device, jacapi):
        # [FUNCTION DOC] Checks that with two outputs a single int argnums yields a tuple of
        # per-output Jacobians, while a one-element tuple argnums yields a tuple of
        # one-element tuples.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a tuple of two linear combinations of x and y with
            # coefficients two,three and four,five.
            return 2 * x + 3 * y, 4 * x + 5 * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected_out0_x = torch.diagflat(torch.full_like(x, 2))
        expected_out1_x = torch.diagflat(torch.full_like(x, 4))

        z = jacapi(f, argnums=0)(x, y)
        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertEqual(z, (expected_out0_x, expected_out1_x))

        z = jacapi(f, argnums=(0,))(x, y)
        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertTrue(isinstance(z[0], tuple))
        self.assertEqual(z, ((expected_out0_x,), (expected_out1_x,)))

    @jacrev_and_jacfwd
    def test_multiple_outputs_pytree(self, device, jacapi):
        # [FUNCTION DOC] Checks that dict outputs produce a dict result whose entries are tuples
        # of Jacobians over both inputs, equal to the expected diagonal matrices.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a dict with a left and a right linear combination of
            # x and y.
            return {"left": 2 * x + 3 * y, "right": 4 * x + 5 * y}

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f, argnums=(0, 1))(x, y)
        expected_left_x = torch.diagflat(torch.full_like(x, 2))
        expected_left_y = torch.diagflat(torch.full_like(y, 3))
        expected_right_x = torch.diagflat(torch.full_like(x, 4))
        expected_right_y = torch.diagflat(torch.full_like(y, 5))
        expected = {
            "left": (expected_left_x, expected_left_y),
            "right": (expected_right_x, expected_right_y),
        }
        self.assertTrue(isinstance(z, dict))
        self.assertTrue(isinstance(z["left"], tuple))
        self.assertTrue(isinstance(z["right"], tuple))
        self.assertEqual(z, expected)

    @jacrev_and_jacfwd
    def test_multiple_inputs_pytree(self, device, jacapi):
        # [FUNCTION DOC] Checks Jacobians with pytree inputs for scalar data and argnums covering
        # all arguments, only the pair argument, and the default; results mirror the input
        # structure with the expected constant coefficients.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Unpacks a pair from a and returns a weighted sum of its
            # elements, b and c with weights one, two, three, four.
            a0, a1 = a
            return a0 + a1 * 2 + b * 3 + c * 4

        x = torch.randn([], device=device)
        args = ((x, x), x, x)

        result = jacapi(f, argnums=(0, 1, 2))(*args)
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
            torch.tensor(3.0, device=device),
            torch.tensor(4.0, device=device),
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0,))(*args)
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
        )
        self.assertEqual(result, expected)

        result = jacapi(f)(*args)
        expected = (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device))
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_dimensionality(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of the identity function: a scalar input gives a
        # zero-dimensional ones result and a length-one vector gives a one by one matrix of one.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input.
            return x

        x = torch.randn([], device=device)
        result = jacapi(f)(x)
        self.assertEqual(result.dim(), 0)
        self.assertEqual(result, torch.ones_like(x))

        x = torch.randn([1], device=device)
        result = jacapi(f)(x)
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result, x.new_ones(1, 1))

    @jacrev_and_jacfwd
    def test_aux_tensor(self, device, jacapi):
        # [FUNCTION DOC] Checks has_aux with a tensor aux: result is the identity Jacobian and the
        # aux value is the cosine of the input.
        def f(x):
            # [FUNCTION DOC] HELPER: Clones the input and returns it with the cosine of the clone
            # as aux.
            y = x.clone()
            return y, y.cos()

        x = torch.randn(3, device=device)
        result, aux = jacapi(f, has_aux=True)(x)

        self.assertEqual(result, torch.eye(3, 3, device=device))
        self.assertEqual(aux, x.cos())

    @jacrev_and_jacfwd
    def test_aux_pytree(self, device, jacapi):
        # [FUNCTION DOC] Checks has_aux with a pytree aux is returned intact and the Jacobian is
        # the identity, and non-tensor aux values (int, float, string, or lists containing them)
        # raise RuntimeError about unsupported types.
        def f(x):
            # [FUNCTION DOC] HELPER: Clones the input and returns it together with an aux dict of
            # cosine and a list holding tangent.
            y = x.clone()
            return y, {"a": y.cos(), "b": [y.tan()]}

        x = torch.randn(3, device=device)

        result, aux = jacapi(f, has_aux=True)(x)
        self.assertEqual(result, torch.eye(3, 3, device=device))
        _, expected_aux = f(x)
        self.assertEqual(aux, expected_aux)

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jacapi(lambda x: (x, aux), has_aux=True)(x)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jacapi(lambda x: (x, [x, aux]), has_aux=True)(x)

    @jacrev_and_jacfwd
    def test_outputs_can_any_pytree(self, device, jacapi):
        # [FUNCTION DOC] Checks which output types the Jacobian APIs accept: None and empty tuple
        # raise RuntimeError about non-empty output, scalars, bool, float and string outputs raise
        # RuntimeError about only tensors; list, dict and nested list/tuple/dict outputs work and
        # keep their structure.
        x = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"(vjp|jvp).+: Expected f to be a function that has non-empty output",
            ):
                jacapi(lambda _: output)(x)

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"(vjp|jvp).+: expected f\(\*primals\) to return only tensors",
            ):
                jacapi(lambda _: output)(x)

        # Check list output
        out = jacapi(lambda x: [x, x.sum()])(x)
        if not (isinstance(out, list) and len(out) == 2):
            raise AssertionError(
                f"Expected list of length 2, got {type(out).__name__} of length {len(out)}"
            )

        # Check dict output
        out = jacapi(lambda x: {"x": x, "xsum": x.sum()})(x)
        if not (isinstance(out, dict) and len(out) == 2 and "xsum" in out):
            raise AssertionError(
                f"Expected dict of length 2 with 'xsum' key, got {type(out).__name__}"
            )

        def composite_output(x):
            # [FUNCTION DOC] HELPER: Returns a list with a tuple of the sum and a dict of the
            # input and a list of the input and the sum.
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        out = jacapi(composite_output)(x)
        if not isinstance(out, list):
            raise AssertionError(f"Expected list, got {type(out).__name__}")
        if not (isinstance(out[0], tuple) and isinstance(out[0][1], dict)):
            raise AssertionError(
                f"Expected (tuple, dict) structure, got ({type(out[0]).__name__}, {type(out[0][1]).__name__})"
            )

    @jacrev_and_jacfwd
    def test_multiple_inputs_outputs_pytree(self, device, jacapi):
        # [FUNCTION DOC] Checks Jacobians with pytree inputs and a pytree output (a tuple of a
        # tensor and a dict) for default argnums, a one-element argnums tuple and an argnums tuple
        # of two. Results mirror the output structure and then the input structure, with constant
        # coefficients and zeros for entries that do not contribute.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Unpacks a pair from a and returns the combination of its
            # elements, plus a dict with foo holding the combination of b and c.
            a0, a1 = a
            return a0 + a1 * 2, {"foo": b * 3 + c * 4}

        x = torch.randn([], device=device)
        zero = torch.zeros([], device=device)
        args = ((x, x), x, x)

        result = jacapi(f)(*args)
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
            {"foo": (zero, zero)},
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0,))(*args)
        expected = (
            ((torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),),
            {"foo": ((zero, zero),)},
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0, 1))(*args)
        expected = (
            (
                (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
                zero,
            ),
            {"foo": ((zero, zero), torch.tensor(3.0, device=device))},
        )
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_multiple_inputs_outputs_pytree_multidim(self, device, jacapi):
        # [FUNCTION DOC] Checks the Jacobian of a function taking a dict of two same-shaped
        # vectors and returning a dict gives a nested dict: each output key maps to a dict by
        # input key holding a diagonal matrix of the derivative or zeros.
        def f(dct):
            # [FUNCTION DOC] HELPER: Returns a dict with c as sin of entry a and d as cosine of
            # entry b.
            a = dct["a"]
            b = dct["b"]
            return {"c": a.sin(), "d": b.cos()}

        x = torch.randn(3, device=device)
        args = ({"a": x, "b": x},)

        result = jacapi(f)(*args)
        expected = {
            "c": {"a": x.cos().diagflat(), "b": x.new_zeros(3, 3)},
            "d": {"a": x.new_zeros(3, 3), "b": -x.sin().diagflat()},
        }
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_unrelated_input(self, device, jacapi):
        # [FUNCTION DOC] Checks that an input unused by the function gets an all-zero Jacobian
        # while the used one gets the identity, with both argnums requested and a tuple result.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the first argument and ignores the second.
            return x

        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)

        result = jacapi(f, argnums=(0, 1))(x, y)
        expected0 = torch.eye(6, 6, device=device).view(2, 3, 2, 3)
        expected1 = y.new_zeros(2, 3, 2, 3)
        expected = (expected0, expected1)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_unrelated_output(self, device, jacapi):
        # [FUNCTION DOC] Checks that a function returning a captured tensor, unrelated to its
        # input, has an all-zero Jacobian of shape output dims followed by input dims.
        y = torch.randn(2, 3, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Ignores its input and returns the captured tensor.
            return y

        x = torch.randn(2, 3, device=device)

        result = jacapi(f)(x)
        expected = x.new_zeros(2, 3, 2, 3)
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_empty_output(self, device, jacapi):
        # [FUNCTION DOC] Checks that a function returning an empty tuple raises RuntimeError whose
        # message contains the text xpected.
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns an empty tuple.
            return ()

        with self.assertRaisesRegex(RuntimeError, "xpected"):
            jacapi(f)(x, y)

    @jacrev_and_jacfwd
    def test_argnums_tuple(self, device, jacapi):
        # [FUNCTION DOC] Checks argnums of a tuple of both inputs on an elementwise product gives
        # a pair of diagonal matrices (of the other input).
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=(0, 1))(x, y)
        expected0 = torch.diagflat(y)
        expected1 = torch.diagflat(x)
        if len(z) != 2:
            raise AssertionError(f"Expected len(z) == 2, got {len(z)}")
        if not torch.allclose(z[0], expected0):
            raise AssertionError("torch.allclose failed: z[0] and expected0 differ")
        if not torch.allclose(z[1], expected1):
            raise AssertionError("torch.allclose failed: z[1] and expected1 differ")

    @jacrev_and_jacfwd
    def test_argnums_effect_on_return(self, device, jacapi):
        # [FUNCTION DOC] Checks that a one-element tuple of argnums yields a one-element tuple
        # result while an int argnums yields a plain tensor.
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=(0,))(x, y)
        expected0 = torch.diagflat(y)
        if not isinstance(z, tuple):
            raise AssertionError(f"Expected tuple, got {type(z).__name__}")
        if len(z) != 1:
            raise AssertionError(f"Expected len(z) == 1, got {len(z)}")
        if not torch.allclose(z[0], expected0):
            raise AssertionError("torch.allclose failed: z[0] and expected0 differ")

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=0)(x, y)
        expected0 = torch.diagflat(y)
        if not isinstance(z, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(z).__name__}")
        if not torch.allclose(z, expected0):
            raise AssertionError("torch.allclose failed: z and expected0 differ")

    @jacrev_and_jacfwd
    def test_argnums_defaults_to_zero(self, device, jacapi):
        # [FUNCTION DOC] Checks that the default argnums differentiates only the first argument.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns two times x plus three times y.
            return x * 2 + y * 3

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f)(x, y)
        expected = torch.diagflat(torch.full_like(x, 2))
        self.assertEqual(z, expected)

    @jacrev_and_jacfwd
    def test_empty_argnums(self, device, jacapi):
        # [FUNCTION DOC] Checks that empty argnums raises RuntimeError saying it must
        # be non-empty.
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            jacapi(torch.sin, argnums=())(x)

    @jacrev_and_jacfwd
    def test_out_of_bounds_argnums(self, device, jacapi):
        # [FUNCTION DOC] Checks that an argnums value beyond the number of inputs raises
        # RuntimeError mentioning only one positional input.
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            jacapi(torch.sin, argnums=2)(x)

    @jacrev_and_jacfwd
    def test_negative_argnums(self, device, jacapi):
        # [FUNCTION DOC] Checks that a too-negative argnums raises RuntimeError mentioning only
        # one positional input.
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            jacapi(torch.sin, argnums=-2)(x)

    @jacrev_and_jacfwd
    def test_repeated_argnums(self, device, jacapi):
        # [FUNCTION DOC] Checks that duplicate argnums raise RuntimeError saying they must
        # be unique.
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            jacapi(torch.sin, argnums=(0, 0))(x)

    @jacrev_and_jacfwd
    def test_float_argnums(self, device, jacapi):
        # [FUNCTION DOC] Checks that a float argnums raises RuntimeError saying it must be int or
        # Tuple, and a tuple with a float element raises RuntimeError saying it must be int.
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be int or Tuple"):
            jacapi(torch.sin, argnums=0.0)(x)
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            jacapi(torch.multiply, argnums=(1, 0.0))(x, x)

    def test_hessian_simple(self, device):
        # [FUNCTION DOC] Smoke test: computes the hessian of an elementwise sin on a vector and
        # does not check the result.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns sin of the input.
            return x.sin()

        x = torch.randn(3, device=device)
        hessian(f)(x)

    def _test_against_reference(self, f, inputs, jacapi):
        # [FUNCTION DOC] HELPER: Compares the Jacobian API under test (a parameter) applied to a
        # wrapper taking one tuple of inputs with the reference Jacobian from
        # torch.autograd.functional called on the function with the inputs; asserts equality.
        def foo(inputs):
            # [FUNCTION DOC] HELPER: Unpacks the tuple of inputs into positional arguments for
            # the function.
            return f(*inputs)

        expected = torch.autograd.functional.jacobian(f, inputs)
        result = jacapi(foo)(inputs)
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_against_reference_simple(self, device, jacapi):
        # [FUNCTION DOC] Compares against the reference Jacobian for an elementwise function of
        # one 3-d input.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns three times the square of the input.
            return 3 * x**2

        x = torch.randn(2, 3, 5, device=device)
        self._test_against_reference(f, (x,), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_multi_input(self, device, jacapi):
        # [FUNCTION DOC] Compares against the reference Jacobian for a function of two matrices
        # (elementwise then matrix product).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of x times its cosine with the
            # sine of y.
            return (x.cos() * x) @ y.sin()

        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_multi_input_multi_output(self, device, jacapi):
        # [FUNCTION DOC] Compares against the reference Jacobian for a function with two inputs
        # and three outputs.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns three outputs: a squared product, a matrix product
            # with a row-summed scaled y, and the sum of y.
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        x = torch.randn(5, 3, device=device)
        y = torch.randn(3, 5, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_unrelated_outputs(self, device, jacapi):
        # [FUNCTION DOC] Compares against the reference Jacobian for a function returning its
        # inputs, so outputs are unrelated to each other and repeated.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x, y, x, y.
            return x, y, x, y

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_zero_dim(self, device, jacapi):
        # [FUNCTION DOC] Compares against the reference Jacobian for zero-dimensional outputs, a
        # zero-dimensional input, and a mix of zero-dimensional input with scalar-reducing output.
        # zero-dim output
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x, the sum of y and their
            # elementwise product.
            return x.sum(), y.sum(), x * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

        # zero-dim input
        def g(x):
            # [FUNCTION DOC] HELPER: Returns three copies of x stacked.
            return torch.stack([x, x, x])

        x = torch.randn([], device=device)
        self._test_against_reference(g, (x,), jacapi)

        # Mixed zero-dim input / zero-dim output
        def h(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of y and the product x times y.
            return y.sum(), x * y

        x = torch.randn([], device=device)
        y = torch.randn(1, device=device)
        self._test_against_reference(h, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_correctness_different_devices(self, device, jacapi):
        # [FUNCTION DOC] Compares against the reference Jacobian where inputs are CPU tensors and
        # one output is moved to the test device (the device parameter), checking
        # cross-device correctness.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the product of x and y and a copy of that product
            # moved to the test device.
            return x * y, (x * y).to(device=device)

        x = torch.randn(3)
        y = torch.randn(3)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_default_arg(self, device, jacapi):
        # [FUNCTION DOC] Compares against the reference Jacobian for a function with a defaulted
        # third parameter not passed as input.
        def f(x, y, z=3.0):
            # [FUNCTION DOC] HELPER: Returns the product of x, y and z.
            return x * y * z

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_inplace(self, device, jacapi):
        # [FUNCTION DOC] Checks Jacobians of functions that mutate an argument in place: copying x
        # into y and returning y gives identity for argnums zero; slice-assigning y into x in a
        # function returning stacked sums with argnums one and two gives the expected block
        # structure of cross terms.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Copies x into y in place and returns y.
            y.copy_(x)
            return y

        out = jacapi(f, argnums=0)  # x is differentiable
        x, y = torch.randn(2, device=device), torch.randn(2, device=device)
        self.assertEqual(out(x, y), torch.eye(y.shape[0]))

        # testing tuple of argnums with the example that raised this issue originally
        def g(x, y, z):
            # [FUNCTION DOC] HELPER: Assigns y into the first two elements of x in place and
            # returns a column stack of the sum of x squared and the sum of z cubed.
            x[:2] = y
            return torch.vstack([(x**2).sum(), (z**3).sum()])

        out = jacapi(g, argnums=(1, 2))
        x, y, z = (
            torch.randn(3, device=device),
            torch.randn(2, device=device),
            torch.randn(2, device=device),
        )

        expected_out = (
            torch.zeros(2, 1, 2, device=device),
            torch.zeros(2, 1, 2, device=device),
        )
        expected_out[0][0][0] = 2 * y  # top left corner
        expected_out[1][1][0] = 3 * (z**2)  # bottom right corner

        out_val = out(x, y, z)
        self.assertEqual(out_val, expected_out)

    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev(self, device, _preallocate_and_copy):
        # [FUNCTION DOC] Parametrized over the preallocate-and-copy option of jacrev. Checks that
        # jacrev with various chunk sizes (including larger than the output) equals the unchunked
        # result for a nested tuple output with two argnums, and that a chunk size of zero or
        # negative raises ValueError about chunk_size being greater than zero.
        x = torch.randn(10, 2, device=device)
        y = torch.randn(1, 2, device=device)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a nested tuple of sin of x, x plus y, x plus two, and
            # the sum of x.
            return (x.sin(), x + y), (x + 2, x.sum())

        for chunk_size in (1, 2, 3, 4, 7, 10, 1000):
            expected = jacrev(f, argnums=(0, 1))(x, y)
            actual = jacrev(
                f,
                argnums=(0, 1),
                chunk_size=chunk_size,
                _preallocate_and_copy=_preallocate_and_copy,
            )(x, y)
            self.assertEqual(actual, expected)

        err_msg = "jacrev: `chunk_size` should be greater than 0."
        with self.assertRaisesRegex(ValueError, err_msg):
            jacrev(f, argnums=(0,), chunk_size=0)(x, y)

        with self.assertRaisesRegex(ValueError, err_msg):
            jacrev(f, argnums=(0,), chunk_size=-2)(x, y)

    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev_composition(self, device, _preallocate_and_copy):
        # [FUNCTION DOC] Parametrized over the preallocate-and-copy option. Checks that vmap over
        # nested jacrev with chunk size three equals the unchunked result.
        x = torch.randn(10, 2, device=device)
        chunk_size = 3

        def f(x):
            # [FUNCTION DOC] HELPER: Returns a nested tuple of sin of x, x, x plus two and the sum
            # of x.
            return (x.sin(), x), (x + 2, x.sum())

        expected = vmap(jacrev(jacrev(f)))(x)
        actual = vmap(
            jacrev(
                jacrev(
                    f,
                    chunk_size=chunk_size,
                    _preallocate_and_copy=_preallocate_and_copy,
                ),
                chunk_size=chunk_size,
            )
        )(x)
        self.assertEqual(actual, expected)

    # https://github.com/pytorch/pytorch/issues/127036
    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev_chunksize_one(self, device, _preallocate_and_copy):
        # [FUNCTION DOC] Regression test for issue 127036, parametrized over the
        # preallocate-and-copy option. With chunk size one, jacrev does not use vmap, so a custom
        # Function whose backward does a data-dependent (nonzero) op still works and matches the
        # reference; with chunk size two it raises RuntimeError about batching operators with
        # dynamic output shape.
        # With chunk_size=1, we shouldn't `vmap` and hence not be limited
        # by it's constraints.
        x = torch.randn(3, 3, device=device)

        # Function with Dynamic Op in Backward.
        # This should cause jacrev/vmap(vjp) to fail.
        class IdentityWithDynamicBackwardOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function that returns its input and whose backward calls
            # nonzero (dynamic output shape) on the gradient and returns the gradient.
            @staticmethod
            def forward(input):
                # [FUNCTION DOC] HELPER: Returns the input.
                return input

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Calls nonzero on the gradient, discarding the result, and
                # returns the gradient.
                # dynamic op in backward pass.
                grad_output.nonzero()
                return grad_output

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the custom function.
            return IdentityWithDynamicBackwardOp.apply(x)

        # With `chunk_size=1`, we don't use vmap. So the following should work.
        jacfn = jacrev(f, chunk_size=1, _preallocate_and_copy=_preallocate_and_copy)
        actual = jacfn(x)
        expected = torch.autograd.functional.jacobian(f, x, vectorize=False)
        self.assertEqual(actual, expected)

        # Should fail with `chunk_size=2`.
        msg = (
            r"vmap: We do not support batching operators that can output dynamic shape."
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            jacrev(f, chunk_size=2, _preallocate_and_copy=_preallocate_and_copy)(x)

    def test_complex_error(self, device):
        # [FUNCTION DOC] Checks that complex inputs and complex outputs raise RuntimeError for
        # both jacrev and jacfwd, with messages naming the API and saying all inputs (or outputs)
        # are expected to be real.
        # Verify complex input raises error
        # C -> C
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the complex conjugate of the input.
            return x.conj()

        x = torch.randn(1, device=device, dtype=torch.cfloat)

        with self.assertRaisesRegex(RuntimeError, "jacrev: Expected all inputs"):
            jacrev(fn)(x)

        with self.assertRaisesRegex(RuntimeError, "jacfwd: Expected all inputs"):
            jacfwd(fn)(x)

        # Verify complex output raises error
        # R -> C
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the conjugate of the input times an imaginary
            # constant, giving a complex output from a real input.
            return torch.conj(x * 0.5j)

        x = torch.randn(1, device=device, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, "jacrev: Expected all outputs"):
            jacrev(fn)(x)

        with self.assertRaisesRegex(RuntimeError, "jacfwd: Expected all outputs"):
            jacfwd(fn)(x)

    @jacrev_and_jacfwd
    def test_jac_with_non_tensor_args(self, device, jacapi):
        # [FUNCTION DOC] Checks that a non-tensor positional argument (an int) is allowed: the
        # Jacobian of an add with the int equals the reference with the int bound by keyword.
        def f(t, int_x):
            # [FUNCTION DOC] HELPER: Returns the tensor plus the int.
            return t + int_x

        t = torch.randn(3, 3, device=device)

        actual = jacapi(f)(t, 3)
        expected = torch.autograd.functional.jacobian(partial(f, int_x=3), t)
        self.assertEqual(actual, expected)


@markDynamoStrictTest
class TestHessian(TestCase):
    # [CLASS DOC] Device-generic tests for hessian, comparing against the reference hessian from
    # torch.autograd.functional and checking vmap composition across Jacobian levels.
    def _test_against_reference(self, f, inputs):
        # [FUNCTION DOC] HELPER: Asserts that the hessian of a wrapper taking a tuple of inputs
        # equals the reference hessian from torch.autograd.functional.
        def foo(inputs):
            # [FUNCTION DOC] HELPER: Unpacks the tuple of inputs into positional arguments for
            # the function.
            return f(*inputs)

        expected = torch.autograd.functional.hessian(f, inputs)
        result = hessian(foo)(inputs)
        self.assertEqual(result, expected)

    def test_hessian_vectorize_correctness_simple(self, device):
        # [FUNCTION DOC] Compares the hessian against the reference for a scalar function of one
        # 3-d input.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of three times the square of the input.
            return (3 * x**2).sum()

        x = torch.randn(2, 3, 5, device=device)
        self._test_against_reference(f, (x,))

    def test_hessian_vectorize_correctness_multi_input(self, device):
        # [FUNCTION DOC] Compares the hessian against the reference for a scalar function of
        # three inputs.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the sum of relu-scaled x times sin of y
            # matrix-multiplied by z.
            return ((x.relu() * x) @ y.sin() @ z).sum()

        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        z = torch.randn(5, 5, device=device)
        self._test_against_reference(f, (x, y, z))

    def test_hessian_vectorize_correctness_unrelated_outputs(self, device):
        # [FUNCTION DOC] Compares the hessian against the reference when the output is unrelated
        # to one input, and when the output is a constant unrelated to all inputs.
        # output unrelated to one input
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of squares of x, ignoring y.
            return (x**2).sum()

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y))

        # output unrelated to all inputs
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a constant scalar one, ignoring both inputs.
            return torch.ones([])

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO and sys.version_info[:2] < (3, 14),
        "Frame Handling Difference between Python versions",
    )
    def test_jacfwd_different_levels(self, device):
        # [FUNCTION DOC] Regression test for functorch issue 597: vmap of nested jacrev and vmap
        # of hessian of a squared residual loss over a batch give the same Hessian. Skipped under
        # Dynamo on Python before 3.14 due to frame handling differences.
        # Test case from:
        # https://github.com/pytorch/functorch/issues/597
        b = 8
        n = 100
        d = 2
        x1 = torch.randn(b, n, d, device=device)
        x2 = x1
        A = 0.1 * torch.randn(b, d, d, device=device)

        def loss(A, x1, x2):
            # [FUNCTION DOC] HELPER: Returns the sum of squares of the difference between x2 and
            # the transposed product of A with the transposed x1.
            x2_hat = (A @ (x1.T)).T
            res = x2 - x2_hat
            res_sqr = res**2
            return res_sqr.sum()

        hess1 = vmap(jacrev(jacrev(loss)))(A, x1, x2)
        hess2 = vmap(hessian(loss))(A, x1, x2)
        self.assertEqual(hess2, hess1)


@markDynamoStrictTest
class TestJvp(TestCase):
    # [CLASS DOC] Device-generic tests for jvp: simple, multiple and pytree inputs, validation
    # errors, unrelated inputs and outputs, strict mode, has_aux and interaction with forward-grad
    # enabling and autograd Functions.
    def test_inplace_on_captures(self, device):
        # [FUNCTION DOC] Checks that mutating a captured tensor in place from a function run under
        # grad raises RuntimeError saying a captured Tensor cannot be mutated.
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        captured = torch.randn(3, device=device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Copies the input into the captured tensor in place and
            # returns the sum of their product.
            captured.copy_(x)
            return (x * captured).sum()

        with self.assertRaisesRegex(RuntimeError, "mutate a captured Tensor"):
            grad(foo)(x)

    def test_simple(self, device):
        # [FUNCTION DOC] Checks jvp of sin returns a tuple of sin and cosine times the tangent.
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)
        result = jvp(torch.sin, (x,), (t,))
        expected = (x.sin(), x.cos() * t)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_multiple_inputs(self, device):
        # [FUNCTION DOC] Checks jvp for a product of two inputs returns the product and the
        # product rule tangent.
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the product of x and y.
            return x * y

        result = jvp(f, (x, y), (tx, ty))
        expected = (x * y, y * tx + x * ty)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_pytree_inputs(self, device):
        # [FUNCTION DOC] Checks jvp with a nested tuple primal and unit tangents gives primal and
        # tangent outputs equal to ten.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Unpacks a pair from x and returns a weighted sum of its
            # elements, y and z with weights one, two, three, four.
            a, b = x
            return a + 2 * b + 3 * y + 4 * z

        one = torch.tensor(1.0, device=device)
        primal_outs, tangent_outs = jvp(
            f, ((one, one), one, one), ((one, one), one, one)
        )
        self.assertEqual(primal_outs, one * 10)
        self.assertEqual(tangent_outs, one * 10)

    def test_pytree_inputs_error_cases(self, device):
        # [FUNCTION DOC] Checks jvp error messages: primals not a tuple; primals and tangents of
        # different structure; non-tensor leaves in primals or tangents; a primals tuple
        # containing only an empty tuple (no tensor).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns its input.
            return x

        one = torch.tensor(1.0, device=device)

        with self.assertRaisesRegex(RuntimeError, "Expected primals to be a tuple"):
            jvp(f, one, one)
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(f, ((one, one), one), (one, one))
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(f, ((one, one), 1), ((one, one), one))
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(f, ((one, one), 1), ((1, one), one))
        with self.assertRaisesRegex(RuntimeError, "at least one Tensor"):
            jvp(f, ((),), ((),))

    def test_unrelated_input(self, device):
        # [FUNCTION DOC] Checks jvp where the function ignores the second input: result is the
        # first input and its tangent.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x and ignores y.
            return x

        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        result = jvp(f, (x, y), (tx, ty))
        expected = (x, tx)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_unrelated_output(self, device):
        # [FUNCTION DOC] Checks jvp where the output is an unrelated captured tensor: the tangent
        # output is zeros.
        y = torch.randn(2, 3, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Ignores the input and returns the captured tensor.
            return y

        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        result = jvp(f, (x,), (tx,))
        expected = (y, torch.zeros_like(y))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_strict_mode(self, device):
        # [FUNCTION DOC] Checks that jvp with strict mode on raises RuntimeError mentioning strict
        # when an output does not depend on the inputs.
        y = torch.randn(2, 3, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input and an unrelated captured tensor.
            return x, y

        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        with self.assertRaisesRegex(RuntimeError, "strict"):
            jvp(f, (x,), (tx,), strict=True)

    def test_multiple_outputs(self, device):
        # [FUNCTION DOC] Checks jvp with two outputs returns primal outputs and tuple of tangents.
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns sin and cosine of the input.
            return torch.sin(x), torch.cos(x)

        result = jvp(f, (x,), (t,))
        expected = (f(x), (x.cos() * t, -x.sin() * t))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_multiple_inputs_outputs(self, device):
        # [FUNCTION DOC] Checks jvp of a linear function with two inputs and outputs equals the
        # function applied to primals and the function applied to tangents.
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns two linear combinations of x and y.
            return 2 * x + 3 * y, 4 * x + 5 * y

        result = jvp(f, (x, y), (tx, ty))
        expected = (f(x, y), f(tx, ty))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_jvp_new_tensor(self):
        # [FUNCTION DOC] Checks that creating a tensor with new_tensor from the input inside jvp
        # works, and with zero tangents the output tangent is zeros.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds a new scalar tensor made by new_tensor to the input.
            y = x.new_tensor(0.5)
            return x + y

        x = torch.rand(10, 10)
        tangents = torch.zeros_like(x)
        actual = jvp(f, (x,), (tangents,))
        expected = (f(x), torch.zeros_like(x))
        self.assertEqual(actual, expected)

    def test_primals_tangents_length_mismatch(self, device):
        # [FUNCTION DOC] Checks jvp raises RuntimeError about the same python structure when the
        # tangents tuple length differs from the primals tuple length.
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        msg = "same python structure"
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x,), (t, t))
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x, x), (t, t, t))

    def test_nonempty_primals_and_tangents(self, device):
        # [FUNCTION DOC] Checks that empty primals and tangents raise RuntimeError saying at least
        # one Tensor is required.
        with self.assertRaisesRegex(RuntimeError, "at least one Tensor"):
            jvp(torch.sin, (), ())

    def test_inputs_are_tuples_of_tensors(self, device):
        # [FUNCTION DOC] Checks jvp argument validation: primals not a tuple, tangents not a tuple
        # or a list instead, and non-tensor entries in primals or tangents, each raising
        # RuntimeError with the matching message fragment.
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        with self.assertRaisesRegex(RuntimeError, "be a tuple"):
            jvp(torch.sin, x, (t,))
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(torch.sin, (x,), t)
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(torch.sin, (x,), [t])
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(torch.sin, (1.0,), (t,))
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(torch.sin, (x,), (1.0,))

    def test_outputs_can_any_pytree(self, device):
        # [FUNCTION DOC] Checks which output types jvp accepts: None and empty tuple raise
        # RuntimeError about non-empty output; scalar, bool, float, string outputs raise
        # RuntimeError about only tensors; list, dict and nested outputs work, and both the primal
        # and tangent outputs keep the output structure.
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"jvp\(f, primals, tangents\): Expected f to be a function that has non-empty output",
            ):
                jvp(lambda _: output, (x,), (t,))

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"jvp\(f, primals, tangents\): expected f\(\*primals\) to return only tensors",
            ):
                jvp(lambda _: output, (x,), (t,))

        # Check list output
        out = jvp(lambda x: [x, x.sum()], (x,), (t,))
        for i in range(2):
            if not (isinstance(out[i], list) and len(out[i]) == 2):
                raise AssertionError(
                    f"Expected list of length 2, got {type(out[i]).__name__} of length {len(out[i])}"
                )

        # Check dict output
        out = jvp(lambda x: {"x": x, "xsum": x.sum()}, (x,), (t,))
        for i in range(2):
            if not (isinstance(out[i], dict) and len(out[i]) == 2 and "xsum" in out[i]):
                raise AssertionError(
                    f"Expected dict of length 2 with 'xsum' key, got {type(out[i]).__name__}"
                )

        def composite_output(x):
            # [FUNCTION DOC] HELPER: Returns a list with a tuple of the sum and a dict of the
            # input and a list of the input and the sum.
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        out = jvp(composite_output, (x,), (t,))
        for i in range(2):
            if not isinstance(out[i], list):
                raise AssertionError(f"Expected list, got {type(out[i]).__name__}")
            if not (isinstance(out[i][0], tuple) and isinstance(out[i][0][1], dict)):
                raise AssertionError(
                    f"Expected (tuple, dict) structure, got ({type(out[i][0]).__name__}, {type(out[i][0][1]).__name__})"
                )

    def test_aux_tensor(self, device):
        # [FUNCTION DOC] Checks jvp with has_aux: a function returning a non-tuple or wrong length
        # tuple raises RuntimeError about the output needing to be a tuple; a valid function
        # returns output, tangent output and aux with expected values.
        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            r"jvp\(f, primals, tangents\): output of function f should be a tuple",
        ):
            jvp(lambda t: [t, t], (x,), (t,), has_aux=True)

        with self.assertRaisesRegex(
            RuntimeError,
            r"jvp\(f, primals, tangents\): output of function f should be a tuple",
        ):
            jvp(lambda t: (t, t + 2, t + 3), (x,), (t,), has_aux=True)

        def f(z):
            # [FUNCTION DOC] HELPER: Returns sin of the input and the cosine as aux.
            y = z.sin()
            return y, z.cos()

        out, jvp_out, aux = jvp(f, (x,), (t,), has_aux=True)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.sin())
        self.assertEqual(jvp_out, t * x.cos())

    def test_aux_pytree(self, device):
        # [FUNCTION DOC] Checks jvp with a pytree aux returns it intact with correct output and
        # tangent, and non-tensor aux values (int, float, string or lists containing them) raise
        # RuntimeError about unsupported types.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns sin of the input and an aux dict of cosine and a list
            # with tangent.
            y = x.sin()
            return y, {"a": x.cos(), "b": [x.tan()]}

        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        out, jvp_out, aux = jvp(f, (x,), (t,), has_aux=True)
        expected_out, expected_aux = f(x)
        self.assertEqual(out, expected_out)
        self.assertEqual(aux, expected_aux)
        self.assertEqual(jvp_out, t * x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jvp(lambda x: (x, aux), (x,), (t,), has_aux=True)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jvp(lambda x: (x, [x, aux]), (x,), (t,), has_aux=True)

    def test_autograd_function_disables_fwd_grad(self, device):
        # [FUNCTION DOC] Sanity check that the forward of a custom autograd Function runs with
        # forward-mode gradient tracking disabled (asserted in the forward through the
        # enclosing test).
        # Sanity check. We don't really assume this anywhere so
        # it's fine if this breaks one day.
        class MySquare(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function for squaring; its forward asserts forward grad
            # is disabled and its backward returns the gradient unchanged.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Asserts forward-mode grad is disabled and returns the
                # square of the input.
                enabled = fwAD._is_fwd_grad_enabled()
                self.assertFalse(enabled)
                return x * x

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        x = torch.randn(3, requires_grad=True)
        MySquare.apply(x)

    def test_disable_fwd_grad_outside(self, device):
        # [FUNCTION DOC] Checks jvp called inside a block disabling forward-mode grad still
        # computes the derivative (cosine).
        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        with fwAD._set_fwd_grad_enabled(False):
            _, y = jvp(torch.sin, (x,), (t,))
        self.assertEqual(y, x.cos())

    def test_disable_fwd_grad_inside(self, device):
        # [FUNCTION DOC] Checks that disabling forward-mode grad inside the function makes the
        # shifted value a constant: jvp gives two x and a nested jvp gives two.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes a square with forward grad disabled as a shift and
            # returns the square minus the shift.
            with fwAD._set_fwd_grad_enabled(False):
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        _, y = jvp(f, (x,), (t,))
        self.assertEqual(y, 2 * x)
        _, y = jvp(lambda x: jvp(f, (x,), (t,))[1], (x,), (t,))
        self.assertEqual(y, 2)

    def test_disable_fwd_grad_mixed(self, device):
        # [FUNCTION DOC] Checks that jvp inside an outer block enabling forward grad and a
        # function that disables it internally still gives two x.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes a square with forward grad disabled as a shift and
            # returns the square minus the shift.
            with fwAD._set_fwd_grad_enabled(False):
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        with fwAD._set_fwd_grad_enabled(True):
            _, y = jvp(f, (x,), (t,))

        self.assertEqual(y, 2 * x)

    def test_jvp_inside_autograd_function(self, device):
        # [FUNCTION DOC] Checks jvp can be used inside the forward and backward of a custom
        # autograd Function: a Function computing sin via negated jvp of cos in forward and jvp of
        # sin in backward returns sin, and regular autograd gives cosine.
        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function implementing sin. Forward negates the tangent
            # output of jvp of cos with unit tangent and saves the input; backward multiplies the
            # gradient by the jvp tangent of sin with unit tangent.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves x for backward and returns minus the tangent output
                # of jvp of cos at x with a ones tangent (which is sin x).
                t = torch.ones_like(x)
                _, neg_sin_x = jvp(torch.cos, (x,), (t,))
                ctx.save_for_backward(x)
                return -neg_sin_x

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Loads the saved input and returns the gradient times the
                # tangent output of jvp of sin at it with a ones tangent (cosine).
                (x,) = ctx.saved_tensors
                t = torch.ones_like(x)
                _, cos_x = jvp(torch.sin, (x,), (t,))
                return gx * cos_x

        x = torch.randn([], device=device, requires_grad=True)
        y = MySin.apply(x)
        self.assertEqual(y, x.sin())

        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    def test_zerotensor_vmapjvp_interaction(self, device):
        # [FUNCTION DOC] Regression check that jvp of covariance nested inside two levels of vmap,
        # with one vmap over an unbatched input, does not raise (no value assertion).
        dummy = torch.ones(4, 1)
        x = torch.randn(4, 2)
        x_tangent = torch.randn(2)

        def push_jvp(dummy, x):
            # [FUNCTION DOC] HELPER: Returns the jvp result of the covariance function at x using
            # the captured tangent.
            result = jvp(torch.cov, (x,), (x_tangent,))
            return result

        # Should not error
        vmap(vmap(push_jvp, (0, None)))(dummy, x)


@markDynamoStrictTest
class TestLinearize(TestCase):
    # [CLASS DOC] Device-generic tests for linearize, which returns the output and a jvp function;
    # each test compares with jvp or vmap/grad composition. Tests run for float dtype only and
    # several skip under the CUDA memory leak check.
    @dtypes(torch.float)
    def test_linearize_basic(self, device, dtype):
        # [FUNCTION DOC] Checks that linearize of cosine gives the same output as jvp and that its
        # returned jvp function applied to a tangent gives the same tangent output.
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of the input.
            return x.cos()

        actual_output, jvp_fn = linearize(fn, x_p)
        actual_jvp = jvp_fn(x_t)
        expected_output, expected_jvp = jvp(fn, (x_p,), (x_t,))
        self.assertEqual(actual_output, expected_output)
        self.assertEqual(actual_jvp, expected_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_return(self, device, dtype):
        # [FUNCTION DOC] Same as the basic check but for a function returning a tuple of cosine
        # and sum. Skipped when the CUDA memory leak check is enabled (known leak).
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a tuple of cosine and the sum of the input.
            return (x.cos(), x.sum())

        actual_output, jvp_fn = linearize(fn, x_p)
        actual_jvp = jvp_fn(x_t)
        expected_output, expected_jvp = jvp(fn, (x_p,), (x_t,))
        self.assertEqual(actual_output, expected_output)
        self.assertEqual(actual_jvp, expected_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_composition_vmap(self, device, dtype):
        # [FUNCTION DOC] Checks that vmap over the jvp function returned by linearize equals vmap
        # over a direct jvp wrapper, for a batch of tangents. Skipped under the CUDA memory
        # leak check.
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 3, 1), device=device, dtype=dtype)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a tuple of cosine and the sum of the input.
            return (x.cos(), x.sum())

        _, jvp_fn = linearize(fn, x_p)
        actual_batched_jvp = vmap(jvp_fn)(x_t)

        def jvp_fn(x_t):
            # [FUNCTION DOC] HELPER: Replacement (shadows the linearize one) returning the tangent
            # output of a direct jvp of the function at the primal.
            return jvp(fn, (x_p,), (x_t,))[1]

        expected_batched_jvp = vmap(jvp_fn)(x_t)

        self.assertEqual(actual_batched_jvp, expected_batched_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_composition_grad(self, device, dtype):
        # [FUNCTION DOC] Checks linearize of a function that calls grad internally gives the same
        # tangent as a direct jvp. Skipped under the CUDA memory leak check.
        x_p = make_tensor((3,), device=device, dtype=dtype)
        x_t = make_tensor((3,), device=device, dtype=dtype)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the grad of a dot product with a ones vector
            # evaluated at the input.
            z = torch.ones(3, device=device, dtype=dtype)
            return grad(lambda x: z @ x)(x)

        _, jvp_fn = linearize(fn, x_p)
        actual_batched_jvp = jvp_fn(x_t)

        def jvp_fn(x_t):
            # [FUNCTION DOC] HELPER: Replacement (shadows the linearize one) returning the tangent
            # output of a direct jvp of the function at the primal.
            return jvp(fn, (x_p,), (x_t,))[1]

        expected_batched_jvp = jvp_fn(x_t)

        self.assertEqual(actual_batched_jvp, expected_batched_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_nested_input_nested_output(self, device, dtype):
        # [FUNCTION DOC] Checks linearize with a dict input (containing a tuple) and a nested dict
        # output equals jvp for output and tangent. Skipped under the CUDA memory leak check.
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)
        y_p = make_tensor((3, 1), device=device, dtype=dtype)
        y_t = make_tensor((3, 1), device=device, dtype=dtype)
        z_p = make_tensor((3, 1), device=device, dtype=dtype)
        z_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(arg):
            # [FUNCTION DOC] HELPER: Reads x and a pair y,z from the dict input and returns a dict
            # with the sum of x and a nested dict of y plus z and a tuple of x times z and exp
            # of y.
            x = arg["x"]
            y = arg["yz"][0]
            z = arg["yz"][1]

            return {"a": x.sum(), "b": {"c": y + z, "d": (x * z, y.exp())}}

        inp_p = {"x": x_p, "yz": (y_p, z_p)}
        inp_t = {"x": x_t, "yz": (y_t, z_t)}
        actual_output, jvp_fn = linearize(fn, inp_p)
        actual_jvp = jvp_fn(inp_t)

        expected_output, expected_jvp = jvp(fn, (inp_p,), (inp_t,))

        self.assertEqual(actual_output, expected_output)
        self.assertEqual(actual_jvp, expected_jvp)

    @onlyCUDA
    def test_linearize_errors(self):
        # [FUNCTION DOC] CUDA only. Using CPU tensors, checks that the jvp function from linearize
        # raises RuntimeError when given tangents with a different structure, shape, dtype or
        # device (moved to cuda) than the primals, with messages naming the mismatch.
        dtype = torch.float
        device = torch.device("cpu")
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sin of the input.
            return x.sin()

        _, jvp_fn = linearize(fn, x_p)

        with self.assertRaisesRegex(
            RuntimeError, "to have the same argspec as the primals"
        ):
            jvp_fn((x_t, x_t))

        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the shape"
        ):
            jvp_fn(x_t.unsqueeze(0))

        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the dtype"
        ):
            jvp_fn(x_t.to(torch.double))

        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the device"
        ):
            jvp_fn(x_t.to(torch.device("cuda")))


# The tests here follow the cases in [Forward Grad View/inplace]
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/autograd_meta.cpp#L18-L43
@markDynamoStrictTest
class TestVmapJvpInplaceView(TestCase):
    # [CLASS DOC] Tests of forward-mode view and in-place semantics under vmap of jvp, following
    # the numbered cases of the forward grad view and in-place notes in the autograd metadata
    # source. Each test compares against vmap of the function alone and checks tangent propagation
    # between views and bases.
    # Case 1 in [Forward Grad View/inplace]
    def test_all_dual_no_view(self, device):
        # [FUNCTION DOC] Case 1: all inputs are dual and there is no view. A function copies y
        # into x in place and returns x. Checks outputs equal the batched primal result and
        # tangents equal the y tangents, including a second run with a non-contiguous, unbatched y
        # and yt.
        B = 2

        def push_jvp(f):
            # [FUNCTION DOC] HELPER: Returns an inner function that runs jvp of f on the given
            # primals and tangents.
            def inner(x, xt, y, yt):
                # [FUNCTION DOC] HELPER: Runs jvp of the outer function at x,y with tangents xt,yt
                # and returns the result.
                return jvp(f, (x, y), (xt, yt))

            return inner

        def f(x, y):
            # [FUNCTION DOC] HELPER: Copies y into x in place and returns x.
            x.copy_(y)
            return x

        x = torch.randn(3, B, device=device)
        xt = torch.randn(3, B, device=device)
        y = torch.randn(3, B, device=device)
        yt = torch.randn(3, B, device=device)
        out, out_tangent = vmap(push_jvp(f), in_dims=1)(x, xt, y, yt)
        self.assertEqual(out, x.movedim(1, 0))
        self.assertEqual(out_tangent, yt.movedim(1, 0))

        x = torch.randn(3, B, device=device)
        xt = torch.randn(3, B, device=device)
        y = torch.randn(3, 3, device=device)[:, 1]
        yt = torch.randn(6, device=device)[::2]
        out, out_tangent = vmap(push_jvp(f), in_dims=(1, 1, None, None))(x, xt, y, yt)
        self.assertEqual(out, x.movedim(1, 0))
        self.assertEqual(out_tangent, yt.expand(B, 3))

    # Case 2 in [Forward Grad View/inplace]
    def test_all_dual_base_view_inplace(self, device):
        # [FUNCTION DOC] Case 2: all inputs dual, an in-place copy into a strided view of x must
        # propagate to the base. Checks outputs against vmap of the plain function, the view
        # tangent equals the y tangent, the base tangent has the strided slice replaced, and the
        # primal x was updated in place.
        B = 2

        def push_jvp(f):
            # [FUNCTION DOC] HELPER: Returns an inner function that runs jvp of f on the given
            # primals and tangents.
            def inner(x, xt, y, yt):
                # [FUNCTION DOC] HELPER: Runs jvp of the outer function at x,y with tangents xt,yt
                # and returns the result.
                return jvp(f, (x, y), (xt, yt))

            return inner

        # with view, propagate from view to base
        def f(x, y):
            # [FUNCTION DOC] HELPER: Takes a strided view of x, copies y into it in place and
            # returns the view and x.
            view = x[:, ::2]
            view.copy_(y)
            return view, x

        orig_x = torch.randn(2, 6, B, device=device)
        orig_xt = torch.randn(2, 6, B, device=device)
        x = orig_x.clone()
        xt = orig_xt.clone()
        y = torch.randn(2, B, 3, device=device)
        yt = torch.randn(2, B, 3, device=device)
        out, out_tangent = vmap(push_jvp(f), in_dims=(2, 2, 1, 1))(x, xt, y, yt)

        expected_out = vmap(f, in_dims=(2, 1))(orig_x.clone(), y)
        self.assertEqual(out[0], expected_out[0])
        self.assertEqual(out[1], expected_out[1])

        self.assertEqual(out_tangent[0], yt.movedim(1, 0))

        expected_x_tangent = orig_xt.movedim(-1, 0).clone()
        expected_x_tangent[:, :, ::2].copy_(yt.movedim(1, 0))
        self.assertEqual(out_tangent[1], expected_x_tangent)

        expected = orig_x.movedim(2, 0).clone()
        expected[:, :, ::2] = y.movedim(1, 0)
        self.assertEqual(x.movedim(2, 0), expected)

    # Case 3 in [Forward Grad View/inplace]
    def test_all_dual_base_inplace(self, device):
        # [FUNCTION DOC] Case 3: all inputs dual, an in-place copy into the base must propagate to
        # previously made views. Checks outputs against vmap of the plain function, the base
        # tangent equals the y tangent, and the view tangent is the corresponding strided slice
        # of it.
        B = 2

        def push_jvp(f):
            # [FUNCTION DOC] HELPER: Returns an inner function that runs jvp of f on the given
            # primals and tangents.
            def inner(x, xt, y, yt):
                # [FUNCTION DOC] HELPER: Runs jvp of the outer function at x,y with tangents xt,yt
                # and returns the result.
                return jvp(f, (x, y), (xt, yt))

            return inner

        # Case 3: with view, propagate from base to view
        def f(x, y):
            # [FUNCTION DOC] HELPER: Takes a strided view of x, copies y into x in place and
            # returns x and the view.
            view = x[0, ::2]
            x.copy_(y)
            return x, view

        x = torch.randn(2, B, 6, device=device)
        xt = torch.randn(2, 6, B, device=device)
        y = torch.randn(2, B, 6, device=device)
        yt = torch.randn(2, B, 6, device=device)
        out, out_tangent = vmap(push_jvp(f), in_dims=(1, 2, 1, 1))(x.clone(), xt, y, yt)

        expected_out = vmap(f, in_dims=(1, 1))(x.clone(), y)
        self.assertEqual(out[0], expected_out[0])
        self.assertEqual(out[1], expected_out[1])

        self.assertEqual(out_tangent[0], yt.movedim(1, 0))
        self.assertEqual(out_tangent[1], yt.movedim(1, 0)[:, 0, ::2])

    # Case 4 in [Forward Grad View/inplace]
    def test_right_dual_view_prop(self, device):
        # [FUNCTION DOC] Case 4: x is a regular tensor and y is dual; an in-place copy into a view
        # of a clone must propagate to its base. Checks outputs against the plain batched
        # function, the view tangent equals the y tangent, and the base tangent is zero except the
        # copied row.
        B = 2

        # Changes on the view must propagate to its base. Also:
        # - x is a regular Tensor
        # - y is a dual tensor
        def f(x, y):
            # [FUNCTION DOC] HELPER: Clones x, takes its first row as a view, copies y into the
            # view in place and returns the view and the clone.
            x = x.clone()
            view = x[0]
            view.copy_(y)
            return view, x

        def push_jvp(x, y, yt):
            # [FUNCTION DOC] HELPER: Runs jvp of f with x fixed by partial application, with
            # primal y and tangent yt.
            return jvp(partial(f, x), (y,), (yt,))

        x = torch.randn(2, B, 6, device=device)
        y = torch.randn(6, B, device=device)
        yt = torch.randn(6, B, device=device)
        outs, tangents = vmap(push_jvp, in_dims=(1, 1, 1))(x, y, yt)

        expected_out = vmap(f, in_dims=(1, 1))(x.clone(), y)
        self.assertEqual(outs[0], expected_out[0])
        self.assertEqual(outs[1], expected_out[1])

        self.assertEqual(tangents[0], yt.movedim(1, 0))

        expected_tangent_1 = torch.zeros_like(x).movedim(1, 0)
        expected_tangent_1[:, 0].copy_(yt.movedim(1, 0))
        self.assertEqual(tangents[1], expected_tangent_1)

    # Case 5 in [Forward Grad View/inplace]
    def test_right_dual_base_prop(self, device):
        # [FUNCTION DOC] Case 5: x is a regular tensor and y is dual; an in-place copy into the
        # base of a clone must propagate to existing views. Checks outputs against the plain
        # batched function and that the base tangent equals the y tangent and the view tangent is
        # its first row.
        B = 2

        # Changes on the base must propagate on all its views. Also:
        # - x is a regular Tensor
        # - y is a dual tensor
        def f(x, y):
            # [FUNCTION DOC] HELPER: Clones x, takes its first row as a view, copies y into the
            # clone in place and returns the view and the clone.
            x = x.clone()
            view = x[0]
            x.copy_(y)
            return view, x

        def push_jvp(x, y, yt):
            # [FUNCTION DOC] HELPER: Runs jvp of f with x fixed by partial application, with
            # primal y and tangent yt.
            return jvp(partial(f, x), (y,), (yt,))

        x = torch.randn(2, B, 6)
        y = torch.randn(2, 6, B)
        yt = torch.randn(2, 6, B)
        outs, tangents = vmap(push_jvp, in_dims=(1, 2, 2))(x, y, yt)

        expected_out = vmap(f, in_dims=(1, 2))(x, y)
        self.assertEqual(outs[0], expected_out[0])
        self.assertEqual(outs[1], expected_out[1])

        self.assertEqual(tangents[0], yt.movedim(2, 0)[:, 0])
        self.assertEqual(tangents[1], yt.movedim(2, 0))


# Use for testing miscellaneous helper functions
@markDynamoStrictTest
class TestHelpers(TestCase):
    # [CLASS DOC] Device-generic tests for miscellaneous helpers: the saved-tensors context
    # wrapper, the debug unwrap utility and the leaf reduction helper of the autograd function
    # support code.
    def test_CtxWithSavedTensors_error_if_name_collision(self, device):
        # [FUNCTION DOC] Checks that wrapping a context with the saved-tensors wrapper raises
        # RuntimeError mentioning a name collision when the context already has either of the two
        # reserved private attribute names (checked with two custom Functions, one per name,
        # in backward).
        x = torch.randn([], device=device, requires_grad=True)
        y = torch.randn([], device=device, requires_grad=True)

        class A(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function that sets the reserved inner context attribute
            # in forward and wraps the context in backward, which should raise.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Sets a reserved attribute on the context, saves x and
                # returns it.
                ctx._pt_inner_ctx = 1
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # [FUNCTION DOC] HELPER: Wraps the context with the saved-tensors wrapper using
                # the outer y; returns the gradient (not reached when the wrapper raises).
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(  # noqa: F841
                    ctx, (y,)
                )
                return gy

        class B(torch.autograd.Function):
            # [CLASS DOC] HELPER: Same as the previous helper Function but sets the other reserved
            # attribute (new saved tensors) in forward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Sets the other reserved attribute on the context, saves x
                # and returns it.
                ctx._pt_new_saved_tensors = 1
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # [FUNCTION DOC] HELPER: Wraps the context with the saved-tensors wrapper using
                # the outer y; returns the gradient (not reached when the wrapper raises).
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(  # noqa: F841
                    ctx, (y,)
                )
                return gy

        out = A.apply(x)
        with self.assertRaisesRegex(RuntimeError, "name collision"):
            out.backward()
        out = B.apply(x)
        with self.assertRaisesRegex(RuntimeError, "name collision"):
            out.backward()

    def test_CtxWithSavedTensors_nesting(self, device):
        # [FUNCTION DOC] Checks that the saved-tensors wrapper nests: wrapping a wrapper replaces
        # saved tensors only in the outer view while the inner one is unchanged, and the backward
        # result uses the outermost override, giving the gradient equal to z.
        CtxWithSavedTensors = torch._functorch.autograd_function.CtxWithSavedTensors
        x = torch.randn([], device=device, requires_grad=True)
        y = torch.randn([], device=device)
        z = torch.randn([], device=device)

        class A(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose backward builds two nested wrappers over
            # the context with override tuples y and z, verifies their saved tensors with plain
            # checks (not test assertions, since autograd runs backward on another thread) and
            # returns the gradient times z.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves x and returns it.
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # [FUNCTION DOC] HELPER: Wraps the context with y then z, raising AssertionError
                # if any wrapper reports the wrong saved tensors; returns the gradient times the
                # outer saved tensor.
                ctx_y = CtxWithSavedTensors(ctx, (y,))
                # Can't use self.assertEqual because that relies on TLS
                # that is not available in multithread autograd
                if len(ctx_y.saved_tensors) != 1:
                    raise AssertionError(
                        f"Expected len(ctx_y.saved_tensors) == 1, got {len(ctx_y.saved_tensors)}"
                    )
                if not torch.allclose(ctx_y.saved_tensors[0], y):
                    raise AssertionError(
                        "torch.allclose failed: ctx_y.saved_tensors[0] and y differ"
                    )

                wrapped = CtxWithSavedTensors(ctx_y, (z,))

                if len(wrapped.saved_tensors) != 1:
                    raise AssertionError(
                        f"Expected len(wrapped.saved_tensors) == 1, got {len(wrapped.saved_tensors)}"
                    )
                if not torch.allclose(wrapped.saved_tensors[0], z):
                    raise AssertionError(
                        "torch.allclose failed: wrapped.saved_tensors[0] and z differ"
                    )

                if len(ctx_y.saved_tensors) != 1:
                    raise AssertionError(
                        f"Expected len(ctx_y.saved_tensors) == 1, got {len(ctx_y.saved_tensors)}"
                    )
                if not torch.allclose(ctx_y.saved_tensors[0], y):
                    raise AssertionError(
                        "torch.allclose failed: ctx_y.saved_tensors[0] and y differ"
                    )

                return gy * wrapped.saved_tensors[0]

        out = A.apply(x)
        out.backward()
        self.assertEqual(x.grad, z)

    def test_CtxWithSavedTensors_overrides_saved_tensors(self, device):
        # [FUNCTION DOC] Checks the saved-tensors wrapper returns exactly the override object
        # given (even non-tensor data) as its saved_tensors, verified inside backward.
        x = torch.randn([], device=device, requires_grad=True)

        class A(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose backward wraps the context with a tuple of
            # integers and checks the wrapper reports it.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves x and returns it.
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # [FUNCTION DOC] HELPER: Wraps the context with an integer tuple, raises
                # AssertionError if saved_tensors differs, returns the gradient.
                # The override can be literally anything
                override = (1, 2, 3)
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(
                    ctx, override
                )
                if wrapped.saved_tensors != override:
                    raise AssertionError(
                        f"Expected wrapped.saved_tensors == {override}, got {wrapped.saved_tensors}"
                    )
                return gy

        out = A.apply(x)
        out.backward()

    def test_CtxWithSavedTensors_passthrough(self, device):
        # [FUNCTION DOC] Checks that the saved-tensors wrapper passes needs_input_grad through
        # from the wrapped context, and that setting an attribute on the wrapper is visible on the
        # original context.
        x = torch.randn([], device=device, requires_grad=True)
        y = torch.randn([], device=device)

        class A(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function of two inputs that multiplies them; its backward
            # checks the wrapper delegates needs_input_grad and attribute assignment to
            # the context.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Saves x and y and returns their product.
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, gz):
                # [FUNCTION DOC] HELPER: Wraps the context, raises AssertionError if the
                # needs_input_grad entries differ or if an attribute set through the wrapper is
                # not visible on both; returns the gradient for both inputs.
                # The override can be literally anything
                override = (1, 2, 3)
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(
                    ctx, override
                )

                if wrapped.needs_input_grad[0] != ctx.needs_input_grad[0]:
                    raise AssertionError(
                        f"needs_input_grad[0] mismatch: wrapped={wrapped.needs_input_grad[0]}, ctx={ctx.needs_input_grad[0]}"
                    )
                if wrapped.needs_input_grad[1] != ctx.needs_input_grad[1]:
                    raise AssertionError(
                        f"needs_input_grad[1] mismatch: wrapped={wrapped.needs_input_grad[1]}, ctx={ctx.needs_input_grad[1]}"
                    )
                wrapped.foo = "bar"
                if wrapped.foo != "bar":
                    raise AssertionError(
                        f"Expected wrapped.foo == 'bar', got {wrapped.foo!r}"
                    )
                if ctx.foo != "bar":
                    raise AssertionError(f"Expected ctx.foo == 'bar', got {ctx.foo!r}")
                return gz, gz

        out = A.apply(x, y)
        out.backward()

    def test_debug_unwrap(self):
        # [FUNCTION DOC] Checks the debug unwrap utility applied under two nested vmaps returns
        # the original unbatched input tensor itself (identity).
        stuff = []

        def f(x):
            # [FUNCTION DOC] HELPER: Appends the debug-unwrapped argument to the outer list and
            # returns the sine.
            stuff.append(torch.func.debug_unwrap(x))
            return x.sin()

        x = torch.randn(2, 3)
        _ = vmap(vmap(f))(x)
        self.assertEqual(stuff[0], x)
        self.assertTrue(stuff[0] is x)

    def test_reductify_leaf(self, device):
        # [FUNCTION DOC] Checks the leaf reduction helper (arguments: gradient, gradient batch
        # dim, input batch dim, batch size, optionally the input shape) for the combinations: no
        # gradient gives None; gradient batched and input not gives a sum over the batch dim;
        # gradient unbatched and input batched gives an expansion (with the input shape given,
        # summed over the extra dims); both batched moves the batch dim, summing extra dims if the
        # input shape is given.
        reductify_leaf = torch._functorch.autograd_function.reductify_leaf
        B = 2

        # grad_input None case
        output = reductify_leaf(None, None, 0, B)
        self.assertIsNone(output)
        output = reductify_leaf(None, None, None, B)
        self.assertIsNone(output)

        # grad_input has bdim, input does not have bdim
        grad_input = torch.randn([B, 3, 4], device=device)
        output = reductify_leaf(grad_input, 0, None, B)
        self.assertEqual(output, grad_input.sum(0))

        grad_input = torch.randn([3, B, 4], device=device)
        output = reductify_leaf(grad_input, 1, None, B, (3,))
        self.assertEqual(output, grad_input.sum(1))

        # grad_input does not have bdim, input has bdim
        # This can happen if the user returns a fresh Tensor from the backward pass
        # that is unrelated to the input
        grad_input = torch.randn([3, 4], device=device)
        output = reductify_leaf(grad_input, None, 1, B)
        self.assertEqual(output, grad_input.view(3, 1, 4).expand(3, B, 4))

        grad_input = torch.randn([3, 4], device=device)
        output = reductify_leaf(grad_input, None, 1, B, (4,))
        self.assertEqual(output, grad_input.view(3, 4, 1).expand(3, 4, B).sum(0))

        # grad_input has bdim, input has bdim
        grad_input = torch.randn([B, 3, 4], device=device)
        output = reductify_leaf(grad_input, 0, 1, B)
        self.assertEqual(output, grad_input.movedim(0, 1))

        grad_input = torch.randn([3, 4, 5, B], device=device)
        output = reductify_leaf(grad_input, 3, 0, B, (5,))
        self.assertEqual(output, grad_input.movedim(-1, 2).sum(0).sum(0))


@markDynamoStrictTest
class TestComposability(TestCase):
    # [CLASS DOC] Device-generic tests for composing transforms (grad, vmap, vjp, jacrev, make_fx)
    # in different orders, deprecation warnings of the functorch namespace and errors for
    # unsupported calls inside transforms.
    def test_deprecation_vmap(self, device):
        # [FUNCTION DOC] Checks the functorch vmap emits a FutureWarning recommending the
        # torch.vmap and that torch.vmap emits no warning (warnings turned to errors).
        # functorch version of the API is deprecated
        with self.assertWarnsRegex(FutureWarning, "Please use `torch.vmap`"):
            vmap(torch.sin)

        # the non-functorch version is not deprecated
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch.vmap(torch.sin)

    # Some of these pass, some of these don't
    @parametrize(
        "transform",
        ["grad", "jacrev", "jacfwd", "grad_and_value", "hessian", "functionalize"],
    )
    def test_deprecation_transforms(self, device, transform):
        # [FUNCTION DOC] Parametrized over several transform names. For each, checks the functorch
        # namespace version emits a FutureWarning recommending the matching torch.func name, and
        # that the torch.func version emits no warning when warnings are errors.
        api = getattr(functorch, transform)
        new_api = getattr(torch.func, transform)

        # functorch version of the API is deprecated
        with self.assertWarnsRegex(
            FutureWarning, f"Please use `torch.func.{transform}`"
        ):
            api(torch.sin)

        # the non-functorch version is not deprecated
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            new_api(torch.sin)

    def test_grad_grad(self, device):
        # [FUNCTION DOC] Checks grad of grad of sin gives minus sin.
        x = torch.randn([], device=device)
        y = grad(grad(torch.sin))(x)
        self.assertEqual(y, -x.sin())

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/179877")
    def test_grad_vmap(self, device):
        # [FUNCTION DOC] Checks grad over a function that uses vmap of sin and sums gives cosine.
        # Skipped under Dynamo (issue 179877).
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the sum of vmap of sin applied to x.
            y = vmap(torch.sin)(x)
            return y.sum()

        x = torch.randn(3, device=device)
        y = grad(foo)(x)
        self.assertEqual(y, x.cos())

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/180894")
    def test_grad_vjp(self, device):
        # [FUNCTION DOC] Checks grad over a function that uses vjp matches grad of x times cosine
        # x summed. Skipped under Dynamo (issue 180894).
        x = torch.randn(3, device=device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the vjp of sin at x with cotangent x.
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)[0].sum()

        y = grad(foo)(x)
        expected = grad(lambda x: (x * x.cos()).sum())(x)
        self.assertEqual(y, expected)

    def test_vmap_grad(self, device):
        # [FUNCTION DOC] Checks vmap of grad of sin gives cosine.
        x = torch.randn(3, device=device)
        y = vmap(grad(torch.sin))(x)
        self.assertEqual(y, x.cos())

    def test_vmap_vmap(self, device):
        # [FUNCTION DOC] Checks nested vmap of sin equals sin.
        x = torch.randn(2, 3, device=device)
        y = vmap(vmap(torch.sin))(x)
        self.assertEqual(y, x.sin())

    def test_vmap_vjp(self, device):
        # [FUNCTION DOC] Checks vmap over a function that builds a vjp inside gives the same as
        # the outside vjp function, and vmap of an outside vjp function over a batch of cotangents
        # equals stacking per-sample calls.
        x = torch.randn(3, device=device)
        _, vjp_fn = vjp(torch.sin, x)

        def foo(x):
            # [FUNCTION DOC] HELPER: Builds a vjp of sin at x and returns its result applied to x.
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)

        y = vmap(foo)(x)
        self.assertEqual(y, vjp_fn(x))

        # TODO: there's a very interesting error message when the following
        # is on CPU
        xs = torch.randn(5, 3, device=device)
        expected = torch.stack([vjp_fn(x)[0] for x in xs])
        result = vmap(lambda x: vjp_fn(x)[0])(xs)
        self.assertEqual(result, expected)

    def test_vjp_grad(self, device):
        # [FUNCTION DOC] Checks vjp of grad of sin: the output is cosine and the vjp function
        # gives minus sin times the cotangent.
        x = torch.randn([], device=device)
        y, vjp_fn = vjp(grad(torch.sin), x)
        self.assertEqual(y, x.cos())

        v = torch.randn([])
        self.assertEqual(vjp_fn(v)[0], -x.sin() * v)

    def test_vjp_vmap(self, device):
        # [FUNCTION DOC] Checks vjp of a vmapped sin: output is sin and the vjp function gives
        # cosine times the cotangent.
        x = torch.randn(3, device=device)
        y, vjp_fn = vjp(vmap(torch.sin), x)
        self.assertEqual(y, x.sin())

        v = torch.randn(3, device=device)
        self.assertEqual(vjp_fn(v)[0], x.cos() * v)

    def test_vjp_vjp(self, device):
        # [FUNCTION DOC] Checks vjp of a function that calls an earlier vjp function: output
        # equals x times cosine x; the final vjp function is called only to confirm it runs
        # (result unchecked).
        x = torch.randn(3, device=device)
        y, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(y, x.sin())

        y, vjp_fn = vjp(lambda x: vjp_fn(x)[0], x)
        self.assertEqual(y, x * x.cos())

        y = vjp_fn(x)[0]
        # Honestly IDK what the result here is... but at least it runs

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/180300")
    def test_make_fx_vmap(self, device):
        # [FUNCTION DOC] Checks make_fx traced over a vmapped function gives a graph module whose
        # result on new input matches the original. Skipped under Dynamo (issue 180300).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns sin of the input.
            return torch.sin(x)

        inp = torch.randn(5, 3)
        f = vmap(f)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(5, 3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/179895")
    def test_make_fx_jacrev(self, device):
        # [FUNCTION DOC] Checks make_fx traced over nested jacrev gives a graph whose result on
        # new input matches the original. Skipped under Dynamo (issue 179895).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input.
            return x.sin().sum()

        inp = torch.randn(3)
        f = jacrev(jacrev(f))
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_vjp(self, device):
        # [FUNCTION DOC] Checks make_fx can trace a vjp function (called with extra boolean flags)
        # and the traced graph gives the same result on a new cotangent as the original
        # vjp function.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input.
            return torch.sin(x).sum()

        primals = torch.randn(3)
        _, vjp_fn = vjp(f, primals)
        cotangent = torch.randn(())
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    # FIXME: test fails in Windows
    @unittest.skipIf(IS_WINDOWS, "fails in Windows; needs investigation")
    @unittest.skipIf(IS_FBCODE, "can't subprocess in fbcode")
    # it is redundant to run this test twice on a machine that has GPUs
    @onlyCPU
    def test_no_warning_on_import_functorch(self, device):
        # [FUNCTION DOC] CPU only; skipped on Windows and in fbcode. Runs a subprocess that
        # imports functorch with all warnings enabled in the test directory and checks the
        # combined output is empty, so importing emits no warning.
        out = subprocess.check_output(
            [sys.executable, "-W", "always", "-c", "import functorch"],
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        ).decode("utf-8")
        self.assertEqual(out, "")

    def test_requires_grad_inside_transform(self, device):
        # [FUNCTION DOC] Checks calling requires_grad_ on a tensor inside vmap, grad, vmap of grad
        # and nested grad raises RuntimeError naming the requires_grad_ method.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls requires_grad_ in place on the input and returns the
            # sum of sin.
            x.requires_grad_()
            return x.sin().sum()

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            vmap(f)(x)
        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            grad(f)(x)
        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            vmap(grad(f))(x)

        x = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            grad(grad(f))(x)

    def test_retain_grad_inside_transform(self, device):
        # [FUNCTION DOC] Checks calling retain_grad on an intermediate inside grad raises
        # RuntimeError naming retain_grad.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls retain_grad on the sine of the input and returns
            # its sum.
            y = x.sin()
            y.retain_grad()
            return y.sum()

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, "Tensor.retain_grad()"):
            grad(f)(x)

    def test_autograd_functional_jacrev_inside_transform(self, device):
        # [FUNCTION DOC] Checks that calling the torch.autograd.functional Jacobian inside vmap or
        # grad raises RuntimeError mentioning torch.autograd.functional.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the autograd functional Jacobian of the sum of sin
            # at x.
            y = torch.autograd.functional.jacobian(lambda x: x.sin().sum(), x)
            return y

        B = 5
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            vmap(f)(x)

        x = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            grad(f)(x)

    def test_autograd_functional_vjp_inside_transform(self, device):
        # [FUNCTION DOC] Checks that calling the torch.autograd.functional vjp inside vmap or grad
        # raises RuntimeError mentioning torch.autograd.functional.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the autograd functional vjp of the sum of sin at x.
            y = torch.autograd.functional.vjp(lambda x: x.sin().sum(), x)
            return y

        B = 5
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            vmap(f)(x)

        x = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            grad(f)(x)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/179876")
    def test_autograd_functional_jvp_inside_transform(self, device):
        # [FUNCTION DOC] Checks that calling the torch.autograd.functional jvp inside vmap or grad
        # raises RuntimeError mentioning torch.autograd.functional. Skipped under Dynamo
        # (issue 179876).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the autograd functional jvp of the sum of sin at x
            # with a ones tangent.
            t = torch.ones_like(x)
            y = torch.autograd.functional.jvp(lambda x: x.sin().sum(), (x,), (t,))
            return y

        B = 5
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            vmap(f)(x)

        x = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            grad(f)(x)

    def test_autograd_functional_jacfwd_inside_transform(self, device):
        # [FUNCTION DOC] Checks that the autograd functional Jacobian with forward-mode strategy
        # and vectorization inside vmap raises RuntimeError saying the batching rule for the
        # make-dual op is not implemented.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the forward-mode vectorized autograd functional
            # Jacobian of the sum of sin at x.
            y = torch.autograd.functional.jacobian(
                lambda x: x.sin().sum(), x, strategy="forward-mode", vectorize=True
            )
            return y

        B = 5
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(
            RuntimeError, "Batching rule not implemented for aten::_make_dual"
        ):
            vmap(f)(x)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/180591")
    @parametrize(
        "transform",
        [
            "vmap",
            "grad",
            "jacrev",
            "jacfwd",
            "grad_and_value",
            "hessian",
            "functionalize",
        ],
    )
    def test_autograd_function_no_setup_context(self, device, transform):
        # [FUNCTION DOC] Parametrized over functorch transform names (vmap, grad, jacrev, jacfwd,
        # grad_and_value, hessian, functionalize). Checks that applying each transform to a custom
        # autograd Function that uses the old style forward with a context argument and no
        # setup_context staticmethod raises RuntimeError saying it must override setup_context.
        # Skipped under Dynamo (issue 180591).
        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function computing sin in the old style (forward takes
            # the context and saves its input there) without a setup_context override; backward
            # multiplies the gradient by cosine.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves x and returns its sine.
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, gy):
                # [FUNCTION DOC] HELPER: Returns the gradient times cosine of the saved input.
                (x,) = ctx.saved_tensors
                return gy * x.cos()

        x = torch.randn(3, device=device)
        transform = getattr(functorch, transform)
        with self.assertRaisesRegex(RuntimeError, "must override the setup_context"):
            transform(MySin.apply)(x)

    # Some of these pass, some of these don't
    @parametrize(
        "transform",
        [
            "grad",
            "jacrev",
            "grad_and_value",
            "hessian",
        ],
    )
    def test_transforms_dont_support_saved_tensor_hooks(self, device, transform):
        # [FUNCTION DOC] Parametrized over grad, jacrev, grad_and_value and hessian. Checks each
        # transform raises RuntimeError mentioning saved tensor hooks when called inside a
        # save_on_cpu context or when the transformed function itself enters such a context. (A
        # branch for functionalize exists but that name is not in the parameter list.)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input.
            return torch.sin(x).sum()

        def g(x):
            # [FUNCTION DOC] HELPER: Calls the sum-of-sin function inside a save_on_cpu
            # saved-tensor hooks context and returns the result.
            with torch.autograd.graph.save_on_cpu():
                return f(x)

        x = torch.randn(3, device=device)

        if transform == "functionalize":
            transform = functorch.experimental.functionalize
        else:
            transform = getattr(functorch, transform)
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            with torch.autograd.graph.save_on_cpu():
                transform(f)(x)

        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            transform(g)(x)

    def test_vjp_doesnt_support_saved_tensor_hooks(self, device):
        # [FUNCTION DOC] Checks vjp raises RuntimeError mentioning saved tensor hooks when called
        # inside save_on_cpu or when the function enters one.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input.
            return torch.sin(x).sum()

        def g(x):
            # [FUNCTION DOC] HELPER: Calls the sum-of-sin function inside a save_on_cpu context
            # and returns the result.
            with torch.autograd.graph.save_on_cpu():
                return f(x)

        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            with torch.autograd.graph.save_on_cpu():
                vjp(f, x)

        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            vjp(g, x)

    def test_jvp_supports_saved_tensor_hooks(self, device):
        # [FUNCTION DOC] Smoke test: jvp works both inside a save_on_cpu context and with a
        # function that enters one (no assertions).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of the input.
            return torch.sin(x).sum()

        def g(x):
            # [FUNCTION DOC] HELPER: Calls the sum-of-sin function inside a save_on_cpu context
            # and returns the result.
            with torch.autograd.graph.save_on_cpu():
                return f(x)

        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        # smoke tests
        with torch.autograd.graph.save_on_cpu():
            jvp(f, (x,), (t,))

        # smoke tests
        jvp(g, (x,), (t,))

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/184862")
    def test_can_use_functionalize_when_key_is_excluded(self, device):
        # [FUNCTION DOC] Checks functionalize still works inside make_fx when the Functionalize
        # dispatch key is in the thread-local exclude set: the traced graph code has no in-place
        # sin, gives the same result as eager, and the exclude set still contains the key
        # afterwards. Skipped under Dynamo (issue 184862).
        def f(x):
            # [FUNCTION DOC] HELPER: Clones the input, applies in-place sin and returns it.
            y = x.clone()
            y.sin_()
            return y

        x = torch.randn([], device=device)
        expected = f(x)

        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
            gm = make_fx(functorch.functionalize(f))(x)
            self.assertTrue("sin_" not in gm.code)
            self.assertEqual(gm(x), expected)

            local_exclude_set = torch._C._dispatch_tls_local_exclude_set()
            self.assertTrue(local_exclude_set.has(DispatchKey.Functionalize))

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/180592")
    def test_can_use_vmap_when_key_is_excluded(self, device):
        # [FUNCTION DOC] Checks vmap works with the batched functorch dispatch key in the
        # thread-local exclude set, gives the same result as without, and that the exclude set
        # still contains the key. Skipped under Dynamo (issue 180592).
        def f(x):
            # [FUNCTION DOC] HELPER: Sums the input over dim zero.
            return x.sum(0)

        x = torch.randn(3, device=device)
        expected = vmap(f)(x)

        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.FuncTorchBatched)):
            result = vmap(f)(x)
            self.assertEqual(result, expected)
            local_exclude_set = torch._C._dispatch_tls_local_exclude_set()
            self.assertTrue(local_exclude_set.has(DispatchKey.FuncTorchBatched))

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/181155")
    def test_can_use_grad_when_key_is_excluded(self, device):
        # [FUNCTION DOC] Checks grad works with the Autograd dispatch key in the thread-local
        # exclude set, gives the same result as without, and that the exclude set still contains
        # the key. Skipped under Dynamo (issue 181155).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns sin of the input.
            return x.sin()

        x = torch.randn([], device=device)
        expected = grad(f)(x)

        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Autograd)):
            result = grad(f)(x)
            self.assertEqual(result, expected)
            local_exclude_set = torch._C._dispatch_tls_local_exclude_set()
            self.assertTrue(local_exclude_set.has(DispatchKey.Autograd))


@markDynamoStrictTest
class TestMakeFunctional(TestCase):
    # [CLASS DOC] Tests for make_functional, make_functional_with_buffers, functional_call,
    # combine_state_for_ensemble and stack_module_state: autograd tracking flags, parameter and
    # buffer tying, correctness against plain modules, error cases and state restoration. Not
    # device-generic (CPU).
    @parametrize("disable_autograd_tracking", [True, False])
    def test_disable_autograd_tracking(self, disable_autograd_tracking):
        # [FUNCTION DOC] Parametrized over the disable_autograd_tracking flag. Checks that
        # make_functional on a single linear layer returns two parameters whose requires_grad is
        # the opposite of the flag.
        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer from 3 to 3 features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                x = self.linear(x)
                return x

        mod = Foo()
        _, params = make_functional(
            mod, disable_autograd_tracking=disable_autograd_tracking
        )
        self.assertEqual(len(params), 2)
        for param in params:
            self.assertEqual(param.requires_grad, not disable_autograd_tracking)

    def test_parameter_tying(self):
        # [FUNCTION DOC] Checks make_functional handles a module whose bias parameter is shared
        # with a linear layer and whose linear layer is also registered twice: only two distinct
        # parameters are returned and the functional call with parameters from a reseeded copy
        # equals the module output.
        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Module with a shared bias parameter (also assigned as the linear
            # layer bias) and a second reference to the same linear layer; the forward applies the
            # layer twice then adds the bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the bias parameter and linear layer, ties the
                # layer bias to the parameter and registers the layer under a second name.
                super().__init__()
                self.bias = nn.Parameter(torch.randn(3))
                self.linear = nn.Linear(3, 3)
                self.linear.bias = self.bias
                self.linear_tied = self.linear

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer, the tied alias, then adds
                # the bias.
                x = self.linear(x)
                x = self.linear_tied(x)
                x = x + self.bias
                return x

        torch.manual_seed(1)
        mod = Foo()
        func, _ = make_functional(mod)

        torch.manual_seed(0)
        mod = Foo()
        _, params = make_functional(mod)
        self.assertEqual(len(params), 2)

        x = torch.randn(2, 3)
        result = func(params, x)
        expected = mod(x)
        self.assertEqual(result, expected)

    def test_buffer_tying(self):
        # [FUNCTION DOC] Checks make_functional_with_buffers with a module that registers the same
        # buffer under two names: three parameters and one buffer are reported, and the functional
        # call with a reseeded copy equals the module output.
        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Module with a bias parameter, a linear layer, and a buffer
            # registered twice; forward applies the layer and adds the bias and both
            # buffer references.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the bias, linear layer and a buffer, plus a
                # second reference to the same buffer.
                super().__init__()
                self.bias = nn.Parameter(torch.randn(3))
                self.linear = nn.Linear(3, 3)
                self.buffer = nn.Buffer(torch.randn(3))
                self.buffer_tied = self.buffer

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and adds the bias, the buffer
                # and its alias.
                x = self.linear(x)
                x = x + self.bias
                x = x + self.buffer
                x = x + self.buffer_tied
                return x

        torch.manual_seed(1)
        mod = Foo()
        func, _, _ = make_functional_with_buffers(mod)

        torch.manual_seed(0)
        mod = Foo()
        _, params, buffers = make_functional_with_buffers(mod)
        self.assertEqual(len(params), 3)
        self.assertEqual(len(buffers), 1)

        x = torch.randn(2, 3)
        result = func(params, buffers, x)
        expected = mod(x)
        self.assertEqual(result, expected)

    @parametrize("disable_autograd_tracking", [True, False])
    def test_with_buffers_disable_autograd_tracking(self, disable_autograd_tracking):
        # [FUNCTION DOC] Parametrized over the disable_autograd_tracking flag. Checks
        # make_functional_with_buffers returns two parameters and one buffer, with parameter
        # requires_grad opposite of the flag.
        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and a buffer added to the output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer and a random buffer.
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.buffer = nn.Buffer(torch.randn(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and adds the buffer.
                x = self.linear(x)
                x = x + self.buffer
                return x

        mod = Foo()
        _, params, buffers = make_functional_with_buffers(
            mod, disable_autograd_tracking=disable_autograd_tracking
        )
        self.assertEqual(len(params), 2)
        self.assertEqual(len(buffers), 1)
        for param in params:
            self.assertEqual(param.requires_grad, not disable_autograd_tracking)

    @parametrize("detach_params", [True, False])
    def test_using_detach_functional_call(self, detach_params):
        # [FUNCTION DOC] Parametrized over whether parameters are detached. Checks that
        # functional_call with a dict of (optionally detached) parameters plus buffers gives an
        # output with no grad_fn exactly when parameters were detached.
        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and a buffer added to the output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer and a random buffer.
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.buffer = nn.Buffer(torch.randn(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and adds the buffer.
                x = self.linear(x)
                x = x + self.buffer
                return x

        def params_dict(mod):
            # [FUNCTION DOC] HELPER: Returns a dict of the module named parameters, with each
            # value detached if the enclosing test flag is set.
            named_params = mod.named_parameters()
            return (
                {k: v.detach() for k, v in named_params}
                if detach_params
                else dict(named_params)
            )

        mod = Foo()
        x = torch.randn(3, 3)
        d = (params_dict(mod), dict(mod.named_buffers()))
        out = functional_call(mod, d, x)
        self.assertEqual(out.grad_fn is None, detach_params)

    def test_parameter_tying_grad(self):
        # [FUNCTION DOC] Checks that grad over the functional form of a module with tied
        # parameters (the layer weight and bias also exposed as attributes) gives the same
        # gradients as plain autograd over the module parameters, using identically seeded copies.
        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Module whose linear layer weight and bias are also kept as
            # attributes and used in a functional linear call after the layer itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer and aliases its weight and bias.
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.weight = self.linear.weight
                self.bias = self.linear.bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and then the functional linear
                # with the aliased weight and bias.
                x = self.linear(x)
                x = F.linear(x, self.weight, self.bias)
                return x

        x = torch.randn(2, 3)
        torch.manual_seed(0)
        mod = Foo()
        loss = mod(x).sum()
        expected = torch.autograd.grad(loss, mod.parameters())

        mod = Foo()
        fmod, _, _ = make_functional_with_buffers(mod)
        torch.manual_seed(0)
        mod = Foo()
        _, params, buffers = make_functional_with_buffers(mod)

        def compute_loss(params, buffers, x):
            # [FUNCTION DOC] HELPER: Returns the sum of the functional module output.
            return fmod(params, buffers, x).sum()

        result = grad(compute_loss)(params, buffers, x)

        self.assertEqual(result, expected)

    def test_parameter_tying_ensemble(self):
        # [FUNCTION DOC] Checks combine_state_for_ensemble with modules having tied parameters and
        # buffers: vmap of the combined functional model over stacked parameters, buffers and
        # inputs equals a stack of the individual model outputs (models reseeded for
        # equal weights).
        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose weight and bias are also kept
            # as attributes, plus a buffer registered twice, all used in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layer, aliases for its weight and bias, and a
                # buffer with an alias.
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.weight = self.linear.weight
                self.bias = self.linear.bias
                self.buffer = nn.Buffer(torch.randn(3))
                self.buffer_tied = self.buffer

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer, a functional linear using the aliases,
                # then adds the buffer and its alias.
                x = self.linear(x)
                x = F.linear(x, self.weight, self.bias)
                x = x + self.buffer
                x = x + self.buffer_tied
                return x

        num_models = 2
        xs = torch.randn(num_models, 64, 3)
        models = [Foo() for _ in range(num_models)]
        fmodel, _, _ = combine_state_for_ensemble(models)

        torch.manual_seed(0)
        models = [Foo() for _ in range(num_models)]
        _, params, buffers = combine_state_for_ensemble(models)
        result = vmap(fmodel)(params, buffers, xs)

        torch.manual_seed(0)
        models = [Foo() for _ in range(num_models)]
        expected = torch.stack([model(x) for model, x in zip(models, xs)])

        self.assertEqual(result, expected)

    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_correctness_mnist(self, mechanism):
        # [FUNCTION DOC] Parametrized over make_functional and functional_call. Checks the
        # functional form of a small convolutional classifier (with dropout) gives the same output
        # as the module when the RNG is seeded identically before building the module and before
        # running it.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Convolutional classifier with two conv layers, 2d dropout, two
            # linear layers, max pooling, relu, dropout and log softmax on 32 by 32 inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two convolutions, the 2d dropout and the two
                # linear layers.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs conv, max pool and relu; conv, dropout, max pool and
                # relu; flattens to 320 features; linear and relu; dropout in training mode;
                # linear; and returns log softmax.
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x)

        x = torch.randn(64, 1, 32, 32)
        torch.manual_seed(301)
        fnet, _ = _get_weights_and_functional_call(Net(), mechanism)

        torch.manual_seed(0)
        _, params = _get_weights_and_functional_call(Net(), mechanism)
        result = fnet(params, x)

        torch.manual_seed(0)
        net = Net()
        expected = net(x)

        self.assertEqual(result, expected)

    def test_combine_state_for_ensemble_error(self):
        # [FUNCTION DOC] Checks combine_state_for_ensemble errors with RuntimeError for an empty
        # model list, models differing in training or eval mode, and models of different classes.
        in_features = 2
        out_features = 2

        models = []
        with self.assertRaisesRegex(RuntimeError, "Expected at least one model"):
            _ = combine_state_for_ensemble(models)

        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1].eval()
        with self.assertRaisesRegex(RuntimeError, "same training/eval mode"):
            _ = combine_state_for_ensemble(models)

        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1] = torch.nn.Conv2d(3, 3, (3, 3))
        with self.assertRaisesRegex(RuntimeError, "models to be of the same class"):
            _ = combine_state_for_ensemble(models)

    def test_combine_state_for_ensemble_smoke(self):
        # [FUNCTION DOC] Smoke test: combine_state_for_ensemble on three linear models runs
        # without error.
        in_features = 2
        out_features = 2
        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        _ = combine_state_for_ensemble(models)

    def test_stack_module_state_smoke(self):
        # [FUNCTION DOC] Smoke test: stack_module_state on three linear models runs without error.
        in_features = 2
        out_features = 2
        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        _ = stack_module_state(models)

    def test_stack_module_state_leaf(self):
        # [FUNCTION DOC] Checks that the stacked parameters returned by stack_module_state are
        # leaf tensors that require grad.
        in_features = 2
        out_features = 2
        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        params, buffers = stack_module_state(models)
        for param in params.values():
            self.assertTrue(param.requires_grad)
            self.assertTrue(param.is_leaf)

    def test_stack_module_state_mismatch_error(self):
        # [FUNCTION DOC] Checks stack_module_state raises RuntimeError mentioning requires_grad
        # when one model has a parameter that does not require grad but the others do.
        in_features = 2
        out_features = 2
        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[0].weight.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "same .requires_grad"):
            params, buffers = stack_module_state(models)

    def test_stack_module_state_error(self):
        # [FUNCTION DOC] Checks stack_module_state errors with RuntimeError for an empty model
        # list, models with different training or eval modes, and models of different classes,
        # each with a message prefixed by the function name.
        in_features = 2
        out_features = 2

        models = []
        with self.assertRaisesRegex(
            RuntimeError, "stack_module_state:.* Expected at least one model"
        ):
            _ = stack_module_state(models)

        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1].eval()
        with self.assertRaisesRegex(
            RuntimeError, "stack_module_state:.* same training/eval mode."
        ):
            _ = stack_module_state(models)

        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1] = torch.nn.Conv2d(3, 3, (3, 3))
        with self.assertRaisesRegex(
            RuntimeError, "stack_module_state:.* models to be of the same class"
        ):
            _ = stack_module_state(models)

    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_make_functional_state_correctly_returned_after_forward(self, mechanism):
        # [FUNCTION DOC] Parametrized over make_functional and functional_call. Checks that after
        # a functional forward pass the original module weight and bias attributes of the
        # stateless module are restored (not None and equal to before).
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer from 3 to 3 features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                x = self.linear(x)
                return x

        def get_module_info(mod):
            # [FUNCTION DOC] HELPER: Returns either the make_functional result for the module, or
            # for functional_call the module plus a dict of its named parameters; raises
            # AssertionError for any other mechanism.
            if mechanism == "make_functional":
                return make_functional(mod)
            else:
                if mechanism != "functional_call":
                    raise AssertionError(
                        f"Expected mechanism 'functional_call', got '{mechanism}'"
                    )
                return mod, dict(mod.named_parameters())

        mod = Net()
        func_mod, params = get_module_info(mod)

        # state in func.names_map
        mod = func_mod.stateless_model if mechanism == "make_functional" else func_mod
        old_state_linear_weight = mod.linear.weight
        old_state_linear_bias = mod.linear.bias

        self.assertIsNotNone(old_state_linear_weight)
        self.assertIsNotNone(old_state_linear_bias)

        x = torch.randn(4, 3)
        if mechanism == "make_functional":
            func_mod(params, x)
        else:
            if mechanism != "functional_call":
                raise AssertionError(
                    f"Expected mechanism 'functional_call', got '{mechanism}'"
                )
            functional_call(func_mod, params, x)

        mod = func_mod.stateless_model if mechanism == "make_functional" else func_mod
        new_state_linear_weight = mod.linear.weight
        new_state_linear_bias = mod.linear.bias

        self.assertIsNotNone(new_state_linear_weight)
        self.assertIsNotNone(new_state_linear_bias)

        self.assertEqual(old_state_linear_weight, new_state_linear_weight)
        self.assertEqual(old_state_linear_bias, new_state_linear_bias)


@markDynamoStrictTest
class TestExamplesCorrectness(TestCase):
    # [CLASS DOC] Device-generic tests that run example workloads (MAML regression and Omniglot,
    # batch norm replacement, Lennard-Jones forces, and later ones) through function transforms
    # and compare with plain autograd.
    def _update_params(self, params, grads, alpha, mechanism):
        # [FUNCTION DOC] HELPER: Performs one gradient descent step with step size alpha. With
        # make_functional it returns a list of updated parameters computed by position; otherwise
        # it requires functional_call (raises AssertionError if not) and returns a dict with
        # updated values per key.
        if mechanism == "make_functional":
            return [(params[i] - alpha * grads[i]) for i in range(len(params))]
        else:
            if mechanism != "functional_call":
                raise AssertionError(
                    f"Expected mechanism 'functional_call', got '{mechanism}'"
                )
            return {k: params[k] - alpha * grads[k] for k in params}

    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_maml_regression(self, device, mechanism):
        # [FUNCTION DOC] Parametrized over the weight mechanism. MAML on sine regression: sample
        # tasks with random amplitude and phase, compute the post-adaptation loss for each task
        # with vmap and grad, average, take the gradient with respect to the initial parameters
        # and compare with a loop using plain autograd with create_graph.
        class ThreeLayerNet(nn.Module):
            # [CLASS DOC] HELPER: Three-layer MLP regressor from 1 to 40 to 40 to 1 with
            # relu activations.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three linear layers and two relu modules.
                super().__init__()
                self.fc1 = nn.Linear(1, 40)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(40, 40)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(40, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, relu, linear, relu, linear and returns
                # the result.
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.fc3(x)
                return x

        # TODO: should replace with F.mse_loss
        def mse_loss(x, y):
            # [FUNCTION DOC] HELPER: Returns the mean squared error between x and y (a TODO notes
            # replacing it with the functional loss).
            return torch.mean((x - y) ** 2)

        net, params = _get_weights_and_functional_call(
            ThreeLayerNet().to(device), mechanism
        )
        K = 20
        num_tasks = 4
        alpha = 0.1

        def sample_tasks(outer_batch_size, inner_batch_size):
            # [FUNCTION DOC] HELPER: Samples random amplitudes and phases for the requested number
            # of tasks and returns two batches of inputs and sine targets, as tensors on the
            # test device.
            # Select amplitude and phase for the task
            As = []
            phases = []
            for _ in range(outer_batch_size):
                As.append(np.random.uniform(low=0.1, high=0.5))
                phases.append(np.random.uniform(low=0.0, high=np.pi))

            def get_batch():
                # [FUNCTION DOC] HELPER: Draws uniform inputs for each task, computes amplitude
                # times sine of input plus phase, and returns both as float tensors on the device.
                xs, ys = [], []
                for A, phase in zip(As, phases):
                    x = np.random.uniform(
                        low=-5.0, high=5.0, size=(inner_batch_size, 1)
                    )
                    y = A * np.sin(x + phase)
                    xs.append(x)
                    ys.append(y)
                return torch.tensor(xs, dtype=torch.float, device=device), torch.tensor(
                    ys, dtype=torch.float, device=device
                )

            x1, y1 = get_batch()
            x2, y2 = get_batch()
            return x1, y1, x2, y2

        def get_loss_for_task(use_transform, x1, y1, x2, y2):
            # [FUNCTION DOC] HELPER: Computes the loss after one inner gradient step for a task.
            # The inner gradient comes from grad when use_transform is set, otherwise from plain
            # autograd with create_graph; then updates parameters and returns the mean squared
            # error on the second batch.
            def inner_loss(params, x1, y1):
                # [FUNCTION DOC] HELPER: Returns the mean squared error of the functional network
                # on the inputs against the targets.
                f = net(params, x1)
                loss = mse_loss(f, y1)
                return loss

            if use_transform:
                grads = grad(inner_loss)(params, x1, y1)
            else:
                loss = inner_loss(params, x1, y1)
                grad_params, spec = tree_flatten(params)
                grads = torch.autograd.grad(loss, grad_params, create_graph=True)
                grads = tree_unflatten(grads, spec)

            new_params = self._update_params(params, grads, alpha, mechanism)

            v_f = net(new_params, x2)
            return mse_loss(v_f, y2)

        task = sample_tasks(num_tasks, K)
        list_params = (
            params if mechanism == "make_functional" else list(params.values())
        )

        # Compute with vmap+grad
        inner_losses = vmap(partial(get_loss_for_task, True))(
            task[0], task[1], task[2], task[3]
        )
        loss2 = sum(inner_losses) / len(inner_losses)
        result_grads = torch.autograd.grad(loss2, list_params)

        # Compute without vmap+grad
        inner_losses = [
            get_loss_for_task(False, task[0][i], task[1][i], task[2][i], task[3][i])
            for i in range(num_tasks)
        ]
        loss2 = sum(inner_losses) / len(inner_losses)
        expected_grads = torch.autograd.grad(loss2, list_params)

        self.assertEqual(result_grads, expected_grads)

    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_maml_omniglot(self, device, mechanism):
        # [FUNCTION DOC] Parametrized over the weight mechanism. MAML on an Omniglot-like conv
        # network in double precision (group norm instead of batch norm, no in-place relu): the
        # query loss after two inner steps is computed with vmap and grad, then compared in outer
        # gradient to a loop with plain autograd.
        # TODO: there appears to be precision issues for float32
        dtype = torch.double

        # TODO: We don't support inplace relu?
        inplace_relu = False
        n_way = 5
        n_inner_iter = 2
        num_tasks = 2

        # real example uses batch norm but it's numerically unstable in the first
        # iteration, when near 0, and won't produce same gradients. Uses group norm instead
        net = (
            nn.Sequential(
                nn.Conv2d(1, 64, 3),
                nn.GroupNorm(64, 64, affine=True),
                nn.ReLU(inplace=inplace_relu),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3),
                nn.GroupNorm(64, 64, affine=True),
                nn.ReLU(inplace=inplace_relu),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3),
                nn.GroupNorm(64, 64, affine=True),
                nn.ReLU(inplace=inplace_relu),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(64, n_way),
            )
            .to(device)
            .to(dtype)
        )

        fnet, params, buffers = _get_weights_and_functional_call_with_buffers(
            net, mechanism
        )
        net = (params, buffers, fnet)

        def loss_for_task(net, n_inner_iter, use_transform, x_spt, y_spt, x_qry, y_qry):
            # [FUNCTION DOC] HELPER: Runs the inner loop of n_inner_iter gradient steps on the
            # support set (using grad or plain autograd depending on use_transform), then returns
            # the query cross entropy loss and accuracy.
            params, buffers, fnet = net
            querysz = x_qry.size(0)

            def compute_loss(new_params, buffers, x, y):
                # [FUNCTION DOC] HELPER: Returns the cross entropy of the functional network
                # output against the labels.
                logits = fnet(new_params, buffers, x)
                loss = F.cross_entropy(logits, y)
                return loss

            new_params = params
            for _ in range(n_inner_iter):
                if use_transform:
                    grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
                else:
                    res = compute_loss(new_params, buffers, x_spt, y_spt)
                    grad_params, spec = tree_flatten(new_params)
                    grads = torch.autograd.grad(res, grad_params, create_graph=True)
                    grads = tree_unflatten(grads, spec)

                new_params = self._update_params(new_params, grads, 1e-1, mechanism)

            qry_logits = fnet(new_params, buffers, x_qry)
            qry_loss = F.cross_entropy(qry_logits, y_qry)
            qry_acc = (qry_logits.argmax(dim=1) == y_qry).sum() / querysz

            return qry_loss, qry_acc

        # Get some sample inputs...
        x_spt = torch.randn(num_tasks, 25, 1, 28, 28, dtype=dtype, device=device)
        y_spt = torch.randint(0, 5, (num_tasks, 25), device=device)
        x_qry = torch.randn(num_tasks, 75, 1, 28, 28, dtype=dtype, device=device)
        y_qry = torch.randint(0, 5, (num_tasks, 75), device=device)

        # compute with vmap + grad
        compute_loss = partial(loss_for_task, net, n_inner_iter, True)
        qry_losses, _ = vmap(compute_loss)(x_spt, y_spt, x_qry, y_qry)
        list_params = (
            params if mechanism == "make_functional" else list(params.values())
        )
        result_grads = torch.autograd.grad(qry_losses.sum(), list_params)

        # compute without vmap + grad
        compute_loss = partial(loss_for_task, net, n_inner_iter, False)
        losses = [
            compute_loss(x_spt[i], y_spt[i], x_qry[i], y_qry[i])[0]
            for i in range(num_tasks)
        ]
        expected_grads = torch.autograd.grad(sum(losses), list_params)

        self.assertEqual(result_grads, expected_grads)

    @parametrize("mechanism", ["make_functional", "functional_call"])
    @parametrize("originally_track_running_stats", [True, False])
    def test_update_batch_norm(self, device, originally_track_running_stats, mechanism):
        # [FUNCTION DOC] Parametrized over the mechanism and whether batch norm originally tracked
        # running stats. Replaces all batch norm modules via the experimental helper, then
        # compares per-sample gradients computed with vmap and grad (parameters unbatched) with a
        # loop of plain autograd gradients, stacked.
        dtype = torch.double
        inplace_relu = False
        classes = 5
        num_batches = 2
        net = (
            nn.Sequential(
                nn.Conv2d(64, 64, 3),
                nn.BatchNorm2d(
                    64, affine=True, track_running_stats=originally_track_running_stats
                ),
                nn.ReLU(inplace=inplace_relu),
                nn.Flatten(),
                nn.Linear(43264, classes),
            )
            .to(device)
            .to(dtype)
        )

        replace_all_batch_norm_modules_(net)
        transformed_net = net
        fnet, params, buffers = _get_weights_and_functional_call_with_buffers(
            transformed_net, mechanism
        )
        criterion = nn.CrossEntropyLoss()

        def compute_loss(x, y, params, buffers):
            # [FUNCTION DOC] HELPER: Returns the cross entropy of the functional network on the
            # input against the labels.
            return criterion(fnet(params, buffers, x), y)

        # Get some sample inputs...
        x = torch.randn(num_batches, 1, 64, 28, 28, device=device, dtype=dtype)
        y = torch.randint(0, classes, (num_batches, 1), device=device)

        # compute some per sample grads with vmap + grad
        result_grads = vmap(grad(compute_loss, argnums=2), in_dims=(0, 0, None, None))(
            x, y, params, buffers
        )

        # compute some per sample grads without vmap + grad
        fnet, params, buffers = _get_weights_and_functional_call_with_buffers(
            transformed_net, mechanism
        )
        flat_params, spec = tree_flatten(params)
        expected_grads = [
            torch.autograd.grad(compute_loss(x[i], y[i], params, buffers), flat_params)
            for i in range(num_batches)
        ]
        expected_grads = [torch.stack(shards) for shards in zip(*expected_grads)]
        expected_grads = tree_unflatten(expected_grads, spec)

        self.assertEqual(result_grads, expected_grads)

    @parametrize("jac", ["jacfwd", "jacrev"])
    def test_lennard_jones_batched_jac(self, device, jac):
        # [FUNCTION DOC] Parametrized over jacfwd and jacrev. Builds Lennard-Jones training
        # energies and forces for 100 distances along one axis, a small tanh network, and compares
        # the parameter gradients of a combined energy and force loss when the forces come from a
        # vmapped batched Jacobian of the model versus per-sample autograd functional Jacobians
        # with create_graph.
        sigma = 0.5
        epsilon = 4.0

        jac = getattr(functorch, jac)

        def lennard_jones(r):
            # [FUNCTION DOC] HELPER: Returns the Lennard-Jones potential energy at distance r
            # using the enclosing sigma and epsilon.
            return epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

        def lennard_jones_force(r):
            # [FUNCTION DOC] HELPER: Returns the magnitude of the Lennard-Jones force at distance
            # r using the enclosing sigma and epsilon.
            """Get magnitude of LJ force"""
            return -epsilon * ((-12 * sigma**12 / r**13) + (6 * sigma**6 / r**7))

        r = torch.linspace(0.5, 2 * sigma, steps=100, requires_grad=True, device=device)
        drs = torch.outer(r, torch.tensor([1.0, 0, 0], device=device))
        norms = torch.norm(drs, dim=1).reshape(-1, 1)
        training_energies = torch.stack(list(map(lennard_jones, norms))).reshape(-1, 1)
        training_forces = torch.stack(
            [force * dr for force, dr in zip(map(lennard_jones_force, norms), drs)]
        )

        model = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        ).to(device)

        def make_prediction(model, drs, use_functorch):
            # [FUNCTION DOC] HELPER: Predicts energies and forces from distance vectors: the
            # energies come from the model on the norms; forces use a vmapped Jacobian of the
            # model when use_functorch is set, otherwise per-sample autograd functional Jacobians
            # with create_graph.
            norms = torch.norm(drs, dim=1).reshape(-1, 1)
            energies = model(norms)

            if use_functorch:
                network_derivs = vmap(jac(model))(norms).squeeze(-1)
                forces = -network_derivs * drs / norms
            else:
                forces = []
                for r, dr in zip(norms, drs):
                    network_deriv = torch.autograd.functional.jacobian(
                        model, r, create_graph=True
                    )
                    force = -network_deriv * dr / r
                    forces.append(force)
                forces = torch.cat(forces)
            return energies, forces

        def loss_fn(energies, forces, predicted_energies, predicted_forces):
            # [FUNCTION DOC] HELPER: Returns the mean squared error of the energies plus 0.01
            # times the mean squared error of the forces divided by three.
            return (
                F.mse_loss(energies, predicted_energies)
                + 0.01 * F.mse_loss(forces, predicted_forces) / 3
            )

        energies, forces = make_prediction(model, drs, use_functorch=True)
        loss = loss_fn(training_energies, training_forces, energies, forces)
        result = torch.autograd.grad(loss, model.parameters())

        energies, forces = make_prediction(model, drs, use_functorch=False)
        loss = loss_fn(training_energies, training_forces, energies, forces)
        expected = torch.autograd.grad(loss, model.parameters())

        self.assertEqual(result, expected)

    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_ensemble_regression(self, device, mechanism):
        # [FUNCTION DOC] Parametrized over the weight mechanism. Ensemble training on a noisy
        # two-spiral classification set: one vmapped training step with grad_and_value over two
        # stacked models (shared data) is compared in loss and updated weights with two separate
        # plain-autograd steps stacked together.
        def make_spirals(n_samples, noise_std=0.0, rotations=1.0):
            # [FUNCTION DOC] HELPER: Generates a two-class spiral dataset of the requested size
            # with random signs and optional Gaussian noise; returns the 2d points and integer
            # labels on the test device.
            ts = torch.linspace(0, 1, n_samples)
            rs = ts**0.5
            thetas = rs * rotations * 2 * math.pi
            signs = torch.randint(0, 2, (n_samples,)) * 2 - 1
            labels = (signs > 0).to(torch.long)

            xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
            ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
            points = torch.stack([xs, ys], dim=1)
            return points.to(device), labels.to(device)

        points, labels = make_spirals(100, noise_std=0.05)

        class MLPClassifier(nn.Module):
            # [CLASS DOC] HELPER: Two-layer perceptron classifier (linear, relu, linear, log
            # softmax) from 2 features to n_classes with hidden width hidden_dim.
            def __init__(self, hidden_dim=32, n_classes=2):
                # [FUNCTION DOC] HELPER: Stores sizes and creates two linear layers (2 to
                # hidden_dim and hidden_dim to n_classes).
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, relu, linear and log softmax over the
                # last dim; returns the result.
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        loss_fn = nn.NLLLoss()

        func_model, weights = _get_weights_and_functional_call(
            MLPClassifier().to(device), mechanism
        )

        def train_step_fn(use_transform, weights, batch, targets, lr=0.2):
            # [FUNCTION DOC] HELPER: Runs one gradient descent step on the loss with step size lr.
            # The gradient and loss come from grad_and_value when use_transform is set, otherwise
            # from plain autograd; returns the loss and the updated weights.
            def compute_loss(weights, batch, targets):
                # [FUNCTION DOC] HELPER: Returns the negative log likelihood loss of the
                # functional model output against the targets.
                output = func_model(weights, batch)
                loss = loss_fn(output, targets)
                return loss

            if use_transform:
                grad_weights, loss = grad_and_value(compute_loss)(
                    weights, batch, targets
                )
            else:
                loss = compute_loss(weights, batch, targets)
                flat_weights, spec = tree_flatten(weights)
                flat_grad_weights = torch.autograd.grad(loss, flat_weights)
                grad_weights = tree_unflatten(flat_grad_weights, spec)

            new_weights = self._update_params(weights, grad_weights, lr, mechanism)
            return (loss, new_weights)

        def unpack(train_result):
            # [FUNCTION DOC] HELPER: Returns the first two entries of a training step result as
            # a pair.
            return train_result[0], train_result[1]

        def init_fn(num_models):
            # [FUNCTION DOC] HELPER: Creates the requested number of fresh classifier models and
            # returns their stacked parameters, via the ensemble combine helper or the module
            # state stack helper depending on the mechanism.
            models = tuple(MLPClassifier().to(device) for _ in range(num_models))
            if mechanism == "make_functional":
                return combine_state_for_ensemble(models)[1]
            else:
                return stack_module_state(models)[0]

        def slice_weights(batched_weights, index):
            # [FUNCTION DOC] HELPER: Selects one model from batched weights by index and returns
            # the weights as detached tensors that require grad, keeping the container structure.
            return tree_map(
                lambda weight: weight[index].detach().requires_grad_(), batched_weights
            )

        batched_weights = init_fn(num_models=2)
        parallel_train_step_fn = vmap(
            partial(train_step_fn, True), in_dims=(0, None, None)
        )

        result_loss, result_weights = unpack(
            parallel_train_step_fn(batched_weights, points, labels)
        )

        loss0, weights0 = unpack(
            train_step_fn(False, slice_weights(batched_weights, 0), points, labels)
        )
        loss1, weights1 = unpack(
            train_step_fn(False, slice_weights(batched_weights, 1), points, labels)
        )
        expected_loss = torch.stack([loss0, loss1])

        weights0, spec0 = tree_flatten(weights0)
        weights1, spec1 = tree_flatten(weights1)
        if spec0 != spec1:
            raise AssertionError(f"Expected spec0 == spec1, got {spec0} != {spec1}")
        expected_weights = tuple(
            torch.stack([w0, w1]) for w0, w1 in zip(weights0, weights1)
        )
        expected_weights = tree_unflatten(expected_weights, spec0)

        self.assertEqual(result_loss, expected_loss)
        self.assertEqual(result_weights, expected_weights)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/180336")
    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/180320")
    @parametrize(
        "dropout_layer",
        [
            subtest(nn.Dropout, "Dropout"),
            subtest(nn.AlphaDropout, "AlphaDropout"),
            subtest(nn.FeatureAlphaDropout, "FeatureAlphaDropout"),
        ],
    )
    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_find_learning_rate_ensembling(self, device, dropout_layer, mechanism):
        # [FUNCTION DOC] Parametrized over three dropout layer classes and the weight mechanism;
        # skipped under Dynamo (issues 180336 and 180320). Trains two identically initialized
        # models with different learning rates under vmap with same randomness, and checks the
        # losses are equal for both while the updated weights differ.
        # This example mimics what a user might do when trying to find the optimal learning rate. They would
        # want to run a bunch of models with the same behavior (including the same dropout!) and have them
        # each run with different learning rates. Specifically, this is an example of using same randomness with vmap
        points, labels = (
            torch.randn(100, 2, 2, 2, 2, device=device),
            torch.randint(0, 2, (100,), device=device),
        )

        class MLPClassifier(nn.Module):
            # [CLASS DOC] HELPER: Classifier with a dropout layer of the test-selected class,
            # flatten, linear (16 to hidden), relu, linear and log softmax.
            def __init__(self, hidden_dim=32, n_classes=2):
                # [FUNCTION DOC] HELPER: Stores sizes, creates the chosen dropout layer and two
                # linear layers.
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.dropout = dropout_layer()
                self.fc1 = nn.Linear(16, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies dropout, flattens from dim one, then linear,
                # relu, linear and log softmax; returns the result.
                x = self.dropout(x)
                x = torch.flatten(x, start_dim=1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        loss_fn = nn.NLLLoss()

        func_model, weights = _get_weights_and_functional_call(
            MLPClassifier().to(device), mechanism
        )

        def train_step_fn(weights, batch, targets, lr):
            # [FUNCTION DOC] HELPER: Computes loss and gradient via grad_and_value, performs a
            # gradient descent step with learning rate lr, and returns a flat tuple of the loss
            # followed by the new weights (flattened to a list for functional_call, since vmap
            # needs tensor outputs).
            def compute_loss(weights, batch, targets):
                # [FUNCTION DOC] HELPER: Returns the negative log likelihood of the functional
                # model output against the targets.
                output = func_model(weights, batch)
                loss = loss_fn(output, targets)
                return loss

            grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)
            new_weights = self._update_params(weights, grad_weights, lr, mechanism)
            if mechanism != "make_functional":
                new_weights = list(new_weights.values())
            # NB: return looks weird because torch.vmap must return Tensors
            return (loss, *new_weights)

        def unpack(train_result):
            # [FUNCTION DOC] HELPER: Splits a training step result into the loss and the tuple of
            # remaining weights.
            return train_result[0], train_result[1:]

        def init_fn(num_models):
            # [FUNCTION DOC] HELPER: Creates one classifier, makes deep copies of it for each
            # requested model (same initialization), and returns stacked parameters via the
            # ensemble combine helper or the module state stack helper by mechanism.
            og_model = MLPClassifier().to(device)
            models = tuple(
                copy.deepcopy(og_model) for _ in range(num_models)
            )  # have same initialization
            if mechanism == "make_functional":
                return combine_state_for_ensemble(models)[1]
            else:
                return stack_module_state(models)[0]

        batched_weights = init_fn(num_models=2)
        parallel_train_step_fn = vmap(
            train_step_fn, in_dims=(0, None, None, 0), randomness="same"
        )

        lrs = torch.tensor([0.2, 0.4], device=device)
        result_loss, result_weights = unpack(
            parallel_train_step_fn(batched_weights, points, labels, lrs)
        )

        self.assertEqual(result_loss[0], result_loss[1])
        self.assertNotEqual(
            tuple(weight[0] for weight in result_weights),
            tuple(weight[1] for weight in result_weights),
        )

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_resnet18_per_sample_grads(self, device, mechanism):
        # [FUNCTION DOC] Parametrized over the weight mechanism; skipped if torchvision is
        # missing, with TF32 turned off. Builds a resnet18 with group norm, computes per-sample
        # gradients with vmap over grad (weights unbatched, images and targets batched), and
        # compares with a loop of plain autograd gradients (absolute tolerance 1e-3, relative
        # tolerance 1.0).
        from torchvision import models

        model = models.__dict__["resnet18"](
            pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(32, c), c))
        ).to(device)
        criterion = nn.CrossEntropyLoss(
            reduction="sum"
        )  # avoid cross batch reductions for for loop comparison

        func_model, weights = _get_weights_and_functional_call(model, mechanism)

        def compute_loss(weights, image, target):
            # [FUNCTION DOC] HELPER: Adds a batch dim of one to the image and target, runs the
            # functional model and returns the summed cross entropy.
            image = image.unsqueeze(0)
            target = target.unsqueeze(0)
            output = func_model(weights, image)
            loss = criterion(output, target)
            return loss

        batch_size = 3
        images = torch.randn(batch_size, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        result_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(
            weights, images, targets
        )

        flat_weights, spec = tree_flatten(weights)
        expected_grads = [
            torch.autograd.grad(
                compute_loss(weights, images[i], targets[i]), flat_weights
            )
            for i in range(batch_size)
        ]
        expected_grads = [torch.stack(shards) for shards in zip(*expected_grads)]
        expected_grads = tree_unflatten(expected_grads, spec)

        self.assertEqual(result_grads, expected_grads, atol=1e-3, rtol=1.0)


def normalize_devices(fx_g):
    # [FUNCTION DOC] HELPER: Rewrites an FX graph module in place so that torch.device arguments
    # (positional or keyword) of every node become the string cpu (to make printed code device
    # independent), recompiles it and returns the same module.
    for node in fx_g.graph.nodes:
        args = list(node.args)
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.device):
                args[idx] = "cpu"
        node.args = tuple(args)
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = "cpu"
            new_kwargs[k] = v
        node.kwargs = new_kwargs
    fx_g.recompile()
    return fx_g


@markDynamoStrictTest
class TestFunctionalize(TestCase):
    # [CLASS DOC] Device-generic tests for functionalize: numerical equivalence with eager (with
    # and without vmap and view removal), composition with grad, vmap and jvp, fake tensors, and
    # golden FX code tests for make_fx of functionalized functions.
    def _check_functionalize_correctness(self, f, inpt, *, skip_vmap=False):
        # [FUNCTION DOC] HELPER: Runs the function eagerly on a clone, with functionalize (under
        # vmap on a batch of one, unless skip_vmap) on a second clone and with functionalize
        # removing mutations and views on a third. Asserts outputs match eager and that all three
        # inputs ended up with the same mutated values.
        inpt1 = inpt.clone()
        inpt2 = inpt.clone()
        inpt3 = inpt.clone()

        expected_outputs = f(inpt1)
        if skip_vmap:
            actual_outputs = functionalize(f)(inpt2)
        else:
            actual_outputs = vmap(functionalize(f))(inpt2.unsqueeze(0))[0].squeeze()
        # Right now the flavor of functionalize that also removes view ops
        # isn't being used with vmap
        # That's because {view}_copy ops don't have batching rules yet
        # (although we should probably fix that)
        actual_outputs_view_copy = functionalize(f, remove="mutations_and_views")(inpt3)
        # Check that outputs are the same
        self.assertEqual(actual_outputs, expected_outputs)
        self.assertEqual(actual_outputs_view_copy, expected_outputs)

        # Inputs might have been mutated by f: check that they were mutated properly
        self.assertEqual(inpt1, inpt2)
        self.assertEqual(inpt1, inpt3)

    def test_simple_view(self, device):
        # [FUNCTION DOC] Checks functionalize correctness for adding to a reshaped view of the
        # input in place and returning the input.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Views x as four by two, adds a ones vector in place through
            # the view and returns x.
            tmp = torch.ones(2, device=device)
            y = x.view(4, 2)
            y.add_(tmp)
            return x

        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))

    def test_multioutput_view(self, device):
        # [FUNCTION DOC] Checks functionalize correctness for an in-place add on a diagonal view
        # of a split piece of the input.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Splits x into two parts, adds a ones vector in place to the
            # diagonal of the first part and returns x.
            tmp = torch.ones(2, device=device)
            y1, y2 = x.split(2)
            y1_view = y1.diagonal()
            y1_view.add_(tmp)
            return x

        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))

    def test_inplace_view(self, device):
        # [FUNCTION DOC] Checks functionalize correctness (without vmap) for an in-place add on a
        # row selected from a transposed intermediate.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Doubles x, transposes it, takes its first row, adds ones in
            # place and returns the intermediate.
            tmp = torch.ones(4, device=device)
            y = x + x
            y2 = y.transpose(1, 0)
            z = y2[0]
            z.add_(tmp)
            return y

        self._check_functionalize_correctness(
            f, torch.zeros(4, 2, device=device), skip_vmap=True
        )

    # See https://github.com/pytorch/functorch/issues/780
    def test_linear(self, device):
        # [FUNCTION DOC] Regression test for functorch issue 780: functionalize of a function
        # calling the internal linear op gives the same result as eager.
        def f(x, y, z) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the internal linear op applied to x, y and z.
            return torch._C._nn.linear(x, y, z)

        x = torch.randn(14, 1, 384, device=device)
        y = torch.randn(96, 384, device=device)
        z = torch.randn(96, device=device)

        out_expected = f(x, y, z)
        out_actual = functionalize(f)(x, y, z)
        self.assertEqual(out_expected, out_actual)

    def test_multioutput_inplace_slice_view(self, device):
        # [FUNCTION DOC] Checks functionalize correctness (without vmap, see the note on slice
        # scatter under vmap) for a chain of view ops including reshape, transpose, in-place
        # unsqueeze and squeeze, and split, followed by an in-place add.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies a chain of view ops to x and an in-place add to one
            # split piece, returns x.
            tmp = torch.ones(2, 2, device=device)
            y = x.view(8)
            z0 = y.reshape(2, 4)
            z1 = z0.transpose(1, 0)
            z1.unsqueeze_(0)
            z1.squeeze_()
            z2, z3 = z1.split(2)
            z2.add_(tmp)
            return x

        # See Note [Fix vmap slice_scatter]
        self._check_functionalize_correctness(
            f, torch.zeros(4, 2, device=device), skip_vmap=True
        )

    # Ensure functionalize works with List[Optional[Tensor]] arguments.
    # See the fix / discussion at https://github.com/pytorch/pytorch/pull/76085
    def test_functionalize_opt_tensor_list(self, device):
        # [FUNCTION DOC] Checks functionalize works with functions taking an optional tensor list
        # argument (index by tensor) both numerically and with make_fx, comparing the traced code
        # to a golden string.
        def f(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns x indexed by the index tensor.
            return x[indices]

        inpta = torch.ones(4, device=device)
        inptb = torch.arange(2, device=device)
        out1 = f(inpta, inptb)
        out2 = functionalize(f)(inpta, inptb)
        self.assertEqual(out1, out2)
        out = make_fx(functionalize(f))(inpta, inptb)
        self.assertExpectedInline(
            (out.code),
            """\



def forward(self, x_1, indices_1) -> torch.Tensor:
    index = torch.ops.aten.index.Tensor(x_1, [indices_1]);  x_1 = indices_1 = None
    return index
    """,
        )

    # Ensure grad(functionalize(f)) works
    def test_functionalize_grad(self, device):
        # [FUNCTION DOC] Checks grad of a functionalized function equals grad of the original and
        # that the input is left equal.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Doubles x, takes a view, adds ones in place to the doubled
            # tensor and returns the sum of the view.
            tmp = torch.ones(2, device=device)
            y = x + x
            z = y.view(4, 2)
            y.add_(tmp)
            return z.sum()

        inpt1 = torch.ones(4, 2, device=device)
        inpt2 = torch.ones(4, 2, device=device)
        out1 = grad(f)(inpt1)
        out2 = grad(functionalize(f))(inpt2)
        self.assertEqual(out1, out2)
        self.assertEqual(inpt1, inpt2)

    @unittest.skipIf(IS_FBCODE, "fails in fbcode")
    def test_vmap_functionalize_jvp(self, device):
        # [FUNCTION DOC] Checks vmap of jvp gives the same result with and without functionalize
        # applied to the jvp wrapper. Skipped in fbcode.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Doubles x, takes a flat view, adds one in place to the
            # doubled tensor and returns the view.
            y = x + x
            z = y.view(-1)
            y.add_(1)
            return z

        def jvp_wrapper(x, t):
            # [FUNCTION DOC] HELPER: Returns the jvp of the view-mutating function at x with
            # tangent t.
            return jvp(
                f,
                (x,),
                (t,),
            )

        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        out1 = vmap(jvp_wrapper)(x, t)
        out2 = vmap(functionalize(jvp_wrapper))(x, t)
        self.assertEqual(out1, out2)

    # TODO: move this test into test_fake_tensor.py
    # once functionalize() can be used in core tests.
    def test_functionalize_fake_tensors(self, device):
        # [FUNCTION DOC] Checks functionalize works with a fake tensor mode on a tensor that
        # requires grad (detach and add), and the original tensor keeps its size. The TODO says to
        # move it to fake tensor tests once functionalize is usable in core tests.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Detaches x and returns the sum of the detached tensor
            # with itself.
            y = x.detach()
            return y + y

        with FakeTensorMode():
            x = torch.ones(2, device=device, requires_grad=True)
            functionalize(f)(x)
        self.assertEqual(x.size(), (2,))

    def test_functionalize_fx_simple(self, device):
        # [FUNCTION DOC] Checks the make_fx graph of functionalize with mutations and views
        # removed for an in-place add through a view: view_copy ops replace views and a final copy
        # into the input preserves the input mutation. Compares printed code (device normalized)
        # with a golden string.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Views x as four by two, adds a ones vector in place through
            # the view and returns x.
            tmp = torch.ones(2, device=device)
            y = x.view(4, 2)
            y.add_(tmp)
            return x

        # There's a copy_ in the graph, because the input (x) was mutated.
        # To preserve semantics, functionalize() needs to propagate the mutation.
        fn = make_fx(functionalize(f, remove="mutations_and_views"))
        out = fn(torch.zeros(4, 2, device=device))
        out = normalize_devices(out)
        self.assertExpectedInline(
            (out.code),
            """\



def forward(self, x_1) -> torch.Tensor:
    ones = torch.ops.aten.ones.default([2], device = 'cpu', pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(x_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2]);  view_copy_2 = None
    copy_ = torch.ops.aten.copy_.default(x_1, view_copy_1);  x_1 = copy_ = None
    return view_copy_1
    """,
        )

    def test_functionalize_fx_transpose_simple(self, device):
        # [FUNCTION DOC] Checks the traced graph for a transpose with views removed is a transpose
        # copy op, using a golden string.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns x transposed.
            return x.transpose(1, 0)

        fn = make_fx(functionalize(f, remove="mutations_and_views"))
        out = fn(torch.zeros(4, 2, device=device))
        out = normalize_devices(out)
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, x_1) -> torch.Tensor:
    transpose_copy = torch.ops.aten.transpose_copy.int(x_1, 1, 0);  x_1 = None
    return transpose_copy
    """,
        )

    def test_functionalize_fx_out_op(self, device):
        # [FUNCTION DOC] Checks the traced functionalized graph for a function using an out
        # variant op followed by an in-place add through a view, with views removed, against a
        # golden string.
        def f(inpt: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocates an empty scalar tensor, writes the doubled input
            # into it with the out argument, adds one in place through a view and returns
            # the tensor.
            out = torch.empty((), dtype=torch.float32)
            torch.add(inpt, inpt, out=out)
            out_view = out.view(4)
            out_view.add_(1)
            return out

        fn = make_fx(functionalize(f, remove="mutations_and_views"))
        out = fn(torch.arange(4, device=device, dtype=torch.float32))
        out = normalize_devices(out)
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, inpt_1) -> torch.Tensor:
    empty = torch.ops.aten.empty.memory_format([], dtype = torch.float32, device = 'cpu', pin_memory = False);  empty = None
    add = torch.ops.aten.add.Tensor(inpt_1, inpt_1);  inpt_1 = None
    view_copy = torch.ops.aten.view_copy.default(add, [4]);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4]);  add = None
    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1);  view_copy_1 = None
    view_copy_2 = torch.ops.aten.view_copy.default(add_1, [4]);  add_1 = None
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [4]);  view_copy_3 = None
    return view_copy_2
    """,
        )

    def test_functionalize_fx_multi_out_op(self, device):
        # [FUNCTION DOC] Checks the traced graph for an op with two out arguments (aminmax writing
        # into a plain tensor and a view of another) after functionalization with views removed,
        # against a golden string.
        def f(inpt: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocates two empty outputs, one used through a flat view,
            # views the input as two by four, runs aminmax along dim zero writing into the
            # outputs, and returns both.
            mins = torch.empty(4, dtype=torch.float32)
            maxs = torch.empty(2, 2, dtype=torch.float32)
            maxs_view = maxs.view(4)
            inpt_view = inpt.view(2, 4)
            torch.aminmax(inpt_view, dim=0, out=(mins, maxs_view))
            return (maxs, mins)

        fn = make_fx(functionalize(f, remove="mutations_and_views"))
        out = fn(torch.arange(8, device=device, dtype=torch.float32))
        out = normalize_devices(out)
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, inpt_1) -> torch.Tensor:
    empty = torch.ops.aten.empty.memory_format([4], dtype = torch.float32, device = 'cpu', pin_memory = False);  empty = None
    empty_1 = torch.ops.aten.empty.memory_format([2, 2], dtype = torch.float32, device = 'cpu', pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(empty_1, [4]);  empty_1 = view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(inpt_1, [2, 4]);  inpt_1 = None
    aminmax = torch.ops.aten.aminmax.default(view_copy_1, dim = 0);  view_copy_1 = None
    getitem = aminmax[0]
    getitem_1 = aminmax[1];  aminmax = None
    view_copy_2 = torch.ops.aten.view_copy.default(getitem_1, [2, 2]);  getitem_1 = None
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [4]);  view_copy_3 = None
    return (view_copy_2, getitem)
    """,
        )

    def test_functionalize_fx_reapply_views_simple(self, device):
        # [FUNCTION DOC] Checks the traced graph of default functionalize (views are reapplied as
        # view ops, not copies) for an in-place add through a view, with a copy into the input,
        # against a golden string.
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Views x as four by two, adds a ones vector in place through
            # the view and returns x.
            tmp = torch.ones(2, device=device)
            y = x.view(4, 2)
            y.add_(tmp)
            return x

        out = make_fx(functionalize(f))(torch.zeros(4, 2, device=device))
        out = normalize_devices(out)
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, x_1) -> torch.Tensor:
    ones = torch.ops.aten.ones.default([2], device = 'cpu', pin_memory = False)
    view = torch.ops.aten.view.default(x_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [4, 2]);  view_2 = None
    copy_ = torch.ops.aten.copy_.default(x_1, view_1);  x_1 = copy_ = None
    return view_1
    """,
        )

    def test_functionalize_nonfunctional_output(self, device):
        # [FUNCTION DOC] Checks the traced graph of functionalize for a function returning a
        # tensor captured from outside is just a read of the captured constant, against a
        # golden string.
        global_out = torch.ones(2, device=device)

        def f() -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the enclosing captured tensor.
            return global_out

        out = make_fx(functionalize(f))()
        out = normalize_devices(out)
        self.assertExpectedInline(
            out.code,
            """\



def forward(self) -> torch.Tensor:
    _tensor_constant0 = self._tensor_constant0
    return _tensor_constant0
    """,
        )

    def test_functionalize_optional_tensorlist1(self, device):
        # [FUNCTION DOC] Checks functionalize handles ops with optional tensor list arguments
        # (indexing with a tensor) and that the traced graph has a plain index op, against a
        # golden string.
        def f(a, b) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns a indexed by b.
            # at::index has OptionalTensorList arguments,
            # test that here
            return a[b]

        a = torch.arange(4).reshape(2, 2)
        b = torch.ones(2, dtype=torch.long)
        out = make_fx(functionalize(f))(a, b)
        out = normalize_devices(out)
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, a_1, b_1) -> torch.Tensor:
    index = torch.ops.aten.index.Tensor(a_1, [b_1]);  a_1 = b_1 = None
    return index
    """,
        )

    @unittest.skipIf(IS_FBCODE, "fails in fbcode")
    def test_functionalize_optional_tensorlist2(self, device):
        # [FUNCTION DOC] Checks functionalize on a direct call of the aten index op with an index
        # tensor (see PR 77846): the traced graph unbinds the tensor and indexes with a list,
        # against a golden string. Skipped in fbcode.
        def f(a, b) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the result of the aten index op on a and b.
            # See https://github.com/pytorch/pytorch/pull/77846
            return torch.ops.aten.index(a, b)

        a = torch.arange(4).reshape(2, 2)
        b = torch.ones(2, dtype=torch.long)
        out = make_fx(functionalize(f))(a, b)
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, a_1, b_1) -> torch.Tensor:
    unbind = torch.ops.aten.unbind.int(b_1);  b_1 = None
    getitem = unbind[0]
    getitem_1 = unbind[1];  unbind = None
    index = torch.ops.aten.index.Tensor(a_1, [getitem, getitem_1]);  a_1 = getitem = getitem_1 = None
    return index
    """,
        )

    def test_resize_program_inputs(self, device):
        # [FUNCTION DOC] Checks functionalize of resizing a program input and filling it: the
        # traced graph shows resize, fill, an in-place resize and a copy back, returning None,
        # against a golden string.
        def f(x):
            # [FUNCTION DOC] HELPER: Resizes x in place to ten elements and fills it with two;
            # returns nothing.
            x.resize_(10)
            x.fill_(2)

        fn = make_fx(functionalize(f))
        out = fn(torch.zeros(0, device=device))
        out = normalize_devices(out)
        self.assertExpectedInline(
            (out.code),
            """\



def forward(self, x_1):
    resize = torch.ops.aten.resize.default(x_1, [10])
    fill = torch.ops.aten.fill.Scalar(resize, 2);  resize = None
    resize_ = torch.ops.aten.resize_.default(x_1, [10]);  x_1 = None
    copy_ = torch.ops.aten.copy_.default(resize_, fill);  resize_ = fill = copy_ = None
    return None
    """,
        )


def construct_sum_pyop():
    # [FUNCTION DOC] HELPER: Builds and returns a custom higher order operator named mysum that
    # sums a tensor over a dim, with implementations for the vmap transform, the grad transform
    # (through a single-level autograd Function), and the CPU and CUDA autograd dispatch keys.
    # Used to test transforms with higher order operators.
    class MySum(HigherOrderOperator):
        # [CLASS DOC] HELPER: Higher order operator class registered under the name mysum; its
        # call just forwards to the parent implementation.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Initializes the base operator with the name mysum.
            super().__init__("mysum")

        def __call__(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Forwards all arguments to the base class call and returns
            # its result.
            return super().__call__(*args, **kwargs)

    mysum = MySum()

    @mysum.py_impl(torch._C._functorch.TransformType.Vmap)
    def mysum_batch_rule(interpreter, x, dim):
        # [FUNCTION DOC] HELPER: Vmap rule for mysum. For an unbatched tensor it lowers the
        # interpreter and calls the operator again (after a no-op view). For a batched tensor it
        # unwraps the value and batch dim, moves the batch dim to the front, sums over the dim
        # shifted by one with the interpreter lowered, and rewraps the result with a batch dim
        # at zero.
        if not torch._C._functorch.is_batchedtensor(x):
            with interpreter.lower():
                x = x.view_as(x)  # unnecessary, just here to test the dispatch
                return mysum(x, dim)

        bdim = torch._C._functorch.maybe_get_bdim(x)
        value = torch._C._functorch.get_unwrapped(x)

        with interpreter.lower():
            value = value.movedim(bdim, 0)
            result = mysum(value, dim + 1)

        return torch._C._functorch._add_batch_dim(result, 0, interpreter.level())

    @mysum.py_impl(torch._C._functorch.TransformType.Grad)
    def mysum_grad_rule(interpreter, x, dim):
        # [FUNCTION DOC] HELPER: Grad rule for mysum. Applies a single-level autograd Function
        # that unwraps the input for the grad level, calls the operator with grad enabled and the
        # interpreter lowered, rewraps the output, and has a backward that expands the gradient
        # over the summed dim. Runs under the single-level autograd function context.
        level = interpreter.level()

        class MySum(torch.autograd.function._SingleLevelFunction):
            # [CLASS DOC] HELPER: Single-level autograd Function for the grad rule of mysum; saves
            # the input shape and the dim on the context and expands the gradient back
            # in backward.
            @staticmethod
            def forward(ctx, x, dim):
                # [FUNCTION DOC] HELPER: Saves the shape and dim, unwraps x for the grad level,
                # calls the operator under grad with the interpreter lowered, rewraps the result
                # for the level and returns it.
                ctx.x_shape = x.shape
                ctx.dim = dim
                x = torch._C._functorch._unwrap_for_grad(x, level)
                with torch.enable_grad(), interpreter.lower():
                    x = x.view_as(x)  # unnecessary, just here to test the dispatch
                    y = mysum(x, dim)

                y = torch._C._functorch._wrap_for_grad(y, level)
                return y

            @staticmethod
            def backward(ctx, gy):
                # [FUNCTION DOC] HELPER: Inserts the summed dim into the gradient and expands it
                # to the saved input shape; returns it with None for the dim.
                return gy.unsqueeze(ctx.dim).expand(ctx.x_shape), None

        with enable_single_level_autograd_function():
            return MySum.apply(x, dim)

    @mysum.py_impl(torch._C.DispatchKey.AutogradCPU)
    def mysum_autograd_cpu(x, dim):
        # [FUNCTION DOC] HELPER: CPU autograd implementation: returns the sum of x over dim.
        return torch.sum(x, dim)

    @mysum.py_impl(torch._C.DispatchKey.AutogradCUDA)
    def mysum_autograd_cuda(x, dim):
        # [FUNCTION DOC] HELPER: CUDA autograd implementation: returns the sum of x over dim.
        return torch.sum(x, dim)

    return mysum


sum_pyop = construct_sum_pyop()


@markDynamoStrictTest
class TestHigherOrderOperatorInteraction(TestCase):
    # [CLASS DOC] Device-generic tests that apply grad and vmap to the custom mysum higher order
    # operator built at module level (basic use, vmap, grad, nested grad, no_grad interplay) plus
    # two name and functional-call checks.
    def test_basic_sum(self, device):
        # [FUNCTION DOC] Checks the operator applied directly equals the ordinary sum.
        x = torch.randn(2, 3, 4, device=device)
        result = sum_pyop(x, 1)
        self.assertEqual(result, torch.sum(x, 1))

    def test_vmap_sum(self, device):
        # [FUNCTION DOC] Checks vmap of the operator (and nested vmap) equals the ordinary sum
        # over the corresponding dim.
        x = torch.randn(2, 3, 4, device=device)
        result = vmap(sum_pyop, (0, None))(x, 0)
        self.assertEqual(result, torch.sum(x, 1))

        result = vmap(vmap(sum_pyop, (0, None)), (0, None))(x, 0)
        self.assertEqual(result, torch.sum(x, 2))

    def test_grad_sum(self, device):
        # [FUNCTION DOC] Checks grad of the operator gives ones.
        x = torch.randn(3, device=device)
        gx = grad(sum_pyop)(x, 0)
        self.assertEqual(gx, torch.ones_like(x))

    def test_grad_grad_sum(self, device):
        # [FUNCTION DOC] Checks second order grad through the operator on a sine (needs a
        # non-linearity) gives minus sin.
        x = torch.randn(3, requires_grad=True, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied to sin of x over dim zero.
            # higher order grad. Requires a non-linearity
            return sum_pyop(x.sin(), 0)

        def grad_f_sum(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the grad of the sine-sum function at x.
            return grad(f)(x).sum()

        ggx = grad(grad_f_sum)(x)
        self.assertEqual(ggx, -x.sin())

    def test_vmap_grad_sum(self, device):
        # [FUNCTION DOC] Checks vmap of grad of the operator gives ones for every row.
        x = torch.randn(2, 3, device=device)
        gx = vmap(grad(sum_pyop), (0, None))(x, 0)
        self.assertEqual(gx, torch.ones_like(x))

    def test_no_grad_outside_grad(self, device):
        # [FUNCTION DOC] Checks grad of the operator inside a no_grad block returns ones that do
        # not require grad.
        x = torch.randn(3, device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(sum_pyop)(x, 0)
        self.assertEqual(y, torch.ones_like(x))
        self.assertFalse(y.requires_grad)

    def test_no_grad_inside_grad(self, device):
        # [FUNCTION DOC] Checks no_grad inside the differentiated function: the operator value
        # computed there is a constant shift, giving gradient two x, second derivative two, and an
        # autograd derivative of two.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes the operator on x squared under no_grad as a shift
            # and returns the operator on x squared minus the shift.
            with torch.no_grad():
                shift = sum_pyop(x**2, 0)
            return sum_pyop(x**2, 0) - shift

        x = torch.randn(3, device=device)
        y = grad(f)(x)
        self.assertEqual(y, 2 * x)
        y = grad(lambda x: grad(f)(x).sum())(x)
        self.assertEqual(y, torch.full_like(x, 2))

        x = torch.randn(3, device=device, requires_grad=True)
        y = grad(f)(x)
        (z,) = torch.autograd.grad(y.sum(), x)
        self.assertEqual(z, torch.full_like(x, 2))

    def test_grad_name_wrapping(self, device):
        # [FUNCTION DOC] Checks the function returned by grad keeps the wrapped function name.
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        grad_fn = grad(my_fn)
        self.assertEqual(grad_fn.__name__, "my_fn")

    def test_functional_call_multiple_dicts(self):
        # [FUNCTION DOC] Smoke test: functional_call accepts a tuple of two parameter dicts (one
        # weight, one bias) for a linear layer and runs without error.
        mod = nn.Linear(1, 1)
        x = torch.randn((1, 1))
        params = ({"weight": torch.zeros(1, 1)}, {"bias": torch.ones(1)})
        functional_call(mod, params, x)


def traceable(f):
    # [FUNCTION DOC] HELPER: Marks the function as allowed in the Dynamo graph and returns a
    # wrapper preserving its metadata that forwards all positional and keyword arguments to it, so
    # a transformed function can be traced by torch.compile as a single call.
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the wrapped function and returns
        # its result.
        return f(*args, **kwargs)

    return wrapper


@markDynamoStrictTest
class TestCompileTransforms(TestCase):
    # [CLASS DOC] Device-generic tests that run function transforms under torch.compile: vmap of
    # nested Jacobians, dynamic shape regression tests for grad with strided slicing and with
    # custom pytree inputs, and the deprecated functorch grad API under full-graph compile.
    # torch.compile is not supported on Windows CUDA.
    # Triton only supports GPU with SM70 or later.
    @expectedFailureIf((IS_WINDOWS and TEST_CUDA) or (TEST_CUDA and not SM70OrLater))
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_compile_vmap_hessian(self, device):
        # [FUNCTION DOC] Compiles a vmapped forward-over-reverse hessian (with aux outputs) of a
        # small linear and relu model, and checks the compiled result equals the eager result.
        # Expected to fail on Windows with CUDA or on CUDA devices older than SM70 (no compile
        # support); skipped under the CUDA memory leak check.
        # The model and inputs are a smaller version
        # of code at benchmark repo:
        # https://github.com/pytorch/benchmark/blob/main/userbenchmark/functorch/vmap_hessian_fc.py
        D = 2
        B = 4

        x = torch.randn(B, D, device=device)

        model = nn.Sequential(nn.Linear(D, D), nn.ReLU()).to(device)

        params_and_buffers = (
            dict(model.named_parameters()),
            dict(model.named_buffers()),
        )

        def predict(params_and_buffers, x):
            # [FUNCTION DOC] HELPER: Runs the module functionally with the given parameters and
            # buffers and returns the output twice (as output and aux).
            out = torch.func.functional_call(model, params_and_buffers, x)
            return out, out

        fn = vmap(
            jacfwd(jacrev(predict, argnums=1, has_aux=True), argnums=1, has_aux=True),
            in_dims=(None, 0),
        )

        expected = fn(params_and_buffers, x)

        opt_fn = torch.compile(traceable(fn))
        actual = opt_fn(params_and_buffers, x)
        self.assertEqual(actual, expected)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_dynamic_grad_stride_slice_mha(self, device, backend):
        # [FUNCTION DOC] Parametrized over the eager and aot_eager backends. Regression test for
        # issue 181305: grad of a function that slices with a stride, averages, and applies a
        # linear layer, compiled with dynamic shapes after resetting Dynamo, equals the
        # eager gradient.
        # Regression test for https://github.com/pytorch/pytorch/issues/181305
        linear = nn.Linear(3, 3).to(device).eval()

        def model(x):
            # [FUNCTION DOC] HELPER: Takes every second element along dim one of x, averages over
            # dim zero, applies the linear layer and returns the mean.
            y = x[:, ::2, :, :]
            y = y.mean(dim=0)
            y = linear(y)
            return y.mean()

        x = torch.randn(4, 8, 2, 3, device=device)
        expected = grad(model)(x)

        torch._dynamo.reset()
        compiled = torch.compile(grad(model), dynamic=True, backend=backend)
        result = compiled(x)
        self.assertEqual(result, expected)

    def test_compile_dynamic_grad_vjp_index_select_pytree(self, device):
        # [FUNCTION DOC] Regression test for issue 171537. Registers a custom pytree node class,
        # then compares eager and dynamic-shape compiled (eager backend) grad_and_value of a
        # function that uses vjp over an input holding a positions tensor and index_select. Always
        # deregisters the pytree node afterwards, even on failure.
        # Regression test for https://github.com/pytorch/pytorch/issues/171537
        from torch.utils import _pytree as pytree

        class Input:
            # [CLASS DOC] HELPER: Simple container holding a positions tensor, registered as a
            # pytree node for this test.
            def __init__(self, positions):
                # [FUNCTION DOC] HELPER: Stores the positions tensor.
                self.positions = positions

        def flatten(inp):
            # [FUNCTION DOC] HELPER: Pytree flatten function: returns a list with the positions
            # tensor and no context.
            return [inp.positions], None

        def unflatten(children, context):
            # [FUNCTION DOC] HELPER: Pytree unflatten function: builds a new container from the
            # first child, ignoring the context.
            return Input(children[0])

        pytree.register_pytree_node(
            Input,
            flatten,
            unflatten,
            serialized_type_name="test_compile_dynamic_grad_vjp_index_select_pytree.Input",
        )
        try:

            def loss(weight, inp):
                # [FUNCTION DOC] HELPER: Computes a vjp of the energy function over the container
                # input with ones as cotangent, and returns the sum of the squared energy plus the
                # sum of the squared input gradient positions.
                def energy(input_pc):
                    # [FUNCTION DOC] HELPER: Index-selects five rows of the positions with an
                    # index range, multiplies by the weight and sums over dim one keeping the dim.
                    idx = torch.arange(5, device=input_pc.positions.device)
                    vals = input_pc.positions.index_select(0, idx) * weight
                    return vals.sum(dim=[1], keepdim=True)

                y, vjp_fn = torch.func.vjp(energy, inp)
                (input_grads,) = vjp_fn(torch.ones_like(y))
                return y.square().sum() + input_grads.positions.square().sum()

            fn = torch.func.grad_and_value(loss, argnums=0)
            weight = torch.randn(3, device=device, requires_grad=True)
            inp = Input(torch.randn(7, 3, device=device, requires_grad=True))

            expected = fn(weight, inp)

            torch._dynamo.reset()
            compiled = torch.compile(fn, dynamic=True, backend="eager")
            result = compiled(weight, inp)
            self.assertEqual(result, expected)
        finally:
            pytree._deregister_pytree_node(Input)

    def test_compile_dynamic_grad_vjp_trace_pytree(self, device):
        # [FUNCTION DOC] Regression test for issue 171537. Same as the index-select variant but
        # the energy is the trace of the elementwise product of a matrix with the weight. Compares
        # eager and dynamic-shape compiled grad_and_value, and deregisters the pytree
        # node afterwards.
        # Regression test for https://github.com/pytorch/pytorch/issues/171537
        from torch.utils import _pytree as pytree

        class Input:
            # [CLASS DOC] HELPER: Simple container holding a matrix tensor, registered as a pytree
            # node for this test.
            def __init__(self, matrix):
                # [FUNCTION DOC] HELPER: Stores the matrix tensor.
                self.matrix = matrix

        def flatten(inp):
            # [FUNCTION DOC] HELPER: Pytree flatten function: returns a list with the matrix
            # tensor and no context.
            return [inp.matrix], None

        def unflatten(children, context):
            # [FUNCTION DOC] HELPER: Pytree unflatten function: builds a new container from the
            # first child, ignoring the context.
            return Input(children[0])

        pytree.register_pytree_node(
            Input,
            flatten,
            unflatten,
            serialized_type_name="test_compile_dynamic_grad_vjp_trace_pytree.Input",
        )
        try:

            def loss(weight, inp):
                # [FUNCTION DOC] HELPER: Computes a vjp of the trace energy over the container
                # with ones as cotangent, and returns the squared energy plus the sum of the
                # squared input gradient matrix.
                def energy(input_pc):
                    # [FUNCTION DOC] HELPER: Returns the trace of the container matrix times
                    # the weight.
                    return torch.trace(input_pc.matrix * weight)

                y, vjp_fn = torch.func.vjp(energy, inp)
                (input_grads,) = vjp_fn(torch.ones_like(y))
                return y.square() + input_grads.matrix.square().sum()

            fn = torch.func.grad_and_value(loss, argnums=0)
            weight = torch.randn(4, 4, device=device, requires_grad=True)
            inp = Input(torch.randn(4, 4, device=device, requires_grad=True))

            expected = fn(weight, inp)

            torch._dynamo.reset()
            compiled = torch.compile(fn, dynamic=True, backend="eager")
            result = compiled(weight, inp)
            self.assertEqual(result, expected)
        finally:
            pytree._deregister_pytree_node(Input)

    # torch.compile is not supported on Windows
    @torch._dynamo.config.patch(suppress_errors=False)
    def test_grad_deprecated_api(self, device):
        # [FUNCTION DOC] Run with Dynamo suppress_errors disabled. Checks that the deprecated
        # functorch grad API (single argnums and tuple argnums on a product) compiles with the
        # eager backend in full-graph mode and equals the eager result.
        x = torch.randn((), device=device)
        y = torch.randn((), device=device)

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the functorch grad of the multiply op at x and y.
            return functorch.grad(torch.mul)(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="eager", fullgraph=True)(x, y)
        torch.compile(wrapper_fn, backend="eager", fullgraph=True)
        self.assertEqual(actual, expected)

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Redefinition returning the functorch grad of the multiply op
            # at x and y with respect to both arguments.
            return functorch.grad(torch.mul, argnums=(0, 1))(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(actual, expected)


class TestGradTrackingTensorToList(TestCase):
    # [CLASS DOC] Tests that tolist works on tensors wrapped by grad (GradTrackingTensor) in
    # plain, nested, multi-dimensional and conjugate cases, and that gradients are still computed.
    # Instantiated for each device type in the list at the end of the file.
    """Tests for tolist() method with GradTrackingTensor (functorch tensors)."""

    def test_tolist_with_grad(self):
        # [FUNCTION DOC] Checks that calling tolist on the input inside grad returns a python list
        # with the right values and that the final gradient is two times the input.
        """Test to see if tolist works inside grad transformation."""

        def f(x):
            # [FUNCTION DOC] HELPER: Asserts through the enclosing test that tolist returns the
            # list of values one, two, three, then returns the sum of squares.
            # inside grad, x is a GradTrackingTensor
            result = x.tolist()
            # tolist should return a python list and not fail
            self.assertIsInstance(result, list)
            self.assertEqual(result, [1.0, 2.0, 3.0])
            return (x**2).sum()

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        grad_f = torch.func.grad(f)
        result = grad_f(x)
        self.assertIsInstance(result, torch.Tensor)
        # gradients should still be computed correctly
        self.assertEqual(result, [2.0, 4.0, 6.0])

    def test_tolist_nested_grad(self):
        # [FUNCTION DOC] Checks tolist works on tensors at two nested grad levels and that the
        # second derivative of the sum of squares is twos.
        """Test `tolist` with nested grad transformations."""

        def f(x):
            # [FUNCTION DOC] HELPER: Calls tolist on the outer input, then returns the sum of the
            # grad of the inner function at it.
            def g(y):
                # [FUNCTION DOC] HELPER: Calls tolist on its input, asserts the result is a list
                # and returns the sum of squares.
                # y is gradTrackingTensor(lvl=1)
                inner_list = y.tolist()
                self.assertIsInstance(inner_list, list)
                return (y**2).sum()

            # x is a gradTrackingTensor(lvl=0)
            outer_list = x.tolist()
            self.assertIsInstance(outer_list, list)
            grad_g = torch.func.grad(g)
            return grad_g(x).sum()

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        grad_f = torch.func.grad(f)
        result = grad_f(x)
        # should compute second derivate
        self.assertIsInstance(result, torch.Tensor)
        # grad_f should return the derivate of g(y) which is (2*x).sum
        self.assertEqual(
            result,
            [
                2.0,
                2.0,
                2.0,
            ],
        )

    def test_tolist_multidimensional_grad(self):
        # [FUNCTION DOC] Checks tolist on a two by three input inside grad returns the nested list
        # of values and that the gradient of the sum is ones.
        """Test tolist with multi-dimensional tensors in grad."""

        def f(x):
            # [FUNCTION DOC] HELPER: Asserts the tolist result is a nested list of length two with
            # the expected values, and returns the sum.
            result = x.tolist()
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            return x.sum()

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        grad_f = torch.func.grad(f)
        result = grad_f(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(
            result,
            [
                [
                    1.0,
                    1.0,
                    1.0,
                ],
                [1.0, 1.0, 1.0],
            ],
        )

    def test_tolist_conj_neg_grad(self):
        # [FUNCTION DOC] Checks tolist of a conjugated complex tensor inside grad returns a list,
        # and that grad of the real part of the product with the conjugate gives the expected
        # complex gradient (twice the input).
        """Test tolist method with conjugate/negative tensors in grad context."""

        def f(x):
            # [FUNCTION DOC] HELPER: Takes the conjugate, asserts its tolist result is a list and
            # returns the sum of the real part of the product of the input with its conjugate.
            # test with the conjugate view
            x_conj = x.conj()
            result_conj = x_conj.tolist()
            self.assertIsInstance(result_conj, list)
            return (x * x.conj()).real.sum()

        x = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], requires_grad=True)
        grad_f = torch.func.grad(f)
        result = grad_f(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result, [2.0 + 4.0j, 6.0 + 8.0j])


only_for = ("cpu", "cuda")
instantiate_device_type_tests(
    TestGradTransform,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestVmapOfGrad,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestJac,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestJvp,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestLinearize,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestVmapJvpInplaceView,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestHessian,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestComposability,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestExamplesCorrectness,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestHigherOrderOperatorInteraction,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestFunctionalize,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestAutogradFunction,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestAutogradFunctionVmapAPI,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestHelpers,
    globals(),
    only_for=only_for,
)
instantiate_parametrized_tests(
    TestMakeFunctional,
)
instantiate_device_type_tests(
    TestCompileTransforms,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestGradTrackingTensorToList, globals(), only_for=only_for
)

if __name__ == "__main__":
    run_tests()
