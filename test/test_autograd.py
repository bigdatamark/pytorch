# Owner(s): ["module: autograd"]
# ruff: noqa: F841
# [MODULE DOC] Large test suite for the autograd engine and Python autograd API: custom Function
# classes, backward and grad semantics, hooks, saved-tensor hooks, anomaly detection, forward-mode
# AD, the profiler, checkpointing, graph and node introspection, and multithreaded or
# device-specific backward behavior. Contains the TestAutograd class, further per-feature test
# classes, and device-generic classes instantiated near the bottom. Importing it only defines
# tests; running it as a script calls the shared run_tests entry point.

import collections
import contextlib
import contextvars
import functools
import gc
import io
import math
import operator
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import weakref
from collections import defaultdict, OrderedDict
from copy import deepcopy
from functools import partial, reduce
from itertools import product
from operator import mul
from typing import TYPE_CHECKING

import torch
import torch.autograd._functions
import torch.autograd.forward_ad as fwAD
from torch import inf, nan, nn
from torch.autograd import (
    _calculate_shape,
    detect_anomaly,
    Function,
    kineto_available,
    Variable,
)
from torch.autograd.function import InplaceFunction, once_differentiable
from torch.autograd.graph import GradientEdge
from torch.autograd.profiler import emit_itt, emit_nvtx, profile, record_function
from torch.autograd.profiler_util import (
    _format_time,
    EventList,
    FunctionEvent,
    FunctionEventAvg,
)
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfCUDA,
    dtypesIfMPS,
    expectedFailureMPS,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_utils import (
    disable_gc,
    gradcheck,
    gradgradcheck,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    scoped_load_inline,
    set_warn_always_context,
    skipCUDANonDefaultStreamIf,
    skipIfMPS,
    skipIfNoLapack,
    skipIfSlowGradcheckEnv,
    skipIfTorchDynamo,
    skipIfWindows,
    skipIfXpu,
    slowTest,
    TEST_WITH_ASAN,
    TEST_WITH_SLOW,
    TEST_WITH_TORCHDYNAMO,
    TEST_XPU,
    TestCase,
    xfailIf,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    checkpoint,
    checkpoint_sequential,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.weak import WeakTensorKeyDictionary


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


def graph_desc(fn):
    # [FUNCTION DOC] Return a string describing the autograd graph rooted at the given grad_fn
    # node. A missing node gives the text None; otherwise the node class name followed by the
    # comma-separated descriptions of its next_functions inside parentheses, computed recursively.
    # Used to compare graph structure against expected strings.
    if fn is None:
        return "None"
    result = type(fn).__name__ + "("
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        result += graph_desc(next_fn)
        result += ", "
    if next_functions:
        result = result[:-2]
    return result + ")"


class TestAutograd(TestCase):
    # [CLASS DOC] Main collection of autograd tests: custom Function behavior, backward and grad,
    # hooks, views and in-place handling, error messages and graph structure. Each test builds its
    # own tensors; the class holds no state besides a few nested helper Function classes.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Resets the global original-view-tracking flag to disabled after
        # every test, then runs the base TestCase teardown, so tests that enable it do not leak.
        torch.autograd._force_original_view_tracking(False)
        super().tearDown()

    def test_copy_slices_graph_task_updates(self):
        # [FUNCTION DOC] Check that copy-slices graph tasks handle an error node correctly. A
        # DelayedError node wraps y so that executing it raises the error text Boom!. Verifies the
        # error appears for backward and grad when the node is needed, that no crash happens when
        # only x or y_safe is requested, and that an extra view node is not executed.
        def f1(x, y):
            # [FUNCTION DOC] HELPER: Clone x, flatten it with a view and add y in place; return
            # the result.
            out = x.clone().view(-1)
            out += y
            return out

        def f2(x, y):
            # [FUNCTION DOC] HELPER: Like f1, but also keeps a copy b of the view scaled by 2
            # before the in-place add, and returns the sum of the updated view and b.
            out = x.clone().view(-1)
            b = out * 2
            out += y
            return out + b

        x = torch.rand(2, requires_grad=True)
        y = torch.rand(2, requires_grad=True)

        y_safe = torch._C._functions.DelayedError("Boom!", 1)(y)

        for f in [f1, f2]:
            # Ensure that the error Node works
            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                out.sum().backward()

            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                torch.autograd.grad(out.sum(), y)

            # Ensure that if we don't ask for y, it doesn't crash
            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), x)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), y_safe)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), (x, y_safe))

        # Ensure that we don't run extra view Node
        def f3(x, y):
            # [FUNCTION DOC] HELPER: Clone x into a flattened view with a hook registered on it
            # that fails the test if run, then add y both in place and out of place; return the
            # in-place result and the out-of-place sum b.
            out = x.clone().view(-1)

            def hook(*args):
                # [FUNCTION DOC] HELPER: Hook that fails the test if it is ever called.
                # This should never be called!
                self.assertTrue(False)

            out.register_hook(hook)

            b = out + y
            out += y
            return out + b, b

        out, b = f3(x, y_safe)
        torch.autograd.grad(out.sum(), (b, y_safe))

    def test_grad_mode_class_decoration(self):
        # [FUNCTION DOC] Check no_grad applied to a class (deprecated) versus to functions and
        # methods. Decorating a class emits a FutureWarning about deprecation, its constructor
        # runs with grad disabled, but its other methods are not wrapped. Decorating functions or
        # methods emits no warning and disables grad inside them.
        # Decorating class is deprecated and should not be used
        with self.assertWarnsRegex(FutureWarning, "Decorating classes is deprecated"):

            @torch.no_grad()
            class Foo:
                # [CLASS DOC] HELPER: Class decorated with no_grad; the constructor asserts
                # grad mode is off, the foo method asserts grad mode is on (decorator not
                # applied to methods).
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Raises AssertionError if grad mode is enabled
                    # during construction.
                    if torch.is_grad_enabled():
                        raise AssertionError("expected grad to be disabled")

                def foo(self):
                    # [FUNCTION DOC] HELPER: Raises AssertionError if grad mode is disabled
                    # when called.
                    # Not applied to methods
                    if not torch.is_grad_enabled():
                        raise AssertionError("expected grad to be enabled")

            # Show that we can actually construct the class
            foo = Foo()
            foo.foo()

        # Decorating functions or methods is fine though
        with warnings.catch_warnings(record=True) as w:

            @torch.no_grad()
            def foo():
                # [FUNCTION DOC] HELPER: Function decorated with no_grad that raises
                # AssertionError if grad mode is enabled inside it.
                if torch.is_grad_enabled():
                    raise AssertionError("expected grad to be disabled")

            foo()

            class Foo2:
                # [CLASS DOC] HELPER: Class whose constructor and foo method are each individually
                # decorated with no_grad.
                @torch.no_grad()
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Constructor that raises AssertionError if grad mode
                    # is enabled.
                    if torch.is_grad_enabled():
                        raise AssertionError("expected grad to be disabled")

                @torch.no_grad()
                def foo(self):
                    # [FUNCTION DOC] HELPER: Method that raises AssertionError if grad mode
                    # is enabled.
                    if torch.is_grad_enabled():
                        raise AssertionError("expected grad to be disabled")

            foo2 = Foo2()
            foo2.foo()

        self.assertEqual(len(w), 0)

    def test_tensor_grad_warnings(self):
        # [FUNCTION DOC] Check the warning behavior of reading the grad attribute: no warning for
        # a leaf requiring grad, one warning for a non-leaf that does not retain grad, and no
        # additional warning after retain_grad is called on it.
        dummy = torch.empty(1)

        with warnings.catch_warnings(record=True) as w:
            # Accessing .grad on leaf
            dummy.requires_grad_()
            foo = dummy.grad
            self.assertEqual(len(w), 0)

            # Accessing .grad on non-leaf
            dummy = dummy.clone()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

            # Accessing .grad on non-leaf that retains gradients
            dummy.retain_grad()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

    def _function_test(self, cls):
        # [FUNCTION DOC] HELPER: Run a custom Function class on two 5x5 inputs and a scalar 2,
        # backward with create_graph and a grad-requiring scalar gradient, and check the grads
        # equal the other input plus (scalar times) ones; also check the grads carry a grad_fn.
        # Returns the two inputs.
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones((), requires_grad=True)
        result.sum().backward(go, create_graph=True)

        self.assertEqual(x.grad, y + torch.ones(5, 5))
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)
        self.assertIsNotNone(x.grad.grad_fn)
        self.assertIsNotNone(y.grad.grad_fn)

        return x, y

    def test_function(self):
        # [FUNCTION DOC] Check a basic custom Function with a saved-tensor-based
        # double-differentiable backward: the graph description of each gradient is compared with
        # stored expected output files, and the grads are reset to None at the end.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function computing t1 + pyscalar*t2 + t1*t2 that saves both
            # tensors and the scalar.
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                # [FUNCTION DOC] HELPER: Store the scalar and save both tensors on ctx; return the
                # combined value.
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return gradients for the two tensors and None for the
                # scalar, after checking the saved values and the incoming gradient are tensors.
                var1, var2 = ctx.saved_tensors
                # NOTE: self is the test case here
                self.assertIsInstance(var1, torch.Tensor)
                self.assertIsInstance(var2, torch.Tensor)
                self.assertIsInstance(grad_output, torch.Tensor)
                return (
                    grad_output + grad_output * var2,
                    None,
                    grad_output * ctx.pyscalar + grad_output * var1,
                )

        x, y = self._function_test(MyFunction)

        x_grad_desc = graph_desc(x.grad.grad_fn)
        y_grad_desc = graph_desc(y.grad.grad_fn)
        self.assertExpected(x_grad_desc, "x_grad_desc")
        self.assertExpected(y_grad_desc, "y_grad_desc")

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_once_differentiable(self):
        # [FUNCTION DOC] Check once_differentiable on a custom backward: grad mode is disabled
        # inside backward, and the resulting gradient graph shows an Error node that raises on
        # double backward (checked via graph description strings).
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Same forward as the plain custom Function test.
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                # [FUNCTION DOC] HELPER: Save tensors and scalar on ctx; return the
                # combined value.
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward marked once_differentiable; asserts grad mode is
                # off and returns the same gradients as the non-decorated version.
                self.assertFalse(torch.is_grad_enabled())
                t1, t2 = ctx.saved_tensors
                return (
                    grad_output + grad_output * t2,
                    None,
                    grad_output * ctx.pyscalar + grad_output * t1,
                )

        x, y = self._function_test(MyFunction)
        self.assertEqual(
            graph_desc(x.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )
        self.assertEqual(
            graph_desc(y.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_function_returns_input(self):
        # [FUNCTION DOC] Check a custom Function whose forward returns its input unchanged
        # still produces gradient 2 for both a leaf and a cloned input, for shapes of one
        # element and scalar.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Identity forward with backward doubling the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input as is.
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return twice the incoming gradient.
                return grad * 2

        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            MyFunction.apply(v).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))

            with torch.no_grad():
                v.grad.zero_()
            MyFunction.apply(v.clone()).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))

    def test_function_returns_undefined_tensor(self):
        # [FUNCTION DOC] Check that a backward returning None gives an undefined gradient (the
        # grad stays None) rather than zeros, both for direct, nonleaf and summed uses, and for
        # grad with allow_unused.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function scaling input by 2 whose backward returns None.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input times 2.
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return None as the gradient.
                return None

        # Test that undefined tensors returned from custom backward function
        # are propagated as undefined and not tensor full of zeroes
        x = torch.ones(1, requires_grad=True)

        MyFunction.apply(x).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x**2).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x).sum().backward()
        self.assertIsNone(x.grad)

        self.assertIsNone(
            torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0]
        )

    def test_materialize_grads(self):
        # [FUNCTION DOC] Check that an undefined incoming gradient (injected by UndefinedGrad) is
        # materialized as zeros by default before reaching the custom backward.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Identity Function whose backward asserts the incoming gradient
            # equals zeros.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Check the gradient equals a zero tensor of one element
                # and return it.
                self.assertEqual(grad, torch.zeros(1))
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_dont_materialize_grads(self):
        # [FUNCTION DOC] Check that turning off set_materialize_grads makes the custom backward
        # receive None for an undefined incoming gradient.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Identity Function that disables gradient materialization.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Disable materialization of grads on ctx and return
                # the input.
                ctx.set_materialize_grads(False)
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Assert the incoming gradient is None and return it.
                self.assertIsNone(grad)
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_set_materialize_non_diff_grads(self):
        # [FUNCTION DOC] Check the private flag controlling materialization of non-differentiable
        # outputs: with it off, a non-differentiable second output reaches backward as None.
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning two clones, one marked non-differentiable.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Mark the second output non-differentiable, turn off
                # private materialization of non-differentiable grads, return both clones.
                out0 = x.clone()
                out1 = x.clone()
                ctx.mark_non_differentiable(out1)
                ctx._materialize_non_diff_grads = False
                return out0, out1

            @staticmethod
            def backward(ctx, g0, g1):
                # [FUNCTION DOC] HELPER: Assert the second gradient is None and return the first.
                self.assertIsNone(g1)
                return g0

        a = torch.tensor(1.0, requires_grad=True)
        out = Func.apply(a)[0]
        out.backward()

    def test_unused_grad_requires_grad_with_materialize(self):
        # [FUNCTION DOC] Check that grad with allow_unused and materialize_grads returns zero
        # gradients for an unused input without requires_grad set on the results.
        x = torch.ones(10, requires_grad=True)
        y = torch.ones(10, requires_grad=True)
        z = (x**2).sum()

        g = torch.autograd.grad(
            z, (x, y), allow_unused=True, materialize_grads=True, create_graph=False
        )

        self.assertFalse(g[0].requires_grad)
        self.assertFalse(g[1].requires_grad)

    def test_legacy_function_deprecation_exception(self):
        # [FUNCTION DOC] Check that using a legacy Function with a non-static forward raises
        # RuntimeError saying such functions are deprecated.
        # Trigger exception
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Legacy-style Function with instance-method forward and backward.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            def backward(self, grad_output):
                # [FUNCTION DOC] HELPER: Return the incoming gradient.
                return grad_output

        # Check exception occurs
        with self.assertRaisesRegex(
            RuntimeError,
            "Legacy autograd function with non-static forward method is deprecated",
        ):
            MyFunction()(torch.randn(3, 4))

    class SimulateBackwardError(Function):
        # [CLASS DOC] HELPER: Function whose backward always raises a generic
        # Exception with the message about a simulated error on the backward pass;
        # used by test_custom_function_exception.
        @staticmethod
        def forward(ctx, input):
            # [FUNCTION DOC] HELPER: Return a clone of the input.
            return input.clone()

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            # [FUNCTION DOC] HELPER: Backward (once_differentiable) that raises an Exception with
            # a simulated error message.
            raise Exception("Simulate error on backward pass")  # noqa: TRY002

    def test_custom_function_exception(self):
        # [FUNCTION DOC] Check that an exception raised inside a custom backward propagates out of
        # backward unchanged.
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        tmp = (t1 + t2) * (t1 + t2)
        t3 = TestAutograd.SimulateBackwardError.apply(tmp)
        with self.assertRaisesRegex(Exception, "Simulate error on backward pass"):
            t3.sum().backward()

    def test_custom_function_non_tensor_inputs_outputs(self):
        # [FUNCTION DOC] Check a custom Function taking and returning non-tensor values mixed
        # with tensors. Verifies the non-tensor outputs pass through, backward receives seven
        # gradients with None for non-tensors, gradients reach only inputs requiring grad, and
        # gradcheck passes.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function with tensor and scalar inputs returning a seven-element
            # tuple mixing scalars, tensors, None, a bool and a string.
            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                # [FUNCTION DOC] HELPER: Compute two scaled tensors, save the scale and tensors on
                # ctx and return the mixed tuple including the first input.
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale

                # Save scale
                ctx.scale = scale
                ctx.save_for_backward(t1, t2, t3)
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *grads):
                # [FUNCTION DOC] HELPER: Check the gradient count and None entries, then compute
                # gradients for t1, t2 and t3 from the tensor gradients and saved values; None for
                # the scale.
                # Verify grads
                self.assertEqual(7, len(grads))
                self.assertIsNone(grads[0])
                self.assertIsNone(grads[2])
                self.assertIsNone(grads[3])
                self.assertIsNone(grads[5])

                scale = ctx.scale
                var1, var2, var3 = ctx.saved_tensors
                return (
                    grads[1] * scale + grads[4] * var2 * scale + grads[6],
                    grads[1] * var3 * scale + grads[4] * var1 * scale,
                    None,
                    grads[1] * var2 * scale + grads[4] * scale,
                )

        t1 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t2 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t3 = torch.rand(10, dtype=torch.double)
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        # Test gradcheck
        def foo(t1, t2, t3):
            # [FUNCTION DOC] HELPER: Wrapper for gradcheck returning only the tensor outputs of
            # the custom Function.
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

        gradcheck(foo, (t1, t2, t3))

    def test_custom_function_no_tensors(self):
        # [FUNCTION DOC] Check a custom Function called with no tensor arguments at all (only
        # Python floats and an int): outputs are computed and returned correctly without
        # autograd involvement.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function with scalar inputs returning a seven-element
            # mixed tuple.
            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                # [FUNCTION DOC] HELPER: Compute scaled sums and products and return them in the
                # mixed tuple.
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *args):
                # [FUNCTION DOC] HELPER: Return gradient values rearranged; never expected to run
                # in this test.
                return (args[0], args[1], None, args[2])

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

    def test_custom_function_apply_kwargs(self):
        # [FUNCTION DOC] Check that apply accepts keyword arguments for forward parameters:
        # positional, keyword, and defaulted calls give the same outputs and gradients.
        class MySin(Function):
            # [CLASS DOC] HELPER: Function computing factor times sin of the input.
            @staticmethod
            def forward(ctx, input, factor=1):
                # [FUNCTION DOC] HELPER: Save the input and factor and return the scaled sine.
                ctx.save_for_backward(input)
                ctx.factor = factor
                return factor * torch.sin(input)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the scaled cosine gradient and None for
                # the factor.
                (input,) = ctx.saved_tensors
                return grad_output * ctx.factor * torch.cos(input), None

        x = torch.tensor([0.812], requires_grad=True)

        # Baseline: positional
        out_pos = MySin.apply(x, 6)
        out_pos.backward()
        grad_pos = x.grad.clone()

        # kwarg
        x.grad = None
        out_kw = MySin.apply(x, factor=6)
        self.assertEqual(out_kw, out_pos)
        out_kw.backward()
        self.assertEqual(x.grad, grad_pos)

        # default value (no kwarg)
        x.grad = None
        out_default = MySin.apply(x)
        out_default_expected = MySin.apply(x, 1)
        self.assertEqual(out_default, out_default_expected)

    def test_custom_function_apply_kwargs_required(self):
        # [FUNCTION DOC] Check apply with all required arguments passed as keyword arguments, in
        # either order, gives the correct output and gradients of one.
        class Add(Function):
            # [CLASS DOC] HELPER: Function adding two tensors.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Save both inputs and return their sum.
                ctx.save_for_backward(x, y)
                return x + y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Pass the gradient through to both inputs.
                return grad_output, grad_output

        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)

        # All required args as kwargs
        out = Add.apply(x=x, y=y)
        self.assertEqual(out, torch.tensor([5.0]))
        out.backward()
        self.assertEqual(x.grad, torch.tensor([1.0]))
        self.assertEqual(y.grad, torch.tensor([1.0]))

        # Swap order of kwargs
        x.grad = None
        y.grad = None
        out = Add.apply(y=y, x=x)
        self.assertEqual(out, torch.tensor([5.0]))
        out.backward()
        self.assertEqual(x.grad, torch.tensor([1.0]))
        self.assertEqual(y.grad, torch.tensor([1.0]))

    def test_custom_function_apply_kwargs_tensor(self):
        # [FUNCTION DOC] Check apply with a mix of positional and keyword arguments for tensor and
        # scalar parameters, verifying output 30 and gradients 15 and 10.
        class ScaledMul(Function):
            # [CLASS DOC] HELPER: Function computing scale times x times y.
            @staticmethod
            def forward(ctx, x, y, scale=1.0):
                # [FUNCTION DOC] HELPER: Save both tensors, store scale and return the product.
                ctx.save_for_backward(x, y)
                ctx.scale = scale
                return scale * x * y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return gradients for both tensors using the saved values
                # and scale, and None for the scale.
                x, y = ctx.saved_tensors
                return (
                    grad_output * ctx.scale * y,
                    grad_output * ctx.scale * x,
                    None,
                )

        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)

        # Tensor args as kwargs
        out = ScaledMul.apply(x, y=y, scale=5.0)
        self.assertEqual(out, torch.tensor([30.0]))
        out.backward()
        self.assertEqual(x.grad, torch.tensor([15.0]))
        self.assertEqual(y.grad, torch.tensor([10.0]))

    def test_custom_function_apply_kwargs_setup_context(self):
        # [FUNCTION DOC] Check keyword arguments with a Function that uses the setup_context style
        # (forward without ctx): factor given by keyword scales the output and the gradient.
        class MyScale(Function):
            # [CLASS DOC] HELPER: Function scaling by a factor using separate setup_context.
            @staticmethod
            def forward(x, factor=1):
                # [FUNCTION DOC] HELPER: Return the input times factor.
                return factor * x

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Unpack inputs, save the tensor and store the factor
                # on ctx.
                x, factor = inputs
                ctx.save_for_backward(x)
                ctx.factor = factor

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient times the factor and None.
                return gO * ctx.factor, None

        x = torch.randn([], requires_grad=True)
        y = MyScale.apply(x, factor=3)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, torch.tensor(3.0))

    def test_custom_function_apply_kwargs_errors(self):
        # [FUNCTION DOC] Check TypeError messages for bad keyword usage in apply: an unexpected
        # keyword argument, and a keyword colliding with an already supplied positional argument
        # (multiple values for argument).
        class MyFn(Function):
            # [CLASS DOC] HELPER: Two-input Function adding its inputs.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Return the sum.
                return x + y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient for each input.
                return grad_output, grad_output

        x = torch.tensor([1.0])
        y = torch.tensor([2.0])

        with self.assertRaisesRegex(TypeError, "got an unexpected keyword argument"):
            MyFn.apply(x, bad_kwarg=y)

        # Duplicate: positional + kwarg for same param
        with self.assertRaisesRegex(TypeError, "got multiple values for argument"):
            MyFn.apply(x, y, y=y)

        # kwarg collides with an earlier positional arg: f(x, y, z) called
        # as f(2, 3, y=4) -- y is already filled by position 1
        class MyFn3(Function):
            # [CLASS DOC] HELPER: Three-input Function adding its inputs, used to check the
            # duplicate-argument error.
            @staticmethod
            def forward(ctx, x, y, z):
                # [FUNCTION DOC] HELPER: Return the sum of the three inputs.
                return x + y + z

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient for each of three inputs.
                return grad_output, grad_output, grad_output

        z = torch.tensor([3.0])
        with self.assertRaisesRegex(TypeError, "got multiple values for argument"):
            MyFn3.apply(x, y, y=z)

    def test_invalid_gradients(self):
        # [FUNCTION DOC] Check that a custom backward returning a gradient of the wrong shape
        # makes backward raise RuntimeError mentioning the expected shape.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function doubling its input with a deliberately
            # wrong-shaped gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input times 2.
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return a random float tensor of ten elements regardless
                # of input shape.
                return torch.randn(10, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, "expected shape"):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            MyFunction.apply(input).sum().backward()

    def test_unrelated_inputs(self):
        # [FUNCTION DOC] Check gradcheck and gradgradcheck succeed when an input is differentiable
        # but unrelated to the output.
        # test to ensure grad(grad)check runs successfully even if there is an
        # unrelated (but differentiable) inputs

        def my_function(x, y):
            # [FUNCTION DOC] HELPER: Function returning x squared elementwise and ignoring y.
            return x * x

        x = torch.rand(10, dtype=torch.double, requires_grad=True)
        y = torch.rand(10, dtype=torch.double, requires_grad=True)

        gradcheck(my_function, (x, y))
        gradgradcheck(my_function, (x, y))

    def test_not_implemented_grad(self):
        # [FUNCTION DOC] Check that an in-place acosh with no implemented derivative raises
        # NotImplementedError saying the derivative is not implemented when backward runs.
        # Test that built-in functions with unimplemented gradients raise appropriate errors
        a = torch.rand(2, requires_grad=True)
        y = torch.acosh_(a.clone()).sum()

        with self.assertRaisesRegex(
            NotImplementedError, "the derivative for .* is not implemented"
        ):
            y.backward()

    def test_not_implemented_fwad(self):
        # [FUNCTION DOC] Check that using forward AD with igamma, which has no forward formula,
        # raises NotImplementedError with a message about not supporting forward AD.
        x = torch.randn(3)
        v = torch.rand(3)

        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, v)

            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"

            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                # if forward AD ends up being implemented for torch.igamma, choose a different op
                torch.igamma(dual_x, dual_x)

    def test_saved_tensor_hooks_extra_exit_during_bw_no_crash(self):
        # [FUNCTION DOC] Check that exiting a saved-tensor hooks context an extra time while
        # backward unpacks does not crash, then clean up the hook stack and verify no hooks remain
        # by registering hooks on a saved result.
        # This usage of saved tensor is not supported, but should not crash
        def unpack(x):
            # [FUNCTION DOC] HELPER: Unpack hook that exits the first hooks context and returns
            # the tensor.
            ctx_1.__exit__()
            return x

        ctx_1 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, unpack)
        ctx_2 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x)

        for _ in range(10):
            with ctx_2:
                ctx_1.__enter__()
                x = torch.randn(3, 3, requires_grad=True)
                x.sin().sum().backward()

        # Clean up
        for _ in range(10):
            ctx_1.__exit__()

        # Validate there are no more hooks on the stack
        a = torch.tensor(1.0, requires_grad=True)
        y = a.exp()
        y.grad_fn._raw_saved_result.register_hooks(lambda x: x, lambda x: x)

    def test_saved_tensor_hooks_extra_enter_during_bw_no_leak(self):
        # [FUNCTION DOC] Check that entering a hooks context again from inside the unpack hook
        # during backward does not leak: after scope ends, a weak reference to the unpack hook is
        # dead (garbage collection disabled).
        # This usage of saved tensor is not supported, but should not leak
        def scope():
            # [FUNCTION DOC] HELPER: Run backward under a self-reentering hooks context and return
            # a weak reference to the unpack hook.
            def unpack(x):
                # [FUNCTION DOC] HELPER: Unpack hook that re-enters its own hooks context through
                # a weak reference and returns the tensor.
                weak_ctx_1().__enter__()
                return x

            ctx_1 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, unpack)
            weak_ctx_1 = weakref.ref(ctx_1)

            x = torch.randn(3, 3, requires_grad=True)
            with ctx_1:
                x.sin().sum().backward()
            return weakref.ref(unpack)

        with disable_gc():
            unpack_hook_ref = scope()
            self.assertIsNone(unpack_hook_ref())

    def test_will_engine_execute_node(self):
        # [FUNCTION DOC] Check the private query that tells whether the engine will execute a
        # node, from inside tensor hooks, for backward with and without inputs, grad with
        # leaf and non-leaf arguments, and the errors raised when called outside backward or
        # with a non-node.
        counter = [0]

        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function doubling its input in both forward and backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input times 2.
                return x * 2

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient times 2.
                return gO * 2

        def get_grad_fn(t):
            # [FUNCTION DOC] HELPER: Return the grad_fn of a tensor, using the AccumulateGrad node
            # of a requires-grad leaf obtained via a clone.
            if t.requires_grad and t.grad_fn is None:
                return t.clone().grad_fn.next_functions[0][0]
            else:
                return t.grad_fn

        a = torch.randn(2, 3, 4, requires_grad=True)
        a2 = torch.randn(2, 3, 4, requires_grad=True)
        b = a * a2
        b2 = b.cos()
        c = MyFunction.apply(b)

        should_execute = list(map(get_grad_fn, (a, b, c)))
        should_not_execute = list(map(get_grad_fn, (a2, b2)))

        def fn(x):
            # [FUNCTION DOC] HELPER: Hook that counts calls and asserts which of the captured
            # nodes will or will not be executed.
            counter[0] += 1

            for g in should_execute:
                self.assertTrue(torch._C._will_engine_execute_node(g))

            for g in should_not_execute:
                self.assertFalse(torch._C._will_engine_execute_node(g))

        h1 = b.register_hook(fn)
        h2 = c.register_hook(fn)

        # .backward(inputs=) is OK
        out = c.sum()
        torch.autograd.backward(out, inputs=(a, b), retain_graph=True)
        self.assertEqual(counter[0], 2)

        # .backward() is OK
        should_execute = list(map(get_grad_fn, (a, a2, b, c)))
        should_not_execute = list(map(get_grad_fn, (b2,)))
        torch.autograd.backward(out, retain_graph=True)

        # .grad is NOT OK when leaf is passed (this is the current state, subject to change)
        with self.assertRaisesRegex(
            RuntimeError, "are currently running autograd.grad()"
        ):
            torch.autograd.grad(out, (a,))

        # .grad is OK when non-leaf is passed
        a = torch.randn(1, 2, 3, requires_grad=True) * 2
        b = a * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Hook that counts calls and asserts the non-leaf grad_fn of b
            # will be executed.
            # Check a non-leaf
            counter[0] += 1
            self.assertTrue(torch._C._will_engine_execute_node(b.grad_fn))

        h3 = b.register_hook(fn)
        counter[0] = 0
        torch.autograd.grad(b.sum(), (a,))
        self.assertEqual(counter[0], 1)

        # Verify other errors are raised
        with self.assertRaisesRegex(RuntimeError, "during the backward pass"):
            torch._C._will_engine_execute_node(out.grad_fn)

        with self.assertRaisesRegex(RuntimeError, "expects an grad_fn"):
            torch._C._will_engine_execute_node(out)

        # Ensure we don't leak memory
        h1.remove()
        h2.remove()
        h3.remove()

    def test_custom_function_vmap_defaults(self):
        # [FUNCTION DOC] Check default vmap attributes of a Function using the setup_context
        # style: generate_vmap_rule is false and a vmap method exists.
        class MySquare(Function):
            # [CLASS DOC] HELPER: Squaring Function in setup_context style.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Return the square.
                return x**2

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Save the input on ctx.
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient times 2 times the saved input.
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        self.assertFalse(MySquare.generate_vmap_rule)
        self.assertTrue(hasattr(MySquare, "vmap"))

    def test_custom_function_setup_context_simple(self):
        # [FUNCTION DOC] Check a simple setup_context-style Function: gradient of squaring equals
        # twice the input.
        class MySquare(Function):
            # [CLASS DOC] HELPER: Squaring Function with separate setup_context.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Return the square.
                return x**2

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Save the input on ctx.
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return gradient times 2 times the saved input.
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        x = torch.randn([], requires_grad=True)
        y = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_output(self):
        # [FUNCTION DOC] Check setup_context with multiple outputs where one is a non-tensor: the
        # second output is stored on ctx and used in backward.
        # Multiple outputs with some non-Tensor outputs.
        class MySquare(Function):
            # [CLASS DOC] HELPER: Function returning a squared tensor and the python float twice
            # the item.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Return the square and twice the scalar value.
                two_x = x.item() * 2
                return x**2, two_x

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Store the non-tensor output on ctx.
                (x,) = inputs
                _, two_x = output
                ctx.two_x = two_x

            @staticmethod
            @once_differentiable
            def backward(ctx, gO, _):
                # [FUNCTION DOC] HELPER: Backward (once_differentiable) returning the gradient
                # times the stored value.
                return gO * ctx.two_x

        x = torch.randn([], requires_grad=True)
        y, _ = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_input(self):
        # [FUNCTION DOC] Check setup_context with multiple inputs against an equivalent reference
        # Function that uses the ctx-in-forward style: outputs and gradients must match for two
        # shapes and scales.
        class MyReshape(Function):
            # [CLASS DOC] HELPER: Reshape and scale Function in setup_context style.
            @staticmethod
            def forward(x, shape, scale_forward, scale_backward):
                # [FUNCTION DOC] HELPER: Return the reshaped input times the forward scale.
                return x.reshape(shape) * scale_forward

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Store the backward scale and input shape on ctx.
                x, shape, scale_forward, scale_backward = inputs
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Reshape the gradient to the input shape, scale it, and
                # return None for non-tensor inputs.
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        class MyReshapeRef(Function):
            # [CLASS DOC] HELPER: Reference reshape and scale Function that stores state
            # in forward.
            @staticmethod
            def forward(ctx, x, shape, scale_forward, scale_backward):
                # [FUNCTION DOC] HELPER: Store backward scale and input shape and return the
                # reshaped, scaled input.
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape
                return x.reshape(shape) * scale_forward

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Same backward as the setup_context version.
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        def test(x, shape, scale_forward, scale_backward):
            # [FUNCTION DOC] HELPER: Compare output and input gradient between the setup_context
            # Function and the reference.
            y = MyReshape.apply(x, shape, scale_forward, scale_backward).sum()
            (gx,) = torch.autograd.grad(y, x)

            y_expected = MyReshapeRef.apply(
                x, shape, scale_forward, scale_backward
            ).sum()
            (gx_expected,) = torch.autograd.grad(y_expected, x)

            self.assertEqual(y_expected, y)
            self.assertEqual(gx_expected, gx)

        test(torch.randn(24, requires_grad=True), (3, 8), 7, 11)
        test(torch.randn(2, 3, 4, requires_grad=True), (6, 4), -1, 2)

    @skipIfTorchDynamo("dynamo inlines setup_context, so the value stays live")
    def test_custom_function_setup_context_releases_return_value(self):
        # [FUNCTION DOC] Check that the value returned from setup_context is released: after
        # apply, a weak reference to the sentinel it created is dead. Skipped under dynamo, which
        # inlines setup_context.
        class Sentinel:
            # [CLASS DOC] HELPER: Empty marker class used to track object lifetime.
            pass

        sentinel_ref = None

        class MyFunc(Function):
            # [CLASS DOC] HELPER: Function whose setup_context creates a sentinel, records a weak
            # reference in an outer variable, and returns it.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Return a clone of the input.
                return x.clone()

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Create a sentinel, record a weak reference to it in the
                # enclosing scope and return it.
                nonlocal sentinel_ref
                sentinel = Sentinel()
                sentinel_ref = weakref.ref(sentinel)
                return sentinel

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the incoming gradient.
                return gO

        MyFunc.apply(torch.randn(3, requires_grad=True))

        self.assertIsNone(sentinel_ref())

    def test_multiple_insert_removal_caching(self):
        # [FUNCTION DOC] Check that adding a tensor to the cached-tensor set twice then
        # removing it once makes it no longer cached; cached-tensor mode is turned off again in
        # a finally block.
        torch._C._set_cached_tensors_enabled(True)
        try:
            x = torch.rand([4])

            torch._C._add_cached_tensor(x)
            self.assertTrue(torch._C._is_cached_tensor(x))

            torch._C._add_cached_tensor(x)
            torch._C._remove_cached_tensor(x)

            self.assertFalse(torch._C._is_cached_tensor(x))
        finally:
            torch._C._set_cached_tensors_enabled(False)

    def test_accumulate_grad(self):
        # [FUNCTION DOC] Check gradient accumulation into the grad attribute: in place when
        # create_graph is false (value doubles), out of place when true (value unchanged for the
        # held reference).
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            # [FUNCTION DOC] HELPER: Backward twice through x plus 2, second time with the
            # given create_graph flag; return the grad reference and a clone taken before the
            # second pass.
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)
            return x_grad, x_grad_clone

        # Accumulate in-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)

        # Accumulate out-of-place when create_graph is True
        x_grad, x_grad_clone = compute_grad(create_graph=True)
        self.assertEqual(x_grad, x_grad_clone)

    def test_accumulate_grad_tensor_reference(self):
        # [FUNCTION DOC] Check when the grad reference is preserved on accumulation:
        # never for dense into sparse, and only when create_graph is false for
        # dense-dense and sparse-sparse.
        def _test_grad_tensor(
            params_grad_tensor,
            backward_grad_tensor,
            should_preserve_reference,
            create_graph,
        ):
            # [FUNCTION DOC] HELPER: Set an initial grad, run backward with a gradient and
            # create_graph flag, and assert whether the grad object identity is kept as expected.
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            params.grad = params_grad_tensor
            grad_saved = params.grad
            params.backward(backward_grad_tensor, create_graph=create_graph)
            self.assertEqual(
                id(grad_saved) == id(params.grad), should_preserve_reference
            )

        for create_graph in (False, True):
            # Accumulate dense gradient to sparse gradient will change the `params.grad` reference
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.tensor([1.5, 1.5]),
                False,  # never accumulates in-place
                create_graph,
            )

            # Accumulate dense gradient to dense gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.tensor([1.5, 1.5]),
                torch.tensor([1.5, 1.5]),
                not create_graph,
                create_graph,
            )

            # Accumulate sparse gradient to sparse gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                not create_graph,
                create_graph,
            )

    def test_accumulate_grad_with_zero_numel_grad(self):
        # [FUNCTION DOC] Check accumulation of gradients of a broadcast to a zero-element
        # shape: the zero-numel operand gets a zero-element grad and the other gets zeros of
        # its own shape.
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        c = a + b
        if c.shape != (4, 0):
            raise AssertionError(f"expected shape (4, 0), got {c.shape}")
        c.sum().backward()

        self.assertEqual(b.grad, torch.zeros(4, 1))
        self.assertEqual(a.grad, torch.zeros(4, 0))

    def test_hessian_vector(self):
        # [FUNCTION DOC] Check Hessian-vector product by backward with create_graph, then second
        # backward on a combination of first gradients, comparing grads to analytic values.
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

        z = x**2 + y * x + y**2
        z.backward(torch.ones(2, 2), create_graph=True)

        with torch.no_grad():
            x_grad = 2 * x + y
            y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x.grad, x_grad + x_hv)
        self.assertEqual(y.grad, y_grad + y_hv)

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_grad(self):
        # [FUNCTION DOC] Check autograd.grad computes a Hessian-vector product with create_graph,
        # leaving the grad attributes untouched by the second derivative, and that a grad_outputs
        # shape differing from the output shape raises RuntimeError with the exact message about a
        # shape mismatch.
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x**2 + y * x + y**2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x + y
        y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(
            outputs=[grad_sum],
            grad_outputs=[torch.ones(2, 2)],
            inputs=[x],
            create_graph=True,
        )
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4

        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        # Avoid leaking memory
        x.grad = None
        y.grad = None

        # Test that grad_outputs and outputs have the same shape
        grad_out = torch.ones(2)
        try:
            torch.autograd.grad(
                outputs=[grad_sum],
                grad_outputs=[grad_out],
                inputs=[x],
                create_graph=True,
            )
            self.assertFail()
        except RuntimeError as error:
            self.assertEqual(
                str(error),
                "Mismatch in shape: grad_output[0] has a shape of "
                + str(grad_out.shape)
                + " and output[0] has a shape of "
                + str(grad_sum.shape)
                + ".",
            )

    def test_grad_to_node(self):
        # [FUNCTION DOC] Check gradients taken with respect to a gradient edge match gradients
        # taken with respect to the tensor, for a plain C++ node, an AccumulateGrad leaf, a legacy
        # Resize Function and a multi-output C++ node.
        def check_matches(out, inp):
            # [FUNCTION DOC] HELPER: Compute the gradient of the summed output with respect to the
            # tensor and with respect to its gradient edge and assert they are equal.
            ref = torch.autograd.grad(out.sum(), inp)

            edge = torch.autograd.graph.get_gradient_edge(inp)
            new = torch.autograd.grad(out.sum(), edge)
            self.assertEqual(ref, new)

        # We need to ensure that our main types of Node work (regular cpp Nodes,
        # AccumulateGrad Nodes and custom Function)
        x = torch.rand(2, requires_grad=True)
        out = x.clone()
        check_matches(out, x)

        x = x.clone()
        out = x.clone()
        check_matches(out, x)

        x = torch.autograd._functions.Resize.apply(x, (2,))
        out = x.clone()
        check_matches(out, x)

        x = torch.var_mean(x)[1]
        out = x.clone()
        check_matches(out, x)

    def test_grad_to_node_set(self):
        # [FUNCTION DOC] Check that a gradient edge captured before a tensor is mutated with set_
        # under no_grad still works, while asking for the tensor itself raises RuntimeError saying
        # the tensor was not used in the graph.
        x = torch.rand(2, requires_grad=True)
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        out = x.clone()

        with torch.no_grad():
            x.set_(torch.rand_like(x))

        with self.assertRaisesRegex(RuntimeError, "to not have been used in the graph"):
            torch.autograd.grad(out.sum(), x)

        # Works
        torch.autograd.grad(out.sum(), x_edge)

    def test_grad_to_node_inplace(self):
        # [FUNCTION DOC] Check gradients with respect to a gradient edge taken before an in-place
        # multiply: the edge reports the gradient through the multiply (2), the updated tensor
        # reports 1.
        x = torch.rand(2, requires_grad=True).clone()
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        x *= 2

        g_old, g_new = torch.autograd.grad(x.sum(), (x_edge, x))
        self.assertEqual(g_old, 2 * torch.ones_like(x))
        self.assertEqual(g_new, torch.ones_like(x))

    def test_grad_to_node_multi(self):
        # [FUNCTION DOC] Check that grad accepts a tuple of GradientEdge objects built from
        # grad_fn and output_nr, matching the result of passing the tensors.
        x = torch.rand(2, requires_grad=True).clone()
        y = torch.rand(2, requires_grad=True).clone()

        out = x + y

        ref = torch.autograd.grad(out.sum(), (x, y))

        inp_edges = (
            GradientEdge(x.grad_fn, x.output_nr),
            GradientEdge(y.grad_fn, y.output_nr),
        )
        new = torch.autograd.grad(out.sum(), inp_edges)

        self.assertEqual(ref, new)

    def test_grad_to_node_materialize(self):
        # [FUNCTION DOC] Check materialize_grads with GradientEdge inputs: allowed when
        # the edge inputs are used, and RuntimeError saying it cannot be used when an
        # input is a GradientEdge when an unused edge input is combined with allow_unused
        # and materialize_grads.
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        out = x.clone()

        # Works
        torch.autograd.grad(
            out.sum(), (edge_x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(
            out.sum(), (x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(out.sum(), (x, edge_y), allow_unused=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "materialize_grads cannot be used when the given input is a GradientEdge",
        ):
            torch.autograd.grad(
                out.sum(), (x, edge_y), allow_unused=True, materialize_grads=True
            )

    def test_backward_to_node(self):
        # [FUNCTION DOC] Check backward accepts the inputs argument containing any mix of tensors
        # and gradient edges.
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        out = x.clone()

        # All should work in this case
        torch.autograd.backward(out.sum(), inputs=(edge_x, y))
        torch.autograd.backward(out.sum(), inputs=(x, y))
        torch.autograd.backward(out.sum(), inputs=(x, edge_y))
        torch.autograd.backward(out.sum(), inputs=(edge_x, edge_y))

    def test_grad_fn_input_metadata(self):
        # [FUNCTION DOC] Check the private input metadata of grad_fn nodes: shape and dtype for a
        # mul node, one entry per output for var_mean, symbolic sizes and nested-tensor flags for
        # a jagged nested tensor clone, and shape for a custom Function node.
        x = torch.rand(2, requires_grad=True, dtype=torch.float32)
        y = torch.rand(2, requires_grad=True, dtype=torch.float32)
        z = x * y
        z_metadata = z.grad_fn._input_metadata[0]
        self.assertEqual(z_metadata.shape, (2,))
        self.assertEqual(z_metadata.dtype, torch.float32)

        # Multiple outputs
        b = torch.rand(3, 3, requires_grad=True)
        var, _ = torch.var_mean(b, dim=0)

        metadata_0 = var.grad_fn._input_metadata[0]
        metadata_1 = var.grad_fn._input_metadata[1]
        self.assertEqual(metadata_0.shape, (3,))
        self.assertEqual(metadata_1.shape, (3,))

        # Preserves symints
        nt = torch.nested.nested_tensor(
            [torch.randn(3, 2), torch.randn(2, 2)],
            layout=torch.jagged,
            requires_grad=True,
        )

        nt_metadata = nt.clone().grad_fn._input_metadata[0]

        self.assertIsInstance(nt_metadata.shape[1], torch.SymInt)
        self.assertEqual(nt_metadata.shape, nt.shape)
        self.assertTrue(nt_metadata.is_nested_tensor)
        self.assertFalse(nt_metadata.is_cpp_nested_tensor)
        self.assertEqual(nt_metadata.dtype, nt.dtype)

        class Test(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity custom Function used to check input metadata of
            # its node.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return grad_output

        x = torch.randn(3, 3, requires_grad=True)
        x = Test.apply(x)
        metadata = x.grad_fn._input_metadata[0]
        self.assertEqual(metadata.shape, (3, 3))

    def test_gradient_edge_output(self):
        # [FUNCTION DOC] Check get_gradient_edge on intermediate tensors and grad or backward with
        # GradientEdge outputs. Two-step and one-step gradients agree; errors for non-scalar
        # implicit grad, wrong grad_outputs shape and wrong dtype; backward with single, tuple and
        # mixed edge and tensor outputs.
        x = torch.tensor([1.0, 2.0], requires_grad=True)

        def fn(x, reduce=True):
            # [FUNCTION DOC] HELPER: Compute a chain of elementwise ops on x, optionally reduced
            # to a scalar, and return the final scalar and the gradient edge of the intermediate.
            tmp = x.sin().cos()
            if reduce:
                tmp = tmp.sum()
            out = tmp.exp().clone().sin().sum()
            tmp_edge = torch.autograd.graph.get_gradient_edge(tmp)
            return out, tmp_edge

        # Compute fn backward in two steps
        out, tmp_edge = fn(x)
        (tmp_grad,) = torch.autograd.grad(out, (tmp_edge,))

        (x_grad,) = torch.autograd.grad(tmp_edge, (x,), grad_outputs=(tmp_grad,))

        # Compare with as if we did it in one go.
        out, _ = fn(x)
        (x_grad_ref,) = torch.autograd.grad(out, (x,))
        self.assertEqual(x_grad, x_grad_ref)

        # Incorrect case: grad_outputs not passed/implicitly None and output is
        # not a scalar
        out, tmp_edge = fn(x, reduce=False)
        with self.assertRaisesRegex(
            RuntimeError, "grad can be implicitly created only for scalar output"
        ):
            torch.autograd.grad(tmp_edge, (x,))

        # grad_outputs is None, and output is a scalar is fine
        out, tmp_edge = fn(x, reduce=True)
        torch.autograd.grad(tmp_edge, (x,))

        # Incorrect case: grad_outputs wrong size
        out, tmp_edge = fn(x)
        with self.assertRaisesRegex(RuntimeError, "Mismatch in shape"):
            torch.autograd.grad(
                tmp_edge, (x,), grad_outputs=torch.tensor([1.0, 2.0, 3.0, 4.0])
            )

        # Incorrect case: wrong dtype
        out, tmp_edge = fn(x)
        (tmp_grad,) = torch.autograd.grad(out, (tmp_edge,))
        with self.assertRaisesRegex(RuntimeError, "required to have the same dtype"):
            torch.autograd.grad(
                tmp_edge,
                (x,),
                grad_outputs=torch.rand_like(tmp_grad, dtype=torch.complex64),
            )

        # Run with .backward() and compare with .grad()
        out, tmp_edge = fn(x)
        torch.autograd.backward(tmp_edge, retain_graph=True)
        (x_grad_ref,) = torch.autograd.grad(tmp_edge, (x,), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # Pass a tuple of GradientEdges
        x.grad = None
        torch.autograd.backward((tmp_edge,), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # Mixing GradientEdge and Tensors
        out1, tmp_edge1 = fn(x)
        out2, tmp_edge2 = fn(x)
        (x_grad_ref,) = torch.autograd.grad((tmp_edge1, out2), (x,), retain_graph=True)
        x.grad = None
        torch.autograd.backward((tmp_edge1, out2), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # .backward(): wrong shape
        out, tmp_edge = fn(x)
        with self.assertRaisesRegex(RuntimeError, "Mismatch in shape"):
            torch.autograd.backward(
                tmp_edge, inputs=(x,), grad_tensors=torch.tensor([1.0, 2.0, 3.0, 4.0])
            )

    def test_gradient_edge_graph_ownership(self):
        # [FUNCTION DOC] Check that a gradient edge keeps its graph alive: backward through the
        # edge works twice even after deleting the output, for both a C++ node and a Python custom
        # Function node.
        # Ensure we own the graph properly
        class Clone(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function cloning in forward and backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone of the input.
                return x.clone()

            @staticmethod
            def backward(ctx, gX):
                # [FUNCTION DOC] HELPER: Return a clone of the gradient.
                return gX.clone()

        inp = torch.rand(1, requires_grad=True).clone()

        # C++ Node
        out = inp.clone()
        edge = torch.autograd.graph.get_gradient_edge(out)
        torch.autograd.backward(edge)
        del out
        torch.autograd.backward(edge)

        # python Node
        out = Clone.apply(inp)
        edge = torch.autograd.graph.get_gradient_edge(out)
        torch.autograd.backward(edge)
        del out
        torch.autograd.backward(edge)

    def test_grad_nonleaf(self):
        # [FUNCTION DOC] Check autograd.grad with respect to a non-leaf tensor in a gradient
        # ascent loop: grad attributes stay None during the loop, the objective increases, and a
        # final backward fills leaf grads.
        x_init = torch.randn(2, 2, requires_grad=True)
        x = x_init
        y = torch.randn(2, 2, requires_grad=True)
        grad_output = torch.ones(2, 2)

        def fn(x):
            # [FUNCTION DOC] HELPER: Elementwise quadratic function of x involving the outer
            # tensor y.
            return x**2 + y * x + y**2

        for _ in range(5):
            (grad_x,) = torch.autograd.grad(
                fn(x), x, grad_outputs=grad_output, create_graph=True
            )

            grad_x_expected = 2 * x + y
            self.assertIsNone(y.grad)
            self.assertIsNone(x.grad)
            self.assertEqual(grad_x, grad_x_expected)

            x = x + 0.05 * grad_x

        val_init = fn(x_init).sum()
        val_final = fn(x).sum()
        self.assertGreater(val_final, val_init)

        x.backward(grad_output)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(x_init.grad)

    def test_grad_nonleaf_many_outputs(self):
        # [FUNCTION DOC] Check grad of two outputs of one chunk node while only one callback can
        # be captured; a tensor hook on the base must not run and the grad attribute of the base
        # stays None.
        # This checks an edge case for function callbacks
        # We want to capture two grads of a function, but can only
        # register a single callback.
        x = torch.randn(4, 2, requires_grad=True)
        a, b = x.chunk(2)

        def hook(*grads):
            # [FUNCTION DOC] HELPER: Hook that records it was called.
            hook_called[0] = True

        hook_called = [False]
        x.register_hook(hook)

        go = torch.randn(2, 2)
        grad_a, grad_b = torch.autograd.grad(
            (a + 2 * b), [a, b], grad_outputs=go, create_graph=True
        )

        self.assertEqual(grad_a, go)
        self.assertEqual(grad_b, go * 2)
        self.assertFalse(hook_called[0])
        self.assertIsNone(x.grad)

    def test_grad_nonleaf_register_hook(self):
        # [FUNCTION DOC] Check register_hook on a non-leaf unbind output: the hook records its
        # gradient on backward of that output, and backward through other outputs does not fire it
        # or crash; grads accumulate into the base.
        # This checks an edge case for register_hook.
        # We want to capture grad of a nonleaf tensor,
        # but avoid segfault during backward of other nonleaf tensors
        x = torch.randn(5, requires_grad=True)
        x_list = x.unbind()

        x0 = x_list[0]
        hook_results = [None]

        def hook(grad):
            # [FUNCTION DOC] HELPER: Hook storing the received gradient in an outer list.
            hook_results[0] = grad

        x0.register_hook(hook)

        x_list[0].backward()
        self.assertEqual(hook_results[0], torch.tensor(1.0))
        expected_grad = torch.tensor([1.0, 0, 0, 0, 0])
        self.assertEqual(x.grad, expected_grad)
        self.assertIsNone(x_list[0].grad)

        for i in range(1, 5, 1):
            x_list[i].backward()
            self.assertEqual(hook_results[0], None)
            expected_grad[i] = 1.0
            self.assertEqual(x.grad, expected_grad)
            self.assertIsNone(x_list[i].grad)

    def test_grad_materialize_grads(self):
        # [FUNCTION DOC] Check grad with materialize_grads: a higher derivative that is unused
        # gives None without it and zero with it, and passing allow_unused as False together with
        # materialize_grads raises ValueError.
        x = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)
        y = x * a
        dydx = torch.autograd.grad(y, x, create_graph=True)
        d2ydx2_none = torch.autograd.grad(dydx, x, create_graph=True, allow_unused=True)
        d2ydx2 = torch.autograd.grad(
            dydx, x, create_graph=True, allow_unused=True, materialize_grads=True
        )
        # `allow_unused` set to True implicitly
        d3ydx3 = torch.autograd.grad(d2ydx2, x, materialize_grads=True)
        self.assertIsNone(d2ydx2_none[0])
        self.assertEqual(d2ydx2[0].item(), 0)
        self.assertEqual(d3ydx3[0].item(), 0)
        with self.assertRaisesRegex(
            ValueError, "Expected allow_unused to be True or not passed when"
        ):
            torch.autograd.grad(y, x, allow_unused=False, materialize_grads=True)

    def test_post_accumulate_grad_hook_on_non_leaf(self):
        # [FUNCTION DOC] Check registering a post-accumulate-grad hook on a non-leaf tensor
        # raises RuntimeError.
        def hook(tensor):
            # [FUNCTION DOC] HELPER: Hook that subtracts one in place from the tensor.
            tensor.sub_(1.0)

        leaf = torch.rand(3, requires_grad=True)
        non_leaf = 2.0 * leaf

        with self.assertRaisesRegex(
            RuntimeError,
            "post accumulate grad hooks cannot be registered on non-leaf tensors",
        ):
            non_leaf.register_post_accumulate_grad_hook(hook)

    def test_post_accumulate_grad_hook_multiple_hooks(self):
        # [FUNCTION DOC] Check multiple post-accumulate-grad hooks on one tensor both run in
        # registration order: the first subtracts the grad, the second multiplies by four.
        def hook1(tensor):
            # [FUNCTION DOC] HELPER: Hook subtracting the tensor grad from the tensor in place.
            tensor.sub_(tensor.grad)

        def hook2(tensor):
            # [FUNCTION DOC] HELPER: Hook multiplying the tensor by four in place.
            tensor.mul_(4.0)

        tensor = torch.rand(3, requires_grad=True)
        tensor_ref = tensor.detach().clone()
        tensor.register_post_accumulate_grad_hook(hook1)
        tensor.register_post_accumulate_grad_hook(hook2)
        sum = tensor.sum()
        sum.backward()
        # both hooks should be called, in order
        self.assertEqual(4.0 * (tensor_ref - 1.0), tensor)

    def test_post_accumulate_grad_hook_multiple_tensors(self):
        # [FUNCTION DOC] Check the same post-accumulate-grad hook registered on two tensors
        # updates both when their own backward runs.
        def hook(tensor):
            # [FUNCTION DOC] HELPER: Hook subtracting the grad from the tensor in place.
            tensor.sub_(tensor.grad)

        tensor1 = torch.rand(3, requires_grad=True)
        tensor1_ref = tensor1.detach().clone()
        tensor2 = torch.rand(5, requires_grad=True)
        tensor2_ref = tensor2.detach().clone()
        tensor1.register_post_accumulate_grad_hook(hook)
        tensor2.register_post_accumulate_grad_hook(hook)
        tensor1.sum().backward()
        tensor2.sum().backward()
        # both tensors should have been modified
        self.assertEqual(tensor1_ref - 1.0, tensor1)
        self.assertEqual(tensor2_ref - 1.0, tensor2)

    def test_post_accumulate_grad_hook_returns_not_None(self):
        # [FUNCTION DOC] Check that a post-accumulate-grad hook returning a value makes backward
        # raise RuntimeError saying hooks should return None.
        def bad_hook(tensor):
            # [FUNCTION DOC] HELPER: Hook that wrongly returns the tensor grad.
            return tensor.grad

        tensor = torch.rand(2, 3, requires_grad=True)
        tensor.register_post_accumulate_grad_hook(bad_hook)
        # should error!
        with self.assertRaisesRegex(RuntimeError, "hooks should return None."):
            tensor.sum().backward()

    def test_post_accumulate_grad_hook_e2e(self):
        # [FUNCTION DOC] End-to-end check of optimizer steps inside post-accumulate-grad hooks: a
        # model updated by per-parameter Adam hooks during backward matches a copy updated by an
        # ordinary Adam, and after removing the hook handles the hooked model stops updating.
        def setup_optim_in_bwd(model):
            # [FUNCTION DOC] HELPER: Give each parameter its own Adam optimizer stepped by a
            # post-accumulate-grad hook; return the hook handles.
            optims = {}
            handles = []

            def optim_step_hook(param):
                # [FUNCTION DOC] HELPER: Hook that steps and zeroes the optimizer belonging to
                # the parameter.
                optims[param].step()
                optims[param].zero_grad()

            for p in model.parameters():
                optims[p] = torch.optim.Adam([p])
                handles.append(p.register_post_accumulate_grad_hook(optim_step_hook))

            return handles

        model = torch.nn.Linear(3, 2)
        input = torch.rand(2, 3)
        handles = setup_optim_in_bwd(model)

        # make a copy for reference
        model_copy = deepcopy(model)
        optim_copy = torch.optim.Adam(model_copy.parameters())

        iters = 5

        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()

            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()

        params_copy = []  # freeze a copy of the params to compare later
        for p_reference, p in zip(model_copy.parameters(), model.parameters()):
            self.assertEqual(p_reference, p)
            params_copy.append(p_reference.detach().clone())

        # After removing the handle, the model should no longer update.
        for h in handles:
            h.remove()

        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()

            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()

        for p_static, p_reference, p in zip(
            params_copy, model_copy.parameters(), model.parameters()
        ):
            self.assertEqual(p_static, p)
            self.assertNotEqual(p_reference, p)

    def test_post_accumulate_grad_hook_gets_cleaned_up(self):
        # [FUNCTION DOC] Check that closed-over objects of a post-accumulate-grad hook are kept
        # alive by the tensor and freed when the tensor is deleted, using a weak reference with
        # garbage collection disabled.
        def fun_stuff_with_hook():
            # [FUNCTION DOC] HELPER: Register a hook that captures a tensor, run backward, and
            # return the tensor and a weak reference to the captured object.
            thing_to_put_in_hook = torch.rand(3)

            def hook(tensor):
                # [FUNCTION DOC] HELPER: Hook subtracting the grad and adding the captured tensor
                # in place.
                tensor.sub_(tensor.grad)
                tensor.add_(thing_to_put_in_hook)

            tensor = torch.rand(3, requires_grad=True)
            tensor.register_post_accumulate_grad_hook(hook)
            tensor.sum().backward()
            ref = weakref.ref(thing_to_put_in_hook)
            gc.collect()
            return tensor, ref

        with disable_gc():
            tensor, ref = fun_stuff_with_hook()
            self.assertIsNotNone(
                ref()
            )  # thing_to_put_in_hook should be kept alive by tensor

            del tensor
            gc.collect()
            self.assertIsNone(ref())  # thing_to_put_in_hook should be cleaned

    def test_post_accumulate_grad_hook_ordering(self):
        # [FUNCTION DOC] Check the ordering of hooks around gradient accumulation: tensor pre
        # hook, AccumulateGrad node pre hook, post-accumulate-grad hook, AccumulateGrad node post
        # hook, with result 3 for each element.
        tensor = torch.rand(3, requires_grad=True)

        def pre_hook(grad):
            # [FUNCTION DOC] HELPER: Tensor hook subtracting 2 from the gradient.
            return grad.sub(2.0)

        def acc_grad_node_pre_hook(grad_out):
            # [FUNCTION DOC] HELPER: Node pre hook dividing the first gradient by 5 and returning
            # it in a tuple.
            return (grad_out[0].div(5.0),)

        def post_acc_grad_hook(tensor):
            # [FUNCTION DOC] HELPER: Post-accumulate hook adding 0.5 to the tensor grad in place.
            tensor.grad.add_(0.5)

        def acc_grad_node_post_hook(grad_in, grad_out):
            # [FUNCTION DOC] HELPER: Node post hook that replaces the tensor grad with ten times
            # the first output gradient.
            tensor.grad = grad_out[0].mul(10)

        acc_grad = tensor.view_as(tensor).grad_fn.next_functions[0][0]
        tensor.register_hook(pre_hook)
        acc_grad.register_prehook(acc_grad_node_pre_hook)
        tensor.register_post_accumulate_grad_hook(post_acc_grad_hook)
        acc_grad.register_hook(acc_grad_node_post_hook)
        tensor.sum().backward()

        # the hooks should run in the order of:
        #   1. tensor prehook
        #   2. acc_grad prehook
        #   3. tensor post acc_grad hook
        #   4. acc_grad posthook
        # so that would be ((1 - 2) / 5 + 0.5) * 10 = 3
        self.assertEqual(torch.tensor([3.0, 3.0, 3.0]), tensor.grad)

    def test_hook_with_no_name(self):
        # [FUNCTION DOC] Check that a callable object without a name attribute can be registered
        # as a tensor hook and backward runs.
        # Create a hook that do not have a __name__ attribute
        class MyHookClass:
            # [CLASS DOC] HELPER: Hook class (no function name) returning a clone of the gradient.
            def __call__(self, grad):
                # [FUNCTION DOC] HELPER: Return a clone of the gradient.
                return grad.clone()

        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()
        # Should run fine

    def test_prehook_ordering(self):
        # [FUNCTION DOC] Check the order of tensor hooks versus grad_fn pre hooks and
        # AccumulateGrad pre hooks, through backward, grad and backward with inputs, using a log
        # of hook ids; also that retain_grad sees the result of hooks registered before it.
        # Hooks registered to tensor are ordered before those
        # that are registered to grad_fn
        log = []

        def hook1(g):
            # [FUNCTION DOC] HELPER: Tensor hook logging 1 and tripling the gradient.
            log.append(1)
            return g * 3

        def hook2(gs):
            # [FUNCTION DOC] HELPER: Node pre hook logging 2 and doubling every gradient.
            log.append(2)
            return tuple(g * 2 for g in gs)

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()

        b.grad_fn.register_prehook(hook2)
        b.register_hook(hook1)
        b.grad_fn.register_prehook(hook2)

        acc = b.grad_fn.next_functions[0][0]
        a.register_hook(hook1)
        acc.register_prehook(hook2)
        a.register_hook(hook1)

        b.sum().backward(retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1, 2])

        # grad also runs hooks on accumulate grad nodes, even though
        # the accumulate grad nodes are not actually executed
        log = []
        torch.autograd.grad(b.sum(), inputs=(a,), retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1])

        log = []
        b.sum().backward(inputs=(b,))
        self.assertEqual(log, [1, 2, 2])
        # retains_grad hooks would not observe modifications by all pre hooks
        # because they are executed after
        self.assertEqual(b.grad.item(), 3)

    def test_retains_grad_can_always_observe_tensor_prehook(self):
        # [FUNCTION DOC] Check that retains_grad always observes tensor pre hooks, including ones
        # registered after retain_grad: gradient 4 in one setup and 2 in another.
        def tensor_prehook(g):
            # [FUNCTION DOC] HELPER: Tensor hook doubling the gradient.
            return g * 2

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.register_hook(tensor_prehook)
        b.retain_grad()
        b.register_hook(tensor_prehook)

        b.clone().backward()
        self.assertEqual(b.grad.item(), 4)

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.retain_grad()
        b.register_hook(tensor_prehook)

        b.clone().backward()
        self.assertEqual(b.grad.item(), 2)

    def test_accumulate_grad_posthooks_can_observe_tensor_prehook(self):
        # [FUNCTION DOC] Check that post hooks and pre hooks on an AccumulateGrad node see the
        # gradient already modified by a tensor pre hook.
        # Post hooks on accumulate should be able to observe changes to
        # grad made by tensor prehooks
        a = torch.tensor(1.0, requires_grad=True)

        def tensor_prehook(g):
            # [FUNCTION DOC] HELPER: Tensor hook doubling the gradient.
            return g * 2

        def posthook(gO, gI):
            # [FUNCTION DOC] HELPER: Node post hook asserting the grad input is twice a and there
            # are no grad outputs.
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gO), 0)

        def prehook(gI):
            # [FUNCTION DOC] HELPER: Node pre hook asserting one gradient equal to twice a.
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gI), 1)

        b = a.clone()
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        a.register_hook(tensor_prehook)

        b.backward()

    def test_accumulate_grad_posthooks_should_not_execute(self):
        # [FUNCTION DOC] Check hooks of AccumulateGrad nodes for inputs not requested via backward
        # inputs are not executed; hooks that always raise are registered and must not fire.
        def tensor_prehook(g):
            # [FUNCTION DOC] HELPER: Hook that raises RuntimeError.
            raise RuntimeError

        def posthook(gO, gI):
            # [FUNCTION DOC] HELPER: Hook that raises RuntimeError.
            raise RuntimeError

        a = torch.tensor(1.0, requires_grad=True)
        a.register_hook(tensor_prehook)
        b = torch.tensor(1.0, requires_grad=True)
        c = a.clone()
        acc = c.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)

        out = a + b + c
        out.sum().backward(inputs=[b])

    def test_hook_edge_case_when_called_with_grad(self):
        # [FUNCTION DOC] Check which hooks run under grad versus backward: grad runs tensor hooks
        # only for the requested next node and no node pre or post hooks unless those nodes
        # execute, while backward runs them all; counts are asserted for each call form.
        # grad executes the tensor hooks of the next node but not
        # grad_fn pre hooks or the post hooks
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        c = b * 2

        tensor_hook_count = [0]
        prehook_count = [0]
        posthook_count = [0]

        def reset_counts():
            # [FUNCTION DOC] HELPER: Reset the three hook counter lists to fresh zeroed lists.
            nonlocal tensor_hook_count, prehook_count, posthook_count
            tensor_hook_count = [0]
            prehook_count = [0]
            posthook_count = [0]

        def tensor_prehook(g):
            # [FUNCTION DOC] HELPER: Tensor hook incrementing the tensor hook counter.
            tensor_hook_count[0] += 1

        def prehook(g):
            # [FUNCTION DOC] HELPER: Node pre hook incrementing the pre hook counter.
            prehook_count[0] += 1

        def posthook(gI, gO):
            # [FUNCTION DOC] HELPER: Node post hook incrementing the post hook counter.
            posthook_count[0] += 1

        a.register_hook(tensor_prehook)
        b.register_hook(tensor_prehook)
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        b.grad_fn.register_hook(posthook)
        b.grad_fn.register_prehook(prehook)

        torch.autograd.grad(c, inputs=(b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 1)
        self.assertEqual(posthook_count[0], 0)
        self.assertEqual(prehook_count[0], 0)
        reset_counts()

        torch.autograd.grad(c, inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 1)
        self.assertEqual(prehook_count[0], 1)
        reset_counts()

        c.backward(retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)
        reset_counts()

        c.backward(inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)

    def test_sharded_grad(self):
        # [FUNCTION DOC] Check that the gradient of intermediates can be computed in shards with
        # grad, then the rest via backward given those gradients, giving the same final leaf grads
        # as the analytic formula.
        leaves = [torch.zeros(5, 5, requires_grad=True) for _ in range(10)]
        intermediates = [l * i + l * l for i, l in enumerate(leaves)]
        loss = sum(v * i for i, v in enumerate(intermediates)).sum()

        # define a helper for dividing intermediates into groups
        def group(l, group_size):
            # [FUNCTION DOC] HELPER: Generator splitting a list into consecutive chunks of the
            # given size.
            return (l[i : i + group_size] for i in range(0, len(l), group_size))

        # Compute the d loss / d intermediates in chunks of shard_size
        shard_size = 2
        d_intermediates = [
            d_i
            for intermediates_batch in group(intermediates, shard_size)
            for d_i in torch.autograd.grad(loss, intermediates_batch)
        ]
        # Compute rest of backward pass
        torch.autograd.backward(intermediates, d_intermediates)

        for i, l in enumerate(leaves):
            self.assertEqual(l.grad, i * i * (1 + l))

    def test_backward_badcalls(self):
        # [FUNCTION DOC] Check backward on a tensor not requiring grad raises RuntimeError.
        x = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            x.backward()

    def test_grad_badcalls(self):
        # [FUNCTION DOC] Check grad raises RuntimeError when the output or the input does not
        # require grad, and succeeds once it does.
        x = torch.ones(1)
        y = x**2
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(x, y)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(y, x)

        x = torch.ones(1, requires_grad=True)
        y = x**2
        torch.autograd.grad(y, x)  # this should succeed now

    def test_grad_empty_inputs(self):
        # [FUNCTION DOC] Check grad with an empty inputs list raises RuntimeError saying it cannot
        # be empty.
        x = torch.tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "cannot be empty"):
            torch.autograd.grad(2 * x, [], grad_outputs=torch.tensor([1.0]))

    def test_grad_fn_badcalls(self):
        # [FUNCTION DOC] Check calling a grad_fn node with the wrong number of arguments raises
        # TypeError, and a correct call succeeds.
        error_regex = "expected .* arguments, got .* instead"
        x = torch.ones(1, requires_grad=True)
        y = x**2
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn(x.detach(), x.detach())  # too many
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn()  # too few

        y.grad_fn(x.detach())  # this should succeed

    def test_grad_unreachable(self):
        # [FUNCTION DOC] Check grad with unreachable inputs: allow_unused yields None for them,
        # and without allow_unused a RuntimeError asks to set allow_unused.
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # Make sure x and y have grad accumulators allocated
        z = x * 2
        w = y * 2

        grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)

        # This is slightly different than the case above, because z doesn't even
        # have a grad accumulator allocated.
        z = torch.ones(1, requires_grad=True)
        grad_x, grad_z = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)

        # allow_unused=False, but grads contains None inside, should throw
        with self.assertRaisesRegex(RuntimeError, "Set allow_unused=True"):
            grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=False)

    def test_grad_unreachable_discovery(self):
        # [FUNCTION DOC] Check unreachable inputs do not trigger execution of unrelated
        # custom nodes (fail if the custom backward runs), and backward with inputs
        # implicitly allows unused.
        # Test that certain nodes are not erroneously executed when an input
        # is unreachable. See #39784
        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function whose backward fails the test if executed.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Fail the test with a message that the node should not
                # be executed.
                self.fail("This node should not be executed!")

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        (gY,) = torch.autograd.grad(x, (y,), allow_unused=True)
        self.assertIsNone(gY)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        z = torch.randn(1, requires_grad=True)
        (gY, gZ) = torch.autograd.grad(x + z, (y, z), allow_unused=True)
        self.assertIsNone(gY)
        self.assertIsNotNone(gZ)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        torch.autograd.backward(x, inputs=(y,))  # allow_unused is implicitly True!
        self.assertIsNone(y.grad)

    def test_grad_batched_grad(self):
        # [FUNCTION DOC] Check grad with is_grads_batched: batched gradients along a new first
        # dimension give batched results for tensor and scalar outputs, and a mismatched leading
        # shape raises RuntimeError.
        x = torch.randn(2, 2, requires_grad=True)

        out = x.clone()  # Size([2, 2])
        batched_grad = (
            torch.arange(3).expand(2, 2, 3).transpose(0, 2)
        )  # Size([3, 2, 2])
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # Detect shape mismatch
        grad_out = torch.ones(2, 2)
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )

        # Scalar outputs
        out = x.sum()  # Size([])
        batched_grad = torch.arange(3)  # Size([3])
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # We consider scalar and sized-1 to be a mismatch. This is consistent with current non-batched behavior.
        grad_out = torch.ones(2).unsqueeze(1)
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )

    def test_hooks(self):
        # [FUNCTION DOC] Check tensor hooks: counts for hooks on a leaf and on an intermediate,
        # removal of a hook through its handle, and hooks that modify the gradient for the
        # intermediate and then the leaf.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)

        counter = [0]

        def bw_hook(inc, grad):
            # [FUNCTION DOC] HELPER: Hook asserting the gradient is a tensor and adding the
            # increment to the counter.
            self.assertIsInstance(grad, torch.Tensor)
            counter[0] += inc

        z = x**2 + x * 2 + x * y + y
        x.register_hook(lambda *args: bw_hook(0, *args))
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 1)

        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 4)

        test2.remove()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            # [FUNCTION DOC] HELPER: Hook returning the gradient doubled.
            return grad.mul(2)

        test.remove()
        z.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad, (x + 1) * 2)

        y.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad, (x + 1) * 4)

    def _get_mul2(self, use_custom_function):
        # [FUNCTION DOC] HELPER: Return a doubling callable: either the apply of a custom Function
        # or a plain lambda, chosen by the flag.
        if use_custom_function:

            class Mul2(Function):
                # [CLASS DOC] HELPER: Custom Function doubling its input and its gradient.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Return the input times 2.
                    return x * 2

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Return the gradient times 2.
                    return gO * 2

            return Mul2.apply
        else:
            return lambda x: x * 2

    def test_grad_fn_prehooks(self):
        # [FUNCTION DOC] Check grad_fn hooks for both custom and builtin nodes: two post hooks and
        # three effective pre hooks (some pre hooks return None or the input unchanged); pre hooks
        # modify the gradient seen by post hooks; returning None keeps the gradient.
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)

            a = torch.tensor([1.0], requires_grad=True)
            b = mul2(a)

            post_counter = [0]
            pre_counter = [0]

            def posthook(grad_input, grad_output):
                # [FUNCTION DOC] HELPER: Node post hook asserting three pre hooks ran and expected
                # values, then counting and returning the grad input.
                self.assertEqual(pre_counter[0], 3)
                self.assertTrue(torch.allclose(grad_output[0], torch.ones(1) * 8))
                self.assertTrue(torch.allclose(grad_input[0], torch.ones(1) * 16))
                post_counter[0] += 1
                return grad_input

            def prehook(grad_output):
                # [FUNCTION DOC] HELPER: Node pre hook incrementing the counter and returning a
                # tuple with the first gradient doubled.
                pre_counter[0] += 1
                return (grad_output[0] * 2,)

            # register posthook x 2
            b.grad_fn.register_hook(posthook)
            b.grad_fn.register_hook(posthook)
            # register prehook x 3
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: None)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: x)
            b.grad_fn.register_prehook(lambda x: None)

            b.sum().backward()

            self.assertEqual(post_counter[0], 2)
            self.assertEqual(pre_counter[0], 3)

            # Return None
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)

            def prehook(grad_output):
                # [FUNCTION DOC] HELPER: Node pre hook incrementing the counter and
                # returning None.
                pre_counter[0] += 1
                return None

            b.grad_fn.register_prehook(prehook)
            b.sum().backward()
            self.assertEqual(pre_counter[0], 4)
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))

    def test_grad_fn_prehooks_multiple_outputs(self):
        # [FUNCTION DOC] Check node pre hooks on a node with multiple outputs, for var_mean and
        # for a custom Function with a non-tensor output whose gradient is None.
        # Compute gradients without hooks
        b = torch.rand(3, 3, requires_grad=True)
        var, mean = torch.var_mean(b, dim=0)
        (var + mean).sum().backward()

        # Compute gradients with hooks
        a = b.detach().requires_grad_()
        counter = [0]

        def prehook(grad_output):
            # [FUNCTION DOC] HELPER: Pre hook doubling both gradients of var_mean and
            # counting calls.
            gvar, gmean = grad_output
            counter[0] += 1
            return (gvar * 2, gmean * 2)

        var, mean = torch.var_mean(a, dim=0)
        mean.grad_fn.register_prehook(prehook)
        (var + mean).sum().backward()

        self.assertEqual(counter[0], 1)
        # Compare
        self.assertTrue(torch.allclose(a.grad, b.grad * 2))

        # Test with custom Function
        class DoubleMul2(Function):
            # [CLASS DOC] HELPER: Custom Function with three outputs, the middle one a
            # non-tensor scalar.
            @staticmethod
            def forward(ctx, x, a, y):
                # [FUNCTION DOC] HELPER: Store the scale on ctx and return two scaled tensors and
                # the scalar.
                ctx.a = a
                return a * x * 2, a, a * y * 2

            @staticmethod
            def backward(ctx, g1, _a, g2):
                # [FUNCTION DOC] HELPER: Return gradients for the two tensors scaled by the stored
                # value and None for the scalar.
                return ctx.a * g1 * 2, None, ctx.a * g2 * 2

        counter = [0]

        def prehook(grad_output):
            # [FUNCTION DOC] HELPER: Pre hook asserting the middle gradient is None, counting, and
            # returning doubled first and third gradients.
            g1, ga, g2 = grad_output
            self.assertIsNone(ga)
            counter[0] += 1
            return (g1 * 2, None, g2 * 2)

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        k = 3
        c, _, d = DoubleMul2.apply(a, k, b)
        c.grad_fn.register_prehook(prehook)
        (c + d).sum().backward()

        self.assertEqual(counter[0], 1)
        self.assertTrue(torch.allclose(a.grad, torch.ones(1) * 4 * k))
        self.assertTrue(torch.allclose(b.grad, torch.ones(1) * 4 * k))

    def test_grad_fn_prehooks_remove_hooks(self):
        # [FUNCTION DOC] Check removing node pre hooks, both before backward and from inside
        # another hook during backward (including removing an already removed handle) for custom
        # and builtin nodes.
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)

            # Simply remove hooks

            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook(grad_output):
                # [FUNCTION DOC] HELPER: Pre hook counting calls and returning None.
                counter[0] += 1
                return None

            handle = b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            handle.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)

            # Remove hooks during backward
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook1(grad_output):
                # [FUNCTION DOC] HELPER: Pre hook that removes two other handles, one of them
                # already removed.
                handle2.remove()
                # Remove hook that is already removed is OK
                handle3.remove()
                return None

            def prehook2(grad_output):
                # [FUNCTION DOC] HELPER: Pre hook counting calls and returning None.
                counter[0] += 1
                return None

            # Hooks that registered first run first
            b.grad_fn.register_prehook(prehook1)
            handle2 = b.grad_fn.register_prehook(prehook2)
            handle3 = b.grad_fn.register_prehook(prehook2)
            handle3.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)

    def test_node_post_hook_registered_during_unpack_hook(self):
        # [FUNCTION DOC] Check that a node post hook registered from inside a saved-tensor unpack
        # hook gets inputs that are all None and is applied, halving the outputs so the gradient
        # grows by half.
        """
        Test that post hooks registered during one of the node's
        unpack hooks are properly restricted and will run properly.
        """
        test_case = self

        class RegisterPostNodeHook(torch.autograd.graph.saved_tensors_hooks):
            # [CLASS DOC] HELPER: Saved-tensors hooks context whose unpack hook registers a post
            # hook on the currently executing node.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Build the pack and unpack functions and pass them to the
                # base saved_tensors_hooks constructor.
                def pack_tensor(tensor: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Pack hook returning the tensor unchanged.
                    return tensor

                def unpack_tensor(tensor: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Unpack hook that registers a post hook halving
                    # outputs on the current autograd node, then returns the tensor.
                    node = torch._C._current_autograd_node()

                    def hook(outputs, inputs):
                        # [FUNCTION DOC] HELPER: Node post hook asserting inputs are all None and
                        # returning the outputs halved.
                        # Assert that inputs passed in are None
                        test_case.assertTrue(all(i is None for i in inputs))
                        halved_outputs = tuple(
                            o / 2.0 if o is not None else None for o in outputs
                        )
                        return halved_outputs

                    node.register_hook(hook)
                    return tensor

                super().__init__(pack_tensor, unpack_tensor)

        a = torch.rand(3, 3, requires_grad=True)

        def model():
            # [FUNCTION DOC] HELPER: Run var_mean on a, sum, and backward.
            var, mean = torch.var_mean(a, dim=0)
            loss = (var + mean).sum()
            loss.backward()

        model()
        ref_grad = a.grad.clone()

        with RegisterPostNodeHook():
            model()

        # Verify that the post hook got called and the grad propagation worked
        self.assertEqual(ref_grad / 2.0 + ref_grad, a.grad)

    def test_hooks_cpp(self):
        # [FUNCTION DOC] Check a tensor hook on the output of a C++ batch norm in eval mode
        # doubles the gradient and is called once.
        # Tests hooks for autograd function implemented in C++
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()

        counter = [0]

        def bw_hook(grad):
            # [FUNCTION DOC] HELPER: Hook counting calls and doubling the gradient.
            counter[0] += 1
            return grad * 2

        x = torch.ones(5, 5, dtype=torch.double, requires_grad=True)
        z = bn(x)
        z.register_hook(bw_hook)
        z.sum().backward()

        self.assertEqual(counter[0], 1, msg="bw_hook not called")
        self.assertEqual(
            x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-5, rtol=0
        )

    def test_hook_none(self):
        # [FUNCTION DOC] Check hooks on outputs of a custom Function that returns a non-grad
        # input: the hook receives a non-None gradient for the differentiable output.
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        class NoneGradientFunction(Function):
            # [CLASS DOC] HELPER: Function returning both inputs and asserting only the first
            # requires grad.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Raise AssertionError unless needs_input_grad is true for
                # the first input and false for the second; return both inputs.
                if not ctx.needs_input_grad[0]:
                    raise AssertionError("expected needs_input_grad[0] to be True")
                if ctx.needs_input_grad[1]:
                    raise AssertionError("expected needs_input_grad[1] to be False")
                return x, y

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                # [FUNCTION DOC] HELPER: Return the first gradient and None.
                return grad_x, None

        was_called = [False]

        def hook(grad):
            # [FUNCTION DOC] HELPER: Hook asserting the gradient is not None and recording it
            # was called.
            self.assertIsNotNone(grad)
            was_called[0] = True

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        rx, ry = NoneGradientFunction.apply(x, y)
        rx.register_hook(hook)
        ry.register_hook(hook)
        sum(rx, ry).sum().backward()
        self.assertTrue(was_called[0])

    @parametrize("num_inputs", [2, 25])
    def test_needs_input_grad_setter_roundtrip(self, num_inputs):
        # [FUNCTION DOC] Parametrized check that the needs_input_grad attribute of ctx can be
        # set to arbitrary values (list, None) and restored inside backward, for two and twenty
        # five inputs.
        sentinel = ([False, True],)

        class NeedsInputGradSetter(Function):
            # [CLASS DOC] HELPER: Function summing the first two of a variable number of inputs.
            @staticmethod
            def forward(ctx, *inputs):
                # [FUNCTION DOC] HELPER: Return the sum of the first two inputs.
                return inputs[0] + inputs[1]

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Check the original needs_input_grad, assign and read back
                # several replacement values, restore it, and return the gradient for the first
                # input only.
                original = ctx.needs_input_grad
                self.assertEqual(original, (True,) + (False,) * (num_inputs - 1))
                ctx.needs_input_grad = sentinel
                self.assertIs(ctx.needs_input_grad, sentinel)
                ctx.needs_input_grad = None
                self.assertIsNone(ctx.needs_input_grad)
                ctx.needs_input_grad = original
                self.assertIs(ctx.needs_input_grad, original)
                return (grad,) + (None,) * (num_inputs - 1)

        x = torch.randn((), requires_grad=True)
        inputs = (x,) + tuple(torch.randn(()) for _ in range(num_inputs - 1))
        NeedsInputGradSetter.apply(*inputs).backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_retain_grad(self):
        # [FUNCTION DOC] Check retain_grad can be called repeatedly on an intermediate, gradients
        # accumulate across backward calls, and it is a no-op on a leaf.
        input = torch.rand(1, 3, requires_grad=True)
        h1 = input * 3
        out = (h1 * h1).sum()

        # It should be possible to call retain_grad() multiple times
        h1.retain_grad()
        h1.retain_grad()

        # Gradient should be accumulated
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)

        with torch.no_grad():
            input.grad.zero_()
        # It should be a no-op for leaves
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    # NB: See test/cpp/api/autograd.cpp for more tests on the interaction between
    #     retains_grad and hooks in cpp
    def test_retain_grad_inplace(self):
        # [FUNCTION DOC] Check retain_grad together with in-place ops: the retained grad equals
        # one after in-place scaling, including in-place over a view, where hooks move to the
        # copy-slices node; and a base not requiring grad has nothing to move.
        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))

        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        # Inplace multiple times is OK
        a.mul_(2)
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # When in-place over view is done, the retains_grad hooks should be
        # moved from base's original grad_fn to the copyslices node.
        x = torch.tensor([1.0], requires_grad=True).clone()
        x.retain_grad()
        x_view = x[:]
        x_view *= 2
        x *= 2
        x.sum().backward()
        # The grad is 1, not 4, because we are computing grad wrt the latest
        # version of x.
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # If the base did not originally require grad, there should be no hook
        # to move. Make sure this case runs without error.
        x = torch.zeros(4)
        y = x.view(2, 2)
        y.add_(torch.randn(2, 2, requires_grad=True))

    def test_retains_grad_inplace_multiple_outputs(self):
        # [FUNCTION DOC] Check retain_grad followed by an in-place op on one output of a
        # multi-output node (custom Function and var_mean): the retained gradients of both outputs
        # equal the gradients computed with grad on a fresh copy.
        class DoubleMul(Function):
            # [CLASS DOC] HELPER: Custom Function returning two scaled copies of the input (by 2
            # and 3).
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input times 2 and times 3.
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, g1, g2):
                # [FUNCTION DOC] HELPER: Combine the two gradients with the same weights, 2 and 3,
                # into one input gradient.
                return g1 * 2 + g2 * 3

        var_mean = partial(torch.var_mean, dim=0)

        for fn in (DoubleMul.apply, var_mean):
            b = torch.rand(3, 3, requires_grad=True)
            var, mean = fn(b)
            var.retain_grad()
            mean.retain_grad()
            # node has two retains_grad hooks
            var.mul_(2)
            # the retain_grad hook multi-output node refers should now be a nullptr
            (var + mean).sum().backward()
            gvar = var.grad
            gmean = mean.grad

            a = b.detach().requires_grad_(True)
            var, mean = fn(a)
            var.mul_(2)
            out = (var + mean).sum()
            gvar_expected, gmean_expected = torch.autograd.grad(out, inputs=(var, mean))
            self.assertTrue(torch.allclose(gvar, gvar_expected))
            self.assertTrue(torch.allclose(gmean, gmean_expected))

    def test_retain_grad_inplace_over_view(self):
        # [FUNCTION DOC] Check retain_grad on two views of one base, then an in-place op on one
        # view: both views end up with retained gradient one, showing retain hooks moved to the
        # new grad_fn.
        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.retain_grad()
        view2.retain_grad()
        view.mul_(2)
        (view + view2).sum().backward()

        # The old grad_fn, slice, wouldn't be part of the graph during backward
        # so if the retains grad were not properly updated to the new grad_fn,
        # the grad would still be None
        self.assertEqual(view.grad, view2.grad)
        self.assertEqual(view.grad, torch.tensor([1.0]))

    def test_tensor_hooks_inplace(self):
        # [FUNCTION DOC] Check tensor hooks registered before and after an in-place op attach to
        # the right version of the tensor: counts and the gradient values seen by each hook, and
        # the final leaf gradient 8.
        # Check that the second hook gets registered to the new version of tensor
        count1 = [0]
        count2 = [0]

        def fn1(grad):
            # [FUNCTION DOC] HELPER: Hook counting calls, asserting the gradient is 4 and
            # returning it doubled.
            count1[0] += 1
            # x2 from mul, x2 from fn2
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        def fn2(grad):
            # [FUNCTION DOC] HELPER: Hook counting calls, asserting the gradient is 1 and
            # returning it doubled.
            count2[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))
            return grad * 2

        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn1)
        b.mul_(2)
        b.register_hook(fn2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(count2[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))

        count3 = [0]

        def fn3(grad):
            # [FUNCTION DOC] HELPER: Hook counting calls, asserting the gradient is 4 and
            # returning it doubled.
            count3[0] += 1
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn3)
        # Inplace multiple times is OK
        b.mul_(2)
        b.mul_(2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))

    def test_tensor_hooks_inplace_multiple_outputs(self):
        # [FUNCTION DOC] Check tensor hooks and in-place modification of one output of a
        # multi-output node (custom Function and var_mean): each hook fires once and sees
        # the expected gradient, with hooks registered after the in-place op attached to the
        # new version.
        class DoubleMul(Function):
            # [CLASS DOC] HELPER: Custom Function returning two scaled copies of the input (by 2
            # and 3).
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input times 2 and times 3.
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, g1, g2):
                # [FUNCTION DOC] HELPER: Combine the two gradients with weights 2 and 3.
                return g1 * 2 + g2 * 3

        var_mean = partial(torch.var_mean, dim=0)

        for fn in (DoubleMul.apply, var_mean):
            counts = [0, 0, 0]

            def fn0(grad):
                # [FUNCTION DOC] HELPER: Hook counting calls in slot zero and asserting the
                # gradient is twice ones.
                counts[0] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 2)

            def fn1(grad):
                # [FUNCTION DOC] HELPER: Hook counting calls in slot one and asserting the
                # gradient is three times ones.
                counts[1] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 3)

            def fn2(grad):
                # [FUNCTION DOC] HELPER: Hook counting calls in slot two and asserting the
                # gradient is ones.
                counts[2] += 1
                self.assertEqual(grad, torch.ones_like(out1))

            b = torch.rand(3, 3, requires_grad=True)
            out1, out2 = fn(b)
            h1 = out1.register_hook(fn0)
            h2 = out2.register_hook(fn1)
            # node refers to two hook dicts
            # out1 no longer no longer points to its old hook dict
            out1.mul_(2)
            # fn2 is registered to out1's new hook dict
            h3 = out1.register_hook(fn2)
            (out1 + out2 * 3).sum().backward()
            self.assertEqual(counts, [1, 1, 1])

            # Avoid leaking memory
            h1.remove()
            h2.remove()
            h3.remove()

    def test_tensor_hooks_inplace_over_view(self):
        # [FUNCTION DOC] Check that hooks registered on views before an in-place op on a view do
        # not fire afterwards; only a hook registered after the grad_fn of the other view is
        # refreshed fires, once.
        # There might be a better UX here, but this is the way it is now
        count = [0]

        def fn0(grad):
            # [FUNCTION DOC] HELPER: Hook that fails the test if called.
            self.fail()

        def fn1(grad):
            # [FUNCTION DOC] HELPER: Hook that fails the test if called.
            self.fail()

        def fn2(grad):
            # [FUNCTION DOC] HELPER: Hook counting calls and asserting the gradient equals one.
            count[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))

        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.register_hook(fn0)
        view2.register_hook(fn1)
        view.mul_(2)
        # We need to explicitly trigger an update to view to update its grad_fn
        view2.grad_fn
        view2.register_hook(fn2)
        (view + view2).sum().backward()
        # The hooks originally registered to view are not fired, one must explicitly
        # trigger an update to the view's grad_fn, and then register a new hook
        self.assertEqual(count[0], 1)

    def test_retain_grad_cycle(self):
        # [FUNCTION DOC] Check a retain_grad tensor does not create a reference cycle: the
        # intermediate is freed (weak reference expired) once the function returns, and backward
        # still works on the derived tensor.
        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            # [FUNCTION DOC] HELPER: Make an intermediate with retain_grad and return a derived
            # tensor and a weak reference to the intermediate.
            y = x * 2
            y.retain_grad()

            return y / 2, torch._C._WeakTensorRef(y)

        z, ref = run_test()
        self.assertTrue(ref.expired())
        z.sum().backward()

    def test_backward(self):
        # [FUNCTION DOC] Check backward with an explicit gradient on a leaf and on a composite
        # expression: gradients equal analytic values times the gradient argument.
        v = torch.randn(5, 5, requires_grad=True)
        x = torch.randn(5, 5, requires_grad=True)
        y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
        z = torch.randn(5, 5, requires_grad=True)
        grad_output = torch.randn(5, 5)

        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)

        a = x + (y * z) + 4 * z**2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z.pow(2) / y + 1
        y_grad = z - 4 * x * z.pow(2) / y.pow(2)
        z_grad = 8 * x * z / y + y
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_to_sparse_backward(self):
        # [FUNCTION DOC] Check backward through conversions between dense, COO, CSR, CSC, BSR and
        # BSC layouts: for each convertible pair the gradient has the source layout and equals the
        # dense gradient when densified.
        to_attr_names = (
            "to_dense",
            "to_sparse",
            "to_sparse_csr",
            "to_sparse_csc",
            "to_sparse_bsr",
            "to_sparse_bsc",
        )
        to_params = ((), (), (), (), (2,), (2,))
        to_attr_names_params = dict(zip(to_attr_names, to_params))

        def check_inversion_possible(
            t, layout1, layout1_params, layout2, layout2_params
        ):
            # [FUNCTION DOC] HELPER: Return whether converting the tensor to one layout and then
            # the other, in both orders, succeeds without RuntimeError.
            l = (layout1, layout2)
            p = (layout1_params, layout2_params)
            for l1, l2, p1, p2 in ((*l, *p), (*l[::-1], *p[::-1])):
                try:
                    to_l1 = getattr(t, l1)(*p1)
                    to_l2 = getattr(to_l1, l2)(*p2)
                except RuntimeError:
                    return False

            return True

        self_strided = torch.rand(4, 4, dtype=torch.double) + 1
        grad_strided = torch.rand(4, 4, dtype=torch.double) + 1

        for from_to_attr in to_attr_names:
            from_params = to_attr_names_params[from_to_attr]
            self_from = getattr(self_strided, from_to_attr)(
                *from_params
            ).requires_grad_(True)

            for to_to_attr in to_attr_names[1:]:
                to_params = to_attr_names_params[to_to_attr]

                if check_inversion_possible(
                    self_strided, from_to_attr, from_params, to_to_attr, to_params
                ):
                    self_to = getattr(self_from, to_to_attr)(*to_params)
                    grad_to = getattr(grad_strided, to_to_attr)(*to_params)

                    # No gradcheck support for BSR/BSC, so the grads are checked explicitly
                    grad_res = torch.autograd.grad(self_to, self_from, grad_to)[0]

                    self.assertEqual(grad_res.layout, self_from.layout)
                    self.assertEqual(grad_res.to_dense(), grad_strided)

    def test_sparse_mm_backward(self):
        # [FUNCTION DOC] Check backward of matrix multiply with sparse operands for every sparse
        # and requires-grad combination, comparing with the same computation on dense copies.
        size = (3, 3)

        mm_test_cases = product(*(([False, True],) * 4))

        for a_req_grad, a_is_sparse, b_req_grad, b_is_sparse in mm_test_cases:
            # We should only be testing cases with sparse inputs, and at least one
            # input needs to require grad so we can call a backward pass
            if not ((a_is_sparse or b_is_sparse) and (a_req_grad or b_req_grad)):
                continue
            a = torch.randn(size)
            if a_is_sparse:
                # detaching as `a` needs to be a leaf
                a = a.to_sparse().detach()
            b = torch.randn(size)
            if b_is_sparse:
                # detaching as `b` needs to be a leaf
                b = b.to_sparse().detach()

            a = a.requires_grad_(a_req_grad)
            b = b.requires_grad_(b_req_grad)

            r = a.mm(b)
            s = r.sum().backward()
            a_grad = None if a.grad is None else a.grad.detach().clone()
            b_grad = None if b.grad is None else b.grad.detach().clone()

            # Redo with only dense tensors
            a = (
                (a.to_dense() if a.is_sparse else a)
                .clone()
                .detach()
                .requires_grad_(a_req_grad)
            )
            b = (
                (b.to_dense() if b.is_sparse else b)
                .clone()
                .detach()
                .requires_grad_(b_req_grad)
            )

            r = a.mm(b)
            r.sum().backward()

            self.assertEqual(a_grad, a.grad)
            self.assertEqual(b_grad, b.grad)

    def test_multi_backward(self):
        # [FUNCTION DOC] Check backward with several roots and gradients accumulates correctly
        # into leaves, including a shared intermediate.
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q = torch.randn(5, 5, requires_grad=True)

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        q2 = q * 2
        z = x + y + q2
        c = a * b + q2
        grad_z = torch.randn(5, 5)
        grad_c = torch.randn(5, 5)
        torch.autograd.backward([z, c], [grad_z, grad_c])

        self.assertEqual(x.grad, grad_z)
        self.assertEqual(y.grad, grad_z)
        self.assertEqual(a.grad, grad_c * b)
        self.assertEqual(b.grad, grad_c * a)
        self.assertEqual(q.grad, (grad_c + grad_z) * 2)

    def test_multi_backward_no_grad(self):
        # [FUNCTION DOC] Check backward over a list including a tensor with no grad history
        # raises RuntimeError.
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=False)

        z = x + y
        q = y * 2

        # NB: we currently raise an exception if any arguments to backwards
        # have requires_grad=False and don't have a grad_fn. We may want to
        # relax that check to a warning.
        def call_backwards():
            # [FUNCTION DOC] HELPER: Call backward on two roots with ones gradients; one root has
            # no grad history.
            torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])

        self.assertRaises(RuntimeError, call_backwards)

    def test_backward_with_inputs(self):
        # [FUNCTION DOC] Check backward with inputs given as a list, a single tensor, a subset and
        # an empty list: only requested inputs get gradients, and an empty list raises
        # RuntimeError saying it cannot be empty.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        def fn():
            # [FUNCTION DOC] HELPER: Compute a quadratic function of x and y.
            return x**2 + y * x + y**2

        gradient = torch.ones(2, 2)
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        @torch.no_grad()
        def reset_grad():
            # [FUNCTION DOC] HELPER: Zero the gradients of x and y in place without
            # recording autograd.
            x.grad.zero_()
            y.grad.zero_()

        torch.autograd.backward(fn(), gradient, inputs=[x, y])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, y_grad_expected)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[x])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[y])
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=y)
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        self.assertRaisesRegex(
            RuntimeError,
            "cannot be empty",
            lambda: torch.autograd.backward(fn(), gradient, inputs=[]),
        )

    def test_grad_dict_inputs(self):
        # [FUNCTION DOC] Check grad accepts a dict of inputs and returns a dict with the same keys
        # and the right gradients.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2 + y * x + y**2

        inputs = {"x": x, "y": y}
        result = torch.autograd.grad(z.sum(), inputs)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"x", "y"})
        self.assertEqual(result["x"], 2 * x + y)
        self.assertEqual(result["y"], x + 2 * y)

    def test_grad_dict_inputs_ordered_dict(self):
        # [FUNCTION DOC] Check grad with an ordered mapping as inputs raises TypeError saying
        # inputs must be a dict.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2 + y * x + y**2

        inputs = OrderedDict([("x", x), ("y", y)])
        with self.assertRaisesRegex(TypeError, "must be a dict"):
            torch.autograd.grad(z.sum(), inputs)

    def test_grad_dict_inputs_materialize_grads(self):
        # [FUNCTION DOC] Check grad with dict inputs and materialize_grads returns zeros for the
        # unused entry.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2

        inputs = {"x": x, "y": y}
        result = torch.autograd.grad(z.sum(), inputs, materialize_grads=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["x"], 2 * x)
        self.assertEqual(result["y"], torch.zeros_like(y))

    def test_grad_dict_inputs_allow_unused(self):
        # [FUNCTION DOC] Check grad with dict inputs and allow_unused returns None for the
        # unused entry.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2

        inputs = {"x": x, "y": y}
        result = torch.autograd.grad(z.sum(), inputs, allow_unused=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["x"], 2 * x)
        self.assertIsNone(result["y"])

    def test_grad_dict_inputs_create_graph(self):
        # [FUNCTION DOC] Check grad with dict inputs and create_graph gives a result that requires
        # grad and supports a second derivative.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**3

        inputs = {"x": x}
        result = torch.autograd.grad(z.sum(), inputs, create_graph=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["x"], 3 * x**2)
        self.assertTrue(result["x"].requires_grad)

        # Second-order derivative
        result2 = torch.autograd.grad(result["x"].sum(), inputs)
        self.assertIsInstance(result2, dict)
        self.assertEqual(result2["x"], 6 * x)

    def test_grad_dict_inputs_empty(self):
        # [FUNCTION DOC] Check grad with an empty dict raises RuntimeError saying inputs cannot
        # be empty.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2
        inputs: dict[str, torch.Tensor] = {}
        self.assertRaisesRegex(
            RuntimeError,
            "cannot be empty",
            lambda: torch.autograd.grad(z.sum(), inputs),
        )

    def test_backward_dict_inputs(self):
        # [FUNCTION DOC] Check backward accepts a dict for inputs and accumulates gradients into
        # the grad attributes.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2 + y * x + y**2

        gradient = torch.ones(2, 2, dtype=torch.double)
        inputs = {"x": x, "y": y}
        torch.autograd.backward(z, gradient, inputs=inputs)
        self.assertEqual(x.grad, 2 * x + y)
        self.assertEqual(y.grad, x + 2 * y)

    def test_backward_dict_inputs_tensor_backward(self):
        # [FUNCTION DOC] Check Tensor.backward accepts a dict for inputs.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = (x**2 + y * x + y**2).sum()

        inputs = {"x": x, "y": y}
        z.backward(inputs=inputs)
        self.assertEqual(x.grad, 2 * x + y)
        self.assertEqual(y.grad, x + 2 * y)

    def test_backward_dict_inputs_empty(self):
        # [FUNCTION DOC] Check backward with an empty dict for inputs raises RuntimeError saying
        # inputs cannot be empty.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2

        gradient = torch.ones(2, 2, dtype=torch.double)
        self.assertRaisesRegex(
            RuntimeError,
            "cannot be empty",
            lambda: torch.autograd.backward(z, gradient, inputs={}),
        )

    @skipIfTorchDynamo("compiled autograd does not support is_grads_batched with vmap")
    def test_grad_dict_inputs_batched_grads(self):
        # [FUNCTION DOC] Check grad with dict inputs and batched grad outputs returns per-key
        # results with the batch size as leading dimension.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = (x**2 + y * x + y**2).sum()

        # Batched grad_outputs: shape (batch_size,) for scalar output
        batch_size = 3
        grad_outputs = torch.ones(batch_size, dtype=torch.double)
        inputs = {"x": x, "y": y}
        result = torch.autograd.grad(
            z, inputs, grad_outputs=grad_outputs, is_grads_batched=True
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"x", "y"})
        # Each result tensor should have batch_size as first dim
        self.assertEqual(result["x"].shape, (batch_size, 2, 2))
        self.assertEqual(result["y"].shape, (batch_size, 2, 2))

    def test_grad_dict_inputs_non_string_keys(self):
        # [FUNCTION DOC] Check dict inputs with integer keys work at runtime and the result keeps
        # the keys.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = x**2 + y * x + y**2

        # Non-string keys work at runtime (type annotations only enforce str)
        inputs = {0: x, 1: y}
        result = torch.autograd.grad(z.sum(), inputs)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {0, 1})
        self.assertEqual(result[0], 2 * x + y)
        self.assertEqual(result[1], x + 2 * y)

    def test_backward_with_scalar_input(self):
        # [FUNCTION DOC] Check backward with a scalar tensor passed directly as inputs.
        x = torch.randn([], dtype=torch.double, requires_grad=True)
        out = x**2
        out.backward(inputs=x)
        self.assertEqual(x.grad, 2 * x)

    def test_backward_with_nonleaf_inputs(self):
        # [FUNCTION DOC] Check backward with inputs including a non-leaf tensor fills its grad,
        # and an input outside the graph is silently left without a grad.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x_nonleaf = x * 1
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        out = x_nonleaf**2 + y * x_nonleaf + y**2

        out.backward(
            torch.ones(2, 2, dtype=torch.double),
            create_graph=True,
            inputs=[x, y, x_nonleaf],
        )
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y
        x_non_leaf_expected = 2 * x_nonleaf + y

        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(x_nonleaf.grad, x_non_leaf_expected)

        # backward doesn't have an allow_unused flag, so the behavior of backward
        # when variable is not part of the graph is as if allow_used were true
        # x.grad will simply be None.
        out.backward(
            torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[z]
        )
        self.assertIsNone(z.grad)

        # Avoid leaking memory
        x.grad = None
        y.grad = None
        x_nonleaf.grad = None

    def test_dependent_backward(self):
        # [FUNCTION DOC] Check backward with two dependent roots (one derived from the other) sums
        # both contributions into the leaf grad.
        x = torch.randn(10, requires_grad=True)
        y = x**2
        z = y**3

        go_y = torch.randn(10)
        go_z = torch.randn(10)
        torch.autograd.backward([y, z], [go_y, go_z])

        xd = x
        self.assertEqual(x.grad, 2 * xd * go_y + 6 * xd.pow(5) * go_z)

    def test_save_output_nr(self):
        # [FUNCTION DOC] Check that output_nr of the second output of a custom Function is one,
        # including when the output is saved and seen again in backward.
        x = torch.randn(10, requires_grad=True)

        class MultiOutputFn(Function):
            # [CLASS DOC] HELPER: Function splitting its input into two halves.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the first five and remaining elements of the input
                # as two outputs.
                return x[:5], x[5:]

            @staticmethod
            def backward(ctx, *grad):
                # [FUNCTION DOC] HELPER: Concatenate the gradients.
                return torch.cat(grad)

        a, b = MultiOutputFn.apply(x)
        self.assertEqual(b.output_nr, 1)

        class TestFn(Function):
            # [CLASS DOC] HELPER: Function saving its input and asserting during backward that the
            # saved tensor has output_nr one.
            @staticmethod
            def forward(ctx, b):
                # [FUNCTION DOC] HELPER: Save the input and return it doubled.
                ctx.save_for_backward(b)
                return b * 2

            @staticmethod
            def backward(ctx, grad_b):
                # [FUNCTION DOC] HELPER: Assert the saved tensor output_nr is one; return nothing.
                (b,) = ctx.saved_tensors
                self.assertEqual(b.output_nr, 1)

        TestFn.apply(b).sum().backward()

    def test_first_grad_fn_access_in_no_grad_mode(self):
        # [FUNCTION DOC] Check that first accessing grad_fn of a view of a complex tensor while
        # grad mode is disabled works after an in-place update of the base.
        a = torch.tensor([1 + 1j], requires_grad=True).clone()
        v = a.real
        a.add_(1)
        with torch.autograd.grad_mode.no_grad():
            v.grad_fn

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph(self):
        # [FUNCTION DOC] Check that freeing a very deep chain graph (150000 nodes) does not
        # overflow the stack. Skipped under dynamo as too slow.
        def scope():
            # [FUNCTION DOC] HELPER: Build a deep chain of operations and let it go out of scope
            # so the graph is freed.
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build a "chain" computation graph
            for _ in range(depth):
                y = y + y * 0.000001

            # graph deletion occurs when the above locals go out of scope.
            # In this case `del y` will trigger it but it's easier to leave
            # it to Python to delete the locals.

        # Should not stack overflow
        scope()

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph_complicated(self):
        # [FUNCTION DOC] Check that freeing a deep graph with skip connections (100000 nodes) does
        # not overflow the stack. Skipped under dynamo as too slow.
        def scope():
            # [FUNCTION DOC] HELPER: Build a chain with occasional skip connections using in-place
            # adds and let it go out of scope.
            depth = 100000
            randchoice = torch.randint(2, [depth, 2])
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # Hold the two previous values
            prev_values = [None, None]

            # Build a "chain with skip connections" graph
            for _ in range(depth):
                prev_tensors = [
                    tensor for tensor in prev_values[:-1] if tensor is not None
                ]
                prev_values.append(y)
                prev_values.pop(0)

                # Definitely pick one tensor to add
                y += y * 0.000001

                # Possibly add other tensors
                nprev = len(prev_tensors)
                if nprev == 2:
                    y += randchoice[depth].mul(torch.cat(prev_tensors)).sum()

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph_pyfunction(self):
        # [FUNCTION DOC] Check that freeing a deep graph made of Python custom Function nodes
        # (150000 deep) does not overflow the stack. Skipped under dynamo as too slow.
        class MyOp(Function):
            # [CLASS DOC] HELPER: Custom Function adding its two inputs.
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                # [FUNCTION DOC] HELPER: Return the sum.
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient for both inputs.
                return grad_output, grad_output

        def scope():
            # [FUNCTION DOC] HELPER: Build a deeply nested graph of the custom Function and let it
            # go out of scope.
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build deeply nested computation graph
            for _ in range(depth):
                y = MyOp.apply(y, y)

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    def test_no_unnecessary_save(self):
        # [FUNCTION DOC] Check that multiplication does not save an input that is needed only for
        # the other operand's grad: modifying the input in place across iterations must not
        # trigger a version counter error in backward.
        # If we kept x in the derivative Function of x * 2 we would
        # get an error in the backward that would complain that we've
        # modified x, which was needed for gradient computation.
        # Since we should elide unnecessary saves, this test should pass.
        mu = torch.ones(1, requires_grad=True)
        x = torch.empty(1)
        loss = 0
        for i in range(3):
            x.detach_()
            x.copy_(mu + i)
            ft = torch.tensor([float(i)])
            multiplied = x * ft
            s = multiplied.sum()
            loss += s
        loss.backward()

    def test_no_grad(self):
        # [FUNCTION DOC] Check no_grad used as a context manager and as a decorator in both call
        # forms disables graph recording, and that grad mode is restored.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        with torch.no_grad():
            w = x + y

        def adder(x, y):
            # [FUNCTION DOC] HELPER: Add two tensors.
            return x + y

        adders = [torch.no_grad()(adder), torch.no_grad(adder)]

        for adder in adders:
            z = adder(x, y)

            self.assertFalse(w.requires_grad)
            self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
            self.assertIsNone(w.grad_fn)
            self.assertFalse(z.requires_grad)
            self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))
            self.assertIsNone(z.grad_fn)

        # test nested decorator and with-statement on no_grad
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            w = adder(x, y)
            self.assertFalse(torch.is_grad_enabled())

    def test_enable_grad_decorator_no_paren(self):
        # [FUNCTION DOC] Check enable_grad used as a bare decorator re-enables grad inside a
        # no_grad region.
        x = torch.ones(1, requires_grad=True)

        @torch.enable_grad
        def doubler(x):
            # [FUNCTION DOC] HELPER: Return the input times 2.
            return x * 2

        with torch.no_grad():
            z = doubler(x)
        self.assertTrue(z.requires_grad)

    def test_set_grad_generator_functions(self):
        # [FUNCTION DOC] Check no_grad and enable_grad decorating generator functions: grad
        # mode inside the generator follows the decorator while the caller keeps its own mode
        # between steps.
        @torch.no_grad()
        def gen_no_grad():
            # [FUNCTION DOC] HELPER: Generator under no_grad that asserts grad is disabled at each
            # step and yields ten integers.
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), False)
                yield i

        with torch.enable_grad():
            for _ in gen_no_grad():
                self.assertEqual(torch.is_grad_enabled(), True)

        @torch.enable_grad()
        def gen_enable_grad():
            # [FUNCTION DOC] HELPER: Generator under enable_grad that asserts grad is enabled at
            # each step and yields ten integers.
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), True)
                yield i

        with torch.no_grad():
            for _ in gen_enable_grad():
                self.assertEqual(torch.is_grad_enabled(), False)

    def test_set_grad_generator_functions_recursive(self):
        # [FUNCTION DOC] Check mutually recursive calls between no_grad and enable_grad decorators
        # and context managers keep the right mode at each level and restore the caller's mode.
        # enable_grad_decorator_recursive and no_grad_decorator_recursive call each other
        # recursively, to ensure that the decorators preserve the caller's setting
        @torch.enable_grad()
        def enable_grad_decorator_recursive(depth):
            # [FUNCTION DOC] HELPER: Under enable_grad assert grad is on, recurse into the no_grad
            # function with smaller depth and assert grad is still on after.
            self.assertTrue(torch.is_grad_enabled())
            if depth > 0:
                no_grad_decorator_recursive(depth - 1)
                self.assertTrue(torch.is_grad_enabled())

        @torch.no_grad()
        def no_grad_decorator_recursive(depth):
            # [FUNCTION DOC] HELPER: Under no_grad assert grad is off, recurse into the
            # enable_grad function with smaller depth and assert grad is still off after.
            self.assertFalse(torch.is_grad_enabled())
            if depth > 0:
                enable_grad_decorator_recursive(depth - 1)
                self.assertFalse(torch.is_grad_enabled())

        # enable_grad_context_manager_recursive and no_grad_context_manager_recursive call
        # each other recursively, to ensure that the decorators preserve the caller's setting
        def enable_grad_context_manager_recursive(depth):
            # [FUNCTION DOC] HELPER: Context-manager version of the enable_grad recursion.
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if depth > 0:
                    no_grad_context_manager_recursive(depth - 1)
                    self.assertTrue(torch.is_grad_enabled())

        def no_grad_context_manager_recursive(depth):
            # [FUNCTION DOC] HELPER: Context-manager version of the no_grad recursion.
            with torch.no_grad():
                self.assertFalse(torch.is_grad_enabled())
                if depth > 0:
                    enable_grad_context_manager_recursive(depth - 1)
                    self.assertFalse(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertFalse(torch.is_grad_enabled())

    def test_set_grad_coroutines(self):
        # [FUNCTION DOC] Check no_grad and enable_grad on coroutines driven with send: mode inside
        # the coroutine follows its decorator, and the caller's mode is unchanged between sends.
        @torch.no_grad()
        def coro_no_grad(n=10):
            # [FUNCTION DOC] HELPER: Coroutine under no_grad yielding integers and asserting the
            # sent value equals the yielded value.
            self.assertFalse(torch.is_grad_enabled())
            for i in range(n):
                self.assertFalse(torch.is_grad_enabled())
                r = yield i
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertFalse(torch.is_grad_enabled())

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            # [FUNCTION DOC] HELPER: Coroutine under enable_grad yielding integers and asserting
            # the sent value equals the yielded value.
            self.assertTrue(torch.is_grad_enabled())
            for i in range(n):
                self.assertTrue(torch.is_grad_enabled())
                r = yield i
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertTrue(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            coro, r = coro_no_grad(), None
            try:
                while True:
                    self.assertTrue(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertTrue(torch.is_grad_enabled())

            except StopIteration:
                pass

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            coro, r = coro_enable_grad(), None
            try:
                while True:
                    self.assertFalse(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertFalse(torch.is_grad_enabled())

            except StopIteration:
                pass

    def test_set_grad_coroutines_benign_exceptions(self):
        # [FUNCTION DOC] Check coroutines under grad mode decorators when an exception is thrown
        # into them and handled: mode stays correct and the generator continues.
        class RecoverableException(Exception):
            # [CLASS DOC] HELPER: Exception class for the recoverable case.
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            # [FUNCTION DOC] HELPER: Generator under no_grad that handles the recoverable
            # exception by negating later yields.
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    has_raised = True

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            # [FUNCTION DOC] HELPER: Generator under enable_grad that handles the recoverable
            # exception by negating later yields.
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    has_raised = True

        with torch.enable_grad():
            coro = coro_no_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

        with torch.no_grad():
            coro = coro_enable_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

    def test_set_grad_coroutines_critical_exceptions(self):
        # [FUNCTION DOC] Check that an exception thrown into a grad-mode decorated coroutine and
        # re-raised as a different exception propagates to the caller.
        class UnrecoverableException(Exception):
            # [CLASS DOC] HELPER: Exception class thrown into the coroutines.
            pass

        class SecondaryException(Exception):
            # [CLASS DOC] HELPER: Exception raised by the coroutines in response.
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            # [FUNCTION DOC] HELPER: Generator under no_grad that converts the thrown exception
            # into the secondary one.
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    raise SecondaryException from None

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            # [FUNCTION DOC] HELPER: Generator under enable_grad that converts the thrown
            # exception into the secondary one.
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    raise SecondaryException from None

        with torch.enable_grad():
            coro = coro_no_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

        with torch.no_grad():
            coro = coro_enable_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

    def test_set_grad_coroutines_exit(self):
        # [FUNCTION DOC] Check closing a grad-mode decorated coroutine delivers GeneratorExit
        # inside it with the right grad mode.
        @torch.no_grad()
        def coro_no_grad(state):
            # [FUNCTION DOC] HELPER: Generator under no_grad that records the exit in the given
            # set when closed.
            for i in range(10):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertFalse(torch.is_grad_enabled())
                    state.add("GeneratorExit")
                    raise

        @torch.enable_grad()
        def coro_enable_grad(state):
            # [FUNCTION DOC] HELPER: Generator under enable_grad that records the exit in the
            # given set when closed.
            for i in range(10):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertTrue(torch.is_grad_enabled())
                    state.add("GeneratorExit")
                    raise

        state = set()
        with torch.enable_grad():
            coro = coro_no_grad(state)
            for _ in range(5):
                next(coro)

            coro.close()
        self.assertTrue("GeneratorExit" in state)

        state = set()
        with torch.no_grad():
            coro = coro_enable_grad(state)
            for _ in range(5):
                next(coro)

            coro.close()
        self.assertTrue("GeneratorExit" in state)

    def test_no_grad_python_function(self):
        # [FUNCTION DOC] Check a custom Function applied under no_grad produces an output that
        # does not require grad.
        """Python Functions should respect grad mode."""
        x = torch.ones(5, 5, requires_grad=True)

        class MyOp(Function):
            # [CLASS DOC] HELPER: Function adding one to its input.
            @staticmethod
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input plus one.
                return x + 1

            @staticmethod
            def backward(self, dy):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return dy

        with torch.no_grad():
            y = MyOp.apply(x)
        self.assertFalse(y.requires_grad)

    def test_indexing(self):
        # [FUNCTION DOC] Check indexing of Variables with many index forms (ints, slices,
        # ellipsis, tensors, boolean masks, lists, advanced indexing) gives the same values as
        # plain tensors and the expected one-hot gradients.
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        def compare(x, y, idx, indexed_tensor, indexed_var):
            # [FUNCTION DOC] HELPER: Compare indexed tensor and variable values, run backward on
            # the variable result, and compare y.grad with a one-valued tensor at the index.
            indexed_var_t = indexed_var.data
            if not isinstance(indexed_tensor, torch.Tensor):
                indexed_var_t = indexed_var_t[0]
            self.assertEqual(indexed_tensor, indexed_var_t)

            indexed_var.sum().backward()
            expected_grad = torch.empty(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad, expected_grad)

        def check_index(x, y, idx):
            # [FUNCTION DOC] HELPER: Zero any existing grad, index both tensor and variable
            # and compare.
            if y.grad is not None:
                with torch.no_grad():
                    y.grad.zero_()
            indexed_tensor = x[idx]
            indexed_var = y[idx]
            compare(x, y, idx, indexed_tensor, indexed_var)

        check_index(x, y, 1)
        check_index(x, y, (1, 1))
        check_index(x, y, slice(1, None))
        check_index(x, y, slice(None, 2))
        check_index(x, y, (slice(None, 2), 2))
        check_index(x, y, (slice(1, 2), 2))
        check_index(x, y, (1, slice(2, None)))
        check_index(x, y, (slice(None, None), slice(2, None)))
        check_index(x, y, torch.LongTensor([0, 2]))
        check_index(x, y, torch.rand(4, 4).bernoulli().bool())
        check_index(x, y, (Ellipsis, slice(2, None)))
        check_index(x, y, ([0], [0]))
        check_index(x, y, ([1, 2, 3], [0]))
        check_index(x, y, ([1, 2], [2, 1]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ((slice(None), [2, 3])))
        check_index(x, y, (([2, 3], slice(None))))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0]))
        check_index(x, y, ([0],))

        x = torch.arange(1.0, 49).view(4, 3, 4)
        y = Variable(x, requires_grad=True)

        check_index(x, y, (slice(None), [0], [0]))
        check_index(x, y, ([0], [0], slice(None)))
        check_index(x, y, (slice(None), [0, 1, 2], [0]))
        check_index(x, y, ([0, 1, 2], [0], slice(None)))
        check_index(x, y, (slice(None), [1, 2], [2, 1]))
        check_index(x, y, ([1, 2], [2, 1], slice(None)))
        check_index(x, y, (slice(None), [[1, 2], [2, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 2]], slice(None)))
        check_index(x, y, (slice(None), slice(None), [2, 1]))
        check_index(x, y, (slice(None), [2, 1], slice(None)))
        check_index(x, y, ([2, 1], slice(None), slice(None)))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0],))
        check_index(x, y, ([0], slice(None)))
        check_index(x, y, ([0], Ellipsis))
        check_index(x, y, ([1, 2], [0, 1]))
        check_index(x, y, ([1, 2], [0, 1], Ellipsis))
        check_index(x, y, (Ellipsis, [1, 2], [0, 1]))

        # advanced indexing, with a tensor wrapped in a variable
        z = torch.LongTensor([0, 1])
        zv = Variable(z, requires_grad=False)
        seq = (z, Ellipsis)
        seqv = (zv, Ellipsis)

        if y.grad is not None:
            with torch.no_grad():
                y.grad.zero_()
        indexed_tensor = x[seq]
        indexed_var = y[seqv]
        compare(x, y, seq, indexed_tensor, indexed_var)

    def test_indexing_duplicates(self):
        # [FUNCTION DOC] Check the backward of indexing with duplicate indices accumulates one per
        # repetition, for integer tensors, advanced lists, nested lists and mixed slices.
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx:
            expected_grad[i] += 1
        self.assertEqual(y.grad, expected_grad)

        # with advanced indexing
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = ([1, 1, 3, 2, 1, 2], [0])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = ([[1, 2], [0, 0]], [[0, 1], [1, 1]])
        y[idx].sum().backward()
        expected_grad = torch.tensor(
            [
                [0.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1.0, 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)

        idx = ([1, 1, 1], slice(None), slice(None))
        y[idx].sum().backward()
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad, expected_grad)

    def test_index_backward_does_not_save_tensor(self):
        # [FUNCTION DOC] Check indexing assignment pattern from a known bug report does not make
        # indexing save the tensor, so no version check error occurs in backward.
        # Example from https://github.com/pytorch/pytorch/issues/24853.
        # if `index(tensor, indices)` saves `tensor` for backwards, then it will
        # trigger a version check on `tensor` during the backward pass, which
        # will cause the following code to error because `tensor` gets modified
        # by the indexing line.
        a = torch.tensor([1.0, 0, 0])
        b = torch.zeros(3, requires_grad=True)
        tensor = b + 0
        tensor[a != 0] = tensor[a != 0]
        tensor.backward(torch.zeros_like(tensor))

    def test_volatile_deprecated(self):
        # [FUNCTION DOC] Check that reading the volatile attribute emits a warning mentioning
        # volatile and returns False.
        v = torch.autograd.torch.randn(3, 3)
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(v.volatile)
        self.assertIn("volatile", str(w[0].message))

    def test_saved_variables_deprecated(self):
        # [FUNCTION DOC] Check that using saved_variables on ctx emits a deprecation warning
        # mentioning saved_variables during backward.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function adding two tensors whose backward uses the deprecated
            # saved_variables attribute.
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                # [FUNCTION DOC] HELPER: Save both inputs and return their sum.
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Read the deprecated attribute and pass the gradient to
                # both inputs.
                var1, var2 = ctx.saved_variables
                return (grad_output, grad_output)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            x = torch.randn((3, 3), requires_grad=True)
            y = torch.randn((3, 3), requires_grad=True)
            MyFunction.apply(x, y).sum().backward()

            has_deprecated = (
                "deprecated" in str(warn) and "saved_variables" in str(warn)
                for warn in warns
            )
            has_deprecated = reduce(lambda x, y: x or y, has_deprecated)
            self.assertTrue(has_deprecated)

    def test_requires_grad(self):
        # [FUNCTION DOC] Check requires_grad propagation and that backward hooks on tensors that
        # do not need grad are not called.
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.randn(5, 5, requires_grad=True)
        a = x + y
        self.assertFalse(a.requires_grad)
        b = a + z
        self.assertTrue(b.requires_grad)

        def error():
            # [FUNCTION DOC] HELPER: Hook that raises RuntimeError if called.
            raise RuntimeError

        # Make sure backward isn't called on these
        a._backward_hooks = OrderedDict()
        x._backward_hooks = OrderedDict()
        y._backward_hooks = OrderedDict()
        a._backward_hooks["test"] = error
        x._backward_hooks["test"] = error
        y._backward_hooks["test"] = error
        b.backward(torch.ones(5, 5))

    def test_requires_grad_(self):
        # [FUNCTION DOC] Check requires_grad_ returns the same tensor, can be set on leaves,
        # cannot be turned off on a non-leaf (RuntimeError), and can be reset on leaves.
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        self.assertIs(x, x.requires_grad_())
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_())
        self.assertTrue(y.requires_grad)
        self.assertIs(x, x.requires_grad_(True))
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_(True))
        self.assertTrue(y.requires_grad)
        z = x * y
        self.assertRaises(RuntimeError, lambda: z.requires_grad_(False))
        self.assertIs(z, z.requires_grad_())
        self.assertTrue(z.requires_grad)
        self.assertIs(z, z.requires_grad_(True))
        self.assertTrue(z.requires_grad)

        self.assertIs(x, x.requires_grad_(False))
        self.assertFalse(x.requires_grad)
        self.assertIs(y, y.requires_grad_(False))
        self.assertFalse(y.requires_grad)

    def test_requires_grad_inplace(self):
        # [FUNCTION DOC] Check in-place ops with a grad-requiring operand make the target require
        # grad, for both leaf and non-leaf targets.
        a = torch.randn(5, 5)
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

        # non-leaf
        a = torch.randn(5, 5) + 0
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

    def test_no_requires_grad_inplace(self):
        # [FUNCTION DOC] Check in-place modification is allowed while requires_grad is False (also
        # via a view), and raises RuntimeError once requires_grad is set on the base and an
        # in-place op is applied to the base or the view.
        # basic case, should be able to modify inplace while requires_grad is False
        a = torch.randn(2, 3)
        a.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # same but with a view
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # should fail if requires_grad = True when we modify inplace
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    def test_attribute_deletion(self):
        # [FUNCTION DOC] Check attribute deletion on a tensor: grad can be deleted (becomes None),
        # but deleting data, requires_grad, _grad_fn or _backward_hooks raises RuntimeError, and
        # setting data to None raises TypeError.
        x = torch.randn((5, 5), requires_grad=True)
        del x.grad
        self.assertIsNone(x.grad)
        with self.assertRaises(RuntimeError):
            del x.data
        with self.assertRaises(TypeError):
            x.data = None
        with self.assertRaises(RuntimeError):
            del x.requires_grad
        with self.assertRaises(RuntimeError):
            del x._grad_fn
        with self.assertRaises(RuntimeError):
            del x._backward_hooks

    def test_duplicate_backward_root(self):
        # [FUNCTION DOC] Check duplicate roots in backward add their gradients (twice the
        # single result).
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        x = a * b
        grad_output = torch.randn_like(x)
        torch.autograd.backward([x, x], [grad_output, grad_output])

        self.assertEqual(a.grad, b * grad_output * 2)
        self.assertEqual(b.grad, a * grad_output * 2)

    def test_backward_no_grad(self):
        # [FUNCTION DOC] Check backward with a None gradient for a non-scalar root
        # raises RuntimeError.
        a = torch.randn(5, 5, requires_grad=True)
        b = a + 2
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    def test_backward_twice_with_saved_values(self):
        # [FUNCTION DOC] Check a second backward through a graph with saved values raises
        # RuntimeError telling to specify retain_graph.
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        self.assertRaisesRegex(
            RuntimeError,
            "Specify retain_graph=True",
            lambda: c.backward(torch.tensor([1, 1, 1], dtype=torch.double)),
        )

    def test_backward_twice_retained_graph_with_saved_values(self):
        # [FUNCTION DOC] Check a second backward works when the first used retain_graph.
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_without_saved_values(self):
        # [FUNCTION DOC] Check a graph with no saved values can be backwarded twice
        # without retain_graph.
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = b + 1
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_retained_graph_without_saved_values(self):
        # [FUNCTION DOC] Check a second backward works when the first used retain_graph, for a
        # graph built by index assignment that holds no saved values.
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_create_graph_warns(self):
        # [FUNCTION DOC] Check that backward with create_graph emits a warning mentioning that
        # usage (forced to always warn), while grad with create_graph does not.
        with set_warn_always_context(True):
            b = torch.randn(3, requires_grad=True, dtype=torch.double)
            c = b * b
            with warnings.catch_warnings(record=True) as ws:
                c.backward(torch.ones_like(c), create_graph=True)
            b.grad = None
            self.assertTrue(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

            # Should not warn for grad
            with warnings.catch_warnings(record=True) as ws:
                torch.autograd.grad(c, b, torch.ones_like(c), create_graph=True)
            self.assertFalse(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

    def test_next_functions(self):
        # [FUNCTION DOC] Check next_functions of a grad_fn: AccumulateGrad nodes with output
        # number zero for leaves, and for a node with a non-tensor operand the second entry
        # is None.
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        self.assertIsNotNone(a.grad_fn)
        next_functions = a.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIsInstance(next_functions[0][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[0][1], 0)
        self.assertIsInstance(next_functions[1][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[1][1], 0)

        b = a + 5
        next_functions = b.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIs(next_functions[0][0], a.grad_fn)
        self.assertIs(next_functions[1][0], None)

    def test_copy_slices_wrapped_node(self):
        # [FUNCTION DOC] Check the private wrapped node of a CopySlices node created by in-place
        # exp on a view: it is an exp backward node with saved result and input metadata matching
        # the view, linked to the old view grad_fn, and it is released (None) after backward.
        leaf = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        base = leaf.clone()
        view = base[:2]
        view_grad_fn = view.grad_fn
        view.exp_()

        copy_slices = base.grad_fn
        wrapped_node = copy_slices._wrapped_node
        self.assertIsInstance(copy_slices, torch._C._functions.CopySlices)
        self.assertEqual(wrapped_node.name(), "ExpBackward0")
        self.assertIs(copy_slices._wrapped_node, wrapped_node)
        # exp_ saves its result after rebase_history creates CopySlices.
        self.assertEqual(wrapped_node._saved_result, view)
        self.assertEqual(wrapped_node._input_metadata[0].shape, view.shape)
        self.assertIs(wrapped_node.next_functions[0][0], view_grad_fn)

        base.sum().backward()
        self.assertEqual(leaf.grad, torch.tensor([math.exp(1), math.exp(2), 1.0, 1.0]))
        self.assertIsNone(copy_slices._wrapped_node)

    def test_inplace(self):
        # [FUNCTION DOC] Check in-place operations and saved-tensor version checks: backward
        # through ops that do not save the modified value works, ops that saved it raise
        # RuntimeError, version counter changes, and a modified clone remains usable.
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5), retain_graph=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5), retain_graph=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5), retain_graph=True)
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        with torch.no_grad():
            x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.empty(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = torch.ones(5, 5, requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x, torch.ones(5, 5) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad, torch.ones(5, 5))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))

    def test_mark_non_differentiable(self):
        # [FUNCTION DOC] Check mark_non_differentiable on a boolean output: the output does not
        # require grad and can be used as a mask in a differentiable op whose backward still runs.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function returning a boolean mask marked non-differentiable.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Compute the mask, mark it non-differentiable and
                # return it.
                output = input > 0
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return zeros of double dtype as the input gradient.
                return (grad_output * 0).to(torch.double)

        x = torch.randn(5, 5, requires_grad=True)
        mask = MyFunction.apply(x)
        self.assertFalse(mask.requires_grad)
        y = x.masked_fill(mask, 0)
        y.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_mark_non_differentiable_mixed(self):
        # [FUNCTION DOC] Check a Function with two outputs, one marked non-differentiable:
        # backward receives zeros for it and ones for the other, only the second requires grad.
        # Skipped under dynamo.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function returning input plus one (marked non-differentiable)
            # and input plus two.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Mark the first output non-differentiable and return both.
                a = input + 1
                b = input + 2
                ctx.mark_non_differentiable(a)
                return a, b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # [FUNCTION DOC] HELPER: Assert the first gradient is all zeros and the second all
                # ones; return the second.
                self.assertTrue((grad_a == 0).all())
                self.assertTrue((grad_b == 1).all())
                return grad_b

        x = torch.randn(5, 5, requires_grad=True)
        a, b = MyFunction.apply(x)
        self.assertFalse(a.requires_grad)
        self.assertTrue(b.requires_grad)
        b.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5))

    def test_mark_non_differentiable_none(self):
        # [FUNCTION DOC] Regression check for a former segfault: a non-differentiable output whose
        # backward returns None feeds a C++ multiply node during backward.
        # This used to segfault because MyFunction would send back null
        # gradients to MulBackward, which is implemented in C++. C++
        # implemented functions expect incoming grad_outputs to be non-null.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function returning a non-differentiable clone with a backward
            # returning None.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Clone the input, mark the clone non-differentiable and
                # return it.
                output = input.clone()
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return None.
                return None

        x = torch.randn(5, 5, requires_grad=True)
        r = MyFunction.apply(x * x)
        (r * x).sum().backward()

    def test_return_duplicate(self):
        # [FUNCTION DOC] Check a Function returning the same tensor twice: both outputs are the
        # same object, and gradcheck plus gradgradcheck pass.
        class DoubleDuplicate(Function):
            # [CLASS DOC] HELPER: Function returning the same doubled tensor twice.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the doubled input as both outputs.
                output = x * 2
                return output, output

            @staticmethod
            def backward(ctx, grad1, grad2):
                # [FUNCTION DOC] HELPER: Sum both gradients, doubled.
                return grad1 * 2 + grad2 * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply the duplicate-output Function, assert both outputs are
            # one object, and return their sum.
            a, b = DoubleDuplicate.apply(x)
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(fn, [x])
        gradgradcheck(fn, [x])

    def test_return_duplicate_inplace(self):
        # [FUNCTION DOC] Check a Function that modifies its input in place and returns it
        # twice: gradcheck passes; applying the in-place base class to a leaf or to a view
        # raises RuntimeError.
        class DoubleInplace(Function):
            # [CLASS DOC] HELPER: Function that doubles its input in place, marks it dirty and
            # returns it twice.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Double the input in place, mark it dirty and return
                # it twice.
                x.mul_(2)
                ctx.mark_dirty(x)
                return x, x

            @staticmethod
            def backward(ctx, grad1, grad2):
                # [FUNCTION DOC] HELPER: Sum both gradients, doubled.
                return grad1 * 2 + grad2 * 2

        def inplace_fn(x):
            # [FUNCTION DOC] HELPER: Apply the in-place Function to a clone, assert both outputs
            # are the same object and return their sum.
            a, b = DoubleInplace.apply(x.clone())
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(inplace_fn, [x])
        gradgradcheck(inplace_fn, [x])

        # Can't modify leaf variables in-place
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x))
        # Functions which modify views in-place must return only one output
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x.clone()[0]))

    def _test_setitem(self, size, index):
        # [FUNCTION DOC] HELPER: For a tensor of the given size, assign the constant 2 at an index
        # of an intermediate, check the version counter changes, and that backward gives zero
        # gradient at the index and one elsewhere.
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        expected_grad[index] = 0
        self.assertEqual(x.grad, expected_grad)

    def _test_setitem_tensor(self, size, index):
        # [FUNCTION DOC] HELPER: Like the constant setitem helper but assigning a tensor requiring
        # grad; check the gradients of both source and base, and the broadcast case where a
        # 4-element tensor is assigned into a row of a bigger one.
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        value = x.new(x[index].size()).fill_(7)
        value.requires_grad = True
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        expected_grad_input[index] = 0
        self.assertEqual(x.grad, expected_grad_input)
        self.assertEqual(value.grad, torch.ones_like(value))

        # case when x broadcasts to as y[1]
        x = torch.randn(4, requires_grad=True)
        y = torch.zeros(2, 3, 4)
        y[1] = x
        y.backward(torch.randn(2, 3, 4))
        self.assertEqual(x.size(), x.grad.size())

    def test_setitem(self):
        # [FUNCTION DOC] Check item assignment with a constant and with a tensor, using integer,
        # list, slice and tensor indices of several shapes.
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem((10,), ([0, 4, 2]))
        self._test_setitem((5, 5), ([0, 4], [2, 2]))
        self._test_setitem((5, 5, 5), (slice(None), slice(None), [1, 3]))
        self._test_setitem((5, 5, 5), (slice(None), [1, 3], slice(None)))
        self._test_setitem((5, 5, 5), ([1, 3], slice(None), slice(None)))
        self._test_setitem((5, 5, 5), (slice(None), [2, 4], [1, 3]))
        self._test_setitem((5, 5, 5), ([1, 3], [2, 4], slice(None)))
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5, 5), ([0, 1], [1, 0]))
        self._test_setitem_tensor((5,), 3)
        self._test_setitem_tensor(
            (5,), Variable(torch.LongTensor([3]), requires_grad=False).sum()
        )
        self._test_setitem_tensor((5,), [[0, 1, 2, 3]])
        self._test_setitem_tensor((5, 5, 5), (slice(None), slice(None), [1, 3]))
        self._test_setitem_tensor((5, 5, 5), (slice(None), [1, 3], slice(None)))
        self._test_setitem_tensor((5, 5, 5), ([1, 3], slice(None), slice(None)))
        self._test_setitem_tensor((5, 5, 5), (slice(None), [2, 4], [1, 3]))
        self._test_setitem_tensor((5, 5, 5), ([1, 3], [2, 4], slice(None)))
        self._test_setitem_tensor(
            (5, 5, 5),
            (
                Variable(torch.LongTensor([1, 3]), requires_grad=False),
                [2, 4],
                slice(None),
            ),
        )

    def test_setitem_mask(self):
        # [FUNCTION DOC] Check item assignment through a boolean mask, with constant and
        # tensor values.
        mask = torch.BoolTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        # [FUNCTION DOC] Check gradcheck and gradgradcheck for select followed by sum.
        # both select and sum return Scalars in ATen; ensure they work together.
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        def func(x):
            # [FUNCTION DOC] HELPER: Select index one of dimension zero and sum.
            return x.select(0, 1).sum()

        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_diagonal_expanded_v(self):
        # [FUNCTION DOC] Check the backward of diagonal with an expanded incoming gradient equals
        # a scaled identity matrix.
        value = torch.rand([])
        v_expanded = torch.tensor(value).expand(10)
        a = torch.rand(10, 10, dtype=torch.double, requires_grad=True)
        (result,) = torch.autograd.grad(a.diagonal(), a, v_expanded)
        self.assertEqual(result, torch.eye(10, dtype=torch.double) * value)

    def test_select_expanded_v(self):
        # [FUNCTION DOC] Check the backward of select with an expanded incoming gradient puts it
        # in the selected slice and zero elsewhere.
        v_expanded = torch.rand(10).expand(10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[0], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[0] = v_expanded
        self.assertEqual(result, expected)

    def test_slice_expanded_v(self):
        # [FUNCTION DOC] Check the backward of a slice with an expanded incoming gradient puts it
        # in the sliced rows and zero elsewhere.
        v_expanded = torch.rand(10, 1).expand(2, 10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[3:5], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[3:5] = v_expanded
        self.assertEqual(result, expected)

    def test_unused_output(self):
        # [FUNCTION DOC] Check backward when only one output of a chunk is used: the gradient is
        # nonzero only for that chunk's rows, both through a derived value and when calling
        # backward directly on a chunk.
        x = torch.randn(10, 10, requires_grad=True)
        outputs = x.chunk(5)
        o = outputs[2]
        o = o * 4 + 2
        o.sum().backward()
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        self.assertEqual(x.grad, expected_grad)

        with torch.no_grad():
            x.grad.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad, expected_grad)

    # TODO: opinfo this or move to the sparse test suite
    def _test_sparse_gather(self, size_x, size_ind, dim):
        # [FUNCTION DOC] HELPER: Compare the dense gradient of gather with the densified gradient
        # from gather with sparse_grad, for the given input size, index size and dim; scalars
        # handled by using a zero index.
        x = torch.randn(size_x, requires_grad=True)
        if len(size_ind) > 0 and len(size_x) > 0:
            ind = torch.randint(x.size(dim), size_ind)
        else:
            ind = torch.zeros(size_ind, dtype=torch.int64)
        out = torch.gather(x, dim, ind, sparse_grad=False)
        grad = torch.rand_like(out)
        out.backward(grad)
        grad_dense = x.grad.clone()
        x.grad = None
        out = torch.gather(x, dim, ind, sparse_grad=True)
        out.backward(grad)
        self.assertEqual(grad_dense, x.grad.to_dense())

    def test_sparse_gather_dim0(self):
        # [FUNCTION DOC] Sparse gather gradient check with dim zero.
        self._test_sparse_gather((10, 10), (5, 10), 0)

    def test_sparse_gather_dim1(self):
        # [FUNCTION DOC] Sparse gather gradient check with dim one.
        self._test_sparse_gather((10, 10, 5), (10, 5, 5), 1)

    def test_sparse_gather_dim_neg(self):
        # [FUNCTION DOC] Sparse gather gradient check with negative dim.
        self._test_sparse_gather((10, 10, 5), (10, 10, 2), -1)

    def test_sparse_gather_ind_scalar(self):
        # [FUNCTION DOC] Sparse gather gradient check with a scalar-shaped index.
        self._test_sparse_gather((10,), (), 0)

    def test_sparse_gather_x_scalar(self):
        # [FUNCTION DOC] Sparse gather gradient check with a scalar input.
        self._test_sparse_gather((), (2,), 0)

    def test_sparse_gather_both_scalar(self):
        # [FUNCTION DOC] Sparse gather gradient check with scalar input and index.
        self._test_sparse_gather((), (), 0)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_grad_dtype(self):
        # [FUNCTION DOC] Check the grad_dtype attribute of leaf tensors: default is the tensor
        # dtype, can be changed or set to None, is only accessible on leaves, controls validation
        # of manually assigned grads, and controls casting of incoming gradients and of
        # accumulation into existing grads. Skipped under dynamo.
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        # Default to tensor's dtype
        self.assertEqual(leaf.grad_dtype, torch.float32)
        leaf.grad_dtype = torch.float16
        self.assertEqual(leaf.grad_dtype, torch.float16)
        leaf.grad_dtype = None  # Allow any dtype
        self.assertIsNone(leaf.grad_dtype)

        # get/set grad_dtype is only allowed on leaf tensors
        non_leaf = leaf * 2
        self.assertFalse(non_leaf.is_leaf)
        with self.assertRaisesRegex(
            RuntimeError, "grad_dtype can only be accessed on leaf tensors"
        ):
            _ = non_leaf.grad_dtype
        with self.assertRaisesRegex(
            RuntimeError, "grad_dtype can only be set on leaf tensors"
        ):
            non_leaf.grad_dtype = torch.float16

        # Manual setting
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        grad_match = torch.tensor([1.0, 1.0])
        x.grad = grad_match
        self.assertEqual(x.grad.dtype, torch.float32)

        x.grad = None
        x.grad_dtype = torch.float16
        grad_mismatch = torch.tensor([1.0, 1.0])
        with self.assertRaisesRegex(
            RuntimeError,
            "attempting to assign a gradient with dtype.*float.*to a tensor with grad_dtype.*Half",
        ):
            x.grad = grad_mismatch

        # When grad_dtype is None, any dtype is allowed
        x.grad = None
        x.grad_dtype = None
        grad_any = torch.tensor([1.0, 1.0], dtype=torch.float64)
        x.grad = grad_any
        self.assertEqual(x.grad.dtype, torch.float64)

        # Incoming gradient case
        class MismatchedGradientFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward returns a double precision gradient
            # regardless of input dtype.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Return the input doubled.
                return inp * 2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient converted to double precision.
                return grad_output.to(torch.float64)

        d = torch.tensor([1.0, 2.0], requires_grad=True)
        output = MismatchedGradientFunction.apply(d)
        loss = output.sum()
        loss.backward()
        # Default behavior is to cast to tensor dtype
        self.assertEqual(d.grad.dtype, torch.float32)
        self.assertTrue(torch.allclose(d.grad, torch.tensor([1.0, 1.0])))

        e = torch.tensor([3.0, 4.0], requires_grad=True)
        e.grad_dtype = None
        output_e = MismatchedGradientFunction.apply(e)
        loss_e = output_e.sum()
        loss_e.backward()
        # No casting is done if set to None.
        self.assertTrue(
            torch.allclose(e.grad, torch.tensor([1.0, 1.0], dtype=torch.float64))
        )

        f = torch.tensor([5.0, 6.0], requires_grad=True)
        f.grad_dtype = torch.float16  # Expect float16 gradients
        output_f = MismatchedGradientFunction.apply(f)
        loss_f = output_f.sum()
        loss_f.backward()
        self.assertTrue(
            torch.allclose(f.grad, torch.tensor([1.0, 1.0], dtype=torch.float16))
        )

        # Setting grad_dtype when gradient already exists
        g = torch.tensor([1.0, 2.0], requires_grad=True)
        g.grad = torch.tensor([1.0, 1.0])
        g.grad_dtype = torch.float32
        self.assertEqual(g.grad_dtype, torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot set grad_dtype.*because there is already a gradient"
        ):
            g.grad_dtype = torch.float16
        g.grad_dtype = None
        self.assertIsNone(g.grad_dtype)
        g.grad = None
        g.grad_dtype = torch.float16
        self.assertEqual(g.grad_dtype, torch.float16)

        # Test the case where there is an existing accumulate grad
        h = torch.tensor([1.0, 2.0], requires_grad=True)
        _ = h.clone()
        h.grad_dtype = None
        output = MismatchedGradientFunction.apply(h)
        output.sum().backward()
        self.assertEqual(h.grad.dtype, torch.float64)

        # Mixed accumulation cases
        k = torch.tensor([1.0, 2.0], requires_grad=True)
        k.grad_dtype = None
        y = k * 2
        y.sum().backward()
        k.grad = k.grad.to(torch.bfloat16)
        y2 = k * 3
        # Doesn't type promote to float32, always coerce to current .grad's dtype.
        # This is because the accumulation is done in-place on the existing grad.
        self.assertEqual(k.grad.dtype, torch.bfloat16)

        l = torch.tensor([3.0, 4.0], requires_grad=True, dtype=torch.bfloat16)
        l.grad_dtype = None
        z = l * 2
        z.sum().backward()
        l.grad = l.grad.to(torch.float32)
        z2 = l * 3
        z2.sum().backward()
        self.assertEqual(l.grad.dtype, torch.float32)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    @parametrize(
        "declared,expected",
        [
            # unset -> slot defaults to storage dtype; explicit up/down casts the
            # incoming fp64 grad; None passes it through uncast; the output's own
            # dtype requests the default explicitly.
            ("unset", torch.bfloat16),
            (torch.float32, torch.float32),
            (torch.float16, torch.float16),
            (None, torch.float64),
            (torch.bfloat16, torch.bfloat16),
        ],
    )
    def test_ctx_output_grad_dtype(self, declared, expected):
        # [FUNCTION DOC] Parametrized check of the Function API for declaring the output gradient
        # dtype: backward receives the declared dtype (default is the output storage dtype, None
        # means no cast) even when downstream returns a double gradient. Skipped under dynamo.
        # ctx.set_output_grad_dtype controls the dtype of the gradient the
        # engine hands to backward, independent of the output's storage dtype.
        # The output is bf16; a downstream node returns an fp64 gradient for it.
        class Downstream(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward returns a double precision gradient
            # of ones.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone of the input.
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return ones in double precision shaped like the gradient.
                return torch.ones_like(g, dtype=torch.float64)

        class DeclareOutput(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function producing a bfloat16 output, optionally declaring the
            # output gradient dtype; records the dtype of the received gradient on the class.
            @staticmethod
            def forward(ctx, x, declared):
                # [FUNCTION DOC] HELPER: Convert the input to bfloat16 and declare the grad dtype
                # unless the sentinel unset is given.
                out = x.to(torch.bfloat16)
                if declared != "unset":
                    ctx.set_output_grad_dtype(declared)
                return out

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Record the dtype of the incoming gradient and return it
                # as float32 plus None.
                DeclareOutput.seen = g.dtype
                return g.to(torch.float32), None

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = DeclareOutput.apply(x, declared)
        self.assertEqual(out.dtype, torch.bfloat16)
        Downstream.apply(out).sum().backward()
        self.assertEqual(DeclareOutput.seen, expected)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_multiple_outputs(self):
        # [FUNCTION DOC] Check declaring output gradient dtypes for several outputs positionally:
        # a concrete dtype, the output's own dtype, and None for a non-tensor output. Skipped
        # under dynamo.
        # Positional declarations line up with returned values: a concrete
        # dtype, the default (via the output's own dtype), and None for a
        # non-tensor output.
        class MultiOutput(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning two tensors and a string, declaring a dtype
            # for each.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Compute sine and cosine, declare dtypes and return both
                # and a string.
                t1 = x.sin()
                t2 = x.cos()
                ctx.set_output_grad_dtype(torch.float64, t2.dtype, None)
                return t1, t2, "not a tensor"

            @staticmethod
            def backward(ctx, g1, g2, g3):
                # [FUNCTION DOC] HELPER: Record dtypes of the received gradients and return their
                # float32 sum.
                MultiOutput.seen = (g1.dtype, g2.dtype, g3)
                return g1.to(torch.float32) + g2.to(torch.float32)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        t1, t2, s = MultiOutput.apply(x)
        self.assertEqual(s, "not a tensor")
        (t1.sum() + t2.sum()).backward()
        self.assertEqual(MultiOutput.seen[0], torch.float64)
        self.assertEqual(MultiOutput.seen[1], torch.float32)
        self.assertIsNone(MultiOutput.seen[2])

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_aliased_outputs(self):
        # [FUNCTION DOC] Check setup_context can declare different gradient dtypes for output
        # slots that alias the same storage. Skipped under dynamo.
        # setup_context can declare distinct grad dtypes for output slots whose
        # tensors alias the same storage.
        class SetupContext(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning two views of one bfloat16 tensor, declaring
            # float32 and float64 gradients in setup_context.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Convert to bfloat16 and return two views of it.
                base = x.to(torch.bfloat16)
                return base.view_as(base), base.view_as(base)

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Declare float32 and float64 output gradient dtypes.
                ctx.set_output_grad_dtype(torch.float32, torch.float64)

            @staticmethod
            def backward(ctx, g0, g1):
                # [FUNCTION DOC] HELPER: Record the dtypes of both received gradients and return
                # their float32 sum.
                SetupContext.output_grad_dtypes = (g0.dtype, g1.dtype)
                return g0.to(torch.float32) + g1.to(torch.float32)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out0, out1 = SetupContext.apply(x)
        (out0.sum() + out1.sum()).backward()
        self.assertEqual(
            SetupContext.output_grad_dtypes, (torch.float32, torch.float64)
        )

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_accumulation(self):
        # [FUNCTION DOC] Check a declared output gradient dtype also fixes the dtype in which
        # gradients from several consumers are accumulated before reaching backward. Skipped
        # under dynamo.
        # A declared grad dtype also governs the dtype the engine accumulates
        # in when the output fans out to multiple consumers: the two fp64
        # incoming gradients are accumulated as fp32 (the declared dtype), so
        # backward sees fp32.
        class Consumer(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function cloning in forward and returning a double precision
            # gradient of ones.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return ones in double precision.
                return torch.ones_like(g, dtype=torch.float64)

        class Fanout(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function producing a bfloat16 output with declared float32
            # gradient dtype, recording the received dtype.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Convert to bfloat16, declare float32 gradients and
                # return it.
                out = x.to(torch.bfloat16)
                ctx.set_output_grad_dtype(torch.float32)
                return out

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Record the received gradient dtype and return it
                # as float32.
                Fanout.seen = g.dtype
                return g.to(torch.float32)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = Fanout.apply(x)
        (Consumer.apply(out).sum() + Consumer.apply(out).sum()).backward()
        self.assertEqual(Fanout.seen, torch.float32)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_setup_context(self):
        # [FUNCTION DOC] Check declaring the output gradient dtype from setup_context. Skipped
        # under dynamo.
        # The API is equally usable from setup_context.
        class UseSetup(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function with separate setup_context declaring float32 gradients
            # for a bfloat16 output.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Convert to bfloat16.
                return x.to(torch.bfloat16)

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Declare float32 output gradient dtype.
                ctx.set_output_grad_dtype(torch.float32)

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Record the received gradient dtype and return it
                # as float32.
                UseSetup.seen = g.dtype
                return g.to(torch.float32)

        class Downstream(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning a double precision gradient of ones.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return ones in double precision.
                return torch.ones_like(g, dtype=torch.float64)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = UseSetup.apply(x)
        Downstream.apply(out).sum().backward()
        self.assertEqual(UseSetup.seen, torch.float32)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_validation(self):
        # [FUNCTION DOC] Check error handling of the output-gradient-dtype declaration: wrong
        # count of declarations, wrong type, a dtype for a non-tensor output, and a dtype for a
        # non-differentiable output each raise RuntimeError. Skipped under dynamo.
        x = torch.tensor([1.0, 2.0], requires_grad=True)

        class CountMismatch(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function declaring two dtypes for one output.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Declare two dtypes and return a clone.
                ctx.set_output_grad_dtype(torch.float32, torch.float32)
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return g

        with self.assertRaisesRegex(RuntimeError, "number of declarations to match"):
            CountMismatch.apply(x)

        class TooFewDeclarations(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function declaring one dtype for two outputs.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Declare one dtype and return two clones.
                ctx.set_output_grad_dtype(torch.float32)
                return x.clone(), x.clone()

            @staticmethod
            def backward(ctx, g0, g1):
                # [FUNCTION DOC] HELPER: Return the sum of both gradients.
                return g0 + g1

        with self.assertRaisesRegex(RuntimeError, "number of declarations to match"):
            TooFewDeclarations.apply(x)

        class BadDeclarationType(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function declaring a string instead of a dtype.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Declare a string and return a clone.
                ctx.set_output_grad_dtype("float32")
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return g

        with self.assertRaisesRegex(RuntimeError, "torch.dtype or None"):
            BadDeclarationType.apply(x)

        class DtypeForNonTensor(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function declaring a dtype for a string output.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Declare None and a dtype and return a clone and a string.
                ctx.set_output_grad_dtype(None, torch.float32)
                return x.clone(), "not a tensor"

            @staticmethod
            def backward(ctx, g, gs):
                # [FUNCTION DOC] HELPER: Return the first gradient.
                return g

        with self.assertRaisesRegex(RuntimeError, "not a differentiable tensor"):
            DtypeForNonTensor.apply(x)

        class DtypeForNonDifferentiable(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function declaring a dtype for an output
            # marked non-differentiable.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Mark the clone non-differentiable, declare a dtype and
                # return it.
                out = x.clone()
                ctx.mark_non_differentiable(out)
                ctx.set_output_grad_dtype(torch.float32)
                return out

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return g

        with self.assertRaisesRegex(RuntimeError, "not a differentiable tensor"):
            DtypeForNonDifferentiable.apply(x)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_called_twice(self):
        # [FUNCTION DOC] Check that declaring output gradient dtypes twice in one call raises
        # RuntimeError saying it can only be called once. Skipped under dynamo.
        # set_output_grad_dtype may be called at most once per invocation.
        class CalledTwice(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function that declares gradient dtypes twice in forward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Declare a dtype twice and return a clone.
                ctx.set_output_grad_dtype(torch.float32)
                ctx.set_output_grad_dtype(torch.float64)
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return g

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "can only be called once"):
            CalledTwice.apply(x)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_return_as_is_non_differentiable(self):
        # [FUNCTION DOC] Check that differentiability is judged on the raw returned output: an
        # input returned as is and marked non-differentiable does not require grad, and the other
        # output receives the declared dtype. Skipped under dynamo.
        # mark_non_differentiable records the raw output's TensorImpl, so
        # differentiability must be checked on the raw output, not its wrapper.
        class ReturnAsIs(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning a bfloat16 output and a non-differentiable
            # passthrough input.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Mark the second input non-differentiable, declare dtypes
                # and return the converted tensor and the input.
                out = x.to(torch.bfloat16)
                ctx.mark_non_differentiable(y)
                ctx.set_output_grad_dtype(torch.float32, None)
                return out, y

            @staticmethod
            def backward(ctx, g_out, g_y):
                # [FUNCTION DOC] HELPER: Record the dtype of the first gradient and return it as
                # float32 plus None.
                ReturnAsIs.seen_grad_dtype = g_out.dtype
                return g_out.to(torch.float32), None

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0])
        out, out_y = ReturnAsIs.apply(x, y)
        self.assertFalse(out_y.requires_grad)
        out.sum().backward()
        self.assertEqual(ReturnAsIs.seen_grad_dtype, torch.float32)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_ctx_output_grad_dtype_no_leak_from_returned_input(self):
        # [FUNCTION DOC] Check that a Function returning its input as is gets the default output
        # gradient dtype of the output, not the leaf's grad_dtype, while accumulation into the
        # leaf uses the leaf's grad_dtype. Skipped under dynamo.
        # The output defaults to its dtype while leaf accumulation uses x.grad_dtype.
        class ReturnInput(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function recording the dtype of the received gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input unchanged.
                return x

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Record the gradient dtype and return the gradient.
                ReturnInput.seen_grad_dtype = g.dtype
                return g

        x = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
        x.grad_dtype = torch.float64
        out = ReturnInput.apply(x)
        out.sum().backward()
        self.assertEqual(ReturnInput.seen_grad_dtype, torch.float32)
        self.assertEqual(x.grad.dtype, torch.float64)

    def test_gc_in_destructor(self):
        # [FUNCTION DOC] Regression check that a Function whose destructor triggers garbage
        # collection does not crash during repeated forward and backward calls.
        """
        Previously, if a Function destructor triggered a garbage collection,
        the Variable's tp_dealloc handler would get called twice leading to a
        segfault.
        """

        class CollectOnDelete(Function):
            # [CLASS DOC] HELPER: Legacy-style Function whose deletion runs a garbage collection.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            def backward(self, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return grad_output

            def __del__(self):
                # [FUNCTION DOC] HELPER: Run a garbage collection when the object is deleted.
                gc.collect()

        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    def test_naughty_autograd_function_attribute_access(self):
        # [FUNCTION DOC] Check a Function instantiated directly: a DeprecationWarning is emitted,
        # calling it raises RuntimeError about non-static forward, apply works, and several
        # attribute and method accesses on the instance succeed.
        class Id(Function):
            # [CLASS DOC] HELPER: Identity Function with static forward and backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, grad_x):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return grad_x

        with self.assertWarnsRegex(DeprecationWarning, "should not be instantiated"):
            f = Id()

        # After raising warning, should still return an instance
        self.assertIsInstance(f, Id)
        x = torch.zeros(1, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "non-static forward method is deprecated"
        ):
            f(x)
        t = Id.apply(x)
        self.assertEqual(t.grad_fn.name(), "IdBackward")

        # THPFunction always has a backing PyNode (cdata), so attribute
        # accesses on a directly-instantiated Function succeed. Previously,
        # cdata was a null weak ref (PyNode was only created during apply),
        # so these would raise "Attribute X is invalid".
        t = torch.ones(1, requires_grad=True)
        t._backward_hooks = {}
        f._register_hook_dict(t)
        f.register_hook(lambda x, y: None)
        f.next_functions  # empty tuple
        f.name()
        f._sequence_nr()
        f._set_sequence_nr(2)
        f._input_metadata  # empty tuple
        f.metadata

    def test_naughty_anomaly_access(self):
        # [FUNCTION DOC] Check metadata of a grad_fn is still accessible after the output tensor
        # is deleted.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Identity Function.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return g

        x = torch.zeros(1, requires_grad=True)
        y = MyFunction.apply(x)
        y.backward()
        y.grad_fn.metadata
        g = y.grad_fn
        del y
        self.assertIsNotNone(g.metadata)  # this used to fail

    def test_naughty_autograd_function_stashing_ctx(self):
        # [FUNCTION DOC] Check a ctx stashed during backward stays valid after the graph is freed
        # and its saved tensors remain accessible.
        saved_ctx = []

        class Id(Function):
            # [CLASS DOC] HELPER: Function saving its input and stashing ctx in an outer list
            # during backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it.
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                # [FUNCTION DOC] HELPER: Append ctx to the outer list and return its
                # saved tensors.
                saved_ctx.append(ctx)
                return ctx.saved_tensors

        p = torch.zeros(1, requires_grad=True)
        loss = Id.apply(p)
        loss.backward(retain_graph=True)
        del loss
        # THPFunction now always has a strong reference to its backing PyNode
        # (cdata), so the stashed ctx stays alive and saved_tensors is
        # accessible. Previously, cdata was a weak ref that expired after the
        # graph was freed, causing a RuntimeError here.
        self.assertEqual(saved_ctx[0].saved_tensors, (p,))

    @xfailIf(TEST_WITH_TORCHDYNAMO and sys.version_info < (3, 13))
    def test_custom_autograd_repeated_grad_grad(self):
        # [FUNCTION DOC] Check repeated double-backward calls give the same second derivative, for
        # a plain product and a custom Function version (regression from an old PR). Expected to
        # fail under dynamo before Python 3.13.
        # This test failed the equality check in PR #22983; it's an interesting
        # and different test case worth enshrining.  mult1 is not testing
        # anything that interesting, but mult2 is the interesting case.

        def mult1(x):
            # [FUNCTION DOC] HELPER: Product over the last two dimensions.
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function computing the double product, saving input and output.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Compute the product, save input and output and return
                # the output.
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient times output divided by the input,
                # broadcast over the last two dimensions.
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        def check_gradgrad_repeated(x, y):
            # [FUNCTION DOC] HELPER: Compute grad with create_graph twice and a second derivative
            # each time, asserting they agree.
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_1,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_2,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])

        x = torch.ones(2, 4, 4).requires_grad_()
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))

    def test_custom_autograd_no_early_free(self):
        # [FUNCTION DOC] Check that a Function saving its output does not free buffers early:
        # gradcheck and gradgradcheck pass for two implementations, and repeated grad calls work.
        # This test failed complaining that buffers had already been freed
        # prior to #22983.  Also pretty interesting test case.
        class Double(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function squaring its input and saving input and output.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Square the input, save input and output, return
                # the output.
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient times twice the saved input.
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            # [CLASS DOC] HELPER: Variant of the squaring Function whose backward uses the
            # saved output.
            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient times twice the saved output over the
                # saved input.
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        x = torch.tensor(2).double().requires_grad_()

        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))

        y = double(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)

        y = double2(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)  # should not error!

    def test_custom_autograd_ac_early_stop(self):
        # [FUNCTION DOC] Check non-reentrant checkpoint early stop with a custom Function: the
        # saved clone is freed after backward (weak references dead, garbage collection disabled).
        refs = []

        class Test(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving a clone of the input and recording a weak
            # reference to it.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Clone the input, save it, store a weak reference and
                # return it.
                y = x.clone()
                ctx.save_for_backward(y)
                refs.append(weakref.ref(y))
                return y

            @staticmethod
            def backward(ctx, *args):
                # [FUNCTION DOC] HELPER: Access the saved tensors and return None.
                _ = ctx.saved_tensors
                return None

        def fn(inp):
            # [FUNCTION DOC] HELPER: Apply the custom Function.
            return Test.apply(inp)

        inp = torch.randn(5, 5, requires_grad=True)

        def scope():
            # [FUNCTION DOC] HELPER: Run non-reentrant checkpoint over the function and backward.
            # Early-stop is true by default in non-reentrant torch.utils.checkpoint
            out = torch.utils.checkpoint.checkpoint(fn, inp, use_reentrant=False)
            out.sum().backward()

        with disable_gc():
            scope()

            for ref in refs:
                self.assertIsNone(ref())

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_compile_no_recompile(self):
        # [FUNCTION DOC] Check that checkpointed compile with a changed default device does not
        # recompile (recompile set to error), requires CUDA; when an ambient function mode is
        # active the run is expected to raise RuntimeError.
        # Check for ambient TorchFunctionMode, e.g. when PYTORCH_TEST_WITH_CROSSREF=1
        expect_fail = len(torch.overrides._get_current_function_mode_stack()) > 0

        @torch.compile(backend="aot_eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager backend) function computing sin then cos.
            return x.sin().cos()

        x = torch.rand(10, 10, requires_grad=True)

        def run():
            # [FUNCTION DOC] HELPER: Run checkpoint and backward with the default device, reset
            # dynamo, then run again with the default device set to cuda, restoring the prior
            # default device afterwards.
            out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
            out.sum().backward()

            torch._dynamo.reset()

            prev = torch.get_default_device()
            had_default_device = (
                getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
                is not None
            )
            try:
                # Using torch.device("cuda") directly doesn't work here because
                # it has some issues. In particular, unlike set_default_device or
                # invoking the TorchFunctionMode directly, it doesn't update the
                # global state dynamo references for guards:
                # torch.utils._device.CURRENT_DEVICE
                torch.set_default_device("cuda")
                out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
                out.sum().backward()
            finally:
                # None clears the DeviceContext mode; "cpu" would leak it.
                torch.set_default_device(prev if had_default_device else None)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            if expect_fail:
                with self.assertRaises(RuntimeError):
                    run()
            else:
                run()

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_device_context_fn(self):
        # [FUNCTION DOC] Check that checkpoint with a context function setting the default device
        # to cuda works under backward. Requires CUDA.
        @contextlib.contextmanager
        def apply_device(device):
            # [FUNCTION DOC] HELPER: Context manager that sets the default device and restores the
            # previous default (or clears it) on exit.
            prev = torch.get_default_device()
            had_default_device = (
                getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
                is not None
            )
            try:
                torch.set_default_device(device)
                yield
            finally:
                # None clears the DeviceContext mode; "cpu" would leak it.
                torch.set_default_device(prev if had_default_device else None)

        def context_fn():
            # [FUNCTION DOC] HELPER: Return a pair of contexts: a null context for forward and the
            # cuda device context for recompute.
            return contextlib.nullcontext(), apply_device("cuda")

        def fn(x):
            # [FUNCTION DOC] HELPER: Compute sin then cos.
            return x.sin().cos()

        with apply_device("cuda"):
            a = torch.tensor(1.0, requires_grad=True)
            out = torch.utils.checkpoint.checkpoint(
                fn, a, context_fn=context_fn, use_reentrant=False
            )
            out.backward()

    def test_detach(self):
        # [FUNCTION DOC] Check detach and in-place detach: results do not require grad, gradients
        # do not flow through them, and in-place detach on a view raises RuntimeError.
        x = torch.randn(10, 10, requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(x.grad, torch.ones(10, 10))

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)

        # in-place detach on a view raises an exception
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, "view", lambda: view.detach_())

    def test_detach_base(self):
        # [FUNCTION DOC] Check that detaching the base in place does not detach an existing view.
        "detaching base does not detach view"
        x = torch.randn(10, 10, requires_grad=True)
        view = x.narrow(0, 1, 4)
        x.detach_()
        self.assertFalse(x.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNotNone(view.grad_fn)
        self.assertIs(view._base, x)

    def test_detach_then_inplace_raises_in_autograd(self):
        # [FUNCTION DOC] Check that detaching and then zeroing in place a tensor saved for
        # backward makes backward raise RuntimeError about modification by an inplace operation.
        x = torch.randn([], requires_grad=True)
        orig_x = x.detach().clone()

        y = x**2  # saves x
        z = x.detach()
        z.zero_()
        with self.assertRaisesRegex(RuntimeError, "has been modified by an inplace"):
            y.backward()

    def _test_type_conversion_backward(self, t):
        # [FUNCTION DOC] HELPER: Check conversions between float and double with a type-conversion
        # function applied, verifying gradients are ones with the same tensor type.
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad), type(fvar))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

    def test_type_conversions(self):
        # [FUNCTION DOC] Check type conversion APIs (type, type_as, dtype argument, cuda and cpu
        # moves, device index, and data pointer sharing) and backward through conversions on CPU
        # and, when available, CUDA.
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIsInstance(x.float().cuda(), torch.cuda.FloatTensor)
            self.assertIsInstance(x.int().cuda(), torch.cuda.IntTensor)
            self.assertIsInstance(x.int().cuda().cpu(), torch.IntTensor)
            if torch.cuda.device_count() >= 2:
                x2 = x.float().cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().get_device(), 1)

        for t in [
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.IntTensor,
            torch.ByteTensor,
        ]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                # TODO: t.dtype should work
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                if torch.cuda.is_available():
                    for x_cuda in (True, False):
                        for y_cuda in (True, False):
                            x_c = x.cuda() if x_cuda else x
                            y_c = y.cuda() if y_cuda else y
                            _, y_type = y_c.type().rsplit(".", 1)
                            y_typestr = ("torch.cuda." if y_cuda else "torch.") + y_type
                            self.assertEqual(y_c.type(), x_c.type(y_typestr).type())
                            self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                            self.assertEqual(
                                y_c.data_ptr(),
                                y_c.cuda().data_ptr() if y_cuda else y_c.data_ptr(),
                            )

        self._test_type_conversion_backward(lambda x: x)
        if torch.cuda.is_available():
            self._test_type_conversion_backward(lambda x: x.cuda())
            if torch.cuda.device_count() >= 2:
                # one of these has to be the non-default device
                self._test_type_conversion_backward(lambda x: x.cuda(0))
                self._test_type_conversion_backward(lambda x: x.cuda(1))

    def test_isolated_node(self):
        # [FUNCTION DOC] Check backward when part of the graph (argmax index conversion) has no
        # gradient path.
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        b = torch.max(a, 1, True)[1].repeat(1, 5).double()
        o = (b + a).sum()
        o.backward()

    def test_shape(self):
        # [FUNCTION DOC] Check the shape attribute length and entries.
        x = torch.randn(3, 4)
        self.assertEqual(2, len(x.shape))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[1], 4)

    def test_numpy_requires_grad(self):
        # [FUNCTION DOC] Check numpy conversion of a tensor requiring grad raises RuntimeError
        # recommending detach, but works under no_grad or when no grad is required.
        x = torch.randn(2, 2, requires_grad=True)
        err_msg_outputs = r"Can't call numpy\(\) on Tensor that requires grad. Use tensor.detach\(\).numpy\(\) instead."
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()

        with torch.no_grad():
            x.numpy()

        x = torch.randn(2, 2)
        x.numpy()

        with torch.no_grad():
            x.numpy()

    def test_return_leaf(self):
        # [FUNCTION DOC] Check a Function returning its input and a derived tensor: a hook on
        # the returned leaf alias only sees the gradient from its own use, and leaf grads
        # accumulate correctly.
        class Identity(Function):
            # [CLASS DOC] HELPER: Function returning its first input and the sum of both inputs.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Return the first input and the sum.
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # [FUNCTION DOC] HELPER: Return the sum of gradients for the first input and the
                # second gradient for the second.
                return grad_a + grad_b, grad_b

        hook_called = [False]
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Identity.apply(x, y)

        # Make sure hooks only receive grad from usage of q, not x.
        def hook(grad):
            # [FUNCTION DOC] HELPER: Hook recording a call and asserting the gradient is ones.
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))

        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        # [FUNCTION DOC] Check an in-place Function on a tensor not requiring grad: the output is
        # the same object, gets the custom backward class and requires grad from the other input.
        class Inplace(InplaceFunction):
            # [CLASS DOC] HELPER: In-place Function adding its second input to the first.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Mark the first input dirty and return the in-place sum
                # and another derived tensor.
                ctx.mark_dirty(a)
                return a.add_(b), b + 2

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # [FUNCTION DOC] HELPER: Return the first gradient and the sum of both gradients.
                return grad_a, grad_a + grad_b

        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Inplace.apply(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn.__class__, Inplace._backward_cls)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        # [FUNCTION DOC] Check assigning grad-requiring tensors into a tensor not requiring grad
        # makes it require grad and backward routes gradients to the sources.
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad, torch.ones(5))
        self.assertEqual(z.grad, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        # [FUNCTION DOC] Check assignment under no_grad does not change requires_grad or create
        # a grad_fn.
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5)
        with torch.no_grad():
            x[0] = y

        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        # [FUNCTION DOC] Check that an in-place update under no_grad still bumps the version so
        # backward raises RuntimeError about in-place modification.
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        z = (x * y).sum()
        with torch.no_grad():
            x *= 2
        self.assertRaisesRegex(
            RuntimeError, "modified by an inplace operation", lambda: z.backward()
        )

    def test_inplace_version_error_shows_forward_op_name(self):
        # [FUNCTION DOC] Check version mismatch error messages name the forward op (Add, Sum1, or
        # the custom Function name) instead of the backward node name.
        # The error message should refer to the forward op (e.g. "Add"),
        # not the backward node (e.g. "AddBackward0").
        # Use b * b (not b * 2) so MulBackward saves b and detects the
        # version mismatch when unpacking.

        a = torch.randn(5, requires_grad=True)
        b = a + 1
        c = b * b
        with torch.no_grad():
            b += 1
        with self.assertRaisesRegex(RuntimeError, r"output 0 of Add,"):
            c.backward(torch.ones(5))

        # sum(dim=...) uses SumBackward1; the non-zero variant number is
        # preserved so the message says "Sum1" not "Sum".
        a = torch.randn(3, 4, requires_grad=True)
        b = a.sum(dim=1)
        c = b * b
        with torch.no_grad():
            b += 1
        with self.assertRaisesRegex(RuntimeError, r"output 0 of Sum1,"):
            c.backward(torch.ones(3))

        # Custom autograd Function: "MyFunc" not "MyFuncBackward".
        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving its input and multiplying the gradient by it.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return a clone.
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient times the saved input.
                (x,) = ctx.saved_tensors
                return grad * x

        a = torch.randn(5, requires_grad=True)
        b = MyFunc.apply(a)
        c = b * b
        with torch.no_grad():
            b += 1
        with self.assertRaisesRegex(RuntimeError, r"output 0 of MyFunc,"):
            c.backward(torch.ones(5))

    def test_increment_version(self):
        # [FUNCTION DOC] Check increment_version bumps the version counter by one for tensors
        # requiring grad and for integer tensors, and does not error inside or outside
        # inference mode.
        a = torch.rand(5, requires_grad=True)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)

        a = torch.zeros(5, dtype=torch.int)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)

        with torch.inference_mode():
            a = torch.rand(5, requires_grad=True)
            # does not error
            torch.autograd.graph.increment_version(a)

        # does not error
        torch.autograd.graph.increment_version(a)

    def test_no_grad_input(self):
        # [FUNCTION DOC] Check a Function applied under no_grad on an input requiring grad has no
        # grad_fn while the input keeps requires_grad.
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Identity Function.
            @staticmethod
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(self, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return grad_output

        x = torch.randn(5, requires_grad=True)
        with torch.no_grad():
            y = MyFunction.apply(x)

        self.assertTrue(x.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_copy(self):
        # [FUNCTION DOC] Regression check for the engine's gradient buffer copying: a gradient
        # shared by three consumers across a long branch must be accumulated without in-place
        # corruption, giving 34 for x and 17 for y.
        # This tests checks backward engine for a very subtle bug that appeared
        # in one of the initial versions of autograd. Gradients tensors were
        # simply stored in lists while the function waited for all its gradients
        # to be computed. However, sometimes an output was used multiple times,
        # so the gradients needed to be summed. Engine used to keep a need_copy
        # set of tensors that will need a clone upon next addition and removed
        # them from the set as soon as the clone was performed. However, this
        # could lead to incorrect results if the same gradient tensor was
        # buffered in three places in the graph:
        # 1. When accumulating gradients in one of these places it was cloned
        #    and removed from need_copy set.
        # 2. When accumulating in second place, it wasn't in the need_copy set,
        #    so the gradients were simply accumulated in-place (which already
        #    modified the grad in 3rd place)
        # 3. When accumulating in the third place, it wasn't in the need_copy set
        #    as well, so the incoming gradient was summed in-place, yielding
        #    incorrect results in all functions, except the first one.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        # Simulate that we're in the middle of the graph
        a = x + 2
        b = y + 2
        c = x + 2
        # This op will just return grad_output two times in backward
        add1 = a + b
        add2 = add1 + c
        # Simulate a long branch, so grad_output will get buffered.
        for _ in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        # expected gradients are:
        # for x: 34 (16 from final a, 16 from final c, 2 from add2)
        # for y: 17 (16 from final b, 1 from add2)
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        # [FUNCTION DOC] Check save_for_backward accepts None entries and they come back as None
        # from saved_tensors in backward.
        test_case = self

        class MyFn(Function):
            # [CLASS DOC] HELPER: Function squaring its input that saves two None values around
            # the input.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Save None, the input and None; return the input squared.
                ctx.save_for_backward(None, input, None)
                return input * input

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Assert the first and last saved values are None and
                # return the gradient for the input.
                n1, input, n2 = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        # [FUNCTION DOC] Check that a backward returning more gradients than inputs (extra values
        # are None) is accepted.
        class MyFn(Function):
            # [CLASS DOC] HELPER: Identity Function whose backward returns extra None gradients.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Return the input.
                return input

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient followed by two None values.
                return grad_output, None, None

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_pickle(self):
        # [FUNCTION DOC] Check pickling tensors with and without requires_grad for protocols zero
        # to two preserves values and the requires_grad flag.
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            # [FUNCTION DOC] HELPER: Assert two tensors are equal and have equal
            # requires_grad flags.
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_dep_nograd(self):
        # [FUNCTION DOC] Check that a node depending on a non-differentiable output of another
        # node still gets gradients through the differentiable path. Skipped under dynamo.
        class F1(Function):
            # [CLASS DOC] HELPER: Function returning its input and a random tensor
            # marked non-differentiable.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Return the input and a non-differentiable random tensor.
                out = torch.randn(input.size())
                ctx.mark_non_differentiable(out)
                return input, out

            @staticmethod
            def backward(ctx, grad_output, ignored):
                # [FUNCTION DOC] HELPER: Return the first gradient and ignore the second.
                return grad_output

        class F2(Function):
            # [CLASS DOC] HELPER: Function taking two inputs and returning the first.
            @staticmethod
            def forward(ctx, input, ignored):
                # [FUNCTION DOC] HELPER: Return the first input.
                return input

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient for the first input and None.
                return grad_output, None

        x = torch.randn(5, requires_grad=True)
        a, b = F1.apply(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2.apply(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad, torch.ones(x.size()))

    def test_set_grad_enabled(self):
        # [FUNCTION DOC] Check set_grad_enabled as a context manager, including that calling it as
        # a function inside a disabled region enables grad globally.
        x = torch.tensor([1.0], requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)

    def test_set_grad_enabled_wraps(self):
        # [FUNCTION DOC] Check set_grad_enabled as a decorator or wrapper of a function does not
        # change the global grad mode at construction after wrapping, and the wrapped function
        # runs without grad.
        for decorator in [True, False]:
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())

                if decorator:
                    # This should not mutate the global grad mode!
                    @torch.set_grad_enabled(False)
                    def inner_func(x):
                        # [FUNCTION DOC] HELPER: Sine function decorated to run with grad disabled
                        # through set_grad_enabled.
                        return x.sin()

                else:

                    def inner_func(x):
                        # [FUNCTION DOC] HELPER: Sine function to be wrapped by the object
                        # returned from set_grad_enabled.
                        return x.sin()

                    # This is non-idiomatic usage!
                    # More idiomatic usage: torch.set_grad_enabled(False)(inner_func)
                    obj = torch.set_grad_enabled(False)
                    self.assertTrue(not torch.is_grad_enabled())

                    # this will consume the set_grad_enabled global mutation!
                    inner_func = obj(inner_func)
                    self.assertTrue(torch.is_grad_enabled())

                self.assertTrue(torch.is_grad_enabled())

                x = torch.zeros(1, requires_grad=True)
                self.assertTrue(not inner_func(x).requires_grad)

    def test_simple_reentrant(self):
        # [FUNCTION DOC] Check a reentrant custom Function: backward itself runs a nested backward
        # on a graph built in forward, and the outer gradient equals the saved y data.
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            # [CLASS DOC] HELPER: Function whose forward builds a graph with grad enabled and
            # whose backward calls backward on it.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Build a small differentiable graph on ctx and return its
                # detached output.
                with torch.enable_grad():
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Run backward on the stored graph and return the stored
                # input grad times the incoming gradient.
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        # Reentrant starts on CPU thread, finishes on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        # [FUNCTION DOC] Check an error raised in a reentrant child backward propagates to the
        # parent backward call.
        # Parent graph.
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a

        # Reentrant child graph.
        b = torch.rand(3, 3, requires_grad=True)
        e = b * b
        f = TestAutograd.SimulateBackwardError.apply(e)
        reentrant_root = f.sum()

        class ReentrantFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward runs a failing reentrant backward
            # through the outer scope's root.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Return a clone of the input.
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Run backward on the captured reentrant root (which
                # raises) and return the gradient.
                # Reentrant backward in child will throw an error.
                reentrant_root.backward()
                return grad

        d = ReentrantFunc.apply(c)
        with self.assertRaisesRegex(Exception, "Simulate error"):
            d.sum().backward()

    def test_var_mean_differentiable(self):
        # [FUNCTION DOC] Check var_mean gradients match separate var and mean calls on a
        # six-dimensional tensor with two reduced dims.
        dim = [2, 4]
        keepdim = False
        input1 = torch.randn(3, 4, 5, 6, 2, 3, requires_grad=True)
        input2 = deepcopy(input1)
        var1, mean1 = torch.var_mean(input1, dim=dim, keepdim=keepdim)
        var2 = input2.var(dim=dim, keepdim=keepdim)
        mean2 = input2.mean(dim=dim, keepdim=keepdim)
        grad = torch.randn(3, 4, 6, 3, requires_grad=True)

        r1 = var1 * var1 * mean1 * mean1
        r2 = var2 * var2 * mean2 * mean2
        self.assertEqual(r1, r2, rtol=0.01, atol=0.0)

        torch.autograd.backward(r1, grad)
        torch.autograd.backward(r2, grad)
        self.assertEqual(input1.grad, input2.grad, rtol=0.01, atol=0.0)

    @skipIfSlowGradcheckEnv
    @skipIfNoLapack
    def test_lobpcg(self):
        # [FUNCTION DOC] Check lobpcg gradients (gradcheck and gradgradcheck) for several batch
        # shapes, k values and largest flag, and that the gradient of A is symmetric. Skipped for
        # slow gradcheck environments and when LAPACK is missing.
        def func(k, A, largest=True, B=None):
            # [FUNCTION DOC] HELPER: Call lobpcg with an identity-based initial subspace and
            # normalize each eigenvector column by the sign of its largest element to remove sign
            # non-determinism; return eigenvalues and vectors.
            X_shape = list(A.shape)
            X_shape[-1] = k
            X = torch.eye(A.size(-2), k, dtype=A.dtype, device=A.device)
            if A.dim() > 2:
                X = X.expand(X_shape)

            D, U = torch.lobpcg(A=A, k=k, B=B, X=X, largest=largest)

            # LOBPCG uses a random initial eigenspace approximation
            # if parameter `X` is not provided.
            # This may cause a non-deterministic behavior
            # when it comes to the sign of an eigenvector
            # (note if v is an eigenvector, so is -v),
            # hence we eliminate this non-determinism
            # by making sure that each column of U
            # gets multiplied by the sign of its max (in absolute value) element.
            # Also, gradcheck changes the content of the input by +/- eps (default to 1e-06)
            # to compute the numerical gradient which can also cause the signs to flip.
            _, idx = U.abs().max(-2, keepdim=True)
            sign = U.gather(-2, idx).sign()
            U = U * sign
            return D, U

        # TODO: review if this can be ported to OpInfos or moved to test_linalg.py
        def run_symeig_test(k, sizes, largest=True):
            # [FUNCTION DOC] HELPER: Build a random symmetric positive matrix of the given sizes
            # and run gradcheck, gradgradcheck with custom grad vectors, and the symmetry check on
            # the gradient.
            A = torch.rand(*sizes).double()
            A = (A @ A.mT) / 10
            A.requires_grad_(True)

            gradcheck(lambda A: func(k, A, largest), A, check_batched_grad=False)

            # Custom gradient vectors for better stability due to some
            # non-determinism in the lobpcg's forward.
            # Note it is not required if symeig is in forward instead (tested).
            D_grad = torch.rand(*A.shape[:-2], k) / 100
            U_grad = torch.rand(*A.shape[:-1], k) / 100
            gradgradcheck(
                lambda A: func(k, A, largest),
                A,
                [D_grad, U_grad],
                atol=1e-4,
                check_batched_grad=False,
            )

            # check whether A.grad is symmetric
            A = A.detach().requires_grad_(True)
            D, U = func(k, A, largest)
            (D.sum() + U.sum()).backward()
            self.assertEqual(A.grad, A.grad.mT)

        for largest in [True, False]:
            run_symeig_test(1, (6, 6), largest=largest)
            run_symeig_test(1, (2, 6, 6), largest=largest)
            run_symeig_test(1, (2, 2, 6, 6), largest=largest)
            run_symeig_test(2, (6, 6), largest=largest)
            run_symeig_test(2, (2, 6, 6), largest=largest)
            run_symeig_test(2, (2, 2, 6, 6), largest=largest)
            run_symeig_test(3, (9, 9), largest=largest)
            run_symeig_test(3, (2, 9, 9), largest=largest)
            run_symeig_test(3, (2, 2, 9, 9), largest=largest)

    def test_variable_traverse(self):
        # [FUNCTION DOC] Regression check that a reference cycle holding an intermediate Variable
        # does not cause crashes after garbage collection and backward.
        def get_out_and_unrefed_cycle():
            # [FUNCTION DOC] HELPER: Build chained views of a leaf, put an intermediate into a
            # self-referencing list, and return the final view.
            inp = torch.randn(10, requires_grad=True)
            tmp = inp.view(10, 1)
            out = tmp.view(10)

            # Create a reference cycle that contains an
            # intermediary Variable in the graph
            my_list = []
            my_list.append(tmp)
            my_list.append(my_list)

            return out

        out = get_out_and_unrefed_cycle()
        gc.collect()
        # This will segfault if things have been erroneously released
        out.backward(torch.randn(out.size()))

    # TODO: review porting these to OpInfo tests
    def test_pow_zero_tensor_gradient(self):
        # [FUNCTION DOC] Check that the gradient of pow at an all-zero input is zero for tensor
        # exponents and a scalar exponent zero.
        def run_test(input_size, exponent):
            # [FUNCTION DOC] HELPER: Backward through pow of a zero input with the given exponent
            # and assert the gradient sums to zero.
            input = torch.zeros(*input_size, requires_grad=True)
            input.pow(exponent).sum().backward()
            self.assertEqual(input.grad.abs().sum(), 0)

        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_node_ordering_when_none_returned(self):
        # [FUNCTION DOC] Check that a tensor hook on a weight runs before the earlier node's
        # backward when a custom Function returns None for that weight's gradient. Requires CUDA.
        class Matmul(torch.autograd.Function):
            # [CLASS DOC] HELPER: Matrix multiply Function whose backward stores the weight
            # gradient on the weight and returns None for it.
            @staticmethod
            def forward(ctx, x, w):
                # [FUNCTION DOC] HELPER: Save both inputs and return their product.
                # x: [M, N]
                # w: [N, K]
                ctx.save_for_backward(x, w)
                return x @ w

            @staticmethod
            def backward(ctx, g_out):
                # [FUNCTION DOC] HELPER: Compute both gradients, attach the weight gradient in
                # float32 to the weight, and return the input gradient and None.
                # g_out: [M, K]
                x, w = ctx.saved_tensors
                g_x = g_out @ w.T
                g_w = x.T @ g_out
                w.main_grad = g_w.float()
                return g_x, None

        executed = []

        class HookFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function whose backward appends A to the outer
            # execution log.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Log A and return the gradient.
                executed.append("A")
                return g

        def hook(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Tensor hook appending B to the execution log.
            executed.append("B")

        x = torch.randn((3, 3), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x = HookFunction.apply(x)
        w = torch.randn((3, 3), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w.register_hook(hook)
        o = Matmul.apply(x, w)
        o.sum().backward()

        self.assertEqual(executed, ["B", "A"])

    def test_current_graph_task_id(self):
        # [FUNCTION DOC] Check the private current graph task id is incremented by one for each
        # backward, visible in a hook, and is minus one outside backward.
        id = [-1]

        def hook(_):
            # [FUNCTION DOC] HELPER: Hook storing the current graph task id.
            id[0] = torch._C._current_graph_task_id()

        t = torch.tensor(1.0, requires_grad=True).clone()
        t.register_hook(hook)

        t.backward(retain_graph=True)
        base = id[0]
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 1)
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 2)

        self.assertEqual(torch._C._current_graph_task_id(), -1)

    @skipIfTorchDynamo(
        "_current_graph_task_execution_order requires active backward pass"
    )
    def test_current_graph_task_execution_order(self):
        # [FUNCTION DOC] Check the private query for the predicted node execution order inside a
        # backward with multithreading disabled: exact node names for single path, multiple roots,
        # and subgraph cases; agreement with the actual hook order for others; errors outside
        # backward or with multithreading enabled. Skipped under dynamo.
        predicted = [None]
        all_hooks = []

        def hook(_):
            # [FUNCTION DOC] HELPER: Hook recording the predicted execution order.
            predicted[0] = torch._C._current_graph_task_execution_order()

        def names(nodes):
            # [FUNCTION DOC] HELPER: Format node names, taking the last word of each name, as a
            # comma-separated line.
            return ", ".join([node.name().split(" ")[-1] for node in nodes]) + "\n"

        def grad_fns(*tensors):
            # [FUNCTION DOC] HELPER: Map tensors to their grad_fn, or to the AccumulateGrad node
            # for leaves requiring grad.
            # or grad accumulator
            out = []
            for t in tensors:
                if t.requires_grad and t.grad_fn is None:
                    out.append(t.clone().grad_fn.next_functions[0][0])
                else:
                    out.append(t.grad_fn)
            return out

        actual = []

        def register_logging_hooks(*tensors):
            # [FUNCTION DOC] HELPER: Register hooks on all tensors that append them to the list of
            # actual execution order.
            # register hooks that log the order in which they are called
            def get_hook(i):
                # [FUNCTION DOC] HELPER: Make a hook that logs the tensor at the given position.
                def hook(t_):
                    # [FUNCTION DOC] HELPER: Hook appending the tensor to the actual order list.
                    actual.append(tensors[i])

                return hook

            for i, t in enumerate(tensors):
                all_hooks.append(t.register_hook(get_hook(i)))

        # Basic example: single path
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        all_hooks.append(t.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            t.backward()
        self.assertExpectedInline(
            names(predicted[0]),
            """\
ExpBackward0, SinBackward0, CloneBackward0, torch::autograd::AccumulateGrad
""",
        )

        # We don't exactly follow sequence_nr order
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        c = b.sin()
        d = a.cos()
        out = c * d
        register_logging_hooks(a, b, c, d, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Accumulate grad node has more than one input
        a = torch.tensor(1.0, requires_grad=True)
        b = a.sin()
        c = a.cos()
        out = b * c
        register_logging_hooks(a, b, c, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Multiple roots are also OK
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        out2 = b.cos()
        out3 = b.cos()
        register_logging_hooks(a, b, out, out2, out3)
        all_hooks.append(out3.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out, out3, out2), inputs=(a,))
        self.assertExpectedInline(
            names(predicted[0]),
            """\
CosBackward0, CosBackward0, SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where next node is nullptr
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where two `inputs` on the same path
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a, b))
        self.assertEqual(
            names(predicted[0]),
            """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where `inputs` specifies a subgraph
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        c = a * b
        out = c.sin()
        register_logging_hooks(a, b, c, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a,))
        self.assertEqual(
            names(predicted[0]),
            """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Errors when not called in a backward
        with self.assertRaisesRegex(
            RuntimeError, "should only be called during the backward pass"
        ):
            torch._C._current_graph_task_execution_order()

        # Errors when context manager not enabled
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        all_hooks.append(t.register_hook(hook))
        with self.assertRaisesRegex(
            RuntimeError,
            "expects the current backward to be executed with multithreading disabled",
        ):
            t.backward()

        # Avoid leaking memory
        for h in all_hooks:
            h.remove()

    @skipIfWindows(msg="node name demangling inconsistent on windows")
    def test_backward_hook_relative_ordering(self):
        # [FUNCTION DOC] Check the relative order of node pre hooks, node post hooks, tensor
        # hooks, post-accumulate-grad hooks and a module full backward hook across a linear layer
        # backward, against an exact expected list. Skipped on Windows (name demangling differs).
        order = []

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a linear layer of size ten.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the linear layer.
                return self.linear(x)

        x = torch.randn(10, 10, requires_grad=True)
        module = MyModule()
        module.register_full_backward_hook(
            lambda _1, _2, _3: order.append(
                "module_full_backward_hook_BackwardHookFunctionBackward0"
            )
        )

        def make_pre_hook(id):
            # [FUNCTION DOC] HELPER: Return a node pre hook appending a labeled entry to the
            # order list.
            return lambda _: order.append(f"pre_hook_{id}")

        def make_post_hook(id):
            # [FUNCTION DOC] HELPER: Return a node post hook appending a labeled entry to the
            # order list.
            return lambda _1, _2: order.append(f"post_hook_{id}")

        count = 0

        def register_hooks_on_all_nodes(nodes):
            # [FUNCTION DOC] HELPER: Recursively register pre and post hooks on every node
            # reachable via next_functions, labeling each with node name and running count.
            nonlocal count
            for node, _ in nodes:
                count += 1
                id = f"{node.name()}_{count}"
                node.register_prehook(make_pre_hook(id))
                node.register_hook(make_post_hook(id))
                register_hooks_on_all_nodes(node.next_functions)

        loss = module(x).sum()
        register_hooks_on_all_nodes(((loss.grad_fn, None),))

        def make_tensor_pre_hook(id):
            # [FUNCTION DOC] HELPER: Return a tensor hook logging a labeled entry.
            return lambda _: order.append(f"tensor_pre_hook_{id}")

        def make_post_acc_grad_hook(id):
            # [FUNCTION DOC] HELPER: Return a post-accumulate-grad hook logging a labeled entry.
            return lambda _: order.append(f"post_acc_grad_hook_{id}")

        x.register_hook(make_tensor_pre_hook("x"))
        module.linear.weight.register_hook(make_tensor_pre_hook("weight"))
        module.linear.bias.register_hook(make_tensor_pre_hook("bias"))

        x.register_post_accumulate_grad_hook(make_post_acc_grad_hook("x"))
        module.linear.weight.register_post_accumulate_grad_hook(
            make_post_acc_grad_hook("weight")
        )
        module.linear.bias.register_post_accumulate_grad_hook(
            make_post_acc_grad_hook("bias")
        )

        loss.backward()

        expected_order = [
            "pre_hook_SumBackward0_1",
            "post_hook_SumBackward0_1",
            "pre_hook_BackwardHookFunctionBackward_2",
            "post_hook_BackwardHookFunctionBackward_2",
            "pre_hook_AddmmBackward0_3",
            "post_hook_AddmmBackward0_3",
            "tensor_pre_hook_bias",
            "pre_hook_torch::autograd::AccumulateGrad_4",
            "post_acc_grad_hook_bias",
            "post_hook_torch::autograd::AccumulateGrad_4",
            "pre_hook_TBackward0_7",
            "post_hook_TBackward0_7",
            "tensor_pre_hook_weight",
            "pre_hook_torch::autograd::AccumulateGrad_8",
            "post_acc_grad_hook_weight",
            "post_hook_torch::autograd::AccumulateGrad_8",
            "pre_hook_BackwardHookFunctionBackward_5",
            "module_full_backward_hook_BackwardHookFunctionBackward0",
            "post_hook_BackwardHookFunctionBackward_5",
            "tensor_pre_hook_x",
            "pre_hook_torch::autograd::AccumulateGrad_6",
            "post_acc_grad_hook_x",
            "post_hook_torch::autograd::AccumulateGrad_6",
        ]

        self.assertEqual(len(expected_order), len(order))
        for expected, actual in zip(expected_order, order):
            self.assertEqual(expected, actual)

    def test_view_replay_enabled(self):
        # [FUNCTION DOC] Check the private view replay setting as a context manager and as a
        # function or decorator: with replay enabled in-place ops on views record view
        # backward nodes, otherwise as-strided backward; state is restored on exit and for the
        # decorator form.
        def f(x):
            # [FUNCTION DOC] HELPER: Clone, flatten into a view, add one in place and return
            # the view.
            out = x.clone().view(-1)
            # mutate the view, triggering autograd view-replay logic
            out.add_(1)
            return out

        x = torch.ones(2, 2, requires_grad=True)

        # Test as a context manager
        with torch.autograd._force_original_view_tracking(False):
            out = f(x)
            self.assertTrue("AsStridedBackward" in str(out.grad_fn))
            self.assertFalse(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        with torch.autograd._force_original_view_tracking(True):
            out = f(x)
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        with torch.autograd._force_original_view_tracking(False):
            torch.autograd._force_original_view_tracking(True)
            out = f(x)
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        prev = torch.autograd.is_view_replay_enabled()
        ctx = torch.autograd._force_original_view_tracking(not prev)
        # Construction eagerly sets state (function-form behavior).
        self.assertEqual(torch.autograd.is_view_replay_enabled(), not prev)
        with ctx:
            self.assertEqual(torch.autograd.is_view_replay_enabled(), not prev)
            out = f(x)
            self.assertTrue(
                ("ViewBackward" if not prev else "AsStridedBackward")
                in str(out.grad_fn)
            )
        self.assertEqual(torch.autograd.is_view_replay_enabled(), prev)

        # Test as a function
        torch.autograd._force_original_view_tracking(False)
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        torch.autograd._force_original_view_tracking(True)
        out = f(x)
        self.assertTrue("ViewBackward" in str(out.grad_fn))
        self.assertTrue(torch.autograd.is_view_replay_enabled())

        prev = torch.autograd.is_view_replay_enabled()

        @torch.autograd._force_original_view_tracking(not prev)
        def g(x):
            # [FUNCTION DOC] HELPER: Decorated wrapper calling the view helper.
            return f(x)

        # __call__ undoes the __init__ mutation, so ambient state is restored.
        self.assertEqual(torch.autograd.is_view_replay_enabled(), prev)
        out = g(x)
        self.assertTrue(
            ("ViewBackward" if not prev else "AsStridedBackward") in str(out.grad_fn)
        )
        self.assertEqual(torch.autograd.is_view_replay_enabled(), prev)

    def test_unsafe_set_version_counter(self):
        # [FUNCTION DOC] Check the unsafe version counter helpers: preserve-version-counter
        # context managers (single tensor and tuple) keep the counter unchanged on exit, setting
        # counters directly works, and a negative counter raises RuntimeError.
        x = torch.ones(2, requires_grad=True).clone()
        x.add_(1)
        x.add_(2)
        self.assertEqual(2, x._version)
        with torch.autograd._unsafe_preserve_version_counter(x):
            x.mul_(2)
            x.mul_(3)
        # version counter doesn't change inside of the context manager
        self.assertEqual(2, x._version)

        torch._C._autograd._unsafe_set_version_counter((x,), (0,))
        self.assertEqual(0, x._version)
        with self.assertRaisesRegex(RuntimeError, "Cannot set"):
            torch._C._autograd._unsafe_set_version_counter((x,), (-1,))

        y = torch.ones(2, requires_grad=True).clone()
        with torch.autograd._unsafe_preserve_version_counter((x, y)):
            x.mul_(2)
            y.mul_(3)
        # version counter doesn't change inside of the context manager
        self.assertEqual(0, x._version)
        self.assertEqual(0, y._version)

    def test_current_node(self):
        # [FUNCTION DOC] Check the private current autograd node is None in forward and shows the
        # running backward node name inside backward, using a dispatch mode that logs each op,
        # compared with an expected log.
        pr = []

        class MyMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode logging each op with the class name of the current
            # autograd node, then running the op.
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Record the op and current node class name, then call the
                # op with its arguments.
                node = torch._C._current_autograd_node()
                # Don't use node.name() here as it is not consistent on windows
                node_name = node.__class__.__name__ if node else "None"
                pr.append(f"Running {func} from within {node_name}")
                return func(*args, **(kwargs or {}))

        with MyMode():
            pr.append("FW")
            a = torch.rand(10, requires_grad=True)
            b = a.mul(2).div(3).sum()
            pr.append("BW")
            b.backward()
            pr.append("Done")

        self.assertExpectedInline(
            "\n".join(pr),
            """\
FW
Running aten.rand.default from within None
Running aten.mul.Tensor from within None
Running aten.div.Tensor from within None
Running aten.sum.default from within None
BW
Running aten.ones_like.default from within None
Running aten.expand.default from within SumBackward0
Running aten.div.Tensor from within DivBackward0
Running aten.mul.Tensor from within MulBackward0
Running aten.detach.default from within AccumulateGrad
Done""",
        )

    def test_profiler(self):
        # [FUNCTION DOC] Check the profiler is enabled inside its context and records multiply and
        # add events.
        x = torch.randn(10, 10)

        with profile(use_kineto=kineto_available()) as p:
            self.assertTrue(torch.autograd._profiler_enabled())
            y = x * 2 + 4

        self.assertFalse(torch.autograd._profiler_enabled())

        names = ["aten::mul", "aten::add"]
        found_indices = set()
        for evt in p.function_events:
            if evt.name in names:
                found_indices.add(names.index(evt.name))
        self.assertEqual(len(found_indices), len(names))

    def test_profiler_seq_nr(self):
        # [FUNCTION DOC] Check profiler sequence numbers: forward add and sum events and their
        # backward events share sequence numbers, with the engine evaluate event in between,
        # forward thread ids set, and nested empty ops have none. Prints the key averages table.
        with profile(use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = z.sum(dim=None)
            s.backward()
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        # expecting aten::add, aten::sum to have the sequence numbers,
        # expecting the corresponding backward nodes to have the same numbers
        # as the forward ops
        autograd_ops = {
            ("aten::add", "Add"): [],
            ("aten::sum", "Sum"): [],
        }
        accumulate_ops = []
        found_empty = False
        for e in p.function_events:
            for (fwd_name, bwd_name), ops in autograd_ops.items():
                if e.name == fwd_name or (bwd_name in e.name and "Backward" in e.name):
                    ops.append(e)

            if "AccumulateGrad" in e.name:
                accumulate_ops.append(e)

            # check that nested ops (e.g. empty) don't have
            # sequence number
            if e.name == "aten::empty":
                self.assertEqual(e.sequence_nr, -1)
                found_empty = True

        for idx, ((fwd_name, bwd_name), ops) in enumerate(autograd_ops.items()):
            self.assertEqual(len(ops), 3)
            self.assertEqual(ops[0].name, fwd_name)
            self.assertEqual(
                ops[1].name,
                f"autograd::engine::evaluate_function: {bwd_name}Backward{idx}",
            )
            self.assertEqual(ops[2].name, f"{bwd_name}Backward{idx}")
            self.assertGreaterEqual(ops[0].sequence_nr, 0)
            self.assertEqual(ops[1].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[2].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[0].fwd_thread, 0)
            self.assertEqual(ops[1].fwd_thread, ops[0].thread)
            self.assertEqual(ops[2].fwd_thread, ops[0].thread)
        self.assertTrue(found_empty)

    def test_profiler_unboxed_only(self):
        # [FUNCTION DOC] Smoke check profiling an op without boxed support (resize) does not fail.
        x = torch.rand(3, 4)

        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            x.resize_([3, 2])

    def test_profiler_propagation(self):
        # [FUNCTION DOC] Check profiler propagation across a traced fork and wait: events for the
        # record_function ranges in foo, bar and bar after wait are each found exactly once.
        def foo(x):
            # [FUNCTION DOC] HELPER: Traced function that doubles the input inside a named
            # record_function range.
            with record_function("in_foo") as rf:
                return x * 2

        x = torch.rand(3, 4)
        traced_foo = torch.jit.trace(foo, x)

        def bar(x):
            # [FUNCTION DOC] HELPER: Traced function that forks the first function, waits, then
            # doubles in a second named range.
            with record_function("in_bar") as rf:
                # we expect that profiler will be able
                # propagate across fork
                fut = torch.jit._fork(traced_foo, x)
                y = torch.jit._wait(fut)
                # note: continuation (and rf's end) can
                # be executed in a different thread
                with record_function("in_bar_after_wait") as rf2:
                    y = y * 2
                return y

        traced_bar = torch.jit.trace(bar, x)

        with profile(use_kineto=kineto_available()) as p:
            traced_bar(x)

        found_foo = False
        found_bar = False
        found_bar_after_wait = False
        for info in p.function_events:
            if info.name == "in_foo":
                self.assertFalse(found_foo)
                found_foo = True
            elif info.name == "in_bar":
                self.assertFalse(found_bar)
                found_bar = True
            elif info.name == "in_bar_after_wait":
                self.assertFalse(found_bar_after_wait)
                found_bar_after_wait = True
        self.assertTrue(found_foo)
        self.assertTrue(found_bar)
        self.assertTrue(found_bar_after_wait)

    def test_record_function_callbacks(self):
        # [FUNCTION DOC] Check a named record_function range appears exactly once as a
        # profiler event.
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            with record_function("foo"):
                y = x * 2 + 4

        function_events = p.function_events
        foo_event = next(event for event in function_events if "foo" in event.name)
        self.assertEqual(foo_event.count, 1)

    def test_record_function_legacy(self):
        # [FUNCTION DOC] Check the lower-level profiler record_function enter and exit operators
        # produce an event with the given name.
        # Test the new _record_function ops work
        # Note: Remove once record_function uses these directly
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            handle = torch.ops.profiler._record_function_enter("bar", None)
            try:
                y = x * 2 + 4
            finally:
                torch.ops.profiler._record_function_exit(handle)

        function_events = p.function_events
        foo_event = next(event for event in function_events if "bar" in event.name)
        self.assertEqual(foo_event.count, 1)

    def test_profiler_aggregation_fake(self):
        # [FUNCTION DOC] Check parent-child assignment of fake function events across two threads
        # with nested time ranges: only one child relationship is expected.
        events = EventList()
        id = [0]

        def get_id():
            # [FUNCTION DOC] HELPER: Return an increasing integer id.
            id[0] = id[0] + 1
            return id[0]

        # [[thread_id, [(start, end, id), ....]], ...]
        # Using list instead of a dict so order is guaranteed for any Python
        # version
        threads = [
            [1, [(0, 1, get_id()), (1, 2, get_id())]],
            [0, [(0, 2, get_id()), (1, 2, get_id()), (1, 3, get_id())]],
        ]
        for thread, ranges in threads:
            for range in ranges:
                if len(range) != 3:
                    raise AssertionError(f"expected len(range) == 3, got {len(range)}")
                events.append(
                    FunctionEvent(
                        id=range[2],
                        node_id=0,
                        name="",
                        thread=thread,
                        start_us=range[0],
                        end_us=range[1],
                    )
                )

        events._populate_cpu_children()

        # Note that [1, 3] pushes out [0, 2] first. Then we record [1, 2]
        # as a child of [1, 3]
        res = [[], [], [], [], [4]]

        def get_children_ids(event):
            # [FUNCTION DOC] HELPER: Return the ids of the CPU children of an event.
            return [child.id for child in event.cpu_children]

        if [get_children_ids(event) for event in events] != res:
            raise AssertionError("children ids mismatch")

    def test_profiler_aggregation_table(self):
        # [FUNCTION DOC] Check the table method of a profile returns the same text as its
        # string conversion.
        """
        Test if the profiling result is aggregated for `str(prof)`

        See: https://github.com/pytorch/pytorch/issues/37500
        """

        x = torch.randn(1024)
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            torch.einsum("i->", x)

        prof_str = str(prof)
        prof_table = prof.table()

        self.assertEqual(prof_table, prof_str)

    def test_profiler_function_event_avg(self):
        # [FUNCTION DOC] Check averaging of function events: count, total times and per-event
        # average after adding two events and the average to itself.
        avg = FunctionEventAvg()
        avg.add(
            FunctionEvent(id=0, node_id=0, name="foo", thread=0, start_us=10, end_us=15)
        )
        avg.add(
            FunctionEvent(id=1, node_id=0, name="foo", thread=0, start_us=20, end_us=30)
        )
        avg.add(avg)
        self.assertEqual(avg.key, "foo")

        # aggregate stats
        self.assertEqual(avg.count, 4)
        self.assertEqual(avg.cpu_time_total, 30)
        self.assertEqual(avg.self_cpu_time_total, 30)
        self.assertEqual(avg.device_time_total, 0)

        # average stats
        self.assertEqual(avg.cpu_time, 7.5)
        self.assertEqual(avg.device_time_total, 0)

    def test_profiler_shapes(self):
        # [FUNCTION DOC] Check profiler records input shapes for linear layers when record_shapes
        # is on.
        print()
        layer1 = torch.nn.Linear(20, 30)
        layer2 = torch.nn.Linear(30, 40)
        input = torch.randn(128, 20)
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            layer2(layer1(input))

        print(prof.function_events)

        linear_expected_shapes = [
            [[128, 20], [30, 20], [30]],
            [[128, 30], [40, 30], [40]],
        ]

        found_indices = set()
        for event in prof.function_events:
            if event.name == "aten::linear":
                self.assertTrue(event.input_shapes in linear_expected_shapes)
                found_indices.add(linear_expected_shapes.index(event.input_shapes))
        self.assertEqual(len(found_indices), len(linear_expected_shapes))

    def test_profiler_aggregation_lstm(self):
        # [FUNCTION DOC] Smoke test of the profiler with an LSTM: prints several tables, compares
        # total time with python timing and exports a chrome trace (except on Windows).
        print()
        rnn = torch.nn.LSTM(10, 20, 2)
        total_time_s = 0
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            for _ in range(20):
                input = torch.randn(5, 3, 10)
                h = torch.randn(2, 3, 20)
                c = torch.randn(2, 3, 20)
                start = time.time()
                rnn(input, (h, c))
                end = time.time()
                total_time_s += end - start

        print(prof.table(sort_by="self_cpu_time_total", row_limit=10, header="TEST"))
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10
            )
        )
        print(
            prof.table(
                sort_by="self_cpu_time_total",
                row_limit=10,
                max_src_column_width=300,
                header="TEST",
                top_level_events_only=True,
            )
        )
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10, top_level_events_only=True
            )
        )

        total_time_us = (
            total_time_s * 1000.0 * 1000.0
        )  # make it us which is profiler default
        print("Total time based on python measurements: ", _format_time(total_time_us))
        print(
            f"CPU time measurement python side overhead: {(total_time_us / prof.self_cpu_time_total - 1.0) * 100.0:.2f}%"
        )

        if sys.platform != "win32":
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)

    def test_record_function(self):
        # [FUNCTION DOC] Check record_function ranges nest around the expected ops in order, and
        # that it also works as a decorator.
        x = torch.randn(10, 10)

        def forward(x):
            # [FUNCTION DOC] HELPER: Run arithmetic ops in nested outer and inner ranges,
            # then divide.
            with record_function("outer"):
                y = x * 2 + 4
                with record_function("inner"):
                    y = y - 1
            y = y / 1

        forward(x)

        with profile(use_kineto=kineto_available()) as p:
            forward(x)

        events = p.function_events
        important_events = [
            "outer",
            "aten::mul",
            "aten::add",
            "inner",
            "aten::sub",
            "aten::div",
        ]
        idx = 0
        for info in events:
            if info.name == important_events[idx]:
                idx = idx + 1
            if idx == len(important_events):
                break
        self.assertEqual(idx, len(important_events))

        # We can also use record_function to decorate arbitrary function
        @record_function("my_func")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Add two numbers; decorated with a named
            # record_function range.
            return x + y

        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)

        self.assertTrue("my_func" in str(p))

    def test_record_function_multithreaded(self):
        # [FUNCTION DOC] Check exiting a record_function range after another one started or ended
        # does not raise.
        rf = record_function("outer")
        rf.__enter__()
        with record_function("inner"):
            # test that exiting the record function after starting another one
            # doesn't throw.
            rf.__exit__(None, None, None)

        with record_function("inner"):
            rf.__enter__()
        # test that exiting the record function after ending another one
        # doesn't throw.
        rf.__exit__(None, None, None)

    def test_dir(self):
        # [FUNCTION DOC] Check dir on a tensor includes shape and every listed attribute is
        # accessible except imag, which raises RuntimeError for non-complex tensors.
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn("shape", keys)

        # real and imag are only implemented for complex tensors.
        y = torch.randn(10, 10, dtype=torch.cfloat)
        imag_key = "imag"
        self.assertRaises(RuntimeError, lambda: hasattr(x, imag_key))
        self.assertTrue(hasattr(y, imag_key))
        keys.remove(imag_key)

        for key in keys:
            self.assertTrue(hasattr(x, key))

    def test_inplace_on_view_saved_output(self):
        # [FUNCTION DOC] Check an in-place op on a view whose op saves its output does not create
        # a reference cycle: a hook object is deallocated exactly once.
        # Test an in-place operation on a view in which the in-place op saves
        # its output. Previously, this created a reference cycle.
        dealloc = [0]

        class IncrementOnDelete:
            # [CLASS DOC] HELPER: Class whose deletion increments a counter in the
            # enclosing scope.
            def __del__(self):
                # [FUNCTION DOC] HELPER: Increment the deallocation counter.
                dealloc[0] += 1

        def test():
            # [FUNCTION DOC] HELPER: Register a deletion-counting hook on a clone grad_fn and
            # apply an in-place relu on a view of it.
            root = torch.randn(3, 3, requires_grad=True)
            copy = root.clone()
            copy.grad_fn.register_hook(IncrementOnDelete())
            view = copy.view(9)
            torch.nn.functional.relu(view, inplace=True)

        test()
        self.assertEqual(dealloc[0], 1)

    def test_inplace_on_view_leaf_errors(self):
        # [FUNCTION DOC] Check an in-place op on a view of a leaf requiring grad fails immediately
        # with RuntimeError.
        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        x = torch.zeros(1, requires_grad=True)
        y = x.view_as(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "a view of a leaf Variable that "
            "requires grad is being used in "
            "an in-place operation.",
        ):
            y.add_(1)

    def test_inplace_on_view_backward(self):
        # [FUNCTION DOC] Check in-place ops on views in a network with instance norm and relu, and
        # second-order gradients through threshold, do not raise and name the expected double
        # backward node.
        # Issue #10532: Make sure that this does not raise RuntimeError.
        net = nn.Sequential(nn.InstanceNorm2d(2), nn.ReLU(True))

        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        (g,) = torch.autograd.grad(
            net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape), create_graph=True
        )
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))

        # https://discuss.pytorch.org/t/freeing-buffer-strange-behavior/31955/8
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)

        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0.0, 0.0, True)
        prob_interpolated = torch.sigmoid(tmp2)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=inputs,
            grad_outputs=torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_penalty = gradients.sum()
        gradient_penalty.backward()

        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), "ThresholdBackwardBackward0")

    def test_inplace_on_view_weak_grad_fn(self):
        # [FUNCTION DOC] Check the weak reference to the grad_fn of an in-place modified view is
        # preserved after the other tensor is deleted and garbage collected, and that in-place
        # relu on a view of a leaf raises RuntimeError.
        # Issue 23502: Test that b's grad_fn is preserved.
        a = torch.arange(10.0, requires_grad=True)

        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()

        c = b.clone()
        del b
        gc.collect()

        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))

        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()

    def test_out_variant_raises_when_inputs_require_grad(self):
        # [FUNCTION DOC] Check the out variant raises RuntimeError mentioning out when inputs or
        # the output require grad, but works under no_grad.
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        x = torch.zeros_like(a)

        # out=... functions don't support automatic differentiation currently
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

        # the inputs can require grad if we're in no_grad() mode
        with torch.no_grad():
            torch.mul(a, b, out=x)
            self.assertEqual(x, a * b)

        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        x = torch.zeros(2, 2, requires_grad=True)
        # we should throw an exception if the output requires grad
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

    def test_anomaly_detect_nan(self):
        # [FUNCTION DOC] Check anomaly detection reports NaN in a custom backward: no error
        # normally, RuntimeError naming the function and output index with a warning about missing
        # forward info, and the forward stack trace when the forward ran inside detect_anomaly.
        size = 10

        class MyFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward creates a NaN gradient for the first or
            # second input.
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                # [FUNCTION DOC] HELPER: Store the failure flag and return the sum of the
                # first input.
                ctx.fail_0th = fail_0th
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Produce a gradient with a NaN element and return it as
                # the first or second gradient depending on the flag.
                gI = gO.clone().expand(size)
                gI[0] = 0
                gI[0] /= 0  # Generate a nan
                if ctx.fail_0th:
                    return gI, None, None
                else:
                    return None, gI, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        out.backward()  # Should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 0th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            self.assertIn("No forward pass information", str(w[0].message))

        inp = torch.rand(size, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 1th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out = MyFunc.apply(inp, inp, False)
                    out.backward()
            self.assertIn("MyFunc.apply", str(w[0].message))

    def test_calculate_shape_util(self):
        # [FUNCTION DOC] Check the private shape helper returns plain shapes for dense tensors and
        # tensor of sizes for nested tensors.
        out = torch.randn(10, 5, requires_grad=True)
        grad = torch.randn(5, 10, requires_grad=True)
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        if out_shape != torch.Size([10, 5]):
            raise AssertionError(f"expected out_shape == (10, 5), got {out_shape}")
        if grad_shape != torch.Size([5, 10]):
            raise AssertionError(f"expected grad_shape == (5, 10), got {grad_shape}")

        out = torch.nested.as_nested_tensor(
            [
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
            ]
        )
        grad = torch.nested.as_nested_tensor(
            [
                torch.randn(5, 10, requires_grad=True),
                torch.randn(5, 10, requires_grad=True),
            ]
        )
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        if not torch.equal(out_shape, torch.tensor([[10, 5], [10, 5], [10, 5]])):
            raise AssertionError("out_shape mismatch")
        if not torch.equal(grad_shape, torch.tensor([[5, 10], [5, 10]])):
            raise AssertionError("grad_shape mismatch")

    def test_nested_anomaly_detect_nan(self):
        # [FUNCTION DOC] Check anomaly detection with a nested custom Function whose backward
        # calls another custom Function that makes NaN: the error names the inner function and
        # warnings include forward traces for both.
        size = 10

        class MyFunc(Function):
            # [CLASS DOC] HELPER: Outer Function whose backward applies the inner Function.
            @staticmethod
            def forward(ctx, inp1, fail_0th):
                # [FUNCTION DOC] HELPER: Store the failure flag, save the input and return
                # its sum.
                ctx.fail_0th = fail_0th
                ctx.save_for_backward(inp1)
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Apply the inner Function to the gradient scaled by the
                # input and shifted by the input.
                (inp,) = ctx.saved_tensors
                fail_0th = ctx.fail_0th
                g = gO.clone().expand(size)
                gI = MyFunc2.apply(g * inp, g + inp, fail_0th)
                return gI, None

        class MyFunc2(Function):
            # [CLASS DOC] HELPER: Inner Function whose backward generates a NaN in one of
            # two gradients.
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                # [FUNCTION DOC] HELPER: Store the flag and return the combined inputs.
                ctx.fail_0th = fail_0th
                return inp1 * 2.0 + inp2

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Zero the first elements of two gradient copies then
                # divide one by zero depending on the flag; return both and None.
                fail_0th = ctx.fail_0th
                g1 = gO.clone()
                g2 = gO.clone()
                g1[0] = 0
                g2[0] = 0
                # generate a nan
                if fail_0th:
                    g1[0] /= 0
                else:
                    g2[0] /= 0
                return g1, g2, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        gsum.backward()  # should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 0th output.",
            ):
                with detect_anomaly():
                    gsum.backward()
        self.assertIn("No forward pass information", str(w[1].message))

        inp = torch.rand(size, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 1th output.",
            ):
                with detect_anomaly():
                    out = MyFunc.apply(inp, False)
                    (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
                    gsum = ginp.sum()
                    gsum.backward()
        self.assertIn("MyFunc2.apply", str(w[1].message))
        self.assertIn("MyFunc.apply", str(w[2].message))

    def test_anomaly_grad_warnings(self):
        # [FUNCTION DOC] Check anomaly detection warnings: with normal warnings they are recorded
        # as warnings (the anomaly enabled warning and the error location), and when warnings are
        # turned into errors the trace is printed to standard error.
        # PyTorch won't throw warnings if there is an error
        # but we'd want to at least see them in stderr

        class StdErrDiverter:
            # [CLASS DOC] HELPER: Context manager capturing standard error into a string buffer.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Replace standard error by a string buffer and
                # return self.
                self.stderr_orig = sys.stderr
                self.stderr_new = io.StringIO()
                sys.stderr = self.stderr_new
                return self

            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: Save the captured text and restore standard error.
                self.captured = self.stderr_new.getvalue()
                sys.stderr = self.stderr_orig

        # if the warnings don't throw, they will be handled as regular warnings
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    a = torch.randn(5, requires_grad=True)
                    d1 = a + 1
                    d2 = d1**2
                    d1 += 1
                    torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 2)
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        self.assertIn("Error detected in PowBackward0", str(w[1].message))

        # if the warning throws, it will be printed to sys.stderr
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    warnings.simplefilter("error")
                    with StdErrDiverter() as s:
                        a = torch.randn(5, requires_grad=True)
                        d1 = a + 1
                        d2 = d1**2
                        d1 += 1
                        torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 1)
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        self.assertIn("Error detected in PowBackward0", s.captured)

    def test_anomaly_assign_parent_cleanup(self):
        # [FUNCTION DOC] Check objects stored in the metadata of a grad_fn created under anomaly
        # mode are freed when the tensor is freed (no leak through parent assignment).
        # Test that python objects created are properly cleaned up when assign_parent is called

        def get_ref():
            # [FUNCTION DOC] HELPER: Build a graph with create_graph under anomaly mode,
            # store a weakly referenced object in the grad_fn metadata and return the tensor
            # and the reference.
            # we use torch.exp here but any function that will construct a new node in its
            # backward call in grad mode will work
            x = torch.randn(2, 2, requires_grad=True)
            t = x.exp()

            # ExpBackward calls mul, creating the MulBackward node when create_graph=True.
            # In anomaly mode, a PyObject referencing MulBackward's "parent" ExpBackward is added to
            # MulBackward's anomaly metadata dict, creating the following reference chain:
            #
            # grad -> MulBackward -> PyObject -> ExpBackward
            #
            with detect_anomaly():
                grad = torch.autograd.grad(t, x, torch.ones_like(t), create_graph=True)

            # We add a weak reference to a new Foo object, which we insert into ExpBackward's metadata dict
            #
            # (PyObject) -> ExpBackward -> dict -> *Foo*
            #            t ----^        WeakRef ---^
            #
            # We want to test that when grad goes out of scope at the end of this function that PyObject is destroyed
            # We can test this by seeing whether Foo is not kept alive once t is destroyed
            class Foo:
                # [CLASS DOC] HELPER: Empty class used to track object lifetime.
                pass

            my_obj = Foo()
            meta_dict = t.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return t, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_nested_anomaly_printstack_cleanup(self):
        # [FUNCTION DOC] Check metadata objects are cleaned up when a nested custom backward
        # triggers anomaly stack printing.
        # Test if metadata dict PyObject is properly destroyed
        def get_ref():
            # [FUNCTION DOC] HELPER: Build a nested custom Function graph that makes a NaN in
            # backward and return a weak reference to an object in the metadata.
            # This is similar to the construction in test_anomaly_assign_parent_cleanup:
            #
            # MyFuncBackward2 -> PyObject -> MyFuncBackward -> dict -> Foo
            #                               out ---^         WeakRef ---^
            #
            # We want to check that Foo is still properly destroyed even when MyFunc2Backward's
            # AnomalyMetadata calls printstack, which does some python object manipulation.
            #
            # You might be wondering why we still have to test_anomaly_assign_parent_cleanup,
            # since if PyObject is not destroyed here, wouldn't this test would detect that also?
            # The answer is that custom function's PyObject (THPFunction) actually only hold
            # a weak reference to the c++ node!
            class MyFunc(Function):
                # [CLASS DOC] HELPER: Function whose backward applies the second custom Function.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Save the input and return it.
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Apply the second Function to the saved input.
                    (x,) = ctx.saved_tensors
                    return MyFunc2.apply(x)

            class MyFunc2(Function):
                # [CLASS DOC] HELPER: Function whose backward adds NaN to the gradient.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Return the input.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Return the gradient plus NaN.
                    return gO + float("NaN")

            inp = torch.rand(1, requires_grad=True)
            out = MyFunc.apply(inp)
            (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)

            with warnings.catch_warnings(record=True) as w:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Function 'MyFunc2Backward' returned nan values in its 0th output.",
                ):
                    with detect_anomaly():
                        ginp.backward()

            class Foo:
                # [CLASS DOC] HELPER: Empty class used to track object lifetime.
                pass

            my_obj = Foo()
            meta_dict = out.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return out, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_anomaly_mode_no_check_nan(self):
        # [FUNCTION DOC] Check detect_anomaly with check_nan disabled lets a NaN-producing
        # backward run, while a nested context with check_nan enabled raises RuntimeError naming
        # the function and output, and the outer setting is restored afterwards.
        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward returns a NaN-filled expanded tensor.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Return a clone of the input.
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return a ten by ten tensor of NaN.
                return torch.tensor(float("nan")).expand(10, 10)

        def run_fn(a):
            # [FUNCTION DOC] HELPER: Apply the Function and sum the result.
            out = MyFunc.apply(a)
            return out.sum()

        with warnings.catch_warnings(record=True) as w:
            with torch.autograd.detect_anomaly(check_nan=False):
                inp = torch.rand(10, 10, requires_grad=True)
                out = run_fn(inp)
                out.backward(retain_graph=True)

                with torch.autograd.detect_anomaly(check_nan=True):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Function 'MyFuncBackward' returned nan values in its 0th output.",
                    ):
                        out.backward(retain_graph=True)

                out.backward()

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_forward_traceback_preserves_exception_with_checkpoint(self):
        # [FUNCTION DOC] Regression check that collecting the forward traceback while a Python
        # exception is pending does not clear it: runs a custom library op under non-reentrant
        # checkpoint with CUDA memory history recording. Requires CUDA.
        # Regression test: gatherForwardTraceback() must not clear a pending
        # Python exception.  See combined_traceback.cpp for the fix.
        #
        # Ingredients: (1) CUDA memory history recording with context="all"
        # so allocator callbacks fire on free, (2) a custom library op whose
        # forward goes through THPFunction_apply (via Generated in
        # torch._library.autograd), (3) non-reentrant checkpoint.
        #
        # During backward recomputation, _StopRecomputationError is raised
        # from the checkpoint pack_hook inside _save_variables.  The exception
        # stays pending in Python thread state while C++ stack-unwinds (the
        # default python_error ctor does not persist).  Destroying the output
        # THPObjectPtr during unwinding frees the recomputed output tensor's
        # CUDA storage, triggering the allocator callback ->
        # CapturedTraceback::gather() -> gatherForwardTraceback().  On
        # Python < 3.13, without the PyErr_Fetch/PyErr_Restore fix, the
        # PyDict_GetItemRef compat shim clears the pending exception ->
        # SystemError.
        with torch.library._scoped_library("_test_autograd", "FRAGMENT"):

            @torch.library.custom_op("_test_autograd::sin_op", mutates_args=())
            def sin_op(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom library op computing the sine of its input.
                return x.sin()

            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Save the input for backward.
                (x,) = inputs
                ctx.save_for_backward(x)

            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient times the cosine of the saved input.
                (x,) = ctx.saved_tensors
                return grad * x.cos()

            torch.library.register_autograd(
                "_test_autograd::sin_op",
                backward,
                setup_context=setup_context,
            )

            def fn(x):
                # [FUNCTION DOC] HELPER: Call the registered custom op.
                return torch.ops._test_autograd.sin_op(x)

            try:
                torch.cuda.memory._record_memory_history("all", stacks="python")
                x = torch.randn(4, device="cuda", requires_grad=True)
                y = checkpoint(fn, x, use_reentrant=False)
                y.sum().backward()
            finally:
                torch.cuda.memory._record_memory_history(None)

    def test_no_grad_copy(self):
        # [FUNCTION DOC] Check when the engine copies incoming gradients: a non-contiguous
        # gradient is copied before accumulation, otherwise one of two consumers can reuse the
        # computed gradient buffer while they never share one buffer.
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            # [CLASS DOC] HELPER: Function that records the data pointer of the received gradient
            # on the class and passes it through to both inputs.
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                # [FUNCTION DOC] HELPER: Return the sum of inputs.
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Record the gradient data pointer and return the
                # gradient twice.
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward returns a non-contiguous
            # expanded gradient.
            @staticmethod
            def forward(ctx, inp1):
                # [FUNCTION DOC] HELPER: Remember the input size and return a one-element tensor.
                ctx.size = inp1.size()
                return torch.tensor([1.0])

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return ones expanded to the saved size.
                return torch.ones(1).expand(ctx.size)

        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        # non-contiguous grad should be copied
        NonContGradFunc.apply(MyFunc.apply(a, b)).backward()
        self.assertFalse(a.grad.data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(b.grad.data_ptr() == MyFunc.static_grad_ptr)
        # test case that should trigger no copy for one of a,b
        a.grad = b.grad = None
        MyFunc.apply(a, b)[1][0].backward()
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad.data_ptr()
        p_b = b.grad.data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

    def test_no_grad_copy_sparse(self):
        # [FUNCTION DOC] Check copy behavior for sparse gradients from embedding bag: contiguous
        # values are reused for one input, non-contiguous values are cloned for both, and repeated
        # backward accumulates without error.
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            # [CLASS DOC] HELPER: Function recording the data pointer of the values of the
            # received sparse gradient and passing it through.
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                # [FUNCTION DOC] HELPER: Return the sum of inputs.
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Record the pointer of the gradient values and return the
                # gradient twice.
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward returns a sparse gradient with expanded
            # (non-contiguous) indices and values and records the pointer of its values.
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                # [FUNCTION DOC] HELPER: Return the sum of inputs.
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Build the non-contiguous sparse gradient, record the
                # values pointer and return it for both inputs.
                # Create a sparse tensor with non-contiguous indices and values
                # and return as grad.
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return ngrad, ngrad

        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        # test case that should trigger no copy for one of a,b
        emb_matrix = MyFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for _ in range(10):
            loss.backward(retain_graph=True)

        # non-contiguous indices and value, we should trigger a copy.
        a.grad = b.grad = None
        emb_matrix = NonContGradFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = NonContGradFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # Verify we cloned both grads.
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for _ in range(10):
            loss.backward(retain_graph=True)

    def test_gradcheck_single_input(self):
        # [FUNCTION DOC] Check gradcheck and gradgradcheck on a simple multiply function in fast
        # and slow modes.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run both checks for the given fast_mode flag.
            def f(inp):
                # [FUNCTION DOC] HELPER: Multiply the input by 5.
                return inp.mul(5)

            gradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )
            gradgradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )

        check(fast_mode=True)
        check(fast_mode=False)

    @parametrize(
        "layout",
        (
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ),
    )
    def test_gradcheck_input(self, layout):
        # [FUNCTION DOC] Parametrized by sparse layout (COO, CSR, CSC, BSR, BSC): gradcheck of a
        # sum over a sparse input for all combinations of fast mode and masked flag.
        if layout in {torch.sparse_bsr, torch.sparse_bsc}:
            blocksize = (2, 2)
            size = (4, 8)
        else:
            blocksize = None
            size = (2, 2)

        def check(fast_mode, masked):
            # [FUNCTION DOC] HELPER: Run gradcheck of the sum on a sparse tensor of the chosen
            # layout and block size.
            def fn(sparse):
                # [FUNCTION DOC] HELPER: Sum a sparse tensor.
                return torch.sum(sparse)

            gradcheck(
                fn,
                torch.rand(size, dtype=torch.double)
                .to_sparse(layout=layout, blocksize=blocksize)
                .requires_grad_(),
                masked=masked,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        for fast_mode, masked in product(*[(True, False)] * 2):
            check(fast_mode=fast_mode, masked=masked)

    def test_gradcheck_nondeterministic(self):
        # [FUNCTION DOC] Check gradcheck handles non-deterministic backward: deterministic passes,
        # jitter makes it raise RuntimeError saying backward is not reentrant (for gradcheck and
        # gradgradcheck), and a nondeterminism tolerance makes it pass.
        class NonDetFunc(Function):
            # [CLASS DOC] HELPER: Identity Function whose backward multiplies the gradient by a
            # random jitter factor.
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                # [FUNCTION DOC] HELPER: Store the jitter and return the input.
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Return the gradient (itself passed through the Function)
                # times a random factor scaled by the jitter, and None.
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )

        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run all the non-determinism checks for the given
            # fast_mode flag.
            inp = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-6),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradgradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-12),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            gradcheck(
                lambda x: NonDetFunc.apply(x, 1e-6),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            gradgradcheck(
                lambda x: NonDetFunc.apply(x, 1e-12),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_validates_inputs(self):
        # [FUNCTION DOC] Check gradcheck input validation: masked behavior for sparse inputs,
        # ValueError when no input requires grad, a warning for non double precision inputs, and
        # RuntimeError for an input with a stride-zero dimension.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run all validation checks for the given fast_mode flag.
            x = torch.rand(10, requires_grad=True).to_sparse()
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    check_batched_grad=False,
                    atol=1e-1,
                    fast_mode=fast_mode,
                    masked=True,
                )
            )
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=False,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(masked_grad=False),
                    (x,),
                    masked=False,
                    atol=1e-1,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when none of the inputs require grad (always raises even if raise_exception=False)
            x = torch.rand(10, requires_grad=False)
            with self.assertRaisesRegex(
                ValueError, "at least one input tensor to require gradient"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

            # (warning) when inputs are not double precision
            x = torch.ones(1, dtype=torch.float32, requires_grad=True)
            with self.assertWarnsRegex(
                UserWarning, "Input #0 requires gradient and is not a double precision"
            ):
                self.assertTrue(
                    gradcheck(lambda x: x, (x,), atol=1e-1, fast_mode=fast_mode)
                )

            # when layout is not mkldnn(aka has strides) and input has a dimension with stride 0. (always raises
            # even if raise_exception=False)
            x = torch.ones(1, dtype=torch.float64, requires_grad=True)
            x = x.expand((2, 2))
            with self.assertRaisesRegex(
                RuntimeError, "The 0th input has a dimension with stride 0"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_validates_input_mkldnn(self):
        # [FUNCTION DOC] Check gradcheck rejects MKLDNN inputs with forward AD with
        # ValueError (plus the double precision warning), in slow and fast modes. Skipped
        # when MKL-DNN is unavailable.
        # when mkldnn inputs, forward mode testing is not allowed
        # Update tolerances below to make sure the gradient match even in single precision floats
        # Use the warning assert to hide the float32 warning
        x = torch.ones(1).to_mkldnn().requires_grad_()
        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not supported for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=False,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not supported for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=True,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_test_outputs(self):
        # [FUNCTION DOC] Check gradcheck raises ValueError for sparse outputs and for MKLDNN
        # outputs even when raise_exception is false. Skipped when MKL-DNN is unavailable.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run both output checks for the given fast_mode flag.
            # when sparse outputs (always raise even if raise_exception=False)
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                ValueError, "Sparse output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x,
                    (x,),
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

            # when mkldnn outputs (always raise even if raise_exception=False)
            root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
            with self.assertRaisesRegex(
                ValueError, "MKLDNN output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x.to_mkldnn(),
                    (root,),
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_no_differentiable_outputs(self):
        # [FUNCTION DOC] Check gradcheck on functions with no differentiable outputs: raises
        # RuntimeError if the numerical gradient is not zero (or returns False when exceptions are
        # disabled), and passes when there are no outputs at all.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the checks for the given fast_mode flag.
            # When none of the outputs are differentiable, but numerical gradient is not zero
            x = torch.ones((1,), requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "Numerical gradient for function expected to be zero"
            ):
                gradcheck(lambda x: torch.tensor([x]), x)
            self.assertFalse(
                gradcheck(
                    lambda x: torch.tensor([x]),
                    x,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # succeed when no outputs at all
            self.assertTrue(gradcheck(lambda x: (), (x,), fast_mode=fast_mode))

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_batched_grad(self):
        # [FUNCTION DOC] Check the batched gradient check failure on a sparse input raises
        # RuntimeError (or returns False without exceptions).
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the checks for the given fast_mode flag.
            x = torch.rand(10, dtype=torch.double, requires_grad=True).to_sparse()
            # runtime error while compute batched grad (print big error)
            with self.assertRaisesRegex(
                RuntimeError,
                "gradcheck or gradgradcheck failed while testing batched gradient",
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_backward_mul_by_grad_output(self):
        # [FUNCTION DOC] Check gradcheck detects bad backward behavior: sparse gradient with wrong
        # sparse dimension, backward not multiplied by grad_output (dense and sparse), and
        # gradient of wrong layout; each as an exception or a False result.
        # when grad_input is sparse and has incorrect sparse_dim/dense_dim
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run all the checks for the given fast_mode flag.
            def fn(x):
                # [FUNCTION DOC] HELPER: Clone with a hook that converts the gradient into a
                # sparse tensor with one sparse dim, then densify.
                def hook(grad):
                    # [FUNCTION DOC] HELPER: Convert the gradient to dense then sparse with one
                    # sparse dim, if defined.
                    if grad is not None:
                        return grad.to_dense().to_sparse(1)
                    return grad

                y = x.clone()
                y.register_hook(hook)
                return y.to_dense()

            x = torch.ones((2, 2), dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                RuntimeError, "grad is sparse tensor, but has incorrect sparse_dim"
            ):
                gradcheck(
                    fn,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    fn,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when backward not multiplied by grad_output (non-sparse case)
            def fn2(x):
                # [FUNCTION DOC] HELPER: Clone with a hook that adds a small constant to
                # the gradient.
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "backward not multiplied by grad_output"
            ):
                gradcheck(fn2, (x,), atol=1e-1, fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(
                    fn2, (x,), atol=1e-1, raise_exception=False, fast_mode=fast_mode
                )
            )

            # when backward not multiplied by grad_output (sparse case)
            def fn3(x):
                # [FUNCTION DOC] HELPER: Densify a clone of a sparse input with a hook adding a
                # small constant to the gradient.
                y = x.clone().to_dense()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                RuntimeError, "backward not multiplied by grad_output"
            ):
                gradcheck(
                    fn3,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    fn3,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when layout of grad_input is not the same as input
            class Test(Function):
                # [CLASS DOC] HELPER: Identity Function whose backward returns a sparse gradient.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Return the input.
                    return x

                @staticmethod
                def backward(ctx, x):
                    # [FUNCTION DOC] HELPER: Return the gradient converted to sparse.
                    return x.to_sparse()

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "grad is incorrect layout"):
                gradcheck(
                    Test.apply, (x,), check_batched_grad=False, fast_mode=fast_mode
                )
            self.assertFalse(
                gradcheck(
                    Test.apply,
                    (x,),
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    # There are two issues:
    # 1. Dynamo uses real fake tensor when speculating so we never trace
    #    the x is none branch.
    # 2. torch.autograd.gradcheck wraps grads with UndefinedGrad which
    #    gets resulted as Zero tensors when getting passed into custom
    #    autograd function in the runtime. Apply materialize grad is tricky,
    #    because user function (dynamo in this case) needs to handle None case
    # this is fine in normal torch.compile case because aot_autograd would
    # never receive None tensors. But it will be a problem when we are directly
    # tracing autograd.grad into graph because now you will get different
    # result from eager. One potential fix is to detect x is None in dynamo
    # bytecode level but that is too complicated so we just YOLO.
    @skipIfTorchDynamo("branching on grad")
    def test_gradcheck_undefined_grad(self):
        # [FUNCTION DOC] Check gradcheck of the undefined gradient handling: a hook that raises on
        # an undefined gradient makes gradcheck raise RuntimeError saying the backward must handle
        # undefined grads (with a compatibility warning), or return False. Skipped under dynamo.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the checks for the given fast_mode flag.
            # when encounter runtime error while running backward
            def fn(x):
                # [FUNCTION DOC] HELPER: Clone with a hook raising on an undefined gradient.
                def hook(x):
                    # [FUNCTION DOC] HELPER: Raise RuntimeError if the gradient is None.
                    if x is None:
                        raise RuntimeError("x is undefined")

                y = x.clone()
                y.register_hook(hook)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertWarnsRegex(
                UserWarning,
                "Backwards compatibility: New undefined gradient support checking feature",
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected backward function to handle undefined output grads",
                ):
                    gradcheck(fn, (x,), fast_mode=fast_mode)
                self.assertFalse(
                    gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
                )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_jacobian_mismatch(self):
        # [FUNCTION DOC] Check gradcheck reports a Jacobian mismatch for a backward hook that
        # perturbs gradients, for real and complex inputs and outputs, as RuntimeError or False.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the checks for the given fast_mode flag.
            def fn(x):  # R -> R, C -> C
                # [FUNCTION DOC] HELPER: Clone with a gradient-perturbing hook (real to real or
                # complex to complex).
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )

            x_c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn, (x_c,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn, (x_c,), raise_exception=False, fast_mode=False)
            )

            def fn2(x):  # R -> C
                # [FUNCTION DOC] HELPER: Make a complex tensor from real input with a
                # gradient-perturbing hook.
                y = torch.complex(x, x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn2, (x,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn2, (x,), raise_exception=False, fast_mode=False)
            )

            def fn3(x):  # C -> R
                # [FUNCTION DOC] HELPER: Take the real part of a complex input with a
                # gradient-perturbing hook.
                y = torch.real(x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn3, (x_c,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn3, (x_c,), raise_exception=False, fast_mode=False)
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_dense_and_sparse_inputs(self):
        # [FUNCTION DOC] Check gradcheck with one dense and one sparse input.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the check for the given fast_mode flag.
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Multiply the dense input by the dense form of the
                # coalesced sparse input.
                return x * y.coalesce().to_dense()

            a = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
            b = torch.rand(2, 2, dtype=torch.double).to_sparse().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn,
                    (a, b),
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_multiple_mkldnn_inputs(self):
        # [FUNCTION DOC] Check gradcheck with mixed dense and MKLDNN inputs. Skipped when MKL-DNN
        # is unavailable.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the checks for the given fast_mode flag.
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Add a dense input and the dense form of an MKLDNN input.
                return x + y.to_dense()

            a = torch.rand(10, requires_grad=True)
            b = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn, (a, b), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

            def fn2(x, y):
                # [FUNCTION DOC] HELPER: Add the dense forms of two MKLDNN inputs.
                return x.to_dense() + y.to_dense()

            c = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn, (a, c), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_output_shape_or_dtype_depend_on_values(self):
        # [FUNCTION DOC] Check gradcheck raises AssertionError when the output shape or dtype
        # depends on the perturbation of input values.
        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the checks for the given fast_mode flag.
            def fn(x):
                # [FUNCTION DOC] HELPER: Return a doubled-length output when all values are at
                # least one, otherwise the input.
                if torch.all(x >= 1):
                    return torch.cat([x, x])
                else:
                    return x

            a = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same shape when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn, (a,), fast_mode=fast_mode))

            def fn2(x):
                # [FUNCTION DOC] HELPER: Return a float32 output when all values are at least one,
                # otherwise the input.
                if torch.all(x >= 1):
                    return x.to(torch.float32)
                else:
                    return x

            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same dtype when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn2, (a,), fast_mode=fast_mode))

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_complex_non_complex_outputs(self):
        # [FUNCTION DOC] Check gradcheck on functions with mixed complex and real outputs.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return a complex tensor built from two reals and a real
            # shifted tensor.
            z = torch.complex(x, y)
            return z, x + 1

        a = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        self.assertTrue(gradcheck(fn, (a, b)))

        def fn2(z):
            # [FUNCTION DOC] HELPER: Return a complex input and its real part.
            return z, torch.real(z)

        c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
        self.assertTrue(gradcheck(fn2, (c)))

    @unittest.skipIf(TEST_CUDA, "CPU-only test")
    def test_gradcheck_adjusted_atol_complex_inputs(self):
        # [FUNCTION DOC] Regression check for the absolute tolerance adjustment with complex
        # inputs: fast and slow gradcheck both pass for a norm function at a seeded random complex
        # input. CPU only.
        # Regression test for incorrect atol transformation for
        # complex inputs, allowing fast gradcheck to fail and slow gradcheck to pass.
        # See issue: https://github.com/pytorch/pytorch/issues/166385

        # this particular seed on CPU triggers a specific input tangent vector u in [0,1)^d such that
        # v^T(j_n - j_a)u is interesting.
        torch.manual_seed(97)

        def sample_func(z):
            # [FUNCTION DOC] HELPER: Reciprocal of the norm of the input.
            return 1.0 / torch.norm(z)

        # Input needs to be at least 2-dim. to trigger
        # in gradcheck an input projection vector u
        # that is not all 1s.
        eps = 10e-3  # eps distance factor from origin, to get
        # some interesting numerical vs analytic discrepancy.
        z = eps * torch.rand(
            2,
            dtype=torch.complex128,
            requires_grad=True,
        )
        atol = 8.3e-6
        rtol = 1e-9

        # check both fast and slow gradcheck pass after the fix to _adjusted_atol()
        self.assertTrue(
            gradcheck(sample_func, (z,), fast_mode=True, atol=atol, rtol=rtol)
        )
        self.assertTrue(
            gradcheck(sample_func, (z,), fast_mode=False, atol=atol, rtol=rtol)
        )

    def test_gradcheck_get_numerical_jacobian(self):
        # [FUNCTION DOC] Check the deprecated numerical Jacobian helper: emits a FutureWarning,
        # returns expected Jacobians for chosen targets or all inputs, and rejects grad_out other
        # than one with ValueError.
        # get_numerical_jacobian is deprecated and no longer used internally by gradcheck
        from torch.autograd.gradcheck import get_numerical_jacobian

        def fn(inputs):
            # [FUNCTION DOC] HELPER: Take a tuple of two tensors and return two linear
            # combinations of them.
            # get_numerical_jacobian requires fn to take inputs as a tuple
            # and returns the jacobian wrt the first output
            x = inputs[0]
            y = inputs[1]
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            jacobian = get_numerical_jacobian(fn, (a, b), target=a, eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))

        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobian[1], 1 * torch.eye(4, dtype=torch.double))

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6, grad_out=2.0)

    def test_gradcheck_get_analytical_jacobian(self):
        # [FUNCTION DOC] Check the deprecated analytical Jacobian helper: FutureWarning, expected
        # Jacobians, reentrant flag true for deterministic and false for non-deterministic
        # backward, and ValueError for grad_out other than one.
        from torch.autograd.gradcheck import get_analytical_jacobian

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return two linear combinations of the inputs.
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        outputs = fn(a, b)
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a, b), outputs[0])
        self.assertEqual(jacobians[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobians[1], 1 * torch.eye(4, dtype=torch.double))
        self.assertTrue(reentrant)

        class NonDetFunc(Function):
            # [CLASS DOC] HELPER: Identity Function with a randomly jittered backward.
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                # [FUNCTION DOC] HELPER: Store the jitter and return the input.
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Return the gradient passed through the Function times a
                # random jitter factor, and None.
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )

        outputs = NonDetFunc.apply(a, 1e-6)
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a,), outputs)
        self.assertFalse(reentrant)

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobians, _, _, _ = get_analytical_jacobian((a,), outputs, grad_out=2.0)

    def test_gradcheck_custom_error(self):
        # [FUNCTION DOC] Check gradcheck raises GradcheckError (a subclass of RuntimeError)
        # for a Jacobian mismatch and that other exceptions are not swallowed when exceptions
        # are disabled.
        from torch.autograd.gradcheck import GradcheckError

        def check(fast_mode):
            # [FUNCTION DOC] HELPER: Run the checks for the given fast_mode flag.
            def fn(x):
                # [FUNCTION DOC] HELPER: Clone with a gradient-perturbing hook.
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                GradcheckError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )

            def fn2(x):
                # [FUNCTION DOC] HELPER: Raise a RuntimeError that is not a GradcheckError.
                raise RuntimeError("Not a GradcheckError!")

            # Checks that when raise_exception=False, non-GradcheckErrors are not caught by gradcheck
            with self.assertRaisesRegex(RuntimeError, "Not a GradcheckError!"):
                gradcheck(fn2, (x,), fast_mode=fast_mode, raise_exception=False)

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_forward_ad(self):
        # [FUNCTION DOC] Check gradcheck with forward AD for real, mixed and complex inputs, where
        # a deliberately wrong forward-mode function raises a RuntimeError about a forward mode
        # Jacobian mismatch.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the sum and the second input.
            return x + y, y

        def bad_fn(x, y):
            # [FUNCTION DOC] HELPER: Same as the plain function but scales the tangent of y by 1.1
            # when running inside a forward AD level.
            # Hacky way to check if we're currently inside a forward ad level
            is_running_forward_ad = fwAD._current_level >= 0

            if is_running_forward_ad:
                y_p, y_d = fwAD.unpack_dual(y)
                y = fwAD.make_dual(y_p, y_d * 1.1)

            return x + y, y

        err_msg = "Jacobian computed with forward mode mismatch for output 0 with respect to input 1"

        for fast_mode in [True, False]:
            # Test for all inputs and outputs being real
            x = torch.rand(2, dtype=torch.double, requires_grad=True)
            y = torch.rand(2, dtype=torch.double, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            def basic_mul(x):
                # [FUNCTION DOC] HELPER: Multiply by an imaginary unit and view the result
                # as real.
                return torch.view_as_real(torch.resolve_conj(x * 1j))

            gradcheck(basic_mul, x, check_forward_ad=True, fast_mode=fast_mode)

            # Test for one input and one output being complex
            x = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            # Test for all inputs and outputs being complex
            y = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

    def test_gradcheck_forward_ad_runs_with_no_requires_grad(self):
        # [FUNCTION DOC] Check inputs passed to the function under forward AD do not require grad
        # even if they did when given to gradcheck.
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        # This test checks that when the inputs are passed to the function they should not have
        # requires_grad=True even though they may have requires_grad=True when passed
        # to gradcheck
        class UserFn(Function):
            # [CLASS DOC] HELPER: Function returning clones; asserts inputs do not require grad
            # when in a forward AD level.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Assert that inputs do not require grad under forward AD,
                # then return clones of both.
                if fwAD._current_level >= 0:
                    self.assertFalse(x.requires_grad)
                    self.assertFalse(y.requires_grad)
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                # [FUNCTION DOC] HELPER: Return the tangents unchanged.
                return x_t, y_t

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=True)

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=False,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=False)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

    def test_gradcheck_forward_ad_respects_requires_grad(self):
        # [FUNCTION DOC] Check which inputs get forward-mode tangents: counts of jvp calls match
        # expected totals for plain forward AD, undefined grad check and batched forward grad, and
        # for a mix with one input not requiring grad.
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        jvp_count = [0]

        class UserFn(Function):
            # [CLASS DOC] HELPER: Function returning clones that counts jvp calls.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Return clones of both inputs.
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                # [FUNCTION DOC] HELPER: Count the call and return the tangents.
                jvp_count[0] += 1
                return x_t, y_t

        # NB: In slow gradcheck we need to loop through numel times so use numel = 1 to ensure
        #     that fast and slow have the same counts
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=True)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=False,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )
        self.assertEqual(jvp_count[0], 2)  # (2) once per input
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )
        self.assertEqual(
            jvp_count[0], 6
        )  # (+4): (once with normal ZT (+1), once with efficient ZT (+1)) for each input (x2)
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )
        self.assertEqual(
            jvp_count[0], 12
        )  # (+6): (compute batch of 2 with vmap (+1), with a loop (+2)) for each input (x2)
        jvp_count = [0]

        # Repeat the previous test except we mark one input with requires_grad=False
        # NB: _test_undefined_forward_mode is only (+1), when function has single differentiable input, not (+2)!
        #     Otherwise, other counts are halved.
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=False)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )
        self.assertEqual(jvp_count[0], 5)  # 1 + 1 + 3

    def test_gradcheck_check_forward_or_backward_only(self):
        # [FUNCTION DOC] Check gradcheck runs only the forward AD or backward AD checks selected
        # by flags: errors from a deliberately failing forward or backward formula appear only
        # when the corresponding check is on (backward failure hides forward failure), and
        # AssertionError if neither is on.
        """Depending on settings for check_forward_ad and check_backward_ad, the
        correct codepaths should be reached (or not reached)
        """
        fwd_fail_err_msg = "FAIL FWD"
        bwd_fail_err_msg = "FAIL BWD"

        class UserFn(Function):
            # [CLASS DOC] HELPER: Function with vjp and jvp rules that can be made to raise
            # by flags.
            @staticmethod
            def forward(ctx, foo, fwd_bad, bwd_bad):
                # [FUNCTION DOC] HELPER: Store the failure flags and double the input.
                ctx.fwd_bad = fwd_bad
                ctx.bwd_bad = bwd_bad
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                # [FUNCTION DOC] HELPER: Raise RuntimeError when the backward is set to fail,
                # otherwise return twice the gradient and None values.
                if ctx.bwd_bad:
                    raise RuntimeError(bwd_fail_err_msg)
                else:
                    return 2 * gO, None, None

            @staticmethod
            def jvp(ctx, gI, _1, _2):
                # [FUNCTION DOC] HELPER: Raise RuntimeError when the forward rule is set to fail,
                # otherwise return twice the tangent.
                if ctx.fwd_bad:
                    raise RuntimeError(fwd_fail_err_msg)
                else:
                    return 2 * gI

        for fast_mode in (True, False):
            for check_forward_ad in (True, False):
                for check_backward_ad in (True, False):
                    for fwd_bad in (True, False):
                        for bwd_bad in (True, False):
                            fwd_should_fail = fwd_bad and check_forward_ad
                            bwd_should_fail = bwd_bad and check_backward_ad

                            def run():
                                # [FUNCTION DOC] HELPER: Run gradcheck of the Function with the
                                # current loop flags.
                                gradcheck(
                                    UserFn.apply,
                                    (x, fwd_bad, bwd_bad),
                                    check_forward_ad=check_forward_ad,
                                    check_backward_ad=check_backward_ad,
                                    check_undefined_grad=check_backward_ad,
                                    check_batched_grad=check_backward_ad,
                                    fast_mode=fast_mode,
                                )

                            x = torch.rand(2, dtype=torch.double, requires_grad=True)

                            if not check_forward_ad and not check_backward_ad:
                                with self.assertRaisesRegex(
                                    AssertionError, "Expected at least one of"
                                ):
                                    run()
                                continue

                            if not fwd_should_fail and not bwd_should_fail:
                                run()
                            else:
                                # If both fail, backward AD failure "hides" forward AD failure
                                if fwd_should_fail:
                                    fail_msg = fwd_fail_err_msg
                                if bwd_should_fail:
                                    fail_msg = bwd_fail_err_msg
                                with self.assertRaisesRegex(RuntimeError, fail_msg):
                                    run()

    def test_gradcheck_forward_ad_batched_grad(self):
        # [FUNCTION DOC] Check batched forward gradients: functions with non-tensor inputs,
        # unrelated inputs, and a random op in the jvp rule that vmap does not support
        # (raising RuntimeError).
        x = torch.rand(2, dtype=torch.double, requires_grad=True)

        # multiple inputs and outputs with non-tensors inputs
        def fn1(a: torch.Tensor, b: int):
            # [FUNCTION DOC] HELPER: Return a clone and an incremented tensor given a tensor and
            # an int.
            return a.clone(), a + 1

        gradcheck(
            fn1,
            (x, 1),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_undefined_grad=False,
            check_batched_forward_grad=True,
        )

        # unrelated inputs: tangent for c is None
        def fn2(a: torch.Tensor, c: torch.Tensor):
            # [FUNCTION DOC] HELPER: Return a clone of the first input, ignoring the second.
            return a.clone()

        gradcheck(
            fn2,
            (x, x.clone()),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_undefined_grad=False,
            check_batched_forward_grad=True,
        )

        class Fn(Function):
            # [CLASS DOC] HELPER: Function whose jvp rule uses a random op.
            @staticmethod
            def forward(ctx, foo):
                # [FUNCTION DOC] HELPER: Double the input.
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient doubled.
                return gO * 2

            @staticmethod
            def jvp(ctx, gI):
                # [FUNCTION DOC] HELPER: Call a random op, then return the tangent doubled.
                torch.randn_like(gI)
                return gI * 2

        msg = "vmap: We do not yet support calling random operations inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            gradcheck(
                Fn.apply, (x,), check_forward_ad=True, check_batched_forward_grad=True
            )

    def test_version_counter(self):
        # [FUNCTION DOC] Check the version counter: increments on in-place ops, is shared between
        # a view and its base, is preserved by assigning to the data attribute, and keeps being
        # shared afterwards.
        x = torch.randn(1, 2)

        # In-place op bumps version
        x_saved_version = x._version
        x.add_(1).add_(1)
        self.assertTrue(x._version > x_saved_version)

        # Differentiable view shares version counter
        xz = x[:]
        self.assertTrue(x._version == xz._version)
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

        # `x.data = y` preserves version counter of `x`
        x_saved_version = x._version
        x.data = torch.randn(2, 3)
        self.assertTrue(x._version == x_saved_version)
        x.add_(1)
        self.assertTrue(x._version > x_saved_version)
        # Make sure `x` is still using the same version counter it shares with `xz`
        self.assertTrue(x._version == xz._version)

        # In-place op on `xz` also updates version of `x`,
        # because they share the version counter
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

    def test_set_data_tensorimpl_type(self):
        # [FUNCTION DOC] Check assigning a sparse tensor to the data attribute of a dense tensor
        # raises RuntimeError about incompatible tensor type.
        # Dense tensor has impl of type `TensorImpl`, while sparse tensor has impl
        # of type `SparseTensorImpl`.
        x = torch.randn(1, 2)
        x_s = torch.sparse_coo_tensor(torch.zeros([1, 1]), torch.ones([1]))
        with self.assertRaisesRegex(RuntimeError, "incompatible tensor type"):
            x.data = x_s

    def test_set_data_preserve_pyobj(self):
        # [FUNCTION DOC] Check assigning the data attribute keeps the Python object identity of
        # the tensor.
        a = torch.randn(1, 2)
        b = torch.randn(1, 2)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

    def test_set_data_self_requires_grad(self):
        # [FUNCTION DOC] Check assigning to the data attribute of a tensor requiring grad works
        # for floats and raises RuntimeError for an integer dtype.
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0)
        c = torch.tensor(3, dtype=torch.int64)
        a.data = b
        with self.assertRaisesRegex(
            RuntimeError, "must be floating point or complex dtype"
        ):
            a.data = c

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    def test_thread_shutdown(self):
        # [FUNCTION DOC] Run a subprocess script using a custom Function and verify the thread
        # shutdown API usage message is logged only when the engine has device worker threads
        # (CUDA, MPS or XPU). Skipped on Windows.
        code = """import torch
from torch.autograd import Function
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad

# Run on cuda if it is available to ensure that the worker thread
# is properly initialized by the time we exit.
device = "cuda" if torch.cuda.is_available() else "cpu"

for shape in [(1,), ()]:
    v = torch.ones(shape, requires_grad=True, device=device)
    MyFunction.apply(v).backward()
"""
        s = TestCase.runWithPytorchAPIUsageStderr(code)
        # The autograd engine creates worker threads only when GPU devices are present.
        # So make sure that we do shutdown threads when we're testing cuda and make sure
        # that there is no thread to shutdown when we're not using cuda.
        if TEST_CUDA or torch.backends.mps.is_available() or torch.xpu.is_available():
            self.assertRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")
        else:
            self.assertNotRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")

    @unittest.skipIf(
        IS_MACOS,
        "Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941",
    )
    def test_deep_reentrant(self):
        # [FUNCTION DOC] Check deep recursive reentrant backward does not overflow the stack and
        # worker threads are reused. Skipped on macOS.
        class DeepReentrant(Function):
            # [CLASS DOC] HELPER: Function whose backward recursively runs a reentrant backward of
            # itself until a counter drops below zero.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Build the decremented value under enable_grad and return
                # it detached.
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the gradient if the counter is negative, otherwise
                # run a nested backward of itself.
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    DeepReentrant.apply(ctx.x).sum().backward()
                return x

        # Test stack overflow escape mechanism
        v = torch.tensor(2000.0, requires_grad=True)
        # This will cause stack overflow if reentrant calls are handled
        # in the same thread recursively
        DeepReentrant.apply(v).sum().backward()

        # Test stack overflow escape mechanism multiple times
        # to ensure reusing workers in the pool works fine
        v2 = torch.tensor(200.0, requires_grad=True)
        DeepReentrant.apply(v2).sum().backward()

    def test_reentrant_priority(self):
        # [FUNCTION DOC] Check reentrant backward tasks have priority over other ready tasks:
        # ten reentrant calls run before the remaining non-reentrant function, regardless of
        # sequence numbers.
        order = []

        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function that records its backward in the order list.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Record the call and return the gradient.
                order.append("MyFunction")
                return x

        class Reentrant(Function):
            # [CLASS DOC] HELPER: Function that records its backward and recursively triggers
            # nested backward until a counter falls below zero.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Build the decremented value under enable_grad and return
                # it detached.
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Record the call, then return the gradient or run a nested
                # backward of itself.
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        v = a * b
        v.backward()
        # The tasks for the Reentrant and MyFunction backward() will be added
        # to the queue in the autograd engine at the same time. The backward
        # for Reentrant will be executed first, which will then add other
        # backward tasks to the queue. We want to ensure all the reentrant tasks
        # are prioritized over the MyFunction backward task regardless of their
        # sequence numbers
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count("Reentrant"), 10)
        self.assertEqual(order[-1], "MyFunction")

    @slowTest
    def test_checkpointing(self):
        # [FUNCTION DOC] Slow test: reentrant checkpointing of a small network over two thousand
        # inputs followed by a combined backward.
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp),
        )

        feat_combined = []
        for _ in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            feat_r = checkpoint(module, data_r, use_reentrant=True)
            feat_combined.append(feat_r)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()

    def _test_checkpointing_non_reentrant_autocast(self, device_type):
        # [FUNCTION DOC] HELPER: For autocast enabled and disabled, run non-reentrant checkpoint
        # under autocast on the given device type and check that recomputation in backward uses
        # the same bfloat16 or float32 precision.
        for enabled in [True, False]:

            def foo(x, y, z):
                # [FUNCTION DOC] HELPER: Chain matrix multiplications and assert the result dtype
                # matches the autocast setting.
                # torch.mm is on autocast's list of ops that should run in
                # the autocast precision
                x = torch.mm(x, y)
                y = torch.mm(x, z)
                z = torch.mm(z, z)
                expected_dtype = torch.float32 if not enabled else torch.bfloat16
                self.assertEqual(expected_dtype, z.dtype)
                return z

            x = torch.randn(3, 3, requires_grad=True)
            y = torch.randn(3, 3, requires_grad=True)
            z = torch.randn(3, 3, requires_grad=True)
            if device_type in ("cuda", "xpu"):
                x = x.to(device_type)
                y = y.to(device_type)
                z = z.to(device_type)

            with torch.autocast(
                enabled=enabled, device_type=device_type, dtype=torch.bfloat16
            ):
                loss = checkpoint(foo, x, y, z, use_reentrant=False)
                loss = loss.sum()

            # Without saving + recasting the autocast type, would raise error in autograd
            # about mismatched dtypes.
            loss.backward()  # triggers recomputation to check it runs in bfloat

    def test_checkpointing_non_reentrant_autocast_cpu(self):
        # [FUNCTION DOC] Check autocast settings such as dtype are preserved when non-reentrant
        # checkpoint recomputes in backward on CPU.
        """
        Test that autocast args such as the dtype are preserved during non-reentrant
        checkpoint recomputation on CPU.
        """
        self._test_checkpointing_non_reentrant_autocast(device_type="cpu")

    @unittest.skipIf(
        (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported())
        and (not torch.xpu.is_available() or not torch.xpu.is_bf16_supported()),
        "Test requires CUDA or XPU bf16 support",
    )
    def test_checkpointing_non_reentrant_autocast_gpu(self):
        # [FUNCTION DOC] Check autocast settings are preserved during non-reentrant checkpoint
        # recomputation on CUDA or XPU. Skipped without bfloat16 support on either device.
        """
        Test that autocast args/kwargs such as the dtype are preserved during
        non-reentrant checkpoint recomputation on GPU.
        """
        device_type = "cuda" if torch.cuda.is_available() else "xpu"
        self._test_checkpointing_non_reentrant_autocast(device_type=device_type)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    @slowTest
    def test_checkpointing_without_reentrant_memory_savings(self):
        # [FUNCTION DOC] Slow CUDA test: compares peak memory of a model with no checkpoint,
        # reentrant checkpoint and non-reentrant checkpoint, asserting both checkpointed variants
        # use less than the baseline.
        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: Stack of sequential three-layer blocks that optionally
            # checkpoint each block, with preallocated grads so memory differences come
            # from activations.
            def __init__(self, n, use_checkpoint, use_reentrant):
                # [FUNCTION DOC] HELPER: Build the blocks and pre-fill every weight and bias grad
                # with ones.
                super().__init__()
                self.n = n
                self.use_checkpoint = use_checkpoint
                self.use_reentrant = use_reentrant
                self.layers = nn.ModuleList()
                for _ in range(self.n):
                    layer = nn.Sequential(
                        nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                    )
                    self.layers.append(layer)
                # pre-allocate the grad so that increased memory usage is mainly
                # due to activations.
                for layer in self.layers:
                    for lin in layer:
                        lin.weight.grad = torch.ones_like(lin.weight)
                        lin.bias.grad = torch.ones_like(lin.bias)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply each block in turn, directly or through checkpoint
                # with the chosen reentrant mode.
                for i in range(self.n):
                    if not self.use_checkpoint:
                        x = self.layers[i](x)
                    else:
                        x = checkpoint(
                            self.layers[i], x, use_reentrant=self.use_reentrant
                        )

                return x

        model_no_checkpoint = MyModel(
            8, use_checkpoint=False, use_reentrant=False
        ).cuda()
        model_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=True
        ).cuda()
        model_no_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=False
        ).cuda()

        x = torch.randn(100, 256, requires_grad=True, device="cuda")

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        self.assertTrue(mem_reentrant_checkpoint < mem_no_checkpoint)
        self.assertTrue(mem_no_reentrant_checkpoint < mem_no_checkpoint)

    def test_checkpointing_without_reentrant_custom_function_works(self):
        # [FUNCTION DOC] Check a custom Function works under non-reentrant checkpoint: accessing
        # saved tensors a second time within one backward raises RuntimeError because recomputed
        # values are cleared once unpacked.
        msg = "Unpack is being triggered for a tensor that was already unpacked once"

        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving five tensors that tries to unpack them twice in
            # backward and expects the second to raise.
            @staticmethod
            def forward(ctx, x, y, z):
                # [FUNCTION DOC] HELPER: Compute a product, save inputs and intermediates, and
                # return a doubled value.
                w = x * y * z
                out = w + w
                ctx.save_for_backward(x, y, z, w, out)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Unpack saved tensors once, assert a second unpack raises
                # RuntimeError, and return three of them as gradients.
                x, y, z, w, out = ctx.saved_tensors
                # Accessing the saved Tensors a second time will raise because
                # recomputed tensors get cleared as soon as they are unpacked.
                # A recomputation is only triggered if your backward has a new
                # graph-task id.
                with self.assertRaisesRegex(RuntimeError, msg):
                    x_2, y_2, z_2, w_2, out_2 = ctx.saved_tensors
                return x, y, z

        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        z = torch.tensor(3.0, requires_grad=True)

        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Compute products of inputs and apply the custom Function.
            x = x * y * z
            y = y * y * z
            z = z * z
            out = MyFunc.apply(x, y, z)
            return out

        out = checkpoint(foo, x, y, z, use_reentrant=False)
        out.sum().backward()

    def test_checkpointing_without_reentrant_with_context_fn(self):
        # [FUNCTION DOC] Check the context_fn option of non-reentrant checkpoint: a dispatch mode
        # supplied for forward sees only the forward op, one supplied for recompute sees the
        # recompute ops, and use_reentrant true with context_fn raises an error.
        class VerboseTorchDispatchMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode recording the names of the ops that run.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Start with an empty operator list.
                self.operators = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Record the op name and run the op.
                if kwargs is None:
                    kwargs = {}
                self.operators.append(func.__name__)
                return func(*args, **kwargs)

        x = torch.tensor(1.0, requires_grad=True)
        verbose_mode = VerboseTorchDispatchMode()

        def context_fn():
            # [FUNCTION DOC] HELPER: Return the recording mode for forward and a null context
            # for recompute.
            return verbose_mode, contextlib.nullcontext()

        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        self.assertEqual(verbose_mode.operators, ["exp.default"])

        verbose_mode.operators = []

        def context_fn():
            # [FUNCTION DOC] HELPER: Return a null context for forward and the recording mode
            # for recompute.
            return contextlib.nullcontext(), verbose_mode

        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        out.backward()
        self.assertEqual(verbose_mode.operators, ["exp.default", "detach.default"])

        with self.assertRaisesRegex(
            Exception, "only supported when use_reentrant=False"
        ):
            out = checkpoint(
                lambda x: x.sin(), x, use_reentrant=True, context_fn=context_fn
            )

    def test_checkpointing_without_reentrant_context_fn_exits_on_forward_exception(
        self,
    ):
        # [FUNCTION DOC] Check the forward context from context_fn is exited with the exception
        # details when the checkpointed function raises, and its state is reset.
        active_context = contextvars.ContextVar("active_context", default=False)
        events = []
        exit_exc = []

        class Context:
            # [CLASS DOC] HELPER: Context manager recording enter and exit events, setting a
            # context variable on enter and resetting it on exit, and storing the exception passed
            # to exit.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Record enter and set the context variable.
                events.append("enter")
                self.token = active_context.set(True)

            def __exit__(self, exc_type, exc_value, traceback):
                # [FUNCTION DOC] HELPER: Record exit with exception info and reset the context
                # variable; does not suppress the error.
                events.append("exit")
                exit_exc.append((exc_type, exc_value))
                active_context.reset(self.token)

        def context_fn():
            # [FUNCTION DOC] HELPER: Return the recording context for forward and a null context
            # for recompute.
            return Context(), contextlib.nullcontext()

        def fn(x):
            # [FUNCTION DOC] HELPER: Assert the context is active then raise RuntimeError saying
            # the forward failed.
            self.assertTrue(active_context.get())
            raise RuntimeError("forward failed")

        exc = None
        x = torch.tensor(1.0, requires_grad=True)
        try:
            checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        except RuntimeError as e:
            exc = e

        self.assertIsNotNone(exc)
        self.assertEqual(events, ["enter", "exit"])
        self.assertEqual(exit_exc[0][0], RuntimeError)
        self.assertEqual(str(exit_exc[0][1]), "forward failed")
        self.assertFalse(active_context.get())

    def test_checkpointing_without_reentrant_context_fn_cannot_suppress_forward_exception(
        self,
    ):
        # [FUNCTION DOC] Check that a forward context from context_fn that tries to suppress the
        # exception cannot: checkpoint raises RuntimeError saying context_fn suppressed it, and
        # the context saw the original ValueError.
        class Context:
            # [CLASS DOC] HELPER: Context manager that records exception info and returns true
            # from exit to try to suppress it.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Do nothing on enter.
                pass

            def __exit__(self, exc_type, exc_value, traceback):
                # [FUNCTION DOC] HELPER: Store the exception type and value, and return true.
                self.exc_type = exc_type
                self.exc_value = exc_value
                return True

        context = Context()

        def context_fn():
            # [FUNCTION DOC] HELPER: Return the suppressing context for forward and a null context
            # for recompute.
            return context, contextlib.nullcontext()

        def fn(x):
            # [FUNCTION DOC] HELPER: Always raise ValueError saying the forward failed.
            raise ValueError("forward failed")

        x = torch.tensor(1.0, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "context_fn suppressed"):
            checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        self.assertEqual(context.exc_type, ValueError)
        self.assertEqual(str(context.exc_value), "forward failed")

    def test_checkpoint_warns_if_use_reentrant_not_passed_explcitly(self):
        # [FUNCTION DOC] Check checkpoint warns once when use_reentrant is not given explicitly
        # and does not warn when it is.
        a = torch.randn(1, requires_grad=True)

        # Passing explicitly should not warn
        self.assertNotWarn(lambda: checkpoint(lambda x: x, a, use_reentrant=False))

        # Not passing explicitly warns
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint(lambda x: x, a)

    def test_checkpoint_sequential_warns_if_use_reentrant_not_passed_explcitly(self):
        # [FUNCTION DOC] Check checkpoint_sequential warns once when use_reentrant is not given
        # explicitly and does not warn when it is.
        a = torch.randn(3, requires_grad=True)
        modules_list = [
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
        ]

        # Passing explicitly should not warn
        self.assertNotWarn(
            lambda: checkpoint_sequential(modules_list, 3, a, use_reentrant=False)
        )

        # Not passing explicitly warns
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint_sequential(modules_list, 3, a)

    @skipIfTorchDynamo("GraphExecGroup does not support compile")
    def test_checkpoint_graph_execution_group(self):
        # [FUNCTION DOC] Check GraphExecGroup around two successive grad calls on a
        # checkpointed region reduces the number of recomputations from three to two, and that
        # inside a group overlapping backward calls raise RuntimeError even with retain_graph.
        # Skipped under dynamo.
        def run(use_graph_execution_group):
            # [FUNCTION DOC] HELPER: Run two grad calls with or without the execution group
            # context and assert the recompute count.
            counter = [0]

            def fn(x):
                # [FUNCTION DOC] HELPER: Count the call and return two successive
                # sine-cosine results.
                counter[0] += 1
                y = x.sin().cos()
                z = y.sin().cos()
                return y, z

            x = torch.randn(3, 3, requires_grad=True)

            y, z = checkpoint(fn, x, use_reentrant=False)

            group = torch.utils.checkpoint.GraphExecGroup()

            ctx = contextlib.nullcontext()
            if use_graph_execution_group:
                ctx = group

            with ctx:
                (grad_y,) = torch.autograd.grad(
                    z, inputs=(y,), grad_outputs=(torch.ones(3, 3),)
                )

                (grad_x,) = torch.autograd.grad(
                    y,
                    inputs=(x,),
                    grad_outputs=(grad_y,),
                )

            if use_graph_execution_group:
                self.assertEqual(counter[0], 2)
            else:
                self.assertEqual(counter[0], 3)

        run(use_graph_execution_group=True)
        run(use_graph_execution_group=False)

        # Test the not actually disjoint case (using retain_graph=True since
        # otherwise autograd itself will catch this)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compute sine then cosine.
            return x.sin().cos()

        x = torch.randn(3, 3, requires_grad=True)
        out = checkpoint(fn, x, use_reentrant=False)
        with torch.utils.checkpoint.GraphExecGroup():
            # Under this context, we will enforce that two backward are disjoint
            # even if retain_graph=True.
            out.sum().backward(retain_graph=True)
            with self.assertRaisesRegex(
                RuntimeError, "Performing two backward calls that overlap"
            ):
                out.sum().backward()

    def test_checkpoint_detects_non_determinism(self):
        # [FUNCTION DOC] Check checkpoint detects non-deterministic recomputation: fewer or more
        # saved tensors, different metadata, and the debug message with the debug flag or global
        # debug setting; more saved tensors does not raise when early stop is on.
        def save_3_tensors(x):
            # [FUNCTION DOC] HELPER: Function saving three tensors.
            out = x.sin().exp()
            out = out.sin()
            return out

        def save_2_tensors(x):
            # [FUNCTION DOC] HELPER: Function saving two tensors.
            return x.sin().exp()

        def save_2_tensors_alt(x):
            # [FUNCTION DOC] HELPER: Function saving two tensors with different shapes from the
            # previous one.
            return x.sin() * torch.tensor([1.0, 2.0])

        def get_non_det_fn(orig_fn, recompute_fn):
            # [FUNCTION DOC] HELPER: Build a function that runs the original function on the first
            # call and the alternative afterwards.
            counter = [0]

            def fn(x):
                # [FUNCTION DOC] HELPER: Dispatch on a call counter between the original and
                # recompute functions.
                if counter[0] == 0:
                    counter[0] += 1
                    return orig_fn(x)
                else:
                    return recompute_fn(x)

            return fn

        a = torch.randn(1, requires_grad=True)

        # Save fewer tensors during recompute
        fn = get_non_det_fn(orig_fn=save_3_tensors, recompute_fn=save_2_tensors)
        with self.assertRaisesRegex(
            RuntimeError, "A different number of tensors was saved"
        ):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()

        # Save more tensors during recompute
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            with self.assertRaisesRegex(
                RuntimeError, "trying to save more tensors during recomputation"
            ):
                out = checkpoint(fn, a, use_reentrant=False)
                out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        # If early stopping is enabled, we would not raise (the results would be correct anyway)
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()

        # Save the same number of tensors but the shape is different
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)
        with self.assertRaisesRegex(RuntimeError, "tensors have different metadata"):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()

        # Get the debug message if debug=True
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError,
            "You are seeing this error because you passed `debug=True` to checkpoint",
        ):
            out = checkpoint(fn, a, use_reentrant=False, debug=True)
            out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError,
            "You are seeing this error because you passed `debug=True` to checkpoint",
        ):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
                out = checkpoint(fn, a, use_reentrant=False, debug=False)
                out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError, "Recomputed values for the following tensors have different"
        ):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(False):
                out = checkpoint(fn, a, use_reentrant=False, debug=True)
                out.backward()

    def test_checkpoint_error_suggests_mark_dynamic(self):
        # [FUNCTION DOC] Check the checkpoint error message for non-deterministic recomputation
        # suggests mark_dynamic and disabling the LRU cache.
        """CheckpointError messages should suggest mark_dynamic and lru config."""
        from torch.utils.checkpoint import checkpoint

        count = [0]

        def save_2_tensors(a):
            # [FUNCTION DOC] HELPER: Function saving two powers.
            b = a**2
            c = a**3
            return b + c

        def save_3_tensors(a):
            # [FUNCTION DOC] HELPER: Function saving three powers.
            b = a**2
            c = a**3
            d = a**4
            return b + c + d

        def get_non_det_fn(orig_fn, recompute_fn):
            # [FUNCTION DOC] HELPER: Build a function that runs the original on the first call and
            # the recompute variant afterwards.
            def fn(a):
                # [FUNCTION DOC] HELPER: Dispatch on a call counter between the original and
                # recompute functions.
                if count[0] == 0:
                    count[0] += 1
                    return orig_fn(a)
                return recompute_fn(a)

            return fn

        a = torch.randn(1, requires_grad=True)
        fn = get_non_det_fn(orig_fn=save_3_tensors, recompute_fn=save_2_tensors)
        with self.assertRaises(RuntimeError) as cm:
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        error_msg = str(cm.exception)
        self.assertIn(
            "Use torch._dynamo.mark_dynamic() to explicitly mark varying dimensions",
            error_msg,
        )
        self.assertIn(
            "Call torch._C._dynamo.eval_frame._set_lru_cache(False) to disable LRU cache",
            error_msg,
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=True,
    )
    def test_checkpoint_automatic_dynamic_graph_shadowing(self):
        # [FUNCTION DOC] Scenario check for automatic dynamic shapes shadowing a static graph
        # during checkpoint recompute: if backward raises RuntimeError, its message must include
        # the mark_dynamic and LRU cache suggestions. Config patched for automatic dynamic shapes.
        """Activation checkpointing + automatic dynamic shapes graph shadowing.

        When a compiled function is used inside checkpoint and another call
        with a different input size triggers automatic dynamic recompilation,
        the new dynamic graph is inserted at the front of the Dynamo cache
        and shadows the original static graph. During backward, checkpoint's
        recompute may hit the dynamic graph instead of the static graph used
        in the original forward, causing a saved-tensor mismatch.

        This test exercises the scenario described in PR #174993: the forward
        of a checkpointed layer uses a static graph (G0), then a subsequent
        call with a different batch size triggers automatic dynamic (G1 at
        cache front). During backward, checkpoint's recompute for the first
        layer hits G1 instead of G0.
        """
        import torch._dynamo
        from torch.utils.checkpoint import checkpoint

        torch._dynamo.reset()

        @torch.compile(backend="aot_eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function chaining sine, cosine and exp.
            # Multiple ops so the partitioner has save/recompute choices.
            # With static shapes the partitioner sees exact costs; with
            # dynamic shapes it uses size_hint fallbacks, potentially
            # making different save-vs-recompute decisions.
            a = x.sin()
            b = a.cos()
            c = (a * b).exp()
            return c

        # Step 1: Checkpointed forward with batch_size=4 → static graph G0
        x = torch.randn(4, 8, requires_grad=True)
        cp_out = checkpoint(fn, x, use_reentrant=False)

        # Step 2: Non-checkpointed call with batch_size=7 triggers automatic
        # dynamic recompilation → dynamic graph G1 inserted at cache front,
        # shadowing G0 for any future call (including checkpoint recompute).
        y = torch.randn(7, 8, requires_grad=True)
        plain_out = fn(y)

        loss = cp_out.sum() + plain_out.sum()

        # Step 3: Backward. Checkpoint recomputes fn(x) but now hits G1
        # (dynamic) instead of G0 (static). If the graphs save different
        # tensors, this raises CheckpointError with our diagnostic message.
        try:
            loss.backward()
        except RuntimeError as e:
            error_msg = str(e)
            # If the mismatch triggers, verify the error message includes
            # the automatic dynamic shapes suggestions.
            self.assertIn(
                "Use torch._dynamo.mark_dynamic() to explicitly mark varying "
                "dimensions",
                error_msg,
            )
            self.assertIn(
                "Call torch._C._dynamo.eval_frame._set_lru_cache(False) to "
                "disable LRU cache",
                error_msg,
            )

        torch._dynamo.reset()

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=True,
    )
    def test_checkpoint_automatic_dynamic_mark_dynamic_workaround(self):
        # [FUNCTION DOC] Check that marking the batch dimension dynamic up front avoids the graph
        # shadowing problem so backward succeeds. Config patched for automatic dynamic shapes.
        """mark_dynamic avoids the graph shadowing problem.

        By marking the varying dimension as dynamic upfront, both the
        checkpointed forward and the subsequent call compile a single
        dynamic graph from the start — no static-to-dynamic transition,
        no cache shadowing, no saved-tensor mismatch.
        """
        import torch._dynamo
        from torch.utils.checkpoint import checkpoint

        torch._dynamo.reset()

        @torch.compile(backend="aot_eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function chaining sine, cosine and exp.
            a = x.sin()
            b = a.cos()
            c = (a * b).exp()
            return c

        x = torch.randn(4, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        cp_out = checkpoint(fn, x, use_reentrant=False)

        y = torch.randn(7, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(y, 0)
        plain_out = fn(y)

        loss = cp_out.sum() + plain_out.sum()
        # Should succeed — both calls use the same dynamic graph.
        loss.backward()

        torch._dynamo.reset()

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=True,
    )
    def test_checkpoint_automatic_dynamic_lru_disabled_workaround(self):
        # [FUNCTION DOC] Check that disabling the dynamo LRU cache reordering avoids the graph
        # shadowing problem; the LRU setting and dynamo state are restored afterwards.
        """Disabling LRU cache reordering avoids the graph shadowing problem.

        With LRU disabled, new dynamic graphs are appended to the back of
        the cache instead of the front. The original static graph is still
        checked first, so checkpoint's recompute hits the same graph that
        the forward used.
        """
        import torch._dynamo
        from torch.utils.checkpoint import checkpoint

        torch._dynamo.reset()
        # Disable LRU cache reordering so static graphs stay at the front.
        torch._C._dynamo.eval_frame._set_lru_cache(False)

        try:

            @torch.compile(backend="aot_eager")
            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled (aot_eager) function chaining sine, cosine
                # and exp.
                a = x.sin()
                b = a.cos()
                c = (a * b).exp()
                return c

            x = torch.randn(4, 8, requires_grad=True)
            cp_out = checkpoint(fn, x, use_reentrant=False)

            y = torch.randn(7, 8, requires_grad=True)
            plain_out = fn(y)

            loss = cp_out.sum() + plain_out.sum()
            # Should succeed — LRU disabled means static graph G0 stays
            # first in cache, so checkpoint recompute still hits G0.
            loss.backward()
        finally:
            torch._C._dynamo.eval_frame._set_lru_cache(True)
            torch._dynamo.reset()

    def test_access_saved_tensor_twice_without_recomputation_works(self):
        # [FUNCTION DOC] Check recompute counts of non-reentrant checkpoint when reading a saved
        # result outside backward: each access recomputes (counts two and three), backward
        # recomputes again, and afterwards access raises RuntimeError about freed saved tensors.
        count = [0]

        def foo(a):
            # [FUNCTION DOC] HELPER: Count calls and compute a few products and an exp, returning
            # the exp.
            count[0] += 1
            b = a * a
            c = a * b
            d = torch.exp(a)
            return d

        a = torch.randn(5, requires_grad=True)
        d = checkpoint(foo, a, use_reentrant=False)
        self.assertEqual(count[0], 1)
        # Recomputed variables only persist within a particular backward call.
        # If _saved_result is accessed outside of a backward, it will trigger
        # a recompute. And afterwards, those recomputed results are immediately
        # cleared.
        d.grad_fn._saved_result
        self.assertEqual(count[0], 2)
        # Second access will trigger another recompute
        d.grad_fn._saved_result
        self.assertEqual(count[0], 3)
        # Backward clears the saved variable
        d.sum().backward()
        self.assertEqual(count[0], 4)
        # Now it raises an error
        with self.assertRaisesRegex(
            RuntimeError,
            "or directly access saved tensors after they have already been freed",
        ):
            d.grad_fn._saved_result

    @slowTest
    @parametrize("input_requires_grad", [True, False])
    def test_checkpointing_without_reentrant(self, input_requires_grad):
        # [FUNCTION DOC] Slow parametrized test: non-reentrant checkpoint with keyword data
        # argument over two thousand inputs gives the same parameter grads as without checkpoint,
        # whether or not inputs require grad.
        """
        Basic test for checkpoint without reentrant autograd.
        """
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp),
        )

        # Module holder for testing activation checkpointing with no_reentrant
        # supports kwargs.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module wrapper around another module taking a data keyword.
            def __init__(self, mod):
                # [FUNCTION DOC] HELPER: Store the wrapped module.
                super().__init__()
                self.module = mod

            def forward(self, data):
                # [FUNCTION DOC] HELPER: Apply the wrapped module to the data.
                return self.module(data)

        module = MyModule(mod=module)

        # Run model with and without checkpointing and verify gradients are
        # equivalent, regardless of if inputs require grads or not.
        module_copy = deepcopy(module)

        feat_combined = []
        feat_combined_no_checkpoint = []
        for _ in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = input_requires_grad
            data_r_copy = data_r.clone()
            feat_r = checkpoint(module, data=data_r, use_reentrant=False)
            feat_combined.append(feat_r)
            feat_r_no_checkpoint = module_copy(data_r)
            feat_combined_no_checkpoint.append(feat_r_no_checkpoint)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()
        mean_combined_no_checkpoint = torch.stack(feat_combined_no_checkpoint).mean()
        mean_combined_no_checkpoint.backward()

        for checkpoint_param, param in zip(
            module.parameters(), module_copy.parameters()
        ):
            self.assertEqual(checkpoint_param.grad, param.grad)

    def test_checkpoint_valid_reset_on_error(self):
        # [FUNCTION DOC] Check that after a reentrant checkpoint error from grad, a later
        # reentrant checkpoint plus backward works.
        a = torch.randn(2, 2, requires_grad=True)

        with self.assertRaisesRegex(
            Exception, "torch.utils.checkpoint is incompatible"
        ):
            b = checkpoint(torch.exp, a, use_reentrant=True).sum()
            torch.autograd.grad(b, (a,))

        c = checkpoint(torch.exp, a, use_reentrant=True).sum()
        c.backward()

    @parametrize("use_reentrant", [True, False])
    def test_checkpointing_without_reentrant_detached_tensor(self, use_reentrant):
        # [FUNCTION DOC] Parametrized by reentrancy: checkpointing a module run under no_grad;
        # reentrant mode raises RuntimeError that no output requires grad, non-reentrant does not.
        class NoGradModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running two linear layers under no_grad.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the two linear layers without bias.
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.lin2 = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Run both layers with grad disabled.
                with torch.no_grad():
                    return self.lin2(self.linear(x))

        module = NoGradModule()

        err_ctx = (
            self.assertRaisesRegex(
                RuntimeError, "none of output has requires_grad=True"
            )
            if use_reentrant
            else contextlib.nullcontext()
        )

        a = torch.randn(2, 2, requires_grad=True)
        for _ in range(3):
            with err_ctx:
                # out does not require grad
                out = checkpoint(module, a, use_reentrant=use_reentrant)
                # Make loss require grad, otherwise we would run into
                # "element 0 of tensors does not require grad and does not have a grad_fn"
                out += a
                out.sum().backward()

    def test_checkpointing_without_reentrant_saved_object_identity(self):
        # [FUNCTION DOC] Check that a tensor saved by a custom Function not requiring grad keeps
        # its object identity in backward, with and without non-reentrant checkpoint.
        x_backward = None

        class Test(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving the second input and storing the unpacked tensor
            # in an outer variable during backward.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Save the second input and return the first.
                ctx.save_for_backward(y)
                return x

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Store the unpacked saved tensor outside and return the
                # gradient and None.
                nonlocal x_backward
                (x_backward,) = ctx.saved_tensors
                return x, None

        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=False)

        Test.apply(a, b).backward()
        self.assertIs(b, x_backward)

        x_backward = None
        checkpoint(Test.apply, a, b, use_reentrant=False).backward()
        self.assertIs(b, x_backward)

    def test_checkpointing_without_reentrant_correct_grad(self):
        # [FUNCTION DOC] Check gradients from non-reentrant checkpoint equal plain gradients via
        # backward and via grad.
        """
        Verifies that correct gradients are calculated for checkpoint
        without reentrant autograd, for both backward() and autograd.grad().
        """
        a = torch.randn(2, 2, requires_grad=True)

        b = torch.exp(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None
        c = checkpoint(torch.exp, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        a.grad = None
        d = checkpoint(torch.exp, a, use_reentrant=False).sum()
        (d_grad,) = torch.autograd.grad(d, (a,))

        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_checkpointing_without_reentrant_with_block_mask(self):
        # [FUNCTION DOC] Check non-reentrant checkpoint accepts a flex attention BlockMask as
        # input (registering it as a pytree node if needed). Requires CUDA.
        from torch.nn.attention.flex_attention import BlockMask, create_block_mask
        from torch.utils._pytree import register_pytree_node, SUPPORTED_NODES

        if BlockMask not in SUPPORTED_NODES:
            register_pytree_node(
                BlockMask,
                BlockMask._flatten,
                BlockMask._unflatten,
                flatten_with_keys_fn=BlockMask._flatten_with_keys,
                serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
            )

        block_mask = create_block_mask(
            lambda b, h, q, kv: q >= kv, B=1, H=1, Q_LEN=128, KV_LEN=128
        )
        x = torch.randn(4, 128, device="cuda")

        result = checkpoint(lambda x, mask: x * 2, x, block_mask, use_reentrant=False)
        self.assertEqual(result, x * 2)

    @skipIfXpu(msg="torch._C._scatter Not implemented on XPU, issue #143239")
    def test_checkpointing_without_reentrant_dataparallel(self):
        # [FUNCTION DOC] Check gradients of non-reentrant checkpoint combined with DataParallel
        # match the plain model. Skipped on XPU.
        """
        Verifies gradient correctness when checkpoint without reentrant autograd
        is used in conjunction with DataParallel.
        """

        class LinearModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one bias-free linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the layer.
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Apply the layer.
                return self.linear(inp)

        a = torch.randn(2, 2, requires_grad=True)
        if torch.cuda.is_available():
            a = a.cuda()

        model = LinearModule()
        if torch.cuda.is_available():
            model = model.cuda()

        b = deepcopy(model)(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None

        module = torch.nn.DataParallel(deepcopy(model))
        c = checkpoint(module, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        self.assertEqual(b_grad, c_grad)

    def test_checkpointing_without_reentrant_parameter_used_in_an_out(self):
        # [FUNCTION DOC] Check a parameter used both outside and inside a non-reentrant checkpoint
        # triggers its tensor hook only once.
        """
        Ensures that gradient hooks are only called once per tensor.
        """
        w = torch.randn(10, 10, requires_grad=True)
        count = 0

        def hook(grad):
            # [FUNCTION DOC] HELPER: Hook counting calls in an enclosing variable.
            nonlocal count
            count += 1

        w.register_hook(hook)
        x = torch.rand(10, 10, requires_grad=True)
        h = w * x  # Using w outside the checkpoint
        out = checkpoint(
            lambda x: w * x, h, use_reentrant=False
        )  # Using w inside the checkpoint

        out.sum().backward()
        # should only call hook once
        self.assertEqual(count, 1)

    def test_checkpointing_without_reentrant_arbitrary_input_output(self):
        # [FUNCTION DOC] Check non-reentrant checkpoint with dict input and dict output gives the
        # same output and parameter grads as no checkpoint.
        """
        Ensures checkpointing without reentrant autograd works with functions
        with arbitrary input/output structures.
        """

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bias-free linear layer taking a dict with a tensor
            # and returning a dict with a result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the layer.
                super().__init__()
                self.layer = torch.nn.Linear(5, 5, bias=False)

            def forward(self, dict_input):
                # [FUNCTION DOC] HELPER: Apply the layer to the tensor entry and wrap the result
                # in a dict.
                tensor = dict_input["tensor"]
                return {"result": self.layer(tensor)}

        model_no_checkpoint = MyModel()
        model_checkpoint_without_reentrant = deepcopy(model_no_checkpoint)

        inp = {"tensor": torch.randn(5, 5)}

        out_no_checkpoint = model_no_checkpoint(inp)["result"].sum()

        out_checkpoint = checkpoint(
            model_checkpoint_without_reentrant, inp, use_reentrant=False
        )["result"].sum()

        self.assertEqual(out_checkpoint, out_no_checkpoint)

        out_no_checkpoint.backward()
        out_checkpoint.backward()

        for param, checkpoint_param in zip(
            model_no_checkpoint.parameters(),
            model_checkpoint_without_reentrant.parameters(),
        ):
            self.assertEqual(param.grad, checkpoint_param.grad)

    def test_callback_adds_callback(self):
        # [FUNCTION DOC] Check a callback queued during a backward callback is also executed (two
        # calls total).
        called = [0]

        def callback_final():
            # [FUNCTION DOC] HELPER: Increment the call counter.
            called[0] += 1

        def callback_adds_callback():
            # [FUNCTION DOC] HELPER: Increment the call counter and queue the final callback.
            called[0] += 1
            Variable._execution_engine.queue_callback(callback_final)

        class MyFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward queues the callback that queues another.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Return the input.
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Queue the first callback with the engine and return
                # the gradient.
                Variable._execution_engine.queue_callback(callback_adds_callback)
                return grad

        a = torch.rand((3, 3), requires_grad=True)
        b = MyFunc.apply(a)
        b.sum().backward()

        self.assertEqual(called[0], 2)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_callback_propagates_errors_from_device_thread(self):
        # [FUNCTION DOC] Check an error raised in a callback queued from a hook on a CUDA tensor
        # propagates from backward. Requires CUDA.
        def callback():
            # [FUNCTION DOC] HELPER: Raise RuntimeError with a short message.
            raise RuntimeError("blah")

        def hook_with_callback(*args):
            # [FUNCTION DOC] HELPER: Hook that queues the failing callback with the engine.
            torch.autograd.Variable._execution_engine.queue_callback(callback)

        t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("cuda"))
        t.register_hook(hook_with_callback)
        output = t**2
        loss = output.sum()

        with self.assertRaisesRegex(RuntimeError, "blah"):
            loss.backward()

    def _test_reentrant_with_callbacks(self, install_callbacks_in_depths):
        # [FUNCTION DOC] HELPER: Run a reentrant backward where callbacks are queued at the given
        # nesting depths, and return counters of outer and inner callback runs.
        counter = {}
        counter["inner"] = 0
        counter["outer"] = 0

        def inc_inner_counter():
            # [FUNCTION DOC] HELPER: Increment the inner counter.
            counter["inner"] += 1

        def inc_outer_counter():
            # [FUNCTION DOC] HELPER: Increment the outer counter.
            counter["outer"] += 1

        class MyFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward queues the inner callback if depth one
            # is enabled.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Return the input.
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                # [FUNCTION DOC] HELPER: Optionally queue the inner callback and return
                # the gradient.
                if 1 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_inner_counter)

                return input

        class MyReentrantFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward optionally queues the outer callback and
            # runs a nested backward through the inner Function.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Return the input.
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                # [FUNCTION DOC] HELPER: Optionally queue the outer callback, run a nested
                # backward and return the gradient.
                if 0 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_outer_counter)
                # Reentrant backward call.
                tmp_inp = input.detach().requires_grad_()
                with torch.enable_grad():
                    tmp_out = (MyFunc.apply(tmp_inp)).sum()
                tmp_out.backward()
                return input

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = MyReentrantFunc.apply(t1)
        t3 = t2.sum()
        torch.autograd.backward([t3])

        return counter

    def test_reentrant_with_callbacks_depth_0(self):
        # [FUNCTION DOC] Check a callback queued at depth zero of a reentrant backward runs once
        # and the inner one not at all.
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([0])
        self.assertEqual(ret["outer"], 1)
        self.assertEqual(ret["inner"], 0)

    def test_reentrant_with_callbacks_depth_1(self):
        # [FUNCTION DOC] Check a callback queued at depth one runs once and the outer one not
        # at all.
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([1])
        self.assertEqual(ret["outer"], 0)
        self.assertEqual(ret["inner"], 1)

    def test_reentrant_with_callbacks_both_depths(self):
        # [FUNCTION DOC] Check callbacks at both depths each run once.
        # Verify callback is called twice.
        ret = self._test_reentrant_with_callbacks([0, 1])
        self.assertEqual(ret["outer"], 1)
        self.assertEqual(ret["inner"], 1)

    def test_reentrant_with_leaf_variable_hook(self):
        # [FUNCTION DOC] Check a tensor hook on a leaf that removes itself and runs a nested
        # backward while modifying the grad attribute does not fail.
        handle = None
        param = torch.rand(10, requires_grad=True)

        def add_gradient_penalty_to_grad(grad):
            # [FUNCTION DOC] HELPER: Hook that removes itself, runs a nested backward penalty
            # computation, and returns the gradient plus the nested gradient, restoring the
            # grad attribute.
            handle.remove()
            old_param_grad = grad
            param.grad = None
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                new_param = param.detach().requires_grad_()
                out = ((g * 2) + new_param).sum()
                out.backward()
            res = g.grad + grad
            param.grad = old_param_grad
            return res

        handle = param.register_hook(add_gradient_penalty_to_grad)
        # Forward pass
        tmp = param * param
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()

    def test_reentrant_with_non_leaf_variable_hook(self):
        # [FUNCTION DOC] Check a hook on a non-leaf that removes itself and runs a nested backward
        # changes the final parameter gradient to six times the parameter.
        handle = None
        param = torch.rand(10, requires_grad=True)

        def manual_increase_gradient(grad):
            # [FUNCTION DOC] HELPER: Hook that removes itself, runs a nested backward and returns
            # the gradient plus the nested gradient.
            handle.remove()
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                out = ((g * 2) + 5).sum()
                out.backward()
            res = g.grad + grad
            return res

        # Forward pass
        tmp = param * param
        handle = tmp.register_hook(manual_increase_gradient)
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()
        self.assertEqual(param.grad, 6 * param)

    def test_grad_fn_attr_bindings(self):
        # [FUNCTION DOC] Check the generated getter bindings of grad_fn saved attributes: types of
        # saved tensors, lists, optionals, sizes, scalars, strings, bools and floats, errors after
        # the saved data is freed, and raw saved tensor hook registration limits.
        # Check that the getter of each type returns what we want
        # See `gen_autograd_functions.py` for how the getters are generated
        #
        # This test is only meant to check if the codegen'd bindings work
        # Please help update this test if you update the names of any the fields we check!
        #
        a = torch.ones(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        out1 = torch.stack([a, b], dim=0)
        out2 = (a * 2) * b
        # TODO: I don't think we have a backward saving a list of tensors
        #       at the moment. It used to be stack, but for no reason...
        #       see discussion in #84993
        # self.assertEqual(out.grad_fn._saved_tensors, (a, b))              # TewnsorList -> Tuple[Tensor]
        self.assertEqual(out2.grad_fn._saved_self, a * 2)
        self.assertIsInstance(out2.grad_fn._saved_self, torch.Tensor)
        self.assertIsInstance(
            out2.grad_fn._raw_saved_self, torch._C._autograd.SavedTensor
        )
        self.assertEqual(out1.grad_fn._saved_dim, 0)  # int64_t -> int
        self.assertIsInstance(out1.grad_fn._saved_dim, int)

        out2.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)

        out2.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out2.grad_fn._saved_self
        # TODO: interestingly, this only happens if indexing into a list grad_fn._raw_saved_tensors[0],
        #       not when using a saved tensor, see discussion in #84993
        # with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
        #     out2.grad_fn._raw_saved_self
        self.assertEqual(out1.grad_fn._saved_dim, 0)

        a = torch.ones(2, 2, requires_grad=True)
        indices = torch.tensor([0, 1])
        out = a[:, indices]
        self.assertEqual(
            out.grad_fn._saved_indices, (None, indices)
        )  # c10::List<std::optional<Tensor>> -> Tuple[Tensor?]
        self.assertIsInstance(out.grad_fn._saved_indices[1], torch.Tensor)
        self.assertIsInstance(
            out.grad_fn._raw_saved_indices[1], torch._C._autograd.SavedTensor
        )
        self.assertEqual(
            out.grad_fn._saved_self_sym_sizes, a.shape
        )  # SymIntArrayRef -> Tuple[SymInt]
        self.assertIsInstance(out.grad_fn._saved_self_sym_sizes[0], int)

        out.grad_fn._raw_saved_indices[1].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, "None is forbidden"):
            out.grad_fn._raw_saved_indices[0].register_hooks(lambda x: x, lambda x: x)

        out = a.mean()
        self.assertEqual(
            out.grad_fn._saved_self_sym_sizes, a.shape
        )  # IntArrayRef -> Tuple[int]

        a = torch.ones(2, 2, requires_grad=True)
        out = a * a
        out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after it has been freed"):
            out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, 4, mode="linear")
        self.assertEqual(
            out.grad_fn._saved_output_size, (4,)
        )  # std::optional<IntArrayRef> -> int[]?
        self.assertIsInstance(out.grad_fn._saved_output_size[0], int)
        self.assertEqual(out.grad_fn._saved_align_corners, False)  # bool -> bool
        self.assertIsInstance(out.grad_fn._saved_align_corners, bool)
        if hasattr(out.grad_fn, "_saved_scale_factors"):
            self.assertIsNone(
                out.grad_fn._saved_scale_factors
            )  # std::optional<ArrayRef<double>> -> float[]?
        else:
            self.assertIsNone(
                out.grad_fn._saved_scales
            )  # std::optional<ArrayRef<double>> -> float[]?

        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = nn.Conv2d(1, 1, 3)(a)
        self.assertEqual(
            out.grad_fn._saved_bias_sym_sizes_opt, (1,)
        )  # std::optional<SymIntArrayRef> -> SymInt[]?
        out = nn.Conv2d(1, 1, 3, bias=False)(a)
        # TODO: This is BAD! we converted a std::nullopt into a (0,)
        self.assertEqual(out.grad_fn._saved_bias_sym_sizes_opt, (0,))

        a = torch.ones(1, 3, 3, requires_grad=True)
        out = torch.addbmm(a.squeeze(0), a, a)
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_0, 1)  # int64_t
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_1, 3)  # int64_t

        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = torch.nn.functional.unfold(a, 3)
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_2, 3)  # SymInt
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_1, 3)  # SymInt

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, scale_factor=0.5, mode="linear")
        self.assertEqual(out.grad_fn._saved_scales, 0.5)

        a = torch.ones(2, 2, requires_grad=True)
        out = torch.pdist(a, p=1)
        self.assertEqual(out.grad_fn._saved_p, 1.0)  # double -> float
        self.assertIsInstance(out.grad_fn._saved_p, float)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.logit(a, 1.0)
        self.assertEqual(out.grad_fn._saved_eps, 1.0)  # c10:optional<double> -> float?
        self.assertIsInstance(out.grad_fn._saved_eps, float)
        out = torch.logit(a)
        self.assertIsNone(out.grad_fn._saved_eps)

        if torch._C.has_lapack:
            a = torch.ones(1, 1, requires_grad=True)
            q, r = torch.linalg.qr(a, mode="reduced")
            self.assertEqual(q.grad_fn._saved_mode, "reduced")  # std::string -> str

        a = torch.tensor([1.0], requires_grad=True)
        out = torch.div(a, 2.0, rounding_mode="trunc")
        self.assertEqual(
            out.grad_fn._saved_rounding_mode, "trunc"
        )  # std::optional<std::string> -> str?
        out = torch.div(a, 2.0, rounding_mode=None)
        self.assertIsNone(
            out.grad_fn._saved_rounding_mode
        )  # std::optional<std::string> -> str?

        x = torch.zeros(5, requires_grad=True)
        out = torch.threshold(x, threshold=(1 + 0j), value=(1 + 0j))
        self.assertIsInstance(
            out.grad_fn._saved_threshold, complex
        )  # Scalar(complex double) -> complex
        cfloat = torch.tensor(1 + 0j, dtype=torch.complex64)
        out = torch.threshold(x, threshold=cfloat, value=(1 + 0j))
        self.assertIsInstance(
            out.grad_fn._saved_threshold, complex
        )  # Scalar(complex float) -> complex
        out = torch.threshold(x, threshold=1.0, value=1.0)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, float
        )  # Scalar(floating point) -> float
        out = torch.threshold(x, threshold=1, value=1)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, int
        )  # Scalar(integral) -> int
        out = torch.threshold(x, threshold=False, value=False)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, bool
        )  # Scalar(bool) -> bool

        a = torch.ones(2, 2, requires_grad=True)
        out = a.as_strided((3,), (1,), 1)
        self.assertEqual(
            out.grad_fn._saved_storage_offset, 1
        )  # c10:optional<int64_t> -> int?
        self.assertIsInstance(out.grad_fn._saved_storage_offset, int)
        out = a.as_strided((3,), (1,))
        self.assertIsNone(out.grad_fn._saved_storage_offset)

        a = torch.ones(2, requires_grad=True)
        out = torch.tanh(a)
        self.assertEqual(out, out.grad_fn._saved_result)  # saved variable when output

        a = torch.randn(3, 5, requires_grad=True)
        b = torch.tensor([1, 0, 4])
        loss = nn.NLLLoss()
        out = loss(a, b)
        self.assertIsNone(out.grad_fn._saved_weight)
        loss = nn.NLLLoss(weight=torch.ones((5,)))
        out = loss(a, b)
        self.assertEqual(
            out.grad_fn._saved_weight, torch.ones((5,))
        )  # c10:optional<Tensor> -> Tensor?

        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out.grad_fn._saved_weight

        num_tensors = 3
        input_tensors = [
            torch.ones(2, 2, requires_grad=True) for _ in range(num_tensors)
        ]
        scalars = [
            0.0 for _ in range(num_tensors)
        ]  # ArrayRef<Scalar> -> Tuple[Scalar, ...]
        results = torch._foreach_maximum(input_tensors, scalars)
        for t in results:
            self.assertEqual(t.grad_fn._saved_scalars, scalars)

    def test_get_data_and_hooks_from_raw_saved_variable(self):
        # [FUNCTION DOC] Check the data and unpack hook of a raw saved tensor: packed object
        # returned as is when hooks are set, detached data for saved outputs, and un-detached data
        # for saved inputs.
        def pack_hook(t):
            # [FUNCTION DOC] HELPER: Pack hook returning the tensor unchanged.
            return t

        def unpack_hook(t):
            # [FUNCTION DOC] HELPER: Unpack hook returning the value unchanged.
            return t

        a = torch.tensor(2.0, requires_grad=True)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            b = a**2

        c = b.exp()
        d = c**2

        pow_sv = b.grad_fn._raw_saved_self
        exp_sv = c.grad_fn._raw_saved_result
        pow2_sv = d.grad_fn._raw_saved_self

        # Returns the packed object as-is
        self.assertTrue(pow_sv.data is a)
        self.assertTrue(pow_sv.unpack_hook is unpack_hook)
        # Returns the detached data when the output/leaf is saved
        self.assertFalse(exp_sv.data is c)
        self.assertIsNone(exp_sv.unpack_hook)
        # Returns the un-detached data when input is saved
        self.assertTrue(pow2_sv.data is c)
        self.assertIsNone(pow2_sv.unpack_hook)

    def test_saved_tensor_constructor_with_hooks(self):
        # [FUNCTION DOC] Check the private saved tensor constructor runs the pack hook once at
        # creation and the unpack hook on each unpack.
        pack_count = [0]
        unpack_count = [0]

        def pack_hook(x):
            # [FUNCTION DOC] HELPER: Pack hook counting calls and returning the tensor.
            pack_count[0] += 1
            return x

        def unpack_hook(x):
            # [FUNCTION DOC] HELPER: Unpack hook counting calls and returning the tensor.
            unpack_count[0] += 1
            return x

        a = torch.randn(5, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            saved = torch._C._autograd._make_saved_tensor(a, is_output=False)
            self.assertEqual(pack_count[0], 1)
            self.assertEqual(unpack_count[0], 0)

            unpacked = saved.unpack()
            self.assertEqual(pack_count[0], 1)
            self.assertEqual(unpack_count[0], 1)

            unpacked2 = saved.unpack()
            self.assertEqual(pack_count[0], 1)
            self.assertEqual(unpack_count[0], 2)

        # Check tensor equality outside the hooks context to avoid
        # triggering additional pack hooks from assertEqual internals
        self.assertEqual(unpacked, a)
        self.assertEqual(unpacked2, a)

    def test_saved_tensor_constructor_forbidden_without_flag(self):
        # [FUNCTION DOC] Check constructing a SavedTensor directly from Python raises RuntimeError
        # saying it is forbidden.
        a = torch.randn(5, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to create a SavedTensor object from Python is forbidden",
        ):
            torch.autograd.SavedTensor()

    def test_custom_function_saved_tensors(self):
        # [FUNCTION DOC] Check saved tensors of a custom Function including a None, the raw saved
        # tensor objects, hook registration errors (None, wrong argument types, already set), and
        # errors after backward frees them.
        def getFn(save=True):
            # [FUNCTION DOC] HELPER: Build a custom Function class that saves its input and None,
            # or nothing, depending on the flag.
            class MyFn(Function):
                # [CLASS DOC] HELPER: Identity Function optionally saving its input and None.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Save the input and None if requested; return
                    # the input.
                    if save:
                        ctx.save_for_backward(x, None)
                    return x

                @staticmethod
                def backward(ctx, g):
                    # [FUNCTION DOC] HELPER: Return the gradient.
                    return g

            return MyFn

        a = torch.randn(5, requires_grad=True)

        y = getFn(True).apply(a)

        self.assertEqual((a, None), y.grad_fn.saved_tensors)
        saved = y.grad_fn._raw_saved_tensors
        self.assertIsInstance(saved[0], torch._C._autograd.SavedTensor)
        # We can't tell the underlying tensor is None without unpacking it
        self.assertIsInstance(saved[1], torch._C._autograd.SavedTensor)

        # We catch that error when the user calls register_hooks on it
        with self.assertRaisesRegex(RuntimeError, "None is forbidden"):
            saved[1].register_hooks(lambda x: x, lambda x: x)

        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(lambda x: x)
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(1, 1)
        saved[0].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, "already been set"):
            saved[0].register_hooks(lambda x: x, lambda x: x)
        y.sum().backward()

        # Using a reference to the SavedTensor object after the
        # saved variables have been released can lead to undefined behavior
        del saved
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn._raw_saved_tensors
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn.saved_tensors

        y = getFn(False).apply(a)
        self.assertEqual(y.grad_fn.saved_tensors, ())
        self.assertEqual(y.grad_fn._raw_saved_tensors, ())

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX, "https://github.com/pytorch/pytorch/issues/180489"
    )
    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads(self):
        # [FUNCTION DOC] Test the boxed gradient calling convention for custom Functions without
        # compile: backward receives one mutable list of grads which it clears. Skipped on Linux
        # and ASAN builds and under dynamo.
        """Test boxed_grads_call mechanism without torch.compile.

        With boxed_grads_call, backward receives a single mutable list
        of grads instead of individual grad arguments."""

        class BoxedFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function with boxed gradient calling convention that doubles
            # its input.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it doubled.
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Assert the gradient container is a one-element list,
                # clear it and return the gradient doubled.
                self.assertIsInstance(grads, list)
                self.assertEqual(len(grads), 1)
                grad = grads[0]
                grads.clear()
                (x,) = ctx.saved_tensors
                return grad * 2

        x = torch.randn(4, requires_grad=True)
        out = BoxedFunc.apply(x)
        out.sum().backward()
        # d/dx (2x).sum() = 2
        self.assertEqual(x.grad, torch.full_like(x, 2.0))

    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_multi_output(self):
        # [FUNCTION DOC] Check boxed gradients with two outputs: the list has one entry per
        # output. Skipped under dynamo.
        """Boxed grads with multiple outputs — grads list has one
        entry per output."""

        class MultiOut(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function returning input doubled and tripled.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return the doubled and tripled copies.
                ctx.save_for_backward(x)
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Assert two boxed gradients, clear the list and combine
                # them with weights 2 and 3.
                self.assertIsInstance(grads, list)
                self.assertEqual(len(grads), 2)
                (x,) = ctx.saved_tensors
                g1, g2 = grads
                grads.clear()
                return g1 * 2 + g2 * 3

        x = torch.randn(4, requires_grad=True)
        a, b = MultiOut.apply(x)
        (a.sum() + b.sum()).backward()
        # forward: a=2x, b=3x; backward: grad_a=1, grad_b=1
        # return grad_a*2 + grad_b*3 = 2+3 = 5
        self.assertEqual(x.grad, torch.full_like(x, 5.0))

    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_no_extra_refs(self):
        # [FUNCTION DOC] Check that after clearing the boxed list in backward and deleting the
        # local reference, the gradient is freed, showing the framework keeps no extra reference.
        # Skipped under dynamo.
        """Framework holds no extra refs to grads with boxed calling convention.

        After removing the grad from the boxed list, a weakref to it should
        become dead — proving the framework released all its refs."""

        class CheckRefs(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function whose backward checks a weak reference
            # to the gradient dies after clearing the list.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it doubled.
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Clone the gradient, clear the list and delete the
                # reference, assert the weak reference is dead and return the clone doubled.
                grad = grads[0]
                grad_for_compute = grad.clone()
                ref = weakref.ref(grad)
                grads.clear()
                del grad
                self.assertTrue(ref() is None)
                (x,) = ctx.saved_tensors
                return grad_for_compute * 2

        x = torch.randn(4, requires_grad=True)
        out = CheckRefs.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, torch.full_like(x, 2.0))

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX, "https://github.com/pytorch/pytorch/issues/180521"
    )
    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_cleanup_on_error(self):
        # [FUNCTION DOC] Check backward raising an error with boxed gradients propagates the error
        # without leaking the grads list. Skipped on Linux and ASAN builds and under dynamo.
        """Grads list is not leaked when backward raises."""

        class FailingBwd(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function whose backward always
            # raises RuntimeError.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input doubled.
                return x * 2

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Raise RuntimeError with an intentional failure message.
                raise RuntimeError("intentional failure")

        x = torch.randn(4, requires_grad=True)
        out = FailingBwd.apply(x)
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            out.sum().backward()

    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_chain(self):
        # [FUNCTION DOC] Check chaining two boxed-gradient Functions: each backward gets its own
        # grads list and runs in order. Skipped under dynamo.
        """Two boxed-grads functions chained — each gets its own grads list."""

        call_order = []

        class Mul2(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function doubling the input and recording its
            # backward call.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it doubled.
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Record the call, clear the gradient list and return the
                # gradient doubled.
                call_order.append("Mul2")
                grad = grads[0]
                grads.clear()
                (x,) = ctx.saved_tensors
                return grad * 2

        class Mul3(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function tripling the input.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it tripled.
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Record the call, clear the gradient list and return the
                # gradient tripled.
                call_order.append("Mul3")
                grad = grads[0]
                grads.clear()
                (x,) = ctx.saved_tensors
                return grad * 3

        x = torch.randn(4, requires_grad=True)
        out = Mul3.apply(Mul2.apply(x))
        out.sum().backward()
        # backward order: Mul3 then Mul2
        self.assertEqual(call_order, ["Mul3", "Mul2"])
        # d/dx (3 * 2 * x).sum() = grad 1 → Mul3 bwd: 1*3=3 → Mul2 bwd: 3*2=6
        self.assertEqual(x.grad, torch.full_like(x, 6.0))

    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_none_grads(self):
        # [FUNCTION DOC] Check boxed gradients when materialize_grads is off: an unused
        # output appears as None in the list, and the callee may set an entry to None.
        # Skipped under dynamo.
        """Boxed grads with materialize_grads=False and partial None grads.

        When only some outputs are used in the loss and materialize_grads
        is False, the grads list contains None for unused outputs. Verify
        that boxed_grads_call handles this correctly."""

        class TwoOutput(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function with two outputs and gradient
            # materialization turned off.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input, disable gradient materialization and
                # return the doubled and tripled copies.
                ctx.save_for_backward(x)
                ctx.set_materialize_grads(False)
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Assert the first boxed gradient is None and the second is
                # not, null out the second entry and return it tripled.
                self.assertIsInstance(grads, list)
                self.assertEqual(len(grads), 2)
                # Only second output is used in loss, first grad is None
                self.assertIsNone(grads[0])
                self.assertIsNotNone(grads[1])
                (x,) = ctx.saved_tensors
                g2 = grads[1]
                grads[1] = None
                return g2 * 3

        x = torch.randn(4, requires_grad=True)
        a, b = TwoOutput.apply(x)
        # Only use b in the loss — a gets no gradient
        b.sum().backward()
        # forward: b=3x; backward: grad_b=1, return 1*3=3
        self.assertEqual(x.grad, torch.full_like(x, 3.0))

    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_materialize_grads(self):
        # [FUNCTION DOC] Check boxed gradients with default materialization: an unused output gets
        # a zero tensor in the list. Skipped under dynamo.
        """boxed_grads_call works with materialize_grads(True).

        When materialize_grads is True (default) and some outputs have no
        gradient, the engine materializes zero tensors. Verify boxed_grads_call
        delivers these materialized zeros in the grads list."""

        class TwoOutputMaterialized(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function with two outputs and
            # materialized gradients.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return the doubled and tripled copies.
                ctx.save_for_backward(x)
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Assert the first gradient is a zero tensor, clear the
                # list and combine the gradients with weights 2 and 3.
                self.assertIsInstance(grads, list)
                self.assertEqual(len(grads), 2)
                # materialize_grads is True (default), so grads[0] is a zero
                # tensor even though first output wasn't used in the loss
                self.assertIsNotNone(grads[0])
                self.assertEqual(grads[0], torch.zeros_like(grads[1]))
                self.assertIsNotNone(grads[1])
                (x,) = ctx.saved_tensors
                g1, g2 = grads[0], grads[1]
                grads.clear()
                return g1 * 2 + g2 * 3

        x = torch.randn(4, requires_grad=True)
        a, b = TwoOutputMaterialized.apply(x)
        # Only use b in the loss — a gets a materialized zero grad
        b.sum().backward()
        # forward: b=3x; backward: g1=0, g2=1, return 0*2 + 1*3 = 3
        self.assertEqual(x.grad, torch.full_like(x, 3.0))

    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_direct_apply(self):
        # [FUNCTION DOC] Check both paths into a boxed-gradient backward: the engine and a direct
        # apply of the grad_fn node, both delivering a list; results are 5 and 8 respectively.
        # Skipped under dynamo.
        """Test both paths into backward with boxed_grads_call.

        Path 1 (C++ engine): .backward() goes through C++ PyNode::apply,
        which calls apply_boxed with grads in a mutable list.

        Path 2 (direct grad_fn.apply()): bypasses C++, apply() boxes
        grads into a list before calling backward."""

        received_grads = []

        class BoxedMultiOut(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function with two outputs that records the
            # received gradient list.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return the doubled and tripled copies.
                ctx.save_for_backward(x)
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Record the gradient list and combine the two gradients
                # with weights 2 and 3.
                received_grads.append(grads)
                (x,) = ctx.saved_tensors
                return grads[0] * 2 + grads[1] * 3

        # Path 1: .backward() — C++ engine calls apply_boxed
        x1 = torch.randn(4, requires_grad=True)
        a1, b1 = BoxedMultiOut.apply(x1)
        (a1.sum() + b1.sum()).backward()
        self.assertIsInstance(received_grads[0], list)
        self.assertEqual(len(received_grads[0]), 2)
        self.assertEqual(x1.grad, torch.full_like(x1, 5.0))

        # Path 2: grad_fn.apply() — apply() boxes grads
        received_grads.clear()
        x2 = torch.randn(4, requires_grad=True)
        a2, b2 = BoxedMultiOut.apply(x2)
        result = a2.grad_fn.apply(torch.ones(4), torch.ones(4) * 2)
        self.assertIsInstance(received_grads[0], list)
        self.assertEqual(len(received_grads[0]), 2)
        # return 1*2 + 2*3 = 8
        self.assertEqual(result, torch.full_like(x2, 8.0))

    @skipIfTorchDynamo("boxed_grads_call is incompatible with compiled autograd")
    def test_custom_function_boxed_grads_single_list_arg(self):
        # [FUNCTION DOC] Check that a plain list passed to the grad_fn apply of a boxed-gradient
        # Function becomes a single element of the boxed list. Skipped under dynamo.
        """A plain list passed via grad_fn.apply() gets boxed into
        a list wrapping it — the user's list becomes an element."""

        received_grads = []

        class SingleOut(torch.autograd.Function):
            # [CLASS DOC] HELPER: Boxed-gradient Function with one output that records the
            # received gradient list.
            boxed_grads_call = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input doubled.
                return x * 2

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Record the list and return its first element.
                received_grads.append(grads)
                return grads[0]

        x = torch.randn(4, requires_grad=True)
        out = SingleOut.apply(x)

        # grad_fn.apply() with a list — apply() boxes it, so backward
        # receives [user_list] (the user's list is an element, not the grads)
        user_list = [torch.ones(4)]
        out.grad_fn.apply(user_list)
        self.assertIsInstance(received_grads[0], list)
        self.assertEqual(len(received_grads[0]), 1)
        self.assertIs(received_grads[0][0], user_list)

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/176112",
    )
    @skipIfTorchDynamo("dynamo accesses saved_tensors multiple times")
    def test_clear_saved_tensors_on_access(self):
        # [FUNCTION DOC] Check clear_saved_tensors_on_access: the unpacked saved tensor is
        # released from the node, so a local variable is the only reference (weak reference dies
        # after delete). Skipped on ASAN, Linux and slow runs and under dynamo.
        class MyFn(Function):
            # [CLASS DOC] HELPER: Function that clears saved tensors on first access and checks
            # the reference count in backward.
            clear_saved_tensors_on_access = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return a clone.
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Unpack the saved tensor, delete it and assert the weak
                # reference is dead; return the gradient.
                (x,) = ctx.saved_tensors
                ref = weakref.ref(x)
                del x
                # Local variable should be the only remaining reference
                self.assertIsNone(ref())
                return grad_output

        x = torch.randn(3, requires_grad=True)
        y = MyFn.apply(x.clone())
        y.sum().backward()

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/176142",
    )
    @skipIfTorchDynamo("test tests an error that dynamo does not reproduce")
    def test_clear_saved_tensors_on_access_double_access_error(self):
        # [FUNCTION DOC] Check that accessing saved tensors twice in a Function with clearing on
        # access raises RuntimeError saying they can only be accessed once. Skipped on ASAN, Linux
        # and slow runs and under dynamo.
        class MyFn(Function):
            # [CLASS DOC] HELPER: Function that clears saved tensors on access.
            clear_saved_tensors_on_access = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return a clone.
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Access the saved tensors twice and return the gradient.
                _ = ctx.saved_tensors  # first access
                _ = ctx.saved_tensors  # second access - should raise
                return grad_output

        x = torch.randn(3, requires_grad=True)
        y = MyFn.apply(x)
        with self.assertRaisesRegex(RuntimeError, "can only be accessed once"):
            y.sum().backward()

    def test_autograd_node_isinstance(self):
        # [FUNCTION DOC] Check the virtual Node base class: isinstance and issubclass work for
        # generated nodes, manually registered nodes, custom Function contexts and backward
        # contexts, but the base class is not in the real method resolution order, and non-nodes
        # are rejected.
        # Node is a "virtual" base class of codegen'd nodes. This means that
        # isinstance and issubclass are overridden, but mro is unchanged
        Node = torch.autograd.graph.Node

        a = torch.rand(3, 3, requires_grad=True)
        b = a.exp()

        # Some nodes have codegened registrations to the torch._C._function module
        self.assertIsInstance(b.grad_fn, Node)
        self.assertTrue(issubclass(type(b.grad_fn), Node))
        self.assertTrue(Node not in type(b.grad_fn).mro())

        # Other nodes have manual registrations to the torch._C._function module
        self.assertNotIsInstance(torch._C._functions.AccumulateGrad, Node)
        self.assertTrue(issubclass(torch._C._functions.AccumulateGrad, Node))
        self.assertIsInstance(b.grad_fn.next_functions[0][0], Node)
        self.assertTrue(issubclass(torch._C._functions.DelayedError, Node))

        # Special cases
        self.assertNotIsInstance(None, Node)
        self.assertNotIsInstance(1, Node)
        self.assertNotIsInstance(Node, Node)
        self.assertTrue(issubclass(Node, Node))

        # Custom function case
        self.assertTrue(issubclass(torch.autograd.function.BackwardCFunction, Node))

        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function asserting ctx is a Node in forward
            # and backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Assert ctx is a Node and return the input.
                self.assertIsInstance(ctx, Node)
                return x

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Assert ctx is a Node and return the gradient.
                self.assertIsInstance(ctx, Node)
                return x

        out = Func.apply(a)
        self.assertIsInstance(out.grad_fn, Node)
        self.assertTrue(issubclass(type(out.grad_fn), Node))
        self.assertTrue(Node not in type(out.grad_fn).mro())
        out.sum().backward()

    def test_autograd_views_codegen(self):
        # [FUNCTION DOC] Check view tracking and in-place rules of generated code for view_as and
        # unbind in various grad modes and requires-grad settings: which outputs are
        # differentiable views and which in-place ops raise and with what message.
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks the behavior of two codegen functions (view_as and unbind)
        # with respect to view tracking and inplace operation on the output.

        def run_test(grad_mode, requires_grad, is_view, should_raise_tuple):
            # [FUNCTION DOC] HELPER: Create view_as and unbind outputs in the given grad mode and
            # assert view status and the expected error (or none) for in-place adds.
            def maybe_check_raise(fn, should_raise):
                # [FUNCTION DOC] HELPER: Assert the expected error type is None or a string, then
                # run the function expecting that error or none.
                self.assertTrue(should_raise is None or isinstance(should_raise, str))
                if should_raise is not None:
                    with self.assertRaisesRegex(RuntimeError, should_raise):
                        fn()
                else:
                    fn()

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.view_as(inp)
            # Are they differentiable views?
            self.assertTrue(out._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out.add_(1), should_raise_tuple[0])

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.unbind()
            # Are they differentiable views?
            self.assertTrue(out[0]._is_view() == is_view)
            self.assertTrue(out[1]._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out[0].add_(1), should_raise_tuple[1])
            maybe_check_raise(lambda: out[1].add_(1), should_raise_tuple[2])

        # should_raise contains None if it should not raise
        # should_raise contains a string of the error if it should raise
        # The 3 elements are for view_as, first output of unbind and second output of unbind
        run_test(
            grad_mode=True,
            requires_grad=False,
            is_view=True,
            should_raise_tuple=(None, None, None),
        )
        inp_change_err = "Output {} of Unbind is a view and is being modified inplace."
        run_test(
            grad_mode=True,
            requires_grad=True,
            is_view=True,
            should_raise_tuple=(
                None,
                inp_change_err.format("0"),
                inp_change_err.format("1"),
            ),
        )
        leaf_grad_err = (
            "A view was created in no_grad mode and is being modified inplace"
        )
        run_test(
            grad_mode=False,
            requires_grad=True,
            is_view=True,
            should_raise_tuple=(leaf_grad_err, leaf_grad_err, leaf_grad_err),
        )
        run_test(
            grad_mode=False,
            requires_grad=False,
            is_view=True,
            should_raise_tuple=(None, None, None),
        )

    def test_inplace_not_requires_grad(self):
        # [FUNCTION DOC] Check in-place ops with a tensor requiring grad on views that are invalid
        # because they were created inside a custom Function or by a multi-view function raise
        # RuntimeError, while valid view writes still work.
        class MyFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning a view of its input.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Return a view of the input.
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return grad

        # Original Tensor does not require grad
        a = torch.rand(1, 2)

        # Tensor being written does require grad
        b = torch.rand(1, requires_grad=True)

        # Take an invalid view on 'a' that should raise an error (warns during deprecation)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            view_a += b

        # Extra test for copy_ that is a manual implementation and could be easily
        # forgotten when the codegen is updated (warns during deprecation)
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            view_a.copy_(b)

        # Functions that should throw must properly throw
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = a.unbind()[0]
        with self.assertRaisesRegex(
            RuntimeError,
            "This view is the output of a function that returns multiple views.",
        ):
            view_a.copy_(b)

        # Sanity check that views that should work still work
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        a.select(1, 0).copy_(b)

    def _do_test_autograd_simple_views_python(self, dtype):
        # [FUNCTION DOC] HELPER: Check Function behavior when outputs are inputs, views of inputs
        # or views of temporaries, over combinations of function kind, in-place use, view creation
        # and pure-view flag, for the given dtype: expected errors, gradcheck success, and how
        # often the custom backward runs.
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks the autograd.Function behavior when we return one or multiple outputs
        # while one of these is an input, a view of an input or of a temporary tensor.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]
        # This indicator is used to check if the argument `ga` contains non-zero values
        ga_nz = [False]

        class IdOneOutput(Function):
            # [CLASS DOC] HELPER: Function returning its input or a view of it, with a pure view
            # flag, counting backward calls.
            @staticmethod
            def forward(ctx, a, make_view, pure_view):
                # [FUNCTION DOC] HELPER: Store the pure view flag and return a view or a clone of
                # the input.
                ctx._is_pure_view = pure_view
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a

            @staticmethod
            def backward(ctx, ga):
                # [FUNCTION DOC] HELPER: Count the call and return the gradient with None
                # for flags.
                bw_called[0] += 1
                return ga, None, None

        class IdTwoOutput(Function):
            # [CLASS DOC] HELPER: Function with two outputs (the input or a view, and a sum
            # with a second input) that counts backward calls and notes whether the first
            # gradient is nonzero.
            @staticmethod
            def forward(ctx, a, b, make_view, pure_view):
                # [FUNCTION DOC] HELPER: Store the pure view flag and return a view or clone of
                # the first input plus the sum.
                ctx._is_pure_view = pure_view
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                # [FUNCTION DOC] HELPER: Count the call, record whether the first gradient is all
                # zero, and return combined gradients.
                bw_called[0] += 1
                if ga.eq(0).all():
                    ga_nz[0] = False
                else:
                    ga_nz[0] = True
                return ga + gab, gab, None, None

        class ViewOfTemp(Function):
            # [CLASS DOC] HELPER: Function returning a view of a temporary clone, saving
            # the input.
            @staticmethod
            def forward(ctx, a, make_view, pure_view):
                # [FUNCTION DOC] HELPER: Store the flag, save the input and return a selected
                # element of a clone of a view or clone of the input.
                ctx._is_pure_view = pure_view
                ctx.save_for_backward(a)
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                b = a.clone()
                return b.select(0, 0)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Count the call and return a zero tensor with the gradient
                # copied into the first element.
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, 0).copy_(grad)
                return res, None, None

        fn_id_to_inplace_on_view_err_msg = {
            "one_output": (
                "Output 0 of IdOneOutput is a view and is being "
                "modified inplace. This view was created inside a custom Function"
            ),
            "two_output": (
                "Output 0 of IdTwoOutput is a view and is being modified inplace."
                " This view is the output of a function that returns multiple views.",
                "Pure view custom Function can only have one input Tensor and one output Tensor."
                " Open an issue if you need to support more.",
            ),
            "view_of_temp": (
                "Output 0 of ViewOfTemp is a view and is being "
                "modified inplace. This view was created inside a custom Function",
                "a view of a leaf Variable that requires grad is being used in an in-place operation",
            ),
        }

        for fn_id in ["one_output", "two_output", "view_of_temp"]:
            for inplace in [True, False]:
                for make_view in [True, False]:
                    for pure_view in [True, False]:
                        # Used for special casing the tests below
                        output_is_a_view = make_view or fn_id == "view_of_temp"

                        def fn(a, b):
                            # [FUNCTION DOC] HELPER: Apply the selected Function with the loop
                            # settings, optionally add in place or out of place, and sum.
                            # never modify a, b inplace for gracheck
                            a = a.clone()
                            b = b.clone()
                            if fn_id == "two_output":
                                tmp1, tmp2 = IdTwoOutput.apply(
                                    a, b, make_view, pure_view
                                )
                                if inplace:
                                    tmp1 += 3
                                    tmp2 += 3
                                else:
                                    tmp1 = tmp1 + 3
                                    tmp2 = tmp2 + 3
                                tmp = tmp1 * tmp2
                            else:
                                if fn_id == "one_output":
                                    tmp = IdOneOutput.apply(a, make_view, pure_view)
                                else:
                                    tmp = ViewOfTemp.apply(a + b, make_view, pure_view)
                                if inplace:
                                    tmp += 3
                                else:
                                    tmp = tmp + 3

                            return tmp.sum()

                        a = torch.ones(2, dtype=dtype, requires_grad=True)
                        b = torch.ones(2, dtype=dtype, requires_grad=True)

                        err_msg = fn_id_to_inplace_on_view_err_msg[fn_id][
                            int(pure_view)
                        ]

                        will_raise_error = (
                            (pure_view and fn_id == "two_output")
                            or (pure_view and fn_id == "view_of_temp" and inplace)
                            or (not pure_view and inplace and output_is_a_view)
                        )

                        if will_raise_error:
                            with self.assertRaisesRegex(RuntimeError, err_msg):
                                gradcheck(fn, (a, b), check_batched_grad=False)
                        else:
                            gradcheck(fn, (a, b), check_batched_grad=False)

                        # Was the custom backward called properly
                        bw_called[0] = 0
                        ga_nz[0] = True  # For the case where the backward is called

                        expected_called = 1
                        expected_ga_nz = True

                        if will_raise_error:
                            expected_called = 0
                            with self.assertRaisesRegex(RuntimeError, err_msg):
                                fn(a, b)
                        else:
                            fn(a, b).abs().backward()

                        if (
                            fn_id == "one_output"
                            and inplace
                            and output_is_a_view
                            and pure_view
                        ):
                            # We expect the op to have been replayed and we leveraged the pure view
                            # to re-create the graph, so the original backward was not called
                            expected_called = 0

                        self.assertTrue(bw_called[0] == expected_called)
                        self.assertTrue(ga_nz[0] == expected_ga_nz)

    def test_autograd_simple_views_python(self):
        # [FUNCTION DOC] Run the simple views test for double and complex double.
        self._do_test_autograd_simple_views_python(torch.double)
        self._do_test_autograd_simple_views_python(torch.cdouble)

    def test_autograd_inplace_views_creation_meta(self):
        # [FUNCTION DOC] Check creation meta of views for in-place view ops (as_strided_,
        # transpose_, t_, squeeze_, unsqueeze_, swapdims_, swapaxes_) across grad modes at view
        # creation and at op time, requires-grad, and view kinds (normal, multi-view, custom),
        # with the expected errors.
        # Tests creation_meta properly handled for inplace views

        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning a view of its input.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a view of the input.
                return x.view_as(x)

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return x

        view_custom = Func.apply

        def run_test(
            fn, fn_type, grad_mode_view, grad_mode_iview, requires_grad, error1, error2
        ):
            # [FUNCTION DOC] HELPER: Create a view of the chosen kind in one grad mode, apply the
            # in-place view op in another, then add one in place, asserting expected error
            # messages (or none) at each step.
            # This test checks the behavior of inplace-view functions when
            # the views are created in grad mode or not
            base = torch.rand(2, 3, requires_grad=requires_grad).clone()
            # 1. Create a view with `grad_mode=grad_mode_view`
            with torch.set_grad_enabled(grad_mode_view):
                if fn_type == "multi_view":
                    inp = base.unbind()[0]
                elif fn_type == "custom":
                    inp = view_custom(base)
                else:
                    inp = base.view_as(base)

            # 2. Perform inplace view with `grad_mode=grad_mode_iview`
            with torch.set_grad_enabled(grad_mode_iview):
                if error1 is not None:
                    with self.assertRaisesRegex(RuntimeError, error1):
                        fn(inp)
                    return
                else:
                    # If error is None, check that runs without error
                    fn(inp)
            # 3. Do inplace on the (new) view
            if error2 is not None:
                with self.assertRaisesRegex(RuntimeError, error2):
                    inp.add_(1)
            else:
                # If error is None, check that runs without error
                inp.add_(1)

        no_grad_err = "A view was created in no_grad mode"
        multi_view_err = "function that returns multiple views"
        custom_err = "view was created inside a custom Function"

        def run_tests(fn):
            # [FUNCTION DOC] HELPER: Enumerate all kinds, grad modes and requires-grad settings,
            # work out the expected errors and run each case.
            for fn_type in ("normal", "multi_view", "custom"):
                for grad_mode_view in (True, False):
                    for grad_mode_iview in (True, False):
                        for requires_grad in (True, False):
                            error1 = None  # expected error when we do inplace_view on original view
                            error2 = None  # expected error when we do inplace on the resulting view

                            if requires_grad:
                                if not grad_mode_view and grad_mode_iview:
                                    error1 = no_grad_err
                                if not grad_mode_view and not grad_mode_iview:
                                    error2 = no_grad_err

                                if fn_type == "multi_view":
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = multi_view_err
                                    if grad_mode_view and not grad_mode_iview:
                                        error2 = multi_view_err

                                if fn_type == "custom":
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = custom_err
                                    if grad_mode_view and not grad_mode_iview:
                                        error2 = custom_err

                            run_test(
                                fn,
                                fn_type,
                                grad_mode_view,
                                grad_mode_iview,
                                requires_grad,
                                error1,
                                error2,
                            )

        # This list was created by logging gen_inplace_or_view_type.py
        #   detach_ is excluded for this test because it cannot be applied to
        #   views and thus does not return a view
        run_tests(lambda v: v.as_strided_((1, 0), (2, 2)))
        run_tests(lambda v: v.transpose_(0, 0))
        run_tests(lambda v: v.t_())
        run_tests(lambda v: v.squeeze_(0))
        run_tests(lambda v: v.unsqueeze_(0))
        run_tests(lambda v: v.swapdims_(0, 0))
        run_tests(lambda v: v.swapaxes_(0, 0))

    def test_autograd_print_tensor(self):
        # [FUNCTION DOC] Check the string form of tensors: requires_grad marker, grad_fn name for
        # a clone, an invalid grad_fn for a view modified in place under no_grad, and a custom
        # Function backward name.
        a = torch.ones(1, requires_grad=True)
        a_clone = a.clone()
        self.assertEqual(repr(a), "tensor([1.], requires_grad=True)")
        self.assertEqual(repr(a_clone), "tensor([1.], grad_fn=<CloneBackward0>)")

        with torch.no_grad():
            b = a[:]
            b *= 2

        # Special handling for printing view created in no-grad and modified
        # in-placed in no-grad.
        self.assertEqual(repr(b), "tensor([2.], grad_fn=<Invalid>)")

        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function used to check the printed grad_fn name.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return x

        c = Func.apply(a)
        self.assertEqual(repr(c), "tensor([2.], grad_fn=<FuncBackward>)")

    def test_autograd_inplace_view_of_view(self):
        # [FUNCTION DOC] Check in-place ops on a view of a view created under no_grad or inference
        # mode raise RuntimeError with the matching message.
        x = torch.zeros(2)
        with torch.no_grad():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the no_grad block"
        ):
            z /= 2

        x = torch.zeros(2)
        with torch.inference_mode():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the inference_mode"
        ):
            z /= 2

    # TODO This is not the correct behavior -
    # See https://github.com/pytorch/pytorch/issues/49825#issuecomment-794466627
    def test_autograd_inplace_views_cross_dtype(self):
        # [FUNCTION DOC] Check gradients of in-place transposes on a real view of a complex
        # tensor: the in-place transpose result is the transpose of the out-of-place result.
        # KNOWN ISSUE: the test comments say this is a bug (see the linked issue); the assertion
        # locks in the transposed gradient and must lose the transpose once the behavior is fixed.
        # This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b = b.transpose(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        non_inplace_grad = a_orig.grad

        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b.transpose_(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        inplace_grad = a_orig.grad

        # TODO: this is a bug!
        # once this is fixed, it should have the transpose removed:
        # self.assertEqual(non_inplace_grad, inplace_grad)
        self.assertEqual(non_inplace_grad.T, inplace_grad)

    def test_autograd_multiple_views_python(self):
        # [FUNCTION DOC] Check a Function returning a selected slice of its input counts backward
        # once, and in-place modifying its output raises RuntimeError because it is a view created
        # in a custom Function.
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks that multiples views in the forward are properly traced and how they
        # behave with respect to inplace operations.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]

        class ComplexView(Function):
            # [CLASS DOC] HELPER: Function returning a select of its input, counting
            # backward calls.
            @staticmethod
            def forward(ctx, a, idx):
                # [FUNCTION DOC] HELPER: Compute a narrow then a select (the second result is
                # returned), save the input and index.
                res = a.narrow(0, idx, 1)
                res = a.select(0, idx)
                ctx.save_for_backward(a)
                ctx.idx = idx
                return res

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Count the call, scatter the gradient into a zero tensor
                # at the saved index and return it with None.
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, ctx.idx).copy_(grad)
                return res, None

        a = torch.ones(2, requires_grad=True)
        idx = 1

        bw_called[0] = 0
        out = ComplexView.apply(a.clone(), idx)
        out.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        out = ComplexView.apply(a.clone(), idx)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of ComplexView is a view and is being modified inplace",
        ):
            out += 1

    def test_autograd_python_custom_function_inplace(self):
        # [FUNCTION DOC] Check custom Functions that modify inputs in place: single output (with
        # extra in-place and view inputs), warning when marking a non-input dirty, multiple
        # outputs, error for a view input, and error when a dirty tensor is not returned.
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks custom autograd.Function that perform inplace operations

        bw_called = [0]

        # I) Single output
        class MyAdder(Function):
            # [CLASS DOC] HELPER: Function adding in place into its first input, marking it dirty
            # and returning it.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Add the second input in place into the first, mark it
                # dirty and return it.
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Count the call and return the gradient for both inputs.
                bw_called[0] += 1
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        # No extra inplace
        c = MyAdder.apply(a.clone(), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c = MyAdder.apply(a.clone(), b)
        c += 2
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        bw_called[0] = 0
        c = MyAdder.apply(a.clone().view_as(a), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # Should not give non-inputs to mark_dirty
        class MyAdderBad(Function):
            # [CLASS DOC] HELPER: Function that marks a non-input tensor dirty, which should warn.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Create a new tensor from the input, add in place, mark it
                # dirty and return it.
                c = 3 * a
                c.add_(b)
                ctx.mark_dirty(c)
                return c

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Count the call and return tripled gradients for
                # both inputs.
                bw_called[0] += 1
                grad = 3 * grad
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            MyAdderBad.apply(a.clone(), b)
        self.assertEqual(len(w), 1)

        # II) Multiple outputs
        class MyBadAdder(Function):
            # [CLASS DOC] HELPER: Function with two outputs, one an in-place modified input.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Add in place, mark dirty and return the input and a sum.
                a.add_(b)
                ctx.mark_dirty(a)
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                # [FUNCTION DOC] HELPER: Count the call and return the sum of gradients for
                # both inputs.
                bw_called[0] += 1
                return ga + gab, ga + gab

        # No extra inplace
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        c += 2
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        inplace_on_view_err = (
            "your Function modifies inplace an input that is a view of another Tensor"
        )
        with self.assertRaisesRegex(RuntimeError, inplace_on_view_err):
            c, d = MyBadAdder.apply(a.clone().view_as(a), b)

        # III) Inplace + other op
        class MyOutPlaceAdder(Function):
            # [CLASS DOC] HELPER: Function that modifies an input in place but returns clones, so
            # the dirty input is not returned.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Add in place, mark dirty and return a clone and a sum.
                a.add_(b)
                ctx.mark_dirty(a)
                return a.clone(), a + b

            @staticmethod
            def backward(ctx, ga, gab):
                # [FUNCTION DOC] HELPER: Count the call and return combined gradients.
                bw_called[0] += 1
                return ga + gab, ga + 2 * gab

        # We don't reuse the input
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Apply the Function to a view of a clone and return the sum of
            # the product of outputs.
            orig_a = a.clone().view_as(a)
            c, d = MyOutPlaceAdder.apply(orig_a, b)
            return (c * d).sum()

        bad_mark_dirty_err = "Some elements marked as dirty during the forward method were not returned as output."
        with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
            fn(a, b)

    def test_custom_function_mark_dirty_not_differentiable(self):
        # [FUNCTION DOC] Check mark_dirty on a Function in forward mode: the same tensor is
        # returned as is, and the jvp rule must modify its tangent in place or an error is raised,
        # with and without requires_grad.
        def get_custom_fn(jvp_err):
            # [FUNCTION DOC] HELPER: Build an in-place multiply Function whose jvp either returns
            # the tangent unchanged or modifies it in place.
            class InplaceMul(torch.autograd.Function):
                # [CLASS DOC] HELPER: Function doubling its input in place and marking it dirty.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Double the input in place, mark it dirty and
                    # return it.
                    result = x.mul_(2)
                    ctx.mark_dirty(result)
                    return result

                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Backward that does nothing.
                    pass

                @staticmethod
                def jvp(ctx, x_t):
                    # [FUNCTION DOC] HELPER: Return the tangent unchanged, or doubled in place,
                    # depending on the flag.
                    if jvp_err:
                        return x_t
                    else:
                        return x_t.mul_(2)

            return InplaceMul

        for requires_grad, jvp_err in product([True, False], repeat=2):
            InplaceMul = get_custom_fn(jvp_err)
            # Make sure that tensor is always returned as-is if marked dirty
            z = torch.tensor(1.0, requires_grad=requires_grad)
            x = z.clone()
            y = InplaceMul.apply(x)
            self.assertTrue(x is y)
            self.assertEqual(x, z * 2)

            # jvp must properly modify the input grad if mark_dirty is set
            with fwAD.dual_level():
                x_tangent = torch.ones_like(x)
                x_dual = fwAD.make_dual(x, x_tangent)

                if jvp_err:
                    bad_mark_dirty_err = (
                        "jvp function must modify the corresponding gradient inplace"
                    )
                    with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
                        InplaceMul.apply(x_dual)
                else:
                    out_dual = InplaceMul.apply(x_dual)
                    _, out_tangent = fwAD.unpack_dual(out_dual)
                    self.assertTrue(out_dual is x_dual)
                    self.assertTrue(out_tangent is x_tangent)

    def test_custom_function_mark_output_view_of_intermediate(self):
        # [FUNCTION DOC] Check mark_dirty with an output that is a view of an intermediate rather
        # than an input raises RuntimeError saying a tensor that was not an input was received.
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function marking a view of a clone dirty and returning it.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Create a view of a clone, mark it dirty and return it.
                out = inp.clone().view_as(inp)
                ctx.mark_dirty(out)
                return out

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Backward that does nothing.
                pass

        a = torch.tensor([1.0], requires_grad=True)
        a_clone = a.clone()

        with self.assertRaisesRegex(
            RuntimeError, "received a tensor that was not an input."
        ):
            Func.apply(a_clone)

    def test_custom_function_inplace_on_non_default_view(self):
        # [FUNCTION DOC] Check an in-place Function applied to one output of a multi-view op
        # raises RuntimeError.
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function modifying its input in place and marking it dirty.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Add one in place, mark dirty and return the input.
                inp.add_(1)
                ctx.mark_dirty(inp)
                return inp

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Backward that does nothing.
                pass

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        a_clone = a.clone()
        b, c = a.split_with_sizes([1, 1], dim=0)

        with self.assertRaisesRegex(
            RuntimeError, "output of a function that returns multiple view"
        ):
            Func.apply(b)

    def test_custom_function_inplace_on_view_of_leaf(self):
        # [FUNCTION DOC] Check an in-place Function applied to a view of a leaf requiring grad
        # raises RuntimeError.
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function modifying its input in place and marking it dirty.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Add one in place, mark dirty and return the input.
                inp.add_(1)
                ctx.mark_dirty(inp)
                return inp

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Backward that does nothing.
                pass

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = a.view_as(a)

        with self.assertRaisesRegex(
            RuntimeError, "a view of a leaf Variable that requires grad"
        ):
            Func.apply(b)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO and sys.version_info >= (3, 14), "Fails in python 3.14.2"
    )
    def test_custom_function_saving_mutated_view_no_leak(self):
        # [FUNCTION DOC] Check a Function that marks dirty and saves a mutated view does not leak:
        # the view is freed after the scope. Skipped under dynamo on Python 3.14 and later.
        class Test(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function marking its input dirty and saving it.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Mark the input dirty, save it and return it.
                ctx.mark_dirty(x)
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward that does nothing.
                pass

        def scope():
            # [FUNCTION DOC] HELPER: Apply the Function to a view of a clone and return a weak
            # reference to the view.
            x = torch.tensor(1.0, requires_grad=True).clone()
            x = x.view_as(x)
            y = Test.apply(x)
            return weakref.ref(x)

        ref = scope()
        self.assertIsNone(ref())

    def test_custom_function_return_view_in_nograd(self):
        # [FUNCTION DOC] Check a custom Function returning a view in no_grad mode behaves like the
        # equivalent op: same requires_grad, and in-place modification raises RuntimeError about a
        # view created in no_grad.
        class Alias(Function):
            # [CLASS DOC] HELPER: Function returning a full slice view of its input.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a full slice of the input.
                return x[:]

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return gx

        inp = torch.rand(2, requires_grad=True)

        with torch.no_grad():
            output = Alias.apply(inp)

        with torch.no_grad():
            expected_output = inp[:]

        # Calling the custom function should operate as if we called an equivalent op
        self.assertEqual(output.requires_grad, expected_output.requires_grad)

        # Check that in-place modification on view throws
        leaf_grad_err = (
            "A view was created in no_grad mode and is being modified inplace"
        )
        with self.assertRaisesRegex(RuntimeError, leaf_grad_err):
            output.zero_()

    def test_custom_function_preserve_torch_function_when_return_as_is(self):
        # [FUNCTION DOC] Check a Function returning its input as is preserves a tensor subclass
        # with its own torch function override.
        class Custom(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass holding its data, whose torch function unwraps
            # instances, runs the op and rewraps tensor results.
            def __init__(self, data):
                # [FUNCTION DOC] HELPER: Store the wrapped data.
                super().__init__()
                self._data = data

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Unwrap subclass arguments, call the function and rewrap
                # tensor outputs in the subclass.
                kwargs = {} if kwargs is None else kwargs
                args = tuple(a._data if isinstance(a, cls) else a for a in args)
                out = func(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    out = cls(out)
                return out

        class Fn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Return the input.
                return input

            @staticmethod
            def backward(ctx):
                # [FUNCTION DOC] HELPER: Backward that does nothing.
                pass

        x = Custom(torch.randn(2, 3))
        y = Fn.apply(x)
        self.assertTrue(isinstance(y, Custom))

    def test_grad_mode_restored_reentrant(self):
        # [FUNCTION DOC] Check that grad mode is restored correctly after a nested grad call
        # inside a custom backward, when the outer mode is disabled or enabled (create_graph).
        class MyFunction(Function):
            # [CLASS DOC] HELPER: Function whose backward enables grad, computes a nested gradient
            # and asserts grad mode is restored.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Return a clone.
                return inp.clone()

            @staticmethod
            def backward(ctx, go):
                # [FUNCTION DOC] HELPER: Compute a nested grad of a cubic under enable_grad,
                # asserting the mode before and after, and return it.
                original = torch._C.is_grad_enabled()
                with torch.enable_grad():
                    self.assertTrue(torch._C.is_grad_enabled())
                    foo = torch.rand(go.size(), requires_grad=True)
                    (grad,) = torch.autograd.grad(foo**3, foo, grad_outputs=go)
                    self.assertTrue(torch._C.is_grad_enabled())
                self.assertTrue(torch._C.is_grad_enabled() == original)
                return grad

        inp = torch.rand(3, requires_grad=True)

        # Case where original==False
        MyFunction.apply(inp).sum().backward()
        # Case where original==True
        MyFunction.apply(inp).sum().backward(create_graph=True)

    def test_power_function(self):
        # [FUNCTION DOC] Check gradients of power with a zero base: negative infinity at exponent
        # minus one for tensor and scalar bases, zero elsewhere.
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(a**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

        s = 0
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(s**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

    def test_custom_function_error(self):
        # [FUNCTION DOC] Check errors for malformed custom Functions: missing forward, missing
        # backward, both backward and vjp, and jvp not implemented in forward mode.
        class BadFw(Function):
            # [CLASS DOC] HELPER: Function with no forward.
            @staticmethod
            def backward(ctx, foo):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return foo

        class BadBw(Function):
            # [CLASS DOC] HELPER: Function with no backward.
            @staticmethod
            def forward(ctx, foo):
                # [FUNCTION DOC] HELPER: Return a clone.
                return foo.clone()

        class BadBw2(Function):
            # [CLASS DOC] HELPER: Function defining both backward and vjp.
            @staticmethod
            def forward(ctx, foo):
                # [FUNCTION DOC] HELPER: Return a clone.
                return foo.clone()

            @staticmethod
            def backward(ctx, foo):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return foo

            @staticmethod
            def vjp(ctx, foo):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return foo

        class BadJvp(Function):
            # [CLASS DOC] HELPER: Function with no jvp.
            @staticmethod
            def forward(ctx, foo):
                # [FUNCTION DOC] HELPER: Return a clone.
                return foo.clone()

        inp = torch.rand(1, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "must implement the forward"):
            BadFw.apply(inp)

        with self.assertRaisesRegex(RuntimeError, "must implement either the backward"):
            BadBw.apply(inp).sum().backward()

        with self.assertRaisesRegex(
            RuntimeError, "Implementing both 'backward' and 'vjp'"
        ):
            BadBw2.apply(inp).sum().backward()

        with self.assertRaisesRegex(RuntimeError, "must implement the jvp function"):
            with fwAD.dual_level():
                d = fwAD.make_dual(inp, torch.rand_like(inp))
                res = BadJvp.apply(d)

    def test_custom_function_forward_mode_view_checks(self):
        # [FUNCTION DOC] Check forward-mode view checks for a custom Function: the jvp rule must
        # return a view of the tangent of the same base, with errors for not a view, not a view of
        # the input, and not the same base.
        flag_to_error = {
            "ok": None,
            "not_a_view": "jvp is not returning a view",
            "not_a_view_of_inp": "jvp is not returning a view of the given",
            "not_a_view_of_inp_base": "jvp is not returning a view of the same base",
        }

        class ViewFn(Function):
            # [CLASS DOC] HELPER: Function returning a narrow view whose jvp breaks the view
            # relationship depending on a flag.
            @staticmethod
            def forward(ctx, foo, flag):
                # [FUNCTION DOC] HELPER: Store flag and size and return a narrow view of
                # the input.
                ctx.flag = flag
                ctx.size = foo.size()
                return foo.narrow(0, 0, 2)

            @staticmethod
            def vjp(ctx, gO):
                # [FUNCTION DOC] HELPER: Scatter the gradient into a zero tensor of the
                # input size.
                gI = gO.new_zeros(ctx.size)
                gI.narrow(0, 0, 2).copy_(gO)
                return gI, None

            @staticmethod
            def jvp(ctx, gI, _):
                # [FUNCTION DOC] HELPER: Return a narrow of the tangent, cloned or rewrapped as a
                # view of itself depending on the flag.
                res = gI.narrow(0, 0, 2)
                if ctx.flag != "ok":
                    # Break the view in the gradients!
                    res = res.clone()
                if ctx.flag in ["not_a_view_of_inp", "not_a_view_of_inp_base"]:
                    # Result should be a view, just of the wrong thing
                    res = res.view_as(res)
                return res

        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        for flag, msg in flag_to_error.items():

            def test_fn(inp):
                # [FUNCTION DOC] HELPER: Optionally make a view of the input, then apply the
                # Function with the loop flag.
                if flag == "not_a_view_of_inp_base":
                    inp = inp.view_as(inp)
                return ViewFn.apply(inp, flag)

            if msg is None:
                gradcheck(test_fn, inp, check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, msg):
                    gradcheck(test_fn, inp, check_forward_ad=True)

    def test_custom_function_forward_mode_inplace_checks(self):
        # [FUNCTION DOC] Check forward-mode in-place checks: a jvp that modifies its tangent in
        # place is accepted, one that returns a new tensor makes gradcheck raise RuntimeError.
        class InplaceFn(Function):
            # [CLASS DOC] HELPER: In-place doubling Function whose jvp is optionally out of place.
            @staticmethod
            def forward(ctx, foo, flag):
                # [FUNCTION DOC] HELPER: Mark dirty, store the flag, double the input in place and
                # return it.
                ctx.mark_dirty(foo)
                ctx.flag = flag
                foo.mul_(2)
                return foo

            @staticmethod
            def vjp(ctx, gO):
                # [FUNCTION DOC] HELPER: Return twice the gradient and None.
                return 2 * gO, None

            @staticmethod
            def jvp(ctx, gI, _):
                # [FUNCTION DOC] HELPER: Return twice the tangent as a new tensor, or double it in
                # place, depending on the flag.
                if ctx.flag:
                    # Don't do the change inplace
                    return 2 * gI
                else:
                    gI.mul_(2)
                    return gI

        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        def test_fn(inp, flag):
            # [FUNCTION DOC] HELPER: Clone the input and apply the in-place Function.
            inp = inp.clone()
            return InplaceFn.apply(inp, flag)

        gradcheck(test_fn, (inp, False), check_forward_ad=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "inplace custom Function is not modifying the forward mode gradients inplace",
        ):
            gradcheck(test_fn, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_wrong_formula(self):
        # [FUNCTION DOC] Check a wrong jvp formula is detected by gradcheck as a forward-mode
        # Jacobian mismatch RuntimeError.
        class UserFn(Function):
            # [CLASS DOC] HELPER: Function doubling its input with a jvp that triples when told
            # to fail.
            @staticmethod
            def forward(ctx, foo, should_fail):
                # [FUNCTION DOC] HELPER: Store the failure flag and return the input doubled.
                ctx.should_fail = should_fail
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient doubled and None.
                return 2 * gO, None

            @staticmethod
            def jvp(ctx, gI, _):
                # [FUNCTION DOC] HELPER: Return the tangent tripled when failing,
                # otherwise doubled.
                if ctx.should_fail:
                    # Wrong gradient formula
                    return 3 * gI
                else:
                    return 2 * gI

        inp = torch.rand(10, dtype=torch.double, requires_grad=True)
        gradcheck(UserFn.apply, (inp, False), check_forward_ad=True)

        with self.assertRaisesRegex(
            RuntimeError, "Jacobian computed with forward mode mismatch for output 0"
        ):
            gradcheck(UserFn.apply, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_non_tensor_before_tensor_args(self):
        # [FUNCTION DOC] Check jvp receives None for non-tensor arguments placed before tensor
        # arguments, and gradcheck forward AD works with such a Function.
        class MyFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function with non-tensor and tensor arguments interleaved.
            @staticmethod
            def forward(ctx, nt, x, nt2, y):
                # [FUNCTION DOC] HELPER: Return a combination of the two tensor inputs.
                return x * 2 + y * 3

            @staticmethod
            def jvp(ctx, nt, x_t, nt2, y_t):
                # [FUNCTION DOC] HELPER: Assert the non-tensor tangents are None and return the
                # combined tangents.
                self.assertIsNone(nt)
                self.assertIsNone(nt2)
                return x_t * 2 + y_t * 3

        x = torch.tensor(1.0, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        y = torch.tensor(1.0, dtype=torch.double)

        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, t)
            MyFn.apply(1, dual_x, 1, y)

        gradcheck(
            MyFn.apply,
            (1, x.requires_grad_(True), 1, y.requires_grad_(True)),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
        )

    def test_custom_function_forward_mode_forward_is_no_op(self):
        # [FUNCTION DOC] Check what happens when a custom Function's jvp returns a view of the
        # tangent, the tangent itself, or a new tensor for an output that forward returned as is:
        # view is accepted, others raise RuntimeError.
        error_regex = (
            "A custom Function's forward is returning a view \\(or an input as-is\\)"
        )

        return_lambdas = {
            # If we return an input as-is in forward, that is treated
            # as if self.view_as(self) is performed. If jvp returns x.view_as(x),
            # this is OK.
            "view_as": lambda x: x.view_as(x),
            # Expect this to raise an error
            "self": lambda x: x,
            # Expect this to raise the same error
            "mul_by_2": lambda x: x * 2,
        }

        for k, fn in return_lambdas.items():

            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Function returning a sum and its first input as is, whose
                # jvp uses the loop's tangent mapping function.
                @staticmethod
                def forward(ctx, x, y):
                    # [FUNCTION DOC] HELPER: Return the sum and the first input.
                    return x + y, x

                @staticmethod
                def vjp(ctx, gO1, gO2):
                    # [FUNCTION DOC] HELPER: Return the sum of gradients and the first gradient.
                    return gO1 + gO2, gO1

                @staticmethod
                def jvp(ctx, x_t, y_t):
                    # [FUNCTION DOC] HELPER: Return the sum of tangents and the mapped tangent.
                    return x_t + y_t, fn(x_t)

            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            t = torch.tensor(1.0, dtype=torch.double)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)

            c = torch.tensor(1.0, dtype=torch.double)
            t2 = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)

            with fwAD.dual_level():
                a_dual = fwAD.make_dual(a, t)
                c_dual = fwAD.make_dual(c, t2)

                if k == "view_as":
                    _, out2 = MyFn.apply(a_dual, b)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t)

                    _, out2 = MyFn.apply(c_dual, d)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t2)
                else:
                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(a_dual, b)

                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(c_dual, d)

            if k == "view_as":
                gradcheck(MyFn.apply, (a, c), check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, error_regex):
                    gradcheck(MyFn.apply, (a, c), check_forward_ad=True)

    def test_custom_function_save_for_forward(self):
        # [FUNCTION DOC] Check save_for_forward: values saved for forward are available in jvp,
        # those saved only for backward are not, and gradcheck forward AD passes.
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving inputs for backward and forward plus an int,
            # computing a product scaled by the int.
            @staticmethod
            def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
                # [FUNCTION DOC] HELPER: Save the tensors for both modes, store the int and the
                # product and return the scaled product.
                ctx.save_for_backward(x, y)
                ctx.save_for_forward(x, y)
                ctx.z = z
                ctx.prod = x * y
                return z * ctx.prod

            @staticmethod
            def jvp(ctx, x_t, y_t, _):
                # [FUNCTION DOC] HELPER: Use the saved primals to compute the scaled product rule
                # of the tangents.
                x_p, y_p = ctx.saved_tensors
                z = ctx.z
                return z * (y_p * x_t + x_p * y_t)

            @staticmethod
            def vjp(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Return the scaled gradients with respect to each tensor
                # input and None for the int.
                x, y = ctx.saved_tensors
                z = ctx.z
                return z * grad_out * y, z * grad_out * x, None

        a = torch.tensor(1.0, requires_grad=True, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        b = torch.tensor(2.0, requires_grad=True, dtype=torch.double)
        c = 4

        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual, b, c)
            out.backward()

        gradcheck(Func.apply, (a, b, c), check_forward_ad=True)

        # When saved for backward, but not saved for forward
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving its input only for backward.
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Save for backward and return a clone.
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def jvp(ctx, x_t):
                # [FUNCTION DOC] HELPER: Assert no saved tensors are visible in jvp and return
                # the tangent.
                self.assertEqual(len(ctx.saved_tensors), 0)
                return x_t

            @staticmethod
            def vjp(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Assert one saved tensor in vjp and return the gradient.
                (x,) = ctx.saved_tensors
                self.assertEqual(len(ctx.saved_tensors), 1)
                return grad_out

        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual)
            out.backward()

        gradcheck(Func.apply, (a,), check_forward_ad=True)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_custom_function_forward_mode_non_differentiable(self):
        # [FUNCTION DOC] Check forward-mode handling of non-differentiable outputs: tangent None
        # for a marked output, None expected for non-differentiable dtypes, RuntimeError if jvp
        # returns a tangent for a non-differentiable output, and non-tensor outputs. Skipped
        # under dynamo.
        # returns differentiable type, marked non-differentiable
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning a clone and a non-differentiable clone.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Clone, mark the second output non-differentiable and
                # return both.
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                # [FUNCTION DOC] HELPER: Return the first tangent and None.
                return x_tangent, None

        x = torch.tensor(2.0)
        x_tangent = torch.tensor(1.0)
        y = torch.tensor(3.0)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            _, out2_dual = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        y = torch.tensor(3)

        # returns non-differentiable type, NOT marked non-differentiable
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning clones of a float and an integer tensor.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Return clones of both inputs.
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                # [FUNCTION DOC] HELPER: Assert the second tangent is None and return the first
                # tangent and None.
                self.assertIsNone(y_tangent)
                return x_tangent, None

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            _, out2_dual = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        class FuncWrong(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function with a jvp that wrongly returns a tangent for a
            # non-differentiable output.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Clone, mark the second output non-differentiable and
                # return both.
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                # [FUNCTION DOC] HELPER: Return the tangent and a clone of it.
                return x_tangent, x_tangent.clone()

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            with self.assertRaisesRegex(
                RuntimeError, "You should return None at that position instead"
            ):
                FuncWrong.apply(x_dual, y)

        # returns non-tensor
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning a clone, a plain object and a clone.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone, a generic object and another clone.
                return x.clone(), object(), x.clone()

            @staticmethod
            def jvp(ctx, x_tangent):
                # [FUNCTION DOC] HELPER: Return the tangent, None and the tangent.
                return x_tangent, None, x_tangent

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            out_dual, _, out2_dual = Func.apply(x_dual)
            self.assertEqual(fwAD.unpack_dual(out_dual).tangent, x_tangent)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, x_tangent)

    def test_custom_function_local_inplace(self):
        # [FUNCTION DOC] Check that a Function whose forward does an in-place op on a local view
        # still gets its own backward node class, with and without the in-place op.
        class MyFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function that makes a local view of a clone, optionally modifies
            # it in place and returns it.
            @staticmethod
            def forward(ctx, inp, inplace):
                # [FUNCTION DOC] HELPER: Clone the first three elements as a view, optionally add
                # two in place, and return the view.
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient and None.
                return grad, None

        base = torch.rand(10, requires_grad=True)

        foo = MyFn.apply(base, False)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

        foo = MyFn.apply(base, True)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

    def test_integer_outputs(self):
        # [FUNCTION DOC] Check ops with integer outputs (argmax, argmin, argsort, searchsorted,
        # bucketize, count_nonzero) do not require grad, and in the unique family only the first
        # output requires grad.
        inp = torch.rand(4, requires_grad=True)

        out = inp.argmax()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argmin()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argsort()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.rand((), requires_grad=True)

        out = torch.searchsorted(inp, val)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        bins = torch.linspace(0, 1.0, steps=100, requires_grad=True)
        vals = torch.rand(5, 5, requires_grad=True)
        out = torch.bucketize(vals, bins)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.empty(5).requires_grad_()
        out = val.count_nonzero()
        self.assertFalse(out.requires_grad)

        def assert_only_first_requires_grad(res):
            # [FUNCTION DOC] HELPER: Assert the first result requires grad and any remaining
            # results do not.
            if not isinstance(res, tuple):
                res = (res,)
            self.assertTrue(res[0].requires_grad)
            for out in res[1:]:
                if out is not None:
                    self.assertFalse(out.requires_grad)

        for sort in [True, False]:
            for return_inverse in [True, False]:
                for return_counts in [True, False]:
                    res = torch.unique(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                        dim=0,
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(
                        inp, return_inverse=return_inverse, return_counts=return_counts
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(
                        inp,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                        dim=0,
                    )
                    assert_only_first_requires_grad(res)

                    # Here we test the internal functions to make sure all of them are
                    # covered on top of the public API
                    res = torch._unique(inp, sorted=sort, return_inverse=return_inverse)
                    assert_only_first_requires_grad(res)

                    # This looks public but is actually manually deleted from the
                    # torch namespace in torch/functional.py
                    res = torch._VF.unique_dim(
                        inp,
                        dim=0,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

                    # We don't test `unique_dim_consecutive` here.
                    # It looks public but the python binding is actually manually disabled in
                    # tools/autograd/gen_python_functions.py

                    res = torch._unique2(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO and sys.version_info >= (3, 14), "Fails in python 3.14.2"
    )
    def test_custom_function_cycle(self):
        # [FUNCTION DOC] Check a reference cycle through a Function's context metadata is
        # collected: the output is freed after garbage collection with or without backward.
        # Skipped under dynamo on Python 3.14 and later.
        class MyFn(Function):
            # [CLASS DOC] HELPER: Function that stores a metadata dict on ctx, saves its clone and
            # checks values in backward.
            @staticmethod
            def forward(ctx, x, metadata):
                # [FUNCTION DOC] HELPER: Clone the input, store metadata and the clone on ctx and
                # return the clone.
                x = x.clone()
                ctx.meta = metadata
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Check the saved value and the metadata entry equal the
                # expected number and return the gradient times the saved value and None.
                (x,) = ctx.saved_tensors
                self.assertEqual(x, 3.14)
                self.assertEqual(ctx.meta["foo"], 3.14)
                return gO * x, None

        def get_refs(with_backward):
            # [FUNCTION DOC] HELPER: Build the cycle (output stored in the metadata), optionally
            # run backward, and return a weak reference to the output.
            a = torch.tensor(3.14, requires_grad=True)

            metadata = {}
            out = MyFn.apply(a, metadata)

            metadata["foo"] = out

            if with_backward:
                out.sum().backward()
                self.assertEqual(a.grad, a)

            return torch._C._WeakTensorRef(out)

        with disable_gc():
            ref = get_refs(False)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())

        # The backward clears the saved_variables but not the __dict__
        with disable_gc():
            ref = get_refs(True)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())

    def test_create_graph_and_full_backward_hook_cycle(self):
        # [FUNCTION DOC] Check a full backward hook on a module does not create a reference cycle
        # when backward uses create_graph, for several hook counts and inputs, and that exactly
        # one create_graph warning is issued.
        # If BackwardHook saves grad_output, it can create a cycle when we perform backward
        # with create_graph=True
        #
        #   grad_output -> grad_output.grad_fn -> graph -> hook -> grad_output
        #
        class TestCls:
            # [CLASS DOC] HELPER: Empty class to create a weak reference to.
            # Dummy class for the purpose of creating a weakref
            pass

        def get_ref(input_requires_grad, nb_hooks):
            # [FUNCTION DOC] HELPER: Run a hooked module with create_graph backward, store a dummy
            # object in the graph metadata and return a weak reference to it.
            t = torch.randn(10, requires_grad=input_requires_grad)
            a = torch.tensor(1.0, requires_grad=True)

            class Test(nn.Module):
                # [CLASS DOC] HELPER: Module computing the square of the input times a captured
                # parameter squared.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Return the square of the input times a captured
                    # squared scalar.
                    return x**2 * a**2

            mod = Test()

            for _ in range(nb_hooks):
                mod.register_full_backward_hook(lambda a, b, c: None)

            tmp = mod(t)

            # Save dummy object to graph and get a weak ref to it
            test = TestCls()
            ref = weakref.ref(test)
            tmp.grad_fn.metadata["a"] = test

            with set_warn_always_context(True):
                with warnings.catch_warnings(record=True) as w:
                    tmp.exp().sum().backward(create_graph=True)
                    self.assertTrue(w)
                    found = 0
                    for warning in w:
                        if "Using backward() with create_graph=True" in str(
                            warning.message
                        ):
                            found += 1
                    self.assertEqual(found, 1)

            # Remove the backward + create_graph=True cycle
            a.grad = None
            t.grad = None

            return ref

        for nb_hooks in (1, 2, 3):
            for input_requires_grad in (True, False):
                ref_ = get_ref(
                    input_requires_grad=input_requires_grad,
                    nb_hooks=nb_hooks,
                )
                gc.collect()
                self.assertIsNone(ref_())

    @parametrize("use_custom_function", [True, False])
    @parametrize("use_tensor_hook", [True, False])
    def test_hook_closure_cycle(self, use_custom_function, use_tensor_hook):
        # [FUNCTION DOC] Parametrized by custom Function use and tensor versus node hook: a hook
        # closure holding the node forms a cycle that must survive garbage collection while the
        # graph is alive (hook still fires) and be freed when the graph is deleted.
        # This creates a cycle between the hook and grad_fn_b
        # hook -> closure -> grad_fn_b (python) -> grad_fn (cpp) -> hook (cpp)
        # -> dict -> hook
        #
        # This test is testing that the grad_fn_b (python) only traverses the
        # dict if it is the only one holding a reference to the grad_fn_b (cpp)
        # shared_ptr
        #
        # See: https://github.com/pytorch/pytorch/issues/102174
        class Function(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return grad

        class Test:
            # [CLASS DOC] HELPER: Empty marker class for a sentinel object.
            pass

        count = [0]

        def scope():
            # [FUNCTION DOC] HELPER: Build the graph, register a hook whose closure holds the node
            # and a sentinel, and return the output and a weak reference to the sentinel.
            a = torch.tensor(1.0, requires_grad=True)
            if use_custom_function:
                b = Function.apply(a)
            else:
                b = a.clone()
            grad_fn_b = b.grad_fn
            obj = Test()

            def hook(*args):
                # [FUNCTION DOC] HELPER: Hook referencing the node and sentinel and
                # counting calls.
                # Make sure this hook's closure holds onto grad_fn_b
                # This forms a cycle between the hook and grad_fn_b
                # We also hold onto a sentinel object 'obj' to track
                # whether this cycle is still alive. See 'ref' below.
                grad_fn_b
                obj
                count[0] += 1

            if use_tensor_hook:
                b.register_hook(hook)
            else:
                b.grad_fn.register_hook(hook)
            c = b.clone()
            ref = weakref.ref(obj)
            return c, ref

        with disable_gc():
            out, ref = scope()
            out.backward(retain_graph=True)

            gc.collect()

            # Make sure gc does not clear the cycle noted above.
            # e.g. the hook is alive and gets fired even after gc runs
            out.backward(retain_graph=True)
            self.assertEqual(count[0], 2)

            # ref is still alive because the use_count of the cpp grad_fn
            # shared_ptr > 1 since (1) the python grad_fn is alive, and (2) the
            # rest of the graph holds onto the shared_ptr
            self.assertIsNotNone(ref())

            # Then delete the rest of the graph and check that ref is dead
            del out
            gc.collect()
            self.assertIsNone(ref())

    def test_full_backward_hook_double_backward(self):
        # [FUNCTION DOC] Check that a full backward hook on a module runs once for the first
        # backward and is not invoked (nor erroring) on the double backward.
        x = torch.rand(1, requires_grad=True)
        y = torch.rand_like(x)

        func = torch.nn.MSELoss()
        counter = [0]

        def hook(module, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Module backward hook counting calls.
            counter[0] += 1

        func.register_full_backward_hook(hook)

        f = func(x, y)

        (gradx_f,) = torch.autograd.grad(f, x, create_graph=True)
        self.assertEqual(counter[0], 1)
        _ = torch.autograd.grad(gradx_f, x)
        # We should not error, and counter should not be incremented
        self.assertEqual(counter[0], 1)

    def test_input_buffer_accum(self):
        # [FUNCTION DOC] Check accumulating a sparse gradient and a dense pass-through gradient
        # does not modify the provided gradients in place.
        leaf = torch.rand(2, 2, requires_grad=True)

        # An op that returns sparse gradients
        ind = torch.tensor([[0, 0]], dtype=torch.long)
        out2 = leaf.gather(0, ind, sparse_grad=True)

        # An op that returns the gradients as-is
        out1 = leaf.clone()

        grad_out1_original = torch.rand_like(out1)
        grad_out1 = grad_out1_original.clone()
        grad_out2 = torch.rand_like(out2)

        torch.autograd.backward((out1, out2), (grad_out1, grad_out2))

        # Given gradients should not be modified inplace
        self.assertEqual(grad_out1, grad_out1_original)

    def test_no_unnecessary_unwrapping(self):
        # [FUNCTION DOC] Check saved leaf and non-output tensors are returned as the same objects
        # while saved outputs are unpacked into new objects, and access after backward raises
        # RuntimeError; the original leaf is untouched.
        a = torch.randn(5, requires_grad=True)
        a_orig = a.detach().clone()
        b = a * a
        c = a * b
        d = torch.exp(a)

        # a is leaf
        self.assertIs(b.grad_fn._saved_self, a)
        self.assertIs(b.grad_fn._saved_other, a)
        self.assertIs(c.grad_fn._saved_self, a)

        # b is not an output
        self.assertIs(c.grad_fn._saved_other, b)

        # d is an output
        self.assertEqual(d.grad_fn._saved_result, d)
        self.assertIsNot(d.grad_fn._saved_result, d)

        c.sum().backward()

        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            c.grad_fn._saved_self

        # a is left untouched
        self.assertEqual(a, a_orig)

    def test_saved_variable_version_counter(self):
        # [FUNCTION DOC] Check an unpacked saved output shares the version counter of the
        # original, including after an in-place update under no_grad.
        a = torch.rand(2, requires_grad=True)

        b = torch.exp(a)

        b_unpacked = b.grad_fn._saved_result
        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)

        with torch.no_grad():
            b += 1

        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)

    def test_saved_variable_packing_unpacking_saved_original_with_hooks(self):
        # [FUNCTION DOC] Check pack and unpack hooks registered on a raw saved tensor, for leaf
        # and non-leaf, including double backward, wrong return types, wrong signatures, and a
        # pack hook that modifies its input in place (error).
        # Tests that packing/unpacking a SavedVariable works correctly with user-defined hooks
        # The saved_original / did_not_save_original distinction corresponds to the `save_original`
        # attribute of `SavedVariable`.

        def test(get_input, is_leaf):
            # [FUNCTION DOC] HELPER: Run the whole set of hook scenarios with an input factory and
            # a flag stating whether the input is a leaf.
            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x / 2)
            self.assertEqual(a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(2 * a, a.grad)

            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x)
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(3 * a, a.grad)

            # double backward
            a = get_input()
            grad_fn = a.grad_fn
            y = a**3
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                g.sum().backward()
            else:
                g.sum().backward()
                self.assertEqual(6 * a, a.grad)

            a = get_input()
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: 1)
            with self.assertRaisesRegex(
                TypeError, "Output of saved tensor unpack_hook expected to be a Tensor"
            ):
                print(y.grad_fn._saved_self)

            a = get_input()
            y = a * a
            with self.assertRaisesRegex(
                TypeError, "missing 1 required positional argument"
            ):
                y.grad_fn._raw_saved_self.register_hooks(lambda x, b: x, lambda x: x)

            a = get_input()
            y = a * a
            with self.assertRaisesRegex(
                TypeError, "missing 1 required positional argument"
            ):
                y.grad_fn._raw_saved_self.register_hooks(
                    lambda x, b: (x, b), lambda x: x
                )

            def inplace_double(x):
                # [FUNCTION DOC] HELPER: Pack hook that doubles its input in place and returns it.
                x *= 2
                return x

            a = get_input()
            t = a * a

            with self.assertRaisesRegex(
                RuntimeError,
                "A saved tensor pack hook is modifying its input in place.",
            ):
                t.grad_fn._raw_saved_self.register_hooks(
                    inplace_double, lambda x: x / 2
                )

        # leaf
        test(lambda: torch.randn(5, requires_grad=True), True)

        # not leaf, not output
        test(lambda: (1 + torch.randn(5, requires_grad=True)), False)

    def test_saved_variable_saved_original_inplace_detach(self):
        # [FUNCTION DOC] Check that detaching in place a tensor saved as an input makes backward
        # raise RuntimeError, while detaching a tensor saved as output is fine.
        # Detaching a tensor that is saved input raises
        a = torch.tensor(1.0, requires_grad=True).clone()
        b = a.sin()
        a.detach_()
        with self.assertRaisesRegex(
            RuntimeError, "Trying to use a saved tensor that has been detached"
        ):
            b.backward()

        # Detaching a tensor that is saved as output is OK
        a = torch.tensor(1.0, requires_grad=True).clone()
        b = a.exp()
        a.detach_()
        b.backward()

    def test_saved_variable_packing_unpacking_did_not_save_original_with_hooks(self):
        # [FUNCTION DOC] Check hooks registered on a raw saved result (saved output): values and
        # grad_fn are preserved and backward gives the right gradient.
        # Tests that packing/unpacking a SavedVariable works correctly with user-defined hooks
        # The saved_original / did_not_save_original distinction corresponds to the `save_original`
        # attribute of `SavedVariable`.

        a = torch.randn(5, requires_grad=True)
        y = torch.exp(a)
        y.grad_fn._raw_saved_result.register_hooks(lambda x: x, lambda x: x)
        self.assertEqual(y, y.grad_fn._saved_result)
        self.assertIs(y.grad_fn, y.grad_fn._saved_result.grad_fn)
        y.sum().backward()
        self.assertEqual(a.grad, y)

    def test_saved_variable_packing_unpacking_saved_original_with_default_hooks(self):
        # [FUNCTION DOC] Check global saved-tensor hooks: pack called per saved tensor (two
        # warnings for a multiply), identity, scale and unscale, and scale only hooks produce
        # expected saved values and gradients, and hooks are reset on exit.
        # Tests that default hooks are properly registered, used and reset
        # The saved_original / did_not_save_original distinction corresponds to the `save_original`
        # attribute of `SavedVariable`.
        # See also:
        #  - test_saved_variable_packing_unpacking_saved_original_with_hooks

        def pack(x):
            # [FUNCTION DOC] HELPER: Pack hook that issues a warning and returns the tensor.
            warnings.warn("pack")
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            a = torch.ones(5, requires_grad=True)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                y = a * a
                # should raise two warnings from a being saved twice
                self.assertEqual(len(w), 2)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x / 2):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            self.assertEqual(2 * a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(4 * a, a.grad)

        # Exited hooks correctly
        a = torch.randn(5, requires_grad=True)
        y = a * a
        self.assertEqual(a, y.grad_fn._saved_self)
        self.assertEqual(a, y.grad_fn._saved_other)
        y.sum().backward()
        self.assertEqual(2 * a, a.grad)

    def test_saved_variable_packing_unpacking_did_not_save_original_with_default_hooks(
        self,
    ):
        # [FUNCTION DOC] Check global saved-tensor hooks for a saved output (exp): the saved
        # result and the gradient are as without hooks.
        # See also test_saved_variable_packing_unpacking_did_not_save_original_with_hooks

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = torch.exp(a)
            self.assertEqual(y, y.grad_fn._saved_result)
            y.sum().backward()
            self.assertEqual(a.grad, y)

    def test_setting_default_saved_variable_hooks_twice_should_not_fail(self):
        # [FUNCTION DOC] Check entering saved-tensor hooks contexts nested twice does not fail.
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                pass

    def test_setting_default_saved_variable_hooks_twice_should_use_inner(self):
        # [FUNCTION DOC] Check the innermost saved-tensor hooks are used for tensors created
        # inside, and the outer hooks for others (gradient factors 50 and 18 times).
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 3 * x, lambda x: 3 * x):
            b = torch.randn(5, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(
                lambda x: 5 * x, lambda x: 5 * x
            ):
                a = torch.randn(5, requires_grad=True)
                y = a * a
            z = b * b
        y.sum().backward()
        z.sum().backward()
        self.assertEqual(2 * 5 * 5 * a, a.grad)
        self.assertEqual(2 * 3 * 3 * b, b.grad)

    def test_disabling_saved_tensor_hooks(self):
        # [FUNCTION DOC] Check disable_saved_tensors_hooks prevents entering a hooks context with
        # its message, in either nesting order, and re-enables hooks afterwards.
        with torch.autograd.graph.disable_saved_tensors_hooks("error message"):
            with self.assertRaisesRegex(RuntimeError, "error message"):
                with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                    pass

        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with self.assertRaisesRegex(RuntimeError, "error message"):
                with torch.autograd.graph.disable_saved_tensors_hooks("error message"):
                    pass

        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_disabling_saved_tensor_hooks_nested(self):
        # [FUNCTION DOC] Check nested disable_saved_tensors_hooks contexts report the innermost
        # message and hooks stay disabled until the outer one exits.
        with torch.autograd.graph.disable_saved_tensors_hooks("outer"):
            with torch.autograd.graph.disable_saved_tensors_hooks("inner"):
                with self.assertRaisesRegex(RuntimeError, "inner"):
                    with torch.autograd.graph.saved_tensors_hooks(
                        lambda x: x, lambda x: x
                    ):
                        pass

            self.assertFalse(torch._C._autograd._saved_tensors_hooks_is_enabled())

        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_saved_tensor_hooks_custom_error_propagation(self):
        # [FUNCTION DOC] Check exceptions raised inside pack or unpack hooks propagate to the
        # caller unchanged (pack during forward, unpack during backward).
        class CustomError(Exception):
            # [CLASS DOC] HELPER: Custom exception class.
            pass

        class error_on_pack_hook(torch.autograd.graph.saved_tensors_hooks):
            # [CLASS DOC] HELPER: Saved-tensor hooks context whose pack hook raises the
            # custom error.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Install a raising pack hook and an identity unpack hook.
                def pack_hook(x):
                    # [FUNCTION DOC] HELPER: Raise the custom error with the text pack.
                    raise CustomError("pack")

                super().__init__(pack_hook, lambda x: x)

        class error_on_unpack_hook(torch.autograd.graph.saved_tensors_hooks):
            # [CLASS DOC] HELPER: Saved-tensor hooks context whose unpack hook raises the
            # custom error.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Install an identity pack hook and a raising unpack hook.
                def unpack_hook(x):
                    # [FUNCTION DOC] HELPER: Raise the custom error with the text unpack.
                    raise CustomError("unpack")

                super().__init__(lambda x: x, unpack_hook)

        a = torch.tensor(1.0, requires_grad=True)

        with error_on_pack_hook():
            with self.assertRaisesRegex(CustomError, "pack"):
                out = torch.sin(a)

        with error_on_unpack_hook():
            out = torch.sin(a)
            with self.assertRaisesRegex(CustomError, "unpack"):
                out.backward()

    def test_saved_tensor_hooks_custom_function_intermediates(self):
        # [FUNCTION DOC] Check saved-tensor hooks around a custom Function that saves an
        # intermediate requiring grad does not break backward.
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving a detached clone of an intermediate that
            # requires grad.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Compute an exp intermediate, save a detached copy
                # requiring grad and return another exp.
                intermediate = x.exp()
                ctx.save_for_backward(
                    intermediate.clone().detach_().requires_grad_(True)
                )
                return x.exp()

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Multiply the gradient by the saved intermediate.
                (intermediate,) = ctx.saved_tensors
                return grad_out * intermediate

        a = torch.tensor(1.0, requires_grad=True)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            out = Func.apply(a)
        out.backward()

    def test_node_creation_hook(self):
        # [FUNCTION DOC] Check node_creation_hook fires for the exp and add nodes created inside
        # its context, in order, with the same node objects as the tensors' grad_fn.
        a = torch.randn(2, requires_grad=True)
        nodes = []

        b = a * 2
        with torch.autograd.graph.node_creation_hook(nodes.append):
            c = b.exp()
            d = c + c
        e = d.sum()

        self.assertEqual([n.name() for n in nodes], ["ExpBackward0", "AddBackward0"])
        self.assertIs(nodes[0], c.grad_fn)
        self.assertIs(nodes[1], d.grad_fn)

    def test_node_creation_hook_multi_output_fires_once(self):
        # [FUNCTION DOC] Check a multi-output op with a shared grad_fn fires the node creation
        # hook once, ignoring AccumulateGrad nodes.
        AccumulateGrad = torch._C._functions.AccumulateGrad
        nodes = []
        a = torch.randn(6, requires_grad=True)
        # split() has multiple differentiable outputs that all share a single
        # grad_fn, so the hook fires exactly once regardless of output count.
        # a's AccumulateGrad also fires (created on first use); ignore it here.
        with torch.autograd.graph.node_creation_hook(nodes.append):
            outs = a.split(2)
        split_nodes = [n for n in nodes if not isinstance(n, AccumulateGrad)]
        self.assertEqual(len(split_nodes), 1)
        for out in outs:
            self.assertIs(split_nodes[0], out.grad_fn)

    def test_node_creation_hook_nesting(self):
        # [FUNCTION DOC] Check nested node_creation_hook contexts: the outer hook sees all nodes
        # created inside, the inner hook only those inside it.
        calls = []
        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(
            lambda node: calls.append(("outer", node))
        ):
            b = a * 2
            with torch.autograd.graph.node_creation_hook(
                lambda node: calls.append(("inner", node))
            ):
                c = b.exp()
        # b = a * 2 also creates a's AccumulateGrad, firing the outer hook.
        acc = b.grad_fn.next_functions[0][0]
        expected = [
            ("outer", acc),
            ("outer", b.grad_fn),
            ("outer", c.grad_fn),
            ("inner", c.grad_fn),
        ]
        self.assertEqual(calls, expected)

    def test_node_creation_hook_custom_function(self):
        # [FUNCTION DOC] Check the node creation hook fires once for a custom Function node
        # (ignoring AccumulateGrad).
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function doubling its input in forward and gradient in backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input doubled.
                return x * 2

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient doubled.
                return gO * 2

        AccumulateGrad = torch._C._functions.AccumulateGrad
        nodes = []
        a = torch.randn(2, requires_grad=True)
        # a's AccumulateGrad also fires (created on first use); ignore it here.
        with torch.autograd.graph.node_creation_hook(nodes.append):
            out = Func.apply(a)
        func_nodes = [n for n in nodes if not isinstance(n, AccumulateGrad)]
        self.assertEqual(len(func_nodes), 1)
        self.assertIs(func_nodes[0], out.grad_fn)

    def test_node_creation_hook_inplace(self):
        # [FUNCTION DOC] Check an in-place op fires the node creation hook once and its node
        # becomes the grad_fn.
        nodes = []
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        with torch.autograd.graph.node_creation_hook(nodes.append):
            b.mul_(2)
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], b.grad_fn)

    def test_node_creation_hook_view_of_modified_base(self):
        # [FUNCTION DOC] Check the lazily regenerated grad_fn of a view after its base is modified
        # in place fires the hook when accessed inside the context.
        # The view's grad_fn is lazily regenerated on access after the base
        # is modified in-place; hooks should fire for it if a hook context is
        # active at that point.
        a = torch.randn(4, requires_grad=True).clone()
        view = a[:2]
        a.mul_(2)
        nodes = []
        with torch.autograd.graph.node_creation_hook(nodes.append):
            fn = view.grad_fn
        self.assertIn(fn, nodes)

    def test_node_creation_hook_saved_tensors_populated(self):
        # [FUNCTION DOC] Check the node creation hook runs after saved tensors are stored on the
        # node, so the hook can read them.
        # Hooks fire only after saved tensors have been stored on the node.
        a = torch.randn(2, requires_grad=True)
        b = torch.randn(2, requires_grad=True)
        saved = []

        def hook(node):
            # [FUNCTION DOC] HELPER: Hook skipping AccumulateGrad nodes and recording the saved
            # self and other of the node.
            # a and b's AccumulateGrad nodes also fire; only the mul node has
            # saved tensors.
            if isinstance(node, torch._C._functions.AccumulateGrad):
                return
            saved.append((node._saved_self, node._saved_other))

        with torch.autograd.graph.node_creation_hook(hook):
            a * b
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0], a)
        self.assertEqual(saved[0][1], b)

    def test_node_creation_hook_saved_tensors_populated_custom_function(self):
        # [FUNCTION DOC] Check the node creation hook sees saved tensors on a custom
        # Function node.
        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving its input and doubling it.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return it doubled.
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Unpack saved tensors and return the gradient doubled.
                (x,) = ctx.saved_tensors
                return gO * 2

        a = torch.randn(2, requires_grad=True)
        saved = []

        def hook(node):
            # [FUNCTION DOC] HELPER: Hook skipping AccumulateGrad nodes and recording the saved
            # tensors of the node.
            # a's AccumulateGrad also fires but has no saved_tensors.
            if isinstance(node, torch._C._functions.AccumulateGrad):
                return
            saved.append(node.saved_tensors)

        with torch.autograd.graph.node_creation_hook(hook):
            Func.apply(a)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0], (a,))

    def test_node_creation_hook_inplace_on_view_fires_on_copy_slices(self):
        # [FUNCTION DOC] Check an in-place op on a view fires the hook for the view's regenerated
        # node and the composed CopySlices node of the base, exactly two nodes, in that order.
        # In-place ops on views rebase the base's history onto a CopySlices
        # node that wraps the op's backward node. The hook must fire on the
        # composed CopySlices, never on the wrapped inner node. The view's
        # own grad_fn is regenerated as a new AsStridedBackward0 node and
        # legitimately fires too.
        a = torch.randn(4, requires_grad=True).clone()
        v = a[:2]
        nodes = []
        with torch.autograd.graph.node_creation_hook(nodes.append):
            v.mul_(2)
        # Exactly two nodes fire: the composed CopySlices (a.grad_fn) and the
        # view's regenerated AsStridedBackward0 (v.grad_fn); the wrapped inner
        # node does not fire on its own.
        self.assertEqual(len(nodes), 2)
        self.assertIs(nodes[0], v.grad_fn)
        self.assertIs(nodes[1], a.grad_fn)

    def test_node_creation_hook_fallback_multi_output_fires_once(self):
        # [FUNCTION DOC] Check an op using the not-implemented fallback with two outputs shares
        # one grad_fn and fires the hook once.
        # A custom op without an autograd formula goes through the
        # not-implemented fallback, which attaches one shared grad_fn to all
        # differentiable outputs in a loop. The hook must fire exactly once
        # for that shared node, not once per output.
        with torch.library._scoped_library("_test_nch_fallback", "FRAGMENT") as lib:
            lib.define("foo(Tensor a) -> (Tensor, Tensor)")
            lib.impl("foo", lambda a: (a.detach().clone(), a.detach().clone()), "CPU")
            AccumulateGrad = torch._C._functions.AccumulateGrad
            a = torch.randn(2, requires_grad=True)
            nodes = []
            # a's AccumulateGrad also fires (created on first use); ignore it.
            with torch.autograd.graph.node_creation_hook(nodes.append):
                out1, out2 = torch.ops._test_nch_fallback.foo(a)
            self.assertIs(out1.grad_fn, out2.grad_fn)
            fallback_nodes = [n for n in nodes if not isinstance(n, AccumulateGrad)]
            self.assertEqual(len(fallback_nodes), 1)
            self.assertIs(fallback_nodes[0], out1.grad_fn)

    def test_node_creation_hook_accumulate_grad(self):
        # [FUNCTION DOC] Check the AccumulateGrad node fires the hook when created (first use) but
        # not on reuse, and fires again after the graph is freed.
        # An AccumulateGrad node fires when created (a leaf's first use in the
        # context) but not when the cached one is reused. Freeing the graph
        # drops the node so it fires again on the next use.
        AccumulateGrad = torch._C._functions.AccumulateGrad
        a = torch.randn(2, requires_grad=True)
        nodes = []
        with torch.autograd.graph.node_creation_hook(nodes.append):
            b = a * 2
            # Reusing a while its AccumulateGrad is still alive does not refire.
            c = a * 3
        acc = [n for n in nodes if isinstance(n, AccumulateGrad)]
        self.assertEqual(len(acc), 1)
        self.assertIs(acc[0], b.grad_fn.next_functions[0][0])
        self.assertIs(acc[0], c.grad_fn.next_functions[0][0])

        # Freeing the graph drops the cached node, so it fires again next use.
        # Drop every strong ref to the node (graph + our list) so it dies.
        del b, c, acc
        nodes.clear()
        with torch.autograd.graph.node_creation_hook(nodes.append):
            d = a * 4
        self.assertEqual(len([n for n in nodes if isinstance(n, AccumulateGrad)]), 1)

    def test_node_creation_hook_accumulate_grad_reuse_no_fire(self):
        # [FUNCTION DOC] Check an AccumulateGrad created before the context and kept alive is
        # reused without firing.
        # An AccumulateGrad created before entering the context (and kept alive
        # by an existing graph) is reused, not recreated, so it does not fire.
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        nodes = []
        with torch.autograd.graph.node_creation_hook(nodes.append):
            c = a * 3
        self.assertEqual(
            [n for n in nodes if isinstance(n, torch._C._functions.AccumulateGrad)],
            [],
        )
        self.assertIs(b.grad_fn.next_functions[0][0], c.grad_fn.next_functions[0][0])

    def test_node_creation_hook_no_fire_outside_context(self):
        # [FUNCTION DOC] Check the hook does not fire for nodes created outside the context, nor
        # for backward afterwards.
        nodes = []
        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(nodes.append):
            pass
        b = a * 2
        self.assertEqual(nodes, [])
        # a's AccumulateGrad was created by b = a * 2 outside any context, so
        # it does not fire; backward reuses it and creates no new nodes here.
        b.sum().backward()
        self.assertEqual(nodes, [])

    def test_node_creation_hook_during_backward(self):
        # [FUNCTION DOC] Check nodes created during backward with create_graph under the hook
        # context fire the hook, via propagation to engine threads.
        # Nodes created during backward (create_graph=True) should fire when
        # backward runs under the context; the TLS must propagate to engine
        # worker threads via ThreadLocalState.
        names = []
        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(
            lambda node: names.append(node.name())
        ):
            b = (a * a).sum()
            (grad,) = torch.autograd.grad(b, a, create_graph=True)
        self.assertIn("MulBackward0", names)
        # Nodes building grad's graph were created while backward ran inside
        # the context.
        self.assertIsNotNone(grad.grad_fn)
        self.assertIn(grad.grad_fn.name(), names)

    def test_node_creation_hook_error_propagates(self):
        # [FUNCTION DOC] Check an exception raised by a node creation hook propagates to
        # the caller.
        class CustomError(Exception):
            # [CLASS DOC] HELPER: Custom exception class.
            pass

        def bad_hook(node):
            # [FUNCTION DOC] HELPER: Hook raising the custom error.
            raise CustomError("node creation hook failed")

        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(bad_hook):
            with self.assertRaisesRegex(CustomError, "node creation hook failed"):
                a * 2

    def test_node_creation_hook_ops_in_hook_raise(self):
        # [FUNCTION DOC] Check running an autograd op from inside a node creation hook
        # raises RuntimeError.
        def hook(node):
            # [FUNCTION DOC] HELPER: Hook that runs an op creating a node.
            # This op would create a node inside the hook; that is an error.
            x = torch.randn(2, requires_grad=True)
            (x * 2).sum()

        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(hook):
            with self.assertRaisesRegex(
                RuntimeError, "from inside a node creation hook"
            ):
                a * 2

    def test_node_creation_hook_must_return_none(self):
        # [FUNCTION DOC] Check a node creation hook returning a value raises RuntimeError saying
        # it must return None.
        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(lambda node: node):
            with self.assertRaisesRegex(RuntimeError, "must return None"):
                a * 2

    def test_node_creation_hook_foreach(self):
        # [FUNCTION DOC] Check a foreach op fires the hook once for its shared grad_fn.
        # Out-of-place foreach ops create one grad_fn shared by all outputs.
        AccumulateGrad = torch._C._functions.AccumulateGrad
        nodes = []
        tensors = [torch.randn(2, requires_grad=True) for _ in range(3)]
        # Each leaf's AccumulateGrad also fires (created on first use); ignore.
        with torch.autograd.graph.node_creation_hook(nodes.append):
            outs = torch._foreach_mul(tensors, 2)
        foreach_nodes = [n for n in nodes if not isinstance(n, AccumulateGrad)]
        self.assertEqual(len(foreach_nodes), 1)
        for out in outs:
            self.assertIs(foreach_nodes[0], out.grad_fn)

    def test_node_creation_hook_foreach_inplace(self):
        # [FUNCTION DOC] Check an in-place foreach op fires the hook with each tensor's grad_fn
        # in order.
        nodes = []
        tensors = [torch.randn(2, requires_grad=True) + 0 for _ in range(3)]
        with torch.autograd.graph.node_creation_hook(nodes.append):
            torch._foreach_mul_(tensors, 2)
        self.assertEqual(nodes, [t.grad_fn for t in tensors])

    def test_node_creation_hook_delayed_error(self):
        # [FUNCTION DOC] Check the DelayedError node fires the hook: AccumulateGrad, multiply
        # node, then the error node.
        # DelayedError attaches its Error node via wrap_outputs
        # (csrc/autograd/functions/utils.cpp), not through codegen.
        nodes = []
        a = torch.randn(2, requires_grad=True)
        err_fn = torch._C._functions.DelayedError("boom", 1)
        with torch.autograd.graph.node_creation_hook(nodes.append):
            b = a * 2
            c = err_fn(b)
        # b = a * 2 also creates a's AccumulateGrad, which fires first, then
        # b's MulBackward0, then the Error node wrapping c.
        self.assertEqual(len(nodes), 3)
        self.assertIsInstance(nodes[0], torch._C._functions.AccumulateGrad)
        self.assertIs(nodes[1], b.grad_fn)
        self.assertIs(nodes[2], c.grad_fn)

    def test_node_creation_hook_register_backward_hooks(self):
        # [FUNCTION DOC] Check a node creation hook can register pre and post hooks on each node
        # that then run in order during backward.
        # The motivating pattern: capture state at node creation, restore it
        # around the node's execution in backward.
        events = []

        def creation_hook(node):
            # [FUNCTION DOC] HELPER: Register logging pre and post hooks on the node.
            name = node.name()
            node.register_prehook(lambda gO: events.append(("pre", name)))
            node.register_hook(lambda gI, gO: events.append(("post", name)))

        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(creation_hook):
            b = (a * a).exp()
        b.sum().backward()
        self.assertIn(("pre", "ExpBackward0"), events)
        self.assertIn(("post", "ExpBackward0"), events)
        self.assertIn(("pre", "MulBackward0"), events)
        self.assertIn(("post", "MulBackward0"), events)
        self.assertLess(
            events.index(("pre", "ExpBackward0")),
            events.index(("post", "ExpBackward0")),
        )

    def test_node_creation_hook_checkpoint_recompute(self):
        # [FUNCTION DOC] Check checkpoint recomputation during backward under the hook context
        # creates more nodes that fire the hook.
        # Recomputation during backward recreates graph nodes; those should
        # fire if backward runs under the context.
        names = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Exp then sine.
            return x.exp().sin()

        a = torch.randn(2, requires_grad=True)
        with torch.autograd.graph.node_creation_hook(
            lambda node: names.append(node.name())
        ):
            out = torch.utils.checkpoint.checkpoint(fn, a, use_reentrant=False)
            forward_count = names.count("ExpBackward0")
            out.sum().backward()
        # Once from forward (discarded graph) and once from recompute.
        self.assertGreater(names.count("ExpBackward0"), forward_count)

    def test_unpack_hooks_exec_count(self):
        # [FUNCTION DOC] Check that pack hooks run once at save time and unpack hooks only at
        # backward time.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Multiply two tensors.
            return x * y

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            # [FUNCTION DOC] HELPER: Pack hook counting calls.
            nonlocal pack_count
            pack_count += 1
            return x

        # unpack hook shouldn't run during compilation, while we trace the forward
        def unpack_hook(x):
            # [FUNCTION DOC] HELPER: Unpack hook counting calls.
            nonlocal unpack_count
            unpack_count += 1
            return x

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out_test = f(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            out_test.sum().backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    def test_saved_tensors_hook_version_counter_not_shared(self):
        # [FUNCTION DOC] Check that a tensor returned by a pack hook does not share its version
        # counter with the saved tensor, so in-place changes in backward do not change the
        # original version.
        class Test(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function saving its input and modifying the unpacked tensor in
            # place in backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Save the input and return its sine.
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Unpack the saved tensor, add one in place and check the
                # version of the original is unchanged.
                (x,) = ctx.saved_tensors
                before = a._version
                x.add_(1)
                self.assertEqual(a._version, before)
                return grad_output

        a = torch.tensor(1.0, requires_grad=True)
        a_replacement = a.clone()

        def pack_hook(x):
            # [FUNCTION DOC] HELPER: Pack hook returning a fixed replacement tensor.
            return a_replacement

        def unpack_hook(x):
            # [FUNCTION DOC] HELPER: Unpack hook returning its input.
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            b = Test.apply(a)

        b.backward()

    def test_save_on_cpu_and_checkpoint(self):
        # [FUNCTION DOC] Check save_on_cpu combined with non-reentrant checkpoint in both nesting
        # orders gives the same gradient as plain computation.
        a = torch.randn(2, 2, requires_grad=True)

        b = a.pow(2).pow(2).pow(2).pow(2)
        b.sum().backward()
        b_grad = a.grad.clone()
        a.grad.zero_()

        with torch.autograd.graph.save_on_cpu():
            h = a.pow(2)
            h = checkpoint(lambda x: x.pow(2).pow(2), h, use_reentrant=False)
            c = h.pow(2)
        c.sum().backward()
        c_grad = a.grad.clone()
        a.grad.zero_()

        def f(a):
            # [FUNCTION DOC] HELPER: Chain of squares with the middle part under save_on_cpu.
            h = a.pow(2)
            with torch.autograd.graph.save_on_cpu():
                h = h.pow(2).pow(2)
            return h.pow(2)

        d = checkpoint(f, a, use_reentrant=False)
        d.sum().backward()
        d_grad = a.grad.clone()

        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    def test_pack_hook_with_inplace_modification_should_fail(self):
        # [FUNCTION DOC] Check a pack hook that modifies its input in place raises RuntimeError,
        # both as a global hook and when registered on a raw saved tensor.
        a = torch.randn(5, requires_grad=True)

        def inc(x):
            # [FUNCTION DOC] HELPER: Add one to the tensor in place and return it.
            x += 1
            return x

        with torch.autograd.graph.saved_tensors_hooks(inc, lambda x: x):
            with self.assertRaisesRegex(
                RuntimeError,
                "A saved tensor pack hook is modifying its input in place.",
            ):
                y = torch.exp(a)

        y = torch.exp(a)
        with self.assertRaisesRegex(
            RuntimeError, "A saved tensor pack hook is modifying its input in place."
        ):
            y.grad_fn._raw_saved_result.register_hooks(inc, lambda x: x)

    def test_saving_variable_to_disk(self):
        # [FUNCTION DOC] Check saved tensors can be packed to disk files and unpacked from them,
        # with the gradient still correct.
        with tempfile.TemporaryDirectory() as tmp_dir:

            def pack(x):
                # [FUNCTION DOC] HELPER: Save the tensor to a uniquely named file and return the
                # file name.
                name = os.path.join(tmp_dir, str(uuid.uuid4()))
                torch.save(x, name)
                return name

            def unpack(name):
                # [FUNCTION DOC] HELPER: Load a tensor from the file name.
                return torch.load(name)

            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                a = torch.ones(5, requires_grad=True)
                y = a * a
                self.assertEqual(a, y.grad_fn._saved_self)

                y.sum().backward()
                self.assertEqual(2 * a, a.grad)

    def test_default_saved_tensors_hooks_double_backward(self):
        # [FUNCTION DOC] Check global saved-tensor hooks combined with double backward and
        # create_graph: gradient factors 6, 12, 24 and 48 depending on how many saved tensors pass
        # through the scale hook.
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a**3
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            self.assertEqual(6 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a**3
            s = torch.sum(y)
        (g,) = torch.autograd.grad(s, (a,), create_graph=True)
        g.sum().backward()
        # factor 2 because only a is saved once
        self.assertEqual(6 * 2 * a, a.grad)

        a = torch.randn(5, requires_grad=True)
        y = a**3
        s = torch.sum(y)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            # factor 4 because pow_backward is grad * (exp * self.pow(exp - 1))
            # so grad is saved and self (i.e. a) is saved
            self.assertEqual(6 * 4 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a**3
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            # combining the two above blocks: 2 * 4 = 8
            # note that in that sense, a is saved twice
            self.assertEqual(6 * 8 * a, a.grad)

    def test_wrapped_number_saved_tensors_hooks(self):
        # [FUNCTION DOC] Check wrapped numbers (Python scalars saved as tensors) do not trigger
        # saved-tensor hooks.
        def err_hook(x):
            # [FUNCTION DOC] HELPER: Hook that always raises RuntimeError.
            raise RuntimeError("this hook should not be called")

        with torch.autograd.graph.saved_tensors_hooks(err_hook, err_hook):
            a = torch.randn(5, requires_grad=True)
            out = (a * 3).sum()
            # 3 is saved as a saved tensor because it is a wrapped number, but
            # wrapped numbers should be special cased to not trigger saved variable hooks
            torch.autograd.grad(out, (a,))

    def test_graph_save_on_cpu(self):
        # [FUNCTION DOC] Check save_on_cpu keeps values, dtype and layout of saved dense and
        # sparse tensors, with and without pinned memory and CUDA, and backward gives the
        # right gradient.
        def test(get_input, cuda, pin_memory):
            # [FUNCTION DOC] HELPER: Under save_on_cpu, build a product of an input, check the
            # saved tensors' values, dtype and layout, run backward and compare the gradient.
            with torch.autograd.graph.save_on_cpu(pin_memory):
                a = get_input()
                if cuda:
                    a.cuda()
                y = a * a
                self.assertEqual(a, y.grad_fn._saved_self)
                self.assertEqual(a, y.grad_fn._saved_other)
                self.assertEqual(a.dtype, y.grad_fn._saved_self.dtype)
                self.assertEqual(a.layout, y.grad_fn._saved_self.layout)
                if y.is_sparse:
                    y = y.to_dense()
                y.sum().backward()

                actual = 2 * a
                expected = a.grad
                if a.is_sparse:
                    actual = actual.coalesce()
                    expected = expected.coalesce()

                self.assertEqual(actual, expected)

        for cuda in [False] + ([True] if torch.cuda.is_available() else []):
            for pin_memory in [True, False]:
                # FloatTensor
                test(lambda: torch.randn(5, requires_grad=True), cuda, pin_memory)
                # DoubleTensor
                test(
                    lambda: torch.randn(5, requires_grad=True, dtype=torch.double),
                    cuda,
                    pin_memory,
                )
                # Sparse tensor
                x = torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(),
                    torch.tensor([1.0, 1.0]),
                    requires_grad=True,
                )
                test(lambda: x, cuda, pin_memory)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "test requires CUDA or XPU")
    def test_graph_save_on_cpu_cuda(self):
        # [FUNCTION DOC] Check GPU memory with save_on_cpu: memory with grad exceeds memory
        # without grad, and with save_on_cpu equals memory without grad. Requires CUDA or XPU.
        device_type = torch.accelerator.current_accelerator().type

        def f(x):
            # [FUNCTION DOC] HELPER: Add one and square.
            a = x + 1
            return a * a

        # with grad
        a = torch.ones(1, requires_grad=True, device=device_type)
        y = f(a)
        memory_with_grad = (
            torch.cuda.memory_allocated() if TEST_CUDA else torch.xpu.memory_allocated()
        )

        del a
        del y

        # without grad
        a = torch.ones(1, requires_grad=True, device=device_type)
        with torch.no_grad():
            y = f(a)
        memory_without_grad = (
            torch.cuda.memory_allocated() if TEST_CUDA else torch.xpu.memory_allocated()
        )

        self.assertGreater(memory_with_grad, memory_without_grad)

        del a
        del y

        # with hooks
        with torch.autograd.graph.save_on_cpu():
            a = torch.ones(1, requires_grad=True, device=device_type)
            y = f(a)
            memory_with_hooks = (
                torch.cuda.memory_allocated()
                if TEST_CUDA
                else torch.xpu.memory_allocated()
            )
            self.assertEqual(memory_with_hooks, memory_without_grad)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "test requires CUDA and XPU")
    def test_scalar_grad_mixed_device(self):
        # [FUNCTION DOC] Check a scalar CPU tensor with a device tensor product backward works.
        # Requires CUDA or XPU.
        device_type = torch.accelerator.current_accelerator().type
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.randn(2, 2, device=device_type)
        out = x * y
        out.sum().backward()

    @scoped_load_inline
    def test_multi_grad_all_hooks(self, load_inline):
        # [FUNCTION DOC] Check multi-grad hooks in all mode across node kinds (C++ node, Python
        # Function, undefined grad node, C++ custom op built inline): the hook fires once when all
        # required gradients arrive, with proper masks, not on errors, and not after removal.
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        # Ensure we properly detect all types of Nodes here
        # C++ Node
        t1 = t1.mul(2)

        # Python custom Function
        class Foo(Function):
            # [CLASS DOC] HELPER: Identity Function.
            @staticmethod
            def forward(ctx, a):
                # [FUNCTION DOC] HELPER: Return a clone.
                return a.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return gO

        t2 = Foo.apply(t2)

        # C++ Node
        t3 = torch._C._functions.UndefinedGrad()(t3)

        # C++ Custom Op
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x.clone();
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_multigrad_all_hooks, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_multigrad_all_hooks",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        t4 = torch.ops.test_multigrad_all_hooks.custom_op_backed_by_autograd_fn(t4)

        res = [None] * 4
        count = [0]

        def hook(grads):
            # [FUNCTION DOC] HELPER: Hook counting calls and recording which gradients
            # are defined.
            nonlocal res
            count[0] += 1
            res = [g is not None for g in grads]

        handle = torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)

        out = t2 * t3

        out.sum().backward(inputs=(t2, t3), retain_graph=True)
        self.assertEqual(count[0], 1)
        self.assertEqual(res, [False, True, True, False])

        out.sum().backward(inputs=(t1, t4), retain_graph=True)
        self.assertEqual(count[0], 1)

        out.sum().backward(inputs=(t1, t3), retain_graph=True)
        self.assertEqual(count[0], 2)
        self.assertEqual(res, [False, False, True, False])

        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward raises RuntimeError.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Raise RuntimeError with the text error message.
                raise RuntimeError("error message")

        out = Func.apply(t2) * t3
        with self.assertRaisesRegex(RuntimeError, "error message"):
            out.sum().backward(inputs=(t2, t3), retain_graph=True)
        self.assertEqual(count[0], 2)

        handle.remove()
        out.sum().backward(inputs=(t1, t3), retain_graph=True)
        self.assertEqual(count[0], 2)

    def test_multi_grad_any_hooks(self):
        # [FUNCTION DOC] Check multi-grad hooks in any mode: per module the any-mode hook runs
        # once before the regular tensor hook, and a removed any-mode hook does not run.
        hook_id = 0
        any_hook_handles: list[RemovableHandle] = []

        class MultiOutputModule(nn.Module):
            # [CLASS DOC] HELPER: Module producing two outputs from a linear layer, registering a
            # tensor hook and an any-mode multi-grad hook on each call.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.lin = nn.Linear(3, 3)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Apply the layer, return sine and cosine, and register the
                # hooks with the running ids.
                z = self.lin(x)
                out = torch.sin(z), torch.cos(z)
                nonlocal hook_id
                z.register_hook(partial(hook, hook_id))
                hook_id += 1
                any_hook_handles.append(
                    torch.autograd.graph.register_multi_grad_hook(
                        out, partial(hook, hook_id), mode="any"
                    )
                )
                hook_id += 1
                return out

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Model chaining two multi-output modules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the two modules.
                super().__init__()
                self.mod1 = MultiOutputModule()
                self.mod2 = MultiOutputModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Run the first module, add its outputs and run the
                # second module.
                y = self.mod1(x)
                z = y[0] + y[1]
                return self.mod2(z)

        hook_order: list[int] = []
        hook_count = 0

        def hook(hook_id: int, *unused):
            # [FUNCTION DOC] HELPER: Count the call and append the id to the order list.
            nonlocal hook_count
            nonlocal hook_order
            hook_count += 1
            hook_order.append(hook_id)

        # Any hooks: IDs 1 and 3; regular hooks: IDs 0 and 2
        model = Model()
        inp = torch.randn((2, 3))
        out = model(inp)
        (out[0] + out[1]).sum().backward()
        # Check that the any-hook runs only once and before the regular hook
        # for each module
        self.assertEqual(len(any_hook_handles), 2)
        self.assertEqual(hook_order, [3, 2, 1, 0])

        hook_id = 0
        hook_order.clear()
        any_hook_handles.clear()
        out = model(inp)
        for handle in any_hook_handles:
            handle.remove()
        (out[0] + out[1]).sum().backward()
        # Check that the any-hook does not run if removed
        self.assertEqual(hook_order, [2, 0])

    def test_multi_grad_hooks_invalid_mode(self):
        # [FUNCTION DOC] Check register_multi_grad_hook with an invalid mode raises ValueError
        # listing the valid modes.
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        regex = r"Expects mode to be one of \('all', 'any'\) but got foo"
        with self.assertRaisesRegex(ValueError, regex):
            torch.autograd.graph.register_multi_grad_hook(
                (t1, t2), lambda _: None, mode="foo"
            )

    def test_pynode_destruction_deadlock(self):
        # [FUNCTION DOC] Run a subprocess script that builds a graph where a Python Function node
        # is destroyed after an error in backward, to check it does not deadlock; fails on timeout
        # of twenty seconds.
        # KNOWN ISSUE: the embedded script defines forward twice in the Function class (the second
        # was probably meant to be backward), so the class has no backward and the scenario may
        # not be the intended one.
        script = """
import torch

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def forward(ctx, gO):
        return gO.clone()

def get_out():
    inp = torch.rand(2, requires_grad=True)

    # The python function is first so that it runs
    # last in the backward pass
    right = Foo.apply(inp)

    # An op that creates new memory
    left1 = inp.clone()
    # An op that saves its input
    left2 = left1 ** 2

    # Inplace modify so that the backward for
    # left2 always raises an error
    left1 += 1

    # An op that takes both side as input.
    # After running, both side's last op will be in
    # the ready queue
    # And the op for left will run first as it was
    # executed last during the forward
    out = left2 + right

    return out

# Nothing should be global variables here as, from what
# I can see, python leaks all the global objects
get_out().sum().backward()

# This used to deadlock when the PyNode is being destroyed after
# the error is raised.
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
                # It is ok to have an extra long timeout here as a timeout means the test failed
                timeout=20,
            )
        except subprocess.TimeoutExpired as e:
            self.fail(
                msg="Example code timed out! See the code sample in the test for details."
            )
        except subprocess.CalledProcessError as e:
            if e.returncode < 0:
                # Sometimes we segfault instead of deadlocking
                self.fail("Subprocess exited with a fatal signal")
            else:
                err_msg = (
                    "RuntimeError: one of the variables needed for gradient computation"
                )
                self.assertTrue(err_msg in e.output.decode("utf-8"))

    def test_view_func_replay(self):
        # [FUNCTION DOC] Check view replay functions (forward replay and the unsafe and reverse
        # variants) for many view ops, view chains, chains with missing replay functions, and
        # nested tensor views: metadata of replayed views matches and the view relationship to the
        # base is kept.
        with torch.autograd._force_original_view_tracking(True):

            def _assert_match_metadata(a, b):
                # [FUNCTION DOC] HELPER: Assert two tensors have equal size, stride, storage
                # offset, device and dtype.
                self.assertEqual(a.size(), b.size())
                self.assertEqual(a.stride(), b.stride())
                self.assertEqual(a.storage_offset(), b.storage_offset())
                self.assertEqual(a.device, b.device)
                self.assertEqual(a.dtype, b.dtype)

            def _test_fn(fn, inp, *args, use_unsafe_view_func=False):
                # [FUNCTION DOC] HELPER: Apply a view op to an input, then for each resulting view
                # check it is a view of the input, replay it on a clone of the input, and replay
                # in reverse on a detached output, comparing values and metadata.
                outs = fn(inp, *args)
                # handle functions that return multiple views (e.g. split)
                if isinstance(outs, torch.Tensor):
                    outs = [outs]

                for out in outs:
                    self.assertTrue(out._is_view())
                    self.assertTrue(out._base is inp)

                    # forward view_func
                    new_inp = inp.clone()
                    _assert_match_metadata(new_inp, inp)
                    if use_unsafe_view_func:
                        new_out = out._view_func_unsafe(new_inp)
                    else:
                        new_out = out._view_func(new_inp)
                    _assert_match_metadata(new_out, out)
                    self.assertEqual(new_out, out)

                    # reverse view_func
                    new_out = out.detach()
                    new_inp = out._rev_view_func_unsafe(new_out)
                    _assert_match_metadata(new_inp, inp)
                    self.assertTrue(new_inp._is_view())
                    self.assertTrue(new_inp._base is new_out)

            # test individual view ops
            _test_fn(torch.ops.aten.alias.default, torch.rand(2, 2))
            _test_fn(torch.as_strided, torch.rand(2, 2), (4,), (1,))
            _test_fn(torch.chunk, torch.rand(2, 4), 2, -1)
            _test_fn(torch.diagonal, torch.rand(4, 4))
            _test_fn(torch.ops.aten.expand.default, torch.rand(4, 1), (-1, 3))
            _test_fn(torch.narrow, torch.rand(2, 2), 0, 1, 1)
            _test_fn(torch.permute, torch.rand(2, 3, 4), (1, 0, 2))
            _test_fn(torch.select, torch.rand(2, 2), 0, 0)
            _test_fn(torch.ops.aten.slice.Tensor, torch.rand(2, 2), 1, 1, 2)
            _test_fn(torch.split, torch.rand(2, 2), 1)
            _test_fn(torch.split_with_sizes, torch.rand(2, 4), [1, 3], -1)
            _test_fn(torch.squeeze, torch.rand(2, 1, 4))
            _test_fn(torch.squeeze, torch.rand(2, 1, 4), 1)
            _test_fn(torch.squeeze, torch.rand(2, 1, 1, 4), [1, 2])
            _test_fn(torch.t, torch.rand(2, 4))
            _test_fn(torch.transpose, torch.rand(2, 4), 0, 1)
            _test_fn(torch.unbind, torch.rand(1, 5))
            _test_fn(torch.ops.aten.unfold.default, torch.rand(1, 5), 1, 3, 2)
            _test_fn(torch.unsqueeze, torch.rand(2, 4), -2)
            _test_fn(torch.ops.aten.view.default, torch.rand(2, 10), (-1, 5, 2))
            _test_fn(torch.view_as_complex, torch.rand(2, 2))
            _test_fn(torch.view_as_real, torch.rand(2, 2, dtype=torch.cfloat))

            # test view chains
            _test_fn(
                lambda x: x.unsqueeze(-1).transpose(-1, -2).squeeze(1),
                torch.randn(2, 4),
            )
            _test_fn(
                lambda x: x.chunk(2, -1)[0].transpose(0, 1).unsqueeze(-1),
                torch.randn(2, 3, 4),
            )
            _test_fn(
                lambda x: x.split_with_sizes([1, 3], -1)[0].chunk(2, 0),
                torch.randn(2, 3, 4),
            )

            # chains with missing view_func()s use as_strided() to cover the gaps
            def chain_with_only_parent_view_func(x):
                # [FUNCTION DOC] HELPER: Split with view tracking on, then chunk with view
                # tracking off, leaving a gap in replay functions.
                with torch.autograd._force_original_view_tracking(True):
                    x = x.split_with_sizes([1, 3], -1)[0]

                with torch.autograd._force_original_view_tracking(False):
                    x = x.chunk(2, 0)

                return x

            _test_fn(chain_with_only_parent_view_func, torch.randn(2, 3, 4))

            def chain_with_only_current_view_func(x):
                # [FUNCTION DOC] HELPER: Split with view tracking off, then chunk with view
                # tracking on.
                with torch.autograd._force_original_view_tracking(False):
                    x = x.split_with_sizes([1, 3], -1)[0]

                with torch.autograd._force_original_view_tracking(True):
                    x = x.chunk(2, 0)

                return x

            _test_fn(chain_with_only_current_view_func, torch.randn(2, 3, 4))

            # TODO: Move this somewhere else
            # test NT views
            from torch.nested._internal.nested_tensor import (
                nested_view_from_values_offsets,
            )

            values = torch.randn(10, 5)
            offsets = torch.tensor([0, 3, 6, 10])
            _test_fn(nested_view_from_values_offsets, values, offsets)

            nt = nested_view_from_values_offsets(values, offsets).detach().clone()
            _test_fn(
                torch.ops.aten._nested_get_values.default, nt, use_unsafe_view_func=True
            )

            def chain_nt_to_dense_back_and_forth(nt):
                # [FUNCTION DOC] HELPER: Convert a nested tensor to its values and back to a
                # nested tensor and its values with cloned offsets.
                # NJT1 -> dense -> NJT2 -> dense
                offsets2 = nt.offsets().detach().clone()
                return nested_view_from_values_offsets(nt.values(), offsets2).values()

            _test_fn(chain_nt_to_dense_back_and_forth, nt, use_unsafe_view_func=True)

            def chain_dense_to_nt_back_and_forth(values, offsets):
                # [FUNCTION DOC] HELPER: Convert dense values and offsets to a nested tensor, its
                # values, and a second nested tensor with cloned offsets.
                offsets2 = offsets.detach().clone()
                # dense -> NJT1 -> dense -> NJT2
                return nested_view_from_values_offsets(
                    nested_view_from_values_offsets(values, offsets).values(), offsets2
                )

            _test_fn(
                chain_dense_to_nt_back_and_forth,
                values,
                offsets,
                use_unsafe_view_func=True,
            )

    def test_view_func_replay_with_modified_state(self):
        # [FUNCTION DOC] Check view replay with modified saved state: a visitor on saved integers
        # changes the selected index, a visitor on saved tensors changes the replayed shape, and
        # the saved state is restored afterwards.
        with torch.autograd._force_original_view_tracking(True):
            base = torch.randn(3, 4, 5)
            view = base.select(1, 2)

            def symint_visitor_fn(x):
                # [FUNCTION DOC] HELPER: Add one to the saved integer value.
                # modify saved index
                return x + 1

            # ensure modifying state changes view replay
            new_base = torch.randn_like(base)
            new_view = view._view_func(new_base, symint_visitor_fn=symint_visitor_fn)
            self.assertEqual(new_view, new_base.select(1, 3))

            # ensure saved state reverts back afterwards
            self.assertEqual(view._view_func(new_base), new_base.select(1, 2))

            # check modifying tensor state. currently, slice_inverse() is the only
            # view that saves a tensor
            base = torch.randn(3, 4, 5)
            sliced = base[:, 2:3, :].detach()
            view = torch.ops.aten.slice_inverse(sliced, base, 1, 2, 3, 1)

            replacement_shape = (1, 2, 3)

            def tensor_visitor_fn(x):
                # [FUNCTION DOC] HELPER: Return a random tensor of a smaller fixed shape.
                # return tensor with a smaller shape than the saved one
                return torch.randn(*replacement_shape)

            # ensure modifying state changes view replay
            new_sliced = torch.ones_like(base)[:, 2:3, :].detach()
            new_view = view._view_func(new_sliced, tensor_visitor_fn=tensor_visitor_fn)
            self.assertEqual(new_view.shape, replacement_shape)
            self.assertEqual(
                new_view, new_sliced.as_strided(replacement_shape, (6, 3, 1))
            )

            # ensure saved state reverts back afterwards
            self.assertEqual(view._view_func(sliced), base)

    def test_setup_context_when_forward_has_default_args(self):
        # [FUNCTION DOC] Check Functions with separate setup_context whose forward has a default
        # argument, both with static methods and class methods: result is 8 and gradient is 12
        # with the default exponent.
        class PowFunction(Function):
            # [CLASS DOC] HELPER: Power Function with default exponent three, using
            # static methods.
            @staticmethod
            def forward(x, y=3):
                # [FUNCTION DOC] HELPER: Raise x to the power y.
                return torch.pow(x, y)

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Save the input and store the exponent on ctx.
                x, y = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient times the exponent times the lower
                # power, and None.
                (x,) = ctx.saved_tensors
                y = ctx.y
                return gO * y * torch.pow(x, y - 1), None

        class PowFunctionWithClassmethod(Function):
            # [CLASS DOC] HELPER: Same power Function using class methods.
            @classmethod
            def forward(cls, x, y=3):
                # [FUNCTION DOC] HELPER: Raise x to the power y.
                return torch.pow(x, y)

            @classmethod
            def setup_context(cls, ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Save the input and store the exponent on ctx.
                x, y = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            @classmethod
            def backward(cls, ctx, gO):
                # [FUNCTION DOC] HELPER: Return the gradient times the exponent times the lower
                # power, and None.
                (x,) = ctx.saved_tensors
                y = ctx.y
                return gO * y * torch.pow(x, y - 1), None

        x = torch.tensor(2.0, requires_grad=True)

        y = torch.tensor(8.0)
        y_expected = torch.tensor(12.0)

        y1 = PowFunction.apply(x)
        (y1_expected,) = torch.autograd.grad(y1, x)

        y2 = PowFunctionWithClassmethod.apply(x)
        (y2_expected,) = torch.autograd.grad(y2, x)

        self.assertEqual(y, y1)
        self.assertEqual(y_expected, y1_expected)
        self.assertEqual(y, y2)
        self.assertEqual(y_expected, y2_expected)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_gradcheck_default_device_placement_context(self):
        # [FUNCTION DOC] Check fast gradcheck works when the default device is set to CUDA by the
        # user. Requires CUDA.
        # During gradcheck with fast_mode=True, we create a random vector on the CPU device using a CPU generator.
        # This test ensures that this still works when the default device is set to something else by the user.
        with torch.device("cuda"):
            x = torch.randn(3, dtype=torch.double, requires_grad=True)

            def func(inp):
                # [FUNCTION DOC] HELPER: Square elementwise with a float exponent.
                return inp**2.0

            self.assertTrue(gradcheck(func, x, fast_mode=True))

    def test_grad_thread_safety(self):
        # [FUNCTION DOC] Check concurrent reads of the grad attribute from several threads, after
        # backward, return the same tensor in every thread.
        import threading
        from concurrent.futures import ThreadPoolExecutor

        NUM_ITERS = 10
        NUM_THREADS = 4

        # Concurrent calls to tensor.untyped_storage()
        def access_grad(tensor, barrier):
            # [FUNCTION DOC] HELPER: Wait on a barrier then return a weak reference to the
            # tensor grad.
            barrier.wait()
            return weakref.ref(tensor.grad)

        for i in range(NUM_ITERS):
            tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            (tensor**2).sum().backward()

            barrier = threading.Barrier(NUM_THREADS)
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [
                    executor.submit(access_grad, tensor, barrier)
                    for _ in range(NUM_THREADS)
                ]

                # Check that all the grad tensors returned were the same
                for future in futures:
                    self.assertEqual(future.result()(), tensor.grad)
                self.assertIsNotNone(tensor.grad)

    def test_batch_norm_errors_on_third_order_grad(self):
        # [FUNCTION DOC] Check batch norm raises RuntimeError saying third and higher order
        # derivatives are unsupported.
        x = torch.randn(8, 3, requires_grad=True)
        y = torch.nn.functional.batch_norm(x, None, None, training=True)
        (g,) = torch.autograd.grad(y.sum(), x, create_graph=True)
        (g2,) = torch.autograd.grad(g.sum(), x, create_graph=True)
        with self.assertRaisesRegex(
            RuntimeError, "batch_norm does not support 3rd\\+ order derivatives"
        ):
            torch.autograd.grad(g2.sum(), x)


def index_perm_variable(shape, max_indices):
    # [FUNCTION DOC] HELPER: Return a random permutation of indices below a maximum, taking
    # as many as the product of the shape, reshaped to the shape (an int shape is treated as
    # one dimension).
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index


def bernoulli_scalar():
    # [FUNCTION DOC] HELPER: Return a zero-dimensional uint8 tensor filled with a random
    # Bernoulli draw.
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()


class TestAutogradForwardModeBatchedGrad(TestCase):
    # [CLASS DOC] Tests of forward mode AD together with batched (vmap) tangents: gradcheck with
    # batched forward grad, and layout and sharing behavior of tangents for out-of-place, in-place
    # on view and metadata-checks.
    def test_out_of_place_basic(self):
        # [FUNCTION DOC] Check gradcheck passes for sine and add with forward AD, batched grad and
        # batched forward grad.
        a = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        b = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            gradcheck(
                torch.sin,
                a,
                check_forward_ad=True,
                check_batched_grad=True,
                check_batched_forward_grad=True,
            )
        )
        self.assertTrue(
            gradcheck(
                torch.add,
                (a, b),
                check_forward_ad=True,
                check_batched_grad=True,
                check_batched_forward_grad=True,
            )
        )

    def test_out_of_place_not_same_layout(self):
        # [FUNCTION DOC] Check a vmapped jvp returns a new tensor for a transposed (non-matching
        # layout) primal rather than sharing the tangent.
        input = torch.zeros([2, 2]).transpose(0, 1)
        tangent = torch.zeros([2, 2, 2])

        def jvp(tangent):
            # [FUNCTION DOC] HELPER: Make a dual from the primal and the tangent and return the
            # unpacked tangent.
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                return fwAD.unpack_dual(x)[1]

        x_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(tangent)

        self.assertIsNot(x_tangent, tangent)

    def test_inplace_on_view_same_layout(self):
        # [FUNCTION DOC] Check in-place copy into a view with the same layout under vmap shares
        # the tangent between view and base and returns the original tangent for the source.
        input = torch.zeros([2, 2])
        tangent = torch.zeros([2, 2, 2])
        base = torch.zeros([2, 2])
        view = base.view_as(base)

        def jvp(tangent):
            # [FUNCTION DOC] HELPER: Copy a dual into a view and return the tangents of the dual,
            # the view and the base.
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                view.copy_(x)
                return (
                    fwAD.unpack_dual(x)[1],
                    fwAD.unpack_dual(view)[1],
                    fwAD.unpack_dual(view._base)[1],
                )

        x_tangent, view_tangent, base_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(
            tangent
        )

        self.assertFalse(
            view_tangent._is_view()
        )  # Optimization to share the same tensor!
        self.assertIs(view_tangent, base_tangent)
        self.assertIs(x_tangent, tangent)

    def test_inplace_on_view_not_same_layout(self):
        # [FUNCTION DOC] Check in-place copy into a transposed view under vmap keeps the view
        # tangent as a view of the base tangent and distinct from the source tangent.
        input = torch.zeros([2, 2])
        tangent = torch.zeros([2, 2, 2])
        view = torch.zeros([2, 2]).transpose(0, 1)

        def jvp(tangent):
            # [FUNCTION DOC] HELPER: Copy a dual into a transposed view and return the tangents of
            # the dual, the view and the base.
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                view.copy_(x)
                return (
                    fwAD.unpack_dual(x)[1],
                    fwAD.unpack_dual(view)[1],
                    fwAD.unpack_dual(view._base)[1],
                )

        x_tangent, view_tangent, base_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(
            tangent
        )

        self.assertIs(view_tangent._base, base_tangent)
        self.assertIs(x_tangent, tangent)
        self.assertIsNot(view_tangent, tangent)

    def test_metadata_check_for_storage_numel_skipped(self):
        # [FUNCTION DOC] Check the tangent metadata check skips storage size under vmap: no copy
        # is made but as_strided beyond the tensor raises RuntimeError.
        # See: test_metadata_check_checks_storage_numel for the reverse of this test
        primal = torch.randn(5)[:4].detach()
        self.assertEqual(len(primal.storage()), 5)
        tangent = torch.randn(10, 4)

        def jvp(tangent):
            # [FUNCTION DOC] HELPER: Make a dual, check the tangent is returned unchanged and that
            # as_strided past the tensor raises, then return the tangent.
            with fwAD.dual_level():
                dual = fwAD.make_dual(primal, tangent)
                _, unpacked_tangent = fwAD.unpack_dual(dual)

                # No copy is made
                self.assertIs(tangent, unpacked_tangent)

                # as_strided raises
                with self.assertRaisesRegex(
                    RuntimeError, "can access memory outside of `tensor`"
                ):
                    dual.as_strided((5,), (1,), 0)
            return unpacked_tangent

        torch._vmap_internals._vmap(jvp, 0, 0)(tangent)


class TestAutogradForwardMode(TestCase):
    # [CLASS DOC] Tests of forward mode automatic differentiation through the dual number API:
    # level management, make and unpack dual, metadata checks, view and in-place handling,
    # cleanup, and error cases.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Exit any dual levels left open by a failing test, then run the
        # base teardown.
        # Ensure that a failing test won't make others fail
        while fwAD._current_level >= 0:
            fwAD.exit_dual_level()

        super().tearDown()

    def test_forward_level_cleanup(self):
        # [FUNCTION DOC] Check dual level cleanup: tangent reused as is, kept alive by the dual,
        # and the weak reference expires when the dual is deleted.
        def get_tensor_and_weak_ref():
            # [FUNCTION DOC] HELPER: Make a tensor and a weak reference to it.
            # Create a new Tensor and weak reference
            t = torch.rand(2, requires_grad=True)
            return t, torch._C._WeakTensorRef(t)

        # Sanity check that the helper function works as expected
        t, t_ref = get_tensor_and_weak_ref()
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        # Main test code
        foo = torch.rand(2)

        with fwAD.dual_level():
            tangent, tangent_ref = get_tensor_and_weak_ref()
            self.assertFalse(tangent_ref.expired())

            dual = fwAD.make_dual(foo, tangent)
            self.assertFalse(tangent_ref.expired())

            # Make sure that the tangent we provided has been reused as is
            self.assertTrue(fwAD.unpack_dual(dual)[1] is tangent)

            # Make sure that dual is keeping the tangent alive
            del tangent
            self.assertFalse(tangent_ref.expired())

            # Make sure that the dual level does not keep the c++
            # version of the tangent alive
            del dual
            self.assertTrue(tangent_ref.expired())

    def test_size_check(self):
        # [FUNCTION DOC] Check make_dual raises RuntimeError when tangent size differs from the
        # primal, and works for a matching size.
        foo = torch.rand(2)
        tangent = torch.rand(3)

        with fwAD.dual_level():
            with self.assertRaisesRegex(
                RuntimeError,
                "Trying to set a forward gradient that has a different size",
            ):
                dual = fwAD.make_dual(foo, tangent)

            dual = fwAD.make_dual(foo, tangent[1:])

    def test_metadata_check_checks_storage_numel(self):
        # [FUNCTION DOC] Check metadata check considers storage size: mutating the unpacked
        # tangent does not change the original, and as_strided over the full storage works.
        primal = torch.randn(5)[:4].detach()
        self.assertEqual(len(primal.storage()), 5)
        tangent = torch.randn(4)

        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            _, unpacked_tangent = fwAD.unpack_dual(dual)

            # # Verify that mutating unpacked tangent does not affect the original tangent
            tangent_clone = tangent.clone()
            unpacked_tangent *= 2
            self.assertTrue(torch.allclose(tangent_clone, tangent))

            # as_strided runs without error
            dual.as_strided((5,), (1,), 0)

    def test_metadata_check_checks_ignores_size_zero(self):
        # [FUNCTION DOC] Check metadata check ignores strides of zero-size tensors, including
        # gradcheck with forward AD on an empty complex input.
        a = torch.ones(0).as_strided((0, 1), (1, 1), 0)
        b = torch.ones(0).as_strided((0, 1), (1, 0), 0)

        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            torch.diagonal(dual, offset=0)

        input = torch.rand([0, 1], dtype=torch.complex128, requires_grad=True)
        func = partial(torch.diagonal, offset=0)
        torch.autograd.gradcheck(func, (input,), check_forward_ad=True)

    def test_metadata_check_when_primal_has_conj_bit(self):
        # [FUNCTION DOC] Check the storage-size check does not materialize the conjugate bit when
        # the primal is conjugated.
        # Make sure the _has_same_storage_numel is a fallthrough, so that
        # conj bit does not materialize. If it materializes it would
        # cause the layout check to fail for views that do not index the
        # the entire storage.
        a = torch.randn(2, 2, dtype=torch.cdouble).conj()
        b = torch.rand_like(a)

        self.assertTrue(torch.is_conj(a))
        self.assertEqual(len(a.storage()), len(b.storage()))

        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_when_primal_has_neg_bit(self):
        # [FUNCTION DOC] Check the storage-size check does not materialize the negation bit when
        # the primal has it.
        # Make sure the _has_same_storage_numel is a fallthrough, so that
        # conj bit does not materialize. If it materializes it would
        # cause the layout check to fail for views that do not index the
        # the entire storage.
        a = torch.randn(2, 2, dtype=torch.cdouble).conj().imag
        b = torch.randn(2, 2, dtype=torch.cdouble).imag

        self.assertTrue(torch.is_neg(a))
        self.assertEqual(len(a.storage()), len(b.storage()))

        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_check_conj(self):
        # [FUNCTION DOC] Check make_dual keeps the tangent as is only when primal and tangent have
        # the same conj or neg view kind, and that real and imag ops work on the duals.
        keys = {
            "NEITHER": lambda x: x,
            "CONJ": lambda x: x.conj(),
            "NEG": lambda x: x._neg_view(),
        }

        for primal_key, tangent_key in product(keys, keys):
            x = keys[primal_key](torch.randn(2, 3, 4, dtype=torch.cdouble))
            t = keys[tangent_key](torch.randn(2, 3, 4, dtype=torch.cdouble))

            if primal_key == tangent_key:
                with fwAD.dual_level():
                    dual = fwAD.make_dual(x, t)
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is t)
                    torch.real(dual)
                    torch.imag(dual)
            else:
                with fwAD.dual_level():
                    dual = fwAD.make_dual(x, t)
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is not t)
                    torch.real(dual)
                    torch.imag(dual)

    def test_metadata_check_ignore_storage_offset_for_zero_numel_tensor(self):
        # [FUNCTION DOC] Check metadata check ignores storage offset for zero-element tensors and
        # still works for views of nonzero ones.
        # See https://github.com/pytorch/pytorch/issues/80507
        a = torch.tensor([1.0]).as_strided((0,), (1,), 1)
        b = torch.tensor([1.0]).as_strided((0,), (1,), 2)

        with fwAD.dual_level():
            dual_input = fwAD.make_dual(a, b)
            # Check that no copy is made
            self.assertIs(fwAD.unpack_dual(dual_input).tangent, b)

        a = torch.tensor([1.0]).as_strided((1,), (2,), 0)
        b = torch.tensor([1.0]).as_strided((1,), (1,), 0)

        with fwAD.dual_level():
            dual_input = fwAD.make_dual(a, b)
            dual_input[1:]

    # The following test functions want to ensure all the following behaviors:
    #   - Ensure that default level system in the python binding works
    #   - Ensure that only level 0 exists and nesting is properly disabled
    #   - Ensure that printing works fine
    #   - Ensure that basic packing/unpacking works
    #   - Ensure that advanced packing/unpacking works
    #     - For memory / version counter share
    #     - For backward AD (regular ops)
    #   - Ensure that view + inplace for both modes work fine
    #   - Ensure we do proper cleanup on exit of a level

    def test_default_level(self):
        # [FUNCTION DOC] Check dual level default: the tangent is returned as is within the level
        # and is gone after the level exits.
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        # We don't actually need to enforce that these two are the exact same python
        # object, feel free to relax in the future
        self.assertIs(baz_tangent, bar)

        baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertEqual(baz_tangent, None)

    def test_fwd_grad_enabled(self):
        # [FUNCTION DOC] Check the private flag controlling forward grad mode can be switched off
        # and on again.
        # Tests some private helper functions to enable/disable fwd grad mode
        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)

        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)

        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)

    def test_set_fwd_grad_enabled(self):
        # [FUNCTION DOC] Check the private context manager for forward grad mode temporarily
        # enables it and restores the disabled state.
        # Tests a private helper function
        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)

            with fwAD._set_fwd_grad_enabled(True):
                enabled = fwAD._is_fwd_grad_enabled()
                self.assertTrue(enabled)

            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)

    def test_nested_level(self):
        # [FUNCTION DOC] Check only level zero exists and entering a nested dual level
        # raises RuntimeError.
        with fwAD.dual_level() as level:
            # For now only level 0 exists
            self.assertEqual(level, 0)

        with fwAD.dual_level():
            with self.assertRaisesRegex(
                RuntimeError, "Nested forward mode AD is not supported at the moment"
            ):
                nest_level = fwAD.enter_dual_level()

    def test_set_fw_grad_having_own_fw_grad_at_same_level(self):
        # [FUNCTION DOC] Check making a dual from a tensor that already has a forward gradient at
        # the same level raises RuntimeError.
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            with self.assertRaisesRegex(
                RuntimeError, "has a forward gradient at the same level"
            ):
                fwAD.make_dual(baz, dual)

    def test_codegen_ignores_undefined_outputs(self):
        # [FUNCTION DOC] Check codegen ignores undefined outputs: a convolution backward with a
        # dual weight and no requested input grad returns None for it.
        # This test checks that codegen silently ignores undefined outputs
        # Below, grad_input is specified as False in grad_output_mask, so
        # convolution backward will return a undefined tensor in that position.
        # Note that for this test to work we need to make sure either grad_output
        # or weight to be a dual tensor, so grad_input requires forward grad
        weight = torch.randn(6, 1, 30, 30)
        inp = torch.rand((1, 1, 32, 32))
        out = torch.nn.functional.conv2d(inp, weight)
        grad_out = torch.ones_like(out)

        with fwAD.dual_level():
            dual_weight = fwAD.make_dual(weight, torch.ones_like(weight))
            grad_input, _, _ = torch.ops.aten.convolution_backward(
                grad_out,
                inp,
                dual_weight,
                (0,),
                (1, 1),
                (0, 0),
                (1, 1),
                False,
                (0, 0),
                1,
                (False, True, False),
            )
        self.assertIsNone(grad_input)

    def test_make_dual_inference_tensor_in_inference_mode(self):
        # [FUNCTION DOC] Check make_dual in inference mode: the dual is not a view and in-place
        # changes modify the primal.
        with torch.inference_mode():
            foo = torch.rand(2)
            bar = torch.rand(2)
            foo_copy = foo.clone()

            with fwAD.dual_level():
                dual = fwAD.make_dual(foo, bar)
                self.assertFalse(dual._is_view())

                dual += 1
                self.assertFalse(torch.allclose(foo, foo_copy))

    def test_make_dual_torch_dispatch(self):
        # [FUNCTION DOC] Check make_dual on a dispatch tensor subclass calls alias only on the
        # primal and allows re-enabling autograd inside dispatch.
        counter = [0]

        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass counting alias calls in its dispatch hook.
            def __new__(cls, data=None):
                # [FUNCTION DOC] HELPER: Build the subclass from existing data.
                return torch.Tensor._make_subclass(cls, data)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: On alias count the call and check autograd can be
                # re-enabled; then run the op with dispatch disabled.
                if func.overloadpacket == torch.ops.aten.alias:
                    counter[0] += 1

                    # Make sure we can re-enable autograd here
                    with torch.overrides.enable_reentrant_dispatch():
                        foo = torch.rand(1, requires_grad=True)
                        self.assertIsNotNone(foo.exp().grad_fn)

                with no_dispatch():
                    return func(*args, **kwargs)

        a = torch.tensor(1.0)
        s = MySubclass(a)

        with fwAD.dual_level():
            # Only the primal has "alias" called on it
            fwAD.make_dual(s, torch.rand_like(s))
            self.assertEqual(counter[0], 1)
            fwAD.make_dual(torch.rand_like(s), s)
            self.assertEqual(counter[0], 1)

    def test_make_dual_forbid_integral_dtype(self):
        # [FUNCTION DOC] Check make_dual raises ValueError for integral primal or tangent dtypes,
        # with distinct messages.
        primal_f = torch.ones(2, 2, dtype=torch.float)
        primal_l = torch.ones(2, 2, dtype=torch.long)

        tangent_f = torch.ones(2, 2, dtype=torch.float)
        tangent_l = torch.ones(2, 2, dtype=torch.long)

        with fwAD.dual_level():
            # Float Primal and Long Tangent
            with self.assertRaisesRegex(
                ValueError, "Expected tangent to be floating point or complex"
            ):
                fwAD.make_dual(primal_f, tangent_l)

            # Long Primal and Long Tangent
            with self.assertRaisesRegex(
                ValueError, "Expected primal to be floating point or complex"
            ):
                fwAD.make_dual(primal_l, tangent_l)

            # Long Primal and Float Tangent
            with self.assertRaisesRegex(
                ValueError, "Expected primal to be floating point or complex"
            ):
                fwAD.make_dual(primal_l, tangent_f)

    def test_print(self):
        # [FUNCTION DOC] Check the string form of a dual shows a tangent while plain tensors and
        # unpacked parts do not.
        with fwAD.dual_level() as level:
            a = torch.rand(3)
            self.assertFalse("tangent=" in str(a))

            b = fwAD.make_dual(a, torch.rand(3))
            self.assertFalse("tangent=" in str(a))
            self.assertTrue("tangent=" in str(b))

            b_primal, b_tangent = fwAD.unpack_dual(b)
            self.assertFalse("tangent=" in str(b_primal))
            self.assertFalse("tangent=" in str(b_tangent))

    def test_basic_packing_unpacking(self):
        # [FUNCTION DOC] Check make_dual and unpack_dual round trip, the named tuple fields, new
        # views per unpack, and that inputs are unchanged.
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
            self.assertEqual(baz_primal, foo)
            self.assertIs(baz_tangent, bar)

            # Check unpacked dual is returned as a named tuple
            # NB: Every invocation of unpack_dual returns a new tensor view
            self.assertIsNot(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertEqual(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertIs(baz_tangent, fwAD.unpack_dual(baz).tangent)

            # Check that packing/unpacking did not change the input
            foo_primal, foo_tangent = fwAD.unpack_dual(foo)
            self.assertEqual(foo_primal, foo)
            self.assertIsNone(foo_tangent)

    def test_advanced_packing_unpacking(self):
        # [FUNCTION DOC] Check unpacking shares memory and version counters, backward gradients
        # flow through make and unpack dual, and detach, no_grad and in-place detach effects on
        # forward gradients.
        foo = torch.rand(2)
        bar = torch.ones(2)

        # Memory and version counter check
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)

            # Ensure that they are sharing memory and version counter
            self.assertEqual(dual.storage().data_ptr(), foo.storage().data_ptr())

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual._version)

            # Unpacking should only create aliases as well
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertEqual(dual_primal.storage().data_ptr(), foo.storage().data_ptr())
            self.assertEqual(
                dual_tangent.storage().data_ptr(), bar.storage().data_ptr()
            )
            # And the tangent is actually reused as-is so it is still the same Tensor
            self.assertIs(dual_tangent, bar)

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual_primal._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual_primal._version)
            self.assertEqual(bar._version, dual_tangent._version)
            bar.add_(1)
            self.assertEqual(bar._version, dual_tangent._version)

        # backward mode check
        with fwAD.dual_level():
            foo.requires_grad_()
            bar.requires_grad_()

            # Check that backward gradients properly propagates through packing/unpacking
            dual = fwAD.make_dual(foo, bar)
            p, t = fwAD.unpack_dual(dual)

            gfoo, gbar = torch.autograd.grad(
                p.sum(), (foo, bar), retain_graph=True, allow_unused=True
            )
            self.assertEqual(gfoo, torch.ones_like(foo))
            self.assertIsNone(gbar)

            gfoo, gbar = torch.autograd.grad(
                t.sum(), (foo, bar), retain_graph=True, allow_unused=True
            )
            self.assertIsNone(gfoo)
            self.assertEqual(gbar, torch.ones_like(bar))

            # Check that forward gradients are impacted by detach()
            detached_dual = dual.detach()
            out = detached_dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

            # Check that forward gradients are not impacted by no_grad
            with torch.no_grad():
                out = dual * 3
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 3)
            self.assertEqual(t, bar * 3)

            # Check that forward gradients are not impacted by inplace detach
            dual = dual.clone()
            dual.detach_()
            out = dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

    def test_view_inplace_non_differentiable_views(self):
        # [FUNCTION DOC] Check in-place ops on duals update primal and tangent but not
        # non-differentiable views of them, and in-place ops on unpacked primal or tangent do not
        # alter the other.
        original_foo = torch.rand(2, dtype=torch.double)
        original_bar = torch.ones(2, dtype=torch.double)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Note that in this test, we use "update" to mean computing the right tangent for the dual
            # All the inplace operations here are expected to update the primal value of the Tensors but
            # not always their tangents.
            # Also all mentions of "non differentiable view" here means non forward differentiable view
            # unless specified otherwise.
            # See note [Forward Grad View/inplace] for more details on how these views work.

            # Check that inplace ops do not update non-differentiable views
            # Non differentiable view
            dual = fwAD.make_dual(foo, bar)
            dual *= 2
            # Check that non differentiable view's tangent was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that the computed result is correct
            self.assertEqual(bar, original_bar * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            self.assertEqual(foo, original_foo * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 2)
            # Other non differentiable view
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertIsNone(fwAD.unpack_dual(dual_primal)[1])
            self.assertIsNone(fwAD.unpack_dual(dual_tangent)[1])
            dual_primal *= 2
            # Ensure dual's tangent did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            dual_tangent *= 2
            # Ensure dual's primal did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 4)

    def test_view_inplace_differentiable_views(self):
        # [FUNCTION DOC] Check in-place ops on differentiable views of duals update the dual's
        # tangent but stop at non-differentiable views, including non-dual tensors updated
        # from duals.
        original_foo = torch.rand(2)
        original_bar = torch.ones(2)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Check that inplace ops do update differentiable view but stop at non differentiable ones
            # A non differentiable view
            dual = fwAD.make_dual(foo, bar)
            # A differentiable view
            view = dual.narrow(0, 0, 1)
            view *= 2
            # Check that non differentiable view was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that differentiable view was updated
            self.assertEqual(fwAD.unpack_dual(dual)[1], torch.tensor([2.0, 1.0]))
            self.assertEqual(fwAD.unpack_dual(view)[1], torch.tensor([2.0]))

            # Check that we track differentiable view even for Tensors that are not dual
            baz = torch.rand(2)
            baz += dual
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])
            # Updates on view should as well
            baz = torch.rand(2)
            baz[0] = dual[0]
            self.assertEqual(fwAD.unpack_dual(baz)[1][0], fwAD.unpack_dual(dual)[1][0])
            # Unused values get a gradient of 0
            self.assertEqual(fwAD.unpack_dual(baz)[1][1], 0.0)

            # Check that forward non-differentiable views do prevent gradient update
            baz = torch.rand(2)
            view = baz.detach()
            view += dual
            self.assertIsNone(fwAD.unpack_dual(baz)[1])

    def test_view_inplace_always_creates_a_view(self):
        # [FUNCTION DOC] Check in-place binary ops on a transposed view with a dual operand
        # create a view relationship for both primal and tangent in the base (add, mul and
        # copy variants).
        # See https://github.com/pytorch/pytorch/issues/67800
        # The codepath may depend on the op. At the time writing, when self is not a dual tensor
        # the resulting forward grad for self for...
        # - add_ has the same layout as self
        # - mul_ has the same layout as other
        # This is kind of fragile because the above depends on how the forward grad expression
        # is written. For add and mul at least, the output inherits the layout of LHS.
        # We want to handle at least these two cases.
        inplace_binary_ops = (  # Add more to this list?
            lambda x, y: x.add_(y),
            lambda x, y: x.mul_(y),
            lambda x, y: x.copy_(y),
        )

        for inplace_binary_op in inplace_binary_ops:
            base = torch.randn(2, 2)
            view = base.transpose(0, 1)

            primal = torch.randn(2, 2)
            tangent = torch.randn(2, 2)

            with fwAD.dual_level():
                dual = fwAD.make_dual(primal, tangent)
                inplace_binary_op(view, dual)

                # Verify that a view relationship is created for both the primal and tangent
                p, t = fwAD.unpack_dual(base)
                p_clone = p.clone()
                t_clone = t.clone()
                view *= 2
                p, t = fwAD.unpack_dual(base)

                self.assertTrue(torch.allclose(p_clone * 2, p))
                self.assertTrue(torch.allclose(t_clone * 2, t))

    def test_grad_cleanup(self):
        # [FUNCTION DOC] Check tangents are dropped when the level exits and a new level starts
        # without old tangents, while new duals work.
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertIs(fwAD.unpack_dual(dual)[1], bar)

        self.assertIsNone(fwAD.unpack_dual(dual)[1])

        with fwAD.dual_level():
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            new_dual = fwAD.make_dual(foo, baz)

            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            new_dual_primal, new_dual_tangent = fwAD.unpack_dual(new_dual)
            self.assertEqual(dual_primal, new_dual_primal)
            self.assertIsNone(dual_tangent)
            self.assertEqual(new_dual_tangent, baz)

    def test_detach_view_tracking(self):
        # [FUNCTION DOC] Check detach produces a tensor that is not kept alive by a view
        # relationship (weak reference expires after deleting the original).
        # Default detach is both forward and backward non-differentiable
        foo = torch.rand(2)
        foo_weak = torch._C._WeakTensorRef(foo)

        out = foo.detach()

        del foo
        self.assertTrue(foo_weak.expired())

    def test_out_variant(self):
        # [FUNCTION DOC] Check out variants with a dual input or output raise RuntimeError
        # mentioning out.
        with fwAD.dual_level():
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)

            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(bar, bar, out=foo)

            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(foo, bar, out=bar)

    def test_non_differentiable(self):
        # [FUNCTION DOC] Check comparison ops (not differentiable) on duals do not error,
        # including in-place.
        with fwAD.dual_level():
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)

            # No differentiable outputs, shouldn't error
            eq = foo == bar

            # Inplace
            foo.eq_(bar)

    def test_create_new_zeros_with_same_meta(self):
        # [FUNCTION DOC] Check the private zero-creation op matches feature sizes, strides,
        # storage offset, storage size and dtype of a target tensor for several layouts and
        # batch dimensions.
        new_zeroes_fn = torch.ops.aten._new_zeros_with_same_feature_meta

        def check(a, b):
            # [FUNCTION DOC] HELPER: Run the metadata assertion in both directions for a pair
            # of tensors.
            def assert_same_meta(t, target):
                # [FUNCTION DOC] HELPER: For each number of batch dims, create zeros like the
                # target and assert dims, strides of feature dims, contiguity, storage offset,
                # storage length scaling and dtype.
                for num_bdim in range(t.dim()):
                    result = new_zeroes_fn(t, target, self_num_batch_dims=num_bdim)

                    self.assertEqual(result.dim(), target.dim() + num_bdim)

                    # Check size/strides match for feature dims only
                    for i in range(num_bdim, result.dim()):
                        self.assertEqual(result.size()[i], target.size()[i - num_bdim])
                        self.assertEqual(
                            result.stride()[i], target.stride()[i - num_bdim]
                        )

                    # Check that we generate strides reasonably
                    if target.is_contiguous():
                        self.assertTrue(result.is_contiguous())

                    self.assertEqual(result.storage_offset(), target.storage_offset())

                    prod_of_t_bdims = reduce(operator.mul, t.size()[:num_bdim], 1)
                    self.assertEqual(
                        len(result.storage()), len(target.storage()) * prod_of_t_bdims
                    )

                    # TensorOptions is same
                    self.assertEqual(result.dtype, target.dtype)

            assert_same_meta(a, b)
            assert_same_meta(b, a)

        a = torch.randn(5, dtype=torch.float)
        b = torch.randn(2, 3, 4, dtype=torch.double)
        check(a, b)

        # non-contiguous case
        a = torch.randn(2, 3, 4).transpose(0, 1).contiguous().transpose(0, 1)
        b = torch.randn(2, 3, 4)
        check(a, b)

        a = torch.randn(5).narrow(0, 1, 2)
        b = torch.randn(2)
        check(a, b)

        # tensor is not a view, but still does not index entirety of storage
        a = torch.randn(5).resize_(4)
        b = torch.randn(4)
        check(a, b)

        # Zero-numel tensors
        a = torch.randn(1, 0, 2)
        b = torch.randn(1, 2)
        check(a, b)

        # Scalar tensor
        a = torch.tensor(1.0)
        b = torch.randn(1, 2)
        check(a, b)

    def test_backward_graph_destruction(self):
        # [FUNCTION DOC] Check destroying a graph with a C++ reference cycle involving forward
        # grads inside a dual level does not deadlock.
        def fn():
            # [FUNCTION DOC] HELPER: Build a dual with a grad-requiring tangent and an exp to
            # create the cycle.
            a = torch.rand(10, requires_grad=True)

            da = fwAD.make_dual(torch.rand_like(a), a)

            # Create an object with a c++ cycle as:
            # db -> AutogradMeta -> ForwardGrad -> db's grad
            # db's grad -> AutogradMeta -> MulBackward
            # MulBackward -> SavedVariable -> db
            db = da.exp()

        with fwAD.dual_level():
            fn()
        # This test make sure that we don't deadlock on exit of this
        # context manager. If you do, there is something wrong with the
        # locking of the forward ad level most likely


# Generic device type autograd tests.
class TestAutogradDeviceType(TestCase):
    # [CLASS DOC] Device-generic autograd tests, instantiated per device by the device type test
    # framework; each test receives the device string.
    def test_min_max_aminmax_median_backprops_to_all_values(self, device):
        # [FUNCTION DOC] Check min, max, median, nanmedian, amin, amax and aminmax split
        # gradient evenly among tied values, also for all-NaN input and when both aminmax
        # gradients are active.
        # 1) Test min/max/median/nanmedian on both a non NaN and all NaN tensor
        for f in [torch.min, torch.max, torch.median, torch.nanmedian]:
            with self.subTest(f=f):
                x1 = torch.tensor(
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=device, requires_grad=True
                )
                x2 = torch.tensor(
                    [float("nan"), float("nan"), float("nan")],
                    device=device,
                    requires_grad=True,
                )
                for x in [x1, x2]:
                    y = f(x)
                    y.backward()
                    self.assertEqual(x.grad.sum(), 1.0)
                    self.assertEqual((x.grad == 1 / 3).sum(), 3)

        # 2) Explicit amin/amax plus the two components of aminmax
        def amin2(x):
            # [FUNCTION DOC] HELPER: Minimum component of aminmax.
            return torch.aminmax(x)[0]  # min part

        def amax2(x):
            # [FUNCTION DOC] HELPER: Maximum component of aminmax.
            return torch.aminmax(x)[1]

        for f in [torch.amin, torch.amax, amax2, amin2]:
            with self.subTest(f=f):
                x1 = torch.tensor(
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    device=device,
                    requires_grad=True,
                )
                y = f(x1)
                y.backward()
                self.assertEqual(x1.grad.sum(), 1.0)
                self.assertEqual((x1.grad == 1.0 / 3.0).sum(), 3)

        # 3) Both min and max grads active simultaneously — exercises the add_ path
        # in aminmax_backward when both grad_min and grad_max are defined.
        # min ties at indices 1,3 → each gets 0.5; max ties at indices 0,2 → each gets 0.5
        with self.subTest("aminmax_both_grads"):
            x = torch.tensor([3.0, 1.0, 3.0, 1.0], device=device, requires_grad=True)
            min_val, max_val = torch.aminmax(x)
            (min_val + max_val).backward()
            self.assertEqual(x.grad, torch.tensor([0.5, 0.5, 0.5, 0.5], device=device))

    def test_scatter_index_reduce_amin_amax_aminmax_backprops_to_all_values(
        self, device
    ):
        # [FUNCTION DOC] Check scatter_reduce and index_reduce with amin or amax pass gradcheck
        # with tied values.
        # tests that gradients are evenly distributed when there are multiple max/min values
        # tested here instead of adding a SampleInput as the backward for this case is non-differentiable for gradgrad
        # as is the case for test_min_max_aminmax_median_backprops_to_all_values above
        fns = (torch.scatter_reduce, torch.index_reduce)
        reduces = ("amin", "amax")
        for fn, reduction in product(fns, reduces):
            input = torch.randn(
                (2, 3), device=device, dtype=torch.float64, requires_grad=True
            )
            src = input.clone().detach_().requires_grad_(True)
            idx = torch.arange(2).to(dtype=torch.long, device=device)
            if fn == torch.scatter_reduce:
                idx = idx.unsqueeze(-1).expand((2, 3))

            gradcheck(fn, (input, 0, idx, src, reduction), check_batched_grad=False)

    def test_scatter_index_reduce_prod_gradgrad_error(self, device):
        # [FUNCTION DOC] Check gradcheck of scatter_reduce and index_reduce with prod passes for
        # duplicate zero sources, while gradgradcheck raises RuntimeError about unsupported
        # double backward.
        # test that double backward raises an error for the case where 2 zeros in src
        # are scattered to the same position in self
        input = torch.tensor(
            [1.0], device=device, dtype=torch.float64, requires_grad=True
        )
        src = torch.tensor(
            [0.0, 0.0], device=device, dtype=torch.float64, requires_grad=True
        )
        idx = torch.tensor([0, 0], device=device, dtype=torch.long)

        for fn in (torch.scatter_reduce, torch.index_reduce):
            # check that this case passes on gradcheck
            gradcheck(fn, (input, 0, idx, src, "prod"), check_batched_grad=False)
            with self.assertRaisesRegex(
                RuntimeError, "Double backward is unsupported for"
            ):
                gradgradcheck(fn, (input, 0, idx, src, "prod"))

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_parameter_resize(self, device):
        # [FUNCTION DOC] Check backward after in-place resizing of a Parameter with set_ twice and
        # concatenation of it with itself. Skipped on MPS.
        asd = torch.nn.Parameter(torch.ones(16, dtype=torch.double, device=device))

        for _ in range(2):
            with torch.no_grad():
                asd.set_(asd[1:])
                asd.grad = None

            m = torch.cat((asd, asd))
            m.sum().backward()

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_ctor_getter_backward(self, device, dtype):
        # [FUNCTION DOC] Check backward through sparse constructor and getters for dense/empty
        # combinations, and that values of a detached sparse tensor cannot be backwarded. Skipped
        # on MPS; for double and complex double.
        # See NOTE [ Sparse: autograd and API ] on the expected behavior of this test
        def _test(size, sparse_dim, nnz, device):
            # [FUNCTION DOC] HELPER: Build a sparse tensor from indices and values, add another
            # sparse tensor, coalesce, apply tanh to values and rebuild; run gradcheck and check
            # that raw values have no grad_fn.
            v_size = [nnz] + list(size[sparse_dim:])
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)

            inp = torch.randn(
                v_size, dtype=torch.double, device=device, requires_grad=True
            )
            other = self.genSparseTensor(
                size, sparse_dim, nnz, is_uncoalesced=True, device=device, dtype=dtype
            )[0]

            def fn(v):
                # [FUNCTION DOC] HELPER: Sparse tensor construction from values, addition,
                # coalesce, tanh of values and reconstruction, returning values.
                x = torch.sparse_coo_tensor(i, v, size, dtype=dtype, device=device)
                y = (x + other).coalesce()
                yv = y.values()
                new_v = yv.tanh()
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                return z.coalesce().values()

            gradcheck(fn, (inp,), check_batched_grad=False)
            # FIXME: make gradgradcheck work.
            # gradgradcheck(fn, (inp,), check_batched_grad=False)

            # assert that _values is non-differentiable
            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.detach().requires_grad_()._values().backward(
                    torch.ones_like(other._values())
                )

        for empty_i, empty_v, empty_nnz in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            _test(sparse_size + dense_size, len(sparse_size), nnz, device)

    @skipIfMPS
    @skipMeta
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_backward(self, device, dtype):
        # [FUNCTION DOC] Check accumulation of sparse and dense gradients in different orders
        # through a Function returning fixed gradients. Skipped on MPS and meta; double and
        # complex double.
        class FixedGradientFunction(Function):
            # [CLASS DOC] HELPER: Function whose backward returns a gradient given in forward.
            @staticmethod
            def forward(ctx, x, grad_x):
                # [FUNCTION DOC] HELPER: Save the given gradient and return the input.
                ctx.save_for_backward(grad_x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                # [FUNCTION DOC] HELPER: Return the saved gradient and None.
                (saved_grad_x,) = ctx.saved_tensors
                return saved_grad_x, None

        size = torch.Size([6, 3, 2])
        i1 = torch.tensor([[0, 3, 4], [0, 2, 2]], dtype=torch.long)
        v1 = make_tensor([3, 2], dtype=dtype, device=device)
        sparse_grad1 = torch.sparse_coo_tensor(i1, v1, size, dtype=dtype, device=device)
        i2 = torch.tensor([[0, 1, 3, 4], [0, 1, 2, 2]], dtype=torch.long)
        v2 = make_tensor([4, 2], dtype=dtype, device=device)
        sparse_grad2 = torch.sparse_coo_tensor(i2, v2, size, dtype=dtype, device=device)
        dense_grad = torch.rand(size, device=device, dtype=dtype)
        fn = FixedGradientFunction

        # sparse first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (
            fn.apply(x, sparse_grad1)
            + fn.apply(x, dense_grad)
            + fn.apply(x, sparse_grad2)
        ).sum().abs().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # dense first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (
            fn.apply(x, dense_grad)
            + fn.apply(x, sparse_grad1)
            + fn.apply(x, sparse_grad2)
        ).sum().abs().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # sparse only
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().abs().backward()
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    def test_sparse_mask_autograd(self, device):
        # [FUNCTION DOC] Check backward through sparse_mask followed by to_dense gives the dense
        # mask as gradient.
        tensor = torch.randn(3, requires_grad=True, device=device)
        mask = torch.ones(3, device=device)
        mask[1] = 0
        mask = mask.to_sparse()
        converted = tensor.sparse_mask(mask).to_dense()
        converted.sum().backward()
        self.assertEqual(tensor.grad, mask.to_dense())

    def test_pyscalar_conversions(self, device):
        # [FUNCTION DOC] Check conversions between Python scalars and tensors: integral and
        # floating types, special float values, precision loss, errors for NaN and infinity
        # converted to integers, and truthiness.
        def _test_pyscalar_conversions(t, integral_conv):
            # [FUNCTION DOC] HELPER: Run the conversion checks with a tensor type function and an
            # integer conversion function.
            # integral -> integral
            l = t(torch.zeros(1, 1, 1, dtype=torch.long))
            pyscalar = -12345
            l[0] = pyscalar
            self.assertEqual(integral_conv(l), pyscalar)

            # floating point -> floating point
            f = Variable(t(torch.randn(1, 1, dtype=torch.double)))
            pyscalar = -12345.1
            f[0] = pyscalar
            self.assertEqual(float(f), pyscalar)
            f[0] = nan
            self.assertTrue(math.isnan(float(f)))
            f[0] = inf
            self.assertEqual(float(f), inf)
            f[0] = -inf
            self.assertEqual(float(f), -inf)

            # integral -> floating point
            # check we can convert something that loses precision
            pyscalar = 1234567890123456789
            self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
            l[0] = pyscalar
            self.assertEqual(float(l), float(pyscalar))

            # floating point -> integral
            f[0] = nan
            self.assertRaises(ValueError, lambda: integral_conv(f[0]))
            f[0] = inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = -inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = sys.float_info.max
            self.assertEqual(integral_conv(f), sys.float_info.max)

            # bool, nonzero
            def test_nonzero(tensor, value, expected):
                # [FUNCTION DOC] HELPER: Assign a value into the tensor and assert its truth value
                # twice equals the expectation.
                tensor[0] = value
                self.assertEqual(expected, bool(tensor))
                self.assertEqual(expected, bool(tensor))

            test_nonzero(l, 0, False)
            test_nonzero(l, -2, True)
            test_nonzero(f, 0.0, False)
            test_nonzero(f, sys.float_info.min, True)
            test_nonzero(f, nan, bool(nan))
            test_nonzero(f, inf, bool(inf))
            test_nonzero(f, -inf, bool(-inf))

        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    @dtypesIfMPS(torch.float32)
    @dtypesIfCUDA(
        torch.half,
        torch.float,
        torch.double,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    @dtypes(
        torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64
    )
    def test_set_requires_grad_only_for_floats(self, device, dtype):
        # [FUNCTION DOC] Check requires_grad can be set (by method, attribute or constructor) on
        # floating dtypes but raises RuntimeError for integer dtypes; turning it off always works.
        # Parametrized by device-specific dtypes.
        def f1():
            # [FUNCTION DOC] HELPER: Set requires_grad using the in-place method.
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad_()

        def f2():
            # [FUNCTION DOC] HELPER: Set requires_grad using attribute assignment.
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad = True

        def f3():
            # [FUNCTION DOC] HELPER: Create a tensor with requires_grad in the constructor.
            torch.ones(1, dtype=dtype, device=device, requires_grad=True)

        a = torch.ones(1, dtype=dtype, device=device)
        a.requires_grad = False  # should always work
        a.requires_grad_(False)

        for f in [f1, f2, f3]:
            if dtype.is_floating_point:
                f()
            else:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "floating point",
                    msg=f"dt: {a.dtype} device: {a.device}",
                ):
                    f()

    @onlyCUDA
    def test_advanced_indexing_backwards_large(self, device):
        # [FUNCTION DOC] Check backward of advanced indexing on a large first dimension.
        # CUDA only.
        # See https://github.com/pytorch/pytorch/issues/22843
        n = 1 << 16
        x = torch.rand(n, 1, device=device, requires_grad=True)
        a = x[:, [0]]
        a.sum().backward()
        self.assertEqual(x.grad, torch.ones(n, 1, device=device))

    def test_advanced_indexing_backwards_memory_format(self, device):
        # [FUNCTION DOC] Check backward of indexing with a channels-last index tensor works.
        # See https://github.com/pytorch/pytorch/issues/36956
        shape = (2, 8, 1, 2)
        i = torch.randint(1, shape, device=device).contiguous(
            memory_format=torch.channels_last
        )
        x = torch.randn(shape, requires_grad=True, device=device)
        x[i].sum().backward()

    def _test_reentrant_parent_error_on_cpu(self, device):
        # [FUNCTION DOC] HELPER: Build a CPU parent graph that errors in backward and a longer
        # reentrant device child graph, run backward on both, and check the error is raised and no
        # grads are accumulated.
        t1 = torch.rand([3, 3], requires_grad=True)
        t2 = torch.rand([3, 3], device=device, requires_grad=True)
        t3 = torch.rand([3, 3], device=device, requires_grad=True)

        # Parent graph cpu graph.
        t4 = t1 * t1
        t5 = TestAutograd.SimulateBackwardError.apply(t4)

        # Child gpu graph (much longer than parent graph).
        prev = t2 * t2
        for _ in range(10):
            prev = prev * t2
        reentrant_root = prev

        class ReentrantFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward runs a long reentrant backward.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Return a clone.
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Run backward on the long child graph and return
                # the gradient.
                # Reentrant backward in child will take much longer.
                reentrant_root.backward(grad)
                return grad

        # Parent gpu graph.
        t6 = ReentrantFunc.apply(t3)
        t7 = t6 * t6

        # Parent graph will error out first, while child graph will continue executing.
        with self.assertRaisesRegex(Exception, "Simulate error"):
            torch.autograd.backward([t5.sum(), t7.sum()])

        # No grads should be accumulated since child graph will stop execution
        # after parent receives error.
        self.assertIsNone(t2.grad)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t3.grad)

    @onlyCUDA
    def test_reentrant_parent_error_on_cpu(self, device):
        # [FUNCTION DOC] Check the parent error with a reentrant child on a device frees device
        # memory afterward: memory usage returns to its previous value within thirty seconds.
        # CUDA only.
        def _get_cuda_memory_usage():
            # [FUNCTION DOC] HELPER: Collect garbage and return the allocated memory per
            # CUDA device.
            # we don't need CUDA synchronize because the statistics are not tracked at
            # actual freeing, but at when marking the block as free.
            num_devices = torch.cuda.device_count()
            gc.collect()
            return tuple(torch.cuda.memory_allocated(i) for i in range(num_devices))

        before = _get_cuda_memory_usage()

        # Run as separate function so that gc can clean up everything when we
        # check for memory usage.
        self._test_reentrant_parent_error_on_cpu(device)

        # Wait for autograd thread to cleanup failed tasks.
        after = _get_cuda_memory_usage()
        start = time.time()
        while before != after and time.time() - start < 30:
            time.sleep(0.1)
            after = _get_cuda_memory_usage()

        self.assertEqual(before, after)

    # TODO: see if these tests can be ported to OpInfos or moved to where's test suite
    @skipIfMPS  # the test doesn't work on MPS
    def test_where_functional(self, device):
        # [FUNCTION DOC] Check gradcheck and gradgradcheck of where with same shape and broadcast
        # shapes. Skipped on MPS.
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where(cond, x, y):
            # [FUNCTION DOC] HELPER: Wrapper calling where.
            return torch.where(cond, x, y)

        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, device=device)])

        x = torch.randn(5, 1, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, 1, dtype=torch.double, device=device, requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, 5, device=device)])

    @skipIfMPS  # the test doesn't work on MPS
    def test_where_scalar(self, device):
        # [FUNCTION DOC] Check gradcheck and gradgradcheck of where with a scalar in either
        # branch. Skipped on MPS.
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        scalar = 4.0
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where_scalar_first(cond, x):
            # [FUNCTION DOC] HELPER: Call where with the scalar as the true branch.
            return torch.where(cond, scalar, x)

        def where_scalar_second(cond, x):
            # [FUNCTION DOC] HELPER: Call where with the scalar as the false branch.
            return torch.where(cond, x, scalar)

        gradcheck(where_scalar_first, (cond, x))
        gradgradcheck(where_scalar_first, (cond, x))

        gradcheck(where_scalar_second, (cond, x))
        gradgradcheck(where_scalar_second, (cond, x))

    @onlyCUDA
    def test_free_unneeded_tensor(self, device):
        # [FUNCTION DOC] Check that tensors not needed for backward are freed: allocated memory
        # after a sum of a scaled shifted tensor matches memory after a plain sum. CUDA only.
        x = torch.randn(2, 3, 10, 10, device=device, requires_grad=True)
        m = torch.randn(1, 3, 1, 1, device=device)

        z = x.sum()
        base_mem = torch.cuda.memory_allocated()
        z = ((x + 2) * m).sum()
        end_mem = torch.cuda.memory_allocated()

        # In the end the memory usage should remain equal, because neither of
        # (x + 2) and ((x + 2) * m) should be kept alive for backward, while the
        # previous allocation of z had the same size as the current one.
        self.assertEqual(base_mem, end_mem)

    @onlyCUDA
    def test_pin_memory(self, device):
        # [FUNCTION DOC] Check pin_memory returns a new equal tensor that still requires grad,
        # with gradcheck and gradgradcheck passing. CUDA only.
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        self.assertEqual(x, x.pin_memory())
        self.assertIsNot(x, x.pin_memory())
        self.assertTrue(x.pin_memory().requires_grad)
        gradcheck(lambda x: x.pin_memory(), [x])
        gradgradcheck(lambda x: x.pin_memory(), [x])

    @onlyCUDA
    def test_profiler_emit_nvtx(self, device):
        # [FUNCTION DOC] Smoke test that constructing the NVTX profiler range context under the
        # CUDA profiler does not fail. CUDA only.
        # This test is not intended to ensure correctness of nvtx ranges.
        # That would require something a great deal more complex (you'd have to create a
        # profile in a subprocess, open it, and parse the sql somehow).
        # This test is merely intended to catch if emit_nvtx breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with torch.cuda.profiler.profile():
            with emit_nvtx():
                a.add(1.0)

    @onlyCUDA
    def test_rnn_backward_to_input_but_not_parameters(self, device):
        # [FUNCTION DOC] Check backward to the input of an LSTM whose parameters do not require
        # grad gives a nonzero input gradient. CUDA only.
        # this checks whether it is possible to not require
        # weight parameters, but require inputs, see #7722
        l = torch.nn.LSTM(2, 3).to(device)
        for p in l.parameters():
            p.requires_grad = False
        s = torch.randn(1, 1, 2, requires_grad=True, device=device)
        out, _ = l(s)
        out.sum().backward()
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    @unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
    def test_profiler_emit_itt(self, device):
        # [FUNCTION DOC] Smoke test that constructing the ITT profiler range context does not
        # fail. Skipped when ITT is unavailable.
        # This test is not intended to ensure correctness of itt ranges.
        # That would require something a great deal more complex (you'd have to create a
        # profile in a subprocess, open it, and parse the sql somehow).
        # This test is merely intended to catch if emit_itt breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with emit_itt():
            a.add(1.0)

    @deviceCountAtLeast(1)
    def test_grad_assignment(self, devices):
        # [FUNCTION DOC] Check errors from assigning bad values to the grad attribute: wrong type
        # (TypeError), wrong shape, wrong dtype, itself, a device gradient on a CPU tensor,
        # cross-device assignment; half type works on CUDA. Takes a list of devices.
        x = torch.randn(5, 5, device=devices[0])

        # Tests that the wrong type raises
        with self.assertRaisesRegex(TypeError, "expected to be a Tensor or None"):
            x.grad = 0

        # Tests that the wrong shape raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2, device=devices[0])

        # Tests that the wrong dtype raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(5, 5, dtype=torch.long, device=devices[0])

        # Tests that self-assignment raises
        with self.assertRaises(RuntimeError):
            x.grad = x

        # Tests device -> cpu grad assignment raises
        if self.device_type != "cpu":
            with self.assertRaises(RuntimeError):
                t_cpu = torch.rand(5, 5)
                t_cpu.grad = torch.randn(5, 5, device=devices[0])

        # Tests half type on CUDA
        if self.device_type == "cuda":
            x = x.to(dtype=torch.half, device=devices[0])
            x.grad = torch.zeros_like(x)

        # Tests cross-device assignment raises
        if len(devices) > 1:
            x = torch.randn(5, 5, device=devices[0])
            with self.assertRaises(RuntimeError):
                x.grad = torch.randn(5, 5, device=devices[1])

    @dtypesIfMPS(torch.float32)
    @deviceCountAtLeast(1)
    @dtypes(torch.float, torch.double)
    def test_requires_grad_factory(self, devices, dtype):
        # [FUNCTION DOC] Check ones_like and randn_like honor requires_grad, dtype and device
        # arguments. Takes the device list and dtype.
        fns = [torch.ones_like, torch.randn_like]
        x = torch.randn(2, 3, dtype=dtype, device=devices[0])

        for fn in fns:
            for requires_grad in [True, False]:
                output = fn(
                    x, dtype=dtype, device=devices[0], requires_grad=requires_grad
                )
                self.assertEqual(requires_grad, output.requires_grad)
                self.assertIs(dtype, output.dtype)
                self.assertEqual(devices[0], str(x.device))

    @deviceCountAtLeast(2)
    def test_unused_output_device(self, devices):
        # [FUNCTION DOC] Check backward works when only the last of the broadcast outputs across
        # devices is used. Needs two devices.
        from torch.nn.parallel._functions import Broadcast

        x = torch.randn(5, 5, dtype=torch.float, device=devices[0], requires_grad=True)
        outputs = Broadcast.apply(list(range(len(devices))), x)
        y = outputs[-1] * 2
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 2)

    @deviceCountAtLeast(2)
    def test_backward_device(self, devices):
        # [FUNCTION DOC] Check the device a custom backward sees for its gradient is the device of
        # the variable. Needs two devices.
        # check that current device matches the variable's device
        device = [None]

        class Identity(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function recording the device of the incoming gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Record the gradient device and return a clone of
                # the gradient.
                device[0] = grad_output.device
                return grad_output.clone()

        v = torch.randn(1, device=devices[1], requires_grad=True)
        Identity.apply(v).backward()
        self.assertEqual(str(device[0]), devices[1])

    @deviceCountAtLeast(2)
    def test_inputbuffer_add_multidevice(self, devices):
        # [FUNCTION DOC] Check backward through a sum of two copies of a tensor moved to another
        # device works. Needs two devices.
        input = torch.randn(1, device=devices[0], requires_grad=True)
        output = input.to(device=devices[1]) + input.to(device=devices[1])
        output.backward()

    @onlyCPU
    def test_copy_(self, device):
        # [FUNCTION DOC] Check copy_ propagates requires_grad for float, half and bfloat16
        # destination dtypes (regression for bfloat16 not counted as floating). CPU only.
        # At the time of writing this test, copy_ is not generated from native_functions.yaml
        # there was a bug that bfloat16 was not recognized as floating.
        x = torch.randn(10, device=device, requires_grad=True)
        floating_dt = floating_types_and(torch.half, torch.bfloat16)
        for dt in floating_dt:
            y = torch.empty(10, device=device, dtype=dt)
            y.copy_(x)
            self.assertTrue(y.requires_grad)
            z = x.to(torch.bfloat16)
            self.assertTrue(z.requires_grad)

    def test_copy_forward_ad_broadcasting(self, device):
        # [FUNCTION DOC] Check forward AD for copy_ with a source that broadcasts to a destination
        # of a different shape works.
        # copy_ allows the src to have a different shape from self as long as src is
        # broadcastable to self. Make sure forward AD handles this case.
        primal = torch.rand(3, 3, device=device)
        tangent = torch.rand(3, 3, device=device)
        non_dual = torch.rand(1, 3, 3, device=device)

        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            non_dual.copy_(dual)

    def test_copy_forward_ad_same_layout_copies_grad(self, device):
        # [FUNCTION DOC] Check forward AD copy_ with the same layout copies the tangent rather
        # than sharing it.
        primal = torch.tensor([[3.0], [4.0]], device=device)
        tangent = torch.tensor([[5.0], [6.0]], device=device)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(primal, tangent)
            non_dual = torch.tensor([[1.0], [2.0]])
            non_dual.copy_(x_dual)
            self.assertTrue(fwAD.unpack_dual(non_dual).tangent is not tangent)

    @onlyCUDA
    def test_simple_reentrant_cross_device(self, device):
        # [FUNCTION DOC] Check reentrant backward started and finished on different thread kinds
        # (GPU to GPU, CPU to GPU, GPU to CPU) via a class flag. CUDA only.
        class ReentrantFunc(Function):
            # [CLASS DOC] HELPER: Function whose backward runs a nested backward on CPU or on the
            # device depending on a class flag.
            _cpu_mode = True

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return x times x plus two.
                return x * (x + 2)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Under enable_grad run a nested backward of a squared
                # random parameter on CPU or device, and return the incoming gradient.
                with torch.enable_grad():
                    if ReentrantFunc._cpu_mode:
                        new_param = torch.randn(2, 2, requires_grad=True)
                        (new_param**2).sum().backward()
                    else:
                        new_param = torch.randn(2, 2, device=device, requires_grad=True)
                        (new_param**2).sum().backward()
                return grad_output

        # Reentrant starts on GPU thread, finishes on GPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on CPU thread, finishes on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        # set ReentrantFunc node to GPU to emit tasks to GPU queue
        ReentrantFunc._cpu_mode = False
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on GPU thread, finishes on CPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        # set ReentrantFunc node to CPU to emit tasks to CPU queue
        ReentrantFunc._cpu_mode = True
        out = ReentrantFunc.apply(x)
        out.sum().backward()

    @onlyCUDA
    def test_cross_device_reentrant_autograd(self, device):
        # [FUNCTION DOC] Check cross-device reentrant autograd with checkpoint where the last
        # reentrant op runs on the CPU thread for the GPU thread, which must handle the empty node
        # task without crashing. CUDA only.
        # Output on gpu so that this task will be associated with the gpu thread
        def fn_on_gpu(inp):
            # [FUNCTION DOC] HELPER: Do some extra multiplications to raise priority and move the
            # input to the device.
            # Artificially increase the priority of the next op to make sure it runs
            # as soon as we reach it before the ops of branch1.
            dummy = inp * 2 * 2 * 2 * 2
            return inp.to(device=device)

        def parent_on_cpu(inp):
            # [FUNCTION DOC] HELPER: Build a slow branch of divides on the device and a reentrant
            # checkpoint branch, and add them.
            # Slow branch of ops on gpu so that the work queue for the gpu thread
            # won't empty too quickly. They also have smaller priorities than the
            # ones created by fn_on_gpu
            branch1 = inp.to(device=device)
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            # Perform checkpoint on cpu tensors. So the last op performed in the reentrant
            # autograd is an AccumulateGrad that runs on the cpu thread for the gpu thread.
            # So the cpu thread will notify the gpu thread with an empty NodeTask.
            branch2 = checkpoint(fn_on_gpu, inp, use_reentrant=True)
            out = branch2 + branch1
            return out

        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)
        # This will segfault if the empty NodeTask is not handled properly in the
        # gpu thread ReadyQueue
        out.sum().backward()

    def test_inplace_on_view_backprop_base(self, device):
        # [FUNCTION DOC] Check in-place op on a view then backward through the base gives
        # expected gradients.
        # modify view and back-prop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v1.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [1, 1]])

    def test_inplace_on_view_backprop_view_of_view(self, device):
        # [FUNCTION DOC] Check in-place op on one view and backward through another view of the
        # same base.
        # modify view and backprop through view-of-view
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = x.narrow(0, 0, 1)
        v1.mul_(2)
        v2.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

    def test_inplace_on_view_of_view(self, device):
        # [FUNCTION DOC] Check in-place op on a view of a view then backward through the base.
        # modify view-of-view and backprop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()

        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1]])

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_then_no_grad(self, device):
        # [FUNCTION DOC] Check forcing a graph update with grad disabled after an in-place op on a
        # view still lets backward work. Skipped on MPS.
        # Perform an in-place operation on a view of a non-leaf variable.
        a = torch.ones(3, 1, dtype=torch.double, device=device, requires_grad=True)
        b = a * 2
        c = b.view_as(b)
        c[0][0] = 3

        # Force a graph update with grad disabled.
        with torch.no_grad():
            c.grad_fn

        c.sum().backward()

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_gradcheck(self, device):
        # [FUNCTION DOC] Check gradcheck and gradgradcheck for in-place multiplies on nested
        # narrow views. Skipped on MPS.
        # gradcheck modifications to views
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            # [FUNCTION DOC] HELPER: Clone the root and multiply two overlapping nested slices in
            # place by b.
            x = root.clone()
            x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
            x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_multiple_outputs(self, device):
        # [FUNCTION DOC] Check in-place op on one output of unbind raises RuntimeError.
        root = torch.arange(9.0, dtype=torch.double).reshape(3, 3).requires_grad_()
        x = root.clone()
        v1 = x.unbind()
        with self.assertRaises(RuntimeError):
            v1[0].mul_(2)

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_of_multiple_output_view(self, device):
        # [FUNCTION DOC] Check in-place op on a view of an unbind output raises RuntimeError.
        # Skipped on MPS.
        a = torch.rand(
            10, dtype=torch.double, device=device, requires_grad=True
        ).clone()
        b = a.unbind(0)
        c = b[0].view_as(b[0])
        with self.assertRaises(RuntimeError):
            c.mul_(2)

    @skipIfMPS  # MPS backend doesn't support double types
    def test_inplace_multiple_output_view_of_view(self, device):
        # [FUNCTION DOC] Check in-place op on an unbind output of a view raises RuntimeError.
        # Skipped on MPS.
        a = torch.rand(
            10, dtype=torch.double, device=device, requires_grad=True
        ).clone()
        b = a.view_as(a)
        c = b.unbind(0)
        with self.assertRaises(RuntimeError):
            c[0].mul_(2)

    @skipIfMPS  # MPS backend doesn't support double types
    def test_inplace_on_view_makes_base_require_grad(self, device):
        # [FUNCTION DOC] Check an in-place op on a view with a grad-requiring operand makes the
        # base require grad, with gradcheck and gradgradcheck. Skipped on MPS.
        # in-place modification to view makes base require grad
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=False)
        b = torch.randn(4, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            # [FUNCTION DOC] HELPER: Clone, check requires_grad is false, multiply a slice in
            # place by b, check it is now true, return the result.
            x = root.clone()
            self.assertFalse(x.requires_grad)
            x.narrow(1, 2, 2).mul_(b)
            self.assertTrue(x.requires_grad)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_backprop_view(self, device):
        # [FUNCTION DOC] Check in-place multiply on a narrow of a non-grad base gives the right
        # gradient for the other operand and none for the base.
        # modify view and backprop through view
        a = torch.tensor([2.0, 5.0], device=device, requires_grad=False)
        b = torch.tensor([3.0], device=device, requires_grad=True)
        res = a.narrow(0, 1, 1).mul_(b)
        res.sum().backward()
        self.assertEqual(b.grad.tolist(), [5])
        self.assertIsNone(a.grad)

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_modify_base(self, device):
        # [FUNCTION DOC] Check an in-place op on the base that makes it require grad also makes
        # previously created views require grad and backprop correctly. Skipped on MPS.
        # Test that an in-place operation on a base that forced it to require
        # grad also forces any previous views to require grad and backprop
        # correctly
        r = torch.ones(1, dtype=torch.double, device=device, requires_grad=True)

        def fn(r):
            # [FUNCTION DOC] HELPER: Create a base and a view before the in-place add of r,
            # checking the view starts without grad and gains it.
            x = torch.ones(5, dtype=torch.double, device=device)
            v = x.select(0, 1)
            self.assertFalse(v.requires_grad)
            self.assertIsNone(v.grad_fn)
            x.add_(r)  # v is now dependent on r due to the in-place op on x
            self.assertTrue(v.requires_grad)
            return v

        gradcheck(fn, [r])
        gradgradcheck(fn, [r])

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_python(self, device):
        # [FUNCTION DOC] Check in-place modification of views through a Python custom Function
        # with gradcheck and gradgradcheck. Skipped on MPS.
        # in-place modifications of Python-autograd created view
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        class PyAdd(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function adding into its first input in place and marking
            # it dirty.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Add in place, mark dirty and return the first input.
                ctx.mark_dirty(x)
                x.add_(y)
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient for both inputs.
                return grad, grad

        def func(root, b):
            # [FUNCTION DOC] HELPER: Clone the root and apply the Function to two overlapping
            # nested slices.
            x = root.clone()
            PyAdd.apply(x.narrow(1, 2, 2).narrow(0, 1, 2), b)
            PyAdd.apply(x.narrow(1, 0, 2).narrow(0, 1, 2), b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_non_contig(self, device):
        # [FUNCTION DOC] Check in-place op on a view of a non-contiguous (transposed) root gives
        # the right gradients.
        root = torch.ones(2, 3, 2, device=device).select(2, 1).t().requires_grad_(True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1], [1, 1]])

    def test_inplace_on_view_multi_output_unsafe(self, device):
        # [FUNCTION DOC] Check in-place ops on pieces of unsafe split and chunk (which are not
        # marked multi-view) work with backward.
        for f in [
            lambda t: t.unsafe_split(1),
            lambda t: t.unsafe_split_with_sizes((1, 1, 1)),
            lambda t: t.unsafe_chunk(3),
        ]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            s1.mul_(s2)
            s1.sum().backward()

    def test_inplace_on_view_multi_output_safe(self, device):
        # [FUNCTION DOC] Check in-place ops on pieces of safe split and chunk raise RuntimeError
        # about functions that return multiple views.
        for f in [
            lambda t: t.split(1),
            lambda t: t.split_with_sizes((1, 1, 1)),
            lambda t: t.chunk(3),
        ]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            error_msg = (
                "This view is the output of a function that returns multiple views."
            )
            with self.assertRaisesRegex(RuntimeError, error_msg):
                s1.mul_(s2)

    def test_inplace_on_view_undefined_grad_output(self, device):
        # [FUNCTION DOC] Check an in-place Function on a view with an input that gets no gradient
        # does not break backward.
        a = torch.tensor([1.0], requires_grad=True)
        c = a.clone()
        v = c[:]
        b = torch.tensor(1.0, requires_grad=True)

        class InplaceFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function doubling its input in place and marking it dirty,
            # ignoring the second input.
            @staticmethod
            def forward(ctx, x, other):
                # [FUNCTION DOC] HELPER: Mark dirty and return the input doubled in place.
                ctx.mark_dirty(x)
                return x.mul_(2)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient doubled and None.
                return grad * 2, None

        out = InplaceFunc.apply(v, b)
        out.backward()
        self.assertIsNone(b.grad)
        self.assertEqual(a.grad.item(), 2)

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_mv_grad_stride_0(self, device):
        # [FUNCTION DOC] Check gradcheck of matrix-vector product with an expanded (stride zero)
        # vector. Skipped on MPS.
        # Reference: https://github.com/pytorch/pytorch/issues/38315
        mat = torch.randn(2, 2, dtype=torch.double, device=device)
        vec = torch.randn(1, dtype=torch.double, device=device).requires_grad_(True)

        def fn(vec):
            # [FUNCTION DOC] HELPER: Expand the vector and compute the sum of the
            # matrix-vector product.
            # Expand inside the function to make sure the input to
            # gradcheck does not have overlapping memory
            vec = vec.expand(2)
            return (mat @ vec).sum()

        gradcheck(fn, (vec))
        gradgradcheck(fn, (vec))

    @onlyCUDA
    def test_gradcheck_input_output_different_device(self, device):
        # [FUNCTION DOC] Check gradcheck works when input and output are on different devices
        # (device to CPU and CPU to device). CUDA only.
        x = torch.ones((1,), dtype=torch.double, device="cuda", requires_grad=True)
        gradcheck(lambda x: x.to("cpu"), (x,))

        x = torch.ones((1,), dtype=torch.double, device="cpu", requires_grad=True)
        gradcheck(lambda x: x.to("cuda"), (x,))

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/181229"
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS or IS_WINDOWS,
        "https://github.com/pytorch/pytorch/issues/181203",
    )
    @unittest.skipIf(
        IS_WINDOWS and IS_ARM64, "Fails on Windows ARM64; see issue #181228"
    )
    def test_strided_leaf_grad_layout(self, device):
        # [FUNCTION DOC] Check gradient layout rules for leaves: dense non-overlapping leaves get
        # grads with the same strides; other leaves get row-major contiguous grads, for broadcast
        # and non-broadcast gradients. Skipped on Linux, slow and several other configurations.
        # (1) If leaf is non-overlapping and dense, grad's layout should match its leaf.
        for fmt_a in (torch.contiguous_format, torch.channels_last):
            for fmt_b in (torch.contiguous_format, torch.channels_last):
                a = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_a)
                b = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_b)
                a.requires_grad_()
                b.requires_grad_()
                # checks (1) for broadcasted gradients
                a.sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                b.sum().backward()
                self.assertEqual(b.grad.stride(), b.stride())
                # checks (1) for non-broadcasted gradients
                a.grad = None
                b.grad = None
                (a * b).sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                self.assertEqual(b.grad.stride(), b.stride())

        # (2) If leaf isn't dense, checks that grads are rowmajor contiguous.
        c = torch.empty_strided((2, 2), (4, 2), device=device).copy_(
            torch.rand((2, 2), device=device)
        )
        c.requires_grad_()
        d = torch.rand((2, 2), device=device)
        # checks (2) for broadcasted gradients
        c.sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))
        # checks (2) for non-broadcasted gradients
        c.grad = None
        (c * d).sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))

    @skipIfTorchDynamo("tests C++ autograd engine layout policy, not dynamo")
    def test_enforce_grad_layout_policy(self, device):
        # [FUNCTION DOC] Check the gradient layout policy flag: enforcement on matches parameter
        # strides; off steals the contiguous gradient; the setting is restored after the context
        # manager, can be used as a function, and as a decorator without leaking at decoration
        # time. Skipped under dynamo.
        # A custom function that returns a fresh contiguous gradient,
        # which has different strides from a channels_last parameter.
        class ContiguousGrad(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward returns a freshly created contiguous
            # gradient of fixed shape.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the sum of the input.
                return x.sum()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return a ones tensor of shape two by three by four by
                # five on the gradient device.
                return torch.ones(2, 3, 4, 5, device=grad_output.device)

        # Parameter is channels_last -- its strides differ from contiguous.
        a = (
            torch.rand(2, 3, 4, 5, device=device)
            .to(memory_format=torch.channels_last)
            .requires_grad_()
        )

        # --- enforcement ON (default): grad strides match param ---
        ContiguousGrad.apply(a).backward()
        self.assertEqual(a.grad.stride(), a.stride())
        a.grad = None

        # --- enforcement OFF: grad is stolen as-is (contiguous strides) ---
        with torch.autograd.enforce_grad_layout_policy(False):
            ContiguousGrad.apply(a).backward()
            self.assertTrue(a.grad.is_contiguous())
            self.assertNotEqual(a.grad.stride(), a.stride())
        a.grad = None

        # --- context manager restores state ---
        ContiguousGrad.apply(a).backward()
        self.assertEqual(a.grad.stride(), a.stride())
        a.grad = None

        # --- used as a bare function (not context manager) ---
        try:
            torch.autograd.enforce_grad_layout_policy(False)
            ContiguousGrad.apply(a).backward()
            self.assertTrue(a.grad.is_contiguous())
            self.assertNotEqual(a.grad.stride(), a.stride())
        finally:
            torch.autograd.enforce_grad_layout_policy(True)
        a.grad = None

        # --- as a decorator ---
        torch._C._set_grad_layout_enforcement_enabled(True)

        @torch.autograd.enforce_grad_layout_policy(False)
        def do_backward():
            # [FUNCTION DOC] HELPER: Backward of the Function with layout enforcement disabled
            # by decoration.
            ContiguousGrad.apply(a).backward()

        # Check that the flag doesn't leak during function decoration.
        self.assertTrue(torch._C._is_grad_layout_enforcement_enabled())

        do_backward()
        self.assertTrue(a.grad.is_contiguous())
        self.assertNotEqual(a.grad.stride(), a.stride())

    @skipIfMPS
    def test_copy_r_to_c(self, device):
        # [FUNCTION DOC] Check copy_ from a real to a complex tensor backpropagates the real part
        # of the sign without warning. Skipped on MPS.
        out_c = torch.empty(3, 2, dtype=torch.cdouble, device=device)
        inp_r = torch.randn(3, 2, dtype=torch.double, device=device, requires_grad=True)

        def do_test():
            # [FUNCTION DOC] HELPER: Copy real into complex, sum, backward the absolute value and
            # compare the gradient with the real part of the sign.
            out_c.copy_(inp_r)
            out_c_inter = out_c.sum()
            out_c_inter.abs().backward()
            with torch.no_grad():
                self.assertEqual(
                    inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_c_inter).real
                )

        self.assertNotWarn(do_test)

    def test_to_r_to_c(self, device):
        # [FUNCTION DOC] Check converting real to complex with to backpropagates correctly
        # without warnings.
        def do_test():
            # [FUNCTION DOC] HELPER: Convert real to complex, sum, backward the absolute value and
            # compare the gradient with the real part of the sign.
            inp_r = torch.randn(
                3, 2, dtype=torch.double, device=device, requires_grad=True
            )
            out = inp_r.to(torch.complex128)
            out_inter = out.sum()
            out_inter.abs().backward()
            with torch.no_grad():
                self.assertEqual(
                    inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_inter).real
                )

        self.assertNotWarn(do_test)

    def test_non_differentiable_ops(self, device):
        # [FUNCTION DOC] Check isin and signbit give outputs that do not require grad.
        # Just make sure the op doesn't raise an error
        # and resulting tensor has requires_grad=False.
        x = torch.tensor([[1, 2], [3, 4.0]], requires_grad=True, device=device)
        out = torch.isin(x, torch.tensor([2, 3], device=device))
        self.assertFalse(out.requires_grad)

        x = torch.randn(3, 3, requires_grad=True)
        out = torch.signbit(x)
        self.assertFalse(out.requires_grad)

    def test_warning_in_backward(self, device):
        # [FUNCTION DOC] Check a warning raised during backward is surfaced as a Python
        # UserWarning, also from device worker threads.
        # Test warning during backward are always propagated as python warnings (gh-50209)
        # NOTE: For device=cuda, warning gets propagated from a worker thread
        a = torch.zeros((), device=device, requires_grad=True)
        b = torch._C._nn._test_warn_in_autograd(a)

        with self.assertWarnsRegex(UserWarning, "Warn from backward"):
            b.backward()

    def test_complex_scalar_backward(self, device):
        # [FUNCTION DOC] Check backward and grad on a complex scalar output raise RuntimeError
        # about implicit grad for real scalar outputs only.
        a = torch.zeros(1, device=device, requires_grad=True)
        b = a * 0.5j

        msg = "grad can be implicitly created only for real scalar outputs"
        with self.assertRaisesRegex(RuntimeError, msg):
            b.backward()

        with self.assertRaisesRegex(RuntimeError, msg):
            torch.autograd.grad(b, a)

    def test_grad_unused_input_error_message(self, device):
        # [FUNCTION DOC] Check the error message for an unused differentiated input names the
        # index of the unused tensor.
        for unused_idx in [0, 1]:
            x = torch.randn(10, requires_grad=True, device=device)
            y = torch.randn(10, requires_grad=True, device=device)
            inputs = (x, y)

            # If unused_idx is 0, use inputs[1]. If unused_idx is 1, use inputs[0].
            out = inputs[1 - unused_idx].sum()

            expected_msg = f"The differentiated Tensor at index {unused_idx} appears"
            with self.assertRaisesRegex(RuntimeError, expected_msg):
                torch.autograd.grad(out, inputs)

    def test_pow_real_negative_base_complex_exponent(self, device):
        # [FUNCTION DOC] Check gradcheck of pow with a negative real base (tensor and python
        # scalar) and a complex exponent.
        # OpInfo doesn't naturally support input of mixed types, hence this test here.
        base = -torch.ones(2, device=device, dtype=torch.double)
        exponent = torch.randn(
            2, device=device, dtype=torch.cdouble, requires_grad=True
        )

        def fn(exponent):
            # [FUNCTION DOC] HELPER: Power with the captured negative tensor base.
            return torch.pow(base, exponent)

        torch.autograd.gradcheck(fn, (exponent,))

        def fn(exponent):
            # [FUNCTION DOC] HELPER: Power with a scalar negative one base.
            return torch.pow(-1, exponent)

        torch.autograd.gradcheck(fn, (exponent,))

    def test_resize_version_bump(self, device):
        # [FUNCTION DOC] Check the version counter is bumped by resize_ and resize_as_ when the
        # size changes and not when it is a no-op.
        x = torch.rand((1,), device=device)
        y = torch.randn((3,), device=device)
        x.resize_((1, 2))
        self.assertEqual(x._version, 1)
        x.resize_as_(y)
        self.assertEqual(x._version, 2)

        # In the following cases, `resize` is no-op,
        # so no version bumps.
        x.resize_((3,))
        self.assertEqual(x._version, 2)

        x.resize_as_(y)
        self.assertEqual(x._version, 2)

    @unittest.skipIf(not torch.accelerator.is_available(), "requires accelerator")
    def test_zero_dim_param_mixed_device_grad(self, device):
        # [FUNCTION DOC] Check zero-dimensional CPU parameters used with accelerator inputs keep
        # CPU gradients. Skipped without an accelerator.
        # cpu 0-dim params with an accelerator device grad
        # https://github.com/pytorch/pytorch/issues/160084
        class RegressionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two scalar CPU parameters computing a linear
            # function of the input.
            def __init__(self, a=0, b=0):
                # [FUNCTION DOC] HELPER: Create the two scalar parameters from the given numbers.
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(a).float())
                self.b = torch.nn.Parameter(torch.tensor(b).float())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input times a plus b.
                return x * self.a + self.b

        # Keep the model on cpu as we do want to test the mixed cpu/accelerator behavior here
        model = RegressionModel()
        inputs = torch.randn(4, 10, device=device)
        out = model(inputs)
        out.sum().backward()
        self.assertIsNotNone(model.a.grad)
        self.assertIsNotNone(model.b.grad)
        self.assertEqual(model.a.grad.device, torch.device("cpu"))
        self.assertEqual(model.b.grad.device, torch.device("cpu"))

    @dtypes(torch.double)
    def test_cdist_gradgrad(self, device, dtype):
        # [FUNCTION DOC] Check second-order gradients (gradcheck and gradgradcheck) of cdist for
        # several norms, batch shapes and compute modes, excluding small p. Double only.
        # Second-order (double) backward for cdist / _cdist_backward. p < 1 is
        # excluded here because |diff|^(p-2) makes the second derivative too
        # singular for a reliable finite-difference gradgradcheck.
        make = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        shape_pairs = [
            ((4, 3), (5, 3)),  # unbatched
            ((2, 4, 3), (2, 5, 3)),  # batched
            ((1, 4, 3), (2, 5, 3)),  # broadcast batch
        ]
        for p in [1.0, 1.5, 2.0, 2.5, 3.0, float("inf")]:
            # p == 2 dispatches to _euclidean_dist (mm) or the fused kernel
            # depending on compute_mode; exercise both backward paths.
            modes = (
                ["use_mm_for_euclid_dist", "donot_use_mm_for_euclid_dist"]
                if p == 2.0
                else ["donot_use_mm_for_euclid_dist"]
            )
            for s1, s2 in shape_pairs:
                for mode in modes:
                    x1, x2 = make(s1), make(s2)

                    def fn(a, b, p=p, mode=mode):
                        # [FUNCTION DOC] HELPER: Compute cdist with the captured norm and
                        # compute mode.
                        return torch.cdist(a, b, p, compute_mode=mode)

                    with self.subTest(p=p, shapes=(s1, s2), mode=mode):
                        self.assertTrue(gradcheck(fn, (x1, x2)))
                        # Compiled autograd cannot reproduce eager's undefined-grad
                        # accumulation. Check with eager compilation.
                        self.assertTrue(
                            gradgradcheck(
                                fn,
                                (x1, x2),
                                check_undefined_grad=not TEST_WITH_TORCHDYNAMO,
                            )
                        )

    @dtypes(torch.double)
    def test_pdist_gradgrad(self, device, dtype):
        # [FUNCTION DOC] Check second-order gradients of pdist for several norms and shapes.
        # Double only.
        make = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        for p in [1.0, 1.5, 2.0, 2.5, 3.0, float("inf")]:
            for shape in [(4, 3), (5, 2)]:
                x = make(shape)

                def fn(a, p=p):
                    # [FUNCTION DOC] HELPER: Compute pdist with the captured norm.
                    return torch.nn.functional.pdist(a, p)

                with self.subTest(p=p, shape=shape):
                    self.assertTrue(gradcheck(fn, (x,)))
                    # See test_cdist_gradgrad: undefined-grad mode is eager-only.
                    self.assertTrue(
                        gradgradcheck(
                            fn, (x,), check_undefined_grad=not TEST_WITH_TORCHDYNAMO
                        )
                    )

    @dtypes(torch.double)
    def test_cdist_gradgrad_small_p(self, device, dtype):
        # [FUNCTION DOC] Check the analytic double backward of cdist for p below one against a
        # reference built from elementwise ops. Double only.
        # p < 1 is too singular for finite-difference gradgradcheck, so compare
        # the analytic double backward to an autograd reference. Point clouds are
        # separated so no pairwise difference is near zero.
        def ref_cdist(a, b, p):
            # [FUNCTION DOC] HELPER: Reference pairwise p-norm distances from
            # broadcasted differences.
            return (a.unsqueeze(-2) - b.unsqueeze(-3)).abs().pow(p).sum(-1).pow(1.0 / p)

        def second_order(fn, x1, x2, v, p):
            # [FUNCTION DOC] HELPER: Differentiate the distance function with respect to the first
            # input with create_graph, then differentiate its sum with respect to all inputs.
            d = fn(x1, x2, p)
            g1 = torch.autograd.grad(d, x1, grad_outputs=v, create_graph=True)[0]
            return torch.autograd.grad(g1.sum(), (x1, x2, v))

        mk = partial(make_tensor, device=device, dtype=dtype)
        for p in [0.5, 0.75]:
            x1 = mk((4, 3), requires_grad=True)
            x2 = (mk((5, 3)).abs() + 20).detach().requires_grad_()
            v = mk((4, 5), requires_grad=True)
            with self.subTest(p=p):
                self.assertEqual(
                    second_order(torch.cdist, x1, x2, v, p),
                    second_order(ref_cdist, x1, x2, v, p),
                )

    @dtypes(torch.double)
    def test_pdist_gradgrad_small_p(self, device, dtype):
        # [FUNCTION DOC] Check the analytic double backward of pdist for p below one against a
        # reference using off-diagonal pairs. Double only.
        # See test_cdist_gradgrad_small_p; rows separated to avoid near-zero diffs.
        def ref_pdist(a, p):
            # [FUNCTION DOC] HELPER: Reference pdist using only off-diagonal pairs, computed from
            # elementwise ops.
            # Gather only the off-diagonal pairs; including self-distances would
            # put a |0|^(p-1) singularity in the graph for p < 1.
            i, j = torch.triu_indices(a.size(0), a.size(0), 1, device=a.device).unbind()
            return (a[i] - a[j]).abs().pow(p).sum(-1).pow(1.0 / p)

        def second_order(fn, x, v, p):
            # [FUNCTION DOC] HELPER: Differentiate the distance function with create_graph, then
            # differentiate the sum of the gradient with respect to the input and the grad output.
            d = fn(x, p)
            g = torch.autograd.grad(d, x, grad_outputs=v, create_graph=True)[0]
            return torch.autograd.grad(g.sum(), (x, v))

        mk = partial(make_tensor, device=device, dtype=dtype)
        for p in [0.5, 0.75]:
            n, m = 4, 3
            sep = torch.arange(n, device=device, dtype=dtype).view(n, 1) * 50
            x = (mk((n, m)) + sep).detach().requires_grad_()
            v = mk((n * (n - 1) // 2,), requires_grad=True)
            with self.subTest(p=p):
                self.assertEqual(
                    second_order(torch.nn.functional.pdist, x, v, p),
                    second_order(ref_pdist, x, v, p),
                )


class TestAllowMutationOnSaved(TestCase):
    # [CLASS DOC] Tests for the context manager that allows modifying saved tensors in place by
    # cloning them on demand, tracking cloned tensors and a map of tensor ids to weak handles on
    # the context object.
    def assertClonedLenEqual(self, ctx, n):
        # [FUNCTION DOC] HELPER: Assert the number of cloned tensors recorded on the context.
        self.assertEqual(len(list(ctx.cloned.items())), n)

    def assertTIDMapLenEqual(self, ctx, n):
        # [FUNCTION DOC] HELPER: Assert the number of tracked tensor ids on the context.
        self.assertEqual(len(list(ctx.tid_to_weakhandle.items())), n)

    def test_basic(self):
        # [FUNCTION DOC] Check an in-place op on a saved tensor errors normally and works inside
        # the context, with the right gradient and no leftover clones.
        a = torch.rand(2, 3, requires_grad=True)

        def fn(a):
            # [FUNCTION DOC] HELPER: Clone, square and sum, then modify the clone in place,
            # backward and return the grad.
            b = a.clone()
            out = (b**2).sum()
            b.sin_()
            out.sum().backward()
            return a.grad

        msg = (
            "variables needed for gradient computation has been modified by an inplace"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)

        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)

        self.assertTrue(torch.allclose(a * 2, da))
        self.assertClonedLenEqual(ctx, 0)

    def test_views(self):
        # [FUNCTION DOC] Check the same as the basic case with the in-place op applied to a view
        # of the saved tensor.
        a = torch.rand(2, 3, requires_grad=True)

        def fn(a):
            # [FUNCTION DOC] HELPER: Clone, view, square sum, modify the view in place, backward
            # and return the grad.
            b = a.clone()
            c = b.view_as(b)
            out = (b**2).sum()  # How does this work?
            c.sin_()
            out.sum().backward()
            return a.grad

        msg = (
            "variables needed for gradient computation has been modified by an inplace"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)

        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)

        self.assertClonedLenEqual(ctx, 0)
        self.assertTrue(torch.allclose(a * 2, da))

    def test_save_base_and_modify_view(self):
        # [FUNCTION DOC] Check saving a base and modifying a view inside the context gives the
        # right gradient and no leftover clones.
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:1]
            out = b**2
            # modify the view
            c *= 10
            # self.assertClonedLenEqual(ctx, 1)
            out.sum().backward()
            self.assertClonedLenEqual(ctx, 0)

        self.assertClonedLenEqual(ctx, 0)
        self.assertTrue(torch.allclose(a * 2, a.grad))

    def test_save_view_modify_base(self):
        # [FUNCTION DOC] Check saving a view and modifying the base in place inside the context
        # gives the right gradient.
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:]
            out = (c**2).sum()
            b *= 2
            out.backward()
            self.assertTrue(torch.allclose(a * 2, a.grad))

    def test_double_backward(self):
        # [FUNCTION DOC] Check double backward inside the context after in-place modification of a
        # saved tensor gives the expected second derivative and cleans up clones.
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            out = (b**2).sum()
            b.sin_()
            torch.autograd.grad(out, a, create_graph=True)
            (da,) = torch.autograd.grad(out, a, create_graph=True)
            (d2a,) = torch.autograd.grad(da.sum(), a)

        self.assertTrue(torch.allclose(torch.ones_like(a) * 2, d2a))
        self.assertClonedLenEqual(ctx, 0)

    def test_saved_but_not_anymore(self):
        # [FUNCTION DOC] Check no clone is made when the saved tensor was released before the
        # in-place op, and a clone is made and then dropped when a saved tensor is still alive.
        # Make sure we don't clone if the tensor was once saved, but
        # by the time we do in-place, it is no longer saved
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            out = (a**2).sum()
            self.assertTIDMapLenEqual(ctx, 1)
            self.assertClonedLenEqual(ctx, 0)
            out.backward()
            a.sin_()
            self.assertClonedLenEqual(ctx, 0)
            out = (a**2).sum()
            a.sin_()
            self.assertClonedLenEqual(ctx, 1)
            del out
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_many_times(self):
        # [FUNCTION DOC] Check a tensor saved several times by different ops is cloned only once
        # and the clone is dropped when the ops are deleted.
        # We should only clone once
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            b = a**2
            c = a**2
            a.sin_()
            self.assertClonedLenEqual(ctx, 1)
            del b, c
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_different_versions(self):
        # [FUNCTION DOC] Check each version of a tensor saved at different times is cloned
        # separately and dropped as its users are deleted.
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            b = a**2
            a.sin_()
            c = a**2
            a.sin_()
            self.assertClonedLenEqual(ctx, 2)
            del b
            self.assertClonedLenEqual(ctx, 1)
            del c
            self.assertClonedLenEqual(ctx, 0)

    def test_with_math_views(self):
        # [FUNCTION DOC] Check conjugate views work with the context for in-place changes on the
        # base and on the view.
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            b = a.conj()
            out = (b**2).sum()
            a.sin_()
            out.abs().backward()

            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            b = a.conj()
            out = (b**2).sum()
            # in this case, it is no longer a view it seems
            b.sin_()
            out.abs().backward()

    def test_with_out_variant(self):
        # [FUNCTION DOC] Check an out variant overwriting a saved input triggers a clone that is
        # released after backward.
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.tensor([1.0], requires_grad=True)
            b = torch.tensor([1.0])
            c = torch.tensor([2.0])
            out = a * b
            self.assertTIDMapLenEqual(ctx, 1)
            torch.sin(c, out=b)
            self.assertClonedLenEqual(ctx, 1)
            out.backward()
            self.assertClonedLenEqual(ctx, 0)

    def test_backward_out_of_context(self):
        # [FUNCTION DOC] Check backward outside the context, or inside a different context, raises
        # AssertionError saying so.
        # Out of context
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            out = (a**2).sum()

        msg = "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
        with self.assertRaisesRegex(AssertionError, msg):
            out.backward()

        # Different context
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            out = (a**2).sum()

        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            with self.assertRaisesRegex(AssertionError, msg):
                out.backward()

    def test_disallow_nesting(self):
        # [FUNCTION DOC] Check nesting allow_mutation_on_saved_tensors contexts
        # raises RuntimeError.
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            msg = "allow_mutation_on_saved_tensors contexts cannot be nested"
            with self.assertRaisesRegex(RuntimeError, msg):
                with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
                    pass

    def test_inplace_foreach(self):
        # [FUNCTION DOC] Check in-place foreach ops on saved results work inside the context.
        with torch.autograd.graph.allow_mutation_on_saved_tensors():
            a = [
                torch.tensor(1.0, requires_grad=True),
                torch.tensor(1.0, requires_grad=True),
            ]
            b = torch._foreach_exp(a)
            torch._foreach_add_(b, 1)
            (b[0] + b[1]).backward()

        self.assertEqual([a[0].grad, a[1].grad], torch._foreach_exp(a))


class TestAutogradInferenceMode(TestCase):
    # [CLASS DOC] Tests of inference mode semantics: inference-tensor status of results,
    # interaction with normal tensors, views, in-place ops, version counters and error messages.
    def _is_inference_tensor(self, tensor):
        # [FUNCTION DOC] HELPER: Return whether accessing the version counter raises the
        # inference-tensor error (true) or not (false).
        try:
            err_msg = "Inference tensors do not track version counter"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                tensor._version
            return True
        except AssertionError as e:
            return False

    def test_inference_mode_context_manager(self):
        # [FUNCTION DOC] Check the inference mode flag toggles through nested context managers.
        self.assertFalse(torch.is_inference_mode_enabled())
        with torch.inference_mode():
            self.assertTrue(torch.is_inference_mode_enabled())
            with torch.inference_mode(False):
                self.assertFalse(torch.is_inference_mode_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_inference_mode_decorator(self):
        # [FUNCTION DOC] Check inference_mode as a decorator with positional or keyword mode, with
        # or without a mode value: the mode inside is as requested, outputs are inference tensors
        # only when enabled, and requires_grad follows.
        def func(x):
            # [FUNCTION DOC] HELPER: Assert the expected inference mode setting and return the
            # square of the input.
            self.assertEqual(torch.is_inference_mode_enabled(), mode)
            return x * x

        for mode, use_kwarg in product((True, False, None), (True, False)):
            if mode is None:
                if use_kwarg:
                    decorated = torch.inference_mode(mode=func)
                else:
                    decorated = torch.inference_mode(func)
                mode = True
            else:
                if use_kwarg:
                    decorated = torch.inference_mode(mode=mode)(func)
                else:
                    decorated = torch.inference_mode(mode)(func)

            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                d = decorated(c)
                self.assertTrue(not mode or torch.is_inference(d))
                self.assertEqual(d.requires_grad, requires_grad and not mode)

    def test_inference_mode_tensor_creation(self):
        # [FUNCTION DOC] Check tensors created in inference mode are inference tensors, regardless
        # of requires_grad requests.
        with torch.inference_mode():
            # new tensors created through constructors are inference tensors
            c = torch.ones(1, 2, 3)
            self.assertFalse(c.requires_grad)
            self.assertTrue(torch.is_inference(c))

            # requires_grad doesn't change inference tensor behavior in InferenceMode
            tmp = torch.ones(1, 2, 3, requires_grad=True)
            self.assertTrue(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))

            tmp = torch.ones(1, 2, 3).requires_grad_(False)
            self.assertFalse(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))

    def test_inference_mode_existing_autograd_session(self):
        # [FUNCTION DOC] Check modifying in inference mode a tensor saved for backward outside of
        # it makes backward raise RuntimeError about in-place modification.
        s = torch.ones(1, 2, 3, requires_grad=True)
        a = s.clone()

        # `a` gets saved outside of inference mode
        out = a * a
        with torch.inference_mode():
            a.add_(2)

        self.assertFalse(torch.is_inference(a))
        # tensors created outside of inference mode aren't
        # inference tensors, so they will still have their
        # version counters tracked
        err_msg = (
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation"
        )
        with self.assertRaisesRegex(RuntimeError, err_msg):
            out.backward(torch.ones_like(out))

    def test_inference_mode_inf_tensor_in_inf_mode_functional_op(self):
        # [FUNCTION DOC] Check functional ops on inference tensors in inference mode give
        # inference tensors that do not require grad.
        def functional_op(x):
            # [FUNCTION DOC] HELPER: Square the input.
            return x * x

        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # performing a non-view operation produces a inference tensor
                # that does not require grad
                func_out = functional_op(c)
                self.assertTrue(torch.is_inference(func_out))
                self.assertFalse(func_out.requires_grad)

    def test_inference_mode_inf_tensor_in_inf_mode_inplace_op(self):
        # [FUNCTION DOC] Check in-place ops (and manual-kernel ones) on inference tensors in
        # inference mode keep them inference tensors with unchanged requires_grad.
        @torch.inference_mode()
        def run_test(fn):
            # [FUNCTION DOC] HELPER: Under inference mode apply the in-place function to
            # an inference tensor and check it is still an inference tensor with the
            # same requires_grad.
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # after performing inplace operation, tensor is still
                # an inference tensor
                fn(c)
                self.assertTrue(torch.is_inference(c))
                self.assertEqual(c.requires_grad, requires_grad)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

        # inplace ops with manual kernel for ADInplaceOrView key in VariableTypeManual.cpp
        run_test(lambda x: x.resize_(1, 2))
        run_test(lambda x: x.resize_as_(torch.ones(1, 2)))
        run_test(lambda x: x.copy_(torch.ones(1, 2, 3)))

    def test_inference_mode_inf_tensor_in_inf_mode_view_op(self):
        # [FUNCTION DOC] Check view ops on inference tensors in inference mode give inference
        # tensors without grad.
        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # perform view operation produces inference tensor
                # that does not require grad
                view_out = c.view(-1)
                self.assertTrue(torch.is_inference(view_out))
                self.assertFalse(view_out.requires_grad)

    def test_inference_mode_inf_tensor_in_normal_mode_functional_op(self):
        # [FUNCTION DOC] Check functional ops on inference tensors in normal mode give normal leaf
        # tensors not requiring grad.
        def functional_op(x):
            # [FUNCTION DOC] HELPER: Square the input.
            return x * x

        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

        func_out = functional_op(c)
        self.assertFalse(torch.is_inference(func_out))
        self.assertFalse(func_out.requires_grad)
        self.assertTrue(func_out.is_leaf)

    @skipIfTorchDynamo(
        "exception from ill-formed graph module is not propagated with eager_noexcept"
    )
    def test_inference_mode_inf_tensor_in_normal_mode_inplace_op(self):
        # [FUNCTION DOC] Check in-place ops on inference tensors in normal mode raise RuntimeError
        # when the tensor does not require grad. Skipped under dynamo.
        def run_test(fn):
            # [FUNCTION DOC] HELPER: For each requires_grad value, apply the in-place function
            # outside inference mode to an inference tensor and check the expected error for the
            # non-grad case.
            for requires_grad in (False, True):
                with torch.inference_mode():
                    c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                if requires_grad:
                    # leaf variable that requires grad is being used in an inplace
                    # operation when requires_grad=True
                    pass
                else:
                    err_msg = "Inplace update to inference tensor outside InferenceMode"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(c)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_inf_tensor_in_normal_mode_view_op(self):
        # [FUNCTION DOC] Check view ops on inference tensors in normal mode give inference tensors
        # that are leaf non-views not requiring grad.
        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            out = c.view(-1)
            self.assertTrue(torch.is_inference(out))
            self.assertFalse(out.requires_grad)
            self.assertFalse(out._is_view())
            self.assertTrue(out.is_leaf)

    def test_normal_tensor_inplace_output_in_inference_mode(self):
        # [FUNCTION DOC] Check in-place ops on normal tensors inside inference mode keep them
        # normal tensors with unchanged requires_grad, including follow-up views.
        def run_test(fn):
            # [FUNCTION DOC] HELPER: Run the in-place function twice inside inference mode on a
            # clone, and a view of it, checking inference status and requires_grad.
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                    # inplace -> inplace
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                    # inplace -> inplace -> view
                    view_out = a.view(-1)
                    self.assertFalse(torch.is_inference(view_out))
                    self.assertEqual(view_out.requires_grad, requires_grad)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_inplace_output_in_normal_mode(self):
        # [FUNCTION DOC] Check in-place ops on normal tensors inside and outside inference mode
        # keep them normal tensors.
        # KNOWN ISSUE: the calls that run the helper are indented inside the helper function body,
        # so the test never calls it from the test body and nothing is actually checked (calling
        # it would also recurse).
        def run_test(fn):
            # [FUNCTION DOC] HELPER: Run in-place function inside and outside inference mode on a
            # clone, checking inference status and requires_grad; its body also contains the
            # (misplaced) calls to itself.
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace
                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace -> view
                view_out = a.view(-1)
                self.assertFalse(torch.is_inference(view_out))
                self.assertEqual(view_out.requires_grad, requires_grad)
            run_test(lambda x: x.add_(2))
            run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_view_output_in_inference_mode(self):
        # [FUNCTION DOC] Check view ops on normal tensors inside inference mode give normal
        # differentiable views with the same requires_grad, and in-place ops on them share the
        # version counter.
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())

                # view -> view
                tmp = out.view(-1)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                self.assertTrue(tmp._is_view())
                self.assertTrue(tmp.is_leaf)

                # view -> view -> inplace
                self.assertTrue(torch.is_inference_mode_enabled())
                tmp.add_(2)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                # Accessing is_leaf in python tries to update grad_fn and raises:
                # A view was created in inference mode and its base or
                # another view of its base has been modified inplace in normal mode
                # tmp.is_leaf
                self.assertEqual(a._version, tmp._version)

    def test_normal_tensor_view_output_in_normal_mode(self):
        # [FUNCTION DOC] Check views created in inference mode used in normal mode: functional ops
        # work, in-place ops raise RuntimeError when requires_grad else work, and views of them
        # are normal tensors.
        def functional_op(x):
            # [FUNCTION DOC] HELPER: Square the input.
            return x * x

        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())
                self.assertTrue(out.is_leaf)

            tmp = functional_op(out)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

            if requires_grad:
                err_msg = (
                    "A view was created in inference mode and is being modified inplace"
                )
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    out.add_(2)
            else:
                out.add_(2)

            tmp = out.view(2, 3)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

    def test_mix_inference_and_normal_tensor_functional_op(self):
        # [FUNCTION DOC] Check functional ops mixing an inference tensor and a normal tensor: add
        # is fine and supports backward, but multiplication needing to save the inference tensor
        # raises RuntimeError.
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            # add is safe since it doesn't save any variable for backward
            out = c.add(s)
            self.assertFalse(torch.is_inference(out))
            self.assertEqual(out.requires_grad, requires_grad)
            if requires_grad:
                # leaf inference tensor with requires_grad=True can still have gradient
                out.backward(torch.ones_like(out))
                self.assertEqual(c.grad, torch.ones_like(c))

            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    c * s

                # TODO: Test this with an autograd.Function when it works
                #       stack stopped capturing a TensorList input
                # # inference tensor in TensorList input
                # inputs = [s, c]
                # with self.assertRaisesRegex(RuntimeError, err_msg):
                #     torch.stack(inputs)

    def test_mix_inference_and_normal_tensor_inplace_op(self):
        # [FUNCTION DOC] Check in-place and out ops mixing an inference tensor and a normal
        # tensor: saving errors when grad is required, out arguments error as expected otherwise.
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            self.assertTrue(torch.is_inference(c))
            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mul_(c)

                # inference tensor in TensorList input
                err_msg = (
                    "out=... arguments don't support automatic differentiation, "
                    "but one of the arguments requires grad"
                )
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)
            else:
                a.mul_(c)
                err_msg = "Inplace update to inference tensor outside InferenceMode is not allowed"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)

    def test_mix_inference_and_normal_tensor_view_op(self):
        # [FUNCTION DOC] Check view_as with mixed inference and normal tensors gives inference or
        # normal results by the receiver.
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            # view_as is a composite op which calls view with only one
            # tensor argument. So there isn't a mixed inference and normal
            # tensor inputs for view ops
            tmp1 = c.view_as(s)
            self.assertTrue(torch.is_inference(tmp1))
            self.assertFalse(tmp1.requires_grad)

            # this is fine since its equivalent as s.view(c.sizes()) which
            # isn't a mixed input scenario
            tmp2 = s.view_as(c)
            self.assertFalse(torch.is_inference(tmp2))
            self.assertEqual(tmp2.requires_grad, requires_grad)

    def test_inference_mode_handle_direct_view_on_rebase(self):
        # [FUNCTION DOC] Check rebase of a view created in inference mode when the view itself is
        # modified in place: error if requires_grad else fine.
        def run_test(fn):
            # [FUNCTION DOC] HELPER: Make a view in inference mode and apply the in-place function
            # with the expected error behavior.
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    view_out = a.view_as(a)

                if requires_grad:
                    err_msg = "A view was created in inference mode and is being modified inplace"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(view_out)
                else:
                    fn(view_out)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_handle_indirect_view_on_rebase(self):
        # [FUNCTION DOC] Check rebase of a view created in inference mode when the base is
        # modified in place: accessing the view grad_fn raises RuntimeError if requires_grad
        # else works.
        def run_test(fn):
            # [FUNCTION DOC] HELPER: Make a view in inference mode, modify the base in place, and
            # access the view grad_fn with the expected error behavior.
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    view_out = a.view(-1)

                fn(a)
                if requires_grad:
                    err_msg = "A view was created in inference mode and its base or another view "
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        view_out.grad_fn
                else:
                    view_out.grad_fn

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))


NUM_GPU_CYCLES_IN_ONE_SEC = 2_000_000_000


@contextlib.contextmanager
def _set_device_index(target_device):
    # [FUNCTION DOC] HELPER: Context manager that sets the current accelerator device index and
    # restores the previous index on exit, even on error.
    orig_device = torch.accelerator.current_device_index()
    try:
        torch.accelerator.set_device_index(target_device)
        yield
    finally:
        torch.accelerator.set_device_index(orig_device)


def _sleep_if_cuda(cycles):
    # [FUNCTION DOC] HELPER: Sleep the given number of cycles when the current accelerator is
    # CUDA, otherwise do nothing and return None.
    if "cuda" == torch.accelerator.current_accelerator().type:
        return torch.cuda._sleep(cycles)
    else:
        # Update this if non-cuda accelerators support something like sleep
        return


def _get_device_name(idx):
    # [FUNCTION DOC] HELPER: Return the accelerator type with an index as a device string.
    return f"{torch.accelerator.current_accelerator().type}:{idx}"


# Although this is written to be generic over all accelerators, non-cuda accelerators
# are not fully tested since sleep is only supported on cuda.
class TestAutogradStreamSynchronization(TestCase):
    # [CLASS DOC] Tests for how autograd synchronizes accelerator streams between forward and
    # backward, written generically but only fully exercised on CUDA (sleep is CUDA only).
    def get_default_streams(self, num_devices=1):
        # [FUNCTION DOC] HELPER: Return a tuple with the default stream of each of the
        # first devices.
        out = []
        for i in range(num_devices):
            with _set_device_index(i):
                acc = torch.accelerator.current_accelerator()
                out.append(torch.get_device_module(acc).default_stream())
        return tuple(out)

    def synchronize_all_devices(self, num_devices=1):
        # [FUNCTION DOC] HELPER: Synchronize the accelerator of each of the first devices.
        for i in range(num_devices):
            torch.accelerator.synchronize(i)

    def assert_all_streams_default(self, num_devices=1):
        # [FUNCTION DOC] HELPER: Assert that the current stream of each of the first devices is
        # its default stream.
        # Sanity check
        default_streams = self.get_default_streams(num_devices)
        for i in range(num_devices):
            with _set_device_index(i):
                acc = torch.accelerator.current_accelerator()
                # Do this instead of using torch.accelerator.current_stream(i)
                # Otherwise, e.g. in the case of cuda, we'd be trying to compare
                # torch.cuda.Stream with torch.Stream
                self.assertEqual(
                    torch.get_device_module(acc).current_stream(), default_streams[i]
                )

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    def test_consumer_to_single_producer_case_2_correctness(self, device):
        # [FUNCTION DOC] Stream sync check, consumer on stream s0 and producer plus its gradient
        # on stream s1 of the same device: the engine must make the producer wait for the consumer
        # so the gradient equals 4. Runs twice (warmup). Skipped on CPU, CUDA non-default stream
        # runs, and expected to fail on MPS.
        if device == "cpu":
            self.skipTest("requires accelerator")

        #                          Device    Stream
        # Consumer (MulBackward):  cuda:0    s0
        # Producer              :  cuda:0    s1
        # Gradient              :  cuda:0    s1
        class Producer(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward sleeps then adds one to a clone of
            # the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Clone the gradient, sleep on CUDA for half a second worth
                # of cycles, add one in place and return it.
                out = gO.clone()
                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                out.add_(1)
                return out

        def test():
            # [FUNCTION DOC] HELPER: Build the consumer on one stream and the producer on a second
            # stream waiting for it, run backward with multithreading off and check the gradient.
            self.synchronize_all_devices()
            self.assert_all_streams_default()

            with torch.Stream(0) as s0:
                a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
                b = a * 2

            with torch.Stream(0) as s1:
                s1.wait_stream(s0)
                out = Producer.apply(b)

                with torch.autograd.grad_mode.set_multithreading_enabled(False):
                    out.sum().backward()

            self.synchronize_all_devices()

            # Expected result: a.grad = (grad_out + 1) * 2 = 4
            self.assertEqual(a.grad, torch.full_like(a, 4))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    def _test_consumer_to_single_producer_case_3_correctness(
        self, non_default_ambient_stream
    ):
        # [FUNCTION DOC] HELPER: Cross-device stream sync check where the producer is on a second
        # device and the gradient comes from the first device default stream, with the ambient
        # stream at backward time either default or non-default; checks the gradient equals 4.
        #                          Device    Stream
        # Consumer (MulBackward):  cuda:0    s0
        # Producer              :  cuda:1    cuda:1 default
        # Gradient              :  cuda:0    cuda:0 default
        class Producer(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function on device 1 whose backward moves the gradient to device
            # 0, sleeps, adds one and syncs its canonical stream.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Remember the current stream of device 1 as the node
                # stream and move the input to device 1.
                # The node's canonical stream is the current stream
                # of the device of the first output.
                ctx.node_stream = torch.accelerator.current_stream(1)
                return x.to(_get_device_name(1))

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Move the gradient to device 0, sleep there, add one
                # in place, make the node stream wait for device 0's current stream and return
                # the gradient.
                out = gO.to(_get_device_name(0))
                with _set_device_index(0):
                    _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                # It's the node's responsibility to sync back to its canonical stream.
                out.add_(1)
                ctx.node_stream.wait_stream(torch.accelerator.current_stream(0))
                return out

        def test():
            # [FUNCTION DOC] HELPER: Build the consumer on a side stream, make the default stream
            # wait, apply the producer, run backward under the chosen ambient stream with
            # multithreading off, and check the gradient.
            self.synchronize_all_devices(2)
            self.assert_all_streams_default(2)

            (default_stream_0,) = self.get_default_streams()

            # Ensure consumer node happens on non-default stream so that
            # when FuncBackward produces a gradient on a default stream
            # a sync is necessary.
            with torch.Stream(0) as s0:
                a = torch.ones(256, 256, requires_grad=True, device="cuda")
                b = a * 2

            default_stream_0.wait_stream(s0)
            out = Producer.apply(b)

            def call_backward(x):
                # [FUNCTION DOC] HELPER: Run backward on the sum with multithreading disabled.
                with torch.autograd.grad_mode.set_multithreading_enabled(False):
                    x.sum().backward()

            if non_default_ambient_stream:
                with torch.Stream(0) as s1:
                    s1.wait_stream(default_stream_0)
                    call_backward(out)
            else:
                call_backward(out)

            self.synchronize_all_devices(2)

            # Expected result: a.grad = (grad_out + 1) * 2 = 4
            self.assertEqual(a.grad, torch.full_like(a, 4))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_single_producer_case_3_correctness_non_default_ambient_stream(
        self, device
    ):
        # [FUNCTION DOC] Cross-device stream sync check (case 3) with a non-default ambient stream
        # during backward. Needs two accelerators; skipped on CPU and CUDA non-default stream
        # runs; expected failure on MPS.
        if device == "cpu":
            self.skipTest("requires accelerator")
        self._test_consumer_to_single_producer_case_3_correctness(
            non_default_ambient_stream=True
        )

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_single_producer_case_3_correctness(self, device):
        # [FUNCTION DOC] Cross-device stream sync check (case 3) with the default ambient stream
        # during backward. Needs two accelerators.
        if device == "cpu":
            self.skipTest("requires accelerator")
        self._test_consumer_to_single_producer_case_3_correctness(
            non_default_ambient_stream=False
        )

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_single_producer_case_4_correctness(self, device):
        # [FUNCTION DOC] Stream sync check, consumer on device 0 default stream, producer and
        # gradient on device 1 stream s1: the engine must sync the gradient to the consumer,
        # result is 2. Needs two accelerators.
        if device == "cpu":
            self.skipTest("requires accelerator")

        #           Device    Stream
        # Consumer: cuda:0    cuda:0 default
        # Producer: cuda:1    s1
        # Gradient: cuda:1    s1
        class Producer(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward sleeps and adds one to a clone of
            # the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Clone the gradient, sleep on CUDA and return it with one
                # added in place.
                out = gO.clone()
                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                return out.add_(1)

        class Consumer(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function with two outputs (a clone on the first device and a
            # copy on the second) whose backward returns the second gradient moved to device 0.
            # In the multi-output case, the node's canonical device and stream correspond to
            # that of its first output. This is required to induce cases 4/5.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone and a copy moved to device 1.
                return x.clone(), x.to(_get_device_name(1))

            @staticmethod
            def backward(ctx, gO_0, gO_1):
                # [FUNCTION DOC] HELPER: Ignore the first gradient and return the second gradient
                # moved to device 0.
                # gO_1 is on s1, but we're currently doing compute in cuda:1 default
                # It's the user's responsibility to sync to consumer (.to() should do this
                # already.)
                # Things would work out if the engine sync'd s1 with consumer.
                # Ignore grad wrt first arg because we don't use it.
                return gO_1.to(_get_device_name(0))

        def test():
            # [FUNCTION DOC] HELPER: Apply the consumer, run the producer on a stream of device 1
            # and run backward with multithreading off, then check the gradient equals 2.
            self.synchronize_all_devices(2)
            self.assert_all_streams_default(2)

            _, default_stream_1 = self.get_default_streams(2)
            a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
            _unused, b = Consumer.apply(a)

            with torch.Stream(1) as s1:
                s1.wait_stream(default_stream_1)
                out = Producer.apply(b)

                with torch.autograd.grad_mode.set_multithreading_enabled(False):
                    out.sum().backward()

            self.synchronize_all_devices(2)

            # Expected result: a.grad = grad_out + 1 = 2
            self.assertEqual(a.grad, torch.full_like(a, 2))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_multi_producer_case_4_correctness(self, device):
        # [FUNCTION DOC] Stream sync check with two producers on different streams feeding one
        # consumer: the accumulation stream must wait for the slow producer so the result is 8
        # rather than 12. Needs two accelerators.
        if device == "cpu":
            self.skipTest("requires accelerator")

        #             Device    Stream
        # Consumer  : cuda:0    cuda:0 default
        #
        # Producer 1: cuda:1    s1
        # Gradient 1: cuda:1    s1
        #
        # Producer 2: cuda:1    s2
        # Gradient 2: cuda:1    s2
        #
        # Accumulation stream: s2 since it is scheduled first
        class ProducerFast(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward doubles a clone of the gradient quickly.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Return a clone of the gradient doubled.
                out = gO.clone()
                return out * 2

        class ProducerSlow(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward sleeps and then doubles a clone of the
            # gradient in place.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Clone the gradient, sleep on CUDA, double in place and
                # return it.
                out = gO.clone()
                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                return out.mul_(2)

        class Consumer(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function with two outputs whose backward waits on its canonical
            # stream and returns the doubled second gradient moved to device 0.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Remember the current stream and return a clone and a copy
                # on device 1.
                ctx.node_stream = torch.accelerator.current_stream(x.device)
                return x.clone(), x.to(_get_device_name(1))

            @staticmethod
            def backward(ctx, gO_0, gO_1):
                # [FUNCTION DOC] HELPER: Make the stream of the second gradient wait for the node
                # stream and return the doubled gradient moved to device 0.
                torch.accelerator.current_stream(gO_1.device).wait_stream(
                    ctx.node_stream
                )
                return (gO_1 * 2).to(_get_device_name(0))

        def test():
            # [FUNCTION DOC] HELPER: Apply the consumer, run two producers on two streams of
            # device 1, merge streams, run backward and check the gradient equals 8.
            self.synchronize_all_devices(2)
            self.assert_all_streams_default(2)

            default_stream_0, default_stream_1 = self.get_default_streams(2)

            a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
            _unused, b = Consumer.apply(a)

            with torch.Stream(1) as s1:
                s1.wait_stream(default_stream_1)
                out1 = ProducerFast.apply(b)

            with torch.Stream(1) as s2:
                s2.wait_stream(default_stream_1)
                out2 = ProducerSlow.apply(b)

            default_stream_1.wait_stream(s1)
            default_stream_1.wait_stream(s2)

            with torch.autograd.grad_mode.set_multithreading_enabled(False):
                (out1 + out2).sum().backward()

            self.synchronize_all_devices(2)

            # If the accumulation stream does not wait for the slow producer stream
            # the in-place mul-by-2 is performed on the accumulated buffer AFTER
            # ProducerFast has already accumulated!
            #
            # Correct: (1.mul_(2) + 2) * 2 = 8
            # Incorrect: (1 + 2).mul_(2) * 2 = 12
            self.assertEqual(a.grad, torch.full_like(a, 8))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    # This test may spuriously fail on non-cuda accelerators (since we won't
    # be calling sleep)
    @onlyCUDA
    @skipCUDANonDefaultStreamIf(True)
    def test_side_stream_backward_overlap(self, device):
        # [FUNCTION DOC] Check that a side-stream backward overlaps with the main backward: event
        # timings show the main backward starts before the side backward ends. CUDA only; non-CUDA
        # accelerators may fail spuriously.
        # In case 2/3, we would designate the consumer as the accumulation
        # stream and naively, one might have the consumer wait for the producer
        # as soon as we've added to the InputBuffer the first time.
        #
        # However, in the case where the stream of the consumer also happens to
        # be the stream of the producer, this is suboptimal because it would
        # prevent the computation of the two producers from being overlapped.
        # what you really want to do is to have that sync between the producer
        # and consumer to be delayed until right before the accumulation.
        # Note that this doesn't address N=3, but the side-stream N=2 case is
        # the common case.
        events = {
            "main_backward_start": None,
            "side_backward_start": None,
            "side_backward_end": None,
        }

        class Main(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function whose backward records an event when
            # it starts.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Record a timing event in the shared events mapping and
                # return the gradient.
                # Record when main backward starts
                evt = torch.Event(enable_timing=True)
                evt.record()
                events["main_backward_start"] = evt
                return gO

        class Side(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function whose backward records start and end events
            # around a long sleep.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Record a start event, sleep, clone the gradient, record
                # an end event and return the clone.
                evt = torch.Event(enable_timing=True)
                evt.record()
                events["side_backward_start"] = evt

                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                result = gO.clone()

                evt = torch.Event(enable_timing=True)
                evt.record()
                events["side_backward_end"] = evt
                return result

        def populate_events():
            # [FUNCTION DOC] HELPER: Run forward with the main and side branches (the side on its
            # own stream), then backward with multithreading off, and synchronize.
            self.synchronize_all_devices()
            self.assert_all_streams_default()

            (default_stream_0,) = self.get_default_streams()

            a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
            b = a.clone()  # not a leaf, does it matter?

            evt = torch.Event()
            evt.record()

            # Overlap during forward
            c_main = Main.apply(b)

            with torch.Stream(0) as s0:
                s0.wait_event(evt)
                c_side = Side.apply(b)

            default_stream_0.wait_stream(s0)

            with torch.autograd.grad_mode.set_multithreading_enabled(False):
                (c_main + c_side).sum().backward()

            self.synchronize_all_devices()

        def check_ordering():
            # [FUNCTION DOC] HELPER: Assert the side backward took positive time and the main
            # start preceded the side end.
            # Sanity check: side backward's end happens after start
            self.assertTrue(
                events["side_backward_start"].elapsed_time(events["side_backward_end"])
                > 0
            )
            # Overlap check: side's backward starts before side backward ends
            self.assertTrue(
                events["main_backward_start"].elapsed_time(events["side_backward_end"])
                > 0
            )

        # Warmup
        for _ in range(2):
            populate_events()

        # Reset events (not really necessary but OK)
        events["side_backward_start"] = None
        events["side_backward_end"] = None
        events["main_backward_start"] = None

        # Test
        populate_events()
        check_ordering()

    @expectedFailureMPS
    def test_warn_on_accumulate_grad_stream_mismatch_flag(self, device):
        # [FUNCTION DOC] Check the flag controlling warnings when AccumulateGrad runs on a
        # different stream than the one the tensor was created on: warns by default only when the
        # gradient accumulator is kept alive, and not when suppressed. Skipped on CPU, expected
        # failure on MPS.
        if device == "cpu":
            self.skipTest("requires accelerator")

        def do_test(suppress_warn, keep_grad_acc):
            # [FUNCTION DOC] HELPER: Run the scenario with the suppress flag toggled around it and
            # compare whether the warning appeared with the expectation.
            def _test():
                # [FUNCTION DOC] HELPER: Create a leaf on one stream (optionally keeping its
                # accumulator alive), sum it on another stream, run backward, and return whether
                # any warning mentions the flag setter.
                with set_warn_always_context(True):
                    with warnings.catch_warnings(record=True) as warns:
                        warnings.simplefilter("always")

                        with torch.Stream(0) as s0:
                            a = torch.ones(8, 8, device=device, requires_grad=True)
                            if keep_grad_acc:
                                # create grad_acc under s1 and keep alive with b
                                b = a.clone()

                        with torch.Stream(0) as s1:
                            s1.wait_stream(s0)
                            c = a.sum()

                        c.backward()

                    filter_str = "set_warn_on_accumulate_grad_stream_mismatch"
                    return sum([filter_str in str(w.message) for w in warns]) > 0

            if suppress_warn:
                try:
                    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(
                        False
                    )
                    actual_warn = _test()
                finally:
                    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(
                        True
                    )
            else:
                actual_warn = _test()

            expect_warn = not suppress_warn and keep_grad_acc
            self.assertEqual(actual_warn, expect_warn)

        # Warn by default
        self.assertTrue(torch._C._warn_on_accumulate_grad_stream_mismatch())

        for suppress_warn in (True, False):
            for keep_grad_acc in (True, False):
                do_test(suppress_warn=suppress_warn, keep_grad_acc=keep_grad_acc)


class TestMultithreadAutograd(TestCase):
    # [CLASS DOC] Tests of autograd under multiple Python threads: exception propagation, shared
    # inputs, hooks, saved tensor hooks with DataParallel, retain_graph requirements, TorchScript
    # fork and join, tracebacks and the multithreading flag.
    def _run_py_multithread_fn(
        self, fn, args=(), num_threads=10, kwargs=None, pass_idx=False
    ):
        # [FUNCTION DOC] HELPER: Run a function in several threads that propagate exceptions on
        # join; optional extra arguments, thread count, keyword arguments and passing the thread
        # index as first argument.
        class PropagatingThread(threading.Thread):
            # [CLASS DOC] HELPER: Thread subclass that stores an exception raised in the target
            # and returns the result.
            """Helper class to propagate exception from child
            thread to main thread on join.

            Reference: https://stackoverflow.com/a/31614591/5602957
            """

            def run(self):
                # [FUNCTION DOC] HELPER: Run the target, saving its return value, or the
                # exception raised.
                self.exception = None
                try:
                    self.ret = super().run()
                except Exception as e:
                    self.exception = e

            def join(self, timeout=None):
                # [FUNCTION DOC] HELPER: Join the thread and re-raise the stored exception in the
                # caller, otherwise return the result.
                super().join(timeout)
                if self.exception:
                    raise self.exception from self.exception
                return self.ret

        threads = []
        for idx in range(num_threads):
            p = PropagatingThread(target=fn, args=((idx, *args) if pass_idx else args))
            p.start()
            threads.append(p)

        for p in threads:
            p.join()

    def test_multithreaded_exception_propagation(self):
        # [FUNCTION DOC] Check that an exception in a child thread is propagated to the main
        # thread by the helper.
        # Test whether exception in child thread
        # are propagated to main thread.
        def fn():
            # [FUNCTION DOC] HELPER: Fail the test with a false assertion.
            self.assertTrue(False)

        with self.assertRaises(AssertionError):
            self._run_py_multithread_fn(fn)

    def test_simple_backward(self):
        # [FUNCTION DOC] Check separate backward passes in separate threads on private inputs all
        # give the right gradient.
        # simple multithreaded backward that create threads in the beginning of training
        # and everything else is training separately, i.e. inputs, operations, etc.
        def train_fn():
            # [FUNCTION DOC] HELPER: Build private inputs, run backward and check the gradient.
            x = torch.ones(5, 5, requires_grad=True)
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()
            self.assertEqual(x.grad, x + 3.5)

        self._run_py_multithread_fn(train_fn)

    def test_simple_backward_same_input(self):
        # [FUNCTION DOC] Check backward from ten threads on a shared input accumulates ten times
        # the gradient, and grad from threads gives the same result in each.
        # simple multithreaded backward with only shared inputs (i.e. This is common
        # for things like Hogwild multithreaded training with multiple CPU threads)
        def train_fn_backward(x):
            # [FUNCTION DOC] HELPER: Run a backward pass on a shared input.
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()

        x = torch.ones(5, 5, requires_grad=True)
        self._run_py_multithread_fn(train_fn_backward, (x,))
        # Since we are calling backward from multiple threads
        # and all threads share the same input, when we do backward
        # concurrently, different backwards will all accumulate to
        # the same .grad for each input, and the gradients should
        # be equal to num_threads * gradient
        self.assertEqual(x.grad, 10 * (x + 3.5))

        def train_fn_grad(x):
            # [FUNCTION DOC] HELPER: Compute the gradient with the functional grad API and
            # assert it.
            y = (x + 3) * (x + 4) * 0.5
            grads = torch.autograd.grad(y.sum(), x)
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0], x + 3.5)

        # since we use functional grad() api, gradients will not
        # be accumulate to the same place and should be the same
        self._run_py_multithread_fn(train_fn_grad, (x,))

    def test_multi_grad_all_hooks(self):
        # [FUNCTION DOC] Check multi-grad hooks in all mode fire once per concurrent backward,
        # with the same mask, and a partially failing backward leaves counts as expected.
        # Multihooks should behave independently per execution of backward
        # Test that the hook fired the number of times we ran backward
        # even if those executions occur concurrently on different threads
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        res = None
        count = [0]
        hook_lock = threading.Lock()

        def hook(grads):
            # [FUNCTION DOC] HELPER: Under a lock count the call and check the mask of defined
            # gradients matches the first recorded mask.
            nonlocal res
            with hook_lock:
                count[0] += 1
                grad_is_none = [g is not None for g in grads]
                if res is None:
                    res = grad_is_none
                else:
                    self.assertEqual(res, grad_is_none)

        handle = torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)

        out = (t2 * t3).sum()

        def backward_retain_graph(out, t2, t3):
            # [FUNCTION DOC] HELPER: Run backward with inputs and retain_graph.
            out.backward(inputs=(t2, t3), retain_graph=True)

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        self.assertEqual(count[0], 5)
        self.assertEqual(res, [False, True, True, False])

        # Leave one hook partially applied
        res = None
        count = [0]
        err_count = [0]
        bw_count = [0]
        bw_count_lock = threading.Lock()
        err_count_lock = threading.Lock()

        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward raises RuntimeError on its first call
            # and passes the gradient on later calls.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Under a lock, count calls, raise on the first, and return
                # the gradient otherwise.
                with bw_count_lock:
                    bw_count[0] += 1
                    if bw_count[0] == 1:
                        raise RuntimeError("error message")
                    else:
                        return gO

        out = (Func.apply(t2) * t3).sum()

        def backward_retain_graph(out, t2, t3):
            # [FUNCTION DOC] HELPER: Run backward with inputs and retain_graph and count
            # RuntimeErrors under a lock.
            try:
                out.backward(inputs=(t2, t3), retain_graph=True)
            except RuntimeError:
                with err_count_lock:
                    err_count[0] += 1

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        self.assertEqual(count[0], 4)
        self.assertEqual(err_count[0], 1)
        self.assertEqual(res, [False, True, True, False])

        handle.remove()

    def test_multi_grad_any_hooks(self):
        # [FUNCTION DOC] Check multi-grad hooks in any mode fire once per concurrent backward
        # (five), including when one thread's backward raises, since the hook runs before the
        # custom backward.
        # Multihooks should behave independently per execution of backward
        # Test that the hook fired the number of times we ran backward
        # even if those executions occur concurrently on different threads
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        res = None
        count = [0]
        hook_lock = threading.Lock()

        def hook(grad):
            # [FUNCTION DOC] HELPER: Under a lock count the call and check the shared
            # marker value.
            nonlocal res
            with hook_lock:
                count[0] += 1
                if res is None:
                    res = "foo"
                else:
                    self.assertEqual(res, "foo")

        torch.autograd.graph.register_multi_grad_hook(
            (t1, t2, t3, t4), hook, mode="any"
        )

        out = (t2 * t3).sum()

        def backward_retain_graph(out, t2, t3):
            # [FUNCTION DOC] HELPER: Run backward with inputs and retain_graph.
            out.backward(inputs=(t2, t3), retain_graph=True)

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)
        self.assertEqual(count[0], 5)
        self.assertEqual(res, "foo")

        # Raise an error in one thread's backward
        res = None
        count = [0]
        err_count = [0]
        bw_count = [0]
        bw_count_lock = threading.Lock()
        err_count_lock = threading.Lock()

        class Func(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose backward raises RuntimeError on its first call
            # and passes the gradient on later calls.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Under a lock, count calls, raise on the first, and return
                # the gradient otherwise.
                with bw_count_lock:
                    bw_count[0] += 1
                    if bw_count[0] == 1:
                        raise RuntimeError("error message")
                    else:
                        return gO

        out = (Func.apply(t2) * t3).sum()

        def backward_retain_graph(out, t2, t3):
            # [FUNCTION DOC] HELPER: Run backward with inputs and retain_graph and count
            # RuntimeErrors under a lock.
            try:
                out.backward(inputs=(t2, t3), retain_graph=True)
            except RuntimeError:
                with err_count_lock:
                    err_count[0] += 1

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        # Expect all 5 threads to increment count since the hook runs before
        # the custom backward
        self.assertEqual(count[0], 5)
        self.assertEqual(err_count[0], 1)
        self.assertEqual(res, "foo")

    def test_dataparallel_saved_tensors_hooks(self):
        # [FUNCTION DOC] Check saved-tensor hooks and DataParallel: with two or more GPUs the
        # replicas run in other threads without the hooks (no warnings), otherwise hooks run;
        # hooks still run in the caller thread.
        def pack(x):
            # [FUNCTION DOC] HELPER: Pack hook that issues a warning and returns the tensor.
            warnings.warn("pack")
            return x

        _self = self

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that multiplies its input and checks the number of
            # warnings depending on the GPU count.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiply the input by itself inside a warning recorder
                # and assert whether the pack hook ran.
                with warnings.catch_warnings(record=True) as w:
                    y = x * x
                    if torch.cuda.device_count() >= 2:
                        # DataParallel is calling the forward in different threads
                        # without propagating TLS, so hooks should not be called here
                        _self.assertEqual(len(w), 0)
                    else:
                        # DataParallel only uses one thread
                        # so hooks should be called here
                        _self.assertGreater(len(w), 0)

        x = torch.ones(5, 5, requires_grad=True)
        model = torch.nn.DataParallel(Model())

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            model(x)
            with warnings.catch_warnings(record=True) as w:
                y = x * x
                # hooks should be called here
                _self.assertGreater(len(w), 0)

    def test_python_thread_in_middle(self):
        # [FUNCTION DOC] Check backward from several Python threads through a shared part of the
        # graph requires retain_graph: without it at least one thread succeeds and the rest raise
        # RuntimeError telling to specify retain_graph; with it all succeed and gradients add up.
        # User might write a network that starts on one CPU thread, then runs its second half
        # concurrently with other threads (either via python threading or fork/join calls),
        # then calls backward()/grad() on BOTH threads, like a Y pattern from input at the
        # bottom to output at the top. This way part of the GraphTask is being shared across
        # different threads and we need to ensure user specify retain_graph=True, otherwise
        # error out with the correct error message

        # Case 1: multiple backward with python threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        success_vs_raises = [0, 0]

        def train_fn_no_retain_graph(x):
            # [FUNCTION DOC] HELPER: Run backward on a shared intermediate, counting successes and
            # failures, and check the error text.
            y = x + x**2
            try:
                y.sum().backward()
                success_vs_raises[0] += 1
            except RuntimeError as error:
                success_vs_raises[1] += 1
                self.assertRegex(str(error), "Specify retain_graph=True")

        x_no_retain = torch.ones(5, 5, requires_grad=True)
        y_no_retain = x_no_retain + x_no_retain**2
        self._run_py_multithread_fn(
            train_fn_no_retain_graph, (y_no_retain,), num_threads=5
        )
        # at least one thread will be success in this case, all other threads should raise
        # with the error that throw to user to recommend them specify retain_graph=True
        self.assertTrue(success_vs_raises[0] >= 1)

        # multiple backward with python threads, no error with retain_graph=True
        def train_fn_retain_graph(x):
            # [FUNCTION DOC] HELPER: Run backward on a shared intermediate with retain_graph.
            y = x + x**2
            y.sum().backward(retain_graph=True)

        x_retain = torch.ones(5, 5, requires_grad=True)
        y_retain = x_retain + x_retain**2
        self._run_py_multithread_fn(train_fn_retain_graph, (y_retain,), num_threads=5)
        # result should equal to num_thread * gradients
        self.assertEqual(
            x_retain.grad,
            5 * (4 * x_retain**3 + 6 * (x_retain**2) + 4 * x_retain + 1),
        )

    def test_fork_join_in_middle(self):
        # [FUNCTION DOC] Check the same retain_graph requirement with TorchScript fork and join:
        # grad calls in forked tasks error without retain_graph (message about specifying it) and
        # succeed with it, giving equal gradients.
        # multiple backward with jit threads (fork/join primitive)
        # similar to test_python_thread_in_middle, we test with retain_graph=False/True

        # Case 1: multiple grad() calls with jit threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        @torch.jit.script
        def train_fn_jit_no_retain(middle, orig_x):
            # [FUNCTION DOC] HELPER: Scripted function computing the gradient of a quadratic of
            # the middle value with respect to the original input.
            y = middle + middle**2
            return torch.autograd.grad([y.sum()], [orig_x])

        @torch.jit.script
        def train_fn_fork_join_calls_no_retain(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the previous function and also
            # calling it directly, then waiting on the future; returns both gradients.
            y_no_retain = (x + 3) * (x + 4) * 0.5

            fut = torch.jit._fork(train_fn_jit_no_retain, y_no_retain, x)
            grad_hat = train_fn_jit_no_retain(y_no_retain, x)
            grad = torch.jit._wait(fut)
            return grad, grad_hat

        try:
            train_fn_fork_join_calls_no_retain(torch.randn(5, 5, requires_grad=True))
        except RuntimeError as error:
            self.assertRegex(str(error), "Specify retain_graph=True")

        # Case 2: no error with retain_graph=True
        @torch.jit.script
        def train_fn_jit_retain(middle, orig_x):
            # [FUNCTION DOC] HELPER: Scripted function like the previous one but passing
            # retain_graph to grad.
            y = middle + middle**2
            return torch.autograd.grad([y.sum()], [orig_x], retain_graph=True)

        @torch.jit.script
        def train_fn_fork_join_calls_retain(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the retaining function twice,
            # calling it once directly, and returning all three gradients.
            y_retain = (x + 3) * (x + 4) * 0.5
            fut1 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            fut2 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            grad = train_fn_jit_retain(y_retain, x)
            grad1 = torch.jit._wait(fut1)
            grad2 = torch.jit._wait(fut2)
            return grad, grad1, grad2

        grad, grad1, grad2 = train_fn_fork_join_calls_retain(
            torch.randn(5, 5, requires_grad=True)
        )
        self.assertEqual(grad, grad1)
        self.assertEqual(grad, grad2)

    def test_preserve_backtrace(self):
        # [FUNCTION DOC] Check the traceback of an exception raised in a custom backward is
        # preserved and includes the raising line text.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity Function whose backward raises ValueError.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Return the input.
                return input

            @staticmethod
            def backward(ctx, *grad):
                # [FUNCTION DOC] HELPER: Raise ValueError with the text something.
                raise ValueError("something")

        t = torch.rand(10, requires_grad=True)
        try:
            Foo.apply(t).sum().backward()
        except Exception:
            import traceback

            tb = sys.exc_info()[2]
            tb_str = "\n".join(traceback.format_tb(tb))
            self.assertTrue('raise ValueError("something")' in tb_str)

    # TODO(@anjali411): add an OpInfo based test for torch.cat
    # Issue: https://github.com/pytorch/pytorch/issues/51627
    #        https://github.com/pytorch/pytorch/issues/75852
    def test_cat_stack_r_to_c(self):
        # [FUNCTION DOC] Check gradcheck with forward AD of cat and stack across real and complex
        # inputs in both orders.
        inp_c = torch.rand(3, 2, dtype=torch.cdouble, requires_grad=True)
        inp_r = torch.randn(3, 2, dtype=torch.double, requires_grad=True)

        def fn(x1, x2):
            # [FUNCTION DOC] HELPER: Concatenate along the last dimension.
            return torch.cat((x1, x2), dim=-1)

        def fn2(x1, x2):
            # [FUNCTION DOC] HELPER: Stack along the last dimension.
            return torch.stack((x1, x2), dim=-1)

        torch.autograd.gradcheck(fn, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn, [inp_c, inp_r], check_forward_ad=True)

        torch.autograd.gradcheck(fn2, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn2, [inp_c, inp_r], check_forward_ad=True)

    def test_set_multithreading_enabled_as_context_manager_and_function(self):
        # [FUNCTION DOC] Check set_multithreading_enabled as a context manager and as a function,
        # and that the state is restored.
        # Test as a context manager
        with torch.autograd.set_multithreading_enabled(False):
            self.assertFalse(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        with torch.autograd.set_multithreading_enabled(True):
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.set_multithreading_enabled(True)
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        torch.autograd.set_multithreading_enabled(False)
        self.assertFalse(torch.autograd.is_multithreading_enabled())

        torch.autograd.set_multithreading_enabled(True)
        self.assertTrue(torch.autograd.is_multithreading_enabled())

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_custom_function_propagates_errors_from_device_thread(self):
        # [FUNCTION DOC] Check an error raised in a custom backward on a CUDA tensor propagates
        # from backward. Requires CUDA.
        class MyFunc(Function):
            # [CLASS DOC] HELPER: Identity Function whose backward raises RuntimeError.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Raise RuntimeError with the text blah.
                raise RuntimeError("blah")
                return gO

        t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("cuda"))
        out = MyFunc.apply(t).sum()

        with self.assertRaisesRegex(RuntimeError, "blah"):
            out.backward()

    def test_remove_obj_from_tls(self):
        # [FUNCTION DOC] Regression check that removing an object from thread local storage
        # actually removes the key (to avoid a destructor needing the GIL at thread exit).
        # Regression test for deadlock introduced by
        # https://github.com/pytorch/pytorch/pull/173568.
        # _remove_obj_from_tls must erase the key from the C++ thread_local
        # map so that no SafePyObject survives to thread exit (where its
        # destructor would try to acquire the GIL via __call_tls_dtors,
        # risking deadlock when another thread holds the GIL).
        torch._C._stash_obj_in_tls("test_obj", {"dummy": True})
        self.assertTrue(torch._C._is_key_in_tls("test_obj"))
        torch._C._remove_obj_from_tls("test_obj")
        self.assertFalse(torch._C._is_key_in_tls("test_obj"))


class TestNestedCheckpoint(TestCase):
    # [CLASS DOC] Tests of nested (including higher-order) non-reentrant checkpoint, with helpers
    # that wrap functions in grad, sum and checkpoint and verify the resulting graphs are freed.
    @staticmethod
    def grad(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it returns the gradient of its output with
        # respect to its input with create_graph, under enable_grad.
        def wrapper(x):
            # [FUNCTION DOC] HELPER: Compute the output, take the gradient with create_graph and
            # return it.
            with torch.enable_grad():
                out = fn(x)
                (grad_input,) = torch.autograd.grad(out, inputs=(x,), create_graph=True)
            return grad_input

        return wrapper

    @staticmethod
    def sum(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so its output is summed.
        def wrapped(x):
            # [FUNCTION DOC] HELPER: Return the summed output of the wrapped function.
            return fn(x).sum()

        return wrapped

    @staticmethod
    def checkpoint(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it runs under non-reentrant checkpoint.
        def wrapped(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Run checkpoint on the wrapped function with the
            # given arguments.
            return torch.utils.checkpoint.checkpoint(
                fn, *args, use_reentrant=False, **kwargs
            )

        return wrapped

    def get_tests(self, fn):
        # [FUNCTION DOC] HELPER: Build the test table pairing a function and its repeated
        # gradients with equivalent variants wrapped in checkpoint in several ways.
        grad, c = self.grad, self.checkpoint

        tests = (
            # function <> tuple of function arbitrarily wrapped in checkpoint in various ways
            (fn, (c(fn), c(c(fn)))),
            (grad(fn), (grad(c(fn)), grad(c(c(fn))))),
            (
                grad(grad(fn)),
                (grad(c(grad(fn))), c(grad(grad(c(fn)))), grad(c(grad(c(fn))))),
            ),
            (
                grad(grad(grad(fn))),
                (grad(c(grad(grad(c(fn))))), grad(c(grad(c(grad(c(fn))))))),
            ),
        )
        return tests

    def check_graph_dies(self, fn):
        # [FUNCTION DOC] HELPER: Verify that, after building a graph with the function, all nodes
        # are freed: attach weakly referenced handles to each node's metadata and assert none
        # remain after the scope ends.
        def iter_graph(roots):
            # [FUNCTION DOC] HELPER: Breadth-first generator over autograd nodes reachable from
            # the roots, skipping missing nodes and duplicates.
            if not roots:
                return
            seen = set()
            q = collections.deque()
            for node in roots:
                if node is not None:
                    seen.add(node)
                    q.append(node)

            while q:
                node = q.popleft()
                for fn, _idx in node.next_functions:
                    if fn in seen or fn is None:
                        continue
                    seen.add(fn)
                    q.append(fn)

                yield node

        class Handle:
            # [CLASS DOC] HELPER: Holder of a node name used to track node lifetime through a
            # weak reference.
            # KNOWN ISSUE: the class attribute is spelled with a singular name instead of the
            # real slots attribute name, so it has no effect and no slots are used (harmless).
            __slot__ = ["node_name"]

            def __init__(self, node_name):
                # [FUNCTION DOC] HELPER: Store the node name.
                self.node_name = node_name

        def scope():
            # [FUNCTION DOC] HELPER: Build a graph with the function, attach a handle to each
            # node's metadata and return weak references to the handles.
            a = torch.randn((), requires_grad=True)
            out = fn(a)
            refs = []
            for node in iter_graph([out.grad_fn]):
                handle = Handle(node.name())
                refs.append(weakref.ref(handle))
                node.metadata["blah"] = handle
            return refs

        refs = scope()
        node_names = [ref().node_name for ref in refs if ref() is not None]
        if len(node_names) > 0:
            print("Nodes still alive:", node_names)

        self.assertEqual(len(node_names), 0)

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint(self, early_stop):
        # [FUNCTION DOC] Parametrized by early stop: nested checkpoint combinations of simple
        # and gradient-taking functions give the same values as the plain function and their
        # graphs die.
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            x = torch.randn((), requires_grad=True)

            def f(x):
                # [FUNCTION DOC] HELPER: Chain of sine, exp and sine.
                out = x.sin().exp().sin()
                return out

            def g(x):
                # [FUNCTION DOC] HELPER: Compute two chains, take their gradients inside, and
                # return sine of the input.
                a = x.sin().exp().sin()
                b = x.sin().exp().sin()
                (ga,) = torch.autograd.grad(a, x)
                (gb,) = torch.autograd.grad(b, x)
                return x.sin()

            for fn in (f, g):
                for expected_fn, actual_fns in self.get_tests(fn):
                    expected = expected_fn(x)

                    for actual_fn in actual_fns:
                        actual = actual_fn(x)
                        self.assertTrue(torch.allclose(expected, actual))
                        self.check_graph_dies(actual_fn)

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_two_children(self, early_stop):
        # [FUNCTION DOC] Parametrized by early stop: nested checkpoint of two child functions
        # composed in a parent, with second-order gradients matching the non-checkpointed version
        # and graphs freed.
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            grad, sum, c = self.grad, self.sum, self.checkpoint

            def f(x):
                # [FUNCTION DOC] HELPER: Chain of sine, exp and sine.
                return x.sin().exp().sin()

            def g(x):
                # [FUNCTION DOC] HELPER: Chain of cosine, sine and exp.
                return x.cos().sin().exp()

            def hc(x):
                # [FUNCTION DOC] HELPER: Compose the two functions each under checkpoint.
                return c(g)(c(f)(x))

            def h(x):
                # [FUNCTION DOC] HELPER: Compose the two functions without checkpoint.
                return g(f(x))

            a = torch.randn(3, 3, requires_grad=True)
            expected = grad(sum(grad(sum(h))))(a)
            actual = grad(sum(grad(sum(c(hc)))))(a)
            self.assertTrue(torch.allclose(expected, actual))

            actual = grad(sum(c(grad(sum(c(hc))))))(a)
            self.assertTrue(torch.allclose(expected, actual))

            self.check_graph_dies(grad(c(hc)))
            self.check_graph_dies(grad(sum(grad(sum(c(hc))))))
            self.check_graph_dies(grad(sum(c(grad(sum(c(hc)))))))

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_non_tensor_inputs_and_outputs(self, early_stop):
        # [FUNCTION DOC] Parametrized by early stop: checkpoint with non-tensor inputs and outputs
        # (an int, a function, an int and a string) gives the same gradients as the plain call.
        def fn(k, a, b, f):
            # [FUNCTION DOC] HELPER: Apply the function to a scaled product and return the result
            # plus an int and a string.
            return f(k * a * b.exp()), 1, "abcd"

        k = 3
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        def f(x):
            # [FUNCTION DOC] HELPER: Sine of the input.
            return x.sin()

        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out, _unused1, _unused2 = checkpoint(fn, k, a, b, f, use_reentrant=False)
        actual_grads = torch.autograd.grad(out, (a, b))

        out, _unused1, _unused2 = fn(k, a, b, f)
        expected_grads = torch.autograd.grad(out, (a, b))
        for actual, expected in zip(actual_grads, expected_grads):
            self.assertTrue(torch.allclose(actual, expected))

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_kwargs(self, early_stop):
        # [FUNCTION DOC] Parametrized by early stop: checkpoint with a keyword tensor argument
        # gives the same gradients as the plain call.
        def fn(a, blah=None):
            # [FUNCTION DOC] HELPER: Compose sine and exp, optionally multiply by the keyword
            # value, then sine and exp.
            out = a.sin().exp()
            if blah is not None:
                out = out * blah
            return out.sin().exp()

        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out = checkpoint(fn, a, blah=b, use_reentrant=False)
            actual_grads = torch.autograd.grad(out, (a, b))

            out = fn(a, blah=b)
            expected_grads = torch.autograd.grad(out, (a, b))
            for actual, expected in zip(actual_grads, expected_grads):
                self.assertTrue(torch.allclose(actual, expected))

    @skipIfTorchDynamo("Dynamo support for curried checkpoint added in later commit")
    def test_checkpoint_curried_kwargs_do_not_collide_with_checkpoint_kwargs(self):
        # [FUNCTION DOC] Check keyword arguments of the wrapped function with the same names as
        # checkpoint options are passed to the function rather than interpreted by checkpoint
        # (curried checkpoint). Skipped under dynamo.
        def fn(
            a,
            preserve_rng_state,
            use_reentrant,
            context_fn,
            determinism_check,
            debug,
            early_stop,
        ):
            # [FUNCTION DOC] HELPER: Multiply the sine of the input by all the other arguments.
            return (
                a.sin()
                * preserve_rng_state
                * use_reentrant
                * context_fn
                * determinism_check
                * debug
                * early_stop
            )

        a = torch.tensor(2.0, requires_grad=True)
        wrapped_fn = checkpoint(use_reentrant=False, preserve_rng_state=False)(fn)
        out = wrapped_fn(
            a,
            preserve_rng_state=2,
            use_reentrant=3,
            context_fn=5,
            determinism_check=7,
            debug=11,
            early_stop=13,
        )
        actual_grad = torch.autograd.grad(out, (a,))

        out = fn(
            a,
            preserve_rng_state=2,
            use_reentrant=3,
            context_fn=5,
            determinism_check=7,
            debug=11,
            early_stop=13,
        )
        expected_grad = torch.autograd.grad(out, (a,))
        self.assertEqual(actual_grad, expected_grad)

    @skipIfTorchDynamo("Dynamo support for curried checkpoint added in later commit")
    def test_checkpoint_zero_arg_function(self):
        # [FUNCTION DOC] Check checkpoint of a function with no arguments, in direct and curried
        # call forms, gives equal outputs and gradients. Skipped under dynamo.
        a = torch.tensor(2.0, requires_grad=True)

        def fn():
            # [FUNCTION DOC] HELPER: Compute the sine then exp of a captured tensor.
            return a.sin().exp()

        old_out = checkpoint(fn, use_reentrant=False)
        new_out = checkpoint(use_reentrant=False)(fn)()
        expected_out = fn()
        self.assertEqual(old_out, expected_out)
        self.assertEqual(new_out, expected_out)

        old_grad = torch.autograd.grad(old_out, (a,), retain_graph=True)
        new_grad = torch.autograd.grad(new_out, (a,), retain_graph=True)
        expected_grad = torch.autograd.grad(expected_out, (a,))
        self.assertEqual(old_grad, expected_grad)
        self.assertEqual(new_grad, expected_grad)

    @skipIfTorchDynamo("Dynamo support for curried checkpoint added in later commit")
    def test_checkpoint_curried_method(self):
        # [FUNCTION DOC] Check a method decorated with curried checkpoint works with arguments and
        # gives the right gradients. Skipped under dynamo.
        class Model:
            # [CLASS DOC] HELPER: Class with a checkpointed method.
            @checkpoint(use_reentrant=False)
            def fn(self, x, scale):
                # [FUNCTION DOC] HELPER: Multiply the sine of x by scale.
                return x.sin() * scale

        x = torch.tensor(2.0, requires_grad=True)
        scale = torch.tensor(3.0, requires_grad=True)
        model = Model()

        actual = model.fn(x, scale)
        expected = x.sin() * scale
        self.assertEqual(actual, expected)
        self.assertEqual(
            torch.autograd.grad(actual, (x, scale)),
            torch.autograd.grad(expected, (x, scale)),
        )

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_same_graph(self, early_stop):
        # [FUNCTION DOC] Parametrized by early stop: a hook registered on the original graph of a
        # checkpointed function runs once during backward.
        counter = [0]

        def hook(*_unused_args):
            # [FUNCTION DOC] HELPER: Count hook calls.
            counter[0] += 1

        def fn(a):
            # [FUNCTION DOC] HELPER: Chain of sine, cosine and sine.
            return a.sin().cos().sin()

        a = torch.tensor(1.0, requires_grad=True)

        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out = checkpoint(fn, a, use_reentrant=False)
        # The hook is registered on the original graph
        out.grad_fn.next_functions[0][0].register_hook(hook)
        # And backward is performed on the original graph
        out.backward()

        self.assertEqual(counter[0], 1)

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_reentrant_backwards(self, early_stop):
        # [FUNCTION DOC] Parametrized by early stop: a hook that runs backward again from a
        # different output of a checkpointed function works.
        def fn(a):
            # [FUNCTION DOC] HELPER: Return an intermediate and the sine of it.
            x = a.sin().cos()
            out = x.sin()
            return x, out

        def hook(*_unused_args):
            # [FUNCTION DOC] HELPER: Run backward again from the intermediate while retaining
            # the graph.
            # do backward again, but skip over the part of the graph where
            # the hook was registered
            x.backward(retain_graph=True)

        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            x, out = checkpoint(fn, a, use_reentrant=False)
        out.grad_fn.register_hook(hook)
        out.backward(retain_graph=True)

    def test_nested_checkpoint_set_early_stop(self):
        # [FUNCTION DOC] Check the early stop setting: default on, local argument, context manager
        # (which applies to checkpoints created inside it), nesting and precedence, through the
        # number of recomputations of a function.
        counter = [0]

        def clone(x):
            # [FUNCTION DOC] HELPER: Count the call and return a clone.
            counter[0] += 1
            return x.clone()

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply sine and cosine, then the counting clone.
            # Since clone does not save anything, it is not recomputed iff
            # early stop is enabled.
            return clone(x.sin().cos())

        # Test default
        # Early stopping is enabled by default
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test local setting
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn, a, use_reentrant=False, early_stop=False)
        out.backward()
        self.assertEqual(counter[0], 2)

        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn, a, use_reentrant=False, early_stop=True)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test context manager
        # Expect early stopping to be disabled for all checkpoints ran under
        # the context manager, even though context manager is no longer active
        # when backward/recomputation is performed.
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 2)

        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(True):
            out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test context manager nesting
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            with torch.utils.checkpoint.set_checkpoint_early_stop(True):
                out = checkpoint(fn, a, use_reentrant=False, early_stop=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test precedence
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn, a, use_reentrant=False, early_stop=True)
        out.backward()
        self.assertEqual(counter[0], 2)

        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(True):
            out = checkpoint(fn, a, use_reentrant=False, early_stop=False)
        out.backward()
        self.assertEqual(counter[0], 1)

    def test_nested_checkpoint_set_early_stop_no_recompution_needed(self):
        # [FUNCTION DOC] Check recomputation counts when no recomputation is needed: when the
        # only saved tensor is the input, the function runs twice either way and the sine op runs
        # once with early stop and twice without; when nothing is saved, the function runs once
        # either way.
        # Case 1: We have one tensor saved and its the input

        # We have two different counters here because in this case we actually
        # do call into x.sin() at the python level during recomputation whether
        # or not early stop is enabled. This is because the early stopping
        # only happens at the autograd level (preventing us from reaching the
        # backend).
        python_dispatch_counter = [0]
        counter = [0]

        class SinCounterMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode counting how many times the sine op runs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Start the count at zero.
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Count sine calls and run the op.
                kwargs = {} if kwargs is None else kwargs
                if func is torch.ops.aten.sin.default:
                    self.count += 1
                return func(*args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Count the call and return the sine of the input.
            counter[0] += 1
            return x.sin()

        # With early stopping (enabled by default)
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:  # noqa: F811
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 1)

        # Without early stopping
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 2)

        # Case 2: Forward saves no tensors

        # Since unpack isn't even called, counter is 1 whether or not early stop
        # is enabled!
        counter = [0]

        def fn2(x):
            # [FUNCTION DOC] HELPER: Count the call and return a clone of the input.
            counter[0] += 1
            return x.clone()

        # With early stopping (enabled by default)
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Without early stopping
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)


@contextlib.contextmanager
def _counter_op(name):
    # [FUNCTION DOC] HELPER: Context manager creating a custom op with the given name inside a
    # scoped library that counts invocations and logs per call index counts, with an autograd
    # rule; yields the op, the counter and the log.
    """Yields (op, counts, idx_log) where op is a custom op that counts
    invocations. idx_log maps call_idx (passed by caller) to replay count."""
    counts = [0]
    idx_log: dict = {}
    with torch.library._scoped_library("test_ckpt", "FRAGMENT"):

        @torch.library.custom_op(f"test_ckpt::{name}", mutates_args=())
        def op(x: torch.Tensor, call_idx: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op computing the sine while counting calls and logging
            # by call index.
            counts[0] += 1
            idx_log[call_idx] = idx_log.get(call_idx, 0) + 1
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Save the first input for backward.
            ctx.save_for_backward(inputs[0])

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Return the gradient times the cosine of the saved input
            # and None.
            (x,) = ctx.saved_tensors
            return grad * x.cos(), None

        op.register_autograd(backward, setup_context=setup_context)

        yield op, counts, idx_log


class _AutoNamingMode(TorchDispatchMode):
    # [CLASS DOC] HELPER: Dispatch mode that names output tensors by module path, op name and
    # per-module op count (adding output index for multi-output ops), recording names in a weak
    # tensor key mapping.
    """Test helper: names output tensors as ``fqn_op_count[_outputidx]``."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Create a module tracker, the per-key counter and the weak
        # name mapping.
        from torch.utils.module_tracker import ModuleTracker

        self._tracker = ModuleTracker()
        self._func_counter: dict = defaultdict(int)
        self.names = WeakTensorKeyDictionary()

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Enter the module tracker, then enter the dispatch mode.
        self._tracker.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        # [FUNCTION DOC] HELPER: Exit the module tracker, then exit the dispatch mode.
        self._tracker.__exit__(*args)
        return super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Run the op, work out the innermost module path, op name and
        # count, and record names for each tensor output.
        out = func(*args, **(kwargs or {}))
        parents = self._tracker.parents - {"Global"}
        fqn = max(parents, key=len) if parents else "Global"
        op_name = func.__name__ if hasattr(func, "__name__") else str(func)
        key = (fqn, func)
        count = self._func_counter[key]
        self._func_counter[key] += 1
        multi_output = (
            isinstance(out, (tuple, list))
            and sum(isinstance(o, torch.Tensor) for o in out) > 1
        )
        if isinstance(out, torch.Tensor):
            self.names[out] = f"{fqn}_{op_name}_{count}"
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    name = f"{fqn}_{op_name}_{count}"
                    if multi_output:
                        name += f"_{i}"
                    self.names[o] = name
        return out


class TestSelectiveActivationCheckpoint(TestCase):
    # [CLASS DOC] Tests for selective activation checkpoint (SAC) built with checkpoint and a
    # context function: memory and flop accounting, policies, op lists, caching lifetime, version
    # counters, hooks, mismatch errors and naming helper mode.
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_flops_and_mem(self):
        # [FUNCTION DOC] Check activation memory and backward flops without checkpoint, with full
        # checkpoint, and with selective checkpoint saving the matrix multiply (given as op list,
        # as policy function, and as boolean policy): memory 2, 0, 1 and flops 2, 3, 2. CUDA only.
        # From https://github.com/pytorch/pytorch/pull/126320
        def get_act_mem(f):
            # [FUNCTION DOC] HELPER: Run forward and backward once, run forward again
            # measuring requested bytes allocated in megabytes, run backward and return the
            # activation memory.
            out = f()
            out.backward()
            # Why do one forward and backward?
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            cur_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            act_mem = (cur_mem - start_mem) / (1024 * 1024)
            out.backward()
            return act_mem

        def get_bw_flops(f):
            # [FUNCTION DOC] HELPER: Run backward under a flop counter after a warmup and return
            # flops normalized by a 512 square matmul.
            # Normalized so that a 512 square matmul returns 1
            f().backward()
            out = f()
            # NB: FlopCounterMode is pushed onto the mode stack before CachedMode, so
            # it will be able to observe whether an op is cached or not.
            with FlopCounterMode(display=False) as mode:
                out.backward()
            return mode.get_total_flops() / (512**3 * 2)

        x = torch.randn(512, 512, requires_grad=True, device="cuda")
        y = torch.randn(512, 512, requires_grad=True, device="cuda")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matrix multiply of cosine of x with y, then sine and sum.
            return torch.mm(x.cos(), y).sin().sum()

        def fn_ac(x, y):
            # [FUNCTION DOC] HELPER: Full non-reentrant checkpoint of the function.
            return checkpoint(fn, x, y, use_reentrant=False)

        def fn_sac(x, y):
            # [FUNCTION DOC] HELPER: Checkpoint with selective contexts saving the mm op given as
            # an op list.
            context_fn = functools.partial(
                create_selective_checkpoint_contexts,
                [torch.ops.aten.mm.default],
            )
            out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
            return out

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy saving the mm op and recomputing others.
            if op == torch.ops.aten.mm.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn_sac2(x, y):
            # [FUNCTION DOC] HELPER: Checkpoint with selective contexts using the policy function.
            context_fn = functools.partial(
                create_selective_checkpoint_contexts,
                policy_fn,
            )
            out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
            return out

        def policy_fn_bool(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Boolean policy returning true for the mm op.
            return op == torch.ops.aten.mm.default

        def fn_sac3(x, y):
            # [FUNCTION DOC] HELPER: Checkpoint with selective contexts using the boolean policy.
            context_fn = functools.partial(
                create_selective_checkpoint_contexts,
                policy_fn_bool,
            )
            out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
            return out

        act_mem_noac = get_act_mem(lambda: fn(x, y))
        bw_flops_noac = get_bw_flops(lambda: fn(x, y))

        self.assertEqual(act_mem_noac, 2.0)
        self.assertEqual(bw_flops_noac, 2.0)

        act_mem_ac = get_act_mem(lambda: fn_ac(x, y))
        bw_flops_ac = get_bw_flops(lambda: fn_ac(x, y))

        self.assertEqual(act_mem_ac, 0.0)
        self.assertEqual(bw_flops_ac, 3.0)

        act_mem_sac = get_act_mem(lambda: fn_sac(x, y))
        bw_flops_sac = get_bw_flops(lambda: fn_sac(x, y))

        self.assertEqual(act_mem_sac, 1.0)
        self.assertEqual(bw_flops_sac, 2.0)

        act_mem_sac2 = get_act_mem(lambda: fn_sac2(x, y))
        bw_flops_sac2 = get_bw_flops(lambda: fn_sac2(x, y))

        self.assertEqual(act_mem_sac2, 1.0)
        self.assertEqual(bw_flops_sac2, 2.0)

        act_mem_sac3 = get_act_mem(lambda: fn_sac3(x, y))
        bw_flops_sac3 = get_bw_flops(lambda: fn_sac3(x, y))

        self.assertEqual(act_mem_sac3, 1.0)
        self.assertEqual(bw_flops_sac3, 2.0)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_output_already_has_autograd_meta(self):
        # [FUNCTION DOC] Check selective checkpoint with an output that is a view of an integer
        # input (already has autograd metadata) works and backward through the other output
        # succeeds. Skipped under dynamo.
        # View of tensor of non-differentiable dtype still has AutogradMeta
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return a flattened view of the first input and a sine-cosine
            # of the second.
            return x.view(-1), y.sin().cos()

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        y = torch.randn(3, requires_grad=True)

        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            [torch.ops.aten.view.default],
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        out[1].sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_subclass_dispatching_sizes(self):
        # [FUNCTION DOC] Check selective checkpoint ignores metadata ops such as symbolic size for
        # a wrapper subclass with dynamic sizes: the saved shape in the recompute must be correct.
        # Skipped under dynamo.
        # Test that we ignore ops that grab metadata like torch.ops.aten.sym_size.default
        # Caching such metadata ops can be problematic when the following are satisfied:
        #
        # 1. size/strides are dispatched upon
        # 2. our policy saves sizes
        ta = torch.randn(6, 2)

        class CustomSizeDynamicShapesTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass that reports sizes via the dispatch path
            # and delegates ops to an inner tensor.
            @staticmethod
            def __new__(cls, inner):
                # [FUNCTION DOC] HELPER: Create a wrapper subclass with the inner tensor's size,
                # stride, dtype, layout, device and requires_grad with sizes dispatch.
                return torch.Tensor._make_wrapper_subclass(
                    # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
                    # Calling the overload that has kwargs causes us to go down the first overload path,
                    # which will **always** specialize sizes.
                    # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
                    cls,
                    inner.size(),
                    inner.stride(),
                    None,
                    None,
                    inner.dtype,
                    inner.layout,
                    inner.device,
                    False,
                    inner.requires_grad,
                    "sizes",
                )

            def __init__(self, inner):
                # [FUNCTION DOC] HELPER: Store the inner tensor.
                self.inner = inner

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Unwrap subclass arguments, run the op on inner tensors
                # and rewrap tensor outputs.
                if kwargs is None:
                    kwargs = {}
                args_inner = torch.utils._pytree.tree_map_only(
                    cls, lambda x: x.inner, args
                )
                out_inner = func(*args_inner, **kwargs)
                return torch.utils._pytree.tree_map_only(
                    torch.Tensor, lambda x: cls(x), out_inner
                )

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy that asks to save the symbolic size op (silently
            # ignored) and recompute everything else.
            if op is torch.ops.aten.sym_size.default:
                # Silently ignored!
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: Create chained views and assert the saved symbolic sizes of
            # the last view equal the flattened size, then return the exp.
            # We avoid the following case
            #
            # saved     :[4, 3], [], [], [4, 3], [4, 3], [4, 3], [12]
            # forward   :sum   ,sum,mul, mul   , mul   ,view   , view
            # recompute :sum   ,sum,mul, view  , view
            #
            # Views save the shape of their input, so we expect the second
            # view to save 12, but because during AC packing during forward
            # saves the shapes of the input for metadata checks later,
            # we would save the wrong shape during the recompute.
            view_out = (x * x.sum()).view(-1).view(4, 3)
            self.assertEqual(view_out.grad_fn._saved_self_sym_sizes, [12])
            return view_out.exp()

        x = torch.randn(4, 3, requires_grad=True)
        x_wrapper = CustomSizeDynamicShapesTensor(x)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x_wrapper, use_reentrant=False, context_fn=context_fn)
        out.sum().backward()

    def test_bad_inputs(self):
        # [FUNCTION DOC] Check create_selective_checkpoint_contexts input validation: ValueError
        # for non-op entries and for an op packet, TypeError for neither function nor list.
        bad_op_list1 = [2]

        with self.assertRaisesRegex(
            ValueError, "Expected op in `op_list` to be an OpOverload"
        ):
            create_selective_checkpoint_contexts(bad_op_list1)

        bad_op_list2 = [torch.ops.aten.sin]

        with self.assertRaisesRegex(
            ValueError, "update the OpOverloadPacket to a specific OpOverload"
        ):
            create_selective_checkpoint_contexts(bad_op_list2)

        with self.assertRaisesRegex(TypeError, "either a function or a list of ops."):
            create_selective_checkpoint_contexts(2)

    # Dynamo fails for various reasons:
    # - some tests using custom op that does not implement Fake
    # - dynamo is trying to trace into saved variable hooks unpack hook for some reason
    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_policy_with_state(self):
        # [FUNCTION DOC] Check the policy is called only during forward (once per op) and not
        # during recompute. Skipped under dynamo.
        # Policy is only called during forward, not during backward/recompute.
        counters = []

        class Policy:
            # [CLASS DOC] HELPER: Callable policy object that counts calls, records the running
            # count and prefers recompute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Start the call counter at zero.
                self.counter = [0]

            def __call__(self, ctx, func, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Increment the counter, record it and return the
                # prefer-recompute decision.
                self.counter[0] += 1
                counters.append(self.counter[0])
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: Chain of three sines.
            return x.sin().sin().sin()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            Policy(),
            allow_cache_entry_mutation=True,
        )
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        out.sum().backward()
        # Policy is only called during forward (3 ops), not during recompute
        self.assertEqual(counters, [1, 2, 3])

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_storage_lifetime(self):
        # [FUNCTION DOC] Check lifetime of values stored by selective checkpoint: stored values
        # are detached, freed when the graph is freed without backward, and cleared after backward
        # even with retain_graph. Skipped under dynamo.
        from torch.utils._python_dispatch import _get_current_dispatch_mode
        from torch.utils.checkpoint import (
            _CachedTorchDispatchMode,
            _CachingTorchDispatchMode,
        )

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy that saves every op.
            return CheckpointPolicy.MUST_SAVE

        ref = None

        def fn(x):
            # [FUNCTION DOC] HELPER: Check the active dispatch mode is the caching or cached mode,
            # compute cosine and exp, record a weak reference to the stored exp output in forward,
            # check it does not require grad, and return its sine.
            nonlocal ref

            self.assertIsInstance(
                _get_current_dispatch_mode(),
                (_CachingTorchDispatchMode, _CachedTorchDispatchMode),
            )

            out = x.cos().exp()

            if isinstance(_get_current_dispatch_mode(), _CachingTorchDispatchMode):
                raw_val = (
                    _get_current_dispatch_mode()
                    .storage[torch.ops.aten.exp.default][0]
                    .val
                )
                # ref should've been detached
                # to avoid graph -> the saved variable hooks -> recompute_context -> storage -> graph
                self.assertFalse(raw_val.requires_grad)
                ref = weakref.ref(raw_val)

            # Careful for early-stop
            return out.sin()

        with disable_gc():
            # Case 1: If graph goes away without backward, make sure there's no reference cycle
            #         keeping storage alive.
            x = torch.randn(3, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            self.assertIsNotNone(ref())
            del out
            self.assertIsNone(ref())

            # Case 2: After backward, even if retain_graph=True, the storage should go away
            x = torch.randn(3, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            self.assertIsNotNone(ref())
            out.sum().backward(retain_graph=True)
            # The dispatch mode's storage should still be alive, but the entries should've
            # been cleared.
            self.assertIsNone(ref())

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_version_counter(self):
        # [FUNCTION DOC] Check version counter handling in selective checkpoint: a saved output
        # mutated in place by a later op raises RuntimeError about mutation, unless cache entry
        # mutation is allowed. Skipped under dynamo.
        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy saving sine outputs and recomputing others.
            if op == torch.ops.aten.sin.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: Sine, in-place multiply by two, cosine and exp.
            return x.sin().mul_(2).cos().exp()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        # 1) Error because the output of sin is saved and mutated by mul_
        with self.assertRaisesRegex(RuntimeError, "has been mutated"):
            out.sum().backward()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            policy_fn,
            allow_cache_entry_mutation=True,
        )
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        # 2) No longer should be an error because of allow_cache_entry_mutation
        out.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_saved_tensors_hooks_fire_for_saved_tensors(self):
        # [FUNCTION DOC] Check that, when respecting saved tensor hooks is requested, pack
        # and unpack hooks around the checkpoint fire for tensors selective checkpoint saves
        # (the input and the mm output), but not for hooks around only the backward. Skipped
        # under dynamo.
        # Tensors SAC decides to save skip SavedVariable, so SAC must simulate
        # pack/unpack with the surrounding user saved-tensors hooks. This is
        # opt-in for now (BC); the default legacy behavior is covered by
        # TestSACAmbientSavedTensorsHooks.
        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy saving mm outputs and recomputing others.
            if op == torch.ops.aten.mm.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        packed, unpacked = [], []

        def pack(x):
            # [FUNCTION DOC] HELPER: Pack hook recording the tensor and returning it doubled.
            packed.append(x)
            return x * 2  # transform so we can tell unpack really ran

        def unpack(x):
            # [FUNCTION DOC] HELPER: Unpack hook recording the tensor and returning it halved.
            unpacked.append(x)
            return x / 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Matrix multiply of x with itself, relu and sum.
            return torch.mm(x, x).relu().sum()

        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        x = torch.randn(4, 4, requires_grad=True)

        def run_checkpoint():
            # [FUNCTION DOC] HELPER: Run the checkpoint with the selective context and the respect
            # flag set.
            return checkpoint(
                fn,
                x,
                use_reentrant=False,
                context_fn=context_fn,
                respect_saved_tensors_hooks=True,
            )

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = run_checkpoint()
        out.backward()
        # Exactly two packs, both during forward: the checkpoint input x
        # (save_inputs) and the mm output (MUST_SAVE, via SAC storage).
        # Recompute saves are shadowed by _recomputation_hook and don't fire.
        self.assertEqual(len(packed), 2)
        self.assertEqual(len(packed), len(unpacked))

        x_ref = x.detach().clone().requires_grad_()
        fn(x_ref).backward()
        self.assertEqual(x.grad, x_ref.grad)

        # Hooks around backward only must not affect tensors SAC saved
        # without hooks in scope
        packed.clear()
        unpacked.clear()
        x.grad = None
        out = run_checkpoint()
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out.backward()
        self.assertEqual(len(packed), 0)
        self.assertEqual(x.grad, x_ref.grad)

    @skipIfTorchDynamo("gc is unreliable under dynamo-wrapped frames")
    def test_saved_tensors_hooks_no_refcycle_with_stateful_owner(self):
        # [FUNCTION DOC] Check that checkpoint's internal hooks do not keep user hooks alive: an
        # object holding the output and used as hooks is garbage collected. Skipped under dynamo.
        # The graph retains checkpoint's internal pack_hook via SavedVariable
        # in a way gc cannot traverse, so if pack_hook kept a permanent ref to
        # the user hooks, a hook owner reaching back to the output would leak
        # uncollectably. _user_hooks must not outlive the hooks TLS scope.
        class Hooks:
            # [CLASS DOC] HELPER: Hook owner object with pack and unpack methods.
            def pack(self, t):
                # [FUNCTION DOC] HELPER: Return the detached tensor.
                return t.detach()

            def unpack(self, t):
                # [FUNCTION DOC] HELPER: Return the tensor unchanged.
                return t

        def scenario():
            # [FUNCTION DOC] HELPER: Checkpoint a function with no tensor arguments under the
            # owner's hooks, store the output on the owner (cycle) and return weak references
            # to both.
            x = torch.randn(4, 4, requires_grad=True)
            hooks = Hooks()

            # No tensor args: keeps save_inputs from packing x through the
            # user hooks, which would reach hooks via a pre-existing path.
            def fn():
                # [FUNCTION DOC] HELPER: Matrix multiply of a captured tensor with itself, relu
                # and sum.
                return torch.mm(x, x).relu().sum()

            with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
                out = checkpoint(fn, use_reentrant=False)
            hooks.output = out  # owner -> graph back-reference
            return weakref.ref(hooks), weakref.ref(out)

        hooks_ref, out_ref = scenario()
        gc.collect()
        self.assertIsNone(hooks_ref())
        self.assertIsNone(out_ref())

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_function_with_more_than_one_output(self):
        # [FUNCTION DOC] Check selective checkpoint of a multi-output op (var_mean): policy called
        # once, gradients match the plain computation. Skipped under dynamo.
        # maybe there is a more systematic way:
        counter = [0]

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy counting calls for var_mean and saving it,
            # recomputing others.
            if op == torch.ops.aten.var_mean.correction:
                counter[0] += 1
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        # var_mean has two outputs
        def fn(x):
            # [FUNCTION DOC] HELPER: Product of variance and mean.
            a, b = torch.var_mean(x)
            return a * b

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        x_grad = torch.autograd.grad(out.sum(), (x,))
        x_grad_ref = torch.autograd.grad(fn(x).sum(), (x,))
        self.assertEqual(x_grad, x_grad_ref)
        # Policy is only called during forward, not backward
        self.assertEqual(counter[0], 1)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_function_with_non_tensor_output(self):
        # [FUNCTION DOC] Check selective checkpoint with an op returning a non-tensor output: the
        # saved op is not computed again in backward (count stays one) and gradients match.
        # Skipped under dynamo.
        # When SAC is enabled, the op is not computed a second time
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            counter = [0]

            @torch.library.custom_op("mylib::sin_with_extra", mutates_args=())
            def sin_with_extra(x: torch.Tensor) -> tuple[torch.Tensor, int]:
                # [FUNCTION DOC] HELPER: Custom op returning the sine and an int while
                # counting calls.
                counter[0] += 1
                return x.sin(), 2

            def setup_context(ctx, inputs, output) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Save the tensor input for backward.
                (x,) = inputs
                ctx.save_for_backward(x)

            def backward(ctx, grad, _unused):
                # [FUNCTION DOC] HELPER: Return the gradient times cosine of the saved input.
                (x,) = ctx.saved_tensors
                return grad * x.cos()

            torch.library.register_autograd(
                "mylib::sin_with_extra", backward, setup_context=setup_context
            )

            x = torch.randn(3, requires_grad=True)

            def fn(x):
                # [FUNCTION DOC] HELPER: Combine the custom op output with an exp of sine and
                # take sine.
                return (torch.ops.mylib.sin_with_extra(x)[0] * x.sin().exp()).sin()

            ops_list = [torch.ops.mylib.sin_with_extra.default]

            x = torch.randn(3, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, ops_list
            )
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            x_grad = torch.autograd.grad(out.sum(), (x,))
            self.assertEqual(counter[0], 1)
            x_grad_ref = torch.autograd.grad(fn(x).sum(), (x,))
            self.assertEqual(x_grad, x_grad_ref)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_mismatch_new_op_during_recompute(self):
        # [FUNCTION DOC] Check that recomputation running a new op not seen in forward raises
        # RuntimeError saying it was not found in storage. Skipped under dynamo.
        call_count = [0]

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy that prefers recompute for everything.
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: Sine, plus an extra exp only on later calls.
            call_count[0] += 1
            x = x.sin()
            if call_count[0] > 1:
                x = x.exp()
            return x

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(
            fn, x, use_reentrant=False, context_fn=context_fn, early_stop=False
        )
        with self.assertRaisesRegex(RuntimeError, "not found in storage"):
            out.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_mismatch_extra_invocation_during_recompute(self):
        # [FUNCTION DOC] Check that an extra invocation of a saved op during recompute raises
        # RuntimeError mentioning the invocation index missing from storage. Skipped under dynamo.
        call_count = [0]

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy saving sine and recomputing the rest.
            if op == torch.ops.aten.sin.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: Sine, an extra sine only on later calls, then cosine.
            call_count[0] += 1
            x = x.sin()
            if call_count[0] > 1:
                x = x.sin()
            return x.cos()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(
            fn, x, use_reentrant=False, context_fn=context_fn, early_stop=False
        )
        with self.assertRaisesRegex(
            RuntimeError, "invocation index .* not found in storage"
        ):
            out.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_can_only_trigger_recompute_once(self):
        # [FUNCTION DOC] Check that only one recompute can be triggered: a second backward with
        # retain_graph raises RuntimeError about an extra time. Skipped under dynamo.
        # We don't support this to avoid adding extra complexity for now.
        # If there's a need, we could probably do some kind of use_count tracking.
        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy saving sine and recomputing the rest.
            if op == torch.ops.aten.sin.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: Sine, cosine and exp.
            return x.sin().cos().exp()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        out.sum().backward(retain_graph=True)

        with self.assertRaisesRegex(RuntimeError, "Trying to backward an extra time"):
            out.sum().backward(retain_graph=True)

    @skipIfTorchDynamo("torch dispatch modes don't support compile")
    def test_auto_naming_mode_names(self):
        # [FUNCTION DOC] Check the naming dispatch mode in a policy: ops are named by module path,
        # op and count, a policy saving selected names causes the expected per-call recompute
        # counts and nine total calls. Skipped under dynamo.
        with _counter_op("my_op") as (my_op, my_count, idx_log):

            class Block(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying a counting custom op three times and
                # incrementing a shared index.
                def forward(self, x, counter):
                    # [FUNCTION DOC] HELPER: Apply the custom op three times with increasing
                    # call indices.
                    x = my_op(x, counter[0])
                    counter[0] += 1
                    x = my_op(x, counter[0])
                    counter[0] += 1
                    x = my_op(x, counter[0])
                    counter[0] += 1
                    return x

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with two blocks.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Create the two blocks in a list.
                    super().__init__()
                    self.layers = torch.nn.ModuleList([Block(), Block()])

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Pass the input through the blocks with a
                    # shared counter.
                    counter = [0]
                    for layer in self.layers:
                        x = layer(x, counter)
                    return x

            mod = Model()
            naming = _AutoNamingMode()

            save_names = {
                "Model.layers.0_my_op.default_1",
                "Model.layers.1_my_op.default_0",
                "Model.layers.1_my_op.default_2",
            }

            def policy_fn(ctx, op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Policy saving outputs whose recorded names are in the
                # chosen set and recomputing others.
                out = ctx.op_output
                if isinstance(out, torch.Tensor):
                    name = naming.names.get(out)
                    if name in save_names:
                        return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            x = torch.randn(4, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            with naming:
                out = checkpoint(
                    lambda x: mod(x),
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                )
                out.sum().backward()

            self.assertEqual(
                idx_log,
                {
                    0: 2,  # Model.layers.0_my_op.default_0 -> recomputed
                    1: 1,  # Model.layers.0_my_op.default_1 -> saved
                    2: 2,  # Model.layers.0_my_op.default_2 -> recomputed
                    3: 1,  # Model.layers.1_my_op.default_0 -> saved
                    4: 2,  # Model.layers.1_my_op.default_1 -> recomputed
                    5: 1,  # Model.layers.1_my_op.default_2 -> saved
                },
            )
            self.assertEqual(my_count[0], 9)


@skipIfTorchDynamo("SAC hook interaction under compile tested elsewhere")
class TestSACAmbientSavedTensorsHooks(TestCase):
    # [CLASS DOC] Characterization tests for how a user saved-tensor-hooks context wrapped around
    # a selective checkpoint region interacts with tensors SAC saves: the legacy default bypasses
    # hooks with a FutureWarning, an explicit false bypasses silently, true routes through the
    # hooks. Skipped under dynamo.
    # Characterizes how a user saved_tensors_hooks context wrapped OUTER around
    # a selective activation checkpoint (SAC) region interacts with the tensors
    # SAC decides to save. Historically SAC held those tensors outside the
    # autograd graph, so an ambient hook (e.g. save_on_cpu) never saw them.
    # The respect_saved_tensors_hooks arg of checkpoint() opts into routing them
    # through the hook; the default (legacy) leaves them untouched and warns.
    # These tests pin both behaviors so the eventual default flip is explicit.
    class RecordingHooks:
        # [CLASS DOC] HELPER: Context manager installing pack and unpack hooks that record the
        # shapes of packed tensors without changing them.
        # Records the shapes pack sees without transforming the tensor, so it is
        # numerically transparent (models a CPU-offload hook on CPU).
        def __init__(self):
            # [FUNCTION DOC] HELPER: Start with an empty record of packed shapes.
            self.packed = []

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Install the recording hooks and return self.
            def pack(t):
                # [FUNCTION DOC] HELPER: Record the shape and return the tensor.
                self.packed.append(tuple(t.shape))
                return t

            def unpack(t):
                # [FUNCTION DOC] HELPER: Return the tensor unchanged.
                return t

            self._ctx = torch.autograd.graph.saved_tensors_hooks(pack, unpack)
            self._ctx.__enter__()
            return self

        def __exit__(self, *args):
            # [FUNCTION DOC] HELPER: Uninstall the hooks.
            return self._ctx.__exit__(*args)

    # Distinct shapes so a recorded shape unambiguously identifies which op's
    # output the ambient hook saw: only the MUST_SAVE mm output is MM_SHAPE.
    INPUT_SHAPE = (4, 6)
    MM_SHAPE = (6, 6)

    @staticmethod
    def _fn(x):
        # [FUNCTION DOC] HELPER: Matrix multiply of the transpose of x with x, relu and sum.
        # mm(x.t(), x) turns an INPUT_SHAPE tensor into an MM_SHAPE one.
        return torch.mm(x.t(), x).relu().sum()

    @staticmethod
    def _mm_save_context_fn():
        # [FUNCTION DOC] HELPER: Build a context function with a policy that saves the mm output
        # and recomputes the rest.
        # MUST_SAVE the mm output; recompute everything else.
        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Policy saving mm outputs and recomputing others.
            if op == torch.ops.aten.mm.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        return functools.partial(create_selective_checkpoint_contexts, policy_fn)

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run the base setup and seed the random generator with zero.
        super().setUp()
        torch.manual_seed(0)

    def _run(self, respects, capture_warning=False):
        # [FUNCTION DOC] HELPER: Run a selective checkpoint region under recording ambient hooks,
        # check numeric parity with the plain function, and return the packed shapes (and
        # FutureWarnings if requested).
        # Wrap an ambient RecordingHooks OUTER around a SAC region whose only
        # saved tensor is the MM_SHAPE mm output. Returns the recorded shapes.
        fn = self._fn
        x = torch.randn(*self.INPUT_SHAPE, requires_grad=True)
        context_fn = self._mm_save_context_fn()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.RecordingHooks() as hooks:
                out = checkpoint(
                    fn,
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                    respect_saved_tensors_hooks=respects,
                )
            out.backward()

        # Numeric parity vs the plain (no-hook, no-checkpoint) reference must
        # hold regardless of the flag: recording/offloading is transparent.
        x_ref = x.detach().clone().requires_grad_()
        fn(x_ref).backward()
        torch.testing.assert_close(x.grad, x_ref.grad)

        if capture_warning:
            fw = [w for w in caught if issubclass(w.category, FutureWarning)]
            return hooks.packed, fw
        return hooks.packed

    def test_hook_outer_sac_inner_default_bypasses(self):
        # [FUNCTION DOC] Check the default setting bypasses the ambient hook for the saved mm
        # output and emits a FutureWarning mentioning selective activation checkpoint.
        # Legacy default: the ambient hook is blind to the MUST_SAVE mm output.
        # A FutureWarning fires because a user hook is in scope.
        packed, fw = self._run(respects=None, capture_warning=True)
        self.assertNotIn(self.MM_SHAPE, packed)
        self.assertTrue(
            any("selective activation checkpoint" in str(w.message) for w in fw)
        )

    def test_hook_outer_sac_inner_opt_in_routes(self):
        # [FUNCTION DOC] Check the opt-in setting routes the saved mm output through the
        # ambient hook.
        # Opt-in: the MUST_SAVE mm output now flows through the ambient hook.
        packed = self._run(respects=True)
        self.assertIn(self.MM_SHAPE, packed)

    def test_hook_outer_sac_inner_opt_out_bypasses_silently(self):
        # [FUNCTION DOC] Check the explicit opt-out bypasses the hook for the mm output with
        # no warning.
        # Explicit legacy opt-out: hook blind to the mm output, and no warning.
        packed, fw = self._run(respects=False, capture_warning=True)
        self.assertNotIn(self.MM_SHAPE, packed)
        self.assertEqual(fw, [])

    def test_save_on_cpu_opt_in_offloads_saved_tensor(self):
        # [FUNCTION DOC] Check save_on_cpu as the ambient hook with the opt-in gives the same
        # gradients as the plain computation.
        # save_on_cpu is the canonical ambient hook. On CPU we can only observe
        # numeric parity; device movement is asserted on CUDA below.
        fn = self._fn
        x = torch.randn(*self.INPUT_SHAPE, requires_grad=True)
        context_fn = self._mm_save_context_fn()
        with torch.autograd.graph.save_on_cpu():
            out = checkpoint(
                fn,
                x,
                use_reentrant=False,
                context_fn=context_fn,
                respect_saved_tensors_hooks=True,
            )
        out.backward()
        x_ref = x.detach().clone().requires_grad_()
        fn(x_ref).backward()
        torch.testing.assert_close(x.grad, x_ref.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "needs CUDA to observe offload")
    def test_save_on_cpu_device_movement(self):
        # [FUNCTION DOC] Check device movement with save_on_cpu-like hooks: the mm output reaches
        # the hook (on cuda) only with the opt-in. Requires CUDA.
        # save_on_cpu moves saved tensors to CPU. Under the default the SAC-kept
        # mm output never reaches the hook, so it stays on GPU; under the opt-in
        # the hook sees it and offloads it to CPU.
        def offloaded_device(respects):
            # [FUNCTION DOC] HELPER: Run a selective checkpoint under moving hooks and return a
            # mapping from packed shapes to the devices they were on.
            device_by_shape = {}

            def pack(t):
                # [FUNCTION DOC] HELPER: Record the device of the tensor by shape and move it
                # to CPU.
                device_by_shape[tuple(t.shape)] = t.device.type
                return t.cpu()

            def unpack(t):
                # [FUNCTION DOC] HELPER: Move the tensor back to CUDA.
                return t.cuda()

            x = torch.randn(*self.INPUT_SHAPE, device="cuda", requires_grad=True)
            context_fn = self._mm_save_context_fn()
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                out = checkpoint(
                    self._fn,
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                    respect_saved_tensors_hooks=respects,
                )
            out.backward()
            return device_by_shape

        self.assertNotIn(self.MM_SHAPE, offloaded_device(False))
        self.assertEqual(offloaded_device(True).get(self.MM_SHAPE), "cuda")

    def test_nested_checkpoint_ignores_internal_hooks(self):
        # [FUNCTION DOC] Check a selective checkpoint nested in a recompute-all checkpoint under a
        # user hook: with the opt-in the mm output is packed exactly once by the user hook (never
        # by internal hooks), never by default, and gradients match.
        # An inner selective checkpoint nested inside an outer recompute-all
        # checkpoint, both under a user hook. Torch installs its own internal
        # saved_tensors_hooks per region; the user hook must be resolved through
        # them, never mistaken for one. The single MUST_SAVE mm output must
        # reach the user hook exactly once under the opt-in (not doubled by
        # torch's internal hooks), and never under the default. Parity holds
        # either way.
        def inner(x, respects):
            # [FUNCTION DOC] HELPER: Selective checkpoint of a matrix multiply and relu with the
            # respect flag.
            return checkpoint(
                lambda x: torch.mm(x.t(), x).relu(),
                x,
                use_reentrant=False,
                context_fn=self._mm_save_context_fn(),
                respect_saved_tensors_hooks=respects,
            )

        def outer(x, respects):
            # [FUNCTION DOC] HELPER: Outer recompute-all checkpoint of the inner function, summed.
            recompute_all = functools.partial(
                create_selective_checkpoint_contexts, lambda *a, **k: False
            )
            return checkpoint(
                inner,
                x,
                respects,
                use_reentrant=False,
                context_fn=recompute_all,
                respect_saved_tensors_hooks=respects,
            ).sum()

        def observe(respects):
            # [FUNCTION DOC] HELPER: Run the nested case under shape-recording hooks and return
            # the packed shapes.
            packed = []

            def pack(t):
                # [FUNCTION DOC] HELPER: Record the shape and return the tensor.
                packed.append(tuple(t.shape))
                return t

            def unpack(t):
                # [FUNCTION DOC] HELPER: Return the tensor unchanged.
                return t

            x = torch.randn(*self.INPUT_SHAPE, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                out = outer(x, respects)
            out.backward()
            return packed

        self.assertNotIn(self.MM_SHAPE, observe(False))
        self.assertEqual(observe(True).count(self.MM_SHAPE), 1)

        x = torch.randn(*self.INPUT_SHAPE, requires_grad=True)
        outer(x, True).backward()
        x_ref = x.detach().clone().requires_grad_()
        outer(x_ref, True).backward()
        torch.testing.assert_close(x.grad, x_ref.grad)

    def test_plain_checkpoint_no_warning(self):
        # [FUNCTION DOC] Check plain checkpoint under an ambient hook sees the input and emits
        # no FutureWarning.
        # Plain (non-selective) checkpoint caches nothing via SAC, so an ambient
        # hook is unaffected and no FutureWarning about SAC should fire.
        packed = []

        def pack(t):
            # [FUNCTION DOC] HELPER: Record the shape and return the tensor.
            packed.append(tuple(t.shape))
            return t

        def unpack(t):
            # [FUNCTION DOC] HELPER: Return the tensor unchanged.
            return t

        x = torch.randn(*self.INPUT_SHAPE, requires_grad=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                out = checkpoint(self._fn, x, use_reentrant=False)
            out.backward()

        self.assertEqual(
            [w for w in caught if issubclass(w.category, FutureWarning)], []
        )
        self.assertIn(self.INPUT_SHAPE, packed)

    def test_reentrant_rejects_flag(self):
        # [FUNCTION DOC] Check the respect_saved_tensors_hooks option with reentrant checkpoint
        # raises ValueError saying it needs non-reentrant mode.
        # respect_saved_tensors_hooks is a non-reentrant-only feature, like
        # context_fn and debug.
        x = torch.randn(*self.INPUT_SHAPE, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            checkpoint(
                self._fn, x, use_reentrant=True, respect_saved_tensors_hooks=True
            )


class TestAutogradMultipleDispatch(TestCase):
    # [CLASS DOC] Device-generic tests for autograd formulas registered per dispatch key (default,
    # CUDA, nested tensor) through test-only operators; instantiated for CPU and CUDA.
    def test_autograd_multiple_dispatch_registrations(self, device):
        # [FUNCTION DOC] Check the per-dispatch-key derivative formulas: default gives gradient
        # plus one, CUDA gives double, nested tensor gives gradient squared.
        t = torch.randn(3, 3, device=device, requires_grad=True)
        # using _test_autograd_multiple_dispatch.fullcoverage which has
        # registrations in derivatives.yaml for Default, AutogradCUDA and NestedTensorAutograd
        out = torch._test_autograd_multiple_dispatch(t)
        grad = torch.randn(3, 3, device=device)
        out.backward(grad)

        if "cuda" not in device:
            # bogus default gradient registered for Autograd is grad + 1
            self.assertEqual(t.grad, grad + 1)
        else:
            # bogus gradient registered for AutogradCUDA is grad * 2
            self.assertEqual(t.grad, grad * 2)

        # test registered AutogradNestedTensor formula
        a = (
            torch.arange(6, dtype=torch.float, device=device)
            .reshape(2, 3)
            .requires_grad_(True)
        )
        b = (
            torch.arange(8, dtype=torch.float, device=device)
            .reshape(2, 4)
            .requires_grad_(True)
        )
        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)

        nt_out = torch._test_autograd_multiple_dispatch(nt)
        c = torch.randn(2, 3, device=device)
        d = torch.randn(2, 4, device=device)
        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        nt_out.backward(nt_grad)

        # bogus gradient for AutogradNestedTensor is grad * grad
        self.assertEqual(a.grad, c * c)
        self.assertEqual(b.grad, d * d)

    def test_autograd_composite_implicit_and_dispatch_registration(self, device):
        # [FUNCTION DOC] Check an op that is composite implicit except for a nested-tensor
        # formula: dense gradient equals the incoming gradient, nested gives gradient squared
        # plus gradient.
        t = torch.randn(3, 3, device=device, requires_grad=True)
        # using _test_autograd_multiple_dispatch.ntonly
        # which has registrations in derivatives.yaml for NestedTensorAutograd and otherwise is CompositeImplicit
        out = torch._test_autograd_multiple_dispatch(t, True)
        grad = torch.randn(3, 3, device=device)
        out.backward(grad)

        # t.grad is just out.grad by composite op since _test_autograd_multiple_dispatch is just a clone
        self.assertEqual(t.grad, grad)

        # test registered AutogradNestedTensor formula
        a = (
            torch.arange(6, dtype=torch.float, device=device)
            .reshape(2, 3)
            .requires_grad_(True)
        )
        b = (
            torch.arange(8, dtype=torch.float, device=device)
            .reshape(2, 4)
            .requires_grad_(True)
        )
        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)

        nt_out = torch._test_autograd_multiple_dispatch(nt, True)
        c = torch.randn(2, 3, device=device)
        d = torch.randn(2, 4, device=device)
        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        nt_out.backward(nt_grad)

        # bogus gradient for AutogradNestedTensor is grad * grad + grad
        self.assertEqual(a.grad, c * c + c)
        self.assertEqual(b.grad, d * d + d)

    def test_foward_mode_AD(self, device):
        # [FUNCTION DOC] Check forward AD is registered only for the default dispatch: raises
        # NotImplementedError on CUDA and works elsewhere.
        # check that forward mode AD is only registered for the Default
        # dispatch for _test_autograd_multiple_dispatch.fullcoverage and not AutogradCUDA

        primal = torch.randn(3, device=device)
        tangent = torch.randn(3, device=device)

        with fwAD.dual_level():
            dual_input = fwAD.make_dual(primal, tangent)

            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"

            if "cuda" in device:
                with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                    torch._test_autograd_multiple_dispatch(dual_input)
            else:
                torch._test_autograd_multiple_dispatch(dual_input)

    def test_view_copy(self, device):
        # [FUNCTION DOC] Check view_copy derivative formulas are generated per dispatch key: same
        # results as the view op, with the CUDA key adding one.
        # tests that view_copy derivative formulas are also generated per dispatch key
        # from their respective view ops in derivatives.yaml
        t = torch.randn(2, 2, device=device, requires_grad=True)
        t_ref = t.detach().clone().requires_grad_()
        # _test_autograd_multiple_dispatch_view does a .view(-1) on the input
        t_view = torch._test_autograd_multiple_dispatch_view(t_ref)
        t_view_copy = torch._test_autograd_multiple_dispatch_view_copy(t)

        grad = torch.randn(4, device=device)
        t_view_copy.backward(grad)
        t_view.backward(grad.clone())

        # forward and backward give the same shape + result
        self.assertEqual(t_view_copy, t_view)
        self.assertEqual(t.grad, t_ref.grad)
        # backward results are per-dispatch-key in derivatives.yaml
        if "cuda" in device:
            # gradient registered to AutogradCUDA is grad.reshape_as(self) + 1
            self.assertEqual(t.grad, grad.reshape_as(t) + 1)
        else:
            # Default gradient registered is grad.reshape_as(self)
            self.assertEqual(t.grad, grad.reshape_as(t))

    @onlyCPU
    def test_per_dispatch_key_input_saving(self, device):
        # [FUNCTION DOC] Check per-dispatch-key input saving: a sum input is not saved for regular
        # tensors but is saved for nested tensors (in-place later modification errors only for
        # nested). CPU only.
        # Tests that sum.dim_IntList's input is not saved for regular tensors but is saved for nested tensors
        def foo(x):
            # [FUNCTION DOC] HELPER: Clone the input, sum along the last dimension, modify the
            # clone in place and return the sum.
            # Don't modify the input inplace
            x = x.clone()
            res = x.sum(-1, keepdim=True)
            x.add_(x)
            return res

        inp = torch.rand(2, device=device, requires_grad=True)
        # sum's input is not saved for regular Tensors
        foo(inp).backward()

        # sum's input is saved for Nested Tensors
        nt = torch.nested.nested_tensor(
            [torch.rand(2), torch.rand(2)], device=device, requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            foo(nt).backward(
                torch.nested.nested_tensor(
                    [torch.rand(1), torch.rand(1)], device=device
                )
            )

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/181272"
    )
    @onlyCUDA
    def test_backward_single_threaded(self):
        # [FUNCTION DOC] Check that with multithreading disabled the custom backward runs on the
        # thread that ran forward, and with it enabled on a different thread. CUDA only; skipped
        # on Linux and slow runs.
        threads_eq = None

        class TestFn(Function):
            # [CLASS DOC] HELPER: Function recording the thread id in forward and comparing it
            # in backward.
            @staticmethod
            def forward(ctx, x, self):
                # [FUNCTION DOC] HELPER: Store the extra argument and the current thread id and
                # return a clone.
                ctx.self = self
                ctx.tid = threading.get_ident()
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Record whether the current thread equals the forward
                # thread and return the gradient and None.
                nonlocal threads_eq
                threads_eq = ctx.tid == threading.get_ident()
                return gO, None

        inp = torch.rand(10, device="cuda", requires_grad=True)

        with torch.autograd.set_multithreading_enabled(False):
            TestFn.apply(inp, None).sum().backward()
        self.assertTrue(threads_eq)

        TestFn.apply(inp, None).sum().backward()
        self.assertFalse(threads_eq)

    @onlyCUDA
    def test_backward_tls_stash(self):
        # [FUNCTION DOC] Check that an object stashed in the C++ thread local storage is visible
        # and mutable from backward threads on CUDA. CUDA only.
        local = threading.local()
        local.my_obj = {}
        local.my_obj[10] = 10
        test_self = self
        torch._C._stash_obj_in_tls("my_obj", local.my_obj)

        class TestFn(Function):
            # [CLASS DOC] HELPER: Function whose backward checks and mutates a stashed object.
            @staticmethod
            def forward(ctx, x, self):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Check the stash key exists and the stored mapping has the
                # expected value, change the value and return the gradient and None.
                test_self.assertTrue(torch._C._is_key_in_tls("my_obj"))
                test_self.assertTrue(torch._C._get_obj_in_tls("my_obj")[10] == 10)
                torch._C._get_obj_in_tls("my_obj")[10] = 5
                return gO, None

        inp = torch.rand(10, device="cuda", requires_grad=True)

        TestFn.apply(inp, None).sum().backward()
        self.assertEqual(local.my_obj[10], 5)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/181323"
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS or IS_WINDOWS,
        "https://github.com/pytorch/pytorch/issues/181228",
    )
    @unittest.skipIf(
        IS_WINDOWS and IS_ARM64, "Fails on Windows ARM64; see issue #181228"
    )
    def test_is_retain_graph(self):
        # [FUNCTION DOC] Check the private query of whether the current graph task keeps its graph
        # reports the retain_graph argument. Skipped on several platform configurations.
        retain_graph_set = False

        class TestFn(Function):
            # [CLASS DOC] HELPER: Function recording the current graph task keep-graph flag
            # in backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Record the keep-graph flag and return the gradient
                # and None.
                nonlocal retain_graph_set
                retain_graph_set = (
                    torch._C._autograd._get_current_graph_task_keep_graph()
                )
                return gO, None

        inp = torch.rand(10, requires_grad=True)

        out = TestFn.apply(inp)
        self.assertFalse(retain_graph_set)
        out.sum().backward(retain_graph=True)
        self.assertTrue(retain_graph_set)
        out.sum().backward(retain_graph=False)
        self.assertFalse(retain_graph_set)

    def test_set_sequence_nr(self):
        # [FUNCTION DOC] Check setting a node sequence number changes backward execution order:
        # higher numbers run first, giving the order c, a, b, and correct gradients.
        x = torch.randn((10,), dtype=torch.float32, requires_grad=True)
        y = torch.randn((10,), dtype=torch.float32, requires_grad=True)
        z = torch.randn((10,), dtype=torch.float32, requires_grad=True)

        a = x + y
        b = y + z
        c = a + b

        self.assertIsNotNone(a.grad_fn)
        self.assertIsNotNone(b.grad_fn)
        self.assertIsNotNone(c.grad_fn)

        a.grad_fn._set_sequence_nr(100)
        b.grad_fn._set_sequence_nr(99)
        c.grad_fn._set_sequence_nr(98)

        self.assertEqual(a.grad_fn._sequence_nr(), 100)
        self.assertEqual(b.grad_fn._sequence_nr(), 99)
        self.assertEqual(c.grad_fn._sequence_nr(), 98)

        def log_grad_order(grad: torch.Tensor, name: str, order):
            # [FUNCTION DOC] HELPER: Hook appending a name to an order list and returning
            # the gradient.
            order.append(name)
            return grad

        order = []
        a.register_hook(partial(log_grad_order, name="a", order=order))
        b.register_hook(partial(log_grad_order, name="b", order=order))
        c.register_hook(partial(log_grad_order, name="c", order=order))

        c.sum().backward()

        # Expect to see that even though c has the smallest sequence number, it is still the first node to get run in autograd.
        # Also check that although a comes first during the forward, after giving it priority with sequence_nr,
        # its autograd node is run before that of b.
        self.assertEqual(order, ["c", "a", "b"])

        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, 2 * torch.ones_like(x))
        self.assertEqual(z.grad, torch.ones_like(x))

    def test_atan2_zero_gradient(self):
        # [FUNCTION DOC] Check atan2 at the origin gives zero gradients for both inputs.
        x = torch.tensor([0.0], requires_grad=True)
        y = torch.tensor([0.0], requires_grad=True)
        z = torch.atan2(x, y)
        z.backward()
        self.assertEqual(x.grad, torch.zeros_like(x))
        self.assertEqual(y.grad, torch.zeros_like(y))

    # Test that torch.autograd.backward respects __torch_function__ on tensor subclasses.
    def test_backward_respects_torch_function(self):
        # [FUNCTION DOC] Check backward dispatches through the torch function protocol of a
        # tensor subclass: a subclass wrapper triggers a flag and the unwrapped call computes the
        # right gradient.
        backward_called_with_subclass = [False]

        class AsyncTensorLike(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass that records when backward is called
            # with it and unwraps its inner tensor in function and dispatch handlers.
            """Tensor subclass that tracks when backward is called with it."""

            @staticmethod
            def __new__(cls, data):
                # [FUNCTION DOC] HELPER: Create a wrapper subclass with the metadata of the
                # data tensor.
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.size(),
                    strides=data.stride(),
                    storage_offset=data.storage_offset(),
                    dtype=data.dtype,
                    layout=data.layout,
                    device=data.device,
                    requires_grad=data.requires_grad,
                )

            def __init__(self, data):
                # [FUNCTION DOC] HELPER: Store the inner tensor.
                # store the inner tensor
                self._data = data

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: For backward, set the flag and call it with inner tensors
                # substituted; for others, call with unwrapped arguments.
                kwargs = kwargs or {}
                if func is torch.autograd.backward:
                    backward_called_with_subclass[0] = True
                    # unwrap inner tensors and call the real backward
                    new_args = []
                    for arg in args:
                        if isinstance(arg, tuple):
                            new_args.append(
                                tuple(a._data if isinstance(a, cls) else a for a in arg)
                            )
                        elif isinstance(arg, cls):
                            new_args.append(arg._data)
                        else:
                            new_args.append(arg)
                    return func(*new_args, **kwargs)
                return func(
                    *tuple(a._data if isinstance(a, cls) else a for a in args), **kwargs
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Unwrap subclass instances in arguments and keyword
                # arguments and run the op.
                def unwrap(t):
                    # [FUNCTION DOC] HELPER: Return the inner tensor of a subclass instance,
                    # otherwise the value as is.
                    return t._data if isinstance(t, cls) else t

                return func(
                    *torch.utils._pytree.tree_map(unwrap, args),
                    **torch.utils._pytree.tree_map(unwrap, kwargs or {}),
                )

        x = torch.randn(3, requires_grad=True)
        y = x * 2
        wrapped = AsyncTensorLike(y)
        torch.autograd.backward(wrapped, torch.ones_like(y))

        self.assertTrue(
            backward_called_with_subclass[0],
            "backward() should invoke __torch_function__ on tensor subclasses",
        )
        self.assertEqual(x.grad, 2 * torch.ones_like(x))

    def test_trace_backward_nonsquare_171704(self):
        # [FUNCTION DOC] Regression check for gradient of trace on non-square matrices: ones on
        # the diagonal for several shapes.
        # https://github.com/pytorch/pytorch/issues/171704
        for shape in [(5, 2), (7, 3), (4, 1), (1, 5), (2, 5)]:
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                torch.trace(x).backward()
                expected = torch.zeros(shape, dtype=torch.float64)
                for i in range(min(shape)):
                    expected[i, i] = 1.0
                self.assertEqual(x.grad, expected)


@skipIfTorchDynamo("tests eager C++ error paths that Dynamo does not reproduce")
class TestFunctionAssertMessages(TestCase):
    # [CLASS DOC] Regression tests that error messages of the Function C++ assertion helper
    # include the offending type name or index when the context attributes have wrong types.
    # Skipped under dynamo.
    # THPFunction_assert forwards to a printf-style formatter. Regression tests
    # that the dynamic content (offending type name / index) is not silently
    # dropped from the error message.
    def _apply(self, forward_fn):
        # [FUNCTION DOC] HELPER: Apply a custom Function whose forward is the given function to a
        # random input requiring grad.
        class F(Function):
            # [CLASS DOC] HELPER: Identity-like Function delegating forward to the given function.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Call the supplied forward function with the context
                # and input.
                return forward_fn(ctx, x)

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return g

        return F.apply(torch.randn(2, requires_grad=True))

    def test_dirty_tensors_not_tuple_reports_type(self):
        # [FUNCTION DOC] Check assigning a non-tuple to the dirty tensors attribute raises
        # RuntimeError reporting the type.
        def fwd(ctx, x):
            # [FUNCTION DOC] HELPER: Set a string as dirty tensors on ctx and return the input.
            ctx.dirty_tensors = "notatuple"
            return x

        with self.assertRaisesRegex(RuntimeError, r"but is .*str"):
            self._apply(fwd)

    def test_mark_dirty_element_reports_index_and_type(self):
        # [FUNCTION DOC] Check a non-tensor element of the dirty tensors raises RuntimeError
        # reporting argument index and type.
        def fwd(ctx, x):
            # [FUNCTION DOC] HELPER: Set a tuple with a string as dirty tensors on ctx and return
            # the input.
            ctx.dirty_tensors = ("notatensor",)
            return x

        with self.assertRaisesRegex(RuntimeError, r"argument 0 is of type .*str"):
            self._apply(fwd)

    def test_to_save_not_tuple_reports_type(self):
        # [FUNCTION DOC] Check assigning a non-tuple to the saved tensors attribute raises
        # RuntimeError reporting the type.
        def fwd(ctx, x):
            # [FUNCTION DOC] HELPER: Set a string as the to-save attribute on ctx and return
            # the input.
            ctx.to_save = "notatuple"
            return x

        with self.assertRaisesRegex(RuntimeError, r"but is .*str"):
            self._apply(fwd)

    def test_non_differentiable_not_tuple_reports_type(self):
        # [FUNCTION DOC] Check assigning a non-tuple to the non-differentiable attribute raises
        # RuntimeError reporting the type.
        def fwd(ctx, x):
            # [FUNCTION DOC] HELPER: Set a string as the non-differentiable attribute on ctx and
            # return the input.
            ctx.non_differentiable = "notatuple"
            return x

        with self.assertRaisesRegex(RuntimeError, r"but is .*str"):
            self._apply(fwd)

    def test_mark_non_differentiable_element_reports_type(self):
        # [FUNCTION DOC] Check a non-tensor element of the non-differentiable attribute raises
        # RuntimeError reporting the type.
        def fwd(ctx, x):
            # [FUNCTION DOC] HELPER: Set a tuple with a string as the non-differentiable attribute
            # on ctx and return the input.
            ctx.non_differentiable = ("notatensor",)
            return x

        with self.assertRaisesRegex(RuntimeError, r"but got .*str"):
            self._apply(fwd)

    def test_saved_for_forward_not_tuple_reports_type(self):
        # [FUNCTION DOC] Check assigning a non-tuple to the saved-for-forward attribute in
        # setup_context raises RuntimeError reporting the type.
        class F(Function):
            # [CLASS DOC] HELPER: Function with separate setup_context that sets a string as the
            # saved-for-forward attribute.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Return a clone.
                return x.clone()

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Set a string on the saved-for-forward attribute.
                ctx.saved_for_forward = "notatuple"

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Return the gradient.
                return g

        with self.assertRaisesRegex(RuntimeError, r"but is .*str"):
            F.apply(torch.randn(2, requires_grad=True))


# Import test cases from below autograd/ here. These are found
# implicitly by the loader, so Flake8 thinks they are unused, hence
# the suppressions.

from autograd.test_complex import TestAutogradComplex  # noqa: F401
from autograd.test_functional import TestAutogradFunctional  # noqa: F401
from autograd.test_logging import TestAutogradLogging  # noqa: F401


# e.g., TestAutogradDeviceTypeCPU and TestAutogradDeviceTypeCUDA
instantiate_device_type_tests(TestAutogradDeviceType, globals(), except_for=None)

instantiate_device_type_tests(
    TestAutogradMultipleDispatch, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestAutogradStreamSynchronization, globals(), except_for=None
)

instantiate_parametrized_tests(TestAutograd)
instantiate_parametrized_tests(TestNestedCheckpoint)

if __name__ == "__main__":
    run_tests()
