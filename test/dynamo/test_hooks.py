# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests dynamo support for autograd hooks: tensor register_hook (lambda, local,
# global and closure hooks, handles returned or removed), hooks on intermediates, post-
# accumulate-grad hooks, nn.Module forward hooks and guards, interaction with compiled
# autograd, aot_eager and inductor backends, and autograd-cache behavior. Defines module-level
# hook functions and a global handle variable. At import time instantiates device-specific
# test classes for accelerators (including xpu) by calling instantiate_device_type_tests.

import contextlib
import functools
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.compile import nop
from torch._dynamo import compiled_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import skipIfRocm
from torch.utils.hooks import RemovableHandle


def compiler_fn(gm):
    # [FUNCTION DOC] HELPER: Compiler callback for compiled autograd; compiles the given graph
    # module with the inductor backend, full graph and dynamic shapes, and returns the result.
    return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=True)


def global_hook_0(grad):
    # [FUNCTION DOC] HELPER: Gradient hook that returns the gradient multiplied by four.
    return grad * 4


def global_hook_1(grad):
    # [FUNCTION DOC] HELPER: Gradient hook that returns the gradient divided by two.
    return grad / 2


def global_hook_2(grad):
    # [FUNCTION DOC] HELPER: Gradient hook that returns the gradient multiplied by three.
    return grad * 3


h0 = None


class ClassWithVal:
    # [CLASS DOC] HELPER: Simple holder with one attribute val, used as a closure or partial
    # argument captured by hooks in guard and recompile tests.
    def __init__(self, val):
        # [FUNCTION DOC] HELPER: Stores the given value as the attribute val.
        self.val = val


class HooksTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Main test case for hook support in dynamo. Each test compiles a function or
    # module that registers hooks, runs backward and checks gradient values, frame counts,
    # recompilation, generated graph text, or errors, often comparing against eager results.
    def test_tensor_only_register_hook_in_graph_lambda(self):
        # [FUNCTION DOC] Compiled function only registers a lambda hook (doubling gradients)
        # on its input and returns it. Backward gives doubled gradients and no frame is
        # compiled (frame count zero).
        def fn(x):
            # [FUNCTION DOC] HELPER: Registers a gradient-doubling lambda hook on x and
            # returns x.
            x.register_hook(lambda grad: grad * 2)
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_register_hook_in_graph_lambda(self):
        # [FUNCTION DOC] Registers a lambda hook (doubling) on the input inside a function
        # that also squares two other tensors. Checks doubled gradients and one compiled
        # frame.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Registers a doubling hook on x and returns x with squares
            # of y and z.
            x.register_hook(lambda grad: grad * 2)
            return x, y * y, z * z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_lambda(self):
        # [FUNCTION DOC] Registers a doubling hook, keeps its handle, does some tensor work,
        # removes the handle, then registers a tripling hook. Checks the gradient reflects
        # only the tripling hook and that one frame was compiled.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Registers and removes a doubling hook, then registers a
            # tripling hook, returning x, y squared and z squared.
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            x.register_hook(lambda grad: grad * 3)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_multi_handle_return(self):
        # [FUNCTION DOC] Returns the hook handle twice (under two names) from the compiled
        # function. Checks doubled gradients, one frame, handles not none and equal to each
        # other.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Registers a doubling hook and returns tensors plus the
            # handle and an alias of it.
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle
            z = z * z
            return x, y * y, z, handle, h2

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertNotEqual(h, None)
        self.assertNotEqual(h2, None)
        self.assertEqual(h2, h)

    def test_tensor_register_hook_repeated_handle_return(self):
        # [FUNCTION DOC] Returns the same handle object twice. Checks doubled gradients, one
        # frame, that the handle is a RemovableHandle and that both returned handles are the
        # identical object.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Registers a doubling hook and returns tensors plus the
            # handle twice.
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle  # noqa: F841
            z = z * z
            return x, y * y, z, handle, handle

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_removed_handle_return(self):
        # [FUNCTION DOC] Registers a hook, removes its handle twice inside a full-graph
        # function and returns the handle twice. Checks the gradient is unchanged (hook
        # removed), one frame, the handle is a RemovableHandle and identical for both returns.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Registers a doubling hook, removes it two times, and
            # returns tensors plus the handle twice.
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            handle.remove()
            return x, y * y, z, handle, handle

        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_tensor_register_hook_repeated_handle_not_local(self):
        # [FUNCTION DOC] Stores the hook handle as an attribute of a module passed in,
        # compiled with a full graph. Checks doubled gradients, one frame and that the
        # attribute is no longer none.
        def fn(x, y, z, mod):
            # [FUNCTION DOC] HELPER: Registers a doubling hook and stores the handle on the
            # module argument; returns x and squares of y and z.
            mod.handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts, fullgraph=True)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

        mod = torch.nn.Module()
        mod.handle = None

        v, y, z = fn(v, torch.randn([2, 2]), torch.randn([2, 2]), mod)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

        self.assertNotEqual(mod.handle, None)

    def test_tensor_only_register_hook_in_graph_local(self):
        # [FUNCTION DOC] Registers a locally defined hook function (doubling) on the input
        # only; checks doubled gradients and zero compiled frames.
        def local_hook(grad):
            # [FUNCTION DOC] HELPER: Gradient hook that doubles the gradient.
            return grad * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Registers the local hook on x and returns x.
            x.register_hook(local_hook)
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_only_register_hook_in_graph_local_inner(self):
        # [FUNCTION DOC] Defines a hook inside the compiled function and registers it on both
        # the input and an intermediate product; backward from one output element. Checks
        # doubled gradient on that element and one compiled frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Defines a doubling hook, registers it on x and on x
            # squared, returns both.
            def local_hook(grad):
                # [FUNCTION DOC] HELPER: Gradient hook that doubles the gradient.
                return grad * 2

            z = x * x
            x.register_hook(local_hook)
            z.register_hook(local_hook)
            return x, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v[0].backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v[0].grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_local(self):
        # [FUNCTION DOC] Registers a module-local doubling hook function inside a function
        # that also squares other tensors. Checks doubled gradients and one frame.
        def local_hook(grad):
            # [FUNCTION DOC] HELPER: Gradient hook that doubles the gradient.
            return grad * 2

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Registers the local hook on x and returns x with squares
            # of y and z.
            x.register_hook(local_hook)
            return x, y * y, z * z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_local(self):
        # [FUNCTION DOC] Registers a doubling local hook, does tensor work, removes the
        # handle, registers a tripling hook. Checks the gradient is tripled (the first hook
        # has no effect).
        def local_hook(grad):
            # [FUNCTION DOC] HELPER: Gradient hook that doubles the gradient.
            return grad * 2

        def local_hook2(grad):
            # [FUNCTION DOC] HELPER: Gradient hook that triples the gradient.
            return grad * 3

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Registers and removes the first hook, registers the
            # second, returns x, y squared and z squared.
            handle = x.register_hook(local_hook)
            z = z * z
            handle.remove()
            x.register_hook(local_hook2)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))

    def test_tensor_register_global_hook(self):
        # [FUNCTION DOC] Registers the module-level hook (times four) on the input. Checks
        # gradients are four times the incoming gradient and one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Registers the global times-four hook and returns x and x
            # squared.
            x.register_hook(global_hook_0)
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks(self):
        # [FUNCTION DOC] Registers three module-level hooks (times four, divided by two, times
        # three) in order. Checks the gradient is six times the incoming one and one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Registers the three global hooks on x and returns x and x
            # squared.
            x.register_hook(global_hook_0)  # * 4
            x.register_hook(global_hook_1)  # / 2
            x.register_hook(global_hook_2)  # * 3
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks_handles_in_list(self):
        # [FUNCTION DOC] Returns the handles of three registered hooks, runs backward
        # (gradient six times), removes all handles, runs backward again and checks the
        # gradient grows only by the unhooked amount. One frame is compiled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Registers the three global hooks and returns x, x squared
            # and the three handles.
            h0 = x.register_hook(global_hook_0)  # * 4
            h1 = x.register_hook(global_hook_1)  # / 2
            h2 = x.register_hook(global_hook_2)  # * 3
            return x, x * x, h0, h1, h2

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r, handle_0, handle_1, handle_2 = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))
        handle_0.remove()
        handle_1.remove()
        handle_2.remove()

        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, torch.tensor([7.0, 14.0, 21.0]))

        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_global_hooks_handles_in_list(self):
        # [FUNCTION DOC] Stores a hook handle in the module-level global variable h0 from
        # compiled code. Checks the global is set, the hook applies (times four), removing the
        # handle stops it, and the frame count is zero (marked not yet implemented in the
        # test).
        def fn(x):
            # [FUNCTION DOC] HELPER: Declares h0 global, stores the registered hook handle in
            # it, returns x and x squared. Side effect: assigns the global.
            global h0
            h0 = x.register_hook(global_hook_0)  # * 4
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r = fn(v)

        self.assertIsNotNone(h0)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))
        h0.remove()

        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, torch.tensor([5.0, 10.0, 15.0]))

        # NYI!
        self.assertEqual(cnts.frame_count, 0)

    def test_hook_on_intermediate(self):
        # [FUNCTION DOC] Registers a plus-one hook on an intermediate tensor in a full-graph
        # eager-backend function; checks the input gradient equals eager's.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, registers a plus-one hook on the result, and
            # returns its sum.
            y = x * 2
            y.register_hook(lambda grad: grad + 1)
            return y.sum()

        x_compiled = torch.randn(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        x_eager = x_compiled.detach().clone().requires_grad_(True)
        result_eager = fn(x_eager)
        result_eager.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)

    def test_hook_on_intermediate_with_container(self):
        # [FUNCTION DOC] Hook on an intermediate that is also stored into a list and
        # dictionary and read back after being appended. Compares the compiled gradient with
        # eager, checks it equals ten everywhere, and compares the recorded graph code with an
        # inline expected string showing that the hook-registration higher-order op output is
        # returned and used in later computation.
        glb_list = []
        glb_dict = {}

        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, stores it in the enclosing list and
            # dictionary, reads it back (a is three times it, b is it unchanged), registers a
            # plus-one hook on y, and returns the sum of a and b.
            y = x * 2
            glb_list.append(y)
            glb_dict["tensor"] = y
            a = glb_list[0] * 3  # Should use output of register_hook
            b = glb_dict["tensor"]
            y.register_hook(lambda grad: grad + 1)
            return (a + b).sum()

        glb_list.clear()
        glb_dict.clear()
        x_eager = torch.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        glb_list.clear()
        glb_dict.clear()
        x_compiled = torch.ones(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)
        # Without hook: dloss/dy = 4, dloss/dx = 8
        # With hook (+1): hooked = 5, dloss/dx = 10
        self.assertEqual(x_compiled.grad, torch.full_like(x_compiled, 10.0))

        glb_list.clear()
        glb_dict.clear()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.ones(4, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    mul = l_x_ * 2;  l_x_ = None
    mul_1 = mul * 3
    hook_body_0 = self.hook_body_0
    register_hook = torch.ops.higher_order.register_hook(mul, hook_body_0);  mul = hook_body_0 = None
    add = mul_1 + register_hook;  mul_1 = None
    sum_1 = add.sum();  add = None
    return (sum_1, register_hook)""",
        )

    def test_hook_on_intermediate_used_before_and_after(self):
        # [FUNCTION DOC] Uses an intermediate both before and after registering a doubling
        # hook on it; checks compiled and eager input gradients match.
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes uses of y before and after hook registration and
            # returns the sum of the combination.
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = torch.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(2, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_with_higher_order_op(self):
        # [FUNCTION DOC] Registers a doubling hook on an intermediate and then passes it into
        # a conditional higher-order op with two branches; checks compiled and eager gradients
        # match.
        def fn(x):
            # [FUNCTION DOC] HELPER: Registers the hook, builds a conditional op over whether
            # x sums positive, and returns the sum of its result.
            y = x * 2
            y.register_hook(lambda g: g * 2)

            def true_fn(t):
                # [FUNCTION DOC] HELPER: Branch that adds one.
                return t + 1

            def false_fn(t):
                # [FUNCTION DOC] HELPER: Branch that subtracts one.
                return t - 1

            z = torch.cond(x.sum() > 0, true_fn, false_fn, (y,))
            return z.sum()

        x_eager = torch.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(3, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_returns_none(self):
        # [FUNCTION DOC] Hook that returns none (leaves gradient unchanged). Checks compiled
        # and eager gradients agree and are all two.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, registers a hook returning none, returns the
            # sum.
            y = x * 2
            y.register_hook(lambda g: None)
            return y.sum()

        x_eager = torch.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)
        self.assertEqual(x_compiled.grad, torch.full_like(x_compiled, 2.0))

    def test_hook_has_side_effect(self):
        # [FUNCTION DOC] Same body as the before-and-after-hook test: uses an intermediate
        # before and after a doubling hook registration and compares compiled to eager
        # gradients.
        # KNOWN ISSUE: duplicates test_hook_on_intermediate_used_before_and_after and does not
        # exercise anything about side effects despite its name.
        def fn(x):
            # [FUNCTION DOC] HELPER: Uses y before and after hook registration and returns the
            # sum of the combination.
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = torch.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(2, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_bwd_inside_side_effects(self):
        # [FUNCTION DOC] Hook that appends its gradient to an enclosing list (side effect
        # inside the backward hook). Expects compiling and calling the function to raise
        # Unsupported with a message about an unsafe side effect.
        global_list = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, defines and registers a hook that mutates the
            # enclosing list, returns the sum of y plus x.
            y = x * 2

            def _hook(grad):
                # [FUNCTION DOC] HELPER: Appends the gradient to the enclosing list and
                # returns the gradient doubled.
                global_list.append(grad)
                return grad * 2

            y.register_hook(_hook)
            z = y + x
            return z.sum()

        x_eager = torch.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        global_list.clear()

        x_compiled = torch.ones(3, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Unsafe side effect"
        ):
            _ = compiled_fn(x_compiled)

    def test_hook_on_intermediate_from_split(self):
        # [FUNCTION DOC] Registers a plus-one hook on one chunk of a split input after the
        # chunks were concatenated. Checks compiled and eager gradients match and compares the
        # recorded graph code against an inline expected text.
        def fn(x):
            # [FUNCTION DOC] HELPER: Splits x, concatenates the pieces, registers a plus-one
            # hook on the first piece, and returns the sum of both.
            splits = x.split(2)
            result = torch.cat(splits)  # use splits before register_hook
            y = splits[0]
            y.register_hook(lambda g: g + 1)
            return result.sum() + y.sum()

        x_eager = torch.ones(6, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(6, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.ones(6, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    split = l_x_.split(2);  l_x_ = None
    getitem = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2];  split = None
    cat = torch.cat((getitem, getitem_1, getitem_2));  getitem_1 = getitem_2 = None
    hook_body_0 = self.hook_body_0
    register_hook = torch.ops.higher_order.register_hook(getitem, hook_body_0);  getitem = hook_body_0 = None
    sum_1 = cat.sum();  cat = None
    sum_2 = register_hook.sum();  register_hook = None
    add = sum_1 + sum_2;  sum_1 = sum_2 = None
    return (add,)""",
        )

    def test_intermediary_hooks(self):
        # [FUNCTION DOC] Registers a doubling hook on an intermediate in a full-graph
        # function. Checks the compiled output matches eager and the gradient on the input is
        # two.
        def simple_hook(g):
            # [FUNCTION DOC] HELPER: Gradient hook that doubles the gradient.
            return g * 2

        def f(x):
            # [FUNCTION DOC] HELPER: Adds one, registers the hook on the result, adds one
            # again and returns it.
            y = x + 1
            y.register_hook(simple_hook)
            z = y + 1
            return z

        out = torch.randn(1, requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(f, backend=cnts, fullgraph=True)
        res = fn(out)
        res.backward()
        self.assertEqual(res, f(out))
        self.assertEqual(out.grad, torch.Tensor([2.0]))

    def test_intermediary_hooks_same_on_aot_eager(self):
        # [FUNCTION DOC] Two partial-wrapped hooks on an intermediate in a module. Runs eager,
        # aot module simplified, and dynamo with aot_eager under compiled autograd; checks
        # outputs and input gradients agree across all three.
        def my_hook(grad, *, k=0):
            # [FUNCTION DOC] HELPER: Hook adding a keyword-only constant k to the gradient.
            return grad + k

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering two partial hooks (adding three and four)
            # on an intermediate and returning a tuple with the tripled intermediate.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Doubles x, registers two partial hooks, triples the
                # result and returns it in a tuple.
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="aot_eager", fullgraph=True)(x2)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_input_hooks_same(self):
        # [FUNCTION DOC] For each of the eager, aot_eager and inductor backends, registers a
        # partial hook on the module's input inside forward; runs eager, aot module simplified
        # and dynamo (with compiled autograd for backward). Checks outputs and input gradients
        # agree.
        backends = ["eager", "aot_eager", "inductor"]
        for backend in backends:

            def my_hook(grad, *, k=0):
                # [FUNCTION DOC] HELPER: Hook adding a keyword-only constant k to the
                # gradient.
                return grad + k

            hook = functools.partial(my_hook, k=3)

            class MyMod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module registering the enclosing partial hook on its
                # input, then multiplying by two and three; returns a tuple.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Registers the hook on x, multiplies by two then
                    # three, returns a tuple.
                    x.register_hook(hook)
                    y = x.mul(2)
                    z = y.mul(3)
                    return (z,)

            mod = MyMod()
            x0 = torch.ones(4, requires_grad=True)
            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            x1 = torch.ones(4, requires_grad=True)
            mod_compiled = aot_module_simplified(mod, (x1,), nop)
            aot_out = mod_compiled(x1)
            aot_out[0].backward(torch.ones(4))

            x2 = torch.ones(4, requires_grad=True)
            dynamo_out = torch.compile(mod, backend=backend, fullgraph=True)(x2)
            with compiled_autograd._enable(compiler_fn):
                dynamo_out[0].backward(torch.ones(4))

            self.assertEqual(dynamo_out, aot_out)
            self.assertEqual(dynamo_out, eager_out)

            self.assertEqual(x0.grad, x1.grad)
            self.assertEqual(x0.grad, x2.grad)

    def test_intermediary_hooks_same_on_inductor(self):
        # [FUNCTION DOC] Same as the aot_eager intermediary hooks test but with the inductor
        # backend for the dynamo run: eager, aot module simplified and compiled autograd
        # outputs and gradients must agree.
        def my_hook(grad, *, k=0):
            # [FUNCTION DOC] HELPER: Hook adding a keyword-only constant k to the gradient.
            return grad + k

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering two partial hooks (adding three and four)
            # on an intermediate and returning a tuple with the tripled intermediate.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Doubles x, registers two partial hooks, triples the
                # result and returns it in a tuple.
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor(self):
        # [FUNCTION DOC] Two hooks sharing a stateful object whose counter alternates between
        # squaring and doubling the gradient. Checks eager leaves the count at two, then the
        # inductor-compiled run with compiled autograd gives the same output and gradient and
        # the count reaches four.
        class SomePyClass:
            # [CLASS DOC] HELPER: Holds a counter; each call to do_stuff squares the gradient
            # on even counts and doubles it on odd counts, then increments the counter.
            count = 0

            def do_stuff(self, grad):
                # [FUNCTION DOC] HELPER: Returns the gradient squared or doubled depending on
                # the parity of the counter, then increments the counter (mutates state).
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            # [FUNCTION DOC] HELPER: Hook delegating to the state object do_stuff method.
            return obj.do_stuff(grad)

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering two partial hooks bound to the passed
            # state object on an intermediate.
            def forward(self, x, obj):
                # [FUNCTION DOC] HELPER: Doubles x, registers two hooks using obj, triples the
                # result and returns a tuple.
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(torch.ones(4))

        # Eager 2
        self.assertEqual(obj.count, 2)
        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, eager_out)

        # Eager 2 + compiled 2
        self.assertEqual(obj.count, 4)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor_with_graph_break(
        self,
    ):
        # [FUNCTION DOC] Like the state-mutation test but the object method also stores the
        # gradient as a string and prints (graph break). With inductor and compiled autograd,
        # expects backward to raise Unsupported mentioning repr on tensor; afterwards the
        # counter stays at the two from the eager run.
        class SomePyClass:
            # [CLASS DOC] HELPER: Holds a string attribute and a counter; its method
            # stringifies the gradient, applies square or double by counter parity, prints,
            # increments the counter.
            grad_as_str = "None"
            count = 0

            def write_grad_as_str_and_do_stuff(self, grad):
                # [FUNCTION DOC] HELPER: Records the gradient string, computes square or
                # double by parity, prints a message, increments the counter and returns the
                # result.
                self.grad_as_str = str(grad)
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                print("Break!")
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            # [FUNCTION DOC] HELPER: Hook delegating to the state object method.
            return obj.write_grad_as_str_and_do_stuff(grad)

        class MyMod(torch.nn.Module):
            # [FUNCTION DOC] HELPER: Module registering two partial hooks bound to the passed
            # state object.
            def forward(self, x, obj):
                # [FUNCTION DOC] HELPER: Doubles x, registers two hooks using obj, triples the
                # result and returns a tuple.
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, r"repr\(\) on tensor"
            ):
                dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(obj.count, 2)

    def test_register_hook_partial_guarding(
        self,
    ):
        # [FUNCTION DOC] Compiles a module whose hook is a partial bound to an object with a
        # val attribute, calling with different objects under compiled autograd. Expects only
        # one frame compiled overall across objects holding tensor values and an object
        # holding an integer.
        def some_hook(grad, *, obj):
            # [FUNCTION DOC] HELPER: Hook adding the object's val to the gradient.
            return grad + obj.val

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering a partial hook bound to the passed object
            # on an intermediate.
            def forward(self, x, obj):
                # [FUNCTION DOC] HELPER: Doubles x, registers the hook, triples the result and
                # returns a tuple.
                y = x.mul(2)
                hook1 = functools.partial(some_hook, obj=obj)
                y.register_hook(hook1)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        obj3 = ClassWithVal(11)
        cnt = torch._dynamo.testing.CompileCounter()

        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)

        with compiled_autograd._enable(compiler_fn):
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x1, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj2)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj3)
            self.assertEqual(cnt.frame_count, 1)

    def test_hook_with_closure(self):
        # [FUNCTION DOC] Hook on the input as a lambda closing over an object value. Compares
        # gradients against eager for two objects under compiled autograd; expects one forward
        # frame and one backward frame (no recompile for different objects).
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Applies sine, registers a hook adding the object's val to
            # the input gradient, applies sine again.
            y = x.sin()
            x.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_hook_with_nested_closure(self):
        # [FUNCTION DOC] Hook closing over a tensor created in a nested function. Compares
        # gradients with eager under compiled autograd and expects one forward frame and one
        # backward frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls an inner function and returns its result.
            def run():
                # [FUNCTION DOC] HELPER: Applies sine, registers a hook adding that
                # intermediate to the gradient of x, applies sine again.
                y = x.sin()
                x.register_hook(lambda grad: grad + y)
                z = y.sin()
                return z

            return run()

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        fn(x0).sum().backward()
        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x1).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x1.grad)

    def test_intermediate_hook_with_closure_eager(self):
        # [FUNCTION DOC] Like the input closure test but the hook is registered on an
        # intermediate with eager forward counting; compared with eager across two objects,
        # one forward and one backward frame.
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Applies sine, registers a closure hook on it, applies
            # sine again.
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_intermediate_hook_with_closure_aot(self):
        # [FUNCTION DOC] Closure hook on an intermediate with the aot_eager backend and
        # compiled autograd compiled by a counting backend. Gradients match eager for two
        # objects, and only one backward frame is compiled.
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Applies sine, registers a closure hook on it, applies
            # sine again.
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend="aot_eager", fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_no_recompile_on_hook_identity_change(self):
        # [FUNCTION DOC] Registers a function as hook twice, compiles with aot_eager, then
        # rebinds the enclosing hook name to a different function and calls again. Checks only
        # one frame compiled both times and gradients equal eager's.
        def my_hook(grad, k=0):
            # [FUNCTION DOC] HELPER: Hook adding a default-valued constant to the gradient.
            return grad + k

        def my_hook2(grad):
            # [FUNCTION DOC] HELPER: Hook doubling the gradient; later substituted for the
            # first hook.
            return grad * 2

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering the enclosing hook function twice on an
            # intermediate and returning a tuple.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Doubles x, registers the enclosing hook twice,
                # triples the result, returns a tuple.
                y = x.mul(2)
                y.register_hook(my_hook)
                y.register_hook(my_hook)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            comp_mod = torch.compile(mod, backend=cnts, fullgraph=True)
            comp_out = comp_mod(x1)
            comp_out[0].backward(torch.ones(4))

            self.assertEqual(cnts.frame_count, 1)
            my_hook = my_hook2
            self.assertEqual(x0.grad, x1.grad)

            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            comp_out = comp_mod(x1)

            self.assertEqual(cnts.frame_count, 1)
            comp_out[0].backward(torch.ones(4))
            self.assertEqual(x0.grad, x1.grad)

    def test_functools_arg_vary(self):
        # [FUNCTION DOC] Compiles a function using a partial hook with constant one, then
        # rebinds the enclosing partial to constant two and checks the second gradient is
        # twice the first under compiled autograd.
        def pre_hook(grad, *, k):
            # [FUNCTION DOC] HELPER: Hook multiplying the gradient by the keyword-only factor
            # k.
            return grad * k

        hook = functools.partial(pre_hook, k=1)

        @torch.compile(backend="eager", fullgraph=True)
        def h(x):
            # [FUNCTION DOC] HELPER: Compiled function registering the enclosing hook partial
            # on an intermediate and returning it tripled.
            y = x.mul(2)
            y.register_hook(hook)
            return y.mul(3)

        with compiled_autograd._enable(torch.compile(backend="eager", fullgraph=True)):
            x = torch.randn(2, requires_grad=True)
            h(x).sum().backward()
            orig_grad = x.grad
            x.grad = None

            hook = functools.partial(pre_hook, k=2)
            h(x).sum().backward()
            self.assertEqual(orig_grad * 2, x.grad)

    def test_post_acc_grad_hook(self):
        # [FUNCTION DOC] Post-accumulate-grad hook that multiplies the tensor in place by its
        # gradient and scales the gradient by five. Runs once eagerly, then compiled across
        # eager, aot_eager and inductor backends with and without compiled backward. All runs
        # must give the same tensor and gradient values, with one frame for compiled runs.
        def hook(input_t):
            # [FUNCTION DOC] HELPER: Multiplies the tensor in place by its gradient and then
            # multiplies the gradient in place by five.
            input_t.mul_(input_t.grad)
            input_t.grad.mul_(5)

        def reg_and_mul(x, y):
            # [FUNCTION DOC] HELPER: Registers the post-accumulate hook on x and returns x
            # times y.
            x.register_post_accumulate_grad_hook(hook)
            return x * y

        cnts = None

        def test_fn(fn):
            # [FUNCTION DOC] HELPER: Calls the function on module-level x and y, backward on x
            # with a given gradient, then checks the frame count (if a counter exists), the
            # tensor values and gradient. Uses enclosing x, y and cnts variables.
            fn(x, y)
            b = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
            x.backward(b)
            if cnts:
                self.assertEqual(cnts.frame_count, 1)
            # These same exact assertions run on both eager and compiled
            # X goes to x*2 because of mul_
            self.assertEqual(x, torch.tensor([0.5, 0.5, 0.5]) * 2)
            # This test proves grad aliasing works -
            self.assertEqual(x.grad, b * 5)

        # Eager values
        x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        test_fn(reg_and_mul)

        # Compiled
        for backend in ["eager", "aot_eager", "inductor"]:
            for compiled_bwd in [False, True]:
                torch._dynamo.reset()
                x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
                y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

                cnts = torch._dynamo.testing.CompileCounterWithBackend(backend)
                compiled_fn = torch.compile(reg_and_mul, backend=cnts, fullgraph=True)

                compiled_bwd_ctx = (
                    compiled_autograd._enable(
                        torch.compile(backend=backend, fullgraph=True)
                    )
                    if compiled_bwd
                    else contextlib.nullcontext()
                )
                with compiled_bwd_ctx:
                    test_fn(compiled_fn)

    def test_recompile(self):
        # [FUNCTION DOC] Registers a post-accumulate hook that doubles the gradient in place
        # on a leaf and, under compiled autograd with error on recompile enabled, runs five
        # backward passes with a changing input, clearing the gradient each time. Passes if no
        # recompile occurs.
        def hook(param):
            # [FUNCTION DOC] HELPER: Doubles the parameter's gradient in place.
            param.grad *= 2

        x = torch.ones(10)
        x.requires_grad = True

        def run(input):
            # [FUNCTION DOC] HELPER: Returns the leaf tensor times the input.
            return x * input

        x.register_post_accumulate_grad_hook(hook)
        with compiled_autograd._enable(compiler_fn):
            for i in range(5):
                with unittest.mock.patch(
                    "torch._dynamo.config.error_on_recompile", True
                ):
                    # Mimic optimizer.zero_grad() to clear the gradient
                    x.grad = None
                    run(i).sum().backward()

    def test_no_recompile_on_same_hook(self):
        # [FUNCTION DOC] Ten linear layers, each with a forward pre-hook calling the same
        # function and each compiled with one counting backend. Running the model once must
        # compile only one frame in total.
        cnts = torch._dynamo.testing.CompileCounter()

        def fw_hook(inp):
            # [FUNCTION DOC] HELPER: Forward-hook helper returning a tuple whose single
            # element is the first input plus one.
            return (inp[0] + 1,)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a list of ten compiled linear layers with
            # pre-hooks.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds ten linear layers, registers a pre-hook on
                # each, compiles each with the counting backend and appends it.
                super().__init__()
                self.layers = torch.nn.ModuleList()
                for _ in range(10):
                    layer = torch.nn.Linear(16, 16)
                    layer.register_forward_pre_hook(lambda _, inp: fw_hook(inp))
                    layer = torch.compile(layer, backend=cnts)
                    self.layers.append(layer)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Passes x through each layer in order.
                for l in self.layers:
                    x = l(x)
                return x

        mod = Mod()
        x = torch.ones(16, 16, requires_grad=True)
        mod(x)

        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(skip_nnmodule_hook_guards=False)
    def test_nnmodule_hook_guards(self):
        # [FUNCTION DOC] With module-hook guards enabled by config, compiles a function
        # calling a module, then registers a forward hook doubling the output. Compiled result
        # must equal eager and a second frame must be compiled (guard failure on hook
        # addition).
        # Compile a model and then apply a hook

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        cnts = torch._dynamo.testing.CompileCounter()

        mod = Mod()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the enclosing module on x.
            return mod(x)

        opt_fn = torch.compile(fn, backend=cnts)

        x = torch.ones(16, 16)
        opt_fn(x)

        # Register a hook
        def forward_hook(self, inputs, out):
            # [FUNCTION DOC] HELPER: Forward hook doubling the module's output.
            return out * 2

        mod.register_forward_hook(forward_hook)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @torch._dynamo.config.patch(wrap_top_frame=True)
    def test_wrap_top_frame_with_hooks(self):
        # [FUNCTION DOC] With the wrap-top-frame option on, compiles a module with a forward
        # pre-hook two ways, by compiling the module and by compiling in place. Each result
        # must equal eager and compile one frame.
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one bias-free linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.net1 = torch.nn.Linear(18, 18, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.net1(x)

        mod = ToyModel()
        mod.register_forward_pre_hook(lambda mod, input: input[0] + 1)

        # Case 1: torch.compile(mod)
        cnts = torch._dynamo.testing.CompileCounter()
        compiled_mod = torch.compile(mod, backend=cnts)

        x = torch.rand(18, 18)
        ref = mod(x)
        res = compiled_mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        # Case 2: mod.compile()
        cnts = torch._dynamo.testing.CompileCounter()
        mod.compile(backend=cnts)
        res = mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

    def test_global_module_forward_pre_hook(self):
        # [FUNCTION DOC] Registers a global forward pre-hook counting calls, runs the module
        # eagerly (count one), then runs a compiled copy expecting a UserWarning about global
        # hooks and the counter to reach three. Compiled output must equal eager. Removes the
        # hook in a final cleanup step.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward subtracts one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus one.
                return x - 1

        counter = 0

        def hook(mod, args):
            # [FUNCTION DOC] HELPER: Increments the enclosing counter and returns the
            # arguments unchanged.
            nonlocal counter
            counter += 1
            return args

        x = torch.rand(18, 18)
        mod = Mod()
        compiled_mod = torch.compile(mod, backend="eager")

        try:
            hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(hook)
            ref = mod(x)
            self.assertEqual(counter, 1)
            with self.assertWarnsRegex(
                UserWarning,
                r"Using `torch.compile\(module\)` when there are global hooks.*",
            ):
                res = compiled_mod(x)
            self.assertEqual(counter, 3)
            self.assertEqual(ref, res)
        finally:
            hook_handle.remove()

    def test_register_hook_on_intermediate_stride_dependent(self):
        # [FUNCTION DOC] Hook branching on whether the gradient is contiguous; expects
        # compiling with aot_eager and a full graph to raise Unsupported because gradient
        # properties are unknown at trace time.
        def hook(grad):
            # [FUNCTION DOC] HELPER: Returns sine of the gradient if contiguous, otherwise
            # cosine.
            if grad.is_contiguous():
                return grad.sin()
            else:
                return grad.cos()

        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, registers the stride-dependent hook, returns
            # the sum.
            y = x * 2
            y.register_hook(hook)
            return y.sum()

        # Hooks that branch on grad metadata (e.g. is_contiguous)
        # graph break because grad properties are unknown at trace time.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)


class HooksTestsDevice(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-parameterized tests of autograd-cache behavior with hooks;
    # instantiated per accelerator at import time.
    @skipIfRocm(msg="pytorch/pytorch/issues/190414")
    @onlyAccelerator
    def test_register_hook_on_intermediate_autograd_cache(self, device):
        # [FUNCTION DOC] Skipped on ROCm and only runs on accelerators. Enables the autograd
        # cache, compiles and runs a function with an intermediate hook twice (resetting
        # dynamo between), and asserts that no autograd cache bypass was counted. Always
        # disables the cache setting at the end. Mutates global config and the shared
        # counters.
        from torch._dynamo.utils import counters

        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, registers a lambda hook halving the gradient,
            # returns the sum.
            y = x * 2
            y.register_hook(lambda g: g * 0.5)
            return y.sum()

        torch._functorch.config.enable_autograd_cache = True
        try:
            # First compile
            torch._dynamo.reset()
            counters.clear()
            x = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn, fullgraph=True)(x).backward()  # noqa: UNSPECIFIED_BACKEND

            # Second compile (force recompile to test cache)
            torch._dynamo.reset()
            x2 = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn, fullgraph=True)(x2).backward()  # noqa: UNSPECIFIED_BACKEND

            aot_counters = counters["aot_autograd"]
            self.assertEqual(aot_counters.get("autograd_cache_bypass", 0), 0)
        finally:
            torch._functorch.config.enable_autograd_cache = False

    @onlyAccelerator
    def test_register_hook_on_intermediate_autograd_cache_different_hooks(self, device):
        # [FUNCTION DOC] Accelerator-only. With the autograd cache on, compiles two functions
        # that differ only in the hook (times one half versus times three). The second must
        # not reuse the first's cache entry: its gradient is six, and no cache bypass is
        # recorded. Resets the cache setting afterwards.
        from torch._dynamo.utils import counters

        def fn_a(x):
            # [FUNCTION DOC] HELPER: Doubles x, registers a hook halving the gradient, returns
            # the sum.
            y = x * 2
            y.register_hook(lambda g: g * 0.5)
            return y.sum()

        def fn_b(x):
            # [FUNCTION DOC] HELPER: Doubles x, registers a hook tripling the gradient,
            # returns the sum.
            y = x * 2
            y.register_hook(lambda g: g * 3.0)
            return y.sum()

        torch._functorch.config.enable_autograd_cache = True
        try:
            torch._dynamo.reset()
            counters.clear()

            # Compile fn_a
            x = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn_a, fullgraph=True)(x).backward()  # noqa: UNSPECIFIED_BACKEND

            # Compile fn_b (different hook — must NOT cache hit from fn_a)
            x2 = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn_b, fullgraph=True)(x2).backward()  # noqa: UNSPECIFIED_BACKEND

            # fn_b should give grad = 2 * 3.0 = 6.0, not 2 * 0.5 = 1.0
            self.assertEqual(x2.grad, torch.tensor([6.0] * 4, device=device))
            self.assertEqual(
                counters["aot_autograd"].get("autograd_cache_bypass", 0), 0
            )
        finally:
            torch._functorch.config.enable_autograd_cache = False


instantiate_device_type_tests(HooksTestsDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
