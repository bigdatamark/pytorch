# Owner(s): ["module: autograd"]
# [MODULE DOC] Tests for the autograd fallback that runs for custom operators which have no
# autograd kernel registered. Each test defines an operator in a scratch library namespace with a
# CPU implementation, runs it under the 'nothing' and 'warn' fallback modes, and checks whether
# backward warns (warn mode) or raises about missing grad (nothing mode), and whether gradients
# flow. Import time only defines helpers and the test class; the runner is invoked from main.

import contextlib
import warnings

import numpy as np

import torch
from torch.library import _scoped_library
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@contextlib.contextmanager
def autograd_fallback_mode(mode):
    # [FUNCTION DOC] HELPER: Context manager that sets the global autograd fallback mode to the
    # given mode string, yields, and always restores the previously active mode on exit, even if
    # the body raises. Reads and writes the mode through the torch C bindings.
    prev = torch._C._get_autograd_fallback_mode()
    try:
        torch._C._set_autograd_fallback_mode(mode)
        yield
    finally:
        torch._C._set_autograd_fallback_mode(prev)


class TestAutogradFallback(TestCase):
    # [CLASS DOC] Tests of the autograd fallback behavior for operators lacking autograd kernels.
    # Per-test state is a list of Library objects (self.libraries) created through get_lib and
    # destroyed in tearDown; all operators live in the namespace named by test_ns. Most tests are
    # parametrized over the fallback modes 'nothing' and 'warn'.
    test_ns = "_test_autograd_fallback"

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setup and creates an empty list that tracks
        # libraries created during the test.
        super().setUp()
        self.libraries = []

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Removes the test operator namespace from torch.ops if present,
        # destroys every library recorded in self.libraries, and deletes that list. Does not call
        # the base tearDown.
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            lib._destroy()
        del self.libraries

    def get_op(self, name):
        # [FUNCTION DOC] HELPER: Looks up the operator with the given name in the test namespace
        # and returns its default overload.
        return getattr(getattr(torch.ops, self.test_ns), name).default

    def get_lib(self):
        # [FUNCTION DOC] HELPER: Creates a FRAGMENT library in the test namespace, records it in
        # self.libraries so tearDown destroys it, and returns it.
        result = torch.library.Library(self.test_ns, "FRAGMENT")  # noqa: SCOPED_LIBRARY
        self.libraries.append(result)
        return result

    @parametrize("mode", ("nothing", "warn"))
    def test_no_grad(self, mode):
        # [FUNCTION DOC] With grad mode disabled, or with inputs that do not require grad,
        # calling a CPU-only operator must not emit any warning (warnings are turned into errors)
        # and the output must not require grad. Runs under both fallback modes.
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a, Tensor b, int c) -> Tensor")
            lib.impl("foo", lambda a, b, c: a + b + c, "CPU")
            op = self.get_op("foo")

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with torch.no_grad():
                    a = torch.randn([], requires_grad=True)
                    b = torch.randn([], requires_grad=True)
                    out = op(a, b, 1)
                self.assertFalse(out.requires_grad)

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                a = torch.randn([])
                b = torch.randn([])
                out = op(a, b, 1)
                self.assertFalse(out.requires_grad)

    @parametrize("mode", ("nothing", "warn"))
    def test_no_autograd_kernel(self, mode):
        # [FUNCTION DOC] An operator with only a CPU kernel is called with one input requiring
        # grad. Backward through its result is checked via _check_ctx with raising enabled for
        # the 'nothing' mode: it raises about the result not requiring grad, or warns about the
        # missing autograd kernel in 'warn' mode. The input that required grad ends up with no
        # gradient.
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a, Tensor b, int c) -> Tensor")
            op = self.get_op("foo")

            def foo_impl(a, b, c):
                # [FUNCTION DOC] HELPER: CPU kernel that adds the detached numpy views of both
                # tensors and the integer, returning a new tensor with no autograd history.
                result = a.detach().numpy() + b.detach().numpy() + c
                return torch.tensor(result)

            lib.impl("foo", foo_impl, "CPU")

            # Some inputs requiring grad
            a = torch.randn([], requires_grad=False)
            b = torch.randn([], requires_grad=True)
            out = op(a, b, 1).sum()
            with self._check_ctx(mode, mode_nothing_raises=True):
                out.backward()
            self.assertIsNone(b.grad)

    def _check_ctx(self, mode, *, mode_nothing_raises=False):
        # [FUNCTION DOC] HELPER: Returns the context manager expected around backward for the
        # given mode. For 'warn' it asserts a UserWarning mentioning the missing autograd kernel.
        # For 'nothing' it asserts a RuntimeError about not requiring grad if mode_nothing_raises
        # is set, otherwise a null context. Any other mode raises an AssertionError.
        if mode == "warn":
            return self.assertWarnsRegex(
                UserWarning, "an autograd kernel was not registered"
            )
        if mode != "nothing":
            raise AssertionError(f"mode should be 'nothing', got {mode!r}")
        if mode_nothing_raises:
            return self.assertRaisesRegex(RuntimeError, "does not require grad")
        return contextlib.nullcontext()

    @parametrize("mode", ("nothing", "warn"))
    def test_no_autograd_kernel_inplace(self, mode):
        # [FUNCTION DOC] Tests an operator that mutates two inputs in place and returns them.
        # After the call, backward through each of the base tensors, the outputs and the views is
        # checked against the mode context. Then an operator with mutable input and no outputs is
        # tested: the fallback does nothing, no warning is emitted, and the gradient of the clone
        # flows back unchanged.
        with autograd_fallback_mode(mode):
            # input modified in-place gets returned as output
            lib = self.get_lib()
            lib.define("foo(Tensor(a!) self, Tensor(b!) y) -> (Tensor(a!), Tensor(b!))")
            op = self.get_op("foo")

            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: Under no-grad applies in-place sine to the first tensor
                # and cosine to the second, then returns both.
                with torch.no_grad():
                    x.sin_()
                    y.cos_()
                return x, y

            lib.impl("foo", foo_impl, "CPU")

            x = torch.randn(3, requires_grad=True)
            w = x.clone()
            v = x.clone()
            y0 = w[0]
            y1 = v[1]
            z0, z1 = op(y0, y1)
            for tensor in [w, v, z0, z1, y0, y1]:
                with self._check_ctx(mode):
                    tensor.sum().backward(retain_graph=True)

            # no outputs: we don't do anything. Maybe we should in the future.
            # This is not a common failure mode.
            lib.define("bar(Tensor(a!) self) -> ()")
            op = self.get_op("bar")

            def bar_impl(x):
                # [FUNCTION DOC] HELPER: Applies in-place sine to the input under no-grad and
                # returns nothing.
                with torch.no_grad():
                    x.sin_()

            lib.impl("bar", bar_impl, "CPU")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                x = torch.randn([], requires_grad=True)
                y = x.clone()
                op(y)
                y.backward()
                self.assertEqual(x.grad, torch.ones_like(x))

    @parametrize("mode", ("nothing", "warn"))
    def test_cpu_return_self(self, mode):
        # [FUNCTION DOC] CPU kernels that return their input as is (including one declared as
        # mutating) are used. Backward through the result is checked against the mode context
        # and, inside it, the input gradient is compared to ones. Uses a scoped library for
        # cleanup.
        with autograd_fallback_mode(mode):
            # To be clear, none of these situations are OK and will lead
            # to other problems down the line. We're testing them because
            # it is fairly common to actually do these things.
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor self) -> Tensor")
                lib.impl("foo", lambda x: x, "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                y = op(x).sum()
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, torch.ones_like(x))

                lib.define("bar(Tensor(a!) self) -> Tensor(a!)")
                lib.impl("bar", lambda x: x, "CPU")
                op = self.get_op("bar")

                x = torch.randn(3, requires_grad=True)
                y = op(x).sum()
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, torch.ones_like(x))

    @parametrize("mode", ("nothing", "warn"))
    def test_composite_registered_to_cpu(self, mode):
        # [FUNCTION DOC] A CPU kernel that internally calls differentiable ops (sine then sum) is
        # registered. Backward through its result is checked under the mode context, and the
        # input gradient is compared with the cosine of the input, showing the composite path
        # still differentiates.
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor self) -> Tensor")
                lib.impl("foo", lambda x: x.sin().sum(), "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                y = op(x)
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, x.cos())

    @parametrize("mode", ("nothing", "warn"))
    def test_autograd_function_registered_to_cpu(self, mode):
        # [FUNCTION DOC] Registers a custom autograd Function apply as the CPU kernel of an
        # operator. Backward under the mode context must yield the cosine of the input as the
        # gradient.
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor self) -> Tensor")

                class NumpySin(torch.autograd.Function):
                    # [CLASS DOC] HELPER: Autograd Function computing sine via numpy; forward
                    # saves the input and backward multiplies the incoming gradient by the cosine
                    # of the saved input.
                    @staticmethod
                    def forward(ctx, x):
                        # [FUNCTION DOC] HELPER: Saves the input for backward and returns a new
                        # tensor holding its numpy sine.
                        ctx.save_for_backward(x)
                        return torch.tensor(np.sin(x.cpu().numpy()))

                    @staticmethod
                    def backward(ctx, gx):
                        # [FUNCTION DOC] HELPER: Returns the incoming gradient times the cosine
                        # of the saved input.
                        (x,) = ctx.saved_tensors
                        return gx * x.cos()

                lib.impl("foo", NumpySin.apply, "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                y = op(x).sum()
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, x.cos())

    @parametrize("mode", ("nothing", "warn"))
    def test_inplace_autograd_function_registered_to_cpu(self, mode):
        # [FUNCTION DOC] Registers an in-place autograd Function as the CPU kernel of a mutating
        # operator and applies it to a view of a clone. Gradients of the result with respect to
        # the original tensor, and of the whole clone, are checked against expected values
        # (cosine at the touched element, ones elsewhere) under the mode context.
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor(a!) self) -> Tensor(a!)")

                class NumpySin_(torch.autograd.Function):
                    # [CLASS DOC] HELPER: In-place autograd Function computing sine on the input
                    # memory through numpy; it saves a clone of the input, marks the input dirty
                    # and returns it. Backward multiplies the gradient by the cosine of the saved
                    # clone.
                    @staticmethod
                    def forward(ctx, x):
                        # [FUNCTION DOC] HELPER: Saves a clone of the input, applies sine in
                        # place via numpy, marks the input as dirty and returns it.
                        ctx.save_for_backward(x.clone())
                        x_np = x.detach().numpy()
                        np.sin(x_np, out=x_np)
                        ctx.mark_dirty(x)
                        return x

                    @staticmethod
                    def backward(ctx, gx):
                        # [FUNCTION DOC] HELPER: Returns the incoming gradient times the cosine
                        # of the saved clone.
                        (x,) = ctx.saved_tensors
                        return gx * x.cos()

                lib.impl("foo", NumpySin_.apply, "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                z = x.clone()
                w = z[0]
                y = op(w)

                expected = torch.zeros_like(x)
                expected[0] = x[0].cos()
                with self._check_ctx(mode):
                    (gx,) = torch.autograd.grad(
                        y, x, torch.ones_like(y), retain_graph=True
                    )
                    self.assertEqual(gx, expected)

                expected = torch.ones_like(x)
                expected[0] = x[0].cos()
                with self._check_ctx(mode):
                    (gx,) = torch.autograd.grad(z, x, torch.ones_like(z))
                    self.assertEqual(gx, expected)

    @parametrize("mode", ("nothing", "warn"))
    def test_inplace_on_tensor_that_does_not_require_grad(self, mode):
        # [FUNCTION DOC] Defines three mutating operators on a tensor that does not require grad:
        # one returning the input, one returning a fresh clone, and one returning nothing. For
        # each, on a non-view and on a view, the other argument requires grad and computing the
        # gradient is expected to raise a RuntimeError about not requiring grad (history is not
        # rebased for such inputs).
        # We don't do anything special (that is, we don't rebase history).
        # See NOTE [autograd fallback and in-place operations] for why
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # Correct usage of (a!)
                lib.define("foo(Tensor(a!) self, Tensor other) -> Tensor(a!)")

                def foo_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                        # [FUNCTION DOC] HELPER: Adds the detached second tensor into a detached
                        # alias of the first, mutating the first, and returns the first tensor
                        # itself.
                    x_d.add_(y)
                    return x

                lib.impl("foo", foo_impl, "CPU")
                foo = self.get_op("foo")

                # Incorrect usage of (a!): user doesn't return tensor as-is
                lib.define("bar(Tensor(a!) self, Tensor other) -> Tensor(a!)")

                def bar_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                        # [FUNCTION DOC] HELPER: Mutates the first tensor through a detached
                        # alias but returns a clone of the alias instead of the input.
                    x_d.add_(y)
                    return x_d.clone()

                lib.impl("bar", bar_impl, "CPU")
                bar = self.get_op("bar")

                # User mutated input tensor but didn't return it.
                lib.define("baz(Tensor(a!) self, Tensor other) -> ()")

                def baz_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                        # [FUNCTION DOC] HELPER: Mutates the first tensor through a detached
                        # alias and returns nothing.
                    x_d.add_(y)

                lib.impl("baz", baz_impl, "CPU")
                baz = self.get_op("baz")

                # Test in-place on non-view
                for op in (foo, bar, baz):
                    x = torch.randn(3)
                    y = torch.randn(3, requires_grad=True)
                    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                        z = x.clone()
                        op(z, y)
                        torch.autograd.grad(z, y, torch.ones_like(z), allow_unused=True)

                # Test in-place on view
                for op in (foo, bar, baz):
                    x = torch.randn(3)
                    y = torch.randn(3, requires_grad=True)
                    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                        z = x[:]
                        op(z, y)
                        torch.autograd.grad(z, x, torch.ones_like(z), allow_unused=True)

    @parametrize("mode", ("nothing", "warn"))
    def test_post_autograd_returns_leaf(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
                # [FUNCTION DOC] The CPU kernel returns a clone of the input plus a new leaf
                # tensor that requires grad. Backward from the leaf is checked under the mode
                # context.
            lib.define("foo(Tensor a) -> (Tensor, Tensor)")
            op = self.get_op("foo")

            lib.impl(
                "foo", lambda a: (a.clone(), a.detach().clone().requires_grad_()), "CPU"
            )
            x = torch.randn(3, requires_grad=True)
            _, z = op(x)
            with self._check_ctx(mode):
                z.sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_undefined_inputs_outputs(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
                # [FUNCTION DOC] Passes None (an undefined tensor) as the first input and the
                # kernel returns an undefined first output plus a clone of the second input.
                # Backward from the defined output is checked under the mode context.
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor)")
            op = self.get_op("foo")

            def foo_impl(a, b):
                return None, b.clone()

    # [FUNCTION DOC] HELPER: Returns None and a clone of the second argument.
            lib.impl("foo", foo_impl, "CPU")

            x = torch.randn(3, requires_grad=True)
            # NB: PyTorch dispatcher treats "None" as undefined Tensor.
            _, z = op(None, x)
            with self._check_ctx(mode):
                z.sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_undefined_grads(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
                # [FUNCTION DOC] Both outputs of a two-output operator are passed through an
                # undefined-gradient function, so backward receives undefined gradients; the
                # summed backward is checked under the mode context.
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor)")
            op = self.get_op("foo")

            def foo_impl(a, b):
                return a.sin(), b.cos()

    # [FUNCTION DOC] HELPER: Returns the sine of the first argument and the cosine of the second.
            lib.impl("foo", foo_impl, "CPU")

            x = torch.randn(3, requires_grad=True)
            y = torch.randn(3)
            w, z = op(x, y)
            w = torch._C._functions.UndefinedGrad()(w)
            z = torch._C._functions.UndefinedGrad()(z)
            with self._check_ctx(mode):
                (z + w).sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_base_does_not_require_grad(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
                # [FUNCTION DOC] A mutating operator zeroes a tensor that is a view of a view
                # whose base does not require grad. Asserts the base of the final view is the
                # original tensor, then backward through the view is checked under the mode
                # context.
            lib.define("foo(Tensor(a!) x) -> Tensor(a!)")
            op = self.get_op("foo")

            def foo_impl(a):
                with torch.no_grad():
                    return a.zero_()
                        # [FUNCTION DOC] HELPER: Zeroes the input in place under no-grad and
                        # returns it.

            lib.impl("foo", foo_impl, "CPU")
            x = torch.randn(3)
            y = x[:]
            y.requires_grad_()
            w = y[:]
            self.assertTrue(w._base is x)

            # Hook should be registered on w, but not w._base
            op(w)
            with self._check_ctx(mode):
                w.sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_post_autograd_returns_mix_of_requires_grad_tensors(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
                # [FUNCTION DOC] The kernel returns three outputs: one computed under no-grad
                # from the first input, one a differentiable product, and one computed under no-
                # grad from the second. Gradients for each are requested: the no-grad outputs
                # raise in 'nothing' mode while the differentiable one does not.
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)")
            op = self.get_op("foo")

            def foo_impl(a, b):
                with torch.no_grad():
                    x = a.clone()
                        # [FUNCTION DOC] HELPER: Returns a clone of a (no-grad), the product of a
                        # and b, and a clone of b (no-grad).
                    z = b.clone()
                y = a * b
                return x, y, z

            lib.impl("foo", foo_impl, "CPU")
            a = torch.randn(3, requires_grad=True)
            b = torch.randn(3, requires_grad=True)
            x, y, z = op(a, b)

            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    x, (a, b), torch.ones_like(x), allow_unused=True, retain_graph=True
                )

            with self._check_ctx(mode, mode_nothing_raises=False):
                torch.autograd.grad(
                    y, (a, b), torch.ones_like(y), allow_unused=True, retain_graph=True
                )

            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    z, (a, b), torch.ones_like(z), allow_unused=True, retain_graph=True
                )

    @parametrize("mode", ("nothing", "warn"))
    def test_supports_tensor_lists(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
                # [FUNCTION DOC] Tests an operator taking and returning tensor lists. Gradients
                # of both outputs with respect to all three inputs are requested with
                # allow_unused and checked under the mode context with raising enabled in
                # 'nothing' mode.
            lib.define("foo(Tensor[] a) -> Tensor[]")
            op = self.get_op("foo")

            def foo_impl(a):
                x, y, z = a
                with torch.no_grad():
                    # [FUNCTION DOC] HELPER: Unpacks three tensors and, under no-grad, returns
                    # their sum and product.
                    return x + y + z, x * y * z

            lib.impl("foo", foo_impl, "CPU")
            x = torch.randn(3, requires_grad=True)
            y = torch.randn(1, requires_grad=True)
            z = torch.randn(2, 1, requires_grad=True)
            a, b = op([x, y, z])
            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    a,
                    (x, y, z),
                    torch.ones_like(a),
                    allow_unused=True,
                    retain_graph=True,
                )
            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    b,
                    (x, y, z),
                    torch.ones_like(b),
                    allow_unused=True,
                    retain_graph=True,
                )


instantiate_parametrized_tests(TestAutogradFallback)

if __name__ == "__main__":
    run_tests()
