# Owner(s): ["module: autograd"]

# [MODULE DOC] Tests for torch.autograd.functional (vjp, jvp, jacobian, hessian, vhp, hvp).
# Defines tensor constructor namespaces (plain and LoggingTensor-wrapped) and parametrize
# decorators selecting them, then one TestCase checking argument validation, strict mode,
# no_grad behavior, output structure, scalar inputs, create_graph with gradcheck, and vectorized
# versus non-vectorized agreement. Instantiates the parametrized tests at import time and runs
# them when executed directly.

import types
import unittest
import warnings

import torch
import torch.autograd.functional as autogradF
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    gradcheck,
    gradgradcheck,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TestCase,
)
from torch.testing._internal.logging_tensor import LoggingTensor


# Utilities for parametrizing the tensor constructors used in autograd tests
#
# TODO: maybe move somewhere so other tests can also use
#
# NB: Not all factory functions included. A complete(?) list can be found here:
#     https://docs.pytorch.org/docs/2.9/torch.html#creation-ops
base_ctors_dict = {
    "ones": torch.ones,
    "zeros": torch.zeros,
    "randn": torch.randn,
    "rand": torch.rand,
    "tensor": torch.tensor,
}
base_ctors = types.SimpleNamespace(**base_ctors_dict)


def wrap_with_logging_tensor(ctor):
    # [FUNCTION DOC] HELPER: Wraps a tensor factory so its result becomes a LoggingTensor.
    # The returned wrapper pops an optional requires_grad keyword (default false), calls the
    # factory with the remaining arguments, and wraps the output with that requires_grad
    # setting.
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: The wrapper built by wrap_with_logging_tensor; forwards
        # all arguments to the captured factory and returns a LoggingTensor.
        requires_grad = kwargs.pop("requires_grad", False)
        return LoggingTensor(ctor(*args, **kwargs), requires_grad=requires_grad)

    return wrapper


logging_tensor_ctors_dict = {
    k: wrap_with_logging_tensor(ctor) for (k, ctor) in base_ctors_dict.items()
}
logging_tensor_ctors = types.SimpleNamespace(**logging_tensor_ctors_dict)

base_and_logging_tensor = parametrize(
    "ctors",
    [
        subtest(base_ctors, name="base_tensor"),
        subtest(logging_tensor_ctors, name="logging_tensor"),
    ],
)

FIXME_base_and_xfail_logging_tensor = parametrize(
    "ctors",
    [
        subtest(base_ctors, name="base_tensor"),
        subtest(
            logging_tensor_ctors,
            name="logging_tensor",
            decorators=[unittest.expectedFailure],
        ),
    ],
)

# NB: This is equivalent to having both @parametrize("vectorized", [True, False]) and
#     FIXME_base_and_xfail_logging_tensor, except the non-vectorized logging_tensor case is
#     actually expected to succeed
FIXME_xfail_vectorized_logging_tensor = parametrize(
    "vectorize,ctors",
    [
        subtest((True, base_ctors), name="vectorized_base_tensor"),
        subtest((False, base_ctors), name="base_tensor"),
        subtest(
            (True, logging_tensor_ctors),
            name="vectorized_logging_tensor",
            decorators=[unittest.expectedFailure],
        ),
        subtest((False, logging_tensor_ctors), name="logging_tensor"),
    ],
)

vectorized_logging_tensor = parametrize(
    "vectorize,ctors",
    [
        subtest((True, base_ctors), name="vectorized_base_tensor"),
        subtest((False, base_ctors), name="base_tensor"),
        subtest((True, logging_tensor_ctors), name="vectorized_logging_tensor"),
        subtest((False, logging_tensor_ctors), name="logging_tensor"),
    ],
)


class TestAutogradFunctional(TestCase):
    # [CLASS DOC] Test case for torch.autograd.functional. Holds two structural assertion
    # helpers and many tests, each parametrized over plain tensors or LoggingTensor via the
    # module-level decorators (ctors argument is a namespace of factory functions such as
    # ones, zeros, randn, rand, tensor). Parametrization is instantiated after the class
    # definition.
    def _assert_same_struct(self, res, base):
        # [FUNCTION DOC] HELPER: Asserts res has the same structure as base: a single
        # tensor with equal size, or a tuple of tensors of equal length and element
        # sizes. Raises RuntimeError if base is neither.
        # base and res should be Tensors or tuple of Tensors with the same size
        if isinstance(base, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(base.size(), res.size())
        elif isinstance(base, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(base), len(res))
            for el_base, el_res in zip(base, res):
                self.assertTrue(isinstance(el_base, torch.Tensor))
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertEqual(el_base.size(), el_res.size())
        else:
            # Wrong base
            raise RuntimeError(
                "The base given to `_assert_same_struct` doesn't have"
                " the right structure."
            )

    def _assert_interleaved_struct(self, res, base1, base2):
        # [FUNCTION DOC] HELPER: Asserts res has the nested structure of a Jacobian-like
        # result between base1 and base2 (each a tensor or a tuple of tensors): each
        # result tensor's size must equal the size of the base1 element followed by the
        # size of the base2 element, with tuple nesting matching which bases are tuples.
        # Raises RuntimeError for invalid bases.
        # KNOWN ISSUE: in the tuple-tuple case the inner length check compares the outer
        # result with base2 instead of the inner result.
        # base1 and base2 can be Tensors or tuples of Tensors.
        # If they are tuples, res should be a tuple as well.
        # The indexing works as follows for base1, base2 being
        # - tuple, tuple: res[i][j][k][l] = (base1[i][k], base2[j][l])
        # - tuple, Tensor: res[i][k][l] = (base1[i][k], base2[l])
        # - Tensor, tuple: res[i][j][l] = (base1[i], base2[j][l])
        # - Tensor, Tensor: res[k][l] = (base1[k], base2[l])
        if isinstance(base1, torch.Tensor) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(res.size(), base1.size() + base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base1, torch.Tensor))
                self.assertEqual(el_res.size(), el_base1.size() + base2.size())
        elif isinstance(base1, torch.Tensor) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base2))
            for el_res, el_base2 in zip(res, base2):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base2, torch.Tensor))
                self.assertEqual(el_res.size(), base1.size() + el_base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, tuple))
                self.assertEqual(len(res), len(base2))
                for el_el_res, el_base2 in zip(el_res, base2):
                    self.assertTrue(isinstance(el_el_res, torch.Tensor))
                    self.assertTrue(isinstance(el_base2, torch.Tensor))
                    self.assertEqual(
                        el_el_res.size(), el_base1.size() + el_base2.size()
                    )
        else:
            # Wrong bases
            raise RuntimeError(
                "The bases given to `_assert_interleaved_struct` don't have"
                " the right structure."
            )

    @base_and_logging_tensor
    def test_vjp_err_check(self, ctors):
        # [FUNCTION DOC] Checks that vjp raises the documented errors: TypeError for bad
        # inputs or non-tensor outputs, RuntimeError for missing v on a non-scalar
        # output, wrong v structure or size. Then checks a valid call returns a result
        # structured like the input.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns three times the first three elements of
            # the input.
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a tuple with a tensor and a string, an
            # invalid output type.
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = ctors.rand(4)
        v = ctors.ones(3)
        with self.assertRaisesRegex(
            TypeError, "The inputs given to vjp must be either a Tensor"
        ):
            res = autogradF.vjp(foo, (inp, 2), v)

        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to vjp must"
        ):
            res = autogradF.vjp(bar, inp, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "The vector v can only be None if the user-provided function returns",
        ):
            res = autogradF.vjp(foo, inp)

        with self.assertRaisesRegex(
            RuntimeError, "The given v should contain a single Tensor."
        ):
            res = autogradF.vjp(foo, inp, (torch.ones_like(inp), torch.ones_like(inp)))

        with self.assertRaisesRegex(
            RuntimeError, "v has invalid size: should be torch.Size"
        ):
            res = autogradF.vjp(foo, inp, v[:2])

        res = autogradF.vjp(foo, inp, v)[1]
        self._assert_same_struct(res, inp)

    @base_and_logging_tensor
    def test_vjp_err_check_strict(self, ctors):
        # [FUNCTION DOC] Checks vjp strict mode: with strict true, raises RuntimeError
        # when the output does not require grad, is independent of the input, or the
        # Jacobian is independent of the input; with strict false returns zeros (or v
        # for the identity case).
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a detached copy of the input, so the
            # output has no grad.
            return a.detach()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns an output that requires grad but is not
            # connected to the input.
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.vjp(foo, inp, v, strict=True)
        res = autogradF.vjp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.vjp(bar, inp, v, strict=True)
        res = autogradF.vjp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        # The Jacobian does not depend on the input
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a clone of the input (Jacobian is the
            # identity).
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function is independent of input 0.",
        ):
            res = autogradF.vjp(foo, inp, v, create_graph=True, strict=True)
        res = autogradF.vjp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)

    @base_and_logging_tensor
    def test_vjp_no_grad(self, ctors):
        # [FUNCTION DOC] Checks vjp under torch.no_grad: results carry no grad_fn
        # without create_graph and do carry one with create_graph on inputs requiring
        # grad.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums over dim 1.
            return x.sum(dim=1)

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4)
        with torch.no_grad():
            res = autogradF.vjp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

    @base_and_logging_tensor
    def test_vjp_output(self, ctors):
        # [FUNCTION DOC] Checks vjp output structure and that results have no grad_fn,
        # for a single-tensor function, a two-input function, and a two-output function
        # with tuple v.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums over dim 1.
            return x.sum(dim=1)

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4)
        res = autogradF.vjp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def adder(x, y):
            # [FUNCTION DOC] HELPER: Returns two times x plus three times y.
            return 2 * x + 3 * y

        inputs = (ctors.rand(2), ctors.rand(2))
        v = ctors.ones(2)
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        self._assert_same_struct(vjp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vjp_val[0].grad_fn)
        self.assertIsNone(vjp_val[1].grad_fn)

        def adder(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair: two times x plus three times y,
            # and x plus y.
            return 2 * x + 3 * y, x + y

        inputs = (ctors.rand(2), ctors.rand(2))
        v = (ctors.tensor([1.0, 0.0]), ctors.tensor([1.0, 0.0]))
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        self._assert_same_struct(vjp_val, inputs)
        self.assertIsNone(out[0].grad_fn)
        self.assertIsNone(out[1].grad_fn)
        self.assertIsNone(vjp_val[0].grad_fn)
        self.assertIsNone(vjp_val[1].grad_fn)

    @base_and_logging_tensor
    def test_vjp_scalar(self, ctors):
        # [FUNCTION DOC] Checks vjp with scalar outputs (explicit and implicit v) and
        # scalar inputs.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums all elements to a scalar.
            return x.sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones([])
        res = autogradF.vjp(reducer, inputs, v)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.vjp(reducer, inputs)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

        def expander(x):
            # [FUNCTION DOC] HELPER: Adds a leading dim and repeats a scalar input
            # into a vector of four.
            return x.unsqueeze(0).repeat(4)

        inputs = ctors.rand([])
        v = ctors.ones(4)
        res = autogradF.vjp(expander, inputs, v)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

    @base_and_logging_tensor
    def test_vjp_create_graph(self, ctors):
        # [FUNCTION DOC] Checks vjp with create_graph: result has grad_fn, and gradcheck
        # and gradgradcheck pass for single and multi-input functions, including a
        # composed function that uses vjp results.
        # KNOWN ISSUE: the single-input gradcheck lambdas ignore their first argument
        # and close over the outer inputs, so the input gradient is not really
        # exercised.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums over dim 1.
            return x.sum(dim=1)

        inputs = ctors.rand(2, 2, dtype=torch.double)
        v = ctors.ones(2, dtype=torch.double)

        inputs.requires_grad_()
        v.requires_grad_()
        res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(
            lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True),
            (inputs, v),
        )
        gradgradcheck(
            lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True),
            (inputs, v),
        )

        def adder(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair: two times x plus three times y,
            # and the product of x and y.
            return 2 * x + 3 * y, x * y

        inputs = (
            ctors.rand(2, dtype=torch.double, requires_grad=True),
            ctors.rand(2, dtype=torch.double, requires_grad=True),
        )
        v = (
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
        )

        gradcheck(
            lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[
                1
            ],
            inputs + v,
        )
        gradgradcheck(
            lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[
                1
            ],
            inputs + v,
        )

        def foo(*args):
            # [FUNCTION DOC] HELPER: Computes vjp of adder at cosine-transformed
            # inputs and returns a sum of exponentials of its outputs and inputs,
            # exercising double backward.
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.vjp(adder, (x, y), v, create_graph=True)

            return (
                val[0].exp()
                + val[1].exp()
                + grad[0].exp()
                + grad[1].exp()
                + x.exp()
                + y.exp()
            )

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    @base_and_logging_tensor
    def test_jvp_err_check(self, ctors):
        # [FUNCTION DOC] Checks that jvp raises the documented errors (TypeError for bad
        # inputs or outputs, RuntimeError for missing or badly structured or sized v),
        # then that a valid call has output structure of the function result.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns three times the first three elements of
            # the input.
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a tuple with a tensor and a string, an
            # invalid output type.
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(
            TypeError, "The inputs given to jvp must be either a Tensor"
        ):
            res = autogradF.jvp(foo, (inp, 2), v)

        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to jvp must"
        ):
            res = autogradF.jvp(bar, inp, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "The vector v can only be None if the input to the user-provided function",
        ):
            res = autogradF.jvp(foo, inp)

        with self.assertRaisesRegex(
            RuntimeError, "The given v should contain a single Tensor."
        ):
            res = autogradF.jvp(foo, inp, (v, v))

        with self.assertRaisesRegex(
            RuntimeError, "v has invalid size: should be torch.Size"
        ):
            res = autogradF.jvp(foo, inp, v[:2])

        res = autogradF.jvp(foo, inp, v)[1]
        self._assert_same_struct(res, foo(inp))

    @base_and_logging_tensor
    def test_jvp_err_check_strict(self, ctors):
        # [FUNCTION DOC] Checks jvp strict mode with the same cases as the vjp strict
        # test: errors with strict true, zero results (or v for the identity case) with
        # strict false.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a detached copy of the input, so the
            # output has no grad.
            return a.detach()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns an output that requires grad but is not
            # connected to the input.
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.jvp(foo, inp, v, strict=True)
        res = autogradF.jvp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.jvp(bar, inp, v, strict=True)
        res = autogradF.jvp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.0)

        # The Jacobian does not depend on the input
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a clone of the input (Jacobian is the
            # identity).
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function is independent of input 0.",
        ):
            res = autogradF.jvp(foo, inp, v, create_graph=True, strict=True)
        res = autogradF.jvp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)

    @base_and_logging_tensor
    def test_jvp_no_grad(self, ctors):
        # [FUNCTION DOC] Checks jvp under torch.no_grad: no grad_fn without
        # create_graph, grad_fn present with create_graph on inputs requiring grad.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums over dim 1.
            return x.sum(dim=1)

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

    @base_and_logging_tensor
    def test_jvp_output(self, ctors):
        # [FUNCTION DOC] Checks jvp output structure and absence of grad_fn for single,
        # multi-input, and multi-output functions.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums over dim 1.
            return x.sum(dim=1)

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def adder(x, y):
            # [FUNCTION DOC] HELPER: Returns two times x plus three times y.
            return 2 * x + 3 * y

        inputs = (ctors.rand(2), ctors.rand(2))
        v = (ctors.ones(2), ctors.ones(2))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

        def adder(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair: two times x plus three times y,
            # and x plus y.
            return 2 * x + 3 * y, x + y

        inputs = (ctors.rand(2), ctors.rand(2))
        v = (ctors.tensor([1.0, 0.0]), ctors.tensor([1.0, 0.0]))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out[0].grad_fn)
        self.assertIsNone(out[1].grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

    @base_and_logging_tensor
    def test_jvp_scalar(self, ctors):
        # [FUNCTION DOC] Checks jvp with scalar outputs, scalar inputs, and implicit v.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums all elements to a scalar.
            return x.sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[0], ctors.zeros([]))
        self._assert_same_struct(res[1], res[0])

        def expander(x):
            # [FUNCTION DOC] HELPER: Adds a leading dim and repeats a scalar input
            # into a vector of four.
            return x.unsqueeze(0).repeat(4)

        inputs = ctors.rand([])
        v = ctors.ones([])
        res = autogradF.jvp(expander, inputs, v)
        self._assert_same_struct(res[0], ctors.zeros(4))
        self._assert_same_struct(res[1], res[0])

        res = autogradF.jvp(expander, inputs)
        self._assert_same_struct(res[0], ctors.zeros(4))
        self._assert_same_struct(res[1], res[0])

    @base_and_logging_tensor
    def test_jvp_create_graph(self, ctors):
        # [FUNCTION DOC] Checks jvp with create_graph: grad_fn present, gradcheck and
        # gradgradcheck pass for single and multi-input functions, including a composed
        # function using jvp results.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums over dim 1.
            return x.sum(dim=1)

        inputs = ctors.rand(2, 2, dtype=torch.double)
        v = ctors.ones(2, 2, dtype=torch.double)

        inputs.requires_grad_()
        v.requires_grad_()
        res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(
            lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True),
            (inputs, v),
        )
        gradgradcheck(
            lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True),
            (inputs, v),
        )

        def adder(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair: two times x plus three times y,
            # and the product of x and y.
            return 2 * x + 3 * y, x * y

        inputs = (
            ctors.rand(2, dtype=torch.double, requires_grad=True),
            ctors.rand(2, dtype=torch.double, requires_grad=True),
        )
        v = (
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
        )

        gradcheck(
            lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[
                1
            ],
            inputs + v,
        )
        gradgradcheck(
            lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[
                1
            ],
            inputs + v,
        )

        def foo(*args):
            # [FUNCTION DOC] HELPER: Computes jvp of adder at cosine-transformed
            # inputs and returns a sum of exponentials of its outputs and inputs.
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.jvp(adder, (x, y), v, create_graph=True)

            return (
                val[0].exp()
                + val[1].exp()
                + grad[0].exp()
                + grad[1].exp()
                + x.exp()
                + y.exp()
            )

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def _test_construct_standard_basis_for(self, inputs):
        # [FUNCTION DOC] HELPER: For a tuple of input tensors, builds the standard basis
        # via the private autograd functional routine and asserts each block keeps its
        # input's dtype and device and that the concatenation (on cpu, float) equals an
        # identity matrix.
        numels = tuple(tensor.numel() for tensor in inputs)
        results = autogradF._construct_standard_basis_for(inputs, numels)
        for result, inp in zip(results, inputs):
            self.assertEqual(result.dtype, inp.dtype)
            self.assertEqual(result.device, inp.device)
        results = torch.cat(
            [result.to(device="cpu", dtype=torch.float) for result in results], dim=1
        )
        expected = torch.eye(results[0].shape[0], dtype=torch.float)
        self.assertEqual(results, expected)

    @base_and_logging_tensor
    def test_construct_standard_basis_for(self, ctors):
        # [FUNCTION DOC] Runs the standard basis check on several shape combinations
        # including scalars, mixed ranks, and mixed dtypes.
        test_cases = [
            (ctors.randn(2, 3),),
            (ctors.randn(1),),
            (ctors.randn([]),),
            (ctors.randn(1), ctors.randn([]), ctors.randn([])),
            (ctors.randn(2), ctors.randn(3), ctors.randn([])),
            (ctors.randn(2), ctors.randn([]), ctors.randn(3)),
            (ctors.randn(2, 3), ctors.randn(3), ctors.randn(3, 4, 2)),
            (ctors.randn(2, dtype=torch.float64), ctors.randn(3, dtype=torch.float32)),
        ]

        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    @base_and_logging_tensor
    def test_construct_standard_basis_for_cuda(self, ctors):
        # [FUNCTION DOC] CUDA-only variant of the standard basis test with inputs split
        # across cpu and cuda devices; skipped without CUDA.
        test_cases = [
            (ctors.randn(2), ctors.randn(3, device="cuda")),
            (ctors.randn(3, device="cuda"), ctors.randn(2)),
        ]

        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    def _test_vectorize_raises_no_warnings(self, api, ctors):
        # [FUNCTION DOC] HELPER: Asserts that calling the given API (jacobian or
        # hessian) with vectorize true on a simple function emits no Python warnings
        # (vmap's experimental warning must not leak).
        # vmap is an experimental prototype. When someone calls torch.vmap,
        # it raises a python warning. This test checks that
        # autogradF.{jacobian, hessian} don't raise that experimental prototype
        # warning; it is not nice for a public-facing API to raise a warning
        # no matter how it is called.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns the sum of squares.
            return (a**2).sum()

        x = ctors.randn(3)
        with warnings.catch_warnings(record=True) as wa:
            api(foo, x, vectorize=True)
        self.assertEqual(len(wa), 0)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/153707")
    @base_and_logging_tensor
    def test_jacobian_vectorize_raises_no_warnings(self, ctors):
        # [FUNCTION DOC] Checks jacobian with vectorize does not warn; skipped under
        # dynamo.
        return self._test_vectorize_raises_no_warnings(autogradF.jacobian, ctors)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/153644")
    @base_and_logging_tensor
    def test_hessian_vectorize_raises_no_warnings(self, ctors):
        # [FUNCTION DOC] Checks hessian with vectorize does not warn; skipped under
        # dynamo.
        return self._test_vectorize_raises_no_warnings(autogradF.hessian, ctors)

    @parametrize("vectorize", [True, False])
    @base_and_logging_tensor
    def test_jacobian_err_check(self, vectorize, ctors):
        # [FUNCTION DOC] Checks jacobian argument validation errors (TypeError for bad
        # inputs or outputs) and output structure for a single and a multi-input
        # function, with and without vectorize.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns three times the first three elements of
            # the input.
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a tuple with a tensor and a string, an
            # invalid output type.
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = ctors.rand(4)
        with self.assertRaisesRegex(
            TypeError, "The inputs given to jacobian must be either a Tensor"
        ):
            res = autogradF.jacobian(foo, (inp, 2), vectorize=vectorize)

        with self.assertRaisesRegex(
            TypeError,
            "The outputs of the user-provided function given to jacobian must",
        ):
            res = autogradF.jacobian(bar, inp, vectorize=vectorize)

        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(inp), inp)

        def foo(a, b):
            # [FUNCTION DOC] HELPER: Returns the second input and three times a
            # slice of the first.
            return b, 3 * a.narrow(0, 0, 3)

        inp = (ctors.rand(4), ctors.rand(5))

        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(*inp), inp)

    @base_and_logging_tensor
    def test_jacobian_err_check_strict(self, ctors):
        # [FUNCTION DOC] Checks jacobian strict mode: errors with strict true for
        # outputs without grad, independent outputs, or input-independent Jacobian; zero
        # (or identity) results with strict false.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a detached copy of the input.
            return a.detach()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns an output that requires grad but is not
            # connected to the input.
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = ctors.rand(4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.jacobian(foo, inp, strict=True)
        res = autogradF.jacobian(foo, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function is independent of input 0.",
        ):
            res = autogradF.jacobian(bar, inp, strict=True)
        res = autogradF.jacobian(bar, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.0)

        # The Jacobian does not depend on the input
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a clone of the input (Jacobian is the
            # identity).
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function is independent of input 0.",
        ):
            res = autogradF.jacobian(foo, inp, create_graph=True, strict=True)
        res = autogradF.jacobian(foo, inp, create_graph=True, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res, torch.eye(4))

    @base_and_logging_tensor
    def test_jacobian_err_check_strict_vectorize(self, ctors):
        # [FUNCTION DOC] Checks that strict mode together with vectorize raises
        # RuntimeError saying they are not supported together.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input unchanged.
            return x

        inp = ctors.rand(4)
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            autogradF.jacobian(foo, inp, strict=True, vectorize=True)

    @base_and_logging_tensor
    def test_jacobian_no_grad(self, ctors):
        # [FUNCTION DOC] Checks jacobian under torch.no_grad: no grad_fn by default,
        # grad_fn present with create_graph.
        def exp_reducer(x):
            # [FUNCTION DOC] HELPER: Sums the exponential over dim 1.
            return x.exp().sum(dim=1)

        inputs = ctors.rand(4, 4)
        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs)
        self.assertIsNone(res.grad_fn)
        self.assertNotEqual(res, ctors.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs, create_graph=True)
        self.assertIsNotNone(res.grad_fn)
        self.assertNotEqual(res, ctors.zeros(4, 4))

    @vectorized_logging_tensor
    def test_jacobian_output(self, vectorize, ctors):
        # [FUNCTION DOC] Checks jacobian output structure, absence of grad_fn, and
        # identity result for a clone function, for single and multi-input functions,
        # vectorized or not.
        def exp_reducer(x):
            # [FUNCTION DOC] HELPER: Sums the exponential over dim 1.
            return x.exp().sum(dim=1)

        inputs = ctors.rand(4, 4)
        res = autogradF.jacobian(exp_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        self.assertIsNone(res.grad_fn)

        def identity(x):
            # [FUNCTION DOC] HELPER: Returns a clone of the input.
            return x.clone()

        inputs = ctors.rand(4)
        res = autogradF.jacobian(identity, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, identity(inputs), inputs)
        self.assertIsNone(res.grad_fn)
        self.assertEqual(res, torch.eye(4))

        def add_exp_reducer(x, y):
            # [FUNCTION DOC] HELPER: Sums over dim 1 of x plus the exponential of y.
            return (x + y.exp()).sum(dim=1)

        inputs = (ctors.rand(4, 4), ctors.rand(4, 4))
        res = autogradF.jacobian(add_exp_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

    @vectorized_logging_tensor
    def test_jacobian_scalar(self, vectorize, ctors):
        # [FUNCTION DOC] Checks jacobian result shapes for a scalar-valued function and
        # for a scalar input.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums all elements to a scalar.
            return x.sum()

        inputs = ctors.rand(4, 4)
        res = autogradF.jacobian(reducer, inputs, vectorize=vectorize)
        self._assert_same_struct(res, inputs)

        def expander(x):
            # [FUNCTION DOC] HELPER: Adds a leading dim and repeats a scalar input
            # into a vector of four.
            return x.unsqueeze(0).repeat(4)

        inputs = ctors.rand([])
        res = autogradF.jacobian(expander, inputs, vectorize=vectorize)
        self._assert_same_struct(res, ctors.zeros(4))

    @parametrize("vectorize", [True, False])
    @base_and_logging_tensor
    def test_jacobian_create_graph(self, vectorize, ctors):
        # [FUNCTION DOC] Checks jacobian with create_graph: grad_fn present, gradcheck
        # and gradgradcheck pass for single and multi-input functions, including a
        # composed function.
        def exp_reducer(x):
            # [FUNCTION DOC] HELPER: Sums the exponential over dim 1.
            return x.exp().sum(dim=1)

        inputs = ctors.rand(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.jacobian(
            exp_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        self.assertIsNotNone(res.grad_fn)

        gradcheck(
            lambda inp: autogradF.jacobian(
                exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )
        gradgradcheck(
            lambda inp: autogradF.jacobian(
                exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )

        def add_exp_reducer(x, y):
            # [FUNCTION DOC] HELPER: Sums the exponential of x plus y over dim 1.
            return (x + y).exp().sum(dim=1)

        inputs = (
            ctors.rand(4, 4, dtype=torch.double, requires_grad=True),
            ctors.rand(4, 4, dtype=torch.double, requires_grad=True),
        )
        res = autogradF.jacobian(
            add_exp_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(
            lambda *inp: autogradF.jacobian(
                add_exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )
        gradgradcheck(
            lambda *inp: autogradF.jacobian(
                add_exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Computes the jacobian of add_exp_reducer at
            # cosine-transformed inputs and combines exponentials of the values and
            # jacobians.
            x = x.cos()
            val, jac = autogradF.jacobian(
                add_exp_reducer, (x, y), create_graph=True, vectorize=vectorize
            )

            res = val[0].exp().sum() + val[1].exp().sum() + jac[0].exp().sum()
            res = res + jac[1].exp().sum() + x.exp().sum() + y.exp().sum()
            return res

        gradcheck(foo, inputs)
        gradgradcheck(foo, inputs)

    def _check_jacobian_vectorize_correctness(self, f, inputs, test_forward_ad=True):
        # [FUNCTION DOC] HELPER: Asserts the vectorized backward-mode jacobian equals
        # the non-vectorized one and, when test_forward_ad is true, that the forward-
        # mode vectorized jacobian also equals it.
        expected = autogradF.jacobian(f, inputs, vectorize=False)
        result_backward_mode = autogradF.jacobian(f, inputs, vectorize=True)
        self.assertEqual(result_backward_mode, expected)

        if test_forward_ad:
            result_forward_mode = autogradF.jacobian(
                f, inputs, strategy="forward-mode", vectorize=True
            )
            self.assertEqual(result_forward_mode, expected)

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_simple(self, ctors):
        # [FUNCTION DOC] Checks vectorized jacobian correctness for a simple single-
        # input function.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns three times the square of the input.
            return 3 * x**2

        x = ctors.randn(2, 3, 5)
        self._check_jacobian_vectorize_correctness(f, x)

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_multi_input(self, ctors):
        # [FUNCTION DOC] Checks vectorized jacobian correctness for a two-input
        # function.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a matrix product of cosine-scaled x and
            # sine of y.
            return (x.cos() * x) @ y.sin()

        x = ctors.randn(2, 3)
        y = ctors.randn(3, 5)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_multi_input_multi_output(self, ctors):
        # [FUNCTION DOC] Checks vectorized jacobian correctness for a multi-input,
        # multi-output function.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns three outputs mixing matrix products
            # and a sum.
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        x = ctors.randn(5, 3)
        y = ctors.randn(3, 5)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_unrelated_outputs(self, ctors):
        # [FUNCTION DOC] Checks vectorized jacobian correctness when outputs are the
        # inputs repeated (unrelated outputs).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the inputs twice each.
            return x, y, x, y

        x = ctors.randn(2)
        y = ctors.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_zero_dim(self, ctors):
        # [FUNCTION DOC] Checks vectorized jacobian correctness for zero-dim outputs,
        # zero-dim inputs, and mixed cases.
        # zero-dim output
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns two scalar sums and an elementwise
            # product.
            return x.sum(), y.sum(), x * y

        x = ctors.randn(3)
        y = ctors.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

        # zero-dim input
        def g(x):
            # [FUNCTION DOC] HELPER: Stacks three copies of a scalar input.
            return torch.stack([x, x, x])

        x = ctors.randn([])
        self._check_jacobian_vectorize_correctness(g, x)

        # Mixed zero-dim input / zero-dim output
        def h(x, y):
            # [FUNCTION DOC] HELPER: Returns a sum of y and the elementwise product
            # of x and y.
            return y.sum(), x * y

        x = ctors.randn([])
        y = ctors.randn(1)
        self._check_jacobian_vectorize_correctness(h, (x, y))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_different_devices(self, ctors):
        # [FUNCTION DOC] CUDA-only: checks vectorized jacobian correctness when outputs
        # live on different devices; skipped without CUDA.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the product on cpu and the same product
            # moved to cuda.
            return x * y, (x * y).cuda()

        x = ctors.randn(3)
        y = ctors.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_different_dtype(self, ctors):
        # [FUNCTION DOC] Checks vectorized jacobian correctness for outputs of different
        # dtypes, skipping the forward-mode comparison because forward AD returns output
        # dtype.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the product cast to float and to
            # double.
            return (x * y).float(), (x * y).double()

        x = ctors.randn(3)
        y = ctors.randn(3)
        # The Jacobian computed using forward AD has the dtype of the output
        # but the Jacobian computed with reverse AD has dtype of input
        self._check_jacobian_vectorize_correctness(f, (x, y), test_forward_ad=False)

    def _check_hessian_vectorize_correctness(self, f, inputs):
        # [FUNCTION DOC] HELPER: Asserts the vectorized hessian equals the non-
        # vectorized one, and that the forward-mode outer-jacobian vectorized hessian
        # also equals it.
        expected = autogradF.hessian(f, inputs, vectorize=False)
        result = autogradF.hessian(f, inputs, vectorize=True)
        self.assertEqual(result, expected)

        result_forward_mode = autogradF.hessian(
            f, inputs, outer_jacobian_strategy="forward-mode", vectorize=True
        )
        self.assertEqual(result_forward_mode, expected)

    @base_and_logging_tensor
    def test_hessian_vectorize_correctness_simple(self, ctors):
        # [FUNCTION DOC] Checks vectorized hessian correctness for a simple function.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of three times the square of
            # the input.
            return (3 * x**2).sum()

        x = ctors.randn(2, 3, 5)
        self._check_hessian_vectorize_correctness(f, x)

    @base_and_logging_tensor
    def test_hessian_vectorize_correctness_multi_input(self, ctors):
        # [FUNCTION DOC] Checks vectorized hessian correctness for a three-input
        # function.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the sum of a matrix product chain using
            # relu and sine.
            return ((x.relu() * x) @ y.sin() @ z).sum()

        x = ctors.randn(2, 3)
        y = ctors.randn(3, 5)
        z = ctors.randn(5, 5)
        self._check_hessian_vectorize_correctness(f, (x, y, z))

    @base_and_logging_tensor
    def test_hessian_vectorize_correctness_unrelated_outputs(self, ctors):
        # [FUNCTION DOC] Checks vectorized hessian correctness when the output is
        # unrelated to one input and when it is unrelated to all inputs.
        # output unrelated to one input
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of squares of x only.
            return (x**2).sum()

        x = ctors.randn(2)
        y = ctors.randn(3)
        self._check_hessian_vectorize_correctness(f, (x, y))

        # output unrelated to all inputs
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a constant scalar one, independent of
            # all inputs.
            return ctors.ones([])

        x = ctors.randn(2)
        y = ctors.randn(3)
        self._check_hessian_vectorize_correctness(f, (x, y))

    @parametrize("vectorize", [True, False])
    @base_and_logging_tensor
    def test_hessian_err_check(self, vectorize, ctors):
        # [FUNCTION DOC] Checks hessian argument validation errors (bad inputs, bad
        # output types, non-single-element output, tuple output) and the output
        # structure for single and multi-input functions.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a tuple with a tensor and a string, an
            # invalid output type.
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            # [FUNCTION DOC] HELPER: Returns a non-scalar tensor.
            return 3 * a.narrow(0, 0, 3)

        def bar3(a):
            # [FUNCTION DOC] HELPER: Returns a tuple of two tensors.
            return 3 * a.narrow(0, 0, 3), 3 * a.narrow(0, 0, 3)

        inp = ctors.rand(4)
        with self.assertRaisesRegex(
            TypeError, "The inputs given to hessian must be either a Tensor"
        ):
            res = autogradF.hessian(foo, (inp, 2), vectorize=vectorize)

        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to hessian must"
        ):
            res = autogradF.hessian(bar, inp, vectorize=vectorize)

        err_msg_out = "The Tensor returned by the function given to hessian should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.hessian(bar2, inp, vectorize=vectorize)

        with self.assertRaisesRegex(
            RuntimeError, "The function given to hessian should return a single Tensor"
        ):
            res = autogradF.hessian(bar3, inp, vectorize=vectorize)

        res = autogradF.hessian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, inp, inp)

        def foo(a, b):
            # [FUNCTION DOC] HELPER: Returns a scalar from slices of both inputs.
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (ctors.rand(4), ctors.rand(5))

        res = autogradF.hessian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, inp, inp)

    @base_and_logging_tensor
    def test_hessian_err_check_strict(self, ctors):
        # [FUNCTION DOC] Checks hessian strict mode errors and zero results for
        # functions with no grad, unconnected output, and linear function.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns the sum of a detached copy of the
            # input.
            return a.detach().sum()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a summed output that requires grad but
            # is unconnected to the input.
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # [FUNCTION DOC] HELPER: Returns a linear function whose Jacobian is
            # independent of the input.
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = ctors.rand(4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.hessian(foo, inp, strict=True)
        res = autogradF.hessian(foo, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0",
        ):
            res = autogradF.hessian(bar, inp, strict=True)
        res = autogradF.hessian(bar, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0 is",
        ):
            res = autogradF.hessian(bar2, inp, strict=True)
        res = autogradF.hessian(bar2, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.0)

    @base_and_logging_tensor
    def test_hessian_err_check_strict_vectorize(self, ctors):
        # [FUNCTION DOC] Checks that strict together with vectorize raises RuntimeError
        # for hessian.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the sum of cubes.
            return (x**3).sum()

        inp = ctors.rand(4)
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            autogradF.hessian(foo, inp, strict=True, vectorize=True)

    @base_and_logging_tensor
    def test_hessian_no_grad(self, ctors):
        # [FUNCTION DOC] Checks hessian under torch.no_grad: no grad_fn by default,
        # grad_fn present with create_graph.
        def pow_reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum of cubes.
            return x.pow(3).sum()

        inputs = ctors.rand(2, 2)
        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs)
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)
        self.assertNotEqual(res, ctors.zeros(2, 2, 2))

        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs, create_graph=True)
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)
        self.assertNotEqual(res, ctors.zeros(2, 2, 2))

    @vectorized_logging_tensor
    def test_hessian_output(self, vectorize, ctors):
        # [FUNCTION DOC] Checks hessian output structure and absence of grad_fn for
        # single and multi-input functions, vectorized or not.
        def pow_reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum of cubes.
            return x.pow(3).sum()

        inputs = ctors.rand(2, 2)
        res = autogradF.hessian(pow_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNone(res.grad_fn)

        def add_pow_reducer(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of cubes of x plus y.
            return (x + y).pow(3).sum()

        inputs = (ctors.rand(2, 2), ctors.rand(2, 2))
        res = autogradF.hessian(add_pow_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)

    @parametrize("vectorize", [True, False])
    @base_and_logging_tensor
    def test_hessian_scalar(self, vectorize, ctors):
        # [FUNCTION DOC] Checks hessian with scalar inputs and with a function returning
        # a single-element tensor of shape one by one by one.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums all elements to a scalar.
            return x.sum()

        inputs = ctors.rand(4, 4)
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)

        inputs = ctors.rand([])
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        self._assert_same_struct(res, inputs)

        def bad_reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum reshaped to a single-element
            # tensor with three dims.
            return x.sum().view(1, 1, 1)

        inputs = ctors.rand(4, 4)
        res = autogradF.hessian(bad_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)

    @parametrize("vectorize", [True, False])
    @base_and_logging_tensor
    def test_hessian_create_graph(self, vectorize, ctors):
        # [FUNCTION DOC] Checks hessian with create_graph: grad_fn present and gradcheck
        # and gradgradcheck pass for single and multi-input functions, including a
        # composed function.
        def pow_reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum of cubes.
            return x.pow(3).sum()

        inputs = ctors.rand(2, 2, dtype=torch.double, requires_grad=True)
        res = autogradF.hessian(
            pow_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNotNone(res.grad_fn)

        gradcheck(
            lambda inp: autogradF.hessian(
                pow_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )
        gradgradcheck(
            lambda inp: autogradF.hessian(
                pow_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )

        def add_pow_reducer(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of cubes of x plus y.
            return (x + y).pow(3).sum()

        inputs = (
            ctors.rand(2, 2, dtype=torch.double, requires_grad=True),
            ctors.rand(2, 2, dtype=torch.double, requires_grad=True),
        )
        res = autogradF.hessian(
            add_pow_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)

        def flatten(inp):
            # [FUNCTION DOC] HELPER: Flattens a nested tuple of tuples into one
            # tuple.
            return tuple(el_lvl2 for el_lvl1 in inp for el_lvl2 in el_lvl1)

        gradcheck(
            lambda *inp: flatten(
                autogradF.hessian(
                    add_pow_reducer, inp, create_graph=True, vectorize=vectorize
                )
            ),
            inputs,
        )
        gradgradcheck(
            lambda *inp: flatten(
                autogradF.hessian(
                    add_pow_reducer, inp, create_graph=True, vectorize=vectorize
                )
            ),
            inputs,
        )

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Computes hessian of add_pow_reducer at cosine-
            # transformed inputs and combines cosines of the values and hessians.
            x = x.cos()
            val, hess = autogradF.hessian(
                add_pow_reducer, (x, y), create_graph=True, vectorize=vectorize
            )

            res = val[0].cos().sum() + val[1].cos().sum() + hess[0].cos().sum()
            res = res + hess[1].cos().sum() + x.cos().sum() + y.cos().sum()
            return res

        gradcheck(foo, inputs)
        gradgradcheck(foo, inputs)

    @base_and_logging_tensor
    def test_vhp_err_check(self, ctors):
        # [FUNCTION DOC] Checks vhp argument validation errors (inputs, outputs, non-
        # single-element output, v size and type) and valid output structure for single
        # and multi-input functions.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a tuple with a tensor and a string, an
            # invalid output type.
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            # [FUNCTION DOC] HELPER: Returns a non-scalar tensor.
            return 3 * a.narrow(0, 0, 3)

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(
            TypeError, "The inputs given to vhp must be either a Tensor"
        ):
            res = autogradF.vhp(foo, (inp, 2), v)

        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to vhp must"
        ):
            res = autogradF.vhp(bar, inp, v)

        err_msg_out = "The Tensor returned by the function given to vhp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.vhp(bar2, inp, v)

        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.vhp(foo, inp, ctors.rand(5))

        with self.assertRaisesRegex(
            TypeError,
            "The v given to vhp must be either a Tensor or a tuple of Tensors",
        ):
            res = autogradF.vhp(foo, inp, (v, 2))

        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

        def foo(a, b):
            # [FUNCTION DOC] HELPER: Returns a scalar from slices of both inputs.
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (ctors.rand(4), ctors.rand(5))
        v = (ctors.rand(4), ctors.rand(5))

        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    @base_and_logging_tensor
    def test_vhp_err_check_strict(self, ctors):
        # [FUNCTION DOC] Checks vhp strict mode errors and zero results for no-grad,
        # unconnected, and linear functions.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns the sum of a detached copy of the
            # input.
            return a.detach().sum()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a summed output that requires grad but
            # is unconnected to the input.
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # [FUNCTION DOC] HELPER: Returns a linear function whose Jacobian is
            # independent of the input.
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.vhp(foo, inp, v, strict=True)
        res = autogradF.vhp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.vhp(bar, inp, v, strict=True)
        res = autogradF.vhp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0 is",
        ):
            res = autogradF.vhp(bar2, inp, v, strict=True)
        res = autogradF.vhp(bar2, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

    @base_and_logging_tensor
    def test_vhp_no_grad(self, ctors):
        # [FUNCTION DOC] Checks vhp under torch.no_grad: no grad_fn by default, grad_fn
        # present with create_graph.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum of exponentials.
            return x.exp().sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        with torch.no_grad():
            res = autogradF.vhp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.vhp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

    @base_and_logging_tensor
    def test_vhp_output(self, ctors):
        # [FUNCTION DOC] Checks vhp output structure and absence of grad_fn for single
        # and two-input functions.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of exponentials
            # of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.vhp(foo, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def bar(a, b):
            # [FUNCTION DOC] HELPER: Returns a scalar sum of exponentials of a
            # combination of both inputs.
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (ctors.rand(3), ctors.rand(4))
        v = (ctors.ones(3), ctors.ones(4))
        out, vhp_val = autogradF.vhp(bar, inputs, v)
        self._assert_same_struct(vhp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vhp_val[0].grad_fn)
        self.assertIsNone(vhp_val[1].grad_fn)

    @base_and_logging_tensor
    def test_vhp_scalar(self, ctors):
        # [FUNCTION DOC] Checks vhp with scalar inputs, implicit v, and a function
        # returning a one by one by one tensor.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Sums all elements to a scalar.
            return x.sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.vhp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        inputs = ctors.rand([])
        v = ctors.rand([])
        res = autogradF.vhp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.vhp(reducer, inputs)
        self._assert_same_struct(res[1], inputs)

        def bad_reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum reshaped to a single-element
            # tensor with three dims.
            return x.sum().view(1, 1, 1)

        inputs = ctors.rand(4, 4)
        v = ctors.rand(4, 4)
        res = autogradF.vhp(bad_reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

    @base_and_logging_tensor
    def test_vhp_create_graph(self, ctors):
        # [FUNCTION DOC] Checks vhp with create_graph: grad_fn present, gradcheck and
        # gradgradcheck pass, including a composed function using vhp results.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of exponentials
            # of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = ctors.rand(4, 4, dtype=torch.double, requires_grad=True)
        v = ctors.ones(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.vhp(foo, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(
            lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v)
        )
        gradgradcheck(
            lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v)
        )

        def bar(a, b):
            # [FUNCTION DOC] HELPER: Returns a scalar sum of exponentials of a
            # combination of both inputs.
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (
            ctors.rand(3, dtype=torch.double, requires_grad=True),
            ctors.rand(4, dtype=torch.double, requires_grad=True),
        )
        v = (
            ctors.ones(3, dtype=torch.double, requires_grad=True),
            ctors.ones(4, dtype=torch.double, requires_grad=True),
        )
        out, vhp_val = autogradF.vhp(bar, inputs, v, create_graph=True)
        self._assert_same_struct(vhp_val, inputs)
        self.assertIsNotNone(out.grad_fn)
        self.assertIsNotNone(vhp_val[0].grad_fn)
        self.assertIsNotNone(vhp_val[1].grad_fn)

        gradcheck(
            lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )
        gradgradcheck(
            lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )

        def foo(*args):
            # [FUNCTION DOC] HELPER: Computes vhp of bar at cosine-transformed
            # inputs and combines cosines of the results.
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.vhp(bar, (x, y), v, create_graph=True)

            return (
                val.cos()
                + grad[0].cos().sum()
                + grad[1].cos()
                + x.cos().sum()
                + y.cos()
            )

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    @base_and_logging_tensor
    def test_hvp_err_check(self, ctors):
        # [FUNCTION DOC] Checks hvp argument validation errors and valid output
        # structure, mirroring the vhp error test.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a tuple with a tensor and a string, an
            # invalid output type.
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            # [FUNCTION DOC] HELPER: Returns a non-scalar tensor.
            return 3 * a.narrow(0, 0, 3)

        inp = ctors.rand(4)
        v = ctors.rand(4)
        res = autogradF.hvp(foo, inp, v)
        with self.assertRaisesRegex(
            TypeError, "The inputs given to hvp must be either a Tensor"
        ):
            res = autogradF.hvp(foo, (inp, 2), v)

        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to hvp must"
        ):
            res = autogradF.hvp(bar, inp, v)

        err_msg_out = "The Tensor returned by the function given to hvp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.hvp(bar2, inp, v)

        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.hvp(foo, inp, ctors.rand(5))

        with self.assertRaisesRegex(
            TypeError,
            "The v given to hvp must be either a Tensor or a tuple of Tensors",
        ):
            res = autogradF.hvp(foo, inp, (v, 2))

        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

        def foo(a, b):
            # [FUNCTION DOC] HELPER: Returns a scalar from slices of both inputs.
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (ctors.rand(4), ctors.rand(5))
        v = (ctors.rand(4), ctors.rand(5))

        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    @base_and_logging_tensor
    def test_hvp_err_check_strict(self, ctors):
        # [FUNCTION DOC] Checks hvp strict mode errors and zero results for no-grad,
        # unconnected, and linear functions.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns the sum of a detached copy of the
            # input.
            return a.detach().sum()

        def bar(a):
            # [FUNCTION DOC] HELPER: Returns a summed output that requires grad but
            # is unconnected to the input.
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # [FUNCTION DOC] HELPER: Returns a linear function whose Jacobian is
            # independent of the input.
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.hvp(foo, inp, v, strict=True)
        res = autogradF.hvp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.hvp(bar, inp, v, strict=True)
        res = autogradF.hvp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0 is",
        ):
            res = autogradF.hvp(bar2, inp, v, strict=True)
        res = autogradF.hvp(bar2, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

    @base_and_logging_tensor
    def test_hvp_no_grad(self, ctors):
        # [FUNCTION DOC] Checks hvp under torch.no_grad: no grad_fn by default, grad_fn
        # present with create_graph.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum of exponentials.
            return x.exp().sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        with torch.no_grad():
            res = autogradF.hvp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.hvp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

    @base_and_logging_tensor
    def test_hvp_output(self, ctors):
        # [FUNCTION DOC] Checks hvp output structure and absence of grad_fn for single
        # and two-input functions.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of exponentials
            # of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.hvp(foo, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def bar(a, b):
            # [FUNCTION DOC] HELPER: Returns a scalar sum of exponentials of a
            # combination of both inputs.
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (ctors.rand(3), ctors.rand(4))
        v = (ctors.ones(3), ctors.ones(4))
        out, hvp_val = autogradF.hvp(bar, inputs, v)
        self._assert_same_struct(hvp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(hvp_val[0].grad_fn)
        self.assertIsNone(hvp_val[1].grad_fn)

    @base_and_logging_tensor
    def test_hvp_scalar(self, ctors):
        # [FUNCTION DOC] Checks hvp with scalar inputs, implicit v, and a function
        # returning a one by one by one tensor.
        def reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum of exponentials.
            return x.exp().sum()

        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.hvp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        inputs = ctors.rand([])
        v = ctors.rand([])
        res = autogradF.hvp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.hvp(reducer, inputs)
        self._assert_same_struct(res[1], inputs)

        def bad_reducer(x):
            # [FUNCTION DOC] HELPER: Returns the sum of exponentials reshaped to a
            # single-element tensor with three dims.
            return x.exp().sum().view(1, 1, 1)

        inputs = ctors.rand(4, 4)
        v = ctors.rand(4, 4)
        res = autogradF.hvp(bad_reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

    @base_and_logging_tensor
    def test_hvp_create_graph(self, ctors):
        # [FUNCTION DOC] Checks hvp with create_graph: grad_fn present, gradcheck and
        # gradgradcheck pass, including a composed function using hvp results.
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of exponentials
            # of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = ctors.rand(4, 4, dtype=torch.double, requires_grad=True)
        v = ctors.ones(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.hvp(foo, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(
            lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v)
        )
        gradgradcheck(
            lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v)
        )

        def bar(a, b):
            # [FUNCTION DOC] HELPER: Returns a scalar sum of exponentials of a
            # combination of both inputs.
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (
            ctors.rand(3, dtype=torch.double, requires_grad=True),
            ctors.rand(4, dtype=torch.double, requires_grad=True),
        )
        v = (
            ctors.ones(3, dtype=torch.double, requires_grad=True),
            ctors.ones(4, dtype=torch.double, requires_grad=True),
        )
        out, hvp_val = autogradF.hvp(bar, inputs, v, create_graph=True)
        self._assert_same_struct(hvp_val, inputs)
        self.assertIsNotNone(out.grad_fn)
        self.assertIsNotNone(hvp_val[0].grad_fn)
        self.assertIsNotNone(hvp_val[1].grad_fn)

        gradcheck(
            lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )
        gradgradcheck(
            lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )

        def foo(*args):
            # [FUNCTION DOC] HELPER: Computes hvp of bar at cosine-transformed
            # inputs and combines cosines of the results.
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.hvp(bar, (x, y), v, create_graph=True)

            return (
                val.cos()
                + grad[0].cos().sum()
                + grad[1].cos()
                + x.cos().sum()
                + y.cos()
            )

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    @base_and_logging_tensor
    def test_jacobian_match_vjp_jvp(self, ctors):
        # [FUNCTION DOC] Checks that the jacobian of a function agrees with its jvp
        # (jacobian times v) and vjp (v times jacobian).
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the cube of the input plus its sum.
            return x**3 + x.sum()

        inputs = ctors.rand(4)
        v = ctors.rand(4)

        jac = autogradF.jacobian(foo, inputs)
        jvp = autogradF.jvp(foo, inputs, v)[1]
        vjp = autogradF.vjp(foo, inputs, v)[1]

        self.assertEqual(jvp, torch.mm(jac, v.unsqueeze(1)).squeeze(1))
        self.assertEqual(vjp, torch.mm(v.unsqueeze(0), jac).squeeze(0))

    @base_and_logging_tensor
    def test_hessian_match_vhp_hvp(self, ctors):
        # [FUNCTION DOC] Checks that the hessian agrees with hvp (hessian times v) and
        # vhp (v times hessian).
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns a scalar from a slice of exponentials
            # of the input.
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = ctors.rand(4)
        v = ctors.rand(4)

        hes = autogradF.hessian(foo, inputs)
        hvp = autogradF.hvp(foo, inputs, v)[1]
        vhp = autogradF.vhp(foo, inputs, v)[1]

        self.assertEqual(hvp, torch.mm(hes, v.unsqueeze(1)).squeeze(1))
        self.assertEqual(vhp, torch.mm(v.unsqueeze(0), hes).squeeze(0))


instantiate_parametrized_tests(TestAutogradFunctional)

if __name__ == "__main__":
    run_tests()
