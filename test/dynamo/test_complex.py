# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that torch.compile with fullgraph handles complex-valued tensors when the
# functorch complex wrapper option is enabled: complex inputs, no complex inputs/outputs, list
# outputs, aliasing and mutation semantics (including known-failing cases) and a RoPE (rotary
# embedding) function. Applies instantiate_parametrized_tests to ComplexTests at import.
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import instantiate_parametrized_tests


class ComplexDynamoTestCase(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Base test case that turns on torch._functorch.config.enable_complex_wrapper
    # for the whole class and turns it off again afterwards.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Run the base setUpClass and set the functorch enable_complex_wrapper
        # config to True (global side effect).
        super().setUpClass()
        torch._functorch.config.enable_complex_wrapper = True

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Run the base tearDownClass and set the functorch enable_complex_wrapper
        # config back to False.
        super().tearDownClass()
        torch._functorch.config.enable_complex_wrapper = False


def sample_function(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Matrix-multiply a and b, take the absolute value and add x.
    y = torch.mm(a, b)
    return y.abs() + x


class ComplexTests(ComplexDynamoTestCase):
    # [CLASS DOC] Tests compiling complex-number functions with fullgraph under the complex
    # wrapper.
    def test_simple(self):
        # [FUNCTION DOC] Compile the sample function with complex inputs and compare against
        # eager.
        fn_c = torch.compile(sample_function, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)

        self.assertEqual(fn_c(a, b, x), sample_function(a, b, x))

    def test_no_complex_in_out(self):
        # [FUNCTION DOC] Compile a function that builds a complex tensor from real and imaginary
        # parts and returns its absolute value; compare with eager.
        def f(re, im):
            # [FUNCTION DOC] HELPER: Build a complex tensor from re and im and return its abs.
            c = torch.complex(re, im)
            return c.abs()

        re = torch.randn(2, 2, dtype=torch.float32)
        im = torch.randn(2, 2, dtype=torch.float32)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(fn_c(re, im), f(re, im))

    def test_no_complex_inputs(self):
        # [FUNCTION DOC] Compile a function that makes complex tensors from real inputs
        # internally and calls the sample function; compare with eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Build two complex tensors from sin/cos of the inputs and
            # return the sample function result.
            a = torch.complex(torch.sin(x), torch.cos(y))
            b = torch.complex(torch.cos(x), torch.sin(y))
            return sample_function(a, b, x)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        self.assertEqual(fn_c(x, y), f(x, y))

    def test_list_out(self):
        # [FUNCTION DOC] Compile a function returning a list of two sample function results and
        # compare with eager.
        def f(a, b, x):
            # [FUNCTION DOC] HELPER: Return a list of sample_function(a, b, x) and
            # sample_function(b, a, x).
            return [sample_function(a, b, x), sample_function(b, a, x)]

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)
        self.assertEqual(fn_c(a, b, x), f(a, b, x))

    def test_incorrect_aliasing_semantics_raises(self):
        # [FUNCTION DOC] Assert that compiling a function that mutates a complex tensor after
        # taking view_as_real of it raises RuntimeError (the view is a copy, so aliasing is not
        # preserved).
        # view_as_real decomposes to stack (a copy), so mutation of the
        # original complex tensor does not propagate through the view.
        def f(a):
            # [FUNCTION DOC] HELPER: Take view_as_real of a, overwrite a with zeros and return
            # the view.
            out = torch.view_as_real(a)
            a[...] = torch.zeros_like(a)
            return out

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertRaises(RuntimeError, fn_c, a)

    @unittest.expectedFailure
    def test_incorrect_aliasing_semantics(self):
        # [FUNCTION DOC] Expected failure: compiled result should equal eager for the
        # view_as_real-then-mutate function; remove once aliasing is supported.
        # Once this succeeds, remove the xfail and the test above
        def f(a):
            # [FUNCTION DOC] HELPER: Take view_as_real of a, overwrite a with zeros and return
            # the view.
            out = torch.view_as_real(a)
            a[...] = torch.zeros_like(a)
            return out

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        ref = f(a)
        test = fn_c(a)
        self.assertEqual(ref, test)

    def test_conj_alias(self):
        # [FUNCTION DOC] Compile a function that mutates a conjugate view of a complex zero
        # tensor and returns the original; compare with eager.
        def f():
            # [FUNCTION DOC] HELPER: Create a zero complex tensor, fill its conj view with ones
            # and return the tensor.
            a = torch.zeros(2, 2, dtype=torch.complex64)
            b = a.conj()
            b[...] = torch.ones_like(b)
            return a

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_neg_alias(self):
        # [FUNCTION DOC] Compile a function that mutates a negative view of a complex tensor;
        # compare with eager.
        def f():
            # [FUNCTION DOC] HELPER: Create a zero complex tensor, fill its _neg_view with ones
            # and return the tensor.
            a = torch.zeros(2, 2, dtype=torch.complex64)
            b = a._neg_view()
            b[...] = torch.ones_like(b)
            return a

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_complex_output_aliases_input(self):
        # [FUNCTION DOC] Check a compiled function returning an alias of a complex input matches
        # eager, and that after zeroing the input in place the eager and compiled outputs still
        # compare equal (both alias the input).
        # KNOWN ISSUE: the comment about removing an xfail and the test above is stale; this
        # test has no xfail.
        # Once this succeeds, remove the xfail and the test above
        def f(a):
            # [FUNCTION DOC] HELPER: Return an ellipsis-indexed view of a.
            return a[...]

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        ref = f(a)
        test = fn_c(a)
        self.assertEqual(ref, test)
        a[...] = torch.zeros_like(a)
        self.assertEqual(ref, test)

    def test_complex_input_mutation_raises(self):
        # [FUNCTION DOC] Assert that compiling a function that mutates a complex input in place
        # raises RuntimeError.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiply a in place by b and return abs of a.
            a.mul_(b)
            return a.abs()

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertRaises(RuntimeError, fn_c, a, b)

    @unittest.expectedFailure
    def test_complex_input_mutation(self):
        # [FUNCTION DOC] Expected failure: compiled in-place mutation of a complex input should
        # give the same output and mutated input as eager; remove once supported.
        # Once this succeeds, remove the xfail and the test above
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiply a in place by b and return abs of a.
            a.mul_(b)
            return a.abs()

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        a_ref = a.clone()
        a_test = a.clone()
        self.assertEqual(f(a_test, b), fn_c(a_ref, b))
        self.assertEqual(a_test, a_ref)

    def test_aliasing_semantics(self):
        # [FUNCTION DOC] Compile a function that creates a complex tensor, takes a view, then
        # zeros the base, and compare with eager.
        def mutate():
            # [FUNCTION DOC] HELPER: Create ones, take an ellipsis view, zero the base and
            # return the view.
            a = torch.ones(2, 2, dtype=torch.complex64)
            out = a[...]
            a[...] = torch.zeros_like(a)
            return out

        mutate_c = torch.compile(mutate, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(mutate(), mutate_c())

    def test_view_as_real(self):
        # [FUNCTION DOC] Compile a function returning a clone of view_as_real of a complex
        # tensor plus another complex tensor; compare with eager.
        def f():
            # [FUNCTION DOC] HELPER: Return the real view clone of a ones complex tensor and a
            # zeros complex tensor.
            a = torch.ones(2, 2, dtype=torch.complex64)
            b = torch.zeros(2, 2, dtype=torch.complex64)
            return torch.view_as_real(a).clone(), b

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_rope(self):
        # [FUNCTION DOC] Compile a RoPE (rotary embedding) function on seeded random float
        # tensors, compiled with fullgraph, and compare with eager.
        """Test the RoPE function, taken from:
        https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L80-L161"""

        def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
            # [FUNCTION DOC] HELPER: Compute rotary frequencies as complex unit-magnitude
            # numbers (polar form) for each position and half-dimension.
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            t = torch.arange(end, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Reshape freqs_cis so it broadcasts against x on dimensions
            # 1 and the last; raises AssertionError if x has fewer than two dims or the shapes
            # disagree.
            ndim = x.ndim
            assert ndim >= 2  # noqa: S101
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # noqa: S101
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        def apply_rotary_emb(
            # [FUNCTION DOC] HELPER: View the last dimension of xq and xk as complex pairs,
            # multiply by the broadcast frequencies, convert back to real, flatten from dim 3
            # and cast to the original dtypes; returns both.
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        def f(xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Compute freqs_cis from the sequence length and head size,
            # then return apply_rotary_emb for xq and xk.
            freqs_cis = precompute_freqs_cis(xq.shape[-1], xq.shape[1])
            return apply_rotary_emb(xq, xk, freqs_cis)

        g = torch.Generator().manual_seed(42)
        xq = torch.randn(4, 32, 64, generator=g)
        xk = torch.randn(4, 32, 64, generator=g)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(xq, xk), fn_c(xq, xk))


instantiate_parametrized_tests(ComplexTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
