# mypy: allow-untyped-defs
# Owner(s): ["module: complex"]
# [MODULE DOC] Tests for complex tensor behavior: list conversion, default-dtype inference for
# complex literals, conjugate copy-in-place, all/any reductions, and equality/inequality
# comparisons (including NaN parts and out arguments). Device-generic tests are instantiated at
# import time. The module level devices tuple (cpu and cuda:0) is defined but not used by the
# tests.

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_dtype import complex_types
from torch.testing._internal.common_utils import run_tests, set_default_dtype, TestCase


devices = (torch.device("cpu"), torch.device("cuda:0"))


class TestComplexTensor(TestCase):
    # [CLASS DOC] Device-generic tests over the complex dtypes; the eq and ne tests run only on
    # CPU.
    @dtypes(*complex_types())
    def test_to_list(self, device, dtype):
        # [FUNCTION DOC] Checks a zero complex tensor of shape 2x2 converts to a nested list of
        # zero complex numbers with no garbage values, for each complex dtype.
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(
            torch.zeros((2, 2), device=device, dtype=dtype).tolist(),
            [[0j, 0j], [0j, 0j]],
        )

    @dtypes(torch.float32, torch.float64, torch.float16)
    def test_dtype_inference(self, device, dtype):
        # [FUNCTION DOC] Under each of float32, float64 and float16 as default dtype, a tensor
        # built from a mixed real/complex literal list must infer chalf, cfloat and cdouble
        # respectively.
        # issue: https://github.com/pytorch/pytorch/issues/36834
        with set_default_dtype(dtype):
            x = torch.tensor([3.0, 3.0 + 5.0j], device=device)
        if dtype == torch.float16:
            self.assertEqual(x.dtype, torch.chalf)
        elif dtype == torch.float32:
            self.assertEqual(x.dtype, torch.cfloat)
        else:
            self.assertEqual(x.dtype, torch.cdouble)

    @dtypes(*complex_types())
    def test_conj_copy(self, device, dtype):
        # [FUNCTION DOC] Regression test: copying the conjugate of a tensor into the tensor
        # itself must give the elementwise conjugate values.
        # issue: https://github.com/pytorch/pytorch/issues/106051
        x1 = torch.tensor([5 + 1j, 2 + 2j], device=device, dtype=dtype)
        xc1 = torch.conj(x1)
        x1.copy_(xc1)
        self.assertEqual(x1, torch.tensor([5 - 1j, 2 - 2j], device=device, dtype=dtype))

    @dtypes(*complex_types())
    def test_all(self, device, dtype):
        # [FUNCTION DOC] Regression test: the all reduction over a complex tensor with no zero
        # element (including a pure imaginary and a pure real entry) must be true.
        # issue: https://github.com/pytorch/pytorch/issues/120875
        x = torch.tensor([1 + 2j, 3 - 4j, 5j, 6], device=device, dtype=dtype)

        self.assertTrue(torch.all(x))

    @dtypes(*complex_types())
    def test_any(self, device, dtype):
        # [FUNCTION DOC] Regression test: the any reduction over a complex tensor of various
        # signed zeros must be false.
        # issue: https://github.com/pytorch/pytorch/issues/120875
        x = torch.tensor(
            [0, 0j, -0 + 0j, -0 - 0j, 0 + 0j, 0 - 0j], device=device, dtype=dtype
        )

        self.assertFalse(torch.any(x))

    @onlyCPU
    @dtypes(*complex_types())
    def test_eq(self, device, dtype):
        # [FUNCTION DOC] CPU only. For three non-vectorized pairs of single-element tensors and
        # one vectorized pair of 12-element tensors (including NaN parts), compares the eq
        # result of different tensors and of a tensor with itself against the expected booleans
        # (NaN never equals, even to itself). Repeats with an out argument pre-filled with a
        # complex constant and expects the out values to be 0 or 1 in the complex dtype. Failure
        # messages include actual and expected.
        "Test eq on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype),
            ),
        ):
            actual = torch.eq(a, b)
            expected = torch.tensor([False], device=device, dtype=torch.bool)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq\nactual {actual}\nexpected {expected}",
            )

            actual = torch.eq(a, a)
            expected = torch.tensor([True], device=device, dtype=torch.bool)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, b, out=actual)
            expected = torch.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq(out)\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, a, out=actual)
            expected = torch.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq(out)\nactual {actual}\nexpected {expected}",
            )

        # Vectorized operations
        for a, b in (
            (
                torch.tensor(
                    [
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(2.8871, nan),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(nan, -3.2650),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
                torch.tensor(
                    [
                        -6.1278 - 8.5019j,
                        0.5886 + 8.8816j,
                        complex(2.8871, nan),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                        -6.1278 - 2.1172j,
                        5.1576 + 8.8816j,
                        complex(nan, -3.2650),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
            ),
        ):
            actual = torch.eq(a, b)
            expected = torch.tensor(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq\nactual {actual}\nexpected {expected}",
            )

            actual = torch.eq(a, a)
            expected = torch.tensor(
                [
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, b, out=actual)
            expected = torch.tensor(
                [
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq(out)\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, a, out=actual)
            expected = torch.tensor(
                [
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\neq(out)\nactual {actual}\nexpected {expected}",
            )

    @onlyCPU
    @dtypes(*complex_types())
    def test_ne(self, device, dtype):
        # [FUNCTION DOC] CPU only. Mirror image of the eq test for ne: same pairs, expecting
        # inverted results for booleans, and with an out argument expecting complex 1 for
        # unequal and 0 for equal entries, with NaN-containing entries counted as unequal even
        # to themselves.
        "Test ne on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype),
            ),
        ):
            actual = torch.ne(a, b)
            expected = torch.tensor([True], device=device, dtype=torch.bool)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne\nactual {actual}\nexpected {expected}",
            )

            actual = torch.ne(a, a)
            expected = torch.tensor([False], device=device, dtype=torch.bool)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, b, out=actual)
            expected = torch.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne(out)\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, a, out=actual)
            expected = torch.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne(out)\nactual {actual}\nexpected {expected}",
            )

        # Vectorized operations
        for a, b in (
            (
                torch.tensor(
                    [
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(2.8871, nan),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(nan, -3.2650),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
                torch.tensor(
                    [
                        -6.1278 - 8.5019j,
                        0.5886 + 8.8816j,
                        complex(2.8871, nan),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                        -6.1278 - 2.1172j,
                        5.1576 + 8.8816j,
                        complex(nan, -3.2650),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
            ),
        ):
            actual = torch.ne(a, b)
            expected = torch.tensor(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne\nactual {actual}\nexpected {expected}",
            )

            actual = torch.ne(a, a)
            expected = torch.tensor(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, b, out=actual)
            expected = torch.tensor(
                [
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne(out)\nactual {actual}\nexpected {expected}",
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, a, out=actual)
            expected = torch.tensor(
                [
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual,
                expected,
                msg=lambda msg: f"{msg}\nne(out)\nactual {actual}\nexpected {expected}",
            )


instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
