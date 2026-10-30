# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests that the scripted power operator matches Python for all int and float base
# and exponent combinations, and that a zero base with a negative exponent raises a runtime error.
# Raises an error if run directly; run via the main jit test file.

import torch
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class TestAtenPow(TestCase):
    # [CLASS DOC] Test case holding the single power-operator behavior test.
    def test_aten_pow_zero_negative_exponent(self):
        # [FUNCTION DOC] Defines four scripted power functions, one per int/float pairing of base
        # and exponent. For each, compares results over positive, negative and zero bases and
        # exponents (including negative zero floats) against Python's own power, then asserts that
        # a zero base with a negative exponent raises a runtime error. Some expected values are
        # written with different operands than the call but give the same number.
        """
        1. Testing a = int, b = int
        """

        @torch.jit.script
        def fn_int_int(a: int, b: int):
            # [FUNCTION DOC] HELPER: Scripted function raising an int base to an int exponent.
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_int_int(2, 1), 2**1)
        self.assertEqual(fn_int_int(2, 0), 2**0)
        self.assertEqual(fn_int_int(2, -2), 2 ** (-2))
        self.assertEqual(fn_int_int(-2, 2), (-2) ** 2)
        self.assertEqual(fn_int_int(-2, 0), (-2) ** 0)
        self.assertEqual(fn_int_int(-2, -2), (-2) ** (-2))
        self.assertEqual(fn_int_int(-2, -1), (-2) ** (-1))
        self.assertEqual(fn_int_int(0, 2), 0**1)
        self.assertEqual(fn_int_int(0, 0), 0**0)
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_int, 0, -2)

        """
        2. Testing a = int, b = float
        """

        @torch.jit.script
        def fn_int_float(a: int, b: float):
            # [FUNCTION DOC] HELPER: Scripted function raising an int base to a float exponent.
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_int_float(2, 2.5), 2**2.5)
        self.assertEqual(fn_int_float(2, -2.5), 2 ** (-2.5))
        self.assertEqual(fn_int_float(2, -0.0), 2 ** (-0.0))
        self.assertEqual(fn_int_float(2, 0.0), 2 ** (0.0))
        self.assertEqual(fn_int_float(-2, 2.0), (-2) ** 2.0)
        self.assertEqual(fn_int_float(-2, -2.0), (-2) ** (-2.0))
        self.assertEqual(fn_int_float(-2, -3.0), (-2) ** (-3.0))
        self.assertEqual(fn_int_float(-2, -0.0), (-2) ** (-0.0))
        self.assertEqual(fn_int_float(-2, 0.0), (-2) ** (0.0))
        self.assertEqual(fn_int_float(0, 2.0), 0**2.0)
        self.assertEqual(fn_int_float(0, 0.5), 0**0.5)
        self.assertEqual(fn_int_float(0, 0.0), 0**0.0)
        self.assertEqual(fn_int_float(0, -0.0), 0 ** (-0.0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_float, 0, -2.5)

        """
        3. Testing a = float, b = int
        """

        @torch.jit.script
        def fn_float_int(a: float, b: int):
            # [FUNCTION DOC] HELPER: Scripted function raising a float base to an int exponent.
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_float_int(2.5, 2), 2.5**2)
        self.assertEqual(fn_float_int(2.5, -2), 2.5 ** (-2))
        self.assertEqual(fn_float_int(2.5, -0), 2.5 ** (-0))
        self.assertEqual(fn_float_int(2.5, 0), 2.5**0)
        self.assertEqual(fn_float_int(-2.5, 2), 2.5**2)
        self.assertEqual(fn_float_int(-2.5, -2), (-2.5) ** (-2))
        self.assertEqual(fn_float_int(-2.5, -3), (-2.5) ** (-3))
        self.assertEqual(fn_float_int(-2.5, -0), (-2.5) ** (-0))
        self.assertEqual(fn_float_int(-2.5, 0), (-2.5) ** 0)
        self.assertEqual(fn_float_int(0.0, 2), 0**2)
        self.assertEqual(fn_float_int(0.0, 0), 0**0)
        self.assertEqual(fn_float_int(0.0, -0), 0 ** (-0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_float_int, 0.0, -2)

        """
        4. Testing a = float, b = float
        """

        @torch.jit.script
        def fn_float_float(a: float, b: float):
            # [FUNCTION DOC] HELPER: Scripted function raising a float base to a float exponent.
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_float_float(2.5, 2.0), 2.5**2.0)
        self.assertEqual(fn_float_float(2.5, -2.0), 2.5 ** (-2.0))
        self.assertEqual(fn_float_float(2.5, -0.0), 2.5 ** (-0.0))
        self.assertEqual(fn_float_float(2.5, 0.0), 2.5**0.0)
        self.assertEqual(fn_float_float(-2.5, 2.0), 2.5**2.0)
        self.assertEqual(fn_float_float(-2.5, -2.0), (-2.5) ** (-2.0))
        self.assertEqual(fn_float_float(-2.5, -3.0), (-2.5) ** (-3.0))
        self.assertEqual(fn_float_float(-2.5, -0.0), (-2.5) ** (-0.0))
        self.assertEqual(fn_float_float(-2.5, 0.0), (-2.5) ** 0.0)
        self.assertEqual(fn_float_float(0.0, 2.0), 0.0**2.0)
        self.assertEqual(fn_float_float(0.0, 0.0), 0.0**0.0)
        self.assertEqual(fn_float_float(0.0, -0.0), 0.0 ** (-0.0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_float_float, 0.0, -2.0)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
