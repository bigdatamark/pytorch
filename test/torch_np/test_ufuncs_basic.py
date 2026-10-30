# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests of ufunc casting, broadcasting, dtype and out-argument behavior for
# torch._numpy, written so they also pass against real NumPy (which is used under
# TorchDynamo). Covers a unary ufunc (sine), a set of binary ufuncs, agreement between ndarray
# arithmetic operators (including reflected and in-place forms) and the corresponding ufuncs,
# and the dtype keyword. Import builds parameter decorators for the ufunc lists and casting
# modes plus the list of operator triples for add, subtract, multiply.

"""
Poking around ufunc casting/broadcasting/dtype/out behavior.

The goal is to validate on numpy, and tests should work when replacing
>>> import numpy as no

by
>>> import torch._numpy as np
"""

import operator
from unittest import skipIf as skip, SkipTest

from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


parametrize_unary_ufuncs = parametrize("ufunc", [np.sin])
parametrize_casting = parametrize(
    "casting", ["no", "equiv", "safe", "same_kind", "unsafe"]
)


@instantiate_parametrized_tests
class TestUnaryUfuncs(TestCase):
    # [CLASS DOC] Behavior tests for unary ufuncs, currently parametrized to sine only.
    def get_x(self, ufunc):
        # [FUNCTION DOC] HELPER: Returns the standard input for the unary tests, a float64
        # array of the integers zero to four; the ufunc argument is ignored.
        return np.arange(5, dtype="float64")

    @parametrize_unary_ufuncs
    def test_scalar(self, ufunc):
        # [FUNCTION DOC] Takes the first element of the input array and checks the ufunc
        # result can be converted to a Python float.
        # check that ufunc accepts a scalar and the result is convertible to scalar
        x = self.get_x(ufunc)[0]
        float(ufunc(x))

    @skip(True, reason="XXX: unary ufuncs ignore the dtype=... parameter")
    @parametrize_unary_ufuncs
    def test_x_and_dtype(self, ufunc):
        # [FUNCTION DOC] Skipped unconditionally because unary ufuncs ignore the dtype
        # keyword. When enabled it would apply the ufunc with dtype float and assert the
        # result dtype is float.
        x = self.get_x(ufunc)
        res = ufunc(x, dtype="float")
        if res.dtype != np.dtype("float"):
            raise AssertionError(f"Expected dtype float, got {res.dtype}")

    @skip(True, reason="XXX: unary ufuncs ignore the dtype=... parameter")
    @parametrize_casting
    @parametrize_unary_ufuncs
    @parametrize("dtype", ["float64", "complex128", "float32"])
    def test_x_and_dtype_casting(self, ufunc, casting, dtype):
        # [FUNCTION DOC] Skipped unconditionally because unary ufuncs ignore the dtype
        # keyword. When enabled it would apply the ufunc with a target dtype and casting mode,
        # expecting TypeError when the cast is not allowed and otherwise a result of the
        # requested dtype.
        x = self.get_x(ufunc)
        if not np.can_cast(x, dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, dtype=dtype, casting=casting)
        else:
            result = ufunc(x, dtype=dtype, casting=casting)
            if result.dtype != dtype:
                raise AssertionError(f"Expected dtype {dtype}, got {result.dtype}")

    @parametrize_casting
    @parametrize_unary_ufuncs
    @parametrize("out_dtype", ["float64", "complex128", "float32"])
    def test_x_and_out_casting(self, ufunc, casting, out_dtype):
        # [FUNCTION DOC] Applies the ufunc with a preallocated output array of the given dtype
        # and a casting mode. If the input cannot be cast to the output dtype under that mode,
        # expects TypeError; otherwise asserts the result has the output dtype and is the very
        # same object as the output array.
        x = self.get_x(ufunc)
        out = np.empty_like(x, dtype=out_dtype)
        if not np.can_cast(x, out_dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            result = ufunc(x, out=out, casting=casting)
            if result.dtype != out_dtype:
                raise AssertionError(f"Expected dtype {out_dtype}, got {result.dtype}")
            if result is not out:
                raise AssertionError("Expected result to be out")

    @parametrize_unary_ufuncs
    def test_x_and_out_broadcast(self, ufunc):
        # [FUNCTION DOC] Checks the out argument when it is larger than the input: for an
        # output of shape five by five and then one by five, the result must be the output
        # object and must equal applying the ufunc to the input broadcast to that shape.
        x = self.get_x(ufunc)
        out = np.empty((x.shape[0], x.shape[0]))

        x_b = np.broadcast_to(x, out.shape)

        res_out = ufunc(x, out=out)
        res_bcast = ufunc(x_b)
        # TODO: switching the order causes a graph break, failing the test.
        # See test/dynamo/test_misc.py -k test_numpy_graph_break
        if res_out is not out:
            raise AssertionError("Expected res_out to be out")
        assert_equal(res_out, res_bcast)

        out = np.empty((1, x.shape[0]))
        x_b = np.broadcast_to(x, out.shape)

        res_out = ufunc(x, out=out)
        res_bcast = ufunc(x_b)
        if res_out is not out:
            raise AssertionError("Expected res_out to be out")
        assert_equal(res_out, res_bcast)


ufunc_op_iop_numeric = [
    (np.add, operator.__add__, operator.__iadd__),
    (np.subtract, operator.__sub__, operator.__isub__),
    (np.multiply, operator.__mul__, operator.__imul__),
]

ufuncs_with_dunders = [ufunc for ufunc, _, _ in ufunc_op_iop_numeric]
numeric_binary_ufuncs = [
    np.float_power,
    np.power,
]

# these are not implemented for complex inputs
no_complex = [
    np.floor_divide,
    np.hypot,
    np.arctan2,
    np.copysign,
    np.fmax,
    np.fmin,
    np.fmod,
    np.heaviside,
    np.logaddexp,
    np.logaddexp2,
    np.maximum,
    np.minimum,
]

parametrize_binary_ufuncs = parametrize(
    "ufunc", ufuncs_with_dunders + numeric_binary_ufuncs + no_complex
)


# TODO: these snowflakes need special handling
"""
 'bitwise_and',
 'bitwise_or',
 'bitwise_xor',
 'equal',
 'lcm',
 'ldexp',
 'left_shift',
 'less',
 'less_equal',
 'gcd',
 'greater',
 'greater_equal',
 'logical_and',
 'logical_or',
 'logical_xor',
 'matmul',
 'not_equal',
"""


@instantiate_parametrized_tests
class TestBinaryUfuncs(TestCase):
    # [CLASS DOC] Behavior tests for binary ufuncs (add, subtract, multiply, float_power,
    # power and the ones that reject complex input), parametrized over the ufunc list.
    def get_xy(self, ufunc):
        # [FUNCTION DOC] HELPER: Returns two float64 input arrays: zero to four, and eight to
        # twelve.
        return np.arange(5, dtype="float64"), np.arange(8, 13, dtype="float64")

    @parametrize_binary_ufuncs
    def test_scalar(self, ufunc):
        # [FUNCTION DOC] Applies the ufunc to the first element of each input and checks the
        # result can be converted to a Python float.
        # check that ufunc accepts a scalar and the result is convertible to scalar
        xy = self.get_xy(ufunc)
        x, y = xy[0][0], xy[1][0]
        float(ufunc(x, y))

    @parametrize_binary_ufuncs
    def test_vector_vs_scalar(self, ufunc):
        # [FUNCTION DOC] Asserts the ufunc applied to the two arrays equals a list built by
        # applying the ufunc to each pair of elements separately.
        x, y = self.get_xy(ufunc)
        assert_equal(ufunc(x, y), [ufunc(a, b) for a, b in zip(x, y)])

    @parametrize_casting
    @parametrize_binary_ufuncs
    @parametrize("out_dtype", ["float64", "complex128", "float32"])
    def test_xy_and_out_casting(self, ufunc, casting, out_dtype):
        # [FUNCTION DOC] Applies the ufunc with an output array of the given dtype and casting
        # mode. Skips the case if the output is complex and the ufunc rejects complex input.
        # If either input cannot be cast to the output dtype, expects TypeError; otherwise
        # asserts the result has that dtype and is the output object.
        # KNOWN ISSUE: the error branch calls the ufunc with only the first input, so the
        # TypeError may come from the missing second argument rather than from the casting
        # rule being tested.
        x, y = self.get_xy(ufunc)
        out = np.empty_like(x, dtype=out_dtype)

        if ufunc in no_complex and np.issubdtype(out_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        can_cast_x = np.can_cast(x, out_dtype, casting=casting)
        can_cast_y = np.can_cast(y, out_dtype, casting=casting)

        if not (can_cast_x and can_cast_y):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            result = ufunc(x, y, out=out, casting=casting)
            if result.dtype != out_dtype:
                raise AssertionError(f"Expected dtype {out_dtype}, got {result.dtype}")
            if result is not out:
                raise AssertionError("Expected result to be out")

    @parametrize_binary_ufuncs
    def test_xy_and_out_broadcast(self, ufunc):
        # [FUNCTION DOC] Applies the ufunc to a row and a column input with a
        # three-dimensional output of shape two by five by five, asserting the result is the
        # output object and equals the ufunc applied to both inputs explicitly broadcast to
        # the output shape.
        x, y = self.get_xy(ufunc)
        y = y[:, None]
        out = np.empty((2, y.shape[0], x.shape[0]))

        x_b = np.broadcast_to(x, out.shape)
        y_b = np.broadcast_to(y, out.shape)

        res_out = ufunc(x, y, out=out)
        res_bcast = ufunc(x_b, y_b)

        # TODO: switching the order causes a graph break, failing the test.
        # See test/dynamo/test_misc.py -k test_numpy_graph_break
        if res_out is not out:
            raise AssertionError("Expected res_out to be out")
        assert_equal(res_out, res_bcast)


dtypes_numeric = [np.int32, np.float32, np.float64, np.complex128]


@instantiate_parametrized_tests
class TestNdarrayDunderVsUfunc(TestCase):
    # [CLASS DOC] Compares ndarray arithmetic operators (normal, reflected, in-place) against
    # the equivalent ufuncs for add, subtract and multiply, across other-operand dtypes.
    """Test ndarray dunders which delegate to ufuncs, vs ufuncs."""

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    def test_basic(self, ufunc, op, iop):
        # [FUNCTION DOC] Checks that the operator, its reflected form and its in-place form
        # give the same values as the ufunc for an integer array combined with a scalar, a
        # list and an array of the same data, and that in-place modification updates the left
        # operand.
        """basic op/rop/iop, no dtypes, no broadcasting"""

        # __add__
        a = np.array([1, 2, 3])
        assert_equal(op(a, 1), ufunc(a, 1))
        assert_equal(op(a, a.tolist()), ufunc(a, a.tolist()))
        assert_equal(op(a, a), ufunc(a, a))

        # __radd__
        a = np.array([1, 2, 3])
        assert_equal(op(1, a), ufunc(1, a))
        assert_equal(op(a.tolist(), a), ufunc(a, a.tolist()))

        # __iadd__
        a0 = np.array([2, 4, 6])
        a = a0.copy()

        iop(a, 2)  # modifies a in-place
        assert_equal(a, op(a0, 2))

        a0 = np.array([2, 4, 6])
        a = a0.copy()
        iop(a, a)
        assert_equal(a, op(a0, a0))

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @parametrize("other_dtype", dtypes_numeric)
    def test_other_scalar(self, ufunc, op, iop, other_dtype):
        # [FUNCTION DOC] Combines an integer array with a scalar of another numeric dtype
        # using the operator and reflected operator, asserting values equal the ufunc and the
        # dtype equals the promoted dtype. The in-place form is only checked when the promoted
        # dtype can be cast back to the array dtype under same-kind rules (dtype must then be
        # the promoted one); otherwise it must raise TypeError or RuntimeError. Skips complex
        # dtypes for ufuncs that reject them.
        """Test op/iop/rop when the other argument is a scalar of a different dtype."""
        a = np.array([1, 2, 3])
        b = other_dtype(3)

        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        # __op__
        result = op(a, b)
        assert_equal(result, ufunc(a, b))

        if result.dtype != np.result_type(a, b):
            raise AssertionError(
                f"Expected dtype {np.result_type(a, b)}, got {result.dtype}"
            )

        # __rop__
        result = op(b, a)
        assert_equal(result, ufunc(b, a))
        if result.dtype != np.result_type(a, b):
            raise AssertionError(
                f"Expected dtype {np.result_type(a, b)}, got {result.dtype}"
            )

        # __iop__ : casts the result to self.dtype, raises if cannot
        can_cast = np.can_cast(
            np.result_type(a.dtype, other_dtype), a.dtype, casting="same_kind"
        )
        if can_cast:
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))
            if result.dtype != np.result_type(a, b):
                raise AssertionError(
                    f"Expected dtype {np.result_type(a0, b)}, got {result.dtype}"
                )

        else:
            with assert_raises((TypeError, RuntimeError)):  # XXX np.UFuncTypeError
                iop(a, b)

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @parametrize("other_dtype", dtypes_numeric)
    def test_other_array(self, ufunc, op, iop, other_dtype):
        # [FUNCTION DOC] Same checks as the scalar variant but with the other operand being an
        # array of a different numeric dtype: operator and reflected operator match the ufunc
        # and the promoted dtype; the in-place form either matches the ufunc or raises
        # TypeError or RuntimeError depending on same-kind castability.
        """Test op/iop/rop when the other argument is an array of a different dtype."""
        a = np.array([1, 2, 3])
        b = np.array([5, 6, 7], dtype=other_dtype)

        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        # __op__
        result = op(a, b)
        assert_equal(result, ufunc(a, b))
        if result.dtype != np.result_type(a, b):
            raise AssertionError(
                f"Expected dtype {np.result_type(a, b)}, got {result.dtype}"
            )

        # __rop__(other array)
        result = op(b, a)
        assert_equal(result, ufunc(b, a))
        if result.dtype != np.result_type(a, b):
            raise AssertionError(
                f"Expected dtype {np.result_type(a, b)}, got {result.dtype}"
            )

        # __iop__
        can_cast = np.can_cast(
            np.result_type(a.dtype, other_dtype), a.dtype, casting="same_kind"
        )
        if can_cast:
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))
            if result.dtype != np.result_type(a, b):
                raise AssertionError(
                    f"Expected dtype {np.result_type(a0, b)}, got {result.dtype}"
                )
        else:
            with assert_raises((TypeError, RuntimeError)):  # XXX np.UFuncTypeError
                iop(a, b)

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    def test_other_array_bcast(self, ufunc, op, iop):
        # [FUNCTION DOC] Checks the operators with broadcasting between a row and a column
        # array: the shape, values and dtype of operator and reflected results must match the
        # ufunc. The in-place form with a column on the right must raise ValueError or
        # RuntimeError because the left operand cannot be broadcast; in-place with a row
        # broadcast on the right must match the ufunc.
        """Test op/rop/iop with broadcasting"""
        # __op__
        a = np.array([1, 2, 3])
        result_op = op(a, a[:, None])
        result_ufunc = ufunc(a, a[:, None])
        if result_op.shape != result_ufunc.shape:
            raise AssertionError(
                f"Expected shape {result_ufunc.shape}, got {result_op.shape}"
            )
        assert_equal(result_op, result_ufunc)

        if result_op.dtype != result_ufunc.dtype:
            raise AssertionError(
                f"Expected dtype {result_ufunc.dtype}, got {result_op.dtype}"
            )

        # __rop__
        a = np.array([1, 2, 3])
        result_op = op(a[:, None], a)
        result_ufunc = ufunc(a[:, None], a)
        if result_op.shape != result_ufunc.shape:
            raise AssertionError(
                f"Expected shape {result_ufunc.shape}, got {result_op.shape}"
            )
        assert_equal(result_op, result_ufunc)

        if result_op.dtype != result_ufunc.dtype:
            raise AssertionError(
                f"Expected dtype {result_ufunc.dtype}, got {result_op.dtype}"
            )

        # __iop__ : in-place ops (`self += other` etc) do not broadcast self
        b = a[:, None].copy()
        with assert_raises((ValueError, RuntimeError)):  # XXX ValueError in numpy
            iop(a, b)

        # however, `self += other` broadcasts other
        aa = np.broadcast_to(a, (3, 3)).copy()
        aa0 = aa.copy()

        result = iop(aa, a)
        result_ufunc = ufunc(aa0, a)

        if result.shape != result_ufunc.shape:
            raise AssertionError(
                f"Expected shape {result_ufunc.shape}, got {result.shape}"
            )
        assert_equal(result, result_ufunc)

        if result_op.dtype != result_ufunc.dtype:
            raise AssertionError(
                f"Expected dtype {result_ufunc.dtype}, got {result_op.dtype}"
            )


class TestUfuncDtypeKwd(TestCase):
    # [CLASS DOC] Tests of the dtype keyword of binary ufuncs, including the combination with
    # out.
    def test_binary_ufunc_dtype(self):
        # [FUNCTION DOC] Adds one and a tiny increment: by default the result is float64 and
        # differs from one; with dtype float32 the result has that dtype and equals one
        # (precision loss); with a boolean dtype and unsafe casting the result dtype is bool.
        # default computation uses float64:
        r64 = np.add(1, 1e-15)
        if r64.dtype != "float64":
            raise AssertionError(f"Expected dtype float64, got {r64.dtype}")
        if not r64 - 1 > 0:
            raise AssertionError("Expected r64 - 1 > 0")

        # force the float32 dtype: loss of precision
        r32 = np.add(1, 1e-15, dtype="float32")
        if r32.dtype != "float32":
            raise AssertionError(f"Expected dtype float32, got {r32.dtype}")
        if r32 != 1:
            raise AssertionError(f"Expected r32 == 1, got {r32}")

        # now force the cast
        rb = np.add(1.0, 1e-15, dtype=bool, casting="unsafe")
        if rb.dtype != bool:
            raise AssertionError(f"Expected dtype bool, got {rb.dtype}")

    def test_binary_ufunc_dtype_and_out(self):
        # [FUNCTION DOC] Adds a tiny increment to a two-element list, checking four
        # combinations of the dtype keyword and an out array: float64 only (no precision
        # loss); float32 only (precision loss); computation in float32 with float64 output
        # (loss kept, result float64); computation in float64 with float32 output (cast
        # truncates, result float32).
        # all in float64: no precision loss
        out64 = np.empty(2, dtype=np.float64)
        r64 = np.add([1.0, 2.0], 1.0e-15, out=out64)

        if not (r64 != [1.0, 2.0]).all():
            raise AssertionError("Expected r64 != [1.0, 2.0]")
        if r64.dtype != np.float64:
            raise AssertionError(f"Expected dtype float64, got {r64.dtype}")

        # all in float32: loss of precision, result is float32
        out32 = np.empty(2, dtype=np.float32)
        r32 = np.add([1.0, 2.0], 1.0e-15, dtype=np.float32, out=out32)
        if not (r32 == [1, 2]).all():
            raise AssertionError("Expected r32 == [1, 2]")
        if r32.dtype != np.float32:
            raise AssertionError(f"Expected dtype float32, got {r32.dtype}")

        # dtype is float32, so computation is in float32: precision loss
        # the result is then cast to float64
        out64 = np.empty(2, dtype=np.float64)
        r = np.add([1.0, 2.0], 1.0e-15, dtype=np.float32, out=out64)
        if not (r == [1, 2]).all():
            raise AssertionError("Expected r == [1, 2]")
        if r.dtype != np.float64:
            raise AssertionError(f"Expected dtype float64, got {r.dtype}")

        # Internal computations are in float64, but the final cast to out.dtype
        # truncates the precision => precision loss.
        out32 = np.empty(2, dtype=np.float32)
        r = np.add([1.0, 2.0], 1.0e-15, dtype=np.float64, out=out32)
        if not (r == [1, 2]).all():
            raise AssertionError("Expected r == [1, 2]")
        if r.dtype != np.float32:
            raise AssertionError(f"Expected dtype float32, got {r.dtype}")


if __name__ == "__main__":
    run_tests()
