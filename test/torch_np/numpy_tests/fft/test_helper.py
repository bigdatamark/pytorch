# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the fft helper functions fftshift, ifftshift, fftfreq, rfftfreq and the
# irfftn axes handling, run against torch._numpy or real numpy under dynamo. Import time only
# selects the numpy implementation and imports test utilities.

"""Test functions for fftpack.helper module

Copied from fftpack.helper by Pearu Peterson, October 2005

"""

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import fft, pi
    from numpy.testing import assert_array_almost_equal
else:
    import torch._numpy as np
    from torch._numpy import fft, pi
    from torch._numpy.testing import assert_array_almost_equal


class TestFFTShift(TestCase):
    # [CLASS DOC] Tests for fftshift and ifftshift on lists and random arrays.
    def test_definition(self):
        # [FUNCTION DOC] Checks fftshift and ifftshift against hand written sequences of odd and
        # even length (zero frequency moves to the center and back).
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)

    def test_inverse(self):
        # [FUNCTION DOC] For several lengths, ifftshift applied to the fftshift of a random
        # vector must return the original vector.
        for n in [1, 4, 9, 100, 211]:
            x = np.random.random((n,))
            assert_array_almost_equal(fft.ifftshift(fft.fftshift(x)), x)

    def test_axes_keyword(self):
        # [FUNCTION DOC] Checks the axes argument on a three by three grid: both axes, a single
        # axis given as an integer or tuple, and the default, against expected shifted grids in
        # both directions.
        freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
        shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shifted)
        assert_array_almost_equal(
            fft.fftshift(freqs, axes=0), fft.fftshift(freqs, axes=(0,))
        )
        assert_array_almost_equal(fft.ifftshift(shifted, axes=(0, 1)), freqs)
        assert_array_almost_equal(
            fft.ifftshift(shifted, axes=0), fft.ifftshift(shifted, axes=(0,))
        )

        assert_array_almost_equal(fft.fftshift(freqs), shifted)
        assert_array_almost_equal(fft.ifftshift(shifted), freqs)

    def test_uneven_dims(self):
        # [FUNCTION DOC] Uses a three by two array to check shifts along axis zero, axis one,
        # both axes, and none or default (all axes), with axes given as ints, tuples or lists.
        """Test 2D input, which has uneven dimension sizes"""
        freqs = [[0, 1], [2, 3], [4, 5]]

        # shift in dimension 0
        shift_dim0 = [[4, 5], [0, 1], [2, 3]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=0), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=0), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # shift in dimension 1
        shift_dim1 = [[1, 0], [3, 2], [5, 4]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=1), shift_dim1)
        assert_array_almost_equal(fft.ifftshift(shift_dim1, axes=1), freqs)

        # shift in both dimensions
        shift_dim_both = [[5, 4], [1, 0], [3, 2]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # axes=None (default) shift in all dimensions
        assert_array_almost_equal(fft.fftshift(freqs, axes=None), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=None), freqs)
        assert_array_almost_equal(fft.fftshift(freqs), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both), freqs)

    def test_equal_to_original(self):
        # [FUNCTION DOC] Compares the library fftshift and ifftshift against local
        # reimplementations of the old take based algorithms for every array shape up to fifteen
        # by fifteen (including empty ones) and for several axes arguments.
        """Test that the new (>=v1.15) implementation (see #10073) is equal to the original (<=v1.14)"""
        if TEST_WITH_TORCHDYNAMO:
            from numpy import arange, asarray, concatenate, take
        else:
            from torch._numpy import arange, asarray, concatenate, take

        def original_fftshift(x, axes=None):
            # [FUNCTION DOC] HELPER: reference fftshift in the old style: for each requested
            # axis (all axes by default, or the single integer given) rolls the array by
            # gathering the index sequence that starts at half the axis length rounded up.
            # Returns the shifted array.
            """How fftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        def original_ifftshift(x, axes=None):
            # [FUNCTION DOC] HELPER: reference ifftshift in the old style, the inverse of the
            # reference fftshift, using a split point of length minus half the length rounded
            # up. Returns the array.
            """How ifftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = n - (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        # create possible 2d array combinations and try all possible keywords
        # compare output to original functions
        for i in range(16):
            for j in range(16):
                for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                    inp = np.random.rand(i, j)

                    assert_array_almost_equal(
                        fft.fftshift(inp, axes_keyword),
                        original_fftshift(inp, axes_keyword),
                    )

                    assert_array_almost_equal(
                        fft.ifftshift(inp, axes_keyword),
                        original_ifftshift(inp, axes_keyword),
                    )


class TestFFTFreq(TestCase):
    # [CLASS DOC] Tests for the fftfreq sample frequency helper.
    def test_definition(self):
        # [FUNCTION DOC] Scaling fftfreq of length nine or ten by the length (and the sample
        # spacing pi) must give the integer frequency bins in standard order.
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        assert_array_almost_equal(9 * fft.fftfreq(9), x)
        assert_array_almost_equal(9 * pi * fft.fftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        assert_array_almost_equal(10 * fft.fftfreq(10), x)
        assert_array_almost_equal(10 * pi * fft.fftfreq(10, pi), x)


class TestRFFTFreq(TestCase):
    # [CLASS DOC] Tests for the rfftfreq sample frequency helper.
    def test_definition(self):
        # [FUNCTION DOC] Scaling rfftfreq of length nine or ten by the length (and the sample
        # spacing pi) must give the non-negative integer bins.
        x = [0, 1, 2, 3, 4]
        assert_array_almost_equal(9 * fft.rfftfreq(9), x)
        assert_array_almost_equal(9 * pi * fft.rfftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, 5]
        assert_array_almost_equal(10 * fft.rfftfreq(10), x)
        assert_array_almost_equal(10 * pi * fft.rfftfreq(10, pi), x)


class TestIRFFTN(TestCase):
    # [CLASS DOC] Test for irfftn over a non-final axis.
    def test_not_last_axis_success(self):
        # [FUNCTION DOC] Builds a random complex array and runs irfftn restricted to the second
        # to last axis; the call must complete without raising an error.
        ar, ai = np.random.random((2, 16, 8, 32))
        a = ar + 1j * ai

        axes = (-2,)

        # Should not raise error
        fft.irfftn(a, axes=axes)


if __name__ == "__main__":
    run_tests()
