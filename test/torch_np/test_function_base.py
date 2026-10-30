# Owner(s): ["module: dynamo"]
# [MODULE DOC] Small tests for torch._numpy function-base routines: append and
# broadcast_shapes. Imports real NumPy when running under TorchDynamo (so it gets traced) and
# torch._numpy otherwise, choosing the matching assert_equal helper.


import pytest

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


class TestAppend(TestCase):
    # [CLASS DOC] Tests for the append function, with cases taken from the NumPy docstring.
    # tests taken from np.append docstring
    def test_basic(self):
        # [FUNCTION DOC] Checks that append flattens both inputs when no axis is given and
        # yields the integers one to nine, that append along axis zero stacks a one-row array
        # onto a two-row array giving a three-by-three result, and that a values array with a
        # mismatched shape raises RuntimeError or ValueError.
        result = np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
        assert_equal(result, np.arange(1, 10, dtype=int))

        # When `axis` is specified, `values` must have the correct shape.
        result = np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
        assert_equal(result, np.arange(1, 10, dtype=int).reshape((3, 3)))

        with pytest.raises((RuntimeError, ValueError)):
            np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)


class TestMisc(TestCase):
    # [CLASS DOC] Miscellaneous function-base tests.
    def test_broadcast_shapes(self):
        # [FUNCTION DOC] Checks that broadcast_shapes gives two-by-two for
        # one-by-two and two-by-two inputs.
        result = np.broadcast_shapes((1, 2), (2, 2))
        assert_equal(result, (2, 2))


if __name__ == "__main__":
    run_tests()
