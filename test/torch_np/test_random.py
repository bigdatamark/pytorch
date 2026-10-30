# Owner(s): ["module: dynamo"]
# [MODULE DOC] Smoke tests for torch._numpy.random under both random streams: the NumPy stream
# and the PyTorch stream, selected through the dynamo config flag use_numpy_random_stream.
# Covers scalar versus array return of the sampling functions, in-place shuffle of arrays (and
# refusal for lists), choice without replacement, and agreement of the NumPy stream with real
# NumPy seeding. Tests are parametrized over the stream choice via the repo test helpers.

"""Light smoke test switching between numpy to pytorch random streams."""

from contextlib import contextmanager
from functools import partial

import numpy as _np
import pytest

import torch._dynamo.config as config
import torch._numpy as tnp
from torch._numpy.testing import assert_equal
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


@contextmanager
def control_stream(use_numpy=False):
    # [FUNCTION DOC] Context manager that temporarily patches the dynamo config so the random
    # module uses the NumPy stream when use_numpy is true and the PyTorch stream otherwise
    # (default false). It yields nothing; the previous config value is restored on exit by the
    # config patch helper.
    with config.patch(use_numpy_random_stream=use_numpy):
        yield


@instantiate_parametrized_tests
class TestScalarReturn(TestCase):
    # [CLASS DOC] Checks the return type of the torch._numpy.random samplers for the default
    # and explicit size arguments. Parametrized over the stream and over each sampler
    # function.
    @parametrize("use_numpy", [True, False])
    @parametrize(
        "func",
        [
            tnp.random.normal,
            tnp.random.rand,
            partial(tnp.random.randint, 0, 5),
            tnp.random.randn,
            subtest(tnp.random.random, name="random_random"),
            subtest(tnp.random.random_sample, name="random_sample"),
            tnp.random.sample,
            tnp.random.uniform,
        ],
    )
    def test_rndm_scalar(self, func, use_numpy):
        # [FUNCTION DOC] Calls the sampler with no size argument inside the chosen stream and
        # asserts the result is a Python int or float rather than an array. The samplers are
        # normal, rand, randint bound to the range zero to five, randn, random, random_sample,
        # sample and uniform.
        # default `size` means a python scalar return
        with control_stream(use_numpy):
            r = func()
        if not isinstance(r, (int, float)):
            raise AssertionError(f"Expected int or float, got {type(r)}")

    @parametrize("use_numpy", [True, False])
    @parametrize(
        "func",
        [
            tnp.random.normal,
            tnp.random.rand,
            partial(tnp.random.randint, 0, 5),
            tnp.random.randn,
            subtest(tnp.random.random, name="random_random"),
            subtest(tnp.random.random_sample, name="random_sample"),
            tnp.random.sample,
            tnp.random.uniform,
        ],
    )
    def test_rndm_array(self, func, use_numpy):
        # [FUNCTION DOC] Calls the sampler asking for ten values inside the chosen stream
        # (rand and randn take the count as a positional argument, the others take it by the
        # size keyword) and asserts the result is a torch._numpy ndarray.
        with control_stream(use_numpy):
            if func in (tnp.random.rand, tnp.random.randn):
                r = func(10)
            else:
                r = func(size=10)
        if not isinstance(r, tnp.ndarray):
            raise AssertionError(f"Expected tnp.ndarray, got {type(r)}")


@instantiate_parametrized_tests
class TestShuffle(TestCase):
    # [CLASS DOC] Tests for in-place shuffling in torch._numpy.random, parametrized over a
    # stream flag that these tests do not actually apply.
    @parametrize("use_numpy", [True, False])
    def test_1d(self, use_numpy):
        # [FUNCTION DOC] Seeds the generator, shuffles a one-dimensional array of six values
        # in place and asserts it is still an ndarray and differs from an unshuffled copy in
        # at least one element. The use_numpy argument is accepted but not used.
        ax = tnp.asarray([1, 2, 3, 4, 5, 6])
        ox = ax.copy()

        tnp.random.seed(1234)
        tnp.random.shuffle(ax)

        if not isinstance(ax, tnp.ndarray):
            raise AssertionError(f"Expected tnp.ndarray, got {type(ax)}")
        if (ax == ox).all():
            raise AssertionError("Expected 1d array to be shuffled")

    @parametrize("use_numpy", [True, False])
    def test_2d(self, use_numpy):
        # [FUNCTION DOC] Same as the one-dimensional case for a three-by-three array; shuffle
        # only permutes the first axis. Asserts the result is an ndarray that differs from the
        # copy made before shuffling. The use_numpy argument is accepted but not used.
        # np.shuffle only shuffles the first axis
        ax = tnp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ox = ax.copy()

        tnp.random.seed(1234)
        tnp.random.shuffle(ax)

        if not isinstance(ax, tnp.ndarray):
            raise AssertionError(f"Expected tnp.ndarray, got {type(ax)}")
        if (ax == ox).all():
            raise AssertionError("Expected 2d array to be shuffled")

    @parametrize("use_numpy", [True, False])
    def test_shuffle_list(self, use_numpy):
        # [FUNCTION DOC] Asserts that shuffling a plain Python list raises NotImplementedError
        # (in eager mode lists are refused; under dynamo the call falls back to NumPy, per the
        # comments). The use_numpy argument is accepted but not used.
        # on eager, we refuse to shuffle lists
        # under dynamo, we always fall back to numpy
        # NB: this means that the random stream is different for
        # shuffling a list or an array when USE_NUMPY_STREAM == False
        x = [1, 2, 3]
        with pytest.raises(NotImplementedError):
            tnp.random.shuffle(x)


@instantiate_parametrized_tests
class TestChoice(TestCase):
    # [CLASS DOC] Tests for the choice function in torch._numpy.random.
    @parametrize("use_numpy", [True, False])
    def test_choice(self, use_numpy):
        # [FUNCTION DOC] Within the chosen stream, seeds the generator and draws three values
        # without replacement from five items with given probabilities, once with an integer
        # population and once (after reseeding) with an arange array; asserts both results are
        # equal.
        kwds = dict(size=3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        with control_stream(use_numpy):
            tnp.random.seed(12345)
            x = tnp.random.choice(5, **kwds)
            tnp.random.seed(12345)
            x_1 = tnp.random.choice(tnp.arange(5), **kwds)
            assert_equal(x, x_1)


class TestNumpyGlobal(TestCase):
    # [CLASS DOC] Checks the relation between the NumPy-stream mode and the real NumPy global
    # generator.
    def test_numpy_global(self):
        # [FUNCTION DOC] With the NumPy stream selected, seeds and draws eleven uniform
        # values, then reseeds real NumPy and draws the same and asserts equality. Then with
        # the PyTorch stream and the same seed draws again and asserts the values are not all
        # equal to the first draw. Fails with AssertionError otherwise.
        with control_stream(use_numpy=True):
            tnp.random.seed(12345)
            x = tnp.random.uniform(0, 1, size=11)

        # check that the stream is identical to numpy's
        _np.random.seed(12345)
        x_np = _np.random.uniform(0, 1, size=11)
        assert_equal(x, tnp.asarray(x_np))

        # switch to the pytorch stream, variates differ
        with control_stream(use_numpy=False):
            tnp.random.seed(12345)
            x_1 = tnp.random.uniform(0, 1, size=11)

        if (x_1 == x).all():
            raise AssertionError("Expected different variates with different stream")


if __name__ == "__main__":
    run_tests()
