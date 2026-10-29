# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for tracing the diffusers BaseOutput dataclass UNet2DOutput
# (create, attribute assign/get, key and integer indexing, to_tuple). Import-time: tries to
# import diffusers unet_2d and sets it to None on ImportError; every test is skipped then.
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same


try:
    from diffusers.models import unet_2d
except ImportError:
    unet_2d = None


def maybe_skip(fn):
    # [FUNCTION DOC] HELPER: Decorator; returns fn wrapped in a unittest skip ("requires
    # diffusers") when unet_2d failed to import, otherwise returns fn unchanged.
    if unet_2d is None:
        return unittest.skip("requires diffusers")(fn)
    return fn


class TestBaseOutput(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase exercising UNet2DOutput inside compiled functions; each
    # test is skipped when diffusers is missing. Shared checking lives in _common.
    @maybe_skip
    def test_create(self):
        # [FUNCTION DOC] Compiles a function that constructs UNet2DOutput from a plus one
        # result and returns it, via the standard_test helper with one argument and an
        # expected op count of 1.
        def fn(a):
            # [FUNCTION DOC] HELPER: Returns a UNet2DOutput wrapping a + 1.
            tmp = unet_2d.UNet2DOutput(a + 1)
            return tmp

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=1)

    @maybe_skip
    def test_assign(self):
        # [FUNCTION DOC] Compiles (optimize_assert, one graph required) a function that
        # builds a UNet2DOutput then reassigns its sample field. Asserts the sample matches
        # eager, one frame was compiled and two ops were captured.
        def fn(a):
            # [FUNCTION DOC] HELPER: Builds UNet2DOutput(a + 1), overwrites sample with a + 2,
            # returns the output object.
            tmp = unet_2d.UNet2DOutput(a + 1)
            tmp.sample = a + 2
            return tmp

        args = [torch.randn(10)]
        obj1 = fn(*args)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1.sample, obj2.sample))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def _common(self, fn, op_count):
        # [FUNCTION DOC] HELPER: Builds a UNet2DOutput with a random size-10 sample, runs fn
        # eagerly and compiled with optimize_assert and a CompileCounter, then asserts equal
        # results, exactly one compiled frame, and op_count captured ops.
        args = [
            unet_2d.UNet2DOutput(
                sample=torch.randn(10),
            )
        ]
        obj1 = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    @maybe_skip
    def test_getattr(self):
        # [FUNCTION DOC] Checks that reading the sample attribute of the output and
        # multiplying by 10 compiles into one graph with one op.
        def fn(obj: unet_2d.UNet2DOutput):
            # [FUNCTION DOC] HELPER: Returns the sample attribute times 10.
            x = obj.sample * 10
            return x

        self._common(fn, 1)

    @maybe_skip
    def test_getitem(self):
        # [FUNCTION DOC] Checks that string-key access to sample followed by a multiply
        # compiles into one graph with one op.
        def fn(obj: unet_2d.UNet2DOutput):
            # [FUNCTION DOC] HELPER: Returns the item under key "sample" times 10.
            x = obj["sample"] * 10
            return x

        self._common(fn, 1)

    @maybe_skip
    def test_tuple(self):
        # [FUNCTION DOC] Checks that to_tuple on the output, taking the first element and
        # multiplying by 10, compiles into one graph with one op.
        def fn(obj: unet_2d.UNet2DOutput):
            # [FUNCTION DOC] HELPER: Converts obj to a tuple, returns its first item times 10.
            a = obj.to_tuple()
            return a[0] * 10

        self._common(fn, 1)

    @maybe_skip
    def test_index(self):
        # [FUNCTION DOC] Checks that integer indexing (position 0) of the output times 10
        # compiles into one graph with one op.
        def fn(obj: unet_2d.UNet2DOutput):
            # [FUNCTION DOC] HELPER: Returns the item at integer position 0 times 10.
            return obj[0] * 10

        self._common(fn, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
