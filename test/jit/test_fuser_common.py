# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests shared by the JIT fuser backends. Currently holds one check that autodiff-
# optimized script functions fall back correctly when the requires-grad pattern of the input
# changes. Not runnable directly: running the file as a script raises an error that points to
# the umbrella test runner.

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestFuserCommon(JitTestCase):
    # [CLASS DOC] JitTestCase holding fuser-independent tests.
    def test_autodiff_fallback(self):
        # [FUNCTION DOC] For both values of a flag, scripts a function computing the elementwise
        # maximum of the square and cube of its input and runs it five times on a random tensor
        # whose requires-grad setting is the negation of the flag, so an optimized version is
        # created. Then calls it with an input whose requires-grad equals the flag, which the
        # optimization does not cover, and asserts the output requires-grad matches the flag.
        for rq in [True, False]:

            @torch.jit.script
            def fn(x):
                # [FUNCTION DOC] HELPER: Scripted function returning the elementwise maximum of
                # the input raised to the powers two and three.
                return torch.max(x**2.0, x**3.0)

            x = torch.randn(5, requires_grad=not rq)
            # cause optimization to be created
            for _ in range(5):
                fn(x)
            # test fallback when optimization is not applicable
            y = fn(torch.randn(5, requires_grad=rq))
            self.assertEqual(y.requires_grad, rq)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit_fuser_te.py")
