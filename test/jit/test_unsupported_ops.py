# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests documenting TorchScript operations that are unsupported but work in eager
# Python: factory functions called with a requires_grad keyword, and the torch.nn.init
# helpers. A failing test here means parity improved and the unsupported-behavior
# documentation page must be updated. Importing adds the test directory to the module search
# path; running the file directly is rejected.

import os
import sys
import unittest

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# NOTE: FIXING FAILING TESTS
# If you are seeing a test failure from this file, congrats, you improved
# parity between JIT and Python API. Before you fix the test, you must also update
# the corresponding section in documentation that states the unsupported behavior.
# see: `jit_unsupported.rst`


class TestUnsupportedOps(JitTestCase):
    # [CLASS DOC] Test case holding checks that specific eager-valid calls fail to compile
    # under TorchScript. Stateless.
    def test_factory_ops_requires_grad_fail(self):
        # [FUNCTION DOC] Verifies that torch.ones, torch.randn and torch.zeros called with the
        # requires_grad keyword run in eager but fail scripting with an error saying that
        # keyword argument is unknown, with the highlighted source text being the factory call
        # name. Only these three of the affected operators are checked.
        # Keyword argument {name} unknown is a JIT-only error message,
        # so these functions are succeeding in eager and failing in JIT

        # Complete issue and set of ops is https://github.com/pytorch/pytorch/issues/30761
        # only testing some because they should be fixed all at once
        # [FUNCTION DOC] HELPER: Returns a ones tensor of size two with requires_grad set.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def ones():
            return torch.ones([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.ones"
        ):
            torch.jit.script(ones)

        # [FUNCTION DOC] HELPER: Returns a random normal tensor of size two with requires_grad
        # set.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def randn():
            return torch.randn([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.randn"
        ):
            torch.jit.script(randn)

        # [FUNCTION DOC] HELPER: Returns a zeros tensor of size two with requires_grad set.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def zeros():
            return torch.zeros([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.zeros"
        ):
            torch.jit.script(zeros)

    @unittest.skipIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")
    def test_init_ops(self):
        # [FUNCTION DOC] Skipped when PyTorch lacks LAPACK. Defines six small functions each
        # calling one torch.nn.init helper (calculate_gain, eye_, dirac_, kaiming_normal_,
        # orthogonal_, sparse_). For each, calls it in eager mode to confirm it works, then
        # asserts that torch.jit.script raises some exception (any message).
        def calculate_gain():
            # [FUNCTION DOC] HELPER: Returns the recommended gain for the leaky relu
            # nonlinearity with slope 0.2.
            return torch.nn.init.calculate_gain("leaky_relu", 0.2)

        def eye_():
            # [FUNCTION DOC] HELPER: Applies the identity initializer to a 2x2 zeros tensor
            # and returns it.
            return torch.nn.init.eye_(torch.zeros([2, 2]))

        def dirac_():
            # [FUNCTION DOC] HELPER: Applies the Dirac delta initializer to an uninitialized
            # 3x16x5x5 tensor and returns it.
            return torch.nn.init.dirac_(torch.empty(3, 16, 5, 5))

        def kaiming_uniform_():
            # [FUNCTION DOC] HELPER: Despite its name, applies the normal Kaiming initializer
            # to an uninitialized 3x5 tensor and returns it.
            return torch.nn.init.kaiming_normal_(torch.empty(3, 5))

        def orthogonal_():
            # [FUNCTION DOC] HELPER: Applies the orthogonal initializer to an uninitialized
            # 3x5 tensor and returns it.
            return torch.nn.init.orthogonal_(torch.empty(3, 5))

        def sparse():
            # [FUNCTION DOC] HELPER: Applies the sparse initializer with sparsity 0.1 to an
            # uninitialized 3x5 tensor and returns it.
            return torch.nn.init.sparse_(torch.empty(3, 5), sparsity=0.1)

        for func in [
            calculate_gain,
            eye_,
            dirac_,
            kaiming_uniform_,
            orthogonal_,
            sparse,
        ]:
            # doesn't error in eager
            func()
            with self.assertRaisesRegex(Exception, ""):
                torch.jit.script(func)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
