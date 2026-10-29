# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo test that unittest.SkipTest raised and caught
# inside a function traced by make_dynamo_test works when the
# enable_trace_unittest config flag is on. Defines one test class.
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class TestUnittest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo test case that enables the enable_trace_unittest
    # config for each test. State: the attribute _prev saves the prior
    # config value in setUp so tearDown can restore it.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, saves the current value of
        # torch._dynamo.config enable_trace_unittest in _prev, then sets
        # that flag to True (global config mutation).
        super().setUp()
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] Calls the base tearDown, then restores
        # enable_trace_unittest to the value saved in setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._prev

    @make_dynamo_test
    def test_SkipTest(self):
        # [FUNCTION DOC] Run under make_dynamo_test (executed eagerly and traced by
        # dynamo). Raises unittest.SkipTest inside a try block, catches it with a generic
        # Exception handler, sets a local flag to 1, and asserts the flag is 1. Checks
        # SkipTest is a normal catchable Exception subclass under tracing.
        z = 0
        SkipTest = unittest.SkipTest
        try:
            raise SkipTest("abcd")
        except Exception:
            z = 1
        self.assertEqual(z, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
