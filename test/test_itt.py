# Owner(s): ["module: intel"]

# [MODULE DOC] Smoke test for the Intel ITT bindings in torch.profiler.itt. Defines one
# test case, skipped entirely when ITT is not available. Imports load_tests for sharding.
import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

@unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
class TestItt(TestCase):
    # [CLASS DOC] Test case skipped unless the ITT profiler bindings report availability.
    # Holds a single symbol-visibility test and no state.
    def test_itt(self):
        # [FUNCTION DOC] Calls range push with a label, mark with a label, and range pop
        # in order, only to confirm these bindings exist and run without raising. No
        # result is checked.
        # Just making sure we can see the symbols
        torch.profiler.itt.range_push("foo")
        torch.profiler.itt.mark("bar")
        torch.profiler.itt.range_pop()

if __name__ == '__main__':
    run_tests()
