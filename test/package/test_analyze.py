# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Tests for the torch.package analyze helpers. Falls back to a non-relative import
# of the shared base test case when the file is run directly. Run as a script, it starts the
# test runner.

import unittest

import torch
from torch.package import analyze
from torch.testing._internal.common_utils import IS_LINUX, run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestAnalyze(PackageTestCase):
    # [CLASS DOC] Test case for the dependency tracing helper in the analyze module.
    """Dependency analysis API tests."""

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/81213")
    def test_trace_dependencies(self):
        # [FUNCTION DOC] Imports the test_trace_dep module, traces dependencies of one of its
        # module instances run on a random tensor of four elements, and checks that the returned
        # set of module names excludes yaml and includes test_trace_dep. Skipped on Linux
        # because of a linked issue.
        import test_trace_dep

        obj = test_trace_dep.SumMod()

        used_modules = analyze.trace_dependencies(obj, [(torch.randn(4),)])

        self.assertNotIn("yaml", used_modules)
        self.assertIn("test_trace_dep", used_modules)


if __name__ == "__main__":
    run_tests()
