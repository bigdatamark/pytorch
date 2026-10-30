# Owner(s): ["module: ci"]
# Sanity check for CI setup in GHA.  This file is expected to fail so it can trigger reruns
# [MODULE DOC] Sanity check for CI setup: tests assert the CI environment variable is NOT set,
# so they are expected to fail on CI and exercise the rerun machinery. No import-time side
# effects.

import os

from torch.testing._internal.common_utils import run_tests, slowTest, TestCase


class TestCISanityCheck(TestCase):
    # [CLASS DOC] Deliberately failing tests (on CI) with a regular and a slow variant.
    def test_env_vars_exist(self):
        # [FUNCTION DOC] Fails when the CI environment variable is set; expected to fail in CI
        # to trigger reruns, so a pass indicates a misconfigured environment.
        # This check should fail and trigger reruns.  If it passes, something is wrong
        self.assertTrue(os.environ.get("CI") is None)

    @slowTest
    def test_env_vars_exist_slow(self):
        # [FUNCTION DOC] Same check as the regular test but marked slow so it runs only in the
        # slow suite.
        # Same as the above, but for the slow suite
        self.assertTrue(os.environ.get("CI") is None)


if __name__ == "__main__":
    run_tests()
