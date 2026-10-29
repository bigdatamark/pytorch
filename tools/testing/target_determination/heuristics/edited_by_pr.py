# [MODULE DOC] Heuristic that gives top priority to test files modified by the current PR.
# Also holds ADDITIONAL_MAPPINGS, which maps generated test names to regexes over changed
# file paths so edits to those directories trigger the generated tests. No import-time
# side effects.

from __future__ import annotations

import re
from typing import Any

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
    query_changed_files,
)
from tools.testing.test_run import TestRun


# Some files run tests in other test files, so we map them to each other here.
# This is a map from file that runs the test to regex that matches the file that
# contains the test. Test file with path test/a/b.py should of the form a/b.
# Regexes should be based on repo root.
ADDITIONAL_MAPPINGS = {
    # Not files that are tracked by git but rather functions defined in
    # run_test.py that generate test files which run tests in test/cpp_extensions.
    "test_cpp_extensions_aot_ninja": [r"test\/cpp_extensions.*"],
    "test_cpp_extensions_aot_no_ninja": [r"test\/cpp_extensions.*"],
}


class EditedByPR(HeuristicInterface):
    # [CLASS DOC] Heuristic rating tests edited by the PR with score 1 and all others 0.
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Computes the set of modified tests and returns a TestPrioritizations over
        # the given tests where each modified test that is in the list gets score 1.
        critical_tests = _get_modified_tests()
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )


def _get_modified_tests() -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of test names that the PR touched. Takes the
    # changed files, converts changed test/*.py paths to test names, and adds each test name in
    # ADDITIONAL_MAPPINGS whose regexes match any changed file path. Queries git or GitHub.
    changed_files = query_changed_files()
    should_run = python_test_file_to_test_name(set(changed_files))
    for test_file, regexes in ADDITIONAL_MAPPINGS.items():
        if any(
            re.search(regex, changed_file) is not None
            for regex in regexes
            for changed_file in changed_files
        ):
            should_run.add(test_file)
    return should_run
