# [MODULE DOC] Heuristic giving top priority to tests that failed previously in the same PR.
# Failures come from the pytest cache dump and from an additional file for failures that
# leave no pytest record. Also provides the writer for that additional file. REPO_ROOT is
# the repository root. No import-time side effects.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_PREVIOUSLY_FAILED,
    TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL,
)
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
)
from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[4]


class PreviouslyFailedInPR(HeuristicInterface):
    # [CLASS DOC] Heuristic rating previously failed test files with score 1. Stateless apart
    # from trial_mode.
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Unions the failures from the pytest cache and from the additional file and
        # returns a TestPrioritizations where each such test in the list has score 1.
        critical_tests = get_previous_failures() | read_additional_test_failures_file()
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )


def get_previous_failures() -> set[str]:
    # [FUNCTION DOC] Reads the pytest cache file of last failed tests from the additional CI
    # files folder and returns the set of test names. Prints a message and returns an empty set
    # if the file is missing.
    path = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PREVIOUSLY_FAILED
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return set()
    with open(path) as f:
        return python_test_file_to_test_name(
            _parse_prev_failing_test_files(json.load(f))
        )


def _parse_prev_failing_test_files(last_failed_tests: dict[str, bool]) -> set[str]:
    # [FUNCTION DOC] HELPER: Takes keys of the form file::class::method and returns the set of
    # file parts. Keys without a separator are ignored.
    prioritized_tests = set()

    # The keys are formatted as "test_file.py::test_class::test_method[params]"
    # We just need the test_file part
    for test in last_failed_tests:
        parts = test.split("::")
        if len(parts) > 1:
            test_file = parts[0]
            prioritized_tests.add(test_file)

    return prioritized_tests


def gen_additional_test_failures_file(tests: list[str]) -> None:
    # [FUNCTION DOC] Writes the list of failed test names as JSON into the pytest cache
    # directory under the repo root, creating the directory if needed. Side effect: file I/O.
    # Segfaults usually result in no xml and some tests don't run through pytest
    # (ex doctests).  In these cases, there will be no entry in the pytest
    # cache, so we should generate a separate file for them and upload it to s3
    # along with the pytest cache
    pytest_cache_dir = REPO_ROOT / ".pytest_cache"
    if not os.path.exists(pytest_cache_dir):
        os.makedirs(pytest_cache_dir)
    with open(pytest_cache_dir / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL, "w") as f:
        json.dump(tests, f, indent=2)


def read_additional_test_failures_file() -> set[str]:
    # [FUNCTION DOC] Reads the additional failures JSON list from the additional CI files
    # folder and returns it as a set. Prints a message and returns an empty set if missing;
    # otherwise prints the contents.
    path = (
        REPO_ROOT
        / ADDITIONAL_CI_FILES_FOLDER
        / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL
    )
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return set()
    with open(path) as f:
        s = set(json.load(f))
        print(f"additional failures: {s}")
        return s
