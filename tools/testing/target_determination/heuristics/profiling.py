# [MODULE DOC] Heuristic based on profiling data: a precomputed mapping from source files to
# the tests that exercised them (all ratings are 1). No import-time side effects.

from __future__ import annotations

from typing import Any

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_PROFILING_FILE,
)
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    get_ratings_for_tests,
    normalize_ratings,
)
from tools.testing.test_run import TestRun


# Profilers were used to gather simple python code coverage information for each
# test to see files were involved in each tests and used to build a correlation
# dict (where all ratings are 1).
class Profiling(HeuristicInterface):
    # [CLASS DOC] Heuristic rating tests that executed code in the PR's changed files.
    # Stateless apart from trial_mode.
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Loads profiling ratings for the changed files, keeps tests in the given
        # list, rescales so the best becomes 0.25 and returns a TestPrioritizations.
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PROFILING_FILE
        )
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
