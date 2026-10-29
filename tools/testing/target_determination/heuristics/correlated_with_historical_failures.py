# [MODULE DOC] Heuristic that ranks test files by how often they failed historically when
# the files changed in the current PR were modified. Ratings come from the precomputed
# test-file ratings JSON in the additional CI files folder. No import-time side effects.

from __future__ import annotations

from typing import Any

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_FILE_RATINGS_FILE,
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


class CorrelatedWithHistoricalFailures(HeuristicInterface):
    # [CLASS DOC] Heuristic subclass scoring test files with historical failure correlation
    # data. Stateless apart from the trial_mode flag set by the base class.
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Loads the per-changed-file test ratings, keeps only entries for tests in
        # the given list, wraps each name in a TestRun and rescales so the best rating becomes
        # 0.25. Returns a TestPrioritizations over all given tests (unrated tests score zero).
        # Reads the ratings file and queries the changed files (git or GitHub API).
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TEST_FILE_RATINGS_FILE
        )
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
