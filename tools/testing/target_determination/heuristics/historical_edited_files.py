# [MODULE DOC] Heuristic based on which files were edited together in past commits on main:
# the precomputed correlation file maps a file to test files edited alongside it. No
# import-time side effects.

from __future__ import annotations

from typing import Any

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_HISTORICAL_EDITED_FILES,
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


# This heuristic assumes that changed files in previous commits are good sources
# of information for what files are related to each other. If fileA and
# testFileA were edited in the same commit on main, that probably means that
# future commits that change fileA should probably run testFileA. Based on this,
# a correlation dict is built based on what files were edited in commits on main.
class HistorialEditedFiles(HeuristicInterface):
    # [CLASS DOC] Heuristic rating test files by co-edit history with the PR's changed files.
    # Stateless apart from trial_mode.
    # KNOWN ISSUE: the class name is misspelled (Historial); the misspelling is exported.
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Loads co-edit ratings for the changed files, keeps tests in the given
        # list, rescales so the best becomes 0.25 and returns a TestPrioritizations.
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_HISTORICAL_EDITED_FILES
        )
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}

        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
