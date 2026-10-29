# [MODULE DOC] Heuristic using historical correlations between edited files and failing
# test classes. Reads a JSON mapping from file path to "test_file::test_class" scores and
# sums the scores for the PR's changed files. Contains private helpers; no import-time
# side effects.

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, cast

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_CLASS_RATINGS_FILE,
)
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    normalize_ratings,
    query_changed_files,
    REPO_ROOT,
)
from tools.testing.test_run import TestRun


class HistoricalClassFailurCorrelation(HeuristicInterface):
    # [CLASS DOC] Heuristic that rates individual test classes (as TestRun objects including
    # one class) by historical failure correlation. Stateless apart from trial_mode.
    # KNOWN ISSUE: the class name is misspelled (Failur); the misspelling is part of the
    # exported API.
    """
    This heuristic prioritizes test classes that have historically tended to fail
    when the files edited by current PR were modified.
    """

    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Gets class-level ratings, builds a TestRun per "file::class" key whose
        # test file is in the given list, rescales so the best becomes 0.25, and returns a
        # TestPrioritizations over the given tests.
        ratings = _get_ratings_for_tests(set(tests))
        test_ratings = {
            TestRun(k): v for (k, v) in ratings.items() if TestRun(k).test_file in tests
        }
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))


def _get_historical_test_class_correlations() -> dict[str, dict[str, float]]:
    # [FUNCTION DOC] HELPER: Loads the test class ratings JSON (file to class to score) from the
    # additional CI files folder under the repo root. Returns an empty mapping, after printing a
    # message, if the file does not exist.
    path = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TEST_CLASS_RATINGS_FILE
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return {}
    with open(path) as f:
        test_class_correlations = cast(dict[str, dict[str, float]], json.load(f))
        return test_class_correlations


def _get_ratings_for_tests(
    tests_to_run: set[str],
) -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Sums, for every changed file, the correlation scores of each
    # "test_file::test_class" entry whose test file is in tests_to_run. Returns a mapping from
    # qualified class name to total score; empty if no correlation data is available.
    # Get the files edited
    changed_files = query_changed_files()

    test_class_correlations = _get_historical_test_class_correlations()
    if not test_class_correlations:
        return {}

    # Find the tests failures that are correlated with the edited files.
    # Filter the list to only include tests we want to run.
    ratings: dict[str, float] = defaultdict(float)
    for file in changed_files:
        for qualified_test_class, score in test_class_correlations.get(
            file, {}
        ).items():
            # qualified_test_class looks like "test_file::test_class"
            test_file, test_class = qualified_test_class.split("::")
            if test_file in tests_to_run:
                ratings[qualified_test_class] += score

    return ratings


def _rank_correlated_tests(
    tests_to_run: list[str],
) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the qualified test classes correlated with the changed
    # files, sorted from highest to lowest score. Appears unused by the heuristic itself.
    # Find the tests failures that are correlated with the edited files.
    # Filter the list to only include tests we want to run.
    # pyrefly: ignore [bad-assignment]
    tests_to_run = set(tests_to_run)
    # pyrefly: ignore [bad-argument-type]
    ratings = _get_ratings_for_tests(tests_to_run)
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    return prioritize
