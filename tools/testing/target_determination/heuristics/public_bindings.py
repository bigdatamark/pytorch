# [MODULE DOC] Heuristic dedicated to the test_public_bindings test. No import-time
# side effects.

from __future__ import annotations

from typing import Any

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import query_changed_files
from tools.testing.test_run import TestRun


class PublicBindings(HeuristicInterface):
    # [CLASS DOC] Heuristic that rates test_public_bindings with 1.0 when any changed file is
    # under torch/ or is the public API allowlist, otherwise rates nothing. Class attributes:
    # test_public_bindings (the test name) and additional_files (extra trigger files).
    # Literally just a heuristic for test_public_bindings.  Pretty much anything
    # that changes the public API can affect this testp
    test_public_bindings = "test_public_bindings"
    additional_files = ["test/allowlist_for_publicAPI.json"]

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Queries the changed files and returns a TestPrioritizations in which
        # test_public_bindings has score 1.0 if a trigger file changed; all else is 0.
        test_ratings = {}
        changed_files = query_changed_files()

        if any(
            file.startswith("torch/") or file in self.additional_files
            for file in changed_files
        ):
            test_ratings[TestRun(self.test_public_bindings)] = 1.0
        return TestPrioritizations(tests, test_ratings)
