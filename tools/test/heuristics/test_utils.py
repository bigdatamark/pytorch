# [MODULE DOC] Unit tests for the target-determination heuristics utilities
# (tools.testing.target_determination.heuristics.utils), currently the rating
# normalization helper. Import-time side effect: temporarily appends the repository root
# to sys.path to import the tools package, then removes it again.
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.testing.target_determination.heuristics.utils as utils
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))


class TestHeuristicsUtils(unittest.TestCase):
    # [CLASS DOC] Test case for heuristics utility functions; defines a custom
    # approximate dictionary assertion. Stateless.
    def assertDictAlmostEqual(
        self, first: dict[TestRun, Any], second: dict[TestRun, Any]
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts two mappings have identical keys and that each
        # pair of values is approximately equal (default places of assertAlmostEqual).
        # Raises an assertion failure otherwise.
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertAlmostEqual(first[key], second[key])

    def test_normalize_ratings(self) -> None:
        # [FUNCTION DOC] Checks normalize_ratings scales ratings so the largest equals the
        # given maximum: unchanged when the max already matches, proportionally shrunk for
        # a smaller max, and shifted into the range from min_value to the max when a
        # minimum is given.
        ratings: dict[TestRun, float] = {
            TestRun("test1"): 1,
            TestRun("test2"): 2,
            TestRun("test3"): 4,
        }
        normalized = utils.normalize_ratings(ratings, 4)
        self.assertDictAlmostEqual(normalized, ratings)

        normalized = utils.normalize_ratings(ratings, 0.1)
        self.assertDictAlmostEqual(
            normalized,
            {
                TestRun("test1"): 0.025,
                TestRun("test2"): 0.05,
                TestRun("test3"): 0.1,
            },
        )

        normalized = utils.normalize_ratings(ratings, 0.2, min_value=0.1)
        self.assertDictAlmostEqual(
            normalized,
            {
                TestRun("test1"): 0.125,
                TestRun("test2"): 0.15,
                TestRun("test3"): 0.2,
            },
        )


if __name__ == "__main__":
    unittest.main()
