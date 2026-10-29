# [MODULE DOC] Tests for the target-determination heuristics interface: TestPrioritizations
# (initialization, setting/adding scores at file and class granularity, validation, JSON
# round trip), AggregatedHeuristics (combining scores, per-test statistics) and TestRun JSON.
# Also defines the shared TestTD base class reused by other heuristic tests. Import-time side
# effect: temporarily appends the repository root to sys.path for imports, then removes it.
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.testing.target_determination.heuristics.interface as interface
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))


class TestTD(unittest.TestCase):
    # [CLASS DOC] Base test case for target-determination tests, offering an approximate
    # score-mapping assertion and a factory of dummy heuristic classes. Stateless.
    def assert_test_scores_almost_equal(
        self, d1: dict[TestRun, float], d2: dict[TestRun, float]
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts two mappings from test runs to scores have the
        # same set of keys and approximately equal values; failure messages name the key.
        # Check that dictionaries are the same, except for floating point errors
        self.assertEqual(set(d1.keys()), set(d2.keys()))
        for k, v in d1.items():
            self.assertAlmostEqual(v, d2[k], msg=f"{k}: {v} != {d2[k]}")

    def make_heuristic(self, classname: str) -> Any:
        # [FUNCTION DOC] HELPER: Creates and returns a new heuristic class with the given
        # name, derived from a dummy implementation of the heuristic interface.
        # Create a dummy heuristic class
        class Heuristic(interface.HeuristicInterface):
            # [CLASS DOC] HELPER: Dummy heuristic whose prediction is always empty.
            def get_prediction_confidence(
                self, tests: list[str]
            ) -> interface.TestPrioritizations:
                # [FUNCTION DOC] HELPER: Ignores the tests and returns an empty
                # prioritization with no scores.
                # Return junk
                return interface.TestPrioritizations([], {})

        return type(classname, (Heuristic,), {})


class TestTestPrioritizations(TestTD):
    # [CLASS DOC] Tests the TestPrioritizations score container: construction, set and add
    # score operations with class-level include/exclude splitting, and validation. Stateless.
    def test_init_none(self) -> None:
        # [FUNCTION DOC] Asserts construction with no scores records the original tests and
        # gives every test a default score of zero.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(tests, {})
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.0, TestRun("test_b"): 0.0},
        )

    def test_init_set_scores_full_files(self) -> None:
        # [FUNCTION DOC] Asserts scores given for whole test files at construction are kept.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a"): 0.5, TestRun("test_b"): 0.25}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.25},
        )

    def test_init_set_scores_some_full_files(self) -> None:
        # [FUNCTION DOC] Asserts a test file missing from the given scores defaults to zero.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a"): 0.5}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.0},
        )

    def test_init_set_scores_classes(self) -> None:
        # [FUNCTION DOC] Asserts a score given for an included class of a file adds a
        # complementary excluded-class entry for the same file with score zero.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a", included=["TestA"]): 0.5}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.5,
                TestRun("test_a", excluded=["TestA"]): 0.0,
                TestRun("test_b"): 0.0,
            },
        )

    def test_init_set_scores_other_class_naming_convention(self) -> None:
        # [FUNCTION DOC] Asserts the file-double-colon-class naming form is treated the
        # same as an explicit included class, with the same complementary excluded entry.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a::TestA"): 0.5}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.5,
                TestRun("test_a", excluded=["TestA"]): 0.0,
                TestRun("test_b"): 0.0,
            },
        )

    def test_set_test_score_full_class(self) -> None:
        # [FUNCTION DOC] Asserts set_test_score on a whole test file replaces its score.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(tests, {})
        test_prioritizations.set_test_score(TestRun("test_a"), 0.5)
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.0},
        )

    def test_set_test_score_mix(self) -> None:
        # [FUNCTION DOC] Applies a sequence of set_test_score calls mixing whole-file,
        # included-class and excluded-class runs, asserting after each step that the stored
        # runs are split or merged so classes never overlap and the remainder run keeps the
        # right score. Finally validates the container.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_b"): -0.5}
        )
        test_prioritizations.set_test_score(TestRun("test_a"), 0.1)
        test_prioritizations.set_test_score(TestRun("test_a::TestA"), 0.2)
        test_prioritizations.set_test_score(TestRun("test_a::TestB"), 0.3)
        test_prioritizations.set_test_score(TestRun("test_a", included=["TestC"]), 0.4)
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.2,
                TestRun("test_a", included=["TestB"]): 0.3,
                TestRun("test_a", included=["TestC"]): 0.4,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(
            TestRun("test_a", included=["TestA", "TestB"]), 0.5
        )
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", included=["TestC"]): 0.4,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(
            TestRun("test_a", excluded=["TestA", "TestB"]), 0.6
        )
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB"]): 0.6,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(TestRun("test_a", included=["TestC"]), 0.7)
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.6,
                TestRun("test_a", included=["TestC"]): 0.7,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(TestRun("test_a", excluded=["TestD"]), 0.8)
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", excluded=["TestD"]): 0.8,
                TestRun("test_a", included=["TestD"]): 0.6,
                TestRun("test_b"): -0.5,
            },
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        test_prioritizations.validate()

    def test_add_test_score_mix(self) -> None:
        # [FUNCTION DOC] Like the set-score mix test but for add_test_score: asserts that
        # added scores accumulate on every overlapping class run, with the runs split as
        # needed and values compared approximately. Finally validates the container.
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_b"): -0.5}
        )
        test_prioritizations.add_test_score(TestRun("test_a"), 0.1)
        test_prioritizations.add_test_score(TestRun("test_a::TestA"), 0.2)
        test_prioritizations.add_test_score(TestRun("test_a::TestB"), 0.3)
        test_prioritizations.add_test_score(TestRun("test_a", included=["TestC"]), 0.4)
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.3,
                TestRun("test_a", included=["TestB"]): 0.4,
                TestRun("test_a", included=["TestC"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(
            TestRun("test_a", included=["TestA", "TestB"]), 0.5
        )
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.8,
                TestRun("test_a", included=["TestB"]): 0.9,
                TestRun("test_a", included=["TestC"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(
            TestRun("test_a", excluded=["TestA", "TestB"]), 0.6
        )
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.8,
                TestRun("test_a", included=["TestB"]): 0.9,
                TestRun("test_a", included=["TestC"]): 1.1,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.7,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(TestRun("test_a", included=["TestC"]), 0.7)
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.8,
                TestRun("test_a", included=["TestB"]): 0.9,
                TestRun("test_a", included=["TestC"]): 1.8,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.7,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(TestRun("test_a", excluded=["TestD"]), 0.8)
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 1.6,
                TestRun("test_a", included=["TestB"]): 1.7,
                TestRun("test_a", included=["TestC"]): 2.6,
                TestRun("test_a", included=["TestD"]): 0.7,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC", "TestD"]): 1.5,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(
            TestRun("test_a", excluded=["TestD", "TestC"]), 0.1
        )
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 1.7,
                TestRun("test_a", included=["TestB"]): 1.8,
                TestRun("test_a", included=["TestC"]): 2.6,
                TestRun("test_a", included=["TestD"]): 0.7,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC", "TestD"]): 1.6,
                TestRun("test_b"): -0.5,
            },
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        test_prioritizations.validate()


class TestAggregatedHeuristics(TestTD):
    # [CLASS DOC] Tests how AggregatedHeuristics sums the scores of several heuristics.
    # Stateless; the check helper drives each scenario.
    def check(
        self,
        tests: list[str],
        test_prioritizations: list[dict[TestRun, float]],
        expected: dict[TestRun, float],
    ) -> None:
        # [FUNCTION DOC] HELPER: Registers one freshly named dummy heuristic per given score
        # mapping on an aggregator for the tests, then asserts the aggregated priorities
        # equal the expected mapping approximately.
        aggregated_heuristics = interface.AggregatedHeuristics(tests)
        for i, test_prioritization in enumerate(test_prioritizations):
            heuristic = self.make_heuristic(f"H{i}")
            aggregated_heuristics.add_heuristic_results(
                heuristic(), interface.TestPrioritizations(tests, test_prioritization)
            )
        final_prioritzations = aggregated_heuristics.get_aggregated_priorities()
        self.assert_test_scores_almost_equal(
            final_prioritzations._test_scores,
            expected,
        )

    def test_get_aggregated_priorities_mix_1(self) -> None:
        # [FUNCTION DOC] Aggregates a whole-file score, a class score on the same file and
        # a score on another file; asserts the class run sums both and others stay separate.
        tests = ["test_a", "test_b", "test_c"]
        self.check(
            tests,
            [
                {TestRun("test_a"): 0.5},
                {TestRun("test_a::TestA"): 0.25},
                {TestRun("test_c"): 0.8},
            ],
            {
                TestRun("test_a", excluded=["TestA"]): 0.5,
                TestRun("test_a", included=["TestA"]): 0.75,
                TestRun("test_b"): 0.0,
                TestRun("test_c"): 0.8,
            },
        )

    def test_get_aggregated_priorities_mix_2(self) -> None:
        # [FUNCTION DOC] Aggregates heuristics that score overlapping included and excluded
        # class sets across several files; asserts the resulting class runs carry the summed
        # scores.
        tests = ["test_a", "test_b", "test_c"]
        self.check(
            tests,
            [
                {
                    TestRun("test_a", included=["TestC"]): 0.5,
                    TestRun("test_b"): 0.25,
                    TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.8,
                },
                {
                    TestRun("test_a::TestA"): 0.25,
                    TestRun("test_b::TestB"): 0.5,
                    TestRun("test_a::TestB"): 0.75,
                    TestRun("test_a", excluded=["TestA", "TestB"]): 0.8,
                },
                {TestRun("test_c"): 0.8},
            ],
            {
                TestRun("test_a", included=["TestA"]): 0.25,
                TestRun("test_a", included=["TestB"]): 0.75,
                TestRun("test_a", included=["TestC"]): 1.3,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 1.6,
                TestRun("test_b", included=["TestB"]): 0.75,
                TestRun("test_b", excluded=["TestB"]): 0.25,
                TestRun("test_c"): 0.8,
            },
        )

    def test_get_aggregated_priorities_mix_3(self) -> None:
        # [FUNCTION DOC] Aggregates six heuristics with many overlapping class include and
        # exclude sets on one file; asserts the final per-class scores are the sums.
        tests = ["test_a"]
        self.check(
            tests,
            [
                {
                    TestRun("test_a", included=["TestA"]): 0.1,
                    TestRun("test_a", included=["TestC"]): 0.1,
                    TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                },
                {
                    TestRun("test_a", excluded=["TestD"]): 0.1,
                },
                {
                    TestRun("test_a", included=["TestC"]): 0.1,
                },
                {
                    TestRun("test_a", included=["TestB", "TestC"]): 0.1,
                },
                {
                    TestRun("test_a", included=["TestC"]): 0.1,
                    TestRun("test_a", included=["TestD"]): 0.1,
                },
                {
                    TestRun("test_a"): 0.1,
                },
            ],
            {
                TestRun("test_a", included=["TestA"]): 0.3,
                TestRun("test_a", included=["TestB"]): 0.3,
                TestRun("test_a", included=["TestC"]): 0.6,
                TestRun("test_a", included=["TestD"]): 0.3,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC", "TestD"]): 0.3,
            },
        )


class TestAggregatedHeuristicsTestStats(TestTD):
    # [CLASS DOC] Tests the per-test statistics report (positions and scores per heuristic
    # and aggregated) built by AggregatedHeuristics. Stateless.
    def test_get_test_stats_with_whole_tests(self) -> None:
        # [FUNCTION DOC] With two heuristics scoring whole files, asserts the stats of one
        # test list each heuristic's rank position and score plus the aggregated and
        # aggregated-trial entries, exactly as expected.
        self.maxDiff = None
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        expected_test3_stats = {
            "test_name": "test3",
            "test_filters": "",
            "heuristics": [
                {
                    "position": 0,
                    "score": 0.3,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 3,
                    "score": 0.0,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 1, "score": 0.3},
            "aggregated_trial": {"position": 1, "score": 0.3},
        }

        test3_stats = aggregator.get_test_stats(TestRun("test3"))

        self.assertDictEqual(test3_stats, expected_test3_stats)

    def test_get_test_stats_only_contains_allowed_types(self) -> None:
        # [FUNCTION DOC] Asserts the stats mappings for a whole test and a class run contain
        # only string keys and values of string, number, list or mapping type, recursively.
        self.maxDiff = None
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5::classA"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        stats3 = aggregator.get_test_stats(TestRun("test3"))
        stats5 = aggregator.get_test_stats(TestRun("test5::classA"))

        def assert_valid_dict(dict_contents: dict[str, Any]) -> None:
            # [FUNCTION DOC] HELPER: Recursively asserts that all keys are strings and all
            # values are strings, numbers, lists or mappings, descending into mappings and
            # into list items (which are treated as mappings).
            for key, value in dict_contents.items():
                self.assertTrue(isinstance(key, str))
                self.assertTrue(
                    isinstance(value, (str, float, int, list, dict)),
                    f"{value} is not a str, float, or dict",
                )
                if isinstance(value, dict):
                    assert_valid_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_valid_dict(item)

        assert_valid_dict(stats3)
        assert_valid_dict(stats5)

    def test_get_test_stats_gets_rank_for_test_classes(self) -> None:
        # [FUNCTION DOC] Asserts stats for an included-class run and the matching
        # excluded-class run report the correct filter strings, positions and scores for
        # each heuristic and in aggregate.
        self.maxDiff = None
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5::classA"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        stats_inclusive = aggregator.get_test_stats(
            TestRun("test5", included=["classA"])
        )
        stats_exclusive = aggregator.get_test_stats(
            TestRun("test5", excluded=["classA"])
        )
        expected_inclusive = {
            "test_name": "test5",
            "test_filters": "classA",
            "heuristics": [
                {
                    "position": 4,
                    "score": 0.0,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 0,
                    "score": 0.5,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 0, "score": 0.5},
            "aggregated_trial": {"position": 0, "score": 0.5},
        }
        expected_exclusive = {
            "test_name": "test5",
            "test_filters": "not (classA)",
            "heuristics": [
                {
                    "position": 4,
                    "score": 0.0,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 5,
                    "score": 0.0,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 5, "score": 0.0},
            "aggregated_trial": {"position": 5, "score": 0.0},
        }

        self.assertDictEqual(stats_inclusive, expected_inclusive)
        self.assertDictEqual(stats_exclusive, expected_exclusive)

    def test_get_test_stats_works_with_class_granularity_heuristics(self) -> None:
        # [FUNCTION DOC] Mixes a file-level and a class-level heuristic and asserts that
        # requesting stats for the class run and for the whole file raises no error.
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test2"): 0.3,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test2::TestFooClass"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        # These should not throw an error
        aggregator.get_test_stats(TestRun("test2::TestFooClass"))
        aggregator.get_test_stats(TestRun("test2"))


class TestJsonParsing(TestTD):
    # [CLASS DOC] Tests JSON serialization round trips of prioritizations and test runs.
    def test_json_parsing_matches_TestPrioritizations(self) -> None:
        # [FUNCTION DOC] Asserts converting a prioritization to JSON and back preserves the
        # original test set and the score mapping.
        tests = ["test1", "test2", "test3", "test4", "test5"]
        tp = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3", included=["ClassA"]): 0.8,
                TestRun("test3", excluded=["ClassA"]): 0.2,
                TestRun("test4"): 0.7,
                TestRun("test5"): 0.6,
            },
        )
        tp_json = tp.to_json()
        tp_json_to_tp = interface.TestPrioritizations.from_json(tp_json)

        self.assertSetEqual(tp._original_tests, tp_json_to_tp._original_tests)
        self.assertDictEqual(tp._test_scores, tp_json_to_tp._test_scores)

    def test_json_parsing_matches_TestRun(self) -> None:
        # [FUNCTION DOC] Asserts a test run with included classes survives a JSON round trip
        # as an equal object.
        testrun = TestRun("test1", included=["classA", "classB"])
        testrun_json = testrun.to_json()
        testrun_json_to_test = TestRun.from_json(testrun_json)

        self.assertTrue(testrun == testrun_json_to_test)


if __name__ == "__main__":
    unittest.main()
