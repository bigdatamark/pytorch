# [MODULE DOC] Core data types for target determination. TestPrioritizations holds a score
# for each TestRun over a fixed set of test files and keeps the runs disjoint and complete.
# AggregatedHeuristics stores each heuristic's TestPrioritizations and sums them (optionally
# including trial heuristics). HeuristicInterface is the base class of every heuristic. No
# import-time side effects.

from __future__ import annotations

from abc import abstractmethod
from copy import copy
from typing import Any, TYPE_CHECKING

from tools.testing.test_run import TestRun


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class TestPrioritizations:
    # [CLASS DOC] Maps TestRun objects to float scores for a fixed set of test files. Invariant
    # (checked by validate): the runs for a file never overlap and together cover the whole
    # file, and the covered files equal the original test set. Unlisted tests start at 0.
    # State: _original_tests (frozenset of names) and _test_scores (TestRun to score).
    # Ordering is by descending score then name so results are deterministic for sharding.
    """
    Describes the results of whether heuristics consider a test relevant or not.

    All the different ranks of tests are disjoint, meaning a test can only be in one category, and they are only
    declared at initialization time.

    A list can be empty if a heuristic doesn't consider any tests to be in that category.

    Important: Lists of tests must always be returned in a deterministic order,
               otherwise it breaks the test sharding logic
    """

    _original_tests: frozenset[str]
    _test_scores: dict[TestRun, float]

    def __init__(
        self,
        tests_being_ranked: Iterable[str],  # The tests that are being prioritized.
        scores: dict[TestRun, float],
    ) -> None:
        # [FUNCTION DOC] Initializes every test with score 0, applies the given scores (entries for
        # tests outside the set are ignored) and validates the invariant.
        self._original_tests = frozenset(tests_being_ranked)
        self._test_scores = {TestRun(test): 0.0 for test in self._original_tests}

        for test, score in scores.items():
            self.set_test_score(test, score)

        self.validate()

    def validate(self) -> None:
        # [FUNCTION DOC] Checks the invariant: unions the runs per test file, raising
        # AssertionError if two runs overlap, if a file is not fully covered, or if the covered
        # files differ from the original test set.
        # Union all TestRuns that contain include/exclude pairs
        all_tests = self._test_scores.keys()
        files = {}
        for test in all_tests:
            if test.test_file not in files:
                files[test.test_file] = copy(test)
            else:
                if not (files[test.test_file] & test).is_empty():
                    raise AssertionError(
                        f"Test run `{test}` overlaps with `{files[test.test_file]}`"
                    )
                files[test.test_file] |= test

        for test in files.values():
            if not test.is_full_file():
                raise AssertionError(
                    f"All includes should have been excluded elsewhere, and vice versa. "
                    f"Test run `{test}` violates that"
                )

        # Ensure that the set of tests in the TestPrioritizations is identical to the set of tests passed in
        if self._original_tests != set(files.keys()):
            raise AssertionError(
                "The set of tests in the TestPrioritizations must be identical "
                "to the set of tests passed in"
            )

    def _traverse_scores(self) -> Iterator[tuple[float, TestRun]]:
        # [FUNCTION DOC] HELPER: Yields (score, TestRun) pairs sorted by descending score, ties
        # broken by the string form of the run.
        # Sort by score, then alphabetically by test name
        for test, score in sorted(
            self._test_scores.items(), key=lambda x: (-x[1], str(x[0]))
        ):
            yield score, test

    def set_test_score(self, test_run: TestRun, new_score: float) -> None:
        # [FUNCTION DOC] Sets the score of a TestRun (ignored if its file is unknown). Existing
        # runs overlapping it are shrunk to the part not covered, keeping their old score, so the
        # disjointness invariant holds. Then validates.
        if test_run.test_file not in self._original_tests:
            return  # We don't need this test

        relevant_test_runs: list[TestRun] = [
            tr for tr in self._test_scores if tr & test_run and tr != test_run
        ]

        # Set the score of all the tests that are covered by test_run to the same score
        self._test_scores[test_run] = new_score
        # Set the score of all the tests that are not covered by test_run to original score
        for relevant_test_run in relevant_test_runs:
            old_score = self._test_scores[relevant_test_run]
            del self._test_scores[relevant_test_run]

            not_to_be_updated = relevant_test_run - test_run
            if not not_to_be_updated.is_empty():
                self._test_scores[not_to_be_updated] = old_score
        self.validate()

    def add_test_score(self, test_run: TestRun, score_to_add: float) -> None:
        # [FUNCTION DOC] Adds a value to the score of every part of existing runs overlapping the
        # given TestRun, splitting runs into the overlapping part (new score) and the remainder
        # (old score). Ignored if the file is unknown. Then validates.
        if test_run.test_file not in self._original_tests:
            return

        relevant_test_runs: list[TestRun] = [
            tr for tr in self._test_scores if tr & test_run
        ]

        for relevant_test_run in relevant_test_runs:
            old_score = self._test_scores[relevant_test_run]
            del self._test_scores[relevant_test_run]

            intersection = relevant_test_run & test_run
            if not intersection.is_empty():
                self._test_scores[intersection] = old_score + score_to_add

            not_to_be_updated = relevant_test_run - test_run
            if not not_to_be_updated.is_empty():
                self._test_scores[not_to_be_updated] = old_score

        self.validate()

    def get_all_tests(self) -> list[TestRun]:
        # [FUNCTION DOC] Returns all TestRuns ordered by descending score.
        """Returns all tests in the TestPrioritizations"""
        return [x[1] for x in self._traverse_scores()]

    def get_top_per_tests(self, n: int) -> tuple[list[TestRun], list[TestRun]]:
        # [FUNCTION DOC] Splits the ordered runs into a leading part and the rest. The leading part
        # has n percent of the runs (integer division) plus one extra run.
        # KNOWN ISSUE: the extra +1 makes the top group larger than the docstring suggests, and
        # the top group is non-empty even when n is 0.
        """Divides list of tests into two based on the top n% of scores.  The
        first list is the top, and the second is the rest."""
        tests = [x[1] for x in self._traverse_scores()]
        index = n * len(tests) // 100 + 1
        return tests[:index], tests[index:]

    def get_info_str(self, verbose: bool = True) -> str:
        # [FUNCTION DOC] Builds a text listing of runs with scores, one per line, in order. When
        # not verbose, zero-score runs are omitted. Returns the text without trailing whitespace.
        info = ""

        for score, test in self._traverse_scores():
            if not verbose and score == 0:
                continue
            info += f"  {test} ({score})\n"

        return info.rstrip()

    def print_info(self) -> None:
        # [FUNCTION DOC] Prints the verbose info string. Side effect: writes to stdout.
        print(self.get_info_str())

    def get_priority_info_for_test(self, test_run: TestRun) -> dict[str, Any]:
        # [FUNCTION DOC] Finds the first run overlapping the given run and returns a mapping with
        # its position and score. Raises AssertionError if no run overlaps.
        """Given a failing test, returns information about it's prioritization that we want to emit in our metrics."""
        for idx, (score, test) in enumerate(self._traverse_scores()):
            #  Different heuristics may result in a given test file being split
            #  into different test runs, so look for the overlapping tests to
            #  find the match
            if test & test_run:
                return {"position": idx, "score": score}
        raise AssertionError(f"Test run {test_run} not found")

    def get_test_stats(self, test: TestRun) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a metrics mapping for a test: file name, pytest filter, position
        # and score of the run, maximum and minimum score, and all scores keyed by run string.
        return {
            "test_name": test.test_file,
            "test_filters": test.get_pytest_filter(),
            **self.get_priority_info_for_test(test),
            "max_score": max(score for score, _ in self._traverse_scores()),
            "min_score": min(score for score, _ in self._traverse_scores()),
            "all_scores": {
                str(test): score for test, score in self._test_scores.items()
            },
        }

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC] Serializes to a mapping with the nonzero (run, score) pairs and the list
        # of original test names. Zero-score entries are dropped.
        """
        Returns a JSON dict that describes this TestPrioritizations object.
        """
        json_dict = {
            "_test_scores": [
                (test.to_json(), score)
                for test, score in self._test_scores.items()
                if score != 0
            ],
            "_original_tests": list(self._original_tests),
        }
        return json_dict

    @staticmethod
    def from_json(json_dict: dict[str, Any]) -> TestPrioritizations:
        # [FUNCTION DOC] Rebuilds a TestPrioritizations from the mapping produced by to_json.
        """
        Returns a TestPrioritizations object from a JSON dict.
        """
        test_prioritizations = TestPrioritizations(
            tests_being_ranked=json_dict["_original_tests"],
            scores={
                TestRun.from_json(testrun_json): score
                for testrun_json, score in json_dict["_test_scores"]
            },
        )
        return test_prioritizations

    def amend_tests(self, tests: list[str]) -> None:
        # [FUNCTION DOC] Replaces the test set: drops runs whose file is not in the given list,
        # adds score-0 entries for new files, and updates the original test set. Then validates.
        """
        Removes tests that are not in the given list from the
        TestPrioritizations.  Adds tests that are in the list but not in the
        TestPrioritizations.
        """
        valid_scores = {
            test: score
            for test, score in self._test_scores.items()
            if test.test_file in tests
        }
        self._test_scores = valid_scores

        for test in tests:
            if test not in self._original_tests:
                self._test_scores[TestRun(test)] = 0
        self._original_tests = frozenset(tests)

        self.validate()


class AggregatedHeuristics:
    # [CLASS DOC] Collects one TestPrioritizations per heuristic over a common test set and
    # exposes combined views. _heuristic_results keeps insertion order (important for
    # sharding); _all_tests is the shared test set. Every stored result must cover exactly it.
    """
    Aggregates the results across all heuristics.

    It saves the individual results from each heuristic and exposes an aggregated view.
    """

    _heuristic_results: dict[
        HeuristicInterface, TestPrioritizations
    ]  # Key is the Heuristic's name. Dicts will preserve the order of insertion, which is important for sharding

    _all_tests: frozenset[str]

    def __init__(self, all_tests: list[str]) -> None:
        # [FUNCTION DOC] Stores the common test set, starts with no heuristic results, validates.
        self._all_tests = frozenset(all_tests)
        self._heuristic_results = {}
        self.validate()

    def validate(self) -> None:
        # [FUNCTION DOC] Validates each stored result and raises AssertionError if its test set
        # differs from the common test set.
        for heuristic, heuristic_results in self._heuristic_results.items():
            heuristic_results.validate()
            if heuristic_results._original_tests != self._all_tests:
                raise AssertionError(
                    f"Tests in {heuristic.name} are not the same as the tests in "
                    "the AggregatedHeuristics"
                )

    def add_heuristic_results(
        self, heuristic: HeuristicInterface, heuristic_results: TestPrioritizations
    ) -> None:
        # [FUNCTION DOC] Records a heuristic's results. Raises ValueError if the same heuristic
        # object was already added. Validates afterwards.
        if heuristic in self._heuristic_results:
            raise ValueError(f"We already have heuristics for {heuristic.name}")

        self._heuristic_results[heuristic] = heuristic_results
        self.validate()

    def get_aggregated_priorities(
        self, include_trial: bool = False
    ) -> TestPrioritizations:
        # [FUNCTION DOC] Sums the scores of all heuristics into a new TestPrioritizations. Trial
        # mode heuristics are skipped unless include_trial is true. Returns the combined result.
        """
        Returns the aggregated priorities across all heuristics.
        """
        valid_heuristics = {
            heuristic: heuristic_results
            for heuristic, heuristic_results in self._heuristic_results.items()
            if not heuristic.trial_mode or include_trial
        }

        new_tp = TestPrioritizations(self._all_tests, {})

        for heuristic_results in valid_heuristics.values():
            for score, testrun in heuristic_results._traverse_scores():
                new_tp.add_test_score(testrun, score)
        new_tp.validate()
        return new_tp

    def get_test_stats(self, test: TestRun) -> dict[str, Any]:
        # [FUNCTION DOC] Returns metrics for a test: name, filter, per-heuristic position and score
        # (with heuristic name and trial flag), plus the aggregated and aggregated-with-trial
        # positions and scores.
        """
        Returns the aggregated statistics for a given test.
        """
        stats: dict[str, Any] = {
            "test_name": test.test_file,
            "test_filters": test.get_pytest_filter(),
        }

        # Get metrics about the heuristics used
        heuristics = []

        for heuristic, heuristic_results in self._heuristic_results.items():
            metrics = heuristic_results.get_priority_info_for_test(test)
            metrics["heuristic_name"] = heuristic.name
            metrics["trial_mode"] = heuristic.trial_mode
            heuristics.append(metrics)

        stats["heuristics"] = heuristics

        stats["aggregated"] = (
            self.get_aggregated_priorities().get_priority_info_for_test(test)
        )

        stats["aggregated_trial"] = self.get_aggregated_priorities(
            include_trial=True
        ).get_priority_info_for_test(test)

        return stats

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC] Serializes to a mapping from heuristic name to its serialized results.
        """
        Returns a JSON dict that describes this AggregatedHeuristics object.
        """
        json_dict: dict[str, Any] = {}
        for heuristic, heuristic_results in self._heuristic_results.items():
            json_dict[heuristic.name] = heuristic_results.to_json()

        return json_dict


class HeuristicInterface:
    # [CLASS DOC] Base class for all heuristics. Subclasses implement get_prediction_confidence
    # and call this constructor. Attributes: description and trial_mode (trial heuristics are
    # only reported in metrics and not used to order tests).
    # KNOWN ISSUE: the class does not use an abstract base metaclass, so the abstractmethod
    # markers are not enforced.
    """
    Interface for all heuristics.
    """

    description: str

    # When trial mode is set to True, this heuristic's predictions will not be used
    # to reorder tests. It's results will however be emitted in the metrics.
    trial_mode: bool

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Sets trial_mode from the keyword arguments, defaulting to False.
        self.trial_mode = kwargs.get("trial_mode", False)  # type: ignore[assignment]

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Returns the heuristic name, which is the class name.
        return self.__class__.__name__

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the heuristic name.
        return self.name

    @abstractmethod
    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Abstract: given the list of test names, returns a TestPrioritizations over
        # them. Subclasses override it.
        # KNOWN ISSUE: the docstring describes a float rating from -1 to 1 and refers to
        # AggregatedHeuristicsRankings, but the method returns TestPrioritizations.
        """
        Returns a float ranking ranging from -1 to 1, where negative means skip,
        positive means run, 0 means no idea, and magnitude = how confident the
        heuristic is. Used by AggregatedHeuristicsRankings.
        """
