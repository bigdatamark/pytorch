# For testing specific heuristics
# [MODULE DOC] Tests for individual target-determination heuristics: historical class
# failure correlation, edited-by-PR, merge base and changed-file queries, previously
# failed tests cache parsing, and filepath keyword matching. Uses mocks for git, GitHub
# and file access. Import-time side effect: temporarily appends the repository root to
# sys.path for imports, then removes it; defines module constants for patch targets.
from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from tools.test.heuristics.test_interface import TestTD
from tools.testing.target_determination.heuristics.edited_by_pr import (
    _get_modified_tests,
)
from tools.testing.target_determination.heuristics.filepath import (
    file_matches_keyword,
    get_keywords,
)
from tools.testing.target_determination.heuristics.historical_class_failure_correlation import (
    HistoricalClassFailurCorrelation,
)
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    get_previous_failures,
)
from tools.testing.target_determination.heuristics.utils import (
    _get_pr_merge_base,
    python_test_file_to_test_name,
    query_changed_files,
)
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

HEURISTIC_CLASS = "tools.testing.target_determination.heuristics.historical_class_failure_correlation."
EDITED_BY_PR = "tools.testing.target_determination.heuristics.edited_by_pr."
HEURISTIC_UTILS = "tools.testing.target_determination.heuristics.utils."


def mocked_file(contents: dict[Any, Any]) -> io.IOBase:
    # [FUNCTION DOC] HELPER: Returns an in-memory text stream holding the JSON encoding of
    # the given mapping, rewound to the start, to stand in for an opened file.
    file_object = io.StringIO()
    json.dump(contents, file_object)
    file_object.seek(0)
    return file_object


def gen_historical_class_failures() -> dict[str, dict[str, float]]:
    # [FUNCTION DOC] HELPER: Returns fixture data mapping three file names to mappings of
    # "test::class" identifiers to correlation scores.
    return {
        "file1": {
            "test1::classA": 0.5,
            "test2::classA": 0.2,
            "test5::classB": 0.1,
        },
        "file2": {
            "test1::classB": 0.3,
            "test3::classA": 0.2,
            "test5::classA": 1.5,
            "test7::classC": 0.1,
        },
        "file3": {
            "test1::classC": 0.4,
            "test4::classA": 0.2,
            "test7::classC": 1.5,
            "test8::classC": 0.1,
        },
    }


ALL_TESTS = [
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
    "test6",
    "test7",
    "test8",
]


class TestHistoricalClassFailureCorrelation(TestTD):
    # [CLASS DOC] Tests the historical class failure correlation heuristic with patched
    # correlation data and changed-file list. Stateless.
    @mock.patch(
        HEURISTIC_CLASS + "_get_historical_test_class_correlations",
        return_value=gen_historical_class_failures(),
    )
    @mock.patch(
        HEURISTIC_CLASS + "query_changed_files",
        return_value=["file1"],
    )
    def test_get_prediction_confidence(
        self,
        historical_class_failures: dict[str, dict[str, float]],
        changed_files: list[str],
    ) -> None:
        # [FUNCTION DOC] With patched correlations and one changed file, runs the heuristic
        # over all test names and asserts the resulting scores: class-level runs related
        # to the changed file get scaled correlation values, the remaining runs (with those
        # classes excluded) get zero. Note the mock arguments are bound in reverse order
        # of patching and are unused; the patches only take effect.
        tests_to_prioritize = ALL_TESTS

        heuristic = HistoricalClassFailurCorrelation()
        test_prioritizations = heuristic.get_prediction_confidence(tests_to_prioritize)

        expected = TestPrioritizations(
            tests_to_prioritize,
            {
                TestRun("test1::classA"): 0.25,
                TestRun("test2::classA"): 0.1,
                TestRun("test5::classB"): 0.05,
                TestRun("test1", excluded=["classA"]): 0.0,
                TestRun("test2", excluded=["classA"]): 0.0,
                TestRun("test3"): 0.0,
                TestRun("test4"): 0.0,
                TestRun("test5", excluded=["classB"]): 0.0,
                TestRun("test6"): 0.0,
                TestRun("test7"): 0.0,
                TestRun("test8"): 0.0,
            },
        )

        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores, expected._test_scores
        )


class TestChangedFiles(TestTD):
    # [CLASS DOC] Tests changed-file and merge-base lookups, with git and GitHub calls
    # patched. Stateless.
    @mock.patch(EDITED_BY_PR + "query_changed_files", side_effect=RuntimeError)
    def test_edited_by_pr_propagates_changed_files_errors(
        self,
        mock_query_changed_files: Any,
    ) -> None:
        # [FUNCTION DOC] Asserts that an error raised by the changed-files query is
        # propagated by the modified-tests lookup, and that the query was called once.
        with self.assertRaises(RuntimeError):
            _get_modified_tests()

        mock_query_changed_files.assert_called_once_with()

    @mock.patch(
        HEURISTIC_UTILS + "_git_merge_base",
        side_effect=[
            subprocess.CalledProcessError(128, ["git", "merge-base"]),
            subprocess.CalledProcessError(128, ["git", "merge-base"]),
        ],
    )
    @mock.patch(HEURISTIC_UTILS + "_git_head", return_value="head-sha")
    @mock.patch(
        HEURISTIC_UTILS + "_github_api_json",
        return_value={"merge_base_commit": {"sha": "merge-base-sha"}},
    )
    def test_pr_merge_base_falls_back_to_github_compare(
        self,
        mock_github_api_json: Any,
        mock_git_head: Any,
        mock_git_merge_base: Any,
    ) -> None:
        # [FUNCTION DOC] With both local merge-base attempts (remote base branch, then base
        # sha) failing, asserts the PR merge base is obtained from the GitHub compare API
        # using the url-quoted base ref and head sha, and that each helper was called as
        # expected.
        pr_info: dict[str, object] = {
            "base": {
                "ref": "gh/bobrenjc93/893/base",
                "sha": "base-sha",
            }
        }

        self.assertEqual(_get_pr_merge_base(pr_info), "merge-base-sha")

        mock_git_merge_base.assert_has_calls(
            [
                mock.call("origin/gh/bobrenjc93/893/base"),
                mock.call("base-sha"),
            ]
        )
        mock_git_head.assert_called_once_with()
        mock_github_api_json.assert_called_once_with(
            "compare/gh%2Fbobrenjc93%2F893%2Fbase...head-sha"
        )

    @mock.patch(HEURISTIC_UTILS + "get_pr_number", return_value=123)
    @mock.patch(HEURISTIC_UTILS + "get_merge_base", side_effect=RuntimeError)
    @mock.patch(
        HEURISTIC_UTILS + "_query_changed_files_from_github",
        return_value=["test/functorch/test_aotdispatch.py"],
    )
    def test_query_changed_files_falls_back_to_github_pr_files(
        self,
        mock_query_changed_files_from_github: Any,
        mock_get_merge_base: Any,
        mock_get_pr_number: Any,
    ) -> None:
        # [FUNCTION DOC] With the local merge-base lookup raising, asserts the changed-file
        # query falls back to the GitHub PR files listing for the PR number and returns
        # its result.
        self.assertEqual(
            query_changed_files(),
            ["test/functorch/test_aotdispatch.py"],
        )

        mock_get_pr_number.assert_called_once_with()
        mock_get_merge_base.assert_called_once_with()
        mock_query_changed_files_from_github.assert_called_once_with(123)

    def test_python_test_file_to_test_name_handles_git_paths(self) -> None:
        # [FUNCTION DOC] Asserts conversion of file paths to test names keeps only python
        # files under the test directory (accepting forward and backward slash separators)
        # and strips the extension.
        self.assertEqual(
            python_test_file_to_test_name(
                {
                    "test/test_jit.py",
                    "test\\test_nn.py",
                    "torch/test_not_a_test.py",
                    "test/test_not_python.txt",
                }
            ),
            {"test_jit", "test_nn"},
        )


class TestParsePrevTests(TestTD):
    # [CLASS DOC] Tests parsing of the pytest last-failed cache by get_previous_failures,
    # with file existence and opening patched. Class attribute
    # lastfailed_with_multiple_tests_per_file is fixture data.
    @mock.patch("os.path.exists", return_value=False)
    def test_cache_does_not_exist(self, mock_exists: Any) -> None:
        # [FUNCTION DOC] Asserts that with no cache file present the previous failures
        # result is an empty set.
        expected_failing_test_files: set[str] = set()

        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)

    @mock.patch("os.path.exists", return_value=True)
    @mock.patch("builtins.open", return_value=mocked_file({"": True}))
    def test_empty_cache(self, mock_exists: Any, mock_open: Any) -> None:
        # [FUNCTION DOC] With an existing cache whose only key is an empty string, asserts
        # the result is an empty set and the file was opened.
        expected_failing_test_files: set[str] = set()

        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)
        mock_open.assert_called()

    lastfailed_with_multiple_tests_per_file = {
        "test/test_car.py::TestCar::test_num[17]": True,
        "test/test_car.py::TestBar::test_num[25]": True,
        "test/test_far.py::TestFar::test_fun_copy[17]": True,
        "test/test_bar.py::TestBar::test_fun_copy[25]": True,
    }

    @mock.patch("os.path.exists", return_value=True)
    @mock.patch(
        "builtins.open",
        return_value=mocked_file(lastfailed_with_multiple_tests_per_file),
    )
    def test_dedupes_failing_test_files(self, mock_exists: Any, mock_open: Any) -> None:
        # [FUNCTION DOC] With a cache listing several failing tests across files, asserts
        # the result is the set of distinct test file names without directory or extension.
        expected_failing_test_files = {"test_car", "test_bar", "test_far"}
        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)


class TestFilePath(TestTD):
    # [CLASS DOC] Tests the filepath heuristic's keyword extraction and matching
    # functions. Stateless.
    def test_get_keywords(self) -> None:
        # [FUNCTION DOC] Asserts keyword extraction from paths: strips directories noise
        # and the test prefix, and maps related names (e.g. mixed precision) to shared
        # keywords such as amp.
        self.assertEqual(get_keywords("test/test_car.py"), ["car"])
        self.assertEqual(get_keywords("test/nn/test_amp.py"), ["nn", "amp"])
        self.assertEqual(get_keywords("torch/nn/test_amp.py"), ["nn", "amp"])
        self.assertEqual(
            get_keywords("torch/nn/mixed_precision/test_something.py"),
            ["nn", "amp", "something"],
        )

    def test_match_keywords(self) -> None:
        # [FUNCTION DOC] Asserts file_matches_keyword is true when the keyword (or a known
        # alias such as quant for quantization) appears in the test path and false for
        # unrelated pairs like onnx versus nn.
        self.assertTrue(file_matches_keyword("test/quantization/test_car.py", "quant"))
        self.assertTrue(file_matches_keyword("test/test_quantization.py", "quant"))
        self.assertTrue(file_matches_keyword("test/nn/test_amp.py", "nn"))
        self.assertTrue(file_matches_keyword("test/nn/test_amp.py", "amp"))
        self.assertTrue(file_matches_keyword("test/test_onnx.py", "onnx"))
        self.assertFalse(file_matches_keyword("test/test_onnx.py", "nn"))

    def test_get_keywords_match(self) -> None:
        # [FUNCTION DOC] Asserts end-to-end relatedness of a changed file to a test file:
        # true when any keyword of the changed file matches the test path, false otherwise.
        def helper(test_file: str, changed_file: str) -> bool:
            # [FUNCTION DOC] HELPER: Returns true if any keyword extracted from the changed
            # file matches the test file path.
            return any(
                file_matches_keyword(test_file, x) for x in get_keywords(changed_file)
            )

        self.assertTrue(helper("test/quantization/test_car.py", "quantize/t.py"))
        self.assertFalse(helper("test/onnx/test_car.py", "nn/t.py"))
        self.assertTrue(helper("test/nn/test_car.py", "nn/t.py"))
        self.assertFalse(helper("test/nn/test_car.py", "test/b.py"))
        self.assertTrue(helper("test/test_mixed_precision.py", "torch/amp/t.py"))
        self.assertTrue(helper("test/test_amp.py", "torch/mixed_precision/t.py"))
        self.assertTrue(helper("test/idk/other/random.py", "torch/idk/t.py"))


if __name__ == "__main__":
    unittest.main()
