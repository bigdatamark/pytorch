# [MODULE DOC] Heuristic that matches keywords drawn from changed file paths (folder names
# and underscore-separated parts of the file name) against test names. Contains synonym and
# custom-matcher tables, keyword helpers, and the Filepath heuristic. If the PR only
# changes documentation files, every test gets a negative score so all are skipped. Running
# the file as a script prints keyword hit counts for a commit given on the command line.
# KNOWN ISSUE: REPO_ROOT is taken as the fourth parent of this file, which is the tools
# directory, not the repository root; it is unused in this module.

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    is_docs_only_change,
    normalize_ratings,
    query_changed_files,
)
from tools.testing.test_run import TestRun


if TYPE_CHECKING:
    from collections.abc import Callable


REPO_ROOT = Path(__file__).parents[3]

keyword_synonyms: dict[str, list[str]] = {
    "amp": ["mixed_precision"],
    "quant": ["quantized", "quantization", "quantize"],
    "decomp": ["decomposition", "decompositions"],
    "numpy": ["torch_np", "numpy_tests"],
    "ops": ["opinfo"],
    "hop": ["higher_order_op"],
    "aot": ["flex_attention", "autograd"],
    "inductor": ["dynamo", "export"],  # not actually synonyms but they interact a lot
}


custom_matchers: dict[str, Callable[[str], bool]] = {
    "nn": lambda x: "nn" in x.replace("onnx", "_"),
    "c10": lambda x: "c10" in x.replace("c10d", "_"),
}


def is_valid_keyword(keyword: str) -> bool:
    # [FUNCTION DOC] Returns True if the keyword is useful: the keyword nn is always valid,
    # otherwise it must be longer than two characters and not in a list of generic words
    # (torch, test, utils, src, tools, and similar).
    not_keyword = [
        "torch",
        "test",
        "tests",
        "util",
        "utils",
        "func",
        "src",
        "c",
        "ns",
        "tools",
        "internal",
    ]
    return keyword == "nn" or (keyword not in not_keyword and len(keyword) > 2)


@lru_cache(maxsize=1)
def get_keywords(file: str) -> list[str]:
    # [FUNCTION DOC] Returns the valid keywords for a path: each parent folder name and each
    # underscore-separated part of the file stem, normalized by sanitize_name and filtered by
    # is_valid_keyword. Results are cached.
    # KNOWN ISSUE: the cache holds a single entry, but callers alternate between many files, so
    # it is almost never hit.
    keywords = []
    for folder in Path(file).parts[:-1]:
        folder = sanitize_name(folder)
        keywords.append(folder)

    file_name = Path(file).stem.split("_")
    keywords.extend([sanitize_name(x) for x in file_name])
    return [kw for kw in keywords if is_valid_keyword(kw)]


def sanitize_name(folder_name: str) -> str:
    # [FUNCTION DOC] Strips one leading underscore and maps any known synonym (or the
    # canonical name itself) to its canonical name from keyword_synonyms; otherwise returns
    # the stripped name.
    folder_name = folder_name.removeprefix("_")

    for syn_rep, syns in keyword_synonyms.items():
        if folder_name in syns or folder_name == syn_rep:
            return syn_rep

    return folder_name


def file_matches_keyword(file: str, keyword: str) -> bool:
    # [FUNCTION DOC] Returns True if the keyword matches the file: it is one of the file's
    # keywords, a synonym of it appears in the keywords or in the path, or the custom matcher
    # (or plain substring test when no custom matcher exists) accepts the path.
    keywords = get_keywords(file)
    return (
        keyword in keywords
        or any(
            syn in keywords or syn in file for syn in keyword_synonyms.get(keyword, [])
        )
        or custom_matchers.get(keyword, lambda x: keyword in x)(file)  # type: ignore[no-untyped-call]
    )


def get_freq_dict(tests: list[str], changed_files: list[str]) -> dict[str, int]:
    # [FUNCTION DOC] Counts how often each keyword appears across the changed files, then for
    # every test adds the keyword's count each time the keyword matches that test. Returns a
    # mapping from test name to integer score; tests with no match are absent.
    keyword_frequency: dict[str, int] = defaultdict(int)
    for cf in changed_files:
        keywords = get_keywords(cf)
        for keyword in keywords:
            keyword_frequency[keyword] += 1

    test_ratings: dict[str, int] = defaultdict(int)

    for test in tests:
        for keyword, frequency in keyword_frequency.items():
            if file_matches_keyword(test, keyword):
                test_ratings[test] += frequency
    return test_ratings


class Filepath(HeuristicInterface):
    # [CLASS DOC] Heuristic rating tests by overlap between changed file path keywords and
    # test names, or skipping all tests for documentation-only changes.
    # Heuristic based on folders in the file path.  Takes each folder of each
    # changed file and attempts to find matches based on those folders
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Queries the changed files. If all are documentation files, prints a
        # message and gives every test score -1 (skip). Otherwise scores tests by keyword
        # frequency, rescaled so the best is 0.25 and a zero rating would map to 0.125 (every
        # rated test lies between 0.125 and 0.25). Returns a TestPrioritizations.
        changed_files = query_changed_files()

        # If only documentation files (.rst, .md) were modified, skip all tests
        if is_docs_only_change(changed_files):
            print("Only documentation files changed, skipping all tests")
            # Return negative scores to indicate all tests should be skipped
            skip_scores = {TestRun(test): -1.0 for test in tests}
            return TestPrioritizations(tests, skip_scores)

        test_ratings = get_freq_dict(tests, changed_files)
        test_ratings = {
            TestRun(k): float(v) for (k, v) in test_ratings.items() if k in tests
        }
        return TestPrioritizations(
            tests, normalize_ratings(test_ratings, 0.25, min_value=0.125)
        )


if __name__ == "__main__":
    # Quick thing so you can call the heuristic from the command line with a sha
    import os
    import sys

    from tools.testing.discover_tests import TESTS

    git_diff = f"git diff --name-only {sys.argv[1]} {sys.argv[1]}^"
    changed_files = os.popen(git_diff).read().split("\n")
    freq_dict = get_freq_dict(
        TESTS, [x for x in changed_files if x != "" and not x.startswith("test")]
    )
    for k, v in sorted(freq_dict.items(), key=lambda x: x[1], reverse=False):
        print(k, v)
    print(changed_files)
