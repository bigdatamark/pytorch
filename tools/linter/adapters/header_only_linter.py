#!/usr/bin/env python3
# [MODULE DOC] Lint adapter ensuring every symbol listed in the header-only API list file is
# exercised by at least one C++ or CUDA test source under the aoti_abi_check test directories,
# which are built without linking libtorch. Defines the test glob list and the repository
# root, and when run as a script checks the header-only API list and prints an error message
# per untested symbol as a JSON line.
"""
Checks that all symbols in torch/header_only_apis.txt are tested in a .cpp
test file to ensure header-only-ness. The .cpp test file must be built
without linking libtorch.
"""

import argparse
import json
import re
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "HEADER_ONLY_LINTER"


class LintSeverity(str, Enum):
    # [CLASS DOC] HELPER: String-valued enum of the severities lintrunner understands: error,
    # warning, advice and disabled.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # [CLASS DOC] HELPER: Record for one lint finding in lintrunner's JSON schema (path, line,
    # char, code, severity, name, original, replacement, description; None means unset). Instances
    # are converted to a mapping and printed as one JSON object per line.
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


CPP_TEST_GLOBS = [
    "test/cpp/aoti_abi_check/*.cpp",
    "test/cpp/aoti_abi_check/cuda/*.cu",
]

REPO_ROOT = Path(__file__).parents[3]


def find_matched_symbols(
    symbols_regex: re.Pattern[str], test_globs: list[str] = CPP_TEST_GLOBS
) -> set[str]:
    # [FUNCTION DOC] Scan the non-comment lines of all test files matching the globs (relative to
    # the repo root) and return the set of symbols that the given compiled regex matches. Lines
    # that are empty or begin with a double slash or hash sign are skipped. The regex is expected
    # to be built from symbols sorted in reverse order so that longer symbols sharing a prefix
    # match first. Reads files; no other side effects.
    """
    Goes through all lines not starting with // in the cpp files and
    accumulates a list of matches with the symbols_regex. Note that
    we expect symbols_regex to be sorted in reverse alphabetical
    order to allow superset regexes to get matched.
    """
    matched_symbols = set()
    # check noncommented out lines of the test files
    for cpp_test_glob in test_globs:
        for test_file in REPO_ROOT.glob(cpp_test_glob):
            with open(test_file) as tf:
                for test_file_line in tf:
                    test_file_line = test_file_line.strip()
                    if test_file_line.startswith(("//", "#")) or test_file_line == "":
                        continue
                    matches = re.findall(symbols_regex, test_file_line)
                    for m in matches:
                        if m != "":
                            matched_symbols.add(m)
    return matched_symbols


def check_file(
    filename: str, test_globs: list[str] = CPP_TEST_GLOBS
) -> list[LintMessage]:
    # [FUNCTION DOC] Parse the header-only API list file (one symbol per line, blank lines and
    # hash comments ignored, recording the last line number of each symbol), build an alternation
    # regex from the symbols in reverse sorted order, find which symbols appear in the tests, and
    # return one error LintMessage for every symbol that is not tested. The test globs argument
    # exists mainly so tests can override the default.
    # KNOWN ISSUE: the message tells users to edit tools/linters/adapters/header_only_linter.py,
    # but the actual directory is tools/linter/adapters; matching is plain substring/regex
    # matching, so one symbol that is a prefix of another may be matched spuriously and symbols
    # are not regex-escaped.
    """
    Goes through the header_only_apis.txt file and verifies that all symbols
    within the file can be found tested in an appropriately independent .cpp
    file.

    Note that we expect CPP_TEST_GLOBS to be passed in as test_globs--the
    only reason this is an argument at all is for ease of testing.
    """
    lint_messages: list[LintMessage] = []

    symbols: dict[str, int] = {}  # symbol -> lineno
    with open(filename) as f:
        for idx, line in enumerate(f):
            # commented out lines should be skipped
            symbol = line.strip()
            if not symbol or symbol[0] == "#":
                continue

            # symbols can in fact be duplicated and come from different headers.
            # we are aware this is a flaw in using simple string matching.
            symbols[symbol] = idx + 1

    # Why reverse the keys? To allow superset regexes to get matched first in
    # find_matched_symbols. For example, we want Float8_e5m2fnuz to match
    # before Float8_e5m2. Otherwise, both Float8_e5m2fnuz and Float8_e5m2 will
    # match Float8_e5m2
    symbols_regex = re.compile("|".join(sorted(symbols.keys(), reverse=True)))
    matched_symbols = find_matched_symbols(symbols_regex, test_globs)

    for s, lineno in symbols.items():
        if s not in matched_symbols:
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=lineno,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[untested-symbol]",
                    original=None,
                    replacement=None,
                    description=(
                        f"{s} has been included as a header-only API "
                        "but is not tested in any of CPP_TEST_GLOBS, which "
                        f"contains {CPP_TEST_GLOBS}.\n"
                        "Please add a .cpp test using the symbol without "
                        "linking anything to verify that the symbol is in "
                        "fact header-only. If you already have a test but it's"
                        " not found, please add the .cpp file to CPP_TEST_GLOBS"
                        " in tools/linters/adapters/header_only_linter.py."
                    ),
                )
            )

    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="header only APIs linter",
        fromfile_prefix_chars="@",
    )
    args = parser.parse_args()

    for lint_message in check_file(
        str(REPO_ROOT) + "/torch/header_only_apis.txt", CPP_TEST_GLOBS
    ):
        print(json.dumps(lint_message._asdict()), flush=True)
