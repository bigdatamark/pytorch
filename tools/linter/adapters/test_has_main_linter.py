# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "libcst",
# ]
# ///
# [MODULE DOC]
# Lintrunner adapter (uses libcst) that verifies each test file has a main guard block that
# calls run_tests or run_rank or raises an exception, so the tests run in OSS CI.
# Main contents: HasMainVisiter, LintSeverity/LintMessage types, check_file and main (parallel
# over files with 8 worker processes, printing JSON lines). No import-time side effects.
"""
This lint verifies that every Python test file (file that matches test_*.py or
*_test.py in the test folder) has a main block which raises an exception or
calls run_tests to ensure that the test will be run in OSS CI.

Takes ~2 minuters to run without the multiprocessing, probably overkill.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from enum import Enum
from typing import NamedTuple

import libcst as cst
import libcst.matchers as m


LINTER_CODE = "TEST_HAS_MAIN"


class HasMainVisiter(cst.CSTVisitor):
    # [CLASS DOC] CST visitor that sets the found flag when a module has a main guard (name
    # compared for equality with the main string in either order) whose body contains a call
    # to run_tests or run_rank (plain or as attribute) or a raise statement.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the visitor with found set to false.
        super().__init__()
        self.found = False

    def visit_Module(self, node: cst.Module) -> bool:
        # [FUNCTION DOC] Inspects only the direct children of the module for a matching main
        # guard conditional and sets found on the first match. Returns False so the visitor
        # does not descend further.
        name = m.Name("__name__")
        main = m.SimpleString('"__main__"') | m.SimpleString("'__main__'")
        run_test_call = m.Call(
            func=m.Name("run_tests") | m.Attribute(attr=m.Name("run_tests"))
        )
        # Distributed tests (i.e. MultiProcContinuousTest) calls `run_rank`
        # instead of `run_tests` in main
        run_rank_call = m.Call(
            func=m.Name("run_rank") | m.Attribute(attr=m.Name("run_rank"))
        )
        raise_block = m.Raise()

        # name == main or main == name
        if_main1 = m.Comparison(
            name,
            [m.ComparisonTarget(m.Equal(), main)],
        )
        if_main2 = m.Comparison(
            main,
            [m.ComparisonTarget(m.Equal(), name)],
        )
        for child in node.children:
            if m.matches(child, m.If(test=if_main1 | if_main2)):
                if m.findall(child, raise_block | run_test_call | run_rank_call):
                    self.found = True
                    break

        return False


class LintSeverity(str, Enum):
    # [CLASS DOC] HELPER: String-valued enumeration of lint severities (error, warning,
    # advice, disabled) serialized into the JSON output consumed by lintrunner.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # [CLASS DOC] HELPER: Immutable record of one lint finding in the lintrunner JSON format:
    # path, line and char (each optional), linter code, severity, short name, optional
    # original and replacement text (for autofix), and an optional description.
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Parses the file with libcst and runs the main-guard visitor. Returns a
    # single error message named no-main (without a line number) if no suitable main block was
    # found, else an empty list. Raises OSError if unreadable and a libcst parse error if the
    # source is invalid.
    lint_messages = []

    with open(filename) as f:
        file = f.read()
        v = HasMainVisiter()
        cst.parse_module(file).visit(v)
        if not v.found:
            message = (
                "Test files need to have a main block which either calls run_tests "
                + "(to ensure that the tests are run during OSS CI) or raises an exception "
                + "and added to the blocklist in test/run_test.py"
            )
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[no-main]",
                    original=None,
                    replacement=None,
                    description=message,
                )
            )
    return lint_messages


def main() -> None:
    # [FUNCTION DOC] Command line entry point. Parses the list of file paths, checks them in a
    # pool of 8 processes, flattens the results and prints each message as a JSON line on
    # standard output.
    parser = argparse.ArgumentParser(
        description="test files should have main block linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    pool = mp.Pool(8)
    lint_messages = pool.map(check_file, args.filenames)
    pool.close()
    pool.join()

    flat_lint_messages = []
    for sublist in lint_messages:
        flat_lint_messages.extend(sublist)

    for lint_message in flat_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
