#!/usr/bin/env python3

# [MODULE DOC] Command-line script that reads JUnit/xunit XML test reports (a single file or a
# directory tree) and prints the failed and errored tests to the console using rich for colour,
# followed by pass/skip/fail counts.
# Import-time side effects: raises ImportError if junitparser is missing; prints a hint (but
# continues) if rich is missing.
# KNOWN ISSUE: if rich is missing, later calls to rich.print fail with a NameError because the
# name is never bound.
from __future__ import annotations

import argparse
import os
from typing import Any


try:
    from junitparser import (  # type: ignore[import]
        Error,
        Failure,
        JUnitXml,
        TestCase,
        TestSuite,
    )
except ImportError as e:
    raise ImportError(
        "junitparser not found, please install with 'pip install junitparser'"
    ) from e

try:
    import rich
except ImportError:
    print("rich not found, for color output use 'pip install rich'")


def parse_junit_reports(path_to_reports: str) -> list[TestCase]:  # type: ignore[no-any-unimported]
    # [FUNCTION DOC] Collect test cases from a report path. Args: path_to_reports, a file or
    # directory. A file is parsed directly; a directory is walked recursively and every file
    # whose name ends in xml is parsed and concatenated. Returns a list of test cases (empty
    # if the path is neither a file nor a directory). Raises FileNotFoundError if the path
    # does not exist. Unreadable files only produce a warning.
    def parse_file(path: str) -> list[TestCase]:  # type: ignore[no-any-unimported]
        # [FUNCTION DOC] HELPER: Parse one XML report file into a flat list of test
        # cases. Any exception while reading or parsing is caught; a warning is printed
        # via rich and an empty list is returned.
        try:
            return convert_junit_to_testcases(JUnitXml.fromfile(path))
        except Exception as err:
            rich.print(
                f":Warning: [yellow]Warning[/yellow]: Failed to read {path}: {err}"
            )
            return []

    if not os.path.exists(path_to_reports):
        raise FileNotFoundError(f"Path '{path_to_reports}', not found")
    # Return early if the path provided is just a file
    if os.path.isfile(path_to_reports):
        return parse_file(path_to_reports)
    ret_xml = []
    if os.path.isdir(path_to_reports):
        for root, _, files in os.walk(path_to_reports):
            for fname in [f for f in files if f.endswith("xml")]:
                ret_xml += parse_file(os.path.join(root, fname))
    return ret_xml


def convert_junit_to_testcases(xml: JUnitXml | TestSuite) -> list[TestCase]:  # type: ignore[no-any-unimported]
    # [FUNCTION DOC] Flatten a parsed JUnit document into test cases. Recurses into nested
    # test suites and returns the leaf test case items in encounter order. Args: xml, a
    # JUnitXml or TestSuite. No side effects.
    testcases = []
    for item in xml:
        if isinstance(item, TestSuite):
            testcases.extend(convert_junit_to_testcases(item))
        else:
            testcases.append(item)
    return testcases


def render_tests(testcases: list[TestCase]) -> None:  # type: ignore[no-any-unimported]
    # [FUNCTION DOC] Print every errored or failed test as '<classname>.<name>' (via rich)
    # followed by its raw result text, then print totals of passed, skipped and failed
    # tests. A test with no result counts as passed; any result that is neither an error nor
    # a failure counts as skipped. Returns None; output goes to stdout.
    # KNOWN ISSUE: the rich markup for the summary lines uses an unclosed tag pair for
    # colour, so the colouring is likely not applied as intended.
    num_passed = 0
    num_skipped = 0
    num_failed = 0
    for testcase in testcases:
        if not testcase.result:
            num_passed += 1
            continue
        for result in testcase.result:
            if isinstance(result, Error):
                icon = ":rotating_light: [white on red]ERROR[/white on red]:"
                num_failed += 1
            elif isinstance(result, Failure):
                icon = ":x: [white on red]Failure[/white on red]:"
                num_failed += 1
            else:
                num_skipped += 1
                continue
            rich.print(
                f"{icon} [bold red]{testcase.classname}.{testcase.name}[/bold red]"
            )
            print(f"{result.text}")
    rich.print(f":white_check_mark: {num_passed} [green]Passed[green]")
    rich.print(f":dash: {num_skipped} [grey]Skipped[grey]")
    rich.print(f":rotating_light: {num_failed} [grey]Failed[grey]")


def parse_args() -> Any:
    # [FUNCTION DOC] Build the argument parser with one positional argument, report_path (a
    # file or directory of xunit reports), parse the process command line and return the
    # namespace. Exits with a usage error on bad arguments.
    parser = argparse.ArgumentParser(
        description="Render xunit output for failed tests",
    )
    parser.add_argument(
        "report_path",
        help="Base xunit reports (single file or directory) to compare to",
    )
    return parser.parse_args()


def main() -> None:
    # [FUNCTION DOC] Entry point: parse arguments, load all test cases from the given report
    # path, and render the failures and summary. Propagates FileNotFoundError for a missing
    # path.
    options = parse_args()
    testcases = parse_junit_reports(options.report_path)
    render_tests(testcases)


if __name__ == "__main__":
    main()
