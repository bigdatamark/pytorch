#!/usr/bin/env python3
# [MODULE DOC] Lint adapter requiring tests to create libraries through the scoped library
# helper rather than constructing a Library directly (it replaces a removed external check).
# Uses the abstract syntax tree to find calls and supports suppression with a noqa comment for
# this linter code on the call line or the line above. Prints error JSON lines when run as
# a script.
"""
This lint verifies that tests under test/ should use ``torch.library._scoped_library`` instead
of ``torch.library.Library()``. It replaces the old ``TOR901`` check from removed torchfix.

Skip linting with ``# noqa: SCOPED_LIBRARY`` on the line or above
"""

from __future__ import annotations

import argparse
import ast
import json
from enum import Enum
from typing import NamedTuple


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


LINTER_CODE = "SCOPED_LIBRARY"


def _is_noqa_suppressed(source_lines: list[str], lineno: int) -> bool:
    # [FUNCTION DOC] HELPER: Return whether the one-based line carries a noqa comment for this
    # linter at its end, or the previous line consists solely of such a comment. Out-of-range line
    # numbers return false.
    if lineno <= 0 or lineno > len(source_lines):
        return False
    if source_lines[lineno - 1].rstrip().endswith(f"# noqa: {LINTER_CODE}"):
        return True
    if lineno > 1 and source_lines[lineno - 2].strip() == f"# noqa: {LINTER_CODE}":
        return True
    return False


def _is_torch_library_call(node: ast.AST) -> bool:
    # [FUNCTION DOC] HELPER: Heuristic check on a call's function expression: true for a bare name
    # Library, or an attribute access ending in Library whose dotted text is a known torch library
    # path or contains the word library.
    # KNOWN ISSUE: any bare name Library matches, even if it is a different class.
    """True if the call is likely torch.library Library construction."""
    if isinstance(node, ast.Name) and node.id == "Library":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Library":
        s = ast.unparse(node)
        return (
            s
            in (
                "torch.library.Library",
                "library.Library",
            )
            or s.endswith(".Library")
            and "library" in s
        )
    return False


def _call_suppressed(source_lines: list[str], node: ast.Call) -> bool:
    # [FUNCTION DOC] HELPER: Return whether any line spanned by the call node is covered by a
    # noqa suppression.
    """True if any line of the call is covered by noqa."""
    start = node.lineno
    end = getattr(node, "end_lineno", None) or start
    for line_no in range(start, end + 1):
        if _is_noqa_suppressed(source_lines, line_no):
            return True
    return False


def check_file(path: str) -> list[LintMessage]:
    # [FUNCTION DOC] Parse the file and return an error LintMessage for every call that looks like
    # direct Library construction and is not suppressed. Files with a syntax error produce no
    # messages. The message column is the zero-based offset of the call. Reads the file; no other
    # side effects.
    # KNOWN ISSUE: the docstring says this applies to tests, but the function checks any file it
    # is given; scoping is left to the lintrunner configuration.
    with open(path, encoding="utf-8") as f:
        content = f.read()
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError:
        return []

    lines = content.splitlines()
    lint_messages = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_torch_library_call(node.func):
            continue
        line_no = node.lineno
        if _call_suppressed(lines, node):
            continue
        lint_messages.append(
            LintMessage(
                path=path,
                line=line_no,
                char=node.col_offset,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="direct-torch-Library",
                original=None,
                replacement=None,
                description=(
                    f"In tests, use torch.library._scoped_library, or skip linting with '# noqa: {LINTER_CODE}'"
                ),
            )
        )
    return lint_messages


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses the list of Python file names, checks each and prints
    # each lint message as a JSON line.
    parser = argparse.ArgumentParser(
        description="Enforce _scoped_library in the PyTorch test tree",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="Python files to check",
    )
    args = parser.parse_args()
    for path in args.filenames:
        for m in check_file(path):
            print(json.dumps(m._asdict()), flush=True)


if __name__ == "__main__":
    main()
