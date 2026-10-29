"""
Lints test/dynamo test files for ``torch.compile`` uses that do not pass an
explicit ``backend=``.

The default backend of ``torch.compile`` is ``inductor``, which pulls in the
full codegen stack.  Most Dynamo tests only exercise tracing/correctness and do
not need Inductor, so relying on the implicit default needlessly slows the
tests down and couples them to codegen flakiness.  This linter forces every
``torch.compile`` in a Dynamo test to consciously pick a backend (``eager``,
``aot_eager``, ``inductor``, ...).

Suppress a deliberate use with a trailing ``# noqa: UNSPECIFIED_BACKEND``.
"""

# [MODULE DOC]
# Lintrunner adapter that reports calls and decorator uses of torch.compile that do not pass
# an explicit backend argument in Dynamo tests, unless the line carries the suppression
# comment for this linter. Main contents: LintSeverity/LintMessage types, helpers that
# recognize torch.compile and check suppression, check_file and main (parallel over files with
# 8 worker processes, printing JSON lines). No import-time side effects.
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "UNSPECIFIED_BACKEND"


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


def _is_torch_compile(func: ast.expr) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the expression is an attribute access named
    # compile on the name torch.
    # Matches ``torch.compile`` (the attribute access), whether called or used
    # bare as a decorator.
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "compile"
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _has_backend_kwarg(call: ast.Call) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the call passes a keyword argument named backend.
    return any(kw.arg == "backend" for kw in call.keywords)


def _suppressed(source_lines: list[str], start: int, end: int) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if any source line in the inclusive 1-based range
    # contains the noqa marker for this linter code.
    # start/end are 1-based inclusive line numbers spanning the call.
    for lineno in range(start, end + 1):
        if f"noqa: {LINTER_CODE}" in source_lines[lineno - 1]:
            return True
    return False


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Parses the file and collects torch.compile calls without a backend keyword
    # plus bare torch.compile decorators, skipping those whose line span has the suppression
    # comment. Returns an implicit-inductor-backend error per offender (with 1-based column), or
    # a single syntax-error message if the file cannot be parsed. Raises OSError if unreadable.
    with open(filename) as f:
        source = f.read()
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as err:
        return [
            LintMessage(
                path=filename,
                line=err.lineno,
                char=err.offset,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="syntax-error",
                original=None,
                replacement=None,
                description=f"Failed to parse file: {err}",
            )
        ]

    # Collect offending nodes: (node-to-report, span-end-line).
    offenders: list[ast.expr] = []

    # ``torch.compile(...)`` call form. This also covers the
    # ``@torch.compile(...)`` call-decorator form, since a decorator call is an
    # ast.Call reachable from ast.walk.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _is_torch_compile(node.func)
            and not _has_backend_kwarg(node)
        ):
            offenders.append(node)

    # Bare ``@torch.compile`` decorator (Attribute, not a Call -> default
    # backend). Only reachable via decorator_list.
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Attribute) and _is_torch_compile(dec):
                offenders.append(dec)

    messages: list[LintMessage] = []
    for node in offenders:
        end = getattr(node, "end_lineno", None) or node.lineno
        if _suppressed(source_lines, node.lineno, end):
            continue
        messages.append(
            LintMessage(
                path=filename,
                line=node.lineno,
                char=node.col_offset + 1,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="implicit-inductor-backend",
                original=None,
                replacement=None,
                description=(
                    "torch.compile in a Dynamo test must pass an explicit "
                    'backend= (e.g. backend="eager"). The implicit default is '
                    '"inductor", which most Dynamo tests do not need. Suppress '
                    "a deliberate use with `# noqa: UNSPECIFIED_BACKEND`."
                ),
            )
        )

    return messages


def main() -> None:
    # [FUNCTION DOC] Command line entry point. Parses the list of file paths, checks them in a
    # pool of 8 processes and prints each message as a JSON line on standard output.
    parser = argparse.ArgumentParser(
        description="Dynamo test torch.compile backend linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    with mp.Pool(8) as pool:
        results = pool.map(check_file, args.filenames)

    for sublist in results:
        for lint_message in sublist:
            print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
