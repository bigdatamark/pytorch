# [MODULE DOC]
# Base class for token-based Python linters run through lintrunner or from the command line.
# FileLinter handles argument parsing, file list expansion, repeated lint-and-fix passes for
# recursive results, and human-readable or JSON output; subclasses supply the _lint method.
# ErrorLines holds the display window sizes. No import-time side effects.
from __future__ import annotations

import json
import sys
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from . import ParseError
from .argument_parser import ArgumentParser
from .messages import LintResult
from .python_file import PythonFile


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator, Sequence
    from typing_extensions import Never


class ErrorLines:
    # [CLASS DOC] HELPER: Constants for the code window shown around an error: total window
    # size, lines shown before the error line, and lines shown after it (derived from the
    # other two). Not instantiated.
    """How many lines to display before and after an error"""

    WINDOW = 5
    BEFORE = 2
    AFTER = WINDOW - BEFORE - 1


class FileLinter:
    # [CLASS DOC] Abstract base for token-based linters. Subclasses set description and
    # linter_name, may set epilog, is_fixer and report_column_numbers, and implement _lint to
    # yield LintResult items for a PythonFile. State: argv, the parser, and a flag recording
    # whether any result has been printed (used for blank-line separation). Key methods:
    # run (entry point), lint_all, _lint_file, _replace (iterative fixing) and _display.
    """The base class that all token-based linters inherit from"""

    description: str
    linter_name: str

    epilog: str | None = None
    is_fixer: bool = True
    report_column_numbers: bool = False

    @abstractmethod
    def _lint(self, python_file: PythonFile) -> Iterator[LintResult]:
        # [FUNCTION DOC] HELPER: Abstract hook that subclasses implement to produce lint
        # results for one parsed file. The base version raises not-implemented.
        raise NotImplementedError

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        # [FUNCTION DOC] Stores the optional argument list (defaulting later to the process
        # arguments), creates the shared argument parser using the class description, epilog
        # and fixer flag, and resets the result-shown flag.
        self.argv = argv
        self.parser = ArgumentParser(
            is_fixer=self.is_fixer,
            description=self.description,
            epilog=self.epilog,
        )
        self.result_shown = False

    @classmethod
    def run(cls) -> Never:
        # [FUNCTION DOC] Entry point: builds a linter, lints all requested files, and exits the
        # process with status 0 on success or when the lintrunner flag is set, else status 1.
        linter = cls()
        sys.exit(not (linter.lint_all() or linter.args.lintrunner))

    def lint_all(self) -> bool:
        # [FUNCTION DOC] Lints every path in order (all are processed even after a failure)
        # and returns true only if every file passed.
        success = True
        for p in self.paths:
            success = self._lint_file(p) and success
        return success

    @classmethod
    def make_file(cls, pc: Path | str | None = None) -> PythonFile:
        # [FUNCTION DOC] Creates a PythonFile for this linter's name from either a path (read
        # lazily from disk) or a string of contents.
        return PythonFile.make(cls.linter_name, pc)

    @cached_property
    def args(self) -> Namespace:
        # [FUNCTION DOC] Cached parsed command line options. Raises a value error if both the
        # fix and lintrunner flags are given.
        args = self.parser.parse_args(self.argv)

        if args.fix and args.lintrunner:
            raise ValueError("--fix and --lintrunner are incompatible")
        return args

    @cached_property
    def code(self) -> str:
        # [FUNCTION DOC] Cached upper-case form of the linter name, used as the lint code.
        return self.linter_name.upper()

    @cached_property
    def paths(self) -> list[Path]:
        # [FUNCTION DOC] Cached sorted list of files to lint. Each file argument is split on
        # colons; an entry starting with an at sign names a text file whose lines are added
        # as paths; a lone double dash is skipped; other entries are paths.
        files = []
        file_parts = (f for fp in self.args.files for f in fp.split(":"))
        for f in file_parts:
            if f.startswith("@"):
                files.extend(Path(f[1:]).read_text().splitlines())
            elif f != "--":
                files.append(f)
        return sorted(Path(f) for f in files)

    def _lint_file(self, p: Path) -> bool:
        # [FUNCTION DOC] HELPER: Lints one file: reads it, computes replacement text and
        # results, prints the results, and in fix mode writes the file when the contents
        # changed. Returns true if there were no results, or in fix mode all results were
        # edits. Side effects: prints to stdout (stderr when verbose) and may write the file.
        if self.args.verbose:
            print(p, "Reading", file=sys.stderr)

        pf = self.make_file(p)
        replacement, results = self._replace(pf)

        if display := list(self._display(pf, results)):
            print(*display, sep="\n")
        if results and self.args.fix and pf.path and pf.contents != replacement:
            pf.path.write_text(replacement)

        return not results or self.args.fix and all(r.is_edit for r in results)

    def _error(self, pf: PythonFile, result: LintResult) -> None:
        # [FUNCTION DOC] HELPER: Hook called when a file cannot be tokenized or parsed, with
        # the resulting error result. The base version does nothing; subclasses may override.
        """Called on files that are unparsable"""

    def _replace(self, pf: PythonFile) -> tuple[str, list[LintResult]]:
        # [FUNCTION DOC] HELPER: Runs the linter and applies its edits repeatedly. Each pass
        # sorts results, converts indentation or parse errors into a single result (and calls
        # the error hook), flags results nested inside earlier ones as recursive, applies
        # non-recursive edits from last to first, and reparses if any recursive results
        # remain. It stops when there are no results, the count does not shrink, or nothing was
        # recursive. Returns the final text and the results of the first pass; in lintrunner
        # mode appends a whole-file suggested-fix result when there were findings.
        # Because of recursive replacements, we need to repeat replacing and reparsing
        # from the inside out until all possible replacements are complete
        previous_result_count = float("inf")
        first_results: list[LintResult] = []
        original = replacement = pf.contents
        results: list[LintResult] = []

        while True:
            try:
                results = sorted(self._lint(pf), key=LintResult.sort_key)
            except IndentationError as e:
                error, (_name, lineno, column, _line) = e.args

                results = [LintResult(error, lineno, column)]
                self._error(pf, *results)

            except ParseError as e:
                results = [LintResult(str(e), *e.token.start)]
                self._error(pf, *results)

            for i, ri in enumerate(results):
                if not ri.is_recursive:
                    for rj in results[i + 1 :]:
                        if ri.contains(rj):
                            rj.is_recursive = True
                        else:
                            break

            first_results = first_results or results
            if not results or len(results) >= previous_result_count:
                break
            previous_result_count = len(results)

            lines = pf.lines[:]
            for r in reversed(results):
                r.apply(lines)
            replacement = "".join(lines)

            if not any(r.is_recursive for r in results):
                break
            pf = pf.with_contents(replacement)

        if first_results and self.args.lintrunner:
            name = f"Suggested fixes for {self.linter_name}"
            msg = LintResult(name=name, original=original, replacement=replacement)
            first_results.append(msg)

        return replacement, first_results

    def _display(self, pf: PythonFile, results: list[LintResult]) -> Iterator[str]:
        # [FUNCTION DOC] HELPER: Yields output strings for the results. In lintrunner mode,
        # one sorted-key JSON message per result. Otherwise a blank separator before every
        # result after the first one shown (tracked across files), then either a
        # path-and-name line when there is no line number, or a code window.
        """Emit a series of human-readable strings representing the results"""
        for r in results:
            if self.args.lintrunner:
                msg = r.as_message(code=self.code, path=str(pf.path))
                yield json.dumps(msg.asdict(), sort_keys=True)
            else:
                if self.result_shown:
                    yield ""
                else:
                    self.result_shown = True
                if r.line is None:
                    yield f"{pf.path}: {r.name}"
                else:
                    yield from (i.rstrip() for i in self._display_window(pf, r))

    def _display_window(self, pf: PythonFile, r: LintResult) -> Iterator[str]:
        # [FUNCTION DOC] HELPER: Yields a header line (path, line, optional one-based column,
        # message) followed by a window of numbered source lines around the error, with a row
        # of carets under the error line marking the column and length (the whole line when
        # no column is known).
        """Display a window onto the code with an error"""
        if r.char is None or not self.report_column_numbers:
            yield f"{pf.path}:{r.line}: {r.name}"
        else:
            yield f"{pf.path}:{r.line}:{r.char + 1}: {r.name}"

        begin = max((r.line or 0) - ErrorLines.BEFORE, 1)
        end = min(begin + ErrorLines.WINDOW, 1 + len(pf.lines))

        for lineno in range(begin, end):
            source_line = pf.lines[lineno - 1].rstrip()
            yield f"{lineno:5} | {source_line}"
            if lineno == r.line:
                spaces = 8 + (r.char or 0)
                carets = len(source_line) if r.char is None else (r.length or 1)
                yield spaces * " " + carets * "^"
