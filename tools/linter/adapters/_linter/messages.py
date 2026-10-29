# [MODULE DOC]
# Data types for linter output: LintSeverity (severity levels), LintMessage (the JSON record
# sent to lintrunner) and LintResult (a finding that may carry a precise edit location and
# nesting information, convertible to a LintMessage). No import-time side effects.
from __future__ import annotations

import dataclasses as dc
from enum import Enum


class LintSeverity(str, Enum):
    # [CLASS DOC] Enumeration of lintrunner severity levels (error, warning, advice,
    # disabled); the values are the lowercase strings used in the JSON output.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


@dc.dataclass
class LintMessage:
    # [CLASS DOC] Record matching lintrunner's JSON message: required code, name and
    # severity, and optional char, description, line, original, path and replacement. The
    # asdict method converts it to a plain mapping.
    """This is a datatype representation of the JSON that gets sent to lintrunner
    as described here:
    https://docs.rs/lintrunner/latest/lintrunner/lint_message/struct.LintMessage.html
    """

    code: str
    name: str
    severity: LintSeverity

    char: int | None = None
    description: str | None = None
    line: int | None = None
    original: str | None = None
    path: str | None = None
    replacement: str | None = None

    asdict = dc.asdict


@dc.dataclass
class LintResult:
    # [CLASS DOC] One linter finding. State: name (message), optional line, char and length
    # locating a span, optional replacement text for that span (making it an edit), optional
    # description and whole-file original, and is_recursive marking results nested inside an
    # earlier result. Key methods: apply, contains, as_message, sort_key.
    """LintResult is a single result from a linter.

    Like LintMessage but the .length member allows you to make specific edits to
    one location within a file, not just replace the whole file.

    Linters can generate recursive results - results that contain other results.

    For example, the annotation linter would find two results in this code sample:

        index = Union[Optional[str], int]

    And the first result, `Union[Optional[str], int]`, contains the second one,
    `Optional[str]`, so the first result is recursive but the second is not.

    If --fix is selected, the linter does a cycle of tokenizing and fixing all
    the non-recursive edits until no edits remain.
    """

    name: str

    line: int | None = None
    char: int | None = None
    replacement: str | None = None
    length: int | None = None  # Not in LintMessage
    description: str | None = None
    original: str | None = None

    is_recursive: bool = False  # Not in LintMessage

    @property
    def is_edit(self) -> bool:
        # [FUNCTION DOC] True when line, char, length and replacement are all present, i.e.
        # the result describes a concrete text replacement.
        return None not in (self.char, self.length, self.line, self.replacement)

    def apply(self, lines: list[str]) -> None:
        # [FUNCTION DOC] Applies the edit in place to the list of source lines: replaces the
        # span of length characters starting at the column on the (one-based) line with the
        # replacement text. Does nothing if the result is not a complete edit or is recursive.
        # Mutates the given list.
        if not (
            self.char is None
            or self.length is None
            or self.line is None
            or self.replacement is None
            or self.is_recursive
        ):
            line = lines[self.line - 1]
            before = line[: self.char]
            after = line[self.char + self.length :]
            lines[self.line - 1] = f"{before}{self.replacement}{after}"

    def contains(self, r: LintResult) -> bool:
        # [FUNCTION DOC] True if r lies on the same line and its span is within this result's
        # span. Raises an assertion error if either result lacks a line or char, or if this
        # result lacks a length.
        if self.char is None or self.line is None:
            raise AssertionError("self.char and self.line must not be None")
        if r.char is None or r.line is None:
            raise AssertionError("r.char and r.line must not be None")
        return self.line == r.line and self.char <= r.char and self.end >= r.end

    @property
    def end(self) -> int:
        # [FUNCTION DOC] Column just past the span (char plus length). Raises an assertion
        # error if char or length is missing.
        if self.char is None or self.length is None:
            raise AssertionError("self.char and self.length must not be None")
        return self.char + self.length

    def as_message(self, code: str, path: str) -> LintMessage:
        # [FUNCTION DOC] Converts to a LintMessage with the given code and path and error
        # severity, dropping the recursion flag and length. For edit results the replacement is
        # cleared so lintrunner is not asked to apply it per span.
        # KNOWN ISSUE: the explanatory comment inside is garbled (a word is missing after "one
        # of our"), so the intent is only inferred.
        d = dc.asdict(self)
        d.pop("is_recursive")
        d.pop("length")
        if self.is_edit:
            # This is one of our , which we don't want to
            # send to lintrunner as a replacement
            d["replacement"] = None

        return LintMessage(code=code, path=path, severity=LintSeverity.ERROR, **d)

    def sort_key(self) -> tuple[int, int, str]:
        # [FUNCTION DOC] Sort key of line, column and name, with missing line or column
        # treated as minus one so location-less results sort first.
        line = -1 if self.line is None else self.line
        char = -1 if self.char is None else self.char
        return line, char, self.name
