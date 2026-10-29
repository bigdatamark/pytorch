# [MODULE DOC]
# Defines PythonFile, a lazily tokenized view of a Python source (from a path or a string)
# with helpers for lines, token lines, imports, blocks, set literals and noqa omissions, and
# OmittedLines, which finds lines marked to be skipped by a given linter. No import-time
# side effects.
from __future__ import annotations

import token
from functools import cached_property
from pathlib import Path
from tokenize import generate_tokens, TokenInfo
from typing import TYPE_CHECKING

from . import is_empty, NO_TOKEN, ParseError, ROOT
from .sets import LineWithSets


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing_extensions import Self

    from .block import Block


class PythonFile:
    # [CLASS DOC] One Python source file for a linter. State: linter_name, optional path
    # (made relative to the repository root when absolute) and optional explicit contents.
    # Most derived views (contents, lines, tokens, token_lines, blocks, omitted lines, set
    # literals, import positions) are computed lazily and cached, so a modified file is
    # represented by a new instance via with_contents.
    path: Path | None
    linter_name: str

    def __init__(
        self,
        linter_name: str,
        *,
        contents: str | None = None,
        path: Path | None = None,
    ) -> None:
        # [FUNCTION DOC] Records the linter name and explicit contents, and the path, which is
        # converted to be relative to the repository root when it is absolute (raises a value
        # error if it is outside the root).
        self.linter_name = linter_name
        self._contents = contents
        self.path = path.relative_to(ROOT) if path and path.is_absolute() else path

    @cached_property
    def contents(self) -> str:
        # [FUNCTION DOC] Cached source text: the explicit contents if given, else the text read
        # from the path, else an empty string.
        if self._contents is not None:
            return self._contents
        return self.path.read_text() if self.path else ""

    @cached_property
    def lines(self) -> list[str]:
        # [FUNCTION DOC] Cached list of source lines, each keeping its line ending.
        return self.contents.splitlines(keepends=True)

    @classmethod
    def make(cls, linter_name: str, pc: Path | str | None = None) -> Self:
        # [FUNCTION DOC] Factory: a Path argument becomes the file path, any other value
        # (string or none) becomes the explicit contents.
        if isinstance(pc, Path):
            return cls(linter_name, path=pc)
        else:
            return cls(linter_name, contents=pc)

    def with_contents(self, contents: str) -> Self:
        # [FUNCTION DOC] Returns a new instance of the same class with the same linter name and
        # path but the given contents; no cached views are shared.
        return self.__class__(self.linter_name, contents=contents, path=self.path)

    @cached_property
    def omitted(self) -> OmittedLines:
        # [FUNCTION DOC] Cached OmittedLines for this file and linter, used to skip lines
        # marked with a suppression comment. Raises an assertion error if linter_name is none.
        if self.linter_name is None:
            raise AssertionError("linter_name is None")
        return OmittedLines(self.lines, self.linter_name)

    @cached_property
    def tokens(self) -> list[TokenInfo]:
        # [FUNCTION DOC] Cached full token list from the standard tokenizer over the lines.
        """This file, tokenized. Raises IndentationError on badly indented code."""
        return list(generate_tokens(iter(self.lines).__next__))

    @cached_property
    def token_lines(self) -> list[list[TokenInfo]]:
        # [FUNCTION DOC] Cached tokens grouped into logical lines, each ending at a newline
        # token. Comments, end marker and non-logical newlines are left out. A trailing empty
        # group is removed.
        """Returns lists of TokenInfo segmented by token.NEWLINE"""
        token_lines: list[list[TokenInfo]] = [[]]

        for t in self.tokens:
            if t.type not in (token.COMMENT, token.ENDMARKER, token.NL):
                token_lines[-1].append(t)
                if t.type == token.NEWLINE:
                    token_lines.append([])
        if token_lines and not token_lines[-1]:
            token_lines.pop()
        return token_lines

    @cached_property
    def import_lines(self) -> list[list[int]]:
        # [FUNCTION DOC] Cached pair of lists of token-line indexes: lines starting with from,
        # and lines starting with import. Scans from the top and stops at the first indented
        # line, so only leading top-level statements are examined.
        froms, imports = [], []
        for i, (t, *_) in enumerate(self.token_lines):
            if t.type == token.INDENT:
                break
            if t.type == token.NAME:
                if t.string == "from":
                    froms.append(i)
                elif t.string == "import":
                    imports.append(i)

        return [froms, imports]

    @cached_property
    def opening_comment_lines(self) -> int:
        # [FUNCTION DOC] Cached count of lines before the first line not starting with a hash
        # character (a blank line also ends the run).
        # KNOWN ISSUE: a file consisting only of comment lines yields zero rather than the
        # number of comment lines.
        """The number of comments at the very top of the file."""
        it = (i for i, s in enumerate(self.lines) if not s.startswith("#"))
        return next(it, 0)

    def __getitem__(self, i: int | slice) -> TokenInfo | Sequence[TokenInfo]:
        # [FUNCTION DOC] Indexing or slicing the file returns the corresponding token(s).
        return self.tokens[i]

    def next_token(self, start: int, token_type: int, error: str) -> int:
        # [FUNCTION DOC] Returns the index of the first token at or after start with the given
        # type. Raises ParseError (attached to the last token) with the error message if none.
        for i in range(start, len(self.tokens)):
            if self.tokens[i].type == token_type:
                return i
        raise ParseError(self.tokens[-1], error)

    def docstring(self, start: int) -> str:
        # [FUNCTION DOC] Scans tokens after index start and returns the first string token's
        # text, or an empty string if an indent, dedent, newline or other empty token comes
        # first or the tokens run out.
        # KNOWN ISSUE: it does not skip the rest of a signature, so a string token inside the
        # header (for example a default value) would be returned if start is the def token.
        for i in range(start + 1, len(self.tokens)):
            tk = self.tokens[i]
            if tk.type == token.STRING:
                return tk.string
            if is_empty(tk):
                return ""
        return ""

    @cached_property
    def indent_to_dedent(self) -> dict[int, int]:
        # [FUNCTION DOC] Cached mapping from the index of each indent token to the index of its
        # matching dedent token, computed with a stack over all tokens.
        dedents = dict[int, int]()
        stack = list[int]()

        for i, t in enumerate(self.tokens):
            if t.type == token.INDENT:
                stack.append(i)
            elif t.type == token.DEDENT:
                dedents[stack.pop()] = i

        return dedents

    @cached_property
    def braced_sets(self) -> list[Sequence[TokenInfo]]:
        # [FUNCTION DOC] Cached list of token sequences for brace-delimited set literals found
        # on all logical lines, excluding those on lines marked as omitted for this linter.
        lines = [t for tl in self._lines_with_sets for t in tl.braced_sets]
        return [s for s in lines if not self.omitted(s)]

    @cached_property
    def sets(self) -> list[TokenInfo]:
        # [FUNCTION DOC] Cached list of tokens that name set constructor calls found on all
        # logical lines, excluding those on lines marked as omitted for this linter.
        tokens = [t for tl in self._lines_with_sets for t in tl.sets]
        return [t for t in tokens if not self.omitted([t])]

    @cached_property
    def insert_import_line(self) -> int | None:
        # [FUNCTION DOC] Cached line number where an ordered-set import should be inserted, or
        # none if some leading import line already mentions OrderedSet. The position is the
        # line after the last from-import (else last import); with no imports, the line after
        # the opening comment block.
        froms, imports = self.import_lines
        for i in froms + imports:
            tl = self.token_lines[i]
            if any(i.type == token.NAME and i.string == "OrderedSet" for i in tl):
                return None
        if section := froms or imports:
            return self._lines_with_sets[section[-1]].tokens[-1].start[0] + 1
        return self.opening_comment_lines + 1

    @cached_property
    def _lines_with_sets(self) -> list[LineWithSets]:
        # [FUNCTION DOC] HELPER: Cached list wrapping each logical token line in a LineWithSets
        # analyzer, index-aligned with token_lines.
        return [LineWithSets(tl) for tl in self.token_lines]

    @cached_property
    def blocks(self) -> list[Block]:
        # [FUNCTION DOC] Cached list of def and class blocks of this file, built on first use
        # by the blocks module (imported lazily to avoid an import cycle).
        from .blocks import blocks

        return blocks(self)


class OmittedLines:
    # [CLASS DOC] Index of lines whose text ends with a suppression comment naming a given
    # linter. State: the source lines and the set of one-based omitted line numbers. Callable
    # on a token range to ask whether any covered line is omitted.
    """Read lines textually and find comment lines that end in 'noqa {linter_name}'"""

    omitted: set[int]

    def __init__(self, lines: Sequence[str], linter_name: str) -> None:
        # [FUNCTION DOC] Stores the lines and records the one-based numbers of all lines that,
        # ignoring trailing whitespace, end with a noqa comment for linter_name.
        self.lines = lines
        suffix = f"# noqa: {linter_name}"
        omitted = ((i, s.rstrip()) for i, s in enumerate(lines))
        self.omitted = {i + 1 for i, s in omitted if s.endswith(suffix)}

    def __call__(
        self, tokens: Sequence[TokenInfo], begin: int = 0, end: int = NO_TOKEN
    ) -> bool:
        # [FUNCTION DOC] Returns true if any physical line spanned by tokens in the index range
        # from begin up to end (default: all tokens) is omitted. The span runs from the
        # smallest start line to the largest end line of those tokens, since a logical line may
        # cover several physical lines.
        if end == NO_TOKEN:
            end = len(tokens)
        # A token_line might span multiple physical lines
        start = min((tokens[i].start[0] for i in range(begin, end)), default=0)
        end = max((tokens[i].end[0] for i in range(begin, end)), default=-1)
        return self.contains_lines(start, end)

    def contains_lines(self, begin: int, end: int) -> bool:
        # [FUNCTION DOC] True if any line number from begin to end inclusive is omitted.
        return bool(self.omitted.intersection(range(begin, end + 1)))
