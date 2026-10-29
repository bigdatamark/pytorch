# [MODULE DOC] Line-oriented code buffer with indentation tracking (IndentedBuffer), a deferred
# line base class whose content can be "unwritten" later (DeferredLineBase), and small records
# for line-number mapping (LineContext, ValueWithLineMap). Used by inductor, dynamo and
# AOTAutograd code generation. Standard library only; no import-time side effects.
"""
IndentedBuffer: a line-oriented buffer that tracks indentation as source is
appended and renders it (optionally with a line map for traceback remapping).

This is the shared primitive underneath inductor's codegen buffers
(torch._inductor.utils re-exports these names and subclasses IndentedBuffer for
its JIT/AOTI variants), dynamo's guard codegen (IndentedBufferWithPrefix), and
AOTAutograd's runtime-wrapper codegen (PySourceBuilder). It lives in torch.utils
so those layers can share it without importing across the _functorch -> _inductor
edge or pulling in the heavy torch._inductor.utils module. The cluster depends
only on the standard library.
"""

from __future__ import annotations

import contextlib
import math
import textwrap
from dataclasses import dataclass
from io import StringIO
from typing import Any, NamedTuple, TYPE_CHECKING
from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class LineContext(NamedTuple):
    # [CLASS DOC] Marker entry stored in a buffer that carries an arbitrary context object (for
    # example origin information). It produces no output text; it records the position in the
    # rendered output where the context applies.
    context: Any


@dataclass
class ValueWithLineMap:
    # [CLASS DOC] Result record of rendering a buffer: value is the full text, line_map is a
    # list of pairs of (1-based output line number, context object) taken from LineContext entries.
    value: str
    line_map: list[tuple[int, LineContext]]


class IndentedBuffer:
    # [CLASS DOC] Accumulates source lines with a current indentation level. State: _lines (list
    # of strings, deferred lines or line-context markers; stored lines already include their
    # indentation prefix) and _indent (current level, in units of tabwidth spaces, default 4).
    # Key methods: writeline/writelines/splice to add text, indent/do_indent/do_unindent to change
    # level, getvalue/getvaluewithlinemap/getrawvalue to render. The *_jit/*_aot variants exist
    # for subclasses with separate buffers; here jit forms act normally, aot forms do nothing.
    tabwidth = 4

    def __init__(self, initial_indent: int = 0) -> None:
        # [FUNCTION DOC] Creates an empty buffer with the given starting indentation level.
        self._lines: list[DeferredLineBase | LineContext | str] = []
        self._indent = initial_indent

    @contextlib.contextmanager
    def set_tabwidth(self, tabwidth: int) -> Iterator[None]:
        # [FUNCTION DOC] Context manager that temporarily sets the spaces-per-level on this
        # instance and restores the previous value on exit, even on error.
        prev = self.tabwidth
        try:
            self.tabwidth = tabwidth
            yield
        finally:
            self.tabwidth = prev

    def getvaluewithlinemap(self) -> ValueWithLineMap:
        # [FUNCTION DOC] Renders the buffer: evaluates deferred lines (skipping ones that return
        # None), writes each line followed by a newline, and records the current output line
        # number for each context marker. Returns the text and the line map. Raises
        # AssertionError if a line is not a string.
        buf = StringIO()
        p = 1
        linemap: list[tuple[int, LineContext]] = []
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                linemap.append((p, li.context))
                continue
            else:
                line = li
            if not isinstance(line, str):
                raise AssertionError(f"Expected str, got {type(line)}")
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return ValueWithLineMap(buf.getvalue(), linemap)

    def getvalue(self) -> str:
        # [FUNCTION DOC] Returns the rendered text, discarding the line map.
        return self.getvaluewithlinemap().value

    def getrawvalue(self) -> str:
        # [FUNCTION DOC] Renders like getvalue but ignores context markers, and for a line ending
        # with a backslash omits that backslash and the newline, joining it with the next line.
        # Raises AssertionError if a line is not a string.
        buf = StringIO()
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                continue
            else:
                line = li
            if not isinstance(line, str):
                raise AssertionError(f"Expected str, got {type(line)}")
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def get_lines_ref(self):
        # [FUNCTION DOC] Returns the live internal list of lines (not a copy); mutating it
        # changes the buffer.
        return self._lines

    def clear(self) -> None:
        # [FUNCTION DOC] Removes all stored lines; the indentation level is unchanged.
        self._lines.clear()

    def __bool__(self) -> bool:
        # [FUNCTION DOC] True if the buffer holds any entry (including markers).
        return bool(self._lines)

    def prefix(self) -> str:
        # [FUNCTION DOC] Returns the run of spaces for the current indentation level.
        return " " * (self._indent * self.tabwidth)

    def newline(self) -> None:
        # [FUNCTION DOC] Appends a blank line (a whitespace-only line is stored as empty).
        self.writeline("\n")

    def writeline(self, line: LineContext | DeferredLineBase | str) -> None:
        # [FUNCTION DOC] Appends one entry. Context markers are stored unchanged, deferred lines
        # get the current prefix applied, strings with non-whitespace content are stored with the
        # prefix, and blank strings are stored as an empty string.
        if isinstance(line, LineContext):
            self._lines.append(line)
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writeline_jit(self, line: LineContext | DeferredLineBase | str) -> None:
        # [FUNCTION DOC] Hook for subclasses; here it simply forwards to writeline.
        """Write to JIT buffer only. On a plain IndentedBuffer, same as writeline."""
        self.writeline(line)

    def writeline_aot(self, line: LineContext | DeferredLineBase | str) -> None:
        # [FUNCTION DOC] Hook for subclasses; does nothing on this class.
        """Write to AOTI buffer only. No-op on a plain IndentedBuffer."""

    def splice_jit(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        # [FUNCTION DOC] Hook for subclasses; here it simply forwards to splice.
        """Splice to JIT buffer only. On a plain IndentedBuffer, same as splice."""
        self.splice(other_code, strip=strip)

    def splice_aot(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        # [FUNCTION DOC] Hook for subclasses; does nothing on this class.
        """Splice to AOTI buffer only. No-op on a plain IndentedBuffer."""

    def writelines(self, lines: Sequence[LineContext | DeferredLineBase | str]) -> None:
        # [FUNCTION DOC] Appends each entry of the sequence in order using writeline.
        for line in lines:
            self.writeline(line)

    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        # [FUNCTION DOC] Returns a context manager that raises the indentation level by offset
        # on entry and lowers it by the same amount on exit (also on error).
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            # [FUNCTION DOC] HELPER: Generator body of the indentation context manager.
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def do_indent(self, offset: int = 1) -> None:
        # [FUNCTION DOC] Raises the indentation level by offset.
        self._indent += offset

    def do_unindent(self, offset: int = 1) -> None:
        # [FUNCTION DOC] Lowers the indentation level by offset (no lower bound check).
        self._indent -= offset

    def splice(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        # [FUNCTION DOC] Appends another buffer or a text block at the current indentation. For
        # a buffer, removes the common leading whitespace of its lines and re-adds lines through
        # the base writeline (markers copied as is). For text, dedents it, optionally strips
        # leading whitespace (strip), drops trailing whitespace, and appends each line; empty text
        # adds nothing. The base writeline is called explicitly, bypassing subclass overrides.
        if isinstance(other_code, IndentedBuffer):
            dedent = float("inf")

            for line in other_code._lines:
                if not isinstance(line, LineContext) and line:
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            if math.isinf(dedent):
                dedent = 0
            for line in other_code._lines:
                if isinstance(line, LineContext):
                    self._lines.append(line)
                else:
                    IndentedBuffer.writeline(self, line[int(dedent) :])
        else:
            other_code = textwrap.dedent(other_code)
            if strip:
                other_code = other_code.lstrip()
            if not other_code:
                return
            other_code = other_code.rstrip()
            for s in other_code.split("\n"):
                IndentedBuffer.writeline(self, s)

    def map(self, func: Callable[[Any], Any]) -> IndentedBuffer:
        # [FUNCTION DOC] Returns a new plain IndentedBuffer with the same indentation whose
        # lines are the results of applying func to each stored entry. Does not modify self.
        res = IndentedBuffer(initial_indent=self._indent)
        res._lines = [func(line) for line in self._lines]
        return res

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the type and the rendered text.
        return f"{type(self)}({self.getvalue()})"

    def __add__(self, other: Self) -> IndentedBuffer:
        # [FUNCTION DOC] Concatenates two buffers into a new plain IndentedBuffer. Raises
        # AssertionError if the indentation levels differ. KNOWN ISSUE: lines are re-added via
        # writeline, so entries that already carry an indentation prefix get the prefix applied
        # again when the level is nonzero; the result type is always the base class.
        if self._indent != other._indent:
            raise AssertionError(f"Indent mismatch: {self._indent} != {other._indent}")
        res = IndentedBuffer(initial_indent=self._indent)
        # TODO(rec): or should this be self.__class__(initial_indent=self._indent)?
        res.writelines(self._lines)
        res.writelines(other._lines)
        return res

    def contains(self, new_line: DeferredLineBase | LineContext | str) -> bool:
        # [FUNCTION DOC] True if an equal entry is already stored (compared exactly, including
        # any indentation prefix).
        return new_line in self._lines


class DeferredLineBase:
    # [CLASS DOC] Abstract base for a line whose text can be withdrawn after being written.
    # State: line (the text, empty if the input was blank). Subclasses implement __call__
    # (return the text or None when unwritten) and _new_line (copy with new text and the same
    # condition). Supports prefixing, left strip, slicing, truthiness and length like a string.
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line: str):
        # [FUNCTION DOC] Stores the text, normalizing whitespace-only text to an empty string.
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> str | None:
        # [FUNCTION DOC] Abstract: subclasses return the line text, or None if it has been
        # unwritten. Raises NotImplementedError here.
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError

    def _new_line(self, line: str) -> Self:
        # [FUNCTION DOC] HELPER: Abstract factory for a copy with different text. Raises
        # NotImplementedError here.
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError

    def with_prefix(self, prefix: str) -> Self:
        # [FUNCTION DOC] Returns a new deferred line with the prefix prepended to the text.
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self) -> Self:
        # [FUNCTION DOC] Returns a new deferred line with leading whitespace removed.
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index: int | slice) -> Self:
        # [FUNCTION DOC] Returns a new deferred line holding the indexed character or slice of
        # the text.
        return self._new_line(self.line[index])

    def __bool__(self) -> bool:
        # [FUNCTION DOC] True if the text is non-empty (independent of whether it is unwritten).
        return bool(self.line)

    def __len__(self) -> int:
        # [FUNCTION DOC] Length of the text.
        return len(self.line)
