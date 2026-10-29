# [MODULE DOC]
# Defines Block, a description of one def or class region in a tokenized Python file (token
# index range, names, docstring, parent and children relations, decorators), plus a helper
# that recovers decorator text from tokens. No import-time side effects.
from __future__ import annotations

import dataclasses as dc
import itertools
import token
from enum import Enum
from functools import cached_property, total_ordering
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from tokenize import TokenInfo
    from typing_extensions import Self


_OVERRIDES = {"@override", "@typing_extensions.override", "@typing.override"}


@total_ordering
@dc.dataclass
class Block:
    # [CLASS DOC] One def or class region of a file, as indexes into a shared token list.
    # State: category, tokens, name, begin and end token indexes, docstring, and family
    # fields (full_name, index, is_local, is_method, parent, children) that are filled in
    # later by the blocks builder. Blocks compare equal when they share the same token list
    # and index. Key members: line properties, decorators, display_name and as_data.
    """A block of Python code starting with either `def` or `class`"""

    class Category(str, Enum):
        # [CLASS DOC] Enumeration of block kinds: class or def; values are the keywords.
        CLASS = "class"
        DEF = "def"

    category: Category

    # The sequence of tokens that contains this Block.
    # Tokens are represented in `Block` as indexes into `self.tokens`
    tokens: Sequence[TokenInfo] = dc.field(repr=False)

    # The name of the function or class being defined
    name: str

    # The index of the very first token in the block (the "class" or "def" keyword)
    begin: int

    # The index of the last token for this block
    end: int

    # The docstring for the block
    docstring: str

    # These next members only get filled in after all blocks have been constructed
    # and figure out family ties

    # The full qualified name of the block within the file.
    # This is the name of this block and all its parents, joined with `.`.
    full_name: str = ""

    # The index of this block within the full list of blocks in the file
    index: int = 0

    # Is this block contained within a function definition?
    is_local: bool = dc.field(default=False, repr=False)

    # Is this block a function definition in a class definition?
    is_method: bool = dc.field(default=False, repr=False)

    # A block index to the parent of this block, or None for a top-level block.
    parent: int | None = None

    # A list of block indexes for the children
    children: list[int] = dc.field(default_factory=list)

    @property
    def start_line(self) -> int:
        # [FUNCTION DOC] Line number of the first token (the def or class keyword).
        """The line number for the def or class statement"""
        return self.tokens[self.begin].start[0]

    @property
    def end_line(self) -> int:
        # [FUNCTION DOC] Line number where the last token of the block starts.
        return self.tokens[self.end].start[0]

    @property
    def line_count(self) -> int:
        # [FUNCTION DOC] Number of lines the block spans, counting both end lines.
        return self.end_line - self.start_line + 1

    @property
    def line_range(self) -> range:
        # [FUNCTION DOC] Range of line numbers covered by the block, end inclusive.
        return range(self.start_line, self.end_line + 1)

    @property
    def is_class(self) -> bool:
        # [FUNCTION DOC] True if the block is a class definition, false for a function.
        return self.category == Block.Category.CLASS

    @property
    def display_name(self) -> str:
        # [FUNCTION DOC] Human-readable label: the keyword and full dotted name, with a pair
        # of parentheses appended for functions.
        """A user-friendly name like 'class One' or 'def One.method()'"""
        ending = "" if self.is_class else "()"
        return f"{self.category.value} {self.full_name}{ending}"

    @cached_property
    def decorators(self) -> list[str]:
        # [FUNCTION DOC] Cached list of decorator texts (at sign plus arguments, whitespace
        # removed) applied to this block, in source order, found by scanning tokens backward
        # from the block start.
        """A list of decorators for this function or method.

        Each decorator both the @ symbol and any arguments to the decorator
        but no extra whitespace.
        """
        return _get_decorators(self.tokens, self.begin)

    @cached_property
    def is_override(self) -> bool:
        # [FUNCTION DOC] Cached flag: true for a function carrying an override decorator (bare,
        # typing, or typing_extensions form); always false for classes.
        return not self.is_class and bool(_OVERRIDES.intersection(self.decorators))

    DATA_FIELDS = (
        "category",
        "children",
        "decorators",
        "display_name",
        "docstring",
        "full_name",
        "index",
        "is_local",
        "is_method",
        "line_count",
        "parent",
        "start_line",
    )

    def as_data(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a plain mapping of the fields listed in DATA_FIELDS, with the
        # category replaced by its string value, suitable for serialization.
        d = {i: getattr(self, i) for i in self.DATA_FIELDS}
        d["category"] = d["category"].value
        return d

    @property
    def is_init(self) -> bool:
        # [FUNCTION DOC] True if the block is a function named __init__.
        return not self.is_class and self.name == "__init__"

    def contains(self, b: Block) -> bool:
        # [FUNCTION DOC] True if block b starts on a later line than this block and ends no
        # later than this block ends (line-based nesting test).
        return self.start_line < b.start_line and self.end_line >= b.end_line

    def __eq__(self, o: object) -> bool:
        # [FUNCTION DOC] Blocks are equal when they come from the same token list object and
        # have the same index. Raises an assertion error if compared with a non-Block.
        # KNOWN ISSUE: comparing with a non-Block raises instead of returning not-implemented,
        # and equality is by index while the hash is identity-based, so equal blocks may hash
        # differently.
        if not isinstance(o, Block):
            raise AssertionError(f"Expected Block, got {type(o)}")
        return o.tokens is self.tokens and o.index == self.index

    def __hash__(self) -> int:
        # [FUNCTION DOC] Identity-based hash from the base object implementation.
        return super().__hash__()

    def __lt__(self, o: Self) -> bool:
        # [FUNCTION DOC] Ordering by block index, requiring the same token list; raises an
        # assertion error otherwise.
        # KNOWN ISSUE: the comparison is inverted (this block is "less" when its index is
        # larger), so sorting yields reverse file order; may be intentional but is surprising.
        if not (isinstance(o, Block) and o.tokens is self.tokens):
            raise AssertionError("Expected Block with same tokens")
        return o.index < self.index


_IGNORE = {token.COMMENT, token.DEDENT, token.INDENT, token.NL}


def _get_decorators(tokens: Sequence[TokenInfo], block_start: int) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the decorator texts that precede the def or class token at
    # block_start, in source order. Walks logical lines backward from the block; a line whose
    # first meaningful token is an at sign is a decorator (its tokens joined without
    # whitespace); the first line that is any other statement ends the scan.
    def decorators() -> Iterator[str]:
        # [FUNCTION DOC] HELPER: Generator yielding decorator texts from nearest to farthest
        # (the caller reverses them), using the enclosing token list and block start.
        rev = reversed(range(block_start))
        newlines = (i for i in rev if tokens[i].type == token.NEWLINE)
        it = iter(itertools.chain(newlines, [-1]))
        # The -1 accounts for the very first line in the file

        end = next(it, -1)  # Like itertools.pairwise in Python 3.10
        for begin in it:
            for i in range(begin + 1, end):
                t = tokens[i]
                if t.type == token.OP and t.string == "@":
                    useful = (t for t in tokens[i:end] if t.type not in _IGNORE)
                    yield "".join(s.string.strip("\n") for s in useful)
                    break
                elif t.type not in _IGNORE:
                    return  # A statement means no more decorators
            end = begin

    out = list(decorators())
    out.reverse()
    return out
