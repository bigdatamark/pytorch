# [MODULE DOC] Token-level helpers for the set linter: LineWithSets wraps one logical line of
# Python tokens and finds uses of the built-in set type (the name set, typing Set[...]) and
# braced set literals. No import-time side effects beyond imports.
from __future__ import annotations

import dataclasses as dc
import token
from functools import cached_property
from typing import TYPE_CHECKING

from . import is_empty
from .bracket_pairs import bracket_pairs


if TYPE_CHECKING:
    from tokenize import TokenInfo


@dc.dataclass
class LineWithSets:
    # [CLASS DOC] Dataclass holding the list of tokens of one logical Python line. Lazily
    # computed (cached) views: sets (tokens naming the set type), braced_sets (token slices
    # of non-empty brace set literals) and bracket_pairs (open index to close index map).
    # is_set and is_braced_set are the classification predicates used by those views.
    """A logical line of Python tokens, terminated by a NEWLINE or the end of file"""

    tokens: list[TokenInfo]

    @cached_property
    def sets(self) -> list[TokenInfo]:
        # [FUNCTION DOC] Cached property returning the tokens of this line for which is_set
        # is true, in token order.
        """A list of tokens which use the built-in set symbol"""
        return [t for i, t in enumerate(self.tokens) if self.is_set(i)]

    @cached_property
    def braced_sets(self) -> list[list[TokenInfo]]:
        # [FUNCTION DOC] Cached property returning, for each matched bracket pair accepted
        # by is_braced_set, the slice of tokens from the open brace through the close brace.
        """A list of lists of tokens, each representing a braced set, like {1}"""
        return [
            self.tokens[b : e + 1]
            for b, e in self.bracket_pairs.items()
            if self.is_braced_set(b, e)
        ]

    @cached_property
    def bracket_pairs(self) -> dict[int, int]:
        # [FUNCTION DOC] Cached property: mapping from the index of each opening bracket
        # token to the index of its matching closing bracket, via the shared helper.
        return bracket_pairs(self.tokens)

    def is_set(self, i: int) -> bool:
        # [FUNCTION DOC] Decide whether the token at index i uses the set type. True for a
        # name token Set directly followed by an open square bracket, or for a name token
        # set that is not preceded by def or a dot and not followed by an equals sign (so
        # method definitions, attribute access and keyword/assignment uses are ignored).
        # Pure; returns a bool (the Set branch may return a falsy non-bool token value).
        t = self.tokens[i]
        after = i < len(self.tokens) - 1 and self.tokens[i + 1]
        if t.string == "Set" and t.type == token.NAME:
            # pyrefly: ignore [bad-return]
            return after and after.string == "[" and after.type == token.OP
        return (
            (t.string == "set" and t.type == token.NAME)
            and not (i and self.tokens[i - 1].string in ("def", "."))
            and not (after and after.string == "=" and after.type == token.OP)
        )

    def is_braced_set(self, begin: int, end: int) -> bool:
        # [FUNCTION DOC] Decide whether the bracket pair at token indices begin..end is a
        # non-empty set literal. False if the pair is empty, the opener is not a curly
        # brace, the brace follows the keyword in (membership test), or the top level
        # contains a colon or double-star (dict or unpacking). Nested bracket groups are
        # skipped as a unit; whitespace-like tokens do not count as content.
        if (
            begin + 1 == end
            or self.tokens[begin].string != "{"
            or begin
            and self.tokens[begin - 1].string == "in"  # skip `x in {1, 2, 3}`
        ):
            return False

        i = begin + 1
        empty = True
        while i < end:
            t = self.tokens[i]
            if t.type == token.OP and t.string in (":", "**"):
                return False
            if brace_end := self.bracket_pairs.get(i):
                # Skip to the end of a subexpression
                i = brace_end
            elif not is_empty(t):
                empty = False
            i += 1
        return not empty
