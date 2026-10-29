# [MODULE DOC]
# Utility that matches brackets in a token stream. Defines the bracket tables and the
# optional f-string token types (absent before Python 3.12), and the bracket_pairs function.
# No import-time side effects beyond looking up token constants.
import token
from collections.abc import Sequence
from tokenize import TokenInfo

from . import NO_TOKEN, ParseError


FSTRING_START: int = getattr(token, "FSTRING_START", NO_TOKEN)
FSTRING_END: int = getattr(token, "FSTRING_END", NO_TOKEN)

BRACKETS = {"{": "}", "(": ")", "[": "]"}
BRACKETS_INV = {j: i for i, j in BRACKETS.items()}


def bracket_pairs(tokens: Sequence[TokenInfo]) -> dict[int, int]:
    # [FUNCTION DOC] Scans the tokens with a stack and returns a mapping from the index of
    # each opening bracket (brace, parenthesis, square bracket) to the index of its closer.
    # Brackets inside f-string spans are ignored. Raises ParseError for a closer with no
    # opener, mismatched bracket kinds, unbalanced f-string markers, or brackets left open.
    # KNOWN ISSUE: the left-open errors refer to the last token variable after the loop, which
    # is undefined if the token list is empty (cannot happen with a real end marker).
    """Returns a dictionary mapping opening to closing brackets"""
    braces: dict[int, int] = {}
    stack: list[int] = []
    fstring_depth = 0

    for i, t in enumerate(tokens):
        if t.type == token.OP and not fstring_depth:
            if t.string in BRACKETS:
                stack.append(i)
            elif inv := BRACKETS_INV.get(t.string):
                if not stack:
                    raise ParseError(t, "Never opened")
                begin = stack.pop()
                braces[begin] = i

                b = tokens[begin].string
                if b != inv:
                    raise ParseError(t, f"Mismatched braces '{b}' at {begin}")
        elif t.type == FSTRING_START:
            fstring_depth += 1
        elif t.type == FSTRING_END:
            if not fstring_depth:
                raise ParseError(t, "Mismatched FSTRING_START/FSTRING_END")
            fstring_depth -= 1
    if fstring_depth:
        raise ParseError(t, "Left open")
    if stack:
        raise ParseError(t, "Left open")
    return braces
