# [PACKAGE DOC]
# Shared infrastructure for token-based Python linters used through lintrunner. Public
# exports: Block (a def or class region), FileLinter (base class for linters), PythonFile
# (tokenized source file), LintResult (one finding or edit), ParseError, is_empty and ROOT
# (the repository root path). Also defines token-class constants shared by the submodules.
# The submodule imports sit at the bottom so they can import names defined above.
from __future__ import annotations

import token
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence
    from tokenize import TokenInfo


__all__ = [
    "Block",
    "FileLinter",
    "is_empty",
    # pyrefly: ignore [bad-dunder-all]
    "LineWithSets",
    "LintResult",
    "ParseError",
    "PythonFile",
    "ROOT",
]

NO_TOKEN = -1

# Python 3.12 and up have two new token types, FSTRING_START and FSTRING_END
_START_OF_LINE_TOKENS = token.DEDENT, token.INDENT, token.NEWLINE
_IGNORED_TOKENS = token.COMMENT, token.ENDMARKER, token.ENCODING, token.NL
_EMPTY_TOKENS = dict.fromkeys(_START_OF_LINE_TOKENS + _IGNORED_TOKENS)

_LINTER = Path(__file__).absolute().parents[0]
ROOT = _LINTER.parents[3]


class ParseError(ValueError):
    # [CLASS DOC] Error raised when source tokens cannot be parsed as expected (for example
    # unbalanced brackets). Carries the offending token in the token attribute so callers can
    # report its position.
    def __init__(self, token: TokenInfo, *args: str) -> None:
        # [FUNCTION DOC] Creates the error with the given message arguments and remembers the
        # token at which parsing failed.
        super().__init__(*args)
        self.token = token


def is_empty(t: TokenInfo) -> bool:
    # [FUNCTION DOC] True if the token is whitespace-like or structural only: indent, dedent,
    # newline, comment, end marker, encoding or non-logical newline.
    return t.type in _EMPTY_TOKENS


from .block import Block
from .file_linter import FileLinter
from .messages import LintResult
from .python_file import PythonFile
