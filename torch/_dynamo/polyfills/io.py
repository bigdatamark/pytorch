# [MODULE DOC] Dynamo polyfill for the C-implemented _io.text_encoding.
# Registers a pure-Python replacement via substitute_in_graph so Dynamo can trace it
# (and constant fold it). Exports text_encoding. Import-time side effect: the
# substitution is registered with Dynamo's polyfill machinery.
"""
Python polyfills for _io
"""

from __future__ import annotations

import _io

import sys
import warnings

from ..decorators import substitute_in_graph


__all__ = ["text_encoding"]


# Copied from Lib/_pyio.py in the standard library
# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(_io.text_encoding, can_constant_fold_through=True)
def text_encoding(encoding: str | None, stacklevel: int = 2, /) -> str:
    # [FUNCTION DOC] Polyfill of _io.text_encoding. If encoding is None, returns the
    # string utf-8 when the interpreter is in UTF-8 mode, else the string locale; in
    # that case it also emits an EncodingWarning if warn_default_encoding is set
    # (side effect: warning with stacklevel plus one). Otherwise returns encoding as is.
    if encoding is None:
        encoding = "utf-8" if sys.flags.utf8_mode else "locale"
        if sys.flags.warn_default_encoding:
            warnings.warn(
                "'encoding' argument not specified.",
                EncodingWarning,
                stacklevel + 1,
            )
    return encoding
