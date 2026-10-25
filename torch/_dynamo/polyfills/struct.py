# [MODULE DOC] Dynamo polyfills for struct.pack and struct.unpack, registered via
# substitute_in_graph with constant folding allowed. Exports pack and unpack.
# Import-time side effect: registers the substitutions.
"""
Python polyfills for struct
"""

from __future__ import annotations

import struct
from typing import Any
from typing_extensions import Buffer

from ..decorators import substitute_in_graph


__all__ = [
    "pack",
    "unpack",
]


@substitute_in_graph(struct.pack, can_constant_fold_through=True)  # type: ignore[arg-type]
def pack(format: bytes | str, /, *v: Any) -> bytes:
    # [FUNCTION DOC] Polyfill of struct.pack: delegates to the real struct.pack with the
    # format and values and returns the packed bytes. Raises struct.error on bad input.
    return struct.pack(format, *v)


@substitute_in_graph(struct.unpack, can_constant_fold_through=True)  # type: ignore[arg-type]
def unpack(format: bytes | str, buffer: Buffer, /) -> tuple[Any, ...]:
    # [FUNCTION DOC] Polyfill of struct.unpack: delegates to the real struct.unpack and
    # returns the tuple of unpacked values. Raises struct.error on size/format mismatch.
    return struct.unpack(format, buffer)
