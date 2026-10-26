# [MODULE DOC] Defines CacheKeyStrategy, a frozen description of how Inductor caches
# hash ordered components into string keys (SHA256, base32 or hex, optional prefix and
# separator), plus named strategy instances for compact, code, fx_graph, system and
# autotune caches. Formats are kept for compatibility with existing cache layouts.
# No import-time side effects beyond constructing the instances.
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from typing import Any, Literal, Protocol
from typing_extensions import assert_never


CacheKeyComponent = str | bytes | bytearray | memoryview


class _HashLike(Protocol):
    # [CLASS DOC] HELPER: Structural type for hash objects offering update, digest and
    # hexdigest.
    # [FUNCTION DOC] HELPER: Protocol method; feeds bytes into the hash.
    def update(self, data: bytes, /) -> None: ...
    # [FUNCTION DOC] HELPER: Protocol method; returns the raw digest bytes.
    def digest(self) -> bytes: ...
    # [FUNCTION DOC] HELPER: Protocol method; returns the hex digest string.
    def hexdigest(self) -> str: ...


@dataclasses.dataclass(frozen=True)
class CacheKeyStrategy:
    # [CLASS DOC] Immutable recipe for building cache keys: name (debug label only),
    # digest_format (base32 or hex), prefix prepended to the key, optional byte
    # separator inserted between components, and base32_length (characters kept from the
    # base32 digest). Methods: digest, key (prefix plus digest) and key_from_json.
    """
    Describes how an Inductor cache turns stable components into a cache key.

    Different caches intentionally use different key formats for compatibility
    with existing on-disk and remote cache layouts. Keeping those choices in
    named strategies makes the composition explicit at the call site.
    """

    # Human-readable label for repr/debugging; it is not part of the cache key.
    name: str
    digest_format: Literal["base32", "hex"]
    prefix: str = ""
    separator: bytes | None = None
    base32_length: int = 51

    @staticmethod
    def _to_bytes(component: CacheKeyComponent) -> bytes:
        # [FUNCTION DOC] HELPER: Converts a str (UTF-8 encoded), bytes, bytearray or
        # memoryview into bytes; raises TypeError for any other type.
        if isinstance(component, str):
            return component.encode("utf-8")
        if isinstance(component, bytes):
            return component
        if isinstance(component, bytearray):
            return bytes(component)
        if isinstance(component, memoryview):
            return component.tobytes()
        raise TypeError(f"Unsupported cache key component: {type(component)!r}")

    def _hasher(self, components: tuple[CacheKeyComponent, ...]) -> _HashLike:
        # [FUNCTION DOC] HELPER: Builds a SHA256 hash fed with each component converted to
        # bytes, inserting the separator between consecutive components when configured.
        hasher = hashlib.sha256()
        for idx, component in enumerate(components):
            if idx > 0 and self.separator is not None:
                hasher.update(self.separator)
            hasher.update(self._to_bytes(component))
        return hasher

    def digest(self, *components: CacheKeyComponent) -> str:
        # [FUNCTION DOC] Hashes the components and returns the digest string: lowercase hex
        # for hex format, or the lowercase base32 text truncated to base32_length for base32
        # format. Raises TypeError for unsupported component types.
        hasher = self._hasher(components)
        if self.digest_format == "hex":
            return hasher.hexdigest()
        if self.digest_format == "base32":
            # [:51] strips the "Q====" suffix common to every SHA256 base32 digest.
            return (
                base64.b32encode(hasher.digest())[: self.base32_length]
                .decode("utf-8")
                .lower()
            )
        assert_never(self.digest_format)

    def key(self, *components: CacheKeyComponent) -> str:
        # [FUNCTION DOC] Returns the strategy prefix followed by the digest of the components.
        return f"{self.prefix}{self.digest(*components)}"

    def key_from_json(self, value: Any, *, sort_keys: bool = True) -> str:
        # [FUNCTION DOC] Serializes the value to JSON text (keys sorted by default) and
        # returns the key for that single text component. Raises if value is not serializable.
        return self.key(json.dumps(value, sort_keys=sort_keys))


COMPACT_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="compact",
    digest_format="base32",
)

CODE_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="code",
    digest_format="base32",
    prefix="c",
    separator=b"||",
)

FX_GRAPH_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="fx_graph",
    digest_format="base32",
    prefix="f",
)

SYSTEM_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="system",
    digest_format="hex",
)

AUTOTUNE_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="autotune",
    digest_format="hex",
    # Preserve the existing autotune cache format, which concatenates components.
    separator=None,
)
