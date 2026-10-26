# [MODULE DOC] Defines IndexedDict, an insertion-ordered mapping that also supports constant-time
# lookup of the next/previous key relative to a given key. Used by AOTAutograd for fast
# neighbor-key access. No import-time side effects beyond defining two type variables and the class.
from collections.abc import Iterator, MutableMapping
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


# Used for fast next key access (using the fact that the dict is ordered)
# Note: doesn't support deletion but we don't need it!
class IndexedDict(MutableMapping[K, V], Generic[K, V]):
    # [CLASS DOC] Insertion-ordered mutable mapping. State: a backing dict of values, a list of keys in
    # insertion order, and a mapping from key to its position in that list. Invariant: the key list and
    # position mapping always contain exactly the keys of the backing dict. Deletion is unsupported.
    # Key methods: next_key and prev_key return the neighbor key in insertion order in constant time.
    """A dict that maintains insertion order with O(1) index access."""

    __slots__ = ("_dict", "_keys", "_key_to_index")

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates an empty mapping with an empty backing dict, key list and key-to-position map.
        self._dict: dict[K, V] = {}
        self._keys: list[K] = []  # typing: ignore[bad-override]
        self._key_to_index: dict[K, int] = {}

    def __setitem__(self, key: K, value: V) -> None:
        # [FUNCTION DOC] Stores value under key. A new key is appended to the key list and its position
        # recorded; an existing key keeps its original position and only has its value replaced.
        if key not in self._dict:
            self._key_to_index[key] = len(self._keys)
            self._keys.append(key)
        self._dict[key] = value

    def __getitem__(self, key: K) -> V:
        # [FUNCTION DOC] Returns the value for key; raises KeyError if the key is absent.
        return self._dict[key]

    def __delitem__(self, key: K) -> None:
        # [FUNCTION DOC] Always raises NotImplementedError; deletion is intentionally unsupported.
        raise NotImplementedError("Deletion not supported for IndexedDict")

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the number of stored keys.
        return len(self._dict)

    def __iter__(self) -> Iterator[K]:
        # [FUNCTION DOC] Returns an iterator over keys in insertion order.
        return iter(self._keys)

    def __contains__(self, key: object) -> bool:
        # [FUNCTION DOC] Returns whether key is present (hash lookup in the backing dict).
        return key in self._dict

    def next_key(self, key: K) -> K | None:
        # [FUNCTION DOC] Returns the key inserted immediately after key, or None if key is the last
        # key or is not in the mapping. Constant time.
        """Get the next key in insertion order. O(1)."""
        idx = self._key_to_index.get(key)
        if idx is not None and idx + 1 < len(self._keys):
            return self._keys[idx + 1]
        return None

    def prev_key(self, key: K) -> K | None:
        # [FUNCTION DOC] Returns the key inserted immediately before key, or None if key is the first
        # key or is not in the mapping. Constant time.
        """Get the previous key in insertion order. O(1)."""
        idx = self._key_to_index.get(key)
        if idx is not None and idx > 0:
            return self._keys[idx - 1]
        return None
