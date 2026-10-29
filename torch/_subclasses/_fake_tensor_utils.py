# [MODULE DOC] Helper types for the FakeTensor dispatch cache. They represent SymInt/SymNode values inside
# cache keys and cached outputs in a form that can be hashed, compared by value, and stored
# without keeping a ShapeEnv alive. Contains the deconstructed SymNode/SymType wrappers,
# input back-reference and input/output stubs, and the key-building state object.
# No import-time side effects (type-only imports are guarded by TYPE_CHECKING).
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import SymInt
from torch.fx.experimental.sym_node import SymNode
from torch.types import py_sym_types, PySymType


if TYPE_CHECKING:
    import sympy

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from .fake_tensor import _DispatchCacheKey, _MetadataIntLike


@dataclass(frozen=True, slots=True)
class _DeconstructedSymNode:
    # [CLASS DOC] HELPER: Frozen snapshot of a SymNode (expression, python type, hint, constant, fx node)
    # without its ShapeEnv, so cached entries do not create ShapeEnv reference cycles.
    # Mirrors the SymNode field protocol; converted back with extract given a ShapeEnv.
    """
    Represents a SymNode without the associated ShapeEnv
    """

    # n.b. keep the same protocol as SymNode
    _expr: sympy.Expr
    pytype: type
    _hint: int | float | bool | None
    constant: int | float | bool | None
    fx_node: torch.fx.Node

    @staticmethod
    def from_node(node: SymNode) -> _DeconstructedSymNode:
        # [FUNCTION DOC] HELPER: Build a deconstructed node by copying the fields of a SymNode and dropping its ShapeEnv.
        return _DeconstructedSymNode(
            node._expr,
            node.pytype,
            # pyrefly: ignore[bad-argument-type]
            node._hint,
            node.constant,
            # pyrefly: ignore[bad-argument-type]
            node.fx_node,
        )

    def extract(self, shape_env: ShapeEnv) -> SymNode:
        # [FUNCTION DOC] HELPER: Rebuild a real SymNode from the stored fields and the supplied ShapeEnv.
        return SymNode(
            self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node
        )

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: String form is the string of the symbolic expression.
        return str(self._expr)

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug representation listing expression, type, hint, constant and fx node.
        return f"_DeconstructedSymNode{{{self._expr!r}, {self.pytype!r}, {self._hint!r}, {self.constant!r}, {self.fx_node!r}}}"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Equality is intentionally unsupported; raises NotImplementedError (use _value_eq).
        raise NotImplementedError

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hashing is intentionally unsupported; raises NotImplementedError (use _value_hash).
        raise NotImplementedError

    # _value_eq to match SymNode
    def _value_eq(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Value equality matching SymNode: true only when other is a SymNode or deconstructed node
        # with equal expression, type, hint, constant and fx node; false for any other object.
        if isinstance(other, (SymNode, _DeconstructedSymNode)):
            return (
                self._expr == other._expr
                and self.pytype == other.pytype
                and self._hint == other._hint
                and self.constant == other.constant
                and self.fx_node == other.fx_node
            )
        else:
            return False

    # _value_hash to match SymNode
    def _value_hash(self) -> int:
        # [FUNCTION DOC] HELPER: Value hash over expression, type, hint, constant and fx node; consistent with _value_eq.
        return hash((self._expr, self.pytype, self._hint, self.constant, self.fx_node))


@dataclass(frozen=True, slots=True)
class _DeconstructedSymType:
    # [CLASS DOC] HELPER: Frozen snapshot of a SymInt/SymFloat/SymBool: the symbolic python type plus a
    # deconstructed node (no ShapeEnv). Rebuilt via extract with a ShapeEnv.
    """
    Represents a SymInt, SymFloat, SymBool without the associated ShapeEnv
    """

    ty: type[PySymType]
    node: _DeconstructedSymNode

    @staticmethod
    def from_sym_type(value: PySymType) -> _DeconstructedSymType:
        # [FUNCTION DOC] HELPER: Capture the type of the given symbolic value and its node (deconstructed form built by caller conventions).
        # KNOWN ISSUE: stores value.node directly, which is a SymNode, not a _DeconstructedSymNode as the field annotation says.
        # pyrefly: ignore [bad-argument-type]
        return _DeconstructedSymType(type(value), value.node)

    def extract(self, shape_env: ShapeEnv) -> PySymType:
        # [FUNCTION DOC] HELPER: Rebuild a symbolic value by extracting the node with the given ShapeEnv and wrapping it in the stored type.
        return self.ty(self.node.extract(shape_env))

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: String form shows the symbolic type and node.
        return f"{self.ty}({self.node})"

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug representation with type and node.
        return f"_DeconstructedSymType({self.ty}, {self.node!r})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Returns NotImplemented rather than raising, so comparisons fall back to identity.
        # KNOWN ISSUE: unlike the sibling classes it does not raise; the frozen dataclass machinery is bypassed.
        return NotImplemented

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns NotImplemented instead of an integer hash.
        # KNOWN ISSUE: hash() on an instance will raise TypeError because a non-integer is returned.
        return NotImplemented


@dataclass(frozen=True, slots=True)
class _InputBackref:
    # [CLASS DOC] HELPER: Frozen marker inside a cache key meaning 'same SymNode as the earlier key element at index value'.
    # Used to preserve aliasing between symbolic inputs.
    value: int


@dataclass(slots=True)
class _PySymInputStub:
    # [CLASS DOC] HELPER: Wrapper for a symbolic int in a cache key, giving it value-based hash/eq (plain SymInt uses
    # identity). Holds a live symbolic value, a deconstructed one (when stored in the cache), or an
    # _InputBackref to an earlier key entry.
    """
    Represents a SymInt in the cached key. Needed because SymInt doesn't
    support __eq__ or __hash__ directly.
    """

    # value can be:
    #   PySymType: This is the 'normal' SymInt value, wrapped so we can use
    #              hash/eq as value hash/eq (normally SymInt does object
    #              hash/eq).
    #   _DeconstructedSymType: This is used when storing the _PySymInputStub in
    #                          the cache to avoid cyclic ShapeEnv references.
    #   _InputBackref: This is a back-reference to a previous _PySymInputStub in
    #                  the key.
    value: PySymType | _DeconstructedSymType | _InputBackref

    def __init__(
        self, value: PySymType | _DeconstructedSymType | _InputBackref
    ) -> None:
        # [FUNCTION DOC] HELPER: Store the value unchanged; inputs keep the original symbolic value so it can be reused as an output.
        # For inputs (values in the `key`) we need to keep the PySymType intact
        # - this way if we need to reuse it as an output we can properly copy
        # the original value.
        self.value = value

    def strip_shape_env(self) -> None:
        # [FUNCTION DOC] HELPER: If holding a live symbolic value, replace it with its deconstructed form so no ShapeEnv is retained. Mutates self.
        if isinstance(self.value, py_sym_types):
            self.value = _DeconstructedSymType.from_sym_type(self.value)

    def extract(self, shape_env: ShapeEnv) -> PySymType:
        # [FUNCTION DOC] HELPER: Return the symbolic value: rebuild it with the ShapeEnv if deconstructed, else return the stored one.
        # Raises AssertionError if the value is a back-reference (callers must use the original entry).
        if isinstance(self.value, _DeconstructedSymType):
            return self.value.extract(shape_env)
        else:
            # We should never see an _InputBackref here - anyone extracting a
            # value should be pulling from the original entry (the one this
            # backref points at).
            if isinstance(self.value, _InputBackref):
                raise AssertionError(
                    "Cannot extract value from _InputBackref - use the original entry"
                )
            return self.value

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: String form is the string of the wrapped value.
        return str(self.value)

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug representation of the wrapped value.
        return f"_PySymInputStub({self.value!r})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Equal only to another stub. Back-references compare by their value; otherwise the underlying nodes
        # are compared by value (_value_eq).
        if not isinstance(other, _PySymInputStub):
            return False
        elif isinstance(self.value, _InputBackref) or isinstance(
            other.value, _InputBackref
        ):
            return self.value == other.value  # type: ignore[bad-return]
        else:
            # pyrefly: ignore [bad-argument-type]
            return self.value.node._value_eq(other.value.node)

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hash of the back-reference, or the value hash of the underlying node; consistent with equality.
        if isinstance(self.value, _InputBackref):
            return hash(self.value)
        else:
            return self.value.node._value_hash()


@dataclass(slots=True)
class _SymIntOutputStub:
    # [CLASS DOC] HELPER: Describes a symbolic int in a cached output: either an index into the cache key to copy the
    # SymNode from (output aliases an input) or a deconstructed node to rebuild.
    """
    Represents a SymInt in the cached output.
    """

    # This is either an `int` which represents the index in the key to copy the
    # SymNode from or it's the deconstructed SymNode itself.
    value: int | _DeconstructedSymNode

    def __init__(self, value: SymInt, key_path: int | None) -> None:
        # [FUNCTION DOC] HELPER: If key_path is None, store the deconstructed node of the SymInt; otherwise store the key index to copy from.
        if key_path is None:
            self.value = _DeconstructedSymNode.from_node(value.node)
        else:
            self.value = key_path

    def extract(self, key: _DispatchCacheKey, shape_env: ShapeEnv) -> SymInt:
        # [FUNCTION DOC] HELPER: Produce the SymInt: rebuild from the stored node with the ShapeEnv, or fetch the SymInt held by the referenced key entry.
        # Raises AssertionError if the referenced key entry is not a stub holding a SymInt.
        if isinstance(self.value, _DeconstructedSymNode):
            return SymInt(self.value.extract(shape_env))
        else:
            src = key.key[self.value]
            if not isinstance(src, _PySymInputStub) or not isinstance(
                src.value, SymInt
            ):
                raise AssertionError(
                    f"Expected _PySymInputStub with SymInt value, got {type(src)} "
                    f"with {type(src.value) if isinstance(src, _PySymInputStub) else 'N/A'}"
                )
            return src.value

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug representation of the stored value.
        return f"_SymIntOutputStub({self.value!r})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Equality is unsupported; raises NotImplementedError.
        raise NotImplementedError

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hashing is unsupported; raises NotImplementedError.
        raise NotImplementedError


@dataclass(slots=True)
class _CacheKeyState:
    # [CLASS DOC] HELPER: Mutable state while building one dispatch cache key. Tracks id(SymNode) to key index (for
    # aliasing detection), the set of sympy symbols seen on input, and the ShapeEnv found (or given).
    """
    State used while building our cache key.
    """

    # We track the SymNodes so when we get the output we can see if it exactly
    # matches one of the inputs so we can uncache it properly.
    sym_node_lookup: dict[int, int]  # id(SymNode) -> index

    # This is a list of all seen input sympy.Symbols. We use it when building
    # the cache entry to see if the output value has any symbols that we didn't
    # see on input. See _has_unrepresented_symbols().
    known_symbols: set[sympy.Symbol]

    # There are cases where we're asked to perform an op when we have no
    # ShapeEnv on the FakeTensorMode - but for SymNodes we MUST have a
    # ShapeEnv. So as we scan if we see a SymNode (with a ShapeEnv) we record it
    # here.
    shape_env: ShapeEnv | None

    def __init__(self, shape_env: ShapeEnv | None = None) -> None:
        # [FUNCTION DOC] HELPER: Start with empty lookup and symbol set and the optional ShapeEnv.
        self.sym_node_lookup = {}
        self.known_symbols = set()
        self.shape_env = shape_env

    def cache_on_shape_env(self) -> bool:
        # [FUNCTION DOC] HELPER: True if any SymNode was seen, meaning the entry must be cached on the ShapeEnv rather than globally,
        # because cached results may depend on guards specific to that ShapeEnv.
        """
        Returns true if the CacheKey needs to be cached on the ShapeEnv
        rather than the global cache.

        If our inputs contain a SymNode then we can't cache this operation on
        the global cache because the cached output will implicitly depend on
        guard values which might not be true on some other ShapeEnv. So unless
        we're also going to cache the guards we need to cache this operation on
        the ShapeEnv instead of globally.
        """
        return bool(self.sym_node_lookup)

    def convert_sym_int(self, result: list[object], arg: SymInt) -> None:
        # [FUNCTION DOC] HELPER: Append the key element for a SymInt to result: a back-reference if its node was seen already,
        # otherwise record its position, add its free symbols to known_symbols, remember its ShapeEnv if none, and append a stub.
        node_id = id(arg.node)
        if node_id in self.sym_node_lookup:
            result.append(_InputBackref(self.sym_node_lookup[node_id]))
        else:
            self.sym_node_lookup[node_id] = len(result)
            self.known_symbols.update(arg.node.expr.free_symbols)
            if self.shape_env is None:
                self.shape_env = arg.node.shape_env
            result.append(_PySymInputStub(arg))

    def convert_output(self, arg: _MetadataIntLike) -> _MetadataIntLike:
        # [FUNCTION DOC] HELPER: Convert an output metadata int: SymInt becomes an output stub (pointing to the matching input index if the node was an input),
        # anything else is returned unchanged.
        if isinstance(arg, SymInt):
            return _SymIntOutputStub(arg, self.sym_node_lookup.get(id(arg.node), None))
        else:
            return arg
