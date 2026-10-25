# [MODULE DOC]
# Hashability support for Dynamo VariableTrackers. Provides raise_unhashable
# (raises an observed TypeError, or graph-breaks if Dynamo wrongly thinks a type
# is unhashable), is_hashable (lazy-aware hashability check), RawHash (wrapper
# that preserves a precomputed hash), and HashableTracker (wrapper making a
# VariableTracker usable as a dict key / set element for ConstDictVariable and
# SetVariable). No import-time side effects beyond imports.
"""
Hashability utilities for PyTorch Dynamo variable tracking.

This module provides the HashableTracker wrapper class and associated utilities
for making VariableTracker instances usable as dictionary keys and set elements
during symbolic execution. Used by both ConstDictVariable and SetVariable.
"""

from typing import TYPE_CHECKING

import torch

from .. import variables
from ..exc import raise_observed_exception
from ..utils import guard_if_dyn, specialize_symnode
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


def raise_unhashable(
    arg: VariableTracker, tx: "InstructionTranslatorBase | None" = None
) -> None:
    # [FUNCTION DOC]
    # Raises, inside the traced program, a TypeError "unhashable type" for the
    # given VariableTracker. If tx is None, the current instruction translator is
    # used. The type name comes from the tracker's python type when available,
    # otherwise from the tracker's own class name.
    # If the real python type is known and defines a hash, Dynamo's hashability
    # check disagrees with CPython, so instead of a wrong TypeError it triggers a
    # graph break (unimplemented, "Hashability mismatch").
    # Always exits by raising (observed exception or graph break); returns None
    # only nominally.
    if tx is None:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()

    try:
        arg_type = arg.python_type()
    except Exception:
        arg_type = None

    # Safety check: if we know the real Python type and it IS hashable,
    # our is_hashable() disagrees with CPython. Graph-break rather than
    # raising a wrong TypeError.
    if arg_type is not None and arg_type.__hash__ is not None:
        from .. import graph_break_hints
        from ..exc import unimplemented

        unimplemented(
            gb_type="Hashability mismatch",
            context=f"raise_unhashable {arg}",
            explanation=f"Dynamo thinks {arg_type.__name__} is unhashable, "
            f"but its __hash__ is not None. This likely indicates a missing "
            f"or incorrect is_hashable() override.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    type_name = arg_type.__name__ if arg_type is not None else type(arg).__name__
    raise_observed_exception(
        TypeError,
        tx,
        args=[
            f"unhashable type: '{type_name}'",
        ],
    )


def is_hashable(x: VariableTracker) -> bool:
    # [FUNCTION DOC]
    # Returns whether the tracker is hashable. For an unrealized lazy tracker whose
    # lazy hashability check succeeds, returns True without realizing it (avoids
    # installing guards); otherwise defers to the tracker's own is_hashable.
    # LazyVT optimization: check hashability without realizing the VT to avoid
    # accidentally inserting guards.
    if (
        isinstance(x, variables.LazyVariableTracker)
        and not x.is_realized()
        and x.is_hashable_lazy()
    ):
        return True

    return x.is_hashable()


class RawHash:
    # [CLASS DOC]
    # Immutable-by-convention holder (slot h) of an already computed integer hash.
    # Its hash returns the stored value unchanged and it equals only other
    # RawHash objects with the same value. Used to build tuples/frozensets of
    # per-item hashes without Python's integer hash reduction altering them.
    """Wraps a pre-computed hash value to bypass int.__hash__'s modular reduction.

    When building a tuple/frozenset of per-item hashes, using bare ints would
    apply long_hash (mod sys.hash_info.modulus), corrupting the values.
    Wrapping in RawHash makes tuplehash/frozenset_hash see the original hash.
    """

    __slots__ = ("h",)

    def __init__(self, h: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the precomputed hash integer h.
        self.h = h

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the stored hash value unchanged.
        return self.h

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: True only if other is a RawHash with an equal stored hash.
        return isinstance(other, RawHash) and self.h == other.h


class HashableTracker:
    # [CLASS DOC]
    # Wraps a VariableTracker so it can be a key in Python dicts/sets used by
    # ConstDictVariable and SetVariable. State: vt (the wrapped, possibly
    # SymNode-specialized tracker), _hash (hash computed at construction), and
    # _hash_is_identity (True when the hash is identity based, e.g. tensors, so
    # equality falls back to identity). Hash is fixed at construction; equality
    # mirrors CPython rich comparison on the traced values.
    """
    Class that wraps a VariableTracker and makes it hashable.
    Note that it's fine to put VTs into dictionaries and sets, but doing so
    does not take into account aliasing.
    """

    _MISSING = object()

    def __init__(self, vt: VariableTracker) -> None:
        # [FUNCTION DOC]
        # Wraps vt: specializes SymNodes, then computes the hash. Fast path: an
        # unrealized lazy tracker that is lazily hashable is hashed from its
        # original value without realization (no guards), with identity flag False.
        # Otherwise hashes via generic_hash_impl using the current translator,
        # which yields (hash, is-identity flag) and raises an observed TypeError
        # for unhashable values. Sets vt, _hash, _hash_is_identity.
        # We specialize SymNodes
        vt = specialize_symnode(vt)

        # Fast path for unrealized LazyVariableTrackers: check and hash without
        # realizing, to avoid inserting guards.  If the fast-path check fails,
        # fall through to realize the VT and try the full is_hashable check.
        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and vt.is_hashable_lazy()
        ):
            self._hash = hash(vt.original_value())
            self._hash_is_identity = False
            self.vt = vt
            return

        # Compute hash via the tp_hash slot (generic_hash_impl).
        # For unhashable types, hash_impl raises ObservedTypeError.
        from torch._dynamo.symbolic_convert import InstructionTranslator

        from .object_protocol import generic_hash_impl

        tx = InstructionTranslator.current_tx()
        # is_fake marks an identity-based hash (e.g. id(fake_tensor)); such VTs
        # have no python-constant value, so their key equality is identity.
        self._hash, self._hash_is_identity = generic_hash_impl(tx, vt)
        self.vt = vt

    @classmethod
    def _maybe_constant_torch_size(cls, vt: VariableTracker) -> object:
        # [FUNCTION DOC] HELPER: Tries to get a concrete torch.Size for vt.
        # Returns the original value for an unrealized lazy tracker holding a
        # torch.Size; for a SizeVariable, builds a torch.Size from items that are
        # python constants or single-element tensor trackers whose fake example
        # value carries a known constant. Returns the _MISSING sentinel if vt is
        # not a size or any item cannot be resolved.
        from .lists import SizeVariable
        from .tensor import TensorVariable

        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and isinstance(vt.original_value(), torch.Size)
        ):
            return vt.original_value()

        if not isinstance(vt, SizeVariable):
            return cls._MISSING

        items = []
        for item in vt.items:
            if item.is_python_constant():
                items.append(item.as_python_constant())
                continue

            if isinstance(item, TensorVariable):
                proxy = getattr(item, "proxy", None)
                node = getattr(proxy, "node", None)
                meta = getattr(node, "meta", None) if node is not None else None
                example_value = (
                    meta.get("example_value") if isinstance(meta, dict) else None
                )
                constant = getattr(example_value, "constant", None)

                if isinstance(constant, torch.Tensor) and constant.numel() == 1:
                    items.append(constant.item())
                    continue

            return cls._MISSING

        return torch.Size(items)

    def __hash__(self) -> int:
        # [FUNCTION DOC] Returns the hash computed at construction.
        return self._hash

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality of wrapped trackers. False for non-HashableTracker; True if the
        # same tracker object. If both resolve to concrete torch.Size values,
        # compares them. If either is a TensorVariable, equal only when both hashes
        # are identity based and equal (never emits an elementwise eq into the
        # graph). Otherwise runs generic_richcompare_bool (may run user __eq__ and
        # observe its exceptions); a constant result is used directly, a symnode
        # result is guarded on, anything else falls back to identity-hash equality.
        # Side effects: may install guards or trace user code.
        """
        Checks equality between two HashableTracker instances.

        Mirrors CPython's PyObject_RichCompareBool: routes the comparison
        through generic_richcompare_bool so any user-defined __eq__ runs.

        Args:
            other: Another HashableTracker instance to compare with

        Returns:
            True if the underlying variable trackers are Python-equal, False otherwise
        """
        if not isinstance(other, HashableTracker):
            return False
        if self.vt is other.vt:
            return True

        self_constant = self._maybe_constant_torch_size(self.vt)
        other_constant = self._maybe_constant_torch_size(other.vt)
        if self_constant is not self._MISSING and other_constant is not self._MISSING:
            return self_constant == other_constant

        # Tensor keys hash by identity (Tensor.__hash__ is id(self)), so CPython
        # only ever compares identical tensors during a dict/set lookup; a
        # tensor's elementwise __eq__ is never consulted for membership. Mirror
        # that with an identity comparison instead of running __eq__, which would
        # otherwise emit a stray elementwise-eq node into the FX graph and
        # corrupt the surrounding trace.
        from .tensor import TensorVariable

        if isinstance(self.vt, TensorVariable) or isinstance(other.vt, TensorVariable):
            return (
                self._hash_is_identity
                and other._hash_is_identity
                and self._hash == other._hash
            )

        # All other keys: mirror PyObject_RichCompareBool and run the
        # comparison through generic_richcompare_bool so any user-defined __eq__
        # runs and its result (or any exception it raises) is observed by the
        # traced program.
        from ..symbolic_convert import InstructionTranslator
        from .object_protocol import generic_richcompare_bool

        tx = InstructionTranslator.current_tx()

        result = generic_richcompare_bool(tx, self.vt, other.vt, op="__eq__")
        if result.is_python_constant():
            return bool(result.as_python_constant())

        if result.is_symnode_like():
            return bool(guard_if_dyn(result))

        # Comparison did not resolve to a constant (e.g. keys whose __eq__ could
        # not be determined). Fall back to identity-based hash equality, which
        # mirrors CPython treating such keys as equal only when identical.
        return (
            self._hash_is_identity
            and other._hash_is_identity
            and self._hash == other._hash
        )
