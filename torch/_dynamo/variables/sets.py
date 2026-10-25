# [MODULE DOC] Dynamo variable trackers for set-like Python values: set (SetVariable),
# frozenset (FrozensetVariable), OrderedSet (OrderedSetVariable, plus
# OrderedSetClassVariable for the class object) and dict_keys views
# (DictKeySetVariable). Elements are stored as an insertion-ordered mapping from
# HashableTracker wrappers to a None placeholder. Also holds set type-check and copy
# helpers. No import-time side effects beyond class definitions.
"""
Set-related variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for different types of set-like objects:
- Regular Python sets (set)
- Frozen sets (frozenset)
- Ordered sets (torch.utils._ordered_set.OrderedSet)
- Dictionary key sets (dict_keys views used as sets)

These classes are responsible for tracking set operations during graph compilation,
maintaining proper guards for set mutations and element existence checks.

The implementation uses a special HashableTracker wrapper to handle set elements
while preserving proper aliasing semantics. Sets are modeled internally as
dictionaries with None values.
"""

import functools
import operator
from collections.abc import Iterable, Iterator
from typing import Any, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, raise_type_error
from ..guards import GuardBuilder, install_guard
from ..source import is_constant_source, is_from_local_source
from ..utils import (
    _item_debug_repr,
    cmp_name_to_op_mapping,
    istype,
    lazily_unpack,
    raise_args_mismatch,
    set_methods,
    tracked_repr,
    unpack_iterable,
)
from .base import Member, Method, ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .hashable import HashableTracker, is_hashable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


# [Adding a new supported class within the keys of SetVariable]
# see steps outlined for ConstDictVariable


def pyanyset_check(obj: VariableTracker) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the tracked value's Python type is a set or
    # frozenset (or subclass). Mirrors the C-level any-set type check.
    return issubclass(obj.python_type(), (set, frozenset))


def pyset_check(obj: VariableTracker) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the tracked value's Python type is a mutable
    # set (or subclass), False for frozenset.
    # ref: https://github.com/python/cpython/blob/v3.13.0/Include/setobject.h#L36-L38
    return issubclass(obj.python_type(), set)


def set_copy(obj: VariableTracker) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Creates a fresh, sourceless set tracker holding a shallow copy of
    # the input's items mapping, marked as newly created (ValueMutationNew). For user-defined
    # set subclasses the underlying base tracker is copied. Raises an error if that base is
    # missing. Used as scratch storage for binary operators so inputs are never mutated.
    """Mirrors CPython's internal `set_copy` (Objects/setobject.c).

    Always allocates a fresh set/frozenset with a shallow-copied items dict.
    Distinct from the user-visible `.copy()` method, which preserves identity
    for exact frozenset (`frozenset_copy`).  Use this for binary-op scratch
    storage so mutations don't bleed into the input.
    """
    base = obj._base_vt if isinstance(obj, variables.UserDefinedSetVariable) else obj
    if base is None:
        raise AssertionError("_base_vt must not be None")
    return base.clone(
        items=base.items.copy(),  # type: ignore[missing-attribute]
        mutation_type=ValueMutationNew(),
        source=None,
    )


class SetVariable(VariableTracker):
    # [CLASS DOC] Tracker for a Python set. State: items (insertion-ordered mapping of
    # HashableTracker to a None placeholder), original_items (snapshot at creation, used to
    # detect changes), should_reconstruct_all (True when the set must be rebuilt wholesale
    # after mutation or when it has no local source). Methods are exposed via tp_methods and
    # the nb_/sq_/tp_ protocol hooks; mutations are recorded through the side-effects tracker.
    # Subclasses cover frozenset, OrderedSet and dict_keys.
    """Represents a Python set during symbolic execution."""

    # PySet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2436
    _cpython_type = set

    CONTAINS_GUARD = GuardBuilder.SET_CONTAINS
    NOT_CONTAINS_GUARD = GuardBuilder.SET_NOT_CONTAINS

    def __init__(
        self,
        items: Iterable[VariableTracker | HashableTracker],
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Builds the tracker from an iterable of VariableTrackers or
        # HashableTrackers. Plain trackers are realized (installing aliasing guards) and wrapped
        # in a HashableTracker; the mapping preserves order and drops duplicates. Discards
        # clone-passed original_items/should_reconstruct_all kwargs, then recomputes
        # should_reconstruct_all (True unless sourced from a local) and snapshots original_items.
        # .clone() passes these arguments in kwargs but they're recreated below
        if "original_items" in kwargs:
            kwargs.pop("original_items")
        if "should_reconstruct_all" in kwargs:
            kwargs.pop("should_reconstruct_all")

        super().__init__(**kwargs)

        # Items can be either VariableTrackers or HashableTrackers (from set ops).
        # For VariableTrackers, realize them to ensure aliasing guards are installed
        # when the same object appears multiple times.
        hashable_items = []
        for item in items:
            if isinstance(item, HashableTracker):
                # Already a HashableTracker from a set operation
                hashable_items.append(item)
            else:
                # VariableTracker - realize to install guards, then wrap
                # pyrefly: ignore [bad-argument-type]
                hashable_items.append(HashableTracker(item.realize()))
        # Internal representation as dict allows for simple integration with
        # OrderedSet, notably polyfills. Using set moves complexity to OrderedSet
        self.items = dict.fromkeys(hashable_items, SetVariable._default_value())
        self.should_reconstruct_all = (
            not is_from_local_source(self.source) if self.source else True
        )
        self.original_items = dict.fromkeys(
            hashable_items, SetVariable._default_value()
        )

    def debug_repr(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a debug string like a set literal of the item debug
        # reprs, or the empty-set form when there are no items.
        if not self.items:
            return "set()"
        else:
            items: list[str] = []
            for v in self.items:
                vt = v.vt if isinstance(v, HashableTracker) else v
                val_str = _item_debug_repr(vt)
                items.append(val_str)
            return "{" + ", ".join(items) + "}"

    @property
    def set_items(self) -> set["HashableTracker"]:
        # [FUNCTION DOC] Property: returns a real (unordered) set of the HashableTracker keys.
        # Insertion order is lost; use items when order matters.
        return set(self.items.keys())

    @staticmethod
    def _default_value() -> VariableTracker:
        # [FUNCTION DOC] HELPER: Returns a fresh None constant tracker used as the mapping value
        # for every set element.
        # Variable to fill in the keys of the dictionary
        return ConstantVariable.create(None)

    def as_proxy(self) -> Any:
        # [FUNCTION DOC] Returns a real Python set of the proxies of every element.
        return {k.vt.as_proxy() for k in self.set_items}

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the set type.
        return set

    def is_python_constant(self) -> bool:
        # [FUNCTION DOC] Returns True iff every element is a Python constant. Checks elements
        # directly so no real set is built and no element hash is invoked.
        # Avoid the base implementation, which probes as_python_constant() and
        # thus rebuilds a real set, re-hashing the elements (wrong for elements
        # with a side-effecting __hash__).  Check element constness directly.
        return all(k.vt.is_python_constant() for k in self.set_items)

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns a real Python set of the element constants. Raises if any
        # element is not a constant.
        return {k.vt.as_python_constant() for k in self.set_items}

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] Implements repr: returns a string constant tracker with the empty-set
        # form for no items, else a brace-delimited list of tracked element reprs. Element
        # order follows the unordered set_items view.
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L763-L822
        if not self.items:
            return VariableTracker.build(tx, f"{self.python_type_name()}()")
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, "{" + items + "}")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that loads each element then builds a new set from
        # them (element order unspecified).
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_SET", arg=len(self.set_items)))

    def __contains__(self, vt: VariableTracker) -> bool:
        # [FUNCTION DOC] Python-level membership test on the tracker (not a traced contains).
        # Returns False for unhashable trackers, otherwise whether it is a key of items.
        # Raises an error if the argument is not a VariableTracker.
        if not isinstance(vt, VariableTracker):
            raise AssertionError(f"Expected VariableTracker, got {type(vt)}")
        # Use is_hashable as a side-effect-free pre-check.  We can't catch
        # ObservedTypeError from HashableTracker because it modifies
        # tx.exn_vt_stack as a side effect.
        if not is_hashable(vt):
            return False
        key = HashableTracker(vt)
        return key in self.items

    def has_new_items(self) -> bool:
        # [FUNCTION DOC] Returns True if the set must be rebuilt: either should_reconstruct_all
        # is set, or some current value differs (by identity after realization) from the
        # snapshot taken in original_items.
        return self.should_reconstruct_all or any(
            # pyrefly: ignore [bad-argument-type]
            self.is_new_item(self.original_items.get(key.vt), value)
            for key, value in self.items.items()
        )

    def is_new_item(
        self, value: VariableTracker | None, other: VariableTracker
    ) -> bool:
        # [FUNCTION DOC] Returns True if the two trackers are different objects; compares the
        # realized trackers when both are realized, otherwise the raw trackers. A missing
        # value (None) always counts as new.
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        # [FUNCTION DOC] Returns the element trackers as a list in insertion order.
        return [x.vt for x in self.items]

    def clone(self, **kwargs: Any) -> VariableTracker:
        # [FUNCTION DOC] Clones the tracker with overridden fields. If the new mutation type
        # marks the object as newly created, the source is forced to None so the copy is
        # sourceless.
        from torch._dynamo.variables.base import AttributeMutationNew, ValueMutationNew

        if isinstance(
            kwargs.get("mutation_type"), (ValueMutationNew, AttributeMutationNew)
        ):
            kwargs["source"] = None
        return super().clone(**kwargs)

    def is_hashable(self) -> bool:
        # [FUNCTION DOC] Mutable sets are unhashable: always False.
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Raises an observed TypeError (unhashable type) in the traced program.
        from ..exc import raise_type_error

        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # [FUNCTION DOC] Returns a bool constant: whether the set's Python type has the named
        # attribute.
        return VariableTracker.build(tx, hasattr(self.python_type(), name))

    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        # [FUNCTION DOC] Installs a guard recording whether args[0] was in the sourced set.
        # No-op when the set has no source, was modified during tracing, or the argument is
        # sourced or non-constant. Chooses the contains or not-contains guard by the current
        # membership result. Side effect: registers a guard.
        if not self.source:
            return

        if tx.output.side_effects.is_modified(self):
            return

        contains = args[0] in self
        if args[0].source is None and args[0].is_python_constant():
            guard_fn = (
                type(self).CONTAINS_GUARD if contains else type(self).NOT_CONTAINS_GUARD
            )
            install_guard(
                self.make_guard(
                    functools.partial(
                        guard_fn,
                        key=args[0].as_python_constant(),
                    )
                )
            )

    def _fast_set_method(
        self,
        tx: "InstructionTranslatorBase",
        fn: Any,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Calls a real set method on the constant values of self and
        # the arguments and wraps the result as a tracker. Any exception from the real call
        # is re-raised as an observed exception in the traced program.
        try:
            res = fn(
                *[x.as_python_constant() for x in [self, *args]],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
        except Exception as exc:
            raise_observed_exception(type(exc), tx, args=list(exc.args))
        return VariableTracker.build(tx, res)

    def _iter_operand_keys(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> "Iterator[HashableTracker]":
        # [FUNCTION DOC] HELPER: Generator yielding HashableTracker keys for a set-operation
        # operand. Set and dict operands reuse their keys directly; other iterables are
        # unpacked lazily one element at a time and wrapped (unhashable elements raise an
        # observed TypeError).
        # Lazily yield HashableTracker keys for a set-operation operand, one
        # element at a time so callers that short-circuit (isdisjoint) observe
        # the same generator side effects as CPython. A set or dict operand's
        # keys are reused directly (no re-hashing), mirroring CPython's
        # set_update_internal fast path for set/frozenset/dict operands.
        # HashableTracker raises ObservedTypeError for an unhashable element,
        # matching CPython's hash-at-insert behavior.
        from .dicts import ConstDictVariable

        if isinstance(other, (SetVariable, ConstDictVariable)):
            yield from other.items.keys()
            return

        yield from (HashableTracker(item) for item in lazily_unpack(tx, other))

    def _operand_keys(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> list[HashableTracker]:
        # [FUNCTION DOC] HELPER: Eager list version of the operand-key generator; set and dict
        # operands give their keys, other iterables are fully unpacked and wrapped.
        # Eager variant: unpack_iterable takes the fast unpack_var_sequence
        # path for builtin iterables instead of the per-element iterator
        # protocol.
        from .dicts import ConstDictVariable

        if isinstance(other, (SetVariable, ConstDictVariable)):
            return list(other.items.keys())
        return [HashableTracker(x) for x in unpack_iterable(tx, other)]

    def _new_set(self, items: "Iterable[HashableTracker]") -> "SetVariable":
        # [FUNCTION DOC] HELPER: Returns a new newly-created tracker of the same class as self
        # holding the given keys in iteration order.
        # Build a fresh set of the same concrete type (set / frozenset /
        # OrderedSet). list() preserves insertion order, which matters for
        # OrderedSet.
        return type(self)(list(items), mutation_type=ValueMutationNew())

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the in operator. An unhashable item raises an observed
        # TypeError unless it is a set, in which case it is converted to a frozenset first.
        # Installs a contains guard, then returns a bool constant tracker.
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2131-L2149
        if not is_hashable(item):
            # Mirror CPython's set_contains: if hashing fails with TypeError due to
            # an unhashable set, coerce the key to frozenset and retry.
            # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2151-L2159
            if not pyset_check(item):
                raise_type_error(tx, f"unhashable type: '{item.python_type_name()}'")
            # CPython NOTE:
            # Note that 'key' could be a set() or frozenset() object.  Unlike most
            # container types, set allows membership testing with a set key, even
            # though it is not hashable.
            item = FrozensetVariable(item.items)  # type: ignore[missing-attribute]
        self.install_set_contains_guard(tx, [item])
        contains = item in self
        return VariableTracker.build(tx, contains)

    def _try_fast_set_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Fast path for set methods. When no argument is a set or dict
        # tracker, all arguments are constants and self is an exact set or frozenset, runs the
        # real method on constants and returns the result tracker; otherwise returns None so
        # the caller uses the slow tracker-based path.
        from ..utils import check_constant_args
        from .dicts import ConstDictVariable

        # Constant, exact-builtin fast path: materialize and call the real
        # method. Set/dict operands take the slow VT path below so their keys
        # are not re-hashed (CPython's do-not-rehash-dict-keys fast path);
        # OrderedSet is excluded because as_python_constant() loses insertion
        # order (it routes through the unordered set_items).
        if (
            not any(isinstance(a, (SetVariable, ConstDictVariable)) for a in args)
            and check_constant_args(args, kwargs)
            and self.python_type() in (set, frozenset)
        ):
            py_type = self.python_type()
            return self._fast_set_method(tx, getattr(py_type, name), args, kwargs)
        return None

    def add(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.add: records a mutation and inserts args[0] (hashed; unhashable
        # raises an observed TypeError). Returns None constant.
        # Convert add to __setitem__ with None value
        tx.output.side_effects.mutation(self)
        self.items[HashableTracker(args[0])] = SetVariable._default_value()
        return ConstantVariable.create(None)

    def pop(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.pop: removes and returns an element. Raises an observed KeyError
        # when empty. Marks the set for full reconstruction and records a mutation.
        # KNOWN ISSUE: the element is chosen from an unordered real set of trackers, so which
        # item is popped is arbitrary rather than matching CPython order.
        # Choose an item at random and pop it
        try:
            result: VariableTracker = self.set_items.pop().vt  # type: ignore[assignment]
        except KeyError as e:
            raise_observed_exception(KeyError, tx, args=list(e.args))
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.pop(HashableTracker(result))
        return result

    def isdisjoint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.isdisjoint: bool constant, True if no operand element is in self.
        # Uses the constant fast path when possible, else iterates operand keys lazily and
        # stops at the first match.
        fast = self._try_fast_set_method(tx, "isdisjoint", args, kwargs)
        if fast is not None:
            return fast
        for key in self._iter_operand_keys(tx, args[0]):
            if key in self.items:
                return ConstantVariable.create(False)
        return ConstantVariable.create(True)

    def intersection(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.intersection: returns a new set of the same class holding elements
        # present in self and all argument iterables. Constant fast path first; no mutation.
        fast = self._try_fast_set_method(tx, "intersection", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            other_keys = set(self._operand_keys(tx, other))
            out_items = {k: v for k, v in out_items.items() if k in other_keys}
        return self._new_set(out_items)

    def intersection_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.intersection_update: keeps only elements also in every argument
        # iterable, replacing contents in place. Records a mutation and forces full
        # reconstruction. Returns None constant.
        kept = dict(self.items)
        for other in args:
            other_keys = set(self._operand_keys(tx, other))
            kept = {k: v for k, v in kept.items() if k in other_keys}
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        self.items.clear()
        self.items.update(kept)
        return ConstantVariable.create(None)

    def union(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.union: returns a new set of the same class with self's elements
        # followed by new elements from each argument iterable. Constant fast path first.
        fast = self._try_fast_set_method(tx, "union", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            for key in self._operand_keys(tx, other):
                out_items.setdefault(key, SetVariable._default_value())
        return self._new_set(out_items)

    def difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.difference: returns a new set of the same class without any
        # element found in the argument iterables. Constant fast path first; no mutation.
        fast = self._try_fast_set_method(tx, "difference", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            for key in self._operand_keys(tx, other):
                out_items.pop(key, None)
        return self._new_set(out_items)

    def difference_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.difference_update: removes in place every element found in the
        # argument iterables. Records a mutation, forces full reconstruction, returns None
        # constant.
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        for other in args:
            for key in self._operand_keys(tx, other):
                self.items.pop(key, None)
        return ConstantVariable.create(None)

    def symmetric_difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.symmetric_difference: returns a new set of the same class with
        # elements in exactly one of self and args[0]. Constant fast path first.
        fast = self._try_fast_set_method(tx, "symmetric_difference", args, kwargs)
        if fast is not None:
            return fast
        other = dict.fromkeys(
            self._operand_keys(tx, args[0]), SetVariable._default_value()
        )
        out_items = {k: v for k, v in self.items.items() if k not in other}
        out_items.update({k: v for k, v in other.items() if k not in self.items})
        return self._new_set(out_items)

    def symmetric_difference_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.symmetric_difference_update: replaces contents in place with the
        # elements in exactly one of self and args[0]. Records a mutation, forces full
        # reconstruction, returns None constant.
        other = dict.fromkeys(
            self._operand_keys(tx, args[0]), SetVariable._default_value()
        )
        new_items = {k: v for k, v in self.items.items() if k not in other}
        new_items.update({k: v for k, v in other.items() if k not in self.items})
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        self.items.clear()
        self.items.update(new_items)
        return ConstantVariable.create(None)

    def update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        # [FUNCTION DOC] set.update: adds all elements of each argument iterable in place.
        # Returns None (no handling) if the tracker is not mutable, else records a mutation
        # and returns a None constant.
        if not self.is_mutable():
            return None
        tx.output.side_effects.mutation(self)
        for other in args:
            for key in self._operand_keys(tx, other):
                self.items.setdefault(key, SetVariable._default_value())
        return ConstantVariable.create(None)

    def remove(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.remove: deletes args[0] (a set argument is normalized to a
        # frozenset). Raises an observed KeyError if absent. Records a mutation and forces
        # full reconstruction. Returns None constant.
        if not self.sq_contains_impl(tx, args[0]).as_python_constant():
            raise_observed_exception(KeyError, tx, args=[args[0]])
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        # sq_contains validated/normalized args[0]; a set key was coerced to
        # a frozenset, so pop that same normalized key.
        key = args[0] if is_hashable(args[0]) else FrozensetVariable(args[0].items)  # type: ignore[missing-attribute]
        self.items.pop(HashableTracker(key))
        return ConstantVariable.create(None)

    def discard(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.discard: like remove but silently does nothing when args[0] is
        # absent. When present, records a mutation and forces full reconstruction. Returns
        # None constant.
        if self.sq_contains_impl(tx, args[0]).as_python_constant():
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            # sq_contains validated/normalized args[0]; a set key was coerced
            # to a frozenset, so pop that same normalized key.
            key = (
                args[0] if is_hashable(args[0]) else FrozensetVariable(args[0].items)  # type: ignore[missing-attribute]
            )
            self.items.pop(HashableTracker(key))
        return ConstantVariable.create(None)

    def _ordering_test(self, tx, args, op):
        # [FUNCTION DOC] HELPER: Shared body of issubset/issuperset. Realizes args[0],
        # converts it to a set tracker if it is not one, then traces the given comparison
        # operator applied to (self, other) and returns its result.
        from .builder import SourcelessBuilder

        other = args[0].realize()
        if not istype(other, SetVariable):
            other = SourcelessBuilder.create(tx, set).call_function(tx, [other], {})
        return SourcelessBuilder.create(tx, op).call_function(tx, [self, other], {})

    def issubset(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.issubset: traces self <= other (other converted to a set).
        return self._ordering_test(tx, args, operator.le)

    def issuperset(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.issuperset: traces self >= other (other converted to a set).
        return self._ordering_test(tx, args, operator.ge)

    def copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.copy: returns a fresh shallow copy tracker via set_copy.
        return set_copy(self)

    def clear(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] set.clear: removes all elements in place, records a mutation and forces
        # full reconstruction. Returns None constant.
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.clear()
        return ConstantVariable.create(None)

    tp_methods = {
        "add": Method(add),
        "pop": Method(pop),
        "isdisjoint": Method(isdisjoint),
        "intersection": Method(intersection),
        "intersection_update": Method(intersection_update),
        "union": Method(union),
        "difference": Method(difference),
        "difference_update": Method(difference_update),
        "symmetric_difference": Method(symmetric_difference),
        "symmetric_difference_update": Method(symmetric_difference_update),
        "update": Method(update),
        "remove": Method(remove),
        "discard": Method(discard),
        "issubset": Method(issubset),
        "issuperset": Method(issuperset),
        "copy": Method(copy),
        "clear": Method(clear),
    }

    def getitem_const(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Always raises a RuntimeError: sets do not support item access.
        raise RuntimeError("Illegal to getitem on a set")

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements __init__ on an existing set: builds a temporary set from
        # the call arguments, then replaces self's contents with it. Records a mutation.
        # Returns None constant.
        from .builder import SourcelessBuilder

        temp_set_vt = SourcelessBuilder.create(tx, set).call_set(tx, *args, **kwargs)
        tx.output.side_effects.mutation(self)
        self.items.clear()
        self.items.update(temp_set_vt.items)  # type: ignore[attr-defined]
        return ConstantVariable.create(None)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Implements iter: returns a SetIterator over the items. For a sourced,
        # non-constant set, registers the source for key-order guarding.
        from .iter import SetIterator

        if self.source and not is_constant_source(self.source):
            tx.output.guard_on_key_order.add(self.source)
        return SetIterator(self.items)

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the | operator (reverse swaps operands). Returns a
        # NotImplemented constant unless both are set/frozenset; otherwise a fresh copy of the
        # left operand with the right operand's items added.
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1318-L1338
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        result = set_copy(self_)
        if self_ is other_:
            return result
        result.items.update(other_.items)  # type: ignore[missing-attribute]
        return result

    def nb_inplace_or_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements |=: NotImplemented constant unless other is a set or
        # frozenset; else records a mutation, merges other's items into self and returns self.
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1340-L1350
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        tx.output.side_effects.mutation(self)
        self.items.update(other.items)  # type: ignore[missing-attribute]
        return self

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the - operator (reverse swaps operands). NotImplemented
        # constant unless both are set/frozenset; otherwise a fresh copy of the left operand
        # minus the right operand's items.
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1801-L1812
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        result = set_copy(self_)
        for k in list(other_.items.keys()):  # type: ignore[missing-attribute]
            result.items.pop(k, None)  # type: ignore[missing-attribute]
        return result

    def nb_inplace_subtract_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements -=: NotImplemented constant unless other is a set or
        # frozenset; else records a mutation, removes other's items from self, returns self.
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1814-L1828
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        tx.output.side_effects.mutation(self)
        for k in list(other.items.keys()):  # type: ignore[missing-attribute]
            self.items.pop(k, None)
        return self

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the & operator (reverse swaps operands). NotImplemented
        # constant unless both are set/frozenset; otherwise delegates to the left operand's
        # intersection method.
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1506-L1518 (set_and)
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        return self_.call_method(tx, "intersection", [other_], {})

    def nb_inplace_and_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements &=: NotImplemented constant unless other is a set or
        # frozenset; else runs intersection_update on self and returns self.
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1520-L1536 (set_iand)
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        self.call_method(tx, "intersection_update", [other], {})
        return self

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the ^ operator (reverse swaps operands). NotImplemented
        # constant unless both are set/frozenset; otherwise delegates to the left operand's
        # symmetric_difference method.
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1984-L1990 (set_xor)
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        return self_.call_method(tx, "symmetric_difference", [other_], {})

    def nb_inplace_xor_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements ^=: NotImplemented constant unless other is a set or
        # frozenset; else runs symmetric_difference_update on self and returns self.
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1992-L2004 (set_ixor)
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        self.call_method(tx, "symmetric_difference_update", [other], {})
        return self

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Implements len: returns an int constant tracker of the element count.
        return VariableTracker.build(tx, len(self.set_items))

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the six rich comparisons against another set. Returns a
        # NotImplemented constant when other is not a set/frozenset type. Equality compares
        # sizes then subset; ordering ops use real-set comparison of the key views. Returns a
        # bool constant tracker.
        """set_richcompare: subset/superset comparisons for all 6 ops.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L2097
        CPython uses PyAnySet_Check: only accepts set/frozenset (not dict views).
        """
        if not isinstance(other, SetVariable):
            try:
                other_type = other.python_type()
            except NotImplementedError:
                return ConstantVariable.create(NotImplemented)
            if not issubclass(other_type, (set, frozenset)):
                return ConstantVariable.create(NotImplemented)

        # Accessing set_items directly is correct: CPython's set_richcompare
        # operates on the internal C struct (PySet_GET_SIZE, set_next,
        # set_contains_entry) -- it never calls __len__ or __contains__.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L2093-L2130
        self_items = self.set_items
        other_items = other.set_items  # type: ignore[attr-defined]
        if op == "__eq__":
            # len check + issubset: same length and subset implies equality.
            if len(self_items) != len(other_items):
                return ConstantVariable.create(False)
            return VariableTracker.build(tx, self_items <= other_items)
        elif op == "__ne__":
            if len(self_items) != len(other_items):
                return ConstantVariable.create(True)
            return VariableTracker.build(tx, not (self_items <= other_items))
        else:
            return VariableTracker.build(
                tx,
                cmp_name_to_op_mapping[op](self_items, other_items),
            )


class OrderedSetClassVariable(VariableTracker):
    # [CLASS DOC] Tracker for the OrderedSet class object itself. Calling it constructs an
    # OrderedSetVariable; method calls on the class (unbound set-style methods) are forwarded
    # to the set tracker passed as the first argument.
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Forwards keyword arguments to the base tracker constructor.
        super().__init__(**kwargs)

    def as_python_constant(self) -> type[OrderedSet[Any]]:
        # [FUNCTION DOC] Returns the OrderedSet class.
        return OrderedSet

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handles method calls on the OrderedSet class. For the constructor
        # method, requires exactly two positional args and no kwargs (else an argument-mismatch
        # error) and returns an empty new OrderedSetVariable. For other names matching known
        # set methods with a set tracker as first arg, calls that method on it with the rest;
        # otherwise defers to the base implementation.
        # KNOWN ISSUE: the mismatch message concatenates strings without a separator, so it
        # reads oddly ("one arg" directly followed by the counts).
        if name == "__new__":
            if len(args) != 2 or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "OrderedSet.__new__ only accepts one arg"
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            return variables.OrderedSetVariable([], mutation_type=ValueMutationNew())

        resolved_fn = getattr(set, name)
        if resolved_fn in set_methods and isinstance(args[0], variables.SetVariable):
            return args[0].call_method(tx, name, args[1:], kwargs)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "OrderedSetVariable":
        # [FUNCTION DOC] Implements calling OrderedSet(...): accepts zero or one positional
        # iterable and no kwargs (else an argument-mismatch error). Returns a new
        # OrderedSetVariable holding the unpacked elements.
        if len(args) > 1 or kwargs:
            raise_args_mismatch(
                tx,
                "OrderedSet",
                "OrderedSet only accepts one arg"
                f"{len(args)} args and {len(kwargs)} kwargs",
            )

        if len(args) == 0:
            # pyrefly: ignore [implicit-any]
            items = []
        else:
            items = unpack_iterable(tx, args[0])
        return variables.OrderedSetVariable(items, mutation_type=ValueMutationNew())


class OrderedSetVariable(SetVariable):
    # [CLASS DOC] Tracker for torch.utils._ordered_set.OrderedSet. Reuses SetVariable storage
    # (insertion-ordered items), exposes its internal dict as the _dict member, and overrides
    # operators to route through named methods since OrderedSet is not a Python set subclass.
    _cpython_type = OrderedSet

    def method_flags_type(self) -> type:
        # [FUNCTION DOC] HELPER: Returns the set type so method arity checks follow set's.
        # OrderedSet is pure-Python (no C ml_flags); its named methods mirror
        # set's arities, so derive MethodFlags from set to enforce them.
        return set

    def _get_internal_dict(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the _dict member; returns a new dict tracker over
        # the same items mapping so inlined OrderedSet code traces without graph breaks.
        # OrderedSet is backed by a dict (self._dict). Expose it so inlined
        # OrderedSet methods (e.g. __contains__ -> `elem in self._dict`) trace
        # natively instead of graph-breaking on an unmodeled attribute.
        from .dicts import ConstDictVariable

        return ConstDictVariable(self.items, mutation_type=ValueMutationNew())  # type: ignore[bad-argument-type]

    tp_members = {"_dict": Member(_get_internal_dict, None)}

    def debug_repr(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string listing item debug reprs in OrderedSet form.
        if not self.items:
            return "OrderedSet([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "OrderedSet([" + ", ".join(items) + "])"

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] Implements repr: string constant tracker of the type name with a
        # bracketed list of element reprs.
        # KNOWN ISSUE: iterates the unordered set_items view, so the printed order may not
        # match insertion order.
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, f"{self.python_type_name()}([{items}])")

    def as_python_constant(self) -> OrderedSet[Any]:
        # [FUNCTION DOC] Returns a real OrderedSet of the element constants.
        # KNOWN ISSUE: built from the unordered set_items view, so insertion order is lost.
        return OrderedSet([k.vt.as_python_constant() for k in self.set_items])

    def python_type(self) -> type[OrderedSet[Any]]:
        # [FUNCTION DOC] Returns the OrderedSet class.
        return OrderedSet

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that imports OrderedSet, loads the elements, builds a
        # list from them and calls the class on it.
        # KNOWN ISSUE: elements come from the unordered set_items view, so the rebuilt
        # OrderedSet may not keep insertion order.
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch.utils._ordered_set", "OrderedSet")
        )
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.set_items)))
        codegen.extend_output(create_call_function(1, False))

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements |: calls the union method on self with other.
        # KNOWN ISSUE: the reverse flag is ignored, so a reflected call still unions as self | other.
        # OrderedSet does not inherit from Python set, so SetVariable.nb_or_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "union", [other], {})

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements &: calls the intersection method on self with other
        # (the reverse flag is ignored).
        # OrderedSet does not inherit from Python set, so SetVariable.nb_and_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "intersection", [other], {})

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements ^: calls the symmetric_difference method on self with
        # other (the reverse flag is ignored).
        # OrderedSet does not inherit from Python set, so SetVariable.nb_xor_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "symmetric_difference", [other], {})

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the - operator (reverse swaps operands): calls the
        # difference method of the left operand with the right one.
        self_, other_ = (other, self) if reverse else (self, other)
        return self_.call_method(tx, "difference", [other_], {})

    def nb_inplace_subtract_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements -=: records a mutation, runs difference_update with other
        # and returns self.
        tx.output.side_effects.mutation(self)
        self.call_method(tx, "difference_update", [other], {})
        return self


class FrozensetVariable(SetVariable):
    # [CLASS DOC] Tracker for frozenset: an immutable, hashable variant of SetVariable. The
    # in-place subtract operator is disabled, copy returns self for the exact type, and
    # difference/intersection/symmetric_difference return frozenset trackers.
    # PyFrozenSet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2526
    _cpython_type = frozenset

    nb_inplace_subtract_impl = None  # type: ignore[bad-override]

    def debug_repr(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string in frozenset form listing item debug reprs.
        if not self.items:
            return "frozenset()"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "frozenset({" + ", ".join(items) + "})"

    @property
    def set_items(self) -> set["HashableTracker"]:
        # [FUNCTION DOC] Property: returns a real (unordered) set of the HashableTracker keys
        # (same as the base class).
        return set(self.items.keys())

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the frozenset type.
        return frozenset

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns a real frozenset of the element constants; raises if any
        # element is not a constant.
        return frozenset({k.vt.as_python_constant() for k in self.set_items})

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] Implements repr: string constant tracker, the empty form when there
        # are no items, else the type name wrapping a braced list of element reprs (unordered).
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L763-L822
        if not self.items:
            return VariableTracker.build(tx, f"{self.python_type_name()}()")
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, f"{self.python_type_name()}({{{items}}})")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that loads the frozenset builtin, the elements, builds
        # a list and calls frozenset on it.
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_global("frozenset"),
                ]
            )
        )
        codegen.foreach([x.vt for x in self.set_items])
        codegen.extend_output(
            [
                create_instruction("BUILD_LIST", arg=len(self.set_items)),
                *create_call_function(1, False),
            ]
        )

    def copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] frozenset.copy: returns self for the exact frozenset type (identity
        # preserved), else a fresh copy via the base set copy.
        if type(self) is FrozensetVariable:
            return self
        return SetVariable.copy(self, tx, args, kwargs)

    def difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] frozenset.difference: runs the base difference and rewraps the items
        # as a frozenset tracker.
        r = SetVariable.difference(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    def intersection(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] frozenset.intersection: runs the base intersection and rewraps the
        # items as a frozenset tracker.
        r = SetVariable.intersection(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    def symmetric_difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] frozenset.symmetric_difference: runs the base operation and rewraps
        # the items as a frozenset tracker.
        r = SetVariable.symmetric_difference(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    tp_methods = {
        "copy": Method(copy),
        "difference": Method(difference),
        "intersection": Method(intersection),
        "symmetric_difference": Method(symmetric_difference),
    }

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements repeated __init__: no-op returning a None constant.
        # frozenset is immutable. Calling __init__ again shouldn't have any effect.
        return ConstantVariable.create(None)

    def is_hashable(self) -> bool:
        # [FUNCTION DOC] frozenset is hashable: always True.
        return True

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Returns (hash, is_fake). A fully constant frozenset uses the real
        # hash and is_fake False. Otherwise hashes each element via the generic hash protocol,
        # wraps the raw hashes, hashes the resulting frozenset and ORs the fake flags.
        # Overrides SetVariable.hash_impl (which raises TypeError for mutable sets).
        # CPython frozenset_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L769
        from .hashable import RawHash
        from .object_protocol import generic_hash_impl

        if self.is_python_constant():
            return hash(self.as_python_constant()), False
        is_fake = False
        raw_hashes = []
        for item in self.set_items:
            h, fake = generic_hash_impl(tx, item.vt)
            is_fake = is_fake or fake
            raw_hashes.append(RawHash(h))
        return hash(frozenset(raw_hashes)), is_fake


class DictKeySetVariable(SetVariable):
    # [CLASS DOC] Tracker for dict_keys views used as set-like values. Shares SetVariable
    # storage, but set_items is the ordered items mapping itself, no contains guard is
    # installed (the dict is already guarded) and the constant form is a real dict_keys view.
    def debug_repr(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string in dict_keys form listing item debug reprs.
        if not self.items:
            return "dict_keys([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "dict_keys([" + ", ".join(items) + "])"

    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        # [FUNCTION DOC] No-op: the underlying dict is already guarded.
        # Already EQUALS_MATCH guarded
        pass

    @property
    def set_items(self) -> Any:
        # [FUNCTION DOC] Property: returns the ordered items mapping itself (not a copied set).
        return self.items

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the dict_keys view type.
        from ..utils import dict_keys

        return dict_keys

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns a real dict_keys view built from a dict whose keys are the
        # element constants.
        return dict.fromkeys(
            {k.vt.as_python_constant() for k in self.set_items}, None
        ).keys()
