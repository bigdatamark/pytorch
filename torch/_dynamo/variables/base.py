# [MODULE DOC]
# Core of Dynamo's variable tracking. Defines VariableTracker (the base wrapper for every
# Python value seen during symbolic tracing), its metaclass machinery, the MutationType
# family that says what mutation is allowed, and the declarative CPython-style tables
# (Method, GetSet, Member, SlotDef and the _SLOTDEFS list) plus the wrapper functions that
# adapt a generic dunder call to the matching *_impl method on a VariableTracker.
# Import-time side effects: builds slot bit-to-field maps, validates that the slot function
# table covers exactly the struct fields (raises AssertionError otherwise), and, at the very
# end, imports the builder and lazy modules (late import to avoid a cycle). Every
# VariableTracker subclass defined afterwards is recorded in
# VariableTrackerMeta.all_subclasses and gets re-entrancy guards on two of its methods.
"""
Core variable tracking functionality for Dynamo. This module defines the fundamental
classes and systems used to track and manage variables during Dynamo's operation.

The module provides:
1. VariableTracker - The base class for tracking variables during compilation
2. MutationType system - Classes for tracking and managing mutations to variables
3. Source type management - Utilities for tracking variable origins and scope
4. Variable state management - Tools for managing variable state and transformations

These components form the foundation of Dynamo's variable handling system,
enabling accurate tracking and transformation of Python code into optimized
computations.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
import logging
import textwrap
from collections.abc import Callable, ItemsView, KeysView, ValuesView
from contextvars import ContextVar
from enum import Enum, IntFlag
from typing import Any, NamedTuple, NoReturn, TYPE_CHECKING, TypeAlias

from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
    PyTypeSlots,
)

from .. import graph_break_hints, variables
from ..current_scope_id import current_scope_id
from ..exc import (
    ObservedAttributeError,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, Source
from ..utils import format_source_range, istype


_RICHCOMPARE_OPS = frozenset(
    {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"}
)


if TYPE_CHECKING:
    from torch._guards import Guard
    from torch.fx.proxy import Node

    from ..codegen import PyCodegen
    from ..side_effects import SideEffects
    from ..symbolic_convert import InstructionTranslatorBase
    from .constant import ConstantVariable
    from .functions import UserFunctionVariable


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SourceLocation:
    # [CLASS DOC] Immutable record of where (file, line, optional end line and column
    # range) the bytecode instruction that produced a VariableTracker lives. Used to
    # render user-facing source excerpts in diagnostics.
    """Source position of the bytecode instruction that generated a VariableTracker."""

    filename: str
    lineno: int
    end_lineno: int | None = None
    col_offset: int | None = None
    end_col_offset: int | None = None

    def format(self) -> str:
        # [FUNCTION DOC] Render the location as a traceback-style text: a File/line
        # header followed by the indented source excerpt (when it can be read from
        # disk). Returns the string; reads the source file via format_source_range.
        result = f'  File "{self.filename}", line {self.lineno}\n'
        source = format_source_range(
            self.filename,
            self.lineno,
            self.end_lineno,
            self.col_offset,
            self.end_col_offset,
        )
        if source:
            result += textwrap.indent(source.rstrip("\n"), "    ") + "\n"
        return result


# Tracks active method calls on VariableTracker instances to detect self-referential
# calls (e.g., as_python_constant on a list that contains itself). Maps
# (id(instance), id(original_method)) tuples to track which calls are in progress.
# We use id(original_method) rather than the method name string so that super()
# delegation within a class hierarchy (e.g. CustomClassObjectVariable.as_python_constant
# calling UserDefinedObjectVariable.as_python_constant) is not a false positive.
_vt_active_calls: ContextVar[set[tuple[int, int]] | None] = ContextVar(
    "_vt_active_calls", default=None
)


class SourceType(Enum):
    # [CLASS DOC] Enum with two members, Existing (value existed before tracing) and
    # New (value created during tracing); used to construct MutationType scopes.
    """
    This Enum divides VariableTracker into 2 cases, depending on the variable
    it represents:
    - already existed that Dynamo began tracking while introspection (Existing)
    - is a new variable that is created during Dynamo introspection (New)

    In general, we have these invariants:
    1. for `VariableTracker` associated with `Existing`, its `source` field must not be None.
    2. for `VariableTracker` associated with `New`, most of the time its
       `source` field is None, except for cases like side effect codegen for
       `AttributeMutationNew`, during which we generate a
       `LocalSource('tmp...')` for such variable, to facilitate codegen.
    """

    Existing = 0
    New = 1


class MutationType:
    # [CLASS DOC] Base marker for VariableTracker.mutation_type. Carries a single
    # attribute, scope: 0 for pre-existing values, otherwise the current
    # higher-order-operator nesting scope id at creation time. Subclasses encode
    # whether the value itself or only its attributes may be mutated and whether it
    # existed before tracing.
    """
    Base class for Variable.mutation_type. It encodes information about
    1. The type of mutation Dynamo allows on the variable.
    2. Whether the value represented by this variable already existed before
    Dynamo tracing.
    """

    def __init__(self, typ: SourceType) -> None:
        # [FUNCTION DOC] Set the scope attribute: zero for Existing, the current scope
        # id for New. Any other typ triggers a graph break (unimplemented) flagged as
        # a Dynamo bug.
        # In HigherOrderOperator tracing, we need to distinguish
        # between MutationTypes inside the HigherOrderOperator and
        # ones outside it. For example, it is not safe to mutate
        # `a` in the following example because it was constructed
        # in a different scope.
        #
        # def f(x):
        #     a = 1
        #     def g(x):
        #         nonlocal a
        #         a = 2
        #         return x
        #     return wrap(g, x) + a
        #
        # We use self.scope to distinguish this.
        # scope == 0: The object was an existing variable
        # scope == 1: The object was created while Dynamo
        #             was introspecting a function
        #             (and no HigherOrderOps were involved)
        # scope >= 2: The object was created through
        #             Dynamo introspection of a HigherOrderOp.
        #             The exact number corresponds to the level
        #             of nested HigherOrderOps.
        if typ is SourceType.Existing:
            self.scope = 0
        elif typ is SourceType.New:
            self.scope = current_scope_id()
        else:
            unimplemented(
                gb_type="Unsupported SourceType",
                context=f"MutationType.__init__ {self} {typ}",
                explanation=f"Dynamo does not support the type `{typ}`",
                hints=[
                    "This branch is not supposed to be reachable.",
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )


class AttrMutationKind(Enum):
    # [CLASS DOC] Enum choosing how a recorded attribute mutation is replayed after the
    # graph runs: GENERIC_SETATTR (ordinary attribute store/delete honoring
    # descriptors) or INSTANCE_DICT (direct write into the instance dictionary).
    # Replay through STORE_ATTR/DELETE_ATTR, preserving descriptor-aware
    # setattr/delattr semantics. Used for slots, getset descriptors, and other
    # ordinary attribute writes that should not bypass descriptors.
    GENERIC_SETATTR = 0
    # Replay a direct object.__dict__ mutation while bypassing descriptors.
    INSTANCE_DICT = 1


class ValueMutationNew(MutationType):
    # [CLASS DOC] Mutation marker for values created during tracing whose contents may
    # be mutated (for example a new list). Hashes and compares by identity.
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created list with this marker,
    indicating that while we need to model mutations to this list, we don't have
    to emit bytecode for these mutations if the list doesn't escape into the
    Python world.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize as a New-source mutation type (scope is the
        # current scope id).
        super().__init__(SourceType.New)

    def __hash__(self) -> int:
        # [FUNCTION DOC] Identity-based hash so each marker instance is a distinct key.
        return id(self)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Identity equality; two markers are equal only if the same object.
        return self is other


class ValueMutationExisting(MutationType):
    # [CLASS DOC] Mutation marker for pre-existing values whose contents may be
    # mutated; mutations must be buffered and replayed after the graph. State:
    # is_modified, set once a mutation happened, so SideEffects can skip untouched values.
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing list with this marker,
    indicating that if we encounter mutations to this list, we need to buffer
    and re-apply those mutations after the graph runs, since the list might be
    used afterwards in Python.
    """

    # A flag to indicate whether mutation happened on the associated
    # `VariableTracker`. This enables SideEffects to accurately and quickly
    # filter out which pre-existing values it needs to generate mutation for.
    is_modified: bool

    def __init__(self, is_modified: bool = False) -> None:
        # [FUNCTION DOC] Initialize as an Existing-source mutation type and store the
        # initial is_modified flag.
        super().__init__(SourceType.Existing)
        self.is_modified = is_modified


class AttributeMutation(MutationType):
    # [CLASS DOC] Common base for mutation markers that allow mutating the value's
    # attributes (not the value itself).
    """
    This case of VariableTracker.mutation_type marker indicates that Dynamo
    allows mutation on the value's attributes.
    """


class AttributeMutationExisting(AttributeMutation):
    # [CLASS DOC] Attribute-mutation marker for objects that existed before tracing;
    # attribute mutations must be buffered and replayed after the graph runs.
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing object with this marker,
    indicating that if we encounter mutations to this object, we need to buffer
    then re-apply those mutations after the graph runs, since the object might
    be used afterwards in Python.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize as an Existing-source mutation type (scope zero).
        super().__init__(SourceType.Existing)


class AttributeMutationNew(AttributeMutation):
    # [CLASS DOC] Attribute-mutation marker for objects created during tracing; if the
    # object does not escape, no mutation bytecode is needed. State: cls_source, an
    # optional Source for the object's class used when reconstructing it.
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created object with this marker,
    indicating that while we need to model mutations to this object, we don't
    have to emit bytecode for these mutations if the object doesn't escape into
    the Python world.
    """

    def __init__(self, cls_source: Source | None = None) -> None:
        # [FUNCTION DOC] Initialize as a New-source mutation type and remember the
        # optional class source.
        super().__init__(SourceType.New)
        self.cls_source = cls_source


def _is_top_level_scope(scope_id: int) -> bool:
    # [FUNCTION DOC] HELPER: True when the scope id is 1, i.e. no higher-order
    # operator is being traced.
    return scope_id == 1


def is_side_effect_safe(m: MutationType) -> bool:
    # [FUNCTION DOC] Decide whether a value with mutation type m may be mutated now.
    # In the top-level scope always true; inside a higher-order operator only when the
    # value was created in the current scope. Reads the current scope id; no side effects.
    scope_id = current_scope_id()

    # In the top-level scope (if no HigherOrderOperators are involved),
    # we are allowed to modify variables created in this scope as well
    # as existing variables.
    if _is_top_level_scope(scope_id):
        return True
    # Otherwise, only allow local mutation of variables created in the current scope
    return m.scope == scope_id


def maybe_get_python_type(obj: VariableTracker) -> type:
    # [FUNCTION DOC] Return obj.python_type(); if the tracker does not implement it
    # (NotImplementedError), graph break via unimplemented flagged as a Dynamo bug.
    try:
        return obj.python_type()
    except NotImplementedError:
        unimplemented(
            gb_type="Unsupported python_type() call",
            context=f"{obj} does not implement python_type()",
            explanation="This VariableTracker does not implement python_type(), "
            "which is required for object protocol operations.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )


class NO_SUCH_SUBOBJ:
    # [CLASS DOC] Sentinel class (never instantiated) returned by
    # get_real_python_backed_value when no concrete Python object backs a tracker.
    """Sentinel indicating no concrete Python object is available."""


class MethodFlags(IntFlag):
    # [CLASS DOC] Bit flags describing a builtin method's calling convention (VARARGS,
    # KEYWORDS, NOARGS, O), mirroring CPython ml_flags; drives arity checking in Method.
    """Calling convention for a `Method` handler, mirroring CPython's
    PyMethodDef ml_flags. `call_method` enforces arity from these before
    invoking the handler, centralizing the raise_args_mismatch boilerplate."""

    VARARGS = 1  # any positional args; kwargs rejected unless KEYWORDS is set
    KEYWORDS = 2  # kwargs allowed (combine with VARARGS)
    NOARGS = 4  # exactly zero args and zero kwargs
    O = 8  # exactly one positional arg, no kwargs


def _check_method_arity(
    vt: VariableTracker,
    tx: InstructionTranslatorBase,
    name: str,
    flags: MethodFlags,
    args: Any,
    kwargs: Any,
) -> None:
    # [FUNCTION DOC] HELPER: Validate args/kwargs of method name on vt against flags.
    # Raises an observed TypeError (via raise_type_error) when keywords are given
    # without the KEYWORDS flag, when NOARGS has any positional argument, or when O
    # does not get exactly one. Returns nothing.
    # Centralized arity check for a tp_methods handler, driven by MethodFlags,
    # raising the same TypeErrors CPython raises for builtin methods. Shared by
    # Method and callers that run the handler directly (e.g. tensor.py).
    n = len(args)
    qualname = f"{vt.python_type_name()}.{name}"
    if kwargs and not (flags & MethodFlags.KEYWORDS):
        raise_type_error(tx, f"{qualname}() takes no keyword arguments")
    if flags & (MethodFlags.NOARGS | MethodFlags.O):
        if flags & MethodFlags.NOARGS and args:
            raise_type_error(tx, f"{qualname}() takes no arguments ({n} given)")
        if flags & MethodFlags.O and n != 1:
            raise_type_error(tx, f"{qualname}() takes exactly one argument ({n} given)")


# CPython PyMethodDef.ml_flags bits (Include/methodobject.h); MethodFlags above
# deliberately mirrors the low four so the mapping is near-identity.
_METH_VARARGS, _METH_KEYWORDS, _METH_NOARGS, _METH_O, _METH_FASTCALL = 1, 2, 4, 8, 0x80


@functools.cache
def _flags_from_ml_flags(python_type: type, name: str) -> MethodFlags:
    # [FUNCTION DOC] HELPER: Cached lookup of the arity flags for attribute name of
    # python_type, translated from the C-level method flags. Defaults to VARARGS plus
    # KEYWORDS if the attribute is missing or has no C flags. Imports torch lazily.
    """Arity convention for `python_type.name`, read from its CPython
    PyMethodDef.ml_flags. Falls back to VARARGS|KEYWORDS when python_type has no
    such attribute or it carries no ml_flags (slot wrappers, pure-Python
    functions)."""
    import torch

    default = MethodFlags.VARARGS | MethodFlags.KEYWORDS
    fn = getattr(python_type, name, None)
    ml = None if fn is None else torch._C._dynamo.eval_frame.get_method_ml_flags(fn)
    if ml is None:
        return default
    flags = MethodFlags(0)
    if ml & _METH_NOARGS:
        flags |= MethodFlags.NOARGS
    if ml & _METH_O:
        flags |= MethodFlags.O
    if ml & (_METH_VARARGS | _METH_FASTCALL):
        flags |= MethodFlags.VARARGS
    if ml & _METH_KEYWORDS:
        flags |= MethodFlags.KEYWORDS
    return flags or default


def _derive_method_flags(vt: VariableTracker, name: str) -> MethodFlags:
    # [FUNCTION DOC] HELPER: Arity flags for method name, computed from the tracker's
    # method_flags_type via the cached C-flag lookup.
    """Resolve the arity flags for method `name` from the VT's arity-reference
    type (see VariableTracker.method_flags_type)."""
    return _flags_from_ml_flags(vt.method_flags_type(), name)


@dataclasses.dataclass(slots=True)
class Method:
    # [CLASS DOC] Slotted dataclass wrapping one handler in a tracker's tp_methods
    # table. Calling it checks arity (derived from the real CPython method flags) then
    # runs the handler; a handler returning None declines so dispatch continues.
    """Declarative entry in a VariableTracker's `tp_methods` table, analogous
    to CPython's PyMethodDef.

    Handlers have signature `(self, tx, args, kwargs)` and return the result
    VariableTracker, or None to decline the call (the equivalent of the old
    `super().call_method` fall-through) so `call_method` continues to the
    object-protocol dispatch below. The method name is the tp_methods key this
    entry is stored under; its arity convention is derived
    on demand from that method's ml_flags (see _derive_method_flags)."""

    handler: Callable[..., VariableTracker | None]

    def __call__(
        self,
        vt: VariableTracker,
        tx: InstructionTranslatorBase,
        name: str,
        args: Any,
        kwargs: Any,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Invoke the entry for method name on vt: derive arity flags,
        # validate args/kwargs (may raise an observed TypeError), then call the handler
        # with vt, tx, args, kwargs. Returns the handler result (None means declined).
        flags = _derive_method_flags(vt, name)
        _check_method_arity(vt, tx, name, flags, args, kwargs)
        return self.handler(vt, tx, args, kwargs)


@dataclasses.dataclass(slots=True)
class GetSet:
    # [CLASS DOC] Slotted dataclass for a tp_getset table entry: a getter and an
    # optional setter (None means read-only). A getter returning None declines.
    """`tp_getset` entry, analogous to CPython's PyGetSetDef. `getter`
    `(self, tx) -> VT | None` (None declines); `setter`
    `(self, tx, value) -> VT | None`, None for read-only."""

    getter: Callable[..., VariableTracker | None]
    setter: Callable[..., VariableTracker | None] | None = None


@dataclasses.dataclass(slots=True)
class Member:
    # [CLASS DOC] Slotted dataclass for a tp_members table entry; same fields as GetSet
    # (getter, optional setter) but a distinct type.
    """`tp_members` entry, analogous to CPython's PyMemberDef. Same shape as
    GetSet; a distinct type so members and getsets never share a class."""

    getter: Callable[..., VariableTracker | None]
    setter: Callable[..., VariableTracker | None] | None = None


def getset_read(
    accessor: Callable[[Any], VariableTracker],
) -> Callable[..., VariableTracker]:
    # [FUNCTION DOC] Build a getter that applies accessor to the tracker and returns
    # the resulting (already built) VariableTracker unchanged.
    """Getter for a GetSet/Member whose value is an already-built VT."""
    return lambda self, tx: accessor(self)


def getset_build(
    accessor: Callable[[Any], Any],
) -> Callable[..., VariableTracker]:
    # [FUNCTION DOC] Build a getter that applies accessor to the tracker to get a raw
    # Python value and wraps it into a new VariableTracker via VariableTracker.build.
    """Getter that builds a VT from the raw value returned by `accessor`."""
    return lambda self, tx: VariableTracker.build(tx, accessor(self))


def unsupported_attr(name: str) -> Callable[..., VariableTracker | None]:
    # [FUNCTION DOC] Return a getter that always graph breaks, used to mark attribute
    # name as unsupported in a tp_getset table.
    def graph_break(
        vt: VariableTracker, tx: InstructionTranslatorBase
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Getter body: always calls unimplemented (graph break, Dynamo
        # bug hint) naming the attribute and the tracker class; never returns.
        unimplemented(
            gb_type="Unsupported attribute",
            context=f"attr_unsupported {vt} {name}",
            explanation=f"{type(vt).__name__} does not support attribute '{name}'",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    return graph_break


# This helps users of `as_python_constant` to catch unimplemented error with
# more information; it inherits `NotImplementedError` for backward
# compatibility reasons.
class AsPythonConstantNotImplementedError(NotImplementedError):
    # [CLASS DOC] NotImplementedError subclass raised when a tracker cannot be turned
    # into a Python constant; keeps the offending tracker in attribute vt so callers
    # can report it.
    vt: VariableTracker

    def __init__(self, vt: VariableTracker, msg: str | None = None) -> None:
        # [FUNCTION DOC] Store vt and build the message (default: vt is not a constant).
        msg = f"{vt} is not a constant" if msg is None else msg
        super().__init__(msg)
        self.vt = vt


# ---------------------------------------------------------------------------
# SLOTDEFS: Dynamo's analog of CPython's static slotdefs[] table
# (Objects/typeobject.c).  One global table, shared by every type, mapping a
# dunder name to the VariableTracker method that implements it (``impl``, the
# slotdefs "function" column) and a ``wrapper`` that adapts a generic call into
# the right impl invocation (the slotdefs "wrapper" column: arity + operand
# order).  CPython also stores a slot offset per entry; Dynamo does not need it
# -- dispatch only cares which impl to call, and MRO lookup already tells us
# whether a name resolves.  Per-VT variation is which impl method is overridden;
# the table itself never varies.  Lives here alongside the ``*_impl`` methods it
# names so mapping and implementations stay together; consumers (call_method
# here, getattro in object_protocol) both read it without an import cycle.
# ---------------------------------------------------------------------------


def _wrap_unaryfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Slot wrapper for zero-argument slots. Raises an observed
    # TypeError if any keyword or positional argument is given; otherwise returns
    # the result of calling the impl with self and tx.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 0:
        raise_type_error(tx, f"expected 0 arguments, got {len(args)}")
    return func(self, tx)


def _wrap_binaryfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Slot wrapper for one-argument slots. Raises an observed
    # TypeError on keywords or if the positional count is not one; returns the impl
    # result called with self, tx and the single operand.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    other = args[0]
    return func(self, tx, other)


def _wrap_binaryfunc_r(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Like the binary wrapper but for reflected operators: same
    # arity checks (TypeError on keywords or not exactly one argument), and calls the
    # impl with the reverse flag set so self acts as the right operand.
    # Reflected binary op (``__radd__``): same impl, operands from the right.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    other = args[0]
    return func(self, tx, other, reverse=True)


def _wrap_ternaryfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Slot wrapper for power: accepts one or two positional
    # arguments (exponent and optional modulus), no keywords; otherwise raises an
    # observed TypeError. Calls the impl with the exponent and the modulus (or none).
    # __pow__ is binary or ternary: ``x.__pow__(y)`` or ``x.__pow__(y, z)`` (mod).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) not in (1, 2):
        raise_type_error(tx, f"expected 1 or 2 arguments, got {len(args)}")
    z = args[1] if len(args) == 2 else None
    return func(self, tx, args[0], z)


def _wrap_ternaryfunc_r(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Reflected variant of the power wrapper: same argument
    # validation, but calls the impl with the reverse flag so self is the base on the
    # right-hand side.
    # Reflected __rpow__: operands from the right.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) not in (1, 2):
        raise_type_error(tx, f"expected 1 or 2 arguments, got {len(args)}")
    z = args[1] if len(args) == 2 else None
    return func(self, tx, args[0], z, reverse=True)


def _wrap_sq_setitem(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for sequence item assignment. Requires exactly
    # two positional arguments (index, value) and no keywords (else observed
    # TypeError); converts the index to an integer via getindex (may raise) and calls
    # the impl with that integer and the value.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import getindex

    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    [arg, value] = args
    i = getindex(tx, self, arg)
    return func(self, tx, i, value)


def _wrap_sq_item(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for sequence item lookup. Requires one positional
    # argument and no keywords (else observed TypeError); converts it to an integer
    # via getindex and calls the impl with that integer.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import getindex

    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")

    [arg] = args
    i = getindex(tx, self, arg)
    return func(self, tx, i)


def _wrap_indexargfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for sequence repeat. Requires one positional
    # argument and no keywords (else observed TypeError); converts it with the
    # number-to-ssize conversion (overflow reported as OverflowError) and calls the
    # impl with the integer count.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import pynumber_as_ssize_t

    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")

    [o] = args
    i = pynumber_as_ssize_t(tx, o, err=OverflowError)
    return func(self, tx, i)


def _wrap_objobjargproc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for mapping item assignment. Requires exactly two
    # positional arguments (key, value) and no keywords (else observed TypeError);
    # calls the impl with key and value.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    [key, value] = args
    return func(self, tx, key, value)


def _wrap_delitem(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for mapping item deletion. Requires one positional
    # argument (key) and no keywords (else observed TypeError); calls the assignment
    # impl with the key and no value (deletion marker).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    [key] = args
    return func(self, tx, key, None)


def _wrap_sq_delitem(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for sequence item deletion. Requires one positional
    # argument and no keywords (else observed TypeError); converts it to an integer via
    # getindex and calls the assignment impl with that integer and no value.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import getindex

    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    [arg] = args
    i = getindex(tx, self, arg)
    return func(self, tx, i, None)


def _wrap_objobjproc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for the membership slot. Requires one positional
    # argument and no keywords (else observed TypeError); calls the impl with it.
    # sq_contains via __contains__: one arg, impl returns a bool.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    [other] = args
    return func(self, tx, other)


def _make_richcmp(
    op: str,
) -> Callable[
    [
        VariableTracker,
        InstructionTranslatorBase,
        Callable[..., VariableTracker],
        list[VariableTracker],
        dict[str, VariableTracker],
    ],
    VariableTracker,
]:
    # [FUNCTION DOC] HELPER: Factory returning a slot wrapper for one comparison
    # operator named op (a dunder name string); the wrapper forwards op to the shared
    # rich-compare impl.
    # tp_richcompare via __lt__/.../__ge__: one impl, op baked per dunder.
    def wrapper(
        self: VariableTracker,
        tx: InstructionTranslatorBase,
        func: Callable[..., VariableTracker],
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Comparison wrapper. Raises an observed TypeError on
        # keywords or if not exactly one argument; returns the impl result for
        # self, the other operand, and the captured operator name.
        if kwargs:
            raise_type_error(tx, "this method takes no keyword arguments")
        if len(args) != 1:
            raise_type_error(tx, f"expected 1 argument, got {len(args)}")
        return func(self, tx, args[0], op)

    return wrapper


richcmp_lt = _make_richcmp("__lt__")
richcmp_le = _make_richcmp("__le__")
richcmp_eq = _make_richcmp("__eq__")
richcmp_ne = _make_richcmp("__ne__")
richcmp_gt = _make_richcmp("__gt__")
richcmp_ge = _make_richcmp("__ge__")


def wrap_call(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Slot wrapper for calling an object; performs no argument
    # validation and forwards args and kwargs unchanged to the impl.
    # tp_call via __call__: variadic (kwargs handled by the call_method branch).
    return func(self, tx, args, kwargs)


def _wrap_init(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Slot wrapper for initialization; forwards args and kwargs
    # unchanged to the impl with no validation.
    # tp_init via __init__: variadic, forwards kwargs (e.g. dict(a=1),
    # defaultdict(list, d, c=3)).
    return func(self, tx, args, kwargs)


def _wrap_setattr(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for attribute assignment. Requires exactly two
    # positional arguments (name, value) and no keywords (else observed TypeError);
    # calls the impl with them. Currently unused by the slot table (entry commented out).
    # tp_setattro via __setattr__: (name, value).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    return func(self, tx, args[0], args[1])


def _wrap_delattr(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for attribute deletion. Requires one positional
    # argument (name) and no keywords (else observed TypeError); calls the impl with
    # it. Currently unused by the slot table (entry commented out).
    # tp_setattro via __delattr__: (name,).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    return func(self, tx, args[0])


def _wrap_getattro(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for attribute lookup. Requires one positional
    # argument and no keywords (else observed TypeError); converts the argument to a
    # Python constant (may raise if it is not constant) and calls the impl with that name.
    # tp_getattro via __getattribute__: impl takes the attribute name as a str.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    return func(self, tx, args[0].as_python_constant())


def _wrap_descr_get(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for the descriptor get protocol. Accepts one or two
    # positional arguments (object, optional owner), no keywords (else observed
    # TypeError). When owner is omitted it is derived from the object's class attribute.
    # Calls the impl with object and owner.
    # tp_descr_get via __get__(obj, owner=None): owner defaults to type(obj).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) not in (1, 2):
        raise_type_error(tx, f"expected 1 or 2 arguments, got {len(args)}")
    obj = args[0]
    owner = args[1] if len(args) > 1 else obj.tp_getattro_impl(tx, "__class__")
    return func(self, tx, obj, owner)


def _wrap_descr_set(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for the descriptor set protocol. Requires exactly
    # two positional arguments (object, value) and no keywords (else observed
    # TypeError); calls the impl with them.
    # tp_descr_set via __set__(obj, value): 2 args.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    return func(self, tx, args[0], args[1])


def _wrap_descr_delete(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Wrapper for the descriptor delete protocol. Requires one
    # positional argument (object) and no keywords (else observed TypeError); calls
    # the shared set impl with the object and no value (deletion marker).
    # tp_descr_set via __delete__(obj): value=None signals delete (CPython
    # passes value==NULL to the shared tp_descr_set slot).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    return func(self, tx, args[0], None)


class SlotGroup(Enum):
    # [CLASS DOC] Enum naming the four CPython slot groups (SEQUENCE, MAPPING, NUMBER,
    # TYPE); each value is the index of that group's mask in the type-slots tuple.
    """A CPython slot group.  The value is the group's index into
    get_type_slots()'s 4-tuple."""

    SEQUENCE = 0
    MAPPING = 1
    NUMBER = 2
    TYPE = 3


@dataclasses.dataclass(frozen=True)
class Slot:
    # [CLASS DOC] Frozen dataclass naming (impl) the VariableTracker method that
    # implements a slot a CPython type fills. Callable: dispatches to that method.
    """A CPython type slot: the name of the VariableTracker method implementing
    the slot function.
    """

    impl: str

    def __call__(
        self,
        vt: VariableTracker,
        tx: InstructionTranslatorBase,
        *args: Any,
        **kwargs: Any,
    ) -> VariableTracker:
        # [FUNCTION DOC] Look up the impl method by name on the tracker's class and call
        # it with the tracker, tx and any extra arguments. Returns its result; errors
        # propagate from the impl.
        return getattr(type(vt), self.impl)(vt, tx, *args, **kwargs)


class PyNumberMethods(NamedTuple):
    # [CLASS DOC] Named tuple mirroring CPython's number-methods struct: one optional
    # Slot per numeric operation (None when the type does not fill that slot).
    nb_bool: Slot | None
    nb_index: Slot | None
    nb_int: Slot | None
    nb_float: Slot | None
    nb_negative: Slot | None
    nb_positive: Slot | None
    nb_absolute: Slot | None
    nb_invert: Slot | None
    nb_add: Slot | None
    nb_inplace_add: Slot | None
    nb_subtract: Slot | None
    nb_inplace_subtract: Slot | None
    nb_multiply: Slot | None
    nb_inplace_multiply: Slot | None
    nb_remainder: Slot | None
    nb_inplace_remainder: Slot | None
    nb_power: Slot | None
    nb_inplace_power: Slot | None
    nb_lshift: Slot | None
    nb_inplace_lshift: Slot | None
    nb_rshift: Slot | None
    nb_inplace_rshift: Slot | None
    nb_and: Slot | None
    nb_inplace_and: Slot | None
    nb_xor: Slot | None
    nb_inplace_xor: Slot | None
    nb_or: Slot | None
    nb_inplace_or: Slot | None
    nb_floor_divide: Slot | None
    nb_inplace_floor_divide: Slot | None
    nb_true_divide: Slot | None
    nb_inplace_true_divide: Slot | None
    nb_divmod: Slot | None
    nb_matrix_multiply: Slot | None
    nb_inplace_matrix_multiply: Slot | None


class PySequenceMethods(NamedTuple):
    # [CLASS DOC] Named tuple mirroring CPython's sequence-methods struct: optional
    # Slot per sequence operation (length, concat, repeat, item, assign, contains, in-place).
    sq_length: Slot | None
    sq_concat: Slot | None
    sq_repeat: Slot | None
    sq_item: Slot | None
    sq_ass_item: Slot | None
    sq_contains: Slot | None
    sq_inplace_concat: Slot | None
    sq_inplace_repeat: Slot | None


class PyMappingMethods(NamedTuple):
    # [CLASS DOC] Named tuple mirroring CPython's mapping-methods struct: optional Slot
    # for length, subscript and subscript assignment.
    mp_length: Slot | None
    mp_subscript: Slot | None
    mp_ass_subscript: Slot | None


class PyTypeObject(NamedTuple):
    # [CLASS DOC] Named tuple mirroring the relevant parts of CPython's type object:
    # optional Slots for repr, hash, call, str, attribute access, rich compare,
    # iteration, descriptors and init, plus the number/sequence/mapping sub-structs.
    tp_repr: Slot | None
    tp_hash: Slot | None
    tp_call: Slot | None
    tp_str: Slot | None
    tp_getattro: Slot | None
    tp_setattro: Slot | None
    tp_richcompare: Slot | None
    tp_iter: Slot | None
    tp_iternext: Slot | None
    tp_descr_get: Slot | None
    tp_descr_set: Slot | None
    tp_init: Slot | None
    tp_as_number: PyNumberMethods
    tp_as_sequence: PySequenceMethods
    tp_as_mapping: PyMappingMethods


# The tp_ slot fields of PyTypeObject (everything but the sub-struct pointers).
_TYPE_FIELDS = tuple(f for f in PyTypeObject._fields if not f.startswith("tp_as_"))

# Per group: C slot bit -> struct field name, keyed via the enum's UPPERCASE name.
_BIT_FIELD: dict[SlotGroup, dict[int, str]] = {
    SlotGroup.NUMBER: {
        getattr(PyNumberSlots, f.upper()): f for f in PyNumberMethods._fields
    },
    SlotGroup.SEQUENCE: {
        getattr(PySequenceSlots, f.upper()): f for f in PySequenceMethods._fields
    },
    SlotGroup.MAPPING: {
        getattr(PyMappingSlots, f.upper()): f for f in PyMappingMethods._fields
    },
    SlotGroup.TYPE: {getattr(PyTypeSlots, f.upper()): f for f in _TYPE_FIELDS},
}


# The actual slot function (a VariableTracker method) each struct field dispatches to
_SLOT_FN: dict[SlotGroup, dict[str, str]] = {
    SlotGroup.NUMBER: {f: f"{f}_impl" for f in PyNumberMethods._fields},
    SlotGroup.SEQUENCE: {f: f"{f}_impl" for f in PySequenceMethods._fields},
    SlotGroup.MAPPING: {f: f"{f}_impl" for f in PyMappingMethods._fields},
    SlotGroup.TYPE: {
        **{f: f"{f}_impl" for f in _TYPE_FIELDS},
        "tp_call": "call_function",
        "tp_hash": "hash_impl",
    },
}

# _SLOT_FN must name a slot fn for exactly each struct field.
for _group, _fns in _SLOT_FN.items():
    if set(_fns) != set(_BIT_FIELD[_group].values()):
        raise AssertionError(f"{_group.name}: _SLOT_FN fields != struct fields")


def _fill(group: SlotGroup, masks: tuple[int, int, int, int]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: For the given slot group, build a mapping from struct field
    # name to a Slot (naming the implementing method) if the type's mask has that
    # slot bit set, else to nothing.
    # {field -> Slot(slot fn) if the type fills the slot, else None}
    mask = masks[group.value]
    fn_map = _SLOT_FN[group]
    return {
        field: Slot(fn_map[field]) if has_slot(mask, bit) else None
        for bit, field in _BIT_FIELD[group].items()
    }


@functools.cache
def _tp_type(obj_type: type) -> PyTypeObject:
    # [FUNCTION DOC] HELPER: Cached. Query the C-level slot masks of obj_type and return
    # its full slot table as a PyTypeObject, with separate sequence and mapping views.
    # obj_type's whole slot table (PyTypeObject-like).  Independent of _tp_dict's
    # cross-group collapse, so a type with both sq_length and mp_length shows up in
    # both the sequence and mapping views (as CPython's separate sub-structs do).
    masks = get_type_slots(obj_type)
    return PyTypeObject(
        tp_as_number=PyNumberMethods(**_fill(SlotGroup.NUMBER, masks)),
        tp_as_sequence=PySequenceMethods(**_fill(SlotGroup.SEQUENCE, masks)),
        tp_as_mapping=PyMappingMethods(**_fill(SlotGroup.MAPPING, masks)),
        **_fill(SlotGroup.TYPE, masks),
    )


WrapperType: TypeAlias = Callable[
    [
        "VariableTracker",
        "InstructionTranslatorBase",
        Callable[..., "VariableTracker"],
        list["VariableTracker"],
        dict[str, "VariableTracker"],
    ],
    "VariableTracker",
]


@dataclasses.dataclass(frozen=True)
class SlotDef:
    # [CLASS DOC] Frozen dataclass for one slot-table entry: dunder name, impl method
    # name, slot group, slot bit, and the wrapper adapting a generic call to the impl.
    # Callable: invokes the impl through the wrapper.
    """One slotdefs entry: dunder name -> (impl method, call wrapper).

    ``impl`` is the VariableTracker method implementing the behavior; ``wrapper``
    adapts a generic ``(self, tx, args)`` call to it (arity + operand order).  A
    reflected op reuses the forward impl with ``_wrap_binaryfunc_r``; an in-place
    op is a plain binary call into its own ``nb_inplace_*`` impl -- so no
    reverse/inplace flags are needed.

    ``group`` is the CPython slot group (``SlotGroup``) the entry lives in and
    ``slot`` is the bit within it; together they decide whether a real type
    implements the slot.
    """

    name: str
    impl: str
    group: SlotGroup
    slot: int
    wrapper: WrapperType

    def __call__(
        self,
        vt: VariableTracker,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Resolve the impl method on the tracker's class and invoke the
        # wrapper with the tracker, tx, impl, args and kwargs. Returns the result;
        # wrapper arity errors surface as observed TypeErrors.
        """Call the impl via the wrapper, passing self, tx, and args."""
        func = getattr(type(vt), self.impl)
        return self.wrapper(vt, tx, func, args, kwargs)


def SQSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    # [FUNCTION DOC] HELPER: Construct a SlotDef in the sequence slot group.
    return SlotDef(name, impl, SlotGroup.SEQUENCE, slot, wrapper)


def MPSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    # [FUNCTION DOC] HELPER: Construct a SlotDef in the mapping slot group.
    return SlotDef(name, impl, SlotGroup.MAPPING, slot, wrapper)


def NBSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    # [FUNCTION DOC] HELPER: Construct a SlotDef in the number slot group.
    return SlotDef(name, impl, SlotGroup.NUMBER, slot, wrapper)


def TPSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    # [FUNCTION DOC] HELPER: Construct a SlotDef in the type slot group.
    return SlotDef(name, impl, SlotGroup.TYPE, slot, wrapper)


# The declarative slot table (mirrors CPython's static slotdefs[] array).  Each
# entry: dunder name, impl method, ``group`` + ``slot`` (which slot sub-struct
# and bit -- reflected ops share the forward slot's group/slot), and wrapper
# (call shape).
_SLOTDEFS: list[SlotDef] = [
    # SlotDef("__getattribute__", ),
    # SlotDef("__getattr__", ),
    # SlotDef("__setattr__", ),
    # SlotDef("__delattr__", ),
    TPSLOT("__repr__", "tp_repr_impl", PyTypeSlots.TP_REPR, _wrap_unaryfunc),
    # TPSLOT("__hash__", "tp_hash_impl", PyTypeSlots.TP_HASH, _wrap_unaryfunc),
    TPSLOT("__call__", "call_function", PyTypeSlots.TP_CALL, wrap_call),
    TPSLOT("__str__", "tp_str_impl", PyTypeSlots.TP_STR, _wrap_unaryfunc),
    TPSLOT(
        "__getattribute__",
        "tp_getattro_impl",
        PyTypeSlots.TP_GETATTRO,
        _wrap_getattro,
    ),
    TPSLOT(
        "__getattr__",
        "tp_getattro_impl",
        PyTypeSlots.TP_GETATTRO,
        _wrap_getattro,
    ),
    # This needs one to model tp_setattro first + PyObject_GenericSetAttr / PyObject_GenericDelAttr
    # TPSLOT(
    #     "__setattr__",
    #     "setattro_impl",
    #     PyTypeSlots.TP_SETATTRO,
    #     _wrap_setattr,
    # ),
    # TPSLOT(
    #     "__delattr__",
    #     "delattro_impl",
    #     PyTypeSlots.TP_SETATTRO,
    #     _wrap_delattr,
    # ),
    TPSLOT(
        "__lt__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_lt,
    ),
    TPSLOT(
        "__le__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_le,
    ),
    TPSLOT(
        "__eq__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_eq,
    ),
    TPSLOT(
        "__ne__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_ne,
    ),
    TPSLOT(
        "__gt__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_gt,
    ),
    TPSLOT(
        "__ge__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_ge,
    ),
    TPSLOT("__iter__", "tp_iter_impl", PyTypeSlots.TP_ITER, _wrap_unaryfunc),
    TPSLOT(
        "__next__",
        "tp_iternext_impl",
        PyTypeSlots.TP_ITERNEXT,
        _wrap_unaryfunc,
    ),
    TPSLOT(
        "__get__",
        "tp_descr_get_impl",
        PyTypeSlots.TP_DESCR_GET,
        _wrap_descr_get,
    ),
    TPSLOT(
        "__set__",
        "tp_descr_set_impl",
        PyTypeSlots.TP_DESCR_SET,
        _wrap_descr_set,
    ),
    TPSLOT(
        "__delete__",
        "tp_descr_set_impl",
        PyTypeSlots.TP_DESCR_SET,
        _wrap_descr_delete,
    ),
    TPSLOT("__init__", "tp_init_impl", PyTypeSlots.TP_INIT, _wrap_init),
    # SlotDef("__new__", ...), # missing
    # SlotDef("__del__", ...), # missing
    # SlotDef("__buffer__", ...), # missing
    # SlotDef("__release_buffer__", ...), # missing
    # SlotDef("__await__", ...), # missing
    # SlotDef("__aiter__", ...), # missing
    # SlotDef("__anext__", ...), # missing
    NBSLOT(
        "__add__",
        "nb_add_impl",
        PyNumberSlots.NB_ADD,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__radd__",
        "nb_add_impl",
        PyNumberSlots.NB_ADD,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__sub__",
        "nb_subtract_impl",
        PyNumberSlots.NB_SUBTRACT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rsub__",
        "nb_subtract_impl",
        PyNumberSlots.NB_SUBTRACT,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__mul__",
        "nb_multiply_impl",
        PyNumberSlots.NB_MULTIPLY,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rmul__",
        "nb_multiply_impl",
        PyNumberSlots.NB_MULTIPLY,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__mod__",
        "nb_remainder_impl",
        PyNumberSlots.NB_REMAINDER,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rmod__",
        "nb_remainder_impl",
        PyNumberSlots.NB_REMAINDER,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__divmod__",
        "nb_divmod_impl",
        PyNumberSlots.NB_DIVMOD,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rdivmod__",
        "nb_divmod_impl",
        PyNumberSlots.NB_DIVMOD,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__pow__",
        "nb_power_impl",
        PyNumberSlots.NB_POWER,
        _wrap_ternaryfunc,
    ),
    NBSLOT(
        "__rpow__",
        "nb_power_impl",
        PyNumberSlots.NB_POWER,
        _wrap_ternaryfunc_r,
    ),
    NBSLOT(
        "__neg__",
        "nb_negative_impl",
        PyNumberSlots.NB_NEGATIVE,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__pos__",
        "nb_positive_impl",
        PyNumberSlots.NB_POSITIVE,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__abs__",
        "nb_absolute_impl",
        PyNumberSlots.NB_ABSOLUTE,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__bool__",
        "nb_bool_impl",
        PyNumberSlots.NB_BOOL,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__invert__",
        "nb_invert_impl",
        PyNumberSlots.NB_INVERT,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__lshift__",
        "nb_lshift_impl",
        PyNumberSlots.NB_LSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rlshift__",
        "nb_lshift_impl",
        PyNumberSlots.NB_LSHIFT,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__rshift__",
        "nb_rshift_impl",
        PyNumberSlots.NB_RSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rrshift__",
        "nb_rshift_impl",
        PyNumberSlots.NB_RSHIFT,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__and__",
        "nb_and_impl",
        PyNumberSlots.NB_AND,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rand__",
        "nb_and_impl",
        PyNumberSlots.NB_AND,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__xor__",
        "nb_xor_impl",
        PyNumberSlots.NB_XOR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rxor__",
        "nb_xor_impl",
        PyNumberSlots.NB_XOR,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__or__",
        "nb_or_impl",
        PyNumberSlots.NB_OR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ror__",
        "nb_or_impl",
        PyNumberSlots.NB_OR,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__int__",
        "nb_int_impl",
        PyNumberSlots.NB_INT,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__float__",
        "nb_float_impl",
        PyNumberSlots.NB_FLOAT,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__iadd__",
        "nb_inplace_add_impl",
        PyNumberSlots.NB_INPLACE_ADD,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__isub__",
        "nb_inplace_subtract_impl",
        PyNumberSlots.NB_INPLACE_SUBTRACT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__imul__",
        "nb_inplace_multiply_impl",
        PyNumberSlots.NB_INPLACE_MULTIPLY,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__imod__",
        "nb_inplace_remainder_impl",
        PyNumberSlots.NB_INPLACE_REMAINDER,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ipow__",
        "nb_inplace_power_impl",
        PyNumberSlots.NB_INPLACE_POWER,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ilshift__",
        "nb_inplace_lshift_impl",
        PyNumberSlots.NB_INPLACE_LSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__irshift__",
        "nb_inplace_rshift_impl",
        PyNumberSlots.NB_INPLACE_RSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__iand__",
        "nb_inplace_and_impl",
        PyNumberSlots.NB_INPLACE_AND,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ixor__",
        "nb_inplace_xor_impl",
        PyNumberSlots.NB_INPLACE_XOR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ior__",
        "nb_inplace_or_impl",
        PyNumberSlots.NB_INPLACE_OR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__floordiv__",
        "nb_floor_divide_impl",
        PyNumberSlots.NB_FLOOR_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rfloordiv__",
        "nb_floor_divide_impl",
        PyNumberSlots.NB_FLOOR_DIVIDE,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__truediv__",
        "nb_true_divide_impl",
        PyNumberSlots.NB_TRUE_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rtruediv__",
        "nb_true_divide_impl",
        PyNumberSlots.NB_TRUE_DIVIDE,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__ifloordiv__",
        "nb_inplace_floor_divide_impl",
        PyNumberSlots.NB_INPLACE_FLOOR_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__itruediv__",
        "nb_inplace_true_divide_impl",
        PyNumberSlots.NB_INPLACE_TRUE_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__index__",
        "nb_index_impl",
        PyNumberSlots.NB_INDEX,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__matmul__",
        "nb_matrix_multiply_impl",
        PyNumberSlots.NB_MATRIX_MULTIPLY,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rmatmul__",
        "nb_matrix_multiply_impl",
        PyNumberSlots.NB_MATRIX_MULTIPLY,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__imatmul__",
        "nb_inplace_matrix_multiply_impl",
        PyNumberSlots.NB_INPLACE_MATRIX_MULTIPLY,
        _wrap_binaryfunc,
    ),
    # Mapping
    MPSLOT(
        "__len__",
        "mp_length_impl",
        PyMappingSlots.MP_LENGTH,
        _wrap_unaryfunc,
    ),
    MPSLOT(
        "__getitem__",
        "mp_subscript_impl",
        PyMappingSlots.MP_SUBSCRIPT,
        _wrap_binaryfunc,
    ),
    MPSLOT(
        "__setitem__",
        "mp_ass_subscript_impl",
        PyMappingSlots.MP_ASS_SUBSCRIPT,
        _wrap_objobjargproc,
    ),
    MPSLOT(
        "__delitem__",
        "mp_ass_subscript_impl",
        PyMappingSlots.MP_ASS_SUBSCRIPT,
        _wrap_delitem,
    ),
    # Sequence
    SQSLOT(
        "__len__",
        "sq_length_impl",
        PySequenceSlots.SQ_LENGTH,
        _wrap_unaryfunc,
    ),
    SQSLOT(
        "__add__",
        "sq_concat_impl",
        PySequenceSlots.SQ_CONCAT,
        _wrap_binaryfunc,
    ),
    SQSLOT(
        "__mul__",
        "sq_repeat_impl",
        PySequenceSlots.SQ_REPEAT,
        _wrap_indexargfunc,
    ),
    SQSLOT(
        "__rmul__",
        "sq_repeat_impl",
        PySequenceSlots.SQ_REPEAT,
        _wrap_indexargfunc,
    ),
    SQSLOT(
        "__getitem__",
        "sq_item_impl",
        PySequenceSlots.SQ_ITEM,
        _wrap_sq_item,
    ),
    SQSLOT(
        "__setitem__",
        "sq_ass_item_impl",
        PySequenceSlots.SQ_ASS_ITEM,
        _wrap_sq_setitem,
    ),
    SQSLOT(
        "__delitem__",
        "sq_ass_item_impl",
        PySequenceSlots.SQ_ASS_ITEM,
        _wrap_sq_delitem,
    ),
    SQSLOT(
        "__contains__",
        "sq_contains_impl",
        PySequenceSlots.SQ_CONTAINS,
        _wrap_objobjproc,
    ),
    SQSLOT(
        "__iadd__",
        "sq_inplace_concat_impl",
        PySequenceSlots.SQ_INPLACE_CONCAT,
        _wrap_binaryfunc,
    ),
    SQSLOT(
        "__imul__",
        "sq_inplace_repeat_impl",
        PySequenceSlots.SQ_INPLACE_REPEAT,
        _wrap_indexargfunc,
    ),
]


@functools.cache
def _compute_slotdefs(obj_type: type) -> dict[str, SlotDef]:
    # [FUNCTION DOC] HELPER: Cached. For obj_type, return a mapping from dunder name to
    # the first SlotDef in table order whose slot bit the type actually fills; later
    # entries with an already-seen name are skipped (so number slots win over sequence).
    masks = get_type_slots(obj_type)
    d: dict[str, SlotDef] = {}
    for sd in _SLOTDEFS:
        if sd.name in d:
            continue
        if has_slot(masks[sd.group.value], sd.slot):
            d[sd.name] = sd
    return d


class VariableTrackerMeta(type):
    # [CLASS DOC] Metaclass of VariableTracker. Keeps a registry (all_subclasses) of
    # every tracker class and picks, per new class, either itself or the
    # implicit-realizing metaclass (which makes isinstance see through lazy trackers)
    # unless the class opts out with the no-implicit-realize flag.
    all_subclasses: list[type] = []

    def __new__(
        mcs: type, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
        # [FUNCTION DOC] Create a tracker class using the plain metaclass when the class
        # (or a base) sets the no-implicit-realize flag or is VariableTracker itself,
        # otherwise using the implicit-realizing metaclass. Returns the new class.
        # Determine which metaclass to use based on the class attributes
        # Classes with _no_implicit_realize = True should NOT implicitly realize
        # (they need standard isinstance behavior to avoid infinite recursion)
        # Check if any base class has _no_implicit_realize set, or if it's in attrs
        no_implicit_realize = attrs.get("_no_implicit_realize", False) or any(
            getattr(base, "_no_implicit_realize", False) for base in bases
        )
        if no_implicit_realize or name == "VariableTracker":
            # Use base VariableTrackerMeta (no custom __instancecheck__)
            return super().__new__(VariableTrackerMeta, name, bases, attrs)
        else:
            # Use ImplicitRealizingVariableTrackerMeta for all other subclasses
            return super().__new__(
                ImplicitRealizingVariableTrackerMeta, name, bases, attrs
            )

    def __init__(
        cls: type, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> None:
        # [FUNCTION DOC] Initialize the class normally and append it to the global
        # all_subclasses registry (side effect on class-level state).
        super().__init__(name, bases, attrs)  # type: ignore[misc]
        VariableTrackerMeta.all_subclasses.append(cls)


class ImplicitRealizingVariableTrackerMeta(VariableTrackerMeta):
    # [CLASS DOC] Metaclass for most tracker subclasses that customizes isinstance so a
    # lazy (not yet realized) tracker answers for the type it will realize to.
    def __instancecheck__(self, instance: object) -> bool:
        # [FUNCTION DOC] isinstance hook: if instance is a lazy tracker, delegate to its
        # lazy_isinstance (which may avoid realizing it); otherwise use the normal check.
        """Make isinstance work with LazyVariableTracker"""
        if instancecheck(LazyVariableTracker, instance):
            return instance.lazy_isinstance(self)  # pyrefly: ignore[missing-attribute]
        return instancecheck(self, instance)


class VariableTracker(metaclass=VariableTrackerMeta):
    # [CLASS DOC] Base class for every tracked local/stack value during tracing. State:
    # source (where the real value comes from, for guards), mutation_type (how mutation
    # is tracked, or none for immutable), source_location, and a cached dict view.
    # Treated as immutable: use clone to change. Declarative tables tp_methods, tp_getset
    # and tp_members (looked up across the class hierarchy) mirror CPython type tables.
    # Subclasses override as_python_constant, as_proxy, reconstruct, call_function,
    # call_method and the *_impl slot methods. Defaults graph break or raise for
    # unsupported operations. Subclass creation installs a re-entrancy guard on
    # as_python_constant and reconstruct to survive self-referential values.
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.

    Prefer the factory function VariableTracker.build() over VariableTracker.__init__().
    """

    # The CPython type(s) this VT is designed to represent.
    # Single type or tuple of types. None means no static CPython type mapping
    # (e.g., dynamic types like UserDefinedObjectVariable, or Dynamo-internal VTs).
    _cpython_type: type | tuple[type, ...] | None = None

    # Declarative named-method table (analogous to CPython's tp_methods): maps a
    # method name to a Method describing its handler and calling convention.
    # `call_method` consults this before the object-protocol (tp_slot) dispatch.
    # Lookup walks the class MRO (see `_lookup_tp_table`), so a subclass table
    # adds to (rather than replaces) the entries it inherits, with the
    # most-derived class winning on name collisions.
    tp_methods: dict[str, Method] = {}
    # Declarative attribute tables, split to match CPython: tp_getset holds the
    # PyGetSetDef attributes, tp_members the PyMemberDef ones.
    tp_getset: dict[str, GetSet] = {}
    tp_members: dict[str, Member] = {}

    def _lookup_tp_table(self, name: str, *table_attrs: str) -> Any:
        # [FUNCTION DOC] HELPER: Search the named class-level tables for key name, walking
        # the class hierarchy from most derived to base and consulting only each class's
        # own tables. Returns the first matching entry or None. Pure lookup.
        """Resolve *name* in the given declarative tables across the class MRO.

        Mirrors CPython method/getset/member inheritance: each class contributes
        its own table entries into its own tp_dict, and lookup walks the MRO with
        the most-derived class winning. A subclass table therefore *adds* to
        (rather than replaces) the entries it inherits. Within a class the tables
        are consulted in the order given.
        """
        for klass in type(self).__mro__:
            d = klass.__dict__
            for attr in table_attrs:
                table = d.get(attr)
                if table is not None and name in table:
                    return table[name]
        return None

    def lookup_tp_getset_member(self, name: str) -> GetSet | Member | None:
        # [FUNCTION DOC] Find the attribute entry for name in the getset tables, then the
        # member tables, across the class hierarchy; None if absent.
        return self._lookup_tp_table(name, "tp_getset", "tp_members")

    def lookup_tp_method(self, name: str) -> Method | None:
        # [FUNCTION DOC] Find the Method entry for name in the methods tables across the
        # class hierarchy; None if absent.
        return self._lookup_tp_table(name, "tp_methods")

    def method_flags_type(self) -> type:
        # [FUNCTION DOC] Return the Python type whose C method flags define arities for
        # this tracker's methods; default is python_type (graph breaks if unavailable).
        """Type whose CPython ml_flags define this VT's tp_methods arities
        (see _derive_method_flags). Defaults to python_type(); a VT whose
        python_type() is a pure-Python stand-in without C ml_flags (e.g.
        OrderedSet) overrides this to the builtin whose method arities it
        mirrors, so MethodFlags still enforces arity."""
        return maybe_get_python_type(self)

    # fields to leave unmodified in apply()
    _nonvar_fields = {
        "value",
        "guards",
        "source",
        "source_location",
        "mutation_type",
        "parents_tracker",
        "user_code_variable_name",
        "dict_vt",
    }

    # Lazily-created view of the instance __dict__, backed by the side effects table
    dict_vt: variables.DunderDictVariable | None = None

    def clone(self, **kwargs: Any) -> VariableTracker:
        # [FUNCTION DOC] Shallow copy: build a new instance of the same class passing the
        # current instance attributes as constructor keywords, overridden by kwargs.
        # Subclass constructors must therefore accept all of their instance attributes.
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def visit(
        cls,
        fn: Callable[[VariableTracker], None],
        value: Any,
        cache: dict[int, Any] | None = None,
        side_effects: SideEffects | None = None,
    ) -> None:
        # [FUNCTION DOC] Pre-order traversal (explicit worklist, no recursion) over value,
        # calling fn on every VariableTracker reached. Descends into tracker attributes
        # (except the non-variable fields), exact lists, tuples, dicts and ordered dicts,
        # and optionally attributes recorded in side_effects. cache (keyed by object id)
        # prevents revisiting; lazy trackers are unwrapped before and after fn. Mutates cache.
        """
        Walk value and call fn on all the VariableTracker instances.

        When side_effects is provided, also walks attributes stored in
        store_attr_mutations (e.g. dataclass fields set during tracing
        that aren't in the VT's __dict__).

        Implemented with an explicit worklist rather than recursion so that
        deeply chained structures (e.g. the ~N-node buffer built by
        itertools.tee over a long iterable) do not overflow the Python stack.
        Children are pushed in reverse so they are popped in their original
        order, matching the pre-order DFS the previous recursive version
        produced (callers rely on the visitation order, e.g. for graph output
        ordering).
        """
        if cache is None:
            cache = {}

        worklist = [value]
        while worklist:
            cur = worklist.pop()
            idx = id(cur)
            if idx in cache:
                continue
            # save `cur` to keep it alive and ensure id() isn't reused
            cache[idx] = cur

            children: list[Any]
            if isinstance(cur, VariableTracker):
                cur = cur.unwrap()
                fn(cur)
                cur = cur.unwrap()  # calling fn() might have realized it
                nonvars = cur._nonvar_fields
                children = [
                    subvalue
                    for key, subvalue in cur.__dict__.items()
                    if key not in nonvars
                ]
                if (
                    side_effects is not None
                    and cur in side_effects.store_attr_mutations
                ):
                    children.extend(side_effects.store_attr_mutations[cur].values())
            elif istype(cur, (list, tuple)):
                children = list(cur)
            elif istype(cur, (dict, collections.OrderedDict)):
                children = list(cur.values())
            else:
                continue
            worklist.extend(reversed(children))

    def __repr__(self) -> str:
        # [FUNCTION DOC] Default text form: the class name with empty parentheses.
        return f"{self.__class__.__name__}()"

    def debug_repr(self) -> str:
        # [FUNCTION DOC] Text for diagnostics: the text form of the Python constant if the
        # tracker is constant, else its normal text form.
        # Intended to be overridden to provide more info
        try:
            return repr(self.as_python_constant())
        except NotImplementedError:
            return repr(self)

    def python_type(self) -> type:
        # [FUNCTION DOC] Return the Python type this tracker represents. Default: the type
        # of its Python constant value; raises NotImplementedError if the tracker is not
        # a constant (subclasses override for non-constants).
        """
        Abstract method to be implemented by subclasses of VariableTracker.

        This method should return the type represented by the instance of the subclass.
        The purpose is to provide a standardized way to retrieve the Python type information
        of the variable being tracked.

        Returns:
            type: The Python type (such as int, str, list, etc.) of the variable tracked by
                the subclass. If the type cannot be determined or is not relevant,
                leaving it undefined or invoking super() is always sound.

        Note:
            This is an abstract method and may be overridden in subclasses.

        Example:
            class SetVariable(VariableTracker):
                def python_type(self):
                    return set

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        try:
            return type(self.as_python_constant())
        except NotImplementedError:
            raise NotImplementedError(f"{self} has no type") from None

    def python_type_name(self) -> str:
        # [FUNCTION DOC] Name of the represented Python type, or a placeholder
        # unknown-type string when python_type is not implemented.
        """
        Return the type name for the Python type this VariableTracker represents.

        Mirrors CPython's tp_name slot (PyTypeObject.tp_name). In Python 3.10+,
        type.__name__ matches CPython's tp_name exactly (e.g., "list", "NoneType").

        Note: There are no external callers outside torch._dynamo that rely on this.
        Internal uses should prefer this over hardcoded type name strings.
        """
        try:
            return self.python_type().__name__
        except NotImplementedError:
            return "<unknown type>"

    def python_qualified_name(self) -> str:
        # [FUNCTION DOC] Module-qualified type name of the represented type (module
        # omitted for the main module and builtins); placeholder string if the type is
        # unknown.
        """
        Equivalent to _PyType_GetFullyQualifiedName

        See https://github.com/python/cpython/blob/v3.15.0b4/Objects/typeobject.c#L1658
        """
        try:
            type_ = self.python_type()
        except NotImplementedError:
            return "<unknown type>"
        # Direct attribute access is safe here because type objects use the getset protocol, which will only return str
        # (and not execute user code)
        mod = type_.__module__
        qn = type_.__qualname__
        if mod not in ("__main__", "builtins"):
            return f"{mod}.{qn}"
        else:
            return qn

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Return the real Python value for constant trackers. Base: always
        # raises AsPythonConstantNotImplementedError (a NotImplementedError) naming the tracker.
        """For constants"""
        raise AsPythonConstantNotImplementedError(self)

    def guard_as_python_constant(self) -> Any:
        # [FUNCTION DOC] Like as_python_constant but converts failure into a graph break
        # (unimplemented, no hints) instead of an exception.
        # KNOWN ISSUE: the docstring says ID_MATCH guards are added, but the base
        # implementation installs no guard; any guarding must come from subclass overrides.
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
        try:
            return self.as_python_constant()
        except NotImplementedError:
            unimplemented(
                gb_type="Not a Python constant",
                context=f"guard_as_python_constant {self}",
                explanation=f"Failed to convert {self} into a Python constant.",
                hints=[],
            )

    def is_python_constant(self) -> bool:
        # [FUNCTION DOC] True if as_python_constant succeeds, False if it raises
        # NotImplementedError. Evaluates the constant as a side effect of the check.
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    def nb_bool_impl(self, tx: InstructionTranslatorBase) -> VariableTracker | None:
        # [FUNCTION DOC] Truth-value slot. Base returns None meaning the type has no bool
        # slot, so truthiness falls back to the length check and then to true.
        # Mirrors CPython's tp_as_number->nb_bool slot.
        # https://github.com/python/cpython/blob/c09ccd9c429/Objects/object.c#L2135-L2158
        #
        # Returns None when the type has no nb_bool, causing generic_is_true to
        # fall through to length check, then truthy default.
        return None

    def is_hashable(self) -> bool:
        # [FUNCTION DOC] Fast, side-effect-free hashability check; base returns True.
        # Unhashable-type trackers override to return False.
        """Whether the underlying Python object is hashable.

        Defaults to True — in CPython, all objects are hashable unless the type
        explicitly sets __hash__ = None (PyObject_HashNotImplemented).
        https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L7288

        Subclasses override to return False for unhashable types (list, dict,
        set, etc.).  This is a fast, side-effect-free check used by
        HashableTracker and __contains__ before attempting to hash.
        """
        return True

    def hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        # [FUNCTION DOC] Default identity hash. Returns (hash, is_fake) with is_fake always
        # true. With a source: resolves the real object and graph breaks if its type has a
        # non-default hash (missing override); otherwise hashes the real object. Sourceless:
        # returns the tracker's own identity. Resolves the source via the output graph.
        """Default tp_hash: object.__hash__ = PyObject_GenericHash (identity hash).

        PyObject_GenericHash: https://github.com/python/cpython/blob/e76aa128fe/Python/pyhash.c#L143-L147
        Assigned on object: https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L7288

        Returns (hash_value, is_fake).  is_fake=True means the hash depends on
        a sourceless object's identity and must not escape into output bytecode.

        VT subclasses override this for types with their own C tp_hash.
        UserDefinedObjectVariable.hash_impl handles the MRO walk for custom
        Python __hash__, builtin-inherited C hash, and unknown C tp_hash.
        """
        # object.__hash__ = Py_HashPointer(obj) — identity hash.
        if self.source:
            real_val = tx.output.resolve_source_value(self.source)
            if type(real_val).__hash__ is not object.__hash__:
                unimplemented(
                    gb_type="Missing hash_impl override",
                    context=f"hash_impl {self}",
                    explanation=f"{type(self).__name__} wraps "
                    f"{type(real_val).__name__} which has a non-default "
                    f"tp_hash. Add a hash_impl override to "
                    f"{type(self).__name__}.",
                    hints=[*graph_break_hints.DYNAMO_BUG],
                )
            # object.__hash__ = id(obj), so the hash is identity-based.
            # Return is_fake=True so the hash is only usable internally
            # (dict keys, set membership) and doesn't escape into bytecode
            # — avoids installing an ID_MATCH guard just for hashing.
            # TODO when LazyConstants are fully landed, we can use them here instead.
            return hash(real_val), True
        # Sourceless: no real object to hash — fake id.
        return id(self), True

    def tp_richcompare_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        # [FUNCTION DOC] Rich-comparison slot for operator op (a dunder name) against other.
        # Base always graph breaks (missing override, Dynamo bug hint). Overrides return a
        # tracker holding the comparison result, or the not-implemented constant to let
        # the other operand try.
        """Per-VT tp_richcompare slot. Subclasses must override.

        Analogous to CPython's tp_richcompare function pointer on PyTypeObject.
        Returns ConstantVariable(NotImplemented) when the type does not handle
        the comparison (signaling do_richcompare to try the other operand).

        Called from two paths:
        - call_method("__eq__") calls tp_richcompare_impl directly (like CPython's
          a.__eq__(b) calling tp_richcompare without do_richcompare).
        - generic_richcompare calls tp_richcompare_impl as part of the 4-step
          do_richcompare algorithm (subclass priority, forward, reflected,
          fallback).
        """
        unimplemented(
            gb_type="Missing tp_richcompare_impl override",
            context=f"tp_richcompare_impl {self} {op}",
            explanation=f"{type(self).__name__} does not implement "
            f"tp_richcompare_impl. Add a tp_richcompare_impl override to "
            f"{type(self).__name__}.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    def is_constant_match(self, *values: Any) -> bool:
        # [FUNCTION DOC] True if this tracker is a constant equal to one of values; base
        # returns False (constant trackers override).
        """
        Check if this variable is a python constant matching one of the given values.

        Examples:
            var.is_constant_match(None)  # True if var is constant None
            var.is_constant_match(True, False)  # True if var is constant True or False
            var.is_constant_match(NotImplemented)  # True if var is constant NotImplemented
        """
        return False

    def is_constant_none(self) -> bool:
        # [FUNCTION DOC] True if this tracker is the constant None; base returns False.
        """Check if this variable is a constant None value."""
        return False

    def make_guard(self, fn: Callable[..., Any]) -> Guard:
        # [FUNCTION DOC] Create a guard of kind fn from this tracker's source. Raises
        # NotImplementedError if the tracker has no source. Does not install the guard.
        if self.source:
            return self.source.make_guard(fn)
        raise NotImplementedError

    def const_getattr(self, tx: InstructionTranslatorBase, name: str) -> Any:
        # [FUNCTION DOC] Fetch attribute name as a raw Python constant. Base always raises
        # NotImplementedError; subclasses with constant attributes override.
        """getattr(self, name) returning a python constant"""
        raise NotImplementedError

    def is_symnode_like(self) -> bool:
        # [FUNCTION DOC] True for values usable in symbolic-number operations; base False.
        """Return True for values that can participate in SymNode operations"""
        return False

    def is_tensor(self) -> bool:
        # [FUNCTION DOC] True for tensor trackers; base False.
        """Return True for TensorVariable instances"""
        return False

    def get_handler_type_for_dispatch(self) -> type[VariableTracker]:
        # [FUNCTION DOC] Tracker class used to pick a builtin-handler; base returns the
        # tracker's own class. Overrides may substitute another class (and must guard).
        """Return the VariableTracker type to use for builtin handler dispatch.

        This is used by BuiltinVariable to look up the appropriate handler for
        a given set of arguments. Most VariableTrackers just return their own
        type, but LazyConstantVariable returns ConstantVariable since it can
        be treated as a constant for most builtin operations.

        Subclasses that override this should install appropriate guards to
        ensure the dispatched handler remains valid.
        """
        return type(self)

    def get_value_for_setattr(self) -> object | None:
        # [FUNCTION DOC] Return the wrapped real object to use for generic attribute-store
        # mutation, or None to decline; base declines.
        """Return the wrapped Python object for generic STORE_ATTR mutation,
        or None to decline.  Only override for VTs with __dict__ and
        standard __setattr__."""
        return None

    def lookup_instance_dict(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Look up name in the instance dictionary; base has none and returns
        # None. Overrides return a tracker when the attribute exists.
        """Look up *name* in the instance __dict__ (tp_dictoffset equivalent).

        Returns a VT if the attribute exists in the instance dict, None if
        the attribute is absent or the VT has no instance dict at all.
        UDOV overrides to check self.value.__dict__ + side effects.
        """
        return None

    def call_getattr_fallback(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Final fallback of generic attribute lookup: call a type-defined
        # missing-attribute hook. Base returns None (no hook).
        """Call __getattr__ fallback (step 6 of GenericGetAttr).

        Returns a VT if the type defines __getattr__ and it succeeds,
        None otherwise.  The base returns None (most types have no
        __getattr__).  UDOV overrides to walk the MRO for __getattr__.
        """
        return None

    def tp_getattro_impl(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute lookup slot. Order: (1) getset/member table entry (a
        # getter returning None declines); (2) the class attribute of the object; (3) the
        # generic descriptor-protocol lookup when the Python type is known (may raise an
        # observed AttributeError; unhandled descriptors fall through); (4) a constant
        # attribute via const_getattr, wrapped as a constant tracker with a sourced
        # attribute guard when applicable. Raises NotImplementedError if the constant is
        # not a literal. Installs a constant-match guard for sourced non-constant owners.
        """Default attribute access via object_generic_getattr.

        Tries the MRO-based descriptor protocol first (PyObject_GenericGetAttr),
        falls back to const_getattr for VTs without a known python_type or
        when the descriptor protocol hits an unhandled type.
        """
        # tp_getset/tp_members are data descriptors: resolve ahead of the
        # object-protocol walk. A getter returning None declines.
        getset = self.lookup_tp_getset_member(name)
        if getset is not None:
            result = getset.getter(self, tx)
            if result is not None:
                return result

        # object.__class__: one shared getset on `object` rather than per-VT.
        if name == "__class__":
            return VariableTracker.build(tx, self.python_type())

        try:
            py_type = self.python_type()
        except NotImplementedError:
            py_type = None

        if py_type is not None:
            from .object_protocol import (
                _UnhandledDescriptorError,
                object_generic_getattr,
            )

            try:
                return object_generic_getattr(tx, self, name)
            except _UnhandledDescriptorError:
                pass

        value = self.const_getattr(tx, name)
        if not variables.ConstantVariable.is_literal(value):
            raise NotImplementedError
        source = self.source and AttrSource(self.source, name)
        if source and not self.is_python_constant():
            # Skip guards on const getattr objects like __code__.co_argcount
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        return variables.ConstantVariable.create(value, source=source)

    def get_dict_vt(
        self, tx: InstructionTranslatorBase
    ) -> variables.DunderDictVariable:
        # [FUNCTION DOC] Return the cached view of the instance dictionary, creating and
        # caching it on first use (mutates the dict_vt attribute despite the tracker being
        # otherwise immutable).
        # Callers gate this on the object actually having an instance __dict__
        # (e.g. the per-VT `__dict__` attribute branches). If a VT without a
        # real instance dict reaches here, DunderDictVariable's example-value
        # lookup graph-breaks rather than fabricating a bogus dict.
        if self.dict_vt is None:
            self.dict_vt = variables.DunderDictVariable.create(tx, self)
        return self.dict_vt

    def is_proxy(self) -> bool:
        # [FUNCTION DOC] True if as_proxy succeeds, False on NotImplementedError.
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self) -> Any:
        # [FUNCTION DOC] Return the graph proxy (or value usable as a graph argument) for
        # this tracker. Base always raises NotImplementedError.
        raise NotImplementedError(str(self))

    def maybe_fx_node(self) -> Node | None:
        # [FUNCTION DOC] Return the FX graph node behind this tracker's proxy, or None if
        # there is no proxy or it is not an FX proxy.
        try:
            proxy = self.as_proxy()
            import torch.fx

            if isinstance(proxy, torch.fx.Proxy):
                return proxy.node
            return None
        except NotImplementedError:
            return None

    def _contains_self_reference(self) -> bool:
        # [FUNCTION DOC] HELPER: True if traversing this tracker's own variable-valued
        # attributes reaches the tracker itself (cycle detection).
        """Check if this variable references itself (directly or indirectly)."""
        found_self = False

        def check(vt: VariableTracker) -> None:
            # [FUNCTION DOC] HELPER: Visitor that records, in the enclosing flag, whether
            # vt is the tracker under inspection.
            nonlocal found_self
            if vt is self:
                found_self = True

        # unwrap first iteration - otherwise we can't detect if we revisit self
        for key, subvalue in self.__dict__.items():
            if key not in self._nonvar_fields:
                VariableTracker.visit(check, subvalue)

        return found_self

    def reconstruct(self, codegen: PyCodegen) -> None:
        # [FUNCTION DOC] Emit bytecode via codegen that rebuilds this value at runtime.
        # Base raises NotImplementedError; subclasses override (re-entrant calls on the
        # same tracker graph break through the installed guard).
        raise NotImplementedError

    def reconstruct_pycode(self, codegen: PyCodegen) -> str:
        # [FUNCTION DOC] Return Python source text that rebuilds this value. Base raises
        # NotImplementedError naming the tracker.
        raise NotImplementedError(f"reconstruct_pycode not implemented for {self}")

    def unpack_var_sequence(
        self, tx: InstructionTranslatorBase
    ) -> list[VariableTracker]:
        # [FUNCTION DOC] Return the elements of the value as a list of trackers when it
        # can be unpacked statically. Base raises NotImplementedError.
        raise NotImplementedError

    def _hasattr_check_side_effects(
        self, tx: InstructionTranslatorBase, name: str
    ) -> ConstantVariable | None:
        # [FUNCTION DOC] HELPER: If attribute name has a pending recorded mutation, return a
        # boolean constant tracker saying whether it still exists (false if deleted);
        # otherwise None. Reads the side-effects table.
        """If *name* has a pending mutation, return the hasattr result; else None."""
        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            value = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
            return variables.ConstantVariable.create(
                not isinstance(value, variables.DeletedVariable)
            )
        return None

    def call_obj_hasattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> ConstantVariable:
        # [FUNCTION DOC] Dynamo's attribute-existence test. Checks pending mutations first;
        # else attempts attribute lookup: success gives true, an observed AttributeError
        # (cleared from the exception stack) gives false. If lookup is unsupported
        # (NotImplementedError or graph break) it graph breaks with a hint. Returns a
        # boolean constant tracker.
        """Dynamo's hasattr(): try tp_getattro_impl, catch AttributeError.

        Mirrors CPython's PyObject_HasAttr (via PyObject_GetOptionalAttr):
        call tp_getattro, suppress AttributeError via PyErr_Clear, return
        True/False.
        https://github.com/python/cpython/blob/848cb25624ab44c9fef2966c777419376b65af1b/Objects/object.c#L1346
        """
        result = self._hasattr_check_side_effects(tx, name)
        if result is not None:
            return result

        try:
            self.tp_getattro_impl(tx, name)
            return variables.ConstantVariable.create(True)
        except ObservedAttributeError:
            tx.exn_vt_stack.clear_current_exception()
            return variables.ConstantVariable.create(False)
        except (NotImplementedError, Unsupported):
            pass

        unimplemented(
            gb_type="Unsupported hasattr call",
            context=f"call_obj_hasattr {self} {name}",
            explanation=f"Dynamo does not know how to trace the function `{self.debug_repr()}`",
            hints=[
                f"Avoid calling `hasattr({self.__class__.__name__}, {name})` in your code.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def tp_iter_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # [FUNCTION DOC] Iteration slot returning an iterator tracker. Base graph breaks
        # (missing tp_iter, supportable hint); iterable subclasses override.
        """
        Implements PyObject_GetIter semantics (tp_iter slot).
        Subclasses override this to support iteration.
        """
        # PyObject_GetIter: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L2847-L2870
        # TODO: raise TypeError instead - Make sure the jit tests for ScriptDict/ScriptList works with
        # this change
        unimplemented(
            gb_type="missing tp_iter",
            context=f"tp_iter_impl not implemented for {self.python_type_name()}",
            explanation=f"Dynamo does not know how to iterate over `{self.debug_repr()}`.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def tp_init_impl(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Initialization slot with positional and keyword argument trackers.
        # Base graph breaks (missing tp_init, Dynamo bug hint); subclasses override.
        """tp_init slot (__init__). VTs override to initialize instances."""
        unimplemented(
            gb_type="missing tp_init",
            context=f"tp_init_impl not implemented for {self.python_type_name()}",
            explanation=f"Dynamo does not know how to trace __init__ on `{self.debug_repr()}`.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    def call_function(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Call this value with args and kwargs. Base fallback: if the Python
        # type is known and not callable, raises an observed TypeError (not callable);
        # otherwise graph breaks as an unsupported call.
        # Reaching this base fallback means no subclass handled the call. If the
        # object's type has no tp_call slot, it is genuinely not callable, so
        # mirror CPython's PyObject_Call TypeError. Otherwise the type is
        # callable in principle but unsupported by Dynamo -> graph break.
        from .object_protocol import pycallable_check

        try:
            obj_type = self.python_type()
        except NotImplementedError:
            obj_type = None
        if obj_type is not None and not pycallable_check(obj_type):
            raise_type_error(tx, f"'{self.python_type_name()}' object is not callable")
        unimplemented(
            gb_type="Unsupported function call",
            context=f"call_function {self} {args} {kwargs}",
            explanation=f"Dynamo does not know how to trace the function `{self.debug_repr()}`",
            hints=[
                f"Avoid calling `{self.debug_repr()}` in your code.",
                "Please report an issue to PyTorch.",
            ],
        )

    def sq_length_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # [FUNCTION DOC] Sequence length slot fallback: raises an observed TypeError saying
        # the type has no length.
        """Called when sq_length is not implemented."""
        raise_observed_exception(
            TypeError,
            tx,
            args=[f"object of type '{self.python_type_name()}' has no len()"],
        )

    def mp_length_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # [FUNCTION DOC] Mapping length slot fallback: raises an observed TypeError saying
        # the type has no length.
        """Called when mp_length is not implemented."""
        raise_observed_exception(
            TypeError,
            tx,
            args=[f"object of type '{self.python_type_name()}' has no len()"],
        )

    def mp_subscript_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Mapping subscript slot (lookup by key). Base graph breaks as
        # unsupported subscripting; subclasses override.
        # PyObject_GetItem: https://github.com/python/cpython/blob/62a6e898e01/Objects/abstract.c#L155-L206
        # generic_getitem handles dispatch and raises TypeError for non-subscriptable
        # objects.  This base fallback fires for types with mp_subscript at the
        # C level but no Dynamo override yet.
        unimplemented(
            gb_type="missing_mp_subscript",
            context=f"mp_subscript_impl not defined for {type(self).__name__}",
            explanation=f"Dynamo does not yet support subscripting '{self.python_type_name()}'.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_item_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Sequence item slot (lookup by integer index; key is already an
        # integer tracker). Base graph breaks; subclasses override.
        # PyObject_GetItem Branch 2: tp_as_sequence->sq_item
        # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L168-L181
        # Key has already been converted to int via nb_index_impl by generic_getitem.
        unimplemented(
            gb_type="unsupported __getitem__ (sq_item)",
            context=f"sq_item_impl {self} {key}",
            explanation=f"Dynamo does not know how to handle sq_item on {self}",
            hints=[],
        )

    def sq_contains_impl(
        self, tx: InstructionTranslatorBase, item: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Membership slot fallback: graph breaks (supportable hint) because
        # Dynamo cannot test whether item is contained in this value.
        """Called when sq_contains is not implemented."""
        unimplemented(
            gb_type="missing sq_contains",
            context=f"sq_contains_impl not implemented for {self.python_type_name()}",
            explanation=f"Dynamo does not know how to check if `{item.debug_repr()}` is in `{self.debug_repr()}`.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def mp_ass_subscript_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # [FUNCTION DOC] Mapping item assignment/deletion slot (value absent means delete).
        # Base graph breaks; subclasses override.
        unimplemented(
            gb_type="missing_mp_ass_subscript",
            context=f"mp_ass_subscript_impl not defined for {self.python_type_name()}",
            explanation=f"Dynamo does not yet support item assignment on '{self.python_type_name()}'.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_ass_item_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # [FUNCTION DOC] Sequence item assignment/deletion slot by integer index (value
        # absent means delete). Base graph breaks; subclasses override.
        unimplemented(
            gb_type="unsupported __setitem__ (sq_ass_item)",
            context=f"sq_ass_item_impl {self} {key} {value}",
            explanation=f"Dynamo does not know how to handle sq_ass_item on {self}",
            hints=[],
        )

    def sq_concat_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Sequence concatenation slot with other. Base graph breaks.
        """Sequence concatenation via + operator (sq_concat slot)."""
        unimplemented(
            gb_type="missing sq_concat",
            context=f"sq_concat not defined for {type(self).__name__}",
            explanation=f"Dynamo does not yet support concatenating '{self.python_type_name()}' and '{other.python_type_name()}'",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_inplace_concat_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place sequence concatenation slot with other. Base graph breaks.
        """In-place sequence concatenation via += operator (sq_inplace_concat slot)."""
        unimplemented(
            gb_type="missing sq_inplace_concat",
            context=f"sq_inplace_concat not defined for {type(self).__name__}",
            explanation=f"Dynamo does not yet support inplace concat of '{self.python_type_name()}' and '{other.python_type_name()}'",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def call_method(
        self,
        tx: InstructionTranslatorBase,
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Call method name with args and kwargs. Dispatch order: (1) the
        # declarative methods table (handler returning None declines); (2) the slot table
        # for the Python type (dunder methods); (3) special cases: hash, class-check on
        # constant classes with default behavior, and a constant-folded pickling protocol
        # hook for constant values; (4) otherwise graph breaks with hints (extra hints for
        # iterator dunder calls on user objects). Result is a tracker.
        # KNOWN ISSUE: the subclass-check special case tests the truthiness of the
        # constants and calls as_python_constant without catching NotImplementedError, so a
        # non-constant argument raises instead of falling through to the graph break.
        # Declarative named-method dispatch (tp_methods). A handler returning
        # None declines the call and falls through to the object-protocol
        # (tp_slot) dispatch below, mirroring the old super().call_method path.
        method = self.lookup_tp_method(name)
        if method is not None:
            result = method(self, tx, name, args, kwargs)
            if result is not None:
                return result

        slotdef = self._slotdefs.get(name)
        if slotdef is not None:
            return slotdef(self, tx, args, kwargs)

        if name == "__hash__" and not args and not kwargs:
            from .object_protocol import generic_hash

            return generic_hash(tx, self)

        if name == "__subclasscheck__" and len(args) == 1 and not kwargs:
            if (self_py := self.as_python_constant()) and (
                derived_py := args[0].as_python_constant()
            ):
                if (
                    inspect.getattr_static(self_py, "__subclasscheck__", None)
                    is type.__subclasscheck__
                ):
                    return variables.ConstantVariable.create(
                        issubclass(derived_py, self_py)
                    )
        # __reduce_ex__ is a C builtin (object.__reduce_ex__) that Dynamo
        # cannot trace into.  Constant-fold it for VTs backed by a real
        # Python object so that copy.deepcopy can trace through.
        if (
            name == "__reduce_ex__"
            and len(args) == 1
            and not kwargs
            and self.is_python_constant()
        ):
            protocol = args[0].as_python_constant()
            return VariableTracker.build(
                tx, self.as_python_constant().__reduce_ex__(protocol)
            )

        hints = [
            f"Avoid calling `{self.python_type_name()}.{name}` in your code.",
            "Please report an issue to PyTorch.",
        ]
        # additional hint for method calls on improperly constructed iterators
        if isinstance(self, variables.UserDefinedObjectVariable) and name in (
            "__iter__",
            "__next__",
        ):
            if isinstance(self.value, (KeysView, ItemsView, ValuesView)):
                hints.append(
                    "Consider moving the creation of dict view object (e.g. `dict.keys()`, `dict.items()`,) "
                    "to the compiled region, instead of passing it as an input to the compiled region."
                )
            hints.append(
                "Dynamo does not fully support tracing builtin iterators (e.g. `map`, `zip`, `enumerate`) "
                "passed in from uncompiled to compiled regions (e.g. `torch.compile(fn)(enumerate(...))`). "
                "This can happen unintentionally if a previous graph break happens with a builtin iterator "
                "in the local scope."
            )
            hints.append(
                "List/dict comprehensions in Python <= 3.11 result in implicit function calls, which Dynamo "
                "cannot trace as a top level frame. Possible workarounds are (1) use a loop instead of a comprehension, "
                "(2) fix any graph breaks in the function above the comprehension, (3) wrap the comprehension in a "
                "function, or (4) use Python 3.12+."
            )
        unimplemented(
            gb_type="Unsupported method call",
            context=f"call_method {self} {name} {args} {kwargs}",
            explanation=f"Dynamo does not know how to trace method `{name}` of class `{self.python_type_name()}`",
            hints=hints,
        )

    def call_tree_map(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Fast path for the tree-map utility on this node. If an is_leaf
        # predicate is supplied, evaluate it on self: non-constant result falls back to
        # tracing the real tree-map; true applies map_fn to self (plus rest) directly.
        # Otherwise defers to call_tree_map_branch.
        """Performance optimization to implement optree.tree_map faster than tracing it"""
        is_leaf_var = tree_map_kwargs.get("is_leaf")
        if is_leaf_var is not None and not is_leaf_var.is_constant_none():
            pred_result = is_leaf_var.call_function(tx, [self], {})
            try:
                leaf_decision = pred_result.as_python_constant()
            except NotImplementedError:
                return self._tree_map_fallback(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                )
            if leaf_decision:
                return map_fn.call_function(tx, [self, *rest], {})

        return self.call_tree_map_branch(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
        )

    def call_tree_map_branch(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Default (leaf) behavior of the tree-map fast path: no fast path for
        # this node type, so fall back to tracing the real tree-map call. Container
        # subclasses override to recurse.
        """Emulate optree.tree_map without is_leaf/none_is_leaf checks (handled above)"""
        return self._tree_map_fallback(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
        )

    def _tree_map_fallback(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Trace the real tree-map: clone the function tracker with its
        # fast path disabled and call it with map_fn, self and the rest. Logs at debug level.
        tree_map_fn_copy = tree_map_fn.clone()
        tree_map_fn_copy._maybe_call_tree_map_fastpath = lambda *args, **kwargs: None  # type: ignore[missing-attribute]
        log.debug(
            "tree_map fastpath fallback triggered for %s (rest=%s, kwargs=%s)",
            self,
            rest,
            tree_map_kwargs,
        )
        return tree_map_fn_copy.call_function(
            tx,
            [map_fn, self, *rest],
            tree_map_kwargs,
        )

    def call_tree_map_with_path(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        # [FUNCTION DOC] Fast path for the path-aware tree-map on this node at key path
        # keypath. If an is_leaf predicate is supplied: non-constant result falls back to
        # tracing; true calls map_fn with the key path tuple, self and rest. Otherwise
        # defers to call_tree_map_with_path_branch.
        """Performance optimization to implement tree_map_with_path faster than tracing it"""
        is_leaf_var = tree_map_kwargs.get("is_leaf")
        if is_leaf_var is not None and not is_leaf_var.is_constant_none():
            pred_result = is_leaf_var.call_function(tx, [self], {})
            try:
                leaf_decision = pred_result.as_python_constant()
            except NotImplementedError:
                return self._tree_map_with_path_fallback(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                    keypath,
                )
            if leaf_decision:
                keypath_var = variables.TupleVariable(
                    [VariableTracker.build(tx, k) for k in keypath]
                )
                return map_fn.call_function(tx, [keypath_var, self, *rest], {})

        return self.call_tree_map_with_path_branch(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
            keypath,
        )

    def call_tree_map_with_path_branch(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        # [FUNCTION DOC] Default leaf behavior: build a tuple tracker from keypath and call
        # map_fn with it, self and rest. Container subclasses override to recurse.
        """Handle tree_map_with_path for leaf nodes (default behavior)"""
        keypath_var = variables.TupleVariable(
            [VariableTracker.build(tx, k) for k in keypath]
        )
        return map_fn.call_function(tx, [keypath_var, self, *rest], {})

    def _tree_map_with_path_fallback(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Trace the real path-aware tree-map: clone the function
        # tracker with its fast path disabled and call it with map_fn, self and rest.
        # KNOWN ISSUE: the call passes the same arguments as the plain fallback (no key path
        # prefix); keypath is only used in the debug log.
        tree_map_fn_copy = tree_map_fn.clone()
        tree_map_fn_copy._maybe_call_tree_map_fastpath = lambda *args, **kwargs: None  # type: ignore[missing-attribute]
        log.debug(
            "tree_map_with_path fastpath fallback triggered for %s (rest=%s, kwargs=%s, keypath=%s)",
            self,
            rest,
            tree_map_kwargs,
            keypath,
        )
        return tree_map_fn_copy.call_function(
            tx,
            [map_fn, self, *rest],
            tree_map_kwargs,
        )

    def set_name_hint(self, name: str) -> None:
        # [FUNCTION DOC] Hook to suggest a variable name for generated graph nodes; base
        # ignores it.
        pass

    def set_source_location(self, source_location: SourceLocation) -> None:
        # [FUNCTION DOC] Record the originating bytecode location on this tracker (mutates
        # the source_location attribute).
        self.source_location = source_location

    def realize(self) -> VariableTracker:
        # [FUNCTION DOC] Force a lazy tracker to build its real tracker; base returns self.
        """Used by LazyVariableTracker to build the real VariableTracker"""
        return self

    def unwrap(self) -> VariableTracker:
        # [FUNCTION DOC] Return the real tracker behind a lazy wrapper if already built
        # (without realizing); base returns self.
        """Used by LazyVariableTracker to return the real VariableTracker if it already exists"""
        return self

    def is_realized(self) -> bool:
        # [FUNCTION DOC] False only for lazy trackers not yet built; base returns True.
        """Used by LazyVariableTracker to indicate an unrealized node"""
        return True

    def tp_iternext_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # [FUNCTION DOC] Next-item slot of an iterator tracker. Base graph breaks (missing
        # iternext, supportable hint); iterator subclasses override.
        """
        Implements tp_iternext slot semantics.
        Subclasses override this to support next(). Reaching this base is a
        bug — it means tp_iternext is missing for that VariableTracker subclass.
        """
        unimplemented(
            gb_type="Missing tp_iternext",
            context=f"next({self.python_type_name()})",
            explanation=(
                f"Dynamo does not support next() on {self.python_type_name()}."
                " Add tp_iternext_impl to this VariableTracker subclass."
            ),
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def next_variable(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # [FUNCTION DOC] Convenience alias that returns the next item via the next-item slot.
        return self.tp_iternext_impl(tx)

    def is_strict_mode(self, tx: InstructionTranslatorBase) -> bool:
        # [FUNCTION DOC] True if the translator has a strict-checks callback and it returns
        # true for this tracker; otherwise False.
        return bool(tx.strict_checks_fn and tx.strict_checks_fn(self))

    def is_mutable(self) -> bool:
        # [FUNCTION DOC] True when mutation is allowed (the negation of is_immutable).
        """Whether Dynamo allows mutation on this variable."""
        return not self.is_immutable()

    def is_immutable(self) -> bool:
        # [FUNCTION DOC] True when the tracker has no mutation type.
        """Whether Dynamo bans mutation on this variable."""
        return self.mutation_type is None

    @staticmethod
    def build(
        tx: InstructionTranslatorBase,
        value: Any,
        source: Source | None = None,
        realize: bool = False,
    ) -> Any:
        # [FUNCTION DOC] Static factory: wrap Python value into a tracker. No source: use the
        # sourceless builder. Source with realize: eagerly build via the sourced builder
        # (installs guards). Source, primitive type: a lazy-constant tracker (deferred
        # guards). Otherwise a lazy tracker. Returns the tracker.
        """Create a new VariableTracker from a value and optional Source"""
        if source is None:
            return builder.SourcelessBuilder.create(tx, value)
        elif realize:
            return builder.VariableBuilder(tx, source)(value)
        elif type(value) in variables.LazyConstantVariable.supported_types:
            # Use LazyConstantVariable for primitives to enable deferred
            # guard installation - constants that are just passed through
            # won't cause recompilation when their values change.
            return variables.LazyConstantVariable.create(value, source)
        else:
            return variables.LazyVariableTracker.create(value, source, tx=tx)

    def get_id(self, tx: InstructionTranslatorBase) -> int | None:
        # [FUNCTION DOC] Identity number of the real object: resolve the source through the
        # output graph and return its id; None if the tracker has no source.
        """Return id() of the underlying Python object, or None if unavailable.

        The base implementation uses source resolution for sourceful VTs.
        Subclasses override for special cases (e.g. NNModuleVariable uses
        get_submodule, ConstantVariable handles singletons).

        Returns None for sourceless VTs — callers use FakeIdVariable in
        that case (see generic_id in object_protocol.py).
        """
        if self.source:
            return id(tx.output.resolve_source_value(self.source))
        return None

    def get_id_guard_type(self) -> Callable[..., Any] | None:
        # [FUNCTION DOC] Guard kind to use when an identity comparison depends on this
        # tracker: the identity-match guard if it has a source, else None.
        if self.source:
            return GuardBuilder.ID_MATCH
        return None

    def get_real_python_backed_value(self) -> object:
        # [FUNCTION DOC] Real Python object for identity comparisons: the constant value if
        # available, else the NO_SUCH_SUBOBJ sentinel.
        """Return the Python object this VT wraps, for `is` comparison.

        Returns NO_SUCH_SUBOBJ if no concrete Python object is available.
        """

        try:
            return self.as_python_constant()
        except NotImplementedError:
            return NO_SUCH_SUBOBJ

    def nb_index_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Integer-conversion-for-indexing slot. Base raises an observed
        # TypeError (cannot be interpreted as an integer); integer-like subclasses override.
        """Mirrors CPython's PyNumber_Index / nb_index slot.

        https://github.com/python/cpython/blob/c09ccd9c429/Objects/abstract.c#L1411-L1450

        The base implementation raises TypeError, matching CPython's behavior
        when tp_as_number->nb_index is NULL (_PyIndex_Check fails).
        """
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                f"'{self.python_type_name()}' object cannot be interpreted as an integer"
            ],
        )

    def tp_repr_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Repr slot returning a string tracker. Base graph breaks (supportable
        # hint) as no override exists.
        """Mirrors CPython's tp_repr slot.

        https://github.com/python/cpython/blob/v3.13.3/Objects/object.c#L745-L778

        Called when type_implements_tp_repr returns True for this type.
        Subclasses override to provide the actual repr implementation.
        """
        unimplemented(
            gb_type="tp_repr_impl not implemented",
            context=f"{type(self).__name__} has tp_repr slot but no tp_repr_impl override",
            explanation=f"The type {self.python_type_name()} has a tp_repr C slot but "
            "the corresponding VariableTracker doesn't implement tp_repr_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def tp_str_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] String-conversion slot returning a string tracker. Base graph breaks
        # (supportable hint); the generic str helper dispatches here and falls back to repr.
        """Dynamo hook for VariableTrackers with dedicated str behavior.
        Subclasses override this for the per-type str result; generic_str()
        handles dispatch and repr fallback.
        """
        unimplemented(
            gb_type="tp_str_impl not implemented",
            context=f"{type(self).__name__} has no tp_str_impl override for {self.python_type_name()}",
            explanation=f"Dynamo does not implement __str__ for {self.python_type_name()} "
            f"in {type(self).__name__}.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_int_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Integer-conversion slot. Base graph breaks (supportable hint).
        """Mirrors CPython's tp_as_number->nb_int slot.

        Called when type_implements_nb_int returns True for this type.
        Subclasses override to provide the actual conversion.
        """
        unimplemented(
            gb_type="nb_int_impl not implemented",
            context=f"{type(self).__name__} has nb_int slot but no nb_int_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_int C slot but "
            "the corresponding VariableTracker doesn't implement nb_int_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_float_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Float-conversion slot. Base graph breaks (supportable hint).
        """Mirrors CPython's tp_as_number->nb_float slot.

        Called when type_implements_nb_float returns True for this type.
        Subclasses override to provide the actual conversion.
        """
        unimplemented(
            gb_type="nb_float_impl not implemented",
            context=f"{type(self).__name__} has nb_float slot but no nb_float_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_float C slot but "
            "the corresponding VariableTracker doesn't implement nb_float_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def _nb_slot_not_implemented(
        self,
        tp_slot_name: str,
        other: VariableTracker,
        reverse: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Graph break (supportable hint) reporting that numeric slot
        # tp_slot_name is not implemented for this type and other; never returns normally.
        unimplemented(
            gb_type="tp_as_number slot not implemented",
            context=f"{tp_slot_name}({self.python_type_name()}, {other.python_type_name()}, reverse={reverse})",
            explanation=f"The type {self.python_type_name()} does not implement the {tp_slot_name}",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_lshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Left-shift slot; base returns the not-implemented constant so
        # dispatch tries the other operand. reverse means self is the right operand.
        """tp_as_number->nb_lshift slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rlshift__`` instead of ``__lshift__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_lshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place left-shift slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_lshift slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_rshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Right-shift slot; base returns the not-implemented constant.
        # reverse means self is the right operand.
        """tp_as_number->nb_rshift slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rrshift__`` instead of ``__rshift__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_rshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place right-shift slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_rshift slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_and_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Bitwise-and slot; base returns the not-implemented constant.
        # reverse means self is the right operand.
        """tp_as_number->nb_and slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rand__`` instead of ``__and__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_and_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place bitwise-and slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_and slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_xor_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Bitwise-xor slot; base returns the not-implemented constant.
        # reverse means self is the right operand.
        """tp_as_number->nb_xor slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rxor__`` instead of ``__xor__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_xor_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place bitwise-xor slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_xor slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_floor_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Floor-division slot; base returns the not-implemented constant.
        # reverse means self is the right operand.
        """tp_as_number->nb_floor_divide slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rfloordiv__`` instead of ``__floordiv__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_floor_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place floor-division slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_floor_divide slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_true_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] True-division slot; base returns the not-implemented constant.
        # reverse means self is the right operand.
        """tp_as_number->nb_true_divide slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rtruediv__`` instead of ``__truediv__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_true_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place true-division slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_true_divide slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_remainder_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Remainder (modulo) slot; base returns the not-implemented constant.
        # reverse means self is the right operand.
        """tp_as_number->nb_remainder slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rmod__`` instead of ``__mod__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_remainder_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place remainder slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_remainder slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_divmod_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Divmod slot; base returns the not-implemented constant. reverse
        # means self is the right operand.
        """tp_as_number->nb_divmod slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rdivmod__`` instead of ``__divmod__``). divmod has no
        in-place form.
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_power_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        z: VariableTracker | None,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Power slot with exponent other and optional modulus z; base returns
        # the not-implemented constant. reverse means self is the right operand.
        return variables.ConstantVariable(NotImplemented)

    def nb_power_z_impl(
        self,
        tx: InstructionTranslatorBase,
        v: VariableTracker,
        w: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Power slot variant used when self is the modulus operand (v is the
        # base, w the exponent); base returns the not-implemented constant.
        """tp_as_number->nb_power slot when self is the modulus (z) operand.

        ``v`` is the base, ``w`` is the exponent, ``self`` is the modulus.
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_power_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        z: VariableTracker | None,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place power slot (exponent other, optional modulus z); base
        # returns the not-implemented constant.
        return variables.ConstantVariable(NotImplemented)

    def nb_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Multiplication slot; base graph breaks (not implemented), unlike
        # most other arithmetic slots which return the not-implemented constant. reverse
        # means self is the right operand.
        """tp_as_number->nb_multiply slot. Default: graph-breaks.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rmul__`` instead of ``__mul__``).  The base default
        graph-breaks so unmigrated VTs surface loudly; ``binary_op1`` only
        invokes this on types whose CPython type advertises the slot.
        """
        return self._nb_slot_not_implemented("nb_multiply_impl", other, reverse=reverse)

    def nb_inplace_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place multiplication slot; base graph breaks.
        """tp_as_number->nb_inplace_multiply slot. Default: graph-breaks."""
        return self._nb_slot_not_implemented("nb_inplace_multiply_impl", other)

    def nb_matrix_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Matrix-multiplication slot; base graph breaks. reverse means self is
        # the right operand.
        """tp_as_number->nb_matrix_multiply slot. Default: graph-breaks.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rmatmul__`` instead of ``__matmul__``).
        """
        return self._nb_slot_not_implemented(
            "nb_matrix_multiply_impl", other, reverse=reverse
        )

    def nb_inplace_matrix_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place matrix-multiplication slot; base graph breaks.
        """tp_as_number->nb_inplace_matrix_multiply slot. Default: graph-breaks."""
        return self._nb_slot_not_implemented("nb_inplace_matrix_multiply_impl", other)

    def sq_repeat_impl(
        self,
        tx: InstructionTranslatorBase,
        count: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Sequence repetition slot; count is an integer tracker. Base graph
        # breaks (supportable hint).
        """tp_as_sequence->sq_repeat slot.

        Implements sequence repetition (e.g. ``[1, 2] * 3``).  ``count`` is
        an int VariableTracker already produced by ``nb_index_impl``.
        """
        unimplemented(
            gb_type="sq_repeat_impl not implemented",
            context=f"{type(self).__name__} has sq_repeat slot but no sq_repeat_impl override",
            explanation=f"The type {self.python_type_name()} has an sq_repeat C slot but "
            "the corresponding VariableTracker doesn't implement sq_repeat_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_inplace_repeat_impl(
        self,
        tx: InstructionTranslatorBase,
        count: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place sequence repetition slot; base graph breaks.
        """tp_as_sequence->sq_inplace_repeat slot."""
        unimplemented(
            gb_type="sq_inplace_repeat_impl not implemented",
            context=f"{type(self).__name__} has sq_inplace_repeat slot but no sq_inplace_repeat_impl override",
            explanation=f"The type {self.python_type_name()} has an sq_inplace_repeat C slot but "
            "the corresponding VariableTracker doesn't implement sq_inplace_repeat_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_or_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Bitwise-or slot; base returns the not-implemented constant. reverse
        # means self is the right operand.
        """tp_as_number->nb_or slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__ror__`` instead of ``__or__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_or_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place bitwise-or slot; base returns the not-implemented constant.
        """tp_as_number->nb_inplace_or slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_negative_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Unary negation slot; base graph breaks (supportable hint).
        """Mirrors CPython's tp_as_number->nb_negative slot.

        Called when type_implements_nb_negative returns True for this type.
        Subclasses override to provide the actual negation.
        """
        unimplemented(
            gb_type="nb_negative_impl not implemented",
            context=f"{type(self).__name__} has nb_negative slot but no nb_negative_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_negative C slot but "
            "the corresponding VariableTracker doesn't implement nb_negative_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_positive_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Unary plus slot; base graph breaks (supportable hint).
        """Mirrors CPython's tp_as_number->nb_positive slot.

        Called when type_implements_nb_positive returns True for this type.
        Subclasses override to provide the actual positive.
        """
        unimplemented(
            gb_type="nb_positive_impl not implemented",
            context=f"{type(self).__name__} has nb_positive slot but no nb_positive_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_positive C slot but "
            "the corresponding VariableTracker doesn't implement nb_positive_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_add_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Addition slot; base returns the not-implemented constant so the
        # sequence-concatenation slot or the other operand can be tried. reverse means self
        # is the right operand.
        # KNOWN ISSUE: the docstring says graph break, but the code returns the
        # not-implemented constant.
        """tp_as_number->nb_add slot. Default: graph break.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__radd__`` instead of ``__add__``).
        """
        # We need to return NotImplemented here because of sq_concat
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_add_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place addition slot; base returns the not-implemented constant so
        # in-place concatenation can be tried.
        # KNOWN ISSUE: the docstring says graph break, but the code returns the
        # not-implemented constant.
        """tp_as_number->nb_inplace_add slot. Default: graph break."""
        # We need to return NotImplemented here because of sq_inplace_concat
        return variables.ConstantVariable(NotImplemented)

    def nb_subtract_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Subtraction slot; base returns the not-implemented constant. reverse
        # means self is the right operand.
        # KNOWN ISSUE: the docstring says graph break, but the code returns the
        # not-implemented constant.
        """tp_as_number->nb_subtract slot. Default: graph break.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rsub__`` instead of ``__sub__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_subtract_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] In-place subtraction slot; base returns the not-implemented constant.
        # KNOWN ISSUE: the docstring says graph break, but the code returns the
        # not-implemented constant.
        """tp_as_number->nb_inplace_subtract slot. Default: graph break."""
        return variables.ConstantVariable(NotImplemented)

    def nb_absolute_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Absolute-value slot; base graph breaks (supportable hint).
        """Mirrors CPython's tp_as_number->nb_absolute slot.

        Called when type_implements_nb_absolute returns True for this type.
        Subclasses override to provide the actual absolute value.
        """
        unimplemented(
            gb_type="nb_absolute_impl not implemented",
            context=f"{type(self).__name__} has nb_absolute slot but no nb_absolute_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_absolute C slot but "
            "the corresponding VariableTracker doesn't implement nb_absolute_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_invert_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # [FUNCTION DOC] Bitwise-inversion slot; base graph breaks (supportable hint).
        """Mirrors CPython's tp_as_number->nb_invert slot.

        Called when type_implements_nb_invert returns True for this type.
        Subclasses override to provide the actual bitwise inversion.
        """
        unimplemented(
            gb_type="nb_invert_impl not implemented",
            context=f"{type(self).__name__} has nb_invert slot but no nb_invert_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_invert C slot but "
            "the corresponding VariableTracker doesn't implement nb_invert_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def __init__(
        self,
        *,
        source: Source | None = None,
        mutation_type: MutationType | None = None,
        source_location: SourceLocation | None = None,
        dict_vt: variables.DunderDictVariable | None = None,
    ) -> None:
        # [FUNCTION DOC] Store source, source_location, mutation_type and dict_vt (keyword
        # only). Validates consistency: a New mutation type requires no source, an Existing
        # one requires a source and must be an Existing type, else raises AssertionError.
        super().__init__()
        self.source = source
        self.source_location = source_location
        self.mutation_type = mutation_type
        # Carried so clone() round-trips the cached __dict__ view.
        self.dict_vt = dict_vt

        # NOTE sometimes mutation_type is set afterwards for implementation
        # convenience, we don't validate those cases at the moment.
        if mutation_type is not None:
            if isinstance(mutation_type, (ValueMutationNew, AttributeMutationNew)):
                # If this fails, it's either
                # 1. one mistakenly passed in a source
                # 2. `mutation_type` is incorrect
                if source is not None:
                    raise AssertionError(
                        "source must be None for ValueMutationNew/AttributeMutationNew"
                    )
            else:
                if not isinstance(
                    mutation_type, (ValueMutationExisting, AttributeMutationExisting)
                ):
                    raise AssertionError(
                        f"Expected ValueMutationExisting or AttributeMutationExisting, "
                        f"got {type(mutation_type)}"
                    )
                # If this fails, it's either
                # 1. one forgot to pass in a source
                # 2. `mutation_type` is incorrect
                if source is None:
                    raise AssertionError(
                        "source must not be None for ValueMutationExisting/AttributeMutationExisting"
                    )

    @property
    def tp_type(self) -> PyTypeObject:
        # [FUNCTION DOC] Property: the full slot table of the represented Python type
        # (graph breaks if the type is unknown).
        return _tp_type(maybe_get_python_type(self))

    @property
    def tp_as_number(self) -> PyNumberMethods:
        # [FUNCTION DOC] Property: number-slot table of the represented Python type.
        return _tp_type(maybe_get_python_type(self)).tp_as_number

    @property
    def tp_as_sequence(self) -> PySequenceMethods:
        # [FUNCTION DOC] Property: sequence-slot table of the represented Python type.
        return _tp_type(maybe_get_python_type(self)).tp_as_sequence

    @property
    def tp_as_mapping(self) -> PyMappingMethods:
        # [FUNCTION DOC] Property: mapping-slot table of the represented Python type.
        return _tp_type(maybe_get_python_type(self)).tp_as_mapping

    @property
    def tp_iter(self) -> Slot | None:
        # [FUNCTION DOC] Property: iteration slot of the represented type, or None.
        return _tp_type(maybe_get_python_type(self)).tp_iter

    @property
    def tp_iternext(self) -> Slot | None:
        # [FUNCTION DOC] Property: next-item slot of the represented type, or None.
        return _tp_type(maybe_get_python_type(self)).tp_iternext

    @property
    def tp_str(self) -> Slot | None:
        # [FUNCTION DOC] Property: string-conversion slot of the represented type, or None.
        return _tp_type(maybe_get_python_type(self)).tp_str

    @property
    def tp_repr(self) -> Slot | None:
        # [FUNCTION DOC] Property: repr slot of the represented type, or None.
        return _tp_type(maybe_get_python_type(self)).tp_repr

    @property
    def _slotdefs(self):
        # [FUNCTION DOC] HELPER: Property: mapping from dunder name to the SlotDef the
        # represented Python type implements (cached per type).
        return _compute_slotdefs(maybe_get_python_type(self))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # [FUNCTION DOC] Class-creation hook: for each new subclass, wrap its own
        # as_python_constant and reconstruct with re-entrancy guards. A repeated call on the
        # same object in the same call chain raises a not-a-constant error (constant case) or
        # graph breaks (reconstruct case). Modifies the subclass in place.
        """
        Wraps all subclasses' `as_python_constant` and `reconstruct` so that it cannot be
        called twice in the same call chain - i.e. self-referential objects.
        For `as_python_constant` - self-referential objects are NOT treated as constants.
        For `reconstruct` - we will graph break. The graph break can be avoided if the VT subclass
        can generate and cache itself before recursively `reconstruct`ing - see ListVariable for an example.
        """
        super().__init_subclass__(**kwargs)

        def as_python_constant_failure(self) -> NoReturn:
            # [FUNCTION DOC] HELPER: Always raises AsPythonConstantNotImplementedError saying
            # the tracker is self-referential.
            raise AsPythonConstantNotImplementedError(
                self, msg=f"{self} is self-referential"
            )

        VariableTracker._add_call_once_guard(
            cls, "as_python_constant", as_python_constant_failure
        )

        def reconstruct_failure(self) -> NoReturn:
            # [FUNCTION DOC] HELPER: Always graph breaks (unimplemented) explaining that a
            # self-referential sourceless tracker cannot be reconstructed.
            unimplemented(
                gb_type="Reconstruction failure (self-referential)",
                context=str(self),
                explanation=f"Dynamo tried to reconstruct sourceless variable {self}, but it is self-referential. "
                "Dynamo must manually implement reconstruction rules for self-referentiable sourceless variables.",
                hints=[
                    "If Dynamo is attempting to trace a return statement and your code is attempting to return a variable "
                    "that Dynamo cannot reconstruct, then remove it from the return statement.",
                    "Remove the self-reference in the variable. A self-referring list, for example, is `l = []; l.append(l)`.",
                    *graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK,
                    "Report an issue to PyTorch if you need self-referential reconstrtuction support.",
                ],
            )

        VariableTracker._add_call_once_guard(cls, "reconstruct", reconstruct_failure)

    @staticmethod
    def _add_call_once_guard(
        cls: type[VariableTracker],
        method: str,
        callback: Callable[[VariableTracker], Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Replace method on cls with a guarded version that tracks
        # in-progress calls per (object, original method) in a context variable; on a
        # re-entrant call it invokes callback (which raises). Skips methods inherited
        # unchanged from the base or already guarded. Mutates cls.
        original_method = getattr(cls, method)

        if original_method is getattr(VariableTracker, method) or hasattr(
            original_method, "_call_once_guarded"
        ):
            return

        @functools.wraps(original_method)
        def guarded_method(self, *args: Any, **kwargs: Any) -> VariableTracker:
            # [FUNCTION DOC] HELPER: Run the original method, registering the call in the
            # active-call set (created on first use); if already active, call the failure
            # callback first. Always unregisters in a finally step.
            active = _vt_active_calls.get()
            if active is None:
                active = set()
                _vt_active_calls.set(active)

            key = (id(self), id(original_method))
            if key in active:
                callback(self)

            active.add(key)
            try:
                return original_method(self, *args, **kwargs)
            finally:
                active.discard(key)

        guarded_method._call_once_guarded = True  # pyrefly: ignore[missing-attribute]
        setattr(cls, method, guarded_method)


def typestr(*objs: object) -> str:
    # [FUNCTION DOC] Describe objs for messages: a single tracker gives its text form,
    # a single other object gives its type name, and several are joined by spaces.
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))


instancecheck = type.__instancecheck__


_CPYTHON_BASE_URL = "https://github.com/python/cpython/blob/v3.13.0/"


def print_cpython_to_vt_mapping() -> None:
    # [FUNCTION DOC] Debug utility: import all tracker modules so the subclass registry is
    # complete, group registered classes by the CPython type they declare themselves (own
    # attribute only), and print one line per pair sorted by type name. Writes to stdout.
    """Print the mapping from CPython types to Dynamo VariableTracker classes.

    Reads _cpython_type from each VT subclass (own attribute only, not inherited).
    """
    # Ensure all VT modules are imported so all_subclasses is complete
    from . import (  # noqa: F401
        builtin,
        constant,
        ctx_manager,
        dicts,
        distributed,
        functions,
        higher_order_ops,
        iter,
        lazy,
        lists,
        misc,
        nn_module,
        streams,
        tensor,
        torch,
        torch_function,
        user_defined,
    )

    mapping: dict[type, list[type]] = {}
    for vt_cls in VariableTrackerMeta.all_subclasses:
        cpython_type = vt_cls.__dict__.get("_cpython_type")
        if cpython_type is None:
            continue
        types = cpython_type if isinstance(cpython_type, tuple) else (cpython_type,)
        for t in types:
            mapping.setdefault(t, []).append(vt_cls)

    for py_type, vt_classes in sorted(mapping.items(), key=lambda x: x[0].__qualname__):
        for vt_cls in vt_classes:
            print(
                f"{py_type.__module__}.{py_type.__qualname__:30s} -> {vt_cls.__name__}"
            )


from . import builder
from .lazy import LazyVariableTracker
