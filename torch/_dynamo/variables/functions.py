# [MODULE DOC]
# Variable trackers for callable objects used by Dynamo symbolic execution:
# user functions/methods, nested (closure-creating) functions, generators,
# skipped functions, functools.partial, polyfilled functions, wrappers
# (torch.jit/lru_cache style), collective rewrites, Triton kernels and TMA
# descriptors, and trackers modelling CPython descriptor types (wrapper/method/
# member/getset descriptors, staticmethod, classmethod, property, tuplegetter).
# Also holds argument binding helpers (FunctionSpec, bind_args_cached).
# Import-time side effects: creates a module-level weak cache of function specs,
# imports the Triton HOP helpers, and optionally imports the FSDP param group
# module (set to None when unavailable).
"""
Function-related variable tracking classes for Dynamo's symbolic execution.

This module contains classes that track different types of functions during graph
compilation, including:
- User-defined functions and methods
- Built-in functions and methods
- Wrapped functions (e.g. from decorators)
- Special function types (e.g. functools.partial)
- Triton kernels and related function types

These classes are responsible for:
- Tracking function calls and their arguments
- Managing function closures and cell variables
- Handling function attributes and special methods
- Maintaining guards for function identity and closure contents
- Supporting function inlining and specialization
- Enabling proper symbolic execution of different function types

The variable trackers here work together with the rest of Dynamo to enable
accurate graph capture while handling Python's various function-related behaviors.
"""

import _collections  # type: ignore[import-not-found]
import builtins
import collections
import functools
import importlib.metadata
import importlib.util
import inspect
import itertools
import logging
import os
import re
import sys
import traceback
import types
import typing
from collections.abc import Callable, Sequence
from types import CellType, FunctionType
from typing import Any, cast, Literal, Optional, TYPE_CHECKING, TypeVar
from typing_extensions import Never
from weakref import WeakKeyDictionary

import torch
from torch._dynamo.exc import get_stack_above_dynamo
from torch._guards import Source
from torch.utils._pytree import is_namedtuple_class

from .. import config, graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function, create_rot_n, is_generator
from ..exc import (
    format_frame_info,
    get_dynamo_observed_exception,
    InfiniteGeneratorError,
    ObservedException,
    ObservedGeneratorExit,
    ObservedUserStopIteration,
    raise_observed_exception,
    raise_type_error,
    raise_value_error,
    StepUnsupported,
    unimplemented,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    CellContentsSource,
    ClosureSource,
    ConstantSource,
    DefaultsSource,
    GetItemSource,
    ImportSource,
    SkipGuardSource,
    TypeMROSource,
    TypeSource,
)
from ..utils import (
    check_constant_args,
    check_unspec_or_constant_args,
    FrameState,
    identity,
    is_function,
    is_lru_cache_wrapper_trace_without_warning_allowed,
    is_tensor_base_attr_getter,
    is_wrapper_or_member_descriptor,
    istype,
    make_cell,
    unpack_iterable,
)
from .base import (
    AsPythonConstantNotImplementedError,
    AttributeMutationNew,
    GetSet,
    getset_build,
    getset_read,
    Member,
    Method,
    NO_SUCH_SUBOBJ,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable
from .user_defined import UserDefinedObjectVariable


try:
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import (
        InliningGeneratorInstructionTranslator,
        InliningInstructionTranslator,
        InstructionTranslatorBase,
    )
    from torch._dynamo.variables.ctx_manager import ContextWrappingVariable
    from torch._higher_order_ops.triton_kernel_wrap import (
        TritonGridType,
        TritonKernelType,
    )

    from .lists import BaseListVariable, ListVariable
    from .tensor import TensorVariable


_F = TypeVar("_F", bound=Callable[..., Any])
CO_VARARGS = 0x04
CO_VARKEYWORDS = 0x08
_SUPPORTED_TREE_MAP_KWARGS = frozenset({"namespace", "none_is_leaf", "is_leaf"})
_TREE_MAP_ONLY_SUPPORTED_KWARGS = frozenset({"is_leaf"})

PT2_ISSUE_TRACKER_URL = "https://github.com/pytorch/pytorch/issues/new?&labels=oncall%3A+pt2&projects=&template=pt2-bug-report.yml"

# Module-level cache keyed by the function object
_spec_cache: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


# Raised when get_function() cannot convert a nested function to a Python function.
class ClosureConversionError(NotImplementedError):
    # [CLASS DOC] HELPER: Error signalling that a nested function (or its closure
    # cells) cannot be converted into a real Python function; subclass of
    # NotImplementedError so callers treating it as "not a constant" still work.
    pass


@functools.lru_cache
def get_pytree_SUPPORTED_NODES_source() -> AttrSource:
    # [FUNCTION DOC] Returns (cached) the attribute source that names the pytree
    # SUPPORTED_NODES registry reached from the torch import source.
    # Callers must install the identity guard on the embedded torch import source.
    # Cached, so callers are responsible for installing the ID_MATCH guard on the
    # embedded ImportSource("torch") themselves.
    return AttrSource(
        AttrSource(AttrSource(ImportSource("torch"), "utils"), "_pytree"),
        "SUPPORTED_NODES",
    )


class FunctionSpec:
    # [CLASS DOC] HELPER: Cached, per-function description of the parameter layout
    # (positional-only, positional-or-keyword, keyword-only names, varargs and
    # varkw names, counts) derived from the code object, plus defaults that are
    # refreshed on every call via update_defaults. Used by bind_args_cached.
    def __init__(self, func: FunctionType) -> None:
        # [FUNCTION DOC] HELPER: Reads the code object of func and records the
        # parameter counts and name tuples, and the names of *args / **kwargs
        # parameters (or None when absent). Defaults are not read here.
        code = func.__code__
        vn = code.co_varnames

        self.posonly_count = code.co_posonlyargcount
        self.arg_count = code.co_argcount
        self.kwonly_count = code.co_kwonlyargcount

        self.posonly_names = vn[: self.posonly_count]
        self.pos_or_kw_names = vn[self.posonly_count : self.arg_count]
        self.all_pos_names = self.posonly_names + self.pos_or_kw_names
        self.kwonly_names = vn[self.arg_count : self.arg_count + self.kwonly_count]

        off = self.arg_count + self.kwonly_count
        self.varargs_name = vn[off] if code.co_flags & CO_VARARGS else None
        off += 1 if self.varargs_name else 0
        self.varkw_name = vn[off] if code.co_flags & CO_VARKEYWORDS else None

    def update_defaults(self, func: FunctionType) -> None:
        # [FUNCTION DOC] HELPER: Refreshes positional defaults, keyword-only
        # defaults, and the mapping from positional parameter name to its index
        # in the defaults tuple. Mutates self; called on every bind since
        # defaults may be reassigned on the function.
        # Defaults can change from function call to function call. So re-update
        # them on every call.
        self.defaults = func.__defaults__ or ()
        self.kwdefaults = func.__kwdefaults__ or {}

        # Map positional-default names → their index in self.defaults
        self.pos_default_map = dict(
            zip(self.all_pos_names[-len(self.defaults) :], range(len(self.defaults)))
        )


def _get_spec(func: FunctionType) -> FunctionSpec:
    # [FUNCTION DOC] HELPER: Returns the FunctionSpec for func, creating and
    # storing it in the module-level weak cache on first use.
    spec = _spec_cache.get(func)
    if spec is None:
        spec = FunctionSpec(func)
        _spec_cache[func] = spec
    return spec


class BindArgsTypeError(TypeError):
    # [CLASS DOC] HELPER: TypeError raised when call arguments cannot be bound to
    # a function signature (missing, extra or unexpected arguments).
    pass


def bind_args_cached(
    func: FunctionType,
    tx: "InstructionTranslatorBase",
    fn_source: Source | None,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> dict[str, VariableTracker]:
    # [FUNCTION DOC] Binds call arguments (args list, kwargs mapping) to the
    # parameter names of func, returning a mapping from local name to
    # VariableTracker. Has a fast path for plain positional calls without
    # defaults/varargs/kwargs. The full path binds positionals (allowing keyword
    # fill of non-positional-only names), then defaults (installing a length guard
    # on __defaults__ when fn_source is given, and lazy sourced default values),
    # then varargs tuple, keyword-only arguments and the varkw mapping.
    # Raises BindArgsTypeError on missing, too many, or unexpected arguments.
    # Side effects: may install guards; wraps raw values via wrap_bound_arg.
    spec = _get_spec(func)

    # Fast path: simple positional-only, no defaults, no varargs/varkw
    # This is the common case for small utility functions called repeatedly.
    if (
        len(args) == spec.arg_count
        and not func.__defaults__
        and not kwargs
        and not spec.varargs_name
        and not spec.varkw_name
        and not spec.kwonly_names
    ):
        return {
            name: wrap_bound_arg(tx, args[i])
            for i, name in enumerate(spec.all_pos_names)
        }

    # Full path with all features
    spec.update_defaults(func)
    ba = {}
    rem_kw = dict(kwargs)
    guarded_pos_defaults_len = False

    # 1) Bind all positional (pos-only + pos-or-kw)
    for i, name in enumerate(spec.all_pos_names):
        if i < len(args):
            ba[name] = wrap_bound_arg(tx, args[i])
        elif name in rem_kw and (
            # `kwargs` can have the same key as a pos-only arg `name`.
            # If this case happens, we should not consume the `name` here and
            # keep it in `kwargs`:
            #   >>> def fn(a, /, **kwargs): return (a, kwargs)
            #   >>> fn(1, a=2)
            #   (1, {'a': 2})
            name not in spec.posonly_names
        ):
            ba[name] = wrap_bound_arg(tx, rem_kw.pop(name))
        elif name in spec.pos_default_map:
            idx = spec.pos_default_map[name]
            if fn_source and not guarded_pos_defaults_len:
                # The parameter-to-default mapping depends on __defaults__
                # length; guard it without wrapping every default value.
                install_guard(
                    AttrSource(fn_source, "__defaults__").make_guard(
                        GuardBuilder.SEQUENCE_LENGTH
                    )
                )
                guarded_pos_defaults_len = True
            default_source = None
            if fn_source and not (
                ConstantVariable.is_literal(spec.defaults[idx])
                and config.skip_guards_on_constant_func_defaults
            ):
                default_source = DefaultsSource(fn_source, idx)
            ba[name] = wrap_bound_arg(tx, spec.defaults[idx], default_source)
        else:
            raise BindArgsTypeError(f"missing required positional argument: {name}")

    # 2) *args
    extra = args[len(spec.all_pos_names) :]
    if spec.varargs_name:
        ba[spec.varargs_name] = wrap_bound_arg(tx, tuple(extra))
    elif extra:
        raise BindArgsTypeError(
            f"Too many positional arguments: got {len(args)}, expected {len(spec.all_pos_names)}"
        )

    # 3) Keyword-only
    for name in spec.kwonly_names:
        if name in rem_kw:
            ba[name] = wrap_bound_arg(tx, rem_kw.pop(name))
        elif name in spec.kwdefaults:
            kwdefault_source = None
            if fn_source:
                kwdefault_source = DefaultsSource(fn_source, name, is_kw=True)
            ba[name] = wrap_bound_arg(tx, spec.kwdefaults[name], kwdefault_source)
        else:
            raise BindArgsTypeError(f"Missing required keyword-only argument: {name}")

    # 4) **kwargs
    if spec.varkw_name:
        ba[spec.varkw_name] = wrap_bound_arg(tx, rem_kw)
    elif rem_kw:
        raise BindArgsTypeError(f"Unexpected keyword arguments: {list(rem_kw)}")

    return ba


def wrap_bound_arg(
    tx: "InstructionTranslatorBase", val: Any, source: Source | None = None
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Converts val to a VariableTracker. Existing trackers
    # are returned unchanged; raw values without a source are built eagerly; raw
    # values with a source become lazy trackers so no guard is installed until
    # the value is actually used.
    # Source propagation is best effort since not every object we encounter has a source to begin with.
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        return VariableTracker.build(tx, val)
    else:
        # Create a lazy variable to avoid guarding on __defaults__ unless really
        # needed.
        return variables.LazyVariableTracker.create(val, source, tx=tx)


def wrap_args_kwargs(tx: "InstructionTranslatorBase", result: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: In place, replaces every tuple or dict entry in the
    # result mapping (the packed varargs / varkw values) with a VariableTracker.
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v)


def init_cellvars(
    parent: "InstructionTranslatorBase",
    result: dict[str, VariableTracker],
    code: types.CodeType,
) -> None:
    # [FUNCTION DOC] HELPER: For each cell variable name of code, creates a new
    # tracked cell in the parent's side effects and stores it in result. If the
    # name is already an argument in result, its value is moved into the new cell.
    """
    Update `result` to add mapping from local name to new cells created
    directly by `code`, or update SideEffects in `parent` if a local cell is
    already in `result` (cell argument).
    """
    side_effects = parent.output.side_effects

    for name in code.co_cellvars:
        new_cell = side_effects.track_cell_new()
        if name in result:
            # This handles when a function argument is a cell (e.g., captured by
            # a nested func). See `MAKE_CELL` bytecode for more info.
            side_effects.store_cell(new_cell, result.pop(name))
        result[name] = new_cell


def _create_nested_fn(
    code: types.CodeType,
    f_globals: dict[str, Any],
    name: str,
    defaults: tuple[object, ...] | None,
    closure: tuple[CellType] | None,
    kwdefaults: dict[str, Any] | None,
    annotations: dict[str, Any] | None,
) -> types.FunctionType:
    # [FUNCTION DOC] HELPER: Runtime constructor used by reconstructed bytecode to
    # build a nested function from code, globals, name, defaults, closure,
    # keyword defaults and annotations (a flat tuple of key/value pairs is
    # converted to a dict). Raises an error if annotations is neither None nor a
    # mapping. Returns the new function.
    from types import FunctionType

    func = FunctionType(code, f_globals, name, defaults, closure)
    func.__kwdefaults__ = kwdefaults

    if isinstance(annotations, tuple):
        from itertools import pairwise

        annotations = dict(pairwise(annotations))

    # TypeError: __annotations__ must be set to a dict object
    if not (annotations is None or isinstance(annotations, dict)):
        raise AssertionError(
            f"annotations must be None or a dict, got {type(annotations)}"
        )
    func.__annotations__ = annotations  # type: ignore[assignment]

    return func


fn_known_dunder_attrs = {
    "__annotations__",
    "__builtins__",
    "__closure__",
    "__code__",
    "__defaults__",
    "__doc__",
    "__globals__",
    "__kwdefaults__",
    "__name__",
    "__module__",
}


def fn_getattro_impl(
    tx: "InstructionTranslatorBase", fn: object, source: Source | None, name: str
) -> VariableTracker:
    # [FUNCTION DOC] Implements attribute lookup on a real function object: looks
    # up name statically (raises an observed AttributeError if absent), reads
    # known dunder attributes through normal getattr, and returns a lazy tracker
    # when a source exists (annotation sources skip guards) else a built tracker.
    # KNOWN ISSUE: per the TODO, instance dict entries (e.g. set by partial-like
    # wrappers) are not consulted before the known-dunder shortcut.
    source = source and AttrSource(source, name)

    if source and name == "__annotations__":
        # We get a large number of silly guards from annotations from inspect
        # module. Changing annotations is rare, and it impacting the extracted
        # graph is even rarer. So skip guards.
        source = SkipGuardSource(source)

    subobj = None
    try:
        subobj = inspect.getattr_static(fn, name)
    except AttributeError:
        # function does not have a __getattr__ or __getattribute__ method,
        # so we can safely assume that this attribute is absent
        raise_observed_exception(AttributeError, tx)

    # Special handling for known dunder attributes
    # TODO(guilhermeleobas): this check should go through fn.__dict__ first as
    # functools.partial can override it
    if name in fn_known_dunder_attrs:
        subobj = getattr(fn, name)
    if source:
        return variables.LazyVariableTracker.create(subobj, source, tx=tx)
    return VariableTracker.build(tx, subobj)


class BaseUserFunctionVariable(VariableTracker):
    # [CLASS DOC] Abstract base for trackers of Python-level callables. Provides
    # function attribute plumbing (dunder getset/member tables, annotations slot,
    # instance dict lookup, hasattr), setattr/delattr handling and the default
    # call path that inlines the function body. Subclasses supply get_code,
    # get_function, get_globals and has_self. The annotations attribute is a
    # per-instance lazily created tracker (or None).
    # funcobject.c func_annotations: a dedicated slot, NOT a __dict__ entry.
    # Materialized per-instance once accessed/assigned.
    annotations: "VariableTracker | None" = None

    def tp_richcompare_impl(self, tx, other, op):
        # [FUNCTION DOC] Rich comparison with identity semantics via the generic
        # object comparison helper (equality by identity, ordering unsupported).
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def self_args(self) -> list[VariableTracker]:
        # [FUNCTION DOC] Returns the list of implicit leading arguments (bound self);
        # empty for plain functions.
        return []

    def get_source(self) -> Source | None:
        # [FUNCTION DOC] Returns the source of this tracker (may be None).
        return self.source

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] Returns a constant string tracker holding the repr of the
        # underlying python constant function (so it must be convertible).
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/funcobject.c
        return VariableTracker.build(tx, repr(self.as_python_constant()))

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handles setattr and delattr on the function: the
        # annotations attribute is stored/cleared in the dedicated slot (returns a
        # None constant); any other name is written to or deleted from the
        # function's tracked dict. Other method names defer to the parent class.
        if name == "__setattr__":
            if args[0].is_constant_match("__annotations__"):
                self.annotations = args[1]
                return ConstantVariable.create(None)
            return self.get_dict_vt(tx).call_method(
                tx, "__setitem__", list(args), kwargs
            )
        elif name == "__delattr__":
            if args[0].is_constant_match("__annotations__"):
                self.annotations = None
                return ConstantVariable.create(None)
            return self.get_dict_vt(tx).call_method(tx, "__delitem__", list(args), {})
        return super().call_method(tx, name, list(args), kwargs)

    def get_filename(self) -> str:
        # [FUNCTION DOC] Returns the source file name of the function's code object.
        return self.get_code().co_filename

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the code object's name.
        return self.get_code().co_name

    def get_qualname(self) -> str:
        # [FUNCTION DOC] Returns the qualified name from the code object on
        # Python 3.11+, otherwise falls back to the plain name.
        if sys.version_info >= (3, 11):
            return self.get_code().co_qualname
        else:
            return self.get_name()

    def get_doc(self) -> str | None:
        # [FUNCTION DOC] Returns the first constant of the code object, which holds
        # the docstring.
        # KNOWN ISSUE: co_consts[0] is not guaranteed to be a docstring on newer
        # Python versions or for functions without one (may be another constant,
        # and raises on empty constants).
        # stored in code.co_consts[0]
        return self.get_code().co_consts[0]

    def get_globals(self) -> dict[str, Any]:
        # [FUNCTION DOC] Abstract: subclasses return the function's globals mapping.
        raise NotImplementedError

    def get_code(self) -> types.CodeType:
        # [FUNCTION DOC] Abstract: subclasses return the function's code object.
        raise NotImplementedError

    def has_self(self) -> bool:
        # [FUNCTION DOC] Abstract: whether the callable is bound to a self object.
        raise NotImplementedError

    def get_function(self) -> types.FunctionType:
        # [FUNCTION DOC] Abstract: returns a real Python function for this tracker.
        raise NotImplementedError

    def get_module(self) -> str:
        # [FUNCTION DOC] Returns the module name taken from the globals mapping.
        return self.get_globals()["__name__"]

    def _get_defaults(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the defaults attribute: the stored
        # defaults tracker, or a None constant if absent.
        d = getattr(self, "defaults", None)
        return d if d is not None else ConstantVariable.create(None)

    def _get_named_attr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Returns a dunder attribute (name, qualname, code,
        # doc, module) preferring an override stored in the function's dict;
        # otherwise calls the matching accessor and wraps the result as a constant
        # with an attribute source derived from this tracker's source.
        fn_dict = self.get_dict_vt(tx)
        if fn_dict.contains(name):
            return fn_dict.getitem(name)
        val = getattr(self, f"get_{name[2:-2]}")()
        return ConstantVariable.create(
            val, source=self.source and AttrSource(self.source, name)
        )

    def _get_annotations(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for annotations; lazily creates and stores
        # a fresh, sourceless, mutable empty dict tracker and returns it.
        # func_get_annotations lazily creates and stores an empty dict. The dict
        # is a fresh value (ValueMutationNew), so it must carry no source.
        if self.annotations is None:
            self.annotations = variables.ConstDictVariable(
                {}, mutation_type=ValueMutationNew()
            )
        return self.annotations

    def _get_type_params(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for type params: reads the entry from the
        # function dict, defaulting to a new empty mutable tuple tracker.
        return self.get_dict_vt(tx).getitem_or_default(
            "__type_params__",
            lambda: variables.TupleVariable([], mutation_type=ValueMutationNew()),
        )

    def _get_kwdefaults(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for keyword defaults: stored tracker or a
        # None constant.
        d = getattr(self, "kwdefaults", None)
        return d if d is not None else ConstantVariable.create(None)

    def _get_closure(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the closure: stored tracker or a None
        # constant.
        c = getattr(self, "closure", None)
        return c if c is not None else ConstantVariable.create(None)

    tp_getset = {
        "__defaults__": GetSet(_get_defaults, None),
        "__kwdefaults__": GetSet(_get_kwdefaults, None),
        "__name__": GetSet(lambda s, tx: s._get_named_attr(tx, "__name__")),
        "__qualname__": GetSet(lambda s, tx: s._get_named_attr(tx, "__qualname__")),
        "__code__": GetSet(lambda s, tx: s._get_named_attr(tx, "__code__")),
        "__dict__": GetSet(lambda s, tx: s.get_dict_vt(tx)),
        "__annotations__": GetSet(_get_annotations),
        "__type_params__": GetSet(_get_type_params),
    }
    tp_members = {
        "__doc__": Member(lambda s, tx: s._get_named_attr(tx, "__doc__")),
        "__module__": Member(lambda s, tx: s._get_named_attr(tx, "__module__")),
        "__closure__": Member(_get_closure),
    }

    def lookup_instance_dict(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] Looks up an arbitrary attribute in the real function's
        # instance dict. Returns None when the function cannot be materialized or
        # the name is absent; otherwise a lazy tracker (with attribute source) or
        # a built tracker for the value.
        # Arbitrary attributes set on a function object live in its __dict__
        # (tp_dictoffset). Resolve them at the instance-dict step so getattr
        # finds them before call_getattr_fallback (which raises). Known dunder
        # slots are handled earlier via tp_getset/tp_members and never reach here.
        try:
            fn = self.get_function()
        except NotImplementedError:
            return None
        fn_dict = getattr(fn, "__dict__", None)
        if not fn_dict or name not in fn_dict:
            return None
        source = self.get_source()
        source = AttrSource(source, name) if source is not None else None
        if source is not None:
            return variables.LazyVariableTracker.create(fn_dict[name], source, tx=tx)
        return VariableTracker.build(tx, fn_dict[name])

    def call_getattr_fallback(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] Fallback when no attribute was found: always raises an
        # observed AttributeError carrying the name.
        # Functions have no __getattr__; reaching the fallback means the
        # attribute genuinely does not exist (CPython
        # _PyObject_GenericGetAttrWithDict step 7 raises AttributeError).
        raise_observed_exception(AttributeError, tx, args=[name])

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the function by inlining its body with any bound self
        # args prepended (nested graph breaks allowed). The lr_scheduler step-
        # patching helper is skipped and returns a None constant.
        # Ignore patch_track_step_called from torch/optim/lr_scheduler.py - it just patches
        # the optimizer.step method and we don't need to trace it
        if (
            self.get_name() == "patch_track_step_called"
            and self.get_filename().endswith("torch/optim/lr_scheduler.py")
        ):
            return ConstantVariable.create(None)
        return tx.inline_user_function_return(
            self,
            [*self.self_args(), *args],
            kwargs,
            allow_nested_graph_breaks=True,
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # [FUNCTION DOC] hasattr: first consults tracked side effects; known dunder
        # names and the dict attribute are always present; otherwise asks the real
        # function (false if it cannot be materialized). Returns a bool tracker.
        se_result = self._hasattr_check_side_effects(tx, name)
        if se_result is not None:
            return se_result

        result = False

        if name in fn_known_dunder_attrs or name == "__dict__":
            result = True
        else:
            try:
                result = hasattr(self.get_function(), name)  # type: ignore[attr-defined]
            except NotImplementedError:
                result = False
        return VariableTracker.build(tx, result)

    def closure_vars(
        self, tx: "InstructionTranslatorBase"
    ) -> dict[str, VariableTracker]:
        # [FUNCTION DOC] Returns the closure variable mapping; empty by default.
        return {}

    # Override to set whether or not nested graph breaks should be allowed
    # if we create an inlining tx for this BaseUserFunctionVariable.
    # See symbolic_convert.py for where this function is called.
    def should_allow_nested_graph_breaks(self) -> bool:
        # [FUNCTION DOC] Whether an inlining translator for this function may take
        # nested graph breaks; True by default.
        return True


class UserFunctionVariable(BaseUserFunctionVariable):
    # [CLASS DOC] Tracker for a real Python function object (attribute fn). Holds
    # an is_constant flag (function marked as returning a constant). Handles
    # special Dynamo API functions in call_function, inlines others, binds
    # arguments and closure cells (bind_args), binds as methods through the
    # descriptor protocol, and has a fast path for tree_map style pytree calls.
    """Some unsupported user-defined global function"""

    # PyFunction_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/funcobject.c#L1046
    _cpython_type = types.FunctionType

    _nonvar_fields = {
        "fn",
        "is_constant",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    _TREE_MAP_MODULES = frozenset(
        {
            "optree",
            "optree.ops",
            "torch.utils._pytree",
            "torch.utils._cxx_pytree",
        }
    )

    @classmethod
    def create_with_source(cls, value: Any, source: Any) -> "UserFunctionVariable":
        # [FUNCTION DOC] Factory: installs a closure-identity guard (code object
        # match) on source and returns a new tracker for value with that source.
        install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        return cls(value, source=source)

    def get_value_for_setattr(self) -> object | None:
        # [FUNCTION DOC] Returns the wrapped function as the value used for setattr.
        return self.fn

    def __init__(
        self,
        fn: types.FunctionType | torch.jit.ScriptFunction,  # type: ignore[type-arg]
        is_constant: bool = False,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores fn (after unwrapping an inline-target attribute set
        # by optimize wrappers). Sets is_constant from the function's constant
        # marker attribute (the is_constant argument itself is ignored). Triggers
        # a graph break if fn is not a Python function or script function.
        # KNOWN ISSUE: the is_constant parameter is never used; only the marker
        # attribute on fn decides the flag.
        super().__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            self.is_constant = False

        # TODO putting this here to avoid duplication, because we could hit this
        # from several paths (e.g., SuperVariable or `tp_getattro_impl`s).
        if not isinstance(fn, (types.FunctionType, torch.jit.ScriptFunction)):
            unimplemented(
                gb_type="can't handle functions not implemented in python ",
                context=f"{fn}",
                explanation="Dynamo can only handle functions defined in python",
                hints=[
                    "Move usage of this function out of `torch.compile` region",
                    *graph_break_hints.INFERENCE_MODE,
                ],
            )
        # TODO(anijain2305) - Replace directly calling UserFunctionVariable with
        # VariableBuilder, which handles the wrapping of _torchdynamo_inline.
        # unpack @torch._dynamo.optimize()(fn) wrapped function
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        self.fn = fn

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns the function itself when this is exactly a plain
        # UserFunctionVariable; subclasses defer to the parent implementation.
        if istype(self, UserFunctionVariable):
            return self.fn
        # subclasses (such as methods) usually aren't a constant
        return super().as_python_constant()

    def reconstruct_pycode(self, codegen):
        # [FUNCTION DOC] Emits Python source reconstruction via this tracker's
        # source; raises NotImplementedError when there is no source.
        if self.source:
            return self.source.reconstruct_pycode(codegen)
        raise NotImplementedError(
            "Python codegen not implemented for sourceless UserFunctionVariable"
        )

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Returns the real function for exact instances; subclasses
        # defer to the parent.
        if istype(self, UserFunctionVariable):
            return self.fn
        return super().get_real_python_backed_value()

    def self_args(self) -> list[VariableTracker]:
        # [FUNCTION DOC] No implicit self arguments for plain functions.
        return []

    def get_function(self) -> types.FunctionType:
        # [FUNCTION DOC] Returns the wrapped function.
        return self.fn

    def get_code(self) -> types.CodeType:
        # [FUNCTION DOC] Returns the wrapped function's code object.
        return self.fn.__code__

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the Python type as the function type.
        return types.FunctionType

    def has_self(self) -> bool:
        # [FUNCTION DOC] True if the wrapped callable has a non-None bound self.
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the wrapped function's globals mapping.
        return self.fn.__globals__

    def should_allow_nested_graph_breaks(self) -> bool:
        # [FUNCTION DOC] Returns False when the file lives under a builtin
        # inline-when-called directory or is registered as suppressing nested graph
        # breaks; otherwise True.
        from torch._dynamo.trace_rules import (
            BUILTIN_INLINE_WHEN_CALLED,
            is_ngb_suppressed_inline,
        )

        filename = self.get_filename()
        if any(filename.startswith(d) for d in BUILTIN_INLINE_WHEN_CALLED):
            return False
        if is_ngb_suppressed_inline(filename):
            return False
        return True

    def get_source(self) -> Source:
        # [FUNCTION DOC] Returns the source of the underlying function; for method
        # trackers this is the function source (source_fn) rather than the bound
        # method source.
        source = self.source

        if source and isinstance(self, variables.UserMethodVariable):
            source = self.source_fn  # type: ignore[assignment]
        return source  # type: ignore[return-value]

    def bind_args(
        self,
        parent: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> dict[str, VariableTracker]:
        # [FUNCTION DOC] Builds the initial locals mapping for inlining a call:
        # binds args/kwargs via the cached binder (using the root translator),
        # creates cells for cell variables, and maps each free variable name to a
        # tracked cell (existing side-effect cell, or a new tracked cell whose
        # contents are built from the closure, with a deleted marker if the cell
        # is empty). Raises an error for constant functions, non-plain functions,
        # or inconsistent closure length. Side effects: registers cells in the
        # parent's side effects.
        """
        Assume `args` and `kwargs` are VariableTracker arguments for a call to
        this function, create new bindings for initial locals.
        """
        if self.is_constant:
            raise AssertionError(
                "bind_args should not be called on a constant function"
            )

        fn: types.FunctionType = self.fn

        if not isinstance(fn, FunctionType):
            raise TypeError("Only supports regular Python functions.")
        root_tx = parent.output.root_tx

        source = self.get_source()
        result = bind_args_cached(fn, root_tx, source, args, kwargs)  # type: ignore[arg-type]

        init_cellvars(parent, result, fn.__code__)
        closure = self.fn.__closure__ or ()
        if len(closure) != len(self.fn.__code__.co_freevars):
            raise AssertionError(
                f"closure length {len(closure)} does not match "
                f"co_freevars length {len(self.fn.__code__.co_freevars)}"
            )
        for idx, name, cell in zip(
            itertools.count(), self.fn.__code__.co_freevars, closure
        ):
            # TODO refactor these 3 branches.
            side_effects = parent.output.side_effects
            if cell in side_effects:
                cell_var = side_effects[cell]

            elif source:
                closure_cell = GetItemSource(ClosureSource(source), idx)
                closure_cell_contents = CellContentsSource(
                    closure_cell, "cell_contents", freevar_name=name
                )
                try:
                    contents_var = VariableTracker.build(
                        parent, cell.cell_contents, closure_cell_contents
                    )
                except ValueError:
                    # Cell has not yet been assigned
                    contents_var = variables.DeletedVariable()
                cell_var = side_effects.track_cell_existing(
                    closure_cell, cell, contents_var
                )

            else:
                # TODO figure out why source isn't available here, and whether
                # we can fix that and remove this branch.
                try:
                    contents_var = VariableTracker.build(parent, cell.cell_contents)
                except ValueError:
                    # Cell has not yet been assigned
                    contents_var = variables.DeletedVariable()
                cell_var = side_effects.track_cell_existing(None, cell, contents_var)

            result[name] = cell_var

        return result

    # func.__get__ binds the function to an instance via the tp_descr_get slot
    # wrapper. https://github.com/python/cpython/blob/v3.13.0/Objects/funcobject.c#L1119
    def _get_dunder_get(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the descriptor get method of the
        # function, built as a tracker with an attribute source when available.
        source = self.get_source()
        source = AttrSource(source, "__get__") if source is not None else None
        return VariableTracker.build(tx, self.fn.__get__, source)

    def _fn_getattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Reads attribute name from the real function with
        # source tracking (delegates to fn_getattro_impl).
        return fn_getattro_impl(tx, self.fn, self.get_source(), name)

    # A real function object backs this VT, so resolve the function slots by
    # reflecting on it (fn_getattro_impl) rather than via the synthesized
    # get_*-based tables inherited from BaseUserFunctionVariable (which assume
    # synthesized fields like self.defaults). These win over the base entries
    # under MRO-merge resolution. __dict__ inherits the base getset (get_dict_vt).
    tp_getset = {
        "__get__": GetSet(_get_dunder_get, None),
        "__name__": GetSet(lambda s, tx: s._fn_getattr(tx, "__name__")),
        "__qualname__": GetSet(lambda s, tx: s._fn_getattr(tx, "__qualname__")),
        "__code__": GetSet(lambda s, tx: s._fn_getattr(tx, "__code__")),
        "__defaults__": GetSet(lambda s, tx: s._fn_getattr(tx, "__defaults__")),
        "__kwdefaults__": GetSet(lambda s, tx: s._fn_getattr(tx, "__kwdefaults__")),
        "__annotations__": GetSet(lambda s, tx: s._fn_getattr(tx, "__annotations__")),
        "__type_params__": GetSet(lambda s, tx: s._fn_getattr(tx, "__type_params__")),
    }
    tp_members = {
        "__doc__": Member(lambda s, tx: s._fn_getattr(tx, "__doc__")),
        "__module__": Member(lambda s, tx: s._fn_getattr(tx, "__module__")),
        "__closure__": Member(lambda s, tx: s._fn_getattr(tx, "__closure__")),
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor get: binds this function to the instance obj,
        # returning a method tracker whose source is an attribute of obj's source
        # (if any) and whose function source is this tracker's source.
        # Mirrors func_descr_get which calls PyMethod_New to bind
        # the function to an instance.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1119
        source = obj.source and AttrSource(obj.source, self.fn.__name__)
        return UserMethodVariable(self.fn, obj, source_fn=self.source, source=source)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the function. Special-cases Dynamo control APIs:
        # patch_dynamo_config (constant args -> config patch tracker),
        # error_on_graph_break, override_cudagraphs, the bytecode debugger
        # breakpoint (sets a flag on the output), and nonstrict_trace (returns an
        # in-graph function tracker; graph-breaks if not a module-level user
        # function). Raises RuntimeError on malformed arguments of those APIs.
        # Constant functions are invoked and stored as constants. Temporarily
        # relaxes side-effect checks for the safety-check disabling helper and
        # for FSDP pre/post forward inside checkpoint tracing. Tries the tree_map
        # fast path, else inlines via the parent.
        # Handle patch_dynamo_config call
        if self.fn is torch._dynamo.patch_dynamo_config:
            try:
                args_const = [arg.as_python_constant() for arg in args]
                kwargs_const = {
                    key: val.as_python_constant() for key, val in kwargs.items()
                }
                changes = torch._dynamo.patch_dynamo_config(
                    *args_const, **kwargs_const
                ).changes
                return variables.DynamoConfigPatchVariable(changes)
            except AsPythonConstantNotImplementedError as e:
                raise RuntimeError(
                    "Cannot convert patch_dynamo_config args/kwargs to constants. "
                    "Please fix your call to patch_dynamo_config by using simpler inputs. "
                    f"args: {args}, kwargs: {kwargs}"
                ) from e
        elif self.fn is torch._dynamo.error_on_graph_break:
            try:
                bound = inspect.signature(self.fn).bind(*args, **kwargs)
                error_on_graph_break = bound.arguments[
                    "error_on_graph_break"
                ].as_python_constant()
                if not isinstance(error_on_graph_break, bool):
                    raise AssertionError(
                        f"error_on_graph_break must be a bool, got {type(error_on_graph_break)}"
                    )
                return variables.ErrorOnGraphBreakVariable(error_on_graph_break)
            except Exception as e:
                raise RuntimeError(
                    "Improper error_on_graph_break() call. Please fix your call to error_on_graph_break(). "
                    f"args: {args}, kwargs: {kwargs}"
                ) from e
        elif self.fn is torch._dynamo.override_cudagraphs:
            try:
                bound = inspect.signature(self.fn).bind(*args, **kwargs)
                bound.apply_defaults()
                fwd = bound.arguments["fwd"]
                bwd = bound.arguments["bwd"]
                if isinstance(fwd, VariableTracker):
                    fwd = fwd.as_python_constant()
                if isinstance(bwd, VariableTracker):
                    bwd = bwd.as_python_constant()
                return variables.CudagraphOverrideVariable(fwd, bwd)
            except Exception as e:
                raise RuntimeError(
                    "Improper override_cudagraphs() call. Please fix your call to override_cudagraphs(). "
                    f"args: {args}, kwargs: {kwargs}"
                ) from e
        elif self.fn is torch._dynamo.bytecode_debugger.breakpoint:
            tx.output._emit_debugger_breakpoint = True
            return variables.ConstantVariable.create(None)
        # Handle a `nonstrict_trace(fn)` call
        elif self.fn is torch._dynamo.nonstrict_trace:
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            fn_var = bound.args[0]
            if not isinstance(fn_var, BaseUserFunctionVariable):
                typ = fn_var.python_type()
                msg = f"`nonstrict_trace` expects a callable, but got value of type <{typ.__name__}>"
                unimplemented(
                    gb_type="TypeError from user code",
                    context=f"call_function({self.value}, {args}, {kwargs})",  # type: ignore[attr-defined]
                    explanation=msg,
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

            if not isinstance(fn_var, UserFunctionVariable):
                fn_name = fn_var.get_name()
                msg = f"Applying `nonstrict_trace` to function <{fn_name}>; however, `nonstrict_trace` currently requires the function to be defined outside `torch.compile` region."
                unimplemented(
                    gb_type="Limitation of `nonstrict_trace",
                    context=f"{self}",
                    explanation=msg,
                    hints=[
                        f"make sure definition of {fn_name} is outside ",
                        "`torch.compile` region",
                    ],
                )

            fn = fn_var.fn
            return variables.TorchInGraphFunctionVariable(
                fn, kind=variables.torch.AllowInGraphKind.NONSTRICT_TRACE
            )

        if self.is_constant:
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), args, kwargs
            )

        if (
            not tx.output.current_tracer.unsafe_allow_externally_visible_side_effects
            and self.fn
            is torch._dynamo.utils._disable_side_effect_safety_checks_for_current_subtracer
        ):
            with torch._dynamo.side_effects.allow_externally_visible_side_effects_in_subtracer(
                tx
            ):
                return super().call_function(tx, args, kwargs)

        if (
            getattr(tx.output.current_tracer, "description", None)
            == "torch.utils.checkpoint.checkpoint"
            and not tx.output.current_tracer.allow_side_effects_in_hop
        ):
            try:
                from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
            except Exception:
                FSDPState = None  # type: ignore[assignment, misc]
            if FSDPState is not None and self.fn in [
                FSDPState._pre_forward,
                FSDPState._post_forward,
            ]:
                with torch._dynamo.side_effects.allow_side_effects_in_hop(tx):
                    return super().call_function(tx, args, kwargs)

        tree_map_result = self._maybe_call_tree_map_fastpath(tx, args, kwargs)
        if tree_map_result is not None:
            return tree_map_result

        return super().call_function(tx, args, kwargs)

    def _maybe_call_tree_map_fastpath(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        # [FUNCTION DOC] HELPER: If this call is a tree_map or tree_map_with_path
        # (possibly after rewriting a tree_map_only call) with supported keyword
        # arguments and at least a function and a tree, dispatches to the first
        # tree tracker's tree-map routine with nested graph breaks disabled.
        # Returns None when the fast path does not apply.
        rewrite = self._rewrite_tree_map_only_call(tx, args, kwargs)
        if rewrite is not None:
            tree_map_fn, tree_map_args, tree_map_kwargs = rewrite
        else:
            tree_map_fn = self
            tree_map_args = args
            tree_map_kwargs = kwargs

        is_tree_map = (
            isinstance(tree_map_fn, UserFunctionVariable)
            and tree_map_fn._is_tree_map_function()
        )
        is_tree_map_with_path = (
            isinstance(tree_map_fn, UserFunctionVariable)
            and tree_map_fn._is_tree_map_with_path_function()
        )

        if not (is_tree_map or is_tree_map_with_path):
            return None
        if {*tree_map_kwargs} - _SUPPORTED_TREE_MAP_KWARGS:
            return None
        if len(tree_map_args) < 2:
            return None

        map_fn = tree_map_args[0]
        first_tree = tree_map_args[1]
        rest = tree_map_args[2:]

        # The tree_map fast path doesn't create an InliningIT for tree_map,
        # so NGB resume functions would miss the tree_map continuation.
        with torch._dynamo.disable_nested_graph_breaks():
            if is_tree_map_with_path:
                return first_tree.call_tree_map_with_path(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                    keypath=(),
                )
            else:
                return first_tree.call_tree_map(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                )

    def _is_tree_map_function(self) -> bool:
        # [FUNCTION DOC] HELPER: True if fn is named tree_map in a known pytree module.
        return (
            getattr(self.fn, "__name__", None) == "tree_map"
            and getattr(self.fn, "__module__", None) in self._TREE_MAP_MODULES
        )

    def _is_tree_map_with_path_function(self) -> bool:
        # [FUNCTION DOC] HELPER: True if fn is tree_map_with_path in a known pytree module.
        return (
            getattr(self.fn, "__name__", None) == "tree_map_with_path"
            and getattr(self.fn, "__module__", None) in self._TREE_MAP_MODULES
        )

    def _is_tree_map_only_function(self) -> bool:
        # [FUNCTION DOC] HELPER: True if fn is tree_map_only in a known pytree module.
        return (
            getattr(self.fn, "__name__", None) == "tree_map_only"
            and getattr(self.fn, "__module__", None) in self._TREE_MAP_MODULES
        )

    def _rewrite_tree_map_only_call(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> (
        tuple[
            "UserFunctionVariable",
            list[VariableTracker],
            dict[str, VariableTracker],
        ]
        | None
    ):
        # [FUNCTION DOC] HELPER: Rewrites a tree_map_only(type_selector, fn, tree)
        # call into a tree_map call whose function applies fn only to leaves of
        # the selected types. Returns (tree_map tracker, new args, new kwargs) or
        # None if this is not tree_map_only, has the wrong arity, has unsupported
        # keywords, a non-constant/invalid type selector, or no tree_map found.
        if not self._is_tree_map_only_function():
            return None

        if len(args) != 3:
            return None
        if {*kwargs} - _TREE_MAP_ONLY_SUPPORTED_KWARGS:
            return None

        type_selector, map_fn, tree_arg = args
        allowed_types = self._extract_tree_map_only_types(type_selector)
        if allowed_types is None:
            return None

        tree_map_callable = self._lookup_tree_map_function()
        if tree_map_callable is None:
            return None

        wrapped_map_fn = TreeMapOnlyFunctionVariable(
            allowed_types,
            map_fn,
            source=getattr(map_fn, "source", None),
        )
        tree_map_variable = VariableTracker.build(tx, tree_map_callable)
        return tree_map_variable, [wrapped_map_fn, tree_arg], dict(kwargs)

    def _lookup_tree_map_function(self) -> types.FunctionType | None:
        # [FUNCTION DOC] HELPER: Finds the tree_map function in the same module as
        # fn; returns None if the module or a plain-function tree_map is missing.
        module_name = getattr(self.fn, "__module__", None)
        if not module_name:
            return None
        module = sys.modules.get(module_name)
        if module is None:
            return None
        tree_map = getattr(module, "tree_map", None)
        if isinstance(tree_map, types.FunctionType):
            return tree_map
        return None

    def _extract_tree_map_only_types(
        self, selector: VariableTracker
    ) -> tuple[type, ...] | None:
        # [FUNCTION DOC] HELPER: Converts a constant type selector tracker (type,
        # tuple of types or union) to a de-duplicated tuple of types; returns None
        # if non-constant, empty, or containing non-types.
        if not selector.is_python_constant():
            return None
        try:
            raw_value = selector.as_python_constant()
        except NotImplementedError:
            return None

        flattened = self._flatten_type_spec(raw_value)
        if not flattened:
            return None
        if not all(isinstance(typ, type) for typ in flattened):
            return None
        return tuple(dict.fromkeys(flattened))

    def _flatten_type_spec(self, value: Any) -> list[type] | None:
        # [FUNCTION DOC] HELPER: Recursively flattens a type, tuple of types, or
        # union type into a list of types; returns None if any member is
        # unsupported.
        if isinstance(value, type):
            return [value]
        if isinstance(value, tuple):
            collected: list[type] = []
            for entry in value:
                flat = self._flatten_type_spec(entry)
                if flat is None:
                    return None
                collected.extend(flat)
            return collected
        union_type = getattr(types, "UnionType", None)
        if union_type is not None and isinstance(value, union_type):
            collected = []
            for entry in typing.get_args(value):
                flat = self._flatten_type_spec(entry)
                if flat is None:
                    return None
                collected.extend(flat)
            return collected
        return None


class InspectSignatureVariable(UserFunctionVariable):
    # [CLASS DOC] Tracker for the signature-inspection function that memoizes
    # results per target function in the output graph's signature cache.
    """
    Variable tracker for inspect.signature with caching support.

    inspect.Signature is expensive to trace. When inspect.signature is called
    repeatedly on the same function during tracing, we cache the result to avoid
    retracing the signature construction each time. Although this is different
    from CPython behavior, it is safe to do so because inspect.signature does
    not change across different calls to the same function.
    """

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] With exactly one positional argument that is a user
        # function or method tracker, returns a cached result keyed by the real
        # function (tracing and storing it on a miss). Otherwise traces normally.
        # Side effect: mutates the output graph's signature cache.
        # Fast path: cache results for repeated calls on the same function
        if len(args) == 1 and not kwargs:
            target_arg = args[0]
            cache_key = None

            if isinstance(target_arg, (UserFunctionVariable, UserMethodVariable)):
                cache_key = target_arg.get_function()

            if cache_key is not None:
                if cache_key in tx.output.signature_cache:
                    return tx.output.signature_cache[cache_key]

                result = super().call_function(tx, args, kwargs)
                tx.output.signature_cache[cache_key] = result
                return result

        return super().call_function(tx, args, kwargs)


class TreeMapOnlyFunctionVariable(BaseUserFunctionVariable):
    # [CLASS DOC] Synthetic function tracker wrapping a user map function so it
    # applies only to leaves whose type is in allowed_types (implementing
    # tree_map_only on top of tree_map); other leaves pass through unchanged.
    _nonvar_fields = {
        "allowed_types",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    def __init__(
        self,
        allowed_types: tuple[type, ...],
        map_fn: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the tuple of allowed leaf types and the tracker of
        # the function to apply to matching leaves.
        super().__init__(**kwargs)
        self.allowed_types = allowed_types
        self.map_fn = map_fn

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the function type.
        return FunctionType

    def _matches_allowed_type(self, node: VariableTracker) -> bool:
        # [FUNCTION DOC] HELPER: True if node's Python type is a subclass of any
        # allowed type; False if its type is unknown.
        try:
            node_type = node.python_type()
        except NotImplementedError:
            return False
        return any(issubclass(node_type, allowed) for allowed in self.allowed_types)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Applies map_fn when called without args, when the first
        # argument matches an allowed type, or when extra args/kwargs are present
        # (fall back to normal tracing); otherwise returns the single leaf
        # unchanged.
        if not args:
            return self.map_fn.call_function(tx, args, kwargs)
        leaf = args[0]
        if self._matches_allowed_type(leaf):
            return self.map_fn.call_function(tx, args, kwargs)
        if len(args) != 1 or kwargs:
            # Defer to the original map function so we fall back to normal
            # tracing instead of triggering a graph break.
            return self.map_fn.call_function(tx, args, kwargs)
        return leaf


class LocalGeneratorObjectVariable(VariableTracker):
    # [CLASS DOC] Tracker for a generator object created from a generator function
    # defined in traced code. Owns an inlining tracer whose frame state models the
    # generator state machine (created, executing, suspended, suspended-yield-from,
    # completed, cleared). Implements next/send/throw/close following CPython
    # semantics; remaining_items caches items when reconstructing as a list
    # iterator. Registers itself with the output graph on construction.
    # PyGen_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/genobject.c#L814
    _cpython_type = types.GeneratorType

    def __init__(
        self,
        code: types.CodeType,
        f_globals: dict[str, Any],
        inline_tracer: "InliningGeneratorInstructionTranslator",
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the code, globals and inline tracer, initializes an
        # empty remaining-items list, and registers this generator with the
        # tracer's output graph for tracking.
        super().__init__(**kwargs)
        self.code = code
        self.f_globals = f_globals
        self.inline_tracer = inline_tracer
        self.remaining_items: list[VariableTracker] = []
        inline_tracer.output.track_generator(self)

    def get_code(self) -> types.CodeType:
        # [FUNCTION DOC] Returns the generator's code object.
        return self.code

    def get_filename(self) -> str:
        # [FUNCTION DOC] Returns the code's file name.
        return self.get_code().co_filename

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the code's function name.
        return self.get_code().co_name

    def get_function(self) -> Never:
        # [FUNCTION DOC] Not supported: always raises NotImplementedError.
        raise NotImplementedError("get_function")

    def has_self(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def __name__(self) -> str:
        # [FUNCTION DOC] Returns the generator function name.
        # KNOWN ISSUE: defined as a method (not a property), so the attribute is a
        # bound method rather than a string.
        return self.get_name()

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the class name with the generator function name.
        return f"{self.__class__.__name__}({self.get_name()})"

    __repr__ = __str__

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Reconstructs the generator as a list iterator over its
        # remaining items. If the generator is not finished, it is exhausted
        # eagerly (unpacked) under a saved/restarted speculation log, with side
        # effects in the generator disallowed and output-graph writes temporarily
        # allowed. Side effect: fills remaining_items and advances the generator.
        from torch._dynamo.side_effects import disallow_side_effects_in_generator
        from torch._dynamo.symbolic_convert import (
            save_and_restart_speculation_log,
            temporarely_allow_writes_to_output_graph,
        )

        tx = codegen.tx
        save = save_and_restart_speculation_log(tx)
        disallow = disallow_side_effects_in_generator(tx)
        temp = temporarely_allow_writes_to_output_graph(tx)

        with save, disallow, temp:
            if not self._frame_state_finished():
                self.remaining_items = unpack_iterable(tx, self)  # type: ignore[bad-argument-type]
            variables.ListIteratorVariable(self.remaining_items).reconstruct(codegen)

    def get_globals(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the generator's globals mapping.
        return self.f_globals

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the generator type.
        return types.GeneratorType

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Identity-based comparison; ordering comparisons raise.
        # Generators have no tp_richcompare: identity for ==/!=, TypeError for
        # ordering.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def pygen_yf(self) -> VariableTracker | None:
        # [FUNCTION DOC] Returns the sub-iterator being delegated to (yield from)
        # when suspended in a yield-from, read from the tracer stack at a version-
        # dependent depth; otherwise None.
        if self.inline_tracer.frame_state == FrameState.FRAME_SUSPENDED_YIELD_FROM:
            if sys.version_info >= (3, 15):
                ind = -2
            else:
                ind = -1
            return self.inline_tracer.stack[ind]
        return None

    def gen_send_ex2(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        exc: bool,
    ) -> VariableTracker:
        # [FUNCTION DOC] Resumes the generator with arg (or throws the pending
        # exception when exc is true) and returns the yielded or returned value.
        # Raises observed TypeError (non-None send to a fresh generator), observed
        # ValueError (already executing) or observed StopIteration (finished).
        # Manages the tracer frame state; on exceptions marks the frame cleared or
        # completed; on pre-3.12 converts an escaping StopIteration to RuntimeError.
        # An infinite-generator error becomes a graph break; an Unsupported error
        # marks the code as skipped and the frame to be skipped, then re-raises.
        # https://github.com/python/cpython/blob/f31a89bb901067dd105b00cfa90523cf7ffdbbdd/Objects/genobject.c#L259
        tracer = self.inline_tracer

        if (
            tracer.frame_state == FrameState.FRAME_CREATED
            and not arg.is_constant_none()
        ):
            raise_type_error(
                tx,
                "can't send non-None value to a just-started generator",
            )

        if tracer.frame_state == FrameState.FRAME_EXECUTING:
            raise_value_error(tx, "generator already executing")

        if self._frame_state_finished():
            raise_observed_exception(StopIteration, tx)

        tracer.frame_state = FrameState.FRAME_EXECUTING

        try:
            # Hierarchically, tx can be seen as the parent of the inline tracer
            # created on call_function. Any exception needs to be propagated to tx
            # for Dynamo to behave correctly
            tracer.push(arg)
            with self.inline_tracer.link_gi_exc_state():
                if exc:
                    self.throw_pending()
                return tracer.inline_call_()
        except ObservedUserStopIteration:
            # PEP 479: pre-3.12 has no STOPITERATION_ERROR opcode, so convert a
            # StopIteration that escapes the generator body to RuntimeError here
            # at the frame boundary. https://github.com/python/cpython/pull/99006
            # A normal return sets FRAME_CLEARED and raises a synthetic
            # StopIteration to signal exhaustion; that one must stay a
            # StopIteration, so only convert when the body was still executing.
            was_executing = tracer.frame_state == FrameState.FRAME_EXECUTING
            tracer.frame_state = FrameState.FRAME_COMPLETED
            if sys.version_info < (3, 12) and was_executing:
                # Match CPython's _PyErr_FormatFromCause: set __context__ and
                # __cause__ directly rather than pushing onto the exception
                # stack (which must stay balanced -- genobject.c pops the
                # generator's gi_exc_state before the conversion runs, so the
                # caller's stack is left unchanged). do_raise sets __cause__;
                # set __context__ first so set_exception_obj's implicit chaining
                # leaves it untouched.
                prev = tracer.exn_vt_stack.get_raised_exception()
                rt = VariableTracker.build(tx, RuntimeError).call_function(
                    tx,
                    [VariableTracker.build(tx, "generator raised StopIteration")],
                    {},
                )
                rt.call_method(
                    tx,
                    "__setattr__",
                    [ConstantVariable.create("__context__"), prev],
                    {},
                )
                tx.do_raise(rt, prev)
            raise
        except ObservedException:
            # An exception propagating out of the generator frame finishes it,
            # mirroring CPython setting gi_frame_state = FRAME_CLEARED.
            tracer.frame_state = FrameState.FRAME_CLEARED
            raise
        except InfiniteGeneratorError:
            # test/dynamo/test_misc.py::test_iterator_limit
            unimplemented(
                gb_type="infinite generator detected",
                context="",
                explanation="Dynamo traced the YIELD_VALUE bytecode too many times. This could mean "
                "that we have attempted to trace an infinite generator.",
                hints=[
                    f"If you are sure that your generator is not infinite, please report a bug at {PT2_ISSUE_TRACKER_URL}.",
                    *graph_break_hints.USER_ERROR,
                ],
            )
        except Unsupported as e:
            torch._dynamo.eval_frame.skip_code(self.get_code())
            e.skip_frame = True
            if not tx.one_graph and not tx.error_on_graph_break:
                e.msg += "\n\nSkipping frame due to graph break in a generator's next() call."
            raise

    def gen_send_ex(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        exc: bool,
    ) -> VariableTracker:
        # [FUNCTION DOC] Like gen_send_ex2 but converts a return (frame not
        # suspended) into an observed StopIteration carrying the return value
        # (no args when None). Returns the yielded value when suspended.
        # rule of thumb for gen_send_ex2:
        # - PYGEN_RETURN => No exception raised
        # - PYGEN_ERROR => Exception raised
        # - PYGEN_NEXT => yielded - frame suspended
        # gen_send_ex raises StopIteration if gen_send_ex2 returns PYGEN_RETURN
        result = self.gen_send_ex2(tx, arg, exc)
        if self._frame_state_suspended():
            # PYGEN_NEXT
            return result
        else:
            # PYGEN_RETURN
            if result.is_constant_none():
                raise_observed_exception(StopIteration, tx)
            else:
                raise_observed_exception(StopIteration, tx, args=[result])

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Next item: resumes the generator by sending None and
        # returns the next yielded value (raises on exhaustion via gen_send_ex2).
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/genobject.c#L832
        return self.gen_send_ex2(tx, ConstantVariable.create(None), False)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # [FUNCTION DOC] hasattr answered statically: True iff the generator type
        # defines name in its class dict.
        if name in self.python_type().__dict__:
            return ConstantVariable.create(True)
        return ConstantVariable.create(False)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Iterating a generator returns the generator itself.
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/genobject.c#L831
        return self

    # no nested graph breaks in generators
    def should_allow_nested_graph_breaks(self) -> Literal[False]:
        # [FUNCTION DOC] Always False: nested graph breaks are not allowed in generators.
        return False

    def _setup_exception(
        self, tx: "InstructionTranslatorBase", exc: VariableTracker
    ) -> None:
        # [FUNCTION DOC] HELPER: Instantiates exc (an exception type or instance
        # tracker) and records it as the raised exception on the generator's
        # tracer so it will be thrown at resume. Raises TypeError for other kinds.
        # Set up the exception to be raised in the generator frame
        from torch._dynamo.symbolic_convert import ExceptionTypes, ExceptionVals

        # Instantiate if an exception type was passed (builtin or user-defined).
        if not isinstance(exc, (ExceptionTypes, ExceptionVals)):
            raise TypeError(
                f"Expected an exception type or instance, got {exc.python_type_name()}"
            )
        val = tx._create_exception_instance(exc)
        if not isinstance(val, ExceptionVals):
            raise AssertionError(f"Expected an exception variable, got {val}")
        self.inline_tracer.exn_vt_stack.set_raised_exception(val)

    def _frame_state_created(self) -> bool:
        # [FUNCTION DOC] HELPER: True if the generator has not started.
        return self.inline_tracer.frame_state == FrameState.FRAME_CREATED

    def _frame_state_finished(self) -> bool:
        # [FUNCTION DOC] HELPER: True if the generator is completed or cleared.
        return self.inline_tracer.frame_state in {
            FrameState.FRAME_COMPLETED,
            FrameState.FRAME_CLEARED,
        }

    def _frame_state_suspended(self) -> bool:
        # [FUNCTION DOC] HELPER: True if suspended at a yield or yield-from.
        return self.inline_tracer.frame_state in {
            FrameState.FRAME_SUSPENDED,
            FrameState.FRAME_SUSPENDED_YIELD_FROM,
        }

    def gen_send(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method send: resumes with the first argument and returns
        # the next yielded value; raises observed StopIteration at exhaustion.
        # Sends a value into the generator function. Returns the next value
        # yielded by the generator, or raises StopIteration if the generator
        # exits without yielding another value
        return self.gen_send_ex(tx, args[0], False)

    def gen_close(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method close: returns None for unstarted (marks completed)
        # or finished generators. Otherwise closes any delegated sub-iterator, then
        # throws GeneratorExit into the generator. Returns None if it exits or
        # raises GeneratorExit (traceback dropped to avoid reference cycles),
        # returns the generator's return value if it returns one, raises observed
        # RuntimeError if it yields, and propagates other exceptions.
        # * Raises a GeneratorExit at the point where the generator function was paused.
        # * If the generator function catches the exception and returns a
        # value, this value is returned from close() - Python 3.13+
        # * If the generator function is already closed, or raises GeneratorExit
        # (by not catching the exception), close() returns None.
        # * If the generator yields a value, a RuntimeError is raised.
        # * If the generator raises any other exception, it is propagated to the caller.
        # * If the generator has already exited due to an exception or normal
        # exit, close() returns None and has no other effect.

        # Return None if close is called on a just-started generator
        # See test GeneratorCloseCpythonTests::test_close_not_started

        tracer = self.inline_tracer
        if self._frame_state_created():
            tracer.frame_state = FrameState.FRAME_COMPLETED
            return ConstantVariable.create(None)

        if self._frame_state_finished():
            return ConstantVariable.create(None)

        err = False
        yf = self.pygen_yf()
        if yf:
            with tracer.temporarily_set_frame_state(FrameState.FRAME_EXECUTING):
                try:
                    yf.call_method(tx, "close", [], {})
                except ObservedException:
                    err = True

        if err is False:
            self._setup_exception(tx, VariableTracker.build(tx, GeneratorExit))

        try:
            self.gen_send_ex(tx, ConstantVariable.create(None), True)
        except ObservedGeneratorExit as e:
            # Drop the traceback to break the exception -> traceback -> frame ->
            # (raised_exception, self) reference cycle. Otherwise the generator's
            # inline_tracer (and the OutputGraph it points at) survives until
            # cyclic GC instead of being freed by refcount at frame exit.
            e.__traceback__ = None
            return ConstantVariable.create(None)
        except ObservedUserStopIteration:
            # generator returned a value while closing. gen_send_ex() raises
            # StopIteration with the value returned
            curr_exc = tracer.exn_vt_stack.get_raised_exception()
            if not isinstance(curr_exc, variables.ExceptionVariable):
                # make pyrefly happy
                raise AssertionError(
                    f"Expected current exception to be an ExceptionVariable, got {curr_exc}"
                ) from None
            if curr_exc.args:
                return curr_exc.args[0]
        else:
            # if it reaches here, the generator yielded a value while closing
            raise_observed_exception(
                RuntimeError,
                tx,
                args=["generator ignored GeneratorExit"],
            )
        return ConstantVariable.create(None)

    def throw_pending(self) -> None:
        # [FUNCTION DOC] Re-raises the exception currently recorded on the tracer
        # inside the generator frame: sets the exception object and runs the
        # tracer's exception handler. Side effect: mutates tracer exception state.
        tracer = self.inline_tracer
        curr_exc = tracer.exn_vt_stack.get_raised_exception()
        observed = get_dynamo_observed_exception(curr_exc.python_type())()
        curr_type = VariableTracker.build(tracer, curr_exc.exc_type)
        tracer.set_exception_obj(curr_type, curr_exc)
        tracer.exception_handler(observed)

    def gen_throw(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method throw: raises the given exception (last argument
        # form accepted) at the suspension point and returns the next yielded value.
        # If delegating via yield-from: for GeneratorExit the sub-iterator is closed
        # first and the exception then thrown here; for other exceptions the throw
        # is forwarded to the sub-iterator, falling back to throwing into this
        # generator if that raises.
        # * Raises an exception at the point where the generator was paused, and
        # returns the next value yielded by the generator.
        # * If the generator exits without yielding, raise StopIteration
        # * If the generator function does not catch the passed-in exception,
        # or raises a different exception, then that exception propagates to the caller.
        def throw_here():
            # [FUNCTION DOC] HELPER: Records the exception and resumes the generator
            # with it, returning the next yielded value.
            self._setup_exception(tx, arg)
            return self.gen_send_ex(tx, ConstantVariable.create(None), True)

        from torch._dynamo.symbolic_convert import pyerr_given_exception_match

        arg = args[1] if len(args) > 1 else args[0]
        yf = self.pygen_yf()
        tracer = self.inline_tracer

        if yf:
            # CPython has an extra flag for handling async generators
            if pyerr_given_exception_match(arg, GeneratorExit):
                try:
                    # CPython uses gen_close_iter here
                    with tracer.temporarily_set_frame_state(FrameState.FRAME_EXECUTING):
                        yf.call_method(tx, "close", [], {})
                except ObservedException:
                    pass
                return throw_here()

            try:
                with tracer.temporarily_set_frame_state(FrameState.FRAME_EXECUTING):
                    return yf.call_method(tx, "throw", [arg], {})
            except ObservedException:
                return self.gen_send_ex(tx, ConstantVariable.create(None), True)

        return throw_here()

    tp_methods = {
        "send": Method(gen_send),
        "close": Method(gen_close),
        "throw": Method(gen_throw),
    }


class ContextlibContextManagerLocalGeneratorObjectVariable(
    LocalGeneratorObjectVariable
):
    # [CLASS DOC] Marker subclass of the generator tracker for generators created
    # by functions decorated with the contextmanager decorator; no added behavior.
    """
    .. note::

        This is only used when the function is annotated with @contextlib.contextmanager

        It is a special case of a generator function as we do not allow return a context manager
        from a torch.compile function.
    """


class LocalGeneratorFunctionVariable(BaseUserFunctionVariable):
    # [CLASS DOC] Wrapper around a user or nested function tracker (vt) whose code
    # is a generator function. Calling it builds an inline tracer, runs it to the
    # first suspension (3.11+), and returns a generator object tracker of class
    # generator_cls. Unknown attributes are forwarded to the wrapped tracker.
    """functions that behaves like iterators

    .. note::

        This is a wrapper around (Nested)UserFunctionVariable
    """

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the function type.
        return types.FunctionType

    def __init__(
        self,
        vt: BaseUserFunctionVariable,
        *,
        generator_cls: type = LocalGeneratorObjectVariable,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the wrapped function tracker and the generator
        # tracker class to instantiate on call.
        super().__init__(**kwargs)
        self.vt = vt
        self.generator_cls = generator_cls

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC] Attribute fallback: names defined on this class are read
        # from self; everything else is forwarded to the wrapped tracker.
        # KNOWN ISSUE: the class-dict branch re-reads the same attribute on self,
        # which would recurse if the name were not actually resolvable.
        if name in self.__class__.__dict__:
            return getattr(self, name)
        return getattr(self.vt, name)

    # These need to be explicit so the custom __getattr__ doesn't fall back to the unimplemented base class version
    def get_code(self) -> types.CodeType:
        # [FUNCTION DOC] Returns the wrapped function's code object.
        return self.vt.get_code()

    def get_globals(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the wrapped function's globals.
        return self.vt.get_globals()

    def has_self(self) -> bool:
        # [FUNCTION DOC] Delegates to the wrapped tracker.
        return self.vt.has_self()

    def _build_inline_tracer(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "InliningInstructionTranslator":
        # [FUNCTION DOC] HELPER: Builds an inlining translator for this function
        # with the given call arguments, as a child of tx.
        from torch._dynamo.symbolic_convert import InliningInstructionTranslator

        return InliningInstructionTranslator.build_inline_tracer(
            tx,
            self,
            args,
            kwargs,
        )

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the generator function: graph-breaks if the wrapped
        # function is skipped or its code is not a generator. Otherwise builds an
        # inline tracer, runs it to its first suspension on Python 3.11+, and
        # returns a new generator tracker (class generator_cls) holding the code,
        # globals, tracer and this tracker's source.
        if isinstance(self.vt, SkipFunctionVariable):
            unimplemented(
                gb_type="generator function over a skipped function",
                context=str(self.vt),
                explanation="Cannot trace a generator whose underlying function is skipped by Dynamo "
                "(e.g. defined in a skip-listed module), since its body cannot be symbolically traced.",
                hints=[*graph_break_hints.FUNDAMENTAL],
            )
        if not is_generator(self.vt.get_code()):
            unimplemented(
                gb_type="non-generator contextlib.contextmanager",
                context=str(self.vt.get_code()),
                explanation="Cannot compile function decorated with `@contextlib.contextmanager` that is not a generator"
                ", i.e. does not use `yield`",
                hints=[
                    "Use `yield` in the function body instead of `return`.",
                    "Remove the `@contextlib.contextmanager` decorator.",
                ],
            )

        inline_tracer = self._build_inline_tracer(tx, list(args), kwargs)
        code = self.vt.get_code()
        f_globals = self.vt.get_globals()

        if sys.version_info >= (3, 11):
            inline_tracer.inline_call_()
        # calling a generator returns a generator object
        return self.generator_cls(
            code,
            f_globals,
            inline_tracer,  # type: ignore[arg-type]
            source=self.source,
        )

    def get_real_python_backed_value(self) -> object:
        # [FUNCTION DOC] Delegates to the wrapped tracker.
        return self.vt.get_real_python_backed_value()


class FunctionDecoratedByContextlibContextManagerVariable(
    LocalGeneratorFunctionVariable
):
    # [CLASS DOC] Generator function tracker for functions decorated with the
    # contextmanager decorator; produces the contextmanager-specific generator
    # tracker class.
    """
    .. note::

        This is only used when the function is annotated with @contextlib.contextmanager
    """

    def __init__(self, vt: BaseUserFunctionVariable, **kwargs: Any) -> None:
        # [FUNCTION DOC] Wraps vt, fixing the generator class to the
        # contextmanager-specific one.
        super().__init__(
            vt,
            generator_cls=ContextlibContextManagerLocalGeneratorObjectVariable,
            **kwargs,
        )


class UserMethodVariable(UserFunctionVariable):
    # [CLASS DOC] Tracker for a function bound to an instance (obj). Attributes:
    # fn (the unbound function), obj (the bound self tracker), source_fn (source of
    # the underlying function, used for guards on binding). Calls prepend obj to
    # the arguments, with special routing for nonstrict-trace and leaf functions,
    # nn.Module methods, and FSDP training-state context managers.
    """Some unsupported user-defined method"""

    # PyMethod_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/classobject.c#L332
    _cpython_type = types.MethodType

    def __init__(
        self,
        fn: Callable[..., Any],
        obj: VariableTracker,
        source_fn: Source | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores fn, the bound object tracker and source_fn. When no
        # source_fn is given but a bound-method source exists, derives source_fn as
        # the function attribute of that source.
        super().__init__(fn=fn, **kwargs)  # type: ignore[arg-type]
        self.obj = obj
        self.source_fn = source_fn
        # Note on source and source_fn
        # Be careful with `source` when delegating to UserFunctionVariable
        # (base-class) methods. In this __init__, `source` is a *bound method*
        # object, but the base class expects the underlying *function* object.
        # One way is to simplly use `__func__` to unwrap it.
        #
        # For recursive dict-tag optimizations, it can be faster to fetch the
        # function directly from `cls.__dict__`; that's why we pass on
        # `source_fn`. Whenever it is possible to access the function from
        # cls.__dict__, we pass that on to `source_fn`. Because bind_args
        # operates on the unbound function, most guards should target
        # `source_fn` rather than the original `source`.
        if source_fn is None and kwargs.get("source") is not None:
            self.source_fn = AttrSource(kwargs.get("source"), "__func__")  # type: ignore[assignment, arg-type]

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns class name with the function and bound object.
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash of the bound method as (hash value, flag). With a
        # source, resolves the real method and hashes it (flag False). Otherwise
        # combines the object's hash with the function hash by xor, mapping -1 to
        # -2, and propagates the object hash's flag.
        # CPython method_hash: hash(self) ^ hash(func)
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/classobject.c#L304
        if self.source:
            real_val = tx.output.resolve_source_value(self.source)
            return hash(real_val), False
        # Sourceless: compute method_hash from components.
        from .object_protocol import generic_hash_impl

        self_hash, self_fake = generic_hash_impl(tx, self.obj)
        func_hash = hash(self.fn)
        h = self_hash ^ func_hash
        if h == -1:
            h = -2
        return h, self_fake

    def self_args(self) -> list[VariableTracker]:
        # [FUNCTION DOC] Returns the bound object as the sole implicit argument.
        return [self.obj]

    def python_type(self) -> type[types.MethodType]:
        # [FUNCTION DOC] Reports the bound-method type.
        return types.MethodType

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the bound method. Nonstrict-trace and leaf functions
        # are routed to an in-graph function tracker with obj prepended. On the
        # root tracer with an nn.Module object, methods defined in torch.nn
        # (except parametrize) or constant methods are delegated to the module
        # tracker's call_method. FSDP use_training_state becomes a context manager
        # class call. Constant methods are invoked and stored as constants.
        # Otherwise inlines via the parent.
        # NOTE this is to handle methods annotated by `nonstrict_trace`.
        # a `nonstrict_trace`-ed function will be wrapped by
        # `VariableTracker.build` and route to `TorchInGraphFunctionVariable`,
        # but in the case of method, we manually wrap it with `UserMethodVariable`
        # inside `UserDefinedObjectVariable.tp_getattro_impl`.
        #
        # We might be able to simplify this away by canonicalizing the
        # function/method wrapping code paths.
        from ..trace_rules import is_leaf_function, is_nonstrict_trace_callable

        if is_nonstrict_trace_callable(self.fn):
            call_args = [*self.self_args(), *args]
            var = variables.TorchInGraphFunctionVariable(
                self.fn, kind=variables.torch.AllowInGraphKind.NONSTRICT_TRACE
            )
            return var.call_function(tx, call_args, kwargs)

        if is_leaf_function(self.fn):
            call_args = [*self.self_args(), *args]
            var = variables.TorchInGraphFunctionVariable(
                self.fn, kind=variables.torch.AllowInGraphKind.LEAF_FUNCTION
            )
            return var.call_function(tx, call_args, kwargs)

        # For nn.Module methods, redirecting to NNModuleVariable.call_method for optimized solution
        # rather than simple inlining. E.g, putting `call_method` op in FX graph for `forward` method
        # since we ensure `forward` of allowed modules can be traced by AOT safely.
        # Note this is not only for allowed modules, as user customized modules can extend from
        # allowed modules but using parent's `forward` method, which is also covered by this branch.

        # If we are tracing the higher order op, we want Dynamo to step inside
        # the module call so that Dynamo can see the underlying parameters and
        # buffers and raise them as inputs to the graph. The is_root_tracer
        # check bypasses the if condition for non-root tracers and directly
        # calls the super().call_function at the end, which is basically
        # equivalent of inlining the method.
        if tx.output.is_root_tracer() and isinstance(
            self.obj, variables.NNModuleVariable
        ):
            module_attr = getattr(self.fn, "__module__", "")
            # inline torch.nn.utils.parametrize
            if (
                module_attr is not None
                and module_attr.startswith("torch.nn.")
                and module_attr != "torch.nn.utils.parametrize"
                or self.is_constant
            ):
                return self.obj.call_method(
                    tx, self.fn.__name__, list(args), kwargs, constant=self.is_constant
                )
        elif (
            _fsdp_param_group is not None
            and self.fn is _fsdp_param_group.FSDPParamGroup.use_training_state  # type: ignore[attr-defined]
        ):
            return variables.TorchCtxManagerClassVariable(self.fn).call_function(
                tx, [self.obj, *args], kwargs
            )
        if self.is_constant:
            fn = getattr(self.obj.value, self.fn.__name__)  # type: ignore[attr-defined]
            return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
        return super().call_function(tx, args, kwargs)

    def _get_func(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the function attribute of the method:
        # builds a tracker for the underlying function using source_fn.
        # We might have a better way to access the function object, this
        # information is stored in self.source_fn, use that to construct the
        # variable tracker.
        return VariableTracker.build(tx, self.fn, self.source_fn)  # type: ignore[arg-type]

    # __self__ / __func__ are read-only members on method objects.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/classobject.c#L20-L24
    tp_members = {
        "__self__": Member(getset_read(lambda s: s.obj)),
        "__func__": Member(_get_func, None),
    }

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Returns the unbound function.
        # KNOWN ISSUE: returns the unbound function rather than the bound method.
        return self.fn


class WrappedUserMethodVariable(UserMethodVariable):
    # [CLASS DOC] A method tracker that runs inside a context manager tracker:
    # attributes wrapped (the original method tracker) and context. Calls enter
    # the context, call the method, and exit; reconstructs as the context applied
    # to the wrapped method.
    def __init__(
        self,
        wrapped: UserMethodVariable,
        context: "ContextWrappingVariable",
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Initializes from the wrapped method's function and object
        # (ignoring any fn/obj passed in kwargs) and stores wrapped and context.
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, wrapped.obj, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the wrapped method inside its context manager. With
        # nested graph breaks enabled, calls a polyfill helper that applies the
        # context around the wrapped callable. Otherwise enters the context, calls
        # the method, exits the context (not on exceptions) and returns the result.
        if config.nested_graph_breaks:
            wrapper_fn = UserFunctionVariable(polyfills._fn_with_ctx)
            return wrapper_fn.call_function(
                tx, [self.context, self.wrapped] + list(args), kwargs
            )
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode applying the context object to the wrapped
        # method (a decorator-style call with one argument).
        codegen.add_push_null(lambda: codegen(self.context))  # type: ignore[arg-type]
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


class WrappedUserFunctionVariable(UserFunctionVariable):
    # [CLASS DOC] A function tracker that runs inside a context manager tracker:
    # attributes wrapped (original function tracker) and context. Calls enter the
    # context, call the function, and exit; reconstructs as the context applied
    # to the wrapped function.
    def __init__(
        self,
        wrapped: UserFunctionVariable,
        context: "ContextWrappingVariable",
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Initializes from the wrapped function (ignoring any fn in
        # kwargs) and stores wrapped and context.
        kwargs.pop("fn", None)
        super().__init__(wrapped.fn, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the function inside its context manager (via a
        # polyfill helper when nested graph breaks are enabled, otherwise enter,
        # call, exit). Returns the call result.
        if config.nested_graph_breaks:
            wrapper_fn = UserFunctionVariable(polyfills._fn_with_ctx)
            return wrapper_fn.call_function(
                tx, [self.context, self.wrapped] + list(args), kwargs
            )
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode applying the context object to the wrapped
        # function (a decorator-style call with one argument).
        codegen.add_push_null(lambda: codegen(self.context))  # type: ignore[arg-type]
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


def invoke_and_store_as_constant(
    tx: "InstructionTranslatorBase",
    fn: Callable[..., Any],
    name: str,
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] Runs fn eagerly at trace time with all arguments converted to
    # real values (tensors via their real value, user objects with an identity
    # guard, others via python constants; graph-breaks if not convertible), then
    # registers the result as a graph attribute named name with a constant source
    # and returns its tracker. Side effects: calls fn, installs guards, registers
    # an attribute on the output graph.
    def convert(x: VariableTracker) -> Any:
        # [FUNCTION DOC] HELPER: Converts one argument tracker to a real Python
        # value; graph-breaks if constant conversion is unsupported.
        if x.is_tensor():
            return cast("TensorVariable", x).get_real_value()
        if isinstance(x, UserDefinedObjectVariable):
            if x.source is not None:
                install_guard(x.make_guard(GuardBuilder.ID_MATCH))
            return x.value
        try:
            return x.as_python_constant()
        except AsPythonConstantNotImplementedError:
            unimplemented(
                gb_type="assume_constant_result argument conversion failed",
                context=f"function {name}, variable type {type(x).__name__}",
                explanation=f"Cannot convert argument of type {type(x).__name__} to a Python constant "
                f"for function {name} marked with torch._dynamo.assume_constant_result. "
                f"The variable tracker does not support constant conversion.",
                hints=[
                    "Remove torch._dynamo.assume_constant_result from this function",
                    "Ensure all arguments passed to the function can be converted to constants",
                ],
            )

    args = [convert(x) for x in args]
    kwargs = {k: convert(v) for k, v in kwargs.items()}
    res = fn(*args, **kwargs)
    return tx.output.register_attr_or_module(
        res,
        name,
        source=ConstantSource(name),
    )


class NestedUserFunctionVariable(BaseUserFunctionVariable):
    # [CLASS DOC] Tracker for a function created during tracing (closure creation
    # via MAKE_FUNCTION). Holds trackers for name, code, defaults, keyword
    # defaults and closure (a tuple of cells), plus the raw globals mapping and
    # optional wrapped_fn (from functools.wraps). Mutation type defaults to a new
    # attribute-mutable object. Can be converted to a real function when the
    # closure is constant, and is reconstructed via a helper that builds the
    # function at runtime.
    _nonvar_fields = {
        "f_globals",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    def __init__(
        self,
        fn_name: VariableTracker,
        code: VariableTracker,
        f_globals: dict[str, Any],
        defaults: VariableTracker | None,
        kwdefaults: VariableTracker | None,
        closure: VariableTracker | None,
        # This is present when this function is created by
        # `functools.wrap(wrapped_fn)(this_fn)`.
        wrapped_fn: VariableTracker | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Validates that the name is a constant string, code is a
        # constant code object and globals is a mapping (raises an error
        # otherwise), defaults the mutation type to a new attribute-mutable
        # object, and stores all components.
        if kwargs.get("mutation_type") is None:
            kwargs.update(mutation_type=AttributeMutationNew())
        super().__init__(**kwargs)
        if not isinstance(fn_name.as_python_constant(), str):
            raise AssertionError(
                f"fn_name must be a str, got {type(fn_name.as_python_constant())}"
            )
        if not isinstance(code.as_python_constant(), types.CodeType):
            raise AssertionError(
                f"code must be a CodeType, got {type(code.as_python_constant())}"
            )
        if not isinstance(f_globals, dict):
            raise AssertionError(f"f_globals must be a dict, got {type(f_globals)}")
        self.fn_name = fn_name
        self.code = code
        self.f_globals = f_globals
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.closure = closure
        self.wrapped_fn: VariableTracker | None = wrapped_fn

    def self_args(self) -> list[VariableTracker]:
        # [FUNCTION DOC] No implicit self arguments.
        return []

    def as_python_constant(self) -> types.FunctionType:
        # [FUNCTION DOC] Materializes and returns a real function (may raise
        # ClosureConversionError when the closure is not constant).
        return self.get_function()

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] repr via the parent; if the closure cannot be converted,
        # graph-breaks with an explanatory message.
        try:
            return super().tp_repr_impl(tx)
        except ClosureConversionError as e:
            unimplemented(
                gb_type="repr() on nested function with non-constructible closure",
                context=f"repr() on nested function {self.fn_name.as_python_constant()}: {e}",
                explanation="Dynamo could not safely evaluate repr() for this "
                "nested function because it could not reconstruct its closure "
                "as Python constants.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

    def get_code(self) -> types.CodeType:
        # [FUNCTION DOC] Returns the real code object.
        return self.code.as_python_constant()

    def python_type(self) -> type[types.FunctionType]:
        # [FUNCTION DOC] Reports the function type.
        return types.FunctionType

    def get_function(
        self,
        _converting: set[int] | None = None,
        *,
        allow_sourced_cells: bool = False,
    ) -> types.FunctionType:
        # [FUNCTION DOC] Builds a real function from the tracked components.
        # _converting is a set of ids of trackers currently being converted, used
        # to detect mutual recursion (raises ClosureConversionError). Any failure
        # to convert closure cells to constants is re-raised as
        # ClosureConversionError. allow_sourced_cells permits source-backed,
        # unmodified user objects in cells (guarded by identity).
        # _converting is used a way to break cycles when
        # two nested_functions refer to each other.
        from .base import AsPythonConstantNotImplementedError

        self_id = id(self)
        if _converting is None:
            _converting = set()
        if self_id in _converting:
            raise ClosureConversionError(
                "cycle detected in mutually recursive closures"
            )
        _converting.add(self_id)
        try:
            return self._get_function_impl(_converting, allow_sourced_cells)
        except AsPythonConstantNotImplementedError as e:
            raise ClosureConversionError(
                "failed to convert closure cell to Python constant"
            ) from e
        finally:
            _converting.discard(self_id)

    def is_python_constant(self) -> bool:
        # [FUNCTION DOC] True if the function can be materialized as a real
        # function without raising NotImplementedError or a graph break.
        try:
            self.as_python_constant()
            return True
        except (NotImplementedError, Unsupported):
            return False

    def _get_function_impl(
        self, _converting: set[int], allow_sourced_cells: bool
    ) -> types.FunctionType:
        # [FUNCTION DOC] HELPER: Core of get_function. Converts each closure cell
        # (loaded through side effects) into a real cell: nested function cells
        # recurse, constants are wrapped, and (if allowed) source-backed
        # unmodified user objects are guarded by identity and registered in side
        # effects so later inlining reuses the original tracker. Self-referential
        # closures raise ClosureConversionError. Then creates the function and
        # applies defaults, keyword defaults and annotations (a flat pair tuple is
        # turned into a mapping; non-mapping raises an error).
        # Side effects: may install guards and track new cells.
        closure_cells = None
        if self.closure:
            from torch._dynamo.symbolic_convert import InstructionTranslator

            tx = InstructionTranslator.current_tx()
            cells = []

            for cell_var in self.closure.items:  # type: ignore[attr-defined]
                # Get the cell contents from side_effects or pre_existing_contents
                # load_cell will replay the side-effects
                cell_contents = tx.output.side_effects.load_cell(cell_var)

                # Check for self-referential closure (function capturing itself for recursion)
                # For example:
                # def outer():
                #     def helper(n):
                #         if n <= 0:
                #             return 0
                #         return n + helper(n - 1)  # helper calls itself
                #     return helper
                if cell_contents is self:
                    raise ClosureConversionError("self-referential nested function")

                # If the cell contents is a NestedUserFunctionVariable, call get_function
                # directly to properly propagate the _converting set for cycle detection
                if isinstance(cell_contents, NestedUserFunctionVariable):
                    value = cell_contents.get_function(
                        _converting,
                        allow_sourced_cells=allow_sourced_cells,
                    )
                    cells.append(make_cell(value))
                    continue

                try:
                    value = cell_contents.as_python_constant()
                    cells.append(make_cell(value))
                    continue
                except (NotImplementedError, Unsupported):
                    if not allow_sourced_cells:
                        raise ClosureConversionError(
                            "failed to convert closure cell to Python constant"
                        ) from None

                # Non-constant closure cell (e.g. a class defined in a compiled
                # region closing over a real object such as a TestCase
                # instance). Only allow source-backed objects that have not
                # been mutated during the trace, and pin their identity with
                # an ID_MATCH guard. Materialize a real cell from the backing
                # object and register it in side_effects mapping back to the
                # traced VT, so that when methods of the built class are
                # later inlined, bind_args reuses the original VT (preserving
                # its source, guards, and identity) via load_cell. Only the
                # __build_class__ path opts in; as_python_constant
                # conversions must stay strict.
                if not (
                    isinstance(cell_contents, UserDefinedObjectVariable)
                    and cell_contents.source is not None
                    and not tx.output.side_effects.is_modified(cell_contents)
                ):
                    raise ClosureConversionError(
                        "failed to convert closure cell to Python constant"
                    )
                value = cell_contents.guard_as_python_constant()
                cell = make_cell(value)
                tx.output.side_effects.track_cell_existing(None, cell, cell_contents)
                cells.append(cell)
            closure_cells = tuple(cells)

        func = types.FunctionType(
            self.code.as_python_constant(),
            self.f_globals,
            self.fn_name.as_python_constant(),
            argdefs=None,
            closure=closure_cells,
        )
        if self.defaults:
            func.__defaults__ = self.defaults.as_python_constant()
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.as_python_constant()
        if self.annotations:
            annotations = self.annotations.as_python_constant()
            if isinstance(annotations, tuple):
                from itertools import pairwise

                annotations = dict(pairwise(annotations))

            # TypeError: __annotations__ must be set to a dict object
            if not isinstance(annotations, dict):
                raise AssertionError(
                    f"annotations must be a dict, got {type(annotations)}"
                )
            func.__annotations__ = annotations
        return func

    def has_closure(self) -> bool:
        # [FUNCTION DOC] True if a closure tracker is present.
        return self.closure is not None

    def const_getattr(self, tx: "InstructionTranslatorBase", name: str) -> Any:
        # [FUNCTION DOC] Constant attribute read for name, code and defaults
        # (defaults as a python constant or None); other names defer to the parent.
        if name == "__name__":
            return self.get_name()
        if name == "__code__":
            return self.get_code()
        if name == "__defaults__":
            d = getattr(self, "defaults", None)
            return d.as_python_constant() if d else None
        return super().const_getattr(tx, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # [FUNCTION DOC] hasattr: code and defaults checked against this tracker's
        # own fields; a name present in the function dict is True; otherwise defers
        # to the parent. Returns a bool tracker.
        # KNOWN ISSUE: the defaults check tests whether the tracker field exists
        # (always set in the constructor), so it is always True even for None.
        if name == "__code__":
            return VariableTracker.build(tx, hasattr(self, "code"))
        if name == "__defaults__":
            return VariableTracker.build(tx, hasattr(self, "defaults"))
        vt = ConstantVariable.create(name)
        if vt in self.get_dict_vt(tx):
            return ConstantVariable.create(True)
        return super().call_obj_hasattr(tx, name)

    def has_self(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def get_globals(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the globals mapping of the nested function.
        return self.f_globals

    def bind_args(
        self,
        parent: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> dict[str, VariableTracker]:
        # [FUNCTION DOC] Builds the initial locals for inlining a call: creates a
        # dummy function with the tracked defaults and placeholder cells, binds
        # args/kwargs using the standard signature binder (applying defaults), wraps
        # packed varargs/varkw values, creates cells for cell variables, and maps
        # each free variable to the corresponding tracked closure cell.
        # Raises BindArgsTypeError when arguments do not match the signature.
        code = self.get_code()
        func = types.FunctionType(
            code,
            self.f_globals,
            self.fn_name.as_python_constant(),
            tuple(self.defaults.items) if self.defaults else None,  # type: ignore[attr-defined]
            tuple(make_cell(None) for _ in range(len(self.get_code().co_freevars))),
        )
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.keys_as_python_constant()  # type: ignore[missing-attribute]
        try:
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            raise BindArgsTypeError(e.args[0]) from e
        result = dict(bound.arguments.items())
        wrap_args_kwargs(parent.output.root_tx, result)  # type: ignore[arg-type]
        init_cellvars(parent, result, code)

        for idx, name in enumerate(code.co_freevars):
            if name in result:
                raise AssertionError(f"free variable {name!r} already in result")
            cell = self.closure.items[idx]  # type: ignore[attr-defined, union-attr]
            result[name] = cell

        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that calls the nested-function factory with
        # code, globals, name, defaults, closure, keyword defaults and annotations
        # (None where absent). If created by functools.wraps, applies that wrapper.
        # Then replays pending attribute mutations by storing each recorded
        # attribute onto the new function.
        codegen.add_push_null(
            lambda: codegen.load_import_from(__name__, "_create_nested_fn")
        )
        codegen(self.code)
        codegen.extend_output([codegen.create_load_const_unchecked(self.f_globals)])
        codegen(ConstantVariable.create(self.code.value.co_name))  # type: ignore[attr-defined]

        if self.defaults:
            codegen(self.defaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.closure:
            codegen(self.closure)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.kwdefaults:
            codegen(self.kwdefaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.annotations is not None:
            try:
                annotations = self.annotations.as_python_constant()
                codegen.extend_output(
                    [codegen.create_load_const_unchecked(annotations)]
                )
            except NotImplementedError:
                codegen(self.annotations)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        codegen.extend_output(create_call_function(7, False))

        if self.wrapped_fn:
            codegen.add_push_null(
                lambda: codegen.load_import_from("functools", "wraps")
            )
            codegen(self.wrapped_fn)
            codegen.extend_output(create_call_function(1, False))
            codegen.extend_output(create_rot_n(2))
            codegen.extend_output(create_call_function(1, True))

        # codegen attributes
        tx = codegen.tx
        if tx.output.side_effects.has_pending_mutation(self):
            for name, value in tx.output.side_effects.store_attr_mutations[
                self
            ].items():
                codegen.dup_top()
                codegen(value)
                codegen.extend_output(create_rot_n(2))
                codegen.store_attr(name)


class WrappedNestedUserFunctionVariable(NestedUserFunctionVariable):
    # [CLASS DOC] A nested function tracker that runs inside a context manager
    # tracker: attributes wrapped (original nested function tracker) and context.
    # Calls enter the context, call the function, and exit; reconstructs as the
    # context applied to the wrapped function.
    def __init__(
        self,
        wrapped: NestedUserFunctionVariable,
        context: "ContextWrappingVariable",
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Copies all components of the wrapped nested function
        # (ignoring same-named kwargs), shares its annotations, and stores wrapped
        # and context.
        kwargs.pop("fn_name", None)
        kwargs.pop("code", None)
        kwargs.pop("f_globals", None)
        kwargs.pop("defaults", None)
        kwargs.pop("kwdefaults", None)
        kwargs.pop("annotations", None)
        kwargs.pop("closure", None)
        kwargs.pop("wrapped_fn", None)
        super().__init__(
            wrapped.fn_name,
            wrapped.code,
            wrapped.f_globals,
            wrapped.defaults,
            wrapped.kwdefaults,
            wrapped.closure,
            wrapped.wrapped_fn,
        )
        self.annotations = wrapped.annotations
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the function inside its context manager (via a
        # polyfill helper when nested graph breaks are enabled, otherwise enter,
        # call, exit). Returns the call result.
        if config.nested_graph_breaks:
            wrapper_fn = UserFunctionVariable(polyfills._fn_with_ctx)
            return wrapper_fn.call_function(
                tx, [self.context, self.wrapped] + list(args), kwargs
            )
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode applying the context object to the wrapped
        # nested function (a decorator-style call with one argument).
        codegen.add_push_null(lambda: codegen(self.context))
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


RE_CONSTANT_FOLD_FNS = {
    re.search,
    re.match,
    re.fullmatch,
    re.compile,
    re.sub,
    re.subn,
    re.split,
    re.findall,
    re.escape,
}


class SkipFunctionVariable(VariableTracker):
    # [CLASS DOC] Tracker for a callable that Dynamo must not trace into (value is
    # the real callable, reason an optional skip reason). Calling it either
    # constant-folds known pure helpers (importlib lookups, re functions with
    # constant arguments), handles special Dynamo APIs (graph_break, skip_frame,
    # step_unsupported, disable wrappers, function descriptor get), or raises a
    # graph break with a detailed explanation and hints.
    _nonvar_fields = {
        "value",
        "reason",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, value: Any, reason: str | None = None, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the skipped callable and the optional reason.
        super().__init__(**kwargs)
        self.value = value
        self.reason = reason

    def get_value_for_setattr(self) -> object | None:
        # [FUNCTION DOC] Returns the callable for setattr unless it belongs to a
        # torch module (module name torch or starting with torch. / torch_), in
        # which case returns None.
        mod = getattr(self.value, "__module__", None) or ""
        if mod == "torch" or mod.startswith(("torch.", "torch_")):
            return None
        return self.value

    def tp_richcompare_impl(self, tx, other, op):
        # [FUNCTION DOC] Identity-based comparison via the generic object helper.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns the skipped callable.
        return self.value

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Returns the skipped callable.
        return self.value

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "SkipFunctionVariable":
        # [FUNCTION DOC] Factory that installs guards depending on the kind of
        # value: for disable-wrapped functions, walks to the original callable (the
        # guard is created but not installed); for builtins without a bound
        # non-module owner, installs a builtin match guard; for other values that
        # are not wrapper/member descriptors, installs a code-identity guard.
        # Returns a new tracker with the given source.
        # KNOWN ISSUE: in the disable-wrapper branch make_guard is called but the
        # result is never passed to install_guard, so no guard is installed.
        # Use closure match guard (i.e. guard on __code__ object instead of
        # function id) to avoid guarding on nested functions.
        if inspect.getattr_static(value, "_torchdynamo_disable", False):
            # For torch._dynamo.disable function, ensure that the original
            # function is guarded. Otherwise, the else branch will guard on the
            # _dynamo.disable.__code__
            guard_on_source = source
            guard_on_value = value

            while inspect.getattr_static(
                guard_on_value, "_torchdynamo_orig_callable", False
            ):
                guard_on_value = guard_on_value._torchdynamo_orig_callable
                guard_on_source = AttrSource(
                    guard_on_source, "_torchdynamo_orig_callable"
                )

            guard_on_source.make_guard(GuardBuilder.CLOSURE_MATCH)
        elif inspect.isbuiltin(value):
            # Bound builtin methods (e.g. obj.__reduce_ex__) are created fresh
            # on every attribute access, so their id() is unstable.  Skip the
            # id-based BUILTIN_MATCH guard for them — the type guard on
            # the owner object is sufficient.
            if not hasattr(value, "__self__") or isinstance(
                value.__self__, types.ModuleType
            ):
                install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        elif not is_wrapper_or_member_descriptor(value):
            # These descriptors are not guaranteed to return the same object on
            # attribute lookup. They are unlikely to be changed, so we can skip
            # guarding them.
            install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        return cls(value, source=source)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handles a call to a skipped callable. Constant-folds
        # importlib find_spec/version and re module functions when all arguments are
        # constants. Raises graph breaks (observed as Unsupported) for direct calls
        # to the disable APIs, calls to disabled functions, graph_break (with
        # optional message), skip_frame (frame skipped), and step_unsupported
        # (converted to StepUnsupported). Function descriptor get with 2-3 args on a
        # user function yields a bound method tracker. Otherwise, if configured to
        # not skip tracing, retries by rebuilding the callable; else builds an
        # explanation/hints (file path, builtin, optree, allow_in_graph cases),
        # may emit a one-time warning, and graph-breaks with the skip reason.
        # importlib functions are frozen builtins that Dynamo cannot trace
        # into.  They are deterministic for a given package name, so
        # constant-fold them when all args are constants.
        if self.value in (importlib.util.find_spec, importlib.metadata.version) and all(
            a.is_python_constant() for a in args
        ):
            return VariableTracker.build(
                tx, self.value(*(a.as_python_constant() for a in args))
            )

        if (
            self.value in RE_CONSTANT_FOLD_FNS
            and all(a.is_python_constant() for a in args)
            and all(v.is_python_constant() for v in kwargs.values())
        ):
            result = self.value(
                *(a.as_python_constant() for a in args),
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            return VariableTracker.build(tx, result)

        def unimplemented_direct_disable_call(api_name: str) -> Never:
            # [FUNCTION DOC] HELPER: Raises a graph break stating that the named
            # disable API was called inside a compiled region, with hints to move
            # it outside or use an explicit graph break.
            # The registry linter keys off this helper name and records concrete
            # entries from the call sites below. Use an alias here so the
            # parameterized helper body is not recorded as a generic
            # `{api_name}` entry.
            _unimplemented = unimplemented
            _unimplemented(
                gb_type=f"Call to `{api_name}()`",
                context=f"Called `{api_name}()` with args `{args}`, kwargs `{kwargs}`",
                explanation=f"`{api_name}()` was called inside a compiled region. "
                "This API disables compilation when used as a decorator or wrapper "
                "outside the compiled region.",
                hints=[
                    f"Move the `{api_name}()` call outside the compiled function and apply it to the function that should run eagerly.",
                    "Use `torch._dynamo.graph_break()` to intentionally insert a graph break at this point.",
                ],
            )

        if self.value is torch._dynamo.disable:
            unimplemented_direct_disable_call("torch._dynamo.disable")
        elif self.value is torch.compiler.disable:
            unimplemented_direct_disable_call("torch.compiler.disable")
        elif inspect.getattr_static(self.value, "_torchdynamo_disable", False):
            msg = inspect.getattr_static(self.value, "_torchdynamo_disable_msg", None)
            unimplemented(
                gb_type="Skip calling `torch.compiler.disable()`d function",
                context=str(self.value),
                explanation=f"Skip calling function `{self.value}` since it was wrapped "
                f"with `torch.compiler.disable` (reason: {msg})",
                hints=[
                    "Remove the `torch.compiler.disable` call",
                ],
            )
        elif self.value is torch._dynamo.graph_break:
            graph_break_msg = kwargs.get("msg")
            if graph_break_msg:
                graph_break_msg = graph_break_msg.as_python_constant()
            unimplemented(
                gb_type="Call to `torch._dynamo.graph_break()`",
                context=f"Called `torch._dynamo.graph_break()` with args `{args}`, kwargs `{kwargs}`",
                explanation=f"User-inserted graph break. Message: {graph_break_msg}",
                hints=[
                    "Remove the `torch._dynamo.graph_break()` call.",
                ],
            )
        elif self.value is torch._dynamo.skip_frame:
            skip_frame_msg = kwargs.get("msg")
            if skip_frame_msg:
                skip_frame_msg = skip_frame_msg.as_python_constant()
            else:
                skip_frame_msg = ""
            unimplemented(
                gb_type="Call to `torch._dynamo.skip_frame()`",
                context=f"Called `torch._dynamo.skip_frame()` with args `{args}`, kwargs `{kwargs}`. "
                f"Skipping frame {format_frame_info(tx.f_code)}.",
                explanation=f"User-inserted skip frame. Message: {skip_frame_msg}",
                hints=[
                    "Remove the `torch._dynamo.skip_frame()` call.",
                ],
                skip_frame=True,
            )
        elif self.value is torch._dynamo.step_unsupported:
            try:
                unimplemented(
                    gb_type="Call to `torch._dynamo.step_unsupported()`",
                    context="",
                    explanation="User-inserted step_unsupported.",
                    hints=[
                        "Remove the `torch._dynamo.step_unsupported()` call.",
                    ],
                )
            except Unsupported as e:
                raise StepUnsupported(e.msg) from None
        elif self.value is types.FunctionType.__get__:
            # function.__get__(func, obj[, cls]) produces a bound method.
            # This is called by inspect._descriptor_get when resolving
            # descriptors during inspect.signature().
            # Note that function.__get__ does not use the 3rd argument. The
            # reason it still has the 3rd argument is because descriptors follow
            # a function signature that takes 3 arguments, and other descriptors
            # (not function.__get__) can use the 3rd argument.
            if len(args) in (2, 3) and not kwargs:
                func_var = args[0]
                obj_var = args[1]
                if isinstance(func_var, UserFunctionVariable):
                    return UserMethodVariable(
                        func_var.fn, obj_var, source_fn=func_var.source
                    )
            unimplemented(
                gb_type="unsupported function.__get__ call",
                context=f"call_function {self}, args: {args}, kwargs: {kwargs}",
                explanation="Dynamo only supports function.__get__(func, obj[, cls]) "
                "where func is a user-defined function.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        else:
            if config.dont_skip_tracing:
                from .builder import SourcelessBuilder

                # re-build the function, attempting to not skip
                rebuilt_fn = SourcelessBuilder.create(tx, self.value)
                # if we still get SkipFunctionVariable, then we *really* should skip this function
                if not isinstance(rebuilt_fn, SkipFunctionVariable):
                    return rebuilt_fn.call_function(tx, args, kwargs)
            qualname = getattr(self.value, "__qualname__", "<unknown qualname>")
            module_or = getattr(self.value, "__module__", None)
            module_name = "<unknown module>" if module_or is None else str(module_or)
            try:
                path = inspect.getfile(self.value)
                explanation = (
                    f"Dynamo developers have intentionally marked that the function `{qualname}` "
                    f"in file `{path}` should not be traced."
                )
                hints = [
                    f"Avoid calling the function `{qualname}`.",
                ]
                # TODO improve trace_rules reasoning to provide better hints.
                # How do we tell that a function/file should NOT be removed from skip files?
                # Do a very basic check for now.
                if "_dynamo" not in path:
                    hints += [
                        f"Apply `@torch._dynamo.dont_skip_tracing` to the function `{qualname}` "
                        "to force tracing into the function. "
                        "More graph breaks may occur as a result of attempting to trace into the function.",
                        "Please file an issue to PyTorch.",
                    ]
            except TypeError:
                known_python_builtin_modules = {"_abc", "_warnings"}
                if module_or in known_python_builtin_modules:
                    explanation = (
                        f"Dynamo does not know how to trace the Python builtin "
                        f"`{module_name}.{qualname}`."
                    )
                    hints = [
                        "If you are attempting to call a logging function (e.g. `_warnings.warn`), "
                        "you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.",
                        "Please file an issue on GitHub "
                        "so the PyTorch team can add support for it. ",
                    ]
                elif module_or is not None and module_or.startswith("optree"):
                    explanation = f"Dynamo cannot trace optree C/C++ function {module_name}.{qualname}."
                    hints = [
                        " Consider using torch.utils._pytree - "
                        "https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py"
                    ]
                    # also warn on it because most users won't see the graph break message
                    torch._dynamo.utils.warn_once(explanation + "\n" + "\n".join(hints))
                else:
                    explanation = (
                        f"Dynamo does not know how to trace the builtin `{module_name}.{qualname}.` "
                        f"This function is either a Python builtin (e.g. _warnings.warn) "
                        f"or a third-party C/C++ Python extension (perhaps created with pybind)."
                    )
                    hints = [
                        "If it is a Python builtin, please file an issue on GitHub "
                        "so the PyTorch team can add support for it and see the next case for a workaround.",
                        "If it is a third-party C/C++ Python extension, please "
                        "either wrap it into a PyTorch-understood custom operator "
                        "(see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html "
                        "for more details) or, if it is traceable, use "
                        "`torch.compiler.allow_in_graph`.",
                    ]
                    # also warn on it because most users won't see the graph break message
                    torch._dynamo.utils.warn_once(explanation + "\n" + "\n".join(hints))
            if qualname == "allow_in_graph":
                explanation = (
                    "torch.compiler.allow_in_graph (or torch._dynamo.allow_in_graph) "
                    "was called inside a compiled region. Dynamically annotating functions "
                    "inside a compiled region is not supported."
                )
                hints = [
                    "Apply @torch.compiler.allow_in_graph as a decorator before compilation, "
                    "not inside the compiled function.",
                ]
            if self.reason:
                reason = self.reason
            else:
                from ..trace_rules import get_skip_reason

                reason = get_skip_reason(self.value)
            unimplemented(
                gb_type="Attempted to call function marked as skipped",
                context=f"module: {module_name}, qualname: {qualname}, skip reason: {reason}",
                explanation=explanation,
                hints=hints,
            )

    def reconstruct_pycode(self, codegen):
        # [FUNCTION DOC] Emits Python source reconstruction via the source; raises
        # NotImplementedError if there is no source.
        if self.source:
            return self.source.reconstruct_pycode(codegen)
        raise NotImplementedError(
            "Python codegen not implemented for sourceless SkipFunctionVariable"
        )


class WrappedSkipFunctionVariable(SkipFunctionVariable):
    # [CLASS DOC] A skipped-function tracker that runs inside a context manager
    # tracker: attributes wrapped (original skip tracker) and context. Calls enter
    # the context, call, and exit; reconstructs as the context applied to the
    # wrapped function.
    def __init__(
        self,
        wrapped: SkipFunctionVariable,
        context: "ContextWrappingVariable",
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Copies value and reason from the wrapped tracker (ignoring
        # same-named kwargs) and stores wrapped and context.
        kwargs.pop("value", None)
        kwargs.pop("reason", None)
        super().__init__(wrapped.value, reason=wrapped.reason, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the skipped function inside its context manager (via
        # a polyfill helper when nested graph breaks are enabled, otherwise enter,
        # call, exit). The skip behavior normally graph-breaks.
        if config.nested_graph_breaks:
            wrapper_fn = UserFunctionVariable(polyfills._fn_with_ctx)
            return wrapper_fn.call_function(
                tx, [self.context, self.wrapped] + list(args), kwargs
            )
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode applying the context object to the wrapped
        # function (a decorator-style call with one argument).
        codegen.add_push_null(lambda: codegen(self.context))
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


class WrapperUserFunctionVariable(BaseUserFunctionVariable):
    # [CLASS DOC] Tracker for a wrapper object (e.g. jit script/trace wrappers,
    # lru_cache wrappers) whose real callable lives in attribute attr_to_trace of
    # wrapper_obj. Calls trace the inner callable through a polyfill, warning once
    # for non-torch lru_cache wrappers.
    """
    Used to represent a wrapper object that contains the actual callable as an
    attribute. For example, torch.jit.script/trace have the original function at
    their _torchdynamo_inline attribute. Similarly, functions with
    __script_if_tracing_wrapper have the original attr at "__original_fn".
    """

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the function type.
        return types.FunctionType

    def __init__(self, wrapper_obj: Any, attr_to_trace: str, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the wrapper object and the name of the attribute
        # that holds the callable to trace.
        super().__init__(**kwargs)
        self.wrapper_obj = wrapper_obj
        self.attr_to_trace = attr_to_trace

    def get_module(self) -> str:
        # [FUNCTION DOC] Returns the wrapper's module name.
        return self.wrapper_obj.__module__

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the wrapper's name.
        return self.wrapper_obj.__name__

    def get_code(self) -> types.CodeType:
        # [FUNCTION DOC] Returns the code object of the traced inner callable.
        return self.get_function().__code__

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute access: the traced attribute name returns a
        # built tracker for the inner callable (with attribute source); all other
        # names defer to the parent implementation.
        if name == self.attr_to_trace:
            val = getattr(self.wrapper_obj, self.attr_to_trace)
            source = self.source and AttrSource(self.source, name)
            return VariableTracker.build(tx, val, source)
        return super().tp_getattro_impl(tx, name)

    def get_function(self):
        # [FUNCTION DOC] Returns the inner callable read from the wrapper.
        return getattr(self.wrapper_obj, self.attr_to_trace)

    def lookup_instance_dict(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] Looks up name in the wrapper object's own instance dict
        # (not the inner callable's); returns None if absent, else a lazy tracker
        # with attribute source, or a built tracker when no source exists.
        # self.source denotes the wrapper, not the inline target reached via
        # attr_to_trace. Resolve instance-dict attrs (e.g. functools.wraps'
        # __wrapped__) against the wrapper so the built value matches the source;
        # the base impl reflects on get_function() (the inline target), which
        # would pair the inline target's value with the wrapper's source and
        # yield a self-inconsistent guard (issue: swap/getfullargspec repro).
        wrapper_dict = getattr(self.wrapper_obj, "__dict__", None)
        if not wrapper_dict or name not in wrapper_dict:
            return None
        source = self.get_source()
        source = AttrSource(source, name) if source is not None else None
        if source is not None:
            return variables.LazyVariableTracker.create(
                wrapper_dict[name], source, tx=tx
            )
        return VariableTracker.build(tx, wrapper_dict[name])

    def self_args(self) -> list[VariableTracker]:
        # [FUNCTION DOC] No implicit self arguments.
        return []

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the inner callable via a tracing polyfill, with self
        # args prepended. For lru_cache wrappers defined outside torch (and not
        # explicitly allowed), warns once and logs a debug stack that the cache
        # is ignored. For an inner torch.compile wrapper (other than a recursive
        # call of the root frame) uses the polyfill variant that disables nested
        # graph breaks. Side effects: warnings and logging.
        if hasattr(self.wrapper_obj, "cache_info"):
            target_fn = getattr(self.wrapper_obj, self.attr_to_trace, None)
            module_name = getattr(target_fn, "__module__", "") or ""
            is_allowed_lru_cache_wrapper = (
                is_lru_cache_wrapper_trace_without_warning_allowed(self.wrapper_obj)
            )

            if (
                module_name.split(".", maxsplit=1)[0] != "torch"
                and not is_allowed_lru_cache_wrapper
            ):
                frame_summary = tx.frame_summary()
                filename = os.path.basename(frame_summary.filename)
                lineno = frame_summary.lineno
                msg = (
                    "Dynamo detected a call to a `functools.lru_cache`-wrapped "
                    f"function at '{filename}:{lineno}'. Dynamo ignores the "
                    "cache wrapper and directly traces the wrapped function. "
                    "Silent incorrectness is only a *potential* risk, not "
                    "something we have observed. "
                    "Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.\n\n"
                    "This call originates from:\n"
                    f"{''.join(traceback.format_list([frame_summary]))}"
                )

                torch._dynamo.utils.warn_once(msg)

                dynamo_logger = torch._dynamo.utils.logging.getLogger("torch._dynamo")
                if dynamo_logger.isEnabledFor(logging.DEBUG):
                    user_stack = torch._guards.TracingContext.extract_stack()
                    user_stack = get_stack_above_dynamo() + user_stack
                    frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                    user_stack_formatted = "".join(traceback.format_list(user_stack))
                    user_stack_trace = f"call to a lru_cache wrapped function at: {frame_loc[0]}:{frame_loc[1]}\n"
                    user_stack_trace += str(user_stack_formatted)
                    dynamo_logger.debug(user_stack_trace)

        all_args = self.self_args() + list(args)
        # Inner torch.compile wrapper: disable nested graph breaks to
        # preserve the inner compile's semantics (e.g. fullgraph=True).
        # Graph breaks inside the inner function should raise Unsupported
        # so they're handled by the outer frame, not as nested breaks.
        # Skip this for recursive calls to the same compiled function
        # (the wrapper's original callable matches the root frame's code).
        is_inner_torch_compile = (
            self.attr_to_trace == "_torchdynamo_inline"
            and inspect.getattr_static(self.wrapper_obj, "_is_torch_compile", False)
            and getattr(
                inspect.getattr_static(
                    self.wrapper_obj, "_torchdynamo_orig_callable", None
                ),
                "__code__",
                None,
            )
            is not tx.output.root_tx.f_code
        )
        polyfill = (
            polyfills.getattr_and_trace_no_nested_graph_breaks
            if is_inner_torch_compile
            else polyfills.getattr_and_trace
        )
        return VariableTracker.build(
            tx,
            polyfill,  # type: ignore[arg-type]
        ).call_function(
            tx,
            [self, VariableTracker.build(tx, self.attr_to_trace), *all_args],
            kwargs,
        )

    def get_real_python_backed_value(self) -> object:
        # [FUNCTION DOC] Returns the wrapper object (not the inner callable).
        # This VT stands for the wrapper, which is also what self.source
        # denotes. The inline target is reached via attr_to_trace and is a
        # different object.
        return self.wrapper_obj


class WrapperUserMethodVariable(WrapperUserFunctionVariable):
    # [CLASS DOC] Method flavor of the wrapper tracker: also stores the bound self
    # tracker (obj), which is prepended to call arguments.
    """
    Similar to WrapperUserFunctionVariable, but for methods. The only delta is
    saving the vt for `self` object of the method which is then used by
    WrapperUserFunctionVariable in `call_function` method.
    """

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the bound-method type.
        return types.MethodType

    def __init__(
        self,
        wrapper_obj: Any,
        attr_to_trace: str,
        self_obj: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Initializes the wrapper and stores the bound self tracker.
        super().__init__(wrapper_obj, attr_to_trace, **kwargs)
        self.obj = self_obj

    def self_args(self) -> list[VariableTracker]:
        # [FUNCTION DOC] Returns the bound self tracker as the implicit argument.
        return [self.obj]


def _traceable_collective_remaps() -> dict[Any, Any]:
    # [FUNCTION DOC] HELPER: Returns the mapping from distributed collective
    # functions to traceable replacements, or an empty mapping when distributed
    # support is not available.
    # We can't rely on importing from distributed, since it's not always built
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import (
            traceable_collective_remaps,
        )

        return traceable_collective_remaps
    return {}


def _traceable_collectives_source(
    tx: "InstructionTranslatorBase", fn: Callable[..., Any]
) -> AttrSource:
    # [FUNCTION DOC] HELPER: Returns the attribute source of the traceable
    # collective replacement fn inside the functional collectives module. Raises
    # an error if distributed is unavailable or fn is not a known replacement.
    if not torch.distributed.is_available():
        raise AssertionError("Illegal invocation.")
    if fn not in _traceable_collective_remaps().values():
        raise AssertionError(f"{fn} is not a traceable collective remap")

    inner_name = fn.__name__
    path_source = tx.import_source("torch.distributed._functional_collectives")
    return AttrSource(path_source, inner_name)


class CollectiveFunctionRewriteVariable(UserFunctionVariable):
    # [CLASS DOC] Tracker for a torch.distributed collective that is rewritten to
    # its traceable functional-collective counterpart (replacement_var). Provides
    # static helpers to test and perform the rewrite; calls normalize arguments,
    # graph-break on unsupported cases, and forward to the replacement.
    """
    Some of the torch.distributed.* collective APIs are possible to rewrite to 'traceable' collectives.

    This class provides both a way to check if a function is remappable, and perform the remapping.

    In the case that a function is 'remappable' but only for some combinations of call-time arguments,
    we check the args at `call_function` time and fall back to graph-breaking if needed.  This is no worse
    than status-quo as we currently graph-break on all distributed.* collectives.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        replacement_var: UserFunctionVariable,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Initializes the original function tracker and stores the
        # replacement tracker; raises an error if it is not a UserFunctionVariable.
        super().__init__(fn, **kwargs)  # type: ignore[arg-type]
        if not isinstance(replacement_var, UserFunctionVariable):
            raise AssertionError(
                f"replacement_var must be a UserFunctionVariable, got {type(replacement_var)}"
            )
        self.replacement_var = replacement_var

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        old_fn: Callable[..., Any],
        source: Source,
        **options: Any,
    ) -> "CollectiveFunctionRewriteVariable":
        # [FUNCTION DOC] Factory: computes the replacement function and its source
        # and returns a rewrite tracker for old_fn (with the given source) holding
        # a user-function tracker of the replacement.
        new_fn, new_source = CollectiveFunctionRewriteVariable.rewrite(tx, old_fn)
        return CollectiveFunctionRewriteVariable(
            old_fn,
            replacement_var=UserFunctionVariable(new_fn, source=new_source, **options),
            source=source,
            **options,
        )

    @staticmethod
    def can_rewrite(variable: Any) -> bool:
        # [FUNCTION DOC] True if variable is a plain function registered in the
        # traceable-collective remap table.
        return (
            inspect.isfunction(variable) and variable in _traceable_collective_remaps()
        )

    @staticmethod
    def rewrite(
        tx: "InstructionTranslatorBase", fn: Callable[..., Any]
    ) -> tuple[Any, AttrSource]:
        # [FUNCTION DOC] Looks up fn in the remap table (KeyError if missing) and
        # returns the replacement function with its attribute source.
        new_fn = _traceable_collective_remaps()[fn]
        return new_fn, _traceable_collectives_source(tx, new_fn)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the collective: merges positional args into keyword
        # args using the original signature, graph-breaks for async_op true, and
        # forwards to the replacement. Special cases: batch_isend_irecv converts a
        # list of P2P operations into parallel lists of op names, peers, tags,
        # tensors plus a group name (needs the p2p compilation config, else graph
        # break; raises internal errors for malformed P2P ops); isend/irecv need the
        # p2p config; reduction collectives map the reduce op to its string name
        # (raises ValueError for unsupported ops).
        # KNOWN ISSUE: after merging, args is rebound to an empty list before the
        # async_op graph-break message formats it, so the context omits positional
        # args; also the reduce-op value is read from the tracker's value attribute.
        # call_function must check any unsupported arguments and graph-break.
        # It's safe to assume args/kwargs from orig_fn map 1:1 to args/kwargs of remapped_fn,
        # since that's the contract for putting a mapping in `traceable_collective_remaps`
        import torch.distributed as dist
        from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

        # Merge args into kwargs so positional and keyword args
        # can be processed the same way.
        signature = inspect.signature(self.fn)
        kwargs = dict(signature.bind(*args, **kwargs).arguments)
        args = []

        if "async_op" in kwargs and kwargs["async_op"].as_python_constant():
            unimplemented(
                gb_type="async_op=True for distributed collectives",
                context=f"{self.fn}, {args=}, {kwargs=}",
                explanation=f"`torch.compile` doesn't support `async_op=True for {self.fn}",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        if self.fn == dist.batch_isend_irecv:
            if not config.enable_p2p_compilation:
                unimplemented(
                    gb_type="P2P compilation disabled for batch_isend_irecv",
                    context=f"{self.fn}",
                    explanation="P2P compilation is disabled.",
                    hints=[
                        "Set TORCHDYNAMO_ENABLE_P2P_COMPILATION=1 to enable.",
                    ],
                )

            p2p_ops = kwargs["p2p_op_list"]
            if not isinstance(p2p_ops, variables.ListVariable):
                raise torch._dynamo.exc.InternalTorchDynamoError(
                    "`P2POp` used incorrectly"
                )

            ops: list[VariableTracker] = list()
            peers = list()
            tags = list()
            tensors = list()
            group_var: VariableTracker | None = None

            for item in p2p_ops.items:
                if item.python_type() is not dist.P2POp:
                    raise torch._dynamo.exc.InternalTorchDynamoError(
                        "`P2POp` used incorrectly"
                    )

                op_var = item.tp_getattro_impl(tx, "op")
                if op_var.is_python_constant():
                    op = op_var.as_python_constant()
                    if op not in (dist.isend, dist.irecv):
                        raise torch._dynamo.exc.InternalTorchDynamoError(
                            f"unexpected P2POp op {op}"
                        )
                    op_var = variables.ConstantVariable.create(op.__name__)
                elif hasattr(op_var, "get_name"):
                    op_var = variables.ConstantVariable.create(op_var.get_name())
                else:
                    raise torch._dynamo.exc.InternalTorchDynamoError(
                        f"unexpected P2POp op variable {op_var}"
                    )

                ops.append(op_var)
                tensors.append(item.tp_getattro_impl(tx, "tensor"))
                # batch_p2p_ops expects a group-local rank, which is what
                # P2POp.group_peer provides.
                peers.append(item.tp_getattro_impl(tx, "group_peer"))
                tags.append(item.tp_getattro_impl(tx, "tag"))
                if group_var is None:
                    group_var = item.tp_getattro_impl(tx, "group")

            if group_var is None:
                raise AssertionError("group_var must be set from P2POp items")
            new_args: list[VariableTracker] = []
            new_kwargs: dict[str, VariableTracker] = {
                "op_list": variables.ListVariable(ops),
                "peer_list": variables.ListVariable(peers),
                "tag_list": variables.ListVariable(tags),
                "tensors": variables.ListVariable(tensors),
                "group_name": group_var,
            }
            return self.replacement_var.call_function(tx, new_args, new_kwargs)

        if self.fn in (dist.isend, dist.irecv):
            if not config.enable_p2p_compilation:
                unimplemented(
                    gb_type="P2P compilation disabled for isend/irecv",
                    context=f"{self.fn}",
                    explanation="P2P compilation is disabled.",
                    hints=[
                        "Set TORCHDYNAMO_ENABLE_P2P_COMPILATION=1 to enable.",
                    ],
                )

            return self.replacement_var.call_function(tx, args, kwargs)

        if self.fn in (
            dist.all_reduce,
            dist.reduce_scatter,
            dist.reduce_scatter_single,
            # pyrefly: ignore [deprecated]
            dist.reduce_scatter_tensor,
            # pyrefly: ignore [deprecated]
            dist._reduce_scatter_base,
        ):
            reduce_op_var = kwargs.get("op")
            reduce_op = (
                reduce_op_var.value  # type: ignore[attr-defined]
                if reduce_op_var is not None
                else signature.parameters["op"].default
            )
            if reduce_op not in REDUCE_OP_TO_STR:
                raise ValueError(f"Unsupported all_reduce op: {reduce_op}")
            kwargs["op"] = VariableTracker.build(tx, REDUCE_OP_TO_STR[reduce_op])
        return self.replacement_var.call_function(tx, args, kwargs)


class CollectionsNamedTupleFunction(UserFunctionVariable):
    # [CLASS DOC] Tracker for the namedtuple factory function. Calls with constant
    # arguments are executed eagerly, producing a class tracker for the new type.
    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns the factory function itself.
        return self.fn

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] If all arguments are constants, calls the real factory and
        # returns a mutable user-defined class tracker for the created type; a
        # TypeError from the factory is re-raised as an observed exception.
        # Otherwise graph-breaks.
        constant_args = check_constant_args(args, kwargs)
        if constant_args:
            try:
                value = self.fn(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                )
            except TypeError as exc:
                raise_observed_exception(
                    type(exc),
                    tx,
                    args=list(exc.args),
                )
            return variables.UserDefinedClassVariable(
                value,
                mutation_type=ValueMutationNew(),
            )
        unimplemented(
            gb_type="namedtuple construction",
            context=f"{args=}, {kwargs=}",
            explanation="`torch.compile` only support certain input types for namedtuple",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class FunctoolsPartialVariable(VariableTracker):
    # [CLASS DOC] Tracker for a partial object: func (callable tracker), args (list
    # of positional trackers), keywords (mapping of name to tracker) and optional
    # original_cache_hash preserved for activation-checkpoint context caching.
    # Calls merge stored and call arguments (call keywords override). Can be
    # reconstructed as a new partial and exposes func/args/keywords as members.
    # partial_type_spec: https://github.com/python/cpython/blob/v3.13.0/Modules/_functoolsmodule.c#L538
    _cpython_type = functools.partial

    _nonvar_fields = {
        "original_cache_hash",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        func: VariableTracker,
        args: list[VariableTracker],
        keywords: dict[str, VariableTracker],
        original_cache_hash: Any = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the wrapped callable, the positional list and the
        # keyword mapping (raises an error if they are not a list / mapping) and the
        # original cache hash.
        super().__init__(**kwargs)
        self.func = func
        if not isinstance(args, list):
            raise AssertionError(f"args must be a list, got {type(args)}")
        self.args = args
        if not isinstance(keywords, dict):
            raise AssertionError(f"keywords must be a dict, got {type(keywords)}")
        self.keywords = keywords
        # Store cache_hash from the original partial for SAC context_fn caching
        self.original_cache_hash = original_cache_hash

    def tp_richcompare_impl(self, tx, other, op):
        # [FUNCTION DOC] Identity-based comparison via the generic object helper.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the partial type.
        return functools.partial

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode creating a new partial from the function,
        # positional args and (via a keyword call) the keyword arguments.
        codegen.add_push_null(lambda: codegen.load_import_from("functools", "partial"))
        codegen(self.func)
        if self.args:
            codegen.foreach(self.args)
        if not self.keywords:
            codegen.extend_output(create_call_function(len(self.args) + 1, False))
            return

        codegen.foreach(self.keywords.values())
        keys = tuple(self.keywords.keys())
        codegen.extend_output(
            codegen.create_call_function_kw(len(keys) + len(self.args) + 1, keys, False)
        )

    def get_function(self) -> Any:
        # [FUNCTION DOC] Returns the real partial object (requires all parts to be
        # python constants).
        return self.as_python_constant()

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the wrapped function with stored positional args
        # followed by call args, and stored keywords overridden by call keywords.
        merged_args = self.args + args
        merged_kwargs = {**self.keywords, **kwargs}
        return self.func.call_function(tx, merged_args, merged_kwargs)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # [FUNCTION DOC] hasattr answered by checking a real dummy partial object;
        # returns a bool tracker.
        # functools.partial uses slots, so attributes are constant
        return VariableTracker.build(tx, hasattr(functools.partial(identity), name))

    # func / args / keywords are read-only members on partial objects.
    # https://github.com/python/cpython/blob/v3.13.0/Modules/_functoolsmodule.c#L295-L299
    def _get_func(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the func member.
        return self.func

    def _get_args(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the args member: a tuple tracker of the
        # stored positional trackers with an attribute source if available.
        source = self.source and AttrSource(self.source, "args")
        return variables.TupleVariable(self.args, source=source)

    def _get_keywords(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for the keywords member: a dict tracker
        # keyed by constant name trackers, with an attribute source if available.
        source = self.source and AttrSource(self.source, "keywords")
        items = {VariableTracker.build(tx, k): v for k, v in self.keywords.items()}
        return variables.ConstDictVariable(items, source=source)

    tp_members = {
        "func": Member(_get_func, None),
        "args": Member(_get_args, None),
        "keywords": Member(_get_keywords, None),
    }

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute access via the parent; a NotImplementedError is
        # converted into an observed AttributeError naming the attribute.
        try:
            return super().tp_getattro_impl(tx, name)
        except NotImplementedError:
            raise_observed_exception(AttributeError, tx, args=[name])

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Builds a real partial object from the python constants of
        # func, args and keywords (raises if any is not constant).
        return functools.partial(
            self.func.as_python_constant(),
            *[arg.as_python_constant() for arg in self.args],
            **{k: v.as_python_constant() for k, v in self.keywords.items()},
        )

    def guard_as_python_constant(self) -> Any:
        # [FUNCTION DOC] Like as_python_constant but converts each part with identity
        # guards, and copies the original cache hash onto the resulting partial.
        # Side effect: installs guards.
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
        result = functools.partial(
            self.func.guard_as_python_constant(),
            *[v.guard_as_python_constant() for v in self.args],
            **{k: v.guard_as_python_constant() for k, v in self.keywords.items()},
        )
        # Preserve cache_hash for SAC context_fn caching
        if self.original_cache_hash is not None:
            result.cache_hash = self.original_cache_hash  # type: ignore[missing-attribute]
        return result


class PolyfilledFunctionVariable(VariableTracker):
    # [CLASS DOC] Tracker for a C/builtin function that has a traceable Python
    # polyfill. Attributes: fn (original), wrapped_fn (handler found for fn) and
    # traceable_fn (the Python implementation traced instead). Calls may be
    # constant-folded, specialized (sum over ints), or inlined through the polyfill.
    _nonvar_fields = {
        "fn",
        "wrapped_fn",
        "traceable_fn",
        *VariableTracker._nonvar_fields,
    }

    @classmethod
    @functools.cache
    def _get_polyfill_handlers(cls) -> dict[Callable[..., Any], types.FunctionType]:
        # [FUNCTION DOC] HELPER: Returns the cached (shared, mutable) registry mapping
        # original functions to polyfill handlers; created empty on first use and
        # populated elsewhere.
        return {}

    @classmethod
    def create_with_source(
        cls, value: Any, source: Source
    ) -> "PolyfilledFunctionVariable":
        # [FUNCTION DOC] Factory: installs a code-identity guard on source and
        # returns a new tracker for value with that source.
        install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))

        return cls(value, source=source)

    def get_value_for_setattr(self) -> object | None:
        # [FUNCTION DOC] Returns the original function.
        return self.fn

    def __init__(self, fn: _F, **kwargs: Any) -> None:
        # [FUNCTION DOC] Resolves the handler for fn from the registry (default fn
        # itself) and finds its traceable implementation from the registered polyfill
        # attribute or the third-party python-implementation attribute. Raises an
        # error if the handler is not callable, the candidate is not callable, or
        # no traceable implementation exists (RuntimeError).
        super().__init__(**kwargs)
        # pyrefly: ignore[invalid-type-var]
        self.fn: _F = fn

        handler = self._get_polyfill_handlers().get(fn, fn)
        traceable_fn = None
        if not callable(handler):
            raise AssertionError(f"Polyfill handler {handler} is not callable for {fn}")
        for candidate_attr in (
            "__torch_dynamo_polyfill__",  # registered polyfill
            "__python_implementation__",  # self handler from third-party libraries
        ):
            candidate = getattr(handler, candidate_attr, None)
            if candidate:
                if not callable(candidate):
                    raise AssertionError(
                        f"Polyfill candidate {candidate} is not callable"
                    )
                traceable_fn = candidate
                break
        else:
            raise RuntimeError(
                f"Polyfill handler {handler} does not have a traceable function"
            )

        self.wrapped_fn = handler
        # pyrefly: ignore[invalid-type-var]
        self.traceable_fn: _F = traceable_fn

    @property
    def polyfill_fn(self) -> Callable[..., Any]:
        # [FUNCTION DOC] Property: the traceable Python implementation.
        return self.traceable_fn

    def can_constant_fold_through(self) -> bool:
        # [FUNCTION DOC] True if the handler is marked as safe to evaluate eagerly
        # when arguments are constants.
        return getattr(
            self.wrapped_fn, "__torch_dynamo_can_constant_fold_through__", False
        )

    def get_function(self) -> Any:
        # [FUNCTION DOC] Returns the original function.
        return self.as_python_constant()

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the polyfilled function. If foldable and all args are
        # constant or unspecialized, runs the original function eagerly and returns
        # a tracker of the result. Special case: builtin sum over a single list or
        # tuple of ints/int symbolic values emits a symbolic sum node in the graph.
        # Otherwise inlines the traceable implementation.
        if self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        ):
            result = (
                self.fn(  # use the original function which is faster than the polyfill
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                )
            )
            return VariableTracker.build(tx, result)

        # Special case for sum on tuple/list of ints
        if (
            self.fn is builtins.sum
            and len(args) == 1
            and not kwargs
            and isinstance(args[0], (variables.ListVariable, variables.TupleVariable))
            and all(
                (x.is_python_constant() and isinstance(x.as_python_constant(), int))
                or (isinstance(x, variables.SymNodeVariable) and x.python_type() is int)
                for x in args[0].items
            )
        ):
            return variables.SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function",
                    torch.sym_sum,
                    (tuple(a.as_proxy() for a in args[0].items),),
                    {},
                ),
                sym_num=torch.sym_sum(
                    [
                        (
                            x.as_python_constant()
                            if x.is_python_constant()
                            else x.sym_num  # type: ignore[attr-defined]
                        )
                        for x in args[0].items
                    ]
                ),
            )

        traceable_function_variable = VariableTracker.build(tx, self.traceable_fn)
        return tx.inline_user_function_return(
            traceable_function_variable,
            list(args),
            dict(kwargs),
        )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method call: the call dunder forwards to call_function;
        # otherwise looks up the named method on the original function (observed
        # TypeError if not found) and calls it as another polyfilled function with
        # an attribute source.
        # KNOWN ISSUE: the lookup check uses "method or is_function(method)", which
        # lets any truthy non-callable through and is likely meant to be a callable
        # check.
        if name == "__call__":
            return self.call_function(tx, args, kwargs)

        method = getattr(self.fn, name, None)
        if not (method or is_function(method)):
            raise_type_error(tx, f"Cannot find callable {name} in {self.fn}")
        options = {}
        if self.source:
            options["source"] = AttrSource(self.source, name)
        polyfilled_method_variable = PolyfilledFunctionVariable(method, **options)
        return polyfilled_method_variable.call_function(tx, args, kwargs)

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns the original function.
        return self.fn


class SysFunctionVariable(VariableTracker):
    # [CLASS DOC] Tracker for the sys exception inspection functions (exc_info and
    # exception); answers from the translator's stack of currently handled
    # exceptions.
    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the sys function being modelled.
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the builtin function type.
        return types.BuiltinFunctionType

    def exc_info(self, tx: "InstructionTranslatorBase") -> "variables.TupleVariable":
        # [FUNCTION DOC] Returns a tuple tracker (type, value, traceback) for the
        # innermost handled exception, or three None constants if none is active.
        if len(tx.exn_vt_stack):
            exn = tx.exn_vt_stack[-1]
            typ = exn.exc_type  # type: ignore[union-attr]
            tb = exn.tp_getattro_impl(tx, "__traceback__")
            items = [VariableTracker.build(tx, typ), exn, tb]
        else:
            items = [
                ConstantVariable.create(None),
                ConstantVariable.create(None),
                ConstantVariable.create(None),
            ]
        return variables.TupleVariable(items)  # type: ignore[arg-type]

    def exception(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Returns the currently handled exception tracker (or None
        # constant), the second item of the exception info tuple.
        return self.exc_info(tx).items[1]

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the modelled function ignoring arguments: exc_info
        # gives the exception info tuple, otherwise the function must be
        # sys.exception (an error is raised if not) and its result is returned.
        if self.value is sys.exc_info:
            return self.exc_info(tx)

        if self.value is not sys.exception:
            raise AssertionError(f"expected sys.exception, got {self.value}")
        return self.exception(tx)


from torch._higher_order_ops.triton_kernel_wrap import (
    create_tma_experimental_metadata,
    create_tma_stable_metadata,
    TMADescriptorMetadata,
    TritonHOPifier,
)


class DynamoTritonHOPifier(TritonHOPifier):
    # [CLASS DOC] Dynamo implementation of the Triton higher-order-op converter
    # hooks: error reporting through graph breaks, callable/value checks on
    # trackers, grid evaluation, wrapping of user objects, config unpacking, grid
    # subscripting and emission of the kernel wrapper call into the FX graph.
    # A single shared instance is used (dynamo_triton_hopifier_singleton).
    def raise_unsupported(self, msg: str) -> Never:
        # [FUNCTION DOC] Raises a graph break stating an unsupported Triton kernel
        # feature with the given message.
        unimplemented(
            gb_type="triton kernel unsupported feature",
            context="",
            explanation=f"Encountered triton kernel unsupported feature: {msg}",
            hints=[],
        )

    def is_callable(self, maybe_callable: VariableTracker) -> bool:
        # [FUNCTION DOC] True if the tracker is a user or nested user function.
        return isinstance(
            maybe_callable, (NestedUserFunctionVariable, UserFunctionVariable)
        )

    def get_value(self, val: VariableTracker) -> Any:
        # [FUNCTION DOC] Returns the real value held by the tracker's value attribute.
        return val.value  # type: ignore[attr-defined]

    def check_grid(self, grid: "BaseListVariable") -> tuple[torch.fx.proxy.Proxy, ...]:
        # [FUNCTION DOC] Converts a list-like grid tracker into a tuple of graph
        # proxies; graph-breaks if the grid is not list-like.
        from .lists import BaseListVariable

        if isinstance(grid, BaseListVariable):
            return grid.as_proxy()
        else:
            unimplemented(
                gb_type="unsupported grid type for triton hop check_grid",
                context=f"grid type = {type(grid)}",
                explanation="`torch.compile` only supports list-like grid for check_grid",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

    def call_grid(
        self, grid: Any, meta: dict[str, Any], tx: "InstructionTranslatorBase"
    ) -> Any:
        # [FUNCTION DOC] Evaluates a callable grid tracker by calling it with a
        # dict tracker built from the meta-parameter mapping; returns the result.
        meta_var = {VariableTracker.build(tx, k): v for k, v in meta.items()}
        grid = grid.call_function(tx, [meta_var], {})
        return grid

    # We use this function to wrap call_prune_configs
    def call_user_defined_fn(
        self,
        user_fn: Callable[..., Any],
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
        tx: Optional["InstructionTranslatorBase"],
        variable: Any,
    ) -> VariableTracker:
        # [FUNCTION DOC] Wraps a user function (e.g. a prune-configs hook) in a
        # sourceless tracker and calls it with the given argument trackers,
        # returning the result tracker.
        from .builder import SourcelessBuilder

        wrapped_user_function = SourcelessBuilder.create(tx, user_fn)  # type: ignore[arg-type]
        result = wrapped_user_function.call_function(tx, args, kwargs)
        return result

    def wrap_user_defined_obj(
        self,
        user_obj: Any,
        tx: Optional["InstructionTranslatorBase"],
        variable: Any,
        name: str,
    ) -> VariableTracker:
        # [FUNCTION DOC] Wraps a user object attached to the kernel as a tracker with
        # a source derived from the kernel source and the attribute name, going
        # through the variable builder so already-tracked objects are reused.
        # Raises an error if tx is None.
        from .builder import VariableBuilder

        if tx is None:
            raise AssertionError("tx must not be None")
        # Route through VariableBuilder.__call__ so already-tracked mutable
        # objects (for example autotuner config lists) are reused instead of
        # being registered for mutation twice in the same trace.
        wrapped_user_obj = VariableBuilder(
            tx, AttrSource(variable.kernel_source, f"{name}")
        )(user_obj)
        return wrapped_user_obj

    def maybe_unpack_configs(
        self, configs: Any, tx: Optional["InstructionTranslatorBase"]
    ) -> list[Any]:
        # [FUNCTION DOC] Unpacks an iterable tracker of autotune configs into a list
        # of real config objects, installing guards so changes to them are detected.
        # Raises an error if tx is None.
        # unpack the list of configs
        if tx is None:
            raise AssertionError("tx must not be None")
        configs = unpack_iterable(tx, configs)

        # guard_as_python_constant inserts guards for Dynamo to check if the configs object changed.
        configs = [config.guard_as_python_constant() for config in configs]

        return configs

    def maybe_unpack_heuristic_result(self, result: VariableTracker) -> Any:
        # [FUNCTION DOC] Returns the guarded constant value of a heuristic result
        # tracker; reports an unsupported feature (graph break) if it is not constant.
        if not result.is_python_constant():
            self.raise_unsupported(
                "@triton.heuristics must return constant values because configs can only contain constant values."
            )

        return result.guard_as_python_constant()

    # We need to override call_getitem here so that we can add the source in the case
    # where we call the triton kernel with a grid
    def call_getitem(  # type: ignore[override]
        self,
        variable: "TritonKernelVariable",
        args: Sequence[Any],
    ) -> "TritonKernelVariable":
        # [FUNCTION DOC] Implements kernel subscripting with a grid: requires that no
        # grid is set yet and exactly one argument (else unsupported/graph break),
        # and returns a new kernel tracker of the same type with that grid and the
        # same kernel, index and kernel source.
        # __getitem__ should only be called if we don't already have a grid
        # Only grid needs to be passed
        if variable.grid is not None or len(args) != 1:
            self.raise_unsupported(
                "Triton kernels should be called with only a single grid"
            )
        return type(variable)(
            kernel=variable.kernel,
            kernel_idx=variable.kernel_idx,
            grid=args[0],
            kernel_source=variable.kernel_source,
        )

    def call_HOP(
        self,
        variable: "TritonKernelVariable",
        grids: Any,
        combined_args: dict[str, Any],
        launch_kwargs: tuple[str, ...],
        kernel_arg_names: set[str],
        tx: "InstructionTranslatorBase",
    ) -> ConstantVariable | None:
        # [FUNCTION DOC] Emits the triton kernel wrapper call into the FX graph.
        # TMA descriptor trackers in the arguments are replaced by their tensors with
        # descriptor metadata passed separately. Constant arguments are stored in the
        # kernel side table; non-constant ones must be tensors or symbolic ints
        # (else unsupported). Launch keyword arguments that are not kernel
        # parameters must be constants (compiler options). Creates a graph node for
        # the wrapper with kernel index, constant args index, grid, TMA metadata,
        # kwargs and launch kwargs. Returns a None constant tracker.
        # Side effects: mutates combined_args, the kernel side table and the graph.
        from .dicts import ConstDictVariable

        # as we can only pass tensors as non-const args in fx graph,
        # here we replace TMA descriptors
        # (TMADescriptorExperimentalVariable and TMADescriptorStableVariable
        # instances) with the underlying tensors, while moving the
        # TMA descriptor-related metadata to a separate argument,
        # so that we can reconstruct the TMA descriptors downstream
        tma_descriptor_metadata: TMADescriptorMetadata = {}
        for k in list(combined_args.keys()):
            v = combined_args[k]
            if isinstance(
                v, (TMADescriptorExperimentalVariable, TMADescriptorStableVariable)
            ):
                tma_descriptor_metadata[k] = v.to_metadata()
                combined_args[k] = v.get_tensor()

        combined_args_vt = {
            VariableTracker.build(tx, k): v for k, v in combined_args.items()
        }

        from torch._higher_order_ops.triton_kernel_wrap import (
            kernel_side_table,
            triton_kernel_wrapper_mutation,
        )

        # Combine args and kwargs and pass as a dict so that if user defined triton
        # kernel uses variables as 'grid' or 'kernel', it does not conflict with
        # parameters of the wrapper function
        constant_args = {
            k: v.as_python_constant()
            for k, v in combined_args.items()
            if isinstance(v, VariableTracker) and v.is_python_constant()
        }
        non_constant_args = {
            k: v
            for k, v in combined_args_vt.items()
            if not (isinstance(v, VariableTracker) and v.is_python_constant())
        }
        # launch_kwargs records the names passed as kwargs at the Triton launch
        # site. A non-kernel launch kwarg can only be a compiler option, so it
        # must be a Python constant before entering the graph. Kernel launch
        # kwargs may also be compiler options, but that target-specific check
        # happens in Inductor after the triton backend is determined and
        # backend.parse_options() is called.
        non_const_options: list[str] = []
        for k in launch_kwargs:
            if k in kernel_arg_names:
                continue
            v = combined_args[k]
            if not (isinstance(v, VariableTracker) and v.is_python_constant()):
                non_const_options.append(k)
        if non_const_options:
            self.raise_unsupported(
                "Triton backend options must be Python constants: "
                f"{sorted(non_const_options)!r}."
            )

        for v in non_constant_args.values():
            v = v.realize()
            if not (v.is_tensor() or v.is_symnode_like()):
                self.raise_unsupported(
                    f"Unexpected argument type for a Triton kernel: {repr(v)}."
                )

        constant_args_idx = kernel_side_table.add_constant_args(constant_args)
        meta = ConstDictVariable(non_constant_args)
        tx.output.create_proxy(
            "call_function",
            triton_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": variable.kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": grids,
                "tma_descriptor_metadata": tma_descriptor_metadata,
                "kwargs": meta.as_proxy(),
                "launch_kwargs": launch_kwargs,
            },
        )

        return VariableTracker.build(
            tx,
            None,
        )


dynamo_triton_hopifier_singleton = DynamoTritonHOPifier()


class TritonKernelVariable(VariableTracker):
    # [CLASS DOC] Tracker for a user-defined Triton kernel (optionally with a launch
    # grid). Attributes: kernel, kernel_idx (index in the kernel side table), grid
    # and kernel_source. Calling it emits a kernel wrapper node via the shared
    # hopifier; subscripting with a grid returns a new tracker.
    grid: "TritonGridType"
    kernel: "TritonKernelType"
    kernel_idx: int | None
    kernel_source: Source | None

    def __init__(
        self, kernel: Any, kernel_idx: int | None, grid: Any, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Takes the kernel source from kwargs (defaulting to source)
        # and lets the shared hopifier initialize the kernel, index and grid fields.
        self.kernel_source = kwargs.pop("kernel_source", kwargs.get("source"))
        super().__init__(**kwargs)
        dynamo_triton_hopifier_singleton.init_variable(self, kernel, kernel_idx, grid)

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the type of the real kernel object.
        return type(self.kernel)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Launches the kernel through the shared hopifier, which
        # validates the call and emits the wrapper node; returns its result tracker.
        return dynamo_triton_hopifier_singleton.call_triton_kernel(  # type: ignore[return-value]
            self, args, kwargs, tx
        )

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Subscript with a grid: returns a new kernel tracker with
        # that grid (see the hopifier's subscript hook).
        # Triton kernel[grid] — triton-specific, not a CPython slot.
        return dynamo_triton_hopifier_singleton.call_getitem(self, [key])

    def run(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the kernel run method through the shared
        # hopifier and returns its result tracker.
        return dynamo_triton_hopifier_singleton.call_run(self, args, kwargs, tx)  # type: ignore[return-value]

    def specialize_symbolic(self, arg: Any) -> Any:
        # [FUNCTION DOC] Replaces a symbolic number tracker by a constant tracker of
        # its evaluated value (specializing, which may add guards); other values are
        # returned unchanged.
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, SymNodeVariable):
            return ConstantVariable.create(arg.evaluate_expr())
        return arg

    tp_methods = {"run": Method(run)}


class TMADescriptorExperimentalVariable(VariableTracker):
    # [CLASS DOC] Tracker for an experimental-API TMA descriptor created from a
    # data pointer: holds data_ptr (DataPtrVariable), dims, block_dims and
    # element_size trackers. Provides metadata for graph emission and bytecode
    # reconstruction through the triton experimental descriptor helpers.
    def __init__(
        self,
        data_ptr: "variables.DataPtrVariable",
        dims: list[VariableTracker],
        block_dims: list[VariableTracker],
        element_size: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the descriptor components; raises an error if
        # data_ptr is not a DataPtrVariable.
        if not isinstance(data_ptr, variables.DataPtrVariable):
            raise AssertionError(
                f"data_ptr must be a DataPtrVariable, got {type(data_ptr)}"
            )
        super().__init__(**kwargs)
        self.data_ptr = data_ptr
        self.dims = dims
        self.block_dims = block_dims
        self.element_size = element_size

    def to_metadata(self) -> Any:
        # [FUNCTION DOC] Builds the experimental TMA descriptor metadata from the
        # proxies of dims, block dims and element size.
        return create_tma_experimental_metadata(
            [dim.as_proxy() for dim in self.dims],
            [dim.as_proxy() for dim in self.block_dims],
            self.element_size.as_proxy(),
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the triton create-Nd-descriptor helper (N is
        # the number of dims) with the data pointer, dims, block dims and element size.
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                "triton.tools.experimental_descriptor",
                f"create_{len(self.dims)}d_tma_descriptor",
            )
        )
        self.data_ptr.reconstruct(codegen)
        args = [*self.dims, *self.block_dims, self.element_size]
        codegen.foreach(args)
        codegen.call_function(len(args) + 1, False)

    def get_tensor(self) -> VariableTracker:
        # [FUNCTION DOC] Returns the tensor tracker the data pointer came from.
        return self.data_ptr.from_tensor


class TMADescriptorStableVariable(VariableTracker):
    # [CLASS DOC] Tracker for a stable-API TMA tensor descriptor: holds the tensor
    # tracker and the block shape list tracker. Provides metadata for graph
    # emission and reconstruction via the tensor descriptor from-tensor factory.
    def __init__(
        self,
        tensor: "TensorVariable",
        block_shape: "ListVariable",
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the tensor and block shape; raises an error if the
        # tensor argument is not a tensor tracker.
        if not tensor.is_tensor():
            raise AssertionError("tensor argument must be a tensor")
        super().__init__(**kwargs)
        self.tensor = tensor
        self.block_shape = block_shape

    def to_metadata(self) -> Any:
        # [FUNCTION DOC] Builds the stable TMA descriptor metadata from the block
        # shape proxy.
        return create_tma_stable_metadata(
            self.block_shape.as_proxy(),
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the tensor descriptor from-tensor factory
        # with the tensor and block shape.
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                "triton.tools.tensor_descriptor",
                "TensorDescriptor",
            )
        )
        codegen.load_method("from_tensor")
        self.tensor.reconstruct(codegen)
        codegen(self.block_shape)
        codegen.call_method(2)

    def get_tensor(self) -> Any:
        # [FUNCTION DOC] Returns the tensor tracker.
        return self.tensor


class CreateTMADescriptorExperimentalVariable(VariableTracker):
    # [CLASS DOC] Tracker for the experimental create-1d/2d-TMA-descriptor function
    # (rank 1 or 2). Calls produce a TMADescriptorExperimentalVariable.
    def __init__(
        self,
        rank: int,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the descriptor rank; raises an error unless it is 1 or 2.
        if rank not in (1, 2):
            raise AssertionError(f"rank must be 1 or 2, got {rank}")
        super().__init__(**kwargs)
        self.rank = rank

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the function type.
        return types.FunctionType

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Creates a descriptor tracker from the call. The pointer
        # argument (keyword or first positional) must be a data pointer tracker
        # (else graph break). Requires exactly 4 total arguments for rank 1 or 6 for
        # rank 2 (observed TypeError otherwise); dims, block dims and element size
        # are read by keyword name or position (the last positional for element
        # size). Rank 2 dims are ordered dim1 then dim0.
        ptr = kwargs["ptr"] if "ptr" in kwargs else args[0]

        if not isinstance(ptr, variables.DataPtrVariable):
            unimplemented(
                gb_type="invalid ptr argument for create_tma_descriptor",
                context=f"args = {args}, kwargs = {kwargs}",
                explanation=f"Expected `ptr` argument of `create_{self.rank}d_tma_descriptor`"
                "to be from a `.data_ptr()` call, represented internally by `DataPtrVariable`",
                hints=[
                    "`torch.compile` may fail to internally represent result of `.data_ptr()` "
                    "with `DataPtrVariable` due to a graph break between the `.data_ptr()` call and "
                    f"`create_{self.rank}d_tma_descriptor`. Please ensure there were no graph breaks "
                    "between these two calls.",
                ],
            )

        if self.rank == 1:
            if len(args) + len(kwargs) != 4:
                raise_type_error(
                    tx,
                    f"TMA metadata rank=1 requires exactly 4 arguments, got {len(args) + len(kwargs)}",
                )
            dims = [
                kwargs["dim"] if "dim" in kwargs else args[1],
            ]
            block_dims = [
                kwargs["block_dim"] if "block_dim" in kwargs else args[2],
            ]
        else:
            if len(args) + len(kwargs) != 6:
                raise_type_error(
                    tx,
                    f"TMA metadata rank=2 requires exactly 6 arguments, got {len(args) + len(kwargs)}",
                )
            dims = [
                kwargs["dim1"] if "dim1" in kwargs else args[1],
                kwargs["dim0"] if "dim0" in kwargs else args[2],
            ]
            block_dims = [
                kwargs["block_dim1"] if "block_dim1" in kwargs else args[3],
                kwargs["block_dim0"] if "block_dim0" in kwargs else args[4],
            ]
        element_size = kwargs["element_size"] if "element_size" in kwargs else args[-1]

        # to make pyrefy happy
        if not isinstance(ptr, variables.DataPtrVariable):
            raise AssertionError(f"ptr must be a DataPtrVariable, got {type(ptr)}")

        return TMADescriptorExperimentalVariable(
            data_ptr=ptr,
            dims=dims,
            block_dims=block_dims,
            element_size=element_size,
        )


class CreateTMADescriptorStableVariable(VariableTracker):
    # [CLASS DOC] Tracker for the stable tensor-descriptor factory function; calls
    # produce a TMADescriptorStableVariable.
    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the function type.
        return types.FunctionType

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Reads the tensor and block shape (by keyword or position)
        # and returns a stable TMA descriptor tracker. No argument validation.
        tensor = kwargs["tensor"] if "tensor" in kwargs else args[0]
        block_shape = kwargs["block_shape"] if "block_shape" in kwargs else args[1]

        return TMADescriptorStableVariable(
            tensor=tensor,  # type: ignore[arg-type]
            block_shape=block_shape,  # type: ignore[arg-type]
        )


class PyTreeGetNodeTypeFunctionVariable(UserFunctionVariable):
    # [CLASS DOC] Fast-path tracker for the pytree node-type lookup function: answers
    # directly with the argument's type (or the namedtuple factory for namedtuple
    # classes) instead of tracing the function body, installing type guards.
    """
    `torch.utils._pytree._get_node_type` function is very hot function. We want to special case it to reduce Dynamo tracing time.

    def _get_node_type(tree: Any) -> Any:
        node_type = type(tree)
        # All namedtuple types are implicitly registered as pytree nodes.
        # XXX: Other parts of the codebase expect namedtuple types always return
        #      `namedtuple` instead of the actual namedtuple type. Even if the type
        #      is explicitly registered.
        if is_namedtuple_class(node_type):
            return collections.namedtuple
        return node_type
    """

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Requires exactly one argument (observed TypeError
        # otherwise). If the argument has a source, installs a type-match guard. For
        # namedtuple classes returns the namedtuple factory (guarded by identity of
        # the collections module); otherwise returns a tracker of the argument's
        # Python type with a type source when available.
        if len(args) != 1:
            raise_type_error(
                tx, f"pytree_get_node_type requires exactly 1 argument, got {len(args)}"
            )
        type_source = None
        if args[0].source:
            install_guard(args[0].source.make_guard(GuardBuilder.TYPE_MATCH))
            type_source = TypeSource(args[0].source)
        python_type = args[0].python_type()
        if is_namedtuple_class(python_type):
            collections_source = ImportSource("collections")
            install_guard(collections_source.make_guard(GuardBuilder.ID_MATCH))
            type_source = AttrSource(collections_source, "namedtuple")
            return VariableTracker.build(
                tx, vars(collections)["namedtuple"], type_source
            )
        return VariableTracker.build(tx, python_type, source=type_source)


class PyTreeTreeIsLeafFunctionVariable(UserFunctionVariable):
    # [CLASS DOC] Fast-path tracker for the pytree is-leaf predicate: when no custom
    # is-leaf function is given, computes whether the node type is absent from the
    # supported-nodes registry without tracing the function body.
    """
    `torch.utils._pytree.tree_is_leaf` function is a hot function. We want to special case it to reduce Dynamo tracing time.

    def tree_is_leaf(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> bool:
        if is_leaf is not None and is_leaf(tree):
            return True
        return _get_node_type(tree) not in SUPPORTED_NODES

    When is_leaf is None (the common case), we can optimize by not tracing into the function.
    When is_leaf is not None, we fall back to regular tracing since it requires executing user code.
    """

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Accepts 1 or 2 positional arguments (observed TypeError
        # otherwise); the leaf predicate may come from keyword or second argument.
        # If a non-None predicate is given, falls back to normal tracing. Otherwise
        # gets the node type via the node-type fast path, checks membership in the
        # pytree supported-nodes registry (using its tracked source so mutations are
        # seen, with an identity guard on the torch import), and returns the negated
        # result as a bool tracker.
        # tree_is_leaf(tree, is_leaf=None)
        if len(args) < 1 or len(args) > 2:
            raise_type_error(
                tx, f"tree_is_leaf requires 1 or 2 arguments, got {len(args)}"
            )

        # Check if is_leaf parameter is provided
        is_leaf = kwargs.get("is_leaf", ConstantVariable.create(None))
        if len(args) == 2:
            is_leaf = args[1]

        if not is_leaf.is_constant_none():
            return super().call_function(tx, args, kwargs)

        # Optimize the case where is_leaf is None
        # return _get_node_type(tree) not in SUPPORTED_NODES
        tree = args[0]
        node_type_var = PyTreeGetNodeTypeFunctionVariable(
            torch.utils._pytree._get_node_type
        ).call_function(tx, [tree], {})

        # If the SUPPORTED_NODES was seen earlier and mutated, there would be a
        # source and that will give us the mutated SUPPORTED_NODES.
        # get_pytree_SUPPORTED_NODES_source is cached, so install its
        # ImportSource("torch") ID_MATCH guard here.
        install_guard(ImportSource("torch").make_guard(GuardBuilder.ID_MATCH))
        supported_nodes_var = VariableTracker.build(
            tx,
            torch.utils._pytree.SUPPORTED_NODES,
            source=get_pytree_SUPPORTED_NODES_source(),
        )
        out = supported_nodes_var.call_method(tx, "__contains__", [node_type_var], {})
        return VariableTracker.build(tx, not out.value)


class SparseTensorCreationSkipVariable(SkipFunctionVariable):
    # [CLASS DOC] Skip tracker for sparse tensor factory functions; always
    # graph-breaks with a sparse-specific message.
    """
    Skip variable for sparse tensor factory functions with clear messaging regarding lack of support.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Initializes the skip tracker with a fixed reason that sparse
        # tensor creation is unsupported.
        reason = "sparse tensor creation is not supported in torch.compile"
        super().__init__(value, reason=reason, **kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Always graph-breaks, reporting that sparse tensor creation
        # via the named function is unsupported, with sparse-tensor hints.
        from .. import graph_break_hints

        fn_name = getattr(self.value, "__name__", str(self.value))
        unimplemented(
            gb_type="Sparse tensor creation not supported",
            context=f"function: {fn_name}",
            explanation=(
                f"torch.compile does not support sparse tensor creation functions like {fn_name}. "
                "Sparse tensors require specialized handling that is not yet implemented in the compiler."
            ),
            hints=[*graph_break_hints.SPARSE_TENSOR],
        )


def emit_noargs_leaf_function_to_graph(
    tx: "InstructionTranslatorBase",
    real_impl: Callable[[], None],
    name: str,
) -> None:
    # [FUNCTION DOC] Adds to the graph an ordered leaf-function invocation node that
    # runs real_impl (no tensor inputs or outputs) at runtime. Wraps real_impl and a
    # no-op fake implementation into leaf callables, registers them and an empty
    # input spec as static graph attributes named after name (with type recorded on
    # the proxy nodes), and creates the call node. Side effects: registers static
    # attributes and creates a graph node.
    """Emit an invoke_leaf_function node for a side-effectful function with no
    tensor inputs or outputs.

    The function is captured as a closure inside _LeafCallable objects and
    registered as a static attribute on the graph module.  Because
    invoke_leaf_function is registered as EffectType.ORDERED, effect tokens
    prevent DCE and maintain execution ordering relative to other ops.

    Use this when Dynamo needs to preserve a pure-side-effect call (like
    setting global runtime state) in the compiled graph so that it replays
    at the correct position at runtime.
    """
    import torch.utils._pytree as pytree
    from torch._higher_order_ops.invoke_leaf_function import (
        _LeafCallable,
        invoke_leaf_function,
        make_leaf_function_wrappers,
    )

    def fake_impl():
        # [FUNCTION DOC] HELPER: No-op fake implementation returning None.
        return None

    captured_out_spec: list[pytree.TreeSpec | None] = [None]
    wrapped_real, wrapped_fake = make_leaf_function_wrappers(
        real_impl, fake_impl, captured_out_spec
    )

    real_callable = _LeafCallable(wrapped_real)
    fake_callable = _LeafCallable(wrapped_fake)
    input_spec = pytree.tree_flatten(((), {}))[1]

    def make_proxy(attr_name: str, val: Any) -> Any:
        # [FUNCTION DOC] HELPER: Registers val as a static graph attribute under
        # attr_name, records its type on the proxy node and returns the proxy.
        proxy = tx.output.register_static_attr_and_return_proxy(attr_name, val)
        proxy.node.type = type(val)
        return proxy

    invoke_args = (
        make_proxy(f"{name}_real_fn", real_callable),
        make_proxy(f"{name}_fake_fn", fake_callable),
        make_proxy(f"{name}_input_spec", input_spec),
        "",  # mutated_flat_indices
    )
    tx.output.create_proxy("call_function", invoke_leaf_function, invoke_args, {})


class TritonSetAllocatorVariable(VariableTracker):
    # [CLASS DOC] Tracker for the Triton set-allocator function; calls are emitted as
    # ordered graph nodes so the allocator is set at the right point at runtime.
    """Trace triton.set_allocator as an invoke_leaf_function node in the
    graph so that it executes at the right point at runtime, ordered by
    effect tokens."""

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the real set-allocator function.
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the type of the real function.
        return type(self.value)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Requires exactly one positional argument and no keywords
        # (raises an error otherwise); the allocator must be a python constant. Emits
        # a graph node that calls the real set-allocator with it at runtime and
        # returns a None constant tracker.
        if len(args) != 1:
            raise AssertionError(f"expected exactly 1 arg, got {len(args)}")
        if kwargs:
            raise AssertionError("unexpected kwargs")
        alloc_fn = args[0].as_python_constant()

        # Emit an invoke_leaf_function node so it runs at runtime.
        set_allocator = self.value

        def real_impl():
            # [FUNCTION DOC] HELPER: Runtime body: calls the real set-allocator with
            # the captured allocator and returns None.
            set_allocator(alloc_fn)
            return None

        emit_noargs_leaf_function_to_graph(tx, real_impl, "set_alloc")

        return ConstantVariable.create(None)


# ---------------------------------------------------------------------------
# CPython descriptor VTs
#
# Each class mirrors a CPython descriptor type (PyWrapperDescr_Type,
# PyMethodDescr_Type, etc.) and implements tp_descr_get_impl to model
# the descriptor binding step faithfully.
# ---------------------------------------------------------------------------


def _check_descriptor_obj_type(
    tx: "InstructionTranslatorBase",
    descriptor: types.MethodDescriptorType
    | types.WrapperDescriptorType
    | types.MemberDescriptorType
    | types.GetSetDescriptorType,
    obj: "VariableTracker",
) -> None:
    # [FUNCTION DOC] HELPER: Validates that obj's Python type is a subtype of the
    # descriptor's owning class. Returns silently for None or an object with unknown
    # type; otherwise raises an observed TypeError describing the mismatch.
    """Check that obj's type is compatible with descriptor.__objclass__.

    Mirrors CPython's descr_check which raises TypeError when a C descriptor
    is bound to an object whose type is not a subtype of the descriptor's
    __objclass__.

    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L79-L96
    """
    if obj is None:
        return
    try:
        obj_type = obj.python_type()
    except NotImplementedError:
        return
    if not issubclass(obj_type, descriptor.__objclass__):
        raise_type_error(
            tx,
            f"descriptor '{descriptor.__name__}' for "
            f"'{descriptor.__objclass__.__name__}' objects "
            f"doesn't apply to a '{obj_type.__name__}' object",
        )


# descr_members: __objclass__ and __name__ are PyMemberDef on all descriptor
# types. https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L641-L645
class WrapperDescriptorVariable(VariableTracker):
    # [CLASS DOC] Models an unbound slot-wrapper descriptor (for example the add
    # method of a built-in type). Holds the real descriptor and an owner class
    # tracker. Direct calls dispatch through the owner; binding to an instance yields
    # a bound method-wrapper tracker.
    """Unbound C slot wrapper (wrapper_descriptor on a type).

    CPython types define behavior through C-level slots on PyTypeObject
    (tp_richcompare, sq_length, nb_add, etc.).  When these slots are
    accessed from Python (e.g. list.__add__), CPython exposes them as
    wrapper_descriptor objects (PyWrapperDescr_Type).  A wrapper_descriptor
    is an unbound descriptor living on the type -- it is not tied to any
    instance.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L867

    When a wrapper_descriptor is accessed on an instance (e.g. [1,2].__add__),
    its tp_descr_get slot (wrapperdescr_get) is invoked, which calls
    PyWrapper_New to produce a bound method-wrapper (_PyMethodWrapper_Type).
    The tp_descr_get_impl method on this class mirrors that binding step.
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.WrapperDescriptorType,
        owner: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real descriptor and the owner class tracker.
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.owner = owner

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the owning class and descriptor name.
        cls_name = self.descriptor.__objclass__.__name__
        return f"WrapperDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the wrapper-descriptor type.
        return types.WrapperDescriptorType

    def as_python_constant(self) -> types.WrapperDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    def get_real_python_backed_value(self) -> types.WrapperDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Unbound call: the first argument is the instance. Raises an
        # observed TypeError when no arguments are given or the instance type does not
        # match the owning class; otherwise dispatches the method by name through the
        # owner class tracker (not the instance's type) with the instance prepended.
        # Unbound call: list.__add__([1,2], [3,4]) -- first arg is self.
        # Mirrors wrapperdescr_call which invokes the C slot directly.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L535
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Dispatch through the owner (UDCV for the defining class) rather
        # than obj.call_method, which would do MRO resolution from type(obj)
        # and find Python overrides on subclasses. Routing through the class
        # mirrors CPython's wrapperdescr_call which invokes the C slot directly.
        return self.owner.call_method(
            tx, self.descriptor.__name__, [obj, *rest], kwargs
        )

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> "MethodWrapperVariable":
        # [FUNCTION DOC] Descriptor get: checks the instance type (observed TypeError
        # on mismatch) and returns a bound method-wrapper tracker for the instance,
        # keeping this tracker's source.
        # Mirrors wrapperdescr_get which calls PyWrapper_New to produce
        # a bound method-wrapper.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L203-L213
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1489-L1505
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        return MethodWrapperVariable(self.descriptor, obj, source=self.source)


class MethodWrapperVariable(VariableTracker):
    # [CLASS DOC] Models a slot-wrapper descriptor bound to an instance (a
    # method-wrapper). Holds the real descriptor and the bound object tracker; calls
    # dispatch the named method on the object (tensor attribute getters are handled
    # specially). Constant only if the object is constant.
    """Bound method-wrapper (wrapper_descriptor bound to an instance).

    Produced by WrapperDescriptorVariable.tp_descr_get_impl, mirroring
    PyWrapper_New which stores a reference to the descriptor and the instance.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1450
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.WrapperDescriptorType,
        obj: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real descriptor and the bound object tracker.
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.obj = obj

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the owning class, descriptor name and object.
        cls_name = self.descriptor.__objclass__.__name__
        return (
            f"MethodWrapperVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"
        )

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the method-wrapper type.
        return types.MethodWrapperType

    # A method-wrapper's own dunders come from the descriptor / bound obj and do
    # not depend on obj's contents. Resolving them via these tables (consulted
    # before the generic getset-descriptor path) avoids forcing self.obj to a
    # python constant, which would break e.g. a list holding non-constant items.
    tp_getset = {
        "__name__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__name__)
        ),
        "__qualname__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__qualname__)
        ),
    }
    tp_members = {
        "__self__": Member(lambda s, tx: s.obj),
    }

    def get_real_python_backed_value(self) -> types.MethodWrapperType:
        # [FUNCTION DOC] Returns the real method-wrapper (requires a constant object).
        return self.as_python_constant()

    def is_python_constant(self) -> bool:
        # [FUNCTION DOC] True exactly when the bound object is a python constant.
        return self.obj.is_python_constant()

    def as_python_constant(self) -> types.MethodWrapperType:
        # [FUNCTION DOC] Binds the real descriptor to the object's constant value and
        # returns the method-wrapper; raises the as-constant-not-implemented error if
        # binding fails with a TypeError.
        obj_value = self.obj.as_python_constant()
        try:
            return self.descriptor.__get__(obj_value, type(obj_value))
        except TypeError:
            raise AsPythonConstantNotImplementedError(self) from None

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the bound slot method. If the first argument is a
        # tensor tracker and the wrapper is a tensor base attribute getter, requires
        # exactly one argument (observed TypeError otherwise) and returns the tensor's
        # attribute of that name. In all other cases calls the named method on the
        # bound object with the given arguments.
        # Only materialize the wrapper (which forces self.obj's python constant)
        # for the tensor-getter special case; the generic path below dispatches
        # via call_method and must not require a constant self.obj (e.g. a set /
        # frozenset whose as_python_constant() re-hashes its keys).
        if args and isinstance(args[0], variables.TensorVariable):
            try:
                method_wrapper = self.as_python_constant()
            except NotImplementedError:
                method_wrapper = None
            if method_wrapper is not None and is_tensor_base_attr_getter(
                method_wrapper
            ):
                if not (len(args) == 1 and len(kwargs) == 0):
                    raise_type_error(
                        tx, "tensor attribute getter takes exactly one argument"
                    )
                # Avoid the generic descriptor path's implicit owner lookup, which
                # would read __class__ on tensor subclasses during __torch_function__.
                descriptor = cast(Any, method_wrapper.__self__)
                return args[0].tp_getattro_impl(tx, descriptor.__name__)

        return self.obj.call_method(tx, self.descriptor.__name__, list(args), kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode loading the object and then its attribute of
        # the descriptor's name.
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.descriptor.__name__))

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash as (value, flag): the hash of the real method-wrapper
        # with flag False when constant; otherwise defers to the parent.
        try:
            # CPython wrapper_hash:
            # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1347
            return hash(self.as_python_constant()), False
        except NotImplementedError:
            return super().hash_impl(tx)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Comparison by the real constant values via the generic helper.
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)


class MethodDescriptorVariable(VariableTracker):
    # [CLASS DOC] Models an unbound C method descriptor (for example the append
    # method of the list type). Holds the real descriptor and an owner class tracker.
    # Direct calls dispatch through the owner; binding to an instance yields a bound
    # builtin method tracker.
    """Unbound C method descriptor (method_descriptor on a type).

    CPython types expose their PyMethodDef-based C methods as
    method_descriptor objects (PyMethodDescr_Type) in the type's tp_dict.
    For example, list.append and dict.get are method_descriptors.  Like
    wrapper_descriptors, these are unbound descriptors living on the type.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L716

    When a method_descriptor is accessed on an instance (e.g. [].append),
    its tp_descr_get slot (method_get) is invoked, which calls
    PyCFunction_NewEx to produce a bound builtin_function_or_method
    (PyCFunction_Type).  The tp_descr_get_impl method mirrors that step.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType,
        owner: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real descriptor and the owner class tracker.
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.owner = owner

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the owning class and descriptor name.
        cls_name = self.descriptor.__objclass__.__name__
        return f"MethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the method-descriptor type.
        return types.MethodDescriptorType

    def as_python_constant(self) -> types.MethodDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    def get_real_python_backed_value(self) -> types.MethodDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Unbound call: the first argument is the instance. Raises an
        # observed TypeError when no arguments are given or the instance type does not
        # match the owning class; otherwise dispatches the method by name through the
        # owner class tracker with the instance prepended.
        # Mirrors methoddescr_call which invokes the C method directly.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L427
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        name = self.descriptor.__name__
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Dispatch through the owner (UDCV for the defining class) rather
        # than obj.call_method, which would do MRO resolution from type(obj)
        # and find Python overrides on subclasses.
        return self.owner.call_method(tx, name, [obj, *rest], kwargs)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> "BoundBuiltinMethodVariable":
        # [FUNCTION DOC] Descriptor get: checks the instance type (observed TypeError
        # on mismatch) and returns a bound builtin method tracker for the instance,
        # keeping this tracker's source.
        # Mirrors method_get which calls PyCFunction_NewEx to produce a
        # bound builtin_function_or_method.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L40
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        return BoundBuiltinMethodVariable(self.descriptor, obj, source=self.source)


class BoundBuiltinMethodVariable(VariableTracker):
    # [CLASS DOC] Models a built-in method bound to an object (or to a class for
    # class methods). Holds the backing descriptor and the bound object tracker; calls
    # dispatch the named method on the object.
    """Bound builtin_function_or_method (PyCFunction_Type).

    Produced by MethodDescriptorVariable.tp_descr_get_impl (binding a
    method_descriptor to an instance, e.g. [].append) or created by the
    builder for bound C methods (e.g. frozenset().__contains__,
    tuple.__new__).  The backing descriptor can be a MethodDescriptorType
    or a BuiltinFunctionType (for methods stored directly in type dicts).
    https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L331
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType
        | types.BuiltinFunctionType
        | types.ClassMethodDescriptorType,
        obj: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the backing descriptor and the bound object tracker.
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.obj = obj

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the owning class (or a placeholder),
        # descriptor name and bound object.
        cls_name = getattr(
            getattr(self.descriptor, "__objclass__", None), "__name__", "?"
        )
        return f"BoundBuiltinMethodVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the builtin method type.
        return types.BuiltinMethodType

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash as (value, flag): the hash of the real bound method with
        # flag False when constant; otherwise the identity of this tracker with flag
        # True.
        # meth_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/methodobject.c#L319
        try:
            return hash(self.as_python_constant()), False
        except AsPythonConstantNotImplementedError:
            return id(self), True

    def tp_richcompare_impl(self, tx, other, op):
        # [FUNCTION DOC] Identity-based comparison via the generic object helper.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Builds the real bound method from the object's constant
        # value: class method descriptors bind to the class; other descriptors with a
        # get method bind to the object; otherwise reads the named attribute.
        obj = self.obj.as_python_constant()
        if isinstance(self.descriptor, types.ClassMethodDescriptorType):
            return self.descriptor.__get__(None, obj)
        if hasattr(self.descriptor, "__get__"):
            return self.descriptor.__get__(obj)  # type: ignore[union-attr]
        return getattr(obj, self.descriptor.__name__)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the named method on the bound object with the given
        # arguments and returns the result tracker.
        return self.obj.call_method(tx, self.descriptor.__name__, list(args), kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode loading the object and then its attribute of
        # the descriptor's name.
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.descriptor.__name__))


class ClassMethodDescriptorVariable(VariableTracker):
    # [CLASS DOC] Models a C-level class method descriptor (for example the
    # constructor-from-keys method of dict). Holds the real descriptor; binding
    # always binds to the class, producing a bound builtin method tracker.
    """C-level classmethod descriptor (classmethod_descriptor on a type).

    CPython exposes C classmethods defined via PyMethodDef with METH_CLASS
    as classmethod_descriptor objects (PyClassMethodDescr_Type).  For
    example, dict.fromkeys is a classmethod_descriptor.  Like
    method_descriptor, these live on the type and are unbound.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L756

    classmethod_get binds the C method to the class (ignoring obj) via
    PyCMethod_New, producing a bound builtin_function_or_method.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.ClassMethodDescriptorType,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real descriptor.
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the owning class and descriptor name.
        cls_name = self.descriptor.__objclass__.__name__
        return f"ClassMethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the class-method-descriptor type.
        return types.ClassMethodDescriptorType

    def as_python_constant(self) -> types.ClassMethodDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    def get_real_python_backed_value(self) -> types.ClassMethodDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> BoundBuiltinMethodVariable:
        # [FUNCTION DOC] Descriptor get: ignores the instance and returns a bound
        # builtin method tracker bound to the owner class, keeping this source.
        # classmethod_get binds the C method to the class (ignoring obj),
        # producing a builtin_function_or_method via PyCMethod_New.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
        return BoundBuiltinMethodVariable(self.descriptor, owner, source=self.source)


class StaticMethodVariable(VariableTracker):
    # [CLASS DOC] Models a static method descriptor wrapping a callable; accessing it
    # (on a class or instance) yields the wrapped function unchanged.
    """staticmethod descriptor wrapping a callable.

    CPython's staticmethod (PyStaticMethod_Type) is a non-data descriptor
    whose tp_descr_get (sm_descr_get) simply returns the wrapped callable,
    ignoring both obj and type.
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1520
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: staticmethod,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real static method object.
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the wrapped function name.
        func_name = getattr(self.descriptor.__func__, "__name__", "?")
        return f"StaticMethodVariable({func_name})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the static method type.
        return staticmethod

    def as_python_constant(self) -> staticmethod:  # type: ignore[type-arg]
        # [FUNCTION DOC] Returns the real static method object.
        return self.descriptor

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor get: returns a tracker for the wrapped function
        # (with a function attribute source when this tracker has a source),
        # ignoring obj and owner.
        # sm_descr_get returns sm->sm_callable unconditionally.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
        func_source = AttrSource(self.source, "__func__") if self.source else None
        return VariableTracker.build(tx, self.descriptor.__func__, func_source)


class ClassMethodVariable(VariableTracker):
    # [CLASS DOC] Models a class method descriptor wrapping a callable; accessing it
    # binds the wrapped function to the owner class as a method tracker.
    """classmethod descriptor wrapping a callable.

    CPython's classmethod (PyClassMethod_Type) is a non-data descriptor
    whose tp_descr_get (cm_descr_get) creates a bound method of the
    wrapped callable bound to the class (via PyMethod_New).
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1314
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: classmethod,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real class method object.
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the wrapped function name.
        func_name = getattr(self.descriptor.__func__, "__name__", "?")
        return f"ClassMethodVariable({func_name})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the class method type.
        return classmethod

    def as_python_constant(self) -> classmethod:  # type: ignore[type-arg]
        # [FUNCTION DOC] Returns the real class method object.
        return self.descriptor

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor get: returns a method tracker binding the wrapped
        # function to the owner class; the function source derives from this source
        # and the bound source from the owner's source (named after the function),
        # when available.
        # cm_descr_get binds the wrapped function to the class.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
        func_source = AttrSource(self.source, "__func__") if self.source else None
        bound_source = (
            AttrSource(owner.source, self.descriptor.__func__.__name__)
            if owner.source
            else None
        )
        return UserMethodVariable(
            self.descriptor.__func__,
            owner,
            source_fn=func_source,
            source=bound_source,
        )


class MemberDescriptorVariable(VariableTracker):
    # [CLASS DOC] Models a member descriptor (slot or C struct field accessor). Holds
    # the real descriptor; get reads the field from the real backing object, set
    # records an attribute store as a side effect.
    """C struct field descriptor (member_descriptor on a type).

    CPython exposes C struct fields defined via PyMemberDef as
    member_descriptor objects (PyMemberDescr_Type).  These are data
    descriptors used by __slots__ and C extension types to provide
    direct access to struct members.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L793

    member_get reads the field via PyMember_GetOne.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MemberDescriptorType,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real descriptor.
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the owning class and descriptor name.
        cls_name = self.descriptor.__objclass__.__name__
        return f"MemberDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the member-descriptor type.
        return types.MemberDescriptorType

    def as_python_constant(self) -> types.MemberDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor get: checks the object type (observed TypeError),
        # obtains the real backing object (raises the unhandled-descriptor error if it
        # cannot be obtained), reads the field from it (an AttributeError or TypeError
        # becomes an observed AttributeError) and returns a tracker of the value with
        # an attribute source when the object has a source.
        # Mirrors member_get which calls PyMember_GetOne to read the
        # C struct field.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        from .object_protocol import _UnhandledDescriptorError

        attr_name = self.descriptor.__name__
        obj_value = obj.get_real_python_backed_value()
        if obj_value is NO_SUCH_SUBOBJ:
            raise _UnhandledDescriptorError(
                f"Cannot resolve member_descriptor '{attr_name}' "
                f"on {type(obj).__name__}"
            )
        try:
            resolved = self.descriptor.__get__(obj_value, type(obj_value))
        except (AttributeError, TypeError):
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj_value).__name__}' object has no attribute '{attr_name}'"
                ],
            )
        result_source = obj.source and AttrSource(obj.source, attr_name)
        return VariableTracker.build(tx, resolved, result_source)

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor set/delete: records an attribute store of value (a
        # deleted marker when value is None) on obj in the side effects, to be replayed
        # later. Returns a None constant tracker. Does no type check.
        # Mirrors member_set (PyMember_SetOne): store into the C struct field.
        # STORE_ATTR itself applies the descriptor, so replay via store_attr on
        # the target (mirrors the __slots__ path in UserDefinedObjectVariable).
        # value is None for __delete__.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L180-L196
        stored = variables.DeletedVariable() if value is None else value
        tx.output.side_effects.store_attr(obj, self.descriptor.__name__, stored)
        return variables.ConstantVariable.create(None)


class GetSetDescriptorVariable(VariableTracker):
    # [CLASS DOC] Models a getset descriptor (C getter/setter computed attribute).
    # Holds the real descriptor; always treated as a python constant. Get resolves the
    # value from the real backing object, with special source handling for type
    # attributes.
    """C getter/setter descriptor (getset_descriptor on a type).

    CPython exposes C getter/setter pairs defined via PyGetSetDef as
    getset_descriptor objects (PyGetSetDescr_Type).  These are data
    descriptors used for computed attributes backed by C functions
    (e.g. object.__class__, type.__dict__).
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L830

    getset_get calls the C getter function.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, descriptor: types.GetSetDescriptorType, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the real descriptor.
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the owning class and descriptor name.
        cls_name = self.descriptor.__objclass__.__name__
        return f"GetSetDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def get_real_python_backed_value(self) -> types.GetSetDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    def _getset_descriptor_get(
        self, tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Getter for the get-method attribute of the descriptor:
        # None without a source, otherwise a tracker for the real bound get method with
        # an attribute source.
        if self.source is None:
            return None
        source = AttrSource(self.source, "__get__")
        return VariableTracker.build(tx, self.descriptor.__get__, source)

    tp_getset = {
        "__get__": GetSet(_getset_descriptor_get, None),
    }

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def is_python_constant(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    def as_python_constant(self) -> types.GetSetDescriptorType:
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Comparison by the real constant values via the generic helper.
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor get: checks the object type (observed TypeError),
        # obtains the real backing object (raises the unhandled-descriptor error if it
        # cannot), calls the real getter (an AttributeError or TypeError becomes an
        # observed AttributeError) and returns a tracker of the value with an attribute
        # source. For type annotations, dict and mro attributes, the source is dropped
        # if a metaclass shadows the descriptor, and uses a type-MRO source for the mro.
        # Mirrors getset_get which calls the C getter function.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
        attr_name = self.descriptor.__name__
        # Try to eagerly call the C getter when we can obtain the
        # concrete Python object (UDOV.value, or as_python_constant
        # for classes/constants). Fall back to tp_getattro_impl for
        # proxy-based VTs like TensorVariable.
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        obj_value = obj.get_real_python_backed_value()
        if obj_value is NO_SUCH_SUBOBJ:
            from .object_protocol import _UnhandledDescriptorError

            raise _UnhandledDescriptorError(
                f"Cannot resolve getset_descriptor '{attr_name}' "
                f"on {type(obj).__name__}"
            )
        try:
            resolved = self.descriptor.__get__(obj_value, type(obj_value))
        except (AttributeError, TypeError):
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj_value).__name__}' object has no attribute '{attr_name}'"
                ],
            )
        result_source = obj.source and AttrSource(obj.source, attr_name)
        if (
            obj.source
            and self.descriptor.__objclass__ is type
            and attr_name in ("__annotations__", "__dict__", "__mro__")
        ):
            # Direct descriptor calls still resolve the standard type slot even
            # when a metaclass shadows the same attribute. Only attach the
            # normal attribute source when runtime attribute lookup agrees.
            static_desc = inspect.getattr_static(type(obj_value), attr_name, None)
            if static_desc is not self.descriptor:
                result_source = None
            elif attr_name == "__mro__":
                result_source = TypeMROSource(obj.source)
        return VariableTracker.build(tx, resolved, result_source)


class PropertyVariable(VariableTracker):
    # [CLASS DOC] Models a Python property descriptor. Holds the real property; get on
    # an instance inlines the getter function, get on the class returns the property.
    """Python property descriptor.

    The property type is a data descriptor with tp_descr_get =
    property_descr_get which calls fget(obj) to compute the value.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L2073
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: property,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real property object.
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the getter function name.
        fget_name = getattr(self.descriptor.fget, "__name__", "?")
        return f"PropertyVariable({fget_name})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the property type.
        return property

    def as_python_constant(self) -> property:
        # [FUNCTION DOC] Returns the real property object.
        return self.descriptor

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor get: with no instance returns the property tracker
        # itself; otherwise builds a (realized) tracker for the getter function, using
        # a getter attribute source when available, and calls it with the instance.
        # Mirrors property_descr_get: if obj is NULL or None, return self.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
        if obj is None:
            return self
        fget_source = AttrSource(self.source, "fget") if self.source else None
        fget_vt = VariableTracker.build(
            tx, self.descriptor.fget, source=fget_source, realize=True
        )
        return fget_vt.call_function(tx, [obj], {})


class TupleGetterVariable(VariableTracker):
    # [CLASS DOC] Models the tuple field accessor descriptor used by named tuples. Holds
    # the real descriptor; get on an instance subscripts the instance by the stored
    # index, get on the class returns the descriptor; set/delete always fail.
    """_tuplegetter descriptor used by namedtuple for field access.

    _tuplegetter is a C data descriptor that stores an index and returns
    self[index] on instance access. When accessed on the class (obj=None),
    it returns the descriptor itself.
    https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2735
    https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2636-L2663
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: "_collections._tuplegetter",
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the real descriptor.
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string with the field index and doc, read from the
        # descriptor's reduce data.
        _, (idx, doc) = self.descriptor.__reduce__()
        return f"TupleGetterVariable(index={idx}, doc={doc!r})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the tuple getter type.
        return _collections._tuplegetter

    def as_python_constant(self) -> "_collections._tuplegetter":
        # [FUNCTION DOC] Returns the real descriptor.
        return self.descriptor

    # _tuplegetter exposes __doc__ as a T_OBJECT member.
    # https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L2717-L2721
    tp_members = {"__doc__": Member(getset_build(lambda s: s.descriptor.__doc__))}

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor get: with no instance returns the descriptor
        # tracker; otherwise subscripts the instance with the stored field index and
        # returns that item tracker.
        # https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2636-L2663
        if obj is None:
            return self
        _, (idx, _) = self.descriptor.__reduce__()
        return obj.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(idx)], {}
        )

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # [FUNCTION DOC] Descriptor set/delete: always raises an observed
        # AttributeError (cannot set / cannot delete attribute depending on whether a
        # value was given).
        # _tuplegetter fields are read-only; tuplegetter_descr_set always
        # raises AttributeError for both set and delete.
        # https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2665-L2673
        msg = "can't delete attribute" if value is None else "can't set attribute"
        raise_observed_exception(AttributeError, tx, args=[msg])
