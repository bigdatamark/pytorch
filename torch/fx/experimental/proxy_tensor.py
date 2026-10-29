# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Proxy-tensor tracing machinery behind make_fx. Provides ProxyTorchDispatchMode,
# PythonKeyTracer and related trackers that intercept ATen/torch calls (via torch dispatch,
# torch function and the python dispatcher) and record them into a torch.fx Graph, attaching
# fake-tensor/SymInt metadata ("val", "tensor_meta") to each node. Main contents: proxy-slot
# bookkeeping (set_proxy_slot/get_proxy_slot/track_tensor), dispatch_trace, make_fx,
# DecompositionInterpreter, selective_decompose, symbolic-int proxying (handle_sym_dispatch,
# thunkify helpers), and pre-dispatch tracing support. Import-time side effects: registers
# torch.Size as a pytree node, creates module-level loggers and aliases (aten, prim).

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import operator
import threading
import typing
import typing_extensions
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Concatenate,
    Literal,
    overload,
    Protocol,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec, Self, TypeVarTuple, Unpack
from weakref import WeakKeyDictionary

import torch
import torch._ops
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import SymBool, SymInt, Tensor
from torch._custom_class_base import CustomClassBase
from torch._dispatch.python import enable_python_dispatcher
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import (
    get_reconstruct_fn,
    is_custom_class_obj,
    is_opaque_constant_type,
    is_opaque_symbolic_type,
    should_hoist,
)
from torch._logging import trace_structured
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_impls import fast_detach
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    get_plain_tensors,
    is_fake,
    is_fake_tensor,
    maybe_get_fake_mode,
    unset_fake_temporarily,
)
from torch._subclasses.functional_tensor import FunctionalTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.experimental.dynamic_spec import (
    _coerce_to_shapes_spec,
    IntVar,
    LeafSpec,
    ParamsSpec,
    ShapesSpec,
    TensorSpec,
)
from torch.fx.graph_module import _assign_attr
from torch.fx.node import (
    _side_effectful_need_to_be_preserved_pre_dispatch,
    Argument,
    Target,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
    _disable_infra_mode,
    _push_mode,
    _unset_infra_mode,
    autograd_would_have_decomposed,
    TorchDispatchMode,
    TraceableWrapperSubclass,
)
from torch.utils._stats import count
from torch.utils._thunk import Thunk
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary, WeakTensorKeyDictionary

from ._backward_state import BackwardState
from .sym_node import SymNode


if TYPE_CHECKING:
    import types
    from collections.abc import MutableMapping

    import sympy

    from torch._guards import Source
    from torch._higher_order_ops.utils import FunctionalizeCtxWrapper
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.types import BoolLikeType, FloatLikeType, IntLikeType

__all__ = [
    "PythonKeyTracer",
    "dispatch_trace",
    "make_fx",
    "DecompositionInterpreter",
    "selective_decompose",
    "py_sym_types",
    "get_innermost_proxy_mode",
    "get_proxy_mode",
    "handle_sym_dispatch",
    "maybe_enable_thunkify",
    "maybe_disable_thunkify",
]

_ProxyTracer = Union["PythonKeyTracer", "_GraphAppendingTracerEx"]
_TracingMode: TypeAlias = Literal["real", "fake", "symbolic"]

_AnyScriptObject = (torch.ScriptObject, FakeScriptObject)
_AnyScriptObjectType = torch.ScriptObject | FakeScriptObject

aten = torch.ops.aten
prim = torch.ops.prim

log = logging.getLogger(__name__)
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

CONSTANT_NUMEL_LIMIT = 1

T = TypeVar("T")
U = TypeVar("U")
_P = ParamSpec("_P")
R = TypeVar("R")
_Ts = TypeVarTuple("_Ts")

# We currently convert all SymInt to proxies before we use them.
# This could plausibly be handled at the Dynamo level.
pytree.register_pytree_node(
    torch.Size,
    lambda xs: (list(xs), None),
    lambda xs, _: tuple(xs),
    # pyrefly: ignore [bad-argument-type]
    flatten_with_keys_fn=lambda xs: (
        [(pytree.SequenceKey(i), x) for i, x in enumerate(xs)],
        None,
    ),
    serialized_type_name="torch.Size",
)
# Ideally unflattening should not lose info, but we unflatten
# torch.Size to tuple (see above). This is necessary because the
# torch.Size constructor only accepts ints whereas our infra often
# transforms them to non-ints, e.g. symint proxies. Anyway, losing
# such info can cause pytree mapping or spec matching to fail, so
# work around this problem using the following dict as needed.
_pytree_subclasses_that_lose_info = {torch.Size: tuple}


def fake_signature(fn: Callable[_P, R], nargs: int) -> Callable[_P, R]:
    # [FUNCTION DOC] HELPER: Wraps fn in a generated function with exactly nargs positional
    # parameters named arg0..argN-1, so FX does not see varargs. Returns the new callable which
    # forwards its arguments to fn. Uses eval to build the wrapper (no side effects otherwise).
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})


@contextmanager
def decompose(
    decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
) -> Generator[Mapping[OpOverload, Callable[..., Any]], None, None]:
    # [FUNCTION DOC] Context manager that enables the given decomposition table on the currently
    # active proxy mode and yields the resulting table. Raises an AssertionError if no proxy mode
    # is active. Decompositions are disabled again on exit by the mode's own context manager.
    mode = get_proxy_mode()
    if mode is None:
        raise AssertionError("Expected proxy mode to be set")
    with mode.enable_decompositions(decomposition_table) as table:
        yield table


# ensure we cannot collide with other properties
proxy_slot = object()


class _NoDefault:
    # [CLASS DOC] HELPER: Sentinel type; its single instance (no_default) marks "no default value
    # was supplied" so None can still be passed as a real default.
    pass


no_default = _NoDefault()

from torch.types import py_sym_types, PySymType


class _HasMeta(Protocol):
    # [CLASS DOC] HELPER: Structural protocol for objects (e.g. fx.Node) carrying a meta mapping
    # of strings to symbolic values; used only for typing is_sym_node.
    meta: dict[str, PySymType]


def is_sym_node(node: _HasMeta) -> bool:
    # [FUNCTION DOC] Returns True if the node's meta mapping has a "val" entry that is a symbolic
    # scalar (SymInt/SymFloat/SymBool). Raises AssertionError if the node has no meta attribute.
    if not hasattr(node, "meta"):
        raise AssertionError("All nodes traced with proxy_tensor should have meta")
    return "val" in node.meta and isinstance(node.meta["val"], py_sym_types)


class _DisableUpdateTensorTracker(threading.local):
    # [CLASS DOC] HELPER: Thread-local holder of a boolean flag (default False) that says whether
    # updates to the tensor tracker are currently suppressed in this thread.
    value: bool = False


_disable_update_tensor_tracker_tls = _DisableUpdateTensorTracker()


_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT: dict[int, torch.fx.Node] = {}


def _is_proxy_tensor_update_tensor_tracker_disabled() -> bool:
    # [FUNCTION DOC] HELPER: Returns the thread-local flag telling whether tensor-tracker updates
    # are currently disabled.
    """
    Returns current state of disabling update tensor tracker.
    """
    return _disable_update_tensor_tracker_tls.value


@contextmanager
def _proxy_tensor_disable_update_tensor_tracker() -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that sets the thread-local flag disabling tensor
    # tracker updates (so repeated in-place ops are not chained through the same fake tensor),
    # then restores the previous value on exit, even on error. See the NOTE in the docstring.
    """
    NOTE "Do not clobber inplace ops"
    By default tensor_tracker is updated every time.
    This leads to chaining every operation by the FakeTensor.
    For example for mutable ops if we have several consecutive mutable operations:

    def f(x, y, z):
        x.copy_(y)
        x.copy_(z)
        return x

    Default graph result:
    def f_graph(x, y, z)
        x_1 = x.copy_(y)
        x_2 = x_1.copy_(z)
        return x_2

    This chaining simplifies the fx passes and helps to prevent the reordering.
    But in some cases, we want those nodes to be disconnected.
    E.g. in case of splitting joint graph into forward and backward.
    If first inplace op happened in forward, second in backward,
    we want them after split to be properly placed.

    Enabling this context manager for copy_ will result in:
    def f_graph_2(x, y, z):
        x_1 = x.copy_(y)
        x_2 = x.copy_(z)
        return x

    Results of copy_ x1 and x2 will have empty users in the graph.
    The reason why this behavior is not enabled for all inplace ops is that
    some fx passes (e.g. fx quantization) rely on chaining inplace ops like add_
    in their fusions passes.
    We could revisit enabling this logic for all inplace ops in future.
    """
    orig_value = _disable_update_tensor_tracker_tls.value
    _disable_update_tensor_tracker_tls.value = True
    try:
        yield
    finally:
        _disable_update_tensor_tracker_tls.value = orig_value


@overload
def set_proxy_slot(obj: Tensor, tracer: _ProxyTracer, proxy: _ProxyTensor) -> None: ...


@overload
def set_proxy_slot(
    obj: _AnyScriptObjectType | CustomClassBase, tracer: _ProxyTracer, proxy: Proxy
) -> None: ...


@overload
def set_proxy_slot(
    obj: PySymType, tracer: _ProxyTracer, proxy: _PySymProxyType
) -> None: ...


def set_proxy_slot(
    obj: PySymType | _AnyScriptObjectType | Tensor | CustomClassBase,
    tracer: _ProxyTracer,
    proxy: object,
) -> None:
    # [FUNCTION DOC] Records proxy as the traced representation of obj for the given tracer.
    # Tensors: proxy must be a _ProxyTensor; stored in the tensor tracker (an existing entry is
    # kept if tracker updates are disabled). Script objects and custom/opaque class objects: proxy
    # must be an fx Proxy; real ScriptObjects go in the script-object tracker, fake/opaque ones in
    # the opaque tracker, where the first proxy seen for the same underlying real object wins.
    # Symbolic scalars: stored only if not already tracked (first one wins), and plain symbols are
    # also added to the sympy expression tracker. Raises AssertionError on wrong types.
    log.debug("set_proxy_slot %s (%s) %s", obj, id(obj), proxy)
    if isinstance(obj, Tensor):
        # We DO want to clobber proxies whenever we run an inplace operation
        # on a tensor, and it affects the metadata on the proxy.
        if not isinstance(proxy, _ProxyTensor):
            raise AssertionError(f"Expected _ProxyTensor, got {type(proxy)}")
        # see NOTE [Do not clobber inplace ops]
        if (
            obj not in tracer.tensor_tracker
            or not _is_proxy_tensor_update_tensor_tracker_disabled()
        ):
            tracer.tensor_tracker[obj] = proxy
    elif isinstance(obj, (_AnyScriptObject)) or is_custom_class_obj(obj):
        if not isinstance(proxy, Proxy):
            raise AssertionError(f"Expected Proxy, got {type(proxy)}")
        # ScriptObject (actual C++ torchbind) uses _WeakHashRef-keyed tracker
        # because the same C++ IValue can produce different Python wrappers.
        # FakeScriptObject/CustomClassBase uses WeakIdRef-keyed tracker because
        # value-equal objects (e.g. primal vs tangent) must be tracked separately.
        if isinstance(obj, torch.ScriptObject):
            tracer.script_object_tracker[obj] = proxy
        else:
            # NB: Never clobber a pre-existing proxy for the same
            # underlying real object.  Multiple FakeScriptObject wrappers
            # can share the same real_obj (e.g. primal vs tangent
            # placeholders during joint graph tracing).  We always keep the
            # first proxy registered, with the same rationale as the
            # symnode_tracker first-one-wins policy below: primals are
            # registered first, so this avoids spurious tangent dependencies
            # in forward outputs (which would break the partitioner).
            real_obj = None
            if isinstance(obj, FakeScriptObject):
                try:
                    real_obj = object.__getattribute__(obj, "real_obj")
                except AttributeError:
                    pass

            if real_obj is not None:
                existing = tracer._opaque_real_obj_proxy.get(id(real_obj))
                if existing is not None:
                    tracer.opaque_tracker[obj] = existing
                else:
                    tracer.opaque_tracker[obj] = proxy
                    tracer._opaque_real_obj_proxy[id(real_obj)] = proxy
            else:
                tracer.opaque_tracker[obj] = proxy
    else:
        # NB: Never clobber pre-existing proxy.  Although the proxies
        # are in principle equivalent, when we do graph partitioning
        # we need there not to be spurious dependencies on tangent inputs.
        # This works because primals get their SymInts set first, and
        # THEN later we allocate tangent inputs.  Make sure if a SymInt
        # is derivable from a primal that we use that.
        if not isinstance(obj, py_sym_types):
            raise AssertionError(f"Expected py_sym_types, got {type(obj)}")
        if obj not in tracer.symnode_tracker:
            proxy = typing.cast(_PySymProxyType, proxy)
            tracer.symnode_tracker[obj] = proxy

            # WAR: python test/dynamo/test_subclasses.py
            # TestNestedTensor.test_basic_autograd
            #
            # AOTAutograd doesn't pass the "outer sizes" as an actual argument
            # to make_fx, but it is made use of internally in AOTAutograd's
            # call to tensor unflatten.  Because the outer sizes isn't passed
            # as an argument, it is therefore untracked.  However, it turns
            # out you luck out, because *Dynamo* will manually add the outer
            # sizes as an argument so you can fix up the proxy'ness.
            #
            # This is probably fixed in
            # https://github.com/pytorch/pytorch/pull/125941/
            import sympy

            if isinstance(obj.node.expr, sympy.Symbol):
                tracer.sympy_expr_tracker[obj.node.expr] = _SympyExprTrackerValue(
                    proxy, obj
                )


def has_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> bool:
    # [FUNCTION DOC] Returns True if obj (a Tensor or SymNode) is tracked by the tracer, by looking
    # it up with a default of False and a transform returning True. Raises AssertionError for other
    # types.
    # KNOWN ISSUE: the type check only admits Tensor or SymNode, but get_proxy_slot itself only
    # accepts tensors, script objects, custom classes and SymInt/SymFloat/SymBool, so a SymNode
    # argument would fail the symbolic-type assertion inside get_proxy_slot.
    if not isinstance(obj, (Tensor, SymNode)):
        raise AssertionError(f"Expected Tensor or SymNode, got {type(obj)}")

    return bool(get_proxy_slot(obj, tracer, False, lambda _: True))


_PySymProxyType: TypeAlias = Thunk[Proxy]
_OpaqueObjectProxyType: TypeAlias = Proxy


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
) -> _ProxyTensor: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
) -> _ProxyTensor | U: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_ProxyTensor], R],
) -> R | U: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
) -> Proxy: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
) -> Proxy | U: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[Proxy], R],
) -> R | U: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
) -> _PySymProxyType: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: T,
) -> T | _PySymProxyType: ...


@overload
def get_proxy_slot(
    obj: CustomClassBase,
    tracer: _ProxyTracer,
    default: T,
) -> T | _OpaqueObjectProxyType: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_PySymProxyType], R],
) -> R | U: ...


# the default argument is what to return if the slot is not set.
# the transform argument is handy if you need to extract a subfield from
# the successfully looked up result (but NOT the default.)
def get_proxy_slot(
    obj: Tensor | _AnyScriptObjectType | PySymType | CustomClassBase,
    tracer: _ProxyTracer,
    default: object = no_default,
    transform: Callable[..., Any] = lambda x: x,
) -> object:
    # [FUNCTION DOC] Looks up the proxy tracked for obj in the tracer's matching tracker (tensor,
    # script object, opaque object, or symbolic-node tracker) and returns transform applied to it.
    # For unknown symbolic values it first tries the sympy-expression tracker and then tries to
    # rebuild the proxy from the expression's parts; for FakeScriptObjects it falls back to the
    # real-object dedup map. If nothing is found, returns default, or raises RuntimeError when no
    # default was given. Raises AssertionError for unsupported object types. The transform is not
    # applied to the default. May register new proxies as a side effect of rebuilding symbols.
    # KNOWN ISSUE: the RuntimeError message lacks a space before "is not tracked".
    tracker: Any
    if isinstance(obj, Tensor):
        tracker = tracer.tensor_tracker
    elif isinstance(obj, _AnyScriptObject) or is_custom_class_obj(obj):
        if isinstance(obj, torch.ScriptObject):
            tracker = tracer.script_object_tracker
        else:
            tracker = tracer.opaque_tracker
    else:
        if not isinstance(obj, py_sym_types):
            raise AssertionError(f"Expected py_sym_types, got {type(obj)}")
        tracker = tracer.symnode_tracker

    # pyrefly: ignore [index-error]
    # pyrefly: ignore [no-matching-overload, bad-argument-type]
    value = tracker.get(obj)

    if value is None and isinstance(obj, py_sym_types):
        if obj.node.is_symbolic():
            # Last ditch - we found a SymInt (SymBool, etc) we don't know
            # about.
            if (tmp := tracer.sympy_expr_tracker.get(obj.node.expr)) is not None:
                value = tmp.proxy

            else:
                # Attempt to build it from first principles.
                _build_proxy_for_sym_expr(tracer, obj.node.expr, obj)
                # pyrefly: ignore [bad-argument-type, no-matching-overload]
                value = tracker.get(obj)

    if value is None and isinstance(obj, FakeScriptObject):
        # A new FakeScriptObject wrapping the same real_obj may have been
        # created (e.g. output flattening in unwrap_tensor_subclasses calls
        # maybe_to_fake_obj which always mints a fresh wrapper).  Fall back
        # to the real-object dedup map that set_proxy_slot maintains.
        value = tracer._opaque_real_obj_proxy.get(id(obj.real_obj))

    if value is None:
        # We don't know this value - return the default.
        if isinstance(default, _NoDefault):
            raise RuntimeError(
                f"{obj} ({type(obj)}, {id(obj)})is not tracked with proxy for {tracer}"
            )
        return default

    res = transform(value)
    return res


# Recursively traverses traceable wrapper subclasses,
# returnining an (unordered) list of Proxy objects that are tracked
# for all inner tensors, given the current extant proxy mode.
# Returns an empty list if no proxy mode is active.
def _get_proxies(t: torch.Tensor | TraceableWrapperSubclass) -> list[Proxy]:
    # [FUNCTION DOC] HELPER: Returns the list of Proxy objects tracked for every inner plain tensor
    # of t (recursing through traceable wrapper subclasses) under the active proxy dispatch mode;
    # returns an empty list if no proxy mode is active. Functional tensors are unwrapped to their
    # inner tensor first. Raises AssertionError if the active mode is not a ProxyTorchDispatchMode.
    # KNOWN ISSUE: get_proxy_slot is called without a default, so an untracked inner tensor raises
    # RuntimeError and the following "is not None" check can never be false.
    proxies = []
    mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    if mode is None:
        return proxies
    if not isinstance(mode, ProxyTorchDispatchMode):
        raise AssertionError(f"Expected ProxyTorchDispatchMode, got {type(mode)}")
    tracer = mode.tracer
    for t_inner in get_plain_tensors(t, out=[]):
        if isinstance(t_inner, FunctionalTensor):
            t_inner = torch._from_functional_tensor(t_inner.elem)
        if not isinstance(t_inner, torch.Tensor):
            continue
        proxy_tensor = get_proxy_slot(t_inner, tracer)
        if proxy_tensor is not None:
            proxies.append(proxy_tensor.proxy)
    return proxies


# sympy Max/Min are n-ary (they flatten, e.g. Max(a, Max(b, c)) -> Max(a, b, c)),
# but torch.sym_max/sym_min are binary. Reduce so _build_proxy_for_sym_expr can
# rebuild flattened expressions as nested binary ops. These are module-level
# (not lambdas) so they have a qualified name and survive FX codegen/pickling
# when used as a graph node target.
def _nary_sym_max(*args: Any) -> Any:
    # [FUNCTION DOC] HELPER: Folds any number of symbolic values with the binary sym_max; module
    # level (not a lambda) so it has a qualified name usable as an FX node target.
    return functools.reduce(torch.sym_max, args)


def _nary_sym_min(*args: Any) -> Any:
    # [FUNCTION DOC] HELPER: Folds any number of symbolic values with the binary sym_min; module
    # level so it has a qualified name usable as an FX node target.
    return functools.reduce(torch.sym_min, args)


@functools.cache
def _sympy_handlers() -> dict[type[sympy.Expr], Callable[..., Any]]:
    # [FUNCTION DOC] HELPER: Cached. Returns a mapping from sympy expression classes to Python
    # callables used to rebuild symbolic expressions: operator functions where an operator exists,
    # n-ary wrappers for Max/Min, power for Pow, and sym_sum for the n-ary Add. Imports sympy
    # lazily.
    """
    Returns a dict mapping sympy types to Python callables
    (e.g. ``sympy.Mul`` -> ``operator.mul``, ``sympy.Add`` -> ``torch.sym_sum``).
    """
    import sympy

    import torch.utils._sympy.interp

    handlers = {}
    for k, v in torch.utils._sympy.interp.handlers().items():
        op = getattr(operator, v, None)
        if op is not None:
            handlers[k] = op
        # sympy Max/Min have no operator.* equivalent. Map them to n-ary-safe
        # wrappers over torch.sym_max/sym_min. Without this,
        # _build_proxy_for_sym_expr cannot rebuild expressions like Max(1, u2)
        # that escape a disable_proxy_modes_tracing region (e.g. DTensor shard
        # propagation size inference).
        elif v == "maximum":
            handlers[k] = _nary_sym_max
        elif v == "minimum":
            handlers[k] = _nary_sym_min
        # sympy.Pow / PowByNatural map to the interp name "pow_by_natural",
        # which has no operator.* equivalent. sympy canonicalizes x * x into
        # Pow(x, 2), so without this _build_proxy_for_sym_expr cannot rebuild
        # any repeated-symbol product (e.g. a reduction numel s0 * s1**2 when
        # two equal dims duck-share a symbol).
        elif v == "pow_by_natural":
            handlers[k] = operator.pow

    # sympy.Add is n-ary (e.g. Add(a, b, c)) but operator.add is binary.
    # torch.sym_sum handles n-ary integer addition and accepts both
    # sym_sum([a, b, c]) and sym_sum(a, b, c).
    handlers[sympy.Add] = torch.sym_sum
    return handlers


def _build_proxy_for_sym_expr(
    tracer: _ProxyTracer, expr: sympy.Expr, out: PySymType | None = None
) -> IntLikeType | FloatLikeType | BoolLikeType | None:
    # [FUNCTION DOC] HELPER: Rebuilds a symbolic value for the sympy expression expr from values
    # already tracked by the tracer. Returns the tracked value if expr is known; returns plain
    # numbers for numeric expressions; otherwise recursively rebuilds each argument and applies
    # the handler for the expression type, returning None if any piece or handler is missing. If
    # out is given, the operation is registered as producing out instead of creating a new value
    # (out must then not coincide with an already tracked expression, else AssertionError).
    """
    Decompose `expr` and look for the pieces as inputs. If `out` is provided
    then that will be the resulting SymNode (and `out.expr` must be the same as
    `expr`).

    This function is used when the ProxyTorchDispatchMode sees a SymNode
    that it hasn't seen before to try to associate it with traced inputs.

    How can this happen?

    First thing to remember is that although sympy.Exprs are interned (so
    `sympy.Expr("s3*s4")` will always have the same `id` and will always compare
    equal) SymNode does not (so doing `SymNode("s3")*SymNode("s4")` twice in a
    row will give two unique SymNodes).

    - On way for this to happen is if we turn off tracing to compute an
      intermediate value and then USE that value with tracing turned on - for
      example if we turn off tracing to do some FakeTensor propagation to
      compute a size (dtensor does this) but then turn tracing back on and use
      that computed size.

    - Another way is if we compute a size in one graph and stash it somewhere
      hidden (such as in some meta-data) and later use it in a different graph
      (dtensor does this too). Since the size was computed in the first graph
      and it's not an official input to the second graph it's not tracked
      properly. This is often going to show up as it usually works in fullgraph
      but a graph break causes a failure.

    To handle this we decompose the sympy.Expr and look for the pieces as
    inputs. But there are problems with this approach:

    - We lose operation provenance: We end up figuring out where to get the
      inputs - but those may not actually be correct. If we have "s1" coming in
      from both tensor1 and tensor2 and we pick the wrong one we could end up
      keeping a tensor alive longer than intended.

    - There's no guarantee that those values are inputs to the graph: If we have
      "s1*s2" computed in a graph #1 and used in graph #2 there's no guarantee
      that the input that holds "s1" is actually an input on graph #2.

    - The decomposition isn't guaranteed to be the same: Sympy can "simplify"
      expressions so it's possible that our inputs are "s1*s2" and "s3" but we
      decompose it into "s1" and "s2*s3" - which wouldn't be found.

    Other ways we could handle this:

    - Don't: Just require that all inputs are tracked properly. This is the
      "correct" solution but harder because you need to track down each
      potential problem one by one and fix them. And when it fails it's a lot of
      work to figure out both why it's failing and the right way to fix it. This
      is complicated by the fact that a stashed value could be incorrect but
      work fine until we happen to get an graph break in the wrong place - so it
      may be a while before the bug is found. (Maybe we need a "dynamo abuse
      mode" where we run tests with as many graph breaks inserted as possible?)

    - Track SymNode ops separately from proxy tracing: Right now SymNode
      operations are tracked as part of the proxy tracing - so when we disable
      proxy tracing we also disable SymNode tracing. But we don't have to do
      that - we could instead always have SymNodes track where they came from
      and just use that when needed. This solves the problem of tracing being
      temporarily turned off but doesn't help if an input isn't present after a
      graph break.

    - Better decomposition: Right now the decomposition is pretty simple. We do
      have a sat-solver available to us so we could theoretically do a better
      job figuring out a "correct" decomposition. But that still relies on
      having the inputs available at all - which isn't a guarantee.
    """

    if (value := tracer.sympy_expr_tracker.get(expr)) is not None:
        if out:
            raise AssertionError(
                "out should be empty when expr is in sympy_expr_tracker"
            )
        return value.value

    if isinstance(expr, (int, float, bool)):
        return expr
    if expr.is_Integer:
        return int(expr)
    if expr.is_Float:
        return float(expr)

    args = []
    for arg in expr.args:
        if (arg_value := _build_proxy_for_sym_expr(tracer, arg)) is None:
            return None
        args.append(arg_value)

    func: OpOverload | None = _sympy_handlers().get(expr.func)  # type: ignore[assignment]
    if not func:
        return None

    if out is None:
        out = func(*args)
    else:
        _sym_register(tracer, func, tuple(args), out)
    return out


def snapshot_fake(val: Tensor, include_real: bool = False) -> Tensor | None:
    # [FUNCTION DOC] Returns a detached copy of val for storing in node metadata. Fake tensors use
    # the fast detach path (optionally keeping the real tensor); other tensors use detach.
    # val.detach() will also eventually call fast_detach(),
    # but this saves us a full trip into __torch_dispatch__
    # (snapshot_fake is called a lot)
    if is_fake_tensor(val):
        return fast_detach(maybe_get_fake_mode(val), val, include_real)
    else:
        return val.detach()


_ExtractValType: TypeAlias = (
    None
    | PySymType
    | CustomClassBase
    | _AnyScriptObjectType
    | BackwardState
    | list["_ExtractValType"]
    | tuple["_ExtractValType", ...]
    | dict[str, "_ExtractValType"]
    | Tensor
    | int
    | float
    | bool
)


def extract_val(val: _ExtractValType, include_real: bool = False) -> _ExtractValType:
    # [FUNCTION DOC] Converts a value into what is stored in node meta "val". Fake tensors are
    # snapshotted; symbolic scalars, script objects, custom class objects, BackwardState and plain
    # Python scalars/None are returned unchanged; lists, tuples and dicts are converted
    # recursively; real dense tensors become a fresh uninitialized strided fake tensor with the
    # same shape, stride, device and dtype (using a detected or new FakeTensorMode); sparse
    # tensors become None. Unsupported types hit an exhaustiveness assertion.
    # KNOWN ISSUE: the include_real flag is not forwarded when recursing into lists, tuples and
    # dicts, so real values are only kept for top-level fake tensors.
    if is_fake(val):
        return snapshot_fake(val, include_real=include_real)
    elif isinstance(val, py_sym_types):
        return val
    elif isinstance(val, (_AnyScriptObject, CustomClassBase)):
        return val
    elif isinstance(val, BackwardState):
        return val
    elif is_custom_class_obj(val):
        return val
    elif isinstance(val, (list, tuple)):
        return val.__class__([extract_val(x) for x in val])
    elif isinstance(val, dict):
        return {k: extract_val(v) for k, v in val.items()}
    elif isinstance(val, Tensor):
        if not val.is_sparse:
            # NB: Kinda hacky, but we should try to get val as the metadata
            # everywhere
            # TODO: This doesn't properly track storages.  A more robust
            # approach would be to maintain a per-trace FakeTensorMode and
            # from_real_tensor to create fake values (don't forget to
            # snapshot_fake)
            from torch._guards import detect_fake_mode

            fake_tensor_mode = detect_fake_mode(val)
            if not fake_tensor_mode:
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            with fake_tensor_mode:
                return torch.empty_strided(
                    val.shape, val.stride(), device=val.device, dtype=val.dtype
                )
        else:
            return None
    elif isinstance(val, (int, float, bool)):
        return val
    elif val is None:
        return None

    typing_extensions.assert_never(val)


@contextmanager
def _enable_thunkify(
    tracer: _ProxyTracer, *, enable: bool = True
) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that sets the tracer's enable_thunkify flag to
    # enable for the duration, restoring the old value on exit even on error. When on, symbolic
    # computation is deferred and only traced into the graph if used.
    """
    Enable thunkification inside the context manager.  Thunkification prevents
    SymNode computation from directly being traced into an FX graph; instead,
    the compute is only added to the graph if it is actually used.  This helps
    us track SymNode compute when it is computed (since we need /something/
    to put in the tracker) even if it is unlikely to be used.
    """
    old = tracer.enable_thunkify
    tracer.enable_thunkify = enable
    try:
        yield
    finally:
        tracer.enable_thunkify = old


@contextmanager
def maybe_disable_thunkify() -> Generator[None, None, None]:
    """Within a context, disable thunkification.  See :func:`maybe_enable_thunkify`
    for more details.  This is helpful if you have a wrapper function which
    you want to enable thunkification on, but in some segment on the inside (say,
    the original user function), you want to disable thunkification as you know
    it is not needed there.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer, enable=False):
            yield
    else:
        yield


@contextmanager
def maybe_enable_thunkify() -> Generator[None, None, None]:
    """Within this context manager, if you are doing make_fx tracing, we will thunkify
    all SymNode compute and avoid tracing it into the graph unless it is actually needed.
    You should prefer to avoid using this as much as possible, as lazy evaluation of
    SymNode tracing can lead to long chains of thunks which will stack overflow
    if you evaluate them.  However, this is currently sometimes necessary as there
    are buggy parts of PT2 which will fail with "s0 is not tracked with proxy" error
    due to insufficient tracing of SymNode computation.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer):
            yield
    else:
        yield


# Note [invariants for node meta 'val']
# What invariants do we have for the 'val' set on the FX node?  It has accurate
# metadata... but only for metadata that exists "below" all other subsystems
# (most notably autograd, but also vmap, functorch transforms, etc).  This means
# you can get the dtype, shape, stride, storage, but you CANNOT get requires_grad,
# grad_fn, _base (_base actually may be set due to recursive call to
# ADInplaceOrView, but you shouldn't rely on it.)
def set_meta(proxy: Proxy, val: _ExtractValType) -> Proxy:
    # [FUNCTION DOC] Stores extract_val of val as the proxy node's meta "val" (keeping real values
    # for placeholders), and on a best-effort basis also sets meta "tensor_meta" for fake tensors
    # and dense real tensors. Returns the proxy. Mutates the node's meta.
    proxy.node.meta["val"] = extract_val(
        val, include_real=(proxy.node.op == "placeholder")
    )

    with _enable_thunkify(proxy.tracer):  # type: ignore[arg-type]
        # Best effort tensor_meta setting; prefer using val!
        if is_fake(val):
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
        elif isinstance(val, Tensor) and not val.is_sparse:
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
    return proxy


def thunkify(
    tracer: _ProxyTracer, f: Callable[_P, R], *args: _P.args, **kwargs: _P.kwargs
) -> Thunk[R]:
    # [FUNCTION DOC] Returns a Thunk for f(*args, **kwargs). If the tracer has thunkification
    # enabled the call is deferred until the thunk is forced (the Thunk caches the result);
    # otherwise f runs immediately and the thunk wraps its result.
    """
    Delays computation of f until it's called again
    Also caches the result
    """
    if tracer.enable_thunkify:
        return Thunk(functools.partial(f, *args, **kwargs))
    else:
        r = f(*args, **kwargs)
        return Thunk(lambda: r)


def track_tensor(
    tensor: Tensor, proxy: Proxy, *, constant: Tensor | None, tracer: _ProxyTracer
) -> None:
    # [FUNCTION DOC] Associates tensor with proxy in the tracer's tensor tracker and also
    # associates each symbolic size and stride of the tensor (dense tensors only for strides)
    # with lazily created sym_size/sym_stride proxy nodes derived from proxy, with metadata set.
    # constant optionally holds a constant tensor value to remember alongside the proxy.
    def try_set_proxy_slot(
        outer_s: IntLikeType,
        proxy_callable: Callable[Concatenate[PySymType, _P], Proxy],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> None:
        # [FUNCTION DOC] HELPER: If outer_s is a SymInt, registers a thunk (calling proxy_callable on
        # it with the extra arguments) as its proxy slot, with thunkification enabled; non-SymInt
        # (plain int) values are ignored.
        if not callable(proxy_callable):
            raise AssertionError("proxy_callable must be callable")
        if isinstance(outer_s, SymInt):
            with _enable_thunkify(tracer):
                set_proxy_slot(
                    outer_s,
                    tracer,
                    thunkify(tracer, proxy_callable, outer_s, *args, **kwargs),
                )

    # The basic idea is that we need to associate each tensor/SymInt
    # with a Proxy.  How do we setup this association?  We just store
    # the proxy on the proxy slot of the object, keyed on the tracer
    # (so that if we have multiple tracers at the same time, they
    # don't clobber each other.)
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(
            s,
            lambda x, i: set_meta(
                tracer.create_proxy(
                    "call_function", torch.ops.aten.sym_size.int, (proxy, i), {}
                ),
                x,
            ),
            i,
        )

    if not is_sparse_any(tensor):
        for i, s in enumerate(tensor.stride()):
            try_set_proxy_slot(
                s,
                lambda x, i: set_meta(
                    tracer.create_proxy(
                        "call_function", torch.ops.aten.sym_stride.int, (proxy, i), {}
                    ),
                    x,
                ),
                i,
            )

    try_set_proxy_slot(
        tensor.numel(),
        lambda x: set_meta(
            tracer.create_proxy(
                "call_function", torch.ops.aten.sym_numel.default, (proxy,), {}
            ),
            x,
        ),
    )
    if not is_sparse_any(tensor):
        try_set_proxy_slot(
            tensor.storage_offset(),
            lambda x: set_meta(
                tracer.create_proxy(
                    "call_function",
                    torch.ops.aten.sym_storage_offset.default,
                    (proxy,),
                    {},
                ),
                x,
            ),
        )
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))


_NestedProxys = Union[  # noqa: UP007
    Proxy, Sequence["_NestedProxys"], Mapping[object, "_NestedProxys"]
]
_NestedTensors = Union[  # noqa: UP007
    Tensor, Sequence["_NestedTensors"], Mapping[object, "_NestedTensors"]
]


def track_tensor_tree(
    inner_res: T,
    proxy_res: _NestedProxys,
    *,
    constant: _NestedTensors | None,
    tracer: _ProxyTracer,
) -> T:
    # [FUNCTION DOC] Walks inner_res (tensors, symbolic scalars, script/opaque objects, tuples, lists,
    # dicts, BackwardState) in parallel with proxy_res and registers each leaf's proxy with the tracer
    # (tracking tensors, setting node meta, and for BackwardState storing the proxy on it). Sequences
    # and dicts are indexed on the proxy to get per-element proxies; constants are split the same way.
    # Unbacked symbol bindings are set once, at the top-level call only. Non-hoisted opaque constant
    # types are only given meta, not tracked. Unsupported primitives are ignored. Returns inner_res
    # unchanged. Raises AssertionError if proxies/constants have unexpected types or a dict result has
    # a non-None constant.
    # NB: We call set_unbacked_bindings only on the *topmost* call to
    # track_tensor_tree, not recursive calls.  This is because there must
    # be only ONE unbacked_binding proxy call, and it should be the one
    # where all of the unbacked SymInts actually first come into existence.
    # If you call this again on the inner proxies for the tuple projections,
    # you will have multiple unbacked_bindings for the same symbol, but
    # they're not going to show up anywhere.
    #
    # I was briefly deceived into setting unbacked bindings recursively when
    # working on https://github.com/pytorch/pytorch/pull/133585 because I
    # observed that some extra unbacked bindings were needed to handle some
    # higher order operator code.  But actually it looks like this was
    # just an unrelated bug that needed to be fixed separately.
    _set_unbacked_bindings(inner_res, proxy_res)

    def wrap_with_proxy(
        e: object, proxy: _NestedProxys, constant: _NestedTensors | None
    ) -> None:
        # [FUNCTION DOC] HELPER: Recursive worker for track_tensor_tree; dispatches on the type of e to
        # register proxy (and constant for tensors) with the enclosing tracer, setting node meta. Mutates
        # tracer trackers and node meta; returns nothing.
        if isinstance(e, Tensor):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            if not (constant is None or isinstance(constant, Tensor)):
                raise AssertionError(f"Expected None or Tensor, got {type(constant)}")
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            # NB: eagerly set meta here, so that the numbering is in order
            set_meta(proxy, e)
            set_proxy_slot(e, tracer, thunkify(tracer, lambda: proxy))
        elif isinstance(e, _AnyScriptObject) or is_custom_class_obj(e):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            # Non-hoisted opaque value types should be baked as constants
            # in the graph, not tracked as proxy references. This matches
            # dynamo's behavior where non-hoisted values are not graph inputs.
            if (
                is_opaque_constant_type(type(e))  # pyrefly: ignore[bad-argument-type]
                and not should_hoist(type(e))
            ):
                set_meta(proxy, e)
                return
            set_proxy_slot(e, tracer, proxy)
            set_meta(proxy, e)
        elif isinstance(e, (tuple, list)):
            # example use case: allreduce_ returns ([tensor], work)
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            def get_constant(
                c: _NestedTensors | None, idx: int
            ) -> _NestedTensors | None:
                # [FUNCTION DOC] HELPER: Returns the idx-th element of the constant container c, or None if c is
                # None. Raises AssertionError if c is not a list or tuple.
                if c is None:
                    return None
                else:
                    if not isinstance(c, (list, tuple)):
                        raise AssertionError(f"Expected list or tuple, got {type(c)}")
                    # pyrefly: ignore [bad-return]
                    return c[idx]

            for idx, ee in enumerate(e):
                # Use an indexer here - if proxy is a List then it will unwrap
                # it. If it's a Proxy then it will proxy the getelem.
                wrap_with_proxy(ee, proxy[idx], get_constant(constant, idx))  # type: ignore[index]

        elif isinstance(e, dict):
            # example use case: triton_kernel_wrapper takes arguments as kwargs

            # In theory we could support const-prop when proxy-tensor-tracing
            # operators that returns dicts of tensors, but we have no use case
            # for it today (since the only op we currently trace that can
            # return a dict is triton_kernel_wrapper_functional/mutation,
            # which does not participate in const-prop)
            if constant is not None:
                raise AssertionError("Expected constant to be None")

            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            for key, val in e.items():
                wrap_with_proxy(val, proxy[key], None)  # type: ignore[index]

        elif isinstance(e, BackwardState):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            set_meta(proxy, e)
            e.proxy = proxy
        else:
            # intentionally pass on unsupported primitives
            pass

    wrap_with_proxy(inner_res, proxy_res, constant)

    return inner_res


@dataclass
class _ProxyTensor:
    # [CLASS DOC] HELPER: Record stored in the tensor tracker: proxy is the fx Proxy for a traced
    # tensor and constant is an optional real tensor value known statically (used for constant
    # propagation), else None.
    proxy: Proxy
    constant: Tensor | None


def fetch_sym_proxy(
    tracer: _ProxyTracer,
) -> Callable[[PySymType], bool | int | float | Proxy]:
    # [FUNCTION DOC] HELPER: Returns a function that maps a symbolic scalar to what should appear in
    # the graph for it: its constant if it has one, a plain bool/int/float if its expression is a
    # number, otherwise the forced proxy tracked for it (all non-constant symbols must be tracked, else
    # get_proxy_slot raises).
    def inner(e: PySymType) -> int | bool | float | Proxy:
        # [FUNCTION DOC] HELPER: Implements the mapping described for fetch_sym_proxy for one symbolic
        # value e.
        n = e.node
        if n.constant is not None:
            return n.constant
        if e.node.expr.is_number:
            if isinstance(e, SymBool):
                return bool(e.node.expr)
            elif isinstance(e, SymInt):
                return int(e.node.expr)
            return float(e.node.expr)
        else:
            if not isinstance(e, py_sym_types):
                raise AssertionError(f"Expected py_sym_types, got {type(e)}")
            # NB: we REQUIRE all symints to be tracked
            return get_proxy_slot(e, tracer).force()

    return inner


@overload
def fetch_object_proxy(tracer: _ProxyTracer, t: Tensor) -> _ProxyTensor | Tensor: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: _AnyScriptObjectType
) -> Proxy | _AnyScriptObjectType: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: PySymType
) -> _PySymProxyType | PySymType: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: CustomClassBase
) -> _OpaqueObjectProxyType | PySymType: ...


def fetch_object_proxy(
    tracer: _ProxyTracer,
    t: Tensor | _AnyScriptObjectType | PySymType | CustomClassBase,
) -> object:
    # [FUNCTION DOC] Returns the tracked proxy object for t (tensor record, script object proxy,
    # symbolic thunk or opaque proxy), or t itself if it is not tracked.
    return get_proxy_slot(t, tracer, t)


HANDLED_TYPES = (Tensor, torch.nn.Parameter, FakeTensor)


def _maybe_record_pointwise_barrier(
    func: object, proxy_mode: ProxyTorchDispatchMode
) -> None:
    # [FUNCTION DOC] HELPER: When emulate_precision_casts is on and no decomposition layer is active,
    # marks the most recently added graph node with meta flag low_precision_pointwise_barrier if func
    # is an OpOverload and its output or any input value is a bfloat16/float16 tensor. Returns nothing.
    """
    Records operators whose tensor outputs or inputs are fp16/bf16 so downstream pointwise code can
    emulate eager's rounding behavior when emulate_precision_casts is enabled.
    """
    if proxy_mode.decomp_layers or not proxy_mode.emulate_precision_casts:
        return

    if not isinstance(func, torch._ops.OpOverload):
        return

    last_node = next(iter(reversed(proxy_mode.tracer.graph.nodes)))
    t = last_node.meta.get("val")
    low_pr_fp = (torch.bfloat16, torch.float16)

    output_low_precision = isinstance(t, torch.Tensor) and t.dtype in low_pr_fp

    if not output_low_precision:
        for input_node in last_node.all_input_nodes:
            val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
            if isinstance(val, torch.Tensor) and val.dtype in low_pr_fp:
                output_low_precision = True
                break

    if not output_low_precision:
        return

    last_node.meta["low_precision_pointwise_barrier"] = True


def _fetch_proxies_and_all_constant_flag(
    flat_args_kwargs: list[object] | tuple[object, ...], tracer: _ProxyTracer
) -> tuple[list[object], tuple[object, ...], bool]:
    # [FUNCTION DOC] HELPER: For flattened args and kwargs, replaces tensors, script objects and custom
    # class objects by their tracked records, and computes (1) that list, (2) a tuple of the
    # corresponding raw proxies (symbolic scalars converted via fetch_sym_proxy), and (3) a flag that is
    # True only if every tracked tensor has a constant and no symbolic scalar is present.
    """
    Given flat arguments, fetch the proxies and whether they are all constants.
    This is later used in proxy_call or when someone is trying to stitch together
    graph node in tf or td modes.
    """
    f_flat_args_kwargs = [
        (
            fetch_object_proxy(tracer, x)
            if isinstance(x, (Tensor, _AnyScriptObject)) or is_custom_class_obj(x)
            else x
        )
        for x in flat_args_kwargs
    ]

    # If there are SymInts, we also should not consider this constant.
    # However, fake tensor handling of SymInts is sufficiently broken that
    # I couldn't write a test for this case
    all_constant = (
        not any(
            t.constant is None
            for t in f_flat_args_kwargs
            if isinstance(t, _ProxyTensor)
        )
        # TODO: maybe constant SymInts should also be allowed?  Not sure if
        # this can happen
        and not any(isinstance(x, py_sym_types) for x in flat_args_kwargs)
    )

    proxy_flat_args_kwargs = [
        e.proxy if isinstance(e, _ProxyTensor) else e for e in f_flat_args_kwargs
    ]

    proxy_flat_args_kwargs = [
        (fetch_sym_proxy(tracer)(e) if isinstance(e, py_sym_types) else e)
        for e in proxy_flat_args_kwargs
    ]

    return f_flat_args_kwargs, tuple(proxy_flat_args_kwargs), all_constant


def _coor_enabled() -> bool:
    # [FUNCTION DOC] HELPER: Returns the compile-on-one-rank flag read from the compiler config.
    """True when compile-on-one-rank (CooR) device-as-parameter rewriting should run.

    Reads the cross-cutting flag from ``torch.compiler.config`` (its canonical home),
    honoring ``config.patch``.
    """
    from torch.compiler import config as compiler_config

    return compiler_config.compile_on_one_rank


def _coor_current_accelerator() -> torch.device | None:
    # [FUNCTION DOC] HELPER: Returns the current accelerator as an indexed device, or None when no
    # accelerator is actually available (build support alone is not enough).
    """The current accelerator as an indexed device (e.g. cuda:0), or None if there is no
    accelerator. Used to classify device operands under compile-on-one-rank."""
    # check_available matters: current_accelerator() reports the accelerator the build
    # supports, so on a CUDA-enabled build with no visible GPUs it returns cuda and the
    # index lookup below then raises "No CUDA GPUs are available". Such a machine has no
    # accelerator for our purposes, and a cpu-only graph must still compile there.
    acc = torch.accelerator.current_accelerator(check_available=True)
    if acc is None:
        return None
    return torch.device(acc.type, torch.accelerator.current_device_index())


def _coor_current_device() -> torch.device:
    # [FUNCTION DOC] HELPER: Returns the current accelerator device with its concrete index; this is
    # the runtime target of the coor::current_device op. Raises RuntimeError if there is no accelerator.
    """The current accelerator as a fully-specialized indexed device (e.g. cuda:0).

    Runtime target of the compile-on-one-rank device node: each rank resolves it to its
    own current device (cuda:0 on rank 0, cuda:3 on rank 3). Carries the concrete index
    (unlike type-only ``current_accelerator()``) so consuming ops -- and inductor -- get a
    fully specified device.
    """
    cur = _coor_current_accelerator()
    if cur is None:
        raise RuntimeError("compile-on-one-rank requires an accelerator")
    return cur


# Registered as an op (not a bare function) so it is a serializable call_function target
# with a stable identity for precompile/export and for consumers to match. It reads only
# torch.accelerator, so it lives in core fx with no torch.distributed coupling.
torch.library.define(
    "coor::current_device",
    "() -> Device",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("coor::current_device", "CompositeExplicitAutograd")
def _coor_current_device_impl() -> torch.device:
    # [FUNCTION DOC] HELPER: Eager implementation of the coor::current_device op; returns the current
    # accelerator device.
    return _coor_current_device()


@torch.library.register_fake("coor::current_device")
def _coor_current_device_fake() -> torch.device:
    # [FUNCTION DOC] HELPER: Fake implementation of the coor::current_device op; same result as eager.
    return _coor_current_device()


def _coor_check_current_accelerator(
    device: torch.device, cur: torch.device | None
) -> None:
    # [FUNCTION DOC] HELPER: Returns silently for cpu/meta devices or a device matching the current
    # accelerator cur; raises RuntimeError if the type differs, cur is None, or an explicit index
    # differs from cur's index (the graph would be rank-specific).
    """Raise if ``device`` is an accelerator other than ``cur`` (``cpu``/``meta`` pass).

    A different accelerator type or a non-current index is rank-specific and cannot be made
    device-agnostic for compile-on-one-rank, so the graph is refused rather than left
    silently non-portable. Callers supply ``cur`` from their own per-scope cache of the
    current accelerator (the tracer while tracing, the wrapper during codegen).
    """
    if device.type in ("cpu", "meta"):
        return
    if cur is None or device.type != cur.type:
        raise RuntimeError(
            f"device_as_parameter: the graph references {device}, which is not the "
            f"current accelerator ({cur}); the traced graph cannot be made "
            f"device-agnostic for compile-on-one-rank."
        )
    if device.index is not None and device.index != cur.index:
        raise RuntimeError(
            f"device_as_parameter: the graph references {device}, whose index differs "
            f"from the current accelerator ({cur}); the traced graph cannot be made "
            f"device-agnostic for compile-on-one-rank."
        )


def _current_device_edge(tracer: _ProxyTracer, device: torch.device) -> Proxy:
    # [FUNCTION DOC] HELPER: Returns the single coor::current_device proxy of the graph, creating it on
    # first use (call_function node whose meta val is the current device) and caching it on the tracer.
    # The device argument is not used.
    """Return a per-graph ``coor::current_device`` proxy for CooR, cached once per graph.

    Replaces a baked accelerator device operand (``cuda:0`` or bare ``cuda``) with a
    reference to a single node that, at runtime, yields each rank's specialized device
    (cuda:0 on rank 0, cuda:3 on rank 3). The node *call* is rank-agnostic; only its
    runtime return and ``node.meta["val"]`` vary, so ``.code`` matches across ranks (a
    graph fingerprint must therefore ignore the per-tensor devices in ``meta``).

    NOTE: a consumer that lowers this graph (e.g. inductor) cannot represent a
    device-returning op, so it must strip this node before lowering.
    """
    cached = tracer._current_device_node
    if cached is not None:
        return cached
    edge = tracer.create_proxy(
        "call_function", torch.ops.coor.current_device.default, (), {}
    )
    edge.node.meta["val"] = _coor_current_device()
    tracer._current_device_node = edge
    return edge


def _coor_traced_current_accelerator(tracer: _ProxyTracer) -> torch.device | None:
    # [FUNCTION DOC] HELPER: Returns the tracer's cached current accelerator, computing and caching it
    # on first use.
    """The current accelerator for this trace, read once (a cudaGetDevice) and cached on the
    tracer beside the current_device node, so the input check and the per-op device-operand
    rewrite together pay a single lookup per trace."""
    if tracer._coor_current_accelerator is None:
        tracer._coor_current_accelerator = _coor_current_accelerator()
    return tracer._coor_current_accelerator


def _coor_match_current_accelerator(tracer: _ProxyTracer, device: torch.device) -> bool:
    # [FUNCTION DOC] HELPER: Returns False for cpu/meta devices; for other devices validates against
    # the traced current accelerator (raising RuntimeError if rank-specific) and returns True.
    """Whether ``device`` should be replaced by the current-device node under CooR.

    True for the current accelerator (bare ``cuda`` or its current index); False for portable
    ``cpu``/``meta``; raises for a rank-specific device.
    """
    if device.type in ("cpu", "meta"):
        return False
    _coor_check_current_accelerator(device, _coor_traced_current_accelerator(tracer))
    return True


def proxy_call(
    proxy_mode: ProxyTorchDispatchMode,
    func: OpOverload,
    pre_dispatch: bool,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # [FUNCTION DOC] Core of proxy-mode dispatch for one operator call. Returns NotImplemented if any
    # tensor argument is of an unhandled subclass without a proxy. Otherwise tries registered
    # decompositions, then CompositeImplicit decomposition (not in pre-dispatch), special-cases
    # is_nonzero, and handles data-dependent ops (constant-evaluates if all inputs are constants, else
    # raises RuntimeError for real tensors when errors are enabled). Under compile-on-one-rank, device
    # operands are replaced by the current-device node. Creates a call_function node (lift_fresh is
    # rewritten to lift_fresh_copy), runs the real op with thunkification on to get outputs, computes a
    # constant for small outputs when inputs are constant (or for lift), tracks the outputs, and
    # returns them. Side effects: adds graph nodes and updates tracker state.
    unrecognized_types: list[type] = []
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))

    def can_handle_tensor(x: Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True if tensor x has an exact handled type, a tracked proxy, or (when fake
        # constants are allowed) is a FakeTensor; records the type of unhandled ones in the enclosing list.
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) is torch._subclasses.FakeTensor
        if not r:
            unrecognized_types.append(type(x))
        return r

    # If there are any tensor subclasses, we need to handle those tensor subclasses first
    # TODO: we could use types to test this
    if not all(can_handle_tensor(x) for x in flat_args_kwargs if isinstance(x, Tensor)):
        not_implemented_log.debug(
            "ProxyTensorMode tensors without proxy had unrecognized subclasses: %s",
            unrecognized_types,
        )
        return NotImplemented

    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        _maybe_record_pointwise_barrier(func, proxy_mode)
        return r

    # For pre-autograd tracing, we do not want to run CompositeImplicit decomps.
    if (
        not pre_dispatch
        and func
        not in [
            torch.ops.aten.size.default,
            torch.ops.aten.stride.default,
            torch.ops.aten.storage_offset.default,
        ]
        and autograd_would_have_decomposed(func, flat_args_kwargs)
    ):
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    if func is torch.ops.aten.is_nonzero.default:
        with proxy_mode:
            torch._check(
                args[0].numel() == 1,  # type: ignore[attr-defined]
                lambda: "Boolean value of Tensor with more than one value is ambiguous",
            )
            return (args[0] != 0).item()  # type: ignore[attr-defined]

    tracer = proxy_mode.tracer
    f_flat_args_kwargs, proxy_flat_args_kwargs, all_constant = (
        _fetch_proxies_and_all_constant_flag(flat_args_kwargs, tracer)
    )

    if torch.Tag.data_dependent_output in func.tags:
        # Check if all of the Tensor inputs are constants
        if all_constant:
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            with unset_fake_temporarily():
                return func(*const_args, **const_kwargs)
        # If any of the Tensor inputs are "real" (not FakeTensor), we may
        # incorrectly burn in constants by allowing this access.  Raise
        # an error in this case
        if proxy_mode._error_on_data_dependent_ops and pytree.tree_all_only(
            Tensor, lambda t: not is_fake(t), (args, kwargs)
        ):
            raise RuntimeError(
                f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! "
                "It's likely that this is caused by data-dependent control flow or similar.  "
                "It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' "
                "in your make_fx call."
            )

    # [device-as-parameter] Under compile-on-one-rank, replace a baked accelerator device
    # operand matching the current accelerator with a reference to a single current_device()
    # node, so the traced graph follows each rank's current device instead of the baked
    # device. Only the operand changes; the fake op below still runs with the real device.
    if _coor_enabled() and any(
        isinstance(e, torch.device) for e in proxy_flat_args_kwargs
    ):
        proxy_flat_args_kwargs = [
            _current_device_edge(tracer, e)
            if isinstance(e, torch.device)
            and _coor_match_current_accelerator(tracer, e)
            else e
            for e in proxy_flat_args_kwargs
        ]

    proxy_args, proxy_kwargs = pytree.tree_unflatten(proxy_flat_args_kwargs, spec)

    # When we trace through a torch.tensor invocation, you never actually
    # see a torch.ops.aten.tensor call. Instead, the way this function is
    # implemented internally is that we allocate a plain tensor (this is
    # *guaranteed* to be a plain tensor, we disable all modes when doing
    # so), and then call at::lift_fresh on it (to give modes a chance to do
    # their stuff).  Furthermore, the tensor argument to lift_fresh is guaranteed
    # to be freshly allocated, so we want lift_fresh to be a no-op (directly
    # returning the input argument).
    #
    # Here is the basic problem: when we trace this sequence of executions
    # into an FX graph, what happens to this call sequence?  Traditionally,
    # tensor constants get interned as buffers on the FX GraphModule.  But
    # this is dangerous.  Consider:
    #
    #       x = torch.tensor(1)
    #       x.add_(2)
    #
    # Naively, this traces into:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh(t)
    #       x.add_(2)
    #
    # If lift_fresh returns t directly, the subsequent add_ call will
    # modify the tensor constant. Really, the problem is we've violated
    # the invariant the argument to lift is fresh.  So what we should
    # preserve the invariant by replacing lift_fresh with lift_fresh_copy:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh_copy(t)
    #       x.add_(2)
    #
    # This is what the overload modification does.
    if func is torch.ops.aten.lift_fresh.default:
        func = torch.ops.aten.lift_fresh_copy.default

    proxy_out = proxy_mode.tracer.create_proxy(
        "call_function",
        func,
        proxy_args,
        proxy_kwargs,
        name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__),
    )

    with _enable_thunkify(proxy_mode.tracer):
        out = func(*args, **kwargs)

    # In some circumstances, we will be tracing in a situation where a tensor
    # is *statically* known to be a constant (currently, this only happens if
    # you run torch.tensor; deterministic factory functions like torch.arange
    # don't get this treatment).  When the tensor in question is small, it's
    # helpful to due constant propagation in case we call item() (in which
    # case we can return the constant value that is known, rather than give
    # an error.)  The logic here tests if constant propagation is possible
    # (because all of the inputs are constant).  If so, we disable fake tensor
    # mode (if it is on) and do true compute on the constant.
    #
    # It's worth highlighting that we're making a policy decision here.
    # There is a potential that the tensor is actually quite large, and we
    # don't actually want to run the compute.  The tensor being quite large
    # is one of the reasons why factory functions don't get this treatment
    # (since they can be quite large; if a parameter is initialized to a
    # constant value it will be!)  Similarly, there is also a potential
    # to run an operator that blows up the size of a small tensor; we don't
    # protect against this case, but we could force, e.g., only single
    # element constant computation by testing the numel of the result before
    # propagating const-ness.  Similarly, we don't require the constant to
    # live on CPU, but we could.
    any_constant = any(
        t.constant is not None
        for t in f_flat_args_kwargs
        if isinstance(t, _ProxyTensor)
    )

    constant = None

    def tensor_numel_in_limit(t: Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True if the tensor has at most CONSTANT_NUMEL_LIMIT elements.
        return t.numel() <= CONSTANT_NUMEL_LIMIT

    # If this is a lift, the input tensor is guaranteed to be a
    # constant, so we keep a copy of the original argument along so
    # we can query it if we're asked to item() it at some later point
    if (
        func is torch.ops.aten.lift_fresh_copy.default
        and out.numel() <= CONSTANT_NUMEL_LIMIT
    ):
        with unset_fake_temporarily():
            if not isinstance(args[0], (Proxy, Tensor)):
                raise AssertionError(f"Expected Proxy or Tensor, got {type(args[0])}")
            constant = args[0].clone()
    elif (
        torch.Tag.nondeterministic_seeded not in func.tags
        and all_constant
        and any_constant
        and pytree.tree_all_only(Tensor, tensor_numel_in_limit, out)
    ):
        # NB: do NOT include factories as constants
        with unset_fake_temporarily():
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None

    track_tensor_tree(
        out,
        proxy_out,
        # pyrefly: ignore[bad-argument-type]
        constant=constant,
        tracer=tracer,
    )
    _maybe_record_pointwise_barrier(func, proxy_mode)
    return out


class _SymNodeDict:
    # [CLASS DOC] HELPER: Mapping-like wrapper keyed by symbolic scalars but storing entries under their
    # underlying SymNode, so different wrappers of one node share an entry. Supports set, get, contains,
    # length; iteration is not supported.
    """
    Wrapper around a dictionary that will hash SymInts with their nodes
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty inner dictionary.
        self.sym_node_dict: dict[PySymType, _PySymProxyType] = {}

    def __setitem__(self, key: PySymType, value: _PySymProxyType) -> None:
        # [FUNCTION DOC] HELPER: Stores value under the node of key.
        # pyrefly: ignore [unsupported-operation]
        self.sym_node_dict[key.node] = value

    def __getitem__(self, key: PySymType) -> _PySymProxyType:
        # [FUNCTION DOC] HELPER: Returns the value stored for the node of key; raises KeyError if missing.
        # pyrefly: ignore [bad-index]
        return self.sym_node_dict[key.node]

    def __contains__(self, key: PySymType) -> bool:
        # [FUNCTION DOC] HELPER: True if the node of key has an entry.
        return key.node in self.sym_node_dict

    def get(
        self, key: PySymType, default: _PySymProxyType | None = None
    ) -> _PySymProxyType:
        # [FUNCTION DOC] HELPER: Returns the value stored for the node of key, or default if missing.
        # dict.get()'s annotation doesn't accept `None` when the value type
        # isn't Optional.
        return self.sym_node_dict.get(key.node, default)  # type: ignore[arg-type, return-value]

    def __iter__(self) -> Any:
        # [FUNCTION DOC] HELPER: Not supported; always raises NotImplementedError.
        raise NotImplementedError

    def __len__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the number of stored entries.
        return len(self.sym_node_dict)


@dataclass
class _SympyExprTrackerValue:
    # [CLASS DOC] HELPER: Record in the sympy expression tracker: proxy is the thunk producing the
    # graph proxy for a symbol and value is the symbolic scalar it came from.
    proxy: _PySymProxyType
    value: PySymType


def _init_proxy_trackers(tracer: PythonKeyTracer | _GraphAppendingTracerEx) -> None:
    # [FUNCTION DOC] HELPER: Sets up empty trackers on the tracer (tensor, symnode, script object,
    # opaque, real-object map, sympy expression) plus the current-device cache, torch function metadata
    # and counts, and disables thunkification. Mutates the tracer attributes.
    """Initialize the tracker dictionaries shared by PythonKeyTracer and _GraphAppendingTracerEx."""
    tracer.tensor_tracker = WeakTensorKeyDictionary()
    tracer.symnode_tracker = _SymNodeDict()
    tracer.script_object_tracker = WeakIdKeyDictionary(dict=None, ref_type=_WeakHashRef)
    tracer.opaque_tracker = WeakIdKeyDictionary()
    tracer._opaque_real_obj_proxy = {}
    tracer.sympy_expr_tracker = {}
    tracer._current_device_node = None
    tracer._coor_current_accelerator = None
    # Stores the torch function that was called during tracing
    tracer.torch_fn_metadata = None
    # Stores the counts for every torch function called. This is to help
    # distinguish between different calls to the same torch function.
    tracer.torch_fn_counts = {}
    tracer.enable_thunkify = False


class PythonKeyTracer(Tracer):
    # [CLASS DOC] fx Tracer used by make_fx in python-key mode. Holds per-trace trackers mapping
    # tensors, script objects, opaque objects and symbolic scalars to their proxies, plus the sympy
    # expression tracker, current-device node cache, torch function metadata/counts and the thunkify
    # flag. Does not make modules into leaves and returns attribute values directly. Key methods:
    # create_arg, create_node (adds metadata), unwrap_proxy.
    # ScriptObject uses _WeakHashRef because the same C++ IValue can produce
    # different Python wrapper objects, so Python id() won't match.
    script_object_tracker: MutableMapping[torch.ScriptObject, Proxy]
    # FakeScriptObject/CustomClassBase uses WeakIdRef because distinct objects that
    # are value-equal (e.g. primal vs tangent opaques) must be tracked separately.
    opaque_tracker: MutableMapping[FakeScriptObject | CustomClassBase, Proxy]
    # Maps id(real_obj) -> proxy for opaque FSOs, so that multiple FSO wrappers
    # of the same real object (e.g. primal vs tangent) resolve to one proxy.
    _opaque_real_obj_proxy: dict[int, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: dict[sympy.Symbol, _SympyExprTrackerValue]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    # [device-as-parameter] the single current_device() node for this graph (CooR)
    _current_device_node: Proxy | None
    # [device-as-parameter] current accelerator for this graph, read once (None = not yet)
    _coor_current_accelerator: torch.device | None
    torch_fn_metadata: OpOverload | None
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the base Tracer with no autowrap modules and sets up the trackers.
        super().__init__(autowrap_modules=())  # type: ignore[arg-type]
        _init_proxy_trackers(self)

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
        self,
        m: Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # [FUNCTION DOC] Never treats module m as a leaf: calls forward directly with args and kwargs and
        # returns its result.
        return forward(*args, **kwargs)

    # We don't want to turn getattr calls into proxies. So we just return the actual value.
    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
    ) -> object:
        # [FUNCTION DOC] Does not proxy attribute accesses: returns attr_val unchanged.
        return attr_val

    def create_arg(self, a: object) -> fx.node.Node:
        # [FUNCTION DOC] Converts a value into an fx argument. Parameters become get_attr nodes (reusing
        # the name of an existing named parameter, else registering a new _param_constant attribute on the
        # root, which mutates it). Symbolic scalars must have a constant node value and yield that value.
        # Untracked opaque objects are first tried via reconstruction from graph inputs; everything else is
        # delegated to the base class.
        # KNOWN ISSUE: the symbolic branch checks the constant of the underlying node but returns a
        # constant attribute of the symbolic scalar itself, which looks inconsistent.
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})

            qualname = self.get_fresh_qualname("_param_constant")
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})
        elif isinstance(a, py_sym_types):
            if a.node.constant is None:
                raise AssertionError("a.node.constant should not be None")
            # pyrefly: ignore [bad-return]
            return a.constant

        # Try reconstructing untracked opaque reference types from existing
        # graph inputs (e.g. derive a DeviceMesh submesh from its root mesh).
        if isinstance(a, (FakeScriptObject, CustomClassBase)):
            node = self._try_reconstruct_opaque(a)
            if node is not None:
                return node

        return super().create_arg(a)  # type: ignore[return-value]

    def _try_reconstruct_opaque(
        self, a: FakeScriptObject | CustomClassBase
    ) -> fx.node.Node | None:
        # [FUNCTION DOC] HELPER: Tries to rebuild an untracked opaque object through the reconstruct
        # function registered for its (symbolic opaque) type, using already tracked objects (matched by
        # real-object map, tracker, or equality with tracked fake objects) as inputs. Returns the resulting
        # graph node or None if not possible. Sets meta val if missing and registers the result for the
        # object and its real object (side effects on trackers).
        """Try to reconstruct an opaque object from existing graph inputs.

        When make_fx encounters an untracked opaque reference type (e.g. a
        DeviceMesh submesh captured by a backward closure), this method checks
        if the type has a registered reconstruct_fn that can derive the object
        from inputs already in the graph.  Returns an FX Node on success,
        None on failure (falls back to get_attr constant).
        """
        real_obj: CustomClassBase = a.real_obj if isinstance(a, FakeScriptObject) else a

        if not is_opaque_symbolic_type(type(real_obj)):
            return None

        reconstruct_fn = get_reconstruct_fn(type(real_obj))
        if reconstruct_fn is None:
            return None

        def get_tracked_proxy(obj: CustomClassBase) -> Proxy | None:
            # [FUNCTION DOC] HELPER: Returns the proxy tracked for obj by real-object id, tracker lookup, or
            # equality against the real objects of tracked fake script objects; None if not found.
            proxy = self._opaque_real_obj_proxy.get(id(obj))
            if proxy is not None:
                return proxy
            proxy = get_proxy_slot(obj, self, None)
            if proxy is not None:
                return proxy
            # The object may be identity-different but equal to a tracked
            # FSO's real_obj. This happens because maybe_to_fake_obj creates
            # a new DeviceMesh wrapper, so the FSO's real_obj is a distinct
            # object from the one held by submesh._root_mesh. Equality
            # comparison is needed, not identity.
            for tracked_obj, p in self.opaque_tracker.items():
                if not isinstance(tracked_obj, FakeScriptObject):
                    continue
                if tracked_obj.real_obj == obj:
                    return p
            return None

        result = reconstruct_fn(real_obj, get_tracked_proxy, self)
        if result is None:
            return None

        # The reconstruct_fn dispatches through a custom op with a Proxy
        # argument, which goes through Proxy.__torch_function__.  That path
        # creates a graph node but does NOT populate meta["val"] (unlike the
        # ProxyTorchDispatchMode.__torch_dispatch__ path which calls
        # track_tensor_tree).  Set it here so downstream consumers (e.g. the
        # min-cut partitioner) can classify the node correctly.
        if "val" not in result.node.meta:
            set_meta(result, a)

        # Also register for the *input* object so dedup works for re-encounters.
        set_proxy_slot(a, self, result)
        if id(real_obj) not in self._opaque_real_obj_proxy:
            self._opaque_real_obj_proxy[id(real_obj)] = result

        return result.node

    @overload
    def unwrap_proxy(self, e: Tensor) -> Proxy | Tensor: ...

    @overload
    def unwrap_proxy(self, e: PySymType) -> Proxy | PySymType: ...

    @overload
    def unwrap_proxy(self, e: _AnyScriptObjectType) -> Proxy | _AnyScriptObjectType: ...

    def unwrap_proxy(self, e: T) -> object:
        # [FUNCTION DOC] Returns the graph-side value for e: tensor record's proxy (or e if untracked),
        # symbolic scalar's forced proxy (or e), tracked script/opaque proxy (or e); other values are
        # returned unchanged.
        if isinstance(e, Tensor):
            return get_proxy_slot(e, self, e, lambda x: x.proxy)  # type: ignore[attr-defined]
        elif isinstance(e, py_sym_types):
            return get_proxy_slot(e, self, e, lambda e: e.force())
        elif isinstance(e, _AnyScriptObject) or is_custom_class_obj(e):
            return get_proxy_slot(e, self, e)
        else:
            return e

    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> torch.fx.Node:
        # [FUNCTION DOC] Creates a node via the base class, drops stack_trace from placeholder/output
        # nodes, sets meta val for get_attr nodes pointing to tensors (with proxy tracing disabled), and for
        # nodes needing it stores meta eager_input_vals, the fake values of the original args and kwargs.
        # Returns the node.
        node = super().create_node(kind, target, args, kwargs, name, type_expr)  # type: ignore[arg-type]

        if node.op in ["placeholder", "output"] and "stack_trace" in node.meta:
            del node.meta["stack_trace"]

        if kind == "get_attr":
            if not isinstance(target, str):
                raise AssertionError(f"Expected str, got {type(target)}")
            attr = getattr(self.root, target)
            if isinstance(attr, torch.Tensor):
                with disable_proxy_modes_tracing():
                    node.meta["val"] = extract_val(attr)

        def map_fn(v: Any) -> _ExtractValType | None:
            # [FUNCTION DOC] HELPER: Maps an fx node argument to its meta val snapshot, or None if the node has
            # no val or holds a non-fake tensor.
            if not isinstance(v, torch.fx.Node) or "val" not in v.meta:
                return None
            val = v.meta["val"]
            # other subclasses like FunctionalTensor error on `extract_val`
            # "Attempting to use FunctionalTensor on its own." just store FakeTensors for now
            if isinstance(val, torch.Tensor) and not is_fake_tensor(val):
                return None
            return extract_val(v.meta["val"])

        if _should_save_eager_input_vals(target, (args, kwargs)):
            # NOTE "eager_input_vals"
            # We save the original (args, kwargs) FakeTensor values for nodes
            # that have exact stride requirements. This is useful downstream.
            # We use this information inside Inductor to ensure that inputs to
            # stride-sensitive operators have the correct strides.
            arg_inp, kwarg_inp = torch.fx.node.map_aggregate((args, kwargs), map_fn)  # type: ignore[misc, arg-type]
            node.meta["eager_input_vals"] = (arg_inp, kwarg_inp)

        return node


def _should_save_eager_input_vals(
    target: Any,
    args_kwargs: tuple[tuple[Argument, ...], dict[str, Argument]] | None = None,
) -> bool:
    # [FUNCTION DOC] HELPER: Decides whether a node with this target should record eager_input_vals:
    # True for triton kernel wrapper HOPs and invoke_subgraph; for auto_functionalized variants it
    # recurses on the wrapped operator (first positional argument, which must be an op or HOP, else
    # AssertionError); non-callable targets give False.
    from torch._higher_order_ops.invoke_subgraph import InvokeSubgraphHOP

    if not callable(target):
        return False
    if isinstance(
        target,
        (
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperMutation,
            InvokeSubgraphHOP,
        ),
    ):
        return True
    if args_kwargs is not None and (
        target is torch.ops.higher_order.auto_functionalized
        or target is torch.ops.higher_order.auto_functionalized_v2
    ):
        args = args_kwargs[0]
        if not isinstance(
            args[0], (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        ):
            raise AssertionError(
                f"Expected OpOverload or HigherOrderOperator, got {type(args[0])}"
            )
        return _should_save_eager_input_vals(args[0], None)
    if target is torch.ops.higher_order.with_effects:
        # TODO: inductor lowering for with_effects needs to be updated to propagate
        # the arg_kwarg_vals
        return False
    if isinstance(target, torch._ops.HigherOrderOperator):
        if pytree.tree_any(_should_save_eager_input_vals, args_kwargs):
            raise RuntimeError(
                f"NYI: The HOP {target} has an input that is an OpOverload that "
                f"needs exact strides. We probably need special logic to "
                f"propagate the FakeTensor vals. Please file an issue."
            )
    if isinstance(target, torch._ops.OpOverload):
        from torch._library.utils import get_layout_constraint_tag

        return get_layout_constraint_tag(target) == torch._C.Tag.needs_exact_strides
    return False


def _make_temp_remove_mode_context_manager(
    mode_ty: type[TorchFunctionMode],
) -> Callable[[], _GeneratorContextManager[TorchFunctionMode | None]]:
    # [FUNCTION DOC] HELPER: Returns a context-manager factory. The context manager pops torch function
    # modes off the global stack until it finds one of type mode_ty, restores the others, and yields the
    # removed mode (or None if none found). On exit, if a mode was removed it is put back at its original
    # position in the stack. Mutates the global torch function mode stack.
    @contextmanager
    def context_manager_fn() -> Generator[TorchFunctionMode | None, None, None]:
        # [FUNCTION DOC] HELPER: The context manager produced by the factory; see the enclosing function.
        from torch.overrides import _len_torch_function_stack, _pop_mode, _push_mode

        temp_elements = []
        removed_mode = None

        while _len_torch_function_stack() > 0:
            mode = _pop_mode()
            if isinstance(mode, mode_ty):
                removed_mode = mode
                break
            else:
                temp_elements.append(mode)

        for mode in reversed(temp_elements):
            _push_mode(mode)

        try:
            yield removed_mode

        finally:
            if removed_mode is not None:
                count = len(temp_elements)
                while count > 0:
                    mode = _pop_mode()
                    count -= 1

                temp_elements.append(removed_mode)

                for mode in reversed(temp_elements):
                    _push_mode(mode)

    return context_manager_fn


@torch._disable_dynamo
def dispatch_trace(
    root: Module | Callable[..., Any],
    tracer: Tracer,
    concrete_args: tuple[object, ...] | None = None,
) -> GraphModule:
    # [FUNCTION DOC] Traces root with the given tracer (concrete_args optional) into a graph, removes
    # dead code using a purity predicate that keeps impure nodes and .item()-style calls but allows
    # removing accessor nodes and symbolic-int-to-symbolic-int computations, deduplicates symbolic int
    # uses, and returns a GraphModule named after the root's class or function. Dynamo is disabled for
    # this function.
    graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]

    # NB: be careful not to DCE .item() calls
    def impure_pred(n: fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: Returns True when node n must be kept by dead code elimination: impure
        # nodes, and anything not recognized as an accessor or as a symbolic computation over symbolic
        # inputs.
        from .symbolic_shapes import is_accessor_node

        # Always defer to the built-in notion of impure
        if n.is_impure():
            return True

        # Accessors always OK to DCE
        if is_accessor_node(n):
            return False

        # If the operator in question takes SymInt args to SymInt output,
        # we assume it's pure and OK to DCE
        if (
            isinstance(n.meta.get("val"), py_sym_types)
            and
            # NB: constant args ok
            all(
                isinstance(a.meta.get("val"), py_sym_types)
                for a in n.args
                if isinstance(a, fx.Node)
            )
        ):
            return False

        # No idea, just assume it's not OK
        return True

    graph.eliminate_dead_code(impure_pred)
    from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints

    dedupe_symints(graph)
    name = root.__class__.__name__ if isinstance(root, Module) else root.__name__
    return fx._lazy_graph_module._make_graph_module(tracer.root, graph, name)


def wrap_key(
    f: Callable[[Unpack[_Ts]], R],
    tensors: tuple[Unpack[_Ts]],
    tracer: _ProxyTracer,
    pre_dispatch: bool,
) -> Callable[_P, R]:
    # [FUNCTION DOC] Wraps f so it can be called with proxies in place of the real example tensors.
    # The returned function registers the proxies for the example inputs (tracking tensors, scalars'
    # meta), checks under compile-on-one-rank that input devices are the current accelerator, optionally
    # replaces module inputs by their proxies, calls f on the original example values, and converts the
    # outputs (tensors, script objects, custom classes, symbolic scalars) back to their proxies.
    # Raises AssertionError if proxy and tensor counts differ.
    flat_tensors, _tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies: _P.args, **_unused: _P.kwargs) -> R:
        # [FUNCTION DOC] HELPER: The wrapper described in wrap_key; takes proxies (extra keyword arguments
        # are ignored) and returns f's output mapped to proxies. May rebind the enclosing tensors list.
        nonlocal tensors

        flat_proxies, _proxies_spec = pytree.tree_flatten(proxies)
        if len(flat_proxies) != len(flat_tensors):
            raise AssertionError(
                f"Expected same length: {len(flat_proxies)} vs {len(flat_tensors)}"
            )
        with disable_proxy_modes_tracing() as m:
            if not isinstance(m, ProxyTorchDispatchMode):
                raise AssertionError(f"Expected ProxyTorchDispatchMode, got {type(m)}")
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)
            # `wrap_key` is the make_fx input-placeholder path. Keep scalar
            # metadata here instead of teaching recursive output tracking to
            # treat arbitrary primitive leaves as proxy-trackable values.
            for input_value, proxy in zip(flat_tensors, flat_proxies):
                if isinstance(input_value, (int, float, bool)) and isinstance(
                    proxy, fx.Proxy
                ):
                    set_meta(proxy, input_value)

        # [device-as-parameter] Under compile-on-one-rank a graph input on a non-current
        # accelerator is rank-specific and can't be made device-agnostic (the wrapper
        # resolves a single current device at runtime, so the whole graph must live on it).
        # Inductor's pattern-matcher templates are traced with CooR off, so this only sees
        # real user graphs, not internal fixed-device template traces.
        if _coor_enabled():
            cur = _coor_traced_current_accelerator(tracer)
            for input_value in flat_tensors:
                if isinstance(input_value, torch.Tensor):
                    _coor_check_current_accelerator(input_value.device, cur)

        if getattr(tracer, "proxy_module_inputs", False):
            tensors = [  # type: ignore[assignment, var-annotated]
                p if isinstance(t, torch.nn.Module) else t
                for t, p in zip(tensors, proxies)  # type: ignore[arg-type]
            ]

        def get_tensor_proxy_slot(t: Tensor) -> Tensor | Proxy:
            # [FUNCTION DOC] HELPER: Returns the proxy tracked for tensor t, or t itself if untracked.
            return get_proxy_slot(t, tracer, t, lambda x: x.proxy)  # type: ignore[attr-defined]

        out = f(*tensors)  # type:ignore[call-arg]
        out = pytree.tree_map_only(Tensor, get_tensor_proxy_slot, out)
        out = pytree.tree_map_only(
            _AnyScriptObject, lambda t: get_proxy_slot(t, tracer, t, lambda x: x), out
        )
        out = pytree.tree_map_only(
            CustomClassBase, lambda t: get_proxy_slot(t, tracer, t, lambda x: x), out
        )

        def get_sym_proxy_slot(t: PySymType) -> Proxy:
            # [FUNCTION DOC] HELPER: Returns the forced proxy for symbolic scalar t; raises if untracked.
            return get_proxy_slot(t, tracer).force()

        out = pytree.tree_map_only(py_sym_types, get_sym_proxy_slot, out)
        return out

    return wrapped


# TODO: Make downstream users of this work with OperatorBase
ORIGINAL_ATEN: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "ORIGINAL_ATEN", default=None
)


@contextmanager
def set_original_aten_op(
    func: OpOverload | torch._ops.HigherOrderOperator,
) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that, when no original aten op is yet recorded and node
    # meta preservation is on, records func in a context variable and in the current preserved meta as
    # original_aten for the duration, resetting both afterwards; otherwise does nothing.
    if ORIGINAL_ATEN.get() is None and fx_traceback.has_preserved_node_meta():
        token = ORIGINAL_ATEN.set(func)
        fx_traceback.current_meta["original_aten"] = func
        try:
            yield
        finally:
            ORIGINAL_ATEN.reset(token)
            fx_traceback.current_meta["original_aten"] = None
    else:
        yield


class TorchFunctionMetadataMode(TorchFunctionMode):
    # [CLASS DOC] Torch function mode that records, on its tracer, the last torch function called and a
    # per-function call count, and rewrites a size call on a tensor with symbolic sizes into a symbolic
    # size call. State: tracer.
    def __init__(self, tracer: _ProxyTracer) -> None:
        # [FUNCTION DOC] Stores the tracer.
        self.tracer = tracer

    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        # [FUNCTION DOC] Records func as the tracer's torch function metadata and increments its count.
        # Calls on size of a tensor with any symbolic dimension are redirected to sym_size; everything else
        # runs func normally and its result is returned.
        kwargs = kwargs or {}
        # pyrefly: ignore [bad-assignment]
        self.tracer.torch_fn_metadata = func
        self.tracer.torch_fn_counts[func] = self.tracer.torch_fn_counts.get(func, 0) + 1
        if (
            func is torch.ops.aten.size.default
            and len(args) == 1
            and isinstance(args[0], Tensor)
            and any(isinstance(s, py_sym_types) for s in args[0].size())
        ):
            return torch.ops.aten.sym_size.default(*args, **kwargs)
        return func(*args, **kwargs)


_temp_remove_metadata_torch_function_mode = _make_temp_remove_mode_context_manager(
    TorchFunctionMetadataMode
)


# This mode is **only** used for pre_dispatch tracing.
# In particular, we need to make sure that autograd/autocast API's
# that do not desugar into dispatcher operators stay in the graph.
class PreDispatchTorchFunctionMode(TorchFunctionMode):
    # [CLASS DOC] Torch function mode used only for pre-dispatch tracing. It keeps in the graph the
    # autograd/autocast/grad-mode and functorch (vmap, jvp, dual) API calls that do not become
    # dispatcher operators. State: tracer and a stack of enter_autocast nodes used to link exit_autocast
    # nodes to their matching enter node.
    def __init__(self, tracer: _ProxyTracer) -> None:
        # [FUNCTION DOC] Stores the tracer and creates the empty stack of enter-autocast nodes.
        self.tracer = tracer
        # The input to torch.amp.autocast_mode._exit_autocast graph node should be the
        # enter_autocast node. So we have to save the enter autocast node here, and assign it
        # to the exit_autocast call_function node.
        self.enter_autocast_nodes: list[torch.fx.Node] = []

    def __torch_function__(
        self,
        func: OpOverload | Callable[..., Any],
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        # [FUNCTION DOC] For side-effectful functions that must be preserved (grad mode, autocast enter/exit
        # and similar) creates a call_function node (exit_autocast gets its matching enter node as argument,
        # meta val set to None for grad/autocast ones; set_grad_enabled is also actually executed) and
        # returns the node. For the listed functorch predispatch functions it creates a proxy from the
        # fetched argument proxies, runs the function, tracks the result (with functorch transforms
        # disabled for jvp functions) and returns the real result. All other functions just run.
        kwargs = kwargs or {}
        if func in _side_effectful_need_to_be_preserved_pre_dispatch:
            # It's for passing the export verifier which needs to verify the meta['val']
            # TODO(tmanlaibaatar): we should systematically couple it with export verifier,
            # instead of hardcoding it here.
            # T203648563
            if func is torch.amp.autocast_mode._exit_autocast:
                enter_node = self.enter_autocast_nodes.pop()
                args = (enter_node,)
            node = self.tracer.create_node("call_function", func, args, {})  # type: ignore[arg-type]
            if func is torch.amp.autocast_mode._enter_autocast:
                self.enter_autocast_nodes.append(node)
            if func in [
                torch._C._set_grad_enabled,
                torch.amp.autocast_mode._enter_autocast,
                torch.amp.autocast_mode._exit_autocast,
            ]:
                node.meta["val"] = None
            # For autocast, the python APIs run so we don't have to run them again
            # here.
            if func is torch._C._set_grad_enabled:
                # pyrefly: ignore [bad-argument-type]
                func(*args, **kwargs)
            return node
        # We need more complicated handling here because the inputs
        # to these functions are sometimes tensors or symints where
        # we need to fetch the proxies properly.
        if func in [
            torch._functorch.predispatch._add_batch_dim,
            torch._functorch.predispatch._remove_batch_dim,
            torch._functorch.predispatch._vmap_increment_nesting,
            torch._functorch.predispatch._vmap_decrement_nesting,
            torch._functorch.vmap.lazy_load_decompositions,
            torch._functorch.predispatch._make_dual,
            torch._functorch.predispatch._unpack_dual,
            torch._functorch.predispatch._jvp_increment_nesting,
            torch._functorch.predispatch._jvp_decrement_nesting,
            torch._functorch.predispatch._unwrap_for_grad,
            torch._functorch.predispatch._enter_dual_level,
            torch._functorch.predispatch._exit_dual_level,
        ]:
            _, proxies, _ = _fetch_proxies_and_all_constant_flag(args, self.tracer)
            out_proxy = self.tracer.create_proxy(
                "call_function",
                func,
                proxies,
                kwargs,
            )
            res = func(*args, **kwargs)
            # When JVP transforms are active, snapshot_fake calls detach
            # which goes through C++ functorch dispatch keys, potentially
            # re-wrapping the result as a grad tracking tensor.
            # Temporarily disable functorch transforms during tracking to
            # prevent this corruption of meta["val"].
            if func in _jvp_predispatch_functions:
                with torch._C._DisableFuncTorch():
                    track_tensor_tree(res, out_proxy, constant=None, tracer=self.tracer)
            else:
                track_tensor_tree(res, out_proxy, constant=None, tracer=self.tracer)
            return res
        return func(*args, **kwargs)


_temp_remove_pre_dispatch_torch_function_mode = _make_temp_remove_mode_context_manager(
    PreDispatchTorchFunctionMode
)


# JVP predispatch functions need special handling during tracing:
# _DisableFuncTorch prevents grad tracking tensor corruption in snapshot_fake.
_jvp_predispatch_functions = frozenset(
    {
        torch._functorch.predispatch._make_dual,
        torch._functorch.predispatch._unpack_dual,
        torch._functorch.predispatch._jvp_increment_nesting,
        torch._functorch.predispatch._jvp_decrement_nesting,
        torch._functorch.predispatch._unwrap_for_grad,
        torch._functorch.predispatch._enter_dual_level,
        torch._functorch.predispatch._exit_dual_level,
    }
)

# These JVP predispatch wrappers manage transform nesting/level state
# and must survive FX dead code elimination even with zero output users.
# Registered here rather than in predispatch.py to avoid circular imports
# (predispatch.py loads before torch.fx during torch.autograd init).
from torch.fx.node import _side_effectful_functions


_side_effectful_functions.update(
    {
        torch._functorch.predispatch._jvp_increment_nesting,
        torch._functorch.predispatch._jvp_decrement_nesting,
        torch._functorch.predispatch._enter_dual_level,
        torch._functorch.predispatch._exit_dual_level,
    }
)


class ProxyTorchDispatchMode(TorchDispatchMode):
    # Ensure this is read-only; this exists only for legacy reasons
    @property
    def enable_tracing(self) -> bool:
        # [FUNCTION DOC] Read-only legacy property; always True.
        return True

    def __init__(
        self,
        tracer: _ProxyTracer,
        tracing_mode: _TracingMode,
        pre_dispatch: bool = False,
        _allow_fake_constant: bool = False,
        _error_on_data_dependent_ops: bool = True,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
    ) -> None:
        # [FUNCTION DOC] Creates the infra proxy mode: stores tracer, tracing mode, pre_dispatch flag,
        # fake-constant and data-dependent-error options and the decomposition table, sets up the stack of
        # previous modes and decomposition tables, decomposition layer count, invoke_subgraph name/cache
        # state, and reads emulate_precision_casts from the inductor config. Uses the PreDispatch key when
        # pre_dispatch is set.
        dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None
        super().__init__(dk)
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.pre_dispatch = pre_dispatch
        self._allow_fake_constant = _allow_fake_constant
        self._error_on_data_dependent_ops = _error_on_data_dependent_ops
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.PROXY
        # Every time we enter a mode, we maintain a stack telling us what the previous
        # ProxyTorchDispatchMode state was (if there was any).
        # This lets us properly reset the state on exit.
        self.enter_stack: list[ProxyTorchDispatchMode | None] = []
        self.decomposition_table: Mapping[OpOverload, Callable[..., Any]] = (
            decomposition_table or {}
        )
        self._decomposition_table_stack: list[
            Mapping[OpOverload, Callable[..., Any]]
        ] = []
        self.decomp_layers: int = 0
        # See invoke_subgraph
        self._invoke_subgraph_names: set[str] = set()
        self._invoke_subgraph_cache: dict[
            torch.fx.GraphModule | FunctionalizeCtxWrapper, str
        ] = {}
        from torch._inductor import config

        self.emulate_precision_casts: bool = config.emulate_precision_casts

    @count
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        # [FUNCTION DOC] Dispatch entry point (counted for stats). Under a context recording the original
        # aten op, returns the plain call result for the device primitive and otherwise delegates to
        # proxy_call. Returns the operator result or NotImplemented.
        with set_original_aten_op(func):
            kwargs = kwargs or {}

            if func == prim.device.default:
                return func(*args, **kwargs)

            return proxy_call(self, func, self.pre_dispatch, args, kwargs)

    def __enter__(self) -> Self:
        # [FUNCTION DOC] Removes any currently active proxy infra mode, remembers it on enter_stack, then
        # enters this mode. Returns self.
        # Stash and store the previous proxy mode (there may or may not be one)
        maybe_prev_proxy_mode = _unset_infra_mode(torch._C._TorchDispatchModeKey.PROXY)
        self.enter_stack.append(maybe_prev_proxy_mode)
        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        # [FUNCTION DOC] Exits the mode, then re-activates the proxy mode that was active before entering
        # (if any). Returns the base class result.
        b = super().__exit__(exc_type, exc_value, traceback)

        # Re-enable the previous proxy mode, if there was one.
        mb_previous_proxy_mode = self.enter_stack.pop()
        if mb_previous_proxy_mode is not None:
            _push_mode(mb_previous_proxy_mode)

        return b

    @classmethod
    def is_infra_mode(cls) -> bool:
        # [FUNCTION DOC] Class method; always True, marking this as an infra mode.
        return True

    @contextmanager
    def enable_decompositions(
        self,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
    ) -> Generator[Mapping[OpOverload, Callable[..., Any]], None, None]:
        # [FUNCTION DOC] Context manager that temporarily installs decomposition_table (empty if None) as the
        # active table, yields it, and restores the previous table on exit.
        table = decomposition_table or {}
        self._decomposition_table_stack.append(self.decomposition_table)
        self.decomposition_table = table
        try:
            yield table
        finally:
            self.decomposition_table = self._decomposition_table_stack.pop()

    def __sym_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        # [FUNCTION DOC] Handles symbolic-scalar operations: multiplication by the integer 1 returns the other
        # operand without tracing; otherwise calls func on args (keyword arguments are not allowed, raising
        # AssertionError), registers the result as a traced symbolic computation and returns it.
        # Peephole optimize multiply by one
        # NB: be careful not to trigger guards here!
        if func is operator.mul:
            if isinstance(args[1], int) and args[1] == 1:
                return args[0]
            elif isinstance(args[0], int) and args[0] == 1:
                return args[1]

        # For speed, we assume there are no nested data structures
        # (otherwise we could use tree_map)
        # We also assume there are no keyword arguments.
        if kwargs:
            raise AssertionError("Expected no keyword arguments")
        out = func(*args, **kwargs)
        _sym_register(self.tracer, func, args, out)
        return out


def _sym_register(
    tracer: _ProxyTracer, func: OpOverload, args: tuple[object, ...], out: object
) -> None:
    # [FUNCTION DOC] HELPER: If out is a symbolic scalar, records a lazily computed proxy for it, derived
    # from func and args, in the tracer's symnode tracker. Non-symbolic results are ignored.
    # If func returned a constant, we don't need to trace; we have
    # determined that the result is constant (no matter if the inputs
    # were symbolic) and it is no longer necessary to trace the
    # computation.  This could occur if func triggered some guards.
    if isinstance(out, py_sym_types):
        p_out_thunk = thunkify(
            tracer, _compute_proxy, tracer, func=func, args=args, out=out
        )
        set_proxy_slot(out, tracer, p_out_thunk)


def _compute_proxy(
    tracer: _ProxyTracer, func: OpOverload, args: tuple[object, ...], out: PySymType
) -> Proxy:
    # [FUNCTION DOC] HELPER: Builds the proxy for a symbolic operation by creating a call_function node
    # for func whose symbolic arguments are replaced by their forced proxy nodes (a single list argument
    # is treated as the argument list of a sum), wraps the node in a Proxy, sets meta from out and returns
    # it. Adds a graph node.
    # Handle torch.sym_sum
    n_args: tuple[object, ...]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        n_args = (
            tuple(
                (
                    get_proxy_slot(a, tracer).force().node
                    if isinstance(a, py_sym_types)
                    else a
                )
                for a in args[0]
            ),
        )
    else:
        n_args = tuple(
            (
                get_proxy_slot(a, tracer).force().node
                if isinstance(a, py_sym_types)
                else a
            )
            for a in args
        )

    # func doesn't have a __torch_function__ that Proxy can interpose, so
    # we gotta do it manually
    n_out = tracer.create_node("call_function", func, n_args, {})  # type: ignore[arg-type]
    p_out = fx.Proxy(n_out, tracer)
    set_meta(p_out, out)
    return p_out


class _GraphAppendingTracerEx(fx.proxy.GraphAppendingTracer):
    # [CLASS DOC] Tracer that appends to an existing graph (instead of building its own) and carries the
    # same tracker attributes as PythonKeyTracer: tensor, symnode, script object, opaque and sympy
    # expression trackers, current-device cache, torch function metadata/counts and thunkify flag.
    script_object_tracker: MutableMapping[torch.ScriptObject, Proxy]
    opaque_tracker: MutableMapping[FakeScriptObject | CustomClassBase, Proxy]
    # Maps id(real_obj) -> proxy for opaque FSOs, so that multiple FSO wrappers
    # of the same real object (e.g. primal vs tangent) resolve to one proxy.
    _opaque_real_obj_proxy: dict[int, Proxy]
    symnode_tracker: _SymNodeDict
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    sympy_expr_tracker: dict[sympy.Symbol, _SympyExprTrackerValue]
    # [device-as-parameter] the single current_device() node for this graph (CooR)
    _current_device_node: Proxy | None
    # [device-as-parameter] current accelerator for this graph, read once (None = not yet)
    _coor_current_accelerator: torch.device | None
    torch_fn_metadata: OpOverload | None
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self, graph: fx.graph.Graph) -> None:
        # [FUNCTION DOC] Initializes the base tracer with graph and sets up the empty trackers.
        super().__init__(graph)
        _init_proxy_trackers(self)


# TODO: I'm not sure what the point of this class is; you can just
# make_fx through a regular Interpreter
class DecompositionInterpreter(fx.Interpreter):
    # [CLASS DOC] fx Interpreter that re-runs a module under a proxy dispatch mode, appending the
    # resulting (decomposed) operations to new_graph. State: new_graph, tracer, decomposition table and
    # mode. Placeholders and attributes become graph inputs/attributes tracked by the tracer; outputs are
    # unwrapped to graph nodes.
    # KNOWN ISSUE: a TODO above the class questions whether this class is needed at all.
    def __init__(
        self,
        module: fx.GraphModule,
        new_graph: fx.Graph,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
        **kwargs: object,
    ) -> None:
        # [FUNCTION DOC] Sets up the interpreter for module, creating the tracer on new_graph and a
        # real-mode proxy dispatch mode using the decomposition table (empty if None).
        super().__init__(module, **kwargs)  # type: ignore[arg-type]
        self.new_graph = new_graph
        self.tracer = _GraphAppendingTracerEx(self.new_graph)
        # Blegh
        self.decomposition_table = decomposition_table or {}
        self.mode = ProxyTorchDispatchMode(
            self.tracer,
            tracing_mode="real",
            decomposition_table=self.decomposition_table,
        )

    # pyrefly: ignore [bad-override]
    def placeholder(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        # [FUNCTION DOC] Runs the base placeholder, adds a placeholder node to the new graph, tracks the result
        # against it (scalars also get meta) and returns the real value.
        # KNOWN ISSUE: a TODO notes that targets starting with a star are not handled.
        out = super().placeholder(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        # Primitive scalar placeholders are real graph inputs. Record their
        # value on the placeholder itself without making recursive output
        # tracking treat all primitive leaves as proxy-trackable values.
        if isinstance(out, (int, float, bool)):
            set_meta(proxy, out)
        # TODO handle case where the first character of target is '*'
        return out

    # pyrefly: ignore [bad-override]
    def get_attr(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        # [FUNCTION DOC] Runs the base get_attr, adds a get_attr node to the new graph, tracks the result
        # against it and returns the value.
        out = super().get_attr(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    # call_function, call_method, call_module get traced automatically by the outer mode.

    # pyrefly: ignore [bad-override]
    def output(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        # [FUNCTION DOC] Runs the base output, then emits the new graph output with tensors replaced by their
        # tracked graph nodes (untracked tensors kept as is). Returns the real value.
        out = super().output(target, args, kwargs)  # type: ignore[arg-type]

        def get_proxy_node(x: _ProxyTensor) -> fx.node.Node:
            # [FUNCTION DOC] HELPER: Returns the graph node of a proxy tensor record.
            return x.proxy.node

        def unwrap(e: Tensor) -> Tensor | fx.Node:
            # [FUNCTION DOC] HELPER: Returns the tracked graph node for tensor e, or e if untracked.
            return get_proxy_slot(e, self.tracer, e, get_proxy_node)

        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args: object, **kwargs: object) -> object:
        # [FUNCTION DOC] Runs the interpreter while the proxy dispatch mode is entered, so operations are
        # traced; returns the base run result.
        # Should enter the mode at least once for being able to restore it later
        # See: https://github.com/pytorch/pytorch/pull/82549#discussion_r934782025
        with self.mode:
            return super().run(*args, **kwargs)  # type: ignore[arg-type]


class _SelectiveDecomposeInterpreter(fx.Interpreter):
    # [CLASS DOC] HELPER: Interpreter that decomposes only the nodes for which should_decompose returns
    # True, using decomposition_table. State: should_decompose and decomposition_table. recursive_wrap
    # builds such interpreters for sub graph modules of higher order operators.
    def __init__(
        self,
        module: fx.GraphModule,
        should_decompose: Callable[[fx.Node], bool],
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
        **kwargs: object,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the predicate and decomposition table after base initialization.
        """
        For all nodes in `module`, selectively decompose if is `should_decompose`,
        following the given `decomposition_table`.
        """
        super().__init__(module, **kwargs)  # type: ignore[arg-type]
        self.should_decompose = should_decompose
        self.decomposition_table = decomposition_table

    @staticmethod
    def recursive_wrap(
        gm: fx.GraphModule,
        should_decompose: Callable[[fx.Node], bool],
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
        **kwargs: object,
    ) -> _SelectiveDecomposeInterpreter:
        # [FUNCTION DOC] HELPER: Static method. Rewrites the args of higher order operator call nodes in gm,
        # replacing each sub GraphModule argument by a recursively wrapped interpreter (mutating the node
        # args in place), then returns an interpreter for gm.
        """
        Recursively wrap gm and its sub graph modules. Specifically, HOP takes
        sub graph module as args. We may not want to decompose all nodes within
        these sub graph modules. So we also need to wrap these sub graph modules.
        As a result:
        - if should_decompose(hop) is True, we decompose all nodes within the hop.
        - if should_decompose(hop) is False, we check each node within the hop
            and decide whether decompose or not.
        """
        for node in gm.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, HigherOrderOperator
            ):
                new_args = []
                for arg in node.args:
                    if isinstance(arg, fx.GraphModule):
                        new_arg = _SelectiveDecomposeInterpreter.recursive_wrap(
                            arg, should_decompose, decomposition_table, **kwargs
                        )
                    else:
                        new_arg = arg
                    new_args.append(new_arg)
                node.args = tuple(new_args)

        return _SelectiveDecomposeInterpreter(
            gm, should_decompose, decomposition_table, **kwargs
        )

    def run_node(self, n: fx.Node) -> Any:
        # [FUNCTION DOC] HELPER: Runs node n, inside a decomposition context if should_decompose(n) is True,
        # otherwise normally; returns the node result.
        if self.should_decompose(n):
            with decompose(self.decomposition_table):
                result = super().run_node(n)
        else:
            result = super().run_node(n)
        return result


def selective_decompose(
    joint_gm: fx.GraphModule,
    *args: object,
    decomposition: Mapping[OpOverload, Callable[..., Any]] | None,
    should_decompose: Callable[..., bool],
    trace_joint_graph: bool,
) -> fx.GraphModule:
    # [FUNCTION DOC] Retraces joint_gm with make_fx (empty decomposition table) using an interpreter that
    # decomposes only nodes selected by should_decompose with the given decomposition table, and returns
    # the new GraphModule. When trace_joint_graph is True the traced function takes arguments named
    # primals and tangents (names relied on by the partitioner).
    # KNOWN ISSUE: in the joint case the traced function ignores its own primals and tangents parameters
    # and runs the interpreter on the outer args instead.
    """Retrace a joint graph module and selectively apply decomposition."""

    if trace_joint_graph:
        # the arg name, primals and tangents, are important.
        # make_fx keeps the name in the traced graph and partitioner later relies
        # on the name to partition joint graph correctly.
        def wrap_fn(primals: list[Any], tangents: list[Any]) -> Any:
            # [FUNCTION DOC] HELPER: Traced function taking lists primals and tangents; runs the selective
            # interpreter on the outer args and returns the result.
            return _SelectiveDecomposeInterpreter.recursive_wrap(
                joint_gm, should_decompose, decomposition
            ).run(*args)
    else:

        def wrap_fn(*args: Any) -> Any:
            # [FUNCTION DOC] HELPER: Traced function taking positional args; runs the selective interpreter on
            # them and returns the result.
            return _SelectiveDecomposeInterpreter.recursive_wrap(
                joint_gm, should_decompose, decomposition
            ).run(*args)

    return make_fx(wrap_fn, decomposition_table={})(*args)


def wrapper_and_args_for_make_fx(
    func: Callable[..., R], args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[Callable[[list[object]], R], list[object]]:
    # [FUNCTION DOC] Flattens args and kwargs together and returns a pair: a function taking a single flat
    # list that rebuilds the original args and kwargs and calls func, and the flat list. Needed because
    # make_fx does not support keyword arguments.
    # make_fx doesn't support kwargs, so we need to do this flattening
    # and then unflatten the args before calling func
    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args: list[object]) -> R:
        # [FUNCTION DOC] HELPER: Unflattens the list into args and kwargs and calls func.
        fn_args, fn_kwargs = pytree.tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)

    return wrapped, flat_args


@contextmanager
def disable_autocast_cache() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that disables the autocast cache and restores the previous setting on
    # exit, even on error.
    old_value = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(old_value)


class _ModuleNotInstalledAsSubmoduleError(NameError):
    # [CLASS DOC] HELPER: NameError subclass raised when a module being looked up is not installed as a
    # submodule of the traced root.
    pass


# Base class for inline _ModuleStackTracer.__init__.AttrProxy
class _AttrProxy:
    # [CLASS DOC] HELPER: Base class for the dynamically created AttrProxy inside _ModuleStackTracer; lets
    # code detect proxies by type and offers a no-op reset hook.
    def reset_proxy_mapping(self, base: Module, path: str) -> None:
        # [FUNCTION DOC] HELPER: No-op by default; subclasses may reset their recorded path mapping.
        pass


class _ModuleStackTracer(PythonKeyTracer):
    # [CLASS DOC] HELPER: PythonKeyTracer that preserves module stack information in node meta. It keeps
    # the original module (scope_root), a mapping from submodules to their first path, and, when a module
    # is shared in several places, turns on AttrProxy wrapping so the accessed path is tracked. State:
    # submodule_paths, proxy_paths, attr_proxy_map, proxy_modules, module_id_cache, enable_attr_proxy,
    # proxy_type.
    r"""Customized version of PythonKeyTracer that retains module stack
    information in node.meta["nn_module_stack"].

    FX symbolic trace actually does this already, but it relies on `self.root`
    being the actual module being traced. Since make_fx traces a lambda of our
    creation, things don't work properly.

    So for this version we hold onto a reference to the original module
    (scope_root) and use that to match the path. Also when we see,
            A
           / \
          B   C
           \ /
            D
    we want to record the path as A.B.D by recording only one path.
    See Note [Preserving the nn module stack metadata during export non-strict mode]  # noqa: W605
    """

    def __init__(self, scope_root: GraphModule) -> None:
        # [FUNCTION DOC] HELPER: Initializes the tracer for scope_root: records stack-trace settings, builds
        # the submodule path map (enabling AttrProxy if a module appears under several names), the weak maps
        # for proxies, a module id to names cache, and defines the AttrProxy class stored as proxy_type.
        super().__init__()
        self.record_stack_traces = not fx.config.do_not_emit_stack_traces
        self._record_forward_stack_traces_only = True
        self.scope_root = scope_root
        self.enable_attr_proxy = False
        self.submodule_paths = {}
        for name, m in self.scope_root.named_modules(remove_duplicate=False):
            if m in self.submodule_paths:
                log.info(
                    "Shared module found between %s and %s, AttrProxy is enabled.",
                    self.submodule_paths[m],
                    name,
                )
                self.enable_attr_proxy = True
            else:
                self.submodule_paths[m] = name

        self.proxy_paths: WeakKeyDictionary[_AttrProxy, str] = WeakKeyDictionary()
        self.attr_proxy_map: WeakKeyDictionary[Module, _AttrProxy] = WeakKeyDictionary()
        self.proxy_modules: WeakKeyDictionary[_AttrProxy, Module] = WeakKeyDictionary()
        self.counter = 0

        self.module_id_cache = defaultdict(list)
        for name, mod in self.scope_root.named_modules(remove_duplicate=False):
            self.module_id_cache[id(mod)].append(name)

        # Build a wrapper around _AttrProxy to provide the tracer. We can't
        # store it on _AttrProxy itself because we mimic the underlying class
        # (including its attributes).
        tracer = self

        class AttrProxy(_AttrProxy):
            # [CLASS DOC] HELPER: Dynamic proxy for a module that mimics the module's class and shares its
            # dictionary, while recording the access path in the tracer's maps so paths stay correct for shared
            # modules.
            def __init__(self, base: Module | _AttrProxy, path: str) -> None:
                # [FUNCTION DOC] HELPER: Wraps base (unwrapping an AttrProxy), turns self into an instance of a
                # dynamic subclass of the base class sharing its attribute dictionary, and records the path and the
                # base module in the tracer. Raises AssertionError if base is not a Module.
                if isinstance(base, _AttrProxy):
                    base = base.get_base()  # type: ignore[attr-defined]

                if not isinstance(base, Module):
                    raise AssertionError(f"Expected Module, got {type(base)}")
                # Class is modified to be a subclass of torch.nn.Module
                # Warning: We blow away our own attributes here to mimic the base class
                # - so don't expect `self.x` to do anything useful.
                # pyrefly: ignore [no-matching-overload]

                self.__class__ = type(
                    base.__class__.__name__,
                    (self.__class__, base.__class__),
                    {},
                )
                self.__dict__ = base.__dict__
                self.__class__.__module__ = base.__class__.__module__
                self.__class__.__qualname__ = base.__class__.__qualname__

                # This overwrites any existing paths if `base` is an AttrProxy
                tracer.proxy_paths[self] = path
                tracer.proxy_modules[self] = base

            def __getattr__(self, name: str) -> AttrProxy:
                # [FUNCTION DOC] HELPER: Looks up an attribute via the base module attribute logic; submodules are
                # returned wrapped with an extended path, other values unchanged.
                if not isinstance(self, Module):
                    raise AssertionError(f"Expected Module, got {type(self)}")
                # Calling into torch.nn.Module.__getattr__ with super(),
                # That __getattr__ is patched to be module_getattr_wrapper in _symbolic_trace.py.
                # which then calls into _ModuleStackTracer.getattr
                attr_val = super().__getattr__(name)  # type: ignore[misc]
                if not isinstance(attr_val, Module):
                    return attr_val

                return AttrProxy(attr_val, tracer.proxy_paths[self] + "." + name)

            def get_base(self) -> Module:
                # [FUNCTION DOC] HELPER: Returns the original wrapped module.
                return tracer.proxy_modules[self]

            def __getitem__(self, idx: int | slice) -> AttrProxy:
                # [FUNCTION DOC] HELPER: Indexing; slices of Sequential or ModuleList produce a new container wrapped
                # with a path including the slice, anything else delegates to the base class.
                if isinstance(idx, slice):
                    if isinstance(self, torch.nn.Sequential):
                        # Copied from nn/modules/container.py
                        res = torch.nn.Sequential(
                            OrderedDict(list(self._modules.items())[idx])
                        )

                        return AttrProxy(res, f"{tracer.proxy_paths[self]}.{idx}")
                    elif isinstance(self, torch.nn.ModuleList):
                        # Copied from nn/modules/container.py
                        res = torch.nn.ModuleList(list(self._modules.values())[idx])

                        return AttrProxy(res, f"{tracer.proxy_paths[self]}.{idx}")

                return super().__getitem__(idx)  # type: ignore[misc]

            @property
            def _modules(self) -> dict[str, AttrProxy]:
                # [FUNCTION DOC] HELPER: Property returning the submodule dictionary with each non-None module wrapped
                # in a path-extended proxy. Raises AssertionError if the dictionary is missing or not a dict.
                if "_modules" not in self.__dict__:
                    raise AssertionError("_modules not in __dict__")
                submodules = self.__dict__["_modules"]
                if not isinstance(submodules, dict):
                    raise AssertionError(f"Expected dict, got {type(submodules)}")
                # pyrefly: ignore [bad-return]
                return {
                    key: (
                        AttrProxy(value, tracer.proxy_paths[self] + "." + str(key))  # type: ignore[misc]
                        if value is not None
                        else value
                    )
                    for key, value in submodules.items()
                }

        self.proxy_type = AttrProxy

    def path_of_module(self, mod: Module) -> str:
        # [FUNCTION DOC] HELPER: Returns the empty string for scope_root, the recorded path for an AttrProxy,
        # otherwise the base class lookup; a NameError there is converted into
        # _ModuleNotInstalledAsSubmoduleError.
        """
        Use tracked access path during tracing instead of the default BFS behavior.
        Still use all the possible module paths to verify the result.
        """
        if mod is self.scope_root:
            return ""

        if isinstance(mod, _AttrProxy):
            return self.proxy_paths[mod]

        try:
            return Tracer.path_of_module(self, mod)
        except NameError as e:
            raise _ModuleNotInstalledAsSubmoduleError from e

    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
    ) -> object:
        # [FUNCTION DOC] HELPER: Returns the attribute unchanged to the base class path for non-modules, graph
        # modules, or when AttrProxy is disabled; otherwise returns an existing AttrProxy as is, or a cached
        # AttrProxy for the module (resetting its path mapping when re-encountered).
        if (
            not isinstance(attr_val, Module)
            or isinstance(attr_val, fx.GraphModule)
            or not self.enable_attr_proxy
        ):
            return super().getattr(attr, attr_val, parameter_proxy_cache)
        if isinstance(attr_val, _AttrProxy):
            return attr_val

        # See NOTE [caching AttrProxy].
        if attr_val not in self.attr_proxy_map:
            self.attr_proxy_map[attr_val] = self.proxy_type(attr_val, attr)
        else:
            self.attr_proxy_map[attr_val].reset_proxy_mapping(attr_val, attr)
        return self.attr_proxy_map[attr_val]

    def trace(  # type: ignore[override]
        self, root: Module | Callable[..., Any], concrete_args: dict[str, object] | None
    ) -> fx.Graph:
        # [FUNCTION DOC] HELPER: Traces root through the base tracer, then rebuilds the global export map from
        # fake tensor ids to proxy nodes (clearing it first), and replaces any AttrProxy modules registered on
        # the root during tracing by the real modules. Returns the graph. Mutates global state and the root.
        res = super().trace(root, concrete_args)

        # NOTE [export non-strict fake tensor leak detection]
        # In non-strict export, we don't have dynamo's side effect
        # tracking logic which makes some cases hard to detect.
        # In general, our detecting strategy is:
        #  (1) We instrument fake tensor creation to log all the fake tensors created during export.
        #  (2) We dump the proxy to fake tensor map from make_fx tracer (_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT))
        #  (3) Filter out fake tensors that are logged during (1):
        #      (1) Associated with TrackedFake (input tracking thing in symbolic_shapes)
        #      (2) Associated with gm.meta
        #  (4) Do ID match with the proxies

        global _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT
        _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT.clear()

        for key, val in self.tensor_tracker.items():
            _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT[id(key)] = val.proxy.node

        # Since we are making _AttrProxy mimic the original
        # submodule, when someone registers a module directly
        # to the tracer while tracing, the proxy object gets registered
        # first. So we need to replace the proxy modules with the real ones
        # This can happen during HOO tracing
        proxy_module_names_to_be_replaced: list[tuple[str, _AttrProxy]] = []
        for name, module in self.root.named_modules():
            if module in self.proxy_modules:
                proxy_module_names_to_be_replaced.append((name, module))

        def _delete_proxy_attr(obj: Module, target: str) -> bool:
            # [FUNCTION DOC] HELPER: Deletes the attribute named by the dotted path target from module obj if every
            # step exists and the final attribute is an AttrProxy; returns True on deletion, False otherwise. Raises
            # AssertionError if obj is not a Module. Mutates the module.
            # Copied from fx/graph_module.py
            # Customized it for proxy type
            atoms = target.split(".")
            path, target_submod = atoms[:-1], atoms[-1]
            if not isinstance(obj, Module):
                raise AssertionError(f"Expected Module, got {type(obj)}")
            mod = obj

            # Get the parent module
            # pyrefly: ignore [bad-assignment]
            for item in path:
                if not hasattr(mod, item):
                    return False

                mod = getattr(mod, item)

                if not isinstance(mod, (_AttrProxy, Module)):
                    return False

            if not hasattr(mod, target_submod):
                return False

            # At least the leaf module should be proxy type.
            if not isinstance(getattr(mod, target_submod), _AttrProxy):
                return False

            delattr(mod, target_submod)
            return True

        for proxy_module_name, proxy_module in proxy_module_names_to_be_replaced:
            _delete_proxy_attr(self.root, proxy_module_name)
            actual_module = self.proxy_modules[proxy_module]
            _assign_attr(actual_module, self.root, proxy_module_name)

        return res

    def call_module(
        self,
        m: Module,
        forward: Callable[..., Any],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        # [FUNCTION DOC] Traces through module m with scope handling: modules that are compiled wrappers or
        # GraphModules are simply inlined by calling forward; otherwise defers to the base tracer, and if the
        # module is not installed as a submodule logs at debug level and inlines forward. Returns the result of
        # forward.
        # KNOWN ISSUE: the return annotation says None although the method returns the forward result.
        """PythonKeyTracer overrides call_module to avoid the scope handling,
        but we actually want it.
        """
        from torch._dynamo import OptimizedModule

        # FIXME (tmanlaibaatar)
        # When we call torch.compile inside HOO, we will end up
        # invoking a module that is not registered on the root. For
        # now, we just inline them. But once we start supporting
        # mark_strict in export, we do need to properly handle this.
        # Right now, it doesn't matter because current non-strict
        # use cases don't need to work with HOO.
        if isinstance(m, (OptimizedModule, GraphModule)):
            return forward(*args, **kwargs)

        try:
            return Tracer.call_module(self, m, forward, args, kwargs)
        except _ModuleNotInstalledAsSubmoduleError:
            log.debug(
                "Unable to find the path of the module %s. "
                "This might be because the module was not properly registered "
                "as a submodule, which is not good practice. We will trace "
                "through the module without recording stack information.",
                m,
            )
            return forward(*args, **kwargs)

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        # [FUNCTION DOC] Never treats a module as a leaf; always returns False.
        return False

    def create_node(self, *args: object, **kwargs: object) -> fx.node.Node:
        # [FUNCTION DOC] Creates a node through the base class and adds metadata: nn_module_stack (copy of the
        # current module stack for non-placeholder/output nodes, with class entries converted to dotted
        # module-plus-qualified-name strings) and torch_fn (name with call count and a class-qualified name) for
        # call_function nodes when torch function metadata is present. Returns the node.
        """
        Create node and add on metadata.
        Add nn_module_stack here instead of TracerBase,
        since calls to make_fx() might not want to record module stack metadata.
        Add torch_fn by looking at torch_fn_metadata and torch_fn_counts.
        Add stack_trace by filtering out forward() stack frames.
        """
        node = super().create_node(*args, **kwargs)  # type: ignore[arg-type]

        # nn_module_stack
        if node.op not in ["placeholder", "output"]:
            if node.meta.get("nn_module_stack") is None:
                node.meta["nn_module_stack"] = self.module_stack.copy()
            # convert nn_module_stack from Dict[key, (FQN, class)] -> Dict[str, Tuple[str, str]]
            for key, (fqn, mod_cls) in node.meta["nn_module_stack"].items():
                if isinstance(mod_cls, type):
                    node.meta["nn_module_stack"][key] = (
                        fqn,
                        mod_cls.__module__ + "." + mod_cls.__qualname__,
                    )

        # torch_fn
        if (
            node.op == "call_function"
            and (torch_fn := self.torch_fn_metadata) is not None
            and "torch_fn" not in node.meta
        ):
            node.meta["torch_fn"] = (
                # pyrefly: ignore[missing-attribute,bad-index]
                f"{torch_fn.__name__}_{self.torch_fn_counts[torch_fn]}",
                # pyrefly: ignore[missing-attribute]
                f"{torch_fn.__class__.__name__}.{torch_fn.__name__}",
            )

        return node


class _MakefxTracer:
    # [CLASS DOC] HELPER: Driver object behind make_fx. Holds configuration (decomposition table with
    # default sym_numel and, when no table was passed and not pre-dispatch, detach decompositions; tracing
    # mode; fake-input and data-dependent-op options; stack-trace and module-stack recording; dynamic shape
    # spec) and the per-trace modes (fake tensor mode, proxy mode, pre-dispatch function mode, fx tracer,
    # python dispatcher mode, torch function metadata mode). Modes are created before tracing and restored
    # afterwards. Sub-tracers (parent_tracer set) trace higher order operator subgraphs. Key methods:
    # trace, trace_subgraph, trace_subgraph_custom_decomp.
    def __init__(
        self,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
        tracing_mode: _TracingMode,
        _allow_non_fake_inputs: bool,
        pre_dispatch: bool,
        record_module_stack: bool,
        _allow_fake_constant: bool,
        _error_on_data_dependent_ops: bool,
        record_stack_traces: bool = False,
        parent_tracer: _MakefxTracer | None = None,
        proxy_module_inputs: bool = False,
        _disable_torch_fn_metadata_mode: bool = False,
        dynamic_shapes: ShapesSpec | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores configuration and initializes all mode slots to inert placeholders.
        # Raises ValueError if dynamic_shapes is given while tracing_mode is not fake.
        # Configurations that are used to initialize the context managers and their states.
        # Should not modify them during tracing.
        self.decomposition_table: dict[OpOverload, Callable[..., Any]] = dict(
            decomposition_table or {}
        )
        self.decomposition_table.setdefault(
            torch.ops.aten.sym_numel.default, torch._decomp.decompositions.sym_numel
        )
        # Only inject the default detach decomp when the caller passed no table
        # at all. An explicit table, even empty, opts out and preserves exact
        # detach semantics for pre-autograd export and compile paths.
        if decomposition_table is None and not pre_dispatch:
            self.decomposition_table.setdefault(
                torch.ops.aten.detach.default,
                torch._decomp.decompositions.nop_decomposition,
            )
        self.tracing_mode: _TracingMode = tracing_mode
        self._allow_non_fake_inputs: bool = _allow_non_fake_inputs
        self.pre_dispatch: bool = pre_dispatch
        self.record_module_stack: bool = record_module_stack
        self._allow_fake_constant: bool = _allow_fake_constant
        self._error_on_data_dependent_ops: bool = _error_on_data_dependent_ops

        # All context managers and their states should be initialized before tracing based on the inputs
        # and configurations. After tracing, their states should be cleaned except for shape_env.
        # Remember to specify how to initialize it from user inputs and from parent tracer whenever
        # adding new modes in _MakefxTracer.
        self.fake_tensor_mode: FakeTensorMode | None = None
        self.proxy_mode: nullcontext[None] | ProxyTorchDispatchMode = nullcontext()
        self.proxy_function_mode: nullcontext[None] | PreDispatchTorchFunctionMode = (
            nullcontext()
        )
        self.fx_tracer: PythonKeyTracer | None = None
        self.python_dispatcher_mode: nullcontext[None] | Any = nullcontext()
        self.torch_fn_metadata_mode: nullcontext[None] | TorchFunctionMetadataMode = (
            nullcontext()
        )
        self.record_stack_traces = record_stack_traces
        self.parent_tracer: _MakefxTracer | None = parent_tracer
        self.proxy_module_inputs = proxy_module_inputs
        self._disable_torch_fn_metadata_mode = _disable_torch_fn_metadata_mode
        # Declares which inputs are dynamic. See
        # ``torch.fx.experimental.dynamic_spec`` for the spec API.
        #
        # Only set on the top-level tracer; sub-tracers default to None.
        # Higher-order ops (``cond``, ``while_loop``, ``scan``, ...) spawn
        # sub-tracers via ``_make_sub_tracer`` to produce a separate
        # subgraph for each branch/body.

        # ``dynamic_shapes`` is pre-normalized by ``make_fx()``; assign as-is.
        self.dynamic_shapes: ShapesSpec | None = dynamic_shapes
        # ``dynamic_shapes`` requires ``tracing_mode="fake"`` so the spec
        # is the *sole* source of dynamism:
        #   - ``"real"``: no fakification → spec would be silently
        #     ignored (footgun).
        #   - ``"symbolic"``: every dim becomes a backed symbol by
        #     default.
        if self.dynamic_shapes is not None and self.tracing_mode != "fake":
            raise ValueError(
                f"make_fx(dynamic_shapes=...) requires tracing_mode='fake' "
                f"(got tracing_mode={self.tracing_mode!r}). 'real' would "
                f"silently ignore the spec; 'symbolic' conflicts with the "
                f"spec's unbacked-by-default semantics."
            )

    def _checkpoint_modes(self) -> list[Any]:
        # [FUNCTION DOC] HELPER: Returns a list of the current mode objects (fake mode, proxy mode, function
        # mode, fx tracer, python dispatcher mode, metadata mode) for later restoration.
        return [
            self.fake_tensor_mode,
            self.proxy_mode,
            self.proxy_function_mode,
            self.fx_tracer,
            self.python_dispatcher_mode,
            self.torch_fn_metadata_mode,
        ]

    def _restore_modes(
        self,
        prev_fake_tensor_mode: FakeTensorMode | None,
        prev_proxy_mode: nullcontext[None] | ProxyTorchDispatchMode,
        prev_proxy_function_mode: nullcontext[None] | PreDispatchTorchFunctionMode,
        prev_fx_tracer: PythonKeyTracer | None,
        prev_python_dispatcher_mode: nullcontext[None] | Any,
        prev_torch_fn_metadata_mode: nullcontext[None] | TorchFunctionMetadataMode,
    ) -> None:
        # [FUNCTION DOC] HELPER: Assigns the given previous values back to the six mode attributes.
        self.fake_tensor_mode = prev_fake_tensor_mode
        self.proxy_mode = prev_proxy_mode
        self.proxy_function_mode = prev_proxy_function_mode
        self.fx_tracer = prev_fx_tracer
        self.python_dispatcher_mode = prev_python_dispatcher_mode
        self.torch_fn_metadata_mode = prev_torch_fn_metadata_mode

    @contextmanager
    def _init_modes_from_inputs(
        self, f: Callable[..., Any], args: tuple[object, ...]
    ) -> Generator[None, None, None]:
        # [FUNCTION DOC] HELPER: Context manager that, for a top-level trace, creates the fx tracer (module stack
        # tracer if f wraps a module and module stack recording is on, else a python key tracer), and the fake
        # tensor mode as needed (detecting one from args, or creating one; symbolic mode requires a shape env,
        # fake mode with a dynamic spec requires one too), then builds the proxy modes and yields. Modes are
        # restored on exit. Raises ValueError/AssertionError for invalid configuration or tracing mode.
        prev_modes = self._checkpoint_modes()
        try:
            # Avoid importing sympy at a module level
            from .symbolic_shapes import ShapeEnv

            if hasattr(f, "_orig_mod") and self.record_module_stack:
                scope_root = f._orig_mod
                # _ModuleStackTracer always try to preserve stack trace
                # in forward functions
                self.fx_tracer = _ModuleStackTracer(scope_root)
            else:
                self.fx_tracer = PythonKeyTracer()
                self.fx_tracer.record_stack_traces = (
                    self.record_stack_traces and not fx.config.do_not_emit_stack_traces
                )
                if self.fx_tracer.record_stack_traces:
                    self.fx_tracer._record_forward_stack_traces_only = True

            if self.tracing_mode == "fake":
                import torch._dynamo

                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    import torch._functorch.config as _config

                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=True,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=ShapeEnv(),
                            static_shapes=True,
                        )
                # dynamic_shapes wires unbacked symbols into the shape env, so
                # it requires one. The fake mode we create above always has it;
                # a detected ambient fake mode might not, and mutating a
                # borrowed mode is unsafe, so reject it with a clear error.
                if (
                    self.dynamic_shapes is not None
                    and fake_tensor_mode.shape_env is None
                ):
                    raise ValueError(
                        "make_fx(dynamic_shapes=...) requires the active "
                        "FakeTensorMode to have a shape_env, but the detected "
                        "fake mode has none."
                    )
                self.fake_tensor_mode = fake_tensor_mode
            elif self.tracing_mode == "symbolic":
                import torch._dynamo

                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    shape_env = ShapeEnv()
                    import torch._functorch.config as _config

                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=False,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=shape_env,
                        )
                if fake_tensor_mode.shape_env is None:
                    raise AssertionError(
                        "shape_env should be set if tracing with 'symbolic'"
                    )
                self.fake_tensor_mode = fake_tensor_mode
            else:
                if not self.tracing_mode == "real":
                    raise AssertionError(
                        f"Unexpected tracing type: {self.tracing_mode}"
                    )

            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)

    def _construct_modes_with_fx_tracer(self, fx_tracer: _ProxyTracer) -> None:
        # [FUNCTION DOC] HELPER: Builds the proxy dispatch mode for fx_tracer; adds the pre-dispatch function mode
        # if pre_dispatch, the python dispatcher for symbolic or pre-dispatch tracing, and the torch function
        # metadata mode unless disabled; sets proxy_module_inputs on the tracer.
        self.proxy_mode = ProxyTorchDispatchMode(
            fx_tracer,
            self.tracing_mode,
            decomposition_table=self.decomposition_table,
            pre_dispatch=self.pre_dispatch,
            _allow_fake_constant=self._allow_fake_constant,
            _error_on_data_dependent_ops=self._error_on_data_dependent_ops,
        )

        if self.pre_dispatch:
            self.proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        if self.tracing_mode == "symbolic" or self.pre_dispatch:
            self.python_dispatcher_mode = enable_python_dispatcher()

        if not self._disable_torch_fn_metadata_mode:
            self.torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)
        fx_tracer.proxy_module_inputs = self.proxy_module_inputs  # type: ignore[union-attr]

    @contextmanager
    def _init_modes_from_parent(
        self, parent_tracer: _MakefxTracer
    ) -> Generator[None, None, None]:
        # [FUNCTION DOC] HELPER: Context manager for sub-tracers: shares the parent's fake tensor mode, creates a
        # fresh fx tracer of the same kind as the parent's, builds modes, yields, and restores modes afterwards.
        # Raises AssertionError if the parent has no fx tracer.
        # By default, subtracer creates new modes based on parent tracer's config.
        # However, there are cases where we want to share the same modes with parent tracer
        # For example, fake_tensor_mode, we want the example value's fake_mode of parent graph and subgraphs to be the same.
        prev_modes = self._checkpoint_modes()
        try:
            self.fake_tensor_mode = parent_tracer.fake_tensor_mode

            def _create_sub_fx_tracer(parent_tracer: _ProxyTracer) -> PythonKeyTracer:
                # [FUNCTION DOC] HELPER: Creates a new tracer of the same type as parent_tracer (plain python key tracer or
                # module stack tracer sharing the scope root); raises RuntimeError for other types.
                if type(parent_tracer) is PythonKeyTracer:
                    return PythonKeyTracer()
                elif type(parent_tracer) is _ModuleStackTracer:
                    return _ModuleStackTracer(parent_tracer.scope_root)
                else:
                    raise RuntimeError(
                        f"Unexpected tracer type: {type(parent_tracer)}."
                    )

            if parent_tracer.fx_tracer is None:
                raise AssertionError("parent_tracer.fx_tracer should not be None")
            self.fx_tracer = _create_sub_fx_tracer(parent_tracer.fx_tracer)
            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)

    def _convert_args_to_fake(self, f: Callable[..., Any], args: T) -> T:
        # [FUNCTION DOC] HELPER: Converts the example args to fake values. Real tracing mode returns args
        # unchanged. Otherwise binds the dynamic shape spec (or an empty one) to the flattened args, wires spec
        # assumptions into the shape env, fakifies each leaf (tensors via the fake mode with symbolic context from
        # the spec, spec-declared ints to unbacked symbolic ints, other ints to symbolic ints in symbolic mode,
        # script/opaque objects to fake objects unless constant types), finalizes spec wiring and returns args
        # rebuilt in the original structure. Raises AssertionError on inconsistent state.
        """Fakify ``args``. If ``self.dynamic_shapes`` is set, create
        tensors with dynamic dims accordingly.
        """
        if self.tracing_mode == "real":
            return args

        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental._spec_binding import _bind_spec_to_args

        # NB: these spec-wiring helpers live in symbolic_shapes, which does
        # ``import sympy`` at module scope. Import them lazily here (once per
        # trace) — a module-level import would pull sympy into ``import torch``
        # and break test_not_import_sympy.
        from torch.fx.experimental.symbolic_shapes import (
            _finalize_spec_wiring,
            _symbolic_context_from_shapes_spec,
            _wire_spec_assumptions,
            _wire_spec_slot,
            _wire_tensor_spec_dims,
        )

        def fakify_leaf(x: object, source: Source, leaf_spec: LeafSpec) -> object:
            # [FUNCTION DOC] HELPER: Fakifies one flat input leaf x according to leaf_spec and the tracing mode (see
            # _convert_args_to_fake); other values are returned unchanged. Raises AssertionError if an already
            # fakified script object is passed or required modes are missing.
            """Fakify a single flat input leaf (closes over the helpers above)."""
            if self.fake_tensor_mode is None:
                raise AssertionError("fake_tensor_mode should not be None")

            if isinstance(x, Tensor):
                if leaf_spec is None:
                    return self.fake_tensor_mode.from_tensor(x, source=source)
                # leaf_spec is guaranteed to be a TensorSpec by _walk_spec.
                tensor_spec = cast(TensorSpec, leaf_spec)
                ctx = _symbolic_context_from_shapes_spec(
                    x, source, tensor_spec, None, {}
                )
                fake_x = self.fake_tensor_mode.from_tensor(
                    x, static_shapes=False, source=source, symbolic_context=ctx
                )
                _wire_tensor_spec_dims(tensor_spec, fake_x)
                return fake_x

            # NB: don't match on bools.
            if type(x) is int:
                if isinstance(leaf_spec, (IntVar, SymInt)):
                    shape_env = self.fake_tensor_mode.shape_env
                    if shape_env is None:
                        raise AssertionError("shape_env should be set for spec inputs")
                    sym_node = shape_env.create_unbacked_symint(source=source)
                    _wire_spec_slot(leaf_spec, sym_node)
                    return sym_node
                if self.tracing_mode == "symbolic":
                    shape_env = self.fake_tensor_mode.shape_env
                    if shape_env is None:
                        raise AssertionError(
                            "shape_env should be set if tracing with 'symbolic'"
                        )
                    return shape_env.create_symintnode(
                        shape_env.create_symbol(x, source, positive=None),
                        hint=x,
                        source=source,
                    )
                # Otherwise: an int not declared in the spec stays static.

            if isinstance(x, torch.ScriptObject) or is_custom_class_obj(x):
                if is_opaque_constant_type(
                    type(x)  # pyrefly: ignore[bad-argument-type]
                ):
                    return x
                return torch._library.fake_class_registry.maybe_to_fake_obj(
                    self.fake_tensor_mode, x
                )

            if isinstance(x, FakeScriptObject):
                raise AssertionError(
                    f"ScriptObject {x} has been fakified. Cannot wrap_fake it again."
                )
            return x

        # Pass an empty ShapesSpec when no user spec was supplied so the
        # binding path is uniform (yields all-None leaf_specs aligned to the
        # flat layout). Both fake_tensor_mode and shape_env are set for tracing.
        shape_env: ShapeEnv = self.fake_tensor_mode.shape_env  # type: ignore[assignment]
        user_spec = self.dynamic_shapes
        # Wire assumptions BEFORE processing inputs so derived / assumption
        # checks can drain as inputs bind.
        if user_spec is not None and user_spec._assumptions:
            _wire_spec_assumptions(shape_env, user_spec)

        leaf_specs, flat_args, args_pytree_spec = _bind_spec_to_args(
            f,
            args,
            {},
            user_spec or ShapesSpec(),
        )

        # ``shape_env`` is None for plain fake tracing with no symbolic shapes;
        # only the spec path allocates unbacked symbols needing the guard.
        unbacked_ctx = (
            shape_env.ignore_fresh_unbacked_symbols()
            if shape_env is not None
            else nullcontext()
        )
        with unbacked_ctx:
            fake_leaves = [
                fakify_leaf(x, ConstantSource(f"input{i}"), leaf_specs[i])
                for i, x in enumerate(flat_args)
            ]

        # Verify every spec assumption / derived check that survived input
        # processing has been emitted.
        if user_spec is not None:
            _finalize_spec_wiring(shape_env)

        # args_pytree_spec wraps (args, {}); unflatten and take the args.
        return pytree.tree_unflatten(fake_leaves, args_pytree_spec)[0]  # type: ignore[return-value]

    def _trace_inner(self, f: Callable[..., Any], *args: object) -> GraphModule:
        # [FUNCTION DOC] HELPER: Performs the actual trace of f with the prepared modes. Converts args to fake,
        # uses a generated fixed-arity wrapper if f has varargs or no code object, enters all modes (plus
        # autocast cache disabling, current tracer registration and, under compile-on-one-rank, the legacy
        # collective rewrite mode), runs dispatch_trace on the key-wrapped function (logging a partial graph via
        # structured tracing and re-raising on failure), and for subgraph tracers or dynamic specs inserts deferred
        # runtime asserts and recompiles. In symbolic mode attaches the shape env to the result. Returns the
        # GraphModule.
        # TODO: We need to explicitly import torch._dynamo before calling dispatch_trace,
        # because dispatch_trace will introduce the lazy import of torch._dynamo,
        # and some contexts set before calling dispatch_trace will cause problems with the import of torch._dynamo,
        # such as some torch API(torch.ones and so on) in populate_builtin_to_tensor_fn_map() will be affected
        # by the context set before dispatch_trace.
        import torch._dynamo

        phs = pytree.tree_map(lambda _: torch.fx._symbolic_trace.PH, args)

        args = self._convert_args_to_fake(f, args)

        # FX doesn't support varargs, so we gotta fake up a wrapper
        # TODO: Would be nice to fix this at the source...
        func: Callable[..., Any] = f
        if (
            not hasattr(inspect.unwrap(f), "__code__")
            or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS
        ):
            func = fake_signature(f, len(phs))
        # We disable the autocast cache as the autocast cache causes type conversions on parameters to
        # check a cache, which introduces untracked tensors into the graph
        #
        # We also disable tracing by any other tensor proxy-based tracers except the current. The
        # purpose of `make_fx` is to produce graphmodules as a side effect; its internal execution is
        # thus irrelevant to any external functional trace.
        proxy_mode: ProxyTorchDispatchMode = typing.cast(
            ProxyTorchDispatchMode, self.proxy_mode
        )
        with ExitStack() as stack:
            if self.fake_tensor_mode:
                stack.enter_context(self.fake_tensor_mode)
            stack.enter_context(self.python_dispatcher_mode)
            stack.enter_context(self.proxy_function_mode)
            stack.enter_context(self.torch_fn_metadata_mode)
            stack.enter_context(proxy_mode)
            stack.enter_context(disable_autocast_cache())
            stack.enter_context(_set_make_fx_tracer(self))

            # Under compile-on-one-rank, redirect legacy in-place c10d
            # collectives to functional collectives so the ProcessGroup flows
            # into the graph as a (serializable) op argument instead of being
            # baked in as a torchbind constant by the in-place op.
            if (
                torch.compiler.config.compile_on_one_rank
                and torch.distributed.is_available()
            ):
                from torch.distributed._functional_collectives import (
                    _LegacyToFunctionalCollectiveMode,
                )

                stack.enter_context(_LegacyToFunctionalCollectiveMode())

            if self.fx_tracer is None:
                raise AssertionError("fx_tracer should not be None")
            try:
                t = dispatch_trace(
                    wrap_key(func, args, self.fx_tracer, self.pre_dispatch),
                    tracer=self.fx_tracer,
                    concrete_args=tuple(phs),
                )
            except Exception:
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "make_fx_fail_partial",
                        "encoding": "string",
                    },
                    payload_fn=lambda: self.fx_tracer.graph.python_code(  # type: ignore[union-attr]
                        root_module="self",
                        verbose=True,
                        include_stride=True,
                        include_device=True,
                    ).src,
                )
                raise

        if (
            # TODO: consider running this unconditionally so data-dependent ops
            # (e.g. ``.item()``, ``nonzero``) get runtime asserts in plain
            # make_fx traces too, not only when an explicit ``dynamic_shapes``
            # spec or a HOP subgraph is involved.
            (self.is_hop_subgraph_tracer() or self.dynamic_shapes is not None)
            and (fake_mode := torch._guards.detect_fake_mode(args))
            and fake_mode.shape_env is not None
        ):
            from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

            if self.dynamic_shapes is not None:
                # Attach ``unbacked_bindings`` to input placeholders so the
                # runtime-assert pass can resolve deferred asserts keyed by
                # spec-introduced unbacked symbols.
                from torch.export._trace import _add_input_unbacked_bindings

                _add_input_unbacked_bindings(t)
            insert_deferred_runtime_asserts(t, fake_mode.shape_env, "reenter_make_fx")
            t.recompile()
        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if self.tracing_mode == "symbolic":
            if self.fake_tensor_mode is None:
                raise AssertionError("fake_tensor_mode should not be None")
            t.shape_env = self.fake_tensor_mode.shape_env  # type: ignore[assignment]
        return t

    def trace(self, f: Callable[..., Any], *args: object) -> fx.GraphModule:
        # [FUNCTION DOC] Top-level trace: sets up modes from the inputs and runs the trace; returns the
        # GraphModule.
        with self._init_modes_from_inputs(f, args):
            return self._trace_inner(f, *args)

    def is_hop_subgraph_tracer(self) -> bool:
        # [FUNCTION DOC] Returns True if this tracer has a parent (traces a higher order operator subgraph).
        return self.parent_tracer is not None

    def _make_sub_tracer(
        self,
        decomp_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
        tracing_mode: _TracingMode = "real",
    ) -> _MakefxTracer:
        # [FUNCTION DOC] HELPER: Creates a child tracer with the same options, using decomp_table (default: this
        # tracer's table) and tracing_mode, with this tracer as parent.
        return _MakefxTracer(
            decomp_table if decomp_table is not None else self.decomposition_table,
            tracing_mode,
            self._allow_non_fake_inputs,
            self.pre_dispatch,
            self.record_module_stack,
            self._allow_fake_constant,
            self._error_on_data_dependent_ops,
            parent_tracer=self,
        )

    def trace_subgraph(self, f: Callable[..., Any], *args: object) -> GraphModule:
        # [FUNCTION DOC] Traces f as a subgraph using a child tracer sharing the parent's fake mode; returns the
        # GraphModule.
        sub_tracer = self._make_sub_tracer()
        with sub_tracer._init_modes_from_parent(self):
            return sub_tracer._trace_inner(f, *args)

    def trace_subgraph_custom_decomp(
        self,
        f: Callable[..., Any],
        decomp_table: Mapping[OpOverload, Callable[..., Any]],
        *args: object,
    ) -> GraphModule:
        # [FUNCTION DOC] Like trace_subgraph but with a custom decomposition table; raises AssertionError if the
        # table is not a mapping.
        if not isinstance(decomp_table, Mapping):
            raise AssertionError(f"Expected Mapping, got {type(decomp_table)}")
        sub_tracer = self._make_sub_tracer(decomp_table=decomp_table)
        with sub_tracer._init_modes_from_parent(self):
            return sub_tracer._trace_inner(f, *args)


_CURRENT_MAKE_FX_TRACER: _MakefxTracer | None = None


@contextmanager
def _set_make_fx_tracer(tracer: _MakefxTracer) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager setting the module-level current make_fx tracer to tracer and
    # restoring the previous value on exit (global state mutation).
    global _CURRENT_MAKE_FX_TRACER
    prev_tracer = _CURRENT_MAKE_FX_TRACER
    try:
        _CURRENT_MAKE_FX_TRACER = tracer
        yield
    finally:
        _CURRENT_MAKE_FX_TRACER = prev_tracer


def make_fx(
    f: Callable[..., Any],
    decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
    tracing_mode: _TracingMode = "real",
    _allow_non_fake_inputs: bool = False,
    *,
    pre_dispatch: bool = False,
    record_module_stack: bool = False,
    _allow_fake_constant: bool = False,
    _error_on_data_dependent_ops: bool = True,
    record_stack_traces: bool = False,
    proxy_module_inputs: bool = False,
    _disable_torch_fn_metadata_mode: bool = False,
    dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
) -> Callable[..., GraphModule]:
    # [FUNCTION DOC] Public entry point. Validates tracing_mode (AssertionError if not real, fake or
    # symbolic), resolves dynamic shapes (including a spec attached to f), builds a _MakefxTracer (stack
    # traces also recorded when inductor provenance tracking level is 1) and returns a function that, when
    # called with example arguments, traces f and returns a GraphModule.
    """
    Given a function f, return a new function which when executed with valid
    arguments to f, returns an FX GraphModule representing the set of operations that
    were executed during the course of execution.

    If record_stack_traces is True, the stack trace will be preserved on node.meta["stack_trace"]

    By default, post-dispatch traces without an explicit decomposition_table
    rewrite detach to alias. Re-differentiating the returned graph does not
    preserve those detach calls as autograd gradient cuts; pass an explicit
    decomposition_table or use pre_dispatch=True if exact detach nodes are needed.

    ``tracing_mode``:
        - ``"real"``: no fakification, traces with real tensors.
        - ``"fake"``: tensors become FakeTensors. By default all dims are
          static (concrete); if ``dynamic_shapes`` is set, the spec
          controls which dims become unbacked symbolic sizes — everything
          not declared by the spec stays static.
        - ``"symbolic"``: tensors become FakeTensors and **every dim becomes a
          backed symbolic size**. NOTE: using backed symbols this way is generally
          **not recommended and not sound** — backed symbols come with assumptions
          and can be constrained by guards silently. Kept for backward compatibility.
          Prefer ``"fake"`` + ``dynamic_shapes`` to declare dynamism explicitly with
          unbacked symbols.
    """

    if tracing_mode not in ["real", "fake", "symbolic"]:
        raise AssertionError(
            f"tracing_mode must be real/fake/symbolic, got {tracing_mode}"
        )

    # If ``f`` carries an ``@dynamic_spec(...)`` decorator, the attached
    # ``ShapesSpec`` is used as ``dynamic_shapes``. Passing both raises.
    from torch.fx.experimental.dynamic_spec import _resolve_dynamic_shapes

    dynamic_shapes = _resolve_dynamic_shapes(f, dynamic_shapes)

    from torch._inductor import config

    make_fx_tracer = _MakefxTracer(
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops,
        record_stack_traces=record_stack_traces
        or config.effective_provenance_tracking_level() == 1,
        proxy_module_inputs=proxy_module_inputs,
        _disable_torch_fn_metadata_mode=_disable_torch_fn_metadata_mode,
        dynamic_shapes=_coerce_to_shapes_spec(dynamic_shapes),
    )

    @functools.wraps(f)
    def wrapped(*args: object) -> GraphModule:
        # [FUNCTION DOC] HELPER: Traces f with the given example args via the shared tracer and returns the
        # GraphModule.
        return make_fx_tracer.trace(f, *args)

    return wrapped


def get_torch_dispatch_modes() -> list[TorchDispatchMode]:
    # [FUNCTION DOC] Returns the current stack of torch dispatch modes.
    return torch.utils._python_dispatch._get_current_dispatch_mode_stack()


# TODO: this is a legacy name, there is only ever one proxy mode as it's an
# infra mode
def get_innermost_proxy_mode() -> ProxyTorchDispatchMode | None:
    # [FUNCTION DOC] Legacy alias of get_proxy_mode.
    return get_proxy_mode()


def get_proxy_mode() -> ProxyTorchDispatchMode | None:
    # [FUNCTION DOC] Returns the active proxy dispatch mode (pre-dispatch or regular) or None when not
    # tracing. Raises AssertionError if both kinds are active.
    # KNOWN ISSUE: the docstring begins with the typo "Current the currently active".
    """
    Current the currently active proxy tracing mode, or None if
    we are not currently tracing.  This includes pre-dispatch proxy
    tracing.
    """
    pre_dispatch_mode = torch._ops._get_dispatch_mode_pre_dispatch(
        torch._C._TorchDispatchModeKey.PROXY
    )
    mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    if not (pre_dispatch_mode is None or mode is None):
        raise AssertionError(f"pre_dispatch_mode={pre_dispatch_mode}, mode={mode}")
    return typing.cast("ProxyTorchDispatchMode | None", pre_dispatch_mode or mode)


def handle_sym_dispatch(
    func: Callable[_P, R],
    args: _P.args,  # type: ignore[valid-type]  # not allowed to use _P.args here
    kwargs: _P.kwargs,  # type: ignore[valid-type]  # not allowed to use _P.kwargs here
) -> R:
    # [FUNCTION DOC] Runs func through the active proxy mode's symbolic dispatch (with proxy tracing disabled
    # around the call) and returns its result. Raises AssertionError if no proxy mode is active.
    """
    Call into the currently active proxy tracing mode to do a
    SymInt/SymFloat/SymBool dispatch trace on a function that operates on
    these arguments.
    """
    mode = get_proxy_mode()
    if not mode:
        raise AssertionError("Expected proxy mode to be set")
    # Have to do it manually, because we're not doing the normal torch
    # dispatch machinery which disables it for us
    with disable_proxy_modes_tracing():
        # TODO: properly compute types
        types: list[type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)  # type: ignore[arg-type, return-value]


@contextmanager
def disable_proxy_modes_tracing() -> Generator[ProxyTorchDispatchMode, None, None]:
    # [FUNCTION DOC] Context manager that disables the proxy infra mode for its duration and yields it.
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.PROXY)


def maybe_handle_decomp(
    proxy_mode: ProxyTorchDispatchMode,
    op: OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # [FUNCTION DOC] If op has a decomposition in the mode's table (and the compiler bisector does not
    # disable it), runs it under the mode while counting a decomposition layer and returns the result;
    # otherwise returns NotImplemented.
    # KNOWN ISSUE: decomp_layers is decremented without try/finally, so an exception in the decomposition
    # leaves the counter incremented.
    from torch._inductor.compiler_bisector import CompilerBisector

    decomp_table = proxy_mode.decomposition_table
    if op in decomp_table:
        if CompilerBisector.disable_subsystem(
            "aot_eager_decomp_partition", "decomposition", lambda: repr(op)
        ):
            return NotImplemented

        with proxy_mode:
            proxy_mode.decomp_layers += 1
            out = decomp_table[op](*args, **kwargs)
            proxy_mode.decomp_layers -= 1
            return out

    return NotImplemented


def get_isolated_graphmodule(
    func: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    tracing_mode: _TracingMode = "real",
    decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> GraphModule:
    # [FUNCTION DOC] Traces func with the given args and kwargs (flattened) into a new GraphModule using
    # make_fx with proxy modes disabled around it, so the result is isolated from the surrounding trace.
    """A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
    wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)

    with disable_proxy_modes_tracing():
        gm = make_fx(
            wrapped, decomposition_table=decomposition_table, tracing_mode=tracing_mode
        )(all_args)
    return gm


def _set_unbacked_bindings(out: object, out_proxy: _NestedProxys) -> None:
    # [FUNCTION DOC] HELPER: If a fake mode with a shape env is active, computes unbacked symbol bindings for
    # the output value, routes each binding to the proxy owning it by walking the output proxy structure (the
    # rest of the path stays as the local accessor), and merges them into each node's meta
    # unbacked_bindings. Raises AssertionError if a path does not match the proxy structure or does not end
    # at a Proxy.
    """A helper function for setting up unbacked_bindings on the destination FX graph."""
    from .symbolic_shapes import compute_unbacked_bindings

    # Can't use detect_fake_mode here,
    #
    # python test/distributed/_tensor/test_dtensor_compile.py -k
    # test_tp_compile_fullgraph_is_seq_parallel_False
    #
    # will fail.  Very strange, it probably isn't right for them to be using
    # two fake modes there...
    fake_mode = typing.cast(
        "FakeTensorMode | None",
        torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE),
    )
    if fake_mode and fake_mode.shape_env:
        if symbol_to_path := compute_unbacked_bindings(fake_mode.shape_env, out):
            # `symbol_to_path` is keyed by the fresh unbacked symbol; each path
            # is relative to the *whole* output pytree (`out`). Route each
            # binding to the proxy that owns it. If `out_proxy` is already a
            # Proxy, it owns the whole output and keeps the original path.
            bindings_by_proxy: dict[Proxy, dict[sympy.Symbol, pytree.KeyPath]] = {}
            for symbol, path in symbol_to_path.items():
                proxy: _NestedProxys = out_proxy
                local_path = path
                # `out` and `out_proxy` are structurally congruent, so the
                # leading container keys index identically into both. Consume
                # them off `out_proxy` until we reach the owning Proxy leaf; the
                # remaining suffix (e.g. `.size()[0]`, or an InnerTensorKey for
                # a wrapper subclass) is the tensor-local accessor that the
                # per-node consumer resolves against that node's own value.
                while not isinstance(proxy, Proxy) and local_path:
                    try:
                        proxy = local_path[0].get(proxy)
                    except (IndexError, KeyError, AttributeError) as e:
                        raise AssertionError(
                            f"unbacked_bindings path {pytree.keystr(path)} does "
                            f"not index into the proxy pytree {out_proxy}"
                        ) from e
                    local_path = local_path[1:]

                if not isinstance(proxy, Proxy):
                    raise AssertionError(
                        f"Expected Proxy at binding path {pytree.keystr(path)}, "
                        f"got {type(proxy)}"
                    )

                bindings_by_proxy.setdefault(proxy, {})[symbol] = local_path

            # Merge rather than overwrite: multiple symbols (e.g. two unbacked
            # dims on one tensor) can route to the same node.
            for proxy, bindings in bindings_by_proxy.items():
                proxy.node.meta.setdefault("unbacked_bindings", {}).update(bindings)
