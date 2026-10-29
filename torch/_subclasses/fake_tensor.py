# [MODULE DOC] FakeTensor and FakeTensorMode: tensors that carry metadata (shape, dtype, strides, a
# "fake" device) backed by meta tensors, so operators can be run without real computation or memory.
# Main contents: FakeTensor (Tensor subclass), FakeTensorMode (TorchDispatchMode that runs ops on
# fake tensors with meta kernels/decompositions and a dispatch cache), FakeTensorConverter (real to
# fake conversion with memoization and constant tracking), cache key/entry machinery for op output
# caching, error types, and helper predicates (is_fake, maybe_get_fake_mode, ...).
# Import-time side effects: creates module loggers, a thread-local state object (fake_tensor_tls),
# parses the Mkldnn dispatch key, and registers an atexit hook later in the file for cache stats.
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import functools
import logging
import math
import os
import threading
import traceback
import types
import typing
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast, Literal, TYPE_CHECKING, TypeGuard, TypeVar, Union
from typing_extensions import Self, TypedDict, Unpack
from weakref import ReferenceType

import torch
import torch._library.utils as library_utils
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C._functorch import is_functorch_wrapped_tensor, is_legacy_batchedtensor
from torch._custom_class_base import CustomClassBase
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.fake_profile import MissingOpProfile
from torch._logging import dtrace_structured
from torch._prims_common import suggest_memory_format
from torch._subclasses.meta_utils import (
    assert_eq,
    assert_metadata_eq,
    is_sparse_any,
    is_sparse_compressed,
    MetaConverter,
)
from torch._utils import _is_privateuse1_backend_available, render_call
from torch.fx.immutable_collections import immutable_dict
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.types import IntLikeType, py_sym_types
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
    TraceableWrapperSubclass,
)
from torch.utils._pytree import KeyPath, keystr, PyTree, tree_map, tree_map_, TreeSpec
from torch.utils._stats import count
from torch.utils._traceback import CapturedTraceback

from ._fake_tensor_utils import _CacheKeyState, _PySymInputStub, _SymIntOutputStub


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
    from types import TracebackType

    from torch._guards import Source
    from torch._ops import OpOverload
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext

log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")

# TODO: Hack to unblock https://github.com/pytorch/pytorch/pull/108186
# Proper fix tracked by https://github.com/pytorch/pytorch/issues/120105
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        not_implemented_log = logging.getLogger(__name__ + ".not_implemented")
    else:
        raise e


DimList = list

pytree = torch.utils._pytree
T = TypeVar("T")

aten = torch._ops.ops.aten

CONSTANT_NUMEL_LIMIT = 1

RECURSION_COUNT = 0

_MKLDNN_DISPATCH_KEY = torch._C._dispatch_key_parse("MkldnnCPU")


def _dispatch_keys_has_mkldnn(dispatch_keys: torch.DispatchKeySet | None) -> bool:
    # [FUNCTION DOC] HELPER: True if the optional dispatch key set is given and contains the MkldnnCPU key.
    return dispatch_keys is not None and dispatch_keys.has(_MKLDNN_DISPATCH_KEY)


class _FakeTensorConstructorIgnoredState(TypedDict, total=False):
    # [CLASS DOC] HELPER: Typed dict (all keys optional) naming private FakeTensor attributes (memo values,
    # memo version counters/epochs, debug trace) that FakeTensor.__new__ silently discards when a tensor
    # is rebuilt from another FakeTensor's attribute dictionary.
    # Recompute memo descriptor state when reconstructing from
    # FakeTensor.__dict__.
    _nonzero_memo: object
    _nonzero_memo_vc: object
    _nonzero_memo_epoch: object
    _item_memo: object
    _item_memo_vc: object
    _item_memo_epoch: object
    _unique_memo: object
    _unique_memo_vc: object
    _unique_memo_epoch: object
    _unique_consecutive_memo: object
    _unique_consecutive_memo_vc: object
    _unique_consecutive_memo_epoch: object
    _nested_int_memo: object
    _nested_int_memo_vc: object
    _nested_int_memo_epoch: object
    _debug_trace: object


_FAKE_TENSOR_CONSTRUCTOR_IGNORED_STATE_ATTRS = frozenset(
    _FakeTensorConstructorIgnoredState.__annotations__
)


# Check if device type supports device index
@functools.cache
def _is_indexed_device_type(device_type: str) -> bool:
    # [FUNCTION DOC] HELPER: Cached check whether a device type string (cuda, hpu, xpu, mps, mtia or the
    # privateuse1 backend name) supports a device index.
    return device_type in [
        "cuda",
        "hpu",
        "xpu",
        "mps",
        "mtia",
        torch._C._get_privateuse1_backend_name(),
    ]


# Small helper that increments recursion count, and
# resets it when the object goes out of scope.  Useful
# if you don't want to increase indentation which is
# what a context manager would do.
class IncrementRecursionCount:
    # [CLASS DOC] Scope guard that increments the module-global RECURSION_COUNT on creation and decrements
    # it when destroyed, giving a recursion depth counter without extra indentation. Relies on object
    # finalization timing (refcount drop) to decrement.
    def __init__(self) -> None:
        # [FUNCTION DOC] Increments the global recursion counter.
        global RECURSION_COUNT
        RECURSION_COUNT += 1

    def __del__(self) -> None:
        # [FUNCTION DOC] Decrements the global recursion counter when the object is destroyed.
        global RECURSION_COUNT
        RECURSION_COUNT -= 1


@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    # [CLASS DOC] Runtime error (also a dataclass with a reason string) raised when an operation or tensor
    # cannot be represented as a fake tensor; the mode usually catches it and falls back.
    reason: str


class FakeTensorDeviceMismatchError(RuntimeError):
    # [CLASS DOC] Runtime error for ops given tensors on different devices; records the op, the first
    # common device and the conflicting device, and builds an explanatory message.
    def __init__(
        self,
        func: OpOverload,
        common_device: torch.device,
        device: torch.device,
    ) -> None:
        # [FUNCTION DOC] Stores func, common_device and device on the exception and initializes the message
        # saying at least two devices were found.
        self.func = func
        self.common_device = common_device
        self.device = device
        super().__init__(
            "Expected all tensors to be on the same device, but found "
            f"at least two devices, {common_device} and {device}!"
        )


@dataclass
class DynamicOutputShapeException(RuntimeError):
    # [CLASS DOC] Error (dataclass holding the op) raised when an operator's output shape depends on
    # something not computable without data, in a dynamic-output-shape way.
    func: OpOverload


@dataclass
class DataDependentOutputException(RuntimeError):
    # [CLASS DOC] Error (dataclass holding the op) raised when an operator's output depends on tensor data
    # values (data-dependent output) and cannot be faked without real values.
    func: OpOverload


@dataclass
class UnsupportedOperatorException(RuntimeError):
    # [CLASS DOC] Error (dataclass holding the op) raised when fake tensor mode has no meta/decomposition
    # support for an operator.
    func: OpOverload


@dataclass
class UnsupportedMutationAliasingException(RuntimeError):
    # [CLASS DOC] Error (dataclass with a reason string) raised when mutation plus aliasing in an
    # operation cannot be modeled by fake tensors.
    reason: str


@dataclass
class MetadataMismatchError(RuntimeError):
    # [CLASS DOC] Error (dataclass with a reason string) raised when fake output metadata does not match
    # the real output metadata during validation.
    reason: str


class FakeTensorTLS(threading.local):
    # [CLASS DOC] Thread-local state: allow_non_fake_inputs_override (None means defer to each mode's own
    # setting, otherwise overrides every mode in this thread) and a weak set tracking fake tensors made
    # during non-strict export, used for leak detection.
    # Default to None, otherwise it'll be used to override _all_
    # `FakeTensorMode.allow_non_fake_inputs` in this thread.
    allow_non_fake_inputs_override: bool | None
    non_strict_export_fake_tensor_tracker: weakref.WeakSet[FakeTensor]

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the override to None and the export tracker to an empty weak set for the
        # current thread.
        self.allow_non_fake_inputs_override = None
        self.non_strict_export_fake_tensor_tracker = weakref.WeakSet()


fake_tensor_tls = FakeTensorTLS()


def ordered_set(*items: T) -> dict[T, Literal[True]]:
    # [FUNCTION DOC] Returns an insertion-ordered mapping with the given items as keys (all mapped to a
    # true flag), i.e. an ordered set with duplicates removed.
    return dict.fromkeys(items, True)


@contextlib.contextmanager
def unset_fake_temporarily() -> Generator[TorchDispatchMode | None, None, None]:
    # [FUNCTION DOC] Context manager that removes the active FAKE dispatch mode, yields it (or None), and on
    # exit restores it if there was one.
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


@contextlib.contextmanager
def disable_fake_tensor_cache(fake_mode: FakeTensorMode) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that turns off the dispatch cache of the given fake mode and restores
    # the previous setting on exit, even on error.
    old_value: bool = fake_mode.cache_enabled
    try:
        fake_mode.cache_enabled = False
        yield
    finally:
        fake_mode.cache_enabled = old_value


def get_plain_tensors(
    subclass: Tensor | TraceableWrapperSubclass,
    *,
    out: list[Tensor | int | SymInt | CustomClassBase],
) -> list[Tensor | int | SymInt | CustomClassBase]:
    # [FUNCTION DOC] Flattens a traceable wrapper subclass into its innermost plain tensors/values by
    # iterating over the inner tensor names of each wrapper depth-first, appending leaves to the supplied
    # out list and returning it. Runtime path: intentionally no assertions.
    # This function is used in Runtime, do not add redundant asserts
    todo = [subclass]
    while todo:
        curr = todo.pop()
        if not is_traceable_wrapper_subclass(curr):
            out.append(curr)
            continue

        inner_keys, _ = curr.__tensor_flatten__()
        todo.extend(getattr(curr, key) for key in reversed(inner_keys))

    return out


def is_fake_tensor(x: object) -> TypeGuard[Tensor]:
    # [FUNCTION DOC] True if the object is a Python FakeTensor or a C++-level fake tensor; False otherwise.
    # True if x is itself a fake tensor: a Python FakeTensor, or a C++ fake.
    if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return True
    return isinstance(x, Tensor) and torch._C._is_fake_tensor(x)


def is_fake(x: object) -> TypeGuard[Tensor]:
    # [FUNCTION DOC] True if x is fake, looking through traceable wrapper subclasses (all tensor members
    # must agree; mixed fake and real raises an assertion error; non-tensor, non-custom-class members also
    # raise), FunctionalTensor, functionalized and functorch-wrapped tensors by unwrapping.
    from torch._subclasses.functional_tensor import FunctionalTensor

    if is_fake_tensor(x):
        return True
    if is_traceable_wrapper_subclass(x):
        attrs, _ = type(x).__tensor_flatten__(x)
        got_fake: bool | None = None
        for attr in attrs:
            match getattr(x, attr):
                case Tensor() as v:
                    fake = is_fake(v)
                    if got_fake is None:
                        got_fake = fake
                    elif got_fake != fake:
                        raise AssertionError("got mixed fake and real tensors!")
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return got_fake or False
    elif isinstance(x, FunctionalTensor):
        return is_fake(x.elem)
    elif isinstance(x, Tensor) and torch._is_functional_tensor(x):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(x, reapply_views)
        return is_fake(unwrapped)
    elif isinstance(x, Tensor) and is_functorch_wrapped_tensor(x):
        unwrapped = torch._C._functorch.get_unwrapped(x)
        return is_fake(unwrapped)
    return False


def maybe_get_fake_mode(t: object) -> FakeTensorMode | None:
    # [FUNCTION DOC] Returns the FakeTensorMode associated with a tensor-like object, recursing through
    # wrapper subclasses (all inner modes must be identical, else an assertion error), functional and
    # functorch wrapped tensors; None if there is none.
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return t.fake_mode
    if is_traceable_wrapper_subclass(t):
        inner_tensor_names, _ = t.__tensor_flatten__()
        mode: FakeTensorMode | None = None
        for t_name in inner_tensor_names:
            match getattr(t, t_name):
                case Tensor() as v:
                    m = maybe_get_fake_mode(v)
                    if mode is None:
                        mode = m
                    elif mode is not m:
                        raise AssertionError("All fake tensor modes must be the same")
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return mode
    elif isinstance(t, FunctionalTensor):
        return maybe_get_fake_mode(t.elem)
    elif isinstance(t, Tensor) and torch._is_functional_tensor(t):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(t, reapply_views)
        return maybe_get_fake_mode(unwrapped)
    elif isinstance(t, Tensor) and is_functorch_wrapped_tensor(t):
        unwrapped = torch._C._functorch.get_unwrapped(t)
        return maybe_get_fake_mode(unwrapped)
    return None


def maybe_get_real_tensor(x: object) -> Tensor | None:
    # [FUNCTION DOC] Returns the real tensor shadowing a FakeTensor, or None for anything else.
    if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return x.real_tensor
    return None


def maybe_get_fake_device(x: object) -> torch.device | None:
    # [FUNCTION DOC] Returns the fake device of a FakeTensor, or None for anything else.
    if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return x.fake_device
    return None


def maybe_get_fake_constant(x: object) -> Tensor | None:
    # [FUNCTION DOC] Returns the constant tensor value tracked by a FakeTensor, or None.
    if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return x.constant
    return None


@functools.cache
def get_schema_info(func: OpOverload) -> torch._C._SchemaInfo:
    # [FUNCTION DOC] Cached lookup building the C++ schema info object for an operator overload.
    return torch._C._SchemaInfo(func._schema)


# many of the decompositions registered to torch/_prims do not at the moment model
# aliasing or strides, so as an incremental step, just enable the decompositions in
# torch/_decomp/decompositions.py.
# decomps are used for aot autograd tracing so we would like to unify on their
# implementation and add additional testing to them
@functools.cache
def torch_decomp_decompositions(func: OpOverload) -> bool:
    # [FUNCTION DOC] Cached check whether the operator's decomposition table entry comes from the
    # torch._decomp package and is a function defined in torch._decomp.decompositions.
    from torch._decomp import decomposition_table

    decompositions = torch._decomp.decompositions
    # Note that the function in the decomposition table might be
    # different from the one in the module because of the difference
    # in out handling in aten API and torch public API
    return decomposition_table[func].__module__.startswith(
        "torch._decomp"
    ) and decomposition_table[func].__name__ in dir(decompositions)


def tree_flatten_only(ty: type[T], tree: PyTree) -> list[T]:
    # [FUNCTION DOC] Returns the leaves of a pytree that are instances of the given type, in order.
    flat_vals = pytree.tree_leaves(tree)
    return [elem for elem in flat_vals if isinstance(elem, ty)]


def _is_plain_tensor(t: object) -> bool:
    # [FUNCTION DOC] HELPER: True if t is exactly a base Tensor with strided layout that is not sparse,
    # nested, functorch-wrapped, batched or functionalized.
    return (
        type(t) is Tensor
        and t.layout == torch.strided
        and not (
            t.is_sparse
            or t.is_nested
            or is_functorch_wrapped_tensor(t)
            or is_legacy_batchedtensor(t)
            or torch._is_functional_tensor(t)
        )
    )


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakIdRef` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverter:
    # [CLASS DOC] Converts real tensors (or meta tensors plus a device) into FakeTensors while preserving
    # view/storage sharing through a MetaConverter memo (weak references). Also tracks, per weak storage
    # reference, the constant fake tensors sharing that storage so that mutation can invalidate all alias
    # constants. State: meta_converter, constant_storage_mapping, export flag (disables item memoization).
    @property
    def tensor_memo(
        self,
    ) -> weakref.WeakValueDictionary[
        torch._subclasses.meta_utils.MetaTensorId, FakeTensor
    ]:
        # [FUNCTION DOC] Property exposing the memo of the underlying meta converter mapping tensor ids to
        # FakeTensors.
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter[FakeTensor]
    constant_storage_mapping: dict[StorageWeakRef, list[ReferenceType[FakeTensor]]]
    export: bool

    def __init__(self, *, copy_data: bool = False, export: bool = False) -> None:
        # [FUNCTION DOC] Creates the meta converter (optionally copying data), records the export flag and
        # starts with an empty storage to constant tensors mapping.
        self.meta_converter = MetaConverter(copy_data=copy_data)
        self.export = export

        # map from to storage to corresponding constant tensors
        self.constant_storage_mapping = {}

    def add_constant_storage_mapping(self, fake_tensor: FakeTensor) -> None:
        # [FUNCTION DOC] Registers a constant FakeTensor under a weak reference to its constant's storage so
        # that later mutation can clear all aliases. Raises an assertion error if the argument is not a
        # FakeTensor with a constant.
        # when you have a constant, aliased tensor:
        # const_tensor.add_(torch.rand([1]))
        # all aliases of it must become no longer const
        if (
            not isinstance(  # noqa: ISINSTANCE_FAKE_TENSOR
                fake_tensor, FakeTensor
            )
            or fake_tensor.constant is None
        ):
            raise AssertionError("fake_tensor must be a FakeTensor with a constant")
        weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())

        # we need a map from a weak storage to all of its corresponding
        # constant tensors. python doesn't have the weak value equivalent
        # of defaultdict(list), so we are using a WeakValueDictionary as one
        if weak_st not in self.constant_storage_mapping:
            self.constant_storage_mapping[weak_st] = []
        self.constant_storage_mapping[weak_st].append(weakref.ref(fake_tensor))

    def invalidate_constant_aliases(self, tensor: Tensor) -> None:
        # [FUNCTION DOC] Given a real tensor mutated in place, clears the constant of every live FakeTensor
        # sharing its storage and drops that storage entry. Raises if given a FakeTensor.
        if isinstance(tensor, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
            raise AssertionError("Expected a real tensor, not a FakeTensor")

        weak_st = StorageWeakRef(tensor._typed_storage())
        if weak_st not in self.constant_storage_mapping:
            return

        for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
            ten = weak_tensor_ref()
            if ten is not None:
                # pyrefly: ignore [missing-attribute]
                ten._fix_weakref()
                ten.constant = None

        del self.constant_storage_mapping[weak_st]

    def clear_non_cpu_constants(self) -> None:
        # [FUNCTION DOC] Drops constants that are not on CPU (invalidating their aliases) and keeps CPU
        # constants, pruning dead or constant-less references from the storage mapping.
        """Clear non-CPU constants while keeping cheap CPU constants for folding."""
        for weak_st, weak_tensor_refs in list(self.constant_storage_mapping.items()):
            live_tensor_refs: list[ReferenceType[FakeTensor]] = []
            constant: Tensor | None = None
            for weak_tensor_ref in weak_tensor_refs:
                ten = weak_tensor_ref()
                if ten is None or ten.constant is None:
                    continue

                live_tensor_refs.append(weak_tensor_ref)
                if constant is None:
                    constant = ten.constant

            if constant is None:
                del self.constant_storage_mapping[weak_st]
            elif constant.device.type == "cpu":
                self.constant_storage_mapping[weak_st] = live_tensor_refs
            else:
                self.invalidate_constant_aliases(constant)

    def _get_memo(self, t: Tensor) -> FakeTensor | None:
        # [FUNCTION DOC] HELPER: Looks up an already created FakeTensor for a real tensor through the describer
        # id, or None.
        tid = self.meta_converter.describer.lookup_tensor.get(t)
        if tid is None:
            return None
        return self.tensor_memo.get(tid)

    def set_tensor_memo(self, t: Tensor, v: FakeTensor) -> None:
        # [FUNCTION DOC] Records v as the memoized fake tensor for the real tensor t.
        tid = self.meta_converter.describer.get_tensor_id(t)
        self.meta_converter.tensor_memo[tid] = v

    # You can have a real tensor that you need to convert into a fake tensor.
    # If you have a meta tensor already, call from_meta_and_device.
    #
    # You're allowed to pass a meta tensor to be turned into a fake
    # tensor; although an odd thing to do, this can occur if you're doing
    # cross ref testing and the inner test is already operating on meta tensors.
    def from_real_tensor(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        make_constant: bool = False,
        shape_env: ShapeEnv | None = None,
        *,
        source: Source | None = None,
        symbolic_context: SymbolicContext | None = None,
        trace: bool = True,
    ) -> FakeTensor:
        # [FUNCTION DOC] Converts a real (or meta) tensor into a FakeTensor, returning the memoized one when it
        # exists. Recovers source and symbolic context from the active tracing context when absent. Rejects
        # quantized tensors (UnsupportedFakeTensorException) and constants for parameters. Delegates to the
        # meta converter with a callback building FakeTensors, copies mkldnn dispatch keys and grad_dtype,
        # and for eligible 0-d CPU int/float64 tensors pre-computes the item value and memoizes it (as a
        # symbolic int/float when a source and shape env exist). Registers constants when make_constant.
        # see note [Tensor Fakification and Symbol Caching]
        if not symbolic_context and not source and shape_env:
            if tracing_context := torch._guards.TracingContext.try_get():
                if t in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[t]
                    from torch.fx.experimental.symbolic_shapes import (
                        StatefulSymbolicContext,
                    )

                    if not isinstance(symbolic_context, StatefulSymbolicContext):
                        raise AssertionError(
                            f"Expected StatefulSymbolicContext, got {type(symbolic_context)}"
                        )
                    source = symbolic_context.tensor_source

        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            if t.is_mkldnn and not maybe_memo.is_mkldnn:
                maybe_memo.dispatch_keys = torch._C._dispatch_keys(t)
            return maybe_memo
        # not yet supported in metatensors
        if t.is_quantized:
            raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        if type(t) is torch.nn.Parameter:
            if make_constant:
                raise AssertionError("make_constant must be False for nn.Parameter")

        constant = t if make_constant else None

        # This callback is used by both subclass and inner tensors. Require the
        # caller to explicitly specify the device in case outer and inner tensors
        # have different devices.
        def mk_fake_tensor(
            make_meta_t: Callable[[], object], device: torch.device | str
        ) -> FakeTensor:
            # [FUNCTION DOC] HELPER: Callback for the meta converter; builds a FakeTensor from a meta-tensor
            # producing function under no_dispatch, with an explicit device and the enclosing constant.
            # KNOWN ISSUE: the enclosing TODO notes the constant is that of the outer tensor, which is wrong when
            # the callback is used for inner tensors of subclasses.
            # NB: don't use in_kernel_invocation_manager. to
            # ensure FakeTensor can internally do constant computation
            # as necessary.  Invocation manager is "more correct" as
            # it works for more operators in make_meta_t, but
            # invariant is that make_meta_t only calls factories
            # for which it is not strictly necessary to use the
            # invocation manager (I think!)
            with no_dispatch():
                return FakeTensor(
                    fake_mode,
                    # pyrefly: ignore [bad-argument-type]
                    make_meta_t(),
                    # pyrefly: ignore [bad-argument-type]
                    device,
                    # TODO: callback might be used in recursive contexts, in
                    # which case using t is wrong!  BUG!
                    constant=constant,
                )

        out = self.meta_converter(
            t,
            shape_env=shape_env,
            callback=mk_fake_tensor,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )
        if out is NotImplemented:
            raise UnsupportedFakeTensorException("meta converter nyi")
        if t.is_mkldnn:
            out.dispatch_keys = torch._C._dispatch_keys(t)

        # Propagate grad_dtype here rather than in meta_converter because
        # meta tensors don't carry autograd metadata.
        # Unwrap FunctionalTensor because accessing is_leaf/grad_fn on a
        # FunctionalTensor view whose base was mutated (e.g. via set_())
        # triggers lazy view replay through __torch_dispatch__, which
        # errors without an active FunctionalTensorMode.
        inner_t = (
            torch._from_functional_tensor(t.elem)
            if isinstance(t, torch._subclasses.functional_tensor.FunctionalTensor)
            else t
        )
        if (
            inner_t.requires_grad
            and inner_t.is_leaf
            and inner_t.grad_dtype != inner_t.dtype
            and out.is_leaf
        ):
            out.grad_dtype = inner_t.grad_dtype

        from torch._dynamo.source import RandomValueSource

        value = None
        if (
            not self.export
            and _is_plain_tensor(t)  # mostly, we want to know if item() works
            and t.dim() == 0
            and t.device.type == "cpu"
            # All integer types are fair game, because signed overflow is UB
            # (and even int64 can overflow, since integers in Python are
            # arbitrary precision). But only float64 is OK for float, because
            # switching between float32 and float64 changes semantics in an
            # observable way without hitting UB.
            and t.dtype
            in [torch.int64, torch.int32, torch.int16, torch.int8, torch.float64]
            # Dynamo may pass shape_env=None for static fakification while
            # the mode still owns a ShapeEnv. In that case, do not install a
            # concrete Python scalar where Dynamo expects a SymNode-like memo.
            #
            # Similarly, compiler tracing can use a FakeTensorMode without a
            # ShapeEnv; keep concrete memos limited to truly manual/plain
            # FakeTensorMode usage so _local_scalar_dense tracing semantics do
            # not change under AOT/Inductor.
            and (
                source is not None
                or (
                    fake_mode.shape_env is None
                    and torch._guards.TracingContext.try_get() is None
                )
            )
            # Impede setting up item() on things coming from random.  These
            # are not "real" item() calls, instead UnspecializedPythonVariable
            # is unsafely pretending an int is a tensor, which can sometimes
            # implicitly cause an item call.  The problem is this is pretty
            # unsound: there's no reason substituting an int with a Tensor is
            # going to give the same results.  Today, you mostly get around
            # this by typically not having capture_scalar_outputs on and graph
            # breaking when someone tries to use the unspec variable in an
            # int-y context.  But allowing it through here would break that.
            # So don't.
            #
            # Once random values are setup to be represented as
            # SymNodeVariable, this condition can be removed.  To check if
            # you've done it right, this is a good test:
            #
            #   PYTORCH_TEST_WITH_DYNAMO=1 python test/test_reductions.py -k
            #   TestReductionsCPU.test_dim_reduction_fns_fn_name_amax_cpu_bfloat16
            and (source is None or not isinstance(source, RandomValueSource))
            # In Dynamo, shape_env is never none (even with static shapes).
            # However, FakeTensorMode can be used by hand and in some cases
            # ShapeEnv is not allocated.
            and (source is None or shape_env is not None)
        ):
            from torch._dynamo.source import CallMethodItemSource, FloatTensorSource
            from torch.fx.experimental.symbolic_shapes import DimDynamic

            with no_dispatch():
                value = t.item()
            if not math.isnan(value) and not math.isinf(value):
                if source is None:
                    # Plain FakeTensorMode may not have a ShapeEnv. Preserve
                    # the concrete scalar memo for real 0-D tensor inputs.
                    out.item_memo = value
                else:
                    if shape_env is None:
                        raise AssertionError("shape_env unexpectedly missing")
                    # Peephole strip out unnecessary torch.as_tensor(x).item()
                    if isinstance(source, FloatTensorSource):
                        item_source = source.base
                    else:
                        item_source = CallMethodItemSource(source)
                    symbol = shape_env.create_unspecified_symbol(
                        value,
                        source=item_source,
                        dynamic_dim=DimDynamic.DYNAMIC,
                        symbolic_context=symbolic_context,
                    )
                    # NB: reusing item_memo here ensures that we invalidate on
                    # mutation
                    if t.dtype == torch.int64:
                        out.item_memo = shape_env.create_symintnode(
                            symbol,
                            hint=value,
                            source=item_source,
                        )
                    elif t.dtype == torch.float64:
                        out.item_memo = shape_env.create_symfloatnode(
                            symbol,
                            hint=value,
                            source=item_source,
                        )
        if make_constant:
            self.add_constant_storage_mapping(out)
        # NB: meta_converter set the memo
        return out

    # If you specify the device, it MUST be a meta tensor.
    def from_meta_and_device(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        device: torch.device,
        pytype: type[torch.Tensor] | None = None,
        dispatch_keys: torch.DispatchKeySet | None = None,
    ) -> FakeTensor:
        # [FUNCTION DOC] Wraps a meta tensor and an explicit device into a FakeTensor (memoized per meta
        # tensor). Raises an assertion error if t is not on the meta device.
        if t.device.type != "meta":
            raise AssertionError(
                f"tensor's device must be `meta`, got {t.device.type} instead"
            )
        # This is a bit abusive (this is not the "real" tensor) but whatever,
        # the meta tensor should be fresh so there's no way to get it wrong
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensor(
            fake_mode, t, device, pytype=pytype, dispatch_keys=dispatch_keys
        )
        self.set_tensor_memo(t, out)
        return out


@functools.cache
def init_gpu_context(device: torch.device) -> None:
    # [FUNCTION DOC] Cached per device: for accelerator devices, allocates a tiny tensor so the backend
    # context is initialized before backward runs on fake tensors.
    # Backward will error with cuda Fake Tensors if no cuda tensors have been initialized first
    if torch.accelerator.current_accelerator(True) == device.type:
        (
            torch.empty(1, device=device)
            if torch.version.hip is None
            else torch.zeros(1, device=device)
        )


@contextlib.contextmanager
def in_kernel_invocation_manager(
    fake_mode: FakeTensorMode,
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that marks the mode as inside a kernel invocation: asserts the
    # thread-local meta include flag agrees with the mode state, disables torch dispatch, sets the meta
    # dispatch-include flag, and restores the mode flag afterwards.
    # See: note [Fake Tensor Dispatch Keys]
    prev_in_kernel = fake_mode.in_kernel_invocation
    meta_in_tls = torch._C._meta_in_tls_dispatch_include()
    if meta_in_tls != prev_in_kernel:
        raise AssertionError(f"{meta_in_tls}, {prev_in_kernel}")

    with torch._C._DisableTorchDispatch():
        fake_mode.in_kernel_invocation = True
        # Unfortunately _set_meta_in_tls_dispatch_include(False) can leave
        # `Dense` turned on (because it's implied by `Meta`)
        with torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                yield
            finally:
                fake_mode.in_kernel_invocation = prev_in_kernel
                # torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)


# Return if the function allows Python numbers to bind to Tensors
def should_allow_numbers_as_tensors(func: OpOverload) -> bool:
    # [FUNCTION DOC] True if the op (by base name) allows Python numbers to bind to tensor arguments.
    return torch._C._should_allow_numbers_as_tensors(
        func.name().split("::")[-1].split(".")[0]
    )


class FakeTensorConfig:
    # [CLASS DOC] Configuration holder; debug is true when the environment variable
    # TORCH_FAKE_TENSOR_DEBUG equals 1, causing FakeTensors to capture creation tracebacks.
    debug = os.environ.get("TORCH_FAKE_TENSOR_DEBUG", "0") == "1"


# This memorizes unbacked SymInt or SymFloats representing quantities like the
# number of nonzero elements in this tensor or learning rate. There is one
# instance of the descriptor per particular quantity to memoize.
#
# Memoization is helpful if you do something like x[mask] and y[mask];
# mask.nonzero() gets repeatedly called and should give a consistent unbacked
# SymInt. It needs to be invalidated in the same way constant is.
#
# Making this a descriptor may seem overly fancy, but actually it's the most
# convenient way to ensure access to FakeTensor during access, which is
# required for testing version counter and epoch validity.
class SymNumberMemoDescriptor:
    # [CLASS DOC] Descriptor memoizing an unbacked symbolic int/float (e.g. nonzero count, item value) on a
    # FakeTensor under private attributes (value, version counter, epoch). Reads return None when the
    # tensor was mutated (version mismatch, skipped for inference tensors and nested ints) or the mode's
    # epoch changed (not for nested ints or backed SymFloats). Writes record the version and epoch.
    _name: str

    # By default, SymInts in this memo are invalidated across versions/epochs.
    # nested_ints however are preserved across epochs and across versions.
    # Preserving across versions is okay for nested int since the association
    # of a nested int is agnostic to the underlying data and nested ints are not
    # shared across multiple distinct tensors.
    _is_nested_int: bool

    def __init__(self, *, is_nested_int: bool = False) -> None:
        # [FUNCTION DOC] Stores whether this memo is for a nested int (preserved across versions and epochs).
        self._is_nested_int = is_nested_int

    def __set_name__(self, owner: str, name: str) -> None:
        # [FUNCTION DOC] Records the attribute name the descriptor is bound to.
        self._name = name

    def _memo(self, obj: FakeTensor) -> str:
        # [FUNCTION DOC] HELPER: Name of the private attribute holding the memo value.
        return f"_{self._name}"

    def _memo_vc(self, obj: FakeTensor) -> str:
        # [FUNCTION DOC] HELPER: Name of the private attribute holding the version counter at memo time.
        return f"_{self._name}_vc"

    # When we retrace, we need to invalidate all the memos so that we can
    # accurately identify the first time unbacked SymInts are allocated.
    # This is only relevant for inputs; for intermediates, they will get fresh
    # fake tensors so you won't have a memo anyway
    def _memo_epoch(self, obj: FakeTensor) -> str:
        # [FUNCTION DOC] HELPER: Name of the private attribute holding the fake mode epoch at memo time.
        return f"_{self._name}_epoch"

    def __get__(
        self, obj: FakeTensor, objtype: type[FakeTensor] | None = None
    ) -> torch.SymInt | torch.SymFloat | None:
        # [FUNCTION DOC] Returns the memoized value for the tensor, or None if unset or invalidated by version
        # counter change or epoch change (invalidation clears the stored value). Backed SymFloats survive epoch
        # changes but not mutation.
        if (r := getattr(obj, self._memo(obj))) is None:
            return None

        # Version counter based tracking isn't 100% sound but it's close
        # enough.  Inference tensors don't track version counters, so
        # skip that check for them.
        if not self._is_nested_int and not obj.is_inference():
            if getattr(obj, self._memo_vc(obj), None) != obj._version:
                setattr(obj, self._memo(obj), None)
                return None

        # Backed SymFloats are stable across retracing epochs, but tensor
        # mutation (version counter check above) still invalidates the memo.
        if isinstance(r, torch.SymFloat) and r.node.hint is not None:
            return r

        if (
            not self._is_nested_int
            and getattr(obj, self._memo_epoch(obj), None) != obj.fake_mode.epoch
        ):
            setattr(obj, self._memo(obj), None)
            return None

        return r

    def __set__(
        self,
        obj: FakeTensor,
        value: torch.SymInt | torch.SymFloat | torch.SymBool | int | float | None,
    ) -> None:
        # [FUNCTION DOC] Stores or clears the memo: None clears value, version and epoch; otherwise stores the
        # value, the current version (unless nested int or inference tensor) and the mode epoch.
        if value is None:
            setattr(obj, self._memo(obj), None)
            setattr(obj, self._memo_vc(obj), None)
            setattr(obj, self._memo_epoch(obj), None)
        else:
            setattr(obj, self._memo(obj), value)
            if not self._is_nested_int and not obj.is_inference():
                setattr(obj, self._memo_vc(obj), obj._version)
            setattr(obj, self._memo_epoch(obj), obj.fake_mode.epoch)


class FakeTensor(Tensor):
    # [CLASS DOC] Tensor subclass wrapping a meta tensor and carrying a fake_device, the owning fake_mode,
    # optional constant (real value for tiny constants) and real_tensor, pytype/dispatch_keys overrides,
    # and memo descriptors (nonzero, item, unique, unique_consecutive, nested_int). Operations dispatch
    # through __torch_dispatch__ into the fake mode. Reports the fake device unless the mode is inside a
    # kernel invocation, when it reports meta.
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """

    _fake_device: torch.device
    fake_mode: FakeTensorMode
    constant: Tensor | None
    real_tensor: Tensor | None

    # TODO: Generalize this as needed, e.g., into a trie of memos, if
    # you do something like x[0].item()  (x[0] is fresh each time, so
    # memo mechanism here won't work)
    nonzero_memo: SymNumberMemoDescriptor | int | None = SymNumberMemoDescriptor()
    item_memo = SymNumberMemoDescriptor()
    unique_memo: SymNumberMemoDescriptor | int | None = SymNumberMemoDescriptor()
    unique_consecutive_memo: SymNumberMemoDescriptor | int | None = (
        SymNumberMemoDescriptor()
    )

    # We expect nested_int_memo to be None when an offsets is a graph
    # intermediate, or an input that has never been associated with a
    # nested int.
    nested_int_memo = SymNumberMemoDescriptor(is_nested_int=True)

    # FakeTensor doesn't fully emulate the original tensor's Python type
    # and dispatch key set, therefore sometimes we want to track them
    # separately.
    pytype: type[Tensor] | None
    dispatch_keys: torch.DispatchKeySet | None

    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FAKE

    @property
    # pyrefly: ignore [bad-override]
    def is_mkldnn(self) -> bool:
        # [FUNCTION DOC] Property: True if the dispatch keys include Mkldnn.
        return _dispatch_keys_has_mkldnn(self.dispatch_keys)

    @property
    # pyrefly: ignore [bad-override]
    def layout(self) -> torch.layout:
        # [FUNCTION DOC] Property: layout; returns the mkldnn layout object for mkldnn tensors, else the base
        # tensor layout.
        if self.is_mkldnn:
            return torch._mkldnn  # type: ignore[attr-defined]
        return Tensor.layout.__get__(self, type(self))  # type: ignore[attr-defined]

    def to_dense(
        self,
        dtype: torch.dtype | None = None,
        *,
        masked_grad: bool | None = None,
    ) -> Tensor:
        # [FUNCTION DOC] Converts to a dense tensor; mkldnn fakes go through the aten _to_dense op, others use
        # the base implementation.
        if self.is_mkldnn:
            return torch.ops.aten._to_dense.default(
                self, dtype=dtype, masked_grad=masked_grad
            )
        return Tensor.to_dense(self, dtype=dtype, masked_grad=masked_grad)

    @property
    # pyrefly: ignore [bad-override]
    def device(self) -> torch.device:
        # [FUNCTION DOC] Property: meta device while inside a kernel invocation, otherwise the fake device.
        if self.fake_mode.in_kernel_invocation:
            return torch.device("meta")
        else:
            return self.fake_device

    @device.setter
    def device(self, _: torch.device) -> None:
        # [FUNCTION DOC] Setter that always raises NotImplementedError (device cannot be assigned).
        raise NotImplementedError

    @property
    def fake_device(self) -> torch.device:
        # [FUNCTION DOC] Property: the fake device.
        return self._fake_device

    @fake_device.setter
    def fake_device(self, device: torch.device) -> None:
        # [FUNCTION DOC] Setter: normalizes the device (index filling, GPU context init) and stores it.
        self._fake_device = self._normalize_fake_device(device)

    # Note: [Fake Tensor Dispatch Keys]
    # In order to model the behavior of device-specific autocast
    # and autograd logic, we update the dispatch keys of FakeTensors
    # to reflect their fake device. This includes the BackendComponent
    # (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
    # related Autocast and Autograd keys. __torch_dispatch__ sits below
    # Autocast and Autograd, and is only invoked when we are at the
    # kernel for the BackendComponent. Then, we add Meta to the
    # thread-local dispatch include set to hit the meta kernel
    # instead of the kernel of the BackendComponent for the fake device.
    # The `device_for_backend_keys` does that below
    # NOTE: this probably will not do the right thing for backends
    # that have dispatch keys which are higher than the "meta" key:
    # https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h#L189

    @staticmethod
    def _normalize_fake_device(device: torch.device) -> torch.device:
        # [FUNCTION DOC] HELPER: Static; initializes GPU context for cuda/xpu devices and, for indexed device
        # types without an index, fills in the current device index if the backend is initialized (never for
        # mps) or index 0.
        """Normalize device by initializing GPU context and setting device index."""
        if device.type in ("cuda", "xpu"):
            init_gpu_context(device)

        if _is_indexed_device_type(device.type) and device.index is None:
            if device.type != "mps" and getattr(torch, device.type).is_initialized():
                device = torch.device(
                    f"{device.type}:{getattr(torch, device.type).current_device()}"
                )
            else:
                device = torch.device(f"{device.type}:0")
        return device

    @staticmethod
    def __new__(
        cls,
        fake_mode_or_elem: FakeTensorMode | Tensor | None = None,
        elem: Tensor | None = None,
        device: torch.device | str | None = None,
        constant: Tensor | None = None,
        real_tensor: Tensor | None = None,
        pytype: type[Tensor] | None = None,
        dispatch_keys: torch.DispatchKeySet | None = None,
        *,
        fake_mode: FakeTensorMode | None = None,
        fake_device: torch.device | str | None = None,
        _fake_device: torch.device | str | None = None,
        requires_grad: bool | None = None,
        _is_param: bool = False,
        **state: Unpack[_FakeTensorConstructorIgnoredState],
    ) -> Self:
        # [FUNCTION DOC] Constructor creating the subclass instance from a meta tensor. Accepts either
        # (fake_mode, elem, ...) or (elem, fake_mode=...) forms, discards known private state keys, rejects
        # unknown keywords and conflicting duplicate mode or device values (TypeError/ValueError), requires
        # a device from device, fake_device or _fake_device, requires elem on meta, forbids meta fake device
        # unless the mode allows it, forbids a FakeTensor real_tensor, sets data pointer access policy, and
        # initializes fields and memos (and a creation traceback in debug mode).
        state_dict = cast(dict[str, object], state)
        for attr_name in _FAKE_TENSOR_CONSTRUCTOR_IGNORED_STATE_ATTRS:
            state_dict.pop(attr_name, None)
        if state_dict:
            unexpected = next(iter(state_dict))
            raise TypeError(
                "FakeTensor.__new__() got an unexpected keyword argument "
                f"{unexpected!r}"
            )

        if isinstance(fake_mode_or_elem, FakeTensorMode):
            if fake_mode is not None and fake_mode is not fake_mode_or_elem:
                raise TypeError("FakeTensor.__new__() got conflicting fake_mode values")
            fake_mode = fake_mode_or_elem
        else:
            # Support bounded state-based reconstruction from known FakeTensor
            # attrs, e.g. type(fake_tensor)(new_elem, **fake_tensor.__dict__).
            if fake_mode_or_elem is not None:
                if not isinstance(fake_mode_or_elem, Tensor):
                    raise TypeError(
                        "FakeTensor.__new__() expected a FakeTensorMode or Tensor "
                        f"as the first argument, got {type(fake_mode_or_elem)}"
                    )
                if elem is not None:
                    raise TypeError(
                        "FakeTensor.__new__() expected either "
                        "(fake_mode, elem, ...) or (elem, fake_mode=..., ...)"
                    )
                elem = fake_mode_or_elem
            if fake_mode is None:
                raise TypeError(
                    "FakeTensor.__new__() missing required argument: 'fake_mode'"
                )

        if elem is None:
            raise TypeError("FakeTensor.__new__() missing required argument: 'elem'")

        specified_devices = [
            (name, FakeTensor._normalize_fake_device(torch.device(value)))
            for name, value in (
                ("device", device),
                ("fake_device", fake_device),
                ("_fake_device", _fake_device),
            )
            if value is not None
        ]
        if specified_devices:
            first_name, resolved_device = specified_devices[0]
            for name, candidate_device in specified_devices[1:]:
                if resolved_device != candidate_device:
                    raise ValueError(
                        "FakeTensor.__new__() got conflicting device values: "
                        f"{first_name}={resolved_device}, {name}={candidate_device}"
                    )
            device = resolved_device
        else:
            raise TypeError(
                "FakeTensor.__new__() missing required argument: "
                "'device' (or 'fake_device')"
            )

        self = Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad if requires_grad is None else requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )
        if not fake_mode._allow_unsafe_data_ptr_access:
            torch._C._set_throw_on_mutable_data_ptr(self)
        else:
            torch._C._set_warn_deprecated_on_mutable_data_ptr(self)

        if elem.device.type != "meta":
            raise AssertionError(
                f"elem.device.type must be 'meta', got {elem.device.type}"
            )
        # NB: it is fine, if a little confusing, for device to be meta
        # (we are faking a meta tensor in that case).  However, it often
        # indicates some sort of confusion (e.g., you accidentally passed
        # in a meta tensor when you should have passed in the real tensor).
        # So by default we disallow meta, and if you are working in a situation
        # where it is helpful (e.g., crossref testing) you can turn it back
        # on
        if not fake_mode.allow_meta:
            if device.type == "meta":
                raise AssertionError(
                    "device.type must not be 'meta' when allow_meta is False"
                )
        self.fake_device = device
        self.fake_mode = fake_mode
        self.constant = constant
        self.pytype = pytype
        self.dispatch_keys = dispatch_keys
        if isinstance(real_tensor, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
            raise AssertionError("real_tensor must not be a FakeTensor")
        self.real_tensor = real_tensor
        self.nonzero_memo = None
        self.item_memo = None
        self.unique_memo = None
        self.unique_consecutive_memo = None
        self.nested_int_memo = None

        if _is_param:
            self._is_param = _is_param

        if FakeTensorConfig.debug:
            self._debug_trace = CapturedTraceback.extract()  # type: ignore[attr-defined]
        return self

    # In some circumstances, a conventional Tensor constructor
    # will get rewritten to call into FakeTensor.  We must provide an
    # __init__ method that can accept the Python interpreters initialization
    # in such a situation; we must also be able to handle direct fake
    # tensor construction via FakeTensor().
    #
    # In particular, the __init__ call will look funny in the following case:
    #
    #   with FakeTensorMode():
    #       x = Tensor([1, 2, 3])
    #
    # this desugars into:
    #
    #   with FakeTensorMode():
    #       x = Tensor.__new__([1, 2, 3])
    #       # NB: x is a fake tensor, because of the mode!
    #       x.__init__([1, 2, 3])  # not the normal fake tensor args!
    #
    def __init__(self, *args: object, **kwargs: object) -> None:
        # [FUNCTION DOC] Initializer accepting arbitrary arguments (required for tensor construction rewrites
        # under a mode); registers the tensor in the export leak tracker when exporting with leak detection.
        super().__init__()
        if (
            torch.compiler.is_exporting()
            and torch._export.config.detect_non_strict_fake_tensor_leaks
        ):
            fake_tensor_tls.non_strict_export_fake_tensor_tracker.add(self)

    @staticmethod
    def from_tensor(t: Tensor, fake_mode: FakeTensorMode) -> FakeTensor:
        # [FUNCTION DOC] Static convenience: converts a real tensor to a FakeTensor via the given mode.
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(  # type: ignore[override] # TODO
        cls,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        # [FUNCTION DOC] Dispatch entry point for FakeTensor arguments. Handles device queries directly,
        # dispatches meta handlers (size/stride etc.), returns NotImplemented for unrecognized subclasses or
        # when a fake mode is already active, otherwise finds the mode from the arguments and re-runs the op
        # inside that mode.
        # need to handle here to avoid infinite recursion
        # see [in_kernel_invocation]
        if func is torch.ops.prim.device.default:
            if len(args) != 1 or not isinstance(args[0], FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                raise AssertionError(
                    "Expected exactly one FakeTensor argument for prim.device.default"
                )
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

        # this handler must be done inside FakeTensor subclass, not mode, because
        # we can end up dispatching here when we have a fake tensor with
        # symbolic sizes running under in_kernel_invocation_manager.
        # The subclass is asked to handle this query because size (not
        # sym_size) was called, but we are unable to serve it directly because
        # there are symbolic sizes in the class.  The use of
        # in_kernel_invocation_manager means it's incorrect to activate a
        # mode to actually handle this (this caused
        # https://github.com/pytorch/pytorch/issues/122772).
        if handler := _DISPATCH_META_HANDLERS.get(func):
            return handler(args)

        # Because fake mode can return NotImplemented (if it sees a subclass
        # it doesn't know how to deal with), this test here is important
        # because the next dispatch after a fake mode will attempt to use
        # subclasses of tensors to dispatch, and any FakeTensor arguments
        # will be considered eligible.
        unrecognized_types = [
            t for t in types if not issubclass(t, FakeTensor) and t is not Tensor
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FakeTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        fake_mode = None
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                fake_mode = arg.fake_mode
                break

        if fake_mode is None:
            raise AssertionError("Could not find a FakeTensor in the arguments")

        # If the fake mode is already active, don't try to reapply it!
        # NotImplemented is the right thing to return here, because the
        # typical situation this can occur is if ProxyTensorMode returned a
        # NotImplemented because of a not implemented subclass; we may have
        # unluckily attempted to hit FakeTensor's dispatch first,
        # NotImplemented lets us keep chaining until we find the actual
        # subclass
        maybe_cur_fake_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.FAKE
        )
        if maybe_cur_fake_mode:
            not_implemented_log.debug(
                "FakeTensor mode already active: %s in %s",
                fake_mode,
                maybe_cur_fake_mode,
            )
            return NotImplemented

        if fake_mode.in_kernel_invocation:
            raise AssertionError("fake_mode.in_kernel_invocation must be False")

        with fake_mode:
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(
        func: OpOverload, flat_args: Sequence[object]
    ) -> tuple[torch.device, bool]:
        # [FUNCTION DOC] HELPER: Static; scans flat args for FakeTensors and returns (common_device,
        # has_scalar_only_inputs). A CPU zero-dim tensor defers to (or is overridden by) other devices unless
        # the op bypasses that rule; some ops (foreach copy, in-place add with meta rhs) may legally mix
        # devices; a configured preferred device type wins ties; otherwise a device mismatch raises
        # FakeTensorDeviceMismatchError. With no tensor but an op accepting numbers, returns cpu and True.
        # Raises an assertion error if no device can be found.
        # Returns: (common_device, has_scalar_only_inputs)

        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite the common_device if it the only existing
        # device comes from a cpu zero-dim tensor
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        # list of ops which can have args(tensor/tensorList) in mixed device
        mixed_device_fns = ordered_set(
            aten._foreach_copy.default,
        )

        # These in-place ops keep the destination tensor's device even if the
        # rhs was explicitly constructed on meta.
        meta_rhs_mixed_device_fns = ordered_set(
            aten.add_.Tensor,
        )

        # list of ops not using zero dim cpu tensor logic to align with the eager mode.
        bypass_zero_dim_cpu_tensor_check_ops = ordered_set(
            aten.nextafter.default,
        )

        def is_device_cpu(device: torch.device) -> bool:
            # [FUNCTION DOC] HELPER: True if device type is cpu.
            return device.type == "cpu"

        def is_device_meta(device: torch.device) -> bool:
            # [FUNCTION DOC] HELPER: True if device type is meta.
            return device.type == "meta"

        def cpu_zero_dim(t: Tensor) -> bool:
            # [FUNCTION DOC] HELPER: True if the tensor is a zero-dim tensor on cpu.
            return is_device_cpu(t.device) and t.dim() == 0

        def merge_devices(t: object) -> None:
            # [FUNCTION DOC] HELPER: Folds one argument's device into the enclosing common_device and
            # is_cpu_zero_dim state (non-FakeTensors ignored); raises FakeTensorDeviceMismatchError for an
            # irreconcilable mismatch.
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            if not isinstance(t, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                return

            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            t_is_cpu_zero_dim = cpu_zero_dim(t)
            if t.device == common_device:
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            is_bypass_zero_dim_cpu_tensor_check_op = (
                func in bypass_zero_dim_cpu_tensor_check_ops
            )

            # mismatching devices !
            # if current tensor is cpu 0 dim, defer to existing device
            if t_is_cpu_zero_dim and not is_bypass_zero_dim_cpu_tensor_check_op:
                return

            # current device is from cpu 0 dim tensor, overwrite
            if is_cpu_zero_dim and not is_bypass_zero_dim_cpu_tensor_check_op:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # if still device mismatches we will check ops which can work
            # on different devices for ex. _foreach_copy, and one of the
            # device must be cpu in this case we will return from here without
            # throwing an error
            if func in mixed_device_fns:
                if any(map(is_device_cpu, (common_device, t.device))):
                    return

            if func in meta_rhs_mixed_device_fns:
                if any(map(is_device_meta, (common_device, t.device))):
                    return

            # if prefer_device_type is set, prefer that device type over others
            prefer_device_type = torch._functorch.config.fake_tensor_prefer_device_type
            if prefer_device_type is not None:
                common_has_preferred = prefer_device_type in common_device.type
                t_has_preferred = prefer_device_type in t.device.type

                if not common_has_preferred and t_has_preferred:
                    # Switch to the preferred device type
                    common_device = t.device
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                    return
                elif common_has_preferred and not t_has_preferred:
                    # Keep the existing preferred device type
                    return

            # mismatching devices of non-zero dim tensors, throw
            # This might be valid behavior and need to be explicitly modeled, e.g. reshape_as
            raise FakeTensorDeviceMismatchError(func, common_device, t.device)

        for arg in flat_args:
            merge_devices(arg)

        # some functions that allow Python numbers to bind to Tensors
        # if we have failed to find a device, and we're running one of these operators,
        # we must have scalar only inputs
        if should_allow_numbers_as_tensors(func) and common_device is None:
            # ops with scalar only inputs always have result on cpu
            has_scalar_only_inputs = True
            common_device = torch.device("cpu")

        if common_device is None:
            raise AssertionError(f"Could not find common device for {func}")

        return common_device, has_scalar_only_inputs

    def get_nested_int(
        self,
        *,
        coeff: int | torch.SymInt = 1,
    ) -> torch.SymInt:
        # [FUNCTION DOC] Returns the nested int associated with this tensor multiplied by coeff, creating and
        # memoizing a symbolic nested int from the fake mode when none exists.
        if self.nested_int_memo is None:
            self.nested_int_memo = self.fake_mode.create_symbolic_nested_int(
                nt_tensor_id=None
            )
        if not isinstance(self.nested_int_memo, torch.SymInt):
            raise AssertionError(
                f"Expected nested_int_memo to be torch.SymInt, got {type(self.nested_int_memo)}"
            )
        return self.nested_int_memo * coeff

    # Similar to FunctionalTensor.tolist
    def tolist(self) -> Any:
        # [FUNCTION DOC] Converts to Python values by recursion: scalar item for 0-dim, list of items for 1-dim, list of nested lists otherwise.
        if self.dim() == 0:
            return self.item()
        elif self.dim() == 1:
            return [elem.item() for elem in self]
        else:
            return [elem.tolist() for elem in self]


_MetadataIntLike = Union[IntLikeType, "_PySymInputStub", "_SymIntOutputStub"]


@dataclass(slots=True)
class TensorMetadata:
    # [CLASS DOC] Slotted dataclass of the tensor properties relevant to cache-key hashing of a FakeTensor:
    # dtype, shape, stride, device, layout, suggested memory format, storage offset/bytes, autograd and
    # conj/neg/inference/quantized/sparse flags and sparse dims.
    """
    The Tensor metadata relevant to hashing FakeTensors when caching.
    """

    dtype: torch.dtype
    shape: tuple[_MetadataIntLike, ...]
    stride: tuple[_MetadataIntLike, ...]
    device: torch.device
    layout: torch.layout
    memory_format: torch.memory_format | None
    storage_offset: _MetadataIntLike
    storage_bytes: _MetadataIntLike | None
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool  # read: is sparse COO
    is_coalesced: bool | None
    dense_dim: int | None
    sparse_dim: int | None

    def _flatten_into(
        self,
        result: list[object],
        mode: FakeTensorMode,
        state: _CacheKeyState,
    ) -> None:
        # [FUNCTION DOC] HELPER: Appends every field value to result for building a cache key; sequences are
        # flattened through the mode's argument preparation and symbolic ints go through the key state's
        # SymInt conversion.
        # Flatten the TensorMetadata out into `result`.  Make sure to call
        # state.convert_sym_int() on any SymInts.
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (tuple, list, torch.Size)):
                # This will recursively flatten the iterable, calling
                # convert_sym_int() as necessary.
                id_hashed_objects: list[object] = []
                mode._prep_args_for_hash(result, value, state, id_hashed_objects)
                id_hashed_objects.clear()
            elif isinstance(value, SymInt):
                state.convert_sym_int(result, value)
            else:
                result.append(value)


def extract_tensor_metadata(t: Tensor) -> TensorMetadata:
    # [FUNCTION DOC] Builds a TensorMetadata snapshot of a tensor. Memory format is suggested only for
    # non-symbolic, non-sparse tensors and only kept if the tensor is contiguous in it; strides empty for
    # non-strided layouts; storage bytes and sparse info only where applicable.
    """
    Extract the TensorMetadata of a tensor.
    """
    # Read layout/sparseness once (hot-path Python properties on FakeTensor).
    layout = t.layout
    _is_sparse_any: bool = is_sparse_any(t)
    # Don't call is_contiguous() on a Tensor which has symbolic sizes or things
    # will go badly (guards will be messed up?). suggest_memory_format walks
    # sizes and strides, which is symbolic arithmetic for such a tensor, so
    # skip it too rather than computing a value we are about to discard.
    memory_format: torch.memory_format | None
    if t._has_symbolic_sizes_strides or _is_sparse_any:
        memory_format = None
    else:
        memory_format = suggest_memory_format(t)
        if not t.is_contiguous(memory_format=memory_format):
            memory_format = None

    storage_offset = t.storage_offset()

    return TensorMetadata(
        t.dtype,
        t.shape,
        t.stride() if layout == torch.strided else (),
        t.device,
        layout,
        memory_format,
        storage_offset,
        # Only set storage_bytes for tensors that have storage (not sparse)
        t.untyped_storage().nbytes() if not _is_sparse_any else None,
        t.requires_grad,
        t.is_quantized,
        t.is_conj(),
        t.is_neg(),
        t.is_inference(),
        t.is_sparse,
        t.is_coalesced() if t.is_sparse else None,
        t.dense_dim() if _is_sparse_any else None,
        t.sparse_dim() if _is_sparse_any else None,
    )


@dataclass(slots=True)
class _DispatchCacheKey:
    # [CLASS DOC] HELPER: Hashable key for the dispatch cache wrapping a tuple of key parts with a
    # precomputed hash; equality compares the tuples.
    """
    Key for the FakeTensor dispatch cache.
    """

    key: tuple[object, ...]
    hashvalue: int

    def __init__(self, tup: tuple[object, ...]) -> None:
        # [FUNCTION DOC] Stores the tuple and its hash.
        self.key = tup
        self.hashvalue = hash(tup)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal if other is a cache key with equal tuple.
        return isinstance(other, _DispatchCacheKey) and self.key == other.key

    def __hash__(self) -> int:
        # [FUNCTION DOC] Returns the precomputed hash.
        return self.hashvalue

    def strip_shape_env(self) -> None:
        # [FUNCTION DOC] Strips shape-env references from symbolic input stubs inside the key so cached keys do not keep shape envs alive.
        # We need to strip the ShapeEnv from any values before we store in the
        # cache so the cache doesn't keep our ShapeEnvs alive.
        for v in self.key:
            if isinstance(v, _PySymInputStub):
                v.strip_shape_env()


# Default value for constant_value in _DispatchCacheEntryOutputInfo. This is
# only for checking and differentiates from None.
class SingletonConstant:
    # [CLASS DOC] HELPER: Sentinel type used as default for constant_value in cache output info to tell
    # "no constant" apart from a constant of None.
    pass


@dataclass(frozen=True, slots=True)
class _DispatchCacheEntryOutputInfo:
    # [CLASS DOC] HELPER: Frozen cache record for one op output: index of the aliased in-place argument,
    # or tensor metadata (plus optional index of the viewed argument) used to synthesize a new
    # FakeTensor, or a constant value (the default sentinel means none).
    """
    Entry type for the FakeTensor dispatch cache for an output. Accounts for three
    possibilities:
    1) The op is inplace, and a hit means we need to alias the argument at a
       given index.
    2) We need to synthesize a new FakeTensor given tensor metadata. For view
       ops, we further capture the index of the arg to alias.
    3) if the tensor related fields are None, then it is a constant value (e.g.
    None or integer)
    """

    inplace_idx: int | None
    metadata: TensorMetadata | None
    view_idx: int | None
    constant_value: Any | None = SingletonConstant


@dataclass(frozen=True, slots=True)
class _DispatchCacheValidEntry:
    # [CLASS DOC] HELPER: Positive cache entry: a tuple of per-output infos and a flag saying whether the
    # op returns a tuple.
    """
    Entry type for the FakeTensor dispatch cache. It supports two types of outputs
    1) tensor
    2) tuple of tensors

    is_output_tuple flag helps in differentiating the return type
    """

    output_infos: tuple[_DispatchCacheEntryOutputInfo]
    is_output_tuple: bool = False


@dataclass(frozen=True, slots=True)
class _DispatchCacheBypassEntry:
    # [CLASS DOC] HELPER: Negative cache entry recording the reason why caching is bypassed for a key.
    """
    Entry type for a negative cache entry.
    """

    reason: str


if TYPE_CHECKING:
    _DispatchCacheEntry = _DispatchCacheValidEntry | _DispatchCacheBypassEntry


@dataclass(frozen=True, slots=True)
class _BypassDispatchCache(Exception):
    # [CLASS DOC] HELPER: Exception raised internally to signal that this call must skip the dispatch
    # cache; carries a reason string.
    """
    Signals cases that should skip FakeTensor caching.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class DispatchCacheInfo:
    # [CLASS DOC] Frozen snapshot of cache statistics: hits, misses, per-reason bypass counts and size.
    """
    Information about the state of the FakeTensor dispatch cache.
    """

    hits: int
    misses: int
    bypasses: dict[str, int]
    size: int


# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with FakeTensorMode()`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly increase.


class FakeTensorMode(TorchDispatchMode):
    # [CLASS DOC] Infra dispatch mode producing FakeTensors. Holds class-level shared dispatch cache and
    # stats (cache, hits, misses, bypasses), an epoch advanced on retrace, in_kernel_invocation flag,
    # instance state: converter, optional ShapeEnv, static_shapes, allow_meta, fallback and non-fake input
    # policies, cache flags, nested tensor id counter, and a stack of enter records so re-entry is
    # idempotent. Entering installs the mode; dispatch resolves ops by cache lookup, meta kernels,
    # decompositions, constant propagation or fallbacks.
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: dict[str, int] = defaultdict(int)
    # Every time you retrace using the same fake tensor mode, you should
    # advance the epoch so we don't reuse unbacked memos
    epoch: int = 0
    in_kernel_invocation: bool = False
    static_shapes: bool
    shape_env: ShapeEnv | None
    _stack: str | None
    allow_meta: bool

    # NestedTensor uses a tensor_id_counter to uniquely identify offsets.
    # This counter is incremented when an offsets is used to create an NJT
    # for the first time. To avoid mutating eager state if we construct NJT
    # during tracing, we maintain a separate counter on the FakeTensorMode.
    # The initial count is set to the current eager tensor_id_counter value
    # upon initialization, and every time you retrace using the same fake tensor
    # mode, you should reset the counter to the initial count.
    nt_tensor_id_counter: int = -1
    nt_tensor_id_initial_count: int = -1

    def __init__(
        self,
        *,
        allow_fallback_kernels: bool = True,
        allow_non_fake_inputs: bool = False,
        shape_env: ShapeEnv | None = None,
        static_shapes: bool | None = None,
        # TODO: This is a temporary measure, see
        # https://github.com/pytorch/pytorch/pull/126245#discussion_r1604185748
        # We're currently solely using this to impede population of
        # item_memo for 0d scalar tensor inputs when export, because this
        # causes things that used to be deferred runtime asserts to turn into
        # guards, and then the guards are just lost.  We can potentially fix
        # this by ensuring guards also get put in the graph, but this is
        # pending a rework of how deferred runtime asserts in export.  Once
        # that's done, we can remove this.
        export: bool = False,
    ) -> None:
        # [FUNCTION DOC] Creates the mode: reads functorch/dynamo config for real tensor propagation, data
        # pointer policy, meta allowance and cache flags; builds the FakeTensorConverter; decides static
        # shapes (default: no shape env); sets mode key to FAKE; initializes enter stack, creation stack
        # trace and nested tensor id counters from the eager counter.
        log.debug("create_mode 0x%x", id(self))
        super().__init__()
        self.allow_fallback_kernels = allow_fallback_kernels

        import torch._dynamo.config
        import torch._functorch.config

        self.propagate_real_tensors = (
            torch._functorch.config.fake_tensor_propagate_real_tensors
        )
        self.fake_tensor_converter = FakeTensorConverter(
            copy_data=self.propagate_real_tensors,
            export=export,
        )

        if static_shapes is not None:
            self.static_shapes = static_shapes
        else:
            self.static_shapes = shape_env is None

        # This is temporarily patched to True in Dynamo to grandfather in some
        # places where we unconditionally allow scalar outputs, TO BE REMOVED
        self.allow_scalar_outputs = False

        self._allow_unsafe_data_ptr_access = (
            torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access
        )
        self.allow_meta = torch._functorch.config.fake_tensor_allow_meta
        self.cache_enabled: bool = (
            torch._dynamo.config.fake_tensor_cache_enabled
            and not self.propagate_real_tensors
        )
        self.cache_crosscheck_enabled = (
            torch._dynamo.config.fake_tensor_cache_crosscheck_enabled
        )

        # A flag that controls, whether we want to invoke ops on mix of
        # real weights/global variables and fake inputs
        self.allow_non_fake_inputs = allow_non_fake_inputs

        # [in_kernel_invocation]
        # when FakeTensor is invoked in user code, .device should return
        # the fake_device of the tensor so that code such as `if x.is_cuda`
        # or torch.zeros([10, 10], device=x.device) continues to execute as if
        # the FakeTensor were real. However, within kernel execution, we return
        # the `Meta` device because all computation within the kernels should
        # behave as if the Tensors are on meta devices. Kernels should allocate
        # new tensors on meta devices, and checks like `is_meta` should return true.
        # within python refs, we always return the real device by defining
        # the device property
        self.in_kernel_invocation = False

        # True if we enter'ed and actually enabled fake tensor mode,
        # false if it was a no-op.  Not thread safe but neither is
        # in_kernel_invocation
        # If another fake mode was already active when we enter, we also stash it here.
        # That way when we exit, we know to re-enable the previous fake mode.
        self.enter_stack: list[tuple[bool, TorchDispatchMode | None, bool | None]] = []

        self.shape_env = shape_env

        self._stack_trace = traceback.extract_stack()
        self._stack = None

        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.FAKE

        import torch.nested._internal.nested_tensor

        self.nt_tensor_id_initial_count = (
            torch.nested._internal.nested_tensor._tensor_id_counter
        )
        self.nt_tensor_id_counter = self.nt_tensor_id_initial_count

    def reset_nt_tensor_id_counter(self) -> None:
        # [FUNCTION DOC] Resets the nested tensor id counter to its initial count (call on retrace).
        self.nt_tensor_id_counter = self.nt_tensor_id_initial_count

    # Typically, there is only one fake tensor mode and you test for it by
    # doing an isinstance test.  However, in some situations, there might be
    # TWO fake tensor modes.  The canonical example of this is exporting
    # a fake model: there is an outer fake mode created by the user, and
    # an inner fake mode created by Dynamo.  The two phase process is required
    # because the outer fake mode typically won't have a ShapeEnv, even if
    # the user is interested in exporting with dynamic shapes (so the inner
    # fake mode will actually have a ShapeEnv and swap in symbolic sizes.)
    #
    # In this case, it's insufficient to test only one FakeTensor: you need
    # to distinguish between our fake tensor and other fake tensors.  That's
    # what this function does.
    def is_our_fake(self, t: object) -> TypeGuard[FakeTensor]:
        # [FUNCTION DOC] True if t is a FakeTensor created by this exact mode (not another fake mode).
        return (
            isinstance(  # noqa: ISINSTANCE_FAKE_TENSOR
                t, FakeTensor
            )
            and t.fake_mode is self
        )

    # If we should avoid device init. This changes the behavior of various APIs:
    # - We avoid constant-prop on Tensors with ops that move them to another device
    # - We change the torch.tensor ctor contract to never materialize
    #   tensors on device
    #   (see NOTE: [torch.tensor, lift_fresh, and device movement])
    @property
    def avoid_device_init(self) -> bool:
        # [FUNCTION DOC] Property: True when there is no usable accelerator (no available cuda/xpu/hpu/privateuse1
        # device), in which case device init is avoided. Raises an assertion error if both xpu and cuda are compiled.
        if torch.xpu._is_compiled():
            if torch.cuda._is_compiled():
                raise AssertionError("Cannot have both xpu and cuda compiled")
            return not torch.xpu.is_available()

        return not (
            torch.cuda.is_available()
            or (hasattr(torch, "hpu") and torch.hpu.is_available())
            or _is_privateuse1_backend_available()
        )

    @property
    def stack(self) -> str:
        # [FUNCTION DOC] Property: lazily formatted creation stack trace string of this mode.
        if self._stack is None:
            self._stack = "".join(traceback.format_list(self._stack_trace))
        return self._stack

    @count
    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        # [FUNCTION DOC] Dispatch hook; asserts no fake mode is currently set, delegates to dispatch, and logs
        # then re-raises TypeError.
        # FakeTensorMode should not be set when we're inside of it.
        if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
            raise AssertionError(
                f"FakeTensorMode should not be set when dispatching {func}"
            )
        try:
            return self.dispatch(func, types, args, kwargs)
        except TypeError:
            log.exception("fake tensor raised TypeError")
            raise

    # No-op if FakeTensorMode is already in use
    def __enter__(self) -> Self:
        # [FUNCTION DOC] Enters the mode (no-op if already the active fake mode). On machines without a usable
        # accelerator, restricts lifting to CPU tensors and sets a no-op CUDA device guard. Pushes an enter
        # record (whether it truly installed the mode, previous fake mode, previous lift flag).
        import torch.nested._internal.nested_tensor

        prev_only_lift_cpu_tensors = None
        if self.avoid_device_init:
            # See NOTE: [torch.tensor, lift_fresh, and device movement]
            prev_only_lift_cpu_tensors = torch._C._only_lift_cpu_tensors()
            torch._C._set_only_lift_cpu_tensors(True)

            # In the case of CPU-only build or cuda device unavailable,
            # we patch the cuda device guard to use NoOpDeviceGuardImpl.
            # This enables us to trace over cuda kernels under FakeTensorMode.
            torch._C._ensureCUDADeviceGuardSet()

        maybe_prev_fake_mode = torch._C._unset_dispatch_mode(self._mode_key)
        if self is not maybe_prev_fake_mode:
            self.enter_stack.append(
                (True, maybe_prev_fake_mode, prev_only_lift_cpu_tensors)
            )
            return super().__enter__()
        else:
            # no-op (still need to re-set the fake mode though since we unset it)
            torch._C._set_dispatch_mode(self)
            self.enter_stack.append((False, None, prev_only_lift_cpu_tensors))

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Pops the enter record and, only if this entry truly installed the mode, exits it and
        # restores the previous fake mode and cpu-lift flag.
        (live, maybe_prev_fake_mode, maybe_prev_only_lift_cpu_tensors) = (
            self.enter_stack.pop()
        )
        if live:
            super().__exit__(exc_type, exc_val, exc_tb)

            # Re-enable the previous fake mode, if there was one.
            if maybe_prev_fake_mode is not None:
                torch._C._set_dispatch_mode(maybe_prev_fake_mode)
            if maybe_prev_only_lift_cpu_tensors is not None:
                torch._C._set_only_lift_cpu_tensors(maybe_prev_only_lift_cpu_tensors)

    @classmethod
    def is_infra_mode(cls) -> bool:
        # [FUNCTION DOC] Class method: always True (this is an infra mode with lower precedence).
        return True

    @classmethod
    def cache_info(cls) -> DispatchCacheInfo:
        # [FUNCTION DOC] Class method returning a DispatchCacheInfo of hits, misses, bypass counts and cache size.
        """
        Query the state of the dispatch cache.
        """
        return DispatchCacheInfo(
            FakeTensorMode.cache_hits,
            FakeTensorMode.cache_misses,
            dict(FakeTensorMode.cache_bypasses),
            len(FakeTensorMode.cache),
        )

    @classmethod
    def cache_clear(cls) -> None:
        # [FUNCTION DOC] Class method: resets cache statistics and clears the dispatch cache.
        """
        Clear the dispatch cache.
        """
        cls.cache_hits = 0
        cls.cache_misses = 0
        cls.cache_bypasses.clear()
        cls.cache.clear()

    def _cached_dispatch_impl(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> object:
        # [FUNCTION DOC] HELPER: Dispatch with cache: builds a cache key (on bypass records the reason and
        # dispatches uncached), looks up the cache in either the shape env cache or the global one, and on
        # hit reconstructs outputs (negative entries fall back to computing).
        """
        Lookup a cache entry for the given arguments. If none exists, dispatch
        and cache the result (if the result is eligible for caching).
        """
        state = None
        key = None
        try:
            state = _CacheKeyState(self.shape_env)
            key = self._cache_key(state, func, args, kwargs)
        except _BypassDispatchCache as e:
            # We couldn't create the cache key at all
            if (
                isinstance(func, torch._ops.HigherOrderOperator)
                and func.name() == "invoke_subgraph"
            ):
                hc_log.debug(
                    "Fake tensor cache failed: identifier = %s, reason = %s",
                    args[1],
                    e.reason,
                )
            FakeTensorMode.cache_bypasses[e.reason] += 1

        if key is None:
            # Do this dispatch outside the above except handler so if it
            # generates its own exception there won't be a __context__ caused by
            # the caching mechanism.

            return self._dispatch_impl(func, types, args, kwargs)

        if state is None:
            raise AssertionError("state must not be None after cache key generation")
        if state.cache_on_shape_env():
            if state.shape_env is None:
                raise AssertionError(
                    "state.shape_env must not be None when caching on shape_env"
                )
            cache = state.shape_env.fake_tensor_cache
            set_cache_key = _set_cache_key_for_shape_env
        else:
            cache = FakeTensorMode.cache
            set_cache_key = _set_cache_key
        entry = cache.get(key, None)

        if entry is not None:
            if isinstance(entry, _DispatchCacheBypassEntry):
                # This represents a negative cache entry - we already saw that the
                # output is uncachable. Compute it from first principals.
                FakeTensorMode.cache_bypasses[entry.reason] += 1

                return self._dispatch_impl(func, types, args, kwargs)

            # We have a cache entry.

            output = self._output_from_cache_entry(state, entry, key, func, args)
            FakeTensorMode.cache_hits += 1
            if self.cache_crosscheck_enabled:
                # For debugging / testing: Validate that the output synthesized
                # from the cache matches the output created by normal dispatch.
                with disable_fake_tensor_cache(self):
                    self._crosscheck_cache_output(output, func, types, args, kwargs)
            return output

        # We don't have a cache entry.

        output = self._dispatch_impl(func, types, args, kwargs)

        try:
            entry = self._make_cache_entry(state, key, func, args, kwargs, output)
        except _BypassDispatchCache as e:
            # We ran "extra" checks on the cache key and determined that it's no
            # good. Record the reason and mark it so we don't bother validating
            # again.
            if (
                isinstance(func, torch._ops.HigherOrderOperator)
                and func.name() == "invoke_subgraph"
            ):
                hc_log.debug(
                    "Fake tensor cache failed: identifier = %s, reason = %s",
                    args[1],
                    e.reason,
                )
            FakeTensorMode.cache_bypasses[e.reason] += 1
            set_cache_key(cache, key, _DispatchCacheBypassEntry(e.reason))
            return output

        set_cache_key(cache, key, entry)
        FakeTensorMode.cache_misses += 1
        return output

    def _cache_key(
        self,
        state: _CacheKeyState,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> _DispatchCacheKey:
        # [FUNCTION DOC] HELPER: Builds the dispatch cache key for a call from the op, default dtype, default
        # device, inference mode, shape env settings, tracing flag, epoch (when symbols are known) and the
        # prepared args/kwargs. Registers weakref finalizers on id-hashed objects (graph modules, subgraphs)
        # to evict the key when they die. Raises _BypassDispatchCache when the call cannot be cached.
        # KNOWN ISSUE: the cache key creator does not call the validity check itself; validation is done when entries are made.
        """
        Create a cache key given the dispatch args. Raises _BypassDispatchCache
        for any situation that precludes caching.
        """
        is_tracing = torch.fx.experimental.proxy_tensor.get_proxy_mode() is not None
        key_values = [
            func,
            # Capture the default_dtype mode since that can affect the output tensor,
            # e.g., when operating on constant float values.
            torch.get_default_dtype(),
            # Capture the current device to support, e.g., cache tensor creation,
            # where there isn't necessarily a tensor to take the device from.
            torch._C._get_default_device(),
            # We want to create tensors from cached metadata only when the inference
            # mode is the same.
            torch.is_inference_mode_enabled(),
            # Shape env settings could affect behavior. One example seen in the wild:
            # Disallowing dynamic shapes can introduce a DynamicOutputShapeException
            # where it wasn't seen on a previous instance of the same op.
            self.shape_env.settings if self.shape_env else None,
            # ProxyTorchDispatchMode needs to track how SymNodes are constructed
            # so we need to handle things a little different depending on
            # whether we're tracing or not.
            is_tracing,
        ]
        if state.known_symbols:
            # If there are symbols then include the epoch - this is really more
            # of a Shape env var which lives on the FakeTensorMode.
            # pyrefly: ignore [bad-argument-type]
            key_values.append(self.epoch)
        # Collect the id_hashed objects to attach a weakref finalize later
        id_hashed_objects: list[object] = []
        # Translate any FakeTensor args to metadata.
        if args:
            # pyrefly: ignore [bad-argument-type]
            self._prep_args_for_hash(key_values, args, state, id_hashed_objects)
        if kwargs:
            # pyrefly: ignore [bad-argument-type]
            self._prep_args_for_hash(key_values, kwargs, state, id_hashed_objects)
        key = _DispatchCacheKey(tuple(key_values))

        for id_hashed_obj in id_hashed_objects:
            weakref.finalize(
                id_hashed_obj, functools.partial(evict_fake_tensor_cache_key, key=key)
            )
        id_hashed_objects.clear()
        return key

    def _validate_cache_key(
        self,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        # [FUNCTION DOC] HELPER: Raises _BypassDispatchCache for ops that cannot be safely cached: data
        # dependent or dynamic output shape ops (except index with only non-bool/int8 indices), inplace views,
        # unsafe view, as_strided prims, lift functions, mkldnn conversion, inductor storage resize, non-builtin
        # ops, and view ops with a CompositeImplicitAutograd kernel. Registered higher-order ops are
        # deferred to entry creation.
        """
        Validate that the cache key generated by _cache_key will be
        reasonable.
        """
        from torch._higher_order_ops.utils import registered_hop_fake_fns

        # For hops, we perform the validity check in _make_cache_entry  because we
        # need to have the output tensor.
        if (
            isinstance(func, torch._ops.HigherOrderOperator)
            and func in registered_hop_fake_fns
        ):
            return

        # Avoid caching for any ops that would require a more sophisticated
        # caching implementation, e.g., data dependent ops or ops that modify
        # the inputs.
        if torch.Tag.data_dependent_output in func.tags:
            raise _BypassDispatchCache("data dependent output")

        if torch.Tag.dynamic_output_shape in func.tags:
            if func is aten.index.Tensor:
                _, new_kwargs = normalize_function(  # type: ignore[misc]
                    func,
                    args=args,  # type: ignore[arg-type]
                    kwargs=kwargs,  # type: ignore[arg-type]
                    normalize_to_only_use_kwargs=True,
                )
                for index in new_kwargs["indices"]:
                    # index calls nonzero for bool or int8 tensors, and
                    # therefore has a dynamic shape output. For other dtypes,
                    # the output shape depends on the input shape (and not data)
                    if isinstance(index, torch.Tensor) and index.dtype in (
                        torch.bool,
                        torch.int8,
                    ):
                        raise _BypassDispatchCache("dynamic output shape")
                return

            raise _BypassDispatchCache("dynamic output shape")

        if torch.Tag.inplace_view in func.tags:
            raise _BypassDispatchCache("inplace view")

        if func is aten._unsafe_view.default:
            raise _BypassDispatchCache("unsafe view")

        if func is torch.ops.prims.as_strided.default:
            raise _BypassDispatchCache("prims.as_strided")

        if func in self.lift_fns:
            raise _BypassDispatchCache("lift")

        if func is aten.to_mkldnn.default:
            raise _BypassDispatchCache("mkldnn tensor")

        if func.name() == "inductor::resize_storage_bytes_":
            raise _BypassDispatchCache("inductor::resize_storage_bytes_")

        if not torch._library.utils.is_builtin(func):
            raise _BypassDispatchCache("non-builtin")

        # In order to handle storage aliasing, we need to establish the alias
        # for any view op on a cache hit. But CompositeImplicitAutograd ops may
        # or may not alias the input, so just punt on caching these.
        if func.is_view and torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            raise _BypassDispatchCache("CompositeImplicitAutograd")

    def _prep_args_for_hash(
        self,
        result: list[object],
        args: Mapping[str, object] | Sequence[object] | Iterable[object],
        state: _CacheKeyState,
        id_hashed_objects: list[object],
    ) -> None:
        # [FUNCTION DOC] HELPER: Appends a hashable translation of args to result: container types and
        # lengths, FakeTensor metadata (bypass if not this mode's fake, has constant, sparse or mkldnn), SymInt
        # via the key state, graph modules/subgraph wrappers by identity (collected in id_hashed_objects), and
        # other values with their type. Bypasses for non-fake tensors, SymBool/SymFloat and function arguments.
        """
        Translate the provided args into a form suitable for caching at FakeTensor
        dispatch, i.e., convert unhashable types like lists & dicts into tuples and
        convert FakeTensors into metadata. Raises _BypassDispatchCache to signal
        unsupported cases that should bypass caching.
        """
        from torch._higher_order_ops.auto_functionalize import (
            FunctionalCallableWithEpilogue,
        )
        from torch._higher_order_ops.utils import SubgraphCallableWrapper

        if isinstance(args, (list, tuple, dict)):
            result.append(type(args))
            result.append(f"length_{len(args)}")

        if isinstance(args, dict):
            self._prep_args_for_hash(result, args.keys(), state, id_hashed_objects)
            self._prep_args_for_hash(result, args.values(), state, id_hashed_objects)
            return

        for arg in args:
            if isinstance(arg, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                if not self.is_our_fake(arg):
                    raise _BypassDispatchCache("not our fake")
                if arg.constant is not None:
                    raise _BypassDispatchCache("constant attribute")
                if is_sparse_any(arg):
                    raise _BypassDispatchCache(f"{arg.layout} tensor")
                if arg.is_mkldnn:
                    raise _BypassDispatchCache("mkldnn tensor")
                metadata = extract_tensor_metadata(arg)
                metadata._flatten_into(result, self, state)
            elif isinstance(arg, Tensor):
                raise _BypassDispatchCache("non-fake tensor")
            elif isinstance(arg, SymInt):
                state.convert_sym_int(result, arg)
            elif isinstance(arg, (SymBool, SymFloat)):
                raise _BypassDispatchCache("symbolic shape")
            elif isinstance(arg, (list, tuple, dict)):
                self._prep_args_for_hash(result, arg, state, id_hashed_objects)
            elif isinstance(arg, types.FunctionType):
                raise _BypassDispatchCache("function argument")
            elif isinstance(arg, torch.fx.GraphModule):
                # This is used for invoke_subgraph where id(graph_module) allows
                # us to cache fake outputs
                result.append(type(arg))
                result.append(id(arg))
                id_hashed_objects.append(arg)
            elif isinstance(arg, SubgraphCallableWrapper):
                result.append(hash(arg))
                id_hashed_objects.append(arg.subgraph)
            elif isinstance(arg, FunctionalCallableWithEpilogue):
                result.append(type(arg))
                result.append(hash(arg))
                id_hashed_objects.append(arg.orig_callable)
            else:
                # It's important to capture the type of the arg since, e.g., 1 and 1.0
                # hash to the same value, but can produce different dtypes for the
                # output tensor.
                result.append(type(arg))
                result.append(arg)

    def _validate_output_for_cache_entry(
        self,
        state: _CacheKeyState,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        output: FakeTensor | None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Checks an output is cacheable: ints and None pass; symbolic content is
        # validated; must be a FakeTensor without constant, not sparse or sparse compressed, and not aliased
        # by any keyword value. Raises _BypassDispatchCache otherwise. The key and func parameters are unused.
        # Is this even possible? According to the signature this can be None but
        # not `int`. So either the signature is a lie or (part of) this line is
        # unnecessary...
        if isinstance(output, (int, type(None))):
            return

        # Check for symbolic content that should bypass caching - raises
        # _BypassDispatchCache if necessary.
        _validate_symbolic_output_for_caching(state, output)

        # Some ops return tuples of Tensors, but it's rare, so avoid
        # the complexity of caching other types.
        if not isinstance(output, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
            raise _BypassDispatchCache("non-FakeTensor output")

        # Avoid caching FakeTensors with constants attached since those
        # can be invalidated.
        if output.constant is not None:
            raise _BypassDispatchCache("constant attribute")

        # TODO: support caching sparse outputs?
        if output.is_sparse:
            raise _BypassDispatchCache("sparse output")

        if is_sparse_compressed(output):
            raise _BypassDispatchCache("sparse compressed output")

        # Can an in-place op really reference a kwarg? If so, then we need
        # to extend the implementation to handle it.
        for kval in kwargs.values():
            if id(kval) == id(output):
                raise _BypassDispatchCache("kwarg aliases output")

    def _get_output_info_for_cache_entry(
        self,
        state: _CacheKeyState,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        output: FakeTensor,
    ) -> _DispatchCacheEntryOutputInfo:
        # [FUNCTION DOC] HELPER: Builds the per-output cache record: constant for int/SymInt/None outputs, the
        # index of the aliased argument for in-place ops, else metadata with outputs converted to stubs (and
        # the viewed argument index for view ops). Verifies that the output synthesized from the record has
        # the same dispatch key set; bypasses on mismatch or data-dependent errors.
        if isinstance(output, (int, torch.SymInt, type(None))):
            return _DispatchCacheEntryOutputInfo(
                inplace_idx=None, metadata=None, view_idx=None, constant_value=output
            )

        # If this is an in-place op, the entry records which input arg is aliased.
        for idx in range(len(args)):
            if id(args[idx]) == id(output):
                return _DispatchCacheEntryOutputInfo(
                    inplace_idx=idx, metadata=None, view_idx=None
                )

        # Otherwise, create an entry that records the output tensor's metadata.
        view_idx = None
        if isinstance(func, torch._ops.OpOverload) and func.is_view:
            idxs = [i for i, t in enumerate(args) if isinstance(t, Tensor)]
            if len(idxs) != 1:
                raise AssertionError(
                    f"Expected exactly one tensor arg for view op, got {len(idxs)}"
                )
            view_idx = idxs[0]

        metadata = extract_tensor_metadata(output)
        metadata.shape = tuple(state.convert_output(v) for v in metadata.shape)
        metadata.stride = tuple(state.convert_output(v) for v in metadata.stride)
        metadata.storage_offset = state.convert_output(metadata.storage_offset)
        metadata.storage_bytes = (
            None
            if metadata.storage_bytes is None
            else state.convert_output(metadata.storage_bytes)
        )

        entry = _DispatchCacheEntryOutputInfo(
            inplace_idx=None,
            metadata=metadata,
            view_idx=view_idx,
        )

        # N.B.: Some checks for bypassing the cache would be performed on the
        # output tensor synthesized from the cached metadata. As an optimization,
        # we can synthesize a tensor here and do the checks on that instance.
        # This approach keeps the (more frequent) cache-hit path as lightweight
        # as possible.
        entry_for_synth_output = _DispatchCacheValidEntry(
            output_infos=(entry,), is_output_tuple=False
        )
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

        try:
            synth_output = self._output_from_cache_entry(
                state, entry_for_synth_output, key, func, args
            )
        except GuardOnDataDependentSymNode:
            # This should probably never really happen. If it does it means that
            # although the original call didn't get a data-dependent error when
            # we tried to reconstruct the output we did - that's almost
            # certainly a bug.
            raise _BypassDispatchCache("data dependent symnode") from None

        # Make sure the dispatch_key_set from the synthesized output tensor will
        # be the same.
        synth_key_set = torch._C._dispatch_key_set(synth_output)
        key_set = torch._C._dispatch_key_set(output)
        if synth_key_set != key_set:
            raise _BypassDispatchCache("dispatch_key_set mismatch")

        return entry

    def _make_cache_entry(
        self,
        state: _CacheKeyState,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        output: FakeTensor | None,
    ) -> _DispatchCacheValidEntry:
        # [FUNCTION DOC] HELPER: Builds a valid cache entry for an op output (single value or tuple) after
        # validating the key and the output. For registered higher-order ops, bypasses if the output has
        # unbacked symbols or the op skips fake-tensor caching. Raises _BypassDispatchCache.
        # KNOWN ISSUE: the bypass reason string for HOPs contains a line continuation backslash with embedded whitespace, so the text includes extra spaces.
        """
        Make a cache entry object for the given 'output' Tensor. Raises
        _BypassDispatchCache if the output tensor has characteristics that
        prevent caching it.
        """
        from torch._higher_order_ops.utils import (
            hops_that_skip_faketensor_cache,
            registered_hop_fake_fns,
        )
        from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

        self._validate_cache_key(func, args, kwargs)

        # For hops, lets look at the output tensor to find any unbacked symints.
        # If there are none, then we rely on the existing checks to validate
        # caching.
        # NB: Note that the HOPs that sta alive till FakeTensor are functional,
        # once they support mutations, we will have to revisit this logic.
        # Skipping caching for HOPs that used to be registered with @py.impl(FakeTensorMode)
        # to preserve original behaviour
        if (
            isinstance(func, torch._ops.HigherOrderOperator)
            and func in registered_hop_fake_fns
        ):
            outs = output if isinstance(output, tuple) else (output,)
            non_cacheable = (
                any(
                    isinstance(o, (torch.Tensor, torch.SymInt))
                    and has_free_unbacked_symbols(o)
                    for o in outs
                )
                or func in hops_that_skip_faketensor_cache
            )
            if non_cacheable:
                raise _BypassDispatchCache(
                    f"unbacked symbol in HOP {func} output \
                    or {func} in hops_that_skip_faketensor_cache"
                )

        if isinstance(output, (int, torch.SymInt, type(None))):
            output_info = _DispatchCacheEntryOutputInfo(
                inplace_idx=None, metadata=None, view_idx=None, constant_value=output
            )
            return _DispatchCacheValidEntry(
                output_infos=(output_info,), is_output_tuple=False
            )

        if isinstance(output, tuple):
            for out_element in output:
                self._validate_output_for_cache_entry(
                    state,
                    key,
                    func,
                    args,
                    kwargs,
                    out_element,
                )
        else:
            self._validate_output_for_cache_entry(
                state,
                key,
                func,
                args,
                kwargs,
                output,
            )

        if isinstance(output, tuple):
            output_infos = [
                self._get_output_info_for_cache_entry(
                    state,
                    key,
                    func,
                    args,
                    kwargs,
                    out_elem,
                )
                for out_elem in output
            ]
            return _DispatchCacheValidEntry(
                # pyrefly: ignore [bad-argument-type]
                output_infos=tuple(output_infos),
                is_output_tuple=True,
            )

        else:
            output_info = self._get_output_info_for_cache_entry(
                state,
                key,
                func,
                args,
                kwargs,
                output,
            )
            return _DispatchCacheValidEntry(
                output_infos=(output_info,), is_output_tuple=False
            )

    def _get_output_tensor_from_cache_entry(
        self,
        state: _CacheKeyState,
        entry: _DispatchCacheEntryOutputInfo,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
    ) -> FakeTensor | None:
        # [FUNCTION DOC] HELPER: Reconstructs one output from a record: returns the stored constant, the
        # aliased argument for in-place ops, or synthesizes a new FakeTensor by allocating a meta strided
        # tensor from cached metadata (resolving symbolic stubs through the shape env), applying conj/neg
        # bits and re-attaching the viewed argument's storage for view ops.
        # KNOWN ISSUE: the sparse check passes a metadata record to the sparse predicate, which is meant for tensors.
        if (
            entry.inplace_idx is None
            and entry.metadata is None
            and entry.view_idx is None
        ):
            if entry.constant_value is SingletonConstant:
                raise AssertionError(
                    "entry.constant_value must not be SingletonConstant"
                )
            return entry.constant_value
        if entry.inplace_idx is not None:
            # This is an in-place op; return the aliased arg.
            inplace_arg = args[entry.inplace_idx]
            if not isinstance(inplace_arg, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                raise AssertionError("inplace_arg must be a FakeTensor")
            return inplace_arg

        # Synthesize a new FakeTensor with the cached metadata.
        metadata = entry.metadata
        if metadata is None:
            return None

        if is_sparse_any(metadata):
            raise AssertionError("Sparse tensors are not supported in cache")

        def check_value(value: _MetadataIntLike, state: _CacheKeyState) -> IntLikeType:
            # [FUNCTION DOC] HELPER: Resolves a cached size/stride/offset value: symbolic output stubs are
            # extracted from the shape env; unresolved input stubs raise an assertion error; plain values pass through.
            if isinstance(value, _SymIntOutputStub):
                if state.shape_env is None:
                    raise AssertionError(
                        "state.shape_env must not be None for _SymIntOutputStub"
                    )
                return value.extract(key, state.shape_env)
            else:
                if isinstance(value, _PySymInputStub):
                    raise AssertionError("Unexpected _PySymInputStub value")
                return value

        shape = tuple(check_value(v, state) for v in metadata.shape)
        stride = tuple(check_value(v, state) for v in metadata.stride)
        storage_offset = check_value(metadata.storage_offset, state)
        if metadata.storage_bytes is not None:
            check_value(metadata.storage_bytes, state)

        maybe_suppress: Callable[[], typing.ContextManager[None]] = (
            contextlib.nullcontext
        )
        if self.shape_env is not None:
            maybe_suppress = self.shape_env.suppress_guards

        with in_kernel_invocation_manager(self), maybe_suppress():
            empty = torch.empty_strided(
                shape,
                stride,
                dtype=metadata.dtype,
                layout=metadata.layout,
                device="meta",
                requires_grad=metadata.requires_grad,
            )

        if metadata.is_conj:
            torch._C._set_conj(empty, True)
        if metadata.is_neg:
            torch._C._set_neg(empty, True)

        if isinstance(func, torch._ops.OpOverload) and func.is_view:
            # For view ops, the storage should be the same as the tensor input.
            view_arg = args[cast(int, entry.view_idx)]
            if not isinstance(view_arg, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                raise AssertionError("view_arg must be a FakeTensor")
            storage = view_arg.untyped_storage()
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(storage, storage_offset, shape, stride)

        return FakeTensor(self, empty, metadata.device)

    def _output_from_cache_entry(
        self,
        state: _CacheKeyState,
        entry: _DispatchCacheValidEntry,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
    ) -> FakeTensor | None | tuple[FakeTensor | None, ...]:
        # [FUNCTION DOC] HELPER: Builds the output of a cached call from a valid entry: a tuple of
        # outputs when the entry says so, otherwise the single output.
        """
        Create a new FakeTensor from the cache entry.
        """

        if entry.is_output_tuple:
            outputs = [
                self._get_output_tensor_from_cache_entry(
                    state, output_info, key, func, args
                )
                for output_info in entry.output_infos
            ]
            return tuple(outputs)
        else:
            return self._get_output_tensor_from_cache_entry(
                state, entry.output_infos[0], key, func, args
            )

    def _crosscheck_cache_output(
        self,
        output: FakeTensor | None | tuple[FakeTensor | None, ...],
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        # [FUNCTION DOC] HELPER: Debug check that recomputes the real dispatch result and compares it with the
        # cache-synthesized output (ints, None, symbolic types, tensor metadata, tuples). Raises RuntimeError
        # describing the call on any failure or mismatch.
        """
        Helper to validate that the output synthesized from the cache matches
        the output created by normal dispatch.
        """

        def assert_helper(a: Any, b: Any) -> None:
            # [FUNCTION DOC] HELPER: Recursively asserts two outputs agree; raises AssertionError on mismatch and
            # RuntimeError for unsupported types.
            if isinstance(a, tuple):
                if not isinstance(b, tuple):
                    raise AssertionError(f"Expected tuple, got {type(b)}")
                if len(a) != len(b):
                    raise AssertionError(f"Tuple length mismatch: {len(a)} != {len(b)}")
                for l, r in zip(a, b):
                    assert_helper(l, r)
            elif isinstance(a, int):
                if not isinstance(b, int) or a != b:
                    raise AssertionError(f"Int mismatch: {a} != {b}")
            elif a is None:
                if b is not None:
                    raise AssertionError(f"Expected None, got {b}")
            elif isinstance(a, py_sym_types):
                if type(a) is not type(b) or a.node is not b.node:
                    raise AssertionError(f"SymType mismatch: {a} != {b}")
            elif isinstance(a, torch.Tensor):
                if not isinstance(b, torch.Tensor):
                    raise AssertionError(f"Expected Tensor, got {type(b)}")
                assert_metadata_eq(assert_eq, a, b)
            else:
                raise RuntimeError(f"Unsupported type {type(a)}")

        try:
            true_output = self._dispatch_impl(func, types, args, kwargs)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}: Dispatch raised={e}"
            ) from e
        try:
            assert_helper(true_output, output)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}"
            ) from e

    def dispatch(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        # [FUNCTION DOC] Mode entry for an op: logs the call, serves metadata-handler ops directly, runs a few
        # attribute-query ops inside a kernel invocation, and otherwise dispatches through the cached path when
        # caching is enabled or straight to the uncached implementation. A recursion counter object is created
        # only for debug logging.
        kwargs = kwargs or {}
        with no_dispatch():
            log.debug("%s %s %s", func, args, kwargs)

        if func in _DISPATCH_META_HANDLERS:
            return _DISPATCH_META_HANDLERS[func](args)

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "%sFakeTensorMode.__torch_dispatch__: %s", " " * RECURSION_COUNT, func
            )
            # NOTE: incr is intentionally unused for a RAII pattern
            incr = IncrementRecursionCount()  # noqa: F841

        # Some attribute queries that can be serviced directly
        # See Note [is_coalesced is dispatched]
        if func in _DISPATCH_HANDLE_DIRECTLY:
            # NB: no_dispatch is ok here too, this func is very simple
            with in_kernel_invocation_manager(self):
                return func(*args, **kwargs)

        if self.cache_enabled:
            return self._cached_dispatch_impl(func, types, args, kwargs)
        else:
            return self._dispatch_impl(func, types, args, kwargs)

    def _maybe_infer_fake(
        self, func: OpOverload, path: KeyPath, fake: object, real: object
    ) -> tuple[object | None, bool]:
        # [FUNCTION DOC] HELPER: Cross-checks one fake output leaf against the real one when real tensor
        # propagation is on. Tensors: compares metadata, then each size (symbols resolved with the shape env);
        # None vs real value; plain values by equality. On mismatch raises MetadataMismatchError, unless the
        # config flag allows regenerating the fake from the real output, in which case a new fake and True are
        # returned (a structured trace is emitted). Returns (fake_or_replacement, was_overwritten).
        """
        Helper to cross-check fake/real output properties & values,
        and create new fake vals if mismatched.
        Returns tuple of object & boolean, for whether or not it was overwrriten
        """
        import sympy

        from torch._subclasses.fake_utils import _check_fake_real_tensors

        def _check_fake_real_vals(fake: Any, real: Any) -> None:
            # [FUNCTION DOC] HELPER: Compares a fake scalar (symbolic or concrete) to a real value; symbolic
            # values that are fully determined by known backed/unbacked values are evaluated statically. Raises
            # MetadataMismatchError on disagreement.
            # use real values + ShapeEnv to check mismatches between potentially symbolic values
            if isinstance(fake, (SymInt, SymFloat)):
                # symbolic expression, ask ShapeEnv to substitute known backed/unbacked values
                if self.shape_env is None:
                    raise AssertionError(
                        "self.shape_env must not be None for symbolic values"
                    )
                if (
                    not fake.node.expr.free_symbols
                    - self.shape_env.backed_var_to_val.keys()
                    - self.shape_env.real_tensor_prop_unbacked_vals.keys()
                ):
                    if (
                        self.shape_env._maybe_evaluate_static(
                            sympy.Eq(fake.node.expr, real), compute_hint=True
                        )
                        is not sympy.S.true
                    ):
                        raise MetadataMismatchError(
                            f"mismatch between fake value {fake} and real value {real} "
                        )
            elif isinstance(
                fake, (int, float, bool)
            ):  # concrete value, check direct equality
                if fake != real:
                    raise MetadataMismatchError(
                        f"mismatch between fake value {fake} and real value {real} "
                    )

        if isinstance(fake, torch.Tensor):
            try:
                _check_fake_real_tensors(
                    real,  # type: ignore[arg-type]
                    fake,  # type: ignore[arg-type]
                    context="Real tensor propagation found",
                    sizes=False,  # manual check below
                    strides=False,  # skip strides
                    storage_offset=True,
                    requires_grad=False,  # issues with FakeTensorConverter preserving requires_grad
                )
            except MetadataMismatchError as exc:
                if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                    dtrace_structured(
                        "mismatched_fake_kernel",
                        metadata_fn=lambda: {
                            "op": str(func),
                            "reason": exc.reason,  # noqa: F821
                        },
                    )
                    return _infer_fake_from_real_tensor(self, func, real), True  # type: ignore[arg-type]
                raise MetadataMismatchError(
                    f"Real tensor propagation found a metadata mismatch between "
                    f"fake tensor {fake} and real tensor {real}, "
                    f" at output{keystr(path)}, for func: {func}"
                ) from exc

            for j, (s_fake, s_real) in enumerate(zip(fake.size(), real.size())):  # type: ignore[attr-defined]
                try:
                    _check_fake_real_vals(s_fake, s_real)
                except MetadataMismatchError as exc:
                    if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                        dtrace_structured(
                            "mismatched_fake_kernel",
                            metadata_fn=lambda: {
                                "op": str(func),
                                "reason": exc.reason,  # noqa: F821
                            },
                        )
                        return _infer_fake_from_real_tensor(self, func, real), True  # type: ignore[arg-type]
                    raise MetadataMismatchError(
                        f"Real tensor propagation found an output size mismatch between "
                        f"fake shape {s_fake} and real shape {s_real}, "
                        f"at output{keystr(path)}.size({j}), for func: {func}"
                    ) from exc
        elif fake is None and real is not None:
            if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                dtrace_structured(
                    "mismatched_fake_kernel",
                    metadata_fn=lambda: {
                        "op": str(func),
                        "reason": f"mismatch between fake value {fake} and real value {real}",
                    },
                )
                return _infer_fake_from_real_tensor(self, func, real), True  # type: ignore[arg-type]
            raise MetadataMismatchError(
                f"Real tensor propagation found a metadata mismatch between "
                f"fake tensor {fake} and real tensor {real}, "
                f" at output{keystr(path)}, for func: {func}"
            )
        else:
            try:
                _check_fake_real_vals(fake, real)
            except MetadataMismatchError as exc:
                raise MetadataMismatchError(
                    f"Real tensor propagation found an output value mismatch between "
                    f"fake output value {fake} and real output value {real}, "
                    f"at output{keystr(path)}, for func: {func}"
                ) from exc
        return fake, False

    def _maybe_infer_fake_kernel_from_pytree_out(
        self,
        func: OpOverload,
        fake_in: object,
        real_in: object,
        fake_out: object,
        real_out: object,
    ) -> object | None:
        # [FUNCTION DOC] HELPER: Kernel-level fake vs real comparison for a whole pytree output: checks
        # aliasing against inputs, then each leaf via _maybe_infer_fake. On an aliasing mismatch, either raises
        # MetadataMismatchError or (config flag) returns non-aliasing fakes inferred from the real output.
        # Clears newly added pending unbacked symbols when a kernel was overridden. Returns the (possibly
        # replaced) fake pytree.
        """
        Helper to cross-check fake/real output properties & values,
        and create new fake vals if mismatched, but at the kernel level.
        Means this handles pytree outputs & checks aliasing.
        """
        from torch._subclasses.fake_utils import _check_alias_info

        # we might have to clear pending unbacked symbols, if we override the kernel
        pending_unbacked = None
        if self.shape_env:
            pending_unbacked = list(self.shape_env.pending_fresh_unbacked_symbols)

        def _clear_pending_unbacked() -> None:
            # [FUNCTION DOC] HELPER: Removes from the shape env's pending fresh unbacked symbols those that were
            # pending before this call started.
            self.shape_env.pending_fresh_unbacked_symbols = list(  # type: ignore[union-attr]
                set(self.shape_env.pending_fresh_unbacked_symbols).difference(  # type: ignore[union-attr]
                    pending_unbacked  # type: ignore[arg-type]
                )
            )

        fake_paths_leaves, fake_spec = pytree.tree_flatten_with_path(fake_out)
        real_leaves, _ = pytree.tree_flatten(real_out)
        try:
            # catch aliasing mismatches between fake/real tensors
            _check_alias_info(
                "Real tensor propagation found", real_out, real_in, fake_out, fake_in
            )
        except MetadataMismatchError as exc:
            # if mismatch found, optionally infer fake kernel
            if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                dtrace_structured(
                    "mismatched_fake_kernel",
                    metadata_fn=lambda: {
                        "op": str(func),
                        "reason": (
                            f"Mismatched aliasing spec between fake kernel and real kernel: {exc.reason}"  # noqa: F821
                        ),
                    },
                )
                # if aliasing mismatches are found, it's likely that the fake tensor impl
                # is incorrectly aliasing, since we don't support aliasing custom ops.
                # in this case we can default to inferring non-aliasing fake kernels from the real outputs.
                _clear_pending_unbacked()
                return tree_map(
                    lambda x: _infer_fake_from_real_tensor(self, func, x), real_out
                )
            else:
                raise MetadataMismatchError(
                    f"Real tensor propagation found an aliasing mismatch between "
                    f"fake output {fake_out} and real output {real_out}, "
                    f" for func: {func}"
                ) from exc

        # if no errors raised, run cross checks on fake/real tensors,
        # optionally overriding individual fake tensors, if individual meta kernel output is incorrect.
        fake_leaves, overrides = zip(
            *[
                self._maybe_infer_fake(func, _fake_path, _fake_out, _real_out)
                for (_fake_path, _fake_out), _real_out in zip(
                    fake_paths_leaves, real_leaves
                )
            ]
        )
        if (
            any(overrides) and pending_unbacked
        ):  # only keep new pending unbacked symbols
            _clear_pending_unbacked()
        return pytree.tree_unflatten(fake_leaves, fake_spec)

    def _dispatch_impl(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> FakeTensor | None:
        # [FUNCTION DOC] Core dispatch. Returns NotImplemented for unrecognized tensor subclasses. Then, in
        # order: constant-propagates lift ops and scalar-accepting ops with only constant inputs; converts
        # real tensors per policy; constant-folds ops whose fake inputs are all constants (invalidating
        # aliases when results cannot stay constants); runs registered higher-order fake kernels; invalidates
        # constants written to; optionally runs the real op for real-tensor propagation and cross-checks
        # fake output; then tries fast op impls, mkldnn to_dense, python meta/decomposition, prim meta
        # impls, custom op profiles, and later registered fake impls, fallback kernels and meta kernels (rest of
        # the function). Raises data-dependent, dynamic-shape or unsupported operator exceptions as appropriate.
        from torch._higher_order_ops.utils import registered_hop_fake_fns

        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # DO NOT PUT LOGIC BEFORE UNRECOGNIZED TYPE CHECKING
        # We must throw NotImplemented in case of unrecognized types to handle subclasses.
        # Throwing the exception will pass the control to the next __torch_dispatch__.
        # See [subclass inputs] below
        # NB: If you're seeing a mysterious infinite loop involving fake
        # tensor, it might be related to this line.  Though I'm not sure
        # how you'll know to read this comment, as this line won't show up
        # in the stack trace.
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "FakeTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        flat_arg_fake_tensors = [t for t in flat_args if self.is_our_fake(t)]
        has_symbolic_sizes = any(
            i._has_symbolic_sizes_strides for i in flat_arg_fake_tensors
        ) or any(isinstance(a, SymInt) for a in flat_args)

        converter = self.fake_tensor_converter

        is_lift_func = func in self.lift_fns

        # If we are trying to avoid device init, then we need to avoid constant
        # prop on constant tensors for ops that change devices.
        avoiding_device_init = False
        if self.avoid_device_init:
            if (
                func is torch.ops.aten._to_copy.default
                and "device" in kwargs
                and kwargs["device"].type != "cpu"  # type: ignore[attr-defined]
            ):
                avoiding_device_init = True
            if func is torch.ops.prims.device_put.default:
                avoiding_device_init = True

        # skip const prop for aten._to_copy if
        # 1. input tensor is on "meta" device
        # 2. destination device is unavailable, captured by `avoiding_device_init`
        device_conversion_skip_const_prop = (
            func is torch.ops.aten._to_copy.default
            and isinstance(args[0], torch.Tensor)
            and args[0].device.type == "meta"
        ) or avoiding_device_init

        # To constant propagate through these functions:
        # 1, If this is a lift due to a torch.tensor call,
        #    the input tensor is guaranteed to be a
        #    constant, so we keep a copy of the original argument along so
        #    we can query it if we're asked to item() it at some later point.
        #    (Note that you can always call a lift fn manually, so we do
        #    have to check if there are any fake tensors!)
        # 2, Some functions that allow Python numbers to bind to Tensors, e.g, torch.div
        if (is_lift_func and not flat_arg_fake_tensors) or (
            should_allow_numbers_as_tensors(func)
            and not has_symbolic_sizes
            and not flat_arg_fake_tensors
            and not device_conversion_skip_const_prop
        ):
            if not all(t.constant is not None for t in flat_arg_fake_tensors):
                raise AssertionError(
                    f"{func} should not have fake inputs without constants"
                )
            const_flat_args = [
                a.constant if self.is_our_fake(a) else a for a in flat_args
            ]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)
            out = func(*const_args, **const_kwargs)
            if type(out) is Tensor and self.may_turn_const(out):
                # NB: not in_kernel_invocation_manager because we're doing real
                # compute here
                # NB: no_dispatch() here is VERY DANGEROUS (like, segfault
                # dangerous) if this is actually a wrapper subclass tensor,
                # therefore the exact type test above
                with no_dispatch():
                    out = out.clone()
                return converter.from_real_tensor(self, out, make_constant=True)

        # if we are in the dispatch mode, we will enter this function even if the inputs
        # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
        # and just support constructors.

        # this is generated from torch.tensor(), which does not use the
        # dispatcher, to allow wrapper subclasses to wrap the new tensor
        if is_lift_func:
            if len(kwargs) != 0 or len(args) != 1:
                raise AssertionError(
                    f"Expected exactly one arg for lift func, got args={args} kwargs={kwargs}"
                )

            if type(args[0]) is Tensor:
                return converter.from_real_tensor(self, args[0])

        # Recompute flat_arg_fake_tensors here again in case some of the inputs
        # were real tensors and fakified in validate_and_convert_non_fake_tensors
        (flat_args, flat_arg_fake_tensors) = self.validate_and_convert_non_fake_tensors(
            func, converter, flat_args, args_spec
        )
        del args, kwargs  # Invalidated

        # The current constant handling only support tracing systems
        # (aot autograd, torchdynamo) where each operation is run consecutively.
        # Because each operation is run in order, we can trace out and support
        # sequences like: x = torch.tensor(0.); y = x.add_(1)
        # Whenever a constant is written to but with inputs that cannot be evaluated
        # statically, such as random_(), we invalidate all constants that alias the input
        # We will rely on functionalization for use of fake tensors constants as persistent
        # objects on an FX Graph.

        all_constant = all(e.constant is not None for e in flat_arg_fake_tensors)
        if (
            isinstance(func, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded not in func.tags
            # We dispatch size/stride/numel on the FakeTensor not its constant, so bail on inplace_view.
            # Example: fake_a.transpose_(0,1) would mutate fake_a.constant in-place, changing its
            # shape from (2,3) to (3,2), while fake_a.shape still reports (2,3) → divergence.
            # However, detach_ is safe: it only mutates requires_grad (not shape/stride/data),
            # and constants are used purely for their values, not autograd.
            and (
                torch.Tag.inplace_view not in func.tags or func is aten.detach_.default
            )
            and all_constant
            and len(flat_arg_fake_tensors) != 0
            and not has_symbolic_sizes
            and not avoiding_device_init
            and func is not aten._nested_tensor_from_tensor_list.default
        ):
            const_flat_args = [
                a.constant if self.is_our_fake(a) else a for a in flat_args
            ]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)

            # NB: not in_kernel_invocation_manager(self) as we want to do REAL
            # compute
            with no_dispatch():
                out = func(*const_args, **const_kwargs)

            flat_out = pytree.tree_leaves(out)
            flat_out_tensors = [t for t in flat_out if isinstance(t, Tensor)]
            all_constant = all(self.may_turn_const(t) for t in flat_out_tensors)

            if all_constant:
                return pytree.tree_map_only(
                    Tensor,
                    lambda t: converter.from_real_tensor(self, t, make_constant=True),
                    out,
                )

            # we weren't able to turn outputs to constants,
            # so invalidate all constants that might be aliases of the outputs
            for ten in flat_out_tensors:
                converter.invalidate_constant_aliases(ten)

        # we are falling through to running non constant tensors, any input constant that
        # is written to must be invalidated
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)

        if (
            isinstance(func, torch._ops.HigherOrderOperator)
            and func in registered_hop_fake_fns
        ):
            # Reenable the fake tensor mode for the registered fake function
            maybe_ignore_fresh_unbacked_symbols = (
                contextlib.nullcontext
                if self.shape_env is None
                else self.shape_env.ignore_fresh_unbacked_symbols
            )

            with self, maybe_ignore_fresh_unbacked_symbols():
                return registered_hop_fake_fns[func](*args, **kwargs)

        self.invalidate_written_to_constants(func, flat_arg_fake_tensors, args, kwargs)

        def maybe_to_real_tensor(
            t: T,
        ) -> T | Tensor | torch._C.ScriptObject | None:
            # [FUNCTION DOC] HELPER: Maps a fake argument to its real counterpart for real tensor propagation:
            # FakeTensor to its real tensor, symbolic scalars to concrete values via the shape env, fake script
            # object to its real object; everything else unchanged.
            if isinstance(t, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                return t.real_tensor
            elif isinstance(t, py_sym_types):
                if self.shape_env is None:
                    raise AssertionError(
                        "self.shape_env must not be None for symbolic types"
                    )
                return t.node.pytype(
                    t.node.expr.xreplace(self.shape_env.backed_var_to_val).xreplace(
                        self.shape_env.real_tensor_prop_unbacked_vals
                    )
                )
            elif isinstance(t, FakeScriptObject):
                return t.real_obj
            else:
                return t

        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            free_unbacked_symbols,
        )

        nil = object()

        real_out = nil
        if (
            self.propagate_real_tensors
            and all(e.real_tensor is not None for e in flat_arg_fake_tensors)
            and not any(
                (
                    isinstance(a, py_sym_types)
                    and (syms := free_unbacked_symbols(a))
                    and self.shape_env is not None
                    and any(
                        s not in self.shape_env.real_tensor_prop_unbacked_vals
                        for s in syms
                    )
                )
                for a in flat_args
            )
        ):
            log.debug("propagate_real_tensors %s", func)
            real_flat_args = [maybe_to_real_tensor(a) for a in flat_args]
            real_args, real_kwargs = pytree.tree_unflatten(real_flat_args, args_spec)

            is_builtin = library_utils.is_builtin(func)
            if not is_builtin:
                mutation_checker = library_utils.MutationChecker(
                    func, real_flat_args, args_spec
                )

            try:
                real_out = func(*real_args, **real_kwargs)
            except ZeroDivisionError as exc:
                # we shouldn't broadly catch all errors here;
                # some come from real-kernel mutation/aliasing checks we want to run.
                # add more exception types as needed.
                log.debug(
                    "real-tensor fallback failed for %s: %s; silently ignoring",
                    func,
                    exc,
                )

            if not is_builtin:
                mutation_checker.check()  # type: ignore[possibly-undefined]
                library_utils.check_aliasing_constraint(func._name, flat_args, real_out)

        elif self.propagate_real_tensors:
            # This can happen occasionally legitimately, specifically when you
            # are inside the meta of a data dependent operation and you create
            # a tensor on an unbacked SymInt; at this point in time we don't
            # know what the unbacked SymInt is, but we will know later.
            # However, if there's a bug in the condition above, this condition
            # will also trigger.
            log.debug(
                "SKIPPED propagate_real_tensors %s(%s, %s) %s",
                func,
                flat_arg_fake_tensors,
                flat_args,
                self.shape_env.real_tensor_prop_unbacked_vals
                if self.shape_env
                else None,
            )

        def maybe_propagate_real_tensors(fake_out: T) -> T:
            # [FUNCTION DOC] HELPER: When real tensor propagation computed a real output, cross-checks it with the
            # fake output (optionally replacing the fake output), records real tensors on fake outputs and unbacked
            # symbol values in the shape env, and peeks unbacked bindings. Returns the fake output.
            import sympy

            log.debug("maybe_propagate_real_tensors %s", func)

            def go(t: object, real_t: Tensor) -> None:
                # [FUNCTION DOC] HELPER: Records the real counterpart on a fake tensor (overwriting) and recursively
                # on its sizes, strides and storage offset; for unbacked symbols stores the real value in the shape env.
                if isinstance(t, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                    # NB: unconditionally overwrite
                    log.debug(
                        "maybe_propagate_real_tensors %s -> %s", id(t), id(real_t)
                    )
                    t.real_tensor = real_t
                    for s, real_s in zip(t.size(), real_t.size()):
                        go(s, real_s)  # type: ignore[arg-type]
                    for s, real_s in zip(t.stride(), real_t.stride()):
                        go(s, real_s)  # type: ignore[arg-type]
                    go(t.storage_offset(), real_t.storage_offset())  # type: ignore[arg-type]
                elif isinstance(t, py_sym_types) and free_unbacked_symbols(t):
                    if isinstance(t.node.expr, sympy.Symbol):
                        if self.shape_env is None:
                            raise AssertionError(
                                "self.shape_env must not be None for symbolic Symbol"
                            )
                        self.shape_env.set_real_tensor_prop_unbacked_vals(
                            t.node.expr, real_t
                        )
                    elif (
                        isinstance(s := t.node.expr, sympy.Eq)
                        and isinstance(s.lhs, sympy.Symbol)
                        and s.rhs == 1
                    ):
                        if self.shape_env is None:
                            raise AssertionError(
                                "self.shape_env must not be None for symbolic Eq"
                            )

                        self.shape_env.set_real_tensor_prop_unbacked_vals(s, real_t)

            if real_out is not nil:
                # cross check fake/real outputs, and optionally override fake kernel mismatches
                if not torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                    self._maybe_infer_fake_kernel_from_pytree_out(
                        func,
                        (args, kwargs),
                        (real_args, real_kwargs),
                        fake_out,
                        real_out,
                    )
                else:
                    # this can override the output only when the flag is True
                    fake_out = self._maybe_infer_fake_kernel_from_pytree_out(  # type: ignore[assignment]
                        func,
                        (args, kwargs),
                        (real_args, real_kwargs),
                        fake_out,
                        real_out,
                    )

                # populate real_tensor_prop_unbacked_vals
                if (
                    not isinstance(fake_out, Tensor)
                    and not isinstance(real_out, Tensor)
                    and type(fake_out) is not type(real_out)
                ):
                    # This can happen when decompositions have different return types,
                    # e.g. namedtuple vs. tuple vs. list.
                    tree_map_(
                        go,
                        tuple(pytree.tree_flatten(fake_out)),
                        tuple(pytree.tree_flatten(real_out)),
                    )
                else:
                    tree_map_(go, fake_out, real_out)

                # If a data-dependent op is used in a decomposition, we
                # may need to get the unbacked settings "early"
                # TODO: Is this really needed?
                compute_unbacked_bindings(self.shape_env, fake_out, peek=True)

            return fake_out

        # Try for fastpath
        if has_symbolic_sizes:
            fast_impl = get_fast_op_impls().get(func)
            if fast_impl is not None:
                return maybe_propagate_real_tensors(fast_impl(self, *args, **kwargs))

        if func is torch.ops.aten.to_dense.default:
            # The registered fake op impl handles the usual path, but symbolic
            # shapes can still reach generic decomposition below. The native
            # composite sees a fake MKLDNN tensor's strided meta backing, so
            # handle this before generic decomposition.
            dtype = args[1] if len(args) > 1 else kwargs.get("dtype")
            masked_grad = kwargs.get("masked_grad")
            op_impl_out = maybe_to_dense_mkldnn(
                self, args[0], dtype=dtype, masked_grad=masked_grad
            )
            if op_impl_out is not NotImplemented:
                return maybe_propagate_real_tensors(cast(FakeTensor, op_impl_out))

        # If there's a Python meta, prefer that over the decomposition
        from torch._decomp import meta_table

        if (
            func not in meta_table
            and not self.cpp_meta_supports_symint(func)
            and not (
                has_symbolic_sizes and func in self._unbacked_special_fake_handling_ops
            )
        ):
            from torch._decomp import decomposition_table

            # Prefer Python decompositions over C++ ones
            if func in decomposition_table and (
                has_symbolic_sizes
                or (
                    # TODO: Remove these exclusions, so that we can remove
                    # this leg entirely
                    torch_decomp_decompositions(func)
                    and all(not is_sparse_any(e) for e in flat_arg_fake_tensors)
                )
            ):
                with self:
                    return maybe_propagate_real_tensors(
                        decomposition_table[func](*args, **kwargs)
                    )

            with self:
                # Decomposes CompositeImplicitAutograd ops
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return maybe_propagate_real_tensors(r)

        # prims already wrap FakeTensor inputs to FakeTensor outputs
        # and do device logic, we don't need do anything but run them
        # and ensure that Meta kernels are dispatched to (see)
        # Fake Tensor Dispatch Keys
        # TODO - we should be use the prim aten impl
        # TODO - fix prims complex ops
        if (
            "prims::" in func._schema.name
            and hasattr(func, "prim_meta_impl")
            and not stride_incorrect_op(func)
        ):
            with self:
                return maybe_propagate_real_tensors(
                    func.prim_meta_impl(*args, **kwargs)
                )

        profiles = torch._dynamo.config._custom_ops_profile
        if profiles is not None:
            if func in profiles.data:
                return profiles.generic_fake_kernel(func, self, *args, **kwargs)

        if (
            self.propagate_real_tensors
            and real_out is not nil
            and not library_utils.is_builtin(func)
            and self.shape_env is not None
        ):
            # Automatically infer a Fake kernel if there isn't one.
            if not library_utils.has_fake_kernel(func):
                result = inferred_fake_kernel_from_real_out(self, func, real_out)

                dtrace_structured(
                    "missing_fake_kernel",
                    metadata_fn=lambda: {
                        "op": str(func),
                    },
                )
                return maybe_propagate_real_tensors(result)

        # Users can register FakeTensor rules for custom operators
        # Call them if they exist.
        maybe_fake_impl = torch._library.simple_registry.singleton.find(
            func.name()
        ).fake_impl.kernel
        if maybe_fake_impl:
            try:
                ctx = torch._library.fake_impl.FakeImplCtx(self, func)
                with torch._library.fake_impl.set_ctx_getter(lambda: ctx), self:
                    result = maybe_fake_impl(*args, **kwargs)
                    return maybe_propagate_real_tensors(result)

            except MissingOpProfile as e:
                # If we have a fake kernel registered generated from OpProfiles
                # but there doesn't exist a profile for the existing inputs, and we are in
                if (
                    self.propagate_real_tensors
                    and real_out is not nil
                    and not library_utils.is_builtin(func)
                    and self.shape_env is not None
                ):
                    result = inferred_fake_kernel_from_real_out(self, func, real_out)

                    dtrace_structured(
                        "missing_fake_kernel",
                        metadata_fn=lambda: {
                            "op": str(func),
                        },
                    )
                    return maybe_propagate_real_tensors(result)
                else:
                    raise e

        # special handling for funcs registered through `register_op_impl`,
        # e.g., manipulating args on constructor calls to construct meta tensors
        # and then afterwards wrapping them to a FakeTensor
        for run_impl_check, op_impl in op_implementations_checks:
            if run_impl_check(func):
                # pyrefly: ignore [bad-argument-count]
                op_impl_out = op_impl(self, func, *args, **kwargs)
                if op_impl_out is not NotImplemented:
                    # pyrefly: ignore [bad-return]
                    return maybe_propagate_real_tensors(op_impl_out)

        def maybe_run_unsafe_fallback(
            error: RuntimeError | None = None,
        ) -> FakeTensor | None:
            # [FUNCTION DOC] HELPER: Fallback when no usable meta kernel exists: generates a trivial fake result
            # for custom ops returning None or out= ops; otherwise raises UnsupportedOperatorException if symbolic
            # sizes are involved or fallback is not allowed for the op; else runs the real kernel on zero
            # tensors via run_fallback_kernel.
            # We infer the meta of custom ops that return None to just
            # return None, and Tag.out ops to return their out= args.
            # Custom ops are not allowed to mutate metadata of their
            # inputs, so this is safe.
            if torch._library.utils.can_generate_trivial_fake_impl(func):
                return torch._library.utils.generate_trivial_fake_impl(
                    func, *args, **kwargs
                )
            # no meta kernel registered, fallback to kernel for the device
            if has_symbolic_sizes or not self.can_run_unsafe_fallback(func):
                raise UnsupportedOperatorException(func)
            if error is None:
                error = UnsupportedOperatorException(func)
            return run_fallback_kernel(self, func, flat_args, args_spec, error)

        # Optimization: If there is no Meta kernel, it takes a surprisingly long
        # amount of time to catch the NotImplementedError, so we check it here.
        if not has_meta(func):
            fallback = maybe_run_unsafe_fallback()
            return maybe_propagate_real_tensors(fallback)

        # run kernel registered to meta for func, which include
        # python meta registrations, prims, decomps, and c++ meta fns (structured kernels)
        # It's possible that the kernel will return NotImplementedError
        try:
            with in_kernel_invocation_manager(self):
                r = func(*args, **kwargs)
        except NotImplementedError as not_implemented_error:
            return maybe_run_unsafe_fallback(not_implemented_error)
        except Exception:
            log.exception("failed while attempting to run meta for %s", func)
            raise

        return maybe_propagate_real_tensors(
            self.wrap_meta_outputs_with_default_device_logic(
                r, func, flat_args, device=kwargs.get("device")
            )
        )

    # WARNING: DO NOT add any additional namespaces/operators here if they refer to operators
    # outside of the pytorch/pytorch library! Any pre-existing things here
    # are either in the pytorch/pytorch library or have been grandfathered in.
    # The fallback does not always work and MAY CRASH and emit unreadable error messages
    # so it should not be allowed by default.
    _can_run_unsafe_fallback_allowed_namespaces = ordered_set(
        "debugprims",
        "prims",
        "aten",
        "xla",
        "vision",
        "torchtext",
        "torchaudio",
        "quantized",
    )

    def can_run_unsafe_fallback(self, func: OpOverload) -> bool:
        # [FUNCTION DOC] True if running the real fallback kernel is allowed: fallback kernels enabled and the
        # op is in an allow-listed namespace (debugprims, prims, aten, xla, vision, torchtext, torchaudio,
        # quantized) or is the fbgemm gmm op.
        if not self.allow_fallback_kernels:
            return False
        # It's OK to try the fallback for built-in ops (e.g. aten, prims)
        # because we control and test these but the fallback leads to unexpected behavior
        # in user-defined custom ops
        return (
            func.namespace in self._can_run_unsafe_fallback_allowed_namespaces
            or func.name() == "fbgemm::gmm"
        )

    def validate_and_convert_non_fake_tensors(
        self,
        func: OpOverload,
        converter: FakeTensorConverter,
        flat_args: Sequence[object],
        args_spec: TreeSpec,
    ) -> tuple[list[object], list[FakeTensor]]:
        # [FUNCTION DOC] Validates every tensor in the flat args: tensors of this mode pass; others are converted
        # to fake only when non-fake inputs are allowed (mode setting or thread override), otherwise an
        # AssertionError is raised (also for metadata-mutating ops on non-fake inputs, or for fake tensors of a
        # different mode). Returns the validated flat args and the list of fake tensors among them.
        """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
        flat_arg_fake_tensors: list[FakeTensor] = []

        def validate(x: T) -> T | FakeTensor:
            # [FUNCTION DOC] HELPER: Validates or converts one argument (non-tensors pass through) and records the
            # resulting fake tensor in the enclosing list.
            if not isinstance(x, Tensor):
                return x

            nonlocal flat_arg_fake_tensors
            if not self.is_our_fake(x):
                if hasattr(func, "tags") and torch.Tag.inplace_view in func.tags:
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}"
                    )
                allow_non_fake_inputs = (
                    self.allow_non_fake_inputs
                    if fake_tensor_tls.allow_non_fake_inputs_override is None
                    else fake_tensor_tls.allow_non_fake_inputs_override
                )
                if not allow_non_fake_inputs:
                    if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                        # pyrefly: ignore [missing-attribute]
                        x_fake_mode = x.fake_mode
                        if x_fake_mode is not self:
                            raise AssertionError(
                                f"Mixing fake modes NYI x.fake_mode={x_fake_mode} vs self={self}"
                            )
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode "
                        f"with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}"
                    )

                out = converter.from_real_tensor(self, x)
            else:
                out = x

            flat_arg_fake_tensors.append(out)
            return out

        validated_args = [validate(a) for a in flat_args]
        return validated_args, flat_arg_fake_tensors

    def wrap_meta_outputs_with_default_device_logic(
        self,
        r: object,
        func: OpOverload,
        flat_args: Sequence[object],
        device: torch.device,
    ) -> PyTree:
        # [FUNCTION DOC] Wraps tensors returned by a meta kernel into FakeTensors on the common input device
        # (or the explicit device argument): own fakes are checked for device agreement, scalar-only-input ops
        # get their real CPU result fakified, others convert from meta plus device. Alias and detach keep the
        # input's dispatch keys. Non-tensors pass through; returns the same pytree structure.
        converter = self.fake_tensor_converter

        # Lazily initialized, in case there are no tensor returns
        common_device = None
        has_scalar_only_inputs = False
        preserve_dispatch_keys = False
        input_dispatch_keys = None
        if (
            (func is aten.alias.default or func is aten.detach.default)
            and len(flat_args) == 1
            and isinstance(flat_args[0], FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
        ):
            input_dispatch_keys = flat_args[0].dispatch_keys
            preserve_dispatch_keys = input_dispatch_keys is not None

        def wrap(e: T) -> T | FakeTensor:
            # [FUNCTION DOC] HELPER: Wraps one output leaf as described by the enclosing method, computing the
            # common device lazily on first tensor.
            nonlocal common_device
            nonlocal has_scalar_only_inputs

            if not isinstance(e, Tensor):
                return e

            if common_device is None:
                (
                    common_device,
                    has_scalar_only_inputs,
                ) = FakeTensor._find_common_device(func, flat_args)

            is_our_fake = self.is_our_fake(e)
            if is_our_fake:
                torch._check(
                    e.device == common_device,
                    lambda: f"FakeTensor is wrapped to wrong device, found {e.device}, expected {common_device}",
                )
                if preserve_dispatch_keys:
                    e.dispatch_keys = input_dispatch_keys
                return cast(T, e)
            elif converter is not None:
                if has_scalar_only_inputs:
                    # Under FakeTensorMode, op accepts scalar only inputs, such as aten.add/sub/mul/div,
                    # returns a real scalar tensor on CPU. See TensorMeta() in _prims/__init__.py for details.
                    # We thus directly convert real tensor to fake tensor.
                    out = converter.from_real_tensor(self, e)
                else:
                    out = converter.from_meta_and_device(
                        self, e, device or common_device
                    )
                if preserve_dispatch_keys:
                    out.dispatch_keys = input_dispatch_keys
                return out
            else:
                # pyrefly: ignore [bad-return]
                return e

        return tree_map(wrap, r)

    def create_symbolic_nested_int(
        self, *, nt_tensor_id: int | None = None
    ) -> IntLikeType:
        # [FUNCTION DOC] Creates a symbolic nested int (coefficient 1) with a ShapeEnv symbol. When no id is
        # given, takes the next nested tensor id from the mode counter (mode must be active) and advances it.
        # Raises an assertion error if there is no shape env.
        # See Note: [Creating symbolic nested int]
        # Returned nested int always has coeff=1; multiply the result by coeff if needed
        import torch.nested._internal.nested_tensor
        from torch.nested._internal.nested_int import NestedIntNode

        if nt_tensor_id is None:
            nt_tensor_id = self.nt_tensor_id_counter
            if not self.enter_stack:
                raise AssertionError(
                    "should only be called while FakeTensorMode is active"
                )
            self.nt_tensor_id_counter += 1
        hint = torch.SymInt(NestedIntNode(nt_tensor_id, 1))

        src = torch._dynamo.source.EphemeralSource("intermediate_offsets_or_lengths")
        if self.shape_env is None:
            raise AssertionError("self.shape_env must not be None")
        ret = self.shape_env.create_symintnode(
            sym=self.shape_env.create_symbol(
                val=hint,
                source=src,
            ),
            hint=hint,
            source=src,
        )
        return ret

    _cpp_meta_supports_symint = ordered_set(
        aten.empty.memory_format,
        aten.empty_strided.default,
        aten.as_strided_scatter.default,
        aten.as_strided.default,
        aten.as_strided_.default,
        aten.zeros.default,
        aten.detach.default,
        aten.view_as_real.default,
        aten.view_as_complex.default,
        aten.set_.source_Storage_storage_offset,
        aten._sparse_coo_tensor_with_dims_and_tensors.default,
        aten.stack.default,
    )

    _unbacked_special_fake_handling_ops = ordered_set(
        aten.view.default,
        aten._unsafe_view.default,
        aten.slice.Tensor,
    )

    def cpp_meta_supports_symint(self, func: OpOverload) -> bool:
        # [FUNCTION DOC] True if the op's C++ meta kernel handles symbolic ints: view_copy ops and an allow-list of ops.
        if torch.Tag.view_copy in func.tags:
            return True
        return func in self._cpp_meta_supports_symint

    lift_fns = ordered_set(aten.lift_fresh.default, aten.lift_fresh_copy.default)

    def may_turn_const(self, t: Tensor) -> bool:
        # [FUNCTION DOC] True if a real tensor may become a tracked constant: at most the constant numel limit
        # elements, not sparse, not already fake, not on meta.
        return (
            t.numel() <= CONSTANT_NUMEL_LIMIT
            and not is_sparse_any(t)
            and not self.is_our_fake(t)
            and t.device.type != "meta"
        )

    def invalidate_written_to_constants(
        self,
        func: OpOverload,
        flat_arg_fake_tensors: Sequence[FakeTensor],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        # [FUNCTION DOC] If the op mutates any argument that is a constant fake tensor, invalidates all
        # constant aliases of that argument's constant in the converter.
        any_constant = any(e.constant is not None for e in flat_arg_fake_tensors)
        schema_info = get_schema_info(func)
        if any_constant and schema_info.is_mutable():
            _, new_kwargs = normalize_function(  # type: ignore[misc]
                func,
                args=args,  # type: ignore[arg-type]
                kwargs=kwargs,  # type: ignore[arg-type]
                normalize_to_only_use_kwargs=True,
            )
            for k, v in new_kwargs.items():
                k = k if (k != "input" or schema_info.has_argument(k)) else "self"
                if (
                    self.is_our_fake(v)
                    and schema_info.is_mutable(k)
                    and v.constant is not None
                ):
                    self.fake_tensor_converter.invalidate_constant_aliases(v.constant)

    def from_tensor(
        self,
        tensor: Tensor,
        *,
        static_shapes: bool | None = None,
        source: Source | None = None,
        symbolic_context: SymbolicContext | None = None,
        trace: bool = True,
    ) -> FakeTensor:
        # [FUNCTION DOC] Converts a real tensor to a FakeTensor of this mode, returning an already-fake own
        # tensor unchanged when no overrides are given. Static shapes (argument or mode default) drop the
        # shape env and conflict with a symbolic context (AssertionError).
        if (
            isinstance(tensor, FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
            and tensor.fake_mode is self
            and static_shapes is None
            and source is None
            and symbolic_context is None
        ):
            return tensor

        shape_env: ShapeEnv | None = self.shape_env
        if static_shapes is None:
            static_shapes = self.static_shapes
        if static_shapes:
            if symbolic_context is not None:
                raise AssertionError(
                    "cannot set both static_shapes and symbolic_context"
                )
            shape_env = None
        return self.fake_tensor_converter.from_real_tensor(
            self,
            tensor,
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )


_StoragePointer = object


def _validate_symbolic_output_for_caching(
    state: _CacheKeyState, output: FakeTensor
) -> None:
    # [FUNCTION DOC] HELPER: Raises _BypassDispatchCache if the output has symbols unknown to the cache key
    # state, or when proxy tracing is active and the output has any symbolic nodes. Returns None otherwise.
    """
    Validate symbolic content in output and raise _BypassDispatchCache if
    caching should be bypassed.

    Args:
        state: Cache key state containing known symbols
        output: Output to validate
        proxy_mode_active: Whether PROXY dispatch mode is currently active

    Raises: _BypassDispatchCache: If output contains symbolic content that
        prevents caching

    Details:

    If our output contains any symbols that didn't appear in the input then we
    need to bypass. Usually this will be unbacked symbols which can't be
    properly reconstructed but there could be "weird" cases where backed symbols
    spontaneously appear (from non-input state)?

    If we're proxy (symbol) tracing and the output contains ANY symbols then we
    need to bypass. The problem is that ProxyTorchDispatchMode relies on SymNode
    object identity and being able to see the construction of SymNodes.

    We could improve the proxy tracing case in a few ways:

    1. If the output SymNodes are directly copied from inputs then this is
       actually fine - they're already tracked. This would probably be the
       biggest bang/buck.

    2. If the output (tensors) are all direct copies of the inputs then this is
       also fine - since they're inputs they must be tracked. We already compute
       this we just don't plumb it around enough.

    3. If the output SymNodes are already tracked by the proxy then this is also
       actually fine - they're properly tracked. This probably wouldn't be
       common since for most outputs we use torch.empty_strided() and recompute
       strides.

    4. We could use the proxy to track "how" the SymNodes were computed and when
       using the cache we could "replay" them properly to teach the proxy how to
       build them.
    """
    from torch.fx.experimental.symbolic_shapes import _iterate_exprs, _iterate_nodes

    is_tracing = torch.fx.experimental.proxy_tensor.get_proxy_mode() is not None
    if is_tracing:
        # Check for SymNode types in PROXY mode - this should bypass caching
        # regardless of whether symbols are known or not
        for _ in _iterate_nodes(output):
            raise _BypassDispatchCache("Proxy mode with SymNode output")
    else:
        # Check for unrepresented symbols in tensor expressions
        for s in _iterate_exprs(output):
            for symbol in s.free_symbols:
                if symbol not in state.known_symbols:
                    raise _BypassDispatchCache("unrepresented symbol in output")


# NB: returns fake tensors
def run_fallback_kernel(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    flat_args: Sequence[object],
    args_spec: PyTree,
    orig_not_implemented_exception: RuntimeError,
) -> FakeTensor:
    # [FUNCTION DOC] Runs the real kernel on zero-filled real tensors standing in for the fake inputs
    # (on their fake devices), then fakifies outputs, reusing the original fake inputs for
    # returned stand-ins. Re-raises the original not-implemented error for metadata-mutating ops or when an
    # output aliases input storage without being a stand-in. Used as a last-resort fallback.
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception

    # pyrefly: ignore [implicit-any]
    inp_impls = {}

    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e: T) -> T | Tensor:
            # [FUNCTION DOC] HELPER: Replaces a fake tensor with a zero real tensor on its fake device (keeping
            # sparse coalesced state) and remembers the mapping; other values unchanged.
            if fake_mode.is_our_fake(e):
                out = torch.zeros_like(e, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                inp_impls[id(out)] = e
                return out
            return e

        flat_args = [to_real_tensor(a) for a in flat_args]
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)

        r = func(*args, **kwargs)

    storages: set[_StoragePointer] = set()

    for e in flat_args:
        if isinstance(e, Tensor):
            if not is_sparse_any(e):
                storages.add(e._typed_storage()._cdata)

    # TODO: also check metadata change on inputs
    # proper aliasing/metadata relationship between outputs and inputs will
    # not be set up, bc of conversion to device, unless we can reuse an
    # input impl

    def map_out(e: T) -> T | FakeTensor:
        # [FUNCTION DOC] HELPER: Maps a real output back: raises the original error if it aliases an input
        # storage without being a stand-in, returns the original fake for stand-ins, fakifies other tensors.
        if id(e) not in inp_impls and (
            isinstance(e, Tensor)
            and not is_sparse_any(e)
            and e._typed_storage()._cdata in storages
        ):
            raise orig_not_implemented_exception

        if isinstance(e, Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, e)
        else:
            return e

    return pytree.tree_map(map_out, r)


def _set_cache_key_for_shape_env(
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry],
    key: _DispatchCacheKey,
    entry: _DispatchCacheEntry,
) -> None:
    # [FUNCTION DOC] HELPER: Stores a cache entry for a shape-env based cache after stripping the shape env from the key.
    key.strip_shape_env()
    cache[key] = entry


def _set_cache_key(
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry],
    key: _DispatchCacheKey,
    entry: _DispatchCacheEntry,
) -> None:
    # [FUNCTION DOC] HELPER: Stores a cache entry in a plain cache mapping.
    cache[key] = entry


# Just for use to allow copying a module to fake tensors,
# does not apply elsewhere
class FakeCopyMode(TorchFunctionMode):
    # [CLASS DOC] Torch function mode used only for copying modules into fake tensors: tensor clone and
    # deepcopy are redirected to the given fake mode (static shapes, with deepcopy memoized), and all other functions run normally.
    def __init__(self, fake_mode: FakeTensorMode) -> None:
        # [FUNCTION DOC] Stores the fake mode.
        self.fake_mode = fake_mode

    def __torch_function__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> FakeTensor:
        # [FUNCTION DOC] Intercepts clone and deepcopy of tensors, returning fake tensors (deepcopy honors the memo); other functions
        # run with subclass torch function disabled. Raises an assertion error on malformed arguments.
        kwargs = kwargs if kwargs else {}

        # clone will get called in Parameter deepcopy
        if func is torch._C.TensorBase.clone:
            if not isinstance(args[0], Tensor):
                raise AssertionError(f"Expected Tensor, got {type(args[0])}")
            return func(
                self.fake_mode.from_tensor(args[0], static_shapes=True), **kwargs
            )
        elif func is Tensor.__deepcopy__:
            if len(args) != 2 or len(kwargs) != 0:
                raise AssertionError(
                    f"Expected 2 args and 0 kwargs for __deepcopy__, got {len(args)} args and {len(kwargs)} kwargs"
                )
            tensor = cast(Tensor, args[0])
            memo = cast(dict[int, FakeTensor], args[1])

            if id(tensor) in memo:
                return memo[id(tensor)]

            out = self.fake_mode.from_tensor(tensor, static_shapes=True)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


def _device_handler(args: Sequence[object]) -> torch.device:
    # [FUNCTION DOC] HELPER: Meta handler for the device query op: returns meta while in kernel invocation, otherwise the fake device of
    # the single FakeTensor argument. Raises an assertion error for other arguments.
    # NB: Don't use is_our_fake, just serve the fake information
    # as is.  Notice we don't use 'self'; we use args[0].fake_mode
    # because they may not be the same.  It would also be possible
    # to return NotImplemented here, in which case the FakeTensor
    # handler on args[0] would handle it, but we're being nice and
    # short-circuiting quickly.
    if len(args) != 1 or not isinstance(args[0], FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        raise AssertionError(
            "Expected exactly one FakeTensor argument for _device_handler"
        )
    if args[0].fake_mode.in_kernel_invocation:
        return torch.device("meta")
    else:
        return args[0].fake_device


# [subclass inputs]
# Suppose we enable fake tensor mode.  This means that fake tensor
# mode will run first.  But what if we do an operation that
# involves a tensor subclass that will desugar into normal tensor
# operations?  Without returning NotImplemented, fake tensor mode will run first,
# decide that a conversion was made (since there was a non fake
# tensor argument), and report an error that converting non
# fake tensor is not supported.  What we actually wanted to happen
# was to give the subclass a chance to figure out what it wants to
# before erroring out. Returning NotImplemented here allows this.
def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    # [FUNCTION DOC] HELPER: True if any flat argument is a non-fake tensor subclass that should be given a chance to dispatch.
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    # [FUNCTION DOC] HELPER: True for tensors that are neither FakeTensor nor plain Tensor, excluding Parameter subclasses that only carry metadata (do not override torch dispatch).
    if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return False
    if not isinstance(x, Tensor):
        return False

    x_type = type(x)
    if x_type is Tensor:
        return False

    # Use the concrete type instead of isinstance because wrapper subclasses
    # marked as Parameters (e.g. Parameter(TwoTensor(...))) still need subclass
    # dispatch. Only metadata-only Parameter subclasses that inherit Tensor's
    # disabled __torch_dispatch__ can follow the regular Parameter path.
    is_metadata_only_parameter_subclass = (
        issubclass(x_type, torch.nn.Parameter)
        and x_type.__torch_dispatch__ is Tensor.__torch_dispatch__
    )
    if is_metadata_only_parameter_subclass:
        return False

    return True


_DISPATCH_META_HANDLERS = {
    torch.ops.prim.device.default: _device_handler,
    torch.ops.aten.size.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).size()
    ),
    torch.ops.aten.stride.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).stride()
    ),
    torch.ops.aten.storage_offset.default: lambda args: int(
        cast(Tensor, args[0]).storage_offset()
    ),
}

_DISPATCH_HANDLE_DIRECTLY = ordered_set(
    torch.ops.aten.is_coalesced.default,
    torch.ops.aten.dense_dim.default,
    torch.ops.aten.sparse_dim.default,
    # _RecordFunction doesn't support __eq__ so make sure not to attempt to
    # cache it.
    torch.ops.profiler._record_function_exit._RecordFunction,
)

from torch._subclasses.fake_impls import (  # noqa: F401
    _device_not_kwarg_ops,
    _is_tensor_constructor,
    _like_tensor_constructors,
    contains_tensor_types,
    get_fast_op_impls,
    has_meta,
    maybe_to_dense_mkldnn,
    op_implementations_checks,
    stride_incorrect_op,
)


def evict_fake_tensor_cache_key(key: _DispatchCacheKey) -> None:
    # [FUNCTION DOC] Weakref finalizer callback: removes the key from the global dispatch cache if present.
    if key in FakeTensorMode.cache:
        FakeTensorMode.cache.pop(key)


@atexit.register
def dump_cache_stats() -> None:
    # [FUNCTION DOC] Exit hook (registered with atexit) logging dispatch cache hits, misses and per-reason bypass counts.
    log.info("FakeTensor cache stats:")
    log.info("  cache_hits: %s", FakeTensorMode.cache_hits)
    log.info("  cache_misses: %s", FakeTensorMode.cache_misses)
    bypasses = FakeTensorMode.cache_bypasses
    if bypasses:
        log.info("  cache_bypasses:")
        width = max(len(k) for k in bypasses)
        for k, v in sorted(bypasses.items(), key=lambda i: -i[1]):
            log.info("    %-*s %s", width + 1, f"{k}:", v)


def _infer_fake_from_real_tensor(
    mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: torch.Tensor
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Infers a fake output for a real tensor by allocating unbacked size symbols per
    # dimension and dense strides computed from the sorted real strides. Raises RuntimeError (via the nested
    # helper) for non-zero storage offset or non-dense real outputs.
    def unsupported(reason: str) -> None:
        # [FUNCTION DOC] HELPER: Raises RuntimeError saying a fake kernel cannot be inferred for the op because of the given reason.
        raise RuntimeError(
            f"propagate_real_tensors: we cannot infer a Fake kernel "
            f"(meta kernel) for operator {op._name} because {reason}. "
            f"Please use torch.library.register_fake to add a Fake kernel."
        )

    if real_out.storage_offset() != 0:
        unsupported(
            f"a return has a non-zero storage offset {real_out.storage_offset()}"
        )

    # Since PT2 is rank specialized, there's no such thing as a symbolic
    # output rank. So we can assume the fake tensor has the same number of
    # dimensions as the real tensor output.
    #
    # We shouldn't assume the Fake sizes/strides are exactly what we see on
    # the real tensor output (perhaps we should give users a lever to toggle
    # this). This is because there's a good amount of operators that return
    # outputs with data-dependent output shape.
    # So we infer the output sizes to all be unbacked symints
    fake_shape = [
        torch._library.fake_impl.allocate_size(mode.shape_env)
        for _ in range(real_out.dim())
    ]

    # We infer what the strides are. We had a couple of options for this:
    # - assume the strides are computable from the sizes
    # - use new fresh unbacked symints in the strides
    #   This doesn't work that well (PT2 doesn't support unbacked symint strides well)
    # - use the real strides
    #   This can only be used if we assume the strides are static.
    # We went with the first option.
    fake_strides = [-1] * real_out.dim()
    strides = [(s, idx) for idx, s in enumerate(real_out.stride())]
    strides.sort(key=lambda x: (x[0], -x[1]))
    expected = 1
    fake_stride = expected
    for s, idx in strides:
        if s != expected:
            unsupported(
                f"a return was not dense in memory (sizes {real_out.shape} strides {real_out.stride()})"
            )
        fake_strides[idx] = fake_stride
        expected = expected * real_out.shape[idx]
        fake_stride = fake_stride * fake_shape[idx]

    with mode:
        return torch.empty_strided(
            fake_shape,
            fake_strides,
            device=real_out.device,
            dtype=real_out.dtype,
            layout=real_out.layout,
        )


def inferred_fake_kernel_from_real_out(
    mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: Any
) -> Any:
    # [FUNCTION DOC] HELPER: Infers fake outputs for all-tensor real outputs of an op lacking a fake kernel,
    # preserving the output structure. Raises AssertionError without a shape env and RuntimeError when any output is not a tensor.
    if mode.shape_env is None:
        raise AssertionError("mode.shape_env must not be None")

    # Only support operators that have all Tensor outputs
    # This is a general limitation on custom ops that we impose for PT2
    # to avoid baking non-symbolic float/int outputs into the graph.
    real_flat_out, spec = pytree.tree_flatten(real_out)
    if not all(isinstance(t, torch.Tensor) for t in real_flat_out):
        raise RuntimeError(
            f"propagate_real_tensors: we don't support operators that return "
            f"non-Tensors. Got {op._schema}"
        )

    fake_flat_out = [_infer_fake_from_real_tensor(mode, op, t) for t in real_flat_out]
    return pytree.tree_unflatten(fake_flat_out, spec)
