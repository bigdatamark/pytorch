# mypy: allow-untyped-defs
# [MODULE DOC] Python-side support for the torch dispatch mode stack and wrapper tensor
# subclasses: the TorchDispatchMode base class, stack push/pop/inspection helpers, infra mode
# (functional/proxy) handling, the traceable wrapper subclass protocol and predicates,
# subclass transformation, and alias/schema analysis used to fix up storage aliasing in
# subclass dispatch handlers. Holds module global flags tracking whether any dispatch mode is
# active; these are updated by mode enter/exit. Importing also imports torchgen and the
# dynamo guards extension.
from __future__ import annotations

import contextlib
import functools
import warnings
from collections import deque
from dataclasses import dataclass
from typing import cast, overload, Protocol, TYPE_CHECKING
from typing_extensions import TypeIs

import torch
import torchgen
import torchgen.model
from torch._C import (
    _get_dispatch_stack_at,
    _len_torch_dispatch_stack,
    _pop_torch_dispatch_stack,
    _push_on_torch_dispatch_stack,
    DispatchKey,
)
from torch._C._dynamo.guards import set_is_in_mode_without_ignore_compile_internals


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch._custom_class_base import CustomClassBase


# TODO: Limitations and things about enable_torch_dispatch_mode we should fix before exposing it:
# - We need a better user-facing api for _DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)

_is_in_torch_dispatch_mode = False
_is_in_non_infra_torch_dispatch_mode = False
# If inside any mode that has ignore_compile_internals() = False
_is_in_any_mode_without_ignore_compile_internals = False


def is_in_torch_dispatch_mode(include_infra_modes: bool = True) -> bool:
    # [FUNCTION DOC] Returns whether execution is inside a dispatch mode context. With
    # include_infra_modes false, only non-infra (user) modes count. Reads module global flags.
    return (
        _is_in_torch_dispatch_mode
        if include_infra_modes
        else _is_in_non_infra_torch_dispatch_mode
    )


def is_in_any_mode_without_ignore_compile_internals() -> bool:
    # [FUNCTION DOC] Returns whether any active mode reports that it does not ignore compile
    # internals (reads a module global flag).
    return _is_in_any_mode_without_ignore_compile_internals


def any_torch_dispatch_mode_on_stack() -> bool:
    # [FUNCTION DOC] Scans the dispatch stack from bottom to top and returns True if any mode is
    # neither an infra mode nor one that ignores compile internals; False otherwise.
    stack_len = torch._C._len_torch_dispatch_stack()

    for idx in range(stack_len):
        mode = _get_dispatch_stack_at(idx)

        # Apply filters first
        if mode.is_infra_mode():
            continue

        if mode.ignore_compile_internals():
            continue

        return True
    return False


class TorchDispatchMode:
    # [CLASS DOC] Base class for dispatch modes that intercept operator calls within a dynamic
    # scope. Subclasses implement the dispatch hook; entering the context pushes the mode on the
    # dispatch stack and exiting pops it. State: three stacks (deques) of saved previous values of
    # the module-level activity flags, one entry per nesting of enter, plus an optional
    # dispatch key stored directly in the instance mapping for pre-dispatch modes. Class
    # attribute supports_higher_order_operators controls handling of higher-order operators.
    # On subclass creation, if dynamo skipping is enabled, a non-classmethod dispatch hook is
    # wrapped to disable dynamo recursively.
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overridable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``self`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    # - When False, custom torch dispatch mode will error out explicitly when a hop
    # is called under the mode.
    # - When True, custom torch dispatch mode's __torch_dispatch__ will be triggered.
    # Mode authors can implement how the mode interacts with higher order operators.
    supports_higher_order_operators = False

    def __init_subclass__(cls, **kwargs):
        # [FUNCTION DOC] Subclass hook: after the normal setup, if the class opts into skipping
        # dynamo and defines its own dispatch hook that is not a classmethod, replaces the hook
        # with a version that recursively disables dynamo. Mutates the new subclass.
        super().__init_subclass__(**kwargs)
        if cls._should_skip_dynamo():
            if "__torch_dispatch__" in cls.__dict__:
                raw = cls.__dict__["__torch_dispatch__"]
                if not isinstance(raw, classmethod):
                    cls.__torch_dispatch__ = torch._disable_dynamo(raw, recursive=True)

    def __init__(self, _dispatch_key=None):
        # [FUNCTION DOC] Optionally records a pre-dispatch key (raises AssertionError if given
        # and not a dispatch key), and creates the three empty stacks of saved flag values.
        if _dispatch_key is not None:
            if not isinstance(_dispatch_key, torch._C.DispatchKey):
                raise AssertionError("_dispatch_key must be a torch._C.DispatchKey")
            self.__dict__["_dispatch_key"] = _dispatch_key

        self.old_dispatch_mode_flags: deque[bool] = deque()
        self.old_non_infra_dispatch_mode_flags: deque[bool] = deque()
        self.old_without_ignore_compile_internals_dispatch_mode_flags: deque[bool] = (
            deque()
        )

    def _lazy_init_old_dispatch_mode_flags(self):
        # [FUNCTION DOC] HELPER: Creates any of the three flag stacks that are missing, for
        # subclasses whose constructor did not call the base constructor.
        if not hasattr(self, "old_dispatch_mode_flags"):
            self.old_dispatch_mode_flags: deque[bool] = deque()  # type: ignore[no-redef]

        if not hasattr(self, "old_non_infra_dispatch_mode_flags"):
            self.old_non_infra_dispatch_mode_flags: deque[bool] = deque()  # type: ignore[no-redef]

        if not hasattr(
            self, "old_without_ignore_compile_internals_dispatch_mode_flags"
        ):
            self.old_without_ignore_compile_internals_dispatch_mode_flags: deque[  # type: ignore[no-redef]
                bool
            ] = deque()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Abstract hook called for each intercepted operator with the operator,
        # participating types, positional and keyword arguments. Subclasses return the result.
        # Raises NotImplementedError here.
        raise NotImplementedError

    def __enter__(self):
        # [FUNCTION DOC] Activates the mode: saves the previous module flags on the per-instance
        # stacks, sets them to reflect this mode (non-infra and without-ignore flags are or-ed
        # with this mode's properties), tells the dynamo guards extension, and pushes the mode on
        # the dispatch stack. Returns self.
        global _is_in_torch_dispatch_mode
        global _is_in_non_infra_torch_dispatch_mode
        global _is_in_any_mode_without_ignore_compile_internals

        # Previously, there wasn't any state in this class' constructor
        # super calls were added to existing modes, but for any new modes
        # this will replicate the previous behavior of not strictly needing
        # to call super().__init__()
        self._lazy_init_old_dispatch_mode_flags()
        self.old_dispatch_mode_flags.append(_is_in_torch_dispatch_mode)
        _is_in_torch_dispatch_mode = True
        self.old_non_infra_dispatch_mode_flags.append(
            _is_in_non_infra_torch_dispatch_mode
        )
        _is_in_non_infra_torch_dispatch_mode = (
            _is_in_non_infra_torch_dispatch_mode or not self.is_infra_mode()
        )
        self.old_without_ignore_compile_internals_dispatch_mode_flags.append(
            _is_in_any_mode_without_ignore_compile_internals
        )
        _is_in_any_mode_without_ignore_compile_internals = (
            _is_in_any_mode_without_ignore_compile_internals
            or not self.ignore_compile_internals()
        )
        set_is_in_mode_without_ignore_compile_internals(
            _is_in_any_mode_without_ignore_compile_internals
        )
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] Deactivates the mode: restores the three module flags from the saved
        # stacks, updates the dynamo guards extension, and pops the mode from the stack (using the
        # stored dispatch key or mode key, if any, to pick the stack). Does not suppress
        # exceptions. Raises IndexError if exit is not paired with an enter.
        mb_dk_or_mode_key = self.__dict__.get("_dispatch_key", None)
        if mb_dk_or_mode_key is None:
            # Today, mode keys are not used at all in the per-dispatch-key-mode logic (for pre-dispatch)
            # We should probably revisit this.
            mb_dk_or_mode_key = self.__dict__.get("_mode_key", None)
        global _is_in_torch_dispatch_mode
        _is_in_torch_dispatch_mode = self.old_dispatch_mode_flags.pop()
        global _is_in_non_infra_torch_dispatch_mode
        _is_in_non_infra_torch_dispatch_mode = (
            self.old_non_infra_dispatch_mode_flags.pop()
        )
        global _is_in_any_mode_without_ignore_compile_internals
        _is_in_any_mode_without_ignore_compile_internals = (
            self.old_without_ignore_compile_internals_dispatch_mode_flags.pop()
        )
        set_is_in_mode_without_ignore_compile_internals(
            _is_in_any_mode_without_ignore_compile_internals
        )
        _pop_mode(mb_dk_or_mode_key)

    @classmethod
    def push(cls, *args, **kwargs):
        # [FUNCTION DOC] Deprecated constructor alias: emits a warning that it is unnecessary and
        # returns a new instance built from the arguments (does not push anything).
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`",
            stacklevel=2,
        )
        instance = cls(*args, **kwargs)
        return instance

    @classmethod
    def is_infra_mode(cls) -> bool:
        # [FUNCTION DOC] Whether this mode class is internal infrastructure (functional or proxy
        # style) rather than user code. Default False; overridden by infra modes.
        return False

    @classmethod
    def _should_skip_dynamo(cls) -> bool:
        # [FUNCTION DOC] HELPER: Class-level switch (default True) deciding whether the dispatch
        # hook is wrapped to skip compilation inside it. Consulted at subclass creation.
        """Skip Dynamo when the flag is set to True

        This is temporary measure to rollout a feature
        that skips PT2 compilation inside __torch_dispatch__
        frames.

        If this flag is off, we would expect following:

        class YoloMode(TorchDispatchMode):
            @classmethod
            def _should_skip_dynamo(cls):
                return False
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

        # instead of recursively disabling, we are compiling into __torch_dispatch__
        assert len(backend.graphs) == 1
        """
        return True

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # [FUNCTION DOC] Whether the mode should be turned off during compilation and see only
        # the compiled artifact's runtime. Default: True for infra modes, False otherwise.
        """Ignore operators that are compiled via torch.compile.

        If ``True``, then this TorchDispatchMode ignores operators that
        are optimized by :func:`torch.compile`. Mechanically, this involves
        turning off the TorchDispatchMode throughout the whole compilation process,
        and turning it back on for the runtime of the compiled artifact(s).
        For example,

        @torch.compile
        def f(x):
            return x.sin().cos()

        with LoggingMode():
            f(x)

        The above example will not log anything if
        ``LoggingMode.ignore_compile_internals()`` is True.
        torch.compile will fuse sin() and cos() into a single operation
        and this TorchDispatchMode will not be passed sin and cos.

        If ``False`` (default), :func:`torch.compile` will respect
        the eager semantics of passing this TorchDispatchMode all
        operators that would have run during eager execution.
        The way this will usually happen is that :func:`torch.compile`
        will just fallback to eager-mode PyTorch.
        """
        if cls.is_infra_mode():
            return True
        return False


def _get_current_dispatch_mode() -> TorchDispatchMode | None:
    # [FUNCTION DOC] HELPER: Returns the mode on top of the dispatch stack, or None if empty.
    """
    Return the top user mode on the stack (the next one that would be
    executed) if there are any.
    """
    stack_len = _len_torch_dispatch_stack()
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    return None


def _detect_infra_mode(key):
    # [FUNCTION DOC] HELPER: Returns the active functional or proxy infra mode for the key,
    # looking at both the pre-dispatch and the regular stacks; None if neither has one. Raises
    # AssertionError for any other key or if both stacks have an active mode.
    if key not in (
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
        torch._C._TorchDispatchModeKey.PROXY,
    ):
        raise AssertionError(
            f"key must be either FUNCTIONAL ({torch._C._TorchDispatchModeKey.FUNCTIONAL}) \
                or PROXY ({torch._C._TorchDispatchModeKey.PROXY}) _TorchDispatchModeKey, \
                    got {key}"
        )
    from torch._ops import _get_dispatch_mode_pre_dispatch

    pre_dispatch_mode = _get_dispatch_mode_pre_dispatch(key)
    post_dispatch_mode = torch._C._get_dispatch_mode(key)

    if pre_dispatch_mode is not None and post_dispatch_mode is not None:
        raise AssertionError(
            "At most one of pre_dispatch_mode and post_dispatch_mode may be active"
        )

    if pre_dispatch_mode is None:
        return post_dispatch_mode

    return pre_dispatch_mode


def _unset_infra_mode(key):
    # [FUNCTION DOC] HELPER: Removes and returns the active infra mode for the key from whichever
    # stack holds it (pre-dispatch or regular); returns None if none. Raises AssertionError if
    # both stacks have one. Side effect: mutates the mode stack.
    from torch._ops import _get_dispatch_mode_pre_dispatch, unset_mode_pre_dispatch

    pre_dispatch_mode = _get_dispatch_mode_pre_dispatch(key)
    post_dispatch_mode = torch._C._get_dispatch_mode(key)
    if pre_dispatch_mode and post_dispatch_mode:
        raise AssertionError(
            "Can't have active infra mode on both pre and post dispatch mode stack"
        )

    if pre_dispatch_mode:
        mode = unset_mode_pre_dispatch(key)
        return mode
    if post_dispatch_mode:
        return torch._C._unset_dispatch_mode(key)


def _disable_infra_mode(key):
    # [FUNCTION DOC] HELPER: Generator that temporarily removes the functional or proxy infra
    # mode for the key, yields it (or None) and pushes it back afterwards. Raises AssertionError
    # for other keys. KNOWN ISSUE: it is not decorated as a context manager, so it cannot be
    # used directly in a with statement as intended.
    if key not in (
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
        torch._C._TorchDispatchModeKey.PROXY,
    ):
        raise AssertionError(
            "key must be either FUNCTIONAL or PROXY _TorchDispatchModeKey"
        )
    mode_unset = _unset_infra_mode(key)
    try:
        yield mode_unset
    finally:
        if mode_unset is not None:
            _push_mode(mode_unset)


def _get_current_dispatch_mode_stack() -> list[TorchDispatchMode]:
    # [FUNCTION DOC] HELPER: Returns a new list of the modes on the dispatch stack, bottom first.
    """
    Returns the current stack of dispatch modes, with the most recent
    (i.e., the one that will be processed first) at the end of the
    list (standard stack convention).
    """
    stack_len = _len_torch_dispatch_stack()
    return [_get_dispatch_stack_at(i) for i in range(stack_len)]


def _push_mode(mode: TorchDispatchMode) -> None:
    # [FUNCTION DOC] HELPER: Pushes a mode. A mode without a dispatch key goes on the regular
    # dispatch stack. A mode with the pre-dispatch key first clears the cached dispatch handlers
    # of all cached ops for the related backend keys, then goes on the pre-dispatch stack.
    # Raises AssertionError for any other dispatch key.
    k = mode._dispatch_key if hasattr(mode, "_dispatch_key") else None
    if k is not None and k != torch._C.DispatchKey.PreDispatch:
        raise AssertionError(
            "mode._dispatch_key must be None or DispatchKey.PreDispatch"
        )
    if k is None:
        _push_on_torch_dispatch_stack(mode)
        return

    from torch._ops import _set_mode_pre_dispatch, get_cached_ops

    # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
    # Clear the cache of every op that has been used so far, for this particular key.
    ks = torch._C._functionality_to_backend_keys(k)
    for op in get_cached_ops():
        for key in ks:
            op._uncache_dispatch(key)
    _set_mode_pre_dispatch(mode)


def _pop_mode(k: DispatchKey | torch._C._TorchDispatchModeKey | None = None):
    # [FUNCTION DOC] HELPER: Pops and returns a mode: from the pre-dispatch stack for the
    # pre-dispatch key, from the regular stack for no key or an infra mode key. Returns None
    # (nothing popped) for any other key.
    if k == torch._C.DispatchKey.PreDispatch:  # type: ignore[attr-defined]
        from torch._ops import _pop_mode_from_pre_dispatch

        return _pop_mode_from_pre_dispatch()

    if k is None or isinstance(k, torch._C._TorchDispatchModeKey):
        return _pop_torch_dispatch_stack(k)


@contextlib.contextmanager
def _pop_mode_temporarily(k: DispatchKey | None = None):
    # [FUNCTION DOC] HELPER: Context manager that pops a mode (per the key), yields it, and
    # pushes it back on exit even on error.
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old)


@contextlib.contextmanager
def _disable_current_modes():
    # [FUNCTION DOC] HELPER: Context manager that pops every mode off the pre-dispatch stack and
    # the regular stack, yields the list of removed modes (pre-dispatch ones first), and
    # restores both stacks on exit. Raises AssertionError if the same infra mode type
    # (functional, proxy, schema check) is present on both stacks.
    from torch._ops import (
        _len_torch_dispatch_stack_pre_dispatch,
        _pop_mode_from_pre_dispatch,
    )
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    from torch._subclasses.schema_check_mode import SchemaCheckMode
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

    mode_len_pre_dispatch = _len_torch_dispatch_stack_pre_dispatch()
    old_pre_dispatch_modes = [
        _pop_mode_from_pre_dispatch() for _ in range(mode_len_pre_dispatch)
    ]

    has_proxy_mode_in_pre_dispatch = False
    has_functional_mode_in_pre_dispatch = False
    has_schema_check_mode_in_pre_dispatch = False

    for i in old_pre_dispatch_modes:
        if isinstance(i, ProxyTorchDispatchMode):
            has_proxy_mode_in_pre_dispatch = True
        if isinstance(i, FunctionalTensorMode):
            has_functional_mode_in_pre_dispatch = True
        if isinstance(i, SchemaCheckMode):
            has_schema_check_mode_in_pre_dispatch = True

    mode_len = _len_torch_dispatch_stack()
    old_modes = [_pop_mode() for _ in range(mode_len)]

    for old in old_modes:
        if (
            isinstance(old, FunctionalTensorMode)
            and has_functional_mode_in_pre_dispatch
        ):
            raise AssertionError(
                "Can't have FunctionalMode available both in PreDispatch and Python Key"
            )
        if isinstance(old, ProxyTorchDispatchMode) and has_proxy_mode_in_pre_dispatch:
            raise AssertionError(
                "Can't have ProxyTorchDispatchMode available both in PreDispatch and Python Key"
            )
        if isinstance(old, SchemaCheckMode) and has_schema_check_mode_in_pre_dispatch:
            raise AssertionError(
                "Can't have SchemaCheckMode available both in PreDispatch and Python Key"
            )

    # Manually disable proxy and fake modes, if any are active
    try:
        yield old_pre_dispatch_modes + old_modes
    finally:
        for mode in reversed(old_modes):
            _push_mode(mode)
        for mode in reversed(old_pre_dispatch_modes):
            _push_mode(mode)


class BaseTorchDispatchMode(TorchDispatchMode):
    # [CLASS DOC] Pass-through mode: a ready-to-use dispatch mode that just runs each operator.
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Calls the operator with the given arguments (defaulting keyword
        # arguments to empty) and returns its result.
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


# Python typing cannot express Intersection[torch.Tensor, Protocol], so this
# protocol repeats the Tensor members that call sites commonly use after
# is_traceable_wrapper_subclass() narrows a value to this protocol.
class TraceableWrapperSubclass(Protocol):
    # [CLASS DOC] Structural type for wrapper tensor subclasses traceable by PT2: requires
    # flatten and unflatten methods plus the commonly used tensor members (shape, stride, size,
    # storage_offset, dim, to). Only declarations with ellipsis bodies; no behavior. The one-line
    # member stubs are not documented individually.
    """
    Canonical protocol for wrapper tensor subclasses that PT2 can trace through.

    ``__tensor_flatten__`` must return stable attribute names for the inner
    values that participate in tracing together with any metadata needed to
    rebuild the outer subclass.

    ``__tensor_unflatten__`` must reconstruct an equivalent wrapper subclass
    instance from the flattened inner values, metadata, and requested outer
    size/stride. Callers may pass tensor attrs and registered reference-type
    opaques in ``inner_tensors``. The returned tensor is expected to preserve
    the requested outer size/stride. If ``attrs, metadata =
    x.__tensor_flatten__()``, then ``type(x).__tensor_unflatten__`` must round-
    trip from ``{name: getattr(x, name) for name in attrs}``, ``metadata``,
    ``x.size()``, and ``x.stride()`` to an equivalent instance of ``x``.

    ``__tensor_unflatten__`` may be implemented as either a ``@staticmethod``
    or a ``@classmethod``; the runtime check below intentionally uses duck
    typing to support both forms, even though static type checkers may only
    recognize the ``@staticmethod`` form as conforming to this protocol.
    """

    def __tensor_flatten__(self) -> tuple[Sequence[str], object]: ...

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Mapping[str, torch.Tensor | CustomClassBase],
        metadata: object,
        outer_size: Sequence[int | torch.SymInt],
        outer_stride: Sequence[int | torch.SymInt],
    ) -> torch.Tensor: ...

    shape: torch._C.Size

    @overload
    def stride(self, dim: None = None) -> tuple[int, ...]: ...

    @overload
    def stride(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = None) -> tuple[int, ...]: ...

    @overload
    def size(self, dim: int) -> int: ...

    def storage_offset(self) -> int: ...

    def dim(self) -> int: ...

    @overload
    def to(
        self,
        dtype: torch.types._dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> torch.Tensor: ...

    @overload
    def to(
        self,
        device: torch._prims_common.DeviceLikeType | None = None,
        dtype: torch.types._dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> torch.Tensor: ...

    @overload
    def to(
        self,
        other: torch.Tensor,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> torch.Tensor: ...


TensorWithFlatten = TraceableWrapperSubclass


def _has_traceable_wrapper_subclass_protocol(t: object) -> bool:
    # [FUNCTION DOC] HELPER: True if the object (instance or class) has both the flatten and
    # the unflatten attributes (duck typing only).
    return hasattr(t, "__tensor_flatten__") and hasattr(t, "__tensor_unflatten__")


def is_traceable_wrapper_subclass(t: object) -> TypeIs[TraceableWrapperSubclass]:
    # [FUNCTION DOC] True if t is an instance of a strict tensor subclass (not a plain tensor)
    # that has the flatten and unflatten attributes. Pure.
    """
    Returns whether ``t`` is a tensor subclass that matches the
    ``TraceableWrapperSubclass`` protocol at runtime.

    See ``TraceableWrapperSubclass`` for the canonical flatten/unflatten
    contract. Matching the protocol is necessary but not sufficient for full
    PT2 support: the subclass's ``__torch_dispatch__`` implementation must also
    desugar into traceable dispatcher operations and preserve aliasing semantics
    with ``return_and_correct_aliasing()`` when needed.
    """
    is_subclass = isinstance(t, torch.Tensor) and type(t) is not torch.Tensor
    return is_subclass and _has_traceable_wrapper_subclass_protocol(t)


def is_traceable_wrapper_subclass_type(
    t: type,
) -> TypeIs[type[TraceableWrapperSubclass]]:
    # [FUNCTION DOC] Class version of the instance check: True if t is a strict tensor subclass
    # type with the flatten and unflatten attributes. Raises TypeError if t is not a class.
    """Same as above, but takes a type argument instead of an instance."""
    return (
        issubclass(t, torch.Tensor)
        and t is not torch.Tensor
        and _has_traceable_wrapper_subclass_protocol(t)
    )


def transform_subclass(t, callback, outer_size=None, outer_stride=None):
    # [FUNCTION DOC] Flattens the wrapper subclass t, applies callback (given attribute name and
    # inner tensor) to each inner tensor, and rebuilds a new instance of the same type using the
    # context, with outer size and stride defaulting to those of t. Raises AssertionError if the
    # rebuilt tensor's shape or stride differs from the requested ones. Returns the new tensor.
    # KNOWN ISSUE: the error messages join the type and the method name without a dot separator.
    """
    Given a traceable, wrapper tensor subclass ``t`` that implements
    ``__torch_dispatch__`` and holds some inner tensors,
    and a callback of type ``Callable[[str, torch.Tensor], torch.Tensor]``,
    `transform_subclass` will construct a fresh instance of the wrapper tensor subclass.
    It will do so by grabbing each inner tensor attribute from the wrapper,
    passing them into ``callback`` to get a transformed tensor,
    and putting each transformed tensor into the fresh tensor subclass instance.

    Note: this function will not handle ensuring that the fresh subclass
    gets the same (autograd, and aliasing) metadata as the original tensor.
    This is generally handled in other subsystems like AOTAutograd.
    """
    outer_size = outer_size if outer_size is not None else t.size()
    outer_stride = outer_stride if outer_stride is not None else t.stride()

    attrs, ctx = t.__tensor_flatten__()
    transformed_tensors_dict = {}
    for attr in attrs:
        transformed_tensors_dict[attr] = callback(attr, getattr(t, attr))
    sub = type(t).__tensor_unflatten__(
        transformed_tensors_dict, ctx, outer_size, outer_stride
    )

    # NB: Purposefully guard here to simplify the inner / outer symbols.
    # Using sym_eq() for symbolic comparison can result in an expression that's too
    # difficult to guard on, so we use == here.
    if sub.shape != outer_size:
        raise AssertionError(
            f"Expected return value from {type(t)}__tensor_unflatten__() to have "
            f"shape equal to {outer_size}, but got: {sub.shape}"
        )
    if sub.stride() != outer_stride:
        raise AssertionError(
            f"Expected return value from {type(t)}__tensor_unflatten__() to have "
            f"stride equal to {outer_stride}, but got: {sub.stride()}"
        )

    return sub


def _correct_storage_aliasing(func, schema_info, args, outs) -> None:
    # [FUNCTION DOC] HELPER: For each (argument index, output index) pair in the schema info's
    # read-only alias matches, makes the output share the input's storage by an unsafe storage
    # swap, leaving output sizes and strides untouched. Mutates the output tensors. Raises
    # AssertionError if func is not an OpOverload, args is not a tuple, outs is not a list or
    # tuple, or if types of aliased arguments and subclass outputs do not match.
    # KNOWN ISSUE: two of the argument-check error messages report the type of args instead of
    # the type of func or outs.
    """
    Given: an OpOverload, a SchemaInfo (cached information from torchgen about schema),
    and the inputs/outputs to the OpOverload,
    this function checks to see if func is a view operator
    (by checking if any of the outputs in the op's schema
     are immutable aliases of inputs).
    If so, this function manually aliases the storage of the output tensor
    with its corresponding input tensor alias.
    It does this by unsafely overwriting the storage field of the output tensor
    to be the same storage as the input.
    """
    if not isinstance(func, torch._ops.OpOverload):
        raise AssertionError(f"func must be an OpOverload, got {type(args)}")
    if not isinstance(args, tuple):
        raise AssertionError(f"args must be a tuple, got {type(args)}")
    if not isinstance(outs, (list, tuple)):
        raise AssertionError(f"outs must be a list or tuple, got {type(args)}")

    def alias_non_inplace_storage(arg, ret) -> None:
        # [FUNCTION DOC] HELPER: Verifies that argument and result types match when either is a
        # traceable wrapper subclass (for a list result, each element), then swaps the storage
        # of the result (each element of a list) to that of arg. Raises AssertionError on a type
        # mismatch or a result that is neither a tensor nor a list.
        # This is hopefully a reasonable assert:
        # subclasses that rely on this API for output aliasing
        # should always return wrapper tensor subclasses for us to manually alias.
        # in theory if a subclass that needs this API wants to sometimes return
        # plain tensors, we could remove the assert and just not perform the aliasing,
        # but it seems safer to learn more about this case first.
        #
        # Performance note: This is all just to assert that the argument and result
        # types match, checking that is cheaper than is_traceable_wrapper_subclass_type,
        # and multiple returns are relatively unlikely, so just check up front!
        arg_type = type(arg)
        ret_type = type(ret)
        if arg_type is not ret_type and (
            is_traceable_wrapper_subclass_type(arg_type)
            or is_traceable_wrapper_subclass_type(ret_type)
        ):
            ret_list = ret if isinstance(ret, list) else [ret]
            for r in ret_list:
                if type(arg) is not type(r):
                    raise AssertionError(
                        f"Called {str(func)} with input of type {type(arg)}\n"
                        f"and output of type {type(ret)}. But expected types to match."
                    )
        # Need to call a non-dispatcher helper, because we explicitly do **not**
        # want our subclass to intercept the set_() call.
        # instead, our subclass should directly have its storage swapped out.
        # we **explicitly** don't want to reset the sizes on ret, if the storage implies a size change.
        # Why?
        # The purpose of this API is *not* to change the size/strides of our output- we assume it's already correct.
        # We just want to "fix up" the storage aliasing, without modifying or output's metadata.
        # Example: out = inp.expand(inp.shape[0], inp.shape[0])
        #     This requires swapping the storage of out to be the same as inp,
        #     but we do *not* want it to change the sizes/strides that were compute for out.

        if isinstance(ret, list):
            for r in ret:
                torch._functionalize_unsafe_set(r, arg)
        else:
            if not isinstance(ret, torch.Tensor):
                raise AssertionError(f"expected torch.Tensor, got {type(ret)}")
            torch._functionalize_unsafe_set(ret, arg)

    for arg_idx, return_idx in schema_info.read_only_alias_match_indexes:
        alias_non_inplace_storage(args[arg_idx], outs[return_idx])


def _get_write_alias(x) -> str | None:
    # [FUNCTION DOC] HELPER: For an alias info record, returns the single alias name if it is a
    # write alias, or None if it has no aliases or is not written. Raises AssertionError if the
    # alias set has more than one name.
    alias_set = x.alias_set
    if not alias_set or not x.is_write:
        return None
    # torchscript allows for complicated alias sets, but our dispatcher ops only really involve simple aliasing
    if len(alias_set) != 1:
        raise AssertionError("Expected alias_set to contain exactly one element")
    # timeit says next(iter(alias_set)) is faster than list(alias_set)[0] even for
    # set of size 1 on Python 3.13.
    return next(iter(alias_set))


# This abstracts over the fact that in return_and_correct_aliasing,
# we sometimes use torchgen schema parsing (for aten ops, since torchscript's schema parsing is sometimes buggy),
# and sometimes use torchscript schema parsing (for custom ops, for which torchgen parsing is untested).
@dataclass
class AliasInfo:
    # [CLASS DOC] Alias annotation of one schema argument or return: alias_set (names of alias
    # sets), is_write (whether the annotation marks mutation) and name (argument name or None).
    alias_set: set[str]
    is_write: bool
    name: str | None


@dataclass
class SchemaInfo:
    # [CLASS DOC] Cached alias summary of an operator schema: alias info for arguments and
    # outputs, whether the op is an in-place view op, the write alias name per output (or None
    # when no output is written) and the read-only (argument index, output index) alias matches.
    args: list[AliasInfo]
    outs: list[AliasInfo]

    is_inplace_view_op: bool

    # [_get_write_alias(x) for x in outs]. Guaranteed to contain no Nones; we coerce
    # all-Nones result to empty list instead, and we don't support
    # some-but-not-all-Nones.
    outs_write_aliases: list[str] | None

    # List of (arg_idx, return_idx) where args[arg_idx].alias_set &
    # outs[out_idx].alias_set is not empty, and not args[arg_idx].is_write.
    read_only_alias_match_indexes: list[tuple[int, int]]


# Given an OpOverload, returns schema information on it.
# This is cached for efficiency, since it can involve running torchgen
@functools.cache
def get_alias_info(func) -> SchemaInfo:
    # [FUNCTION DOC] Builds (and caches per op) the alias summary of an operator overload. ATen
    # ops are parsed with torchgen after text fixes to the schema string; other ops use the
    # runtime schema's alias info. Computes read-only alias matches, per-output write aliases
    # and the in-place view tag. Raises AssertionError if an ATen schema string lacks the
    # namespace prefix, and RuntimeError if only some outputs are written aliases.
    # For ATen ops: use torchgen (since torchscript parser doesn't handle alias annotations
    # properly for some ops that output tensorlists)
    if func.namespace == "aten":
        torchgen_schema_str = str(func._schema)
        if not torchgen_schema_str.startswith("aten::"):
            raise AssertionError(
                "Expected torchgen schema string to start with 'aten::'"
            )
        # remove the aten:: namespace, which is added by the torchscript parser,
        # and torchgen doesn't know how to handle
        torchgen_schema_str = torchgen_schema_str[6:]
        import re

        # the torchscript parser ends up converting int[2]=1 into int[2]=[1, 1],
        # which torchgen chokes on.
        torchgen_schema_str = re.sub(r"=\[[0, ]+\]", "=0", torchgen_schema_str)
        torchgen_schema_str = re.sub(r"=\[[1, ]+\]", "=1", torchgen_schema_str)
        # for aten::rot90 / aten:fft_*
        torchgen_schema_str = re.sub(
            r"=\[(-?[0-9]+), (-?[0-9]+)\]", r"=[\1,\2]", torchgen_schema_str
        )
        torchgen_schema = torchgen.model.FunctionSchema.parse(torchgen_schema_str)
        arg_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.annotation is None else set(a.annotation.alias_set)
                ),
                is_write=a.annotation is not None and a.annotation.is_write,
                name=a.name,
            )
            for a in torchgen_schema.arguments.flat_all
        ]
        out_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.annotation is None else set(a.annotation.alias_set)
                ),
                is_write=a.annotation is not None and a.annotation.is_write,
                name=a.name,
            )
            for a in torchgen_schema.returns
        ]
    else:
        # For non-aten ops, torchgen is untested so we rely on torchscript schema parsing
        arg_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.alias_info is None else set(a.alias_info.before_set)
                ),
                is_write=a.alias_info is not None and a.alias_info.is_write,
                name=a.name,
            )
            for a in func._schema.arguments
        ]
        out_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.alias_info is None else set(a.alias_info.before_set)
                ),
                is_write=a.alias_info is not None and a.alias_info.is_write,
                name=a.name,
            )
            for a in func._schema.returns
        ]
    read_only_alias_match_indexes = []
    for arg_idx, schema_arg in enumerate(arg_schemas):
        for return_idx, schema_out in enumerate(out_schemas):
            is_read_only_alias_match = (
                schema_arg.alias_set & schema_out.alias_set
            ) and not schema_arg.is_write
            if is_read_only_alias_match:
                read_only_alias_match_indexes.append((arg_idx, return_idx))

    outs_write_aliases_list: list[str | None] = [
        _get_write_alias(r) for r in out_schemas
    ]
    non_nones = sum(x is not None for x in outs_write_aliases_list)
    if non_nones == 0:
        outs_write_aliases: list[str] | None = None
    elif non_nones != len(outs_write_aliases_list):
        # simplifying assumption: we don't have **any** ops with return types like "-> (Tensor(a!), Tensor)"
        raise RuntimeError("Unsupported schema: " + str(func._schema))
    else:
        outs_write_aliases = cast(list[str], outs_write_aliases_list)

    schema_info = SchemaInfo(
        args=arg_schemas,
        outs=out_schemas,
        # This check is surprisingly expensive because pybind11 enum_s are
        # inefficient. Just cache it.
        is_inplace_view_op=torch.Tag.inplace_view in func.tags,
        outs_write_aliases=outs_write_aliases,
        read_only_alias_match_indexes=read_only_alias_match_indexes,
    )
    return schema_info


def autograd_would_have_decomposed(
    func: torch._ops.OpOverload, flat_args: Sequence[torch.Tensor | object]
) -> bool:
    # [FUNCTION DOC] Decides, from the first tensor in flat_args, whether autograd would have
    # decomposed the operator: returns True unless the operator has a kernel registered for that
    # tensor's device backend key. With no tensors returns True. Raises AssertionError if the
    # device type has no dispatch key.
    """
    Suppose that an operator has CompositeImplicitAutograd decomp registered.
    Would autograd have used this decomposition?  It will only use it if there
    isn't an explicit backend registration for the device as well.  This function
    will tell if this would have occurred.

    Why do we need to apply these decompositions later?  When inference mode is
    on, the autograd key is bypassed entirely, so a lower level mode cannot rely
    on the decomposition having been applied.  It's easy to accidentally never apply
    the decomposition, resulting in an operator showing up in a graph that
    is unexpected.

    Why do we need to AVOID applying the decomposition when autograd wouldn't
    have decomposed?  If autograd doesn't decompose, this means in eager mode
    we would have run the fused kernel.  It must be possible to trace this
    fused kernel directly into the graph for fidelity with eager (NB: a user
    has the option of then further decomposing at proxy tensor mode via
    decomposition table, but we must preserve it to proxy mode to have the
    choice.)

    Why does functionalization need to also perform the test here?  This is
    because some CompositeImplicitAutograd decompositions are not functional.
    If we are eventually going to decompose, we need to do this while we can
    still turn functionalization back on, so those decompositions get functionalized.
    So an early decomposition in functionalization may still be necessary.  Note that
    if proxy tensor decomposition process could turn functionalization back on, this
    wouldn't be necessary, and maybe that is a useful thing to do anyway because
    the decomposition table is user specified and a user could violate the functional
    decomp requirement with a bad decomp.  If this happened, then you could always
    pass through functionalization.
    """
    has_backend_registration = False
    for a in flat_args:
        if isinstance(a, torch.Tensor):
            backend_key = torch._C._parse_dispatch_key(
                torch._C._dispatch_key_for_device(a.device.type)
            )
            if backend_key is None:
                raise AssertionError(
                    f"failed to parse dispatch key for device {a.device.type}"
                )
            # TODO: use func.has_kernel_for_dispatch_key(backend_key)
            # but this one checks py_impl and CompositeImplicitAutograd
            # incorrectly shows up as has backend reg here
            has_backend_registration = torch._C._dispatch_has_kernel_for_dispatch_key(
                func.name(), backend_key
            )

            # in theory we should take all backend keys and take the highest priority one
            # to properly mimic the dispatcher,
            # this just grabs the first tensor and takes its device key
            break
    return not has_backend_registration


def return_and_correct_aliasing(func, args, kwargs, out):
    # [FUNCTION DOC] Post-processes a subclass's operator result to honor aliasing: aliases
    # output storages for view ops; for in-place view ops, re-runs the op under no-dispatch with
    # the meta-in-TLS flag set to update wrapper metadata (skipped for functional tensors); and
    # for ops with mutable output aliases returns the corresponding input argument(s) (as the
    # same container type as out when several) instead of out. Otherwise returns out.
    # Raises AssertionError on unexpected schema shapes.
    """
    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses
    that would like to work with torch.compile. It ensures that the subclass
    properly implements the aliasing behavior of every op,
    which is needed for correctness in AOTAutograd.
    This function will handle:

        * When we see a view op, we will alias the storages of any
          input and output tensor subclasses

        * When we see an inplace or out= op, we will directly
          return the corresponding input tensor, instead of returning
          a (potentially) fresh output tensor.
    """

    # Caching here because torchgen parsing is definitely not fast, and this function is called
    # once for every op in the graph during functionalization.
    schema_info = get_alias_info(func)

    def get_arg_from_alias(output_alias, schema_info, args, kwargs):
        # [FUNCTION DOC] HELPER: Normalizes the call arguments and returns the one input argument
        # whose alias set contains output_alias (by keyword name if present, else by position).
        # Raises AssertionError unless exactly one argument matches.
        new_args, new_kwargs = torch.fx.operator_schemas.normalize_function(  # type: ignore[misc]
            func, args=args, kwargs=kwargs
        )

        arg_indices = [
            i for i, a in enumerate(schema_info.args) if output_alias in a.alias_set
        ]
        # For any dispatcher op with an output alias, we expect it to map to exactly one alias in the schema's input arguments.
        if len(arg_indices) != 1:
            raise AssertionError(
                "Expected exactly one argument index for the given output alias"
            )
        idx = arg_indices[0]
        arg_info = schema_info.args[idx]
        if arg_info.name is not None and arg_info.name in new_kwargs:
            return new_kwargs[arg_info.name]
        return new_args[idx]

    # Fix up the storages of any outs so that they point to the same storage as the input,
    # if func is a view op.
    _correct_storage_aliasing(
        func, schema_info, args, (out,) if not isinstance(out, tuple) else out
    )

    # For inplace_view ops in particular, we'll try hard to make sure that the wrapper subclass's
    # metadata is set correctly.
    if schema_info.is_inplace_view_op:
        # no_dispatch() to make sure that we secretly change the metadata on the wrapper,
        # but don't end up dispatching the op anywhere else.
        mutated_args = [
            x
            for i, x in enumerate(args)
            if _get_write_alias(schema_info.args[i]) is not None
        ]
        # Assumption: we have a very small number of inplace_view ops that follow a strict schema:
        # there is only a single argument that gets its metadata mutated.
        if len(mutated_args) != 1:
            raise AssertionError(
                "expected exactly one mutated arg for inplace_view ops"
            )
        # This check exists because we generally *do* want to update the metadata of any wrapper subclasses,
        # but FunctionalTensor is special: it overrides all size/stride calls to plumb to the inner tensor.
        # so we don't actually need to update the metadata (and attempting to do so causes errors)
        from torch._subclasses.functional_tensor import FunctionalTensor

        if not isinstance(mutated_args[0], FunctionalTensor):
            with torch.utils._mode_utils.no_dispatch():
                # See Note: [Fake Tensor Dispatch Keys]
                # we're borrowing the way it modifies dispatch key TLS.
                meta_in_tls = torch._C._meta_in_tls_dispatch_include()
                torch._C._set_meta_in_tls_dispatch_include(True)
                try:
                    func(*args, **kwargs)
                finally:
                    torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

    # Next: we need to make sure to return inputs directly, if the output is a mutable alias (e.g. add_()).

    schema_info_outs_write_aliases = schema_info.outs_write_aliases
    # simple case: none of our outputs have mutable aliases, so we can return the output as-is
    if schema_info_outs_write_aliases is None:
        return out

    if len(schema_info_outs_write_aliases) == 1:
        return get_arg_from_alias(
            schema_info_outs_write_aliases[0], schema_info, args, kwargs
        )

    # In the multi-return case, all aten ops return a tuple / list, so cast accordingly.
    outs_to_return = type(out)(
        [
            (get_arg_from_alias(write_alias, schema_info, args, kwargs))
            for write_alias in schema_info_outs_write_aliases
        ]
    )
    return outs_to_return
