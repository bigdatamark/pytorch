# [MODULE DOC] Small helper functions and classes that Dynamo is explicitly allowed to trace
# through or call during compilation: compile-state check, inline/numpy/context-manager
# wrappers, compiled-autograd hook and backward helpers, and constant-masking utilities.
# Import-time side effects: tries to import numpy (np is None if missing); defines type vars.
"""
This module contains utility functions that are explicitly allowed to be called during
TorchDynamo compilation. These functions are carefully vetted to ensure they work
correctly within the TorchDynamo tracing and compilation process.

Key functionality groups:

- Compilation State:
  Functions for checking compilation state (is_compiling)

- Function Wrapping:
  Utilities for wrapping functions (wrap_inline, wrap_numpy) to work with
  TorchDynamo compilation

- Autograd Hooks:
  Functions and classes for handling autograd hooks and backward passes
  (call_hook, FakeBackwardCFunction, etc.)

- Tensor Operations:
  Utility functions for tensor operations and transformations
"""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import deprecated, ParamSpec

import torch
import torch.utils._pytree as pytree


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

_P = ParamSpec("_P")
_R = TypeVar("_R")

if TYPE_CHECKING:
    # TorchScript does not support `@deprecated`
    # This is a workaround to avoid breaking TorchScript
    @deprecated(
        "`torch._dynamo.external_utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
        category=FutureWarning,
    )
    def is_compiling() -> bool:
        # [FUNCTION DOC] Type-checking-only deprecated stub; returns whether the compiler is
        # currently tracing (delegates to the compiler module). Only defined for type checkers.
        return torch.compiler.is_compiling()

else:

    def is_compiling() -> bool:
        # [FUNCTION DOC] Returns True while tracing under torch.compile or torch.export, by
        # delegating to the compiler module's check. No side effects.
        # Note: under the eager backend this function itself is traced and yields true.
        """
        Indicates whether we are tracing/compiling with torch.compile() or torch.export().
        """
        # NOTE: With `@torch.compile(backend="eager")`, torch._dynamo.is_compiling() will get traced
        # and return true. torch.compiler.is_compiling() is skipped and will return false.
        return torch.compiler.is_compiling()


def wrap_inline(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Returns a wrapper (preserving fn metadata) that simply forwards all arguments
    # to fn, so that an extra traceable frame exists that is not in the skip list.
    """
    Create an extra frame around fn that is not in skipfiles.
    """

    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Calls fn with the given arguments and returns its result.
        return fn(*args, **kwargs)

    return inner


def call_hook(
    hook: Callable[..., torch.Tensor | None], *args: Any, **kwargs: Any
) -> torch.Tensor:
    # [FUNCTION DOC] Calls hook with the positional args (kwargs are not forwarded). If the hook
    # returns None, returns the first positional argument unchanged; otherwise returns the
    # hook result. Raises RuntimeError if the hook returned a value while the hook_type keyword
    # is post_acc_grad_hook (those hooks must return None).
    # KNOWN ISSUE: the post_acc_grad_hook error is checked only after the None case is handled.
    """
    Used by compiled autograd to handle hook returning None.
    """
    result = hook(*args)
    if result is None:
        return args[0]
    elif kwargs.get("hook_type") == "post_acc_grad_hook":
        raise RuntimeError("Tensor post accumulate grad hooks should return None.")
    return result


def wrap_numpy(f: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Decorator: returns f unchanged when numpy is unavailable; otherwise returns a
    # wrapper that converts all tensor arguments to numpy arrays, calls f, and converts numpy
    # array results back to tensors.
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.
    """
    if not np:
        return f

    @functools.wraps(f)
    def wrap(*args: _P.args, **kwargs: _P.kwargs) -> pytree.PyTree:
        # [FUNCTION DOC] HELPER: Maps tensors in args/kwargs to numpy arrays, calls f, maps numpy
        # arrays in the output (any pytree) to tensors, and returns it.
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, lambda x: x.numpy(), (args, kwargs)
        )
        out = f(*args, **kwargs)
        # pyrefly: ignore [missing-attribute]
        return pytree.tree_map_only(np.ndarray, lambda x: torch.as_tensor(x), out)

    return wrap


class FakeBackwardCFunction:
    # [CLASS DOC] Proxy for an autograd backward context object used by compiled autograd.
    # Attributes: real (the underlying backward context) and saved_tensors (explicit list of
    # saved tensors that overrides the real one). Unknown attributes are forwarded to real.
    def __init__(
        self,
        real: torch.autograd.function.BackwardCFunction,
        saved_tensors: list[torch.Tensor],
    ) -> None:
        # [FUNCTION DOC] Stores the real context and the saved tensors list.
        self.real = real
        self.saved_tensors = saved_tensors

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC] Called only for attributes not found normally. For the deprecated name
        # saved_variables, emits a DeprecationWarning and returns saved_tensors; for any other
        # name returns the attribute from the real context (raising AttributeError if absent).
        if name == "saved_variables":
            warnings.warn(
                "'saved_variables' is deprecated; use 'saved_tensors'",
                DeprecationWarning,
            )
            return self.saved_tensors

        return getattr(self.real, name)


def call_backward(
    backward_c_function: torch.autograd.function.BackwardCFunction,
    saved_tensors: list[torch.Tensor],
    *args: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Runs the user-defined backward of the custom autograd function that owns
    # backward_c_function, using a fake context carrying the given saved tensors, with args as
    # the incoming gradients. Returns the gradients always as a tuple (a single result is
    # wrapped). Propagates any error from the user backward.
    # KNOWN ISSUE: the annotated return type allows a bare tensor but a tuple is always returned.
    fake = FakeBackwardCFunction(backward_c_function, saved_tensors)
    grads = fake._forward_cls.backward(fake, *args)  # type: ignore[attr-defined]

    if not isinstance(grads, tuple):
        grads = (grads,)

    return grads


def normalize_as_list(x: Any) -> list[Any]:
    # [FUNCTION DOC] Returns a new list for a tuple, the same list for a list, and a one-element
    # list wrapping any other value.
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


def untyped_storage_size(x: torch.Tensor) -> int:
    # [FUNCTION DOC] Returns the size of the tensor's underlying untyped storage.
    return x.untyped_storage().size()


class FakeCompiledAutogradEngine:
    # [CLASS DOC] Stateless stand-in for the autograd engine inside compiled autograd graphs;
    # offers static methods to queue and run final callbacks held in a caller-owned list.
    @staticmethod
    def queue_callback(
        final_callbacks: list[Callable[[], None]], cb: Callable[[], None]
    ) -> None:
        # [FUNCTION DOC] Appends cb to the final_callbacks list (mutates it in place).
        final_callbacks.append(cb)

    @staticmethod
    def exec_final_callbacks(final_callbacks: list[Callable[[], None]]) -> None:
        # [FUNCTION DOC] Runs every callback in the list in order, including ones appended while
        # running, then clears the list. An exception from a callback propagates and leaves the
        # list uncleared.
        i = 0
        while i < len(final_callbacks):
            cb = final_callbacks[i]
            cb()
            i += 1
        final_callbacks.clear()

    @staticmethod
    def _exec_final_callbacks_stub() -> None:
        # [FUNCTION DOC] HELPER: No-op placeholder with no arguments, result or side effects.
        pass


def call_hook_from_backward_state(
    *args: Any, bw_state: Any, hook_name: str, **kwargs: Any
) -> Any:
    # [FUNCTION DOC] Looks up the hook stored on bw_state under hook_name and calls it with the
    # given positional and keyword arguments; returns its result.
    return getattr(bw_state, hook_name)(*args, **kwargs)


def call_module_hooks_from_backward_state(
    _: Any, result: Any, *args: Any, bw_state: Any, hooks_name: str, module_name: str
) -> Any:
    # [FUNCTION DOC] Fetches the module (by module_name) and its list of hooks (by hooks_name)
    # from bw_state, then calls each hook in order with the module, the current result and the
    # extra args; a non-None hook return replaces the result. The first (ignored) argument is
    # unused. Returns the final result.
    module = getattr(bw_state, module_name)
    hooks = getattr(bw_state, hooks_name)
    for hook in hooks:
        new_result = hook(module, result, *args)
        if new_result is not None:
            result = new_result
    return result


# used for torch._dynamo.disable(recursive=False)
def get_nonrecursive_disable_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Returns a wrapper around fn (metadata preserved) used for the non-recursive
    # form of the compiler disable decorator. Defined here so the frame converter does not skip it.
    # wrap function to get the right error message
    # this function is in external_utils so that convert_frame doesn't skip it.
    @functools.wraps(fn)
    def nonrecursive_disable_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Raises RuntimeError if called while exporting; otherwise calls
        # fn with the arguments and returns its result.
        if torch.compiler.is_exporting():
            raise RuntimeError(
                "Non-recursive torch.compiler.disable is not supported with torch.export."
            )
        return fn(*args, **kwargs)

    return nonrecursive_disable_wrapper


def wrap_dunder_call_ctx_manager(self: Any, func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Returns a function that enters self as a context manager around a call to
    # func, forwarding all arguments and returning func's result. Intentionally not
    # functools-wrapped so the frame is never skipped.
    """
    Apply self as a ctx manager around a call to func
    """

    # NOTE: do not functools.wraps(func) because we don't ever want this frame to be skipped!
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Runs func inside the with-block of self and returns its result.
        with self:
            return func(*args, **kwargs)

    return inner


# Use only on ints marked dynamic via torch.empty(0, integer)
# Currently only way to mark ints as dynamic: https://github.com/pytorch/pytorch/issues/129623
def unwrap_maybe_dynamic_int(x: torch.Tensor | int) -> int:
    # [FUNCTION DOC] If x is a tensor (the marker used for dynamic ints, shaped zero by N),
    # returns its size along dimension 1 as the int; otherwise returns x unchanged.
    if isinstance(x, torch.Tensor):
        # x.size() is expected to be [0, dynamic_int]
        return x.size(1)
    return x


def call_accumulate_grad(
    variable: torch.Tensor,
    variable_grad: torch.Tensor | None,
    grad: torch.Tensor | bool,
    has_post_hooks: bool | None = None,
) -> torch.Tensor | None:
    # [FUNCTION DOC] Accumulates grad into variable's gradient via the compiled-autograd
    # AccumulateGrad op, assigns the updated gradient to variable.grad (mutation) and returns it.
    # Legacy calling convention: when has_post_hooks is None, the third argument is really the
    # has-post-hooks flag, the second argument is the new grad, and the current grad is read
    # from variable.grad.
    if has_post_hooks is None:
        # Backward compatibility for graphs captured before current grad became
        # an explicit argument.
        has_post_hooks = bool(grad)
        grad = variable_grad  # type: ignore[assignment]
        variable_grad = variable.grad
    updated_grad = torch._dynamo.compiled_autograd.ops.AccumulateGrad(  # type: ignore[attr-defined]
        [grad], variable, variable_grad, has_post_hooks
    )
    variable.grad = updated_grad[0]
    return variable.grad


def wrap_inline_with_error_on_graph_break(
    fn: Callable[_P, _R], error_on_graph_break: bool
) -> Callable[_P, _R]:
    # [FUNCTION DOC] Returns a wrapper that calls fn inside the error-on-graph-break context set
    # to the given flag (True or False). Two separate wrapper definitions avoid capturing the
    # flag as a free variable. Wrappers are intentionally not functools-wrapped.
    # NB: need multiple definitions in order to prevent `fullgraph` from
    # being a freevar of wrapper
    # NOTE: do not functools.wraps(fn) because we don't ever want these wrappers to be skipped!
    if error_on_graph_break:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # [FUNCTION DOC] HELPER: Calls fn with graph-break errors enabled; returns result.
            with torch._dynamo.error_on_graph_break(True):
                return fn(*args, **kwargs)

    else:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # [FUNCTION DOC] HELPER: Calls fn with graph-break errors disabled; returns result.
            with torch._dynamo.error_on_graph_break(False):
                return fn(*args, **kwargs)

    return wrapper


def filter_out_const_values(tup: tuple[Any, ...], masks: list[bool]) -> tuple[Any, ...]:
    # [FUNCTION DOC] Returns a tuple of the elements of tup whose mask entry is False (constants
    # are dropped). Iterates over masks, so tup must be at least as long as masks.
    """
    masks is a list of bools, where True means the corresponding element in tup
    is a const value. Filter out the const values.
    """
    out = []
    for mask_idx, mask in enumerate(masks):
        if not mask:
            out.append(tup[mask_idx])
    return tuple(out)


def insert_const_values_with_mask(
    tup: tuple[Any, ...], masks: list[bool], values: tuple[Any, ...]
) -> tuple[Any, ...]:
    # [FUNCTION DOC] Inverse of filter_out_const_values: builds a tuple with one slot per mask
    # entry; where the mask is True takes the value at the same position in values, otherwise
    # takes the next unused element of tup. Raises IndexError if tup or values is too short.
    """
    masks and values are of same length. For indices where the mask is True, use
    the const_values to fill in.
    """
    out = []
    idx = 0
    for mask_idx, mask in enumerate(masks):
        if mask:
            out.append(values[mask_idx])
        else:
            out.append(tup[idx])
            idx += 1
    return tuple(out)
