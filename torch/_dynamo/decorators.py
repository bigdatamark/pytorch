# [MODULE DOC]
# Public decorators and context managers controlling Dynamo tracing: run,
# disable, skip, set_stance, allow_in_graph, disallow_in_graph, forbid_in_graph,
# nonstrict_trace, leaf_function, substitute_in_graph (polyfills), dimension
# markers (mark_dynamic, mark_unbacked, maybe_mark_dynamic, mark_static,
# mark_static_address), traceable config patching (patch_dynamo_config,
# dont_skip_tracing, disable_nested_graph_breaks), error_on_graph_break,
# override_cudagraphs, override_optimization_hint, allow_c_slot.
# Import-time side effects: copies public names of the C eval_frame extension
# into module globals; marks justknobs_check as a Dynamo constant; registers the
# code of the nonrecursive disable wrapper as skipped; validates the
# allowed-config-patch names (AssertionError if one is missing from config);
# registers an einops module-init hook with trace_rules.
"""
This module provides decorators and utilities for controlling TorchDynamo's behavior during compilation.
"""

import functools
import inspect
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, overload, Protocol, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch._custom_class_base import CustomClassBase
from torch._vendor.packaging.version import InvalidVersion, Version
from torch.compiler import is_compiling
from torch.utils._contextlib import _DecoratorContextManager
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._typing_utils import not_none

from .._utils_internal import justknobs_check
from . import trace_rules, variables
from .comptime import comptime
from .eval_frame import (
    _set_stance,
    DisableContext,
    DynamoStance,
    innermost_fn,
    RunOnlyContext,
    skip_code,
    StanceStr,
)
from .external_utils import (
    get_nonrecursive_disable_wrapper,
    wrap_dunder_call_ctx_manager,
)
from .utils import (
    _get_cudagraph_override,
    _get_error_on_graph_break,
    _set_cudagraph_override,
    _set_error_on_graph_break,
    allow_lru_cache_wrapper_trace_without_warning,
    is_function,
    is_lru_cache_wrapped_function,
)


justknobs_check._dynamo_marked_constant = True  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from types import FunctionType

    from torch._C._dynamo.eval_frame import (  # noqa: F401
        reset_code,
        set_eval_frame,
        set_guard_complete_hook,
        set_guard_error_hook,
        unsupported,
    )

    from .variables import VariableTracker
else:
    for name in dir(torch._C._dynamo.eval_frame):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(torch._C._dynamo.eval_frame, name)


_P = ParamSpec("_P")
_R = TypeVar("_R")
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def run(fn: Callable[_P, _R] | None = None) -> Any:
    # [FUNCTION DOC] Run-only mode: with fn, returns fn (innermost unwrapped)
    # wrapped by RunOnlyContext so only existing cached compilations are used and
    # no new compilation happens; without fn, returns the context object for use
    # as decorator/context manager. Raises AssertionError if fn is not callable.
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        if not callable(fn):
            raise AssertionError("fn must be callable")
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None, recursive=True, *, reason=None, wrapping=True):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Disables Dynamo for fn. Recursive mode returns a
    # DisableContext applied to fn (or the context itself if fn is None); callee
    # frames are skipped too. Non-recursive mode returns a wrapper tagged with
    # disable attributes (reason, original callable, recursive=False) that skips
    # only fn's own frame; with fn None it returns a decorator. reason is shown
    # when tracing reaches the function. Raises AssertionError for non-callables.
    """
    Decorator to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.

    If reason is provided, it will be printed when Dynamo attempts to trace the disabled function.
    """
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            if not callable(fn):
                raise AssertionError("fn must be callable")
            return DisableContext(msg=reason, wrapping=wrapping)(fn)
        return DisableContext(msg=reason, wrapping=wrapping)
    else:

        def wrap(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            # [FUNCTION DOC] HELPER: Builds the non-recursive disable wrapper for
            # fn and sets the disable marker attributes on it; returns it.
            # Raises AssertionError if fn is not callable.
            fn = innermost_fn(fn)
            if not callable(fn):
                raise AssertionError("fn must be callable")

            nonrecursive_disable_wrapper = get_nonrecursive_disable_wrapper(fn)
            nonrecursive_disable_wrapper._torchdynamo_disable = True  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_disable_msg = reason  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_wrapper_id = id(  # type: ignore[attr-defined]
                nonrecursive_disable_wrapper
            )
            nonrecursive_disable_wrapper._torchdynamo_disable_recursive = False  # type: ignore[attr-defined]
            return nonrecursive_disable_wrapper

        if fn is None:
            return wrap
        return wrap(fn)


_nonrecursive_disable_wrapper_code = disable(lambda: None, recursive=False).__code__  # type: ignore[attr-defined]
skip_code(_nonrecursive_disable_wrapper_code)


def skip(fn: Callable[_P, _R] | None = None) -> Callable[..., Any]:
    # [FUNCTION DOC] Marks fn's code object as skipped (frame not compiled, but
    # callees still processed) and tags fn as disabled; mutates fn and returns
    # it. With no fn, returns skip itself so it works as a decorator factory.
    # Raises AssertionError if fn is not callable.
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    if not callable(fn):
        raise AssertionError("fn must be callable")
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True  # type: ignore[attr-defined]
    return fn


class set_stance(_DecoratorContextManager):
    # [CLASS DOC] Sets Dynamo's global compiler stance. State: stance (a
    # DynamoStance value) and prev (the stance in effect at construction).
    # Construction applies the stance immediately (so a plain call works as a
    # function); as a context manager it applies on enter and restores prev on
    # exit; as a decorator it restores prev right away and re-applies per call.
    # Marked forbidden in graph.
    """
    Decorator, context manager, function to set the current stance of the compiler.

    Stances documented in corresponding function in torch/compiler/__init__.py
    """

    _dynamo_forbidden = True

    def __init__(
        self,
        stance: StanceStr = "default",
        *,
        skip_guard_eval_unsafe: bool = False,
        force_backend: str | Callable[..., Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds the DynamoStance and sets it globally, saving the
        # previous stance in prev. Raises RuntimeError if force_backend is given
        # with a non-default stance.
        if force_backend is not None and stance != "default":
            raise RuntimeError("non-default stance cannot have force_backend set")

        self.stance = DynamoStance(stance, skip_guard_eval_unsafe, force_backend)
        self.prev = _set_stance(self.stance)

    def __call__(self, fn: F) -> F:
        # [FUNCTION DOC] Decorator use: undoes the stance set in the constructor
        # (restores prev), then wraps fn via the base class so the stance applies
        # per call; marks the wrapper as forbidden in graph and returns it.
        _set_stance(self.prev)
        wrapper = super().__call__(fn)
        # forbid wrapper in graph
        wrapper._dynamo_forbidden = True  # type: ignore[attr-defined]
        return wrapper

    def __enter__(self) -> None:
        # [FUNCTION DOC] Applies this stance globally.
        _set_stance(self.stance)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Restores the stance saved in prev; exceptions are not
        # suppressed.
        _set_stance(self.prev)

    def clone(self) -> "set_stance":
        # [FUNCTION DOC] Creates a new set_stance with the same stance name and
        # backend (which applies it globally as a side effect). The
        # skip_guard_eval_unsafe flag is not carried over.
        # KNOWN ISSUE: skip_guard_eval_unsafe is dropped, so the clone may differ
        # from the original.
        return self.__class__(self.stance.stance, force_backend=self.stance.backend)


def assume_constant_result(fn):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Marks fn (mutating it) so Dynamo treats its call result as a
    # compile-time constant; returns fn.
    fn._dynamo_marked_constant = True  # type: ignore[attr-defined]
    return fn


def allow_in_graph(fn):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Registers a callable (or each in a list/tuple, returning a
    # list) in trace_rules so Dynamo places it directly in the graph. Moves the id
    # from the disallowed to the allowed set unless already treated as in-graph,
    # and removes the id again when fn is garbage collected. Returns fn.
    # Raises AssertionError if fn is not callable.
    """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    See :func:`torch.compiler.allow_in_graph`'s docstring for the full documentation

    WARNING: this API can be a footgun, please read the documentation carefully.
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    if not callable(fn):
        raise AssertionError("allow_in_graph expects a callable")
    if trace_rules.lookup_callable(fn) != variables.TorchInGraphFunctionVariable:
        fn_id = id(fn)
        trace_rules._disallowed_callable_ids.remove(fn_id)
        trace_rules._allowed_callable_ids.add(fn_id)

        # Avoid id reuse which creates subtle bugs.
        def deregister() -> None:
            # [FUNCTION DOC] HELPER: Finalizer removing fn's id from the allowed
            # set.
            trace_rules._allowed_callable_ids.remove(fn_id)

        weakref.finalize(fn, deregister)
    return fn


# pyrefly: ignore [implicit-any]
def _check_mutually_exclusive_decorators(fn: Callable, decorator_name: str) -> None:
    # [FUNCTION DOC] HELPER: Raises ValueError if fn is already marked with the
    # other of leaf_function / nonstrict_trace than decorator_name.
    mutually_exclusive = {
        "leaf_function": trace_rules.is_leaf_function,
        "nonstrict_trace": trace_rules.is_nonstrict_trace_callable,
    }

    for other_name, check_fn in mutually_exclusive.items():
        if other_name != decorator_name and check_fn(fn):
            first, second = sorted([decorator_name, other_name])
            raise ValueError(
                f"Function {fn} cannot be both marked as @{first} and "
                f"@{second}. Please use only one decorator."
            )


def nonstrict_trace(traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Returns a wrapper of traceable_fn whose id is registered as
    # allowed in graph and as nonstrict-traced (deregistered on garbage
    # collection). Raises AssertionError for non-callables and ValueError if the
    # function is already a leaf_function.
    """
    Decorator to mark a function as nonstrict-traceable for dynamo.

    See :func:`torch.compiler.nonstrict_trace` for the full documentation.
    """
    if not callable(traceable_fn):
        raise AssertionError("nonstrict_trace expects a callable")

    _check_mutually_exclusive_decorators(traceable_fn, "nonstrict_trace")

    @functools.wraps(traceable_fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Forwards the call to traceable_fn.
        return traceable_fn(*args, **kwargs)

    wrapped_id = id(wrapped)

    # This line allows us to reuse much of the `allow_in_graph` impl.
    trace_rules._allowed_callable_ids.add(wrapped_id)

    # This line allows us to diverge the impl from `allow_in_graph`.
    trace_rules._nonstrict_trace_callable_ids.add(wrapped_id)

    # Avoid id reuse which creates subtle bugs.
    def deregister() -> None:
        # [FUNCTION DOC] HELPER: Finalizer removing the wrapper id from the
        # allowed and nonstrict-trace sets.
        trace_rules._allowed_callable_ids.remove(wrapped_id)
        trace_rules._nonstrict_trace_callable_ids.remove(wrapped_id)

    weakref.finalize(wrapped, deregister)

    return wrapped


def _invoke_leaf_function_python(
    real_impl: Callable[..., Any],
    fake_impl: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    mutates_args: frozenset[str] | None = None,
    hook_fn: Callable[..., Any] | None = None,
    hook_fake_fn: Callable[..., Any] | None = None,
) -> Any:
    # [FUNCTION DOC] HELPER: Executes a leaf function through the
    # invoke_leaf_function higher-order op without Dynamo. Collects nn.Module
    # arguments (deduplicated), registers them for make_fx, replaces them by
    # state, flattens inputs, wraps real/fake implementations (recording the
    # output tree spec), optionally attaches the backward hook pair and resolves
    # mutated argument indices from mutates_args, then returns the unflattened
    # outputs. Raises AssertionError if the output spec was never captured.
    """Call invoke_leaf_function HOP directly from Python.

    This enables @leaf_function to work with make_fx
    without relying on Dynamo to intercept the call.
    """
    from torch._higher_order_ops.invoke_leaf_function import (
        _LeafCallable,
        convert_modules_to_states,
        invoke_leaf_function,
        make_leaf_function_wrappers,
        store_makefx_modules,
    )

    captured_modules: list[torch.nn.Module] = []
    seen_module_ids: dict[int, int] = {}  # id(module) -> position in captured_modules
    for val in pytree.tree_flatten(
        (args, kwargs), is_leaf=lambda x: isinstance(x, torch.nn.Module)
    )[0]:
        if isinstance(val, torch.nn.Module) and id(val) not in seen_module_ids:
            seen_module_ids[id(val)] = len(captured_modules)
            captured_modules.append(val)

    global_indices = store_makefx_modules(captured_modules)
    module_to_index = {
        mod_id: global_indices[pos] for mod_id, pos in seen_module_ids.items()
    }

    processed = convert_modules_to_states((args, kwargs), module_to_index)
    flat_args, input_spec = pytree.tree_flatten(processed)

    # Single-element mutable list so the wrappers can write back the output
    # TreeSpec. Read captured_out_spec[0] after the wrappers have been called.
    captured_out_spec: list[pytree.TreeSpec | None] = [None]
    wrapped_real, wrapped_fake = make_leaf_function_wrappers(
        real_impl, fake_impl, captured_out_spec
    )

    real_fn_callable = _LeafCallable(wrapped_real)
    fake_fn_callable = _LeafCallable(wrapped_fake)

    if hook_fn is not None:
        real_fn_callable._leaf_hook_real_fn = hook_fn  # type: ignore[attr-defined]
        real_fn_callable._leaf_hook_fake_fn = hook_fake_fn  # type: ignore[attr-defined]

    mutated_flat_indices = ""
    if mutates_args:
        from torch._higher_order_ops.invoke_leaf_function import (
            _resolve_mutated_flat_indices,
        )

        mutated_flat_indices = _resolve_mutated_flat_indices(
            real_impl, mutates_args, len(flat_args), input_spec
        )

    flat_out = invoke_leaf_function(
        real_fn_callable, fake_fn_callable, input_spec, mutated_flat_indices, *flat_args
    )

    if captured_out_spec[0] is None:
        raise AssertionError("captured_out_spec was not set by leaf function wrappers")
    return pytree.tree_unflatten(flat_out, captured_out_spec[0])


@overload
def leaf_function(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...


@overload
def leaf_function(
    *, mutates_args: set[str]
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def leaf_function(
    fn: Callable[_P, _R] | None = None, *, mutates_args: set[str] | None = None
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # [FUNCTION DOC] Decorator marking fn as an opaque leaf function. If fn is
    # None, returns a partial applying mutates_args. Otherwise validates that fn
    # is not nonstrict_trace and that every mutates_args root name is a
    # parameter of fn (ValueError otherwise), then returns a wrapper that runs fn
    # eagerly through the invoke_leaf_function op. The wrapper carries attributes
    # for real/fake implementations, mutated args and hook functions, plus
    # register_fake and register_multi_grad_hook setters; calling it before a
    # fake is registered raises ValueError. Registers the wrapper id as allowed
    # and as a leaf function (deregistered on garbage collection).
    """
    Decorator to mark a function as a leaf function for :func:`torch.compile`.

    A leaf function appears as an opaque operation in the compiled graph. During
    compilation, Dynamo and AOT autograd do not trace into it. At runtime, the
    original eager Python code is executed directly.

    Quick Start:
        Suppose we want to log per-sample statistics during the forward pass::

            def compute_and_log_stats(x):
                stats = x.mean(dim=1)
                logger.info(f"Per-sample means: {stats}")
                return (stats,)

        With ``torch.compile(..., fullgraph=True)``, this fails because
        ``logger.info`` causes a graph break. To fix it, follow the steps below:

        1. Decorate your function with ``@leaf_function``
        2. Define a shape-inference function using ``@your_fn.register_fake``

        Concrete implementation::

            >>> import logging
            >>> import torch
            >>> from torch._dynamo.decorators import leaf_function
            >>>
            >>> logging.basicConfig(level=logging.INFO)
            >>> logger = logging.getLogger(__name__)
            >>>
            >>> @leaf_function
            ... def compute_and_log_stats(x):
            ...     stats = x.mean(dim=1)  # Shape: (x.shape[0],)
            ...     logger.info(f"Per-sample means: {stats}")
            ...     return (stats,)
            ...
            >>> @compute_and_log_stats.register_fake
            ... def compute_and_log_stats_fake(x):
            ...     # Match the output shape: (x.shape[0],)
            ...     return (x.new_empty(x.shape[0]),)
            ...
            >>> def fn(x):
            ...     return compute_and_log_stats(x)
            ...
            >>> x = torch.randn(3, 4)
            >>> compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
            >>> out = compiled_fn(x)[0]

    When to Use:
        Use ``leaf_function`` when you need eager execution semantics that tracing
        cannot preserve:

        - **Non-traceable code**: The function consists of code that Dynamo or
          AOT Autograd cannot trace.
        - **Runtime side effects**: Operations that must happen exactly once per call
          at runtime (e.g., logging, external library calls).

        If your function has a static computation graph and no runtime side effects,
        prefer :func:`allow_in_graph` or :func:`nonstrict_trace` instead. They allow AOT autograd
        to trace through and potentially optimize the code.

    Usage:
        **Inputs and Outputs**:
        - Both inputs and outputs must use pytree-compatible types.
        Tensors, Python primitives (int, float, bool, str), and built-in containers
        (list, tuple, dict) are supported by default.

        Note: We recommend leaf_functions only accept and return tensors. Though primitive
        types (int, float, bool, str) are supported in inputs and outputs, they may cause
        surprising behavior: primitive inputs are guarded and may trigger recompilation,
        while primitive output values are captured from the fake implementation at compile time and
        remain fixed for all subsequent executions. Even though the real function runs
        at runtime, its primitive return values are silently replaced with the compile-time
        values from the fake implementation.

        Example::

            # BAD: primitive output varies at runtime
            counter = 0


            @leaf_function
            def count_calls(x):
                global counter
                counter += 1
                return (x, counter)


            @count_calls.register_fake
            def count_calls_fake(x):
                return (x, 999)  # placeholder value


            # At runtime: real function runs (counter increments to 1, 2, 3, ...)
            # But returned count is always 999 (from fake implementation at compile time)

        - User-defined classes must be registered via :func:`torch.utils._pytree.register_pytree_node`
        or :func:`torch.utils._pytree.register_dataclass`.

        - :class:`torch.nn.Module` can also be passed as input; its parameters and buffers
        are tracked for autograd. The module must exist outside the compile region.
        Running the module multiple times must not modify its parameters, buffers, or attributes.

        **register_fake (required)**:
        Since the function body is not traced, you must
        provide a shape-inference function via ``@fn.register_fake``. It runs at compile
        time with FakeTensor inputs (tensors with no data, only metadata) and must
        satisfy the following requirements:

        - Must have the same input and output signature (e.g., same pytree structure, same tensor metadata
          such as shapes, dtypes, device, strides) as the real function
        - Must be runnable with FakeTensor inputs
        - Must only use its explicit arguments (no closures over tensors or modules)

        Note: The input and output signature must be determinable at compile time.
        If your function's output structure depends on runtime values, adjust the leaf function
        boundary until outputs become predictable.

        To validate that your fake implementation matches the real function's outputs, set
        ``torch._dynamo.config.leaf_function_validate_outputs = True``.

        **register_multi_grad_hook (optional)**:
        You can register a backward hook via ``@fn.register_multi_grad_hook``
        to run code when gradients have been computed for all requires_grad
        tensor inputs during backward. The hook fires exactly once per
        backward pass. The hook receives only the gradients of the leaf
        function's requires_grad tensor inputs, in the order they appeared;
        non-tensor arguments and tensors without ``requires_grad`` are not
        forwarded to the hook. The hook must return ``None``. The hook is
        called as a leaf function itself, so it is also opaque to the
        compiler.

        Example::

            >>> @leaf_function
            ... def debug_log(t):
            ...     return None
            ...
            >>> @debug_log.register_fake
            ... def debug_log_fake(t):
            ...     return None
            ...
            >>> @debug_log.register_multi_grad_hook
            ... def debug_log_hook(t_grad):
            ...     print(f"[bwd] norm={t_grad.norm().item()}")
            ...
            >>> x = torch.randn(4, requires_grad=True)
            >>> debug_log(x)  # no assignment needed
            >>> (x * 2).sum().backward()
            [bwd] norm=...

    Limitations:
        Currently, inductor backend and :func:`torch.export.export` are not yet supported.

    Training / Autograd:
        Training is supported automatically if your leaf function is differentiable
        in eager mode (e.g., it's implemented with PyTorch ops, :class:`torch.autograd.Function`,
        or differentiable custom ops).

        **Restriction**: Calling ``.backward()`` *inside* the leaf function is not
        supported.

        **Escaped gradients check**: If the leaf function closes over a tensor with
        ``requires_grad=True``, gradients will not flow back to it (only explicit inputs
        receive gradients). To detect such cases, set
        ``torch._dynamo.config.leaf_function_check_escaped_gradients = True``.
        When enabled, a ``RuntimeError`` is raised with details about the escaped tensors.

        Internally, inputs and outputs are detached from the outer autograd graph at the
        leaf function boundary. The leaf function builds its own local autograd
        graph. In backward, gradients propagate through this local graph to the
        leaf function's inputs. If a :class:`torch.nn.Module` is passed in as input, the gradients
        will also flow back to its parameters and buffers.

    How leaf_function Works:
        Understanding this helps avoid pitfalls:

        1. **Compilation**: Dynamo and AOT autograd do not trace into the leaf function.
           Only your fake implementation runs during compilation to determine output signatures,
           shapes, and dtypes. The real function runs at runtime as eager Python.

        2. **Isolation**: Mutations to shared state (globals, closures) may not be
           visible across the boundary between leaf functions and compiled regions. Pass data
           explicitly as function arguments and return results as outputs.

    Dangerous Patterns:
        These patterns may cause silent incorrectness or errors:

        - **Side effects between leaf functions and compiled regions**: Mutations to
          Python state inside a leaf function may not be visible to the compiled
          region, and vice versa. The compiled graph may reorder or eliminate
          operations in ways that break assumptions about side effect ordering.

          Bad::

            # Compiled region
            self.counter += 1
            y = my_leaf_fn(self, x)  # Don't depend on self.counter inside leaf_fn

            y = my_leaf_fn(self, x)
            result = self.state  # Don't depend on state mutated by leaf_fn

          Logging/printing inside the leaf function is fine since it doesn't affect
          correctness.

        - **In-place mutations on inputs**: Undeclared in-place mutations on input
          tensors are detected and will raise an error. Either declare mutations via
          ``mutates_args`` or clone inputs before mutating.

          Bad::

            @leaf_function
            def my_leaf_fn(x):
                x.add_(1)  # Will raise: "Undeclared in-place mutation"
                return (x,)

          Good::

            @leaf_function(mutates_args={"buf"})
            def my_leaf_fn(x, buf):
                buf.add_(1)  # OK: declared in mutates_args
                return (x + buf,)

          For pytree args (lists, tuples, dicts of tensors), use the parameter name
          to declare mutation on all contained tensors::

            @leaf_function(mutates_args={"buffers"})
            def my_leaf_fn(x, buffers):
                for buf in buffers:
                    buf.add_(1)  # OK: 'buffers' declared in mutates_args
                return (x + sum(buffers),)

          Or use bracket notation for fine-grained control::

            @leaf_function(mutates_args={"buffers[0]"})
            def my_leaf_fn(x, buffers):
                buffers[0].add_(1)  # OK: 'buffers[0]' declared
                return (x + buffers[1],)  # buffers[1] is not cloned

        - **Closures in fake implementation**: Tensors or modules captured from enclosing scopes
          in the fake implementation will cause compilation errors. The real function can
          close over them if they don't require gradient, but the fake implementation must
          only use its arguments.

          Bad::

            weight = torch.randn(3, 3)


            @leaf_function
            def my_leaf_fn(x):
                return (x @ weight,)


            @my_leaf_fn.register_fake
            def my_leaf_fn_fake(x):
                return (x @ weight,)  # Error: weight used in fake implementation

          Good::

            weight = torch.randn(3, 3)


            @leaf_function
            def my_leaf_fn(x):
                return (x @ weight,)  # OK: real function can use closure


            @my_leaf_fn.register_fake
            def my_leaf_fn_fake(x):
                return (x @ torch.empty_like(x),)  # OK: uses only args

    Example::
        Wrapping a linear function with logging as a leaf_function. Gradients
        flow back because the inner operations are differentiable::

            >>> import torch
            >>> import torch.nn.functional as F
            >>> from torch._dynamo.decorators import leaf_function
            >>>
            >>> @leaf_function
            ... def custom_forward(linear, x):
            ...     # Logging at runtime
            ...     print(f"Input: shape={x.shape}, mean={x.mean().item():.4f}")
            ...     out = F.linear(x, linear.weight, linear.bias)
            ...     print(f"Output: shape={out.shape}, norm={out.norm().item():.4f}")
            ...     return (out,)
            ...
            >>> @custom_forward.register_fake
            ... def custom_forward_fake(linear, x):
            ...     # Return tensor with correct shape: (batch, out_features)
            ...     return (x.new_empty(x.shape[0], linear.weight.shape[0]),)
            ...
            >>> class MyModel(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 20)
            ...     def forward(self, x):
            ...         return custom_forward(self.linear, x)
            ...
            >>> model = MyModel()
            >>> compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            >>> x = torch.randn(32, 10, requires_grad=True)
            >>> out = compiled(x)[0]
            >>> out.sum().backward()  # Gradients flow to model.linear.weight/bias and x

    Args:
        fn: The function being decorated.
        mutates_args: Set of Python expressions (as strings) identifying arguments
            that the function mutates in-place. Each string is evaluated against
            the function's parameters. Examples: ``'buf'`` for a plain tensor,
            ``'model.running_mean'`` for an nn.Module buffer,
            ``'buffers'`` to mark all tensors in a list, ``'buffers[0]'`` for a
            specific element, ``'state["key"]'`` for a dict entry.
    """
    if fn is None:
        return functools.partial(leaf_function, mutates_args=mutates_args)

    from . import trace_rules

    _check_mutually_exclusive_decorators(fn, "leaf_function")

    if mutates_args:
        import inspect
        import re

        params = set(inspect.signature(fn).parameters)
        for expr in mutates_args:
            root = re.split(r"[.\[]", expr, maxsplit=1)[0]
            if root not in params:
                raise ValueError(
                    f"mutates_args expression '{expr}' refers to parameter '{root}', "
                    f"which is not a parameter of '{fn.__name__}'. "
                    f"Available parameters: {', '.join(params)}"
                )

    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Eager entry point of the leaf function. Raises
        # ValueError if no fake implementation is registered; otherwise invokes
        # the real function via the leaf-function op and returns its outputs.
        if inner._torchdynamo_leaf_fake_fn is None:  # type: ignore[attr-defined]
            raise ValueError(
                f"leaf_function '{getattr(fn, '__name__', fn)}' "
                "requires a fake implementation. Please provide one using the @<func>.register_fake "
                "decorator. See the leaf_function docstring for details."
            )

        return _invoke_leaf_function_python(
            fn,  # pyrefly: ignore [bad-argument-type]
            # pyrefly: ignore [bad-argument-type]
            inner._torchdynamo_leaf_fake_fn,
            args,
            kwargs,
            mutates_args=inner._torchdynamo_leaf_mutates_args,  # pyrefly: ignore [missing-attribute]
            hook_fn=inner._torchdynamo_leaf_hook_fn,  # type: ignore[attr-defined]
            hook_fake_fn=inner._torchdynamo_leaf_hook_fake_fn,  # type: ignore[attr-defined]
        )  # type: ignore[attr-defined]

    inner._torchdynamo_leaf_real_fn = fn  # type: ignore[attr-defined]
    inner._torchdynamo_leaf_fake_fn = None  # type: ignore[attr-defined]
    inner._torchdynamo_leaf_mutates_args = (  # pyrefly: ignore [missing-attribute]
        frozenset(mutates_args) if mutates_args else frozenset()
    )  # type: ignore[attr-defined]
    inner._torchdynamo_leaf_hook_fn = None  # type: ignore[attr-defined]
    inner._torchdynamo_leaf_hook_fake_fn = None  # type: ignore[attr-defined]

    # Follow nonstrict_trace implementation
    wrapped_id = id(inner)
    trace_rules._allowed_callable_ids.add(wrapped_id)
    trace_rules._leaf_function_ids.add(wrapped_id)

    def deregister() -> None:
        # [FUNCTION DOC] HELPER: Finalizer removing the wrapper id from the
        # allowed and leaf-function sets.
        trace_rules._allowed_callable_ids.remove(wrapped_id)
        trace_rules._leaf_function_ids.remove(wrapped_id)

    weakref.finalize(inner, deregister)

    def register_fake_setter(fake_fn: Callable[..., Any]) -> Callable[..., Any]:
        # [FUNCTION DOC] HELPER: Stores fake_fn as the fake implementation and
        # returns the leaf function wrapper (not fake_fn).
        inner._torchdynamo_leaf_fake_fn = fake_fn  # type: ignore[attr-defined]
        return inner

    inner.register_fake = register_fake_setter  # type: ignore[attr-defined]

    def register_hook_setter(hook_fn: Callable[..., Any]) -> Callable[..., Any]:
        # [FUNCTION DOC] HELPER: Stores hook_fn as the multi-grad backward hook,
        # sets a no-op fake hook returning None, and returns the leaf function
        # wrapper.
        inner._torchdynamo_leaf_hook_fn = hook_fn  # type: ignore[attr-defined]
        inner._torchdynamo_leaf_hook_fake_fn = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        return inner

    inner.register_multi_grad_hook = register_hook_setter  # type: ignore[attr-defined]

    return inner


def _disallow_in_graph_helper(throw_if_not_allowed: bool) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Returns a decorator that moves a callable's id from
    # the allowed and nonstrict-trace sets into the disallowed set (forcing a
    # graph break). When throw_if_not_allowed is true, raises RuntimeError for
    # callables that are not already in-graph torch callables.
    def inner(fn: Any) -> Any:
        # [FUNCTION DOC] HELPER: Applies the disallow registration to fn (or each
        # element of a list/tuple, returning a list) and returns fn. Raises
        # AssertionError for non-callables.
        if isinstance(fn, (list, tuple)):
            return [disallow_in_graph(x) for x in fn]
        if not callable(fn):
            raise AssertionError("disallow_in_graph expects a callable")
        if (
            throw_if_not_allowed
            and trace_rules.lookup_callable(fn)
            != variables.TorchInGraphFunctionVariable
            and trace_rules.lookup(fn) != variables.TorchInGraphFunctionVariable
        ):
            raise RuntimeError(
                "disallow_in_graph is expected to be used on an already allowed callable (like torch.* ops). "
                "Allowed callables means callables that TorchDynamo puts as-is in the extracted graph."
            )
        trace_rules._allowed_callable_ids.remove(id(fn))
        trace_rules._nonstrict_trace_callable_ids.remove(id(fn))
        trace_rules._disallowed_callable_ids.add(id(fn))
        return fn

    return inner


def disallow_in_graph(fn: Callable[..., Any]) -> Any:
    # [FUNCTION DOC] Makes Dynamo graph-break on fn (an already allowed torch
    # callable, or a list/tuple of them). Returns fn; raises RuntimeError if fn
    # was not in-graph. Mutates global trace_rules id sets.
    """
    Customize which functions TorchDynamo will exclude in the generated
    graph and force a graph break on.
    ::

        torch._dynamo.disallow_in_graph(torch.sub)


        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will break the graph on `torch.sub`, and give two graphs each with a
    single `torch.add()` op.
    """
    return _disallow_in_graph_helper(throw_if_not_allowed=True)(fn)


@_disallow_in_graph_helper(throw_if_not_allowed=False)
def graph_break(msg: str = "") -> None:
    # [FUNCTION DOC] Marker that forces a graph break when traced by Dynamo
    # (registered as disallowed). Does nothing when run eagerly; msg is unused.
    """Force a graph break"""


# NOTE: primarily used for internal debugging purposes!
@_disallow_in_graph_helper(throw_if_not_allowed=False)
def skip_frame(msg: str = "") -> None:
    # [FUNCTION DOC] Internal debugging marker that makes Dynamo skip the
    # current frame when traced. Does nothing when run eagerly; msg is unused.
    """Force a skipped frame"""


@_disallow_in_graph_helper(throw_if_not_allowed=False)
def step_unsupported(msg: str = "") -> None:
    # [FUNCTION DOC] Marker that makes Dynamo compile the graph traced so far and
    # skip the rest of the frame when traced. Does nothing when run eagerly.
    """Force a step unsupported graph break, which results in compiling
    the traced FX graph so far, then skipping the rest of the frame.
    In order to get expected behavior, there should be at least 2 ops
    and a part of the code not contained in any try/with blocks."""


def forbid_in_graph(fn: Any) -> Any:
    # [FUNCTION DOC] Marks fn (or each element of a list/tuple, returning a
    # list) with a forbidden attribute so Dynamo asserts if it is traced.
    # Mutates fn and returns it. Raises AssertionError for non-callables.
    """
    Customize which functions TorchDynamo will assert are not present while tracing.

    If you want a graph break on this function instead, use disallow_in_graph.
    TODO(voz): We now have allow_in_graph, disallow_in_graph, forbid_in_graph - some more robust
    documentation would not be amiss.
    """
    if isinstance(fn, (list, tuple)):
        return [forbid_in_graph(x) for x in fn]
    if not callable(fn):
        raise AssertionError("forbid_in_graph applies only to callables")
    # pyrefly: ignore [missing-attribute]
    fn._dynamo_forbidden = True
    return fn


def substitute_in_graph(
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = False,
    skip_signature_check: bool = False,
    # type that is embedded in the Python interpreter
    is_embedded_type: bool = False,  # internal use only
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # [FUNCTION DOC] Returns a decorator that registers the decorated function
    # as the polyfill used when Dynamo inlines original_fn. Raises TypeError if
    # original_fn (or the polyfill) is not a function (a class is allowed when
    # is_embedded_type). For embedded itertools types, marks them polyfilled.
    # The decorator validates signatures unless skipped, checks for duplicate
    # registrations (ValueError), then registers dispatch, rule-map and handler
    # entries globally and returns a wrapper that calls original_fn eagerly.
    """
    Register a polyfill handler for a function, usually a C function from the C extension, to be
    used in place of the original function when inlining the original function in the graph.

    .. note::

        The polyfill handler is only used when inlining the original function. It is not used when
        the original function is called directly. In the eager mode, the decorated function calls
        the performant C function rather than the polyfill handler.

    The polyfill handler is a function that will be called in place of the original function when
    inlining the original function. The polyfill handler should have the same signature and the same
    behavior as the original function.

    Args:
        original_fn (callable): The original function, usually a C function, to register a polyfill
            handler for.
        can_constant_fold_through (bool, optional): Whether the polyfill handler can be constant
            folded through. That is, if the polyfill handler is a pure function and its arguments
            are constant, the result of the polyfill handler can be constant folded during the
            compilation. Defaults to ``False``.
        skip_signature_check (bool, optional): Whether to skip the signature check between the
            original function and the polyfill handler. Defaults to ``False``.

    Returns:
        A decorator that registers the polyfill handler for the original function.

    Example::

        >>> # xdoctest: +SKIP("conflict with the tests: duplicate polyfill handlers")
        >>> import operator
        >>> operator.indexOf([1, 2, 3, 4, 5], 3)
        2
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        Traceback (most recent call last):
        ...
        torch._dynamo.exc.Unsupported: ...

        >>> @torch.compiler.substitute_in_graph(operator.indexOf)
        ... def indexOf(a, b, /):
        ...     for i, item in enumerate(a):
        ...         if item is b or item == b:
        ...             return i
        ...     raise ValueError("sequence.index(x): x not in sequence")
        >>>
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        2
    """
    if not is_function(original_fn) and not (
        is_embedded_type and inspect.isclass(original_fn)
    ):
        raise TypeError(
            f"substitute_in_graph expects a function but got {type(original_fn)!r}"
        )
    if is_embedded_type:
        if not inspect.isclass(original_fn):
            raise TypeError(
                f"substitute_in_graph expects a class but got {type(original_fn)!r}"
            )

        from .variables.builder import ITERTOOLS_POLYFILLED_TYPE_IDS, ITERTOOLS_TYPE_IDS

        if id(original_fn) in ITERTOOLS_TYPE_IDS:
            ITERTOOLS_POLYFILLED_TYPE_IDS.add(id(original_fn))

    def wrapper(traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]:
        # [FUNCTION DOC] HELPER: Performs the registration of traceable_fn as the
        # polyfill for original_fn; returns the eager wrapper carrying polyfill
        # attributes. Raises TypeError on signature mismatch or non-function,
        # ValueError on duplicate registration.
        if not is_function(traceable_fn):
            raise TypeError(
                f"@substitute_in_graph(...) expects a function but got {type(traceable_fn)!r}"
            )

        if not skip_signature_check:
            try:
                original_sig = inspect.signature(original_fn)
            except ValueError:
                pass
            else:
                traceable_sig = inspect.signature(traceable_fn)

                def sig_ident(
                    sig: inspect.Signature,
                ) -> tuple[tuple[str, ...], set[str], dict[str, Any]]:
                    # [FUNCTION DOC] HELPER: Returns a comparable signature key:
                    # names of non-keyword, non-variadic parameters, the set of
                    # keyword-only names, and a mapping of names to defaults.
                    # Ignore annotations for parameters and return type
                    return (
                        tuple(
                            p.name
                            for p in sig.parameters.values()
                            if (
                                p.kind
                                not in {
                                    p.KEYWORD_ONLY,
                                    # the name of *args and **kwargs is not important
                                    p.VAR_POSITIONAL,
                                    p.VAR_KEYWORD,
                                }
                            )
                        ),
                        {
                            p.name
                            for p in sig.parameters.values()
                            if p.kind == p.KEYWORD_ONLY
                        },
                        {
                            p.name: p.default
                            for p in sig.parameters.values()
                            # the name of *args and **kwargs is not important
                            if p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
                        },
                    )

                wildcard_sig = inspect.signature(lambda *args, **kwargs: None)

                if (
                    sig_ident(original_sig) != sig_ident(traceable_sig)
                    and sig_ident(original_sig) != sig_ident(wildcard_sig)
                    and sig_ident(traceable_sig) != sig_ident(wildcard_sig)
                ):
                    raise TypeError(
                        f"Signature mismatch between {original_fn} and {traceable_fn}: "
                        f"{original_sig} != {traceable_sig}"
                    )

        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.trace_rules import (
            _polyfilled_function_ids,
            get_torch_obj_rule_map,
        )
        from torch._dynamo.variables import PolyfilledFunctionVariable
        from torch._dynamo.variables.builder import VariableBuilder

        id_dispatch_map = VariableBuilder._id_dispatch()
        if id(original_fn) in id_dispatch_map:
            raise ValueError(
                f"Duplicate dispatch rule for {original_fn}: "
                "already registered in VariableBuilder's id dispatch map"
            )

        if id(original_fn) in _polyfilled_function_ids:
            raise ValueError(f"Duplicate polyfilled object {original_fn}")

        rule_map: dict[Any, type[VariableTracker]] = get_torch_obj_rule_map()
        if original_fn in rule_map:
            raise ValueError(
                f"Duplicate object {original_fn} with different rules: "
                f"{PolyfilledFunctionVariable}, {rule_map[original_fn]}"
            )

        polyfill_handlers: dict[Callable[..., Any], FunctionType]
        polyfill_handlers = PolyfilledFunctionVariable._get_polyfill_handlers()
        if original_fn in polyfill_handlers:
            raise ValueError(
                f"Duplicate polyfill handlers for {original_fn}: "
                f"already handled by {polyfill_handlers[original_fn]}"
            )

        # Need to wrap the function because we may cannot assign __torch_dynamo_polyfill__ to a
        # C++ function.
        @functools.wraps(traceable_fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # [FUNCTION DOC] HELPER: Calls the original function eagerly.
            return original_fn(*args, **kwargs)

        def dispatch_fn(
            self: VariableBuilder, value: Callable[_P, _R]
        ) -> PolyfilledFunctionVariable:
            # [FUNCTION DOC] HELPER: VariableBuilder dispatch handler: installs a
            # class/module/id match guard for value and returns a
            # PolyfilledFunctionVariable. Raises AssertionError if no guards.
            if inspect.isclass(value):
                guard_type = GuardBuilder.CLASS_MATCH
            elif inspect.ismodule(value):
                guard_type = GuardBuilder.MODULE_MATCH
            else:
                guard_type = GuardBuilder.ID_MATCH
            guards = self.install_guards(guard_type)
            if guards is None:
                raise AssertionError("install_guards returned None")
            return PolyfilledFunctionVariable(
                value,
                source=self.source,
                **guards,
            )

        id_dispatch_map[id(original_fn)] = id_dispatch_map[id(wrapped)] = dispatch_fn
        _polyfilled_function_ids.add(id(original_fn))
        _polyfilled_function_ids.add(id(wrapped))
        rule_map[original_fn] = rule_map[wrapped] = PolyfilledFunctionVariable
        polyfill_handlers[original_fn] = polyfill_handlers[wrapped] = wrapped  # type: ignore[assignment]

        wrapped.__torch_dynamo_original__ = original_fn  # type: ignore[attr-defined]
        wrapped.__torch_dynamo_polyfill__ = traceable_fn  # type: ignore[attr-defined]
        wrapped.__torch_dynamo_can_constant_fold_through__ = can_constant_fold_through  # type: ignore[attr-defined]

        return wrapped  # type: ignore[return-value]

    return wrapper


# Helper function to flatten a tensor subclass and apply a function to
# all inner tensors that match the outer dim. Used to reduce duplication
# across the various marking APIs.
def _apply_func_to_inner_tensors_of_same_dim(
    func: Callable[..., Any], t: object, *args: Any, **kwargs: Any
) -> None:
    # [FUNCTION DOC] HELPER: For a traceable wrapper tensor subclass t, calls
    # func on each inner tensor whose dimension count equals t's, passing the
    # extra arguments; non-tensor custom class attributes are ignored. Raises
    # AssertionError if t is not a traceable wrapper subclass tensor or an
    # attribute is neither tensor nor custom class.
    if not is_traceable_wrapper_subclass(t):
        raise AssertionError(f"Expected a traceable wrapper subclass, got {type(t)}")

    attrs, _ctx = t.__tensor_flatten__()
    if not isinstance(t, torch.Tensor):
        raise AssertionError(f"Expected a Tensor, got {type(t)}")
    for attr in attrs:
        match getattr(t, attr):
            case torch.Tensor() as inner:
                if inner.dim() == t.dim():
                    func(inner, *args, **kwargs)
            case CustomClassBase():
                pass
            case unexpected:
                raise AssertionError(
                    f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                )


@dataclass(frozen=True, slots=True)
class _DimRange:
    # [CLASS DOC] HELPER: Immutable record of a tensor dimension index and its
    # allowed min/max sizes (either may be None), stored on tensors by
    # mark_dynamic.
    """
    This represents a dimension of a tensor and the corresponding
    min and max values it can take.  Don't create this
    class directly; instead, use :func:`mark_dynamic`.
    """

    dim: int
    min: int
    max: int


@forbid_in_graph
def mark_unbacked(
    t: Any,
    index: int | list[Any] | tuple[Any],
    hint_override: int | None = None,
    strict: bool = False,
    specialize_on: list[Any] | None = None,
    shape_id: str | None = None,
    min: int | None = None,
    max: int | None = None,
) -> None:
    # [FUNCTION DOC] Marks dimension(s) index of tensor t as unbacked by setting
    # bookkeeping attributes on t (unbacked indices, hint overrides, bounds,
    # shape ids, specialization lists, or strict indices when strict). A list or
    # tuple index recurses per element (only shape_id, min and max are forwarded
    # then). DTensors are marked via their local tensor. Raises AssertionError
    # for traceable wrapper subclasses or invalid index types. Must be called
    # outside compiled code.
    # KNOWN ISSUE: the recursion and DTensor paths drop hint_override, strict and
    # specialize_on; also the hasattr check uses _specialized_on while the
    # attribute set is _specialize_on, so it is reset on every call.
    """
    Mark a tensor as having an unbacked dimension. This changes the semantics of operations:
    - The size of the specified dimension will always be reported as not equal to zero or one.
    - Assertions on this index will be turned into runtime asserts.
    - Attempting to get the real value of this dimension will raise an exception.
    - In effect, this dimension is treated as data-dependent (its value is unknown).

    Args:
        t (Any): The tensor to mark as having an unbacked dimension.
        index (int or list/tuple of int): The dimension(s) to mark as unbacked. Can be a single
            integer or a list/tuple of integers. Calls are additive: mark_unbacked(x, 0) followed
            by mark_unbacked(x, 1) marks both dims.
        hint_override (Optional[int], default=None): An optional integer to override the size hint for this dimension.
            This is only used by the inductor backend for size hint queries, such as during autotuning.
            NOTE: changing hint_override values will cause FxGraphCache misses, since hint overrides
            affect inductor codegen decisions and are included in the cache key via
            ShapeEnv.var_to_hint_override.
        strict (bool, default=False): If True, an error will be raised if the unbacked dimension is specialized.
            By default (strict=False), specialization is allowed and will proceed without error.
        specialize_on (Optional[list[Any]], default=None): A list of specialization criteria (e.g., lambdas) for this dimension.
            If provided, Dynamo will generate specialized compiled regions for each criterion in addition to a generic trace.
        shape_id (Optional[str], default=None): An optional identifier to group unbacked dimensions together.
            All unbacked dimensions with the same shape_id will share the same unbacked symbol. This is useful when multiple tensors
            are known to have the same batch size at runtime. A runtime assertion is added
            to ensure this property at runtime.
        min (Optional[int], default=None): Minimum value constraint for this dimension.
            If provided, a runtime check will be added to ensure the dimension is >= min.
        max (Optional[int], default=None): Maximum value constraint for this dimension.
            If provided, a runtime check will be added to ensure the dimension is <= max.
    """
    if torch.distributed.is_available() and isinstance(
        t, torch.distributed.tensor.DTensor
    ):
        # apply on inner tensor sizes/strides
        mark_unbacked(t._local_tensor, index, shape_id=shape_id)
    else:
        # You could have copied the mark_dynamic behavior but I'm not convinced
        # it's what you want
        if is_traceable_wrapper_subclass(t):
            raise AssertionError("not implemented yet")

    if isinstance(index, int):
        if strict:
            if not hasattr(t, "_dynamo_strict_unbacked_indices"):
                t._dynamo_strict_unbacked_indices = set()

            t._dynamo_strict_unbacked_indices.add(index)
            t._has_dynamo_dim_marking = True  # type: ignore[attr-defined]
            return

        if not hasattr(t, "_specialized_on"):
            # pyrefly: ignore [implicit-any]
            t._specialize_on = {}

        if not hasattr(t, "_dynamo_unbacked_indices"):
            t._dynamo_unbacked_indices = set()

        if not hasattr(t, "_dynamo_hint_overrides"):
            # pyrefly: ignore [implicit-any]
            t._dynamo_hint_overrides = {}

        if hint_override:
            t._dynamo_hint_overrides[index] = hint_override

        if min is not None or max is not None:
            if not hasattr(t, "_dynamo_unbacked_bounds"):
                # pyrefly: ignore [implicit-any]
                t._dynamo_unbacked_bounds = {}
            t._dynamo_unbacked_bounds[index] = (min, max)

        if shape_id is not None:
            if not hasattr(t, "_dynamo_shape_ids"):
                # pyrefly: ignore [implicit-any]
                t._dynamo_shape_ids = {}
            t._dynamo_shape_ids[index] = shape_id

        # FX tracers don't respect @forbid_in_graph and choke on the following error since it passes in proxies:
        # TypeError: 'Attribute' object does not support item assignment

        if isinstance(t._specialize_on, dict):
            t._specialize_on[index] = specialize_on if specialize_on is not None else []

        t._dynamo_unbacked_indices.add(index)
        t._has_dynamo_dim_marking = True  # type: ignore[attr-defined]
        return

    if not isinstance(index, (list, tuple)):
        raise AssertionError(
            f"Expected index to be int, list, or tuple, got {type(index)}"
        )
    for i in index:
        mark_unbacked(t, i, shape_id=shape_id, min=min, max=max)


@forbid_in_graph
def mark_dynamic(
    t: Any,
    index: int | list[Any] | tuple[Any],
    *,
    hint_override: int | None = None,
    min: int | None = None,
    max: int | None = None,
    specialize_on: list[Any] | None = None,
) -> None:
    # [FUNCTION DOC] Marks dimension(s) index of tensor t as dynamic by setting
    # attributes on t: dynamic indices, a set of _DimRange(min, max), optional
    # hint overrides and per-dim specialization criteria. Wrapper subclasses
    # mirror the call onto same-rank inner tensors. A list or tuple index
    # recurses (hint_override is not forwarded). Raises AssertionError for
    # invalid index types. Must be called outside compiled code.
    """
    Mark a tensor as having a dynamic dim and set corresponding min and max range for the dim.

    [Note - on the state of mark_dynamic]

    The behavior of having a dynamic dimension on a tensor is governed by a few factors:

    1) torch._dynamo.config dynamic_shapes True or False.
        a) dynamic_shapes=True - dynamic_shapes must be True for mark_dynamic to work.
        a) dynamic_shapes=False - This config will raise an exception when used in conjunction with
        mark_dynamic. We will eventually support this.

    2) If the dimension is fully constrained - as in, it does not allow more than a single value
    in both eager (torch.compile, torch._dynamo.optimize) mode and export mode (torch._dynamo.export),
    we will raise an error

    3) If the dimension is partially constrained - allowing at least 2 values but not the full unbounded
    range of shapes, in eager we will pass it through, but export will raise an error.

    4) Attempts to trace this function will explicitly raise. As such, all calls to mark_dynamic must be made
    before torch.compile.

    5) If hint_override is passed, the hint_override for the specified dimension will replace the provided value
    from the first example input as the official size hint. Note: changing hint_override values will cause
    FxGraphCache misses, since hint overrides affect inductor codegen decisions (autotuning, reduction
    strategy, etc.) and are included in the cache key via ShapeEnv.var_to_hint_override.

    6) If specialize_on is passed in, we will perform a single generic Dynamo trace followed by
    multiple specialized compilations in addition to a single generic compilation. NB: For now we only support
    per dimension specialization, or in other words we do not generate a cross product of specializations.
    At runtime, we will dispatch to a specialized compiled region if the input matches the specialization criteria.

    For example:
        mark_dynamic(..., specialize_on=[
            lambda x: x == 8,
            lambda x: x == 16
        ])

    This approach results in one Dynamo trace and two backend compilations. When the input dimension equals 8 or 16
    at runtime, execution will be directed to the specialized compiled region. Performance measurements indicate
    2-8x speedups depending on the specific specialization and model architecture.
    """
    if is_traceable_wrapper_subclass(t):
        # default behavior: mirror mark_dynamic() on all inner tensors with same dim as t
        # TODO: Make this configurable via a supported public API
        _apply_func_to_inner_tensors_of_same_dim(
            mark_dynamic, t, index, min=min, max=max
        )

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_dynamic_indices"):
            t._dynamo_dynamic_indices = set()

            t._dynamo_dynamic_range = set()

            # pyrefly: ignore [implicit-any]
            t._dynamo_hint_overrides = {}

        if not hasattr(t, "_specialize_on"):
            # pyrefly: ignore [implicit-any]
            t._specialize_on = {}

        if hint_override:
            t._dynamo_hint_overrides[index] = hint_override
        # TODO(voz): Should we bounds check?

        t._dynamo_dynamic_indices.add(index)
        t._dynamo_dynamic_range.add(_DimRange(index, min, max))  # type: ignore[arg-type]
        t._has_dynamo_dim_marking = True  # type: ignore[attr-defined]

        # FX tracers don't respect @forbid_in_graph and choke on the following error since it passes in proxies:
        # TypeError: 'Attribute' object does not support item assignment

        if isinstance(t._specialize_on, dict):
            t._specialize_on[index] = specialize_on if specialize_on is not None else []

        return

    if not isinstance(index, (list, tuple)):
        raise AssertionError(
            f"Expected index to be int, list, or tuple, got {type(index)}"
        )
    for i in index:
        mark_dynamic(t, i, min=min, max=max, specialize_on=specialize_on)


@forbid_in_graph
def maybe_mark_dynamic(t: Any, index: int | list[Any] | tuple[Any]) -> None:
    # [FUNCTION DOC] Records dimension(s) index of t as weakly dynamic (no error
    # if specialized) in a weak-dynamic-indices attribute and sets the dim
    # marking flag; mirrors onto same-rank inner tensors of wrapper subclasses
    # and recurses over list/tuple indices. Raises AssertionError for invalid
    # index types.
    """
    Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this
    dimension ends up getting specialized, don't error).
    """
    if is_traceable_wrapper_subclass(t):
        # default behavior: mirror maybe_mark_dynamic() on all inner tensors with same dim as t
        # TODO: Make this configurable via a supported public API
        _apply_func_to_inner_tensors_of_same_dim(maybe_mark_dynamic, t, index)

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_weak_dynamic_indices"):
            t._dynamo_weak_dynamic_indices = set()
        # TODO(voz): Should we bounds check?

        t._dynamo_weak_dynamic_indices.add(index)
        t._has_dynamo_dim_marking = True  # type: ignore[attr-defined]
        return

    if not isinstance(index, (list, tuple)):
        raise AssertionError(
            f"Expected index to be int, list, or tuple, got {type(index)}"
        )
    for i in index:
        maybe_mark_dynamic(t, i)


def mark_static(t: Any, index: int | list[Any] | tuple[Any] | None = None) -> None:
    # [FUNCTION DOC] When called during compilation, forces the sizes of t (all
    # dims or the given one) to be static. Otherwise marks tensor dims static
    # (all dims if index is None, or an int / list of ints) by setting static
    # index attributes, mirroring onto inner tensors of wrapper subclasses; for
    # an nn.Module class sets a static marker and returns the class. Raises
    # TypeError for other types and AssertionError for invalid index types.
    """
    Mark a tensor as having a static dim or mark a nn module class as static.

    For tensors
    ===========
    This will prevent us from attempting to compile it dynamically
    when dynamic=True; this can improve trace-time performance.

    This has lower precedence than mark_dynamic.

    Unlike mark_dynamic, this can be done inside a graph, in which case it
    induces specialization on the tensor.

    For nn.Module classes
    =====================
    For static nn.Module classes, TorchDynamo assumes that the module instance
    attributes will not be modified after compilation. This will ensure that
    TorchDynamo keeps integer attributes CONSTANT and not symints.

    From TorchDynamo implementation side, the instances of static-marked
    nn.Module class will be converted to UnspecializedBuiltinNNModuleVariable,
    which have the same properties.

    Note that we still have to guard on the attributes, because different
    instances of the nn.Module can have different values of the attributes. The
    key point here is that the attributes are static.
    """
    if is_compiling():
        if index is None:
            for s in t.size():
                comptime.force_static(s)
        else:
            comptime.force_static(t.size(index))
        return

    if is_traceable_wrapper_subclass(t):
        # default behavior: mirror mark_static() on all inner tensors with same dim as t
        # TODO: Make this configurable via a supported public API
        _apply_func_to_inner_tensors_of_same_dim(mark_static, t, index)

    if not isinstance(t, torch.Tensor) and issubclass(t, torch.nn.Module):
        t._dynamo_marked_static = True
        # pyrefly: ignore [bad-return]
        return t

    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"mark_static expects a tensor/nn.Module class but received {type(t)}"
        )

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_static_indices"):
            t._dynamo_static_indices = set()  # type: ignore[attr-defined]
        # TODO(voz): Should we bounds check?
        t._dynamo_static_indices.add(index)  # type: ignore[attr-defined]
        t._has_dynamo_dim_marking = True  # type: ignore[attr-defined]
    elif index is None:
        for i in range(t.dim()):
            mark_static(t, i)
    else:
        if not isinstance(index, (list, tuple)):
            raise AssertionError(
                f"Expected index to be int, list, or tuple, got {type(index)}"
            )
        for i in index:
            mark_static(t, i)


@forbid_in_graph
def mark_static_address(t: Any, guard: bool = False) -> None:
    # [FUNCTION DOC] Tags tensor t with a static input type of guarded (data
    # pointer is guarded, recompiling on change) or unguarded, so cudagraphs can
    # avoid copying it. Raises TypeError if t is not a tensor.
    """
    Marks an input tensor whose address should be treated as constant across calls to the
    same dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True, and cause a full
    recompile if the data_ptr changes. Note: If this address changes, cudagraphs will re-record
    if guard=False.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"mark_static_address expects a tensor but received {type(t)}")

    if guard:
        t._dynamo_static_input_type = "guarded"  # type: ignore[attr-defined]
    else:
        t._dynamo_static_input_type = "unguarded"  # type: ignore[attr-defined]


def _patch_einops_symint_compat(einops_mod: Any) -> None:
    # [FUNCTION DOC] HELPER: For old einops, replaces two lru_cached helper
    # functions in einops_mod with wrappers that fall back to the uncached
    # function on TypeError (unhashable SymInt). Mutates the module.
    """Backport the SymInt lru_cache fix from einops 0.7.0 into einops <= 0.6.1."""
    for name in ("_reconstruct_from_shape", "_prepare_transformation_recipe"):
        cached = getattr(einops_mod, name)
        uncached = cached.__wrapped__

        def make_wrapper(cached_fn: Any, uncached_fn: Any) -> Any:
            # [FUNCTION DOC] HELPER: Builds the cached-with-fallback wrapper.
            @functools.wraps(cached_fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # [FUNCTION DOC] HELPER: Calls the cached function; on TypeError
                # calls the uncached one instead.
                try:
                    return cached_fn(*args, **kwargs)
                except TypeError:
                    return uncached_fn(*args, **kwargs)

            return wrapper

        setattr(einops_mod, name, make_wrapper(cached, uncached))


_EINOPS_DYNAMO_TRACING_MIN_VERSION = Version("0.8.2")
# Known pure parser/shape helpers in einops 0.8.2. Keep this explicit so
# unrelated or future lru_cache wrappers continue through the normal warning path.
_EINOPS_LRU_CACHE_WRAPPER_ALLOWLIST = (
    (
        "einops.einops",
        (
            "_reconstruct_from_shape",
            "_prepare_transformation_recipe",
            "_compactify_pattern_for_einsum",
        ),
    ),
    ("einops.packing", ("analyze_pattern",)),
)


def _einops_supports_dynamo_tracing(einops_mod: Any) -> bool:
    # [FUNCTION DOC] HELPER: True if the einops version is at least the minimum
    # that Dynamo can trace directly; false if too old or unparseable.
    try:
        return Version(einops_mod.__version__) >= _EINOPS_DYNAMO_TRACING_MIN_VERSION
    except InvalidVersion:
        return False


def _allow_lru_cache_trace_without_warning_for_einops() -> None:
    # [FUNCTION DOC] HELPER: Imports each allowlisted einops module (skipping
    # missing ones) and exempts the listed lru_cache wrapped helpers from the
    # trace warning.
    import importlib

    for module_name, attr_names in _EINOPS_LRU_CACHE_WRAPPER_ALLOWLIST:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        for attr_name in attr_names:
            obj = getattr(module, attr_name, None)
            if is_lru_cache_wrapped_function(obj):
                allow_lru_cache_wrapper_trace_without_warning(obj)


# Dynamo can trace through einops 0.8.2+ directly (no allow_in_graph needed).
# Older versions still need the allow_in_graph registration below.
def _allow_in_graph_einops() -> None:
    # [FUNCTION DOC] HELPER: Module-init hook run when einops is imported. For
    # new einops, pre-registers its backend and exempts lru_cache helpers. For
    # older versions, relies on einops' own registration or else patches SymInt
    # compatibility and registers rearrange/reduce/repeat/einsum/pack/unpack
    # (when present) as allowed in graph.
    import einops

    if _einops_supports_dynamo_tracing(einops):
        if hasattr(einops, "einops") and hasattr(einops.einops, "get_backend"):
            # trigger backend registration up front to avoid a later guard failure
            # that would otherwise cause a recompilation
            einops.rearrange(torch.empty(1), "i -> i")
        # einops uses lru_cache for pure library-internal recipe helpers. Mark
        # only those wrappers so general lru_cache warnings are unchanged.
        _allow_lru_cache_trace_without_warning_for_einops()
        return

    try:
        # requires einops > 0.6.1, torch >= 2.0
        from einops._torch_specific import (  # type: ignore[attr-defined]  # noqa: F401
            _ops_were_registered_in_torchdynamo,
        )

        # einops > 0.6.1 will call the op registration logic as it is imported.
    except ImportError:
        # einops <= 0.6.1 doesn't handle unhashable SymInt in its lru_cache'd
        # helpers. Backport the try/except TypeError fallback from einops 0.7.0+
        # so allow_in_graph works during fake tensor validation.
        _patch_einops_symint_compat(einops.einops)  # type: ignore[attr-defined]
        allow_in_graph(einops.rearrange)
        allow_in_graph(einops.reduce)
        if hasattr(einops, "repeat"):
            allow_in_graph(einops.repeat)  # available since einops 0.2.0
        if hasattr(einops, "einsum"):
            allow_in_graph(einops.einsum)  # available since einops 0.5.0
        if hasattr(einops, "pack"):
            allow_in_graph(einops.pack)  # available since einops 0.6.0
        if hasattr(einops, "unpack"):
            allow_in_graph(einops.unpack)  # available since einops 0.6.0


# Note: this carefully avoids eagerly import einops.
trace_rules.add_module_init_func("einops", _allow_in_graph_einops)


class _ConfigPatchProtocol(Protocol):
    # [CLASS DOC] HELPER: Structural type for the object returned by
    # config.patch: has a changes mapping and enter/exit methods.
    """The config.patch() context manager object wrapped by DynamoConfigPatchProxy."""

    changes: dict[str, Any]

    def __enter__(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


# Proxy class for torch._dynamo.config patching - so dynamo can identify context managers/decorators
# created by patch_dynamo_config, compared to ones created by a raw torch._dynamo.config.patch.
class DynamoConfigPatchProxy:
    # [CLASS DOC] Wrapper around a config.patch object so Dynamo can recognize
    # and trace config patches made through patch_dynamo_config. State:
    # config_patch (the wrapped patch). Works as context manager and decorator
    # (the decorator form is implemented in external_utils).
    def __init__(self, config_patch: _ConfigPatchProtocol) -> None:
        # [FUNCTION DOC] Stores the wrapped config patch object.
        self.config_patch = config_patch

    @property
    def changes(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the mapping of config names to patched values
        # from the wrapped patch.
        return self.config_patch.changes

    # Decorator implementation that simply sets up `self` as a context manager.
    # Placed in external_utils so that we can trace through it.
    __call__ = wrap_dunder_call_ctx_manager

    def __enter__(self) -> None:
        # [FUNCTION DOC] Applies the wrapped config patch.
        return self.config_patch.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Restores the config by delegating to the wrapped patch.
        return self.config_patch.__exit__(exc_type, exc_val, exc_tb)


# Criteria for patchable config:
# - Config values must be constants (i.e. int, float, str, bool, None).
#     - in particular, NO list, set, dict.
# - Traceable config patches are only useful for configs that change dynamo behavior
#   from symbolic_convert and below.
#     - e.g. patching recompile_limit won't really do anything.
# - For patching configs that affect Dynamo behavior above symbolic_convert,
#   ensure that Dynamo behaves soundly even if tracing is done with different config.
#     - e.g. be careful if patching guard-related configs as configs may have changed
#       between guard creation and evaluation.
_allowed_config_patches = (
    "verbose",
    "verify_correctness",
    "rewrite_assert_with_torch_assert",
    "capture_scalar_outputs",
    "allow_unspec_int_on_nn_module",
    "skip_torchrec",
    "dont_skip_tracing",
    "nested_graph_breaks",
)

from . import config


for name in _allowed_config_patches:
    if not hasattr(config, name):
        raise AssertionError(f"nonexistent config: {name}")
del config


def _patch_dynamo_config_check(changes: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Validates a config patch mapping. Raises ValueError
    # if any key is not in the allowed list or any value is not a safe constant.
    for k, v in changes.items():
        if k not in _allowed_config_patches:
            raise ValueError(
                f"patch_dynamo_config does not support patching config {k}"
            )
        if not torch._dynamo.utils.is_safe_constant(v):
            raise ValueError(
                f"patch_dynamo_config does not support patching config {k} "
                f"with non-safe-constant value {v}"
            )


# TODO: also implement nonrecursive patch_dynamo_config/dont_skip_tracing.
# Unlike config.patch, we also need to accept tuple as input in order to
# deal with context manager reconstruction.
def patch_dynamo_config(
    arg1: str | dict[str, Any] | tuple[tuple[str, Any], ...] | None = None,
    arg2: Any = None,
    **kwargs: Any,
) -> DynamoConfigPatchProxy:
    # [FUNCTION DOC] Creates a config.patch from the same arguments (a tuple of
    # pairs is converted to a mapping), validates its changes against the
    # allowed list of traceable patches (ValueError on disallowed key or
    # non-constant value) and returns it wrapped in a DynamoConfigPatchProxy.
    """
    A wrapper around torch._dynamo.config.patch that can be traced by Dynamo to
    temporarily change config values DURING tracing.

    See _allowed_config_patches for the list of allowed config patches.

    Arguments are the same as with torch._dynamo.config.patch.

    Can be used as a decorator or a context manager.

    User code SHOULD NOT MODIFY the return value of this function.

    WARNING: changing Dynamo config during tracing can lead to unpredictable tracing behavior!
        Proceed only as advised!
    """
    if isinstance(arg1, tuple):
        arg1 = dict(arg1)
    config_patch = torch._dynamo.config.patch(arg1, arg2, **kwargs)
    _patch_dynamo_config_check(config_patch.changes)
    # check for valid patching using config_patch.changes
    return DynamoConfigPatchProxy(config_patch)


@overload
def dont_skip_tracing(fn: None = None) -> DynamoConfigPatchProxy: ...


@overload
def dont_skip_tracing(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...


def dont_skip_tracing(fn: Any | None = None) -> Any:
    # [FUNCTION DOC] Returns a traceable config patch enabling the
    # dont_skip_tracing option; if fn is given returns fn decorated with it,
    # otherwise the patch to use as a context manager.
    """
    Context manager/decorator to trace into functions intentionally marked by developers to be skipped
    when tracing.

    This decorator will also apply to recursively invoked functions.
    """
    ctx = patch_dynamo_config(dont_skip_tracing=True)
    if fn:
        return ctx(fn)
    return ctx


@overload
def disable_nested_graph_breaks(fn: None = None) -> DynamoConfigPatchProxy: ...


@overload
def disable_nested_graph_breaks(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...


def disable_nested_graph_breaks(fn: Any | None = None) -> Any:
    # [FUNCTION DOC] Returns a traceable config patch turning off nested graph
    # breaks; if fn is given returns fn decorated with it, otherwise the patch to
    # use as a context manager.
    """
    Context manager/decorator to disable nested graph breaks when tracing
    this function and any nested functions. Used when nested graph breaks
    is causing problems.
    """
    ctx = patch_dynamo_config(nested_graph_breaks=False)
    if fn:
        return ctx(fn)
    return ctx


class ErrorOnGraphBreakDecoratorContextManager:
    # [CLASS DOC] Context manager/decorator that sets the global
    # error-on-graph-break flag for its scope. State: error_on_graph_break
    # (target value) and prev_error_on_graph_break (stack of previous values, or
    # None where no change was made, to support re-entrancy).
    def __init__(self, error_on_graph_break: bool) -> None:
        # [FUNCTION DOC] Stores the target flag value and an empty previous-value
        # stack.
        self.error_on_graph_break = error_on_graph_break
        self.prev_error_on_graph_break: list[bool | None] = []

    __call__ = wrap_dunder_call_ctx_manager

    def __enter__(self) -> None:
        # [FUNCTION DOC] Pushes the previous flag value (or None if already
        # equal) and sets the global flag to the target value.
        current_error_on_graph_break = _get_error_on_graph_break()
        if current_error_on_graph_break != self.error_on_graph_break:
            self.prev_error_on_graph_break.append(current_error_on_graph_break)
            _set_error_on_graph_break(self.error_on_graph_break)
        else:
            self.prev_error_on_graph_break.append(None)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Pops the saved value and restores the global flag if it
        # had been changed; exceptions are not suppressed.
        prev_error_on_graph_break = self.prev_error_on_graph_break.pop()
        if prev_error_on_graph_break is not None:
            _set_error_on_graph_break(prev_error_on_graph_break)


def error_on_graph_break(
    error_on_graph_break: bool,
) -> ErrorOnGraphBreakDecoratorContextManager:
    # [FUNCTION DOC] Returns a context manager/decorator that sets Dynamo's
    # error_on_graph_break flag to the given value within its scope (no effect
    # under fullgraph).
    """
    Context manager/decorator to toggle torch.compile's `error_on_graph_break` setting at compile time.

    If `fullgraph` is set, then `error_on_graph_break` does nothing
    (i.e. `fullgraph = True` takes higher precedence). If `fullgraph` is False, then
    `error_on_graph_break` determines whether `torch.compile` throws an error upon
    encountering a graph break, or attempts to continue tracing.

    `error_on_graph_break` can be toggled during compile time with this decorator to allow graph breaks in some
    compiled regions but not others. One key difference from `fullgraph` is that `error_on_graph_break = True`
    does NOT guarantee that a single graph is captured from the compiled function.

    The default value of torch.compile's `error_on_graph_break` setting is False.
    """
    return ErrorOnGraphBreakDecoratorContextManager(error_on_graph_break)


class CudagraphOverrideContextManager:
    # [CLASS DOC] Context manager/decorator that sets a global cudagraph override
    # pair (forward, backward) for its scope. State: fwd and bwd (True force on,
    # False disable, None no override) and prev_cudagraph_override (stack of
    # earlier override values).
    """Context manager that overrides cudagraph recording during tracing."""

    def __init__(self, fwd: bool | None = None, bwd: bool | None = None) -> None:
        # [FUNCTION DOC] Stores the forward/backward override settings and an
        # empty previous-value stack.
        self.fwd = fwd
        self.bwd = bwd
        self.prev_cudagraph_override: list[tuple[bool | None, bool | None] | None] = []

    __call__ = wrap_dunder_call_ctx_manager

    def __enter__(self) -> None:
        # [FUNCTION DOC] Pushes the current global cudagraph override and sets
        # the (fwd, bwd) override globally.
        # Set the override at runtime so it follows the dynamic call stack.
        # Dynamo handles the lexical `with` symbolically and does not run this
        # __enter__ at trace time; it runs when the generated/resumed bytecode
        # re-enters the context, before invoking callees that are compiled as
        # separate frames. OutputGraph seeds its annotation from this value via
        # _get_cudagraph_override. Mirrors ErrorOnGraphBreakDecoratorContextManager.
        self.prev_cudagraph_override.append(_get_cudagraph_override())
        _set_cudagraph_override((self.fwd, self.bwd))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Pops and restores the previous global cudagraph
        # override; exceptions are not suppressed.
        _set_cudagraph_override(self.prev_cudagraph_override.pop())


def override_cudagraphs(
    fwd: bool | None = None, bwd: bool | None = None
) -> CudagraphOverrideContextManager:
    # [FUNCTION DOC] Returns a CudagraphOverrideContextManager for the given
    # forward/backward settings, usable as a context manager or decorator.
    """
    Context manager/decorator to override cudagraph recording for compiled graphs.

    When used as a context manager, overrides cudagraphs for all graph segments
    within the block (including across graph breaks).

    When used as a decorator, marks a function so that any compiled graph
    inlining it will have cudagraphs overridden.

    Args:
        fwd: If False, disable cudagraphs for forward. If True, force enable.
             If None, don't override.
        bwd: If False, disable cudagraphs for backward. If True, force enable.
             If None, don't override.
    """
    return CudagraphOverrideContextManager(fwd=fwd, bwd=bwd)


def override_optimization_hint(x: Any, val: int) -> None:
    # [FUNCTION DOC] Records val as the optimization hint of the unbacked symbol
    # behind SymInt x in its shape env (a no-op for plain int x). Raises
    # TypeError if val is not an int or x is not SymInt/int, and ValueError if x
    # is a derived expression or a backed symbol. Mutates the shape env.
    """Override the optimization hint for a scalar unbacked symbol.

    When the compiler or runtime needs a non-guarding integer hint for an
    unbacked ``SymInt`` — for example during FX passes, graph partitioning,
    or inductor autotuning — it calls
    ``_optimization_hint_base``
    (see ``torch/fx/experimental/_size_hinting.py``).  By default
    that function uses internal heuristics to choose a hint and a global fixed
    fallback;
    this function lets user code override that choice.  This is similar to
    the ``hint_override`` parameter in ``mark_unbacked``, but applies to
    symbols that already exist (e.g. from ``.item()`` calls).

    Typical usage::

        u = x.item()  # unbacked SymInt
        torch._dynamo.override_optimization_hint(u, 42)
        # From now on, any call to shape_env.optimization_hint(u, ...)
        # returns 42 instead of the default heuristic value.

    This updates ``shape_env.var_to_hint_override`` so that any consumer
    of ``_optimization_hint_base`` sees *val* as the hint for the
    unbacked symbol behind *x*.

    Works both eagerly (during FX passes or outside dynamo) and inside
    ``torch.compile`` regions.

    Behavior during compilation:
        The dynamo handler applies the hint as a **side effect** on
        ``shape_env.var_to_hint_override`` during tracing.  No FX graph
        node is emitted — the call is fully consumed at trace time and
        does not appear in the pre-grad, joint, or post-autograd graphs.
        ``FXGraphCache`` includes ``var_to_hint_override`` in its cache
        key, so cache hits/misses correctly reflect hint changes.

    .. note::

        To maximize performance, it is recommended to pass hints for
        **all** unbacked symbols in the program to guide optimizations.

    Args:
        x: A ``torch.SymInt`` wrapping an unbacked symbol (e.g. from
            ``.item()``), or a plain ``int``.  If *x* is a plain ``int``
            the call is a no-op.
        val: The integer hint value to record.
    """
    if not isinstance(val, int):
        raise TypeError(
            f"override_optimization_hint expects val to be an int, got {type(val)}"
        )
    if isinstance(x, int):
        return
    if not isinstance(x, torch.SymInt):
        raise TypeError(
            f"override_optimization_hint expects a torch.SymInt or int, got {type(x)}"
        )
    shape_env = not_none(x.node.shape_env)
    expr = x.node.expr
    import sympy

    if not isinstance(expr, sympy.Symbol):
        raise ValueError(
            f"override_optimization_hint expects a single unbacked symbol, "
            f"got derived expression: {expr}"
        )
    if not shape_env.is_unbacked_symint(expr):
        raise ValueError(
            f"override_optimization_hint expects an unbacked symbol, "
            f"but {expr} is backed"
        )
    shape_env._set_unbacked_var_to_hint_override(x, val)


def is_dynamo_disable_recursive(method: Callable[[Any], Any]) -> bool | None:
    # [FUNCTION DOC] Returns the recursive flag recorded by disable on method:
    # true, false, or None if method was not wrapped by disable.
    """
    Check if a method is marked as `dynamo_disable` recursively. It returns:
    - True if disable(recursive=True)
    - False if disable(recursive=False)
    - None if method is not a disable decorator
    """
    return getattr(method, "_torchdynamo_disable_recursive", None)


_HASH_SLOTS = ("__hash__",)
_RICHCOMPARE_SLOTS = ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__")


def allow_c_slot(
    tp: type,
    *,
    tp_hash: bool = True,
    tp_richcompare: bool = True,
) -> type:
    # [FUNCTION DOC] Adds the type's C-level hash and/or comparison slot
    # functions (those not inherited from object) to Dynamo's safe slot set so
    # tracing does not graph-break on them. Returns tp unchanged. Raises
    # TypeError if tp is not a type. Mutates global state.
    """Register a C extension type's slots as safe to call at trace time.

    By default, ``torch.compile`` graph-breaks when it encounters ``hash()``
    or comparison operators on a C extension type with custom C slots (e.g.,
    types defined in C extension modules).  This function tells Dynamo that the
    type's C slots are safe to evaluate during tracing, avoiding graph breaks.

    The slot functions must satisfy these requirements:

    - **Pure**: depends only on the object's value, with no observable side
      effects (no I/O, no mutation of global state).
    - **Deterministic**: returns the same result for the same object across
      calls within a process.
    - **Immutable objects**: the object's state must not change after
      construction in a way that affects the slot's return value.

    Builtin types (``int``, ``str``, etc.) and Python-level dunder methods
    are already handled and do not need registration.  This API is only needed
    for C extension types whose C slots Dynamo cannot trace into.

    Args:
        tp: The C extension type whose C slots should be allowed.
        tp_hash: Register ``__hash__`` as safe (default True).
        tp_richcompare: Register comparison dunders as safe (default True).

    Returns:
        The type, unchanged (so it can be used as a decorator).

    Example::

        import torch._dynamo
        from my_extension import MyType

        # Register all slots (hash + comparison)
        torch._dynamo.allow_c_slot(MyType)

        # Register only hash
        torch._dynamo.allow_c_slot(MyType, tp_richcompare=False)

        # Register only comparison
        torch._dynamo.allow_c_slot(MyType, tp_hash=False)
    """
    from .variables.user_defined import _safe_c_slots

    if not isinstance(tp, type):
        raise TypeError(f"allow_c_slot expects a type, got {type(tp).__name__}")
    safe = _safe_c_slots()
    dunders = ()
    if tp_hash:
        dunders += _HASH_SLOTS
    if tp_richcompare:
        dunders += _RICHCOMPARE_SLOTS
    for dunder in dunders:
        fn = getattr(tp, dunder, None)
        if fn is not None and fn is not getattr(object, dunder, None):
            safe.add(fn)
    return tp
