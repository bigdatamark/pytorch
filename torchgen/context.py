# [MODULE DOC]
# Context-manager helpers used by torchgen code generators. Provides a manager that
# attaches native_functions.yaml error context and sets thread-local codegen flags for a
# function, plus decorators that wrap generator functions so they run inside it.
# Import-time side effects: only defines type variables and functions.
from __future__ import annotations

import contextlib
import functools
from typing import Any, TYPE_CHECKING, TypeVar

import torchgen.local as local
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
from torchgen.utils import context, S, T


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


# Helper functions for defining generators on things in the model

F = TypeVar(
    "F",
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    NativeFunction | NativeFunctionsGroup,
    NativeFunction | NativeFunctionsViewGroup,
)

F2 = TypeVar(
    "F2",
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunction | None,
    bool,
    str,
)

F3 = TypeVar("F3", tuple[NativeFunction, Any], list[NativeFunction])


@contextlib.contextmanager
def native_function_manager(
    g: NativeFunctionsGroup | NativeFunctionsViewGroup | NativeFunction,
) -> Iterator[None]:
    # [FUNCTION DOC]
    # Context manager taking a native function, a structured group (uses its out variant)
    # or a view group (uses its view operator). While active, errors are annotated with
    # the yaml location and signature, and the thread-local codegen flags
    # (const-ref for mutable tensors, ilistref for tensor lists) are set from the function.
    # Yields nothing; the flags are restored on exit.
    if isinstance(g, NativeFunctionsGroup):
        # By default, we associate all errors with structured native functions
        # with the out variant.  In some cases, it might be better to have
        # a more specific place to hang things; if so, use
        # native_function_manager again on the inside
        f = g.out
    elif isinstance(g, NativeFunctionsViewGroup):
        # We associate errors with the view operator
        f = g.view
    else:
        f = g
    with context(lambda: f"in native_functions.yaml line {f.loc}:\n  {f.func}"):
        with local.parametrize(
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=f.part_of_structured_group,
        ):
            yield


# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    # [FUNCTION DOC]
    # Decorator: returns a wrapper that calls the given function with its single argument
    # inside native_function_manager for that argument. Return value passes through.
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        # [FUNCTION DOC] HELPER: Runs the wrapped function under the manager for f.
        with native_function_manager(f):
            return func(f)

    return wrapper


def with_native_function_and(func: Callable[[F, F2], T]) -> Callable[[F, F2], T]:
    # [FUNCTION DOC]
    # Decorator for two-argument functions: the wrapper sets up the native function
    # context from the first argument only and forwards both arguments.
    @functools.wraps(func)
    def wrapper(f: F, f2: F2) -> T:
        # [FUNCTION DOC] HELPER: Runs the wrapped function under the manager for f.
        # The first native_function is assumed to be the one with the appropriate context.
        with native_function_manager(f):
            return func(f, f2)

    return wrapper


def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    # [FUNCTION DOC]
    # Decorator for methods taking (self, f): the wrapper sets up the native function
    # context from f and forwards both arguments.
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        # [FUNCTION DOC] HELPER: Runs the wrapped method under the manager for f.
        with native_function_manager(f):
            return func(slf, f)

    return wrapper


def method_with_nested_native_function(
    func: Callable[[S, F3], T],
) -> Callable[[S, F3], T]:
    # [FUNCTION DOC]
    # Decorator for methods taking (self, f) where f is a tuple or list whose first element
    # is a native function; the context is set up from that first element.
    @functools.wraps(func)
    def wrapper(slf: S, f: F3) -> T:
        # [FUNCTION DOC] HELPER: Runs the wrapped method under the manager for f[0].
        with native_function_manager(f[0]):
            return func(slf, f)

    return wrapper


# Convenience decorator for functions that explicitly take in a BackendIndex,
# instead of indirectly taking one in as a closure
def with_native_function_and_index(
    func: Callable[[F, BackendIndex], T],
) -> Callable[[F, BackendIndex], T]:
    # [FUNCTION DOC]
    # Decorator for functions taking (f, backend_index): sets up the native function
    # context from f and forwards both arguments.
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        # [FUNCTION DOC] HELPER: Runs the wrapped function under the manager for f.
        with native_function_manager(f):
            return func(f, backend_index)

    return wrapper


# Convenience decorator for functions that explicitly take in a Dict of BackendIndices
def with_native_function_and_indices(
    func: Callable[[F, dict[DispatchKey, BackendIndex]], T],
) -> Callable[[F, dict[DispatchKey, BackendIndex]], T]:
    # [FUNCTION DOC]
    # Decorator for functions taking (f, mapping of dispatch key to backend index): sets up
    # the native function context from f and forwards both arguments.
    @functools.wraps(func)
    def wrapper(f: F, backend_indices: dict[DispatchKey, BackendIndex]) -> T:
        # [FUNCTION DOC] HELPER: Runs the wrapped function under the manager for f.
        with native_function_manager(f):
            return func(f, backend_indices)

    return wrapper
