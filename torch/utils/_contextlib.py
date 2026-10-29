# mypy: allow-untyped-defs

# [MODULE DOC] Helpers for using context managers as decorators: a wrapper that applies a
# context manager around each step of a generator, a general decorating function, and base
# classes that turn a context manager into a decorator (with or without parentheses).
# Extra utilities for working with context managers that should have been
# in the standard library but are not

import functools
import inspect
import sys
import warnings
from collections.abc import Callable
from typing import Any, cast, overload, TypeVar
from typing_extensions import Self


# Used for annotating the decorator usage of _DecoratorContextManager (e.g.,
# 'no_grad' and 'enable_grad').
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def _wrap_generator(ctx_factory, func):
    # [FUNCTION DOC] HELPER: Given a factory producing context managers and a generator
    # function, returns a generator function that enters a fresh context around every resume
    # of the wrapped generator (start, send, throw, close) and forwards values, thrown errors
    # and the final return value. Exits the context on each suspension at a yield.
    """
    Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Generator that drives the wrapped generator, entering the
        # context for each step, relaying yielded values and requests; closes or throws into
        # the inner generator as needed; returns the inner generator's return value.
        gen = func(*args, **kwargs)

        # Generators are suspended and unsuspended at `yield`, hence we
        # make sure the grad mode is properly set every time the execution
        # flow returns into the wrapped generator and restored when it
        # returns through our `yield` to our caller (see PR #49017).
        try:
            # Issuing `None` to a generator fires it up
            with ctx_factory():
                response = gen.send(None)

            while True:
                try:
                    # Forward the response to our caller and get its next request
                    request = yield response

                except GeneratorExit:
                    # Inform the still active generator about its imminent closure
                    with ctx_factory():
                        gen.close()
                    raise

                except BaseException:
                    # Propagate the exception thrown at us by the caller
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())

                else:
                    # Pass the last request to the generator and get its response
                    with ctx_factory():
                        response = gen.send(request)

        # We let the exceptions raised above by the generator's `.throw` or
        # `.send` methods bubble up to our caller, except for StopIteration
        except StopIteration as e:
            # The generator informed us that it is done: take whatever its
            # returned value (if any) was and indicate that we're done too
            # by returning it (see docs for python's return-statement).
            return e.value

    return generator_context


def context_decorator(ctx, func):
    # [FUNCTION DOC] Returns func wrapped so each call runs inside the context given by ctx
    # (a reusable context manager, or a factory called on every use). Generator functions
    # get per-step wrapping. Raises AssertionError if ctx is both callable and a context
    # manager, and RuntimeError if func is a class.
    """
    Like contextlib.ContextDecorator.

    But with the following differences:
    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """
    if callable(ctx) and hasattr(ctx, "__enter__"):
        raise AssertionError(
            f"Passed in {ctx} is both callable and also a valid context manager "
            "(has __enter__), making it ambiguous which interface to use.  If you "
            "intended to pass a context manager factory, rewrite your call as "
            "context_decorator(lambda: ctx()); if you intended to pass a context "
            "manager directly, rewrite your call as context_decorator(lambda: ctx)"
        )

    if not callable(ctx):

        def ctx_factory():
            # [FUNCTION DOC] HELPER: Returns the captured context manager object itself.
            return ctx

    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped function inside a new context and returns
        # its result.
        # pyrefly: ignore [bad-context-manager]
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    # [CLASS DOC] HELPER: Base class for context managers usable as decorators. Subclasses
    # implement enter and exit and may override clone when their constructor takes
    # parameters; each decorated call uses a fresh clone.
    """Allow a context manager to be used as a decorator."""

    def __call__(self, orig_func: F) -> F:
        # [FUNCTION DOC] Decorates a function with a clone of this manager. A class argument
        # triggers a FutureWarning (deprecated) and is wrapped through a function calling it.
        # Returns the decorated callable.
        if inspect.isclass(orig_func):
            warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else.",
                FutureWarning,
                stacklevel=2,
            )
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func

        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        # [FUNCTION DOC] Abstract; subclasses must override. Raises NotImplementedError.
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # [FUNCTION DOC] Abstract; subclasses must override. Raises NotImplementedError.
        raise NotImplementedError

    def clone(self):
        # [FUNCTION DOC] Returns a new instance of the same class built with no arguments.
        # override this method if your children class takes __init__ parameters
        return self.__class__()


class _NoParamDecoratorContextManager(_DecoratorContextManager):
    # [CLASS DOC] HELPER: Decorator-capable context manager that can be applied directly
    # to a function without parentheses; constructing with a function returns the
    # decorated function, constructing with nothing returns a manager instance.
    """Allow a context manager to be used as a decorator without parentheses."""

    @overload
    def __new__(cls, orig_func: F) -> F: ...  # type: ignore[misc]
    @overload
    def __new__(cls, orig_func: None = None) -> Self: ...

    def __new__(cls, orig_func: F | None = None) -> Self | F:  # type: ignore[misc]
        # [FUNCTION DOC] With no function returns a normal new instance; with a function
        # returns that function decorated by a fresh instance.
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)
