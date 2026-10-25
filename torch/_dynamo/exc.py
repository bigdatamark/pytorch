# [MODULE DOC] Exception hierarchy and error-reporting helpers for TorchDynamo. Defines the base
# TorchDynamoException (picklable), restart/skip exceptions used for control flow in tracing,
# Unsupported (graph break) and the unimplemented helper that raises it, user-facing errors,
# observed-exception classes (and the map from builtin exception types to them), helpers to
# raise exceptions inside symbolic tracing, and functions to format messages and stacks.
# Import-time side effects: creates module loggers (including a graph_breaks artifact logger)
# and the observed_exception_map / allowed-fallback exception tuple.
from __future__ import annotations


"""Exception handling and error reporting for TorchDynamo.

This module provides a comprehensive set of exception classes and utilities for error
handling in TorchDynamo. It includes:

Base Exceptions:
    - TorchDynamoException: Base class for all TorchDynamo-specific exceptions
    - Various specialized subclasses for different error scenarios

User Error Handling:
    - UserError: Exceptions for user-facing errors in TorchDynamo usage
    - UserErrorType: Enumeration of different categories of user errors
    - Formatted error messages with debugging information

Observed Exceptions:
    - Classes for handling exceptions observed during tracing
    - Special handling for StopIteration, LookupError, etc.
    - Exception state management during compilation

    When an exception is raised inside a ``torch.compile`` region,
    TorchDynamo does **not** immediately propagate it to the caller.
    Instead, Dynamo symbolically traces the exception so that
    ``try``/``except`` blocks inside the compiled function can be
    captured in the FX graph and reasoned about statically.

    Common Python exception types are wrapped in ``ObservedException``
    subclasses (e.g. ``StopIteration`` → ``ObservedUserStopIteration``,
    ``KeyError`` → ``ObservedKeyError``).  The full mapping is defined in
    ``observed_exception_map``.  For exception types not present in that
    map, :func:`get_dynamo_observed_exception` creates a new
    ``ObservedException`` subclass dynamically at tracing time.

    If an observed exception is **not caught** inside the compiled region
    it propagates out and is re-raised as the original exception type in
    eager mode, preserving the original traceback for the user.

    .. note::
        This behaviour means that exception-based control flow (e.g.
        ``StopIteration`` from a custom iterator, ``KeyError`` from a
        dict access) is generally supported inside ``torch.compile``
        regions.  However, raising or catching exceptions that depend on
        *data-dependent* conditions may still cause a graph break because
        Dynamo cannot statically determine which branch will be taken at
        compile time.

Error Formatting:
    - Stack trace filtering and formatting
    - Error message augmentation
    - Debugging utilities for error reporting
"""

import json
import logging
import re
import sys
import textwrap
import types
import typing
from enum import auto, Enum
from functools import lru_cache
from pathlib import Path
from traceback import format_exc, format_list, FrameSummary, StackSummary
from typing import Any, NoReturn, TYPE_CHECKING

import torch._guards
from torch._utils_internal import get_file_path_2

from . import config
from .utils import counters


if TYPE_CHECKING:
    from torch._dynamo.variables import VariableTracker
    from torch._guards import CompileId

    from .output_graph import DynamoTracerOutput
    from .symbolic_convert import InstructionTranslatorBase
    from .types import DynamoFrameType, FrameExecStrategy


def exportdb_error_message(case_name: str) -> str:
    # [FUNCTION DOC] Returns a sentence with the export-database documentation URL for the given
    # case name (underscores replaced by hyphens in the anchor). Pure.
    return (
        "For more information about this error, see: "
        + "https://pytorch.org/docs/main/generated/exportdb/index.html#"
        + case_name.replace("_", "-")
    )


log = logging.getLogger(__name__)
graph_breaks_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")


_EXCEPTION_STATE_ATTRS_TO_DROP = frozenset(
    (
        "_torch_dynamo_tracer_output",
        "first_useful_frame",
        "frame_exec_strategy",
    )
)


def _safe_exception_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    # [FUNCTION DOC] HELPER: Hook for sanitizing exception args before pickling; currently returns
    # the args unchanged.
    return args


class _SerializedException(RuntimeError):
    # [CLASS DOC] HELPER: Picklable stand-in for an arbitrary inner exception. Carries the
    # original message as its text and the fully qualified original exception type name in
    # _dynamo_original_exception_type.
    _dynamo_original_exception_type: str

    def __init__(self, exc: BaseException) -> None:
        # [FUNCTION DOC] HELPER: Stores str of exc as the message and records the original
        # exception's module and qualified type name.
        super().__init__(str(exc))
        self._dynamo_original_exception_type = (
            f"{type(exc).__module__}.{type(exc).__qualname__}"
        )


def _safe_inner_exception(exc: BaseException) -> _SerializedException:
    # [FUNCTION DOC] HELPER: Wraps exc in a picklable serialized exception and returns it.
    return _SerializedException(exc)


def _safe_exception_state(state: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a copy of an exception's attribute mapping that is safe to
    # pickle: tracer output, first useful frame, frame exec strategy and any frame objects are
    # replaced by None, an inner_exception attribute is replaced by its serialized form, and
    # everything else is kept as is.
    result: dict[str, Any] = {}
    for name, value in state.items():
        if name in _EXCEPTION_STATE_ATTRS_TO_DROP or isinstance(value, types.FrameType):
            result[name] = None
        elif name == "inner_exception" and isinstance(value, BaseException):
            result[name] = _safe_inner_exception(value)
        else:
            result[name] = value
    return result


def _reconstruct_torch_dynamo_exception(
    exc_type: type[TorchDynamoException], args: tuple[Any, ...]
) -> TorchDynamoException:
    # [FUNCTION DOC] HELPER: Unpickling factory. Creates an instance of exc_type without calling
    # its constructor and initializes only the base exception with args, so subclasses with
    # custom constructor signatures can be rebuilt. Attribute state is restored separately.
    exc = exc_type.__new__(exc_type)
    BaseException.__init__(exc, *args)
    return exc


class TorchDynamoException(RuntimeError):
    """Base exception class for all TorchDynamo-specific exceptions.

    Attributes:
        _torch_dynamo_tracer_output: Optional tracer output attached to the exception
        frame_exec_strategy: Optional frame execution strategy to control how convert_frame
            should handle this exception. When set, convert_frame will use this strategy
            instead of the default behavior. This allows exceptions to signal specific
            execution strategies (e.g., SKIP, RUN_ONLY) without requiring separate
            exception types for control flow.
    """

    # [CLASS DOC] Base class of all Dynamo exceptions (a RuntimeError). Attributes: tracer output
    # and frame exec strategy, both initially None. Supports pickling via a custom reduce that
    # drops unpicklable state (frames, tracer output).
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Initializes the base error with the arguments and sets tracer output and
        # frame exec strategy to None.
        super().__init__(*args, **kwargs)
        self._torch_dynamo_tracer_output: DynamoTracerOutput | None = None
        self.frame_exec_strategy: FrameExecStrategy | None = None

    def __reduce__(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Pickle support: returns the reconstruction factory with the exception
        # type and sanitized args, plus the sanitized attribute state.
        return (
            _reconstruct_torch_dynamo_exception,
            (type(self), _safe_exception_args(self.args)),
            self.__getstate__(),
        )

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a pickle-safe copy of the instance attribute mapping.
        return _safe_exception_state(self.__dict__)

    def __setstate__(self, state: dict[str, Any] | None, /) -> None:
        # [FUNCTION DOC] Restores attributes by merging the given state mapping (if not None) into
        # the instance attributes.
        if state is not None:
            self.__dict__.update(state)


class InternalTorchDynamoError(TorchDynamoException):
    # [CLASS DOC] Marker for internal Dynamo errors (bugs); no extra state.
    pass


class ResumePrologueTracingError(TorchDynamoException):
    # [CLASS DOC] Marker for failures while tracing a resume function prologue; no extra state.
    pass


class RestartAnalysis(TorchDynamoException):
    # [CLASS DOC] Control-flow exception asking the tracer to discard the current analysis and
    # start over (possibly with changed behavior). Attribute: restart_reason (optional text).
    restart_reason: str | None

    def __init__(self, *args: Any, restart_reason: str | None = None) -> None:
        # [FUNCTION DOC] Records the optional restart reason, then initializes the base with args.
        self.restart_reason = restart_reason
        super().__init__(*args)


class SpeculationRestartAnalysis(RestartAnalysis):
    # [CLASS DOC] Restart signal used when speculation state requires re-tracing; no extra state.
    pass


class AutogradGradRestartAnalysis(RestartAnalysis):
    # [CLASS DOC] Restart signal: autograd grad consumed returned grad functions; on restart the
    # grad call causes a graph break instead of being traced.
    """Raised when autograd.grad consumed grad_fns that are returned.

    On restart, autograd.grad will graph break instead of being traced.
    """


class RequiresGradRestartAnalysis(RestartAnalysis):
    # [CLASS DOC] Restart signal: a source-less intermediate set to require grad leaked as an
    # output; on restart that call graph breaks instead of being traced.
    """Raised when a source-less requires_grad_() intermediate leaks as output.

    On restart, requires_grad_() will graph break instead of being traced,
    preserving partial acceleration for code before the call.
    """


class UnspecializeRestartAnalysis(RestartAnalysis):
    # [CLASS DOC] Restart signal used to retrace after unspecializing a value; no extra state.
    pass


class CompileCollectiveRestartAnalysis(RestartAnalysis):
    # [CLASS DOC] Restart signal used for compile-time collective (cross-rank) decisions; no extra state.
    pass


class TensorifyScalarRestartAnalysis(RestartAnalysis):
    # [CLASS DOC] Restart signal used when scalars must be converted to tensors; no extra state.
    pass


# Used (primarily for backends) to skip tracing the current frame
# and all future invocations of it.
# NOTE: this does NOT cause a graph break, and thus no graph break messages
# will be issued!
class SkipFrame(TorchDynamoException):
    # [CLASS DOC] Signals that the current frame and all future calls of it should not be traced;
    # not a graph break and produces no graph break messages. No extra state.
    pass


class TorchRuntimeError(TorchDynamoException):
    # [CLASS DOC] Wraps a runtime failure raised by torch during tracing. Attributes: msg and
    # real_stack (user stack summary).
    def __init__(self, msg: str, real_stack: StackSummary | None = None) -> None:
        # [FUNCTION DOC] Stores msg and the user stack; if real_stack is None, extracts the stack
        # from the current tracing context.
        super().__init__(msg)
        self.msg = msg
        self.real_stack = (
            real_stack
            if real_stack is not None
            else torch._guards.TracingContext.extract_stack()
        )


class InvalidBackend(TorchDynamoException):
    # [CLASS DOC] Raised when a requested backend name is unknown; message lists suggestions.
    def __init__(self, name: str, suggestions: list[str] | None = None) -> None:
        # [FUNCTION DOC] Builds a message naming the invalid backend, optionally suggesting
        # similar names, and pointing to the list-backends API; passes it to the base.
        msg = f"Invalid backend: {name!r}"
        msg += (
            f", did you mean: {', '.join(map(repr, suggestions))}?"
            if suggestions
            else "."
        )
        msg += " See `torch._dynamo.list_backends()` for available backends."
        super().__init__(msg)


class ResetRequired(TorchDynamoException):
    # [CLASS DOC] Raised when compile is called with different backend arguments without a reset.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes with a fixed message telling the user to call reset first.
        super().__init__(
            textwrap.dedent(
                """
                Must call `torch._dynamo.reset()` before changing backends.  Detected two calls to
                `torch.compile()` with a different backend compiler arguments.
                """
            )
        )


class ShortenTraceback(TorchDynamoException):
    # [CLASS DOC] Exception whose traceback can be trimmed to start at the first user-relevant
    # frame. Attribute: first_useful_frame (frame object or None).
    def __init__(
        self, *args: Any, first_useful_frame: types.FrameType | None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Initializes the base with args and stores first_useful_frame.
        super().__init__(*args, **kwargs)
        self.first_useful_frame = first_useful_frame

    def remove_dynamo_frames(self) -> typing.Self:
        # [FUNCTION DOC] Returns self with its traceback cut to begin at first_useful_frame. No
        # change if that frame is None, there is no traceback, or verbose config is on. Raises
        # AssertionError if the frame is not found in the traceback.
        tb = self.__traceback__
        if self.first_useful_frame is None or tb is None or config.verbose:
            return self
        while tb.tb_frame is not self.first_useful_frame:
            tb = tb.tb_next
            if tb is None:
                raise AssertionError("internal error, please report a bug")
        return self.with_traceback(tb)


class BackendCompilerFailed(ShortenTraceback):
    # [CLASS DOC] Raised when the user's backend compiler raised. Attributes: backend_name,
    # inner_exception (the original error).
    def __init__(
        self,
        backend_fn: Any,
        inner_exception: Exception,
        first_useful_frame: types.FrameType | None,
    ) -> None:
        # [FUNCTION DOC] Records the backend name (function name or a placeholder) and the inner
        # exception, and builds a message containing both the inner type and text.
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        msg = f"backend={self.backend_name!r} raised:\n{type(inner_exception).__name__}: {inner_exception}"
        super().__init__(msg, first_useful_frame=first_useful_frame)


# NOTE: important invariant! Almost any exception handler that handles Unsupported
# should NOT suppress the exception if skip_frame is set!
# skip_frame is used by symbolic_convert.py to bubble up Unsupported exceptions to convert_frame to cause
# a frame skip. Once the Unsupported exn is in convert_frame, we will always skip, so skip_frame
# won't be checked
class Unsupported(TorchDynamoException):
    # [CLASS DOC] The graph-break exception: Dynamo hit code it cannot trace. Attributes: msg,
    # gb_type (category), skip_frame (if set, handlers must not suppress it so convert_frame
    # skips the frame), real_stack (user stack), category (stats counter name), logged flag.
    # Creating one increments a global stats counter keyed by message.
    def __init__(
        self,
        msg: str,
        # TODO: make this argument required once we remove Unsupported subclasses
        gb_type: str = "",
        skip_frame: bool = False,
        *,
        case_name: str | None = None,
        real_stack: StackSummary | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores message, graph break type and skip flag; if no real_stack is given
        # (or empty) extracts it from the tracing context. Registers the message in the global
        # "unimplemented" stats counter. The case_name argument is accepted but unused.
        super().__init__(msg)
        if not real_stack:
            real_stack = torch._guards.TracingContext.extract_stack()
        self.real_stack = real_stack
        self.msg = msg
        self.skip_frame = skip_frame
        self.category: str | None = None
        self.add_to_stats()
        self.gb_type: str | None = gb_type
        self.logged = False

    def remove_from_stats(self) -> None:
        # [FUNCTION DOC] Decrements the global stats counter for this message under its category,
        # deleting the entry when it drops to zero or below. Raises AssertionError if category
        # is unset.
        if self.category is None:
            raise AssertionError("category must be set before removing from stats")
        counters[self.category][self.msg] -= 1
        if counters[self.category][self.msg] <= 0:
            del counters[self.category][self.msg]

    def add_to_stats(self, category: str = "unimplemented") -> None:
        # [FUNCTION DOC] Sets this exception's category and increments the global stats counter
        # for its message under that category.
        self.category = category
        counters[category][self.msg] += 1


class UnknownPropertiesDuringBackwardTrace(TorchDynamoException):
    # [CLASS DOC] Raised when properties needed while tracing backward are unknown; no extra state.
    pass


class RecompileError(TorchDynamoException):
    # [CLASS DOC] Raised for recompilation-related failures; no extra state.
    pass


class InfiniteGeneratorError(TorchDynamoException):
    # [CLASS DOC] Raised when a generator yields more values than the iterator limit, suggesting
    # it is infinite; no extra state.
    # Raised when the number of yielded values is greater than MAX_ITERATOR_LIMIT
    pass


class CondOpArgsMismatchError(TorchDynamoException):
    # [CLASS DOC] Internal error from the conditional higher-order op when branch arguments mismatch.
    """
    Internal error from cond() due to arguments mismatch.
    """


class UserErrorType(Enum):
    # [CLASS DOC] Enumeration of categories of user errors: dynamic control flow, anti-pattern,
    # standard library use, constraint violation, dynamic dim, invalid input, invalid output,
    # and unsupported aliased-mutated dynamic inputs. Values are auto-assigned.
    DYNAMIC_CONTROL_FLOW = auto()
    ANTI_PATTERN = auto()
    STANDARD_LIBRARY = auto()
    CONSTRAINT_VIOLATION = auto()
    DYNAMIC_DIM = auto()
    INVALID_INPUT = auto()
    INVALID_OUTPUT = auto()
    UNSUPPORTED_ALIASED_MUTATED_DYNAMIC_INPUTS = auto()


class UserError(TorchDynamoException):
    # [CLASS DOC] Error for code that is valid in eager but unsupported by Dynamo. Attributes:
    # error_type, msg/message (actionable text, possibly with an exportdb link), real_stack,
    # skip_frame (always False), logged flag.
    def __init__(
        self, error_type: UserErrorType, msg: str, case_name: str | None = None
    ) -> None:
        # [FUNCTION DOC] If case_name is given (must be a string, else AssertionError), appends
        # the exportdb link to msg (after a space if msg ends with a period, else after a
        # newline). Initializes base with the message and records the user stack from the tracing
        # context, error type and message attributes.
        """
        Type of errors that would be valid in Eager, but not supported in TorchDynamo.
        The error message should tell user about next actions.

        error_type: Type of user error
        msg: Actionable error message
        case_name: (Optional) Unique name (snake case) for the usage example in exportdb.
        """
        if case_name is not None:
            if not isinstance(case_name, str):
                raise AssertionError(f"case_name must be a str, got {type(case_name)}")
            if msg.endswith("."):
                msg += " "
            else:
                msg += "\n"
            msg += exportdb_error_message(case_name)
        super().__init__(msg)
        self.real_stack = torch._guards.TracingContext.extract_stack()
        self.skip_frame = False
        self.logged = False
        self.error_type = error_type
        self.msg = msg
        self.message = msg


# debug exception thrown when tracing torch._dynamo.step_unsupported()
class StepUnsupported(TorchDynamoException):
    # [CLASS DOC] Debug exception raised when tracing the step-unsupported marker. Attributes:
    # msg, real_stack, logged flag.
    def __init__(self, msg: str, real_stack: StackSummary | None = None) -> None:
        # [FUNCTION DOC] Stores msg, the user stack (extracted from the tracing context if not
        # given or empty) and logged=False.
        super().__init__(msg)
        self.msg = msg
        if not real_stack:
            real_stack = torch._guards.TracingContext.extract_stack()
        self.real_stack = real_stack
        self.logged = False


class UnsafeScriptObjectError(TorchDynamoException):
    # [CLASS DOC] Raised when a script object is used unsafely during tracing; no extra state.
    pass


class UncapturedHigherOrderOpError(TorchDynamoException):
    # [CLASS DOC] Raised when a higher-order op body could not be captured. Attributes: msg,
    # real_stack.
    def __init__(self, msg: str, real_stack: StackSummary | None = None) -> None:
        # [FUNCTION DOC] Stores msg and the user stack (taken from the tracing context if None).
        super().__init__(msg)
        self.msg = msg
        self.real_stack = (
            real_stack
            if real_stack is not None
            else torch._guards.TracingContext.extract_stack()
        )


# TODO: I'm a little uncertain about what error classification we should have
# for this.  This is potentially a user error, but regressions in
# specialization in PyTorch proper could also trigger this problem
class FailOnRecompileLimitHit(Exception):
    # [CLASS DOC] Plain Exception (not a Dynamo exception) raised when the recompile limit is hit
    # and failing is requested; no extra state.
    pass


class PackageError(TorchDynamoException):
    # [CLASS DOC] Raised for compile package errors; no extra state.
    pass


class ObservedException(TorchDynamoException):
    # [CLASS DOC] Base for exceptions raised by traced user code and observed by Dynamo while
    # tracing, so that try/except in compiled code can be simulated. Attribute: real_stack.
    # An exception observed during the tracing. This exception is used by Dynamo to handle exceptions.
    def __init__(
        self, *args: Any, real_stack: StackSummary | None = None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Initializes the base with args/kwargs and stores the user stack (taken
        # from the tracing context when not provided).
        super().__init__(*args, **kwargs)
        self.real_stack: StackSummary = (
            real_stack
            if real_stack is not None
            else torch._guards.TracingContext.extract_stack()
        )


class ObservedUserStopIteration(ObservedException):
    # [CLASS DOC] Observed stop-iteration signal (e.g. while tracing a next method). Attribute:
    # value (the first constructor argument, or None), mirroring the value of a normal
    # stop-iteration.
    # An UserStopIteration exception observed during the Dynamo tracing (e.g Dynamo tracing __next__)
    value: Any | None

    # Reference `StopIteration_init` in CPython
    # https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L568-L584
    def __init__(
        self, *args: Any, real_stack: StackSummary | None = None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Initializes the base with a fixed "unhandled raise" message and the
        # stack; sets value to the first positional argument if any, else None. Other
        # arguments and kwargs are ignored.
        super().__init__("unhandled `raise StopIteration`", real_stack=real_stack)
        if len(args) > 0:
            self.value = args[0]
        else:
            self.value = None


class ObservedLookupError(ObservedException):
    # [CLASS DOC] Observed lookup error; no extra state.
    # A LookupError exception to be raised from inside Dynamo tracing. This can happen on __getitem__
    pass


class ObservedIndexError(ObservedLookupError):
    # [CLASS DOC] Observed index error; no extra state.
    # An IndexError exception to be raised from inside Dynamo tracing. This can happen on list __getitem__
    pass


class ObservedKeyError(ObservedLookupError):
    # [CLASS DOC] Observed key error; no extra state.
    # A KeyError exception to be raised from inside Dynamo tracing. This can happen on dict __getitem__
    pass


class ObservedGeneratorExit(ObservedException):
    # [CLASS DOC] Observed generator-exit signal; no extra state.
    pass


class ObservedAttributeError(ObservedException):
    # [CLASS DOC] Observed attribute error; no extra state.
    # An AttributeError exception to be raised from inside Dynamo tracing. This can happen on user defined object __getattr__
    pass


class ObservedRuntimeError(ObservedException):
    # [CLASS DOC] Observed runtime error; no extra state.
    # A RuntimeError exception to be raised from inside Dynamo tracing. This can happen on generator.throw(..) method
    pass


class ObservedNotImplementedError(ObservedException):
    # [CLASS DOC] Observed not-implemented error; no extra state.
    pass


class ObservedTypeError(ObservedException):
    # [CLASS DOC] Observed type error; no extra state.
    # A TypeError exception to be raised from inside Dynamo tracing. This can happen on generator.send(..) method
    pass


class FakeTensorObservedException(ObservedException):
    # [CLASS DOC] Observed exception originating from fake tensor propagation; no extra state.
    pass


observed_exception_map = {
    StopIteration: ObservedUserStopIteration,
    LookupError: ObservedLookupError,
    IndexError: ObservedIndexError,
    GeneratorExit: ObservedGeneratorExit,
    KeyError: ObservedKeyError,
    AttributeError: ObservedAttributeError,
    RuntimeError: ObservedRuntimeError,
    NotImplementedError: ObservedNotImplementedError,
    TypeError: ObservedTypeError,
}


class UnhandledDescriptorError(NotImplementedError):
    # [CLASS DOC] Not-implemented error raised by the generic attribute getter when a descriptor
    # type is not recognized; distinguishable from other not-implemented errors.
    """Raised by object_generic_getattr when a descriptor type is not
    recognized by _resolve_descriptor_get.  Subclasses NotImplementedError
    so callers that catch NotImplementedError (e.g., generic_getattr's
    graph-break fallback) still work, but callers that want to
    distinguish unhandled descriptors from other NotImplementedErrors can
    catch this specifically.
    """


class TritonUnavailableError(RuntimeError):
    # [CLASS DOC] Runtime error signaling that a device cannot run Triton; lets callers react to
    # that case specifically without swallowing unrelated runtime errors.
    """
    Raised by DeviceInterface.raise_if_triton_unavailable to signal that a
    device cannot run Triton (e.g. no Triton backend was built for it).

    Subclasses RuntimeError so existing callers that catch RuntimeError keep
    working, while callers that only want to react to Triton unavailability -
    such as has_triton() - can catch this specific type instead of swallowing
    every RuntimeError, which would hide unrelated bugs.
    """


def get_dynamo_observed_exception(exc_type: type[Exception]) -> type[ObservedException]:
    # [FUNCTION DOC] Returns the observed-exception class for a builtin/user exception type. If
    # the type is not in the global map, dynamically creates a new subclass of the observed
    # base named after it and adds it to the map (global mutation).
    if exc_type not in observed_exception_map:
        name = getattr(exc_type, "__name__", str(exc_type))
        observed_exception_map[exc_type] = type(  # type: ignore[assignment]
            f"Observed{name}Error", (ObservedException,), {}
        )
    # pyrefly: ignore [bad-index]
    return observed_exception_map[exc_type]


def raise_observed_exception(
    exc_type: type[Exception],
    tx: InstructionTranslatorBase,
    *,
    args: list[VariableTracker] | list[str] | None = None,
    kwargs: dict[str, VariableTracker] | None = None,
) -> NoReturn:
    # [FUNCTION DOC] Simulates raising a Python exception inside traced code. Converts string
    # args to symbolic constants, constructs the exception object symbolically by calling the
    # exception type with args and kwargs, then asks the translator to raise it (which sets up
    # the symbolic exception stack). Never returns normally.
    from .variables.builder import SourcelessBuilder

    if args:
        args_ = [
            SourcelessBuilder.create(tx, arg) if isinstance(arg, str) else arg
            for arg in args
        ]
    else:
        args_: list[VariableTracker] = []

    # CPython here raises an exception. Since there is no python code, we have to manually setup the exception
    # stack and raise the exception.
    exception_vt = SourcelessBuilder.create(tx, exc_type).call_function(
        tx, args_, kwargs or {}
    )
    tx.do_raise(exception_vt, None)


def raise_type_error(tx: InstructionTranslatorBase, msg: str) -> NoReturn:
    # [FUNCTION DOC] Raises a type error with message msg as an observed exception in tracing.
    """Raise a TypeError as an observed exception during tracing."""
    raise_observed_exception(TypeError, tx, args=[msg])


def raise_value_error(tx: InstructionTranslatorBase, msg: str) -> NoReturn:
    # [FUNCTION DOC] Raises a value error with message msg as an observed exception in tracing.
    """Raise a ValueError as an observed exception during tracing."""
    raise_observed_exception(ValueError, tx, args=[msg])


def handle_observed_exception(tx: Any) -> None:
    # [FUNCTION DOC] Marks the current symbolic exception as handled by clearing it from the
    # translator's exception stack (equivalent of an except clause that does nothing). Used
    # where CPython handles the exception implicitly, e.g. in loop iteration.
    # This is essentially exception handling code, equivalent of this pseudo code
    #
    # try:
    #     ... somebody raising StopIteration
    # except StopIteration
    #     pass
    #
    # If this was going through the python code, we would have called exception_handler method, but FOR_ITER
    # handles the exception completely in CPython. For example for 3.11, the resulting bytecode is
    #
    #
    #   6          46 LOAD_GLOBAL              2 (StopIteration)
    #              58 RAISE_VARARGS            1
    #         >>   60 PUSH_EXC_INFO

    #   7          62 LOAD_GLOBAL              2 (StopIteration)
    #              74 CHECK_EXC_MATCH
    #              76 POP_JUMP_FORWARD_IF_FALSE     3 (to 84)
    #              78 POP_TOP

    #   8          80 POP_EXCEPT
    #

    # Fortunately this translates to a simple pop from the exn_vt_stack
    tx.exn_vt_stack.clear_current_exception()


# These exceptions are ok to fallback to eager/graph_break.
exceptions_allowed_to_be_fallback = (
    torch._subclasses.fake_tensor.DataDependentOutputException,
    torch._subclasses.fake_tensor.DynamicOutputShapeException,
    torch._subclasses.fake_tensor.UnsupportedOperatorException,
    torch._subclasses.fake_tensor.UnsupportedFakeTensorException,
    torch._subclasses.fake_tensor.UnsupportedMutationAliasingException,
)


def unimplemented_with_warning(
    e: Exception,
    code: types.CodeType,
    *,
    gb_type: str,
    context: str,
    explanation: str,
    hints: list[str],
) -> NoReturn:
    # [FUNCTION DOC] Like the graph-break helper but first logs the formatted verbose error for
    # exception e and code as a structured-trace artifact and a debug log entry, and then
    # raises the graph-break exception chained from e with a logged warning. Always raises.
    # This function calls unimplemented internally and eventually graph breaks
    # or falls to eager. unimplemented itself does not print any user warnings,
    # i.e., its very silent. This helper function is intended when an error is
    # encountered in the torch.compile stack which is worth showing as warning
    # to the user. For example, if AOT Autograd backend fails with a fake tensor
    # exception, its ok to fallback to eager but not silently. Here, we can use
    # this function to log the message and the stack trace.
    graph_break_msg = format_error_msg_verbose(e, code)
    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "dynamo_graph_break_reason",
            "encoding": "string",
        },
        payload_fn=lambda: graph_break_msg,
    )
    graph_breaks_log.debug("%s", graph_break_msg)
    _unimplemented = unimplemented
    # to prevent a graph break registry entry
    _unimplemented(
        gb_type=gb_type,
        context=context,
        explanation=explanation,
        hints=hints,
        from_exc=e,
        log_warning=True,
    )


def format_graph_break_message(
    gb_type: str,
    context: str,
    explanation: str,
    hints: list[str],
) -> str:
    # [FUNCTION DOC] Builds the multi-line graph break message: type line, indented explanation,
    # one hint line per hint, developer context, and (if the type is in the registry) a link to
    # documentation. Pure string formatting.
    explanation = textwrap.indent(explanation, "    ").lstrip()
    hints_str = "\n".join(
        "  Hint: " + textwrap.indent(hint, "    ").lstrip() for hint in hints
    )
    context = textwrap.indent(context, "    ").lstrip()

    msg = f"""\
{gb_type}
  Explanation: {explanation}
{hints_str}

  Developer debug context: {context}"""
    documentation_link = get_gbid_documentation_link(gb_type)

    if documentation_link:
        msg += f"\n\n For more details about this graph break, please visit: {documentation_link}"

    return msg


@lru_cache(maxsize=1)
def _load_gb_type_to_gb_id_map() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Cached (single entry) loader that reads the graph break registry JSON
    # next to this module and returns a mapping from every recorded graph break type (including
    # historical ones) to its registry id. On any read/parse error, logs the exception and uses
    # an empty registry. File I/O on first call only.
    """
    Loads the gb_type to gb_id map from the graph break registry from JSON file with caching.

    Includes historical gb_type (mapping behavior of duplicate gb_types with different gb_ids is undefined).
    """
    try:
        script_dir = Path(__file__).resolve().parent
        registry_path = get_file_path_2(
            "", str(script_dir), "graph_break_registry.json"
        )
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception:
        log.exception("Error accessing the registry file")
        # pyrefly: ignore [implicit-any]
        registry = {}

    mapping = {}
    for k, v in registry.items():
        for entry in v:
            mapping[entry["Gb_type"]] = k

    return mapping


def get_gbid_documentation_link(gb_type: str) -> str | None:
    # [FUNCTION DOC] Returns the documentation page URL for a graph break type (built from the
    # registry id with its GB prefix removed), or None if the type is not in the registry.
    """
    Retrieves the GBID documentation link for a given graph break type.

    Args:
        gb_type: The graph break type to look up.

    Returns:
        A string containing the documentation URL if found, otherwise None.
    """
    GRAPH_BREAK_SITE_URL = (
        "https://meta-pytorch.github.io/compile-graph-break-site/gb/"  # @lint-ignore
    )

    gb_type_to_gb_id_map = _load_gb_type_to_gb_id_map()

    if gb_type in gb_type_to_gb_id_map:
        return (
            f"{GRAPH_BREAK_SITE_URL}gb{gb_type_to_gb_id_map[gb_type].lstrip('GB')}.html"
        )

    return None


_NOTHING = object()


def unimplemented(
    *,
    gb_type: str,
    context: str,
    explanation: str,
    hints: list[str],
    from_exc: Any = _NOTHING,
    log_warning: bool = False,
    skip_frame: bool = False,
) -> NoReturn:
    # [FUNCTION DOC] Raises the graph-break exception. Formats the message from gb_type, context,
    # explanation and hints; optionally logs it as a warning. If from_exc is given: when it is
    # itself a graph break, the new message is prefixed by the old one and raised without
    # chaining; otherwise raised chained from it; the previous real stack is reused if
    # present. skip_frame is passed through. Always raises.
    """
    Called within dynamo to cause a graph break.
    Args:
        gb_type: Context-free graph break type. It should be a short string without any
                 information specific to the tracing context (i.e. no dynamically-generated strings)
        context: Developer context for the graph break. It can contain tracing context/dynamic strings.
        explanation: User-facing context-dependent explanation for the graph break. Can be dynamic.
        hints: List of user-facing hints for the graph break.
    """

    msg = format_graph_break_message(gb_type, context, explanation, hints)

    if log_warning:
        log.warning(msg)
    if from_exc is not _NOTHING:
        past_real_stack = None
        if hasattr(from_exc, "real_stack"):
            past_real_stack = from_exc.real_stack
        if isinstance(from_exc, Unsupported):
            msg = f"{from_exc.msg}\n\n*** While handling this graph break, another graph break occurred: ***\n\n{msg}"
            # noqa: GB_REGISTRY
            raise Unsupported(msg, gb_type, skip_frame, real_stack=past_real_stack)
        # noqa: GB_REGISTRY
        raise Unsupported(
            msg, gb_type, skip_frame, real_stack=past_real_stack
        ) from from_exc
    # noqa: GB_REGISTRY
    raise Unsupported(msg, gb_type, skip_frame)


# KeyError has special handling for its args
# see https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L2534 for details
class KeyErrorMsg:
    # [CLASS DOC] Wrapper letting a plain-text message be used as a key error argument without
    # the extra quoting a key error normally applies; both str and repr give the plain text.
    def __init__(self, value: Any) -> None:
        # [FUNCTION DOC] Stores the wrapped value.
        self.value = value

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the text of the wrapped value.
        return str(self.value)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Same as str: returns the plain text of the wrapped value.
        return self.__str__()


def augment_exc_message_with_hop_name(exc: Exception, msg: str) -> str:
    # [FUNCTION DOC] If exc has a higher-order-op name attribute, inserts a line naming it in msg,
    # right before the explanation section if present, else at the end of the first line
    # portion (before the partition remainder). Returns the possibly modified message.
    # Add HOP context right after before the explanation if present;
    # otherwise after the message
    if hasattr(exc, "_hop_name"):
        lines = msg.partition("\n  Explanation:")
        msg = (
            f"{lines[0]}\n  Higher Order Operator: {exc._hop_name}{lines[1]}{lines[2]}"  # type: ignore[attr-defined]
        )

    return msg


def augment_exc_message(exc: Exception, msg: str = "\n", export: bool = False) -> None:
    # [FUNCTION DOC] Mutates exc in place to enrich its first argument (message): appends the
    # user code stack, the replay file hint (if enabled and recorded), a verbose-mode hint (when
    # not verbose and exc has a real stack), and minifier info from an inner exception. Also
    # sets innermost_user_frame_summary on exc (last real stack frame or None) and applies the
    # higher-order-op name. For key errors the message is wrapped so it is not re-quoted.
    # The export argument is unused.
    import traceback

    exc.innermost_user_frame_summary = None  # type: ignore[attr-defined]

    real_stack = get_real_stack(exc)
    if real_stack is not None and len(real_stack) > 0:
        exc.innermost_user_frame_summary = real_stack[-1]  # type: ignore[attr-defined]
        msg += f"\nfrom user code:\n {''.join(traceback.format_list(real_stack))}"

    if config.replay_record_enabled and hasattr(exc, "record_filename"):
        msg += (
            f"\nLast frame execution written to {exc.record_filename}. To run only this frame while debugging, run\
 torch._dynamo.replay('{exc.record_filename}').\n"
        )

    if not config.verbose and hasattr(exc, "real_stack"):
        msg += (
            "\nSet TORCHDYNAMO_VERBOSE=1 for the internal stack trace "
            "(please do this especially if you're reporting a bug to PyTorch). "
            'For even more developer context, set TORCH_LOGS="+dynamo"\n'
        )

    if hasattr(exc, "inner_exception") and hasattr(
        exc.inner_exception, "minifier_path"
    ):
        if hasattr(exc.inner_exception, "buck_command"):
            msg += (
                f"\nMinifier script written to {exc.inner_exception.minifier_path}. Run "
                f"this buck command to find the smallest traced graph "
                f"which reproduces this error: {exc.inner_exception.buck_command}\n"
            )
        else:
            msg += (
                f"\nMinifier script written to {exc.inner_exception.minifier_path}. Run "
                "this script to find the smallest traced graph which reproduces this error.\n"
            )

    old_msg = "" if len(exc.args) == 0 else str(exc.args[0])

    old_msg = augment_exc_message_with_hop_name(exc, old_msg)

    if isinstance(exc, KeyError):
        exc.args = (KeyErrorMsg(old_msg + msg),) + exc.args[1:]
    else:
        new_msg = old_msg + msg
        exc.args = (new_msg,) + exc.args[1:]


def get_exc_message(
    e: Exception, compile_id: CompileId
) -> tuple[str | None, int | None]:
    # [FUNCTION DOC] Returns (filename, line number) of the exception's innermost user frame, or
    # (None, None). Also sets the compile_id attribute on e. Requires that e already has the
    # innermost user frame attribute (set by the message augmenter), else AttributeError.
    filename = None
    lineno = None
    if e.innermost_user_frame_summary is not None:  # type: ignore[attr-defined]
        filename = e.innermost_user_frame_summary.filename  # type: ignore[attr-defined]
        lineno = e.innermost_user_frame_summary.lineno  # type: ignore[attr-defined]
    e.compile_id = compile_id  # type: ignore[attr-defined]
    return filename, lineno


def _extract_stack_with_positions() -> StackSummary:
    # [FUNCTION DOC] HELPER: Captures the current call stack (excluding this function) as a stack
    # summary ordered outermost first, filling in column positions for each frame where the
    # interpreter provides them (3.11+), so caret annotations can be rendered. No source lines
    # are looked up.
    # traceback.extract_stack() does not populate colno/end_colno on
    # FrameSummary, even though the data is available on live frames via
    # f_lasti + co_positions(). This means traceback.format_list() never
    # renders caret annotations for stacks captured this way. We do it
    # manually here so that "stack above dynamo" frames get carets.
    stack = StackSummary()
    frame = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        # colno/end_colno kwargs were added to FrameSummary in 3.11
        kwargs: dict[str, Any] = {}
        if sys.version_info >= (3, 11) and frame.f_lasti >= 0:
            positions = list(code.co_positions())
            idx = frame.f_lasti // 2
            if idx < len(positions):
                _, _, kwargs["colno"], kwargs["end_colno"] = positions[idx]
        stack.append(
            FrameSummary(
                code.co_filename,
                frame.f_lineno,
                code.co_name,
                lookup_line=False,
                **kwargs,
            )
        )
        frame = frame.f_back
    stack.reverse()
    return stack


def get_stack_above_dynamo() -> StackSummary:
    # [FUNCTION DOC] Returns the current stack with Dynamo-internal frames filtered out (the user
    # frames that called into Dynamo).
    return filter_stack(_extract_stack_with_positions())


def get_real_stack(
    exc: Exception, frame: DynamoFrameType | None = None
) -> StackSummary | None:
    # [FUNCTION DOC] Returns the exception's real stack, or None if it has none. If a frame is
    # supplied, prepends the filtered stack above Dynamo (taken from the current stack, not from
    # the frame itself). An empty real stack is still reported with the prefix.
    real_stack = getattr(exc, "real_stack", None)
    if real_stack is None:
        return None

    # NB: it's possible for real_stack to be []; we still attempt to
    # report a stack anyway because the stack_above_dynamo may still
    # be useful for debugging

    if frame is not None:
        # NB: frame is PyInterpreterFrame on Python 3.11 and later,
        # not a TRUE frame object.  You can't actually feed it
        # to traceback because it doesn't have enough information.
        # To solve this problem, we technically should just materialize
        # the frame, the same way _PyFrame_GetFrameObject would do
        # (but we cannot actually do this, because this populates
        # frame_obj field, which default eval frame doesn't like).
        #
        # Fortunately, in this case, we can hack it: there's no need
        # to actually use the truly top frame, we can just extract
        # from where we are right now and rely on filter_stack to
        # get rid of all the dynamo frames.  For ease of testing
        # we apply this behavior to ALL Python versions
        stack_above_dynamo = get_stack_above_dynamo()
    else:
        stack_above_dynamo = StackSummary()

    return StackSummary.from_list(stack_above_dynamo + real_stack)


# filter out all frames after entering dynamo
def filter_stack(stack: StackSummary) -> StackSummary:
    # [FUNCTION DOC] Returns the frames of stack before the first frame from the frame converter
    # file, skipping frames without a filename, frames from the eval-frame file and frames whose
    # source line is the legacy optimize call. Pure.
    user_stack = StackSummary()
    for frame in stack:
        if frame.filename is None:
            continue
        if "convert_frame" in frame.filename:
            break
        if "eval_frame" in frame.filename or (
            frame.line and "torch._dynamo.optimize(" in frame.line
        ):
            continue
        user_stack.append(frame)

    return user_stack


def remove_resume_prefix(name: str) -> str:
    # [FUNCTION DOC] If name looks like a resume function name (resume prefix, original name,
    # and an "at" offset), returns the original function name; otherwise returns name.
    from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

    match = re.match(f"{TORCH_DYNAMO_RESUME_IN_PREFIX}_(\\w+)_at_\\d+", name)
    if match:
        return match.group(1)
    return name


def collapse_resume_frames(stack: StackSummary | list[FrameSummary]) -> StackSummary:
    # [FUNCTION DOC] Returns a new stack where resume-function frames are renamed to the original
    # function name and consecutive frames with the same name are collapsed into the latest
    # one. Frames without a filename are dropped. Mutates the name of the frames it keeps.
    """
    When we graph break, we create a resume function and make a regular Python call
    to it, which gets intercepted by Dynamo. This behavior is normally shown in the
    traceback, which can be confusing to a user. So we can filter out resume frames
    for better traceback clarity.

    Example:
    File "..." line 3, in f
        <line 3>
    File "..." line 5, in torch_dynamo_resume_in_f_at_80
        <line 5>
    File "..." line 10, in torch_dynamo_resume_in_f_at_120
        <line 10>

    becomes
    File "..." line 10, in f
        <line 10>
    """

    new_stack = StackSummary()
    for frame in stack:
        if frame.filename is None:
            continue
        name = remove_resume_prefix(frame.name)
        if new_stack and name and new_stack[-1].name == name:
            new_stack[-1] = frame
            frame.name = name
        else:
            frame.name = name
            new_stack.append(frame)

    return new_stack


def format_error_msg_verbose(
    exc: Exception,
    code: types.CodeType,
    record_filename: str | None = None,
    frame: DynamoFrameType | None = None,
) -> str:
    # [FUNCTION DOC] Builds a verbose report for a failed conversion: header with code name,
    # file and first line, the Dynamo traceback of the exception being handled, and (if
    # available) the real user stack. The record_filename argument is unused.
    msg = (
        f"WON'T CONVERT {code.co_name} {code.co_filename} line {code.co_firstlineno}\n"
    )
    msg += "=" * 10 + " TorchDynamo Stack Trace " + "=" * 10 + "\n"
    msg += format_exc()
    real_stack = get_real_stack(exc, frame)
    if real_stack is not None:
        msg += (
            "\n"
            + "=" * 10
            + " The above exception occurred while processing the following code "
            + "=" * 10
            + "\n\n"
        )
        msg += "".join(format_list(real_stack))
        msg += "\n"
        msg += "=" * 10

    return msg


def format_frame_info(code: types.CodeType) -> str:
    # [FUNCTION DOC] Returns a one-line description of a code object: name, file and first line,
    # using placeholders/defaults for missing attributes.
    return (
        f"{getattr(code, 'co_name', '<unknown>')} "
        f"({getattr(code, 'co_filename', '<unknown>')} "
        f"line {getattr(code, 'co_firstlineno', 0)})"
    )


def format_skip_frame_message(code: types.CodeType | None, reason: str) -> str:
    # [FUNCTION DOC] Returns a message saying compile intentionally skipped the frame (described
    # by code if given) and fell back to eager, followed by the reason.
    if code is not None:
        frame_info = format_frame_info(code)
        return (
            f"torch.compile intentionally decided to skip the frame {frame_info} and fall back to eager.\n"
            f"Reason: {reason}"
        )
    else:
        return (
            f"torch.compile intentionally decided to skip the frame and fall back to eager.\n"
            f"Reason: {reason}"
        )


def format_error_msg(
    exc: Exception,
    code: types.CodeType,
    record_filename: str | None = None,
    frame: DynamoFrameType | None = None,
) -> str:
    # [FUNCTION DOC] Returns the "won't convert" error text for a failed frame: the verbose
    # report when verbose config is on, else a short message with code name, file, line and the
    # traceback of the exception currently being handled.
    if config.verbose:
        return format_error_msg_verbose(exc, code, record_filename, frame)
    return f"WON'T CONVERT {code.co_name} {code.co_filename}\
 line {code.co_firstlineno} \ndue to: \n{format_exc()}"
