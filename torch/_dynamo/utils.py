# [MODULE DOC]
# Shared utility grab-bag for TorchDynamo: compile-event/metrics logging (CompileEventLogger,
# ChromiumEventLogger, dynamo_timed, CompilationMetrics), global counters and timers, tensor and
# fake-tensor helpers, guard/graph debugging helpers, bytecode/frame helpers, container and
# type-checking helpers, RNG/state save-restore, and misc testing utilities. Import-time side
# effects: creates module-level counters, thread-local state, loggers, optional numpy/torch._numpy
# tables (NP_SUPPORTED_MODULES, NP_TO_TNP_MODULE), and captures unpatched nn.Module methods.
"""
Utility functions and classes used throughout the TorchDynamo system.

This module contains a collection of helper utilities used by various parts of Dynamo for:
- Performance metrics collection and reporting
- Compilation timing and debugging
- Graph manipulation and tensor operations
- Runtime guards and checks
- Common data structure operations
- Testing and development tools

This is an internal module that provides shared functionality used across the Dynamo codebase.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import copy
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import importlib
import inspect
import itertools
import json
import linecache
import logging
import math
import operator
import os
import re
import sys
import textwrap
import threading
import time
import traceback
import types
import typing
import uuid
import warnings
import weakref
from collections import Counter, OrderedDict
from contextlib import AbstractContextManager, contextmanager
from dataclasses import is_dataclass
from functools import lru_cache
from types import CodeType, MethodWrapperType
from typing import (
    Any,
    cast,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    overload,
    TypeAlias,
    TypeGuard,
    TypeVar,
)
from typing_extensions import ParamSpec, TypeIs

import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
import torch.utils._pytree as pytree
from torch import fx
from torch._C import (
    _instruction_counter,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.metrics_context import MetricsContext, RuntimeMetricsContext
from torch._guards import CompileId, Source, TracingContext
from torch._subclasses.meta_utils import is_sparse_compressed
from torch._utils_internal import (
    justknobs_check,
    log_chromium_event_internal,
    log_compilation_event,
    record_chromium_event_internal,
    signpost_event,
)
from torch.fx._utils import _format_graph_code, lazy_format_graph_code
from torch.monitor import _WaitCounter
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._triton import has_triton, has_triton_package
from torch.utils.hooks import RemovableHandle

from .graph_utils import _get_flat_args


if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Container,
        Generator,
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
        Sequence,
        ValuesView,
    )

    from torch._dynamo.bytecode_transformation import Instruction
    from torch._dynamo.replay_record import ExecutionRecord
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._dynamo.variables.base import VariableTracker
    from torch._prims_common import DeviceLikeType
    from torch._subclasses import FakeTensorMode


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    import torch._logging
    import torch._numpy as tnp
    from torch._guards import detect_fake_mode  # noqa: F401
    from torch._logging import LazyString

    from . import config

    # NOTE: Make sure `NP_SUPPORTED_MODULES` and `NP_TO_TNP_MODULE` are in sync.
    if np:
        NP_SUPPORTED_MODULES: tuple[types.ModuleType, ...] = (
            np,
            np.fft,
            np.linalg,
            np.random,
        )

        NP_TO_TNP_MODULE = {
            np: tnp,
            np.fft: tnp.fft,
            np.linalg: tnp.linalg,
            np.random: tnp.random,
        }
    else:
        NP_SUPPORTED_MODULES = ()

        # pyrefly: ignore [implicit-any]
        NP_TO_TNP_MODULE = {}
    from torch._subclasses.fake_tensor import is_fake
except ImportError:
    pass


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
T6 = TypeVar("T6")
T7 = TypeVar("T7")
T8 = TypeVar("T8")
T9 = TypeVar("T9")
T10 = TypeVar("T10")
R = TypeVar("R")
_P = ParamSpec("_P")

unpatched_nn_module_getattr = torch.nn.Module.__getattr__
unpatched_nn_module_call = torch.nn.Module.__call__
unpatched_nn_module_call_impl = torch.nn.Module._call_impl

counters: collections.defaultdict[str, Counter[str]] = collections.defaultdict(
    collections.Counter
)
optimus_scuba_log: dict[str, Any] = {}
troubleshooting_url = "https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html"
nnmodule_doc_url = "https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_nn_module.html"
nnmodule_doc_url_msg = f"See {nnmodule_doc_url} for more information and limitations."
log = logging.getLogger(__name__)

# profiling compilation time by function
compilation_time_metrics: dict[str, list[float]] = {}

# This supports calculate_time_spent(), which reports cumulative times
# across the process for any "phase" populated by dynamo_timed. Reset if
# reset_frame_count() is called.
cumulative_time_spent_ns: dict[str, float] = collections.defaultdict(float)

timer_counter = itertools.count()


# Abstraction on top of counters.
class ReInplaceTrigger(enum.Enum):
    # [CLASS DOC] Enum naming the source of a missed re-inplace opportunity (auto-functionalized
    # v1, v2, or triton ops); its name is used as a suffix in ReinplaceCounters keys.
    AUTO_FUNC_V1 = 1
    AUTO_FUNC_V2 = 2
    TRITON_OPS = 3


class ReinplaceCounters:
    # [CLASS DOC] Class-level (global, never instantiated) counters of re-inplace misses, kept in
    # one defaultdict mapping "missed_bytes_<trigger>" / "missed_tensors_<trigger>" to totals.
    # Methods are classmethods that add, clear, total, and log (via signpost_event) the counts.
    _values: collections.defaultdict[str, int] = collections.defaultdict(int)

    # Track sizes of known not re-inplaced tensors (exclude dynamic shapes).
    @classmethod
    def add_missed_bytes(cls, trigger: ReInplaceTrigger, bytes: int) -> None:
        # [FUNCTION DOC] Adds a nonzero byte count to the missed-bytes counter for the trigger.
        # Mutates class-level state; zero is ignored.
        if bytes != 0:
            cls._values[f"missed_bytes_{trigger.name}"] += bytes

    # Track number of not re-inplaced tensors.
    @classmethod
    def add_missed_opportunities(cls, trigger: ReInplaceTrigger, count: int) -> None:
        # [FUNCTION DOC] Adds a nonzero count of missed re-inplaced tensors to the counter for the
        # trigger. Mutates class-level state; zero is ignored.
        if count != 0:
            cls._values[f"missed_tensors_{trigger.name}"] += count

    @classmethod
    def clear(cls) -> None:
        # [FUNCTION DOC] Resets all re-inplace counters (clears class-level mapping).
        cls._values.clear()

    @classmethod
    def get_total_missed(cls) -> int:
        # [FUNCTION DOC] Returns the sum of missed-tensor counts across all triggers.
        sum = 0
        for trigger in ReInplaceTrigger:
            sum += cls._values.get(f"missed_tensors_{trigger.name}", 0)
        return sum

    @classmethod
    def get_total_missed_bytes(cls) -> int:
        # [FUNCTION DOC] Returns the sum of missed-byte counts across all triggers.
        sum = 0
        for trigger in ReInplaceTrigger:
            sum += cls._values.get(f"missed_bytes_{trigger.name}", 0)
        return sum

    @classmethod
    def log(cls) -> None:
        # [FUNCTION DOC] If any counters are nonempty, emits them as an "inductor"
        # "reinplace_counters" signpost event. No-op when empty.
        # if not empty log.
        if cls._values:
            signpost_event("inductor", "reinplace_counters", cls._values)


def tabulate(
    rows: list[tuple[str, Any]] | list[list[Any]],
    headers: tuple[str, ...] | list[str],
) -> str:
    # [FUNCTION DOC] Formats rows with headers as a table string using the optional tabulate
    # package; if it is not installed, falls back to comma-separated lines (headers first).
    try:
        import tabulate

        return tabulate.tabulate(rows, headers=headers)
    except ImportError:
        return "\n".join(
            ", ".join(map(str, row)) for row in itertools.chain([headers], rows)
        )


curr_frame = 0


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def increment_frame() -> None:
    # [FUNCTION DOC] Increments the global frame counter curr_frame by one (mutates module state).
    global curr_frame
    curr_frame = curr_frame + 1


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def reset_frame_count() -> None:
    # [FUNCTION DOC] Resets the frame counter to zero and clears cumulative phase times and
    # per-function compilation time metrics (module-level state).
    global curr_frame
    cumulative_time_spent_ns.clear()
    compilation_time_metrics.clear()
    curr_frame = 0


_recompile_user_contexts: list[Callable[[], str]] | None = None


def register_hook_for_recompile_user_context(hook: Callable[[], str]) -> None:
    # [FUNCTION DOC] Appends a hook (no-arg callable returning a string) to the global list of
    # recompile user-context hooks, creating the list on first use.
    """
    Register a hook to be called when a recompile is triggered. The hook
    should return a string describing user contexts that are not available
    to the compiler, such as the current training epoch. This is useful for
    debugging and data analysis for recompile. For data retention purposes,
    the user context string is capped at 256 characters.
    """
    global _recompile_user_contexts
    if _recompile_user_contexts is None:
        _recompile_user_contexts = []
    _recompile_user_contexts.append(hook)


def get_hook_for_recompile_user_context() -> list[Callable[[], str]] | None:
    # [FUNCTION DOC] Returns the registered recompile user-context hooks, or None if none were
    # ever registered (or after a reset).
    return _recompile_user_contexts


def reset_recompile_user_contexts() -> None:
    # [FUNCTION DOC] HELPER: Sets the global hook list back to None (test helper).
    """Clear any registered recompile user-context hooks (test helper)."""
    global _recompile_user_contexts
    _recompile_user_contexts = None


op_count = 0


def increment_op_count(cnt: int) -> None:
    # [FUNCTION DOC] Adds cnt to the global op_count (total traced ops).
    global op_count
    op_count += cnt


# Get the total time in seconds for each "phase"
# For example, {'entire_frame_compile':8.574629999999999, 'backend_compile':5.26806}
def calculate_time_spent() -> dict[str, float]:
    # [FUNCTION DOC] Converts cumulative per-phase nanosecond times to seconds and adds a
    # "total_wall_time" entry equal to entire_frame_compile plus entire_backward_compile (missing
    # phases count as zero). Returns the new mapping; does not mutate state.
    total_by_key = {}
    for phase, timing in cumulative_time_spent_ns.items():
        total_by_key[phase] = timing / 1e9

    total_by_key["total_wall_time"] = total_by_key.get(
        "entire_frame_compile", 0
    ) + total_by_key.get("entire_backward_compile", 0)
    return total_by_key


# Print a report of time spent so far
# Ex:
# TIMING:
# entire_frame_compile:8.574629999999999
# backend_compile:5.26806
def print_time_report() -> None:
    # [FUNCTION DOC] Prints one line "TIMING:" followed by each phase and its seconds (rounded to
    # 5 places) from calculate_time_spent. Side effect: writes to stdout.
    total_by_key = calculate_time_spent()

    out = "TIMING:"
    for key, value in total_by_key.items():
        out = f"{out} {key}:{round(value, 5)}"

    print(out)


# Use the following singleton to capture and log CompilationMetrics. Entering the context
# manager allocates a new record to be logged when it exits. (You should not need to use
# this directly unless you introduce a new code path where compilation metrics would be
# gathered). While compiling, use the setters or timer in MetricsContext to update fields
# in the current context. For example:
#
# To set a single field once (use overwrite=True to overwrite):
#   get_metrics_context().set("metric_name", value)
#
# To set multiple fields at once (use overwrite=True to overwrite):
#   get_metrics_context().update({"name1": val1, "name2": val2})
#
# To increment an integer field:
#   get_metrics_context().increment("metric_name", value)
#
# To record execution time, MetricsContext works with dynamo_timed:
#    def foo(...):
#        # Updates the "metric_us" field.
#        with dynamo_timed("metric", dynamo_compile_column_us="metric_us")
#            ...
#
_metrics_context_tls = threading.local()


def get_metrics_context() -> MetricsContext:
    # [FUNCTION DOC] Returns the thread-local MetricsContext, lazily creating it with
    # record_compilation_metrics as its on-exit callback.
    if not hasattr(_metrics_context_tls, "metrics_context"):
        _metrics_context_tls.metrics_context = MetricsContext(
            on_exit=record_compilation_metrics
        )
    return _metrics_context_tls.metrics_context


def get_runtime_metrics_context() -> RuntimeMetricsContext:
    # [FUNCTION DOC] Returns the thread-local RuntimeMetricsContext, lazily creating it with
    # record_compilation_metrics as its on-exit callback.
    if not hasattr(_metrics_context_tls, "runtime_metrics_context"):
        _metrics_context_tls.runtime_metrics_context = RuntimeMetricsContext(
            on_exit=record_compilation_metrics
        )
    return _metrics_context_tls.runtime_metrics_context


class CompileEventLogLevel(enum.Enum):
    # [CLASS DOC] Enum of increasingly broad logging destinations: CHROMIUM (tlparse only),
    # PT2_COMPILE (plus PT2 Compile Events), COMPILATION_METRIC (plus dynamo_compile).
    """
    Enum that loosely corresponds with a "log level" of a given event.

    CHROMIUM_EVENT: Logs only to tlparse.
    COMPILE_EVENT: Logs to tlparse + PT2 Compile Events
    COMPILATION_METRIC: Logs to tlparse, PT2 Compile Events, and dynamo_compile
    """

    CHROMIUM = 1
    PT2_COMPILE = 2
    COMPILATION_METRIC = 3


class CompileEventLogger:
    # [CLASS DOC] Stateless namespace of static methods for attaching metadata to compile events
    # at a chosen CompileEventLogLevel. Delegates to the global chromium event logger and the
    # thread-local metrics context; COMPILATION_METRIC writes require the toplevel event and an
    # in-progress metrics context, else RuntimeError.
    """
    Helper class for representing adding metadata(i.e. columns) to various compile events.
    Use CompileEventLogger to add event data to:
    - Chromium events
    - PT2 Compile Events
    - CompilationMetrics

    This should be used in conjunction with dynamo_timed() and metrics contexts, which create
    timed spans and events. CompileEventLogger uses three log levels (described in CompileEventLogLevel),
    where each log level logs to all sources below it in the hierarchy.

    Example usages:
    - I want to log to an existing chromium event within dynamo timed:
    with dynamo_timed("my_event"):
        CompileEventLogger.chromium("my_event", foo=bar)

    - I want to log my event to both chromium + pt2_compile_events:
    with dynamo_timed("my_event", log_pt2_compile_event=True):
        CompileEventLogger.pt2_compile("my_event", foo=bar)

    - I want to add information to dynamo events and dynamo_compile
        CompileEventLogger.compilation_metric(foo=bar)
    """

    @staticmethod
    def log_instant_event(
        event_name: str,
        metadata: dict[str, Any],
        time_ns: int | None = None,
        log_level: CompileEventLogLevel = CompileEventLogLevel.CHROMIUM,
    ) -> None:
        # [FUNCTION DOC] Logs an instant event with metadata at time_ns (default now) to the
        # chromium logger; PT2_COMPILE level also logs to PT2 compile events. Raises RuntimeError
        # for COMPILATION_METRIC level.
        if time_ns is None:
            time_ns = time.time_ns()
        chromium_log = get_chromium_event_logger()
        if log_level == CompileEventLogLevel.CHROMIUM:
            log_pt2_compile_event = False
        elif log_level == CompileEventLogLevel.PT2_COMPILE:
            log_pt2_compile_event = True
        else:
            raise RuntimeError(
                "Cannot log instant event at COMPILATION_METRIC level. Please choose one of CHROMIUM_EVENT or COMPILE_EVENT"
            )
        chromium_log.log_instant_event(
            event_name, time_ns, metadata, log_pt2_compile_event
        )

    @staticmethod
    def add_data(
        event_name: str,
        log_level: CompileEventLogLevel,
        overwrite: bool = False,
        **metadata: object,
    ) -> None:
        # [FUNCTION DOC] Adds metadata to the named event according to log_level. PT2_COMPILE
        # requires the event to be in the pt2_compile substack; COMPILATION_METRIC requires the
        # event to be the outermost one and a metrics context in progress, and also updates the
        # metrics context (overwrite flag). Raises RuntimeError on violations.
        # KNOWN ISSUE: the PT2_COMPILE error passes the event name as a second argument rather
        # than formatting it into the message.
        """
        Centralized API for adding data to various events
        Log an event to a toplevel "dynamo" event or metrics context
        depending on log level.
        """
        chromium_log = get_chromium_event_logger()
        pt2_compile_substack = chromium_log.get_pt2_compile_substack()

        if log_level == CompileEventLogLevel.CHROMIUM:
            chromium_log.add_event_data(event_name, **metadata)
        elif log_level == CompileEventLogLevel.PT2_COMPILE:
            pt2_compile_substack = chromium_log.get_pt2_compile_substack()
            if event_name not in pt2_compile_substack:
                raise RuntimeError(
                    "Error: specified log level PT2_COMPILE, but the event %s"
                    " is not logged to pt2_compile_events. Make sure the event is active and you passed "
                    "log_pt2_compile_event=True to dynamo_timed",
                    event_name,
                )
            chromium_log.add_event_data(event_name, **metadata)
        else:
            if log_level != CompileEventLogLevel.COMPILATION_METRIC:
                raise AssertionError(
                    f"Expected log_level to be COMPILATION_METRIC, got {log_level}"
                )
            top_event = chromium_log.get_outermost_event()

            if event_name != top_event:
                raise RuntimeError(
                    "Log level is COMPILATION_METRIC, but event_name isn't the toplevel event. "
                    "CompilationMetrics must be logged to the toplevel event. Consider using `log_toplevel_event_data` directly."
                )
            metrics_context = get_metrics_context()
            if not metrics_context.in_progress():
                raise RuntimeError(
                    "No metrics context is in progress. Please only call this function within a metrics context."
                )

            # TODO: should we assert that the keys of metadata are in CompilationMetrics?
            metrics_context.update(metadata, overwrite)
            chromium_log.add_event_data(event_name, **metadata)

    @staticmethod
    def add_toplevel(
        log_level: CompileEventLogLevel, overwrite: bool = False, **metadata: object
    ) -> None:
        # [FUNCTION DOC] Adds metadata to the outermost (toplevel) active event via add_data.
        # Raises RuntimeError if no toplevel event is active.
        """
        Syntactic sugar for logging to the toplevel event
        """
        top_event = get_chromium_event_logger().get_outermost_event()
        if top_event is None:
            raise RuntimeError(
                "No toplevel event active. Please only call this function within a dynamo_timed context."
            )
        CompileEventLogger.add_data(top_event, log_level, overwrite, **metadata)

    @staticmethod
    def increment(
        event_name: str, log_level: CompileEventLogLevel, key: str, value: int
    ) -> None:
        # [FUNCTION DOC] Increments (or creates) integer field key by value on the event; for
        # COMPILATION_METRIC also increments it in the metrics context and requires the event to
        # be toplevel and a metrics context in progress (RuntimeError otherwise).
        """
        Increments an existing field, or adds it
        """
        chromium_log = get_chromium_event_logger()
        if (
            log_level == CompileEventLogLevel.CHROMIUM
            or log_level == CompileEventLogLevel.PT2_COMPILE
        ):
            chromium_log.increment(event_name, key, value)
        else:
            if log_level != CompileEventLogLevel.COMPILATION_METRIC:
                raise AssertionError(
                    f"Expected log_level to be COMPILATION_METRIC, got {log_level}"
                )
            top_event = chromium_log.get_outermost_event()
            if event_name != top_event:
                raise RuntimeError(
                    "Log level is COMPILATION_METRIC, but event_name isn't the toplevel event. "
                    "CompilationMetrics must be logged to the toplevel event. Consider using `increment_toplevel` directly."
                )

            metrics_context = get_metrics_context()
            if not metrics_context.in_progress():
                raise RuntimeError(
                    "No metrics context is in progress. Please only call this function within a metrics context/dynamo_timed."
                )

            metrics_context.increment(key, value)
            chromium_log.increment(event_name, key, value)

    @staticmethod
    def increment_toplevel(
        key: str,
        value: int = 1,
        log_level: CompileEventLogLevel = CompileEventLogLevel.COMPILATION_METRIC,
    ) -> None:
        # [FUNCTION DOC] Increments key by value (default 1) on the toplevel event at log_level
        # (default COMPILATION_METRIC). Raises RuntimeError if no toplevel event is active.
        """
        Increments a value on the toplevel metric. By default, logs to metric.
        """
        chromium_log = get_chromium_event_logger()
        top_event = chromium_log.get_outermost_event()
        if top_event is None:
            raise RuntimeError(
                "No toplevel event active. Please only call this function within a metrics context/dynamo_timed."
            )
        CompileEventLogger.increment(top_event, log_level, key, value)

    @staticmethod
    def add_to_set(
        event_name: str, log_level: CompileEventLogLevel, key: str, value: Any
    ) -> None:
        # [FUNCTION DOC] Adds value to the set stored under key on the event (creating the set).
        # COMPILATION_METRIC level also updates the metrics context and has the same toplevel and
        # in-progress requirements (RuntimeError otherwise).
        """
        Add metadata <value> to a set of values with key <key>. Creates a set if it doesn't exist.
        """
        chromium_log = get_chromium_event_logger()
        if (
            log_level == CompileEventLogLevel.CHROMIUM
            or log_level == CompileEventLogLevel.PT2_COMPILE
        ):
            chromium_log.add_to_set(event_name, key, value)
        else:
            if log_level != CompileEventLogLevel.COMPILATION_METRIC:
                raise AssertionError(
                    f"Expected log_level to be COMPILATION_METRIC, got {log_level}"
                )
            top_event = chromium_log.get_outermost_event()
            if event_name != top_event:
                raise RuntimeError(
                    "Log level is COMPILATION_METRIC, but event_name isn't the toplevel event. "
                    "CompilationMetrics must be logged to the toplevel event. Consider using `add_to_set_metric` directly."
                )

            metrics_context = get_metrics_context()
            if not metrics_context.in_progress():
                raise RuntimeError(
                    "No metrics context is in progress. Please only call this function within a metrics context/dynamo_timed."
                )

            metrics_context.add_to_set(key, value)
            chromium_log.add_to_set(event_name, key, value)

    @staticmethod
    def add_to_set_toplevel(
        key: str,
        value: Any,
        log_level: CompileEventLogLevel = CompileEventLogLevel.COMPILATION_METRIC,
    ) -> None:
        # [FUNCTION DOC] Like add_to_set but targets the toplevel event automatically. Raises
        # RuntimeError if no toplevel event is active.
        """
        Same as add to set, just does it automatically to the toplevel event instead of having to explicitly name it.
        Defaults to COMPILATION_METRIC log level.
        """
        chromium_log = get_chromium_event_logger()
        top_event = chromium_log.get_outermost_event()
        if top_event is None:
            raise RuntimeError(
                "No toplevel event active. Please only call this function within a metrics context/dynamo_timed."
            )
        CompileEventLogger.add_to_set(top_event, log_level, key, value)

    # Helper functions that are syntactic sugar

    @staticmethod
    def chromium(event_name: str, **metadata: object) -> None:
        # [FUNCTION DOC] Adds metadata (no overwrite) to the named event at CHROMIUM level.
        """
        Add <metadata> to <event_name> in chromium. Each key/value of metadata will appear in the chromium trace.
        <event_name> should be the name of a timed event span passed to `dynamo_timed`.
        """
        CompileEventLogger.add_data(
            event_name, CompileEventLogLevel.CHROMIUM, overwrite=False, **metadata
        )

    @staticmethod
    def pt2_compile(event_name: str, **metadata: object) -> None:
        # [FUNCTION DOC] Adds metadata (no overwrite) to the named event at PT2_COMPILE level; the
        # event must have been started with PT2 compile event logging enabled.
        """
        Add <metadata> to <event_name> in chromium and PT2 Compile Events.
        Each key/value of metadata will appear in the chromium trace. Each kwarg name becomes
        a column in PT2 Compile Events, with the corresponding kwarg value.
        <event_name> should be the name of a timed event span passed to `dynamo_timed`,
        with log_to_pt2_compile_events=True.
        """
        CompileEventLogger.add_data(
            event_name, CompileEventLogLevel.PT2_COMPILE, overwrite=False, **metadata
        )

    @staticmethod
    def add_record_function_data(event_name: str, **metadata: object) -> None:
        # [FUNCTION DOC] If the autograd profiler is enabled and metadata is nonempty, opens and
        # immediately closes a record_function range named with the event and key=value pairs so
        # it appears in profiler traces. Otherwise no-op.
        """
        Add record function data to the profiler event.

        This emits profiler event data so compilation events show up in stack profilers
        like the PyTorch profiler.

        Args:
            event_name: Name of the event to record
            **metadata: Additional metadata to attach to the record function
        """
        if torch.autograd.profiler._is_profiler_enabled and metadata:
            metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            with torch.autograd.profiler.record_function(
                f"{event_name}_data: {metadata_str}"
            ):
                pass

    @staticmethod
    def compilation_metric(overwrite: bool = False, **metadata: object) -> None:
        # [FUNCTION DOC] Adds metadata to the toplevel event at COMPILATION_METRIC level (also
        # updates the metrics context); overwrite controls replacing existing fields.
        """
        Add <metadata> to the CompilationMetrics context. Also logs to PT2 Compile Events
        and chromium.
        Each key/value of metadata will appear in the chromium trace. Each kwarg name becomes
        a column in PT2 Compile Events and Dynamo Compile, with the corresponding kwarg value.
        """
        CompileEventLogger.add_toplevel(
            CompileEventLogLevel.COMPILATION_METRIC, overwrite, **metadata
        )

    @staticmethod
    def instant(
        event_name: str, metadata: dict[str, Any], time_ns: int | None = None
    ) -> None:
        # [FUNCTION DOC] Logs an instant event to chromium logs at CHROMIUM level via
        # log_instant_event; time_ns defaults to now.
        """
        Log an instant event to chromium logs with name <event_name> at time <time_ns>. The `args` field in
        Perfetto will point to metadata. <time_ns> should be a value obtained from time.time_ns().
        """
        CompileEventLogger.log_instant_event(
            event_name, metadata, time_ns, CompileEventLogLevel.CHROMIUM
        )

    @staticmethod
    def try_add_pt2_compile(event_name: str, **metadata: object) -> None:
        # [FUNCTION DOC] Adds metadata to an existing event if the chromium logger is active;
        # silently returns otherwise (and the logger ignores a missing event).
        """
        Adds to an existing pt2_compile event, but silently returns if the event doesn't exist
        or ChromiumEventLogger is not initialized.
        This function is syntactic sugar for chromium_event_logger().try_add_event_data.
        """
        if not chromium_event_log_active():
            return
        chromium_log = get_chromium_event_logger()
        chromium_log.try_add_event_data(event_name, **metadata)

    @staticmethod
    def try_(method_fn: Callable[_P, Any], *args: _P.args, **kwargs: _P.kwargs) -> None:
        # [FUNCTION DOC] Calls method_fn with the given arguments only if the chromium event
        # logger is active and a metrics context is in progress; otherwise does nothing. Return
        # value is discarded.
        """
        Special function that quietly runs a given method, returning if CHROMIUM_EVENT_LOG is None or metrics context is not set
        """
        if not chromium_event_log_active():
            return
        metrics_context = get_metrics_context()
        if not metrics_context.in_progress():
            return
        method_fn(*args, **kwargs)


_dynamo_timed_tls = threading.local()


@contextmanager
def compile_time_record_function(name: str) -> Generator[Any, None, None]:
    # [FUNCTION DOC] Context manager that, when the autograd profiler is enabled, wraps the body
    # in a low-overhead fast record-function range named name; otherwise just yields.
    """
    A context manager for compile-time profiling that uses _RecordFunctionFast
    for lower overhead than torch.profiler.record_function.

    This is intended for use during compilation (dynamo, inductor, etc.) where
    we want profiling support but with minimal overhead. Moreover, we do not
    want the record_function call inside torch.compile to be dispatched.

    Args:
        name: The name of the record function event that will appear in profiles.
    """
    if torch.autograd.profiler._is_profiler_enabled:
        rf = torch._C._profiler._RecordFunctionFast(name)
        rf.__enter__()
        try:
            yield
        finally:
            rf.__exit__(None, None, None)
    else:
        yield


@contextmanager
def dynamo_timed(
    key: str,
    # TODO(masneral): Deprecate this param.
    phase_name: str | None = None,
    log_pt2_compile_event: bool = False,
    metadata: dict[str, object] | None = None,
    dynamo_compile_column_us: str | None = None,
    compile_id: CompileId | None = None,
    is_backward: bool | None = None,
    log_waitcounter: bool = False,
    waitcounter_name_override: str | None = None,
) -> Generator[Any, None, None]:
    # [FUNCTION DOC] Context manager that times its body. Logs chromium start/end events (and PT2
    # compile events if requested), records the elapsed seconds in compilation_time_metrics under
    # key, optionally enters a wait counter, and when dynamo_compile_column_us is given (must end
    # with an underscore-us suffix, else AssertionError) accumulates microseconds into the metrics
    # context (compile-time) or runtime metrics context, plus a total duration for outermost
    # nested calls (tracked with a thread-local depth). Also adds to cumulative_time_spent_ns.
    # Timing/logging happens in a finally clause so it runs on exceptions too.
    """
    dynamo_timed is a context manager
    By wrapping a function in dynamo_timed, we can get a few things:

    1) Optionally log timings to pt2_compile_events.
    2) Optionally log timings to CompilationMetrics (dynamo_compile).
    3) Optionally log chromium events.
    4) Optionally increment a WaitCounter.
    5) Store a record in compilation_time_metrics
       For example:

        def _foo(...):
            with dynamo_timed("_foo"):
                ...

        Would show up as an entry in our timing dict:
        OrderedDict([('_foo', [0.083690, 0.23949, 3.1425e-05])])
        This is extremely useful for granular debugging.

    Although it is tempting to use dynamo_timed as a decorator, please do not.
    In its decorator form it makes cProfile traces less useful as dynamo_timed
    suddenly becomes a bottleneck for lots of function calls (as only one parent
    pointer is recorded).

    Params:
    - key: key into compile_time_metrics. If phase_name is not provided, this is
      also the event name used for pt2_compile_events logs and chromium events.
    - phase_name: Optional override for the event name.
    - log_pt2_compile_event: Whether to log a pt2 compile event internally.
    - metadata: Extra metadata to put in pt2_compile_events.
    - dynamo_compile_column_us: If provided, updates the specified CompilationMetrics
      field to be logged to dyname_compile column. We expect all columns to be _us;
      therefore, the field name must end with "_us".
    - compile_id: In the typical case, this parameter should not be needed. Use to
      supply the compile_id for those cases where we want to log a compile_id where
      it's not naturally available, e.g., for runtime autotuning.
    - is_backward: Specify forward/backward directly when not available in a
      CompileContext, e.g., during runtime autotuning.
      that support it.
    - log_waitcounter: If set, we'll log a waitcounter of the form "pytorch.dynamo_timed.{key}"
    """
    if phase_name:
        event_name = phase_name
        fn_name = key
    else:
        event_name = key
        fn_name = None

    if key not in compilation_time_metrics:
        compilation_time_metrics[key] = []

    metrics = compilation_time_metrics[key]
    event_metadata = {}
    if metadata:
        event_metadata.update(metadata)
    if fn_name:
        event_metadata.update({"fn_name": fn_name})
    if is_backward is not None:
        event_metadata.update({"is_backward": is_backward})

    chromium_log: ChromiumEventLogger = get_chromium_event_logger()
    start_ns = time.time_ns()
    chromium_log.log_event_start(
        event_name, start_ns, event_metadata, log_pt2_compile_event, compile_id
    )

    cx_mgrs: list[typing.Any] = [compile_time_record_function(f"{key} (dynamo_timed)")]
    if log_waitcounter:
        wc_name = waitcounter_name_override or key
        cx_mgrs.append(_WaitCounter(f"pytorch.wait_counter.{wc_name}").guard())

    is_compile_time = torch._guards.CompileContext.current_compile_id() is not None
    if dynamo_compile_column_us:
        # We're standardizing on microseconds for dynamo_compile timings.
        if not dynamo_compile_column_us.endswith("_us"):
            raise AssertionError(
                f"dynamo_compile_column_us must end with '_us', got {dynamo_compile_column_us!r}"
            )

        # Track nested dynamo_timed calls that update CompilationMetrics so we can
        # bump a total duration only for the outermost metric.
        if not hasattr(_dynamo_timed_tls, "depth"):
            _dynamo_timed_tls.depth = 0
        _dynamo_timed_tls.depth += 1

        # The corresponding WaitCounters that we bump for all overheads
        if _dynamo_timed_tls.depth == 1:
            cx_mgrs.append(_WaitCounter("pytorch.wait_counter.dynamo_compile").guard())
            if not is_compile_time:
                runtime_wc = "pytorch.wait_counter.compile_runtime_overheads"
                cx_mgrs.append(_WaitCounter(runtime_wc).guard())

    try:
        with contextlib.ExitStack() as stack:
            for cx in cx_mgrs:
                stack.enter_context(cx)
            yield
    finally:
        end_ns = time.time_ns()
        time_spent_ns = end_ns - start_ns
        metrics.append(time_spent_ns / 1e9)
        chromium_log.log_event_end(
            event_name, end_ns, {}, start_ns, log_pt2_compile_event, compile_id
        )
        if dynamo_compile_column_us:
            # TODO: the events that we capture in calculate_time_spent() seem a little
            # arbitrary. Currently, it's only those fields that are present in
            # CompilationMetrics (but note that we accumulate by the associated event
            # name, not the field name in CompilationMetrics). Do we want to keep it
            # this way?
            cumulative_time_spent_ns[event_name] += time_spent_ns

            # Bump the total duration for every outer event.
            _dynamo_timed_tls.depth -= 1
            is_outer_event = _dynamo_timed_tls.depth == 0

            duration_us = time_spent_ns // 1000
            if is_compile_time:
                metrics_context = get_metrics_context()
                if metrics_context.in_progress():
                    metrics_context.increment(dynamo_compile_column_us, duration_us)
                    if is_outer_event:
                        metrics_context.increment("duration_us", duration_us)
            else:
                runtime_context = get_runtime_metrics_context()
                runtime_context.increment(dynamo_compile_column_us, duration_us)
                if is_outer_event:
                    extra = {
                        "compile_id": compile_id,
                        "is_runtime": True,
                        "is_forward": not is_backward,
                    }
                    runtime_context.increment("duration_us", duration_us, extra)


@overload
def compile_times(repr: Literal["str"], aggregate: bool = False) -> str: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def compile_times(
    repr: Literal["csv"], aggregate: bool = False
) -> tuple[list[str], list[object]]: ...


def compile_times(  # type: ignore[misc]
    repr: str = "str", aggregate: bool = False
) -> str | None | tuple[list[str], list[str]]:
    """
    Get metrics about torchdynamo frontend/backend compilation times.

    Accumulates information from functions tagged with `dynamo_timed`.

    repr='str' returns a printable string for user interaction, and 'csv'
    returns headers, rows which can be logged for output

    aggregate causes values from multiple compilations (e.g. split graphs)
    to be accumulated into one value.  If false, expect more than one value
    per metric.
    """
    # [FUNCTION DOC] Reports compilation times recorded by dynamo_timed. With repr "str" returns a
    # printable table (Function / Runtimes) string; with "csv" returns a (headers, row values)
    # pair; any other repr returns None. If aggregate, values per key are summed, else joined.

    def fmt_fn(values: list[float], item_fn: Callable[[float], str] = str) -> str:
        # [FUNCTION DOC] HELPER: Formats a metric's list of times: the sum if aggregate is set,
        # otherwise a comma-joined list, applying item_fn to each number.
        if aggregate:
            return item_fn(sum(values))
        return ", ".join(map(item_fn, values))

    if repr == "str":
        rows = [
            (k, fmt_fn(compilation_time_metrics[k], item_fn=lambda x: f"{x:.4f}"))
            for k in compilation_time_metrics
        ]
        out = "TorchDynamo compilation metrics:\n"
        out += tabulate(rows, headers=("Function", "Runtimes (s)"))
        return out
    elif repr == "csv":
        values = [
            fmt_fn(v, item_fn=lambda x: f"{x:.6f}")
            for v in compilation_time_metrics.values()
        ]
        headers = list(compilation_time_metrics.keys())
        return headers, values
    return None


@atexit.register
def dump_compile_times() -> None:
    # [FUNCTION DOC] Registered to run at interpreter exit; logs the aggregated compile-times
    # string at info level.
    log.info(compile_times(repr="str", aggregate=True))


tensortype_to_dtype = {
    torch.FloatTensor: (torch.float32, torch.float),
    torch.DoubleTensor: (torch.float64, torch.double),
    torch.HalfTensor: (torch.float16, torch.half),
    torch.BFloat16Tensor: (torch.bfloat16,),
    torch.ByteTensor: (torch.uint8,),
    torch.CharTensor: (torch.int8,),
    torch.LongTensor: (torch.int64, torch.long),
    torch.IntTensor: (torch.int32, torch.int),
    torch.ShortTensor: (torch.int16, torch.short),
    torch.BoolTensor: (torch.bool,),
}


class DuplicateWarningChecker:
    # [CLASS DOC] Bounded LRU set of already-seen warning keys (OrderedDict self.set, capacity
    # maxsize) used to suppress duplicate graph-break warnings unless config.verbose is set.
    def __init__(self, maxsize: int = 4096) -> None:
        # [FUNCTION DOC] Stores the capacity and initializes an empty seen-set via reset.
        self.maxsize = maxsize
        self.reset()

    def reset(self) -> None:
        # [FUNCTION DOC] Discards all remembered keys by creating a fresh ordered mapping.
        self.set: OrderedDict[Any, Any] = OrderedDict()

    def add(self, key: str | tuple[object, object]) -> bool:
        # [FUNCTION DOC] Records key and returns True if it should be reported. A key already seen
        # is moved to most-recent and returns False unless config.verbose is true; a new key is
        # inserted, evicting the oldest entries beyond maxsize, and returns True.
        if key in self.set:
            self.set.move_to_end(key, last=True)
            if not config.verbose:
                return False
        else:
            self.set[key] = None
            while len(self.set) > self.maxsize:
                self.set.popitem(last=False)
        return True


graph_break_dup_warning_checker = DuplicateWarningChecker()


def setup_compile_debug() -> contextlib.ExitStack:
    # [FUNCTION DOC] If env var TORCH_COMPILE_DEBUG is "1", installs a debug-log file handler and
    # returns the exit stack that removes it; otherwise returns an empty exit stack.
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    if compile_debug:
        return add_file_handler()

    return contextlib.ExitStack()


def reset_graph_break_dup_checker() -> None:
    # [FUNCTION DOC] Clears the global graph-break duplicate warning checker.
    graph_break_dup_warning_checker.reset()


# Matches ANSI escape sequences (CSI)
ANSI_ESCAPE_PATTERN = re.compile(
    r"""
    \x1B            # ESC
    \[              # [
    [0-?]*          # Parameter bytes
    [ -/]*          # Intermediate bytes
    [@-~]           # Final byte
    """,
    re.VERBOSE,
)


class StripAnsiFormatter(logging.Formatter):
    # [CLASS DOC] Logging formatter that removes ANSI escape sequences from formatted messages.
    """Logging formatter that strips ANSI escape codes."""

    def format(self, record: logging.LogRecord) -> str:
        # [FUNCTION DOC] Formats the record normally, then strips ANSI escape sequences.
        msg = super().format(record)
        return ANSI_ESCAPE_PATTERN.sub("", msg)


def add_file_handler() -> contextlib.ExitStack:
    # [FUNCTION DOC] Creates the debug dir's torchdynamo folder, attaches an ANSI-stripping file
    # handler (debug.log) to the torch._dynamo logger, and returns an exit stack that detaches it.
    # Side effects: creates directories and a file.
    log_path = os.path.join(get_debug_dir(), "torchdynamo")
    os.makedirs(log_path, exist_ok=True)

    log_file_handler = logging.FileHandler(os.path.join(log_path, "debug.log"))
    log_file_handler.setFormatter(StripAnsiFormatter("%(message)s"))
    logger = logging.getLogger("torch._dynamo")
    logger.addHandler(log_file_handler)

    exitstack = contextlib.ExitStack()
    exitstack.callback(lambda: logger.removeHandler(log_file_handler))
    return exitstack


def setup_log_file() -> contextlib.ExitStack:
    # [FUNCTION DOC] If config.log_file_name is set, attaches a file handler to every torch
    # logger and returns an exit stack that removes the handlers; otherwise returns an empty stack.
    # KNOWN ISSUE: the removal callback closes over the loop variable, so on exit only the last
    # logger's handler is removed (late binding).
    exitstack = contextlib.ExitStack()
    if config.log_file_name is not None:
        log_file_handler = logging.FileHandler(config.log_file_name)
        for logger in torch._logging._internal.get_loggers():
            logger.addHandler(log_file_handler)
            exitstack.callback(lambda: logger.removeHandler(log_file_handler))
        return exitstack

    return exitstack


def gen_record_file_name(exc: Exception, code: CodeType) -> str:
    # [FUNCTION DOC] Builds the path of an execution-record file under the debug dir's
    # error_recordings folder from the code name, exception type name and first line number.
    return f"{get_debug_dir()}/error_recordings/\
{code.co_name}_{type(exc).__name__}_{code.co_firstlineno}.rec"


def write_record_to_file(filename: str, exec_record: ExecutionRecord) -> None:
    # [FUNCTION DOC] Dumps the execution record to filename (binary), creating parent dirs. If the
    # file exists, logs a warning and skips. Any exception is logged, never raised.
    try:
        if os.path.exists(filename):
            log.warning(
                "Unable to write execution record %s; file already exists.", filename
            )
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                exec_record.dump(f)
    except Exception:
        log.exception("Unable to write execution record %s", filename)


def count_calls(g: fx.Graph) -> int:
    # [FUNCTION DOC] Returns the number of graph nodes whose op name contains "call"
    # (call_function, call_method, call_module).
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x: T) -> T:
    # [FUNCTION DOC] HELPER: Returns its argument unchanged.
    return x


def hashable(x: object) -> bool:
    # [FUNCTION DOC] Returns True if hashing x succeeds; False if it raises TypeError or ValueError.
    try:
        hash(x)
        return True
    except TypeError:
        return False
    # cannot hash writable memoryview object
    except ValueError:
        return False


def nothing(*args: Any, **kwargs: Any) -> None:
    # [FUNCTION DOC] HELPER: No-op accepting any arguments.
    pass


class ExactWeakKeyDictionary:
    # [CLASS DOC] Mapping keyed by object identity (id) with weak references to keys. State:
    # values (id -> value) and refs (id -> weakref). Entries vanish automatically when the key
    # object is garbage collected. Keys need not be hashable.
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates empty values and refs mappings.
        self.values: dict[int, Any] = {}
        self.refs: dict[int, weakref.ReferenceType[Any]] = {}

    def __getitem__(self, key: Any) -> Any:
        # [FUNCTION DOC] Returns the value stored for the key's identity; raises KeyError if absent.
        return self.values[id(key)]

    def get(self, key: Any, default: Any = None) -> Any:
        # [FUNCTION DOC] Returns the value for the key's identity, or default if absent.
        return self.values.get(id(key), default)

    def __contains__(self, key: Any) -> bool:
        # [FUNCTION DOC] True if a value is stored for the key's identity.
        return id(key) in self.values

    def __setitem__(self, key: Any, value: Any) -> None:
        # [FUNCTION DOC] Stores value under the key's identity, registering a weak reference whose
        # callback removes the entry when the key dies (only on first insertion of that key).
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx: int) -> None:
        # [FUNCTION DOC] HELPER: Deletes the value and weakref entries for an id if present.
        if idx in self.values:
            del self.values[idx]
        if idx in self.refs:
            del self.refs[idx]

    def clear(self) -> None:
        # [FUNCTION DOC] Removes all entries.
        self.refs.clear()
        self.values.clear()


@overload
def istype(obj: object, allowed_types: type[T]) -> TypeIs[T]: ...


# This can be simplified once TypeVarTuple objects can be expanded into TypeIs.
@overload
def istype(
    obj: object, allowed_types: tuple[type[T1], type[T2]]
) -> TypeIs[T1 | T2]: ...


@overload
def istype(
    obj: object, allowed_types: tuple[type[T1], type[T2], type[T3]]
) -> TypeIs[T1 | T2 | T3]: ...


@overload
def istype(
    obj: object, allowed_types: tuple[type[T1], type[T2], type[T3], type[T4]]
) -> TypeIs[T1 | T2 | T3 | T4]: ...


@overload
def istype(
    obj: object, allowed_types: tuple[type[T1], type[T2], type[T3], type[T4], type[T5]]
) -> TypeIs[T1 | T2 | T3 | T4 | T5]: ...


@overload
def istype(
    obj: object,
    allowed_types: tuple[type[T1], type[T2], type[T3], type[T4], type[T5], type[T6]],
) -> TypeIs[T1 | T2 | T3 | T4 | T5 | T6]: ...


@overload
def istype(
    obj: object,
    allowed_types: tuple[
        type[T1], type[T2], type[T3], type[T4], type[T5], type[T6], type[T7]
    ],
) -> TypeIs[T1 | T2 | T3 | T4 | T5 | T6 | T7]: ...


@overload
def istype(
    obj: object,
    allowed_types: tuple[
        type[T1], type[T2], type[T3], type[T4], type[T5], type[T6], type[T7], type[T8]
    ],
) -> TypeIs[T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8]: ...


@overload
def istype(
    obj: object,
    allowed_types: tuple[
        type[T1],
        type[T2],
        type[T3],
        type[T4],
        type[T5],
        type[T6],
        type[T7],
        type[T8],
        type[T9],
    ],
) -> TypeIs[T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9]: ...


@overload
def istype(
    obj: object,
    allowed_types: tuple[
        type[T1],
        type[T2],
        type[T3],
        type[T4],
        type[T5],
        type[T6],
        type[T7],
        type[T8],
        type[T9],
        type[T10],
    ],
) -> TypeIs[T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10]: ...


@overload
def istype(
    obj: object, allowed_types: tuple[type, ...] | list[type] | set[type]
) -> bool: ...


def istype(obj: object, allowed_types: Any) -> bool:
    # [FUNCTION DOC] Exact-type check (no subclasses): if allowed_types is a tuple, list or set,
    # True when the type of obj is a member; otherwise True when the type is exactly that type.
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


_builtin_final_typing_classes: tuple[Any, ...] = tuple()
if sys.version_info >= (3, 12):
    # Some typing classes moved to C in 3.12,
    # which no longer have the _Final mixin.
    # Check for consistency e.g. here:
    # https://github.com/python/cpython/blob/f2b82b3b3b1f8c7a81e84df35ee921e44517cf32/Lib/typing.py#L32
    _builtin_final_typing_classes = (
        typing.ParamSpecArgs,
        typing.ParamSpecKwargs,
        typing.ParamSpec,
        typing.TypeVar,
        typing.TypeVarTuple,
        typing.TypeAliasType,
    )


def get_inputs_devices(
    inputs: collections.abc.Sequence[object],
    model: torch.fx.GraphModule,
) -> list[torch.device | None]:
    # [FUNCTION DOC] Collects the distinct devices of the flattened inputs and of the "val"
    # metadata of the model's graph nodes, drops non-device and meta-device entries, and appends
    # None as a final fallback entry. Returns the list.
    all_inputs = pytree.tree_flatten(inputs)[0] + [
        node.meta["val"] for node in list(model.graph.nodes) if "val" in node.meta
    ]
    devices: list[torch.device | None] = list(
        OrderedSet([i.device for i in all_inputs if hasattr(i, "device")])
    )
    return [
        i for i in devices if (isinstance(i, torch.device) and i.type != "meta")
    ] + [None]


if sys.version_info >= (3, 14):
    _builtin_final_typing_classes += (typing.Union,)


def is_typing(value: object) -> bool:
    # [FUNCTION DOC] True if value is a typing construct (typing internals, union types, Generic,
    # Union, or the 3.12+ C-implemented typing classes). Generic subclasses are intentionally
    # excluded.
    # _Final catches most of typing classes:
    #   - Any
    #   - Callable
    #   - Union (Python < 3.14)
    #   ...
    #
    # NB: we intentionally ignore classes that inherit from Generic, since they
    # can be used as both TypingVariable as well as UserDefinedClassVariable.
    if sys.version_info >= (3, 12) and isinstance(value, _builtin_final_typing_classes):
        return True
    return (
        isinstance(value, (types.UnionType, typing._Final))  # type: ignore[attr-defined]
        or value is typing.Generic
        or value is typing.Union
    )


def is_numpy_int_type(value: object) -> bool:
    # [FUNCTION DOC] True if value is exactly an instance of a numpy sized int/uint scalar type;
    # False when numpy is unavailable.
    if not np:
        return False

    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


def is_numpy_float_type(value: object) -> bool:
    # [FUNCTION DOC] True if value is exactly an instance of a numpy float16/32/64 scalar type;
    # False when numpy is unavailable.
    if not np:
        return False

    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )


_unpack_fast_types_cache: tuple[type, ...] | None = None


def _unpack_fast_types() -> tuple[type, ...]:
    # [FUNCTION DOC] HELPER: Lazily builds (and caches globally) the tuple of VariableTracker
    # classes whose elements can be unpacked directly; imported lazily to avoid circular imports.
    # Builtin iterables whose elements we can get directly via
    # unpack_var_sequence, skipping the generic iter/getiter/iternext protocol
    # (a bottleneck for large iterables). Built lazily since `variables` is a
    # circular import at module load.
    global _unpack_fast_types_cache
    if _unpack_fast_types_cache is None:
        from . import variables

        _unpack_fast_types_cache = (
            variables.ConstDictVariable,
            variables.DictViewVariable,
            variables.MappingProxyVariable,
            variables.DequeVariable,
            variables.ListVariable,
            variables.ListIteratorVariable,
            variables.RangeVariable,
            variables.SetVariable,
            variables.TensorVariable,
            variables.TupleVariable,
        )
    return _unpack_fast_types_cache


def unpack_iterable(
    tx: InstructionTranslatorBase, iterable: VariableTracker
) -> list[VariableTracker]:
    # [FUNCTION DOC] Returns all elements of a symbolic iterable as a list of VariableTrackers:
    # directly via unpack_var_sequence for fast-path types, else by exhausting lazily_unpack.
    if isinstance(iterable, _unpack_fast_types()):
        # unpack_var_sequence returns a fresh list, so hand it back directly:
        # no generator, no per-element callback, single allocation.
        return iterable.unpack_var_sequence(tx)
    return list(lazily_unpack(tx, iterable))


def unpack_and_apply_fn(
    tx: InstructionTranslatorBase,
    iterable: VariableTracker,
    apply_fn,
) -> None:
    # [FUNCTION DOC] Calls apply_fn on each element produced by lazily_unpack of the iterable.
    for item in lazily_unpack(tx, iterable):
        apply_fn(item)


def lazily_unpack(
    tx: InstructionTranslatorBase,
    iterable: VariableTracker,
):
    # [FUNCTION DOC] Generator yielding the elements of a symbolic iterable one at a time: fast
    # path via unpack_var_sequence, otherwise runs the iterator protocol (getiter / next) and
    # stops when an observed StopIteration is handled.
    from .exc import handle_observed_exception, ObservedUserStopIteration
    from .variables.object_protocol import generic_getiter, pyiter_next

    if isinstance(iterable, _unpack_fast_types()):
        yield from iterable.unpack_var_sequence(tx)
        return

    iterator = generic_getiter(tx, iterable)  # type: ignore[bad-argument-type]
    while True:
        try:
            yield pyiter_next(tx, iterator)  # type: ignore[bad-argument-type]
        except ObservedUserStopIteration:
            handle_observed_exception(tx)
            break


@overload
def is_lru_cache_wrapped_function(
    value: Callable[..., T],
) -> TypeGuard[functools._lru_cache_wrapper[T]]: ...


@overload
def is_lru_cache_wrapped_function(
    value: Any,
) -> TypeGuard[functools._lru_cache_wrapper[Any]]: ...


def is_lru_cache_wrapped_function(
    value: Any,
) -> bool:
    # [FUNCTION DOC] True if value is an lru_cache wrapper whose statically-looked-up wrapped
    # target is a plain function.
    return isinstance(value, functools._lru_cache_wrapper) and is_function(
        inspect.getattr_static(value, "__wrapped__")
    )


_lru_cache_wrappers_allowed_to_trace_without_warning: weakref.WeakSet[
    functools._lru_cache_wrapper[Any]
] = weakref.WeakSet()


def allow_lru_cache_wrapper_trace_without_warning(
    value: functools._lru_cache_wrapper[Any],
) -> None:
    # [FUNCTION DOC] Adds the lru_cache wrapper to a global weak set of wrappers that may be
    # traced without emitting a warning.
    _lru_cache_wrappers_allowed_to_trace_without_warning.add(value)


def is_lru_cache_wrapper_trace_without_warning_allowed(value: object) -> bool:
    # [FUNCTION DOC] True if value is in the global weak set of wrappers allowed to be traced
    # silently.
    return value in _lru_cache_wrappers_allowed_to_trace_without_warning


_FuncTypes: TypeAlias = (
    types.FunctionType
    | types.BuiltinFunctionType
    | types.MethodDescriptorType
    | types.WrapperDescriptorType
)


def is_function_or_wrapper(
    value: Any,
) -> TypeIs[_FuncTypes | torch._ops.OpOverloadPacket | torch._ops.OpOverload]:
    # [FUNCTION DOC] True if value is a function-like object (see is_function) or a torch op
    # overload / overload packet.
    return is_function(value) or isinstance(
        value, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
    )


def is_function(
    value: Any,
) -> TypeIs[_FuncTypes]:
    # [FUNCTION DOC] True if value is a Python function, builtin function, method descriptor or
    # wrapper descriptor (subclass instances count, since isinstance is used).
    return isinstance(
        value,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodDescriptorType,
            types.WrapperDescriptorType,
        ),
    )


cmp_name_to_op_mapping = {
    "__eq__": operator.eq,
    "__ne__": operator.ne,
    "__lt__": operator.lt,
    "__le__": operator.le,
    "__gt__": operator.gt,
    "__ge__": operator.ge,
}


cmp_name_to_op_str_mapping = {
    "__eq__": "==",
    "__ne__": "!=",
    "__lt__": "<",
    "__le__": "<=",
    "__gt__": ">",
    "__ge__": ">=",
}


def is_wrapper_or_member_descriptor(
    value: Any,
) -> TypeIs[
    types.GetSetDescriptorType
    | types.MethodDescriptorType
    | types.WrapperDescriptorType
    | types.MemberDescriptorType
    | types.MethodWrapperType
]:
    # [FUNCTION DOC] True if value is a getset/method/wrapper/member descriptor or a method
    # wrapper, i.e. a C-level attribute or slot wrapper object.
    return isinstance(
        value,
        (
            # set up by PyGetSetDef
            types.GetSetDescriptorType,
            # set by PyMethodDef, e.g. list.append
            types.MethodDescriptorType,
            # slots - list.__add__
            types.WrapperDescriptorType,
            # set up by PyMemberDef
            types.MemberDescriptorType,
            # wrapper over C functions
            types.MethodWrapperType,
        ),
    )


def tracked_repr(tx: InstructionTranslatorBase, vt: VariableTracker) -> str:
    # [FUNCTION DOC] Returns the Python string produced by symbolically evaluating the repr of vt
    # under the translator tx (may trace user code or graph break).
    from .variables.object_protocol import generic_repr

    return generic_repr(tx, vt).as_python_constant()


def _item_debug_repr(vt: VariableTracker) -> str:
    # [FUNCTION DOC] HELPER: Repr of the constant value if vt is a Python constant, else its
    # debug_repr string.
    if vt.is_python_constant():
        return repr(vt.as_python_constant())
    return vt.debug_repr()


def unwrap_if_wrapper(fn: Any) -> Any:
    # [FUNCTION DOC] HELPER: Returns only the unwrapped function from
    # unwrap_with_attr_name_if_wrapper.
    return unwrap_with_attr_name_if_wrapper(fn)[0]


def unwrap_with_attr_name_if_wrapper(fn: Any) -> tuple[Any, str | None]:
    # [FUNCTION DOC] If fn is a function carrying a truthy static attribute "_torchdynamo_inline",
    # returns (that attribute value, attribute name); otherwise (fn, None).
    # TODO(anijain2305) - Investigate if we can get rid of this function
    # unpack @torch._dynamo.optimize()(fn) wrapped function
    if is_function(fn) and inspect.getattr_static(fn, "_torchdynamo_inline", False):
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        attr_name = "_torchdynamo_inline"
    else:
        attr_name = None
    return fn, attr_name


def is_numpy_ndarray(value: Any) -> TypeGuard[np.ndarray]:  # type: ignore[type-arg]
    # [FUNCTION DOC] True if value is exactly a numpy ndarray; False if numpy is unavailable.
    if not np:
        return False

    return istype(value, np.ndarray)


def istensor(obj: Any) -> bool:
    # [FUNCTION DOC] True if the exact type of obj is Tensor, Parameter, FakeTensor or a
    # configured traceable tensor subclass.
    """Check of obj is a tensor"""
    tensor_list: tuple[type, ...] = (
        torch.Tensor,
        torch.nn.Parameter,
        *config.traceable_tensor_subclasses,
    )
    tensor_list = tensor_list + (torch._subclasses.FakeTensor,)
    return istype(obj, tensor_list)


def is_lazy_module(mod: Any) -> bool:
    # [FUNCTION DOC] True if mod is an instance of LazyModuleMixin.
    return isinstance(mod, LazyModuleMixin)


@functools.lru_cache(4096)
def print_once(*args: Any) -> None:
    # [FUNCTION DOC] Prints args to stdout; memoized with lru_cache so each distinct (hashable)
    # argument tuple prints at most once (until evicted past 4096 entries).
    print(*args)


def make_cell(val: Any = None) -> types.CellType:
    # [FUNCTION DOC] Creates a closure cell object holding val by capturing it in a throwaway
    # inner function and returning that function's only closure cell. Raises AssertionError if
    # the closure is missing or does not have exactly one cell.
    """Some black magic to create a cell object that usually only exists in a closure"""
    x = val

    def f() -> Any:
        # [FUNCTION DOC] HELPER: Returns the captured value; exists only to create the closure.
        return x

    if f.__closure__ is None:
        raise AssertionError("Expected f to have a closure, but f.__closure__ is None")
    if len(f.__closure__) != 1:
        raise AssertionError(
            f"Expected f.__closure__ to have exactly 1 element, got {len(f.__closure__)}"
        )
    return f.__closure__[0]


def proxy_args_kwargs(args: Any, kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # [FUNCTION DOC] Converts a sequence of VariableTracker args and a mapping of kwargs into a
    # tuple of FX proxies and a dict of proxies via as_proxy. If a conversion raises
    # NotImplementedError, triggers a graph break (unimplemented) chained from that error.
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except NotImplementedError as e:
        from .exc import unimplemented
        from .variables.base import typestr

        unimplemented(
            gb_type="Failed to convert args/kwargs to proxy",
            context=f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}",
            explanation="Missing `as_proxy()` implementation for some arg/kwarg.",
            hints=[],
            from_exc=e,
        )


def to_int_ms(v: float | None) -> int | None:
    # [FUNCTION DOC] Converts seconds to integer milliseconds (truncating); None stays None.
    return None if v is None else int(v * 1000)


# float64 timestamp has a quarter microsecond precision in 2024, so while
# this is suboptimal we shouldn't meaningfully lose precision
def to_int_us(v: float | None) -> int | None:
    # [FUNCTION DOC] Converts seconds to integer microseconds (truncating); None stays None.
    return None if v is None else int(v * 1_000_000)


# Version field added to every log. Increment to make it easier to distinguish new
# vs. old entries when you make a substantive change to how the logs are populated.
LOG_FORMAT_VERSION = 3


@dataclasses.dataclass
class CompilationMetrics:
    # [CLASS DOC] Dataclass record of everything logged to the dynamo_compile table for one
    # compile (or runtime-compile) event: ids, timings in seconds/ms/us, guard and graph counts,
    # failure info, cache stats, config dumps, etc. All fields optional (default None, or 0 for
    # hit/miss counters, or LOG_FORMAT_VERSION). Provides factory/aggregation helpers below.
    compile_id: str | None = None
    frame_key: str | None = None
    co_name: str | None = None
    co_filename: str | None = None
    co_firstlineno: int | None = None
    cache_size: int | None = None
    accumulated_cache_size: int | None = None
    guard_count: int | None = None
    shape_env_guard_count: int | None = None
    graph_op_count: int | None = None
    graph_node_count: int | None = None
    graph_input_count: int | None = None
    start_time: float | None = None
    entire_frame_compile_time_s: float | None = None
    backend_compile_time_s: float | None = None
    inductor_compile_time_s: float | None = None
    code_gen_time_s: float | None = None
    fail_type: str | None = None
    fail_reason: str | None = None
    fail_user_frame_filename: str | None = None
    fail_user_frame_lineno: int | None = None
    non_compliant_ops: set[str] | None = None
    compliant_custom_ops: set[str] | None = None
    restart_reasons: set[str] | None = None
    dynamo_time_before_restart_s: float | None = None
    stack_trace: list[str] | None = None
    exception_stack_trace: list[str] | None = None
    graph_node_shapes: str | None = None
    # Sometimes, we will finish analyzing a frame but conclude we don't want
    # to install any guarded code.  True means we actually decided to install
    # a compiled frame
    has_guarded_code: bool | None = None
    remote_cache_time_saved_s: float | None = None
    structured_logging_overhead_s: float | None = None
    config_suppress_errors: bool | None = None
    config_inline_inbuilt_nn_modules: bool | None = None
    specialize_float: bool | None = None
    dynamo_config: str | None = None
    compiler_config: str | None = None
    is_forward: bool | None = None
    num_triton_bundles: int | None = None
    remote_fx_graph_cache_get_time_ms: int | None = None
    remote_fx_graph_cache_put_time_ms: int | None = None
    start_time_us: int | None = None
    duration_us: int | None = None
    dynamo_cumulative_compile_time_us: int | None = None
    aot_autograd_cumulative_compile_time_us: int | None = None
    inductor_cumulative_compile_time_us: int | None = None
    inductor_code_gen_cumulative_compile_time_us: int | None = None
    triton_compile_time_us: int | None = None
    runtime_cudagraphify_time_us: int | None = None
    runtime_triton_autotune_time_us: int | None = None
    dynamo_compile_time_before_restart_us: int | None = None
    distributed_ephemeral_timeout_us: int | None = None
    structured_logging_overhead_us: int | None = None
    remote_fx_graph_cache_get_time_us: int | None = None
    remote_fx_graph_cache_put_time_us: int | None = None
    backward_cumulative_compile_time_us: int | None = None
    end_time_us: int | None = None
    pre_grad_pass_time_us: int | None = None
    post_grad_pass_time_us: int | None = None
    joint_graph_pass_time_us: int | None = None
    log_format_version: int = LOG_FORMAT_VERSION
    inductor_config: str | None = None
    remote_cache_version: int | None = None
    inductor_fx_remote_cache_hit_count: int | None = 0
    inductor_fx_remote_cache_miss_count: int | None = 0
    inductor_fx_remote_cache_backend_type: str | None = None
    inductor_fx_remote_cache_hit_keys: str | None = None
    inductor_fx_remote_cache_miss_keys: str | None = None
    inductor_fx_local_cache_hit_count: int | None = 0
    inductor_fx_local_cache_miss_count: int | None = 0
    aotautograd_remote_cache_hit_count: int | None = 0
    aotautograd_remote_cache_miss_count: int | None = 0
    aotautograd_local_cache_hit_count: int | None = 0
    aotautograd_local_cache_miss_count: int | None = 0
    cuda_version: str | None = None
    triton_version: str | None = None
    feature_usage: dict[str, bool] | None = None
    compile_time_autotune_time_us: int | None = None
    is_runtime: bool | None = False
    gc_time_us: int | None = None
    tensorify_float_attempt: bool | None = None
    tensorify_float_success: bool | None = None
    tensorify_float_failure: set[str] | None = None
    guard_latency_us: float | None = None
    recompile_reason: str | None = None
    num_graph_breaks: int | None = None
    triton_kernel_compile_times_us: str | None = None
    ir_count: int | None = None
    cudagraph_skip_reason: str | None = None
    python_version: str | None = None
    pgo_put_remote_code_state_time_us: int | None = None
    pgo_get_remote_code_state_time_us: int | None = None
    # The number of elements within parameters. This is classically what people
    # think of when they think of parameters in a ML model.
    param_numel: int | None = None
    # The number of elements counted by bytes - i.e. a float32 is 4 bytes
    # per element.
    param_bytes: int | None = None
    # The number of parameters counted by fields. This is mostly a proxy for
    # the number of distinct type of params.
    param_count: int | None = None
    recompile_user_contexts: set[str] | None = None
    inline_inbuilt_nn_modules_candidate: bool | None = False
    pytorch_version: str | None = None
    inductor_provenance: set[str] | None = None
    functorch_config: str | None = None

    @classmethod
    def create(cls, metrics: dict[str, Any]) -> CompilationMetrics:
        # [FUNCTION DOC] Builds a CompilationMetrics from a mapping of field names to values.
        # Derives legacy second/millisecond fields from the microsecond fields (explicit values in
        # metrics win), converts hit/miss key collections to comma-joined strings, triton kernel
        # times to a JSON string, and compile_id to a string (None if falsy). Unknown keys make
        # the constructor raise TypeError.
        """
        Factory method to create a CompilationMetrics from a dict of fields.
        Includes the logic to add legacy fields and any pre-processing, e.g.,
        we transform some fields to comma-separated strings for scuba logging.
        """

        def us_to_s(metric: int | None) -> float | None:
            # [FUNCTION DOC] HELPER: Microseconds to seconds (float); None stays None.
            return metric / 1e6 if metric is not None else None

        def us_to_ms(metric: int | None) -> int | None:
            # [FUNCTION DOC] HELPER: Microseconds to whole milliseconds (floor); None stays None.
            return metric // 1000 if metric is not None else None

        def collection_to_str(metric: Any | None) -> str | None:
            # [FUNCTION DOC] HELPER: Joins a sorted set or list as a comma-separated string of
            # item strings; None stays None and any other type becomes "<unknown>".
            def safe_str(item: Any) -> str:
                # [FUNCTION DOC] HELPER: String of item, or "<unknown>" if conversion raises.
                try:
                    return str(item)
                except Exception:
                    return "<unknown>"

            if metric is None:
                return None

            if not isinstance(metric, (set, list)):
                return "<unknown>"

            return ",".join(safe_str(item) for item in sorted(metric))

        def collection_to_json_str(metric: Any | None) -> str | None:
            # [FUNCTION DOC] HELPER: JSON-encodes the collection as a list; None stays None and
            # failures give "<unknown>".
            if metric is None:
                return None
            try:
                return json.dumps(list(metric))
            except Exception:
                return "<unknown>"

        # TODO: The following are legacy fields, populated from the fields that replace
        # them. Remove these when we decide we can really deprecate them.
        legacy_metrics = {
            "start_time": us_to_s(metrics.get("start_time_us")),
            "entire_frame_compile_time_s": us_to_s(
                metrics.get("dynamo_cumulative_compile_time_us")
            ),
            "backend_compile_time_s": us_to_s(
                metrics.get("aot_autograd_cumulative_compile_time_us")
            ),
            "inductor_compile_time_s": us_to_s(
                metrics.get("inductor_cumulative_compile_time_us")
            ),
            "code_gen_time_s": us_to_s(
                metrics.get("inductor_code_gen_cumulative_compile_time_us")
            ),
            "remote_cache_time_saved_s": us_to_s(
                metrics.get("distributed_ephemeral_timeout_us")
            ),
            "remote_fx_graph_cache_get_time_ms": us_to_ms(
                metrics.get("remote_fx_graph_cache_get_time_us")
            ),
            "remote_fx_graph_cache_put_time_ms": us_to_ms(
                metrics.get("remote_fx_graph_cache_put_time_us")
            ),
            "structured_logging_overhead_s": us_to_s(
                metrics.get("structured_logging_overhead_us")
            ),
        }

        all_metrics = {**legacy_metrics, **metrics}

        # Processing before logging:
        all_metrics["inductor_fx_remote_cache_hit_keys"] = collection_to_str(
            all_metrics.get("inductor_fx_remote_cache_hit_keys")
        )
        all_metrics["inductor_fx_remote_cache_miss_keys"] = collection_to_str(
            all_metrics.get("inductor_fx_remote_cache_miss_keys")
        )
        all_metrics["triton_kernel_compile_times_us"] = collection_to_json_str(
            all_metrics.get("triton_kernel_compile_times_us")
        )
        compile_id = all_metrics.get("compile_id")
        all_metrics["compile_id"] = str(compile_id) if compile_id else None

        # pyrefly: ignore [bad-argument-type]
        return cls(**all_metrics)


DEFAULT_COMPILATION_METRICS_LIMIT = 64


_compilation_metrics: collections.deque[CompilationMetrics] = collections.deque(
    maxlen=DEFAULT_COMPILATION_METRICS_LIMIT
)


def add_compilation_metrics_to_chromium(c: CompilationMetrics) -> None:
    # [FUNCTION DOC] Copies a fixed subset of CompilationMetrics fields (frame info, counts,
    # failure info, restart reasons, config) as event data onto the outermost active chromium
    # event, converting sets to lists. No-op if no event is active.
    """
    These are the common fields in CompilationMetrics that existed before
    metrics_context, and aren't set by MetricsContext.set(). We add the subset
    of them that make sense in `dynamo`/toplevel events in PT2 Compile Events
    directly.

    If you're tempted to add to this list, consider using CompileEventLogger.compilation_metric()
    instead, which will automatically also add it to tlparse and PT2 Compile Events.
    TODO: Get rid of this function and replace it with CompileEventLogger directly instead.
    """
    event_logger = get_chromium_event_logger()
    event_name = event_logger.get_outermost_event()
    if not event_name:
        return
    event_logger.add_event_data(
        event_name=event_name,
        frame_key=c.frame_key,
        co_name=c.co_name,
        co_filename=c.co_filename,
        co_firstlineno=c.co_firstlineno,
        cache_size=c.cache_size,
        accumulated_cache_size=c.accumulated_cache_size,
        guard_count=c.guard_count,
        shape_env_guard_count=c.shape_env_guard_count,
        graph_op_count=c.graph_op_count,
        graph_node_count=c.graph_node_count,
        graph_input_count=c.graph_input_count,
        fail_type=c.fail_type,
        fail_reason=c.fail_reason,
        fail_user_frame_filename=c.fail_user_frame_filename,
        fail_user_frame_lineno=c.fail_user_frame_lineno,
        # Sets aren't JSON serializable
        non_compliant_ops=(
            list(c.non_compliant_ops) if c.non_compliant_ops is not None else None
        ),
        compliant_custom_ops=(
            list(c.compliant_custom_ops) if c.compliant_custom_ops is not None else None
        ),
        restart_reasons=(
            list(c.restart_reasons) if c.restart_reasons is not None else None
        ),
        dynamo_time_before_restart_s=c.dynamo_time_before_restart_s,
        has_guarded_code=c.has_guarded_code,
        dynamo_config=c.dynamo_config,
    )


def _get_dynamo_config_for_logging() -> str | None:
    # [FUNCTION DOC] HELPER: Returns a key-sorted JSON string of a copy of the dynamo config with
    # blocklisted keys removed, sets sorted, and values exposing a JSON conversion converted.
    def clean_for_json(d: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Filters out blocklisted config keys and converts values to
        # JSON-friendly forms.
        blocklist = {
            "TYPE_CHECKING",
            "log_file_name",
            "verbose",
            "repro_after",
            "repro_level",
            "repro_forward_only",
            "repro_tolerance",
            "repro_ignore_non_fp",
            "same_two_models_use_fp64",
            "base_dir",
            "debug_dir_root",
            "_save_config_ignore",
            "log_compilation_metrics",
            "inject_BUILD_SET_unimplemented_TESTING_ONLY",
            "_autograd_backward_strict_mode_banned_ops",
            "reorderable_logging_functions",
            "ignore_logger_methods",
            "ignore_logging_functions",
            "traceable_tensor_subclasses",
            "nontraceable_tensor_subclasses",
            "_custom_ops_profile",
        }

        return {
            key: (
                sorted(value)
                if isinstance(value, set)
                else value.to_jsonable()
                if hasattr(value, "to_jsonable")
                else value
            )
            for key, value in d.items()
            if key not in blocklist
        }

    config_dict = clean_for_json(config.get_config_copy())
    return json.dumps(config_dict, sort_keys=True)


def _compiler_config_for_logging() -> str | None:
    # [FUNCTION DOC] HELPER: Returns a key-sorted JSON string of torch.compiler.config (sets
    # sorted); None if the config is empty/falsy, or a message string if it cannot be copied.
    def clean_for_json(d: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Drops blocklisted keys and sorts set values.
        blocklist = {
            "TYPE_CHECKING",
        }

        return {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in d.items()
            if key not in blocklist
        }

    if not torch.compiler.config:
        return None

    try:
        compiler_config_copy = torch.compiler.config.get_config_copy()  # type: ignore[attr-defined]
    except (TypeError, AttributeError):
        return "Compiler Config cannot be pickled"

    config_dict = clean_for_json(compiler_config_copy)
    return json.dumps(config_dict, sort_keys=True)


def _scrubbed_inductor_config_for_logging() -> str | None:
    # [FUNCTION DOC] HELPER: Returns a JSON string of a copy of the inductor config with
    # non-string keys removed, sets turned into lists, and unserializable values replaced by a
    # placeholder. Returns an explanatory message if the config cannot be copied or serialized;
    # never raises for logging errors. May return None if inductor config is falsy.
    """
    Method to parse and scrub uninteresting configs from inductor config
    """

    # TypeSafeSerializer for json.dumps()
    # Skips complex types as values in config dict
    class TypeSafeSerializer(json.JSONEncoder):
        # [CLASS DOC] HELPER: JSON encoder that substitutes a placeholder string for values it
        # cannot serialize.
        def default(self, o: Any) -> Any:
            # [FUNCTION DOC] HELPER: Defers to the base encoder, returning a placeholder string
            # on any error.
            try:
                return super().default(o)
            except Exception:
                return "Value is not JSON serializable"

    keys_to_scrub: set[Any] = set()
    inductor_conf_str = None
    inductor_config_copy = None

    if torch._inductor.config:
        try:
            inductor_config_copy = torch._inductor.config.get_config_copy()
        except (TypeError, AttributeError, RuntimeError, AssertionError):
            inductor_conf_str = "Inductor Config cannot be pickled"

    if inductor_config_copy is not None:
        try:
            for key, val in inductor_config_copy.items():
                if not isinstance(key, str):
                    keys_to_scrub.add(key)
                # Convert set() to list for json.dumps()
                if isinstance(val, set):
                    inductor_config_copy[key] = list(val)
            # Evict unwanted keys
            for key in keys_to_scrub:
                del inductor_config_copy[key]
            # Stringify Inductor config
            inductor_conf_str = json.dumps(
                inductor_config_copy,
                cls=TypeSafeSerializer,
                skipkeys=True,
                sort_keys=True,
            )
        except Exception:
            # Don't crash because of runtime logging errors
            inductor_conf_str = "Inductor Config is not JSON serializable"
    return inductor_conf_str


def _functorch_config_for_logging() -> str | None:
    # [FUNCTION DOC] HELPER: Returns a key-sorted JSON string of the functorch config (blocklisted
    # and non-string keys removed, sets to lists), gated by a justknob. Returns None if the knob
    # is off or the config cannot be copied; a message string if serialization fails.
    """Serialize functorch config (AOT autograd settings) for Scuba logging."""
    if not justknobs_check("pytorch/dynamo:log_functorch_config"):
        return None

    try:
        functorch_config_copy = torch._functorch.config.get_config_copy()
    except (TypeError, AttributeError, RuntimeError, AssertionError):
        return None

    try:
        blocklist = {
            "TYPE_CHECKING",
            "_save_config_ignore",
        }
        clean = {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in functorch_config_copy.items()
            if k not in blocklist and isinstance(k, str)
        }
        return json.dumps(clean, sort_keys=True, default=str)
    except Exception:
        return "Functorch Config is not JSON serializable"


def record_compilation_metrics(
    start_time_ns: int,
    end_time_ns: int,
    metrics: dict[str, Any],
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
) -> None:
    # [FUNCTION DOC] On-exit callback of the metrics contexts. Merges common fields (ids, times,
    # failure type/reason, config dumps, versions, remote cache info) with the recorded metrics
    # into a CompilationMetrics, appends it to the bounded in-memory deque, emits it as a
    # structured trace (name gets bwd_/_runtime variants), adds a subset to the active chromium
    # event, and logs it externally if config.log_compilation_metrics.
    if torch._inductor.utils.should_use_remote_fx_graph_cache():
        try:
            from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION

            remote_cache_version = REMOTE_CACHE_VERSION
            inductor_fx_remote_cache_backend_type = "_ManifoldCache"
        except ModuleNotFoundError:
            remote_cache_version = None
            inductor_fx_remote_cache_backend_type = None
    else:
        inductor_fx_remote_cache_backend_type = None
        remote_cache_version = None

    # Populate the compile_id from the metrics context if it's set. Otherwise,
    # look for it in the current compile context.
    compile_id = metrics.get("compile_id")
    if not compile_id:
        compile_id = torch._guards.CompileContext.current_compile_id()

    common_metrics = {
        "compile_id": compile_id,
        "start_time_us": start_time_ns // 1000,
        "end_time_us": end_time_ns // 1000,
        "fail_type": exc_type.__qualname__ if exc_type else None,
        "fail_reason": str(exc_value) if exc_value else None,
        "structured_logging_overhead_us": to_int_us(
            torch._logging.get_structured_logging_overhead()
        ),
        "dynamo_config": _get_dynamo_config_for_logging(),
        "config_suppress_errors": config.suppress_errors,
        "config_inline_inbuilt_nn_modules": True,
        "inductor_config": _scrubbed_inductor_config_for_logging(),
        "compiler_config": _compiler_config_for_logging(),
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__ if has_triton() else "",
        "remote_cache_version": remote_cache_version,
        "inductor_fx_remote_cache_backend_type": inductor_fx_remote_cache_backend_type,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "functorch_config": _functorch_config_for_logging(),
    }

    compilation_metrics = CompilationMetrics.create({**common_metrics, **metrics})
    _compilation_metrics.append(compilation_metrics)

    name = "compilation_metrics"
    if compilation_metrics.is_forward is False:
        name = "bwd_" + name
    if compilation_metrics.is_runtime is True:
        name = name + "_runtime"

    torch._logging.trace_structured(
        name,
        lambda: {
            k: list(v) if isinstance(v, set) else v
            for k, v in dataclasses.asdict(compilation_metrics).items()
        },
        # NB: Because compilation metrics *includes* the logging overhead time,
        # we can't both *measure* the logging overhead of compilation metrics
        # without making it inconsistent with compilation metrics itself, so
        # we ignore the (hopefully small) time spent logging compilation metrics
        record_logging_overhead=False,
        # These may be runtime logs, e.g., runtime autotunning, so we provide
        # the CompileId from the compilation metrics in case it's not available
        # in the current trace.
        compile_id=compile_id,
    )

    # If there's a chromium event in flight, add the CompilationMetrics to it.
    add_compilation_metrics_to_chromium(compilation_metrics)

    # Finally log the compilation metrics.
    if config.log_compilation_metrics:
        log_compilation_event(compilation_metrics)


def set_compilation_metrics_limit(new_size: int) -> None:
    # [FUNCTION DOC] Resizes the global compilation-metrics deque to new_size, dropping the oldest
    # entries first if needed (replaces the global deque).
    global _compilation_metrics
    while len(_compilation_metrics) > new_size:
        _compilation_metrics.popleft()
    new_deque = collections.deque(_compilation_metrics, maxlen=new_size)
    _compilation_metrics = new_deque


def clear_compilation_metrics() -> None:
    # [FUNCTION DOC] Empties the global compilation-metrics deque.
    global _compilation_metrics
    _compilation_metrics.clear()


def get_compilation_metrics() -> list[CompilationMetrics]:
    # [FUNCTION DOC] Returns a list copy of the recorded CompilationMetrics (oldest first).
    return list(_compilation_metrics)


class ChromiumEventLogger:
    # [CLASS DOC] Emits Chromium-trace-format begin/end/instant events to structured logging
    # (tlparse) and optionally internal PT2 compile event logging, and profiler record functions.
    # State: a unique id_ per logger and thread-local tls holding the event stack, a pt2-compile
    # substack, per-event metadata, and open record functions. Events are expected to nest.
    """Logs chromium events to structured logs. tlparse will concatenate these into a perfetto UI link.

    Also emits RecordFunction calls to torch.profiler when enabled.

    See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.yr4qxyxotyw for
    a specification of the Chromium Event JSON format.
    """

    def get_stack(self) -> list[str]:
        # [FUNCTION DOC] Returns the thread-local list of active event names (outermost first),
        # creating it if missing.
        """
        The main event stack, with every chromium event.
        Logged to tlparse.
        """
        if hasattr(self.tls, "stack"):
            return self.tls.stack
        else:
            self.tls.stack = []
            return self.tls.stack

    def get_outermost_event(self) -> str | None:
        # [FUNCTION DOC] Returns the first (outermost) active event name, or None if no events.
        """
        Get the outermost event name (i.e. the longest running event)
        or None if the stack is empty.
        """
        stack = self.get_stack()
        return stack[0] if stack else None

    def get_pt2_compile_substack(self) -> list[str]:
        # [FUNCTION DOC] Returns the thread-local list of active events that are also logged to
        # PT2 compile events, creating it if missing.
        """
        A smaller subset of the main stack that gets used to log
        PT2 Compile Events internally.
        """
        if hasattr(self.tls, "pt2_compile_substack"):
            return self.tls.pt2_compile_substack
        else:
            self.tls.pt2_compile_substack = []
            return self.tls.pt2_compile_substack

    def get_event_data(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the thread-local mapping of event name to accumulated metadata,
        # creating it if missing.
        if not hasattr(self.tls, "event_data"):
            self.tls.event_data = {}
        return self.tls.event_data

    def get_record_functions(self) -> dict[str, AbstractContextManager[None]]:
        # [FUNCTION DOC] Returns the thread-local mapping of event name to open profiler record
        # function objects, creating it if missing.
        if not hasattr(self.tls, "record_functions"):
            self.tls.record_functions = {}
        return self.tls.record_functions

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates thread-local storage and a unique logger id (optionally prefixed
        # by config.pt2_compile_id_prefix) and logs its creation at info level.
        self.tls = threading.local()

        from . import config

        # Generate a unique id for this logger, which we can use in scuba to filter down
        # to a single python run.
        if config.pt2_compile_id_prefix:
            self.id_ = f"{config.pt2_compile_id_prefix}-{uuid.uuid4()}"
        else:
            self.id_ = str(uuid.uuid4())

        # TODO: log to init/id tlparse after I add support for it
        log.info("ChromiumEventLogger initialized with id %s", self.id_)

    def try_add_event_data(self, event_name: str, **kwargs: Any) -> None:
        # [FUNCTION DOC] Like add_event_data but silently does nothing if the event is not active.
        """
        Same as add_event_data, but will silently not log if the event isn't in the stack.
        """
        if event_name not in self.get_stack():
            return
        self.add_event_data(event_name, **kwargs)

    def add_event_data(
        self,
        event_name: str,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Merges kwargs into the pending metadata of an active event (reported on
        # its end event). Raises RuntimeError if the event is not in the stack.
        """
        Adds additional metadata info to an in-progress event
        This metadata is recorded in the END event
        """
        if event_name not in self.get_stack():
            raise RuntimeError(
                f"Event {repr(event_name)} not in {self.get_stack()}. "
                "Cannot add metadata to events that aren't in progress. "
                "Please make sure the event has started and hasn't ended."
            )
        event_data = self.get_event_data()
        if event_name not in event_data:
            event_data[event_name] = {}
        event_data[event_name].update(kwargs)

    def increment(self, event_name: str, key: str, value: int) -> None:
        # [FUNCTION DOC] Adds value to integer field key in the active event's metadata (starting
        # from zero). Raises RuntimeError if the event is not active.
        """
        Increment an integer event data field by the given amount
        """
        if event_name not in self.get_stack():
            raise RuntimeError(
                f"Event {repr(event_name)} not in {self.get_stack()}. "
                "Cannot add metadata to events that aren't in progress. "
                "Please make sure the event has started and hasn't ended."
            )

        event_data = self.get_event_data()
        if event_name not in event_data:
            event_data[event_name] = {}
        if key not in event_data[event_name]:
            event_data[event_name][key] = 0
        event_data[event_name][key] += value

    def add_to_set(
        self,
        event_name: str,
        key: str,
        value: Any,
    ) -> None:
        # [FUNCTION DOC] Adds value to the set stored under key in the active event's metadata
        # (creating the set). Raises RuntimeError if the event is not active.
        """
        Add a value to a set within a event_name's metadata if it exists
        """
        if event_name not in self.get_stack():
            raise RuntimeError(
                f"Event {repr(event_name)} not in {self.get_stack()}. "
                "Cannot add metadata to events that aren't in progress. "
                "Please make sure the event has started and hasn't ended."
            )
        event_data = self.get_event_data()
        if event_name not in event_data:
            event_data[event_name] = {}
        if key not in event_data[event_name]:
            event_data[event_name][key] = set()
        event_data[event_name][key].add(value)

    def log_event_start(
        self,
        event_name: str,
        time_ns: int,
        metadata: dict[str, Any],
        log_pt2_compile_event: bool = False,
        compile_id: CompileId | None = None,
    ) -> None:
        # [FUNCTION DOC] Logs a begin ("B") chromium event with metadata (mutated to include the
        # compile_id string), pushes the event on the stack (and pt2 substack if requested),
        # records the start metadata as event data, and opens a profiler record function when the
        # profiler is enabled.
        """
        Logs the start of a single event.
        :param str event_name Name of event to appear in trace
        :param time_ns Timestamp in nanoseconds
        :param metadata: Any extra metadata associated with this event
        :param log_pt2_compile_event: If True, log to pt2_compile_events
        :param compile_id: Explicit compile_id (rather than using the current context)
        """
        compile_id = compile_id or torch._guards.CompileContext.current_compile_id()
        metadata["compile_id"] = str(compile_id)
        self._log_timed_event(
            event_name,
            time_ns,
            "B",
            metadata,
        )
        self.get_stack().append(event_name)
        # Add metadata from start event
        self.add_event_data(event_name, **metadata)
        if log_pt2_compile_event:
            self.get_pt2_compile_substack().append(event_name)

        # Emit profiler event so compilation events show up in stock PyTorch profiler
        if torch.autograd.profiler._is_profiler_enabled:
            rf = torch._C._profiler._RecordFunctionFast(event_name)
            rf.__enter__()
            self.get_record_functions()[event_name] = rf

            # Add metadata to the profiler event if present
            if metadata:
                CompileEventLogger.add_record_function_data(event_name, **metadata)

    def _reset_to_state(
        self,
        stack_depth: int,
        pt2_compile_substack_depth: int,
        event_data_keys: set[str],
        record_function_keys: set[str],
    ) -> None:
        # [FUNCTION DOC] HELPER: Truncates the event stack and pt2 substack to the given depths,
        # drops event data for events not in event_data_keys, and exits and removes record
        # functions not in record_function_keys.
        stack = self.get_stack()
        del stack[stack_depth:]
        substack = self.get_pt2_compile_substack()
        del substack[pt2_compile_substack_depth:]
        event_data = self.get_event_data()
        for event_name in list(event_data):
            if event_name not in event_data_keys:
                del event_data[event_name]
        # Clean up any lingering record functions (shouldn't happen in normal operation)
        record_functions = self.get_record_functions()
        for event_name, rf in list(record_functions.items()):
            if event_name not in record_function_keys:
                rf.__exit__(None, None, None)
                del record_functions[event_name]

    def reset(self) -> None:
        # [FUNCTION DOC] Clears all event stacks, event data and record functions (recovery after
        # crashed or restarted compiles).
        # We do this on every compile in case a compile crashes or restarts and
        # we haven't cleared the stack.
        self._reset_to_state(0, 0, set(), set())

    def log_event_end(
        self,
        event_name: str,
        time_ns: int,
        metadata: dict[str, Any],
        start_time_ns: int,
        log_pt2_compile_event: bool,
        compile_id: CompileId | None = None,
    ) -> None:
        # [FUNCTION DOC] Logs an end ("E") chromium event whose args combine the metadata gathered
        # during the span with the passed metadata. If the event was never started, logs a warning
        # and returns without popping; otherwise pops stacks (dropping overlapping inner events
        # with a warning), reports to internal pt2 compile logging if requested, and closes the
        # profiler record function.
        # KNOWN ISSUE: docstring names the parameter log_pt_compile_event but it is
        # log_pt2_compile_event.
        """
        Logs the end of a single event. This function should only be
        called after log_event_start with the same event_name.
        :param event_name: Name of event to appear in trace
        :param time_ns: Timestamp in nanoseconds
        :param metadata: Any extra metadata associated with this event
        :param start_time_ns: The start time timestamp in nanoseconds
        :param log_pt_compile_event: If True, log to pt2_compile_events
        :param compile_id: Explicit compile_id (rather than using the current context)
        """
        compile_id = compile_id or torch._guards.CompileContext.current_compile_id()
        metadata["compile_id"] = str(compile_id)

        # Grab metadata collected during event span
        all_event_data = self.get_event_data()
        if event_name in all_event_data:
            event_metadata = all_event_data[event_name]
            del all_event_data[event_name]
        else:
            # pyrefly: ignore [implicit-any]
            event_metadata = {}
        # Add the passed in metadata
        event_metadata.update(metadata)

        event = self._log_timed_event(
            event_name,
            time_ns,
            "E",
            event_metadata,
        )

        def pop_stack(stack: list[str]) -> None:
            # [FUNCTION DOC] HELPER: Pops entries off the given stack until the ending event is on
            # top, warning for each overlapping event removed.
            while event_name != stack[-1]:
                # If the event isn't the most recent one to end, pop
                # off the stack until it is.
                # Since event_name in self.stack, this pop is always safe
                log.warning(
                    "ChromiumEventLogger: Detected overlapping events, fixing stack"
                )
                stack.pop()

        event_stack = self.get_stack()
        # These stack health checks currently never happen,
        # but they're written this way to future proof any weird event
        # overlaps in the future.
        if event_name not in event_stack:
            # Something went wrong, we never called start on this event,
            # or it was skipped due to overlapping events below
            log.warning("ChromiumEventLogger: Start event not in stack, ignoring")
            return

        pop_stack(event_stack)

        if log_pt2_compile_event:
            pt2_compile_substack = self.get_pt2_compile_substack()
            pop_stack(pt2_compile_substack)
            log_chromium_event_internal(
                event, pt2_compile_substack, self.id_, start_time_ns
            )
            # Pop actual event off of stack
            pt2_compile_substack.pop()

        # Finally pop the actual event off the stack
        event_stack.pop()

        # End profiler event so compilation events show up in stock PyTorch profiler
        record_functions = self.get_record_functions()
        if event_name in record_functions:
            rf = record_functions.pop(event_name)
            rf.__exit__(None, None, None)

    def _log_timed_event(
        self,
        event_name: str,
        time_ns: int,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Builds a chromium event dict (name, microsecond timestamp, args,
        # phase) and sends it to structured logging and the internal recorder. Returns the dict.
        """
        Logs a timed event in chromium format. See log_event_start, log_event_end, etc.
        """
        event = {
            "name": event_name,
            "ts": time_ns / 1000,  # Chromium events are in micro seconds
            "args": metadata,
            "ph": phase,
            # These categories are needed in all chromium traces
            "cat": "dynamo_timed",
            "tid": 0,
            "pid": 0,  # pid should be specified on all logs, we don't personally care about the actual process id
        }
        torch._logging.trace_structured(
            "chromium_event",
            payload_fn=lambda: event,
            suppress_context=False,
            expect_trace_id=False,  # Not every chromium event will have a trace_id
        )
        record_chromium_event_internal(event)
        return event

    def log_instant_event(
        self,
        event_name: str,
        time_ns: int,
        metadata: dict[str, Any] | None = None,
        # By default, an instant event isn't logged internally, only to structured logging.
        log_pt2_compile_event: bool = False,
    ) -> None:
        # [FUNCTION DOC] Logs a process-scoped instant ("i") chromium event with metadata (mutated
        # to include compile_id) to structured logging, and also to internal pt2 compile logging
        # if requested.
        """
        Log an instant event with no associated duration.
        :param str event_name: Name of event to appear in trace
        :param int time_ns Timestamp in nanoseconds
        :param Optional[Dict[str, Any]] metadata: Any extra metadata associated with this event
        :param str cname optional color for the arrow in the trace
        """
        if metadata is None:
            metadata = {}
        compile_id = str(torch._guards.CompileContext.current_compile_id())
        metadata["compile_id"] = compile_id
        event = {
            "name": event_name,
            "ts": time_ns / 1000,
            "args": metadata,
            "ph": "i",
            # These categories are needed in all chromium traces
            "cat": "dynamo_timed",
            "tid": 0,
            "pid": 0,
            "s": "p",  # We use "process" level instant events so they all appear on the same row in the trace.
        }
        torch._logging.trace_structured(
            "chromium_event",
            payload_fn=lambda: event,
            suppress_context=False,
            expect_trace_id=True,
        )
        if log_pt2_compile_event:
            # Log an instant event with the same start and end time
            log_chromium_event_internal(
                event, self.get_pt2_compile_substack(), self.id_, time_ns
            )


CHROMIUM_EVENT_LOG: ChromiumEventLogger | None = None


def get_chromium_event_logger() -> ChromiumEventLogger:
    # [FUNCTION DOC] Returns the process-global ChromiumEventLogger, creating it on first use.
    global CHROMIUM_EVENT_LOG
    if CHROMIUM_EVENT_LOG is None:
        CHROMIUM_EVENT_LOG = ChromiumEventLogger()
    return CHROMIUM_EVENT_LOG


def chromium_event_log_active() -> bool:
    # [FUNCTION DOC] True if the global chromium event logger has been created.
    global CHROMIUM_EVENT_LOG
    return CHROMIUM_EVENT_LOG is not None


@contextmanager
def chromium_event_timed(
    event_name: str,
    reset_event_log_on_exit: bool = False,
    log_pt2_compile_event: bool = False,
) -> Generator[Any, None, None]:
    # [FUNCTION DOC] Context manager that logs a chromium start event on entry and the end event on
    # exit (also on exceptions), using the global logger; optionally resets the logger state on
    # exit. Prefer dynamo_timed.
    """
    Context manager that creates a chromium start and end event. Chromium event
    logging is integrated with dynamo_timed, so you probably want to use that
    instead. Use this context manager only if you want to avoid dynamo_timed.
    """
    chromium_event_log = get_chromium_event_logger()
    reset_state: tuple[int, int, set[str], set[str]] | None = None
    if reset_event_log_on_exit:
        reset_state = (
            len(chromium_event_log.get_stack()),
            len(chromium_event_log.get_pt2_compile_substack()),
            set(chromium_event_log.get_event_data()),
            set(chromium_event_log.get_record_functions()),
        )
    chromium_start_time = time.time_ns()
    chromium_event_log.log_event_start(
        event_name,
        chromium_start_time,
        {},
        log_pt2_compile_event,
    )
    try:
        yield
    finally:
        chromium_event_log.log_event_end(
            event_name,
            time.time_ns(),
            {},
            chromium_start_time,
            log_pt2_compile_event,
        )
        if reset_state is not None:
            chromium_event_log._reset_to_state(*reset_state)


# (id(scope), name) -> the token of whichever hook currently owns that
# binding. A value comparison isn't enough here: CompilePackage.install() can
# re-materialize a name like __builtins_dict__ with the *same* object a
# pre-reset hook already installed, so an identity-of-value check would still
# let the stale hook delete it. A token makes ownership explicit instead.
_cleanup_owners: dict[tuple[int, str], object] = {}


@dataclasses.dataclass
class CleanupHook:
    # [CLASS DOC] Callable (usually a weakref/code-collection callback) that removes a name from a
    # scope dict. Fields: scope (the dict), name, and token (identity marker, excluded from
    # compare/repr) which must still be the registered owner of (scope, name) in _cleanup_owners
    # for the removal to happen. Created via the create static method.
    """Remove a global variable when hook is called"""

    scope: dict[str, Any]
    name: str
    token: object = dataclasses.field(compare=False, repr=False)

    def __call__(self, *args: Any) -> None:
        # [FUNCTION DOC] Decrements the CleanupManager count (unless the manager is already torn
        # down at shutdown), then removes the name from the scope only if this hook's token still
        # owns it; otherwise leaves the binding alone. Uses a tolerant pop so missing names do not
        # raise. Arguments are ignored.
        # Make sure we're not shutting down
        if CleanupManager is not None:
            CleanupManager.count -= 1
        # Hooks fire when the owning code object is collected, which can happen
        # after something else has taken over this name -- CompilePackage.install()
        # reinstalls precompiled state under names a pre-reset compile still
        # owns. Only clean up while nothing has claimed the name out from under us.
        key = (id(self.scope), self.name)
        if _cleanup_owners.pop(key, None) is not self.token:
            return
        # During interpreter shutdown the scope's module __dict__ may already
        # have been cleared, so the name can be gone. Use pop to avoid a
        # KeyError surfacing as an "Exception ignored in weakref callback".
        self.scope.pop(self.name, None)

    @staticmethod
    def create(scope: dict[str, Any], name: str, val: Any) -> CleanupHook:
        # [FUNCTION DOC] Installs val as scope[name] (AssertionError if the name already exists),
        # bumps CleanupManager.count, registers a fresh ownership token for (scope, name) and
        # returns a CleanupHook holding that token.
        if name in scope:
            raise AssertionError(f"Name {name!r} already exists in scope")
        CleanupManager.count += 1
        scope[name] = val
        token = object()
        _cleanup_owners[(id(scope), name)] = token
        return CleanupHook(scope, name, token)

    @staticmethod
    def disown(scope: dict[str, Any], name: str) -> None:
        # [FUNCTION DOC] Drops the ownership record for (scope, name) so any outstanding hook for
        # it becomes a no-op when fired.
        """Invalidate whichever hook currently owns (scope, name), so that if
        its code object outlives this call, firing later is a no-op instead of
        deleting a binding something else has since taken over."""
        _cleanup_owners.pop((id(scope), name), None)


class CleanupManager(ExactWeakKeyDictionary):
    # [CLASS DOC] Identity-keyed weak dictionary mapping objects (e.g. code objects) to lists of
    # cleanup hooks; when a key dies, its hooks are run. count tracks live hooks globally; a
    # singleton lives in the instance class attribute.
    count = 0
    instance: ClassVar[CleanupManager]

    def _remove_id(self, idx: int) -> None:
        # [FUNCTION DOC] HELPER: Calls every hook registered for the dead key, then performs the
        # normal entry removal.
        for hook in self.values[idx]:
            hook()
        super()._remove_id(idx)


CleanupManager.instance = CleanupManager()


def clone_tensor(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Returns a clone of x with the same requires_grad flag; if x is a leaf with a
    # gradient, the clone also gets a cloned gradient.
    """Clone the tensor and its gradient"""
    y = x.clone().requires_grad_(x.requires_grad)
    if x.is_leaf and x.grad is not None:
        y.grad = x.grad.clone()
    return y


def _copy_dynamo_attr(src: torch.Tensor, dst: torch.Tensor, attr: str) -> None:
    # [FUNCTION DOC] HELPER: Copies attribute attr from src to dst (using the value's copy method
    # when it has one, else the value itself); if src lacks it but dst has it, deletes it from dst.
    """Copy a single dynamo attribute from src to dst, or remove it from dst
    if src doesn't have it. Uses .copy() for container types (set, dict) to
    avoid shared references between src and dst, and assigns directly for
    primitive types like bool (e.g., _has_dynamo_dim_marking)."""
    if hasattr(src, attr):
        val = getattr(src, attr)
        setattr(dst, attr, val.copy() if hasattr(val, "copy") else val)
    elif hasattr(dst, attr):
        delattr(dst, attr)


def copy_dynamo_tensor_attributes(src: torch.Tensor, dst: torch.Tensor) -> None:
    # [FUNCTION DOC] Mirrors the dynamic-shape marking attributes (dynamic/unbacked indices, hint
    # overrides, shape ids, strict-unbacked, weak and propagated dynamic indices, dim-marking
    # flag) from src onto dst, removing those absent on src. Mutates dst.
    """
    Copy dynamo-specific tensor attributes from src to dst.
    These attributes are used for dynamic shape marking and must be preserved
    when cloning or casting tensors. If src doesn't have an attribute but dst does,
    the attribute is removed from dst.
    """
    _copy_dynamo_attr(src, dst, "_dynamo_dynamic_indices")
    _copy_dynamo_attr(src, dst, "_dynamo_unbacked_indices")
    _copy_dynamo_attr(src, dst, "_dynamo_hint_overrides")
    _copy_dynamo_attr(src, dst, "_dynamo_shape_ids")
    _copy_dynamo_attr(src, dst, "_dynamo_strict_unbacked_indices")
    _copy_dynamo_attr(src, dst, "_dynamo_weak_dynamic_indices")
    _copy_dynamo_attr(src, dst, "_dynamo_propagated_dynamic_indices")
    _copy_dynamo_attr(src, dst, "_has_dynamo_dim_marking")


def clone_input(x: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    # [FUNCTION DOC] Copies a tensor while preserving its strides (allocates a padded flat buffer
    # and as_strided view aligned like the source), optionally casting to dtype, preserving pinned
    # memory, requires_grad and leaf gradients, and dynamo attributes. Fake tensors are returned
    # as-is. Sparse COO/compressed, XLA and traceable-subclass tensors use a plain clone path; a
    # RuntimeError from copying into an overlapping layout falls back to a plain clone.
    """copy while preserving strides"""
    # TODO: this is questionable
    if is_fake(x):
        # this func fails on fake tensors in __torch_dispatch__
        return x

    def is_pinned_cpu_tensor(x: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True for non-subclass CPU tensors that are pinned.
        return (
            x.device.type == "cpu"
            and not is_traceable_wrapper_subclass(x)
            and x.is_pinned()
        )

    def torch_clone(x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Plain clone that re-pins memory if x was pinned, keeps
        # requires_grad for leaves, clones the leaf gradient (with dtype), and copies dynamo attrs.
        y = torch.clone(x)
        if is_pinned_cpu_tensor(x):
            y = y.pin_memory()
        if x.is_leaf:
            y.requires_grad_(x.requires_grad)
        if x.is_leaf and x.grad is not None:
            y.grad = clone_input(x.grad, dtype=dtype)
        copy_dynamo_tensor_attributes(x, y)
        return y

    with torch.no_grad():
        if x.device.type == "xla":
            # Access data_ptr() for a xla tensor will cause crash
            return torch_clone(x)

        # Handle sparse storage (no stride).
        if x.layout is torch.sparse_coo:
            return torch.sparse_coo_tensor(
                torch_clone(x._indices()),
                torch_clone(x._values()),
                x.shape,
                is_coalesced=x.is_coalesced(),
            )
        elif is_sparse_compressed(x):
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices = x.crow_indices()
                plain_indices = x.col_indices()
            else:
                compressed_indices = x.ccol_indices()
                plain_indices = x.row_indices()
            return torch.sparse_compressed_tensor(
                torch_clone(compressed_indices),
                torch_clone(plain_indices),
                torch_clone(x.values()),
                x.shape,
                layout=x.layout,
            )
        elif is_traceable_wrapper_subclass(x):
            # Questionable - but this is required to not fail executorch related
            # torchao tests.
            return torch_clone(x)

        needed_size = sum(
            (shape - 1) * stride for shape, stride in zip(x.size(), x.stride())
        )
        if x.is_quantized:
            result = torch.empty_quantized((needed_size + 32,), x)
        else:
            result = torch.empty(
                needed_size + 32,
                dtype=dtype or x.dtype,
                device=x.device,
                pin_memory=is_pinned_cpu_tensor(x),
            )
        cache_line_offset = (
            (x.data_ptr() - result.data_ptr()) % 32
        ) // x.element_size()
        result.as_strided_(x.size(), x.stride(), cache_line_offset)
        try:
            result.copy_(x.clone())
            if x.is_leaf:
                result.requires_grad_(x.requires_grad)
            if x.is_leaf and x.grad is not None:
                result.grad = clone_input(x.grad, dtype=dtype)
        except RuntimeError:
            # RuntimeError: unsupported operation: more than one element of the written-to
            # tensor refers to a single memory location. Please clone() the tensor before
            # performing the operation.
            return torch_clone(x)
        copy_dynamo_tensor_attributes(x, result)
        return result


@overload
def clone_inputs(
    example_inputs: dict[str, T | tuple[T, ...]],
) -> dict[str, list[T]]: ...


@overload
def clone_inputs(example_inputs: Sequence[T]) -> list[T]: ...


def clone_inputs(example_inputs: Any) -> Any:
    # [FUNCTION DOC] Clones example inputs. For an exact dict: tuple values are cloned recursively
    # and other values must be tensors (AssertionError otherwise), returning a new dict. Otherwise
    # returns a list in which tensor entries are cloned with clone_input and others kept as-is.
    res: dict[str, Any] | list[Any]
    if type(example_inputs) is dict:
        res = dict(example_inputs)
        for key, value in res.items():
            if isinstance(value, tuple):
                res[key] = clone_inputs(value)
            else:
                if not isinstance(value, torch.Tensor):
                    raise AssertionError(
                        f"Expected value to be a torch.Tensor, got {type(value)}"
                    )
                res[key] = clone_input(value)
        return res

    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = clone_input(res[i])
    return res


def skip_frame_if_in_functorch_mode(val: torch.Tensor) -> None:
    # [FUNCTION DOC] Probes val.data_ptr(); if that raises RuntimeError (functorch wrapper
    # tensors), triggers a graph break that skips the frame, naming the functorch subclass.
    # Returns None otherwise.
    try:
        val.data_ptr()  # will throw for functorch tensors
    except RuntimeError as e:
        from .exc import unimplemented

        # This will be GradTrackingTensor/BatchedTensor/etc
        functorch_subclass_name = re.sub(r"\(.*", "", repr(val))

        unimplemented(
            gb_type="skip frame due to being in functorh mode",
            context="",
            explanation=f"torch.compile cannot be run in context: {functorch_subclass_name}. Skipping frame.",
            hints=[],
            from_exc=e,
            skip_frame=True,
        )


@contextmanager
def preserve_rng_state() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that snapshots the CPU (and CUDA/XPU if available) RNG state
    # on entry, with functorch and dispatch modes disabled, and restores it on exit even after
    # exceptions. May skip the frame if running under functorch.
    disable_functorch = torch._C._DisableFuncTorch
    disable_current_modes = torch.utils._python_dispatch._disable_current_modes
    with disable_current_modes(), disable_functorch():
        rng_state = torch.clone(torch.random.get_rng_state())
        skip_frame_if_in_functorch_mode(rng_state)
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        if torch.xpu.is_available():
            xpu_rng_state = torch.clone(torch.xpu.get_rng_state())
    try:
        yield
    finally:
        with torch.utils._python_dispatch._disable_current_modes():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]
            if torch.xpu.is_available():
                torch.xpu.set_rng_state(xpu_rng_state)  # type: ignore[possibly-undefined]


def is_jit_model(
    model0: Any,
) -> TypeIs[
    torch.jit._trace.TopLevelTracedModule
    | torch.jit._script.RecursiveScriptModule
    | torch.jit.ScriptFunction[Any, Any]
    | torch.jit.ScriptModule
]:
    # [FUNCTION DOC] True if model0 is a TorchScript traced/scripted module or script function.
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
            torch.jit.ScriptModule,
        ),
    )


def torchscript(model: Any, example_inputs: Any, verbose: bool = False) -> Any:
    # [FUNCTION DOC] Returns model if already a JIT model; otherwise tries jit trace, then jit
    # script, returning the first success. If both fail, logs an error (with traceback if
    # verbose) and returns None.
    if is_jit_model(model):
        # already done?
        return model

    try:
        return torch.jit.trace(model, example_inputs)
    except Exception:
        try:
            return torch.jit.script(model)
        except Exception:
            if verbose:
                log.exception("jit error")
            else:
                log.error("Both torch.jit.trace and torch.jit.script failed")
    return None


def getfile(obj: object) -> str | None:
    # [FUNCTION DOC] Returns the source file path of obj via inspect, or None on TypeError/OSError.
    try:
        # inspect.getfile is typed to accept only a narrow union of callable
        # and module-like objects, but at runtime it accepts any object and
        # raises TypeError for unsupported ones, which this function catches.
        return inspect.getfile(obj)  # type: ignore[arg-type]
    except (TypeError, OSError):
        return None


def is_namedtuple(obj: object) -> bool:
    # [FUNCTION DOC] True if obj's type is a namedtuple or torch return-type quasi-namedtuple.
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    return is_namedtuple_cls(type(obj))


def is_namedtuple_cls(cls: object) -> bool:
    # [FUNCTION DOC] True if cls is a tuple subclass from torch.return_types / forward_ad, a
    # direct collections namedtuple (fields tuple and make method, base only tuple apart from
    # Generic), or a subclass of a namedtuple that does not override the constructor.
    # Non-class input and TypeError during inspection give False.
    """Test if an object is a namedtuple or a (torch.return_types|torch.autograd.forward_ad).* quasi-namedtuple"""
    if not isinstance(cls, type):
        return False
    try:
        if issubclass(cls, tuple):
            module = getattr(cls, "__module__", None)
            if module in ("torch.return_types", "torch.autograd.forward_ad"):
                return True
            if isinstance(getattr(cls, "_fields", None), tuple) and callable(
                getattr(cls, "_make", None)
            ):
                # The subclassing style namedtuple can have an extra base `typing.Generic`
                bases = tuple(t for t in cls.__bases__ if t is not Generic)
                if bases == (tuple,):
                    # This is a namedtuple type directly created by `collections.namedtuple(...)`
                    return True
                if bases and any(
                    (
                        # Subclass of namedtuple
                        is_namedtuple_cls(t)
                        # For subclasses of namedtuple, the __new__ method should not be customized
                        and cls.__new__ is t.__new__
                    )
                    for t in bases
                ):
                    return True
    except TypeError:
        pass
    return False


@functools.lru_cache(1)
def namedtuple_fields(cls: type) -> tuple[str, ...]:
    # [FUNCTION DOC] Returns the field names of a namedtuple class, ("start","stop","step") for
    # slice, or for torch.return_types classes discovers names by instantiating with marker
    # objects and ordering attributes by position. Raises AssertionError for non-tuple classes or
    # inconsistent field counts. Memoized with a cache of size one.
    # KNOWN ISSUE: lru_cache(1) keeps only the most recent class, so alternating classes defeat
    # the cache (possibly unintentional).
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    if cls is slice:
        return ("start", "stop", "step")

    if not issubclass(cls, tuple):
        raise AssertionError(f"Expected cls to be a subclass of tuple, got {cls}")
    if hasattr(cls, "_fields"):
        # normal namedtuples
        return cls._fields

    @dataclasses.dataclass
    class Marker:
        # [CLASS DOC] HELPER: Placeholder carrying a positional index, used to discover field
        # names of return-type tuples.
        index: int

    # frustrating ones e.g. torch.return_types.max
    if cls.__module__ != "torch.return_types":
        raise AssertionError(
            f"Expected cls.__module__ to be 'torch.return_types', got {cls.__module__!r}"
        )
    obj = cls(map(Marker, range(cls.n_fields)))  # type: ignore[attr-defined]
    fields: dict[str, int] = {}
    for name in dir(obj):
        if name[0] != "_" and isinstance(getattr(obj, name), Marker):
            fields[name] = getattr(obj, name).index
    if len(fields) != cls.n_fields:  # type: ignore[attr-defined]
        raise AssertionError(
            f"Expected {cls.n_fields} fields, got {len(fields)}"  # type: ignore[attr-defined]
        )
    return tuple(sorted(fields, key=fields.get))  # type: ignore[arg-type]


def checkpoint_params(gm: torch.fx.GraphModule) -> Callable[[], None]:
    # [FUNCTION DOC] Snapshots the RNG state (CPU and CUDA) and clones every parameter and buffer
    # of gm along with its version counter, returning a restore callable that resets the RNG and
    # copies back any parameter whose version changed.
    with torch.no_grad():
        rng_state = torch.clone(torch.random.get_rng_state())
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        saved_state = [
            (param, param._version, torch.clone(param))
            # pyrefly: ignore [bad-argument-type]
            for param in itertools.chain(gm.parameters(), gm.buffers())
        ]

    def restore() -> None:
        # [FUNCTION DOC] HELPER: Restores saved RNG state and any parameters/buffers modified
        # since the checkpoint (detected via version counters).
        with torch.no_grad():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            for param, version, original_value in saved_state:
                if param._version != version:
                    param.copy_(original_value)

    return restore


def timed(
    model: Any, example_inputs: Iterable[Any], times: int = 1
) -> tuple[Any, float]:
    # [FUNCTION DOC] Runs model on example_inputs the given number of times and returns (last
    # result, elapsed seconds). Synchronizes the accelerator (if any), garbage-collects, and
    # seeds the RNG with 1337 before timing. Side effects: gc and global RNG seeding.
    if torch.accelerator.is_available():
        synchronize = torch.accelerator.synchronize
    else:
        synchronize = nothing

    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0  # type: ignore[possibly-undefined]


def check_is_cuda(gm: torch.fx.GraphModule, example_inputs: Iterable[Any]) -> bool:
    # [FUNCTION DOC] True if every example input and every parameter of gm is on a CUDA device.
    return all(x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True)))


@lru_cache(32)
def rot_n_helper(n: int) -> Callable[..., Any]:
    # [FUNCTION DOC] Returns (cached) a generated function taking n positional args and returning
    # a tuple: the first n-1 arguments in reverse order followed by the last argument (used to
    # emulate stack rotation bytecode). The function is named for n. Raises AssertionError if n
    # is not greater than 1.
    if n <= 1:
        raise AssertionError(f"Expected n > 1, got {n}")
    vars = [f"v{i}" for i in range(n)]
    rotated = reversed(vars[-1:] + vars[:-1])
    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
    fn.__name__ = f"rot_{n}_helper"
    return fn


common_constant_types: set[type] = {
    int,
    float,
    complex,
    bool,
    str,
    bytes,
    type(None),
    Ellipsis.__class__,
    NotImplemented.__class__,
    types.CodeType,
    # Commonly used immutable types from torch.
    torch.device,
    torch.dtype,
    torch.memory_format,
    torch.layout,
    torch.finfo,
    torch.iinfo,
    torch.nn.attention.SDPBackend,
    torch.cuda._CudaDeviceProperties,
    # Pytree key types (frozen dataclasses used in tree_map_with_path)
    torch.utils._pytree.SequenceKey,
    torch.utils._pytree.MappingKey,
    torch.utils._pytree.GetAttrKey,
}

if has_triton_package():
    import triton

    common_constant_types.add(triton.language.dtype)

"""
    Difference between is_safe_constant and common_constant_types.
    * common_constant_types: Constants would be wrapped by VariableBuilder.wrap_literal
                             as ConstantVariable.
    * is_safe_constant: Constants can be loaded by LOAD_CONST bytecode.
"""


def is_safe_constant(v: object) -> bool:
    # [FUNCTION DOC] True if v can be loaded as a constant in bytecode: tuples/frozensets of safe
    # constants, enums, classes, torch.Size, generic aliases, or exact instances of the common
    # constant types or slice.
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    return isinstance(
        v,
        (
            enum.Enum,
            type,
            torch.Size,
            typing._GenericAlias,  # type: ignore[attr-defined]
            types.GenericAlias,
        ),
    ) or istype(
        v,
        common_constant_types | {slice},
    )


@functools.cache
def common_constants() -> set[int]:
    # [FUNCTION DOC] Returns (cached) the set of ints, zero and one, that are specialized because
    # shapes are zero-one specialized.
    return {
        # We zero-one specialize shapes, so specialize these constants
        # too
        0,
        1,
    }


def is_torch_sym(value: object) -> TypeGuard[torch.SymBool | torch.SymInt]:
    # [FUNCTION DOC] True if value is a SymInt or SymBool whose node is not a nested-int node.
    return isinstance(value, (torch.SymBool, torch.SymInt)) and not isinstance(
        value.node, torch.nested._internal.nested_int.NestedIntNode
    )


def is_int_specialization_case(value: Any, source: Any) -> bool:
    # [FUNCTION DOC] Decides whether a Python int from source should be specialized (made a
    # constant) rather than traced as symbolic. Requires the tracing context not to force
    # unspecialized unbacked ints; true for non-local (e.g. global) sources, nn-module sources
    # unless config.allow_unspec_int_on_nn_module, defaults sources, or (while a justknob rollout
    # is off) the values zero and one.
    from .source import is_from_defaults

    return not TracingContext.get().force_unspec_int_unbacked_size_like and (
        # Assume integers from global variables want to be specialized
        not source.guard_source.is_local()
        # Assume that integers that came from NN modules want to be
        # specialized (as we don't expect users to be changing the
        # NN modules on the fly), unless explicitly disabled
        or (
            source.guard_source.is_specialized_nn_module()
            and not config.allow_unspec_int_on_nn_module
        )
        or (
            source.guard_source.is_unspecialized_builtin_nn_module()
            and not config.allow_unspec_int_on_nn_module
        )
        or (
            source.guard_source.is_unspecialized_nn_module()
            and not config.allow_unspec_int_on_nn_module
        )
        or is_from_defaults(source)
        # TODO: Delete this condition when rollout is done.  NB: this
        # condition never evaluates True in open source
        or (
            not justknobs_check("pytorch/dynamo:enable_unspecialize_zero_one_plain_int")
            and value in common_constants()
        )
    )


def specialize_symnode(arg: Any) -> Any:
    # [FUNCTION DOC] If arg is an unrealized lazy variable that would not become a symbolic node,
    # returns it untouched; otherwise realizes it. A SymNodeVariable result is specialized to a
    # ConstantVariable of its evaluated expression (installing guards). Other values pass through.
    from .variables import ConstantVariable, LazyVariableTracker, SymNodeVariable
    from .variables.lazy import ComputedLazyConstantVariable

    # Guard and specialize
    if isinstance(arg, LazyVariableTracker):
        if not arg.is_realized() and not isinstance(arg, ComputedLazyConstantVariable):
            # Find if the arg would be realized as SymNodeVariable later on. If yes,
            # realize it and specialize. Else return the arg.

            source = arg.original_source()
            value = arg.original_value()

            is_symnode_vt = is_torch_sym(value) or (
                not config.specialize_int
                and type(value) is int
                and not is_int_specialization_case(value, source)
            )

            if not is_symnode_vt:
                return arg

        # Realize to get the underlying variable (handles both realized and unrealized)
        arg = arg.realize()

    if isinstance(arg, SymNodeVariable):
        return ConstantVariable.create(arg.evaluate_expr())
    return arg


def guard_if_dyn(arg: Any) -> Any:
    # [FUNCTION DOC] Specializes symbolic nodes in arg (see specialize_symnode) and, if the result
    # is a VariableTracker holding a Python constant, returns that raw constant; else the value.
    from .variables import VariableTracker

    arg = specialize_symnode(arg)

    if isinstance(arg, VariableTracker) and arg.is_python_constant():
        return arg.as_python_constant()

    return arg


def check_constant_args(args: Iterable[Any], kwargs: Mapping[Any, Any]) -> bool:
    # [FUNCTION DOC] True if all positional and keyword argument trackers are Python constants.
    return all(x.is_python_constant() for x in itertools.chain(args, kwargs.values()))


def check_unspec_python_args(args: Iterable[Any], kwargs: Mapping[Any, Any]) -> bool:
    # [FUNCTION DOC] True if every argument is a Python constant or an unspecialized Python
    # variable and at least one is unspecialized.
    from .variables import VariableTracker
    from .variables.tensor import UnspecializedPythonVariable

    unspec_count = 0
    for x in itertools.chain(args, kwargs.values()):
        if isinstance(x, UnspecializedPythonVariable):
            unspec_count += 1
        elif not (isinstance(x, VariableTracker) and x.is_python_constant()):
            return False
    return unspec_count > 0


def check_unspec_or_constant_args(
    args: Iterable[Any], kwargs: Mapping[Any, Any]
) -> bool:
    # [FUNCTION DOC] True if every argument is a Python constant or an unspecialized Python
    # variable (fused form of the constant check or the unspec check; note that it also returns
    # True when all arguments are constants).
    # A fused version of:
    # return check_constant_args(args, kwargs) or check_unspec_python_args(args, kwargs)
    from .variables.tensor import UnspecializedPythonVariable

    for x in itertools.chain(args, kwargs.values()):
        if not (x.is_python_constant() or isinstance(x, UnspecializedPythonVariable)):
            return False
    return True


def check_numpy_ndarray_args(args: Iterable[Any], kwargs: Mapping[Any, Any]) -> bool:
    # [FUNCTION DOC] True if any positional or keyword argument is a NumpyNdarrayVariable.
    from .variables.tensor import NumpyNdarrayVariable

    return any(
        isinstance(x, NumpyNdarrayVariable)
        for x in itertools.chain(args, kwargs.values())
    )


dict_keys: type[KeysView[Any]] = type({}.keys())
dict_values: type[ValuesView[Any]] = type({}.values())
dict_items: type[ItemsView[Any, Any]] = type({}.items())
odict_values: type[ValuesView[Any]] = type(OrderedDict().values())
# pyrefly: ignore [bad-assignment]
tuple_iterator: type[Iterator[Any]] = type(iter(()))
# pyrefly: ignore [bad-assignment]
range_iterator: type[Iterator[Any]] = type(iter(range(0)))
tuple_iterator_len = tuple_iterator.__length_hint__  # type: ignore[attr-defined]
deque_iterator = type(iter(collections.deque()))
deque_rev_iterator = type(reversed(collections.deque()))
object_new = object.__new__
dict_new = dict.__new__
dict_methods = {
    method
    for method in itertools.chain(dict.__dict__.values(), OrderedDict.__dict__.values())
    if callable(method)
}
set_methods = {method for method in set.__dict__.values() if callable(method)}
frozenset_methods = {
    method for method in frozenset.__dict__.values() if callable(method)
}
base_exception_methods = {
    method for method in BaseException.__dict__.values() if callable(method)
}
exception_methods = {
    method for method in Exception.__dict__.values() if callable(method)
}


tuple_new = tuple.__new__
tuple_methods = {method for method in tuple.__dict__.values() if callable(method)}
list_methods = {method for method in list.__dict__.values() if callable(method)}
list_getitem = list.__getitem__

deque_methods = {
    method for method in collections.deque.__dict__.values() if callable(method)
}

str_methods = {method for method in str.__dict__.values() if callable(method)}

# EnumType is the metaclass for Enum classes
enum_type_methods = {
    method for method in type(enum.Enum).__dict__.values() if callable(method)
}

K = TypeVar("K")
V = TypeVar("V")


def builtin_dict_keys(d: dict[K, V]) -> KeysView[K]:
    # [FUNCTION DOC] Returns the keys view using the builtin dict implementation (ignoring any
    # override). Raises AssertionError if d is not a dict.
    # Avoids overridden keys method of the dictionary
    if not isinstance(d, dict):
        raise AssertionError(f"Expected d to be a dict, got {type(d)}")
    return dict.keys(d)


def get_items_from_dict(obj: dict[K, V]) -> Iterable[tuple[K, V | Any]]:
    # [FUNCTION DOC] Returns the (key, value) pairs of a dict without calling user-overridden
    # item lookup or keys methods: items directly for exact dict/OrderedDict, otherwise a list
    # built with the base class lookups. Raises AssertionError if obj is not a dict.
    # Get items without calling the user defined __getitem__ or keys method.
    if not isinstance(obj, dict):
        raise AssertionError(f"Expected obj to be a dict, got {type(obj)}")
    if istype(obj, (dict, OrderedDict)):
        return obj.items()
    elif isinstance(obj, OrderedDict):
        return [(k, OrderedDict.__getitem__(obj, k)) for k in OrderedDict.keys(obj)]
    else:
        return [(k, dict.__getitem__(obj, k)) for k in dict.keys(obj)]


def enumerate_items_with_dict_position(
    obj: dict[K, V],
) -> Iterable[tuple[int, K, V | Any]]:
    """Enumerate dict items yielding (dict_keys_position, key, value).

    For OrderedDicts where move_to_end/prepend has been used, the OrderedDict
    iteration order can differ from dict.keys() order.  We iterate in
    OrderedDict order (correct execution semantics) but return each key's
    dict.keys() position so that ConstDictKeySource indices stay consistent
    with PyDict_Next / C++ DictGuardManager.
    """
    # [FUNCTION DOC] Returns an iterable of (position, key, value) triples, where position is the
    # key's index in the underlying dict key order (differs from iteration order for reordered
    # OrderedDicts) and the iteration order follows the object's own order.
    items = get_items_from_dict(obj)
    if isinstance(obj, OrderedDict):
        key_to_pos = {k: i for i, k in enumerate(dict.keys(obj))}
        return ((key_to_pos[k], k, v) for k, v in items)
    return ((i, k, v) for i, (k, v) in enumerate(items))


def nn_module_new(cls: Any) -> Any:
    # [FUNCTION DOC] Allocates an instance of cls without running its constructor and initializes
    # only the base nn.Module state. Returns the instance.
    obj = object_new(cls)
    torch.nn.Module.__init__(obj)
    return obj


def product(it: Iterable[T]) -> int:
    # [FUNCTION DOC] Multiplies all elements together, starting from one (one for empty input).
    return functools.reduce(operator.mul, it, 1)


def tuple_iterator_getitem(it: Any, index: int) -> Any:
    # [FUNCTION DOC] Returns the element at offset index from the current position of a tuple
    # iterator, using its reduce data (underlying tuple and start position) without consuming it.
    _, (obj,), start = it.__reduce__()
    return obj[start + index]


def dataclass_fields(cls: Any) -> Any:
    # [FUNCTION DOC] Returns the dataclass fields of cls, calling the fields function through
    # dynamo disable so it is not traced.
    return torch._dynamo.disable(dataclasses.fields)(cls)


iter_next = next


def normalize_count_iter(count_iter: Iterator[Any]) -> tuple[Any, Any]:
    # [FUNCTION DOC] Returns (start, step) of an itertools count object from its reduce data;
    # step defaults to 1. If reduce is unsupported (newer Python), parses literal arguments from
    # its repr. Returns (NotImplemented, NotImplemented) if they cannot be recovered.
    try:
        _, args = count_iter.__reduce__()
    except TypeError:
        # Python 3.14 no longer pickles itertools.count, so fall back to the
        # repr and only recover literal arguments. Non-literal arguments still
        # fall back to user-defined handling via the NotImplemented sentinel.
        import ast

        count_repr = repr(count_iter)
        if not count_repr.startswith("count(") or not count_repr.endswith(")"):
            return (NotImplemented, NotImplemented)
        try:
            args = ast.literal_eval(f"({count_repr[6:-1]},)")
        except (SyntaxError, ValueError):
            return (NotImplemented, NotImplemented)
        if not isinstance(args, tuple) or not 1 <= len(args) <= 2:
            return (NotImplemented, NotImplemented)
    if len(args) == 1:
        return (args[0], 1)
    return (args[0], args[1])


def normalize_range_iter(range_iter: Any) -> tuple[int, int, int]:
    # [FUNCTION DOC] Returns (start, stop, step) of the remaining portion of a range iterator,
    # advancing start by the number of steps already taken when the iterator reports an index.
    _, (range_obj,), maybe_idx = range_iter.__reduce__()
    # In 3.12+, `maybe_idx` could be None, and `range_obj.start` would've been
    # already incremented by the current index.
    # The index (maybe_idx) is the number of steps taken so far. To get the
    # correct start value, one must add (maybe_idx * step) to the original
    # start. See:
    # https://github.com/python/cpython/blob/ea77feecbba389916af8f90b2fc77f07910a2963/Objects/rangeobject.c#L885-L899
    start = range_obj.start + (maybe_idx or 0) * range_obj.step
    stop = range_obj.stop
    step = range_obj.step
    return (start, stop, step)


def to_subclass(t: Any, cls: type) -> Any:
    # [FUNCTION DOC] Views tensor t as an instance of the tensor subclass cls.
    return t.as_subclass(cls)


dict_getitem = dict.__getitem__


@torch.fx.wrap
def dict_keys_getitem(d: dict[Any, Any], n: int) -> Any:
    # [FUNCTION DOC] FX-wrapped function returning the n-th key of dict d in the dict's own key
    # order (matching C++ guard iteration). Raises StopIteration if n is out of range.
    # Use dict.keys() to match the iteration order of PyDict_Next used by
    # the C++ DictGuardManager and by ConstDictKeySource._name_template.
    # pyrefly: ignore [bad-argument-type]
    return next(itertools.islice(dict.keys(d), n, n + 1))


def set_base_iter(s: Iterable[T]) -> Iterator[T]:
    # [FUNCTION DOC] Returns an iterator over s using the base set or frozenset iteration for
    # (sub)instances of those types, bypassing overrides; the generic iterator otherwise.
    if isinstance(s, set):
        return set.__iter__(s)
    if isinstance(s, frozenset):
        return frozenset.__iter__(s)
    return iter(s)


def set_getitem(s: set[T], n: int) -> T:
    # [FUNCTION DOC] Returns the n-th element in the base iteration order of set s. Raises
    # IndexError if n is out of range. Order may not be stable across runs.
    # Set ordering might not be stable. For set/frozenset subclasses, use the
    # base iterator so guard reconstruction observes the internal set contents.
    try:
        return next(itertools.islice(set_base_iter(s), n, n + 1))
    except StopIteration:
        raise IndexError("set index out of range") from None


def enum_repr(value: Any, local: bool) -> str:
    # [FUNCTION DOC] Returns source-like text for an enum member: the locals or globals mapping
    # (chosen by local) indexed by the enum class name, then attribute access by member name.
    # enum class can override __str__ method. Use __class__ and name attribute
    # to extract the class name and key name.
    name = value.__class__.__name__
    val = value.name
    scope = "L" if local else "G"
    local_name = f'{scope}["{name}"].{val}'
    return local_name


def set_example_value(node: torch.fx.Node, example_value: Any) -> None:
    # [FUNCTION DOC] Stores example_value (a fake tensor or similar) in node metadata under
    # example_value, and records any unbacked symbol bindings computed from it in metadata.
    # Requires the tracing context to have a fake mode (AssertionError otherwise).
    # NB: example_value is a bit of a misnomer, because this is always a fake
    # tensor of some sort.  Furthermore, these example values serve as the
    # runtime state of Dynamo tracing, which means if metadata mutation
    # occurs, the example_value gets directly updated (so you can't rely on
    # this to accurately reflect what the state of the value was at the time
    # the program was traced).
    node.meta["example_value"] = example_value
    fake_mode = TracingContext.get().fake_mode
    if fake_mode is None:
        raise AssertionError("fake_mode must not be None")
    shape_env = fake_mode.shape_env
    if (
        symbol_to_path
        := torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings(
            shape_env, example_value
        )
    ):
        node.meta["unbacked_bindings"] = symbol_to_path


def _get_fake_tensor(vt: VariableTracker) -> Any:
    # [FUNCTION DOC] HELPER: Returns the fake example value on vt's proxy node; triggers a graph
    # break (Dynamo bug hint) if it is not a fake tensor.
    fake_tensor = vt.as_proxy().node.meta.get("example_value")
    if not is_fake(fake_tensor):
        from . import graph_break_hints
        from .exc import unimplemented

        unimplemented(
            gb_type="Cannot check Tensor object identity without its fake value",
            context=str(fake_tensor),
            explanation="TensorVariable is missing a fake example_value.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )
    return fake_tensor


def slice_length(s: slice, seq_len: int) -> int:
    # [FUNCTION DOC] Returns the number of elements selected when slice s is applied to a
    # sequence of length seq_len (never negative).
    start, stop, step = s.indices(seq_len)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def raise_args_mismatch(
    tx: InstructionTranslatorBase,
    name: str,
    expect: str = "",
    actual: str = "",
) -> NoReturn:
    # [FUNCTION DOC] Raises (inside the traced program) an observed TypeError whose message says
    # the call to name had the wrong number of positional or keyword arguments, showing the
    # expected and actual descriptions. Never returns.
    from torch._dynamo.exc import raise_observed_exception

    msg_str = (
        f"wrong number of arguments or keyword arguments for {name}() call.\n"
        f"  Expect: {expect}\n"
        f"  Actual: {actual}"
    )

    raise_observed_exception(
        TypeError,
        tx,
        args=[msg_str],
    )


def check_positional(
    tx: InstructionTranslatorBase,
    funcname: str,
    nargs: int,
    min_args: int,
    max_args: int,
) -> None:
    # [FUNCTION DOC] Validates that the positional argument count nargs lies between min_args and
    # max_args inclusive, otherwise raises an observed TypeError with CPython-style wording
    # ("expected at least/at most N argument(s), got M").
    # Mirrors CPython _PyArg_CheckPositional (Python/getargs.c): enforce
    # min_args <= nargs <= max_args with CPython's exact TypeError text. Used
    # for METH_FASTCALL methods, whose positional count MethodFlags (derived
    # from ml_flags) cannot check.
    from torch._dynamo.exc import raise_type_error

    if nargs < min_args:
        rel = "" if min_args == max_args else "at least "
        s = "" if min_args == 1 else "s"
        raise_type_error(
            tx, f"{funcname} expected {rel}{min_args} argument{s}, got {nargs}"
        )
    if nargs > max_args:
        rel = "" if min_args == max_args else "at most "
        s = "" if max_args == 1 else "s"
        raise_type_error(
            tx, f"{funcname} expected {rel}{max_args} argument{s}, got {nargs}"
        )


def no_positional(
    tx: InstructionTranslatorBase, funcname: str, args: list[VariableTracker]
) -> None:
    # [FUNCTION DOC] Raises an observed TypeError saying funcname takes no positional arguments if
    # args is nonempty; otherwise does nothing.
    # Mirrors CPython _PyArg_NoPositional (Python/getargs.c).
    from torch._dynamo.exc import raise_type_error

    if args:
        raise_type_error(tx, f"{funcname}() takes no positional arguments")


def no_keywords(
    tx: InstructionTranslatorBase, funcname: str, kwargs: dict[str, VariableTracker]
) -> None:
    # [FUNCTION DOC] Raises an observed TypeError saying funcname takes no keyword arguments if
    # kwargs is nonempty; otherwise does nothing.
    # Mirrors CPython _PyArg_NoKeywords / _PyArg_NoKwnames (Python/getargs.c).
    # For methods reached via tp_methods, MethodFlags already rejects kwargs;
    # use this only where MethodFlags does not run (e.g. tp_init constructors).
    from torch._dynamo.exc import raise_type_error

    if kwargs:
        raise_type_error(tx, f"{funcname}() takes no keyword arguments")


def iter_contains(
    items: Iterable[VariableTracker],
    search: VariableTracker,
    tx: InstructionTranslatorBase,
) -> VariableTracker:
    # [FUNCTION DOC] Symbolic membership test (like list/set contains). If search and all items are
    # Python constants, answers directly (identity or equality) as a ConstantVariable boolean.
    # Otherwise compares each item to search in order with the richcompare protocol, returning the
    # first definitely-true result, skipping definitely-false ones, and or-combining unknown
    # results; yields constant False if none matched.
    from .variables import ConstantVariable
    from .variables.object_protocol import generic_richcompare_bool

    items = list(items)
    # CPython's list_contains/set_contains use PyObject_RichCompareBool(item,
    # search, Py_EQ) with an identity shortcut. The constant fast path is only
    # valid when every element is a constant too; a non-constant element earlier
    # in the sequence has an __eq__ that must be honored in order (it may match,
    # or raise), so fall through to the per-element richcompare loop otherwise.
    if search.is_python_constant() and all(x.is_python_constant() for x in items):
        search_val = search.as_python_constant()
        found_const = any(
            x.as_python_constant() is search_val or x.as_python_constant() == search_val
            for x in items
        )
        return ConstantVariable.create(found_const)
    found: VariableTracker | None = None
    for x in items:
        check = generic_richcompare_bool(tx, x, search, "__eq__")
        if check.is_constant_match(True):
            return check
        if check.is_constant_match(False):
            continue
        if found is None:
            found = check
        else:
            from torch._dynamo.variables.builder import SourcelessBuilder

            found = SourcelessBuilder.create(tx, operator.or_).call_function(
                tx, [check, found], {}
            )
    if found is None:
        found = ConstantVariable.create(False)
    return found


def key_is_id(
    k: Any,
) -> TypeIs[torch.Tensor | torch.nn.Module | MethodWrapperType]:
    # [FUNCTION DOC] True if k is a tensor, nn.Module or method wrapper, i.e. a type that dict
    # guards identify by id.
    """Returns whether it indexes dictionaries using its id"""
    return isinstance(k, (torch.Tensor, torch.nn.Module, MethodWrapperType))


def key_to_id(value: Any) -> list[Any]:
    # [FUNCTION DOC] Returns a list of the keys in value, replacing id-keyed objects (see
    # key_is_id) by their integer id.
    return [id(k) if key_is_id(k) else k for k in value]


def const_repr(x: Any, *, local: Any) -> str:
    # [FUNCTION DOC] Renders a constant as source text for guard code: lists and tuples recursively
    # (one-element tuple with trailing comma), enums via enum_repr without quotes, builtin
    # callables by name, classes by fully qualified name of their class, and everything else by
    # its repr. local selects locals vs globals naming for enums.
    # KNOWN ISSUE: for a class x the helper uses the class of x (its metaclass), so the printed
    # name is the metaclass name (normally type), not the name of x itself.
    from .trace_rules import is_builtin_callable

    if isinstance(x, (list, tuple)):
        elems_repr = ",".join(const_repr(s, local=local) for s in x)
        if isinstance(x, list):
            return f"[{elems_repr}]"
        else:
            if not isinstance(x, tuple):
                raise AssertionError(f"Expected x to be a tuple, got {type(x)}")
            if len(x) == 1:
                return f"({elems_repr},)"
            else:
                return f"({elems_repr})"
    elif isinstance(x, enum.Enum):
        # To workaround repr(Enum) returning invalid global reference before python 3.11
        # by calling enum_repr and removing quotes to render enum in guard code.
        return enum_repr(x, local=local).replace("'", "")
    elif is_builtin_callable(x):
        return x.__name__
    elif isinstance(x, type):

        def fullname(o: Any) -> str:
            # [FUNCTION DOC] HELPER: Qualified name of o's class, omitting the module for builtins.
            klass = o.__class__
            module = klass.__module__
            if module == "builtins":
                return klass.__qualname__  # avoid outputs like 'builtins.str'
            return module + "." + klass.__qualname__

        return fullname(x)
    else:
        return f"{x!r}"


def dict_keys_repr(const_keys: Any, *, local: Any) -> str:
    # [FUNCTION DOC] Returns a bracketed, comma-joined source text of the constant keys using
    # const_repr.
    keys_str = ",".join(const_repr(s, local=local) for s in const_keys)
    return "[" + keys_str + "]"


GLOBAL_KEY_PREFIX = "__dict_key"


from torch._subclasses import UnsupportedFakeTensorException


def get_safe_global_name(tx: InstructionTranslatorBase, root: str, obj: Any) -> str:
    # [FUNCTION DOC] Returns a global name made from root, the object id, and the current output
    # graph's compile id so different torch.compile invocations never share a global name.
    # The global_mangled_class_name should be different for different
    # invocations of torch.compile. Otherwise, we can run into a situation
    # where multiple torch.compile invocations reuse the same global name,
    # but the global's lifetime is tied to the first invocation (and
    # may be deleted when the first torch.compile invocation is deleted)
    # We mangle it based off of the output_graph's id.
    return f"{root}_{id(obj)}_c{tx.output.compile_id}"


def is_in(item: T, *containers: Container[T]) -> bool:
    # [FUNCTION DOC] True if item is contained in any of the given containers.
    for container in containers:
        if item in container:
            return True
    return False


def get_unique_name_wrt(
    prefix: str, *containers: Any, requires_suffix: bool = False
) -> str:
    # [FUNCTION DOC] Returns prefix itself if unused (and a suffix is not required); otherwise the
    # first prefix_N (N counting from zero) not present in any container.
    """
    Return a name that starts with `prefix` and is not in any of the
    `containers` (e.g., map, set).
    """
    if not requires_suffix and not is_in(prefix, *containers):
        return prefix

    for i in itertools.count():
        candidate = f"{prefix}_{i}"
        if not is_in(candidate, *containers):
            return candidate

    raise AssertionError("unreachable")


def wrap_fake_exception(fn: Callable[[], Any]) -> Any:
    # [FUNCTION DOC] Calls fn and returns its result; converts UnsupportedFakeTensorException into
    # a logged warning and a graph break (unimplemented) chained from the original error.
    try:
        return fn()
    except UnsupportedFakeTensorException as e:
        from .exc import unimplemented

        msg = f"Encountered exception ({e.reason}) during fake tensor propagation."
        log.warning(msg)
        unimplemented(
            gb_type="Fake tensor propagation exception",
            context=str(e.reason),
            explanation=msg,
            hints=[],
            from_exc=e,
        )


def deepcopy_to_fake_tensor(
    obj: Any, fake_mode: torch._subclasses.fake_tensor.FakeTensorMode
) -> Any:
    # [FUNCTION DOC] Deep-copies obj with tensors converted to fake tensors under fake_mode (copy
    # mode), turning unsupported fake tensor errors into graph breaks. Returns the copy.
    with torch._subclasses.fake_tensor.FakeCopyMode(fake_mode):
        return wrap_fake_exception(lambda: copy.deepcopy(obj))


def rmse(ref: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Returns the root mean squared error between the two tensors as a scalar
    # tensor.
    """
    Calculate root mean squared error
    """
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def bitwise_same(ref: Any, res: Any, equal_nan: bool = False) -> bool:
    # [FUNCTION DOC] Exact-equality variant of same: compares ref and res with zero tolerance.
    return same(
        ref,
        res,
        tol=0.0,
        equal_nan=equal_nan,
    )


def same(
    ref: Any,
    res: Any,
    fp64_ref: Any = None,
    cos_similarity: bool = False,
    tol: float = 1e-4,
    equal_nan: bool = False,
    exact_dtype: bool = True,
    relax_numpy_equality: bool = False,
    ignore_non_fp: bool = False,
    log_error: Callable[..., None] = log.error,
    use_larger_multiplier_for_smaller_tensor: bool = False,
    force_max_multiplier: bool = False,
    use_iou_for_bool: bool = False,
    iou_threshold: float = 0.99,
) -> bool:
    # [FUNCTION DOC] Recursive accuracy comparison of a reference result and a result. Handles
    # sequences (length and elementwise), dicts (key sets then per key), sets, tensors/floats
    # (dtype check, bool tensors via uint8 compare or IoU, cosine similarity mode, allclose with
    # tolerance, then an RMSE-vs-fp64-reference test with a dtype/size-dependent multiplier),
    # str/int/None/bool/device (equality), numpy scalars and arrays, and known HF-style output
    # objects (compared attribute by attribute). Returns True if matching; failures are reported
    # through log_error and return False. Raises AssertionError for structural mismatches such as
    # differing types, dict keys or set contents, and RuntimeError for unsupported types.
    # KNOWN ISSUE: sequence length mismatch, dict key mismatch and set mismatch behave
    # inconsistently (return False vs raise AssertionError).
    """Check correctness to see if ref and res match"""
    if fp64_ref is None:
        fp64_ref = ref
    if isinstance(
        ref, (list, tuple, collections.deque, torch.nn.ParameterList, torch.Size)
    ):
        if not isinstance(res, (list, tuple, collections.deque)):
            raise AssertionError(f"type mismatch {type(ref)} {type(res)}")
        if len(ref) != len(res):
            log_error("Length mismatch")
            return False
        return len(ref) == len(res) and all(
            same(
                ai,
                bi,
                fp64_refi,
                cos_similarity,
                tol,
                equal_nan,
                exact_dtype,
                relax_numpy_equality,
                ignore_non_fp,
                log_error=log_error,
                use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
                force_max_multiplier=force_max_multiplier,
                use_iou_for_bool=use_iou_for_bool,
                iou_threshold=iou_threshold,
            )
            for ai, bi, fp64_refi in zip(ref, res, fp64_ref)
        )
    elif type(ref).__name__ == "QuestionAnsweringModelOutput":
        # This skips checking accuracy for start_logits/end_logits.
        # Tentatively, start_logits/end_logits appear to be very prone to
        # inaccuracies and is somewhat subsumed by checking the loss.
        return same(
            ref.loss,
            res.loss,
            fp64_ref.loss,
            cos_similarity,
            tol,
            equal_nan,
            exact_dtype,
            relax_numpy_equality,
            ignore_non_fp,
            log_error=log_error,
            use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
            force_max_multiplier=force_max_multiplier,
            use_iou_for_bool=use_iou_for_bool,
            iou_threshold=iou_threshold,
        )
    elif isinstance(ref, dict):
        if not isinstance(res, dict):
            raise AssertionError(f"Expected res to be a dict, got {type(res)}")
        if set(ref.keys()) != set(res.keys()):
            raise AssertionError(
                f"keys mismatch {set(ref.keys())} == {set(res.keys())}"
            )
        for k in sorted(ref.keys()):
            if not (
                same(
                    ref[k],
                    res[k],
                    fp64_ref[k],
                    cos_similarity=cos_similarity,
                    tol=tol,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                    relax_numpy_equality=relax_numpy_equality,
                    ignore_non_fp=ignore_non_fp,
                    log_error=log_error,
                    use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
                    force_max_multiplier=force_max_multiplier,
                    use_iou_for_bool=use_iou_for_bool,
                    iou_threshold=iou_threshold,
                )
            ):
                log_error("Accuracy failed for key name %s", k)
                return False
        return True
    elif isinstance(ref, set):
        if not isinstance(res, set):
            raise AssertionError(f"Expected res to be a set, got {type(res)}")
        if set(ref) != set(res):
            raise AssertionError(f"elements mismatch {set(ref)} == {set(res)}")
        return True
    elif isinstance(ref, (torch.Tensor, float)):
        if torch._subclasses.fake_tensor.is_fake_tensor(ref):
            raise AssertionError("ref should not be a FakeTensor")
        if torch._subclasses.fake_tensor.is_fake_tensor(res):
            raise AssertionError("res should not be a FakeTensor")

        def to_tensor(t: Any) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns t if already a tensor, else wraps it as a tensor.
            return t if isinstance(t, torch.Tensor) else torch.tensor(t)

        ref, res, fp64_ref = (to_tensor(val) for val in (ref, res, fp64_ref))

        if ref.is_sparse:
            if not res.is_sparse:
                raise AssertionError("Expected res to be sparse since ref is sparse")
            ref = ref.to_dense()
            res = res.to_dense()
        if not isinstance(res, torch.Tensor):
            raise AssertionError(f"type mismatch {type(ref)} {type(res)}")
        if exact_dtype:
            if ref.dtype != res.dtype:
                log_error("dtype mismatch %s, %s", ref.dtype, res.dtype)
                return False
            if ref.dtype == torch.bool:
                if ignore_non_fp:
                    return True
                if use_iou_for_bool:
                    # Use IoU (Intersection over Union) metric for boolean mask comparison.
                    # This is useful for segmentation models where small floating-point
                    # differences get thresholded into boolean masks.
                    intersection = (ref & res).sum().float()
                    union = (ref | res).sum().float()
                    if union == 0:
                        # Both masks are empty
                        return bool(intersection == 0)
                    iou = (intersection / union).item()
                    if iou < iou_threshold:
                        log_error(
                            "IoU accuracy failed: %.4f < %.2f (intersection=%d, union=%d, ref_sum=%d, res_sum=%d, shape=%s)",
                            iou,
                            iou_threshold,
                            int(intersection.item()),
                            int(union.item()),
                            int(ref.sum().item()),
                            int(res.sum().item()),
                            list(ref.shape),
                        )
                        return False
                    return True
                # triton stores bool as int8, so add this for more accurate checking
                r = torch.allclose(
                    ref.to(dtype=torch.uint8),
                    res.to(dtype=torch.uint8),
                    atol=tol,
                    rtol=tol,
                    equal_nan=equal_nan,
                )
                if not r:
                    log_error("Accuracy failed: uint8 tensor did not match")
                return r

        if cos_similarity:
            ref = ref.flatten().to(torch.float32)
            res = res.flatten().to(torch.float32)
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=True):
                # early exit that handles zero/nan better
                # cosine_similarity(zeros(10), zeros(10), dim=0) is 0
                return True
            score = torch.nn.functional.cosine_similarity(ref, res, dim=0, eps=1e-6)
            if score < 0.99:
                log.warning("Similarity score=%s", score.detach().cpu().item())
            return bool(score >= 0.99)
        else:
            if not exact_dtype:
                ref = ref.to(res.dtype)

            # First try usual allclose
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=equal_nan):
                return True

            # Check error from fp64 version
            if fp64_ref.dtype == torch.float64:
                # Fix a corner case that res and fp64_ref does not contains NaN and match (with loose tolerance)
                # while the ref contains NaN. In this case, RMSE should not match any ways.
                # But res is 'BETTER' than ref so we count it pass.
                #
                # This happens for Super_SloMo when loop ordering after fusion is enabled:
                # https://gist.github.com/shunting314/11f235c70f7db0d52718d26f4a701cab
                loose_tol = 1e-2 * 4
                if (
                    not fp64_ref.isnan().any()
                    and not res.isnan().any()
                    and ref.isnan().any()
                    and torch.allclose(
                        fp64_ref.to(dtype=res.dtype),
                        res,
                        atol=loose_tol,
                        rtol=loose_tol,
                        equal_nan=equal_nan,
                    )
                ):
                    return True
                ref_error = rmse(fp64_ref, ref).item()
                # ref unable to produce this with stable numerics in this precision, ignore
                if math.isnan(ref_error):
                    log.warning(
                        "Found nan in reference. Consider running in higher precision."
                    )

                res_error = rmse(fp64_ref, res).item()

                def get_multiplier() -> float:
                    # [FUNCTION DOC] HELPER: Chooses how many times the reference's RMSE error
                    # the result may have: 10 if forced; otherwise 2, or 3 for half precision,
                    # raised to 10 or 8 for tiny tensors when requested, or to 3 for small
                    # tensors, 1x1 spatial 4D tensors, or large tolerances.
                    # In some particular cases, we expect high difference in results.
                    # At the moment one of these cases is inductor freezing bfloat16 convolution const folding.
                    # In this case, the res_error is at least one order of magnitude higher.
                    if force_max_multiplier:
                        return 10.0
                    # In the case of using AMP (Automatic Mixed Precision), certain models have
                    # failed the benchmark's correctness check. However, the end-to-end model's
                    # accuracy when comparing AMP with FP32 is within a difference of less than 0.1%.
                    # Thus, it's possible that the correctness check failures for these models are
                    # false alarms. We use multiplier of 3 instead of 2 to avoid these false alarms.
                    multiplier = (
                        3.0 if res.dtype in (torch.float16, torch.bfloat16) else 2.0
                    )

                    if use_larger_multiplier_for_smaller_tensor and (
                        fp64_ref.numel() <= 10
                    ):
                        multiplier = 10.0
                    elif use_larger_multiplier_for_smaller_tensor and (
                        fp64_ref.numel() <= 500
                    ):
                        multiplier = 8.0
                    elif (
                        fp64_ref.numel() < 1000
                        or (ref.ndim == 4 and ref.shape[-1] == ref.shape[-2] == 1)
                        # large tol means a benchmark has been specified as REQUIRE_HIGHER_TOLERANCE
                        or tol >= 2 * 1e-2
                    ):
                        # In the presence of noise, noise might dominate our error
                        # metric for smaller tensors.
                        # Similarly, for 1x1 kernels, there seems to be high noise with amp.
                        multiplier = 3.0
                    return multiplier

                multiplier = get_multiplier()

                passes_test = res_error <= (multiplier * ref_error + tol / 10.0)
                if (
                    not passes_test
                    and equal_nan
                    and math.isnan(ref_error)
                    and math.isnan(res_error)
                    # Some unit test for the accuracy minifier relies on
                    # returning false in this case.
                    and not torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY
                ):
                    passes_test = True
                if not passes_test:
                    log_error(
                        "RMSE (res-fp64): %.5f, (ref-fp64): %.5f and shape=%s. res.dtype: %s, multiplier: %f, tol: %f"
                        ", use_larger_multiplier_for_smaller_tensor: %d",
                        res_error,
                        ref_error,
                        res.size(),
                        res.dtype,
                        multiplier,
                        tol,
                        use_larger_multiplier_for_smaller_tensor,
                    )
                return passes_test

            if ignore_non_fp:
                return True

            log_error("Accuracy failed: allclose not within tol=%s", tol)
            return False
    elif isinstance(ref, (str, int, type(None), bool, torch.device)):
        if ignore_non_fp:
            return True
        r = ref == res
        if not r:
            log_error("Accuracy failed (%s): %s != %s", type(ref), ref, res)
        return r
    elif is_numpy_int_type(ref) or is_numpy_float_type(ref):
        if relax_numpy_equality and not (
            is_numpy_int_type(res) or is_numpy_float_type(res)
        ):
            ref = ref.item()
        r = (type(ref) is type(res)) and (ref == res)
        if not r:
            log_error("Accuracy failed (numpy): %s != %s", ref, res)
        return r
    elif is_numpy_ndarray(ref):
        return (type(ref) is type(res)) and same(
            torch.as_tensor(ref),
            torch.as_tensor(res),
            fp64_ref,
            cos_similarity=cos_similarity,
            tol=tol,
            equal_nan=equal_nan,
            exact_dtype=exact_dtype,
            relax_numpy_equality=relax_numpy_equality,
            ignore_non_fp=ignore_non_fp,
            log_error=log_error,
            use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
            force_max_multiplier=force_max_multiplier,
            use_iou_for_bool=use_iou_for_bool,
            iou_threshold=iou_threshold,
        )
    elif type(ref).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        if type(ref) is not type(res):
            raise AssertionError(f"type mismatch {type(ref)} {type(res)}")
        return all(
            same(
                getattr(ref, key),
                getattr(res, key),
                getattr(fp64_ref, key),
                cos_similarity=cos_similarity,
                tol=tol,
                equal_nan=equal_nan,
                exact_dtype=exact_dtype,
                relax_numpy_equality=relax_numpy_equality,
                ignore_non_fp=ignore_non_fp,
                log_error=log_error,
                use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
                force_max_multiplier=force_max_multiplier,
                use_iou_for_bool=use_iou_for_bool,
                iou_threshold=iou_threshold,
            )
            for key in ref.__dict__
        )
    else:
        raise RuntimeError(f"unsupported type: {type(ref).__name__}")


def format_func_info(code: CodeType) -> str:
    # [FUNCTION DOC] Returns a description of a code object as quoted name followed by
    # (basename:first line number).
    short_filename = code.co_filename.split("/")[-1]
    return f"'{code.co_name}' ({short_filename}:{code.co_firstlineno})"


@contextlib.contextmanager
def disable_cache_limit() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that sets the dynamo recompile limit and accumulated
    # recompile limit to the maximum integer, restoring both previous values on exit (even on
    # exceptions). Mutates global config temporarily.
    prior = config.recompile_limit
    # pyrefly: ignore [bad-assignment]
    config.recompile_limit = sys.maxsize
    prior_acc_limit = config.accumulated_recompile_limit
    # pyrefly: ignore [bad-assignment]
    config.accumulated_recompile_limit = sys.maxsize

    try:
        yield
    finally:
        config.recompile_limit = prior
        config.accumulated_recompile_limit = prior_acc_limit


# map from transformed code back to original user code
orig_code_map = ExactWeakKeyDictionary()

# keep a record of code_obj -> list of guard failure reasons for logging
guard_failures: collections.defaultdict[Any, list[Any]] = collections.defaultdict(list)

# Keep a record of graph break reasons for logging
graph_break_reasons: list[torch._dynamo.output_graph.GraphCompileReason] = []

# keep record of compiled code, if we are in "error if recompile"
# to track code that dynamo has compiled previously
seen_code_map = ExactWeakKeyDictionary()


# return same dir unless user changes config between calls
@functools.cache
def _get_debug_dir(root_dir: str) -> str:
    # [FUNCTION DOC] HELPER: Returns root_dir joined with a run directory name made of the current
    # timestamp and process id. Cached per root_dir so the same directory is reused until the
    # config root changes.
    dir_name = (
        "run_"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # use pid to avoid conflicts among ranks
        + "-pid_"
        + str(os.getpid())
    )
    return os.path.join(root_dir, dir_name)


def get_debug_dir() -> str:
    # [FUNCTION DOC] Returns the per-run debug directory path for the configured debug root.
    debug_root = config.debug_dir_root
    return _get_debug_dir(debug_root)


def extract_fake_example_value(node: torch.fx.Node, required: bool = True) -> Any:
    # [FUNCTION DOC] Returns the fake tensor stored as the node's example value. If missing or not
    # fake: triggers a graph break (Dynamo bug hint) when required, else returns None.
    if "example_value" in node.meta and is_fake(node.meta["example_value"]):
        return node.meta["example_value"]
    elif required:
        from torch._dynamo.exc import unimplemented

        from . import graph_break_hints

        unimplemented(
            gb_type="Missing FakeTensor example value",
            context=str(node),
            explanation=f"`FakeTensor` example value was required for {node} but not available.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )
    else:
        return None


def ensure_graph_fake(e: Any, tx: InstructionTranslatorBase) -> Any:
    # [FUNCTION DOC] Returns e after asserting its fake mode is the translator's fake mode
    # (AssertionError otherwise).
    from torch._subclasses.fake_tensor import maybe_get_fake_mode

    if maybe_get_fake_mode(e) is not tx.fake_mode:
        raise AssertionError(
            f"Expected fake mode of e to be tx.fake_mode, got {maybe_get_fake_mode(e)} vs {tx.fake_mode}"
        )
    return e


def get_fake_values_from_nodes(
    tx: InstructionTranslatorBase, nodes: Any, allow_non_graph_fake: bool
) -> Any:
    # [FUNCTION DOC] Maps a nested structure of FX nodes to their fake values: computes missing
    # call_function values via get_fake_value, returns the submodule for get_attr nodes, and
    # otherwise the stored example value (validated as belonging to this graph's fake mode for
    # tensors unless allow_non_graph_fake).
    def visit(n: torch.fx.Node) -> Any:
        # [FUNCTION DOC] HELPER: Resolves a single node to its fake value, submodule, or stored
        # example value as described by the enclosing function.
        if n.op == "call_function" and "example_value" not in n.meta:
            # fake tensor validity is checked inside get_fake_value using
            # ensure_graph_fake
            return get_fake_value(n, tx, allow_non_graph_fake)

        elif n.op == "get_attr" and "example_value" not in n.meta:
            if n.target not in tx.output.nn_modules:
                raise AssertionError(
                    f"Expected n.target {n.target!r} to be in tx.output.nn_modules"
                )
            gm = tx.output.nn_modules[n.target]  # type: ignore[index]
            if not isinstance(gm, torch.fx.GraphModule):
                raise AssertionError(
                    f"Expected gm to be a torch.fx.GraphModule, got {type(gm)}"
                )
            return gm

        out = n.meta["example_value"]
        if not allow_non_graph_fake and isinstance(out, torch.Tensor):
            return ensure_graph_fake(out, tx)
        return out

    return torch.fx.node.map_arg(nodes, visit)


def get_concrete_sizes_from_symints(msg: str, fake_mode: FakeTensorMode | None) -> str:
    """
    Replace symbolic size expressions (like 's0', 's94') in error messages
    with their concrete runtime values for better readability.

    Example: "size (s94)" -> "size (s94: hint= 10)" if s94's value is 10.
    """
    # [FUNCTION DOC] Rewrites occurrences of a parenthesized symbol name (s followed by digits) in
    # msg to include the backed hint value from the fake mode's shape env when it is an integer;
    # other matches are unchanged. Returns msg as-is if fake_mode is None; AssertionError if the
    # shape env is missing.
    # KNOWN ISSUE: the docstring example shows "hint= 10" but the code emits "hint = 10".
    import re

    from sympy.core.numbers import Integer

    if fake_mode is None:
        return msg

    pattern = r"\(s(\d+)\)"
    if fake_mode.shape_env is None:
        raise AssertionError("fake_mode.shape_env must not be None")
    shape_env = fake_mode.shape_env
    backed_var_to_val = shape_env.backed_var_to_val

    def replace_sym(match: Any) -> str:
        # [FUNCTION DOC] HELPER: Regex substitution callback giving the replacement text for one
        # symbol match (with hint if an integer value is known, else the original text).
        sym_name = f"s{match.group(1)}"
        val = next(
            (v for k, v in backed_var_to_val.items() if k.name == sym_name),
            None,
        )
        if isinstance(val, (int, Integer)):
            return f"({sym_name}: hint = {str(val)})"
        return match.group(0)

    msg = re.sub(pattern, replace_sym, msg)
    return msg


def _wrap_graph_break_with_torch_runtime_err(gb_fn: Callable[[], NoReturn]) -> NoReturn:
    # [FUNCTION DOC] HELPER: Calls gb_fn (expected to raise a graph break) and converts the
    # resulting Unsupported error into a TorchRuntimeError with the same message, real stack and
    # traceback. Raises AssertionError if gb_fn returns normally.
    from .exc import TorchRuntimeError, Unsupported

    try:
        gb_fn()
    except Unsupported as e:
        exc = TorchRuntimeError(str(e), getattr(e, "real_stack", None))
        raise exc.with_traceback(e.__traceback__) from None
    raise AssertionError("should be unreachable")


def _custom_op_fake_impl_pending(tx: InstructionTranslatorBase, target: Any) -> bool:
    # [FUNCTION DOC] HELPER: True if target is a custom op overload whose definition object is
    # tracked in the current trace's side effects with a pending (unapplied) mutation of its fake
    # implementation attribute; False for non-overloads, ops without a definition, or untracked
    # definitions.
    """True if ``target`` is a custom op whose ``register_fake`` ran earlier in
    this trace -- i.e. its ``_abstract_fn`` has a pending (not-yet-applied)
    attribute mutation. Distinguishes an op defined+register_fake'd inside the
    compiled function (recoverable via graph break) from a genuinely
    unregistered op (should hard-error with actionable guidance).

    Only detects the ``CustomOpDef.register_fake`` method form (a tracked
    pending ``_abstract_fn`` store on ``tx.output.side_effects``); other forms
    (e.g. an op with no ``CustomOpDef``, or a definition inside a HOP body whose
    mutation lives in a subtracer) fall through to the hard error, preserving
    pre-existing behavior."""
    if not isinstance(target, torch._ops.OpOverload):
        return False
    from torch._library.custom_ops import _maybe_get_opdef

    opdef = _maybe_get_opdef(target)
    if opdef is None:
        return False
    side_effects = tx.output.side_effects
    vt = side_effects.id_to_variable.get(id(opdef))
    if vt is None:
        return False
    return side_effects.has_pending_mutation_of_attr(vt, "_abstract_fn")


def get_fake_value(
    node: torch.fx.Node,
    tx: InstructionTranslatorBase,
    allow_non_graph_fake: bool = False,
) -> Any:
    # [FUNCTION DOC] Computes the fake value of an FX node by delegating to _get_fake_value_impl
    # and adds the elapsed nanoseconds to the output graph's get_fake_value timing counter
    # (even on exceptions).
    _t0 = time.time_ns()
    try:
        return _get_fake_value_impl(node, tx, allow_non_graph_fake)
    finally:
        tx.output.bytecode_tracing_timings.get_fake_value_ns += time.time_ns() - _t0


def _get_fake_value_impl(
    node: torch.fx.Node,
    tx: InstructionTranslatorBase,
    allow_non_graph_fake: bool = False,
) -> Any:
    # [FUNCTION DOC] HELPER: Runs the node's operation on fake inputs under the fake mode and
    # returns the fake result (cached example value if already present). Copies nn.Module
    # arguments and call_module targets to fake mode, initializes lazy modules, specializes
    # symbolic floats in some cases, and translates failures into graph breaks or user errors
    # (data-dependent or dynamic-shape ops, unsupported operators, device mismatch, constraint
    # violations, bad arguments). Unless allow_non_graph_fake, verifies tensor results belong to
    # this graph's fake mode, and when graph deduplication tracking is enabled records input
    # mutations with the region tracker.
    """
    Run the computation represented by `node` using fake tensors and return the result.

    allow_non_graph_fake: whether to allow the return result to be:
        1. non-fake or 2. fake that is not created by this instance of Dynamo.
        If `True`, you must be prepared to deal with such return values, ideally
        by further wrapping them as this graph's fakes.
    """
    from torch.utils._sympy.value_ranges import ValueRangeError

    from . import graph_break_hints
    from .exc import unimplemented, Unsupported, UserError, UserErrorType

    op = node.op

    # FX Node should always return the same fake value
    if "example_value" in node.meta and is_fake(node.meta["example_value"]):
        return node.meta["example_value"]

    args, kwargs = get_fake_values_from_nodes(
        tx, (node.args, node.kwargs), allow_non_graph_fake
    )

    if (
        torch._dynamo.config.use_graph_deduplication
        or torch._dynamo.config.track_nodes_for_deduplication
    ):
        flat_args_kwargs = get_fake_values_from_nodes(
            tx, _get_flat_args(node, {}), allow_non_graph_fake
        )
        id_to_initial_version = {
            id(arg): arg._version for arg in flat_args_kwargs if is_fake(arg)
        }
    else:
        # pyrefly: ignore [implicit-any]
        flat_args_kwargs = []
        # pyrefly: ignore [implicit-any]
        id_to_initial_version = {}

    nnmodule = None
    fake_mode = tx.fake_mode
    if fake_mode is None:
        raise AssertionError("fake_mode must not be None")
    if op == "call_method" and len(args) > 0 and isinstance(args[0], torch.nn.Module):
        # If the first argument is nn.Module, should copy to fake mode.
        args = (deepcopy_to_fake_tensor(args[0], fake_mode),) + tuple(args[1:])

    if op == "call_module":
        nnmodule = tx.output.nn_modules[node.target]  # type: ignore[index]

        if (
            is_lazy_module(nnmodule)
            and inspect.getattr_static(nnmodule, "_initialize_hook", None) is not None
        ):
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it.
            # Afterwards, lazy module deletes its pre-hooks
            # to avoid treating it as lazy on subsequent recompile.
            nnmodule._infer_parameters(nnmodule, args)

        # no matter it's lazy module or not, we should copy to fake mode.
        nnmodule = deepcopy_to_fake_tensor(nnmodule, fake_mode)

    if node.name in ["interpolate", "is_integer", "wrapped_gradient"] or any(
        isinstance(a, complex) for a in args
    ):
        # We need to specialize symfloats for now. Eventually we should do a tensorify pass in dynamo.
        args = tuple(
            (
                float(arg)
                if isinstance(arg, torch.SymFloat) and arg.node.hint is not None
                else arg
            )
            for arg in args
        )

    try:
        from torch._dynamo.eval_frame import _use_eager_on_nested_compile

        with fake_mode, enable_python_dispatcher(), _use_eager_on_nested_compile():
            ret_val = wrap_fake_exception(
                lambda: run_node(tx.output, node, args, kwargs, nnmodule)
            )
    except Unsupported:
        raise
    except RuntimeError as e:
        cause: BaseException = e
        if e.__cause__ is not None:
            cause = e.__cause__

        if isinstance(
            cause, torch._subclasses.fake_tensor.DataDependentOutputException
        ):
            # capture_scalar_outputs only works for these ops right now
            # see torch/_subclasses/fake_impls.py
            if cause.func in (
                torch.ops.aten.item.default,
                torch.ops.aten._local_scalar_dense.default,
            ):
                # does this actually get triggered?
                hints = [
                    "Enable tracing of data-dependent output operators with "
                    "`torch._dynamo.config.capture_scalar_outputs = True`",
                ]
            else:
                hints = [
                    "Consider wrapping the operator into a PyTorch-understood custom operator "
                    "(see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)",
                ]
            unimplemented(
                gb_type="Data dependent operator",
                context=str(cause.func),
                explanation=f"Operator `{cause.func}` has a non-Tensor output "
                "whose value is dependent on the data of Tensor inputs.",
                hints=hints,
                from_exc=cause,
            )
        elif isinstance(
            cause, torch._subclasses.fake_tensor.DynamicOutputShapeException
        ):
            if not torch._dynamo.config.capture_dynamic_output_shape_ops:
                unimplemented(
                    gb_type="Dynamic shape operator",
                    context=str(cause.func),
                    explanation=f"Operator `{cause.func}`'s output shape depends on input Tensor data.",
                    hints=[
                        "Enable tracing of dynamic shape operators with "
                        "`torch._dynamo.config.capture_dynamic_output_shape_ops = True`",
                    ],
                    from_exc=cause,
                )
            else:
                unimplemented(
                    gb_type="Dynamic shape operator (no meta kernel)",
                    context=str(cause.func),
                    explanation=f"Operator `{cause.func}` does not have a meta kernel that supports dynamic output shapes",
                    hints=[
                        "Please report an issue to PyTorch",
                    ],
                    from_exc=cause,
                )
        elif isinstance(
            cause, torch._subclasses.fake_tensor.UnsupportedOperatorException
        ):
            op = cause.func  # type: ignore[assignment]
            import_suggestion = ""
            if isinstance(op, torch._ops.OpOverload):
                maybe_pystub = torch._C._dispatch_pystub(
                    op._schema.name, op._schema.overload_name
                )
                if maybe_pystub is not None:
                    module, ctx = maybe_pystub
                    import_suggestion = (
                        f"It's possible that the support was implemented in "
                        f"module `{module}` and you may need to `import {module}`"
                        f"({ctx}), otherwise "
                    )
            unimplemented(
                gb_type="Operator does not support running with fake tensors",
                context=f"unsupported operator: {cause.func}",
                explanation="",
                hints=[
                    f"{import_suggestion}see "
                    "https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0"
                    " for how to fix",
                ],
                from_exc=cause,
            )
        elif isinstance(
            cause, torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        ):
            raise UserError(
                UserErrorType.CONSTRAINT_VIOLATION,
                str(cause),
                case_name="constrain_as_size_example",
            ) from cause
        elif isinstance(
            cause, torch._subclasses.fake_tensor.FakeTensorDeviceMismatchError
        ):
            _wrap_graph_break_with_torch_runtime_err(
                lambda: unimplemented(
                    gb_type="Tensor device mismatch",
                    context=f"{op} {node.target}",
                    explanation=str(cause),
                    hints=["Move inputs, parameters, and buffers to the same device."],
                    from_exc=cause,
                )
            )
            raise AssertionError("should not be reachable") from None
        elif isinstance(cause, ValueRangeError):
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, e.args[0]) from e
        elif isinstance(cause, TypeError) and "argument" in str(cause):
            unimplemented(
                gb_type="TypeError when making fake tensor call",
                context=f"TypeError {node.target}: {cause}",
                explanation="",
                hints=[*graph_break_hints.USER_ERROR],
                from_exc=cause,
            )
        elif "no fake impl registered" in str(cause) and _custom_op_fake_impl_pending(
            tx, node.target
        ):
            # A custom op was defined and register_fake'd inside the compiled
            # function, then called. Under nested graph breaks Dynamo traces
            # into the inlined op definition, so register_fake's mutation of
            # _abstract_fn is recorded as a pending side effect rather than run
            # eagerly; it is not yet applied to the real op when the op is called
            # here, so its fake kernel raises "no fake impl registered". (Without
            # nested graph breaks the definition graph-breaks and the op is fully
            # registered before the call, so this path is not hit.) Graph break:
            # the definition (and register_fake) run eagerly and the op call
            # itself falls back to eager -- it is not captured into the resumed
            # graph. Unlike the generic RuntimeError below this is recoverable,
            # so do NOT wrap it as a hard TorchRuntimeError.
            # NB: gated on a pending _abstract_fn mutation so a genuinely
            # unregistered op still hard-errors below with actionable guidance.
            unimplemented(
                gb_type="Custom op missing fake impl during tracing",
                context=f"{node.target}",
                explanation="A custom operator was defined and register_fake'd "
                "inside the compiled function, then called before the "
                "register_fake side effect was applied.",
                hints=[*graph_break_hints.SUPPORTABLE],
                from_exc=cause,
            )
        msg = get_concrete_sizes_from_symints(str(e), fake_mode)
        from .exc import (
            FakeTensorObservedException,
            ObservedException,
            raise_observed_exception,
        )

        if not node.users:
            tx.output.graph.erase_node(node)
        try:
            raise_observed_exception(RuntimeError, tx, args=[msg])
        except ObservedException as e:
            raise FakeTensorObservedException(msg, real_stack=e.real_stack) from None

    if not allow_non_graph_fake:
        _ = pytree.tree_map_only(
            torch.Tensor,
            functools.partial(ensure_graph_fake, tx=tx),
            ret_val,  # type: ignore[unbound-name]
        )

    if (
        torch._dynamo.config.use_graph_deduplication
        or torch._dynamo.config.track_nodes_for_deduplication
    ):
        tx.output.region_tracker.track_node_mutations(
            node,
            flat_args_kwargs,
            id_to_initial_version,
        )

    return ret_val


_current_node = threading.local()


def get_current_node() -> torch.fx.Node | None:
    # [FUNCTION DOC] Returns the FX node currently being run on this thread (set by
    # set_current_node), or None.
    return getattr(_current_node, "value", None)


@contextmanager
def set_current_node(node: torch.fx.Node) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that records node as the thread-local current node for the
    # duration of the body and restores the previous value on exit (even on exceptions).
    old = get_current_node()
    _current_node.value = node
    try:
        yield
    finally:
        _current_node.value = old


def run_node(
    tracer: Any, node: torch.fx.Node, args: Any, kwargs: Any, nnmodule: Any
) -> Any:
    """
    Runs a given node, with the given args and kwargs.

    Behavior is dictated by a node's op.

    run_node is useful for extracting real values out of nodes.
    See get_real_value for more info on common usage.

    Note: The tracer arg is only used for 'get_attr' ops
    Note: The nnmodule arg is only used for 'call_module' ops

    Nodes that are not call_function, call_method, call_module, or get_attr will
    raise an AssertionError.
    """
    # [FUNCTION DOC] Executes one FX node with concrete (real or fake) args: call_function calls
    # the target, call_method calls the named method on the first arg (graph break if the
    # attribute is missing), call_module calls nnmodule, get_attr fetches the submodule from the
    # tracer's output graph, placeholder returns its example_value. NotImplementedError and
    # unsupported fake tensor errors become graph breaks; Unsupported and IndexError propagate;
    # other exceptions are rewrapped as RuntimeError with context. Other ops raise AssertionError.
    op = node.op

    with set_current_node(node):

        def make_error_message(e: Any) -> str:
            # [FUNCTION DOC] HELPER: Builds the failure message naming the node op, target, args
            # and the underlying error.
            return (
                f"Dynamo failed to run FX node with fake tensors: {op} {node.target}(*{args}, **{kwargs}): got "
                + repr(e)
            )

        from .exc import Unsupported

        try:
            if op == "call_function":
                return node.target(*args, **kwargs)  # type: ignore[operator]
            elif op == "call_method":
                if not hasattr(args[0], node.target):  # type: ignore[arg-type]
                    from . import graph_break_hints
                    from .exc import unimplemented

                    unimplemented(
                        gb_type="Missing attribute when running call_method node",
                        context="",
                        explanation=make_error_message("attribute not defined"),
                        hints=[*graph_break_hints.USER_ERROR],
                    )
                return getattr(args[0], node.target)(*args[1:], **kwargs)  # type: ignore[arg-type]
            elif op == "call_module":
                if nnmodule is None:
                    raise AssertionError("nnmodule must not be None for call_module op")
                return nnmodule(*args, **kwargs)
            elif op == "get_attr":
                return tracer.output_graph.get_submodule(node.target)
            elif op == "placeholder":
                if "example_value" not in node.meta:
                    raise AssertionError(
                        "Expected 'example_value' in node.meta for placeholder op"
                    )
                return node.meta["example_value"]

        except (NotImplementedError, UnsupportedFakeTensorException) as e:
            # NB: mimic how wrap_fake_exception does it
            from . import graph_break_hints
            from .exc import unimplemented

            hints = [*graph_break_hints.USER_ERROR]
            if isinstance(e, NotImplementedError):
                hints += [
                    "If the op is a custom op, did you implement a fake tensor implementation? "
                    "(e.g. with `@my_custom_op.register_fake`)",
                    "If the op is a PyTorch op, please file an issue to PyTorch.",
                ]

            unimplemented(
                gb_type="NotImplementedError/UnsupportedFakeTensorException when running FX node",
                context="",
                explanation=make_error_message(e),
                hints=hints,
                from_exc=e,
            )
        except Unsupported:
            raise
        except IndexError:
            # Re-raise IndexError from tensor dim validation (e.g. canonicalize_dim,
            # maybe_wrap_dim) so it reaches the user as IndexError, not RuntimeError.
            # This is intentionally broad: an internal Dynamo indexing bug would also
            # propagate this way, but such bugs are rare and the traceback is preserved.
            raise
        except Exception as e:
            raise RuntimeError(make_error_message(e)).with_traceback(
                e.__traceback__
            ) from e

    raise AssertionError(op)


def get_real_value(node: torch.fx.Node, tracer: Any) -> Any:
    """
    Run the actual computation represented by `node` and return the result.
    This will execute any dependent nodes in the graph as well.
    """
    # [FUNCTION DOC] Computes (and caches in tracer.real_value_cache) the real value of node by
    # recursively computing its argument nodes, deep-copying nn modules for call_module (or running
    # lazy modules to initialize them), then running the node via run_node. Placeholders with a
    # graph arg return its example. A RuntimeError is converted into a graph break wrapped as a
    # TorchRuntimeError.
    from . import graph_break_hints
    from .exc import unimplemented

    cache = tracer.real_value_cache
    if node in cache:
        return cache[node]

    op = node.op
    args, kwargs = torch.fx.node.map_arg(  # type: ignore[misc]
        (node.args, node.kwargs),
        lambda n: get_real_value(n, tracer),
    )

    if op == "placeholder" and "grapharg" in node.meta:
        return node.meta["grapharg"].example

    if op == "call_module":
        nn_module = tracer.output_graph.nn_modules[node.target]
        if not is_lazy_module(nn_module):
            nn_module = copy.deepcopy(nn_module)
        else:
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it
            nn_module(*args, **kwargs)
    else:
        nn_module = None

    try:
        real_value = run_node(tracer, node, args, kwargs, nn_module)
        cache[node] = real_value
    except RuntimeError as e:
        exn = e  # to make typing happy for the lambda
        _wrap_graph_break_with_torch_runtime_err(
            lambda: unimplemented(
                gb_type="RuntimeError when trying to get real value from fx.Node",
                context="",
                explanation="",
                hints=[*graph_break_hints.USER_ERROR],
                from_exc=exn,
            )
        )
        raise AssertionError("should not be reachable") from None
    return real_value


def assert_no_fake_params_or_buffers(gm: torch.nn.Module) -> None:
    # [FUNCTION DOC] Raises AssertionError naming the first buffer or parameter of gm that is a
    # fake tensor (with creation traceback or a hint to enable debug mode); returns None if none.
    from torch._subclasses.fake_tensor import FakeTensorConfig, is_fake

    def stack_or_hint(t: Any) -> str:
        # [FUNCTION DOC] HELPER: Returns the fake tensor creation traceback if fake tensor debug is
        # enabled, else a hint on how to enable it.
        if FakeTensorConfig.debug:
            import traceback

            return f"FAKE TENSOR CREATION TRACEBACK: \n {traceback.format_list(t._debug_trace)}"
        else:
            return "Enable TORCH_FAKE_TENSOR_DEBUG=1 to get creation stack traces on fake tensors."

    for name, buffer in gm.named_buffers():
        if is_fake(buffer):
            raise AssertionError(
                f"Unexpected fake buffer {name} {stack_or_hint(buffer)}"
            )
    for name, param in gm.named_parameters():
        if is_fake(param):
            raise AssertionError(f"Unexpected fake param {name} {stack_or_hint(param)}")


def fqn(obj: Any) -> str:
    # [FUNCTION DOC] Returns the fully qualified name of obj as its module and qualified name
    # joined by a dot.
    """
    Returns the fully qualified name of the object.
    """
    return f"{obj.__module__}.{obj.__qualname__}"


def ifdynstaticdefault(count1: Any, count2: Any) -> Any:
    # [FUNCTION DOC] Returns count1 if config.assume_static_by_default is on, else count2.
    if torch._dynamo.config.assume_static_by_default:
        return count1
    else:
        return count2


def import_submodule(mod: types.ModuleType) -> None:
    """
    Ensure all the files in a given submodule are imported
    """
    # [FUNCTION DOC] Imports every non-underscore-prefixed .py file in the directory of the given
    # module (sorted by file name) as a submodule of it. Side effect: executes those modules.
    for filename in sorted(os.listdir(os.path.dirname(cast(str, mod.__file__)))):
        if filename.endswith(".py") and filename[0] != "_":
            importlib.import_module(f"{mod.__name__}.{filename[:-3]}")


def object_has_getattribute(value: object) -> bool:
    # [FUNCTION DOC] True if the type of value defines a Python-level attribute-lookup override.
    return class_has_getattribute(type(value))


def object_setattr_ignore_descriptor(obj: Any, name: str, value: Any) -> None:
    # [FUNCTION DOC] Sets name to value directly in obj's instance dictionary, bypassing
    # descriptors and custom setattr (like CPython's generic path). For thread-local objects uses
    # the per-thread dictionary. Re-raises AttributeError for objects without a dictionary.
    # Mirror the instance-dict update path in _PyObject_GenericSetAttrWithDict:
    # https://github.com/python/cpython/blob/v3.13.0/Objects/object.c#L1679-L1741
    try:
        d = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        if not isinstance(obj, threading.local):
            raise
        # threading.local stores attributes in a per-thread C-backed dict that
        # object.__getattribute__ cannot see. CPython passes that dict directly
        # to _PyObject_GenericSetAttrWithDict:
        # https://github.com/python/cpython/blob/v3.13.0/Modules/_threadmodule.c#L1707-L1730
        d = threading.local.__getattribute__(obj, "__dict__")
    d[name] = value


def object_delattr_ignore_descriptor(obj: Any, name: str) -> None:
    # [FUNCTION DOC] Deletes name from obj's instance dictionary, bypassing descriptors and custom
    # delattr (thread-local objects use the per-thread dictionary). Raises KeyError if absent.
    # Same path as object_setattr_ignore_descriptor with a NULL value.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/object.c#L1679-L1741
    try:
        d = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        if not isinstance(obj, threading.local):
            raise
        d = threading.local.__getattribute__(obj, "__dict__")
    del d[name]


def class_has_getattribute(cls: type) -> bool:
    # [FUNCTION DOC] True if the statically looked-up attribute-lookup method on cls is a plain
    # Python function (user override); False otherwise or if missing.
    try:
        if isinstance(
            inspect.getattr_static(cls, "__getattribute__"),
            types.FunctionType,
        ):
            return True
    except AttributeError:
        pass
    return False


def get_custom_getattr(
    value: Any, ignore_nn_module_getattr: bool = False
) -> Any | None:
    # [FUNCTION DOC] Returns the class-level fallback attribute hook of type(value) found
    # statically, or None if absent; the default nn.Module hook is treated as absent when
    # ignore_nn_module_getattr is set.
    try:
        getattr_fn = inspect.getattr_static(type(value), "__getattr__")
    except AttributeError:
        getattr_fn = None
    if ignore_nn_module_getattr and getattr_fn is torch.nn.Module.__getattr__:
        # ignore this case of getattr
        getattr_fn = None
    return getattr_fn


class TensorStaticReason(enum.Enum):
    # [CLASS DOC] Enum of reasons a tensor is forced to have static shapes: PARAMETER, NOT_TENSOR,
    # NN_MODULE_PROPERTY.
    PARAMETER = 2
    NOT_TENSOR = 4
    NN_MODULE_PROPERTY = 5


def tensor_static_reason_to_message(reason: TensorStaticReason) -> str:
    # [FUNCTION DOC] Returns a human-readable explanation for a TensorStaticReason; raises
    # AssertionError for an unknown reason.
    if reason == TensorStaticReason.PARAMETER:
        return "mark_dynamic on parameter, parameters are always static today."
    if reason == TensorStaticReason.NOT_TENSOR:
        return "mark_dynamic on a non tensor, how did this happen?"
    if reason == TensorStaticReason.NN_MODULE_PROPERTY:
        return "tensor is static because it is nn module associated."
    raise AssertionError(f"Illegal reason {reason}")


def tensor_always_has_static_shape(
    tensor: torch.Tensor | Any,
    is_tensor: bool,
    tensor_source: Source,
) -> tuple[bool, TensorStaticReason | None]:
    """
    Given a tensor, source, and is_tensor flag, determine if a shape should be static.

    Args:
    tensor - the real tensor to evaluate, parameters force a static shape.
    is_tensor - internal dynamo check, essentially "is_tensor": target_cls is TensorVariable,
    tensors not in a TensorVariable for whatever reason are forced static.

    Returns a tuple, where the first element is the bool of whether or not this tensor should have a static shape.
    The second element is a TensorStaticReason, useful for passing to tensor_static_reason_to_message if needed.
    """
    # [FUNCTION DOC] Returns (is_static, reason). Static with NN_MODULE_PROPERTY for tensors from
    # nn module sources when configured; static with PARAMETER for parameters or parameter/buffer
    # sources when configured; static with NOT_TENSOR if is_tensor is false; else (False, None).
    from .source import is_from_unspecialized_param_buffer_source

    if (
        tensor_source.guard_source.is_specialized_nn_module()
        or tensor_source.guard_source.is_unspecialized_builtin_nn_module()
    ) and config.force_nn_module_property_static_shapes:
        return True, TensorStaticReason.NN_MODULE_PROPERTY

    if (
        type(tensor) is torch.nn.Parameter
        or is_from_unspecialized_param_buffer_source(tensor_source)
    ) and config.force_parameter_static_shapes:
        return True, TensorStaticReason.PARAMETER
    if not is_tensor:
        return True, TensorStaticReason.NOT_TENSOR
    return False, None


def lazy_format_graph_tabular(fn_name: str, gm: torch.fx.GraphModule) -> Any:
    # [FUNCTION DOC] Returns a lazy string that, when rendered, formats gm's graph nodes as a
    # table (opcode, name, target, args, kwargs) with the tabulate package, or falls back to the
    # normal graph code format if tabulate is missing. Nothing is computed until rendered.
    def inner() -> str:
        # [FUNCTION DOC] HELPER: Performs the deferred table formatting.
        try:
            from tabulate import tabulate  # TODO: Check that this is installed
        except ImportError:
            return (
                "Tabulate module missing, please install tabulate to log the graph in tabular format, logging code instead:\n"
                + str(lazy_format_graph_code(fn_name, gm))
            )

        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes
        ]
        graph_str = tabulate(
            node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
        )
        return _format_graph_code(fn_name, gm.forward.__code__.co_filename, graph_str)

    return LazyString(inner)


def format_bytecode(
    prefix: str, name: str, filename: str, line_no: int, code: Any
) -> str:
    # [FUNCTION DOC] Returns a header line (prefix, name, file, line) followed by the
    # disassembly of code.
    return f"{prefix} {name} {filename} line {line_no} \n{dis.Bytecode(code).dis()}\n"


forward_hook_names = [
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_with_kwargs",
    "_forward_hooks",
]
backward_hook_names = ["_backward_pre_hooks", "_backward_hooks"]
state_dict_hook_names = [
    "_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]
all_hook_names = forward_hook_names + backward_hook_names + state_dict_hook_names


def nn_module_has_global_hooks() -> bool:
    # [FUNCTION DOC] True if any global module backward hook or backward pre-hook is registered
    # (forward hooks are excluded because module variable tracking handles them).
    # This is limited to backward hooks for now because NNModuleVariable
    # supports fwd hooks underneath.
    return bool(
        len(torch.nn.modules.module._global_backward_hooks)
        or len(torch.nn.modules.module._global_backward_pre_hooks)
    )


def nn_module_get_all_hooks(
    mod: torch.nn.Module,
    check_forward_hooks: bool = False,
    check_backward_hooks: bool = False,
    check_state_dict_hooks: bool = False,
) -> list[Any]:
    """
    Sometimes its useful to differentiate between types of hooks such as forward/backward/pre
    hooks executed during module.__call__, and state_dict hooks which are executed separately.
    """
    # [FUNCTION DOC] Returns a list of all hook callables registered on mod from the selected
    # categories: forward, backward, and/or state-dict hooks. If no flag is set, forward and
    # backward hooks are included (state-dict hooks only when explicitly requested).
    hook_dicts_to_check = []
    check_all_hooks = (
        not check_forward_hooks
        and not check_backward_hooks
        and not check_state_dict_hooks
    )
    if check_forward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(forward_hook_names)
    if check_backward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(backward_hook_names)
    if check_state_dict_hooks:
        hook_dicts_to_check.extend(state_dict_hook_names)

    all_hooks = []
    for hook_dict_name in hook_dicts_to_check:
        hooks = getattr(mod, hook_dict_name, [])
        for hook_name in hooks:
            hook = hooks[hook_name]

            all_hooks.append(hook)
    return all_hooks


def nnmodule_has_hooks(
    mod: torch.nn.Module,
    check_forward_hooks: bool = False,
    check_backward_hooks: bool = False,
    check_state_dict_hooks: bool = False,
) -> bool:
    """
    Helper function to check if a module has any hooks attached to it.
    """
    # [FUNCTION DOC] True if nn_module_get_all_hooks finds any hook for the selected categories.
    hooks = nn_module_get_all_hooks(
        mod,
        check_forward_hooks=check_forward_hooks,
        check_backward_hooks=check_backward_hooks,
        check_state_dict_hooks=check_state_dict_hooks,
    )
    return bool(hooks)


def to_numpy_helper(value: Any) -> Any:
    """Convert tensor and tnp.ndarray to numpy.ndarray."""
    # [FUNCTION DOC] Recursively converts tnp arrays and tensors (forced to CPU, detached) to numpy
    # arrays, preserving list/tuple containers; fake tensors and other values pass through.
    if is_fake(value):
        return value
    if isinstance(value, tnp.ndarray):
        return to_numpy_helper(value.tensor)
    elif isinstance(value, torch.Tensor):
        return value.numpy(force=True)
    elif isinstance(value, (tuple, list)):
        return type(value)(to_numpy_helper(obj) for obj in value)
    else:
        return value


def numpy_to_tensor(value: Any) -> Any:
    """Convert tnp.ndarray to tensor, leave other types intact. If a list/tuple, loop through it to convert."""
    # [FUNCTION DOC] Recursively converts numpy arrays (via as_tensor) and tnp arrays (underlying
    # tensor) to tensors, preserving list/tuple containers; other values are returned unchanged.
    # Raises AssertionError if numpy is unavailable.
    if np is None:
        raise AssertionError("numpy must be available (np is None)")
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    if isinstance(value, tnp.ndarray):
        return value.tensor
    elif isinstance(value, (tuple, list)):
        return type(value)(numpy_to_tensor(obj) for obj in value)
    else:
        return value


class numpy_to_tensor_wrapper(Generic[_P, R]):
    # [CLASS DOC] Callable wrapper around a function f that converts f's numpy-like results to
    # tensors with numpy_to_tensor. State: f and a derived __name__ ("wrapped_" plus f's name).
    def __init__(self, f: Callable[_P, R]) -> None:
        # [FUNCTION DOC] Stores f and sets the wrapper's __name__.
        self.f = f
        self.__name__ = "wrapped_" + self.f.__name__

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a repr naming the wrapped function.
        return f"<Wrapped function <original {self.f.__name__}>>"

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> Any:
        # [FUNCTION DOC] Calls f with the given arguments and converts the result to tensors.
        out = self.f(*args, **kwargs)
        return numpy_to_tensor(out)


def numpy_attr_wrapper(obj: Any, name: str) -> Any:
    # [FUNCTION DOC] Reads attribute name from a tnp array (or a tensor viewed as one) and
    # converts the result to tensors. Returns None implicitly for other object types.
    if isinstance(obj, tnp.ndarray):
        out = getattr(obj, name)
        return numpy_to_tensor(out)
    elif isinstance(obj, torch.Tensor):
        out = getattr(tnp.ndarray(obj), name)
        return numpy_to_tensor(out)


class numpy_method_wrapper:
    # [CLASS DOC] Callable that invokes the named method on its first argument (a tensor is first
    # viewed as a tnp array) with the remaining arguments and converts the result back to tensors.
    # State: method name and derived __name__.
    """Convert obj from torch.Tensor to tnp.ndarray and call method. Then convert result back to torch.Tensor."""

    def __init__(self, method: str) -> None:
        # [FUNCTION DOC] Stores the method name and sets __name__.
        self.method = method
        self.__name__ = "wrapped_" + self.method

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a repr naming the wrapped method.
        return f"<Wrapped method <original {self.method}>>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Treats args[0] as the object (converting tensors to tnp arrays), calls
        # its method with the other arguments, and converts the result to tensors.
        obj = args[0]
        if isinstance(obj, torch.Tensor):
            obj = tnp.ndarray(obj)
        method_callable = getattr(obj, self.method)
        out = method_callable(*args[1:], **kwargs)
        return numpy_to_tensor(out)


class numpy_operator_wrapper(Generic[_P, R]):
    # [CLASS DOC] Callable wrapping an operator-library function so tensor arguments are viewed as
    # tnp arrays before applying it and the result is converted back to tensors. State: op and
    # derived __name__.
    """Implements dunder methods for tnp.ndarray via functions from the operator library"""

    def __init__(self, op: Callable[..., Any]) -> None:
        # [FUNCTION DOC] Stores the operator and sets __name__.
        self.op = op
        self.__name__ = f"wrapped_{op.__name__}"

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a repr naming the wrapped operator.
        return f"<Wrapped operator <original {self.__name__}>>"

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> Any:
        # [FUNCTION DOC] Applies the operator to the positional arguments (tensors converted to
        # tnp arrays) and converts the result to tensors. Raises AssertionError if keyword
        # arguments are passed.
        if kwargs:
            raise AssertionError(f"Expected no kwargs, got {kwargs}")

        # pyrefly: ignore [bad-assignment]
        args = (
            tnp.ndarray(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        out = self.op(*args)
        return numpy_to_tensor(out)


@functools.lru_cache(maxsize=1)
def _torch_numpy_callable_id_map() -> dict[int, tuple[Callable[..., Any], Any, str]]:
    # [FUNCTION DOC] HELPER: Builds (once, cached) a mapping from id of each torch._numpy callable
    # to (that callable, the matching numpy callable, dotted numpy path), for names present as
    # callables in both libraries across the supported module pairs.
    result: dict[int, tuple[Callable[..., Any], Any, str]] = {}
    for np_mod, tnp_mod in NP_TO_TNP_MODULE.items():
        module = np_mod.__name__
        for name, tnp_callable in tnp_mod.__dict__.items():
            if not callable(tnp_callable):
                continue

            np_callable = getattr(np_mod, name, None)
            if not callable(np_callable):
                continue

            result[id(tnp_callable)] = (
                tnp_callable,
                np_callable,
                f"{module}.{name}",
            )
    return result


@functools.lru_cache(maxsize=1024)
def _torch_numpy_callable_cache_key_by_id(
    obj_id: int,
) -> tuple[Callable[..., Any], str] | None:
    # [FUNCTION DOC] HELPER: For an id known to the id map, returns (torch._numpy callable, dotted
    # numpy path) if the numpy function is importable at that path and is not a random function;
    # otherwise None. Cached per id.
    tnp_callable, np_callable, cache_key = _torch_numpy_callable_id_map()[obj_id]

    module, _, name = cache_key.rpartition(".")
    # Random functions depend on global RNG state, so they are not safe to
    # identify only by their canonical NumPy path in an AOTAutograd cache key.
    if module == "numpy.random" or module.startswith("numpy.random."):
        return None

    try:
        if np_callable is not getattr(importlib.import_module(module), name):
            return None
    except (AttributeError, ImportError):
        return None

    return (tnp_callable, cache_key)


def _torch_numpy_callable_cache_key(obj: Callable[..., Any]) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the dotted numpy path used as a cache key for obj if obj is
    # exactly a recognized, non-random torch._numpy function; None otherwise. Misses are not
    # cached by id.
    obj_id = id(obj)
    # Arbitrary Python callables can be short-lived, so do not cache misses by
    # id; a later object may reuse the same id.
    if obj_id not in _torch_numpy_callable_id_map():
        return None

    entry = _torch_numpy_callable_cache_key_by_id(obj_id)
    if entry is None:
        return None

    tnp_callable, cache_key = entry
    if obj is not tnp_callable:
        return None

    return cache_key


def is_safe_numpy_wrapper(obj: object) -> bool:
    # [FUNCTION DOC] True if obj is a numpy_to_tensor_wrapper around a recognized deterministic
    # torch._numpy function (has a cache key).
    return numpy_wrapper_cache_key(obj) is not None


def numpy_wrapper_cache_key(obj: Any) -> tuple[str, str] | None:
    # [FUNCTION DOC] Returns (wrapper class qualified name, dotted numpy path) for an exact
    # numpy_to_tensor_wrapper whose wrapped function is recognized; None otherwise.
    if type(obj) is not numpy_to_tensor_wrapper:
        return None

    key = _torch_numpy_callable_cache_key(obj.f)
    if key is None:
        return None

    return (type(obj).__qualname__, key)


def defake(x: Any) -> Any:
    # [FUNCTION DOC] Converts a fake tensor into a real zero-filled tensor with the same strides,
    # dtype, device and requires_grad (symbolic sizes replaced by optimization hints). Non-fake
    # values are returned unchanged.
    from torch._subclasses.fake_tensor import is_fake_tensor

    if not is_fake_tensor(x):
        return x
    size: torch._prims_common.ShapeType
    stride: torch._prims_common.StrideType
    if x._has_symbolic_sizes_strides:
        # optimization_hint is appropriate here because defake only needs a
        # plausible concrete shape to allocate a real tensor; it does not need
        # to install guards. For unbacked symbols the heuristic fallback is fine.
        size = [
            torch.fx.experimental.symbolic_shapes.optimization_hint(s) for s in x.size()
        ]
        stride = [
            torch.fx.experimental.symbolic_shapes.optimization_hint(s)
            for s in x.stride()
        ]
    else:
        size = x.size()
        stride = x.stride()
    y = torch.empty_strided(
        size,
        stride,
        dtype=x.dtype,
        device=x.device,
        requires_grad=x.requires_grad,
    )
    y.zero_()
    return y


def _disable_side_effect_safety_checks_for_current_subtracer(
    fn: Callable[_P, R], *args: _P.args, **kwargs: _P.kwargs
) -> R:
    # [FUNCTION DOC] HELPER: Plain pass-through that calls fn with the arguments; exists as a named
    # marker function that tracing recognizes to disable side-effect safety checks.
    return fn(*args, **kwargs)


def is_utils_checkpoint(obj: object) -> bool:
    # [FUNCTION DOC] True if obj is the torch.utils.checkpoint.checkpoint function.
    # Lazy import to avoid circular dependencies
    import torch.utils.checkpoint

    return obj is torch.utils.checkpoint.checkpoint


def is_invoke_subgraph(obj: object) -> bool:
    # [FUNCTION DOC] True if obj is the invoke_subgraph placeholder object.
    from torch._higher_order_ops.invoke_subgraph import invoke_subgraph_placeholder

    return obj is invoke_subgraph_placeholder


def build_invoke_subgraph_variable(**options: Any) -> Any:
    # [FUNCTION DOC] Builds a higher-order-operator variable for invoke_subgraph, forwarding the
    # options.
    from .variables.higher_order_ops import TorchHigherOrderOperatorVariable

    return TorchHigherOrderOperatorVariable.make(
        torch._higher_order_ops.invoke_subgraph,
        **options,
    )


def build_checkpoint_variable(**options: Any) -> Any:
    # [FUNCTION DOC] Builds a higher-order-operator variable for activation checkpointing,
    # choosing the RNG-functionalizing wrapper op when functorch's functionalize_rng_ops is on and
    # the tag-based op otherwise; forwards the options.
    import torch._higher_order_ops.wrap as higher_order_ops

    from .variables.higher_order_ops import TorchHigherOrderOperatorVariable

    # TODO - This is a temporary situation where we have two versions of
    # checkpointing implementation. We will converge on one and remove the other.
    activation_checkpoint_op: torch._ops.HigherOrderOperator = (
        higher_order_ops.tag_activation_checkpoint
    )
    if torch._functorch.config.functionalize_rng_ops:
        activation_checkpoint_op = higher_order_ops.wrap_activation_checkpoint

    return TorchHigherOrderOperatorVariable.make(
        activation_checkpoint_op,
        **options,
    )


def is_compile_supported(device_type: DeviceLikeType) -> Any:
    # [FUNCTION DOC] Returns whether compilation is supported for the device type: CPU needs
    # only dynamo support; cuda, xpu and mtia additionally need triton; other devices are
    # unsupported.
    from .eval_frame import is_dynamo_supported

    type = torch.device(device_type).type
    compile_supported = is_dynamo_supported()
    if type == "cpu":
        pass
    elif type in ["cuda", "xpu", "mtia"] and compile_supported:
        compile_supported = has_triton()
    else:
        compile_supported = False
    return compile_supported


# The following 3.11 source code functions are adapted from
# https://github.com/python/cpython/blob/v3.11.4/Lib/traceback.py
# in order to output source code corresponding to bytecode in 3.11+.
# We need our own versions since we want to support multiline expressions.
def _fix_offset(str: str, offset: int) -> int:
    # [FUNCTION DOC] HELPER: Converts a UTF-8 byte offset into a character offset for the string
    # (invalid bytes replaced). Adapted unchanged from CPython.
    """
    Convert byte offset `offset` of `str` into character offset.
    Byte offset is used for 3.11+ instruction column data.
    Takes things like unicode characters into consideration.

    Unchanged from CPython implementation.
    """
    as_utf8 = str.encode("utf-8")
    return len(as_utf8[:offset].decode("utf-8", errors="replace"))


@dataclasses.dataclass
class _Anchors:
    # [CLASS DOC] HELPER: Dataclass of line/column positions of an operator or bracket pair within
    # a source segment (left end inclusive, right start exclusive column).
    # inclusive
    left_end_lineno: int
    left_end_offset: int
    right_start_lineno: int
    # exclusive
    right_start_offset: int


def _extract_anchors_from_expr(segment: str) -> _Anchors | None:
    """
    Given source code `segment` corresponding to a bytecode
    instruction, determine:
        - for binary ops, the location of the binary op
        - for indexing, the location of the brackets.
    `segment` is expected to be a valid Python expression
    """
    # [FUNCTION DOC] HELPER: Parses segment (wrapped in brackets for multi-line support) and
    # returns the anchor location of a binary operator, subscript brackets or call parentheses
    # when the expression is such a node; None for unparseable or other expressions. Raises
    # AssertionError on Python older than 3.11 or inconsistent positions.
    if sys.version_info < (3, 11):
        raise AssertionError(f"Python >= 3.11 required, got {sys.version_info}")

    import ast

    tree: Any | None = None
    try:
        # Without brackets, `segment` is parsed as a statement.
        # We expect an expression, so wrap `segment` in
        # brackets to handle multi-line expressions.
        tree = ast.parse("(\n" + segment + "\n)")
    except SyntaxError:
        return None
    if tree is None:
        raise AssertionError("ast.parse returned None unexpectedly")

    if len(tree.body) != 1:
        return None

    lines = segment.split("\n")

    # get character index given byte offset
    def normalize(lineno: int, offset: int) -> int:
        # [FUNCTION DOC] HELPER: Converts a byte offset on the given segment line to a character
        # column.
        return _fix_offset(lines[lineno], offset)

    # Gets the next valid character index in `lines`, if
    # the current location is not valid. Handles empty lines.
    def next_valid_char(lineno: int, col: int) -> tuple[int, int]:
        # [FUNCTION DOC] HELPER: Skips past line ends (and empty lines) to the nearest valid
        # (line, column) position at or after the given one; AssertionError if none exists.
        while lineno < len(lines) and col >= len(lines[lineno]):
            col = 0
            lineno += 1
        if lineno >= len(lines):
            raise AssertionError(
                f"lineno {lineno} out of range (len(lines)={len(lines)})"
            )
        if col >= len(lines[lineno]):
            raise AssertionError(
                f"col {col} out of range (len(lines[{lineno}])={len(lines[lineno])})"
            )
        return lineno, col

    # Get the next valid character index in `lines`.
    def increment(lineno: int, col: int) -> tuple[int, int]:
        # [FUNCTION DOC] HELPER: Moves one character forward to the next valid position, crossing
        # lines if needed; AssertionError if out of range.
        col += 1
        lineno, col = next_valid_char(lineno, col)
        if lineno >= len(lines):
            raise AssertionError(
                f"lineno {lineno} out of range (len(lines)={len(lines)})"
            )
        if col >= len(lines[lineno]):
            raise AssertionError(
                f"col {col} out of range (len(lines[{lineno}])={len(lines[lineno])})"
            )
        return lineno, col

    # Get the next valid character at least on the next line
    def nextline(lineno: int, col: int) -> tuple[int, int]:
        # [FUNCTION DOC] HELPER: Moves to the first valid character on a following line;
        # AssertionError if out of range.
        col = 0
        lineno += 1
        lineno, col = next_valid_char(lineno, col)
        if lineno >= len(lines):
            raise AssertionError(
                f"lineno {lineno} out of range (len(lines)={len(lines)})"
            )
        if col >= len(lines[lineno]):
            raise AssertionError(
                f"col {col} out of range (len(lines[{lineno}])={len(lines[lineno])})"
            )
        return lineno, col

    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        expr = statement.value
        if isinstance(expr, ast.BinOp):
            # ast gives locations for BinOp subexpressions, e.g.
            # ( left_expr ) + ( right_expr )
            #   left^^^^^       right^^^^^
            # -2 since end_lineno is 1-indexed and because we added an extra
            # bracket to `segment` when calling ast.parse
            cur_lineno = cast(int, expr.left.end_lineno) - 2
            if expr.left.end_col_offset is None:
                raise AssertionError("expr.left.end_col_offset must not be None")
            cur_col = normalize(cur_lineno, expr.left.end_col_offset)
            cur_lineno, cur_col = next_valid_char(cur_lineno, cur_col)

            # Heuristic to find the operator character.
            # The original CPython implementation did not look for ), \, or #,
            # leading to incorrect anchor location, e.g.
            # (x) + (y)
            # ~~^~~~~~~
            while (ch := lines[cur_lineno][cur_col]).isspace() or ch in ")\\#":
                if ch in "\\#":
                    cur_lineno, cur_col = nextline(cur_lineno, cur_col)
                else:
                    cur_lineno, cur_col = increment(cur_lineno, cur_col)

            # binary op is 1 or 2 characters long, on the same line
            right_col = cur_col + 1
            if (
                right_col < len(lines[cur_lineno])
                and not (ch := lines[cur_lineno][right_col]).isspace()
                and ch not in "\\#"
            ):
                right_col += 1
            # right_col can be invalid since it is exclusive

            return _Anchors(cur_lineno, cur_col, cur_lineno, right_col)
        elif isinstance(expr, ast.Subscript):
            # ast gives locations for value and slice subexpressions, e.g.
            # ( value_expr ) [ slice_expr ]
            #   value^^^^^     slice^^^^^
            # subscript^^^^^^^^^^^^^^^^^^^^
            # find left bracket (first '[' after value)
            left_lineno = cast(int, expr.value.end_lineno) - 2
            if expr.value.end_col_offset is None:
                raise AssertionError("expr.value.end_col_offset must not be None")
            left_col = normalize(left_lineno, expr.value.end_col_offset)
            left_lineno, left_col = next_valid_char(left_lineno, left_col)
            while lines[left_lineno][left_col] != "[":
                left_lineno, left_col = increment(left_lineno, left_col)
            # find right bracket (final character of expression)
            right_lineno = cast(int, expr.end_lineno) - 2
            if expr.end_col_offset is None:
                raise AssertionError("expr.end_col_offset must not be None")
            right_col = normalize(right_lineno, expr.end_col_offset)
            return _Anchors(left_lineno, left_col, right_lineno, right_col)
        elif isinstance(expr, ast.Call):
            # ( func_expr ) (args, kwargs)
            #   func^^^^^
            # call^^^^^^^^^^^^^^^^^^^^^^^^
            # find left bracket (first '(' after func)
            left_lineno = cast(int, expr.func.end_lineno) - 2
            if expr.func.end_col_offset is None:
                raise AssertionError("expr.func.end_col_offset must not be None")
            left_col = normalize(left_lineno, expr.func.end_col_offset)
            left_lineno, left_col = next_valid_char(left_lineno, left_col)
            while lines[left_lineno][left_col] != "(":
                left_lineno, left_col = increment(left_lineno, left_col)
            # find right bracket (final character of expression)
            right_lineno = cast(int, expr.end_lineno) - 2
            if expr.end_col_offset is None:
                raise AssertionError("expr.end_col_offset must not be None")
            right_col = normalize(right_lineno, expr.end_col_offset)
            return _Anchors(left_lineno, left_col, right_lineno, right_col)

    return None


def format_source_range(
    filename: str,
    lineno: int | None,
    end_lineno: int | None = None,
    col_offset: int | None = None,
    end_col_offset: int | None = None,
    *,
    function_name: str = "<unknown>",
) -> str:
    # [FUNCTION DOC] Renders the source lines for a location range (from the line cache) with a
    # marker line underneath: tildes under the whole span and carets under the operator, brackets
    # or call parentheses when anchors can be found (otherwise the whole span uses carets). On
    # Python 3.13+ multi-line ranges use the stdlib traceback formatter. Returns an empty string
    # when lineno is None or no source is available; with incomplete column data returns only the
    # first source line.
    if lineno is None:
        return ""

    last_lineno = end_lineno if end_lineno is not None else lineno
    source_lines = [
        linecache.getline(filename, current_lineno).rstrip()
        for current_lineno in range(lineno, last_lineno + 1)
    ]
    if not any(source_lines):
        return ""

    if (
        sys.version_info >= (3, 13)
        and end_lineno is not None
        and end_lineno != lineno
        and col_offset is not None
        and end_col_offset is not None
    ):
        # Keep single-line ranges on Dynamo's manual path. The stdlib traceback
        # formatter is useful for multiline spans on 3.13+, but for single-line
        # spans it can emit version-specific mixed `~`/`^` markers or omit the
        # marker line entirely for comments.
        frame_summary = traceback.FrameSummary(
            filename,
            lineno,
            function_name,
            end_lineno=end_lineno,
            colno=col_offset,
            end_colno=end_col_offset,
        )
        result = traceback.format_list([frame_summary])[0]
        result = "\n".join(result.splitlines()[1:])
        source_lines_dedent = textwrap.dedent("\n".join(source_lines)).splitlines()
        if not source_lines_dedent:
            return result
        indent_len = len(source_lines[0]) - len(source_lines_dedent[0])
        indent = source_lines[0][:indent_len]
        return textwrap.indent(textwrap.dedent(result), indent)

    first_line = source_lines[0]
    if end_lineno is None or col_offset is None or end_col_offset is None:
        return first_line

    start_offset = _fix_offset(first_line, col_offset)
    segment = ""
    markers: list[str] = []

    if end_lineno == lineno:
        end_offset = _fix_offset(first_line, end_col_offset)
        segment = first_line[start_offset:end_offset]
        markers.append(" " * start_offset + "~" * (end_offset - start_offset))
    else:
        segment = first_line[start_offset:] + "\n"
        markers.append(" " * start_offset + "~" * (len(first_line) - start_offset))
        last_line = source_lines[-1]
        end_offset = _fix_offset(last_line, end_col_offset)
        for line in source_lines[1:-1]:
            segment += line + "\n"
            num_spaces = len(line) - len(line.lstrip())
            markers.append(" " * num_spaces + "~" * (len(line) - num_spaces))
        segment += last_line[:end_offset]
        num_spaces = len(last_line) - len(last_line.lstrip())
        markers.append(" " * num_spaces + "~" * (end_offset - num_spaces))

    anchors: _Anchors | None = None
    try:
        anchors = _extract_anchors_from_expr(segment)
    except AssertionError:
        pass

    if anchors is None:
        markers = [marker.replace("~", "^") for marker in markers]
    else:
        mutable_markers: list[list[str]] = [list(marker) for marker in markers]

        if anchors.left_end_lineno == 0:
            anchors.left_end_offset += start_offset
        if anchors.right_start_lineno == 0:
            anchors.right_start_offset += start_offset

        for marker_lineno in range(len(markers)):
            for col in range(len(mutable_markers[marker_lineno])):
                if marker_lineno < anchors.left_end_lineno:
                    continue
                if (
                    marker_lineno == anchors.left_end_lineno
                    and col < anchors.left_end_offset
                ):
                    continue
                if (
                    marker_lineno == anchors.right_start_lineno
                    and col >= anchors.right_start_offset
                ):
                    continue
                if marker_lineno > anchors.right_start_lineno:
                    continue
                if mutable_markers[marker_lineno][col] == "~":
                    mutable_markers[marker_lineno][col] = "^"

        markers = ["".join(marker) for marker in mutable_markers]

    result = ""
    for line, marker in zip(source_lines, markers):
        result += line + "\n"
        result += marker + "\n"
    return result


def get_instruction_source_311(code: types.CodeType, inst: Instruction) -> str:
    # [FUNCTION DOC] Returns the underlined source text for an instruction using its position data
    # (via format_source_range) with the code object's file and name. Raises AssertionError if the
    # instruction lacks position data.
    """
    Python 3.11+ only. Returns lines of source code (from code object `code`)
    corresponding to `inst`'s location data, and underlines relevant code to `inst`.

    Example: CALL on `g`:
    f(g(
      ^^
        h(x)))
        ^^^^^

    We need our own implementation in < 3.13 since `format_frame_summary` in
    Python's `traceback` module doesn't handle multi-line expressions
    (and their anchor extraction code is not completely correct).
    """
    if not hasattr(inst, "positions"):
        raise AssertionError("inst must have a 'positions' attribute")
    if inst.positions is None:
        raise AssertionError("inst.positions must not be None")
    return format_source_range(
        code.co_filename,
        inst.positions.lineno,
        inst.positions.end_lineno,
        inst.positions.col_offset,
        inst.positions.end_col_offset,
        function_name=code.co_name,
    )


# The two flavors of mark_static_address: "guarded" recompiles when the data_ptr
# changes, "unguarded" re-records cudagraphs instead. Set in decorators.py.
StaticInputType = Literal["guarded", "unguarded"]


def get_static_address_type(t: Any) -> StaticInputType | None:
    # [FUNCTION DOC] Returns the static-input type marker ("guarded"/"unguarded") stored on a
    # tensor, or None for non-tensors and unmarked tensors.
    if isinstance(t, torch.Tensor):
        return getattr(t, "_dynamo_static_input_type", None)

    return None


def is_rng_state_getter_or_setter(value: object) -> bool:
    # [FUNCTION DOC] True if value is one of the known RNG state get/set functions (generator,
    # default generator, global and CUDA variants).
    getters = (
        # The following two functions are not identical, so don't remove anyone!
        torch._C.Generator.get_state,
        torch.default_generator.get_state,
        torch.get_rng_state,
        torch.cuda.get_rng_state,
    )
    setters = (
        torch._C.Generator.set_state,
        torch.default_generator.set_state,
        torch.set_rng_state,
        torch.cuda.set_rng_state,
    )
    return value in (*setters, *getters)


def is_tensor_base_attr_getter(value: object) -> bool:
    # [FUNCTION DOC] True if value is a descriptor's bound get method whose owner class is the
    # C tensor base class (i.e. a tensor attribute getter).
    return (
        isinstance(value, types.MethodWrapperType)
        and value.__name__ == "__get__"
        and hasattr(value.__self__, "__objclass__")
        and value.__self__.__objclass__ is torch._C._TensorBase  # type: ignore[attr-defined]
    )


def is_tensor_getset_descriptor(name: str) -> bool:
    # [FUNCTION DOC] True if the attribute name on torch.Tensor (looked up statically) is exactly a
    # getset descriptor; False if missing or of another kind.
    try:
        attr = inspect.getattr_static(torch.Tensor, name)
        return type(attr) is types.GetSetDescriptorType
    except AttributeError:
        return False


def is_torch_class(cls: type) -> bool:
    """Check if cls is defined in torch or a torch submodule.

    Torch-internal C-level descriptors (e.g. torch.Tensor.unsqueeze_) need
    to go through VariableTracker.build / trace_rules so they get the right
    VT (TorchInGraphFunctionVariable), which has guards like
    inplace-view-on-input-tensor detection. This helper identifies classes
    whose descriptors should take that path instead of descriptor VTs.
    """
    # [FUNCTION DOC] True if the class's module is torch or a torch submodule.
    module = getattr(cls, "__module__", None)
    return module is not None and (module == "torch" or module.startswith("torch."))


def is_torch_function_object(value: object) -> bool:
    # [FUNCTION DOC] True if value has a torch function protocol attribute.
    return hasattr(value, "__torch_function__")


def has_torch_function(vt: VariableTracker) -> bool:
    # [FUNCTION DOC] Emulates the C++ torch-function-override check for a tracked value. Looks at
    # realized trackers (or unrealized lazy ones whose peeked value has the protocol, without
    # realizing others) and returns True if the tensor subclass or user object defines a
    # torch-function hook other than the disabled default implementation.
    # This emulates
    # https://github.com/pytorch/pytorch/blob/8d81806211bc3c0ee6c2ef235017bacf1d775a85/torch/csrc/utils/disable_torch_function.cpp#L315-L323
    from torch._dynamo.variables import UserDefinedObjectVariable
    from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable

    # Note on lazy vars: The value will either be realized or not throughout the course of execution
    # if the value has a torch function, it will eventually be realized so we can realize it here
    # if the value does not have a torch function, it may or may not be realized
    # if it is realized it will be used and guards will be installed properly
    # if it is not used, guards won't be installed, and it doesn't matter
    # if the value has a torch function or not, so we should *not* realize it.
    # NB: We technically know that if is_realized is False, LazyVariableTracker has the peek_value method
    # but mypy does not unfortunately
    if vt.is_realized() or (
        hasattr(vt, "peek_value") and hasattr(vt.peek_value(), "__torch_function__")
    ):
        func = None
        if isinstance(vt, TensorWithTFOverrideVariable):
            func = getattr(vt.class_type, "__torch_function__", None)

        elif isinstance(vt, UserDefinedObjectVariable):
            func = getattr(vt.value, "__torch_function__", None)

        return func not in (None, torch._C._disabled_torch_function_impl)

    return False


# see note [Tensor Fakification and Symbol Caching]
def to_fake_tensor(
    t: torch.Tensor, fake_mode: torch._subclasses.fake_tensor.FakeTensorMode
) -> Any:
    # [FUNCTION DOC] Converts a real tensor to a fake tensor under fake_mode with non-static
    # shapes, reusing the symbolic context and source recorded for t in the current tracing
    # context when available.
    symbolic_context = None
    source = None
    if tracing_context := torch._guards.TracingContext.try_get():
        if t in tracing_context.tensor_to_context:
            symbolic_context = tracing_context.tensor_to_context[t]
            source = symbolic_context.tensor_source

    return fake_mode.from_tensor(
        t, static_shapes=False, symbolic_context=symbolic_context, source=source
    )


# NB: this works for both classes and instances
def is_frozen_dataclass(value: object) -> bool:
    # [FUNCTION DOC] True if value (instance or class) is a frozen dataclass whose type does not
    # customize attribute lookup.
    # value may be either a dataclass instance or a dataclass type; both are
    # accepted by getattr_static, so the class_has_getattribute call is safe
    # even though it is annotated to take a type.
    return bool(
        not object_has_getattribute(value)
        and not class_has_getattribute(value)  # type: ignore[arg-type]
        and is_dataclass(value)
        and hasattr(value, "__dataclass_params__")
        and hasattr(value.__dataclass_params__, "frozen")
        and value.__dataclass_params__.frozen
    )


def get_first_attr(obj: Any, *attrs: str) -> Any:
    """
    Return the first available attribute or throw an exception if none is present.
    """
    # [FUNCTION DOC] Returns the value of the first attribute name in attrs that obj has; raises
    # AssertionError if it has none.
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


@contextlib.contextmanager
def maybe_enable_compiled_autograd(
    should_enable: bool, fullgraph: bool = True, dynamic: bool = True
) -> Generator[Any, None, None]:
    # [FUNCTION DOC] Context manager: if should_enable, enables compiled autograd with a compiler
    # that torch.compiles the autograd graph (fullgraph/dynamic as given) using inductor and
    # counts compiles; yields the enable context. Otherwise just yields.
    if not should_enable:
        yield
    else:

        def compiler_fn(gm: Any) -> Any:
            # [FUNCTION DOC] HELPER: Compiler hook returning a torch.compiled version of gm.
            def inner_compiler(gm_: Any, example_inputs_: Any) -> Any:
                # [FUNCTION DOC] HELPER: Backend that bumps the compiled_autograd compiles counter
                # and compiles with inductor.
                torch._dynamo.utils.counters["compiled_autograd"]["compiles"] += 1
                return torch._inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=fullgraph, dynamic=dynamic
            )

        with torch._dynamo.compiled_autograd._enable(compiler_fn) as ctx:
            yield ctx


def invalid_removeable_handle() -> RemovableHandle:
    # [FUNCTION DOC] Returns a RemovableHandle backed by an empty throwaway dict, so removing it is
    # a harmless no-op.
    # need a subclass so weakref works
    class Invalid(dict):  # type: ignore[type-arg]
        # [CLASS DOC] HELPER: Empty dict subclass (plain dicts cannot be weakly referenced).
        pass

    return RemovableHandle(Invalid())


# Returns a "proxy" (new object with the same class and dict) for (non-GraphModule) nn.Module's.
# Attribute changes to the original object/proxy will be reflected in the other.
# This is useful for cases where we want a keep-alive reference to a module without increasing
# its reference count.
def nn_module_proxy(mod: Any) -> Any:
    # [FUNCTION DOC] For a plain nn.Module returns a new instance of the same class sharing the
    # original's attribute dictionary (so mutations are visible both ways) without a strong
    # reference to the original module. Non-modules and GraphModules are returned as-is.
    if not isinstance(mod, torch.nn.Module):
        return mod
    if isinstance(mod, torch.fx.GraphModule):
        # Dynamo-generated GM's shouldn't contain user-created GM's
        return mod
    proxy = mod.__class__.__new__(mod.__class__)
    proxy.__dict__ = mod.__dict__
    return proxy


class GmWrapper(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping a graph module so it can be called with flat positional
    # inputs; an unflatten function rebuilds the structured arguments the graph expects.
    def __init__(
        self, gm: torch.fx.GraphModule, unflatten_fn: Callable[[list[Any]], Any]
    ) -> None:
        # [FUNCTION DOC] Stores the graph module and the unflatten function.
        super().__init__()
        self.gm = gm
        self.unflatten_fn = unflatten_fn

    def forward(self, *args: Any) -> Any:
        # [FUNCTION DOC] Unflattens the flat args into the graph's expected structure and calls
        # the wrapped graph module with them, returning its output.
        # pyrefly: ignore [redefinition]
        args: list[Any] = list(args)
        return self.gm(*self.unflatten_fn(args))


def flatten_graph_inputs(
    gm: torch.fx.GraphModule, inputs: Any, compile_gm: Callable[[Any, Any], Any]
) -> Callable[..., Any]:
    """
    Mutate inputs so that they are flat and wrap gm such that it
    accepts those inputs.  This is needed for graphs that take
    bumpy inputs.
    """
    # [FUNCTION DOC] Compiles gm (wrapped in GmWrapper) on flattened example inputs via compile_gm
    # and returns a wrapper callable that flattens call-time args, clears the original list
    # arguments for placeholders marked as stolen (so they release references), and calls the
    # compiled function with the flat list (boxed call). Inside a compiled autograd region uses a
    # fast path that assumes a single stolen list input; otherwise uses pytree flattening.
    # Raises AssertionError if the compiled autograd fast-path assumptions are violated.
    inputs_idx_to_clear = [
        i
        for i, node in enumerate(gm.graph.nodes)
        if node.op == "placeholder" and node.meta.get("steal_arg", False)
    ]

    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        # fast path, avoid pytree overhead
        # compiled autograd inputs are always a list of tensors, maybe followed by symints
        if inputs_idx_to_clear != [0]:
            raise AssertionError(
                f"Expected inputs_idx_to_clear == [0], got {inputs_idx_to_clear}"
            )
        if not isinstance(inputs[0], list):
            raise AssertionError(
                f"Expected inputs[0] to be a list, got {type(inputs[0])}"
            )
        boxed_inputs_count = len(inputs[0])

        def flatten_fn(args: Any) -> Any:
            # [FUNCTION DOC] HELPER: Concatenates the boxed list (first arg) with the remaining
            # args into one flat list.
            return args[0] + list(args[1:])

        def unflatten_fn(flat_args: Any) -> Any:
            # [FUNCTION DOC] HELPER: Splits a flat list back into (boxed list, remaining args).
            return (flat_args[:boxed_inputs_count], *flat_args[boxed_inputs_count:])

        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flatten_fn(inputs))
    else:
        # slow path, don't know inputs structure
        flat_inputs, spec = pytree.tree_flatten(inputs)
        unflatten_fn = functools.partial(pytree.tree_unflatten, treespec=spec)
        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flat_inputs)
        # note this doesn't check the spec, assuming it is the same
        flatten_fn = pytree.arg_tree_leaves

    def wrapper(*args: Any) -> Any:
        # [FUNCTION DOC] HELPER: Runtime entry: flattens args, clears stolen list args, and calls
        # the compiled function with the flat list.
        flat_args = flatten_fn(args)

        # flat_args is a new list, so we need to clear references from the old list
        for i in inputs_idx_to_clear:
            args[i].clear()

        # this call is boxed to avoid increasing refcount until we reach aot_module_simplified forward
        return compiled_fn(flat_args)

    return wrapper


def get_locals_to_steal(maybe_gm: Any) -> list[Any]:
    # [FUNCTION DOC] Returns the locals-to-steal list from a graph module's metadata, or an empty
    # list if the object is not a graph module, lacks metadata, or has no entry.
    if not isinstance(maybe_gm, torch.fx.GraphModule) or not hasattr(maybe_gm, "meta"):
        return []
    return maybe_gm.meta.get("locals_to_steal", [])


def set_locals_to_steal(gm: torch.fx.GraphModule, locals_to_steal: list[Any]) -> None:
    # [FUNCTION DOC] Stores the locals-to-steal list in the graph module's metadata (mutates gm).
    gm.meta["locals_to_steal"] = locals_to_steal


class Lit:
    # [CLASS DOC] HELPER: Wrapper around a string whose repr is the raw string itself, used to
    # embed literal text (without quotes) when formatting.
    def __init__(self, s: str) -> None:
        # [FUNCTION DOC] Stores the literal string.
        self.s = s

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the stored string unchanged.
        return self.s


warn_once_cache: set[str] = set()


def warn_once(msg: str, stacklevel: int = 1) -> None:
    # [FUNCTION DOC] Emits a warning for msg at most once per process (tracked in a global set),
    # attributing it to the caller at the given stacklevel. Side effect: warnings output and cache.
    # Dynamo causes all warnings.warn (in user code and in Dynamo code) to print all the time.
    # https://github.com/pytorch/pytorch/issues/128427.
    # warn_once is a workaround: if the msg has been warned on before, then we will not
    # warn again.
    # NB: it's totally ok to store a cache of all the strings: this is what warnings.warn does as well.
    if msg in warn_once_cache:
        return
    warn_once_cache.add(msg)
    warnings.warn(msg, stacklevel=stacklevel + 1)


def strip_color_from_string(text: str) -> str:
    # [FUNCTION DOC] Returns text with ANSI escape sequences removed.
    # This regular expression matches ANSI escape codes
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


@contextlib.contextmanager
def _disable_saved_tensors_hooks_during_tracing() -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that turns on the autograd tracing flag for saved
    # tensor hooks (deferring pack/unpack hooks to runtime) and restores the prior flag on exit.
    # KNOWN ISSUE: if enabling the flag raises, prior is unbound in the finally clause.
    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    try:
        prior = torch._C._autograd._saved_tensors_hooks_set_tracing(True)
        yield
    finally:
        torch._C._autograd._saved_tensors_hooks_set_tracing(prior)


def is_parameter_freezing() -> bool:
    # [FUNCTION DOC] True if inductor freezing is configured and grad mode is currently disabled.
    return torch._inductor.config.freezing and not torch.is_grad_enabled()


def get_torch_function_mode_stack() -> list[Any]:
    # [FUNCTION DOC] Returns the current torch function mode stack as a list (bottom first).
    return [
        get_torch_function_mode_stack_at(i) for i in range(_len_torch_function_stack())
    ]


def get_torch_function_mode_stack_at(ind: int) -> Any:
    # [FUNCTION DOC] Returns the torch function mode at index ind of the stack; raises
    # AssertionError if ind is negative or beyond the stack length.
    if ind < 0:
        raise AssertionError(f"Expected ind >= 0, got {ind}")
    if ind >= _len_torch_function_stack():
        raise AssertionError(
            f"Expected ind < _len_torch_function_stack() ({_len_torch_function_stack()}), got {ind}"
        )
    return torch._C._get_function_stack_at(ind)


def set_torch_function_mode_stack(stack: list[Any]) -> None:
    # [FUNCTION DOC] Replaces the global torch function mode stack: pops every current mode then
    # pushes the given modes in order. Mutates global state.
    for _ in range(_len_torch_function_stack()):
        _pop_torch_function_stack()

    for mode in stack:
        _push_on_torch_function_stack(mode)


def clear_torch_function_mode_stack() -> None:
    # [FUNCTION DOC] Pops all modes off the global torch function mode stack.
    for _ in range(_len_torch_function_stack()):
        _pop_torch_function_stack()


def get_current_stream(device: torch.device) -> torch.Stream:
    # [FUNCTION DOC] Returns the accelerator's current stream for the given device.
    return torch.accelerator.current_stream(device)


# call from C dynamo in order to inspect values in pdb
def _breakpoint_for_c_dynamo(*args: Any) -> None:
    # [FUNCTION DOC] HELPER: Drops into the debugger; called from the C dynamo layer to inspect
    # values. Arguments are ignored.
    breakpoint()


def verify_guard_fn_signature(value: Any) -> None:
    # [FUNCTION DOC] Validates a tensor subclass's metadata-guard method: it must take exactly two
    # parameters and be bound to the class (a classmethod). Raises InternalTorchDynamoError
    # otherwise.
    fn = value.__metadata_guard__
    sig = inspect.signature(fn)
    if len(sig.parameters) != 2:
        from .exc import InternalTorchDynamoError

        raise InternalTorchDynamoError(
            "Tensor subclass method __metadata_guard__ must take exactly two subclass metadata arguments"
        )
    if fn.__self__ != value.__class__:
        from .exc import InternalTorchDynamoError

        raise InternalTorchDynamoError(
            "Tensor subclass method __metadata_guard__ must be a classmethod"
        )


def does_not_override_dict_iter_methods(user_cls: Any) -> bool:
    # [FUNCTION DOC] True if the class's items, values, keys and iteration methods are all the
    # builtin dict or OrderedDict implementations (no overrides).
    return (
        user_cls.items in (dict.items, OrderedDict.items)
        and user_cls.values in (dict.values, OrderedDict.values)
        and user_cls.keys in (dict.keys, OrderedDict.keys)
        and user_cls.__iter__ in (dict.__iter__, OrderedDict.__iter__)
    )


# Helper functions below are to prevent TorchDynamo to prevent tracing of
# __torch_function__ calls triggered on tensor properties in the pre graph
# bytecode.
@torch._disable_dynamo
def call_size(x: Any, i: int) -> int:
    # [FUNCTION DOC] Returns x.size(i) with dynamo disabled so the call is not traced.
    return x.size(i)


@torch._disable_dynamo
def call_stride(x: Any, i: int) -> int:
    # [FUNCTION DOC] Returns x.stride(i) with dynamo disabled so the call is not traced.
    return x.stride(i)


@torch._disable_dynamo
def call_storage_offset(x: Any) -> int:
    # [FUNCTION DOC] Returns x.storage_offset() with dynamo disabled so the call is not traced.
    return x.storage_offset()


# Helper function to extract relevant parts of a tensor's __dict__ to store in node meta.
# To avoid ref cycles, it's important that no tensors are present here, so leave those out.
def _extract_tensor_dict(t: torch.Tensor) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns shallow copies of the whitelisted entries (static input type
    # and tag) present in the tensor's attribute dictionary, for storing in node metadata.
    KEYS_TO_COPY = [
        "_dynamo_static_input_type",
        "tag",
    ]

    tensor_dict = {
        key: copy.copy(t.__dict__[key]) for key in KEYS_TO_COPY if key in t.__dict__
    }

    return tensor_dict


def build_stream(args: tuple[Any], kwargs: dict[Any, Any]) -> torch.Stream:
    # [FUNCTION DOC] Constructs a new stream object, forwarding the positional and keyword args.
    return torch._C.Stream(*args, **kwargs)


def build_event(args: tuple[Any], kwargs: dict[Any, Any]) -> torch.Event:
    # [FUNCTION DOC] Constructs a new event object, forwarding the positional and keyword args.
    return torch._C.Event(*args, **kwargs)


class FrameState(enum.Enum):
    # [CLASS DOC] Enum mirroring CPython's frame execution states (created, suspended, executing,
    # completed, cleared) by their numeric values.
    FRAME_CREATED = -3
    FRAME_SUSPENDED = -2
    FRAME_SUSPENDED_YIELD_FROM = -1
    FRAME_EXECUTING = 0
    FRAME_COMPLETED = 1
    FRAME_CLEARED = 4


class PySendResult(enum.Enum):
    # [CLASS DOC] Enum mirroring CPython's PySendResult codes (return, error, next).
    PYGEN_RETURN = 0
    PYGEN_ERROR = -1
    PYGEN_NEXT = 1


class CompileTimeInstructionCounter:
    # [CLASS DOC] Class-level (global, not instantiated) accumulator of CPU instruction counts spent
    # in compilation. State: _counter (accumulated total), _id (active native counter handle), and
    # _depth (nesting depth, only the outermost start/end pair measures). Not thread-safe.
    _counter: int = 0
    _id: int = -1
    _depth = 0

    @classmethod
    def start(cls) -> None:
        # [FUNCTION DOC] Increases nesting depth; at depth one starts the native instruction
        # counter and remembers its id.
        cls._depth = cls._depth + 1
        if cls._depth == 1:
            cls._id = _instruction_counter.start()

    @classmethod
    def end(cls) -> None:
        # [FUNCTION DOC] Decreases nesting depth; when it returns to zero, stops the native counter
        # and adds its count to the total, resetting the id.
        cls._depth = cls._depth - 1
        if cls._depth == 0:
            cls._counter += _instruction_counter.end(cls._id)
            cls._id = -1

    @classmethod
    def clear(cls) -> None:
        # [FUNCTION DOC] Resets the accumulated instruction count to zero.
        cls._counter = 0

    @classmethod
    def value(cls) -> int:
        # [FUNCTION DOC] Returns the accumulated instruction count.
        return cls._counter

    @classmethod
    @contextmanager
    def record(cls) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that, if config.record_compile_time_instruction_count is
        # on, brackets the body with start and end (end runs even on exceptions); otherwise it
        # only yields.
        try:
            if config.record_compile_time_instruction_count:
                cls.start()
            yield
        finally:
            if config.record_compile_time_instruction_count:
                cls.end()


class CompileCounterInt(int):
    # [CLASS DOC] Int subclass whose addition preserves the subclass type, letting trackers
    # recognize a compile-counter integer after arithmetic.
    def __add__(self, other: Any) -> CompileCounterInt:
        # [FUNCTION DOC] Returns the sum as a CompileCounterInt.
        return CompileCounterInt(super().__add__(other))


def set_feature_use(feature: str, usage: bool) -> None:
    """
    Records whether we are using a feature
    Generally a feature is a JK.
    """
    # [FUNCTION DOC] Records feature -> usage in the metrics context's feature_usage mapping if a
    # metrics context is in progress; otherwise does nothing.
    # Note that sometimes (tests etc...) we're not in a context which we can record into
    if get_metrics_context().in_progress():
        get_metrics_context().set_key_value("feature_usage", feature, usage)


OptimizeDDPMode = Literal[
    "ddp_optimizer",
    "python_reducer",
    "python_reducer_without_compiled_forward",
    "no_optimization",
]

_ddp_optimization_mode: tuple[OptimizeDDPMode, ...] = (
    "ddp_optimizer",
    "python_reducer",  # experimental mode
    "python_reducer_without_compiled_forward",
    "no_optimization",
)


def get_optimize_ddp_mode() -> OptimizeDDPMode:
    # [FUNCTION DOC] Resolves config.optimize_ddp to a mode string: True maps to ddp_optimizer,
    # False to no_optimization, a string is used as-is. Raises ValueError for other types and
    # AssertionError for unknown mode strings.
    optimize_ddp = config.optimize_ddp
    if isinstance(optimize_ddp, bool):
        mode = "ddp_optimizer" if optimize_ddp else "no_optimization"
    elif isinstance(optimize_ddp, str):
        mode = optimize_ddp
    else:
        raise ValueError(
            f"Invalid dynamo config optimize_ddp type {type(optimize_ddp)=}"
        )

    if mode not in _ddp_optimization_mode:
        raise AssertionError(f"Invalid dynamo config optimize_ddp value {mode=}")
    return cast("OptimizeDDPMode", mode)


@contextmanager
def maybe_disable_inference_mode() -> Generator[None, None, None]:
    """
    Disables torch.inference_mode for the compilation (still on at runtime).
    This simplifies the compile stack where we can assume that inference_mode
    will always be off.

    Since inference_mode is equivalent to no_grad + some optimizations (version
    counts etc), we turn on no_grad here. The other optimizations are not
    relevant to torch.compile.
    """
    # [FUNCTION DOC] Context manager: if the config flag is set and inference mode is active, runs
    # the body with inference mode off and grad disabled; otherwise just yields.
    is_inference_mode_on = (
        config.fake_tensor_disable_inference_mode and torch.is_inference_mode_enabled()
    )
    if is_inference_mode_on:
        with (
            torch.inference_mode(False),
            torch.no_grad(),
        ):
            yield
    else:
        yield


@contextmanager
def maybe_disable_inference_mode_for_fake_prop() -> Generator[None, None, None]:
    """
    Turns off tracking of inference_mode for fake tensor propagation. With this
    context manager, when a real tensor is converted to fake tensor, the fake
    tensor looses its inference-ness.
    """
    # [FUNCTION DOC] Context manager: if config.fake_tensor_disable_inference_mode is set, disables
    # inference-mode tracking during fake tensor conversion for the body; otherwise just yields.
    if config.fake_tensor_disable_inference_mode:
        with torch._subclasses.meta_utils.disable_inference_mode_for_fake_prop():
            yield
    else:
        yield


def is_node_meta_valid(node: torch.fx.Node | None) -> bool:
    # [FUNCTION DOC] True if node is None or its metadata has an example value or a val entry.
    return node is None or "example_value" in node.meta or "val" in node.meta


# If True, enforce fullgraph=True - raise errors on graph break
_error_on_graph_break = False


def _get_error_on_graph_break() -> bool:
    # [FUNCTION DOC] HELPER: Returns the global flag that makes graph breaks errors (fullgraph).
    return _error_on_graph_break


def _set_error_on_graph_break(value: bool) -> None:
    # [FUNCTION DOC] HELPER: Sets the global error-on-graph-break flag.
    global _error_on_graph_break
    _error_on_graph_break = value


@functools.lru_cache(1)
def _is_tensorify_enabled() -> bool:
    # [FUNCTION DOC] HELPER: Whether python-scalar tensorification is enabled: the environment
    # variable decides if set (enabled unless "0" or "FALSE"), else a justknob. Cached after the
    # first call.
    if (env := os.getenv("TENSORIFY_PYTHON_SCALARS")) is not None:
        return env not in ("0", "FALSE")
    return justknobs_check("pytorch/compiler:tensorify_python_scalars")


@torch._disable_dynamo
def record_pregraph_bytecode_enter() -> AbstractContextManager[None]:
    # [FUNCTION DOC] Starts a "Pregraph bytecode" profiler range (when the profiler is enabled, else
    # a null context), enters it, and returns the entered context manager for the exit function.
    # Dynamo-disabled so it is not traced.
    cm: AbstractContextManager[None] = (
        torch._C._profiler._RecordFunctionFast("Pregraph bytecode")
        if torch.autograd.profiler._is_profiler_enabled
        else contextlib.nullcontext()
    )
    cm.__enter__()
    return cm


@torch._disable_dynamo
def record_pregraph_bytecode_exit(cm: AbstractContextManager[None]) -> None:
    # [FUNCTION DOC] Exits the context manager returned by record_pregraph_bytecode_enter.
    cm.__exit__(None, None, None)


# Active `torch._dynamo.override_cudagraphs` override, set/restored at RUNTIME by
# CudagraphOverrideContextManager.__enter__/__exit__ (mirrors _error_on_graph_break
# above). This lets the override follow the dynamic call stack: a callee that
# cannot be inlined (e.g. it contains a graph break) is compiled as a separate
# frame and would otherwise not inherit the override, which lives lexically in a
# caller's `with` block / decorator. OutputGraph seeds its cudagraph_annotation
# from this value. None means no override is active; otherwise it is the
# (fwd, bwd) pair. NOTE: we intentionally do not guard on this, so the override
# is sticky per code object (first compile wins) for functions reached both
# inside and outside it.
_cudagraph_override: tuple[bool | None, bool | None] | None = None


def _get_cudagraph_override() -> tuple[bool | None, bool | None] | None:
    # [FUNCTION DOC] HELPER: Returns the active cudagraph override pair (forward, backward) or None.
    return _cudagraph_override


def _set_cudagraph_override(value: tuple[bool | None, bool | None] | None) -> None:
    # [FUNCTION DOC] HELPER: Sets the global cudagraph override pair (or None to clear it).
    global _cudagraph_override
    _cudagraph_override = value


# Returns a set of code objects present traced in the current TracingContext, or None
# if there is no current TracingContext.
def get_traced_code() -> list[CodeType] | None:
    # [FUNCTION DOC] Returns the code objects traced so far in the current tracing context, or None
    # if there is no current context.
    from torch._guards import TracingContext

    return TracingContext.get_traced_code()


def is_pybind11_enum_member(value: object) -> bool:
    """Check if value is a pybind11 enum member (with stable hash and eq).

    Pybind11 enums have __members__ on their type. Unlike Python's enum.Enum,
    pybind11 injects __hash__ and __eq__ directly into the type's __dict__,
    which trips raise_on_overridden_hash. But these are safe: members have
    deterministic hash and equality, same as Python enums.

    Note: pybind11 doesn't always return the singleton for enum values (e.g.
    a C++ function returning an enum may construct a new Python wrapper), so
    we check by name membership rather than identity.
    """
    # [FUNCTION DOC] True if value's type has a members mapping and the value's name is one of its
    # keys (pybind11 enum member); False otherwise.
    t = type(value)
    members = getattr(t, "__members__", None)
    if members is None:
        return False
    name = getattr(value, "name", None)
    return name is not None and name in members


def _make_inlined(
    tx: InstructionTranslatorBase, f: Callable[..., Any]
) -> Callable[..., VariableTracker]:
    # [FUNCTION DOC] HELPER: Returns a callable that inlines the Python function f into the current
    # trace via the translator, forcing inlining, and returns the resulting tracker. Raises
    # AssertionError if f is not callable.
    if not callable(f):
        raise AssertionError("Expect f to be a python callable.")

    def inline_call(
        *args: VariableTracker, **kwargs: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Inlines f with the given tracker args and kwargs under forced
        # inlining and returns the call's result tracker.
        from torch._dynamo.trace_rules import _force_inline
        from torch._dynamo.variables.functions import UserFunctionVariable

        with _force_inline():
            return tx.inline_user_function_return(
                UserFunctionVariable(f),
                list(args),
                dict(kwargs),
            )

    return inline_call
