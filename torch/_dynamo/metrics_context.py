# [MODULE DOC]
# Metrics accumulation for Dynamo: TopN (bounded most-expensive list),
# MetricsContext (re-entrant context manager collecting compile-time metrics and
# reporting them to an on_exit callback), and RuntimeMetricsContext (explicitly
# finished runtime collector). Import-time side effect: creates a module logger.
"""Metrics collection and management system for Dynamo.

This module provides context managers for gathering and reporting metrics during
compilation and runtime.

It includes two main components:
- MetricsContext: A context manager for collecting metrics during compilation, supporting
  nested contexts and various metric types (counters, sets, key-value pairs)
- RuntimeMetricsContext: A specialized context for runtime metrics collection that doesn't
  require explicit context management

The metrics system enables comprehensive monitoring and analysis of both compilation and
execution performance.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeAlias
from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Iterator

from torch.utils._traceback import CapturedTraceback


log = logging.getLogger(__name__)


class TopN:
    # [CLASS DOC]
    # Keeps at most at_most (default 25) (value, key) entries with the largest
    # values, stored in a min-heap so the smallest is evicted first. Supports
    # length and iteration; iteration yields (key, value) pairs sorted by value
    # descending.
    """
    Helper to record a list of metrics, keeping only the top N "most expensive" elements.
    """

    def __init__(self, at_most: int = 25) -> None:
        # [FUNCTION DOC]
        # Stores the capacity at_most and creates an empty heap.
        self.at_most = at_most
        self.heap: list[tuple[int, Any]] = []

    def add(self, key: Any, val: int) -> None:
        # [FUNCTION DOC]
        # Records key with score val. If the heap is below capacity, pushes;
        # otherwise pushes and pops the smallest entry. Mutates the heap.
        # KNOWN ISSUE: entries are (val, key) tuples, so equal vals compare keys,
        # which fails for keys that are not orderable.
        # Push if we haven't reached the max size, else push and pop the smallest
        fn = heapq.heappush if len(self.heap) < self.at_most else heapq.heappushpop
        fn(self.heap, (val, key))

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of entries currently kept.
        return len(self.heap)

    def __iter__(self) -> Iterator[tuple[Any, int]]:
        # [FUNCTION DOC]
        # Returns an iterator of (key, value) pairs ordered by value from largest
        # to smallest. Does not modify the heap.
        return ((key, val) for val, key in sorted(self.heap, reverse=True))


OnExitType: TypeAlias = Callable[
    [int, int, dict[str, Any], type[BaseException] | None, BaseException | None],
    None,
]


class MetricsContext:
    # [CLASS DOC]
    # Re-entrant context manager accumulating a mapping of metrics. State:
    # _metrics (name to value), _start_time_ns, _level (nesting depth), _edits
    # (captured tracebacks with the metric names they set, for error messages).
    # Metrics reset on entering the outermost level; the on_exit callback is
    # invoked only when the outermost level exits. Setter methods raise outside
    # a context.
    def __init__(self, on_exit: OnExitType) -> None:
        # [FUNCTION DOC]
        # Stores the on_exit callback (start ns, end ns, metrics, exception type,
        # exception) and initializes empty metrics, zero level and no edits.
        """
        Use this class as a contextmanager to create a context under which to accumulate
        a set of metrics, e.g., metrics gathered during a compilation. On exit of the
        contextmanager, call the provided 'on_exit' function and pass a dictionary of
        all metrics set during the lifetime of the contextmanager.
        """
        self._on_exit = on_exit
        self._metrics: dict[str, Any] = {}
        self._start_time_ns: int = 0
        self._level: int = 0
        self._edits: list[tuple[CapturedTraceback, set[str]]] = []

    def __enter__(self) -> Self:
        # [FUNCTION DOC]
        # Enters the context. At the outermost level, clears metrics and records
        # the start time. Increments the nesting level and returns self.
        # KNOWN ISSUE: _edits is never cleared, so it accumulates across contexts.
        """
        Initialize metrics recording.
        """
        if self._level == 0:
            # In case of recursion, track at the outermost context.
            self._metrics = {}
            self._start_time_ns = time.time_ns()

        self._level += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: Any,
    ) -> None:
        # [FUNCTION DOC]
        # Decrements the nesting level; raises AssertionError if it goes
        # negative. When the outermost level exits, calls on_exit with start and
        # end times, the metrics and the exception info. Exceptions raised by
        # on_exit are logged and swallowed. Never suppresses the user exception.
        """
        At exit, call the provided on_exit function.
        """
        self._level -= 1
        if self._level < 0:
            raise AssertionError("MetricsContext level cannot become negative")
        if self._level == 0:
            try:
                end_time_ns = time.time_ns()
                self._on_exit(
                    self._start_time_ns, end_time_ns, self._metrics, exc_type, exc_value
                )
            except Exception:
                log.exception("Unexpected exception logging compilation metrics")

    def in_progress(self) -> bool:
        # [FUNCTION DOC]
        # Returns true if the context is currently entered (level above zero).
        """
        True if we've entered the context.
        """
        return self._level > 0

    def increment(self, metric: str, value: int) -> None:
        # [FUNCTION DOC]
        # Adds value to the named metric, starting from zero if absent. Raises
        # RuntimeError if called outside the context.
        """
        Increment a metric by a given amount.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot increment {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = 0
        self._metrics[metric] += value

    def _render_edits(self, pred: set[str]) -> str:
        # [FUNCTION DOC] HELPER: Builds a string of formatted previous tracebacks
        # for every recorded edit whose metric names intersect pred, for use in
        # duplicate-set error messages. Always starts with blank lines.
        return "\n\n" + "\n\n".join(
            "Previous Traceback:\n" + "".join(e.format())
            for e, k in self._edits
            if k & pred
        )

    def set(self, metric: str, value: Any, overwrite: bool = False) -> None:
        # [FUNCTION DOC]
        # Sets the named metric to value and records the caller traceback.
        # Raises RuntimeError outside the context, or if the metric is already
        # set and overwrite is false (message includes previous tracebacks).
        """
        Set a metric to a given value. Raises if the metric has been assigned previously
        in the current context.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot set {metric} outside of a MetricsContext")
        if metric in self._metrics and not overwrite:
            raise RuntimeError(
                self._render_edits({metric})
                + f"\n\nRuntimeError: Metric '{metric}' has already been set in the current context "
                "(see above for current and previous traceback)."
            )
        self._edits.append((CapturedTraceback.extract(skip=1), {metric}))
        self._metrics[metric] = value

    def set_key_value(self, metric: str, key: str, value: Any) -> None:
        # [FUNCTION DOC]
        # Treats the metric as a mapping (created empty if absent) and stores
        # value under key; repeated calls are allowed and overwrite. Raises
        # RuntimeError outside the context. Does not record an edit traceback.
        """
        Treats a given metric as a dictionary and set the k and value within it.
        Note that the metric must be a dictionary or not present.

        We allow this to be called multiple times (i.e. for features, it's not uncommon
        for them to be used multiple times within a single compilation).
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot set {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = {}
        self._metrics[metric][key] = value

    def update(self, values: dict[str, Any], overwrite: bool = False) -> None:
        # [FUNCTION DOC]
        # Sets many metrics from the values mapping (no incrementing) and records
        # the caller traceback. Raises RuntimeError outside the context, or if
        # any name is already set and overwrite is false.
        """
        Set multiple metrics directly. This method does NOT increment. Raises if any
        metric has been assigned previously in the current context and overwrite is
        not set to True.
        """
        if self._level == 0:
            raise RuntimeError("Cannot update metrics outside of a MetricsContext")
        existing = self._metrics.keys() & values.keys()
        if existing and not overwrite:
            raise RuntimeError(
                self._render_edits(set(values.keys()))
                + f"\n\nRuntimeError: Metric(s) {existing} have already been set in the current context.  "
                "(see above for current and previous traceback)."
            )
        self._edits.append((CapturedTraceback.extract(skip=1), set(values.keys())))
        self._metrics.update(values)

    def update_outer(self, values: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Calls update (without overwrite) only when at the outermost level; does
        # nothing when nested. Raises RuntimeError outside the context.
        """
        Update, but only when at the outermost context.
        """
        if self._level == 0:
            raise RuntimeError("Cannot update metrics outside of a MetricsContext")
        if self._level == 1:
            self.update(values)

    def add_to_set(self, metric: str, value: Any) -> None:
        # [FUNCTION DOC]
        # Adds value to the set stored under the metric (created if absent).
        # Raises RuntimeError outside the context.
        """
        Records a metric as a set() of values.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot add {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = set()
        self._metrics[metric].add(value)

    def add_top_n(self, metric: str, key: Any, val: int) -> None:
        # [FUNCTION DOC]
        # Adds (key, val) to the TopN stored under the metric (created if
        # absent). Silently does nothing outside the context, unlike the other
        # setters which raise.
        """
        Records a metric as a TopN set of values.
        """
        if self._level == 0:
            return
        if metric not in self._metrics:
            self._metrics[metric] = TopN()
        self._metrics[metric].add(key, val)


class RuntimeMetricsContext:
    # [CLASS DOC]
    # Collector for runtime metrics not tied to a compilation scope. State:
    # _on_exit callback, _metrics mapping, _start_time_ns (set when the first
    # metric arrives). Metrics are flushed explicitly via finish.
    def __init__(self, on_exit: OnExitType) -> None:
        # [FUNCTION DOC]
        # Stores the on_exit callback and initializes empty metrics and start time.
        """
        Similar to MetricsContext, but used to gather the runtime metrics that are
        decoupled from compilation, where there's not a natural place to insert a
        context manager.
        """
        self._on_exit = on_exit
        self._metrics: dict[str, Any] = {}
        self._start_time_ns: int = 0

    def increment(
        self, metric: str, value: int, extra: dict[str, Any] | None = None
    ) -> None:
        # [FUNCTION DOC]
        # Adds value to the named metric (starting at zero). Starts the timer when
        # the metrics are empty. Optional extra mapping: each key not yet present
        # and with a non-none value is stored as is. Never raises on its own.
        """
        Increment a metric by a given amount.
        """
        if not self._metrics:
            # Start timing on the first entry
            self._start_time_ns = time.time_ns()
        if metric not in self._metrics:
            self._metrics[metric] = 0
        self._metrics[metric] += value

        if extra:
            for k, v in extra.items():
                if k not in self._metrics and v is not None:
                    self._metrics[k] = v

    def finish(self) -> None:
        # [FUNCTION DOC]
        # If any metrics exist, calls on_exit with start and end times, the
        # metrics, and no exception info; callback errors are logged and
        # swallowed. Always clears the metrics afterwards. Does nothing if empty.
        """
        Call the on_exit function with the metrics gathered so far and reset.
        """
        if self._metrics:
            try:
                end_time_ns = time.time_ns()
                self._on_exit(
                    self._start_time_ns, end_time_ns, self._metrics, None, None
                )
            except Exception:
                log.exception("Unexpected exception logging runtime metrics")
            finally:
                self._metrics = {}
