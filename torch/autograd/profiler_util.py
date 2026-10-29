# mypy: allow-untyped-defs
import bisect
import itertools
import json
import math
from collections import defaultdict, namedtuple
from collections.abc import Callable
from operator import attrgetter
from typing import Any, NamedTuple
from typing_extensions import deprecated

import torch
from torch.autograd import DeviceType


__all__ = [
    "EventList",
    "EventMetadata",
    "FormattedTimesMixin",
    "Interval",
    "Kernel",
    "FunctionEvent",
    "FunctionEventAvg",
    "StringTable",
    "MemRecordsAcc",
]


class EventList(list):
    # [CLASS DOC] List subclass holding FunctionEvent (or FunctionEventAvg) objects with analysis helpers: building the CPU parent/child tree, table formatting,
    # Chrome trace and stack export, and averaging by key. Extra state: the profiled device name, whether memory profiling and flop counting are on, and whether the tree has been built
    # (key_averages requires it). Usually created by the profiler, not by users.
    """A list of profiling events with helper methods for analysis and visualization.

    EventList extends the standard Python list to provide specialized methods for
    working with profiling events (FunctionEvent or FunctionEventAvg objects).
    It includes utilities for aggregating statistics, formatting output tables,
    and exporting profiling data.

    This class is typically returned by profiler methods and should not be
    instantiated directly by users.

    Args:
        *args: Standard list arguments.
        use_device (str, optional): Device type for profiling ("cuda", "xpu", etc.).
        profile_memory (bool, optional): Whether memory profiling was enabled. Default: False.
        with_flops (bool, optional): Whether to include FLOP counts. Default: False.

    Attributes:
        _use_device (str): Device type being profiled.
        _profile_memory (bool): Whether memory profiling is enabled.
        _with_flops (bool): Whether FLOP counting is enabled.
        _tree_built (bool): Whether the event tree structure has been built.

    Key Methods:
        table(...): Format events as a table string for display.
        export_chrome_trace(path): Export to Chrome tracing format.
        export_stacks(path, metric): Export stack traces with metrics.
        key_averages(...): Compute averaged statistics grouped by operation name.
        total_average(): Compute aggregate totals across all events (sums, not averages).

    Properties:
        self_cpu_time_total: Sum of self CPU time across all events.

    Example::

        import torch
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            x = torch.randn(100, 100)
            y = torch.matmul(x, x)

        # EventList is returned by prof.events()
        events = prof.events()

        # Display as formatted table
        print(
            events.table(
                sort_by="cpu_time_total", row_limit=20, top_level_events_only=False
            )
        )

        # Export to Chrome tracing format
        events.export_chrome_trace("trace.json")

        # Get averaged statistics
        avg_events = events.key_averages()
        print(avg_events.table())

        # Export stack traces
        events.export_stacks("stacks.txt", "self_cpu_time_total")

    See Also:
        - :class:`FunctionEvent`: Individual profiling event
        - :class:`FunctionEventAvg`: Averaged profiling statistics
        - :meth:`table`: Format events as a readable table
        - :meth:`key_averages`: Aggregate events by operation name
    """

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Pops use_device, profile_memory and with_flops from the keyword arguments (defaults None, False, False), passes the rest to list, and marks the tree as not built.
        use_device = kwargs.pop("use_device", None)
        profile_memory = kwargs.pop("profile_memory", False)
        with_flops = kwargs.pop("with_flops", False)
        # pyrefly: ignore [not-iterable]
        super().__init__(*args, **kwargs)
        self._use_device = use_device
        self._profile_memory = profile_memory
        self._tree_built = False
        self._with_flops = with_flops

    def _build_tree(self):
        # [FUNCTION DOC] HELPER: Links CPU parents and children, removes duplicate parent/child nodes of the same name, assigns backward stack traces from forward events, and marks the tree as built.
        self._populate_cpu_children()
        self._remove_dup_nodes()
        self._set_backward_stacktraces()
        self._tree_built = True

    def __str__(self):
        # [FUNCTION DOC] Returns the default formatted table of the events.
        return self.table()

    def _remove_dup_nodes(self):
        # [FUNCTION DOC] HELPER: Repeatedly removes events whose only-child parent has the same name: the child's children and kernels are lifted onto the parent, and the child is deleted from the list in place. Loops until no more removals occur.
        while True:
            to_delete = set()

            for idx in range(len(self)):
                if (
                    self[idx].cpu_parent is not None
                    and self[idx].cpu_parent.name == self[idx].name
                    and len(self[idx].cpu_parent.cpu_children) == 1
                ):
                    self[idx].cpu_parent.cpu_children = self[idx].cpu_children
                    self[idx].cpu_parent.kernels = self[idx].kernels  # lift kernels up
                    for ch in self[idx].cpu_children:
                        ch.cpu_parent = self[idx].cpu_parent
                    to_delete.add(idx)
            if len(to_delete) == 0:
                break

            new_evts = [ev for ind, ev in enumerate(self) if ind not in to_delete]

            self.clear()

            self.extend(new_evts)

    def _populate_cpu_children(self):
        # [FUNCTION DOC] HELPER: Computes parent and child links among synchronous CPU events. Events are grouped by (thread, node id), sorted by start then descending end, and nested using a stack
        # so that an event fully inside another becomes its child; partially overlapping events get no relation. Raises AssertionError if an event would receive a second parent. Mutates the events.
        """Populate child events into each underlying FunctionEvent object.

        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """
        # Some events can be async (i.e. start and end on different threads),
        # since it's generally undefined how to attribute children ranges to
        # async ranges, we do not use them when calculating nested ranges and stats
        sync_events = [
            evt
            for evt in self
            if not evt.is_async and evt.device_type == DeviceType.CPU
        ]
        events = sorted(
            sync_events,
            key=attrgetter("thread"),
        )
        # Group by both thread and node_id, so that events that happen to have
        # the same thread_id but are from different nodes aren't incorrectly
        # grouped together.
        threads = itertools.groupby(
            events, key=lambda event: (event.thread, event.node_id)
        )

        # For each thread we keep a stack of current nested parents.
        # We maintain the invariant that each interval is a subset of all other
        # intervals lower in the stack.
        #
        # First we sort the intervals by their start time. Then we iterate over them.
        # Every time we see a new interval we remove several parents from
        # the top until we restore the invariant. Then parent child relationship
        # if recorded if the stack is not empty.
        # Finally we add new interval to the list
        #
        # Algorithm has O(N * log(N)) complexity where N is number of
        # intervals
        for _thread_id, thread_events in threads:
            thread_events_ = sorted(
                thread_events,
                key=lambda event: [event.time_range.start, -event.time_range.end],
            )
            current_events: list[FunctionEvent] = []
            for event in thread_events_:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if (
                        event.time_range.start >= parent.time_range.end
                        or event.time_range.end > parent.time_range.end
                    ):
                        # this can't be a parent
                        current_events.pop()
                    else:
                        parent.append_cpu_child(event)
                        if event.cpu_parent is not None:
                            raise AssertionError(
                                f"There is already a CPU parent event for {event.key}"
                            )
                        event.set_cpu_parent(parent)
                        break

                current_events.append(event)

    def _set_backward_stacktraces(self):
        # [FUNCTION DOC] HELPER: For events under a backward-function ancestor, replaces the stack with the stack of the forward event having the same sequence number and forward thread (empty if none).
        # Forward stacks are collected from events with no backward ancestor. Raises AssertionError if a backward ancestor lacks a forward thread. Mutates event stacks.
        def bw_parent(evt):
            # [FUNCTION DOC] HELPER: Returns the nearest ancestor (including itself) with backward-function scope, or None if there is none.
            if evt is None:
                return None
            elif evt.scope == 1:  # BACKWARD_FUNCTION
                return evt
            else:
                return bw_parent(evt.cpu_parent)

        fwd_stacks = {}
        for evt in self:
            if bw_parent(evt) is None and evt.stack is not None:
                t = (evt.sequence_nr, evt.thread)
                if t not in fwd_stacks:
                    fwd_stacks[t] = evt.stack

        for evt in self:
            p = bw_parent(evt)
            if p is not None:
                if p.fwd_thread is None:
                    raise AssertionError(
                        "Expected fwd_thread to be set for backward parent"
                    )
                t = (p.sequence_nr, p.fwd_thread)
                evt.stack = fwd_stacks.get(t, [])

    @property
    def self_cpu_time_total(self):
        # [FUNCTION DOC] Property: sum of self CPU time over all events.
        return sum(event.self_cpu_time_total for event in self)

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
        time_unit=None,
    ):
        # [FUNCTION DOC] Returns the events formatted as a text table, delegating to the table builder with this list's memory and flops settings. See the docstring for argument meanings.
        """Print an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``xpu_time``,
                ``cpu_time_total``, ``cuda_time_total``, ``xpu_time_total``,
                ``cpu_memory_usage``, ``cuda_memory_usage``, ``xpu_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``,
                ``self_xpu_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda/xpu ops events are omitted for profiler result readability.
            time_unit(str, optional): A time unit to be used for all values in the
                table. Valid options are: ``s``, ``ms`` and ``us``.

        Returns:
            A string containing the table.
        """
        return _build_table(
            self,
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
            top_level_events_only=top_level_events_only,
            time_unit=time_unit,
        )

    def export_chrome_trace(self, path):
        # [FUNCTION DOC] Writes a minimal Chrome trace JSON file to path (overwritten): one complete-duration entry per event having a trace name, plus a flow-start entry per kernel.
        # Written manually for speed, trimming the trailing separator. Side effect: file I/O only; the device name used in flow categories defaults to cuda.
        """Export an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
        import os

        device_name = "cuda" if not self._use_device else self._use_device
        with open(path, "w") as f:
            next_id = 0
            # Use file IO over using json.dump since JSON dumping is very slow and
            # this technique is proven to give a 4x speedup.
            f.write("[")
            for evt in self:
                if evt.trace_name is None:
                    continue
                name_json = json.dumps(evt.trace_name)
                f.write(
                    '{{"name": {}, '
                    '"ph": "X", '
                    '"ts": {}, '
                    '"dur": {}, '
                    '"tid": {}, '
                    '"pid": "CPU functions", '
                    '"args": {{}}}}, '.format(
                        name_json,
                        evt.time_range.start,
                        evt.time_range.elapsed_us(),
                        evt.thread
                        if not evt.is_remote
                        else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "',
                    )
                )
                for _ in evt.kernels:
                    # 's' and 'f' draw Flow arrows from
                    # the CPU launch to the GPU kernel
                    f.write(
                        f'{{"name": {name_json}, '
                        '"ph": "s", '
                        f'"ts": {evt.time_range.start}, '
                        f'"tid": {evt.thread}, '
                        '"pid": "CPU functions", '
                        f'"id": {next_id}, '
                        f'"cat": "cpu_to_{device_name}", '
                        '"args": {}}, '
                    )
                    # Note: use torch.profiler to get device kernel trace
                    next_id += 1
            if len(self) > 0:
                # remove trailing whitespace and comma
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.truncate()
            f.write("]")

    def supported_export_stacks_metrics(self):
        # [FUNCTION DOC] Returns the list of metric names accepted by export_stacks.
        return [
            "self_cpu_time_total",
            "self_cuda_time_total",
            "self_xpu_time_total",
            "self_privateuse1_time_total",
        ]

    def export_stacks(self, path: str, metric: str):
        # [FUNCTION DOC] Writes flamegraph-style lines to path for events having stacks and a positive integer metric value: stack entries (spaces, semicolons, tabs and newlines replaced by underscores)
        # joined by semicolons from outermost to innermost, then a space and the value. Device metrics are mapped to the generic device attribute. Raises ValueError for an unsupported metric.
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError(
                "metric should be one of: "
                + str(self.supported_export_stacks_metrics())
            )
        translate_table = str.maketrans(" ;\t\n", "____")
        with open(path, "w") as f:
            for evt in self:
                if evt.stack and len(evt.stack) > 0:
                    metric_value = getattr(
                        evt,
                        metric.replace("cuda", "device")
                        .replace("xpu", "device")
                        .replace("privateuse1", "device"),
                    )
                    if int(metric_value) > 0:
                        stack_str = ""
                        for entry in reversed(evt.stack):
                            stack_str += entry.translate(translate_table)
                            stack_str += ";"
                        stack_str = stack_str[:-1] + " " + str(int(metric_value))
                        f.write(stack_str + "\n")

    def key_averages(
        self,
        group_by_input_shapes=False,
        group_by_stack_n=0,
        group_by_overload_name=False,
        include_python_functions=False,
    ):
        # [FUNCTION DOC] Aggregates events into one FunctionEventAvg per key (name, node, device, optionally shapes, stack prefix, overload name) and returns them as an EventList carrying this list's settings.
        # Requires the tree to be built (raises AssertionError otherwise). Python-function events are excluded unless requested. See the docstring for argument meanings.
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

            group_by_overload_name: Differentiate operators by their overload name e.g. aten::add.Tensor
            and aten::add.out will be aggregated separately

            include_python_functions: include Python function events (e.g. individual
                Python callsite entries captured with ``with_stack=True``) in the
                averages. By default these are excluded because they tend to appear as
                misleading hotspots (e.g. ``threading.py: wait``) that obscure the
                real operator-level breakdown. Set to ``True`` to restore the raw
                per-callsite view.

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        if not self._tree_built:
            raise AssertionError(
                "Expected tree to be built before calling key_averages"
            )
        stats: dict[tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(
        # [FUNCTION DOC] HELPER: Builds the grouping key tuple for an event: its key, node id, device type, legacy flag and user-annotation flag, plus optionally the overload name, input shapes and top stack entries.
        # KNOWN ISSUE: the overload name is read from the enclosing loop variable evt instead of the event parameter; it only works because the caller passes that same event.
            event, group_by_input_shapes, group_by_stack_n, group_by_overload_name
        ) -> tuple[str, ...]:
            key = [
                str(event.key),
                str(event.node_id),
                str(event.device_type),
                str(event.is_legacy),
                str(event.is_user_annotation),
            ]
            if group_by_overload_name:
                key.append(evt.overload_name)
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)

        for evt in self:
            if evt.is_python_function and not include_python_functions:
                continue
            stats[
                get_key(
                    evt, group_by_input_shapes, group_by_stack_n, group_by_overload_name
                )
            ].add(evt)

        avg_list = EventList(
            stats.values(),
            use_device=self._use_device,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
        )
        for evt in avg_list:
            evt.stack = evt.stack[:group_by_stack_n]
            if not group_by_input_shapes:
                evt.input_shapes = ""
            if not group_by_overload_name:
                evt.overload_name = ""
        return avg_list

    def total_average(self):
        # [FUNCTION DOC] Sums all events into a single FunctionEventAvg whose key is set to Total and returns it (averaged metrics then mix all operations).
        """Compute aggregate statistics across all events.

        Accumulates statistics from all events into a single FunctionEventAvg object.
        This is primarily useful for computing total metrics (total CPU time, total
        memory usage, etc.) across the entire profiling session, regardless of
        operation type.

        Note:
            This sums up times and counts across ALL different operations, so the
            "average" metrics (like cpu_time) represent the average time per operation
            call across the entire session, mixing all operation types together.
            For per-operation averages, use :meth:`key_averages` instead.

        Returns:
            FunctionEventAvg: A single aggregate object with key="Total" containing
                accumulated statistics.

        """
        total_stat = FunctionEventAvg()
        for evt in self:
            total_stat += evt
            total_stat.key = None
        total_stat.key = "Total"
        return total_stat


def _format_time(time_us):
    # [FUNCTION DOC] HELPER: Formats a microsecond value as a string with three decimals and unit s, ms or us depending on magnitude.
    """Define how to format time in FunctionEvent."""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"


def _format_time_share(time_us, total_time_us):
    # [FUNCTION DOC] HELPER: Returns the percentage of time_us in total_time_us with two decimals, or NaN if the total is zero (raises AssertionError if the total is zero but the part is not).
    # KNOWN ISSUE: the docstring is copied from the time formatter and is inaccurate.
    """Define how to format time in FunctionEvent."""
    if total_time_us == 0:
        if time_us != 0:
            raise AssertionError(f"Expected time_us == 0 but got {time_us}")
        return "NaN"
    return f"{time_us * 100.0 / total_time_us:.2f}%"


def _format_memory(nbytes):
    # [FUNCTION DOC] HELPER: Formats a byte count (sign-aware) as GB, MB, KB with two decimals, or bytes.
    """Return a formatted memory size string."""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} GB"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} MB"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} KB"
    else:
        return str(nbytes) + " B"


def _attr_formatter(name):
    # [FUNCTION DOC] HELPER: Returns a read-only property that formats the named time attribute of the instance as a time string.
    return property(lambda self: _format_time(getattr(self, name)))


class FormattedTimesMixin:
    # [CLASS DOC] Mixin giving FunctionEvent and FunctionEventAvg formatted-time string properties and per-call average times. Subclasses must define the total time attributes and count.
    """Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """

    cpu_time_str = _attr_formatter("cpu_time")
    device_time_str = _attr_formatter("device_time")
    cpu_time_total_str = _attr_formatter("cpu_time_total")
    device_time_total_str = _attr_formatter("device_time_total")
    self_cpu_time_total_str = _attr_formatter("self_cpu_time_total")
    self_device_time_total_str = _attr_formatter("self_device_time_total")

    @property
    def cpu_time(self):
        # [FUNCTION DOC] Property: average CPU time per call (zero when count is zero).
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count  # type: ignore[attr-defined]

    @property
    def device_time(self):
        # [FUNCTION DOC] Property: average device time per call (zero when count is zero).
        return 0.0 if self.count == 0 else 1.0 * self.device_time_total / self.count  # type: ignore[attr-defined]

    @property
    @deprecated(
        "`cuda_time` is deprecated, please use `device_time` instead.",
        category=FutureWarning,
    )
    def cuda_time(self):  # To be deprecated
        # [FUNCTION DOC] Deprecated property alias of device_time (emits a FutureWarning).
        return self.device_time


class Interval:
    # [CLASS DOC] Simple closed time range holding start and end timestamps in microseconds.
    def __init__(self, start, end):
        # [FUNCTION DOC] Stores start and end.
        self.start = start
        self.end = end

    def elapsed_us(self):
        # [FUNCTION DOC] Returns end minus start.
        r"""
        Returns the length of the interval
        """
        return self.end - self.start


Kernel = namedtuple("Kernel", ["name", "device", "duration"])


class EventMetadata(NamedTuple):
    # [CLASS DOC] Immutable record of structured Kineto metadata for an event, grouped as kernel, memory and NCCL collective fields; every field may be None when absent.
    # Kernel fields
    registers_per_thread: int | None
    shared_memory: int | None
    grid: list[int] | None
    block: list[int] | None
    priority: int | None
    blocks_per_sm: float | None
    warps_per_sm: float | None
    occupancy: dict[str, Any] | None
    est_occupancy_pct: float | None
    queued: int | None
    graph_id: int | None
    graph_node_id: int | None
    stream: int | None
    context: int | None
    channel: int | None
    channel_type: int | None
    # Memory fields
    bytes: int | None
    bandwidth_gb_s: float | None
    # NCCL fields
    collective_name: str | None
    dtype: str | None
    in_msg_nelems: int | None
    out_msg_nelems: int | None
    in_split_size: str | None
    out_split_size: str | None
    global_rank_start: int | None
    global_rank_stride: int | None
    group_size: int | None
    process_group_name: str | None
    process_group_desc: str | None
    group_ranks: str | None
    rank: int | None
    src_rank: int | None
    dst_rank: int | None
    seq: int | None
    is_async: bool | None


def _to_str(v: str) -> str:
    # [FUNCTION DOC] HELPER: Strips double quote characters from both ends of the string.
    return v.strip('"')


def _to_bool(v: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns True for the strings 1 and true, else False.
    return v in ("1", "true")


# Kineto key → (EventMetadata field name, converter from string)
_EVENT_METADATA_KEYS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "registers per thread": ("registers_per_thread", int),
    "shared memory": ("shared_memory", int),
    "grid": ("grid", json.loads),  # list[int]
    "block": ("block", json.loads),  # list[int]
    "priority": ("priority", int),
    "blocks per SM": ("blocks_per_sm", float),
    "warps per SM": ("warps_per_sm", float),
    "est. achieved occupancy %": ("est_occupancy_pct", float),
    "occupancy": ("occupancy", json.loads),  # dict[str, Any]
    "queued": ("queued", int),
    "graph id": ("graph_id", int),
    "graph node id": ("graph_node_id", int),
    "stream": ("stream", int),
    "context": ("context", int),
    "channel": ("channel", int),
    "channel_type": ("channel_type", int),
    "bytes": ("bytes", int),
    "memory bandwidth (GB/s)": ("bandwidth_gb_s", float),
    "Collective name": ("collective_name", _to_str),
    "dtype": ("dtype", _to_str),
    "In msg nelems": ("in_msg_nelems", int),
    "Out msg nelems": ("out_msg_nelems", int),
    "In split size": ("in_split_size", _to_str),
    "Out split size": ("out_split_size", _to_str),
    "Global rank start": ("global_rank_start", int),
    "Global rank stride": ("global_rank_stride", int),
    "Group size": ("group_size", int),
    "Process Group Name": ("process_group_name", _to_str),
    "Process Group Description": ("process_group_desc", _to_str),
    "Process Group Ranks": ("group_ranks", _to_str),
    "Rank": ("rank", int),
    "Src Rank": ("src_rank", int),
    "Dst Rank": ("dst_rank", int),
    "Seq": ("seq", int),
    "Is asynchronized op": ("is_async", _to_bool),
}


def _build_metadata(extra_meta):
    # [FUNCTION DOC] HELPER: Converts a mapping of Kineto metadata strings into an EventMetadata using the key table and converters; absent keys become None.
    # Returns None if no known key was present.
    fields: dict[str, Any] = {}
    any_populated = False
    for kineto_key, (field_name, convert) in _EVENT_METADATA_KEYS.items():
        v = extra_meta.get(kineto_key)
        if v is not None:
            fields[field_name] = convert(v)
            any_populated = True
        else:
            fields[field_name] = None
    return EventMetadata(**fields) if any_populated else None


class FunctionEvent(FormattedTimesMixin):
    # [CLASS DOC] Record of a single profiled operation or device activity: id, name, thread, time interval, kernels, CPU parent/children, shapes, stack, memory usage, device info, flops,
    # metadata and flow ids. Provides derived totals (CPU/device time, self times, memory) computed from the tree links and kernels. Normally created by the profiler.
    """Profiling information about a single function.

    FunctionEvent records the execution of a single operation during profiling.
    These events are obtained from the profiler/kineto and contain detailed
    timing and memory usage information.

    .. note::
        FunctionEvent objects are typically created by the profiler/kineto and should not
        be instantiated directly by users. Access them through the profiler's output.

    Attributes:
        id (int): Unique identifier for this event.
        node_id (int): Node identifier for distributed profiling (-1 if not applicable).
        name (str): Name of the profiled function/operator.
        overload_name (str): Overload name for the operator (requires _ExperimentalConfig(capture_overload_names=True) set).
        trace_name (str): Same as name, just changes ProfilerStep* to ProfilerStep#
        time_range (Interval): Time interval containing start and end timestamps in microseconds.
        thread (int): Thread ID where the operation started.
        fwd_thread (int): Thread ID of the corresponding forward operation.
        kernels (List[Kernel]): List of device kernels launched by this operation.
        count (int): Number of times this event was called (usually 1).
        cpu_children (List[FunctionEvent]): Direct CPU child operations.
        cpu_parent (FunctionEvent): Direct CPU parent operation.
        input_shapes (List[List[int]]): Shapes of input tensors (requires record_shapes=True).
            For plain tensor inputs, each entry is a list of dimensions (e.g. ``[16, 16]``).
            TensorList inputs are represented as an empty list ``[]``; use
            ``structured_input_shapes`` to get per-element shapes for TensorList inputs.
        concrete_inputs (List[Any]): Concrete input values (requires record_shapes=true).
        kwinputs (Dict[str, Any]): Keyword arguments (requires record_shapes=true).
        stack (List[str]): Python stack trace where the operation was called (requires with_stack=true).
        scope (int): at::RecordScope identifier (0=forward, 1=backward, etc.).
        use_device (str): Device type being profiled ("cuda", "xpu", etc.).
        cpu_memory_usage (int): CPU memory allocated in bytes.
        device_memory_usage (int): Device memory allocated in bytes.
        is_async (bool): Whether this is an asynchronous operation.
        is_remote (bool): Whether this operation occurred on a remote node.
        sequence_nr (int): Sequence number for autograd operations.
        device_type (DeviceType): Type of device (CPU, CUDA, XPU, PrivateUse1, etc.).
        device_index (int): Index of the device (e.g., GPU 0, 1, 2).
        device_resource_id (int): Resource ID on the device (ie. stream ID).
        is_legacy (bool): Whether this is from the legacy profiler.
        flops (int): Estimated floating point operations.
        is_user_annotation (bool): Whether this is a user-annotated region.
        metadata (Dict[str, Any]): Additional metadata keyed by the field names
            used in exported traces. Use
            ``_ExperimentalConfig(expose_kineto_event_metadata=True)`` to expose
            Kineto activity metadata. Available fields vary by activity and backend.
        metadata_json (str): Deprecated. Use event_metadata instead.
        event_metadata (EventMetadata): Additional metadata in structured format.
        structured_input_shapes (List[List[int] | List[List[int]]]): Like ``input_shapes``
            but distinguishes TensorList inputs.  Plain tensor inputs are ``List[int]``;
            TensorList inputs are ``List[List[int]]`` containing one shape per tensor in the list.
            Matches the ``"Input Dims"`` field in the Chrome trace JSON.
        structured_input_strides (List[List[int] | List[List[int]]]): Strides of input
            tensors in the same format as ``structured_input_shapes`` (requires
            record_shapes=True).

    Properties:
        cpu_time_total (float): Total CPU time in microseconds.
        device_time_total (float): Total device (CUDA/XPU/etc) time in microseconds.
        self_cpu_time_total (float): CPU time excluding child operations.
        self_device_time_total (float): Device time excluding child operations.
        self_cpu_memory_usage (int): CPU memory usage excluding child operations.
        self_device_memory_usage (int): Device memory usage excluding child operations.
        cpu_time (float): Average CPU time per call.
        device_time (float): Average device time per call.
        key (str): Key used for grouping events (usually same as name).

    See Also:
        - :class:`torch.profiler.profile`: Context manager for profiling
        - :class:`EventList`: List container for FunctionEvent objects with helper methods
        - :class:`FunctionEventAvg`: Averaged statistics over multiple FunctionEvent objects
    """

    def __init__(
        self,
        id,
        name,
        thread,
        start_us,
        end_us,
        overload_name=None,
        fwd_thread=None,
        input_shapes=None,
        stack=None,
        scope=0,
        use_device=None,
        cpu_memory_usage=0,
        device_memory_usage=0,
        is_async=False,
        is_remote=False,
        sequence_nr=-1,
        node_id=-1,
        device_type=DeviceType.CPU,
        device_index=0,
        device_resource_id=None,
        is_legacy=False,
        flops=None,
        trace_name=None,
        concrete_inputs=None,
        kwinputs=None,
        is_user_annotation=False,
        is_python_function=False,
        activity_type=None,
        metadata_json=None,
        flow_id=None,
        flow_type=None,
        flow_start=None,
        external_id=0,
        linked_correlation_id=0,
        extra_meta=None,
        structured_input_shapes=None,
        structured_input_strides=None,
        input_dtypes=None,
        python_id=-1,
        python_parent_id=-1,
        python_module_id=-1,
        typed_metadata=None,
    ):
        # [FUNCTION DOC] Stores all event fields as attributes: identifiers, naming, time interval, thread, empty kernel list and CPU children, shapes, stack, memory, device info, flops, flow and python ids.
        # The device resource id defaults to the thread. Structured metadata is built from the extra_meta mapping when given; typed_metadata is stored as the metadata mapping. Count starts at 1.
        self.id: int = id
        self.node_id: int = node_id
        self.name: str = name
        # pyrefly: ignore [bad-assignment]
        self.overload_name: str = overload_name
        # pyrefly: ignore [bad-assignment]
        self.trace_name: str = trace_name
        self.time_range: Interval = Interval(start_us, end_us)
        self.thread: int = thread
        self.fwd_thread: int | None = fwd_thread
        self.kernels: list[Kernel] = []
        self.count: int = 1
        self.cpu_children: list[FunctionEvent] = []
        self.cpu_parent: FunctionEvent | None = None
        # pyrefly: ignore [bad-assignment]
        self.input_shapes: tuple[int, ...] = input_shapes
        # pyrefly: ignore [bad-assignment]
        self.concrete_inputs: list[Any] = concrete_inputs
        # pyrefly: ignore [bad-assignment]
        self.kwinputs: dict[str, Any] = kwinputs
        # pyrefly: ignore [bad-assignment]
        self.stack: list = stack
        self.scope: int = scope
        self.use_device: str | None = use_device
        self.cpu_memory_usage: int = cpu_memory_usage
        self.device_memory_usage: int = device_memory_usage
        self.is_async: bool = is_async
        self.is_remote: bool = is_remote
        self.sequence_nr: int = sequence_nr
        self.device_type: DeviceType = device_type
        self.device_index: int = device_index
        self.device_resource_id: int = (
            thread if device_resource_id is None else device_resource_id
        )
        self.is_legacy: bool = is_legacy
        self.flops: int | None = flops
        self.is_user_annotation: bool | None = is_user_annotation
        self.is_python_function: bool = is_python_function
        self.activity_type: str | None = activity_type
        self.self_cpu_percent = -1
        self.total_cpu_percent = -1
        self.total_device_percent = -1
        self._metadata_json = metadata_json
        self.flow_id: int | None = flow_id
        self.flow_type: int | None = flow_type
        self.flow_start: bool | None = flow_start
        self.external_id: int = external_id
        self.linked_correlation_id: int = linked_correlation_id
        self.metadata: dict[str, Any] | None = typed_metadata
        self.event_metadata: EventMetadata | None = (
            _build_metadata(extra_meta) if extra_meta else None
        )
        # pyrefly: ignore [bad-assignment]
        self.structured_input_shapes: list = structured_input_shapes
        # pyrefly: ignore [bad-assignment]
        self.structured_input_strides: list = structured_input_strides
        # pyrefly: ignore [bad-assignment]
        self.input_dtypes: list[str] = input_dtypes
        self.python_id: int = python_id
        self.python_parent_id: int = python_parent_id
        self.python_module_id: int = python_module_id

    def append_kernel(self, name, device, duration):
        # [FUNCTION DOC] Adds a Kernel record (name, device, duration) to this event. Raises AssertionError if this is not a CPU event.
        if self.device_type != DeviceType.CPU:
            raise AssertionError("Expected device_type to be CPU")
        self.kernels.append(Kernel(name, device, duration))

    def append_cpu_child(self, child):
        # [FUNCTION DOC] Appends a direct CPU child. Raises AssertionError if self or the child is not a CPU event or the child is not a FunctionEvent.
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
        if self.device_type != DeviceType.CPU:
            raise AssertionError("Expected device_type to be CPU")
        if not isinstance(child, FunctionEvent):
            raise AssertionError("Expected child to be a FunctionEvent")
        if child.device_type != DeviceType.CPU:
            raise AssertionError("Expected child device_type to be CPU")
        self.cpu_children.append(child)

    def set_cpu_parent(self, parent):
        # [FUNCTION DOC] Sets the immediate CPU parent. Raises AssertionError if self or the parent is not a CPU event or the parent is not a FunctionEvent.
        """Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
        if self.device_type != DeviceType.CPU:
            raise AssertionError("Expected device_type to be CPU")
        if not isinstance(parent, FunctionEvent):
            raise AssertionError("Expected parent to be a FunctionEvent")
        if parent.device_type != DeviceType.CPU:
            raise AssertionError("Expected parent device_type to be CPU")
        self.cpu_parent = parent

    # Note: async events don't have children, are not used when computing 'self'
    # metrics of other events, have only total cpu time
    @property
    def self_cpu_memory_usage(self):
        # [FUNCTION DOC] Property: CPU memory usage minus that of direct CPU children; zero for async or non-CPU events.
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_memory_usage - sum(
            child.cpu_memory_usage for child in self.cpu_children
        )

    @property
    def self_device_memory_usage(self):
        # [FUNCTION DOC] Property: device memory usage minus that of direct CPU children; zero for async or non-CPU events.
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.device_memory_usage - sum(
            child.device_memory_usage for child in self.cpu_children
        )

    @property
    @deprecated(
        "`metadata_json` is deprecated. Use `event_metadata` instead.",
        category=FutureWarning,
    )
    def metadata_json(self):
        # [FUNCTION DOC] Deprecated property returning the raw metadata JSON string (emits a FutureWarning).
        return self._metadata_json

    @property
    @deprecated(
        "`self_cuda_memory_usage` is deprecated. Use `self_device_memory_usage` instead.",
        category=FutureWarning,
    )
    def self_cuda_memory_usage(self):  # To be deprecated
        # [FUNCTION DOC] Deprecated alias of self_device_memory_usage (emits a FutureWarning).
        return self.self_device_memory_usage

    @property
    def cpu_time_total(self):
        # [FUNCTION DOC] Property: duration of the event if it is a CPU event, else zero.
        if self.device_type == DeviceType.CPU:
            return self.time_range.elapsed_us()
        else:
            return 0

    @property
    def self_cpu_time_total(self):
        # [FUNCTION DOC] Property: CPU duration minus children durations; zero for async or non-CPU events.
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_time_total - sum(
            child.cpu_time_total for child in self.cpu_children
        )

    @property
    def device_time_total(self):
        # [FUNCTION DOC] Property: device time. Zero if async or no device is profiled. For CPU events sums the launched kernels' durations (plus children's totals unless legacy);
        # for device events returns the event duration. Raises AssertionError for an unsupported device type.
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            if not self.is_legacy:
                # account for the kernels in the children ops
                return sum(kinfo.duration for kinfo in self.kernels) + sum(
                    ch.device_time_total for ch in self.cpu_children
                )
            else:
                # each legacy cpu events has a single (fake) kernel
                return sum(kinfo.duration for kinfo in self.kernels)
        else:
            if self.device_type not in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.HPU,
                DeviceType.XPU,
            ]:
                raise AssertionError(
                    f"Expected device_type to be CUDA, PrivateUse1, MTIA, HPU or XPU, but got {self.device_type}"
                )
            return self.time_range.elapsed_us()

    @property
    @deprecated(
        "`cuda_time_total` is deprecated. Use `device_time_total` instead.",
        category=FutureWarning,
    )
    def cuda_time_total(self):  # To be deprecated
        # [FUNCTION DOC] Deprecated alias of device_time_total (emits a FutureWarning).
        return self.device_time_total

    @property
    def self_device_time_total(self):
        # [FUNCTION DOC] Property: device time excluding children for CPU events; the full device time for device events. Zero if async or no device. Raises AssertionError for an unsupported device type.
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            return self.device_time_total - sum(
                child.device_time_total for child in self.cpu_children
            )
        else:
            if self.device_type not in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.HPU,
                DeviceType.XPU,
            ]:
                raise AssertionError(
                    f"Expected device_type to be CUDA, PrivateUse1, MTIA, HPU or XPU, but got {self.device_type}"
                )
            return self.device_time_total

    @property
    @deprecated(
        "`self_cuda_time_total` is deprecated. Use `self_device_time_total` instead.",
        category=FutureWarning,
    )
    def self_cuda_time_total(self):  # To be deprecated
        # [FUNCTION DOC] Deprecated alias of self_device_time_total (emits a FutureWarning).
        return self.self_device_time_total

    @property
    def key(self):
        # [FUNCTION DOC] Property: the grouping key, which is the event name.
        return self.name

    def __repr__(self):
        # [FUNCTION DOC] Returns a single-line description of the event including ids, times, children ids, device, memory and flags.
        device_name = self.use_device
        device_time = self.device_time_str
        device_memory_usage = self.device_memory_usage
        return (
            f"<FunctionEvent id={self.id} name={self.name} overload_name={self.overload_name} "
            f"device_type={self.device_type} node_id={self.node_id} cpu_time={self.cpu_time_str} "
            f"start_us={self.time_range.start} end_us={self.time_range.end} "
            f"cpu_children={str([child.id for child in self.cpu_children])} {device_name}_time={device_time} "
            f"name={self.name} thread={self.thread} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory_usage} "
            f"is_async={self.is_async} is_remote={self.is_remote} seq_nr={self.sequence_nr} is_legacy={self.is_legacy}>"
        )


class FunctionEventAvg(FormattedTimesMixin):
    # [CLASS DOC] Accumulator of statistics over events sharing a key: summed counts, CPU/device times (total and self) and memory, plus fields copied from the first event added
    # (key, node, flags, parent/children, shapes, stack, scope, device type). Built by adding events with add or the in-place add operator.
    """Averaged profiling statistics over multiple FunctionEvent objects.

    FunctionEventAvg aggregates statistics from multiple FunctionEvent objects
    with the same key (typically same operation name). This is useful for getting
    average performance metrics across multiple invocations of the same operation.

    This class is typically created by calling :meth:`EventList.key_averages()` on
    a profiler's event list.

    Attributes:
        key (str): Grouping key for the events (typically operation name).
        count (int): Total number of events aggregated.
        node_id (int): Node identifier for distributed profiling (-1 if not applicable).
        is_async (bool): Whether the operations are asynchronous.
        is_remote (bool): Whether the operations occurred on a remote node.
        use_device (str): Device type being profiled ("cuda", "xpu", etc.).
        cpu_time_total (int): Accumulated total CPU time in microseconds.
        device_time_total (int): Accumulated total device time in microseconds.
        self_cpu_time_total (int): Accumulated self CPU time (excluding children) in microseconds.
        self_device_time_total (int): Accumulated self device time (excluding children) in microseconds.
        input_shapes (List[List[int]]): Input tensor shapes (requires record_shapes=true).
        overload_name (str): Operator overload name (requires _ExperimentalConfig(capture_overload_names=True) set).
        stack (List[str]): Python stack trace where the operation was called (requires with_stack=true).
        scope (int): at::RecordScope identifier (0=forward, 1=backward, etc.).
        cpu_memory_usage (int): Accumulated CPU memory usage in bytes.
        device_memory_usage (int): Accumulated device memory usage in bytes.
        self_cpu_memory_usage (int): Accumulated self CPU memory usage in bytes.
        self_device_memory_usage (int): Accumulated self device memory usage in bytes.
        cpu_children (List[FunctionEvent]): CPU child events.
        cpu_parent (FunctionEvent): CPU parent event.
        device_type (DeviceType): Type of device (CPU, CUDA, XPU, PrivateUse1, etc.).
        is_legacy (bool): Whether from legacy profiler.
        flops (int): Total floating point operations.
        is_user_annotation (bool): Whether this is a user-annotated region.

    Properties:
        cpu_time (float): Average CPU time per invocation.
        device_time (float): Average device time per invocation.

    See Also:
        - :class:`EventList.key_averages`: Method that creates FunctionEventAvg objects
        - :class:`FunctionEvent`: Individual profiling event
        - :class:`EventList`: Container for profiling events
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes an empty accumulator: no key, zero counts, times, memory and flops, CPU device type.
        # KNOWN ISSUE: is_user_annotation is not initialized here; it only exists after the first add.
        self.key: str | None = None
        self.count: int = 0
        self.node_id: int = 0
        self.is_async: bool = False
        self.is_remote: bool = False
        self.use_device: str | None = None
        self.cpu_time_total: int = 0
        self.device_time_total: int = 0
        self.self_cpu_time_total: int = 0
        self.self_device_time_total: int = 0
        self.input_shapes: list[list[int]] | None = None
        self.overload_name: str | None = None
        self.stack: list | None = None
        self.scope: int | None = None
        self.cpu_memory_usage: int = 0
        self.device_memory_usage: int = 0
        self.self_cpu_memory_usage: int = 0
        self.self_device_memory_usage: int = 0
        self.cpu_children: list[FunctionEvent] | None = None
        self.cpu_parent: FunctionEvent | None = None
        self.device_type: DeviceType = DeviceType.CPU
        self.is_legacy: bool = False
        self.flops: int = 0

    def add(self, other):
        # [FUNCTION DOC] Adds another FunctionEvent or FunctionEventAvg. On the first call copies identity fields from it. Raises AssertionError if the type is wrong or keys differ,
        # then sums times, memory, count and flops. Returns self.
        if self.key is None:
            # First function being recorded as part of FunctionEventAvg, propagate
            # fields.
            self.key = other.key
            self.node_id = other.node_id
            self.is_async = other.is_async
            self.is_remote = other.is_remote
            self.cpu_parent = other.cpu_parent
            self.cpu_children = other.cpu_children

            self.overload_name = other.overload_name
            self.input_shapes = other.input_shapes
            self.stack = other.stack
            self.scope = other.scope
            self.device_type = other.device_type
            self.is_legacy = other.is_legacy
            self.use_device = other.use_device
            self.is_user_annotation = other.is_user_annotation

        if not isinstance(other, (FunctionEvent, FunctionEventAvg)):
            raise AssertionError(
                "Expected other to be a FunctionEvent or FunctionEventAvg"
            )
        if other.key != self.key:
            raise AssertionError(
                f"Expected keys to match, but got {other.key} vs {self.key}"
            )

        self.cpu_time_total += other.cpu_time_total
        self.device_time_total += other.device_time_total
        self.self_cpu_time_total += other.self_cpu_time_total
        self.self_device_time_total += other.self_device_time_total
        self.cpu_memory_usage += other.cpu_memory_usage
        self.device_memory_usage += other.device_memory_usage
        self.self_cpu_memory_usage += other.self_cpu_memory_usage
        self.self_device_memory_usage += other.self_device_memory_usage
        self.count += other.count
        if self.flops is None:
            # pyrefly: ignore [bad-assignment]
            self.flops = other.flops
        elif other.flops is not None:
            self.flops += other.flops
        return self

    def __iadd__(self, other):
        # [FUNCTION DOC] In-place add; delegates to add and returns self.
        return self.add(other)

    def __repr__(self):
        device_name = "cuda" if not self.use_device else self.use_device
        self_device_time = self.self_device_time_total_str
        device_time = self.device_time_str
        device_memory = self.device_memory_usage
        return (
            f"<FunctionEventAvg key={self.key} self_cpu_time={self.self_cpu_time_total_str} cpu_time={self.cpu_time_str} "
            f" self_{device_name}_time={self_device_time} {device_name}_time={device_time} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory}>"
        )


class StringTable(defaultdict):
    # [CLASS DOC] Mapping that lazily demangles missing keys (C++ symbol names) via the C++ demangler, caching the result; single-character keys are kept as is.
    def __missing__(self, key):
        # [FUNCTION DOC] Computes, stores and returns the demangled name (or the key itself when it has length one).
        # manage cases like 't' (demangled to 'unsigned short') separately,
        # for now simply check the length to avoid unexpected results for
        # the short sequences
        self[key] = torch._C._demangle(key) if len(key) > 1 else key
        return self[key]


class MemRecordsAcc:
    # [CLASS DOC] Index over memory records sorted by start time enabling interval lookups by binary search. Each record is a pair whose first element has a start time in nanoseconds.
    """Acceleration structure for accessing mem_records in interval."""

    def __init__(self, mem_records):
        # [FUNCTION DOC] Stores the records and builds sorted start-time and index arrays (empty if no records).
        self._mem_records = mem_records
        self._start_nses: list[int] = []
        self._indices: list[int] = []
        if len(mem_records) > 0:
            tmp = sorted([(r[0].start_ns(), i) for i, r in enumerate(mem_records)])
            self._start_nses, self._indices = zip(*tmp)  # type: ignore[assignment]

    def in_interval(self, start_ns, end_ns):
        # [FUNCTION DOC] Generator yielding the records whose start time lies within the inclusive interval, in start-time order.
        r"""
        Return all records in the given interval
        """
        start_idx = bisect.bisect_left(self._start_nses, start_ns)
        end_idx = bisect.bisect_right(self._start_nses, end_ns)
        for i in range(start_idx, end_idx):
            yield self._mem_records[self._indices[i]]


def _filter_stack_entry(entry):
    # [FUNCTION DOC] HELPER: Returns False for stack entries belonging to known autograd or test-harness wrapper frames, True otherwise.
    filtered_entries = [
        ("autograd/__init__", "_make_grads"),
        ("autograd/__init__", "backward"),
        ("torch/tensor", "backward"),
        ("_internal/common_utils", "prof_callable"),
        ("_internal/common_utils", "prof_func_call"),
        ("_internal/common_utils", "prof_meth_call"),
    ]
    return all(not (f[0] in entry and f[1] in entry) for f in filtered_entries)


MEMORY_EVENT_NAME = "[memory]"
OUT_OF_MEMORY_EVENT_NAME = "[OutOfMemory]"


def _filter_name(name):
    # [FUNCTION DOC] HELPER: Returns True if the event name is one of the internal utility ops (memory markers, record-function enter/exit, leaf/output_nr/version queries) that should be skipped.
    # ignoring the following utility ops
    filtered_out_names = [
        MEMORY_EVENT_NAME,  # used only for the top-level memory events
        OUT_OF_MEMORY_EVENT_NAME,
        "profiler::_record_function_enter",
        "profiler::_record_function_enter_new",
        "profiler::_record_function_exit",
        "aten::is_leaf",
        "aten::output_nr",
        "aten::_version",
    ]
    return name in filtered_out_names


# Demangles and optionally rewrites the provided event name,
# with_wildcard - whether to replace certain numbered event names
# with a wildcard name to aggregate them together in the profiler table
# output
def _rewrite_name(name, with_wildcard=False):
    # [FUNCTION DOC] HELPER: Demangles the event name via a StringTable and, if with_wildcard, collapses numbered ProfilerStep names into a single wildcard name. Returns the name.
    string_table = StringTable()
    name = string_table[name]
    if with_wildcard:
        if name.startswith("ProfilerStep#"):
            name = "ProfilerStep*"
    return name


def _build_table(
    events,
    sort_by=None,
    header=None,
    row_limit=100,
    max_src_column_width=75,
    max_name_column_width=55,
    max_shapes_column_width=80,
    with_flops=False,
    profile_memory=False,
    top_level_events_only=False,
    time_unit=None,
):
    # [FUNCTION DOC] HELPER: Renders events (FunctionEvent or FunctionEventAvg) as a fixed-width text table and returns it ('' if there are no events).
    # Optionally sorts descending by an attribute (cuda/xpu/privateuse1 names mapped to device), sizes columns, adds optional columns (overload, device times, memory, node id, shapes, source, flops),
    # limits rows, optionally shows only top-level events, and finishes with total self CPU/device time lines. Also updates percent fields on the events.
    # Raises RuntimeError if device data exists without a device name; AssertionError for non-positive flops scaling input.
    """Print a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    has_device_time = any(event.self_device_time_total > 0 for event in events)
    has_device_mem = any(event.self_device_memory_usage > 0 for event in events)
    use_device = events[0].use_device
    # Running on PrivateUse1 device with profiler but not enable
    # ProfilerActivity.PrivateUse1 can also catch privateuse1 memory usage.
    # Here only need to check has_privateuse1_time if not use_device.
    if not use_device and has_device_time:
        raise RuntimeError("use_device is None, but there is device performance data.")

    has_input_shapes = any(
        (event.input_shapes is not None and len(event.input_shapes) > 0)
        for event in events
    )

    has_overload_names = any(
        (event.overload_name is not None and len(event.overload_name) > 0)
        for event in events
    )

    if sort_by is not None:
        events = EventList(
            sorted(
                events,
                key=lambda evt: getattr(
                    evt,
                    sort_by.replace("cuda", "device")
                    .replace("xpu", "device")
                    .replace("privateuse1", "device"),
                ),
                reverse=True,
            ),
            use_device=use_device,
            profile_memory=profile_memory,
            with_flops=with_flops,
        )

    name_column_width = max(len(evt.key) for evt in events) + 4
    if max_name_column_width is not None:
        name_column_width = min(name_column_width, max_name_column_width)

    shapes_column_width = max(len(str(evt.input_shapes)) for evt in events) + 4
    if max_shapes_column_width is not None:
        shapes_column_width = min(shapes_column_width, max_shapes_column_width)

    DEFAULT_COLUMN_WIDTH = 12
    flops_column_width = DEFAULT_COLUMN_WIDTH

    src_column_width = None
    stacks = [
        evt.stack for evt in events if evt.stack is not None and len(evt.stack) > 0
    ]
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = (
            max(max(len(entry) for entry in stack) for stack in stacks) + 4
        )
        if max_src_column_width is not None:
            src_column_width = min(src_column_width, max_src_column_width)

    headers = ["Name"]
    if has_overload_names:
        headers.append("Overload Name")
    headers += [
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
    ]

    device_name = use_device.upper() if use_device is not None else "None"
    if has_device_time:
        headers.extend(
            [
                f"Self {device_name}",
                f"Self {device_name} %",
                f"{device_name} total",
                f"{device_name} time avg",
            ]
        )
    if profile_memory:
        headers.extend(
            [
                "CPU Mem",
                "Self CPU Mem",
            ]
        )
        if use_device and has_device_mem:
            headers.extend(
                [
                    f"{device_name} Mem",
                    f"Self {device_name} Mem",
                ]
            )
    headers.append("# of Calls")
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any(evt.node_id != -1 for evt in events)
    if append_node_id:
        headers.append("Node ID")

    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]

    def add_column(padding, text_dir=">"):
        # [FUNCTION DOC] HELPER: Appends a column of the given width and alignment to the row format, separator and total line length.
        row_format_lst[0] += (
            "{: " + text_dir + str(padding) + "}" + (" " * SPACING_SIZE)
        )
        header_sep_lst[0] += "-" * padding + (" " * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        # [FUNCTION DOC] HELPER: Chooses a units prefix (FLOPs to PFLOPs) for a positive flops value; returns the scale multiplier and the header label. Raises AssertionError for non-positive input.
        flop_headers = [
            "FLOPs",
            "KFLOPs",
            "MFLOPs",
            "GFLOPs",
            "TFLOPs",
            "PFLOPs",
        ]
        if flops <= 0:
            raise AssertionError(f"Expected flops to be positive, but got {flops}")
        # pyrefly: ignore [no-matching-overload]
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        if not (log_flops >= 0 and log_flops < len(flop_headers)):
            raise AssertionError(
                f"Expected log_flops to be in range [0, {len(flop_headers)}), but got {log_flops}"
            )
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    add_column(name_column_width)
    if has_overload_names:
        add_column(name_column_width)
    for _ in headers[1 + has_overload_names :]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append("Input Shapes")
        add_column(shapes_column_width)

    if has_stack:
        headers.append("Source Location")
        add_column(src_column_width, text_dir="<")

    if with_flops:
        # Auto-scaling of flops header
        raw_flops = [evt.flops for evt in events if evt.flops > 0]
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            headers.append(f"Total {flops_header}")
            add_column(flops_column_width)
        else:
            with_flops = False  # can't find any valid flops

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore[assignment]

    result = []

    def append(s):
        # [FUNCTION DOC] HELPER: Appends the text and a newline to the result buffer.
        result.append(s)
        result.append("\n")  # Yes, newline after the end as well

    sum_self_cpu_time_total = 0
    sum_self_device_time_total = 0
    for evt in events:
        sum_self_cpu_time_total += evt.self_cpu_time_total
        if evt.device_type == DeviceType.CPU and evt.is_legacy:
            # in legacy profiler, kernel info is stored in cpu events
            sum_self_device_time_total += evt.self_device_time_total
        elif (
            evt.device_type
            in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.XPU,
            ]
            and not evt.is_user_annotation
        ):
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_device_time_total += evt.self_device_time_total

    # Actual printing
    if header is not None:
        append("=" * line_length)
        append(header)
    if top_level_events_only:
        append("=" * line_length)
        append("This report only display top-level ops statistics")
    append(header_sep)
    append(row_format.format(*headers))

    append(header_sep)

    def trim_path(path, src_column_width):
        # [FUNCTION DOC] HELPER: Truncates a source path from the left to fit the column width, marking the cut with an ellipsis.
        if len(path) > src_column_width:
            offset = len(path) - src_column_width
            path = path[offset:]
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    def override_time_unit(time_us, default_str, time_unit):
        # [FUNCTION DOC] HELPER: Formats microseconds in the requested unit (s, ms or us), or returns the default string for any other unit.
        US_IN_SECOND = 1000.0 * 1000.0
        US_IN_MS = 1000.0
        if time_unit == "s":
            return f"{time_us / US_IN_SECOND:.3f}s"
        elif time_unit == "ms":
            return f"{time_us / US_IN_MS:.3f}ms"
        elif time_unit == "us":
            return f"{time_us:.3f}us"
        else:
            return default_str

    event_limit = 0
    for evt in events:
        if event_limit == row_limit:
            break
        if top_level_events_only and evt.cpu_parent is not None:
            continue
        else:
            event_limit += 1
        name = evt.key
        if max_name_column_width is not None and len(name) >= max_name_column_width - 3:
            name = name[: (max_name_column_width - 3)] + "..."

        evt.self_cpu_percent = _format_time_share(
            evt.self_cpu_time_total, sum_self_cpu_time_total
        )
        evt.total_cpu_percent = (
            _format_time_share(evt.cpu_time_total, sum_self_cpu_time_total)
            if not evt.is_async
            else 0
        )

        row_values = [name]
        if has_overload_names:
            overload_name = evt.overload_name
            if (
                max_name_column_width is not None
                and len(overload_name) >= max_name_column_width - 3
            ):
                overload_name = overload_name[: (max_name_column_width - 3)] + "..."
            row_values += [overload_name]
        row_values += [
            # Self CPU total %, 0 for async events.
            evt.self_cpu_percent,
            override_time_unit(
                evt.self_cpu_time_total, evt.self_cpu_time_total_str, time_unit
            ),  # Self CPU total
            # CPU total %, 0 for async events.
            evt.total_cpu_percent,
            override_time_unit(
                evt.cpu_time_total, evt.cpu_time_total_str, time_unit
            ),  # CPU total
            override_time_unit(
                evt.cpu_time, evt.cpu_time_str, time_unit
            ),  # CPU time avg
        ]
        if has_device_time:
            evt.total_device_percent = _format_time_share(
                evt.self_device_time_total, sum_self_device_time_total
            )
            row_values.extend(
                [
                    override_time_unit(
                        evt.self_device_time_total,
                        evt.self_device_time_total_str,
                        time_unit,
                    ),
                    # device time total %
                    evt.total_device_percent,
                    override_time_unit(
                        evt.device_time_total, evt.device_time_total_str, time_unit
                    ),
                    override_time_unit(
                        evt.device_time, evt.device_time_str, time_unit
                    ),  # device time avg
                ]
            )
        if profile_memory:
            row_values.extend(
                [
                    # CPU Mem Total
                    _format_memory(evt.cpu_memory_usage),
                    # Self CPU Mem Total
                    _format_memory(evt.self_cpu_memory_usage),
                ]
            )
            if use_device and has_device_mem:
                row_values.extend(
                    [
                        # Device Mem Total
                        _format_memory(evt.device_memory_usage),
                        # Self Device Mem Total
                        _format_memory(evt.self_device_memory_usage),
                    ]
                )
        row_values.append(
            evt.count,  # Number of calls
        )

        if append_node_id:
            row_values.append(evt.node_id)
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:shapes_column_width])
        if with_flops:
            if evt.flops <= 0:
                row_values.append("--")
            else:
                row_values.append(f"{evt.flops * flops_scale:8.3f}")  # type: ignore[possibly-undefined]
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = trim_path(evt.stack[0], src_column_width)
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:]:
                append(
                    row_format.format(
                        *(empty_headers + [trim_path(entry, src_column_width)])
                    )
                )
            empty_headers.append("")
            append(row_format.format(*empty_headers))

    append(header_sep)
    append(
        f"Self CPU time total: {override_time_unit(sum_self_cpu_time_total, _format_time(sum_self_cpu_time_total), time_unit)}"
    )
    if has_device_time:
        append(
            f"Self {use_device.upper() if use_device is not None else 'None'} "
            f"time total: {override_time_unit(sum_self_device_time_total, _format_time(sum_self_device_time_total), time_unit)}"
        )
    return "".join(result)


# Collect all events with stack traces and format them canonically
def _canonicalize_profiler_events(events):
    # [FUNCTION DOC] HELPER: Converts trace event dicts into a deterministic multi-line string of event name (30 chars), node name and the last real source line of the stack trace
    # (caret/tilde marker lines skipped), ordered by start time. KNOWN ISSUE: a comment says sorting is by node name but the code sorts by start time.
    """
    Extract and format all events with stack traces in a canonical way
    for deterministic testing.
    """
    events_with_traces = []

    for event in events:
        # Extract relevant fields
        event_name = event.get("name", "")
        node_name = event["args"].get("node_name", "")
        stack_trace = event["args"].get("stack_trace", "")

        # Get the last non-empty line of the stack trace that is actual source,
        # not a caret-marker line. Python 3.11+ appends "^^^^"/"~~~~" indicator
        # lines below the source when a FrameSummary has colno/end_colno set
        # (e.g. dynamo-generated stack traces); those lines must be skipped so
        # we still surface the source code line in canonicalized output.
        lines = [s.strip() for s in stack_trace.split("\n") if s.strip()]
        source_lines = [s for s in lines if not set(s).issubset({"^", "~", " "})]
        stack_trace = source_lines[-1] if source_lines else ""

        events_with_traces.append(
            {
                "event_name": event_name[:30],
                "node_name": node_name,
                "stack_trace": stack_trace,
                "start_time": event.get("ts", 0),
            }
        )

    # Sort by node_name for deterministic ordering
    events_with_traces.sort(key=lambda x: x["start_time"])

    # Format as a string
    lines: list[str] = []
    for evt in events_with_traces:
        lines.append(
            f"event={evt['event_name']} node={evt['node_name']} stack_trace={evt['stack_trace']}"
        )

    return "\n".join(lines)
