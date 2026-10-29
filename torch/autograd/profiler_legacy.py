# mypy: allow-untyped-defs
# [MODULE DOC] Deprecated legacy autograd profiler (use torch.profiler instead). Exports profile, a context manager over the legacy C++ profiler,
# plus a parser turning raw records into FunctionEvent lists. No import-time side effects.
import itertools
import warnings
from typing_extensions import deprecated

import torch
import torch.cuda
from torch.autograd import (
    _disable_profiler_legacy,
    _enable_profiler_legacy,
    DeviceType,
    ProfilerConfig,
    ProfilerState,
)
from torch.autograd.profiler_util import (
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    EventList,
    FunctionEvent,
    MEMORY_EVENT_NAME,
)


__all__ = ["profile"]


@deprecated(
    "`torch.autograd.profiler_legacy.profile` is deprecated and will be removed in a future release. "
    "Please use `torch.profiler` instead.",
    category=None,  # TODO: change to `FutureWarning`
)
class profile:
    # [CLASS DOC] Deprecated legacy context manager that profiles CPU (and optionally CUDA) ops via the legacy C++ profiler.
    # State: enabled flag, config options (use_cuda, record_shapes, with_flops, profile_memory, with_stack, with_modules), entered flag,
    # profiler_kind and function_events (None until the context exits). Not reentrant. When disabled, most attributes are never set.
    """DEPRECATED: use torch.profiler instead."""

    def __init__(
        self,
        enabled=True,
        *,
        use_cuda=False,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    ):
        # [FUNCTION DOC] Stores options. If enabled is false returns early leaving other attributes unset. with_flops forces record_shapes on.
        # If use_cuda is requested but CUDA is unavailable, warns and falls back to CPU. Chooses ProfilerState.CUDA or CPU as profiler_kind.
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        self.function_events = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules

        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available, disabling CUDA profiling",
                stacklevel=2,
            )
            self.use_cuda = False

        if self.use_cuda:
            self.profiler_kind = ProfilerState.CUDA
        else:
            self.profiler_kind = ProfilerState.CPU

    def config(self):
        # [FUNCTION DOC] Builds and returns a ProfilerConfig from the stored options with a default experimental config.
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            # avoid exposing _ExperimentalConfig this in legacy public API
            torch._C._profiler._ExperimentalConfig(),
        )

    def __enter__(self):
        # [FUNCTION DOC] Starts profiling and returns self (returns None if disabled). Raises RuntimeError if already entered.
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self.entered = True
        self._start_trace()
        return self

    def _start_trace(self):
        # [FUNCTION DOC] HELPER: Enables the legacy C++ profiler with the current config.
        _enable_profiler_legacy(self.config())

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] Stops profiling (synchronizing CUDA first if used), parses the legacy records into events, stores them as an EventList
        # in function_events and builds the event tree. Does nothing if disabled. Returns False so exceptions propagate.
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()

        records = _disable_profiler_legacy()
        parsed_results = _parse_legacy_records(records)
        # pyrefly: ignore [bad-assignment]
        self.function_events = EventList(
            parsed_results,
            use_device="cuda" if self.use_cuda else None,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        # pyrefly: ignore [missing-attribute]
        self.function_events._build_tree()
        return False

    def __repr__(self):
        # [FUNCTION DOC] Returns a placeholder string if profiling is unfinished, else the repr of the event list.
        if self.function_events is None:
            return "<unfinished profiler_legacy.profile>"
        return repr(self.function_events)

    def __str__(self):
        # [FUNCTION DOC] Returns a placeholder string if profiling is unfinished, else the string form of the event list.
        if self.function_events is None:
            return "<unfinished profile.profiler_legacy.profile>"
        return str(self.function_events)

    def _check_finish(self):
        # [FUNCTION DOC] HELPER: Raises RuntimeError if profiling results are not yet available.
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        # [FUNCTION DOC] Checks that profiling finished, then returns the formatted table string from the event list, forwarding all arguments.
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        # [FUNCTION DOC] Checks that profiling finished, then writes a Chrome trace JSON to path via the event list.
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.export_chrome_trace(path)

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        # [FUNCTION DOC] Writes stack-based profile data for the given metric to path. Raises RuntimeError if unfinished, AssertionError if with_stack was not set.
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        if not self.with_stack:
            raise AssertionError("export_stacks() requires with_stack=True")
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        # [FUNCTION DOC] Returns the event list averaged by key, optionally grouped by input shape and stack depth. Raises RuntimeError if unfinished.
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        # [FUNCTION DOC] Returns a single event aggregating all events' averages. Raises RuntimeError if unfinished.
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        # [FUNCTION DOC] Property: sum of self CPU time over all events. Raises RuntimeError if unfinished.
        """Return CPU time as the sum of self times across all events."""
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.self_cpu_time_total


def _parse_legacy_records(thread_records):
    # [FUNCTION DOC] HELPER: Converts per-thread lists of legacy profiler records into a sorted list of FunctionEvent.
    # Finds the '__start_profile' record (AssertionError if missing or remote) as the time origin. For each thread pairs push/pop records by (handle, node_id),
    # skips filtered names and duplicate consecutive pushes, accumulates memory allocation per open range, and emits standalone memory events when no range is open.
    # Raises AssertionError on a pop without push or mismatched memory tables. Sorts by start time then descending end time.
    def _get_record_key(record):
        # [FUNCTION DOC] HELPER: Returns the (handle, node_id) pair used to match push and pop records.
        """Return a tuple for correlating start and end records in `_parse_legacy_records`."""
        return (record.handle(), record.node_id())

    start_record = None
    functions = []

    # '__start_profile' is not guaranteed to be first, so we must find it here
    for record in itertools.chain.from_iterable(thread_records):
        name = record.name()
        if start_record is None and name == "__start_profile":
            start_record = record

    if start_record is None or start_record.is_remote():
        raise AssertionError("Expected a valid local start_record")

    for thread_record_list in thread_records:
        # accumulated memory allocations per handle
        cpu_memory_allocs = {}
        cuda_memory_allocs = {}
        # ranges per handle
        range_starts = {}

        filtered_handles = set()
        prev_record = None
        for record in thread_record_list:
            record_key = _get_record_key(record)
            if _filter_name(record.name()) or record_key in filtered_handles:
                filtered_handles.add(record_key)
                continue

            if record.kind() == "push":
                # workaround to reduce double logging from operator
                # wrappers and redispatch
                if prev_record is not None:
                    duplicate = (
                        prev_record.name() == record.name()
                        and prev_record.kind() == record.kind()
                        and prev_record.node_id() == record.node_id()
                    )
                    if duplicate:
                        filtered_handles.add(record_key)
                        continue

                range_starts[record_key] = record
                cpu_memory_allocs[record_key] = 0
                cuda_memory_allocs[record_key] = 0
            elif record.kind() == "pop":
                if record_key not in range_starts:
                    raise AssertionError(
                        f"Expected record with key {record_key} to exist in range_starts. "
                        "This means that the pop event did not have a corresponding push."
                    )

                start = range_starts[record_key]

                cpu_memory_usage = cpu_memory_allocs[record_key]
                cuda_memory_usage = cuda_memory_allocs[record_key]
                is_async = start.is_async() or (start.thread_id() != record.thread_id())
                is_remote_event = record.is_remote()
                start_flops = start.flops()

                fe = FunctionEvent(
                    id=record.handle(),
                    node_id=record.node_id(),
                    name=_rewrite_name(name=start.name(), with_wildcard=True),
                    trace_name=_rewrite_name(name=start.name(), with_wildcard=False),
                    thread=start.thread_id(),
                    start_us=start_record.cpu_elapsed_us(start),
                    end_us=start_record.cpu_elapsed_us(record),
                    fwd_thread=start.fwd_thread_id(),
                    input_shapes=start.shapes(),
                    stack=[
                        entry for entry in start.stack() if _filter_stack_entry(entry)
                    ],
                    scope=start.scope(),
                    use_device="cuda" if start.has_cuda() else None,
                    cpu_memory_usage=cpu_memory_usage,
                    device_memory_usage=cuda_memory_usage,
                    is_async=is_async,
                    is_remote=is_remote_event,
                    sequence_nr=start.sequence_nr(),
                    device_type=DeviceType.CPU,
                    is_legacy=True,
                    flops=start_flops,
                )
                # note: async events have only cpu total time
                if not is_async and start.has_cuda():
                    duration = start.cuda_elapsed_us(record)
                    if duration > 0:
                        fe.append_kernel(start.name(), start.device(), duration)
                functions.append(fe)
                del range_starts[record_key]
                del cpu_memory_allocs[record_key]
                del cuda_memory_allocs[record_key]
            elif record.kind() == "memory_alloc":
                num_open_handles_cpu = len(cpu_memory_allocs)
                num_open_handles_cuda = len(cuda_memory_allocs)
                if num_open_handles_cpu != num_open_handles_cuda:
                    raise AssertionError(
                        f"Expected CPU and CUDA memory allocation handles to match, "
                        f"but got {num_open_handles_cpu} CPU and {num_open_handles_cuda} CUDA"
                    )
                for handle in cpu_memory_allocs:
                    cpu_memory_allocs[handle] += record.cpu_memory_usage()
                for handle in cuda_memory_allocs:
                    cuda_memory_allocs[handle] += record.cuda_memory_usage()
                if num_open_handles_cpu == 0:
                    # output event as a top-level memory event
                    fe = FunctionEvent(
                        id=0,
                        name=MEMORY_EVENT_NAME,
                        trace_name=None,
                        thread=0,
                        start_us=0,
                        end_us=0,
                        stack=[],
                        cpu_memory_usage=record.cpu_memory_usage(),
                        device_memory_usage=record.cuda_memory_usage(),
                        is_legacy=True,
                    )
                    functions.append(fe)
            prev_record = record

    # Sort functions by start time then by end time ascending.
    # This ensures that--in the case of nested events which
    # have the same start time (which may happen due to the
    # granularity of the given clock tick)--we always show
    # the outermost nested call first. This adds stability
    # in how FunctionEvents appear
    functions.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])
    return functions
