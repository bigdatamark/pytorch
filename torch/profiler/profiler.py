# mypy: allow-untyped-defs
# [MODULE DOC] Public profiler API built on the autograd/Kineto profiler: the profile context
# manager (with its low-level base class), the step schedule factory and action enum, the
# tensorboard trace handler, supported_activities, and ExecutionTraceObserver for recording
# PyTorch execution traces. Also contains an optional CUPTI-monitor backend selected through
# the experimental custom profiler config. Import-time side effect: creates a module-level
# set used to deduplicate one-time warnings.
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, TYPE_CHECKING
from typing_extensions import deprecated, Self
from warnings import warn

import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
    _add_execution_trace_observer,
    _disable_execution_trace_observer,
    _enable_execution_trace_observer,
    _ExperimentalConfig,
    _remove_execution_trace_observer,
)
from torch._environment import is_fbcode
from torch._utils_internal import profiler_allow_cudagraph_cupti_lazy_reinit_cuda12
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = [
    "supported_activities",
    "ProfilerAction",
    "schedule",
    "tensorboard_trace_handler",
    "profile",
    "ExecutionTraceObserver",
]
PROFILER_STEP_NAME = "ProfilerStep"

_WARNINGS_SHOWN = set()


def _warn_once(msg, category=UserWarning, stacklevel=2):
    # [FUNCTION DOC] HELPER: Emits the warning with the given category and stacklevel only
    # the first time this exact message is seen, tracking messages in a module-level set.
    if msg not in _WARNINGS_SHOWN:
        _WARNINGS_SHOWN.add(msg)
        warn(msg, category=category, stacklevel=stacklevel)


def _parse_custom_profiler_config(
    experimental_config: _ExperimentalConfig | None,
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Reads the custom_profiler_config string of the experimental
    # config and returns it as a mapping. Returns an empty mapping when the config or string
    # is missing/empty or parses to a non-mapping; if the string is not valid JSON it is
    # treated as a bare backend name and returned as a mapping with a single backend entry.
    if experimental_config is None:
        return {}
    raw = getattr(experimental_config, "custom_profiler_config", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"backend": raw}
    return parsed if isinstance(parsed, dict) else {}


class _NumpyEncoder(json.JSONEncoder):
    # [CLASS DOC] HELPER: JSON encoder subclass that serializes NumPy integers, floats and
    # arrays as plain ints, floats and lists; defers to the standard encoder (which raises
    # for unsupported types) for everything else or when NumPy is not installed.
    """
    Json encoder for numpy types (np.int, np.float, np.array etc.)
    Returns default encoder if numpy is not available
    """

    def default(self, obj):
        # [FUNCTION DOC] HELPER: Converts a NumPy scalar or array to a JSON-serializable
        # Python value; otherwise delegates to the base encoder, which raises TypeError.
        """Encode NumPy types to JSON"""
        try:
            import numpy as np
        except ImportError:
            return json.JSONEncoder.default(self, obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


def supported_activities():
    # [FUNCTION DOC] Returns the set of ProfilerActivity values this build can trace, as
    # reported by the autograd engine. Takes no arguments and has no side effects.
    """
    Returns a set of supported profiler tracing activities.

    Note: profiler uses CUPTI library to trace on-device CUDA kernels.
    In case when CUDA is enabled but CUPTI is not available, passing
    ``ProfilerActivity.CUDA`` to profiler results in using the legacy CUDA
    profiling code (same as in the legacy ``torch.autograd.profiler``).
    This, in turn, results in including CUDA time in the profiler table output,
    but not in the JSON trace.
    """
    return torch.autograd._supported_activities()


class _ITraceObserver(ABC):
    # [CLASS DOC] HELPER: Abstract interface for execution trace observers used by the
    # profiler; implementations must provide start, stop and cleanup.
    """Abstract interface for a Trace observer.
    This satisfies 3 methods: start, stop and cleanup"""

    @abstractmethod
    def start(self):
        # [FUNCTION DOC] HELPER: Abstract; begin observing.
        pass

    @abstractmethod
    def stop(self):
        # [FUNCTION DOC] HELPER: Abstract; stop observing.
        pass

    @abstractmethod
    def cleanup(self):
        # [FUNCTION DOC] HELPER: Abstract; release resources held by the observer.
        pass


def _parse_activities(
    activities: Iterable[ProfilerActivity | dict[ProfilerActivity, list[str]]],
) -> tuple[set[ProfilerActivity], dict[ProfilerActivity, set[str]]]:
    # [FUNCTION DOC] HELPER: Splits a mixed list of activities into a set of activity groups
    # and a mapping from group to the set of named sub-types to collect (only for mapping
    # items). Raises ValueError if a group appears more than once and TypeError for items
    # that are neither an activity nor a mapping. Returns both containers.
    """Parse a mixed activities list into a set of activities and a filter dict.

    Each item is either a bare ``ProfilerActivity`` (collect all defaults) or a
    ``dict[ProfilerActivity, list[str]]`` (collect only the named subset).
    An empty list value (e.g. ``{CUDA: []}``) means collect nothing for that group.
    """
    parsed_activities: set[ProfilerActivity] = set()
    activity_filters: dict[ProfilerActivity, set[str]] = {}
    for item in activities:
        if isinstance(item, ProfilerActivity):
            if item in parsed_activities:
                raise ValueError(f"Activity {item} specified more than once")
            parsed_activities.add(item)
        elif isinstance(item, dict):
            for key, val in item.items():
                if key in parsed_activities:
                    raise ValueError(f"Activity {key} specified more than once")
                parsed_activities.add(key)
                activity_filters[key] = set(val)
        else:
            raise TypeError(f"Expected ProfilerActivity or dict, got {type(item)}")
    return parsed_activities, activity_filters


class _KinetoProfile:
    # [CLASS DOC] Low-level wrapper around the autograd profiler with Kineto enabled. Holds
    # the configuration (activities and per-activity filters, shape/memory/stack/flops/module
    # flags, experimental config, execution trace observer, accumulation and timeout
    # options), the underlying autograd profiler once prepared, user metadata, and optional
    # CUPTI-monitor state (observer, window id, async export flag). Lifecycle: prepare_trace,
    # start_trace, stop_trace; then export or inspect results. Base class of profile.
    """Low-level profiler wrap the autograd profile

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.

            Each item can be a ``ProfilerActivity`` enum (collects all default
            activity types for that group) or a ``dict`` mapping a ``ProfilerActivity``
            to a list of individual activity type names to collect, e.g.
            ``{ProfilerActivity.CUDA: ["GPU_MEMCPY", "CUDA_RUNTIME"]}``.
            An empty list (e.g. ``{ProfilerActivity.CUDA: []}``) means collect
            nothing for that group.
            The same activity group must not appear more than once.
            See :class:`~torch.profiler.ProfilerActivity` for the valid
            activity type names and device-specific behavior.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation (see ``export_memory_timeline``
            for more details).
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPS of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B

            .. deprecated::
                ``with_modules`` is deprecated and will be removed in a future version.
                It only collects data for TorchScript models, which are themselves
                deprecated, and does nothing in eager mode. Use ``with_stack=True``,
                which records ``nn.Module`` events for eager models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
        post_processing_timeout_s (float): Optional timeout in seconds for post-processing profiler
            results. In this context, post-processing happens after the profiling itself has finished.
            If specified, event parsing will stop after this duration and return partial results. Useful
            for handling large traces that may take too long to process.


    .. note::
        This API is experimental and subject to change in the future.

        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.
    """

    def __init__(
        self,
        *,
        activities: Iterable[ProfilerActivity | dict[ProfilerActivity, list[str]]]
        | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: _ExperimentalConfig | None = None,
        execution_trace_observer: _ITraceObserver | None = None,
        acc_events: bool = False,
        custom_trace_id_callback: Callable[[], str] | None = None,
        post_processing_timeout_s: float | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores all options. Activities default to the supported set; given
        # activities are parsed into groups plus filters (errors propagate). If trace_only is
        # combined with stack recording, warns and uses a copy of the config with trace_only
        # cleared. Picks use_device from the first present accelerator group (CUDA, XPU, MTIA,
        # HPU, privateuse1). Parses the custom profiler config; for the cupti_monitor backend
        # requires CPU activity (ValueError otherwise), reads the async-export flag, and arms
        # graph-dependency / event-node recorders when configured. Raises ValueError if the
        # async-export option is given without that backend.
        if activities is not None:
            self.activities, self.activity_filters = _parse_activities(activities)
        else:
            self.activities = supported_activities()
            self.activity_filters: dict[ProfilerActivity, set[str]] = {}
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        if (
            experimental_config is not None
            and experimental_config.trace_only
            and with_stack
        ):
            import copy

            warn(
                "trace_only=True is incompatible with with_stack=True "
                "(stack traces require event post-processing). "
                "Disabling trace_only."
            )
            experimental_config_copy: _ExperimentalConfig = copy.copy(
                experimental_config
            )
            experimental_config_copy.trace_only = False
            experimental_config = experimental_config_copy
        self.experimental_config = experimental_config
        self.execution_trace_observer = execution_trace_observer
        self.acc_events = acc_events
        self.custom_trace_id_callback = custom_trace_id_callback
        self.post_processing_timeout_s = post_processing_timeout_s
        self.profiler: prof.profile | None = None
        self.has_cudagraphs = False
        self.mem_tl: MemoryProfileTimeline | None = None
        self.use_device = None
        if ProfilerActivity.CUDA in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "cuda"
        elif ProfilerActivity.XPU in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "xpu"
        elif ProfilerActivity.MTIA in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "mtia"
        elif ProfilerActivity.HPU in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "hpu"
        elif ProfilerActivity.PrivateUse1 in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = _get_privateuse1_backend_name()

        # user-defined metadata to be amended to the trace
        self.preset_metadata: dict[str, str] = {}
        self._trace_metadata: dict[str, str] = {}
        self._custom_profiler_config = _parse_custom_profiler_config(
            self.experimental_config
        )
        self._use_cupti_monitor = self._custom_profiler_config.get(
            "backend"
        ) == "cupti_monitor" or bool(self._custom_profiler_config.get("cupti_monitor"))
        # The ProfilerObserver driving the shared CUPTI monitor this session; window opened
        # at start, closed at stop (_monitor_window_id), exported by export_chrome_trace.
        self._cupti_profiler_observer: Any = None
        self._monitor_window_id: int | None = None
        # cupti_monitor exports synchronously by default (like the stock profiler).
        # {"cupti_monitor_async_export": true} hands the merge+write off-thread, joined by
        # wait_for_exports; cupti_monitor-only, rejected elsewhere.
        self._cupti_async_export = False
        if self._use_cupti_monitor:
            if ProfilerActivity.CPU not in self.activities:
                raise ValueError(
                    "cupti_monitor profiler backend currently requires CPU activity"
                )
            self._cupti_async_export = bool(
                self._custom_profiler_config.get("cupti_monitor_async_export", False)
            )
            # Arm graph-dependency recording now, at profiler construction -- before the
            # training loop captures its CUDA graphs. The recording hook must observe each
            # graph's one-time instantiate(); the per-window ProfilerObserver registers it
            # too late (at prepare_trace, after warm-up capture) to catch replay-only graphs.
            if self._custom_profiler_config.get("enable_graph_dependencies"):
                from torch.profiler._cupti._graph_deps import _GraphDependencyRecorder

                _GraphDependencyRecorder().arm()
            # Same early-arm rationale for the CUDA_EVENT -> graph event-record node bridge:
            # the recorder reads each graph's event nodes at its one-time instantiate().
            if self._custom_profiler_config.get("enable_event_node_ids"):
                from torch.profiler._cupti._event_nodes import _EventNodeRecorder

                _EventNodeRecorder().arm()
        elif "cupti_monitor_async_export" in self._custom_profiler_config:
            raise ValueError(
                "cupti_monitor_async_export is only supported with the cupti_monitor "
                "backend"
            )

    def start(self) -> None:
        # [FUNCTION DOC] Prepares and then starts tracing in one call.
        self.prepare_trace()
        self.start_trace()

    def stop(self) -> None:
        # [FUNCTION DOC] Stops tracing (alias for stop_trace).
        self.stop_trace()

    def prepare_trace(self) -> None:
        # [FUNCTION DOC] Records whether inductor cudagraphs are enabled, warns once if a
        # previous profiler exists and events are not accumulated, and creates a fresh
        # autograd profiler from the stored options (unless accumulating across cycles). For
        # the cupti_monitor backend also creates its observer (starting collection with the
        # configured options) and publishes it as the active observer. Finally prepares the
        # underlying trace. Side effects: global active-observer state.
        # KNOWN ISSUE: sets a trace-window attribute that is never read or declared in init.
        if hasattr(torch, "_inductor"):
            import torch._inductor.config as inductor_config

            self.has_cudagraphs = inductor_config.triton.cudagraphs
        if (self.profiler is not None) and (not self.acc_events):
            _warn_once(
                "Warning: Profiler clears events at the end of each cycle. "
                "Only events from the current cycle will be reported. "
                "To keep events across cycles, set acc_events=True."
            )
        if (self.profiler is None) or (not self.acc_events):
            use_device = None if self._use_cupti_monitor else self.use_device
            self.profiler = prof.profile(
                use_cpu=(ProfilerActivity.CPU in self.activities),
                use_device=use_device,
                record_shapes=self.record_shapes,
                with_flops=self.with_flops,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_modules=self.with_modules,
                use_kineto=True,
                experimental_config=self.experimental_config,
                acc_events=self.acc_events,
                custom_trace_id_callback=self.custom_trace_id_callback,
                post_processing_timeout_s=self.post_processing_timeout_s,
                activity_filters=self.activity_filters
                if self.activity_filters
                else None,
            )
        if self._use_cupti_monitor:
            from torch.profiler._cupti.observers.profiler import ProfilerObserver

            self._monitor_trace_window = None
            # Constructing the observer registers it with the shared monitor and starts
            # collection. cuda_sync events are opt-in via the config, matching kineto's flag.
            self._cupti_profiler_observer = ProfilerObserver(
                enable_cuda_sync=bool(
                    self._custom_profiler_config.get("enable_cuda_sync_events")
                ),
                # GPU environment counters (power/clock/thermal/cooling) are periodically
                # sampled; opt-in since the sampling adds overhead.
                enable_environment_counters=bool(
                    self._custom_profiler_config.get("enable_environment_counters")
                ),
                # PM sampling (true SM-active % + DRAM-throughput % counters) is a CUPTI-monitor
                # feature, opt-in like the env counters. The metrics are per-profile
                # (custom_profiler_config["pm_metrics"], a list of CUPTI metric names).
                enable_pm_sampling=bool(
                    self._custom_profiler_config.get("enable_pm_sampling")
                ),
                pm_metrics=self._custom_profiler_config.get("pm_metrics"),
                # Node->node CUDA-graph dependency arrows are opt-in (extra work at graph
                # instantiate + arrow rendering); off unless the config requests them.
                enable_graph_dependencies=bool(
                    self._custom_profiler_config.get("enable_graph_dependencies")
                ),
                # Join CUDA_EVENT records (graph event-record nodes, e.g. NCCL under
                # NCCL_GRAPH_MIXING_SUPPORT) back to their graph_node_id. Opt-in: pulls in the
                # CUDA_EVENT record kind.
                enable_event_node_ids=bool(
                    self._custom_profiler_config.get("enable_event_node_ids")
                ),
                # gzip level for the native .pftrace encoder (0-9; 1 = fast, the default).
                pftrace_compression_level=int(
                    self._custom_profiler_config.get("pftrace_compression_level", 1)
                ),
                # Synchronous export finalizes on the calling thread, so skip the poll thread.
                defer_export=self._cupti_async_export,
            )
            # Publish the observer so record_function routes annotations to it. The reference
            # lives in torch.autograd (not the cupti package), so record_function never
            # imports the cupti chain on a non-cupti run.
            prof._set_active_cupti_profiler_observer(self._cupti_profiler_observer)
        self.profiler._prepare_trace()

    def start_trace(self) -> None:
        # [FUNCTION DOC] Starts the execution trace observer if any, then the underlying
        # trace (AssertionError if the profiler was never prepared) and opens the cupti window
        # when that backend is active. Writes metadata flags for enabled options, distributed
        # info when Kineto is available and a process group exists, and, with inductor
        # cudagraphs on old CUDA or when not allowed, sets environment variables disabling
        # CUPTI lazy re-init and teardown. Finally writes preset user metadata to the trace.
        # Side effects: mutates the process environment and global trace metadata.
        if self.execution_trace_observer:
            self.execution_trace_observer.start()
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before starting trace")
        self.profiler._start_trace()
        if self._use_cupti_monitor and self._cupti_profiler_observer is not None:
            # Open the trace window here (stamps the start boundary, native clock, no
            # device sync); records before this are excluded from the window.
            self._cupti_profiler_observer.open_window()

        if self.profile_memory:
            self.add_metadata_json("profile_memory", "1")
        if self.with_stack:
            self.add_metadata_json("with_stack", "1")
        if self.record_shapes:
            self.add_metadata_json("record_shapes", "1")
        if self.with_modules:
            self.add_metadata_json("with_modules", "1")
        if self.with_flops:
            self.add_metadata_json("with_flops", "1")

        if kineto_available():
            dist_info = self._get_distributed_info()
            if dist_info:
                self.add_metadata_json(
                    "distributedInfo", json.dumps(dist_info, cls=_NumpyEncoder)
                )

            cuda_version = None
            if hasattr(torch, "version"):
                from torch.torch_version import TorchVersion

                cuda_version = TorchVersion(getattr(torch.version, "cuda", "0.0"))

            if self.has_cudagraphs and (
                (cuda_version and cuda_version < "12.6")
                or not profiler_allow_cudagraph_cupti_lazy_reinit_cuda12()
            ):
                os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
                self.add_metadata_json("DISABLE_CUPTI_LAZY_REINIT", "1")
                # FIXME: CUDA Graph does not work well with CUPTI teardown.
                #   1) crashes on 1st lazy CUPTI re-init after teardown (CUDA 11)
                #   2) crashes on 2nd non-lazy CUPTI re-init after teardown (CUDA 12)
                # Workaround: turn off CUPTI teardown when using CUDA Graphs.
                os.environ["TEARDOWN_CUPTI"] = "0"

            # Insert the preset user metadata to the trace
            for k, v in self.preset_metadata.items():
                self.add_metadata_json(k, v)

    def stop_trace(self) -> None:
        # [FUNCTION DOC] Stops the execution trace observer if any, raises AssertionError if
        # the profiler was never prepared, and for the cupti backend unpublishes the active
        # observer and closes the trace window (keeping the observer for export). Then exits
        # the underlying autograd profiler, finishing collection.
        if self.execution_trace_observer:
            self.execution_trace_observer.stop()
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before stopping trace")
        if self._use_cupti_monitor:
            # Unpublish the observer (record_function stops routing here) and close the trace
            # window (end boundary, native clock, no device sync), queuing it for deferred
            # export; the observer is kept alive past stop for the async write.
            prof._set_active_cupti_profiler_observer(None)
            if self._cupti_profiler_observer is not None:
                self._monitor_window_id = self._cupti_profiler_observer.close_window()
        self.profiler.__exit__(None, None, None)

    def export_chrome_trace(self, path: str, use_python_export: bool = False):
        # [FUNCTION DOC] Writes the collected trace as Chrome JSON to path (gzipped when the
        # path ends with .gz; optional Python-based exporter). Raises AssertionError if never
        # prepared. For the cupti backend: if the observer or window is unavailable, warns
        # once, cleans up and returns without writing; otherwise exports the CPU side to a
        # temporary file and registers it with the observer, which merges and writes path
        # (immediately when synchronous, later when async export is on). Includes user trace
        # metadata.
        # KNOWN ISSUE: the cupti branch leaves its temporary file undeleted (delete is off).
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        if self.profiler is None:
            raise AssertionError(
                "Profiler must be initialized before exporting chrome trace"
            )
        if self._use_cupti_monitor:
            obs = self._cupti_profiler_observer
            if obs is None or not obs.available or self._monitor_window_id is None:
                # Nothing to export this cycle: the per-cycle ProfilerObserver didn't register
                # with the CUPTI monitor (available is False -- the intermittent case), or its
                # window wasn't opened/closed (window id None). Skip rather than crash -- a
                # profiler-trace hiccup must not take down a training run -- and clean up below.
                _warn_once(
                    "CUPTI monitor observer unavailable; skipping chrome trace export"
                )
                # join() tears down the poll thread + monitor registration, which exist only
                # when the observer registered (available). An unavailable observer never
                # started either, so just drop the reference and let it be GC'd.
                if obs is not None and obs.available:
                    obs.join()
                self._cupti_profiler_observer = None
                self._monitor_window_id = None
                return
            # Capture the profiler's CPU-side trace (cheap, no device sync) and hand it + the
            # output path to the observer. Async: the poller merges + writes `path` once the
            # GPU records arrive; wait_for_exports() blocks for it.
            fp = tempfile.NamedTemporaryFile(  # noqa: SIM115
                "w+t", suffix=".json", delete=False
            )
            fp.close()
            self.profiler.export_chrome_trace(fp.name, self._trace_metadata)
            self._cupti_profiler_observer.set_export(
                self._monitor_window_id, fp.name, path
            )
            if not self._cupti_async_export:
                # Synchronous: finalize + write now, so `path` exists on return.
                self._cupti_profiler_observer.join()
                self._cupti_profiler_observer = None
                self._monitor_window_id = None
            return
        if use_python_export:
            self.profiler.export_chrome_trace(
                path, self._trace_metadata, use_python_export=True
            )
        elif path.endswith(".gz"):
            with tempfile.NamedTemporaryFile("w+b", suffix=".json") as fp:
                self.profiler.export_chrome_trace(fp.name, self._trace_metadata)
                with open(fp.name, "rb") as fin, gzip.open(path, "wb") as fout:
                    fout.writelines(fin)
        else:
            self.profiler.export_chrome_trace(path, self._trace_metadata)

    def wait_for_exports(self) -> None:
        # [FUNCTION DOC] For the cupti_monitor backend, blocks until the pending observer has
        # written its exports, then drops the observer and window id. Otherwise does nothing.
        """Block until every deferred cupti_monitor export is written, then unregister.
        No-op unless the cupti_monitor backend is active. Call on the training thread when
        you need the file(s) on disk; the finalize force-flushes CUPTI (safe here)."""
        if self._use_cupti_monitor and self._cupti_profiler_observer is not None:
            self._cupti_profiler_observer.join()
            self._cupti_profiler_observer = None
            self._monitor_window_id = None

    def take_pending_cupti_export(self) -> Any:
        # [FUNCTION DOC] Hands the caller the current cupti observer (with its unwritten
        # window) and clears this profiler's references so the export can be finished on a
        # worker thread. Returns None for non-cupti backends or when none is pending.
        """Detach this cycle's cupti_monitor ProfilerObserver (with its unwritten window) so
        the deferred export can be finalized OFF the training thread (call
        ``obs.join(force=False)`` on a worker). None for non-cupti backends."""
        if not self._use_cupti_monitor:
            return None
        obs = self._cupti_profiler_observer
        self._cupti_profiler_observer = None
        self._monitor_window_id = None
        return obs

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        # [FUNCTION DOC] Writes stack traces with the chosen time metric to the file at path
        # via the underlying profiler. Raises AssertionError if never prepared.
        """Save stack traces to a file

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_cuda_time_total"
        """
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before exporting stacks")
        return self.profiler.export_stacks(path, metric)

    def toggle_collection_dynamic(
        self, enable: bool, activities: Iterable[ProfilerActivity]
    ) -> None:
        # [FUNCTION DOC] Turns collection of the given activity groups on or off while a
        # trace is running by forwarding to the underlying profiler; silently does nothing
        # if no profiler exists yet.
        """Toggle collection of activities on/off at any point of collection. Currently supports toggling Torch Ops
        (CPU) and CUDA activity supported in Kineto

        Args:
            activities (iterable): list of activity groups to use in profiling, supported values:
                ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``
        Examples:

        .. code-block:: python

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                code_to_profile_0()
                // turn off collection of all CUDA activity
                p.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_1()
                // turn on collection of all CUDA activity
                p.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_2()
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
        """
        if self.profiler is None:
            return
        self.profiler.toggle_collection_dynamic(enable, activities)

    def key_averages(
        self,
        group_by_input_shape: bool = False,
        group_by_stack_n: int = 0,
        group_by_overload_name: bool = False,
        include_python_functions: bool = False,
    ):
        # [FUNCTION DOC] Returns the aggregated event list from the underlying profiler,
        # grouped by operator name and optionally input shape, stack depth and overload name;
        # may include Python function events. Raises AssertionError if never prepared.
        """Averages events, grouping them by operator name and (optionally) input shapes, stack
        and overload name.

        Returns an :class:`~torch.autograd.profiler_util.EventList` of the aggregated events.

        .. note::
            To use shape/stack functionality make sure to set record_shapes/with_stack
            when creating profiler context manager.
        """
        if self.profiler is None:
            raise AssertionError(
                "Profiler must be initialized before getting key averages"
            )
        return self.profiler.key_averages(
            group_by_input_shape,
            group_by_stack_n,
            group_by_overload_name,
            include_python_functions,
        )

    def events(self):
        # [FUNCTION DOC] Returns the unaggregated function events of the underlying profiler.
        # Raises AssertionError if never prepared and RuntimeError when the experimental
        # config has trace_only enabled (events are not parsed in that mode).
        """
        Return the list of unaggregated :class:`~torch.autograd.profiler_util.FunctionEvent`
        objects, for use in the trace callback or after profiling has finished.
        """
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before accessing events")
        if self.experimental_config is not None and self.experimental_config.trace_only:
            raise RuntimeError(
                "events() is not available when trace_only=True in "
                "ExperimentalConfig. Use export_chrome_trace() instead."
            )
        return self.profiler.function_events

    def add_metadata(self, key: str, value: str) -> None:
        # [FUNCTION DOC] Adds a string metadata entry to the trace: wraps the value in double
        # quotes with embedded quotes escaped, records it in the local metadata mapping and
        # sends it to the global trace metadata. Other special characters are not escaped.
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
        wrapped_value = '"' + value.replace('"', '\\"') + '"'
        self._trace_metadata[key] = wrapped_value
        torch.autograd._add_metadata_json(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str) -> None:
        # [FUNCTION DOC] Adds a metadata entry whose value is already valid JSON text, storing
        # it locally and sending it to the global trace metadata unchanged.
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
        self._trace_metadata[key] = value
        torch.autograd._add_metadata_json(key, value)

    def preset_metadata_json(self, key: str, value: str) -> None:
        # [FUNCTION DOC] Remembers a JSON-valued metadata entry to be written to the trace
        # when tracing starts; use when the profiler is not running yet.
        """
        Preset a user defined metadata when the profiler is not started
        and added into the trace file later.
        Metadata is in the format of a string key and a valid json value
        """
        self.preset_metadata[key] = value

    def _get_distributed_info(self):
        # [FUNCTION DOC] HELPER: Returns a mapping with backend, rank, world size, process
        # group count and group configs (plus NCCL version for the nccl backend), or None if
        # distributed is unavailable or not initialized.
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None

        backend = dist.get_backend()
        dist_info = {
            "backend": backend,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "pg_count": dist.get_pg_count(),
            "pg_config": dist.distributed_c10d._get_all_pg_configs(),
        }
        if backend == "nccl":
            nccl_version = torch.cuda.nccl.version()
            # pyrefly: ignore [bad-typed-dict-key, unsupported-operation]
            dist_info["nccl_version"] = ".".join(str(v) for v in nccl_version)
        return dist_info

    def _memory_profile(self) -> MemoryProfile:
        # [FUNCTION DOC] HELPER: Builds a memory profile from the Kineto results. Raises
        # ValueError naming any of record_shapes, profile_memory, with_stack that were not
        # enabled, and AssertionError if there are no Kineto results.
        required = ("record_shapes", "profile_memory", "with_stack")
        missing = [f"{i}=True" for i in required if not getattr(self, i)]
        if missing:
            raise ValueError(f"{', '.join(missing)} required for memory profiling.")

        if self.profiler is None or self.profiler.kineto_results is None:
            raise AssertionError(
                "Profiler and kineto_results must be initialized for memory profiling"
            )
        return MemoryProfile(self.profiler.kineto_results)

    @deprecated(
        "`export_memory_timeline` is deprecated and will be removed in a future version. "
        "Please use `torch.cuda.memory._record_memory_history` and `torch.cuda.memory._export_memory_snapshot` instead.",
        category=FutureWarning,
    )
    def export_memory_timeline(self, path: str, device: str | None = None) -> None:
        # [FUNCTION DOC] Deprecated (emits a FutureWarning). Builds a memory timeline from the
        # profile and writes it for the device (default: the profiler's non-cuda device at
        # index 0, else cuda:0 if available, else cpu). Output format follows the path suffix:
        # html (embedded plot), raw json gz (raw events), other gz (gzipped JSON), else JSON.
        # Stores the timeline on the instance. Raises as _memory_profile does.
        """Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.

        .. deprecated::
            ``export_memory_timeline`` is deprecated and will be removed in a future version.
            Please use ``torch.cuda.memory._record_memory_history`` and
            ``torch.cuda.memory._export_memory_snapshot`` instead.
        """
        # Default to device 0, if unset. Fallback on cpu.
        if device is None:
            if self.use_device and self.use_device != "cuda":
                device = self.use_device + ":0"
            else:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Construct the memory timeline plot data
        self.mem_tl = MemoryProfileTimeline(self._memory_profile())

        # Depending on the file suffix, save the data as json.gz or json.
        # For html, we can embed the image into an HTML file.
        if path.endswith(".html"):
            self.mem_tl.export_memory_timeline_html(path, device)
        elif path.endswith(".gz"):
            with tempfile.NamedTemporaryFile("w+t", suffix=".json") as fp:
                if path.endswith("raw.json.gz"):
                    self.mem_tl.export_memory_timeline_raw(fp.name, device)
                else:
                    self.mem_tl.export_memory_timeline(fp.name, device)
                with open(fp.name) as fin, gzip.open(path, "wt") as fout:
                    fout.writelines(fin)
        else:
            self.mem_tl.export_memory_timeline(path, device)


class ProfilerAction(Enum):
    # [CLASS DOC] Enumeration of per-step profiler actions: NONE, WARMUP, RECORD,
    # RECORD_AND_SAVE (end of cycle, triggers the trace callback) and DEVICE_STOPPED
    # (internal only; set when device collection stops early, forbidden from schedules).
    """
    Profiler actions that can be taken at the specified intervals.

    NONE, WARMUP, RECORD, and RECORD_AND_SAVE are user-facing values that may
    be returned from a user-provided schedule. DEVICE_STOPPED is set
    internally by the profiler when device collection stops early due to
    errors; it must not be returned from a user-provided schedule.
    """

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3
    DEVICE_STOPPED = 4


def _unreachable_transition(prev: str, current: str) -> None:
    # [FUNCTION DOC] HELPER: Always raises RuntimeError reporting an internal error for an
    # impossible transition from prev to current action.
    raise RuntimeError(
        f"Profiler internal error: {prev} -> {current} should be unreachable"
    )


def schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
    skip_first_wait: int = 0,
) -> Callable:
    # [FUNCTION DOC] Builds and returns a step-to-action function. Validates arguments first
    # (AssertionError unless wait, warmup, repeat, skip_first are non-negative and active is
    # positive) and warns if warmup is zero. The returned function maps a step number to
    # NONE during the skipped steps and waiting, WARMUP during warmup, RECORD during active
    # steps and RECORD_AND_SAVE on the last active step of each cycle; after repeat cycles
    # (when repeat is positive) it returns NONE. skip_first_wait shifts the first cycle
    # earlier by the wait length. The returned function raises AssertionError for negative
    # steps.
    """
    Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip
    the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
    then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.

    The ``skip_first_wait`` parameter controls whether the first ``wait`` stage should be skipped.
    This can be useful if a user wants to wait longer than ``skip_first`` between cycles, but not
    for the first profile. For example, if ``skip_first`` is 10 and ``wait`` is 20, the first cycle will
    wait 10 + 20 = 30 steps before warmup if ``skip_first_wait`` is zero, but will wait only 10
    steps if ``skip_first_wait`` is non-zero. All subsequent cycles will then wait 20 steps between the
    last active and warmup.
    """

    def schedule_fn(step: int) -> ProfilerAction:
        # [FUNCTION DOC] HELPER: Pure mapping from a non-negative step number to the profiler
        # action according to the enclosing schedule parameters; see schedule for the rules.
        # Raises AssertionError for negative steps.
        if step < 0:
            raise AssertionError(f"Step must be non-negative. Got {step}.")
        if step < skip_first:
            return ProfilerAction.NONE
        else:
            step -= skip_first
        # If wait >> skip_first and we want to grab profiling early, shift left by wait if skip_first_wait is True
        if skip_first_wait != 0:
            step += wait
        num_steps = wait + warmup + active
        if repeat > 0 and step / num_steps >= repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < wait:
            return ProfilerAction.NONE
        elif mod_step < wait + warmup:
            return ProfilerAction.WARMUP
        else:
            return (
                ProfilerAction.RECORD
                if mod_step < num_steps - 1
                else ProfilerAction.RECORD_AND_SAVE
            )

    if wait < 0 or warmup < 0 or active <= 0 or repeat < 0 or skip_first < 0:
        raise AssertionError(
            f"Invalid profiler schedule arguments. Got wait={wait} (need >= 0), warmup={warmup} (need >= 0), "
            f"active={active} (need > 0), repeat={repeat} (need >= 0), skip_first={skip_first} (need >= 0)."
        )
    if warmup == 0:
        warn(
            "Profiler won't be using warmup, this can skew profiler results",
            stacklevel=2,
        )
    return schedule_fn


def _default_schedule_fn(_: int) -> ProfilerAction:
    # [FUNCTION DOC] HELPER: Default schedule; ignores the step and always returns RECORD.
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.
    """
    return ProfilerAction.RECORD


def tensorboard_trace_handler(
    dir_name: str,
    worker_name: str | None = None,
    use_gzip: bool = False,
    use_python_export: bool = False,
):
    # [FUNCTION DOC] Returns a trace-ready callback that exports each finished trace into
    # dir_name (created if missing, RuntimeError if it cannot be) with a unique file name
    # built from the worker name (default host name and process id) and a nanosecond
    # timestamp, optionally gzipped and optionally using the Python exporter. The callback
    # takes the profiler object and returns None.
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import socket
    import time

    def handler_fn(prof) -> None:
        # [FUNCTION DOC] HELPER: Ensures the output directory exists, fills in the default
        # worker name once (remembered across calls), and exports the profiler's chrome trace
        # to a new uniquely named file there. Raises RuntimeError if the directory cannot be
        # created.
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(
            os.path.join(dir_name, file_name),
            use_python_export=use_python_export,
        )

    return handler_fn


class profile(_KinetoProfile):
    # [CLASS DOC] Public profiler context manager. Adds a step schedule, a trace-ready
    # callback, a step counter, per-step record markers and an action state machine on top of
    # the Kineto profile. State: schedule function, current action (possibly overridden to
    # DEVICE_STOPPED), the raw previous schedule action, step number, and a table mapping
    # (previous action, current action) to the list of operations to run. Entering starts
    # profiling, step() advances the schedule, exiting stops and cleans up.
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.

            Each item can be a ``ProfilerActivity`` enum (collects all default
            activity types for that group) or a ``dict`` mapping a ``ProfilerActivity``
            to a list of individual activity type names to collect, e.g.
            ``{ProfilerActivity.CUDA: ["GPU_MEMCPY", "CUDA_RUNTIME"]}``.
            An empty list (e.g. ``{ProfilerActivity.CUDA: []}``) means collect
            nothing for that group.
            The same activity group must not appear more than once.
            See :class:`~torch.profiler.ProfilerActivity` for the valid
            activity type names and device-specific behavior.
        schedule (Callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (Callable): callable invoked at the end of each profiling cycle
            (when ``schedule`` returns ``ProfilerAction.RECORD_AND_SAVE``). Receives the
            :class:`profile` instance as its only argument, typically used to export the
            trace (e.g. via :meth:`export_chrome_trace`) or print a summary.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B

            .. deprecated::
                ``with_modules`` is deprecated and will be removed in a future version.
                It only collects data for TorchScript models, which are themselves
                deprecated, and does nothing in eager mode. Use ``with_stack=True``,
                which records ``nn.Module`` events for eager models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used for Kineto library features. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler. See the examples section below for a code sample.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
        post_processing_timeout_s (float): Optional timeout in seconds for post-processing profiler
            results. If specified, event parsing will stop after this duration and return partial
            results. Useful for handling large traces that may take too long to process.
        custom_trace_id_callback (Callable[[], str], optional): User-supplied trace ID generator,
            invoked once per profiling cycle. Defaults to a random UUID; retrieve via
            :meth:`get_trace_id`.

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --logdir dir_name``

        to see the results in TensorBoard.
        For more information, see
        `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.


    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            )
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the fourth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=trace_handler,
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                # send a signal to the profiler that the next iteration has started
                p.step()

    The following sample shows how to setup up an Execution Trace Observer (`execution_trace_observer`)

    .. code-block:: python

        with torch.profiler.profile(
            ...
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback("./execution_trace.json")
            ),
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                p.step()

    You can also refer to test_execution_trace_with_kineto() in tests/profiler/test_profiler.py.
    Note: One can also pass any object satisfying the _ITraceObserver interface.
    """

    def __init__(
        self,
        *,
        activities: Iterable[ProfilerActivity | dict[ProfilerActivity, list[str]]]
        | None = None,
        schedule: Callable[[int], ProfilerAction] | None = None,
        on_trace_ready: Callable[..., Any] | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: _ExperimentalConfig | None = None,
        execution_trace_observer: _ITraceObserver | None = None,
        acc_events: bool = False,
        custom_trace_id_callback: Callable[[], str] | None = None,
        post_processing_timeout_s: float | None = None,
    ) -> None:
        # [FUNCTION DOC] Validates activities (AssertionError if the resulting set is empty),
        # initializes the base class (using the given execution trace observer or one built
        # from environment variables), picks the schedule (default records every step; markers
        # per step only when a schedule is given), evaluates the schedule for step zero
        # (ValueError if it returns DEVICE_STOPPED), and builds the transition table from
        # (previous, current) action pairs to operation lists, including warnings for
        # malformed schedules and recovery rules for early device stop. Finally initializes the
        # global Kineto step counter. Side effect: global step-tracker state.
        if activities is not None:
            activities_set, _ = _parse_activities(activities)
        else:
            activities_set = supported_activities()
        if len(activities_set) == 0:
            raise AssertionError("No valid profiler activities found")

        super().__init__(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            execution_trace_observer=execution_trace_observer
            or ExecutionTraceObserver.build_execution_trace_obs_from_env(),
            acc_events=acc_events,
            custom_trace_id_callback=custom_trace_id_callback,
            post_processing_timeout_s=post_processing_timeout_s,
        )

        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = _default_schedule_fn
            self.record_steps = False
        self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.step_rec_fn: prof.record_function | None = None

        schedule_action = self.schedule(self.step_num)
        if schedule_action == ProfilerAction.DEVICE_STOPPED:
            raise ValueError(
                "ProfilerAction.DEVICE_STOPPED is set internally by the "
                "profiler and must not be returned by a user-provided schedule"
            )
        self.current_action = schedule_action
        # Raw schedule output of the previous step, separate from
        # current_action which step() may override to DEVICE_STOPPED. Used to
        # detect cycle boundaries (RECORD_AND_SAVE -> next) when warmup=0,
        # so DEVICE_STOPPED can exit and resume profiling on the new cycle.
        self._prev_schedule_action = schedule_action

        self.action_map: dict[
            tuple[ProfilerAction, ProfilerAction | None], list[Any]
        ] = {
            # key is (prev_action, current_action), value is action list corresponding to the state pair.
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.prepare_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: WARMUP followed by NONE"),
                self.start_trace,
                self.stop_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self.start_trace],
            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: RECORD followed by NONE"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(warn, "Incorrect schedule: RECORD followed by WARMUP"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            # DEVICE_STOPPED: entered when device collection stops early (e.g.
            # CUPTI buffer overflow). Absorbs the remainder of the current
            # profiling cycle, then exits at the next cycle boundary.
            #
            # All three entry transitions fire _trace_ready so the user's
            # callback is invoked even when entry happens from WARMUP. This
            # matters for active=1 schedules: the only active step (R&S) gets
            # converted to DEVICE_STOPPED via WARMUP -> R&S override, and
            # without _trace_ready firing here the user would never see a
            # callback for that cycle. The trace will be empty (no recording
            # ran), but the callback firing is the signal that the cycle
            # completed.
            (ProfilerAction.WARMUP, ProfilerAction.DEVICE_STOPPED): [
                self.start_trace,
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.RECORD, ProfilerAction.DEVICE_STOPPED): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.DEVICE_STOPPED): [],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.WARMUP): [
                self.prepare_trace,
            ],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.NONE): [],
            # Cycle boundary recovery: when the schedule has no WARMUP phase
            # (warmup=0), DEVICE_STOPPED exits directly into the next cycle's
            # active phase. Mirrors (NONE, RECORD) / (NONE, RECORD_AND_SAVE).
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.RECORD): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.RECORD_AND_SAVE): [
                self.prepare_trace,
                self.start_trace,
            ],
            # Unreachable transitions:
            # - prev=NONE: step()'s entry guard excludes it, and start() and
            #   __init__ both reject DEVICE_STOPPED from a user schedule.
            # - prev=RECORD_AND_SAVE: entry guard excludes it because the
            #   natural (R&S, *) transitions already do the right thing to
            #   recover.
            (ProfilerAction.NONE, ProfilerAction.DEVICE_STOPPED): [
                partial(_unreachable_transition, "NONE", "DEVICE_STOPPED"),
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.DEVICE_STOPPED): [
                partial(_unreachable_transition, "RECORD_AND_SAVE", "DEVICE_STOPPED"),
            ],
            # used for exit action
            (ProfilerAction.WARMUP, None): [self.start_trace, self.stop_trace],
            (ProfilerAction.RECORD, None): [self.stop_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.DEVICE_STOPPED, None): [],
        }
        # Start tracking increments to profiler step, this will be used
        # by Kineto
        prof.KinetoStepTracker.init_step_count(PROFILER_STEP_NAME)

    def __enter__(self):
        # [FUNCTION DOC] Starts profiling and returns the profiler itself.
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] Stops profiling, erases the global step counter entry, and cleans up
        # the execution trace observer if any. Does not suppress exceptions (returns None).
        self.stop()
        prof.KinetoStepTracker.erase_step_count(PROFILER_STEP_NAME)
        if self.execution_trace_observer:
            self.execution_trace_observer.cleanup()

    def start(self) -> None:
        # [FUNCTION DOC] Runs the transition from the NONE action to the current action (which
        # prepares and starts tracing as needed) and, when a schedule is in use, opens the
        # record marker for the current step.
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def stop(self) -> None:
        # [FUNCTION DOC] Closes the open step marker, runs the exit transition for the current
        # action (stopping the trace and invoking the trace-ready callback where applicable),
        # then resets the current action to the raw schedule value so a later start works.
        # Raises AssertionError if the stored raw action is DEVICE_STOPPED.
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
            self.step_rec_fn = None
        self._transit_action(self.current_action, None)
        # Reset current_action to the schedule's view in case step() had
        # overridden it to DEVICE_STOPPED. Without this, a subsequent start()
        # would transit (NONE, DEVICE_STOPPED) — unreachable by contract —
        # and leave the profiler running with no Kineto session.
        if self._prev_schedule_action == ProfilerAction.DEVICE_STOPPED:
            raise AssertionError(
                "_prev_schedule_action must never be DEVICE_STOPPED here"
            )
        self.current_action = self._prev_schedule_action

    def step(self) -> None:
        # [FUNCTION DOC] Advances to the next step: closes the old step marker, evaluates the
        # schedule for the new step (ValueError, with no state change, if it returns
        # DEVICE_STOPPED), updates step number and actions, and applies the early device stop
        # logic (warns and enters DEVICE_STOPPED when Kineto reports a stop during warmup or
        # recording; leaves it at a cycle boundary). Then runs the transition operations for
        # the old/new action pair, bumps the Kineto step counter when the daemon or forced hook
        # is enabled, and opens a new step marker if a schedule is in use.
        """
        Signals the profiler that the next profiling step has started.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
            # Drop our reference so a subsequent stop() / step() — e.g. after
            # this step() raises — doesn't double-exit the same instance.
            self.step_rec_fn = None
        prev_action = self.current_action
        prev_schedule_action = self._prev_schedule_action
        next_step = self.step_num + 1
        schedule_action = self.schedule(next_step)

        # Note that we check schedule validity BEFORE changing the profiler's
        # internal state. This prevents the profiler from being in an invalid
        # state.
        if schedule_action == ProfilerAction.DEVICE_STOPPED:
            raise ValueError(
                "ProfilerAction.DEVICE_STOPPED is set internally by the "
                "profiler and must not be returned by a user-provided schedule"
            )

        self.step_num = next_step
        self.current_action = schedule_action
        self._prev_schedule_action = schedule_action

        # DEVICE_STOPPED handling: when Kineto signals that device collection
        # has stopped early (e.g. CUPTI buffer overflow), we enter
        # DEVICE_STOPPED and stay there until the current profiling cycle ends.
        #
        # Once in DEVICE_STOPPED (prev_action == DEVICE_STOPPED):
        # - We exit at the next cycle boundary, defined by either:
        #   - prev_schedule_action == RECORD_AND_SAVE: since active > 0,
        #     every cycle ends with RECORD_AND_SAVE, so the next step starts
        #     a new cycle regardless of warmup/wait shape.
        #   - current_action == NONE: safety hatch for user-defined schedules
        #     that may emit NONE without a preceding RECORD_AND_SAVE. (The
        #     standard schedule() builder always pairs them.)
        # - Otherwise we stay in DEVICE_STOPPED, absorbing the rest of the
        #   cycle.
        #
        # Entering DEVICE_STOPPED (override current_action to DEVICE_STOPPED):
        # - We enter when all of the following hold:
        #   - _is_kineto_stopped() is True: Kineto reports the stop.
        #   - use_device is set: a CPU-only profiler must not react to a
        #     stale stopped flag from a previous device profiler.
        #   - prev_action is WARMUP or RECORD: these have no built-in cleanup
        #     in their natural transitions.
        #   - current_action is not NONE: if the schedule is already stopping
        #     us (e.g. RECORD_AND_SAVE -> NONE), respect that rather than
        #     overriding.
        # - Otherwise we leave current_action as the schedule's output.
        # - prev_action == RECORD_AND_SAVE is excluded: the natural (R&S, *)
        #   transitions already call prepare_trace, so overriding here would
        #   skip the reset and waste the next cycle.

        if prev_action == ProfilerAction.DEVICE_STOPPED:
            at_cycle_boundary = (
                prev_schedule_action == ProfilerAction.RECORD_AND_SAVE
                or self.current_action == ProfilerAction.NONE
            )
            if not at_cycle_boundary:
                self.current_action = ProfilerAction.DEVICE_STOPPED
        elif (
            torch.autograd._is_kineto_stopped()
            and self.use_device is not None
            and prev_action in (ProfilerAction.WARMUP, ProfilerAction.RECORD)
            and self.current_action != ProfilerAction.NONE
        ):
            warn(
                "Device profiling activity collection was stopped early "
                f"at step {self.step_num}. Profiler schedule is proceeding "
                "until next cycle without actual profiler activity collection."
            )
            self.current_action = ProfilerAction.DEVICE_STOPPED

        self._transit_action(prev_action, self.current_action)
        if os.environ.get("KINETO_USE_DAEMON", "") or (
            is_fbcode() and os.environ.get("KINETO_FORCE_STEP_HOOK", "")
        ):
            prof.KinetoStepTracker.increment_step(PROFILER_STEP_NAME)

        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def set_custom_trace_id_callback(self, callback) -> None:
        # [FUNCTION DOC] Replaces the trace id generator callback, used when the next cycle
        # prepares its trace.
        """
        Set the trace ID generator. Called at the start of each cycle, so updating
        it between cycles yields distinct IDs per cycle.
        """
        self.custom_trace_id_callback = callback

    def get_trace_id(self):
        # [FUNCTION DOC] Returns the trace id of the underlying profiler, or None if no
        # profiler has been created yet.
        """
        Returns the current trace ID.
        """
        if self.profiler is None:
            return None
        return self.profiler.trace_id

    def _trace_ready(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the user's trace-ready callback with this profiler, if
        # one was provided.
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def _transit_action(self, prev_action, current_action) -> None:
        # [FUNCTION DOC] HELPER: Looks up the operation list for the (previous, current) action
        # pair in the transition table and calls each operation in order; unknown pairs do
        # nothing.
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()

    def _stats(self) -> prof._ProfilerStats | None:
        # [FUNCTION DOC] HELPER: Returns the underlying profiler's stats object, or None if
        # no profiler exists yet.
        if self.profiler is None:
            return None
        return self.profiler._stats


class ExecutionTraceObserver(_ITraceObserver):
    # [CLASS DOC] Records a PyTorch Execution Trace to a file by registering a native
    # observer with record-function callbacks. State: registered and running flags, the
    # requested output path (optionally gzip), the actual observer path (a temp file when
    # gzip), whether to gather extra resources (generated kernels), and the resources
    # directory. Intended as one instance per process. Key methods: register_callback,
    # start, stop, unregister_callback (finalizes outputs); cleanup and deletion also
    # unregister.
    """Execution Trace Observer

    Each process can have a single ExecutionTraceObserver instance. The observer
    can be added to record function callbacks via calling register_callback()
    explicitly. Without calling unregister_callback(), repeated calls to
    register_callback() will not add additional observers to record function
    callbacks. Once an ExecutionTraceObserver is created, the start() and stop()
    methods control when the event data is recorded.

    Deleting or calling unregister_callback() will remove the observer from the
    record function callbacks, finalize the output file, and will stop
    incurring any overheads.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Sets all state to defaults: unregistered, not running, no extra
        # resource collection, empty resources directory and output paths.
        """
        Initializes the default states.
        """
        self._registered = False
        self._execution_trace_running = False
        self.extra_resources_collection = False
        self.resources_dir: str = ""
        self.output_file_path: str = ""
        self.output_file_path_observer: str = ""

    def __del__(self) -> None:
        # [FUNCTION DOC] Finalizer; unregisters the callback so outputs are finalized.
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    @staticmethod
    def build_execution_trace_obs_from_env() -> ExecutionTraceObserver | None:
        # [FUNCTION DOC] When the environment variable enabling execution traces equals 1,
        # creates a temporary json output file (warns and returns None on failure), builds an
        # observer registered to it, and enables extra resource collection if the second
        # environment variable equals 1. Returns the observer, or None when not enabled.
        # Side effects: creates a file that is not deleted, registers a native observer.
        """
        Returns an ExecutionTraceObserver instance if the environment variable
        ENABLE_PYTORCH_EXECUTION_TRACE is set to 1, otherwise returns None.

        Configures the observer to also collect extra resources if the environment variable
        ``ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS=1``. These are resources such as generated kernels,
        index tensor data etc. that are required to make the Execution Trace replayable.
        """
        if os.environ.get("ENABLE_PYTORCH_EXECUTION_TRACE", "0") == "1":
            try:
                with tempfile.NamedTemporaryFile(
                    "w+t", suffix=".et.json", delete=False
                ) as fp:
                    filename = fp.name
            except Exception as e:
                warn(
                    f"Execution trace will not be recorded. Exception on creating default temporary file: {e}",
                    stacklevel=2,
                )
                return None
            et = ExecutionTraceObserver()
            et.register_callback(filename)
            # additionally, check if the env requires us to collect extra resources
            if os.environ.get("ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS", "0") == "1":
                et.set_extra_resource_collection(True)
            else:
                et.set_extra_resource_collection(False)
            return et
        return None

    def set_extra_resource_collection(self, val) -> None:
        # [FUNCTION DOC] Sets the flag controlling collection of extra resources; when true,
        # eagerly creates the resources directory next to the output file (so register the
        # callback first). Returns None.
        """
        Collects extra resources such as generated kernels, index tensor data, and any other
        metadata that is required to complete the Execution Trace content.

        The caller should call this method with val=True after calling register_callback() if they want
        to collect the extra resources.
        """
        self.extra_resources_collection = val
        if self.extra_resources_collection:
            self.get_resources_dir(can_create=True)
        return

    def register_callback(self, output_file_path: str) -> Self:
        # [FUNCTION DOC] If not yet registered, remembers the output path and registers the
        # native observer writing to it; for a path ending in .gz it writes to a temporary
        # uncompressed file instead (compressed on unregister). Repeated calls while
        # registered change nothing. The registered flag is set from the native call result.
        # Returns the observer itself.
        """
        Adds ET observer to record function callbacks. The data will be
        written to output_file_path.
        """

        def get_temp_uncompressed_file() -> str:
            # [FUNCTION DOC] HELPER: Creates a persistent temporary json file and returns its
            # path.
            with tempfile.NamedTemporaryFile("w+b", suffix=".json", delete=False) as fp:
                return fp.name

        if not self._registered:
            self.output_file_path = output_file_path
            if output_file_path.endswith(".gz"):
                output_file_path = get_temp_uncompressed_file()
            self.output_file_path_observer = output_file_path
            self._registered = _add_execution_trace_observer(output_file_path)
        return self

    def get_resources_dir(self, can_create=False) -> str | None:
        # [FUNCTION DOC] Returns the resources directory path: None if extra resource
        # collection is off, the cached path if already known, otherwise the directory derived
        # from the output path (created only if can_create is true), or None if it does not
        # exist and cannot be created. Caches a found path on the instance.
        """
        Generates the resources directory for the generated kernels,
        or index tensor data or any other metadata that is required
        to complete the Execution Trace content.

        The directory is created right where the ET file is being output.

        Only works if the observer has called set_extra_resource_collection(val=True).

        Returns None if the observer is not configured with extra resource collection.
        """
        if not self.extra_resources_collection:
            return None
        if self.resources_dir:
            # already created
            return self.resources_dir
        generated_path = ExecutionTraceObserver.get_resources_dir_for_et_path(
            self.output_file_path, create_dir=can_create
        )
        if not generated_path:
            # could not find or create the resources dir
            return None
        self.resources_dir = generated_path
        return self.resources_dir

    @staticmethod
    def get_resources_dir_for_et_path(
        trace_path, create_dir: bool = False
    ) -> str | None:
        # [FUNCTION DOC] Computes the sibling resources directory for a trace file: same folder,
        # file name stem plus a resources suffix. If missing, creates it when create_dir is
        # true (warning and returning None on failure), else returns None. Returns the path
        # when it exists or was created. Side effect: may create a directory.
        work_dir, file_name = os.path.split(trace_path)
        resource_dir = os.path.join(
            work_dir, os.path.splitext(file_name)[0] + "_resources"
        )
        if not os.path.exists(resource_dir):
            if create_dir:
                try:
                    os.mkdir(resource_dir)
                except Exception:
                    warn(
                        f"Execution trace exception when creating {resource_dir}",
                        stacklevel=2,
                    )
                    return None
            else:
                return None
        return resource_dir

    def unregister_callback(self) -> None:
        # [FUNCTION DOC] If registered: stops the observer, copies generated inductor kernel
        # source files into the resources directory (failures only warn), removes the native
        # observer, compresses the temporary file to the requested gzip output (deleting the
        # temporary) when the output path ends with gz, and marks the observer unregistered.
        # Does nothing when not registered. Side effects: file copies and writes, console
        # print during compression.
        """
        Removes ET observer from record function callbacks.
        """

        def _save_triton_kernels() -> None:
            # [FUNCTION DOC] HELPER: Copies the source file of every loaded inductor-generated
            # module into the resources directory. Returns early (warning on error) if the
            # directory is unavailable. Side effect: file copies.
            try:
                resource_dir = self.get_resources_dir()
            except Exception as e:
                warn(
                    f"Execution trace exception when generating resource directory: {e}",
                    stacklevel=2,
                )
                return
            if not resource_dir:
                return

            # Save the kernel paths for the generated kernels
            from torch._inductor.codecache import PyCodeCache

            kernel_files = [
                v.__file__
                for v in PyCodeCache.modules
                if getattr(v, "__file__", None) is not None
            ]

            for kernel_file in kernel_files:
                if kernel_file is None:
                    continue
                name = os.path.basename(kernel_file)
                dst = os.path.join(resource_dir, name)
                shutil.copyfile(kernel_file, dst)

        def _save_gz_file(uncompressed_file: str, output_file: str) -> None:
            # [FUNCTION DOC] HELPER: Prints a progress message, gzip-compresses the input file
            # into the output file, then deletes the input file.
            print(f"Execution Trace: compressing {uncompressed_file} to {output_file}")
            with open(uncompressed_file, "rb") as fin:
                with gzip.open(output_file, "wb") as fout:
                    fout.writelines(fin)
            os.remove(uncompressed_file)

        if self._registered:
            self.stop()

            try:
                _save_triton_kernels()
            except Exception as e:
                warn(f"Execution trace failed to save kernels: {e}", stacklevel=2)

            _remove_execution_trace_observer()
            if self.output_file_path.endswith("gz"):
                _save_gz_file(self.output_file_path_observer, self.output_file_path)

            self._registered = False

    @property
    def is_registered(self):
        # [FUNCTION DOC] Property; True if the native observer is currently registered.
        """
        Returns True if the execution trace observer is registered, otherwise False.
        """
        return self._registered

    def is_running(self):
        # [FUNCTION DOC] Returns True if capture is currently enabled.
        """
        Returns True if the observer is running, otherwise False.
        """
        return self._execution_trace_running

    def start(self) -> None:
        # [FUNCTION DOC] If registered and not already running, enables the native observer,
        # marks it running, and records the process group configuration into the trace.
        """
        Starts to capture.
        """
        if self._registered and not self._execution_trace_running:
            _enable_execution_trace_observer()
            self._execution_trace_running = True
            self._record_pg_config()

    def stop(self) -> None:
        # [FUNCTION DOC] If running, disables the native observer and marks it not running.
        """
        Stops to capture.
        """
        if self._execution_trace_running:
            _disable_execution_trace_observer()
            self._execution_trace_running = False

    def cleanup(self) -> None:
        # [FUNCTION DOC] Finalizes outputs by unregistering the callback.
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    def get_output_file_path(self) -> str | None:
        # [FUNCTION DOC] Returns the requested output file path, or None if none was set.
        """
        Returns the output file name or None.
        """
        if self.output_file_path:
            return self.output_file_path
        else:
            return None

    def _record_pg_config(self) -> None:
        # [FUNCTION DOC] HELPER: When registered and distributed is available and initialized,
        # writes the process group configuration (as JSON) into the trace by entering a
        # record-function scope named for process group initialization.
        # KNOWN ISSUE: the record-function scope is entered but never exited here.
        # Records the PG config info to the trace as node:
        #  ## process_group:init ##
        if (
            self.is_registered
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            pg_config_info = torch.distributed.distributed_c10d._world.pg_config_info
            torch.autograd._record_function_with_args_enter(
                "## process_group:init ##",
                json.dumps(pg_config_info, cls=_NumpyEncoder),
            )
