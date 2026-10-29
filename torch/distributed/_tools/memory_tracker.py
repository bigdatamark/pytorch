# mypy: allow-untyped-defs
# [MODULE DOC] Operator-level GPU memory tracking for nn.Module runs. Provides MemoryProfileDispatchMode (a dispatch mode recording memory after every operator) and MemoryTracker (module hooks, recording, summary printing, matplotlib plotting, pickle save/load).
# Constant BYTES_PER_MB converts bytes to megabytes. No import-time side effects beyond imports.
import operator
import pickle
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, no_type_check, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


BYTES_PER_MB = 1024 * 1024.0


class MemoryProfileDispatchMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that, after executing each operator, asks its owning MemoryTracker to record memory stats under a unique operator name.
    # State: the memory_tracker reference. Detach operations are executed but not recorded.
    """Run in ``TorchDispatchMode`` to get memory stats at operator level."""

    def __init__(self, memory_tracker) -> None:
        # [FUNCTION DOC] Stores the memory tracker that receives per-operator records.
        self.memory_tracker = memory_tracker

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        # [FUNCTION DOC] Runs the intercepted operator and returns its result unchanged. Except for detach, builds a name from the tracker's current module name, the operator name and a per-operator running count, increments that count, and calls the tracker to record memory stats under that name.
        # Side effects: mutates the tracker's operator-name counters and stat tables.
        rs = func(*args, **kwargs)
        if func is torch.ops.aten.detach.default:
            return rs
        func_name: str = (
            self.memory_tracker._cur_module_name
            + "."
            + func.__name__
            + "_"
            + str(self.memory_tracker._operator_names[func.__name__])
        )
        self.memory_tracker._operator_names[func.__name__] = (
            self.memory_tracker._operator_names[func.__name__] + 1
        )
        self.memory_tracker._record_memory_stats(func_name)

        return rs


class MemoryTracker:
    # [CLASS DOC] Collects allocated, active and reserved device memory (in MB) after each operator while a root module runs, with markers for forward start and the forward/backward boundary.
    # State: forward-hook handles, per-operator-name counters, three tables mapping operator index to (name, MB), marker positions, current module name, operator index, allocation-retry count, and the device module. Use start_monitor, run the model, stop, then summary, show_traces, save_stats or load.
    """
    Collect and plot the memory stats at operator level.

    Includes ``memories_allocated``, ``memories_active`` and ``memories_reserved``.
    It also prints a summary for the top 20 operators that generate the most memories.

    Example usage:

        >>> # xdoctest: +SKIP(failing)
        >>> net.cuda()
        >>> input = input.cuda()

        >>> mem_tracker = MemoryTracker()
        >>> mem_tracker.start_monitor(net)

        >>> net.zero_grad(True)
        >>> loss = net(input)
        >>> if isinstance(loss, dict):
        >>>    loss = loss['out']
        >>> loss.sum().backward()
        >>> net.zero_grad(set_to_none=True)

        >>> mem_tracker.stop()
        >>> mem_tracker.summary()
        >>> mem_tracker.show_traces()
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Logs API usage once and initializes empty hooks, counters, stat tables, markers, operator index, retry count and the current device module.
        torch._C._log_api_usage_once("torch.distributed.memory_tracker")
        self._hooks: list[RemovableHandle] = []
        self._operator_names: dict[str, int] = defaultdict(int)
        self.memories_allocated: dict[int, dict[str, float]] = defaultdict()
        self.memories_active: dict[int, dict[str, float]] = defaultdict()
        self.memories_reserved: dict[int, dict[str, float]] = defaultdict()
        self._markers: dict[str, int] = defaultdict(int)
        self._cur_module_name: str = ""
        self._op_index: int = 0
        self._num_alloc_retries: int = 0
        self._device_module = torch.get_device_module()

    @no_type_check
    def start_monitor(self, root_module: nn.Module) -> None:
        # [FUNCTION DOC] Resets state, tags the root module as root and every other submodule as non-root (via an attribute), and registers a forward pre-hook and forward hook on each module except those whose name contains the fused grouped embedding bag marker. Empties the device cache, then creates and enters a MemoryProfileDispatchMode.
        # Raises AssertionError if a profile mode is already active. Side effects: mutates modules, installs hooks, enters the dispatch mode.
        """
        Register module hooks and entering ``MemoryProfileDispatchMode``.

        This enables operator level memory stats can be tracked during module runtime.
        """
        self._clear_state()
        root_module.__setattr__("_memory_tracker_is_root", True)
        for name, m in root_module.named_modules():
            if m is not root_module:
                m.__setattr__("_memory_tracker_is_root", False)
            # fused_proxy_group does not support hooks
            if ".fused_proxy_grouped_embedding_bag" in name:
                continue
            # hook ordering with other hooks added by users is not managed, so
            # the memory stats tracked here may not completely accurate.
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            # it does not work well with jagged tensor somehow, the root cause is not
            # clear and remove it for now as it does not really capture important info.
            # h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2])
        self._device_module.empty_cache()
        if getattr(self, "profile_mode", None) is not None:
            raise AssertionError
        self.profile_mode = MemoryProfileDispatchMode(self)
        self.profile_mode.__enter__()

    @no_type_check
    def stop(self) -> None:
        # [FUNCTION DOC] Reads the allocation retry count from the device memory stats, removes all registered hooks, exits and clears the profile dispatch mode.
        # Raises AssertionError if monitoring was not started.
        """
        Remove module hooks and exit ``MemoryProfileDispatchMode`` to stop tracking memory stats at operator level.

        Get some aggregated stats when the memory_tracker() is enabled, like ``num_alloc_retries``.
        """
        self._num_alloc_retries = self._device_module.memory_stats().get(
            "num_alloc_retries", 0
        )

        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if getattr(self, "profile_mode", None) is None:
            raise AssertionError
        self.profile_mode.__exit__(None, None, None)
        self.profile_mode = None

    @no_type_check
    def summary(self, top: int = 20) -> None:
        # [FUNCTION DOC] Prints the number of allocation retries and the top operators by memory growth. For each operator index after the first, computes the allocated memory difference from the previous record, then prints the largest values in MB, up to the given count.
        # Requires recorded data (raises a key error if none). Output goes to standard output.
        """
        Print out the top operators that generate the most memories.

        The number of the top operators can be configured.
        """
        op_diff: dict[str, float] = defaultdict(float)
        op_name, previous_allocated_memory = self.memories_allocated[0]
        for i in range(1, self._op_index):
            op_name, current_allocated_memory = self.memories_allocated[i]
            op_diff[op_name] = current_allocated_memory - previous_allocated_memory
            previous_allocated_memory = current_allocated_memory

        print("------------------------------------------------")
        print(f"The number of alloc retries are: {self._num_alloc_retries}")
        print(f"Top {top} ops that generates memory are:")
        for k, v in sorted(op_diff.items(), key=operator.itemgetter(1), reverse=True)[
            :top
        ]:
            print(f"{k}: {v}MB")
        print("------------------------------------------------")

    @no_type_check
    def show_traces(self, path: str = "") -> None:
        # [FUNCTION DOC] Plots the recorded memory traces with matplotlib: one figure with allocated, active and reserved lines, plus one figure per series, with vertical marker lines (red for the forward/backward boundary, black otherwise). If a path is given, first loads stats from that file.
        # Does not call show or save; figures are only created. Imports matplotlib lazily.
        import matplotlib.pyplot as plt

        def _plot_figure(x, y_values, labels):
            # [FUNCTION DOC] HELPER: Creates a new figure, draws each series against the operator index with labels and axis titles, and draws a vertical line for every marker spanning slightly beyond the data range.
            min_val = min(chain.from_iterable(y_values)) * 0.999
            max_val = max(chain.from_iterable(y_values)) * 1.001
            plt.figure()
            for y, label in zip(y_values, labels):
                plt.plot(x, y, label=label)
            plt.xlabel("# Operator Calls")
            plt.ylabel("Memory (MB)")
            plt.legend()
            for marker_name, marker in self._markers.items():
                if marker_name == "fw_bw_boundary":
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "r",
                        lw=2,
                        label=marker_name,
                    )
                else:
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "k-",
                        lw=2,
                        label=marker_name,
                    )

        if path != "":
            self.load(path)

        y_1 = [gb for (name, gb) in self.memories_allocated.values()]
        y_2 = [gb for (name, gb) in self.memories_active.values()]
        y_3 = [gb for (name, gb) in self.memories_reserved.values()]
        x = list(range(len(y_1)))
        # Split figures when there is big difference between
        # "reserved_memory" and "allocated_memory" or "active_memory".
        _plot_figure(
            x,
            [list(y_1), list(y_2), list(y_3)],
            ["allocated_memory", "active_memory", "reserved_memory"],
        )
        _plot_figure(x, [list(y_1)], ["allocated_memory"])
        _plot_figure(x, [list(y_2)], ["active_memory"])
        _plot_figure(x, [list(y_3)], ["reserved_memory"])

    def save_stats(self, path: str) -> None:
        # [FUNCTION DOC] Pickles the three memory tables, markers and retry count to the given file path (binary write, highest protocol). Raises on file errors.
        """Save the stats using pickle during runtime if users want to plot the traces in other places like notebook."""
        stats = {
            "memories_allocated": self.memories_allocated,
            "memories_active": self.memories_active,
            "memories_reserved": self.memories_reserved,
            "markers": self._markers,
            "num_alloc_retries": self._num_alloc_retries,
        }

        with open(path, "wb") as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        # [FUNCTION DOC] Reads a pickle file and replaces the memory tables, markers and retry count with its contents. Raises on file or key errors. KNOWN ISSUE: unpickling executes arbitrary code if the file is untrusted.
        """Load the pickled memory stats to plot the traces or print the summary."""
        with open(path, "rb") as f:
            stats = pickle.load(f)

        self.memories_allocated = stats["memories_allocated"]
        self.memories_active = stats["memories_active"]
        self.memories_reserved = stats["memories_reserved"]
        self._markers = stats["markers"]
        self._num_alloc_retries = stats["num_alloc_retries"]

    def _create_pre_forward_hook(self, name: str) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a forward pre-hook for the module with the given name. The hook sets the current module name to the name plus forward and, for the root module, adds the forward start marker.
        """Prefix operator name with current module and 'forward', and insert 'fw_start' marker at forward pass start."""

        def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
            # [FUNCTION DOC] HELPER: Pre-forward hook body; sets the current module name and adds the forward start marker when the module is the root.
            self._cur_module_name = f"{name}.forward"
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_start")

        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a forward hook that adds the forward/backward boundary marker when the root module finishes forward. The name argument is unused.
        """Insert the marker 'fw_bw_boundary' at the boundary of forward and backward pass."""

        def _post_forward_hook(
            module: nn.Module,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
        ) -> None:
            # [FUNCTION DOC] HELPER: Post-forward hook body; adds the boundary marker for the root module only.
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_bw_boundary")

        return _post_forward_hook

    def _create_backward_hook(self, name: str) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a backward hook that sets the current module name to the name plus backward. Currently not registered anywhere (its registration is commented out in start_monitor).
        """Insert the current module name with backward prefix for the operator name."""

        def _backward_hook(
            module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
        ) -> None:
            # [FUNCTION DOC] HELPER: Backward hook body; sets the current module name for subsequent operator naming.
            self._cur_module_name = f"{name}.backward"

        return _backward_hook

    @no_type_check
    def _record_memory_stats(self, fn_name: str) -> None:
        # [FUNCTION DOC] HELPER: Records the device's allocated, reserved and active memory (converted to MB) as (name, value) pairs under the current operator index in the three tables, then increments the index.
        """
        Record current memory allocated, current memory active and current memory reserved.

        The memory stats dict is indexed with ``self._op_index``.
        """
        memory_allocated: float = self._device_module.memory_allocated() / BYTES_PER_MB
        memory_reserved: float = self._device_module.memory_reserved() / BYTES_PER_MB
        memory_active: float = (
            self._device_module.memory_stats().get("active_bytes.all.current", 0)
            / BYTES_PER_MB
        )
        self.memories_allocated[self._op_index] = (fn_name, memory_allocated)
        self.memories_reserved[self._op_index] = (fn_name, memory_reserved)
        self.memories_active[self._op_index] = (fn_name, memory_active)
        self._op_index += 1

    def _add_marker(self, marker_name: str) -> None:
        # [FUNCTION DOC] HELPER: Records a marker at the position equal to the number of operators recorded so far.
        """Set the marker's x-axis value."""
        marker_val = len(self.memories_allocated.values())
        self._markers[marker_name] = marker_val

    def _clear_state(self) -> None:
        # [FUNCTION DOC] HELPER: Clears operator counters, memory tables and markers, and resets module name, operator index and retry count.
        """Clear states when start_monitor() is called."""
        self._operator_names.clear()
        self.memories_allocated.clear()
        self.memories_active.clear()
        self.memories_reserved.clear()
        self._markers.clear()
        self._cur_module_name = ""
        self._op_index = 0
        self._num_alloc_retries = 0
