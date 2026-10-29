# [MODULE DOC] NUMA binding support for worker processes (torchrun --numa-binding).
# Public API: AffinityMode (node, socket, exclusive, core-complex) and the NumaOptions
# dataclass. Internals pick the set of logical CPUs near the accelerator device used by a
# worker by reading Linux sysfs topology files and the process CPU affinity, then either
# prefix a command with the numactl tool or set the affinity of all threads of the current
# process. Failures are reported through signpost events and logging, and are optionally
# swallowed. Linux only. Import time only creates a logger and type variables.
"""
In NUMA (Non-Uniform Memory Access) systems, accessing memory on remote NUMA
nodes incurs additional latency. PyTorch provides NUMA binding utilities to
promote memory locality by binding worker processes to CPUs near their assigned accelerator devices.

In practice, NUMA binding typically results in 1-10% overall performance improvements,
but some workloads may obtain much greater benefits or none at all.

To enable NUMA binding, use the ``--numa-binding`` flag with :ref:`torchrun <launcher-api>`, e.g.:

.. code-block:: bash

    torchrun --numa-binding=node --nproc_per_node=8 train.py

Alternatively, pass :class:`NumaOptions` to ``LaunchConfig``
when using ``elastic_launch``.

See :class:`AffinityMode` for available binding modes.
"""

import os
import shutil
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from logging import getLogger
from typing import ParamSpec, TypeVar

import torch
from torch._utils_internal import signpost_event


__all__ = [
    "AffinityMode",
    "NumaOptions",
]

logger = getLogger(__name__)


class AffinityMode(str, Enum):
    # [CLASS DOC] String-valued enum of strategies for choosing CPUs near a worker's device.
    # Members: NODE (all CPUs of the device's NUMA node), SOCKET (all CPUs of every NUMA node on
    # that node's socket), EXCLUSIVE (an even share of the node's physical cores per device),
    # CORE_COMPLEX (one group of cores sharing the last-level cache per device).
    NODE = "node"
    """
    Each worker process and its threads will be bound to all the CPUs
    on the NUMA node containing the accelerator device whose local index equals the worker's
    local rank. If in doubt, use this option rather than the others.

    **Ex.:** If device 3 lives on NUMA node 1, then the worker whose local rank is 3 will
    only be able to run on the CPUs of NUMA node 1.
    """

    SOCKET = "socket"
    """
    Each worker process and its threads will be bound to all the CPUs on all the NUMA nodes of the
    socket containing the accelerator device whose local index equals the worker's local rank.

    **Ex.:** If socket 0 contains device 3 and NUMA nodes 0-1, then the worker whose
    local rank is 3 will be bound to the CPUs of NUMA nodes 0-1.

    For cases where there is only one NUMA node per socket anyway, this is equivalent to NODE.
    """

    EXCLUSIVE = "exclusive"
    """
    Each worker process and its threads will be bound to an exclusive subset of CPUs
    on the NUMA node containing the accelerator device whose local index equals the worker's
    local rank. The CPUs on the NUMA node are divided evenly among all devices on that node,
    so no two workers share the same CPU cores.

    **Ex.:** If NUMA node 1 has 16 physical cores and devices 2 and 3, then the worker whose
    local rank is 2 will be bound to cores 0-7, and the worker whose local rank is 3 will
    be bound to cores 8-15.
    """

    CORE_COMPLEX = "core-complex"
    """
    Each worker process and its threads will be bound to a single core complex (a group of cores
    sharing the same L3 cache) on the NUMA node containing the accelerator device whose local
    index equals the worker's local rank. Each worker is bound to a different core complex when
    possible.

    **Ex.:** If NUMA node 1 has two core complexes (cores 0-7 sharing one L3 cache, cores 8-15
    sharing another) and devices 2 and 3, then the worker whose local rank is 2 will be bound to
    cores 0-7, and the worker whose local rank is 3 will be bound to cores 8-15.
    """


@dataclass(frozen=True)
class NumaOptions:
    # [CLASS DOC] Frozen dataclass of NUMA binding settings.
    # Fields: affinity_mode (an AffinityMode) and should_fall_back_if_binding_fails (bool,
    # default False) which, when True, makes binding errors get logged instead of raised.
    affinity_mode: AffinityMode

    should_fall_back_if_binding_fails: bool = False
    """
    If ``True``, we will silence any exceptions that occur during NUMA binding itself
    rather than raising them.

    There are no expected exceptions, so avoid using this option. Its purpose is simply
    to mitigate crash risk while conducting mass rollouts of NUMA binding.
    """


def _maybe_wrap_command_args_with_numa_binding(
    command_args: tuple[str, ...],
    *,
    device_index: int,
    numa_options: NumaOptions | None,
) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Prefix a command with numactl pinning to the device's CPUs.
    # Returns the command unchanged when numa_options is None. Otherwise computes the validated
    # CPU set for device_index, builds the numactl-prefixed tuple, and emits a success
    # signpost event. On any exception it goes through _handle_exception (which re-raises
    # unless fallback is enabled) and then returns the original command.
    """
    Wraps command arguments with numactl to apply NUMA CPU binding.

    This function prepends numactl with appropriate CPU affinity flags to the
    provided command arguments, binding the process to CPUs associated with
    the specified accelerator device's NUMA node.

    Args:
        command_args: The original command arguments to wrap.
        device_index: The index of the accelerator device that will be used by the subprocess.
        numa_options: Configuration for NUMA binding behavior. If None, returns
            the original command_args unchanged.

    Returns:
        Tuple of command arguments, potentially wrapped with numactl for NUMA binding.
        Returns the original command_args if numa_options is None or if binding fails
        and fallback is enabled.
    """
    if numa_options is None:
        return command_args

    kwargs = {
        "command_args": command_args,
        "device_index": device_index,
        "numa_options": asdict(numa_options),
    }

    try:
        logical_cpu_indices = _get_validated_logical_cpus_to_bind_to(
            device_index=device_index,
            numa_options=numa_options,
        )

        wrapped_command_args = _assemble_numactl_command_args(
            original_command_args=command_args,
            logical_cpu_indices=logical_cpu_indices,
        )
        signpost_event(
            category="numa_binding",
            name="apply_success",
            parameters={
                **kwargs,
                "wrapped_command": wrapped_command_args,
            },
        )
        return wrapped_command_args
    except Exception:
        # pyrefly: ignore [bad-argument-type]
        _handle_exception(numa_options=numa_options, logger_kwargs=kwargs)
        return command_args


_TParams = ParamSpec("_TParams")
_TReturn = TypeVar("_TReturn")


def _maybe_wrap_with_numa_binding(
    func: Callable[_TParams, _TReturn],
    *,
    device_index: int,
    numa_options: NumaOptions | None,
) -> Callable[_TParams, _TReturn]:
    # [FUNCTION DOC] HELPER: Wrap a callable so NUMA binding is applied before each call.
    # Returns func itself when numa_options is None. Otherwise returns a wrapper (preserving
    # metadata) that first binds the current process threads for device_index and then
    # calls func with the original arguments.
    """
    Wraps a function to apply NUMA CPU binding before execution.

    This decorator applies NUMA CPU affinity to all threads in the current process
    before calling the wrapped function, binding them to CPUs associated with the
    specified accelerator device's NUMA node.

    Args:
        func: The function to wrap with NUMA binding.
        device_index: The index of the accelerator device that will be used.
        numa_options: Configuration for NUMA binding behavior. If None, returns
            the original function unchanged.

    Returns:
        A wrapped function that applies NUMA binding before execution, or the
        original function if numa_options is None.
    """
    if numa_options is None:
        return func

    @wraps(func)
    def wrapped(*args: _TParams.args, **kwargs: _TParams.kwargs) -> _TReturn:
        # [FUNCTION DOC] HELPER: Apply NUMA binding to the current process, then call the wrapped
        # function and return its result.
        _maybe_apply_numa_binding_to_current_process(
            device_index=device_index,
            # pyrefly: ignore [bad-argument-type]
            numa_options=numa_options,
        )
        return func(*args, **kwargs)

    return wrapped


def _maybe_apply_numa_binding_to_current_process(
    *, device_index: int, numa_options: NumaOptions
) -> None:
    # [FUNCTION DOC] HELPER: Bind every thread of the current process to the device's CPUs.
    # Computes and validates the CPU set, sets affinity for all threads, and emits a success
    # signpost event. Exceptions are routed to _handle_exception (re-raised unless fallback
    # is enabled). Returns None.
    kwargs = {
        "device_index": device_index,
        "numa_options": asdict(numa_options),
    }

    try:
        logical_cpu_indices = _get_validated_logical_cpus_to_bind_to(
            device_index=device_index,
            numa_options=numa_options,
        )

        _bind_all_threads_in_current_process_to_logical_cpus(
            logical_cpu_indices=logical_cpu_indices
        )

        signpost_event(
            category="numa_binding",
            name="apply_success",
            parameters={
                **kwargs,
                "logical_cpu_indices": _get_ranges_str_from_ints(logical_cpu_indices),
            },
        )
    except Exception:
        # pyrefly: ignore [bad-argument-type]
        _handle_exception(numa_options=numa_options, logger_kwargs=kwargs)


def _assemble_numactl_command_args(
    *, original_command_args: tuple[str, ...], logical_cpu_indices: set[int]
) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Build the argument tuple running a command under numactl.
    # Result is numactl, a physcpubind option listing the CPUs as a compact range string,
    # then the original arguments.
    return (
        "numactl",
        f"--physcpubind={_get_ranges_str_from_ints(logical_cpu_indices)}",
        *original_command_args,
    )


def _handle_exception(
    *, numa_options: NumaOptions, logger_kwargs: dict[str, object]
) -> None:
    # [FUNCTION DOC] HELPER: Record and handle a failed NUMA binding; call only inside an except.
    # Emits an exception signpost event with the traceback and logs the failure. If fallback is
    # enabled it logs a warning and returns; otherwise it re-raises the active exception.
    signpost_event(
        category="numa_binding",
        name="apply_exception",
        parameters={
            **logger_kwargs,
            "traceback": traceback.format_exc(),
        },
    )
    logger.exception("Failed to apply NUMA binding for input=%r", logger_kwargs)
    if numa_options.should_fall_back_if_binding_fails:
        logger.warning(
            "Continuing executing without applying NUMA binding, despite exception %s",
            traceback.format_exc(),
        )
        return
    # This function is called within an except block, so silence the warning
    # about raise without an exception.
    raise  # noqa: PLE0704


def _get_validated_logical_cpus_to_bind_to(
    *,
    device_index: int,
    numa_options: NumaOptions,
) -> set[int]:
    # [FUNCTION DOC] HELPER: Compute the CPU set for a device and check it is usable.
    # Returns the set of logical CPU indices; raises RuntimeError (from the validator) if
    # numactl is missing or the set is empty.
    logical_cpu_indices = _get_logical_cpus_to_bind_to(
        device_index=device_index, numa_options=numa_options
    )
    _raise_if_binding_invalid(logical_cpu_indices=logical_cpu_indices)

    return logical_cpu_indices


def _raise_if_binding_invalid(*, logical_cpu_indices: set[int]) -> None:
    # [FUNCTION DOC] HELPER: Raise RuntimeError if the numactl tool is not on PATH or the CPU set
    # is empty.
    # NOTE: numactl CLI is only actually necessary for the str entrypoint path,
    # but for simplicity we will just check it no matter what.
    if shutil.which("numactl") is None:
        raise RuntimeError("numactl CLI is required for NUMA binding")

    if not logical_cpu_indices:
        raise RuntimeError("Must bind to a non-empty set of CPU indices")


def _bind_all_threads_in_current_process_to_logical_cpus(
    *, logical_cpu_indices: set[int]
) -> None:
    # [FUNCTION DOC] HELPER: Set CPU affinity of all threads in this process (Linux only).
    # First binds the calling thread (errors propagate). Then walks the task entries of the
    # process and sets affinity for each thread whose current affinity still equals the
    # calling thread's original affinity (threads already customized are left alone).
    # Per-thread errors, for example exited threads, are ignored.
    # Save the original affinity of the main thread before changing it
    # pyrefly: ignore [missing-attribute]
    original_main_thread_affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]

    # 0 represents the current thread.
    # This is outside the try/except because the main thread should always bind successfully.
    # pyrefly: ignore [missing-attribute]
    os.sched_setaffinity(0, logical_cpu_indices)  # type: ignore[attr-defined]

    for tid_str in os.listdir("/proc/self/task"):
        try:
            tid = int(tid_str)
            # pyrefly: ignore [missing-attribute]
            tid_affinity = os.sched_getaffinity(tid)  # type: ignore[attr-defined]

            # Defensive check to ensure we do not overwrite affinity on any threads
            # that have already had their affinity set elsewhere.
            if tid_affinity == original_main_thread_affinity:
                # pyrefly: ignore [missing-attribute]
                os.sched_setaffinity(tid, logical_cpu_indices)  # type: ignore[attr-defined]
        except Exception:
            # Thread may have exited or otherwise become invalid
            pass


def _get_logical_cpus_to_bind_to(
    *,
    device_index: int,
    numa_options: NumaOptions,
) -> set[int]:
    # [FUNCTION DOC] HELPER: Dispatch on the affinity mode to choose CPUs for a device.
    # Returns the set of logical CPU indices from the strategy function matching the mode.
    # Raises ValueError for an unsupported mode.
    """
    Args:
        device_index: The index of the accelerator device that will be used by the subprocess.
            Example: 0
        numa_options: See NumaOptions for details.

    Returns:
        Set of logical CPU indices to bind to.
    """
    if numa_options.affinity_mode == AffinityMode.NODE:
        logical_cpus = _node_get_logical_cpus_to_bind_to(device_index=device_index)
    elif numa_options.affinity_mode == AffinityMode.SOCKET:
        logical_cpus = _socket_get_logical_cpus_to_bind_to(device_index=device_index)
    elif numa_options.affinity_mode == AffinityMode.EXCLUSIVE:
        logical_cpus = _exclusive_get_logical_cpus_to_bind_to(device_index=device_index)
    elif numa_options.affinity_mode == AffinityMode.CORE_COMPLEX:
        logical_cpus = _core_complex_get_logical_cpus_to_bind_to(
            device_index=device_index
        )
    else:
        raise ValueError(f"Affinity mode {numa_options.affinity_mode} not supported.")

    return logical_cpus


def _node_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    # [FUNCTION DOC] HELPER: Node strategy: all allowed CPUs of the device's NUMA node.
    """
    Core logic of 'node' numa strategy.
    """
    numa_node_index = _get_numa_node_index_for_device_index(device_index=device_index)

    return _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )


def _socket_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    # [FUNCTION DOC] HELPER: Socket strategy: union of allowed CPUs over all NUMA nodes that share
    # a socket with the NUMA node of the device.
    """
    Core logic of 'socket' numa strategy.
    """
    numa_node_index_of_device = _get_numa_node_index_for_device_index(
        device_index=device_index
    )
    socket_index = _get_socket_index_for_numa_node(
        numa_node_index=numa_node_index_of_device
    )
    numa_node_indices = _get_numa_node_indices_for_socket_index(
        socket_index=socket_index
    )

    logical_cpus = set()
    for numa_node_index in numa_node_indices:
        logical_cpus.update(
            _get_allowed_logical_cpu_indices_for_numa_node(
                numa_node_index=numa_node_index
            )
        )

    return logical_cpus


def _exclusive_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    # [FUNCTION DOC] HELPER: Exclusive strategy: give each device on the node its own cores.
    # Groups the node's allowed logical CPUs by physical core, orders cores, and splits them
    # into contiguous slices across the node's sorted device indices (leftover cores go one
    # each to the first devices). Returns the logical CPUs of this device's slice. Raises
    # RuntimeError if there are fewer physical cores than devices on the node.
    """
    Core logic of 'exclusive' numa strategy.
    """
    numa_node_index = _get_numa_node_index_for_device_index(device_index=device_index)

    dev_indices = _get_device_indices_for_numa_node(numa_node_index=numa_node_index)
    dev_indices = sorted(dev_indices)
    original_device_relative_index = dev_indices.index(device_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the physical core to
    # represent the physical core.
    physical_core_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpu_indices_sharing_same_physical_core_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )
    # Sort the dict for consistency (dicts maintain order in Python)
    physical_core_to_allowed_logical_cpu_indices = dict(
        sorted(physical_core_to_allowed_logical_cpu_indices.items())
    )

    num_physical_cores_per_device = len(
        physical_core_to_allowed_logical_cpu_indices
    ) // len(dev_indices)
    num_devices_to_give_one_extra_physical_core = len(
        physical_core_to_allowed_logical_cpu_indices
    ) % len(dev_indices)

    if num_physical_cores_per_device < 1:
        raise RuntimeError(
            f"There are only {len(physical_core_to_allowed_logical_cpu_indices)} physical cores on {numa_node_index=},"
            + f" but there are {len(dev_indices)} devices associated with this NUMA node."
        )

    start = original_device_relative_index * num_physical_cores_per_device + min(
        original_device_relative_index, num_devices_to_give_one_extra_physical_core
    )
    end = (
        start
        + num_physical_cores_per_device
        + (
            1
            if original_device_relative_index
            < num_devices_to_give_one_extra_physical_core
            else 0
        )
    )

    # Slice and flatten the logical CPUs from the selected physical cores
    logical_cpu_indices_for_device = {
        logical_cpu_index
        for logical_cpu_indices in list(
            physical_core_to_allowed_logical_cpu_indices.values()
        )[start:end]
        for logical_cpu_index in logical_cpu_indices
    }

    return logical_cpu_indices_for_device


def _core_complex_get_logical_cpus_to_bind_to(*, device_index: int) -> set[int]:
    # [FUNCTION DOC] HELPER: Core-complex strategy: give each device one last-level-cache group.
    # Groups the node's allowed CPUs by the CPUs sharing their highest-level data or unified
    # cache, orders groups by size (largest first, then lowest index), and picks the group by
    # the device's rank among the node's devices, wrapping around if there are more devices
    # than groups. Returns that group's CPU set.
    # KNOWN ISSUE: a CPU with no readable cache data yields an empty sharing set, which makes
    # the group-key computation fail with an error from taking the minimum of an empty set.
    """
    Core logic of 'core-complex' numa strategy.

    Each accelerator device is assigned a full core complex (group of cores sharing
    L3 cache) within its affined NUMA node.
    """
    numa_node_index = _get_numa_node_index_for_device_index(device_index=device_index)

    dev_indices = _get_device_indices_for_numa_node(numa_node_index=numa_node_index)
    dev_indices = sorted(dev_indices)
    original_device_relative_index = dev_indices.index(device_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the max level cache
    # to represent the max level cache.
    max_level_cache_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpus_sharing_same_max_level_cache_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )

    max_level_cache_to_allowed_logical_cpu_indices = dict(
        sorted(
            max_level_cache_to_allowed_logical_cpu_indices.items(),
            # First, prioritize caches with more available cpus
            # Second, prioritize lower index cpus (just for clarity/consistency)
            key=lambda item: (-len(item[1]), item[0]),
        )
    )

    cache_index_for_device = original_device_relative_index % len(
        max_level_cache_to_allowed_logical_cpu_indices
    )
    logical_cpu_indices_for_device = list(
        max_level_cache_to_allowed_logical_cpu_indices.values()
    )[cache_index_for_device]

    return logical_cpu_indices_for_device


K = TypeVar("K")
V = TypeVar("V")


def _group_by(values: Iterable[V], get_key: Callable[[V], K]) -> dict[K, set[V]]:
    # [FUNCTION DOC] HELPER: Group values by a key function into a mapping from key to set.
    """
    Groups elements with same key into sets.
    """
    key_to_values: defaultdict[K, set[V]] = defaultdict(set)
    for value in values:
        key = get_key(value)
        key_to_values[key].add(value)
    return key_to_values


def _get_logical_cpu_indices_sharing_same_physical_core_as(
    *, logical_cpu_index: int
) -> set[int]:
    # [FUNCTION DOC] HELPER: Read from sysfs the hyperthread sibling CPUs of a CPU.
    # Returns a set of CPU indices (including the CPU). Raises a file error if sysfs is missing.
    thread_siblings_list_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/thread_siblings_list"
    )
    with open(thread_siblings_list_absolute_path) as f:
        return _get_set_of_int_from_ranges_str(f.read())


def _get_logical_cpus_sharing_same_max_level_cache_as(
    *, logical_cpu_index: int
) -> set[int]:
    # [FUNCTION DOC] HELPER: Find the CPUs sharing the highest-level data/unified cache with a CPU.
    # Scans the cache entries under sysfs for the CPU, skipping instruction caches, keeps the
    # entry with the highest level, and returns its shared CPU set (empty if none found).
    cpu_cache_dir_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache"
    )

    max_level = -1
    logical_cpus_sharing_max_level_cache = set()
    for entry in os.listdir(cpu_cache_dir_absolute_path):
        if not entry.startswith("index") or not entry[5:].isdecimal():
            continue
        cache_index_absolute_path = os.path.join(cpu_cache_dir_absolute_path, entry)

        # Filter out other cache types like Instruction
        type_absolute_path = os.path.join(cache_index_absolute_path, "type")
        with open(type_absolute_path) as type_file:
            if type_file.read().strip() not in {"Unified", "Data"}:
                continue

        level_absolute_path = os.path.join(cache_index_absolute_path, "level")
        with open(level_absolute_path) as level_file:
            level = int(level_file.read())
        if level <= max_level:
            continue

        max_level = level
        shared_cpu_list_absolute_path = os.path.join(
            cache_index_absolute_path, "shared_cpu_list"
        )
        with open(shared_cpu_list_absolute_path) as share_cpu_list_file:
            logical_cpus_sharing_max_level_cache = _get_set_of_int_from_ranges_str(
                share_cpu_list_file.read()
            )

    return logical_cpus_sharing_max_level_cache


def _get_allowed_logical_cpu_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    # [FUNCTION DOC] HELPER: CPUs of a NUMA node intersected with the current thread's allowed set.
    all_cpu_indices = _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
        numa_node_index=numa_node_index
    )
    allowed_cpu_indices = _get_allowed_cpu_indices_for_current_thread()
    return all_cpu_indices & allowed_cpu_indices


def _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
    *, numa_node_index: int
) -> set[int]:
    # [FUNCTION DOC] HELPER: Read all CPU indices of a NUMA node from sysfs, ignoring affinity.
    # Raises RuntimeError if the node's cpulist file does not exist.
    """
    Returns:
        Indices of all CPUs associated with numa_node_index. However, the list
        is not filtered based on whether the thread is allowed to use them.
    """
    cpulist_absolute_path = f"/sys/devices/system/node/node{numa_node_index}/cpulist"
    try:
        with open(cpulist_absolute_path) as f:
            cpu_range_str = f.read()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not determine CPUs corresponding to {numa_node_index=}."
        ) from e
    return _get_set_of_int_from_ranges_str(cpu_range_str)


def _get_device_count() -> int:
    # [FUNCTION DOC] HELPER: Return the number of accelerator devices.
    return torch.accelerator.device_count()


def _get_numa_node_index_for_device_index(*, device_index: int) -> int:
    # [FUNCTION DOC] HELPER: Map an accelerator device index to its NUMA node via PCI sysfs info.
    # Raises RuntimeError if no accelerator is available. Builds the PCI address from the
    # device properties and reads the numa_node file; negative values (single-node systems)
    # become 0. File errors propagate.
    if not torch.accelerator.is_available():
        raise RuntimeError("No accelerator available for NUMA binding")
    acc = torch.accelerator.current_accelerator()
    device_module = torch.get_device_module(acc)
    device_properties = device_module.get_device_properties(device_index)

    domain = device_properties.pci_domain_id  # type: ignore[attr-defined]
    bus = device_properties.pci_bus_id  # type: ignore[attr-defined]
    device = device_properties.pci_device_id  # type: ignore[attr-defined]

    # Format to sysfs PCI address: "0000:dc:00.0"
    pci_addr = f"{domain:04x}:{bus:02x}:{device:02x}.0"

    pci_numa_node_absolute_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    with open(pci_numa_node_absolute_path) as f:
        # In systems with only one NUMA node, this will
        # often be saved as -1. In those cases, there is obviously
        # at least one numa node, 0, so we use that.
        return max(int(f.read().strip()), 0)


def _get_device_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    # [FUNCTION DOC] HELPER: Device indices whose PCI-reported NUMA node equals the given one.
    return {
        device_index
        for device_index in range(_get_device_count())
        if _get_numa_node_index_for_device_index(device_index=device_index)
        == numa_node_index
    }


def _get_socket_index_for_numa_node(*, numa_node_index: int) -> int:
    # [FUNCTION DOC] HELPER: Socket index of a NUMA node, taken from its lowest allowed CPU.
    # KNOWN ISSUE: raises an error from taking the minimum of an empty set if the node has no
    # CPUs allowed for this process.
    arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
        numa_node_index=numa_node_index
    )

    return _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index)


def _get_socket_index_for_cpu(*, cpu_index: int) -> int:
    # [FUNCTION DOC] HELPER: Read a CPU's physical package (socket) id from sysfs.
    # Raises RuntimeError if the topology file does not exist.
    package_id_absolute_path = (
        f"/sys/devices/system/cpu/cpu{cpu_index}/topology/physical_package_id"
    )
    try:
        with open(package_id_absolute_path) as f:
            return int(f.read().strip())
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not determine socket for {cpu_index=}") from e


def _get_arbitrary_allowed_cpu_index_for_numa_node(*, numa_node_index: int) -> int:
    # [FUNCTION DOC] HELPER: Lowest allowed CPU index in a NUMA node (raises if none are allowed).
    return min(
        _get_allowed_logical_cpu_indices_for_numa_node(numa_node_index=numa_node_index)
    )


def _get_set_of_int_from_ranges_str(ranges_str: str) -> set[int]:
    # [FUNCTION DOC] HELPER: Parse a sysfs range list such as zero-to-two, four, six-to-seven.
    # Splits on commas, ignores empty items, expands inclusive ranges, and returns a set of
    # ints. Raises ValueError on malformed numbers.
    """
    Util for parsing a string of int ranges, as in a sysfs file.

    Args:
        ranges_str: E.g., "0-2,4,6-7"

    Returns:
        E.g., {0, 1, 2, 4, 6, 7}
    """
    ints: set[int] = set()
    for range_str in ranges_str.split(","):
        range_str = range_str.strip()
        if not range_str:
            continue
        if "-" in range_str:
            start_str, end_str = range_str.split("-")
            start, end = int(start_str), int(end_str)
            ints.update(range(start, end + 1))
        else:
            ints.add(int(range_str))
    return ints


def _get_ranges_str_from_ints(ints: Iterable[int]) -> str:
    # [FUNCTION DOC] HELPER: Format integers as a compact comma-separated range string.
    # Sorts the values and collapses consecutive runs into start-end items; empty input gives
    # an empty string. Inverse of the range-string parser.
    """
    Convert a set of integers to a compact string with ranges.

    Args:
        ints: E.g., {0, 1, 2, 4, 6, 7}

    Returns:
        E.g., "0-2,4,6-7"
    """
    if not ints:
        return ""

    sorted_ints = sorted(ints)
    ranges = []
    start = prev = sorted_ints[0]

    for num in sorted_ints[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = num

    # Append the last range
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


def _get_systemwide_numa_node_indices() -> set[int]:
    # [FUNCTION DOC] HELPER: Read the set of possible NUMA node indices on this system from sysfs.
    with open("/sys/devices/system/node/possible") as f:
        possible_nodes_str = f.read()

    return _get_set_of_int_from_ranges_str(possible_nodes_str)


def _get_numa_node_indices_for_socket_index(*, socket_index: int) -> set[int]:
    # [FUNCTION DOC] HELPER: NUMA nodes on the system whose socket equals the given socket index.
    # Determines each node's socket from its lowest allowed CPU.
    systemwide_numa_node_indices = _get_systemwide_numa_node_indices()

    matching_numa_node_indices = set()
    for numa_node_index in systemwide_numa_node_indices:
        arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
            numa_node_index=numa_node_index
        )
        if socket_index == _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index):
            matching_numa_node_indices.add(numa_node_index)

    return matching_numa_node_indices


def _get_allowed_cpu_indices_for_current_thread() -> set[int]:
    # [FUNCTION DOC] HELPER: Return the CPU affinity set of the current thread (Linux only).
    # 0 denotes current thread
    # pyrefly: ignore [missing-attribute]
    return os.sched_getaffinity(0)  # type:ignore[attr-defined]
