# mypy: allow-untyped-defs
# [MODULE DOC] Pinned-memory support for the DataLoader: a thread loop that moves fetched
# batches into page-locked memory, and a recursive pin_memory function that handles tensors and
# nested containers. No import-time side effects.
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **need** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import collections
import copy
import queue

import torch
from torch._utils import ExceptionWrapper

from . import MP_STATUS_CHECK_INTERVAL


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device) -> None:
    # [FUNCTION DOC] HELPER: Body of the pin-memory thread. Sets the thread's torch thread count
    # to one, names the thread and selects the accelerator device index. Then repeats until
    # done_event is set: take an (index, data) pair from in_queue, pin the data unless it is
    # an ExceptionWrapper (a failure while pinning becomes an ExceptionWrapper), and put the
    # pair on out_queue, retrying while full until done_event is set. Returns when done.
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.multiprocessing._set_thread_name("pt_data_pin")
    torch.accelerator.set_device_index(device_id)

    def do_one_step() -> None:
        # [FUNCTION DOC] HELPER: Performs one iteration of the loop: waits up to the status
        # check interval for an input item (returns if none), pins it unless done or already
        # an error, and forwards the result to out_queue, retrying while the queue is full.
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(
                    where=f"in pin memory thread for device {device_id}"
                )
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()


def pin_memory(data, device=None):
    # [FUNCTION DOC] Returns a copy of data with all tensors moved to pinned memory. Objects with
    # their own pin_memory method use it; strings and bytes pass through; mappings, tuples
    # (including named tuples) and sequences are rebuilt recursively, preferring a shallow copy
    # for mutable containers and falling back to a plain dict or list when the type cannot be
    # rebuilt. Other objects are returned unchanged. The device argument is passed down
    # recursively but is not used for tensor pinning.
    # KNOWN ISSUE: device is never applied to Tensor.pin_memory, so it has no effect.
    if isinstance(data, torch.Tensor):
        return data.pin_memory()

    if hasattr(data, "pin_memory"):
        return data.pin_memory()

    if isinstance(data, (str, bytes)):
        return data

    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                # pyrefly: ignore [bad-instantiation]
                return type(data)(
                    # pyrefly: ignore [bad-argument-count]
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}

    if isinstance(data, tuple):
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(*(pin_memory(sample, device) for sample in data))
        return type(data)(pin_memory(sample, device) for sample in data)

    if isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, item in enumerate(data):
                    clone[i] = pin_memory(item, device)
                return clone
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]

    return data
