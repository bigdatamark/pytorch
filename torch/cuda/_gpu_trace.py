# [MODULE DOC] Registry of process-wide callback lists used by the CUDA GPU sanitizer/tracing hooks.
# Defines one CallbackRegistry per CUDA event kind (event creation, deletion, record, wait; memory
# allocation/deallocation; stream creation; device, stream and event synchronization) and a
# register_callback_for_* function for each that appends a callback to the matching registry.
# Callbacks receive integer handles (event/stream ids or pointers); device synchronization takes no args.
# Import-time side effect: creates the ten registry objects. No other state.
from collections.abc import Callable

from torch._utils import CallbackRegistry


EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event creation"
)
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event deletion"
)
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event record"
)
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry("CUDA event wait")
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory allocation"
)
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory deallocation"
)
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream creation"
)
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "CUDA device synchronization"
)
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream synchronization"
)
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event synchronization"
)


def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA event creation (callback args: event ids).
    # Appends the callback to the module-level registry; returns nothing.
    EventCreationCallbacks.add_callback(cb)


def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA event deletion (callback args: event id).
    # Appends the callback to the module-level registry; returns nothing.
    EventDeletionCallbacks.add_callback(cb)


def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA event record (callback args: event id and stream id).
    # Appends the callback to the module-level registry; returns nothing.
    EventRecordCallbacks.add_callback(cb)


def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA event wait (callback args: event id and stream id).
    # Appends the callback to the module-level registry; returns nothing.
    EventWaitCallbacks.add_callback(cb)


def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA memory allocation (callback args: pointer).
    # Appends the callback to the module-level registry; returns nothing.
    MemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA memory deallocation (callback args: pointer).
    # Appends the callback to the module-level registry; returns nothing.
    MemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA stream creation (callback args: stream id).
    # Appends the callback to the module-level registry; returns nothing.
    StreamCreationCallbacks.add_callback(cb)


def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA device synchronization (callback args: no arguments).
    # Appends the callback to the module-level registry; returns nothing.
    DeviceSynchronizationCallbacks.add_callback(cb)


def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA stream synchronization (callback args: stream id).
    # Appends the callback to the module-level registry; returns nothing.
    StreamSynchronizationCallbacks.add_callback(cb)


def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Register a callback to be invoked on CUDA event synchronization (callback args: event id).
    # Appends the callback to the module-level registry; returns nothing.
    EventSynchronizationCallbacks.add_callback(cb)
