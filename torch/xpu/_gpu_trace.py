# [MODULE DOC] HELPER: Global callback registries for tracing XPU runtime events (event
# creation/deletion/record/wait, memory allocation/deallocation, stream creation, and
# device/stream/event synchronization), plus register_callback_for_* functions that add
# a callable to the matching registry. Import time creates the registry objects only.
from collections.abc import Callable

from torch._utils import CallbackRegistry


EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry("XPU event creation")
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry("XPU event deletion")
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "XPU event record"
)
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry("XPU event wait")
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU memory allocation"
)
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU memory deallocation"
)
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU stream creation"
)
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "XPU device synchronization"
)
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU stream synchronization"
)
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU event synchronization"
)


def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the event creation registry; cb receives the event id.
    EventCreationCallbacks.add_callback(cb)


def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the event deletion registry; cb receives the event id.
    EventDeletionCallbacks.add_callback(cb)


def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the event record registry; cb receives event id and stream.
    EventRecordCallbacks.add_callback(cb)


def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the event wait registry; cb receives event id and stream.
    EventWaitCallbacks.add_callback(cb)


def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the memory allocation registry; cb receives the pointer.
    MemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the memory deallocation registry; cb receives the pointer.
    MemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the stream creation registry; cb receives the stream id.
    StreamCreationCallbacks.add_callback(cb)


def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    # [FUNCTION DOC] Adds cb to the device synchronization registry; cb takes no arguments.
    DeviceSynchronizationCallbacks.add_callback(cb)


def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the stream synchronization registry; cb receives the stream.
    StreamSynchronizationCallbacks.add_callback(cb)


def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    # [FUNCTION DOC] Adds cb to the event synchronization registry; cb receives the event id.
    EventSynchronizationCallbacks.add_callback(cb)
