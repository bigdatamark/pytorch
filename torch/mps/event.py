# [MODULE DOC] Defines the Event class, a thin Python wrapper over MPS synchronization events
# held by integer id in the native layer. No import-time side effects.
import torch


class Event:
    # [CLASS DOC] Wrapper around an MPS event identified by an integer id acquired from the
    # native layer at construction and released on deletion. Supports recording, waiting,
    # querying, synchronizing and measuring elapsed time between two events (timing must be
    # enabled at construction for elapsed time to be meaningful).
    r"""Wrapper around an MPS event.

    MPS events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MPS streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __init__(self, enable_timing: bool = False) -> None:
        # [FUNCTION DOC] Acquires a native event id, optionally with timing enabled, and stores it.
        self.__eventId = torch._C._mps_acquireEvent(enable_timing)

    def __del__(self) -> None:
        # [FUNCTION DOC] Releases the native event if the release hook still exists (guards
        # interpreter shutdown) and the id is positive.
        # KNOWN ISSUE: if the constructor failed before storing the id, this raises AttributeError.
        # checks if torch._C is already destroyed
        if hasattr(torch._C, "_mps_releaseEvent") and self.__eventId > 0:
            torch._C._mps_releaseEvent(self.__eventId)

    def record(self) -> None:
        # [FUNCTION DOC] Records this event on the default stream.
        r"""Records the event in the default stream."""
        torch._C._mps_recordEvent(self.__eventId)

    def wait(self) -> None:
        # [FUNCTION DOC] Makes later work on the default stream wait for this event (device side).
        r"""Makes all future work submitted to the default stream wait for this event."""
        torch._C._mps_waitForEvent(self.__eventId)

    def query(self) -> bool:
        # [FUNCTION DOC] Returns True if all work captured by the event has completed.
        r"""Returns True if all work currently captured by event has completed."""
        return torch._C._mps_queryEvent(self.__eventId)

    def synchronize(self) -> None:
        # [FUNCTION DOC] Blocks the calling CPU thread until the event has completed.
        r"""Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        torch._C._mps_synchronizeEvent(self.__eventId)

    def elapsed_time(self, end_event: "Event") -> float:
        # [FUNCTION DOC] Returns milliseconds between this event (start) and end_event, both of
        # which must have been recorded; errors come from the native layer if timing is disabled.
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return torch._C._mps_elapsedTimeOfEvents(self.__eventId, end_event.__eventId)
