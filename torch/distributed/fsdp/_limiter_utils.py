# [MODULE DOC] Rate-limiting helper for FSDP: defines _FreeEventQueue, a FIFO of
# CUDA/device free events used to bound the number of in-flight all-gathers.
# No import-time side effects beyond importing collections and torch.
import collections

import torch


class _FreeEventQueue:
    # [CLASS DOC] HELPER: FIFO queue (a deque) of torch events, one per pending
    # unsharded-parameter free tied to an in-flight all-gather. Holds the deque
    # and a fixed limit of 2 in-flight all-gathers. Callers enqueue after each
    # all-gather and dequeue (at most one) once the limit is reached.
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates an empty event deque and sets the
        # maximum number of in-flight all-gathers to 2 (empirically chosen).
        self._queue: collections.deque[torch.Event] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen

    def enqueue(self, free_event: torch.Event) -> None:
        # [FUNCTION DOC] HELPER: Appends the given free event to the back of
        # the queue. Mutates the queue; returns nothing.
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> torch.Event | None:
        # [FUNCTION DOC] HELPER: If the queue length has reached the in-flight
        # limit, removes and returns the oldest event; otherwise returns None
        # and leaves the queue unchanged.
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> torch.Event | None:
        # [FUNCTION DOC] HELPER: Pops and returns the oldest event, or returns
        # None when the queue is empty. Does not raise.
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None
