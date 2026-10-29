# mypy: allow-untyped-defs
# pylint: disable=useless-parent-delegation
# [MODULE DOC] Python wrappers Stream and Event over the C++ XPU stream/event base classes.
# At import time, if the C extension lacks those base classes, registers dummy placeholder
# base classes in the torch._C namespace so the module can still be imported.
from __future__ import annotations

import ctypes

import torch
from torch._utils import _dummy_type


if not hasattr(torch._C, "_XpuStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_XpuStreamBase"] = _dummy_type("_XpuStreamBase")
    torch._C.__dict__["_XpuEventBase"] = _dummy_type("_XpuEventBase")


class Stream(torch._C._XpuStreamBase):
    # [CLASS DOC] A XPU stream: an ordered queue of work on one device. Usable as a context
    # manager target for stream guards, hashable and comparable (equal only to other Stream
    # objects), and convertible to a ctypes pointer of the SYCL queue. Key methods: wait_event,
    # wait_stream, record_event, query, synchronize.
    r"""Wrapper around a XPU stream.

    A XPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. By default, the priority is set to 0.
            If the value falls outside of the allowed priority range, it will automatically be
            mapped to the nearest valid priority (lowest for large positive numbers or
            highest for large negative numbers).
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # [FUNCTION DOC] Creates a stream with the given priority. If device is None, or the
        # keyword arguments already carry both stream_id and device_index (wrapping an
        # existing stream), constructs directly; otherwise constructs inside a device context
        # for device so the stream lands on that device. Extra keyword arguments are forwarded.
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.xpu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event: Event | torch.Event) -> None:
        # [FUNCTION DOC] Makes future work on this stream wait for event (delegates to the
        # event's wait method with this stream).
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (Event, torch.Event): an event to wait for.
        """
        event.wait(self)

    def wait_stream(self, stream: Stream | torch.Stream) -> None:
        # [FUNCTION DOC] Makes future work on this stream wait for all work currently submitted
        # to stream, by recording an event on stream and waiting on it.
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream, torch.Stream): a stream to synchronize.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event: Event | torch.Event | None = None):
        # [FUNCTION DOC] Records event on this stream (a new Event is allocated when none is
        # given) and returns the recorded event.
        r"""Record an event.

        Args:
            event (Event, torch.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        # [FUNCTION DOC] Returns True if all work submitted to this stream has completed.
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self) -> None:
        # [FUNCTION DOC] Blocks the calling thread until all work on this stream completes.
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    @property
    def _as_parameter_(self):
        # [FUNCTION DOC] Property used by ctypes: returns the SYCL queue handle as a void pointer.
        return ctypes.c_void_p(self.sycl_queue)

    def __eq__(self, o):
        # [FUNCTION DOC] Equal only if o is a Stream and the base class considers them equal;
        # any other type compares unequal.
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        # [FUNCTION DOC] Hash combining the SYCL queue handle and the device.
        return hash((self.sycl_queue, self.device))

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns text with the device and the SYCL queue address in hex.
        return f"torch.xpu.Stream(device={self.device} sycl_queue={self.sycl_queue:#x})"


class Event(torch._C._XpuEventBase):
    # [CLASS DOC] A XPU event: a synchronization marker, optionally timing-enabled, lazily
    # created on first record. Only streams on the event's device may record it, but any
    # device's streams may wait on it. Key methods: record, wait, query, elapsed_time,
    # synchronize.
    r"""Wrapper around a XPU event.

    XPU events are synchronization markers that can be used to monitor the
    device's progress, and to synchronize XPU streams.

    The underlying XPU events are lazily initialized when the event is first
    recorded. After creation, only streams on the same device may record the
    event. However, streams on any device can wait on the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False):
        # [FUNCTION DOC] Creates an event; enable_timing selects whether it can measure time.
        return super().__new__(cls, enable_timing=enable_timing)

    def record(self, stream: Stream | torch.Stream | None = None) -> None:
        # [FUNCTION DOC] Records this event on stream, or on the current XPU stream if None.
        # The stream's device must match the event's device (checked by the base class).
        r"""Record the event in a given stream.

        Args:
            stream (Stream, torch.Stream, optional): Uses ``torch.xpu.current_stream()`` if no stream is specified.
                The stream's device must match the event's device.
        """
        if stream is None:
            stream = torch.xpu.current_stream()
        super().record(stream)

    def wait(self, stream: Stream | torch.Stream | None = None) -> None:
        # [FUNCTION DOC] Makes future work on stream (default: current XPU stream) wait
        # for this event.
        r"""Make all future work submitted to the given stream wait for this event.

        Args:
            stream (Stream, torch.Stream, optional): Uses ``torch.xpu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = torch.xpu.current_stream()
        super().wait(stream)

    def query(self) -> bool:
        # [FUNCTION DOC] Returns True if all work captured by this event has completed.
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event: Event):
        # [FUNCTION DOC] Returns milliseconds between this event and end_event, both of which
        # must have been recorded (and timing enabled); errors come from the base class.
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.

        Args:
            end_event (Event): the end event.
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        # [FUNCTION DOC] Blocks the calling thread until all work captured by this event ends.
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        super().synchronize()

    @property
    def _as_parameter_(self):
        # [FUNCTION DOC] Property used by ctypes: returns the SYCL event handle as a void pointer.
        return ctypes.c_void_p(self.sycl_event)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns text with the SYCL event address in hex, or a marker
        # saying the event is uninitialized when it has no underlying event yet.
        if self.sycl_event:
            return f"torch.xpu.Event(sycl_event={self.sycl_event:#x})"
        else:
            return "torch.xpu.Event(uninitialized)"
