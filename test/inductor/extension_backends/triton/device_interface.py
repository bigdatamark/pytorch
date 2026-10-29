from __future__ import annotations

# [MODULE DOC] HELPER: Fake device interface for a Triton-based extension backend in tests.
# Provides a properties holder with fixed numbers and a DeviceInterface subclass that has a
# single device (index 0), no real streams, and an Event that measures wall-clock time on the
# host. No import-time side effects.
import time

import torch
from torch._dynamo import device_interface


class DeviceProperties:
    # [CLASS DOC] HELPER: Fixed fake device properties: major compute version 8 (set to bypass
    # an H100 check in the Triton heuristics), one thread per multiprocessor and 80
    # multiprocessors.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the three fixed attributes major,
        # max_threads_per_multi_processor and multi_processor_count to constants; no arguments.
        self.major = 8  # TODO: bypass check for H100 in triton_heuristics.py
        self.max_threads_per_multi_processor = 1
        self.multi_processor_count = 80


class DeviceInterface(device_interface.DeviceInterface):
    # [CLASS DOC] HELPER: Device interface for a fake single-device backend. Nested classes
    # supply Event, device and Worker; static methods report device index 0, one device, no raw
    # stream and zero compute capability, and perform no real synchronization.
    class Event(torch.Event):
        # [CLASS DOC] HELPER: Host-clock event. Records time with the nanosecond performance
        # counter, only when timing was enabled at construction; other operations do nothing.
        def __init__(
            self,
            enable_timing: bool = False,
            blocking: bool = False,
            interprocess: bool = False,
        ) -> None:
            # [FUNCTION DOC] HELPER: Stores the enable_timing flag and initializes the recorded
            # time to None. The blocking and interprocess arguments are accepted but ignored, and
            # the parent constructor is not called.
            self.enable_timing = enable_timing
            self.recorded_time: int | None = None

        def record(self, stream) -> None:
            # [FUNCTION DOC] HELPER: If timing is disabled, returns without doing anything.
            # Otherwise stores the current performance counter time in nanoseconds; raises an
            # AssertionError if this event was already recorded. The stream argument is unused.
            if not self.enable_timing:
                return
            if self.recorded_time is not None:
                raise AssertionError
            self.recorded_time = time.perf_counter_ns()

        def elapsed_time(self, end_event: DeviceInterface.Event) -> float:
            # [FUNCTION DOC] HELPER: Returns milliseconds elapsed from this event to end_event as
            # a float (nanosecond difference divided by one million). Raises an AssertionError if
            # either event has no recorded time (never recorded, or timing disabled).
            if not self.recorded_time:
                raise AssertionError
            if not end_event.recorded_time:
                raise AssertionError
            # convert to ms
            return (end_event.recorded_time - self.recorded_time) / 1000000

        def wait(self, stream) -> None:
            # [FUNCTION DOC] HELPER: No-op wait on a stream.
            pass

        def query(self) -> None:
            # [FUNCTION DOC] HELPER: No-op completion query; returns None.
            pass

        def synchronize(self) -> None:
            # [FUNCTION DOC] HELPER: No-op synchronize.
            pass

    class device:
        # [CLASS DOC] HELPER: Minimal device context holder; stores the given device in an
        # attribute named device and does nothing else (no enter or exit behavior).
        def __init__(self, device) -> None:
            # [FUNCTION DOC] HELPER: Saves the device argument on the instance.
            self.device = device

    class Worker(device_interface.DeviceInterface.Worker):
        # [CLASS DOC] HELPER: Worker hooks of the device interface for the fake backend, all
        # static: no device index is tracked, so the current device is always index 0.
        @staticmethod
        def set_device(device: int) -> None:
            # [FUNCTION DOC] HELPER: Intentionally does nothing; the backend has no device index.
            # No device index for our backend
            pass

        @staticmethod
        def current_device() -> int:
            # [FUNCTION DOC] HELPER: Always returns device index 0.
            # No device index for our backend
            return 0

        @staticmethod
        def get_device_properties(
            device=None,
        ) -> DeviceProperties:
            # [FUNCTION DOC] HELPER: Returns a new properties object with the fixed fake values;
            # the device argument is ignored.
            return DeviceProperties()

    @staticmethod
    def current_device() -> int:
        # [FUNCTION DOC] HELPER: Always returns device index 0.
        return 0

    @staticmethod
    def set_device(device) -> None:
        # [FUNCTION DOC] HELPER: Intentionally does nothing; the device argument is ignored.
        pass

    @staticmethod
    def device_count() -> int:
        # [FUNCTION DOC] HELPER: Always reports exactly one device.
        return 1

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        # [FUNCTION DOC] HELPER: Accepts only device index 0 (raises an AssertionError for any
        # other index) and returns 0 as the previous device.
        if device != 0:
            raise AssertionError(
                f"Only device index 0 is supported, tried to set index to {device}"
            )
        return 0  # previous device is always 0

    @staticmethod
    def exchange_device(device: int) -> int:
        # [FUNCTION DOC] HELPER: Same as maybe_exchange_device: only index 0 is accepted, an
        # AssertionError is raised otherwise, and 0 is returned as the previous device.
        if device != 0:
            raise AssertionError(
                f"Only device index 0 is supported, tried to set index to {device}"
            )
        return 0  # previous device is always 0

    @staticmethod
    def get_raw_stream(device_index: int):
        # [FUNCTION DOC] HELPER: Returns None; there are no real streams.
        return None

    @staticmethod
    def synchronize(device) -> None:
        # [FUNCTION DOC] HELPER: No-op device synchronization.
        pass

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] HELPER: Reports the device as available (always true); tests may patch it.
        return True

    @staticmethod
    def get_compute_capability(device) -> int:
        # [FUNCTION DOC] HELPER: Returns compute capability 0 for any device.
        return 0
