# Owner(s): ["module: cuda"]
# [MODULE DOC] Tests for the CUDA GPU trace hooks in torch.cuda._gpu_trace. Each test
# registers a mock callback for one event kind (event, memory, stream or synchronization) and
# checks it is invoked with the expected handle or pointer. Import-time: prints a message and
# swaps TestCase for NoTest when CUDA is unavailable.

import sys
import unittest
import unittest.mock

import torch
import torch.cuda._gpu_trace as gpu_trace
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase


# NOTE: Each test needs to be run in a brand new process, to reset the registered hooks
# and make sure the CUDA streams are initialized for each test that uses them.

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaTrace(TestCase):
    # [CLASS DOC] Test case holding one test per GPU trace callback kind. Each test relies on
    # a fresh process because registered hooks are global. self.mock is a MagicMock created in
    # setUp and used as the callback.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, activates GPU tracing in the C++ layer, and
        # creates the mock callback stored on self.mock.
        super().setUp()
        torch._C._activate_gpu_trace()
        self.mock = unittest.mock.MagicMock()

    def test_event_creation_callback(self):
        # [FUNCTION DOC] Registers the mock for event creation, records a new CUDA event, and
        # checks the mock was called once with the event's raw handle value.
        gpu_trace.register_callback_for_event_creation(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_event_deletion_callback(self):
        # [FUNCTION DOC] Registers the mock for event deletion, records and then deletes an
        # event, and checks the mock was called once with the deleted event's handle value.
        gpu_trace.register_callback_for_event_deletion(self.mock)

        event = torch.cuda.Event()
        event.record()
        event_id = event._as_parameter_.value
        del event
        self.mock.assert_called_once_with(event_id)

    def test_event_record_callback(self):
        # [FUNCTION DOC] Registers the mock for event record, records an event, and checks the
        # mock got the event handle and the default stream's raw handle.
        gpu_trace.register_callback_for_event_record(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )

    def test_event_wait_callback(self):
        # [FUNCTION DOC] Registers the mock for event wait, records then waits on an event,
        # and checks the mock got the event handle and the default stream's raw handle.
        gpu_trace.register_callback_for_event_wait(self.mock)

        event = torch.cuda.Event()
        event.record()
        event.wait()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )

    def test_memory_allocation_callback(self):
        # [FUNCTION DOC] Registers the mock for memory allocation, allocates a CUDA tensor,
        # and checks the mock was called once with the tensor's data pointer.
        gpu_trace.register_callback_for_memory_allocation(self.mock)

        tensor = torch.empty(10, 4, device="cuda")
        self.mock.assert_called_once_with(tensor.data_ptr())

    def test_memory_deallocation_callback(self):
        # [FUNCTION DOC] Registers the mock for memory deallocation, frees a CUDA tensor, and
        # checks the mock was called once with the freed data pointer.
        gpu_trace.register_callback_for_memory_deallocation(self.mock)

        tensor = torch.empty(3, 8, device="cuda")
        data_ptr = tensor.data_ptr()
        del tensor
        self.mock.assert_called_once_with(data_ptr)

    def test_stream_creation_callback(self):
        # [FUNCTION DOC] Registers the mock for stream creation and creates a stream. On HIP
        # the stream is lazily created, so work is run on it to force creation. Checks the
        # mock was called.
        gpu_trace.register_callback_for_stream_creation(self.mock)

        # see Note [HIP Lazy Streams]
        if torch.version.hip:
            user_stream = torch.cuda.Stream()
            with torch.cuda.stream(user_stream):
                torch.ones(5, device="cuda")
        else:
            torch.cuda.Stream()

        self.mock.assert_called()

    def test_device_synchronization_callback(self):
        # [FUNCTION DOC] Registers the mock for device synchronization, calls the device-wide
        # synchronize, and checks the mock was called.
        gpu_trace.register_callback_for_device_synchronization(self.mock)

        torch.cuda.synchronize()
        self.mock.assert_called()

    def test_stream_synchronization_callback(self):
        # [FUNCTION DOC] Registers the mock for stream synchronization, synchronizes a new
        # stream, and checks the mock was called once with that stream's raw handle.
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        stream = torch.cuda.Stream()
        stream.synchronize()
        self.mock.assert_called_once_with(stream.cuda_stream)

    def test_event_synchronization_callback(self):
        # [FUNCTION DOC] Registers the mock for event synchronization, records and
        # synchronizes an event, and checks the mock was called once with the
        # event's handle value.
        gpu_trace.register_callback_for_event_synchronization(self.mock)

        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_memcpy_synchronization(self):
        # [FUNCTION DOC] Registers the mock for stream synchronization, then runs nonzero on a
        # CUDA tensor, which does an implicit device-to-host copy. Checks the mock was called
        # once with the default stream's raw handle.
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        tensor = torch.rand(5, device="cuda")
        tensor.nonzero()
        self.mock.assert_called_once_with(torch.cuda.default_stream().cuda_stream)

    def test_all_trace_callbacks_called(self):
        # [FUNCTION DOC] Registers two callbacks (self.mock and a second mock) for memory
        # allocation, allocates a CUDA tensor, and checks that both callbacks were called once
        # with the tensor's data pointer.
        other = unittest.mock.MagicMock()
        gpu_trace.register_callback_for_memory_allocation(self.mock)
        gpu_trace.register_callback_for_memory_allocation(other)

        tensor = torch.empty(10, 4, device="cuda")
        self.mock.assert_called_once_with(tensor.data_ptr())
        other.assert_called_once_with(tensor.data_ptr())


if __name__ == "__main__":
    run_tests()
