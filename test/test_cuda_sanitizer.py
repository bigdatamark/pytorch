# Owner(s): ["module: cuda"]
# [MODULE DOC] Tests for the CUDA sanitizer in torch.cuda._sanitizer: argument parsing of
# operator schemas into read and write pointer sets, the event handler's data-race detection
# over streams, events and synchronization, log and error message text, and subclass
# compatibility. Import-time: prints a message and swaps TestCase for NoTest when CUDA is
# unavailable. Defines small helpers that map integers to data pointer, stream and event ids.

import sys
import textwrap
import traceback

import torch
import torch.cuda._sanitizer as csan
from torch.cuda._sanitizer import DataPtr, EventId, StreamId
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase
from torch.testing._internal.two_tensor import TwoTensor


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


class TestArgumentHandler(TestCase):
    # [CLASS DOC] Tests that the argument handler classifies each tensor's data pointer as read,
    # written or output for several ATen operators, using their schemas.
    def test_add(self):
        # [FUNCTION DOC] For the add operator on two CUDA tensors, checks both inputs are
        # recorded as read and the result as written.
        add_func = torch.ops.aten.add.Tensor
        a = torch.ones(5, 3, device="cuda")
        b = torch.randn(5, 3, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(add_func._schema, (a, b), {}, is_factory=False)
        c = torch.add(a, b)
        argument_handler.parse_outputs(add_func._schema, c, is_factory=False)

        self.assertEqual({a.data_ptr(), b.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({c.data_ptr()}, argument_handler.dataptrs_written)

    def test_cat(self):
        # [FUNCTION DOC] For the cat operator with a list of three tensors, checks all list
        # elements are recorded as read and the result as written.
        cat_func = torch.ops.aten.cat.default
        a = torch.ones(2, 4, 5, device="cuda")
        b = torch.zeros(2, 1, 5, device="cuda")
        c = torch.rand(2, 7, 5, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            cat_func._schema, ([a, b, c], 1), {}, is_factory=False
        )
        d = torch.cat((a, b, c), dim=1)
        argument_handler.parse_outputs(cat_func._schema, d, is_factory=False)

        self.assertEqual(
            {a.data_ptr(), b.data_ptr(), c.data_ptr()}, argument_handler.dataptrs_read
        )
        self.assertEqual({d.data_ptr()}, argument_handler.dataptrs_written)

    def test_split(self):
        # [FUNCTION DOC] For split, a view operation, checks no pointers are recorded as read
        # or written.
        split_func = torch.ops.aten.split.Tensor
        a = torch.arange(10, device="cuda").reshape(5, 2)

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(split_func._schema, (a, 2), {}, is_factory=False)
        out = torch.split(a, 2)
        argument_handler.parse_outputs(split_func._schema, out, is_factory=False)

        # Split is a view op, no data is read or written!
        self.assertEqual(len(argument_handler.dataptrs_read), 0)
        self.assertEqual(len(argument_handler.dataptrs_written), 0)

    def test_inplace(self):
        # [FUNCTION DOC] For the in-place add, checks nothing is read and the mutated tensor's
        # pointer is written.
        add_inplace_func = torch.ops.aten.add_.Tensor
        a = torch.rand(4, 2, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            add_inplace_func._schema, (a, 5), {}, is_factory=False
        )
        a.add_(5)
        argument_handler.parse_outputs(add_inplace_func._schema, a, is_factory=False)

        self.assertEqual(set(), argument_handler.dataptrs_read)
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_written)

    def test_out(self):
        # [FUNCTION DOC] For the multiply operator with an out argument, checks the input is
        # read and the out tensor is written.
        mul_out_func = torch.ops.aten.mul.out
        a = torch.arange(8, device="cuda")
        b = torch.empty(8, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            mul_out_func._schema, (a, 3), {"out": b}, is_factory=False
        )
        torch.mul(a, 3, out=b)
        argument_handler.parse_outputs(mul_out_func._schema, b, is_factory=False)

        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({b.data_ptr()}, argument_handler.dataptrs_written)

    def test_nonzero(self):
        # [FUNCTION DOC] For nonzero with the tuple option (several output tensors), checks
        # the input is read and every returned tensor is written.
        nonzero_func = torch.ops.aten.nonzero.default
        a = torch.ones(5, 3, 2, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            nonzero_func._schema, (a,), {"as_tuple": True}, is_factory=False
        )
        out = torch.nonzero(a, as_tuple=True)
        argument_handler.parse_outputs(nonzero_func._schema, out, is_factory=False)

        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual(outputs, argument_handler.dataptrs_written)

    def test_tensor_names(self):
        # [FUNCTION DOC] For the addr operator called with the same vector twice, checks the
        # alias names recorded per pointer (self, and both vec names for the shared vector,
        # none for the output) and that the output set holds only the result pointer.
        addr_func = torch.ops.aten.addr.default
        vec = torch.arange(1, 4, device="cuda")
        M = torch.zeros(3, 3, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            addr_func._schema, (M, vec, vec), {}, is_factory=False
        )
        out = torch.addr(M, vec, vec)
        argument_handler.parse_outputs(addr_func._schema, out, is_factory=False)

        self.assertEqual(
            argument_handler.tensor_aliases,
            {
                M.data_ptr(): ["self"],
                vec.data_ptr(): ["vec1", "vec2"],
                out.data_ptr(): [],
            },
        )
        self.assertEqual({out.data_ptr()}, argument_handler.outputs)


def tensor_id(i: int) -> DataPtr:
    # [FUNCTION DOC] HELPER: Returns the given integer unchanged as a data pointer id.
    return i


def stream_id(i: int) -> StreamId:
    # [FUNCTION DOC] HELPER: Returns a stream id equal to 1000 plus the given integer.
    return 1000 + i


def event_id(i: int) -> EventId:
    # [FUNCTION DOC] HELPER: Returns an event id equal to 2000 plus the given integer.
    return 2000 + i


class TestEventHandler(TestCase):
    # [CLASS DOC] Tests the sanitizer's event handler, which tracks per-stream happens-before
    # state, using synthetic kernel launches, event records and waits, and synchronizations.
    # State: self.handler, a fresh event handler per test. Helper methods launch kernels and
    # assert on the number of synchronization errors.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp and creates a new event handler in self.handler.
        super().setUp()
        self.handler = csan.EventHandler()

    def kernel_launch(
        self,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
    ) -> list[csan.SynchronizationError]:
        # [FUNCTION DOC] HELPER: Simulates a kernel launch on the given stream
        # through the handler. Missing read-only and read-write pointer lists
        # default to empty; every pointer gets a single empty alias name. Returns
        # the list of synchronization errors detected.
        if read_only is None:
            read_only = []
        if read_write is None:
            read_write = []
        return self.handler._handle_kernel_launch(
            stream,
            read_only,
            read_write,
            {},
            "",
            {k: [""] for k in read_only + read_write},
        )

    def assert_good_kernel_launch(
        self,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Launches a kernel and asserts that no synchronization errors
        # are reported.
        self.assertEqual(self.kernel_launch(stream, read_only, read_write), [])

    def assert_bad_kernel_launch(
        self,
        number_of_errors: int,
        stream: StreamId,
        read_only: list[DataPtr] | None = None,
        read_write: list[DataPtr] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Launches a kernel and asserts that exactly the given number
        # of synchronization errors is reported.
        errors = self.kernel_launch(stream, read_only, read_write)
        self.assertEqual(len(errors), number_of_errors)

    def test_empty_kernel_launch(self):
        # [FUNCTION DOC] A launch with no accessed pointers is accepted.
        self.assert_good_kernel_launch(stream_id(0))

    def test_simple_passing(self):
        # [FUNCTION DOC] Two different streams may both read the same pointer
        # without synchronization.
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

    def test_simple_error(self):
        # [FUNCTION DOC] A write on a second stream to a pointer read on another stream with
        # no synchronization yields one error.
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_simple_sync(self):
        # [FUNCTION DOC] After a read, recording an event on the first stream and waiting for
        # it on the second makes a later write on the second stream valid.
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_reads_check_last_write(self):
        # [FUNCTION DOC] Checks that every read, not just the first, is checked against the
        # last write: a synchronized stream reads fine, but an unsynchronized third stream's
        # read gives one error.
        # Tests that not only the first read operation checks if it is in conflict
        # with the last write operation, but all read operations do.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(3), read_only=[tensor_id(1)])

    def test_branch_sync(self):
        # [FUNCTION DOC] Two streams that both wait on an event recorded after a write can
        # read, but a write by one of them without further synchronization gives one error.
        # Tests that two streams can read after both waiting for a third, but they
        # cannot write without further synchronization.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.handler._handle_event_wait(event_id(0), stream_id(3))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_chain_sync(self):
        # [FUNCTION DOC] Chains ten record and wait pairs across consecutive streams after a
        # read, then checks a write on the final stream is valid.
        iterations = 10

        self.assert_good_kernel_launch(stream_id(0), read_only=[tensor_id(1)])
        for i in range(iterations):
            self.handler._handle_event_record(event_id(i), stream_id(i))
            self.handler._handle_event_wait(event_id(i), stream_id(i + 1))
        self.assert_good_kernel_launch(stream_id(iterations), read_write=[tensor_id(1)])

    def test_expired_record(self):
        # [FUNCTION DOC] Work launched on the recording stream after the event record is not
        # covered by the event, so a write by the waiting stream gives one error.
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_wait(event_id(0), stream_id(2))

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_deleted_record(self):
        # [FUNCTION DOC] For each combination of deleting and re-creating an event after it
        # was recorded (each run on a freshly reset handler), waiting on it does not
        # synchronize, so a later write on the waiting stream gives one error.
        for should_delete, should_create in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            self.setUp()
            with self.subTest(should_delete=should_delete, should_create=should_create):
                self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
                self.handler._handle_event_record(event_id(0), stream_id(1))

                if should_delete:
                    self.handler._handle_event_deletion(event_id(0))
                if should_create:
                    self.handler._handle_event_creation(event_id(0))

                self.handler._handle_event_wait(event_id(0), stream_id(2))
                self.assert_bad_kernel_launch(
                    1, stream_id(2), read_write=[tensor_id(1)]
                )

    def test_all_reads_checked_failing(self):
        # [FUNCTION DOC] Nine streams read a pointer and record events, stream zero waits on
        # all of them, and then another stream reads and records. Stream zero's write is not
        # synchronized with that last read, so one error is expected.
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))

        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        self.assert_good_kernel_launch(stream_id(iterations), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(iterations), stream_id(i))

        # Does not synchronize with the last read.
        self.assert_bad_kernel_launch(1, stream_id(0), read_write=[tensor_id(1)])

    def test_all_reads_checked_passing(self):
        # [FUNCTION DOC] Nine streams read a pointer and record events, stream zero waits on
        # all of them, and its write is then accepted with no errors.
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))

        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])

    def test_multiple_errors(self):
        # [FUNCTION DOC] Writing ten pointers on one stream and then on another unsynchronized
        # stream yields ten errors, one per pointer.
        iterations = 10
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(iterations)]
        )
        self.assert_bad_kernel_launch(
            iterations,
            stream_id(1),
            read_write=[tensor_id(i) for i in range(iterations)],
        )

    def test_correct_state_merging(self):
        # [FUNCTION DOC] Checks that waiting on an event merges the recorded state into the
        # waiting stream's state, so after crossed record and wait pairs one stream can write
        # both pointers.
        # Tests that after waiting for an event, a stream's state is indeed set
        # to the pointwise maximum of its old state and the recorded state.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(2), stream_id(2))

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(2), stream_id(1))

        self.handler._handle_event_record(event_id(3), stream_id(2))
        self.handler._handle_event_wait(event_id(3), stream_id(1))
        self.assert_good_kernel_launch(
            stream_id(1), read_write=[tensor_id(1), tensor_id(2)]
        )

    def test_record_override(self):
        # [FUNCTION DOC] Recording the same event on a second stream overrides the first
        # record; a stream waiting on it is synchronized only with the second stream, so a
        # write gives one error.
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(1), stream_id(2))

        self.handler._handle_event_wait(event_id(1), stream_id(3))
        self.assert_bad_kernel_launch(1, stream_id(3), read_write=[tensor_id(1)])

    def test_multiple_wait(self):
        # [FUNCTION DOC] Two streams waiting on the same recorded event are both synchronized
        # with the recording stream and can read its written pointer.
        # Tests that a wait operation can be performed multiple times on the same event
        # by different streams.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(1), stream_id(3))

        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

    def test_device_synchronize(self):
        # [FUNCTION DOC] Nine streams each write their own pointer, a device synchronization
        # is handled, and stream zero can then write all of those pointers.
        # Tests that a device synchronization does correctly cause all streams
        # to synchronize with each other.

        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_write=[tensor_id(i)])

        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(1, iterations)]
        )

    def test_device_synchronization_expired(self):
        # [FUNCTION DOC] A device synchronization only covers earlier work: a later write on
        # another stream after further work on the first stream gives one error.
        # Tests that a device synchronization is a one-time synchronization.
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_new_stream_is_synchronized(self):
        # [FUNCTION DOC] A stream created after a device synchronization is already
        # synchronized with earlier work, so it may write a pointer written before.
        # Tests that after synchronizing operations with the host, any newly created
        # stream is guaranteed to be synchronized with them as well.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.handler._handle_stream_creation(stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_stream_synchronize(self):
        # [FUNCTION DOC] A stream synchronization makes the host, and thus later streams, wait
        # for that stream only: reads of its pointer pass, while a read of another stream's
        # pointer gives one error.
        # Tests that a stream synchronization does correctly cause all streams to wait
        # for one specific stream, but does not synchronize all streams with each other.

        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])
        self.handler._handle_stream_synchronization(stream_id(0))

        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(4), read_only=[tensor_id(2)])

    def test_event_synchronize(self):
        # [FUNCTION DOC] An event synchronization covers work up to the record only: a write
        # to the pointer written before the record passes, but one to a pointer written after
        # it gives one error.
        # Tests that an event synchronization does correctly cause all streams to wait
        # for a recorded event, but does not guarantee synchronization with the current
        # state of the stream that recorded the event.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])

        self.handler._handle_event_synchronization(event_id(1))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(2)])


class TestMessages(TestCase):
    # [CLASS DOC] Tests the text of log messages and error messages. State: self.handler, a fresh
    # event handler per test.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp and creates a new event handler in self.handler.
        super().setUp()
        self.handler = csan.EventHandler()

    def test_ensure_exists(self):
        # [FUNCTION DOC] For event deletion and memory deallocation of an id never created,
        # checks the first captured log message states the item was not found in the trace and
        # is backfilled.
        ARG = 0
        for func, out in [
            (
                self.handler._handle_event_deletion,
                f"Found Event with id: {ARG}, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
            (
                self.handler._handle_memory_deallocation,
                f"Found tensor with pointer: {ARG}, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
        ]:
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    def test_ensure_does_not_exist(self):
        # [FUNCTION DOC] After creating an event and a stream with the same id, repeats the
        # creation of each and checks the first log message: event duplicates are backfilled,
        # while a duplicate stream creation is ignored.
        ARG = 0
        self.handler._handle_event_creation(ARG)
        self.handler._handle_stream_creation(ARG)
        for func, out in [
            (
                self.handler._handle_event_creation,
                "Found duplicate event creation in the trace for event with "
                f"id: {ARG}. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
            (
                self.handler._handle_stream_creation,
                "Found duplicate Stream creation in the trace for Stream with "
                f"id: {ARG}. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
            ),
        ]:
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    def test_error_message(self):
        # [FUNCTION DOC] Builds a write access and a read access with stack traces, wraps them
        # in an unsynchronized access error with an allocation trace, and checks its string
        # form equals the expected multi-line report.
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=1,
            stream=stream_id(1),
            operator="schema",
            aliases=["b"],
            is_output=True,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace a")]
            ),
        )
        previous_access = csan.Access(
            type=csan.AccessType.READ,
            seq_num=2,
            stream=stream_id(0),
            operator="schema",
            aliases=["a"],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace b")]
            ),
        )
        error = csan.UnsynchronizedAccessError(
            data_ptr=tensor_id(1),
            allocation_stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "alloc")]
            ),
            current_access=current_access,
            previous_access=previous_access,
        )
        self.assertEqual(
            str(error),
            textwrap.dedent(
                """\
                ============================
                CSAN detected a possible data race on tensor with data pointer 1
                Access by stream 1001 during kernel:
                schema
                writing to argument(s) b, and to the output
                With stack trace:
                  File "file", line 0, in name
                    trace a

                Previous access by stream 1000 during kernel:
                schema
                reading from argument(s) a
                With stack trace:
                  File "file", line 0, in name
                    trace b

                Tensor was allocated with stack trace:
                  File "file", line 0, in name
                    alloc
                """
            ),
        )

    def test_subclass(self):
        # [FUNCTION DOC] Enables the CUDA sanitizer and constructs a TwoTensor and a plain
        # tensor subclass instance to check subclass creation works under the sanitizer's
        # dispatch mode; the sanitizer is always disabled in a final cleanup.
        class MyT(torch.Tensor):
            # [CLASS DOC] HELPER: Minimal tensor subclass whose constructor clones the data
            # and returns it viewed as this subclass.
            def __new__(cls, data):
                # [FUNCTION DOC] HELPER: Clones the data tensor and returns the clone as an
                # instance of the class.
                new_data = data.clone()
                return new_data.as_subclass(cls)

        try:
            csan.enable_cuda_sanitizer()

            # These two tests ensure that subclass creation
            # happens smoothly under the mode used by csan
            TwoTensor(torch.rand(2), torch.rand(2))
            MyT(torch.rand(2))
        finally:
            csan.cuda_sanitizer.disable()


if __name__ == "__main__":
    run_tests()
