# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Unit tests for torch.monitor: Stat aggregation (interval and fixed-count),
# Event creation and log_event, register/unregister of event handlers, the _WaitCounter
# guard, and the TensorboardEventHandler writing scalars to a SummaryWriter.
# Importing registers nothing globally; running the file calls run_tests.

import tempfile
import time

from datetime import datetime, timedelta

from torch.monitor import (
    _WaitCounter,
    Aggregation,
    Event,
    log_event,
    register_event_handler,
    Stat,
    TensorboardEventHandler,
    unregister_event_handler,
)
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestMonitor(TestCase):
    # [CLASS DOC] Tests for the core torch.monitor API (Stat, Event, log_event, event handlers,
    # _WaitCounter). Stateless; each test registers and unregisters its own handlers.
    def test_interval_stat(self) -> None:
        # [FUNCTION DOC] Registers a handler that collects events in a list, creates a Stat
        # with SUM and COUNT aggregations over a 1 millisecond window, and adds samples in a
        # loop of up to 100 sleep-and-add rounds until the window expires and an event
        # arrives. Asserts the stat name and that at least one event was delivered, then
        # unregisters the handler.
        events = []

        def handler(event):
            # [FUNCTION DOC] HELPER: Event handler; appends each received event to the
            # enclosing list.
            events.append(event)

        handle = register_event_handler(handler)
        s = Stat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            timedelta(milliseconds=1),
        )
        self.assertEqual(s.name, "asdf")

        s.add(2)
        for _ in range(100):
            # NOTE: different platforms sleep may be inaccurate so we loop
            # instead (i.e. win)
            time.sleep(1 / 1000)  # ms
            s.add(3)
            if len(events) >= 1:
                break
        self.assertGreaterEqual(len(events), 1)
        unregister_event_handler(handle)

    def test_fixed_count_stat(self) -> None:
        # [FUNCTION DOC] Creates a Stat with SUM and COUNT, a 100 hour window and a maximum
        # of three samples. After two adds the name is checked and the pending count is 2;
        # the third add reaches the maximum, so the count resets to 0 and the stat result
        # mapping holds SUM 6.0 and COUNT 3.
        s = Stat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            timedelta(hours=100),
            3,
        )
        s.add(1)
        s.add(2)
        name = s.name
        self.assertEqual(name, "asdf")
        self.assertEqual(s.count, 2)
        s.add(3)
        self.assertEqual(s.count, 0)
        self.assertEqual(s.get(), {Aggregation.SUM: 6.0, Aggregation.COUNT: 3})

    def test_log_event(self) -> None:
        # [FUNCTION DOC] Builds an Event with a name, a timestamp of now and string, float
        # and int data, checks the name, timestamp and data fields are populated, and logs
        # it. No handler is registered, so the only check is that logging does not raise.
        e = Event(
            name="torch.monitor.TestEvent",
            timestamp=datetime.now(),
            data={
                "str": "a string",
                "float": 1234.0,
                "int": 1234,
            },
        )
        self.assertEqual(e.name, "torch.monitor.TestEvent")
        self.assertIsNotNone(e.timestamp)
        self.assertIsNotNone(e.data)
        log_event(e)

    @skipIfTorchDynamo("Really weird error")
    def test_event_handler(self) -> None:
        # [FUNCTION DOC] Skipped under TorchDynamo. Registers a handler that records events,
        # logs the same Event twice and checks the handler saw it each time (equal to the
        # original), then unregisters and checks a further log reaches no handler (count
        # stays at 2).
        events = []

        def handler(event: Event) -> None:
            # [FUNCTION DOC] HELPER: Event handler; appends each received event to the
            # enclosing list.
            events.append(event)

        handle = register_event_handler(handler)
        e = Event(
            name="torch.monitor.TestEvent",
            timestamp=datetime.now(),
            data={},
        )
        log_event(e)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], e)
        log_event(e)
        self.assertEqual(len(events), 2)

        unregister_event_handler(handle)
        log_event(e)
        self.assertEqual(len(events), 2)

    def test_wait_counter(self) -> None:
        # [FUNCTION DOC] Creates a _WaitCounter with a name and enters its guard context manager
        # with an empty body. Only verifies that creation and the guard do not raise.
        wait_counter = _WaitCounter(
            "test_wait_counter",
        )
        with wait_counter.guard():
            pass


@skipIfTorchDynamo("Really weird error")
class TestMonitorTensorboard(TestCase):
    # [CLASS DOC] Tests for TensorboardEventHandler. Skipped under TorchDynamo, and each test is
    # skipped in setUp when tensorboard is not installed. Holds temp_dirs, a list of temporary
    # directories that receive SummaryWriter output and are removed in tearDown.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, then imports the tensorboard event multiplexer and
        # SummaryWriter and publishes them as module globals. Skips the test if the import fails;
        # otherwise initializes the temp_dirs list.
        super().setUp()
        global SummaryWriter, event_multiplexer
        try:
            from tensorboard.backend.event_processing import (
                plugin_event_multiplexer as event_multiplexer,
            )
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return self.skipTest("Skip the test since TensorBoard is not installed")
        self.temp_dirs = []

    def create_summary_writer(self):
        # [FUNCTION DOC] HELPER: Creates a temporary directory, records it in temp_dirs so it is
        # cleaned up later, and returns a SummaryWriter that writes into it.
        temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
        self.temp_dirs.append(temp_dir)
        return SummaryWriter(temp_dir.name)

    def tearDown(self):
        # [FUNCTION DOC] Cleans up every temporary directory recorded in temp_dirs.
        # KNOWN ISSUE: does not chain to the base class tearDown, unlike setUp.
        # Remove directories created by SummaryWriter
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()

    def test_event_handler(self):
        # [FUNCTION DOC] Registers a TensorboardEventHandler on a SummaryWriter, creates a
        # SUM and COUNT Stat with a 1 hour window and a maximum of five samples, adds ten
        # values and checks the pending count is 0, then unregisters the handler. Reads the
        # written run directory back with an event multiplexer, extracts the first float
        # value of each scalar tensor per tag and asserts the result maps the sum tag to 10
        # and the count tag to 5.
        with self.create_summary_writer() as w:
            handle = register_event_handler(TensorboardEventHandler(w))

            s = Stat(
                "asdf",
                (Aggregation.SUM, Aggregation.COUNT),
                timedelta(hours=1),
                5,
            )
            for i in range(10):
                s.add(i)
            self.assertEqual(s.count, 0)

            unregister_event_handler(handle)

        mul = event_multiplexer.EventMultiplexer()
        mul.AddRunsFromDirectory(self.temp_dirs[-1].name)
        mul.Reload()
        scalar_dict = mul.PluginRunToTagToContent("scalars")
        raw_result = {
            tag: mul.Tensors(run, tag)
            for run, run_dict in scalar_dict.items()
            for tag in run_dict
        }
        scalars = {
            tag: [e.tensor_proto.float_val[0] for e in events]
            for tag, events in raw_result.items()
        }
        self.assertEqual(
            scalars,
            {
                "asdf.sum": [10],
                "asdf.count": [5],
            },
        )


if __name__ == "__main__":
    run_tests()
