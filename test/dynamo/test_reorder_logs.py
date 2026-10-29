# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo handling of logging and print calls: functions listed
# in the ignore_logging_functions config are treated as no-ops, and functions listed
# in reorderable_logging_functions are replayed after the compiled graph runs instead
# of causing graph breaks. Import time defines module loggers, a Logger subclass with
# a warning_once method and a non-propagating instance of it with a null handler. It
# temporarily swaps the global logger class while creating that instance.
import io
import logging
import warnings
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


logger = logging.getLogger(__name__)
logger_test = logging.getLogger("test")


class CustomLogger(logging.Logger):
    # [CLASS DOC] HELPER: Logger subclass adding a warning_once method for tests.
    def warning_once(self, msg):
        # [FUNCTION DOC] HELPER: Emit the message as an ordinary warning (no deduplication).
        self.warning(msg)


_previous_logger_class = logging.getLoggerClass()
try:
    logging.setLoggerClass(CustomLogger)
    custom_logger = logging.getLogger("test_custom_logger")
finally:
    logging.setLoggerClass(_previous_logger_class)
custom_logger.addHandler(logging.NullHandler())
custom_logger.propagate = False


def f_info(x):
    # [FUNCTION DOC] HELPER: Double the input, log an info message through the module
    # logger, square the result and return it.
    x = x + x
    logger.info("moo")
    x = x * x
    return x


def f_isEnabledFor(x):
    # [FUNCTION DOC] HELPER: Like f_info but logs only if the module logger has info
    # level enabled.
    x = x + x
    if logger.isEnabledFor(logging.INFO):
        logger.info("moo")
    x = x * x
    return x


def f_warning_once(x):
    # [FUNCTION DOC] HELPER: Double the input, call warning_once on the custom logger,
    # square the result and return it.
    x = x + x
    custom_logger.warning_once("moo")
    x = x * x
    return x


@instantiate_parametrized_tests
class IgnoreLogsTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for the ignore_logging_functions config: calls to ignored
    # functions are dropped during compilation without graph breaks.
    @parametrize(
        "ignore_method, fn, captured_logger, should_ignore_logger",
        [
            (None, f_info, logger, False),
            (logger_test.info, f_info, logger, False),
            (None, f_isEnabledFor, logger, False),
            (logger_test.isEnabledFor, f_isEnabledFor, logger, False),
            (logger.info, f_info, logger, True),
            (logging.Logger.info, f_info, logger, True),
            (logger.isEnabledFor, f_isEnabledFor, logger, True),
            (logging.Logger.isEnabledFor, f_isEnabledFor, logger, True),
            (custom_logger.warning_once, f_warning_once, custom_logger, True),
            (CustomLogger.warning_once, f_warning_once, custom_logger, True),
            (logging.Logger.warning, f_warning_once, custom_logger, False),
        ],
    )
    def test_ignore_logger(
        self, ignore_method, fn, captured_logger, should_ignore_logger
    ):
        # [FUNCTION DOC] Parametrized on the function to ignore, the function to compile,
        # the logger to capture and whether the logger call should be dropped. Compiles
        # with the eager backend (graph breaks allowed) and captures logs; if the call is
        # ignored, the message must not be logged and no graph break recorded, otherwise
        # it must be logged and at least one graph break recorded. Output must match eager.
        counters.clear()
        x = torch.randn(3, 3)
        orig_out = fn(x)
        with torch._dynamo.config.patch(ignore_logging_functions={ignore_method}):
            opt_f = torch.compile(backend="eager")(fn)
            with self.assertLogs(captured_logger, level="INFO") as captured:
                captured_logger.info("call logger info to avoid error")
                opt_out = opt_f(x)
                printed_output = [entry.split(":", 2)[2] for entry in captured.output]

        self.assertTrue(same(orig_out, opt_out))
        if should_ignore_logger:
            self.assertNotIn("moo", printed_output)
            self.assertEqual(len(counters["graph_break"]), 0)
        else:
            self.assertIn("moo", printed_output)
            self.assertGreater(len(counters["graph_break"]), 0)

    def test_ignore_arbitrary_function_noop(self):
        # [FUNCTION DOC] An ignored arbitrary function must not run under full graph
        # compilation; the output matches eager and no graph break is recorded.
        counters.clear()
        calls = []

        def dbg_fn(x):
            # [FUNCTION DOC] HELPER: Appends a marker to the enclosing list (must never run).
            calls.append("ran")

        def f(x):
            # [FUNCTION DOC] HELPER: Calls the ignored function and returns the input plus one.
            dbg_fn(x)  # must be no-op inside Dynamo
            return x + 1

        x = torch.randn(3, 3)

        with torch._dynamo.config.patch(ignore_logging_functions={dbg_fn}):
            opt_f = torch.compile(backend="eager", fullgraph=True)(f)
            opt_out = opt_f(x)

        # function must never run
        self.assertEqual(calls, [])

        # output must match eager
        self.assertTrue(same(opt_out, x + 1))

        # no graph breaks allowed
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_ignore_function_returns_none(self):
        # [FUNCTION DOC] An ignored function's call evaluates to None in the compiled code
        # and never runs; the other output matches eager and no graph break is recorded.
        counters.clear()
        calls = []

        def ignore_me(x):
            # [FUNCTION DOC] HELPER: Appends a marker and returns a string (must never run).
            calls.append("ran")
            return "should_not_run"

        with torch._dynamo.config.patch(ignore_logging_functions={ignore_me}):

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the ignored function,
                # returns the doubled input and its result.
                y = ignore_me(x)  # Dynamo must replace with Constant(None)
                return x * 2, y

            x = torch.randn(3, 3)
            opt_f = torch.compile(backend="eager", fullgraph=True)(f)
            opt_out = opt_f(x)

        # ignored function must NOT run
        self.assertEqual(calls, [])

        # y must be None
        self.assertIs(opt_out[1], None)

        # output correct
        self.assertTrue(same(opt_out[0], x * 2))

        # no graph breaks
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_ignore_function_does_not_conflict_with_reorderable(self):
        # [FUNCTION DOC] One function is ignored and another reorderable in the same
        # config; the ignored one must not run, the reorderable one must run, and the
        # output must be correct.
        counters.clear()
        log = []

        def ignored(x):
            # [FUNCTION DOC] HELPER: Appends the ignored marker to the shared log.
            log.append("ignored")

        def reordered(x):
            # [FUNCTION DOC] HELPER: Appends the reordered marker to the shared log.
            log.append("reordered")

        def f(x):
            # [FUNCTION DOC] HELPER: Calls both functions and returns the input plus one.
            ignored(x)
            reordered(x)
            return x + 1

        x = torch.ones(3, 3)

        with torch._dynamo.config.patch(
            ignore_logging_functions={ignored},
            reorderable_logging_functions={reordered},
        ):
            opt_f = torch.compile(backend="eager", fullgraph=True)(f)
            opt_out = opt_f(x)

        # ignored must NOT run
        self.assertNotIn("ignored", log)

        # reordered MUST run
        self.assertIn("reordered", log)
        # output is correct
        self.assertTrue(same(opt_out, x + 1))


class ReorderLogsTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for the reorderable_logging_functions config: print, logger
    # methods, warnings and custom functions are replayed after the graph runs.
    def test_dont_reorder_print(self):
        # [FUNCTION DOC] Without configuration print causes a graph break: output is the
        # single message, the result matches eager and one graph break is counted.
        def f(x):
            # [FUNCTION DOC] HELPER: Double, print a message, square and return.
            x = x + x
            print("moo")
            x = x * x
            return x

        counters.clear()
        x = torch.randn(3, 3)
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertEqual(printed_output, "moo")
        self.assertEqual(len(counters["graph_break"]), 1)

    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_reorder_print(self):
        # [FUNCTION DOC] With print reorderable, three prints (a string, a tensor, numbers)
        # compile in one graph; printed text equals the expected sequence in order and the
        # result matches eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Prints interleaved with tensor math; returns two tensors.
            print("moo")
            x1 = x + x
            print(x1)
            x2 = x1 * x1
            print(1, 2, 3)
            x3 = x2 + x2
            return (x1, x3)

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(printed_output, f"moo\n{torch.ones(3, 3) * 2}\n1 2 3")
        self.assertTrue(same(orig_out, opt_out))

    def test_reorder_logger_method(self):
        # [FUNCTION DOC] For both the bound and unbound logger info method as reorderable,
        # compiles with full graph; exactly one record is captured whose text holds the
        # tensor sum and the output is all fours. Resets Dynamo for each variant.
        def f(x):
            # [FUNCTION DOC] HELPER: Double, log the sum through the module logger, square, return.
            x = x + x
            logger.info("moo %s", x.sum())
            x = x * x
            return x

        x = torch.ones(3)
        for reordered in (logger.info, logging.Logger.info):
            torch._dynamo.reset()
            with torch._dynamo.config.patch(reorderable_logging_functions={reordered}):
                opt_f = torch.compile(backend="eager", fullgraph=True)(f)
                with self.assertLogs(logger, logging.INFO) as captured:
                    opt_out = opt_f(x)
            self.assertEqual(len(captured.output), 1)
            self.assertIn("moo tensor(6.)", captured.output[0])
            self.assertTrue(same(opt_out, torch.full((3,), 4.0)))

    @torch._dynamo.config.patch(reorderable_logging_functions={logging.Logger.info})
    def test_dont_reorder_logger_method_kwargs(self):
        # [FUNCTION DOC] Logger call with a keyword argument cannot be replayed; compiling
        # fully must raise Unsupported saying it attempted to reorder a function that cannot
        # actually be reordered.
        # kwargs must graph break rather than be silently dropped at replay
        def f(x):
            # [FUNCTION DOC] HELPER: Double, log with a stack level keyword, square and return.
            x = x + x
            logger.info("moo %s", x.sum(), stacklevel=2)
            return x * x

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "attempted to reorder a debugging function that can't actually be reordered",
        ):
            torch.compile(backend="eager", fullgraph=True)(f)(torch.ones(3))

    @torch._dynamo.config.patch(reorderable_logging_functions={logging.Logger.info})
    def test_dont_reorder_logger_method_unsupported_arg(self):
        # [FUNCTION DOC] Logger call whose argument is a list holding a tensor is
        # unsupported for reordering; full compilation raises the same Unsupported error.
        def f(x):
            # [FUNCTION DOC] HELPER: Double, log with a list
            # argument holding a tensor, square, return.
            x = x + x
            logger.info("moo %s", [x.sum()])
            return x * x

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "attempted to reorder a debugging function that can't actually be reordered",
        ):
            torch.compile(backend="eager", fullgraph=True)(f)(torch.ones(3))

    @torch._dynamo.config.patch(reorderable_logging_functions={warnings.warn})
    def test_reorder_warnings(self):
        # [FUNCTION DOC] With the warnings.warn function reorderable, two warnings compile
        # in one graph; recorded warnings include the first message and the output matches eager.
        import warnings

        def f(x):
            # [FUNCTION DOC] HELPER: Two warnings interleaved
            # with tensor math; returns the final tensor.
            x1 = x + x
            warnings.warn("moo")
            x2 = x1 * x1
            warnings.warn(f"{x2}")
            x3 = x2 + x2
            return x3

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        with warnings.catch_warnings(record=True) as w:
            opt_out = opt_f(x)
            warning_messages = [str(i.message) for i in w]
            orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertIn("moo", warning_messages)

    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_reorder_print_graph_break(self):
        # [FUNCTION DOC] Print is reorderable and there is an explicit graph break between
        # two prints; printed text equals both messages in order and output matches eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Print, explicit graph break, print; returns a tensor.
            x1 = x + x
            print(f"res: {x1}")
            x2 = x1 * x1
            torch._dynamo.graph_break()
            x3 = x2 + x2
            print(1, 2, 3)
            return x3

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(printed_output, f"res: {torch.ones(3, 3) * 2}\n1 2 3")
        self.assertTrue(same(orig_out, opt_out))

    def test_reorder_custom_log_fn(self):
        # [FUNCTION DOC] A user function that itself graph breaks is reorderable; the
        # messages are recorded in order with the right values, and exactly one graph break
        # is counted in total.
        custom_logs = []

        def custom_log(s: str):
            # [FUNCTION DOC] HELPER: Trigger a graph break,
            # then append the string to the shared list.
            torch._dynamo.graph_break()
            custom_logs.append(s)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls the custom logger twice around a tensor addition.
            custom_log("moo")
            x1 = x + x
            custom_log(f"{x1}")
            return x + x

        x = torch.ones(3, 3)
        counters.clear()
        with torch._dynamo.config.patch(reorderable_logging_functions={custom_log}):
            opt_f = torch.compile(backend="eager")(f)
            opt_f(x)

        self.assertEqual(sum(counters["graph_break"].values()), 1)
        self.assertEqual(custom_logs[0], "moo")
        self.assertEqual(custom_logs[1], f"{torch.ones(3, 3) * 2}")

    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_constant_mutation(self):
        # [FUNCTION DOC] Print is reorderable and a list is mutated between prints, with
        # graph breaks from scalar extraction. The printed text must reflect the list state
        # at each print and the single distinct graph break message must equal the expected
        # text about the capture scalar outputs setting.
        def f(x):
            # [FUNCTION DOC] HELPER: Append to a list, print its last element, extract a scalar
            # (graph break), pop, print again, extract a scalar again, return the popped tensor.
            alist = [x]
            alist.append(x + 1)
            print(alist[-1])
            alist[0].sum().item()  # graph break
            res = alist.pop()
            print(alist[-1])
            res.sum().item()  # graph break
            return res

        inputs = (torch.tensor([1]),)
        counters.clear()
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(*inputs)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(*inputs)

        self.assertEqual(printed_output, "tensor([2])\ntensor([1])")
        self.assertTrue(same(orig_out, opt_out))

        graph_break_key = counters["graph_break"].keys()
        self.assertEqual(len(graph_break_key), 1)
        self.assertExpectedInline(
            next(iter(graph_break_key)),
            """\
Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html""",
        )

    def test_reorder_print_data_dependent_fstring(self):
        # [FUNCTION DOC] A print of a value derived from tensor data fails with Unsupported
        # under full compilation when print is not reorderable. With print reorderable and
        # scalar capture enabled it compiles fully, prints the expected text and the
        # result matches eager.
        """Print with data-dependent bool in f-string should graph break on print,
        but work when print is reorderable."""

        def f(x, mask):
            # [FUNCTION DOC] HELPER: Evaluate whether the mask is all zeros, print it in a
            # formatted string and return the input plus one.
            make_causal = bool((mask == 0).all())
            print(f"make_causal={make_causal}")
            return x + 1

        x = torch.randn(2, 3)
        mask = torch.zeros(2, 3)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Dynamo does not know how to trace builtin operator `print`",
        ):
            torch.compile(backend="eager", fullgraph=True)(f)(x, mask)

        with torch._dynamo.config.patch(
            reorderable_logging_functions={print}, capture_scalar_outputs=True
        ):
            opt_f = torch.compile(backend="eager", fullgraph=True)(f)
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                opt_out = opt_f(x, mask)
                printed_output = mock_stdout.getvalue().strip()

        self.assertTrue(same(opt_out, x + 1))
        self.assertEqual(printed_output, "make_causal=True")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
