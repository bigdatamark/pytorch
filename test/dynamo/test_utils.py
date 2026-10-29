# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo utility and logging helpers: tolerance-based output comparison,
#   cleanup hooks, graph-break counting, traced-code queries, record-function data, reinplace
#   counters, compilation-metrics events (timings, node shapes, parameter counts, IR counts,
#   provenance, feature usage, config snapshots), accelerator sync in the timing helper, and JSON
#   scrubbing of inductor and functorch configs for logging.
# Defines a small linear model used by the tests, a Windows-detection flag, and instantiates
#   device-generic tests for one class at import time (allowing XPU).
import dataclasses
import json
import os
import pprint
import sys
import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch.compiler.config as compiler_config
from torch._dynamo import utils
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_LINUX,
    IS_MACOS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
)


_IS_WINDOWS = sys.platform == "win32"


class TestUtils(TestCase):
    # [CLASS DOC] Tests for assorted helpers in the Dynamo utils module: cleanup hooks, output
    #   comparison, graph-break counting, traced code lookup, profiler record-function logging,
    #   reinplace counters, and config logging.
    def test_cleanup_hook_tolerates_missing_name(self):
        # [FUNCTION DOC] A cleanup hook must not raise when the installed global was already
        #   removed from its scope (as happens at interpreter shutdown), and in the normal path
        #   calling it removes the installed global from the scope.
        # During interpreter shutdown the scope's module __dict__ may already be
        # cleared before the weakref callback fires the hook. The hook must not
        # raise KeyError (which would surface as an "Exception ignored in
        # weakref callback").
        scope = {}
        hook = utils.CleanupHook.create(scope, "myglobal", object())
        del scope["myglobal"]
        hook()

        # The normal path still removes the installed global.
        hook2 = utils.CleanupHook.create(scope, "myglobal", object())
        hook2()
        self.assertNotIn("myglobal", scope)

    def test_nan(self):
        # [FUNCTION DOC] The same-output comparison treats two NaN tensors as equal when
        #   equal-NaN is requested, given a double-precision reference.
        a = torch.Tensor([float("nan")])
        b = torch.Tensor([float("nan")])
        fp64_ref = torch.DoubleTensor([5.0])
        res = utils.same(a, b, fp64_ref=fp64_ref, equal_nan=True)
        self.assertTrue(res)

    def test_same_iou_for_bool_propagates_to_instances(self):
        # [FUNCTION DOC] The comparison on arbitrary objects with a boolean tensor attribute: one
        #   mask element differs, so the default comparison reports a mismatch while the
        #   IoU-for-bool option reports a match.
        Instances = type("Instances", (), {})
        ref = Instances()
        res = Instances()
        ref.pred_masks = torch.ones((4, 16, 16), dtype=torch.bool)
        res.pred_masks = ref.pred_masks.clone()
        res.pred_masks[0, 0, 0] = False

        self.assertFalse(utils.same(ref, res))
        self.assertTrue(utils.same(ref, res, use_iou_for_bool=True))

    def test_larger_multiplier_for_smaller_tensor(self):
        # [FUNCTION DOC] For tensors with a medium element count, a result off by factor 2 passes
        #   within the tolerance but factor 4 fails by default; the larger-multiplier option
        #   accepts factor 4 and still rejects factor 9.
        """
        Tensor numel between (10, 500]
        """
        N = 100
        fp64_ref = torch.full([N], 0.0, dtype=torch.double)
        a = torch.full([N], 1.0)
        tol = 4 * 1e-2
        self.assertTrue(utils.same(a, a * 2, fp64_ref=fp64_ref, tol=tol))
        self.assertFalse(utils.same(a, a * 4, fp64_ref=fp64_ref, tol=tol))
        self.assertTrue(
            utils.same(
                a,
                a * 4,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )
        self.assertFalse(
            utils.same(
                a,
                a * 9,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )

    def test_larger_multiplier_for_even_smaller_tensor(self):
        # [FUNCTION DOC] For a one-element tensor, factor 2 passes within the tolerance but
        #   factor 7 fails by default; the larger-multiplier option accepts factor 7 but still
        #   rejects factor 20.
        """
        Tensor numel <=10
        """
        fp64_ref = torch.DoubleTensor([0.0])
        a = torch.Tensor([1.0])
        tol = 4 * 1e-2
        self.assertTrue(utils.same(a, a * 2, fp64_ref=fp64_ref, tol=tol))
        self.assertFalse(utils.same(a, a * 7, fp64_ref=fp64_ref, tol=tol))
        self.assertTrue(
            utils.same(
                a,
                a * 7,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )
        self.assertFalse(
            utils.same(
                a,
                a * 20,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    def test_graph_break_counting(self):
        # [FUNCTION DOC] Verifies the graph-break count recorded in compilation events (collected
        #   by patching the event logger): a clean model has zero breaks, a function with one
        #   data-dependent branch under the inductor backend shows one break (also one when
        #   recompiled after reset, not two), and a function with two nested branches shows two.
        #   Runs with compilation-metric logging enabled.
        """
        Run a compilation that includes a graph break and validate that the
        graph break counter is incremented.
        """

        def run_forward_backward():
            # [FUNCTION DOC] HELPER: Compile the module-level linear model, run it forward on a
            #   random input requiring grad, compute a mean-squared-error loss against a fixed
            #   target, and run backward.
            model = torch.compile(TestModel())  # noqa: UNSPECIFIED_BACKEND
            x = torch.rand([3], requires_grad=True)
            output = model(x)
            loss_fn = torch.nn.MSELoss()
            target = torch.tensor([1.0])
            loss = loss_fn(output, target)
            loss.backward()

        @torch.compile(backend="eager")
        def add(x, y):
            # [FUNCTION DOC] HELPER: Eager-backend compiled function returning the sum of its two
            #   arguments.
            return x + y

        @torch.compile(backend="eager")
        def break_it(x):
            # [FUNCTION DOC] HELPER: Eager-backend compiled function with a data-dependent branch
            #   on the sum of the input; each branch adds or subtracts a scalar item, causing a
            #   graph break.
            y = x.sum()
            if y > 0:
                return x + y.item()
            return x - y.item()

        @torch.compile(backend="eager")
        def break_it2(x):
            # [FUNCTION DOC] HELPER: Eager-backend compiled function with two nested
            #   data-dependent branches, each using a scalar item (multiply, add or subtract),
            #   causing two graph breaks.
            y = x.sum()
            if y > 0:
                if y > 1:
                    return x * y.item()
                return x + y.item()
            return x - y.item()

        add(torch.rand([10]), torch.rand([10]))
        utils.reset_frame_count()

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 0)

            # We should fallback to normal mode and increment the graph break counter
            torch.compile(break_it, backend="inductor")(torch.ones(3, 3))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 1)

            # Graph break counter should be incremented by 1 (after a reset), not 2
            torch.compile(break_it, backend="inductor")(torch.ones(3, 3))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 1)

            # Graph break counter should be incremented by 2
            torch.compile(break_it2, backend="inductor")(torch.ones(3, 3))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 2)

    def test_traced_code_query(self):
        # [FUNCTION DOC] Checks the traced-code query inside a custom backend: records file names
        #   of traced code lists for a function with no inlining (only this file), with
        #   successful inlining of a helper from a sibling utils file (both files in one list),
        #   with a graph break during inlining and nested graph breaks disabled (two separate
        #   lists, one per file), and with an empty graph (no list recorded). Imports the helpers
        #   with a fallback for running outside a package.
        try:
            from .utils import add, break_it
        except ImportError:
            from utils import add, break_it

        traced_code_lists = []

        def get_filenames(traced_code_lists):
            # [FUNCTION DOC] HELPER: Convert each list of code objects into a list of their file
            #   names.
            return [
                [code.co_filename for code in code_list]
                for code_list in traced_code_lists
            ]

        def my_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend that appends the currently traced code list to an
            #   enclosing list and returns the graph forward function.
            from torch._dynamo.utils import get_traced_code

            nonlocal traced_code_lists
            traced_code_lists.append(get_traced_code())
            return gm.forward

        # === no inlining ===
        @torch.compile(backend=my_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled with the recording backend: doubles the
            #   input (no inlining).
            return x * 2

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        self.assertEqual(get_filenames(traced_code_lists), [[__file__]])

        # === successful inlining ===
        @torch.compile(backend=my_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled with the recording backend: calls an
            #   imported helper add then doubles the result.
            return add(x) * 2

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        utils_path = os.path.join(os.path.dirname(__file__), "utils.py")
        self.assertEqual(get_filenames(traced_code_lists), [[__file__, utils_path]])

        # === graph break occurs during inlining ===
        @torch._dynamo.disable_nested_graph_breaks
        @torch.compile(backend=my_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled with the recording backend and nested
            #   graph breaks disabled: adds one, calls an imported helper that graph-breaks, and
            #   doubles the result.
            z = x + 1
            y = break_it(z)
            return y * 2

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        self.assertEqual(get_filenames(traced_code_lists), [[__file__], [utils_path]])

        # === empty graph ===
        @torch.compile(backend=my_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled with the recording backend that returns
            #   its input unchanged (empty graph).
            return x

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        self.assertEqual(traced_code_lists, [])

    def test_add_record_function_data(self):
        # [FUNCTION DOC] Checks that adding record-function data logs nothing to the profiler
        #   record-function context when the profiler is disabled, or when enabled but no
        #   key-value data is given, and with data and an enabled profiler calls it once with a
        #   name plus a formatted key=value list.
        with (
            mock.patch("torch.autograd.profiler._is_profiler_enabled", False),
            mock.patch("torch.autograd.profiler.record_function") as mock_rf,
        ):
            utils.CompileEventLogger.add_record_function_data(
                "test_event", key1="value1", key2="value2"
            )
            mock_rf.assert_not_called()

        with (
            mock.patch("torch.autograd.profiler._is_profiler_enabled", True),
            mock.patch("torch.autograd.profiler.record_function") as mock_rf,
        ):
            utils.CompileEventLogger.add_record_function_data("test_event")
            mock_rf.assert_not_called()

        with (
            mock.patch("torch.autograd.profiler._is_profiler_enabled", True),
            mock.patch("torch.autograd.profiler.record_function") as mock_rf,
        ):
            utils.CompileEventLogger.add_record_function_data(
                "test_event", key1="value1", key2="value2"
            )
            mock_rf.assert_called_once_with("test_event_data: key1=value1, key2=value2")

    def test_reinplace_counters_use_trigger_name_not_enum_value(self):
        # [FUNCTION DOC] Verifies that the reinplace counters build their dictionary keys from
        #   the trigger name rather than the enum value: after adding missed tensors and bytes
        #   for one trigger, keys with the trigger name exist with the right values; after
        #   clearing and using another trigger, only the new name-based key exists; no key
        #   containing the numeric enum value exists. Clears counters at start and between
        #   phases.
        """Test that ReinplaceCounters uses trigger.name in dictionary keys instead of the enum value"""
        from torch._dynamo.utils import ReinplaceCounters, ReInplaceTrigger

        # Clear any existing state
        ReinplaceCounters.clear()

        # Test with AUTO_FUNC_V1 trigger
        trigger = ReInplaceTrigger.AUTO_FUNC_V1

        # Add some values
        ReinplaceCounters.add_missed_opportunities(trigger, 2)
        ReinplaceCounters.add_missed_bytes(trigger, 512)

        # Check that the dictionary keys use the trigger name, not the enum value
        expected_tensor_key = "missed_tensors_AUTO_FUNC_V1"
        expected_bytes_key = "missed_bytes_AUTO_FUNC_V1"

        # Verify the keys exist with the correct format
        self.assertIn(
            expected_tensor_key,
            ReinplaceCounters._values,
            lambda msg: f"{msg}\nExpected key {expected_tensor_key} not found",
        )
        self.assertIn(
            expected_bytes_key,
            ReinplaceCounters._values,
            lambda msg: f"{msg}\nExpected key {expected_bytes_key} not found",
        )

        # Verify the values are correct
        self.assertEqual(ReinplaceCounters._values[expected_tensor_key], 2)
        self.assertEqual(ReinplaceCounters._values[expected_bytes_key], 512)

        # Clear for next test
        ReinplaceCounters.clear()

        # Test with a different trigger to ensure it's not hardcoded
        trigger2 = ReInplaceTrigger.TRITON_OPS
        ReinplaceCounters.add_missed_opportunities(trigger2, 3)

        expected_key2 = "missed_tensors_TRITON_OPS"
        self.assertIn(
            expected_key2,
            ReinplaceCounters._values,
            lambda msg: f"{msg}\nExpected key {expected_key2} not found",
        )
        self.assertEqual(ReinplaceCounters._values[expected_key2], 3)

        # Verify the old key doesn't exist
        self.assertNotIn("missed_tensors_AUTO_FUNC_V1", ReinplaceCounters._values)

        # Test edge case: check that we don't use the enum integer value
        # ReInplaceTrigger.AUTO_FUNC_V1 has value 1, so we verify "missed_tensors_1" doesn't exist
        self.assertNotIn(
            f"missed_tensors_{trigger.value}",
            ReinplaceCounters._values,
            "Should not use enum value (integer) in key, should use trigger.name instead",
        )

    def test_get_dynamo_config_for_logging_ignores_logging_functions(self):
        # [FUNCTION DOC] The config snapshot prepared for logging is a JSON string that parses to
        #   a dictionary and omits the ignore-logging-functions option even when it is set to a
        #   set of functions.
        with dynamo_config.patch(ignore_logging_functions={print}):
            result = utils._get_dynamo_config_for_logging()
            parsed = json.loads(result)

        self.assertIsInstance(parsed, dict)
        self.assertNotIn("ignore_logging_functions", parsed)


class TestModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny module with a single linear layer from 3 inputs to 1 output,
    #   compiled by several tests.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Create the linear layer.
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Apply the linear layer to the input.
        return self.linear(x)


class TestDynamoTimed(TestCase):
    # [CLASS DOC] Tests of compilation metrics produced around timed compile phases; they capture
    #   the events passed to a patched compilation-event logger after compiling a small model
    #   forward and backward.
    """
    Test utilities surrounding dynamo_timed.
    """

    def setUp(self):
        # [FUNCTION DOC] Run the base setup, then reset recompile user contexts when that
        #   function exists.
        super().setUp()
        if hasattr(torch._dynamo, "reset_recompile_user_contexts"):
            torch._dynamo.reset_recompile_user_contexts()

    def run_forward_backward(self):
        # [FUNCTION DOC] HELPER: Compile the module-level linear model, run it forward on a
        #   random input requiring grad, compute a mean-squared-error loss against a fixed
        #   target, and run backward.
        model = torch.compile(TestModel())  # noqa: UNSPECIFIED_BACKEND
        x = torch.rand([3], requires_grad=True)
        output = model(x)
        loss_fn = torch.nn.MSELoss()
        target = torch.tensor([1.0])
        loss = loss_fn(output, target)
        loss.backward()

    def warmup(self):
        # [FUNCTION DOC] HELPER: Warm up process-global caches by compiling and running a small
        #   add function with the inductor backend, then reset the frame count and clear the
        #   structured-logging overhead record.
        # Helper to make sure any process-global lru_caches (e.g., torch_key())
        # have already executed. Just compile something.
        @torch.compile(backend="inductor")
        def add(x, y):
            # [FUNCTION DOC] HELPER: Inductor-compiled function returning the sum of its two
            #   arguments.
            return x + y

        add(torch.rand([10]), torch.rand([10]))
        utils.reset_frame_count()
        torch._logging._internal.structured_logging_overhead.clear()

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_stack_trace(self):
        # [FUNCTION DOC] Runs a forward-backward compile with metrics logging and caches
        #   disabled, collects the stack trace of each compilation event, and asserts that the
        #   combined text mentions this test's name.
        self.warmup()

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            self.run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        stack_trace_list = []
        for e in compilation_events:
            stack_trace_list.append(e.stack_trace)

        self.assertGreater(len(stack_trace_list), 0)
        result = "\n".join(
            item
            for sublist in stack_trace_list
            if sublist
            for item in (sublist if isinstance(sublist, list) else [sublist])
        )
        self.assertIn(
            "test_stack_trace",
            result,
            "Log file does not contain the expected string: 'test_stack_trace'",
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_graph_node_shapes(self):
        # [FUNCTION DOC] Runs a forward-backward compile with metrics logging and caches disabled
        #   and asserts that the first event records the graph node names mapped to their shapes
        #   exactly as expected for the linear model (bias, weight, input, output).
        self.warmup()

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            self.run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]

        self.assertEqual(
            compilation_events[0].graph_node_shapes,
            "{'l_self_modules_linear_parameters_bias_': [1], "
            "'l_self_modules_linear_parameters_weight_': [1, 3], "
            "'l_x_': [3], 'linear': [1]}",
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_log_dynamo_start(self):
        # [FUNCTION DOC] Runs a warm-up compile, then asserts that the Dynamo start log for a
        #   sample function's code object returns stack strings whose last entry contains the
        #   code's first line number, its name and this file name, and that there is more than
        #   one entry.
        import torch._dynamo.convert_frame as convert_frame

        self.warmup()
        self.run_forward_backward()

        # Dummy code object
        def sample_func():
            # [FUNCTION DOC] HELPER: Empty function whose code object is passed to the start-log
            #   function.
            pass

        code = sample_func.__code__
        stack_strings = convert_frame.log_dynamo_start(code)
        last_entry = stack_strings[-1]
        # Check if the last entry is a valid stack trace i.e for the sample_func
        self.assertIn(
            f"Line: {code.co_firstlineno}",
            last_entry,
            "Log does not contain a Line no.",
        )
        self.assertIn(
            f"Name: {code.co_name}", last_entry, "Log does not contain a Name"
        )
        self.assertIn(
            "test_utils.py",
            last_entry,
            "Log file does not contain the expected Filename: 'test_utils.py'",
        )

        # Since the remaining logs are env specific, we just check if they are present instead of checking the exact string
        self.assertGreater(len(stack_strings), 1)

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_exception_stack_trace(self):
        # [FUNCTION DOC] A full-graph compile of a function calling the print builtin raises the
        #   unsupported-operation error with a message about print; the logged compilation event
        #   must contain a non-empty exception stack trace whose first entry includes that
        #   message.
        from torch._dynamo.exc import Unsupported

        def backward(grad_output):
            # [FUNCTION DOC] HELPER: Function that prints a message (not traceable) and returns
            #   its argument.
            print("graph break!")  # This should trigger a Dynamo error
            return grad_output

        compiled_backward = torch.compile(backward, backend="eager", fullgraph=True)
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            with self.assertRaisesRegex(
                Unsupported,
                "Dynamo does not know how to trace builtin operator `print`",
            ):
                compiled_backward(torch.ones(3))

        compilation_events = [arg[0][0] for arg in log_event.call_args_list]

        self.assertGreater(len(compilation_events), 0)
        self.assertGreater(len(compilation_events[0].exception_stack_trace), 0)
        self.assertIn(
            "Dynamo does not know how to trace builtin operator `print`",
            compilation_events[0].exception_stack_trace[0],
            "exception_stack_trace does not contain the expected string: "
            "'Dynamo does not know how to trace builtin operator `print`'",
        )

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/148093",
    )
    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    @inductor_config.patch(
        {
            "bundle_triton_into_fx_graph_cache": False,
            "bundled_autotune_remote_cache": False,
            "force_disable_caches": True,
        }
    )
    # We can't easily test that timing is actually accurate. Mock time to always
    # return the same value; all durations will be zero.
    @mock.patch("time.time", return_value=0.001)
    @mock.patch("time.time_ns", return_value=100000)
    @dynamo_config.patch(specialize_float=False)
    def test_dynamo_timed(self, mock_time, mock_time_ns):
        # [FUNCTION DOC] Single-compile test of the timing instrumentation, skipped on several
        #   platforms and builds due to an open issue. With mocked clocks (all durations zero),
        #   checks the compilation time metrics mapping, the time-spent summary, and the full
        #   dictionaries of the forward and backward compilation events against expected text,
        #   with variants for Windows. Config-dependent entries (garbage collection) are filtered
        #   from expectations. Takes the mocked clock arguments, which are not otherwise used.
        # The expected output contains no source line numbers, so comment insertion does not
        #   affect it.
        """
        Run a compilation that includes a forward and a backward and validate
        various recorded metrics. This test could be broken into several, but the
        compilation is somewhat expensive. Instead of resetting and compiling the
        same thing multiple times, we may as well compile once and just check all
        the things that are affected by dynamo_timed.
        """
        self.warmup()

        # The logging function is different for OSS vs. internal. Let's just mock
        # and capture all the CompilationMetric objects logged.
        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            self.run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]

        def filter_expected(s: str) -> str:
            # [FUNCTION DOC] HELPER: Evaluate an expected-dictionary text, drop
            #   garbage-collection entries (and null the gc time) if garbage collection after
            #   compile is disabled in config, and return its pretty-printed form.
            d = eval(s)
            if not dynamo_config.run_gc_after_compile:
                d.pop("gc", None)
                if "gc_time_us" in d:
                    d["gc_time_us"] = None
            return pprint.pformat(d)

        # Validate utils.compile_times(). Unfortunately, we can't test the output
        # reliably because it depends on whether 'tabulate' is installed. So we'll
        # directly inspect the dict it prints instead:
        self.assertExpectedInline(
            pprint.pformat(utils.compilation_time_metrics),
            filter_expected(
                """\
{'CacheBase.get_system.triton_key': [0.0],
 'GraphLowering.codegen': [0.0, 0.0],
 'GraphLowering.compile_to_fn': [0.0, 0.0],
 'GraphLowering.compile_to_module': [0.0, 0.0],
 'GraphLowering.run': [0.0, 0.0],
 'OutputGraph.call_user_compiler': [0.0],
 'PyCodeCache.load_by_key_path': [0.0, 0.0],
 'PythonWrapperCodegen.generate': [0.0, 0.0],
 'Scheduler.__init__': [0.0, 0.0],
 'Scheduler.codegen': [0.0, 0.0],
 'Scheduler.fused_nodes': [0.0, 0.0],
 '_compile.compile_inner': [0.0],
 '_recursive_joint_graph_passes': [0.0],
 '_recursive_post_grad_passes': [0.0, 0.0],
 '_recursive_pre_grad_passes': [0.0],
 'additional_fake_tensor_prop': [0.0, 0.0],
 'aot_collect_metadata': [0.0],
 'aot_trace_joint_graph': [0.0],
 'async_compile.wait': [0.0, 0.0],
 'backward._backward_impl': [0.0],
 'build_guards': [0.0],
 'bytecode_tracing': [0.0],
 'compile_attempt_0': [0.0],
 'compile_file': [0.0, 0.0],
 'compile_fx.<locals>.bw_compiler': [0.0],
 'compile_fx.<locals>.fw_compiler_base': [0.0],
 'compile_fx_inner': [0.0, 0.0],
 'create_aot_dispatcher_function': [0.0],
 'fx_codegen_and_compile': [0.0, 0.0],
 'gc': [0.0],
 'insert_deferred_runtime_asserts': [0.0],
 'min_cut_rematerialization_partition': [0.0],
 'pass.joint_graph_passes.constant_fold_uniform_value': [0.0],
 'pass.joint_graph_passes.pass_pattern_0': [0.0],
 'pass.joint_graph_passes.pass_pattern_1': [0.0],
 'pass.joint_graph_passes.remove_noop_ops': [0.0],
 'pass.post_grad_passes.decompose_auto_functionalized': [0.0, 0.0],
 'pass.post_grad_passes.decompose_map_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_scan_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_triton_kernel_wrapper_functional': [0.0, 0.0],
 'pass.post_grad_passes.fix_auto_functionalized_dtype_views': [0.0, 0.0],
 'pass.post_grad_passes.move_constructors_to_cuda': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_0': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_1': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_2': [0.0, 0.0],
 'pass.post_grad_passes.post_grad_custom_pre_pass': [0.0, 0.0],
 'pass.post_grad_passes.reinplace_inplaceable_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_assert_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_noop_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_profiler_ops': [0.0, 0.0],
 'pass.post_grad_passes.stable_sort': [0.0, 0.0],
 'pass.pre_grad_passes.apply_gumbel_max_trick_pass': [0.0],
 'pass.pre_grad_passes.efficient_conv_bn_eval_pass': [0.0],
 'pass.pre_grad_passes.group_batch_fusion_passes': [0.0]}"""
                if _IS_WINDOWS
                else """\
{'CacheBase.get_system.triton_key': [0.0],
 'GraphLowering.codegen': [0.0, 0.0],
 'GraphLowering.compile_to_fn': [0.0, 0.0],
 'GraphLowering.compile_to_module': [0.0, 0.0],
 'GraphLowering.run': [0.0, 0.0],
 'OutputGraph.call_user_compiler': [0.0],
 'PyCodeCache.load_by_key_path': [0.0, 0.0],
 'PythonWrapperCodegen.generate': [0.0, 0.0],
 'Scheduler.__init__': [0.0, 0.0],
 'Scheduler.codegen': [0.0, 0.0],
 'Scheduler.fused_nodes': [0.0, 0.0],
 '_compile.compile_inner': [0.0],
 '_recursive_joint_graph_passes': [0.0],
 '_recursive_post_grad_passes': [0.0, 0.0],
 '_recursive_pre_grad_passes': [0.0],
 'additional_fake_tensor_prop': [0.0, 0.0],
 'aot_collect_metadata': [0.0],
 'aot_trace_joint_graph': [0.0],
 'async_compile.wait': [0.0, 0.0],
 'backward._backward_impl': [0.0],
 'build_guards': [0.0],
 'bytecode_tracing': [0.0],
 'compile_attempt_0': [0.0],
 'compile_file': [0.0, 0.0],
 'compile_fx.<locals>.bw_compiler': [0.0],
 'compile_fx.<locals>.fw_compiler_base': [0.0],
 'compile_fx_inner': [0.0, 0.0],
 'create_aot_dispatcher_function': [0.0],
 'fx_codegen_and_compile': [0.0, 0.0],
 'gc': [0.0],
 'insert_deferred_runtime_asserts': [0.0],
 'min_cut_rematerialization_partition': [0.0],
 'pass.joint_graph_passes.constant_fold_uniform_value': [0.0],
 'pass.joint_graph_passes.pass_pattern_0': [0.0],
 'pass.joint_graph_passes.pass_pattern_1': [0.0],
 'pass.joint_graph_passes.remove_noop_ops': [0.0],
 'pass.post_grad_passes.decompose_auto_functionalized': [0.0, 0.0],
 'pass.post_grad_passes.decompose_map_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_scan_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_triton_kernel_wrapper_functional': [0.0, 0.0],
 'pass.post_grad_passes.fix_auto_functionalized_dtype_views': [0.0, 0.0],
 'pass.post_grad_passes.move_constructors_to_cuda': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_0': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_1': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_2': [0.0, 0.0],
 'pass.post_grad_passes.post_grad_custom_pre_pass': [0.0, 0.0],
 'pass.post_grad_passes.reinplace_inplaceable_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_assert_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_noop_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_profiler_ops': [0.0, 0.0],
 'pass.post_grad_passes.stable_sort': [0.0, 0.0],
 'pass.pre_grad_passes.apply_gumbel_max_trick_pass': [0.0],
 'pass.pre_grad_passes.efficient_conv_bn_eval_pass': [0.0],
 'pass.pre_grad_passes.group_batch_fusion_passes': [0.0]}"""
            ),
        )

        # Now validate utils.calculate_time_spent(). Formatting the return
        # value makes reading diffs much easier.
        time_spent = utils.calculate_time_spent()
        self.assertExpectedInline(
            pprint.pformat(time_spent),
            filter_expected(
                """\
{'_recursive_joint_graph_passes': 0.0,
 '_recursive_post_grad_passes': 0.0,
 '_recursive_pre_grad_passes': 0.0,
 'backend_compile': 0.0,
 'code_gen': 0.0,
 'entire_backward_compile': 0.0,
 'entire_frame_compile': 0.0,
 'gc': 0.0,
 'inductor_compile': 0.0,
 'total_wall_time': 0.0}"""
                if _IS_WINDOWS
                else """\
{'_recursive_joint_graph_passes': 0.0,
 '_recursive_post_grad_passes': 0.0,
 '_recursive_pre_grad_passes': 0.0,
 'async_compile.wait': 0.0,
 'backend_compile': 0.0,
 'code_gen': 0.0,
 'entire_backward_compile': 0.0,
 'entire_frame_compile': 0.0,
 'gc': 0.0,
 'inductor_compile': 0.0,
 'total_wall_time': 0.0}"""
            ),
        )

        # Now validate the CompilationMetrics logs. We expect a log for the
        # forward and a log for the backward.
        self.assertTrue(len(compilation_events) == 2)
        self.assertTrue(
            all(isinstance(e, utils.CompilationMetrics) for e in compilation_events)
        )

        # Remove a few fields that aren't helpful for test stability.
        for e in compilation_events:
            e.dynamo_config = None
            e.co_filename = None
            e.co_firstlineno = None
            e.inductor_config = None
            e.functorch_config = None
            e.compiler_config = None
            e.cuda_version = None
            e.triton_version = None
            e.python_version = None
            e.pytorch_version = None
            e.stack_trace = None
            e.graph_node_shapes = None
            e.exception_stack_trace = None

        # First event is for the forward. Formatting makes reading diffs
        # much easier.
        raw = dataclasses.asdict(compilation_events[0])
        del raw["feature_usage"]
        del raw["ir_count"]
        del raw["inductor_provenance"]
        del raw["param_numel"]
        del raw["param_bytes"]
        del raw["param_count"]
        # guard_latency_us is not deterministic
        del raw["guard_latency_us"]
        self.assertExpectedInline(
            pprint.pformat(raw),
            filter_expected(
                """\
{'accumulated_cache_size': 0,
 'aot_autograd_cumulative_compile_time_us': 0,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': 0.0,
 'backward_cumulative_compile_time_us': None,
 'cache_size': 0,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': 'forward',
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': set(),
 'config_inline_inbuilt_nn_modules': True,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': 0,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': 0,
 'dynamo_time_before_restart_s': 0.0,
 'end_time_us': 100,
 'entire_frame_compile_time_s': 0.0,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': '1',
 'functorch_config': None,
 'gc_time_us': 0,
 'graph_input_count': 3,
 'graph_node_count': 5,
 'graph_node_shapes': None,
 'graph_op_count': 1,
 'guard_count': 31,
 'has_guarded_code': True,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': True,
 'is_runtime': False,
 'joint_graph_pass_time_us': 0,
 'log_format_version': 3,
 'non_compliant_ops': set(),
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': 0,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': set(),
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': 0,
 'specialize_float': False,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': 0,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
                if _IS_WINDOWS
                else """\
{'accumulated_cache_size': 0,
 'aot_autograd_cumulative_compile_time_us': 0,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': 0.0,
 'backward_cumulative_compile_time_us': None,
 'cache_size': 0,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': 'forward',
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': set(),
 'config_inline_inbuilt_nn_modules': True,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': 0,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': 0,
 'dynamo_time_before_restart_s': 0.0,
 'end_time_us': 100,
 'entire_frame_compile_time_s': 0.0,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': '1',
 'functorch_config': None,
 'gc_time_us': 0,
 'graph_input_count': 3,
 'graph_node_count': 5,
 'graph_node_shapes': None,
 'graph_op_count': 1,
 'guard_count': 31,
 'has_guarded_code': True,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': True,
 'is_runtime': False,
 'joint_graph_pass_time_us': 0,
 'log_format_version': 3,
 'non_compliant_ops': set(),
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': 0,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': set(),
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': 0,
 'specialize_float': False,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': 0,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
            ),
        )

        # Second event is for the backward
        raw = dataclasses.asdict(compilation_events[1])
        del raw["feature_usage"]
        del raw["ir_count"]
        del raw["inductor_provenance"]
        del raw["guard_latency_us"]
        del raw["param_numel"]
        del raw["param_bytes"]
        del raw["param_count"]
        self.assertExpectedInline(
            pprint.pformat(raw),
            (
                """\
{'accumulated_cache_size': None,
 'aot_autograd_cumulative_compile_time_us': None,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': None,
 'backward_cumulative_compile_time_us': 0,
 'cache_size': None,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': None,
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': None,
 'config_inline_inbuilt_nn_modules': True,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': None,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': None,
 'dynamo_time_before_restart_s': None,
 'end_time_us': 100,
 'entire_frame_compile_time_s': None,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': None,
 'functorch_config': None,
 'gc_time_us': None,
 'graph_input_count': None,
 'graph_node_count': None,
 'graph_node_shapes': None,
 'graph_op_count': None,
 'guard_count': None,
 'has_guarded_code': None,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': False,
 'is_runtime': False,
 'joint_graph_pass_time_us': None,
 'log_format_version': 3,
 'non_compliant_ops': None,
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': None,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': None,
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': None,
 'specialize_float': None,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': None,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
                if _IS_WINDOWS
                else """\
{'accumulated_cache_size': None,
 'aot_autograd_cumulative_compile_time_us': None,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': None,
 'backward_cumulative_compile_time_us': 0,
 'cache_size': None,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': None,
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': None,
 'config_inline_inbuilt_nn_modules': True,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': None,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': None,
 'dynamo_time_before_restart_s': None,
 'end_time_us': 100,
 'entire_frame_compile_time_s': None,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': None,
 'functorch_config': None,
 'gc_time_us': None,
 'graph_input_count': None,
 'graph_node_count': None,
 'graph_node_shapes': None,
 'graph_op_count': None,
 'guard_count': None,
 'has_guarded_code': None,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': False,
 'is_runtime': False,
 'joint_graph_pass_time_us': None,
 'log_format_version': 3,
 'non_compliant_ops': None,
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': None,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': None,
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': None,
 'specialize_float': None,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': 0,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
            ),
        )

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    @compiler_config.patch({"job_id": "test_job_id"})
    def test_compiler_config(self):
        # [FUNCTION DOC] With a job id set in the compiler config, a compiled function's first
        #   compilation event carries a compiler config string including that job id.
        def test1(x):
            # [FUNCTION DOC] HELPER: Function returning the square of its input.
            return x * x

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(test1)(torch.randn(1))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertIn(
            '"job_id": "test_job_id"',
            compilation_events[0].compiler_config,
        )

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    def test_ir_count(self):
        # [FUNCTION DOC] The IR count recorded in the compilation event matches a
        #   per-Python-version expected value for two functions (one with an extra
        #   multiplication); the test also asserts that the running Python version is one of the
        #   supported listed versions.
        # Different python versions have different potential IR counts.
        version = (sys.version_info[0], sys.version_info[1])
        self.assertIn(version, ((3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15)))
        first, second = {
            (3, 10): (10, 6),
            (3, 11): (11, 7),
            (3, 12): (11, 7),
            (3, 13): (11, 7),
            (3, 14): (11, 7),
            (3, 15): (11, 7),
        }[version]

        def test1(x):
            # [FUNCTION DOC] HELPER: Function computing the square of the doubled input through
            #   intermediate variables.
            y = x + x
            z = y * y
            return z

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(test1)(torch.randn(10, 10))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].ir_count, first)

        def test2(x):
            # [FUNCTION DOC] HELPER: Function returning the doubled input.
            y = x + x
            return y

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(test2)(torch.randn(10, 10))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].ir_count, second)

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    @inductor_config.patch(
        {"trace.enabled": True, "trace.provenance_tracking_level": 1},
    )
    def test_inductor_provenance(self):
        # [FUNCTION DOC] With inductor tracing and provenance tracking enabled, compiling a
        #   traced linear module yields an event whose inductor provenance equals the expected
        #   set containing a single JSON string for the matrix-multiply extern kernel.
        module = torch.nn.Linear(6, 66)
        graph_module = torch.fx.symbolic_trace(module)

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(graph_module)(torch.randn(6, 6))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(
            compilation_events[0].inductor_provenance,
            {'{"extern_kernels.addmm:1": []}'},
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_dynamic_shape_feature_use(self):
        # [FUNCTION DOC] The feature-usage record in the second compilation event shows automatic
        #   dynamic shapes enabled by default after calling with a different size, and disabled
        #   when that option is turned off in config.
        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:

            @torch.compile()  # noqa: UNSPECIFIED_BACKEND
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning the square of its input.
                return x * x

            f(torch.randn(4))
            f(torch.randn(3))
            compilation_events = [
                arg[0][0].feature_usage for arg in log_event.call_args_list
            ]
        self.assertIn(
            ("dynamo.automatic_dynamic_shapes", True), compilation_events[1].items()
        )

        compilation_events = []
        with (
            dynamo_config.patch({"automatic_dynamic_shapes": False}),
            mock.patch("torch._dynamo.utils.log_compilation_event") as log_event,
        ):

            @torch.compile()  # noqa: UNSPECIFIED_BACKEND
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning the square of its input.
                return x * x

            f(torch.randn(4))
            f(torch.randn(3))
            compilation_events = [
                arg[0][0].feature_usage for arg in log_event.call_args_list
            ]
        self.assertIn(
            ("dynamo.automatic_dynamic_shapes", False), compilation_events[1].items()
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    def test_num_params(self):
        # [FUNCTION DOC] Checks parameter number, byte and count metrics in compilation events: a
        #   conv model (520 elements, 2 parameters), a wrapper holding two such models (1040, 4),
        #   a sequential with a shared layer (counted once per distinct parameter: 40 elements, 4
        #   parameters), and layers sharing one weight (24 elements, 3 parameters). Bytes are
        #   four times elements.
        import torch.nn as nn
        import torch.nn.functional as F

        class ModelSimple(nn.Module):
            # [CLASS DOC] HELPER: Module with one convolution layer (1 to 20 channels, kernel 5)
            #   followed by relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the convolution layer.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the convolution then relu.
                return F.relu(self.conv1(x))

        self.assertEqual([x.numel() for x in ModelSimple().parameters()], [500, 20])

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            m = ModelSimple()
            torch.compile(m)(torch.randn(1, 10, 10))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 520)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 520)
        self.assertEqual(compilation_events[0].param_count, 2)

        class ModelWrapped(nn.Module):
            # [CLASS DOC] HELPER: Module holding two simple conv models and summing their
            #   outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create two simple conv model submodules.
                super().__init__()
                self.m1 = ModelSimple()
                self.m2 = ModelSimple()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the sum of the two submodule outputs.
                return self.m1(x) + self.m2(x)

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            m = ModelWrapped()
            torch.compile(m)(torch.randn(1, 10, 10))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 1040)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 1040)
        self.assertEqual(compilation_events[0].param_count, 4)

        # Test a tied module
        l1 = nn.Linear(4, 4)
        l2 = nn.Linear(4, 4)
        m = nn.Sequential(l1, nn.Sequential(l1, l2))
        self.assertEqual([x.numel() for x in m.parameters()], [16, 4, 16, 4])
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(m)(torch.randn(4, 4))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 40)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 40)
        self.assertEqual(compilation_events[0].param_count, 4)

        # Test tied weights
        l1 = nn.Linear(4, 4)
        l2 = nn.Linear(4, 4)
        l1.weight = l2.weight
        m = nn.Sequential(l1, nn.Sequential(l2))
        self.assertEqual([x.numel() for x in m.parameters()], [16, 4, 4])
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(m)(torch.randn(4, 4))  # noqa: UNSPECIFIED_BACKEND
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 24)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 24)
        self.assertEqual(compilation_events[0].param_count, 3)


class TestTimedSync(TestCase):
    # [CLASS DOC] Device-generic tests (instantiated per device at import) of the timing helper's
    #   accelerator synchronization.
    def test_timed_syncs_on_accelerator(self, device):
        # [FUNCTION DOC] On non-CPU devices, the timing helper on a traced identity module with
        #   one repetition must call the accelerator synchronize function exactly twice (once
        #   before the loop, once per iteration); skipped on CPU. The synchronize function is
        #   patched to record calls.
        if torch.device(device).type == "cpu":
            self.skipTest("sync only triggered for non-CPU devices")

        from torch._dynamo.utils import timed

        called = []
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        inputs = [torch.randn(4, device=device)]

        with patch(
            "torch.accelerator.synchronize", side_effect=lambda: called.append(1)
        ):
            timed(gm, inputs, times=1)

        # once before loop + once per iteration for times=1
        self.assertEqual(len(called), 2)


instantiate_device_type_tests(TestTimedSync, globals(), allow_xpu=True)


class TestInductorConfigParsingForLogging(TestCase):
    # [CLASS DOC] Tests of the function that converts the inductor config into a JSON string for
    #   compilation metrics logging.
    """
    Test for parsing inductor config for logging in CompilationMetrics.
    """

    class TestObject:
        # [CLASS DOC] HELPER: Simple object with two attributes; not used by any test in this
        #   file.
        def __init__(self, a, b):
            # [FUNCTION DOC] HELPER: Store the two arguments.
            self.a = a
            self.b = b

    def test_inductor_config_jsonify(self):
        # [FUNCTION DOC] The real inductor config converts to a string that contains a trace key.
        """
        Sanity check if the actual inductor config is parsed correctly
        """

        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertTrue(isinstance(inductor_config_json, str))
        self.assertIn('trace"', inductor_config_json)

    @mock.patch("torch._dynamo.utils.torch._inductor.config")
    def test_inductor_config_parsing_non_conforming_items(self, mocked_inductor_config):
        # [FUNCTION DOC] With the inductor config object patched, checks scrubbing:
        #   unserializable values nested in dicts and lists become a placeholder text and keys
        #   sorted; a mapping with a non-string key yields an empty JSON object; a non-mapping
        #   config gives a not-serializable message; a missing config gives none. Takes the
        #   patched mock config.
        """
        Test if the inductor config is parsed correctly when the config is
            - None
            - not a dict
            - not json serializable
            - complex unserializable objects
        """
        obj = TestCase
        test_mock_config = {
            "some": {"name": obj, "some": True},
            "data": {"name": obj, "some": True},
            "list": [
                {"name": obj, "some": True},
                {"name": obj, "some": True},
            ],
            "object": {
                "name": obj,
                "some": True,
                "data": {"name": obj, "some": True},
            },
        }
        expected = (
            """{"data": {"name": "Value is not JSON serializable", "some": true}, """
            """"list": [{"name": "Value is not JSON serializable", "some": true}, """
            """{"name": "Value is not JSON serializable", "some": true}], """
            """"object": {"data": {"name": "Value is not JSON serializable", "some": true}, """
            """"name": "Value is not JSON serializable", "some": true}, """
            """"some": {"name": "Value is not JSON serializable", "some": true}}"""
        )
        mocked_inductor_config.get_config_copy.return_value = test_mock_config
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)

        expected = "{}"
        mocked_inductor_config.get_config_copy.return_value = {obj: obj}
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)

        expected = "Inductor Config is not JSON serializable"
        mocked_inductor_config.get_config_copy.return_value = obj
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)

        expected = None
        mocked_inductor_config.get_config_copy.return_value = None
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)


class TestFunctorchConfigParsingForLogging(TestCase):
    # [CLASS DOC] Tests of the function that converts the functorch config into a JSON string for
    #   compilation metrics logging.
    def test_functorch_config_jsonify(self):
        # [FUNCTION DOC] The functorch config string parses as JSON and includes the activation
        #   memory budget with value 1.0.
        functorch_config_json = utils._functorch_config_for_logging()
        self.assertIsInstance(functorch_config_json, str)
        parsed = json.loads(functorch_config_json)
        self.assertIn("activation_memory_budget", parsed)
        self.assertEqual(parsed["activation_memory_budget"], 1.0)

    def test_functorch_config_blocklist(self):
        # [FUNCTION DOC] The functorch config string leaves out the type-checking flag and the
        #   private ignore-list entry.
        functorch_config_json = utils._functorch_config_for_logging()
        parsed = json.loads(functorch_config_json)
        self.assertNotIn("TYPE_CHECKING", parsed)
        self.assertNotIn("_save_config_ignore", parsed)

    @mock.patch("torch._dynamo.utils.torch._functorch.config")
    def test_functorch_config_none(self, mocked_config):
        # [FUNCTION DOC] When reading a copy of the functorch config raises AttributeError, the
        #   logging function returns none. Takes the patched mock config.
        mocked_config.get_config_copy.side_effect = AttributeError
        result = utils._functorch_config_for_logging()
        self.assertIsNone(result)

    @mock.patch("torch._dynamo.utils.torch._functorch.config")
    def test_functorch_config_runtime_error(self, mocked_config):
        # [FUNCTION DOC] When reading a copy of the functorch config raises RuntimeError, the
        #   logging function returns none. Takes the patched mock config.
        mocked_config.get_config_copy.side_effect = RuntimeError
        result = utils._functorch_config_for_logging()
        self.assertIsNone(result)

    @mock.patch("torch._dynamo.utils.justknobs_check", return_value=False)
    def test_functorch_config_jk_disabled(self, mocked_jk):
        # [FUNCTION DOC] When the just-knobs check is patched to return false, the logging
        #   function returns none and the check was consulted once with the functorch-config
        #   logging knob name. Takes the patched mock.
        result = utils._functorch_config_for_logging()
        self.assertIsNone(result)
        mocked_jk.assert_called_once_with("pytorch/dynamo:log_functorch_config")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
