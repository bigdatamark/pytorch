# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the interaction between dynamo and the torch profiler: dynamo_timed and
# compile events showing up in profiles, chromium event logger scoping and thread-locality, cache
# lookup events, compiled region annotations, execution trace with dynamic shapes, and capture of
# record_function contexts into graphs. No import-time side effects besides imports.
import threading
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch._dynamo.utils import chromium_event_timed, ChromiumEventLogger, dynamo_timed
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.ctx_manager import ProfilerRecordFunctionContextVariable
from torch.profiler import record_function
from torch.testing._internal.common_utils import TemporaryFileName


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo test case with profiler-related tests. Most tests run small functions
    # under a profiler and inspect event names; others use a private ChromiumEventLogger patched
    # into the dynamo utils module.
    def test_dynamo_timed_profiling_isolated(self):
        # [FUNCTION DOC] Profiles a call to a function using the dynamo_timed context manager and
        # checks an event named with the timed label plus a dynamo_timed suffix appears.
        # dynamo_timed functions should appear in profile traces.
        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of the input inside a dynamo_timed region
            # labeled inner_fn.
            with dynamo_timed("inner_fn"):
                return x.sin()

        def outer_fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the timed helper and multiplies its result by the
            # second input.
            return inner_fn(x) * y

        x, y = (torch.rand((2, 2)) for _ in range(2))

        with torch.profiler.profile(with_stack=False) as prof:
            outer_fn(x, y)

        self.assertTrue(
            any("inner_fn (dynamo_timed)" in evt.name for evt in prof.events())
        )

    def test_dynamo_timed_profiling_backend_compile(self):
        # [FUNCTION DOC] Profiles a real torch.compile with the aot_eager backend and checks a
        # profiler event for the user compiler call (dynamo_timed suffix) is present.
        # dynamo_timed functions should appear in profile traces.
        # this checks whether these actually appear in actual dynamo execution.
        # "backend_compile" is just chosen as an example; if it gets renamed
        # this test can be replaced or deleted

        fn_name = "call_user_compiler"

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test: sine of first input times cosine of
            # second.
            return x.sin() * y.cos()

        x, y = (torch.rand((2, 2)) for _ in range(2))

        with torch.profiler.profile(with_stack=False) as prof:
            torch.compile(fn, backend="aot_eager")(x, y)

        self.assertTrue(
            any(f"{fn_name} (dynamo_timed)" in evt.name for evt in prof.events())
        )

    def test_compile_events_visible_in_profiler(self):
        # [FUNCTION DOC] Resets dynamo, profiles compiling a simple add function and checks at
        # least one event name carries the dynamo_timed marker.
        torch._dynamo.reset()

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: adds one to the input.
            return x + 1

        with torch.profiler.profile(with_stack=False) as prof:
            torch.compile(fn, backend="aot_eager")(torch.randn(2, 2))

        event_names = [e.name for e in prof.events()]

        # Check for the main dynamo event (follows pattern from test_dynamo_timed_profiling_backend_compile)
        self.assertTrue(
            any("(dynamo_timed)" in name for name in event_names),
            lambda msg: f"{msg}\nExpected dynamo_timed events in profiler: {event_names}",
        )

    def test_record_functions_thread_local(self):
        # [FUNCTION DOC] Starts three threads each fetching the logger's record functions
        # mapping, adding one entry, and recording its size; checks every thread saw size one,
        # proving the mapping is thread-local.
        """Verify record_functions dict is thread-local (no cross-thread contamination)"""

        logger = ChromiumEventLogger()
        results = {}

        def thread_func(thread_id):
            # [FUNCTION DOC] HELPER: Thread body: gets the record functions mapping from the
            # shared logger, adds a thread-specific entry and stores the mapping size in the
            # shared results mapping under the thread id.
            # Each thread creates its own record_functions dict
            rf = logger.get_record_functions()
            rf[f"thread_{thread_id}"] = f"value_{thread_id}"
            results[thread_id] = len(rf)

        threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should only see its own entry (length 1)
        for thread_id, length in results.items():
            self.assertEqual(
                length,
                1,
                lambda msg: f"{msg}\nThread {thread_id} saw {length} entries instead of 1",
            )

    def test_chromium_event_timed_scoped_reset_preserves_outer_event(self):
        # [FUNCTION DOC] With a logger installed as the global chromium log, starts an outer
        # event, then enters a timed inner event with reset on exit. Checks the stack, pt2
        # compile substack and event data keep the outer event after the inner scope exits, no
        # warnings are logged, and after ending the outer event everything is empty.
        logger = ChromiumEventLogger()

        with patch.object(torch._dynamo.utils, "CHROMIUM_EVENT_LOG", logger):
            logger.log_event_start(
                "outer", 0, {"outer_metadata": True}, log_pt2_compile_event=True
            )
            with self.assertNoLogs("torch._dynamo.utils", level="WARNING"):
                with chromium_event_timed("inner", reset_event_log_on_exit=True):
                    self.assertEqual(logger.get_stack(), ["outer", "inner"])

                self.assertEqual(logger.get_stack(), ["outer"])
                self.assertEqual(logger.get_pt2_compile_substack(), ["outer"])
                self.assertIn("outer", logger.get_event_data())

                logger.log_event_end(
                    "outer",
                    1,
                    {},
                    0,
                    log_pt2_compile_event=True,
                )

            self.assertEqual(logger.get_stack(), [])
            self.assertEqual(logger.get_pt2_compile_substack(), [])
            self.assertEqual(logger.get_event_data(), {})

    def test_chromium_event_timed_scoped_reset_preserves_outer_record_function(self):
        # [FUNCTION DOC] Under a profiler with a patched global logger, starts an outer event and
        # enters a timed inner event with reset on exit; checks record function entries for outer
        # and inner exist inside and only the outer remains afterwards, that ending the outer
        # event empties the mapping, and that both names appear in profiler events.
        logger = ChromiumEventLogger()

        with (
            patch.object(torch._dynamo.utils, "CHROMIUM_EVENT_LOG", logger),
            torch.profiler.profile(with_stack=False) as prof,
        ):
            logger.log_event_start("outer", 0, {})
            self.assertIn("outer", logger.get_record_functions())

            with chromium_event_timed("inner", reset_event_log_on_exit=True):
                self.assertIn("outer", logger.get_record_functions())
                self.assertIn("inner", logger.get_record_functions())

            self.assertEqual(logger.get_stack(), ["outer"])
            self.assertIn("outer", logger.get_record_functions())
            self.assertNotIn("inner", logger.get_record_functions())

            logger.log_event_end(
                "outer",
                1,
                {},
                0,
                log_pt2_compile_event=False,
            )
            self.assertEqual(logger.get_record_functions(), {})

        event_names = [event.name for event in prof.events()]
        self.assertIn("outer", event_names)
        self.assertIn("inner", event_names)

    def test_nested_compilation_events_in_profiler(self):
        # [FUNCTION DOC] Profiles compiling a function with the inductor backend and checks that
        # some event names mention dynamo and some mention inductor (lowercased comparison).
        """Verify nested compilation events (dynamo→inductor) show hierarchy"""
        torch._dynamo.reset()

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: input doubled plus its sine.
            return x * 2 + torch.sin(x)

        with torch.profiler.profile(with_stack=False) as prof:
            torch.compile(fn, backend="inductor")(torch.randn(10, 10))

        event_names = [e.name.lower() for e in prof.events()]

        # Should see both parent and child events
        has_dynamo = any("dynamo" in name for name in event_names)
        has_inductor = any("inductor" in name for name in event_names)

        self.assertTrue(has_dynamo, "Missing dynamo event in nested compilation")
        self.assertTrue(has_inductor, "Missing inductor event in nested compilation")

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_runtime(self):
        # [FUNCTION DOC] With assume_static_by_default patched off, compiles a matmul-add with
        # dynamic shapes, runs it for two shape sets to get a dynamic graph, then runs a third
        # shape set under a profiler that records shapes. Only checks nothing raises.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Matrix multiply of the first two inputs plus the third.
            return x @ y + z

        opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)

        inputs = [
            (torch.rand(a, b), torch.rand(b, c), torch.rand(a, c))
            for (a, b, c) in [(15, 16, 17), (15, 15, 16), (16, 16, 16)]
        ]

        opt_fn(*inputs[0])
        opt_fn(*inputs[1])

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs[2])

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_compilation(self):
        # [FUNCTION DOC] With assume_static_by_default patched off, compiles a dynamic-shape
        # matmul-add for the first time under a shape-recording profiler. Only checks nothing
        # raises.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Matrix multiply of the first two inputs plus the third.
            return x @ y + z

        opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)

        inputs = (torch.rand(15, 16), torch.rand(16, 17), torch.rand(15, 17))

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs)

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_list_compilation(self):
        # [FUNCTION DOC] Like the previous compile-under-profiler test but the function
        # concatenates a list of two tensors along the first dimension before adding the third;
        # exercises shape recording of list inputs. Only checks nothing raises.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Concatenates the first two inputs along dimension zero and
            # adds the third.
            return torch.cat([x, y], dim=0) + z

        opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)

        inputs = (torch.rand(4, 16), torch.rand(12, 16), torch.rand(16, 16))

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs)

    def test_execution_trace_dynamic_shapes(self):
        # [FUNCTION DOC] Compiles a dynamic-shape function and runs it while an execution trace
        # observer is registered to a temporary file; only checks nothing raises.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Matrix multiply of the first two inputs plus the third.
            return x @ y + z

        et = torch.profiler.ExecutionTraceObserver()
        opt_fn = torch.compile(fn, dynamic=True, backend="aot_eager")
        inputs = [torch.rand((4, 4)) for _ in range(3)]

        with TemporaryFileName() as fname:
            et.register_callback(fname)
            et.start()
            opt_fn(*inputs)
            et.stop()
            et.unregister_callback()

    def test_profiler_cache_lookup(self):
        # [FUNCTION DOC] For both the autograd profiler and the torch profiler, warms up a
        # compiled function then profiles one call; checks the result equals eager and exactly
        # one event with the cache lookup name appears.
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares the input, adds two and cubes the result.
            y = x**2
            y = y + 2
            z = y**3
            return z

        for profiler, get_events in (
            (torch.autograd.profiler.profile, lambda prof: prof.function_events),
            (torch.profiler.profiler.profile, lambda prof: prof.events()),
        ):
            x = torch.randn((2, 2), requires_grad=True)
            ref = fn(x)
            opt_fn = torch.compile(fn, backend="aot_eager")

            # warmup
            opt_fn(x)

            with profiler() as prof:
                res = opt_fn(x)
            events = list(
                filter(
                    lambda event: "TorchDynamo Cache Lookup" in event.name,
                    get_events(prof),
                )
            )

            self.assertEqual(ref, res)
            self.assertTrue(
                len(events) == 1,
                "Expected one lookup profiler event for one opt_fn run",
            )

    def test_profiler_guardless_cache_hit_skips_guard_lookup(self):
        # [FUNCTION DOC] Compiles with all guards skipped, runs once to compile, then profiles a
        # second call; checks the result is correct and no cache lookup event was emitted.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one to the input.
            return x + 1

        x = torch.randn((2, 2))
        opt_fn = torch.compile(
            fn,
            backend="eager",
            options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
        )
        opt_fn(x)

        with torch.profiler.profile() as prof:
            self.assertEqual(opt_fn(x), fn(x))

        events = [
            event for event in prof.events() if "TorchDynamo Cache Lookup" in event.name
        ]
        self.assertEqual(events, [])

    def test_profiler_cache_lookup_profiler_step(self):
        # [FUNCTION DOC] Runs a compiled function ten times under a profiler with a wait, warmup
        # and active schedule, stepping the profiler each iteration; checks the cache lookup
        # event is recorded in the profile events.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Subtracts the second input from the first and adds the
            # third.
            return torch.add(torch.sub(x, y), z)

        opt_fn = torch.compile(fn, backend="aot_eager")

        (
            x,
            y,
            z,
        ) = (torch.rand(4, 4) for _ in range(3))

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=2, repeat=1)
        )

        for _ in range(10):
            opt_fn(x, y, z)
            prof.step()

        self.assertTrue(
            any(e.name == "TorchDynamo Cache Lookup" for e in prof.events())
        )

    def test_profiler_enabled_export(self):
        # [FUNCTION DOC] Exports a module whose forward branches on whether the autograd profiler
        # is enabled, once without a profiler and once under one (re-exporting because export
        # skips guards), checking the exported graph module matches eager each time.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies sine and then cosine if the
            # autograd profiler is enabled, else sigmoid.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent constructor only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies sine, then returns cosine if the profiler is
                # enabled, otherwise sigmoid.
                x = torch.sin(x)
                if torch.autograd._profiler_enabled():
                    return torch.cos(x)
                else:
                    return torch.sigmoid(x)

        mod = Mod()

        x = torch.randn(4)
        opt_mod = torch._dynamo.export(mod, (x))

        ref = mod(x)
        res = opt_mod.graph_module(x)
        self.assertEqual(ref, res)

        with torch.autograd.profiler.profile():
            ref = mod(x)
            # Reexport because export skips guards
            opt_mod = torch._dynamo.export(mod, (x))
            res = opt_mod.graph_module(x)
            self.assertEqual(ref, res)

    def test_profiler_dynamo_compiled_region(self):
        # [FUNCTION DOC] Compiles a function containing a print (a graph break) and calls it with
        # two input shapes under a profiler; checks the list of compiled region annotation event
        # names equals the expected sequence of region ids with their recompile indices.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums the second input over dimension one, prints the shape
            # and multiplies the first input by it.
            r = y.sum(dim=1)
            print(r.shape)
            return x * r

        with torch.profiler.profile() as prof:
            fn_c = torch.compile(fn, backend="eager")

            fn_c(
                torch.randn(10),
                torch.randn(10, 10),
            )

            fn_c(
                torch.randn(10),
                torch.randn(10, 15),
            )

        annotations = [e.name for e in prof.events() if "Torch-Compiled" in e.name]
        self.assertEqual(
            annotations,
            [
                "Torch-Compiled Region: 0/0",
                "Torch-Compiled Region: 1/0",
                "Torch-Compiled Region: 0/1",
                "Torch-Compiled Region: 1/0",
            ],
        )

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_dynamo_preserve_record_func(self):
        # [FUNCTION DOC] With capture of profiler record function enabled, compiles a function
        # with three record_function regions and checks the dynamo graph and the AOT forward
        # graph contain enter and exit profiler ops for each region (expected text), and that
        # running under a profiler emits the three region names in order.
        def fn(x):
            # [FUNCTION DOC] HELPER: Runs sine, cosine and add two in separate named record
            # function regions.
            with record_function("my_net1"):
                a = x.sin()
            with record_function("my_cos"):
                b = a.cos()
            with record_function("my_net2"):
                c = b + 2
            return c

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        fn_c = torch.compile(fn, backend=backend)
        fn_c(
            torch.randn(10),
        )
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new('my_net1', None)
    sin = l_x_.sin();  l_x_ = None
    _record_function_exit__record_function = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit__record_function = None
    _record_function_enter_new_1 = torch.ops.profiler._record_function_enter_new('my_cos', None)
    cos = sin.cos();  sin = None
    _record_function_exit__record_function_1 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_1);  _record_function_enter_new_1 = _record_function_exit__record_function_1 = None
    _record_function_enter_new_2 = torch.ops.profiler._record_function_enter_new('my_net2', None)
    add = cos + 2;  cos = None
    _record_function_exit__record_function_2 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_2);  _record_function_enter_new_2 = _record_function_exit__record_function_2 = None
    return (add,)""",
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new.default('my_net1')
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    _record_function_exit = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit = None
    _record_function_enter_new_1 = torch.ops.profiler._record_function_enter_new.default('my_cos')
    cos = torch.ops.aten.cos.default(sin);  sin = None
    _record_function_exit_1 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_1);  _record_function_enter_new_1 = _record_function_exit_1 = None
    _record_function_enter_new_2 = torch.ops.profiler._record_function_enter_new.default('my_net2')
    add = torch.ops.aten.add.Tensor(cos, 2);  cos = None
    _record_function_exit_2 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_2);  _record_function_enter_new_2 = _record_function_exit_2 = None
    return (add,)""",
        )
        with torch.profiler.profile() as prof:
            fn_c(
                torch.randn(10),
            )

        annotations = [e.name for e in prof.events() if "my_" in e.name]
        self.assertEqual(
            annotations,
            [
                "my_net1",
                "my_cos",
                "my_net2",
            ],
        )

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_profiler_record_function_create_uses_record_args_and_kwargs(self):
        # [FUNCTION DOC] With record function capture enabled, checks the context variable
        # factory takes its target values from positional arguments and from keyword arguments
        # (name and args), giving the expected lists of constant values.
        ctx = ProfilerRecordFunctionContextVariable.create(
            func=record_function,
            record_args=[ConstantVariable.create("with_args")],
            record_kwargs={"args": ConstantVariable.create("payload")},
        )

        self.assertEqual(ctx.target_values, ["with_args", "payload"])

        ctx = ProfilerRecordFunctionContextVariable.create(
            func=record_function,
            record_args=[],
            record_kwargs={
                "name": ConstantVariable.create("with_kwargs"),
                "args": ConstantVariable.create("kw_payload"),
            },
        )

        self.assertEqual(ctx.target_values, ["with_kwargs", "kw_payload"])

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_dynamo_preserve_record_func_decorator(self):
        # [FUNCTION DOC] With record function capture enabled, compiles a function decorated with
        # record_function; checks the dynamo graph has enter and exit ops around the body
        # (expected text) and the profiler shows exactly one event named for the region.
        @record_function("my_net")
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated function that computes the sine of the input plus
            # two.
            a = x.sin()
            return a + 2

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        fn_c = torch.compile(fn, backend=backend)
        fn_c(torch.randn(10))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_args_0_ : torch.Tensor):
    l_args_0_ = L_args_0_
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new('my_net', None)
    sin = l_args_0_.sin();  l_args_0_ = None
    add = sin + 2;  sin = None
    _record_function_exit__record_function = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit__record_function = None
    return (add,)""",
        )
        with torch.profiler.profile() as prof:
            fn_c(torch.randn(10))

        annotations = [e.name for e in prof.events() if "my_" in e.name]
        self.assertEqual(annotations, ["my_net"])

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_dynamo_preserve_record_func_with_graph_break(self):
        # [FUNCTION DOC] With record function capture enabled, compiles a function with two
        # record_function regions separated by a graph break. Checks two graphs are produced,
        # each containing the matching region name and enter and exit ops, and the profiler shows
        # both names in order.
        # Test that record_function works correctly with graph breaks
        def fn(x):
            # [FUNCTION DOC] HELPER: Sine in a pre-break region, a graph break, then cosine in a
            # post-break region.
            with record_function("pre_graph_break"):
                a = x.sin()
            # This causes a graph break
            torch._dynamo.graph_break()
            with record_function("post_graph_break"):
                b = a.cos()
            return b

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        fn_c = torch.compile(fn, backend=backend)
        fn_c(
            torch.randn(10),
        )

        # We expect 2 graphs due to the graph break
        self.assertEqual(len(backend.graphs), 2)

        # First graph should have the pre_graph_break record_function
        self.assertIn("pre_graph_break", backend.graphs[0].code)
        self.assertIn("_record_function_enter_new", backend.graphs[0].code)
        self.assertIn("_record_function_exit", backend.graphs[0].code)

        # Second graph should have the post_graph_break record_function
        self.assertIn("post_graph_break", backend.graphs[1].code)
        self.assertIn("_record_function_enter_new", backend.graphs[1].code)
        self.assertIn("_record_function_exit", backend.graphs[1].code)

        # Verify profiler events work correctly
        with torch.profiler.profile() as prof:
            fn_c(
                torch.randn(10),
            )

        annotations = [
            e.name
            for e in prof.events()
            if e.name in ["pre_graph_break", "post_graph_break"]
        ]
        # Both record_function contexts should appear in profiler events
        self.assertEqual(
            annotations,
            [
                "pre_graph_break",
                "post_graph_break",
            ],
        )

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_dynamo_preserve_record_func_spanning_graph_break(self):
        # [FUNCTION DOC] With record function capture enabled, compiles a function whose
        # record_function region spans a graph break. Despite the leading comment about an error,
        # it only checks that the result matches eager and the profiler has exactly one event
        # with that name.
        # KNOWN ISSUE: the inline comment says a spanning region raises an error, but the test
        #     asserts a normal result and a single profiler event rather than an exception.
        # Test that record_function that spans across a graph break raises an error
        # This prevents the confusing behavior where the context gets duplicated across graphs
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, then inside a record_function region takes sine,
            # graph breaks, takes cosine, and finally subtracts one.
            x = x + 1
            with record_function("spanning_context"):
                a = x.sin()
                torch._dynamo.graph_break()
                b = a.cos()
            b = b - 1
            return b

        fn_c = torch.compile(fn, backend="aot_eager")
        x = torch.randn(10)
        fn_c(x)
        with torch.profiler.profile() as prof:
            result = fn_c(x)

        self.assertEqual(fn(x), result)

        annotations = [e.name for e in prof.events() if e.name == "spanning_context"]
        # record_function contexts should appear in profiler events once
        self.assertEqual(
            annotations,
            [
                "spanning_context",
            ],
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
