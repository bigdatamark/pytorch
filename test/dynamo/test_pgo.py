# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for dynamo profile-guided optimization (PGO), which remembers dynamic-shape
# information across compilations (locally and via a mocked remote cache) so later runs start
# with dynamic dimensions. Covers dynamic source allowlists, job ids, extra read/write keys, file
# path unification and NJT inputs. No import-time side effects besides imports.

import contextlib
import importlib.util
import os
import re
import tempfile

import torch._dynamo.config
import torch._dynamo.test_case
import torch._inductor.mock_cache as mock_cache
import torch.compiler.config
import torch.nested
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    normalize_gm,
)
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.utils import clear_caches, fresh_cache
from torch.testing._internal.common_utils import IS_WINDOWS


class PgoTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case that enables local PGO, sets the compiler job id to the test id, and
    # (unless disabled by an environment variable) uses a fresh cache directory and mock caches.
    # Holds an exit stack of the patches so tearDown can undo them. The reset method clears
    # dynamo and inductor caches between simulated runs.
    def setUp(self):
        # [FUNCTION DOC] Fixture: after the parent setUp, enters on an exit stack a job id patch
        # equal to the test id, a config patch enabling local automatic dynamic PGO, a fresh
        # inductor cache unless the INDUCTOR_TEST_DISABLE_FRESH_CACHE variable equals 1, and then
        # starts the mock cache patching.
        super().setUp()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(torch.compiler.config.patch(job_id=self.id()))
        self._test_stack.enter_context(
            torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
        )
        if os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1":
            self._test_stack.enter_context(fresh_cache())
        mock_cache.PatchCaches.setUp()

    def tearDown(self):
        # [FUNCTION DOC] Fixture: calls the parent tearDown, resets dynamo, closes the exit stack
        # to undo the config and cache patches, and stops the mock cache patching.
        super().tearDown()
        torch._dynamo.reset()
        self._test_stack.close()
        mock_cache.PatchCaches.tearDown()

    def reset(self):
        # [FUNCTION DOC] HELPER: Simulates a new process by resetting dynamo state and clearing
        # inductor caches, leaving PGO state that was persisted to cache.
        torch._dynamo.reset()
        clear_caches()

    def test_basic(self):
        # [FUNCTION DOC] Two calls with different second dimensions compile twice; after
        # resetting, two calls with new sizes compile only once because PGO makes that dimension
        # dynamic from the start.
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input doubled.
            return x * 2

        f(torch.randn(2, 3))
        f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        f(torch.randn(2, 5))
        f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(
        force_parameter_static_shapes=False,
        force_nn_module_property_static_shapes=False,
    )
    def test_whitelist_suggestion(self):
        # [FUNCTION DOC] With parameter and module property static shape forcing disabled, checks
        # the suggested dynamic-sources allowlist rendered from PGO state grows as more inputs,
        # weights, biases and attributes change shape, that five sources are reported missing
        # from the allowlist, and that after a reset, using the suggested allowlist as the
        # dynamic sources config yields a single compiled frame and no missing sources.
        from torch._dynamo.pgo import (
            _collect_dynamic_sources,
            _collect_missing_sources,
            get_code_state,
            render_code_state,
        )

        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Compiled module with a linear layer of 4 by 4 and a random
            # vector attribute, whose forward adds the layer output, the attribute and a second
            # input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer and the random attribute.
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.attr = torch.randn(4)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to the first input plus
                # the attribute plus the second input.
                return self.lin(x) + self.attr + y

        sources = [
            "L['x']",
            "L['self']._modules['lin']._parameters['weight']",
            "L['self']._modules['lin']._parameters['bias']",
            "L['self'].attr",
            "L['y']",
        ]

        def check_whitelist(sources_):
            # [FUNCTION DOC] HELPER: Renders the current PGO code state, extracts the dynamic
            # sources allowlist string from it, and asserts every given source name appears in
            # that string.
            state = render_code_state(get_code_state())
            whitelist = re.search(r'TORCH_COMPILE_DYNAMIC_SOURCES="(.*)"', state).group(
                1
            )
            for src in sources_:
                self.assertTrue(src in whitelist)

        def check_num_missing_whitelist(expected):
            # [FUNCTION DOC] HELPER: Collects dynamic sources from the first frame state, finds
            # those missing from the dynamic sources config, and asserts their count equals the
            # expected number.
            frame_state = next(iter(get_code_state().values()))
            all_dynamic_sources = _collect_dynamic_sources(frame_state)
            missing_whitelist = _collect_missing_sources(all_dynamic_sources)
            self.assertEqual(len(missing_whitelist), expected)

        # check growing whitelist
        f = Foo()
        f(torch.randn(2, 4), torch.randn(4))
        # only x
        f(torch.randn(4, 4), torch.randn(4))
        check_whitelist(sources[:1])
        # x, lin.weight
        f.lin = torch.nn.Linear(8, 4)
        f(torch.randn(8, 8), torch.randn(4))
        check_whitelist(sources[:2])
        # x, y, lin.weight, lin.bias, attr
        f.lin = torch.nn.Linear(8, 8)
        f.attr = torch.randn(8)
        f(torch.randn(8, 8), torch.randn(8))
        check_whitelist(sources)
        check_num_missing_whitelist(5)

        # now use suggested whitelist
        self.reset()
        cnts.clear()
        code_state = get_code_state()
        state = render_code_state(code_state)
        whitelist = re.search(r'TORCH_COMPILE_DYNAMIC_SOURCES="(.*)"', state).group(1)
        with torch.compiler.config.patch(dynamic_sources=whitelist):
            f = Foo()
            f(torch.randn(2, 4), torch.randn(4))
            f(torch.randn(4, 4), torch.randn(4))
            f.lin = torch.nn.Linear(8, 8)
            f.attr = torch.randn(8)
            f(torch.randn(8, 8), torch.randn(8))
            self.assertEqual(cnts.frame_count, 1)
            check_num_missing_whitelist(0)

    def test_no_empty_graph_allowlist(self):
        # [FUNCTION DOC] Checks the allowlist-logged flag in the pgo module: a compiled function
        # that only calls a disabled function yields an empty graph and the flag stays false
        # after two shapes; a function that has real ops around the disabled call sets the flag
        # to true.
        @torch._dynamo.disable
        def g(x):
            # [FUNCTION DOC] HELPER: Function excluded from dynamo that returns twice the input
            # plus the input.
            return x * 2 + x

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (eager backend) function that only calls the
            # disabled function.
            return g(x)

        self.reset()
        f(torch.randn(4))
        f(torch.randn(8))
        self.assertEqual(torch._dynamo.pgo._LOGGED_DYNAMIC_ALLOWLIST, False)

        @torch.compile(backend="eager")
        def f1(x):
            # [FUNCTION DOC] HELPER: Compiled (eager backend) function that adds two to the
            # input, calls the disabled function and adds two.
            return g(x + 2) + 2

        self.reset()
        f1(torch.randn(4))
        f1(torch.randn(8))
        self.assertEqual(torch._dynamo.pgo._LOGGED_DYNAMIC_ALLOWLIST, True)

    def test_pgo_dynamic_false(self):
        # [FUNCTION DOC] Compiles a module with dynamic disabled, whose forward has a graph
        # break, and runs it with two different shape pairs. Checks that in every frame code
        # state the recorded sizes for both inputs are overwritten with the last static sizes.
        @torch.compile(backend="eager", dynamic=False)
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module compiled with dynamic disabled; forward mutates both
            # inputs, forces a graph break, then undoes and returns them.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Adds two to both inputs in place, graph breaks,
                # subtracts two from the first and doubles the second in place, returns both.
                x += 2
                y += 2
                torch._dynamo.graph_break()
                x -= 2
                y *= 2
                return x, y

        self.reset()
        f = Foo()
        f(torch.randn(2, 4), torch.randn(2, 4))
        f(torch.randn(4, 4), torch.randn(6, 8))

        # check PGO code state is overwritten with static value, both before/after graph break
        for code_state in torch._dynamo.pgo.get_code_state().values():
            self.assertEqual(code_state.automatic_dynamic["L['x']"].size, (4, 4))
            self.assertEqual(code_state.automatic_dynamic["L['y']"].size, (6, 8))

    def test_whitelist_ints_floats(self):
        # [FUNCTION DOC] Runs a compiled module whose int property and inputs change, then checks
        # the rendered allowlist includes the changing tensor-like int and float inputs and the
        # changing int property, but not the ephemeral float tensor source, the constant float
        # input or the constant float property.
        @torch.compile(backend="eager", fullgraph=True)
        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Compiled module storing a float and an int, whose forward adds
            # one to the first input if the int equals two and, if the float equals one, returns
            # the sum of inputs and a tensor made from the third argument.
            def __init__(self, c, d):
                # [FUNCTION DOC] HELPER: Stores the two constructor values as attributes.
                super().__init__()
                self.c = c
                self.d = d

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Forward described by the class; returns nothing when the
                # float is not one.
                if self.d == 2:
                    x += 1
                if self.c == 1.0:
                    return x + y + torch.tensor([z])

        f = Bar(1.0, 2)
        f(2, 1.0, 2.0)
        f.d = 3
        f(3, 1.2, 2.0)
        state = torch._dynamo.pgo.render_code_state(torch._dynamo.pgo.get_code_state())
        whitelist = re.search(r'TORCH_COMPILE_DYNAMIC_SOURCES="(.*)"', state).group(1)
        self.assertTrue("L['x']" in whitelist)
        self.assertTrue("L['y']" in whitelist)
        self.assertTrue(
            "___as_tensor(L['y'])" not in whitelist
        )  # ephemeral FloatTensor source
        self.assertTrue("L['z']" not in whitelist)  # static float
        self.assertTrue("L['self'].c" not in whitelist)  # static float property
        self.assertTrue("L['self'].d" in whitelist)  # dynamic int property

    def test_pgo_dynamic_params(self):
        # [FUNCTION DOC] Counts compiled frames for a module whose linear layer is replaced by
        # new shapes. By default parameters are static, so each of the three runs recompile; with
        # static-shape forcing disabled the second run compiles two frames and a following run
        # only one, since parameters now participate in PGO.
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Compiled module holding an initially empty linear layer slot
            # which is applied to the input in forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the layer attribute to nothing.
                super().__init__()
                self.lin = None

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the layer applied to the input.
                return self.lin(x)

        f = Foo()

        def run():
            # [FUNCTION DOC] HELPER: Resets caches and counter, installs a 4 by 4 layer and calls
            # with two batch sizes, then installs an 8 by 8 layer and calls once.
            self.reset()
            cnts.clear()
            f.lin = torch.nn.Linear(4, 4)
            f(torch.randn(2, 4))
            f(torch.randn(4, 4))
            f.lin = torch.nn.Linear(8, 8)
            f(torch.randn(8, 8))

        # recompile each run
        run()
        self.assertEqual(cnts.frame_count, 3)

        # parameter static shapes are forced static, so we recompile once
        with torch._dynamo.config.patch(
            force_parameter_static_shapes=False,
            force_nn_module_property_static_shapes=False,
        ):
            run()
            self.assertEqual(cnts.frame_count, 2)

            # because flags were flipped, params were included in PGO
            run()
            self.assertEqual(cnts.frame_count, 1)

    def test_njt(self):
        # [FUNCTION DOC] Checks PGO handles nested jagged tensors without pickling problems: two
        # jagged inputs with different nested sizes compile once, and after a reset two more
        # jagged inputs again compile once.
        cnts = CompileCounter()

        # NB: PGO doesn't do anything here, the point is to catch pickle
        # problem with nested int

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input doubled.
            return x * 2

        x = torch.nested.nested_tensor_from_jagged(
            torch.randn(10, 3), torch.tensor([0, 3, 7, 10]), torch.tensor([1, 2, 3])
        )
        y = torch.nested.nested_tensor_from_jagged(
            torch.randn(13, 3), torch.tensor([0, 3, 7, 13]), torch.tensor([1, 2, 6])
        )

        f(x)
        f(y)
        self.assertEqual(cnts.frame_count, 1)

        self.reset()
        cnts.clear()

        a = torch.nested.nested_tensor_from_jagged(
            torch.randn(14, 3), torch.tensor([0, 3, 7, 14]), torch.tensor([1, 2, 7])
        )
        b = torch.nested.nested_tensor_from_jagged(
            torch.randn(15, 3), torch.tensor([0, 3, 7, 15]), torch.tensor([1, 2, 8])
        )

        f(a)
        f(b)
        self.assertEqual(cnts.frame_count, 1)

    def test_distinct_compile_id(self):
        # [FUNCTION DOC] Checks the job id separates PGO state: with job foo two shapes compile
        # twice; after reset, with job bar also twice; after clearing caches again, with job foo,
        # the stored state makes it compile once.
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input doubled.
            return x * 2

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        with torch.compiler.config.patch(job_id="bar"):
            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 2)

        torch._dynamo.reset()
        clear_caches()
        cnts.clear()

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 7))
            f(torch.randn(2, 8))
        self.assertEqual(cnts.frame_count, 1)

    # TODO: to test local need to ensure the local filesystem gets cleared out
    @torch._dynamo.config.patch(
        automatic_dynamic_remote_pgo=True, automatic_dynamic_local_pgo=False
    )
    def test_remote_basic(self):
        # [FUNCTION DOC] With remote PGO enabled and local PGO disabled under mock caches, checks
        # compile counts and the mock cache statistics (put, hit, miss counters) after several
        # runs, including a run under a different cache key tag which behaves like a fresh cache.
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input doubled.
            return x * 2

        with mock_cache.PatchCaches():
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 0, 1)
            )

            self.reset()
            cnts.clear()

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 1, 1)
            )

            self.reset()
            cnts.clear()

            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                f(torch.randn(2, 7))
                f(torch.randn(2, 8))
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(
                    mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(4, 1, 2)
                )

    # Test that if the same file appears in two different paths for two different compilations PGO still works.
    def test_different_file_paths_local_pgo(self):
        # [FUNCTION DOC] Writes the same module source into two different temporary directories,
        # loads and runs it for each path, and checks the first run compiles two frames, the
        # rendered code state mentions the file and a dynamic size, and the second run compiles
        # only one frame because PGO unifies the same file at different paths.
        # KNOWN ISSUE: the helper loads the module spec from the first path even when asked to
        #     run the second path, so the second run does not actually load the second file. The
        #     hash check uses a conditional expression whose Windows branch is always true.
        content = """
import torch
def run(cnt):
    @torch.compile(backend=cnt, fullgraph=True)
    def func(x):
        return x*10
    func(torch.rand(10))
    func(torch.rand(20))
    func(torch.rand(30))
"""
        temp_dir1 = tempfile.TemporaryDirectory()
        temp_dir2 = tempfile.TemporaryDirectory()

        # We need normalize_path_separator for Windows file path.
        path1 = normalize_path_separator(os.path.join(temp_dir1.name, "example.py"))
        path2 = normalize_path_separator(os.path.join(temp_dir2.name, "example.py"))
        cnts = CompileCounter()

        if path1 == path2:
            raise AssertionError("Expected path1 != path2")

        def write_load_and_run(path):
            # [FUNCTION DOC] HELPER: Writes the module source to the given path, imports a module
            # from the first path through importlib, and runs its run entry point with the
            # compile counter.
            with open(path, "w") as file:
                file.write(content)
            spec = importlib.util.spec_from_file_location("example", path1)
            if spec is None:
                raise AssertionError("Expected spec to not be None")
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise AssertionError("Expected spec.loader to not be None")
            spec.loader.exec_module(module)
            module.run(cnts)

        write_load_and_run(path1)
        self.assertEqual(cnts.frame_count, 2)
        state = torch._dynamo.pgo.render_code_state(torch._dynamo.pgo.get_code_state())

        # Windows can't create unification temp path:
        #   hash(a18a3259)C:/Users/Xuhan/AppData/Local/Temp/tmpx3hfkuqa/example.py
        # Skip hash check
        self.assertTrue("hash" if IS_WINDOWS else "hash(390fe689)" in state)
        self.assertTrue("/example.py:4:func:" in state)
        self.assertTrue(" L['x']: tensor size=[?] stride=[1]" in state)
        # We should compile this only once due to PGO.
        cnts.clear()
        write_load_and_run(path2)
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(
        automatic_dynamic_remote_pgo=True, automatic_dynamic_local_pgo=False
    )
    def test_sticky_pgo_read_write(self):
        # [FUNCTION DOC] With remote PGO and mock caches, checks the extra write and extra read
        # PGO keys: different job ids do not share state; writing with a sticky key then reading
        # it in another job gives dynamic x with a single compile; a later job writing another
        # sticky key sees three compiles; when default remote PGO exists the extra read key is
        # ignored and varying a different input recompiles.
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning both inputs scaled by two and
            # three.
            return x * 2, y * 3

        def t(x, y):
            # [FUNCTION DOC] HELPER: Returns a random tensor of the given two dimensions.
            return torch.randn(x, y)

        with mock_cache.PatchCaches():
            # we pretend to disable the default remote cache, by keying different job ids per run
            with torch.compiler.config.patch(job_id="a"):
                f(t(2, 2), t(2, 2))
                f(t(2, 4), t(2, 2))
                self.assertEqual(cnts.frame_count, 2)

            # first test we're not reading from local/default remote cache;
            # we should recompile when x wobbles
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(
                job_id="b", pgo_extra_write_key="sticky_0"
            ):
                f(t(2, 2), t(2, 2))
                f(t(2, 4), t(2, 2))
                self.assertEqual(cnts.frame_count, 2)

            # now with the extra sticky_0 key, we start with dynamic x;
            # no recompiles
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(job_id="c", pgo_extra_read_key="sticky_0"):
                f(t(2, 2), t(2, 2))
                f(t(2, 4), t(2, 2))
                self.assertEqual(cnts.frame_count, 1)

            # last test: wobble y and write to sticky_1 key
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(
                job_id="d", pgo_extra_write_key="sticky_1"
            ):
                f(t(2, 2), t(2, 2))
                f(t(2, 2), t(2, 4))
                f(t(2, 2), t(4, 4))
                self.assertEqual(cnts.frame_count, 3)

            # start using default remote PGO, create run that wobbles y
            self.reset()
            cnts.clear()
            f(t(2, 2), t(2, 2))
            f(t(2, 4), t(2, 2))
            f(t(4, 2), t(2, 2))

            # with both default remote present, we ignore extra remote.
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(pgo_extra_read_key="sticky_1"):
                f(t(2, 2), t(2, 2))
                f(t(6, 8), t(2, 2))
                self.assertEqual(cnts.frame_count, 1)
                f(t(2, 2), t(2, 4))
                self.assertEqual(cnts.frame_count, 2)

    def test_factory_created_functions_separate_pgo_state(self):
        # [FUNCTION DOC] Compiles two functions made by a factory wrapper with a counting
        # aot_eager backend and shows they keep separate PGO state: the second function keeps
        # static shapes and its graph text matches the expected one. Calling both again with the
        # same shapes triggers no recompilation.
        cc_backend = CompileCounterWithBackend(backend="aot_eager")

        def wrap_compile(func):
            # [FUNCTION DOC] HELPER: Factory that returns a fully-graph compiled wrapper calling
            # the given function.
            @torch.compile(backend=cc_backend, fullgraph=True)
            def _func(x):
                # [FUNCTION DOC] HELPER: Compiled wrapper that calls the captured function on its
                # input.
                return func(x)

            return _func

        def impl_a(x):
            # [FUNCTION DOC] HELPER: Returns the input flattened with a view.
            return x.view(-1)

        def impl_b(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the input.
            return x.sum()

        compiled_a = wrap_compile(impl_a)
        compiled_b = wrap_compile(impl_b)

        # Call with different shapes - they should NOT share PGO state
        compiled_a(torch.randn(2, 3))  # shape (2, 3)
        compiled_b(torch.randn(5, 7))  # shape (5, 7)

        actual = normalize_gm(cc_backend.graphs[1].print_readable(print_output=False))

        # If PGO state is properly separated, compiled_b should NOT have
        # been triggered to use dynamic shapes based on compiled_a's shapes
        # This should result in 2 separate static compilations
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5, 7]"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum();  l_x_ = None
        return (sum_1,)
""",
        )

        # Verify no dynamic shapes were introduced incorrectly
        # by checking that calling with same shapes doesn't recompile
        cc_backend.clear()
        compiled_a(torch.randn(2, 3))
        compiled_b(torch.randn(5, 7))
        self.assertEqual(cc_backend.frame_count, 0)  # No recompilation needed


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
