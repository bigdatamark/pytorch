# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch._inductor.compiler_bisector.CompilerBisector, run through
# dynamo's test directory. Each test supplies a pass/fail function to do_bisect, injects a bug
# (bad decomposition, custom graph pass, random fallback, crossref mismatch, precision casts,
# relu lowering bug, pre-grad graph split), and asserts the reported backend, subsystem, bisect
# number and debug info. Also covers cudagraph skipping via environment variables and the
# command-line run mode. Defines dtype aliases and an aten alias at import; the test class is
# skipped without a GPU and Triton.

import unittest
from contextlib import contextmanager
from importlib import import_module

import torch
import torch._prims_common as utils
from torch._dynamo.utils import preserve_rng_state
from torch._inductor import config
from torch._inductor.compiler_bisector import CompilerBisector
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.test_case import TestCase
from torch.library import _scoped_library, Library
from torch.testing._internal.common_utils import requires_cuda
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


aten = torch.ops.aten


f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


@unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
class TestCompilerBisector(TestCase):
    # [CLASS DOC] Test case for CompilerBisector.do_bisect, skipped unless a GPU with Triton is
    # available. Uses a private operator namespace (test_ns) and an optional Library stored in
    # self.lib that tearDown destroys.
    test_ns = "_test_bisector"

    def tearDown(self):
        # [FUNCTION DOC] Delete the test operator namespace from torch.ops if present and
        # destroy self.lib if one was created.
        # KNOWN ISSUE: does not call the base class tearDown, so base cleanup (e.g. dynamo
        # reset) is skipped.
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        if hasattr(self, "lib"):
            self.lib._destroy()

    def get_op(self, name):
        # [FUNCTION DOC] HELPER: Return the default overload of the named operator in the test
        # namespace.
        return getattr(getattr(torch.ops, self.test_ns), name).default

    def get_lib(self):
        # [FUNCTION DOC] HELPER: Create a FRAGMENT Library in the test namespace, remember it as
        # self.lib and return it.
        lib = Library(self.test_ns, "FRAGMENT")  # noqa: SCOPED_LIBRARY
        self.lib = lib
        return lib

    def test_bad_decomp(self):
        # [FUNCTION DOC] Bisect a failure caused by a wrong decomposition of exponential_
        # (returns NaNs): patch the inductor decomposition table, then assert the result points
        # at backend aot_eager_decomp_partition, subsystem decomposition, bisect number 1 and
        # debug info mentioning aten.exponential.
        import_module("torch._inductor.compile_fx")

        def bad_exp_decomp(self, rate=1, generator=None):
            # [FUNCTION DOC] HELPER: Bad decomposition of exponential_ that validates dtype and
            # rate with checks, requires generator to be None (raises AssertionError otherwise)
            # and returns random values multiplied by NaN.
            if generator is not None:
                raise AssertionError("Expected generator to be None")
            torch._check(
                not utils.is_complex_dtype(self.dtype)
                and not utils.is_integer_dtype(self.dtype)
                and not utils.is_boolean_dtype(self.dtype),
                lambda: f"Exponential distribution is a continuous probability distribution. \
                dtype must be a floating point but you specified {self.dtype}",
            )
            torch._check(
                rate > 0.0,
                lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
            )
            return torch.rand_like(self) * float("nan")

        @contextmanager
        def patch_exp_decomp():
            # [FUNCTION DOC] HELPER: Context manager that temporarily replaces
            # compile_fx.select_decomp_table with one whose table maps aten.exponential to the
            # bad decomposition, restoring the original in a finally block.
            from torch._inductor.compile_fx import select_decomp_table as old_decomp

            def get_decomp():
                # [FUNCTION DOC] HELPER: Return a copy of the original decomposition table with
                # the exponential entry overridden by the bad decomposition.
                out = old_decomp()
                out = out.copy()
                out[aten.exponential.default] = bad_exp_decomp
                return out

            torch._inductor.compile_fx.select_decomp_table = get_decomp
            try:
                yield

            finally:
                torch._inductor.compile_fx.select_decomp_table = old_decomp

        def vq(x):
            # [FUNCTION DOC] HELPER: Add 3, apply exponential_ in place, scale by 10.5.
            return (x + 3).exponential_() * 10.5

        def test_fn():
            # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, under the patched
            # decomposition run eager and compiled versions with the same RNG state and return
            # True if the compiled output has no NaN.
            torch._dynamo.reset()
            with patch_exp_decomp():
                vq_compiled = torch.compile(vq)  # noqa: UNSPECIFIED_BACKEND
                x = torch.randn(4, 400, 256, device=GPU_TYPE)
                with torch._dynamo.utils.preserve_rng_state():
                    vq(x)
                out_compiled = vq_compiled(x)

            return not out_compiled.isnan().any()

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "aot_eager_decomp_partition")
        self.assertEqual(out.subsystem, "decomposition")
        self.assertEqual(out.bisect_number, 1)
        self.assertTrue("aten.exponential" in out.debug_info)

    def test_pre_grad(self):
        # [FUNCTION DOC] Bisect a wrong pre-grad custom pass (rewrites the add constant to 2):
        # install it with config, assert backend inductor, subsystem pre_grad_passes, bisect
        # number 3 and debug info naming pre_grad_custom_pass.
        import operator

        from torch._inductor import config

        # similar setup to test_joint_graph (see below)
        class CustomPrePass(CustomGraphPass):
            # [CLASS DOC] HELPER: Custom graph pass that changes the single operator.add node's
            # second argument to 2.
            def __call__(self, graph: torch.fx.Graph):
                # [FUNCTION DOC] HELPER: Find operator.add call nodes in the graph, raise
                # AssertionError unless there is exactly one, and set its second argument to 2.
                nodes = graph.find_nodes(op="call_function", target=operator.add)
                if len(nodes) != 1:
                    raise AssertionError(f"Expected 1 node, got {len(nodes)}")
                args = list(nodes[0].args)
                args[1] = 2
                nodes[0].args = tuple(args)

            def uuid(self):
                # [FUNCTION DOC] HELPER: Return a hash of a fixed string identifying this pass.
                return hash("TestCompilerBisector.test_pre_grad.pass_class")

        def foo(x):
            # [FUNCTION DOC] HELPER: Return x plus 1.
            return x + 1

        def test_fn():
            # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, compare eager and
            # compiled foo on a random CPU tensor and return whether they are close.
            torch._dynamo.reset()

            inp = torch.rand([10])

            out = foo(inp)
            out_c = torch.compile(foo)(inp)  # noqa: UNSPECIFIED_BACKEND

            return torch.allclose(out, out_c)

        with config.patch(pre_grad_custom_pass=CustomPrePass()):
            out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "pre_grad_passes")
        self.assertEqual(out.bisect_number, 3)
        self.assertTrue("pre_grad_custom_pass" in out.debug_info)

    def test_joint_graph(self):
        # [FUNCTION DOC] Bisect a wrong joint-graph custom post pass (rewrites aten.add.Tensor
        # constant to 2): assert backend inductor, subsystem joint_graph_passes, bisect number 4
        # and debug info naming joint_custom_post_pass.
        from torch._inductor import config

        class CustomPostPass(CustomGraphPass):
            # [CLASS DOC] HELPER: Custom graph pass that changes the single aten.add.Tensor
            # node's second argument to 2.
            def __call__(self, graph: torch.fx.Graph):
                # [FUNCTION DOC] HELPER: Find aten.add.Tensor call nodes, raise AssertionError
                # unless exactly one, and set its second argument to 2.
                nodes = graph.find_nodes(
                    op="call_function", target=torch.ops.aten.add.Tensor
                )
                if len(nodes) != 1:
                    raise AssertionError(f"Expected 1 node, got {len(nodes)}")
                args = list(nodes[0].args)
                args[1] = 2
                nodes[0].args = tuple(args)

            def uuid(self):
                # [FUNCTION DOC] HELPER: Return a hash of a fixed string identifying this pass.
                return hash("TestCompilerBisector.test_joint_graph.pass_class")

        def foo(x):
            # [FUNCTION DOC] HELPER: Return x plus 1.
            return x + 1

        def test_fn():
            # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, compare eager and
            # compiled foo on a random GPU tensor and return whether they are close.
            torch._dynamo.reset()

            inp = torch.rand([10], device=GPU_TYPE)

            out = foo(inp)
            out_c = torch.compile(foo)(inp)  # noqa: UNSPECIFIED_BACKEND

            return torch.allclose(out, out_c)

        with config.patch(joint_custom_post_pass=CustomPostPass()):
            out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "joint_graph_passes")
        self.assertEqual(out.bisect_number, 4)
        self.assertTrue("joint_custom_post_pass" in out.debug_info)

    def test_rng(self):
        # [FUNCTION DOC] Bisect a mismatch between eager and compiled random numbers (RNG state
        # preserved for each run): assert backend inductor, subsystem inductor_fallback_random
        # and matching debug info.
        def foo():
            # [FUNCTION DOC] HELPER: Return a random GPU vector plus 1.
            return torch.rand([10], device=GPU_TYPE) + 1

        def test_fn():
            # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, run eager and compiled
            # foo each under preserve_rng_state and return whether results are close.
            torch._dynamo.reset()

            with preserve_rng_state():
                out = foo()
            with preserve_rng_state():
                out_c = torch.compile(foo)()  # noqa: UNSPECIFIED_BACKEND

            return torch.allclose(out, out_c)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "inductor_fallback_random")
        self.assertTrue("inductor_fallback_random" in out.debug_info)

    def test_crossref(self):
        # [FUNCTION DOC] Define a custom operator with an autograd function, CPU and Meta
        # implementations, in a scoped library; bisect a failing compiled-vs-eager comparison on
        # a tensor requiring grad and assert the backend is aot_eager_decomp_partition_crossref.
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            lib.define("foo(Tensor x) -> Tensor")
            op = self.get_op("foo")

            class Foo(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Call the operator with autograd and the
                    # ADInplaceOrView dispatch key excluded and return the result.
                    # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                    with torch._C._AutoDispatchBelowAutograd():
                        with torch._C._ExcludeDispatchKeyGuard(
                            torch._C.DispatchKeySet(
                                torch._C.DispatchKey.ADInplaceOrView
                            )
                        ):
                            return op(x)

                @staticmethod
                def backward(ctx, gx):
                    # [FUNCTION DOC] HELPER: Return the incoming gradient unchanged.
                    return gx

            def foo_impl(x):
                # [FUNCTION DOC] HELPER: CPU implementation: clone of a view of x.
                return x.view_as(x).clone()

            def foo_meta(x):
                # [FUNCTION DOC] HELPER: Meta implementation: a view of x.
                return x.view_as(x)

            lib.impl("foo", Foo.apply, "Autograd")
            lib.impl("foo", foo_impl, "CPU")
            lib.impl("foo", foo_meta, "Meta")

            x = torch.tensor(3.14159 / 3, requires_grad=True)

            def test_fn():
                # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, return True if the
                # compiled operator output is close to eager, False if any exception occurs.
                torch._dynamo.reset()

                try:
                    torch.testing.assert_close(torch.compile(op)(x), op(x))  # noqa: UNSPECIFIED_BACKEND
                except Exception:
                    return False
                return True

            out = CompilerBisector.do_bisect(test_fn)
            self.assertEqual(out.backend, "aot_eager_decomp_partition_crossref")

    def test_emulate_precision_casts(self):
        # [FUNCTION DOC] Bisect a precision difference between eager and compiled scale
        # computation on bfloat16 data; assert backend inductor and subsystem
        # inductor_emulate_precision_casts.
        def test_fn():
            # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, seed the RNG, compute
            # the scale eagerly and compiled and return whether they are exactly equal.
            torch._dynamo.reset()

            def calculate_scale(inp):
                # [FUNCTION DOC] HELPER: Compute 448 divided by the clamped absolute max of inp,
                # cast to float32.
                amax = torch.abs(torch.max(inp))
                scale = 448.0 / torch.clamp(amax, min=1e-12)
                scale = scale.to(torch.float32)
                return scale

            dtype = torch.bfloat16
            torch.manual_seed(0)
            inp = torch.randn(16, 16, 768, dtype=dtype, device=GPU_TYPE)
            eager_scale = calculate_scale(inp)
            compile_scale = torch.compile(calculate_scale)(inp)  # noqa: UNSPECIFIED_BACKEND

            return torch.equal(eager_scale, compile_scale)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "inductor_emulate_precision_casts")

    def test_bad_lowering(self):
        # [FUNCTION DOC] Bisect an injected relu accuracy bug in the Triton backend; assert
        # backend inductor, subsystem lowerings, bisect number 2 and debug info mentioning relu.
        def test_fn():
            # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, with the relu bug
            # injected compare compiled and eager my_func and return whether close.
            torch._dynamo.reset()
            with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):

                def my_func(x):
                    # [FUNCTION DOC] HELPER: Return relu of the negated input minus 0.01.
                    return ((x * -1) - 0.01).relu()

                inp = torch.rand([100], device=GPU_TYPE)

                return torch.allclose(torch.compile(my_func)(inp), my_func(inp))  # noqa: UNSPECIFIED_BACKEND

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "lowerings")
        self.assertEqual(out.bisect_number, 2)
        self.assertTrue("relu" in out.debug_info)

    def test_eager_backend(self):
        # [FUNCTION DOC] A predicate that always fails should blame the eager backend with no
        # subsystem.
        # should indicate problem with first backend
        def test_fn():
            # [FUNCTION DOC] HELPER: Predicate that always returns False.
            return False

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "eager")
        self.assertEqual(out.subsystem, None)

    @config.patch(
        {
            "test_configs.bisect_pre_grad_graph": True,
            "test_configs.bisect_keep_custom_backend_for_inductor": True,
        }
    )
    def test_bisect_pre_grad_graph(self):
        # [FUNCTION DOC] With the pre-grad graph bisect test config enabled and a custom backend
        # that splits the graph and compiles the pieces with standalone_compile, bisect an
        # injected relu bug; assert backend inductor, subsystem pre_grad_graph and bisect number
        # 1.
        def f(x):
            # [FUNCTION DOC] HELPER: Add 1 five times and apply relu.
            for _ in range(5):
                x = x + 1
            return x.relu()

        class MyBackend:
            # [CLASS DOC] HELPER: Backend that splits the graph module after the first three
            # nodes and compiles each submodule with standalone_compile.
            def __call__(self, gm, example_inputs):
                # [FUNCTION DOC] HELPER: Split the module using a node-order counter, replace
                # each submodule's forward with a standalone-compiled version, and return the
                # split module.
                node_idx = 0

                def node_to_graph_id(node):
                    # [FUNCTION DOC] HELPER: Return graph id 0 for the first three nodes seen
                    # and 1 afterwards, incrementing the shared counter.
                    nonlocal node_idx
                    out = 0 if node_idx < 3 else 1
                    node_idx += 1
                    return out

                split_gm = torch.fx.passes.split_module.split_module(
                    gm, None, node_to_graph_id, keep_original_order=True
                )

                for name, submod in split_gm.named_modules():
                    if "submod_" in name:
                        # the test case is simple enough that using
                        # the original example_inputs works for sub
                        # module
                        submod.forward = torch._inductor.standalone_compile(
                            submod,
                            example_inputs,
                            dynamic_shapes="from_example_inputs",
                            options={},
                        )

                return split_gm

        def test_fn():
            # [FUNCTION DOC] HELPER: Bisection predicate: reset dynamo, with the relu bug
            # injected compare compiled (custom backend) and eager results.
            torch._dynamo.reset()

            x = torch.randn(1024, device=GPU_TYPE)
            with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):
                opt_f = torch.compile(f, backend=MyBackend())
                return torch.allclose(opt_f(x), f(x))

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "pre_grad_graph")
        self.assertEqual(out.bisect_number, 1)

    # XPU doesn't support cudagrah
    @requires_cuda
    def test_cudagraph_bisect_max(self):
        # [FUNCTION DOC] Requires CUDA. With bisect environment variables selecting the inductor
        # cudagraphs subsystem and max 0, compile two functions in reduce-overhead mode and
        # assert the cudagraph_skips counter is positive; always disables bisection and clears
        # the env cache afterwards.
        """Test that cudagraph bisector can limit number of cudagraphed graphs."""
        import os
        from unittest.mock import patch

        from torch._dynamo.utils import counters
        from torch._inductor.compiler_bisector import get_env_val

        def foo(x):
            # [FUNCTION DOC] HELPER: Return x plus 1.
            return x + 1

        def bar(x):
            # [FUNCTION DOC] HELPER: Return x times 2.
            return x * 2

        env = {
            "TORCH_BISECT_BACKEND": "inductor",
            "TORCH_BISECT_SUBSYSTEM": "cudagraphs",
            "TORCH_BISECT_MAX": "0",
        }

        with patch.dict(os.environ, env):
            get_env_val.cache_clear()
            CompilerBisector.reset_counters()
            torch._dynamo.reset()
            counters.clear()
            CompilerBisector.bisection_enabled = True
            try:
                foo_c = torch.compile(foo, mode="reduce-overhead")  # noqa: UNSPECIFIED_BACKEND
                bar_c = torch.compile(bar, mode="reduce-overhead")  # noqa: UNSPECIFIED_BACKEND
                x = torch.randn(10, device=GPU_TYPE)
                foo_c(x)
                bar_c(x)

                # With max=0, all graphs should be skipped
                self.assertGreater(counters["inductor"]["cudagraph_skips"], 0)
            finally:
                CompilerBisector.bisection_enabled = False
                get_env_val.cache_clear()

    def test_bisect_run_debuginfo(self):
        # [FUNCTION DOC] Run the compiler_bisector command line in a subprocess (timeout 300
        # seconds) on the helper script with only the aot_eager_decomp_partition backend
        # searched, and assert stdout contains the expected debug info for aten.exponential.
        import os
        import subprocess
        from pathlib import Path
        from unittest.mock import patch

        test_file = (
            Path(__file__).resolve().parent / "_test_compiler_bisector_run_helper.py"
        )
        # Minimize test runtime by searching only the subsystem that's broken.
        with patch.dict(
            os.environ, {"TORCH_BISECT_BACKEND": "aot_eager_decomp_partition"}
        ):
            output = subprocess.run(
                [
                    "python",
                    "-m",
                    "torch._inductor.compiler_bisector",
                    "run",
                    "python",
                    str(test_file),
                ],
                stdout=subprocess.PIPE,
                check=True,
                text=True,
                timeout=300,
            )
        expected_result = (
            "Debug info: <OpOverload(op='aten.exponential', overload='default')>"
        )
        self.assertIn(expected_result, output.stdout)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
