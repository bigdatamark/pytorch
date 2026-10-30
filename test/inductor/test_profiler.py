# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for torch.profiler integration: checks that compiled triton
# kernels show up in profiler traces with expected names, input shapes and launch metadata, and
# that profiler record_function ops are removed from post-grad graphs. Most tests need a GPU
# with triton; the entry point only runs the suite when a GPU and triton are available. Defines
# HAS_TRITON at import time from the triton availability check.
import json
import os
import tempfile
import unittest
from collections.abc import Callable

import torch
import torch._inductor.test_case
import torch._inductor.utils
from torch import _dynamo as torchdynamo
from torch._inductor import config
from torch.profiler import ProfilerActivity, record_function
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    IS_BIG_GPU,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.torch_version import TorchVersion
from torch.utils._triton import has_triton


HAS_TRITON = has_triton()


class DynamoProfilerTests(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Test case holding all profiler/inductor integration tests. Stateless; each
    # test compiles a small function, runs it (often under a profiler) and inspects events or
    # logs.
    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_triton_launch(self):
        # [FUNCTION DOC] Compiles a pointwise function, profiles one call, exports a chrome
        # trace to a temporary file, and asserts the trace has a launch event: a CUDA or HIP
        # kernel launch name, or the named inductor triton kernel. Skipped without triton.
        # Verify that we get some sort of CPU-side indication of triton kernel launches
        # in the profile traces. Currently, those appear as `cuLaunchKernel`. If this
        # detail changes, the test can be updated or removed.
        @torch.compile
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Pointwise function adding two tensors then applying sin and
            # cos; compiled and run by the enclosing test.
            return (x + y).sin().cos()

        x, y = (torch.rand((4, 4), device=GPU_TYPE) for _ in range(2))

        with torch.profiler.profile() as prof:
            fn(x, y)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace_json = json.load(f)

        self.assertTrue("traceEvents" in trace_json)
        events = trace_json["traceEvents"]

        valid_names = {
            "hipModuleLaunchKernel",
            "cuLaunchKernel",
            "triton_poi_fused_add_cos_sin_0",
        }
        self.assertTrue(any((event.get("name") in valid_names) for event in events))

    def _test_profiling_kernel_names(
        self, fn, args, kernel_name_str: str, check_fn: Callable | None = None
    ):
        # [FUNCTION DOC] HELPER: Compiles the given function, runs it twice on the given
        # arguments to warm up, then calls the optional check callback (it may raise SkipTest).
        # It then profiles one CPU-activity run with shapes recorded and asserts some event name
        # contains both the kernel substring and the word triton. Returns the profiler's event
        # list for further checks.
        """
        We expect a record_function event to be added on the CPU side, surrounding
        the launch of each triton kernel.
        """
        fn_opt = torch.compile(fn)

        for _ in range(2):
            fn_opt(*args)

        if check_fn is not None:
            check_fn()

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            fn_opt(*args)

        # The name of the kernel is expected to match the name of the kernel in debug
        # files etc. The name could change in the future, but it seems reasonable that
        # the name should always contain "triton" and "kernel_name_str" - e.g. if the
        # kernel contains a sin op, it should probably contain "str" in the name.
        # If this changes in the future, feel free to change the assertion here.
        # Debugging tips: you can add prof.export_chrome_trace("test.json") inline in
        # this test, and then view test.json in chrome://tracing to see the trace.
        self.assertTrue(
            any(
                (
                    hasattr(event, "name")
                    and kernel_name_str in event.name
                    and "triton" in event.name
                )
                for event in prof.events()
            )
        )
        return prof.events()

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names_pointwise(self):
        # [FUNCTION DOC] Profiles a compiled add/sin/cos function and finds the event for the
        # fused pointwise triton kernel. Asserts its first four recorded input shapes are three
        # 4x4 tensors followed by an empty (scalar) entry, and that such an event exists. Needs
        # triton.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Eager function computing sin and cos of the sum of its
            # inputs.
            return (x + y).sin().cos()

        args = [torch.rand((4, 4), device=GPU_TYPE) for _ in range(2)]

        events = self._test_profiling_kernel_names(fn, args, "sin")
        event_found = False
        for event in events:
            if event.name == "triton_poi_fused_add_cos_sin_0":
                event_found = True
                # Note: depending on the triton version, we might get 4 or 5 args
                # (including / not including the constexpr args). The last two are
                # both empty args, so we just truncate the event.input_shapes to the
                # first 4.
                self.assertEqual(event.input_shapes[:4], [[4, 4], [4, 4], [4, 4], []])
        self.assertTrue(event_found)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_inductor_profiling_kernel_names_template(self):
        # [FUNCTION DOC] With max-autotune on and only the triton GEMM backend, profiles a
        # compiled matrix multiply, finds the template kernel event, and asserts its first three
        # input shapes are all 4x4. May skip if autotuning never ran. Skipped when the GPU is
        # not a big GPU.
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Eager matrix multiply of its two inputs.
                return x @ y

            args = [torch.rand((4, 4), device=GPU_TYPE) for _ in range(2)]

            def check_fn():
                # [FUNCTION DOC] HELPER: Callback passed to the profiling helper. Raises
                # SkipTest when the inductor counter for select_algorithm autotune is zero,
                # since then no template kernel profile is expected (the machine may lack enough
                # SMs).
                # test_profiling_kernel_names will check this before asserting mm is in the trace.
                # reason: sometimes testing runs on machines with not enough SMs, and autotuning is skipped.
                if (
                    torch._dynamo.utils.counters["inductor"][
                        "select_algorithm_autotune"
                    ]
                    == 0
                ):
                    raise unittest.SkipTest(
                        "select_algorithm didn't run, we probably won't get profiling data. GPU might not have enough SMs."
                    )

            events = self._test_profiling_kernel_names(fn, args, "mm", check_fn)

            event_found = False
            for event in events:
                if event.name == "triton_tem_fused_mm_0":
                    event_found = True
                    self.assertEqual(event.input_shapes[:3], [[4, 4], [4, 4], [4, 4]])
            self.assertTrue(event_found)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names_foreach(self):
        # [FUNCTION DOC] With max-autotune enabled, profiles a compiled foreach add over lists
        # of three 4x4 tensors, finds the fused foreach triton kernel event and asserts its
        # input shapes are nine 4x4 entries. Needs triton.
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Eager foreach add of two tensor lists.
                return torch._foreach_add(x, y)

            x = [torch.rand((4, 4), device=GPU_TYPE) for _ in range(3)]
            y = [torch.rand((4, 4), device=GPU_TYPE) for _ in range(3)]

            args = (x, y)

            events = self._test_profiling_kernel_names(fn, args, "_for_")
            event_found = False
            for event in events:
                if event.name == "triton_for_fused_0":
                    event_found = True
                    self.assertTrue(
                        event.input_shapes
                        == [
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                        ]
                    )
            self.assertTrue(event_found)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    @config.patch(
        "compile_threads", 1
    )  # This test monkey patches global variables, which workers don't see
    def test_inductor_profiling_triton_hooks(self):
        # [FUNCTION DOC] Installs launch-enter and launch-exit hooks on triton (via the knobs
        # runtime when available, else on the CompiledKernel class), runs a compiled foreach
        # add, and asserts both hooks were called. Side effect: leaves the hooks installed
        # globally after the test; compile threads are forced to 1 so hooks are visible in-
        # process.
        from triton.compiler import CompiledKernel  # @manual

        from torch._inductor.runtime.triton_compat import knobs

        hooks_called = {"enter": False, "exit": False}

        def launch_enter_hook(lazy_dict):
            # [FUNCTION DOC] HELPER: Records in the shared mapping that the launch-enter hook
            # ran.
            hooks_called["enter"] = True

        def launch_exit_hook(lazy_dict):
            # [FUNCTION DOC] HELPER: Records in the shared mapping that the launch-exit hook
            # ran.
            hooks_called["exit"] = True

        if knobs:
            knobs.runtime.launch_enter_hook = launch_enter_hook
            knobs.runtime.launch_exit_hook = launch_exit_hook
        else:
            CompiledKernel.launch_enter_hook = launch_enter_hook
            CompiledKernel.launch_exit_hook = launch_exit_hook

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Eager foreach add of two tensor lists.
            return torch._foreach_add(x, y)

        x = [torch.rand((4, 4), device=GPU_TYPE) for _ in range(3)]
        y = [torch.rand((4, 4), device=GPU_TYPE) for _ in range(3)]

        args = (x, y)
        fn_opt = torch.compile(fn)
        fn_opt(*args)

        self.assertTrue(hooks_called["enter"])
        self.assertTrue(hooks_called["exit"])

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_pt2_triton_attributes(self):
        # [FUNCTION DOC] Runs a compiled linear+add+cos function under a scheduled profiler for
        # ten steps, exports a chrome trace to a temporary file and collects events with
        # kernel_backend metadata. Asserts exactly two such events and that each carries the
        # triton backend, a stream, an existing kernel file, a hash matching the file source,
        # and kernel kwargs starting with the XBLOCK setting. Prints the trace path and events.
        # Needs triton.
        from torch._inductor.codecache import code_hash

        device = GPU_TYPE
        debug = False  # set to True to get output file

        @torchdynamo.optimize("inductor")
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Linear layer followed by a bias add and cos, compiled with
            # the inductor backend via dynamo.
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        a, b, c = (torch.randn(4, 4, requires_grad=True).to(device) for _ in range(3))

        inputs = [a, b, c]
        with config.patch(compile_threads=1):
            fn(*inputs)

        fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=not debug)  # noqa: SIM115
        fp.close()

        with torch.profiler.profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
        ) as prof:
            for _ in range(10):
                fn(*inputs)
                prof.step()

        prof.export_chrome_trace(fp.name)
        print(f"Trace written to {fp.name}, set debug=True to retain file.")

        triton_events = []
        with open(fp.name) as f:
            trace_json = json.load(f)
            triton_events = [
                event
                for event in trace_json["traceEvents"]
                if "kernel_backend" in event.get("args", {})
            ]

        print(triton_events)
        self.assertEqual(len(triton_events), 2)

        def get_hash(kernel_file: str) -> str:
            # [FUNCTION DOC] HELPER: Reads the given kernel source file and returns the inductor
            # code hash of its stripped contents.
            with open(kernel_file) as f:
                kernel_src = f.read()
            return code_hash(kernel_src.strip())

        def check_triton_event(e) -> None:
            # [FUNCTION DOC] HELPER: Asserts that one trace event has non-empty args with triton
            # backend, stream, kernel file existing on disk, kernel hash equal to the file's
            # hash, and kernel kwargs starting with XBLOCK. Failure messages include the event.
            args = e.get("args", {})
            self.assertNotEqual(args, {}, msg=lambda msg: f"{msg}\nevent = {e}")

            self.assertEqual(
                args["kernel_backend"], "triton", msg=lambda msg: f"{msg}\nevent = {e}"
            )

            self.assertTrue("stream" in args, msg=lambda msg: f"{msg}\nevent = {e}")
            self.assertTrue(
                "kernel_file" in args, msg=lambda msg: f"{msg}\nevent = {e}"
            )
            kernel_file = args["kernel_file"]
            self.assertTrue(
                os.path.isfile(kernel_file), msg=lambda msg: f"{msg}\nevent = {e}"
            )

            self.assertTrue(
                "kernel_hash" in args, msg=lambda msg: f"{msg}\nevent = {e}"
            )
            self.assertEqual(
                args["kernel_hash"],
                get_hash(kernel_file),
                msg=lambda msg: f"{msg}\nevent = {e}",
            )

            self.assertTrue(
                "kernel_kwargs" in args, msg=lambda msg: f"{msg}\nevent = {e}"
            )
            self.assertTrue(
                args["kernel_kwargs"].startswith("XBLOCK="),
                msg=lambda msg: f"{msg}\nevent = {e}",
            )

        for e in triton_events:
            check_triton_event(e)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_cupti_lazy_reinit(self):
        # [FUNCTION DOC] Runs a function compiled in reduce-overhead mode under a profiler, then
        # checks the DISABLE_CUPTI_LAZY_REINIT environment variable: unset or zero when the CUDA
        # version is at least 12.6, one otherwise. Needs triton.
        x, y = (torch.randn(4, 4, device=GPU_TYPE) for _ in range(2))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Eager add followed by sin.
            return (x + y).sin()

        fn_c = torch.compile(fn, mode="reduce-overhead")

        with torch.profiler.profile():
            fn_c(x, y)

        if TorchVersion(torch.version.cuda) >= "12.6":
            self.assertEqual("0", os.environ.get("DISABLE_CUPTI_LAZY_REINIT", "0"))
        else:
            self.assertEqual("1", os.environ.get("DISABLE_CUPTI_LAZY_REINIT", "0"))

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_inductor_remove_profiler_ops(self):
        # [FUNCTION DOC] Compiles (full graph) a function wrapping ops in profiler
        # record_function regions, with dynamo capture of those regions enabled, and captures
        # the post-grad graph log. Asserts the printed graph contains only sin, cos and add, so
        # profiler ops were removed.
        """
        Test that inductor post_grad graph doesn't have profiler ops even when
        dynamo produce those ops.
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies sin, cos and add-two, each in a profiler region.
            with record_function("my_net1"):
                a = x.sin()
            with record_function("my_cos"):
                b = a.cos()
            with record_function("my_net2"):
                c = b + 2
            return c

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        with ctx():
            torch.compile(fn, fullgraph=True)(torch.randn(10))
        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
        sin: "f32[10][1]cpu" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
        cos: "f32[10][1]cpu" = torch.ops.aten.cos.default(sin);  sin = None
        add: "f32[10][1]cpu" = torch.ops.aten.add.Tensor(cos, 2);  cos = None
        return (add,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU_AND_TRITON:
        run_tests()
