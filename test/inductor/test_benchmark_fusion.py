# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor benchmark fusion: when benchmark_kernel and benchmark_fusion are
# enabled the scheduler benchmarks candidate fused kernels. Defines a base test case that patches
# those configs for the whole class, a device-agnostic template class of tests that is copied (with
# a device suffix) into GPU and CPU test classes, a benchmarking test on a non-zero device, and
# multi-template (max-autotune epilogue fusion) tests. Import-time side effects: appends the test
# directory to the import path and registers the copied tests when the GPU with Triton or the CPU
# (non-MPS) is available. Run as a script it runs tests if CPU or GPU with Triton is available.
import math
import os
import sys

import torch
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.test_operators import realize
from torch._inductor.utils import fresh_cache, is_big_gpu, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import slowTest
from torch.testing._internal.inductor_utils import (
    get_func_call,
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU_AND_TRITON,
    IS_BIG_GPU,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

import contextlib
import unittest

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
    skip_if_cpp_wrapper,
)
from torch._inductor import config
from torch._inductor.scheduler import Scheduler


class TestCase(InductorTestCase):
    # [CLASS DOC] HELPER: Inductor test case base. Enters a config patch (kernel benchmarking and
    # fusion benchmarking on) at class setup through an exit stack stored on the class, and closes
    # it at class teardown, so every test in subclasses runs with those options.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Class setup; calls the parent setup, then creates an exit stack on
        # the class and enters a config patch enabling benchmark_kernel and benchmark_fusion.
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "benchmark_kernel": True,
                    "benchmark_fusion": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] HELPER: Class teardown; closes the exit stack (undoing the config patch)
        # and then calls the parent teardown.
        cls._stack.close()
        super().tearDownClass()


class BenchmarkFusionTestTemplate:
    # [CLASS DOC] Template of device-generic benchmark fusion tests. It is not a test case itself;
    # it relies on the host class providing a common checker and a device attribute, and its test
    # methods are copied into the concrete GPU and CPU classes by copy_tests.
    def test_softmax(self):
        # [FUNCTION DOC] Compares a compiled softmax over the last dim of a 2 by 8192 random tensor
        # against eager using the host class checker.
        def f(x):
            # [FUNCTION DOC] HELPER: Function under test; softmax over the last dimension.
            return torch.nn.functional.softmax(x, dim=-1)

        self.common(f, (torch.rand(2, 8192),))

    @slowTest
    def test_resnet18(self):
        # [FUNCTION DOC] Slow test: skips if torchvision is missing; otherwise runs a resnet18
        # model in eval mode on a random batch of 16 images and compares against eager with loose
        # tolerances.
        try:
            import torchvision
        except ImportError:
            self.skipTest("TorchVision not available")

        model = torchvision.models.resnet18()
        model.eval()
        batch_size = 16
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
        self.common(model, inputs, atol=1e-2, rtol=1e-2)

    def test_register_spills(self):
        # [FUNCTION DOC] Exercises the register-spill handling path of benchmark fusion. Replaces
        # the scheduler fused-node benchmark with a wrapper reporting latency 1.0 for any
        # non-infinite result, and disables dynamic rblock scaling. Compiles and runs a function
        # that does a multiply, add and row sum on many inputs (count from the NINP environment
        # variable, default 30) of shape 512 by 2560 and accumulates their sums. Only checks that
        # this runs without error.
        """
        The test can potentially trigger register spills
        """
        old_benchmark_fn = Scheduler.benchmark_fused_nodes

        def new_benchmark_fn(scheduler, nodes):
            # [FUNCTION DOC] HELPER: Replacement for the scheduler benchmark of fused nodes; calls
            # the saved original and, if the returned time is not infinite, overrides it with 1.0.
            # Returns the time and the path from the original.
            """
            We override Scheduler.benchmark_fused_nodes to return latency 1.0
            if there are no register spills. Without this, we may not able to
            test the code path handling register spilling because before register
            start spilling, the related fusion may have already been skipped
            due to longer lantency.
            """
            ms, path = old_benchmark_fn(scheduler, nodes)
            if not math.isinf(ms):
                ms = 1.0
            return ms, path

        # Disable dynamic_scale_rblock to make it easier to trigger register
        # spilling.
        with (
            unittest.mock.patch.object(
                Scheduler, "benchmark_fused_nodes", new_benchmark_fn
            ),
            config.patch("dynamic_scale_rblock", False),
        ):
            S = 512

            def f(*inputs):
                # [FUNCTION DOC] HELPER: Function under test; for each input applies a multiply,
                # add and sum over the last dim, collecting each result and a running sum. Returns
                # both.
                inputs = list(inputs)
                outputs = []
                out = torch.zeros(S, device=self.device)
                for x in inputs:
                    x = x * 2
                    x = x + 1
                    x = x.sum(dim=-1)
                    outputs.append(x)
                    out = out + x
                return outputs, out

            N = int(os.environ.get("NINP", "30"))
            inputs = [torch.randn(S, 2560, device=self.device) for _ in range(N)]
            opt_f = torch.compile(f)
            opt_f(*inputs)

    def test_foreach_kernel(self):
        # [FUNCTION DOC] Checks that benchmark fusion handles a foreach kernel (absolute value on a
        # list of two tensors followed by separate adds) by skipping benchmarking for it, compared
        # against eager using the host class checker.
        """
        Benchmark fusion should skip benchmarking kernels involves foreach kernel
        for now. Without the skipping logic, `codegen_node_schedule` may fail.
        """
        a = torch.randn(1024, 256, device=self.device)
        b = torch.randn(1024, 512, device=self.device)

        def f(a, b):
            # [FUNCTION DOC] HELPER: Function under test; foreach abs on two tensors, then adds one
            # and two respectively; returns both.
            a, b = torch._foreach_abs([a, b])
            return a + 1, b + 2

        self.common(f, (a, b))

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch(max_autotune_gemm_backends="TRITON")
    def test_avoid_register_spilling(self):
        # [FUNCTION DOC] Skipped unless on a big GPU. For GPU only, compiles a function with a
        # linear layer followed by repeated gelu and accumulation under max-autotune with Triton
        # gemm backend. Verifies that with benchmark fusion the generated code has two triton
        # kernel definitions and exactly two run calls (the test returns early if CI produced a
        # single kernel), then recompiles with benchmark_fusion and epilogue_fusion off and checks
        # that both outputs allocate exactly one buffer via empty_strided and reuse it before
        # returning.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("GPU only")

        from torch.nn.functional import gelu

        def foo(m, inp):
            # [FUNCTION DOC] HELPER: Function under test; applies the module, then four rounds of
            # gelu, adding all previous round outputs to the current value, and returns the last
            # value.
            curr = m(inp)
            tmps = []
            for _ in range(4):
                curr = gelu(curr)
                for t in tmps:
                    curr = curr + t
                tmps.append(curr)

            return curr

        m = torch.nn.Linear(2048, 2048, bias=True).half().to(GPU_TYPE)
        inp = torch.rand([2048, 2048]).half().to(GPU_TYPE)

        with torch.no_grad():
            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            _, out_code = run_and_get_code(foo_c, m, inp)

            # occasionally, CI will make this one kernel. just skip in this case
            if out_code[0].count("def triton_") != 2:
                return

            # should be multiple triton invocations
            FileCheck().check("async_compile.wait").check_count(
                ".run", 2, exactly=True
            ).run(out_code[0])

        with (
            config.patch({"benchmark_fusion": False, "epilogue_fusion": False}),
            torch.no_grad(),
        ):
            torch._dynamo.reset()

            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            _, out_code2 = run_and_get_code(foo_c, m, inp)

        for c in out_code[0], out_code2[0]:
            FileCheck().check("async_compile.wait").check("DeviceGuard").check_count(
                f"empty_strided_{GPU_TYPE}", 1, exactly=True
            ).check_regex("buf[0-9]* = buf[0-9]*; del buf[0-9]*").check("return").run(c)

    def test_tield_kernel_fusion(self):
        # [FUNCTION DOC] Checks fusion of a kernel with a tied transposed read: compiled function
        # realizes the sum of the input and its transpose, then adds one; 1024 by 1024 input,
        # compared with eager with the low-precision check disabled.
        def f(x):
            # [FUNCTION DOC] HELPER: Function under test; realizes the input plus its transpose,
            # adds one.
            y = realize(x + x.t())
            return y + 1

        x = torch.randn(1024, 1024, device=self.device)
        # Disable lowp check due to non-deterministic kernel fusion strategies affecting fp16 precision.
        self.common(f, (x,), check_lowp=False)


if HAS_GPU_AND_TRITON:

    class BenchmarkFusionGpuTest(TestCase):
        # [CLASS DOC] GPU concrete class: uses the GPU model checker and the GPU device. Template
        # tests are copied into it just below with the GPU device type as suffix.
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionGpuTest, GPU_TYPE)

    class BenchmarkingTest(TestCase):
        # [CLASS DOC] Holds a test that checks fused-kernel benchmarking runs on a device other
        # than the current default device.
        @unittest.skipIf(
            getattr(torch, GPU_TYPE).device_count() < 2,
            "The test need at least 2 devices",
        )
        @skip_if_cpp_wrapper("This tests triton scheduling directly")
        def test_benchmark_on_non_zero_device(self):
            # [FUNCTION DOC] Needs at least two GPUs and a non-cpp-wrapper mode. With device zero
            # current, compiles a relu-plus-input function on a large tensor placed on device one,
            # while wrapping the Triton scheduler benchmark of codegened modules to count calls and
            # assert every measured time is positive. Asserts the wrapper was called at least once.
            hit_count = 0
            with getattr(torch, GPU_TYPE).device(f"{GPU_TYPE}:0"):

                @torch.compile
                def relu(x):
                    # [FUNCTION DOC] HELPER: Compiled function; realizes relu of the input and adds
                    # the input.
                    return realize(x.relu()) + x

                x = torch.randn(int(16e6), device=f"{GPU_TYPE}:1")

                orig_benchmark_codegened_module = (
                    TritonScheduling.benchmark_codegened_module
                )

                def benchmark_codegened_module(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Wrapper around the Triton scheduler module benchmark;
                    # increments the outer hit counter, calls the original, asserts the time is
                    # positive and returns the original time and path.
                    nonlocal hit_count
                    hit_count += 1
                    ms, path = orig_benchmark_codegened_module(*args, **kwargs)
                    self.assertTrue(ms > 0)
                    return ms, path

                with unittest.mock.patch.object(
                    TritonScheduling,
                    "benchmark_codegened_module",
                    benchmark_codegened_module,
                ):
                    relu(x)
                self.assertTrue(hit_count > 0)

    class BenchmarkMultiTemplateFusionGpuTest(InductorTestCase):
        # [CLASS DOC] Tests for max-autotune epilogue fusion with benchmark_epilogue_fusion. Class
        # setup patches benchmark_kernel, benchmark_fusion and benchmark_epilogue_fusion on; each
        # test is skipped without a big GPU. Compares generated code with and without epilogue
        # benchmarking.
        @classmethod
        def setUpClass(cls):
            # [FUNCTION DOC] HELPER: Class setup; calls the parent setup, then creates an exit
            # stack on the class and enters a config patch enabling all three benchmarking options.
            super().setUpClass()
            cls._stack = contextlib.ExitStack()
            cls._stack.enter_context(
                config.patch(
                    {
                        "benchmark_kernel": True,
                        "benchmark_fusion": True,
                        "benchmark_epilogue_fusion": True,
                    }
                )
            )

        @classmethod
        def tearDownClass(cls):
            # [FUNCTION DOC] HELPER: Class teardown; closes the exit stack (undoing the config
            # patch) and then calls the parent teardown.
            cls._stack.close()
            super().tearDownClass()

        def setUp(self):
            # [FUNCTION DOC] HELPER: Per-test setup; calls the parent setup and skips the test if
            # the GPU is not big enough for max-autotune.
            super().setUp()
            if not is_big_gpu():
                return self.skipTest("Need a big GPU to run max_autotune=True")

        def _equivalent_output_code_impl(self, size, first_dim=None, activation=True):
            # [FUNCTION DOC] HELPER: Compiles a linear layer (optionally followed by relu) in
            # max-autotune mode with benchmark epilogue fusion on, runs it on half-precision GPU
            # tensors of the given size (first dim defaults to size), resets dynamo, then compiles
            # again with benchmark epilogue fusion off. Asserts the two results are close and
            # returns both generated code lists.
            def foo(m, inp):
                # [FUNCTION DOC] HELPER: Function under test; applies the module and, if enabled in
                # the enclosing helper, relu.
                a = m(inp)
                if activation:
                    return torch.nn.functional.relu(a)
                return a

            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)
            first_dim = first_dim if first_dim is not None else size

            m = torch.nn.Linear(size, size, bias=True).half().to(GPU_TYPE)
            inp = torch.rand([first_dim, size]).half().to(GPU_TYPE)

            with torch.no_grad():
                res, code = run_and_get_code(foo_c, m, inp)

            torch._dynamo.reset()
            with config.patch(benchmark_epilogue_fusion=False):
                foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)
                with torch.no_grad():
                    res2, code2 = run_and_get_code(foo_c, m, inp)

            self.assertEqual(res, res2, atol=1e-4, rtol=1.1)
            return code, code2

        @fresh_cache()
        @config.patch(
            {
                "max_autotune_gemm_backends": "TRITON",
                "benchmark_epilogue_fusion": False,
            }
        )
        def test_equivalent_template_code(self):
            # [FUNCTION DOC] With Triton gemm backend and epilogue benchmarking patched off and a
            # fresh cache, compares code from both compile modes for size 256: each must call the
            # wrapper once, allocate one buffer, contain the fused addmm relu template kernel and
            # delete three buffers (or reset three times with the cpp wrapper) before returning.
            code, code2 = self._equivalent_output_code_impl(256)
            for out_code in [code, code2]:
                FileCheck().check(get_func_call()).check_count(
                    "empty_strided", 1, exactly=True
                ).check("triton_tem_fused_addmm_relu_t_0").check_count(
                    ".reset()" if config.cpp_wrapper else "del", 3, exactly=True
                ).check("" if config.cpp_wrapper else "return").run(out_code[0])

        @fresh_cache()
        @config.patch(max_autotune_gemm_backends="ATEN")
        def test_equivalent_extern_code(self):
            # [FUNCTION DOC] With the ATEN gemm backend and a fresh cache, resets dynamo and
            # compares code from both compile modes for a 512 size, 1 row, no activation case: each
            # must call the wrapper, allocate once, use an extern kernel and delete three buffers
            # (reset three times with the cpp wrapper) before returning.
            torch._dynamo.reset()

            code, code2 = self._equivalent_output_code_impl(512, 1, False)

            for out_code in [code, code2]:
                FileCheck().check(get_func_call()).check_count(
                    "empty_strided", 1, exactly=True
                ).check("" if config.cpp_wrapper else "extern_kernels.").check_count(
                    ".reset()" if config.cpp_wrapper else "del", 3, exactly=True
                ).check("" if config.cpp_wrapper else "return").run(out_code[0])

        def test_changed_layout(self):
            # [FUNCTION DOC] Checks that layout changes made while planning addmm results feeding a
            # concat are propagated: compares a max-autotune compiled concat of two addmm results
            # over 4 by 4 tensors against eager, then resets dynamo.
            # cat addmm planning will change layout - make sure propagated
            def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
                # [FUNCTION DOC] HELPER: Function under test; concatenates two addmm results along
                # dim 1.
                return torch.cat(
                    [
                        torch.addmm(a, b, c),
                        torch.addmm(b, c, a),
                    ],
                    1,
                )

            args = [
                torch.randn(4, 4, device=GPU_TYPE),
                torch.randn(4, 4, device=GPU_TYPE),
                torch.randn(4, 4, device=GPU_TYPE),
            ]

            expected = fn(*args)
            actual = torch.compile(fn, mode="max-autotune")(*args)
            self.assertEqual(expected, actual)

            torch._dynamo.reset()


if HAS_CPU and not torch.backends.mps.is_available():

    class BenchmarkFusionCpuTest(TestCase):
        # [CLASS DOC] CPU concrete class: uses the CPU model checker and device name cpu. Template
        # tests are copied into it just below with the cpu suffix.
        common = check_model
        device = "cpu"

    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCpuTest, "cpu")

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU_AND_TRITON:
        run_tests()
