# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the inductor back-to-back GEMM (b2b_gemm) pass, which fuses two chained
# matmuls (optionally with a pointwise op between) into one Triton kernel. Contains
# correctness/pattern tests that check the b2b_gemm counter or the presence of the B2B entrance
# markers in generated code, and opt-in performance tests enabled by DO_PERF_TEST. Tests run only
# when a GPU is available when executed as a script; the class is skipped on XPU.
import os
import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


@skipIfXpu(msg="Segmentation fault on CI machine")
class B2BGEMMTest(TestCase):
    # [CLASS DOC] Test case for the b2b_gemm pass on the GPU device. Each correctness test compiles
    # a three-matrix chain with the pass enabled and either compares against a float32 baseline and
    # checks the b2b_gemm counter, or checks the pass did not trigger. Perf tests print speedups.
    device = GPU_TYPE

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_left_assoc_good_shape(self):
        # [FUNCTION DOC] Left-associated pattern (pointwise GELU of the first product, then times
        # the third matrix) with shapes good for the pass. Compiled output must match a float32
        # baseline within tolerance and the b2b_gemm counter must be positive.
        """
        left_assoc means the pattern is (subgraph(A @ B) @ C)
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference function; GELU of the product of the first two
            # inputs, then matrix product with the third.
            g = torch.nn.GELU()
            return torch.mm(g(torch.mm(m1, m2)), m3)

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Baseline that casts inputs to float32, runs the reference
            # function and casts the result back to float16, so precision is compared fairly with
            # the fused kernel.
            """
            When the optimization is applied,
            the Triton kernel is more precise than the above f,
            because it internally uses float32 for accumulation while the above f uses float16.
            To ensure a fair comparison,
            we promote the baseline f to float32 for precision comparison.
            This actually reduced some atol's in the tests from 0.2 to 0.1.
            """
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_right_assoc_good_shape(self):
        # [FUNCTION DOC] Right-associated pattern (first matrix times the ReLU of the product of
        # the last two) with good shapes. Result must match the float32 baseline within tolerance
        # and the b2b_gemm counter must be positive.
        """
        right_assoc means the pattern is (A @ subgraph(B @ C))
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference function; first input times the ReLU of the product
            # of the second and third.
            g = torch.nn.ReLU()
            return torch.mm(m1, g(torch.mm(m2, m3)))

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Float32 baseline wrapper around the reference function; casts
            # the result back to float16.
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_trivial_left_assoc_good_shape(self):
        # [FUNCTION DOC] Plain left-associated chain (no pointwise op) with good shapes. Result
        # must match the float32 baseline and the b2b_gemm counter must be positive.
        """
        trivial_left_assoc means the pattern is ((A @ B) @ C)
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference function; product of the first two inputs times the
            # third.
            return torch.mm(torch.mm(m1, m2), m3)

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Float32 baseline wrapper around the reference function; casts
            # the result back to float16.
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_trivial_right_assoc_good_shape(self):
        # [FUNCTION DOC] Plain right-associated chain (no pointwise op) with good shapes. Result
        # must match the float32 baseline and the b2b_gemm counter must be positive.
        """
        trivial_right_assoc means the pattern is (A @ (B @ C))
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference function; first input times the product of the other
            # two.
            return torch.mm(m1, torch.mm(m2, m3))

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Float32 baseline wrapper around the reference function; casts
            # the result back to float16.
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_bad_pattern_good_shape(self):
        # [FUNCTION DOC] Negative test with a pattern the pass does not support (the first product
        # is reused by two later matmuls). Compiles with the pass on, checks numerical closeness to
        # eager and asserts neither B2B Triton entrance marker appears in the generated code.
        """
        bad_pattern means the code does not contain the supported patterns
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Function under test; computes two chained products that both
            # consume the first product and returns the product of the first and second results.
            mm1 = torch.mm(m1, m2)
            mm2 = torch.mm(mm1, m3)
            return torch.mm(mm1, mm2)

        f_opt = torch.compile(f)
        A = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        res, codes = run_and_get_code(f_opt, A, B, C)
        code = "\n".join(codes)
        self.assertTrue(torch.allclose(f(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" not in code)
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" not in code)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_good_pattern_bad_shape(self):
        # [FUNCTION DOC] Negative test with a supported pattern but square 100 by 100 inputs, which
        # are bad shapes for the pass. Checks numerical closeness to eager and that no B2B Triton
        # entrance marker appears in the generated code.
        """
        bad_shape means the sizes are not good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Function under test; product of the first two inputs times the
            # third.
            return torch.mm(torch.mm(m1, m2), m3)

        f_opt = torch.compile(f)
        A = torch.randn((100, 100), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((100, 100), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((100, 100), device=GPU_TYPE, dtype=torch.float16)
        res, codes = run_and_get_code(f_opt, A, B, C)
        code = "\n".join(codes)
        self.assertTrue(torch.allclose(f(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" not in code)
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" not in code)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @torch._dynamo.config.patch(recompile_limit=32)
    def test_plain_b2b_gemm_performance(self):
        # [FUNCTION DOC] Opt-in performance test (skipped unless DO_PERF_TEST is 1) for the plain
        # chain. For a grid of sizes, benchmarks the compiled chain with the pass off and on,
        # prints a table of speedups and the geometric mean, and asserts nothing (the check is
        # disabled as flaky).
        """compare torch.compile(f, b2b_gemm = off) with torch.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            # [FUNCTION DOC] HELPER: Compiles the plain chain without dynamic shapes using default
            # config and returns the benchmarker time for it (warmup 100, repeat 500).
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Function being benchmarked; product of the first two
                # inputs times the third.
                return torch.mm(torch.mm(m1, m2), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @torch._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            # [FUNCTION DOC] HELPER: Same as the pass-off helper but run with the b2b_gemm pass
            # enabled through a config patch; returns the benchmarker time.
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Function being benchmarked; product of the first two
                # inputs times the third.
                return torch.mm(torch.mm(m1, m2), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for Plain B2B-GEMM:")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = M, N
                A = torch.randn((M, N), device=GPU_TYPE, dtype=torch.float16)
                B = torch.randn((N, O), device=GPU_TYPE, dtype=torch.float16)
                C = torch.randn((O, P), device=GPU_TYPE, dtype=torch.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @torch._dynamo.config.patch(recompile_limit=32)
    def test_gelu_b2b_gemm_performance(self):
        # [FUNCTION DOC] Opt-in performance test (skipped unless DO_PERF_TEST is 1) for the GELU
        # pattern. Uses square-ish shapes over a grid of sizes, compares pass off against pass on,
        # prints a table of speedups and the geometric mean, and asserts nothing (disabled as
        # flaky).
        """compare torch.compile(f, b2b_gemm = off) with torch.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            # [FUNCTION DOC] HELPER: Compiles the GELU chain without dynamic shapes using default
            # config and returns the benchmarker time for it (warmup 100, repeat 500).
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Function being benchmarked; GELU of the first product,
                # then times the third matrix.
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @torch._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            # [FUNCTION DOC] HELPER: Same as the pass-off helper but run with the b2b_gemm pass
            # enabled through a config patch; returns the benchmarker time.
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Function being benchmarked; GELU of the first product,
                # then times the third matrix.
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for GELU B2B-GEMM:")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = M, N
                A = torch.randn((M, N), device=GPU_TYPE, dtype=torch.float16)
                B = torch.randn((N, O), device=GPU_TYPE, dtype=torch.float16)
                C = torch.randn((O, P), device=GPU_TYPE, dtype=torch.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @torch._dynamo.config.patch(recompile_limit=32)
    def test_gelu_mlp_b2b_gemm_performance(self):
        # [FUNCTION DOC] Opt-in performance test (skipped unless DO_PERF_TEST is 1) for the GELU
        # MLP shape, where the inner and output dims equal the second size of the grid. Compares
        # pass off against on, prints the speedup table and geometric mean, and asserts nothing
        # (disabled).
        """compare torch.compile(f, b2b_gemm = off) with torch.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            # [FUNCTION DOC] HELPER: Compiles the GELU chain without dynamic shapes using default
            # config and returns the benchmarker time for it (warmup 100, repeat 500).
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Function being benchmarked; GELU of the first product,
                # then times the third matrix.
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @torch._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            # [FUNCTION DOC] HELPER: Same as the pass-off helper but run with the b2b_gemm pass
            # enabled through a config patch; returns the benchmarker time.
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Function being benchmarked; GELU of the first product,
                # then times the third matrix.
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for GELU B2B-GEMM (MLP):")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = N, N
                A = torch.randn((M, N), device=GPU_TYPE, dtype=torch.float16)
                B = torch.randn((N, O), device=GPU_TYPE, dtype=torch.float16)
                C = torch.randn((O, P), device=GPU_TYPE, dtype=torch.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
