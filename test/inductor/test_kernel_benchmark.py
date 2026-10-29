# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# [MODULE DOC] Tests for Inductor kernel benchmarking support (the benchmark_kernel config).
# Tests compile small functions, then run the generated kernel files or compiled modules in a
# subprocess with benchmark flags and check the printed memory bandwidth (GB/s and gigabyte
# counts) against hand computed byte counts for pointwise, reduction, slice, cat, matmul,
# scatter and scan kernels. Others check that the cleaned triton output no longer depends on
# Inductor runtime and that the generated benchmark entry passes the device name. Requires a
# GPU; the entry point only runs when one exists.
import contextlib
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import torch._inductor.async_compile
from torch._dynamo.testing import rand_strided
from torch._inductor import config
from torch._inductor.codecache import PyCodeCache
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, run_and_get_code, run_and_get_kernels
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import xfailIfSM89
from torch.testing._internal.common_utils import recover_orig_fp32_precision
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_BIG_GPU


class TestKernelBenchmark(TestCase):
    # [CLASS DOC] Kernel benchmark tests. The class setup enables the benchmark_kernel config
    # for all tests and records an augmented PYTHONPATH in a class attribute so subprocesses see
    # the same import path as the parent. Helper methods fetch the single compiled module from
    # the code cache and run kernel files or modules in subprocesses to inspect their output.
    device_type = GPU_TYPE

    # to make sure the subprocess runs on the exact same path as the parent process
    # we augment the PYTHONPATH env var
    python_path = ""

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: open an exit stack with the benchmark_kernel config
        # patched to true, and store a colon joined copy of the interpreter path (prefixed by
        # any existing PYTHONPATH value) as the class attribute used by subprocess runs. It does
        # not call the parent class setup.
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(patch.object(config, "benchmark_kernel", True))
        # setup the augmented PYTHONPATH to pass to the subprocess calls
        augmented_pp = ":".join(sys.path)
        if os.environ.get("PYTHONPATH"):
            augmented_pp = f"{os.environ.get('PYTHONPATH')}:{augmented_pp}"
        cls.python_path = augmented_pp

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: close the exit stack, restoring the patched config.
        cls.exit_stack.close()

    def setUp(self):
        # [FUNCTION DOC] Call the parent setUp and clear the Python code cache so each test
        # starts with no compiled modules.
        super().setUp()
        PyCodeCache.cache_clear()

    def get_compiled_module(self):
        # [FUNCTION DOC] HELPER: Scan the code cache modules for those exposing a benchmark
        # entry function, asserting there is at most one, and then that one exists; returns it.
        compiled_module = None
        for v in PyCodeCache.modules:
            if hasattr(v, "benchmark_compiled_module"):
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)
        return compiled_module

    def run_kernel_benchmark(self, kernel_path):
        # [FUNCTION DOC] HELPER: Run the given kernel file with the current Python interpreter
        # in a subprocess using the augmented PYTHONPATH and return its decoded combined output.
        # On a non-zero exit it prints the error and the captured output and re-raises the
        # error.
        try:
            bench_out = subprocess.check_output(
                f"{sys.executable} {kernel_path}".split(),
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": self.python_path},
            ).decode()
        except subprocess.CalledProcessError as e:
            print("Failed when running output code", e)
            print(e.output.decode())
            raise e
        return bench_out

    def verify_compiled_kernels(self, GB_count=1):
        # [FUNCTION DOC] HELPER: Run the single compiled module in a subprocess with the flag
        # that benchmarks all autotuning configs, then check the output contains the bandwidth
        # marker GB/s at least the given number of times. On subprocess failure prints the error
        # and output and re-raises.
        compiled_module = self.get_compiled_module()
        # now run the compiled module in subprocess and check its output
        try:
            bench_out = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__} -kc".split(),
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": self.python_path},
            ).decode()
        except subprocess.CalledProcessError as e:
            print("Failed when running output code", e)
            print(e.output.decode())
            raise e

        # make sure we have the bandwidth information in the output
        # -kc flag benchmarks all autotuning configs,
        # so we check for at least GB_count occurrences rather than exactly.
        FileCheck().check_count(
            "GB/s",
            GB_count,
        ).run(bench_out)

    def verify_remove_inductor_deps(self, compiled_module):
        # [FUNCTION DOC] HELPER: Run the compiled module in a subprocess with launch parameter
        # dumping enabled, then produce a cleaned triton version of the file with the clean
        # triton utility. Asserts the cleaned text has no triton heuristics decorator and no run
        # calls, runs the cleaned file in a subprocess, and returns the cleaned text. Subprocess
        # failures print diagnostics and are re-raised. Side effect: writes a cleaned file next
        # to the module.
        try:
            out = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__}".split(),
                env={
                    **os.environ.copy(),
                    "TORCHINDUCTOR_DUMP_LAUNCH_PARAMS": "1",
                    "PYTHONPATH": self.python_path,
                },
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(
                "Failed when runinng triton code with TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1",
                e,
            )
            print(e.output.decode())
            raise e
        from torch.utils._get_clean_triton import get_clean_triton

        cleaned_triton = get_clean_triton(
            compiled_module.__file__, f"{compiled_module.__file__}.cleaned"
        )
        self.assertTrue("@triton_heuristics" not in cleaned_triton)
        self.assertTrue(".run(" not in cleaned_triton)
        try:
            out = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__}.cleaned".split(),
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": self.python_path},
            )
        except subprocess.CalledProcessError as e:
            print("Failed when when running cleaned triton", e)
            print(e.output.decode())
            print(cleaned_triton)
            raise e
        return cleaned_triton

    def check_bandwidth(self, compiled_module, num_gb):
        # [FUNCTION DOC] HELPER: Run the compiled module in a subprocess with the kernel
        # benchmark flag and assert the output contains the given gigabyte figure followed by
        # the unit exactly once. On subprocess failure prints the error and output and
        # re-raises.
        # now run the compiled module in subprocess and check its output
        try:
            bench_out = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__} -k".split(),
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": self.python_path},
            ).decode()
        except subprocess.CalledProcessError as e:
            print("Failed when running output code", e)
            print(e.output.decode())
            raise e

        # make sure we have the bandwidth information in the output
        FileCheck().check_count(
            f"{num_gb} GB ",
            1,
            exactly=1,
        ).run(bench_out)

    def test_plus1_kernel_benchmark(self):
        # [FUNCTION DOC] Compiles a function adding one, extracts the generated kernel source,
        # writes it to the code cache, runs it as a script, and checks the output reports
        # bandwidth.
        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Return x plus one.
            return x + 1

        x = torch.randn(1024, device=GPU_TYPE)
        _, (kernel_code,) = run_and_get_kernels(f, x, remove_quote=True)
        _, path = PyCodeCache.write(kernel_code)
        bench_output = self.run_kernel_benchmark(path)
        self.assertTrue("GB/s" in bench_output)

    def test_pw_kernel_benchmark(self):
        # [FUNCTION DOC] Compiles a pointwise sine plus cosine function, runs it, and verifies
        # the compiled module reports bandwidth for its kernels.
        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Return the sum of sine and cosine of x.
            return torch.sin(x) + torch.cos(x)

        inp = torch.rand(2, 3).to(device=GPU_TYPE)
        out = f(inp)
        self.verify_compiled_kernels()

    # TODO: Currently the Triton mm template +  relu fusion causes slowdown on XPU,
    # Need to refine the template and config for XPU.
    @config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @fresh_cache()
    def test_matmul_triton_kernel_benchmark(self):
        # [FUNCTION DOC] With max autotune restricted to the triton gemm backend and shape
        # padding forced (skipped on small GPUs), compiles a half precision matmul followed by
        # relu and verifies the compiled module reports bandwidth.
        M = 12544
        N = 256
        K = 64
        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(N, K, dtype=torch.float16, device=GPU_TYPE).t()

        @torch.compile
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return the relu of the matrix product of a and b.
            return torch.relu(a @ b)

        f(a, b)
        self.verify_compiled_kernels()

    @config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", shape_padding=False
    )
    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @fresh_cache()
    def test_mm_triton_kernel_benchmark(self):
        # [FUNCTION DOC] With max autotune on the triton gemm backend and no shape padding
        # (skipped on small GPUs), compiles a matmul on a narrowed slice of an oddly strided
        # input and verifies bandwidth is reported at least once.
        M = 2048
        N = 2432
        K = 1949
        K_2 = 3581
        a = rand_strided((M, K_2), (K_2, 1), device=GPU_TYPE, dtype=torch.float16)
        b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float16)

        @torch.compile
        def f(a, b):
            # [FUNCTION DOC] HELPER: Narrow the second dimension of a to its first K columns and
            # return the matrix product with b.
            a_1 = torch.narrow(a, 1, 0, K)
            c = torch.mm(a_1, b)
            return c

        f(a, b)

        self.verify_compiled_kernels(GB_count=1)

    @recover_orig_fp32_precision
    def test_matmul_bandwidth_computation(self):
        # [FUNCTION DOC] Does a matmul followed by an elementwise square, where the matmul stays
        # in aten so only the multiply is a generated kernel. That kernel updates a million
        # element float tensor in place, so traffic is counted twice, about 0.008 gigabytes; the
        # test asserts that figure appears. Sets float32 matmul precision to high as a side
        # effect.
        """
        The test does a matmul and then mul. Without max-autotune, we use
        the matmul in aten. So there is a single triton kernel for mul.
        The kernel we generated is like:

            @triton.jit
            def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):

        Note the in_out_ptr0 argument. It's for a 1000x1000 tensor, but it's
        inplace updated, so when computing the bandwidth, we should count
        the total memory access as 2 * 1000 * 1000 * 4 = 8MB. This amount is
        what this test asserts.
        """
        torch.set_float32_matmul_precision("high")  # suggested by a warning

        @torch.compile
        def f(x, y):
            # [FUNCTION DOC] HELPER: Return the square of the matrix product of x and y.
            z = x @ y
            w = z * z
            return w

        M, N, K = 1000, 1000, 10
        x = torch.rand(M, K).to(device=GPU_TYPE)
        y = torch.rand(K, N).to(device=GPU_TYPE)
        out = f(x, y)

        compiled_module = self.get_compiled_module()

        self.check_bandwidth(compiled_module, 0.008)

    def test_unused_input_bandwidth_computation(self):
        # [FUNCTION DOC] Compiles an addition of two of three half precision inputs, with
        # dynamic first dimension. The unused input must not be counted; expected traffic is two
        # inputs and the output, about 0.030 gigabytes.
        M, N = 5, 1000000

        @torch.compile
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Return a plus c, ignoring b.
            return a + c

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(c, 0)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_c + size_out
        # num_gb = (5 * 1000000 + 5 * 1000000 + 5 * 1000000) * 2 / 1e9
        #        = 0.030
        self.check_bandwidth(compiled_module, "0.030")

    def test_reduction_bandwidth_computation(self):
        # [FUNCTION DOC] Compiles a sum reduction over the middle dimension of a half precision
        # tensor. Expected traffic is input plus output, about 0.042 gigabytes.
        @torch.compile
        def f(a):
            # [FUNCTION DOC] HELPER: Return the sum of a over dimension 1.
            return torch.sum(a, dim=1)

        a = torch.rand(1000, 20, 1000, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a,)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_out
        # num_gb = (1000 * 20 * 1000 + 1000 * 1000) * 2 / 1e9
        #        = 0.042
        self.check_bandwidth(compiled_module, "0.042")

    @config.patch(max_autotune=True)
    def test_fused_layernorm_bandwidth_computation(self):
        # [FUNCTION DOC] With max autotune on, compiles an add, layer norm with weight and bias,
        # sigmoid and multiply on half precision inputs. Expected traffic is the matrix, three
        # vectors and the output, about 0.046 gigabytes.
        M, N = 10, 1000000

        @torch.compile
        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Add a and b, layer normalize with weight c and bias d,
            # apply sigmoid, and return the sum times the sigmoid.
            x0 = a + b
            x1 = torch.nn.functional.layer_norm(
                x0, normalized_shape=(N,), weight=c, bias=d, eps=1e-05
            )
            x2 = torch.sigmoid(x1)
            return x0 * x2

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        d = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c, d)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_b + size_c + size_d + size_out
        # num_gb = (10 * 1000000 + 1000000 + 1000000 + 1000000 + 10 * 1000000) * 2 / 1e9
        #        = 0.046
        self.check_bandwidth(compiled_module, "0.046")

    def test_slice_add_cat_bandwidth_computation(self):
        # [FUNCTION DOC] Compiles a function that narrows b, adds a broadcast vector, and
        # concatenates with a. The slice size is overestimated because of the cat, giving about
        # 0.052 gigabytes as expected traffic.
        M, N = 5, 1000000

        @torch.compile
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Slice b along dimension 1, add c with broadcasting, and
            # concatenate a with the result along dimension 1.
            x0 = torch.narrow(b, 1, N, N)
            # broadcasting
            x1 = x0 + c
            return torch.cat([a, x1], dim=1)

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N * 5, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # we overestimate the size of "slice_b" due to torch.cat
        # num_gp = size_a + size_slice_b + size_c + size_out
        # num_gb = (5 * 1000000 + 5 * 2000000 + 1000000 + 5 * 2000000) * 2 / 1e9
        #        = 0.052
        self.check_bandwidth(compiled_module, "0.052")

    def test_slice_add_bandwidth_computation(self):
        # [FUNCTION DOC] Compiles a function adding a, a slice of b and a broadcast vector.
        # Expected traffic is about 0.032 gigabytes.
        M, N = 5, 1000000

        @torch.compile
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Slice b along dimension 1 and return a plus the slice plus
            # c.
            x0 = torch.narrow(b, 1, N, N)
            return a + x0 + c

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N * 5, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_slice_b + size_c + out_size
        # num_gb = (5 * 1000000 + 5 * 1000000 + 1000000 + 5 * 1000000) * 2 / 1e9
        #        = 0.032
        self.check_bandwidth(compiled_module, "0.032")

    def test_mm_slice_add_bandwidth_computation(self):
        # [FUNCTION DOC] A matmul plus two different slices. The matmul plus first slice becomes
        # an extern addmm, so only the pointwise add of its output and the second slice is
        # counted, about 0.006 gigabytes.
        M, N, K = 1000, 1000, 30

        @torch.compile
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Return the matrix product of a and b plus two different
            # narrowed slices of c.
            x0 = torch.mm(a, b)
            x1 = torch.narrow(c, 1, 20 * N, N)
            x2 = torch.narrow(c, 1, 21 * N, N)
            return x0 + x1 + x2

        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, N * 100, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # torch.mm + x1 becomes an extern addmm kernel, so we measure the nbytes
        # for the pointwise kernel adding that result to x2:
        # num_gb = addmm_out + size_slice_c + size_out
        # num_gb = (1000 * 1000 + 1000 * 1000 + 1000 * 1000) * 2 / 1e9
        #        = 0.006
        num_gb = "0.006"
        self.check_bandwidth(compiled_module, num_gb)

    def test_mm_slice_add_bandwidth_computation_2(self):
        # [FUNCTION DOC] Like the previous test but both slices are identical, so the shared
        # access is counted once. The matmul is extern and the pointwise add is expected to move
        # about 0.006 gigabytes.
        M, N, K = 1000, 1000, 30

        @torch.compile
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Return the matrix product of a and b plus two identical
            # narrowed slices of c.
            x0 = torch.mm(a, b)
            x1 = torch.narrow(c, 1, 20 * N, N)
            x2 = torch.narrow(c, 1, 20 * N, N)
            return x0 + x1 + x2

        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, N * 100, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # torch.mm becomes an extern kernel, so we measure the nbytes
        # for the pointwise add kernel:
        # num_gb = x0 + size_slice_c + size_out
        # num_gb = (1000 * 1000 + 1000 * 1000 + 1000 * 1000) * 2 / 1e9
        #        = 0.006
        # note that we only count one size_slice_c because two accesses
        # have the same index.
        self.check_bandwidth(compiled_module, "0.006")

    @xfailIfSM89
    @config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_slice_mm_bandwidth_computation(self):
        # [FUNCTION DOC] Compiles a matmul on a narrowed slice of a, with the triton gemm
        # backend and forced shape padding (expected failure on SM89; raises skip on small XPU
        # devices). Expected traffic is the output, the sliced input and the weights, about
        # 0.022 gigabytes.
        if GPU_TYPE == "xpu" and not torch._inductor.utils.is_big_gpu():
            raise unittest.SkipTest("unsupported device")

        M, N, K = 1000, 2000, 3000

        @torch.compile
        def f(a, b):
            # [FUNCTION DOC] HELPER: Narrow a along dimension 1 to K columns starting at K and
            # return the matrix product with b.
            x = torch.narrow(a, 1, K, K)
            return torch.mm(x, b)

        a = torch.rand(M, 3 * K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        inputs = (a, b)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()

        # c[1000, 2000] = x[1000, 3000] @ b[3000, 2000]
        # num_gb = (1000 * 2000 + 1000 * 3000 + 3000 * 2000) * 2 / 1e9
        #        = 0.022
        self.check_bandwidth(compiled_module, "0.022")

    def test_star_dep(self):
        # [FUNCTION DOC] Tests bandwidth estimation for an indexed assignment, which uses a star
        # dependency. Counts the 80 KB index tensor plus the 200 MB of rows written, expecting
        # 0.200 gigabytes.
        """
        Test the bandwidth estimation for StarDep
        """

        @torch.compile
        def f(a, b):
            # [FUNCTION DOC] HELPER: Assign the constant 3.0 into a at the indices in b, in
            # place.
            a[b] = 3.0

        a = torch.rand(10000, 5000, device=GPU_TYPE)
        b = torch.randint(
            0, 10000, [20000], device=GPU_TYPE, dtype=torch.int32
        ).unsqueeze(1)
        f(a, b)
        compiled_module = self.get_compiled_module()
        # 20000 * 4 = 80KB for b
        # 20000 * 5000 * 4 = 200MB for a
        self.check_bandwidth(compiled_module, "0.200")

    def test_split_scan(self):
        # [FUNCTION DOC] Compiles a cumulative sum over a flattened tensor, which is a split
        # scan. Counts the 200 MB input and an equal sized output, expecting 0.400 gigabytes.
        @torch.compile
        def f(a):
            # [FUNCTION DOC] HELPER: Return the cumulative sum of a over its last dimension.
            return a.cumsum(-1)

        a = torch.rand(10000, 5000, device=GPU_TYPE)
        f(a.reshape(-1))
        compiled_module = self.get_compiled_module()
        # 10000 * 5000 * 4 = 200 MB for a
        # Double that for output as well
        self.check_bandwidth(compiled_module, "0.400")

    @config.patch("triton.unique_kernel_names", True)
    @config.patch(benchmark_kernel=False)
    @config.patch(compile_threads=1)
    def test_remove_inductor_deps(self):
        # [FUNCTION DOC] With unique kernel names, benchmark_kernel off and one compile thread,
        # compiles a cos then sin function and runs the dependency removal check on the compiled
        # module.
        @torch.compile
        def f(a):
            # [FUNCTION DOC] HELPER: Return sine of cosine of a.
            return a.cos().sin()

        a = torch.randn(5, device=GPU_TYPE)
        f(a)
        compiled_module = self.get_compiled_module()
        cleaned_triton = self.verify_remove_inductor_deps(compiled_module)

    @config.patch("triton.unique_kernel_names", True)
    @config.patch(benchmark_kernel=False)
    @config.patch(compile_threads=1)
    def test_remove_inductor_deps_multiple_kernels(self):
        # [FUNCTION DOC] Same dependency removal check for a function with two matmuls, a cos
        # and sin pair, and a softmax, producing several kernels.
        @torch.compile
        def f(a):
            # [FUNCTION DOC] HELPER: Matrix multiply a with itself, apply cos then sin, multiply
            # again, and return the softmax over the last dimension.
            a = torch.mm(a, a)
            a = a.cos().sin()
            a = torch.mm(a, a)
            a = torch.softmax(a, dim=-1)
            return a

        a = torch.randn(5, 5, device=GPU_TYPE)
        f(a)
        compiled_module = self.get_compiled_module()
        self.verify_remove_inductor_deps(compiled_module)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch("triton.unique_kernel_names", True)
    @config.patch(benchmark_kernel=False)
    @config.patch(compile_threads=1)
    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_remove_inductor_deps_templates(self):
        # [FUNCTION DOC] Same dependency removal check with max autotune on the triton gemm
        # backend (skipped on small GPUs), so the generated code includes template kernels.
        @torch.compile
        def f(a):
            # [FUNCTION DOC] HELPER: Matrix multiply a with itself, apply cos, multiply again,
            # and apply sin.
            a = torch.mm(a, a)
            a = a.cos()
            a = torch.mm(a, a)
            a = a.sin()
            return a

        a = torch.randn(128, 128, device=GPU_TYPE)
        f(a)
        compiled_module = self.get_compiled_module()
        self.verify_remove_inductor_deps(compiled_module)

    @config.patch("triton.unique_kernel_names", True)
    @config.patch(benchmark_kernel=False)
    @config.patch(compile_threads=1)
    def test_remove_inductor_deps_scalar(self):
        # [FUNCTION DOC] Same dependency removal check for adding two zero dimensional tensors.
        @torch.compile
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return a plus b.
            return a + b

        a = torch.tensor(1.0, device=GPU_TYPE)
        b = torch.tensor(2.0, device=GPU_TYPE)
        f(a, b)
        compiled_module = self.get_compiled_module()
        self.verify_remove_inductor_deps(compiled_module)

    def test_benchmark_compiled_module_device_arg(self):
        # [FUNCTION DOC] Regression test for the benchmark entry in generated code: compiles an
        # add one function, captures the generated source, and checks it passes the current GPU
        # device name to the performance printing call.
        """Regression test for https://github.com/pytorch/pytorch/issues/181954."""

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Return x plus one.
            return x + 1

        x = torch.randn(1024, device=GPU_TYPE)
        _, (src,) = run_and_get_code(f, x)

        FileCheck().check(
            f"print_performance(fn, times=times, repeat=repeat, device='{GPU_TYPE}')"
        ).run(src)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
