# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Tests for the Halide backend of inductor. Defines a factory that clones existing
# test classes from test_torchinductor with config patches selecting the halide backend, plus a
# small HalideTests class that exercises the Halide code cache, scheduling modes, random number
# consistency with triton, compile options and an in-place alias case. Import-time side effects:
# on Windows CI it writes to stderr and exits or skips; HAS_HALIDE is probed by importing
# halide; the generated Halide test classes are registered in module globals when CPU (or GPU
# with the TEST_HALIDE_GPU env var set to one) tests and halide are available.
import functools
import itertools
import os
import sys
import textwrap
import unittest

import torch
import torch._inductor.async_compile
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.codecache import HalideCodeCache
from torch._inductor.runtime.hints import HalideInputSpec, HalideMeta
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import parallel_num_threads, run_and_get_code
from torch.testing._internal.common_utils import IS_CI, IS_MACOS, IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CPU
from torch.utils._triton import has_triton


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    import halide  # @manual

    HAS_HALIDE = halide is not None
except ImportError:
    HAS_HALIDE = False


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library


test_classes = {}


def make_halide(cls):
    # [FUNCTION DOC] Builds a Halide variant of a test class using make_test_cls_with_patches
    # with the Halide name prefix and suffix. Patches enable halide scan kernels, set both cpu
    # and cuda backends to halide and disable runtime triton dtype and shape asserts; tests
    # listed under the halide expected-failure property are treated as expected failures.
    # Records the new class in test_classes, publishes it into module globals (required for test
    # discovery), sets its module name to this module and returns it.
    suffix = "_halide"

    cls_prefix = "Halide"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "halide.scan_kernels", True),
        (config, "cpu_backend", "halide"),
        (config, "cuda_backend", "halide"),
        (config, "test_configs.runtime_triton_dtype_assert", False),
        (config, "test_configs.runtime_triton_shape_assert", False),
        xfail_prop="_expected_failure_halide",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


@unittest.skipUnless(HAS_HALIDE, "requires halide")
class HalideTests(TestCase):
    # [CLASS DOC] Halide-specific tests, skipped entirely when halide is not importable. Covers
    # direct use of the Halide code cache with an autoscheduler and a manual schedule, plus end
    # to end checks through torch.compile.
    def test_codecache(self):
        # [FUNCTION DOC] Compiles a hand written Halide generator (elementwise add of two 1024
        # float buffers) through the code cache with the Mullapudi2016 autoscheduler and a
        # thread count from parallel_num_threads. Calls the resulting function on random CPU
        # tensors and asserts the output buffer equals the sum of the inputs.
        fn = HalideCodeCache.generate_halide(
            HalideMeta(
                argtypes=[
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr1",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                ],
                target="host-no_runtime",
                scheduler="Mullapudi2016",
                scheduler_flags={
                    "parallelism": parallel_num_threads(),
                },
            ),
            textwrap.dedent(
                """
                import halide as hl

                @hl.generator(name="kernel")
                class Kernel:
                    in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                    in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                    out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                    def generate(g):
                        in_ptr0 = g.in_ptr0
                        in_ptr1 = g.in_ptr1
                        out_ptr0 = g.out_ptr0
                        xindex = hl.Var('xindex')
                        x0 = xindex
                        tmp0 = hl.Func()
                        tmp0[xindex] = in_ptr0[x0]
                        tmp1 = hl.Func()
                        tmp1[xindex] = in_ptr1[x0]
                        tmp2 = hl.Func()
                        tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                        out_ptr0[x0] = tmp2[xindex]

                        assert g.using_autoscheduler()
                        in_ptr0.set_estimates([hl.Range(1024, 1024)])
                        in_ptr1.set_estimates([hl.Range(1024, 1024)])
                        out_ptr0.set_estimates([hl.Range(1024, 1024)])

                __name__ == '__main__' and hl.main()
                """,
            ),
        )
        a = torch.randn(1024)
        b = torch.randn(1024)
        c = torch.randn(1024)
        fn(a, b, c)
        self.assertEqual(c, a + b)

    def test_manual_schedule(self):
        # [FUNCTION DOC] Same elementwise add generator as the code cache test but with no
        # scheduler, so the generator applies a manual schedule (split, parallel, vectorize,
        # compute_at/store_at, inline). Runs it on random CPU tensors and asserts the output
        # equals the sum.
        fn = HalideCodeCache.generate_halide(
            HalideMeta(
                argtypes=[
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr1",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                ],
                target="host-no_runtime",
                scheduler=None,
            ),
            textwrap.dedent(
                """
                import halide as hl

                @hl.generator(name="kernel")
                class Kernel:
                    in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                    in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                    out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                    def generate(g):
                        in_ptr0 = g.in_ptr0
                        in_ptr1 = g.in_ptr1
                        out_ptr0 = g.out_ptr0
                        xindex = hl.Var('xindex')
                        x0 = xindex
                        tmp0 = hl.Func()
                        tmp0[xindex] = in_ptr0[x0]
                        tmp1 = hl.Func()
                        tmp1[xindex] = in_ptr1[x0]
                        tmp2 = hl.Func()
                        tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                        out_ptr0[x0] = tmp2[xindex]

                        assert not g.using_autoscheduler()
                        i = hl.Var()
                        j = hl.Var()
                        out_ptr0.compute_root()
                        out_ptr0.split(xindex, i, j, 32)
                        out_ptr0.parallel(i)
                        out_ptr0.vectorize(j)
                        tmp2.compute_at(out_ptr0, i)
                        tmp2.store_at(out_ptr0, i)
                        tmp1.compute_inline()

                __name__ == '__main__' and hl.main()
                """,
            ),
        )
        a = torch.randn(1024)
        b = torch.randn(1024)
        c = torch.randn(1024)
        fn(a, b, c)
        self.assertEqual(c, a + b)

    @unittest.skipUnless(has_triton(), "requires triton")
    def test_random_consistency(self):
        # [FUNCTION DOC] Requires triton. For each of three CUDA random generators (uniform,
        # normal, random integer) compiles a function with the halide cuda backend and one with
        # the triton cuda backend, seeds the global RNG identically before each call and gets
        # both outputs.
        # KNOWN ISSUE: the final equality assertion sits after the loop, so only the last
        # generator (the integer one) is actually compared; the earlier two are computed but
        # never checked.
        seed = 1234
        shape = (3, 3)
        dtype = torch.float32

        for (rand_fn,) in itertools.product(
            (
                functools.partial(torch.rand, shape, dtype=dtype, device="cuda"),
                functools.partial(torch.randn, shape, dtype=dtype, device="cuda"),
                functools.partial(
                    torch.randint,
                    -1000,
                    1000,
                    size=shape,
                    dtype=torch.int64,
                    device="cuda",
                ),
            )
        ):

            @torch.compile(backend="inductor", options={"cuda_backend": "halide"})
            def get_rand_halide():
                # [FUNCTION DOC] HELPER: Compiled with inductor using the halide cuda backend;
                # returns the output of the current loop iteration's random generator.
                return rand_fn()

            @torch.compile(backend="inductor", options={"cuda_backend": "triton"})
            def get_rand_triton():
                # [FUNCTION DOC] HELPER: Compiled with inductor using the triton cuda backend;
                # returns the output of the current loop iteration's random generator.
                return rand_fn()

            torch.manual_seed(seed)
            halide_output = get_rand_halide()
            torch.manual_seed(seed)
            triton_output = get_rand_triton()

        self.assertEqual(halide_output, triton_output)

    def test_compile_options(self):
        # [FUNCTION DOC] Compiles a sum of two softmaxes with inductor using halide for cpu and
        # cuda and the Adams2019 (cpu) and Anderson2021 (cuda) schedulers. Runs it on CPU and,
        # if CUDA is available, on GPU, asserting the generated source contains a Halide
        # generator declaration.
        @torch.compile(
            backend="inductor",
            options={
                "cuda_backend": "halide",
                "cpu_backend": "halide",
                "halide.scheduler_cuda": "Anderson2021",
                "halide.scheduler_cpu": "Adams2019",
            },
        )
        def halide(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of the last-dimension
            # softmaxes of its two inputs.
            return torch.softmax(a, -1) + torch.softmax(b, -1)

        _, (code,) = run_and_get_code(
            halide, torch.randn(1024, 1024), torch.randn(1024, 1024)
        )
        self.assertIn("@hl.generator", code)

        if torch.cuda.is_available():
            _, (code,) = run_and_get_code(
                halide,
                torch.randn(1024, 1024, device="cuda"),
                torch.randn(1024, 1024, device="cuda"),
            )
            self.assertIn("@hl.generator", code)

    def test_inplace_add_broadcast_input_alias(self):
        # [FUNCTION DOC] Regression test: an in-place add on a transposed tensor with a
        # broadcast (size one dimension) second input, compiled with the halide cpu backend.
        # Asserts the returned value and the mutated input both equal the eager in-place result.
        @torch.compile(backend="inductor", options={"cpu_backend": "halide"})
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function that adds y into x in place and returns
            # x.
            return x.add_(y)

        x = torch.ones([2, 12, 13, 17]).transpose(1, 2)
        y = torch.ones([2, 13, 1, 17])
        expected = x.clone()
        expected.add_(y)

        result = fn(x, y)

        self.assertEqual(result, expected)
        self.assertEqual(x, expected)


if test_torchinductor.HAS_CPU and HAS_HALIDE:
    make_halide(test_torchinductor.SweepInputsCpuTest)
    make_halide(test_torchinductor.CpuTests)

if (
    test_torchinductor.HAS_GPU
    and HAS_HALIDE
    and os.environ.get("TEST_HALIDE_GPU") == "1"
):
    make_halide(test_torchinductor.SweepInputsGPUTest)
    make_halide(test_torchinductor.GPUTests)

if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS and HAS_HALIDE:
        run_tests(needs="filelock")
