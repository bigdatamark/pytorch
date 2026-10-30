# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor with dynamic shapes. Defines the xfail/skip table for the
# dynamic-shape variants of the shared CommonTemplate tests, a factory that clones a test
# class with assume_static_by_default turned off, CPU and GPU test classes filled with copies
# of those tests, and TestInductorDynamic, a device-generic class of regression tests for
# dynamic and unbacked symbols. Import-time side effects: appends the test directory to
# sys.path, requires filelock, edits the failure table for debug and ASAN builds, creates
# classes only when CPU or GPU support exists, wraps some generated tests with ROCm skips, and
# instantiates the device-type tests into module globals.
import contextlib
import importlib
import math
import operator
import os
import sys
import unittest
from functools import partial

import torch
import torch.library
from torch._dynamo.testing import CompileCounterWithBackend, make_test_cls_with_patches
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config, metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import IS_SM89
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyOn,
)
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_FBCODE,
    MI350_ARCH,
    parametrize,
    serialTest,
    skipIfRocmArch,
    skipIfRocmVersionAtLeast,
    TEST_CUDA_MEM_LEAK_CHECK,
    TEST_WITH_ASAN,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_MPS,
    patch_inductor_backend,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    CommonTemplate,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    "test_kwargs_dynamic_shapes": TestFailure(("cpu",)),
    # PDL tests are CUDA SM90+ only, skip on CPU
    "test_pdl_mutation_dynamic_shapes": TestFailure(("cpu",), is_skip=True),
    "test_pdl_template_and_delay_dynamic_shapes": TestFailure(("cpu",), is_skip=True),
    # With tensorify enabled, SymInt div no longer graph-breaks; the full
    # model compiles but Inductor hangs on the complex dynamic-shape graph.
    "test_AllenaiLongformerBase_repro_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),
    "test_argmax_argmin_with_duplicates_dynamic_shapes": TestFailure(("mps",)),
    "test_index_propagation_abs_dynamic_shapes": TestFailure(("mps",)),
    "test_index_propagation_floordiv_dynamic_shapes": TestFailure(("mps",)),
    "test_index_propagation_remainder_dynamic_shapes": TestFailure(("mps",)),
    # This fails on periodic CPU shards but passes on other CPU configurations,
    # so skip it instead of producing unexpected successes.
    "test_index_propagation_to_dtype_inf_dynamic_shapes": TestFailure(
        ("cpu",), is_skip=True
    ),
    "test_roll_dynamic_shapes": TestFailure(("mps",)),
    "test_reflection_pad2d_backward_dynamic_shapes": TestFailure(
        ("mps",), is_skip=True
    ),
}

if any(os.getenv("BUILD_ENVIRONMENT", "").endswith(x) for x in ("-debug", "-asan")):
    # Fails with TORCH_INTERNAL_ASSERT(!is_heap_allocated()), see https://github.com/pytorch/pytorch/issues/130073
    # After https://github.com/pytorch/pytorch/pull/161586, starts failing UBSAN so we can't even xfail.
    # Root cause seems to be SymInt issues in StorageImpl, see
    # https://github.com/pytorch/pytorch/pull/161586#issuecomment-3246530671
    test_failures["test_resize_as_dynamic_shapes"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )
    test_failures["test_resize_dynamic_shapes"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )


def make_dynamic_cls(cls, xfail_prop="_expected_failure_dynamic"):
    # [FUNCTION DOC] Returns a new test class built by make_test_cls_with_patches from cls,
    # named with a DynamicShapes prefix, whose copied test methods get the suffix
    # _dynamic_shapes and run with the dynamo config assume_static_by_default set to false.
    # Test methods carrying the attribute named by xfail_prop (default
    # _expected_failure_dynamic) are marked expected failures in the copy.
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        "_dynamic_shapes",
        (torch._dynamo.config, "assume_static_by_default", False),
        xfail_prop=xfail_prop,
    )


DynamicShapesCommonTemplate = make_dynamic_cls(CommonTemplate)


class DynamicShapesTestCase(TestCase):
    # [CLASS DOC] Base test case for dynamic-shape test classes. For the whole class it
    # enables the inductor test_configs flags for runtime Triton dtype and shape assertions,
    # using a config patch held open in an ExitStack stored on the class.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Runs the parent setUpClass, creates an ExitStack stored as
        # _triton_assert_stack and enters an inductor config patch turning on the runtime
        # Triton dtype assert and shape assert. Side effect: those global config values stay
        # patched until tearDownClass.
        super().setUpClass()
        cls._triton_assert_stack = contextlib.ExitStack()
        cls._triton_assert_stack.enter_context(
            config.patch(
                {
                    "test_configs.runtime_triton_dtype_assert": True,
                    "test_configs.runtime_triton_shape_assert": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Closes the ExitStack so the config patch is undone, then runs
        # the parent tearDownClass.
        cls._triton_assert_stack.close()
        super().tearDownClass()


class DynamicShapesOpTests:
    # [CLASS DOC] Mixin, not a test case itself, holding op-level tests shared by the CPU and
    # GPU dynamic-shape classes. The host class must provide common (a model checker such as
    # check_model) and a device attribute.
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_prod_backward_keeps_input_shape_dynamic(self):
        # [FUNCTION DOC] Dynamo assume_static_by_default is off. Runs a product reduction over
        # dim 3 (keeping the dimension) on an 8 by 10 by 3 by 2 input that requires grad,
        # through the host class's common checker with gradient checking on, and asserts that
        # all four dims of input 0 are symbolic in the graphs given to inductor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the product of x over dim 3 with the
            # reduced dimension kept.
            return torch.prod(x, 3, keepdim=True)

        self.common(
            fn,
            (torch.randn(8, 10, 3, 2, requires_grad=True),),
            check_gradient=True,
            assert_dynamic_dims={0: (0, 1, 2, 3)},
        )


if HAS_CPU:

    class DynamicShapesCpuTests(DynamicShapesOpTests, TestCase):
        # [CLASS DOC] CPU dynamic-shape test class: DynamicShapesOpTests plus TestCase, with
        # check_model as the common checker and device cpu. Copies of the common dynamic-shape
        # template tests are attached afterwards by copy_tests.
        common = check_model
        device = "cpu"

        @torch._dynamo.config.patch(assume_static_by_default=False)
        def test_assert_dynamic_dims_disables_aot_autograd_cache(self):
            # [FUNCTION DOC] Dynamo assume_static_by_default is off. Clears the counters and
            # the AOT autograd cache, runs the same product-backward case twice with
            # assert_dynamic_dims, then asserts the autograd cache hit and miss counters are
            # both zero, i.e. the cache is bypassed when dynamic dims are asserted.
            counters.clear()
            AOTAutogradCache.clear()

            def fn(x):
                # [FUNCTION DOC] HELPER: Returns the product of x over dim 3 with the
                # reduced dimension kept.
                return torch.prod(x, 3, keepdim=True)

            for _ in range(2):
                self.common(
                    fn,
                    (torch.randn(8, 10, 3, 2, requires_grad=True),),
                    check_gradient=True,
                    assert_dynamic_dims={0: (0, 1, 2, 3)},
                )

            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)

        def test_bincount_weighted_count_nonzero_dtype(self):
            # [FUNCTION DOC] Compiles a function that takes the weighted bincount (minlength
            # 6) of an int64 index tensor with float32 weights and also counts the nonzero
            # entries of the result, using fullgraph with dynamic false and then true (dynamo
            # reset in between). Both outputs must equal the eager results.
            def fn(x, weights):
                # [FUNCTION DOC] HELPER: Returns the weighted bincount and the number of
                # nonzero elements in it.
                counts = torch.bincount(x, weights=weights, minlength=6)
                return counts, counts.count_nonzero()

            x = torch.tensor(
                [0, 2, 2, 3, 1, 0, 3, 3],
                dtype=torch.int64,
            )
            weights = torch.tensor(
                [1.0, -2.0, 3.0, 0.0, 4.0, -5.0, 6.0, 7.0],
                dtype=torch.float32,
            )

            expected = fn(x, weights)
            for dynamic in [False, True]:
                torch._dynamo.reset()
                compiled_fn = torch.compile(
                    fn,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=dynamic,
                )
                actual = compiled_fn(x, weights)
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])

    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCpuTests, "cpu", test_failures)

    if hasattr(DynamicShapesCpuTests, "test_tmp_not_defined_issue3_dynamic_shapes_cpu"):
        # The vec-ISA probe child cannot resolve ROCm SDK libraries in CI.
        DynamicShapesCpuTests.test_tmp_not_defined_issue3_dynamic_shapes_cpu = (
            skipIfRocmVersionAtLeast([7, 14])(
                DynamicShapesCpuTests.test_tmp_not_defined_issue3_dynamic_shapes_cpu
            )
        )


if (HAS_GPU or HAS_MPS) and not TEST_WITH_ASAN:

    class DynamicShapesGPUTests(DynamicShapesOpTests, DynamicShapesTestCase):
        # [CLASS DOC] GPU dynamic-shape test class (also created for MPS, never under ASAN):
        # DynamicShapesOpTests plus DynamicShapesTestCase, with check_model_gpu as the common
        # checker and device set to the GPU type. Copies of the common template tests are
        # attached afterwards by copy_tests.
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        DynamicShapesCommonTemplate, DynamicShapesGPUTests, GPU_TYPE, test_failures
    )

    if HAS_GPU and hasattr(
        DynamicShapesGPUTests, "test_conv_with_as_strided_dynamic_shapes_cuda"
    ):
        # gfx950 shows a deterministic numerical mismatch for this generated test.
        DynamicShapesGPUTests.test_conv_with_as_strided_dynamic_shapes_cuda = (
            skipIfRocmArch(MI350_ARCH)(
                DynamicShapesGPUTests.test_conv_with_as_strided_dynamic_shapes_cuda
            )
        )

    if HAS_GPU and hasattr(
        DynamicShapesGPUTests, "test_randint_distribution_dynamic_shapes_cuda"
    ):
        # gfx950 shows a deterministic randint64 distribution mismatch for high bounds.
        DynamicShapesGPUTests.test_randint_distribution_dynamic_shapes_cuda = (
            skipIfRocmArch(MI350_ARCH)(
                DynamicShapesGPUTests.test_randint_distribution_dynamic_shapes_cuda
            )
        )


class TestInductorDynamic(DynamicShapesTestCase):
    # [CLASS DOC] Device-generic tests for dynamic and unbacked shapes in inductor. compile_fn
    # is torch.compile with dynamic enabled. Every test is skipped by setUp when no GPU with
    # Triton is available, even for CPU variants. setUp and tearDown reset dynamo and manage
    # an inductor config patch kept in an ExitStack. The class is expanded per device type at
    # the end of the module.
    compile_fn = partial(torch.compile, dynamic=True)

    def setUp(self):
        # [FUNCTION DOC] Skips the test when HAS_GPU is false (even for the CPU variant).
        # Otherwise resets dynamo, runs the parent setUp, and enters through an ExitStack
        # stored as _stack an inductor config patch: debug off, cpp min_chunk_size one, Triton
        # pointwise autotune off and implicit fallbacks off.
        # HAS_CUDA_AND_TRITON also checks compute capability to skip tests
        # on older devices
        if not HAS_GPU:
            self.skipTest("Triton not available")
        torch._dynamo.reset()
        super().setUp()
        # this should be in setUpClass, but device-generic tests
        # don't work with setUpClass well (non-deterministically the wrong setUpClass is resolved),
        # so put it in test setUp, it's cheap
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            torch._inductor.config.patch(
                {
                    "debug": False,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Closes the config-patch ExitStack, calls TestCase tearDown directly
        # (bypassing intermediate base classes) and resets dynamo.
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()

    def test_constant_fold_uniform_value_dynamic(self, device):
        # [FUNCTION DOC] For dynamic false and true, compiles each of three functions that
        # fold to returning their input (add of a zero tensor built from a full tensor,
        # multiply by one built from a full tensor, multiply by a broadcast ones tensor),
        # collects the generated source, checks no fused C++ kernel (cpu) or no Triton kernel
        # (otherwise) is present, and compares compiled and eager results on a 2 by 4 and a 3
        # by 4 input.
        # KNOWN ISSUE: The fourth helper, full_mul_symint, is never included in the tested
        # tuple, so the branch that exempts it from the no-kernel check is dead.
        def full_add_zero(x):
            # [FUNCTION DOC] HELPER: Builds a tensor of ones shaped like x, subtracts one to
            # get zeros and adds it to x.
            a = torch.full(x.shape, 1, dtype=x.dtype, device=x.device)
            b = a - 1
            return x + b

        def full_mul_one(x):
            # [FUNCTION DOC] HELPER: Builds a tensor filled with minus one shaped like x, adds
            # two to get ones and multiplies x by it.
            a = torch.full(x.shape, -1, dtype=x.dtype, device=x.device)
            b = 2 + a
            return x * b

        def full_view_op(x):
            # [FUNCTION DOC] HELPER: Makes a one-element ones tensor, adds a leading axis and
            # multiplies x by it (a broadcast of ones).
            a = torch.ones([1], dtype=x.dtype, device=x.device)
            a = a[:, None]
            return x * a

        def full_mul_symint(x):
            # [FUNCTION DOC] HELPER: Builds a minus-one filled tensor shaped like x, adds two
            # and multiplies by the symbolic first dimension size of x. Not used by the test.
            a = torch.full(x.shape, -1, dtype=x.dtype, device=x.device)
            b = 2 + a
            return b * x.shape[0]

        fns = (full_add_zero, full_mul_one, full_view_op)

        x = torch.randn((2, 4), device=device)
        y = torch.randn((3, 4), device=device)

        for dynamic in [False, True]:
            torch._dynamo.reset()
            for fn in fns:
                ref = fn(x)
                fn_c = torch.compile(fn, dynamic=dynamic)

                actual, source_codes = run_and_get_code(fn_c, x)

                if fn is not full_mul_symint:
                    # due to constant folding, fn returns x directly.
                    if device == "cpu":
                        FileCheck().check_not("cpp_fused").run(source_codes[0])
                    else:
                        FileCheck().check_not("triton.jit").run(source_codes[0])

                self.assertEqual(ref, actual)
                self.assertEqual(fn(x), fn_c(x))
                self.assertEqual(fn(y), fn_c(y))

    def test_constant_fold_uniform_value_self_referential_shape(self, device):
        # [FUNCTION DOC] Regression test described in its docstring. Sets the dynamo config
        # flags capture_scalar_outputs and capture_dynamic_output_shape_ops by direct
        # assignment, runs the function in eager with backward, then a dynamic fullgraph
        # compiled version with backward on a 401120 by 3 input and a scalar input, and
        # finally compares compiled with eager on fresh inputs that do not require grad.
        # KNOWN ISSUE: Config is assigned directly without a patch or restore, so the flags
        # leak into later tests.
        """
        Test that constant_fold_uniform_value correctly handles the case where
        a tensor's shape depends on a sym_size computed from the tensor itself.

        This is a regression test for a bug where creating a replacement full()
        node with a shape that includes a sym_size_int derived from the original
        tensor would create a circular dependency in the graph, causing
        stable_topological_sort to fail with an assertion error.
        """
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        def fn(arg0, arg1):
            # [FUNCTION DOC] HELPER: Clones the first argument, zeroes it and reshapes to
            # 109 by 115 by 96; takes relu of the contiguous second argument, clones the
            # reshaped tensor and fills it with that relu value as a Python scalar; returns
            # the filled tensor.
            t0 = arg0
            t1 = t0.clone()
            t1.zero_()
            t2 = t1.reshape((109, 115, 96))
            t3 = arg1
            t4 = t3.contiguous()
            t5 = torch.nn.functional.relu(t4)
            t6 = t2.clone()
            t6.fill_(t5.item())
            return t6

        arg0 = torch.rand(
            [401120, 3], dtype=torch.float32, device=device, requires_grad=True
        )
        arg1 = torch.rand([], dtype=torch.float32, device=device, requires_grad=True)

        # Test eager mode
        out_eager = fn(arg0, arg1)
        out_eager.sum().backward()

        # Reset grads
        arg0 = torch.rand(
            [401120, 3], dtype=torch.float32, device=device, requires_grad=True
        )
        arg1 = torch.rand([], dtype=torch.float32, device=device, requires_grad=True)

        # Test compiled mode - this would fail before the fix with:
        # AssertionError in stable_topological_sort due to circular dependency
        compiled_fn = torch.compile(fn, fullgraph=True, dynamic=True)
        out_compiled = compiled_fn(arg0, arg1)
        out_compiled.sum().backward()

        # Verify outputs match
        arg0_test = torch.rand(
            [401120, 3], dtype=torch.float32, device=device, requires_grad=False
        )
        arg1_test = torch.rand(
            [], dtype=torch.float32, device=device, requires_grad=False
        )
        self.assertEqual(fn(arg0_test, arg1_test), compiled_fn(arg0_test, arg1_test))

    def test_arange_dynamic(self, device):
        # [FUNCTION DOC] Compiles (dynamic) a mask builder over a ten-element integer tensor
        # whose maximum is forced to 29 and compares with eager.
        def fn(a):
            # [FUNCTION DOC] HELPER: Builds a range up to the maximum of a cast to a's dtype,
            # repeats it once per element, compares each row to the matching element of a, and
            # returns the logical negation.
            batch_size = a.numel()
            max_len = a.max()
            return ~(
                torch.arange(0, max_len, device=a.device)
                .type_as(a)
                .repeat(batch_size, 1)
                .lt(a.unsqueeze(1))
            )

        a = torch.randint(10, 30, (10,), device=device)
        a[0] = 29  # fix max_len
        opt = self.compile_fn(fn)
        res = opt(a)
        ref = fn(a)
        self.assertEqual(res, ref)

    def test_shape_as_constant_reciprocal_float_exp(self, device):
        # [FUNCTION DOC] Compiles (dynamic) a function of a tensor and its first dimension
        # size, as an integer argument, and compares the outputs with eager.
        def fn(x, a):
            # [FUNCTION DOC] HELPER: Returns x together with minus one divided by the integer
            # argument raised to a float power of one.
            return x, -1 / a**1.0

        x = torch.rand(10, 20, device=device)
        opt = self.compile_fn(fn)
        res = opt(x, x.size(0))
        ref = fn(x, x.size(0))
        self.assertEqual(res, ref)

    # not supported yet on cpu, https://github.com/pytorch/pytorch/issues/109897
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_bool_mask_nobreak(self, device):
        # [FUNCTION DOC] Dynamic-output-shape ops capture on. Compiles the masked sum with
        # fullgraph (so no graph break on the data-dependent size) and compares to eager on a
        # five-element input. The comment notes it is not yet supported on cpu.
        def f(x, b):
            # [FUNCTION DOC] HELPER: Selects elements of x with the boolean mask b, doubles
            # them and returns the sum.
            return (x[b] * 2).sum()

        opt_f = torch.compile(f, fullgraph=True)
        x = torch.randn(5, device=device)
        b = torch.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    def test_adaptive_max_pool3d_with_indices(self, device):
        # [FUNCTION DOC] Compiles (dynamic) adaptive 3D max pooling with indices, output
        # size five, on a random 9 by 10 by 9 by 8 by 6 tensor and compares both returned
        # tensors with eager.
        x = 5
        y = torch.rand([9, 10, 9, 8, 6], dtype=torch.float32, device=device)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls adaptive max pool 3d with indices with output size
            # x on input y, returning values and indices.
            return torch.nn.functional.adaptive_max_pool3d_with_indices(
                output_size=x, input=y, return_indices=True
            )

        opt_f = self.compile_fn(fn)
        r = fn(x, y)
        opt_r = opt_f(x, y)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unwrap_storage_didnt_work_repro(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile of a no-argument function
        # that makes a scalar tensor holding 11, converts it to a Python integer and returns a
        # zeros tensor of that length; compares with eager.
        def f():
            # [FUNCTION DOC] HELPER: Creates a scalar full tensor of eleven, reads it as a
            # Python int and returns an int-filled zeros tensor of that length.
            full = torch.full((), 11)
            i0 = full.item()
            return torch.full((i0,), 0)

        opt_f = torch.compile(f, fullgraph=True)
        r = f()
        opt_r = opt_f()
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_sum_unbacked(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile of a function that
        # converts a hundred random small integers to a list, adds them with the Python sum
        # and wraps the result in a tensor; compares with eager.
        def f(a):
            # [FUNCTION DOC] HELPER: Converts a to a Python list, sums it and returns a tensor
            # of the sum.
            xs = a.tolist()
            y = sum(xs)
            return torch.tensor(y)

        splits = torch.randint(10, (100,), device=device)

        opt_f = torch.compile(f, fullgraph=True)
        r = f(splits)
        opt_r = opt_f(splits)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_size_factory_nobreak(self, device):
        # [FUNCTION DOC] Dynamic-output-shape ops capture on. Fullgraph compile of a function
        # that uses the number of nonzero entries of a boolean tensor as the length of new
        # zeros; compares with eager.
        def f(x, b):
            # [FUNCTION DOC] HELPER: Returns a zeros tensor shaped by the count of true
            # entries of b, created like x.
            y = torch.nonzero(b)
            return x.new_zeros(y.size(0))

        opt_f = torch.compile(f, fullgraph=True)
        x = torch.randn(5, device=device)
        b = torch.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_no_realloc(self, device):
        # [FUNCTION DOC] Dynamic-output-shape ops capture on. Runs a dynamic fullgraph
        # function that splits a nonzero result into one chunk whose size comes from another
        # tensor. There is no value check; it passes if compile and run complete.
        @torch.compile(fullgraph=True, dynamic=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Takes the nonzero indices of x and splits them into
            # pieces sized by the first dimension of y.
            z = x.nonzero()
            return torch.split(z, [y.size(0)])

        f(torch.tensor([1, 0, 1, 1, 0, 1, 0]), torch.randn(4))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_nobreak(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile that reads a
        # one-element tensor as a Python int and builds an empty tensor of that length; runs
        # once with no output check.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Reads x as a Python int and returns an uninitialized
            # tensor with that length.
            y = x.item()
            return torch.empty(y)

        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_bool_nobreak(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile of a function returning
        # the item of a one-element boolean tensor; run once, no output check.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the Python value of the single element of x.
            return x.item()

        f(torch.tensor([True], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_noops_tensor_repropagate(self, device):
        # [FUNCTION DOC] Both scalar output and dynamic-output-shape capture on. Fullgraph
        # compile of a function that converts to int64, takes nonzero and doubles it; run once
        # on an int64 tensor of five elements with no output check.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Converts x to int64 with the prims conversion op, takes
            # nonzero indices and doubles them.
            b = torch.ops.prims.convert_element_type.default(x, torch.int64)
            r = b.nonzero()
            return r * 2

        f(torch.tensor([0, 4, 2, 0, 1], dtype=torch.int64, device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_zeros_nobreak(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile of a function that
        # reads an integer, makes an unused empty tensor of that size, and returns new zeros
        # of the same size from x; the comment says this avoids a no-op scheduler node. Run
        # once, no output check.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Reads x as an integer y, creates and discards an empty
            # tensor of length y and returns zeros of length y.
            y = x.item()
            torch.empty(y)
            # This will avoid a NopSchedulerNode
            return x.new_zeros(y)

        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_return(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile of a function reading
        # the same one-element tensor twice and returning the sum of the two values; run
        # once, no output check.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Reads x as a Python number twice and returns their sum.
            y = x.item()
            z = x.item()
            return y + z

        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_inf(self, device):
        # [FUNCTION DOC] Both capture flags on. Fullgraph compile of a function comparing the
        # item of a float tensor to positive infinity; run once.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns whether the single value of x
            # equals positive infinity.
            return x.item() == math.inf

        f(torch.tensor([3.0], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_neginf(self, device):
        # [FUNCTION DOC] Both capture flags on. Fullgraph compile of a function comparing the
        # item of a float tensor to negative infinity; run once.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns whether the single value of x
            # equals negative infinity.
            return x.item() == -math.inf

        f(torch.tensor([3.0], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_item_to_inputs_kernel_nobreak(self, device):
        # [FUNCTION DOC] Scalar output capture on and implicit fallbacks on. Registers a
        # custom op in the test_inductor_dynamic_shapes namespace taking a tensor and an int
        # and a matching fake implementation, then fullgraph compiles a function that reads an
        # int from one tensor and passes it to the op together with another tensor; run once,
        # no output check. Side effect: registers a library op in the process.
        @torch.library.custom_op(
            "test_inductor_dynamic_shapes::nobreak_test", mutates_args=()
        )
        def nobreak_test(x: torch.Tensor, y: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation: returns a clone of x
            # and ignores y.
            return x.clone()

        @nobreak_test.register_fake
        def _(x: torch.Tensor, y: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake (meta) implementation of the custom op: returns a
            # clone of x.
            return x.clone()

        @torch.compile(fullgraph=True)
        def f(x, r):
            # [FUNCTION DOC] HELPER: Reads x as an integer and calls the registered custom op
            # on r and that integer.
            y = x.item()
            return torch.ops.test_inductor_dynamic_shapes.nobreak_test(r, y)

        f(torch.tensor([3], device=device), torch.randn(10, device=device))

    @unittest.skipUnless(IS_FBCODE, "")
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_return(self, device):
        # [FUNCTION DOC] Runs only on FB code builds. Both capture flags on. Fullgraph compile
        # of a function returning the item of a float tensor; run once.
        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the Python float of the single element of x.
            return x.item()

        f(torch.tensor([3.0], device=device))

    @unittest.skipIf(TEST_CUDA_MEM_LEAK_CHECK, "failing memory leak check")
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_index_select(self, device):
        # [FUNCTION DOC] Skipped when the CUDA memory leak check is on. Scalar output capture
        # on. Fullgraph compile of a function that reads an int and does index_select of three
        # indices from a ones tensor of that length, covering unbacked symbols used by an
        # inner function; compares with eager.
        # Tests if unbacked symbols captured by inner_fn are properly tracked
        def f(x):
            # [FUNCTION DOC] HELPER: Reads x as an int y and selects elements 0, 2, 1 from a
            # ones tensor with y elements.
            y = x.item()
            return torch.index_select(
                torch.ones(y, device=device), 0, torch.tensor([0, 2, 1], device=device)
            )

        cf = torch.compile(fullgraph=True)(f)
        arg = torch.tensor(5, device=device)
        self.assertEqual(f(arg), cf(arg))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_return_unbacked_view_split(self, device):
        # [FUNCTION DOC] Both capture flags on. Fullgraph compile of a function that converts
        # a two-element tensor to a list and splits a values tensor into two pieces of those
        # sizes (returned as outputs, values require grad); compares both outputs with eager.
        def f(values, length_per_key):
            # [FUNCTION DOC] HELPER: Splits values into two pieces whose sizes come from the
            # list form of length_per_key and returns both.
            u0, u1 = length_per_key.tolist()
            v1, v2 = torch.functional.split(values, [u0, u1])
            return v1, v2

        cf = torch.compile(fullgraph=True)(f)
        args = (
            torch.randn(8, requires_grad=True, device=device),
            torch.tensor([3, 5], device=device),
        )
        self.assertEqual(f(*args), cf(*args))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_matmul(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile of a function that reads
        # an int y and multiplies a one by y ones matrix with a y by one ones matrix;
        # compares with eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Reads x as int y and returns the matrix product of a 1 by
            # y ones tensor with a y by 1 ones tensor.
            y = x.item()
            return torch.ones(1, y, device=device) @ torch.ones(y, 1, device=device)

        cf = torch.compile(fullgraph=True)(f)
        arg = torch.tensor(5, device=device)
        self.assertEqual(f(arg), cf(arg))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_unbacked_save_for_backwards(self, device) -> None:
        # [FUNCTION DOC] Both capture flags on, implicit fallbacks on. Registers a custom op
        # taking a tensor and a list of ints, with a fake implementation, and registers
        # autograd for it. Compiles dynamic fullgraph a function calling it with sizes from a
        # CPU tensor and multiplying the result by the input, then compares the gradient of
        # the scalar input from compiled backward with eager. Side effect: registers library
        # ops in the process.
        @torch.library.custom_op("_test::_cat", mutates_args=())
        def _cat(t: torch.Tensor, ds: list[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation: multiplies t by a ones tensor
            # whose length is the sum of ds.
            return t * t.new_ones([sum(ds)])

        @torch.library.register_fake("_test::_cat")
        def _cat_fake(t: torch.Tensor, ds: list[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation: returns an uninitialized tensor of
            # length equal to the sum of ds.
            return t.new_empty([sum(ds)])

        def _cat_setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup hook that saves nothing.
            pass

        def _cat_backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward: returns the sum of the incoming gradient for t
            # and no gradient for ds.
            return grad.sum(), None

        torch.library.register_autograd(
            "_test::_cat",
            _cat_backward,
            setup_context=_cat_setup_context,
        )

        def fn(t, sizes):
            # [FUNCTION DOC] HELPER: Calls the custom op with the list form of sizes and
            # multiplies the result by t.
            r = torch.ops._test._cat(t, sizes.tolist())
            return r * t

        t = torch.randn((), requires_grad=True, device=device)
        sizes = torch.tensor([4, 8], dtype=torch.int64, device="cpu")
        out = fn(t, sizes)
        out.sum().backward()
        expect = t.grad
        t.grad = None
        torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)(
            t, sizes
        ).sum().backward()
        self.assertEqual(t.grad, expect)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_reduction(self, device):
        # [FUNCTION DOC] Scalar output capture on. A fullgraph compile of a function that
        # reads an int and sums a ones tensor of that length is expected to fail only on
        # CPU, non-ARM64 machines without the C++ wrapper. Any exception is swallowed
        # when failure is expected and re-raised otherwise; a pass when failure was
        # expected calls self.fail.
        expect_fail = (
            device == "cpu" and not IS_ARM64 and not torch._inductor.config.cpp_wrapper
        )
        try:

            def f(x):
                # [FUNCTION DOC] HELPER: Reads x as an int and returns the sum of a ones
                # tensor with that length.
                y = x.item()
                return torch.ones(y, device=device).sum()

            cf = torch.compile(fullgraph=True)(f)
            arg = torch.tensor(5, device=device)
            self.assertEqual(f(arg), cf(arg))
        except Exception:
            if not expect_fail:
                raise
        else:
            if expect_fail:
                self.fail("expected to fail, but actually passed")

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_cat_unbacked_duplicate_size(self, device):
        # [FUNCTION DOC] Both capture flags on. Fullgraph compile of a function that converts
        # a two-element tensor to two sizes, builds zeros and ones tensors, and concatenates
        # the first twice with the second with the aten cat op; compares with eager.
        # KNOWN ISSUE: The input tensor is created on the GPU type rather than the device
        # argument, so CPU variants use a GPU tensor.
        def f(x):
            # [FUNCTION DOC] HELPER: Converts x to two list sizes, makes zeros of the first
            # and ones of the second length, and concatenates the first, the first again
            # and the second.
            device = x.device
            s, s2 = x.tolist()
            g = torch.zeros(s, device=device)
            g2 = torch.ones(s2, device=device)
            return torch.ops.aten.cat.default([g, g, g2])

        cf = torch.compile(fullgraph=True)(f)
        arg = torch.tensor([4, 6], device=GPU_TYPE)
        self.assertEqual(f(arg), cf(arg))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_cat_backwards(self, device):
        # [FUNCTION DOC] Both capture flags on. Computes eager backward through a function
        # that builds two ones tensors with sizes from a list, multiplies by a
        # gradient-requiring weight, concatenates and sums; then runs the fullgraph-compiled
        # version's backward and compares the weight gradient.
        def f(x, w):
            # [FUNCTION DOC] HELPER: Converts x to sizes a and b, builds ones tensors of those
            # lengths, scales both by w, concatenates and returns the sum.
            device = w.device
            a, b = x.tolist()
            ta = torch.ones(a, device=device)
            tb = torch.ones(b, device=device)
            pa = ta * w  # make it require gradients
            pb = tb * w
            r = torch.cat([pa, pb])
            return r.sum()

        x = torch.tensor([4, 9])
        w = torch.randn(1, requires_grad=True)
        f(x, w).backward()
        orig_w = w.grad
        w.grad = None

        torch.compile(fullgraph=True)(f)(x, w).backward()
        self.assertEqual(orig_w, w.grad)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_cat_backwards_save_data_dependent(self, device):
        # [FUNCTION DOC] Both capture flags on. Like the previous test but the function
        # returns the concatenated tensor and the sum is taken by the caller before backward;
        # compares the weight gradients of eager and compiled runs.
        def f(x, w):
            # [FUNCTION DOC] HELPER: Converts x to sizes a and b, builds ones tensors of those
            # lengths, scales both by w and returns their concatenation.
            device = w.device
            a, b = x.tolist()
            ta = torch.ones(a, device=device)
            tb = torch.ones(b, device=device)
            pa = ta * w  # make it require gradients
            pb = tb * w
            r = torch.cat([pa, pb])
            return r

        x = torch.tensor([4, 9])
        w = torch.randn(1, requires_grad=True)
        f(x, w).sum().backward()
        orig_w = w.grad
        w.grad = None

        torch.compile(fullgraph=True)(f)(x, w).sum().backward()
        self.assertEqual(orig_w, w.grad)

    @onlyOn(GPU_TYPE)
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_embedding_backward_dynamic_shapes_large_grid(self, device):
        # [FUNCTION DOC] GPU only. Both capture flags on and emulate_precision_casts on,
        # though nothing is compiled. Calls the grid-limit helper with a size hint of 600
        # million and an x block of 64 using the warp size of the device, then checks: on
        # HIP the total thread count of the returned block count must not exceed the 32-bit
        # grid maximum; otherwise the x block must remain 64 and the block count must not
        # exceed that maximum.
        """Test _check_max_grid_x correctly applies platform-specific grid limits.

        On CUDA: uses num_blocks only (not num_blocks * num_warps * warp_size).
        On ROCm: uses num_blocks * num_warps * warp_size (total threads limit).
        """
        from torch._inductor.runtime.hints import get_warp_size
        from torch._inductor.runtime.triton_heuristics import (
            _check_max_grid_x,
            _num_warps,
        )

        size_hints = {"x": 600_000_000}
        x = 64
        warp_size = get_warp_size(torch.device(device))
        num_warps = _num_warps(8, warp_size=warp_size)

        result_x, result_num_blocks = _check_max_grid_x(
            size_hints, x, num_warps, warp_size=warp_size
        )

        max_grid_x = 2147483647
        if torch.version.hip:
            # ROCm limits total threads (num_blocks * num_warps * warp_size)
            self.assertLessEqual(
                result_num_blocks * num_warps * warp_size,
                max_grid_x,
                "ROCm total-threads grid limit should be satisfied",
            )
        else:
            # CUDA limits number of blocks only — 600M/64 ≈ 9.4M blocks,
            # well within 2^31-1, so no scaling should occur
            self.assertEqual(
                result_x,
                64,
                lambda msg: f"{msg}\nXBLOCK should remain 64 (got {result_x})",
            )
            self.assertLessEqual(result_num_blocks, max_grid_x)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_dynamic_stride_nobreak(self, device):
        # [FUNCTION DOC] Both capture flags on, implicit fallbacks on. Registers a custom op
        # returning a one-element strided tensor whose stride comes from the input value, with
        # a fake implementation using a new dynamic size, then fullgraph compiles a function
        # using the result's stride as an allocation size. Run once, no output check. Side
        # effect: registers a library op.
        @torch.library.custom_op("test_dynamic_stride_nobreak::foo", mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation: reads x as the stride and
            # returns an empty one-element tensor with that stride.
            stride = x.item()
            return torch.empty_strided((1,), (stride,), device=x.device)

        @foo.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation: gets the library context, creates a
            # new dynamic size and returns an empty one-element tensor with it as stride.
            ctx = torch.library.get_ctx()
            stride = ctx.new_dynamic_size()
            return torch.empty_strided((1,), (stride,), device=x.device)

        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls the registered op on x and returns an uninitialized
            # tensor whose length is the first stride of the result.
            r = torch.ops.test_dynamic_stride_nobreak.foo(x)
            y = r.stride(0)
            return torch.empty(y, device=x.device)

        f(torch.tensor([3], device=device))

    @unittest.skipIf(
        IS_SM89,
        "Fails(with OOMS) on SM89, see https://github.com/pytorch/pytorch/issues/141915",
    )
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_multi_output_unbacked_custom_op(self, device):
        # [FUNCTION DOC] Skipped on SM89 (out-of-memory). Both capture flags on, implicit
        # fallbacks on. Registers a custom op that returns a pair of tensors with lengths two
        # and three and a fake implementation whose first length is a new dynamic size, then
        # fullgraph compiles a function summing both outputs. Run once, no output check. Side
        # effect: registers a library op.
        @torch.library.custom_op(
            "test_inductor_dynamic_shapes::unbacked_test", mutates_args=()
        )
        def unbacked_test(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Custom op implementation: returns two uninitialized
            # tensors, of length two and three.
            return torch.empty(2, device=x.device), torch.empty(3, device=x.device)

        @unbacked_test.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation: returns uninitialized tensors of
            # length equal to a new dynamic size and of length three.
            ctx = torch.library.get_ctx()
            u0 = ctx.new_dynamic_size()
            return torch.empty(u0, device=x.device), torch.empty(3, device=x.device)

        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls the op and returns the sum of the sums
            # of both outputs.
            a, b = torch.ops.test_inductor_dynamic_shapes.unbacked_test(x)
            return a.sum() + b.sum()

        f(torch.tensor([3], device=device))

    def test_meta_dynamic_shapes(self):
        # [FUNCTION DOC] Compiles with dynamic enabled a function multiplying two inputs by 2
        # and 3. Called with a one by 16 by 128 by 128 meta tensor and a random 64-element CPU
        # tensor; compares with eager. Takes no device argument.
        def foobar(x, y):
            # [FUNCTION DOC] HELPER: Returns x times two and y times three.
            return x * 2, y * 3

        foo_c = torch.compile(dynamic=True)(foobar)
        t = torch.empty((1, 16, 128, 128), device="meta")
        y = torch.rand([64])

        self.assertEqual(foo_c(t, y), foobar(t, y))

    def test_floor(self):
        # [FUNCTION DOC] Compiles a function that adds an integer computed from the last
        # dimension times 0.2 plus one. First runs with size five (static), then eight
        # (triggering dynamic shapes), comparing with eager each time.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds to x the truncated integer of 0.2 times its last
            # dimension size, plus one.
            n = x.size(-1)
            y = x + int(n * 0.2) + 1
            return y

        opt = self.compile_fn(fn)
        # The first run doesn't trigger dynamic shapes.
        x0 = torch.rand(5)
        ref0 = fn(x0)
        res0 = opt(x0)
        self.assertEqual(ref0, res0)
        # The second run triggers dynamic shapes.
        x1 = torch.rand(8)
        ref1 = fn(x1)
        res1 = opt(x1)
        self.assertEqual(ref1, res1)

    @onlyOn(GPU_TYPE)
    def test_pad_dynamic(self, device):
        # [FUNCTION DOC] GPU only. Compiles with dynamic enabled a function that applies
        # TF-style same padding to the last two dimensions of a 2 by 24 by 110 by 110 input
        # with kernel 5 and stride 2, and requires exact equality with eager.
        def get_same_padding(x: int, k: int, s: int, d: int):
            # [FUNCTION DOC] HELPER: Returns the padding amount needed along one dimension:
            # the larger of zero and the quantity for output size ceil of x divided by s minus
            # one, times s, plus (k minus one) times d plus one minus x.
            return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

        def pad_same(x, k, s, d=(1, 1), value=0):
            # [FUNCTION DOC] HELPER: Pads the last two dimensions of x so that output size
            # matches ceil of input over stride, splitting each padding amount between both
            # sides (smaller part first) with the given fill value; returns x unchanged if no
            # padding is needed.
            ih, iw = x.size()[-2:]
            pad_h, pad_w = (
                get_same_padding(ih, k[0], s[0], d[0]),
                get_same_padding(iw, k[1], s[1], d[1]),
            )
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(
                    x,
                    [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                    value=value,
                )
            return x

        x = torch.randn(2, 24, 110, 110, device=device)
        opt = self.compile_fn(pad_same)
        res = opt(x, (5, 5), (2, 2))
        ref = pad_same(x, (5, 5), (2, 2))
        self.assertEqual(res, ref, atol=0, rtol=0)

    def test_slice_scatter(self, device):
        # [FUNCTION DOC] Compiles a function building ones tensors sized by the input length
        # and half of it, scattering the smaller into the larger along dim 1 from half to
        # twice half; compares with eager for a 16-element input.
        def fn(i):
            # [FUNCTION DOC] HELPER: Takes s3 as the size of i, makes a 64 by s3 ones
            # tensor and 64 by half of s3 ones tensor, and uses slice scatter on dim 1 over
            # the middle range.
            s3 = i.size(0)
            x = torch.ones(64, s3, device=device)
            y = torch.ones(64, s3 // 2, device=device)
            return torch.slice_scatter(x, y, 1, s3 // 2, 2 * (s3 // 2))

        a = torch.randn(16, device=device)
        cfn = self.compile_fn(fn)
        expect = fn(a)
        actual = cfn(a)
        self.assertEqual(expect, actual)

    def test_slice_index_changing_sign(self, device):
        # [FUNCTION DOC] Compiles one function and runs it with the slice bound
        # difference positive (16 by 2 second input), then negative (2 by 16), comparing
        # with eager both times.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a clone of the leading rows of x, with row count
            # equal to the first dimension of y minus the second.
            y0, y1 = y.shape
            return x[: (y0 - y1)].clone()

        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)

        # y0 > y1 -> y0 - y1 is positive
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

        # y0 < y1 -> y0 - y1 is negative
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    def test_sym_stride_lowering(self, device):
        # [FUNCTION DOC] Compiles (dynamic) a function that multiplies the input by
        # the symbolic first stride of the input plus one; compares with eager on a 32
        # by 32 tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Gets the stride of dim 0 of x plus one and multiplies
            # x by it.
            s0 = (x + 1).stride(0)
            return x * s0

        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)
        self.assertEqual(fn(a), cfn(a))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_materialize(self, device):
        # [FUNCTION DOC] Scalar output capture on. Fullgraph compile of a function summing an
        # int64 3 by 4 tensor over dim 0 and returning the four values as a Python list;
        # compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sums x over dim 0, views as four elements and converts to
            # a Python list.
            return x.sum(dim=0).view(4).tolist()

        cfn = torch.compile(fullgraph=True)(fn)

        a = torch.ones(3, 4, dtype=torch.int64, device=device)
        self.assertEqual(cfn(a), fn(a))

    def test_abs(self, device):
        # [FUNCTION DOC] Compiles one function that slices by and multiplies with the absolute
        # difference of the two dimensions of y; runs with positive and negative differences
        # and compares with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the leading rows of x, as many as the absolute
            # difference of the dims of y, multiplied by that same difference.
            y0, y1 = y.shape
            # Slicing checks abs in wrapper code,
            # multiplication tests abs in kernel code
            return x[: abs(y0 - y1)] * abs(y0 - y1)

        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)

        # y0 > y1 -> y0 - y1 is positive
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

        # y0 < y1 -> y0 - y1 is negative
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    def test_float_is_integer(self, device):
        # [FUNCTION DOC] Compiles a function that divides a size by an int and branches on
        # whether the float result is an integer; compares with eager on a tensor whose last
        # dim is two.
        def fn(x, mul, dim=-1):
            # [FUNCTION DOC] HELPER: Divides the size of x along dim by mul; returns that
            # float if it is integral, otherwise the integer size.
            size = x.size(dim)
            m = size / mul
            if m.is_integer():
                return m
            return size

        a = torch.randn((3, 6, 4, 2), device=device)
        cfn = self.compile_fn(fn)

        expect = fn(a, 2)
        actual = cfn(a, 2)
        self.assertEqual(expect, actual)

    def test_magic_method_lowerings_with_symbolic_scalars(self, device):
        # [FUNCTION DOC] Defines seven functions using symbolic scalar arithmetic on the first
        # dimension (modulo, floor division, shifts, conditional select, min and max, a
        # log-based power of two, a numel based float constant), compiles each with fullgraph
        # in its own subtest, and compares with eager on a 7 by 5 input.
        def mod(x):
            # [FUNCTION DOC] HELPER: Returns a ones tensor whose length is the first dimension
            # modulo three plus one.
            return x.new_ones((x.shape[0] % 3) + 1)

        def floordiv(x):
            # [FUNCTION DOC] HELPER: Returns a ones tensor whose length is the first dimension
            # floor-divided by three plus one.
            return x.new_ones((x.shape[0] // 3) + 1)

        def shifts(x):
            # [FUNCTION DOC] HELPER: Returns two ones tensors with lengths from the first
            # dimension shifted left by one and shifted right by one, each plus one.
            return (
                x.new_ones((x.shape[0] << 1) + 1),
                x.new_ones((x.shape[0] >> 1) + 1),
            )

        def comparison(x):
            # [FUNCTION DOC] HELPER: Chooses the first dimension if it is below ten, else that
            # plus one, via a symbolic conditional, and returns a ones tensor of that length.
            n = torch.sym_ite(x.shape[0] < 10, x.shape[0], x.shape[0] + 1)
            return x.new_ones(n)

        def sym_min_max(x):
            # [FUNCTION DOC] HELPER: Returns two ones tensors with lengths from symbolic
            # minimum and maximum of the first dimension and three, each plus one.
            return (
                x.new_ones(torch.sym_min(x.shape[0], 3) + 1),
                x.new_ones(torch.sym_max(x.shape[0], 3) + 1),
            )

        def pow_log2(x):
            # [FUNCTION DOC] HELPER: Adds to x two raised to the floor of log base two of the
            # first dimension plus one.
            return x + 2 ** (math.floor(math.log2(x.shape[0]) + 1))

        def float_constant(x):
            # [FUNCTION DOC] HELPER: Adds to x the constant one, computed as the number of
            # elements raised to power zero divided by one.
            return x + ((x.numel() ** 0) / 1.0)

        x = torch.randn(7, 5, device=device)
        for fn in (
            mod,
            floordiv,
            shifts,
            comparison,
            sym_min_max,
            pow_log2,
            float_constant,
        ):
            with self.subTest(fn=fn.__name__):
                cfn = self.compile_fn(fn, fullgraph=True)
                self.assertEqual(fn(x), cfn(x))

    @onlyCPU
    def test_arithmetic_constant_folding(self, device):
        # [FUNCTION DOC] CPU only. Compiles three functions combining an
        # integer-argument input with a zeros or ones tensor of size three (add,
        # multiply, divide) and compares compiled and eager on the integer three, for
        # constant folding of arithmetic identities.
        def test(fn):
            # [FUNCTION DOC] HELPER: Compiles the function with compile_fn, evaluates eager
            # and compiled with integer three and asserts equal.
            cfn = self.compile_fn(fn)
            expect = fn(3)
            actual = cfn(3)
            self.assertEqual(expect, actual)

        def add(x):
            # [FUNCTION DOC] HELPER: Adds a zeros tensor of three elements to x.
            return x + torch.zeros(3)

        test(add)

        def mul(x):
            # [FUNCTION DOC] HELPER: Multiplies x by a ones tensor of three elements.
            return x * torch.ones(3)

        test(mul)

        def div(x):
            # [FUNCTION DOC] HELPER: Divides x by a ones tensor of three elements.
            return x / torch.ones(3)

        test(div)

    @onlyCPU
    def test_sub_constant_folding(self, device):
        # [FUNCTION DOC] CPU only. Compiles a function subtracting a three-element zeros
        # tensor from an integer argument and compares compiled and eager on the integer
        # three, for constant folding.
        def sub(x):
            # [FUNCTION DOC] HELPER: Subtracts a zeros tensor of three elements from x.
            return x - torch.zeros(3)

        cfn = self.compile_fn(sub)
        expect = sub(3)
        actual = cfn(3)
        self.assertEqual(expect, actual)

    def test_full_symbolic_value(self, device):
        # [FUNCTION DOC] Compiles a function that makes a three-element full tensor from an
        # integer argument and another from its float conversion; compares compiled and
        # eager for five.
        def fn(a):
            # [FUNCTION DOC] HELPER: Returns two three-element tensors filled with a and with
            # the symbolic float of a.
            return torch.full((3,), a), torch.full((3,), torch.sym_float(a))

        cfn = self.compile_fn(fn)
        expect = fn(5)
        actual = cfn(5)
        self.assertEqual(expect, actual)

    def test_interpolate_ceil_eq(self, device):
        # [FUNCTION DOC] Compiles a function whose output shape uses ceiling of true division
        # of symbolic expressions, building a bfloat16 zeros tensor and upsampling by two with
        # nearest mode; compares with eager on a 4 by 16 by 18 CPU input.
        ceiling = math.ceil
        IntTrueDiv = operator.truediv

        def fn(t):
            # [FUNCTION DOC] HELPER: Reads three sizes of t, makes a zeros bfloat16
            # tensor shaped from them with ceiling expressions and returns its
            # two-times nearest upsample.
            s0, s2, s3 = t.size()
            x = torch.zeros(
                (
                    s0,
                    2048,
                    ceiling(IntTrueDiv(2 * ((s2 - 1) // 8) + 2, 1)),
                    ceiling(IntTrueDiv(2 * ((s3 - 1) // 8) + 2, 1)),
                ),
                dtype=torch.bfloat16,
            )
            return torch.nn.functional.interpolate(
                x,
                scale_factor=2,
                mode="nearest",
            )

        cfn = self.compile_fn(fn)
        arg = torch.randn(4, 16, 18)
        expect = fn(arg)
        actual = cfn(arg)
        self.assertEqual(expect, actual)

    def test_full_recompiles(self, device):
        # [FUNCTION DOC] Compiles a function creating a full float16-minimum square tensor
        # sized by the second input dimension; runs on shapes 2 by 3 then 2 by 4 expecting no
        # recompilation, and asserts the compile counter for frame 0 equals one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes L as the second dimension of x and returns an L by
            # L tensor filled with the float16 minimum.
            _, L = x.shape
            return torch.full((L, L), torch.finfo(torch.float16).min, device=device)

        cfn = self.compile_fn(fn)

        import functools

        input_fn = functools.partial(torch.randint, 10, 1000, device=device)

        cfn(input_fn((2, 3)))
        cfn(input_fn((2, 4)))  # expect don't recompile here

        # check compiled times of frame 0
        from torch._dynamo.convert_frame import FRAME_COMPILE_COUNTER

        self.assertEqual(FRAME_COMPILE_COUNTER[0], 1)

    @parametrize(
        "op",
        [
            math.sqrt,
            math.sin,
            math.cos,
            math.cosh,
            math.sin,
            math.sinh,
            math.tan,
            math.tanh,
            math.asin,
            math.acos,
            math.atan,
        ],
    )
    def test_math_ops(self, device, op):
        # [FUNCTION DOC] Parametrized over math functions such as square root, trig and
        # inverse trig (sin appears twice in the list). Fullgraph compiles a function adding
        # the math function of a Python number argument to a tensor, with argument minus one
        # for asin and acos and twelve otherwise; compares with eager.
        def func(x, fn, a):
            # [FUNCTION DOC] HELPER: Returns x plus the result of fn applied to a.
            return x + fn(a)

        cfunc = self.compile_fn(func, fullgraph=True)
        x = torch.rand(10, device=device)
        a = -1 if op in (math.asin, math.acos) else 12
        expected = func(x, op, a)
        output = cfunc(x, op, a)
        self.assertEqual(output, expected)

    @serialTest()
    def test_wrapper_codegen_statically_known_int_or_none(self):
        # [FUNCTION DOC] Serial test. Resets dynamo, marks dim 0 of a 5 by 3 by 3 tensor
        # maybe-dynamic, defines three functions with different constraints on that dim, and
        # compiles each with a patched CPU Python wrapper codegen subclass which, at generate
        # time, checks whether the batch dim is statically known: it must not be on the first
        # call and must equal five on the second.
        # KNOWN ISSUE: The third branch tests call count two a second time instead of three,
        # so the check for the equality-constrained function never runs.
        torch._dynamo.reset()

        _x = torch.randn([5, 3, 3])
        torch._dynamo.maybe_mark_dynamic(_x, 0)

        # Simple functions introducing constraints on x.shape[0]
        def fn_1(x):
            # [FUNCTION DOC] HELPER: No constraint: returns the sine of x.
            # no constraint
            return x.sin()

        def fn_2(x):
            # [FUNCTION DOC] HELPER: Constrains dim 0 from both sides: cosine if above five,
            # doubled if below five, otherwise sine.
            # constrain in two directions
            if x.shape[0] > 5:
                return x.cos()
            if x.shape[0] < 5:
                return x * 2
            # x.shape[0] == 5 at this point
            return x.sin()

        def fn_3(x):
            # [FUNCTION DOC] HELPER: Branches on dim 0 equal to five: sine if so, else cosine.
            # equality constraint, which matches example shape
            if x.size(0) == 5:
                return x.sin()
            else:
                return x.cos()

        call_count = 0

        def _test_wrapper_codegen_statically_known_int_or_none_in_context():
            # [FUNCTION DOC] HELPER: Hook called from the codegen subclass. Increments the
            # shared call counter, reads the layout of the first graph input that has one,
            # and checks the statically known value of its first size: it must be None on
            # call one and five on call two; otherwise raises AssertionError. The third
            # branch is unreachable.
            nonlocal call_count
            call_count += 1
            graph = V.graph
            input_layouts = [
                inp.layout
                for inp in graph.graph_inputs.values()
                if hasattr(inp, "layout")
            ]
            batch_dim = input_layouts[0].size[0]
            if call_count == 1:
                # testing fn_1
                if (
                    PythonWrapperCodegen.statically_known_int_or_none(batch_dim)
                    is not None
                ):
                    raise AssertionError("Should not be statically known on first call")
            elif call_count == 2:
                # testing fn_2
                if PythonWrapperCodegen.statically_known_int_or_none(batch_dim) != 5:
                    raise AssertionError(
                        "Should be limited to exactly 5 on second call due to multiple constraints"
                    )
            elif call_count == 2:
                # testing fn_3
                if PythonWrapperCodegen.statically_known_int_or_none(batch_dim) != 5:
                    raise AssertionError("Should be exactly 5 on third call")

        class TestWrapperCodegen(PythonWrapperCodegen):
            # [CLASS DOC] HELPER: Python wrapper codegen subclass whose generate first runs
            # the checking hook, then defers to the parent.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the
                # parent initializer unchanged.
                super().__init__(*args, **kwargs)

            def generate(self, is_inference, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Runs the checking hook, then returns the parent
                # generate result for the same arguments.
                _test_wrapper_codegen_statically_known_int_or_none_in_context()
                return super().generate(is_inference, *args, **kwargs)

        with patch_inductor_backend("cpu", python_wrapper_codegen=TestWrapperCodegen):
            # Compile each of the functions above, with an example input
            # that has 5 in the first dimension, but is marked as dynamic

            torch.compile(backend="inductor", dynamic=None)(fn_1)(_x)
            torch.compile(backend="inductor", dynamic=None)(fn_2)(_x)
            torch.compile(backend="inductor", dynamic=None)(fn_3)(_x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_unbacked_stride_nobreak(self, device):
        # [FUNCTION DOC] Scalar output capture on. Dynamic fullgraph compile of a function
        # that reads an int, asserts bounds one to ten with torch check and returns a square
        # ones tensor of that size; run once.
        @torch.compile(fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Reads x as int a, checks a is between one and ten and
            # returns an a by a ones tensor.
            a = x.item()
            torch._check(a >= 1)
            torch._check(a <= 10)
            return torch.ones(a, a)

        f(torch.tensor([5], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symint_sum_list(self, device):
        # [FUNCTION DOC] Scalar output capture on. Compiles (no fullgraph) a function that
        # converts a tensor of 320 fives to a list, sums it and returns zeros of that length;
        # run once, no value check.
        @torch.compile()
        def f(xt):
            # [FUNCTION DOC] HELPER: Converts xt to a list, sums it and returns a zeros tensor
            # of that length on device.
            xs = xt.tolist()
            y = sum(xs)
            return torch.zeros(y, device=device)

        f(torch.tensor([5] * 320))

    def test_mark_unbacked_slice(self):
        # [FUNCTION DOC] Compiles with reduce-overhead mode and fullgraph a sum, then calls it
        # on a GPU tensor of shape one by four with strides five and one marked unbacked on
        # dim 0; just runs.
        @torch.compile(backend="inductor", mode="reduce-overhead", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        x = torch.empty_strided((1, 4), (5, 1), device=GPU_TYPE)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        f(x)

    @torch._dynamo.config.patch(specialize_float=False, capture_scalar_outputs=True)
    def test_unspecialized_float_operations(self):
        # [FUNCTION DOC] Float specialization off, scalar output capture on. For each of
        # multiply, add, subtract and divide in turn (as subtests) compiles a function
        # applying it to a tensor and a float and calls with 2.0, 3.0, 4.0, comparing with
        # eager. The compile counter must show two frames for the first operator (automatic
        # dynamic state persists across compiles) and one for the rest.
        operations = {
            "multiply": operator.mul,
            "add": operator.add,
            "subtract": operator.sub,
            "divide": operator.truediv,
        }

        for i, (name, op) in enumerate(operations.items()):
            with self.subTest(operation=name):

                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Applies the current loop's operator to x and y.
                    return op(x, y)

                cnt = CompileCounterWithBackend("inductor")
                fn_opt = torch.compile(fn, backend=cnt)

                x = torch.arange(3)
                self.assertEqual(fn(x, 2.0), fn_opt(x, 2.0))
                self.assertEqual(fn(x, 3.0), fn_opt(x, 3.0))
                self.assertEqual(fn(x, 4.0), fn_opt(x, 4.0))
                if i == 0:
                    # Automatic dynamic state persists across
                    # compiles so only the first compile
                    # goes through the automatic dynamic step.
                    self.assertEqual(cnt.frame_count, 2)
                else:
                    self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_fallback_specialization(self):
        # [FUNCTION DOC] Float specialization off. Compiles a function with a float argument
        # used by tensor creation, exp, size based math functions and math.floor of sqrt;
        # calls with several second-argument floats and constant z; compares with eager and
        # expects two compiled frames because of automatic dynamic float arguments.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Returns a tuple: a tensor of z, exp of that tensor times
            # x times y, the first dimension size, its square root, floors of square roots of
            # size, element count and number of dims, and floor of square root of z.
            return (
                torch.tensor(z),
                torch.exp(torch.tensor(z)) * (x * y),
                x.size(0),
                math.sqrt(x.size(0)),
                math.floor(math.sqrt(x.size(0))),
                math.floor(math.sqrt(x.numel())),
                math.floor(math.sqrt(x.dim())),
                math.floor(math.sqrt(z)),
            )

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch.compile(fn, backend=cnt)
        x = torch.arange(3)
        z = 1.3

        self.assertEqual(fn(x, 2.0, z), fn_opt(x, 2.0, z))
        self.assertEqual(fn(x, 3.0, z), fn_opt(x, 3.0, z))
        self.assertEqual(fn(x, 4.0, z), fn_opt(x, 4.0, z))
        # Automatic dynamic float arguments
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_softshrink(self):
        # [FUNCTION DOC] Float specialization off. Compares softshrink with a float lambda
        # argument for 2.0, 3.0 and 4.0 against eager and expects two compiled frames. The
        # comment says it exercises both arithmetic and comparison replacements of aten ops.
        # KNOWN ISSUE: Contains a stray debug print of the outputs.
        # This test is particularly interesting since it exercises
        # both standard operator replacements ie. torch.ops.aten.mul.Tensor
        # as well as comparison replacements ie. torch.ops.aten.ge.Scalar
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies the internal softshrink op to x with threshold y.
            return torch._C._nn.softshrink(x, lambd=y)

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch.compile(fn, backend=cnt)
        x = torch.randn(5, 5)

        print(fn(x, 2.0), fn_opt(x, 2.0))

        self.assertEqual(fn(x, 2.0), fn_opt(x, 2.0))
        self.assertEqual(fn(x, 3.0), fn_opt(x, 3.0))
        self.assertEqual(fn(x, 4.0), fn_opt(x, 4.0))
        self.assertEqual(cnt.frame_count, 2)

    @onlyOn(GPU_TYPE)
    def test_dynamic_rblock_bounds(self):
        # [FUNCTION DOC] GPU only. Using a choices handler that forces persistent and
        # non-cooperative reductions, marks dim 0 of a 31-element tensor dynamic with bounds
        # one to 62, compiles a sum and checks the generated code sets the reduction block
        # constant to 64; resets dynamo and repeats with dim 2 marked dynamic with bounds
        # one to 64.
        # KNOWN ISSUE: The tensor is one-dimensional, so marking dim 2 looks like
        # a copy-paste error.
        class ForcePersistent(InductorChoices):
            # [CLASS DOC] HELPER: Choices handler that never uses cooperative reductions and
            # always uses persistent reductions.
            @staticmethod
            def should_use_cooperative_reduction(*args, **kwargs) -> bool:
                # [FUNCTION DOC] HELPER: Static method that ignores its arguments
                # and returns false.
                return False

            @staticmethod
            def should_use_persistent_reduction(*args, **kwargs) -> bool:
                # [FUNCTION DOC] HELPER: Static method that ignores its arguments
                # and returns true.
                return True

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        x = torch.rand([31], device=GPU_TYPE)

        with V.set_choices_handler(ForcePersistent()):
            torch._dynamo.mark_dynamic(x, 0, min=1, max=62)
            fn_c = torch.compile(fn)
            actual, source_codes = run_and_get_code(fn_c, x)
            self.assertEqual(fn(x), actual)
            FileCheck().check("R0_BLOCK: tl.constexpr = 64").run(source_codes[0])
            torch._dynamo.reset()

            torch._dynamo.mark_dynamic(x, 2, min=1, max=64)
            fn_c = torch.compile(fn)
            actual, source_codes = run_and_get_code(fn_c, x)
            self.assertEqual(fn(x), actual)
            FileCheck().check("R0_BLOCK: tl.constexpr = 64").run(source_codes[0])

    def test_non_persistent_dynamic_rblock(self):
        # [FUNCTION DOC] Takes no device argument and uses the GPU type directly. Builds 256
        # by 5536 by 6 permuted tensors, marks the last dim dynamic with bounds 6 to 64,
        # compiles the reduction, checks the generated code does not use the persistent
        # reduction heuristic, and compares with eager.
        def reduce_bounded(x, y):
            # [FUNCTION DOC] HELPER: Asserts the third dimension of x is at most 64 and
            # returns the sum of x times y over that dimension.
            # KNOWN ISSUE: The assert message says 128 while the checked bound is 64.
            """Reduce over a dimension with bounded size."""
            # x shape: [batch, features, reduction_dim]
            # reduction_dim is dynamic but bounded to max 128
            assert x.shape[2] <= 64, f"Reduction dim {x.shape[2]} exceeds max 128"  # noqa: S101

            # Perform reduction (sum) over the last dimension
            result = torch.sum(x * y, dim=2)
            return result

        # Create tensors where reduction dimension is 6 (but could be up to 128)
        batch = 256
        features = 5536
        reduction_dim = 6  # Actual size is small

        x = torch.randn(reduction_dim, batch, features, device=GPU_TYPE).permute(
            1, 2, 0
        )
        y = torch.randn(reduction_dim, batch, features, device=GPU_TYPE).permute(
            1, 2, 0
        )

        torch._dynamo.mark_dynamic(x, 2, min=6, max=64)
        torch._dynamo.mark_dynamic(y, 2, min=6, max=64)

        compiled_fn = torch.compile(reduce_bounded)
        result, source_codes = run_and_get_code(compiled_fn, x, y)

        FileCheck().check_not("@triton_heuristics.persistent").run(source_codes[0])
        expected = reduce_bounded(x, y)

        if not torch.allclose(result, expected, atol=1e-3, rtol=1e-3):
            raise AssertionError

    def test_unspecialized_float_dynamic(self):
        # [FUNCTION DOC] Compiles with dynamic enabled and a compile counter a function
        # multiplying a tensor by a float; calls with 2.0, 3.0, 4.0, compares with eager and
        # expects a single compiled frame.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x times y.
            return x * y

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch.compile(fn, dynamic=True, backend=cnt)
        x = torch.randn(5, 5)

        self.assertEqual(fn(x, 2.0), fn_opt(x, 2.0))
        self.assertEqual(fn(x, 3.0), fn_opt(x, 3.0))
        self.assertEqual(fn(x, 4.0), fn_opt(x, 4.0))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_fallback_symint_specialization(self):
        # [FUNCTION DOC] Float specialization off. Compiles a function taking floor of a float
        # squared times a tensor; calls with 2.0, 3.0, 4.0 and expects four compiled frames
        # (comment: N plus one for automatic dynamic floats).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the floor of x squared, times y.
            return math.floor(x**2) * y

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch.compile(fn, backend=cnt)
        y = torch.arange(3)

        self.assertEqual(fn(2.0, y), fn_opt(2.0, y))
        self.assertEqual(fn(3.0, y), fn_opt(3.0, y))
        self.assertEqual(fn(4.0, y), fn_opt(4.0, y))
        # N + 1 for automatic dynamic float arguments
        self.assertEqual(cnt.frame_count, 4)

    def test_sort_dynamic_shape_with_check(self, device):
        # [FUNCTION DOC] Resets metrics, then compiles dynamically a stable sort on a 10 by
        # 128 float tensor with tied values and checks results against eager; the number of
        # generated kernels is expected to be zero off-GPU and one after the ascending run,
        # two after the descending run, and still two after repeating both.
        # KNOWN ISSUE: The line meant to build a non power of two slice is a bare expression
        # with no effect, so that case is not exercised.
        if torch.device(device).type != GPU_TYPE:

            def check_count(n):
                # [FUNCTION DOC] HELPER: Used when the device is not the GPU type: asserts the
                # generated kernel count is zero, ignoring n.
                self.assertEqual(metrics.generated_kernel_count, 0)

        else:

            def check_count(n):
                # [FUNCTION DOC] HELPER: Used for the GPU type: asserts the generated kernel
                # count equals n.
                self.assertEqual(metrics.generated_kernel_count, n)

        # Test dynamic shapes with statically known small enough to generate
        # persistent sort kernel
        def fn(a, descending):
            # [FUNCTION DOC] HELPER: Asserts the last dimension of a is at most 256 and
            # returns the stable sort of a along the last dim in the given direction.
            torch._check(a.shape[-1] <= 256)
            return a.sort(dim=-1, stable=True, descending=descending)

        inp = torch.rand(10, 128, dtype=torch.float32, device=device)
        inp[:, 10:20] = 1.0
        inp[:, 30:40] = 1.0
        metrics.reset()

        opt_fn = torch.compile(fn, dynamic=True)
        expect = fn(inp, False)
        actual = opt_fn(inp, False)
        self.assertEqual(actual, expect)
        check_count(1)

        expect = fn(inp, True)
        actual = opt_fn(inp, True)
        self.assertEqual(actual, expect)
        check_count(2)

        # Non-power of two
        inp[:, :120]

        expect = fn(inp, False)
        actual = opt_fn(inp, False)
        self.assertEqual(actual, expect)
        check_count(2)  # Reused existing kernel

        expect = fn(inp, True)
        actual = opt_fn(inp, True)
        self.assertEqual(actual, expect)
        check_count(2)  # Reused existing kernel

    def test_coalescing_analysis_sympy_is_constant(self, device):
        # [FUNCTION DOC] Regression test (see its comment) for sympy numeric evaluation
        # hitting a custom modulo assertion during coalescing analysis. Compiles dynamic
        # fullgraph a many-op function on float16 inputs and compares with eager using loose
        # tolerances.
        # Regression test for issue where sympy's is_constant() would trigger
        # numerical evaluation that caused assertion errors in our custom Mod function
        def fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6):
            # [FUNCTION DOC] HELPER: Chains attention, min, variance, reshape, clone and zero,
            # transpose, pow, layer norm, division and subtractions of seven inputs and
            # returns the final tensor.
            t3 = torch.nn.functional.scaled_dot_product_attention(arg0, arg1, arg2)
            t4 = t3.min(dim=3).values
            t6 = arg3.var(dim=0)
            t7 = t6.reshape((29, 50, 32))
            t10 = arg5.clone()
            t10.zero_()
            t11 = t10.transpose(0, 2)
            t12 = torch.pow(torch.pow(t4, arg4), t11)
            t15 = torch.nn.functional.layer_norm(arg6, (32,))
            t16 = t12 / t15
            t17 = ((((t4) - t7) - t16) - t11) - t16
            return t17

        arg0 = torch.rand([29, 50, 32, 5], dtype=torch.float16, device=device)
        arg1 = torch.rand([29, 50, 32, 5], dtype=torch.float16, device=device)
        arg2 = torch.rand([29, 50, 32, 5], dtype=torch.float16, device=device)
        arg3 = torch.rand([3, 10, 4640], dtype=torch.float16, device=device)
        arg4 = torch.rand([29, 50, 32], dtype=torch.float16, device=device)
        arg5 = torch.rand([32, 50, 29], dtype=torch.float16, device=device)
        arg6 = torch.rand([29, 50, 32], dtype=torch.float16, device=device)

        compiled_fn = torch.compile(fn, fullgraph=True, dynamic=True)
        expected = fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        actual = compiled_fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)

    @onlyOn(GPU_TYPE)
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_sympy_infinity_bounds_in_persistent_reduction(self):
        # [FUNCTION DOC] GPU only; both capture flags on. Regression test per its docstring:
        # compiles dynamic fullgraph a function and runs forward and backward on GPU
        # tensors, with no value check, to confirm infinite sympy bounds are handled. Takes
        # no device argument.
        """
        Regression test for sympy Infinity bounds handling in should_use_persistent_reduction.

        When bound_sympy() returns infinity bounds for dynamic reduction dimensions,
        the inductor compiler should handle them correctly without attempting to
        convert infinity to an integer.

        Previously this would fail with: AttributeError: 'Infinity' object has no attribute '_mpf_'
        """

        def fn(arg0, arg2, arg3):
            # [FUNCTION DOC] HELPER: Chains layer norm, conv1d with a viewed weight, square
            # root, group norm and subtractions of three inputs and returns the result.
            t0 = arg0
            t2 = torch.nn.functional.layer_norm(t0, (1024, 10))
            t3 = arg2
            t4 = t3.contiguous().view((67, 1024, 5))
            t5 = torch.nn.functional.conv1d(t2, t4, stride=1, padding=0)
            t6 = arg3
            t7 = torch.sqrt(t6)
            t14 = torch.nn.functional.group_norm(t7, 1)
            t15 = (((t5) - t14) - t5) - t5
            return t15

        arg0 = torch.rand(
            [65, 1024, 10], dtype=torch.float32, device=GPU_TYPE, requires_grad=True
        )
        arg2 = torch.rand(
            [134, 4, 640], dtype=torch.float32, device=GPU_TYPE, requires_grad=True
        )
        arg3 = torch.rand(
            [65, 67, 6], dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True
        )

        compiled_fn = torch.compile(fn, fullgraph=True, dynamic=True)
        out_compiled = compiled_fn(arg0, arg2, arg3)
        # Test backward pass as well - this is where the bug manifested
        out_compiled.sum().backward()

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_combinations_dynamic(self):
        # [FUNCTION DOC] Dynamic-output-shape ops capture on. Compiles dynamic fullgraph
        # pairwise combinations and compares with eager for 1-D inputs of lengths 3, 5 and 7.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns all two-element combinations of the
            # elements of x.
            return torch.combinations(x, r=2)

        compiled_f = torch.compile(f, fullgraph=True, dynamic=True)
        for n in [3, 5, 7]:
            x = torch.randn(n)
            expected = f(x)
            actual = compiled_f(x)
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestInductorDynamic, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # Slow on ASAN after https://github.com/pytorch/pytorch/pull/94068
    if (HAS_CPU or HAS_GPU or HAS_MPS) and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
