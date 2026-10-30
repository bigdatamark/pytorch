# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Tests for the Inductor Pallas backend. Defines a mixin of
# device-parameterized end-to-end tests (compile with the pallas backend
# option, compare against eager, and in a few cases inspect generated
# code), plus make_pallas which clones existing Inductor test classes with
# the pallas backend patched in. At import time it reads the sentinel
# directories pallas_expected_failures and pallas_skip_tests, exits or
# skips on Windows CI, imports test_torchinductor, and registers CPU, CUDA
# and TPU test classes only when the matching Pallas support is available.
import functools
import os
import re
import sys
import unittest

import numpy as np

import torch
import torch._dynamo
import torch._inductor.async_compile
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS
from torch.utils._pallas import has_cpu_pallas, has_cuda_pallas, has_tpu_pallas
from torch.utils._triton import has_triton


# Load pallas expected failures from sentinel files
_pallas_expected_failures_dir = os.path.join(
    os.path.dirname(__file__), "pallas_expected_failures"
)
if os.path.isdir(_pallas_expected_failures_dir):
    PALLAS_EXPECTED_FAILURES = set(os.listdir(_pallas_expected_failures_dir))
else:
    PALLAS_EXPECTED_FAILURES = set()

# Load pallas skip tests from sentinel files (for flaky tests)
_pallas_skip_tests_dir = os.path.join(os.path.dirname(__file__), "pallas_skip_tests")
if os.path.isdir(_pallas_skip_tests_dir):
    PALLAS_SKIP_TESTS = set(os.listdir(_pallas_skip_tests_dir))
else:
    PALLAS_SKIP_TESTS = set()


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library


test_classes = {}


def make_pallas(cls):
    # [FUNCTION DOC] Build a Pallas variant of an existing Inductor test
    # class. Patches the cpu and cuda backend config to pallas and disables
    # the runtime triton dtype and shape asserts. Test methods named in the
    # expected-failure sentinel set get an xfail marker attribute, and those
    # in the skip set get a skip marker attribute, keyed by class name and
    # method name. Creates the class with make_test_cls_with_patches (prefix
    # Pallas, suffix _pallas), records double as an unsupported input
    # generation type, registers it in test_classes and module globals (needed
    # for test collection), sets its module name, and returns it.
    # KNOWN ISSUE: the comment says float64 and int64 are unsupported but only
    #   double is recorded.
    """Create a test class variant that uses Pallas backend.

    Args:
        cls: The test class to create a Pallas variant of.
    """
    patches = [
        (config, "cpu_backend", "pallas"),
        (config, "cuda_backend", "pallas"),
        (config, "test_configs.runtime_triton_dtype_assert", False),
        (config, "test_configs.runtime_triton_shape_assert", False),
    ]
    cls_prefix = "Pallas"
    suffix = "_pallas"

    # Mark tests based on sentinel files in pallas_expected_failures/ and pallas_skip_tests/
    for name in cls.__dict__:
        if name.startswith("test_"):
            fn = cls.__dict__[name]
            if callable(fn):
                key = f"{cls.__name__}.{name}"
                if key in PALLAS_EXPECTED_FAILURES:
                    fn._expected_failure_pallas = True
                elif key in PALLAS_SKIP_TESTS:
                    fn._skip_pallas = True

    def skip_decorator(fn):
        # [FUNCTION DOC] HELPER: Decorator passed to the class builder. Returns
        # the test unchanged unless it carries the skip marker attribute, in which
        # case it is wrapped in an unconditional unittest skip.
        if hasattr(fn, "_skip_pallas"):
            return unittest.skip("Skipped in Pallas backend")(fn)
        return fn

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        *patches,
        xfail_prop="_expected_failure_pallas",
        decorator=skip_decorator,
    )

    # Pallas does not support float64 or int64
    test_class._unsupported_input_gen_types = {"double"}

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


def _skip_if(condition_fn):
    # [FUNCTION DOC] HELPER: Factory for skip decorators. Given a predicate
    # taking the test instance, returns a decorator usable bare or with a
    # reason keyword; the decorated test calls skipTest when the predicate is
    # true.
    def skip(fn=None, *, reason=None):
        # [FUNCTION DOC] HELPER: Decorator entry point. If a function is passed
        # directly it is decorated at once; otherwise returns the decorator so the
        # reason keyword form works.
        def decorator(fn):
            # [FUNCTION DOC] HELPER: Wraps the test function, preserving its metadata.
            @functools.wraps(fn)
            def wrapper(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Runs the predicate on the test instance; if true
                # skips with the given reason or a default message naming the device,
                # otherwise calls the original test. The return value of the test is
                # discarded.
                if condition_fn(self):
                    self.skipTest(reason or f"Not yet working on {self.DEVICE}")
                fn(self, *args, **kwargs)

            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator

    return skip


skip_if_tpu = _skip_if(lambda self: self.DEVICE == "tpu")
skip_if_cpu = _skip_if(lambda self: self.DEVICE == "cpu")
skip_if_cuda = _skip_if(lambda self: self.DEVICE == "cuda")


class PallasTestsMixin:
    # [CLASS DOC] Mixin holding the Pallas backend tests, parameterized by the
    # DEVICE class attribute (cpu, cuda or tpu); concrete subclasses are
    # created below only when the device supports Pallas. On CUDA it disables
    # the per-test fresh cache through an environment variable and clears
    # dynamo and JAX caches before each test to avoid Mosaic GPU state
    # problems. Tests compile small functions through _compile and compare
    # with eager results.
    """Basic tests for Pallas backend functionality (parameterized by DEVICE). Mixin only, not collected.

    NOTE: CUDA tests disable per-test fresh_cache to avoid JAX Mosaic GPU backend state issues.
    The Mosaic backend fails with "Failed to construct pass pipeline" when dynamically loading
    jit-compiled kernels across mock.patch.dict context boundaries (which fresh_cache uses).
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Class setup. For CUDA sets an environment
        # variable that disables the per-test fresh cache; chains to the parent
        # setup.
        super().setUpClass()
        # Disable per-test fresh_cache for CUDA to avoid JAX Mosaic state corruption
        if getattr(cls, "DEVICE", None) == "cuda":
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] HELPER: Class teardown. For CUDA removes the environment
        # variable set at setup (no error if absent), then chains to the parent
        # teardown.
        if getattr(cls, "DEVICE", None) == "cuda":
            os.environ.pop("INDUCTOR_TEST_DISABLE_FRESH_CACHE", None)
        super().tearDownClass()

    def setUp(self):
        # [FUNCTION DOC] HELPER: Per-test setup. For CUDA only: resets dynamo,
        # runs garbage collection and clears JAX caches (ignored if JAX is not
        # importable), to avoid Mosaic GPU backend state leakage.
        super().setUp()
        # Clear caches between tests to avoid Mosaic GPU backend state issues
        if self.DEVICE == "cuda":
            import gc

            torch._dynamo.reset()
            gc.collect()
            try:
                import jax

                jax.clear_caches()
            except ImportError:
                pass

    def _compile(self, fn):
        # [FUNCTION DOC] HELPER: Compile a function with inductor using the pallas
        # backend for the current device. Maps the device name to the matching
        # backend option key (cuda_backend, cpu_backend or tpu_backend) and
        # compiles with static shapes. Raises a key error for an unknown device.
        device_to_backend_key = {
            "cuda": "cuda_backend",
            "cpu": "cpu_backend",
            "tpu": "tpu_backend",
        }
        key = device_to_backend_key[self.DEVICE]
        return torch.compile(
            fn, backend="inductor", options={key: "pallas"}, dynamic=False
        )

    def test_simple_add(self):
        # [FUNCTION DOC] Element-wise addition of two random tensors matches
        # eager. Shapes: one 1D size, plus larger 1D and 2D shapes when not on
        # CUDA.
        """Test basic element-wise addition."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return a + b

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    def test_simple_mul(self):
        # [FUNCTION DOC] Element-wise multiplication matches eager over the same
        # shape set as the add test.
        """Test basic element-wise multiplication."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Multiplies the two inputs.
            return a * b

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    def test_sin(self):
        # [FUNCTION DOC] Sine of a random tensor matches eager; an extra larger
        # shape is used off CUDA.
        """Test sin operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies sine.
            return torch.sin(x)

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def test_fused_ops(self):
        # [FUNCTION DOC] A fused sin plus add of two 1024-element tensors matches
        # eager.
        """Test fused operations (sin + add)."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sine of the first input plus the second.
            return x.sin() + y

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        y = torch.randn(1024, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_exp_log(self):
        # [FUNCTION DOC] Log of exp of a random tensor matches eager.
        """Test exp and log operations."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies exp then log.
            return torch.log(torch.exp(x))

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="sqrt primitive not implemented in Pallas Mosaic GPU")
    def test_sqrt(self):
        # [FUNCTION DOC] Square root of absolute values matches eager. Skipped on
        # CUDA (sqrt primitive missing in Pallas Mosaic GPU).
        """Test sqrt operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies square root.
            return torch.sqrt(x)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE).abs()  # Ensure positive for sqrt
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_tanh(self):
        # [FUNCTION DOC] Hyperbolic tangent of a random tensor matches eager.
        """Test tanh operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies tanh.
            return torch.tanh(x)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_abs_neg(self):
        # [FUNCTION DOC] Absolute value of a negated tensor matches eager.
        """Test abs and neg operations."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Negates then takes the absolute value.
            return torch.abs(-x)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_maximum_minimum(self):
        # [FUNCTION DOC] Sum of element-wise maximum and minimum of two tensors
        # matches eager.
        """Test maximum and minimum operations."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns maximum plus minimum of the two inputs.
            return torch.maximum(a, b) + torch.minimum(a, b)

        compiled = self._compile(fn)

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    @unittest.skipUnless(has_triton(), "requires triton")
    @unittest.skip("Random ops not yet implemented in Pallas backend")
    def test_random_consistency(self):
        # [FUNCTION DOC] Intended to check that seeded random generation agrees
        # between the pallas and triton backends on CUDA; currently
        # unconditionally skipped because random ops are not implemented. Loops
        # over rand and randn generators, compiling each with both backends,
        # seeding before each call and comparing outputs.
        """Test that random number generation is consistent across backends."""
        seed = 1234
        shape = (3, 3)
        dtype = torch.float32

        for rand_fn in [
            functools.partial(torch.rand, shape, dtype=dtype, device="cuda"),
            functools.partial(torch.randn, shape, dtype=dtype, device="cuda"),
        ]:

            @torch.compile(backend="inductor", options={"cuda_backend": "pallas"})
            def get_rand_pallas():
                # [FUNCTION DOC] HELPER: Compiled with the pallas CUDA backend; returns
                # the generator output.
                return rand_fn()

            @torch.compile(backend="inductor", options={"cuda_backend": "triton"})
            def get_rand_triton():
                # [FUNCTION DOC] HELPER: Compiled with the triton CUDA backend; returns
                # the generator output.
                return rand_fn()

            torch.manual_seed(seed)
            pallas_output = get_rand_pallas()
            torch.manual_seed(seed)
            triton_output = get_rand_triton()

            self.assertEqual(pallas_output, triton_output)

    def test_compile_options(self):
        # [FUNCTION DOC] Compiles a sin plus cos function with the pallas backend,
        # fetches the generated source with run_and_get_code and asserts it
        # contains the jax, jax.numpy and pallas imports.
        """Test that Pallas backend is properly configured."""

        pallas_fn = self._compile(lambda a, b: a.sin() + b.cos())

        _, (code,) = run_and_get_code(
            pallas_fn,
            torch.randn(128, device=self.DEVICE),
            torch.randn(128, device=self.DEVICE),
        )
        # Verify Pallas-specific code generation
        self.assertIn("import jax", code)
        self.assertIn("import jax.numpy as jnp", code)
        self.assertIn("from jax.experimental import pallas as pl", code)

    def test_jax_jit_wrapper_is_emitted(self):
        # [FUNCTION DOC] Checks the generated source for a jit wrapper. Finds the
        # kernel name by regular expression, requires a wrapper function named
        # after it and isolates the wrapper text up to the main function. Asserts
        # the source uses jax.jit with donated arguments. On CUDA the wrapper must
        # use plgpu.kernel and the source must have no copy_ call; elsewhere it
        # must use input_output_aliases. In all cases the wrapper must not mention
        # torch.
        """Ensure generated Pallas code wraps pl.pallas_call in jax.jit."""

        pallas_fn = self._compile(lambda a, b: a + b)

        _, (code,) = run_and_get_code(
            pallas_fn,
            torch.randn(128, device=self.DEVICE),
            torch.randn(128, device=self.DEVICE),
        )

        kernel_match = re.search(r"def (pallas_[A-Za-z0-9_]+)_kernel", code)
        self.assertIsNotNone(kernel_match)
        kernel_name = kernel_match.group(1)
        wrapper_name = f"{kernel_name}_jit_wrapper"
        self.assertIn(wrapper_name, code)
        start = code.index(f"def {wrapper_name}")
        end = code.index(f"def {kernel_name}_main", start)
        wrapper_block = code[start:end]

        self.assertIn("jax.jit", code)
        self.assertIn("donate_argnums", code)
        if self.DEVICE == "cuda":
            # Mosaic GPU backend uses plgpu.kernel instead of pl.pallas_call
            self.assertIn("plgpu.kernel", wrapper_block)
            self.assertNotIn(".copy_(", code)
        else:
            # CPU backend uses pl.pallas_call with input_output_aliases
            self.assertIn("input_output_aliases", wrapper_block)
        self.assertNotIn("torch.", wrapper_block)

    def test_2d_tensor(self):
        # [FUNCTION DOC] Addition of 2D tensors matches eager: a 32 by 32 shape,
        # plus 16 by 2048 off CUDA.
        """Test with 2D tensors."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return x + y

        shapes = [(32, 32)]
        if self.DEVICE != "cuda":
            shapes.append((16, 2048))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                y = torch.randn(shape, device=self.DEVICE)
                result = compiled(x, y)
                expected = fn(x, y)
                self.assertEqual(result, expected)

    @skip_if_cuda(
        reason="iteration variables not supported in Pallas GPU (Mosaic) backend"
    )
    def test_different_shapes(self):
        # [FUNCTION DOC] Multiplying by a constant for 1D inputs of sizes 64 to
        # 1024 matches eager, reusing one compiled function. Skipped on CUDA
        # (iteration variables unsupported).
        """Test with different tensor shapes."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by two.
            return x * 2.0

        compiled = self._compile(fn)

        for shape in [(64,), (128,), (256,), (1024,)]:
            x = torch.randn(shape, device=self.DEVICE)
            result = compiled(x)
            expected = fn(x)
            self.assertEqual(result, expected)

    @skip_if_cuda
    def test_contiguous_index_validation(self):
        # [FUNCTION DOC] End-to-end check of contiguous and non-contiguous inputs.
        # Runs an add and a scale on contiguous tensors, then for several shapes
        # feeds a transposed (non-contiguous) view and its contiguous copy through
        # a compiled sine, asserting the views are as expected and results match
        # eager. Skipped on CUDA.
        """Test that contiguous index validation works correctly end-to-end."""

        # Test 1: Contiguous operations should work
        def contiguous_add(a, b):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return a + b

        compiled = self._compile(contiguous_add)

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = contiguous_add(a, b)
        self.assertEqual(result, expected)

        # Test 2: Operations on contiguous tensors should work
        def contiguous_mul(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by two.
            return x * 2.0

        compiled = self._compile(contiguous_mul)

        x = torch.randn(128, 8, device=self.DEVICE)
        result = compiled(x)
        expected = contiguous_mul(x)
        self.assertEqual(result, expected)

        # Test 3: Non-contiguous views should work with the simplified dlpack approach
        # The direct dlpack conversion handles non-contiguous tensors correctly
        def operate_on_tensor(x):
            # [FUNCTION DOC] HELPER: Applies sine.
            return x.sin()

        compiled = self._compile(operate_on_tensor)

        for rows, cols in [(64, 32), (5, 8), (3215, 23), (8, 128), (128, 8)]:
            with self.subTest(rows=rows, cols=cols):
                # Create a transposed (non-contiguous) view
                x = torch.randn(rows, cols, device=self.DEVICE)
                x_t = x.t()  # Non-contiguous view
                self.assertFalse(x_t.is_contiguous())

                # With the simplified dlpack approach, non-contiguous tensors now work
                result = compiled(x_t)
                expected = operate_on_tensor(x_t)
                self.assertEqual(result, expected)

                # Contiguous tensors should also continue to work
                x_t_contiguous = x_t.contiguous()
                self.assertTrue(x_t_contiguous.is_contiguous())
                result = compiled(x_t_contiguous)
                expected = operate_on_tensor(x_t_contiguous)
                self.assertEqual(result, expected)

    @skip_if_cuda(reason="strided access not supported in Pallas GPU (Mosaic) backend")
    def test_strided_int_pallas(self):
        # [FUNCTION DOC] Strided slice with step two of a 1D range, scaled by two,
        # matches eager. Skipped on CUDA.
        """Test strided access patterns with the Pallas backend."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes every second element and doubles it.
            return x[::2] * 2.0

        compiled = self._compile(fn)

        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="strided access not supported in Pallas GPU (Mosaic) backend")
    def test_strided_offset_pallas(self):
        # [FUNCTION DOC] Strided slice starting at offset one with step two, plus
        # one, matches eager. Skipped on CUDA.
        """Test strided access with offset."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes odd-index elements and adds one.
            return x[1::2] + 1.0

        compiled = self._compile(fn)

        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_strided_2d_pallas(self):
        # [FUNCTION DOC] Scaling a 2D tensor by three matches eager.
        """Test strided access on 2D tensors."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by three.
            # Simple operation on 2D tensor
            return x * 3.0

        compiled = self._compile(fn)

        x = torch.randn(8, 16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_stride_non_contiguous_1d(self):
        # [FUNCTION DOC] Compiled scale-and-shift on several 1D non-contiguous
        # inputs (strides two, four and a double stride) matches eager; asserts
        # each input is non-contiguous.
        """Test 1D non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_1d = torch.arange(256, dtype=torch.float32, device=self.DEVICE)
        for x in [base_1d[::2], base_1d[::4], base_1d[::2][::2]]:
            self.assertFalse(x.is_contiguous())
            self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_row_stride(self):
        # [FUNCTION DOC] Scale-and-shift on a 2D view taking every second row
        # matches eager.
        """Test 2D row-strided input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, :]  # (16, 32) with stride (64, 1)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_col_stride(self):
        # [FUNCTION DOC] Scale-and-shift on a 2D view taking every second column
        # matches eager.
        """Test 2D col-strided input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[:, ::2]  # (32, 16) with stride (32, 2)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_both_stride(self):
        # [FUNCTION DOC] Scale-and-shift on a 2D view striding both dimensions
        # matches eager.
        """Test 2D both-strided input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, ::2]  # (16, 16) with stride (64, 2)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_cuda
    def test_stride_non_contiguous_2d_transpose(self):
        # [FUNCTION DOC] Scale-and-shift on transposed inputs of several shapes
        # (including odd sizes) matches eager. Skipped on CUDA.
        """Test 2D transposed input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        for rows, cols in [
            (32, 32),
            (2048, 2048),
            (64, 32),
            (5, 8),
            (3215, 23),
            (8, 128),
            (128, 8),
        ]:
            with self.subTest(rows=rows, cols=cols):
                base_2d = torch.randn(rows, cols, device=self.DEVICE)
                x = base_2d.t()
                self.assertFalse(x.is_contiguous())
                self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_3d(self):
        # [FUNCTION DOC] Scale-and-shift on a 3D view striding every dimension by
        # two matches eager.
        """Test 3D non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_3d = torch.randn(8, 8, 8, device=self.DEVICE)
        x = base_3d[::2, ::2, ::2]
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_cuda
    def test_stride_non_contiguous_permuted(self):
        # [FUNCTION DOC] Scale-and-shift on a permuted 3D tensor matches eager.
        # Skipped on CUDA.
        """Test permuted non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_3d = torch.randn(8, 8, 8, device=self.DEVICE)
        x = base_3d.permute(2, 0, 1)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_cuda
    def test_stride_non_contiguous_channels_last(self):
        # [FUNCTION DOC] Scale-and-shift on a channels-last 4D tensor, asserted
        # non-contiguous, matches eager. Skipped on CUDA.
        """Test channels-last (NHWC) non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        x = torch.randn(2, 3, 4, 5, device=self.DEVICE).to(
            memory_format=torch.channels_last
        )
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_diagonal(self):
        # [FUNCTION DOC] Scale-and-shift on the diagonal of a square matrix (a
        # large-stride 1D view) matches eager.
        """Test diagonal (large stride) non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d.diagonal()
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_as_strided(self):
        # [FUNCTION DOC] Scale-and-shift on a tensor built with an explicit size
        # and stride layout matches eager.
        """Test as_strided (custom layout) non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_flat = torch.randn(256, device=self.DEVICE)
        x = torch.as_strided(base_flat, size=(4, 8), stride=(16, 2))
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_select_stride(self):
        # [FUNCTION DOC] Scale-and-shift on a strided row selected from a 2D
        # tensor matches eager.
        """Test select then stride on non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[3, ::2]
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_unsqueeze(self):
        # [FUNCTION DOC] Scale-and-shift on a strided 2D view with a leading
        # dimension added matches eager.
        """Test unsqueeze on strided non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, ::2].unsqueeze(0)
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_tpu(reason="TPU doesn't support float 64")
    def test_stride_non_contiguous_dtypes(self):
        # [FUNCTION DOC] Scale-and-shift on strided views of float64, int32 and
        # complex64 data matches eager. Skipped on TPU (no float64).
        """Test non-contiguous patterns with various dtypes."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        for dtype in [torch.float64, torch.int32, torch.complex64]:
            if dtype == torch.int32:
                base = torch.arange(32, dtype=dtype, device=self.DEVICE)
            else:
                base = torch.randn(32, dtype=dtype, device=self.DEVICE)
            x = base[::2]
            self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @unittest.skip(
        "Expanded tensors (stride=0) generate index expressions that don't match "
        "the contiguous layout after .contiguous() is called at runtime, causing "
        "gather ops on GPU which aren't supported by Pallas GPU lowering"
    )
    def test_stride_expanded_tensors(self):
        # [FUNCTION DOC] Intended to cover expanded (zero-stride) inputs for
        # single- and multi-dimension expansion; currently unconditionally skipped
        # since gather ops are unsupported in Pallas GPU lowering.
        """Test expanded tensors with stride=0 (distinct code path)."""
        compiled = self._compile(lambda x, y: x + y)

        # Single dim expansion
        x = torch.randn(1, 16, device=self.DEVICE).expand(8, 16)
        y = torch.randn(8, 16, device=self.DEVICE)
        self.assertEqual(x.stride()[0], 0)
        self.assertEqual(compiled(x, y), x + y)

        # Multi-dim expansion
        x = torch.randn(1, 1, 16, device=self.DEVICE).expand(4, 8, 16)
        self.assertEqual(compiled(x, x), x + x)

    def test_stride_multiple_inputs(self):
        # [FUNCTION DOC] Adds and multiplies several strided inputs with identical
        # stride patterns, then separately checks a strided matrix combined with a
        # broadcast vector and a 0-D scalar tensor, all against eager. Same-stride
        # inputs avoid scatter operations.
        """Test multiple strided inputs and broadcasting."""
        compiled = self._compile(lambda a, b, c: a * b + c)

        # Use separate base tensors to create strided inputs with the SAME stride pattern
        # This avoids triggering scatter operations which aren't supported in Pallas GPU lowering
        base_a = torch.randn(32, 32, device=self.DEVICE)
        base_b = torch.randn(32, 32, device=self.DEVICE)

        # Multiple strided inputs with the same stride pattern
        a = base_a[::2, ::2]  # (16, 16) with stride (64, 2)
        b = base_b[::2, ::2]  # (16, 16) with stride (64, 2)
        c = torch.randn(16, 16, device=self.DEVICE)
        self.assertEqual(a.stride(), b.stride())
        self.assertFalse(a.is_contiguous())
        self.assertFalse(b.is_contiguous())
        self.assertEqual(compiled(a, b, c), a * b + c)

        # Broadcasting with strided
        x = base_a[::2, ::2]  # (16, 16)
        y = torch.randn(16, device=self.DEVICE)  # broadcasts
        s = torch.tensor(2.0, device=self.DEVICE)  # scalar
        compiled_bcast = self._compile(lambda x, y, s: x + y * s)
        self.assertEqual(compiled_bcast(x, y, s), x + y * s)

    @skip_if_cuda
    def test_scalar_scalar_ops(self):
        # [FUNCTION DOC] Binary arithmetic (add, multiply, subtract, divide) on
        # 0-D float32 tensors matches eager; also asserts the result is 0-D
        # float32. Skipped on CUDA.
        """Test scalar-scalar operations."""

        def test_add(a, b):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return a + b

        def test_mul(a, b):
            # [FUNCTION DOC] HELPER: Multiplies the two inputs.
            return a * b

        def test_sub(a, b):
            # [FUNCTION DOC] HELPER: Subtracts the second input from the first.
            return a - b

        def test_div(a, b):
            # [FUNCTION DOC] HELPER: Divides the first input by the second.
            return a / b

        for fn in [test_add, test_mul, test_sub, test_div]:
            with self.subTest(op=fn.__name__):
                compiled = self._compile(fn)

                # Test with 0-D tensors (scalars)
                a = torch.tensor(3.5, dtype=torch.float32, device=self.DEVICE)
                b = torch.tensor(2.0, dtype=torch.float32, device=self.DEVICE)

                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

                # Ensure result is also scalar
                self.assertEqual(result.dim(), 0)
                self.assertEqual(result.dtype, torch.float32)

    def test_scalar_tensor_ops(self):
        # [FUNCTION DOC] Arithmetic between a 0-D tensor and tensors of rank one
        # to three, with the scalar on either side, for add and multiply. Chooses
        # the argument order from the helper name, and asserts value, shape and
        # float32 dtype.
        """Test scalar-tensor operations."""

        def test_scalar_add_tensor(s, t):
            # [FUNCTION DOC] HELPER: Scalar plus tensor.
            return s + t

        def test_tensor_add_scalar(t, s):
            # [FUNCTION DOC] HELPER: Tensor plus scalar.
            return t + s

        def test_scalar_mul_tensor(s, t):
            # [FUNCTION DOC] HELPER: Scalar times tensor.
            return s * t

        def test_tensor_mul_scalar(t, s):
            # [FUNCTION DOC] HELPER: Tensor times scalar.
            return t * s

        shapes = [(16,), (8, 8), (4, 4, 4)]

        for shape in shapes:
            for fn in [
                test_scalar_add_tensor,
                test_tensor_add_scalar,
                test_scalar_mul_tensor,
                test_tensor_mul_scalar,
            ]:
                with self.subTest(op=fn.__name__, shape=shape):
                    compiled = self._compile(fn)

                    # Create 0-D scalar tensor
                    scalar = torch.tensor(2.5, dtype=torch.float32, device=self.DEVICE)
                    tensor = torch.randn(shape, dtype=torch.float32, device=self.DEVICE)

                    if "scalar" in fn.__name__.split("_")[0]:
                        result = compiled(scalar, tensor)
                        expected = fn(scalar, tensor)
                    else:
                        result = compiled(tensor, scalar)
                        expected = fn(tensor, scalar)

                    self.assertEqual(result, expected)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, torch.float32)

    def test_non_power_of_2_sizes(self):
        # [FUNCTION DOC] Addition of two 10-element tensors (size not a power of
        # two) matches eager. On GPU the backend relies on TMA out-of-bounds
        # masking; on CPU and TPU on masked loads and stores.
        """Test that non-power-of-2 tensor sizes work correctly.

        On GPU (Mosaic backend), TMA automatically handles OOB masking for
        non-power-of-2 sizes. On CPU/TPU, masked loads/stores are used.
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return a + b

        compiled = self._compile(fn)

        # Test a specific non-power-of-2 size (10)
        a = torch.randn(10, device=self.DEVICE)
        b = torch.randn(10, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_non_power_of_2_multiple_ops(self):
        # [FUNCTION DOC] A multi-op expression of sin, cos and product on
        # 17-element tensors matches eager.
        """Test non-power-of-2 sizes with multiple operations."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sine of x plus cosine of y minus their product.
            return x.sin() + y.cos() - (x * y)

        compiled = self._compile(fn)

        # Non-power-of-2 size: 17
        x = torch.randn(17, device=self.DEVICE)
        y = torch.randn(17, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    @skip_if_tpu(reason="Cannot do int indexing on TPU")
    @skip_if_cuda(reason="gather not supported in Pallas GPU (Mosaic) backend")
    def test_complex_indexing_gather(self):
        # [FUNCTION DOC] Index a 1D float tensor with an int64 index vector of
        # eight entries; result matches eager. Skipped on TPU (integer indexing)
        # and CUDA (gather).
        """Test complex indexing with gather-like operations."""

        def fn(x, indices):
            # [FUNCTION DOC] HELPER: Gathers elements of the first input by the index
            # tensor.
            # Use indices to gather elements from x
            return x[indices]

        compiled = self._compile(fn)

        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        # Use power-of-2 size for indices (Pallas Mosaic requirement)
        indices = torch.tensor(
            [0, 2, 5, 7, 11, 13, 14, 15], dtype=torch.int64, device=self.DEVICE
        )
        result = compiled(x, indices)
        expected = fn(x, indices)
        self.assertEqual(result, expected)

    @skip_if_tpu(reason="Cannot do int indexing on TPU")
    # Pallas Mosaic backend doesn't support gather operations with array indices
    # This limitation is in the Pallas/Mosaic lowering, not our implementation
    @skip_if_cuda(
        reason="Multi-dimensional gather not supported on Pallas Mosaic (CUDA) backend"
    )
    def test_complex_indexing_2d(self):
        # [FUNCTION DOC] Selects rows of a 2D tensor using an int64 index vector
        # of four entries; matches eager. Skipped on TPU and CUDA (multi-
        # dimensional gather).
        """Test complex indexing on 2D tensors with integer array indexing."""

        def fn(x, row_indices):
            # [FUNCTION DOC] HELPER: Selects the rows given by the index tensor.
            # Select specific rows using integer array indexing
            return x[row_indices, :]

        compiled = self._compile(fn)

        x = torch.randn(16, 8, device=self.DEVICE)
        # Use power-of-2 sizes (Pallas Mosaic requirement)
        row_indices = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device=self.DEVICE)
        result = compiled(x, row_indices)
        expected = fn(x, row_indices)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex64_mul(self):
        # [FUNCTION DOC] Complex64 multiplication matches eager for a 128-element
        # input, plus 2048 off CUDA. Skipped on TPU.
        """Test complex64 multiplication."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Multiplies the two inputs.
            return a * b

        sizes = [128]
        if self.DEVICE != "cuda":
            sizes.append(2048)
        for size in sizes:
            with self.subTest(size=size):
                compiled = self._compile(fn)
                a = torch.randn(size, dtype=torch.complex64, device=self.DEVICE)
                b = torch.randn(size, dtype=torch.complex64, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_conj(self):
        # [FUNCTION DOC] Complex conjugate of a complex64 tensor matches eager.
        # Skipped on TPU.
        """Test complex conjugate."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Conjugates the input.
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_real(self):
        # [FUNCTION DOC] Real part of a complex64 tensor matches eager. Skipped on
        # TPU.
        """Test extracting real part of complex tensor."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Extracts the real part.
            return torch.real(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_imag(self):
        # [FUNCTION DOC] Imaginary part of a complex64 tensor matches eager.
        # Skipped on TPU.
        """Test extracting imaginary part of complex tensor."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Extracts the imaginary part.
            return torch.imag(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_abs(self):
        # [FUNCTION DOC] Magnitude of a complex64 tensor matches eager. Skipped on
        # TPU.
        """Test complex absolute value (magnitude)."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the absolute value.
            return torch.abs(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex128_conj(self):
        # [FUNCTION DOC] Complex conjugate of a complex128 tensor matches eager.
        # Skipped on TPU.
        """Test complex128 conjugate operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Conjugates the input.
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex128, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_mul_scalar(self):
        # [FUNCTION DOC] Multiplying a complex64 tensor by a real constant matches
        # eager. Skipped on TPU.
        """Test complex multiplication with scalar."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by 2.5.
            return x * 2.5

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_conj_mul(self):
        # [FUNCTION DOC] Conjugate of one complex64 tensor times another matches
        # eager. Skipped on TPU.
        """Test conjugate followed by multiplication."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Conjugates the first input and multiplies by the
            # second.
            return torch.conj(x) * y

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        y = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_where(self):
        # [FUNCTION DOC] Conditional select between two tensors based on the sign
        # of the first matches eager over 1D and (off CUDA) larger 1D and 2D
        # shapes.
        """Test torch.where operation."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Chooses the first input where positive, else the
            # second.
            return torch.where(x > 0, x, y)

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                y = torch.randn(shape, device=self.DEVICE)
                result = compiled(x, y)
                expected = fn(x, y)
                self.assertEqual(result, expected)

    def test_clamp(self):
        # [FUNCTION DOC] Clamping values to the range minus one to one matches
        # eager over several shapes.
        """Test torch.clamp operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Clamps the input to the range minus one to one.
            return torch.clamp(x, -1.0, 1.0)

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE) * 2
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def test_comparison_ops(self):
        # [FUNCTION DOC] Sums of float conversions of greater-than, less-than and
        # equal comparisons match eager.
        """Test comparison operations."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the float results of three comparisons
            # between the inputs.
            gt = a > b
            lt = a < b
            eq = a == b
            return gt.float() + lt.float() + eq.float()

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    def test_logical_ops(self):
        # [FUNCTION DOC] Logical and of two positivity masks, converted to float,
        # matches eager.
        """Test logical operations."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Logical and of the two positivity masks as float.
            return torch.logical_and(a > 0, b > 0).float()

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    def test_sign(self):
        # [FUNCTION DOC] Sign of a 16-element random tensor matches eager.
        """Test sign operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies sign.
            return torch.sign(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="integer_pow primitive not implemented in Pallas Mosaic GPU")
    def test_reciprocal(self):
        # [FUNCTION DOC] Reciprocal of shifted random values matches eager.
        # Skipped on CUDA (integer_pow primitive missing).
        """Test reciprocal operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies reciprocal.
            return torch.reciprocal(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE) + 1.0  # Avoid zeros
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_square(self):
        # [FUNCTION DOC] Elementwise square matches eager.
        """Test square operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies square.
            return torch.square(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu(
        reason="Pallas loweing crash: https://github.com/jax-ml/jax/issues/36149"
    )
    def test_erf(self):
        # [FUNCTION DOC] Error function matches eager. Skipped on TPU because of a
        # Pallas lowering crash.
        """Test erf operation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies erf.
            return torch.erf(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu(
        reason="Pallas loweing crash: https://github.com/jax-ml/jax/issues/36149"
    )
    def test_atan2(self):
        # [FUNCTION DOC] Two-argument arctangent matches eager. Skipped on TPU
        # because of a Pallas lowering crash.
        """Test atan2 operation."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Applies atan2 to the two inputs.
            return torch.atan2(a, b)

        compiled = self._compile(fn)

        a = torch.randn(16, device=self.DEVICE)
        b = torch.randn(16, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_sum_reduction(self):
        # [FUNCTION DOC] Full sum reduction to a scalar matches eager over a 1D
        # size, plus a larger size off CUDA.
        """Test sum reduction."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums all elements.
            return x.sum()

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def test_max_reduction(self):
        # [FUNCTION DOC] Full max reduction matches eager over a 1D size, plus a
        # larger size off CUDA.
        """Test max reduction."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the overall maximum.
            return x.max()

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def test_min_reduction(self):
        # [FUNCTION DOC] Full min reduction matches eager over a 16-element input,
        # plus a larger size off CUDA.
        """Test min reduction."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the overall minimum.
            return x.min()

        shapes = [(16,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_tpu(reason="reduce_prod primitive not implemented in Pallas TPU lowering")
    @skip_if_cuda(reason="reduce_prod primitive not implemented in Pallas Mosaic GPU")
    def test_prod_reduction(self):
        # [FUNCTION DOC] Full product reduction of inputs scaled down to avoid
        # overflow matches eager. Skipped on TPU and CUDA (reduce_prod primitive
        # missing).
        """Test prod reduction."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Product of all elements after scaling by 0.1.
            # Use smaller values to avoid overflow
            return (x * 0.1).prod()

        shapes = [(16,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_softmax_two_pass(self):
        # [FUNCTION DOC] Softmax over the last dimension (max and sum reductions)
        # for two 2D shapes; resets dynamo each iteration and compares with eager.
        # Skipped on CUDA.
        """Test two-pass softmax (max reduction + sum reduction)."""

        for shape in [(32, 64), (2048, 64)]:
            with self.subTest(shape=shape):
                torch._dynamo.reset()

                def fn(x):
                    # [FUNCTION DOC] HELPER: Softmax along the last dimension.
                    return torch.softmax(x, dim=-1)

                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_non_stride1_reduction(self):
        # [FUNCTION DOC] Sum along dim 0 and dim 1 of a square 8 by 8 tensor, to
        # check reduction axis detection uses strides not shapes. Resets dynamo
        # per dim. Skipped on CUDA.
        """Test reductions along non-innermost axis on square tensors.

        On square tensors (e.g. 8x8), the reduction axis cannot be inferred
        from shape alone since both dims have the same size. This verifies
        that stride-based axis detection works for both dim=0 and dim=1.
        """
        x = torch.randn(8, 8, device=self.DEVICE)
        for dim in [0, 1]:
            with self.subTest(dim=dim):
                torch._dynamo.reset()

                def fn(x, dim=dim):
                    # [FUNCTION DOC] HELPER: Sums over the dimension bound at definition time.
                    return x.sum(dim)

                compiled = self._compile(fn)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_rms_norm(self):
        # [FUNCTION DOC] RMS normalization (mean of squares, reciprocal square
        # root, weight scale) for two shapes matches eager; resets dynamo per
        # shape. Skipped on CUDA.
        """Test RMS normalization (mean-of-squares reduction + rsqrt)."""

        for rows, cols in [(32, 64), (2048, 64)]:
            with self.subTest(rows=rows, cols=cols):
                torch._dynamo.reset()

                def fn(x, weight):
                    # [FUNCTION DOC] HELPER: Normalizes x by its root mean square over the
                    # last dimension and scales by weight.
                    variance = x.pow(2).mean(-1, keepdim=True)
                    x = x * torch.rsqrt(variance + 1e-6)
                    return x * weight

                compiled = self._compile(fn)
                x = torch.randn(rows, cols, device=self.DEVICE)
                weight = torch.randn(cols, device=self.DEVICE)
                result = compiled(x, weight)
                expected = fn(x, weight)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_welford(self):
        # [FUNCTION DOC] Variance and mean over the last dimension for two shapes.
        # Compares against numpy (sample variance, ddof one) because eager
        # torch_tpu cannot lower var_mean. Skipped on CUDA.
        """Test Welford variance/mean computation (two-pass fallback)."""

        for shape in [(32, 64), (2048, 64)]:
            with self.subTest(shape=shape):
                torch._dynamo.reset()

                def fn(x):
                    # [FUNCTION DOC] HELPER: Returns variance and mean along the last
                    # dimension, kept as a dimension.
                    return torch.var_mean(x, dim=-1, keepdim=True)

                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                var_result, mean_result = compiled(x)
                # Eager mode torch_tpu doesn't support lowering var_mean, so comparing with numpy
                var_expected = np.var(x.cpu().numpy(), axis=-1, keepdims=True, ddof=1)
                mean_expected = np.mean(x.cpu().numpy(), axis=-1, keepdims=True)
                self.assertEqual(mean_result.cpu().numpy(), mean_expected)
                self.assertEqual(var_result.cpu().numpy(), var_expected)

    @skip_if_cuda
    def test_layer_norm(self):
        # [FUNCTION DOC] Layer normalization with weight and bias for two shapes
        # matches eager; resets dynamo per shape. Skipped on CUDA.
        """Test layer normalization (mean + variance reduction, normalize, scale + shift)."""

        for rows, cols in [(32, 64), (2048, 64)]:
            with self.subTest(rows=rows, cols=cols):
                torch._dynamo.reset()

                def fn(x, weight, bias):
                    # [FUNCTION DOC] HELPER: Layer-norms x over the last dimension with
                    # epsilon 1e-6, then applies weight and bias.
                    mean = x.mean(-1, keepdim=True)
                    variance = (x - mean).pow(2).mean(-1, keepdim=True)
                    x = (x - mean) * torch.rsqrt(variance + 1e-6)
                    return x * weight + bias

                compiled = self._compile(fn)
                x = torch.randn(rows, cols, device=self.DEVICE)
                weight = torch.randn(cols, device=self.DEVICE)
                bias = torch.randn(cols, device=self.DEVICE)
                result = compiled(x, weight, bias)
                expected = fn(x, weight, bias)
                self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_rope(self):
        # [FUNCTION DOC] Rotary position embedding using contiguous halves,
        # rotated with cos and sin and concatenated; exercises non-contiguous
        # output aliases. Skipped on CUDA and TPU.
        """Test Rotary Position Embedding with slice + cat.

        Splits input into halves, applies cos/sin rotation, and concatenates
        back. Exercises non-contiguous output aliases via torch.cat.
        """

        def fn(x, cos, sin):
            # [FUNCTION DOC] HELPER: Splits the last dimension in half, applies the
            # rotation to both halves and concatenates.
            d = x.shape[-1]
            x1 = x[..., : d // 2]
            x2 = x[..., d // 2 :]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        compiled = self._compile(fn)

        x = torch.randn(32, 64, device=self.DEVICE)
        cos = torch.randn(32, 32, device=self.DEVICE)
        sin = torch.randn(32, 32, device=self.DEVICE)
        result = compiled(x, cos, sin)
        expected = fn(x, cos, sin)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu  # stack+where fusion doesn't broadcast correctly on TPU yet
    def test_rope_interleaved(self):
        # [FUNCTION DOC] Rotary embedding using even and odd strided slices,
        # recombined by stacking and reshaping; exercises strided input. Skipped
        # on CUDA and TPU (stack plus where fusion).
        """Test Rotary Position Embedding with interleaved halves.

        Uses even/odd stride-2 slicing instead of contiguous halves, then
        reassembles via stack + reshape. Exercises strided input access.
        """

        def fn(x, cos, sin):
            # [FUNCTION DOC] HELPER: Rotates the even and odd interleaved elements and
            # reassembles to the input shape.
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.stack([o1, o2], dim=-1).reshape_as(x)

        compiled = self._compile(fn)

        x = torch.randn(32, 64, device=self.DEVICE)
        cos = torch.randn(32, 32, device=self.DEVICE)
        sin = torch.randn(32, 32, device=self.DEVICE)
        result = compiled(x, cos, sin)
        expected = fn(x, cos, sin)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu  # output last dim 10 not 128-aligned, Mosaic rejects it
    def test_chained_stride_slice(self):
        # [FUNCTION DOC] Chained strided slices on the last dimension (which
        # compose into a single strided access), plus one, match eager for a 4 by
        # 480 input. Skipped on CUDA and TPU.
        """Test that chained stride slices compose into a single strided access.

        x[:, 1::2][:, 2::3][:, 3::4] should compose to x[:, 23::24].
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies three chained strided slices and adds
            # one.
            return x[:, 1::2][:, 2::3][:, 3::4] + 1

        compiled = self._compile(fn)

        # last dim = 480 which is divisible by 24
        x = torch.randn(4, 480, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu  # store uses flatten+scatter, unsupported on Mosaic
    def test_strided_multi_dim(self):
        # [FUNCTION DOC] Strided access on both dimensions of an 8 by 12 input,
        # plus one, matches eager. Skipped on CUDA and TPU (flatten plus scatter
        # store).
        """Test strided access on multiple dimensions simultaneously."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Strides rows by two and columns by three and adds
            # one.
            return x[::2, ::3] + 1.0

        compiled = self._compile(fn)

        # 8 % 2 == 0 and 12 % 3 == 0
        x = torch.randn(8, 12, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu  # falls back to flatten+gather, unsupported on Mosaic
    def test_strided_non_divisible(self):
        # [FUNCTION DOC] Stride three over a 16-element range (not divisible),
        # doubled, matches eager; this takes the flatten plus gather fallback.
        # Skipped on CUDA and TPU.
        """Test strided access where dim is not divisible by stride.

        Falls back to flatten+gather on CPU (blocks tiling).
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes every third element and doubles it.
            return x[::3] * 2.0

        compiled = self._compile(fn)

        # 16 % 3 != 0 → should fall back
        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    def test_strided_large_offset(self):
        # [FUNCTION DOC] Slice from offset five with step two over ten elements,
        # plus one, matches eager. Skipped on CUDA.
        """Test strided access where offset >= stride (skip blocks)."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes elements from index five with step two and
            # adds one.
            return x[5::2] + 1.0

        compiled = self._compile(fn)

        # offset=5, stride=2: skip=2, r=1 → reshape(5,2)[2:,1]
        x = torch.arange(10, dtype=torch.float32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu  # store uses scatter, unsupported on Mosaic
    def test_strided_large_offset_2d(self):
        # [FUNCTION DOC] 2D slice on the last dimension from offset five with step
        # two, doubled, matches eager. Skipped on CUDA and TPU (scatter store).
        """Test 2D strided access where offset >= stride on last dim."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Strides the last dimension from offset five and
            # doubles.
            return x[:, 5::2] * 2.0

        compiled = self._compile(fn)

        x = torch.randn(4, 8, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    @skip_if_cuda(reason="arange not supported in Pallas GPU (Mosaic) backend")
    def test_arange_multi_output(self):
        # [FUNCTION DOC] Two arange tensors (one viewed 8 by 8, one offset range)
        # combined with an input, returning two outputs; checks tuple length and
        # each element against eager. Skipped on TPU and CUDA.
        """Test arange with view and multiple outputs."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the aranges on the input device and
            # returns the product and the sum with the second range.
            rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
            rng2 = torch.arange(10, 18, device=x.device)
            tmp = x * rng1
            return tmp, tmp + rng2

        compiled = self._compile(fn)

        x = torch.randn(8, 8, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertEqual(r, e)

    def test_dtype_bitcast(self):
        # [FUNCTION DOC] Reinterpreting float32 data as int32 matches eager for a
        # small size, plus a larger size off CUDA.
        """Test dtype bitcast (view tensor as different dtype)."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Bitcast view to int32.
            # View float32 tensor as int32 (same byte size)
            return x.view(torch.int32)

        sizes = [128]
        if self.DEVICE != "cuda":
            sizes.append(2048)
        for size in sizes:
            with self.subTest(size=size):
                compiled = self._compile(fn)
                x = torch.randn(size, device=self.DEVICE, dtype=torch.float32)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def test_dtype_bitcast_float16_to_int16(self):
        # [FUNCTION DOC] Reinterpreting float16 data as int16 matches eager.
        """Test dtype bitcast from float16 to int16."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Bitcast view to int16.
            return x.view(torch.int16)

        compiled = self._compile(fn)

        x = torch.randn(128, device=self.DEVICE, dtype=torch.float16)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_warpgroup_size_exact_128(self):
        # [FUNCTION DOC] Addition on exactly 128 elements (one warpgroup) matches
        # eager.
        """Test with exactly 128 elements (1 warpgroup)."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return a + b

        compiled = self._compile(fn)

        # 128 = WARPGROUP_SIZE (exactly 1 warpgroup)
        a = torch.randn(128, device=self.DEVICE)
        b = torch.randn(128, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_multiple_256(self):
        # [FUNCTION DOC] Multiply-add on 256 elements matches eager.
        """Test with 256 elements (multiple of 128)."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Product of the inputs plus the first input.
            return a * b + a

        compiled = self._compile(fn)

        # 256 = 2 * WARPGROUP_SIZE
        a = torch.randn(256, device=self.DEVICE)
        b = torch.randn(256, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_multiple_512(self):
        # [FUNCTION DOC] Add with doubling on 512 elements matches eager.
        """Test with 512 elements (multiple of 128)."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: First input plus twice the second.
            return a + b * 2

        compiled = self._compile(fn)

        # 512 = 4 * WARPGROUP_SIZE
        a = torch.randn(512, device=self.DEVICE)
        b = torch.randn(512, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_below_128(self):
        # [FUNCTION DOC] Addition on 100 elements (below the warpgroup size,
        # padded for GPU) matches eager.
        """Test with fewer than 128 elements.

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return a + b

        compiled = self._compile(fn)

        # 100 < 128, automatic padding aligns to 128
        a = torch.randn(100, device=self.DEVICE)
        b = torch.randn(100, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_non_aligned(self):
        # [FUNCTION DOC] Multiplication on 200 elements (not a multiple of 128,
        # padded for GPU) matches eager.
        """Test with non-aligned size (not a multiple of 128).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Multiplies the two inputs.
            return a * b

        compiled = self._compile(fn)

        # 200 is not a multiple of 128, automatic padding aligns to 256
        a = torch.randn(200, device=self.DEVICE)
        b = torch.randn(200, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_128x128(self):
        # [FUNCTION DOC] Addition of 2D tensors, 128 by 128 plus 2048 by 2048 off
        # CUDA, matches eager.
        """Test 2D tensor with 128x128 and tiling-exercising sizes."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return x + y

        shapes = [(128, 128)]
        if self.DEVICE != "cuda":
            shapes.append((2048, 2048))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                y = torch.randn(shape, device=self.DEVICE)
                result = compiled(x, y)
                expected = fn(x, y)
                self.assertEqual(result, expected)

    def test_warpgroup_size_small_tensor(self):
        # [FUNCTION DOC] Addition on 64 elements (smaller than a warpgroup)
        # matches eager.
        """Test with very small tensor (less than warpgroup size).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return a + b

        compiled = self._compile(fn)

        # Small tensor, automatic padding aligns to 128
        a = torch.randn(64, device=self.DEVICE)
        b = torch.randn(64, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_non_aligned_10x10(self):
        # [FUNCTION DOC] Addition of 10 by 10 tensors (100 elements, unaligned)
        # matches eager.
        """Test 2D tensor with 10x10 = 100 elements (not multiple of 128).

        Mosaic GPU requires total tensor size to be multiples of 128.
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return x + y

        compiled = self._compile(fn)

        # 10x10 = 100 elements, automatic padding aligns to 128
        x = torch.randn(10, 10, device=self.DEVICE)
        y = torch.randn(10, 10, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_non_aligned_15x15(self):
        # [FUNCTION DOC] Multiply-add of 15 by 15 tensors (225 elements) matches
        # eager.
        """Test 2D tensor with 15x15 = 225 elements (not multiple of 128).

        Mosaic GPU requires total tensor size to be multiples of 128.
        Automatic padding aligns tensors to 256 for GPU.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Product of the inputs plus the first input.
            return x * y + x

        compiled = self._compile(fn)

        # 15x15 = 225 elements, automatic padding aligns to 256
        x = torch.randn(15, 15, device=self.DEVICE)
        y = torch.randn(15, 15, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_3d_non_aligned(self):
        # [FUNCTION DOC] Addition of 5 by 5 by 5 tensors (125 elements) matches
        # eager.
        """Test 3D tensor with non-aligned size (5x5x5 = 125 elements).

        Mosaic GPU requires total tensor size to be multiples of 128.
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return x + y

        compiled = self._compile(fn)

        # 5x5x5 = 125 elements, automatic padding aligns to 128
        x = torch.randn(5, 5, 5, device=self.DEVICE)
        y = torch.randn(5, 5, 5, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_3d_aligned(self):
        # [FUNCTION DOC] Addition of 4 by 4 by 8 tensors (exactly 128 elements)
        # matches eager.
        """Test 3D tensor with aligned size (4x4x8 = 128 elements)."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return x + y

        compiled = self._compile(fn)

        # 4x4x8 = 128 elements, exactly 1 warpgroup
        x = torch.randn(4, 4, 8, device=self.DEVICE)
        y = torch.randn(4, 4, 8, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_non_aligned_7x19(self):
        # [FUNCTION DOC] Subtraction of 7 by 19 tensors (133 elements) matches
        # eager.
        """Test 2D tensor with 7x19 = 133 elements (not multiple of 128).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        The backend automatically pads inputs and unpads outputs to handle this.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Subtracts the second input from the first.
            return x - y

        compiled = self._compile(fn)

        # 7x19 = 133 elements, just above 128 but not aligned
        x = torch.randn(7, 19, device=self.DEVICE)
        y = torch.randn(7, 19, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_4d_non_aligned(self):
        # [FUNCTION DOC] Addition of 4D tensors with 120 elements matches eager.
        """Test 4D tensor with non-aligned size (2x3x4x5 = 120 elements).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        The backend automatically pads inputs and unpads outputs to handle this.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return x + y

        compiled = self._compile(fn)

        # 2x3x4x5 = 120 elements, not a multiple of 128
        x = torch.randn(2, 3, 4, 5, device=self.DEVICE)
        y = torch.randn(2, 3, 4, 5, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_4d_aligned(self):
        # [FUNCTION DOC] Multiplication of 4D tensors with 256 elements matches
        # eager.
        """Test 4D tensor with aligned size (2x2x4x16 = 256 elements)."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies the two inputs.
            return x * y

        compiled = self._compile(fn)

        # 2x2x4x16 = 256 elements, multiple of 128
        x = torch.randn(2, 2, 4, 16, device=self.DEVICE)
        y = torch.randn(2, 2, 4, 16, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_aligned_16x8(self):
        # [FUNCTION DOC] First input plus twice the second on 16 by 8 tensors (128
        # elements) matches eager.
        """Test 2D tensor with 16x8 = 128 elements (exactly 1 warpgroup)."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: First input plus twice the second.
            return x + y * 2

        compiled = self._compile(fn)

        # 16x8 = 128 elements, exactly 1 warpgroup
        x = torch.randn(16, 8, device=self.DEVICE)
        y = torch.randn(16, 8, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_simple_mlp(self):
        # [FUNCTION DOC] Three chained matmul plus relu layers on 32 by 32 inputs
        # match eager.
        def fn(w_0, w_1, w_2, x):
            # [FUNCTION DOC] HELPER: Applies three matrix multiplies each followed by
            # relu.
            x = (x @ w_0).relu()
            x = (x @ w_1).relu()
            x = (x @ w_2).relu()
            return x

        compiled = self._compile(fn)

        ws = [torch.rand(32, 32, device=self.DEVICE) for _ in range(3)]
        x = torch.randn(32, 32, device=self.DEVICE)

        result = compiled(*ws, x)
        expected = fn(*ws, x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    def test_nanogpt(self):
        # [FUNCTION DOC] A minimal GPT-style block: single-head causal self-
        # attention with softmax, residual, GELU (tanh form) MLP and residual, on
        # 2D tensors with small random weights and an upper triangular negative-
        # infinity mask. Compares compiled and eager outputs. Skipped on CUDA
        # (axis reductions unsupported).
        """Test a minimal NanoGPT-style transformer block.

        Tests the core transformer operations used in GPT-style models:
        - Single-head self-attention with causal masking
        - MLP (feed-forward) block with GELU activation
        - Residual connections

        Uses 2D tensors (seq_len, embed_dim) to match the existing test_simple_mlp
        pattern and avoid 3D tensor shape issues in Pallas codegen.

        Note: Skipped on CUDA because JAX's Mosaic GPU backend doesn't support
        axis-based reductions (reduce with axis= parameter), which is required
        for softmax. Full reductions work, but partial reductions do not.
        """
        seq_len = 32
        n_embd = 64

        def transformer_block(x, w_q, w_k, w_v, w_proj, w_fc, w_out, mask):
            # [FUNCTION DOC] HELPER: The block forward pass: q, k, v projections,
            # scaled masked softmax attention, output projection with residual, then
            # the MLP with residual; returns the new activations.
            T, C = x.shape

            # === Self-Attention ===
            q = x @ w_q  # (T, C)
            k = x @ w_k
            v = x @ w_v

            # Scaled dot-product attention
            scale = 1.0 / (C**0.5)
            att = (q @ k.t()) * scale  # (T, T)
            att = att + mask  # Apply causal mask
            att = torch.softmax(att, dim=-1)
            attn_out = att @ v  # (T, C)

            # Output projection + residual
            x = x + (attn_out @ w_proj)

            # === MLP (Feed-Forward) ===
            h = x @ w_fc  # (T, 4C)
            # GELU activation (tanh approximation)
            h = 0.5 * h * (1.0 + torch.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))
            x = x + (h @ w_out)  # Residual + project back

            return x

        compiled = self._compile(transformer_block)

        # Initialize weights
        w_q = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_k = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_v = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_proj = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_fc = torch.randn(n_embd, 4 * n_embd, device=self.DEVICE) * 0.02
        w_out = torch.randn(4 * n_embd, n_embd, device=self.DEVICE) * 0.02

        # Causal mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.DEVICE),
            diagonal=1,
        )

        x = torch.randn(seq_len, n_embd, device=self.DEVICE)

        result = compiled(x, w_q, w_k, w_v, w_proj, w_fc, w_out, mask)
        expected = transformer_block(x, w_q, w_k, w_v, w_proj, w_fc, w_out, mask)
        self.assertEqual(result, expected)

    def _run_transformer_layer(
        self,
        seq_len,
        hidden_dim,
        num_heads,
        head_dim,
        ffn_dim,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        # [FUNCTION DOC] HELPER: Shared driver for transformer layer tests. Resets
        # dynamo, compiles a Llama-style layer (RMSNorm, multi-head attention with
        # causal mask, residual, RMSNorm, SwiGLU FFN, residual), builds weights
        # scaled by inverse square root of the hidden size plus unit norm weights
        # and a causal mask, and compares with eager under the given absolute and
        # relative tolerances.
        """Run a Llama-style transformer layer forward pass and verify correctness.

        Architecture: RMSNorm -> Multi-Head Attention -> Residual ->
                      RMSNorm -> SwiGLU FFN -> Residual
        """
        torch._dynamo.reset()

        def transformer_layer(
            x,
            rms_w1,
            rms_w2,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            mask,
        ):
            # [FUNCTION DOC] HELPER: The layer forward pass with per-head reshape and
            # permute for attention, causal mask broadcast over heads, and a SwiGLU
            # feed-forward; returns the updated activations.
            T, C = x.shape

            # Pre-attention RMSNorm
            variance = x.pow(2).mean(-1, keepdim=True)
            h = x * torch.rsqrt(variance + 1e-6) * rms_w1

            # Multi-head self-attention
            q = (h @ w_q).view(T, num_heads, head_dim).permute(1, 0, 2)  # (H, T, D)
            k = (h @ w_k).view(T, num_heads, head_dim).permute(1, 0, 2)
            v = (h @ w_v).view(T, num_heads, head_dim).permute(1, 0, 2)

            scale = 1.0 / (head_dim**0.5)
            att = (q @ k.transpose(-2, -1)) * scale  # (H, T, T)
            att = att + mask  # causal mask broadcasts (T, T) -> (H, T, T)
            att = torch.softmax(att, dim=-1)
            attn_out = (att @ v).permute(1, 0, 2).contiguous().view(T, C)  # (T, C)

            x = x + (attn_out @ w_o)

            # Pre-FFN RMSNorm
            variance = x.pow(2).mean(-1, keepdim=True)
            h = x * torch.rsqrt(variance + 1e-6) * rms_w2

            # SwiGLU FFN
            gate = torch.nn.functional.silu(h @ w_gate)
            up = h @ w_up
            x = x + ((gate * up) @ w_down)

            return x

        compiled = self._compile(transformer_layer)

        # Scale weights by 1/sqrt(fan_in) to keep activations O(1) and
        # avoid rsqrt amplification of reduction-order diffs.
        s = hidden_dim**-0.5
        w_q = torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s
        w_k = torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s
        w_v = torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s
        w_o = torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s
        w_gate = torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s
        w_up = torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s
        w_down = torch.randn(ffn_dim, hidden_dim, device=self.DEVICE) * s
        rms_w1 = torch.ones(hidden_dim, device=self.DEVICE)
        rms_w2 = torch.ones(hidden_dim, device=self.DEVICE)

        # Causal mask (T, T) - broadcasts over heads
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.DEVICE),
            diagonal=1,
        )

        x = torch.randn(seq_len, hidden_dim, device=self.DEVICE)

        result = compiled(
            x,
            rms_w1,
            rms_w2,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            mask,
        )
        expected = transformer_layer(
            x,
            rms_w1,
            rms_w2,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            mask,
        )
        self.assertEqual(result, expected, atol=atol, rtol=rtol)

    @skip_if_cuda
    def test_transformer_layer_tiny(self):
        # [FUNCTION DOC] Runs the single-layer driver with tiny sizes (sequence
        # 32, hidden 64, 2 heads, FFN 256). Skipped on CUDA.
        """Test full Llama-style transformer layer at tiny dimensions."""
        self._run_transformer_layer(
            seq_len=32,
            hidden_dim=64,
            num_heads=2,
            head_dim=32,
            ffn_dim=256,
        )

    @skip_if_cuda
    def test_transformer_layer_medium(self):
        # [FUNCTION DOC] Runs the single-layer driver at Llama-7B sizes with
        # loosened tolerances. Skipped on CUDA.
        """Test full Llama-style transformer layer at Llama-7B dimensions."""
        self._run_transformer_layer(
            seq_len=128,
            hidden_dim=4096,
            num_heads=32,
            head_dim=128,
            ffn_dim=11008,
            atol=2e-2,
            rtol=1e-2,
        )

    @skip_if_cuda
    def test_transformer_layer_large(self):
        # [FUNCTION DOC] Runs the single-layer driver at Llama-405B sizes with
        # loosened tolerances. Skipped on CUDA.
        """Test full Llama-style transformer layer at Llama-405B dimensions."""
        self._run_transformer_layer(
            seq_len=32,
            hidden_dim=16384,
            num_heads=128,
            head_dim=128,
            ffn_dim=53248,
            atol=2e-2,
            rtol=1e-2,
        )

    @skip_if_cuda
    def test_permute_contiguous_3d(self):
        # [FUNCTION DOC] Permute of a 3D tensor followed by contiguous matches
        # eager. Skipped on CUDA.
        """Test that permute + contiguous on a 3D tensor produces correct results."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Permutes dims one, zero, two and makes the result
            # contiguous.
            return x.permute(1, 0, 2).contiguous()

        compiled = self._compile(fn)
        x = torch.randn(2, 32, 32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    def test_transpose_contiguous_2d(self):
        # [FUNCTION DOC] Transpose plus contiguous on a 2 by 32 tensor matches
        # eager. Skipped on CUDA.
        """Test that transpose + contiguous on a 2D tensor compiles and runs."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Transposes and makes contiguous.
            return x.transpose(0, 1).contiguous()

        compiled = self._compile(fn)
        x = torch.randn(2, 32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    def test_permute_contiguous_2d_asymmetric(self):
        # [FUNCTION DOC] Transpose plus contiguous over several asymmetric 2D
        # shapes matches eager. Skipped on CUDA.
        """Test transpose+contiguous on asymmetric 2D shapes."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Transposes and makes contiguous.
            return x.transpose(0, 1).contiguous()

        compiled = self._compile(fn)
        for shape in [(10000, 2), (2, 10000), (8, 128), (128, 8), (3, 256)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_permute_contiguous_3d_all_perms(self):
        # [FUNCTION DOC] Three 3D permutations on a 2 by 1152 by 2048 tensor, each
        # compiled and compared with eager. Skipped on CUDA.
        """Test non-identity 3D permutations with distinct dim sizes."""
        all_perms = [
            (1, 0, 2),
            (0, 2, 1),
            (2, 1, 0),  # full-rank detection
        ]
        x = torch.randn(2, 1152, 2048, device=self.DEVICE)

        for perm in all_perms:
            with self.subTest(perm=perm):

                def fn(x, p=perm):
                    # [FUNCTION DOC] HELPER: Applies the permutation bound at definition time
                    # and makes contiguous.
                    return x.permute(*p).contiguous()

                compiled = self._compile(fn)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_permute_contiguous_3d_collapsed(self):
        # [FUNCTION DOC] Two 3D permutations needing collapsed-dimension
        # detection, on a 2 by 1152 by 2048 tensor, compared with eager. Skipped
        # on CUDA and TPU.
        """Test 3D permutations that require collapsed-dim detection."""
        all_perms = [
            (2, 0, 1),
            (1, 2, 0),
        ]
        x = torch.randn(2, 1152, 2048, device=self.DEVICE)

        for perm in all_perms:
            with self.subTest(perm=perm):

                def fn(x, p=perm):
                    # [FUNCTION DOC] HELPER: Applies the permutation bound at definition time
                    # and makes contiguous.
                    return x.permute(*p).contiguous()

                compiled = self._compile(fn)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_permute_contiguous_3d_large_notile(self):
        # [FUNCTION DOC] Three 3D permutations on a 2 by 1152 by 1024 tensor
        # (large, no tiling), compared with eager. Skipped on CUDA.
        """Test 3D perms with large shape but no tiling (dim=1024 exact fit)."""
        perms = [(1, 0, 2), (0, 2, 1), (2, 1, 0)]

        x = torch.randn(2, 1152, 1024, device=self.DEVICE)

        for perm in perms:
            with self.subTest(perm=perm):

                def fn(x, p=perm):
                    # [FUNCTION DOC] HELPER: Applies the permutation bound at definition time
                    # and makes contiguous.
                    return x.permute(*p).contiguous()

                compiled = self._compile(fn)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_permute_contiguous_3d_medium(self):
        # [FUNCTION DOC] Three 3D permutations on a 2 by 8 by 2048 tensor (tiling
        # on the last dimension), compared with eager. Skipped on CUDA.
        """Test 3D perms with medium shape that triggers tiling on last dim."""
        perms = [(1, 0, 2), (0, 2, 1), (2, 1, 0)]

        x = torch.randn(2, 8, 2048, device=self.DEVICE)

        for perm in perms:
            with self.subTest(perm=perm):

                def fn(x, p=perm):
                    # [FUNCTION DOC] HELPER: Applies the permutation bound at definition time
                    # and makes contiguous.
                    return x.permute(*p).contiguous()

                compiled = self._compile(fn)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_permute_contiguous_3d_small(self):
        # [FUNCTION DOC] Five 3D permutations on a small 2 by 8 by 16 tensor,
        # compared with eager; on TPU one permutation is skipped because of a
        # Mosaic shape cast bug. Skipped on CUDA.
        """Test 3D perms with small shapes that produce grid=(1,)."""
        all_perms = [
            (1, 0, 2),
            (0, 2, 1),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
        ]
        # (2,0,1) on (2,8,16) triggers a Mosaic "unsupported shape cast"
        # bug on TPU due to internal tile padding (128x16 -> 16x128).
        tpu_skip = {(2, 0, 1)} if self.DEVICE == "tpu" else set()
        x = torch.randn(2, 8, 16, device=self.DEVICE)

        for perm in all_perms:
            if perm in tpu_skip:
                continue
            with self.subTest(perm=perm):

                def fn(x, p=perm):
                    # [FUNCTION DOC] HELPER: Applies the permutation bound at definition time
                    # and makes contiguous.
                    return x.permute(*p).contiguous()

                compiled = self._compile(fn)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_permute_contiguous_4d(self):
        # [FUNCTION DOC] All 23 non-identity 4D permutations, each on a shape
        # chosen to be large in the last or first dimension, compared with eager.
        # Skipped on CUDA and TPU.
        """Test all 23 non-identity 4D permutations with multi-tile grids."""
        all_perms = [
            (0, 1, 3, 2),
            (0, 2, 1, 3),
            (0, 2, 3, 1),
            (0, 3, 1, 2),
            (0, 3, 2, 1),
            (1, 0, 2, 3),
            (1, 0, 3, 2),
            (1, 2, 0, 3),
            (1, 2, 3, 0),
            (1, 3, 0, 2),
            (1, 3, 2, 0),
            (2, 0, 1, 3),
            (2, 0, 3, 1),
            (2, 1, 0, 3),
            (2, 1, 3, 0),
            (2, 3, 0, 1),
            (2, 3, 1, 0),
            (3, 0, 1, 2),
            (3, 0, 2, 1),
            (3, 1, 0, 2),
            (3, 1, 2, 0),
            (3, 2, 0, 1),
            (3, 2, 1, 0),
        ]
        needs_large_first = {
            (0, 3, 2, 1),
            (1, 3, 0, 2),
            (1, 3, 2, 0),
            (2, 3, 1, 0),
            (3, 0, 2, 1),
            (3, 1, 0, 2),
            (3, 1, 2, 0),
            (3, 2, 0, 1),
            (3, 2, 1, 0),
        }
        shapes = {
            "large_last": (2, 4, 128, 2048),
            "large_first": (1152, 1152, 2, 4),
        }

        for perm in all_perms:
            key = "large_first" if perm in needs_large_first else "large_last"
            shape = shapes[key]
            with self.subTest(perm=perm, shape=shape):
                x = torch.randn(*shape, device=self.DEVICE)

                def fn(x, p=perm):
                    # [FUNCTION DOC] HELPER: Applies the permutation bound at definition time
                    # and makes contiguous.
                    return x.permute(*p).contiguous()

                compiled = self._compile(fn)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def _run_transformer(
        self,
        num_layers,
        seq_len,
        hidden_dim,
        num_heads,
        head_dim,
        ffn_dim,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        # [FUNCTION DOC] HELPER: Driver for multi-layer transformer tests. Resets
        # dynamo, compiles a function looping over layers with nine parameters
        # each, builds unit norm weights and scaled random matrices plus a causal
        # mask, and compares with eager under the given tolerances.
        """Run a multi-layer Llama-style transformer and verify correctness."""
        torch._dynamo.reset()

        def transformer(x, mask, *layer_params):
            # [FUNCTION DOC] HELPER: Applies each layer in turn, pulling its nine
            # parameters from the flat argument list by offset, and returns the final
            # activations.
            T, C = x.shape
            params_per_layer = (
                9  # rms_w1, rms_w2, w_q, w_k, w_v, w_o, w_gate, w_up, w_down
            )

            for i in range(num_layers):
                offset = i * params_per_layer
                rms_w1 = layer_params[offset]
                rms_w2 = layer_params[offset + 1]
                w_q = layer_params[offset + 2]
                w_k = layer_params[offset + 3]
                w_v = layer_params[offset + 4]
                w_o = layer_params[offset + 5]
                w_gate = layer_params[offset + 6]
                w_up = layer_params[offset + 7]
                w_down = layer_params[offset + 8]

                # Pre-attention RMSNorm
                variance = x.pow(2).mean(-1, keepdim=True)
                h = x * torch.rsqrt(variance + 1e-6) * rms_w1

                # Multi-head self-attention
                q = (h @ w_q).view(T, num_heads, head_dim).permute(1, 0, 2)
                k = (h @ w_k).view(T, num_heads, head_dim).permute(1, 0, 2)
                v = (h @ w_v).view(T, num_heads, head_dim).permute(1, 0, 2)

                scale = 1.0 / (head_dim**0.5)
                att = (q @ k.transpose(-2, -1)) * scale
                att = att + mask
                att = torch.softmax(att, dim=-1)
                attn_out = (att @ v).permute(1, 0, 2).contiguous().view(T, C)

                x = x + (attn_out @ w_o)

                # Pre-FFN RMSNorm
                variance = x.pow(2).mean(-1, keepdim=True)
                h = x * torch.rsqrt(variance + 1e-6) * rms_w2

                # SwiGLU FFN
                gate = torch.nn.functional.silu(h @ w_gate)
                up = h @ w_up
                x = x + ((gate * up) @ w_down)

            return x

        compiled = self._compile(transformer)

        s = hidden_dim**-0.5
        all_params = []
        for _ in range(num_layers):
            all_params.extend(
                [
                    torch.ones(hidden_dim, device=self.DEVICE),  # rms_w1
                    torch.ones(hidden_dim, device=self.DEVICE),  # rms_w2
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,  # w_q
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,  # w_k
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,  # w_v
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,  # w_o
                    torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s,  # w_gate
                    torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s,  # w_up
                    torch.randn(ffn_dim, hidden_dim, device=self.DEVICE) * s,  # w_down
                ]
            )

        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.DEVICE),
            diagonal=1,
        )
        x = torch.randn(seq_len, hidden_dim, device=self.DEVICE)

        result = compiled(x, mask, *all_params)
        expected = transformer(x, mask, *all_params)
        self.assertEqual(result, expected, atol=atol, rtol=rtol)

    @skip_if_cuda
    def test_transformer_tiny(self):
        # [FUNCTION DOC] Four-layer transformer at tiny sizes with loose
        # tolerances. Skipped on CUDA.
        """Test a 4-layer Llama-style transformer at tiny dimensions."""
        self._run_transformer(
            num_layers=4,
            seq_len=32,
            hidden_dim=64,
            num_heads=2,
            head_dim=32,
            ffn_dim=256,
            atol=1e-2,
            rtol=1e-2,
        )

    @skip_if_cuda
    def test_transformer_medium(self):
        # [FUNCTION DOC] Four-layer transformer at Llama-7B-like sizes with loose
        # tolerances. Skipped on CUDA.
        """Test a 4-layer transformer at Llama-7B-like dimensions."""
        self._run_transformer(
            num_layers=4,
            seq_len=128,
            hidden_dim=4096,
            num_heads=32,
            head_dim=128,
            ffn_dim=11008,
            atol=0.1,
            rtol=1e-2,
        )

    def _run_transformer_lm(
        self,
        num_layers,
        seq_len,
        vocab_size,
        hidden_dim,
        num_heads,
        head_dim,
        ffn_dim,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        # [FUNCTION DOC] HELPER: Driver for a full language model test. Resets
        # dynamo, compiles an embedding lookup, the layer stack, a final RMSNorm
        # and a vocabulary projection, builds random parameters and a causal mask,
        # and compares logits with eager.
        """Run a full Llama-style LM (embedding + layers + norm + lm_head)."""
        torch._dynamo.reset()

        def transformer_lm(
            token_ids, embed_table, final_rms_w, lm_head_w, mask, *layer_params
        ):
            # [FUNCTION DOC] HELPER: Embeds the token ids by table lookup, applies
            # each layer using parameters taken by offset, then the final norm and the
            # projection; returns logits.
            # Token embedding via gather
            x = embed_table[token_ids]  # (T, C)
            T, C = x.shape
            params_per_layer = 9

            for i in range(num_layers):
                offset = i * params_per_layer
                rms_w1 = layer_params[offset]
                rms_w2 = layer_params[offset + 1]
                w_q = layer_params[offset + 2]
                w_k = layer_params[offset + 3]
                w_v = layer_params[offset + 4]
                w_o = layer_params[offset + 5]
                w_gate = layer_params[offset + 6]
                w_up = layer_params[offset + 7]
                w_down = layer_params[offset + 8]

                # Pre-attention RMSNorm
                variance = x.pow(2).mean(-1, keepdim=True)
                h = x * torch.rsqrt(variance + 1e-6) * rms_w1

                # Multi-head self-attention
                q = (h @ w_q).view(T, num_heads, head_dim).permute(1, 0, 2)
                k = (h @ w_k).view(T, num_heads, head_dim).permute(1, 0, 2)
                v = (h @ w_v).view(T, num_heads, head_dim).permute(1, 0, 2)

                scale = 1.0 / (head_dim**0.5)
                att = (q @ k.transpose(-2, -1)) * scale
                att = att + mask
                att = torch.softmax(att, dim=-1)
                attn_out = (att @ v).permute(1, 0, 2).contiguous().view(T, C)

                x = x + (attn_out @ w_o)

                # Pre-FFN RMSNorm
                variance = x.pow(2).mean(-1, keepdim=True)
                h = x * torch.rsqrt(variance + 1e-6) * rms_w2

                # SwiGLU FFN
                gate = torch.nn.functional.silu(h @ w_gate)
                up = h @ w_up
                x = x + ((gate * up) @ w_down)

            # Final RMSNorm
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + 1e-6) * final_rms_w

            # LM head
            logits = x @ lm_head_w  # (T, V)
            return logits

        compiled = self._compile(transformer_lm)

        s = hidden_dim**-0.5
        embed_table = torch.randn(vocab_size, hidden_dim, device=self.DEVICE) * s
        final_rms_w = torch.ones(hidden_dim, device=self.DEVICE)
        lm_head_w = torch.randn(hidden_dim, vocab_size, device=self.DEVICE) * s

        all_params = []
        for _ in range(num_layers):
            all_params.extend(
                [
                    torch.ones(hidden_dim, device=self.DEVICE),
                    torch.ones(hidden_dim, device=self.DEVICE),
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s,
                    torch.randn(ffn_dim, hidden_dim, device=self.DEVICE) * s,
                ]
            )

        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.DEVICE),
            diagonal=1,
        )
        token_ids = torch.randint(0, vocab_size, (seq_len,), device=self.DEVICE)

        result = compiled(
            token_ids, embed_table, final_rms_w, lm_head_w, mask, *all_params
        )
        expected = transformer_lm(
            token_ids, embed_table, final_rms_w, lm_head_w, mask, *all_params
        )
        self.assertEqual(result, expected, atol=atol, rtol=rtol)

    @unittest.skip("numerical mismatch in embedding + RMSNorm fusion")
    def test_transformer_lm_tiny(self):
        # [FUNCTION DOC] Full language model at tiny sizes; currently
        # unconditionally skipped due to a numerical mismatch in embedding plus
        # RMSNorm fusion.
        """Test a full LM (embed + 4 layers + norm + lm_head) at tiny dims."""
        self._run_transformer_lm(
            num_layers=4,
            seq_len=32,
            vocab_size=256,
            hidden_dim=64,
            num_heads=2,
            head_dim=32,
            ffn_dim=256,
        )

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_embedding_lookup(self):
        # [FUNCTION DOC] Embedding table lookup by integer token ids matches
        # eager. Skipped on CUDA (scalar prefetch unsupported).
        """Test simple embedding table lookup (integer indexing)."""

        def fn(token_ids, embed_table):
            # [FUNCTION DOC] HELPER: Indexes the table with the token ids.
            return embed_table[token_ids]

        compiled = self._compile(fn)
        embed_table = torch.randn(256, 64, device=self.DEVICE)
        token_ids = torch.randint(0, 256, (32,), device=self.DEVICE)
        result = compiled(token_ids, embed_table)
        expected = fn(token_ids, embed_table)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_basic(self):
        # [FUNCTION DOC] Plain table row lookup through indirect access matches
        # eager. Skipped on CUDA.
        """Test bare embedding lookup via indirect access detection."""

        def fn(indices, table):
            # [FUNCTION DOC] HELPER: Indexes the table with the indices.
            return table[indices]

        compiled = self._compile(fn)
        table = torch.randn(256, 64, device=self.DEVICE)
        indices = torch.randint(0, 256, (32,), device=self.DEVICE)
        result = compiled(indices, table)
        expected = fn(indices, table)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_non_128_dim(self):
        # [FUNCTION DOC] Row lookup from a table whose width is 100, not a
        # multiple of 128. Skipped on CUDA.
        """Test indirect access with D not divisible by 128."""

        def fn(indices, table):
            # [FUNCTION DOC] HELPER: Indexes the table with the indices.
            return table[indices]

        compiled = self._compile(fn)
        table = torch.randn(256, 100, device=self.DEVICE)
        indices = torch.randint(0, 256, (32,), device=self.DEVICE)
        result = compiled(indices, table)
        expected = fn(indices, table)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_large_vocab(self):
        # [FUNCTION DOC] Row lookup from a 2048 by 128 table with 64 indices.
        # Skipped on CUDA.
        """Test indirect access with larger vocabulary."""

        def fn(indices, table):
            # [FUNCTION DOC] HELPER: Indexes the table with the indices.
            return table[indices]

        compiled = self._compile(fn)
        table = torch.randn(2048, 128, device=self.DEVICE)
        indices = torch.randint(0, 2048, (64,), device=self.DEVICE)
        result = compiled(indices, table)
        expected = fn(indices, table)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_fused_add(self):
        # [FUNCTION DOC] Row lookup fused with adding a per-row bias matrix.
        # Skipped on CUDA.
        """Test embedding + pointwise add fused."""

        def fn(indices, table, bias):
            # [FUNCTION DOC] HELPER: Looks up rows and adds the bias.
            return table[indices] + bias

        compiled = self._compile(fn)
        table = torch.randn(256, 64, device=self.DEVICE)
        indices = torch.randint(0, 256, (32,), device=self.DEVICE)
        bias = torch.randn(32, 64, device=self.DEVICE)
        result = compiled(indices, table, bias)
        expected = fn(indices, table, bias)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_fused_mul(self):
        # [FUNCTION DOC] Row lookup fused with multiplication by a broadcast
        # vector. Skipped on CUDA.
        """Test embedding + pointwise multiply with broadcast."""

        def fn(indices, table, scale):
            # [FUNCTION DOC] HELPER: Looks up rows and multiplies by the scale.
            return table[indices] * scale

        compiled = self._compile(fn)
        table = torch.randn(256, 64, device=self.DEVICE)
        indices = torch.randint(0, 256, (32,), device=self.DEVICE)
        scale = torch.randn(64, device=self.DEVICE)
        result = compiled(indices, table, scale)
        expected = fn(indices, table, scale)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_fused_chain(self):
        # [FUNCTION DOC] Row lookup followed by an add and a multiply. Skipped on
        # CUDA.
        """Test embedding + chained add and multiply."""

        def fn(indices, table, bias, scale):
            # [FUNCTION DOC] HELPER: Looks up rows, adds bias, then multiplies by the
            # scale.
            x = table[indices] + bias
            return x * scale

        compiled = self._compile(fn)
        table = torch.randn(256, 64, device=self.DEVICE)
        indices = torch.randint(0, 256, (32,), device=self.DEVICE)
        bias = torch.randn(32, 64, device=self.DEVICE)
        scale = torch.randn(64, device=self.DEVICE)
        result = compiled(indices, table, bias, scale)
        expected = fn(indices, table, bias, scale)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_nn_embedding(self):
        # [FUNCTION DOC] The functional embedding op through the indirect access
        # path matches eager. Skipped on CUDA.
        """Test nn.functional.embedding path through indirect access."""

        def fn(indices, weight):
            # [FUNCTION DOC] HELPER: Calls the functional embedding with the indices
            # and weight.
            return torch.nn.functional.embedding(indices, weight)

        compiled = self._compile(fn)
        weight = torch.randn(256, 64, device=self.DEVICE)
        indices = torch.randint(0, 256, (32,), device=self.DEVICE)
        result = compiled(indices, weight)
        expected = fn(indices, weight)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_single_token(self):
        # [FUNCTION DOC] Row lookup with a single index. Skipped on CUDA.
        """Test indirect access with seq=1."""

        def fn(indices, table):
            # [FUNCTION DOC] HELPER: Indexes the table with the indices.
            return table[indices]

        compiled = self._compile(fn)
        table = torch.randn(256, 64, device=self.DEVICE)
        indices = torch.randint(0, 256, (1,), device=self.DEVICE)
        result = compiled(indices, table)
        expected = fn(indices, table)
        self.assertEqual(result, expected)

    @skip_if_cuda(reason="scalar prefetch not supported in Pallas GPU (Mosaic) backend")
    def test_indirect_access_duplicate_indices(self):
        # [FUNCTION DOC] Row lookup with repeated indices in an eight-entry index
        # tensor. Skipped on CUDA.
        """Test indirect access with repeated indices."""

        def fn(indices, table):
            # [FUNCTION DOC] HELPER: Indexes the table with the indices.
            return table[indices]

        compiled = self._compile(fn)
        table = torch.randn(256, 64, device=self.DEVICE)
        indices = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3], device=self.DEVICE)
        result = compiled(indices, table)
        expected = fn(indices, table)
        self.assertEqual(result, expected)

    @skip_if_cuda
    def test_transformer_with_final_norm_and_lm_head(self):
        # [FUNCTION DOC] Four-layer transformer on given activations (no
        # embedding) followed by a final RMSNorm and vocabulary projection, built
        # inline with fixed small sizes and compared with eager. Resets dynamo
        # first. Skipped on CUDA.
        """Test multi-layer transformer + final RMSNorm + LM head (no embedding)."""
        torch._dynamo.reset()
        num_layers = 4
        seq_len = 32
        hidden_dim = 64
        num_heads = 2
        head_dim = 32
        ffn_dim = 256
        vocab_size = 256

        def transformer_with_head(x, final_rms_w, lm_head_w, mask, *layer_params):
            # [FUNCTION DOC] HELPER: Runs the layer loop with parameters taken by
            # offset, then the final norm and projection; returns logits.
            T, C = x.shape
            params_per_layer = 9
            for i in range(num_layers):
                offset = i * params_per_layer
                rms_w1 = layer_params[offset]
                rms_w2 = layer_params[offset + 1]
                w_q = layer_params[offset + 2]
                w_k = layer_params[offset + 3]
                w_v = layer_params[offset + 4]
                w_o = layer_params[offset + 5]
                w_gate = layer_params[offset + 6]
                w_up = layer_params[offset + 7]
                w_down = layer_params[offset + 8]
                variance = x.pow(2).mean(-1, keepdim=True)
                h = x * torch.rsqrt(variance + 1e-6) * rms_w1
                q = (h @ w_q).view(T, num_heads, head_dim).permute(1, 0, 2)
                k = (h @ w_k).view(T, num_heads, head_dim).permute(1, 0, 2)
                v = (h @ w_v).view(T, num_heads, head_dim).permute(1, 0, 2)
                scale = 1.0 / (head_dim**0.5)
                att = (q @ k.transpose(-2, -1)) * scale
                att = att + mask
                att = torch.softmax(att, dim=-1)
                attn_out = (att @ v).permute(1, 0, 2).contiguous().view(T, C)
                x = x + (attn_out @ w_o)
                variance = x.pow(2).mean(-1, keepdim=True)
                h = x * torch.rsqrt(variance + 1e-6) * rms_w2
                gate = torch.nn.functional.silu(h @ w_gate)
                up = h @ w_up
                x = x + ((gate * up) @ w_down)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + 1e-6) * final_rms_w
            return x @ lm_head_w

        compiled = self._compile(transformer_with_head)
        s = hidden_dim**-0.5
        final_rms_w = torch.ones(hidden_dim, device=self.DEVICE)
        lm_head_w = torch.randn(hidden_dim, vocab_size, device=self.DEVICE) * s
        all_params = []
        for _ in range(num_layers):
            all_params.extend(
                [
                    torch.ones(hidden_dim, device=self.DEVICE),
                    torch.ones(hidden_dim, device=self.DEVICE),
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, hidden_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s,
                    torch.randn(hidden_dim, ffn_dim, device=self.DEVICE) * s,
                    torch.randn(ffn_dim, hidden_dim, device=self.DEVICE) * s,
                ]
            )
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.DEVICE),
            diagonal=1,
        )
        x = torch.randn(seq_len, hidden_dim, device=self.DEVICE)
        result = compiled(x, final_rms_w, lm_head_w, mask, *all_params)
        expected = transformer_with_head(x, final_rms_w, lm_head_w, mask, *all_params)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_aligned_32x8(self):
        # [FUNCTION DOC] Addition of 32 by 8 tensors (256 elements) matches eager.
        """Test 2D tensor with 32x8 = 256 elements (2 warpgroups)."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs.
            return x + y

        compiled = self._compile(fn)

        # 32x8 = 256 elements, 2 warpgroups
        x = torch.randn(32, 8, device=self.DEVICE)
        y = torch.randn(32, 8, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)


if test_torchinductor.RUN_CPU and has_cpu_pallas():

    class PallasTestsCPU(PallasTestsMixin, TestCase):
        # [CLASS DOC] Concrete CPU test class: the mixin with the device set to
        # cpu. Defined only when CPU tests are enabled and CPU Pallas support
        # exists.
        DEVICE = "cpu"

    make_pallas(test_torchinductor.SweepInputsCpuTest)
    make_pallas(test_torchinductor.CpuTests)


if test_torchinductor.RUN_GPU and has_cuda_pallas():

    class PallasTestsCUDA(PallasTestsMixin, TestCase):
        # [CLASS DOC] Concrete CUDA test class: the mixin with the device set to
        # cuda. Defined only when GPU tests are enabled and CUDA Pallas support
        # exists.
        DEVICE = "cuda"

    make_pallas(test_torchinductor.SweepInputsGPUTest)
    # make_pallas(test_torchinductor.GPUTests)

if test_torchinductor.RUN_TPU and has_tpu_pallas():

    class PallasTestsTPU(PallasTestsMixin, TestCase):
        # [CLASS DOC] Concrete TPU test class: the mixin with the device set to
        # tpu. Defined only when TPU tests are enabled and TPU Pallas support
        # exists.
        DEVICE = "tpu"

    make_pallas(test_torchinductor.SweepInputsTpuTest)


if __name__ == "__main__":
    run_tests(needs="filelock")
