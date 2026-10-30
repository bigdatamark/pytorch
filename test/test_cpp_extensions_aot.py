# Owner(s): ["module: cpp-extensions"]
# [MODULE DOC] Tests for ahead-of-time built C++ extensions (torch_test_cpp_extension package):
# plain function and module bindings, CUDA/MPS/SYCL/cuBLAS/cuSOLVER extensions, pybind type
# casters, a fake MAIA device backend, a custom RNG generator extension, and a torch library
# operator. At import time it probes for pytest, imports the cpp, maia and rng extension modules
# (using pytest importorskip when pytest exists) and raises RuntimeError telling the user to run
# the test through the test runner if any import fails. Intended to be run via run_test.py.

import os
import re
import unittest
from itertools import repeat
from typing import get_args, get_origin, Union

import torch
import torch.backends.cudnn
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    skipIfTorchDynamo,
    TEST_XPU,
    xfailIfTorchDynamo,
)


try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# TODO: Rewrite these tests so that they can be collected via pytest without
# using run_test.py
try:
    if HAS_PYTEST:
        cpp_extension = pytest.importorskip("torch_test_cpp_extension.cpp")
        maia_extension = pytest.importorskip("torch_test_cpp_extension.maia")
        rng_extension = pytest.importorskip("torch_test_cpp_extension.rng")
    else:
        import torch_test_cpp_extension.cpp as cpp_extension
        import torch_test_cpp_extension.maia as maia_extension
        import torch_test_cpp_extension.rng as rng_extension
except ImportError as e:
    raise RuntimeError(
        "test_cpp_extensions_aot.py cannot be invoked directly. Run "
        "`python run_test.py -i test_cpp_extensions_aot_ninja` instead."
    ) from e


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionAOT(common.TestCase):
    # [CLASS DOC] Tests for ahead-of-time cpp extensions; run_test.py runs it both with and
    # without ninja, and many tests are skipped when the needed hardware, platform or ninja
    # build is unavailable.
    """Tests ahead-of-time cpp extensions

    NOTE: run_test.py's test_cpp_extensions_aot_ninja target
    also runs this test case, but with ninja enabled. If you are debugging
    a test failure here from the CI, check the logs for which target
    (test_cpp_extensions_aot_no_ninja vs test_cpp_extensions_aot_ninja)
    failed.
    """

    def test_extension_function(self):
        # [FUNCTION DOC] Calls the extension sigmoid-add on two random 4x4 tensors and compares
        # with the sum of their sigmoids; also checks the extension's pybind dtype cast maps
        # half to float32 as the math type.
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = cpp_extension.sigmoid_add(x, y)
        self.assertEqual(z, x.sigmoid() + y.sigmoid())
        # test pybind support torch.dtype cast.
        self.assertEqual(
            str(torch.float32), str(cpp_extension.get_math_type(torch.half))
        )

    def test_extension_module(self):
        # [FUNCTION DOC] Builds the extension matrix multiplier class with a 4 by 8 weight, and
        # checks its forward result for a random double matrix equals multiplying the stored
        # tensor by it.
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double)
        expected = mm.get().mm(weights)
        result = mm.forward(weights)
        self.assertEqual(expected, result)

    def test_backward(self):
        # [FUNCTION DOC] Runs forward of the extension module on a weight requiring grad,
        # backpropagates the sum and checks the gradients of both the input weights and the
        # stored tensor against the analytic matrix product with a tensor of ones.
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double, requires_grad=True)
        result = mm.forward(weights)
        result.sum().backward()
        tensor = mm.get()

        expected_weights_grad = tensor.t().mm(torch.ones([4, 4], dtype=torch.double))
        self.assertEqual(weights.grad, expected_weights_grad)

        expected_tensor_grad = torch.ones([4, 4], dtype=torch.double).mm(weights.t())
        self.assertEqual(tensor.grad, expected_tensor_grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_extension(self):
        # [FUNCTION DOC] Skipped without CUDA. Imports the CUDA extension, adds two zero cuda
        # tensors with its sigmoid-add and checks the cpu result is all ones (two times sigmoid
        # of zero).
        import torch_test_cpp_extension.cuda as cuda_extension

        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)

        z = cuda_extension.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not found")
    def test_mps_extension(self):
        # [FUNCTION DOC] Skipped without MPS. Computes an addition of 100000 random floats on
        # CPU and on MPS through the extension and checks the outputs agree.
        import torch_test_cpp_extension.mps as mps_extension

        tensor_length = 100000
        x = torch.randn(tensor_length, device="cpu", dtype=torch.float32)
        y = torch.randn(tensor_length, device="cpu", dtype=torch.float32)

        cpu_output = mps_extension.get_cpu_add_output(x, y)
        mps_output = mps_extension.get_mps_add_output(x.to("mps"), y.to("mps"))

        self.assertEqual(cpu_output, mps_output.to("cpu"))

    @unittest.skipIf(not TEST_XPU, "XPU not found")
    @unittest.skipIf(
        os.getenv("USE_NINJA", "0") == "0",
        "sycl extension requires ninja to build",
    )
    def test_sycl_extension(self):
        # [FUNCTION DOC] Skipped without XPU or when USE_NINJA is unset or zero. Imports the
        # SYCL extension, adds two zero XPU tensors with its sigmoid-add and checks the result
        # is all ones.
        import torch_test_cpp_extension.sycl as sycl_extension

        x = torch.zeros(100, device="xpu", dtype=torch.float32)
        y = torch.zeros(100, device="xpu", dtype=torch.float32)

        z = sycl_extension.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, "Windows not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cublas_extension(self):
        # [FUNCTION DOC] Skipped on ROCm, Windows, or without CUDA. Calls the extension no-op
        # cuBLAS function on a zero cuda tensor and checks the output equals the input.
        from torch_test_cpp_extension import cublas_extension

        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        z = cublas_extension.noop_cublas_function(x)
        self.assertEqual(z, x)

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, "Windows not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cusolver_extension(self):
        # [FUNCTION DOC] Skipped on ROCm, Windows, or without CUDA. Calls the extension no-op
        # cuSOLVER function on a zero cuda tensor and checks the output equals the input.
        from torch_test_cpp_extension import cusolver_extension

        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        z = cusolver_extension.noop_cusolver_function(x)
        self.assertEqual(z, x)

    @unittest.skipIf(IS_WINDOWS, "Not available on Windows")
    def test_no_python_abi_suffix_sets_the_correct_library_name(self):
        # [FUNCTION DOC] Skipped on Windows. Relies on a separate pip build of a test extension
        # with the no-python-abi-suffix option; walks the relative build folder for files ending
        # in so, and expects exactly one whose name is exactly the plain library name without
        # ABI suffix.
        # For this test, run_test.py will call
        # `python -m pip install . -v --no-build-isolation` in the
        # cpp_extensions/no_python_abi_suffix_test folder, where the
        # `BuildExtension` class has a `no_python_abi_suffix` option set to
        # `True`. This *should* mean that on Python 3, the produced shared
        # library does not have an ABI suffix like
        # "cpython-37m-x86_64-linux-gnu" before the library suffix, e.g. "so".
        root = os.path.join("cpp_extensions", "no_python_abi_suffix_test", "build")
        matches = [f for _, _, fs in os.walk(root) for f in fs if f.endswith("so")]
        self.assertEqual(len(matches), 1, msg=str(matches))
        self.assertEqual(matches[0], "no_python_abi_suffix_test.so", msg=str(matches))

    def test_optional(self):
        # [FUNCTION DOC] Checks the extension function taking an optional tensor returns true
        # for a tensor and false for none.
        has_value = cpp_extension.function_taking_optional(torch.ones(5))
        self.assertTrue(has_value)
        has_value = cpp_extension.function_taking_optional(None)
        self.assertFalse(has_value)

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, "Windows not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(
        os.getenv("USE_NINJA", "0") == "0",
        "cuda extension with dlink requires ninja to build",
    )
    def test_cuda_dlink_libs(self):
        # [FUNCTION DOC] Skipped on ROCm, Windows, without CUDA, or when USE_NINJA is unset or
        # zero. Imports the extension built with device linking and checks its add of two random
        # cuda tensors equals plain addition.
        from torch_test_cpp_extension import cuda_dlink

        a = torch.randn(8, dtype=torch.float, device="cuda")
        b = torch.randn(8, dtype=torch.float, device="cuda")
        ref = a + b
        test = cuda_dlink.add(a, b)
        self.assertEqual(test, ref)


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestPybindTypeCasters(common.TestCase):
    # [CLASS DOC] Verifies the Python types produced by C++ functions using custom pybind type
    # casters. The expected type of each function is parsed from the return annotation in its
    # docstring.
    """Pybind tests for ahead-of-time cpp extensions

    These tests verify the types returned from cpp code using custom type
    casters. By exercising pybind, we also verify that the type casters work
    properly.

    For each type caster in `torch/csrc/utils/pybind.h` we create a pybind
    function that takes no arguments and returns the type_caster type. The
    second argument to `PYBIND11_TYPE_CASTER` should be the type we expect to
    receive in python, in these tests we verify this at run-time.
    """

    @staticmethod
    def expected_return_type(func):
        # [FUNCTION DOC] Static helper: extracts the text after the arrow in the first line of
        # the pybind function docstring that carries the return annotation and evaluates it to
        # obtain the expected Python type (List and Tuple are imported locally so evaluation
        # works). Fails with an error if the docstring has no match.
        """
        Our Pybind functions have a signature of the form `() -> return_type`.
        """
        # Imports needed for the `eval` below.
        from typing import List, Tuple  # noqa: F401, UP035

        return eval(re.search("-> (.*)\n", func.__doc__).group(1))

    def check(self, func):
        # [FUNCTION DOC] HELPER: Calls the function with no arguments, and checks the value
        # against the expected return type: list and tuple annotations delegate to the element-
        # wise helpers, others use a plain instance check.
        val = func()
        expected = self.expected_return_type(func)
        origin = get_origin(expected)
        if origin is list:
            self.check_list(val, expected)
        elif origin is tuple:
            self.check_tuple(val, expected)
        else:
            self.assertIsInstance(val, expected)

    def check_list(self, vals, expected):
        # [FUNCTION DOC] HELPER: Asserts the value is a list and every element is an instance of
        # the annotation's element type.
        self.assertIsInstance(vals, list)
        list_type = get_args(expected)[0]
        for val in vals:
            self.assertIsInstance(val, list_type)

    def check_tuple(self, vals, expected):
        # [FUNCTION DOC] HELPER: Asserts the value is a tuple whose elements match the annotated
        # element types; a variable-length annotation repeats the first type for all elements.
        self.assertIsInstance(vals, tuple)
        tuple_types = get_args(expected)
        if tuple_types[1] is ...:
            tuple_types = repeat(tuple_types[0])
        for val, tuple_type in zip(vals, tuple_types):
            self.assertIsInstance(val, tuple_type)

    def check_union(self, funcs):
        # [FUNCTION DOC] HELPER: For a group of functions that return the same union type,
        # verifies all share one annotation which is a Union, and that across all functions each
        # member type (except SymInt, which is not required) is produced exactly once. Raises
        # AssertionError if a returned value matches no remaining type or if some types are left
        # unmatched.
        """Special handling for Union type casters.

        A single cpp type can sometimes be cast to different types in python.
        In these cases we expect to get exactly one function per python type.
        """
        # Verify that all functions have the same return type.
        union_type = {self.expected_return_type(f) for f in funcs}
        if len(union_type) != 1:
            raise AssertionError(f"expected 1 union type, got {len(union_type)}")
        union_type = union_type.pop()
        self.assertIs(Union, get_origin(union_type))
        # SymInt is inconvenient to test, so don't require it
        expected_types = set(get_args(union_type)) - {torch.SymInt}
        for func in funcs:
            val = func()
            for tp in expected_types:
                if isinstance(val, tp):
                    expected_types.remove(tp)
                    break
            else:
                raise AssertionError(f"{val} is not an instance of {expected_types}")
        self.assertFalse(
            expected_types, f"Missing functions for types {expected_types}"
        )

    def test_pybind_return_types(self):
        # [FUNCTION DOC] Runs the check on each pybind getter for complex, device, generator,
        # int array ref, memory format, storage, symbolic float, symbolic int array ref and
        # tensor values; the symbolic int getter is checked with the union helper. Each runs in
        # a subTest.
        functions = [
            cpp_extension.get_complex,
            cpp_extension.get_device,
            cpp_extension.get_generator,
            cpp_extension.get_intarrayref,
            cpp_extension.get_memory_format,
            cpp_extension.get_storage,
            cpp_extension.get_symfloat,
            cpp_extension.get_symintarrayref,
            cpp_extension.get_tensor,
        ]
        union_functions = [
            [cpp_extension.get_symint],
        ]
        for func in functions:
            with self.subTest(msg=f"check {func.__name__}"):
                self.check(func)
        for funcs in union_functions:
            with self.subTest(msg=f"check {[f.__name__ for f in funcs]}"):
                self.check_union(funcs)

    def test_pybind_layout_types(self):
        # [FUNCTION DOC] For each layout (strided, the sparse formats, mkldnn and jagged) passes
        # it through the extension roundtrip function and checks the same layout comes back.
        layouts = [
            torch.strided,
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
            torch._mkldnn,
            torch.jagged,
        ]
        for layout in layouts:
            with self.subTest(msg=f"check {layout}"):
                self.assertEqual(cpp_extension.roundtrip_layout(layout), layout)


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestMAIATensor(common.TestCase):
    # [CLASS DOC] Tests for the out-of-tree MAIA device backend extension; operator calls are
    # counted by the extension and read back with its get-test-int function.
    def test_unregistered(self):
        # [FUNCTION DOC] Checks that creating a tensor on the cpu works while the same call on
        # the maia device raises RuntimeError mentioning that it could not run.
        torch.arange(0, 10, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "Could not run"):
            torch.arange(0, 10, device="maia")

    @skipIfTorchDynamo("dynamo cannot model maia device")
    def test_zeros(self):
        # [FUNCTION DOC] Skipped under dynamo. Creates empty tensors on cpu and maia and checks
        # device, default dtype and an explicit int64 dtype; the extension's call counter must
        # stay zero.
        a = torch.empty(5, 5, device="cpu")
        self.assertEqual(a.device, torch.device("cpu"))

        b = torch.empty(5, 5, device="maia")
        self.assertEqual(b.device, torch.device("maia", 0))
        self.assertEqual(maia_extension.get_test_int(), 0)
        self.assertEqual(torch.get_default_dtype(), b.dtype)

        c = torch.empty((5, 5), dtype=torch.int64, device="maia")
        self.assertEqual(maia_extension.get_test_int(), 0)
        self.assertEqual(torch.int64, c.dtype)

    def test_add(self):
        # [FUNCTION DOC] Creates two empty maia tensors (counter stays zero), then adds them and
        # checks the extension's operator counter increased to one.
        a = torch.empty(5, 5, device="maia", requires_grad=True)
        self.assertEqual(maia_extension.get_test_int(), 0)

        b = torch.empty(5, 5, device="maia")
        self.assertEqual(maia_extension.get_test_int(), 0)

        a + b
        self.assertEqual(maia_extension.get_test_int(), 1)

    def test_conv_backend_override(self):
        # [FUNCTION DOC] Runs conv2d on maia tensors with a four-dimensional input and checks
        # the counter reaches two and the output shape follows batch and output channels. Then
        # takes the gradient with graph creation and checks the counter reaches three and the
        # gradient has the input shape.
        # To simplify tests, we use 4d input here to avoid doing view4d( which
        # needs more overrides) in _convolution.
        input = torch.empty(2, 4, 10, 2, device="maia", requires_grad=True)
        weight = torch.empty(6, 4, 2, 2, device="maia", requires_grad=True)
        bias = torch.empty(6, device="maia")

        # Make sure forward is overridden
        out = torch.nn.functional.conv2d(input, weight, bias, 2, 0, 1, 1)
        self.assertEqual(maia_extension.get_test_int(), 2)
        self.assertEqual(out.shape[0], input.shape[0])
        self.assertEqual(out.shape[1], weight.shape[0])

        # Make sure backward is overridden
        # Double backward is dispatched to _convolution_double_backward.
        # It is not tested here as it involves more computation/overrides.
        grad = torch.autograd.grad(out, input, out, create_graph=True)
        self.assertEqual(maia_extension.get_test_int(), 3)
        self.assertEqual(grad[0].shape, input.shape)

    def test_autocast_apis_for_maia_device(self):
        # [FUNCTION DOC] Checks the autocast dtype for maia is bfloat16 and that autocast is
        # reported as available for maia.
        # Default low-precision type in MAIA's autocast.
        fast_dtype = torch.get_autocast_dtype("maia")
        self.assertEqual(fast_dtype, torch.bfloat16)
        self.assertTrue(torch._C._is_autocast_available("maia"))

    @skipIfTorchDynamo(
        "dynamo cannot handle maia device. Output tensor may have wrong dtype."
    )
    def test_matmul_autocast_float16_precision(self):
        # [FUNCTION DOC] Skipped under dynamo. Within an autocast region on maia with float16, a
        # matmul of two float inputs runs with autocast enabled and yields a float16 result of
        # shape 2 by 2.
        # Ensure we can change low precision dtype.
        x = torch.empty((2, 4), dtype=torch.float, device="maia")
        w = torch.empty((4, 2), dtype=torch.float, device="maia")
        with torch.autocast(device_type="maia", dtype=torch.float16):
            self.assertTrue(torch.is_autocast_enabled("maia"))
            y = torch.ops.aten.matmul(x, w)
            self.assertEqual(y.dtype, torch.float16)
            self.assertEqual(y.shape, (2, 2))

    @skipIfTorchDynamo(
        "dynamo cannot handle maia device. Output tensor may have wrong dtype."
    )
    def test_matmul_autocast_default_precision(self):
        # [FUNCTION DOC] Skipped under dynamo. Within an autocast region on maia with default
        # dtype, a matmul of two float inputs yields a bfloat16 result of shape 2 by 2.
        # Use default lower precision dtype, bfloat16.
        x = torch.empty((2, 4), dtype=torch.float, device="maia")
        w = torch.empty((4, 2), dtype=torch.float, device="maia")
        with torch.autocast(device_type="maia"):
            self.assertTrue(torch.is_autocast_enabled("maia"))
            y = torch.ops.aten.matmul(x, w)
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertEqual(y.shape, (2, 2))


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestRNGExtension(common.TestCase):
    # [CLASS DOC] Tests for the custom RNG generator extension and its instance bookkeeping.
    def setUp(self):
        # [FUNCTION DOC] Calls the base class setup only.
        super().setUp()

    @xfailIfTorchDynamo
    def test_rng(self):
        # [FUNCTION DOC] Marked expected failure under dynamo. First verifies default and cpu
        # generator random values differ from 42. Then creates the extension test generator with
        # seed constant (instance count 1), checks aliases and the identity function keep count
        # at one, that values drawn from it are all 42, and that the count returns to zero only
        # after all references are deleted.
        fourty_two = torch.full((10,), 42, dtype=torch.int64)

        t = torch.empty(10, dtype=torch.int64).random_()
        self.assertNotEqual(t, fourty_two)

        gen = torch.Generator(device="cpu")
        t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
        self.assertNotEqual(t, fourty_two)

        self.assertEqual(rng_extension.getInstanceCount(), 0)
        gen = rng_extension.createTestCPUGenerator(42)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        copy = gen
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(gen, copy)
        copy2 = rng_extension.identity(copy)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(gen, copy2)
        t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(t, fourty_two)
        del gen
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        del copy
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        del copy2
        self.assertEqual(rng_extension.getInstanceCount(), 0)


@torch.testing._internal.common_utils.markDynamoStrictTest
@unittest.skipIf(not TEST_CUDA, "CUDA not found")
class TestTorchLibrary(common.TestCase):
    # [CLASS DOC] Tests for operators registered through the torch library API in an extension;
    # skipped without CUDA.
    def test_torch_library(self):
        # [FUNCTION DOC] Imports the torch library extension (registering a logical-and
        # operator), checks the operator through a plain Python function for all four boolean
        # combinations, then scripts that function and checks the same results and that the
        # graph mentions the operator name.
        import torch_test_cpp_extension.torch_library  # noqa: F401

        def f(a: bool, b: bool):
            # [FUNCTION DOC] HELPER: Calls the registered logical-and operator on two booleans
            # and returns its result.
            return torch.ops.torch_library.logical_and(a, b)

        self.assertTrue(f(True, True))
        self.assertFalse(f(True, False))
        self.assertFalse(f(False, True))
        self.assertFalse(f(False, False))
        s = torch.jit.script(f)
        self.assertTrue(s(True, True))
        self.assertFalse(s(True, False))
        self.assertFalse(s(False, True))
        self.assertFalse(s(False, False))
        self.assertIn("torch_library::logical_and", str(s.graph))


if __name__ == "__main__":
    common.run_tests()
