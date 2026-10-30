# Owner(s): ["module: tensor creation"]
# ruff: noqa: F841
# [MODULE DOC] Tests for tensor creation ops: eye, arange, linspace, logspace, empty, full,
# zeros, ones, the random creation functions, the like-functions, cat, roll, diag helpers,
# complex/polar, tril/triu indices, tensor construction from data, and DLPack interop.
# Defines device-generic test classes instantiated per device type at import time; view ops
# and most numpy interop live elsewhere. Also holds two small random input generators used by
# several tests.

import torch
import numpy as np

import sys
import math
import warnings
import unittest
from itertools import product, combinations, combinations_with_replacement, permutations
import random
import tempfile
from typing import Any

from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    do_test_empty_full,
    TEST_WITH_ROCM,
    suppress_warnings,
    torch_to_numpy_dtype_dict,
    numpy_to_torch_dtype_dict,
    numpy_to_torch_dtype,
    slowTest,
    set_default_dtype,
    set_default_tensor_type,
    TEST_SCIPY,
    IS_PPC,
    IS_WINDOWS,
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_S390X,
    IS_ARM64,
    parametrize,
    TEST_WITH_TORCHDYNAMO,
    xfailIfTorchDynamo,
)
from torch.testing._internal.common_device_type import (
    expectedFailureMeta, instantiate_device_type_tests, deviceCountAtLeast, onlyNativeDeviceTypes,
    onlyCPU, largeTensorTest, precisionOverride, dtypes,
    onlyCUDA, skipCPUIf, dtypesIfCUDA, dtypesIfCPU, skipMeta, onlyAccelerator)
from torch.testing._internal.common_dtype import (
    all_types_and_complex, all_types_and_complex_and, all_types_and, floating_and_complex_types, complex_types,
    floating_types, floating_and_complex_types_and, integral_types, integral_types_and, get_all_dtypes,
    float_to_corresponding_complex_type_map, all_types_complex_float8_and
)

from torch.utils.dlpack import to_dlpack

# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    # [FUNCTION DOC] HELPER: Build a random test tensor of the given shape, dtype and device. An
    # empty shape tuple yields an empty 1-D tensor. Floating and complex dtypes get scaled normal
    # values with about half zeroed; with_extremal also sprinkles nan and +/-inf. Bool gets
    # random True entries, other dtypes get random integers from 15 to 99. Returns the tensor.
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x


# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    # [FUNCTION DOC] HELPER: Return a tuple of dim random integers, each between min_size and
    # max_size inclusive, usable as a random tensor shape.
    shape = []
    for _ in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

# Test suite for tensor creation ops
#
# Includes creation functions like torch.eye, random creation functions like
#   torch.rand, and *like functions like torch.ones_like.
# DOES NOT INCLUDE view ops, which are tested in TestViewOps (currently in
#   test_torch.py) OR numpy interop (which is also still tested in test_torch.py)
#
# See https://pytorch.org/docs/main/torch.html#creation-ops

class TestTensorCreation(TestCase):
    # [CLASS DOC] Device-generic tests for tensor creation and joining ops (cat, roll, diag
    # helpers, complex, polar, and similar). exact_dtype is True so assertEqual checks
    # dtypes strictly.
    exact_dtype = True

    @onlyCPU
    @dtypes(torch.float)
    def test_diag_embed(self, device, dtype):
        # [FUNCTION DOC] Check diag_embed on a 2-D CPU float tensor against stacking per-row diag
        # results, for the default call and for a call with an offset and explicit dim1/dim2.
        x = torch.arange(3 * 4, dtype=dtype, device=device).view(3, 4)
        result = torch.diag_embed(x)
        expected = torch.stack([torch.diag(r) for r in x], 0)
        self.assertEqual(result, expected)

        result = torch.diag_embed(x, offset=1, dim1=0, dim2=2)
        expected = torch.stack([torch.diag(r, 1) for r in x], 1)
        self.assertEqual(result, expected)

    def test_cat_mem_overlap(self, device):
        # [FUNCTION DOC] Check that cat with an out argument that is an expanded (overlapping
        # memory) tensor raises a RuntimeError about an unsupported operation.
        x = torch.rand((1, 3), device=device).expand((6, 3))
        y = torch.rand((3, 3), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.cat([y, y], out=x)

    @onlyNativeDeviceTypes
    def test_vander(self, device):
        # [FUNCTION DOC] Check vander edge cases: an empty input with zero columns gives shape
        # (0, 0), a negative N raises RuntimeError, and a non-1-D input raises RuntimeError.
        x = torch.tensor([1, 2, 3, 5], device=device)

        self.assertEqual((0, 0), torch.vander(torch.tensor([]), 0).shape)

        with self.assertRaisesRegex(RuntimeError, "N must be non-negative."):
            torch.vander(x, N=-1)

        with self.assertRaisesRegex(RuntimeError, "x must be a one-dimensional tensor."):
            torch.vander(torch.stack((x, x)))

    @onlyNativeDeviceTypes
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.short, torch.int, torch.long,
            torch.float, torch.double,
            torch.cfloat, torch.cdouble)
    def test_vander_types(self, device, dtype):
        # [FUNCTION DOC] Compare vander against numpy for bool, uint8, integer, float and complex
        # dtypes over several inputs, column counts (including default) and both orderings. Uses
        # loose absolute tolerance and does not require equal dtypes.
        if dtype is torch.uint8:
            # Note: no negative uint8 values
            X = [[1, 2, 3, 5], [0, 1 / 3, 1, math.pi, 3 / 7]]
        elif dtype is torch.bool:
            # Note: see https://github.com/pytorch/pytorch/issues/37398
            # for why this is necessary.
            X = [[True, True, True, True], [False, True, True, True, True]]
        elif dtype in [torch.cfloat, torch.cdouble]:
            X = [[1 + 1j, 1 + 0j, 0 + 1j, 0 + 0j],
                 [2 + 2j, 3 + 2j, 4 + 3j, 5 + 4j]]
        else:
            X = [[1, 2, 3, 5], [-math.pi, 0, 1 / 3, 1, math.pi, 3 / 7]]

        N = [None, 0, 1, 3]
        increasing = [False, True]

        for x, n, inc in product(X, N, increasing):
            numpy_dtype = torch_to_numpy_dtype_dict[dtype]
            pt_x = torch.tensor(x, device=device, dtype=dtype)
            np_x = np.array(x, dtype=numpy_dtype)

            pt_res = torch.vander(pt_x, increasing=inc) if n is None else torch.vander(pt_x, n, inc)
            np_res = np.vander(np_x, n, inc)

            self.assertEqual(
                pt_res,
                torch.from_numpy(np_res),
                atol=1e-3,
                rtol=0,
                exact_dtype=False)

    def test_cat_all_dtypes_and_devices(self, device):
        # [FUNCTION DOC] For many dtypes (including half, bool, bfloat16, chalf and float8
        # variants), check that cat of a 2x2 tensor with itself along dim 0 and dim 1 gives the
        # expected values.
        for dt in all_types_and_complex_and(
            torch.half,
            torch.bool,
            torch.bfloat16,
            torch.chalf,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ):
            x = torch.tensor([[1, 2], [3, 4]], dtype=dt, device=device)

            expected1 = torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 0), expected1)

            expected2 = torch.tensor([[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 1), expected2)

    def test_fill_all_dtypes_and_devices(self, device):
        # [FUNCTION DOC] For many dtypes, fill small and large (10000 element) tensors in place
        # with a range of integer values (smaller range for 8-bit types) and check contents and
        # dtype remain correct.
        for dt in all_types_complex_float8_and(torch.half, torch.bool, torch.bfloat16, torch.chalf):
            for x in [torch.tensor((10, 10), dtype=dt, device=device),
                      torch.empty(10000, dtype=dt, device=device)]:  # large tensor
                numel = x.numel()
                bound_dtypes = (torch.uint8, torch.int8, torch.float8_e4m3fn,
                                torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)
                bound = 100 if dt in bound_dtypes else 2000
                for n in range(-bound, bound, bound // 10):
                    x.fill_(n)
                    self.assertEqual(x, torch.tensor([n] * numel, dtype=dt, device=device))
                    self.assertEqual(dt, x.dtype)

    def test_roll(self, device):
        # [FUNCTION DOC] Exercise roll: positive and negative shifts, multi-dimensional data,
        # full-loop rolls returning the original, non-contiguous input, no dims (flatten then
        # roll), multiple shifts and dims, error cases for empty or misaligned shifts and dims,
        # and bool and complex tensors.
        numbers = torch.arange(1, 9, device=device)

        single_roll = numbers.roll(1, 0)
        expected = torch.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device)
        self.assertEqual(single_roll, expected, msg=lambda msg: f"{msg}\n{single_roll} did not equal expected result")

        roll_backwards = numbers.roll(-2, 0)
        expected = torch.tensor([3, 4, 5, 6, 7, 8, 1, 2], device=device)
        self.assertEqual(roll_backwards, expected, msg=lambda msg: f"{msg}\n{roll_backwards} did not equal expected result")

        data = numbers.view(2, 2, 2)
        rolled = data.roll(1, 0)
        expected = torch.tensor([5, 6, 7, 8, 1, 2, 3, 4], device=device).view(2, 2, 2)
        self.assertEqual(expected, rolled, msg=lambda msg: f"{msg}\n{rolled} did not equal expected result: {expected}")

        data = data.view(2, 4)
        # roll a loop until back where started
        loop_rolled = data.roll(2, 0).roll(4, 1)
        self.assertEqual(data, loop_rolled, msg=lambda msg: f"{msg}\n{loop_rolled} did not equal the original: {data}")
        # multiple inverse loops
        self.assertEqual(data, data.roll(-20, 0).roll(-40, 1))
        self.assertEqual(torch.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device), numbers.roll(1, 0))

        # test non-contiguous
        # strided equivalent to numbers.as_strided(size=(4, 2), stride=(1, 4))
        strided = numbers.view(2, 4).transpose(0, 1)
        self.assertFalse(strided.is_contiguous(), "this test needs a non-contiguous tensor")
        expected = torch.tensor([4, 8, 1, 5, 2, 6, 3, 7]).view(4, 2)
        rolled = strided.roll(1, 0)
        self.assertEqual(expected, rolled,
                         msg=lambda msg: f"{msg}\nnon contiguous tensor rolled to {rolled} instead of {expected} ")

        # test roll with no dimension specified
        expected = numbers.roll(1, 0).view(2, 4)
        self.assertEqual(expected, data.roll(1), msg="roll with no dims should flatten and roll.")
        self.assertEqual(expected, data.roll(1, dims=None), msg="roll with no dims should flatten and roll.")

        # test roll over multiple dimensions
        expected = torch.tensor([[7, 8, 5, 6], [3, 4, 1, 2]], device=device)
        double_rolled = data.roll(shifts=(2, -1), dims=(1, 0))
        self.assertEqual(double_rolled, expected,
                         msg=lambda msg: f"{msg}\nshould be able to roll over two dimensions, got {double_rolled}")

        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=()))
        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=1))
        # shifts/dims should align
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1, 2), dims=(1,)))
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1,), dims=(1, 2)))

        # test bool tensor
        t = torch.zeros(6, dtype=torch.bool, device=device)
        t[0] = True
        t[3] = True
        self.assertEqual(torch.tensor([False, True, False, False, True, False]), t.roll(1, 0))

        # test complex tensor
        t = torch.tensor([1, 2 + 1j, 3.5, 4. + 2j, 5j, 6.], device=device)
        t[0] = 1 + 0.5j
        t[3] = 4.
        expected = torch.tensor([6., 1 + 0.5j, 2 + 1j, 3.5, 4., 5j], device=device)
        self.assertEqual(expected, t.roll(1, 0))

    def test_diagflat(self, device):
        # [FUNCTION DOC] Check diagflat against diag of the flattened input: 1-D input, an
        # offset, a multi-dimensional input, a non-contiguous input, and complex128 input.
        dtype = torch.float32
        # Basic sanity test
        x = torch.randn((100,), dtype=dtype, device=device)
        result = torch.diagflat(x)
        expected = torch.diag(x)
        self.assertEqual(result, expected)

        # Test offset
        x = torch.randn((100,), dtype=dtype, device=device)
        result = torch.diagflat(x, 17)
        expected = torch.diag(x, 17)
        self.assertEqual(result, expected)

        # Test where input has more than one dimension
        x = torch.randn((2, 3, 4), dtype=dtype, device=device)
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Noncontig input
        x = torch.randn((2, 3, 4), dtype=dtype, device=device).transpose(2, 0)
        self.assertFalse(x.is_contiguous())
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Complex number support
        result = torch.diagflat(torch.ones(4, dtype=torch.complex128))
        expected = torch.eye(4, dtype=torch.complex128)
        self.assertEqual(result, expected)

    def test_block_diag(self, device):
        # [FUNCTION DOC] Check block_diag against a manual reference for mixed 0-D, 1-D and 2-D
        # inputs, the single-tensor cases, the no-argument case (empty 1x0 on cpu), type
        # promotion over all pairs of dtypes, errors for inputs with more than 2 dims, and
        # (non-cpu) mixed-device inputs.
        def block_diag_workaround(*arrs):
            # [FUNCTION DOC] HELPER: Reference block_diag built by hand. Promotes each 1-D or 0-D
            # input to 2-D, allocates a zero matrix with summed shapes in the first input's
            # dtype, and copies each block along the diagonal. Returns that matrix.
            arrs_expanded = []
            for a in arrs:
                if a.dim() == 2:
                    arrs_expanded.append(a)
                elif a.dim() == 1:
                    arrs_expanded.append(a.expand(1, a.size(0)))
                elif a.dim() == 0:
                    arrs_expanded.append(a.expand(1, 1))
            shapes = torch.tensor([a.shape for a in arrs_expanded], device=device)
            out = torch.zeros(
                torch.sum(shapes, dim=0).tolist(),
                dtype=arrs_expanded[0].dtype,
                device=device
            )
            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                out[r:r + rr, c:c + cc] = arrs_expanded[i]
                r += rr
                c += cc
            return out

        tensors = [
            torch.rand((2, 2), device=device),
            torch.rand((2, 3), device=device),
            torch.rand(10, device=device),
            torch.rand((8, 1), device=device),
            torch.rand(1, device=device)[0]
        ]
        result = torch.block_diag(*tensors)
        result_check = block_diag_workaround(*tensors)
        self.assertEqual(result, result_check)

        tensor = torch.rand(1, device=device)[0]
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, 1)
        self.assertEqual(result, result_check)

        tensor = torch.rand(10, device=device)
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, tensor.size(0))
        self.assertEqual(result, result_check)

        result = torch.block_diag()
        result_check = torch.empty(1, 0, device=device)
        self.assertEqual(result, result_check)
        self.assertEqual(result.device.type, 'cpu')

        test_dtypes = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128
        ]
        # Test pairs of different dtypes
        for dtype1 in test_dtypes:
            for dtype2 in test_dtypes:
                a = torch.tensor(1, device=device, dtype=dtype1)
                b = torch.tensor(2, device=device, dtype=dtype2)
                result = torch.block_diag(a, b)
                result_dtype = torch.result_type(a, b)
                result_check = torch.tensor([[1, 0], [0, 2]], device=device, dtype=result_dtype)
                self.assertEqual(result, result_check)

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 1 has 3 dimensions"
        ):
            torch.block_diag(torch.tensor(5), torch.tensor([[[6]]]))

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 0 has 4 dimensions"
        ):
            torch.block_diag(torch.tensor([[[[6]]]]))

        if device != 'cpu':
            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "torch.block_diag: input tensors must all be on the same device."
                    " Input 0 is on device cpu and input 1 is on device "
                )
            ):
                torch.block_diag(torch.ones(2, 2).cpu(), torch.ones(2, 2, device=device))

    @skipCPUIf(TEST_WITH_TORCHDYNAMO, "test doesn't currently work with dynamo on CPU")
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    def test_block_diag_scipy(self, device):
        # [FUNCTION DOC] Compare block_diag with scipy block_diag on several nested-list inputs
        # (scalars, empty lists, mixed int/float/complex). Checks result dtypes of both (they
        # differ in float width) and that values match after casting scipy output to the torch
        # dtype. Skipped without scipy.
        import scipy.linalg
        scipy_tensors_list = [
            [
                1,
                [2],
                [],
                [3, 4, 5],
                [[], []],
                [[6], [7.3]]
            ],
            [
                [[1, 2], [3, 4]],
                [1]
            ],
            [
                [[4, 9], [7, 10]],
                [4.6, 9.12],
                [1j + 3]
            ],
            []
        ]

        expected_torch_types = [
            torch.float32,
            torch.int64,
            torch.complex64,
            torch.float32
        ]

        expected_scipy_types = [
            torch.float64,
            numpy_to_torch_dtype(np.array(0).dtype),
            torch.complex128,
            torch.float64
        ]

        for scipy_tensors, torch_type, scipy_type in zip(scipy_tensors_list, expected_torch_types, expected_scipy_types):
            torch_tensors = [torch.tensor(t, device=device) for t in scipy_tensors]
            torch_result = torch.block_diag(*torch_tensors)
            self.assertEqual(torch_result.dtype, torch_type)

            scipy_result = torch.tensor(
                scipy.linalg.block_diag(*scipy_tensors),
                device=device
            )
            self.assertEqual(scipy_result.dtype, scipy_type)
            scipy_result = scipy_result.to(torch_type)

            self.assertEqual(torch_result, scipy_result)

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.float32, torch.float64)
    def test_torch_complex(self, device, dtype):
        # [FUNCTION DOC] Check torch.complex builds a complex tensor from matching real and
        # imaginary half, float or double tensors, with the corresponding complex dtype.
        real = torch.tensor([1, 2], device=device, dtype=dtype)
        imag = torch.tensor([3, 4], device=device, dtype=dtype)
        z = torch.complex(real, imag)
        complex_dtype = float_to_corresponding_complex_type_map[dtype]
        self.assertEqual(torch.tensor([1.0 + 3.0j, 2.0 + 4.0j], dtype=complex_dtype), z)

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.float32, torch.float64)
    def test_torch_polar(self, device, dtype):
        # [FUNCTION DOC] Check torch.polar from magnitude and angle tensors against precomputed
        # complex values; half precision uses a looser tolerance.
        abs = torch.tensor([1, 2, -3, -4.5, 1, 1], device=device, dtype=dtype)
        angle = torch.tensor([math.pi / 2, 5 * math.pi / 4, 0, -11 * math.pi / 6, math.pi, -math.pi],
                             device=device, dtype=dtype)
        z = torch.polar(abs, angle)
        complex_dtype = float_to_corresponding_complex_type_map[dtype]
        # float16 loses precision in cos/sin and the interleaved storage; relax tol.
        atol, rtol = (1e-2, 1e-2) if dtype == torch.half else (1e-5, 1e-5)
        self.assertEqual(torch.tensor([1j, -1.41421356237 - 1.41421356237j, -3,
                                       -3.89711431703 - 2.25j, -1, -1],
                                      dtype=complex_dtype),
                         z, atol=atol, rtol=rtol)

    @onlyNativeDeviceTypes
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.complex64, torch.complex128, torch.bool)
    def test_torch_complex_floating_dtype_error(self, device, dtype):
        # [FUNCTION DOC] Check that complex and polar raise a RuntimeError when inputs are not
        # half, float or double (integer, complex or bool dtypes).
        for op in (torch.complex, torch.polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            error = r"Expected both inputs to be Half, Float or Double tensors but " \
                    r"got [A-Za-z]+ and [A-Za-z]+"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_same_dtype_error(self, device, dtype):
        # [FUNCTION DOC] Check that complex and polar raise a RuntimeError naming both dtypes
        # when the two inputs have different float dtypes.

        def dtype_name(dtype):
            # [FUNCTION DOC] HELPER: Return the error-message name, Float for float32 and
            # Double otherwise.
            return 'Float' if dtype == torch.float32 else 'Double'

        for op in (torch.complex, torch.polar):
            other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=other_dtype)
            error = f"Expected object of scalar type {dtype_name(dtype)} but got scalar type " \
                    f"{dtype_name(other_dtype)} for second argument"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_out_dtype_error(self, device, dtype):
        # [FUNCTION DOC] Check that complex and polar raise a RuntimeError when the out tensor
        # has a real dtype instead of the matching complex dtype.

        def dtype_name(dtype):
            # [FUNCTION DOC] HELPER: Return the error-message name, Float for float32 and
            # Double otherwise.
            return 'Float' if dtype == torch.float32 else 'Double'

        def complex_dtype_name(dtype):
            # [FUNCTION DOC] HELPER: Return the error-message name, ComplexFloat for complex64
            # and ComplexDouble otherwise.
            return 'ComplexFloat' if dtype == torch.complex64 else 'ComplexDouble'

        for op in (torch.complex, torch.polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            out = torch.zeros(2, device=device, dtype=dtype)
            expected_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            error = f"Expected object of scalar type {complex_dtype_name(expected_dtype)} but got scalar type " \
                    f"{dtype_name(dtype)} for argument 'out'"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b, out=out)

    def test_cat_empty_legacy(self, device):
        # [FUNCTION DOC] Legacy behavior: cat of a 4-D tensor with an empty 1-D tensor on either
        # side along dim 1 works and gives equal results; cat of two such empties returns the
        # empty tensor.
        # FIXME: this is legacy behavior and should be removed
        # when we support empty tensors with arbitrary sizes
        dtype = torch.float32

        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = torch.randn((0,), dtype=dtype, device=device)

        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = torch.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

    def test_cat_empty(self, device):
        # [FUNCTION DOC] Check cat of a 4-D tensor with a size-zero-in-dim-1 tensor on either
        # side gives equal results, and cat of two such empties returns the empty tensor.
        dtype = torch.float32

        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = torch.randn((4, 0, 32, 32), dtype=dtype, device=device)

        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = torch.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

    def test_concat_empty_list_error(self, device):
        # [FUNCTION DOC] Regression test: concat and concatenate of an empty list raise
        # ValueError with a message about needing a non-empty list of Tensors.
        # Regression test for https://github.com/pytorch/pytorch/issues/155306
        msg = "expected a non-empty list of Tensors"
        with self.assertRaisesRegex(ValueError, msg):
            torch.concat([], dim=0)
        with self.assertRaisesRegex(ValueError, msg):
            torch.concatenate([], dim=0)

    def test_cat_out(self, device):
        # [FUNCTION DOC] Check cat with out pointing at a slice of a larger tensor writes the
        # right values without disturbing neighbours, then check out being a contiguous or
        # non-contiguous narrowed view along dim 0 and dim 1 of a 3-D tensor.
        x = torch.zeros((0), device=device)
        y = torch.randn((4, 6), device=device)

        w = y.view(-1).clone()
        a = torch.cat([w[:2], w[4:6]])
        b = torch.cat([w[:2], w[4:6]], out=w[6:10])
        self.assertEqual(a, b)
        self.assertEqual(a, w[6:10])
        self.assertEqual(w[:6], y.view(-1)[:6])

        # Case:
        # Reference: https://github.com/pytorch/pytorch/issues/49878
        for dim in [0, 1]:
            x = torch.zeros((10, 5, 2), device=device)

            random_length = random.randint(1, 4)
            y = x.narrow(dim, 0, x.shape[dim] - random_length)
            val = torch.full_like(y[0], 3., device=device)

            if dim == 0:
                self.assertTrue(y.is_contiguous())
            else:
                self.assertFalse(y.is_contiguous())

            torch.cat((val[None],) * y.shape[0], dim=0, out=y)

            expected_y = torch.cat((val[None],) * y.shape[0], dim=0)
            expected_x = torch.zeros((10, 5, 2), device=device)
            if dim == 0:
                expected_x[:x.shape[dim] - random_length, :, :] = expected_y
            elif dim == 1:
                expected_x[:, :x.shape[dim] - random_length, :] = expected_y

            self.assertEqual(y, expected_y)
            self.assertEqual(x, expected_x)

    @dtypes(*all_types_and_complex(), torch.uint16, torch.uint32, torch.uint64)
    def test_cat_out_fast_path_dim0_dim1(self, device, dtype):
        # [FUNCTION DOC] Check cat fast paths on dim 0 and dim 1 for many dtypes: results with
        # and without out and with contiguous inputs agree, slices of tensor_split work, and for
        # non-integer dtypes the backward pass gives all-ones gradients (compared by magnitude
        # for complex).
        int_types = integral_types_and(torch.uint16, torch.uint32, torch.uint64)
        x = torch.zeros((0), device=device, dtype=dtype)
        if dtype in int_types:
            y = torch.randint(low=0, high=100, size=(4, 6), device=device, dtype=dtype)
        else:
            y = torch.randn((4, 6), device=device, dtype=dtype)
        # Test concat on dimension 0
        w = y.view(-1).clone()
        a = torch.cat([w[:2], w[4:6]])
        b = torch.cat([w[:2], w[4:6]], out=w[6:10])
        # Note that there is no guarantee that slicing here will result in
        # contiguous tensors
        self.assertEqual(a, b)
        self.assertEqual(a, w[6:10])
        self.assertEqual(w[:6], y.view(-1)[:6])
        # If inputs are contiguous tensors, then fast concat paths will be invoked
        a_fastcat = torch.cat([w[:2].contiguous(), w[4:6].contiguous()])
        self.assertEqual(a_fastcat, a)
        # Test concat on dimension 1
        w = y.clone()
        w_slices = torch.tensor_split(w, (2, 4), dim=1)
        # Note that the tensor in w_slices[] here may not be a contiguous
        # tensor and we need to make sure this is not broken by fast concat
        b = torch.cat([w_slices[0], w_slices[1]], dim=1)
        expected_b = torch.index_select(w, 1, torch.tensor([0, 1, 2, 3], device=device))
        self.assertEqual(b, expected_b)
        # If inputs are contiguous tensors, then fast concat paths will be invoked
        b_fastcat = torch.cat([w_slices[0].contiguous(), w_slices[1].contiguous()], dim=1)
        self.assertEqual(b_fastcat, expected_b)
        # Finally, we need to make sure backward is not broken
        # Integral types will not have grad
        if dtype not in int_types:
            a = torch.randn((4, 3), device=device, dtype=dtype, requires_grad=True)
            b = torch.randn((2, 3), device=device, dtype=dtype, requires_grad=True)
            c = torch.randn((5, 3), device=device, dtype=dtype, requires_grad=True)
            d = torch.randn((5, 2), device=device, dtype=dtype, requires_grad=True)
            expected_a_grad = torch.ones((4, 3), device=device, dtype=dtype)
            expected_b_grad = torch.ones((2, 3), device=device, dtype=dtype)
            expected_c_grad = torch.ones((5, 3), device=device, dtype=dtype)
            expected_d_grad = torch.ones((5, 2), device=device, dtype=dtype)
            # All the new tensors should be contiguous here. Let us make sure
            # to explicitly set them contiguous to enforce fast cat
            dim0_cat = torch.cat([a.contiguous(), b.contiguous()], dim=0)
            if dtype in complex_types():
                dim0_cat.sum().abs().backward()
                self.assertEqual(a.grad.abs(), expected_a_grad.abs())
                self.assertEqual(b.grad.abs(), expected_b_grad.abs())
            else:
                dim0_cat.sum().backward()
                self.assertEqual(a.grad, expected_a_grad)
                self.assertEqual(b.grad, expected_b_grad)
            dim1_cat = torch.cat([c.contiguous(), d.contiguous()], dim=1)
            if dtype in complex_types():
                dim1_cat.sum().abs().backward()
                self.assertEqual(c.grad.abs(), expected_c_grad.abs())
                self.assertEqual(d.grad.abs(), expected_d_grad.abs())
            else:
                dim1_cat.sum().backward()
                self.assertEqual(c.grad, expected_c_grad)
                self.assertEqual(d.grad, expected_d_grad)

    def test_cat_out_channels_last(self, device):
        # [FUNCTION DOC] Check cat with out given as a channels_last tensor gives the same values
        # as cat without out.
        x = torch.randn((4, 3, 8, 8))
        y = torch.randn(x.shape)
        res1 = torch.cat((x, y))
        z = res1.clone().contiguous(memory_format=torch.channels_last)
        res2 = torch.cat((x, y), out=z)
        self.assertEqual(res1, res2)

    @onlyNativeDeviceTypes
    def test_cat_in_channels_last(self, device):
        # [FUNCTION DOC] Check cat of channels_last inputs along every dim, for small and
        # large sizes, keeps the channels_last format and matches the result of the
        # default-format inputs.
        for dim in range(4):
            x = torch.randn((4, 15, 8, 8), device=device)
            y = torch.randn(x.shape, device=device)
            res1 = torch.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            res2 = torch.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res1, res2)

            # Size larger than grain size.
            x = torch.randn((4, 15, 256, 256), device=device)
            y = torch.randn(x.shape, device=device)
            res1 = torch.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            res2 = torch.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res1, res2)

    @onlyNativeDeviceTypes
    def test_cat_preserve_channels_last(self, device):
        # [FUNCTION DOC] Check cat of channels_last inputs preserves channels_last memory format
        # and values, including slices that are discontiguous in that format.
        x = torch.randn((4, 3, 8, 8), device=device)
        y = torch.randn(x.shape, device=device)
        res1 = torch.cat((x, y))
        res2 = torch.cat((x.contiguous(memory_format=torch.channels_last), y.contiguous(memory_format=torch.channels_last)))
        self.assertEqual(res1, res2)
        self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
        # discontiguous channels-last inputs
        x = torch.arange(24, dtype=torch.float, device=device).reshape(2, 2, 3, 2).to(memory_format=torch.channels_last)
        x1 = x[:, :, :2]
        x2 = x[:, :, 1:]
        res1 = torch.cat((x1, x2), dim=-1)
        res2 = torch.cat((x1.contiguous(), x2.contiguous()), dim=-1)
        self.assertEqual(res1, res2)
        self.assertTrue(res1.is_contiguous(memory_format=torch.channels_last))

    @onlyCUDA
    def test_cat_channels_last_large_inputs(self, device):
        # [FUNCTION DOC] CUDA only: cat of 130 channels_last tensors along dim 1 matches the CPU
        # result and stays channels_last.
        num_tensors = 130
        inputs_cuda = [
            torch.randn((2, 3, 4, 4), device=device).contiguous(memory_format=torch.channels_last)
            for _ in range(num_tensors)
        ]
        inputs_cpu = [t.cpu() for t in inputs_cuda]

        result = torch.cat(inputs_cuda, dim=1)
        expected = torch.cat(inputs_cpu, dim=1)

        self.assertEqual(result.cpu(), expected)
        self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

    @onlyCUDA
    def test_cat_out_memory_format(self, device):
        # [FUNCTION DOC] CUDA only: check memory format of cat with out for CUDA and CPU inputs.
        # A correctly shaped out keeps its own contiguous format; an empty out is resized and
        # gets channels_last only when all inputs are channels_last, otherwise the default
        # contiguous format.
        inp_size = (4, 4, 4, 4)
        expected_size = (8, 4, 4, 4)
        a_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.channels_last)
        a_cpu = torch.randn(inp_size, device='cpu').contiguous(memory_format=torch.channels_last)
        b_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.contiguous_format)
        b_cpu = torch.randn(inp_size, device='cpu').contiguous(memory_format=torch.contiguous_format)
        c_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.channels_last)

        # Case 1: if out= is the correct shape then the memory format of out= is respected

        out_cuda = torch.empty(expected_size, device=device).contiguous(memory_format=torch.contiguous_format)
        res1_cuda = torch.cat((a_cuda, b_cuda), out=out_cuda)

        out_cpu = torch.empty(expected_size, device='cpu').contiguous(memory_format=torch.contiguous_format)
        res1_cpu = torch.cat((a_cpu, b_cpu), out=out_cpu)

        self.assertTrue(res1_cuda.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(res1_cpu.is_contiguous(memory_format=torch.contiguous_format))

        # Case 2: if out= is not the correct shape then the output it is resized internally
        # - For both CPU and CUDA variants, it only propagates memory format if all the tensors have
        #   the same memory format, otherwise it just uses contiguous_format as a default

        out_cuda = torch.empty((0), device=device).contiguous(memory_format=torch.contiguous_format)
        # a_cuda and b_cuda have different memory_format
        res2_cuda = torch.cat((a_cuda, b_cuda), out=out_cuda)

        out_cpu = torch.empty((0), device='cpu').contiguous(memory_format=torch.contiguous_format)
        res2_cpu = torch.cat((a_cpu, b_cpu), out=out_cpu)

        self.assertTrue(res2_cuda.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(res2_cpu.is_contiguous(memory_format=torch.contiguous_format))

        out_cuda = torch.empty((0), device=device).contiguous(memory_format=torch.contiguous_format)
        # a_cuda and c_cuda have same memory_format
        res3_cuda = torch.cat((a_cuda, c_cuda), out=out_cuda)

        self.assertTrue(res3_cuda.is_contiguous(memory_format=torch.channels_last))

    @onlyCUDA
    def test_cat_stack_cross_devices(self, device):
        # [FUNCTION DOC] CUDA only: stack of a CUDA tensor and a CPU tensor, in either order,
        # raises a RuntimeError saying all tensors must be on the same device.
        cuda = torch.randn((3, 3), device=device)
        cpu = torch.randn((3, 3), device='cpu')

        # Stack
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cuda, cpu))
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cpu, cuda))

    # TODO: reconcile with other cat tests
    # TODO: Compare with a NumPy reference instead of CPU
    @onlyCUDA
    def test_cat(self, device):
        # [FUNCTION DOC] CUDA only: cat of three transposed random tensors along each dim from -3
        # to 2 places each input in the right narrow slice; also cat of split and chunk pieces
        # rebuilds the original, and cat with an extra row gives the expected size.
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE, device=device).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE, device=device).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE, device=device).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = torch.randn(20, SIZE, SIZE, device=device)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randn(1, SIZE, SIZE, device=device)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    # TODO: update this test to compare against NumPy instead of CPU
    @onlyCUDA
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_device_rounding(self, device, dtype):
        # [FUNCTION DOC] CUDA only: round on device values uses half-to-even and matches the
        # expected values computed as a CPU tensor.
        # test half-to-even
        a = [-5.8, -3.5, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, 5.8]
        res = [-6., -4., -2., -2., 0., 0., 2., 2., 4., 6.]

        a_tensor = torch.tensor(a, device=device).round()
        res_tensor = torch.tensor(res, device='cpu')
        self.assertEqual(a_tensor, res_tensor)

    # Note: This test failed on XLA since its test cases are created by empty_strided which
    #       doesn't support overlapping sizes/strides in XLA impl
    @onlyNativeDeviceTypes
    def test_like_fn_stride_proparation_vs_tensoriterator_unary_op(self, device):
        # [FUNCTION DOC] Check that zeros_like, ones_like, randint_like (both forms), randn_like,
        # rand_like, full_like and empty_like give the same size and stride as the unary op exp
        # does, for many dense, sliced, gapped, zero-stride and overlapping layouts under every
        # dim permutation.
        # Test like functions against tensoriterator based unary operator (exp) to
        # make sure the returned tensor from like function follows the same stride propergation
        # rule as what tensoriterator does for unary operator. The like function's  output strides
        # is computed on CPU side always, no need to test GPU here.

        def compare_helper_(like_fn, t):
            # [FUNCTION DOC] HELPER: Apply exp and the given like function to tensor t and assert
            # that the strides and sizes of the two results are equal.
            te = torch.exp(t)
            tl = like_fn(t)
            self.assertEqual(te.stride(), tl.stride())
            self.assertEqual(te.size(), tl.size())

        like_fns = [
            lambda t, **kwargs: torch.zeros_like(t, **kwargs),
            lambda t, **kwargs: torch.ones_like(t, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 10, 100, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 100, **kwargs),
            lambda t, **kwargs: torch.randn_like(t, **kwargs),
            lambda t, **kwargs: torch.rand_like(t, **kwargs),
            lambda t, **kwargs: torch.full_like(t, 7, **kwargs),
            lambda t, **kwargs: torch.empty_like(t, **kwargs)]

        # dense non-overlapping tensor,
        # non-dense non-overlapping sliced tensor
        # non-dense non-overlapping gapped tensor
        # non-dense non-overlapping 0 strided tensor
        # non-dense overlapping general tensor
        # non-dense overlapping sliced tensor
        # non-dense overlapping gapped tensor
        # non-dense overlapping 0 strided tensor
        # non-dense overlapping equal strides
        tset = (
            torch.randn(4, 3, 2, device=device),
            torch.randn(4, 3, 2, device=device)[:, :, ::2],
            torch.empty_strided((4, 3, 2), (10, 3, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 0, 3), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (4, 2, 1), device=device)[:, :, ::2].fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 1, 1, 2), (10, 0, 0, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 2, 3), (10, 3, 3), device=device).fill_(1.0))

        for like_fn in like_fns:
            for t in tset:
                for p in permutations(range(t.dim())):
                    tp = t.permute(p)
                    compare_helper_(like_fn, tp)

    def _hvd_split_helper(self, torch_fn, np_fn, op_name, inputs, device, dtype, dim):
        # [FUNCTION DOC] HELPER: Shared driver for hsplit, vsplit and dsplit tests. For each pair
        # of shape and split argument, decides whether an error is expected (too few dims, or an
        # integer split that does not divide the size). Without error, compares the torch result
        # with numpy; with error, checks torch raises RuntimeError (with the matching message),
        # and numpy raises ValueError.
        dimension_error_message = op_name + " requires a tensor with at least "
        divisibiliy_error_message = op_name + " attempted to split along dimension "

        for shape, arg in inputs:
            direction = dim - (len(shape) == 1 and dim == 1)
            bound = dim + 2 * (dim == 0) + (dim == 2)
            error_expected = len(shape) < bound or (not isinstance(arg, list) and shape[direction] % arg != 0)

            t = make_tensor(shape, dtype=dtype, device=device)
            t_np = t.cpu().numpy()

            if not error_expected:
                self.assertEqual(torch_fn(t, arg), np_fn(t_np, arg))
            else:
                self.assertRaises(RuntimeError, lambda: torch_fn(t, arg))
                self.assertRaises(ValueError, lambda: np_fn(t, arg))
                expected_error_message = dimension_error_message if len(shape) < bound else divisibiliy_error_message
                self.assertRaisesRegex(RuntimeError, expected_error_message, lambda: torch_fn(t, arg))

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_hsplit(self, device, dtype):
        # [FUNCTION DOC] Run the split helper for hsplit (split dimension 1) over scalar, 1-D,
        # 2-D and 3-D inputs, integer and list split arguments, including empty dims and
        # out-of-range indices.
        inputs = (
            ((), 3),
            ((), [2, 4, 6]),
            ((6,), 2),
            ((6,), 4),
            ((6,), [2, 5]),
            ((6,), [7, 9]),
            ((3, 8), 4),
            ((3, 8), 5),
            ((3, 8), [1, 5]),
            ((3, 8), [3, 8]),
            ((5, 5, 5), 2),
            ((5, 5, 5), [1, 4]),
            ((5, 0, 5), 3),
            ((5, 5, 0), [2, 6]),
        )
        self._hvd_split_helper(torch.hsplit, np.hsplit, "torch.hsplit", inputs, device, dtype, 1)

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_vsplit(self, device, dtype):
        # [FUNCTION DOC] Run the split helper for vsplit (split dimension 0) over 1-D to 3-D
        # inputs with integer and list split arguments.
        inputs = (
            ((6,), 2),
            ((6,), 4),
            ((6, 5), 2),
            ((6, 5), 4),
            ((6, 5), [1, 2, 3]),
            ((6, 5), [1, 5, 9]),
            ((6, 5, 5), 2),
            ((6, 0, 5), 2),
            ((5, 0, 5), [1, 5]),
        )
        self._hvd_split_helper(torch.vsplit, np.vsplit, "torch.vsplit", inputs, device, dtype, 0)

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_dsplit(self, device, dtype):
        # [FUNCTION DOC] Run the split helper for dsplit (split dimension 2) over 1-D to 3-D
        # inputs with integer and list split arguments, including empty dims.
        inputs = (
            ((6,), 4),
            ((6, 6), 3),
            ((5, 5, 6), 2),
            ((5, 5, 6), 4),
            ((5, 5, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
            ((5, 5, 0), 2),
            ((5, 0, 6), 4),
            ((5, 0, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
        )
        self._hvd_split_helper(torch.dsplit, np.dsplit, "torch.dsplit", inputs, device, dtype, 2)

    def _test_special_stacks(self, dim, at_least_dim, torch_fn, np_fn, device, dtype):
        # [FUNCTION DOC] HELPER: Shared driver for hstack, column_stack, vstack, row_stack and
        # dstack tests. Checks a TypeError for a bare tensor argument, 0-D inputs against numpy,
        # then for random shapes differing along one axis: if valid (given the concatenating dim
        # and the minimum dims) compares with numpy, otherwise expects RuntimeError from torch
        # and ValueError from numpy.
        # Test error for non-tuple argument
        t = torch.randn(10)
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn(t)
        # Test error for a single array
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn(t)

        # Test 0-D
        num_tensors = random.randint(1, 5)
        input_t = [torch.tensor(random.uniform(0, 10), device=device, dtype=dtype) for i in range(num_tensors)]
        actual = torch_fn(input_t)
        expected = np_fn([input.cpu().numpy() for input in input_t])
        self.assertEqual(actual, expected)

        for ndims in range(1, 5):
            base_shape = list(_rand_shape(ndims, min_size=1, max_size=5))
            for i in range(ndims):
                shape = list(base_shape)
                num_tensors = random.randint(1, 5)
                torch_input = []
                # Create tensors with shape being different along one axis only
                for _ in range(num_tensors):
                    shape[i] = random.randint(1, 5)
                    torch_input.append(_generate_input(tuple(shape), dtype, device, with_extremal=False))

                # Determine if input tensors have valid dimensions.
                valid_dim = True
                for k in range(len(torch_input) - 1):
                    for tdim in range(ndims):
                        # Test whether all tensors have the same shape except in concatenating dimension
                        # Unless the number of dimensions is less than the corresponding at_least function dimension
                        # Since the original concatenating dimension would shift after applying at_least and would no
                        # longer be the concatenating dimension
                        if (ndims < at_least_dim or tdim != dim) and torch_input[k].size()[tdim] != torch_input[k + 1].size()[tdim]:
                            valid_dim = False

                # Special case for hstack is needed since hstack works differently when ndims is 1
                if valid_dim or (torch_fn is torch.hstack and ndims == 1):
                    # Valid dimensions, test against numpy
                    np_input = [input.cpu().numpy() for input in torch_input]
                    actual = torch_fn(torch_input)
                    expected = np_fn(np_input)
                    self.assertEqual(actual, expected)
                else:
                    # Invalid dimensions, test for error
                    with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match except in dimension"):
                        torch_fn(torch_input)
                    with self.assertRaises(ValueError):
                        np_input = [input.cpu().numpy() for input in torch_input]
                        np_fn(np_input)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_hstack_column_stack(self, device, dtype):
        # [FUNCTION DOC] Run the special stack driver for hstack and column_stack against numpy,
        # then compare column_stack of mixed 1-D and 2-D inputs with numpy.
        ops = ((torch.hstack, np.hstack), (torch.column_stack, np.column_stack))
        for torch_op, np_op in ops:
            self._test_special_stacks(1, 1, torch_op, np_op, device, dtype)

        # Test torch.column_stack with combinations of 1D and 2D tensors input
        one_dim_tensor = torch.arange(0, 10).to(dtype=dtype, device=device)
        two_dim_tensor = torch.arange(0, 100).to(dtype=dtype, device=device).reshape(10, 10)
        inputs = two_dim_tensor, one_dim_tensor, two_dim_tensor, one_dim_tensor
        torch_result = torch.column_stack(inputs)

        np_inputs = [input.cpu().numpy() for input in inputs]
        np_result = np.column_stack(np_inputs)

        self.assertEqual(np_result,
                         torch_result)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_vstack_row_stack(self, device, dtype):
        # [FUNCTION DOC] Run the special stack driver for vstack and row_stack against numpy, and
        # check vstack handles a 1-D tensor together with a 1-row 2-D tensor.
        ops = ((torch.vstack, np.vstack), (torch.row_stack, np.vstack))
        for torch_op, np_op in ops:
            self._test_special_stacks(0, 2, torch_op, np_op, device, dtype)
            for _ in range(5):
                # Test dimension change for 1D tensor of size (N) and 2D tensor of size (1, N)
                n = random.randint(1, 10)
                input_a = _generate_input((n,), dtype, device, with_extremal=False)
                input_b = _generate_input((1, n), dtype, device, with_extremal=False)
                torch_input = [input_a, input_b]
                np_input = [input.cpu().numpy() for input in torch_input]
                actual = torch_op(torch_input)
                expected = np_op(np_input)
                self.assertEqual(actual, expected)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_dstack(self, device, dtype):
        # [FUNCTION DOC] Run the special stack driver for dstack against numpy, and check dstack
        # of mixed 1-D, 2-D and 3-D inputs, repeated several times with random sizes.
        self._test_special_stacks(2, 3, torch.dstack, np.dstack, device, dtype)
        for _ in range(5):
            # Test dimension change for 1D tensor of size (N), 2D tensor of size (1, N), and 3D tensor of size (1, N, 1)
            n = random.randint(1, 10)
            input_a = _generate_input((n,), dtype, device, with_extremal=False)
            input_b = _generate_input((1, n), dtype, device, with_extremal=False)
            input_c = _generate_input((1, n, 1), dtype, device, with_extremal=False)
            torch_input = [input_a, input_b, input_c]
            np_input = [input.cpu().numpy() for input in torch_input]
            actual = torch.dstack(torch_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

            # Test dimension change for 2D tensor of size (M, N) and 3D tensor of size (M, N, 1)
            m = random.randint(1, 10)
            n = random.randint(1, 10)
            input_a = _generate_input((m, n), dtype, device, with_extremal=False)
            input_b = _generate_input((m, n, 1), dtype, device, with_extremal=False)
            torch_input = [input_a, input_b]
            np_input = [input.cpu().numpy() for input in torch_input]
            actual = torch.dstack(torch_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

    @dtypes(torch.int32, torch.int64)
    def test_large_linspace(self, device, dtype):
        # [FUNCTION DOC] Check linspace between the min and a large max of int32 or int64
        # gives a first step larger than the exact fractional step (no truncation toward zero
        # of the step).
        start = torch.iinfo(dtype).min
        end = torch.iinfo(dtype).max & ~0xfff
        steps = 15
        x = torch.linspace(start, end, steps, dtype=dtype, device=device)
        self.assertGreater(x[1] - x[0], (end - start) / steps)

    @dtypes(torch.float32, torch.float64)
    def test_unpack_double(self, device, dtype):
        # [FUNCTION DOC] Regression test: build float32 or float64 tensors from big Python
        # numbers (above 2 to the 24 or 53, integer limits, and out-of-range floats) and compare
        # with numpy conversions.
        # Reference: https://github.com/pytorch/pytorch/issues/33111
        vals = (2 ** 24 + 1, 2 ** 53 + 1,
                np.iinfo(np.int64).max, np.iinfo(np.uint64).max, np.iinfo(np.uint64).max + 1,
                -1e500, 1e500)
        for val in vals:
            t = torch.tensor(val, dtype=dtype, device=device)
            a = np.array(val, dtype=torch_to_numpy_dtype_dict[dtype])
            self.assertEqual(t, torch.from_numpy(a))

    def _float_to_int_conversion_helper(self, vals, device, dtype, refs=None):
        # [FUNCTION DOC] HELPER: Convert values to a float tensor then to the target dtype and
        # compare on CPU with the reference. If no reference is given, it is numpy casting of a
        # float32 array to the matching numpy dtype.
        if refs is None:
            a = np.array(vals, dtype=np.float32).astype(torch_to_numpy_dtype_dict[dtype])
            refs = torch.from_numpy(a)
        t = torch.tensor(vals, device=device, dtype=torch.float).to(dtype)
        self.assertEqual(refs, t.cpu())

    # Checks that float->integer casts don't produce undefined behavior errors.
    # Note: In C++, casting from a floating value to an integral dtype
    # is undefined if the floating point value is not within the integral
    # dtype's dynamic range. This can (and should) cause undefined behavior
    # errors with UBSAN. These casts are deliberate in PyTorch, however, and
    # NumPy may have the same behavior.
    @onlyNativeDeviceTypes
    @unittest.skipIf(IS_PPC, "Test is broken on PowerPC, see https://github.com/pytorch/pytorch/issues/39671")
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_finite(self, device, dtype):
        # [FUNCTION DOC] Check finite float to integer or bool casts against numpy references.
        # The value list is trimmed per device and dtype where conversion of out-of-range values
        # is divergent, and uint8 on CPU uses hand-written expected results.
        min = torch.finfo(torch.float).min
        max = torch.finfo(torch.float).max

        # Note: CUDA/MTIA max float -> integer conversion is divergent on some dtypes
        vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2, max)
        refs = None
        if self.device_type in ('cuda', 'mtia'):
            if torch.version.hip:
                # HIP min float -> int64 conversion is divergent
                vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)
            elif dtype == torch.uint8:
                # CUDA out-of-range float -> uint8 is divergent under clang-21:
                # out-of-range float->int is UB and codegen differs from numpy's
                # reference. Test only in-range values so torch and numpy agree
                # deterministically.
                vals = (0, .5, 1.5, 2)
            elif dtype in (torch.int8, torch.int16):
                # CUDA min float -> int8/int16 is divergent under clang-21
                # (out-of-range float->int is UB). Drop the out-of-range extreme.
                vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)
            else:
                vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
        elif dtype == torch.uint8:
            # Note: CPU max float -> uint8 conversion is divergent
            vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
            # Note: numpy -2.0 or -1.5 -> uint8 conversion is undefined
            #       see https://github.com/pytorch/pytorch/issues/97794
            refs = (0, 254, 255, 0, 0, 0, 1, 2)
        elif dtype in (torch.int8, torch.int16):
            # CPU min and max float -> int8/int16 conversion is divergent.
            # (int8 also diverges under clang-21: out-of-range float->int is UB
            # and codegen differs from numpy's reference.) Drop the out-of-range
            # extremes.
            vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)

        self._float_to_int_conversion_helper(vals, device, dtype, refs)

    # Note: CUDA will fail this test on most dtypes, often dramatically.
    # Note: This test validates undefined behavior consistency in float-to-ints casts
    # NB: torch.uint16, torch.uint32, torch.uint64 excluded as this
    # nondeterministically fails, warning "invalid value encountered in cast"
    @onlyCPU
    @unittest.skipIf(IS_S390X, "Test fails for int16 on s390x. Needs investigation.")
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_nonfinite(self, device, dtype):
        # [FUNCTION DOC] CPU only: check casts of negative infinity, positive infinity and nan
        # to integer types against references that depend on platform (arm64 or not) and on
        # the dtype.
        vals = (float('-inf'), float('inf'), float('nan'))

        if dtype == torch.bool:
            refs = (True, True, True)
        elif IS_ARM64:
            refs = (torch.iinfo(dtype).min, torch.iinfo(dtype).max, 0)
            if dtype in (torch.int8, torch.int16):
                refs = (0, -1, 0)
        else:
            refs = (0, 0, 0)
            if dtype in (torch.int32, torch.int64):
                refs = (torch.iinfo(dtype).min, ) * 3
        self._float_to_int_conversion_helper(vals, device, dtype, refs)

    @onlyNativeDeviceTypes
    def test_complex_type_conversions(self, device):
        # [FUNCTION DOC] Check converting random tensors among float, complex64 and complex128:
        # complex to real keeps the real part, real to complex has zero imaginary part, otherwise
        # values are equal.
        dtypes = [torch.float, torch.complex64, torch.complex128]
        for from_type in dtypes:
            for to_type in dtypes:
                from_tensor = torch.randn(4, dtype=from_type, device=device)
                to_tensor = from_tensor.to(to_type)
                if from_type.is_complex and not to_type.is_complex:
                    self.assertEqual(torch.real(from_tensor), to_tensor, exact_dtype=False)
                elif not from_type.is_complex and to_type.is_complex:
                    self.assertEqual(from_tensor, torch.real(to_tensor), exact_dtype=False)
                    self.assertEqual(torch.zeros_like(torch.imag(to_tensor)), torch.imag(to_tensor), exact_dtype=False)
                else:
                    self.assertEqual(from_tensor, to_tensor, exact_dtype=False)

    @slowTest
    @onlyCPU
    def test_cat_big(self, device):
        # [FUNCTION DOC] Slow CPU test: cat of two huge uint8 tensors gives a result whose first
        # dimension is the sum of the inputs.
        SIZE1 = 6500
        SIZE2 = 4500
        concat_list = []
        concat_list.append(torch.ones((SIZE1, 1024 * 512), dtype=torch.uint8, device=device))
        concat_list.append(torch.ones((SIZE2, 1024 * 512), dtype=torch.uint8, device=device))
        result = torch.cat(concat_list)
        self.assertEqual(result.size(0), SIZE1 + SIZE2)

    @onlyCPU
    @dtypes(torch.half, torch.double, torch.int)
    def test_cat2(self, device, dtype):
        # [FUNCTION DOC] CPU only: cat of transposed random integer-valued tensors for half,
        # double and int along each dim from -3 to 2 puts each input in its slice; also cat of
        # split and chunk pieces.
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.randint(low=-100, high=100, size=(13, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            y = torch.randint(low=-100, high=100, size=(17, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            z = torch.randint(low=-100, high=100, size=(19, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = torch.randint(low=-100, high=100, size=(20, SIZE, SIZE), device=device).to(dtype)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randint(low=-100, high=100, size=(1, SIZE, SIZE), device=device).to(dtype)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    @dtypes(torch.float)
    def test_cat_size1(self, device, dtype):
        # [FUNCTION DOC] Check cat of single-element slices (aligned stride, unaligned slice)
        # along the last dim matches cat of copies with fresh strides.
        # create a tensor that has aligned stride along dim - 1 dimension
        # but catted slice size is not aligned
        x1 = torch.randn(16, 16, device=device, dtype=dtype)[:1, :1]
        xref = x1.clone().view(-1).view(x1.shape)
        # make sure output size is aligned, need at least 4 elements for this
        res = torch.cat([x1, x1, x1, x1], dim=-1)
        ref = torch.cat([xref, xref, xref, xref], dim=-1)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    def test_cat_trailing_dim(self, device, dtype):
        # [FUNCTION DOC] Check cat along dim 1 of two 3-D tensors with a trailing dim of 23
        # matches the CPU result.
        x1 = torch.randn(16, 16, 23, device=device, dtype=dtype)
        x2 = torch.rand_like(x1)
        res = torch.cat([x1, x2], dim=1)
        ref = torch.cat([x1.cpu(), x2.cpu()], dim=1)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    def test_cat_misaligned(self, device, dtype):
        # [FUNCTION DOC] Check cat of two misaligned (offset start) 1-D slices matches the
        # CPU result.
        x1 = torch.randn(14, device=device, dtype=dtype)[2:]
        x2 = torch.rand_like(x1)
        res = torch.cat([x1, x2], dim=-1)
        ref = torch.cat([x1.cpu(), x2.cpu()], dim=-1)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    def test_cat_multi_batch(self, device, dtype):
        # [FUNCTION DOC] Check cat of 130 tensors (many kernel batches) along the last dim and
        # along dim 0, with two differently sized trailing inputs, matches the CPU result.
        xs = [torch.randn(16, 16, device=device, dtype=dtype) for _ in range(130)]
        xs_cpu = [x.cpu() for x in xs]
        res = torch.cat(xs, dim=-1)
        ref = torch.cat(xs_cpu, dim=-1)
        self.assertEqual(res, ref)
        xs = [torch.randn(16, 15, 15, device=device, dtype=dtype) for _ in range(130)]
        xs[128] = torch.randn(15, 15, 15, device=device, dtype=dtype)
        xs[129] = torch.randn(17, 15, 15, device=device, dtype=dtype)
        xs_cpu = [x.cpu() for x in xs]
        res = torch.cat(xs, dim=0)
        ref = torch.cat(xs_cpu, dim=0)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    @largeTensorTest("16GB")
    def test_cat_large_tensor(self, device, dtype):
        # [FUNCTION DOC] Large tensor test requiring 16GB: cat of a tensor with 2 to the 32 bytes
        # of data and a much smaller one matches the CPU result.
        N = 2 ** 32 // dtype.itemsize
        inps = [torch.randn(N, device=device, dtype=dtype), torch.randn(N // 128, device=device, dtype=dtype)]
        res = torch.cat(inps, dim=0)
        ref = torch.cat([x.cpu() for x in inps])
        self.assertEqual(res, ref)

    # FIXME: Create an OpInfo-based tensor creation method test that verifies this for all tensor
    #   creation methods and verify all dtypes and layouts
    @dtypes(torch.bool, torch.uint8, torch.int16, torch.int64, torch.float16, torch.float32, torch.complex64)
    def test_zeros_dtype_layout_device_match(self, device, dtype):
        # [FUNCTION DOC] Check zeros with explicit dtype, strided layout and device produces a
        # tensor whose dtype, layout and device match what was requested.
        layout = torch.strided
        t = torch.zeros((2, 3), device=device, dtype=dtype, layout=layout)
        self.assertIs(dtype, t.dtype)
        self.assertIs(layout, t.layout)
        self.assertEqual(torch.device(device), t.device)

    def test_stack(self, device):
        # [FUNCTION DOC] Check stack of three tensors for half, double and int along every
        # positive and negative dim: matches, has an inserted size-3 axis, and selecting each
        # index recovers the input.
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            for dim in range(4):
                res = torch.stack((x, y, z), dim)
                res_neg = torch.stack((x, y, z), dim - 4)
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                self.assertEqual(res, res_neg)
                self.assertEqual(res.size(), expected_size)
                self.assertEqual(res.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 2), z, atol=0, rtol=0)

    def test_stack_out(self, device):
        # [FUNCTION DOC] Check stack with an out tensor for positive and negative dims: results
        # equal, size is correct, the out storage pointer is unchanged, and each slice matches
        # its input.
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            for dim in range(4):
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                res_out = x.new(expected_size)
                res_neg_out = x.new(expected_size)
                res_out_dp = res_out.data_ptr()
                res_out_neg_dp = res_neg_out.data_ptr()
                torch.stack((x, y, z), dim, out=res_out)
                torch.stack((x, y, z), dim - 4, out=res_neg_out)
                self.assertEqual(res_out, res_neg_out)
                self.assertEqual(res_out.size(), expected_size)
                self.assertEqual(res_out_dp, res_out.data_ptr())
                self.assertEqual(res_out_neg_dp, res_neg_out.data_ptr())
                self.assertEqual(res_out.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 2), z, atol=0, rtol=0)

    def test_repeat_interleave(self, device):
        # [FUNCTION DOC] Check repeat_interleave: repeats-only form, errors for
        # multi-dimensional, float or negative repeats, negative repeats rejected even with
        # output_size (CPU only), scalar and tensor repeats with and without dim, mismatched
        # repeats length errors, and zero-sized inputs.
        x = torch.tensor([0, 1, 2, 3], device=device)
        expected = torch.tensor([1, 2, 2, 3, 3, 3], device=device)
        self.assertEqual(torch.repeat_interleave(x), expected)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4, device=device).reshape(2, 2))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0, device=device))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4], device=device))

        # Negative repeats must be rejected even when output_size is passed,
        # otherwise the CPU kernel can write out of bounds: the corrupted cumsum
        # makes a later non-negative element produce start < 0, and in a
        # multi-threaded run that OOB write races ahead of the sibling thread's
        # check. See https://github.com/pytorch/pytorch/issues/188938
        # On CUDA the kernel rejects these inputs via a device-side assert,
        # which cannot be caught cleanly and poisons the context, so this is
        # validated on CPU only.
        if torch.device(device).type == "cpu":
            with self.assertRaisesRegex(RuntimeError, "repeats can not be negative"):
                torch.repeat_interleave(
                    torch.tensor([-1, -1, 2], device=device), output_size=0)
            with self.assertRaisesRegex(RuntimeError, "repeats can not be negative"):
                torch.repeat_interleave(
                    torch.tensor([5, -2, 1], device=device), output_size=4)

        y = torch.tensor([[1, 2], [3, 4]], device=device)

        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2, device=device))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2], device=device))
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        y2 = torch.repeat_interleave(y, 3, dim=1)
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        self.assertEqual(y2, y2_expect)

        y3 = torch.repeat_interleave(y, torch.tensor([1, 2], device=device), dim=0)
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        self.assertEqual(y3, y3_expect)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3], device=device), dim=0)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9, device=device).reshape(3, 3), dim=0)

        # test zero sized dimension
        x = torch.zeros((5, 0), device=device)
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        x = torch.tensor([], dtype=torch.int64, device=device)
        y = torch.repeat_interleave(x, x)
        self.assertEqual(y, x)

    def test_new_methods_requires_grad(self, device):
        # [FUNCTION DOC] Check new_full, new_empty, new_zeros and new_ones honor requires_grad
        # True and False on a float tensor, and raise RuntimeError when requires_grad is set on
        # an integer tensor.
        size = (10,)
        test_cases = [
            # method name, args
            ('new_full', [size, 1]),
            ('new_empty', [size]),
            ('new_zeros', [size]),
            ('new_ones', [size]),
        ]
        for method_name, args in test_cases:
            x = torch.randn(size, device=device)
            for requires_grad in [True, False]:
                x_new = x.__getattribute__(method_name)(*args, requires_grad=requires_grad)
                self.assertEqual(x_new.requires_grad, requires_grad)
            x = torch.randint(10, size, device=device)
            with self.assertRaisesRegex(
                    RuntimeError,
                    r'Only Tensors of floating point and complex dtype can require gradients'):
                x_new = x.__getattribute__(method_name)(*args, requires_grad=True)

    def test_tensor_from_sequence(self, device):
        # [FUNCTION DOC] Check torch.tensor on a user-defined sequence: one whose item access
        # raises a TypeError fails with ValueError about not determining the shape; a working one
        # equals a list.
        class MockSequence:
            # [CLASS DOC] HELPER: Sequence stub holding a list, reporting its length but whose
            # item access always raises TypeError, so torch cannot read its elements.
            def __init__(self, lst):
                # [FUNCTION DOC] HELPER: Store the given list in the lst attribute.
                self.lst = lst

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the length of the stored list.
                return len(self.lst)

            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: Always raises TypeError regardless of the index.
                raise TypeError

        class GoodMockSequence(MockSequence):
            # [CLASS DOC] HELPER: Subclass of the mock sequence whose item access works and
            # returns the stored list element, so torch can build a tensor from it.
            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: Return the stored list element at the given index.
                return self.lst[item]

        bad_mock_seq = MockSequence([1.0, 2.0, 3.0])
        good_mock_seq = GoodMockSequence([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, 'could not determine the shape'):
            torch.tensor(bad_mock_seq, device=device)
        self.assertEqual(torch.tensor([1.0, 2.0, 3.0], device=device), torch.tensor(good_mock_seq, device=device))

    def test_simple_scalar_cast(self, device):
        # [FUNCTION DOC] Check int, float and complex conversion of one-element tensors, complex
        # of a complex scalar, ValueError for multi-element or empty tensors, and RuntimeError
        # when converting a complex tensor to int or float.
        ok = [torch.tensor([1.5], device=device), torch.zeros(1, 1, 1, 1, device=device)]
        ok_values = [1.5, 0]

        not_ok = map(torch.Tensor, [[], [1, 2], [[1, 2], [3, 4]]])

        for tensor, value in zip(ok, ok_values):
            self.assertEqual(int(tensor), int(value))
            self.assertEqual(float(tensor), float(value))
            self.assertEqual(complex(tensor), complex(value))

        self.assertEqual(complex(torch.tensor(1.5j)), 1.5j)

        for tensor in not_ok:
            self.assertRaises(ValueError, lambda: int(tensor))
            self.assertRaises(ValueError, lambda: float(tensor))
            self.assertRaises(ValueError, lambda: complex(tensor))

        self.assertRaises(RuntimeError, lambda: float(torch.tensor(1.5j)))
        self.assertRaises(RuntimeError, lambda: int(torch.tensor(1.5j)))

    def test_offset_scalar_cast(self, device):
        # [FUNCTION DOC] Check int conversion of a one-element slice view that has a
        # storage offset.
        x = torch.tensor([1., 2., 3.], device=device)
        y = x[2:]
        self.assertEqual(int(y), 3)

    def test_meshgrid_empty(self):
        # [FUNCTION DOC] Check meshgrid with no arguments raises RuntimeError about a
        # non-empty list.
        with self.assertRaisesRegex(RuntimeError,
                                    'expects a non-empty TensorList'):
            torch.meshgrid()

    def test_meshgrid_unsupported_indexing(self):
        # [FUNCTION DOC] Check meshgrid with an empty indexing string raises RuntimeError saying
        # indexing must be xy or ij.
        with self.assertRaisesRegex(RuntimeError,
                                    'indexing must be one of "xy" or "ij"'):
            torch.meshgrid(torch.tensor([1, 2]), indexing='')

    def test_meshgrid_non_1d_tensor(self):
        # [FUNCTION DOC] Check meshgrid of a 2-D tensor raises RuntimeError expecting 0-D or
        # 1-D tensors.
        with self.assertRaisesRegex(RuntimeError,
                                    'Expected 0D or 1D tensor'):
            torch.meshgrid(torch.tensor([[1, 2], [3, 4]]))

    def test_meshgrid_inconsistent_dtype(self):
        # [FUNCTION DOC] Check meshgrid of tensors with different dtypes raises RuntimeError.
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same dtype'):
            torch.meshgrid(torch.tensor([1], dtype=torch.int),
                           torch.tensor([2], dtype=torch.float))

    def test_meshgrid_inconsistent_device(self):
        # [FUNCTION DOC] Check meshgrid of a cpu tensor and a meta tensor raises RuntimeError
        # about differing devices.
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same device'):
            torch.meshgrid(torch.tensor([1], device='cpu'),
                           torch.tensor([2], device='meta'))

    def test_meshgrid_warns_if_no_indexing(self):
        # [FUNCTION DOC] Check meshgrid without the indexing argument emits a UserWarning (once)
        # saying the argument will be required.
        with self.assertWarnsOnceRegex(
                UserWarning, '.*will be required to pass the indexing arg.*'):
            torch.meshgrid(torch.tensor([1, 2]))

    def test_meshgrid_default_indexing(self, device):
        # [FUNCTION DOC] Check meshgrid without indexing on a 0-D and two 1-D tensors, passed as
        # a list or as separate arguments, gives ij-style grids of the expected shape and values.
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c])
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_xy_indexing(self, device):
        # [FUNCTION DOC] Same as the default-indexing check but with xy indexing: the first
        # two axes are swapped, giving shape 3x1x2 with the expected values, for list and
        # varargs calls.
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c], indexing='xy')
        self.assertEqual(grid_a.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_b.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_c.shape, torch.Size([3, 1, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c, indexing='xy')
        self.assertEqual(grid_a2.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([3, 1, 2]))
        expected_grid_a = torch.ones(3, 1, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1]],
                                        [[2, 2]],
                                        [[3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2]],
                                        [[1, 2]],
                                        [[1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_ij_indexing(self, device):
        # [FUNCTION DOC] Same as the default-indexing check but with explicit ij indexing, giving
        # shape 1x3x2 with the expected values, for list and varargs calls.
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c], indexing='ij')
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c, indexing='ij')
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_ij_indexing_is_default(self, device):
        # [FUNCTION DOC] Check that meshgrid results with explicit ij indexing equal those
        # without the indexing argument.
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid(a, b, c, indexing='ij')
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        self.assertTrue(grid_a.equal(grid_a2))
        self.assertTrue(grid_b.equal(grid_b2))
        self.assertTrue(grid_c.equal(grid_c2))

    @skipMeta
    def test_meshgrid_vs_numpy(self, device):
        # [FUNCTION DOC] Compare meshgrid with numpy meshgrid over 0-D and 1-D inputs (including
        # empty cases) and for the matching indexing-argument pairs; each case runs as a subtest.
        # Shapes to the random tensors. Each line is a test case, and
        # each list within that line is the shape of a single
        # tensor. The shapes are restricted to 0D (represented by [])
        # and 1D tensors.
        cases = [
            [[]],
            [[1], [1], [1]],
            [[], [], []],
            [[3], [5], [7]],
            [[3], [], [7]],
            [[11], [13]],
            [[15]],
        ]

        # We also need to test the different indexing modes. We can't
        # just enumerate them because we don't presently support the
        # same modes as numpy.meshgrid, nor does our default
        # correspond to their default.
        #
        # TODO Eliminate this and replace it with a list of all
        # supported indexing modes when we have full compatibility.
        indexing_correspondence = [
            # No indexing in PyTorch corresponds to "ij" indexing in
            # NumPy.
            ({}, {'indexing': 'ij'}),

            # No indexing in NumPy corresponds to "xy" indexing in
            # PyTorch.
            ({'indexing': 'xy'}, {}),

            # "ij" and "xy" are implemented identically in both.
            ({'indexing': 'ij'}, {'indexing': 'ij'}),
            ({'indexing': 'xy'}, {'indexing': 'xy'}),
        ]
        for shapes, (torch_kwargs, numpy_kwargs) in product(cases, indexing_correspondence):
            with self.subTest(shapes=shapes, torch_kwargs=torch_kwargs, numpy_kwargs=numpy_kwargs):
                tensors = [make_tensor(shape, device=device, dtype=torch.int) for shape in shapes]
                torch_grids = torch.meshgrid(*tensors, **torch_kwargs)
                numpy_grids = np.meshgrid(*(tensor.cpu().numpy() for tensor in tensors), **numpy_kwargs)
                self.assertEqual(torch_grids, numpy_grids)


    def test_cartesian_prod(self, device):
        # [FUNCTION DOC] Check cartesian_prod against itertools product for three tensors, an
        # empty input (result has shape 0 by 4), and a single input (returned unchanged).
        a = torch.tensor([1], device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        prod = torch.cartesian_prod(a, b, c)
        expected = torch.tensor(list(product([a], b, c)), device=device)
        self.assertEqual(expected, prod)

        # test 0 size input
        d = torch.empty(0, dtype=b.dtype, device=device)
        prod = torch.cartesian_prod(a, b, c, d)
        expected = torch.empty(0, 4, dtype=b.dtype, device=device)
        self.assertEqual(expected, prod)

        # test single input
        prod = torch.cartesian_prod(b)
        self.assertEqual(b, prod)

    def test_combinations(self, device):
        # [FUNCTION DOC] Check combinations for r from 0 to 5, with and without replacement,
        # against itertools results, with r larger than the length giving an empty result, and
        # for empty input.
        a = torch.tensor([1, 2, 3], device=device)

        c = torch.combinations(a, r=0)
        expected = torch.empty(0, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=1)
        expected = torch.tensor(list(combinations(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=1, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a)
        expected = torch.tensor(list(combinations(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=3)
        expected = torch.tensor(list(combinations(a, r=3)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=4)
        expected = torch.empty(0, 4, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=5)
        expected = torch.empty(0, 5, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        # test empty input
        a = torch.empty(0, device=device)
        c1 = torch.combinations(a)
        c2 = torch.combinations(a, with_replacement=True)
        expected = torch.empty(0, 2, dtype=a.dtype, device=device)
        self.assertEqual(c1, expected)
        self.assertEqual(c2, expected)

    @skipMeta
    def test_linlogspace_mem_overlap(self, device):
        # [FUNCTION DOC] Check linspace and logspace with out set to an expanded (overlapping)
        # tensor raise RuntimeError about an unsupported operation.
        x = torch.rand(1, device=device).expand(10)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.linspace(1, 10, 10, out=x)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.logspace(1, 10, 10, out=x)

    def test_ctor_with_numpy_array(self, device):
        # [FUNCTION DOC] Check legacy DoubleTensor, FloatTensor and HalfTensor constructors
        # accept numpy arrays of several dtypes and preserve element values.
        correct_dtypes = [
            np.double,
            float,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            bool,
        ]

        incorrect_byteorder = '>' if sys.byteorder == 'little' else '<'
        incorrect_dtypes = [incorrect_byteorder + t for t in ['d', 'f']]

        for dtype in correct_dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)

            # Upcast
            tensor = torch.DoubleTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            # Downcast (sometimes)
            tensor = torch.FloatTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            tensor = torch.HalfTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_random(self, device, dtype):
        # [FUNCTION DOC] Check random_ with a lower and upper bound, and with only an upper
        # bound, yields values spanning exactly the range of lower to upper minus one, for float
        # and integer dtypes.
        # This test is flaky with p<=(2/(ub-lb))^200=6e-36
        t = torch.empty(200, dtype=dtype, device=device)
        lb = 1
        ub = 4

        t.fill_(-1)
        t.random_(lb, ub)
        self.assertEqual(t.min(), lb)
        self.assertEqual(t.max(), ub - 1)

        t.fill_(-1)
        t.random_(ub)
        self.assertEqual(t.min(), 0)
        self.assertEqual(t.max(), ub - 1)

    def test_random_bool(self, device):
        # [FUNCTION DOC] Check random_ on a bool tensor produces both values with roughly equal
        # frequency, starting from all-False and from all-True.
        size = 2000
        t = torch.empty(size, dtype=torch.bool, device=device)

        t.fill_(False)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

        t.fill_(True)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

    def test_random_from_to_bool(self, device):
        # [FUNCTION DOC] Check random_ with from and to arguments on bool tensors: out-of-range
        # from or to minus one, and from not less than to, raise RuntimeError with specific
        # messages; valid ranges produce values within bounds.
        size = 2000

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        min_val = 0
        max_val = 1

        froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
        tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=torch.bool, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = 1
                        self.assertTrue(from_ <= t.to(torch.int).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.int).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    # NB: uint64 is broken because its max value is not representable in
    # int64_t, but this is what random expects
    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch.uint16, torch.uint32))
    def test_random_full_range(self, device, dtype):
        # [FUNCTION DOC] Check random_ with only a from argument spans the whole representable
        # range (limited by float mantissa for floating dtypes) by checking min and max fall near
        # both ends.
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype == torch.double:
            fp_limit = 2**53
        elif dtype == torch.float:
            fp_limit = 2**24
        elif dtype == torch.half:
            fp_limit = 2**11
        elif dtype == torch.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        t = torch.empty(size, dtype=dtype, device=device)

        if dtype in [torch.float, torch.double, torch.half, torch.bfloat16]:
            from_ = int(max(-fp_limit, int64_min_val))
            to_inc_ = int(min(fp_limit, int64_max_val))
        else:
            from_ = int(max(torch.iinfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.iinfo(dtype).max, int64_max_val))
        range_ = to_inc_ - from_ + 1

        t.random_(from_, None)
        delta = max(1, alpha * range_)
        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
        self.assertTrue((to_inc_ - delta) < t.to(torch.double).max() <= to_inc_)

    # NB: uint64 is broken because its max value is not representable in
    # int64_t, but this is what random expects
    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch .uint16, torch.uint32))
    def test_random_from_to(self, device, dtype):
        # [FUNCTION DOC] Check random_ with from and to arguments for each dtype over lists of
        # candidate bounds. Out-of-range from or to minus one raise RuntimeError; from not below
        # to raises RuntimeError; floating dtypes whose bounds exceed exact integer range only
        # warn; valid cases check min and max fall near the ends (looser for bfloat16).
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, -42, min_val - 1, min_val, 42, max_val, max_val + 1]
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1]
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        if dtype == torch.double:
            fp_limit = 2**53
        elif dtype == torch.float:
            fp_limit = 2**24
        elif dtype == torch.half:
            fp_limit = 2**11
        elif dtype == torch.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=dtype, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        if dtype.is_floating_point and (
                                not (-fp_limit <= from_ <= fp_limit) or not (-fp_limit <= (to_ - 1) <= fp_limit)):
                            if not (-fp_limit <= from_ <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "from is out of bounds",
                                                      lambda: t.random_(from_, to_))
                            if not (-fp_limit <= (to_ - 1) <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "to - 1 is out of bounds",
                                                      lambda: t.random_(from_, to_))
                        else:
                            t.random_(from_, to_)
                            range_ = to_ - from_
                            delta = max(1, alpha * range_)
                            if dtype == torch.bfloat16:
                                # Less strict checks because of rounding errors
                                # TODO investigate rounding errors
                                self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                            else:
                                self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch.uint16, torch.uint32))
    def test_random_to(self, device, dtype):
        # [FUNCTION DOC] Check random_ with a single upper bound (from defaults to 0) for each
        # dtype: an upper bound whose last value is out of range raises RuntimeError, a
        # non-positive bound raises RuntimeError about from not below to, and valid bounds give
        # min and max near the range ends.
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        from_ = 0
        for to_ in tos:
            t = torch.empty(size, dtype=dtype, device=device)
            if to_ > from_:
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertRaisesRegex(
                        RuntimeError,
                        "to - 1 is out of bounds",
                        lambda: t.random_(from_, to_)
                    )
                else:
                    t.random_(to_)
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    if dtype == torch.bfloat16:
                        # Less strict checks because of rounding errors
                        # TODO investigate rounding errors
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                    else:
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )

    @dtypes(*all_types_and(torch.bfloat16, torch.half))
    def test_random_default(self, device, dtype):
        # [FUNCTION DOC] Check random_ with no arguments spans from zero up to the largest
        # exactly representable integer for the dtype (mantissa limited for floats), checking min
        # and max.
        size = 2000
        alpha = 0.1

        if dtype == torch.float:
            to_inc = 1 << 24
        elif dtype == torch.double:
            to_inc = 1 << 53
        elif dtype == torch.half:
            to_inc = 1 << 11
        elif dtype == torch.bfloat16:
            to_inc = 1 << 8
        else:
            to_inc = torch.iinfo(dtype).max

        t = torch.empty(size, dtype=dtype, device=device)
        t.random_()
        self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)

    # TODO: this test should be updated
    @onlyNativeDeviceTypes
    def test_empty_full(self, device):
        # [FUNCTION DOC] Run the shared empty and full helper for all dtypes (without half and
        # bfloat16 but with complex32) on the strided layout, with the device given as a device
        # object, and on cuda also with no device.
        torch_device = torch.device(device)
        device_type = torch_device.type

        dtypes = get_all_dtypes(include_half=False, include_bfloat16=False, include_complex32=True)
        if device_type == 'cpu':
            do_test_empty_full(self, dtypes, torch.strided, torch_device)
        if device_type == 'cuda':
            do_test_empty_full(self, dtypes, torch.strided, None)
            do_test_empty_full(self, dtypes, torch.strided, torch_device)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyNativeDeviceTypes
    @deviceCountAtLeast(1)
    def test_tensor_device(self, devices):
        # [FUNCTION DOC] Check the device of tensors made by tensor and ones for cpu, and on cuda
        # for the explicit index and string forms, current device context, error moving to cpu
        # through the cuda method, and the second device when available. Warnings are suppressed.
        device_type = torch.device(devices[0]).type
        if device_type == 'cpu':
            self.assertEqual('cpu', torch.tensor(5).device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu').device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu:0').device.type)
            self.assertEqual('cpu',
                             torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cpu:0').device.type)
            self.assertEqual('cpu', torch.tensor(np.random.randn(2, 3), device='cpu').device.type)
        if device_type == 'cuda':
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda(0).device))
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda('cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device=0).device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device='cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cuda:0').device))

            self.assertEqual('cuda:0', str(torch.tensor(np.random.randn(2, 3), device='cuda:0').device))

            for device in devices:
                with torch.cuda.device(device):
                    device_string = 'cuda:' + str(torch.cuda.current_device())
                    self.assertEqual(device_string,
                                     str(torch.tensor(5, dtype=torch.int64, device='cuda').device))

            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu')
            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu:0')

            if len(devices) > 1:
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda(1).device))
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda('cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device=1).device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device='cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(torch.ones((2, 3), dtype=torch.float32),
                                     device='cuda:1').device))

                self.assertEqual('cuda:1',
                                 str(torch.tensor(np.random.randn(2, 3), device='cuda:1').device))

    # TODO: this test should be updated
    @onlyNativeDeviceTypes
    def test_as_strided_neg(self, device):
        # [FUNCTION DOC] Check as_strided with a negative stride raises RuntimeError with the
        # message that negative strides are not supported, for two different inputs.
        error = r'as_strided: Negative strides are not supported at the ' \
                r'moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]'
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(3, 3, device=device), (1, 1), (2, -1))
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(14, device=device), (2,), (-11,))

    # TODO: this test should be updated
    def test_zeros(self, device):
        # [FUNCTION DOC] Check zeros with out, and zeros of bool, half, bfloat16, complex64 and
        # complex32 dtypes equal the expected zero tensors.
        res1 = torch.zeros(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.zeros(100, 100, device=device, out=res2)

        self.assertEqual(res1, res2)

        boolTensor = torch.zeros(2, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[False, False], [False, False]],
                                device=device, dtype=torch.bool)
        self.assertEqual(boolTensor, expected)

        halfTensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[0.]], device=device, dtype=torch.float16)
        self.assertEqual(halfTensor, expected)

        bfloat16Tensor = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
        expected = torch.tensor([[0.]], device=device, dtype=torch.bfloat16)
        self.assertEqual(bfloat16Tensor, expected)

        complexTensor = torch.zeros(2, 2, device=device, dtype=torch.complex64)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex64)
        self.assertEqual(complexTensor, expected)

        complexHalfTensor = torch.zeros(2, 2, device=device, dtype=torch.complex32)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex32)
        self.assertEqual(complexHalfTensor, expected)

    def test_zeros_bounds_checking(self, device):
        # [FUNCTION DOC] Check zeros with a huge negative size raises RuntimeError saying the
        # dimension size must be non-negative.
        # Test negative large integer
        with self.assertRaisesRegex(RuntimeError, r"zeros: Dimension size must be non-negative."):
            torch.zeros(-6744789213055875072, device=device)

    # TODO: this test should be updated
    def test_zeros_out(self, device):
        # [FUNCTION DOC] Check zeros with out: changing dtype or using sparse layout with an
        # existing dense out raises RuntimeError, while leaving dtype, layout and shape
        # matching works.
        shape = (3, 4)
        out = torch.zeros(shape, device=device)
        torch.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, dtype=torch.int64, out=out)
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, layout=torch.sparse_coo, out=out)

        # leave them the same
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, dtype=out.dtype, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, layout=torch.strided, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, out=out))

    # TODO: this test should be updated
    def test_ones(self, device):
        # [FUNCTION DOC] Check ones with out, as a bool tensor, and with chalf dtype versus
        # complex64 values (dtype check relaxed).
        res1 = torch.ones(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.ones(100, 100, device=device, out=res2)
        self.assertEqual(res1, res2)

        # test boolean tensor
        res1 = torch.ones(1, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[True, True]], device=device, dtype=torch.bool)
        self.assertEqual(res1, expected)

        # test chalf
        self.assertEqual(torch.ones(100, 100, device=device, dtype=torch.chalf),
                         torch.ones(100, 100, device=device, dtype=torch.cfloat), exact_dtype=False)

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_dtypes(self, device):
        # [FUNCTION DOC] CPU only: check default dtypes of the legacy tensor types and storages
        # under context managers that set the default tensor type or dtype (including cuda when
        # available); setting a non-float legacy type or a dtype object as tensor type raises
        # TypeError, and only half, float, double and bfloat16 are accepted as default dtype,
        # others raise TypeError.
        self.assertIs(torch.tensor([]).dtype, torch.get_default_dtype())

        self.assertIs(torch.uint8, torch.ByteTensor.dtype)
        self.assertIs(torch.float32, torch.FloatTensor.dtype)
        self.assertIs(torch.float64, torch.DoubleTensor.dtype)

        with set_default_tensor_type('torch.FloatTensor'):
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.FloatStorage, torch.Storage)

        # only floating-point types are supported as the default type
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type('torch.IntTensor'))

        with set_default_dtype(torch.float64):
            self.assertIs(torch.float64, torch.get_default_dtype())
            self.assertIs(torch.DoubleStorage, torch.Storage)

        with set_default_tensor_type(torch.FloatTensor):
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.FloatStorage, torch.Storage)

        if torch.cuda.is_available():
            with set_default_tensor_type(torch.cuda.FloatTensor):
                self.assertIs(torch.float32, torch.get_default_dtype())
                self.assertIs(torch.float32, torch.cuda.FloatTensor.dtype)
                self.assertIs(torch.cuda.FloatStorage, torch.Storage)

                with set_default_dtype(torch.float64):
                    self.assertIs(torch.float64, torch.get_default_dtype())
                    self.assertIs(torch.cuda.DoubleStorage, torch.Storage)

        # don't allow passing dtype to set_default_tensor_type
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type(torch.float32))

        # don't allow passing dtype to set_default_dtype
        for t in all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.qint8):
            # only floating-point types are supported as the default type
            if t in (
                    torch.half,
                    torch.float,
                    torch.double,
                    torch.bfloat16):
                with set_default_dtype(t):
                    pass
            else:
                self.assertRaises(TypeError, lambda: torch.set_default_dtype(t))

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_device_legacy(self, device):
        # [FUNCTION DOC] CPU only: legacy tensor constructors and new called with a mismatched
        # device argument (cuda from cpu types, or cpu from cuda types when available) or with a
        # tensor and a device raise RuntimeError.
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor((2.0, 3.0), device='cuda'))

        self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cuda'))

        # Tensor constructor/new with Tensor argument shouldn't work with device specified
        i = torch.tensor([1], device='cpu')
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))

        x = torch.randn((3,), device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cuda'))

        if torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor((2.0, 3.0), device='cpu'))

            # Tensor constructor/new with Tensor argument shouldn't work with device specified
            i = torch.tensor([1], device='cuda')
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))

            with set_default_tensor_type(torch.cuda.FloatTensor):
                self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cpu'))
                self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cpu'))
                self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cpu'))
            x = torch.randn((3,), device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cpu'))

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory(self, device):
        # [FUNCTION DOC] CPU only: check torch.tensor from lists with and without dtype, copy
        # semantics for tensors and numpy arrays, and bool conversion of floats, nan, huge ints,
        # complex and infinity. Warnings are suppressed.
        # TODO: This test probably doesn't make too much sense now that
        # torch.tensor has been established for a while; it makes more
        # sense to test the legacy behavior in terms of the new behavior
        expected = torch.Tensor([1, 1])
        # test data
        res1 = torch.tensor([1, 1])
        self.assertEqual(res1, expected, exact_dtype=False)

        res1 = torch.tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy
        res2 = torch.tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))

        res2 = torch.tensor(expected, dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy with numpy
        for dtype in [np.float64, np.int64, np.int8, np.uint8]:
            a = np.array([5.]).astype(dtype)
            res1 = torch.tensor(a)
            self.assertEqual(5., res1[0].item())
            a[0] = 7.
            self.assertEqual(5., res1[0].item())

        # test boolean tensor
        a = torch.tensor([True, True, False, True, True], dtype=torch.bool)
        b = torch.tensor([-1, -1.1, 0, 1, 1.1], dtype=torch.bool)
        self.assertEqual(a, b)
        c = torch.tensor([-0.1, -1.1, 0, 1, 0.1], dtype=torch.bool)
        self.assertEqual(a, c)
        d = torch.tensor((-.3, 0, .3, 1, 3 / 7), dtype=torch.bool)
        e = torch.tensor((True, False, True, True, True), dtype=torch.bool)
        self.assertEqual(e, d)
        f = torch.tensor((-1, 0, -1.1, 1, 1.1), dtype=torch.bool)
        self.assertEqual(e, f)

        int64_max = torch.iinfo(torch.int64).max
        int64_min = torch.iinfo(torch.int64).min
        float64_max = torch.finfo(torch.float64).max
        float64_min = torch.finfo(torch.float64).min
        g_1 = torch.tensor((float('nan'), 0, int64_min, int64_max, int64_min - 1), dtype=torch.bool)
        self.assertEqual(e, g_1)
        g_2 = torch.tensor((int64_max + 1, 0, (int64_max + 1) * 2, (int64_max + 1) * 2 + 1, float64_min), dtype=torch.bool)
        self.assertEqual(e, g_2)
        g_3 = torch.tensor((float64_max, 0, float64_max + 1, float64_min - 1, float64_max + 1e291), dtype=torch.bool)
        self.assertEqual(e, g_3)

        h = torch.tensor([True, False, False, True, False, True, True], dtype=torch.bool)
        i = torch.tensor([1e-323, 1e-324, 0j, 1e-323j, 1e-324j, 1 + 2j, -1j], dtype=torch.bool)
        self.assertEqual(h, i)
        j = torch.tensor((True, True, True, True), dtype=torch.bool)
        k = torch.tensor((1e323, -1e323, float('inf'), -float('inf')), dtype=torch.bool)
        self.assertEqual(j, k)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory_copy_var(self, device):
        # [FUNCTION DOC] CPU only: check copy semantics of tensor, new_tensor and as_tensor on a
        # leaf double tensor that requires grad: copies are leaves with the requested
        # requires_grad, as_tensor without a dtype change shares storage and graph state, and
        # with a dtype change copies and keeps the graph. Warnings are suppressed.
        def check_copy(copy, is_leaf, requires_grad, data_ptr=None):
            # [FUNCTION DOC] HELPER: Assert that a tensor equals the outer source tensor and has
            # the given leaf and requires_grad flags, and that its data_ptr attribute equals the
            # expected value (its own when none is given).
            if data_ptr is None:
                data_ptr = copy.data_ptr
            self.assertEqual(copy, source, exact_dtype=False)
            self.assertTrue(copy.is_leaf == is_leaf)
            self.assertTrue(copy.requires_grad == requires_grad)
            self.assertTrue(copy.data_ptr == data_ptr)

        source = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        # test torch.tensor()
        check_copy(torch.tensor(source), True, False)
        check_copy(torch.tensor(source, requires_grad=False), True, False)
        check_copy(torch.tensor(source, requires_grad=True), True, True)

        # test tensor.new_tensor()
        copy = torch.randn(1)
        check_copy(copy.new_tensor(source), True, False)
        check_copy(copy.new_tensor(source, requires_grad=False), True, False)
        check_copy(copy.new_tensor(source, requires_grad=True), True, True)

        # test torch.as_tensor()
        check_copy(torch.as_tensor(source), source.is_leaf, source.requires_grad, source.data_ptr)  # not copy
        check_copy(torch.as_tensor(source, dtype=torch.float), False, True)  # copy and keep the graph

    # TODO: this test should be updated
    @onlyCPU
    def test_tensor_factory_type_inference(self, device):
        # [FUNCTION DOC] CPU only: check the dtype of tensors built from Python scalars and
        # sequences under different default dtypes, via a nested helper.
        def test_inference(default_dtype):
            # [FUNCTION DOC] HELPER: For a given default dtype, assert the dtype inferred by
            # torch.tensor for empty, float, int, bool, explicit dtype, mixed int/float, complex,
            # and numpy-derived inputs (numpy scalars keep their own dtype; mixed numpy and
            # Python values follow promotion rules).
            default_complex_dtype = torch.complex64 if default_dtype == torch.float32 else torch.complex128
            self.assertIs(default_dtype, torch.tensor(()).dtype)
            self.assertIs(default_dtype, torch.tensor(5.).dtype)
            self.assertIs(torch.int64, torch.tensor(5).dtype)
            self.assertIs(torch.bool, torch.tensor(True).dtype)
            self.assertIs(torch.int32, torch.tensor(5, dtype=torch.int32).dtype)
            self.assertIs(default_dtype, torch.tensor(((7, 5), (9, 5.))).dtype)
            self.assertIs(default_dtype, torch.tensor(((5., 5), (3, 5))).dtype)
            self.assertIs(torch.int64, torch.tensor(((5, 3), (3, 5))).dtype)
            self.assertIs(default_complex_dtype, torch.tensor(((5, 3 + 2j), (3, 5 + 4j))).dtype)

            self.assertIs(torch.float64, torch.tensor(np.array(())).dtype)
            self.assertIs(torch.float64, torch.tensor(np.array(5.)).dtype)
            if np.array(5).dtype == np.int64:  # np long, which can be 4 bytes (e.g. on windows)
                self.assertIs(torch.int64, torch.tensor(np.array(5)).dtype)
            else:
                self.assertIs(torch.int32, torch.tensor(np.array(5)).dtype)
            self.assertIs(torch.uint8, torch.tensor(np.array(3, dtype=np.uint8)).dtype)
            self.assertIs(default_dtype, torch.tensor(((7, np.array(5)), (np.array(9), 5.))).dtype)
            self.assertIs(torch.float64, torch.tensor(((7, 5), (9, np.array(5.)))).dtype)
            self.assertIs(torch.int64, torch.tensor(((5, np.array(3)), (np.array(3), 5))).dtype)

        for dtype in [torch.float64, torch.float32]:
            with set_default_dtype(dtype):
                test_inference(dtype)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_new_tensor(self, device):
        # [FUNCTION DOC] CPU only: check new_tensor on a byte tensor: values and dtype override,
        # copy semantics for tensors and numpy arrays (later source edits do not show through),
        # and when two cuda devices exist, that the result stays on the source device unless a
        # device is given. Warnings are suppressed.
        expected = torch.autograd.Variable(torch.ByteTensor([1, 1]))
        # test data
        res1 = expected.new_tensor([1, 1])
        self.assertEqual(res1, expected)
        res1 = expected.new_tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy
        res2 = expected.new_tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))
        res2 = expected.new_tensor(expected, dtype=torch.int)
        self.assertEqual(res2, expected, exact_dtype=False)
        self.assertIs(torch.int, res2.dtype)

        # test copy with numpy
        a = np.array([5.])
        res1 = torch.tensor(a)
        res1 = res1.new_tensor(a)
        self.assertEqual(5., res1[0].item())
        a[0] = 7.
        self.assertEqual(5., res1[0].item())

        if torch.cuda.device_count() >= 2:
            expected = expected.cuda(1)
            res1 = expected.new_tensor([1, 1])
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor([1, 1], dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

            res2 = expected.new_tensor(expected)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=torch.int, device=0)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res2.get_device(), 0)

            res1 = expected.new_tensor(1)
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor(1, dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

    # TODO: this test should be updated
    @onlyCPU
    def test_as_tensor(self, device):
        # [FUNCTION DOC] CPU only: check as_tensor against tensor: equal for Python data,
        # TypeError for mixed-type or self-referential lists, no copy for a tensor of unchanged
        # dtype and device, shared memory with numpy arrays of several dtypes, and a copy when
        # dtype changes (or device, with cuda).
        # from python data
        x = [[0, 1], [2, 3]]
        self.assertEqual(torch.tensor(x), torch.as_tensor(x))
        self.assertEqual(torch.tensor(x, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32))

        # python data with heterogeneous types
        z = [0, 'torch']
        with self.assertRaisesRegex(TypeError, "invalid data type"):
            torch.tensor(z)
            torch.as_tensor(z)

        # python data with self-referential lists
        z = [0]
        z += [z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            torch.tensor(z)
            torch.as_tensor(z)

        z = [[1, 2], z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            torch.tensor(z)
            torch.as_tensor(z)

        # from tensor (doesn't copy unless type is different)
        y = torch.tensor(x)
        self.assertIs(y, torch.as_tensor(y))
        self.assertIsNot(y, torch.as_tensor(y, dtype=torch.float32))
        if torch.cuda.is_available():
            self.assertIsNot(y, torch.as_tensor(y, device='cuda'))
            y_cuda = y.to('cuda')
            self.assertIs(y_cuda, torch.as_tensor(y_cuda))
            self.assertIs(y_cuda, torch.as_tensor(y_cuda, device='cuda'))

        # doesn't copy
        for dtype in [np.float64, np.int64, np.int8, np.uint8]:
            n = np.random.rand(5, 6).astype(dtype)
            n_astensor = torch.as_tensor(n)
            self.assertEqual(torch.tensor(n), n_astensor)
            n_astensor[0][0] = 25.7
            self.assertEqual(torch.tensor(n), n_astensor)

        # changing dtype causes copy
        n = np.random.rand(5, 6).astype(np.float32)
        n_astensor = torch.as_tensor(n, dtype=torch.float64)
        self.assertEqual(torch.tensor(n, dtype=torch.float64), n_astensor)
        n_astensor[0][1] = 250.8
        self.assertNotEqual(torch.tensor(n, dtype=torch.float64), n_astensor)

        # changing device causes copy
        if torch.cuda.is_available():
            n = np.random.randn(5, 6)
            n_astensor = torch.as_tensor(n, device='cuda')
            self.assertEqual(torch.tensor(n, device='cuda'), n_astensor)
            n_astensor[0][2] = 250.9
            self.assertNotEqual(torch.tensor(n, device='cuda'), n_astensor)

    # TODO: this test should be updated
    @suppress_warnings
    @dtypesIfCPU(torch.float, torch.bfloat16, torch.float16)
    @dtypes(torch.float)
    def test_range(self, device, dtype):
        # [FUNCTION DOC] Check the deprecated range function (inclusive end): equal to the out
        # form, writing into a non-contiguous narrowed out view, negative step, and equal bounds.
        # Warnings are suppressed; also runs for bfloat16 and float16 on CPU.
        res1 = torch.range(0, 1, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.range(0, 1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check range for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        torch.range(0, 3, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        res2 = torch.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=dtype)
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = torch.tensor((1, 0), device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.range(1, 0, -1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = torch.ones(1, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.range(1, 1, -1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        torch.range(1, 1, 1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

    # TODO: this test should be updated
    def test_range_warning(self, device):
        # [FUNCTION DOC] Check calling range emits exactly one warning.
        with warnings.catch_warnings(record=True) as w:
            torch.range(0, 10, device=device)
            self.assertEqual(len(w), 1)

    # TODO: this test should be updated
    def test_arange(self, device):
        # [FUNCTION DOC] Broad arange check: matches Python range for 10000 elements, out into
        # non-contiguous views, one-argument form, negative and equal bounds, float steps, bool
        # inputs (giving int64), exclusive end semantics, rounding cases, resize warning for out,
        # RuntimeError for an overlapping out, unsupported range for nan or infinite bounds,
        # overflow error, and a stable length for a tricky float step.
        res = torch.tensor(range(10000), device=device)
        res1 = torch.arange(0, 10000, device=device)  # Use a larger number so vectorized code can be triggered
        res2 = torch.tensor([], dtype=torch.int64, device=device)
        torch.arange(0, 10000, out=res2)
        self.assertEqual(res, res1, atol=0, rtol=0)
        self.assertEqual(res, res2, atol=0, rtol=0)

        # Vectorization on non-contiguous tensors
        res = torch.rand(3, 3, 300000, device=device).to(torch.int64)
        res = res.permute(2, 0, 1)
        torch.arange(0, 300000 * 3 * 3, out=res)
        self.assertEqual(res.flatten(), torch.arange(0, 300000 * 3 * 3, device=device))

        # Check arange with only one argument
        res1 = torch.arange(10, device=device)
        res2 = torch.arange(0, 10, device=device)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check arange for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device)
        torch.arange(0, 4, out=x.narrow(1, 1, 2))
        res2 = torch.tensor(((0., 0., 1.), (0., 2., 3.)), device=device)
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = torch.tensor((1., 0.), device=device)
        res2 = torch.tensor([], device=device)
        torch.arange(1, -1, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = torch.ones(1, device=device)
        res2 = torch.tensor([], device=device)
        torch.arange(1, 0, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        torch.arange(1, 2, 1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # FloatTensor
        out = torch.tensor([], dtype=torch.float, device=device)
        res1 = torch.arange(0.6, 0.89, 0.1, out=out)
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        out = torch.tensor([], dtype=torch.float, device=device)
        res1 = torch.arange(1, 10, 0.3, out=out)
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # DoubleTensor
        out = torch.tensor([], dtype=torch.double, device=device)
        res1 = torch.arange(0.6, 0.89, 0.1, out=out)
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        out = torch.tensor([], dtype=torch.double, device=device)
        res1 = torch.arange(1, 10, 0.3, out=out)
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # Bool Input matching numpy semantics
        r = torch.arange(True, device=device)
        self.assertEqual(r[0], 0)
        r2 = torch.arange(False, device=device)
        self.assertEqual(len(r2), 0)
        self.assertEqual(r.dtype, torch.int64)
        self.assertEqual(r2.dtype, torch.int64)

        # Check that it's exclusive
        r = torch.arange(0, 5, device=device)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 5)

        r = torch.arange(0, 6, 3, device=device)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 3)
        self.assertEqual(r.numel(), 2)

        r = torch.arange(0, 5, 2, device=device)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 3)

        r = torch.arange(0, -5, -2, device=device)
        self.assertEqual(r.min(), -4)
        self.assertEqual(r.max(), 0)
        self.assertEqual(r.numel(), 3)

        r1 = torch.arange(0, 5 + 1e-6, device=device)
        # NB: without the dtype, we'll infer output type to be int64
        r2 = torch.arange(0, 5, dtype=torch.float32, device=device)
        r3 = torch.arange(0, 5 - 1e-6, device=device)
        self.assertEqual(r1[:-1], r2, atol=0, rtol=0)
        self.assertEqual(r2, r3, atol=0, rtol=0)

        r1 = torch.arange(10, -1 + 1e-6, -1, device=device)
        # NB: without the dtype, we'll infer output type to be int64
        r2 = torch.arange(10, -1, -1, dtype=torch.float32, device=device)
        r3 = torch.arange(10, -1 - 1e-6, -1, device=device)
        self.assertEqual(r1, r2, atol=0, rtol=0)
        self.assertEqual(r2, r3[:-1], atol=0, rtol=0)

        w = 1449629115440469
        r = torch.arange(0, 100 * w, w, device=device)
        self.assertEqual(r.numel(), 100)

        # Test Rounding Errors
        line = torch.zeros(size=(1, 49), device=device)
        self.assertWarnsRegex(UserWarning, 'The out tensor will be resized',
                              lambda: torch.arange(-1, 1, 2. / 49, dtype=torch.float32, out=line))
        self.assertEqual(line.shape, [50])

        x = torch.empty(1).expand(10)
        self.assertRaises(RuntimeError, lambda: torch.arange(10, out=x))

        msg = "unsupported range"
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(-5, float('nan'), device=device))
        # check with step size
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('-inf'), -1, device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('inf'), device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('-inf'), 10, device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('nan'), 10, device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('inf'), device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('nan'), device=device))

        self.assertRaisesRegex(
            RuntimeError, "overflow",
            lambda: torch.arange(1.175494351e-38, 3.402823466e+38, device=device))

        # check that it holds a consistent output shape on precision-cornered step sizes
        d = torch.arange(-4.0, 4.0, 0.01, dtype=torch.float32, device=device)
        self.assertEqual(d.shape[0], 800)

    # TODO: this test should be updated
    @onlyCPU
    def test_arange_inference(self, device):
        # [FUNCTION DOC] CPU only: check the dtype arange infers from Python numbers and 0-D
        # tensors: float32 if any argument is floating (even float64 tensors), otherwise int64
        # (even int16 tensors).
        # end only
        self.assertIs(torch.float32, torch.arange(1.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1.)).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1)).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1, dtype=torch.int16)).dtype)

        # start, end, [step]
        self.assertIs(torch.float32, torch.arange(1., 3).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64), 3).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1, dtype=torch.int16), torch.tensor(3.)).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3, 1.).dtype)
        self.assertIs(torch.float32,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3, dtype=torch.int16),
                                   torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1, 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), torch.tensor(3, dtype=torch.int16)).dtype)
        self.assertIs(torch.int64, torch.arange(1, 3, 1).dtype)
        self.assertIs(torch.int64,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3),
                                   torch.tensor(1, dtype=torch.int16)).dtype)

    # cannot call storage() on meta tensor
    @skipMeta
    def test_empty_strided(self, device):
        # [FUNCTION DOC] Check empty_strided has the same shape and strides as an as_strided view
        # over a storage of the same size, for several odd shape and stride pairs. Skipped for
        # meta tensors.
        for shape in [(2, 3, 4), (0, 2, 0)]:
            # some of these cases are pretty strange, just verifying that if as_strided
            # allows them then empty_strided can as well.
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                empty_strided = torch.empty_strided(shape, strides, device=device)
                # as_strided checks the storage size is big enough to support such a strided tensor;
                # instead of repeating this calculation, we just use empty_strided which does the same
                # calculation when setting the storage size.
                as_strided = torch.empty(empty_strided.storage().size(),
                                         device=device).as_strided(shape, strides)
                self.assertEqual(empty_strided.shape, as_strided.shape)
                self.assertEqual(empty_strided.stride(), as_strided.stride())

    def test_new_empty_strided(self, device):
        # [FUNCTION DOC] Check new_empty_strided matches empty_strided on shape, stride, dtype
        # and device for assorted sizes and strides, and raises RuntimeError when sizes and
        # strides lengths differ.
        def _test(sizes, strides, dtype):
            # [FUNCTION DOC] HELPER: Make a zeros source tensor of the dtype, call
            # new_empty_strided with the sizes and strides and assert the result has the same
            # shape, strides, dtype and device as empty_strided with the source dtype and device.
            x = torch.zeros(5, 5, dtype=dtype, device=device)
            result = x.new_empty_strided(sizes, strides)
            expected = torch.empty_strided(sizes, strides, dtype=x.dtype, device=x.device)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.stride(), expected.stride())
            self.assertEqual(result.dtype, expected.dtype)
            self.assertEqual(result.device, expected.device)

        _test([2, 3], [3, 1], torch.float)
        _test([5, 3], [0, 1], torch.int)
        _test([], [], torch.float)

        # Some really weird cases
        for shape in [(2, 3, 4), (0, 2, 0)]:
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                _test(shape, strides, torch.float)

        # Make sure sizes and strides have the same length
        # https://github.com/pytorch/pytorch/issues/82416
        with self.assertRaisesRegex(
                RuntimeError,
                r"dimensionality of sizes \(1\) must match dimensionality of strides \(0\)"):
            dtype = torch.float64
            x = torch.tensor(-4.8270, dtype=dtype, device=device)
            size = (2,)
            stride = ()
            x.new_empty_strided(size, stride, dtype=dtype, device=device)

    def test_strided_mismatched_stride_shape(self, device):
        # [FUNCTION DOC] Check as_strided and as_strided_ raise RuntimeError about a mismatch in
        # the length of strides and shape when the two lengths differ.
        for shape, strides in [((1, ), ()), ((1, 2), (1, ))]:
            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                torch.tensor(0.42, device=device).as_strided(shape, strides)

            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                torch.tensor(0.42, device=device).as_strided_(shape, strides)

    def test_empty_tensor_props(self, device):
        # [FUNCTION DOC] Check empty tensors with zero-size dims are contiguous, have the
        # requested shape, and have the same strides as a tensor with zero sizes replaced by one.
        sizes = [(0,), (0, 3), (5, 0), (5, 0, 3, 0, 2), (0, 3, 0, 2), (0, 5, 0, 2, 0)]
        for size in sizes:
            x = torch.empty(tuple(size), device=device)
            self.assertEqual(size, x.shape)
            self.assertTrue(x.is_contiguous())
            size_ones_instead_of_zeros = (x if x != 0 else 1 for x in size)
            y = torch.empty(tuple(size_ones_instead_of_zeros), device=device)
            self.assertEqual(x.stride(), y.stride())

    @onlyNativeDeviceTypes
    def test_empty_overflow(self, device):
        # [FUNCTION DOC] Check empty and empty_strided raise RuntimeError for sizes whose storage
        # size or strides overflow.
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            torch.empty([2, 4, 2**29, 2**29], dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            torch.empty([8, 8, 2**29, 2**29], dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            torch.empty_strided([8, 8], [2**61, 1], dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'Stride calculation overflowed'):
            torch.empty([0, 4, 2305843009213693952], dtype=torch.float32)

    def test_eye(self, device):
        # [FUNCTION DOC] Check eye for all dtypes except bfloat16: negative n or m raises
        # RuntimeError, and one-argument and two-argument forms equal a zero matrix with ones on
        # the diagonal, also through the out form.
        for dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if dtype == torch.bfloat16:
                continue
            # Test the RuntimeError is raised when either m or n is a negative number
            for n, m in ((-1, 1), (1, -1), (-1, -1)):
                with self.assertRaisesRegex(RuntimeError, 'must be greater or equal to'):
                    torch.eye(n, m, device=device, dtype=dtype)

            # Test when the `m` parameter is not provided
            for n in (3, 5, 7):
                res1 = torch.eye(n, device=device, dtype=dtype)
                naive_eye = torch.zeros(n, n, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # Check eye_out outputs
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch.eye(n, out=res2)
                self.assertEqual(res1, res2)

            for n, m in product([3, 5, 7], repeat=2):
                # Construct identity using diagonal and fill
                res1 = torch.eye(n, m, device=device, dtype=dtype)
                naive_eye = torch.zeros(n, m, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # Check eye_out outputs
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch.eye(n, m, out=res2)
                self.assertEqual(res1, res2)

    @precisionOverride({torch.float: 1e-8, torch.double: 1e-10})
    @dtypes(*floating_and_complex_types())
    def test_linspace_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compare linspace with numpy for floating and complex dtypes over many
        # step counts (up to 4 million), requiring exact first and last values, with tight
        # tolerance overrides.
        start = -0.0316082797944545745849609375 + (0.8888888888j if dtype.is_complex else 0)
        end = .0315315723419189453125 + (0.444444444444j if dtype.is_complex else 0)

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = torch.linspace(start, end, steps, device=device, dtype=dtype)
            a = np.linspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))
            self.assertTrue(t[0].item() == a[0])
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    @dtypes(*integral_types())
    def test_linspace_vs_numpy_integral(self, device, dtype):
        # [FUNCTION DOC] Compare linspace with numpy for integer dtypes between 1 and 127 with 25
        # and 50 steps, requiring exact first and last values.
        start = 1
        end = 127

        for steps in [25, 50]:
            t = torch.linspace(start, end, steps, device=device, dtype=dtype)
            a = np.linspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))
            self.assertTrue(t[0].item() == a[0])
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    def _test_linspace_logspace_complex_helper(self, torch_fn, np_fn, device, dtype):
        # [FUNCTION DOC] HELPER: Pick a random complex start and an end shifted by a random
        # amount, then for many step counts compare the given torch function with the given numpy
        # function using a nested comparison helper. Only linspace is passed by the caller in
        # this file.
        start = torch.randn(1, dtype=dtype).item()
        end = (start + torch.randn(1, dtype=dtype) + random.randint(5, 15)).item()

        def test_fn(torch_fn, numpy_fn, steps):
            # [FUNCTION DOC] HELPER: Call the torch and numpy functions with the outer start and
            # end and the given step count, move the torch result to CPU and assert it equals the
            # numpy result.
            t = torch_fn(start, end, steps, device=device)
            a = numpy_fn(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            test_fn(torch.linspace, np.linspace, steps)

    @dtypes(torch.complex64)
    def test_linspace_vs_numpy_complex(self, device, dtype):
        # [FUNCTION DOC] Run the linspace complex helper for complex64 using linspace and
        # numpy linspace.
        self._test_linspace_logspace_complex_helper(torch.linspace, np.linspace,
                                                    device, dtype)

    @dtypes(torch.complex64)
    def test_logspace_vs_numpy_complex(self, device, dtype):
        # [FUNCTION DOC] Run the linspace complex helper for complex64 using logspace and
        # numpy logspace.
        self._test_linspace_logspace_complex_helper(torch.logspace, np.logspace,
                                                    device, dtype)

    @precisionOverride({torch.float: 1e-6, torch.double: 1e-10})
    @dtypes(*floating_types())
    def test_logspace_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compare logspace with numpy for floating dtypes over many step counts
        # (up to 4 million) with tight precision overrides, also requiring matching first and
        # last elements.
        start = -0.0316082797944545745849609375
        end = .0315315723419189453125

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = torch.logspace(start, end, steps, device=device, dtype=dtype)
            a = np.logspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))
            self.assertEqual(t[0], a[0])
            self.assertEqual(t[steps - 1], a[steps - 1])

    @onlyCUDA
    @largeTensorTest('16GB')
    def test_range_factories_64bit_indexing(self, device):
        # [FUNCTION DOC] CUDA only, needs 16GB: check arange, linspace and logspace with more
        # than 2 to the 31 elements return the right last value (index math must be 64-bit). On
        # ROCm, also tries sizes just above and far above 2 to the 32 when free memory allows,
        # checking the int32 wrapped trailing values.
        bigint = 2 ** 31 + 1
        t = torch.arange(bigint, dtype=torch.long, device=device)
        self.assertEqual(t[-1].item(), bigint - 1)
        del t
        t = torch.linspace(0, 1, bigint, dtype=torch.float, device=device)
        self.assertEqual(t[-1].item(), 1)
        del t
        t = torch.logspace(0, 1, bigint, 2, dtype=torch.float, device=device)
        self.assertEqual(t[-1].item(), 2)
        del t

        # On ROCm, launches with gridDim.x * blockDim.x >= 2^32 are not
        # supported and either return hipErrorInvalidConfiguration or fail
        # silently. Exercise the just-over case (~16 GB at int32) and a
        # far-above case (~33 GB) when memory permits. arange computes
        # int64 values then casts down, so for int32 the trailing
        # 2^32 - 2, 2^32 - 1, 2^32 wrap to -2, -1, 0.
        if TEST_WITH_ROCM:
            for bigint in (2 ** 32 + 1, 2 ** 33 + 1):
                free, _ = torch.cuda.mem_get_info(device)
                if free < bigint * 4 + (3 << 30):
                    continue
                t = torch.arange(bigint, dtype=torch.int32, device=device)
                self.assertEqual(t.numel(), bigint)
                self.assertEqual(
                    t[-3:].cpu(), torch.tensor([-2, -1, 0], dtype=torch.int32)
                )
                del t

    @expectedFailureMeta  # RuntimeError: The tensor has a non-zero number of elements
    @onlyNativeDeviceTypes
    def test_tensor_ctor_device_inference(self, device):
        # [FUNCTION DOC] Check that tensor, as_tensor and sparse_coo_tensor keep the device of
        # tensor inputs unless a cpu input is given under a cuda device context, in which case
        # the result stays on cpu. Expected to fail on meta device.
        torch_device = torch.device(device)
        values = torch.tensor((1, 2, 3), device=device)

        # Tests tensor and as_tensor
        # Note: warnings are suppressed (suppresses warnings)
        for op in (torch.tensor, torch.as_tensor):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(op(values).device, torch_device)
                self.assertEqual(op(values, dtype=torch.float64).device, torch_device)

                if self.device_type == 'cuda':
                    with torch.cuda.device(device):
                        self.assertEqual(op(values.cpu()).device, torch.device('cpu'))

        # Tests sparse ctor
        indices = torch.tensor([[0, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0]], device=device)
        sparse_size = (3, 3, 3)

        sparse_default = torch.sparse_coo_tensor(indices, values, sparse_size)
        self.assertEqual(sparse_default.device, torch_device)

        sparse_with_dtype = torch.sparse_coo_tensor(indices, values, sparse_size, dtype=torch.float64)
        self.assertEqual(sparse_with_dtype.device, torch_device)

        if self.device_type == 'cuda':
            with torch.cuda.device(device):
                sparse_with_dtype = torch.sparse_coo_tensor(indices.cpu(), values.cpu(),
                                                            sparse_size, dtype=torch.float64)
                self.assertEqual(sparse_with_dtype.device, torch.device('cpu'))

    @onlyCUDA
    @onlyNativeDeviceTypes
    def test_new_tensor_device(self, device):
        # [FUNCTION DOC] CUDA only: check new_tensor's device for all combinations of cpu and
        # device data as source and as argument: no device argument keeps the source tensor's
        # device, an explicit device argument wins.
        torch_device = torch.device(device)
        cpu_device = torch.device('cpu')
        tensor = torch.tensor((1, 2, 3), device=device)

        # need a non-CPU device (with its own device memory) to test this
        if self.device_type == 'cpu':
            raise AssertionError(f"expected a non-CPU device, got {self.device_type!r}")
        for left, right in product([tensor, tensor.cpu()], [tensor, tensor.cpu()]):
            for device_arg in [torch_device, cpu_device, None]:
                if device_arg is None:
                    self.assertEqual(left.new_tensor(right).device, left.device)
                else:
                    self.assertEqual(left.new_tensor(right, device=device_arg).device, device_arg)

    def _test_signal_window_functions(self, name, dtype, device, **kwargs):
        # [FUNCTION DOC] HELPER: Compare a torch window function (named by prefix plus window)
        # against scipy signal windows for several sizes in periodic and symmetric forms.
        # Non-floating dtypes must raise RuntimeError mentioning floating point; sparse layout
        # raises RuntimeError; requires_grad is honored. Extra keyword arguments are forwarded
        # and also used as scipy parameters.
        import scipy.signal as signal

        torch_method = getattr(torch, name + '_window')
        if not dtype.is_floating_point:
            with self.assertRaisesRegex(RuntimeError, r'floating point'):
                torch_method(3, dtype=dtype)
            return
        for size in [1, 2, 5, 10, 50, 100, 1024, 2048]:
            for periodic in [True, False]:
                res = torch_method(
                    size,
                    periodic=periodic,
                    layout=torch.strided,
                    requires_grad=False,
                    **kwargs,
                    device=device,
                    dtype=dtype,
                )
                # NB: scipy always returns a float64 result
                ref = torch.from_numpy(
                    signal.get_window(
                        (name, *(kwargs.values())), size, fftbins=periodic
                    )
                )
                self.assertEqual(res, ref.to(dtype))
        with self.assertRaisesRegex(RuntimeError, r'not implemented for sparse types'):
            torch_method(3, layout=torch.sparse_coo)
        self.assertTrue(torch_method(3, requires_grad=True).requires_grad)
        self.assertFalse(torch_method(3).requires_grad)

    @onlyNativeDeviceTypes
    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16, torch.half, torch.long)
    @dtypes(torch.float, torch.double, torch.long)
    @parametrize("window", ['hann', 'hamming', 'bartlett', 'blackman'])
    def test_signal_window_functions(self, device, dtype, window):
        # [FUNCTION DOC] Run the signal window helper for hann, hamming, bartlett and
        # blackman windows.
        self._test_signal_window_functions(window, dtype, device)

    @onlyNativeDeviceTypes
    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16, torch.half, torch.long)
    @dtypes(torch.float, torch.double, torch.long, torch.bfloat16, torch.float16)
    def test_kaiser_window(self, device, dtype):
        # [FUNCTION DOC] Run the signal window helper fifty times for the kaiser window with a
        # random beta between 0 and 30.
        for _ in range(50):
            self._test_signal_window_functions('kaiser', dtype, device, beta=random.random() * 30)

    def _test_signal_windows_functions(self, name, dtype, device, **kwargs):
        # [FUNCTION DOC] HELPER: Same idea as the window helper above but for functions in the
        # signal windows module, which take a symmetric flag (opposite of periodic). Checks
        # non-floating dtypes raise RuntimeError, values match scipy without requiring equal
        # dtype, and requires_grad works.
        import scipy.signal as signal

        torch_method = getattr(torch.signal.windows, name)
        if not dtype.is_floating_point:
            with self.assertRaisesRegex(RuntimeError, r'floating point'):
                torch_method(3, dtype=dtype)
            return
        for size in [1, 2, 5, 10, 50, 100, 1024, 2048]:
            for periodic in [True, False]:
                res = torch_method(size, sym=not periodic, **kwargs, device=device, dtype=dtype)
                # NB: scipy always returns a float64 result
                ref = torch.from_numpy(signal.get_window((name, *(kwargs.values())), size, fftbins=periodic))
                self.assertEqual(res, ref, exact_dtype=False)
        self.assertTrue(torch_method(3, requires_grad=True).requires_grad)
        self.assertFalse(torch_method(3).requires_grad)

    # torch.signal.windows functions (except any with extra parameters)
    @onlyNativeDeviceTypes
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypes(torch.float, torch.double)
    @parametrize("window", ['bartlett', 'blackman', 'cosine', 'hamming', 'hann', 'nuttall'])
    def test_signal_windows_functions(self, device, dtype, window):
        # [FUNCTION DOC] Run the signal windows helper for bartlett, blackman, cosine, hamming,
        # hann and nuttall windows.
        self._test_signal_windows_functions(window, dtype, device)

    # torch.signal.windows.kaiser
    @onlyNativeDeviceTypes
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypes(torch.float, torch.double)
    def test_kaiser(self, device, dtype):
        # [FUNCTION DOC] Run the signal windows helper fifty times for kaiser with a random beta.
        for _ in range(50):
            self._test_signal_windows_functions('kaiser', dtype, device, beta=random.random() * 30)

    def test_tensor_factories_empty(self, device):
        # [FUNCTION DOC] Check that every factory (zeros, full, ones, empty, empty_strided,
        # randint, rand, randn, their like variants, arange, eye, linspace, logspace,
        # randperm, window functions, and tensor from nested empty lists) creates empty
        # tensors of the right shape for zero-size shapes and many dtypes, with errors for
        # unsupported dtype combinations.
        # ensure we can create empty tensors from each factory function
        shapes = [(5, 0, 1), (0,), (0, 0, 1, 0, 2, 0, 0)]

        for shape in shapes:
            for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.chalf):

                self.assertEqual(shape, torch.zeros(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.zeros_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.full(shape, 3, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.full_like(torch.zeros(shape, device=device, dtype=dt), 3).shape)
                self.assertEqual(shape, torch.ones(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.ones_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.empty(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.empty_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.empty_strided(shape, (0,) * len(shape), device=device, dtype=dt).shape)

                if dt == torch.bool:
                    self.assertEqual(shape, torch.randint(2, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randint_like(torch.zeros(shape, device=device, dtype=dt), 2).shape)
                elif dt.is_complex:
                    self.assertRaises(RuntimeError, lambda: torch.randint(6, shape, device=device, dtype=dt).shape)
                else:
                    self.assertEqual(shape, torch.randint(6, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randint_like(torch.zeros(shape, device=device, dtype=dt), 6).shape)

                if dt not in {torch.double, torch.float, torch.half, torch.bfloat16,
                              torch.complex32, torch.complex64, torch.complex128}:
                    self.assertRaises(RuntimeError, lambda: torch.rand(shape, device=device, dtype=dt).shape)

                if dt == torch.double or dt == torch.float or dt.is_complex:
                    self.assertEqual(shape, torch.randn(shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randn_like(torch.zeros(shape, device=device, dtype=dt)).shape)

        self.assertEqual((0,), torch.arange(0, device=device).shape)
        self.assertEqual((0, 0), torch.eye(0, device=device).shape)
        self.assertEqual((0, 0), torch.eye(0, 0, device=device).shape)
        self.assertEqual((5, 0), torch.eye(5, 0, device=device).shape)
        self.assertEqual((0, 5), torch.eye(0, 5, device=device).shape)
        self.assertEqual((0,), torch.linspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), torch.logspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), torch.randperm(0, device=device).shape)
        self.assertEqual((0,), torch.bartlett_window(0, device=device).shape)
        self.assertEqual((0,), torch.bartlett_window(0, periodic=False, device=device).shape)
        self.assertEqual((0,), torch.hamming_window(0, device=device).shape)
        self.assertEqual((0,), torch.hann_window(0, device=device).shape)
        self.assertEqual((0,), torch.kaiser_window(0, device=device).shape)
        self.assertEqual((1, 1, 0), torch.tensor([[[]]], device=device).shape)
        self.assertEqual((1, 1, 0), torch.as_tensor([[[]]], device=device).shape)

    @onlyCUDA
    def test_tensor_factory_gpu_type_inference(self, device):
        # [FUNCTION DOC] CUDA only: tensor dtype follows set_default_dtype and device is cuda
        # while the default tensor type is a cuda double type.
        with set_default_tensor_type(torch.cuda.DoubleTensor):
            with set_default_dtype(torch.float32):
                self.assertIs(torch.float32, torch.tensor(0.).dtype)
                self.assertEqual(torch.device(device), torch.tensor(0.).device)
            with set_default_dtype(torch.float64):
                self.assertIs(torch.float64, torch.tensor(0.).dtype)
                self.assertEqual(torch.device(device), torch.tensor(0.).device)

    @onlyCUDA
    def test_tensor_factory_gpu_type(self, device):
        # [FUNCTION DOC] CUDA only: zeros uses the dtype and cuda device of the default tensor
        # type set to a cuda float or double type.
        with set_default_tensor_type(torch.cuda.FloatTensor):
            x = torch.zeros((5, 5))
            self.assertIs(torch.float32, x.dtype)
            self.assertTrue(x.is_cuda)
        with set_default_tensor_type(torch.cuda.DoubleTensor):
            x = torch.zeros((5, 5))
            self.assertIs(torch.float64, x.dtype)
            self.assertTrue(x.is_cuda)

    @skipCPUIf(True, 'compares device with cpu')
    @dtypes(torch.int, torch.long, torch.float, torch.double)
    def test_arange_device_vs_cpu(self, device, dtype):
        # [FUNCTION DOC] Non-CPU only: arange on the device equals arange on cpu for int, long,
        # float and double.
        cpu_tensor = torch.arange(0, 10, dtype=dtype, device='cpu')
        device_tensor = torch.arange(0, 10, dtype=dtype, device=device)
        self.assertEqual(cpu_tensor, device_tensor)

    @dtypes(torch.bfloat16, torch.float16)
    def test_arange_lowp(self, device, dtype):
        # [FUNCTION DOC] Check arange for bfloat16 and float16, with and without a step, against
        # explicit tensors.
        ref_tensor = torch.tensor([0, 1, 2, 3], dtype=dtype, device=device)
        f16_tensor = torch.arange(0, 4, dtype=dtype, device=device)
        self.assertEqual(ref_tensor, f16_tensor)

        # step=2
        ref_tensor = torch.tensor([0, 2, 4], dtype=dtype, device=device)
        f16_tensor = torch.arange(0, 6, step=2, dtype=dtype, device=device)
        self.assertEqual(ref_tensor, f16_tensor)

    @dtypes(*all_types_and_complex_and(torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.bfloat16))
    def test_linspace(self, device, dtype):
        # [FUNCTION DOC] Broad linspace check across dtypes: matches out form, small and large
        # exact integer ranges, non-contiguous vectorized output, negative steps error, one and
        # zero steps, missing steps TypeError, inferred-dtype RuntimeError for float with complex
        # endpoints, value accuracy with device tolerances, start above end, monotonic result for
        # a million steps, and writing to a narrowed out view.
        _from = random.random()
        to = _from + random.random()
        res1 = torch.linspace(_from, to, 137, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.linspace(_from, to, 137, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # small tensor
        self.assertEqual(torch.linspace(10, 20, 11, device=device, dtype=dtype),
                         torch.tensor(list(range(10, 21)), device=device, dtype=dtype))
        # large tensor
        if dtype not in (torch.int8, torch.uint8):
            self.assertEqual(torch.linspace(10, 2000, 1991, device=device, dtype=dtype),
                             torch.tensor(list(range(10, 2001)), device=device, dtype=dtype))

        # Vectorization on non-contiguous tensors
        if dtype not in (torch.int8, torch.uint8):  # int8 and uint8 are too small for this test
            res = torch.rand(3, 3, 1000, device=device).to(dtype)
            res = res.permute(2, 0, 1)
            torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, out=res)
            self.assertEqual(res.flatten(), torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, device=device, dtype=dtype))

        self.assertRaises(RuntimeError, lambda: torch.linspace(0, 1, -1, device=device, dtype=dtype))
        # steps = 1
        self.assertEqual(torch.linspace(0, 1, 1, device=device, dtype=dtype),
                         torch.zeros(1, device=device, dtype=dtype), atol=0, rtol=0)
        # steps = 0
        self.assertEqual(torch.linspace(0, 1, 0, device=device, dtype=dtype).numel(), 0, atol=0, rtol=0)

        # steps not provided
        self.assertRaises(TypeError, lambda: torch.linspace(0, 1, device=device, dtype=dtype))

        if dtype == torch.float:
            # passed dtype can't be safely casted to inferred dtype
            with self.assertRaisesRegex(RuntimeError, r"torch.linspace\(\): inferred dtype"):
                torch.linspace(0, 1j, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"torch.linspace\(\): inferred dtype"):
                torch.linspace(0j, 1, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"torch.linspace\(\): inferred dtype"):
                torch.linspace(0j, 1j, 5, device=device, dtype=dtype)

        # Check linspace for generating the correct output for each dtype.
        start = 0 if dtype == torch.uint8 else -100
        expected_lin = torch.tensor([start + .5 * i for i in range(401)], device=device, dtype=torch.double)
        actual_lin = torch.linspace(start, start + 200, 401, device=device, dtype=dtype)
        # If on GPU, allow for minor error depending on dtype.
        tol = 0.
        if device != 'cpu':
            if dtype == torch.half:
                tol = 1e-1
            elif dtype == torch.float:
                tol = 1e-5
            elif dtype == torch.double:
                tol = 1e-10

        self.assertEqual(expected_lin.to(dtype), actual_lin, atol=tol, rtol=0)

        # Check linspace for generating with start > end.
        self.assertEqual(torch.linspace(2, 0, 3, device=device, dtype=dtype),
                         torch.tensor((2, 1, 0), device=device, dtype=dtype),
                         atol=0, rtol=0)

        # Check for race condition (correctness when applied on a large tensor).
        if dtype not in (torch.int8, torch.uint8, torch.int16, torch.half, torch.bfloat16):
            y = torch.linspace(0, 999999 + (999999j if dtype.is_complex else 0),
                               1000000, device=device, dtype=dtype)
            if dtype.is_complex:
                cond = torch.logical_and(y[:-1].real < y[1:].real, y[:-1].imag < y[1:].imag)
            else:
                cond = y[:-1] < y[1:]
            correct = all(cond)
            self.assertTrue(correct)

        # Check linspace for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        y = torch.linspace(0, 3, 4, out=x.narrow(1, 1, 2), dtype=dtype)
        self.assertEqual(x, torch.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=dtype), atol=0, rtol=0)

    def _test_linspace_logspace_deduction_helper(self, fn, device):
        # [FUNCTION DOC] HELPER: For several real and complex start and end pairs, check the
        # function (linspace or logspace) returns complex64 if either endpoint is complex,
        # otherwise float32.
        for start, end in [(1, 2), (1., 2), (1., -2.), (1j, 2j), (0., 2j), (1j, 2)]:
            dtype = torch.float32
            if isinstance(start, complex) or isinstance(end, complex):
                dtype = torch.cfloat

            self.assertEqual(fn(start, end, steps=100, device=device).dtype, dtype)

    def test_linspace_deduction(self, device):
        # [FUNCTION DOC] Check dtype deduction of linspace from its endpoints.
        # Test deduction from input parameters.
        self._test_linspace_logspace_deduction_helper(torch.linspace, device)

    def test_logspace_deduction(self, device):
        # [FUNCTION DOC] Check dtype deduction of logspace from its endpoints.
        # Test deduction from input parameters.
        self._test_linspace_logspace_deduction_helper(torch.logspace, device)

    # The implementation of linspace+logspace goes through a different path
    # when the steps arg is equal to 0 or 1. For other values of `steps`
    # they call specialized linspace (or logspace) kernels.
    LINSPACE_LOGSPACE_SPECIAL_STEPS = [0, 1]

    # NOTE [Linspace+Logspace precision override]
    # Our Linspace and logspace torch.half CUDA kernels are not very precise.
    # Since linspace/logspace are deterministic, we can compute an expected
    # amount of error (by testing without a precision override), adding a tiny
    # amount (EPS) to that, and using that value as the override.
    LINSPACE_LOGSPACE_EXTRA_EPS = 1e-5

    # Compares linspace device vs. cpu
    def _test_linspace(self, device, dtype, steps):
        # [FUNCTION DOC] HELPER: Compare linspace from zero to ten on the device versus default
        # cpu, with dtype not required to match.
        a = torch.linspace(0, 10, steps=steps, dtype=dtype, device=device)
        b = torch.linspace(0, 10, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.0039 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def test_linspace_device_vs_cpu(self, device, dtype):
        # [FUNCTION DOC] Compare device and cpu linspace for ten steps across float, complex,
        # half and bfloat16; skipped on cpu. Uses a precision override for half.
        self._test_linspace(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def test_linspace_special_steps(self, device, dtype):
        # [FUNCTION DOC] Compare device and cpu linspace for the special step counts zero and
        # one; skipped on cpu.
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_linspace(device, dtype, steps=steps)

    # Compares logspace device vs cpu
    def _test_logspace(self, device, dtype, steps):
        # [FUNCTION DOC] HELPER: Compare logspace with default base from 1 to 1.1 on the device
        # versus cpu, with dtype not required to match.
        a = torch.logspace(1, 1.1, steps=steps, dtype=dtype, device=device)
        b = torch.logspace(1, 1.1, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # Compares logspace device vs cpu
    def _test_logspace_base2(self, device, dtype, steps):
        # [FUNCTION DOC] HELPER: Compare logspace with base 2 from 1 to 1.1 on the device versus
        # cpu, with dtype not required to match.
        a = torch.logspace(1, 1.1, steps=steps, base=2, dtype=dtype, device=device)
        b = torch.logspace(1, 1.1, steps=steps, base=2)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.025 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_device_vs_cpu(self, device, dtype):
        # [FUNCTION DOC] Compare device and cpu logspace for ten steps (half also on cuda);
        # skipped on cpu.
        self._test_logspace(device, dtype, steps=10)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.0201 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_base2(self, device, dtype):
        # [FUNCTION DOC] Compare device and cpu logspace with base 2 for ten steps; skipped
        # on cpu.
        self._test_logspace_base2(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_special_steps(self, device, dtype):
        # [FUNCTION DOC] Compare device and cpu logspace, default and base 2, for the special
        # step counts zero and one; skipped on cpu.
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_logspace(device, dtype, steps=steps)
            self._test_logspace_base2(device, dtype, steps=steps)

    @dtypes(*all_types_and(torch.bfloat16))
    @dtypesIfCUDA(*integral_types_and(torch.half, torch.bfloat16, torch.float32, torch.float64) if TEST_WITH_ROCM else
                  all_types_and(torch.half, torch.bfloat16))
    def test_logspace(self, device, dtype):
        # [FUNCTION DOC] Broad logspace check across dtypes: matches out form, negative steps
        # error, missing steps TypeError, one step, inferred-dtype RuntimeError for float
        # with complex endpoints, exact values for base 2 (tolerance only off cpu), and
        # non-default bases.
        _from = random.random()
        to = _from + random.random()
        res1 = torch.logspace(_from, to, 137, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.logspace(_from, to, 137, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        self.assertRaises(RuntimeError, lambda: torch.logspace(0, 1, -1, device=device, dtype=dtype))
        # steps not provided
        self.assertRaises(TypeError, lambda: torch.logspace(0, 1, device=device, dtype=dtype))
        self.assertEqual(torch.logspace(0, 1, 1, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype), atol=0, rtol=0)

        if dtype == torch.float:
            # passed dtype can't be safely casted to inferred dtype
            with self.assertRaisesRegex(RuntimeError, r"torch.logspace\(\): inferred dtype"):
                torch.logspace(0, 1j, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"torch.logspace\(\): inferred dtype"):
                torch.logspace(0j, 1, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"torch.logspace\(\): inferred dtype"):
                torch.logspace(0j, 1j, 5, device=device, dtype=dtype)

        # Check precision - start, stop and base are chosen to avoid overflow
        # steps is chosen so that step size is not subject to rounding error
        # a tolerance is needed for gpu tests due to differences in computation
        atol = None
        rtol = None
        if self.device_type == 'cpu':
            atol = 0
            rtol = 0
        self.assertEqual(torch.tensor([2. ** (i / 8.) for i in range(49)], device=device, dtype=dtype),
                         torch.logspace(0, 6, steps=49, base=2, device=device, dtype=dtype),
                         atol=atol, rtol=rtol)

        # Check non-default base=2
        self.assertEqual(torch.logspace(1, 1, 1, 2, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype) * 2)
        self.assertEqual(torch.logspace(0, 2, 3, 2, device=device, dtype=dtype),
                         torch.tensor((1, 2, 4), device=device, dtype=dtype))

        # Check logspace_ for generating with start > end.
        self.assertEqual(torch.logspace(1, 0, 2, device=device, dtype=dtype),
                         torch.tensor((10, 1), device=device, dtype=dtype), atol=0, rtol=0)

        # Check logspace_ for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        y = torch.logspace(0, 3, 4, base=2, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        self.assertEqual(x, torch.tensor(((0, 1, 2), (0, 4, 8)), device=device, dtype=dtype), atol=0, rtol=0)

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.float, torch.double)
    def test_full_inference(self, device, dtype):
        # [FUNCTION DOC] With each of half, float and double set as the default dtype, check full
        # infers bool for a bool fill, int64 for an int fill, the default dtype for a float fill,
        # and the matching complex dtype for a complex fill.
        size = (2, 2)

        with set_default_dtype(dtype):
            # Tests bool fill value inference
            t = torch.full(size, True)
            self.assertEqual(t.dtype, torch.bool)

            # Tests integer fill value inference
            t = torch.full(size, 1)
            self.assertEqual(t.dtype, torch.long)

            # Tests float fill value inference
            t = torch.full(size, 1.)
            self.assertEqual(t.dtype, dtype)

            # Tests complex inference
            t = torch.full(size, (1 + 1j))
            ctype = torch.complex128 if dtype is torch.double else torch.complex64
            self.assertEqual(t.dtype, ctype)

    def test_full_out(self, device):
        # [FUNCTION DOC] Check full with out: a dtype that conflicts with out raises
        # RuntimeError, and without explicit dtype the result takes out's dtype instead of the
        # inferred one.
        size = (5,)
        o = torch.empty(size, device=device, dtype=torch.long)

        # verifies dtype/out conflict throws a RuntimeError
        with self.assertRaises(RuntimeError):
            torch.full(o.shape, 1., dtype=torch.float, out=o)

        # verifies out dtype overrides inference
        self.assertEqual(torch.full(o.shape, 1., out=o).dtype, o.dtype)
        self.assertEqual(torch.full(size, 1, out=o).dtype, o.dtype)

    # check that warning for numpy being not writable is suppressed
    # when a copy of it is being created.
    # see issue #47160
    def test_tensor_from_non_writable_numpy(self, device):
        # [FUNCTION DOC] Regression test: building a tensor from a non-writable numpy array emits
        # no warnings, because a copy is made.
        with warnings.catch_warnings(record=True) as w:
            a = np.arange(5.)
            a.flags.writeable = False
            t = torch.tensor(a)
            self.assertEqual(len(w), 0)

    @onlyCPU
    @parametrize('shared', [True, False])
    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
    def test_from_file(self, device, shared):
        # [FUNCTION DOC] CPU only, parametrized on shared: write a float64 array to a
        # temporary file, map it with from_file and check the storage filename is set only
        # when shared, the values match, and UntypedStorage from_file reports the same
        # filename. Skipped on Windows.
        dtype = torch.float64
        t = torch.randn(2, 5, dtype=dtype, device=device)
        with tempfile.NamedTemporaryFile() as f:
            expected_filename = f.name if shared else None
            t.numpy().tofile(f)
            t_mapped = torch.from_file(f.name, shared=shared, size=t.numel(), dtype=dtype)
            self.assertTrue(t_mapped.untyped_storage().filename == expected_filename)
            self.assertEqual(torch.flatten(t), t_mapped)

            s = torch.UntypedStorage.from_file(f.name, shared, nbytes=t.numel() * dtype.itemsize)
            self.assertTrue(s.filename == expected_filename)

    @onlyCPU
    def test_storage_filename(self, device):
        # [FUNCTION DOC] CPU only: check a regular tensor's storage has no filename.
        t = torch.randn(2, 5, device=device)
        self.assertIsNone(t.untyped_storage().filename)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_refs_tensor(self, device, dtype):
        # [FUNCTION DOC] Check the reference implementation of tensor given an empty list gives
        # the same result as torch.tensor, for many dtypes.
        self.assertEqual(torch._refs.tensor([], device=device, dtype=dtype), torch.tensor([], device=device, dtype=dtype))


# Class for testing random tensor creation ops, like torch.randint
class TestRandomTensorCreation(TestCase):
    # [CLASS DOC] Device-generic tests for random tensor creation ops: normal, uniform_, random_
    # with negative values, randint, randn, rand and randperm. exact_dtype is True.
    exact_dtype = True

    # TODO: add torch.complex64, torch.complex128
    @dtypes(torch.float, torch.double)
    def test_normal(self, device, dtype):
        # [FUNCTION DOC] Check normal sampling for float and double through a nested helper,
        # using real parts and imaginary parts for complex dtypes; the in-place, tensor mean,
        # tensor std, scalar forms, out forms, zero std, empty tensors and full-shape forms are
        # covered. For complex dtypes also check that a complex std raises RuntimeError.
        # Statistical checks use loose tolerances.

        def helper(self, device, dtype, ptype, t_transform, std_transform):
            # [FUNCTION DOC] HELPER: Run normal_ and torch.normal in all mean and std argument
            # combinations (tensor or scalar, with and without out, zero std) and assert dtype,
            # device, sample mean and sample std using the supplied transform for the tested
            # component and std scaling function.
            q = torch.empty(100, 100, dtype=dtype, device=device)

            q.normal_()
            self.assertEqual(t_transform(q).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(q).std(), std_transform(1), atol=0.2, rtol=0)

            q.normal_(2, 3)
            self.assertEqual(t_transform(q).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(q).std(), std_transform(3), atol=0.3, rtol=0)

            q = torch.empty(100, 100, dtype=dtype, device=device)
            q_row1 = q[0:1].clone()
            q[99:100].normal_()
            self.assertEqual(t_transform(q[99:100]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(q[99:100]).std(), std_transform(1), atol=0.2, rtol=0)
            self.assertEqual(t_transform(q[0:1]).clone(), t_transform(q_row1))

            mean = torch.empty(100, 100, dtype=dtype, device=device)
            mean[:50].fill_(ptype(0))
            mean[50:].fill_(ptype(1))

            std = torch.empty(100, 100, dtype=torch.float, device=device)
            std[:, :50] = 4
            std[:, 50:] = 1

            r = torch.normal(mean)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(1), atol=0.2, rtol=0)

            r.fill_(42)
            r = torch.normal(mean, 3)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)

            r.fill_(42)
            torch.normal(mean, 3, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)

            r.fill_(42)
            r = torch.normal(2, std)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, atol=0.2, rtol=0)
            self.assertEqual(r[:, :50].std(), 4, atol=0.3, rtol=0)
            self.assertEqual(r[:, 50:].std(), 1, atol=0.2, rtol=0)

            r.fill_(42)
            torch.normal(2, std, out=r)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, atol=0.2, rtol=0)
            self.assertEqual(r[:, :50].std(), 4, atol=0.3, rtol=0)
            self.assertEqual(r[:, 50:].std(), 1, atol=0.2, rtol=0)

            r.fill_(42)
            r = torch.normal(mean, std)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), atol=0.3, rtol=0)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), atol=0.2, rtol=0)

            r.fill_(42)
            torch.normal(mean, std, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), atol=0.3, rtol=0)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), atol=0.2, rtol=0)

            # test empty mean/std
            out = torch.normal(mean=torch.empty((0, 2)), std=torch.empty((0, 1)))
            self.assertEqual(out.size(), torch.Size([0, 2]))

            r.fill_(42)
            r = torch.normal(2, 3, (100, 100), dtype=dtype, device=device)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.3, rtol=0)

            r.fill_(42)
            torch.normal(2, 3, (100, 100), dtype=dtype, device=device, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.3, rtol=0)

            # float std 0 with float mean
            r.fill_(42)
            torch.normal(2, 0, (10, 10), dtype=dtype, device=device, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertTrue(r.eq(2).all())

            # float std 0 with tensor mean
            r.fill_(42)
            mean_rand = torch.randn(10, 10, dtype=dtype, device=device)
            torch.normal(mean_rand, 0, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(mean_rand, r, atol=0, rtol=0)

            # tensor std 0 with float mean
            r.fill_(42)
            std_zeros = torch.zeros(10, 10, dtype=dtype, device=device)
            torch.normal(2, std_zeros, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertTrue(r.eq(2).all())

            # tensor std 0 with tensor mean
            r.fill_(42)
            torch.normal(mean_rand, std_zeros, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(mean_rand, r, atol=0, rtol=0)

        if dtype.is_complex:
            helper(self, device, dtype, lambda x: complex(x, x),
                   lambda t: torch.real(t).to(torch.float), lambda mean: mean / math.sqrt(2))
            helper(self, device, dtype, lambda x: complex(x, x),
                   lambda t: torch.imag(t).to(torch.float), lambda mean: mean / math.sqrt(2))
            self.assertRaisesRegex(
                RuntimeError, "normal expects standard deviation to be non-complex",
                lambda: torch.normal(0, torch.empty(100, 100, dtype=dtype, device=device)))
            out = torch.empty(100, 100, dtype=dtype, device=device)
            self.assertRaisesRegex(
                RuntimeError, "normal expects standard deviation to be non-complex",
                lambda: torch.normal(0, torch.empty(100, 100, dtype=dtype, device=device), out=out))
        else:
            helper(self, device, dtype, lambda x: x, lambda t: t, lambda mean: mean)

    # Ensure that normal raises appropriate error when `std` < 0
    def test_normal_std_error(self, device):
        # [FUNCTION DOC] Check normal raises RuntimeError for a negative scalar std, and for a
        # negative tensor std unless on GPU where the error is raised asynchronously.
        a = torch.tensor(0, dtype=torch.float32, device=device)
        std = torch.tensor(-1, dtype=torch.float32, device=device)

        for input in [0, a]:
            with self.assertRaisesRegex(RuntimeError, r'normal expects std >= 0.0, but found std'):
                torch.normal(input, -1, (10,))

            if not std.is_cuda:  # on GPU this error is raised asynchronously
                with self.assertRaisesRegex(RuntimeError, r'normal expects all elements of std >= 0.0'):
                    torch.normal(input, std)

    # https://github.com/pytorch/pytorch/issues/126834
    @xfailIfTorchDynamo
    @dtypes(torch.float, torch.double, torch.half)
    @dtypesIfCUDA(torch.float, torch.double, torch.half, torch.bfloat16)
    def test_uniform_from_to(self, device, dtype):
        # [FUNCTION DOC] Check uniform_ over pairs of bounds from huge negative to huge positive
        # values: bounds outside the dtype range are skipped, reversed bounds and an overflowing
        # range raise RuntimeError, and otherwise min and max land within a tenth of the range at
        # each end (skipped for bfloat16, half on cpu, and all-nan results). Expected failure
        # under Dynamo.
        size = 2000
        alpha = 0.1

        float_min = torch.finfo(torch.float).min
        float_max = torch.finfo(torch.float).max
        double_min = torch.finfo(torch.double).min
        double_max = torch.finfo(torch.double).max

        if dtype == torch.bfloat16:
            min_val = -3.389531389251535e+38
            max_val = 3.389531389251535e+38
        else:
            min_val = torch.finfo(dtype).min
            max_val = torch.finfo(dtype).max

        values = [double_min, float_min, -42, 0, 42, float_max, double_max]

        for from_ in values:
            for to_ in values:
                t = torch.empty(size, dtype=dtype, device=device)
                if not (min_val <= from_ <= max_val) or not (min_val <= to_ <= max_val):
                    pass
                elif to_ < from_:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to return",
                        lambda: t.uniform_(from_, to_)
                    )
                elif to_ - from_ > max_val:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to-from",
                        lambda: t.uniform_(from_, to_)
                    )
                else:
                    t.uniform_(from_, to_)
                    range_ = to_ - from_
                    if dtype != torch.bfloat16 and not (
                            dtype == torch.half and device == 'cpu') and not torch.isnan(t).all():
                        delta = alpha * range_
                        double_t = t.to(torch.double)
                        if range_ == 0:
                            self.assertTrue(double_t.min() == from_)
                            self.assertTrue(double_t.max() == to_)
                        elif dtype == torch.half:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() <= to_)
                        else:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() < to_)

    def test_random_neg_values(self, device):
        # [FUNCTION DOC] Check random_ with negative bounds on signed dtypes keeps values between
        # the lower bound and the upper bound minus one.
        SIZE = 10
        signed_dtypes = [torch.double, torch.float, torch.long, torch.int, torch.short]
        for dtype in signed_dtypes:
            res = torch.rand(SIZE, SIZE).to(device=device, dtype=dtype)
            res.random_(-10, -1)
            self.assertLessEqual(res.max().item(), 9)
            self.assertGreaterEqual(res.min().item(), -10)

    # TODO: this test should be updated
    @onlyCPU
    def test_randint_inference(self, device):
        # [FUNCTION DOC] CPU only: check randint output dtype is int64 by default and with layout
        # or generator, follows dtype or out when given; check errors for empty or reversed
        # ranges, float bound (TypeError), and bounds out of range for int16 and uint32.
        size = (2, 1)
        for args in [(3,), (1, 3)]:  # (low,) and (low, high)
            self.assertIs(torch.int64, torch.randint(*args, size=size).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, layout=torch.strided).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, generator=torch.default_generator).dtype)
            self.assertIs(torch.float32, torch.randint(*args, size=size, dtype=torch.float32).dtype)
            out = torch.empty(size, dtype=torch.float32)
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out).dtype)
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out, dtype=torch.float32).dtype)
            out = torch.empty(size, dtype=torch.int64)
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out, dtype=torch.int64).dtype)

        self.assertRaisesRegex(RuntimeError,
                               "random_ expects 'from' to be less than 'to', but got from=0 >= to=0",
                               lambda: torch.randint(0, size=size))
        self.assertRaisesRegex(RuntimeError,
                               "random_ expects 'from' to be less than 'to', but got from=-1 >= to=-2",
                               lambda: torch.randint(-1, -2, size=size))
        self.assertRaisesRegex(TypeError,
                               r"randint\(\): argument 'high' \(position 1\) must be int, not float",
                               lambda: torch.randint(.5, size=size))
        self.assertRaisesRegex(RuntimeError,
                               "from is out of bounds for",
                               lambda: torch.randint(-32769, 0, size=size, dtype=torch.int16))
        self.assertRaisesRegex(RuntimeError,
                               "from is out of bounds for",
                               lambda: torch.randint(-1, 1, size=size, dtype=torch.uint32))

    # TODO: this test should be updated
    @onlyCPU
    def test_randint(self, device):
        # [FUNCTION DOC] CPU only: with the default generator and a new generator, check randint
        # results with the same seed are identical for the low/high and high-only forms and for
        # out forms, and that values lie within zero and five.
        SIZE = 100

        def seed(generator):
            # [FUNCTION DOC] HELPER: Seed the global generator when none is given, otherwise seed
            # the given generator, with a fixed seed; returns the generator.
            if generator is None:
                torch.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, torch.Generator()):
            generator = seed(generator)
            res1 = torch.randint(0, 6, (SIZE, SIZE), generator=generator)
            res2 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            torch.randint(0, 6, (SIZE, SIZE), generator=generator, out=res2)
            generator = seed(generator)
            res3 = torch.randint(6, (SIZE, SIZE), generator=generator)
            res4 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            torch.randint(6, (SIZE, SIZE), out=res4, generator=generator)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res3)
            self.assertEqual(res1, res4)
            self.assertEqual(res2, res3)
            self.assertEqual(res2, res4)
            self.assertEqual(res3, res4)
            self.assertTrue((res1 < 6).all().item())
            self.assertTrue((res1 >= 0).all().item())


    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "For fb compatibility random not changed in fbcode")
    def test_randint_distribution(self, device):
        # [FUNCTION DOC] Check that randint over a range of three quarters of 2 to the 32, with a
        # million samples, is evenly spread over eight bins within a statistical error bound.
        # Skipped in fbcode.
        size = 1_000_000
        n_max = int(0.75 * 2 ** 32)
        n_bins = 8

        def bin(index, max_size):
            # [FUNCTION DOC] HELPER: Map an index to its bin number by integer division of the
            # range into the outer number of bins.
            return index // (max_size // n_bins)
        res = torch.randint(n_max, (size,), device=device)
        # histogram implemented for float only
        bins = bin(res, n_max).float().cpu()
        hist, _ = bins.histogram(8, range=(0, n_bins))
        expected_bin = res.shape[0] / 8
        expected_error = math.sqrt(expected_bin) / expected_bin * 3
        error = (hist - expected_bin).abs().max() / expected_bin
        self.assertTrue(error < expected_error)


    @dtypes(torch.half, torch.float, torch.bfloat16, torch.double,
            torch.complex32, torch.complex64, torch.complex128)
    def test_randn(self, device, dtype):
        # [FUNCTION DOC] Check randn with the same seed gives the same values through the out
        # form, for sizes 0 and 100 and half, bfloat16, float, double and complex dtypes.
        SIZE = 100
        for size in [0, SIZE]:
            torch.manual_seed(123456)
            res1 = torch.randn(size, size, dtype=dtype, device=device)
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            torch.randn(size, size, out=res2)
            self.assertEqual(res1, res2)

    @dtypes(torch.float, torch.double, torch.complex32, torch.complex64, torch.complex128)
    def test_rand(self, device, dtype):
        # [FUNCTION DOC] Check rand with the same seed gives the same values through the out
        # form, for sizes 0 and 100 and float, double and complex dtypes.
        SIZE = 100
        for size in [0, SIZE]:
            torch.manual_seed(123456)
            res1 = torch.rand(size, size, dtype=dtype, device=device)
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            torch.rand(size, size, out=res2)
            self.assertEqual(res1, res2)

    def test_randperm(self, device):
        # [FUNCTION DOC] Broad randperm check: for several n and dtypes values equal the out form
        # under a forked RNG and form a permutation, default dtype is long, n of zero gives an
        # empty tensor, RuntimeError when n is too large for the dtype, and out being
        # non-contiguous works.
        if device == 'cpu' or device == 'meta':
            rng_device = None
        else:
            # TODO: This won't actually work for non-CUDA device
            # see https://github.com/pytorch/pytorch/issues/54282
            rng_device = [device]

        # Test core functionality. On CUDA, different value of n has different
        # code path
        for n in (5, 100, 50000, 100000):
            # Ensure both integer and floating-point numbers are tested. Half follows an execution path that is
            # different from others on CUDA.
            for dtype in (torch.long, torch.half, torch.float, torch.bfloat16):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                if dtype == torch.bfloat16 and device != 'cpu':
                    continue
                if n > 256 and dtype == torch.bfloat16:
                    continue
                with torch.random.fork_rng(devices=rng_device):
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1, res2, atol=0, rtol=0)
                self.assertEqual(res1.sort().values.long(), torch.arange(n, device=device))

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test exceptions when n is too large for a floating point type
        for dtype, small_n, large_n in ((torch.uint8, 2**8, 2**8 + 1),
                                        (torch.half, 2**11 + 1, 2**11 + 2),
                                        (torch.float, 2**24 + 1, 2**24 + 2),
                                        (torch.double, 2**25,  # 2**53 + 1 is too large to run
                                         2**53 + 2)):
            res = torch.empty(0, dtype=dtype, device=device)
            torch.randperm(small_n, out=res)  # No exception expected
            self.assertRaises(RuntimeError, lambda: torch.randperm(large_n, out=res, device=device))

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):
                res = torch.randperm(n, dtype=torch.long, device=device)
            torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(non_contiguous_tensor, res)
            self.assertEqual(res.sort().values.long(), torch.arange(n, device=device))


    @largeTensorTest("10GB", "cpu")
    @largeTensorTest("40GB", "cuda")
    @slowTest
    def test_randperm_large(self, device):
        # [FUNCTION DOC] Large-memory slow test: randperm of 700 million items is checked to be
        # evenly distributed by deciles for the first million entries (to detect skew from 32-bit
        # random numbers).
        # Test even distribution where rand32 might produce skewed "uniform" distribution
        # n_items is chosen to not evenly divide 2**32 and be sufficiently large
        # to easily detect skew
        def decile(index, collection_size):
            # [FUNCTION DOC] HELPER: Map an index to its decile by integer division of the
            # collection size into ten parts.
            return index // (collection_size // 10)

        n_items = 700_000_000
        shuffled = torch.randperm(n_items, device=device)
        interval = 1_000_000
        shuffled_interval = shuffled[:interval]
        # histogram implemented for float only
        deciles = decile(shuffled_interval, shuffled.shape[0]).float().cpu()
        hist, _ = deciles.histogram(10, range=(0, 10))
        expected_bin = shuffled_interval.shape[0] / 10
        expected_error = math.sqrt(expected_bin) / expected_bin * 3
        error = (hist - expected_bin).abs().max() / expected_bin
        self.assertTrue(error < expected_error, lambda msg: f"{msg}\nerror {error} > {expected_error}")

    # Test exceptions when device and generator types are incompatible
    @onlyCUDA
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Produces inconsistent errors when run in fbcode.")
    def test_randperm_device_compatibility(self, device):
        # [FUNCTION DOC] CUDA only: check randperm and randint accept generators on matching
        # device types (including a different cuda index) and for n of zero any generator; a cpu
        # generator for a cuda output or a cuda generator for a cpu output raises RuntimeError
        # about device type, also with out. Skipped in fbcode.
        cuda_gen = torch.Generator(device='cuda')
        cpu_gen = torch.Generator(device='cpu')

        # n=0 is a special case that we don't need to use generator, thus no error even if
        # device and generator don't match
        torch.randperm(0, device='cuda:0', generator=torch.Generator(device='cuda:1'))
        if torch.cuda.device_count() > 1:
            torch.randperm(0, device='cuda:1', generator=torch.Generator(device='cuda:0'))
        torch.randperm(0, device='cuda', generator=torch.Generator(device='cpu'))
        torch.randperm(0, device='cpu', generator=torch.Generator(device='cuda'))

        for n in (1, 3, 100, 30000):
            torch.randperm(n, device='cuda', generator=torch.Generator(device='cuda:0'))
            torch.randperm(n, device='cuda:0', generator=torch.Generator(device='cuda'))
            # For cuda:0 to match cuda:1, we are making consistent device type matching
            # behavior just like torch.randint. Longer term, generator should ignore
            # device ordinal, since it's not used anyway.
            torch.randint(low=0, high=n + 1, size=(1,), device="cuda:0", generator=torch.Generator(device='cuda:1'))
            torch.randperm(n, device='cuda:0', generator=torch.Generator(device='cuda:1'))
            if torch.cuda.device_count() > 1:
                torch.randint(low=0, high=n + 1, size=(1,), device="cuda:1", generator=torch.Generator(device='cuda:0'))
                torch.randperm(n, device='cuda:1', generator=torch.Generator(device='cuda:0'))

            regex = 'Expected a .* device type for generator but found .*'
            cuda_t = torch.tensor(n, device='cuda')
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cuda', generator=cpu_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cuda', generator=cpu_gen, out=cuda_t))
            cpu_t = torch.tensor(n, device='cpu')
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cpu', generator=cuda_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cpu', generator=cuda_gen, out=cpu_t))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, generator=cuda_gen))  # implicitly on CPU

# Class for testing *like ops, like torch.ones_like
class TestLikeTensorCreation(TestCase):
    # [CLASS DOC] Device-generic tests for the like creation ops (ones_like, zeros_like,
    # empty_like, full_like, rand_like, randn_like, randint_like). exact_dtype is True.
    exact_dtype = True

    # TODO: this test should be updated
    def test_ones_like(self, device):
        # [FUNCTION DOC] Check ones_like of a float tensor and of a bool tensor equals the
        # expected ones.
        expected = torch.ones(100, 100, device=device)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

        # test boolean tensor
        expected = torch.tensor([True, True], device=device, dtype=torch.bool)
        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

    # TODO: this test should be updated
    @onlyCPU
    def test_empty_like(self, device):
        # [FUNCTION DOC] CPU only: check empty_like keeps shape and type string for an empty
        # tensor, a float matrix and an int tensor (wrapped in the legacy Variable wrapper).
        x = torch.autograd.Variable(torch.tensor([]))
        y = torch.autograd.Variable(torch.randn(4, 4))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        for a in (x, y, z):
            self.assertEqual(torch.empty_like(a).shape, a.shape)
            self.assertEqualTypeString(torch.empty_like(a), a)

    def test_zeros_like(self, device):
        # [FUNCTION DOC] Check zeros_like of a zeros tensor equals the expected zeros.
        expected = torch.zeros((100, 100,), device=device)

        res1 = torch.zeros_like(expected)
        self.assertEqual(res1, expected)

    @deviceCountAtLeast(2)
    def test_zeros_like_multiple_device(self, devices):
        # [FUNCTION DOC] With at least two devices, check zeros_like of a tensor on the second
        # device is equal to zeros created on the first device.
        expected = torch.zeros(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.zeros_like(x)
        self.assertEqual(output, expected)

    @deviceCountAtLeast(2)
    def test_ones_like_multiple_device(self, devices):
        # [FUNCTION DOC] With at least two devices, check ones_like of a tensor on the second
        # device is equal to ones created on the first device.
        expected = torch.ones(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.ones_like(x)
        self.assertEqual(output, expected)

    # Full-like precedence is the explicit dtype then the dtype of the "like"
    # tensor.
    @onlyNativeDeviceTypes
    def test_full_like_inference(self, device):
        # [FUNCTION DOC] Check full_like uses the dtype of the like tensor (long) even for a
        # float fill value, unless an explicit dtype (complex64) is given.
        size = (2, 2)
        like = torch.empty((5,), device=device, dtype=torch.long)

        self.assertEqual(torch.full_like(like, 1.).dtype, torch.long)
        self.assertEqual(torch.full_like(like, 1., dtype=torch.complex64).dtype,
                         torch.complex64)

    def test_rand_like(self, device):
        # [FUNCTION DOC] Check rand_like with the global and a device generator: same seed gives
        # equal results, values lie in zero to one, shapes match; separate generators with equal
        # seeds agree and a different seed or an advanced generator gives different values.
        like_tensor = torch.zeros(100, 100, device=device)

        def seed(generator):
            # [FUNCTION DOC] HELPER: Seed the global generator when none is given, otherwise seed
            # the given generator, with a fixed seed; returns the generator.
            if generator is None:
                torch.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, torch.Generator(device)):
            generator = seed(generator)
            res1 = torch.rand_like(like_tensor, generator=generator)

            generator = seed(generator)
            res2 = torch.empty_like(like_tensor)
            res2 = torch.rand_like(like_tensor, generator=generator)

            self.assertEqual(res1, res2)
            self.assertTrue((res1 >= 0).all().item())
            self.assertTrue((res1 < 1).all().item())
            self.assertEqual(res1.shape, like_tensor.shape)

        gen0 = torch.Generator(device)
        gen1 = torch.Generator(device)
        gen2 = torch.Generator(device)
        gen0.manual_seed(42)
        gen1.manual_seed(42)
        gen2.manual_seed(123456)

        tensor0 = torch.rand_like(like_tensor, generator=gen0)
        tensor1 = torch.rand_like(like_tensor, generator=gen1)
        tensor2 = torch.rand_like(like_tensor, generator=gen2)
        self.assertEqual(tensor0, tensor1)
        self.assertNotEqual(tensor2, tensor0)
        self.assertNotEqual(tensor2, tensor1)

        tensor0 = torch.rand_like(like_tensor, generator=gen0)
        self.assertNotEqual(tensor0, tensor1)

    def test_randn_like(self, device):
        # [FUNCTION DOC] Check randn_like with the global and a device generator: same seed gives
        # equal results with matching shapes; generators with equal seeds agree and differently
        # seeded or advanced generators differ.
        like_tensor = torch.zeros(100, 100, device=device)

        def seed(generator):
            # [FUNCTION DOC] HELPER: Seed the global generator when none is given, otherwise seed
            # the given generator, with a fixed seed; returns the generator.
            if generator is None:
                torch.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, torch.Generator(device)):
            generator = seed(generator)
            res1 = torch.randn_like(like_tensor, generator=generator)

            generator = seed(generator)
            res2 = torch.empty_like(like_tensor)
            res2 = torch.randn_like(like_tensor, generator=generator)

            self.assertEqual(res1, res2)
            self.assertEqual(res1.shape, like_tensor.shape)

        gen0 = torch.Generator(device)
        gen1 = torch.Generator(device)
        gen2 = torch.Generator(device)
        gen0.manual_seed(42)
        gen1.manual_seed(42)
        gen2.manual_seed(123456)

        tensor0 = torch.randn_like(like_tensor, generator=gen0)
        tensor1 = torch.randn_like(like_tensor, generator=gen1)
        tensor2 = torch.randn_like(like_tensor, generator=gen2)
        self.assertEqual(tensor0, tensor1)
        self.assertNotEqual(tensor2, tensor0)
        self.assertNotEqual(tensor2, tensor1)

        tensor0 = torch.randn_like(like_tensor, generator=gen0)
        self.assertNotEqual(tensor0, tensor1)


    def test_randint_like(self, device):
        # [FUNCTION DOC] Check randint_like (low and high form, and high only form) with the
        # global and a device generator: same seed gives equal results in the range zero to
        # ten, shapes match; equal seeded generators agree, differently seeded or advanced
        # generators differ.
        like_tensor = torch.zeros(100, 100, device=device, dtype=torch.long)

        def seed(generator):
            # [FUNCTION DOC] HELPER: Seed the global generator when none is given, otherwise seed
            # the given generator, with a fixed seed; returns the generator.
            if generator is None:
                torch.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, torch.Generator(device)):
            generator = seed(generator)
            res1 = torch.randint_like(like_tensor, 0, 10, generator=generator)

            generator = seed(generator)
            res2 = torch.empty_like(like_tensor)
            res2 = torch.randint_like(like_tensor, 0, 10, generator=generator)

            generator = seed(generator)
            res3 = torch.randint_like(like_tensor, 10, generator=generator)

            generator = seed(generator)
            res4 = torch.empty_like(like_tensor)
            res4 = torch.randint_like(like_tensor, 10, generator=generator)

            self.assertEqual(res1, res2)
            self.assertEqual(res3, res4)
            self.assertTrue((res1 >= 0).all().item())
            self.assertTrue((res1 < 10).all().item())
            self.assertTrue((res3 >= 0).all().item())
            self.assertTrue((res3 < 10).all().item())
            self.assertEqual(res1.shape, like_tensor.shape)
            self.assertEqual(res3.shape, like_tensor.shape)

        gen0 = torch.Generator(device)
        gen1 = torch.Generator(device)
        gen2 = torch.Generator(device)
        gen0.manual_seed(42)
        gen1.manual_seed(42)
        gen2.manual_seed(123456)

        tensor0 = torch.randint_like(like_tensor, 0, 10, generator=gen0)
        tensor1 = torch.randint_like(like_tensor, 0, 10, generator=gen1)
        tensor2 = torch.randint_like(like_tensor, 0, 10, generator=gen2)
        self.assertEqual(tensor0, tensor1)
        self.assertNotEqual(tensor2, tensor0)
        self.assertNotEqual(tensor2, tensor1)

        tensor0 = torch.randint_like(like_tensor, 0, 10, generator=gen0)
        self.assertNotEqual(tensor0, tensor1)


# Tests for the `frombuffer` function (only work on CPU):
#   Constructs tensors from Python objects that implement the buffer protocol,
#   without copying data.
SIZE = 5
SHAPE = (SIZE,)

def may_require_grad(dtype):
    # [FUNCTION DOC] HELPER: Return True if the dtype is floating point or complex, meaning a
    # tensor of that dtype may require gradients.
    return dtype.is_floating_point or dtype.is_complex

def get_dtype_size(dtype):
    # [FUNCTION DOC] HELPER: Return the size in bytes of one element of the dtype, as an integer.
    return int(torch.empty((), dtype=dtype).element_size())

class TestBufferProtocol(TestCase):
    # [CLASS DOC] Tests for frombuffer, which builds a tensor sharing memory with a Python buffer
    # object. Each test compares against numpy frombuffer using the module-level SIZE and SHAPE.
    def _run_test(self, shape, dtype, count=-1, first=0, offset=None, **kwargs):
        # [FUNCTION DOC] HELPER: Create a random numpy array of the shape and dtype, wrap it in a
        # memoryview, and call both frombuffer functions with the count and offset (offset
        # defaults to the first element index times the element size; extra keywords go to the
        # torch version only). Asserts equal values and the same data address; returns the numpy
        # array and the torch tensor.
        numpy_dtype = torch_to_numpy_dtype_dict[dtype]

        if offset is None:
            offset = first * get_dtype_size(dtype)

        numpy_original = make_tensor(shape, dtype=dtype, device="cpu").numpy()
        original = memoryview(numpy_original)
        # First call PyTorch's version in case of errors.
        # If this call exits successfully, the NumPy version must also do so.
        torch_frombuffer = torch.frombuffer(original, dtype=dtype, count=count, offset=offset, **kwargs)
        numpy_frombuffer = np.frombuffer(original, dtype=numpy_dtype, count=count, offset=offset)

        self.assertEqual(numpy_frombuffer, torch_frombuffer)
        self.assertEqual(numpy_frombuffer.__array_interface__["data"][0], torch_frombuffer.data_ptr())
        return (numpy_original, torch_frombuffer)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_same_type(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer matches numpy for scalar, 1-D and 2-D arrays of
        # each dtype.
        self._run_test((), dtype)
        self._run_test((4,), dtype)
        self._run_test((10, 10), dtype)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_requires_grad(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer honors the requires_grad keyword for arrays of each
        # dtype: true is only valid for floating or complex dtypes, and false works for all.
        def _run_test_and_check_grad(requires_grad, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Run the buffer test helper with requires_grad added to the
            # keywords and assert the resulting tensor has that requires_grad value.
            kwargs["requires_grad"] = requires_grad
            _, tensor = self._run_test(*args, **kwargs)
            self.assertTrue(tensor.requires_grad == requires_grad)

        requires_grad = may_require_grad(dtype)
        _run_test_and_check_grad(requires_grad, (), dtype)
        _run_test_and_check_grad(requires_grad, (4,), dtype)
        _run_test_and_check_grad(requires_grad, (10, 10), dtype)
        _run_test_and_check_grad(False, (), dtype)
        _run_test_and_check_grad(False, (4,), dtype)
        _run_test_and_check_grad(False, (10, 10), dtype)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_with_offset(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer with every valid element offset in a short array.
        # Offset should be valid whenever there is, at least,
        # one remaining element
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, first=i)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_with_count(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer with every valid count from minus one to the array
        # length, except zero.
        # Count should be valid for any valid in the interval
        # [-1, len(input)], except for 0
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_with_count_and_offset(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer with explicit counts, explicit offsets, and every valid
        # pair of count and offset.
        # Explicit default count [-1, 1, 2, ..., len]
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)
        # Explicit default offset [0, 1, ..., len - 1]
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, first=i)
        # All possible combinations of count and dtype aligned
        # offset for 'input'
        # count:[1, 2, ..., len - 1] x first:[0, 1, ..., len - count]
        for i in range(1, SIZE):
            for j in range(SIZE - i + 1):
                self._run_test(SHAPE, dtype, count=i, first=j)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_invalid_positional_args(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer raises ValueError with specific messages for an empty
        # buffer, count zero, a negative offset, an offset at the buffer end, a non-multiple
        # offset, and a count that runs past the buffer end.
        bytes = get_dtype_size(dtype)
        in_bytes = SIZE * bytes
        # Empty array
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length \(0\) and count"):
            empty = np.array([])
            torch.frombuffer(empty, dtype=dtype)
        # Count equals 0
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length .* and count \(0\)"):
            self._run_test(SHAPE, dtype, count=0)
        # Offset negative and bigger than total length
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \(-{bytes} bytes\) must be"):
            self._run_test(SHAPE, dtype, first=-1)
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \({in_bytes} bytes\) must be .* "
                                    rf"buffer length \({in_bytes} bytes\)"):
            self._run_test(SHAPE, dtype, first=SIZE)
        # Non-multiple offset with all elements
        if bytes > 1:
            offset = bytes - 1
            with self.assertRaisesRegex(ValueError,
                                        rf"buffer length \({in_bytes - offset} bytes\) after "
                                        rf"offset \({offset} bytes\) must be"):
                self._run_test(SHAPE, dtype, offset=bytes - 1)
        # Count too big for each good first element
        for first in range(SIZE):
            count = SIZE - first + 1
            with self.assertRaisesRegex(ValueError,
                                        rf"requested buffer length \({count} \* {bytes} bytes\) "
                                        rf"after offset \({first * bytes} bytes\) must .*"
                                        rf"buffer length \({in_bytes} bytes\)"):
                self._run_test(SHAPE, dtype, count=count, first=first)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_shared_buffer(self, device, dtype):
        # [FUNCTION DOC] Check the tensor from frombuffer shares memory with the array: writing
        # through the tensor changes the array (full range and for many counts and offsets), and
        # changing an array element is visible in the tensor.
        x = make_tensor((1,), dtype=dtype, device=device)
        # Modify the whole tensor
        arr, tensor = self._run_test(SHAPE, dtype)
        tensor[:] = x
        self.assertEqual(arr, tensor)
        self.assertTrue((tensor == x).all().item())

        # Modify the whole tensor from all valid offsets, given
        # a count value
        for count in range(-1, SIZE + 1):
            if count == 0:
                continue

            actual_count = count if count > 0 else SIZE
            for first in range(SIZE - actual_count):
                last = first + actual_count
                arr, tensor = self._run_test(SHAPE, dtype, first=first, count=count)
                tensor[:] = x
                self.assertEqual(arr[first:last], tensor)
                self.assertTrue((tensor == x).all().item())

                # Modify the first value in the array
                arr[first] = x.item() - 1
                self.assertEqual(arr[first:last], tensor)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_not_a_buffer(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer on a plain list raises ValueError saying the object
        # does not implement the buffer protocol.
        with self.assertRaisesRegex(ValueError,
                                    r"object does not implement Python buffer protocol."):
            torch.frombuffer([1, 2, 3, 4], dtype=dtype)

    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_non_writable_buffer(self, device, dtype):
        # [FUNCTION DOC] Check frombuffer on a read-only bytes object emits a UserWarning (once)
        # saying the buffer is not writable.
        numpy_arr = make_tensor((1,), dtype=dtype, device=device).numpy()
        byte_arr = numpy_arr.tobytes()
        with self.assertWarnsOnceRegex(UserWarning,
                                       r"The given buffer is not writable."):
            torch.frombuffer(byte_arr, dtype=dtype)

    def test_byte_to_int(self):
        # [FUNCTION DOC] Check frombuffer reinterprets eight bytes as two int32 values of 255
        # each, using byte order matching the platform.
        byte_array = np.array([-1, 0, 0, 0, -1, 0, 0, 0], dtype=np.byte) if sys.byteorder == 'little' \
            else np.array([0, 0, 0, -1, 0, 0, 0, -1], dtype=np.byte)
        tensor = torch.frombuffer(byte_array, dtype=torch.int32)
        self.assertEqual(tensor.numel(), 2)
        self.assertSequenceEqual(tensor, [255, 255])

class TestFromBlob(TestCase):
    # [CLASS DOC] Tests for the private from_blob wrapper, which builds a tensor over a raw
    # memory address supplied as an integer, using numpy arrays as the memory owner.
    def _make_data(self, dtype, numel):
        # [FUNCTION DOC] HELPER: Build a numpy array holding one through numel in the dtype's
        # numpy equivalent; return the array and the address of its data.
        numpy_dtype = torch_to_numpy_dtype_dict[dtype]
        arr = np.arange(1, numel + 1, dtype=numpy_dtype)
        return arr, arr.ctypes.data

    @dtypes(*all_types_and_complex_and(torch.half))
    def test_basic(self, device, dtype):
        # [FUNCTION DOC] Check from_blob over six elements gives a tensor equal to the
        # numpy array.
        arr, ptr = self._make_data(dtype, 6)
        t = torch._from_blob(ptr, [6], dtype=dtype)
        self.assertEqual(t, torch.from_numpy(arr))

    @dtypes(*all_types_and_complex_and(torch.half))
    def test_2d(self, device, dtype):
        # [FUNCTION DOC] Check from_blob with a 2 by 3 shape equals the reshaped numpy array.
        arr, ptr = self._make_data(dtype, 6)
        t = torch._from_blob(ptr, [2, 3], dtype=dtype)
        self.assertEqual(t, torch.from_numpy(arr.reshape(2, 3)))

    @dtypes(torch.float32, torch.float64, torch.int32, torch.int64)
    def test_strides(self, device, dtype):
        # [FUNCTION DOC] Check from_blob with explicit strides one and two on a 2 by 3 shape
        # equals the numpy array reshaped in Fortran order.
        arr, ptr = self._make_data(dtype, 6)
        t = torch._from_blob(ptr, [2, 3], [1, 2], dtype=dtype)
        expected = torch.from_numpy(arr.reshape(2, 3, order="F"))
        self.assertEqual(t, expected)

    @dtypes(torch.float32, torch.int64)
    def test_shared_memory(self, device, dtype):
        # [FUNCTION DOC] Check memory is shared in both directions between a numpy array and a
        # from_blob tensor.
        numpy_dtype = torch_to_numpy_dtype_dict[dtype]
        arr = np.array([1, 2, 3, 4], dtype=numpy_dtype)
        t = torch._from_blob(arr.ctypes.data, [4], dtype=dtype)
        arr[0] = 99
        self.assertEqual(t[0].item(), 99)
        t[1] = 77
        self.assertEqual(arr[1], 77)

    def test_explicit_device(self, device):
        # [FUNCTION DOC] Check from_blob with an explicit cpu device gives a cpu tensor equal to
        # the array.
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = torch._from_blob(arr.ctypes.data, [2], dtype=torch.float32, device="cpu")
        self.assertEqual(t.device, torch.device("cpu"))
        self.assertEqual(t, torch.from_numpy(arr))

    def test_default_dtype(self, device):
        # [FUNCTION DOC] Check from_blob without dtype takes the current default dtype, for
        # float32 and float64 defaults.
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with set_default_dtype(torch.float32):
            t = torch._from_blob(arr.ctypes.data, [2])
            self.assertEqual(t.dtype, torch.float32)
        with set_default_dtype(torch.float64):
            t = torch._from_blob(arr.ctypes.data, [2])
            self.assertEqual(t.dtype, torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_invalid_device(self, device):
        # [FUNCTION DOC] Only when CUDA is available: check from_blob over host memory with a
        # cuda device raises RuntimeError.
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            torch._from_blob(arr.ctypes.data, [2], dtype=torch.float32, device="cuda")


# Tests for the `asarray` function:
#   Constructs tensors from a Python object that has one of the following
#   characteristics:
#       1. is a Tensor
#       2. is a DLPack capsule
#       3. implements the Python Buffer protocol
#       4. is an arbitrary list
#   The implementation itself is based on the Python Array API:
#   https://data-apis.org/array-api/latest/API_specification/creation_functions.html
def get_another_device(device):
    # [FUNCTION DOC] HELPER: Return a device type name different from the given one: cpu
    # for any non-cpu device, otherwise the current accelerator type, or None when there is
    # no accelerator.
    device = torch.device(device)

    if device.type != "cpu":
        return "cpu"

    acc = torch.accelerator.current_accelerator(True)

    return acc.type if acc is not None else None

def identity(tensor):
    # [FUNCTION DOC] HELPER: Return the tensor unchanged (an identity converter for
    # asarray tests).
    return tensor
def to_numpy(tensor):
    # [FUNCTION DOC] HELPER: Convert a tensor to a numpy array sharing its memory.
    return tensor.numpy()
def to_memview(tensor):
    # [FUNCTION DOC] HELPER: Convert a tensor to a memoryview over its numpy array.
    return memoryview(to_numpy(tensor))

class TestAsArray(TestCase):
    # [CLASS DOC] Tests for torch.asarray, which builds a tensor from a tensor, numpy array,
    # DLPack capsule, buffer object or list, aliasing the memory where possible. Helper methods
    # verify aliasing, dtype, device and requires_grad of the result.
    def _check(self, original, cvt=lambda t: t, is_alias=True, same_dtype=True, same_device=True, **kwargs):
        # [FUNCTION DOC] HELPER: Convert the original tensor with cvt, call asarray with the
        # extra keyword arguments, and assert: the result is a tensor; it shares memory with the
        # original only when is_alias is set; values are equal when same_dtype (else only shape
        # and requested dtype); the device type and index match the original device when
        # same_device (else the requested device, default cpu); and requires_grad equals the
        # keyword or the converted tensor's flag.
        """Check the output of 'asarray', given its input and assertion information.

        Besides calling 'asarray' itself, this function does 4 different checks:
            1. Whether the result is aliased or not, depending on 'is_alias'
            2. Whether the result has the expected dtype and elements
            3. Whether the result lives in the expected device
            4. Whether the result has its 'requires_grad' set or not
        """
        converted_original = cvt(original)
        result = torch.asarray(converted_original, **kwargs)
        self.assertTrue(isinstance(result, torch.Tensor))

        # 1. The storage pointers should be equal only if 'is_alias' is set
        if is_alias:
            self.assertEqual(result.data_ptr(), original.data_ptr())
        else:
            self.assertNotEqual(result.data_ptr(), original.data_ptr())

        # 2. Comparison of the elements only takes place if the original
        # sequence and the resulting tensor have the same data type
        if same_dtype:
            self.assertEqual(original, result)
        else:
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            self.assertEqual(original.shape, result.shape)
            self.assertEqual(dtype, result.dtype)

        # 3. Given the specified target device, we first check whether
        # its type is the same, and then if its index is the same (if it
        # is not None)
        if same_device:
            device = original.device
        else:
            device = torch.device(kwargs.get("device", "cpu"))

        # Compare the target device type, and its index
        self.assertEqual(device.type, result.device.type)
        if device.index is not None:
            self.assertEqual(device.index, result.device.index)

        # 4. By default, 'requires_grad' mirrors the original tensor's requires_grad, if
        # present.
        original_requires_grad = converted_original.requires_grad if isinstance(converted_original, torch.Tensor) else False
        self.assertEqual(result.requires_grad, kwargs.get("requires_grad", original_requires_grad))

    def _test_alias_with_cvt(self, cvt, device, dtype, shape=(5, 5), only_with_dtype=False):
        # [FUNCTION DOC] HELPER: Make a random tensor, then check asarray results are aliases
        # through the converter for several keyword combinations (copy false, device, dtype,
        # requires_grad). When only_with_dtype is set, the combinations that omit dtype are
        # skipped (needed for buffers).
        original = make_tensor(shape, dtype=dtype, device=device)

        def check(**kwargs):
            # [FUNCTION DOC] HELPER: Run the alias check on the outer original tensor and
            # converter with the given asarray keyword arguments.
            self._check(original, cvt=cvt, **kwargs)

        if not only_with_dtype:
            check(copy=False)
            check(device=device)
            check(device=device, copy=False)

        check(dtype=dtype)
        check(dtype=dtype, copy=False)
        check(requires_grad=False, dtype=dtype)
        check(requires_grad=may_require_grad(dtype), dtype=dtype)
        check(device=device, dtype=dtype)
        check(device=device, dtype=dtype, copy=False)

    # Skipping 'meta' devices, since there's no point in comparing their
    # data pointer (which is basically the point here), since they all
    # return 0.
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_alias_from_tensor(self, device, dtype):
        # [FUNCTION DOC] Check asarray aliases a tensor input across dtypes. Skipped for meta
        # tensors since their data pointers are all zero.
        self._test_alias_with_cvt(identity, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_alias_from_numpy(self, device, dtype):
        # [FUNCTION DOC] CPU only: check asarray aliases a numpy array input for all
        # numpy dtypes.
        self._test_alias_with_cvt(to_numpy, device, dtype)

    # Skipping 'meta', since 'to_dlpack' does not work for them.
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_alias_from_dlpack(self, device, dtype):
        # [FUNCTION DOC] Check asarray aliases a DLPack capsule input across dtypes. Skipped
        # for meta.
        self._test_alias_with_cvt(to_dlpack, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_alias_from_buffer(self, device, dtype):
        # [FUNCTION DOC] CPU only: check asarray aliases a memoryview input for numpy dtypes,
        # always passing dtype.
        self._test_alias_with_cvt(to_memview, device, dtype, shape=(5,), only_with_dtype=True)

    def _test_copy_with_cvt(self, cvt, device, dtype, shape=(5, 5), only_with_dtype=False):
        # [FUNCTION DOC] HELPER: Make a random tensor and check that asarray with copy forced (or
        # forced by a differing device or dtype) never shares memory. Covers requires_grad and
        # dtype keywords, a different device (skipping the test if the other accelerator lacks
        # the dtype), and every other dtype (skipped when only_with_dtype is set).
        original = make_tensor(shape, dtype=dtype, device=device)

        def check(**kwargs):
            # [FUNCTION DOC] HELPER: Run the shared check on the outer original tensor and
            # converter, expecting no aliasing, with the given asarray keyword arguments.
            self._check(original, cvt=cvt, is_alias=False, **kwargs)

        if not only_with_dtype:
            check(copy=True)
            check(device=device, copy=True)

        check(requires_grad=False, dtype=dtype, copy=True)
        check(requires_grad=may_require_grad(dtype), dtype=dtype, copy=True)
        check(dtype=dtype, copy=True)
        check(device=device, dtype=dtype, copy=True)

        # Copy is forced because of different device
        other = get_another_device(device)
        if other is not None:
            try:
                caps = torch.accelerator.get_device_capability(other)
            except Exception:
                pass
            else:
                if dtype not in caps["supported_dtypes"]:
                    self.skipTest(f"{other} doesn't support {dtype}")

            check(same_device=False, device=other, dtype=dtype)
            check(same_device=False, device=other, dtype=dtype, copy=True)

        # Copy is forced because of different dtype
        if not only_with_dtype:
            for other in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
                if dtype != other:
                    check(same_dtype=False, dtype=other)
                    check(same_dtype=False, dtype=other, copy=True)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_copy_tensor(self, device, dtype):
        # [FUNCTION DOC] Check asarray copies a tensor input in the copy-forced scenarios.
        # Skipped for meta.
        self._test_copy_with_cvt(identity, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_copy_from_numpy(self, device, dtype):
        # [FUNCTION DOC] CPU only: check asarray copies a numpy array input in the
        # copy-forced scenarios.
        self._test_copy_with_cvt(to_numpy, device, dtype)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_copy_from_dlpack(self, device, dtype):
        # [FUNCTION DOC] Check asarray copies a DLPack capsule input in the copy-forced
        # scenarios. Skipped for meta.
        self._test_copy_with_cvt(to_dlpack, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_copy_from_buffer(self, device, dtype):
        # [FUNCTION DOC] CPU only: check asarray copies a memoryview input in the copy-forced
        # scenarios, always passing dtype.
        self._test_copy_with_cvt(to_memview, device, dtype, shape=(5,), only_with_dtype=True)

    def _test_copy_mult_devices(self, devices, dtype, cvt):
        # [FUNCTION DOC] HELPER: Make a tensor on the first device and check asarray to the
        # second device (through the converter) copies, with no keyword, with copy true, and with
        # dtype given.
        cuda1 = devices[0]
        cuda2 = devices[1]
        original = make_tensor((5, 5), dtype=dtype, device=cuda1)

        def check(**kwargs):
            # [FUNCTION DOC] HELPER: Run the shared check on the outer tensor expecting a copy
            # onto the second device, with the given asarray keyword arguments.
            self._check(original, cvt, is_alias=False, same_device=False, device=cuda2, **kwargs)

        check()
        check(copy=True)
        check(dtype=dtype, copy=True)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_copy_from_tensor_mult_devices(self, devices, dtype):
        # [FUNCTION DOC] CUDA, two devices: check asarray of a tensor to the other cuda
        # device copies.
        self._test_copy_mult_devices(devices, dtype, identity)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_copy_from_dlpack_mult_devices(self, devices, dtype):
        # [FUNCTION DOC] CUDA, two devices: check asarray of a DLPack capsule to the other cuda
        # device copies.
        self._test_copy_mult_devices(devices, dtype, to_dlpack)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_copy_list(self, device, dtype):
        # [FUNCTION DOC] Check asarray of a nested list built from a cpu tensor copies, producing
        # the requested dtype and device with default, explicit requires_grad and copy settings.
        original = make_tensor((5, 5), dtype=dtype, device=torch.device("cpu"))

        def check(**kwargs):
            # [FUNCTION DOC] HELPER: Run the shared check with the tolist converter on the outer
            # original tensor, expecting no aliasing, with the given asarray keyword arguments.
            self._check(original, torch.Tensor.tolist, is_alias=False, **kwargs)

        same_device = torch.device("cpu") == device
        check(same_device=same_device, device=device, dtype=dtype)
        check(same_device=same_device, device=device, dtype=dtype, requires_grad=False)
        check(same_device=same_device, device=device, dtype=dtype, requires_grad=may_require_grad(dtype))
        check(same_device=same_device, device=device, dtype=dtype, copy=True)

    @dtypes(torch.float32)
    def test_unsupported_alias(self, device, dtype):
        # [FUNCTION DOC] Check asarray with copy disabled raises ValueError when it would need a
        # device change, a dtype change, or when the input is an arbitrary list.
        original = make_tensor((5, 5), dtype=dtype, device=device)

        other_device = get_another_device(device)
        if other_device is not None:
            with self.assertRaisesRegex(ValueError,
                                        f"from device '{device}' to '{other_device}'"):
                torch.asarray(original, device=other_device, copy=False)

        with self.assertRaisesRegex(ValueError,
                                    "with dtype '.*' into dtype '.*'"):
            torch.asarray(original, dtype=torch.float64, copy=False)

        with self.assertRaisesRegex(ValueError,
                                    "can't alias arbitrary sequence"):
            torch.asarray(original.tolist(), copy=False)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(torch.float32)
    def test_unsupported_alias_mult_devices(self, devices, dtype):
        # [FUNCTION DOC] CUDA, two devices: check asarray with copy disabled across cuda devices
        # raises ValueError naming both devices.
        dev1, dev2 = devices[:2]
        original = make_tensor((5, 5), dtype=dtype, device=dev1)
        with self.assertRaisesRegex(ValueError,
                                    f"from device '{dev1}' to '{dev2}'"):
            torch.asarray(original, device=dev2, copy=False)

    @dtypes(torch.float32, torch.complex64)
    def test_retain_autograd_history(self, device, dtype):
        # [FUNCTION DOC] Check asarray of a non-leaf tensor keeps requires_grad from the input by
        # default, honors an explicit flag, and only retains autograd history when the result
        # requires grad.
        original = make_tensor((5, 5), dtype=dtype, device=device, requires_grad=True)
        # 'cloned' has 'grad_fn=<CloneBackwards>'
        cloned = original.clone()

        def check(**kwargs):
            # [FUNCTION DOC] HELPER: Call asarray on the outer non-leaf tensor with the keywords
            # and assert requires_grad equals the keyword or the input's flag, and the grad_fn is
            # absent exactly when requires_grad is false.
            a = torch.asarray(cloned, **kwargs)
            requires_grad = kwargs.get("requires_grad", cloned.requires_grad)
            self.assertEqual(a.requires_grad, requires_grad)
            # Autograd history shouldn't be retained when requires_grad is False
            self.assertEqual(a.grad_fn is None, not requires_grad)

        check()
        check(requires_grad=True)
        check(copy=True)
        check(requires_grad=True, copy=True)
        check(requires_grad=False)
        check(requires_grad=False, copy=True)

    @onlyCPU
    def test_astensor_consistency(self, device):
        # [FUNCTION DOC] CPU only: check asarray gives results equal to as_tensor for Python
        # scalars, homogeneous and mixed lists (bool, int, float, complex) and a range.
        # See issue: https://github.com/pytorch/pytorch/pull/71757

        examples = [
            # Scalars
            True,
            42,
            1.0,
            # Homogeneous Lists
            [True, True, False],
            [1, 2, 3, 42],
            [0.0, 1.0, 2.0, 3.0],
            # Mixed Lists
            [True, False, 0],
            [0.0, True, False],
            [0, 1.0, 42],
            [0.0, True, False, 42],
            # With Complex
            [0.0, True, False, 42, 5j],
            # With Range
            range(5),
        ]

        for e in examples:
            original = torch.as_tensor(e)
            t = torch.asarray(e)
            self.assertEqual(t, original)

    # Dynamo changes numpy scalar to array, thus skips the asserted error.
    @xfailIfTorchDynamo
    @onlyCPU
    def test_numpy_scalars(self, device):
        # [FUNCTION DOC] CPU only, expected to fail under Dynamo: check asarray with copy
        # disabled on a numpy scalar raises RuntimeError; with default copy it gives a 0-D
        # float64 tensor; a 0-D numpy array with dtype int32 gives a 0-D int32 tensor (regression
        # test for a reported issue).
        scalar = np.float64(0.5)

        with self.assertRaisesRegex(RuntimeError, "can't alias NumPy scalars."):
            torch.asarray(scalar, copy=False)

        tensor = torch.asarray(scalar)
        self.assertEqual(tensor.dim(), 0)
        self.assertEqual(tensor.item(), scalar.item())
        self.assertEqual(tensor.dtype, torch.float64)
        # Regression test for https://github.com/pytorch/pytorch/issues/97021
        zerodim_arr = np.array(1.)
        tensor = torch.asarray(zerodim_arr, dtype=torch.int32)
        self.assertEqual(tensor.dim(), 0)
        self.assertEqual(tensor.item(), zerodim_arr.item())
        self.assertEqual(tensor.dtype, torch.int32)

    def test_default_device(self, device):
        # [FUNCTION DOC] Check asarray under a default-device context places the result on that
        # device, for an integer, a tensor, a numpy array and a memoryview, and that contents
        # are preserved.
        original = torch.arange(5)

        examples: list[tuple[Any, dict]] = [
            (3, {}),
            (original, {}),
            (to_numpy(original), {}),
            (to_memview(original), {"dtype": original.dtype}),
        ]

        for data, kwargs in examples:
            with torch.device(device):
                tensor = torch.asarray(data, **kwargs)
                self.assertEqual(tensor.device, torch.device(device))

                # Check the contents of the tensor.
                if isinstance(data, int):
                    self.assertEqual(data, tensor.item())
                else:
                    self.assertEqual(data, tensor)

    @onlyAccelerator
    def test_device_without_index(self, device):
        # [FUNCTION DOC] Accelerator only: check asarray with a device without index aliases the
        # input unless copy is requested, which gives a different data address.
        original = torch.arange(5, device=device)

        tensor = torch.asarray(original, device=device)
        # The storage pointers should be equal
        self.assertEqual(original.data_ptr(), tensor.data_ptr())

        tensor = torch.asarray(original, copy=True, device=device)
        # The storage pointers should not be equal
        self.assertNotEqual(original.data_ptr(), tensor.data_ptr())


instantiate_device_type_tests(TestTensorCreation, globals())
instantiate_device_type_tests(TestRandomTensorCreation, globals())
instantiate_device_type_tests(TestLikeTensorCreation, globals())
instantiate_device_type_tests(TestBufferProtocol, globals(), only_for="cpu")
instantiate_device_type_tests(TestFromBlob, globals(), only_for="cpu")
instantiate_device_type_tests(TestAsArray, globals())

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
