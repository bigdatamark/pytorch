# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Owner(s): ["module: tests"]
# [MODULE DOC] Main unit-test module for core torch functionality. Defines TestTorchDeviceType
# (device-generic tests instantiated per device type), TestDevicePrecision, TestTorch
# (CPU-oriented API tests), a Tracker helper and disable_gc context manager, a generator of
# negative-dimension tests, and two empty classes that receive generated view/device-op tests.
# Import time: asserts that the default dtype is float32, computes AMPERE_OR_ROCM and is_cuda_sm86
# flags, and picks the TestCase base class (the inductor one when testing under inductor). Run as
# a script it runs the whole suite via run_tests.

import torch
import torch.utils.data
import numpy as np

import contextlib
import gc
import io
import inspect
import itertools
import math
import random
import re
import copy
import tempfile
import unittest
import warnings
import types
import pickle
import textwrap
import subprocess
import weakref
import sys
import copyreg
from torch import inf, nan
from itertools import product, combinations, permutations, chain
from functools import partial
from torch import multiprocessing as mp
from torch.testing import make_tensor
from torch.testing._internal.common_optimizers import (
    optim_db, optims, _get_optim_inputs_including_global_cliquey_kwargs)

from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    MI200_ARCH, TEST_WITH_TORCHINDUCTOR, TEST_WITH_ROCM, run_tests, IS_JETSON,
    IS_FILESYSTEM_UTF8_ENCODING,
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, skipIfRocmArch, skipIfTorchInductor, load_tests, slowTest, slowTestIf,
    skipIfCrossRef, TEST_WITH_CROSSREF, skipIfTorchDynamo, set_default_dtype,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfRocmVersionAtLeast, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, CudaSyncGuard,
    bytes_to_scalar, parametrize, noncontiguous_like,
    AlwaysWarnTypedStorageRemoval, TEST_WITH_TORCHDYNAMO, xfailIfTorchDynamo,
    xfailIfS390X, set_warn_always_context, decorateIf, isRocmArchAnyOf,
    IS_MACOS,
)
from multiprocessing.reduction import ForkingPickler
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    expectedFailureXLA,
    instantiate_device_type_tests,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta, PYTORCH_CUDA_MEMCHECK, largeTensorTest, onlyNativeDeviceTypes, skipCUDAIfNotRocm,
    get_all_device_types, skipXLA)
import torch.backends.quantized
import torch.testing._internal.data
from torch.testing._internal.common_cuda import (
    tf32_on_and_off, TEST_CUDNN, TEST_MULTIGPU,
    _create_scaling_case, _create_scaling_models_optimizers)
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.testing._internal.common_dtype import (
    floating_types_and, get_all_math_dtypes, all_types_and_complex_and, complex_types,
    all_types_and, floating_types, floating_and_complex_types, integral_types_and,
    get_all_qint_dtypes, all_types_complex_float8_and, all_passthru_types_and,
)
from torch.testing._internal.two_tensor import TwoTensor
from torch.testing._internal.common_utils import IS_WINDOWS

if TEST_WITH_TORCHINDUCTOR:
    from torch._inductor.test_case import TestCase
else:
    from torch.testing._internal.common_utils import TestCase  # type: ignore[assignment]


# Protects against includes accidentally setting the default dtype
if torch.get_default_dtype() is not torch.float32:
    raise AssertionError(f"default dtype should be float32, got {torch.get_default_dtype()}")

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

AMPERE_OR_ROCM = TEST_WITH_ROCM or torch.cuda.is_tf32_supported()


is_cuda_sm86 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 6)

class TestTorchDeviceType(TestCase):
    # [CLASS DOC] Device-generic test case for tensors, storages,
    # deterministic/nondeterministic-alert behavior, random sampling statistics, cdist and many
    # other ops. Sets exact_dtype so comparisons check dtypes. Test methods take a device (and
    # often a dtype) supplied by instantiate_device_type_tests, which creates per-device
    # subclasses. Underscore-prefixed or check_* methods are shared helpers.
    exact_dtype = True

    # TODO: move all tensor creation to common ops
    def _rand_shape(self, dim, min_size, max_size):
        # [FUNCTION DOC] HELPER: Return a tuple of dim random integers, each drawn uniformly
        # (inclusive) from min_size to max_size, for use as a random tensor shape.
        shape = []
        for _ in range(dim):
            shape.append(random.randint(min_size, max_size))
        return tuple(shape)

    # Validates that mathematical constants are defined properly, as required by
    # the Python Array API (https://data-apis.org/array-api/latest/API_specification/constants.html)
    @onlyCPU
    def test_constants(self, device):
        # [FUNCTION DOC] CPU only. Verifies that torch.e, torch.pi, torch.nan and torch.inf
        # are plain Python floats and equal the math module constants exactly (NaN compared
        # with equal_nan).
        self.assertIsInstance(torch.e, float)
        self.assertEqual(torch.e, math.e, atol=0, rtol=0)

        self.assertIsInstance(torch.pi, float)
        self.assertEqual(torch.pi, math.pi, atol=0, rtol=0)

        self.assertIsInstance(torch.nan, float)
        self.assertEqual(torch.nan, math.nan, equal_nan=True)

        self.assertIsInstance(torch.inf, float)
        self.assertEqual(torch.inf, math.inf)

    @onlyNativeDeviceTypes
    @slowTestIf(IS_WINDOWS)
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    def test_bytes_to_scalar(self, device, dtype):
        # [FUNCTION DOC] For each of 10 iterations, builds a random list of element_size bytes for
        # the dtype, converts it with bytes_to_scalar, and checks the resulting scalar's untyped
        # storage bytes equal the original list.
        def rand_byte():
            # [FUNCTION DOC] HELPER: Return one random byte value as a Python number, using a
            # boolean-range value (0 or 1) when the dtype under test is bool and 0..255 otherwise.
            if dtype == torch.bool:
                return torch.randint(0, 2, ()).item()
            else:
                return torch.randint(0, 256, ()).item()

        element_size = torch._utils._element_size(dtype)

        for _ in range(10):
            bytes_list = [rand_byte() for _ in range(element_size)]
            scalar = bytes_to_scalar(bytes_list, dtype, device)
            self.assertEqual(scalar.storage().untyped().tolist(), bytes_list)

    # For testing in64 support in upsample_nearest3d
    @skipIfRocmArch(MI200_ARCH)
    @onlyCUDA
    @largeTensorTest('56GB', device='cuda')
    @dtypes(torch.bfloat16)
    @unittest.skipIf(IS_JETSON, "Large tensor tests are too large for Jetson.")
    @decorateIf(unittest.expectedFailure, lambda params: isRocmArchAnyOf(MI200_ARCH))
    def test_int64_upsample3d(self, device, dtype):
        # [FUNCTION DOC] CUDA-only large-tensor test (about 56GB, skipped on Jetson and ROCm
        # MI200). Runs nearest-neighbor interpolation with scale factor 2 on a very large bfloat16
        # tensor to exercise int64 indexing in the upsample_nearest3d kernel, and fails the test
        # if any exception is raised.
        x = torch.ones((1, 256, 16, 720, 1280), dtype=dtype, device=device)
        try:
            torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    @slowTestIf(IS_WINDOWS)
    def test_storage(self, device, dtype):
        # [FUNCTION DOC] Creates a random small tensor and checks that the legacy storage indexing
        # returns the same elements as the tensor, both for the typed storage (element by element)
        # and for the untyped byte storage (decoded per element with bytes_to_scalar).
        v = make_tensor((3, 5), dtype=dtype, device=device, low=-9, high=9)
        self.assertEqual(v.storage()[0], v[0][0])
        self.assertEqual(v.storage()[14], v[2][4])
        v_s = v.storage()

        for el_num in range(v.numel()):
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            self.assertEqual(
                v_s[el_num],
                v[dim0][dim1])

        v_s_byte = v.storage().untyped()
        el_size = v.element_size()

        for el_num in range(v.numel()):
            start = el_num * el_size
            end = start + el_size
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            self.assertEqual(
                bytes_to_scalar(v_s_byte[start:end], dtype, device),
                v[dim0][dim1])

    @onlyNativeDeviceTypes
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.quint8, torch.qint8, torch.qint32,
            torch.quint4x2)
    @slowTestIf(IS_WINDOWS)
    def test_storage_setitem(self, device, dtype):
        # [FUNCTION DOC] Exercises item and slice assignment on legacy typed storage classes
        # looked up by dtype name (torch or torch.cuda namespace). Fill with zeros, assign per
        # index, then assign a slice, comparing against a storage built from a mirrored Python
        # list each time. Quantized dtypes are skipped on CUDA by returning early.
        # Skip quantized dtypes for CUDA, since they're not supported
        if torch.device(device).type == 'cuda':
            if dtype in [torch.quint8, torch.qint8, torch.qint32, torch.quint4x2]:
                return

        storage_type_name = torch.storage._dtype_to_storage_type_map()[dtype]
        if torch.device(device).type == 'cuda':
            storage_type = eval('torch.cuda.' + storage_type_name)
        else:
            storage_type = eval('torch.' + storage_type_name)

        N = 10

        s = storage_type(N)
        s[:] = 0
        l = [0] * N
        self.assertEqual(s, storage_type(l))

        for i in range(N):
            s[i] = i
            l[i] = i

        self.assertEqual(s, storage_type(l))

        l[2:7] = [1] * 5
        s[2:7] = 1
        self.assertEqual(s, storage_type(l))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @onlyNativeDeviceTypes
    @slowTestIf(IS_WINDOWS)
    def test_storage_use_count(self, device):
        # [FUNCTION DOC] Checks reference counts on storages: a fresh tensor's untyped storage has
        # use count 2 (tensor plus the temporary wrapper), and creating a view increases the count
        # by one. Skipped under TorchDynamo.
        a = torch.randn(10, device=device)
        prev_cf = torch._C._storage_Use_Count(a.untyped_storage()._cdata)
        # Two references: 'a' and the wrapper returned by untyped_storage()
        self.assertEqual(prev_cf, 2)
        b = a.view(2, 5)
        self.assertEqual(torch._C._storage_Use_Count(b.untyped_storage()._cdata), prev_cf + 1)

    @onlyNativeDeviceTypes
    def test_storage_throws_on_data_ptr_access(self, device):
        # [FUNCTION DOC] Checks the private helper that reports whether data_ptr access would
        # raise. A normal storage reports False; a FakeTensorMode storage with unsafe data_ptr
        # access disabled reports True; a storage armed through the C-level error-message setter
        # reports True and goes back to False once the message is cleared. Each case is
        # cross-checked against the local helper.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.storage import _throws_on_data_ptr_access

        def raises_on_data_ptr(storage):
            # [FUNCTION DOC] HELPER: Return True if calling data_ptr on the given storage raises
            # RuntimeError, otherwise False.
            try:
                storage.data_ptr()
            except RuntimeError:
                return True
            return False

        # Normal storage: data_ptr() works, so we report False.
        s = torch.randn(4, device=device).untyped_storage()
        self.assertFalse(_throws_on_data_ptr_access(s))
        self.assertFalse(raises_on_data_ptr(s))

        # FakeTensor with unsafe access disallowed: data_ptr() raises.
        with FakeTensorMode() as mode:
            mode._allow_unsafe_data_ptr_access = False
            fs = torch.randn(4, device=device).untyped_storage()
            self.assertTrue(_throws_on_data_ptr_access(fs))
            self.assertTrue(raises_on_data_ptr(fs))

        # Storage armed to throw on the immutable data_ptr path.
        s2 = torch.randn(4, device=device).untyped_storage()
        torch._C._set_storage_data_ptr_access_error_msg(s2._cdata, "invalid")
        self.assertTrue(_throws_on_data_ptr_access(s2))
        self.assertTrue(raises_on_data_ptr(s2))
        torch._C._clear_storage_data_ptr_access_error_msg(s2._cdata)
        self.assertFalse(_throws_on_data_ptr_access(s2))
        self.assertFalse(raises_on_data_ptr(s2))

    @xfailIfTorchDynamo
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_tensor_storage_type(self, device, dtype):
        # [FUNCTION DOC] Checks that a tensor's storage_type method returns the legacy storage
        # class (looked up in the torch or torch.cuda namespace by dtype name) matching its dtype.
        a = make_tensor((10,), dtype=dtype, device=device, low=-9, high=9)

        module = torch.cuda if (torch.device(device).type == 'cuda') else torch
        expected_storage_type = getattr(module, torch.storage._dtype_to_storage_type_map()[dtype])

        self.assertEqual(a.storage_type(), expected_storage_type)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    @slowTestIf(IS_WINDOWS)
    def test_tensor_from_storage(self, device, dtype):
        # [FUNCTION DOC] Builds a tensor from a typed storage and from the untyped storage of an
        # existing tensor and checks it equals the original. Also checks that constructing a
        # tensor of a different dtype from a mismatched storage raises RuntimeError mentioning an
        # expected storage type.
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        a_s = a.storage()
        b = torch.tensor(a_s, device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, b)
        c = torch.tensor(a_s.untyped(), device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, c)

        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if error_dtype == dtype:
                continue
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                error_storage = a.to(error_dtype).storage()
                torch.tensor(error_storage, device=device, dtype=dtype)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_set_storage(self, device, dtype):
        # [FUNCTION DOC] Same as the tensor-from-storage test but using the in-place set_ method
        # on an empty tensor, with typed and untyped storage; a storage of the wrong dtype must
        # raise RuntimeError.
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        a_s = a.storage()
        b = torch.tensor([], device=device, dtype=dtype).set_(a_s).reshape(a.size())
        self.assertEqual(a, b)
        c = torch.tensor([], device=device, dtype=dtype).set_(a_s.untyped()).reshape(a.size())
        self.assertEqual(a, c)

        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if error_dtype == dtype:
                continue
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                error_storage = a.to(error_dtype).storage()
                b = torch.tensor([], device=device, dtype=dtype).set_(error_storage)

    def _check_storage_meta(self, s, s_check):
        # [FUNCTION DOC] HELPER: Recursively validate a meta-device storage s against a real
        # storage s_check of the same kind: both must be typed or both untyped, s must be on the
        # meta device with the same nbytes and size, data_ptr zero, element access raising
        # NotImplementedError, and for typed storages the same dtype and an equivalent check of
        # the untyped storage.
        self.assertTrue(
            isinstance(s, (torch.UntypedStorage, torch.TypedStorage)) and
            isinstance(s_check, type(s)),
            lambda msg: (
                f"{msg}\ns and s_check must both be one of UntypedStorage or "
                "TypedStorage, but got"
                f" {type(s).__name__} and {type(s_check).__name__}"
            ),
        )

        self.assertEqual(s.device.type, 'meta')
        self.assertEqual(s.nbytes(), s_check.nbytes())
        self.assertEqual(s.size(), s_check.size())
        self.assertEqual(s.data_ptr(), 0)

        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s[0]

        if isinstance(s, torch.TypedStorage):
            self.assertEqual(s.dtype, s_check.dtype)
            self._check_storage_meta(s.untyped(), s_check.untyped())

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_typed_storage_meta(self, device, dtype):
        # [FUNCTION DOC] Creates typed storages from several constructor argument sets (empty,
        # size 0, size 100, list of values) on the real device and on meta, and validates each
        # meta storage against its real counterpart.
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        for args in args_list:
            s_check = torch.TypedStorage(*args, dtype=dtype, device=device)
            s = torch.TypedStorage(*args, dtype=dtype, device='meta')
            self._check_storage_meta(s, s_check)

    @onlyNativeDeviceTypes
    @slowTestIf(IS_WINDOWS)
    def test_untyped_storage_meta(self, device):
        # [FUNCTION DOC] Same as the typed meta test but for untyped storages built from several
        # constructor argument sets on the device and on meta.
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        for args in args_list:
            s_check = torch.UntypedStorage(*args, device=device)
            s = torch.UntypedStorage(*args, device='meta')
            self._check_storage_meta(s, s_check)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_storage_meta_from_tensor(self, device, dtype):
        # [FUNCTION DOC] Converts a random tensor to the meta device and checks its storage
        # matches the storage of the original tensor via the meta storage check helper.
        t_check = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        t = t_check.to('meta')

        s_check = t_check.storage()
        s = t.storage()
        self._check_storage_meta(s, s_check)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_storage_meta_errors(self, device, dtype):
        # [FUNCTION DOC] Checks that unsupported operations on a meta typed storage raise: cpu and
        # cuda copies raise NotImplementedError, share_fd/share_filename/share_cuda/share_memory
        # raise RuntimeError, pin_memory raises TypeError, tolist and writing to a file raise
        # NotImplementedError, is_shared is False, and copying a meta storage into a real CPU or
        # CUDA storage raises NotImplementedError.
        s0 = torch.TypedStorage([1, 2, 3, 4], device='meta', dtype=dtype)

        with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
            s0.cpu()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_fd_cpu_()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_filename_cpu_()

        if torch.cuda.is_available():
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s0.cuda()

            with self.assertRaisesRegex(RuntimeError, r'only available on CUDA'):
                s0._share_cuda_()

            with self.assertRaisesRegex(TypeError, r"cannot pin 'torch.storage.UntypedStorage' only CPU memory can be pinned"):
                s0.pin_memory()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0.share_memory_()

        self.assertFalse(s0.is_shared())

        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s0.tolist()

        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s0._write_file(f, True, True, s0.element_size())

        for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
            s1 = torch.TypedStorage([1, 2, 3, 4], device=device, dtype=dtype)

            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s1.copy_(s0)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_storage_meta_ok(self, device, dtype):
        # [FUNCTION DOC] CPU only. Checks that resizing a meta typed storage to 10 elements works
        # without allocation and without raising.
        s0 = torch.TypedStorage([1, 2, 3, 4], device='meta', dtype=dtype)

        # This is OK, it changes the meta storage size without allocating
        s0.resize_(10)

    @onlyCUDA
    def test_module_share_memory(self):
        # [FUNCTION DOC] CUDA only. Regression test for issue 80733: moving a small Linear module
        # to CUDA and calling share_memory on it must not raise.
        # Test fix for issue #80733
        # See https://github.com/pytorch/pytorch/issues/80733
        model = torch.nn.Linear(3, 1)
        _model_cuda = model.to('cuda')
        model.share_memory()

    @dtypes(torch.float32, torch.complex64)
    @slowTestIf(IS_WINDOWS)
    def test_deepcopy(self, device, dtype):
        # [FUNCTION DOC] Deep-copies a nested list containing tensors, views and their storages;
        # checks values are equal, that sharing between a tensor and its storage is preserved
        # (in-place edits of the copy show up through the copied storage and view), and that extra
        # Python attributes on a tensor survive the copy.
        from copy import deepcopy
        a = torch.randn(5, 5, dtype=dtype, device=device)
        b = torch.randn(5, 5, dtype=dtype, device=device)
        c = a.view(25)
        q = [a, [a.storage(), b.storage()], b, c]
        w = deepcopy(q)
        self.assertEqual(w[0], q[0], atol=0, rtol=0)
        self.assertEqual(w[1][0], q[1][0], atol=0, rtol=0)
        self.assertEqual(w[1][1], q[1][1], atol=0, rtol=0)
        self.assertEqual(w[1], q[1], atol=0, rtol=0)
        self.assertEqual(w[2], q[2], atol=0, rtol=0)

        # Check that deepcopy preserves sharing
        w[0].add_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][0][i], q[1][0][i] + 1)
        self.assertEqual(w[3], c + 1)
        w[2].sub_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][1][i], q[1][1][i] - 1)

        # Check that deepcopy preserves attributes
        a.foo = 3
        self.assertEqual(deepcopy(a).foo, 3)

    @dtypes(torch.float32, torch.complex64)
    @slowTestIf(IS_WINDOWS)
    def test_deepcopy_scalar(self, device, dtype):
        # [FUNCTION DOC] Deep-copies a zero-dimensional tensor and checks the size and value
        # are unchanged.
        from copy import deepcopy
        a = torch.tensor(5, dtype=dtype, device=device)
        self.assertEqual(a.size(), deepcopy(a).size())
        self.assertEqual(a, deepcopy(a))

    def check_internal_mem_overlap(self, inplace_op, num_inputs,
                                   dtype, device,
                                   expected_failure=False):
        # [FUNCTION DOC] HELPER: Shared helper: checks that an in-place op (given as a callable or
        # a method name looked up on Tensor) raises RuntimeError about a single memory location
        # when applied to an expanded input. With expected_failure True it instead asserts that
        # the expected error is not raised (AssertionError).
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input)
                            for i in range(num_inputs - 1)]
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                inplace_op(*inputs)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                    inplace_op(*inputs)

    def unary_check_input_output_mem_overlap(self, data, sz, op,
                                             expected_failure=False):
        # [FUNCTION DOC] HELPER: Shared helper for out= ops: runs a function with output identical
        # to input, with independent output, and checks results match a reference out tensor; then
        # checks that partially overlapping and transposed output/input raise RuntimeError about
        # an unsupported operation. With expected_failure True those checks assert that the error
        # is not raised.

        def _test(op, output, input):
            # [FUNCTION DOC] HELPER: Compute the expected result by calling the op into a fresh
            # empty_like output, then call it with the given output and assert the returned value
            # equals the expected one.
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output is identical to input:
        _test(op, output=data[0:sz], input=data[0:sz])
        # output and input are independent:
        _test(op, output=data[0:sz], input=data[sz:2 * sz])
        # output partially overlaps with input:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, data[0:sz], data[1:sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, data[0:sz], data[1:sz + 1])
        # output is transpose of input:
        length = int(math.sqrt(sz))
        input = data[:length**2].view([length, length])
        out = input.t()
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, out, input)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, out, input)

    def ternary_check_input_output_mem_overlap(self, op, device,
                                               expected_failure=False):
        # [FUNCTION DOC] HELPER: Shared helper: checks an op taking three tensor inputs and out=
        # for memory-overlap handling by placing the data tensor in each of the three input
        # positions (other two are random tensors) and delegating to the unary overlap checker.
        sz = 9
        data = torch.randn(2 * sz, device=device)
        other1 = torch.randn(sz, device=device)
        other2 = torch.randn(sz, device=device)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(input, other1.view(input.shape), other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), input, other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), other2.view(input.shape), input, out=out),
            expected_failure=expected_failure)

    def _select_broadcastable_dims(self, dims_full=None):
        # [FUNCTION DOC] HELPER: Randomly picks broadcast-compatible shapes. Returns a tuple
        # (smaller dims, larger dims, full dims) where the full dims are either given or random (1
        # to 4 dims, sizes 1 to 8), and the two others randomly replace sizes with 1 (and the
        # smaller may have fewer dims) so they broadcast against the full shape.
        # select full dimensionality
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # select actual dimensions for ops:
        # larger: full ndims, individual sizes may be reduced
        # smaller: possibly reduced ndims, sizes may be reduced
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # no reduced singleton dimension
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # larger may have reduced singleton dimension
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # smaller may have reduced singleton dimension
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    # collected tests of ops that used scalar_check in Declarations.cwrap for
    # correctness
    def test_scalar_check(self, device):
        # [FUNCTION DOC] Collection of result-shape checks for ops that once used scalar_check in
        # the legacy declarations. Using a zero-dim and a one-element tensor, it verifies that
        # unary/binary pointwise ops, clamp, cumulative ops, sort/topk, max/min, shifts and
        # bitwise or/and, clone, masked_select, mode, amax/amin, set_, take, gather and normal
        # give an empty shape for zero-dim inputs and a one-element shape when mixed with a
        # vector. Also checks conv2d with a zero-dim input raises RuntimeError, nll_loss with
        # zero-dim input raises ValueError, and nll_loss with mean/sum reduction returns a
        # zero-dim result.
        zero_d = torch.randn((), device=device)
        one_d = torch.randn((1,), device=device)

        # remainder
        self.assertEqual((), torch.remainder(zero_d, zero_d).shape)
        self.assertEqual((), torch.remainder(zero_d, 2).shape)
        self.assertEqual((1,), torch.remainder(zero_d, one_d).shape)
        self.assertEqual((1,), torch.remainder(one_d, zero_d).shape)

        # fmod
        self.assertEqual((), torch.fmod(zero_d, zero_d).shape)
        self.assertEqual((), torch.fmod(zero_d, 2).shape)
        self.assertEqual((1,), torch.fmod(zero_d, one_d).shape)
        self.assertEqual((1,), torch.fmod(one_d, zero_d).shape)

        # exp, cos, cosh, tan, atan, tanh, erf, erfc, reciprocal
        self.assertEqual((), torch.exp(zero_d).shape)
        self.assertEqual((), torch.cos(zero_d).shape)
        self.assertEqual((), torch.cosh(zero_d).shape)
        self.assertEqual((), torch.tan(zero_d).shape)
        self.assertEqual((), torch.atan(zero_d).shape)
        self.assertEqual((), torch.acosh(zero_d).shape)
        self.assertEqual((), torch.asinh(zero_d).shape)
        self.assertEqual((), torch.atanh(zero_d).shape)
        self.assertEqual((), torch.tanh(zero_d).shape)
        self.assertEqual((), torch.erf(zero_d).shape)
        self.assertEqual((), torch.erfc(zero_d).shape)
        self.assertEqual((), torch.reciprocal(zero_d).shape)
        self.assertEqual((1,), torch.exp(one_d).shape)
        self.assertEqual((1,), torch.cos(one_d).shape)
        self.assertEqual((1,), torch.cosh(one_d).shape)
        self.assertEqual((1,), torch.tan(one_d).shape)
        self.assertEqual((1,), torch.atan(one_d).shape)
        self.assertEqual((1,), torch.acosh(one_d).shape)
        self.assertEqual((1,), torch.asinh(one_d).shape)
        self.assertEqual((1,), torch.atanh(one_d).shape)
        self.assertEqual((1,), torch.tanh(one_d).shape)
        self.assertEqual((1,), torch.erf(one_d).shape)
        self.assertEqual((1,), torch.erfc(one_d).shape)
        self.assertEqual((1,), torch.reciprocal(one_d).shape)

        # clamp
        self.assertEqual((), torch.clamp(zero_d, min=0, max=1).shape)
        self.assertEqual((), torch.clamp(zero_d, min=0).shape)
        self.assertEqual((), torch.clamp(zero_d, max=1).shape)
        self.assertEqual((1,), torch.clamp(one_d, min=0, max=1).shape)
        self.assertEqual((1,), torch.clamp(one_d, min=0).shape)
        self.assertEqual((1,), torch.clamp(one_d, max=1).shape)

        # cumsum, cumprod, cummax, cummin
        self.assertEqual((), torch.logcumsumexp(zero_d, 0).shape)
        self.assertEqual((), torch.cumsum(zero_d, 0).shape)
        self.assertEqual((), torch.cumprod(zero_d, 0).shape)
        self.assertEqual((), torch.cummax(zero_d, 0)[0].shape)
        self.assertEqual((), torch.cummin(zero_d, 0)[0].shape)

        # sort, topk
        self.assertEqual([(), ()], [x.shape for x in torch.sort(zero_d, 0, False)])
        self.assertEqual([(), ()], [x.shape for x in torch.sort(zero_d, 0, True)])
        self.assertEqual([(), ()], [x.shape for x in torch.topk(zero_d, 1, 0, False)])
        self.assertEqual([(), ()], [x.shape for x in torch.topk(zero_d, 1, 0, True)])

        # max, min
        self.assertEqual((), torch.max(zero_d, zero_d).shape)
        self.assertEqual((1,), torch.max(one_d, zero_d).shape)
        self.assertEqual((1,), torch.max(zero_d, one_d).shape)
        self.assertEqual((), torch.min(zero_d, zero_d).shape)
        self.assertEqual((1,), torch.min(one_d, zero_d).shape)
        self.assertEqual((1,), torch.min(zero_d, one_d).shape)

        zero_d_int = torch.tensor(1, device=device)
        one_d_int = torch.tensor([1], device=device)

        # lshift, rshift
        self.assertEqual((), (zero_d_int >> zero_d_int).shape)
        self.assertEqual((), (zero_d_int >> 1).shape)
        self.assertEqual((1,), (one_d_int >> zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int >> one_d_int).shape)
        self.assertEqual((1,), (one_d_int >> 1).shape)

        self.assertEqual((), (zero_d_int << zero_d_int).shape)
        self.assertEqual((), (zero_d_int << 1).shape)
        self.assertEqual((1,), (one_d_int << zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int << one_d_int).shape)
        self.assertEqual((1,), (one_d_int << 1).shape)

        # or
        self.assertEqual((), (zero_d_int | zero_d_int).shape)
        self.assertEqual((), (zero_d_int | 1).shape)
        self.assertEqual((1,), (one_d_int | zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int | one_d_int).shape)
        self.assertEqual((1,), (one_d_int | 1).shape)

        # and
        self.assertEqual((), (zero_d_int & zero_d_int).shape)
        self.assertEqual((), (zero_d_int & 1).shape)
        self.assertEqual((1,), (one_d_int & zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int & one_d_int).shape)
        self.assertEqual((1,), (one_d_int & 1).shape)

        # clone
        self.assertEqual((), zero_d.clone().shape)

        zero_d_bool = torch.tensor(True, device=device)
        one_d_bool = torch.tensor([True], device=device)

        # masked_select
        self.assertEqual((1,), torch.masked_select(zero_d_bool, zero_d_bool).shape)
        self.assertEqual((1,), torch.masked_select(zero_d_bool, one_d_bool).shape)
        self.assertEqual((1,), torch.masked_select(one_d_bool, zero_d_bool).shape)

        torch.tensor(1, dtype=torch.uint8, device=device)
        torch.tensor([1], dtype=torch.uint8, device=device)

        # mode
        self.assertEqual([(), ()], [x.shape for x in torch.mode(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.mode(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.mode(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.mode(one_d, dim=0, keepdim=False)])

        # max
        self.assertEqual([(), ()], [x.shape for x in torch.max(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.max(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.max(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.max(one_d, dim=0, keepdim=False)])

        # amax
        self.assertEqual((), torch.amax(zero_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amax(zero_d, dim=0, keepdim=False).shape)
        self.assertEqual((1,), torch.amax(one_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amax(one_d, dim=0, keepdim=False).shape)

        # min
        self.assertEqual([(), ()], [x.shape for x in torch.min(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.min(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.min(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.min(one_d, dim=0, keepdim=False)])

        # amin
        self.assertEqual((), torch.amin(zero_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amin(zero_d, dim=0, keepdim=False).shape)
        self.assertEqual((1,), torch.amin(one_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amin(one_d, dim=0, keepdim=False).shape)

        # set_
        zero_d_clone = zero_d.clone()
        one_d_clone = one_d.clone()
        self.assertEqual((), zero_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), zero_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)
        self.assertEqual((), one_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), one_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)

        self.assertEqual((), zero_d.clone().set_(zero_d).shape)
        self.assertEqual((), one_d.clone().set_(zero_d).shape)
        self.assertEqual((1,), zero_d.clone().set_(one_d).shape)
        self.assertEqual((1,), one_d.clone().set_(one_d).shape)

        # take
        self.assertEqual((), torch.randn((2, 3), device=device).take(zero_d_int).shape)
        self.assertEqual((1,), torch.randn((2, 3), device=device).take(one_d_int).shape)

        # gather
        self.assertEqual((), torch.gather(zero_d, 0, torch.zeros((), dtype=torch.int64, device=device)).shape)
        self.assertEqual((1,), torch.gather(zero_d, 0, torch.zeros((1,), dtype=torch.int64, device=device)).shape)
        self.assertEqual((), torch.gather(one_d, 0, torch.zeros((), dtype=torch.int64, device=device)).shape)
        self.assertEqual((1,), torch.gather(one_d, 0, torch.zeros((1,), dtype=torch.int64, device=device)).shape)

        # normal
        # std must be >= 0
        zero_d_ge_0 = torch.rand((), device=device)
        # documentation says out shape matches shape of mean
        self.assertEqual((), torch.normal(zero_d, zero_d_ge_0).shape)
        self.assertEqual((1,), torch.normal(one_d, zero_d_ge_0).shape)
        self.assertEqual((), torch.normal(1, zero_d_ge_0).shape)
        self.assertEqual((), torch.normal(zero_d, 1).shape)
        self.assertEqual((1,), torch.normal(one_d, 1).shape)
        # TODO: this behavior differs on CPU and GPU, see https://github.com/pytorch/pytorch/issues/30480.
        # self.assertEqual((), torch.normal(zero_d, one_d).shape)
        # self.assertEqual((), torch.normal(1, one_d).shape)

        # convolutions.  Yes, we are testing nn.functional here; seems justified
        # given its similar to the other tests
        w = torch.randn(2, 1, 3, 3, device=device).div_(2).requires_grad_()
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.conv2d(zero_d, w, groups=1))
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.conv2d(zero_d, w, groups=2))

        # nll_loss -- verify input can't be 0-dimensional.
        self.assertRaises(ValueError, lambda: torch.nn.functional.nll_loss(zero_d, zero_d, reduction='none'))
        self.assertRaises(ValueError, lambda: torch.nn.functional.nll_loss(zero_d, one_d, reduction='none'))
        # verify output is 0-dimensional when reduction != 'none'
        for (input, target) in ((torch.randn(1, 1, device=device), torch.tensor([0], device=device)),
                                (torch.randn(1, 1, 1, 1, device=device), torch.tensor([[[0]]], device=device))):
            self.assertEqual((), torch.nn.functional.nll_loss(input, target, reduction='mean').shape)
            self.assertEqual((), torch.nn.functional.nll_loss(input, target, reduction='sum').shape)

    # Test that `torch._check_tensor_all` raises errors in the correct cases
    def test_check_tensor_all(self, device):
        # [FUNCTION DOC] Tests torch._check_tensor_all error behavior: a non-tensor condition
        # raises TypeError, a non-bool tensor raises TypeError, and for bool tensors of many
        # shapes an all-True tensor passes while all-False (or all-True with one random element
        # cleared) raises RuntimeError with the default message. Also checks that custom messages
        # (a plain string, a message built from a tensor, and a formatted string) appear in the
        # error text.
        default_message = 'Expected cond to be True'
        check_fn = torch._check_tensor_all
        expected_error = RuntimeError

        # cond must be a tensor
        with self.assertRaisesRegex(TypeError, 'cond must be a tensor'):
            check_fn(True)

        # cond tensor must be boolean
        with self.assertRaisesRegex(TypeError, 'cond tensor must have dtype torch.bool'):
            check_fn(torch.ones(1, device=device))

        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        for size in test_sizes:
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # Should not raise error
            check_fn(t_all_true)

            with self.assertRaisesRegex(expected_error, default_message):
                check_fn(t_all_false)

            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # Choose a random element to set to false
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                with self.assertRaisesRegex(expected_error, default_message):
                    check_fn(t_all_true_but_one)

            # Test a simple failure message
            message = 'message'
            with self.assertRaisesRegex(expected_error, message):
                check_fn(t_all_false, lambda: message)

            # Test message with tensor
            def message():
                # [FUNCTION DOC] HELPER: Return a small arange tensor; used as a lazily computed
                # failure message containing a tensor.
                return torch.arange(4)

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)

            # Test format string message
            def message():
                # [FUNCTION DOC] HELPER: Return a formatted string combining a string, a list
                # of mixed values, a bool, an int and a tensor; used as a lazily computed
                # failure message.
                return f"{'test'} {[1, 2, 'a', True]} {True} {100} {torch.arange(4)}"

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)

    # Test that `TORCH_CHECK_TENSOR_ALL` raises errors that propagate from C++ to Python
    def test_check_tensor_internal(self, device):
        # [FUNCTION DOC] Checks the C++ TORCH_CHECK_TENSOR_ALL macro through the internal test op:
        # for bool tensors of many shapes an all-True tensor passes, while all-False or
        # all-True-with-one-False raises RuntimeError with the macro's test message.
        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        for size in test_sizes:
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # Should not raise error
            torch._test_check_tensor(t_all_true)

            with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                torch._test_check_tensor(t_all_false)

            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # Choose a random element to set to false
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                    torch._test_check_tensor(t_all_true_but_one)

    # Uses mismatched arange out size to trigger a warning
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref perturbs line numbering")
    def test_cpp_warnings_have_python_context(self, device):
        # [FUNCTION DOC] Verifies that warnings raised from C++ carry Python context. For an eager
        # call that triggers an arange out-size warning, checks the message mentions the C++
        # source location, the warning line number equals the line of the offending statement in
        # the helper (computed relative to the current frame line) and that exactly one warning is
        # recorded. Repeats for the scripted version of the helper (message and count only) and
        # for a scripted function that emits a Python warning, whose line number is again checked
        # relative to the frame line. Skipped under TorchDynamo and crossref.
        # KNOWN ISSUE: the expected line offsets (8 and 12) are hard coded; the doc comments in the
        # nested helpers sit above the statements so the offsets still hold, but any extra line
        # added between a warning statement and the later frame query breaks the test.
        # Creates long string in advance to avoid a too-long Python line
        s = ".+Triggered internally at.+RangeFactories.+"

        def cpp_warn_fn():
            # [FUNCTION DOC] HELPER: Create an empty tensor of 5 elements and call arange into it
            # with a size-3 range, which emits a C++ resize warning; returns the output tensor.
            out = torch.empty((5,))
            torch.arange(0, 3, out=out)
            return out

        # Checks eager-mode cpp warning
        with warnings.catch_warnings(record=True) as w:
            # nvfuser deprecation warning filter
            warnings.filterwarnings("ignore", "torch::jit::fuser::cuda", UserWarning)
            cpp_warn_fn()
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            warning = w[0]

            # Checks for cpp context in the warning message
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # Checks the Python features of the warning
            # Note: the eager mode warning refers to the line in the function
            # that throws the warning.
            self.assertEqual(frameinfo.lineno - 8, warning.lineno)
            self.assertEqual(len(w), 1)

        # Checks jitted cpp warning
        with warnings.catch_warnings(record=True) as w:
            # nvfuser deprecation warning filter
            warnings.filterwarnings("ignore", "torch::jit::fuser::cuda", UserWarning)
            # ignore all deprecation warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            # torch.jit.script emits a visible FutureWarning; ignore it too
            warnings.filterwarnings("ignore", category=FutureWarning)
            scripted_cpp_warn_fn = torch.jit.script(cpp_warn_fn)
            scripted_cpp_warn_fn()
            warning = w[0]

            # Checks for cpp context in the warning message
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # Checks the Python features of the warning
            # Note: the jitted warning's lineno refers to the call to the jitted
            # function, which in our test suite has a layer of indirection
            # that makes checking the Python lineno fragile
            self.assertEqual(len(w), 1)

        # Checks jitted Python warning
        def warn_fn():
            # [FUNCTION DOC] HELPER: Emit a Python UserWarning with the text Warning!; scripted
            # with torch.jit.script to check the warning line number.
            warnings.warn("Warning!")

        # The jit mimics an eager-mode Python warning in this case
        with warnings.catch_warnings(record=True) as w:
            # nvfuser deprecation warning filter
            warnings.filterwarnings("ignore", "torch::jit::fuser::cuda", UserWarning)
            # ignore all deprecation warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            # torch.jit.script emits a visible FutureWarning; ignore it too
            warnings.filterwarnings("ignore", category=FutureWarning)
            scripted_warn_fn = torch.jit.script(warn_fn)
            scripted_warn_fn()
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            warning = w[0]

            self.assertTrue(re.search('Warning!', str(warning.message)) is not None)

            # Checks the Python features of the warning
            self.assertEqual(frameinfo.lineno - 12, warning.lineno)
            self.assertEqual(len(w), 1)

    # FIXME: move to test_testing
    @onlyCPU
    def test_warn_always_caught(self, device):
        # [FUNCTION DOC] CPU only. Checks that a warn-once warning about non-writable NumPy arrays
        # can be caught repeatedly with the assertWarnsOnceRegex context manager (calling it twice
        # in a row, and with two warnings in one block).
        # Check that we can catch a TORCH_WARN_ONCE warning twice
        # since assertWarnsOnceRegex uses set_warn_always(True) which changes
        # TORCH_WARN_ONCE to TORCH_WARN
        a = np.arange(10)
        a.flags.writeable = False
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # OK, got it once, now try again
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # Make sure emitting two warnings will pass the assertWarnsOnceRegex
        # context manager
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)
            torch.from_numpy(a)

    @onlyNativeDeviceTypes
    def test_complex_half_experimental_warning(self, device):
        # [FUNCTION DOC] Checks that creating chalf (complex half) tensors through randn, rand,
        # empty, ones, zeros, their _like variants and a simple add each emit the once-only
        # experimental-support UserWarning.
        msg = 'ComplexHalf support is experimental'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            t = torch.randn(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.randn_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            # t + 1 allocates a new tensor for result using empty
            t + 1

    @onlyCUDA
    def test_dtypetensor_warnings(self, device):
        # [FUNCTION DOC] CUDA only. Checks that using the legacy torch.cuda FloatTensor and
        # DoubleTensor constructors emits the warning that dtype tensor constructors are no
        # longer recommended.
        msg = 'The torch.cuda.*DtypeTensor constructors are no longer recommended'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.cuda.FloatTensor([0])

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.cuda.DoubleTensor([0])

    def test_set_default_tensor_type_warnings(self, device):
        # [FUNCTION DOC] Checks that torch.set_default_tensor_type emits the deprecation warning
        # for the CPU float tensor type (and CUDA float type when CUDA is available); restores the
        # original default tensor type in a finally clause.
        msg = '.*is deprecated as of PyTorch 2.1, please use torch.set_default_dtype().*'
        default_type = torch.tensor([]).type()
        try:
            with self.assertWarnsOnceRegex(UserWarning, msg):
                torch.set_default_tensor_type(torch.FloatTensor)

            if torch.cuda.is_available():
                with self.assertWarnsOnceRegex(UserWarning, msg):
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
        finally:
            torch.set_default_tensor_type(default_type)

    # TODO: this test should be in test_nn.py
    def test_conv_transposed_backward_agnostic_to_memory_format(self, device):
        # [FUNCTION DOC] Runs forward and backward through ConvTranspose1d followed by LayerNorm,
        # and through ConvTranspose3d with a transposed (non-contiguous) output gradient, to
        # check backward works regardless of memory format; no value assertions, only that
        # nothing raises.
        in_channels = 64
        out_channels = 128
        scale_factor = 8
        batch_size = 8
        length = 16

        conv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor).to(device)
        layer_norm = torch.nn.LayerNorm(out_channels).to(device)

        input_ = torch.randn(batch_size, in_channels, length).to(device).contiguous()
        input_ = conv(input_).contiguous()
        input_ = layer_norm(input_.transpose(1, 2).contiguous()).contiguous()
        input_.sum().backward()

        # 3d
        conv = torch.nn.ConvTranspose3d(3, 3, kernel_size=3).to(device)
        input = torch.randn(batch_size, 3, length, length, length, device=device)
        out = conv(input)
        out.backward(torch.ones_like(out).transpose(-2, -1))

    # TODO: this test should be in test_nn.py
    @onlyCUDA
    @largeTensorTest('12GB')
    def test_conv_transposed_large(self, device):
        # [FUNCTION DOC] CUDA only, needs 12GB. Runs a ConvTranspose3d forward on a very large
        # input to check large-tensor support (issue 32866); passes if nothing raises.
        # ConvTranspose3d works for large input tensors (gh-32866)
        in_channels = 64
        out_channels = 128
        kernel_size = 5

        conv = torch.nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=2, padding=2, output_padding=1).to(device)

        x = torch.rand([1, 64, 8, 128, 172]).to(device)
        conv(x)

    def test_is_set_to(self, device):
        # [FUNCTION DOC] Tests Tensor.is_set_to: False for separate tensors, True (symmetrically)
        # after set_ to share storage, False after a resize to a different shape, False for
        # storage-less empty tensors, works for bool dtype, and requires matching sizes/shapes
        # including the empty-size case.
        t1 = torch.empty(3, 4, 9, 10, device=device)
        t2 = torch.empty(3, 4, 9, 10, device=device)
        t3 = torch.tensor([], device=device).set_(t1)
        t4 = t3.clone().resize_(12, 90)
        self.assertFalse(t1.is_set_to(t2))
        self.assertTrue(t1.is_set_to(t3))
        self.assertTrue(t3.is_set_to(t1), "is_set_to should be symmetric")
        self.assertFalse(t1.is_set_to(t4))
        self.assertFalse(torch.tensor([]).is_set_to(torch.tensor([])),
                         "Tensors with no storages should not appear to be set "
                         "to each other")

        t1 = torch.tensor([True, True], dtype=torch.bool, device=device)
        t2 = torch.tensor([0], dtype=torch.bool, device=device).set_(t1)
        self.assertTrue(t1.is_set_to(t2))

        # test that sizes must match
        t1 = torch.empty([2, 3, 4], device=device)
        t2 = t1.view(4, 3, 2)
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

        # test that legacy empty size behavior used to be respected (i.e. all
        # empty tensors were logically collapsed to size [0]).
        t1 = torch.empty([2, 5, 0], device=device)
        t2 = t1.view([0])
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

    # See https://github.com/pytorch/pytorch/issues/72650
    @skipMeta
    @parametrize(
        "fn",
        [
            "dist", "atan2", "pow", "lerp", "add", "sub", "mul", "div", "fmod", "remainder", "eq", "ge", "gt", "le",
            "lt", "max", "min", "ne", "addcdiv", "addcmul", "masked_scatter", "masked_select", "masked_fill", "map",
            "map2", "copy",
        ],
    )
    def test_broadcast(self, fn, device):
        # [FUNCTION DOC] Parametrized over many op names (skipped for meta). Builds random
        # broadcast-compatible tensors and checks that for the Tensor-method form, the torch
        # function form and the in-place form, calling with broadcast (unexpanded) arguments gives
        # the same result as with manually expanded arguments. In-place variants are only checked
        # where the in-place tensor is not itself broadcast, and incompatible shapes must raise
        # RuntimeError (except under inductor). Map and map2 are skipped on CUDA.
        # functions with three tensor arguments
        fns_3_args = {"map2"}
        fns_value_kwarg = {"addcdiv", "addcmul"}

        (dims_small, dims_large, dims_full) = self._select_broadcastable_dims()
        full1d = torch.randn(*dims_full, device=device).flatten().float()
        small = torch.randn(*dims_small, device=device).float()
        large = torch.randn(*dims_large, device=device).float()
        small_expanded = small.expand(*dims_full)
        large_expanded = large.expand(*dims_full)
        small2 = None
        small2_expanded = None
        if fn in fns_3_args or fn in fns_value_kwarg:
            # create another smaller tensor
            (dims_small2, _, _) = self._select_broadcastable_dims(dims_full)
            small2 = torch.randn(*dims_small2, device=device).float()
            small2_expanded = small2.expand(*dims_full)

        if small.is_cuda and fn in ['map', 'map2']:
            # map and map2 are not implemented on CUDA tensors
            return

        if hasattr(large_expanded, fn):
            # run through tensor versions of functions
            # and verify fully expanded inputs give same results
            expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

            def tensorfn(myfn, t1, t2):
                # [FUNCTION DOC] HELPER: Call the given Tensor method myfn with arguments chosen
                # according to the op name of the enclosing test: lerp with weight 0.5, masked ops
                # with a mask built from t1, three-argument ops with scalar 1 first, value-kwarg
                # ops with value 1, otherwise a single argument.
                if fn == "lerp":
                    return myfn(t1, 0.5)
                elif fn == "masked_select":
                    return myfn(t1 < 0)
                elif fn == "masked_scatter":
                    return myfn(t1 < 0.5, full1d)
                elif fn == "masked_fill":
                    return myfn(t1 < 0.5, 1.0)
                elif fn in fns_3_args:
                    return myfn(1, t1, t2)
                elif fn in fns_value_kwarg:
                    return myfn(t1, t2, value=1)
                else:
                    return myfn(t1)

            # test various orders
            for first, second, third in [(large, small, small2), (small, large, small2),
                                         (small2, small, large), (small2, large, small)]:
                if first is None:
                    break  # ignore last iter when small2 is None
                method_expanded = getattr(expanded[first], fn)
                method = getattr(first, fn)
                r1 = tensorfn(method_expanded, expanded[second], expanded[third])
                r2 = tensorfn(method, second, third)
                self.assertEqual(r1, r2)

        # now for torch. versions of functions
        if hasattr(torch, fn):
            fntorch = getattr(torch, fn)
            expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

            def torchfn(t1, t2, t3):
                # [FUNCTION DOC] HELPER: Call the torch function for the op name of the
                # enclosing test with arguments appropriate to it (lerp weight, masks built from
                # t2, scalar multiplier for three-argument ops, value 1.0 for addcdiv/addcmul,
                # or plain two tensors).
                if fn == "lerp":
                    return fntorch(t1, t2, 0.5)
                elif fn == "masked_select":
                    return fntorch(t1, t2 < 0)
                elif fn == "masked_scatter":
                    return fntorch(t1, t2 < 0.5, full1d)
                elif fn == "masked_fill":
                    return fntorch(t1, t2 < 0.5, 1.0)
                elif fn in fns_3_args:
                    return fntorch(t1, 1.0, t2, t3)
                elif fn in fns_value_kwarg:
                    return fntorch(t1, t2, t3, value=1.0)
                else:
                    return fntorch(t1, t2)

            # test various orders
            for first, second, third in [(large, small, small2), (small, large, small2),
                                         (small2, small, large), (small2, large, small)]:
                if first is None:
                    break  # ignore last iter when small2 is None
                r1 = torchfn(expanded[first], expanded[second], expanded[third])
                r2 = torchfn(first, second, third)
                self.assertEqual(r1, r2)

        # now for in place functions
        # in-place tensor is not broadcastable; test only guaranteed
        # to work by broadcasting other argument(s)
        if not hasattr(large_expanded, fn + "_"):
            return

        # need to clone largeExpanded so we can reuse, since functions are in-place
        large_expanded_clone = large_expanded.clone()

        def tensorfn_inplace(t0, t1, t2=None):
            # [FUNCTION DOC] HELPER: Call the in-place method named by the op name plus underscore
            # on t0 with arguments appropriate to the op (lerp weight, masks, lambdas for map and
            # map2, scalar for three-argument ops, value kwarg), and return its result.
            t0_fn = getattr(t0, fn + "_")
            if fn == "lerp":
                return t0_fn(t1, 0.5)
            elif fn == "masked_scatter":
                return t0_fn(t1 < 0.5, full1d)
            elif fn == "masked_fill":
                return t0_fn(t1 < 0.5, 1.0)
            elif fn == "map":
                return t0_fn(t1, lambda x, y: x + y)
            elif fn == "map2":
                return t0_fn(t1, t2, lambda x, y, z: x + y + z)
            elif fn in fns_3_args:
                return t0_fn(1.0, t1, t2)
            elif fn in fns_value_kwarg:
                return t0_fn(t1, t2, value=1.0)
            else:
                return t0_fn(t1)
        # in-place pointwise operations don't actually work if the in-place
        # tensor is 0-strided (numpy has the same issue)
        if (0 not in large_expanded.stride() and 0 not in large_expanded_clone.stride()):
            r1 = tensorfn_inplace(large_expanded, small_expanded, small2_expanded)
            r2 = tensorfn_inplace(large_expanded_clone, small, small2)
            self.assertEqual(r1, r2)

        def broadcastable(t0, t1, t2=None):
            # [FUNCTION DOC] HELPER: Return True if t1 (and t2 if given) can be expanded to the
            # shape of t0, and False if expand_as raises RuntimeError.
            try:
                t1.expand_as(t0)
                if t2 is not None:
                    t2.expand_as(t0)
            except RuntimeError:
                return False
            return True

        def _test_in_place_broadcastable(t0, t1, t2=None):
            # [FUNCTION DOC] HELPER: Check an in-place op for broadcasting: if the arguments
            # cannot broadcast to t0 and their element counts differ, require RuntimeError
            # (skipped under inductor); if they can broadcast, simply run the op.
            if not broadcastable(t0, t1, t2):
                same_size = t0.numel() == t1.numel() and (t0.numel() == t2.numel() if t2 is not None else True)
                if not same_size:
                    # Functionalization converts the inplace to an out-of-place, which causes us to error.
                    # We should fix this, but "error probably on bad inputs" isn't a hi-pri PT2 item.
                    if not TEST_WITH_TORCHINDUCTOR:
                        self.assertRaises(RuntimeError, lambda: tensorfn_inplace(t0, t1, t2))
            else:
                tensorfn_inplace(t0, t1, t2)

        if fn not in fns_3_args and fn not in fns_value_kwarg:
            _test_in_place_broadcastable(small, large_expanded)
            _test_in_place_broadcastable(small, large)
        else:
            _test_in_place_broadcastable(small2, small_expanded, large_expanded)
            _test_in_place_broadcastable(small2, small, large)

    @onlyCPU
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*get_all_qint_dtypes())
    def test_nondeterministic_resize_quantized(self, device, dtype):
        # [FUNCTION DOC] CPU only, quantized dtypes. Checks that resizing a quantized tensor emits
        # the nondeterministic alert for the CPU quantized resize kernel. Skipped under inductor.
        a = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.float, device=device)
        b = torch.quantize_per_tensor(a, 0.1, 10, dtype)
        self.check_nondeterministic_alert(
            lambda: b.resize_((10,)),
            'quantized_resize_cpu_')

    @skipXLA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    def test_deterministic_resize(self, device, dtype):
        # [FUNCTION DOC] Over a table of (size, stride, resize target) cases, creates a
        # zero-filled tensor, resizes it in place under DeterministicGuard with
        # fill_uninitialized_memory enabled and compares storages. If the storage grew, the old
        # prefix must be unchanged and the new section must be NaN for float/complex dtypes, True
        # for bool, or the dtype's max integer value; otherwise the storage must be unchanged.
        # Skipped on XLA and inductor.
        test_cases = [
            # size, stride, resize_size
            ((10,), (1,), (5,)),
            ((10,), (0,), (10,)),
            ((10,), (1,), (20,)),
            ((2, 3, 4), None, (2, 3, 4)),
            ((2, 3, 4), None, (6, 3, 4)),
            ((2, 3, 4), None, (2, 5, 4)),
            ((2, 3, 4), None, (2, 3, 6)),
            ((2, 3, 4), None, (3, 4, 5)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (4, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 4, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 5)),
            ((2, 3, 4), (1, 4, 12), (3, 4, 5)),
            ((2, 3, 4), (1, 0, 1), (2, 4, 5)),
        ]

        for size, stride, resize_size in test_cases:
            if stride is None:
                a = torch.zeros(size, dtype=dtype, device=device)
            else:
                a = torch.empty_strided(size, stride, dtype=dtype, device=device).fill_(0)
            old_storage = a.untyped_storage().clone()
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                a.resize_(resize_size)

            new_storage = a.untyped_storage()

            # If storage size was increased, check that the new section is
            # filled with NaN/MAX_INT. Otherwise, check that the storages are
            # equal.
            old_tensor = torch.tensor(old_storage, dtype=dtype)
            old_numel = old_tensor.numel()
            new_tensor = torch.tensor(new_storage, dtype=dtype)
            new_numel = new_tensor.numel()

            if new_numel > old_numel:
                self.assertEqual(new_tensor[:old_numel], old_tensor)
                fill_section = new_tensor[old_numel:]

                if dtype.is_floating_point or dtype.is_complex:
                    self.assertTrue(fill_section.isnan().all())
                else:
                    if dtype == torch.bool:
                        max_val = True
                    else:
                        max_val = torch.iinfo(dtype).max
                    self.assertTrue(fill_section.eq(max_val).all())
            else:
                self.assertEqual(old_tensor, new_tensor)

    # When deterministic algorithms are enabled, `torch.empty` should fill floating
    # point tensors with NaN and integer tensors with MAX_INT
    @skipXLA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*all_types_and_complex_and(
        torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64, torch.complex32))
    def test_deterministic_empty(self, device, dtype):
        # [FUNCTION DOC] Under DeterministicGuard with fill_uninitialized_memory, builds
        # uninitialized tensors with empty, empty with out, empty_like (with and without memory
        # format), empty_strided and empty_permuted and checks every element is NaN
        # (float/complex) or the max integer value (True for bool). Skipped on XLA and inductor.
        gen_fns = [
            lambda: torch.empty(10, 9, device=device, dtype=dtype),
            lambda: torch.empty(10, 9, out=torch.zeros(1, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype), memory_format=torch.contiguous_format),
            lambda: torch.empty_strided((10, 9), (1, 5), device=device, dtype=dtype),
            lambda: torch.empty_permuted((2, 3, 5), (1, 0, 2), device=device, dtype=dtype),
        ]

        for gen_fn in gen_fns:
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                res = gen_fn()

            if dtype.is_floating_point or dtype.is_complex:
                self.assertTrue(res.isnan().all())
            else:
                if dtype == torch.bool:
                    max_val = True
                else:
                    max_val = torch.iinfo(dtype).max
                self.assertTrue(res.eq(max_val).all())

    # FIXME: update OpInfos to support "nondeterministic samples" and port these tests
    #   to that architecture
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AvgPool3d(self, device):
        # [FUNCTION DOC] Runs a 3D average pooling forward then its backward and checks, via
        # check_nondeterministic_alert, that the CUDA backward kernel for avg_pool3d raises or
        # warns only when the device is CUDA. Skipped under inductor.
        module = torch.nn.AvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveAvgPool2d(self, device):
        # [FUNCTION DOC] Same pattern for AdaptiveAvgPool2d: backward on a small 3D input must
        # trigger the nondeterministic alert for the CUDA adaptive_avg_pool2d backward only on
        # CUDA devices.
        module = torch.nn.AdaptiveAvgPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveAvgPool3d(self, device):
        # [FUNCTION DOC] Same pattern for AdaptiveAvgPool3d: the alert for the CUDA
        # adaptive_avg_pool3d backward is expected only on CUDA devices.
        module = torch.nn.AdaptiveAvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_MaxPool3d(self, device):
        # [FUNCTION DOC] Same pattern for MaxPool3d: backward through a max pool with indices
        # expects the nondeterministic alert for the CUDA max_pool3d backward kernel only on CUDA.
        module = torch.nn.MaxPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'max_pool3d_with_indices_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveMaxPool2d(self, device):
        # [FUNCTION DOC] Same pattern for AdaptiveMaxPool2d: the alert for the CUDA
        # adaptive_max_pool2d backward is expected only on CUDA devices.
        module = torch.nn.AdaptiveMaxPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_FractionalMaxPool2d(self, device):
        # [FUNCTION DOC] Same pattern for FractionalMaxPool2d with output ratio 0.5: the alert for
        # the CUDA fractional_max_pool2d backward is expected only on CUDA devices.
        module = torch.nn.FractionalMaxPool2d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_FractionalMaxPool3d(self, device):
        # [FUNCTION DOC] Same pattern for FractionalMaxPool3d with output ratio 0.5: the alert for
        # the CUDA fractional_max_pool3d backward is expected only on CUDA devices.
        module = torch.nn.FractionalMaxPool3d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool1d(self, device, dtype):
        # [FUNCTION DOC] For floating dtypes and native devices, runs MaxUnpool1d forward with
        # zero indices and checks the nondeterministic alert for the max_unpooling2d forward
        # kernel (which the 1d module reuses) is produced on every device.
        module = torch.nn.MaxUnpool1d(3, 1)
        input = torch.randn(1, 1, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool2d(self, device, dtype):
        # [FUNCTION DOC] For floating dtypes and native devices, runs MaxUnpool2d forward with
        # zero indices and checks the nondeterministic alert for the max_unpooling2d forward
        # kernel is produced on every device.
        module = torch.nn.MaxUnpool2d(3, 1)
        input = torch.randn(1, 1, 7, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool3d(self, device, dtype):
        # [FUNCTION DOC] For floating dtypes and native devices, runs MaxUnpool3d forward with
        # zero indices and checks the nondeterministic alert for the max_unpooling3d forward
        # kernel is produced on every device.
        module = torch.nn.MaxUnpool3d(3, 1)
        input = torch.randn(1, 1, 7, 7, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling3d_forward_out')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_linear(self, device):
        # [FUNCTION DOC] Linear interpolation (size 12, align_corners False) backward must trigger
        # the nondeterministic alert for the CUDA upsample_linear1d backward only on CUDA. Skipped
        # under inductor.
        input = torch.randn(1, 2, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='linear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_linear1d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_bilinear(self, device):
        # [FUNCTION DOC] Bilinear interpolation backward must trigger the nondeterministic alert
        # for the CUDA upsample_bilinear2d backward only on CUDA. Skipped under inductor.
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bilinear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bilinear2d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    def test_no_nondeterministic_alert_interpolate_bilinear(self, device):
        # [FUNCTION DOC] Checks that bilinear interpolation backward does not raise or warn the
        # nondeterministic alert for the bilinear kernel when no CUDA-only expectation is set
        # (should_alert is False).
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)

        def fn():
            # [FUNCTION DOC] HELPER: Run bilinear interpolation of the shared input to size 12 and
            # backpropagate a ones gradient; the callable checked for the absence of an alert.
            res = torch.nn.functional.interpolate(
                input,
                size=12,
                mode='bilinear',
                align_corners=False)
            grad = torch.ones_like(res)
            return res.backward(grad)

        self.check_nondeterministic_alert(
            fn,
            'upsample_bilinear2d_backward_out_cuda',
            False)

    def test_no_nondeterministic_alert_interpolate_trilinear(self, device):
        # [FUNCTION DOC] Checks that trilinear interpolation backward does not produce a
        # nondeterministic alert for the trilinear kernel (should_alert False).
        input = torch.randn(1, 2, 4, 4, 4, device=device, requires_grad=True)

        def fn():
            # [FUNCTION DOC] HELPER: Run trilinear interpolation of the shared input to size 12
            # and backpropagate a ones gradient; the callable checked for the absence of an alert.
            res = torch.nn.functional.interpolate(
                input,
                size=12,
                mode='trilinear',
                align_corners=False)
            grad = torch.ones_like(res)
            return res.backward(grad)

        self.check_nondeterministic_alert(
            fn,
            'upsample_trilinear3d_backward_out_cuda',
            False)

    @skipIfTorchInductor("aot-autograd issue")
    def test_deterministic_max_pool3d(self, device):
        # [FUNCTION DOC] Over several max_pool3d parameter sets (batched and unbatched), runs
        # forward with indices and backward five times under DeterministicGuard and checks the
        # input gradient is bitwise identical across runs. Skipped under inductor.
        test_cases = [
            # size, kernel_size, stride, padding, dilation, ceil_mode
            [(2, 3, 8, 8, 8), 3, 1, 1, 1, False],
            [(2, 3, 8, 8, 8), 3, 2, 1, 1, False],
            [(2, 3, 8, 8, 8), 2, 2, 0, 1, False],
            [(2, 3, 8, 8, 8), 3, 2, 1, 1, True],
            [(3, 8, 8, 8), 3, 1, 1, 1, False],  # unbatched
        ]

        for size, ks, st, pa, di, cm in test_cases:
            input = torch.randn(*size, device=device, requires_grad=True)
            grad = None
            with DeterministicGuard(True):
                for _ in range(5):
                    res, _ = torch.nn.functional.max_pool3d(
                        input, ks, st, pa, di, cm, return_indices=True)
                    res.backward(torch.ones_like(res))
                    if grad is None:
                        grad = input.grad
                    else:
                        self.assertEqual(grad, input.grad, atol=0, rtol=0)
                    input.grad = None

    @skipIfTorchInductor("aot-autograd issue")
    def test_deterministic_replication_pad2d(self, device):
        # [FUNCTION DOC] For several tensor sizes and padding tuples (including negative padding
        # except on XLA), runs replicate padding forward and backward under DeterministicGuard and
        # checks the gradient is consistent. Skipped under inductor.
        test_cases = [
            # size, padding
            [(1, 2, 4, 4), (0, 0, 0, 0)],
            [(1, 2, 4, 4), (3, 4, 5, 6)],
            [(3, 8, 7), (0, 0, 0, 0)],
            [(3, 8, 7), (4, 3, 2, 7)],
        ]

        if torch.device(device).type != 'xla':
            test_cases += [
                [(4, 3, 5, 10), (-9, 4, 5, 6)],
                [(3, 8, 7), (-4, -2, -2, -3)],
            ]

        for size, padding in test_cases:
            input = torch.randn(*size, device=device, requires_grad=True)
            grad = None
            with DeterministicGuard(True):
                res = torch.nn.functional.pad(
                    input,
                    padding,
                    mode='replicate')
                res.backward(torch.ones_like(res))
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                input.grad = None

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_deterministic_interpolate_bilinear(self, device):
        # [FUNCTION DOC] Runs bilinear interpolation forward and backward five times under
        # DeterministicGuard and checks the input gradient is bitwise equal each time. Skipped
        # under inductor.
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        grad = None
        with DeterministicGuard(True):
            for _ in range(5):
                res = torch.nn.functional.interpolate(
                    input,
                    size=12,
                    mode='bilinear',
                    align_corners=False)
                res.backward(torch.ones_like(res))
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                input.grad = None

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_bicubic(self, device):
        # [FUNCTION DOC] Bicubic interpolation backward must trigger the nondeterministic alert
        # for the CUDA upsample_bicubic2d backward only on CUDA. Skipped under inductor.
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bicubic',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bicubic2d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_trilinear(self, device):
        # [FUNCTION DOC] Trilinear interpolation backward must trigger the nondeterministic alert
        # for the CUDA upsample_trilinear3d backward only on CUDA. Skipped under inductor.
        input = torch.randn(1, 2, 4, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='trilinear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_trilinear3d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReflectionPad1d(self, device):
        # [FUNCTION DOC] ReflectionPad1d backward must trigger the nondeterministic alert for the
        # CUDA reflection_pad1d backward only on CUDA. Skipped under inductor.
        module = torch.nn.ReflectionPad1d((1, 2))
        input = torch.randn(2, 3, 8, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'reflection_pad1d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReflectionPad3d(self, device):
        # [FUNCTION DOC] ReflectionPad3d backward must trigger the nondeterministic alert for the
        # CUDA reflection_pad3d backward only on CUDA. Skipped under inductor.
        module = torch.nn.ReflectionPad3d((1, 2, 3, 4, 5, 6))
        input = torch.randn(2, 3, 8, 8, 8, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'reflection_pad3d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad1d(self, device):
        # [FUNCTION DOC] ReplicationPad1d backward must trigger the nondeterministic alert for the
        # CUDA replication_pad1d backward only on CUDA. Skipped under inductor.
        module = torch.nn.ReplicationPad1d((1, 2))
        input = torch.randn(2, 3, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad1d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad2d(self, device):
        # [FUNCTION DOC] ReplicationPad2d backward triggers the alert for the CUDA
        # replication_pad2d backward only on CUDA when the forward ran normally; if the forward
        # ran under DeterministicGuard, the backward must not alert on any device.
        module = torch.nn.ReplicationPad2d((1, 2, 3, 4))
        input = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        # Nondeterministic alert should only be raised if the forward call was
        # nondeterministic
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            torch.device(device).type == 'cuda')

        with DeterministicGuard(True):
            res = module(input)

        grad = torch.ones_like(res)

        # If the forward call was deterministic, nondeterministic alert should
        # not be raised
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            False)

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad3d(self, device):
        # [FUNCTION DOC] ReplicationPad3d backward must trigger the nondeterministic alert for the
        # CUDA replication_pad3d backward only on CUDA. Skipped under inductor.
        module = torch.nn.ReplicationPad3d((1, 2, 3, 4, 5, 6))
        input = torch.randn(2, 3, 4, 4, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchDynamo("Warning is not raised.")
    def test_nondeterministic_alert_NLLLoss(self, device):
        # [FUNCTION DOC] NLLLoss forward on a 4D input with 2D targets must trigger the
        # nondeterministic alert for the CUDA nll_loss2d forward only on CUDA. Skipped
        # under TorchDynamo.
        module = torch.nn.NLLLoss()
        input = torch.randn(2, 3, 5, 5, device=device)
        target = torch.rand(2, 5, 5, device=device).mul(3).floor().long()


        self.check_nondeterministic_alert(
            lambda: module(input, target),
            'nll_loss2d_forward_out_cuda_template',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_CTCLoss(self, device):
        # [FUNCTION DOC] CTCLoss backward with random log-probabilities, targets and lengths must
        # trigger the nondeterministic alert for the GPU ctc_loss backward only on CUDA. Skipped
        # under inductor.
        module = torch.nn.CTCLoss()
        input = torch.randn(50, 3, 15, device=device, requires_grad=True)
        target = torch.randint(0, 14, (3, 30), device=device)
        input_lengths = [50, 50, 50]
        target_lengths = [30, 25, 20]
        res = module(input, target, input_lengths, target_lengths)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'ctc_loss_backward_gpu',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_EmbeddingBag_max(self, device):
        # [FUNCTION DOC] EmbeddingBag in max mode (explicit weight with grad) backward must
        # trigger the nondeterministic alert for the CUDA embedding_bag max backward only on CUDA.
        # Skipped under inductor.
        module = torch.nn.EmbeddingBag(
            4, 3, None, 2., False, 'max',
            _weight=torch.randn(4, 3, device=device, requires_grad=True))
        input = torch.randint(0, 3, (4, 3), device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'embedding_bag_backward_cuda_max',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @onlyCUDA
    def test_deterministic_cumsum(self, device):
        # [FUNCTION DOC] CUDA only. For several large 1D sizes computes cumsum under
        # DeterministicGuard four times and checks repeatability bitwise, and agreement with the
        # CPU result within tolerance. Also does the same for a complex128 tensor sized to exceed
        # the number of thread blocks on a 132-SM GPU. Skipped under inductor.
        test_cases = [
            # size, dim
            [(1025,), 0],
            [(8193,), 0],
            [(8191,), 0],
            [(128256,), 0],
            [(1282560,), 0],
            [(12825600,), 0],
        ]
        for size, dim in test_cases:
            input = 100 * torch.rand(*size, device=device)
            with DeterministicGuard(True):
                res0 = input.cumsum(dim)
                for _ in range(3):
                    res1 = input.cumsum(dim)
                    self.assertEqual(res0, res1, atol=0, rtol=0)

            res_cpu = input.cpu().cumsum(dim)
            self.assertEqual(res0, res_cpu, atol=1e-3, rtol=1e-2)
            # test double complex that has fewer threads than CTAs
            # that's a problem for large GPUS (H100 with 132 sms)
            # test is very tailored for that and will provide 0 signal
            # on smaller GPUs
            num_sms = 132
            elems_per_cta = 256 * 16
            N = num_sms * elems_per_cta
            input = torch.rand(N, dtype=torch.complex128, device=device)
            with DeterministicGuard(True):
                res0 = input.cumsum(dim)
                for _ in range(3):
                    res1 = input.cumsum(dim)
                    self.assertEqual(res0, res1, atol=0, rtol=0)

            res_cpu = input.cpu().cumsum(dim)
            self.assertEqual(res0, res_cpu, atol=1e-3, rtol=1e-2)

    @onlyCUDA
    @largeTensorTest('49GB')
    def test_cumsum_64bit_indexing(self, device):
        # [FUNCTION DOC] CUDA only, needs 49GB. Builds a huge float tensor, computes cumsum along
        # the last dim out of place and in chunks in place, and compares the first and last rows
        # within tolerance to check 64-bit indexing.
        b = torch.ones(2 * 4096 * 8, 100000, dtype=torch.float, device='cuda')
        b /= 100000
        d = b.cumsum(dim=-1)
        chunk = 2**30 // b.shape[-1]
        for i in range(0, b.shape[0], chunk):
            end = min(i + chunk, b.shape[0])
            b[i:end, :].cumsum_(dim=-1)
        # cheat a bit to avoid OOM
        self.assertEqual(b[0, :], d[0, :], atol=3e-5, rtol=3e-5)
        self.assertEqual(b[-1, :], d[-1, :], atol=3e-5, rtol=3e-5)

    @onlyCUDA
    @largeTensorTest('48GB')
    def test_cumsum_outer_dim_64bit_indexing(self, device):
        # [FUNCTION DOC] CUDA only, needs 48GB. Runs cumsum over the middle dimension of a huge
        # zero tensor and checks the maximum is zero (64-bit indexing along the outer dimension).
        x = torch.zeros(309504, 1, 16384, device=device)
        torch.exp(x)
        cumsum = torch.cumsum(x, dim=1)
        self.assertEqual(cumsum.max().item(), 0., atol=0., rtol=0.)

    @expectedFailureMeta  # expected a non-determinitic error, but it was not raised
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_put(self, device):
        # [FUNCTION DOC] Calls Tensor.put and put_ with duplicate indices (not accumulating) and
        # expects the nondeterministic alert for put_ on every native device. Expected to fail on
        # meta because no alert is raised.
        a = torch.randn(10, device=device)
        indices = torch.tensor([0, 0], device=device)
        values = torch.tensor([0., 1.], device=device)

        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=False),
                'put_')

    # warn_only=False correctly raises RuntimeError: put_ does not have a deterministic implementation
    # warn_only=True logs warning from the FallbackKernel: torch.ops.aten.put_.default, instead of as UserWarning:
    # [W Context.cpp:%(lineno)] Warning: put_ does not have a deterministic implementation
    @skipIfTorchInductor("warning is logged from the FallbackKernel: torch.ops.aten.put_.default when warn_only=True")
    def test_nondeterministic_alert_put_accumulate(self, device):
        # [FUNCTION DOC] Calls Tensor.put and put_ with duplicate indices and accumulate on,
        # expecting the put_ nondeterministic alert only on CUDA. Skipped under inductor since the
        # warning is logged elsewhere in warn-only mode.
        a = torch.randn(10, device=device)
        indices = torch.tensor([0, 0], device=device)
        values = torch.tensor([0., 1.], device=device)

        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=True),
                'put_',
                torch.device(device).type == 'cuda')

    @dtypes(torch.float32)
    @dtypesIfCUDA(torch.float32, torch.int32)
    def test_nondeterministic_alert_histc(self, device, dtype):
        # [FUNCTION DOC] Calls torch.histc and Tensor.histc on an empty tensor; expects the
        # nondeterministic alert for the CUDA histc only when running on CUDA with a
        # floating dtype.
        a = torch.tensor([], device=device, dtype=dtype)
        for op_call in [torch.histc, torch.Tensor.histc]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, min=0, max=3),
                '_histc_cuda with floating point input',
                torch.device(device).type == 'cuda' and dtype.is_floating_point)

    def test_nondeterministic_alert_bincount(self, device):
        # [FUNCTION DOC] Calls torch.bincount and Tensor.bincount on an empty long tensor: with
        # weights an alert is expected only on CUDA, without weights no alert is expected.
        a = torch.tensor([], device=device, dtype=torch.long)
        weights = torch.tensor([], device=device)

        for op_call in [torch.bincount, torch.Tensor.bincount]:
            # Error should only be raised when device is CUDA and weights are
            # given
            self.check_nondeterministic_alert(
                lambda: op_call(a, weights),
                '_bincount_cuda',
                torch.device(device).type == 'cuda')

            self.check_nondeterministic_alert(
                lambda: op_call(a),
                '_bincount_cuda',
                False)

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_2d(self, device):
        # [FUNCTION DOC] 2D grid_sample backward must trigger the nondeterministic alert for the
        # CUDA grid_sampler_2d backward only on CUDA. Skipped under inductor.
        input = torch.empty(1, 1, 2, 2, device=device, requires_grad=True)
        grid = torch.empty(1, 1, 1, 2, device=device)
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    @skipIfRocm
    @onlyCUDA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_2d_cudnn(self, device):
        # [FUNCTION DOC] CUDA only with cuDNN available (not ROCm). Runs grid_sample with cuDNN
        # enabled and backward and expects the alert for the cuDNN grid sampler backward. Skipped
        # under inductor.
        def fn():
            # [FUNCTION DOC] HELPER: Create small input and grid tensors, run grid_sample with the
            # cuDNN backend enabled, and backpropagate a ones gradient.
            input = torch.empty(1, 1, 2, 2, device=device, requires_grad=True)
            grid = torch.empty(1, 1, 1, 2, device=device)
            with torch.backends.cudnn.flags(enabled=True):
                res = torch.nn.functional.grid_sample(input, grid, align_corners=True)
                res.backward(torch.ones_like(res))

        self.check_nondeterministic_alert(
            fn,
            'cudnn_grid_sampler_backward')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_3d(self, device):
        # [FUNCTION DOC] 3D grid_sample backward must trigger the nondeterministic alert for the
        # CUDA grid_sampler_3d backward only on CUDA. Skipped under inductor.
        input = torch.empty(1, 1, 2, 2, 2, device=device, requires_grad=True)
        grid = torch.empty(1, 1, 1, 2, 3, device=device)
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_3d_backward_cuda',
            torch.device(device).type == 'cuda')

    def test_invalid_shapes_grid_sampler(self, device):
        # [FUNCTION DOC] For a 3D and a 2D input/grid pair with mismatching batch sizes, checks
        # that grid_sampler, grid_sampler_2d, grid_sampler_3d, the CPU fallback and (on non-ROCm
        # CUDA with cuDNN) cudnn_grid_sampler all raise RuntimeError saying input and grid must
        # have the same batch size.
        make_arg = partial(
            make_tensor, device=device, dtype=torch.float64, requires_grad=True)

        inputs = (
            # input, grid
            ((5, 5, 5, 5, 5,), (1, 1, 1, 4, 4,)),  # 3d
            ((5, 5, 5, 5,), (1, 1, 4, 4,)),  # 2d
        )

        interpolation_mode = 0
        padding_mode = 0
        align_corners = True

        err = "expected grid and input to have same batch size"

        for input, grid in inputs:
            input = make_arg(input)
            grid = make_arg(grid, low=-1, high=1)

            # Wrapper for the 2d, 3d, and cuDNN functions listed below.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_2d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 3d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_3d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch._grid_sampler_2d_cpu_fallback(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input, on CUDA.
            # Doesn't work on CPU and ROCm.
            if device != 'cpu' and TEST_CUDNN and not TEST_WITH_ROCM:
                with self.assertRaisesRegex(RuntimeError, err):
                    torch.cudnn_grid_sampler(input, grid)

    def test_dist(self, device):
        # [FUNCTION DOC] Compares torch.dist with the norm of the difference for p in 0, 1, 2, 3,
        # 4, inf and -inf on random vectors and on a mostly-zero pair.
        def run_test(x, y):
            # [FUNCTION DOC] HELPER: For each tested p value, assert that the distance of x and y
            # equals the norm of their difference.
            for p in [0, 1, 2, 3, 4, inf, -inf]:
                dist_xy = torch.dist(x, y, p)
                dist_xy_norm = torch.norm(x - y, p)
                self.assertEqual(dist_xy, dist_xy_norm)

        run_test(torch.randn(5, device=device), torch.randn(5, device=device))

        x = torch.zeros(3, device=device)
        y = torch.zeros(3, device=device)
        y[1] = 1.
        run_test(x, y)

    # Ensures that median throws nondeterministic alerts in the correct cases
    @dtypes(torch.double)
    def test_nondeterministic_alert_median(self, device, dtype):
        # [FUNCTION DOC] Checks median alerts: calls with no dim (function and method) never
        # alert, calls with a dim or out indices alert only on CUDA (kernel message for median
        # with indices output).
        def test_func(call_type):
            # [FUNCTION DOC] HELPER: Invoke median on a fresh random vector in the way named by
            # call_type (function, function with indices, method, method with indices, out with
            # indices); fails the test for an unknown call type.
            S = 10
            a = torch.randn(S, device=device)
            if call_type == 'function':
                torch.median(a)
            elif call_type == 'function with indices':
                torch.median(a, 0)
            elif call_type == 'method':
                a.median()
            elif call_type == 'method with indices':
                a.median(0)
            elif call_type == 'out with indices':
                result = torch.empty_like(a)
                indices = torch.empty((), dtype=torch.long, device=device)
                torch.median(a, 0, out=(result, indices))
            else:
                self.fail(f"'{call_type}' is not a valid call type")

        def test_func_expect_error(call_type, should_error):
            # [FUNCTION DOC] HELPER: Run the median variant through check_nondeterministic_alert,
            # expecting an alert or not according to should_error.
            self.check_nondeterministic_alert(
                lambda: test_func(call_type),
                'median CUDA with indices output',
                should_error)

        is_cuda = torch.device(device).type == 'cuda'

        test_func_expect_error('function', False)
        test_func_expect_error('function with indices', is_cuda)
        test_func_expect_error('method', False)
        test_func_expect_error('method with indices', is_cuda)
        test_func_expect_error('out with indices', is_cuda)

    # FIXME: move to test_scatter_gather_ops
    def _test_gather_backward_one_dim(self, device, deterministic: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Shared helper for the gather backward tests. Under
        # DeterministicGuard (flag given), runs gather on a random vector with random indices,
        # backpropagates random weights, and on CUDA/MTIA checks the gradient is bitwise
        # repeatable over two reruns while on other devices it compares against a loop-accumulated
        # expected gradient exactly. Raises AssertionError if the gradient is missing.
        with DeterministicGuard(deterministic):
            m = random.randint(2000, 3000)
            elems = random.randint(10 * m, 20 * m)
            dim = 0
            src = torch.randn(m, device=device, requires_grad=True)
            idx = torch.randint(m, (elems,), device=device)
            res = torch.gather(src, dim, idx)
            weight = torch.rand_like(res, device=device) * 10 ** 6
            res.backward(weight)
            if src.grad is None:
                raise AssertionError("expected src.grad to be not None")
            grad = src.grad.detach().clone()

            if torch.device(device).type == 'cuda' or torch.device(device).type == 'mtia':
                for _ in range(2):
                    src.grad.data.zero_()
                    res = torch.gather(src, dim, idx)
                    res.backward(weight)
                    self.assertEqual(src.grad, grad, atol=0, rtol=0)
            else:
                expected = torch.zeros_like(src, device=device)
                for i in range(elems):
                    expected[idx[i]] += weight[i]
                self.assertEqual(grad, expected, atol=0, rtol=0)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_gather_backward_deterministic_path(self, device) -> None:
        # [FUNCTION DOC] On native devices, runs the gather backward helper with deterministic
        # mode enabled.
        self._test_gather_backward_one_dim(device, True)

    # FIXME: move to test_scatter_gather_ops
    @onlyCPU
    def test_gather_backward_one_dim(self, device) -> None:
        # [FUNCTION DOC] CPU only. Runs the gather backward helper with deterministic
        # mode disabled.
        self._test_gather_backward_one_dim(device, False)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.int)
    def test_scatter_add_one_dim_deterministic(self, device, dtype) -> None:
        # [FUNCTION DOC] Under DeterministicGuard, performs scatter_add of many random values into
        # a small vector five times, checking results are bitwise equal across runs, then compares
        # to a loop-accumulated reference within tolerance. Float and int dtypes.
        with DeterministicGuard(True):
            m = random.randint(20, 30)
            elems = random.randint(2000 * m, 3000 * m)
            dim = 0
            # initialize in a way that would produce
            # different ints
            src = (torch.rand(elems, device=device) * 10).to(dtype)
            idx = torch.randint(m, (elems,), device=device)

            x = torch.zeros(m, device=device, dtype=dtype)
            res = x.scatter_add(dim, idx, src)

            # Checking if scatter_add is deterministic
            for _ in range(5):
                res_next = x.scatter_add(dim, idx, src)
                self.assertEqual(res, res_next, atol=0, rtol=0)
                res = res_next

            expected = torch.zeros(m, device=device, dtype=dtype)
            for i in range(elems):
                expected[idx[i]] += src[i]

            self.assertEqual(res, expected, atol=1e-4, rtol=1e-5)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_scatter_zero_size_index(self, device) -> None:
        # [FUNCTION DOC] Checks that scatter with an empty (zero-row) index and source returns a
        # tensor equal to the original.
        null_index = torch.zeros((0, 4), dtype=torch.int64)
        null_arr = torch.zeros((0, 4))
        original = torch.arange(4, dtype=torch.float32)
        result = original.scatter(0, null_index, null_arr)
        self.assertEqual(result, original, atol=0, rtol=0)

    @onlyCUDA
    @skipIfTorchInductor("FIXME")
    def test_sync_warning(self, device):
        # [FUNCTION DOC] CUDA only. Uses CudaSyncGuard at levels 1 (warn) and 2 (error) with two
        # lists of callables on a random CUDA vector: operations expected not to synchronize
        # (index put/get with CUDA masks or indices, one_hot with num_classes, randperm,
        # repeat_interleave with output_size, any, combinations, normal) must run silently, and
        # operations that synchronize (boolean mask on CUDA, CPU index tensors, nonzero, boolean
        # conversion, one_hot without num_classes, repeat_interleave without output_size) must
        # warn at level 1 and raise RuntimeError at level 2. Skipped under inductor.

        def _sync_raises_helper(f, level):
            # [FUNCTION DOC] HELPER: Under CudaSyncGuard at the given level, call f and expect a
            # UserWarning (level 1) or RuntimeError (level 2) saying a synchronizing operation
            # was called.
            with CudaSyncGuard(level):
                if level == 1:
                    with self.assertWarnsRegex(UserWarning, "called a synchronizing "):
                        f()
                elif level == 2:
                    with self.assertRaisesRegex(RuntimeError, "called a synchronizing "):
                        f()

        def _no_sync_helper(f, level):
            # [FUNCTION DOC] HELPER: Call f under CudaSyncGuard at the given level without
            # expecting any warning or error.
            with CudaSyncGuard(level):
                f()

        def _ind_put_fn(x, ind, val):
            # [FUNCTION DOC] HELPER: Assign val into x at index ind in place and return x.
            x[ind] = val
            return x

        def _ind_get_fn(x, ind):
            # [FUNCTION DOC] HELPER: Return the elements of x selected by the index ind.
            return x[ind]

        def _cond_fn(x):
            # [FUNCTION DOC] HELPER: Return x if its boolean value is true, otherwise 2 times x;
            # converting a tensor to bool forces a CUDA sync.
            if x:  # taking boolean value of a tensor synchronizes
                return x
            else:
                return 2 * x

        # prepare inputs for subsequent ops
        size = 4
        x = torch.rand(size, device=device)
        y = torch.rand((), device=device)
        ind = torch.randint(size, (3,), device=device)
        ind_cpu = ind.cpu()
        repeats = torch.full((1,), 2, device=device)
        mask = torch.randint(2, (size,), device=device, dtype=bool)
        mask_cpu = mask.cpu()
        expect_no_sync = (lambda: _ind_put_fn(x, mask, 1.),
                          lambda: _ind_put_fn(x, mask_cpu, y),
                          lambda: _ind_put_fn(x, ind, y),
                          lambda: _ind_get_fn(x, mask_cpu),
                          lambda: _ind_get_fn(x, ind),
                          lambda: torch.nn.functional.one_hot(ind, num_classes=size),
                          lambda: torch.randperm(20000, device=device),
                          lambda: torch.repeat_interleave(x, 2, output_size=2 * size),
                          lambda: torch.repeat_interleave(x, repeats, output_size=2 * size),
                          lambda: torch.any(y),
                          lambda: torch.combinations(x, r=2),
                          lambda: torch.normal(x, x))
        expect_sync = (lambda: _ind_put_fn(x, mask, y),
                       lambda: _ind_put_fn(x, ind_cpu, y),
                       lambda: _ind_get_fn(x, mask),
                       lambda: _ind_get_fn(x, ind_cpu),
                       lambda: x.nonzero(),
                       lambda: _cond_fn(y),
                       lambda: torch.nn.functional.one_hot(ind),
                       lambda: torch.repeat_interleave(x, repeats))
        for f, level in product(expect_no_sync, (1, 2)):
            _no_sync_helper(f, level)
        for f, level in product(expect_sync, (1, 2)):
            _sync_raises_helper(f, level)


    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_log_normal(self, device, dtype):
        # [FUNCTION DOC] Calls log_normal_ on a one-element tensor and checks dtype and size
        # are preserved.
        a = torch.tensor([10], dtype=dtype, device=device).log_normal_()
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_geometric(self, device, dtype):
        # [FUNCTION DOC] Calls geometric_ with probability 0.5 on a one-element tensor and checks
        # dtype and size are preserved.
        a = torch.tensor([10], dtype=dtype, device=device).geometric_(0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

    def test_repeat_interleave(self, device):
        # [FUNCTION DOC] Checks repeat_interleave: the single-argument form flattens and doubles
        # (size 8), and the tensor-repeats form (int, long, plus small ints on CPU) gives shape 3
        # by 2 with the input dtype, with and without the output_size argument.
        y = torch.tensor([[1, 2], [3, 4]], device=device)
        # exercise single argument function signature
        temp = y.repeat_interleave(2)
        self.assertEqual(torch.Size([8]), temp.size())

        repeat_dtypes = [torch.int, torch.long]
        if device == "cpu":
            repeat_dtypes.extend([torch.int8, torch.uint8, torch.int16])

        for dtype in repeat_dtypes:
            lengths = torch.tensor([1, 2], dtype=dtype, device=device)
            output_size = torch.sum(lengths)
            a = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
            )
            self.assertEqual(a.dtype, y.dtype)
            self.assertEqual(a.size(), torch.Size([3, 2]))

            a_with_output = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
                output_size=output_size,
            )
            self.assertEqual(a_with_output.dtype, y.dtype)
            self.assertEqual(a_with_output.size(), torch.Size([3, 2]))

    @dtypes(*floating_types())
    @dtypesIfCPU(*floating_types_and(torch.bfloat16, torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half))
    def test_bernoulli_p(self, device, dtype):
        # [FUNCTION DOC] Checks bernoulli with a probability tensor: trivial probabilities of 0
        # and 1 are reproduced exactly; random probabilities (including an expanded tensor and the
        # out= form) yield only 0 or 1 values.
        for trivial_p in ([0, 1], [1, 0, 1, 1, 0, 1]):
            x = torch.tensor(trivial_p, dtype=dtype, device=device)
            self.assertEqual(x.bernoulli().tolist(), trivial_p)

        def isBinary(t):
            # [FUNCTION DOC] HELPER: Return True if the tensor contains only values 0 or 1 (no
            # element differs from both).
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        p = torch.rand(5, 5, dtype=dtype, device=device)
        self.assertTrue(isBinary(p.bernoulli()))

        p = torch.rand(5, dtype=dtype, device=device).expand(5, 5)
        self.assertTrue(isBinary(p.bernoulli()))

        p = torch.rand(5, 5, dtype=dtype, device=device)
        torch.bernoulli(torch.rand_like(p), out=p)
        self.assertTrue(isBinary(p))

    # RngUniform not implemented for Integral type in XLA test
    @dtypes(*floating_types())
    @dtypesIfCPU(*all_types_and(torch.bool, torch.half))
    @dtypesIfCUDA(*all_types_and(torch.bool, torch.half))
    def test_bernoulli_self(self, device, dtype):
        # [FUNCTION DOC] Checks in-place bernoulli_ on a 10 by 10 tensor starting filled with 2:
        # with a scalar probability, with an expanded probability tensor, via torch.bernoulli with
        # out=, and with a probability tensor of the same shape, for several probability dtypes;
        # each result must contain only 0 and 1.

        def isBinary(t):
            # [FUNCTION DOC] HELPER: Return True if the tensor contains only values 0 or 1 (no
            # element differs from both).
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        t = torch.empty(10, 10, dtype=dtype, device=device)

        t.fill_(2)
        t.bernoulli_(0.5)
        self.assertTrue(isBinary(t))

        for p_dtype in floating_types_and(*[torch.half] if device.startswith('cuda') else []):
            p = torch.rand(10, dtype=p_dtype, device=device).expand(10, 10)
            t.fill_(2)
            t.bernoulli_(p)
            self.assertTrue(isBinary(t))

            t.fill_(2)
            torch.bernoulli(torch.rand_like(t, dtype=p_dtype), out=t)
            self.assertTrue(isBinary(t))

            t.fill_(2)
            t.bernoulli_(torch.rand_like(t, dtype=p_dtype))
            self.assertTrue(isBinary(t))

    @slowTest
    @dtypes(*floating_types_and(torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half))
    def test_bernoulli_edge_cases(self, device, dtype):
        # [FUNCTION DOC] Slow test. Draws 100M Bernoulli samples with probability 0 and with
        # probability 1 and checks no ones appear in the first case and no zeros in the second.
        # Need to draw a lot of samples to cover every random floating point number.
        a = torch.zeros(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 0
        num_ones = (torch.bernoulli(a) == 1).sum()
        self.assertEqual(num_ones, 0)

        b = torch.ones(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 1
        num_zeros = (torch.bernoulli(b) == 0).sum()
        self.assertEqual(num_zeros, 0)

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_exponential(self, device, dtype):
        # [FUNCTION DOC] Checks exponential_: dtype and size preserved, rate infinity gives
        # exactly 0, and a negative rate raises RuntimeError.
        a = torch.tensor([10], dtype=dtype, device=device).exponential_(0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

        # Tests extremal behavior
        t = torch.empty((1,), device=device, dtype=dtype).exponential_(float('inf'))
        self.assertTrue(t.item() == 0)

        # Tests that negative lambda fails
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device=device, dtype=dtype).exponential_(-0.5)

    @onlyCUDA
    @dtypes(torch.half, torch.float)
    def test_exponential_no_zero(self, device, dtype):
        # [FUNCTION DOC] CUDA only. Draws 50M exponential samples in half and float and checks the
        # minimum is strictly positive (zeros must not be generated).
        # naively, 0 in exponential can be generated with probability 2^-24
        # so we need more samples to check if it's not generated
        # instead of doing one
        # don't test CPU, that would be a long test
        x = torch.empty(50000000, device=device, dtype=dtype).exponential_()
        self.assertTrue(x.min() > 0)

    def _generate_correlation_tensors(self, device, dtype):
        # [FUNCTION DOC] HELPER: Generator yielding input tensors for corrcoef/cov tests: empty
        # shapes, vectors, small matrices, a noncontiguous matrix, and (for non-int dtypes) a
        # vector containing NaN and inf.
        yield make_tensor((0, 0), dtype=dtype, device=device)
        yield make_tensor((1, 0), dtype=dtype, device=device)
        yield make_tensor((0, 1), dtype=dtype, device=device)
        yield make_tensor((2,), dtype=dtype, device=device)
        yield make_tensor((2, 1), dtype=dtype, device=device)
        yield make_tensor((2, 2), dtype=dtype, device=device)
        yield make_tensor((2, 3), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        if dtype != torch.int:
            yield torch.tensor([0, -2, nan, 10.2, inf], dtype=dtype, device=device)

    @tf32_on_and_off(0.005)
    @onlyNativeDeviceTypes
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_corrcoef(self, device, dtype):
        # [FUNCTION DOC] Compares torch.corrcoef with numpy corrcoef on the generated inputs (int,
        # float, complex float). For complex results the real part is always compared and the
        # imaginary part only if it contains finite values. Runs with TF32 tolerance handling.
        for x in self._generate_correlation_tensors(device, dtype):
            res = torch.corrcoef(x)
            ref = np.corrcoef(x.cpu().numpy())
            # complex cov can produce rounding errors, and when the norm factor is 0,
            # the imag portion of the answer could be +/- inf instead of nan, causing
            # the numpy answer to differ from the torch answer. We always check the real
            # part, but only check the imag part if it contains some normal values
            # (indicating that the norm factor was not 0).
            if res.is_complex():
                self.assertEqual(res.real, ref.real, atol=1e-05, rtol=1e-05, exact_dtype=False)
                if not (torch.isnan(res.imag) | torch.isinf(res.imag)).all():
                    self.assertEqual(res.imag, ref.imag, atol=1e-05, rtol=1e-05, exact_dtype=False)
            else:
                self.assertEqual(res, ref, atol=1e-05, rtol=1e-05, exact_dtype=False)

    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_cov(self, device, dtype):
        # [FUNCTION DOC] Compares torch.cov with numpy cov for generated correlation inputs (int,
        # float, complex float): first with defaults, then for non-empty observations with random
        # integer frequency weights and float analytic weights over corrections 0, 1, 2 and every
        # combination of weights present or absent.
        def check(t, correction=1, fweights=None, aweights=None):
            # [FUNCTION DOC] HELPER: Compute torch.cov with given correction and optional
            # frequency/analytic weights, compute numpy cov on CPU copies with the same options,
            # and assert they match (real part always, imaginary part only if it has finite values
            # for complex results).
            res = torch.cov(t, correction=correction, fweights=fweights, aweights=aweights)
            t = t.cpu().numpy()
            fweights = fweights.cpu().numpy() if fweights is not None else None
            aweights = aweights.cpu().numpy() if aweights is not None else None
            ref = np.cov(t, ddof=correction, fweights=fweights, aweights=aweights)
            # complex cov can produce rounding errors, and when the norm factor is 0,
            # the imag portion of the answer could be +/- inf instead of nan, causing
            # the numpy answer to differ from the torch answer. We always check the real
            # part, but only check the imag part if it contains some normal values
            # (indicating that the norm factor was not 0).
            if res.is_complex():
                self.assertEqual(res.real, ref.real, atol=1e-05, rtol=1e-05, exact_dtype=False)
                if not (torch.isnan(res.imag) | torch.isinf(res.imag)).all():
                    self.assertEqual(res.imag, ref.imag, atol=1e-05, rtol=1e-05, exact_dtype=False)
            else:
                self.assertEqual(res, ref, atol=1e-05, rtol=1e-05, exact_dtype=False)

        for x in self._generate_correlation_tensors(device, dtype):
            check(x)
            num_observations = x.numel() if x.ndim < 2 else x.size(1)
            if num_observations > 0:
                fweights = torch.randint(1, 10, (num_observations,), device=device)
                aweights = make_tensor((num_observations,), dtype=torch.float, device=device, low=1)
                for correction, fw, aw in product([0, 1, 2], [None, fweights], [None, aweights]):
                    check(x, correction, fw, aw)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_uniform_kstest(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. Samples uniform_ for several lower/upper bound pairs
        # and runs a Kolmogorov-Smirnov test against the uniform CDF, requiring the statistic
        # below 0.1.
        from scipy import stats
        size = 1000
        for from_ in [-42, 0, 4.2]:
            for to_ in [-4.2, 0, 42]:
                if to_ > from_:
                    t = torch.empty(size, dtype=dtype, device=device).uniform_(from_, to_)
                    res = stats.kstest(t.cpu().to(torch.double), stats.uniform(from_, (to_ - from_)).cdf)
                    self.assertTrue(res.statistic < 0.1)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_normal_kstest(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. Samples normal_ for several means and stds and runs a KS
        # test against the normal CDF, requiring statistic below 0.1.
        from scipy import stats
        size = 1000
        for mean in [-10, 0, 50]:
            for std in [1, 5, 10]:
                t = torch.empty(size, dtype=dtype, device=device).normal_(mean=mean, std=std)
                res = stats.kstest(t.cpu().to(torch.double), stats.norm(mean, std).cdf)
                self.assertTrue(res.statistic < 0.1)

    @skipIfNoSciPy
    @dtypes(torch.half, torch.bfloat16)
    def test_normal_small_std_kstest(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. For half and bfloat16 with small stds, draws 50000 normal
        # samples and requires the KS statistic against the normal CDF to be below 0.01.
        from scipy import stats
        size = 50000
        for std in [0.005, 0.01]:
            t = torch.empty(size, dtype=dtype, device=device).normal_(mean=0, std=std)
            res = stats.kstest(t.cpu().to(torch.double), stats.norm(0, std).cdf)
            self.assertTrue(
                res.statistic < 0.01,
                msg=lambda msg: f"{msg}\nKS statistic {res.statistic:.4f} for {dtype} std={std}",
            )

    @skipIfNoSciPy
    @dtypes(torch.half, torch.bfloat16)
    def test_normal_kurtosis(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. For half and bfloat16 and several stds draws 50000 normal
        # samples and requires the excess kurtosis to be within 0.15 of zero.
        from scipy import stats
        size = 50000
        for std in [0.01, 0.1, 1.0]:
            t = torch.empty(size, dtype=dtype, device=device).normal_(mean=0, std=std)
            kurtosis = stats.kurtosis(t.cpu().to(torch.double).numpy())
            self.assertAlmostEqual(
                kurtosis, 0.0, delta=0.15,
                msg=lambda msg: f"{msg}\nExcess kurtosis {kurtosis:.3f} too far from 0 for {dtype} std={std}",
            )

    @dtypes(torch.bfloat16)
    def test_normal_tail_reach(self, device, dtype):
        # [FUNCTION DOC] Draws one million bfloat16 normal samples and requires the maximum
        # absolute value exceed 3.5 sigma, guarding against tail truncation from a low-precision
        # Box-Muller implementation.
        # With the broken bf16 Box-Muller (256 uniform inputs), the maximum
        # output is capped at sqrt(-2*log(1/256)) ≈ 3.33 sigma. A correct
        # implementation generates in float and reaches 4+ sigma.
        size = 1000000
        t = torch.empty(size, dtype=dtype, device=device).normal_(mean=0, std=1.0)
        max_sigma = t.float().abs().max().item()
        self.assertGreater(
            max_sigma, 3.5,
            msg=lambda msg: f"{msg}\nMax |z| = {max_sigma:.2f} sigma, expected > 3.5 (tail truncation)",
        )

    @dtypes(torch.half, torch.bfloat16)
    def test_normal_unique_values(self, device, dtype):
        # [FUNCTION DOC] Draws 10000 half or bfloat16 normal samples with small std and requires
        # many distinct values (more than 100 for bfloat16, more than 50 for half), guarding
        # against quantized random inputs.
        size = 10000
        t = torch.empty(size, dtype=dtype, device=device).normal_(mean=0, std=0.01)
        n_unique = t.unique().numel()
        # bf16 has ~200 representable values in [-0.03, 0.03] (3-sigma range).
        # fp16 has ~60 representable values in [-0.03, 0.03].
        # A correct implementation generates in fp32 and casts, so we should
        # hit most representable values. The broken implementation produces
        # far fewer due to quantized Box-Muller inputs.
        if dtype == torch.bfloat16:
            self.assertGreater(n_unique, 100)
        else:
            self.assertGreater(n_unique, 50)

    @dtypes(torch.half, torch.bfloat16)
    def test_uniform_unique_values(self, device, dtype):
        # [FUNCTION DOC] Draws 10000 half or bfloat16 uniform samples in a wide range and requires
        # many unique values (above 1000 for bfloat16, above 4000 for half), guarding against
        # generating directly in low precision.
        size = 10000
        t = torch.empty(size, dtype=dtype, device=device).uniform_(-100, 100)
        n_unique = t.unique().numel()
        # With a correct implementation (generate in float, cast to dtype),
        # bf16 gives ~1400 unique and fp16 gives ~5200 for this range/size.
        # The broken implementation (generate directly in dtype) gives only
        # 256 for bf16 and ~2000 for fp16 due to using too few random bits.
        if dtype == torch.bfloat16:
            self.assertGreater(n_unique, 1000)
        else:
            self.assertGreater(n_unique, 4000)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_lognormal_kstest(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. Samples log_normal_ for several means and stds and runs a
        # KS test against the lognormal distribution; threshold is 0.3 for half and 0.1 otherwise.
        from scipy import stats
        size = 1000
        for mean in [-3, 0, 7]:
            for std in [1, 5, 7]:
                t = torch.empty(size, dtype=dtype, device=device).log_normal_(mean=mean, std=std)
                res = stats.kstest(t.cpu().to(torch.double), 'lognorm', args=(std, 0, math.exp(mean)))
                if dtype == torch.half:
                    self.assertTrue(res.statistic < 0.3)
                else:
                    self.assertTrue(res.statistic < 0.1)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_exponential_kstest(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. Samples exponential_ for several rates and runs a KS test
        # against the exponential distribution requiring statistic below 0.1.
        from scipy import stats
        size = 1000
        for lambd in [0.5, 1.0, 5.0]:
            t = torch.empty(size, dtype=dtype, device=device).exponential_(lambd=lambd)
            res = stats.kstest(t.cpu().to(torch.double), 'expon', args=(0, 1 / lambd,))
            self.assertTrue(res.statistic < 0.1)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy_kstest(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. Samples cauchy_ for several medians and sigmas and runs a
        # KS test against the Cauchy distribution requiring statistic below 0.1.
        from scipy import stats
        size = 1000
        for median in [-10, 0, 50]:
            for sigma in [0.5, 1.0, 10.0]:
                t = torch.empty(size, dtype=dtype, device=device).cauchy_(median=median, sigma=sigma)
                res = stats.kstest(t.cpu().to(torch.double), 'cauchy', args=(median, sigma))
                self.assertTrue(res.statistic < 0.1)

    @slowTest
    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float32)
    def test_cauchy_no_inf(self, device, dtype):
        # [FUNCTION DOC] Slow, CUDA only. Repeatedly draws 65536 Cauchy samples in bfloat16 and
        # float and checks none are infinite.
        # torch.float16 will have `inf` because of its smaller range.
        for _ in range((2**16) * 2):
            x = torch.empty((2**16), dtype=dtype, device=device)
            x.cauchy_()
            self.assertFalse(x.isinf().sum())

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy(self, device, dtype):
        # [FUNCTION DOC] Checks cauchy_ keeps dtype and size, that an infinite median yields
        # infinity, and that a zero sigma raises RuntimeError.
        a = torch.tensor([10], dtype=dtype, device=device).cauchy_(0.0, 0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

        # Tests extremal behavior
        t = torch.empty((1,), device=device, dtype=dtype).cauchy_(float('inf'), 0.5)
        self.assertTrue(t.item() == float('inf'))

        # Tests non-positive rate fails
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device=device, dtype=dtype).cauchy_(0.0, 0.0)

    @skipIfNoSciPy
    @skipIfTorchDynamo("FIXME: Skip due to dynamo failure on scipy.stats: https://github.com/pytorch/pytorch/issues/170509")
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_geometric_kstest(self, device, dtype):
        # [FUNCTION DOC] Requires SciPy. Samples geometric_ for several probabilities, bins the
        # values, and checks a chi-square test against the geometric pmf gives a p-value close to
        # 1.0. Skipped under Dynamo.
        from scipy import stats
        size = 1000
        for p in [0.2, 0.5, 0.8]:
            t = torch.empty(size, dtype=dtype, device=device).geometric_(p=p)
            actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
            expected = stats.geom(p).pmf(np.arange(1, 99)) * size
            res = stats.chisquare(actual, expected)
            self.assertEqual(res.pvalue, 1.0, atol=0.1, rtol=0)

    # FIXME: find test suite for pdist and cdist
    def test_pairwise_distance_empty(self, device):
        # [FUNCTION DOC] Checks pairwise_distance on empty inputs: shape (2,0) gives zeros of size
        # 2 (or 2 by 1 with keepdim) and shape (0,2) gives empty results.
        shape = (2, 0)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)

        self.assertEqual(torch.zeros(2, device=device), torch.pairwise_distance(x, y))
        self.assertEqual(torch.zeros((2, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

        shape = (0, 2)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)
        self.assertEqual(torch.zeros(0, device=device), torch.pairwise_distance(x, y))
        self.assertEqual(torch.zeros((0, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

    def test_pdist_empty(self, device):
        # [FUNCTION DOC] Checks pdist on empty or single-row inputs gives an empty result and on
        # shape (3,0) gives zeros of length 3.
        shape = (0, 2)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        shape = (1, 2)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        shape = (3, 0)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.zeros(3, device=device), torch.pdist(x))

    def test_cdist_empty(self, device):
        # [FUNCTION DOC] Checks cdist on empty inputs: zero rows give empty results of the
        # expected 2D shape, and zero feature columns give zeros for non-empty row sets.
        x = torch.randn((0, 5), device=device)
        y = torch.randn((4, 5), device=device)
        self.assertEqual(torch.empty(0, 4, device=device), torch.cdist(x, y))

        x = torch.randn((2, 5), device=device)
        y = torch.randn((0, 5), device=device)
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y))

        x = torch.randn((2, 0), device=device)
        y = torch.randn((3, 0), device=device)
        self.assertEqual(torch.zeros(2, 3, device=device), torch.cdist(x, y))

        x = torch.randn((2, 0), device=device)
        y = torch.randn((0, 0), device=device)
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y))

    def _brute_cdist(self, x, y, p=2):
        # [FUNCTION DOC] HELPER: Reference cdist: returns an empty tensor if either set has no
        # rows, otherwise the p-norm of pairwise differences computed by broadcasting.
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    def test_cdist_norm(self, device):
        # [FUNCTION DOC] Compares torch.cdist with the brute-force reference on random 2D inputs
        # over several row/column counts and p values; for p equal to 2 both Euclidean compute
        # modes are tested with a loose tolerance.
        for r1 in [3, 4, 5, 6]:
            for m in [2, 3, 4, 10]:
                for r2 in [4, 6, 7, 8]:
                    for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                        x = torch.randn(r1, m, device=device)
                        y = torch.randn(r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    def test_cdist_norm_batch(self, device):
        # [FUNCTION DOC] Same as the cdist norm test but with batched inputs of shape 2 by 3 by 6
        # rows by columns.
        for r1 in [3, 4, 5, 6]:
            for m in [2, 3, 4, 10]:
                for r2 in [4, 6, 7, 8]:
                    for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                        x = torch.randn(2, 3, 6, r1, m, device=device)
                        y = torch.randn(2, 3, 6, r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    @onlyCUDA
    def test_cdist_cuda_backward(self, device):
        # [FUNCTION DOC] CUDA only. Compares cdist backward gradients against the brute-force
        # reference for several sizes and p values (both Euclidean modes for p equal to 2).
        # KNOWN ISSUE: for p not equal to 2 the backward is never called on the cdist result or
        # the reference, so both gradients are None and the comparisons pass vacuously.
        for l1 in [1, 511, 513]:
            for l2 in [1, 511, 513]:
                for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                    x1 = torch.randn(4, l1, 32, device=device, requires_grad=True)
                    x2 = x1.clone().detach_().requires_grad_()
                    y1 = torch.randn(4, l2, 32, device=device, requires_grad=True)
                    y2 = y1.clone().detach_().requires_grad_()
                    if p == 2:
                        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                            z1 = torch.cdist(x1, y1, p=2, compute_mode=cm).mean()
                            z2 = self._brute_cdist(x2, y2, p=2).mean()
                            z1.backward()
                            z2.backward()
                            self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                            self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)
                    else:
                        z1 = torch.cdist(x1, y1, p=p).mean()
                        z2 = self._brute_cdist(x2, y2, p=p).mean()
                        self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                        self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)

    # On ROCm, AMD XF32 round-down bias in the K=10 mm-based cdist path
    # (||a-b||² via a·b) pushes max_abs to ~5.8e-3, exceeding 0.005; ideal
    # NVIDIA TF32 stays under. See https://github.com/jeffdaily/tf32_analysis.
    @tf32_on_and_off(0.01 if TEST_WITH_ROCM else 0.005)
    @reduced_f32_on_and_off(0.08)
    def test_cdist_large(self, device):
        # [FUNCTION DOC] Compares cdist for 1000 by 10 inputs against the reference in three
        # compute modes, with TF32/reduced-precision tolerance decorators.
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(1000, 10, device=device)
            y = torch.randn(1000, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    @slowTest
    @tf32_on_and_off(0.01)
    @reduced_f32_on_and_off(0.08)
    def test_cdist_large_batch(self, device):
        # [FUNCTION DOC] Slow. Same as the large cdist test but for batched 4 by 3 by 1000 by
        # 10 inputs.
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 1000, 10, device=device)
            y = torch.randn(4, 3, 1000, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    @tf32_on_and_off(0.005)
    @reduced_f32_on_and_off(0.04)
    def test_cdist_non_contiguous(self, device):
        # [FUNCTION DOC] Checks cdist on inputs where x, y or both are non-contiguous (transposed
        # views) in both Euclidean compute modes, asserting the contiguity of each input and
        # comparing to the reference.
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(5, 7, device=device).mT
            y = torch.randn(5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 5, device=device)
            y = torch.randn(5, 3, device=device).t()
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(5, 7, device=device).t()
            y = torch.randn(3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    @tf32_on_and_off(0.005)
    @reduced_f32_on_and_off(0.04)
    def test_cdist_non_contiguous_batch(self, device):
        # [FUNCTION DOC] Batched version of the non-contiguous cdist test, using transposed views
        # of 5D/4D/3D inputs.
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 2, 5, 7, device=device).mT
            y = torch.randn(4, 3, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 2, 7, 5, device=device)
            y = torch.randn(7, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(4, 5, 7, device=device).mT
            y = torch.randn(4, 3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    # Maybe merge into OpInfo?
    def test_cdist_euclidean_large(self, device):
        # [FUNCTION DOC] Runs a forward and backward cdist of a 2000 by 5 pair of float tensors to
        # check large inputs work; no value assertions.
        def _test_euclidean_large_cdist(sizex, sizey=None):
            # [FUNCTION DOC] HELPER: Create float inputs of the given sizes nudged away from ties,
            # require grad, compute the Euclidean cdist and backpropagate its sum.
            if sizey is None:
                sizey = sizex
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # to avoid extremum
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            dist = torch.cdist(x, y, p=2)
            # Do a backward pass to check that it is valid for large
            # matrices
            loss = dist.sum()
            loss.backward()

        _test_euclidean_large_cdist((2000, 5))

    # Ensure that cdist backward with p<1 does not produce NaNs
    def test_cdist_grad_p_lt_1_no_nan(self, device):
        # [FUNCTION DOC] For several p below 1, checks cdist backward on a single pair of points
        # yields no NaN gradients.
        for p in [0.99, 0.7, 0.5, 0.1, 0.01]:
            x = torch.randn(1, 2, device=device)
            y = x.detach().clone() + torch.tensor([[1., 0.]], device=device)
            x.requires_grad = True
            y.requires_grad = True
            result = torch.cdist(x, y, p=p)
            result.backward(torch.ones_like(result))
            self.assertFalse(torch.isnan(x.grad).any())
            self.assertFalse(torch.isnan(y.grad).any())

    def test_cdist_same_inputs(self, device):
        # [FUNCTION DOC] For several p values, computes cdist of a tensor with an identical copy
        # (all diagonal distances zero), backpropagates random gradients, and raises
        # AssertionError if x.grad has any non-finite value.
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            x.requires_grad = True
            d = torch.cdist(x, y, p=p)
            d.backward(dist_grad)
            # Check that the backward pass does not contain invalid
            # values such as nan or inf
            if not torch.isfinite(x.grad).all():
                raise AssertionError("expected x.grad to be finite")

    def test_cumsum(self, device):
        # [FUNCTION DOC] Checks cumsum: functional, out= and in-place forms agree on a 100 by 100
        # tensor; bool and byte inputs give equal integer results matching fixed expected matrices
        # along both dims; backward through cumsum over several shapes (including zero-length
        # dims) runs and the gradient has the input shape; and a scalar input works.
        x = torch.rand(100, 100, device=device)
        res1 = torch.cumsum(x, 1)
        res2 = torch.tensor([]).to(device)
        torch.cumsum(x, 1, out=res2)
        self.assertEqual(res1, res2)
        x.cumsum_(1)
        self.assertEqual(res1, x)

        a = torch.tensor([[True, False, True],
                          [False, False, False],
                          [True, True, True]], device=device)
        b = a.byte()
        aRes = torch.cumsum(a, 0)
        bRes = torch.cumsum(b, 0)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 0, 1],
                                             [1, 0, 1],
                                             [2, 1, 2]]))

        aRes = torch.cumsum(a, 1)
        bRes = torch.cumsum(b, 1)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 1, 2],
                                             [0, 0, 0],
                                             [1, 2, 3]]))

        # Check that cumulative sum over a zero length dimension doesn't crash on backprop.
        # Also check that cumsum over other dimensions in a tensor with a zero-length
        # dimensiuon also works
        # Also include a basic suite of similar tests for other bases cases.
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                integrated = raw_tensor.cumsum(dim=dim)
                # Check that backward does not crash
                integrated.sum().backward()
                # Check that output maintained correct shape
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        # Check a scalar example
        raw_tensor = torch.tensor(3., requires_grad=True)
        integrated = raw_tensor.cumsum(dim=-1)
        self.assertEqual(raw_tensor, integrated)
        # Check that backward does not crash
        integrated.sum().backward()
        # Check that output maintained correct shape
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

    def test_cumprod(self, device):
        # [FUNCTION DOC] Same coverage as the cumsum test but for cumprod (the out= comparison is
        # skipped under inductor), with expected matrices for bool/byte inputs and backward checks
        # on several shapes and a scalar.
        x = torch.rand(100, 100, device=device)
        res1 = torch.cumprod(x, 1)
        res2 = torch.tensor([]).to(device)
        if not TEST_WITH_TORCHINDUCTOR:
            torch.cumprod(x, 1, out=res2)
            self.assertEqual(res1, res2)
        x.cumprod_(1)
        self.assertEqual(res1, x)

        a = torch.tensor([[True, False, True],
                          [False, False, False],
                          [True, True, True]], dtype=torch.bool, device=device)
        b = a.byte()
        aRes = torch.cumprod(a, 0)
        bRes = torch.cumprod(b, 0)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 0, 1],
                                             [0, 0, 0],
                                             [0, 0, 0]]))

        aRes = torch.cumprod(a, 1)
        bRes = torch.cumprod(b, 1)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 0, 0],
                                             [0, 0, 0],
                                             [1, 1, 1]]))

        # Check that cumulative prod over a zero length dimension doesn't crash on backprop.
        # Also check that cumprod over other dimensions in a tensor with a zero-length
        # dimensiuon also works
        # Also include a basic suite of similar tests for other bases cases.
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                integrated = raw_tensor.cumprod(dim=dim)
                # Check that backward does not crash
                integrated.sum().backward()
                # Check that output maintained correct shape
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        # Check a scalar example
        raw_tensor = torch.tensor(3., requires_grad=True)
        integrated = raw_tensor.cumprod(dim=-1)
        self.assertEqual(raw_tensor, integrated)
        # Check that backward does not crash
        integrated.sum().backward()
        # Check that output maintained correct shape
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

    def test_cummax_cummin(self, device):
        # [FUNCTION DOC] Checks cummax and cummin via a nested driver run for each op: out=
        # matches functional results, bool and byte inputs agree and match expected values,
        # inf/nan inputs match an expected vector, out tensors with the wrong dtype raise
        # RuntimeError, invalid dims on scalar or empty tensors raise IndexError, backward
        # on several shapes preserves gradient shape, and complex inputs raise RuntimeError
        # not implemented.
        def test_ops(op, string_of_function_name, expected_output1, expected_output2):
            # [FUNCTION DOC] HELPER: Driver run for one cumulative max/min op (passed together
            # with its method name and two expected outputs): checks out= consistency, bool/byte
            # inputs, inf and nan handling, mismatched out dtype error, scalar and empty dim range
            # errors, and backward shape on multiple shapes.
            x = torch.rand(100, 100, device=device)
            out1 = op(x, 1)
            res2 = torch.empty(0, device=device)
            indices2 = torch.empty(0, dtype=torch.int64, device=device)
            op(x, 1, out=(res2, indices2))
            self.assertEqual(out1[0], res2)
            self.assertEqual(out1[1], indices2)

            a = torch.tensor([[True, False, True],
                              [False, False, False],
                              [True, True, True]], dtype=torch.bool, device=device)
            b = a.byte()
            aRes = op(a, 0)
            bRes = op(b, 0)
            self.assertEqual(aRes[0], bRes[0].bool())
            self.assertEqual(aRes[0], expected_output1.bool())

            # test inf and nan input
            x = torch.tensor([4, inf, 1.5, -inf, 0, nan, 1])
            xRes = op(x, 0)[0]
            self.assertEqual(xRes, expected_output2)

            # op shouldn't support values, indices with a dtype, device type or layout
            # different from that of input tensor
            t = torch.randn(10)
            values = torch.empty(0, dtype=torch.int16)
            indices = torch.empty(0, dtype=torch.int64)
            with self.assertRaisesRegex(
                    RuntimeError,
                    'expected scalar_type Float but found Short'):
                op(t, 0, out=(values, indices))

            # Range-check 0-d tensors
            x = torch.rand([])
            dim = 100
            with self.assertRaisesRegex(
                    IndexError,
                    'Expected reduction dim -1 or 0 for scalar but got 100'):
                op(x, dim)

            # Check that zero-size tensors with invalid dims raise IndexError
            # consistently, matching non-zero tensor behavior
            x = torch.tensor([], device=device)
            with self.assertRaisesRegex(
                    IndexError,
                    'Dimension out of range'):
                op(x, -46)

            # Check that op over a zero length dimension doesn't crash on backprop.
            # Also check that op over other dimensions in a tensor with a zero-length
            # dimension also works
            # Also include a basic suite of similar tests for other bases cases.
            shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
            for shape in shapes:
                for dim in range(len(shape)):
                    raw_tensor = torch.zeros(*shape, requires_grad=True)
                    integrated = getattr(raw_tensor, string_of_function_name)(dim=dim)
                    # Check that backward does not crash
                    integrated[0].sum().backward()
                    # Check that output maintained correct shape
                    self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

            # Check a scalar example
            raw_tensor = torch.tensor(3., requires_grad=True)
            integrated = getattr(raw_tensor, string_of_function_name)(dim=-1)
            # Check that backward does not crash
            integrated[0].sum().backward()
            # Check that output maintained correct shape
            self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        expected_out = torch.tensor([4, inf, inf, inf, inf, nan, nan])
        test_ops(torch.cummax, "cummax", torch.tensor([[1, 0, 1],
                                                       [1, 0, 1],
                                                       [1, 1, 1]]), expected_out)

        expected_out = torch.tensor([4, 4, 1.5, -inf, -inf, nan, nan])
        test_ops(torch.cummin, "cummin", torch.tensor([[1, 0, 1],
                                                       [0, 0, 0],
                                                       [0, 0, 0]]), expected_out)

        for op in [torch.cummax, torch.cummin]:
            x = torch.randn(5, dtype=torch.complex64, device=device)
            with self.assertRaisesRegex(RuntimeError, "not implemented for 'ComplexFloat'"):
                op(x, 0)

    def test_logcumsumexp(self, device):
        # [FUNCTION DOC] Compares logcumsumexp with a reference (log of cumsum of exp) on random
        # data, on a vector containing -inf, inf and nan (also expanded to 2D), checks the out=
        # form writes into the given tensor, and that a float32 out with a float64 input raises
        # RuntimeError about scalar types.
        def logcumsumexp(a, axis):
            # [FUNCTION DOC] HELPER: Reference logcumsumexp: takes the exponential of a,
            # cumulative sum along axis, then the in-place log.
            return torch.cumsum(a.exp(), axis=axis).log_()

        axis = -1
        a = torch.randn(100, 100, device=device)

        actual = a.logcumsumexp(axis)
        expected = logcumsumexp(a, axis)
        self.assertEqual(a.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected, actual)

        # check -inf and nan handling
        x = torch.tensor([-float('inf'), -float('inf'), 1.0, 1.0, float('inf'),
                         float('inf'), float('nan'), 1.0, 1.0], device=device)
        x2d = x.unsqueeze(0).expand(2, -1)

        for inp in (x, x2d):
            actual = inp.logcumsumexp(axis)
            expected = logcumsumexp(inp, axis)
            self.assertEqual(expected, actual)

        # Check that out is actually inplace
        b = torch.randn(5, 2, device=device)
        inplace_out = torch.zeros(5, 2, device=device)

        expected = logcumsumexp(b, axis)
        torch.logcumsumexp(b, axis=axis, out=inplace_out)

        self.assertEqual(inplace_out, expected)

        # Check input and inplace_output type mismatch
        b = torch.randn(5, 2, device=device, dtype=torch.float64)
        inplace_out = torch.zeros(5, 2, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
                RuntimeError,
                'expected scalar_type Double but found Float'):
            torch.logcumsumexp(b, axis, out=inplace_out)

    def _test_diff_numpy(self, t, dims=None):
        # [FUNCTION DOC] HELPER: Compare torch.diff with numpy diff on a tensor for each dim (or
        # the given dims): without prepend/append for each order up to the size, with prepend and
        # append of length 1 for larger orders, and with the tensor itself prepended and appended.
        # bfloat16 is converted to float for numpy.
        # Helper for test_diff to compare with NumPy reference implementation
        def to_np(t):
            # [FUNCTION DOC] HELPER: Convert a tensor to a numpy array on CPU, going through float
            # for bfloat16.
            if t.dtype == torch.bfloat16:
                return t.to(dtype=torch.float, device="cpu").numpy()
            else:
                return t.cpu().numpy()

        for dim in dims if dims else range(t.dim()):
            prepend = t.narrow(dim, 0, 1)
            append = t.narrow(dim, 0, 1)
            np_t = to_np(t)

            # test when no prepend and append
            for n in range(t.size(dim)):
                actual = torch.diff(t, dim=dim, n=n)
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n))
                self.assertEqual(actual, expected.to(t.dtype))

            # test when prepend and append's size along dim is 1
            for n in range(1, t.size(dim) + 4):
                actual = torch.diff(t, dim=dim, n=n, prepend=prepend, append=append)
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n, prepend=to_np(prepend), append=to_np(append)))
                self.assertEqual(actual, expected.to(t.dtype))

            # test when prepend and append's size along dim != 1
            for n in range(1, t.size(dim) * 3):
                actual = torch.diff(t, dim=dim, n=n, prepend=t, append=t)
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n, prepend=np_t, append=np_t))
                self.assertEqual(actual, expected.to(t.dtype))

    # All tensors appear contiguous on XLA
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    def test_diff_noncontig(self, device, dtype):
        # [FUNCTION DOC] Native devices only. For several shapes, builds a non-contiguous tensor
        # by selecting from a larger tensor, copies in values and runs the numpy diff comparison
        # helper on it.
        shapes = (
            (1,),
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5))

        for shape in shapes:
            contig = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)

            non_contig = torch.empty(shape + (2, 2), device=device, dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertTrue(not non_contig.is_contiguous() or shape == (1,))

            self._test_diff_numpy(non_contig)

    # RngNormal not implemented for type f16 for XLA
    @dtypes(*all_types_and_complex_and(torch.bool))
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool))
    def test_diff(self, device, dtype):
        # [FUNCTION DOC] Compares diff with numpy on contiguous tensors of several shapes, and
        # checks the RuntimeError messages for prepend with wrong dimensionality, prepend with
        # mismatched shape, and a zero-dimensional input.
        shapes = (
            (1,),
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5))

        for shape in shapes:
            contig = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            self._test_diff_numpy(contig)

        t = torch.ones(2, 3)

        with self.assertRaisesRegex(
                RuntimeError, 'diff expects prepend or append to be the same dimension as input'):
            invalid_prepend = torch.tensor([1, 2, 3], device=device, dtype=dtype)
            t.diff(dim=0, prepend=invalid_prepend)

        with self.assertRaisesRegex(
                RuntimeError, 'diff expects the shape of tensor to prepend or append to match that of input'):
            invalid_prepend = torch.tensor([[0, 1]], device=device, dtype=dtype)
            t.diff(dim=0, prepend=invalid_prepend)

        with self.assertRaisesRegex(
                RuntimeError, 'diff expects input to be at least one-dimensional'):
            scalar = torch.tensor(2, device=device, dtype=dtype)
            torch.diff(scalar)

    # if the given input arg is not a list, it returns a list of single element: [arg]
    def _wrap_to_list(self, input_array):
        # [FUNCTION DOC] HELPER: Return the argument as a list if it is a list or tuple, otherwise
        # a one-element list containing it.
        return list(input_array) if isinstance(input_array, (list, tuple)) else [input_array]

    # To ensure inf, -inf, and nan values do not cause divergence between Numpy and PyTorch.
    # There are two types of possible divergence:
    # 1. When we compute a,b both real numbers and has very small absolute values (i.e. very near to 0.0)
    # then, result of a/b be inf, -inf and nan, and this cause divergence.
    # 2. When we are dividing complex numbers by zero. For example, when a = torch.tensor(3+5j) we have
    # a/0 to be equal to nan + nan*j in PyTorch and inf + inf*j in Numpy.
    def _inf_nan_preprocess(self, actual, expected):
        # [FUNCTION DOC] HELPER: In-place normalize results for comparison: converts inf, -inf
        # and nan in each expected numpy array to nan, and does the same to the corresponding
        # actual tensors (real and imaginary parts separately for complex64). Returns the pair
        # (actual, expected).
        for i in range(len(expected)):
            expected[i] = np.nan_to_num(expected[i], nan=nan, posinf=nan, neginf=nan)
            # nan_to_num is not defined for complex tensors in PyTorch.
            if actual[i].dtype == torch.complex64 :
                actual[i].real = torch.nan_to_num(actual[i].real, nan=nan, posinf=nan, neginf=nan)
                actual[i].imag = torch.nan_to_num(actual[i].imag, nan=nan, posinf=nan, neginf=nan)
            else:
                actual[i] = torch.nan_to_num(actual[i], nan=nan, posinf=nan, neginf=nan)

        return actual, expected

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_gradient_all(self, device, dtype):
        # [FUNCTION DOC] Compares torch.gradient with numpy gradient over many shape and dimension
        # tuples, contiguous and noncontiguous inputs, edge orders 1 and 2, and three spacing
        # styles (scalar, list of scalars, list of coordinate tensors), after normalizing inf/nan.
        def create_scalar(shape):
            # [FUNCTION DOC] HELPER: Return one random scalar of the test dtype (at least 1).
            return make_tensor((1,), device='cpu', dtype=dtype, low=1.).item()

        def create_list(shape):
            # [FUNCTION DOC] HELPER: Return a list of random scalars of the test dtype, one per
            # dimension of the shape.
            return make_tensor((len(shape),), device='cpu', dtype=dtype, low=1.).tolist()

        def create_coordinate_tensors(shape):
            # [FUNCTION DOC] HELPER: Return a list of random 1D coordinate tensors on the device,
            # one per entry of the shape.
            tensor_list = []
            for i in range(len(shape)):
                tensor_list.append(make_tensor((shape[i],), device=device, dtype=dtype))
            return tensor_list

        def filter_shape(shape, dim):
            # [FUNCTION DOC] HELPER: Return the sizes of the shape selected by the given dimension
            # indices, in order.
            filtered_shape = []
            for i in range(len(dim)):
                filtered_shape.append(shape[dim[i]])
            return filtered_shape

        # shape, dims format
        test_cases = (
            ((5,), (0,)),
            ((4, 4), (0, 1)),
            ((3, 3, 3), (-1, 0)),
            ((4, 4, 4), (2,)),
            ((4, 4, 4), (0, 1)),
            ((4, 4, 4, 3), (0, 2, 3)),
            ((4, 5, 3, 4, 3), (1, 2)),
            ((4, 3, 6, 5, 3), (2, 4)),
            ((4, 3, 3, 5, 3), (0, 1, 2, 3, 4)),
            ((1, 3, 3), (1, 2)),
            ((1, 5), (1,)),
        )

        for case, contig, edge_order, space_fn in product(test_cases, [True, False], [1, 2],
                                                          (create_scalar, create_list, create_coordinate_tensors)):
            shape, dims = case
            # filter shape by dims before passing filtered shape to create_* functions
            filtered_shape = filter_shape(shape, dims)

            spacing = space_fn(filtered_shape)
            t = make_tensor(shape, device=device, dtype=dtype, noncontiguous=not contig)
            t_np = t.cpu().numpy()

            actual = torch.gradient(t, spacing=spacing, dim=dims, edge_order=edge_order)
            if space_fn is create_coordinate_tensors and spacing[0].device != 'cpu':
                spacing = [space.cpu().detach().numpy() for space in spacing]
            expected = np.gradient(t_np, *self._wrap_to_list(spacing), axis=dims, edge_order=edge_order)
            actual, expected = self._inf_nan_preprocess(list(actual), self._wrap_to_list(expected))
            self.assertEqual(actual, expected, equal_nan=True, atol=1e-4, rtol=0, exact_dtype=False)

    @onlyNativeDeviceTypes
    @slowTestIf(TEST_WITH_TORCHINDUCTOR)
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_gradient_extreme_cases(self, device, dtype):
        # [FUNCTION DOC] Checks gradient on a vector with inf and nan against numpy, and on a
        # 100000-element tensor with random coordinates for edge orders 1 and 2.
        # Test behaviour for inf and nan values
        actual = torch.gradient(torch.tensor([2, -2, inf, inf, -inf, -inf, inf, 3, -inf, 2, nan, nan, 3, inf, nan]))
        expected = np.gradient(np.array([2, -2, inf, inf, -inf, -inf, inf, 3, -inf, 2, nan, nan, 3, inf, nan]))
        self.assertEqual(actual, self._wrap_to_list(expected), exact_dtype=False)

        # Test behaviour in very big tensors
        large_size = 100000
        t = make_tensor((large_size,), dtype=dtype, device=device)
        t_np = t.cpu().numpy()
        coordinates_np = np.random.randn(large_size)
        coordinates = [torch.tensor(coordinates_np, device=device)]
        actual = torch.gradient(t, spacing=coordinates, dim=0, edge_order=1)
        expected = [np.gradient(t_np, coordinates_np, axis=0, edge_order=1)]
        self.assertEqual(actual, expected, exact_dtype=False)

        actual = torch.gradient(t, spacing=coordinates, dim=0, edge_order=2)
        expected = [np.gradient(t_np, coordinates_np, axis=0, edge_order=2)]
        self.assertEqual(actual, expected, exact_dtype=False)

    @onlyNativeDeviceTypes
    def test_gradient_type_promotion(self, device):
        # [FUNCTION DOC] Checks torch.gradient against numpy for float, complex and int inputs
        # combined with many spacing types (scalar and list of float, int, complex, and coordinate
        # tensors) and both edge orders; real inputs with complex spacing are compared on the real
        # part and the imaginary part is expected to be zero.
        inputs = (
            make_tensor((4, 4), device=device, dtype=torch.float32),
            make_tensor((4, 4), device=device, dtype=torch.complex64),
            make_tensor((4, 4), device=device, dtype=torch.int64),
        )

        spacing = (
            make_tensor((1,), device='cpu', dtype=torch.float32).item(),
            make_tensor((1,), device='cpu', dtype=torch.int64).item(),
            make_tensor((1,), device='cpu', dtype=torch.complex64).item(),
            make_tensor((2,), device='cpu', dtype=torch.float32, low=0.1).tolist(),
            make_tensor((2,), device='cpu', dtype=torch.int64, low=1).tolist(),
            make_tensor((2,), device='cpu', dtype=torch.complex64).tolist(),
            [make_tensor((4,), device=device, dtype=torch.float32),
             make_tensor((4,), device=device, dtype=torch.float32)],
            [make_tensor((4,), device=device, dtype=torch.int64),
             make_tensor((4,), device=device, dtype=torch.int64)],
            [make_tensor((4,), device=device, dtype=torch.complex64),
             make_tensor((4,), device=device, dtype=torch.complex64)],
        )

        for input, spacing_or_coord, edge_order in product(inputs, spacing, [1, 2]):
            input_np = input.cpu().numpy()
            input_np = input.cpu().numpy()
            actual = torch.gradient(input, spacing=spacing_or_coord, dim=(0, 1), edge_order=edge_order)
            spacing_or_coord_wrapped = self._wrap_to_list(spacing_or_coord)
            spacing_or_coord_np = []
            if torch.is_tensor(spacing_or_coord_wrapped[0]) and torch.device(spacing_or_coord_wrapped[0].device).type != 'cpu':
                for i in range(len(spacing_or_coord_wrapped)):
                    spacing_or_coord_np.append(spacing_or_coord_wrapped[i].detach().clone().cpu().numpy())
            else:
                spacing_or_coord_np = spacing_or_coord_wrapped
            expected = np.gradient(input_np, *spacing_or_coord_np, axis=(0, 1), edge_order=edge_order)
            if actual[0].dtype == torch.complex64 and input.dtype != torch.complex64:
                for i in range(len(actual)):
                    self.assertEqual(actual[i].real, expected[i].real, exact_dtype=False)
                    # Type promotion fails on Numpy when spacing is given as complex number and input is given as real.
                    # Result is given just as real number and all the imaginary parts to be equal to zero.
                    self.assertEqual(expected[i].imag, torch.zeros(actual[i].shape), exact_dtype=False)
            else:
                actual, expected = self._inf_nan_preprocess(list(actual), list(expected))
                self.assertEqual(actual, expected, equal_nan=True, exact_dtype=False)

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_gradient_spacing_list_length_error(self, device, dtype):
        # [FUNCTION DOC] Checks that torch.gradient raises RuntimeError mentioning expected
        # spacing when the number of spacing entries (tensors or scalars) is neither 1 for all
        # dims nor equal to the number of dims being differentiated (tested with 1 or 3 entries
        # failing and 2 passing).
        t = make_tensor((2, 2), device=device, dtype=dtype)

        spacing = (make_tensor((2,), device=device, dtype=dtype),)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        spacing = (make_tensor((2,), device=device, dtype=dtype),) * 2
        torch.gradient(t, spacing=spacing)

        spacing = (make_tensor((2,), device=device, dtype=dtype),) * 3
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        spacing = (2,)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        spacing = (2, 2)
        torch.gradient(t, spacing=spacing)

        spacing = (2, 2, 2)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

    def _test_large_cum_fn_helper(self, x, fn):
        # [FUNCTION DOC] HELPER: Compare a cumulative function on the CPU float copy with the
        # result on the device using assert_close (not assertEqual, to save memory).
        expected = fn(x.cpu().float())
        actual = fn(x).cpu().float()
        # Avoid self.assertEqual to save memory.
        torch.testing.assert_close(expected, actual)

    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @unittest.skipIf(IS_JETSON, "psutil issue for largeTensorTest. Too large for Jetson.")
    @onlyCUDA
    @dtypes(torch.half)  # only small dtype not to get oom
    @largeTensorTest('25GB', device='cpu')
    @largeTensorTest('4GB', device='cuda')
    def test_large_cumsum(self, device, dtype):
        # [FUNCTION DOC] CUDA only, large (25GB host, 4GB device). Builds a half tensor of
        # 2**30 plus 200 elements with a repeating pattern and compares cumsum against the CPU
        # float result.
        # initialization to avoid overflow and half caveats
        x = torch.empty(2**30 + 200, device=device, dtype=dtype)
        x[::3] = -3
        x[1::3] = 2
        x[2::3] = 1
        self._test_large_cum_fn_helper(x, lambda x: torch.cumsum(x, 0))

    @onlyCUDA
    @dtypes(torch.half)  # only small dtype not to get oom
    @largeTensorTest('25GB', device='cpu')
    @largeTensorTest('4GB', device='cuda')
    @unittest.skipIf(IS_JETSON, "psutil issue for largeTensorTest. Too large for Jetson.")
    def test_large_cumprod(self, device, dtype):
        # [FUNCTION DOC] CUDA only, large. Builds a half tensor of 2**30 plus 200 elements with
        # factors 8, 0.25 and 0.5 repeating and compares cumprod against the CPU float result.
        # initialization to avoid overflow and half caveats
        x = torch.empty(2**30 + 200, device=device, dtype=dtype)
        x[::3] = 8
        x[1::3] = .25
        x[2::3] = .5
        self._test_large_cum_fn_helper(x, lambda x: torch.cumprod(x, 0))

    @skipIfTorchDynamo("Torchdynamo fails with unknown reason")
    def test_discontiguous_out_cumsum(self, device):
        # [FUNCTION DOC] Checks cumsum with an out= tensor that is non-contiguous gives the same
        # result as the functional form and keeps its non-contiguity. Skipped under Dynamo.
        x = torch.randn(4, 8, device=device)
        y = torch.empty(4, 16, device=device)[:, ::2]
        out = torch.cumsum(x, 0)
        torch.cumsum(x, 0, out=y)
        self.assertFalse(y.is_contiguous())
        self.assertEqual(out, y, atol=0., rtol=0.)

    def _test_cumminmax_helper(self, x, fn, expected_val, expected_ind):
        # [FUNCTION DOC] HELPER: Check a cummax/cummin function along the last dim against
        # expected values and indices exactly, then call it with non-contiguous out tensors and
        # check they stay non-contiguous (except under inductor) and hold the expected values.
        val, ind = fn(x, -1)
        self.assertEqual(val, expected_val, atol=0, rtol=0)
        self.assertEqual(ind, expected_ind, atol=0, rtol=0)
        out_val = torch.empty_like(val).t().contiguous().t()
        out_ind = torch.empty_like(ind).t().contiguous().t()
        fn(x, -1, out=(out_val, out_ind))
        # TODO: Fix this. It reproduces with aot_eager too, and looks like a functionalization bug.
        # (the problematic case seems rare, as we're calling an out= op directly from user code,
        # where the passed-in out tensors are non-contiguous).
        if not TEST_WITH_TORCHINDUCTOR:
            self.assertFalse(out_val.is_contiguous())
            self.assertFalse(out_ind.is_contiguous())
        self.assertEqual(out_val, expected_val, atol=0, rtol=0)
        self.assertEqual(out_ind, expected_ind, atol=0, rtol=0)

    def test_cummax_discontiguous(self, device):
        # [FUNCTION DOC] Checks cummax on a non-contiguous float input against fixed expected
        # values and indices using the shared helper.
        x = torch.tensor([[0, 1, 2, 3, 2, 1], [4, 5, 6, 5, 6, 7]], device=device, dtype=torch.float).t().contiguous().t()
        expected_val = torch.tensor([[0, 1, 2, 3, 3, 3], [4, 5, 6, 6, 6, 7]], device=device, dtype=torch.float)
        expected_ind = torch.tensor([[0, 1, 2, 3, 3, 3], [0, 1, 2, 2, 4, 5]], device=device, dtype=torch.long)
        self._test_cumminmax_helper(x, torch.cummax, expected_val, expected_ind)

    def test_cummin_discontiguous(self, device):
        # [FUNCTION DOC] Checks cummin on a non-contiguous float input against fixed expected
        # values and indices using the shared helper.
        x = torch.tensor([[3, 2, 1, 0, 1, 2], [7, 6, 5, 4, 5, 2]], device=device, dtype=torch.float).t().contiguous().t()
        expected_val = torch.tensor([[3, 2, 1, 0, 0, 0], [7, 6, 5, 4, 4, 2]], device=device, dtype=torch.float)
        expected_ind = torch.tensor([[0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 3, 5]], device=device, dtype=torch.long)
        self._test_cumminmax_helper(x, torch.cummin, expected_val, expected_ind)

    def test_bool_tensor_value_change(self, device):
        # [FUNCTION DOC] Sets elements of a bool tensor through item assignment and checks the
        # stored values.
        x = torch.tensor([True, False], dtype=torch.bool, device=device)
        x[0] = False
        x[1] = True
        self.assertEqual(x, torch.tensor([False, True], dtype=torch.bool, device=device))

    # FIXME: move to data movement test suite
    def test_copy_all_dtypes_and_devices(self, device):
        # [FUNCTION DOC] For all dtypes, makes a shallow copy of a tensor, fills the copy with
        # ones and checks the original sees the change.
        from copy import copy
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            x = torch.tensor([1, 2, 3, 4], dtype=dt, device=device)
            _x_clone = x.clone()
            y = copy(x)
            y.fill_(1)
            # copy is a shallow copy, only copies the tensor view,
            # not the data
            self.assertEqual(x, y)

    @onlyCPU
    def test_bfloat16_neg_abs(self, device):
        # [FUNCTION DOC] CPU only. Checks neg and abs on bfloat16 inputs (including nan, -nan,
        # inf, -inf) equal the bfloat16 conversion of the float results.
        src = torch.randn(256)
        src[0] = torch.nan
        src[1] = -torch.nan
        src[2] = torch.inf
        src[3] = -torch.inf
        src_bf16 = src.bfloat16()
        self.assertEqual(src.neg().bfloat16(), src_bf16.neg())
        self.assertEqual(src.abs().bfloat16(), src_bf16.abs())

    @onlyCPU
    @dtypes(torch.bfloat16, torch.half)
    def test_reduced_type_float_copy(self, device, dtype):
        # [FUNCTION DOC] CPU only. For bfloat16 and half and several shapes, checks converting
        # float to the reduced type and back (also from a strided slice) equals the original
        # within default tolerance and round-trips exactly.
        for shape in [(20, 7), (249, 137), (1029, 917), (1, 7, 19, 17), (3, 77, 1091)]:
            input = torch.randn(shape, dtype=torch.float, device=device)
            out1 = input.to(dtype=dtype)
            self.assertEqual(input, out1, atol=None, rtol=None, exact_dtype=False)
            out2 = out1.to(torch.float)
            self.assertEqual(out2, out1, atol=0, rtol=0, exact_dtype=False)

            input_s = input[..., ::2, :]
            out1 = input_s.to(dtype=dtype)
            self.assertEqual(input_s, out1, atol=None, rtol=None, exact_dtype=False)
            out2 = out1.to(torch.float)
            self.assertEqual(out2, out1, atol=0, rtol=0, exact_dtype=False)

    # FIXME: move to data movement test suite
    @onlyNativeDeviceTypes
    def test_copy_math_view(self, device):
        # [FUNCTION DOC] Checks copy_ with negative-view sources/destinations (and conjugate views
        # for complex) across dtype pairs gives the expected negated or conjugated values,
        # including the self-copy case from issue 106051.
        for dst_dtype, src_dtype in [
                (torch.float32, torch.float32),
                (torch.float64, torch.float32),
                (torch.int64, torch.int32),
                (torch.complex128, torch.complex64),
        ]:
            src = make_tensor((100,), dtype=src_dtype, device=device)
            dst = torch.empty(100, dtype=dst_dtype, device=device)

            dst.copy_(src)
            self.assertEqual(dst, src, exact_dtype=False)

            dst.copy_(src._neg_view())
            self.assertEqual(dst, src.neg(), exact_dtype=False)

            dst._neg_view().copy_(torch._neg_view(src))
            self.assertEqual(dst, src, exact_dtype=False)

            dst._neg_view().copy_(src)
            self.assertEqual(dst, src.neg(), exact_dtype=False)

            # issue: https://github.com/pytorch/pytorch/issues/106051
            dst._neg_view().copy_(dst)
            self.assertEqual(dst, src, exact_dtype=False)

        for dst_dtype, src_dtype in [
                (torch.complex64, torch.complex64),
                (torch.complex128, torch.complex64),
        ]:
            src = make_tensor((100,), dtype=src_dtype, device=device)
            dst = torch.empty(100, dtype=dst_dtype, device=device)

            dst.conj().copy_(src)
            self.assertEqual(dst, src.conj_physical(), exact_dtype=False)

            dst.conj().copy_(src._neg_view())
            self.assertEqual(dst, src.neg().conj_physical(), exact_dtype=False)

    # FIXME: move to data movement test suite
    @onlyNativeDeviceTypes
    @dtypes(torch.int64, torch.float32, torch.complex64)
    def test_copy_transpose_math_view(self, device, dtype):
        # [FUNCTION DOC] Checks copy_ from a transposed source into a destination with neg-view
        # and conj-view combinations gives expected values for int64, float32 and complex64.
        src = make_tensor((100, 100), dtype=dtype, device=device).transpose(0, 1)
        dst = torch.empty((100, 100), dtype=dtype, device=device)

        dst._neg_view().copy_(src)
        self.assertEqual(dst, -src)
        dst._neg_view().copy_(src._neg_view())
        self.assertEqual(dst, src)
        dst.copy_(src._neg_view())
        self.assertEqual(dst, -src)

        if dtype.is_complex:
            dst.conj().copy_(src)
            self.assertEqual(dst, src.conj_physical())
            dst.conj().copy_(src.conj())
            self.assertEqual(dst, src)
            dst.copy_(src.conj())
            self.assertEqual(dst, src.conj_physical())

    def test_clone_all_dtypes_and_devices(self, device):
        # [FUNCTION DOC] Clones a small tensor of every dtype and checks equality.
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            x = torch.tensor((1, 1), dtype=dt, device=device)
            y = x.clone()
            self.assertEqual(x, y)

    def test_clone_zero_stride_dim(self, device):
        # [FUNCTION DOC] Clones a strided view with a zero stride and size-1 axis and checks the
        # clone equals the view.
        # stride zero, size 1 axis, not contiguous
        x = torch.randn(10)
        y = x.as_strided([2, 1, 5], [1, 0, 2])
        self.assertEqual(y, y.clone())

    def test_clone_not_memory_dense(self):
        # [FUNCTION DOC] Checks that cloning a non-memory-dense strided view keeps the permuted
        # stride layout (issue 64176).
        # github issue: https://github.com/pytorch/pytorch/issues/64176
        x = torch.randn(10, 8).t()[::2, ::2]
        y = x.clone()
        # should retain permutation after densification
        self.assertTrue(y.stride() == (1, 4))

    # FIXME: move to elementwise ternary test suite
    @parametrize("use_cpu_scalar", [True, False])
    @dtypesIfCUDA(*set(get_all_math_dtypes('cuda')))
    @dtypes(*set(get_all_math_dtypes('cpu')))
    def test_addcmul(self, device, dtype, use_cpu_scalar):
        # [FUNCTION DOC] Checks addcmul with value argument against a reference expression,
        # optionally with a CPU scalar tensor, checks the deprecated positional-scalar overload
        # warns once, and on CUDA half checks a value near overflow gives a finite result.
        # Returns floating or integral scalar corresponding to dtype
        def _number(floating, integer, dtype):
            # [FUNCTION DOC] HELPER: Return the floating value for float types, a complex number
            # built from it for complex types, and the integer otherwise, according to the dtype.
            if dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
                return floating
            elif dtype in [torch.cfloat, torch.cdouble]:
                return floating * (1 + 1j)
            else:
                return integer

        def rand_tensor(size, dtype, device):
            # [FUNCTION DOC] HELPER: Return a random tensor of the size and dtype: uniform for
            # float/complex, integers 1..4 for uint8, integers -5..4 otherwise.
            if dtype.is_floating_point or dtype.is_complex:
                return torch.rand(size=size, dtype=dtype, device=device)
            if dtype == torch.uint8:
                return torch.randint(1, 5, size=size, dtype=dtype, device=device)
            else:
                return torch.randint(-5, 5, size=size, dtype=dtype, device=device)

        a = rand_tensor((2, 2), dtype=dtype, device=device)
        b = rand_tensor((2, 2), dtype=dtype, device=device)
        if use_cpu_scalar:
            c = rand_tensor([], device="cpu", dtype=dtype)
        else:
            c = rand_tensor((2, 2), dtype=dtype, device=device)

        alpha = _number(0.5, 3, dtype)

        actual = torch.addcmul(a, b, c, value=alpha)
        expected = a + alpha * b * c

        self.assertEqual(expected, actual)

        with self.assertWarnsOnceRegex(
                UserWarning, "This overload of addcmul is deprecated"):
            self.assertEqual(actual, torch.addcmul(a, alpha, b, c))

        if self.device_type == 'cuda' and dtype == torch.half:
            a = torch.tensor([60000.0], device=device, dtype=dtype)
            b = torch.tensor([60000.0], device=device, dtype=dtype)
            c = torch.tensor([2.0], device=device, dtype=dtype)
            out = torch.addcmul(a, b, c, value=-1)
            self.assertTrue(not (out.isnan() or out.isinf()))

    @onlyCUDA
    def test_addcmul_cuda_errors_with_cpu_scalars(self, device):
        # [FUNCTION DOC] CUDA only. Checks addcmul raises RuntimeError mentioning CPU scalar
        # support when a CPU scalar tensor is passed as tensor1 or as self.
        # Logic is dtype agnostic, so dtype isn't tested
        alpha = 0.5

        a = torch.rand((2, 2), device=device)
        b = torch.rand((2, 2), device=device)
        c = torch.rand((2, 2), device=device)
        scalar = torch.rand([], device="cpu")

        with self.assertRaisesRegex(RuntimeError, r'CPU Scalar support for tensor1 argument'):
            torch.addcmul(a, scalar, c, value=alpha)
        with self.assertRaisesRegex(RuntimeError, r'CPU Scalar support for self argument'):
            torch.addcmul(scalar, b, c, value=alpha)

    # FIXME: move to shape ops test suite
    def test_narrow_empty(self, device):
        # [FUNCTION DOC] Checks narrow at the end of each dim with length zero gives an empty
        # tensor of the expected size.
        x = torch.randn(2, 3, 4, device=device)
        for d in range(x.dim()):
            y = x.narrow(d, x.size(d), 0)
            sz = list(x.size())
            sz[d] = 0
            self.assertEqual(sz, y.size())

    def test_narrow_copy_non_contiguous(self, device):
        # [FUNCTION DOC] Checks narrow_copy on a non-contiguous (moved dim) input equals
        # narrow_copy on its contiguous version (issue 91690).
        # see https://github.com/pytorch/pytorch/issues/91690.
        inp = torch.randn(10, 2, device=device).movedim(-1, 0)
        expected = torch.narrow_copy(inp.contiguous(), 1, 0, 10)
        actual = torch.narrow_copy(inp, 1, 0, 10)
        self.assertEqual(expected, actual)

    # FIXME: find a test suite for the take operator
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_take(self, device, dtype):
        # [FUNCTION DOC] Compares torch.take against numpy take for contiguous/noncontiguous
        # sources and indices, with 1D and reshaped 2D indices, and all four combinations of
        # scalar and one-element shapes.
        idx_size = (4,)

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        make_idx = partial(make_tensor, low=0, device=device, dtype=torch.int64)

        def ref_take(src, idx):
            # [FUNCTION DOC] HELPER: Reference take implemented with numpy: converts to numpy
            # (half for bfloat16), takes, and moves the result back to the device with the dtype.
            if dtype == torch.bfloat16:
                src = src.half()
            src = src.cpu().numpy()
            idx = idx.cpu().numpy()
            out = torch.from_numpy(np.take(src, idx)).to(device=device, dtype=dtype)
            return out

        for src_contig, idx_contig, idx_reshape in product([True, False], repeat=3):
            for src_size in ((5,), (4, 5)):
                src = make_arg(src_size, noncontiguous=not src_contig)
                idx = make_idx(idx_size, high=src.numel(), noncontiguous=not idx_contig)
                if idx_reshape:
                    idx = idx.reshape(2, 2)
                out = torch.take(src, idx)
                out2 = ref_take(src, idx)
                self.assertEqual(out, out2)

        # Create the 4 possible combinations of scalar sizes for source / index
        for size_s, size_i in product([(), (1,)], repeat=2):
            source = make_arg(size_s)
            idx = make_idx(size_i, high=1)
            out = source.take(idx)
            self.assertEqual(out.item(), source.item())

    # FIXME: find a test suite for the put operator
    # The bool instance does not work on GPU. See
    # https://github.com/pytorch/pytorch/issues/54317
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_put(self, device, dtype):
        # [FUNCTION DOC] Compares torch.put and Tensor.put_ (accumulate or not) with a reference
        # built from index_add_/index_copy_ across contiguity combinations, checks scalar-shape
        # combinations and the empty-index case.
        src_size = (4,)

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        make_idx = partial(make_tensor, low=0, device=device, dtype=torch.int64)

        def ref_put(dst, idx, src, accumulate):
            # [FUNCTION DOC] HELPER: Reference put: flattens destination, index and source into
            # 1D, applies index_add_ if accumulating or index_copy_ otherwise, and views the
            # result in the destination shape.
            new_dst = dst.clone(memory_format=torch.contiguous_format).view(-1)
            new_idx = idx.contiguous().view(-1)
            new_src = src.contiguous().view(-1)
            method = new_dst.index_add_ if accumulate else new_dst.index_copy_
            return method(0, new_idx, new_src).view_as(dst)

        for dst_contig, src_contig, idx_contig, idx_reshape, accumulate in product([True, False], repeat=5):
            for dst_size in ((5,), (4, 5)):
                dst = make_arg(dst_size, noncontiguous=not dst_contig)
                src = make_arg(src_size, noncontiguous=not src_contig)

                # If accumulate=True, `put_` should be deterministic regardless of the inputs on CPU
                # On CUDA it may not be, but the test has enough tolerance to account for this
                if accumulate:
                    idx = make_idx(src_size, high=dst.numel())
                else:
                    idx = torch.randperm(dst.numel(), dtype=torch.int64, device=device)[:src_size[0]]
                if not idx_contig:
                    idx = torch.repeat_interleave(idx, 2, dim=-1)[..., ::2]
                if idx_reshape:
                    idx = idx.reshape(2, 2)
                out = torch.put(dst, idx, src, accumulate)
                # out-place
                reference = ref_put(dst, idx, src, accumulate)
                self.assertEqual(out, reference)

                # in-place
                dst.put_(idx, src, accumulate)
                self.assertEqual(dst, reference)


        # Create the 8 possible combinations of scalar sizes for target / index / source
        scalars = ((make_arg(size_t),
                    make_idx(size_i, high=1),
                    make_arg(size_s))
                   for size_t, size_i, size_s in product([(), (1,)], repeat=3))
        for (dest, idx, source), accumulate in product(scalars, [True, False]):
            dest_init = dest.clone()
            # out-place
            out = torch.put(dest, idx, source, accumulate=accumulate)
            # in-place
            dest1 = dest.clone()
            dest1.put_(idx, source, accumulate=accumulate)
            for d in [out, dest1]:
                if accumulate:
                    self.assertEqual(d.item(), (dest_init + source).item())
                else:
                    self.assertEqual(d.item(), source.item())

        # Empty case
        dest = make_arg((3, 2))
        reference = dest.clone()
        idx = make_idx((0,), high=1)
        source = make_arg((0,))
        for accumulate in [True, False]:
            out = torch.put(dest, idx, source, accumulate=accumulate)
            self.assertEqual(out, reference)
            dest.put_(idx, source, accumulate=accumulate)
            self.assertEqual(dest, reference)

    # FIXME: find a test suite for the put operator
    # The bool instance does not work on GPU. See
    # https://github.com/pytorch/pytorch/issues/54317
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_put_accumulate(self, device, dtype):
        # [FUNCTION DOC] Tests put with accumulate on, where all indices point to position zero:
        # for one or two large 1D sizes (smaller for half and bfloat16, to avoid overflow; large
        # enough to trigger CPU parallel loops) checks the result equals the original value plus
        # the sum of the source, with looser tolerances for low precision types.
        # Test for parallel adds with accumulate == True
        low_precision = dtype == torch.half or dtype == torch.bfloat16
        # Less numbers to avoid overflow with low_precision
        # Grainsize is 3000 for the for_loop to be parallelized on CPU
        sizes = ((100,)) if low_precision else ((200,), (3002,))
        # Bfloat16 has a particularly bad performance here
        # This operation is nondeterministic on GPU, so we are generous with the rtol
        rtol, atol = (1e-1, 1e-2) if low_precision else (1e-3, 1e-4)

        make_arg = partial(make_tensor, low=-2, high=3, device=device, dtype=dtype)
        # Dump everything into the 0-th position
        make_idx = partial(torch.zeros, device=device, dtype=torch.int64)
        args = ((make_idx(size), make_arg(size)) for size in sizes)

        for idx, source in args:
            orig = make_arg((1,))
            out = orig.put(idx, source, accumulate=True)
            self.assertEqual(out, orig + source.sum(), rtol=rtol, atol=atol)

    # FIXME: find a test suite for the take operator
    def test_take_empty(self, device):
        # [FUNCTION DOC] Checks take with empty inputs and empty index tensors returns an empty
        # result with the index shape.
        for input_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                input = torch.empty(input_shape, device=device)
                indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                self.assertEqual(indices, torch.take(input, indices), exact_dtype=False)

    # FIXME: find a test suite for the put operator
    def test_put_empty(self, device):
        # [FUNCTION DOC] Checks put_ with empty destination or empty indices (accumulate on or
        # off) leaves the destination equal to the returned tensor.
        for dst_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                for accumulate in [False, True]:
                    dst = torch.randn(dst_shape, device=device)
                    indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                    src = torch.randn(indices_shape, device=device)
                    self.assertEqual(dst, dst.put_(indices, src, accumulate=accumulate))

    # FIXME: port to test_scatter_gather_ops.py
    def scatter_allow_reduce(self, device, dtype, reduceop):
        # [FUNCTION DOC] HELPER: Return True unless the device is CUDA; on CUDA only the multiply
        # reduction of floating dtypes is allowed.
        device_type = torch.device(device).type
        return device_type != 'cuda' or (reduceop == 'multiply' and dtype.is_floating_point)

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_operations_to_large_input(self, device, dtype):
        # [FUNCTION DOC] Checks scatter_ with reduce modes add and multiply from a 2 by 2 source
        # and a 2 by 1 index into a 4 by 4 tensor against fixed expected matrices; combinations
        # unsupported on CUDA (anything except multiply on floating types) are skipped via
        # scatter_allow_reduce.
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        test_data = [
            (torch.zeros(4, 4, device=device, dtype=dtype),
             torch.ones(2, 2, device=device, dtype=dtype),
             torch.tensor([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 0]],
                          device=device, dtype=dtype), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(4, 4),
             torch.tensor([6], device=device, dtype=dtype).repeat(2, 2),
             torch.tensor([[2, 2, 2, 2],
                           [12, 2, 2, 2],
                           [12, 2, 2, 2],
                           [2, 2, 2, 2]], device=device, dtype=dtype), "multiply"),
        ]

        for input, src, result, operation in test_data:
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            input.scatter_(0, index, src, reduce=operation)
            self.assertEqual(input, result)

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_scalar(self, device, dtype):
        # [FUNCTION DOC] Same as the large-input reduce test but with a scalar source value (1
        # with add, 2 with multiply), comparing against fixed expected matrices.
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        test_data = [
            (torch.zeros(4, 4, device=device, dtype=dtype), 1,
             torch.tensor([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 0]],
                          device=device, dtype=dtype), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(4, 4), 2,
             torch.tensor([[2, 2, 2, 2],
                           [4, 2, 2, 2],
                           [4, 2, 2, 2],
                           [2, 2, 2, 2]], device=device, dtype=dtype), "multiply"),
        ]

        for input, src, result, operation in test_data:
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            input.scatter_(0, index, src, reduce=operation)
            self.assertEqual(input, result)

    # FIXME: port to test_scatter_gather_ops.py
    # TODO: remove this after scatter_add_ is deprecated.
    def test_scatter_add_non_unique_index(self, device):
        # [FUNCTION DOC] Checks scatter_add_ with all indices pointing to column zero on a 2 by
        # 65536 tensor of ones: column zero ends with 3 in row 0 and 1 in row 1, others unchanged.
        height = 2
        width = 65536
        input = torch.ones(height, width, device=device)
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        src = torch.ones(height, width, device=device)
        input.scatter_add_(0, index, src)

        self.assertEqual(input,
                         torch.tensor([[3], [1]], device=device,
                                      dtype=torch.float32).repeat(1, width))

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_non_unique_index(self, device, dtype):
        # [FUNCTION DOC] Checks scatter_ with reduce add and multiply when several source elements
        # target the same location (index all zeros); compares to fixed expected values and
        # includes a message with the operation on failure.
        height = 2
        width = 2
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        test_data = [
            (torch.ones(height, width, device=device, dtype=dtype),
             torch.ones(height, width, device=device, dtype=dtype),
             torch.tensor([[3], [1]], device=device, dtype=dtype).repeat(1, width), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(height, width),
             torch.tensor([2], device=device, dtype=dtype).repeat(height, width),
             torch.tensor([[8], [2]], device=device,
                          dtype=dtype).repeat(1, width), "multiply"),
        ]

        for input, src, result, operation in test_data:
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            input.scatter_(0, index, src, reduce=operation)
            self.assertEqual(input, result, msg=lambda msg: f"{msg}\nresult: {result} input: {input} method: {str(operation)}")

    @onlyCUDA
    @dtypes(*complex_types())
    def test_scatter_reduce_multiply_unsupported_dtypes(self, device, dtype):
        # [FUNCTION DOC] CUDA only, complex dtypes. Checks that scatter_ with reduce multiply
        # raises RuntimeError.
        height = 2
        width = 2
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        input = torch.ones(height, width, device=device, dtype=dtype)
        src = torch.ones(height, width, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            input.scatter_(0, index, src, reduce="multiply")

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_to_large_input(self, device):
        # [FUNCTION DOC] Checks scatter_ of a 2 by 2 block of ones into a 4 by 4 zero tensor with
        # a 2 by 1 index matches the expected pattern.
        input = torch.zeros(4, 4, device=device)
        src = torch.ones(2, 2, device=device)
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        input.scatter_(0, index, src)
        self.assertEqual(input, torch.tensor([[0, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 0]], device=device, dtype=torch.float32))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_add_to_large_input(self, device):
        # [FUNCTION DOC] Same as the scatter large-input test but using scatter_add_.
        input = torch.zeros(4, 4, device=device)
        src = torch.ones(2, 2, device=device)
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        input.scatter_add_(0, index, src)
        self.assertEqual(input, torch.tensor([[0, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 0]], device=device, dtype=torch.float32))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_bool(self, device):
        # [FUNCTION DOC] Checks scatter_ with a bool source into a bool zero matrix yields an
        # identity-like pattern.
        x = torch.tensor([[True, True, True], [True, True, True]], device=device)
        res = torch.zeros(3, 3, dtype=torch.bool, device=device)
        res = res.scatter_(0, torch.tensor([[0, 1, 2], [0, 1, 2]], device=device), x)
        self.assertEqual(res, torch.tensor([[True, False, False],
                                            [False, True, False],
                                            [False, False, True]], device=device))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_add_bool(self, device):
        # [FUNCTION DOC] Checks scatter_add_ with bool source and repeated indices into a 3 by 5
        # bool tensor gives the expected logical-or style result.
        x = torch.tensor([[True, True, True, True, True], [True, True, True, True, True]], device=device)
        res = torch.zeros(3, 5, dtype=torch.bool, device=device)
        res = res.scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=device), x)
        self.assertEqual(res, torch.tensor([[True, True, True, True, True],
                                            [False, True, False, True, False],
                                            [True, False, True, False, True]], device=device))

    # FIXME: find a test suite for the masked scatter operator
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_masked_scatter(self, device, dtype):
        # [FUNCTION DOC] Native devices. Checks masked_scatter_ copies consecutive source elements
        # into masked positions (with zero and ones sources against a loop-built reference),
        # raises RuntimeError when the source is too small (not tested on CUDA), and accepts empty
        # destinations with broadcast masks.
        dt = dtype
        num_copy, num_dest = 3, 10
        dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dt, device=device)
        dest2 = dest.clone()
        dest_ones = dest.clone()
        dest_ones_expected = dest.clone()
        src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt, device=device)
        src_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dt, device=device)
        mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device=device)

        dest.masked_scatter_(mask, src)
        j = 0
        for i in range(num_dest):
            if mask[i]:
                dest2[i] = src[j]
                dest_ones_expected[i] = src_ones[j]
                j += 1
        self.assertEqual(dest, dest2, atol=0, rtol=0)

        dest_ones.masked_scatter_(mask, src_ones)
        self.assertEqual(dest_ones, dest_ones_expected, atol=0, rtol=0)

        # Bound checking in CUDA is done inside a kernel
        # in order to avoid synchronization, but this means
        # we can not clear the failures. So there is no way
        # to test it then recover.
        if self.device_type != 'cuda':
            # make src smaller. this should fail
            src = torch.zeros(num_copy - 1, dtype=dt, device=device)
            with self.assertRaises(RuntimeError):
                dest.masked_scatter_(mask, src)

        # empty tensor
        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        mask = torch.ones_like(dest, dtype=torch.bool, device=device)
        src = torch.empty((0,), dtype=dt, device=device)
        dest.masked_scatter_(mask, src)

        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        mask = torch.ones((5, 1, 5), dtype=torch.bool, device=device)
        src = torch.empty((0,), dtype=dt, device=device)
        dest.masked_scatter_(mask, src)

    # FIXME: find a test suite for the masked scatter operator
    def test_masked_scatter_bool_tensor(self, device):
        # [FUNCTION DOC] Checks masked_scatter_ and masked_scatter with bool tensors for two
        # different masks.
        src = torch.tensor([True, True, True], device=device)
        dst = torch.tensor([False, False, False], device=device)
        mask = torch.tensor([False, True, False], device=device)

        dst.masked_scatter_(mask, src)
        self.assertEqual(dst, torch.tensor([False, True, False], device=device))

        mask = torch.tensor([True, False, True], device=device)
        dst = dst.masked_scatter(mask, src)
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

    # FIXME: find a test suite for the masked scatter operator
    #   test_scatter_gather_ops or test_masked_ops?
    @onlyCUDA
    @largeTensorTest('30GB')
    def test_masked_scatter_large_tensor(self, device):
        # [FUNCTION DOC] CUDA only, large (30GB). Compares masked_scatter on a CUDA bool tensor of
        # 2**31 plus 1 elements with the CPU result.
        t_cpu = torch.empty(2**31 + 1, dtype=torch.bool).random_()
        t = t_cpu.to(device)
        result_cpu = t_cpu.masked_scatter(t_cpu, t_cpu)
        result = t.masked_scatter(t, t)
        self.assertEqual(result, result_cpu)

    # FIXME: find a test suite for the masked select operator
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_masked_select(self, device, dtype):
        # [FUNCTION DOC] For each integral mask dtype and bool: non-bool masks must raise
        # RuntimeError about expecting a BoolTensor; bool masks give a result equal to a
        # loop-built reference, including with out=. Also checks the mask broadcasts against the
        # tensor, and the tensor broadcasts against a column mask.
        for maskType in integral_types_and(torch.bool):
            num_src = 10
            src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device)
            mask = torch.randint(2, (num_src,), device=device, dtype=maskType)

            if maskType is not torch.bool:
                with self.assertRaisesRegex(RuntimeError, r'expected BoolTensor for mask'):
                    dst = src.masked_select(mask)
                continue
            else:
                dst = src.masked_select(mask)
            dst2 = []
            for i in range(num_src):
                if mask[i]:
                    dst2 += [src[i]]
            self.assertEqual(dst, torch.tensor(dst2), atol=0, rtol=0)

            dst3 = torch.empty(0, device=device, dtype=dtype)
            torch.masked_select(src, mask, out=dst3)
            self.assertEqual(dst3, torch.tensor(dst2, dtype=dst3.dtype), atol=0, rtol=0)

        # Ensure that masks are expanded to match tensor properly
        a = torch.rand(100, 100, device=device).mul(100).to(dtype)
        mask_first_el_each_row = torch.zeros(100, device=device, dtype=torch.bool)
        mask_first_el_each_row[0] = True
        a_masked = a.masked_select(mask_first_el_each_row)
        self.assertEqual(a_masked, a[:, 0])

        mask_first_row = torch.zeros(100, 1, device=device, dtype=torch.bool)
        mask_first_row[0][0] = True
        a_masked = a.masked_select(mask_first_row)
        self.assertEqual(a_masked, a[0, :])

        # Ensure that tensor is expanded to match mask properly
        a = torch.rand(100, device=device).mul(100).to(dtype)
        mask_copy_3_times = torch.tensor([[True], [True], [False], [True]], device=device)
        a_masked = a.masked_select(mask_copy_3_times)
        self.assertEqual(a_masked, a.unsqueeze(0).expand(3, 100).flatten())

    # FIXME: find a test suite for the masked select operator
    def test_masked_select_discontiguous(self, device):
        # [FUNCTION DOC] Checks masked_select with contiguous or transposed values and masks of
        # sizes 10 and 200, and with a non-contiguous out= tensor, against a reference slice.
        for size in (10, 200):
            vals = torch.rand(size, size, device=device)
            mask = torch.full((size, size), False, dtype=torch.bool, device=device)
            mask[:, ::2] = True
            vals_list = (vals, vals.t())
            mask_list = (mask, mask.t())
            out_dc = torch.empty(size * size, device=device)[::2]
            for v, m in product(vals_list, mask_list):
                if m.is_contiguous():
                    expected = v[:, ::2].clone().reshape((-1, ))
                else:
                    expected = v[::2].clone().reshape((-1, ))
                out = torch.masked_select(v, m)
                self.assertEqual(out, expected, atol=0, rtol=0)
                torch.masked_select(v, m, out=out_dc)
                self.assertEqual(out_dc, expected, atol=0, rtol=0)

    # FIXME: find a test suite for the masked fill operator
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16), (torch.uint8, torch.bool)))
    def test_masked_fill(self, device, dtypes):
        # [FUNCTION DOC] Over dtype and mask dtype pairs: non-bool masks (uint8) must raise
        # RuntimeError saying only boolean masks are supported; with bool masks, masked_fill_
        # matches a loop-built reference, and on a permuted non-contiguous tensor it matches the
        # result on a contiguous copy.
        dtype = dtypes[0]
        mask_dtype = dtypes[1]

        num_dest = 10
        dst = torch.zeros(num_dest, dtype=dtype)
        mask = torch.randint(2, (num_dest,), dtype=mask_dtype)
        val = random.random()
        dst2 = dst.clone()

        if mask_dtype is not torch.bool:
            with self.assertRaisesRegex(RuntimeError, 'only supports boolean masks'):
                dst.masked_fill_(mask, val)
            return

        dst.masked_fill_(mask, val)
        for i in range(num_dest):
            if mask[i]:
                dst2[i] = val
        self.assertEqual(dst, dst2, atol=0, rtol=0)

        # test non-contiguous case
        dst = make_tensor(
            (num_dest, num_dest, num_dest), dtype=dtype, device=device, low=-30, high=30
        ).permute((2, 0, 1))
        dst2 = dst.contiguous()
        if dtype.is_complex:
            mask = dst.abs() > 0
        else:
            mask = dst > 0
        self.assertTrue(not dst.is_contiguous())
        self.assertTrue(dst2.is_contiguous())
        dst.masked_fill_(mask.to(mask_dtype), val)
        dst2.masked_fill_(mask.to(mask_dtype), val)
        self.assertEqual(dst, dst2, atol=0, rtol=0)

    # FIXME: find a test suite for the masked fill operator
    def test_masked_fill_bool_tensor(self, device):
        # [FUNCTION DOC] Checks masked_fill_ and masked_fill on bool tensors.
        dst = torch.tensor([True, False, True], device=device)
        mask = torch.tensor([False, True, False], device=device)

        dst.masked_fill_(mask, True)
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

        dst = dst.masked_fill(mask, False)
        self.assertEqual(dst, torch.tensor([True, False, True], device=device))

    def test_tensor_shape_empty(self, device):
        # [FUNCTION DOC] Checks result shapes of many ops on empty tensors: flatten,
        # squeeze/unsqueeze, transpose, select, repeat, permute, diagonal (including far-off
        # offsets), diagflat, stack, chunk, split (including that a zero split size on a nonzero
        # dim raises RuntimeError).
        x = torch.randn((0, 1, 3, 0), device=device)
        # flatten
        self.assertEqual((0,), torch.flatten(x, 0, 3).shape)
        self.assertEqual((0, 0), torch.flatten(x, 0, 2).shape)
        self.assertEqual((0, 3, 0), torch.flatten(x, 1, 2).shape)

        # squeeze, unsqueeze
        self.assertEqual((0, 1, 1, 3, 0), torch.unsqueeze(x, 1).shape)
        self.assertEqual((0, 3, 0), torch.squeeze(x, 1).shape)
        self.assertEqual((0, 3, 0), torch.squeeze(x).shape)

        # transpose, t
        self.assertEqual((0, 0, 3, 1), torch.transpose(x, 1, 3).shape)
        y = torch.randn((5, 0), device=device)
        self.assertEqual((0, 5), y.t().shape)

        # select
        self.assertEqual((0, 1, 0), torch.select(x, 2, 2).shape)

        # repeat, permute
        self.assertEqual((9, 0, 5, 6, 0), x.repeat(9, 7, 5, 2, 3).shape)
        self.assertEqual((3, 0, 0, 1), x.permute(2, 3, 0, 1).shape)

        # diagonal, diagflat
        self.assertEqual((0,), torch.diagonal(torch.randn((5, 0), device=device)).shape)
        self.assertEqual((0,), torch.diagonal(torch.randn((0, 5), device=device)).shape)
        # off the end offsets are valid
        self.assertEqual((0,), torch.diagonal(torch.randn((5, 0), device=device), offset=1).shape)
        self.assertEqual((0,), torch.diagonal(torch.randn((0, 5), device=device), offset=1).shape)
        # check non-zero sized offsets off the end
        self.assertEqual((5, 6, 0), torch.diagonal(torch.randn((3, 4, 5, 6), device=device), offset=45252).shape)
        self.assertEqual((5, 6, 0), torch.diagonal(torch.randn((3, 4, 5, 6), device=device), offset=-45252).shape)

        self.assertEqual((0, 0), torch.diagflat(torch.tensor([], device=device)).shape)
        self.assertEqual(torch.zeros(1, 1), torch.diagflat(torch.tensor([], device=device), offset=1))
        self.assertEqual((0, 0), torch.diagflat(torch.tensor([[]], device=device)).shape)
        self.assertEqual(torch.zeros(1, 1), torch.diagflat(torch.tensor([[]], device=device), offset=1))

        # stack, split, chunk
        self.assertEqual((4, 0, 1, 3, 0), torch.stack((x, x, x, x)).shape)
        self.assertEqual([(0, 1, 3, 0)],
                         [z.shape for z in torch.chunk(x, 1, dim=0)])

        self.assertEqual([(0, 1, 3, 0), ] * 3, [z.shape for z in torch.chunk(x, 3, dim=0)])
        self.assertEqual([(0, 1, 1, 0), ] * 3, [z.shape for z in torch.chunk(x, 3, dim=2)])

        # NOTE: split_with_sizes behaves differently than NumPy in that it
        # takes sizes rather than offsets
        self.assertEqual([(0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 2, 0)],
                         [z.shape for z in torch.split(x, (0, 1, 2), dim=2)])

        self.assertRaises(RuntimeError, lambda: torch.split(x, 0, dim=1))
        # This is strange because the split size is larger than the dim size, but consistent with
        # how split handles that case generally (when no 0s are involved).
        self.assertEqual([(0, 1, 3, 0)], [z.shape for z in torch.split(x, 1, dim=0)])
        self.assertEqual([(0, 1, 3, 0)], [z.shape for z in torch.split(x, 0, dim=0)])

    # functions that operate over a dimension but don't reduce.
    def test_dim_function_empty(self, device):
        # [FUNCTION DOC] Checks ops that act over a dimension on tensors with zero-size dims:
        # size/stride, glu, softmax, log_softmax, cumulative ops, flip, roll, unbind, cross,
        # renorm, sort, topk, gather, scatter, scatter_add, index_fill, index_copy, index_add and
        # index_select, verifying shapes or unchanged values, plus RuntimeError cases for
        # index_select on CPU with invalid empty-axis or out-of-bounds indices and for a scalar
        # with a two-element index.
        shape = (0, 1, 2, 0)
        x = torch.randn(shape, device=device)

        # size stride
        self.assertEqual(0, x.size(3))
        self.assertEqual(2, x.size(2))
        self.assertEqual(2, x.stride(0))
        self.assertEqual(1, x.stride(2))

        self.assertEqual(x, torch.nn.functional.glu(x, 0))
        self.assertEqual((0, 1, 1, 0), torch.nn.functional.glu(x, 2).shape)

        # softmax, logsoftmax
        self.assertEqual(x, torch.nn.functional.softmax(x, 0))
        self.assertEqual(x, torch.nn.functional.softmax(x, 2))
        self.assertEqual(x, torch.nn.functional.softmax(x, 3))

        self.assertEqual(x, torch.nn.functional.log_softmax(x, 0))
        self.assertEqual(x, torch.nn.functional.log_softmax(x, 2))
        self.assertEqual(x, torch.nn.functional.log_softmax(x, 3))

        # cumsum, cumprod, cummax, cummin
        self.assertEqual(shape, torch.cumsum(x, 0).shape)
        self.assertEqual(shape, torch.cumsum(x, 2).shape)
        self.assertEqual(shape, torch.cumprod(x, 0).shape)
        self.assertEqual(shape, torch.cumprod(x, 2).shape)
        self.assertEqual(shape, torch.cummax(x, 0)[0].shape)
        self.assertEqual(shape, torch.cummax(x, 2)[0].shape)
        self.assertEqual(shape, torch.cummin(x, 0)[0].shape)
        self.assertEqual(shape, torch.cummin(x, 2)[0].shape)
        self.assertEqual(shape, torch.logcumsumexp(x, 0).shape)
        self.assertEqual(shape, torch.logcumsumexp(x, 2).shape)

        # flip
        self.assertEqual(x, x.flip(0))
        self.assertEqual(x, x.flip(2))

        # roll
        self.assertEqual(x, x.roll(0, 1).roll(0, -1))
        self.assertEqual(x, x.roll(1, x.size(1)))
        self.assertEqual(x, x.roll(1))
        self.assertEqual(x, x.roll((1, 1), (3, 1)))

        # unbind
        self.assertEqual((), x.unbind(0))
        self.assertEqual((torch.empty((0, 1, 0), device=device), torch.empty((0, 1, 0), device=device)),
                         x.unbind(2))

        # cross
        y = torch.randn((0, 1, 3, 0), device=device)
        self.assertEqual(y.shape, torch.cross(y, y).shape)

        # renorm
        self.assertEqual(shape, torch.renorm(x, 1, 0, 5).shape)
        self.assertEqual(shape, torch.renorm(x, 1, 2, 5).shape)

        # sort
        self.assertEqual([shape, shape], [z.shape for z in torch.sort(x, dim=0)])
        self.assertEqual([shape, shape], [z.shape for z in torch.sort(x, dim=2)])

        # topk
        self.assertEqual([shape, shape], [z.shape for z in torch.topk(x, 0, dim=0)])
        self.assertEqual([(0, 1, 1, 0), (0, 1, 1, 0)], [z.shape for z in torch.topk(x, 1, dim=2)])

        y = torch.randn((2, 3, 4), device=device)
        self.assertEqual([(2, 3, 0), (2, 3, 0)], [z.shape for z in torch.topk(y, 0)])

        # gather
        self.assertEqual(shape, torch.gather(x, 0, torch.empty(shape, dtype=torch.int64, device=device)).shape)
        self.assertEqual(shape, torch.gather(x, 2, torch.empty(shape, dtype=torch.int64, device=device)).shape)
        larger_shape = torch.empty((0, 1, 3, 0), dtype=torch.int64, device=device)
        self.assertEqual(larger_shape.shape, torch.gather(x, 2, larger_shape).shape)
        smaller_shape = torch.empty((0, 1, 0, 0), dtype=torch.int64, device=device)
        self.assertEqual(smaller_shape.shape, torch.gather(x, 2, smaller_shape).shape)
        y = torch.randn((2, 3, 4), device=device)
        self.assertEqual((0, 3, 4),
                         torch.gather(y, 0, torch.empty((0, 3, 4), dtype=torch.int64, device=device)).shape)

        # scatter, scatter_add
        for dim in [0, 2]:
            y = torch.randn(shape, device=device)
            y_src = torch.randn(shape, device=device)
            ind = torch.empty(shape, dtype=torch.int64, device=device)
            self.assertEqual(shape, y.scatter_(dim, ind, y_src).shape)
            self.assertEqual(shape, y.scatter_add_(dim, ind, y_src).shape)

        z = torch.randn((2, 3, 4), device=device)
        z_src = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.scatter_(2, torch.empty((2, 3, 0), dtype=torch.int64, device=device), z_src))
        self.assertEqual(z, z.scatter_add_(2, torch.empty((2, 3, 0), dtype=torch.int64, device=device), z_src))

        # index_fill, index_copy, index_add
        c = x.clone()
        c_clone = c.clone()
        ind_empty = torch.tensor([], dtype=torch.int64, device=device)
        ind_01 = torch.tensor([0, 1], dtype=torch.int64, device=device)
        self.assertEqual(c_clone, c.index_fill_(0, ind_empty, -1))
        self.assertEqual(c_clone, c.index_fill_(2, ind_empty, -1))
        self.assertEqual(c_clone, c.index_fill_(2, ind_01, -1))
        self.assertEqual(c_clone, c.index_copy_(0, ind_empty, torch.empty((0, 1, 2, 0), device=device)))
        self.assertEqual(c_clone, c.index_copy_(2, ind_empty, torch.empty((0, 1, 0, 0), device=device)))
        self.assertEqual(c_clone, c.index_copy_(2, ind_01, torch.empty((0, 1, 2, 0), device=device)))
        self.assertEqual(c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2, 0), device=device)))
        self.assertEqual(c_clone, c.index_add_(2, ind_empty, torch.empty((0, 1, 0, 0), device=device)))
        self.assertEqual(c_clone, c.index_add_(2, ind_01, torch.empty((0, 1, 2, 0), device=device)))

        c = torch.randn((0, 1, 2), device=device)
        c_clone = c.clone()
        self.assertEqual(c_clone, c.index_fill_(0, ind_empty, -1))
        self.assertEqual(c_clone, c.index_copy_(0, ind_empty, torch.empty((0, 1, 2), device=device)))
        self.assertEqual(c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2), device=device)))
        self.assertEqual(c_clone, c.index_fill_(0, ind_empty, -1))
        self.assertEqual(c_clone, c.index_copy_(0, ind_empty, torch.empty((0, 1, 2), device=device)))
        self.assertEqual(c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2), device=device)))

        # index fill/copy/add non-empty
        z = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.index_fill_(0, ind_empty, -1))
        z = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.index_copy_(0, ind_empty, torch.empty((0, 3, 4), device=device)))
        z = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.index_add_(0, ind_empty, torch.empty((0, 3, 4), device=device)))

        # index_select
        self.assertEqual(x, x.index_select(0, ind_empty))
        self.assertEqual((0, 1, 0, 0), x.index_select(2, ind_empty).shape)
        self.assertEqual(x, x.index_select(2, ind_01))
        z = torch.randn((2, 3, 4), device=device)  # non-empty
        self.assertEqual((0, 3, 4), z.index_select(0, ind_empty).shape)
        c = torch.randn((0, 1, 2), device=device)
        self.assertEqual(c, c.index_select(0, ind_empty))
        c = torch.randn((0, 1, 2), device=device)
        self.assertEqual(c, c.index_select(0, ind_empty))
        w = torch.randn((0, 3), device=device)
        self.assertEqual((0, 2), w.index_select(1, ind_01).shape)
        w = torch.randn((3, 0), device=device)
        self.assertEqual((2, 0), w.index_select(0, ind_01).shape)
        ind_01_int32 = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self.assertEqual((2, 0), w.index_select(0, ind_01_int32).shape)
        s = torch.randn([], device=device)
        ind_0 = torch.tensor([0], dtype=torch.int32, device=device)
        self.assertEqual([], s.index_select(0, ind_0).shape)
        if device == 'cpu':
            w = torch.randn((0, 3), device=device)
            with self.assertRaisesRegex(RuntimeError, "self indexing axis dim should be positive"):
                torch.index_select(w, 0, ind_01)
            ind_05 = torch.tensor([0, 5], dtype=torch.int64, device=device)
            with self.assertRaisesRegex(RuntimeError, "INDICES element is out of DATA bounds"):
                torch.index_select(w, 1, ind_05)
            with self.assertRaisesRegex(RuntimeError, "Index to scalar can have only 1 value"):
                torch.index_select(s, 0, ind_empty)
        with self.assertRaisesRegex(RuntimeError, "Index to scalar can have only 1 value"):
            torch.ones([]).index_select(0, torch.Tensor([0, 0]).int())

    # FIXME: find a test suite for the pdist operator
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @skipIfRocm
    @onlyCUDA
    @largeTensorTest('32GB', device='cpu')
    @largeTensorTest('5GB', device='cuda')
    def test_pdist_norm_large(self, device):
        # [FUNCTION DOC] CUDA only, large. Computes pdist (p=2) of 50000 by 1 float tensor on
        # CPU and GPU and compares the results, exercising the large-size forward kernel
        # (issue 30583).
        # use dim0>=46342 for forward, see:
        # https://github.com/pytorch/pytorch/issues/30583
        # Compare output using GPU with the CPU implementation
        x = torch.randn(50000, 1, dtype=torch.float32)      # 50k * 4 bytes = 200 KB
        # Will require 1249975000 float32s
        expected_cpu = torch.pdist(x, p=2)                  # ~1250M * 4 bytes = 5 GB on CPU
        actual_cpu = torch.pdist(x.to(device), p=2).cpu()         # 5 GB on GPU + 5GB on CPU
        # Workaround for large memory overhead of self.assertTrue (see #84944)
        self.assertTrue(torch.allclose(expected_cpu, actual_cpu))  # ~20GB in allclose

    # FIXME: move to elementwise ternary test suite
    @onlyNativeDeviceTypes
    @dtypesIfCUDA(*set(get_all_math_dtypes('cuda')))
    @dtypes(*set(get_all_math_dtypes('cpu')))
    def test_addcdiv(self, device, dtype):
        # [FUNCTION DOC] Native devices. Checks addcdiv against a reference expression for
        # float/complex dtypes (including the deprecated positional-scalar overload warning once);
        # integer dtypes must raise RuntimeError because integer division is prohibited. On CUDA
        # half also runs a near-overflow check.
        # KNOWN ISSUE: the final CUDA half overflow check calls addcmul rather than addcdiv, so
        # the addcdiv half-precision path is not what is actually exercised there.
        # Returns floating or integral scalar corresponding to dtype
        def _number(floating, integer, dtype):
            # [FUNCTION DOC] HELPER: Return the floating value for float types, a complex number
            # built from it for complex types, and the integer otherwise, according to the dtype.
            if dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
                return floating
            elif dtype in [torch.cfloat, torch.cdouble]:
                return floating * (1 + 1j)
            else:
                return integer

        def non_zero_rand(size, dtype, device):
            # [FUNCTION DOC] HELPER: Return a random tensor of the size and dtype with no zero
            # entries (zeros are replaced by one).
            if dtype.is_floating_point or dtype.is_complex:
                a = torch.rand(size=size, dtype=dtype, device=device)
            elif dtype == torch.uint8:
                a = torch.randint(1, 5, size=size, dtype=dtype, device=device)
            else:
                a = torch.randint(-5, 5, size=size, dtype=dtype, device=device)
            return a + (a == 0).to(dtype)

        def _test_addcdiv():
            # [FUNCTION DOC] HELPER: Compare addcdiv with a reference expression on non-zero
            # random 2 by 2 tensors, and check the deprecated overload warns once.
            a = non_zero_rand((2, 2), dtype=dtype, device=device)
            b = non_zero_rand((2, 2), dtype=dtype, device=device)
            c = non_zero_rand((2, 2), dtype=dtype, device=device)
            alpha = _number(0.5, 3, dtype)

            expected = a + (alpha * b) / c
            actual = torch.addcdiv(a, b, c, value=alpha)
            self.assertEqual(expected, actual)

            with self.assertWarnsOnceRegex(
                    UserWarning, "This overload of addcdiv is deprecated"):
                self.assertEqual(actual, torch.addcdiv(a, alpha, b, c))

        if not (dtype.is_floating_point or dtype.is_complex):
            # Integer division with addcdiv is prohibited
            with self.assertRaises(RuntimeError):
                _test_addcdiv()
        else:
            _test_addcdiv()

        if self.device_type == 'cuda' and dtype == torch.half:
            a = torch.tensor([60000.0], device=device, dtype=dtype)
            b = torch.tensor([60000.0], device=device, dtype=dtype)
            c = torch.tensor([1.0], device=device, dtype=dtype)
            out = torch.addcmul(a, b, c, value=-2)
            self.assertTrue(not (out.isnan() or out.isinf()))

    def test_nullary_op_mem_overlap(self, device):
        # [FUNCTION DOC] For each random-fill op (random_, uniform_, cauchy_, log_normal_,
        # exponential_, geometric_, normal_) calls it in place on an expanded tensor and expects
        # RuntimeError saying the operation is unsupported due to overlap.
        ops = (
            ("random_", ()),
            ("uniform_", ()),
            ("cauchy_", ()),
            ("log_normal_", ()),
            ("exponential_", ()),
            ("geometric_", (0.5,)),
            ("normal_", ()),
        )

        x = torch.rand((1, 3)).expand((3, 3))
        for op, args in ops:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                getattr(x, op)(*args)

    # FIXME: move to an elementwise ternary test suite and make this an OpInfo test
    # https://github.com/pytorch/pytorch/issues/126474
    @xfailIfTorchDynamo
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/126474")
    @dtypes(torch.double)
    def test_ternary_op_mem_overlap(self, device, dtype):
        # [FUNCTION DOC] Runs memory-overlap checks (internal overlap and input/output overlap)
        # for addcmul, addcdiv and lerp in-place and out forms for the matching device only;
        # skipped under Dynamo and inductor.
        if device == "cpu" and TEST_WITH_TORCHINDUCTOR:
            self.skipTest("Failing on cpu")

        ops = [
            ("addcmul", True, True, 'cpu'),
            ("addcmul", True, True, 'cuda'),
            ("addcdiv", True, True, 'cpu'),
            ("addcdiv", True, True, 'cuda'),
            ("lerp", True, True, 'cpu'),
            ("lerp", True, True, 'cuda')
        ]

        for (fn, has_input_output_mem_overlap_check,
             has_internal_mem_overlap_check, dev) in ops:
            if dev != device:
                continue
            out_op = getattr(torch, fn)
            inplace_op = getattr(torch.Tensor, fn + '_')
            self.check_internal_mem_overlap(
                inplace_op, 3, dtype, device,
                expected_failure=not has_internal_mem_overlap_check)
            self.ternary_check_input_output_mem_overlap(out_op, dev,
                                                        expected_failure=not has_input_output_mem_overlap_check)

    @expectedFailureMeta  # RuntimeError not raised
    @dtypes(torch.double)
    @onlyNativeDeviceTypes
    def test_copy_mem_overlap(self, device, dtype):
        # [FUNCTION DOC] Checks copy_ internal overlap and input/output overlap handling; expected
        # to fail on meta devices because no error is raised.
        self.check_internal_mem_overlap(
            torch.Tensor.copy_, num_inputs=2, dtype=dtype, device=device)
        sz = 9
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: out.copy_(input))

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    def test_index_add_mem_overlap(self, device):
        # [FUNCTION DOC] Checks index_add_ raises RuntimeError (unsupported operation) for an
        # expanded self, overlapping source, and index tensors aliasing self or source.
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.index_add_(0, ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_add_(0, ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_add_(0, ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_add_(0, ind.clone(), ind)

    @onlyCUDA
    @skipCUDAIfNotRocm  # This UT throws an OOM error on CUDA
    def test_index_add_large_inputs(self, device):
        # [FUNCTION DOC] CUDA with ROCm only. Runs index_add_ with a large bfloat16 destination
        # and many repeated indices and compares to a loop-accumulated reference.
        D = 6144
        x = torch.zeros([16384, D], device=device, dtype=torch.bfloat16)
        index = torch.randint(0, 16384, (1, 32, 16384), device=device, dtype=torch.int64)
        output = torch.ones([1, 32, 16384, D], device=device, dtype=torch.bfloat16)  # Use random values for test

        x_before = x.clone()
        # Manually update x_before to generate expected values
        for batch in range(output.shape[1]):  # Loop over batch size (32 in this case)
            for idx in range(output.shape[2]):  # Loop over index
                idx_val = index[0, batch, idx].item()
                x_before[idx_val] += output[0, batch, idx]

        # Run index_add to get actual values
        x.index_add_(0, index.view(-1), output.view(-1, D))

        self.assertEqual(x_before, x)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    def test_index_copy_mem_overlap(self, device):
        # [FUNCTION DOC] Same as the index_add_ overlap test but for index_copy_.
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.index_copy_(0, ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_copy_(0, ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_copy_(0, ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_copy_(0, ind.clone(), ind)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # Warning not triggered
    @onlyNativeDeviceTypes
    def test_index_fill_mem_overlap(self, device):
        # [FUNCTION DOC] Checks index_fill_ on an expanded tensor emits a UserWarning about
        # expanded tensors and raises RuntimeError when the index tensor is also the destination;
        # expected to fail on meta.
        x = torch.rand((1,), device=device).expand((6,))
        ind = torch.tensor([2, 1, 0], device=device)

        with self.assertWarnsRegex(UserWarning, "index_fill_ on expanded tensors"):
            x.index_fill_(0, ind, 1.0)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_fill_(0, ind, 0)

    # FIXME: convert to ErrorInputs
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_shift_mem_overlap(self, device):
        # [FUNCTION DOC] Checks that in-place left and right shifts with overlapping slices of the
        # same tensor raise RuntimeError for unsupported operations; expected to fail on meta.
        x = torch.rand(3, device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x[:-1] <<= x[1:]
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x[:-1] >>= x[1:]

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors)
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_bernoulli_mem_overlap(self, device):
        # [FUNCTION DOC] Checks bernoulli_ on an expanded tensor (default, scalar p, tensor p)
        # raises RuntimeError for unsupported operation; expected to fail on meta.
        x = torch.rand((1,), device=device).expand((6,))

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.bernoulli_()
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.bernoulli_(p=0.1)
        p = torch.rand(6, device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.bernoulli_(p=p)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_put_mem_overlap(self, device):
        # [FUNCTION DOC] Checks put_ raises RuntimeError for overlapping arguments (expanded self,
        # aliasing index/source); expected to fail on meta.
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.put_(ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.put_(ind[0], y[0])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.put_(ind, ind)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.put_(ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.put_(ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.put_(ind.clone(), ind)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    def test_index_put_mem_overlap(self, device):
        # [FUNCTION DOC] Checks index_put_ warns about expanded tensors for an expanded self and
        # raises RuntimeError for aliased index/value arguments; expected to fail on meta.
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            x.index_put_((ind,), value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_put_((ind,), y[0])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_put_((ind,), ind)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_put_((ind,), y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_put_((ind,), ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_put_((ind.clone(),), ind)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    def test_masked_fill_mem_overlap(self, device):
        # [FUNCTION DOC] Checks masked_fill_ on an expanded tensor warns about expanded tensors
        # for scalar and tensor values and raises RuntimeError for overlapping mask; expected to
        # fail on meta.
        x = torch.rand((1,), device=device).expand((6,))
        mask = torch.tensor([True, False, True, True, False, False], device=device)
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            x.masked_fill_(mask, 0.)

        fill_val = torch.tensor(0., device=device)
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            x.masked_fill_(mask, fill_val)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            mask[1:].masked_fill_(mask[:-1], False)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_masked_scatter_mem_overlap(self, device):
        # [FUNCTION DOC] Checks masked_scatter_ on an expanded tensor raises RuntimeError for
        # unsupported operation; expected to fail on meta.
        x = torch.rand((1,), device=device).expand((6,))
        src = torch.rand((3,), device=device)
        mask = torch.tensor([True, False, True, True, False, False], device=device)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.masked_scatter_(mask, src)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    def test_scatter_mem_overlap(self, device):
        # [FUNCTION DOC] Checks scatter_ raises RuntimeError for unsupported operation when self
        # is expanded or aliases the source or index.
        x = torch.rand((1,), device=device).expand((6,))
        src = torch.rand((3,), device=device)
        ind = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.scatter_(0, ind, src)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            src.scatter_(0, ind, src)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.scatter_(0, ind, ind.clone())

    # FIXME: move to test distributions
    @onlyCUDA
    def test_multinomial_device_constrain(self, device):
        # [FUNCTION DOC] CUDA only. Checks multinomial with a CPU input and CUDA out tensor raises
        # RuntimeError about tensors on different devices.
        x = torch.empty(3, device="cpu")
        y = torch.empty(3, device=device)
        self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device",
            lambda: torch.multinomial(x, 2, out=y))

    # FIXME: move to test distributions
    @deviceCountAtLeast(2)
    @onlyCUDA
    @skipIfTorchInductor("FIXME: error not thrown")
    def test_multinomial_gpu_device_constrain(self, devices):
        # [FUNCTION DOC] Needs two CUDA devices. Checks multinomial with inputs on different GPUs
        # raises RuntimeError about tensors on different devices (skipped under inductor).
        x = torch.empty(3, device=devices[0])
        y = torch.empty(3, device=devices[1], dtype=torch.long)
        self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device",
            lambda: torch.multinomial(x, 2, out=y))

    # FIXME: convert this to an automated OpInfo test
    @deviceCountAtLeast(2)
    @onlyCUDA
    def test_device_guard(self, devices):
        # [FUNCTION DOC] Needs two CUDA devices. Calls a large list of operators that do not use a
        # device guard (property queries, sparse property ops, in-place resizing/viewing ops,
        # shape modification, chunk/split, empty, to) on tensors that live on the second device
        # while the current device is the first, to check they do not fail; no value assertions.
        # verify that all operators with `device_guard: False` behave properly with multiple devices.
        # TODO: if we had operator introspection we could figure out this set of operators automatically...
        x = torch.randn((1, 2, 3), device=devices[1])
        y = torch.zeros((1, 3, 2), device=devices[1])
        scalar = torch.tensor(5, device=devices[1])

        # property ops
        torch.cudnn_is_acceptable(x)
        x.is_distributed()
        x.is_floating_point()
        x.is_complex()
        x.is_same_size(y)
        x.is_signed()
        x.size(0)
        x.stride(0)
        x.numel()
        x.is_set_to(y)
        x.data_ptr()
        scalar.is_nonzero()

        # sparse property ops
        y[0][1] = 5
        y_sparse = y.to_sparse()
        y_sparse.sparse_dim()
        y_sparse._dimI()
        y_sparse.dense_dim()
        y_sparse._dimV()
        y_sparse._nnz()
        y_sparse.is_coalesced()
        y_sparse._indices()
        y_sparse._values()
        y_sparse.indices()
        y_sparse.values()

        # in-place ops
        def inplace():
            # [FUNCTION DOC] HELPER: Create a fresh random tensor on the second device for
            # in-place op checks.
            return torch.randn((1, 2, 3), device=devices[1])
        inplace().as_strided_(y.size(), y.stride())
        inplace().resize_(y.size())
        inplace().squeeze_()
        inplace().squeeze_(0)
        inplace().unsqueeze_(2)
        inplace().transpose_(1, 2)
        inplace().squeeze_().t_()
        inplace().set_(x.storage())
        inplace().set_(x.storage(), x.storage_offset(), x.size(), x.stride())
        inplace().set_(x)
        inplace().set_()
        y_sparse._coalesced_(True)

        # shape modification
        x.as_strided(y.size(), y.stride())
        x.expand((5, 2, 3))
        x.expand_as(x)
        x.sum_to_size((1,))
        torch.broadcast_tensors(x , x)
        x.reshape((1, 3, 2))
        x.reshape_as(y)
        x.squeeze()
        x.squeeze(0)
        x.squeeze().t()
        x.transpose(1, 2)
        x.unsqueeze(2)
        x.view((1, 3, 2))
        x.view_as(y)

        # chunk, split, etc.
        x.chunk(2, dim=1)
        x.split(1, dim=2)
        x.split_with_sizes([1, 2], dim=2)
        x.unfold(dimension=2, size=1, step=1)

        x.narrow(1, 1, 1)
        x.select(1, 1)
        torch.isnan(x)

        torch.empty((1, 3, 2), out=y)
        torch.empty_like(x)
        torch.empty_like(x, dtype=torch.int64)

        # to
        x.to(x)
        x.to(y)
        x.to(x, copy=True)

    def test_is_signed(self, device):
        # [FUNCTION DOC] Checks is_signed for int, byte, char, float and half tensors moved to
        # the device.
        self.assertEqual(torch.IntTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.ByteTensor(5).to(device).is_signed(), False)
        self.assertEqual(torch.CharTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.FloatTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.HalfTensor(10).to(device).is_signed(), True)

    def test_tensor_type(self):
        # [FUNCTION DOC] For every legacy tensor class checks is_cuda and is_xpu are consistent
        # with the class module name.
        for t in torch._tensor_classes:
            if 'cuda' in t.__module__:
                self.assertEqual(t.is_cuda, True)
            else:
                self.assertEqual(t.is_cuda, False)
            if 'xpu' in t.__module__:
                self.assertEqual(t.is_xpu, True)
            else:
                self.assertEqual(t.is_xpu, False)

    # Note - reports a leak of 512 bytes on CUDA device 1
    @deviceCountAtLeast(2)
    @skipCUDAMemoryLeakCheckIf(True)
    @onlyCUDA
    def test_tensor_set_errors_multigpu(self, devices):
        # [FUNCTION DOC] Needs two CUDA devices. Checks that set_ with a storage or tensor from
        # another GPU raises RuntimeError.
        f_cuda0 = torch.randn((2, 3), dtype=torch.float32, device=devices[0])
        f_cuda1 = torch.randn((2, 3), dtype=torch.float32, device=devices[1])

        self.assertRaises(RuntimeError, lambda: f_cuda0.set_(f_cuda1.storage()))
        self.assertRaises(RuntimeError,
                          lambda: f_cuda0.set_(f_cuda1.storage(), 0, f_cuda1.size(), f_cuda1.stride()))
        self.assertRaises(RuntimeError, lambda: f_cuda0.set_(f_cuda1))

    # FIXME: move to test_serialization
    @onlyCUDA
    @deviceCountAtLeast(1)  # Note: Tests works with one but prefers more devices
    def test_serialization(self, devices):
        # [FUNCTION DOC] CUDA only. Saves and loads a tuple of tensors on different devices with
        # torch.save/torch.load, through both a named temporary file and an in-memory buffer, and
        # checks equality and the devices of the loaded tensors.
        def _test_serialization(filecontext_lambda):
            # [FUNCTION DOC] HELPER: Create a CUDA tensor on the current device and one on the
            # last device, save the tuple into the file context given by the factory, reload, and
            # compare values and devices.
            t0 = torch.cuda.FloatTensor(5).fill_(1)
            with torch.cuda.device(devices[-1]):
                tn = torch.cuda.FloatTensor(3).fill_(2)
            torch.cuda.set_device(devices[0])
            b = (t0, tn)
            with filecontext_lambda() as f:
                torch.save(b, f)
                f.seek(0)
                c = torch.load(f)
                self.assertEqual(b, c, atol=0, rtol=0)
                u0, un = c
                self.assertEqual(str(u0.device), devices[0])
                self.assertEqual(str(un.device), devices[-1])

        _test_serialization(tempfile.NamedTemporaryFile)
        _test_serialization(BytesIOContext)

    # FIXME: move memory format tests to their own test class/suite
    def test_memory_format_preserved_after_permute(self, device):
        # [FUNCTION DOC] Checks permuting a channels-last (4D) or channels-last-3d (5D) tensor
        # twice (swapping and swapping back) leaves it contiguous in that memory format.
        x = torch.randn(4, 3, 8, 8, device=device)
        nhwc = x.contiguous(memory_format=torch.channels_last)
        y = nhwc.permute(0, 1, 3, 2).permute(0, 1, 3, 2)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))

        x = torch.randn(4, 3, 8, 8, 8, device=device)
        ndhwc = x.contiguous(memory_format=torch.channels_last_3d)
        y = ndhwc.permute(0, 1, 4, 3, 2).permute(0, 1, 4, 3, 2)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last_3d))

    def test_memory_format_propagation_rules(self, device):
        # [FUNCTION DOC] Checks memory-format propagation in addition: for ambiguous, contiguous,
        # channels-last and bias tensors the result of adding two operands is contiguous in the
        # expected format, both before and after converting inputs; also checks that for ambiguous
        # tensors the result keeps the input strides for add with bias and scalar multiply.

        contiguous = torch.rand(10, 3, 5, 5, device=device)
        cl = torch.rand(10, 3, 5, 5, device=device).contiguous(memory_format=torch.channels_last)
        ambiguous = torch.rand(10, 3, 1, 1, device=device).contiguous(memory_format=torch.channels_last)
        self.assertTrue(ambiguous.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ambiguous.is_contiguous(memory_format=torch.contiguous_format))
        bias = torch.rand(1, 1, 1, 1, device=device).contiguous(memory_format=torch.channels_last)

        def _test_propagation_rules(self, contiguous, cl, ambiguous, bias):
            # [FUNCTION DOC] HELPER: For several (a, b, expected format) operand pairs, check that
            # the sum is contiguous in the expected memory format. (The first argument is the test
            # case itself.)
            options = ((ambiguous, contiguous, torch.contiguous_format),
                       (ambiguous, cl, torch.channels_last),
                       (contiguous, ambiguous, torch.contiguous_format),
                       (contiguous, cl, torch.contiguous_format),
                       (cl, ambiguous, torch.channels_last),
                       (cl, contiguous, torch.channels_last),
                       (bias, cl, torch.channels_last),
                       (cl, bias, torch.channels_last),)

            for a, b, mf in options:
                result = a + b
                self.assertTrue(result.is_contiguous(memory_format=mf))

        _test_propagation_rules(self, contiguous, cl, ambiguous, bias)

        cl = cl.to(memory_format=torch.channels_last)
        ambiguous = ambiguous.to(memory_format=torch.channels_last)
        bias = bias.to(memory_format=torch.channels_last)

        _test_propagation_rules(self, contiguous, cl, ambiguous, bias)

        # test cases when strides matter in ambiguous tensors
        for mf in (torch.channels_last, torch.contiguous_format):
            ambiguous = torch.rand(10, 3, 1, 1, device=device).to(memory_format=mf)
            bias = torch.rand(3, 1, 1, device=device)
            result = ambiguous + bias
            self.assertEqual(ambiguous.stride(), result.stride())
            result = bias + ambiguous
            self.assertEqual(ambiguous.stride(), result.stride())
            result = ambiguous * 5
            self.assertEqual(ambiguous.stride(), result.stride())

    def test_memory_format_empty_like(self, device):
        # [FUNCTION DOC] Checks empty_like with preserve_format, explicit memory format,
        # contiguous_format and default behavior for a channels-last 4D and a channels-last-3d 5D
        # tensor, and that a sparse tensor with preserve_format raises RuntimeError.
        def test_helper(x, memory_format):
            # [FUNCTION DOC] HELPER: For tensor x and a memory format, check the contiguity of
            # the results of empty_like under different memory_format arguments, and that sparse
            # input raises.
            xc = x.contiguous(memory_format=memory_format)

            like = torch.empty_like(xc, memory_format=torch.preserve_format)
            self.assertFalse(like.is_contiguous())
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            like_x = torch.empty_like(x, memory_format=torch.preserve_format)
            self.assertTrue(like_x.is_contiguous())
            self.assertFalse(like_x.is_contiguous(memory_format=memory_format))

            like = torch.empty_like(x, memory_format=memory_format)
            self.assertFalse(like.is_contiguous())
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            like = torch.empty_like(xc, memory_format=torch.contiguous_format)
            self.assertTrue(like.is_contiguous())
            self.assertFalse(like.is_contiguous(memory_format=memory_format))

            like = torch.empty_like(xc)
            self.assertFalse(like.is_contiguous())
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            sparse = x.to_sparse()
            with self.assertRaises(RuntimeError):
                torch.empty_like(sparse, memory_format=torch.preserve_format)

        test_helper(torch.randn(4, 3, 8, 8, device=device), torch.channels_last)
        test_helper(torch.randn(4, 3, 8, 8, 8, device=device), torch.channels_last_3d)

    def test_memory_format_consistency(self, device):
        # [FUNCTION DOC] Checks that an as_strided view with identical size and stride reports
        # the same size, stride and contiguity (default, channels-last and channels-last-3d) as
        # the original.
        x = torch.randn(10, 3, 1, 1, device=device)
        x_rep = x.as_strided(x.size(), x.stride())
        self.assertEqual(x.size(), x_rep.size())
        self.assertEqual(x.stride(), x_rep.stride())
        self.assertEqual(x.is_contiguous(), x_rep.is_contiguous())
        self.assertEqual(x.is_contiguous(memory_format=torch.channels_last), x_rep.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(
            x.is_contiguous(memory_format=torch.channels_last_3d), x_rep.is_contiguous(memory_format=torch.channels_last_3d))

    # FIXME: make this a elementwise unary and elementwise binary OpInfo test
    def test_memory_format_operators(self, device):
        # [FUNCTION DOC] Checks that many unary, binary and ternary tensor ops (listed as lambdas,
        # including in-place variants and some helper functions) give the same values on
        # channels-last (4D and 5D) inputs as on contiguous inputs, that outputs keep the
        # channels-last format, that bias-add keeps it, and that a few ops return
        # contiguous-format results.
        def _chunk_op(x, y):
            # [FUNCTION DOC] HELPER: Split x in two halves along dim 1 and return their sum.
            x1, x2 = x.chunk(2, dim=1)
            return x1 + x2

        def _unsqueeze_op_add(x, y):
            # [FUNCTION DOC] HELPER: Return the first batch element of x with a leading dimension
            # re-added, plus 3.
            return x[0].unsqueeze(0) + 3

        def _unsqueeze_op_clone(x, y):
            # [FUNCTION DOC] HELPER: Return a clone of the first batch element of x with a leading
            # dimension re-added.
            return x[0].unsqueeze(0).clone()

        def _test_helper(x, y, bias, memory_format):
            # [FUNCTION DOC] HELPER: Run all operator lambdas on a memory-format tensor and on
            # its contiguous copy, comparing values and checking the output format (in-place ops
            # are applied to clones; the source text of each function is inspected to detect
            # in-place names).
            return_contig_fns = [
                lambda x, y: y + x,
                lambda x, y: y * x,
                lambda x, y: y.addcdiv(x, y, value=2),
                lambda x, y: y.addcmul(x, y, value=2),
            ]
            bias_fns = [
                lambda x, b: x + b,
                lambda x, b: b + x,
            ]
            fns = [
                lambda x, y: x.clone(),
                lambda x, y: x + 3,
                lambda x, y: 3 * x,
                lambda x, y: x + y,
                lambda x, y: x * y,
                lambda x, y: abs(x),
                lambda x, y: x.abs(),
                lambda x, y: x.abs_(),
                lambda x, y: x.acos(),
                lambda x, y: x.acos_(),
                lambda x, y: x.add(y, alpha=3),
                lambda x, y: x.add_(y, alpha=3),
                lambda x, y: x.addcdiv(y, y, value=2),
                lambda x, y: x.addcdiv_(y, y, value=2),
                lambda x, y: x.addcmul(y, y, value=2),
                lambda x, y: x.addcmul_(y, y, value=2),
                lambda x, y: x.acosh(),
                lambda x, y: x.acosh_(),
                lambda x, y: x.asinh(),
                lambda x, y: x.asinh_(),
                lambda x, y: x.atanh(),
                lambda x, y: x.atanh_(),
                lambda x, y: x.asin(),
                lambda x, y: x.asin_(),
                lambda x, y: x.atan(),
                lambda x, y: x.atan2(y),
                lambda x, y: x.atan2_(y),
                lambda x, y: x.ceil(),
                lambda x, y: x.ceil_(),
                lambda x, y: x.clamp(-1, 1),
                lambda x, y: x.cos(),
                lambda x, y: x.cosh(),
                lambda x, y: x.div(0.5),
                lambda x, y: x.div_(0.5),
                lambda x, y: x.div(y),
                lambda x, y: x.div_(y),
                lambda x, y: x.digamma(),
                lambda x, y: x.digamma_(),
                lambda x, y: x.erf(),
                lambda x, y: x.erfc(),
                lambda x, y: x.erfinv(),
                lambda x, y: x.erfinv_(),
                lambda x, y: x.exp(),
                lambda x, y: x.expm1(),
                lambda x, y: x.expm1_(),
                lambda x, y: x.floor(),
                lambda x, y: x.floor_(),
                lambda x, y: x.fmod(2),
                lambda x, y: x.frac(),
                lambda x, y: x.hypot(y),
                lambda x, y: x.hypot_(y),
                lambda x, y: x.i0(),
                lambda x, y: x.i0_(),
                lambda x, y: x.lerp(y, 0.5),
                lambda x, y: x.log(),
                lambda x, y: x.log_(),
                lambda x, y: x.log10(),
                lambda x, y: x.log10_(),
                lambda x, y: x.log1p(),
                lambda x, y: x.log1p_(),
                lambda x, y: x.log2(),
                lambda x, y: x.log2_(),
                lambda x, y: x.mul(3),
                lambda x, y: x.mul_(3),
                lambda x, y: x.neg(),
                lambda x, y: x.neg_(),
                lambda x, y: x.pow(3),
                lambda x, y: x.pow_(3),
                lambda x, y: x.pow(0.0),
                lambda x, y: x.pow(1.0),
                lambda x, y: x.reciprocal(),
                lambda x, y: x.remainder(2),
                lambda x, y: x.round(),
                lambda x, y: x.round_(),
                lambda x, y: x.rsqrt(),
                lambda x, y: x.rsqrt_(),
                lambda x, y: x.sigmoid(),
                lambda x, y: x.sigmoid_(),
                lambda x, y: x.logit(),
                lambda x, y: x.logit_(),
                lambda x, y: x.logit(1e-6),
                lambda x, y: x.logit_(1e-6),
                lambda x, y: x.sign(),
                lambda x, y: x.sign_(),
                lambda x, y: x.sgn(),
                lambda x, y: x.sgn_(),
                lambda x, y: x.sin(),
                lambda x, y: x.sin_(),
                lambda x, y: x.sinh(),
                lambda x, y: x.sinh_(),
                lambda x, y: x.sqrt(),
                lambda x, y: x.sqrt_(),
                lambda x, y: x.tan(),
                lambda x, y: x.tanh(),
                lambda x, y: x.trunc(),
                lambda x, y: x.trunc_(),
                _chunk_op,
                _unsqueeze_op_add,
                _unsqueeze_op_clone,
            ]
            x_c = x.contiguous()
            y_c = y.contiguous()
            b_c = bias.contiguous()
            for fn in fns:
                is_inplace = '_(' in inspect.getsource(fn)
                x_clone = x.clone() if is_inplace else x
                x_c_clone = x_c.clone() if is_inplace else x_c
                result_c = fn(x_c_clone, y_c)
                result = fn(x_clone, y)
                self.assertEqual(result, result_c, lambda msg: f"{msg}\nFailed for '{inspect.getsource(fn).strip()}'")
                self.assertTrue(
                    result.is_contiguous(memory_format=memory_format),
                    lambda msg: f"{msg}\nresult of the '{inspect.getsource(fn).strip()}' is not in '{memory_format}' format")

            for fn in bias_fns:
                result_c = fn(x_c, b_c)
                result = fn(x, bias)
                self.assertEqual(result, result_c, lambda msg: f"{msg}\nFailed for '{inspect.getsource(fn).strip()}'")
                self.assertTrue(
                    result.is_contiguous(memory_format=memory_format),
                    lambda msg: f"{msg}\nresult of the '{inspect.getsource(fn).strip()}' is not in '{memory_format}' format")

            for fn in return_contig_fns:
                result_c = fn(x_c, y_c)
                result = fn(x, y)
                self.assertEqual(result, result_c, lambda msg: f"{msg}\nFailed for '{inspect.getsource(fn).strip()}'")
                self.assertTrue(
                    result.is_contiguous(memory_format=torch.contiguous_format),
                    lambda msg: f"{msg}\nresult of the '{inspect.getsource(fn).strip()}' is not in '{torch.contiguous_format}' format")

        _test_helper(
            torch.randn((4, 3, 8, 8), device=device).contiguous(memory_format=torch.channels_last),
            abs(torch.randn((4, 3, 8, 8), device=device)) + 1,
            torch.randn((1, 3, 1, 1), device=device).contiguous(memory_format=torch.channels_last),
            torch.channels_last)
        _test_helper(
            torch.randn((4, 3, 8, 8, 8), device=device).contiguous(memory_format=torch.channels_last_3d),
            abs(torch.randn((4, 3, 8, 8, 8), device=device)) + 1,
            torch.randn((1, 3, 1, 1, 1), device=device).contiguous(memory_format=torch.channels_last_3d),
            torch.channels_last_3d)

    # FIXME: make this a elementwise unary and elementwise binary OpInfo test
    def test_strides_propagation(self, device):
        # [FUNCTION DOC] Checks stride propagation of results for a binary op that defines its
        # output up front (eq) and one that does not (add), plus a unary op (exp), over every
        # permutation of memory-dense, sliced and ambiguous-sliced inputs: both the functional
        # result and the out= result must have memory-dense strides matching the input permutation
        # (after dividing by the last-dim stride) and the input size.
        def _test_helper(x, op, unary=False):
            # [FUNCTION DOC] HELPER: For one input tensor and op, loop over all dimension
            # permutations, and for binary ops over combinations of the permuted tensor and a
            # broadcast vector, and compare strides and sizes of results (with and without a
            # preallocated out tensor).
            def compare_strides(s1, s2, div):
                # [FUNCTION DOC] HELPER: Divide every entry of the first stride list by the given
                # divisor and assert it equals the second list.
                sdiv = [s // div for s in s1]
                self.assertEqual(sdiv, s2)

            dim = x.dim()
            # we produce memory dense outputs, so when input is strided on the last dimension
            # we need to divide by that dimension stride to compare input and result strides
            div = x.stride(-1)
            for p in permutations(range(dim)):
                xp = x.permute(p)
                if not unary:
                    y = torch.randn(xp.size(-1), device=x.device, dtype=x.dtype)
                    for inputs in ((xp, xp), (xp, y), (y, xp)):
                        res = op(*inputs)
                        compare_strides(xp.stride(), res.stride(), div)
                        self.assertEqual(xp.size(), res.size())
                        out = torch.empty(0, device=xp.device, dtype=res.dtype)
                        res = op(*inputs, out=out)
                        compare_strides(xp.stride(), res.stride(), div)
                        self.assertEqual(xp.size(), res.size())
                else:
                    res = op(xp)
                    compare_strides(xp.stride(), res.stride(), div)
                    self.assertEqual(xp.size(), res.size())
                    out = torch.empty(0, device=xp.device, dtype=res.dtype)
                    res = op(xp, out=out)
                    compare_strides(xp.stride(), res.stride(), div)
                    self.assertEqual(xp.size(), res.size())

        # torch.eq by default calls TensorIterator with defined output, torch.add with undefined
        binary_ops = (torch.eq, torch.add)
        unary_ops = (torch.exp,)
        # memory dense, sliced and ambiguous sliced (ambiguous dense loses permutation information)
        xs = (torch.randn(2, 3, 4, device=device), torch.randn(2, 3, 8, device=device)[:, :, ::2],
              torch.randn(1, 1, 4, 12, device=device)[:, :, :, ::2])
        for op in binary_ops:
            for x in xs:
                _test_helper(x, op)
        for op in unary_ops:
            for x in xs:
                _test_helper(x, op, unary=True)

    @onlyCUDA
    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    @skipIfTorchDynamo("NotImplementedError: PrimTorch does not support pinned memory")
    def test_pin_memory_from_constructor(self, device):
        # [FUNCTION DOC] CUDA only. Checks the pin_memory option of tensor factories (tensor,
        # randn, rand, zeros, randperm, empty, ones, eye, arange) and of the _like factories gives
        # pinned tensors, and that the same factories without it, including _like of a pinned
        # source, give unpinned tensors. Skipped under Dynamo and CUDA memcheck.
        def _get_like(t, **kwargs):
            # [FUNCTION DOC] HELPER: Return a list of the six _like factory results (rand, randn,
            # empty, full with 4, zeros, ones) for tensor t, forwarding the keyword arguments.
            return [
                torch.rand_like(t, **kwargs),
                torch.randn_like(t, **kwargs),
                torch.empty_like(t, **kwargs),
                torch.full_like(t, 4, **kwargs),
                torch.zeros_like(t, **kwargs),
                torch.ones_like(t, **kwargs),
            ]

        def _get_tensors(**kwargs):
            # [FUNCTION DOC] HELPER: Return a list of nine tensors created by different factory
            # functions (tensor, randn, rand, zeros, randperm, empty, ones, eye, arange),
            # forwarding the keyword arguments.
            return [
                torch.tensor([10, 11], **kwargs),
                torch.randn(3, 5, **kwargs),
                torch.rand(3, **kwargs),
                # torch.randint(3, 5, **kwargs), // unsupported
                torch.zeros(3, **kwargs),
                torch.randperm(3, **kwargs),
                torch.empty(6, **kwargs),
                torch.ones(6, **kwargs),
                torch.eye(6, **kwargs),
                torch.arange(3, 5, **kwargs)]

        pinned_tensors = _get_tensors(pin_memory=True) + _get_like(torch.empty(5, dtype=torch.float64), pin_memory=True)
        for x in pinned_tensors:
            self.assertTrue(x.is_pinned())

        tensors = _get_tensors() + _get_like(torch.empty(5, dtype=torch.float64, pin_memory=True))
        for x in tensors:
            self.assertFalse(x.is_pinned())

    @deviceCountAtLeast(1)
    @onlyCUDA
    @parametrize("non_blocking", (True, False))
    def test_storage_all_devices(self, devices, non_blocking):
        # [FUNCTION DOC] Needs a CUDA device; parametrized by non_blocking. For each device copies
        # a tensor's untyped storage to CPU and checks the storage dtype, that the copy is pinned
        # only when non-blocking is requested (after synchronizing), and that a tensor wrapping
        # the copy equals the original.
        for device in devices:
            t = torch.randn(6, device=device)
            self.assertEqual(t.dtype, t.storage().dtype)
            s = t.untyped_storage()
            s_cpu = s.to(device='cpu', non_blocking=non_blocking)
            if non_blocking:
                torch.cuda.synchronize()
                self.assertTrue(s_cpu.is_pinned())
            else:
                self.assertFalse(s_cpu.is_pinned())
            t_cpu = torch.empty(()).set_(s_cpu)
            self.assertEqual(t.cpu(), t_cpu)

    # Note [lazy_clone_ tests with inductor enabled]
    # These `lazy_clone_` tests are written in a way that makes them pass in
    # both eager mode and compiled mode (`PYTORCH_TEST_WITH_INDUCTOR=1`). There
    # are cases where COW tensors can materialize at different times and in
    # different ways in compiled mode versus eager mode, and those cases need to
    # be avoided. There are two main wrinkles the be aware of.
    #
    # The first wrinkle is that these tests have to check the internal
    # properties of tensors to make sure they materialize in the expected way,
    # and those checks cause dynamo graph breaks. Depending on the situation, a
    # graph break in-between two compiled graphs that operate on the same COW
    # tensor can make the tensor materialize when it would not materialize in
    # eager mode, causing the checks to fail. The strategy for avoiding this is
    # to make all the operations on COW tensors get compiled into the same
    # graph, by not doing any checks between the operations, and just do all the
    # checks at the end of the test. If we really do want to perform checks
    # between two operations, `op1` and `op2`, the solution is to create two
    # different tests. One test performs just `op1` and then checks. The other
    # test performs `op1` followed immediately by `op2` and then checks.
    #
    # The second wrinkle is that in eager mode, if we perform writes on two COW
    # tensors where one is a lazy clone of the other, the first tensor to be
    # written will be materialized with a new data pointer, and the second
    # tensor will just reuse the original data pointer when it is materialized.
    # But in compiled mode, if these writes happen in the same graph, the order
    # in which the tensors materialize can be different than in eager mode. So
    # in this case the strategy is to purposefully cause a graph break to happen
    # in-between the two write operations, by adding checks between them, so
    # that they have to materialize in the expected order.
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_const_data_ptr(self, device, dtype):
        # [FUNCTION DOC] Checks const_data_ptr equals data_ptr for a regular tensor, and that
        # after a lazy clone both tensors stay copy-on-write (no materialization) when reading the
        # pointer of the clone, which equals that of the original.
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)

        # For a regular tensor, const_data_ptr and data_ptr return the same address
        self.assertEqual(t.const_data_ptr(), t.data_ptr())

        clone = t._lazy_clone()

        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(clone))

        # const_data_ptr should not trigger COW materialization
        addr = clone.const_data_ptr()
        self.assertEqual(addr, t.const_data_ptr())

        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(clone))

    # See Note [lazy_clone_ tests with inductor enabled]
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone(self, device, dtype):
        # [FUNCTION DOC] Checks that a lazy clone makes both the original and clone copy-on-write,
        # with different storage objects but the same data pointer.
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        clone = t._lazy_clone()

        # Lazy cloning a tensor should cause both it and its clone to become COW
        # tensors. They should have different storages, but the same data
        # pointer.

        self.assertTrue(torch._C._is_cow_tensor(clone))
        self.assertTrue(torch._C._is_cow_tensor(t))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        self.assertTrue(torch._C._data_address(t) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

    # See Note [lazy_clone_ tests with inductor enabled]
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_view(self, device, dtype):
        # [FUNCTION DOC] Checks that viewing a lazily cloned tensor does not materialize: all
        # three tensors remain copy-on-write, the view shares the original storage, the clone has
        # a different storage, and all share the data pointer.
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        clone = t._lazy_clone()
        view = t.view([4])

        # Viewing `t` should not cause a copy (materialize) to happen. All the
        # tensors should still be COW and have the same data pointer. `view` and
        # `t` should have the same storage, and `clone` should have a different
        # storage.

        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(view))
        self.assertTrue(torch._C._is_cow_tensor(clone))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        self.assertTrue(torch._C._data_address(t) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(view) == orig_data_ptr)

    # See Note [lazy_clone_ tests with inductor enabled]
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_view_materialize(self, device, dtype):
        # [FUNCTION DOC] Checks materialization order: an in-place add on a view of the original
        # materializes original and view (new data pointer, same storage object) while the clone
        # stays copy-on-write; then writing the clone materializes it without allocating, reusing
        # the original data pointer.
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        clone = t._lazy_clone()
        view = t.view([4])
        view += torch.ones(1, device=device, dtype=dtype)

        # Writing to `t` should cause the storage under `t` and `view` to be
        # copied (materialized), but should not affect `clone`.

        self.assertFalse(torch._C._is_cow_tensor(t))
        self.assertFalse(torch._C._is_cow_tensor(view))
        self.assertTrue(torch._C._is_cow_tensor(clone))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        t_new_data_addr = torch._C._data_address(t)
        self.assertTrue(t_new_data_addr != orig_data_ptr)
        self.assertTrue(torch._C._data_address(view) == t_new_data_addr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

        clone += torch.ones(1, device=device, dtype=dtype)

        # Writing to `clone` should materialize it, so it should no longer
        # be COW. However, since `clone`'s storage is the only COW storage
        # left that holds a reference to the original data pointer, this
        # materialization should not actually cause a copy--it should
        # just reuse the original data pointer.

        self.assertFalse(torch._C._is_cow_tensor(t))
        self.assertFalse(torch._C._is_cow_tensor(view))
        self.assertFalse(torch._C._is_cow_tensor(clone))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        self.assertTrue(torch._C._data_address(t) == t_new_data_addr)
        self.assertTrue(torch._C._data_address(view) == t_new_data_addr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_binary_op_no_materialize(self, device, dtype):
        # [FUNCTION DOC] Checks that adding a tensor to its own lazy clone does not materialize
        # either; both remain copy-on-write.
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        clone = t._lazy_clone()
        t + clone
        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(clone))

    # This tests that if a COW materialization is attempted inside an
    # `at::parallel_for` loop function, then an error is raised. This test is
    # implemented in Python rather than C++ because the C++ tests are built
    # without multithreading support in `at::parallel_for`.
    @skipXLA
    @skipIfTorchDynamo("Torchdynamo fails and we do not need to test it here anyway")
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_parallel_cow_materialize_error(self, device, dtype):
        # [FUNCTION DOC] Checks that materializing a copy-on-write tensor inside a parallel loop
        # raises RuntimeError about materializing a storage, for combinations of thread counts,
        # parallel counts and a skip-first flag, and that non-copy-on-write tensors never raise.
        # Skipped under XLA and Dynamo.

        def run(num_threads, num_parallel, skip_first, should_error):
            # [FUNCTION DOC] HELPER: Set the number of threads, create a lazily cloned tensor
            # and call the parallel materialization test op expecting an error or not; also run
            # it on a normal tensor without errors. Restores the original thread count in a
            # finally clause.
            orig_num_threads = torch.get_num_threads()

            try:
                torch.set_num_threads(num_threads)

                a = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)._lazy_clone()

                if should_error:
                    with self.assertRaisesRegex(RuntimeError, r'Materializing a storage'):
                        torch._test_parallel_materialize(
                            a, num_parallel, skip_first)
                else:
                    torch._test_parallel_materialize(a, num_parallel, skip_first)

                # Error should not raise in any case if the tensor is not COW
                b = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
                torch._test_parallel_materialize(b, num_parallel, skip_first)

            finally:
                torch.set_num_threads(orig_num_threads)

        run(1, 1, False, True)
        run(1, 1, True, False)
        run(1, 10, False, True)
        run(1, 10, True, True)
        run(10, 1, False, True)
        run(10, 1, True, False)
        run(10, 10, False, True)
        run(10, 10, True, True)
        run(10, 2, False, True)
        run(10, 2, True, True)

    # FIXME: move to test distributions
    @dtypesIfCUDA(torch.float, torch.double, torch.half)
    @dtypes(torch.float, torch.double, torch.half)
    def test_multinomial(self, device, dtype):
        # [FUNCTION DOC] Checks torch.multinomial sampling for float, double and half, contiguous
        # and non-contiguous probability tensors: with replacement and without replacement never
        # returns an index of zero probability, without replacement never repeats an index, shapes
        # are right for matrices and vectors, and a small misalignment-regression case (issue
        # 46702) works.
        def make_prob_dist(shape, is_contiguous):
            # [FUNCTION DOC] HELPER: Build a random probability tensor of the given shape (1D or
            # 2D), either contiguous or a non-contiguous view obtained by slicing/transposing a
            # larger tensor, handling half precision by converting after generation; raises
            # AssertionError if the non-contiguous 2D result is contiguous.
            if is_contiguous:
                if dtype == torch.half:
                    return torch.zeros(shape, device=device).uniform_().to(dtype=torch.half)
                return torch.zeros(shape, device=device, dtype=dtype).uniform_()
            elif len(shape) == 1:
                if dtype == torch.half:
                    return torch.zeros((shape + [5]), device=device).uniform_().to(dtype=torch.half)[:, 2]
                return torch.zeros((shape + [5]), device=device, dtype=dtype).uniform_()[:, 2]
            else:
                # num dim = 2
                new_shape = [2, shape[1], 7, 1, shape[0], 1, 10]
                if dtype == torch.half:
                    prob_dist = torch.zeros(new_shape, device=device).uniform_().to(dtype=torch.half)
                else:
                    prob_dist = torch.zeros(new_shape, device=device, dtype=dtype).uniform_()
                prob_dist = prob_dist.transpose(1, 4)
                prob_dist = prob_dist[1, :, 5, 0, :, 0, 4]
                if prob_dist.is_contiguous():
                    raise AssertionError("expected prob_dist to be non-contiguous")
                return prob_dist

        for is_contiguous in (True, False):
            # with replacement
            n_row = 3
            for n_col in range(4, 5 + 1):
                prob_dist = make_prob_dist([n_row, n_col], is_contiguous)
                # indices that shouldn't be sampled (<0 means none)
                zero_prob_indices = torch.LongTensor(n_row).random_(-2, n_col).tolist()
                for i, j in enumerate(zero_prob_indices):
                    if j >= 0:
                        prob_dist[i, j] = 0
                n_sample = n_col * 3
                sample_indices = torch.multinomial(prob_dist, n_sample, True)
                self.assertEqual(prob_dist.dim(), 2)
                self.assertEqual(sample_indices.size(1), n_sample)
                for i in range(n_row):
                    zero_prob_idx = zero_prob_indices[i]
                    if zero_prob_idx < 0:
                        continue
                    for j in range(n_sample):
                        self.assertNotEqual(sample_indices[i, j], zero_prob_idx,
                                            msg="sampled an index with zero probability")

            # without replacement
            n_row = 3
            for n_col in range(2, 10 + 1, 2):
                prob_dist = make_prob_dist([n_row, n_col], is_contiguous)
                # indices that shouldn't be sampled (<0 means none)
                zero_prob_indices = torch.LongTensor(n_row).random_(-1, n_col).tolist()
                for i, j in enumerate(zero_prob_indices):
                    if j >= 0:
                        prob_dist[i, j] = 0
                n_sample = max(1, n_col - 2)
                sample_indices = torch.multinomial(prob_dist, n_sample, False)
                self.assertEqual(prob_dist.dim(), 2)
                self.assertEqual(sample_indices.size(1), n_sample)
                for i in range(n_row):
                    row_samples = {}
                    zero_prob_idx = zero_prob_indices[i]
                    for j in range(n_sample):
                        sample_idx = sample_indices[i, j]
                        if zero_prob_idx >= 0:
                            self.assertNotEqual(sample_idx, zero_prob_idx,
                                                msg="sampled an index with zero probability")
                        self.assertNotIn(sample_idx, row_samples, "sampled an index twice")
                        row_samples[sample_idx] = True

            # vector
            n_col = 4
            prob_dist = make_prob_dist([n_col], is_contiguous).fill_(1)
            zero_prob_idx = 1  # index that shouldn't be sampled
            prob_dist[zero_prob_idx] = 0
            n_sample = 20
            sample_indices = torch.multinomial(prob_dist, n_sample, True)
            for sample_index in sample_indices:
                self.assertNotEqual(sample_index, zero_prob_idx, msg="sampled an index with zero probability")
            self.assertEqual(sample_indices.dim(), 1, msg="wrong number of dimensions")
            self.assertEqual(prob_dist.dim(), 1, msg="wrong number of prob_dist dimensions")
            self.assertEqual(sample_indices.size(0), n_sample, msg="wrong number of samples")

        # CUDA misalignment issue (#46702)
        n_row, n_col = 2, 3
        prob_dist = make_prob_dist([n_row, n_col], True)
        n_sample = 1
        sample_indices = torch.multinomial(prob_dist, n_sample, True)
        self.assertEqual(sample_indices.dim(), 2, msg="wrong number of dimensions")
        self.assertEqual(sample_indices.size(1), n_sample, msg="wrong number of samples")

    # FIXME: move to test distributions
    @onlyCUDA
    @dtypes(torch.float, torch.double, torch.half)
    def test_multinomial_deterministic(self, device, dtype):
        # [FUNCTION DOC] CUDA only. Seeds a generator identically twice and checks multinomial
        # on a large probability matrix returns the same samples, with the expected shape, over
        # five trials.
        gen = torch.Generator(device=device)

        trials = 5
        seed = 0
        prob_dist = torch.rand(10000, 1000, device=device, dtype=dtype)
        n_sample = 1

        for _ in range(trials):
            gen.manual_seed(seed)
            samples_1 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

            gen.manual_seed(seed)
            samples_2 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

            self.assertEqual(samples_1, samples_2)
            self.assertEqual(samples_1.dim(), 2, msg="wrong number of dimensions")
            self.assertEqual(samples_1.size(1), n_sample, msg="wrong number of samples")

    # FIXME: move to test distributions
    @slowTest
    @dtypes(torch.float)
    def test_multinomial_rng_state_advance(self, device, dtype):
        # [FUNCTION DOC] Slow. Draws 100 samples twice from a 100000-element uniform distribution,
        # with and without replacement, and checks that few values repeat across the two draws
        # (the RNG state must advance between calls).
        corpus_size = 100000
        freqs = torch.ones(corpus_size, dtype=torch.float, device=device)
        n_sample = 100
        samples1 = torch.multinomial(freqs, n_sample, replacement=True)
        samples2 = torch.multinomial(freqs, n_sample, replacement=True)
        samples = torch.cat([samples1, samples2])
        # expect no more than 1 repeating elements generated in 2 attempts
        # the probability of at least element being repeated is surprisingly large, 18%
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 2)
        samples1 = torch.multinomial(freqs, n_sample, replacement=False)
        samples2 = torch.multinomial(freqs, n_sample, replacement=False)
        samples = torch.cat([samples1, samples2])
        # expect no more than 1 repeating elements generated in 2 attempts
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 1)

    def _test_memory_format_transformations(self, device, input_generator_fn, transformation_fn,
                                            memory_format, compare_data=True, default_is_preserve=False):
        # [FUNCTION DOC] HELPER: Shared helper for memory-format tests. Given an input generator,
        # a transformation callable and a channels-last or channels-last-3d format (anything else
        # raises AssertionError), checks the transformation with preserve_format keeps the format
        # on a strided sliced input (result not contiguous but in that format), with
        # contiguous_format returns a plain contiguous tensor, and with the default follows
        # default_is_preserve; optionally compares data, and checks that stride permutations are
        # preserved on a random 7D tensor (except under inductor).

        if memory_format not in (torch.channels_last, torch.channels_last_3d):
            raise AssertionError(f"unexpected memory_format: {memory_format}")

        # xc is a channels last tensor
        xc = input_generator_fn(device)
        # xc is not memory dense, but looks like channels last
        # We don't preserve non-dense striding
        if not TEST_WITH_TORCHINDUCTOR:
            if memory_format == torch.channels_last:
                xc = xc[..., ::2, ::2]
            else:
                xc = xc[..., ::2, ::2, ::2]

        clone = transformation_fn(xc, memory_format=torch.preserve_format)


        self.assertFalse(clone.is_contiguous())
        self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        if not TEST_WITH_TORCHINDUCTOR:
            self.assertFalse(xc.is_contiguous())
            self.assertFalse(xc.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        self.assertTrue(clone.is_contiguous())
        self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc)

        if default_is_preserve:
            self.assertFalse(clone.is_contiguous())
            self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        else:
            self.assertTrue(clone.is_contiguous())
            self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        # TODO copy _like constructors to stride permutation instead of just layout
        if not TEST_WITH_TORCHINDUCTOR:
            x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=device)
            for _ in range(10):
                permutation = list(range(len(x.shape)))
                random.shuffle(permutation)
                x = x.permute(permutation)
                self.assertEqual(x.stride(), transformation_fn(x, memory_format=torch.preserve_format).stride())

    def test_memory_format_to(self, device):
        # [FUNCTION DOC] Runs the memory-format transformation helper for the dtype conversion
        # method to float64 with channels-last and channels-last-3d inputs, expecting preserve
        # behavior by default.
        def get_generator(memory_format, shape):
            # [FUNCTION DOC] HELPER: Return an input generator that builds a float32 random tensor
            # of the given shape in the given memory format.
            def input_generator_fn(device):
                # [FUNCTION DOC] HELPER: Create a random float32 tensor of the captured shape on
                # the device in the captured memory format.
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            # [FUNCTION DOC] HELPER: Convert the tensor to float64 forwarding keyword arguments
            # (such as the memory format).
            return tensor.to(dtype=torch.float64, **kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, default_is_preserve=True)

    def test_memory_format_type(self, device):
        # [FUNCTION DOC] Same as the memory-format-to test but the conversion is expressed through
        # the dtype-positional form of the same method.
        def get_generator(memory_format, shape):
            # [FUNCTION DOC] HELPER: Return an input generator that builds a float32 random tensor
            # of the given shape in the given memory format.
            def input_generator_fn(device):
                # [FUNCTION DOC] HELPER: Create a random float32 tensor of the captured shape on
                # the device in the captured memory format.
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            # [FUNCTION DOC] HELPER: Convert the tensor to float64 (positional dtype) forwarding
            # keyword arguments.
            return tensor.to(torch.float64, **kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, default_is_preserve=True)

    def test_memory_format_clone(self, device):
        # [FUNCTION DOC] Runs the memory-format transformation helper for Tensor.clone with data
        # comparison, expecting preserve behavior by default.
        def get_generator(memory_format, shape):
            # [FUNCTION DOC] HELPER: Return an input generator that builds a float32 random tensor
            # of the given shape in the given memory format.
            def input_generator_fn(device):
                # [FUNCTION DOC] HELPER: Create a random float32 tensor of the captured shape on
                # the device in the captured memory format.
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            # [FUNCTION DOC] HELPER: Clone the tensor forwarding keyword arguments.
            return tensor.clone(**kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, True, default_is_preserve=True)

    def test_memory_format_factory_like_functions_preserve(self, device):
        # [FUNCTION DOC] Runs the memory-format transformation helper (without comparing data) for
        # the zeros_like, ones_like, randint_like (two forms), randn_like, rand_like, full_like
        # and empty_like factories, expecting preserve behavior by default.
        def get_generator(memory_format, shape):
            # [FUNCTION DOC] HELPER: Return an input generator that builds a float32 random tensor
            # of the given shape in the given memory format.
            def input_generator_fn(device):
                # [FUNCTION DOC] HELPER: Create a random float32 tensor of the captured shape on
                # the device in the captured memory format.
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        transformation_fns = [
            lambda t, **kwargs: torch.zeros_like(t, **kwargs),
            lambda t, **kwargs: torch.ones_like(t, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 10, 100, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 100, **kwargs),
            lambda t, **kwargs: torch.randn_like(t, **kwargs),
            lambda t, **kwargs: torch.rand_like(t, **kwargs),
            lambda t, **kwargs: torch.full_like(t, 7, **kwargs),
            lambda t, **kwargs: torch.empty_like(t, **kwargs)]

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape, in formats_shapes:
            for transformation_fn in transformation_fns:
                self._test_memory_format_transformations(
                    device, get_generator(mf, shape), transformation_fn, mf, compare_data=False, default_is_preserve=True)

    def test_memory_format_type_shortcuts(self, device):
        # [FUNCTION DOC] Runs the memory-format transformation helper for the dtype shortcut
        # methods (byte, char, double, bool, half, int, long, short, plus bfloat16 on CPU) on
        # rounded 0/1 inputs, and for float using float64 inputs to avoid a no-op conversion.
        def get_generator(memory_format, shape, dtype):
            # [FUNCTION DOC] HELPER: Return an input generator producing a random tensor of the
            # given dtype clamped to 0..1, rounded and put in the given memory format.
            def input_generator_fn(device):
                # [FUNCTION DOC] HELPER: Create a rounded random tensor of the captured shape and
                # dtype in the captured memory format.
                return torch.randn(shape, device=device, dtype=dtype).clamp(0, 1) \
                    .round().contiguous(memory_format=memory_format)
            return input_generator_fn


        def get_fn(fn_name):
            # [FUNCTION DOC] HELPER: Return a transformation function that calls the tensor method
            # named fn_name with keyword arguments.
            def transformation_fn(tensor, **kwargs):
                # [FUNCTION DOC] HELPER: Look up the method named by the enclosing name on the
                # tensor and call it forwarding keyword arguments.
                fn = getattr(tensor, fn_name)
                return fn(**kwargs)
            return transformation_fn

        shortcuts = ['byte', 'char', 'double', 'bool', 'half', 'int', 'long', 'short']
        if device == 'cpu':
            shortcuts += ['bfloat16']

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            for fn_name in shortcuts:
                self._test_memory_format_transformations(
                    device, get_generator(mf, shape, torch.float32), get_fn(fn_name), mf, default_is_preserve=True)

        # Test 'float' separately to avoid float->float no-op.
        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape, torch.float64), get_fn('float'), mf, default_is_preserve=True)

    @onlyCUDA
    def test_memory_format_cpu_and_cuda_ops(self, device):
        # [FUNCTION DOC] CUDA only. Runs the memory-format transformation helper for moving
        # channels-last tensors with the cpu and cuda methods (CUDA inputs to CPU and CPU inputs
        # to CUDA), expecting preserve behavior.
        def get_generator(memory_format, shape):
            # [FUNCTION DOC] HELPER: Return an input generator that builds a float32 random tensor
            # of the given shape in the given memory format.
            def input_generator_fn(device):
                # [FUNCTION DOC] HELPER: Create a random float32 tensor of the captured shape on
                # the device in the captured memory format.
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_cpu_fn(tensor, **kwargs):
            # [FUNCTION DOC] HELPER: Move the tensor to CPU forwarding keyword arguments.
            return tensor.cpu(**kwargs)

        def transformation_cuda_fn(tensor, **kwargs):
            # [FUNCTION DOC] HELPER: Move the tensor to CUDA forwarding keyword arguments.
            return tensor.cuda(**kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                'cuda', get_generator(mf, shape), transformation_cpu_fn, mf, default_is_preserve=True)
            self._test_memory_format_transformations(
                'cpu', get_generator(mf, shape), transformation_cuda_fn, mf, default_is_preserve=True)

    # FIXME: move to test_serialization
    @onlyNativeDeviceTypes
    def test_pickle_gradscaler(self, device):
        # [FUNCTION DOC] Checks that GradScaler objects can be pickled and unpickled on native
        # devices, with and without lazily initialized scale; when enabled, the restored object
        # keeps init scale 3, growth factor 4, backoff factor 0.5, growth interval 2, growth
        # tracker 0, a working default per-optimizer state, and scales values correctly.
        # This test should pass in 3 cases for cuda:
        #  1. cuda is not available.
        #  2. cuda is available but device is not cuda.
        #  3. cuda is available and device is cuda.
        # In case 1, a and b disable themselves on construction and shouldn't try to pickle workhorse attributes.
        # In case 2, a and b are enabled.  Workhorse attributes participate in pickling, but none are lazy-inited
        # to cuda Tensors, because I don't want to do cuda things if device is not cuda.
        # In case 3, a and b are enabled and we may also try lazy-initing _scale to a cuda tensor.
        device = torch.device(device)
        try_lazy_inits = (True, False)
        GradScaler = partial(torch.GradScaler, device=device.type)
        for lazy_init_scale in try_lazy_inits:
            a = GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
            if device.type == "cuda":
                self.assertTrue(not a.is_enabled() if torch.cuda.amp.common.amp_definitely_not_available() else a.is_enabled())
            else:
                self.assertTrue(a.is_enabled())
            if lazy_init_scale:
                # Dummy a.scale() call lazy-inits a._scale Tensor.
                a.scale(torch.tensor([4.0], dtype=torch.float32, device=device))
                self.assertTrue(a._scale.device.type == device.type)
            # The following three lines should work whether or not cuda is available.
            serialized = pickle.dumps(a)
            b = pickle.loads(serialized)
            self.assertEqual(b.is_enabled(), a.is_enabled())
            if a.is_enabled():
                self.assertEqual(b.get_scale(), 3.)
                self.assertEqual(b.get_growth_factor(), 4.)
                self.assertEqual(b.get_backoff_factor(), .5)
                self.assertEqual(b.get_growth_interval(), 2)
                self.assertEqual(b._init_growth_tracker, 0)
                # supplies a dummy key to test the defaultdict's default_factory
                self.assertEqual(b._per_optimizer_states["fdsa"],
                                 torch.amp.grad_scaler._refresh_per_optimizer_state())
                if lazy_init_scale:
                    self.assertEqual(b.scale(torch.tensor([4.0], dtype=torch.float32, device=device)), 12.0)

    # FIXME: move to test distributions
    def _test_multinomial_empty(self, device, replacement, num_samples):
        # [FUNCTION DOC] HELPER: Check multinomial on a zero-row probability matrix returns an
        # empty int64 result with the requested number of columns.
        probs = torch.ones(0, 3, device=device)
        expected = torch.empty(0, num_samples, dtype=torch.int64)
        out = torch.multinomial(probs, num_samples=num_samples, replacement=replacement)
        self.assertEqual(out, expected)

    # FIXME: move to test distributions
    def test_multinomial_empty_w_replacement(self, device):
        # [FUNCTION DOC] Checks multinomial on empty input with replacement for one and
        # two samples.
        self._test_multinomial_empty(device, True, 1)
        self._test_multinomial_empty(device, True, 2)

    # FIXME: move to test distributions
    def test_multinomial_empty_wo_replacement(self, device):
        # [FUNCTION DOC] Checks multinomial on empty input without replacement for one and
        # two samples.
        self._test_multinomial_empty(device, False, 1)
        self._test_multinomial_empty(device, False, 2)

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    def test_grad_scaling_unscale(self, device, dtype):
        # [FUNCTION DOC] Checks the unscale op of the amp foreach kernel on float and double
        # grads: for lists of contiguous, transposed and sliced grads, with and without an
        # injected inf or nan, grads are multiplied by the inverse scale and the found-inf flag is
        # set only when a non-finite value exists. Also checks mixed dtype lists fall back to a
        # per-tensor path, mismatched devices raise RuntimeError (multi-GPU CUDA), and GradScaler
        # unscale groups grads by dtype and device and finds an inf injected into any of them.
        device = torch.device(device)
        device0 = "cuda:0" if device.type == "cuda" else "cpu"
        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device=device0)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device=device0)

        size = 20
        g = torch.full((size, size), 4.0, dtype=dtype, device=device0)
        ginf = g.clone()
        ginf[2, 2] = float('inf')
        gnan = g.clone()
        gnan[2, 2] = float('nan')

        # Tries selected combinations of
        #  - contiguous grads
        #  - g.clone().t() which is not contiguous but still non overlapping and dense
        #  - variants of g.clone()[:, :5] which are not non overlapping and dense
        # Non overlapping and dense grads route into a multi tensor apply kernel,
        # others use a fallback per-tensor kernel, so we should try both.
        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            if has_inf:
                self.assertEqual(found_inf, 1.0)
            else:
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        # When passing lists with mismatched dtypes to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call on CUDA,
        # it's expected to fall back to single-tensor TensorIterator kernel.
        grads = [g.clone(), g.to(dtype=torch.float16)]
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
        for grad in grads:
            self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        # Passing lists with mismatched devices to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call should raise errors.
        if device.type == "cuda" and TEST_MULTIGPU:
            with self.assertRaisesRegex(RuntimeError, r"Expected all tensors to be on the same device"):
                torch._amp_foreach_non_finite_check_and_unscale_([g.clone(), g.to(device="cuda:1")],
                                                                 found_inf,
                                                                 inv_scale)

        # Creates a list of grads with mismatched dtypes and devices, to ensure
        # scaler._unscale_grads_ organizes grads by dtype and device before calling
        # _amp_foreach_non_finite_check_and_unscale_ on each set.
        # If inject_inf >= 0, writes an inf into one grad for _unscale_grads_ to find.
        def perfect_storm_grads(inject_inf):
            # [FUNCTION DOC] HELPER: Build a list of cloned gradients with mixed layouts, dtypes
            # (float and half) and devices (a second GPU when available), writing an infinity into
            # one of them if the given index is non-negative.
            grads = [g.clone(), g.clone()[:, :5], g.to(dtype=torch.float16), g.to(dtype=torch.float16)]
            if device.type == "cuda" and TEST_MULTIGPU:
                grads += [g.to(device="cuda:1"),
                          g.to(device="cuda:1")[:, :5],
                          g.to(device="cuda:1", dtype=torch.float16),
                          g.to(device="cuda:1", dtype=torch.float16)]
            if inject_inf >= 0:
                grads[inject_inf][2, 2] = float('inf')
            return grads

        GradScaler = partial(torch.GradScaler, device=device.type)
        scaler = GradScaler()
        dummy_params = [torch.empty_like(g) for g in perfect_storm_grads(-1)]
        dummy_opt = torch.optim.SGD(dummy_params, lr=1.)

        # Ensures the inf/nan checking can find an inf injected onto any grad in the perfect storm.
        for inject_inf in range(-1, len(dummy_params)):
            found_inf = torch.full((1,), 0.0, dtype=torch.float, device=device0)
            grads = perfect_storm_grads(inject_inf)
            for i, p in enumerate(dummy_params):
                p.grad = grads[i]
            found_inf_per_device = scaler._unscale_grads_(dummy_opt, inv_scale, found_inf, True)
            if inject_inf < 0:
                # No inf was injected, ensures unscaling worked normally.
                self.assertTrue(sum(v.item() for v in found_inf_per_device.values()) == 0)
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)
            else:
                # inf was injected, ensures inf was found.
                self.assertTrue(sum(v.item() for v in found_inf_per_device.values()) == 1)

    @onlyNativeDeviceTypes
    @dtypes(torch.float)
    def test_grad_scaling_update_scale(self, device, dtype):
        # [FUNCTION DOC] Simulates the amp update-scale op: with no infs, the scale stays after
        # one iteration (growth tracker 1) then doubles and tracker resets after the growth
        # interval; with an inf found, the scale is multiplied by the backoff factor and the
        # tracker resets.
        growth = 2.0
        backoff = 0.25
        growth_interval = 2
        scale = torch.full((1,), 4.0, dtype=dtype, device=device)
        growth_tracker = torch.full((1,), 0.0, dtype=torch.int32, device=device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device=device)

        # Simulates 2 consecutive unskipped iterations
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 1)
        self.assertEqual(scale, 4.0)
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 8.0)

        # Simulates a skipped iteration
        found_inf.fill_(1.0)
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 2.0)

    @skipIfTorchDynamo("Failed running call_function for sparse_coo_tensor. See https://github.com/pytorch/pytorch/issues/118856")
    @onlyNativeDeviceTypes
    @dtypes(torch.float)
    def test_grad_scaling_unscale_sparse(self, device, dtype):
        # [FUNCTION DOC] Checks GradScaler unscale on sparse COO gradients: float grads are
        # divided by 4 with no inf flag; sparse grads containing inf or nan set the found-inf
        # flag; half sparse grads unscale correctly with fp16 allowed; and an uncoalesced half
        # sparse gradient whose duplicate entries overflow when summed is reported as an overflow.
        # Skipped under Dynamo.
        device = torch.device(device)
        scaler = torch.GradScaler(device=device.type)

        inv_scale = torch.full((1,), 0.25, dtype=dtype, device=device)
        found_inf = torch.empty((1,), dtype=dtype, device=device)
        cur = found_inf.device

        i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]], device=device, dtype=torch.int64)
        v = torch.tensor([16., 32., 64.], device=device, dtype=torch.float)
        s = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=dtype)

        p = s.clone()
        if not p.is_sparse:
            raise AssertionError("expected p to be sparse")
        opt = torch.optim.SGD([p], lr=1.)

        p.grad = s.clone()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertEqual(p.grad.to_dense(), (s / 4).to_dense())

        v = torch.FloatTensor([16., 32., float('inf')])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=dtype)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        v = torch.FloatTensor([16., 32., float('nan')])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=dtype)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        p = s.clone().half()
        if not p.is_sparse:
            raise AssertionError("expected p to be sparse")
        opt = torch.optim.SGD([p], lr=1.)

        p.grad = s.clone().half()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertEqual(p.grad.to_dense(), (s.half() / 4).to_dense())

        # Creates fp16 sparse tensor with duplicated indices (uncoalesced).  The uncoalesced representation
        # does not overflow in fp16, but the coalesced representation would, because 64000 + 64000 > fp16 max.
        # _amp_non_finite_check_and_unscale_ should report an overflow here.
        i = torch.LongTensor([[0, 1, 0],
                              [2, 0, 2]])
        v = torch.FloatTensor([64000., 32., 64000.])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=torch.float16)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 1.0)

    @onlyNativeDeviceTypes
    def test_grad_scaling_state_dict(self, device):
        # [FUNCTION DOC] For lazily initialized and not lazily initialized scale, loads one
        # GradScaler's state dict into another with different settings and checks scale, growth
        # factor, backoff factor, growth interval and growth tracker are copied.
        device = torch.device(device)
        GradScaler = partial(torch.GradScaler, device=device.type)
        for lazy_init_scale in True, False:
            s0 = GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
            s1 = GradScaler(init_scale=6., growth_factor=7., backoff_factor=.8, growth_interval=1)

            # sets a random value for load_state_dict to overwrite
            s1._init_growth_tracker = 7

            if lazy_init_scale:
                # Dummy scale() call to ensure the scale tensor is lazily initialized.
                s1.scale(torch.full((1,), 4.0, dtype=torch.float32, device=device))
                if "cuda" == device.type:
                    self.assertTrue(isinstance(s1._scale, torch.cuda.FloatTensor))
                else:
                    self.assertTrue(isinstance(s1._scale, torch.FloatTensor))

            s1.load_state_dict(s0.state_dict())

            self.assertEqual(s1.get_scale(), 3.)
            self.assertEqual(s1.get_growth_factor(), 4.)
            self.assertEqual(s1.get_backoff_factor(), .5)
            self.assertEqual(s1.get_growth_interval(), 2)
            self.assertEqual(s1._init_growth_tracker, 0)

    # _run_scaling_case generalizes some single-optimizer test logic to avoid too much copy-pasting below.
    def _run_scaling_case(self, device, run, unskipped, skipped, atol=1e-7, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
        # [FUNCTION DOC] HELPER: Shared harness for single-optimizer GradScaler tests. For scaling
        # enabled and disabled, builds control and scaling models/optimizers, runs the supplied
        # run callable on both (the scaling one with the scaler API), then checks the final scale
        # equals the initial 128 times the growth factor to the number of unskipped steps times
        # the backoff factor to the number of skipped steps (or 1 when disabled), and that
        # gradients, optimizer state and parameters match within atol.
        # Ensure scaling can be disabled without changing user control flow.
        for enabled in True, False:
            (
                mod_control, mod_scaling, opt_control, opt_scaling, data, loss_fn, skip_iter,
            ) = _create_scaling_case(device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs)

            # For functionality, test with a modest initial scale, and an unrealistically-large growth factor
            # so any potential errors with the growth factor handling will be magnified.
            GradScaler = partial(torch.GradScaler, device=device)
            scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            _ = run(device, data, mod_control, opt_control, scaler, loss_fn, skip_iter, False)
            ret = run(device, data, mod_scaling, opt_scaling, scaler, loss_fn, skip_iter, True)

            # Allows run() to optionally return a different scaler instance.
            scaler = ret if ret else scaler

            # If scaling was enabled, the scale factor should have been multiplied by the growth factor
            # len(data) - skipped times and the backoff factor "skipped" times.
            if enabled:
                net_growth = scaler.get_growth_factor()**unskipped if unskipped > 0 else 1.0
                net_backoff = scaler.get_backoff_factor()**skipped if skipped > 0 else 1.0
                self.assertTrue(scaler.get_scale() == (128. * net_growth * net_backoff))
            else:
                self.assertTrue(scaler.get_scale() == 1.0)

            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                self.assertEqual(c.grad, s.grad, atol=atol, rtol=1e-05)

                c_state, s_state = opt_control.state[c], opt_scaling.state[s]
                for k in c_state:
                    self.assertEqual(c_state[k], s_state[k], atol=atol, rtol=1e-05, msg=k)

                self.assertEqual(c, s, atol=atol, rtol=1e-05)

    @onlyNativeDeviceTypes
    @parametrize("foreach, fused", [(None, None), (True, None), (None, True)])
    @optims(
        [optim for optim in optim_db if optim.optim_cls in [torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD]],
        dtypes=[torch.float32]
    )
    def test_grad_scaling_autocast(self, device, dtype, optim_info, foreach, fused):
        # [FUNCTION DOC] Runs the scaling harness with autocast: a run callable trains with fp16
        # autocast and the scaler (injecting an inf gradient on the skip iteration), compared
        # against a plain fp32 run, for Adam, AdamW and SGD with foreach/fused options; a second
        # pass pickles the scaler every step. For Adam and AdamW the comparison is expected to
        # raise AssertionError.
        try_pickle = False

        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # [FUNCTION DOC] HELPER: Training loop used by the autocast test: for each batch zero
            # grads, forward (autocast to half when the scaling API is used), and either
            # scale/backward/step/update (injecting inf into a weight gradient on the skip
            # iteration and optionally pickling the scaler) or plain backward and step skipping
            # the skip iteration. Returns the scaler.
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                with torch.autocast(device_type=device, dtype=torch.half, enabled=try_scaling_api):
                    output = model(input)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        with torch.no_grad():
                            model[1].weight.grad.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        optimizer_ctor = optim_info.optim_cls

        # Compares no scaling + no autocasting against scaling + autocasting.
        # NOTE(mkozuki): With current way of testing, `torch.optim.Adam` is failing in spite of `foreach` and `fused`.
        #   Giving some flexibility to this test might help.
        context = contextlib.nullcontext
        if optimizer_ctor in (torch.optim.Adam, torch.optim.AdamW):
            from functools import partial
            context = partial(self.assertRaises, AssertionError)
        with context():
            # sets atol=1e-3 because we're comparing pure fp32 arithmetic vs a mixture of fp16 and fp32
            self._run_scaling_case(
                device, run, unskipped=3, skipped=1, atol=1e-3,
                optimizer_ctor=optimizer_ctor, optimizer_kwargs={"foreach": foreach, "fused": fused},
            )
            # this will be picked up by try_pickle within run():
            try_pickle = True
            self._run_scaling_case(
                device, run, unskipped=3, skipped=1, atol=1e-3,
                optimizer_ctor=optimizer_ctor, optimizer_kwargs={"foreach": foreach, "fused": fused},
            )

    # Make sure that the parameters become nonsense when scaled gradients are finite
    # but they get invalidated before `optimizer.step`, after `GradScaler.unscale_`

    def _test_params_invalidated_with_grads_invalidated_between_unscale_and_step(self, device, dtype, optim_info):
        # [FUNCTION DOC] HELPER: For each optimizer input configuration, trains a model with
        # GradScaler under autocast but overwrites all gradients with inf or nan after
        # unscale, then steps and updates; asserts that every model has parameters containing
        # nan or inf afterward.
        optimizer_ctor = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",))

        for optim_input in all_optim_inputs:
            model, _, optimizer, _, data, loss_fn, _ = _create_scaling_case(
                device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optim_input.kwargs,
            )
            scaler = torch.GradScaler(device=device, init_scale=128.0)

            for input, target in data:
                optimizer.zero_grad()
                with torch.autocast(device_type=device, dtype=torch.half):
                    output = model(input)
                    loss = loss_fn(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # deliberately break grads
                for j, param in enumerate(model.parameters()):
                    param.grad.copy_(torch.inf if j % 2 else torch.nan)

                scaler.step(optimizer)
                scaler.update()

            self.assertTrue(all((p.isnan().any() or p.isinf().any()) for p in model.parameters()))

    @onlyNativeDeviceTypes
    @optims(
        [optim for optim in optim_db if optim.optim_cls in [torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD]],
        dtypes=[torch.float32]
    )
    def test_params_invalidated_with_grads_invalidated_between_unscale_and_step(self, device, dtype, optim_info):
        # [FUNCTION DOC] Native devices; runs the invalidated-gradients helper for Adam, AdamW and
        # SGD in float32.
        self._test_params_invalidated_with_grads_invalidated_between_unscale_and_step(device, dtype, optim_info)

    @onlyNativeDeviceTypes
    @optims(
        [optim for optim in optim_db if optim.optim_cls in [torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD]],
        dtypes=[torch.float32]
    )
    @torch._inductor.config.patch("graph_partition", True)
    def test_params_invalidated_with_grads_invalidated_and_graph_partition(self, device, dtype, optim_info):
        # [FUNCTION DOC] Same as the invalidated-gradients test but with the inductor
        # graph_partition config patched on.
        self._test_params_invalidated_with_grads_invalidated_between_unscale_and_step(device, dtype, optim_info)

    @onlyNativeDeviceTypes
    def test_grad_scale_will_not_overflow(self, device):
        # [FUNCTION DOC] Checks that GradScaler with a huge initial scale (1e38) and large growth
        # factor does not produce inf or nan in its scale after a step and update; raises
        # AssertionError otherwise.
        device = torch.device(device)
        model = torch.nn.Linear(5, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.GradScaler(device=device.type, growth_interval=1, growth_factor=2**4, init_scale=1e38)
        optimizer.zero_grad()
        x = torch.randn(1, 5).to(device)
        y = 1e-30 * torch.randn(1, 1).to(device)
        l = ((model(x) - y) ** 2).mean()
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler._scale == float("inf") or scaler._scale != scaler._scale:
            raise AssertionError(f"scaler._scale should not be inf or nan, got {scaler._scale}")

    @onlyNativeDeviceTypes
    def test_grad_scaling_clipping(self, device):
        # [FUNCTION DOC] Runs the scaling harness with gradient clipping by norm applied to scaled
        # gradients (max norm multiplied by the scale), with one skipped iteration.
        device = torch.device(device)

        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # [FUNCTION DOC] HELPER: Training loop: per batch zero grads, forward, backward
            # (scaled when using the scaler API, clipping to max norm 0.2 or the scaled
            # equivalent), injecting inf on the skip iteration, then step/update or plain
            # optimizer step.
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * scaler.get_scale())
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(device.type, run, unskipped=3, skipped=1, atol=1e-5)

    @onlyNativeDeviceTypes
    def test_grad_scaling_clipping_separate_unscale(self, device):
        # [FUNCTION DOC] Runs the scaling harness with gradient clipping done after an explicit
        # unscale call, with one skipped iteration and no nonfinite error.
        device = torch.device(device)

        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # [FUNCTION DOC] HELPER: Training loop: backward on the scaled loss, inject inf on the
            # skip iteration, unscale the optimizer, clip norm to 0.2 without raising on
            # nonfinite, then step and update; the control path clips and steps normally.
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(device.type, run, unskipped=3, skipped=1)

    @onlyNativeDeviceTypes
    def test_grad_scaling_penalty(self, device):
        # [FUNCTION DOC] Runs the scaling harness with a gradient-norm penalty added to the loss
        # (double backward with create_graph), unscaling the differentiated gradients by the
        # inverse scale.
        device = torch.device(device)

        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # [FUNCTION DOC] HELPER: Training loop: compute gradients of the (scaled) loss with
            # create_graph, rescale them, add the gradient norm to the loss, backward, inject inf
            # on the skip iteration, then step/update or plain step.
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                if try_scaling_api:
                    grad_params = torch.autograd.grad(scaler.scale(loss),
                                                      model.parameters(), create_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(device.type, run, unskipped=3, skipped=1)

    @onlyNativeDeviceTypes
    def test_grad_scaling_accumulation(self, device):
        # [FUNCTION DOC] Runs the scaling harness with gradient accumulation over two iterations
        # (loss divided by two) and no skipped steps.
        device = torch.device(device)

        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # [FUNCTION DOC] HELPER: Training loop that accumulates gradients over two iterations
            # before stepping (and updating the scaler) then zeroing grads.
            iters_to_accumulate = 2
            for i, (input, target) in enumerate(data):
                output = model(input)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate
                if try_scaling_api:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if try_scaling_api:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

        self._run_scaling_case(device.type, run, unskipped=2, skipped=0)

    @onlyNativeDeviceTypes
    def test_grad_scaling_multiple(self, device):
        # [FUNCTION DOC] Tests GradScaler with two models and two optimizers fed by two losses:
        # backward both scaled losses, inject inf into one model on the skip iteration, separately
        # unscale one optimizer, step both, update; compares parameters against an unscaled
        # control and checks the final scale (growth three times, backoff once) or 1 if disabled.
        device = torch.device(device)
        # Tests gradient scaling with 2 models and 2 optimizers that both receive gradients from 2 losses.
        # Some of the logic here cannot reuse the generic helper functions created for the 1-optimizer cases.
        for enabled in True, False:
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = \
                _create_scaling_case(device.type)
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = \
                _create_scaling_models_optimizers(device.type)

            GradScaler = partial(torch.GradScaler, device=device.type)
            scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                # [FUNCTION DOC] HELPER: Run the two-model training loop over the data, either
                # with the scaler API (scale both losses, inject inf, unscale one optimizer,
                # step both, update) or plainly (step the first optimizer always and the second
                # unless skipped).
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input)
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float('inf'))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)
                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            self.assertTrue(scaler.get_scale() == (128. * scaler.get_growth_factor()**3 *
                                                   scaler.get_backoff_factor()**1) if enabled else 1.0)

            for c, s in zip(chain(mod_control0.parameters(), mod_control1.parameters()),
                            chain(mod_scaling0.parameters(), mod_scaling1.parameters())):
                self.assertEqual(c, s, rtol=1e-5, atol=1e-7)

    @onlyNativeDeviceTypes
    def test_grad_scaler_pass_itself(self, device):
        # [FUNCTION DOC] Checks that GradScaler step passes itself to optimizers whose step
        # accepts a grad_scaler argument (warning with FutureWarning), and sets grad scale and
        # found-inf tensors on optimizers that use the older attribute protocol. Defines
        # placeholder optimizers that assert these conditions inside their step.
        device = torch.device(device)
        GradScaler = partial(torch.amp.GradScaler, device=device.type)

        class _PlaceHolderOptimizer(torch.optim.Optimizer):
            # [CLASS DOC] Optimizer subclass used as a base for the test optimizers; stores the
            # test case as a class attribute so steps can assert, and marks that its step supports
            # amp scaling.
            tester = self

            def __init__(self, params, defaults=None):
                # [FUNCTION DOC] Initialize the base optimizer with the given parameters and
                # default settings (an empty dict if none) and set the flag stating step supports
                # amp scaling.
                if defaults is None:
                    defaults = {}
                super().__init__(params, defaults)
                self._step_supports_amp_scaling = True

        class Optimizer1(_PlaceHolderOptimizer):
            # [CLASS DOC] Test optimizer whose step must receive a grad_scaler argument that is a
            # GradScaler and must not have grad_scale or found_inf attributes set.
            def step(self, closure=None, *, grad_scaler=None):
                # [FUNCTION DOC] Assert that the grad_scaler argument is a GradScaler instance
                # and that the optimizer has no grad_scale and no found_inf attributes; performs
                # no update.
                self.tester.assertTrue(isinstance(grad_scaler, torch.amp.GradScaler))
                self.tester.assertFalse(hasattr(self, "grad_scale"))
                self.tester.assertFalse(hasattr(self, "found_inf"))

        class Optimizer2(_PlaceHolderOptimizer):
            # [CLASS DOC] Test optimizer whose step takes no scaler argument and must find
            # grad_scale and found_inf tensors set on itself.
            def step(self, closure=None):
                # [FUNCTION DOC] Assert that grad_scale and found_inf attributes are tensors;
                # performs no update.
                self.tester.assertTrue(isinstance(self.grad_scale, torch.Tensor))
                self.tester.assertTrue(isinstance(self.found_inf, torch.Tensor))

        x = torch.randn(4, 4).to(device)
        m = torch.nn.Linear(4, 1).to(device)
        o1 = Optimizer1(m.parameters())
        o2 = Optimizer2(m.parameters())
        scaler = GradScaler(init_scale=2.0)

        with torch.autocast(device_type=device.type, dtype=torch.half):
            y = m(x)
            loss = y.mean()
        scaler.scale(loss).backward()
        with self.assertWarns(FutureWarning):
            scaler.step(o1)
        scaler.step(o2)
        scaler.update()

    @onlyNativeDeviceTypes
    def test_grad_scaler_deprecated_warning(self, device):
        # [FUNCTION DOC] Checks that constructing the CUDA or CPU specific GradScaler emits a
        # FutureWarning saying it is deprecated.
        device = torch.device(device)
        GradScaler = torch.cuda.amp.GradScaler if "cuda" == device.type else torch.cpu.amp.GradScaler

        with self.assertWarnsRegex(
            FutureWarning,
            rf"`torch.{device.type}.amp.GradScaler\(args...\)` is deprecated.",
        ):
            _ = GradScaler(init_scale=2.0)

    @dtypesIfCUDA(torch.float, torch.double, torch.half)
    @dtypesIfCPU(torch.float, torch.double, torch.bfloat16, torch.half)
    @dtypes(torch.float, torch.double)
    def test_multinomial_cpu(self, device, dtype):
        # [FUNCTION DOC] Defines a helper that builds contiguous or non-contiguous probability
        # tensors but never calls it or any multinomial op, so effectively nothing is tested.
        # KNOWN ISSUE: the test body only defines the probability-distribution helper and ends; no
        # assertions are made and the CPU multinomial behavior is not exercised.
        def make_prob_dist(shape, is_contiguous):
            # [FUNCTION DOC] HELPER: Build a random probability tensor of the given shape (1D or
            # 2D) either contiguous or as a non-contiguous view of a larger tensor, converting
            # after generation for half and bfloat16; raises AssertionError if the non-contiguous
            # 2D tensor is contiguous. Not used by the enclosing test.
            if is_contiguous:
                if dtype == torch.half or dtype == torch.bfloat16:
                    return torch.zeros(shape, device=device).uniform_().to(dtype=dtype)
                return torch.zeros(shape, device=device, dtype=dtype).uniform_()
            elif len(shape) == 1:
                if dtype == torch.half or dtype == torch.bfloat16:
                    return torch.zeros((shape + [5]), device=device).uniform_().to(dtype=dtype)[:, 2]
                return torch.zeros((shape + [5]), device=device, dtype=dtype).uniform_()[:, 2]
            else:
                # num dim = 2
                new_shape = [2, shape[1], 7, 1, shape[0], 1, 10]
                if dtype == torch.half or dtype == torch.bfloat16:
                    prob_dist = torch.zeros(new_shape, device=device).uniform_().to(dtype=dtype)
                else:
                    prob_dist = torch.zeros(new_shape, device=device, dtype=dtype).uniform_()
                prob_dist = prob_dist.transpose(1, 4)
                prob_dist = prob_dist[1, :, 5, 0, :, 0, 4]
                if prob_dist.is_contiguous():
                    raise AssertionError("expected prob_dist to be non-contiguous")
                return prob_dist

    # FIXME: move to elementwise ternary test suite
    # As the test fails with Runtime Error not raised on XLA
    @onlyNativeDeviceTypes
    def test_where_scalar_handcrafted_values(self, device):
        # [FUNCTION DOC] Compares torch.where with scalar and tensor arguments (scalar by scalar,
        # scalar by tensor, tensor by scalar variants, many dtypes and shapes) against numpy where
        # cast to the promoted dtype; on CUDA also tries CPU scalar tensors and 0-dim conditions.
        # Tests ScalarxScalar, ScalarxTensor and TensorxScalar
        # variant of `where` against NumPy version with
        # handcrafted values.
        condition_shape = (5, 5)
        dtypes = (
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int64,
            torch.float16, torch.float32, torch.float64,
            torch.complex64, torch.complex128,
        )
        shapes = ((), (5,), (1, 5),)

        with torch.no_grad():
            tensors = (torch.empty(shape, dtype=dtype, device=device).fill_(17)
                       for shape, dtype in product(shapes, dtypes))

        # Use different values for `x` and `y`
        # as they are the output values which are compared.
        x_vals = (True, 3, 7.0, 1 + 0.5j)
        y_vals = itertools.chain((False, 4, 8.0, 2 + 0.5j), tensors)
        for x in x_vals:
            for y in y_vals:
                condition = torch.empty(*condition_shape, dtype=torch.bool, device=device).bernoulli_()
                common_dtype = torch.result_type(x, y)

                def check_equal(condition, x, y):
                    # [FUNCTION DOC] HELPER: Compute numpy where on CPU copies of the condition
                    # and operands, cast to the promoted dtype of the test, and assert it equals
                    # torch.where of the same arguments.
                    condition_np = condition.cpu().numpy()
                    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

                    # NumPy aggressively promotes to double, hence cast to output to correct dtype
                    expected = torch.from_numpy(np.where(condition_np, x_np, y_np)).to(common_dtype)
                    result = torch.where(condition, x, y)
                    self.assertEqual(expected, result)

                check_equal(condition, x, y)
                check_equal(condition, y, x)
                if self.device_type == "cuda":
                    check_equal(condition, torch.tensor(x), y)
                    check_equal(condition, y, torch.tensor(x))
                    if not isinstance(y, torch.Tensor):
                        check_equal(condition, torch.tensor(y), torch.tensor(x))
                    if isinstance(y, torch.Tensor) and y.ndim > 0:
                        check_equal(torch.tensor(True), x, y)
                        check_equal(torch.tensor(True), y, x)

    @onlyNativeDeviceTypes
    @dtypes(torch.uint16, torch.uint32, torch.uint64)
    def test_fill_barebones_unsigned(self, device, dtype):
        # [FUNCTION DOC] Checks fill_ on uint16, uint32 and uint64 for several shapes, a
        # non-contiguous row slice, and the maximum value of each dtype against torch.full.
        # Both FillKernel.cpp and FillKernel.cu dispatch over barebones
        # unsigned via AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), but the fill
        # OpInfo does not enumerate them. Cover that dispatch directly.
        for shape in ((5,), (4, 5), (2, 3, 4)):
            t = torch.empty(shape, dtype=dtype, device=device).fill_(7)
            self.assertEqual(t.cpu(), torch.full(shape, 7, dtype=dtype))

        # Non-contiguous tensor (every other row of a larger allocation).
        big = torch.empty((8, 5), dtype=dtype, device=device)
        nc = big[::2]
        nc.fill_(11)
        self.assertEqual(nc.cpu(), torch.full((4, 5), 11, dtype=dtype))

        # Boundary fill value: max of each dtype.
        max_val = torch.iinfo(dtype).max
        t = torch.empty(8, dtype=dtype, device=device).fill_(max_val)
        self.assertEqual(t.cpu().tolist(), [max_val] * 8)

    @onlyNativeDeviceTypes
    @dtypes(torch.uint16, torch.uint32, torch.uint64)
    def test_where_barebones_unsigned(self, device, dtype):
        # [FUNCTION DOC] Checks torch.where on same-dtype unsigned tensors (contiguous and
        # non-contiguous) against the CPU result.
        # The barebones unsigned dtypes are excluded from the broader
        # test_where_scalar_handcrafted_values because torch.result_type
        # does not support promoting them. Cover the same-dtype path
        # for where() against a CPU reference here.
        for shape in ((5,), (1, 5), (4, 5)):
            a = make_tensor(shape, dtype=dtype, device=device)
            b = make_tensor(shape, dtype=dtype, device=device)
            cond = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            out = torch.where(cond, a, b)
            expected = torch.where(cond.cpu(), a.cpu(), b.cpu()).to(device)
            self.assertEqual(out, expected)

        # Non-contiguous inputs.
        big = make_tensor((8, 10), dtype=dtype, device=device)
        a = big[::2]
        b = big[1::2]
        cond = torch.randint(0, 2, a.shape, dtype=torch.bool, device=device)
        out = torch.where(cond, a, b)
        expected = torch.where(cond.cpu(), a.cpu(), b.cpu()).to(device)
        self.assertEqual(out, expected)

    @onlyNativeDeviceTypes
    @dtypes(torch.uint16, torch.uint32, torch.uint64)
    def test_eq_ne_barebones_unsigned(self, device, dtype):
        # [FUNCTION DOC] Checks eq and ne for uint16/32/64: tensors differing in one element
        # (low bit flipped), identical tensors, non-contiguous row slices, and maximum-value
        # boundary comparisons.
        # The eq/ne kernels dispatch over uint16/uint32/uint64 via
        # AT_DISPATCH_V2(... AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)),
        # but the standard test_torch.py comparison sweeps don't list
        # those dtypes. Cover the same-dtype path here.
        for shape in ((5,), (4, 5), (2, 3, 4)):
            a = make_tensor(shape, dtype=dtype, device=device)
            b = a.clone()
            # Differ exactly one element so eq/ne both have True and False
            # to report. XOR with 1 toggles the low bit, so the perturbed
            # value is always in range and differs from old.
            old = int(a.view(-1)[0].item())
            b.view(-1)[0] = old ^ 1
            self.assertEqual(torch.eq(a, b), (a.cpu() == b.cpu()).to(device))
            self.assertEqual(torch.ne(a, b), (a.cpu() != b.cpu()).to(device))
            self.assertEqual(torch.eq(a, a), torch.ones_like(a, dtype=torch.bool))
            self.assertEqual(torch.ne(a, a), torch.zeros_like(a, dtype=torch.bool))

        # Non-contiguous: every-other-row slice.
        big = make_tensor((8, 4), dtype=dtype, device=device)
        a = big[::2]
        b = big[1::2]
        self.assertEqual(torch.eq(a, b), (a.cpu() == b.cpu()).to(device))
        self.assertEqual(torch.ne(a, b), (a.cpu() != b.cpu()).to(device))

        # Boundary value: max-int comparison.
        max_val = torch.iinfo(dtype).max
        a = torch.tensor([0, max_val, max_val - 1], dtype=dtype, device=device)
        b = torch.tensor([0, max_val, max_val], dtype=dtype, device=device)
        self.assertEqual(torch.eq(a, b).tolist(), [True, True, False])
        self.assertEqual(torch.ne(a, b).tolist(), [False, False, True])


    @skipIfTorchInductor("FIXME")
    def test_hook_remove(self, device):
        # [FUNCTION DOC] Reference issue 58354: registers a gradient hook that returns zeros and,
        # optionally, removes itself on first call; checks the first backward gives zero gradient
        # and the second gives zeros if the hook stays or the normal gradient (0.2) if it was
        # removed. Skipped under inductor.
        # Reference: https://github.com/pytorch/pytorch/issues/58354
        def _test_helper(remove_hook):
            # [FUNCTION DOC] HELPER: Create a ones tensor requiring grad, install a hook (removing
            # itself if remove_hook), run backward twice and check the gradient accumulation
            # results described by the outer test.
            def install_hook(tensor):
                # [FUNCTION DOC] HELPER: Register a hook on the tensor that returns zeros and
                # optionally removes itself when called; stores the handle for the hook to use.
                handle = None

                def hook(tensor):
                    # [FUNCTION DOC] HELPER: Gradient hook: remove the handle when enabled and
                    # return a zero tensor in place of the gradient.
                    if remove_hook:
                        handle.remove()
                    return torch.zeros_like(tensor)
                handle = tensor.register_hook(hook)

            t = torch.ones((1, 5), device=device, requires_grad=True)
            install_hook(t)

            # First call to backward
            t.mean().backward()
            self.assertEqual(t.grad, torch.zeros_like(t))

            # Second call to backward
            t.mean().backward()
            if remove_hook:
                # After removing the hook, make sure the usual gradient is returned
                self.assertEqual(t.grad, 0.2 * torch.ones_like(t))
            else:
                self.assertEqual(t.grad, torch.zeros_like(t))

        _test_helper(remove_hook=True)
        _test_helper(remove_hook=False)

    # FIXME: get PyTorch/XLA to run test_testing
    # This test should ideally be in test_testing.py,
    # but since pytorch/xla runs tests from test_torch.py, we have it here.
    @skipXLA
    def test_skip_xla(self, device):
        # [FUNCTION DOC] Test helper checking the skipXLA decorator: asserts False if it runs on
        # an XLA device (it must be skipped there).
        if self.device_type == 'xla':
            # Should not reach here!
            self.assertTrue(False)

    # FIXME: get PyTorch/XLA to run test_testing
    # This test should ideally be in test_testing.py,
    # but since pytorch/xla runs tests from test_torch.py, we have it here.
    @expectedFailureXLA
    def test_expected_failure_xla(self, device):
        # [FUNCTION DOC] Test helper checking the expectedFailureXLA decorator: fails if run on an
        # XLA device (so it is reported as expected failure there).
        if self.device_type == 'xla':
            self.assertTrue(False)

    # FIXME: get PyTorch/XLA to run test_testing
    # This test should ideally be in test_testing.py,
    # but since pytorch/xla runs tests from test_torch.py, we have it here.
    def test_assertRaisesRegex_ignore_msg_non_native_device(self, device):
        # [FUNCTION DOC] Checks that nll_loss with a wrongly sized class weight raises
        # RuntimeError with the weight-size message (for non-native devices the assertRaisesRegex
        # helper only checks the exception type).
        # Verify that self.assertRaisesRegex only checks the Error and ignores
        # message for non-native devices.
        x = torch.randn((10, 3), device=device)
        t = torch.empty(10, dtype=torch.int64, device=device).random_(0, 3)
        invalid_weight = torch.randn(4, device=device)
        msg = "weight tensor should be defined either for all 3 classes or no classes"

        # XLA raises RuntimeError with a different message.
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.nn.functional.nll_loss(x, t, weight=invalid_weight)

    @dtypes(*all_passthru_types_and(torch.chalf))
    def test_copy_(self, device, dtype):
        # [FUNCTION DOC] For each pair of passthrough dtypes (including complex half),
        # copies random data from a source of one dtype into the dtype under test, and when
        # the cast is considered safe checks copying back reproduces the source within
        # dtype-specific tolerances.
        def can_cast(src_dtype, dst_dtype):
            # [FUNCTION DOC] HELPER: Return whether copying from src_dtype to dst_dtype is
            # lossless: for unsigned destinations only bool sources or unsigned sources no wider
            # than the destination; otherwise defers to torch.can_cast.
            # torch.can_cast(torch.int16, torch.uint8) returns True which
            # isn't actually safe-cast. Override for unsigned destinations:
            # only a bool source, or an unsigned source no wider than the
            # destination, round-trips losslessly. Narrowing (uint16 ->
            # uint8) is rejected.
            unsigned = {torch.uint8, torch.uint16, torch.uint32, torch.uint64}

            if dst_dtype in unsigned:
                return (src_dtype is torch.bool
                        or (src_dtype in unsigned
                            and src_dtype.itemsize <= dst_dtype.itemsize))
            return torch.can_cast(src_dtype, dst_dtype)

        def make_tensor_wrapper(shape, dtype):
            # [FUNCTION DOC] HELPER: Create a random tensor with make_tensor, or with randn for
            # complex half which make_tensor cannot generate.
            if dtype is not torch.complex32:
                # Make tensor does not support generating
                # complex32 tensor
                return make_tensor(shape, device=device, dtype=dtype)
            return torch.randn(shape, device=device, dtype=dtype)

        t = make_tensor_wrapper((50,), dtype)
        src_dtypes = all_passthru_types_and(torch.chalf)
        for src_dtype in src_dtypes:
            src = make_tensor_wrapper((50,), dtype=src_dtype)
            t.copy_(src)
            dst = make_tensor_wrapper((50, ), dtype=src_dtype)
            if can_cast(src_dtype, dtype):
                rtol = None
                atol = None
                if dtype in (torch.half, torch.complex32):
                    rtol = 1e-3
                    atol = 1e-3
                if dtype == torch.bfloat16:
                    rtol = 1e-2
                    atol = 1e-2
                self.assertEqual(src, dst.copy_(t), rtol=rtol, atol=atol)

    @dtypes(*all_types_complex_float8_and(
        torch.bool, torch.half, torch.bfloat16, torch.complex32,
        torch.uint16, torch.uint32, torch.uint64))
    def test_item(self, device, dtype):
        # [FUNCTION DOC] Checks item on a one-element tensor of many dtypes (including float8,
        # unsigned and complex half) returns 1; skips XLA for unsupported dtypes.
        xla_unsupported_dtypes = [
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]
        if torch.device(device).type == 'xla' and dtype in xla_unsupported_dtypes:
            self.skipTest('uint16,32,64,float8 not implemented on XLA')
        t = torch.ones((), device=device, dtype=dtype)
        self.assertEqual(1, t.item())

    def test__local_scalar_dense_with_empty_tensor(self, device):
        # [FUNCTION DOC] Checks the internal local-scalar-dense op on an empty tensor raises
        # RuntimeError saying empty tensors are not supported.
        input = torch.randn(0, device=device)
        with self.assertRaisesRegex(RuntimeError, "Empty tensor not supported"):
            torch.ops.aten._local_scalar_dense(input)

    @onlyNativeDeviceTypes
    def test_masked_scatter_inplace_noncontiguous(self, device):
        # [FUNCTION DOC] Checks in-place masked_scatter_ gives the same result for contiguous and
        # non-contiguous self and mask combinations; raises AssertionError if the contiguity setup
        # is not as intended.
        t = torch.zeros(5, 2, dtype=torch.long, device=device)
        t_non_contig = t.transpose(0, 1)
        t_contig = t_non_contig.contiguous()

        if not t_contig.is_contiguous():
            raise AssertionError("expected t_contig to be contiguous")
        if t_non_contig.is_contiguous():
            raise AssertionError("expected t_non_contig to be non-contiguous")

        mask = torch.tensor([[False, True], [False, True], [False, False], [True, True], [True, True]], device=device)
        mask_non_contig = mask.transpose(0, 1)
        mask_contig = mask_non_contig.contiguous()

        if not mask_contig.is_contiguous():
            raise AssertionError("expected mask_contig to be contiguous")
        if mask_non_contig.is_contiguous():
            raise AssertionError("expected mask_non_contig to be non-contiguous")

        # source is always converted to contiguous by the op.
        source = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 9]], device=device)

        # t: contig, mask: contig
        expected = t_contig.masked_scatter_(mask_contig, source)

        # t: non-contig, mask: non-contig
        actual = t_non_contig.masked_scatter_(mask_non_contig, source)
        self.assertEqual(actual, expected)

        # t: contig, mask: non-contig
        actual = t_contig.masked_scatter_(mask_non_contig, source)
        self.assertEqual(actual, expected)

        # t: non-contig, mask: contig
        actual = t_non_contig.masked_scatter_(mask_contig, source)
        self.assertEqual(actual, expected)

    def test_tolist_with_grad(self, device):
        # [FUNCTION DOC] Calls torch.func.grad on a function that converts a tracked tensor with
        # tolist, asserts the list equals the expected values (raises AssertionError otherwise)
        # and returns the sum; checks the gradient is all ones.
        def f(x):
            # [FUNCTION DOC] HELPER: Convert x to a list, assert it equals 1, 2, 3 element by
            # element (AssertionError if not), and return the sum of x.
            result = x.tolist()
            for l, r in zip(result, [1.0, 2.0, 3.0]):
                if l != r:
                    raise AssertionError(f"expected {l} == {r}")
            return sum(x)

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
        grad_f = torch.func.grad(f)
        result = grad_f(x)
        self.assertEqual(result.tolist() , [1.0, 1.0, 1.0])

    def test_swap_data_ptr_with_empty(self, device):
        # [FUNCTION DOC] Checks swapping the data pointer of a storage resized to zero bytes with
        # another storage: x now holds y's data, and y's storage has zero bytes.
        x = torch.randn(4, device=device)
        y = torch.randn(4, device=device)
        y_data = y.clone()
        x_storage = x.untyped_storage()
        y_storage = y.untyped_storage()
        x_storage.resize_(0)
        self.assertEqual(x_storage.nbytes(), 0)
        x_storage._swap_data_ptr_(y_storage)
        self.assertEqual(x, y_data)
        self.assertEqual(y_storage.nbytes(), 0)

    def test_swap_data_ptr_non_empty(self, device):
        # [FUNCTION DOC] Checks swapping the data pointers of two non-empty storages exchanges
        # tensor contents.
        x = torch.randn(4, device=device)
        y = torch.randn(4, device=device)
        x_data = x.clone()
        y_data = y.clone()
        x_storage = x.untyped_storage()
        y_storage = y.untyped_storage()
        x_storage._swap_data_ptr_(y_storage)
        self.assertEqual(x, y_data)
        self.assertEqual(y, x_data)

    def test_swap_data_ptr_preserves_views(self, device):
        # [FUNCTION DOC] Checks that after swapping into a resized-to-empty storage, a
        # pre-existing view of the first tensor sees the second tensor's data slice.
        x = torch.randn(4, 4, device=device)
        x_view = x[1:3]
        y = torch.randn(4, 4, device=device)
        y_data = y.clone()
        x_storage = x.untyped_storage()
        y_storage = y.untyped_storage()
        x_storage.resize_(0)
        x_storage._swap_data_ptr_(y_storage)
        self.assertEqual(x, y_data)
        self.assertEqual(x_view, y_data[1:3])

    def test_swap_data_ptr_self(self, device):
        # [FUNCTION DOC] Checks swapping a storage's data pointer with itself leaves the
        # tensor unchanged.
        x = torch.randn(4, device=device)
        x_data = x.clone()
        s = x.untyped_storage()
        s._swap_data_ptr_(s)
        self.assertEqual(x, x_data)

    def test_swap_data_ptr_device_mismatch(self, device):
        # [FUNCTION DOC] Checks that swapping data pointers between storages on different devices
        # (real and meta) raises RuntimeError mentioning the same device requirement.
        x = torch.randn(4, device=device)
        y = torch.randn(4, device="meta")
        with self.assertRaisesRegex(RuntimeError, "same device"):
            x.untyped_storage()._swap_data_ptr_(y.untyped_storage())

    def test_swap_data_ptr_nbytes_mismatch(self, device):
        # [FUNCTION DOC] Checks that swapping data pointers between storages of different byte
        # sizes raises RuntimeError mentioning the same nbytes requirement.
        x = torch.randn(4, device=device)
        y = torch.randn(8, device=device)
        with self.assertRaisesRegex(RuntimeError, "same nbytes"):
            x.untyped_storage()._swap_data_ptr_(y.untyped_storage())


# Tests that compare a device's computation with the (gold-standard) CPU's.
class TestDevicePrecision(TestCase):
    # [CLASS DOC] Test case comparing device computations or device interactions with CPU results
    # (serialization, copies, indexing across devices, clamp). Sets exact_dtype so dtype is
    # compared. Methods get a device or a list of devices from the device-type instantiation.
    exact_dtype = True

    # FIXME: move to indexing test suite
    @onlyCUDA
    def test_index_add_bfloat16(self, device):
        # [FUNCTION DOC] CUDA only. Compares index_add on bfloat16 tensors on CPU and on the CUDA
        # device within an absolute tolerance of 0.01.
        inp_tensor = torch.randn(5, 3, device='cpu').bfloat16()
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16, device='cpu')
        index = torch.tensor([0, 4, 2], device='cpu')
        out_cpu = inp_tensor.index_add(0, index, t)

        inp_tensor = inp_tensor.to(device=device)
        t = t.to(device=device)
        index = index.to(device=device)
        out_gpu = inp_tensor.index_add(0, index, t)

        self.assertEqual(out_cpu, out_gpu, atol=1e-2, rtol=0)

    # FIXME: move to serialization test suite
    def test_device_serialization(self, device):
        # [FUNCTION DOC] Saves a tensor to a temporary file with torch.save, loads it back, and
        # checks equal values, same Python type and same device.
        x = torch.randn(4, 4, device=device)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)

        self.assertEqual(x_copy, x)
        self.assertIs(type(x_copy), type(x))
        self.assertEqual(x_copy.device, x.device)

    # FIXME: move to serialization test suite
    @deviceCountAtLeast(2)
    def test_multidevice_serialization(self, devices):
        # [FUNCTION DOC] Needs two devices. Saves a list of tensors on two different devices,
        # loads them, and checks each has the original value, type and device.
        x = [torch.randn(4, 4, device=devices[0]),
             torch.randn(4, 4, device=devices[1])]

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)

        for original, cp in zip(x, x_copy):
            self.assertEqual(cp, original)
            self.assertIs(type(cp), type(original))
            self.assertEqual(cp.device, original.device)

    # FIXME: move to data movement test suite
    @deviceCountAtLeast(1)
    def test_copy_noncontig(self, devices):
        # [FUNCTION DOC] Needs at least one device. For several device pairs (CPU to device,
        # device to CPU, and between two devices when available) copies strided slices from a
        # float tensor into an int tensor and checks the copied positions got truncated values and
        # the others stayed zero.
        def do_test(d0, d1):
            # [FUNCTION DOC] HELPER: Copy every other element of a float tensor on d0 into a zero
            # int tensor on d1 and assert the result is the expected truncated pattern; also
            # asserts the two dtypes differ.
            x = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], device=d0)
            y = torch.tensor([0, 0, 0, 0, 0, 0], device=d1)
            self.assertNotEqual(x.dtype, y.dtype)

            y[::2].copy_(x[::2])
            self.assertEqual(y, [1, 0, 3, 0, 5, 0])

        do_test('cpu', devices[0])
        do_test(devices[0], 'cpu')

        if len(devices) > 1:
            do_test(devices[0], devices[1])

    @deviceCountAtLeast(2)
    def test_type_conversions_same_device(self, devices):
        # [FUNCTION DOC] Needs two devices. Checks that int conversion, type and to with an int
        # dtype keep a tensor on the second device.
        x = torch.randn(5, 5, device=devices[1])
        self.assertEqual(x.int().device, torch.device(devices[1]))
        self.assertEqual(x.type(torch.int).device, torch.device(devices[1]))
        self.assertEqual(x.to(torch.int).device, torch.device(devices[1]))

    @dtypesIfCUDA(torch.half, torch.float, torch.double,
                  torch.int8, torch.short, torch.int, torch.long,
                  torch.uint8)
    @dtypes(torch.float, torch.double,
            torch.int8, torch.short, torch.int, torch.long,
            torch.uint8)
    def test_from_sequence(self, device, dtype):
        # [FUNCTION DOC] Builds a tensor from a nested Python list of integers for several dtypes
        # and compares it with an arange reshaped to 5 by 4 ignoring dtype.
        seq = [list(range(i * 4, i * 4 + 4)) for i in range(5)]
        reference = torch.arange(0, 20).resize_(5, 4)
        self.assertEqual(torch.tensor(seq, dtype=dtype, device=device), reference, exact_dtype=False)

    # FIXME: moved to indexing test suite
    @deviceCountAtLeast(1)
    def test_advancedindex_mixed_cpu_devices(self, devices) -> None:
        # [FUNCTION DOC] For each device, indexes a 5D tensor using integer index tensors on CPU,
        # or a mix of CPU and device, in several positions and compares getitem to the CPU result,
        # and runs setitem with random values to check nothing raises.
        def test(x: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Compare getitem with mixed index tensors against the CPU
            # computation, for both multi-index and single-index forms, then perform both
            # assignment forms with random values of the right shape.
            # test getitem
            self.assertEqual(x[:, ia, None, ib, 0].cpu(),
                             x.cpu()[:, ia.cpu(), None, ib.cpu(), 0])
            self.assertEqual(x[ia], x.cpu()[ia.cpu()])
            # test setitem
            x_clone1 = x.clone()
            x_clone2 = x.clone()
            first_shape = x[:, ia, None, ib, 0].shape
            second_shape = x[ia].shape
            x_clone1[:, ia, None, ib, 0] = torch.randn(first_shape).to(x_clone1)
            x_clone2[ia] = torch.randn(second_shape).to(x_clone2)

        cpu = torch.device('cpu')
        for device in devices:
            x = torch.randn(3, 4, 4, 4, 3)
            ia = torch.tensor([0, 2, 1])
            ib = torch.tensor([0, 2, 1])

            # Index device tensor with cpu tensor
            x = x.to(device)
            ia = ia.to(cpu)
            ib = ib.to(cpu)
            test(x, ia, ib)

            # Index device tensor with mixed cpu, device tensors
            x = x.to(device)
            ia = ia.to(cpu)
            ib = ib.to(device)
            test(x, ia, ib)

    @deviceCountAtLeast(1)
    def test_advancedindex_mixed_devices_error(self, devices) -> None:
        # [FUNCTION DOC] Checks that indexing with index tensors on a device that does not match
        # the indexed tensor (cpu tensor with device index, mixed devices, or two different
        # devices) raises RuntimeError saying indices should be CPU or on the tensor device.
        def test(x: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Assert that a multi-index getitem and a single-index
            # getitem with the given index tensors both raise RuntimeError whose message names
            # the tensor device.
            # test getitem
            with self.assertRaisesRegex(RuntimeError, fr"indices should be either .* \({x.device}\)"):
                x[:, ia, None, ib, 0]
            with self.assertRaisesRegex(RuntimeError, fr"indices should be either .* \({x.device}\)"):
                x[ib]

        cpu = torch.device('cpu')
        for device in devices:
            # Index cpu tensor with device tensor
            x = torch.randn(3, 4, 4, 4, 3)
            ia = torch.tensor([0, 2, 1]).to(device)
            ib = torch.tensor([0, 2, 1]).to(device)
            test(x, ia, ib)

            # Index cpu tensor with mixed cpu, device tensors
            x = x.to(cpu)
            ia = ia.to(cpu)
            ib = ib.to(device)
            test(x, ia, ib)

            if len(devices) > 1:
                other_device = devices[0] if device == devices[1] else devices[1]

                # Index device tensor with mixed cpu, device tensors on different devices
                x = x.to(device)
                ia = ia.to(cpu)
                ib = ib.to(other_device)
                test(x, ia, ib)

    # FIXME: move to data movement test suite
    def test_copy_broadcast(self, device) -> None:
        # [FUNCTION DOC] Checks copy_ broadcasts a 1D tensor into a 2D tensor between CPU and the
        # device in both directions.
        x = torch.randn(10, 5)
        y = torch.randn(5, device=device)
        x.copy_(y)
        self.assertEqual(x[3], y)

        x = torch.randn(10, 5, device=device)
        y = torch.randn(5)
        x.copy_(y)
        self.assertEqual(x[3], y)

    # FIXME: move to an elementwise ternary test suite
    @dtypes(torch.int64, torch.float32, torch.float64)
    def test_clamp(self, device, dtype):
        # [FUNCTION DOC] For int64, float32 and float64 and several shapes (contiguous and not),
        # checks clamp with tensor bounds equals max then min composition and numpy clip, clamp
        # with only min or only max, and broadcasting of the bounds and of the input.
        test_args = [
            *product(
                [(100, 50), (10, 64), (97,)],  # shape
                (True, False),  # non-contiguous
            )
        ]

        for shape, noncontig in test_args:
            x = make_tensor(shape, device=device, dtype=dtype,
                            noncontiguous=noncontig)
            ub = make_tensor(shape, device=device, dtype=dtype,
                             noncontiguous=noncontig)
            lb = make_tensor(shape, device=device, dtype=dtype,
                             noncontiguous=noncontig)

            expect = x.max(lb).min(ub)
            actual = x.clamp(lb, ub)
            self.assertEqual(expect, actual)

            expect = np.clip(x.cpu().numpy(), lb.cpu().numpy(), ub.cpu().numpy())
            self.assertEqual(expect, actual)

            expect = x.max(lb)
            actual = x.clamp(min=lb)
            self.assertEqual(expect, actual)

            expect = x.min(ub)
            actual = x.clamp(max=ub)
            self.assertEqual(expect, actual)

            # Test broadcasting min & max
            expect = x.max(lb[0]).min(ub[..., :1])
            actual = x.clamp(lb[0], ub[..., :1])
            self.assertEqual(expect, actual)

            # Test broadcasting x
            expect = x[..., :1].max(lb).min(ub)
            actual = x[..., :1].clamp(lb, ub)
            self.assertEqual(expect, actual)

    def test_cuda_device_idx(self, device):
        # [FUNCTION DOC] Checks that a regular zero tensor and an efficient zero tensor created on
        # the same device report the same device.
        x = torch.zeros(3, device=device)
        y = torch._efficientzerotensor(3, device=device)
        self.assertEqual(x.device, y.device)

# we implemented custom deallocation for subclasses, so it behooves
# us to make sure all of these bits work.  We'll use __del__ to
# track if objects die or not
class Tracker:
    # [CLASS DOC] Helper used to detect object destruction: each instance holds a one-element
    # list marker which is set to True when the instance is deleted. Used by subclass
    # deallocation tests.
    def __init__(self, marker):
        # [FUNCTION DOC] Store the marker list on the instance.
        self.marker = marker

    @staticmethod
    def make():
        # [FUNCTION DOC] Static factory returning a pair of a fresh marker list (initially False)
        # and a new Tracker wrapping it.
        marker = [False]
        return marker, Tracker(marker)

    def __del__(self):
        # [FUNCTION DOC] Destructor: sets the first element of the marker list to True to record
        # the instance was destroyed.
        self.marker[0] = True

@contextlib.contextmanager
def disable_gc():
    # [FUNCTION DOC] Context manager that disables the garbage collector for the block if it was
    # enabled and re-enables it afterward (even on exception); if collection was already disabled
    # it simply yields and changes nothing.
    if gc.isenabled():
        try:
            gc.disable()
            yield
        finally:
            gc.enable()
    else:
        yield

class TestTorch(TestCase):
    # [CLASS DOC] Main CPU-oriented test case for torch API behaviors (indexing, serialization,
    # subclasses, storages, views, numerics, random, deallocation). Sets exact_dtype; many tests
    # also use module-level helpers such as Tracker and disable_gc.
    exact_dtype = True

    def test_dir(self):
        # [FUNCTION DOC] Smoke test: calls dir on the torch module and expects no exception.
        dir(torch)

    def test_wildcard_import(self):
        # [FUNCTION DOC] Smoke test: executes a wildcard import from torch and expects
        # no exception.
        exec('from torch import *')

    def test_newaxis_numpy_comparison(self):
        # [FUNCTION DOC] Compares indexing a tensor with combinations of None, Ellipsis and
        # integers against numpy for 1D and 2D tensors.
        def run_test(tensor, *idx):
            # [FUNCTION DOC] HELPER: Index the tensor and its numpy view with the same index tuple
            # and assert equal results.
            npt = tensor.numpy()
            self.assertEqual(tensor[idx], npt[idx])

        # 1D Tensor Tests
        x = torch.arange(0, 10)
        cases = [
            [None],
            [None, None],
            [Ellipsis, None],
            [None, Ellipsis],
            [2, None],
            [None, 2],
            [Ellipsis, None, 2],
            [Ellipsis, 2, None],
            [2, Ellipsis, None],
            [2, None, Ellipsis],
            [None, 2, Ellipsis],
            [None, Ellipsis, 2],
        ]

        for case in cases:
            run_test(x, *case)

        # 2D Tensor Tests
        x = torch.arange(0, 12).view(3, 4)
        cases = [
            [None],
            [None, None],
            [None, None, None],
            [Ellipsis, None],
            [Ellipsis, None, None],
            [None, Ellipsis],
            [None, Ellipsis, None],
            [None, None, Ellipsis],
            [2, None],
            [2, None, Ellipsis],
            [2, Ellipsis, None],
            [None, 2, Ellipsis],
            [Ellipsis, 2, None],
            [Ellipsis, None, 2],
            [None, Ellipsis, 2],
            [1, 2, None],
            [1, 2, Ellipsis, None],
            [1, Ellipsis, 2, None],
            [Ellipsis, 1, None, 2],
            [Ellipsis, 1, 2, None],
            [1, None, 2, Ellipsis],
            [None, 1, Ellipsis, 2],
            [None, 1, 2, Ellipsis],
        ]

        for case in cases:
            run_test(x, *case)

    def _consecutive(self, size, start=1):
        # [FUNCTION DOC] HELPER: Return a float tensor holding the consecutive numbers beginning
        # at start (default 1) with the given size, built from a cumulative sum.
        sequence = torch.ones(torch.tensor(size).prod(0)).cumsum(0)
        sequence.add_(start - 1)
        return sequence.resize_(*size)

    def test_newindex(self):
        # [FUNCTION DOC] Tests index assignment: partial assignment with integers, tuples and long
        # index tensors round-trips and can be reset to zero, and invalid indices (too many
        # indices, float indices, float slices) raise IndexError or TypeError as expected.
        reference = self._consecutive((3, 3, 3))
        # This relies on __index__() being correct - but we have separate tests for that

        def checkPartialAssign(index):
            # [FUNCTION DOC] HELPER: Assign the consecutive-values slice at the given index
            # into a zero tensor, check it matches, set it back to zero, and check the tensor
            # is all zeros.
            reference = torch.zeros(3, 3, 3)
            reference[index] = self._consecutive((3, 3, 3))[index]
            self.assertEqual(reference[index], self._consecutive((3, 3, 3))[index], atol=0, rtol=0)
            reference[index] = 0
            self.assertEqual(reference, torch.zeros(3, 3, 3), atol=0, rtol=0)

        checkPartialAssign(0)
        checkPartialAssign(1)
        checkPartialAssign(2)
        checkPartialAssign((0, 1))
        checkPartialAssign((1, 2))
        checkPartialAssign((0, 2))
        checkPartialAssign(torch.LongTensor((0, 2)))

        with self.assertRaises(IndexError):
            reference[1, 1, 1, 1] = 1
        with self.assertRaises(IndexError):
            reference[1, 1, 1, (1, 1)] = 1
        with self.assertRaises(IndexError):
            reference[3, 3, 3, 3, 3, 3, 3, 3] = 1
        with self.assertRaises(IndexError):
            reference[0.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, :, 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, ..., 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, :, 0.0] = 1

    # Test `torch._check*` functions
    def test_check(self):
        # [FUNCTION DOC] Tests the torch._check family (check, index, value, type,
        # not-implemented): true passes, false raises the matching exception type with a default
        # message, a callable message, a plain string message, and messages built from tensors or
        # formatted strings; non-bool conditions raise TypeError mentioning cond must be a bool,
        # with a hint to check_tensor_all for tensors.
        test_cases = [
            # check function, expected error
            (torch._check, RuntimeError),
            (torch._check_index, IndexError),
            (torch._check_value, ValueError),
            (torch._check_type, TypeError),
            (torch._check_not_implemented, NotImplementedError),
        ]

        for check_fn, expected_error in test_cases:
            # cond=True should not raise an error
            check_fn(True)

            # Test default failure message for cond=False
            default_message = 'Expected cond to be True'
            with self.assertRaisesRegex(expected_error, default_message):
                check_fn(False)

            # Test a simple failure message
            message = 'message'
            with self.assertRaisesRegex(expected_error, message):
                check_fn(False, lambda: message)

            # A plain string message (not wrapped in a callable) should be
            # accepted and used directly as the error message.
            with self.assertRaisesRegex(expected_error, 'plain string error text'):
                check_fn(False, 'plain string error text')

            # Test message with tensor
            def message():
                # [FUNCTION DOC] HELPER: Return a small arange tensor; used as a lazily computed
                # failure message containing a tensor.
                return torch.arange(4)

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(False, message)

            # Test format string message
            def message():
                # [FUNCTION DOC] HELPER: Return a formatted string combining a string, a list
                # of mixed values, a bool, an int and a tensor; used as a lazily computed
                # failure message.
                return f"{'test'} {[1, 2, 'a', True]} {True} {100} {torch.arange(4)}"

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(False, message)

            # Test incorrect `cond` arg type
            with self.assertRaisesRegex(TypeError, 'cond must be a bool'):
                check_fn('wrong type')

            with self.assertRaisesRegex(
                TypeError, r'cond must be a bool.*torch[.]_check_tensor_all'
            ):
                check_fn(torch.tensor(True))

    # FIXME: move to indexing test suite
    def test_index_add(self):
        # [FUNCTION DOC] Tests index_add_ over all device types, contiguity combinations of
        # destination, source and index, int and long index dtypes, with and without alpha,
        # against a loop-based reference.
        for device in get_all_device_types():
            for dest_contig, src_contig, index_contig in product([True, False], repeat=3):
                for other_sizes in ((), (4, 5)):
                    for dtype in [torch.int, torch.long]:
                        num_copy, num_dest = 3, 3
                        dest = torch.randn(num_dest, *other_sizes, device=device)
                        if not dest_contig:
                            dest = make_tensor(dest.shape, device=device, dtype=dest.dtype, noncontiguous=True)
                        src = torch.randn(num_copy, *other_sizes, device=device)
                        if not src_contig:
                            src = noncontiguous_like(src)
                        idx = torch.randperm(num_dest, dtype=dtype, device=device).narrow(0, 0, num_copy)
                        if not index_contig:
                            idx = noncontiguous_like(idx)
                        # index_add_ without alpha argument
                        dest2 = dest.clone()
                        dest.index_add_(0, idx, src)
                        for i in range(idx.size(0)):
                            dest2[idx[i]] += src[i]
                        self.assertEqual(dest, dest2)
                        # index_add_ with alpha argument
                        dest2 = dest.clone()
                        dest.index_add_(0, idx, src, alpha=2)
                        for i in range(idx.size(0)):
                            dest2[idx[i]] += src[i] * 2
                        self.assertEqual(dest, dest2)

    # FIXME: resolve comment below and move this to indexing test suite
    # add coverage for issue with atomic add that appeared only for
    # specific dtypes on cuda:
    # https://github.com/pytorch/pytorch/issues/29153
    def test_index_add_all_dtypes(self):
        # [FUNCTION DOC] For all math dtypes on all device types and int/long index dtypes, adds a
        # tensor into zeros with index_add (also with alpha of negative one) and checks the result
        # equals the tensor or its negation (coverage for an atomic-add issue on CUDA).
        for device in get_all_device_types():
            for dtype in get_all_math_dtypes(device):
                for idx_dtype in [torch.int, torch.long]:
                    size = [5, 5]
                    if dtype.is_floating_point or dtype.is_complex:
                        tensor = torch.rand(size, dtype=dtype, device=device)
                    elif dtype.is_signed:
                        tensor = torch.randint(-5, 15, size, dtype=dtype, device=device)
                    else:
                        tensor = torch.randint(0, 10, size, dtype=dtype, device=device)

                    # index_add calls atomicAdd on cuda.
                    zeros = torch.zeros(size, dtype=dtype, device=device)

                    added = zeros.index_add(0, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor)
                    self.assertEqual(added, tensor)

                    added = zeros.index_add(0, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor, alpha=-1)
                    self.assertEqual(added, -tensor)

    @unittest.mock.patch.object(torch._dynamo.config, "suppress_errors", False)
    @set_default_dtype(torch.double)
    def test_index_add_correctness(self):
        # [FUNCTION DOC] Runs with a double default dtype and Dynamo error suppression off. For
        # many dtypes, devices, dims and shapes, compares index_add with long indices (the
        # scatter-add path) against a reference scaled by alpha, with dtype- and device-dependent
        # tolerances; also checks out-of-range indices raise RuntimeError.
        # Check whether index_add can get correct result when
        # alpha is 1, and dtype of index is torch.long,
        # i.e., using scatter_add
        def helper(dim, dtype, device, size_result, size_source):
            # [FUNCTION DOC] HELPER: Build a zero tensor and random long index and source for the
            # dtype, compute a reference by index_add with alpha 2 divided by 2, compute
            # index_add, and assert closeness with tolerances chosen by device and dtype.
            tensor = torch.zeros(size_result, dtype=dtype, device=device)
            index = torch.randint(0, size_result[dim], (size_source[dim],),
                                  dtype=torch.long, device=device)
            if dtype.is_floating_point or dtype.is_complex:
                source = torch.rand(size_source, dtype=dtype, device=device)
            elif dtype.is_signed:
                source = torch.randint(-2, 5, size_source, dtype=dtype, device=device)
            else:
                source = torch.randint(0, 5, size_source, dtype=dtype, device=device)

            ref_out = tensor.index_add(dim, index, source, alpha=2.) / 2.
            ref_out = ref_out.to(dtype=dtype)
            out = tensor.index_add(dim, index, source)

            # Determine tolerances based on dtype and device
            # Low-precision types (float16, bfloat16) on GPU have non-deterministic
            # accumulation order, leading to larger rounding differences.
            # See: https://github.com/pytorch/pytorch/issues/91184
            if device == 'cuda' and dtype in (torch.half, torch.bfloat16):
                # Relaxed tolerance for low-precision GPU accumulation
                atol, rtol = 1e-1, 1e-1
            elif device == 'cuda':
                atol, rtol = 1e-2, 1e-2
            else:
                # scatter_add uses fp32 as accumulate type, while index_add doesn't.
                atol, rtol = 1e-2, 1e-2

            self.assertEqual(out, ref_out.to(dtype=dtype), atol=atol, rtol=rtol)

        for dim in [-1, -2, -3]:
            for dtype in all_types_and_complex_and(torch.half, torch.bfloat16):
                for device in get_all_device_types():
                    for size in [(2, 512, 256), (5, 256, 256)]:
                        helper(dim, dtype, device, size, size)

                # Check bound
                result = torch.zeros(1, 512, 256, dtype=dtype)
                source = torch.ones(1, 512, 256, dtype=dtype)
                index = torch.ones(257).to(dtype=torch.long)
                self.assertRaises(RuntimeError, lambda: result.index_add_(dim, index, source))
                index = (torch.ones(256) * 257).to(dtype=torch.long)
                self.assertRaises(RuntimeError, lambda: result.index_add_(dim, index, source))

    def test_index_add_cornercase(self):
        # [FUNCTION DOC] For each device, checks index_add of a 3D source into a scalar
        # destination raises RuntimeError saying the source shape must match the destination shape
        # excluding the dimension.
        for device in get_all_device_types():
            dest = torch.randn((), device=device)
            index = torch.tensor([0], device=device)
            source = torch.randn(1, 1, 1, device=device)
            with self.assertRaisesRegex(
                RuntimeError,
                r"source tensor shape must match self tensor shape, excluding the specified dimension",
            ):
                dest.index_add(0, index, source)

    def test_linspace_logspace(self):
        # [FUNCTION DOC] Checks that linspace and logspace outputs do not require grad even when
        # start and/or end tensors require grad, for step counts 0, 1 and 2.
        # Ensure the output does not require grad regardless of inputs requiring guard or not.
        # The output of factory functions should not be part of any computational graph.
        start = 0.0
        end = 3.0

        for step in [0, 1, 2]:
            self.assertFalse(
                torch.linspace(
                    torch.tensor(start, requires_grad=True),
                    torch.tensor(end, requires_grad=True), step
                ).requires_grad
            )
            self.assertFalse(torch.linspace(torch.tensor(start, requires_grad=True), end, step).requires_grad)
            self.assertFalse(torch.linspace(start, torch.tensor(end, requires_grad=True), step).requires_grad)
            self.assertFalse(
                torch.logspace(
                    torch.tensor(start, requires_grad=True),
                    torch.tensor(end, requires_grad=True), step
                ).requires_grad
            )
            self.assertFalse(torch.logspace(torch.tensor(start, requires_grad=True), end, step).requires_grad)
            self.assertFalse(torch.logspace(start, torch.tensor(end, requires_grad=True), step).requires_grad)

    # FIXME: move to shape ops test suite
    def test_unflatten(self):
        # [FUNCTION DOC] Tests Tensor.unflatten with tuple, list and Size sizes, a -1 inferred
        # size, empty tensors and zero-size dims, and that a string dim raises TypeError with the
        # argument message.
        # test args: tensor, int, sizes
        self.assertEqual(torch.tensor([]).unflatten(0, (0, 1)), torch.empty(0, 1))
        self.assertEqual(torch.tensor([1]).unflatten(0, (1, 1)), torch.tensor([[1]]))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, (2, 2)), torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, [2, 2]), torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, torch.Size([2, 2])), torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.ones(2, 10).unflatten(1, (5, 2)), torch.ones(2, 5, 2))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, (-1, 2)),
                         torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.ones(2, 10).unflatten(1, (5, -1)),
                         torch.ones(2, 5, 2))
        self.assertEqual(torch.ones(2, 10).unflatten(1, (-1,)),
                         torch.ones(2, 10))
        self.assertEqual(torch.ones(2, 3 * 4 * 5 * 6).unflatten(1, (3, 4, -1, 6)),
                         torch.ones(2, 3, 4, 5, 6))
        self.assertEqual(torch.ones(2, 0, 2).unflatten(1, (3, -1, 4, 5)),
                         torch.ones(2, 3, 0, 4, 5, 2))

        # test invalid args: tensor, str, sizes
        with self.assertRaisesRegex(TypeError, r"unflatten\(\): argument 'dim' \(position 1\) must be int, not str"):
            torch.tensor([1]).unflatten('A', (1, 1))

        # test other invalid arguments
        with self.assertRaisesRegex(RuntimeError, r"sizes must be non-empty"):
            torch.tensor([1]).unflatten(0, [])
        with self.assertRaisesRegex(RuntimeError, r"Provided sizes \[2, 2\] don't multiply up to the size of dim 0 \(1\)"):
            torch.tensor([1]).unflatten(0, [2, 2])
        with self.assertRaisesRegex(RuntimeError, r".*Dimension specified as 0 but tensor has no dimensions.*"):
            torch.tensor(1).unflatten(0, [0])
        with self.assertRaisesRegex(RuntimeError, r"only one dimension can be inferred"):
            torch.randn(5, 10).unflatten(1, (-1, -1))
        with self.assertRaisesRegex(RuntimeError,
                                    r"Provided sizes \[-1, 4\] don't multiply up to the size of dim 1 \(10\)"):
            torch.randn(5, 10).unflatten(1, (-1, 4))
        with self.assertRaisesRegex(RuntimeError,
                                    r"the unspecified dimension size -1 can be any value and is ambiguous"):
            torch.randn(2, 0).unflatten(1, (2, -1, 0))

    # Test that warnings generated from C++ are translated to the correct type
    def test_warn_types(self):
        # [FUNCTION DOC] Checks warnings emitted from C++ map to the right Python categories: the
        # plain warn helper yields exactly one UserWarning and the deprecation helper exactly one
        # DeprecationWarning, each matching its test message.
        test_cases = [
            # function, warning type, message
            (torch._C._warn, UserWarning, r"Test message for TORCH_WARN"),
            (torch._C._warn_deprecation, DeprecationWarning, r"Test message for TORCH_WARN_DEPRECATION"),
        ]

        for fn, warning_type, message in test_cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.resetwarnings()
                warnings.filterwarnings('always', category=warning_type)
                fn()

                self.assertEqual(len(w), 1, msg=lambda msg: f'{msg}\n{warning_type} not raised')
                warning = w[0].message
                self.assertTrue(isinstance(warning, warning_type), msg=lambda msg: f'{msg}\n{warning_type} not raised')
                self.assertTrue(re.search(
                    message,
                    str(warning)))

    def test_structseq_repr(self):
        # [FUNCTION DOC] Checks the repr of a structured return value (result of max over a dim)
        # equals the expected multi-line string showing values and indices tensors.
        a = torch.arange(250).reshape(5, 5, 10)
        expected = """
        torch.return_types.max(
        values=tensor([[ 40,  41,  42,  43,  44,  45,  46,  47,  48,  49],
                [ 90,  91,  92,  93,  94,  95,  96,  97,  98,  99],
                [140, 141, 142, 143, 144, 145, 146, 147, 148, 149],
                [190, 191, 192, 193, 194, 195, 196, 197, 198, 199],
                [240, 241, 242, 243, 244, 245, 246, 247, 248, 249]]),
        indices=tensor([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]))"""
        self.assertEqual(repr(a.max(1)), textwrap.dedent(expected).strip())

    def test_is_same_size(self):
        # [FUNCTION DOC] Checks is_same_size for regular tensors with equal and differing shapes
        # and for nested tensors, and that mixing a regular and a nested tensor raises
        # RuntimeError about both needing to be nested.
        t1 = torch.empty(3, 4, 9, 10)
        t2 = torch.empty(3, 4)
        t3 = torch.empty(1, 9, 3, 3)
        t4 = torch.empty(3, 4, 9, 10)

        self.assertFalse(t1.is_same_size(t2))
        self.assertFalse(t1.is_same_size(t3))
        self.assertTrue(t1.is_same_size(t4))

        nt1 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(3, 4), torch.ones(5, 4)])
        nt2 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4)])
        nt3 = torch.nested.nested_tensor([torch.ones(2, 4, 5), torch.ones(2, 6, 5)])
        nt4 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(3, 4), torch.ones(5, 4)])

        self.assertFalse(nt1.is_same_size(nt2))
        self.assertFalse(nt1.is_same_size(nt3))
        self.assertTrue(nt1.is_same_size(nt4))
        with self.assertRaisesRegex(RuntimeError, "Expected both self and other to be nested tensors."):
            t1.is_same_size(nt1)

        with self.assertRaisesRegex(RuntimeError, "Expected both self and other to be nested tensors."):
            nt1.is_same_size(t1)

    def test_tensor_set(self):
        # [FUNCTION DOC] Checks Tensor.set_ with a tensor, with a storage plus offset, size and
        # optionally stride (positional and keyword forms, with keyword names
        # source/storage_offset/size/stride), and with bool tensors; storages are shared and
        # sizes/strides are as requested.
        t1 = torch.tensor([])
        t2 = torch.empty(3, 4, 9, 10).uniform_()
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        size = torch.Size([9, 3, 4, 10])
        t1.set_(t2.storage(), 0, size)
        self.assertEqual(t1.size(), size)
        t1.set_(t2.storage(), 0, tuple(size))
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), (120, 40, 10, 1))
        stride = (10, 360, 90, 1)
        t1.set_(t2.storage(), 0, size, stride)
        self.assertEqual(t1.stride(), stride)
        t1.set_(t2.storage(), 0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        # test argument names
        t1 = torch.tensor([])
        # 1. case when source is tensor
        t1.set_(source=t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 2. case when source is storage
        t1.set_(source=t2.storage())
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 3. case when source is storage, and other args also specified
        t1.set_(source=t2.storage(), storage_offset=0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        t1 = torch.tensor([True, True], dtype=torch.bool)
        t2 = torch.tensor([False, False], dtype=torch.bool)
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)

    def test_tensor_set_errors(self):
        # [FUNCTION DOC] Checks set_ errors: an absurd storage offset raises RuntimeError
        # (out of bounds), a size larger than the storage resizes the storage in place to
        # cover the offset, and mismatched dtype or device (CPU/CUDA) storages or tensors
        # raise RuntimeError.
        f_cpu = torch.randn((2, 3), dtype=torch.float32)
        d_cpu = torch.randn((2, 3), dtype=torch.float64)

        storage_offset = 0x41414141
        with self.assertRaisesRegex(RuntimeError, "out of bounds for storage of size"):
            t = torch.randn(1)
            t.set_(t.untyped_storage(), storage_offset, t.size())

        # if size changes, set_ will resize the storage inplace
        t = torch.randn(1)
        size = torch.Size([2, 3])
        t.set_(t.untyped_storage(), storage_offset, size)
        self.assertEqual(t.storage_offset(), storage_offset)
        self.assertEqual(t.untyped_storage().nbytes(), (storage_offset + size[0] * size[1]) * 4)

        # change dtype
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu.storage()))
        self.assertRaises(RuntimeError,
                          lambda: f_cpu.set_(d_cpu.storage(), 0, d_cpu.size(), d_cpu.stride()))
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu))

        # change device
        if torch.cuda.is_available():
            f_cuda = torch.randn((2, 3), dtype=torch.float32, device='cuda')

            # cpu -> cuda
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_cuda.storage()))
            self.assertRaises(RuntimeError,
                              lambda: f_cpu.set_(f_cuda.storage(), 0, f_cuda.size(), f_cuda.stride()))
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_cuda))

            # cuda -> cpu
            self.assertRaises(RuntimeError, lambda: f_cuda.set_(f_cpu.storage()))
            self.assertRaises(RuntimeError,
                              lambda: f_cuda.set_(f_cpu.storage(), 0, f_cpu.size(), f_cpu.stride()))
            self.assertRaises(RuntimeError, lambda: f_cuda.set_(f_cpu))

    # FIXME: move this test test_testing.py (along with allclose testing)
    # NOTE: test_equal will be deprecated in favor of torch.testing.assert_close
    #   once torch.testing is out of beta
    def test_equal(self):
        # [FUNCTION DOC] Checks Tensor.equal and torch.equal on CPU (and CUDA when available):
        # contiguous and non-contiguous tensors, shape mismatch, empty tensors, different dtypes,
        # negative and conjugate views (equal only when the logical values match, except
        # inductor), same-storage different dtype/strides, and that tensors containing NaN are not
        # equal to themselves.
        for device in ["cpu", "cuda"]:
            if device == "cuda" and not torch.cuda.is_available():
                continue

            # Contiguous, 1D
            t1 = torch.tensor((3., 4., 9., 10.), device=device)
            t2 = t1.contiguous()
            t3 = torch.tensor((1., 9., 3., 10.), device=device)
            t4 = torch.tensor((3., 4., 9.), device=device)
            t5 = torch.tensor([], device=device)
            self.assertTrue(t1.equal(t2))
            self.assertFalse(t1.equal(t3))
            self.assertFalse(t1.equal(t4))
            self.assertFalse(t1.equal(t5))
            self.assertTrue(torch.equal(t1, t2))
            self.assertFalse(torch.equal(t1, t3))
            self.assertFalse(torch.equal(t1, t4))
            self.assertFalse(torch.equal(t1, t5))

            # Non contiguous, 2D
            s = torch.tensor(((1, 2, 3, 4), (5, 6, 7, 8)), device=device)
            s1 = s[:, 1:3]
            s2 = s1.clone()
            s3 = torch.tensor(((2, 3), (6, 7)), device=device)
            s4 = torch.tensor(((0, 0), (0, 0)), device=device)

            self.assertFalse(s1.is_contiguous())
            self.assertTrue(s1.equal(s2))
            self.assertTrue(s1.equal(s3))
            self.assertFalse(s1.equal(s4))
            self.assertTrue(torch.equal(s1, s2))
            self.assertTrue(torch.equal(s1, s3))
            self.assertFalse(torch.equal(s1, s4))

            # Different dtypes
            x = torch.tensor((1, 2, 3), dtype=torch.float, device=device)
            y = torch.tensor((1, 2, 3), dtype=torch.int, device=device)
            z = torch.tensor((1, -1), dtype=torch.int, device=device)
            self.assertTrue(torch.equal(x, y))
            self.assertFalse(torch.equal(z, x))

            # Fast path test: tensor flags, like neg and conj
            neg_0 = torch.tensor((1, 2, 3), dtype=torch.float, device=device)
            neg_1 = neg_0._neg_view()
            self.assertTrue(neg_1.is_neg())
            self.assertEqual(neg_0.data_ptr(), neg_1.data_ptr())
            self.assertEqual(neg_0.storage_offset(), neg_1.storage_offset())
            self.assertEqual(neg_0.stride(), neg_1.stride())
            self.assertEqual(neg_0.size(), neg_1.size())
            self.assertFalse(torch.equal(neg_0, neg_1))
            # FIXME: Disable the following check due to the inductor failure
            # See https://github.com/pytorch/pytorch/issues/100340 and
            # https://github.com/pytorch/pytorch/issues/98175
            if not TEST_WITH_TORCHINDUCTOR:
                self.assertTrue(torch.equal(neg_0, neg_1._neg_view()))

            conj_0 = torch.tensor([1.0 + 2.0j, 2.0 + 1.0j], device=device)
            conj_1 = conj_0.conj()
            self.assertTrue(conj_1.is_conj())
            self.assertEqual(conj_0.data_ptr(), conj_1.data_ptr())
            self.assertEqual(conj_0.storage_offset(), conj_1.storage_offset())
            self.assertEqual(conj_0.stride(), conj_1.stride())
            self.assertEqual(conj_0.size(), conj_1.size())
            self.assertFalse(torch.equal(conj_0, conj_1))
            # FIXME: Disable the following check due to the inductor failure
            # See https://github.com/pytorch/pytorch/issues/100340 and
            # https://github.com/pytorch/pytorch/issues/98175
            if not TEST_WITH_TORCHINDUCTOR:
                self.assertTrue(torch.equal(conj_0, conj_1.conj()))

            # Fast path test: two tensors share the same storage, but different dtype
            s_0 = torch.rand((2, 3), dtype=torch.float, device=device)
            s_1 = s_0.view(dtype=torch.int32)
            self.assertEqual(s_0.data_ptr(), s_1.data_ptr())
            self.assertEqual(s_0.storage_offset(), s_1.storage_offset())
            self.assertEqual(s_0.stride(), s_1.stride())
            self.assertEqual(s_0.size(), s_1.size())
            self.assertFalse(torch.equal(s_0, s_1))

            # Fast path test: two tensors share the same storage, but different strides
            t_0 = torch.rand((2, 3), dtype=torch.float, device=device)
            t_1 = t_0.t()
            self.assertEqual(t_0.data_ptr(), t_1.data_ptr())
            self.assertEqual(t_0.storage_offset(), t_1.storage_offset())
            self.assertNotEqual(t_0.stride(), t_1.stride())
            self.assertNotEqual(t_0.size(), t_1.size())
            self.assertFalse(torch.equal(t_0, t_1))

            # Fast path: tensor containing `nan` is not equal to self
            for dtype in floating_and_complex_types():
                t = torch.tensor([1., float('nan')], dtype=dtype)
                self.assertFalse(torch.equal(t, t))

    def test_element_size(self):
        # [FUNCTION DOC] Compares element_size and itemsize between storage classes and tensors
        # for all basic dtypes, checks they are positive, and checks portable size relations
        # (byte, char and bool are 1; short and int at least 2; long at least 4; ordering
        # between sizes).
        byte = torch.ByteStorage().element_size()
        char = torch.CharStorage().element_size()
        short = torch.ShortStorage().element_size()
        int = torch.IntStorage().element_size()
        long = torch.LongStorage().element_size()
        float = torch.FloatStorage().element_size()
        double = torch.DoubleStorage().element_size()
        bool = torch.BoolStorage().element_size()
        bfloat16 = torch.BFloat16Storage().element_size()
        complexfloat = torch.ComplexFloatStorage().element_size()
        complexdouble = torch.ComplexDoubleStorage().element_size()

        self.assertEqual(byte, torch.ByteTensor().element_size())
        self.assertEqual(byte, torch.ByteTensor().itemsize)
        self.assertEqual(char, torch.CharTensor().element_size())
        self.assertEqual(char, torch.CharTensor().itemsize)
        self.assertEqual(short, torch.ShortTensor().element_size())
        self.assertEqual(short, torch.ShortTensor().itemsize)
        self.assertEqual(int, torch.IntTensor().element_size())
        self.assertEqual(int, torch.IntTensor().itemsize)
        self.assertEqual(long, torch.LongTensor().element_size())
        self.assertEqual(long, torch.LongTensor().itemsize)
        self.assertEqual(float, torch.FloatTensor().element_size())
        self.assertEqual(float, torch.FloatTensor().itemsize)
        self.assertEqual(double, torch.DoubleTensor().element_size())
        self.assertEqual(double, torch.DoubleTensor().itemsize)
        self.assertEqual(bool, torch.BoolTensor().element_size())
        self.assertEqual(bool, torch.BoolTensor().itemsize)
        self.assertEqual(bfloat16, torch.tensor([], dtype=torch.bfloat16).element_size())
        self.assertEqual(bfloat16, torch.tensor([], dtype=torch.bfloat16).itemsize)
        self.assertEqual(complexfloat, torch.tensor([], dtype=torch.complex64).element_size())
        self.assertEqual(complexfloat, torch.tensor([], dtype=torch.complex64).itemsize)
        self.assertEqual(complexdouble, torch.tensor([], dtype=torch.complex128).element_size())
        self.assertEqual(complexdouble, torch.tensor([], dtype=torch.complex128).itemsize)

        self.assertGreater(byte, 0)
        self.assertGreater(char, 0)
        self.assertGreater(short, 0)
        self.assertGreater(int, 0)
        self.assertGreater(long, 0)
        self.assertGreater(float, 0)
        self.assertGreater(double, 0)
        self.assertGreater(bool, 0)
        self.assertGreater(bfloat16, 0)
        self.assertGreater(complexfloat, 0)
        self.assertGreater(complexdouble, 0)

        # These tests are portable, not necessarily strict for your system.
        self.assertEqual(byte, 1)
        self.assertEqual(char, 1)
        self.assertEqual(bool, 1)
        self.assertGreaterEqual(short, 2)
        self.assertGreaterEqual(int, 2)
        self.assertGreaterEqual(int, short)
        self.assertGreaterEqual(long, 4)
        self.assertGreaterEqual(long, int)
        self.assertGreaterEqual(double, float)

    def test_permute(self):
        # [FUNCTION DOC] Checks permute with a random permutation of a 7-dimensional tensor
        # permutes the size correctly and leaves the original tensor size unchanged.
        orig = [1, 2, 3, 4, 5, 6, 7]
        perm = torch.randperm(7).tolist()
        x = torch.empty(*orig).fill_(0)
        new = [i - 1 for i in x.permute(*perm).size()]
        self.assertEqual(perm, new)
        self.assertEqual(x.size(), orig)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_reversed(self):
        # [FUNCTION DOC] Checks the builtin reversed on tensors: 1D, 2D (rows reversed) and 0-dim
        # tensors. Skipped under Dynamo.
        val = torch.arange(0, 10)
        self.assertEqual(reversed(val), torch.arange(9, -1, -1))

        val = torch.arange(1, 10).view(3, 3)
        self.assertEqual(reversed(val), torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]]))

        val = torch.tensor(42)
        self.assertEqual(reversed(val), torch.tensor(42))

    def test_contains(self):
        # [FUNCTION DOC] Checks the in operator for tensors with scalars and tensors (found and
        # not found) and that unsupported operand types (string, list) raise RuntimeError with the
        # message naming the type.
        x = torch.arange(0, 10)
        self.assertEqual(4 in x, True)
        self.assertEqual(12 in x, False)

        x = torch.arange(1, 10).view(3, 3)
        val = torch.arange(1, 4)
        self.assertEqual(val in x, True)
        val += 10
        self.assertEqual(val in x, False)

        self.assertRaisesRegex(
            RuntimeError,
            f"Tensor.__contains__ only supports Tensor or scalar, but you passed in a {str}.",
            lambda: "foo" in x)
        self.assertRaisesRegex(
            RuntimeError,
            f"Tensor.__contains__ only supports Tensor or scalar, but you passed in a {type([1, 2])}.",
            lambda: [1, 2] in x)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_deepcopy_parameter(self):
        # [FUNCTION DOC] Checks deepcopy of a state dict containing Parameters (kept as variables)
        # keeps the Parameter type. Skipped under Dynamo.
        from copy import deepcopy
        l = torch.nn.Linear(10, 1)
        s = l.state_dict(keep_vars=True)
        self.assertEqual(torch.nn.Parameter, type(s['weight']))
        self.assertEqual(torch.nn.Parameter, type(s['bias']))

        s2 = deepcopy(s)
        self.assertEqual(torch.nn.Parameter, type(s2['weight']))
        self.assertEqual(torch.nn.Parameter, type(s2['bias']))

    def test_pickle(self):
        # [FUNCTION DOC] Pickles and unpickles a random tensor and checks equality.
        import pickle
        a = torch.randn(5, 5)
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertEqual(a, b)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_pickle_parameter(self):
        # [FUNCTION DOC] Pickles and unpickles a Parameter and checks it is still a Parameter with
        # the same requires_grad flag. Skipped under Dynamo.
        import pickle
        a = torch.nn.Parameter(torch.randn(5, 5))
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.nn.Parameter))
        self.assertEqual(a.requires_grad, b.requires_grad)
        self.assertEqual(a, b)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_pickle_parameter_no_requires_grad(self):
        # [FUNCTION DOC] Pickles and unpickles a Parameter created with requires_grad off and
        # checks it is a Parameter with the same flag and values.
        import pickle
        a = torch.nn.Parameter(torch.randn(5, 5), requires_grad=False)
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.nn.Parameter))
        self.assertEqual(a.requires_grad, b.requires_grad)
        self.assertEqual(a, b)

    def test_pickle_dtype(self):
        # [FUNCTION DOC] Pickles and unpickles torch.float32 and checks the result is a dtype and
        # is the same singleton object.
        t = torch.float32
        serialized = pickle.dumps(t)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.dtype))
        self.assertEqual(id(b), id(t))

    def test_pickle_size(self):
        # [FUNCTION DOC] Pickles and unpickles a tensor size and checks it is a torch.Size equal
        # to the original.
        a = torch.rand(10).size()
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.Size))
        self.assertEqual(a, b)

    def test_pickle_function(self):
        # [FUNCTION DOC] Pickles and unpickles the tanh function object (issue 37703) and checks
        # it equals the original.
        # https://github.com/pytorch/pytorch/issues/37703
        a = torch.tanh
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertEqual(a, b)

    def test_generator_cpu(self):
        # [FUNCTION DOC] Exercises the CPU Generator API: the default generator equals itself;
        # manual seeds give equal initial seeds; seed gives different initial seeds; set_state on
        # a fresh generator reproduces draws from the source; and restoring the default generator
        # state into another generator reproduces normal_ samples.
        # test default generators are equal
        self.assertEqual(torch.default_generator, torch.default_generator)

        # tests Generator API
        # manual_seed, seed, initial_seed, get_state, set_state
        g1 = torch.Generator()
        g2 = torch.Generator()
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        self.assertEqual(g1.initial_seed(), g2.initial_seed())

        g1.seed()
        g2.seed()
        self.assertNotEqual(g1.initial_seed(), g2.initial_seed())

        g1 = torch.Generator()
        g2_state = g2.get_state()
        g2_randn = torch.randn(1, generator=g2)
        g1.set_state(g2_state)
        g1_randn = torch.randn(1, generator=g1)
        self.assertEqual(g1_randn, g2_randn)

        default_state = torch.default_generator.get_state()
        q = torch.empty(100)
        g1_normal = q.normal_()
        g2 = torch.Generator()
        g2.set_state(default_state)
        g2_normal = q.normal_(generator=g2)
        self.assertEqual(g1_normal, g2_normal)

    def test_invalid_generator_raises(self):
        # [FUNCTION DOC] Checks creating a Generator for an unsupported device name
        # raises RuntimeError.
        self.assertRaises(RuntimeError, lambda: torch.Generator('opengl'))

    def test_pickle_generator(self) -> None:
        # [FUNCTION DOC] For CPU (and CUDA when available) creates a seeded Generator, advances
        # it, pickles and unpickles it, and checks device, initial seed, CUDA offset and full
        # state match.
        devices = ['cpu']
        if torch.cuda.is_available():
            devices += ['cuda']

        for device in devices:
            with self.subTest(device=device):
                generator = torch.Generator(device=device).manual_seed(12345)
                if device != "cpu":
                    generator.set_offset(100)
                torch.randn((100, 100), generator=generator, device=device)  # progress the RNG state

                reserialized: torch.Generator = pickle.loads(pickle.dumps(generator))

                self.assertEqual(generator.device, reserialized.device)
                self.assertEqual(generator.initial_seed(), reserialized.initial_seed())
                if device != "cpu":
                    self.assertEqual(generator.get_offset(), reserialized.get_offset())
                torch.testing.assert_close(generator.get_state(), reserialized.get_state())

    def _sobol_reference_samples(self, scramble: bool) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Return the 8 by 2 expected Sobol samples: the known unscrambled
        # sequence values when scramble is false, otherwise stored reference values for a seeded
        # scrambled engine (only convergence properties are known).
        if not scramble:
            # theoretical values from Joe Kuo 2010
            return torch.tensor(
                [
                    [0., 0.],
                    [0.5, 0.5],
                    [0.75, 0.25],
                    [0.25, 0.75],
                    [0.375, 0.375],
                    [0.875, 0.875],
                    [0.625, 0.125],
                    [0.125, 0.625],
                ],
            )
        else:
            # theoretical values unknown: convergence properties checked
            return torch.tensor(
                [
                    [0.50860737, 0.29320504],
                    [0.07116939, 0.89594537],
                    [0.49354145, 0.11524881],
                    [0.93097717, 0.70244044],
                    [0.87266153, 0.23887917],
                    [0.31021884, 0.57600391],
                    [0.13687253, 0.42054182],
                    [0.69931293, 0.77336788],
                ],
            )

    def test_sobolengine_bounds(self, scramble: bool = False):
        # [FUNCTION DOC] Draws 512 samples from a 100-dimensional SobolEngine and checks all
        # values lie in 0 to 1; scramble selects the scrambled variant.
        engine = torch.quasirandom.SobolEngine(100, scramble=scramble, seed=123456)
        sample = engine.draw(512)
        self.assertTrue(torch.all(sample >= 0))
        self.assertTrue(torch.all(sample <= 1))

    def test_sobolengine_bounds_scrambled(self):
        # [FUNCTION DOC] Runs the Sobol bounds test with scrambling enabled.
        self.test_sobolengine_bounds(scramble=True)

    def test_sobolengine_draw(self, scramble: bool = False):
        # [FUNCTION DOC] Draws as many samples as the reference from a 2-dimensional SobolEngine
        # and compares them (and the count generated) with the reference samples.
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        sample = engine.draw(n=len(ref_sample))
        self.assertEqual(sample, ref_sample)
        self.assertEqual(engine.num_generated, len(ref_sample))

    def test_sobolengine_draw_scrambled(self):
        # [FUNCTION DOC] Runs the Sobol draw test with scrambling enabled.
        self.test_sobolengine_draw(scramble=True)

    def test_sobolengine_first_point(self):
        # [FUNCTION DOC] Checks the first drawn point: all zeros and requested dtype for an
        # unscrambled engine, nonzero for a scrambled one, for float and double.
        for dtype in (torch.float, torch.double):
            engine = torch.quasirandom.SobolEngine(2, scramble=False)
            sample = engine.draw(1, dtype=dtype)
            self.assertTrue(torch.all(sample == 0))
            self.assertEqual(sample.dtype, dtype)
        for dtype in (torch.float, torch.double):
            engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=123456)
            sample = engine.draw(1, dtype=dtype)
            self.assertTrue(torch.all(sample != 0))
            self.assertEqual(sample.dtype, dtype)

    def test_sobolengine_continuing(self, scramble: bool = False):
        # [FUNCTION DOC] Draws half the reference samples, then the rest, and checks the second
        # batch matches the second half of the reference.
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        n_half = len(ref_sample) // 2
        _ = engine.draw(n=n_half)
        sample = engine.draw(n=n_half)
        torch.testing.assert_close(sample, ref_sample[n_half:])

    def test_sobolengine_continuing_scrambled(self):
        # [FUNCTION DOC] Runs the Sobol continuing test with scrambling enabled.
        self.test_sobolengine_continuing(scramble=True)

    def test_sobolengine_reset(self, scramble: bool = False):
        # [FUNCTION DOC] Draws half, resets the engine, checks the generated counter is zero, and
        # checks drawing everything again reproduces the reference.
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        _ = engine.draw(n=len(ref_sample) // 2)
        engine.reset()
        self.assertEqual(engine.num_generated, 0)
        sample = engine.draw(n=len(ref_sample))
        torch.testing.assert_close(sample, ref_sample)

    def test_sobolengine_reset_scrambled(self):
        # [FUNCTION DOC] Runs the Sobol reset test with scrambling enabled.
        self.test_sobolengine_reset(scramble=True)

    def test_sobolengine_fast_forward(self, scramble: bool = False):
        # [FUNCTION DOC] Checks fast_forward skips samples: after skipping 4, the next 4 equal the
        # reference tail; and alternating draw with fast_forward of 1 yields the even-indexed
        # reference samples.
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        engine.fast_forward(4)
        sample = engine.draw(n=4)
        torch.testing.assert_close(sample, ref_sample[4:])
        # alternate fast forwarding with sampling
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.draw())
            else:
                engine.fast_forward(1)
        torch.testing.assert_close(
            ref_sample[[i for i in range(8) if i % 2 == 0]],
            torch.from_numpy(np.concatenate(even_draws)),
        )

    def test_sobolengine_fast_forward_scrambled(self):
        # [FUNCTION DOC] Runs the Sobol fast-forward test with scrambling enabled.
        self.test_sobolengine_fast_forward(scramble=True)

    def test_sobolengine_default_dtype(self):
        # [FUNCTION DOC] Checks SobolEngine draws use the default dtype (float32, or float64 when
        # the default dtype is changed), that an explicit dtype argument overrides it, and that
        # this holds for the first draw of a new engine.
        engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123456)
        # Check that default dtype is correctly handled
        self.assertEqual(engine.draw(n=5).dtype, torch.float32)
        with set_default_dtype(torch.float64):
            engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123456)
            # Check that default dtype is correctly handled (when set to float64)
            self.assertEqual(engine.draw(n=5).dtype, torch.float64)
            # Check that explicitly passed dtype is adhered to
            self.assertEqual(engine.draw(n=5, dtype=torch.float32).dtype, torch.float32)
            # Reinitialize the engine and check that first draw dtype is correctly handled
            engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123456)
            self.assertEqual(engine.draw(n=5, dtype=torch.float32).dtype, torch.float32)

    @skipIfTorchDynamo("np.float64 restored as float32 after graph break.")
    def test_sobolengine_distribution(self, scramble=False):
        # [FUNCTION DOC] Draws 1024 points from a 50-dimensional engine and checks the mean and
        # quartiles per dimension are close to expected uniform values using very loose
        # tolerances. Skipped under Dynamo.
        d = 50
        engine = torch.quasirandom.SobolEngine(d, scramble=scramble, seed=123456)
        sample = engine.draw(1024)
        torch.testing.assert_close(
            torch.mean(sample, dim=0), torch.full((d,), 0.5), atol=2, rtol=2
        )
        torch.testing.assert_close(
            np.percentile(sample, 25, axis=0).astype(np.float64), np.repeat(0.25, d), atol=2, rtol=2
        )
        torch.testing.assert_close(
            np.percentile(sample, 75, axis=0).astype(np.float64), np.repeat(0.75, d), atol=2, rtol=2
        )

    @skipIfTorchDynamo("np.float64 restored as float32 after graph break.")
    def test_sobolengine_distribution_scrambled(self):
        # [FUNCTION DOC] Runs the Sobol distribution test with scrambling enabled. Skipped
        # under Dynamo.
        self.test_sobolengine_distribution(scramble=True)

    def test_sobolengine_draw_base2(self, scramble=False):
        # [FUNCTION DOC] Checks draw_base2 with exponent 2 returns the first four reference
        # samples and then the next four.
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        sample = engine.draw_base2(2)
        self.assertEqual(ref_sample[:4], sample)
        # resampling still having N=2**n
        sample = engine.draw_base2(2)
        self.assertEqual(ref_sample[4:8], sample)

    def test_sobolengine_draw_base2_scrambled(self):
        # [FUNCTION DOC] Runs the Sobol draw_base2 test with scrambling enabled.
        self.test_sobolengine_draw_base2(scramble=True)

    def test_sobolengine_raise(self):
        # [FUNCTION DOC] Checks constructing a SobolEngine with a dimension above the maximum
        # raises ValueError.
        maxdim = torch.quasirandom.SobolEngine.MAXDIM
        with self.assertRaises(ValueError):
            torch.quasirandom.SobolEngine(maxdim + 1)

    def test_sobol_invalid_inputs(self):
        # [FUNCTION DOC] Checks the low-level Sobol ops raise ValueError for invalid inputs:
        # mismatched dimension, too many points requested to fast-forward, and a dimension outside
        # the supported range.
        quasi = torch.ones(2, dtype=torch.long)
        sobolstate = torch.ones(2, 30, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "dimension must match"):
            torch._sobol_engine_ff_(quasi, 1, sobolstate, 1250999896764, 0)
        with self.assertRaisesRegex(ValueError, "at most"):
            torch._sobol_engine_ff_(quasi, 1, sobolstate, 2, 2**30 - 1)
        with self.assertRaisesRegex(ValueError, "dimension must be between"):
            torch._sobol_engine_initialize_state_(sobolstate, 1250999896764)

    def test_sobolengine_high_dim(self):
        # [FUNCTION DOC] Draws two points from a 1111-dimensional unscrambled engine and checks
        # the first is all zeros and the second all 0.5.
        engine = torch.quasirandom.SobolEngine(1111, scramble=False, seed=123456)
        samples1 = engine.draw()
        vals1, counts1 = torch.unique(samples1, return_counts=True)
        samples2 = engine.draw()
        vals2, counts2 = torch.unique(samples2, return_counts=True)
        self.assertEqual(vals1.item(), 0.0)
        self.assertEqual(counts1.item(), 1111)
        self.assertEqual(vals2.item(), 0.5)
        self.assertEqual(counts1.item(), 1111)

    def test_parsing_int64(self):
        # [FUNCTION DOC] Checks integer arguments parsing: a 0-dim integer tensor works as a dim
        # argument and a float tensor raises TypeError.
        # accepts integer arguments
        x = torch.cumsum(torch.ones(5, 5), 0)
        self.assertEqual(x, torch.cumsum(torch.ones(5, 5), torch.tensor(0)))
        # doesn't accept floating point variables
        self.assertRaises(TypeError, lambda: torch.cumsum(torch.ones(5, 5), torch.tensor(0.)))

    def test_parsing_double(self):
        # [FUNCTION DOC] Checks floating arguments parsing for isclose tolerances: Python numbers
        # and int or float tensors are accepted, but a tensor requiring grad raises TypeError.
        # accepts floating point and integer arguments
        x = torch.randn(2, 3)
        torch.isclose(x, x, 1, 1)
        self.assertTrue(torch.isclose(x, x, 1, 1).all())
        self.assertTrue(torch.isclose(x, x, 1.5, 1.).all())
        # accepts floating point and integer tensors
        self.assertTrue(torch.isclose(x, x, torch.tensor(1), torch.tensor(1)).all())
        self.assertTrue(torch.isclose(x, x, torch.tensor(1.5), torch.tensor(1.)).all())
        # doesn't accept variables with requires_grad
        self.assertRaises(TypeError,
                          lambda: torch.isclose(x, x, torch.tensor(1.5), torch.tensor(1., requires_grad=True)).all())

    def test_parsing_intlist(self):
        # [FUNCTION DOC] Checks size arguments given as tensors or numpy integers are accepted by
        # ones, while float tensors or floats, multi-element tensors, extra positional arguments
        # and invalid combinations raise TypeError.
        #  parse with integer variables
        self.assertEqual(torch.Size([3, 4]), torch.ones((torch.tensor(3), torch.tensor(4))).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones(torch.tensor(3), torch.tensor(4)).shape)
        # parse with numpy integers
        self.assertEqual(torch.Size([3, 4]), torch.ones((np.array(3), np.int64(4))).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones(np.array(3), np.int64(4)).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones((np.int64(3), np.array(4))).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones(np.int64(3), np.array(4)).shape)

        # fail parse with float variables
        self.assertRaises(TypeError, lambda: torch.ones((torch.tensor(3.), torch.tensor(4))))
        # fail parse with numpy floats
        self.assertRaises(TypeError, lambda: torch.ones((3., torch.tensor(4))))
        self.assertRaises(TypeError, lambda: torch.ones((np.array(3.), torch.tensor(4))))

        # fail parse with > 1 element variables
        self.assertRaises(TypeError, lambda: torch.ones(torch.tensor(3, 3)))
        self.assertRaises(TypeError, lambda: torch.ones(torch.tensor(3, 3)))
        self.assertRaises(TypeError, lambda: torch.ones(np.array(3, 3)))
        self.assertRaises(TypeError, lambda: torch.ones(np.array(3, 3)))

        # fail parse with additional positional args after intlist arg
        self.assertRaisesRegex(TypeError,
                               "received an invalid combination of arguments",
                               lambda: torch.LongTensor((6, 0), 1, 1, 0))
        self.assertRaisesRegex(TypeError,
                               "missing 1 required positional arguments",
                               lambda: torch.tensor().new_zeros((5, 5), 0))

        # ensure ones() throws an error when extra positional (non-keyword) arguments are given.
        self.assertRaises(TypeError, lambda: torch.ones((3, 3), torch.float32))

    def test_from_buffer(self):
        # [FUNCTION DOC] Checks Storage.from_buffer for byte, short, int, float and bool storages
        # with big and little endian byte orders on bytearrays, including element counts, values,
        # type string and class.
        a = bytearray([1, 2, 3, 4])
        self.assertEqual(torch.ByteStorage.from_buffer(a).tolist(), [1, 2, 3, 4])
        shorts = torch.ShortStorage.from_buffer(a, 'big')
        self.assertEqual(shorts.size(), 2)
        self.assertEqual(shorts.tolist(), [258, 772])
        ints = torch.IntStorage.from_buffer(a, 'little')
        self.assertEqual(ints.size(), 1)
        self.assertEqual(ints[0], 67305985)
        f = bytearray([0x40, 0x10, 0x00, 0x00])
        floats = torch.FloatStorage.from_buffer(f, 'big')
        self.assertEqual(floats.size(), 1)
        self.assertEqual(floats[0], 2.25)

        f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
        bools = torch.BoolStorage.from_buffer(f, 'big')
        self.assertEqual(bools.size(), 8)
        self.assertEqual(bools.tolist(), [False, True, True, True, True, True, True, True])
        self.assertEqual(bools.type(), 'torch.BoolStorage')
        self.assertTrue(isinstance(bools, torch.BoolStorage))

        f = bytearray(b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9')
        bools = torch.BoolStorage.from_buffer(f, 'big')
        self.assertEqual(bools.size(), 19)

        f = bytearray(b'\0x4A')
        bools = torch.BoolStorage.from_buffer(f, 'big')
        self.assertEqual(bools.size(), 4)
        self.assertEqual(bools.tolist(), [False, True, True, True])
        bytes = torch.ByteStorage.from_buffer(a)
        self.assertEqual(bytes.nbytes(), 4)
        self.assertEqual(bytes.tolist(), [1, 2, 3, 4])
        self.assertTrue(isinstance(bytes, torch.ByteStorage))

    def test_storage_error(self):
        # [FUNCTION DOC] Exhaustively checks constructor errors of the legacy typed storage
        # classes (all registered classes except the generic ones, CUDA ones only if CUDA is
        # available): instantiating the legacy base class, passing device or dtype, unexpected
        # keywords, too many positional arguments, invalid data or argument types, wrap_storage
        # misuse and device mismatch, quantized CUDA creation, and the corresponding TypedStorage
        # constructor errors, plus fill_ with a mismatched storage.
        quantized_storages = [
            torch.QInt32Storage,
            torch.QInt8Storage,
            torch.QUInt2x4Storage,
            torch.QUInt4x2Storage,
            torch.QUInt8Storage,
        ]

        with self.assertRaisesRegex(RuntimeError, r"Only child classes of _LegacyStorage can be instantiated"):
            torch.storage._LegacyStorage()

        for storage_class in torch._storage_classes:
            if storage_class in [torch.UntypedStorage, torch.TypedStorage]:
                continue

            device = 'cuda' if storage_class.__module__ == 'torch.cuda' else 'cpu'
            dtype = storage_class.dtype

            if device == 'cuda' and not torch.cuda.is_available():
                continue

            # Legacy <type>Storage constructor errors
            with self.assertRaisesRegex(RuntimeError, r"'device' cannot be specified"):
                storage_class(device='cpu')

            with self.assertRaisesRegex(RuntimeError, r"'dtype' cannot be specified"):
                storage_class(dtype=torch.float)

            with self.assertRaisesRegex(TypeError, r"got an unexpected keyword"):
                storage_class(sdlkjf=torch.float)

            with self.assertRaisesRegex(RuntimeError, r"Too many positional arguments"):
                storage_class(0, 0)

            with self.assertRaisesRegex(TypeError, r"invalid data type"):
                storage_class('string')

            with self.assertRaisesRegex(TypeError, r"Argument type not recognized"):
                storage_class(torch.tensor([]))

            s = storage_class()

            with self.assertRaisesRegex(RuntimeError, r"No positional arguments"):
                storage_class(0, wrap_storage=s.untyped())

            with self.assertRaisesRegex(TypeError, r"must be UntypedStorage"):
                storage_class(wrap_storage=s)

            if torch.cuda.is_available():
                if storage_class in quantized_storages:
                    with self.assertRaisesRegex(RuntimeError, r"Cannot create CUDA storage with quantized dtype"):
                        s.cuda()

                else:

                    if s.is_cuda:
                        s_other_device = s.cpu()
                    else:
                        s_other_device = s.cuda()

                    with self.assertRaisesRegex(RuntimeError, r"Device of 'wrap_storage' must be"):
                        storage_class(wrap_storage=s_other_device.untyped())

            # TypedStorage constructor errors
            with self.assertRaisesRegex(RuntimeError, r"No positional arguments"):
                torch.TypedStorage(0, wrap_storage=s.untyped(), dtype=dtype)

            with self.assertRaisesRegex(RuntimeError, r"Argument 'dtype' must be specified"):
                torch.TypedStorage(wrap_storage=s.untyped())

            with self.assertRaisesRegex(TypeError, r"Argument 'dtype' must be torch.dtype"):
                torch.TypedStorage(wrap_storage=s.untyped(), dtype=0)

            with self.assertRaisesRegex(RuntimeError, r"Argument 'device' should not be specified"):
                torch.TypedStorage(wrap_storage=s.untyped(), dtype=dtype, device=device)

            with self.assertRaisesRegex(TypeError, r"Argument 'wrap_storage' must be UntypedStorage"):
                torch.TypedStorage(wrap_storage=s, dtype=dtype)

            with self.assertRaisesRegex(RuntimeError, r"Storage device not recognized"):
                torch.TypedStorage(dtype=dtype, device='xla')

            if torch.cuda.is_available():
                if storage_class in quantized_storages:
                    with self.assertRaisesRegex(RuntimeError, r"Cannot create CUDA storage with quantized dtype"):
                        torch.TypedStorage(dtype=dtype, device='cuda')

            with self.assertRaisesRegex(TypeError, r"Argument type not recognized"):
                torch.TypedStorage(torch.tensor([]), dtype=dtype, device=device)

            with self.assertRaisesRegex(RuntimeError, r"Too many positional arguments"):
                torch.TypedStorage(0, 0, dtype=dtype, device=device)

            if isinstance(s, torch.TypedStorage):
                s_other = torch.TypedStorage([1, 2, 3, 4], device=device, dtype=dtype)

                with self.assertRaisesRegex(RuntimeError, r'cannot set item'):
                    s.fill_(s_other)

    def test_storage_error_no_attribute(self):
        # [FUNCTION DOC] Checks that CUDA storage classes raise RuntimeError (not available
        # for CUDA storage) for from_buffer, the weak-pointer constructor and the
        # shared-filename constructor.
        storage_classes = [
            torch.cuda.ByteStorage,
            torch.cuda.FloatStorage,
        ]
        for storage_class in storage_classes:
            with self.assertRaisesRegex(RuntimeError, r'Not available for CUDA storage'):
                storage_class.from_buffer()

            with self.assertRaisesRegex(RuntimeError, r'Not available for CUDA storage'):
                storage_class._new_with_weak_ptr()

            with self.assertRaisesRegex(RuntimeError, r'Not available for CUDA storage'):
                storage_class._new_shared_filename(0, 0, 0)

    def test_storage_casts(self):
        # [FUNCTION DOC] Builds an int storage and checks conversions to float, half, bfloat16,
        # long, short, double, char, byte (wraps -1 to 255) and bool storages, verifying sizes,
        # values, type strings and dtypes, plus complex float to complex double conversion.
        storage = torch.IntStorage([-1, 0, 1, 2, 3, 4])
        self.assertEqual(storage.size(), 6)
        self.assertEqual(storage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(storage.type(), 'torch.IntStorage')
        self.assertIs(storage.dtype, torch.int32)

        floatStorage = storage.float()
        self.assertEqual(floatStorage.size(), 6)
        self.assertEqual(floatStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(floatStorage.type(), 'torch.FloatStorage')
        self.assertEqual(floatStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(floatStorage.dtype, torch.float32)

        halfStorage = storage.half()
        self.assertEqual(halfStorage.size(), 6)
        self.assertEqual(halfStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(halfStorage.type(), 'torch.HalfStorage')
        self.assertEqual(halfStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(halfStorage.dtype, torch.float16)

        bfloat16Storage = storage.bfloat16()
        self.assertEqual(bfloat16Storage.size(), 6)
        self.assertEqual(bfloat16Storage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(bfloat16Storage.type(), 'torch.BFloat16Storage')
        self.assertEqual(bfloat16Storage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(bfloat16Storage.dtype, torch.bfloat16)

        longStorage = storage.long()
        self.assertEqual(longStorage.size(), 6)
        self.assertEqual(longStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(longStorage.type(), 'torch.LongStorage')
        self.assertEqual(longStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(longStorage.dtype, torch.int64)

        shortStorage = storage.short()
        self.assertEqual(shortStorage.size(), 6)
        self.assertEqual(shortStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(shortStorage.type(), 'torch.ShortStorage')
        self.assertEqual(shortStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(shortStorage.dtype, torch.int16)

        doubleStorage = storage.double()
        self.assertEqual(doubleStorage.size(), 6)
        self.assertEqual(doubleStorage.tolist(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(doubleStorage.type(), 'torch.DoubleStorage')
        self.assertEqual(doubleStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(doubleStorage.dtype, torch.float64)

        charStorage = storage.char()
        self.assertEqual(charStorage.size(), 6)
        self.assertEqual(charStorage.tolist(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(charStorage.type(), 'torch.CharStorage')
        self.assertEqual(charStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(charStorage.dtype, torch.int8)

        byteStorage = storage.byte()
        self.assertEqual(byteStorage.size(), 6)
        self.assertEqual(byteStorage.tolist(), [255, 0, 1, 2, 3, 4])
        self.assertEqual(byteStorage.type(), 'torch.ByteStorage')
        self.assertEqual(byteStorage.int().tolist(), [255, 0, 1, 2, 3, 4])
        self.assertIs(byteStorage.dtype, torch.uint8)

        boolStorage = storage.bool()
        self.assertEqual(boolStorage.size(), 6)
        self.assertEqual(boolStorage.tolist(), [True, False, True, True, True, True])
        self.assertEqual(boolStorage.type(), 'torch.BoolStorage')
        self.assertEqual(boolStorage.int().tolist(), [1, 0, 1, 1, 1, 1])
        self.assertIs(boolStorage.dtype, torch.bool)

        complexfloat_storage = torch.ComplexFloatStorage([-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j])
        self.assertEqual(complexfloat_storage.size(), 6)
        self.assertEqual(complexfloat_storage.tolist(), [-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j])
        self.assertEqual(complexfloat_storage.type(), 'torch.ComplexFloatStorage')
        self.assertIs(complexfloat_storage.dtype, torch.complex64)

        complexdouble_storage = complexfloat_storage.complex_double()
        self.assertEqual(complexdouble_storage.size(), 6)
        self.assertEqual(complexdouble_storage.tolist(), [-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j])
        self.assertEqual(complexdouble_storage.type(), 'torch.ComplexDoubleStorage')
        self.assertIs(complexdouble_storage.dtype, torch.complex128)

    def test_storage_byteswap(self):
        # [FUNCTION DOC] Checks untyped storage byteswap for each dtype (float64/32/16, bfloat16,
        # int64/32/16/8, uint8, bool, complex128/64) against expected byte orders for 8, 4, 2 and
        # 1 byte element widths.
        input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        swapped_8bytes = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]
        swapped_4bytes = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
        swapped_2bytes = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
        swapped_1byte = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        storage = torch.storage.TypedStorage(input, dtype=torch.uint8)._untyped_storage

        storage_f64 = storage.__copy__()
        storage_f64.byteswap(torch.float64)
        self.assertEqual(storage_f64.tolist(), swapped_8bytes)

        storage_f32 = storage.__copy__()
        storage_f32.byteswap(torch.float32)
        self.assertEqual(storage_f32.tolist(), swapped_4bytes)

        storage_f16 = storage.__copy__()
        storage_f16.byteswap(torch.float16)
        self.assertEqual(storage_f16.tolist(), swapped_2bytes)

        storage_bf16 = storage.__copy__()
        storage_bf16.byteswap(torch.bfloat16)
        self.assertEqual(storage_bf16.tolist(), swapped_2bytes)

        storage_i64 = storage.__copy__()
        storage_i64.byteswap(torch.int64)
        self.assertEqual(storage_i64.tolist(), swapped_8bytes)

        storage_i32 = storage.__copy__()
        storage_i32.byteswap(torch.int32)
        self.assertEqual(storage_i32.tolist(), swapped_4bytes)

        storage_i16 = storage.__copy__()
        storage_i16.byteswap(torch.int16)
        self.assertEqual(storage_i16.tolist(), swapped_2bytes)

        storage_i8 = storage.__copy__()
        storage_i8.byteswap(torch.int8)
        self.assertEqual(storage_i8.tolist(), swapped_1byte)

        storage_ui8 = storage.__copy__()
        storage_ui8.byteswap(torch.uint8)
        self.assertEqual(storage_ui8.tolist(), swapped_1byte)

        storage_bool = storage.__copy__()
        storage_bool.byteswap(torch.bool)
        self.assertEqual(storage_bool.tolist(), swapped_1byte)

        storage_c128 = storage.__copy__()
        storage_c128.byteswap(torch.complex128)
        self.assertEqual(storage_c128.tolist(), swapped_8bytes)

        storage_c64 = storage.__copy__()
        storage_c64.byteswap(torch.complex64)
        self.assertEqual(storage_c64.tolist(), swapped_4bytes)

    # Test that internal versions of functions related to TypedStorage do not
    # produce a deprecation warning
    def test_typed_storage_internal_no_warning(self):
        # [FUNCTION DOC] Checks that internal TypedStorage helpers (internal constructors, size,
        # resize, sharing, pickling type, element size, deepcopy, data pointer, nbytes, typed
        # storage of a tensor; CUDA variants when available) do not emit the TypedStorage
        # deprecation warning, by turning that warning into an error while calling each.
        s0 = torch.FloatStorage(10)
        s0_untyped = s0.untyped()
        t0 = torch.randn(10)

        funcs = [
            lambda: torch.FloatStorage(_internal=True),
            lambda: torch.TypedStorage(
                dtype=torch.float,
                device='cpu',
                _internal=True),
            lambda: torch.TypedStorage(
                wrap_storage=s0_untyped,
                dtype=s0.dtype,
                _internal=True),
            lambda: torch.FloatStorage._dtype,
            lambda: s0._resize_(20),
            lambda: s0._size(),
            lambda: s0._untyped_storage,
            lambda: s0._is_shared(),
            lambda: s0._share_memory_(),
            lambda: s0._pickle_storage_type(),
            lambda: s0._setitem(slice(0, s0._size()), 1),
            lambda: s0._element_size(),
            lambda: s0._deepcopy({}),
            lambda: s0._data_ptr(),
            lambda: s0._nbytes(),
            lambda: t0._typed_storage(),
        ]

        if torch.cuda.is_available():
            s1 = torch.cuda.FloatStorage(10)
            s1_untyped = s1.untyped()
            t1 = torch.randn(10, device='cuda')

            funcs += [
                lambda: torch.cuda.FloatStorage(_internal=True),
                lambda: torch.TypedStorage(
                    dtype=torch.float,
                    device='cuda',
                    _internal=True),
                lambda: torch.TypedStorage(
                    wrap_storage=s1_untyped,
                    dtype=s1.dtype,
                    _internal=True),
                lambda: torch.cuda.FloatStorage._dtype,
                lambda: s1._resize_(20),
                lambda: s1._size(),
                lambda: s1._untyped_storage,
                lambda: s1._is_shared(),
                lambda: s1._share_memory_(),
                lambda: s1._pickle_storage_type(),
                lambda: s1._setitem(slice(0, s1._size()), 1),
                lambda: s1._element_size(),
                lambda: s1._deepcopy({}),
                lambda: s1._data_ptr(),
                lambda: s1._nbytes(),
                lambda: t1._typed_storage(),
            ]

        # Check that each of the TypedStorage internal function calls do not
        # produce a deprecation warning
        for f in funcs:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', "TypedStorage is deprecated")
                f()

    # Test that public functions related to TypedStorage produce a deprecation
    # warning
    @skipIfTorchInductor("FIXME")
    def test_typed_storage_deprecation_warning(self):
        # [FUNCTION DOC] Checks public TypedStorage functions emit exactly one DeprecationWarning
        # starting with TypedStorage is deprecated when warnings are forced on, that by default
        # only the first warning appears (and its source line shows the triggering call, read from
        # the file), and that no further warning appears afterward. Skipped under inductor.
        s0 = torch.FloatStorage(10)
        funcs = [
            lambda: torch.FloatStorage(),
            lambda: torch.FloatStorage.dtype,
            lambda: s0.fill_(0),
            lambda: s0.is_cuda,
            lambda: s0.untyped(),
            lambda: len(s0),
            lambda: s0[0],
        ]

        if torch.cuda.is_available():
            s1 = torch.cuda.FloatStorage(10)
            funcs += [
                lambda: torch.cuda.FloatStorage(),
                lambda: torch.cuda.FloatStorage.dtype,
                lambda: s1.fill_(0),
                lambda: s1.is_cuda,
                lambda: s1.untyped(),
                lambda: len(s1),
                lambda: s1[0],
            ]

        # Check that each of the TypedStorage function calls produce a warning
        # if warnings are reset between each
        for f in funcs:
            with AlwaysWarnTypedStorageRemoval(True):
                with warnings.catch_warnings(record=True) as w:
                    warnings.resetwarnings()
                    f()
                    self.assertEqual(len(w), 1, msg=str([str(a) for a in w]))
                    warning = w[0].message
                    self.assertTrue(warning, DeprecationWarning)
                    self.assertTrue(re.search(
                        '^TypedStorage is deprecated',
                        str(warning)))

        # Test that only the first warning is raised by default
        torch.storage._reset_warn_typed_storage_removal()
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            torch.FloatStorage()
            torch.randn(10).storage()
            self.assertEqual(len(w), 1, msg=str([str(a) for a in w]))
            warning = w[0].message
            self.assertTrue(re.search(
                '^TypedStorage is deprecated',
                str(warning)))
            # Check the line of code from the warning's stack
            with open(w[0].filename, encoding="utf-8") as f:
                code_line = f.readlines()[w[0].lineno - 1]
            self.assertTrue(re.search(re.escape('torch.FloatStorage()'), code_line))

        # Check that warnings are not emitted if it happened in the past
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            torch.FloatStorage()
            torch.randn(10).storage()
            self.assertEqual(len(w), 0, msg=str([str(a) for a in w]))

    def test_from_file(self):
        # [FUNCTION DOC] Runs the file-mapping check for FloatStorage.from_file on a temporary
        # file name, and on a temporary path under a non-ASCII directory name when the filesystem
        # encoding is UTF-8.
        def assert_with_filename(filename):
            # [FUNCTION DOC] HELPER: Map a shared 10000-element float storage from the file twice
            # and check both views see identical data and that fills through either tensor are
            # visible through the other; then release the objects.
            size = 10000
            s1 = torch.FloatStorage.from_file(filename, True, size)
            t1 = torch.FloatTensor(s1).copy_(torch.randn(size))
            self.assertEqual(s1.data_ptr(), torch.FloatTensor(s1).data_ptr())

            # check mapping
            s2 = torch.FloatStorage.from_file(filename, True, size)
            t2 = torch.FloatTensor(s2)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t1 from t2
            rnum = random.uniform(-1, 1)
            t1.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t2 from t1
            rnum = random.uniform(-1, 1)
            t2.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # release the tensors
            del s1, t1, s2, t2

        with TemporaryFileName() as fname:
            assert_with_filename(fname)

        if IS_FILESYSTEM_UTF8_ENCODING:
            with TemporaryDirectoryName(suffix='\u4e2d\u6587') as dname, TemporaryFileName(dir=dname) as fname:
                assert_with_filename(fname)

    def test_torch_from_file(self):
        # [FUNCTION DOC] Same as the storage from_file test but using torch.from_file with a float
        # dtype to create the mapped tensors.
        def assert_with_filename(filename):
            # [FUNCTION DOC] HELPER: Map a shared 10000-element float tensor from the file
            # twice and check both views stay in sync after fills through either; then release
            # the objects.
            size = 10000
            s1 = torch.from_file(filename, True, size, dtype=torch.float)
            t1 = torch.FloatTensor(s1).copy_(torch.randn(size))

            # check mapping
            s2 = torch.from_file(filename, True, size, dtype=torch.float)
            t2 = torch.FloatTensor(s2)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t1 from t2
            rnum = random.uniform(-1, 1)
            t1.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t2 from t1
            rnum = random.uniform(-1, 1)
            t2.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # release the tensors
            del s1, t1, s2, t2

        with TemporaryFileName() as fname:
            assert_with_filename(fname)

        if IS_FILESYSTEM_UTF8_ENCODING:
            with TemporaryDirectoryName(suffix='\u4e2d\u6587') as dname, TemporaryFileName(dir=dname) as fname:
                assert_with_filename(fname)

    def test_print(self):
        # [FUNCTION DOC] Printing tests. Smoke-calls repr and str for every non-sparse legacy
        # tensor class, a half tensor and every legacy storage class (skipping unavailable CUDA
        # and unsupported fills), then compares str and repr and checks exact expected strings
        # for complex, complex half, scientific notation (automatic and via set_printoptions,
        # which is reset to the default), big integers, sign padding, inf/nan (real and complex),
        # float64 under different default dtypes, summarization of long tensors and the internal
        # summary helper, device suffixes on CUDA (including a different GPU and a CUDA default
        # tensor type, which is reset), integral floats with requires_grad, non-contiguous
        # slices, zero-dim and bool tensors, and numpy-style scientific switching rules for
        # small/large magnitudes.
        default_type = torch.tensor([]).type()
        for t in torch._tensor_classes:
            if t == torch.HalfTensor:
                continue  # HalfTensor does not support fill
            if t.is_sparse:
                continue
            if t.is_cuda and not torch.cuda.is_available():
                continue
            obj = t(100, 100).fill_(1)
            obj.__repr__()
            str(obj)
        # test half tensor
        obj = torch.rand(100, 100, device='cpu').half()
        obj.__repr__()
        str(obj)
        for t in torch._storage_classes:
            if t == torch.BFloat16Storage:
                continue  # Fix once fill is enabled for bfloat16
            if t.is_cuda and not torch.cuda.is_available():
                continue
            if t == torch.BoolStorage or t == torch.cuda.BoolStorage:
                obj = t(100).fill_(True)
            else:
                obj = t(100).fill_(1)
            obj.__repr__()
            str(obj)

        # test complex tensor
        # complex tensor print uses two formatters, one for real values
        # and the other for imag values. this is consistent with numpy
        x = torch.tensor([2.3 + 4j, 7 + 6j])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([2.3000+4.j, 7.0000+6.j])''')

        # test complex half tensor
        x = torch.tensor([1.25 + 4j, -7. + 6j], dtype=torch.chalf)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([ 1.2500+4.j, -7.0000+6.j], dtype=torch.complex32)''')

        # test scientific notation for complex tensors
        x = torch.tensor([1e28 + 2j , -1e-28j])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+28+2.0000e+00j, -0.0000e+00-1.0000e-28j])''')

        # test big integer
        x = torch.tensor(2341234123412341)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor(2341234123412341)''')

        # test scientific notation
        x = torch.tensor([1e28, 1e-28])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+28, 1.0000e-28])''')

        # test scientific notation using set_printoptions
        x = torch.tensor([1e2, 1e-2])
        torch.set_printoptions(sci_mode=True)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+02, 1.0000e-02])''')
        torch.set_printoptions(sci_mode=False)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([100.0000,   0.0100])''')
        torch.set_printoptions(sci_mode=None)  # reset to the default value

        # test no leading space if all elements positive
        x = torch.tensor([1, 2])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1, 2])''')

        # test for leading space if there are negative elements
        x = torch.tensor([1, -2])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([ 1, -2])''')

        # test inf and nan
        x = torch.tensor([4, inf, 1.5, -inf, 0, nan, 1])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([4.0000,    inf, 1.5000,   -inf, 0.0000,    nan, 1.0000])''')

        y = torch.tensor([4, inf, complex(1.5, inf), complex(-inf, 4), 0, complex(nan, inf), complex(3, nan)])
        self.assertEqual(y.__repr__(), str(y))
        expected_str = '''\
tensor([4.0000+0.j,    inf+0.j, 1.5000+infj,   -inf+4.j, 0.0000+0.j,    nan+infj,
        3.0000+nanj])'''
        self.assertExpectedInline(str(y), expected_str)

        # test dtype
        with set_default_dtype(torch.float):
            x = torch.tensor([1e-324, 1e-323, 1e-322, 1e307, 1e308, 1e309], dtype=torch.float64)
            self.assertEqual(x.__repr__(), str(x))
            expected_str = '''\
tensor([ 0.0000e+00, 9.8813e-324, 9.8813e-323, 1.0000e+307, 1.0000e+308,
                inf], dtype=torch.float64)'''
            self.assertExpectedInline(str(x), expected_str)

        # test changing default dtype
        with set_default_dtype(torch.float64):
            self.assertEqual(x.__repr__(), str(x))
            expected_str = '''\
tensor([ 0.0000e+00, 9.8813e-324, 9.8813e-323, 1.0000e+307, 1.0000e+308,
                inf])'''
            self.assertExpectedInline(str(x), expected_str)

        # test summary
        x = torch.zeros(10000)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([0., 0., 0.,  ..., 0., 0., 0.])''')

        # test internal summary function
        x = torch.rand(1, 20, 5, 30)
        summary = torch._tensor_str.get_summarized_data(x)
        self.assertEqual(summary.shape, (1, 6, 5, 6))
        first_and_last = [0, 1, 2, -3, -2, -1]
        self.assertEqual(summary, x[:, first_and_last][..., first_and_last])

        # test device
        if torch.cuda.is_available():
            x = torch.tensor([123], device='cuda:0')
            self.assertEqual(x.__repr__(), str(x))
            self.assertExpectedInline(str(x), '''tensor([123], device='cuda:0')''')

            # test changing default to cuda
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.assertEqual(x.__repr__(), str(x))
            self.assertExpectedInline(str(x), '''tensor([123])''')

            # test printing a tensor on a different gpu than current one.
            if torch.cuda.device_count() >= 2:
                with torch.cuda.device(1):
                    self.assertEqual(x.__repr__(), str(x))
                    self.assertExpectedInline(str(x), '''tensor([123], device='cuda:0')''')

            # test printing cpu tensor when default device is cuda
            y = torch.tensor([123], device='cpu')
            self.assertEqual(y.__repr__(), str(y))
            self.assertExpectedInline(str(y), '''tensor([123], device='cpu')''')
        torch.set_default_tensor_type(default_type)


        # test integral floats and requires_grad
        x = torch.tensor([123.], requires_grad=True)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([123.], requires_grad=True)''')

        # test non-contiguous print
        # sliced tensor should have > PRINT_OPTS.threshold elements
        x = torch.ones(100, 2, 2, 10)
        y = x.as_strided(size=(100, 2, 10), stride=(2 * 2 * 10, 2 * 10, 1))
        self.assertEqual(str(y), y.__repr__())
        expected_str = '''\
tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        ...,

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]]])\
'''

        self.assertExpectedInline(str(y), expected_str)

        x = torch.ones(100, 2, 2, 10) * (1 + 1j)
        y = x.as_strided(size=(100, 2, 10), stride=(2 * 2 * 10, 2 * 10, 1))
        self.assertEqual(str(y), y.__repr__())
        expected_str = '''\
tensor([[[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        ...,

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]]])\
'''
        self.assertExpectedInline(str(y), expected_str)

        # test print 0-dim tensor: there's no 0-dim in Numpy, we match arrayprint style
        x = torch.tensor(0.00002)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor(2.0000e-05)''')

        # test print boolean tensor
        x = torch.tensor([True])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([True])''')

        x = torch.tensor(True)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor(True)''')

        # [Numpy] test print float in sci_mode when min < 0.0001.
        x = torch.tensor([0.00002])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([2.0000e-05])''')

        # [Numpy] test print complex in sci_mode when real_min < 0.0001 and (or) imag_min < 0.0001.
        x = torch.tensor([0.00002]) * (1 + 1j)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([2.0000e-05+2.0000e-05j])''')

        # [Numpy] test print float in sci_mode when max > 1e8.
        # TODO: Pytorch uses fixed precision to print, while Numpy uses dragon4_scientific
        # to do automatic trimming and padding.
        x = torch.tensor([123456789.])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.2346e+08])''')

        # [Numpy] test print float in sci_mode when max / min > 1000.
        x = torch.tensor([0.01, 11])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e-02, 1.1000e+01])''')

        # [Numpy] test print int max / min > 1000, no sci_mode
        x = torch.tensor([1, 1010])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([   1, 1010])''')

        # [Numpy] test print int > 1e8, no sci_mode
        x = torch.tensor([1000000000])  # 1e9
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1000000000])''')

        # [Numpy] test printing float in int_mode
        x = torch.tensor([1., 1000.])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([   1., 1000.])''')

        # [Numpy] test printing float in int_mode in sci format when max / min > 1000.
        x = torch.tensor([1., 1010.])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+00, 1.0100e+03])''')

    def test_sizeof(self) -> None:
        # [FUNCTION DOC] Checks that the size reported by a storage object grows linearly with the
        # number of elements, for float and uint8 storages, by comparing size differences for 0,
        # 10 and 100 elements.
        sizeof_empty = torch.randn(0).storage().__sizeof__()
        sizeof_10 = torch.randn(10).storage().__sizeof__()
        sizeof_100 = torch.randn(100).storage().__sizeof__()
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

        sizeof_empty = torch.randn(0).to(torch.uint8).storage().__sizeof__()
        sizeof_10 = torch.randn(10).to(torch.uint8).storage().__sizeof__()
        sizeof_100 = torch.randn(100).to(torch.uint8).storage().__sizeof__()
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_resizable(self) -> None:
        # [FUNCTION DOC] Checks that a fresh tensor's storage is resizable and becomes not
        # resizable after the tensor is converted to a numpy array. Skipped under Dynamo.
        x = torch.randn(5)
        self.assertTrue(x.storage().resizable())
        x.numpy()
        self.assertFalse(x.storage().resizable())

    def test_iter(self) -> None:
        # [FUNCTION DOC] Checks iterating a 2D tensor yields its rows and iterating an empty
        # tensor yields nothing.
        x = torch.randn(5, 5)
        for i, sub in enumerate(x):
            self.assertEqual(sub, x[i])  # noqa: PLR1736

        x = torch.tensor([])
        self.assertEqual(list(x), [])

    def test_new(self) -> None:
        # [FUNCTION DOC] Checks the legacy new method: with no args, size args, Size, lists,
        # tuples, numpy values, scalar tensors, size keyword and empty tuple; with a storage or
        # tensor it shares the data pointer (but is a new object); and that an integer tensor of
        # mismatched type raises TypeError while a storage of mismatched type raises RuntimeError.
        x = torch.autograd.Variable(torch.tensor([]))
        y = torch.autograd.Variable(torch.randn(4, 4))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        self.assertEqual(x.new().shape, [0])
        self.assertEqual(x.new(), x)
        self.assertEqual(x.new(1, 2).shape, [1, 2])
        self.assertEqual(x.new(torch.Size([3, 4])).shape, [3, 4])
        self.assertEqual(x.new([3, 4]).shape, [2])
        self.assertEqual(x.new([3, 4]).tolist(), [3, 4])
        self.assertEqual(x.new((3, 4)).tolist(), [3, 4])
        self.assertEqual(x.new([np.int32(3), np.float64(4)]).tolist(), [3, 4])
        self.assertEqual(x.new(np.array((3, 4))).tolist(), [3, 4])
        self.assertEqual(x.new([z[2], z[0] + 3]).tolist(), [3, 4])
        self.assertEqual(x.new(size=(3, 4)).shape, [3, 4])
        self.assertEqual(x.new(()).shape, [0])
        self.assertEqual(x.new(y.storage()).data_ptr(), y.data_ptr())
        self.assertEqual(x.new(y).data_ptr(), y.data_ptr())
        self.assertIsNot(x.new(y), y)

        self.assertRaises(TypeError, lambda: x.new(z))
        # TypeError would be better
        self.assertRaises(RuntimeError, lambda: x.new(z.storage()))

    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    def test_pin_memory(self):
        # [FUNCTION DOC] Checks is_pinned is False for a normal CPU tensor, and with CUDA
        # available that pin_memory returns an equal pinned copy with different data pointer, and
        # pinning again returns the same object. Skipped under CUDA memcheck.
        x = torch.randn(3, 5)
        self.assertFalse(x.is_pinned())
        if torch.cuda.is_available():
            pinned = x.pin_memory()
            self.assertTrue(pinned.is_pinned())
            self.assertEqual(pinned, x)
            self.assertNotEqual(pinned.data_ptr(), x.data_ptr())
            # test that pin_memory on already pinned tensor has no effect
            self.assertIs(pinned, pinned.pin_memory())
            self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())

    def test_error_msg_type_translation(self):
        # [FUNCTION DOC] Checks that a convolution with Long input and Double weights raises
        # RuntimeError whose message names both Double and Long.
        with self.assertRaisesRegex(
                RuntimeError,
                # message includes both Double and Long
                '(?=.*Double)(?=.*Long)'):

            # Calls model with a LongTensor input but DoubleTensor weights
            input = torch.zeros(1, 1, 1, 6, dtype=torch.long)
            weight = torch.nn.Parameter(torch.zeros(1, 1, 1, 3, dtype=torch.double))
            model = torch.nn.Conv2d(1, 1, (1, 3), stride=1, padding=0, bias=False)
            model.weight = weight
            model(input)

    def test_apply(self):
        # [FUNCTION DOC] Checks apply_ maps a Python function over a CPU tensor in place
        # (doubling) and raises TypeError when the function returns a string.
        x = torch.arange(1, 6)
        res = x.clone().apply_(lambda k: k + k)
        self.assertEqual(res, x * 2)
        self.assertRaises(TypeError, lambda: x.apply_(lambda k: "str"))

    def test_map(self):
        # [FUNCTION DOC] Checks map_ combines two tensors elementwise with a Python function (with
        # broadcasting) and raises TypeError mentioning not callable for a string argument.
        x = torch.autograd.Variable(torch.randn(3, 3))
        y = torch.autograd.Variable(torch.randn(3))
        res = x.clone()
        res.map_(y, lambda a, b: a + b)
        self.assertEqual(res, x + y)
        self.assertRaisesRegex(TypeError, "not callable", lambda: res.map_(y, "str"))

    def test_map2(self):
        # [FUNCTION DOC] Checks map2_ combines three tensors with a Python function and raises
        # RuntimeError mentioning requires grad if one argument requires grad.
        x = torch.autograd.Variable(torch.randn(3, 3))
        y = torch.autograd.Variable(torch.randn(3))
        z = torch.autograd.Variable(torch.randn(1, 3))
        res = x.clone()
        res.map2_(y, z, lambda a, b, c: a + b * c)
        self.assertEqual(res, x + y * z)
        z.requires_grad = True
        self.assertRaisesRegex(
            RuntimeError, "requires grad",
            lambda: res.map2_(y, z, lambda a, b, c: a + b * c))

    def test_Size(self):
        # [FUNCTION DOC] Checks torch.Size behaviors: construction from a tensor raises TypeError,
        # it is a tuple with int elements, slicing and addition and multiplication (both operand
        # orders) return torch.Size with the right values including empty cases.
        # expects iterable of int, not Tensor
        self.assertRaises(TypeError, lambda: torch.Size(torch.ones(3)))
        # initialization
        empty_size = torch.Size([])
        size = torch.Size([1, 2, 3])
        self.assertIsInstance(empty_size, tuple)
        self.assertIsInstance(size, tuple)
        # value check __len__
        self.assertEqual(len(empty_size), 0)
        self.assertEqual(len(size), 3)
        # type check __getitem__[int]
        self.assertIsInstance(size[0], int)
        self.assertIsInstance(size[1], int)
        self.assertIsInstance(size[2], int)
        # value check __getitem__[int]
        self.assertEqual(size[0], 1)
        self.assertEqual(size[1], 2)
        self.assertEqual(size[2], 3)
        # type check __getitem__[slice]
        self.assertIsInstance(size[:], torch.Size)
        self.assertIsInstance(size[:-1], torch.Size)
        self.assertIsInstance(size[0:0], torch.Size)
        # value check __getitem__[slice]
        self.assertEqual(size[:], (1, 2, 3))
        self.assertEqual(size[:-1], (1, 2))
        self.assertEqual(size[0:0], ())
        # type check __add__
        self.assertIsInstance(empty_size + (), torch.Size)
        self.assertIsInstance(size + (), torch.Size)
        self.assertIsInstance(size + (4, 5), torch.Size)
        self.assertIsInstance(size + size, torch.Size)
        # value check __add__
        self.assertEqual(empty_size + (), ())
        self.assertEqual(size + (), (1, 2, 3))
        self.assertEqual(size + (4, 5), (1, 2, 3, 4, 5))
        self.assertEqual(size + size, (1, 2, 3, 1, 2, 3))
        # type check __radd__
        self.assertIsInstance(() + empty_size, torch.Size)
        self.assertIsInstance((4, 5) + size, torch.Size)
        # value check __radd__
        self.assertEqual(() + size, (1, 2, 3))
        self.assertEqual((4, 5) + size, (4, 5, 1, 2, 3))
        # type check __mul__
        self.assertIsInstance(empty_size * 0, torch.Size)
        self.assertIsInstance(size * 0, torch.Size)
        self.assertIsInstance(size * 1, torch.Size)
        self.assertIsInstance(size * 2, torch.Size)
        # value check __mul__
        self.assertEqual(empty_size * 0, ())
        self.assertEqual(size * 0, ())
        self.assertEqual(size * 1, (1, 2, 3))
        self.assertEqual(size * 2, (1, 2, 3, 1, 2, 3))
        # type check __rmul__
        self.assertIsInstance(0 * empty_size, torch.Size)
        self.assertIsInstance(0 * size, torch.Size)
        self.assertIsInstance(1 * size, torch.Size)
        self.assertIsInstance(2 * size, torch.Size)
        # value check __rmul__
        self.assertEqual(0 * empty_size, ())
        self.assertEqual(0 * size, ())
        self.assertEqual(1 * size, (1, 2, 3))
        self.assertEqual(2 * size, (1, 2, 3, 1, 2, 3))

    def test_Size_concat_non_tuple_sequence(self):
        # [FUNCTION DOC] Checks that adding a torch.Size and a non-tuple Sequence raises TypeError
        # ("can only concatenate tuple" for Size plus sequence, "unsupported operand type" for the
        # reverse order). Defines a minimal Sequence subclass for this.
        # check that TypeError gets raised on adding non-tuple sequences.
        from collections.abc import Sequence

        class DummySequence(Sequence):
            # [CLASS DOC] Minimal collections.abc Sequence over a class-level list of the integers
            # zero to four, used to check torch.Size concatenation rejects non-tuples.
            vals = list(range(5))
            # [FUNCTION DOC] Return the number of stored values.
            def __len__(self): return len(self.vals)
            # [FUNCTION DOC] Return the stored value at the given index.
            def __getitem__(self, i): return self.vals[i]
            # [FUNCTION DOC] Return an iterator over the stored values.
            def __iter__(self): return iter(self.vals)

        size = torch.Size([1, 2, 3])
        seq = DummySequence()
        msg = r"can only concatenate tuple \(not \w+\) to torch.Size"
        self.assertRaisesRegex(TypeError, msg, lambda: size + seq)
        msg = r"unsupported operand type"
        self.assertRaisesRegex(TypeError, msg, lambda: seq + size)

    def test_Size_concat_wildcard(self):
        # [FUNCTION DOC] Checks that a third-party class that defines add and reverse add can be
        # added to a torch.Size in either order, returning the class result.
        # check that 3rd party classes can support addition with torch.Size
        class Wildcard:
            # [FUNCTION DOC] Return 42 for any right operand.
            # [CLASS DOC] Helper class whose addition and reflected addition both return the
            # integer 42, regardless of operand.
            def __add__(self, other): return 42
            # [FUNCTION DOC] Return 42 for any left operand.
            def __radd__(self, other): return 42

        size = torch.Size([1, 2, 3])
        wildcard = Wildcard()
        self.assertEqual(wildcard + size, 42)
        self.assertEqual(size + wildcard, 42)

    def test_Size_scalar(self):
        # [FUNCTION DOC] Checks a torch.Size can be built from a list including zero-dim integer
        # tensors, which are converted to the integer values.
        three = torch.tensor(3)
        two = torch.tensor(2)
        x = torch.Size([0, 1, two, three, 4])
        for i in range(1, 5):
            self.assertEqual(x[i], i)

    def test_Size_iter(self):
        # [FUNCTION DOC] Checks torch.Size can be constructed from an iterator and from a range
        # and indexed correctly.
        for sizes in [iter([1, 2, 3, 4, 5]), range(1, 6)]:
            x = torch.Size(sizes)
            for i in range(5):
                self.assertEqual(x[i], i + 1)

    def test_t_not_2d_error(self):
        # [FUNCTION DOC] Checks that t and in-place t_ on a three-dimensional tensor
        # raise RuntimeError.
        self.assertRaises(RuntimeError, lambda: torch.randn(2, 3, 4).t())
        self.assertRaises(RuntimeError, lambda: torch.randn(2, 3, 4).t_())

    # skip this test for now as it affects all tests
    @unittest.skipIf(True, "flush_denormal not supported")
    def test_set_flush_denormal(self):
        # [FUNCTION DOC] Skipped unconditionally. Would test set_flush_denormal turning tiny float
        # values to zero for floats but not doubles and then restore the setting.
        tiny_float = 1e-42
        tiny_double = 1e-320
        float_tensor = torch.FloatTensor([1.0, tiny_float])
        double_tensor = torch.DoubleTensor([1.0, tiny_float, tiny_double])

        self.assertEqual(float_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(float_tensor[1], tiny_float, atol=tiny_float / 16, rtol=0)
        self.assertEqual(double_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[1], tiny_float, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[2], tiny_double, atol=0.0, rtol=0)

        torch.set_flush_denormal(True)
        self.assertEqual(float_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(float_tensor[1], 0.0, atol=0.0, rtol=0)  # tiny_float to zero
        self.assertEqual(double_tensor[0], 1.0, atol=0.0, rtol=0)
        # tiny_float is not converted to zero in double type
        self.assertEqual(double_tensor[1], tiny_float, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[2], 0.0, atol=0.0, rtol=0)  # tiny_double to zero
        torch.set_flush_denormal(False)

    def test_show_config(self):
        # [FUNCTION DOC] Smoke test that the build configuration show function does not crash.
        # We can't usefully test the output; just make sure this doesn't crash
        torch.__config__.show()

    @unittest.skipIf(IS_FBCODE, "CXX_FLAGS is only for OSS build.")
    @skipIfRocmVersionAtLeast([7, 14])
    def test_cxx_flags(self):
        # [FUNCTION DOC] Smoke test that querying the C++ compile flags does not crash (skipped on
        # fbcode and recent ROCm versions).
        torch.__config__._cxx_flags()

    def test_parallel_info(self):
        # [FUNCTION DOC] Smoke test that the parallel info function does not crash.
        torch.__config__.parallel_info()

    def test_get_cpu_capability(self):
        # [FUNCTION DOC] Calls the CPU capability getter and also scripts it with torch.jit.script
        # to ensure it stays scriptable (used by torchvision resize).
        # This method is primarily exposed for torchvision's resize
        torch.backends.cpu.get_cpu_capability()

        # We have to ensure that method is torchscriptable as torchvision's resize
        # should be torchscriptable
        torch.jit.script(torch.backends.cpu.get_cpu_capability)

    @slowTest
    def test_slow_test(self):
        # [FUNCTION DOC] Smoke test that the slowTest decorator works; the body does nothing.
        # Just a smoketest to make sure our slowTest decorator works.
        pass

    def test_is_nonzero(self):
        # [FUNCTION DOC] Checks is_nonzero: errors (RuntimeError, ambiguous) for empty or
        # multi-element tensors and the right boolean for single-element int, float, bool and
        # complex tensors.
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch.tensor([]).is_nonzero()
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch.tensor([0, 0]).is_nonzero()
        self.assertFalse(torch.tensor(0).is_nonzero())
        self.assertTrue(torch.tensor(1).is_nonzero())
        self.assertFalse(torch.tensor([0]).is_nonzero())
        self.assertTrue(torch.tensor([1]).is_nonzero())
        self.assertFalse(torch.tensor([[0]]).is_nonzero())
        self.assertTrue(torch.tensor([[1]]).is_nonzero())
        self.assertTrue(torch.tensor(0.1).is_nonzero())
        self.assertTrue(torch.tensor(-0.1).is_nonzero())
        self.assertFalse(torch.tensor(0.0).is_nonzero())
        self.assertTrue(torch.tensor(True).is_nonzero())
        self.assertFalse(torch.tensor(False).is_nonzero())
        self.assertFalse(torch.tensor(0 + 0j).is_nonzero())
        self.assertTrue(torch.tensor(0 + 0.1j).is_nonzero())

    def test_assert_async(self):
        # [FUNCTION DOC] Checks torch._assert_async: errors for empty and multi-element tensors,
        # raises RuntimeError for zero-valued single elements of any dtype, and passes for
        # nonzero values.
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch._assert_async(torch.tensor([]))
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch._assert_async(torch.tensor([0, 0]))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0))
        torch._assert_async(torch.tensor(1))
        torch._assert_async(torch.tensor(0.1))
        torch._assert_async(torch.tensor(-0.1))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0.0))
        torch._assert_async(torch.tensor(True))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(False))
        torch._assert_async(torch.tensor(0 + 0.1j))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0 + 0j))

    # NB: we must not be built with CUDA; if we are built with CUDA but no CUDA
    # is available, we get a different error.
    @unittest.skipIf(torch.backends.cuda.is_built() or IS_SANDCASTLE, "CUDA is built, can't test CUDA not built error")
    def test_cuda_not_built(self):
        # [FUNCTION DOC] Only on builds without CUDA: checks that CUDA-related calls
        # (current_device, creating tensors or moving to CUDA, CUDA tensor types, setting default
        # type) raise the "not compiled with CUDA" error with the expected exception type
        # (AssertionError or TypeError).
        msg = "Torch not compiled with CUDA enabled"
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.cuda.current_device())
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1], device="cuda"))
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1]).cuda())
        self.assertRaisesRegex(TypeError, msg, lambda: torch.cuda.FloatTensor())
        self.assertRaisesRegex(TypeError, msg, lambda: torch.set_default_tensor_type(torch.cuda.FloatTensor))
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1]).to(device="cuda"))

    def test_has_internal_overlap(self):
        # [FUNCTION DOC] Checks the internal overlap debug function: contiguous gives no overlap,
        # an expanded tensor gives overlap, a zero-stride size-1 axis gives no overlap for a
        # contiguous layout, and a sliced zero-stride layout gives too hard to tell.
        OVERLAP_NO = 0
        OVERLAP_YES = 1
        OVERLAP_TOO_HARD = 2

        # Check for contiguous tensors
        a = torch.randn(3, 3)
        self.assertEqual(torch._debug_has_internal_overlap(a), OVERLAP_NO)

        # Checks for zero strides
        b = torch.randn(1, 3)
        b_expanded = b.expand(4, 3)
        self.assertEqual(torch._debug_has_internal_overlap(b_expanded), OVERLAP_YES)

        # Check for zero strided, size 1 axis, in non-contiguous storage (gh-33812)
        c = torch.randn(10).as_strided([2, 1, 5], [1, 0, 2])
        self.assertEqual(torch._debug_has_internal_overlap(c), OVERLAP_NO)
        c = torch.randn(2, 1, 10)[::2].as_strided((2, 1, 5), (10, 0, 2))
        self.assertEqual(torch._debug_has_internal_overlap(c), OVERLAP_TOO_HARD)

    def test_allow_tensor_metadata_change(self):
        # [FUNCTION DOC] Placeholder test: creates a tensor and does not check anything (the
        # comment describes allowed metadata changes on detached views).
        torch.ones(2, 3)
        # Metadata changes are allowed on view tensors that are created from detach().

    def test_memory_format(self):
        # [FUNCTION DOC] Checks contiguous with channels-last and channels-last-3d memory formats
        # on 4D and 5D tensors gives a tensor not default-contiguous but contiguous in the
        # requested format with equal values.
        def test_helper(x, memory_format):
            # [FUNCTION DOC] HELPER: Convert x to the given memory format via contiguous and check
            # contiguity flags and equal values.
            y = x.contiguous(memory_format=memory_format)
            self.assertFalse(y.is_contiguous())
            self.assertTrue(y.is_contiguous(memory_format=memory_format))
            self.assertEqual(y, x)

        test_helper(torch.randn(4, 3, 8, 8), torch.channels_last)
        test_helper(torch.randn(4, 3, 8, 8, 8), torch.channels_last_3d)

    def test_memory_format_contiguous_returns_same_tensor_if_already_satisfies(self):
        # [FUNCTION DOC] Checks contiguous with a memory format returns the same tensor (an alias)
        # when the input is already in that format: filling the alias also changes the input.
        def test_helper(x, memory_format):
            # [FUNCTION DOC] HELPER: Call contiguous in the given format on x, fill the result
            # with 7, and check x was changed too.
            alias = x.contiguous(memory_format=memory_format)
            alias.fill_(7)
            self.assertEqual(x, alias)

        test_helper(torch.randn(4, 8, 8, 3).permute(0, 3, 1, 2), torch.channels_last)
        test_helper(torch.randn(4, 8, 8, 8, 3).permute(0, 4, 1, 2, 3), torch.channels_last_3d)

    def test_memory_format_empty(self):
        # [FUNCTION DOC] Checks empty with a channels-last format raises RuntimeError for a wrong
        # number of dimensions and succeeds with the right number, giving a tensor contiguous in
        # that format (4D and 5D).
        def test_helper(dim1, dim2, memory_format):
            # [FUNCTION DOC] HELPER: Check empty raises RuntimeError for dim1 and works for dim2
            # with the memory format, producing a tensor contiguous in that format.
            with self.assertRaises(RuntimeError):
                x = torch.empty(dim1, memory_format=memory_format)
            x = torch.empty(dim2, memory_format=memory_format)
            self.assertTrue(x.is_contiguous(memory_format=memory_format))

        test_helper((3, 3), (3, 3, 3, 3), torch.channels_last)
        test_helper((3, 3, 3), (3, 3, 3, 3, 3), torch.channels_last_3d)

    @skipIfCrossRef
    def test_dim_order(self):
        # [FUNCTION DOC] Tests Tensor.dim_order on default, transposed, channels-last,
        # channels-last-3d and permuted-empty tensors; the ambiguity_check option raising
        # RuntimeError for ambiguous shapes unless an acceptable format list resolves them;
        # TypeError for an invalid ambiguity_check value; and AttributeError for sparse tensors.
        # Skipped under crossref.
        shape = (2, 3, 5, 7)

        t = torch.empty(shape)
        self.assertSequenceEqual(t.dim_order(), (0, 1, 2, 3), seq_type=tuple)
        self.assertSequenceEqual(t.dim_order(ambiguity_check=True), (0, 1, 2, 3), seq_type=tuple)
        # transpose doesn't really change the underlying physical memory
        # so expecting dim_order change to reflect that (like strides)
        self.assertSequenceEqual(t.transpose(0, 1).dim_order(), (1, 0, 2, 3))

        t = torch.empty(shape, memory_format=torch.channels_last)
        self.assertSequenceEqual(t.dim_order(), (0, 2, 3, 1))

        t = torch.empty((2, 3, 5, 7, 8), memory_format=torch.channels_last_3d)
        self.assertSequenceEqual(t.dim_order(), (0, 2, 3, 4, 1))

        for dim_order in itertools.permutations(range(4)):
            self.assertSequenceEqual(
                dim_order, torch.empty_permuted(shape, dim_order).dim_order()
            )

        target_shapes = [[2, 2, 1, 2], [1, 2, 2, 2], [2, 2, 2, 1], [1, 2, 2, 1], [1, 2, 1, 2]]

        for shape in target_shapes:
            for memory_format in (torch.contiguous_format, torch.channels_last):
                t = torch.empty(shape).to(memory_format=memory_format)
                with self.assertRaises(RuntimeError):
                    t.dim_order(ambiguity_check=True)

                if memory_format == torch.contiguous_format:
                    dim_order_target = list(range(len(shape)))
                elif memory_format == torch.channels_last:
                    dim_order_target = [0, *list(range(2, len(shape))), 1]

                self.assertSequenceEqual(
                    dim_order_target, t.dim_order(ambiguity_check=[torch.contiguous_format, torch.channels_last])
                )


        ambiguous_shapes = [[2, 1, 2, 2], [2, 2, 1, 1], [1, 2, 1, 1], [2, 1, 1, 2], [2, 1, 2, 1],
                            [1, 1, 1, 2], [1, 1, 2, 2], [1, 1, 1, 1], [2, 1, 1, 1], [1, 1, 2, 1]]

        for shape in ambiguous_shapes:
            for memory_format in (torch.contiguous_format, torch.channels_last):
                t = torch.empty(shape).to(memory_format=memory_format)
                with self.assertRaises(RuntimeError):
                    t.dim_order(ambiguity_check=True)
                    t.dim_order(ambiguity_check=[torch.contiguous_format, torch.channels_last])

        with self.assertRaises(TypeError):
            torch.empty((1, 2, 3, 4)).dim_order(ambiguity_check="ILLEGAL_STR")

        # sparse tensor does not support dim order
        with self.assertRaises(AttributeError):
            indices = torch.tensor([[0, 1, 2], [0, 1, 2]])  # (row, column) indices
            values = torch.tensor([1.0, 2.0, 3.0])  # values at those indices
            sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(3, 3))
            sparse_tensor.dim_order()

    def test_subclass_tensors(self):
        # [FUNCTION DOC] Checks that subclassing the legacy FloatTensor type raises TypeError (not
        # an acceptable base type) while subclassing Tensor works and the subclass instance can
        # call its own method.
        # raise an error when trying to subclass FloatTensor
        with self.assertRaisesRegex(TypeError, "type 'torch.FloatTensor' is not an acceptable base type"):
            class Foo1(torch.FloatTensor):
                # [CLASS DOC] Attempted subclass of the legacy FloatTensor type, defined only to
                # check that class creation raises TypeError; it has no body of its own.
                pass

        # but allow subclassing Tensor:
        class Foo2(torch.Tensor):
            # [CLASS DOC] Tensor subclass adding one method, used to check that Tensor can
            # be subclassed.
            def foo(self):
                # [FUNCTION DOC] Return the integer 5.
                return 5
        f = Foo2()
        self.assertEqual(f.foo(), 5)

    def test_ndim(self):
        # [FUNCTION DOC] Checks ndim for 3D, zero-dim and empty (1 by 0) tensors.
        a = torch.randn(1, 2, 3)
        self.assertEqual(3, a.ndim)
        b = torch.randn(())
        self.assertEqual(0, b.ndim)
        c = torch.randn(1, 0)
        self.assertEqual(2, c.ndim)

    def test_nbytes(self):
        # [FUNCTION DOC] Checks nbytes equals numel times element size for several tensors
        # including zero-dim and empty.
        a = torch.randn(1, 2, 3, dtype=torch.float64)
        self.assertEqual(a.numel() * a.element_size(), a.nbytes)
        b = torch.randn(())
        self.assertEqual(b.numel() * b.element_size(), b.nbytes)
        c = torch.randn(1, 0)
        self.assertEqual(c.numel() * c.element_size(), c.nbytes)

    def test_fill_diagonal(self):
        # [FUNCTION DOC] Checks fill_diagonal_ on 2D (tall) tensors with and without wrap, on a 3D
        # cube, and on non-contiguous slices, by comparing to loop-assigned references.
        a1 = torch.randn(7, 3)
        a2 = a1.clone()
        v = 1
        for i in range(3):
            a2[i][i] = v
        a1.fill_diagonal_(v)
        self.assertEqual(a1, a2)

        b1 = torch.randn(7, 3)
        b2 = b1.clone()
        for i in range(3):
            b2[i][i] = v
            b2[i + 4][i] = v
        b1.fill_diagonal_(v, wrap=True)
        self.assertEqual(b1, b2)

        c1 = torch.rand(3, 3, 3)
        c2 = c1.clone()
        for i in range(3):
            c2[i][i][i] = v
        c1.fill_diagonal_(v)
        self.assertEqual(c1, c2)

        # non-contiguous tensor
        d1 = torch.rand(3, 3, 3)[:, 1, ...]
        d2 = d1.clone()
        for i in range(3):
            d2[i][i] = v
        d1.fill_diagonal_(v)
        self.assertEqual(d1, d2)

        e1 = torch.rand(7, 3, 3)[:, 1, ...]
        e2 = e1.clone()
        for i in range(3):
            e2[i][i] = v
            e2[i + 4][i] = v
        e1.fill_diagonal_(v, wrap=True)
        self.assertEqual(e1, e2)

    def test_setting_real_imag_to_a_number(self):
        # [FUNCTION DOC] Checks assigning plain numbers to the real and imaginary attributes of a
        # complex tensor sets both parts to zero.
        x = torch.randn(4, dtype=torch.cfloat)
        x.real = 0
        x.imag = 0
        zeros = torch.zeros(4)
        self.assertEqual(x.real, zeros)
        self.assertEqual(x.imag, zeros)

    def test_batch_norm_cpu_inference(self):
        # [FUNCTION DOC] Checks BatchNorm2d in eval mode on CPU for two small inputs, with and
        # without affine parameters: contiguous, permuted (non-contiguous) and channels-last
        # inputs give the same result as the stored expected outputs.
        # input nchw in (2,1,1,1), (2,2,2,2)
        inputs = [
            torch.tensor([[[[-0.5000]]], [[[0.5000]]]]),
            torch.tensor([
                [
                    [[-0.5000, 0.5000], [-1.0000, 1.0000]],
                    [[-0.2500, -0.5000], [0.2500, 0.5000]]
                ],
                [
                    [[0.1000, 1.0000], [1.0000, 0.1000]],
                    [[1.0000, 0.5000], [1.5000, -1.5000]]
                ]])]
        # output nchw in (2,1,1,1), (2,2,2,2)
        outputs = [
            torch.tensor([
                [[[-0.499997496604919433593750000]]],
                [[[0.499997496604919433593750000]]]]),
            torch.tensor([
                [[[-0.499997496604919433593750000, 0.499997496604919433593750000],
                  [-0.999994993209838867187500000, 0.999994993209838867187500000]],
                 [[-0.249998748302459716796875000, -0.499997496604919433593750000],
                  [0.249998748302459716796875000, 0.499997496604919433593750000]]],
                [[[0.099999502301216125488281250, 0.999994993209838867187500000],
                  [0.999994993209838867187500000, 0.099999502301216125488281250]],
                 [[0.999994993209838867187500000, 0.499997496604919433593750000],
                  [1.499992489814758300781250000, -1.499992489814758300781250000]]]])]


        for i in range(len(inputs)):
            for affine in [False, True]:
                m = torch.nn.BatchNorm2d(inputs[i].size()[1], 1e-05, 0.1, affine=affine)
                m.eval()
                # contiguous case
                input1 = inputs[i].contiguous()
                output1 = m(input1)
                # non-contiguous case
                input2 = input1.permute(0, 1, 3, 2)
                output2 = m(input2).permute(0, 1, 3, 2)
                # channels last case
                input3 = input1.contiguous(memory_format=torch.channels_last)
                output3 = m(input3)
                self.assertEqual(output3, outputs[i])
                self.assertEqual(output3, output1)
                self.assertEqual(output3, output2)

    # FIXME: move these meta tests to their own test suite/class or
    #   distribute them among the appropriate test suites for their ops
    @skipIfTorchDynamo("Fails after Triton update, see https://github.com/pytorch/pytorch/issues/94687")
    def test_empty_meta(self):
        # [FUNCTION DOC] Checks a huge meta-device tensor can be created and added with
        # broadcasting, giving the right size, and reading an element raises RuntimeError. Skipped
        # under Dynamo.
        x = torch.empty(2 ** 20, 2 ** 20, device='meta')
        y = torch.empty(2 ** 20, device='meta')
        z = x + y
        self.assertEqual(z.size(), (2 ** 20, 2 ** 20))
        self.assertRaises(RuntimeError, lambda: z[0][0].item())

    @skipIfTorchDynamo("Fails after Triton update, see https://github.com/pytorch/pytorch/issues/94687")
    def test_format_scalar_meta(self):
        # [FUNCTION DOC] Checks formatting a zero-dim meta tensor with format gives the same
        # string as repr. Skipped under Dynamo.
        x = torch.empty((), device='meta')
        self.assertEqual(format(x), repr(x))

    def test_upsample_nearest1d_meta(self):
        # [FUNCTION DOC] Checks nearest 1D upsampling on a very large meta tensor produces the
        # doubled size (and reading raises RuntimeError), including the out= variant which resizes
        # an empty meta output.
        # TODO: this test should be triggered by test_nn.py but right
        # now meta is not enabled (and even if it was, we are probably
        # missing too many meta functions to get through the test unmolested)

        # NB: Can't make the exponent too big, or it will overflow
        # signed 64-bit integer
        x = torch.empty(2 * 10 ** 8, 3, 2 * 10 ** 8, device='meta')
        z = torch.nn.functional.interpolate(x, scale_factor=2)
        self.assertEqual(z.size(), (2 * 10 ** 8, 3, 4 * 10 ** 8))
        self.assertRaises(RuntimeError, lambda: z[0][0][0].item())

        # TODO: the out tests cannot be triggered by test_nn.py because
        # we don't actually do out= arguments for nn functions, so there
        # is no public API by which to get the out version

        # interpolate doesn't seem to support out=
        # (not sure why passing None here doesn't work? How strange...)
        z = torch.empty(0, device='meta')
        torch._C._nn.upsample_nearest1d(x, (4 * 10 ** 8,), 2, out=z)
        self.assertEqual(z.size(), (2 * 10 ** 8, 3, 4 * 10 ** 8))
        self.assertRaises(RuntimeError, lambda: z[0][0][0].item())

    def test_upsample_nearest2d_meta(self):
        # [FUNCTION DOC] Checks upsample_nearest2d out= semantics on meta tensors: output memory
        # format is preserved when no resize is needed, replaced by the input's format when the
        # output is resized from empty, and dtype or device mismatch raises RuntimeError with
        # exact messages (device mismatch not checked under inductor).
        # TODO: the out tests cannot be triggered by test_nn.py because
        # we don't actually do out= arguments for nn functions, so there
        # is no public API by which to get the out version

        # Make sure we don't clobber strides of out tensor.  NB: this
        # test must be done on 2d/3d, because 1d doesn't have any meaningful
        # layout support
        x = torch.empty(4, 3, 8, 8, device='meta')
        out = torch.empty(4, 3, 16, 16, device='meta', memory_format=torch.channels_last)
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        x = torch.empty(4, 3, 8, 8, device='meta', memory_format=torch.channels_last)
        out = torch.empty(4, 3, 16, 16, device='meta')
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous())

        # But if resize occurs, do clobber
        x = torch.empty(4, 3, 8, 8, device='meta', memory_format=torch.channels_last)
        out = torch.empty(0, device='meta')
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        # Complain if out dtype mismatch
        x = torch.empty(4, 3, 8, 8, device='meta', dtype=torch.float)
        out = torch.empty(4, 3, 16, 16, device='meta', dtype=torch.double)
        self.assertExpectedRaisesInline(
            RuntimeError, lambda: torch._C._nn.upsample_nearest2d(x, (16, 16), out=out),
            """Expected out tensor to have dtype torch.float32 but got torch.float64 instead"""
        )

        # Complain if out device mismatch
        x = torch.empty(0, 3, 8, 8, device='meta')
        out = torch.empty(0, 3, 16, 16, device='cpu')
        # FIXME: compiling should properly error with a device mismatch.
        if not TEST_WITH_TORCHINDUCTOR:
            self.assertExpectedRaisesInline(
                RuntimeError, lambda: torch._C._nn.upsample_nearest2d(x, (16, 16), out=out),
                """Attempting to copy from device meta to device cpu, but cross-device copies are not allowed!"""
            )

    def test_add_meta_scalar(self):
        # [FUNCTION DOC] Checks adding a Python scalar to a meta tensor keeps the size
        # (issue 53815).
        # From https://github.com/pytorch/pytorch/issues/53815
        x = torch.empty(2, device='meta')
        y = x + 2
        self.assertEqual(y.size(), x.size())

    def test_normal_shape(self):
        # [FUNCTION DOC] For each device type, checks torch.normal with tensor mean and std for
        # equal, expandable and non-expandable shapes (result sizes or RuntimeError messages about
        # mismatched sizes), and the out= argument either compatible or incompatible (resize
        # deprecation warning or size mismatch error).
        for device in get_all_device_types():
            tensor1 = torch.rand(1, device=device)
            tensor4 = torch.rand(4, device=device)
            tensor120 = torch.rand(120, device=device)
            tensor2145 = torch.rand(2, 1, 4, 5, device=device)
            tensor2345 = torch.rand(2, 3, 4, 5, device=device)
            tensor2345_non_contiguous = torch.rand(2, 4, 3, 5, device=device).permute(0, 2, 1, 3)
            tensor2345_channels_last = tensor2345.contiguous(memory_format=torch.channels_last)
            output2345 = torch.zeros(2, 3, 4, 5, device=device)
            output345 = torch.zeros(3, 4, 5, device=device)

            # inputs have same size
            self.assertEqual(torch.normal(tensor2345, tensor2345).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2345_non_contiguous, tensor2345).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2345, tensor2345_channels_last).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2345_non_contiguous, tensor2345_channels_last).size(), (2, 3, 4, 5))

            # scalar case
            self.assertEqual(torch.normal(tensor2345, 2).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(2, tensor2345).size(), (2, 3, 4, 5))

            # inputs are expandable tensors
            self.assertEqual(torch.normal(tensor2345, tensor1).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2145, tensor2345).size(), (2, 3, 4, 5))

            # inputs are non-expandable tensors, but they have same number of elements
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(120\) must match the size of "
                    r"tensor b \(5\) at non-singleton dimension 3"):
                self.assertEqual(torch.normal(tensor120, tensor2345).size(), (120,))
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(5\) must match the size of "
                    r"tensor b \(120\) at non-singleton dimension 3"):
                self.assertEqual(torch.normal(tensor2345, tensor120).size(), (2, 3, 4, 5))

            # inputs are non-expandable tensors and they don't have same number of elements
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(5\) must match the size of "
                    r"tensor b \(4\) at non-singleton dimension 3"):
                torch.normal(tensor2345, tensor4)

            # output and inputs are size compatible
            self.assertEqual(torch.normal(tensor2345, tensor2345, out=output2345).size(), (2, 3, 4, 5))

            # output and inputs are not size compatible
            with self.assertWarnsRegex(
                    UserWarning,
                    "This behavior is deprecated, and in a future PyTorch "
                    "release outputs will not be resized unless they have "
                    "zero elements"):
                self.assertEqual(torch.normal(tensor2345, tensor2145, out=output345).size(), (2, 3, 4, 5))
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(5\) must match the size of "
                    r"tensor b \(120\) at non-singleton dimension 3"):
                # inputs are not expandable, output size is not the same as mean
                torch.normal(tensor2345, tensor120, out=output345)

    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/157246")
    def test_tensoriterator_output_setup(self):
        # [FUNCTION DOC] Checks TensorIterator output setup for contiguous, channels-last,
        # permuted (same strides) and mixed-stride inputs, for float and quantized add, and that a
        # pre-strided out argument keeps its strides. Skipped on macOS.
        # Test whether the output's memory layout is correct
        def test_memory_layout(x, y, scale, zero_point, out):
            # [FUNCTION DOC] HELPER: Check that sizes of x, y and out match and that every element
            # of out equals the sum of the corresponding elements of x and y (using quantized add
            # with scale and zero point when they are given).
            self.assertEqual(x.dim(), 4)
            self.assertEqual(x.size(), y.size())
            self.assertEqual(y.size(), out.size())

            shape = x.size()
            for n in range(shape[0]):
                for c in range(shape[1]):
                    for h in range(shape[2]):
                        for w in range(shape[3]):
                            if scale is not None and zero_point is not None:
                                self.assertEqual(
                                    out[n][c][h][w],
                                    torch.ops.quantized.add(x[n][c][h][w], y[n][c][h][w], scale, zero_point))
                            else:
                                self.assertEqual(out[n][c][h][w], x[n][c][h][w] + y[n][c][h][w])

        xraw = torch.rand(2, 3, 4, 4)
        yraw = torch.rand(2, 3, 4, 4)
        qxraw = torch.quantize_per_tensor(xraw, 0.1, 5, torch.quint8)
        qyraw = torch.quantize_per_tensor(yraw, 0.1, 5, torch.quint8)

        # contiguous case fast setup
        test_memory_layout(xraw, yraw, None, None, xraw + yraw)
        test_memory_layout(qxraw, qyraw, 0.1, 5, torch.ops.quantized.add(qxraw, qyraw, 0.1, 5))

        # channels last case fast setup
        x = xraw.contiguous(memory_format=torch.channels_last)
        y = yraw.contiguous(memory_format=torch.channels_last)
        test_memory_layout(x, y, None, None, x + y)
        qx = qxraw.contiguous(memory_format=torch.channels_last)
        qy = qyraw.contiguous(memory_format=torch.channels_last)
        test_memory_layout(qx, qy, 0.1, 5, torch.ops.quantized.add(qx, qy, 0.1, 5))

        # non contiguous case fast setup (dense, non-overlapping, same shape and strides)
        x = xraw.permute(0, 2, 3, 1)
        y = yraw.permute(0, 2, 3, 1)
        test_memory_layout(x, y, None, None, x + y)
        qx = qxraw.permute(0, 2, 3, 1)
        qy = qyraw.permute(0, 2, 3, 1)
        test_memory_layout(qx, qy, 0.1, 5, torch.ops.quantized.add(qx, qy, 0.1, 5))

        # non contiguous case fast setup (dense, non-overlapping)
        # input tensors have same shape and strides
        # output tensor have same shape as input tensors but different stride
        # output tensor should preserve its strides in this case
        x = xraw.permute(0, 2, 3, 1)
        y = yraw.permute(0, 2, 3, 1)
        out = torch.empty_like(xraw)
        out = out.permute(0, 3, 2, 1)
        expected_stride = out.stride()
        test_memory_layout(x, y, None, None, torch.add(x, y, out=out))
        self.assertEqual(expected_stride, out.stride())

        # non contiguous case non fast setup
        x = xraw.permute(0, 2, 3, 1)
        y = yraw.permute(0, 3, 2, 1)
        test_memory_layout(x, y, None, None, x + y)
        qx = qxraw.permute(0, 2, 3, 1)
        qy = qyraw.permute(0, 3, 2, 1)
        test_memory_layout(qx, qy, 0.1, 5, torch.ops.quantized.add(qx, qy, 0.1, 5))

    def test_conj_physical_meta_stride(self):
        # [FUNCTION DOC] Checks conj_physical of a complex meta tensor produced by an FFT keeps
        # the same stride as its input.
        a = torch.zeros((5, 3, 6), dtype=torch.complex128, device='meta')
        b = torch._fft_c2c(a, [1], 1, True)
        c = torch.conj_physical(b)
        self.assertEqual(b.stride(), c.stride())

    # Tests to make sure we still handle .data properly until it is removed
    def test_dot_data_use(self):
        # [FUNCTION DOC] Checks that replacing a module weight's data with a tensor of a different
        # dtype (complex) and calling it raises RuntimeError naming both Double and ComplexFloat.
        # .data allows to change the Tensors types inplace, check that we still
        # raise a nice error.
        with self.assertRaisesRegex(
                RuntimeError,
                # message includes both Double and ComplexFloat
                '(?=.*Double)(?=.*ComplexFloat)'):

            # Calls model with a LongTensor input but DoubleTensor weights
            input = torch.randn(1, 1, 1, 6, dtype=torch.double)
            weight = torch.zeros(1, 1, 1, 3, dtype=torch.complex64)
            model = torch.nn.Conv2d(1, 1, (1, 3), stride=1, padding=0, bias=False)
            model.weight.data = weight
            model(input)

    def test_empty_storage_view(self):
        # [FUNCTION DOC] Checks in-place multiplying a slice of a zero-element array tensor
        # created from numpy does not raise.
        # we should be able to "modify" slices of a 0-element
        # array without an error being raised due to
        # trying to resize its storage
        t = torch.from_numpy(np.empty((0, 4)))
        t[:, 1::2] *= 1

    def test_has_storage(self):
        # [FUNCTION DOC] Checks that several kinds of empty and derived tensors (empty list,
        # empty, clone, nonzero result, new) all report a non-None storage.
        self.assertIsNotNone(torch.tensor([]).storage())
        self.assertIsNotNone(torch.empty(0).storage())
        self.assertIsNotNone(torch.tensor([]).clone().storage())
        self.assertIsNotNone(torch.tensor([0, 0, 0]).nonzero().storage())
        self.assertIsNotNone(torch.tensor([]).new().storage())

    # FIXME: Extend this test and put in a TensorProperties test class
    def test_numel(self):
        # [FUNCTION DOC] Checks nelement and numel agree for a three-dimensional byte tensor.
        b = torch.ByteTensor(3, 100, 100)
        self.assertEqual(b.nelement(), 3 * 100 * 100)
        self.assertEqual(b.numel(), 3 * 100 * 100)

    # Verifies that (deep)copies of dtypes are the same objects
    def test_copy_dtypes(self):
        # [FUNCTION DOC] Checks that deep-copying each dtype returns the same dtype object.
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
            copied_dtype = copy.deepcopy(dtype)
            self.assertIs(dtype, copied_dtype)

    def test_dtype_is_signed(self):
        # [FUNCTION DOC] Checks every non-quantized dtype's is_signed attribute matches
        # torch.is_signed on a zero-dim tensor of that dtype, and that accessing is_signed on
        # quint8, qint8 and qint32 raises RuntimeError saying it is not supported for
        # quantized types.
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.half):
            self.assertEqual(dtype.is_signed, torch.is_signed(torch.tensor(0, dtype=dtype)))

        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.quint8.is_signed)
        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.qint8.is_signed)
        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.qint32.is_signed)

    def test_dtype_abbr(self):
        # [FUNCTION DOC] Checks the abbreviation string of each dtype (for example f32, bf16, c64,
        # u8, bool as b8, bit-packed types) against an expected table, and that the table covers
        # exactly the set returned by the internal all-dtypes query.
        expected = {
            torch.float64: "f64",
            torch.float32: "f32",
            torch.float16: "f16",
            torch.bfloat16: "bf16",
            torch.float8_e4m3fn: "f8e4m3fn",
            torch.float8_e5m2: "f8e5m2",
            torch.float8_e4m3fnuz: "f8e4m3fnuz",
            torch.float8_e5m2fnuz: "f8e5m2fnuz",
            torch.float8_e8m0fnu: "f8e8m0fnu",
            torch.float4_e2m1fn_x2: "f4e2m1fnx2",
            torch.complex32: "c32",
            torch.bcomplex32: "bc32",
            torch.complex64: "c64",
            torch.complex128: "c128",
            torch.int8: "i8",
            torch.int16: "i16",
            torch.int32: "i32",
            torch.int64: "i64",
            torch.bool: "b8",
            torch.uint8: "u8",
            torch.uint16: "u16",
            torch.uint32: "u32",
            torch.uint64: "u64",
            torch.bits16: "b16x1",
            torch.bits1x8: "b1x8",
            torch.bits2x4: "b2x4",
            torch.bits4x2: "b4x2",
            torch.bits8: "b8x1",
        }
        for dtype, abbr in expected.items():
            self.assertEqual(dtype.abbr, abbr)
        self.assertEqual(set(torch._C._get_all_dtypes()), set(expected.keys()))

    def test_get_all_dtypes(self):
        # [FUNCTION DOC] Pins the set of dtypes returned by the internal all-dtypes
        # function against a hand-written expected set (excluding quantized and sub-byte
        # placeholder types).
        # The C++ implementation iterates AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS
        # and filters out quantized and sub-byte placeholder dtypes; pin the
        # result against a manually computed list so regressions are caught.
        expected = {
            torch.uint8, torch.uint16, torch.uint32, torch.uint64,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.float32, torch.float64, torch.bfloat16,
            torch.complex32, torch.bcomplex32, torch.complex64, torch.complex128,
            torch.bool,
            torch.float8_e5m2, torch.float8_e4m3fn,
            torch.float8_e5m2fnuz, torch.float8_e4m3fnuz,
            torch.float8_e8m0fnu, torch.float4_e2m1fn_x2,
            torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16,
        }
        self.assertEqual(set(torch._C._get_all_dtypes()), expected)

    # FIXME: Put the following random tests into their own test class or test suite
    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNGState(self):
        # [FUNCTION DOC] Checks get_rng_state and set_rng_state: reading the state does not alter
        # it, and restoring the state reproduces the same random numbers. Skipped under Dynamo.
        state = torch.get_rng_state()
        stateCloned = state.clone()
        before = torch.rand(1000)

        self.assertEqual(state.ne(stateCloned).long().sum(), 0, atol=0, rtol=0)

        torch.set_rng_state(state)
        after = torch.rand(1000)
        self.assertEqual(before, after, atol=0, rtol=0)

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNGStateAliasing(self):
        # [FUNCTION DOC] Checks that a generator seeded from the global RNG state produces the
        # same values as the global generator even after the global one has advanced (state is
        # copied, not aliased). Skipped under Dynamo.
        # Fork the random number stream at this point
        gen = torch.Generator()
        gen.set_state(torch.get_rng_state())
        self.assertEqual(gen.get_state(), torch.get_rng_state())

        target_value = torch.rand(1000)
        # Dramatically alter the internal state of the main generator
        _ = torch.rand(100000)
        forked_value = torch.rand(1000, generator=gen)
        self.assertEqual(target_value, forked_value, atol=0, rtol=0, msg="RNG has not forked correctly.")

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNG_after_pickle(self):
        # [FUNCTION DOC] Checks that pickling a tensor with ForkingPickler does not consume
        # random numbers: after reseeding to 100, random values equal those drawn before. Skipped
        # under Dynamo.
        torch.random.manual_seed(100)
        before = torch.rand(10)

        torch.random.manual_seed(100)
        buf = io.BytesIO()
        tensor = torch.tensor([1, 2, 3])
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(tensor)
        after = torch.rand(10)

        self.assertEqual(before, after, atol=0, rtol=0)

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_boxMullerState(self):
        # [FUNCTION DOC] Checks the cached Box-Muller state: after restoring the saved RNG state,
        # drawing an odd number of normal samples gives the same sequence, and reseeding gives the
        # same sequence as the first seeded draw. Skipped under Dynamo.
        torch.manual_seed(123)
        odd_number = 101
        seeded = torch.randn(odd_number)
        state = torch.get_rng_state()
        midstream = torch.randn(odd_number)
        torch.set_rng_state(state)
        repeat_midstream = torch.randn(odd_number)
        torch.manual_seed(123)
        reseeded = torch.randn(odd_number)
        self.assertEqual(midstream, repeat_midstream, atol=0, rtol=0,
                         msg='get_rng_state/set_rng_state not generating same sequence of normally distributed numbers')
        self.assertEqual(seeded, reseeded, atol=0, rtol=0,
                         msg='repeated calls to manual_seed not generating same sequence of normally distributed numbers')

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_manual_seed(self):
        # [FUNCTION DOC] Checks manual_seed and initial_seed: same seed reproduces samples,
        # boundary seeds (max int64, above it, max uint64, zero) are kept, negative seeds wrap
        # around to the unsigned range, out-of-range seeds raise ValueError (overflow), and the
        # original RNG state is restored at the end. Skipped under Dynamo.
        rng_state = torch.get_rng_state()
        torch.manual_seed(2)
        x = torch.randn(100)
        self.assertEqual(torch.initial_seed(), 2)
        torch.manual_seed(2)
        y = torch.randn(100)
        self.assertEqual(x, y)

        max_int64 = 0x7fff_ffff_ffff_ffff
        min_int64 = -max_int64 - 1
        max_uint64 = 0xffff_ffff_ffff_ffff
        # Check all boundary cases of valid seed value inputs
        test_cases = [
            # (seed, expected_initial_seed)
            # Positive seeds should be unchanged
            (max_int64, max_int64),
            (max_int64 + 1, max_int64 + 1),
            (max_uint64, max_uint64),
            (0, 0),
            # Negative seeds wrap around starting from the largest seed value
            (-1, max_uint64),
            (min_int64, max_int64 + 1)
        ]
        for seed, expected_initial_seed in test_cases:
            torch.manual_seed(seed)
            actual_initial_seed = torch.initial_seed()
            msg = (f"expected initial_seed() = {expected_initial_seed:x} "
                   f"after calling manual_seed({seed:x}), but got {actual_initial_seed:x} instead")
            self.assertEqual(expected_initial_seed, actual_initial_seed, msg=msg)
        for invalid_seed in [min_int64 - 1, max_uint64 + 1]:
            with self.assertRaisesRegex(ValueError, r'Overflow when unpacking long long'):
                torch.manual_seed(invalid_seed)

        torch.set_rng_state(rng_state)

    # FIXME: Describe this test and port to the generic device framework in a more
    #   appropriate test suite for the copy operation
    def test_copy_transpose(self):
        # [FUNCTION DOC] Checks copy_ from a transposed 100 by 100 tensor into float, double,
        # complex float and complex half contiguous destinations by comparing selected columns
        # with ranges (regression for issue 45269).
        x = torch.arange(100 * 100, dtype=torch.float).reshape(100, 100).t()
        y = torch.empty(100, 100, dtype=torch.float)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

        y = torch.empty(100, 100, dtype=torch.double)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

        # Validates regression reported in https://github.com/pytorch/pytorch/issues/45269
        x = torch.arange(100 * 100).reshape(100, 100).to(dtype=torch.cfloat).t()
        y = torch.empty(100, 100, dtype=torch.cfloat)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

        x = torch.arange(100 * 100).reshape(100, 100).to(dtype=torch.complex32).t()
        y = torch.empty(100, 100, dtype=torch.complex32)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

    # FIXME: Port to a more appropriate test suite
    def test_copy_broadcast(self):
        # [FUNCTION DOC] Checks copy_ broadcasting a 6-element tensor into 5 by 6 succeeds and
        # copying a 30-element tensor into it raises RuntimeError.
        torch.zeros(5, 6).copy_(torch.zeros(6))
        self.assertRaises(RuntimeError, lambda: torch.zeros(5, 6).copy_(torch.zeros(30)))

    # FIXME: Port to a more appropriate test suite
    # Fails with inductor (and aot_eager) because functionalization replaces copy_ with copy,
    # which doesn't properly error on bad inputs.
    def test_copy_many_to_one(self):
        # [FUNCTION DOC] Checks copy_ into an expanded tensor (many writes to one location)
        # raises RuntimeError.
        # Testing in-place copy where it attempt to write from many memory
        # storage to a single storage would cause RuntimeError to be thrown
        self.assertRaises(RuntimeError, lambda: torch.zeros(1, 6).expand(5, 6).copy_(torch.zeros(5, 6)))

    def test_copy_float16(self):
        # [FUNCTION DOC] Checks copy_ between half and float tensors over many shape pairs (and
        # the float to float TensorIterator path): shape combinations that are invalid must raise
        # RuntimeError, valid ones must succeed and, when CUDA is available, match the CUDA
        # result. Regression for out-of-bounds reads in the fbgemm half-to-float conversion.
        # Check that fbgemm code no longer reads memory out of bounds, see
        # copy_impl and fbgemm::Float16ToFloat_ref.
        # https://github.com/pytorch/pytorch/issues/88543

        # Types to test different code paths in copy_impl.
        dtypes = (
            # out_dtype, src_dtype
            (torch.float32, torch.float16),  # fbgemm
            (torch.float16, torch.float32),  # fbgemm
            (torch.float32, torch.float32),  # TensorIterator
        )

        cases = (
            # out_shape, src_shape, is_ok
            # These cases used to crash with fbgemm, make sure these also raise
            # exceptions with TensorIterator.
            ((1, 2, 3), (0, 2, 3), False),  # same strides, not allowed by TI
            ((1, 5, 6), (4, 5, 6), False),  # same strides, not allowed by TI
            (1, (0, 2, 3), False),  # different strides
            ((4, 5, 6), (0, 2, 3), False),  # different strides
            ((4, 5, 6), (1, 2, 3), False),  # different strides
            ((4, 5, 6), (6, 5, 4), False),  # same numel

            # These cases should pass with fbgemm and TensorIterator.
            ((4, 5, 6), (1, 5, 6), True),  # same strides
            ((4, 5, 6), (4, 5, 6), True),  # same strides
            ((0, 2, 3), 1, True),  # different strides, allowed by TI
            ((4, 5, 6), (4, 5, 1), True),  # different strides, allowed by TI
        )

        for (out_shape, src_shape, is_ok), (out_dtype, src_dtype) in itertools.product(cases, dtypes):
            out = torch.zeros(out_shape, dtype=out_dtype, device=torch.device('cpu'))
            src = torch.ones(src_shape, dtype=src_dtype, device=torch.device('cpu'))
            if is_ok:
                if torch.cuda.is_available():
                    out_cuda = out.cuda()
                    src_cuda = src.cuda()
                res = out.copy_(src)
                if torch.cuda.is_available():
                    res_cuda = out_cuda.copy_(src_cuda)
                    self.assertEqual(res, res_cuda)
            else:
                self.assertRaises(RuntimeError, lambda: out.copy_(src))

    # FIXME: Port to a more appropriate test suite
    def _test_to_with_layout(self, layout):
        # [FUNCTION DOC] HELPER: Shared helper for the to tests with a given layout (strided, or
        # compressed sparse row for the second call). Checks that to returns the same object when
        # nothing changes and a new object when copy is forced, with dtype, device and tensor
        # arguments; that device and dtype conversions work; that data pointers are preserved or
        # copied accordingly (using the component tensors for sparse layouts); and runs the same
        # checks across CUDA devices and non_blocking options when CUDA is available.
        def test_copy_behavior(t, non_blocking=False):
            # [FUNCTION DOC] HELPER: For tensor t, assert that to with the tensor, its dtype, or a
            # like-tensor returns t itself, that copy=True returns a different object, and repeat
            # for the equivalent device spellings (index-less CUDA names, current device).
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True))

            devices = [t.device]
            if t.device.type == 'cuda':
                if t.device.index == -1:
                    devices.append(f'cuda:{torch.cuda.current_device()}')
                elif t.device.index == torch.cuda.current_device():
                    devices.append('cuda')
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True))

        a = torch.tensor(5)
        if layout == torch.sparse_csr:
            a = torch.tensor([[0, 1, 2], [2, 0, 3]]).to_sparse_csr()
        test_copy_behavior(a)
        self.assertEqual(a.device, a.to('cpu').device)
        self.assertEqual(a.device, a.to('cpu', dtype=torch.float32).device)
        self.assertIs(torch.float32, a.to('cpu', dtype=torch.float32).dtype)
        self.assertEqual(a.device, a.to(torch.float32).device)
        self.assertIs(torch.float32, a.to(dtype=torch.float32).dtype)

        def test_data_ptr(getter):
            # [FUNCTION DOC] HELPER: Assert that the data pointer found by the getter is the same
            # for to with the same device, with copy off, and different with copy on.
            self.assertEqual(getter(a), getter(a.to('cpu')))
            self.assertEqual(getter(a), getter(a.to(dtype=a.dtype, device=a.device, copy=False)))
            self.assertEqual(getter(a), getter(a.to('cpu', copy=False)))
            self.assertNotEqual(getter(a), getter(a.to('cpu', copy=True)))
        if layout == torch.sparse_csr:
            # TODO: compressed sparse tensors currently don't support data_ptr.
            # Exercising failure will allow us to widen coverage of this test once it does.
            with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer of Tensor that doesn't have storage"):
                a.data_ptr()
            # While compressed sparse tensors don't have a concept of data_ptr
            # the underlying tensors do. The implementation of to appropriately forwards
            # the call to the components, which is what we're test here.
            test_data_ptr(lambda a: a.values().data_ptr())
            test_data_ptr(lambda a: a.crow_indices().data_ptr())
            test_data_ptr(lambda a: a.col_indices().data_ptr())
        else:
            test_data_ptr(lambda a: a.data_ptr())

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5., device=cuda)
                    test_copy_behavior(b, non_blocking)
                    self.assertEqual(b.device, b.to(cuda, non_blocking=non_blocking).device)
                    self.assertEqual(a.device, b.to('cpu', non_blocking=non_blocking).device)
                    self.assertEqual(b.device, a.to(cuda, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, b.to('cpu', dtype=torch.int32, non_blocking=non_blocking).dtype)
                    self.assertEqual(a.device, b.to('cpu', dtype=torch.int32, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, b.to(dtype=torch.int32).dtype)
                    self.assertEqual(b.device, b.to(dtype=torch.int32).device)

    def test_to(self):
        # [FUNCTION DOC] Runs the to-layout checks for the strided layout, and also for the
        # compressed sparse row layout if the build has CUDA.
        self._test_to_with_layout(torch.strided)
        if torch.version.cuda is not None:
            self._test_to_with_layout(torch.sparse_csr)

    # FIXME: describe this test
    def test_as_subclass(self):
        # [FUNCTION DOC] Checks as_subclass: results have the subclass type and share data with
        # the original (including after in-place edits), class members are visible, autograd flows
        # through a cast tensor back to the original, and a class that is not a Tensor subclass
        # raises TypeError.
        class SubTensor(torch.Tensor):
            # [CLASS DOC] Tensor subclass with a class attribute holding a unique object, used to
            # check as_subclass passes class members through.
            member_var = object()

        t0 = torch.tensor(0)
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([[3, 4], [5, 6]])

        s0 = t0.as_subclass(SubTensor)
        s1 = t1.as_subclass(SubTensor)
        s2 = t2.as_subclass(SubTensor)

        # Check that the correct type is returned.
        self.assertTrue(type(s0) is SubTensor)
        self.assertTrue(type(s1) is SubTensor)
        self.assertTrue(type(s2) is SubTensor)

        # Check that the data is equal.
        self.assertEqual(t0, s0)
        self.assertEqual(t1, s1)
        self.assertEqual(t2, s2)

        t0[()] = 1
        t1[1] = 3
        t2[1, 1] = 7

        # Check that the data is equal even after modification.
        self.assertEqual(t0, s0)
        self.assertEqual(t1, s1)
        self.assertEqual(t2, s2)

        # Check that member variables are passed through.
        self.assertTrue(s0.member_var is SubTensor.member_var)
        self.assertTrue(s1.member_var is SubTensor.member_var)
        self.assertTrue(s2.member_var is SubTensor.member_var)

        # Test that autograd is propagated.
        t = torch.tensor(5, dtype=torch.float32, requires_grad=True)

        # Run a calculation on the tensor.
        exp_t = torch.exp(t)

        # Cast exp_t to a subclass.
        exp_s = exp_t.as_subclass(SubTensor)

        # Make sure that t.grad was initially None
        self.assertTrue(t.grad is None)

        # Run the autograd calculation.
        exp_s.backward()

        # Make sure autograd was propagated to the original tensor
        # declared with requires_grad.
        self.assertTrue(t.grad is not None)

        # Make sure invalid subclasses raise nice errors
        class BadSubTensor:
            # [CLASS DOC] Plain class (not a Tensor subclass) with a class attribute, used to
            # check as_subclass rejects it with TypeError.
            member_var = object()

        err_msg = "Creating a Tensor subclass from a class that does not inherit from Tensor"
        with self.assertRaisesRegex(TypeError, err_msg):
            s0 = t0.as_subclass(BadSubTensor)

    # FIXME: Port to a test suite that better fits slicing
    def test_slice(self):
        # [FUNCTION DOC] Checks tensor slicing semantics on a 4 by 4 matrix: full slices, clamped
        # stop beyond the size, empty results when start is not before stop or is out of bounds,
        # negative bounds and steps.
        empty = torch.empty(0, 4)
        x = torch.arange(0., 16).view(4, 4)
        self.assertEqual(x[:], x)
        self.assertEqual(x[:4], x)
        # start and stop are clamped to the size of dim
        self.assertEqual(x[:5], x)
        # if start >= stop then the result is empty
        self.assertEqual(x[2:1], empty)
        self.assertEqual(x[2:2], empty)
        # out of bounds is also empty
        self.assertEqual(x[10:12], empty)
        # additional correctness checks
        self.assertEqual(x[:1].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(x[:-3].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(x[:, -2:3].tolist(), [[2], [6], [10], [14]])
        self.assertEqual(x[0:-1:2].tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])

    def test_split_with_sizes_copy_out(self):
        # [FUNCTION DOC] Checks torch.split_with_sizes_copy with out= for several dims (positive
        # and negative) and size lists (including a zero-size piece): the out tensors start
        # different from the expected views and equal them afterward; when CUDA is available the
        # same copy is captured in a CUDA graph and replayed.
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        shape = (30, 40, 50)
        x = torch.rand(*shape, device=device)
        cases = [
            (0, [3, 7, 8, 12]),
            (1, [3, 7, 10, 20]),
            (-2, [3, 7, 10, 20]),
            (2, [3, 7, 10, 12, 18]),
            (-1, [3, 7, 10, 12, 18]),
            (2, [3, 7, 10, 0, 30]),
        ]
        for dim, split_sizes in cases:
            views = x.split_with_sizes(split_sizes, dim=dim)
            expects = [v.clone() for v in views]
            out = [torch.zeros_like(v) for v in views]
            for expect, t in zip(expects, out):
                if expect.numel() != 0:
                    self.assertFalse(expect.eq(t).all().item())

            torch.split_with_sizes_copy(x, split_sizes, dim=dim, out=out)
            for expect, t in zip(expects, out):
                self.assertTrue(expect.eq(t).all().item())

            if not torch.cuda.is_available():
                continue

            # Test with cuda graph
            out = [torch.zeros_like(v) for v in views]
            for expect, t in zip(expects, out):
                if expect.numel() != 0:
                    self.assertFalse(expect.eq(t).all().item())

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                torch.split_with_sizes_copy(x, split_sizes, dim=dim, out=out)

            g.replay()
            for expect, t in zip(expects, out):
                self.assertTrue(expect.eq(t).all().item())

    def test_type(self):
        # [FUNCTION DOC] Checks Tensor.type with a type string, legacy tensor class, the generic
        # Tensor class (default dtype) and a dtype.
        x = torch.randn(3, 3).double()
        self.assertEqual(x.type('torch.FloatTensor').dtype, torch.float32)
        self.assertEqual(x.type(torch.FloatTensor).dtype, torch.float32)
        self.assertEqual(x.int().type(torch.Tensor).dtype, torch.get_default_dtype())
        self.assertEqual(x.type(torch.int32).dtype, torch.int32)

    # FIXME: port to a quantization test suite
    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/157245")
    @xfailIfS390X
    def test_qengine(self):
        # [FUNCTION DOC] Iterates the supported quantized engines, sets each as the active engine
        # and raises AssertionError if the engine was not set, restoring the original engine
        # afterward. Skipped on macOS and s390x (expected failure).
        qengines = torch.backends.quantized.supported_engines
        original_qe = torch.backends.quantized.engine
        for qe in qengines:
            torch.backends.quantized.engine = qe
            if torch.backends.quantized.engine != qe:
                raise AssertionError(f"qengine not set successfully: expected {qe}, got {torch.backends.quantized.engine}")
        torch.backends.quantized.engine = original_qe

    def test_terminate_handler_on_crash(self):
        # [FUNCTION DOC] Runs a subprocess that enables the custom terminate handler and calls the
        # internal abort; checks it exits with a nonzero status and prints the unhandled-exception
        # message from the abort handler.
        cmd = [sys.executable, '-c', "import os; os.environ[\"TORCH_CUSTOM_TERMINATE\"] ='1'; \
               import torch; import torch._C; torch._C._abort()"]
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            subprocess.check_output(cmd, shell=False)
        e = cm.exception
        output = e.stdout.decode("utf-8")
        self.assertNotEqual(e.returncode, 0)
        self.assertNotEqual(output, None)
        self.assertIn('Unhandled exception caught in c10/util/AbortHandler.h', output)

    # FIXME: port to a distributed test suite
    @slowTest
    def test_multinomial_invalid_probs(self):
        # [FUNCTION DOC] Slow test meant to check multinomial with invalid probabilities
        # (negative, inf, -inf, nan) raises a RuntimeError message in a spawned process.
        # KNOWN ISSUE: the four spawn calls are indented inside the probe function after its
        # try/except that always returns, so they are unreachable and the test body only defines
        # the helpers; effectively nothing is tested.
        def _spawn_method(self, method, arg):
            # [FUNCTION DOC] HELPER: Set the multiprocessing start method to spawn (ignoring if
            # already set), run the given method on the argument in a one-worker pool and assert
            # the result is truthy.
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            with mp.Pool(1) as pool:
                out = pool.map(method, [arg])
                self.assertTrue(out[0])

        def _test_multinomial_invalid_probs(probs):
            # [FUNCTION DOC] HELPER: Call multinomial with two samples on CPU for the given
            # probabilities; return False if no error occurs, or True if a RuntimeError mentioning
            # invalid probabilities (inf, nan or negative) is raised.
            try:
                # n_sample = 1 is a special case, test n_sample=2 which is more general
                torch.multinomial(probs.to('cpu'), 2)
                return False  # Should not be reached
            except RuntimeError as e:
                return 'probability tensor contains either `inf`, `nan` or element < 0' in str(e)

            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., -1., 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., inf, 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., -inf, 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., 1., nan]))

    # FIXME: port to more appropriate test suite
    def test_to_with_tensor(self):
        # [FUNCTION DOC] Checks Tensor.to with another tensor as target keeps the target's device
        # (CPU, and across CUDA devices with and without non_blocking when CUDA is available).
        a = torch.tensor(5)
        self.assertEqual(a.device, a.to(a).device)

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5., device=cuda)
                    self.assertEqual(b.device, b.to(b, non_blocking=non_blocking).device)
                    self.assertEqual(a.device, b.to(a, non_blocking=non_blocking).device)
                    self.assertEqual(b.device, a.to(b, non_blocking=non_blocking).device)

    def test_device(self):
        # [FUNCTION DOC] Tests torch.device: string and (type, index) construction gives expected
        # str, type and index for cpu and cuda; malformed strings and invalid indexes raise
        # RuntimeError; hashes are distinct for distinct devices; and repr shows the type and
        # index form.
        cpu = torch.device('cpu')
        self.assertEqual('cpu', str(cpu))
        self.assertEqual('cpu', cpu.type)
        self.assertEqual(None, cpu.index)

        cpu0 = torch.device('cpu:0')
        self.assertEqual('cpu:0', str(cpu0))
        self.assertEqual('cpu', cpu0.type)
        self.assertEqual(0, cpu0.index)

        cpu0 = torch.device('cpu', 0)
        self.assertEqual('cpu:0', str(cpu0))
        self.assertEqual('cpu', cpu0.type)
        self.assertEqual(0, cpu0.index)

        cuda = torch.device('cuda')
        self.assertEqual('cuda', str(cuda))
        self.assertEqual('cuda', cuda.type)
        self.assertEqual(None, cuda.index)

        cuda1 = torch.device('cuda:1')
        self.assertEqual('cuda:1', str(cuda1))
        self.assertEqual('cuda', cuda1.type)
        self.assertEqual(1, cuda1.index)

        cuda1 = torch.device('cuda', 1)
        self.assertEqual('cuda:1', str(cuda1))
        self.assertEqual('cuda', cuda1.type)
        self.assertEqual(1, cuda1.index)

        cuda90 = torch.device('cuda', 90)
        self.assertEqual('cuda:90', str(cuda90))
        self.assertEqual('cuda', cuda90.type)
        self.assertEqual(90, cuda90.index)

        self.assertRaises(RuntimeError, lambda: torch.device('cpu:-1'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:-1'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 '))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda: 2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2.'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2?'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:?2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2.232'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2+cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device(-1))

        self.assertRaises(RuntimeError, lambda: torch.device('other'))
        self.assertRaises(RuntimeError, lambda: torch.device('other:0'))

        device_set = {'cpu', 'cpu:0', 'cuda', 'cuda:0', 'cuda:1', 'cuda:10', 'cuda:100'}
        device_hash_set = set()
        device_hash_set.update(hash(torch.device(device)) for device in device_set)
        self.assertEqual(len(device_set), len(device_hash_set))

        def get_expected_device_repr(device):
            # [FUNCTION DOC] HELPER: Return the expected repr string for a device: type and index
            # when the index is set, otherwise only the type.
            if device.index is not None:
                return f"device(type='{device.type}', index={device.index})"

            return f"device(type='{device.type}')"

        for device in device_set:
            dev = torch.device(device)
            self.assertEqual(repr(dev), get_expected_device_repr(dev))

    # Tests that the use_deterministic_flag can be set as expected
    @wrapDeterministicFlagAPITest
    def test_deterministic_flag(self):
        # [FUNCTION DOC] Tests the deterministic algorithms flags:
        # use_deterministic_algorithms with every combination of deterministic and warn_only
        # gives the expected getters and debug mode (0 default, 1 warn, 2 error);
        # set_deterministic_debug_mode accepts numbers and strings; non-bool arguments raise
        # TypeError. The decorator restores global flag state.
        for deterministic, warn_only in product([True, False], [True, False]):
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
            self.assertEqual(warn_only, torch.is_deterministic_algorithms_warn_only_enabled())

            if deterministic:
                if warn_only:
                    debug_mode = 1
                else:
                    debug_mode = 2
            else:
                debug_mode = 0

            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())

        for debug_mode in [0, 1, 2]:
            torch.set_deterministic_debug_mode(debug_mode)
            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())
            deterministic = debug_mode in [1, 2]
            warn_only = debug_mode == 1

            self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
            self.assertEqual(warn_only, torch.is_deterministic_algorithms_warn_only_enabled())

        for debug_mode, debug_mode_str in [(0, 'default'), (1, 'warn'), (2, 'error')]:
            torch.set_deterministic_debug_mode(debug_mode_str)
            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())

        with self.assertRaisesRegex(
                TypeError,
                r"_set_deterministic_algorithms\(\): argument 'mode' \(position 1\) must be bool, not int"):
            torch.use_deterministic_algorithms(1)

        with self.assertRaisesRegex(
                TypeError,
                r"_set_deterministic_algorithms\(\): argument 'warn_only' must be bool, not int"):
            torch.use_deterministic_algorithms(False, warn_only=1)

    # Tests that torch.utils.deterministic.fill_uninitialized_memory can be set as expected
    def test_deterministic_fill_uninitialized_memory(self):
        # [FUNCTION DOC] Tests the fill_uninitialized_memory setting through DeterministicGuard
        # (nested), the Python module attribute, and the internal C setter; getters stay in sync,
        # and assigning a non-bool raises RuntimeError.
        with DeterministicGuard(True, fill_uninitialized_memory=False):
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            with DeterministicGuard(True, fill_uninitialized_memory=True):
                self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
                self.assertTrue(torch._C._get_deterministic_fill_uninitialized_memory())

            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            torch.utils.deterministic.fill_uninitialized_memory = False
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            torch.utils.deterministic.fill_uninitialized_memory = True
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertTrue(torch._C._get_deterministic_fill_uninitialized_memory())

            torch._C._set_deterministic_fill_uninitialized_memory(False)
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            torch._C._set_deterministic_fill_uninitialized_memory(True)
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertTrue(torch._C._get_deterministic_fill_uninitialized_memory())

            with self.assertRaisesRegex(RuntimeError, r"expected a bool, but got int"):
                torch.utils.deterministic.fill_uninitialized_memory = 1

    def test_type_conversion_via_dtype_name(self):
        # [FUNCTION DOC] Checks the dtype conversion methods (byte, bool, char, double, float,
        # half, int, bfloat16, cfloat, cdouble, chalf) give the right dtypes, with zero imaginary
        # part and matching real part for complex.
        x = torch.tensor([1])
        self.assertEqual(x.byte().dtype, torch.uint8)
        self.assertEqual(x.bool().dtype, torch.bool)
        self.assertEqual(x.char().dtype, torch.int8)
        self.assertEqual(x.double().dtype, torch.float64)
        self.assertEqual(x.float().dtype, torch.float32)
        self.assertEqual(x.half().dtype, torch.float16)
        self.assertEqual(x.int().dtype, torch.int32)
        self.assertEqual(x.bfloat16().dtype, torch.bfloat16)
        cfloat = x.cfloat()
        self.assertEqual(cfloat.dtype, torch.complex64)
        self.assertEqual(cfloat.real, x.float())
        self.assertEqual(cfloat.imag, torch.zeros_like(cfloat.imag))
        cdouble = x.cdouble()
        self.assertEqual(cdouble.dtype, torch.complex128)
        self.assertEqual(cdouble.real, x.double())
        self.assertEqual(cdouble.imag, torch.zeros_like(cdouble.imag))
        chalf = x.chalf()
        self.assertEqual(chalf.dtype, torch.complex32)
        self.assertEqual(chalf.real, x.half())
        self.assertEqual(chalf.imag, torch.zeros_like(chalf.imag))

    def test_type_alias(self):
        # [FUNCTION DOC] Checks that the dtype aliases (double, float, int, long, short, half,
        # chalf, cfloat) are the same objects as the canonical dtypes.
        type_alias_map = {torch.float64: torch.double,
                          torch.float32: torch.float,
                          torch.int32: torch.int,
                          torch.int64: torch.long,
                          torch.int16: torch.short,
                          torch.float16: torch.half,
                          torch.complex32: torch.chalf,
                          torch.complex64: torch.cfloat}
        for dtype, alias in type_alias_map.items():
            self.assertIs(alias, dtype)

    def test_doc_template(self) -> None:
        # [FUNCTION DOC] Reads the torch docs source file, extracts each documented function name
        # and docstring with a regular expression, and checks no docstring repeats the text of a
        # common argument description (multi-dim, single-dim, factory arguments) that should use a
        # template placeholder; the failure message names the argument and function.
        """
        Test that all public API doc strings use the same standard template for
        all common arguments such as tensor or dim
        """
        from torch._torch_docs import __file__ as doc_file
        from torch._torch_docs import multi_dim_common, single_dim_common, factory_common_args, factory_like_common_args

        with open(doc_file, encoding="utf-8") as f:
            doc_strs = f.read()

        matches = re.findall(
            r'add_docstr\(([^,]+?),[^"\']*?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')(?:\.|,?[^,\)]*?\))',
            doc_strs,
            re.MULTILINE | re.DOTALL,
        )
        self.assertTrue(matches)

        for m in matches:
            func = m[0].strip()
            desc = m[1].strip()

            for common_args in [multi_dim_common, single_dim_common, factory_common_args, factory_like_common_args]:
                for k, v in common_args.items():
                    self.assertNotIn(v, desc, lambda msg: f'{msg}\nThe argument description "{v}" in {func} can be replaced by {{{k}}}')

    def test_doc(self):
        # [FUNCTION DOC] Intended to check that public methods of tensor, torch.nn and
        # torch.nn.functional have docstrings (with skip lists for known missing ones), but it
        # only defines a namespace-checking helper and never calls it.
        # KNOWN ISSUE: the helper is never invoked by the test, and its trailing block (indented
        # as part of the helper body) calls a differently named, undefined function, so no
        # documentation is actually checked.
        checked_types = (types.MethodType, types.FunctionType,
                         types.BuiltinFunctionType, types.BuiltinMethodType)

        def _test_namespace(ns, *skips):
            # [FUNCTION DOC] HELPER: For a namespace object, walk its public attribute names
            # (using fresh values for real, imag and matrix transpose attributes), and for each
            # function or method assert it has a non-empty docstring unless it matches a skip
            # entry, in which case assert it has none. The trailing calls after the loop use an
            # undefined name and never run.
            if isinstance(ns, object):
                ns_name = ns.__class__.__name__
            else:
                ns_name = ns.__name__
            skip_regexes = []
            for r in skips:
                if isinstance(r, str):
                    skip_regexes.append(re.compile(f'^{re.escape(r)}$'))
                else:
                    skip_regexes.append(r)

            for name in dir(ns):
                if name.startswith('_'):
                    continue
                if name in ['real', 'imag']:
                    y = torch.randn(1, dtype=torch.cfloat)
                    var = getattr(y, name)
                elif name in ["H", "mT", "mH"]:
                    y = torch.randn(1, 1)
                    var = getattr(y, name)
                else:
                    var = getattr(ns, name)
                if not isinstance(var, checked_types):
                    continue
                doc = var.__doc__
                has_doc = doc is not None and len(doc.strip()) > 0
                full_name = ns_name + '.' + name
                if any(r.match(name) for r in skip_regexes):
                    self.assertFalse(has_doc,
                                     lambda msg: f'{msg}\nNew docs have been added for {full_name}, please remove '
                                     'it from the skipped list in TestTorch.test_doc')
                else:
                    self.assertTrue(has_doc, lambda msg: f'{msg}\n{full_name} is missing documentation')

            # FIXME: All of the following should be marked as expected failures
            # so that it is easier to tell when missing has been added.
            # FIXME: fix all the skipped ones below!
            test_namespace(torch.randn(1),  # noqa: F821
                           'as_strided_',
                           re.compile('^clamp_(min|max)_?$'),
                           'is_distributed',
                           'is_nonzero',
                           'is_same_size',
                           'log_softmax',
                           'map2_',
                           'new',
                           'reinforce',
                           'relu',
                           'relu_',
                           'prelu',
                           'resize',
                           'resize_as',
                           'softmax',
                           'split_with_sizes',
                           'unsafe_split_with_sizes',
                           '_autocast_to_fp16',
                           '_autocast_to_fp32',
                           )

            test_namespace(torch.nn)  # noqa: F821
            test_namespace(torch.nn.functional, 'assert_int_or_pair')  # noqa: F821
            # TODO: add torch.* tests when we have proper namespacing on ATen functions
            # test_namespace(torch)

    # FIXME: deprecate torch.Tensor constructor
    def test_tensor_ctor_scalar(self):
        # [FUNCTION DOC] Checks constructing a legacy Tensor from a zero-dim tensor gives an
        # equal tensor.
        x = torch.Tensor(torch.tensor(1.0))
        self.assertEqual(x, torch.tensor(1.0))

    def test_deepcopy_gradient(self):
        # [FUNCTION DOC] Checks deepcopy preserves dense and sparse gradients and keeps the grad
        # object shared when copying a list holding a tensor and its grad.
        from copy import deepcopy
        a = torch.zeros(10)
        a.grad = torch.ones(10)
        self.assertEqual(a.grad, deepcopy(a).grad)
        s = torch.zeros(10).to_sparse()
        s.grad = torch.ones(10).to_sparse()
        self.assertEqual(s.grad, deepcopy(s).grad)

        # ensure sharing is not broken
        c = deepcopy([a, a.grad])
        self.assertTrue(c[0].grad is c[1])

    def test_tensor_base_init(self):
        # [FUNCTION DOC] Checks that the C base class of tensors cannot be instantiated directly
        # (RuntimeError) or subclassed directly (RuntimeError mentioning cannot subclass), but a
        # Tensor subclass can be created.
        # Direct construction not OK
        self.assertRaises(RuntimeError, lambda: torch._C.TensorBase())

        # Subclassing it directly no OK
        with self.assertRaisesRegex(RuntimeError, "Cannot subclass"):
            class Tfail(torch._C.TensorBase):
                # [CLASS DOC] Direct subclass of the C tensor base type, defined only to check
                # that class creation raises an error; no body.
                pass

        # Doing so with Tensor is ok though
        class T(torch.Tensor):
            # [CLASS DOC] Plain subclass of Tensor used to check that Tensor subclasses can be
            # instantiated; no body.
            pass

        T()

    def test_storage_base_init(self):
        # [FUNCTION DOC] Checks that the C storage base class cannot be instantiated directly
        # (RuntimeError) but a plain subclass of it can.
        # Direct construction not OK
        self.assertRaises(RuntimeError, lambda: torch._C.StorageBase())

        # But construction of subclass is OK
        class T(torch._C.StorageBase):
            # [CLASS DOC] Subclass of the C storage base type, defined to check subclassing
            # behavior; no body.
            pass

        T()

    def test_tensor_base_new(self):
        # [FUNCTION DOC] Regression for issue 57421: a Tensor subclass whose static new method
        # forwards extra arguments to the parent new can be instantiated.

        # OK to call super().__new__, see
        # https://github.com/pytorch/pytorch/issues/57421
        class TestTensor(torch.Tensor):
            # [CLASS DOC] Tensor subclass overriding the allocation hook so extra arguments are
            # accepted and ignored.
            @staticmethod
            def __new__(cls, x, *args, **kwargs):
                # [FUNCTION DOC] Allocation hook for the test subclass: delegates to the parent
                # implementation with only the first positional value.
                return super().__new__(cls, x, *args, **kwargs)

        x = torch.ones(5)
        TestTensor(x)

    def test_storage_base_new(self):
        # [FUNCTION DOC] Regression for issue 57421 on storages: a storage-base subclass whose
        # static new method forwards extra arguments to the parent new can be instantiated from an
        # untyped storage.

        # OK to call super().__new__, see
        # https://github.com/pytorch/pytorch/issues/57421
        class TestStorage(torch._C.StorageBase):
            # [CLASS DOC] Storage-base subclass overriding the allocation hook so extra arguments
            # are accepted and ignored.
            @staticmethod
            def __new__(cls, x, *args, **kwargs):
                # [FUNCTION DOC] Allocation hook for the test storage subclass: delegates to the
                # parent implementation with only the first positional value.
                return super().__new__(cls, x, *args, **kwargs)

        x = torch.UntypedStorage(5)
        TestStorage(x)

    def test_pyobj_preserved(self):
        # [FUNCTION DOC] Checks the Python wrapper of a tensor is kept alive by the C++ object: a
        # tensor with a custom attribute stored as the grad of another keeps that attribute after
        # the Python reference is dropped and re-acquired.
        x = torch.empty(2)
        x.foo = 2  # put something on __dict__
        y = torch.empty(2)
        y.grad = x
        del x  # x is dead in Python
        self.assertEqual(y.grad.foo, 2)
        z = y.grad  # it's live
        del z  # it's dead again
        self.assertEqual(y.grad.foo, 2)

    def test_subclass_preserved(self):
        # [FUNCTION DOC] Checks the Python subclass type of a tensor stored as another tensor's
        # grad is preserved after the Python reference is dropped and reacquired.
        class MyTensor(torch.Tensor):
            # [CLASS DOC] Empty Tensor subclass used to check the subclass type survives when only
            # C++ holds a reference.
            pass

        x = MyTensor(torch.empty(2))
        y = torch.empty(2)
        y.grad = x
        del x  # x is dead in Python
        self.assertEqual(type(y.grad), MyTensor)
        z = y.grad  # it's live
        del z  # it's dead again
        self.assertEqual(type(y.grad), MyTensor)

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc(self):
        # [FUNCTION DOC] Using a Tracker marker, checks an untyped storage object with a
        # tracker attribute is destroyed only when its last Python reference is deleted.
        # Skipped under Dynamo.
        m, t = Tracker.make()
        s0 = torch.UntypedStorage(10)
        s1 = s0
        s0._tracker = t
        del t

        self.assertFalse(m[0])
        del s0
        self.assertFalse(m[0])
        del s1
        self.assertTrue(m[0])

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_from_tensor_dealloc(self):
        # [FUNCTION DOC] Checks that a storage obtained from a tensor keeps its Python object
        # (with its tracker attribute) identical across calls and is destroyed only after both the
        # tensor and the storage references are gone. Skipped under Dynamo.
        m, t = Tracker.make()
        a = torch.randn(10)
        s0 = a.untyped_storage()
        s0._tracker = t
        del t

        s1 = a.untyped_storage()
        self.assertTrue(s0 is s1)
        self.assertTrue(hasattr(s1, '_tracker'))

        del a

        self.assertFalse(m[0])
        del s0
        self.assertFalse(m[0])
        del s1
        self.assertTrue(m[0])

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_from_tensor_dealloc_zombie(self):
        # [FUNCTION DOC] Checks the storage Python object stays alive (zombie) while the tensor
        # lives, even after all storage references are deleted, and the tracker fires only after
        # the tensor is deleted. Skipped under Dynamo.
        m, t = Tracker.make()
        a = torch.randn(10)
        s0 = a.untyped_storage()
        s0._tracker = t
        del t

        s1 = a.untyped_storage()
        self.assertTrue(s0 is s1)
        self.assertTrue(hasattr(s1, '_tracker'))

        self.assertFalse(m[0])
        del s0
        self.assertFalse(m[0])
        del s1
        self.assertFalse(m[0])
        del a
        self.assertTrue(m[0])

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_from_tensor_dealloc_resurrected(self):
        # [FUNCTION DOC] Checks a storage object can be resurrected: after deleting storage
        # references while the tensor lives, taking the storage again gives an untyped
        # storage, and the tracker fires only when both tensor and new reference are gone.
        # Skipped under Dynamo.
        m, t = Tracker.make()
        a = torch.randn(10)
        s0 = a.untyped_storage()
        s0._tracker = t
        del t

        s1 = a.untyped_storage()
        self.assertTrue(s0 is s1)
        self.assertTrue(hasattr(s1, '_tracker'))

        self.assertFalse(m[0])
        del s0
        self.assertFalse(m[0])
        del s1
        self.assertFalse(m[0])

        s0 = a.untyped_storage()
        self.assertTrue(isinstance(s0, torch.UntypedStorage))

        del a
        self.assertFalse(m[0])
        del s0
        self.assertTrue(m[0])

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc_resurrected(self):
        # [FUNCTION DOC] Checks a storage with tracker wrapped by a tensor stays alive when the
        # Python reference is deleted, can be recovered from the tensor, and dies after the tensor
        # and recovered reference are deleted. Skipped under Dynamo.
        m, t = Tracker.make()
        s = torch.UntypedStorage(10)
        s._tracker = t
        del t

        a = torch.tensor(s)
        self.assertFalse(m[0])
        del s

        self.assertFalse(m[0])

        s = a.untyped_storage()
        self.assertTrue(isinstance(s, torch.UntypedStorage))

        del a
        self.assertFalse(m[0])
        del s
        self.assertTrue(m[0])

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc_subclass_zombie(self):
        # [FUNCTION DOC] Checks that a storage subclass with a finalizer is not finalized while a
        # tensor holds it (zombie), and is finalized once when the tensor goes away, firing the
        # tracker. Skipped under Dynamo.
        class MyStorage(torch.UntypedStorage):
            # [CLASS DOC] Untyped storage subclass with a class-level finalization counter
            # incremented by its destructor.
            finalized_count = 0

            def __del__(self):
                # [FUNCTION DOC] Destructor: increment the class-level finalization counter.
                MyStorage.finalized_count += 1

        m, t = Tracker.make()
        s = MyStorage(10)
        s._tracker = t
        del t

        a = torch.tensor(s)
        self.assertFalse(m[0])
        del s

        self.assertEqual(MyStorage.finalized_count, 0)
        self.assertFalse(m[0])

        del a
        self.assertEqual(MyStorage.finalized_count, 1)
        self.assertTrue(m[0])

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc_subclass_resurrected(self):
        # [FUNCTION DOC] Checks a storage subclass with a finalizer can be resurrected through the
        # tensor's storage accessor, is not finalized until the new reference is dropped, and is
        # then finalized exactly once. Skipped under Dynamo.
        class MyStorage(torch.UntypedStorage):
            # [CLASS DOC] Untyped storage subclass with a class-level finalization counter
            # incremented by its destructor.
            finalized_count = 0

            def __del__(self):
                # [FUNCTION DOC] Destructor: increment the class-level finalization counter.
                MyStorage.finalized_count += 1

        m, t = Tracker.make()
        s = MyStorage(10)
        s._tracker = t
        del t

        a = torch.tensor(s)
        self.assertFalse(m[0])
        del s

        self.assertEqual(MyStorage.finalized_count, 0)
        self.assertFalse(m[0])

        s = a.untyped_storage()
        del a
        self.assertFalse(m[0])
        self.assertEqual(MyStorage.finalized_count, 0)
        self.assertTrue(isinstance(s, MyStorage))
        del s
        self.assertEqual(MyStorage.finalized_count, 1)
        self.assertTrue(m[0])

    @skipIfTorchDynamo("cannot trace t2.grad_fn._saved_self")
    def test_tensor_ressurecting_clear(self):
        # [FUNCTION DOC] Regression for issue 136358: a tensor with a custom attribute, part of a
        # reference cycle and also kept alive by the autograd graph, keeps its attribute after the
        # Python references are cleared and garbage collection runs. Skipped under Dynamo.
        # Regression test for https://github.com/pytorch/pytorch/issues/136358
        # A Tensor with custom __dict__
        # Autograd here is for the c++ reference later
        t = torch.rand(2, requires_grad=True).clone()
        t.foo = 2

        # that is part of a cycle
        l = []
        l.append(42)
        l.append(t)

        # Keep the Tensor alive from c++
        # Using autograd graph here (any other mean would work)
        t2 = t ** 2
        self.assertIs(t2.grad_fn._saved_self, t)

        # Clear all python references and trigger the gc
        del t, l
        gc.collect()

        # We used to loose the dict!
        self.assertTrue(hasattr(t2.grad_fn._saved_self, "foo"))

    def test_tensor_slot_dealloc(self):
        # [FUNCTION DOC] Checks that tensor subclasses using slots release their slot values when
        # the tensor is deleted (two slot levels, tracked with markers).

        class SlotTensor1(torch.Tensor):
            # [CLASS DOC] Tensor subclass declaring one slot.
            __slots__ = ['slot1']

        class SlotTensor2(SlotTensor1):
            # [CLASS DOC] Tensor subclass of the previous one declaring a second slot.
            __slots__ = ['slot2']

        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()
        slot_tensor = SlotTensor2(torch.empty(2))
        slot_tensor.slot1 = t1
        slot_tensor.slot2 = t2
        del t1
        del t2
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])
        del slot_tensor
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    def test_storage_slot_dealloc(self):
        # [FUNCTION DOC] Same as the tensor slot deallocation test but for subclasses of the C
        # storage base type.

        class SlotStorage1(torch._C.StorageBase):
            # [CLASS DOC] Storage-base subclass declaring one slot.
            __slots__ = ['slot1']

        class SlotStorage2(SlotStorage1):
            # [CLASS DOC] Subclass of the previous storage class declaring a second slot.
            __slots__ = ['slot2']

        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()
        slot_storage = SlotStorage2(torch.UntypedStorage(2))
        slot_storage.slot1 = t1
        slot_storage.slot2 = t2
        del t1
        del t2
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])
        del slot_storage
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_tensor_dict_dealloc(self):
        # [FUNCTION DOC] Checks that an attribute stored in a tensor's dictionary is released when
        # the tensor is deleted. Skipped under Dynamo.
        m, t = Tracker.make()
        x = torch.empty(2)
        x.arf = t
        del t
        self.assertFalse(m[0])
        del x
        self.assertTrue(m[0])

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_storage_dict_dealloc(self):
        # [FUNCTION DOC] Same as the tensor dictionary deallocation test but for an untyped
        # storage: an attribute set on the storage is released when the storage is deleted.
        # Skipped under Dynamo.
        m, t = Tracker.make()
        x = torch.UntypedStorage(2)
        x.arf = t
        del t
        self.assertFalse(m[0])
        del x
        self.assertTrue(m[0])

    def test_tensor_finalizer_dealloc(self):
        # [FUNCTION DOC] Checks a Tensor subclass with a finalizer method runs it when the
        # instance is deleted.
        m = [False]

        class FinalizerTensor(torch.Tensor):
            # [CLASS DOC] Tensor subclass whose destructor records in a shared one-element list
            # that it ran.
            def __del__(self):
                # [FUNCTION DOC] Destructor: set the shared flag to True.
                m[0] = True

        fin_tensor = FinalizerTensor(torch.empty(2))
        self.assertFalse(m[0])
        del fin_tensor
        self.assertTrue(m[0])

    def test_storage_finalizer_dealloc(self):
        # [FUNCTION DOC] Checks a C storage-base subclass with a finalizer method runs it when the
        # instance is deleted.
        m = [False]

        class FinalizerStorage(torch._C.StorageBase):
            # [CLASS DOC] Storage-base subclass whose destructor records in a shared one-element
            # list that it ran.
            def __del__(self):
                # [FUNCTION DOC] Destructor: set the shared flag to True.
                m[0] = True

        fin_storage = FinalizerStorage(torch.UntypedStorage(2))
        self.assertFalse(m[0])
        del fin_storage
        self.assertTrue(m[0])

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_tensor_weakref_dealloc(self):
        # [FUNCTION DOC] Checks a weak reference with a callback to a tensor fires the callback
        # when the tensor is deleted and the reference then returns None. Skipped under Dynamo.
        x = torch.empty(2)
        m = [False]

        def cb(r):
            # [FUNCTION DOC] HELPER: Weak reference callback: set the shared flag to True.
            m[0] = True

        wref = weakref.ref(x, cb)
        del x
        self.assertTrue(m[0])
        self.assertEqual(wref(), None)

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_storage_weakref_dealloc(self):
        # [FUNCTION DOC] Same as the tensor weakref deallocation test but for an untyped storage.
        # Skipped under Dynamo.

        x = torch.UntypedStorage(2)
        m = [False]

        def cb(r):
            # [FUNCTION DOC] HELPER: Weak reference callback: set the shared flag to True.
            m[0] = True

        wref = weakref.ref(x, cb)
        del x
        self.assertTrue(m[0])
        self.assertEqual(wref(), None)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_tensor_cycle_via_dict(self):
        # [FUNCTION DOC] Builds a reference cycle between two tensors through their attribute
        # dictionaries, with a third tensor holding one of them as grad (a C++ reference);
        # deleting Python names and running garbage collection must not free the cycle while the
        # C++ reference exists, and after dropping the holder (with garbage collection disabled)
        # both are freed only on the next collection. Skipped under Dynamo.
        m1, t1 = Tracker.make()
        x = torch.empty(2)
        x._tracker = t1
        del t1

        m2, t2 = Tracker.make()
        y = torch.empty(2)
        y._tracker = t2
        del t2

        x._loop = y
        y._loop = x

        # C++ reference should keep the cycle live!
        # This exercise THPVariable_subtype_traverse
        # NB: Because z.grad is a reference done entirely in C++, cycles
        # involving it directly are NOT broken by Python GC; you've
        # set up a good old C++ reference cycle which we cannot safely
        # break (because C++ references are allowed to be accessed
        # multithreaded-ly) (TODO: except maybe if you can prove that
        # only Python has access to the C++ object, in which case you can
        # also prove that no multithreaded access occurs)
        z = torch.empty(2)
        z.grad = x

        del x
        del y

        gc.collect()
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])

        with disable_gc():
            del z
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_storage_cycle_via_dict(self):
        # [FUNCTION DOC] Same as the tensor dictionary-cycle test but with untyped storages
        # (attribute set as a plain Python attribute on the holder). Skipped under Dynamo.
        m1, t1 = Tracker.make()
        x = torch.UntypedStorage(2)
        x._tracker = t1
        del t1

        m2, t2 = Tracker.make()
        y = torch.UntypedStorage(2)
        y._tracker = t2
        del t2

        x._loop = y
        y._loop = x

        # C++ reference should keep the cycle live!
        # This exercise THPVariable_subtype_traverse
        # NB: Because z.grad is a reference done entirely in C++, cycles
        # involving it directly are NOT broken by Python GC; you've
        # set up a good old C++ reference cycle which we cannot safely
        # break (because C++ references are allowed to be accessed
        # multithreaded-ly) (TODO: except maybe if you can prove that
        # only Python has access to the C++ object, in which case you can
        # also prove that no multithreaded access occurs)
        z = torch.UntypedStorage(2)
        z.grad = x

        del x
        del y

        gc.collect()
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])

        with disable_gc():
            del z
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    def test_tensor_cycle_via_slots(self):
        # [FUNCTION DOC] Builds a cycle between tensor subclass instances through slots whose
        # destructors set flags; deleting the names with garbage collection disabled must not run
        # finalizers, a collection must run both, the weak reference must be cleared and no
        # instance of the subclass may remain tracked.
        m1 = [False]
        m2 = [False]

        class SlotTensor1(torch.Tensor):
            # [CLASS DOC] Tensor subclass with one slot whose destructor sets the first flag.
            __slots__ = ['slot1']

            def __del__(self):
                # [FUNCTION DOC] Destructor: set the first shared flag to True.
                m1[0] = True

        class SlotTensor2(SlotTensor1):
            # [CLASS DOC] Tensor subclass of the previous one with a second slot whose destructor
            # sets the second flag.
            __slots__ = ['slot2']

            def __del__(self):
                # [FUNCTION DOC] Destructor: set the second shared flag to True.
                m2[0] = True

        x = SlotTensor1(torch.empty(2))
        x_ref = weakref.ref(x)
        y = SlotTensor2(torch.empty(2))

        x.slot1 = y
        y.slot2 = x

        del x
        with disable_gc():
            del y
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

        self.assertIsNone(x_ref())

        # At this point, we know the finalizer ran and the weakref
        # was cleared. But is the object really gone?
        self.assertFalse(any(isinstance(o, SlotTensor1) for o in gc.get_objects()))


    def test_storage_cycle_via_slots(self):
        # [FUNCTION DOC] Same as the tensor slot-cycle test but for C storage-base subclasses
        # (without the weak reference and tracked-object checks).
        m1 = [False]
        m2 = [False]

        class SlotStorage1(torch._C.StorageBase):
            # [CLASS DOC] Storage-base subclass with one slot whose destructor sets the
            # first flag.
            __slots__ = ['slot1']

            def __del__(self):
                # [FUNCTION DOC] Destructor: set the first shared flag to True.
                m1[0] = True

        class SlotStorage2(SlotStorage1):
            # [CLASS DOC] Storage-base subclass of the previous one with a second slot whose
            # destructor sets the second flag.
            __slots__ = ['slot2']

            def __del__(self):
                # [FUNCTION DOC] Destructor: set the second shared flag to True.
                m2[0] = True

        x = SlotStorage1(torch.UntypedStorage(2))
        y = SlotStorage2(torch.UntypedStorage(2))

        x.slot1 = y
        y.slot2 = x

        del x
        with disable_gc():
            del y
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_storage_preserve_nonhermetic_in_hermetic_context(self):
        # [FUNCTION DOC] Defines a custom library operator that deletes a global holding a tracked
        # storage; checks the storage tracker is not triggered while the storage object is still
        # reachable from the tensor, and fires only after it is released (the storage must be
        # preserved across hermetic operator contexts). Skipped under Dynamo.
        from torch.library import _scoped_library, impl
        global _my_storage

        with _scoped_library("my_lib", "DEF") as my_lib:
            my_lib.define('my_func() -> None')

            a = torch.tensor([1.])
            _my_storage = a.untyped_storage()

            m, t = Tracker.make()
            _my_storage._tracker = t
            del t

            @impl(my_lib, 'my_func', '')
            def my_func():
                # [FUNCTION DOC] HELPER: Custom operator implementation: delete the module-level
                # global that holds the tracked storage.
                global _my_storage
                del _my_storage

            self.assertFalse(m[0])
            torch.ops.my_lib.my_func()
            self.assertFalse(m[0])

            s = a.untyped_storage()
            del a
            del s
            self.assertTrue(m[0])

    # FIXME: move to test_autograd?
    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_backward_hooks_traverse(self):
        # [FUNCTION DOC] Builds two tensors that reference each other through the special
        # backward-hooks attribute and checks the cycle is freed by garbage collection (not
        # before, with collection disabled). Skipped under Dynamo.
        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()
        x = torch.empty(2, requires_grad=True)
        x._tracker = t1
        y = torch.empty(2, requires_grad=True)
        y._tracker = t2
        del t1
        del t2

        # this hits a special setter, it's not just a __dict__ entry
        x._backward_hooks = y
        y._backward_hooks = x

        del x
        with disable_gc():
            del y
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()

        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_tensor_dead_weak_ref(self):
        # [FUNCTION DOC] Checks a tensor referenced only through another tensor's grad and a
        # weakref is kept alive by a re-acquired Python reference after the holder is deleted,
        # still computes correctly, and the weakref is cleared when finally deleted. Skipped
        # under Dynamo.
        x = torch.ones(2)
        w_x = weakref.ref(x)
        y = torch.ones(2)
        y.grad = x
        del x

        x = w_x()
        # x should keep the tensor live. This didn't happen in earlier PyTorch
        # versions.
        del y

        self.assertEqual(2, x.sum())

        del x
        self.assertIsNone(w_x())

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_storage_dead_weak_ref(self):
        # [FUNCTION DOC] Checks the weak reference to an untyped storage stays valid while a
        # tensor built from the storage lives and is cleared after the tensor is deleted. Skipped
        # under Dynamo.
        x = torch.UntypedStorage(2)
        w_x = weakref.ref(x)
        y = torch.tensor(x)
        del x
        self.assertIsNotNone(w_x())
        del y
        self.assertIsNone(w_x())

    def test_tensor_resurrected_weak_ref(self):
        # [FUNCTION DOC] Checks that after re-acquiring a tensor through a weak reference when
        # only the C++ holder (grad) kept it alive, calling the manual weakref fix and deleting
        # the holder leaves a usable tensor (sigmoid does not crash).
        x = torch.empty(2)
        w_x = weakref.ref(x)
        y = torch.empty(2)
        y.grad = x
        del x

        x = w_x()
        # Use this to manually fix weak references after dereferencing them
        x._fix_weakref()
        del y
        x.sigmoid()

    def test_storage_resurrected_weak_ref(self):
        # [FUNCTION DOC] Checks the same weakref resurrection and manual fix scenario for an
        # untyped storage (the float conversion afterward must not crash).
        x = torch.UntypedStorage(2)
        w_x = weakref.ref(x)
        y = torch.tensor(x)
        del x

        x = w_x()
        # Use this to manually fix weak reference after dereferencing them
        x._fix_weakref()
        del y
        x.float()

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_tensor_fix_weakref_no_leak(self):
        # [FUNCTION DOC] Checks that a weakref callback on a tensor still fires after the manual
        # weakref fix when the tensor is deleted (no leak). Skipped under Dynamo.
        import weakref

        called = False

        a = torch.randn(1)

        def callback(w):
            # [FUNCTION DOC] HELPER: Weakref callback: set the enclosing flag to True.
            nonlocal called
            called = True
        _wa = weakref.ref(a, callback)
        a._fix_weakref()
        del a

        self.assertTrue(called)

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_storage_fix_weakref_no_leak(self):
        # [FUNCTION DOC] Checks that a weakref callback on an untyped storage still fires after
        # the manual weakref fix when the storage is deleted. Skipped under Dynamo.
        import weakref

        called = False

        a = torch.UntypedStorage(1)

        def callback(w):
            # [FUNCTION DOC] HELPER: Weakref callback: set the enclosing flag to True.
            nonlocal called
            called = True
        _wa = weakref.ref(a, callback)
        a._fix_weakref()
        del a

        self.assertTrue(called)

    # FIXME: move to test_linalg
    @torch.inference_mode()
    def test_bmm_multithreaded(self):
        # [FUNCTION DOC] Under inference mode with four CPU threads, checks bmm and bmm with out
        # (output in various permuted layouts) against numpy matmul for permuted, broadcast
        # (expanded) and zero-size batch inputs with batch sizes 1 and 10; restores the thread
        # count afterward.
        device = 'cpu'
        num_threads = torch.get_num_threads()

        torch.set_num_threads(4)
        batch_sizes = [1, 10]
        M, N, O = 23, 8, 12
        dtype = torch.float32
        numpy_dtype = dtype

        def invert_perm(p):
            # [FUNCTION DOC] HELPER: Return the inverse of a permutation of the three axes, as a
            # tuple of the position of each axis.
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_inputs(num_batches):
            # [FUNCTION DOC] HELPER: Generator yielding pairs of float tensors for bmm:
            # permuted-layout inputs for all permutation pairs, broadcast-expanded inputs for all
            # size/one combinations, and zero-sized inputs.
            # transposed tensors
            for perm1, perm2 in itertools.product(itertools.permutations((0, 1, 2)), repeat=2):
                b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1)
                b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1)
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                yield b1, b2
            # broadcasting tensors
            for b1, b2, b3, b4, b5, b6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)
                shape2 = (num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, N, O)
                yield b1, b2
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = torch.randn(shape1, dtype=dtype, device=device)
                b2 = torch.randn(shape2, dtype=dtype, device=device)
                yield b1, b2

        try:
            for num_batches in batch_sizes:
                for (b1, b2), perm3 in itertools.product(generate_inputs(num_batches), itertools.permutations((0, 1, 2))):
                    res1 = torch.bmm(b1, b2)
                    res2 = torch.full((num_batches, M, O), math.nan, dtype=dtype, device=device) \
                        .permute(perm3).contiguous().permute(invert_perm(perm3))
                    torch.bmm(b1, b2, out=res2)
                    expect = torch.from_numpy(
                        b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                    self.assertEqual(expect, res1)
                    self.assertEqual(expect, res2)
        finally:
            torch.set_num_threads(num_threads)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_bmm_matmul_mixed_dtype_error(self):
        # [FUNCTION DOC] CUDA only. Checks that bmm and matmul on half and float inputs raise
        # RuntimeError about mismatched scalar types, in eager mode and under compile with
        # full graph.
        a = torch.randn(2, 8, 8, device="cuda", dtype=torch.float16)
        b = torch.randn(2, 8, 64, device="cuda", dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "expected scalar type .* but found"):
            torch.bmm(a, b)

        with self.assertRaisesRegex(RuntimeError, "expected scalar type .* but found"):
            torch.compile(lambda x, y: torch.bmm(x, y), fullgraph=True)(a, b)

        with self.assertRaisesRegex(RuntimeError, "expected scalar type .* but found"):
            torch.matmul(a, b)

        with self.assertRaisesRegex(RuntimeError, "expected scalar type .* but found"):
            torch.compile(lambda x, y: torch.matmul(x, y), fullgraph=True)(a, b)

    def test_conj_neg_tolist(self):
        # [FUNCTION DOC] Checks tolist on a conjugated complex view and its imaginary part gives
        # the same values as the physically conjugated tensor.
        x = torch.randn(2, dtype=torch.cfloat)
        y1 = x.conj()
        y1_expect = x.conj_physical()
        y2 = y1.imag
        self.assertEqual(y1, y1_expect.tolist())
        self.assertEqual(y2, y1_expect.imag.tolist())

    @unittest.skipIf(torch.backends.cuda.is_built(), "Skipped for cuda-enabled build")
    def test_no_cuda_monkeypatch(self):
        # [FUNCTION DOC] Only for builds without CUDA: checks that constructing CUDA Stream, Event
        # and CUDAGraph objects raises RuntimeError with the "requires CUDA support" or "dummy
        # base class" messages.
        # Note that this is not in test_cuda.py as this whole file is skipped when cuda
        # is not available.
        with self.assertRaisesRegex(RuntimeError, "torch.cuda.Stream requires CUDA support"):
            torch.cuda.Stream()

        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class Event"):
            torch.cuda.Event()

        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class CUDAGraph"):
            torch.cuda.graphs.CUDAGraph()

    def test_tensor_where_scalar(self):
        # [FUNCTION DOC] Checks torch.where and Tensor.where with a scalar fallback value give the
        # same result.

        a = torch.arange(4.0)
        not_zero = 0.001

        # b is generated through torch.where function with not_zero being a scalar parameter
        b = torch.where(a != 0, a, not_zero)
        # c is generated through Tensor.where method with not_zero being a scalar parameter
        c = a.where(a != 0, not_zero)

        self.assertEqual(b, c)

    def test_data_ptr_of_empty_tensor_with_storage(self):
        # [FUNCTION DOC] Checks an empty-shaped tensor resized to zero elements reports a zero
        # data pointer while keeping its storage, and a normal tensor has nonzero.
        t = torch.empty((2, 2))
        self.assertNotEqual(t.data_ptr(), 0)
        t.resize_((0, 2))
        self.assertEqual(t.data_ptr(), 0)

    def test_data_ptr_of_empty_view_with_storage(self):
        # [FUNCTION DOC] Checks a zero-element view of a tensor with storage reports a zero
        # data pointer.
        t = torch.empty((2, 2))
        self.assertNotEqual(t.data_ptr(), 0)
        t2 = t[0:0].view(0, 1)
        self.assertEqual(t2.data_ptr(), 0)

    def test_size_stride(self) -> None:
        # [FUNCTION DOC] Checks size with a dim, size with a None dim returning the full Size,
        # stride with None returning the full stride, and the stride of a transposed tensor.
        t = torch.rand(2, 3, dtype=torch.float32)
        self.assertEqual(t.size(0), 2)
        self.assertEqual(t.size(dim=None), torch.Size([2, 3]))
        self.assertEqual(t.stride(dim=None), torch.Size([3, 1]))
        self.assertEqual(t.t().stride(), torch.Size([1, 3]))

    def test_invalid_arg_error_handling(self) -> None:
        # [FUNCTION DOC] Checks that invalid thread counts (negative and huge) to set_num_threads
        # and set_num_interop_threads raise ValueError or RuntimeError.
        """ Tests that errors from old TH functions are propagated back """
        for invalid_val in [-1, 2**65]:
            self.assertRaises((ValueError, RuntimeError), lambda: torch.set_num_threads(invalid_val))
            self.assertRaises((ValueError, RuntimeError), lambda: torch.set_num_interop_threads(invalid_val))

    def test_intlist_arg_with_misbehaving_list_subclass(self) -> None:
        # [FUNCTION DOC] Regression for issue 184877: passing a list subclass whose length and
        # iteration raise errors, containing a string, as a size argument must give TypeError
        # mentioning the bad element type rather than an internal assertion.
        # Regression test for https://github.com/pytorch/pytorch/issues/184877:
        # an INTERNAL ASSERT fired when the arg parser tried to format a type
        # error for a list subclass whose __len__/__iter__ raised.
        class Foo(list):
            # [CLASS DOC] List subclass whose iteration and length both raise RuntimeError, used
            # to check argument parsing error formatting.
            def __iter__(self):
                # [FUNCTION DOC] Always raises RuntimeError saying it should not be called.
                raise RuntimeError("should not be called")

            def __len__(self):
                # [FUNCTION DOC] Always raises RuntimeError saying it should not be called.
                raise RuntimeError("should not be called")

        x = Foo()
        x.append("a")
        with self.assertRaisesRegex(
            TypeError, r"found element of type str at pos 0"
        ):
            torch.randn(x)

    def _get_tensor_prop(self, t):
        # [FUNCTION DOC] HELPER: Collect properties of a tensor for the swap tests: a pair of
        # preserved properties (object id and, unless under Dynamo, the reference count) and moved
        # properties (slot names, id and keys of the instance dictionary, and slot values).
        preserved = (
            id(t),
            # Refcount values get modified by Dynamo resume frames
            0 if TEST_WITH_TORCHDYNAMO else sys.getrefcount(t),
        )
        slotnames = copyreg._slotnames(t.__class__)
        moved = (
            slotnames,
            id(t.__dict__),
            tuple(t.__dict__.keys()),
            [getattr(t, name, None) for name in slotnames]
        )
        return preserved, moved

    def _checked_swap(self, t1, t2):
        # [FUNCTION DOC] HELPER: Swap two tensors with torch.utils.swap_tensors and check object
        # identity and reference counts are unchanged, the attribute dictionaries and slots
        # moved to the other tensor, and that in-place operations still return the same Python
        # object afterward.
        t1_pres, t1_moved = self._get_tensor_prop(t1)
        t2_pres, t2_moved = self._get_tensor_prop(t2)

        torch.utils.swap_tensors(t1, t2)

        new_t1_pres, new_t1_moved = self._get_tensor_prop(t1)
        new_t2_pres, new_t2_moved = self._get_tensor_prop(t2)
        self.assertEqual(t1_pres, new_t1_pres)
        self.assertEqual(t2_pres, new_t2_pres)
        self.assertEqual(t1_moved, new_t2_moved)
        self.assertEqual(t2_moved, new_t1_moved)

        # tests that PyObject slots on TensorImpl are correctly swapped by
        # checking that when the function applied on a swapped tensor is
        # returns doesn't change the TensorImpl, the returned value (which is
        # given by returning the reference to the PyObject in the TensorImpl's
        # PyObjectSlot) is still correct
        self.assertEqual(id(t1.fill_(0.5)), id(t1))
        self.assertEqual(id(t2.fill_(0.5)), id(t2))

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo adds weakrefs")
    def test_swap_basic(self):
        # [FUNCTION DOC] Checks swap_tensors for plain, matrix, int and TwoTensor tensors in all
        # pairs: custom attributes move, holders keep the same object, grad accumulation nodes of
        # swapped tensors are poisoned (RuntimeError on backward), and swapping tensors with a
        # weak reference raises RuntimeError. Skipped under Dynamo.
        ts = [
            torch.rand(2),
            torch.rand(3, 3),
            torch.empty(3, dtype=torch.int),
            TwoTensor(torch.rand(4), torch.rand(4))
        ]

        for t1, t2 in itertools.combinations(ts, 2):
            t1 = t1.clone()
            t2 = t2.clone()
            t2.foo = "bar"
            holder = []
            holder.append(t1)

            self._checked_swap(t1, t2)

            self.assertIs(holder[0], t1)
            self.assertEqual(t1.foo, "bar")

            if t1.is_floating_point():
                t3 = t1.detach().clone().requires_grad_(True)
                out = t3 * 2
                torch.utils.swap_tensors(t3, t2)
                with self.assertRaisesRegex(RuntimeError, "AccumulateGrad node that was poisoned by swap_tensors"):
                    out.sum().backward()

            _wr = weakref.ref(t1)
            with self.assertRaisesRegex(RuntimeError, "has weakref"):
                torch.utils.swap_tensors(t1, t2)


    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo adds weakrefs")
    def test_swap_fail_slots(self):
        # [FUNCTION DOC] Checks swap_tensors with TwoTensor subclasses declaring different slot
        # sets: swapping a plain tensor or TwoTensor with a slotted subclass, or subclasses with
        # different slot names, raises RuntimeError saying different slots are not allowed;
        # swapping subclasses with the same slot set (even in different order) works, and extra
        # attributes set on instances move with the swap. Skipped under Dynamo.
        class MyTwoTensor(TwoTensor):
            # [CLASS DOC] TwoTensor subclass declaring slots a and b.
            __slots__ = ("a", "b")

        class MyTwoTensor2(TwoTensor):
            # [CLASS DOC] TwoTensor subclass declaring the same slots as the previous class in the
            # opposite order.
            __slots__ = ("b", "a")

        class MyTwoTensor3(TwoTensor):
            # [CLASS DOC] TwoTensor subclass declaring four slots: a, b, c and d.
            __slots__ = ("a", "b", "c", "d")

        class MyTwoTensor4(TwoTensor):
            # [CLASS DOC] TwoTensor subclass declaring slots a and c.
            __slots__ = ("a", "c")


        t1 = torch.rand(4)
        t2 = TwoTensor(torch.rand(4), torch.rand(4))
        t3 = MyTwoTensor(torch.rand(4), torch.rand(4))
        t4 = MyTwoTensor(torch.rand(4), torch.rand(4))
        t5 = MyTwoTensor2(torch.rand(4), torch.rand(4))
        t6 = MyTwoTensor3(torch.rand(4), torch.rand(4))
        t7 = MyTwoTensor3(torch.rand(4), torch.rand(4))
        t8 = MyTwoTensor4(torch.rand(4), torch.rand(4))

        self._checked_swap(t1, t2)
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t1, t3)
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t2, t3)
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t2, t8)
        self._checked_swap(t3, t4)
        self._checked_swap(t3, t5)
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t3, t6)
        t3.c = "foo"
        t4.d = "bar"
        self._checked_swap(t3, t4)
        self.assertEqual(t4.c, "foo")
        self.assertEqual(t3.d, "bar")
        t6.c = "cat"
        t7.d = "dog"
        self._checked_swap(t6, t7)

    @unittest.skipIf(torch.cuda.is_available(), "Test specific for CPU")
    def test_bf16_supported_on_cpu(self):
        # [FUNCTION DOC] CPU-only build check: asserts torch.cuda.is_bf16_supported is False when
        # CUDA is not available.
        self.assertFalse(torch.cuda.is_bf16_supported())

    def test_tensor_with_grad_to_scalar_warning(self) -> None:
        # [FUNCTION DOC] With always-warn enabled, converts a tensor requiring grad to a Python
        # float through math.pow and checks exactly one UserWarning is raised mentioning that
        # converting a tensor with requires_grad to a scalar may lead to unexpected behavior.
        with (warnings.catch_warnings(record=True) as w,
                set_warn_always_context(True)):
            warnings.simplefilter("always")

            x = torch.tensor(2.0, requires_grad=True)
            math.pow(x, 3)  # calling this results in a warning

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn(
                "Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.",
                str(w[0].message)
            )

    def test_tensor_item_no_warning(self):
        # [FUNCTION DOC] Checks that comparing a grad-requiring tensor through max and calling
        # item raises no warnings even with always-warn enabled.
        with (warnings.catch_warnings(record=True) as w,
                set_warn_always_context(True)):
            warnings.simplefilter("always")

            x = torch.tensor(2.0, requires_grad=True)
            max(x, 3)  # No warning
            x.item()  # No warning

            self.assertEqual(len(w), 0)

# The following block extends TestTorch with negative dim wrapping tests
# FIXME: replace these with OpInfo sample inputs or systemic OpInfo tests
# Functions to test negative dimension wrapping
METHOD = 1
INPLACE_METHOD = 2
FUNCTIONAL = 4
DIM_ARG: None = None

def make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim=0):
    # [FUNCTION DOC] Factory for negative-dimension tests. Given an op name, the tensor shape (or
    # list of shapes), a function building the argument list (with placeholders for dimension
    # arguments), the call styles to test (method, in-place method, functional) and an optional
    # extra dimension count, returns a test function that builds the inputs and compares results
    # using each non-negative dimension combination against the equivalent negative dimensions.
    # Raises AssertionError inside the test if method styles are used with a list of tensors.
    def neg_dim_test(self):
        # [FUNCTION DOC] Generated test body: for each combination of dimension values, compares
        # results of the op with positive and negative dimension arguments through each requested
        # call style (method, in-place on a clone, torch function) using equality assertions.
        if isinstance(tensor_arg, list):
            if METHOD in types or INPLACE_METHOD in types:
                raise AssertionError("METHOD and INPLACE_METHOD should not be in types for list tensor_arg")
            x = [torch.randn(arg) for arg in tensor_arg]
            ndim = len(tensor_arg[-1])
        else:
            x = torch.randn(*tensor_arg)
            ndim = len(tensor_arg)
        ndim += extra_dim

        n_dim_to_test = sum(e is DIM_ARG for e in arg_constr())

        for dims_val in combinations(range(ndim), n_dim_to_test):
            arg = arg_constr()
            arg_neg = copy.deepcopy(arg)
            idx = 0
            for i, v in enumerate(arg):
                if v is DIM_ARG:
                    arg[i] = dims_val[idx]
                    arg_neg[i] = dims_val[idx] - ndim
                    idx += 1

            if METHOD in types:
                a = getattr(x, name)(*arg)
                b = getattr(x, name)(*arg_neg)
                self.assertEqual(a, b)

            if INPLACE_METHOD in types:
                a = x.clone()
                getattr(a, name + '_')(*arg)
                b = x.clone()
                getattr(b, name + '_')(*arg_neg)
                self.assertEqual(a, b)

            if FUNCTIONAL in types:
                a = getattr(torch, name)(x, *arg)
                b = getattr(torch, name)(x, *arg_neg)
                self.assertEqual(a, b)

    return neg_dim_test

def idx_tensor(size, max_val):
    # [FUNCTION DOC] Return a long tensor of the given size filled with random integers below
    # max_val minus one, used as index arguments.
    return torch.LongTensor(*size).random_(0, max_val - 1)

def add_neg_dim_tests():
    # [FUNCTION DOC] Builds the declarative table of ops (name, input shape, argument builder,
    # call styles, optional extra dimension) and for each entry creates a negative-dimension test
    # method named after the op and attaches it to TestTorch; raises AssertionError if a generated
    # test name already exists.
    neg_dim_tests = [
        ('narrow', (10, 20, 30), lambda: [DIM_ARG, 0, 5], [METHOD]),
        ('transpose', (10, 20, 30), lambda: [DIM_ARG, DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        ('size', (10, 20, 30), lambda: [DIM_ARG], [METHOD]),
        ('cat', [(2, 3, 4), (2, 3, 4)], lambda: [DIM_ARG], [FUNCTIONAL]),
        ('chunk', (10, 20, 30), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        ('gather', (10, 20), lambda: [DIM_ARG, idx_tensor((10, 20), 10)], [METHOD, FUNCTIONAL]),
        ('index_select', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10)], [METHOD, FUNCTIONAL]),
        ('split', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        ('squeeze', (10, 1, 20, 1), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        ('unbind', (2, 3, 4), lambda: [DIM_ARG], [FUNCTIONAL]),
        ('unsqueeze', (10, 20), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL], 1),
        ('logcumsumexp', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cumprod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cumsum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cummax', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cummin', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('mean', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('median', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('nanmedian', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('mode', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('norm', (10, 20), lambda: [2, DIM_ARG], [METHOD, FUNCTIONAL]),
        ('prod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('std', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('sum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('var', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('kthvalue', (10, 20), lambda: [3, DIM_ARG], [METHOD, FUNCTIONAL]),
        ('max', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('min', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('sort', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('topk', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        ('renorm', (10, 20), lambda: [2, DIM_ARG, 1], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        ('index_add', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
        ('index_copy', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
        ('index_fill', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), 12], [INPLACE_METHOD]),
        ('scatter', (10, 10), lambda: [DIM_ARG, idx_tensor((10, 10), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
        ('select', (10, 20), lambda: [DIM_ARG, 3], [METHOD]),
        ('unfold', (10, 20), lambda: [DIM_ARG, 5, 2], [METHOD]),
    ]

    for decl in neg_dim_tests:
        if len(decl) == 4:
            name, tensor_arg, arg_constr, types = decl
            extra_dim = 0
        elif len(decl) == 5:
            name, tensor_arg, arg_constr, types, extra_dim = decl

        test_name = 'test_' + name + '_neg_dim'

        if hasattr(TestTorch, test_name):
            raise AssertionError("Duplicated test name: " + test_name)
        setattr(TestTorch, test_name, make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim))

# TODO: these empty classes are temporarily instantiated for XLA compatibility
#   once XLA updates their test suite it should be removed
class TestViewOps(TestCase):
    # [CLASS DOC] Empty test case kept only so the device-type instantiation creates view-op test
    # classes that external XLA test suites expect.
    pass

class TestTensorDeviceOps(TestCase):
    # [CLASS DOC] Empty test case kept only so the device-type instantiation creates device-op
    # test classes that external XLA test suites expect.
    pass

# Generates tests
# Note: test generation must be done at file scope, not within main, or
# pytest will fail.
add_neg_dim_tests()
instantiate_device_type_tests(TestViewOps, globals())
instantiate_device_type_tests(TestTensorDeviceOps, globals())
instantiate_device_type_tests(TestTorchDeviceType, globals())
instantiate_device_type_tests(TestDevicePrecision, globals(), except_for='cpu')

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
