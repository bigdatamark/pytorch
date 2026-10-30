# mypy: ignore-errors

# Owner(s): ["module: numpy"]
# [MODULE DOC] Tests for interoperability between torch tensors and NumPy: tensor to
# array conversion (numpy method, force argument, zero tensors), array to tensor
# creation (from_numpy and the tensor constructor), NumPy scalars as indices and
# operands, the array protocol methods on legacy tensor types, comparisons with
# arrays, and a few dynamo compile checks. Defines one TestCase, TestNumPyInterop,
# instantiated per device type at import time by instantiate_device_type_tests.

import sys
from itertools import product
from unittest import skipIf

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    skipMeta,
)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


# For testing handling NumPy objects and sending tensors to / accepting
#   arrays from NumPy.
class TestNumPyInterop(TestCase):
    # [CLASS DOC] Device-generic test case for NumPy interop. Stateless; each test is
    # independent. Many tests are restricted to cpu with the onlyCPU decorator because
    # numpy only shares memory with cpu tensors.
    # Note: the warning this tests for only appears once per program, so
    # other instances of this warning should be addressed to avoid
    # the tests depending on the order in which they're run.
    @onlyCPU
    def test_numpy_non_writeable(self, device):
        # [FUNCTION DOC] Checks that creating a tensor from a read-only numpy array with
        # from_numpy emits a UserWarning. The warning is only emitted once per process,
        # so other code should avoid triggering it to keep this test order-independent.
        arr = np.zeros(5)
        arr.flags["WRITEABLE"] = False
        self.assertWarns(UserWarning, lambda: torch.from_numpy(arr))

    @onlyCPU
    @skipIf(
        sys.version_info[:2] == (3, 14)
        and np.lib.NumpyVersion(np.__version__) < "2.4.0",
        "Broken in older numpy versions, see https://github.com/numpy/numpy/issues/30265",
    )
    def test_numpy_unresizable(self, device) -> None:
        # [FUNCTION DOC] Checks storage sharing prevents resizing. An array backing a
        # tensor cannot be resized (ValueError), a tensor that is viewed as an array
        # cannot be resized in place (RuntimeError), and that array cannot be resized
        # either (ValueError). Skipped on Python 3.14 with numpy older than 2.4.
        x = np.zeros((2, 2))
        y = torch.from_numpy(x)  # noqa: F841
        with self.assertRaises(ValueError):
            x.resize((5, 5))

        z = torch.randn(5, 5)
        w = z.numpy()
        with self.assertRaises(RuntimeError):
            z.resize_(10, 10)
        with self.assertRaises(ValueError):
            w.resize((10, 10))

    @onlyCPU
    def test_to_numpy(self, device) -> None:
        # [FUNCTION DOC] Checks converting tensors to arrays for many dtypes: 1D, with
        # storage offset, empty, 2D contiguous (array is C contiguous), 2D with offset,
        # transposed non-contiguous (array is not C contiguous), and transposed with
        # holes. Elements must equal those of the tensor. For non-half dtypes also checks
        # the array is writeable and writes through it are visible in the tensor, both for
        # a tensor and for its transpose.
        def get_castable_tensor(shape, dtype):
            # [FUNCTION DOC] HELPER: Make a random tensor of the given shape and dtype whose
            # values are safely castable. Floating dtypes sample a double uniformly in the
            # range of the dtype clipped to plus or minus ten billion, then cast; integer
            # dtypes sample 64-bit integers in the dtype range clipped likewise, then cast.
            # Clipping avoids ranges wider than the sampling type (infinities, undefined behavior).
            if dtype.is_floating_point:
                dtype_info = torch.finfo(dtype)
                # can't directly use min and max, because for double, max - min
                # is greater than double range and sampling always gives inf.
                low = max(dtype_info.min, -1e10)
                high = min(dtype_info.max, 1e10)
                t = torch.empty(shape, dtype=torch.float64).uniform_(low, high)
            else:
                # can't directly use min and max, because for int64_t, max - min
                # is greater than int64_t range and triggers UB.
                low = max(torch.iinfo(dtype).min, int(-1e10))
                high = min(torch.iinfo(dtype).max, int(1e10))
                t = torch.empty(shape, dtype=torch.int64).random_(low, high)
            return t.to(dtype)

        dtypes = [
            torch.uint8,
            torch.int8,
            torch.short,
            torch.int,
            torch.half,
            torch.float,
            torch.double,
            torch.long,
        ]

        for dtp in dtypes:
            # 1D
            sz = 10
            x = get_castable_tensor(sz, dtp)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            # 1D > 0 storage offset
            xm = get_castable_tensor(sz * 2, dtp)
            x = xm.narrow(0, sz - 1, sz)
            self.assertTrue(x.storage_offset() > 0)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            def check2d(x, y):
                # [FUNCTION DOC] HELPER: Assert that two 2D objects have equal elements. Uses the
                # loop sizes sz1 and sz2 from the enclosing test scope (assigned before each call)
                # and compares element by element with the test's equality assertion.
                for i in range(sz1):
                    for j in range(sz2):
                        self.assertEqual(x[i][j], y[i][j])

            # empty
            x = torch.tensor([]).to(dtp)
            y = x.numpy()
            self.assertEqual(y.size, 0)

            # contiguous 2D
            sz1 = 3
            sz2 = 5
            x = get_castable_tensor((sz1, sz2), dtp)
            y = x.numpy()
            check2d(x, y)
            self.assertTrue(y.flags["C_CONTIGUOUS"])

            # with storage offset
            xm = get_castable_tensor((sz1 * 2, sz2), dtp)
            x = xm.narrow(0, sz1 - 1, sz1)
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)
            self.assertTrue(y.flags["C_CONTIGUOUS"])

            # non-contiguous 2D
            x = get_castable_tensor((sz2, sz1), dtp).t()
            y = x.numpy()
            check2d(x, y)
            self.assertFalse(y.flags["C_CONTIGUOUS"])

            # with storage offset
            xm = get_castable_tensor((sz2 * 2, sz1), dtp)
            x = xm.narrow(0, sz2 - 1, sz2).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            # non-contiguous 2D with holes
            xm = get_castable_tensor((sz2 * 2, sz1 * 2), dtp)
            x = xm.narrow(0, sz2 - 1, sz2).narrow(1, sz1 - 1, sz1).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            if dtp != torch.half:
                # check writeable
                x = get_castable_tensor((3, 4), dtp)
                y = x.numpy()
                self.assertTrue(y.flags.writeable)
                y[0][1] = 3
                self.assertTrue(x[0][1] == 3)
                y = x.t().numpy()
                self.assertTrue(y.flags.writeable)
                y[0][1] = 3
                self.assertTrue(x[0][1] == 3)

    def test_to_numpy_bool(self, device) -> None:
        # [FUNCTION DOC] Checks that boolean tensors (two elements and one element)
        # convert to arrays with the numpy bool dtype and equal elements.
        x = torch.tensor([True, False], dtype=torch.bool)
        self.assertEqual(x.dtype, torch.bool)

        y = x.numpy()
        self.assertEqual(y.dtype, np.bool_)
        for i in range(len(x)):
            self.assertEqual(x[i], y[i])

        x = torch.tensor([True], dtype=torch.bool)
        self.assertEqual(x.dtype, torch.bool)

        y = x.numpy()
        self.assertEqual(y.dtype, np.bool_)
        self.assertEqual(x[0], y[0])

    @skipIfTorchDynamo(
        "can't check if value is ZeroTensor since _is_zerotensor returns a bool and not a TensorVariable"
    )
    def test_to_numpy_zero_tensor(self, device) -> None:
        # [FUNCTION DOC] Checks, for several dtypes, that an efficient zero tensor cannot be
        # converted by default (RuntimeError) but converting with force produces an array of
        # zeros. Skipped under dynamo.
        dtypes = [
            torch.uint8,
            torch.int8,
            torch.short,
            torch.int,
            torch.half,
            torch.float,
            torch.double,
            torch.long,
            torch.bool,
        ]
        for dtype in dtypes:
            x = torch._efficientzerotensor((10), dtype=dtype)
            self.assertRaises(RuntimeError, lambda: x.numpy())
            y = x.numpy(force=True)
            for i in range(10):
                self.assertEqual(y[i], 0)

    @skipIfTorchDynamo("conj bit not implemented in TensorVariable yet")
    def test_to_numpy_force_argument(self, device) -> None:
        # [FUNCTION DOC] Checks the force argument of conversion to numpy on a complex tensor
        # over combinations of force, requires_grad, sparse and conjugate views (sparse with
        # requires_grad is skipped). Without force, cases that need it (grad, sparse, conj
        # or non-cpu device) must raise RuntimeError or TypeError with a hint message, also
        # when force is passed as false. With force, sparse still raises TypeError. In all
        # other cases the result must equal the resolved-conjugate tensor. Skipped under dynamo.
        for force in [False, True]:
            for requires_grad in [False, True]:
                for sparse in [False, True]:
                    for conj in [False, True]:
                        data = [[1 + 2j, -2 + 3j], [-1 - 2j, 3 - 2j]]
                        x = torch.tensor(
                            data, requires_grad=requires_grad, device=device
                        )
                        y = x
                        if sparse:
                            if requires_grad:
                                continue
                            x = x.to_sparse()
                        if conj:
                            x = x.conj()
                            y = x.resolve_conj()
                        expect_error = (
                            requires_grad or sparse or conj or device != "cpu"
                        )
                        error_msg = r"Use (t|T)ensor\..*(\.numpy\(\))?"
                        if not force and expect_error:
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError), error_msg, lambda: x.numpy()
                            )
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError),
                                error_msg,
                                lambda: x.numpy(force=False),
                            )
                        elif force and sparse:
                            self.assertRaisesRegex(
                                TypeError, error_msg, lambda: x.numpy(force=True)
                            )
                        else:
                            self.assertEqual(x.numpy(force=force), y)

    def test_from_numpy(self, device) -> None:
        # [FUNCTION DOC] Checks from_numpy: elements are preserved for many dtypes
        # (including a remainder-of-two array as a Windows regression), an array with
        # unsupported string dtype raises TypeError, and storage offsets, transposed
        # arrays, arrays with holes and zero-sized shapes convert to equal tensors. An
        # array with a stride that does not fit its dtype raises ValueError.
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.complex64,
            np.complex128,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            np.longlong,
            np.bool_,
        ]
        complex_dtypes = [
            np.complex64,
            np.complex128,
        ]

        for dtype in dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)
            tensor_from_array = torch.from_numpy(array)
            # TODO: change to tensor equality check once HalfTensor
            # implements `==`
            for i in range(len(array)):
                self.assertEqual(tensor_from_array[i], array[i])
            # ufunc 'remainder' not supported for complex dtypes
            if dtype not in complex_dtypes:
                # This is a special test case for Windows
                # https://github.com/pytorch/pytorch/issues/22615
                array2 = array % 2
                tensor_from_array2 = torch.from_numpy(array2)
                for i in range(len(array2)):
                    self.assertEqual(tensor_from_array2[i], array2[i])

        # Test unsupported type
        array = np.array(["foo", "bar"], dtype=np.dtype(np.str_))
        with self.assertRaises(TypeError):
            tensor_from_array = torch.from_numpy(array)

        # check storage offset
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[1]
        expected = torch.arange(1, 126, dtype=torch.float64).view(5, 5, 5)[1]
        self.assertEqual(torch.from_numpy(x), expected)

        # check noncontiguous
        x = np.linspace(1, 25, 25)
        x.shape = (5, 5)
        expected = torch.arange(1, 26, dtype=torch.float64).view(5, 5).t()
        self.assertEqual(torch.from_numpy(x.T), expected)

        # check noncontiguous with holes
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[:, 1]
        expected = torch.arange(1, 126, dtype=torch.float64).view(5, 5, 5)[:, 1]
        self.assertEqual(torch.from_numpy(x), expected)

        # check zero dimensional
        x = np.zeros((0, 2))
        self.assertEqual(torch.from_numpy(x).shape, (0, 2))
        x = np.zeros((2, 0))
        self.assertEqual(torch.from_numpy(x).shape, (2, 0))

        # check ill-sized strides raise exception
        x = np.array([3.0, 5.0, 8.0])
        x.strides = (3,)
        self.assertRaises(ValueError, lambda: torch.from_numpy(x))

    @skipIfTorchDynamo("No need to test invalid dtypes that should fail by design.")
    def test_from_numpy_no_leak_on_invalid_dtype(self):
        # [FUNCTION DOC] Regression test for a reference leak (issue 121138). Repeatedly
        # calls from_numpy on an unsupported bytes-string array, swallowing the TypeError,
        # and asserts the array reference count does not change. Skipped under dynamo.
        # This used to leak memory as the `from_numpy` call raised an exception and didn't decref the temporary
        # object. See https://github.com/pytorch/pytorch/issues/121138
        x = np.array(b"value")
        initial_refcount = sys.getrefcount(x)
        for _ in range(1000):
            try:
                torch.from_numpy(x)
            except TypeError:
                pass
        final_refcount = sys.getrefcount(x)
        self.assertEqual(
            final_refcount,
            initial_refcount,
            lambda msg: f"{msg}\nMemory leak detected: refcount increased from {initial_refcount} to {final_refcount}",
        )

    @skipIfTorchDynamo("No need to test invalid dtypes that should fail by design.")
    @onlyCPU
    def test_from_numpy_zero_element_type(self):
        # [FUNCTION DOC] Checks the dtype validity check runs before the stride check: from_numpy
        # on a zero-itemsize string array raises TypeError instead of dividing by zero.
        # cpu only; skipped under dynamo.
        # This tests that dtype check happens before strides check
        # which results in div-by-zero on-x86
        x = np.ndarray((3, 3), dtype=str)
        self.assertRaises(TypeError, lambda: torch.from_numpy(x))

    @skipMeta
    def test_from_list_of_ndarray_warning(self, device):
        # [FUNCTION DOC] Checks that building a tensor from a list of numpy arrays emits the
        # "extremely slow" UserWarning once. Not run for the meta device.
        warning_msg = (
            r"Creating a tensor from a list of numpy.ndarrays is extremely slow"
        )
        with self.assertWarnsOnceRegex(UserWarning, warning_msg):
            torch.tensor([np.array([0]), np.array([1])], device=device)

    def test_ctor_with_invalid_numpy_array_sequence(self, device):
        # [FUNCTION DOC] Checks the tensor constructor rejects ragged nested sequences of numpy
        # arrays with ValueError ("expected sequence of length") at several nesting levels
        # and in a mixed list of list and array, and raises TypeError ("not a sequence") when a
        # zero-dimensional array appears where a sequence is expected.
        # Invalid list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor(
                [np.random.random(size=(3, 3)), np.random.random(size=(3, 0))],
                device=device,
            )

        # Invalid list of list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor(
                [[np.random.random(size=(3, 3)), np.random.random(size=(3, 2))]],
                device=device,
            )

        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor(
                [
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 3))],
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 2))],
                ],
                device=device,
            )

        # expected shape is `[1, 2, 3]`, hence we try to iterate over 0-D array
        # leading to type error : not a sequence.
        with self.assertRaisesRegex(TypeError, "not a sequence"):
            torch.tensor(
                [[np.random.random(size=(3)), np.random.random()]], device=device
            )

        # list of list or numpy array.
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor([[1, 2, 3], np.random.random(size=(2,))], device=device)

    @onlyCPU
    def test_ctor_with_numpy_scalar_ctor(self, device) -> None:
        # [FUNCTION DOC] Checks that for several numpy scalar types, creating a tensor from the
        # scalar and taking its item gives the original value. cpu only.
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.uint8,
            np.bool_,
        ]
        for dtype in dtypes:
            self.assertEqual(dtype(42), torch.tensor(dtype(42)).item())

    @onlyCPU
    def test_numpy_index(self, device):
        # [FUNCTION DOC] Checks that numpy integer scalars (which are not Python ints) can
        # index a tensor and select the same row as the equivalent Python int. cpu only.
        i = np.array([0, 1, 2], dtype=np.int32)
        x = torch.randn(5, 5)
        for idx in i:
            self.assertFalse(isinstance(idx, int))
            self.assertEqual(x[idx], x[int(idx)])

    @onlyCPU
    def test_numpy_index_multi(self, device):
        # [FUNCTION DOC] Checks indexing a 3D tensor with a boolean mask made by comparing a
        # numpy integer array: the count of selected elements equals the sum of the array.
        for dim_sz in [2, 8, 16, 32]:
            i = np.zeros((dim_sz, dim_sz, dim_sz), dtype=np.int32)
            i[: dim_sz // 2, :, :] = 1
            x = torch.randn(dim_sz, dim_sz, dim_sz)
            self.assertTrue(x[i == 1].numel() == np.sum(i))

    @onlyCPU
    def test_numpy_array_interface(self, device):
        # [FUNCTION DOC] Checks the numpy array protocol on legacy tensor types. Converting
        # with asarray gives arrays of the matching dtype and values, also when a dtype is
        # requested. Applying numpy ufuncs (abs, sin, sqrt, ceil) returns the same legacy
        # tensor type with values equal to numpy results, and a comparison ufunc returns a
        # byte tensor. Unsigned dtypes use non-negative data. cpu only.
        types = [
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.HalfTensor,
            torch.LongTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.ByteTensor,
        ]
        dtypes = [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.uint8,
        ]
        for tp, dtype in zip(types, dtypes):
            # Only concrete class can be given where "Type[number[_64Bit]]" is expected
            if np.dtype(dtype).kind == "u":  # type: ignore[misc]
                # .type expects a XxxTensor, which have no type hints on
                # purpose, so ignore during mypy type checking
                x = torch.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
                array = np.array([1, 2, 3, 4], dtype=dtype)
            else:
                x = torch.tensor([1, -2, 3, -4]).type(tp)  # type: ignore[call-overload]
                array = np.array([1, -2, 3, -4], dtype=dtype)

            # Test __array__ w/o dtype argument
            asarray = np.asarray(x)
            self.assertIsInstance(asarray, np.ndarray)
            self.assertEqual(asarray.dtype, dtype)
            for i in range(len(x)):
                self.assertEqual(asarray[i], x[i])

            # Test __array_wrap__, same dtype
            abs_x = np.abs(x)
            abs_array = np.abs(array)
            self.assertIsInstance(abs_x, tp)
            for i in range(len(x)):
                self.assertEqual(abs_x[i], abs_array[i])

        # Test __array__ with dtype argument
        for dtype in dtypes:
            x = torch.IntTensor([1, -2, 3, -4])
            asarray = np.asarray(x, dtype=dtype)
            self.assertEqual(asarray.dtype, dtype)
            # Only concrete class can be given where "Type[number[_64Bit]]" is expected
            if np.dtype(dtype).kind == "u":  # type: ignore[misc]
                wrapped_x = np.array([1, -2, 3, -4]).astype(dtype)
                for i in range(len(x)):
                    self.assertEqual(asarray[i], wrapped_x[i])
            else:
                for i in range(len(x)):
                    self.assertEqual(asarray[i], x[i])

        # Test some math functions with float types
        float_types = [torch.DoubleTensor, torch.FloatTensor]
        float_dtypes = [np.float64, np.float32]
        for tp, dtype in zip(float_types, float_dtypes):
            x = torch.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
            array = np.array([1, 2, 3, 4], dtype=dtype)
            for func in ["sin", "sqrt", "ceil"]:
                ufunc = getattr(np, func)
                res_x = ufunc(x)
                res_array = ufunc(array)
                self.assertIsInstance(res_x, tp)
                for i in range(len(x)):
                    self.assertEqual(res_x[i], res_array[i])

        # Test functions with boolean return value
        for tp, dtype in zip(types, dtypes):
            x = torch.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
            array = np.array([1, 2, 3, 4], dtype=dtype)
            geq2_x = np.greater_equal(x, 2)
            geq2_array = np.greater_equal(array, 2).astype("uint8")
            self.assertIsInstance(geq2_x, torch.ByteTensor)
            for i in range(len(x)):
                self.assertEqual(geq2_x[i], geq2_array[i])

    @onlyCPU
    def test_multiplication_numpy_scalar(self, device) -> None:
        # [FUNCTION DOC] Checks multiplying a float or double tensor that requires grad by a
        # numpy scalar of various types, on either side: the result is a tensor that keeps
        # the tensor dtype and still requires grad. cpu only.
        for np_dtype in [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.int16,
            np.uint8,
        ]:
            for t_dtype in [torch.float, torch.double]:
                # mypy raises an error when np.floatXY(2.0) is called
                # even though this is valid code
                np_sc = np_dtype(2.0)  # type: ignore[abstract, arg-type]
                t = torch.ones(2, requires_grad=True, dtype=t_dtype)
                r1 = t * np_sc
                self.assertIsInstance(r1, torch.Tensor)
                self.assertTrue(r1.dtype == t_dtype)
                self.assertTrue(r1.requires_grad)
                r2 = np_sc * t
                self.assertIsInstance(r2, torch.Tensor)
                self.assertTrue(r2.dtype == t_dtype)
                self.assertTrue(r2.requires_grad)

    @onlyCPU
    @skipIfTorchDynamo()
    def test_parse_numpy_int_overflow(self, device):
        # [FUNCTION DOC] Checks that passing a numpy unsigned 64-bit scalar with a negative
        # (wrapped) value as an integer argument to mean fails: OverflowError for numpy 2
        # and later, ValueError for older versions. cpu only; skipped under dynamo.
        # assertRaises uses a try-except which dynamo has issues with
        # Only concrete class can be given where "Type[number[_64Bit]]" is expected
        if np.__version__ > "2":
            self.assertRaisesRegex(
                OverflowError,
                "out of bounds",
                lambda: torch.mean(torch.randn(1, 1), np.uint64(-1)),
            )  # type: ignore[call-overload]
        else:
            self.assertRaisesRegex(
                ValueError,
                "(Overflow|an integer is required)",
                lambda: torch.mean(torch.randn(1, 1), np.uint64(-1)),
            )  # type: ignore[call-overload]

    @onlyCPU
    def test_parse_numpy_int(self, device):
        # [FUNCTION DOC] Regression test for issue 29252: numpy integer scalars of several
        # sizes are accepted wherever an int is expected (native function arguments, custom
        # storage construction, item assignment) and keep the integer dtype of a tensor
        # when added to it as a Scalar in either order. cpu only.
        # https://github.com/pytorch/pytorch/issues/29252
        for nptype in [np.int16, np.int8, np.uint8, np.int32, np.int64]:
            scalar = 3
            np_arr = np.array([scalar], dtype=nptype)
            np_val = np_arr[0]

            # np integral type can be treated as a python int in native functions with
            # int parameters:
            self.assertEqual(torch.ones(5).diag(scalar), torch.ones(5).diag(np_val))
            self.assertEqual(
                torch.ones([2, 2, 2, 2]).mean(scalar),
                torch.ones([2, 2, 2, 2]).mean(np_val),
            )

            # numpy integral type parses like a python int in custom python bindings:
            self.assertEqual(torch.Storage(np_val).size(), scalar)  # type: ignore[attr-defined]

            tensor = torch.tensor([2], dtype=torch.int)
            tensor[0] = np_val
            self.assertEqual(tensor[0], np_val)

            # Original reported issue, np integral type parses to the correct
            # PyTorch integral type when passed for a `Scalar` parameter in
            # arithmetic operations:
            t = torch.from_numpy(np_arr)
            self.assertEqual((t + np_val).dtype, t.dtype)
            self.assertEqual((np_val + t).dtype, t.dtype)

    def test_has_storage_numpy(self, device):
        # [FUNCTION DOC] Checks tensors made from numpy arrays with various dtype arguments
        # all report a non-None storage.
        for dtype in [np.float32, np.float64, np.int64, np.int32, np.int16, np.uint8]:
            arr = np.array([1], dtype=dtype)
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.float32).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.double).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.int).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.long).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.uint8).storage()
            )

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_numpy_scalar_cmp(self, device, dtype):
        # [FUNCTION DOC] Checks equality between scalars taken from tensors and numpy arrays
        # of many dtypes (scalar, 1D, 2D; complex or real data). Every combination of a
        # tensor element or Python value with a numpy element or Python value must compare
        # equal; bfloat16 conversion to numpy must raise TypeError. For complex64 a tensor
        # compared with a numpy complex64 scalar must be reported unequal.
        # KNOWN ISSUE: that unequal result reflects a known bug where the imaginary part is
        # dropped (issue 43579); the test asserts the buggy behavior.
        if dtype.is_complex:
            tensors = (
                torch.tensor(complex(1, 3), dtype=dtype, device=device),
                torch.tensor([complex(1, 3), 0, 2j], dtype=dtype, device=device),
                torch.tensor(
                    [[complex(3, 1), 0], [-1j, 5]], dtype=dtype, device=device
                ),
            )
        else:
            tensors = (
                torch.tensor(3, dtype=dtype, device=device),
                torch.tensor([1, 0, -3], dtype=dtype, device=device),
                torch.tensor([[3, 0, -1], [3, 5, 4]], dtype=dtype, device=device),
            )

        for tensor in tensors:
            if dtype == torch.bfloat16:
                with self.assertRaises(TypeError):
                    np_array = tensor.cpu().numpy()
                continue

            np_array = tensor.cpu().numpy()
            for t, a in product(
                (tensor.flatten()[0], tensor.flatten()[0].item()),
                (np_array.flatten()[0], np_array.flatten()[0].item()),
            ):
                self.assertEqual(t, a)
                if (
                    dtype == torch.complex64
                    and torch.is_tensor(t)
                    and type(a) is np.complex64
                ):
                    # TODO: Imaginary part is dropped in this case. Need fix.
                    # https://github.com/pytorch/pytorch/issues/43579
                    self.assertFalse(t == a)
                else:
                    self.assertTrue(t == a)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    def test___eq__(self, device, dtype):
        # [FUNCTION DOC] Checks tensor to numpy array equality in both operand orders for
        # many dtypes: all elements equal, one element changed, random elements changed,
        # and all elements changed; the result must equal the expected boolean tensor
        # each time. cpu only.
        a = make_tensor((5, 7), dtype=dtype, device=device, low=-9, high=9)
        b = a.detach().clone()
        b_np = b.numpy()

        # Check all elements equal
        res_check = torch.ones_like(a, dtype=torch.bool)
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check one element unequal
        if dtype == torch.bool:
            b[1][3] = not b[1][3]
        else:
            b[1][3] += 1
        res_check[1][3] = False
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check random elements unequal
        rand = torch.randint(0, 2, a.shape, dtype=torch.bool)
        res_check = rand.logical_not()
        b.copy_(a)

        if dtype == torch.bool:
            b[rand] = b[rand].logical_not()
        else:
            b[rand] += 1

        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check all elements unequal
        if dtype == torch.bool:
            b.copy_(a.logical_not())
        else:
            b.copy_(a + 1)
        res_check.fill_(False)
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

    @onlyCPU
    def test_empty_tensors_interop(self, device):
        # [FUNCTION DOC] Regression test for issues 115068, 115066 and 113037: arithmetic
        # between a zero-dimensional half tensor and an empty tensor (true divide, mul, and
        # floor division) gives a result with the empty tensor's shape. cpu only.
        x = torch.rand((), dtype=torch.float16)
        y = torch.tensor(np.random.rand(0), dtype=torch.float16)
        # Same can be achieved by running
        # y = torch.empty_strided((0,), (0,), dtype=torch.float16)

        # Regression test for https://github.com/pytorch/pytorch/issues/115068
        self.assertEqual(torch.true_divide(x, y).shape, y.shape)
        # Regression test for https://github.com/pytorch/pytorch/issues/115066
        self.assertEqual(torch.mul(x, y).shape, y.shape)
        # Regression test for https://github.com/pytorch/pytorch/issues/113037
        self.assertEqual(torch.div(x, y, rounding_mode="floor").shape, y.shape)

    def test_ndarray_astype_object_graph_break(self):
        # [FUNCTION DOC] Checks that compiling, with the eager backend and full graph, a
        # function calling astype with the string O on a numpy array raises a dynamo
        # Unsupported error mentioning astype object.
        @torch.compile(backend="eager", fullgraph=True)
        def f(xs):
            # [FUNCTION DOC] HELPER: Function to compile; applies astype with the object type
            # code to its argument and returns nothing.
            xs.astype("O")

        xs = np.array([1, 2])
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "ndarray.astype\\(object\\)"
        ):
            f(xs)

    def test_ndarray_astype_object_graph_break_2(self):
        # [FUNCTION DOC] Same as the previous test but astype is given the Python object
        # type instead of a string; also expects the dynamo Unsupported error.
        @torch.compile(backend="eager", fullgraph=True)
        def f(xs):
            # [FUNCTION DOC] HELPER: Function to compile; applies astype with the object type
            # to its argument and returns nothing.
            xs.astype(object)

        xs = np.array([1, 2])
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "ndarray.astype\\(object\\)"
        ):
            f(xs)

    def test_copy_mode(self):
        # [FUNCTION DOC] Checks that a function creating a numpy array with the IF_NEEDED
        # copy mode compiles with the eager backend and full graph without error, and
        # the compiled result equals the uncompiled result.
        def f(x):
            # [FUNCTION DOC] HELPER: Function to compile; returns a numpy array built from its
            # argument using the if-needed copy mode.
            return np.array(x, copy=np._CopyMode.IF_NEEDED)

        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        x = np.array([1, 2, 3])
        # Should run without throwing an exception
        y = opt_f(x)
        self.assertEqual(y, f(x))


instantiate_device_type_tests(TestNumPyInterop, globals())

if __name__ == "__main__":
    run_tests()
