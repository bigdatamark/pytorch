# Owner(s): ["module: scatter & gather ops"]
# [MODULE DOC] Tests for the private segment reduction operator torch._segment_reduce, which
# reduces contiguous segments of a data tensor along an axis (max, mean, min, sum, prod) given
# segment lengths or offsets. Defines the list of reduction names, a helper giving the empty
# segment default for each reduction, and a device-generic test case covering 1-D, empty, multi-D
# data, pytorch_scatter reference cases, gradients, and the unsafe flag. Instantiated per device
# type at import time.

from itertools import product
from functools import partial

import numpy as np
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
    gradcheck,
    parametrize,
)


reductions = ["max", "mean", "min", "sum", "prod"]


def get_default_value(initial_value, reduction):
    # [FUNCTION DOC] HELPER: Returns the value expected for an empty segment: initial_value if it
    # is not None, else the identity of the reduction (negative infinity for max, nan for mean,
    # positive infinity for min, zero for sum, one for prod). Returns None for an unknown
    # reduction.
    if initial_value is not None:
        return initial_value
    if reduction == "max":
        return -float("Inf")
    elif reduction == "mean":
        return float("nan")
    elif reduction == "min":
        return float("Inf")
    elif reduction == "sum":
        return 0.0
    elif reduction == "prod":
        return 1.0


class TestSegmentReductions(TestCase):
    # [CLASS DOC] Device-generic tests of torch._segment_reduce forward values, gradients and
    # error checking. The class attribute marks the tests as needing an accelerator-class
    # hardware.
    hw_classification = HardwareClassification.ACCELERATOR

    def _test_common(
        self,
        reduction,
        device,
        dtype,
        unsafe,
        axis,
        initial_value,
        data_arr,
        lengths_arr,
        expected_arr,
        expected_grad_arr,
        check_backward,
        lengths_dtype=torch.int,
    ):
        # [FUNCTION DOC] HELPER: Shared checker. Builds lengths and the matching offsets
        # (cumulative sum with a leading zero) from lengths_arr, builds a grad-requiring data
        # tensor and expected result and gradient tensors of dtype dtype, and runs
        # torch._segment_reduce with lengths and then with offsets, comparing with loose tolerance
        # and equal nan. When check_backward is true, sums the result, backpropagates and compares
        # the data gradient with expected_grad, then for float64 and other dtypes except half,
        # bfloat16 and float runs gradcheck on nan-free data.
        # KNOWN ISSUE: the return that skips backward when check_backward is false sits inside the
        # loop over the two modes, so in that case only the lengths mode is ever checked and the
        # offsets mode is skipped.
        lengths = torch.tensor(lengths_arr, device=device, dtype=lengths_dtype)
        # generate offsets from lengths
        zeros_shape = list(lengths.shape)
        zeros_shape[-1] = 1
        offsets = torch.cat((lengths.new_zeros(zeros_shape), lengths), -1).cumsum_(-1)

        data = torch.tensor(
            data_arr,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        expected_result = torch.tensor(expected_arr, device=device, dtype=dtype)
        expected_grad = torch.tensor(expected_grad_arr, device=device, dtype=dtype)
        for mode in ['lengths', 'offsets']:
            segment_reduce_kwargs = dict(
                axis=axis,
                unsafe=unsafe,
                initial=initial_value)
            if (mode == 'lengths'):
                segment_reduce_kwargs['lengths'] = lengths
            else:
                segment_reduce_kwargs['offsets'] = offsets
            actual_result = torch._segment_reduce(
                data=data,
                reduce=reduction,
                **segment_reduce_kwargs
            )
            self.assertEqual(
                expected_result, actual_result, rtol=1e-02, atol=1e-05, equal_nan=True
            )

            if not check_backward:
                return

            # Test backward
            actual_result.sum().backward()
            self.assertEqual(
                expected_grad, data.grad, rtol=1e-02, atol=1e-05, equal_nan=True
            )
            data = data.detach().clone().requires_grad_(True)

            # gradcheck does not work well with bfloat16 or fp16 cpu types
            # also there is small numerical difference with fp32
            if dtype not in [torch.half, torch.bfloat16, torch.float]:
                # gradcheck does not like "nan" input, setting to random 10
                d_non_nan = np.nan_to_num(data_arr, nan=10)
                new_data = torch.tensor(
                    # [10 if v == float("nan") else v for v in data],
                    d_non_nan,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                self.assertTrue(
                    gradcheck(
                        lambda x: torch._segment_reduce(
                            data=x,
                            reduce=reduction,
                            **segment_reduce_kwargs
                        ),
                        (new_data,),
                    )
                )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    def test_simple_1d(self, device, dtypes):
        # [FUNCTION DOC] Tests a one-dimensional data with a nan value and four segments
        # (including an empty one) for each reduction, initial value zero or none, both axes
        # (first and last) and both unsafe settings, with value dtypes half, bfloat16, float,
        # double and lengths dtypes int and int64. Table of expected values and gradients per
        # reduction; backward is checked only when an initial value is given. Delegates to the
        # shared checker.
        val_dtype, length_type = dtypes
        lengths = [1, 2, 3, 0]
        data = [1, float("nan"), 3, 4, 5, 5]

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = initial is not None
                initial_value = initial
                default_value = get_default_value(initial_value, reduction)
                if reduction == "max":
                    expected_result = [1, float("nan"), 5, default_value]
                    expected_grad = [1, 1, 0, 0, 0.5, 0.5]
                elif reduction == "mean":
                    expected_result = [1, float("nan"), 4.666, default_value]
                    expected_grad = [1.0, 0.5, 0.5, 0.333, 0.333, 0.333]
                elif reduction == "min":
                    if initial is not None:
                        initial_value = 1000  # some high number
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [1, float("nan"), 4, default_value]
                    expected_grad = [1.0, 1.0, 0, 1, 0, 0]
                elif reduction == "sum":
                    expected_result = [1, float("nan"), 14, default_value]
                    expected_grad = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                elif reduction == "prod":
                    if initial is not None:
                        initial_value = 2  # 0 initial_value will zero out everything for prod
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [2, float("nan"), 200, default_value]
                        expected_grad = [2.0, 6.0, float("nan"), 50.0, 40.0, 40.0]
                    else:
                        expected_result = [1, float("nan"), 100, default_value]
                        expected_grad = [1.0, 3.0, float("nan"), 25.0, 20.0, 20.0]
                for axis in [0, -1]:
                    for unsafe in [True, False]:
                        self._test_common(
                            reduction,
                            device,
                            val_dtype,
                            unsafe,
                            axis,
                            initial_value,
                            data,
                            lengths,
                            expected_result,
                            expected_grad,
                            check_backward,
                            length_type,
                        )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    def test_simple_zero_length(self, device, dtypes):
        # [FUNCTION DOC] Tests two empty segments over empty data for each reduction, with and
        # without an initial value, both unsafe settings; expects every segment to equal the
        # reduction default and an empty gradient.
        val_dtype, length_type = dtypes
        lengths = [0, 0]
        data = torch.ones(0)

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = initial is not None
                initial_value = initial
                default_value = get_default_value(initial_value, reduction)
                if reduction == "max":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "mean":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "min":
                    if initial is not None:
                        initial_value = 1000  # some high number
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "sum":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "prod":
                    if initial is not None:
                        initial_value = 2  # 0 initial_value will zero out everything for prod
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [default_value, default_value]
                        expected_grad = []
                    else:
                        expected_result = [default_value, default_value]
                        expected_grad = []
                for axis in [0]:
                    for unsafe in [True, False]:
                        self._test_common(
                            reduction,
                            device,
                            val_dtype,
                            unsafe,
                            axis,
                            initial_value,
                            data,
                            lengths,
                            expected_result,
                            expected_grad,
                            check_backward,
                            length_type,
                        )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    def test_multi_d_simple(self, device, dtypes):
        # [FUNCTION DOC] Tests a two-dimensional data (segments along axis zero, a nan in some
        # columns) for every reduction with and without an initial value, and both unsafe
        # settings; expected results and gradients are listed per reduction. The lengths dtype is
        # ignored, so the shared checker uses its default integer lengths.
        val_dtype, _ = dtypes
        axis = 0
        lengths = [1, 2, 3, 0]
        data = [[1, 1], [float("nan"), 1], [3, float("nan")], [4, 1], [3, 2], [2, 3]]

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = initial is not None
                initial_value = initial
                default_value = get_default_value(initial_value, reduction)
                if reduction == "max":
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [4, 3],
                        [default_value, default_value],
                    ]
                    expected_grad = [
                        [1, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 0],
                        [0, 1],
                    ]
                elif reduction == "mean":
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [3, 2],
                        [default_value, default_value],
                    ]
                    expected_grad = [
                        [1.0, 1.0],
                        [0.5, 0.5],
                        [0.5, 0.5],
                        [0.333, 0.333],
                        [0.333, 0.333],
                        [0.333, 0.333],
                    ]
                elif reduction == "min":
                    if initial is not None:
                        initial_value = 1000  # some high number
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [2, 1],
                        [default_value, default_value],
                    ]
                    expected_grad = [
                        [1.0, 1.0],
                        [1, 0],
                        [0, 1],
                        [0, 1],
                        [0, 0],
                        [1, 0],
                    ]
                elif reduction == "sum":
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [9, 6],
                        [default_value, default_value],
                    ]
                    expected_grad = [
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                    ]
                elif reduction == "prod":
                    if initial is not None:
                        initial_value = 2  # 0 initial_value will zero out everything for prod
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [
                            [2, 2],
                            [float("nan"), float("nan")],
                            [48, 12],
                            [default_value, default_value],
                        ]
                        expected_grad = [
                            [2.0, 2.0],
                            [6.0, float("nan")],
                            [float("nan"), 2.0],
                            [12.0, 12.0],
                            [16.0, 6.0],
                            [24.0, 4.0],
                        ]
                    else:
                        expected_result = [
                            [1, 1],
                            [float("nan"), float("nan")],
                            [24, 6],
                            [default_value, default_value],
                        ]
                        expected_grad = [
                            [1.0, 1.0],
                            [3.0, float("nan")],
                            [float("nan"), 1.0],
                            [6.0, 6.0],
                            [8.0, 3.0],
                            [12.0, 2.0],
                        ]
                for unsafe in [True, False]:
                    self._test_common(
                        reduction,
                        device,
                        val_dtype,
                        unsafe,
                        axis,
                        initial_value,
                        data,
                        lengths,
                        expected_result,
                        expected_grad,
                        check_backward,
                    )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    @parametrize("reduce", ['sum', 'prod', 'min', 'max', 'mean'])
    def test_pytorch_scatter_test_cases(self, device, dtypes, reduce):
        # [FUNCTION DOC] Parametrized over the reduction. Runs a set of reference cases taken from
        # pytorch_scatter (1-D to 3-D data with index-pointer arrays of segment boundaries)
        # through torch._segment_reduce using both lengths (derived by differencing the pointers)
        # and offsets and compares with expected results; empty segments are filled with the
        # reduction identity. For float64 also tries gradcheck with a nested function with chosen
        # initial values.
        # KNOWN ISSUE: in the nested function the positional arguments are built as a set holding
        # the data and the reduction name, whose order is undefined, and gradcheck receives the
        # data as a bare tensor rather than a tuple, so the gradcheck part is unreliable.
        val_dtype, length_dtype = dtypes
        # zero-length segments are filled with reduction inits contrary to pytorch_scatter.
        tests = [
            {
                'src': [1, 2, 3, 4, 5, 6],
                'index': [0, 0, 1, 1, 1, 3],
                'indptr': [0, 2, 5, 5, 6],
                'sum': [3, 12, 0, 6],
                'prod': [2, 60, 1, 6],
                'mean': [1.5, 4, float('nan'), 6],
                'min': [1, 3, float('inf'), 6],
                'max': [2, 5, -float('inf'), 6],
            },
            {
                'src': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
                'index': [0, 0, 1, 1, 1, 3],
                'indptr': [0, 2, 5, 5, 6],
                'sum': [[4, 6], [21, 24], [0, 0], [11, 12]],
                'prod': [[3, 8], [315, 480], [1, 1], [11, 12]],
                'mean': [[2, 3], [7, 8], [float('nan'), float('nan')], [11, 12]],
                'min': [[1, 2], [5, 6], [float('inf'), float('inf')], [11, 12]],
                'max': [[3, 4], [9, 10], [-float('inf'), -float('inf')], [11, 12]],
            },
            {
                'src': [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],
                'index': [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],
                'indptr': [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
                'sum': [[4, 21, 0, 11], [12, 18, 12, 0]],
                'prod': [[3, 315, 1, 11], [48, 80, 12, 1]],
                'mean': [[2, 7, float('nan'), 11], [4, 9, 12, float('nan')]],
                'min': [[1, 5, float('inf'), 11], [2, 8, 12, float('inf')]],
                'max': [[3, 9, -float('inf'), 11], [6, 10, 12, -float('inf')]],
            },
            {
                'src': [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
                'index': [[0, 0, 1], [0, 2, 2]],
                'indptr': [[0, 2, 3, 3], [0, 1, 1, 3]],
                'sum': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
                'prod': [[[3, 8], [5, 6], [1, 1]], [[7, 9], [1, 1], [120, 143]]],
                'mean': [[[2, 3], [5, 6], [float('nan'), float('nan')]],
                         [[7, 9], [float('nan'), float('nan')], [11, 12]]],
                'min': [[[1, 2], [5, 6], [float('inf'), float('inf')]],
                        [[7, 9], [float('inf'), float('inf')], [10, 11]]],
                'max': [[[3, 4], [5, 6], [-float('inf'), -float('inf')]],
                        [[7, 9], [-float('inf'), -float('inf')], [12, 13]]],
            },
            {
                'src': [[1, 3], [2, 4]],
                'index': [[0, 0], [0, 0]],
                'indptr': [[0, 2], [0, 2]],
                'sum': [[4], [6]],
                'prod': [[3], [8]],
                'mean': [[2], [3]],
                'min': [[1], [2]],
                'max': [[3], [4]],
            },
            {
                'src': [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
                'index': [[0, 0], [0, 0]],
                'indptr': [[0, 2], [0, 2]],
                'sum': [[[4, 4]], [[6, 6]]],
                'prod': [[[3, 3]], [[8, 8]]],
                'mean': [[[2, 2]], [[3, 3]]],
                'min': [[[1, 1]], [[2, 2]]],
                'max': [[[3, 3]], [[4, 4]]],
            },
        ]
        for test in tests:
            data = torch.tensor(test['src'], dtype=val_dtype, device=device, requires_grad=True)
            indptr = torch.tensor(test['indptr'], dtype=length_dtype, device=device)
            dim = indptr.ndim - 1
            # calculate lengths from indptr
            lengths = torch.diff(indptr, dim=dim)
            expected = torch.tensor(test[reduce], dtype=val_dtype, device=device)

            actual_result = torch._segment_reduce(
                data=data,
                reduce=reduce,
                lengths=lengths,
                axis=dim,
                unsafe=True,
            )
            self.assertEqual(actual_result, expected)

            # test offsets
            actual_result = torch._segment_reduce(
                data=data,
                reduce=reduce,
                offsets=indptr,
                axis=dim,
                unsafe=True,
            )
            self.assertEqual(actual_result, expected)

            if val_dtype == torch.float64:
                def fn(x, mode='lengths'):
                    # [FUNCTION DOC] HELPER: Function under gradcheck. Picks an initial value
                    # (one, or a large positive or negative number for min and max so empty
                    # segments do not produce nan or infinity in the numerical Jacobian) and calls
                    # torch._segment_reduce on x with the lengths or offsets captured from the
                    # enclosing test, selected by mode. Returns the reduced tensor.
                    initial = 1
                    # supply initial values to prevent gradcheck from failing for 0 length segments
                    # where nan/inf are reduction identities that produce nans when calculating the numerical jacobian
                    if reduce == 'min':
                        initial = 1000
                    elif reduce == 'max':
                        initial = -1000
                    segment_reduce_args = {x, reduce}
                    segment_reduce_kwargs = dict(axis=dim, unsafe=True, initial=initial)
                    if mode == 'lengths':
                        segment_reduce_kwargs[mode] = lengths
                    elif mode == 'offsets':
                        segment_reduce_kwargs[mode] = indptr
                    return torch._segment_reduce(*segment_reduce_args, **segment_reduce_kwargs)
                self.assertTrue(gradcheck(partial(fn, mode='lengths'), (data.detach().clone().requires_grad_(True))))
                self.assertTrue(gradcheck(partial(fn, mode='offsets'), (data.detach().clone().requires_grad_(True))))


    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    def test_multi_d(self, device, dtypes):
        # [FUNCTION DOC] Tests three-dimensional data (5 by 2 by 5) with segments along axis zero
        # including empty segments at both ends, for every reduction. Expected results are
        # computed with numpy over the data slices and the reduction initial value fills empty
        # segments; both unsafe settings. Gradients are not checked.
        val_dtype, _ = dtypes
        axis = 0
        lengths = [0, 2, 3, 0]
        data = np.arange(50).reshape(5, 2, 5).tolist()
        expected_grad = []

        # TODO: calculate grad and check correctness
        check_backward = False

        for reduction in reductions:
            initial_value = 0
            if reduction == "max":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.max(data[:2], axis=0).tolist(),
                    np.max(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "mean":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.mean(data[:2], axis=0).tolist(),
                    np.mean(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "min":
                initial_value = 1000  # some high number
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.min(data[:2], axis=0).tolist(),
                    np.min(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "sum":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.sum(data[:2], axis=0).tolist(),
                    np.sum(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "prod":
                initial_value = 1
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.prod(data[:2], axis=0).tolist(),
                    np.prod(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            for unsafe in [True, False]:
                self._test_common(
                    reduction,
                    device,
                    val_dtype,
                    unsafe,
                    axis,
                    initial_value,
                    data,
                    lengths,
                    expected_result,
                    expected_grad,
                    check_backward,
                )

    @dtypes(torch.int, torch.int64)
    def test_unsafe_flag(self, device, dtype):
        # [FUNCTION DOC] Checks that with unsafe false and lengths that do not sum to the data
        # size, torch._segment_reduce raises a RuntimeError mentioning the expected row length
        # sums, for one-dimensional lengths and for a two-dimensional case along the last axis.
        length_type = dtype
        lengths = torch.tensor([0, 2, 3, 0], device=device, dtype=length_type)
        data = torch.arange(6, dtype=torch.float, device=device)

        # test for error on 1-D lengths
        with self.assertRaisesRegex(RuntimeError, "Expected all rows of lengths along axis"):
            torch._segment_reduce(data, 'sum', lengths=lengths, axis=0, unsafe=False)

        # test for error on multi-D lengths
        nd_lengths = torch.tensor([[0, 3, 3, 0], [2, 3, 0, 0]], dtype=length_type, device=device)
        nd_data = torch.arange(12, dtype=torch.float, device=device).reshape(2, 6)
        with self.assertRaisesRegex(RuntimeError, "Expected all rows of lengths along axis"):
            torch._segment_reduce(nd_data, 'sum', lengths=nd_lengths, axis=1, unsafe=False)




instantiate_device_type_tests(TestSegmentReductions, globals())

if __name__ == "__main__":
    run_tests()
