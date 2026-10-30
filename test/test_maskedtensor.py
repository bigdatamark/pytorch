# Owner(s): ["module: masked operators"]
# [MODULE DOC] Tests for torch.masked MaskedTensor. Defines comparison helpers, sample data
# generation, and test classes for construction and conversion (TestBasics), unary and binary
# ufuncs (TestUnary, TestBinary), reductions and gradient behavior (TestReductions), and
# OpInfo-driven equality checks against plain tensors for strided, sparse COO and sparse CSR
# layouts (TestOperators). At import time builds the op lists and instantiates the device-type
# classes for cpu and cuda and the parametrized classes.

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    make_tensor,
    parametrize,
    instantiate_parametrized_tests,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    SampleInput,
    binary_ufuncs,
    reduction_ops,
    unary_ufuncs,
)

from torch.masked import as_masked_tensor, masked_tensor, _combine_input_and_mask
from torch.masked.maskedtensor.core import _masks_match, _tensors_match
from torch.masked.maskedtensor.unary import NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS, UNARY_NAMES
from torch.masked.maskedtensor.binary import NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS, BINARY_NAMES
from torch.masked.maskedtensor.reductions import REDUCE_NAMES


def _compare_mt_t(mt_result, t_result, rtol=1e-05, atol=1e-05):
    # [FUNCTION DOC] HELPER: Checks a MaskedTensor result against a plain tensor result. Takes
    # the mask and data of the masked result (densifying sparse COO or CSR ones), zeroes
    # masked-out positions of detached views of both data and the tensor (this writes into the
    # shared storage of the arguments), and compares with the given tolerances. Raises
    # ValueError on mismatch.
    mask = mt_result.get_mask()
    mt_result_data = mt_result.get_data()
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()
    if mt_result_data.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_result_data = mt_result_data.to_dense()
    a = mt_result_data.detach().masked_fill_(~mask, 0)
    b = t_result.detach().masked_fill_(~mask, 0)
    if not _tensors_match(a, b, exact=False, rtol=rtol, atol=atol):
        raise ValueError("The data in MaskedTensor a and Tensor b do not match")

def _compare_mts(mt1, mt2, rtol=1e-05, atol=1e-08):
    # [FUNCTION DOC] HELPER: Checks two MaskedTensors for equality. Raises ValueError if data
    # layouts differ, masks do not match, or mask layouts differ. Densifies sparse data and
    # masks, zeroes masked-out positions of detached views (writing into shared storage), then
    # compares the data within tolerances and raises ValueError if they differ.
    mt_data1 = mt1.get_data()
    mt_data2 = mt2.get_data()
    if mt_data1.layout != mt_data2.layout:
        raise ValueError("mt1's data and mt2's data do not have the same layout. "
                         f"mt1.get_data().layout = {mt_data1.layout} while mt2.get_data().layout = {mt_data2.layout}")

    mask = mt1.get_mask()
    mask2 = mt2.get_mask()
    if not _masks_match(mt1, mt2):
        raise ValueError("mt1 and mt2 must have matching masks")
    if mask.layout != mask2.layout:
        raise ValueError("mt1's mask and mt2's mask do not have the same layout. "
                         f"mt1.get_mask().layout = {mask.layout} while mt2.get_mask().layout = {mask2.layout}")
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()

    if mt_data1.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_data1 = mt_data1.to_dense()
        mt_data2 = mt_data2.to_dense()
    a = mt_data1.detach().masked_fill_(~mask, 0)
    b = mt_data2.detach().masked_fill_(~mask, 0)

    if not _tensors_match(a, b, exact=False, rtol=rtol, atol=atol):
        raise ValueError("The data in MaskedTensor mt1 and MaskedTensor mt2 do not match")

def _compare_forward_backward(data, mask, fn):
    # [FUNCTION DOC] HELPER: Runs a function on a masked tensor with requires_grad and on a
    # plain tensor whose masked-out entries are set to negative infinity, backpropagates the
    # sum of each result, and compares both the results and the gradients (gradient tolerance
    # 1e-6) using the masked-versus-tensor comparison helper.
    mt = masked_tensor(data, mask, requires_grad=True)
    masked_res = fn(mt)
    masked_res.sum().backward()

    t = data.masked_fill(~mask, float("-inf")).detach().clone().requires_grad_()
    tensor_res = fn(t)
    tensor_res.sum().backward()

    _compare_mt_t(masked_res, tensor_res)
    _compare_mt_t(mt.grad, t.grad, atol=1e-06)

def _create_random_mask(shape, device):
    # [FUNCTION DOC] HELPER: Returns a random boolean mask of the given shape on the
    # given device.
    return make_tensor(shape, device=device, dtype=torch.bool)

def _generate_sample_data(
    device="cpu", dtype=torch.float, requires_grad=True, layout=torch.strided
):
    # [FUNCTION DOC] HELPER: Builds a list of SampleInput objects for shapes: scalar, 1D,
    # 2D and 4D, each with random data and a random mask given in the kwargs. For sparse
    # COO the mask is coalesced and the data restricted to it; for sparse CSR non-2D shapes
    # are skipped and the data is restricted to the mask. Raises AssertionError for any
    # other layout.
    if layout not in {
        torch.strided,
        torch.sparse_coo,
        torch.sparse_csr,
    }:
        raise AssertionError("Layout must be strided/sparse_coo/sparse_csr")
    shapes = [
        [],
        [2],
        [3, 5],
        [3, 2, 1, 2],
    ]
    inputs = []
    for s in shapes:
        data = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)  # type: ignore[arg-type]
        mask = _create_random_mask(s, device)
        if layout == torch.sparse_coo:
            mask = mask.to_sparse_coo().coalesce()
            data = data.sparse_mask(mask).requires_grad_(requires_grad)
        elif layout == torch.sparse_csr:
            if data.ndim != 2 and mask.ndim != 2:
                continue
            mask = mask.to_sparse_csr()
            data = data.sparse_mask(mask)
        inputs.append(SampleInput(data, kwargs={"mask": mask}))
    return inputs

def _fix_fn_name(fn_name):
    # [FUNCTION DOC] HELPER: Removes a single trailing underscore from an op name, so an in-
    # place variant maps to the name of its out-of-place op.
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    return fn_name


class TestBasics(TestCase):
    # [CLASS DOC] Device-generic tests for MaskedTensor construction validation (type,
    # layout, dim and size errors, gradient warning), arithmetic, where, unfold/stack, device
    # and dtype conversion, dense/sparse conversion with gradients, sparse index validation
    # and contiguity.
    def test_invalid_tensor_inputs(self, device):
        # [FUNCTION DOC] Checks that the constructor raises TypeError when data or mask is not
        # a plain tensor (a masked tensor or a number).
        data = torch.randn((3, 4), device=device)
        mask = _create_random_mask((3, 4), device=device)
        mt = masked_tensor(data, mask)

        with self.assertRaisesRegex(TypeError, "data must be a Tensor"):
            masked_tensor(mt, mask)
        with self.assertRaisesRegex(TypeError, "data must be a Tensor"):
            masked_tensor(0, mask)
        with self.assertRaisesRegex(TypeError, "mask must be a Tensor"):
            masked_tensor(data, mt)
        with self.assertRaisesRegex(TypeError, "mask must be a Tensor"):
            masked_tensor(data, 0)

    def test_diff_layouts(self, device):
        # [FUNCTION DOC] Checks that sparse COO data with a strided mask raises TypeError
        # about the layouts.
        data = torch.randn((3, 4), device=device).to_sparse_coo()
        mask = _create_random_mask((3, 4), device=device)
        with self.assertRaisesRegex(TypeError, "data and mask must have the same layout"):
            masked_tensor(data, mask)

    def test_diff_dim(self, device):
        # [FUNCTION DOC] Checks that data and mask with different number of dims
        # raise ValueError.
        data = torch.randn((3, 4, 5), device=device)
        mask = _create_random_mask((3, 4), device=device)
        with self.assertRaisesRegex(ValueError, "data.dim\\(\\) must equal mask.dim\\(\\)"):
            masked_tensor(data, mask)

    def test_diff_sizes(self, device):
        # [FUNCTION DOC] Checks that data and mask with different sizes raise ValueError.
        data = torch.randn((3, 4), device=device)
        mask = _create_random_mask((3, 3), device=device)
        with self.assertRaisesRegex(ValueError, "data.size\\(\\) must equal mask.size\\(\\)"):
            masked_tensor(data, mask)

    def test_grad_warning(self, device):
        # [FUNCTION DOC] Checks that building a MaskedTensor from data that requires grad
        # emits a UserWarning recommending against it.
        data = torch.randn((3, 4), device=device, requires_grad=True)
        mask = _create_random_mask((3, 4), device=device)
        msg = "It is not recommended to create a MaskedTensor with a tensor that requires_grad."
        with self.assertWarnsRegex(UserWarning, msg):
            masked_tensor(data, mask)

    def test_add(self, device):
        # [FUNCTION DOC] Checks that adding two masked tensors with different masks raises
        # ValueError, and that adding a masked tensor to itself doubles the data under the
        # same mask.
        data = torch.arange(5.0, device=device)
        mask = torch.tensor([True, True, False, True, False], device=device)
        m0 = masked_tensor(data, mask)
        m1 = masked_tensor(data, ~mask)
        with self.assertRaisesRegex(ValueError, "Input masks must match."):
            m0 + m1
        _compare_mts(m0 + m0, masked_tensor(torch.tensor([0., 2, 0, 6, 0], device=device), mask))

    def test_softmax(self, device):
        # [FUNCTION DOC] Compares softmax over the last dim on a masked tensor with softmax on
        # a tensor with masked-out entries set to negative infinity, forward and backward.
        data = torch.randn((3, 4), device=device) * 0.1
        mask = torch.tensor(
            [
                [True, True, True, False],
                [False, True, False, True],
                [True, True, False, False],
            ],
            device=device
        )

        _compare_forward_backward(data, mask, lambda t: torch.softmax(t, -1))

    def test_where(self, device):
        # [FUNCTION DOC] Checks where with a masked tensor of exp values and a masked tensor
        # of ones with the opposite mask, against plain tensors; compares results and
        # gradients of both inputs.
        data = torch.tensor([-10.0, -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], device=device)
        mask = data < 0

        mx = masked_tensor(data, mask, requires_grad=True)
        my = masked_tensor(torch.ones_like(data), ~mask, requires_grad=True)
        masked_res = torch.where(mask, torch.exp(mx), my)
        masked_res.sum().backward()

        x = data.detach().clone().requires_grad_()
        y = torch.ones_like(x, device=device, requires_grad=True)
        tensor_res = torch.where(mask, torch.exp(x), y)
        tensor_res.sum().backward()

        _compare_mt_t(masked_res, tensor_res)
        _compare_mt_t(mx.grad, x.grad)
        _compare_mt_t(my.grad, y.grad)

    def test_unfold(self, device):
        # [FUNCTION DOC] Compares the unfold method along dim 1 on masked and plain tensors,
        # forward and backward.
        data = torch.rand(5, 5, device=device)
        mask = torch.rand(5, 5, device=device) > 0.5
        _compare_forward_backward(data, mask, lambda t: t.unfold(1, 2, 2))

    def test_nn_unfold(self, device):
        # [FUNCTION DOC] Compares torch.nn.functional.unfold with a (2, 3) kernel on masked
        # and plain 4D tensors, forward and backward.
        data = torch.rand(2, 5, 3, 4, device=device)
        mask = torch.rand(2, 5, 3, 4, device=device) > 0.5
        _compare_forward_backward(data, mask, lambda t: torch.nn.functional.unfold(t, kernel_size=(2, 3)))

    def test_stack(self, device):
        # [FUNCTION DOC] Stacks three masked tensors and three plain copies of their data,
        # backpropagates the sums, and compares the stacked results and each input's gradient.
        masked_tensors = [
            masked_tensor(
                torch.rand(2, 5, 3, 4, device=device),
                torch.rand(2, 5, 3, 4, device=device) > 0.5,
                requires_grad=True,
            ) for _ in range(3)
        ]

        data_tensors = [mt.get_data().detach().clone().requires_grad_() for mt in masked_tensors]
        masked_res = torch.stack(masked_tensors)
        tensor_res = torch.stack(data_tensors)

        masked_res.sum().backward()
        tensor_res.sum().backward()
        _compare_mt_t(masked_res, tensor_res)
        for mt, t in zip(masked_tensors, data_tensors):
            _compare_mt_t(mt.grad, t.grad, atol=1e-06)

    def test_to_sparse(self, device):
        # [FUNCTION DOC] For each sample, converts a masked tensor to sparse and the data
        # to sparse then dense, backpropagates both and compares results and gradients with
        # the data.
        for sample in _generate_sample_data(device=device):
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data.detach().clone(), mask, requires_grad=True)

            sparse_mt = mt.to_sparse()
            data.to_sparse().to_dense().sum().backward()
            sparse_mt.to_dense().sum().backward()

            _compare_mt_t(sparse_mt, data)
            _compare_mt_t(mt.grad, data.grad)

    def test_to_device(self, device):
        # [FUNCTION DOC] For each sample, moves the masked tensor to another device
        # (accelerator if the test device is cpu and one exists, else cpu) and checks the
        # tensor, its mask and its data all are on that device type.
        for sample in _generate_sample_data(device=device):
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data, mask, requires_grad=True)

            if device == "cpu":
                new_device = torch.device(
                    torch.accelerator.current_accelerator().type
                    if torch.accelerator.is_available()
                    else "cpu"
                )
            else:
                new_device = torch.device("cpu")
            mt_device = mt.to(new_device)

            self.assertEqual(mt_device.device.type, new_device.type)
            self.assertEqual(mt_device.get_mask().device.type, new_device.type)
            self.assertEqual(mt_device.get_data().device.type, new_device.type)

    def test_to_dtype(self, device):
        # [FUNCTION DOC] For each sample, converts the masked tensor to the other of float32
        # and float64 and checks the tensor and its data have the new dtype while the mask
        # stays boolean.
        for sample in _generate_sample_data(device=device):
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data, mask, requires_grad=True)

            new_dtype = torch.float64 if data.dtype == torch.float32 else torch.float32
            mt_dtype = mt.to(new_dtype)

            self.assertEqual(mt_dtype.dtype, new_dtype)
            self.assertEqual(mt_dtype.get_mask().dtype, torch.bool)
            self.assertEqual(mt_dtype.get_data().dtype, new_dtype)

    def test_to_dense(self, device):
        # [FUNCTION DOC] For sparse COO and CSR samples, converts the masked tensor to dense,
        # backpropagates it and the densified data, and compares results and gradients (the
        # gradient is densified).
        samples = _generate_sample_data(
            device=device,
            layout=torch.sparse_coo
        ) + _generate_sample_data(device=device, layout=torch.sparse_csr)
        for sample in samples:
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data, mask, requires_grad=True)

            dense_data = data.to_dense().detach().clone().requires_grad_(True)
            dense_mt = mt.to_dense()
            dense_data.sum().backward()
            dense_mt.sum().backward()

            _compare_mt_t(dense_mt, dense_data)
            _compare_mt_t(mt.grad.to_dense(), dense_data.grad)

    def test_to_dense_and_sparse_coo(self, device):
        # [FUNCTION DOC] For each strided sample, compares converting a strided masked tensor
        # to sparse COO then dense against a sparse COO masked tensor made to dense, results
        # and gradients.
        for sample in _generate_sample_data(device=device, layout=torch.strided):
            data = sample.input
            mask = sample.kwargs["mask"]
            ms = mask.to_sparse_coo().coalesce()

            mt = masked_tensor(data, mask, requires_grad=True)
            mts = masked_tensor(data.sparse_mask(ms), ms, requires_grad=True)

            converted = mt.to_sparse().to_dense()
            converted.sum().backward()

            converted2 = mts.to_dense()
            converted2.sum().backward()

            _compare_mts(converted, converted2)
            _compare_mts(mt.grad, mts.grad.to_dense())

    def test_to_dense_and_sparse_csr(self, device):
        # [FUNCTION DOC] Same as the COO round-trip test but for sparse CSR; non-2D samples
        # are skipped.
        for sample in _generate_sample_data(device=device, layout=torch.strided):
            data = sample.input
            mask = sample.kwargs["mask"]
            if data.ndim != 2:
                continue
            ms = mask.to_sparse_csr()

            mt = masked_tensor(data, mask, requires_grad=True)
            mts = masked_tensor(data.sparse_mask(ms), ms, requires_grad=True)

            converted = mt.to_sparse_csr().to_dense()
            converted.sum().backward()

            converted2 = mts.to_dense()
            converted2.sum().backward()

            _compare_mts(converted, converted2)
            _compare_mts(mt.grad, mts.grad.to_dense())

    def test_invalid_sparse_layout(self, device):
        # [FUNCTION DOC] Checks that data and mask with sparse CSC layout raise TypeError
        # saying the layout is not supported.
        data = torch.randn((3, 4), device=device).to_sparse_csc()
        mask = _create_random_mask((3, 4), device=device).to_sparse_csc()
        with self.assertRaisesRegex(TypeError, "data layout of torch.sparse_csc is not supported"):
            masked_tensor(data, mask)

    def test_invalid_sparse_coo_values(self, device):
        # [FUNCTION DOC] Checks that sparse COO data and mask with different indices
        # raise ValueError.
        v = torch.tensor([3, 4, 5], dtype=torch.float32)
        i1 = torch.tensor([[0, 1, 1], [2, 0, 2]])
        i2 = torch.tensor([[0, 1, 1], [2, 1, 2]])

        t = torch.sparse_coo_tensor(i1, v, (2, 4), device=device)
        mask = torch.sparse_coo_tensor(i2, torch.tensor([True, True, True]), (2, 4), device=device)

        msg = "data and mask are both sparse COO tensors but do not have the same indices."
        with self.assertRaisesRegex(ValueError, msg):
            masked_tensor(t, mask)

    def test_invalid_sparse_csr_values(self, device):
        # [FUNCTION DOC] Checks that sparse CSR data and mask with different crow indices, or
        # with different col indices, raise ValueError.
        crow_indices1 = [0, 2, 3]
        crow_indices2 = [0, 1, 3]
        col_indices1 = [0, 1, 2]
        col_indices2 = [1, 2, 3]

        values = [2, 3, 4]
        mask_values = [True, True, True]

        t1 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices1, dtype=torch.int64),
            torch.tensor(col_indices1, dtype=torch.int64),
            torch.tensor(values),
            size=(2, 4)
        )
        mask1 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices2, dtype=torch.int64),
            torch.tensor(col_indices1, dtype=torch.int64),
            torch.tensor(mask_values),
            dtype=torch.bool,
            size=(2, 4),
        )
        t2 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices2, dtype=torch.int64),
            torch.tensor(col_indices1, dtype=torch.int64),
            torch.tensor(values),
            size=(2, 4),
        )
        mask2 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices2, dtype=torch.int64),
            torch.tensor(col_indices2, dtype=torch.int64),
            torch.tensor(mask_values),
            dtype=torch.bool,
            size=(2, 4),
        )

        msg = "data and mask are both sparse CSR tensors but do not share either crow or col indices."
        with self.assertRaisesRegex(ValueError, msg):
            masked_tensor(t1, mask1)
        with self.assertRaisesRegex(ValueError, msg):
            masked_tensor(t2, mask2)

    def test_contiguous(self, device):
        # [FUNCTION DOC] Checks contiguity queries: a contiguous and a strided non-contiguous
        # masked tensor report correctly; sparse ones raise ValueError for both the query and
        # making contiguous; making the strided one contiguous yields an equal masked tensor
        # with contiguous data.
        data = torch.randn((3, 3), device=device)

        contiguous_data = data.clone()
        mask1 = (contiguous_data > 0).bool()
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))
        mask2 = (not_contiguous_data > 0).bool()

        contiguous_mt = masked_tensor(contiguous_data, mask1)
        not_contiguous_mt = masked_tensor(not_contiguous_data, mask2)

        contiguous_mt_sparse = masked_tensor(
            contiguous_data.to_sparse_coo(), mask1.to_sparse_coo()
        )
        not_contiguous_mt_sparse = masked_tensor(
            not_contiguous_data.to_sparse_coo(), mask2.to_sparse_coo()
        )

        self.assertEqual(contiguous_data.is_contiguous(), True)
        self.assertEqual(not_contiguous_data.is_contiguous(), False)

        self.assertEqual(contiguous_mt.is_contiguous(), True)
        self.assertEqual(not_contiguous_mt.is_contiguous(), False)

        error_msg = "MaskedTensors with sparse data do not have is_contiguous"
        for t in [contiguous_mt_sparse, not_contiguous_mt_sparse]:
            with self.assertRaisesRegex(ValueError, error_msg):
                t.is_contiguous()
            with self.assertRaisesRegex(ValueError, error_msg):
                t.contiguous()

        now_contiguous_mt = not_contiguous_mt.contiguous()

        _compare_mts(not_contiguous_mt, now_contiguous_mt)

        self.assertEqual(now_contiguous_mt.is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.get_data().is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.is_contiguous(), True)

class TestUnary(TestCase):
    # [CLASS DOC] Tests unary ops on masked tensors against plain tensors for the native unary
    # function lists, with per-op input domain fixing helpers. Parametrized per function.
    def _get_test_data(self, fn_name):
        # [FUNCTION DOC] HELPER: Returns random 10x10 data and a random boolean mask. The data
        # is adjusted to the valid domain of the named op (abs, clamp, offset, or integer
        # conversion for bitwise_not).
        data = torch.randn(10, 10)
        mask = torch.rand(10, 10) > 0.5
        fn_name = _fix_fn_name(fn_name)
        if fn_name in ["log", "log10", "log1p", "log2", "sqrt"]:
            data = data.mul(0.5).abs()
        if fn_name == "rsqrt":
            data = data.abs() + 1  # Void division by zero
        if fn_name in ["acos", "arccos", "asin", "arcsin", "logit"]:
            data = data.abs().mul(0.5).clamp(0, 1)
        if fn_name in ["atanh", "arctanh", "erfinv"]:
            data = data.mul(0.5).clamp(-1, 1)
        if fn_name in ["acosh", "arccosh"]:
            data = data.abs() + 1
        if fn_name == "bitwise_not":
            data = data.mul(128).to(torch.int8)
        return data, mask

    def _get_sample_kwargs(self, fn_name):
        # [FUNCTION DOC] HELPER: Returns keyword arguments for the op: min and max bounds for
        # clamp and clip, otherwise an empty mapping.
        fn_name = _fix_fn_name(fn_name)
        kwargs = {}
        if fn_name in ["clamp", "clip"]:
            kwargs["min"] = -0.5
            kwargs["max"] = 0.5
        return kwargs

    def _get_sample_args(self, fn_name, data, mask):
        # [FUNCTION DOC] HELPER: Builds argument lists for the plain tensor call and the
        # masked tensor call; for pow an extra exponent of two is appended to both. Returns
        # the two lists.
        fn_name = _fix_fn_name(fn_name)
        mt = masked_tensor(data, mask)
        t_args = [data]
        mt_args = [mt]
        if fn_name == "pow":
            t_args += [2.0]
            mt_args += [2.0]
        return t_args, mt_args

    @parametrize("fn", NATIVE_UNARY_FNS)
    def test_unary(self, fn):
        # [FUNCTION DOC] Seeds the RNG, runs each native unary function on masked and plain
        # inputs and compares the results.
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        t_args, mt_args = self._get_sample_args(fn_name, data, mask)

        mt_result = fn(*mt_args, **kwargs)
        t_result = fn(*t_args, **kwargs)
        _compare_mt_t(mt_result, t_result)

    @parametrize("fn", NATIVE_INPLACE_UNARY_FNS)
    def test_inplace_unary(self, fn):
        # [FUNCTION DOC] Seeds the RNG, runs each native in-place unary function on masked and
        # plain inputs and compares the results.
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        t_args, mt_args = self._get_sample_args(fn_name, data, mask)

        mt_result = fn(*mt_args, **kwargs)
        t_result = fn(*t_args, **kwargs)
        _compare_mt_t(mt_result, t_result)

class TestBinary(TestCase):
    # [CLASS DOC] Tests binary ops on masked tensors against plain tensors for the native
    # binary and in-place binary function lists, plus mask mismatch errors. Parametrized
    # per function.
    def _get_test_data(self, fn_name):
        # [FUNCTION DOC] HELPER: Returns two random 10x10 tensors and a random mask; integers
        # for bitwise and/or/xor, non-negative int64 for shifts.
        fn_name = _fix_fn_name(fn_name)
        data0 = torch.randn(10, 10)
        data1 = torch.randn(10, 10)
        mask = torch.rand(10, 10) > 0.5
        if fn_name in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
            data0 = data0.mul(128).to(torch.int8)
            data1 = data1.mul(128).to(torch.int8)
        if fn_name in ["bitwise_left_shift", "bitwise_right_shift"]:
            data0 = data0.abs().to(torch.int64)
            data1 = data1.abs().to(torch.int64)
        return data0, data1, mask

    def _get_sample_kwargs(self, fn_name):
        # [FUNCTION DOC] HELPER: Returns the keyword arguments for the op; always an
        # empty mapping.
        fn_name = _fix_fn_name(fn_name)
        kwargs = {}
        return kwargs

    def _yield_sample_args(self, fn_name, data0, data1, mask):
        # [FUNCTION DOC] HELPER: Generator yielding pairs of argument lists: plain tensors
        # versus two masked tensors sharing a mask, then plain tensors versus a masked tensor
        # and a plain tensor.
        """ Returns two sets of Tensor and MaskedTensor args for a binary function to compute.
            Tensor args are all the same (just the two provided data tensors),
            while the MaskedTensor args tests both (MaskedTensor, MaskedTensor) and (MaskedTensor, Tensor)
        """
        fn_name = _fix_fn_name(fn_name)
        mt0 = masked_tensor(data0, mask)
        mt1 = masked_tensor(data1, mask)

        t_args = [data0, data1]
        mt_args = [mt0, mt1]
        yield t_args, mt_args

        t_args = [data0, data1]
        mt_args = [mt0, data1]
        yield t_args, mt_args

    @parametrize("fn", NATIVE_BINARY_FNS)
    def test_binary(self, fn):
        # [FUNCTION DOC] For each native binary function, compares masked results to plain
        # results for both argument pairings.
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data0, data1, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        for (t_args, mt_args) in self._yield_sample_args(fn_name, data0, data1, mask):
            mt_result = fn(*mt_args, **kwargs)
            t_result = fn(*t_args, **kwargs)
            _compare_mt_t(mt_result, t_result)

    @parametrize("fn", NATIVE_INPLACE_BINARY_FNS)
    def test_inplace_binary(self, fn):
        # [FUNCTION DOC] Same as the binary test for in-place binary functions.
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data0, data1, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        for (t_args, mt_args) in self._yield_sample_args(fn_name, data0, data1, mask):
            mt_result = fn(*mt_args, **kwargs)
            t_result = fn(*t_args, **kwargs)
            _compare_mt_t(mt_result, t_result)

    @parametrize("fn_name", ["add", "add_"])
    def test_masks_match(self, fn_name):
        # [FUNCTION DOC] For the add op and its in-place form through the aten namespace,
        # checks that masked tensors with different masks raise ValueError with the exact
        # message about matching masks; raises AssertionError on a different message or when
        # no error occurs.
        torch.random.manual_seed(0)
        fn = getattr(torch.ops.aten, fn_name)
        data0, data1, mask = self._get_test_data(fn_name)
        mask0 = mask
        mask1 = torch.rand(mask.size()) > 0.5
        mt0 = masked_tensor(data0, mask0)
        mt1 = masked_tensor(data1, mask1)
        try:
            fn(mt0, mt1)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            if str(e) != "Input masks must match. If you need support for this, please open an issue on Github.":
                raise AssertionError(f"unexpected error message: {e}") from None

class TestReductions(TestCase):
    # [CLASS DOC] Tests reductions on masked tensors (sum, mean, amax, amin, prod, all, any-
    # true) with values and gradients, the constructor differentiability cases for mean, and
    # error cases.
    def test_max_not_implemented(self):
        # [FUNCTION DOC] Checks that max is not implemented and raises TypeError naming the
        # aten op.
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m)
        with self.assertRaisesRegex(TypeError, "torch._ops.aten.max.default"):
            mt.max()

    def test_sum(self):
        # [FUNCTION DOC] Checks sum over all elements and along dim 0 considers only masked-
        # in values.
        d = torch.tensor([[0, 1, 2, 6], [3, 4, 5.0, 7]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(17.0), torch.tensor(True)), mt.sum())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 13]),
                torch.tensor([True, True, False, True]),
            ),
            mt.sum(dim=0),
        )

    def test_sum_grad(self):
        # [FUNCTION DOC] Checks the sum gradient is one at masked-in positions under the
        # same mask.
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.sum().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor(1.0).expand_as(m), m))

    def test_mean(self):
        # [FUNCTION DOC] Checks mean over all elements and along dim 0 uses only masked-
        # in values.
        d = torch.tensor([[0, 1, 3, 2], [3, 4, 1.0, 4]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(2.5), torch.tensor(True)), mt.mean())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.mean(dim=0),
        )

    """
        The following block of tests "test_mean_grad_case_1[a through e] are used to test the functionality of
        the two different ways of constructing MaskedTensors:
            masked_tensor(data, mask, requires_grad=True/False) -- NO differentiable constructor and always a leaf
            as_masked_tensor(data, mask) -- differentiable constructor

        Like torch.tensor(data), masked_tensor(data, mask) will provide a UserWarning if data.requires_grad=True
        as_masked_tensor does not take in requires_grad -- it just takes on the requires_grad from data

        Therefore, there are 6 cases to test and we use `mean` as a proxy to test the different combinations

        Assuming mt.mean().backward() is run after each constructor:

        Case 1a:
            values.requires_grad = True
            mt = masked_tensor(values, mask, requires_grad=True)
        yields
            - Provide a UserWarning because values.requires_grad=True
            - values.grad = None
            - mt.grad is a MaskedTensor with the correct gradient

        Case 1b:
            values.requires_grad = False
            mt = masked_tensor(values, mask, requires_grad=True)
        yields
            - values.grad = None
            - mt.grad is a MaskedTensor with the correct gradient

        Case 2a/2b:
            values.requires_grad = True/False
            mt = masked_tensor(values, mask, requires_grad=False)

            will both yield a RuntimeError of "element 0 of tensors does not require grad and does not have a grad_fn"
            as expected. When values.requires_grad=True, we will also get a UserWarning

        Case 3a:
            values.requires_grad = True
            mt = as_masked_tensor(values, mask)
        yields
            - values.grad is a MaskedTensor with the correct gradient
            - mt.grad is None and gives a UserWarning that
              "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad"

        Case 3b:
            values.requires_grad = False
            mt = as_masked_tensor(values, mask)

            will yield a RuntimeError of "element 0 of tensors does not require grad and does not have a grad_fn"
            as expected.
    """
    def test_mean_grad_case_1a(self):
        # [FUNCTION DOC] Case: data requires grad, masked tensor made with requires_grad true.
        # Expects a warning, no gradient on the data, and masked gradient of mean (half at
        # masked-in elements).
        """ values.requires_grad = True
            mt = masked_tensor(values, mask, requires_grad=True)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        with self.assertWarnsRegex(UserWarning, "It is not recommended to create a MaskedTensor"):
            mt = masked_tensor(d, m, requires_grad=True)
        mt.mean().backward()
        self.assertIsNone(d.grad)
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))

    def test_mean_grad_case_1b(self):
        # [FUNCTION DOC] Case: data without grad, masked tensor made with requires_grad true.
        # Expects no gradient on the data and a correct masked gradient on the masked tensor.
        """ values.requires_grad = False
            mt = masked_tensor(values, mask, requires_grad=True)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.mean().backward()
        self.assertIsNone(d.grad)
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))

    def test_mean_grad_case_1c(self):
        # [FUNCTION DOC] Case: data requires grad, masked tensor made with requires_grad
        # false. Expects a warning on construction and RuntimeError on backward since no
        # graph exists.
        # KNOWN ISSUE: the stray string literal above this group of tests labels cases 1a,
        # 1b, 2a/2b, 3a/3b, which do not match these test names 1a to 1f.
        """ values.requires_grad = True
            mt = masked_tensor(values, mask, requires_grad=False)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        with self.assertWarnsRegex(UserWarning, "It is not recommended to create a MaskedTensor"):
            mt = masked_tensor(d, m, requires_grad=False)
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()


    def test_mean_grad_case_1d(self):
        # [FUNCTION DOC] Case: data without grad, masked tensor made with requires_grad false.
        # Expects RuntimeError on backward.
        """ values.requires_grad = False
            mt = masked_tensor(values, mask, requires_grad=False)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=False)
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()

    def test_mean_grad_case_1e(self):
        # [FUNCTION DOC] Case: data requires grad, masked tensor via the differentiable
        # constructor. The data gets the masked gradient; reading the gradient of the non-leaf
        # masked tensor warns and returns none.
        """ values.requires_grad = True
            mt = as_masked_tensor(values, mask)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = as_masked_tensor(d, m)
        mt.mean().backward()
        _compare_mts(d.grad, masked_tensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))
        msg = "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad"
        with self.assertWarnsRegex(UserWarning, msg):
            self.assertIsNone(mt.grad)

    def test_mean_grad_case_1f(self):
        # [FUNCTION DOC] Case: data without grad, masked tensor via the differentiable
        # constructor. Expects RuntimeError on backward.
        """ values.requires_grad = False
            mt = as_masked_tensor(values, mask)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = as_masked_tensor(d, m)
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()

    def test_mean_dim_grad(self):
        # [FUNCTION DOC] Checks the gradient of the sum of the mean along dim 1 under
        # the mask.
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, True, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.mean(1).sum().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.5, 0.5, 0], [0, 1, 0]]), m))

    def test_amax(self):
        # [FUNCTION DOC] Checks amax over all elements and along dim 0 using only masked-
        # in values.
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(3.0), torch.tensor(True)), mt.amax())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, -4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amax(dim=0),
        )

    def test_amax_grad(self):
        # [FUNCTION DOC] Checks the amax gradient is one at the maximum masked-in
        # element only.
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.amax().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.0, 0, 0], [0, 1, 0]]), m))

    def test_amin(self):
        # [FUNCTION DOC] Checks amin over all elements and along dim 0 using only masked-
        # in values.
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(-4.0), torch.tensor(True)), mt.amin())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, -4.0, 1.0, -3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amin(dim=0),
        )

    def test_amin_grad(self):
        # [FUNCTION DOC] Checks the amin gradient is one at the minimum masked-in
        # element only.
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.amin().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[1.0, 0, 0], [0, 0, 0]]), m))

    def test_prod(self):
        # [FUNCTION DOC] Checks prod over all elements and along dim 0, with a masked-out
        # nan ignored.
        d = torch.tensor([[0, 1, 3, 0.0], [float("nan"), 4, 1.0, 5.0]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(0.0), torch.tensor(True)), mt.prod())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 0.0]),
                torch.tensor([True, True, False, True]),
            ),
            mt.prod(dim=0),
        )

    def test_prod_grad(self):
        # [FUNCTION DOC] Checks the prod gradient at masked-in positions is the product of the
        # others, with a masked-out nan ignored.
        d = torch.tensor([[2, float("nan"), 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.prod().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[4.0, 0, 0], [0, 2, 0]]), m))

    def test_all(self):
        # [FUNCTION DOC] Checks all over all elements and along dim 0 for two different masks,
        # using only masked-in values.
        d = torch.tensor([[True, True, False, False], [False, True, True, True]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(False), torch.tensor(True)), mt.all())
        _compare_mts(
            masked_tensor(
                torch.tensor([True, True, True, False]),
                torch.tensor([True, True, False, True]),
            ),
            mt.all(dim=0),
        )

        m = torch.tensor([[True, False, True, False], [False, True, False, False]])
        mt = masked_tensor(d, m)
        _compare_mts(
            masked_tensor(
                torch.tensor([True, True, False, True]),
                torch.tensor([True, True, True, False]),
            ),
            mt.all(dim=0),
        )

    def test_grad_dtype(self):
        # [FUNCTION DOC] Checks that boolean data with requires_grad raises RuntimeError
        # about dtypes.
        d = torch.tensor([[True, True, False], [False, True, True]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        msg = "Only Tensors of floating point and complex dtype can require gradients"
        with self.assertRaisesRegex(RuntimeError, msg):
            masked_tensor(d, m, requires_grad=True)

    def test_any_true_dtype(self):
        # [FUNCTION DOC] Checks the any-true check on non-boolean data raises ValueError.
        mt = torch.masked.MaskedTensor(
            torch.rand(2, 2),
            torch.rand(2, 2) > 0.5
        )
        msg = "expected a boolean tensor"
        with self.assertRaisesRegex(ValueError, msg):
            mt._is_any_true()

    def test__is_any_true(self):
        # [FUNCTION DOC] Checks the any-true check is true when some masked-in value is true.
        mt = torch.masked.MaskedTensor(
            torch.tensor([[True, True, False], [False, False, True]]),
            torch.tensor([[True, False, False], [False, True, False]]),
        )
        _compare_mts(
            masked_tensor(torch.tensor(True), torch.tensor(True)),
            mt._is_any_true(),
        )

    def test__is_any_true_false(self):
        # [FUNCTION DOC] Checks the any-true check is false when every element is masked out.
        mt = torch.masked.MaskedTensor(
            torch.tensor([[True, True, False], [False, False, True]]),
            torch.tensor([[False, False, False], [False, False, False]]),
        )
        _compare_mts(
            masked_tensor(torch.tensor(False), torch.tensor(True),),
            mt._is_any_true(),
        )

    def test_backward(self):
        # [FUNCTION DOC] Regression test for issue 128557: backward of sum on a
        # masked tensor with grad must not trigger an error under anomaly detection.
        # See https://github.com/pytorch/pytorch/issues/128557
        with torch.autograd.detect_anomaly():
            mt = torch.masked.MaskedTensor(
                torch.rand(2, 2),
                torch.rand(2, 2) > 0.5,
                requires_grad=True
            )
            mt.sum().backward()


def is_unary(op):
    # [FUNCTION DOC] HELPER: Returns whether the op name is in the masked unary op names.
    return op.name in UNARY_NAMES

def is_binary(op):
    # [FUNCTION DOC] HELPER: Returns whether the op name is in the masked binary op names.
    return op.name in BINARY_NAMES

def is_reduction(op):
    # [FUNCTION DOC] HELPER: Returns whether the op is a supported masked reduction name,
    # excluding all, mean, std and var.
    return op.name in REDUCE_NAMES and op.name not in {"all", "mean", "std", "var"}

mt_unary_ufuncs = [op for op in unary_ufuncs if is_unary(op)]
mt_binary_ufuncs = [op for op in binary_ufuncs if is_binary(op)]
mt_reduction_ufuncs = [op for op in reduction_ops if is_reduction(op)]

MASKEDTENSOR_FLOAT_TYPES = {
    torch.float16,
    torch.float32,
    torch.float64,
}

class TestOperators(TestCase):
    # [CLASS DOC] OpInfo-driven tests comparing masked tensor results with plain tensor
    # results for unary ufuncs, binary ufuncs and reductions, on float16, float32 and
    # float64 only.
    def _convert_mt_args(self, args, mask, layout):
        # [FUNCTION DOC] HELPER: Converts every tensor in the argument list to a masked tensor
        # with the mask (data restricted to the mask when the layout is not strided); other
        # arguments pass through.
        return [
            masked_tensor(
                arg.sparse_mask(mask) if layout != torch.strided else arg, mask
            )
            if torch.is_tensor(arg)
            else arg
            for arg in args
        ]

    def _test_unary_binary_equality(self, device, dtype, op, layout=torch.strided):
        # [FUNCTION DOC] HELPER: For each requires-grad sample, takes the mask from the kwargs
        # (removing it) or makes a random one, converts input and mask to the layout (CSR
        # skips non-2D), skips binary samples whose shapes differ and drops kwargs for binary
        # ops, then compares the masked result with the plain result. For strided binary ops
        # also checks a masked-times-plain pairing.
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            input = sample.input
            sample_args, sample_kwargs = sample.args, sample.kwargs
            mask = (
                _create_random_mask(input.shape, device)
                if "mask" not in sample_kwargs
                else sample_kwargs.pop("mask")
            )

            if layout == torch.sparse_coo:
                mask = mask.to_sparse_coo().coalesce()
                input = input.sparse_mask(mask)
            elif layout == torch.sparse_csr:
                if input.ndim != 2 or mask.ndim != 2:
                    continue
                mask = mask.to_sparse_csr()
                input = input.sparse_mask(mask)

            # Binary operations currently only support same size masks
            if is_binary(op):
                if input.shape != sample_args[0].shape:
                    continue
                # Binary operations also don't support kwargs right now
                else:
                    sample_kwargs = {}

            mt = masked_tensor(input, mask)
            mt_args = self._convert_mt_args(sample_args, mask, layout)

            mt_result = op(mt, *mt_args, **sample_kwargs)
            t_result = op(sample.input, *sample_args, **sample_kwargs)

            _compare_mt_t(mt_result, t_result)

            # If the operation is binary, check that lhs = masked, rhs = regular tensor also works
            if is_binary(op) and layout == torch.strided:
                mt_result2 = op(mt, *sample_args, **sample_kwargs)
                _compare_mt_t(mt_result2, t_result)

    def _test_reduction_equality(self, device, dtype, op, layout=torch.strided):
        # [FUNCTION DOC] HELPER: For each sample, skips zero-dim, empty and all-false-mask
        # inputs, builds the reference by filling masked-out entries with the op identity,
        # converts input and mask to the layout (CSR skips non-2D), applies the op with no
        # extra args to the masked tensor and compares.
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            input = sample.input
            # Reduction operations don't support more advanced args/kwargs right now
            sample_args, sample_kwargs = (), {}

            if input.dim() == 0 or input.numel() == 0:
                continue

            mask = _create_random_mask(input.shape, device)

            if torch.count_nonzero(mask) == 0:
                continue

            tensor_input = _combine_input_and_mask(op.op, input, mask)
            if layout == torch.sparse_coo:
                mask = mask.to_sparse_coo().coalesce()
                input = input.sparse_mask(mask)
            elif layout == torch.sparse_csr:
                if input.ndim != 2 or mask.ndim != 2:
                    continue
                mask = mask.to_sparse_csr()
                input = input.sparse_mask(mask)

            mt = masked_tensor(input, mask)
            mt_args = self._convert_mt_args(sample_args, mask, layout)

            mt_result = op(mt, *mt_args, **sample_kwargs)
            t_result = op(tensor_input, *sample_args, **sample_kwargs)

            _compare_mt_t(mt_result, t_result)

    @ops(mt_unary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    @parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
    def test_unary_core(self, device, dtype, op, layout):
        # [FUNCTION DOC] Unary op test, parametrized by layout. Skips round variants that use
        # keyword arguments.
        # KNOWN ISSUE: the layout is not forwarded to the helper, so every layout runs the
        # strided case.
        # Skip tests that don't have len(kwargs) == 0
        skip_variants = {
            "decimals_0",
            "decimals_3",
            "decimals_neg_3",
        }
        if op.name == "round" and op.variant_test_name in skip_variants:
            return
        self._test_unary_binary_equality(device, dtype, op)

    @ops(mt_binary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    @parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
    def test_binary_core(self, device, dtype, op, layout):
        # [FUNCTION DOC] Binary op test for the given layout; delegates to the
        # equality helper.
        self._test_unary_binary_equality(device, dtype, op, layout)

    @ops(mt_reduction_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    @parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
    def test_reduction_all(self, device, dtype, op, layout):
        # [FUNCTION DOC] Reduction op test for the given layout; returns early for arg min and
        # arg max on sparse CSR, otherwise delegates to the reduction equality helper.
        # argmin and argmax are not currently supported for torch.sparse_csr
        if op.name in {"argmin", "argmax"} and layout == torch.sparse_csr:
            return

        self._test_reduction_equality(device, dtype, op, layout)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

instantiate_device_type_tests(TestBasics, globals(), only_for=only_for)
instantiate_parametrized_tests(TestUnary)
instantiate_parametrized_tests(TestBinary)
instantiate_parametrized_tests(TestReductions)

if __name__ == '__main__':
    run_tests()
