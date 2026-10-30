# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841
# [MODULE DOC] Tests for quantized tensors (per-tensor and per-channel affine, sub-byte
# types): creation, quantize/dequantize, memory formats, device moves, fill, masked_fill,
# index_put, index_select, view/resize/reshape/unsqueeze, copy/clone, serialization
# (save/load, pickle, TorchScript), fp16 saturation, optimized qparam search, and the
# quantized_decomposed ops. Also holds numpy reference helpers. Import calls
# hu.assert_deadline_disabled to turn off hypothesis deadlines. Direct execution raises
# RuntimeError.

import numpy as np
import math
import random
import torch
import io
import unittest
from copy import deepcopy
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TestCase, DeterministicGuard
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_quantization import get_supported_device_types

hu.assert_deadline_disabled()

import itertools
import tempfile

class Foo(torch.nn.Module):
    # [CLASS DOC] HELPER: Minimal module holding a qscheme attribute set to
    # per_tensor_symmetric; used to test pickling of a qscheme.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Call the base initializer and store the
        # per_tensor_symmetric qscheme in the qscheme attribute.
        super().__init__()
        self.qscheme = torch.per_tensor_symmetric

def _calculate_dynamic_qparams(X, dtype, reduce_range=False):
    # [FUNCTION DOC] HELPER: Numpy reference for dynamic quantization parameters. Accepts a
    # tensor or array (tensors are moved to cpu and converted), a quantized dtype (qint8 or
    # otherwise treated as quint8) and an optional reduce_range flag that narrows the integer
    # range. Extends the observed min and max to include zero, derives a scale from the range,
    # picks the zero point from whichever end has smaller error, and clamps and rounds it into
    # the integer range. Returns a list of a float32 scale and an int zero point.
    # KNOWN ISSUE: when the scale is zero or its inverse is infinite the scale is replaced by
    # 0.1 and a local zero_point is set, but that value is never used; the zero point is
    # recomputed afterwards from the replaced scale.
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.cpu().data.numpy()
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:  # dtype == torch.quint8
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255

    min_val = X.min().astype(dtype=np.float32)
    max_val = X.max().astype(dtype=np.float32)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    scale = (np.float64(max_val) - min_val) / (qmax - qmin)
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = np.float64(0.1)
        zero_point = 0

    zero_point_from_min = qmin - min_val / float(scale)
    zero_point_from_max = qmax - max_val / float(scale)
    zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
    zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
    if zero_point_from_min_error < zero_point_from_max_error:
        initial_zero_point = zero_point_from_min
    else:
        initial_zero_point = zero_point_from_max
    nudged_zero_point = 0

    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = int(round(initial_zero_point))

    return [scale.astype(np.float32), int(nudged_zero_point)]

# Note we explicitly cast variables to np.float32 in a couple of places to avoid
# the default casting in Python often resulting in double precision and to make
# sure we're doing the same numerics as C++ code.
def param_search_greedy(x, bit_rate, n_bins=200, ratio=0.16):
    # [FUNCTION DOC] HELPER: Numpy reference of the greedy range search used by optimized
    # qparam selection (mirrors C++ numerics). Starting from the full min and max of the input
    # it repeatedly tries shrinking either the left or right end by one step (range divided by
    # n_bins) using the simplified compress routine to measure loss, moves toward the lower
    # loss, and records local optima, stopping once the range drops below n_bins times one
    # minus ratio of steps. Returns a pair holding the max and then the min of the lowest-loss
    # local optimum, or of the full range when none was found.
    xmin, xmax = np.min(x), np.max(x)
    stepsize = (xmax - xmin) / np.float32(n_bins)
    min_bins = np.float32(n_bins) * (np.float32(1) - np.float32(ratio))
    xq, loss = _compress_uniform_simplified(x, bit_rate, xmin, xmax)

    solutions = []  # [(left, right, loss)] # local optima solution

    cur_min, cur_max, cur_loss = xmin, xmax, loss
    thr = min_bins * stepsize
    while cur_min + thr < cur_max:
        # move left
        xq, loss1 = _compress_uniform_simplified(
            x, bit_rate, cur_min + stepsize, cur_max
        )
        # move right
        xq, loss2 = _compress_uniform_simplified(
            x, bit_rate, cur_min, cur_max - stepsize
        )

        if cur_loss < loss1 and cur_loss < loss2:
            # found a local optima
            solutions.append((cur_min, cur_max, cur_loss))
        if loss1 < loss2:
            cur_min, cur_loss = cur_min + stepsize, loss1
        else:
            cur_max, cur_loss = cur_max - stepsize, loss2
    if solutions:
        best = solutions[0]
        for solution in solutions:
            if solution[-1] < best[-1]:
                best = solution
        return best[1], best[0]  # xmax, xmin
    return xmax, xmin


def _compress_uniform_simplified(X, bit_rate, xmin, xmax, fp16_scale_bias=True):
    # [FUNCTION DOC] HELPER: Numpy reference that quantizes a vector to bit_rate bits over
    # the interval xmin to xmax and reconstructs it. Optionally rounds xmin and the scale
    # through float16 (fp16_scale_bias). A zero data range uses scale one. Computes the l2
    # loss between input and reconstruction with eight interleaved partial sums to match the
    # C++ accumulation order. Returns the reconstructed values and the loss.
    # affine transform to put Xq in [0,2**bit_rate - 1]
    # Xq = (2 ** bit_rate - 1) * (Xq - xmin) / data_range
    if fp16_scale_bias:
        xmin = xmin.astype(np.float16).astype(np.float32)
    data_range = xmax - xmin
    scale = np.where(
        data_range == 0, np.float32(1), data_range / np.float32(2 ** bit_rate - 1)
    )
    if fp16_scale_bias:
        scale = scale.astype(np.float16).astype(np.float32)
    inverse_scale = np.float32(1) / scale
    Xq = np.clip(np.round((X - xmin) * inverse_scale), 0, np.float32(2 ** bit_rate - 1))
    Xq = Xq * scale + xmin

    # Manually compute loss instead of using np.linalg.norm to use the same
    # accumulation order used by C++ code
    vlen = 8
    loss_v = np.zeros(vlen).astype(np.float32)
    for i in range(len(Xq) // vlen * vlen):
        loss_v[i % vlen] += (X[i] - Xq[i]) * (X[i] - Xq[i])
    loss = np.float32(0)
    for i in range(vlen):
        loss += loss_v[i]
    for i in range(len(Xq) // vlen * vlen, len(Xq)):
        loss += (X[i] - Xq[i]) * (X[i] - Xq[i])
    loss = np.sqrt(loss)

    return Xq, loss

class TestQuantizedTensor(TestCase):
    # [CLASS DOC] Test case for quantized tensor behavior; most tests have cpu and cuda
    # variants built on private helper methods that take a device.
    def test_quantized_tensor_creation_deprecation_warning(self):
        # [FUNCTION DOC] Asserts that creating a quantized tensor with the per-tensor
        # quantize function emits a UserWarning matching the deprecated-and-will-be-removed
        # text, exactly once.
        with self.assertWarnsOnceRegex(UserWarning, ".*deprecated and will be removed"):
            torch.quantize_per_tensor(torch.randn(4), 0.1, 10, torch.quint8)

    def test_qtensor_equal(self):
        # [FUNCTION DOC] Quantizes the same random five-element input twice with identical
        # parameters to quint4x2 and asserts that torch.equal reports the two quantized
        # tensors equal.
        x = torch.rand(5)
        x_q = torch.quantize_per_tensor(x, 0.1, 10, torch.quint4x2)
        y_q = torch.quantize_per_tensor(x, 0.1, 10, torch.quint4x2)
        self.assertTrue(torch.equal(x_q, y_q))

    def test_per_tensor_qtensor_to_memory_format(self):
        # [FUNCTION DOC] With a random four dimensional float input and random qint dtype,
        # checks that converting a per-tensor quantized tensor to channels_last via the to
        # method gives the same strides as calling contiguous with that format and as the
        # float tensor. For two degenerate shapes (trailing size-one dims, or one channel)
        # asserts the strides from the two paths differ.
        n = np.random.randint(1, 10)
        c = np.random.randint(2, 10)
        h = np.random.randint(2, 10)
        w = np.random.randint(2, 10)
        x = torch.rand(n, c, h, w)
        scale = np.random.uniform(0.1, 1.0)
        zero_point = np.random.randint(0.0, 10)
        qints = [torch.qint8, torch.quint8, torch.qint32]
        dtype = qints[np.random.randint(0, len(qints))]
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        x_nhwc = x.to(memory_format=torch.channels_last)
        qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=torch.channels_last)
        self.assertEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())
        self.assertEqual(qx_nhwc_using_to.stride(), x_nhwc.stride())

        # When the last two dimensions of a 4D tensor are both size 1 or if c == 1, we have a degenerate case
        # see https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        # In this case, the output of torch.Tensor.to and torch.Tensor.contiguous should not be the same
        x = torch.rand(10, 2, 1, 1)
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=torch.channels_last)
        self.assertNotEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())

        x = torch.rand(10, 1, 2, 2)
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=torch.channels_last)
        self.assertNotEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())

    def test_per_channel_qtensor_to_memory_format(self):
        # [FUNCTION DOC] For a random four dimensional input and random qint dtype,
        # quantizes per channel along every axis with random scales and zero points and
        # asserts that converting to channels_last gives the same strides as the float tensor
        # converted the same way.
        n = np.random.randint(1, 10)
        c = np.random.randint(2, 10)
        h = np.random.randint(2, 10)
        w = np.random.randint(2, 10)
        x = torch.rand(n, c, h, w)
        x_nhwc = x.to(memory_format=torch.channels_last)
        scale = np.random.uniform(0.1, 1.0)
        zero_point = np.random.randint(0.0, 10)
        qints = [torch.qint8, torch.quint8, torch.qint32]
        dtype = qints[np.random.randint(0, len(qints))]
        for axis in range(x.ndim):
            scales = torch.rand(x.size(axis)) + 0.00001
            zero_points = torch.randint(low=0, high=10, size=(x.size(axis), ))
            qx = torch.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=dtype, axis=axis)
            qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
            self.assertEqual(qx_nhwc_using_to.stride(), x_nhwc.stride())

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_cuda(self):
        # [FUNCTION DOC] CUDA test: runs the shared qtensor and dynamic-qtensor checks on
        # the cuda device; skipped if no gpu.
        self._test_qtensor(torch.device('cuda'))
        self._test_qtensor_dynamic(torch.device('cuda'))

    def test_qtensor_cpu(self):
        # [FUNCTION DOC] Runs the shared qtensor and dynamic-qtensor checks on the cpu
        # device.
        self._test_qtensor(torch.device('cpu'))
        self._test_qtensor_dynamic(torch.device('cpu'))

    def _test_qtensor_dynamic(self, device):
        # [FUNCTION DOC] HELPER: Builds a random float tensor of one to three dims (sizes up
        # to 19) on the device, then for qint8 and quint8 compares dynamic per-tensor
        # quantization against the per-tensor quantize call using parameters from the numpy
        # reference function; asserts equality.
        # max number of tensor dimensions
        max_tensor_order = 4
        # max size for any tensor dimension
        max_dim_sz = 20

        num_dim = np.random.randint(low=1, high=max_tensor_order)
        dims = np.random.randint(low=1, high=max_dim_sz, size=num_dim)
        mat2quant = torch.randn(*dims, dtype=torch.float, device=device)
        reduce_flag = False

        for dtype in [torch.qint8, torch.quint8]:
            q_d = torch.quantize_per_tensor_dynamic(mat2quant, dtype, reduce_flag)
            scale, zero_pt = _calculate_dynamic_qparams(mat2quant, dtype, reduce_flag)
            q_s = torch.quantize_per_tensor(mat2quant, scale, zero_pt, dtype)

            self.assertEqual(q_d, q_s)

    def _test_qtensor(self, device):
        # [FUNCTION DOC] HELPER: For qint8, quint8 and qint32 quantizes a ones vector of ten
        # with scale one and zero point two on the device, checks the scale, zero point,
        # is_quantized flag, qscheme and integer representation (all three), slicing,
        # dequantized round trip, and that printing an empty quantized tensor yields the
        # expected whitespace-normalized text including device, size, dtype, scheme, scale and
        # zero point.
        device = str(device)
        num_elements = 10
        scale = 1.0
        zero_point = 2
        for dtype in [torch.qint8, torch.quint8, torch.qint32]:
            r = torch.ones(num_elements, dtype=torch.float, device=device)
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            self.assertEqual(qr.q_scale(), scale)
            self.assertEqual(qr.q_zero_point(), zero_point)
            self.assertTrue(qr.is_quantized)
            self.assertFalse(r.is_quantized)
            self.assertEqual(qr.qscheme(), torch.per_tensor_affine)
            self.assertTrue(isinstance(qr.qscheme(), torch.qscheme))
            # slicing and int_repr
            int_repr = qr.int_repr()
            for num in int_repr:
                self.assertEqual(num, 3)
            for num in qr[2:].int_repr():
                self.assertEqual(num, 3)
            # dequantize
            rqr = qr.dequantize()
            for i in range(num_elements):
                self.assertEqual(r[i], rqr[i])
            # we can also print a qtensor
            empty_r = torch.ones((0, 1), dtype=torch.float, device=device)
            empty_qr = torch.quantize_per_tensor(empty_r, scale, zero_point, dtype)

            device_msg = "" if device == 'cpu' else "device='" + device + ":0', "
            dtype_msg = str(dtype) + ", "
            self.assertEqual(' '.join(str(empty_qr).split()),
                             "tensor([], " + device_msg + "size=(0, 1), dtype=" + dtype_msg +
                             "quantization_scheme=torch.per_tensor_affine, " +
                             "scale=1.0, zero_point=2)")

    def test_qtensor_int_repr(self):
        # [FUNCTION DOC] Quantizes a single-element tensor to quint2x4 and asserts the
        # packed integer representation has one byte holding the value three, so at least one
        # byte is allocated when the bit count is below eight.
        # to catch edge case when num elements * bit rate < 8, make sure at lease allocate one byte to hold the int repr
        num_elements = 1
        device = torch.device('cpu')
        scale = 1.0
        zero_point = 2
        dtype = torch.quint2x4
        r = torch.ones(num_elements, dtype=torch.float, device=device)
        qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
        int_repr = qr.int_repr()
        self.assertEqual(int_repr.numel(), 1)
        # Packed one entry looks like 00000011
        self.assertEqual(int_repr[0], 3)

    def test_qtensor_sub_byte_aligned_cols(self):
        # [FUNCTION DOC] Runs the sub-byte helper for one row of four columns with quint4x2
        # expecting two bytes of value 51 and quint2x4 (expect one byte of 255), where columns
        # fill whole bytes.
        # Packed 4 entries, each of value 3, look like 00110011, 00110011 for torch.qunit4x2, or 11111111 for torch.quint2x4
        self._test_qtensor_sub_byte(1, 4, torch.quint4x2, 2, [51, 51])
        self._test_qtensor_sub_byte(1, 4, torch.quint2x4, 4, [255])

    def test_qtensor_sub_byte_not_aligned_cols(self):
        # [FUNCTION DOC] Runs the sub-byte helper for one row of five columns with quint4x2
        # expecting bytes 51, 51 and 3 and quint2x4 expecting bytes 255 and 3, where the last
        # byte is partially filled.
        # Packed 5 entries, each of value 3, look like 00110011, 00110011, 00000011 for torch.qunit4x2,
        # or 11111111, 00000011 for torch.quint2x4
        self._test_qtensor_sub_byte(1, 5, torch.quint4x2, 2, [51, 51, 3])
        self._test_qtensor_sub_byte(1, 5, torch.quint2x4, 4, [255, 3])

    def _test_qtensor_sub_byte(self, rows, cols, dtype, elements_per_byte, expected_packed_vals):
        # [FUNCTION DOC] HELPER: Quantizes a ones matrix of the given rows and columns to
        # the sub-byte dtype (scale one, zero point two), checks scale, zero point, flags,
        # storage size equal to rows times the ceiling of cols over elements per byte, and
        # that the integer representation equals the expected packed byte values. Also checks
        # an empty affine tensor storage size, and that saving and loading with weights_only
        # both true and false preserves the integer representation.
        num_elements = rows * cols
        scale = 1.0
        zero_point = 2

        r = torch.ones((rows, cols), dtype=torch.float)
        qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
        self.assertEqual(qr.q_scale(), scale)
        self.assertEqual(qr.q_zero_point(), zero_point)
        self.assertTrue(qr.is_quantized)
        self.assertFalse(r.is_quantized)
        self.assertEqual(qr.storage().size(), rows * math.ceil(cols / elements_per_byte), lambda msg: f"{msg}\nwith {dtype}, {elements_per_byte}")

        int_repr = qr.int_repr()
        self.assertEqual(int_repr.numel(), len(expected_packed_vals))
        for num, expected in zip(int_repr, expected_packed_vals):
            self.assertEqual(num, expected, lambda msg: f"{msg}\nwith dtype={dtype}, elements_per_byte={elements_per_byte}, rows={rows}, cols={cols}")

        # Test tensor creation
        q = torch._empty_affine_quantized([num_elements], scale=scale, zero_point=zero_point, dtype=dtype)
        self.assertEqual(q.storage().size(), math.ceil(num_elements / elements_per_byte), lambda msg: f"{msg}\nwith {dtype}, {elements_per_byte}")

        # Test save/load
        with tempfile.NamedTemporaryFile() as f:
            torch.save(qr, f)
            for weights_only in [True, False]:
                f.seek(0)
                loaded_q = torch.load(f, weights_only=weights_only)
                loaded_int_repr = loaded_q.int_repr()
                self.assertEqual(int_repr, loaded_int_repr)

    def test_qtensor_channel_float_assignment(self):
        # [FUNCTION DOC] For every axis of a four dimensional tensor and each qint dtype,
        # quantizes two float tensors per channel with the same random parameters, then
        # assigns float values into the first via scalar, 1D through 4D indexing and slices
        # and checks each equals the corresponding part of the second. Assigning into a non-
        # contiguous slice must raise RuntimeError about quantized copy needing contiguous or
        # NHWC tensors.
        t1 = torch.rand(2, 3, 5, 5)
        t2 = torch.rand(2, 3, 5, 5)
        for axis in range(t1.ndim):
            scales = np.random.rand(t1.size()[axis])
            zero_points = np.random.randint(low=0, high=50, size=t1.size()[axis])
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                qt1 = torch.quantize_per_channel(t1, scales=torch.tensor(scales),
                                                 zero_points=torch.tensor(zero_points), dtype=dtype, axis=axis)
                qt2 = torch.quantize_per_channel(t2, scales=torch.tensor(scales),
                                                 zero_points=torch.tensor(zero_points), dtype=dtype, axis=axis)
                i = 0
                j = 1
                k = 2
                l = 4
                # scalar assignment verification
                qt1[i][j][k][l] = t2[i][j][k][l]
                self.assertEqual(qt1[i][j][k][l], qt2[i][j][k][l])
                # 1D tensor assignment verification
                qt1[i][j][k][2:l] = t2[i][j][k][2:l]
                self.assertEqual(qt1[i][j][k][2:l], qt2[i][j][k][2:l])
                qt1[i][j][k] = t2[i][j][k]
                self.assertEqual(qt1[i][j][k], qt2[i][j][k])
                # 2D tensor assignment verification
                qt1[i][j][k:] = t2[i][j][k:]
                self.assertEqual(qt1[i][j][k:], qt2[i][j][k:])
                qt1[i][j] = t2[i][j]
                self.assertEqual(qt1[i][j], qt2[i][j])
                # 3D tensor assignment verification
                qt1[i][j:] = t2[i][j:]
                self.assertEqual(qt1[i][j:], qt2[i][j:])
                qt1[i] = t2[i]
                self.assertEqual(qt1[i], qt2[i])
                # 4D tensor assignment verification
                qt1[:1] = t2[:1]
                self.assertEqual(qt1[:1], qt2[:1])
                qt1[:] = t2[:]
                self.assertEqual(qt1[:], qt2[:])
                # non-contiguous case **this should raise an exception**
                with self.assertRaisesRegex(RuntimeError, "Quantized copy only works with contiguous and NHWC Tensors"):
                    qt1[:, 0] = t2[:, 0]

    def test_qtensor_float_assignment(self):
        # [FUNCTION DOC] For cpu (and cuda when available) and each qint dtype, quantizes a
        # one-element ones tensor, checks item values, assigns a float into an element and a
        # slice, and verifies the resulting value (11, then 15) and the printed text including
        # device on cuda.
        # Scalar Tensor
        # item
        scale = 1.0
        zero_point = 2
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            r = torch.ones(1, dtype=torch.float).to(device=device)
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                self.assertEqual(qr.item(), 1)
                self.assertEqual(qr[0].item(), 1)
                # assignment
                self.assertTrue(qr[0].is_quantized)
                qr[0] = torch.Tensor([11.3]).to(device=device)  # float assignment
                self.assertEqual(qr.item(), 11)
                x = torch.ones(1, dtype=torch.float).to(device=device) * 15.3
                # Copying from a float Tensor
                qr[:] = x
                self.assertEqual(qr.item(), 15)

                dtype_msg = str(dtype) + ", "
                if device == "cuda":
                    self.assertEqual(' '.join(str(qr).split()),
                                     "tensor([15.], device='" + str(qr.device) + "', size=(1,), dtype=" + dtype_msg +
                                     "quantization_scheme=torch.per_tensor_affine, " +
                                     "scale=1.0, zero_point=2)")
                else:
                    self.assertEqual(' '.join(str(qr).split()),
                                     "tensor([15.], size=(1,), dtype=" + dtype_msg +
                                     "quantization_scheme=torch.per_tensor_affine, " +
                                     "scale=1.0, zero_point=2)")

    def test_qtensor_quant_dequant(self):
        # [FUNCTION DOC] For every supported device, both memory formats and each qint
        # dtype, quantizes a random tensor in range minus two to two and checks the
        # dequantized result is close to the input with a loose tolerance; repeats for 5D
        # tensors.
        scale = 0.02
        zero_point = 2
        for device in get_supported_device_types():
            r = torch.rand(3, 2, 4, 5, dtype=torch.float, device=device) * 4 - 2
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                r = r.contiguous(memory_format=memory_format)
                for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                    qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
                    rqr = qr.dequantize()
                    self.assertTrue(np.allclose(r.cpu().numpy(), rqr.cpu().numpy(), atol=2 / scale))
        # Also check 5D tensors work.
        for device in get_supported_device_types():
            r = torch.rand(3, 2, 4, 5, 6, dtype=torch.float, device=device) * 4 - 2
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
                rqr = qr.dequantize()
                self.assertTrue(np.allclose(r.cpu().numpy(), rqr.cpu().numpy(), atol=2 / scale))

    # legacy constructor/new doesn't support qtensors
    def test_qtensor_legacy_new_failure(self):
        # [FUNCTION DOC] Asserts the legacy new constructor path raises RuntimeError for a
        # quantized tensor when given a device, storage, tensor, size or list.
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        scale = 0.02
        zero_point = 2
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.quint8)
        self.assertRaises(RuntimeError, lambda: qr.new(device='cpu'))
        self.assertRaises(RuntimeError, lambda: qr.new(r.storage()))
        self.assertRaises(RuntimeError, lambda: qr.new(r))
        self.assertRaises(RuntimeError, lambda: qr.new(torch.Size([2, 3])))
        self.assertRaises(RuntimeError, lambda: qr.new([6]))

    def test_per_channel_qtensor_creation_cpu(self):
        # [FUNCTION DOC] Runs the per-channel creation helper on the cpu device.
        self._test_per_channel_qtensor_creation(torch.device('cpu'))

    def _test_dequantize_fp16(self, device):
        # [FUNCTION DOC] HELPER: Converts a random float tensor to float16, calls dequantize
        # on it and asserts the result has float32 dtype and equals the plain float32
        # conversion.
        data_orig = torch.randn(1, 2, 4, 4, dtype=torch.float, device=device)
        data_fp16 = data_orig.to(torch.float16)
        data_fp16_dequant = data_fp16.dequantize()
        data_fp16_fp32 = data_fp16.to(torch.float)
        self.assertTrue(data_fp16_dequant.dtype == torch.float)
        self.assertTrue(torch.allclose(data_fp16_fp32, data_fp16_dequant))

    def test_dequantize_fp16_cpu(self):
        # [FUNCTION DOC] Runs the fp16 dequantize helper on cpu.
        self._test_dequantize_fp16(torch.device('cpu'))

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_dequantize_fp16_cuda(self):
        # [FUNCTION DOC] CUDA test: runs the fp16 dequantize helper on cuda; skipped without
        # a gpu.
        self._test_dequantize_fp16(torch.device('cuda'))

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_channel_qtensor_creation_cuda(self):
        # [FUNCTION DOC] CUDA test: runs the per-channel creation helper on cuda; skipped
        # without a gpu.
        self._test_per_channel_qtensor_creation(torch.device('cuda'))

    def _test_per_channel_qtensor_creation(self, device):
        # [FUNCTION DOC] HELPER: Creates empty per-channel affine quantized tensors for
        # qint8 and quint8 with float and integer zero points, checks the stored scales, zero
        # points and axis; then builds per-channel tensors from a uint8 tensor and checks
        # integer representation, scales, zero points and axis.
        numel = 10
        ch_axis = 0
        scales = torch.rand(numel, device=device)
        zero_points_int = torch.randint(0, 10, size=(numel,), device=device)
        zero_points_float = torch.randn(numel, device=device)
        for dtype, zero_points in itertools.product([torch.qint8, torch.quint8], [zero_points_float, zero_points_int]):
            q = torch._empty_per_channel_affine_quantized(
                [numel], scales=scales, zero_points=zero_points, axis=ch_axis, dtype=dtype, device=device)
            self.assertEqual(scales, q.q_per_channel_scales(), exact_dtype=False)
            self.assertEqual(zero_points, q.q_per_channel_zero_points())
            self.assertEqual(ch_axis, q.q_per_channel_axis())

        # create Tensor from uint8_t Tensor, scales and zero_points
        for zero_points in [zero_points_float, zero_points_int]:
            int_tensor = torch.randint(0, 100, size=(numel,), dtype=torch.uint8, device=device)
            q = torch._make_per_channel_quantized_tensor(int_tensor, scales, zero_points, ch_axis)
            self.assertEqual(int_tensor, q.int_repr())
            self.assertEqual(scales, q.q_per_channel_scales(), exact_dtype=False)
            self.assertEqual(zero_points, q.q_per_channel_zero_points())
            self.assertEqual(ch_axis, q.q_per_channel_axis())

    def test_qtensor_creation(self):
        # [FUNCTION DOC] For each supported device creates an empty per-tensor affine
        # quantized tensor and checks scale and zero point; builds one from a uint8 tensor and
        # checks integer representation; checks empty_like keeps scale, zero point and dtype,
        # and that empty_like with a different dtype raises RuntimeError.
        scale = 0.5
        zero_point = 10
        numel = 10
        for device in get_supported_device_types():
            q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point,
                                              device=device, dtype=torch.quint8)
            self.assertEqual(scale, q.q_scale())
            self.assertEqual(zero_point, q.q_zero_point())

            # create Tensor from uint8_t Tensor, scale and zero_point
            int_tensor = torch.randint(0, 100, size=(10,), device=device, dtype=torch.uint8)
            q = torch._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point)
            self.assertEqual(int_tensor, q.int_repr())
            self.assertEqual(scale, q.q_scale())
            self.assertEqual(zero_point, q.q_zero_point())

            # create via empty_like
            q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point,
                                              device=device, dtype=torch.quint8)
            q_el = torch.empty_like(q)
            self.assertEqual(q.q_scale(), q_el.q_scale())
            self.assertEqual(q.q_zero_point(), q_el.q_zero_point())
            self.assertEqual(q.dtype, q_el.dtype)

            # create via empty_like but change the dtype (currently not supported)
            with self.assertRaises(RuntimeError):
                torch.empty_like(q, dtype=torch.qint8)

    def test_qtensor_dtypes(self):
        # [FUNCTION DOC] Quantizes a random float matrix to qint8, quint8, qint32, quint4x2
        # and quint2x4, dequantizes it and asserts closeness to the input within a loose
        # tolerance.
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        scale = 0.2
        zero_point = 2
        for dtype in [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]:
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            rqr = qr.dequantize()
            self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_tensor_to_device(self):
        # [FUNCTION DOC] CUDA test: for each qint dtype moves a cpu quantized tensor to cuda
        # and a cuda-quantized tensor to cpu and asserts the resulting device types.
        dtypes = [
            torch.quint8,
            torch.qint8,
            torch.qint32,
        ]
        device = torch.device('cuda')
        for dtype in dtypes:
            r = torch.rand(2, 2, dtype=torch.float) * 10
            scale = torch.rand(2).abs().max().item()
            zero_point = (torch.rand(2) * 10).round().to(torch.long).max().item()

            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            qr = qr.to(device)
            qr_cuda = torch.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            qr_cuda = qr_cuda.to('cpu')
            self.assertEqual('cuda', qr.device.type)
            self.assertEqual('cpu', qr_cuda.device.type)

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_channel_to_device(self):
        # [FUNCTION DOC] CUDA test: for several dtype and zero point type combinations
        # quantizes per channel, moves the tensor between cpu and cuda and asserts the device
        # type of the tensor, its scales and zero points follow the move.
        dtype_and_zero_types = [
            (torch.quint8, torch.float),
            (torch.qint8, torch.float),
            #  (torch.qint32, torch.float) not supported for quantize_per_channel
            (torch.quint8, torch.long),
            (torch.qint8, torch.long),
            (torch.qint32, torch.long),
        ]
        axis = 1
        device = torch.device('cuda')
        for dtype, zero_type in dtype_and_zero_types:
            r = torch.rand(2, 2, dtype=torch.float) * 10
            scales = torch.rand(2).abs()
            zero_points = (torch.rand(2) * 10).round().to(zero_type)

            dqr = torch.quantize_per_channel(r, scales, zero_points, axis, dtype)
            dqr = dqr.to(device)
            dqr_cuda = torch.quantize_per_channel(r.to(device), scales.to(
                device), zero_points.to(device), axis, dtype)
            dqr_cuda = dqr_cuda.to('cpu')

            self.assertEqual('cuda', dqr.device.type)
            self.assertEqual('cuda', dqr.q_per_channel_scales().device.type)
            self.assertEqual('cuda', dqr.q_per_channel_zero_points().device.type)

            self.assertEqual('cpu', dqr_cuda.device.type)
            self.assertEqual('cpu', dqr_cuda.q_per_channel_scales().device.type)
            self.assertEqual('cpu', dqr_cuda.q_per_channel_zero_points().device.type)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_compare_per_tensor_device_numerics(self):
        # [FUNCTION DOC] CUDA test: quantizes the same random data per tensor on cpu and
        # cuda for each qint dtype and asserts equal integer representations and close
        # dequantized values.
        dtypes = [
            torch.quint8,
            torch.qint8,
            torch.qint32,
        ]
        device = torch.device('cuda')
        for dtype in dtypes:
            r = torch.rand(2, 2) * 10
            r[0, 0] = 2.5
            scale = torch.rand(2).abs().max().item()
            zero_point = (torch.rand(2) * 10).round().to(torch.long).max().item()

            qtr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            dqtr = qtr.dequantize()
            qtr_cuda = torch.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            dqtr_cuda = qtr_cuda.dequantize()
            self.assertEqual(qtr.int_repr(), qtr_cuda.int_repr())
            self.assertTrue(np.allclose(dqtr, dqtr_cuda.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_compare_per_channel_device_numerics(self):
        # [FUNCTION DOC] CUDA test: repeated twenty times, per channel quantization on cpu
        # and cuda for several dtype and zero point type combinations must give equal integer
        # representations and close dequantized values.
        dtype_and_zero_types = [
            (torch.quint8, torch.float),
            (torch.qint8, torch.float),
            #  (torch.qint32, torch.float) not supported for quantize_per_channel
            (torch.quint8, torch.long),
            (torch.qint8, torch.long),
            (torch.qint32, torch.long),
        ]
        axis = 1
        device = torch.device('cuda')
        for _ in range(20):
            for dtype, zero_type in dtype_and_zero_types:
                r = torch.rand(2, 2) * 10
                r[0, 0] = 2.5
                scales = torch.rand(2).abs()
                zero_points = (torch.rand(2) * 10).round().to(zero_type)

                qr = torch.quantize_per_channel(r, scales, zero_points, axis, dtype)
                dqr = qr.dequantize()
                qr_cuda = torch.quantize_per_channel(r.to(device), scales.to(
                    device), zero_points.to(device), axis, dtype)
                dqr_cuda = qr_cuda.dequantize()
                self.assertEqual(qr.int_repr(), qr_cuda.int_repr())
                self.assertTrue(np.allclose(dqr, dqr_cuda.cpu()))

    def _test_quantize_per_channel(self, r, scales, zero_points, axis, float_params):
        # [FUNCTION DOC] HELPER: Checks per channel quantization of a 4D or 5D tensor along
        # the given axis for contiguous and channels-last (or 3d) layouts. Compares the
        # integer representation to a reference and the dequantized values to the input within
        # tolerance. Uses the flag float_params to choose the float zero point formula.

        def _quantize_per_channel_ref_nd(data, scales, zero_points, float_params):
            # [FUNCTION DOC] HELPER: Reference implementation: reshapes data to three dims
            # around the axis and loops over every element, quantizing with the per-channel
            # scale and zero point (multiply by inverse scale and add zero point when
            # float_params, else divide and then add), rounds and clips to 0..255, then
            # restores the original shape and returns a tensor.
            dims = data.size()
            data = data.view(-1, dims[axis], np.prod(dims[axis + 1:]))
            res = torch.empty_like(data)
            quant_min, quant_max = 0, 255
            for i in range(res.size()[0]):
                for j in range(res.size()[1]):
                    for k in range(res.size()[2]):
                        if float_params:
                            inv_scale = 1.0 / scales[j]
                            res[i][j][k] = np.clip(
                                np.round(data[i][j][k] * inv_scale + zero_points[j]), quant_min, quant_max)
                        else:
                            res[i][j][k] = np.clip(
                                np.round(data[i][j][k] / scales[j]) + zero_points[j], quant_min, quant_max)
            res = res.view(*dims)
            return res

        contig_format = torch.channels_last if r.ndim == 4 else torch.channels_last_3d
        for memory_format in [torch.contiguous_format, contig_format]:
            ref_res = _quantize_per_channel_ref_nd(r, scales, zero_points, float_params)
            r_contig = r.contiguous(memory_format=memory_format)
            qr = torch.quantize_per_channel(r_contig, scales, zero_points, axis, torch.quint8)
            rqr = qr.dequantize()
            self.assertTrue(np.allclose(qr.int_repr(), ref_res))
            self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))

    def test_qtensor_quantize_per_channel(self):
        # [FUNCTION DOC] Compares per channel quantization of a 3x2 tensor (axis one, double
        # scales, long zero points) with a loop reference and checks dequantization closeness;
        # then runs the shared helper on 4D and 5D tensors for axes one and zero.
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        scales = torch.tensor([0.2, 0.03], dtype=torch.double)
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        axis = 1

        def quantize_c(data, scales, zero_points):
            # [FUNCTION DOC] HELPER: Reference for the 3x2 case: for each element divides by
            # the channel scale, rounds, adds the channel zero point, clips to 0..255 and
            # returns a 3x2 float tensor.
            res = torch.empty((3, 2))
            quant_min, quant_max = 0, 255
            for i in range(3):
                for j in range(2):
                    res[i][j] = np.clip(np.round(data[i][j] / scales[j]) + zero_points[j], quant_min, quant_max)
            return res
        qr = torch.quantize_per_channel(r, scales, zero_points, axis, torch.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(qr.int_repr(), quantize_c(r, scales, zero_points)))
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))

        # Check 4D tensor with 2 different memory formats.
        r = torch.rand(3, 2, 4, 5, dtype=torch.float) * 4 - 2
        scales = torch.tensor([0.2, 0.03], dtype=torch.double)
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 1 , False)

        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.double)
        zero_points = torch.tensor([5, 10, 7], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 0, False)

        # Check 5D tensor.
        r = torch.rand(3, 2, 4, 5, 7, dtype=torch.float) * 4 - 2
        scales = torch.tensor([0.2, 0.03], dtype=torch.double)
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 1, False)

        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.double)
        zero_points = torch.tensor([5, 10, 7], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 0, False)

    def test_quantize_per_channel_float_qparams(self):
        # [FUNCTION DOC] Same checks as the integer qparam test but with float scales and
        # float zero points, using a reference with float zero points; also runs the shared
        # helper with the float flag for 4D and 5D tensors.
        r = torch.rand(3, 2, dtype=torch.float) * 4
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        axis = 1

        # Reference quantize function with FP zero_point.
        def quantize_ref(data, scales, zero_points):
            # [FUNCTION DOC] HELPER: Reference for float zero points: multiplies each
            # element by the inverse channel scale, adds the float zero point, rounds and
            # clips to 0..255, returns a 3x2 float tensor.
            res = torch.empty((3, 2))
            quant_min, quant_max = 0, 255
            for i in range(3):
                for j in range(2):
                    inv_scale = 1.0 / scales[j]
                    res[i][j] = np.clip(np.round(data[i][j] * inv_scale + zero_points[j]), quant_min, quant_max)
            return res

        qr = torch.quantize_per_channel(r, scales, zero_points, axis, torch.quint8)
        dequant_tensor = qr.dequantize()
        ref = quantize_ref(r, scales, zero_points)
        self.assertTrue(np.allclose(qr.int_repr(), ref))
        self.assertTrue(np.allclose(r.numpy(), dequant_tensor.numpy(), atol=1))

        # Check 4D tensor with 2 different memory formats.
        r = torch.rand(3, 2, 4, 5, dtype=torch.float) * 4
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 1, True)

        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2, 1.], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 0, True)

        # Check 5D tensor.
        r = torch.rand(3, 2, 4, 5, 7, dtype=torch.float) * 4 - 2
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 1, True)

        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2, 1.], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 0, True)

    def test_quantize_per_channel_sub_byte(self):
        # [FUNCTION DOC] Checks per channel quantization to quint4x2 with float scales and
        # zero points for a 3x2 input on axis zero and a 4D input on axis one, comparing the
        # packed bytes with a reference packer and, for the first case, dequantization
        # closeness.
        """ Tests the per channel quantization scheme for 4-bit qtensors.
        The scale and zero point for this have to be in floating point. """
        r = torch.rand(3, 2, dtype=torch.float) * 4
        scales = torch.tensor([0.2, 0.3, 0.1], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float)
        qr = torch.quantize_per_channel(r, scales, zero_points, 0, torch.quint4x2)
        dequant_tensor = qr.dequantize()

        def _get_qranges(bit_width):
            # [FUNCTION DOC] HELPER: Returns the quantization min and max (0 and 15) for
            # four bits; returns nothing (None) for any other bit width.
            if bit_width == 4:
                return 0, 15

        def _quantize_per_channel_sub_byte_ref(data, scales, zero_points, axis, bit_width):
            # [FUNCTION DOC] HELPER: Reference packer: reshapes data around the axis,
            # quantizes each element with the float scale and zero point, clips to the bit
            # range, and packs elements into bytes, first element in the low bits. Returns a
            # uint8 tensor of the packed bytes.
            dims = data.size()
            data = data.view(-1, dims[axis], np.prod(dims[axis + 1:]))
            qtensor_size = math.ceil(data.numel() / 2)
            res = torch.empty(qtensor_size, dtype=torch.uint8)
            elem_per_byte = 8 // bit_width
            quant_min, quant_max = _get_qranges(bit_width)
            for i in range(data.size()[0]):
                for j in range(data.size()[1]):
                    for k in range(data.size()[2]):
                        inv_scale = 1.0 / scales[j]
                        index = i * data.size()[1] * data.size()[2] + j * data.size()[2] + k
                        qvalue = np.clip(
                            np.round(data[i][j][k] * inv_scale + zero_points[j]), quant_min, quant_max).to(dtype=torch.int)
                        res_idx = int(index / elem_per_byte)
                        if (index % elem_per_byte == 0):
                            res[res_idx] = qvalue
                        else:
                            res[res_idx] |= (qvalue << ((index % elem_per_byte) * bit_width))
            return res

        ref_res = _quantize_per_channel_sub_byte_ref(r, scales, zero_points, 0, 4)
        self.assertTrue(np.allclose(qr.int_repr(), ref_res))
        self.assertTrue(np.allclose(r.numpy(), dequant_tensor.numpy(), atol=1 / np.min(scales.numpy())))

        # Check 4D tensor with non-zero axis.
        r = torch.rand(3, 2, 4, 5, dtype=torch.float) * 4
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        qr = torch.quantize_per_channel(r, scales, zero_points, axis=1, dtype=torch.quint4x2)
        ref_res = _quantize_per_channel_sub_byte_ref(r, scales, zero_points, 1, 4)
        self.assertTrue(np.allclose(qr.int_repr(), ref_res))

    def test_qtensor_permute(self):
        # [FUNCTION DOC] For each device and qint dtype, checks transpose then dequantize
        # against numpy transpose, permute versus transpose equality of integer data, scale,
        # zero point and dequantized values, contiguous after transpose, channels_last
        # conversion preserving data, and that permuting a larger 64 by 64 tensor works.
        scale = 0.02
        zero_point = 1
        for device in get_supported_device_types():
            r = torch.rand(10, 30, 2, 2, device=device, dtype=torch.float) * 4 - 2
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                qr = qr.transpose(0, 1)
                rqr = qr.dequantize()
                # compare transpose + dequantized result with original transposed result
                self.assertTrue(np.allclose(r.cpu().numpy().transpose([1, 0, 2, 3]), rqr.cpu().numpy(), atol=2 / scale))

                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                qr1 = qr.permute([1, 0, 2, 3])
                qr2 = qr.transpose(0, 1)
                # compare int representation after transformations
                self.assertEqual(qr1.int_repr(), qr2.int_repr())
                self.assertEqual(qr1.q_scale(), qr2.q_scale())
                self.assertEqual(qr1.q_zero_point(), qr2.q_zero_point())
                # compare dequantized result
                self.assertEqual(qr1.dequantize(), qr2.dequantize())
                # compare permuted + dequantized result with original transposed result
                self.assertTrue(np.allclose(qr2.dequantize().cpu().numpy(),
                                            r.cpu().numpy().transpose([1, 0, 2, 3]), atol=2 / scale))
                # make permuted result contiguous
                self.assertEqual(qr2.contiguous().int_repr(), qr2.int_repr())

                # change memory format
                qlast = qr.contiguous(memory_format=torch.channels_last)
                self.assertEqual(qr.stride(), sorted(qr.stride(), reverse=True))
                self.assertNotEqual(qlast.stride(), sorted(qlast.stride(), reverse=True))
                self.assertEqual(qr.int_repr(), qlast.int_repr())
                self.assertEqual(qr.q_scale(), qlast.q_scale())
                self.assertEqual(qr.q_zero_point(), qlast.q_zero_point())
                self.assertEqual(qlast.dequantize(), qr.dequantize())

                # permuting larger tensors
                x = torch.randn(64, 64, device=device)
                qx = torch.quantize_per_tensor(x, 1.0, 0, dtype)
                # should work
                qx.permute([1, 0])

    def test_qtensor_per_channel_permute(self):
        # [FUNCTION DOC] For each device checks that transposing a per-channel quantized
        # tensor across its axis raises RuntimeError, while converting to channels_last keeps
        # integer data, scales, zero points, axis one and dequantized values.
        for device in get_supported_device_types():
            r = torch.rand(20, 10, 2, 2, dtype=torch.float, device=device) * 4 - 2
            dtype = torch.qint8
            scales = torch.rand(10, device=device) * 0.02 + 0.01
            zero_points = torch.round(torch.rand(10, device=device) * 2 - 1).to(torch.long)
            qr = torch.quantize_per_channel(r, scales, zero_points, 1, dtype)

            # we can't reorder the axis
            with self.assertRaises(RuntimeError):
                qr.transpose(0, 1)

            # but we can change memory format
            qlast = qr.contiguous(memory_format=torch.channels_last)
            self.assertEqual(qr.stride(), sorted(qr.stride(), reverse=True))
            self.assertNotEqual(qlast.stride(), sorted(qlast.stride(), reverse=True))
            self.assertEqual(qr.int_repr(), qlast.int_repr())
            self.assertEqual(scales.to(dtype=torch.float64), qlast.q_per_channel_scales())
            self.assertEqual(zero_points, qlast.q_per_channel_zero_points())
            self.assertEqual(1, qlast.q_per_channel_axis())
            self.assertEqual(qlast.dequantize(), qr.dequantize())

    def test_qtensor_load_save(self):
        # [FUNCTION DOC] Saves a quantized tensor and a view of one of its columns together
        # to a temp file for each qint dtype, loads them with weights_only true and false, and
        # asserts equality and that both loaded tensors share one storage pointer. Uses cpu
        # only.
        scale = 0.2
        zero_point = 10
        # storage is not accessible on the cuda right now
        device = "cpu"
        r = torch.rand(15, 2, dtype=torch.float32, device=device) * 2
        for dtype in [torch.qint8, torch.quint8, torch.qint32]:
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
            qrv = qr[:, 1]
            with tempfile.NamedTemporaryFile() as f:
                # Serializing and Deserializing Tensor
                torch.save((qr, qrv), f)
                for weights_only in [True, False]:
                    f.seek(0)
                    qr2, qrv2 = torch.load(f, weights_only=weights_only)
                    self.assertEqual(qr, qr2)
                    self.assertEqual(qrv, qrv2)
                    self.assertEqual(qr2.storage().data_ptr(), qrv2.storage().data_ptr())

    def test_qtensor_per_channel_load_save(self):
        # [FUNCTION DOC] For quint8, qint8 and quint4x2 quantizes per channel, then saves to
        # a temp file and loads with weights_only true and false, asserting equality; quint4x2
        # uses float zero points of ones.
        r = torch.rand(20, 10, dtype=torch.float) * 4 - 2
        scales = torch.rand(10, dtype=torch.double) * 0.02 + 0.01
        zero_points = torch.round(torch.rand(10) * 20 + 1).to(torch.long)
        # quint32, cuda is not supported yet
        for dtype in [torch.quint8, torch.qint8, torch.quint4x2]:
            if dtype == torch.quint4x2:
                zero_points = torch.ones(10, dtype=torch.float)
            qr = torch.quantize_per_channel(r, scales, zero_points, 1, dtype)
            with tempfile.NamedTemporaryFile() as f:
                # Serializing and Deserializing Tensor
                torch.save(qr, f)
                for weights_only in [True, False]:
                    f.seek(0)
                    qr2 = torch.load(f, weights_only=weights_only)
                    self.assertEqual(qr, qr2)

    def test_qtensor_copy(self):
        # [FUNCTION DOC] For each qint dtype and device tests copy_ between quantized
        # tensors with equal and with different scale and zero point (the destination takes
        # the source parameters), asserts copying quantized into a float tensor raises
        # RuntimeError mentioning dequantize, and on cpu that copying float into quantized
        # equals quantizing the float.
        scale = 0.5
        zero_point = 10
        numel = 10
        for dtype in [torch.qint8, torch.quint8, torch.qint32]:
            for device in get_supported_device_types():
                # copy from same scale and zero_point
                q = torch._empty_affine_quantized([numel], scale=scale,
                                                  zero_point=zero_point, device=device, dtype=dtype)
                q2 = torch._empty_affine_quantized([numel], scale=scale,
                                                   zero_point=zero_point, device=device, dtype=dtype)
                q.copy_(q2)
                self.assertEqual(q.int_repr(), q2.int_repr())
                self.assertEqual(q.q_scale(), q2.q_scale())
                self.assertEqual(q.q_zero_point(), q2.q_zero_point())
                # copying from different scale and zero_point
                new_scale = 3.2
                new_zero_point = 5
                q = torch._empty_affine_quantized([numel], scale=new_scale,
                                                  zero_point=new_zero_point, device=device, dtype=dtype)
                # check original scale and zero_points are set correctly
                self.assertEqual(q.q_scale(), new_scale)
                self.assertEqual(q.q_zero_point(), new_zero_point)
                q.copy_(q2)
                # check scale and zero_points has been copied
                self.assertEqual(q, q2)
                # can't copy from quantized tensor to non-quantized tensor
                r = torch.empty([numel], dtype=torch.float)
                q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, "please use dequantize"):
                    r.copy_(q)
            # copy from float doesn't support cuda
            device = 'cpu'
            # check copy from non-quantized to quantized
            r = torch.randn([numel], dtype=torch.float, device=device)
            q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=dtype, device=device)
            q.copy_(r)
            qr = torch.quantize_per_tensor(r, scale=scale, zero_point=zero_point, dtype=dtype)
            self.assertEqual(q, qr)

    def test_torch_qtensor_deepcopy(self):
        # [FUNCTION DOC] Builds a per-tensor quantized tensor from a uint8 tensor on cpu,
        # deep copies it and asserts equality.
        # cuda is not supported yet
        device = "cpu"
        q_int = torch.randint(0, 100, [3, 5], device=device, dtype=torch.uint8)
        scale, zero_point = 2.0, 3
        q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
        qc = deepcopy(q)
        self.assertEqual(qc, q)

    def test_clone(self):
        # [FUNCTION DOC] For each device and qint dtype, creates empty per-tensor and per-
        # channel quantized tensors, clones them and asserts the clones equal the originals,
        # so scale and zero point are preserved.
        numel = 10
        scale = 0.5
        zero_point = 10

        options = itertools.product(
            get_supported_device_types(),
            [torch.qint8, torch.quint8, torch.qint32])

        for device, dtype in options:
            per_tensor_quantized = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point,
                device=device, dtype=dtype)
            per_channel_quantized = torch._empty_per_channel_affine_quantized(
                [numel],
                scales=torch.tensor([scale] * numel, device=device),
                zero_points=torch.tensor([zero_point] * numel, device=device),
                axis=0,
                device=device,
                dtype=dtype
            )
            qtensors = [per_tensor_quantized, per_channel_quantized]

            for q in qtensors:
                q2 = q.clone()
                # Check to make sure the scale and zero_point has been copied.
                self.assertEqual(q, q2)

    def test_qtensor_fill_per_tensor(self):
        # [FUNCTION DOC] For each qint dtype, fill value (including negative and overflow
        # values) and device, fills an empty per-tensor tensor and compares integer data and
        # dequantized values with quantizing a tensor of that value; also asserts scale and
        # zero point are unchanged.
        numel = 10
        scale = 0.5
        zero_point = 10

        ones = torch.ones(numel).to(torch.float)

        qtypes = [torch.qint8, torch.quint8, torch.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow

        devices = get_supported_device_types()
        for qtype, val2fill, device in itertools.product(qtypes, vals2fill, devices):
            ones = ones.to(device)
            q_filled = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = torch.quantize_per_tensor(ones * val2fill, scale,
                                              zero_point, qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_scale(), scale)
            self.assertEqual(q_filled.q_zero_point(), zero_point)

    # Adapted from test_qtensor_fill_per_tensor but for a NHWC tensor (requires 4D)
    def test_qtensor_fill_per_tensor_nhwc(self):
        # [FUNCTION DOC] Same as the per-tensor fill test but on a random 4D shape for both
        # contiguous and channels_last memory formats.
        dims = torch.randint(low=1, high=10, size=(4, )).tolist()
        scale = 0.5
        zero_point = 10

        ones = torch.ones(dims).to(torch.float)

        qtypes = [torch.qint8, torch.quint8, torch.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow
        memory_formats = [torch.contiguous_format, torch.channels_last]
        devices = get_supported_device_types()
        for qtype, val2fill, memory_format, device in itertools.product(qtypes, vals2fill, memory_formats, devices):
            q_filled = torch._empty_affine_quantized(
                dims, scale=scale, zero_point=zero_point, device=device,
                dtype=qtype, memory_format=memory_format)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = torch.quantize_per_tensor(ones * val2fill, scale,
                                              zero_point, qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_scale(), scale)
            self.assertEqual(q_filled.q_zero_point(), zero_point)

    # adapted from test_qtensor_fill_per_tensor
    def test_qtensor_fill_per_channel(self):
        # [FUNCTION DOC] Fill test for per-channel tensors: fills empty per-channel
        # quantized tensors with several values and compares against per-channel quantizing a
        # constant tensor, and checks scales and zero points are unchanged.
        dims = [4, 5]
        axis = 0
        # adding a constant to avoid too small of a scale
        scales = torch.rand(dims[axis], dtype=torch.float64) + 0.1
        zero_points = torch.randint(low=0, high=10, size=(dims[axis], ))

        ones = torch.ones(dims).to(torch.float)

        qtypes = [torch.qint8, torch.quint8, torch.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow

        devices = get_supported_device_types()
        for qtype, val2fill, device in itertools.product(qtypes, vals2fill, devices):
            scales = scales.to(device)
            zero_points = zero_points.to(device)
            ones = ones.to(device)
            q_filled = torch._empty_per_channel_affine_quantized(
                dims, scales=scales, zero_points=zero_points, device=device,
                axis=axis, dtype=qtype)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = torch.quantize_per_channel(ones * val2fill, scales=scales,
                                               zero_points=zero_points, axis=axis, dtype=qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_per_channel_scales(), scales)
            self.assertEqual(q_filled.q_per_channel_zero_points(), zero_points)

    def test_qtensor_masked_fill_cpu(self):
        # [FUNCTION DOC] Runs the masked fill helper on cpu.
        self._test_qtensor_masked_fill('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_masked_fill_cuda(self):
        # [FUNCTION DOC] CUDA test: runs the masked fill helper on cuda; skipped without a
        # gpu.
        self._test_qtensor_masked_fill('cuda')

    # adapted from test_qtensor_fill_per_tensor
    def _test_qtensor_masked_fill(self, device):
        # [FUNCTION DOC] HELPER: Checks masked_fill with a true scalar mask equals fill and
        # the quantized reference, with scale and zero point unchanged; then for a random
        # boolean mask over the tensor compares masked_fill with elementwise assignment for
        # each dtype and value.
        numel = 10
        scale = 0.5
        zero_point = 10

        ones = torch.ones(numel, dtype=torch.float, device=device)

        types = [torch.qint8, torch.quint8, torch.qint32]
        fills = [-1, 1, 2**32]  # positive, negative, overflow

        for qtype, fill_with in itertools.product(types, fills):
            q_filled = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            q_filled.fill_(fill_with)
            q_masked_fill = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            # mask fill the whole tensor, equivalent to calling plain vanilla fill
            mask = torch.tensor(True, device=device)
            q_masked_fill.masked_fill_(mask, fill_with)
            int_repr = torch.quantize_per_tensor(ones * fill_with, scale,
                                                 zero_point, qtype)
            fill_with = int_repr.dequantize()
            int_repr = int_repr.int_repr()

            self.assertEqual(q_filled, q_masked_fill)
            self.assertEqual(q_masked_fill.int_repr(), int_repr)
            self.assertEqual(q_masked_fill.dequantize(), fill_with)
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_masked_fill.q_scale(), scale)
            self.assertEqual(q_masked_fill.q_zero_point(), zero_point)

        # the above loop does the same test as test_qtensor_fill
        # now we will check masked_fill for subset of indices
        mask = torch.randint(0, 2, (numel, ), device=device)
        mask = mask.bool()
        x = torch.rand(numel, device=device)
        for qtype, fill_with in itertools.product(types, fills):
            qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)
            q_masked_fill = qx.clone()
            q_masked_fill.masked_fill_(mask, fill_with)
            ref = qx.clone()

            for i in range(numel):
                if mask[i]:
                    # this assignment doesn't end up calling masked_fill, allowing us to compare the different implementations
                    ref[i] = torch.tensor([fill_with], device=device, dtype=torch.float)

            self.assertEqual(q_masked_fill, ref)
            self.assertEqual(q_masked_fill.int_repr(), ref.int_repr())
            self.assertEqual(q_masked_fill.dequantize(), ref.dequantize())

    def test_qtensor_index_put_cpu(self):
        # [FUNCTION DOC] Runs the index_put helpers (plain and deterministic non-accumulate)
        # on cpu.
        self._test_qtensor_index_put('cpu')
        self._test_qtensor_index_put_non_accumulate_deterministic('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_index_put_cuda(self):
        # [FUNCTION DOC] CUDA test: runs both index_put helpers on cuda; skipped without a
        # gpu.
        self._test_qtensor_index_put('cuda')
        self._test_qtensor_index_put_non_accumulate_deterministic('cuda')

    def _test_qtensor_index_put(self, device):
        # [FUNCTION DOC] HELPER: For each qint dtype and for a scalar value and a vector of
        # random values, checks that assigning through index tuples into a quantized tensor
        # equals quantizing the float tensor after the same assignment.
        n = 10
        m = 10
        x_orig = torch.rand(n, m, device=device)
        indices = tuple(torch.tensor([[0, 0], [1, 1], [5, 5], [7, 3], [0, 5], [6, 9], [-1, -1]], device=device).t())
        # for the scalar tensor case, index_put routes to masked_fill
        values_list = [torch.tensor(2.5, device=device), torch.rand(len(indices[0]), device=device) * 1000]
        scale = 0.5
        zero_point = 10
        types = [torch.qint8, torch.quint8, torch.qint32]
        for qtype, values in itertools.product(types, values_list):
            x_ref = x_orig.clone()
            x_ref[indices] = values.to(dtype=x_ref.dtype)
            qx_ref = torch.quantize_per_tensor(x_ref, scale=scale, zero_point=zero_point, dtype=qtype)

            x = x_orig.clone()
            qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)
            qx[indices] = values

            self.assertEqual(qx_ref, qx)

    def _test_qtensor_index_put_non_accumulate_deterministic(self, device):
        # [FUNCTION DOC] HELPER: With deterministic algorithms enabled, for each qint dtype
        # repeats three times a non-accumulating index_put on a quantized tensor with many
        # repeated indices and checks it equals quantizing the float index_put output.
        with DeterministicGuard(True):
            scale = 0.5
            zero_point = 10
            types = [torch.qint8, torch.quint8, torch.qint32]
            for qtype in types:
                for _ in range(3):
                    m = random.randint(10, 20)
                    elems = random.randint(20000, 30000)
                    values = torch.rand(elems, device=device)
                    indices = torch.randint(m, (elems,), device=device)
                    x_orig = torch.rand(m, device=device)

                    x = x_orig.clone()
                    qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)
                    output = qx.index_put((indices,), values, accumulate=False)


                    x_ref = x_orig.clone()
                    output_ref = x_ref.index_put((indices,), values, accumulate=False)
                    qx_ref = torch.quantize_per_tensor(output_ref, scale=scale, zero_point=zero_point, dtype=qtype)

                    self.assertEqual(output, qx_ref)

    # adapted from test_qtensor_fill_per_channel and test_qtensor_fill_per_tensor_nhwc
    def test_qtensor_fill_per_channel_nhwc(self):
        # [FUNCTION DOC] Per-channel fill test on a random 4D shape for contiguous and
        # channels_last memory formats, comparing fill results with quantizing a constant
        # tensor and checking parameters stay unchanged.
        dims = torch.randint(low=1, high=10, size=(4, )).tolist()
        axis = 0
        # adding a constant to avoid too small of a scale
        scales = torch.rand(dims[axis], dtype=torch.float64) + 0.1
        zero_points = torch.randint(low=0, high=10, size=(dims[axis], ))

        ones = torch.ones(dims).to(torch.float)

        qtypes = [torch.qint8, torch.quint8, torch.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow
        memory_formats = [torch.contiguous_format, torch.channels_last]
        devices = get_supported_device_types()
        for qtype, val2fill, memory_format, device in itertools.product(qtypes, vals2fill, memory_formats, devices):
            scales = scales.to(device)
            zero_points = zero_points.to(device)
            ones = ones.to(device)
            q_filled = torch._empty_per_channel_affine_quantized(
                dims, scales=scales, zero_points=zero_points, device=device,
                axis=axis, dtype=qtype, memory_format=memory_format)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = torch.quantize_per_channel(ones * val2fill, scales=scales,
                                               zero_points=zero_points, axis=axis, dtype=qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_per_channel_scales(), scales)
            self.assertEqual(q_filled.q_per_channel_zero_points(), zero_points)

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_index_select_cuda(self):
        # [FUNCTION DOC] CUDA test: runs the index_select helper on cuda; skipped without a
        # gpu.
        self._test_qtensor_index_select('cuda')

    def test_qtensor_index_select_cpu(self):
        # [FUNCTION DOC] Runs the index_select helper on cpu.
        self._test_qtensor_index_select('cpu')

    def _test_qtensor_index_select(self, device):
        # [FUNCTION DOC] HELPER: For quint8 and qint8 checks that selecting indices along a
        # random dimension of a quantized tensor equals quantizing the selected float values.
        for quant_type in [torch.quint8, torch.qint8]:
            dims = 3
            index = torch.randint(dims, [1]).item()
            selected = torch.randperm(dims)[:2].to(device)
            scale = 1
            zp = 0
            x = torch.randn([3] * dims, device=device) * 10

            x_selected = torch.index_select(x, index, selected)
            x_selected_quantized = torch.quantize_per_tensor(x_selected, scale, zp, quant_type)

            x_quantized = torch.quantize_per_tensor(x, scale, zp, quant_type)
            x_quantized_selected = torch.index_select(x_quantized, index, selected)

            self.assertEqual(x_quantized_selected, x_selected_quantized)

    def test_qtensor_view(self):
        # [FUNCTION DOC] Tests view on quantized tensors: shape change keeps element count,
        # a negative dimension infers, view after transpose changes strides and data so the
        # results differ, and viewing a non-contiguous tensor raises RuntimeError about
        # incompatible size and stride while the contiguous version works.
        scale, zero_point, dtype = 1.0, 2, torch.uint8
        for device in get_supported_device_types():
            q_int = torch.randint(0, 100, [1, 2, 3], device=device, dtype=dtype)
            q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            q2 = q.view(1, 3, 2)
            self.assertEqual(q.numel(), q2.numel())
            # testing -1
            self.assertEqual(q, q2.view(1, -1, 3))

            a_int = torch.randint(0, 100, [1, 2, 3, 4], device=device, dtype=dtype)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = a.view(1, 3, 2, 4)  # does not change tensor layout in memory
            self.assertEqual(b.size(), c.size())
            self.assertEqual(b.q_scale(), c.q_scale())
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            self.assertNotEqual(b.stride(), c.stride())
            # size is the same but the underlying data is different
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # torch.equal is not supported for the cuda backend
            if device == 'cpu':
                self.assertFalse(torch.equal(b, c))

            # a case can't view non-contiguous Tensor
            a_int = torch.randint(0, 100, [1, 2, 3, 4], device=device, dtype=dtype)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            err_str = "view size is not compatible with input tensor's size and stride*"
            with self.assertRaisesRegex(RuntimeError, err_str):
                b.view(1, 4, 2, 3)
            # view on contiguous tensor is fine
            b.contiguous().view(1, 4, 2, 3)

    def test_qtensor_resize(self):
        # [FUNCTION DOC] Tests resize on quantized tensors: chains resizes between several
        # shapes with the same element count, compares transposed versus resized original,
        # asserts that resizing to a different element count raises RuntimeError, and that
        # resizing contiguous and non-contiguous tensors is allowed.
        for device in get_supported_device_types():
            scale, zero_point, dtype = 1.0, 2, torch.uint8
            sizes1 = [1, 2, 3, 4]
            sizes2 = [1 * 2, 3 * 4]
            sizes3 = [1, 2 * 3, 4]
            sizes4 = [1 * 2 * 3 * 4]
            sizes5 = [1, 2, 1, 3, 1, 4]

            q1_int = torch.randint(0, 100, sizes1, dtype=dtype, device=device)
            q1 = torch._make_per_tensor_quantized_tensor(q1_int, scale=scale, zero_point=zero_point)
            q2 = q1.resize(*sizes2)
            q3 = q2.resize(*sizes3)
            q4 = q3.resize(*sizes4)
            q5 = q4.resize(*sizes5)

            self.assertEqual(q1.numel(), q2.numel())
            self.assertEqual(q1.numel(), q3.numel())
            self.assertEqual(q1.numel(), q4.numel())
            self.assertEqual(q1.numel(), q5.numel())

            # Compare original and post-transpose
            a_int = torch.randint(0, 100, sizes1, dtype=dtype, device=device)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = b.resize(*sizes1)  # Change the sizes back to the original

            self.assertEqual(a.size(), c.size())
            self.assertEqual(b.q_scale(), c.q_scale())
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            self.assertNotEqual(b.stride(), c.stride())
            # size is the same but the underlying data is different
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # torch.equal is not supported for the cuda backend
            if device == 'cpu':
                self.assertFalse(torch.equal(b, c))

            # Throws an error if numel is wrong
            q1_int = torch.randint(0, 100, sizes1, dtype=dtype, device=device)
            q1 = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            err_str = "requested resize to*"
            with self.assertRaisesRegex(RuntimeError, err_str):
                q2 = q1.resize(*sizes1[:-1])
            # resize on both contiguous and non-contiguous tensor should be fine
            q3 = q1.resize(*sizes2)
            q4 = q1.contiguous().resize(*sizes2)

    def test_qtensor_reshape(self):
        # [FUNCTION DOC] Tests reshape on quantized tensors: flattening and inferring a
        # dimension keeps equality, reshape after transpose differs from the layout-preserving
        # reshape, and reshaping a non-contiguous tensor succeeds.
        scale, zero_point, dtype = 1.0, 2, torch.uint8
        for device in get_supported_device_types():
            q_int = torch.randint(0, 100, [3, 5], dtype=dtype, device=device)
            q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            q2 = q.reshape([15])
            self.assertEqual(q.numel(), q2.numel())
            self.assertEqual(q2.size(), [15])
            # testing -1
            self.assertEqual(q, q2.reshape([3, -1]))

            a_int = torch.randint(0, 100, [1, 2, 3, 4], dtype=dtype, device=device)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = a.reshape(1, 3, 2, 4)  # does not change tensor layout
            self.assertEqual(b.size(), c.size())
            self.assertEqual(b.q_scale(), c.q_scale())
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            self.assertNotEqual(b.stride(), c.stride())
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # torch.equal is not supported for the cuda backend
            if device == 'cpu':
                self.assertFalse(torch.equal(b, c))

            # we can use reshape for non-contiguous Tensor
            a_int = torch.randint(0, 100, [1, 2, 3, 4], dtype=dtype, device=device)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = b.reshape(1, 4, 2, 3)

    def test_qtensor_unsqueeze(self):
        # [FUNCTION DOC] Tests unsqueeze and squeeze on per-tensor and per-channel quantized
        # tensors, including the axis being shifted by unsqueeze, and that squeezing the
        # channel axis raises RuntimeError saying squeeze is only possible on a non-axis
        # dimension for per-channel; also squeeze without a dim argument.
        for device in get_supported_device_types():
            x = torch.randn((1, 3, 4), device=device)
            qx = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)
            qy = qx.unsqueeze(2)
            self.assertEqual(qy.size(), (1, 3, 1, 4))
            qy = qy.squeeze(2)
            self.assertEqual(qy.size(), qx.size())

            # Per channel qtensor
            scales = torch.tensor([1.0], device=device)
            zero_points = torch.tensor([0], device=device)
            qx = torch.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=torch.quint8, axis=0)
            qy = qx.unsqueeze(0)
            self.assertEqual(qy.size(), (1, 1, 3, 4))
            self.assertEqual(qy.q_per_channel_axis(), 1)

            qz = qy.squeeze(0)
            self.assertEqual(qz.size(), x.size())
            self.assertEqual(qz.q_per_channel_axis(), 0)
            with self.assertRaisesRegex(RuntimeError, "Squeeze is only possible on non-axis dimension for Per-Channel"):
                qz = qy.squeeze(1)

            # squeeze without dim specified
            x = torch.randn((3, 1, 2, 1, 4), device=device)
            scales = torch.tensor([1.0, 1.0], device=device)
            zero_points = torch.tensor([0, 0], device=device)
            qx = torch.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=torch.quint8, axis=2)
            qz = qx.squeeze()
            self.assertEqual(qz.size(), (3, 2, 4))
            self.assertEqual(qz.q_per_channel_axis(), 1)
            with self.assertRaisesRegex(RuntimeError, "Squeeze is only possible on non-axis dimension for Per-Channel"):
                qz = qy.squeeze()

    def test_repeat(self):
        # [FUNCTION DOC] For each device, repeats a quantized tensor and asserts equality
        # with a quantized tensor built from the repeated integer data.
        scale, zero_point, dtype = 1.0, 2, torch.uint8
        for device in get_supported_device_types():
            q_int = torch.randint(0, 100, [3], dtype=dtype, device=device)
            q_int_repeat = q_int.repeat(4, 2)
            q_ref = torch._make_per_tensor_quantized_tensor(q_int_repeat, scale=scale, zero_point=zero_point)

            q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            q_repeat = q.repeat(4, 2)
            self.assertEqual(q_ref, q_repeat)

    def test_qscheme_pickle(self):
        # [FUNCTION DOC] Saves a Foo module to an in-memory buffer, loads it back with
        # weights_only false and asserts its qscheme attribute is per_tensor_symmetric.
        f = Foo()
        buf = io.BytesIO()
        torch.save(f, buf)

        buf.seek(0)
        # weights_only=False as this is legacy code that saves the model
        f2 = torch.load(buf, weights_only=False)

        self.assertEqual(f2.qscheme, torch.per_tensor_symmetric)

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=2, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           reduce_range=st.booleans()
           )
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_choose_qparams(self, X, reduce_range):
        # [FUNCTION DOC] Skipped test (hypothesis based): compares the chosen per-tensor
        # qparams from the library function with the numpy reference, scale to three decimals
        # and zero point exactly.
        X, (scale, zero_point, torch_type) = X
        X = torch.from_numpy(X)
        X_scale, X_zp = _calculate_dynamic_qparams(X, torch.quint8, reduce_range=reduce_range)
        qparams = torch._choose_qparams_per_tensor(X, reduce_range)
        np.testing.assert_array_almost_equal(X_scale, qparams[0], decimal=3)
        self.assertEqual(X_zp, qparams[1])

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_quantization_does_not_pin_memory(self):
        # [FUNCTION DOC] CUDA test for a regression about issue 41115: creating a quantized
        # tensor on cuda must not make later cpu allocations pinned; checks is_pinned is false
        # before and after.
        # Context - https://github.com/pytorch/pytorch/issues/41115
        x = torch.randn(3)
        self.assertEqual(x.is_pinned(), False)

        q_int = torch.randint(0, 100, [1, 2, 3], device="cuda", dtype=torch.uint8)
        q = torch._make_per_tensor_quantized_tensor(q_int, scale=0.1, zero_point=0)

        x = torch.randn(3)
        self.assertEqual(x.is_pinned(), False)

    # There's no way to actually pin the memory of a quantized tensor
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_quant_pin_memory(self):
        # [FUNCTION DOC] CUDA test: a pinned float tensor quantized is not pinned,
        # empty_quantized with pin memory true is also not pinned, and calling pin_memory on a
        # quantized tensor raises RuntimeError.
        x = torch.randn(3).pin_memory()
        self.assertEqual(x.is_pinned(), True)
        x_q = torch.quantize_per_tensor(x, 1, 0, torch.quint8)
        self.assertEqual(x_q.is_pinned(), False)
        x_pin = torch.empty_quantized([3], x_q, pin_memory=True, dtype=torch.quint8)
        self.assertEqual(x_pin.is_pinned(), False)
        self.assertRaises(RuntimeError, lambda: x_q.pin_memory())

    def test_fp16_saturate_op(self):
        # [FUNCTION DOC] Checks the saturate-to-fp16 op clamps values of plus and minus
        # 65532 to the float16 limits plus and minus 65504.
        x = torch.ones(5, 5, dtype=torch.float32) * 65532
        x[0] = torch.ones(5) * -65532
        # range of fp16 value is [-65504, + 65504]
        ref = torch.ones(5, 5) * 65504
        ref[0] = torch.ones(5) * -65504
        y = torch._saturate_weight_to_fp16(x)
        self.assertEqual(y, ref)

    def test_choose_qparams_optimized(self):
        # [FUNCTION DOC] For bit widths four and two compares the optimized qparam chooser
        # on a random vector with the numpy greedy reference; also asserts it raises
        # ValueError for an empty input saying the tensor has no data.
        for bit_width in [4, 2]:
            x = torch.randn(64, dtype=torch.float)
            y = torch.choose_qparams_optimized(x, numel=64, n_bins=200, ratio=0.16, bit_width=bit_width)
            ref = param_search_greedy(x.numpy(), bit_rate=bit_width)
            self.assertEqual(y[0].numpy(), ref[0])
            self.assertEqual(y[1].numpy(), ref[1])

        with self.assertRaisesRegex(ValueError, "input tensor is empty and has no data"):
            torch.choose_qparams_optimized(torch.tensor([]), numel=0, n_bins=200, ratio=0.16, bit_width=8)

    def _test_pickle_checkpoint_qtensor(self, device):
        # [FUNCTION DOC] HELPER: Inside a temporary file name, defines a scripted module
        # that saves its two inputs to that file path during forward, runs it with a per-
        # tensor and a per-channel quantized tensor, then loads the file with weights_only
        # true and false and asserts equality with the originals.
        with TemporaryFileName() as fname:
            class M(torch.jit.ScriptModule):
                # [CLASS DOC] HELPER: Script module with the file name as a constant;
                # forward saves both tensor inputs to that file and returns the second.
                __constants__ = ['fname']

                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Stores the captured temp file name in the fname
                    # attribute.
                    super().__init__()
                    self.fname = fname

                @torch.jit.script_method
                def forward(self, x, y):
                    # [FUNCTION DOC] HELPER: Scripted method saving the pair of inputs to
                    # the stored file name and returning y.
                    torch.save((x, y), self.fname)
                    return y

            q = torch.quantize_per_tensor(
                torch.rand(2, 3, dtype=torch.float), scale=0.1, zero_point=10, dtype=torch.quint8).to(device)
            qc = torch.quantize_per_channel(
                torch.rand(2, 3, dtype=torch.float),
                scales=torch.tensor([0.1, 0.5, 0.01]),
                zero_points=torch.tensor([10, 0, 20]),
                axis=1, dtype=torch.quint8).to(device)
            m = M()
            m(q, qc)
            with open(fname, "rb") as handle:
                for weights_only in [True, False]:
                    loaded_q, loaded_qc = torch.load(fname, weights_only=weights_only)
                    self.assertEqual(loaded_q, q)
                    self.assertEqual(loaded_qc, qc)

    def test_pickle_checkpoint_qtensor(self):
        # [FUNCTION DOC] Runs the pickle checkpoint helper on cpu.
        self._test_pickle_checkpoint_qtensor('cpu')

    def test_jit_serialization(self):
        # [FUNCTION DOC] For per-tensor and per-channel cases builds a scripted module
        # holding a quantized buffer, saves it with jit save into a buffer, loads it and
        # asserts calling the loaded model equals the original output.
        class SimpleQTensor(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module that stores a quantized random 5x5 tensor
            # (per tensor or per channel according to its constructor flag) as a buffer and
            # returns it in forward.
            def __init__(self, per_channel):
                # [FUNCTION DOC] HELPER: Creates a random float matrix, quantizes it per
                # tensor or per channel depending on the flag and registers it as the buffer
                # x.
                super().__init__()
                x = torch.rand(5, 5).float()
                if not per_channel:
                    x_q = torch.quantize_per_tensor(x, 0.2, 10, torch.quint8)
                else:
                    s = torch.rand(5, dtype=torch.float64) + 0.1
                    zp = torch.randint(5, 15, (5,))
                    x_q = torch.quantize_per_channel(x, s, zp, 1, torch.quint8)
                self.x = torch.nn.Buffer(x_q)

            @torch.jit.script_method
            def forward(self):
                # [FUNCTION DOC] HELPER: Scripted method returning the stored quantized
                # buffer.
                return self.x

        for per_channel in [False, True]:
            model = SimpleQTensor(per_channel)
            buffer = io.BytesIO()
            torch.jit.save(model, buffer)
            buffer.seek(0)
            model_loaded = torch.jit.load(buffer)
            self.assertEqual(model_loaded(), model())

    def test_bfp16_quantize(self):
        # [FUNCTION DOC] Converts a random tensor to bfloat16 and back to float32 and
        # asserts closeness with loose tolerances.
        X = torch.randn(5 , 10)
        quantized_X = X.to(torch.bfloat16)
        dedequantized_X = quantized_X.to(torch.float32)
        torch.testing.assert_close(X, dedequantized_X, rtol=1e-4, atol=5e-3)

    def test_decomposed_quantize_per_tensor(self):
        # [FUNCTION DOC] Imports the decomposed op module to register ops, then for quint8,
        # qint8 and qint32 compares the decomposed per-tensor quantize op with the regular
        # quantize: output dtype and integer data must match.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        X = torch.randn(5, 10)
        test_cases = [
            (torch.quint8, torch.uint8, 0, 255),
            (torch.qint8, torch.int8, -128, 127),
            (torch.qint32, torch.int32, -2**31, 2**31 - 1),
        ]
        for qdtype, dtype, quant_min, quant_max in test_cases:
            scale, zero_point = _calculate_dynamic_qparams(X, qdtype)
            quantized_X = torch.quantize_per_tensor(X, scale, zero_point, qdtype)
            quantized_decomposed_X = \
                torch.ops.quantized_decomposed.quantize_per_tensor(
                    X, scale, zero_point, quant_min, quant_max, dtype)
            self.assertEqual(quantized_decomposed_X.dtype, dtype)
            self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_quantize_per_tensor_bfloat16_input(self):
        # [FUNCTION DOC] Checks the decomposed per-tensor quantize op accepts bfloat16 input
        # and gives uint8 output equal to the integer data of regular quantization of the
        # float32 input.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        X = torch.randint(1, 10, (5, 5)).to(torch.float32)
        scale, zero_point = _calculate_dynamic_qparams(X, torch.quint8)
        quantized_X = torch.quantize_per_tensor(X, scale, zero_point, torch.quint8)
        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_tensor(
                X.to(torch.bfloat16), scale, zero_point, 0, 255, torch.uint8)
        self.assertEqual(quantized_decomposed_X.dtype, torch.uint8)
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_dequantize_per_tensor(self):
        # [FUNCTION DOC] For three quantized dtypes compares decomposed quantize and
        # dequantize ops with the regular ones: integer data and dequantized values must
        # match.
        import torch.ao.quantization.fx._decomposed
        X = torch.randn(5, 10)
        test_cases = [
            (torch.quint8, torch.uint8, 0, 255),
            (torch.qint8, torch.int8, -128, 127),
            (torch.qint32, torch.int32, -2**31, 2**31 - 1),
        ]

        for qdtype, dtype, quant_min, quant_max in test_cases:
            scale, zero_point = _calculate_dynamic_qparams(X, qdtype)
            quantized_X = torch.quantize_per_tensor(X, scale, zero_point, qdtype)
            dequantized_X = torch.dequantize(quantized_X)

            quantized_decomposed_X = torch.ops.quantized_decomposed.quantize_per_tensor(
                X, scale, zero_point, quant_min, quant_max, dtype)
            dequantized_decomposed_X = torch.ops.quantized_decomposed.dequantize_per_tensor(
                quantized_decomposed_X, scale, zero_point, quant_min, quant_max, dtype
            )
            self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
            self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_dynamic_quant_pattern(self):
        # [FUNCTION DOC] Compares the decomposed dynamic quantization pattern (choose
        # qparams, quantize and dequantize tensor ops) with the regular dynamic path for
        # uint8, checking integer data and dequantized values.
        import torch.ao.quantization.fx._decomposed
        X = torch.randn(5, 10)
        dtype = torch.uint8
        qdtype = torch.quint8
        scale, zero_point = torch._choose_qparams_per_tensor(X, False)
        quant_min, quant_max = 0, 255

        quantized_X = torch.quantize_per_tensor(X, scale, zero_point, qdtype)
        dequantized_X = torch.dequantize(quantized_X)

        # Now try decomposed pattern
        (scale_decomposed, zero_point_decomposed) = torch.ops.quantized_decomposed.choose_qparams.tensor(
            X, quant_min, quant_max, torch.Tensor([torch.finfo(torch.float32).eps]), dtype)
        quantized_decomposed_X = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
            X, scale_decomposed, zero_point_decomposed, quant_min, quant_max, dtype)

        dequantized_decomposed_X = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            quantized_decomposed_X, scale_decomposed, zero_point_decomposed, quant_min, quant_max, dtype
        )
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
        self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_quantize_per_channel(self):
        # [FUNCTION DOC] Compares decomposed per-channel quantize with regular per-channel
        # quantization for uint8 on axis zero: dtype and integer data.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        X = torch.randn(5, 10)
        qdtype = torch.quint8
        dtype = torch.uint8
        scales = torch.randn(5,)
        zero_points = torch.randint(0, 100, (5,))
        quant_min, quant_max = 0, 255
        axis = 0

        quantized_X = torch.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_channel(
                X, scales, zero_points, axis, quant_min, quant_max, dtype)
        self.assertEqual(quantized_decomposed_X.dtype, dtype)
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_quantize_per_channel_bfloat16_input(self):
        # [FUNCTION DOC] Same as the per-channel decomposed test but the decomposed op
        # receives bfloat16 input.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        X = torch.randint(1, 10, (5, 5)).to(torch.float32)
        qdtype = torch.quint8
        dtype = torch.uint8
        scales = torch.randn(5,)
        zero_points = torch.randint(0, 100, (5,))
        quant_min, quant_max = 0, 255
        axis = 0

        quantized_X = torch.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_channel(
                X.to(torch.bfloat16), scales, zero_points, axis, quant_min, quant_max, dtype)
        self.assertEqual(quantized_decomposed_X.dtype, dtype)
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_dequantize_per_channel(self):
        # [FUNCTION DOC] Compares decomposed per-channel quantize and dequantize with the
        # regular ones for uint8: integer data and dequantized values.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        X = torch.randn(5, 10)
        qdtype = torch.quint8
        dtype = torch.uint8
        scales = torch.randn(5,)
        zero_points = torch.randint(0, 100, (5,))
        quant_min, quant_max = 0, 255
        axis = 0

        quantized_X = torch.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        dequantized_X = torch.dequantize(quantized_X)

        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_channel(
                X, scales, zero_points, axis, quant_min, quant_max, dtype)
        dequantized_decomposed_X = \
            torch.ops.quantized_decomposed.dequantize_per_channel(
                quantized_decomposed_X, scales, zero_points, axis, quant_min, quant_max, dtype)

        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
        self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_choose_qparams_per_token_asymmetric_backward(self):
        # [FUNCTION DOC] Computes per-token asymmetric qparams via the decomposed
        # implementation on an input that requires grad, applies the affine transform and
        # rounding and runs backward, checking only that no error is raised.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        x = torch.randn(2, 3).requires_grad_()
        (s, zp) = torch.ops.quantized_decomposed._choose_qparams_per_token_asymmetric_impl(x, torch.int8)
        out = x.div(s).add(zp).round()
        out.sum().backward()

    def test_decomposed_quantize_per_channel_group(self):
        # [FUNCTION DOC] Quantizes and dequantizes a 100x256 tensor with the decomposed per-
        # channel-group ops (group size 128, int8 values in minus eight to seven) and asserts
        # exact equality with fake quantize per channel affine applied to the grouped reshape.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        qmin, qmax = (-8, 7)
        group_size = 128
        x = torch.randn(100, 256)
        s = torch.randn(100, 2)
        zp = torch.randint(qmax, size=(100, 2), dtype=torch.int32)

        # simulate fake quantize per channel group with qdq
        q = torch.ops.quantized_decomposed.quantize_per_channel_group(
            x, s, zp, qmin, qmax, torch.int8, group_size,
        )
        dq = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            q, s, zp, qmin, qmax, torch.int8, group_size, torch.float32
        )

        # express per group fake quant using `torch.fake_quantize_per_channel_affine`
        x_grouped = x.reshape(-1, group_size)
        s_flattened = s.flatten()
        zp_flattened = zp.flatten()
        fq = torch.fake_quantize_per_channel_affine(
            x_grouped, s_flattened, zp_flattened, 0, qmin, qmax,
        )
        fq = fq.reshape_as(x)
        torch.testing.assert_close(dq, fq, rtol=0, atol=0)

    def test_decomposed_quantize_per_token(self):
        # [FUNCTION DOC] Same as the group test but with the decomposed per-token ops and
        # per-row scales and zero points, compared with fake quantize per channel affine over
        # rows with zero tolerance.
        # register the ops
        import torch.ao.quantization.fx._decomposed
        qmin, qmax = (-8, 7)
        x = torch.randn(100, 256)
        s = torch.randn(100, 1)
        zp = torch.randint(qmax, size=(100, 1), dtype=torch.int32)

        # simulate fake quantize per token with qdq
        q = torch.ops.quantized_decomposed.quantize_per_token(
            x, s, zp, qmin, qmax, torch.int8,
        )
        dq = torch.ops.quantized_decomposed.dequantize_per_token(
            q, s, zp, qmin, qmax, torch.int8, torch.float32
        )

        # express per token fake quant using `torch.fake_quantize_per_channel_affine`
        s_flattened = s.flatten()
        zp_flattened = zp.flatten()
        fq = torch.fake_quantize_per_channel_affine(
            x, s_flattened, zp_flattened, 0, qmin, qmax,
        )
        torch.testing.assert_close(dq, fq, rtol=0, atol=0)


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
