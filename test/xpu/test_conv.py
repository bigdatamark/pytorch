# Owner(s): ["module: intel"]
# [MODULE DOC] XPU (Intel GPU) convolution tests, adapted from the CUDA convolution tests. One
# device-type test class checks forward and backward behavior of convolution and transposed
# convolution modules and functional calls: depthwise and grouped equivalences, same and valid
# padding versus explicit padding and scipy, double backward, empty and non-contiguous inputs,
# very large inputs, channels-last layouts, misaligned memory, low-level aten backward ops and
# the oneDNN tf32 flag. At import time it reads the scipy availability flag (importing scipy
# submodules if present), computes a module constant for Ampere/ROCm support, and instantiates
# the test class for the xpu device only (with xpu allowed). Runs the tests when executed as a
# script.

import copy
import itertools
import math
import unittest
from itertools import product

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch._C._dynamo.guards import assert_size_stride
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyXPU,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_nn import _test_module_empty_input, NNTestCase
from torch.testing._internal.common_utils import (
    dtype2prec_DONTUSE,
    gradcheck,
    gradgradcheck,
    parametrize as parametrize_test,
    run_tests,
    set_default_dtype,
    TEST_SCIPY,
    TEST_WITH_ROCM,
)


AMPERE_OR_ROCM = TEST_WITH_ROCM or torch.cuda.is_tf32_supported()
if TEST_SCIPY:
    import scipy.ndimage
    import scipy.signal


class TestConvolutionNNDeviceType(NNTestCase):
    # [CLASS DOC] Device-type test case (derived from the NN test base class) holding the
    # convolution tests for xpu. Tests receive a device string and, where decorated, a dtype; a
    # few helper methods build the reference computations. Some tests still use CUDA-flavored
    # names and calls (cudnn flags, CUDA device-type checks) inherited from the originals.
    def run_conv_double_back_test(
        self,
        kern,
        stride,
        padding,
        chan_in,
        chan_out,
        batch_size,
        inp_size,
        dilation,
        no_weight,
        groups=1,
        use_xpu=False,
        use_bias=True,
        dtype=torch.double,
    ):
        # [FUNCTION DOC] HELPER: Builds a 2D convolution problem and exercises its double
        # backward. Creates a random input of batch by chan_in channels by a square inp_size, a
        # random weight of shape chan_out by chan_in divided by groups by kern by kern
        # (requiring grad unless no_weight is true) and an optional bias, all on xpu if use_xpu
        # else cpu in the given dtype (default double). If dtype is float32, returns whether the
        # first-order input gradient (built with create_graph) requires grad; otherwise returns
        # the result of gradgradcheck on the functional convolution, which raises on mismatch
        # and returns True on success.
        device = torch.device("xpu" if use_xpu else "cpu")
        x = torch.randn(
            batch_size,
            chan_in,
            inp_size,
            inp_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.randn(
            chan_out,
            chan_in // groups,
            kern,
            kern,
            device=device,
            dtype=dtype,
            requires_grad=not no_weight,
        )
        if use_bias:
            bias = torch.randn(chan_out, device=device, dtype=dtype, requires_grad=True)
        else:
            bias = None

        def func(*inputs):
            # [FUNCTION DOC] HELPER: Nested function under test for gradient checking: takes the
            # inputs (input, weight and, when use_bias is set, bias), and returns the functional
            # 2D convolution using the enclosing stride, padding, dilation and groups.
            if use_bias:
                lx, lweight, lbias = inputs
            else:
                lx, lweight = inputs
                lbias = None
            out = F.conv2d(lx, lweight, lbias, stride, padding, dilation, groups)
            return out

        if use_bias:
            inputs = x, weight, bias
        else:
            inputs = x, weight

        dummy_out = func(*inputs)
        grad_y = torch.randn_like(
            dummy_out, device=device, dtype=dtype, requires_grad=True
        )

        if dtype == torch.float:
            (g,) = torch.autograd.grad(dummy_out.sum(), x, create_graph=True)
            return g.requires_grad

        return gradgradcheck(func, inputs, (grad_y,))

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_Conv2d_large_workspace(self, device, dtype):
        # [FUNCTION DOC] Runs forward and backward of a 256 to 256 channel 3x3 convolution
        # (padding 1) for three large input sizes, for floating dtypes including half and
        # bfloat16. Only checks that nothing crashes or runs out of workspace; there are no
        # value assertions.
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        def run_test(benchmark):
            # [FUNCTION DOC] HELPER: Builds the convolution on the device, then for each size
            # runs forward on a fresh requires-grad copy and backward with ones. Returns
            # nothing.
            # KNOWN ISSUE: the benchmark argument is never used, so the two calls (false and
            # true) run the identical computation.
            conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to(device, dtype)
            for size in sizes:
                x = torch.randn(size, device=device, dtype=dtype)
                out = conv(x.detach().clone().requires_grad_())
                out.backward(torch.ones_like(out))

        run_test(benchmark=False)
        run_test(benchmark=True)

    @dtypes(torch.half, torch.float)
    def test_ConvTranspose2d_large_output_padding(self, device, dtype):
        # [FUNCTION DOC] Chains three transposed 2D convolutions (128 to 64 to 32 to 3 channels,
        # kernel 3, stride 2, padding 1, output padding 1) on a small random input in half or
        # float, then runs backward with random gradient. A crash/shape smoke test with no value
        # assertions.
        net1 = torch.nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net2 = torch.nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net3 = torch.nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        x = torch.rand(1, 128, 6, 6, device=device, dtype=dtype, requires_grad=True)
        x = net1(x)
        x = net2(x)
        x = net3(x)
        x.backward(torch.randn_like(x))

    @dtypes(torch.float, torch.double, torch.half)
    def test_Conv2d_depthwise_naive_groups(self, device, dtype):
        # [FUNCTION DOC] Checks that a grouped (depthwise, groups equal to 2) 2D convolution
        # equals two independent single-channel convolutions whose weights and biases are slices
        # of the grouped layer, for depth multipliers 1 and 2. Skips half precision on xpu
        # (known oneDNN accuracy issue). Compares concatenated outputs, input gradient, bias
        # gradient and weight gradient, with tolerance from the per-dtype precision table and
        # zero relative tolerance.
        if dtype == torch.half and "xpu" in device:
            self.skipTest(
                "The accuracy issue of dtype fp16 would be fixed in oneDNN v3.4"
            )
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            i = (
                torch.randn(2, 2, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            output = m(i)
            grad_output = (
                torch.randn(2, 2 * depth_multiplier, 4, 4, device=device, dtype=dtype)
                / 2
            )
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(
                output,
                torch.cat([output1, output2], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

    @dtypes(torch.float, torch.double, torch.half)
    def test_Conv3d_depthwise_naive_groups(self, device, dtype):
        # [FUNCTION DOC] 3D counterpart of the depthwise grouped-convolution equivalence test:
        # grouped convolution with 2 groups versus two single-channel convolutions, depth
        # multipliers 1 and 2, skipping half on xpu. Output and weight gradient use absolute
        # tolerance 3e-4 and relative 3e-2; input and bias gradients use the per-dtype table
        # with zero relative tolerance.
        if dtype == torch.half and "xpu" in device:
            self.skipTest(
                "The accuracy issue of dtype fp16 would be fixed in oneDNN v3.4"
            )
        for depth_multiplier in [1, 2]:
            m = nn.Conv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            i = (
                torch.randn(2, 2, 6, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            output = m(i)
            grad_output = (
                torch.randn(
                    2, 2 * depth_multiplier, 4, 4, 4, device=device, dtype=dtype
                )
                / 2
            )
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())
            atol, rtol = (3e-4, 3e-2)

            self.assertEqual(
                output, torch.cat([output1, output2], 1), atol=atol, rtol=rtol
            )
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=atol,
                rtol=rtol,
            )

    @dtypes(torch.float, torch.double, torch.half)
    def test_noncontig_conv_grad(self, device, dtype):
        # [FUNCTION DOC] Verifies that backward gives the same input gradient for a non-
        # contiguous output gradient (a slice of a larger tensor) as for its contiguous copy.
        # Raises an assertion error if the constructed gradient happens to be contiguous. Runs
        # backward twice with retained graph, zeroing the stored input gradient between runs.
        module = nn.Conv2d(3, 5, kernel_size=3, padding=1).to(device, dtype)
        input = torch.randn(
            2, 3, 10, 10, dtype=dtype, device=device, requires_grad=True
        )
        output = module(input)

        grad = torch.randn(2, 2, 5, 10, 10, dtype=dtype, device=device)[:, 1]
        if grad.is_contiguous():
            raise AssertionError("Expected grad to not be contiguous")
        output.backward(grad, retain_graph=True)
        self.assertIsNotNone(input.grad)
        result = input.grad.data.clone()
        input.grad.data.zero_()

        output.backward(grad.contiguous())
        self.assertEqual(
            result, input.grad.data, atol=dtype2prec_DONTUSE[dtype], rtol=0
        )

    @dtypes(torch.double)
    def test_conv_double_backward(self, device, dtype):
        # [FUNCTION DOC] Double-precision double backward check run on xpu under cudnn flags
        # (enabled, deterministic). Loops over kernel/input-size/dilation combinations (kernel 3
        # with size 5 and dilations 1 and 2, kernel 4 with size 9 and dilation 1) with stride 1,
        # padding 2, 2 input and 3 output channels, calls the double-backward helper and asserts
        # it succeeded. The no-weight flag is always false because the stride is always 1.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            batch_size = 1
            for kern, inp_size, dilations in [(3, 5, [1, 2]), (4, 9, [1])]:
                for stride, padding, chan_in, chan_out, dilation in product(
                    [1], [2], [2], [3], dilations
                ):
                    no_weight = stride == 2
                    result = self.run_conv_double_back_test(
                        kern,
                        stride,
                        padding,
                        chan_in,
                        chan_out,
                        batch_size,
                        inp_size,
                        dilation,
                        no_weight,
                        use_xpu=True,
                        dtype=dtype,
                    )
                    self.assertTrue(result, "Conv double backward test failed")

    def test_conv_double_backward_no_bias(self):
        # [FUNCTION DOC] Double backward check on the CPU with stride 2, kernel 3, padding 1, 2
        # to 4 channels, batch 2, input size 5, using the default double dtype. Asserts the
        # helper succeeded.
        # KNOWN ISSUE: despite the name, the helper is called with the bias enabled, so the no-
        # bias path is not exercised; also runs on CPU.
        kern, stride = 3, 2
        chan_in, chan_out = 2, 4
        batch_size, inp_size = 2, 5
        padding, dilation = 1, 1
        no_weight, use_bias = False, True
        result = self.run_conv_double_back_test(
            kern,
            stride,
            padding,
            chan_in,
            chan_out,
            batch_size,
            inp_size,
            dilation,
            no_weight,
            use_bias=use_bias,
        )
        self.assertTrue(result, "Conv double backward test failed")

    def test_conv_double_backward_groups(self):
        # [FUNCTION DOC] Double backward check on the CPU with groups equal to 2 (channels
        # scaled by the group count), kernel 3, stride 1, padding 2, batch 2, input size 6.
        # Asserts the helper succeeded.
        kern, stride, padding = 3, 1, 2
        chan_in, chan_out = 2, 4
        batch_size, inp_size, dilation = 2, 6, 1
        no_weight = False
        groups = 2
        result = self.run_conv_double_back_test(
            kern,
            stride,
            padding,
            chan_in * groups,
            chan_out * groups,
            batch_size,
            inp_size,
            dilation,
            no_weight,
            groups=groups,
        )
        self.assertTrue(result, "Conv double backward test failed")

    def test_conv_double_backward_stride(self):
        # [FUNCTION DOC] Runs the double-backward helper on the CPU for stride 2 over kernels of
        # 3 with input sizes 5 (dilations 1 and 2) and 7 (dilation 1), paddings 0 and 1, 1 to 2
        # channels, batch 2.
        # KNOWN ISSUE: the result returned by the helper is discarded and never asserted, so
        # only failures that raise inside the helper are detected.
        batch_size = 2
        for kern, inp_size, dilations in [(3, 5, [1, 2]), (3, 7, [1])]:
            for stride, padding, chan_in, chan_out, dilation in product(
                [2], [0, 1], [1], [2], dilations
            ):
                no_weight = False
                self.run_conv_double_back_test(
                    kern,
                    stride,
                    padding,
                    chan_in,
                    chan_out,
                    batch_size,
                    inp_size,
                    dilation,
                    no_weight,
                )

    @dtypes(torch.float)
    def test_conv1d_large_input(self, device, dtype):
        # [FUNCTION DOC] Compares a large 1D convolution (batch 4, 512 to 512 channels, length
        # 441, kernel 3, padding 1, bias) computed on CPU in float32 with the same weights
        # computed on the device in the test dtype. Seeds the generator with 42; asserts outputs
        # match with tolerance 1e-5 ignoring device differences.
        N, C_in, L = 4, 512, 441
        C_out, K, P = 512, 3, 1
        torch.manual_seed(42)

        conv_cpu = (
            nn.Conv1d(C_in, C_out, kernel_size=K, padding=P, bias=True)
            .to(torch.float32)
            .requires_grad_()
        )
        x_cpu = torch.randn(N, C_in, L, dtype=torch.float32)
        out_cpu = conv_cpu(x_cpu)

        conv_dev = nn.Conv1d(C_in, C_out, kernel_size=K, padding=P, bias=True).to(
            device, dtype
        )
        conv_dev.weight.data.copy_(conv_cpu.weight.data.to(dtype))
        conv_dev.bias.data.copy_(conv_cpu.bias.data.to(dtype))

        x_dev = x_cpu.to(device, dtype).requires_grad_()
        out_dev = conv_dev(x_dev)

        self.assertEqual(out_cpu, out_dev, atol=1e-5, rtol=1e-5, exact_device=False)

    @dtypes(torch.float)
    def test_conv1d_same_padding(self, device, dtype):
        # [FUNCTION DOC] Checks 1D same-padding convolution. First, for input sizes 50 to 54,
        # kernel sizes 1, 2, 3 and 8, dilations 1 to 3 and stride 1, asserts the output length
        # is the ceiling of input size over stride. Then compares same padding with explicit
        # numeric padding for kernel 3 (padding 1), kernel 4 with dilation 2 (padding 3), and
        # dilation 3 (padding 5 with the first output element dropped, since the total padding
        # is asymmetric).
        test_args = [
            range(50, 55),
            [1, 2, 3, 8],
            range(1, 4),
            [1],
        ]
        for in_size, k_size, dilation, stride in itertools.product(*test_args):
            x = torch.rand(1, 1, in_size, device=device, dtype=dtype)
            y = torch.rand(1, 1, k_size, device=device, dtype=dtype)
            z = F.conv1d(x, y, padding="same", dilation=dilation, stride=stride)
            self.assertEqual(z.size(2), int(math.ceil(in_size / stride)))

        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 3, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=1)
        actual = F.conv1d(x, y, padding="same")
        self.assertEqual(expect, actual)

        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 4, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=3, dilation=2)
        actual = F.conv1d(x, y, padding="same", dilation=2)
        self.assertEqual(expect, actual)

        expect = F.conv1d(x, y, padding=5, dilation=3)[..., 1:]
        actual = F.conv1d(x, y, padding="same", dilation=3)
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv3d_same_padding(self, device, dtype):
        # [FUNCTION DOC] Checks 3D same-padding convolution against explicit padding: kernel
        # 1x2x5 (padding 0,1,2, dropping the first row of the middle output dimension), the same
        # with dilation 2 (padding 0,1,4), and kernel 4x4x4 with dilation 3 (padding 5, dropping
        # the first element of each spatial dimension). Uses default comparison tolerances.
        rtol, atol = None, None
        x = torch.rand(1, 1, 10, 11, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 1, 2, 5, device=device, dtype=dtype)
        expect = F.conv3d(x, y, padding=(0, 1, 2))[..., :, 1:, :]
        actual = F.conv3d(x, y, padding="same")
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        expect = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        actual = F.conv3d(x, y, padding="same", dilation=2)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        y = torch.rand(1, 1, 4, 4, 4, device=device, dtype=dtype)
        expect = F.conv3d(x, y, padding=5, dilation=3)[..., 1:, 1:, 1:]
        actual = F.conv3d(x, y, padding="same", dilation=3)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

    @dtypes(torch.float)
    def test_conv1d_valid_padding(self, device, dtype):
        # [FUNCTION DOC] Asserts that 1D convolution with the string valid padding gives the
        # same result as default (no) padding.
        x = torch.rand(1, 1, 10, device=device, dtype=dtype)
        y = torch.rand(1, 1, 4, device=device, dtype=dtype)
        expect = F.conv1d(x, y)
        actual = F.conv1d(x, y, padding="valid")
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv2d_valid_padding(self, device, dtype):
        # [FUNCTION DOC] Asserts that 2D convolution with the string valid padding equals the
        # default unpadded convolution for a 1x1x1x10 input and a 1x1x1x4 kernel.
        x = torch.rand(1, 1, 1, 10, device=device, dtype=dtype)
        y = torch.rand(1, 1, 1, 4, device=device, dtype=dtype)
        expect = F.conv2d(x, y)
        actual = F.conv2d(x, y, padding="valid")
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv3d_valid_padding(self, device, dtype):
        # [FUNCTION DOC] Asserts that 3D convolution with the string valid padding equals the
        # default unpadded convolution on a small tensor with singleton leading spatial
        # dimensions.
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device)
        expect = F.conv3d(x, y)
        actual = F.conv3d(x, y, padding="valid")
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv1d_same_padding_backward(self, device, dtype):
        # [FUNCTION DOC] Compares input and weight gradients of 1D convolution with same padding
        # against those of an equivalent explicit-padding convolution (padding 3 with dilation
        # 2; then padding 2 with the first output element dropped) using sum-abs of the output
        # as the loss. Resets the stored gradients between runs.
        x = torch.rand(1, 1, 12, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, dtype=dtype, device=device, requires_grad=True)

        z = F.conv1d(x, y, padding=3, dilation=2)
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding=2)[..., 1:]
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding="same")
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    @dtypes(torch.float)
    def test_conv2d_same_padding_backward(self, device, dtype):
        # [FUNCTION DOC] 2D version of the same-padding gradient comparison: kernel 4x5 with
        # dilation 2 against explicit padding of 3 by 4, and kernel 4x4 against padding 2 with the
        # first row and column of the output dropped. Compares input and weight gradients.
        x = torch.rand(1, 1, 10, 11, device=device, dtype=dtype, requires_grad=True)
        y = torch.rand(1, 1, 4, 5, device=device, dtype=dtype, requires_grad=True)

        z = F.conv2d(x, y, padding=(3, 4), dilation=2)
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv2d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        y = torch.rand(1, 1, 4, 4, device=device, dtype=dtype, requires_grad=True)
        z = F.conv2d(x, y, padding=2)[..., 1:, 1:]
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv2d(x, y, padding="same")
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    @dtypes(torch.double)
    def test_conv3d_same_padding_backward(self, device, dtype):
        # [FUNCTION DOC] 3D same-padding backward test in double precision. Compares input and
        # weight gradients with explicit padding for dilation 2 (kernel 1x2x5) and for kernel
        # 1x4x4 (padding 2, output cropped), and additionally runs gradcheck (with forward-mode
        # AD and nondeterminism tolerance 1e-5) and gradgradcheck (forward over reverse) on
        # same-padding convolution for both kernels.
        x = torch.rand(1, 1, 1, 11, 12, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 2, 5, dtype=dtype, device=device, requires_grad=True)
        z = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
            (x, y),
            check_forward_ad=True,
            nondet_tol=1e-5,
        )
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
            (x, y),
            check_fwd_over_rev=True,
        )

        y = torch.rand(1, 1, 1, 4, 4, dtype=dtype, device=device, requires_grad=True)
        z = F.conv3d(x, y, padding=2)[..., 1:, 1:]
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding="same")
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same"),
            (x, y),
            check_forward_ad=True,
            nondet_tol=1e-5,
        )
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="same"),
            (x, y),
            check_fwd_over_rev=True,
        )

    @dtypes(torch.float)
    def test_conv1d_valid_padding_backward(self, device, dtype):
        # [FUNCTION DOC] Compares input and weight gradients of 1D convolution with valid
        # padding against those with explicit zero padding, using sum-abs of the output as the
        # loss.
        x = torch.rand(1, 1, 10, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, dtype=dtype, device=device, requires_grad=True)
        F.conv1d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None
        F.conv1d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv1d_vs_scipy(self, device, dtype, mode):
        # [FUNCTION DOC] Skipped without scipy. Parametrized over valid and same mode. Compares
        # 1D torch convolution with scipy signal convolution for an even-length (size 4) and odd-
        # length (size 5) kernel on a random 1x10 input, running under the float default dtype.
        t = make_tensor((1, 10), device=device, dtype=dtype)
        feat_dim = t.shape[1]
        weight_even = make_tensor((1, 1, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # [FUNCTION DOC] HELPER: Computes the scipy convolution of the flattened input and
            # kernel as the expected result, then the torch result using the flipped kernel
            # (torch computes correlation). For same mode it pads the input manually by half the
            # kernel size on both sides, uses no padding argument, and crops the result to the
            # input length; for valid mode it passes the mode as the padding string. Asserts
            # closeness with 2e-5 tolerances.
            t_a = t.view(-1).cpu().numpy()
            w_a = weight.view(-1).cpu().numpy()
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            if mode == "same":
                p = weight.shape[2] // 2
                t = torch.nn.functional.pad(t, (p, p))
                kwargs.pop("padding")

            weight_flipped = torch.flip(weight, (2,))
            actual = torch.nn.functional.conv1d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                actual = actual[:feat_dim]

            self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)

        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv2d_vs_scipy(self, device, dtype, mode):
        # [FUNCTION DOC] Skipped without scipy. Parametrized over valid and same mode. Compares
        # 2D torch convolution to scipy 2D convolution for a 1x5x10 input with kernels 2x4
        # (even) and 3x5 (odd), under float default dtype.
        t = make_tensor((1, 5, 10), device=device, dtype=dtype)
        weight_even = make_tensor((1, 1, 2, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 3, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # [FUNCTION DOC] HELPER: Computes the scipy 2D convolution of the squeezed input and
            # kernel, then the torch result with the kernel flipped in both spatial dimensions.
            # For same mode it pads by half the kernel size in each dimension manually and crops
            # the result to 5 by 10; otherwise passes the mode as padding. Compares with rtol
            # 2e-5 and atol 5e-6.
            t_a = t.squeeze(0).cpu().numpy()
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            expected = scipy.signal.convolve2d(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            if mode == "same":
                left_right_pad = weight.shape[3] // 2
                top_bottom_pad = weight.shape[2] // 2
                p = (left_right_pad, left_right_pad, top_bottom_pad, top_bottom_pad)
                t = torch.nn.functional.pad(t, p)
                kwargs.pop("padding")

            weight_flipped = torch.flip(weight, (2, 3))
            actual = torch.nn.functional.conv2d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                actual = actual[:5, :10]

            self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv3d_vs_scipy(self, device, dtype, mode):
        # [FUNCTION DOC] Skipped without scipy. Parametrized over valid and same mode. Compares
        # 3D torch convolution to scipy N-dimensional convolution for a 1x5x5x10 input and
        # kernels 2x2x4 and 2x3x5, under float default dtype.
        t = make_tensor((1, 5, 5, 10), device=device, dtype=dtype)
        weight_even = make_tensor((1, 1, 2, 2, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 2, 3, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # [FUNCTION DOC] HELPER: Computes the scipy convolution of the squeezed input and
            # kernel, then the torch result with the kernel flipped in all three spatial
            # dimensions. For same mode it pads manually by half the kernel size in each
            # dimension and crops the result to 5 by 5 by 10; otherwise passes the mode as
            # padding. Compares with rtol 2e-5 and atol 5e-6.
            t_a = t.squeeze(0).cpu().numpy()
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)
            kwargs = {"padding": mode}
            if mode == "same":
                left_right_pad = weight.shape[4] // 2
                top_bottom_pad = weight.shape[3] // 2
                front_back_pad = weight.shape[2] // 2
                p = (
                    left_right_pad,
                    left_right_pad,
                    top_bottom_pad,
                    top_bottom_pad,
                    front_back_pad,
                    front_back_pad,
                )
                t = torch.nn.functional.pad(t, p)
                kwargs.pop("padding")
            weight_flipped = torch.flip(weight, (2, 3, 4))
            actual = torch.nn.functional.conv3d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                actual = actual[:5, :5, :10]
            self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @dtypes(torch.float)
    def test_conv2d_valid_padding_backward(self, device, dtype):
        # [FUNCTION DOC] Compares input and weight gradients of 2D convolution with valid
        # padding against explicit zero padding, using sum-abs of the output as the loss.
        x = torch.rand(1, 1, 1, 10, device=device, dtype=dtype, requires_grad=True)
        y = torch.rand(1, 1, 1, 4, device=device, dtype=dtype, requires_grad=True)
        F.conv2d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None
        F.conv2d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    @dtypes(torch.double)
    def test_conv3d_valid_padding_backward(self, device, dtype):
        # [FUNCTION DOC] Double-precision gradient comparison of 3D valid padding against
        # explicit zero padding, followed by gradcheck (forward AD enabled) and gradgradcheck
        # (forward over reverse) on valid-padding 3D convolution.
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device, requires_grad=True)
        F.conv3d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv3d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_forward_ad=True,
        )
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_fwd_over_rev=True,
        )

    @parametrize_test("N", range(2, 4), name_fn=lambda N: f"ConvTranspose{N}d")
    def test_conv_transpose_with_output_size_and_no_batch_dim(self, device, N):
        # [FUNCTION DOC] Parametrized over 2 and 3 spatial dimensions (test names mention
        # ConvTranspose2d/3d). Applies a transposed convolution (one channel, kernel 16, stride
        # 16, padding 7, no bias) to an input without a batch dimension while passing an
        # explicit output size of 240 by 200 (plus another 200 for 3D), and asserts the output
        # shape equals that size.
        inp = torch.randn((1, 15, 13) if N == 2 else (1, 15, 13, 13), device=device)
        output_size = (1, 240, 200) if N == 2 else (1, 240, 200, 200)
        ConvTransposeNd = getattr(nn, f"ConvTranspose{N}d")
        m = ConvTransposeNd(
            1, 1, kernel_size=16, stride=16, padding=7, bias=False, device=device
        )
        output = m(inp, output_size=output_size)
        self.assertEqual(output.shape, output_size)

    @dtypes(torch.float)
    def test_conv_empty_channel(self, device, dtype):
        # [FUNCTION DOC] Checks convolutions with zero input channels in 1D, 2D and 3D: a module
        # with zero input channels run on an empty input must pass the shared empty-input check
        # (forward, backward and zero gradients), while an input with a zero-sized spatial
        # dimension but one channel must raise a RuntimeError whose message mentions the
        # groups/weight mismatch.
        in_channels = 0
        mod = torch.nn.Conv1d(in_channels, 8, 2, stride=2, dtype=dtype).to(device)
        inp = torch.randn(2, 0, 15, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 0, device=device, dtype=dtype)
            mod(inp)

        mod = torch.nn.Conv2d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        inp = torch.randn(2, 0, 50, 100, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 40, 0, device=device, dtype=dtype)
            mod(inp)

        mod = torch.nn.Conv3d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        inp = torch.randn(2, 0, 50, 20, 40, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 50, 0, 40, device=device, dtype=dtype)
            mod(inp)

    def test_group_conv_empty(self, device):
        # [FUNCTION DOC] Runs a grouped (4 groups) strided 2D convolution on an input with batch
        # size zero and checks it with the shared empty-input helper (forward, backward and zero
        # gradients, without size check).
        mod = torch.nn.Conv2d(4, 4, stride=2, kernel_size=3, padding=1, groups=4).to(
            device
        )
        inp = torch.randn(0, 4, 4, 4, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    def test_group_convTranspose_empty(self, device):
        # [FUNCTION DOC] Runs a grouped (4 groups) strided 2D transposed convolution on a zero-
        # batch input through the shared empty-input check.
        mod = torch.nn.ConvTranspose2d(
            4, 4, stride=2, kernel_size=3, padding=1, groups=4
        ).to(device)
        inp = torch.randn(0, 4, 4, 4, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    def test_convTranspose_empty(self, device):
        # [FUNCTION DOC] Runs a strided 2D transposed convolution on a zero-batch input through
        # the shared empty-input check.
        mod = torch.nn.ConvTranspose2d(4, 4, stride=2, kernel_size=3, padding=1).to(
            device
        )
        inp = torch.randn(0, 4, 4, 4, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    def test_conv_large_nosplit(self, device):
        # [FUNCTION DOC] Smoke test in half precision: runs forward of a 2 to 2 channel
        # convolution (kernel 8, stride 8) on a very large input (one by two by 1024 by about a
        # million) and of a 1 to 1024 channel pointwise convolution on a 2048 by 1024 input. No
        # assertions; checks that large launches do not fail.
        dtype = torch.half
        conv1 = nn.Conv2d(2, 2, 8, 8).to(device).to(dtype)
        input_large = torch.randn(1, 2, 1024, 1024 * 1024, dtype=dtype, device=device)
        conv1(input_large)
        conv2 = torch.nn.Conv2d(1, 1024, 1, 1).to(device).to(dtype)
        input_large = torch.randn(1, 1, 2048, 1024, dtype=dtype, device=device)
        conv2(input_large)

    def test_conv_noncontig_weights(self, device):
        # [FUNCTION DOC] For 1D, 2D and 3D, with and without grouping, builds an expanded (non-
        # contiguous) weight and runs the functional convolution and transposed convolution
        # forward and backward. Smoke test: only checks that non-contiguous weights are
        # accepted, no value assertions.
        for dim in (1, 2, 3):
            for grouped in (False, True):
                nc = 3
                groups = 3 if grouped else 1
                w = torch.randn([3] * dim, device=device)
                w = w.expand([nc, int(nc / groups)] + list(w.shape))
                w = w.detach().requires_grad_()
                x = torch.randn(
                    [1, nc] + ([5] * dim), device=device, requires_grad=True
                )
                y = getattr(F, f"conv{dim}d")(x, w, groups=groups)
                y.sum().backward()
                y = getattr(F, f"conv_transpose{dim}d")(x, w, groups=groups)
                y.sum().backward()

    def test_conv_noncontig_weights_and_bias(self, device):
        # [FUNCTION DOC] With and without bias, runs a 3 to 64 channel 7x7 strided convolution
        # with non-contiguous input, weight and bias (taken as slices of larger tensors) and
        # again with their contiguous copies, and asserts the two outputs are equal.
        for bias in [True, False]:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=bias).to(
                device, torch.float
            )
            input_nc = torch.randn(
                (1, 3, 224, 224, 2), device=device, dtype=torch.float
            )[:, :, :, :, 1]
            input_c = input_nc.contiguous()
            weight_nc = torch.randn((64, 3, 7, 7, 2), device=device, dtype=torch.float)[
                :, :, :, :, 1
            ]
            conv1.weight = nn.Parameter(weight_nc)
            weight_c = conv1.weight.contiguous()
            if bias:
                bias_nc = torch.randn((64, 2), device=device, dtype=torch.float)[:, 1]
                conv1.bias = nn.Parameter(bias_nc)
                bias_c = conv1.bias.contiguous()
            out1 = conv1(input_nc)
            conv1.weight = nn.Parameter(weight_c)
            if bias:
                conv1.bias = nn.Parameter(bias_c)
            out2 = conv1(input_c)
            self.assertEqual(out1, out2)

    def test_conv_transposed_large(self, device):
        # [FUNCTION DOC] Applies a 1 to 1 transposed convolution to a very large batch (4096)
        # and compares each of four 1024-sample slices of the result with the convolution of the
        # matching input slice, requiring maximum absolute difference zero. Uses half precision
        # on CUDA device type and float otherwise.
        dtype = torch.half if self.device_type == "cuda" else torch.float
        conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False).to(device).to(dtype)
        input_large = torch.randn(4096, 1, 512, 1024, dtype=dtype, device=device)
        ret = conv(input_large)
        maxdiff0 = (
            (ret.narrow(0, 0, 1024) - conv(input_large.narrow(0, 0, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff1 = (
            (ret.narrow(0, 1024, 1024) - conv(input_large.narrow(0, 1024, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff2 = (
            (ret.narrow(0, 2048, 1024) - conv(input_large.narrow(0, 2048, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff3 = (
            (ret.narrow(0, 3072, 1024) - conv(input_large.narrow(0, 3072, 1024)))
            .abs_()
            .max()
            .item()
        )
        self.assertEqual(maxdiff0, 0)
        self.assertEqual(maxdiff1, 0)
        self.assertEqual(maxdiff2, 0)
        self.assertEqual(maxdiff3, 0)

    def test_conv_large(self, device):
        # [FUNCTION DOC] Checks that a convolution (2 to 2 channels, kernel 8, stride 8) on a
        # batch of 4097 gives the same output as when the batch is processed in three chunks,
        # and that weight gradients accumulated from the whole batch match those from the chunks
        # (after normalizing by mean magnitude) within tolerance 5e-2 absolute and 5e-3
        # relative.
        dtype = torch.half if self.device_type == "cuda" else torch.float
        conv = nn.Conv2d(2, 2, 8, 8, bias=False).to(device).to(dtype)
        input_large = torch.randn(4097, 2, 512, 512, dtype=dtype, device=device)
        ret = conv(input_large)
        self.assertEqual(ret[:2048], conv(input_large[:2048]))
        self.assertEqual(ret[2048:4096], conv(input_large[2048:4096]))
        self.assertEqual(ret[4096:], conv(input_large[4096:]))

        conv.zero_grad()
        ret.view(4097, -1).max(dim=1).values.sum().backward()
        del ret
        grad1 = conv.weight.grad.detach().clone()
        conv.zero_grad()
        conv(input_large[:2048]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[2048:4096]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[4096:]).view(1, -1).max(dim=1).values.sum().backward()
        grad2 = conv.weight.grad.detach().clone()
        scale = 1 / grad2.abs().mean()
        grad1 = grad1 * scale
        grad2 = grad2 * scale
        self.assertEqual(grad1, grad2, atol=5e-2, rtol=5e-3)

    def test_Conv2d_size_1_kernel(self, device):
        # [FUNCTION DOC] Compares a 3-channel 1x1 convolution on CPU with the device version
        # (cudnn disabled for the device part), copying weights and bias, using a shared random
        # gradient. Asserts outputs and bias/weight gradients are close (atol 1e-5, devices
        # ignored).
        x_cpu = torch.randn(2, 3, 5, 5)
        conv_cpu = torch.nn.Conv2d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)

        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.Conv2d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )

    def test_ConvTranspose2d_size_1_kernel(self, device):
        # [FUNCTION DOC] Compares a 3-channel 1x1 transposed 2D convolution on CPU with the
        # device version using copied weights and bias and a shared random output gradient;
        # asserts the output and bias and weight gradients are close (atol 1e-5, devices
        # ignored).
        x_cpu = torch.randn(2, 3, 5, 5)
        conv_cpu = torch.nn.ConvTranspose2d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)
        conv_cuda = torch.nn.ConvTranspose2d(3, 3, kernel_size=1).to(device)
        conv_cuda.bias.data.copy_(conv_cpu.bias.data)
        conv_cuda.weight.data.copy_(conv_cpu.weight.data)
        y_cuda = conv_cuda(x_cpu.to(device))
        y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )

    def test_ConvTranspose3d_size_1_kernel(self, device):
        # [FUNCTION DOC] Same CPU-versus-device comparison for a 1x1 transposed 3D convolution,
        # executed with the default dtype temporarily set to double; asserts output, bias
        # gradient and weight gradient match with atol 1e-5.
        with set_default_dtype(torch.double):
            x_cpu = torch.randn(2, 3, 3, 5, 5)
            conv_cpu = torch.nn.ConvTranspose3d(3, 3, kernel_size=1)
            y_cpu = conv_cpu(x_cpu)
            y = torch.rand_like(y_cpu)
            y_cpu.backward(y)
            conv_cuda = torch.nn.ConvTranspose3d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

            self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
            self.assertEqual(
                conv_cpu.bias.grad.data,
                conv_cuda.bias.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )
            self.assertEqual(
                conv_cpu.weight.grad.data,
                conv_cuda.weight.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )

    @dtypes(torch.float)
    def test_Conv2d_naive_groups(self, device, dtype):
        # [FUNCTION DOC] Checks a 4 to 4 channel convolution with 2 groups against two 2 to 2
        # channel convolutions on the halves of the channels (weights and biases copied from
        # slices). Asserts concatenated outputs and the input, bias and weight gradients match,
        # using the per-dtype precision table for gradients.
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(
            i.grad.data,
            torch.cat([i1.grad.data, i2.grad.data], 1),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        self.assertEqual(
            m.bias.grad.data,
            torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        self.assertEqual(
            m.weight.grad.data,
            torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )

    @dtypes(torch.double)
    def test_Conv2d_backward_depthwise(self, device, dtype):
        # [FUNCTION DOC] Runs gradcheck in double precision on a depthwise (2 groups) 2D
        # functional convolution with stride 1 by 10 and no bias.
        x = torch.randn(2, 2, 4, 20, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(2, 1, 3, 5, device=device, dtype=dtype, requires_grad=True)

        def conv2d_depthwise(x, weight):
            # [FUNCTION DOC] HELPER: Takes an input and weight and returns the grouped
            # functional convolution with 2 groups, stride of 1 by 10 and no bias.
            return torch.nn.functional.conv2d(
                x, weight, bias=None, stride=(1, 10), groups=2
            )

        torch.autograd.gradcheck(conv2d_depthwise, (x, weight))

    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_nhwc(self, device, dtype):
        # [FUNCTION DOC] Runs the nested channels-last 2D convolution comparison for four
        # configurations (small and 56 by 56 inputs, each with 1 group and with groups equal to
        # the channel count), for half and float dtypes.
        def helper(n, c, h, w, out_channels, kernel_size, groups):
            # [FUNCTION DOC] HELPER: Builds a channels-last random-integer input and convolution
            # of the given size, group count and dtype, and a double-precision contiguous
            # reference with the same weights. Runs forward and backward with the same random-
            # integer gradient, asserts the tested output and gradients are channels-last
            # contiguous and the reference ones are contiguous, and asserts output, weight, bias
            # and input gradients match the reference (dtype not required to match).
            input = torch.randint(-3, 3, (n, c, h, w), dtype=dtype, device=device).to(
                memory_format=torch.channels_last
            )
            input.requires_grad_()
            conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups).to(
                device=device, dtype=dtype, memory_format=torch.channels_last
            )
            for p in conv.parameters():
                p.data = torch.randint_like(p, -3, 3)

            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(
                device=device, dtype=torch.double, memory_format=torch.contiguous_format
            )

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -3, 3)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last)
            )

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=16)

    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_ndhwc(self, device, dtype):
        # [FUNCTION DOC] 3D version: runs the nested channels-last-3d comparison for four
        # configurations (small 4-cube and 18-cube inputs, each with 1 group and with groups
        # equal to the channel count), for half and float dtypes.
        def helper(n, c, d, h, w, out_channels, kernel_size, groups):
            # [FUNCTION DOC] HELPER: Same as the 2D helper but with a 3D convolution and
            # channels-last-3d layout, with random integers in the range minus 2 to 2: compares
            # against a double-precision contiguous reference and asserts layout of
            # outputs/gradients and numerical match.
            input = torch.randint(
                -2, 2, (n, c, d, h, w), dtype=dtype, device=device
            ).to(memory_format=torch.channels_last_3d)
            input.requires_grad_()
            conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups).to(
                device=device, dtype=dtype, memory_format=torch.channels_last_3d
            )
            for p in conv.parameters():
                p.data = torch.randint_like(p, -2, 2)

            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(
                device=device, dtype=torch.double, memory_format=torch.contiguous_format
            )

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -2, 2)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(
                input.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=16)

    def _run_conv(
        self,
        layer,
        device,
        inp,
        grad,
        ref_conv,
        ref_input,
        ref_out,
        input_format,
        weight_format,
        grad_format,
        output_format,
    ):
        # [FUNCTION DOC] HELPER: Builds a float convolution of the given class on the device
        # with channels from the input and gradient sizes and the reference kernel size, loads
        # the reference state, forces the weight, input and gradient into the given memory
        # formats, runs forward and backward, asserts the output is contiguous in the expected
        # output format, and asserts output, weight gradient, bias gradient and input gradient
        # equal the reference values.
        conv = (
            layer(inp.size(1), grad.size(1), ref_conv.weight.size(2)).float().to(device)
        )
        conv.load_state_dict(ref_conv.state_dict())
        weight_data = (
            conv.weight.detach().clone().contiguous(memory_format=weight_format)
        )
        conv.weight.data = weight_data.resize_(
            weight_data.size(), memory_format=weight_format
        )
        input = inp.clone().contiguous(memory_format=input_format)
        input.resize_(input.size(), memory_format=input_format)
        input = input.requires_grad_()
        grad = grad.contiguous(memory_format=grad_format)
        grad.resize_(grad.size(), memory_format=grad_format)
        out = conv(input)
        out.backward(grad)
        self.assertTrue(out.is_contiguous(memory_format=output_format))
        self.assertEqual(out, ref_out)
        self.assertEqual(conv.weight.grad, ref_conv.weight.grad)
        self.assertEqual(conv.bias.grad, ref_conv.bias.grad)
        self.assertEqual(input.grad, ref_input.grad)

    def _test_conv_cudnn_nhwc_nchw(self, layer, n, c, h, w, k, filter_size, device):
        # [FUNCTION DOC] HELPER: Builds random-integer float data, a reference convolution layer
        # and its reference output and gradient in default layout, then for every combination of
        # weight, gradient and input format (contiguous or channels-last) calls the run helper,
        # expecting channels-last output if the input or weight is channels-last and contiguous
        # otherwise.
        # KNOWN ISSUE: not prefixed as a test and never called in this file, so it is dead code
        # here.
        data = torch.randint(1, 10, (n, c, h, w), dtype=torch.float32, device=device)
        ref_input = data.clone().contiguous().requires_grad_(True)
        ref_conv = layer(c, k, filter_size).float().to(device)
        ref_out = ref_conv(ref_input)
        grad = torch.randint(1, 10, ref_out.size(), dtype=torch.float32, device=device)
        ref_out.backward(grad)

        for w_f in [torch.contiguous_format, torch.channels_last]:
            for g_f in [torch.contiguous_format, torch.channels_last]:
                for input_format in [torch.contiguous_format, torch.channels_last]:
                    output_format = torch.contiguous_format
                    if input_format == torch.channels_last:
                        output_format = torch.channels_last
                    if w_f == torch.channels_last:
                        output_format = torch.channels_last
                    self._run_conv(
                        layer,
                        device,
                        data,
                        grad,
                        ref_conv,
                        ref_input,
                        ref_out,
                        input_format,
                        w_f,
                        g_f,
                        output_format,
                    )

    @dtypes(torch.float, torch.double)
    def test_conv_cudnn_nhwc_support(self, device, dtype):
        # [FUNCTION DOC] Convolves a 1x16x1x1 input with a channels-last 8 to 16 weight with 3x3
        # kernel, stride of 2 by 1, padding 1, and asserts the output is channels-last contiguous,
        # then runs backward on the sum of the output. For float and double.
        input = torch.randn(
            (1, 16, 1, 1), dtype=dtype, device=device, requires_grad=True
        )
        weight = torch.randn(
            (8, 16, 3, 3), dtype=dtype, device=device, requires_grad=True
        )
        weight = weight.to(memory_format=torch.channels_last)
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        self.assertTrue(o.is_contiguous(memory_format=torch.channels_last))
        o.sum().backward()

    @dtypes(torch.float)
    def test_conv2d_no_grad(self, device, dtype):
        # [FUNCTION DOC] For batch sizes 1 to 3 and group counts 1, 2 and 4, builds a grouped
        # 8-output-channel 3x3 convolution and asserts that the output with gradient tracking
        # matches the output computed under no-grad (rtol 1e-2, atol 1e-5).
        for batch in [1, 2, 3]:
            for groups in [1, 2, 4]:
                input = torch.rand(batch, groups, 8, 8, dtype=dtype, device=device)
                m = nn.Conv2d(
                    groups,
                    8,
                    kernel_size=(3, 3),
                    groups=groups,
                    dtype=dtype,
                    device=device,
                )
                with torch.no_grad():
                    output_ng = m(input)
                output = m(input)
                self.assertEqual(output, output_ng, rtol=1e-2, atol=1e-5)

    @unittest.skipIf(torch.xpu.device_count() < 2, "only one GPU detected")
    @dtypes(torch.double, torch.float, torch.half)
    def test_conv2d_on_multi_device(self, dtype):
        # [FUNCTION DOC] Skipped if fewer than two xpu devices. Runs the same convolution on xpu
        # 0 and xpu 1 with copied modules and the same CPU input, asserts the two outputs match,
        # and runs backward on both with the same output gradient. Takes only a dtype and uses
        # explicit device names.
        # KNOWN ISSUE: the final assertion compares the two output-gradient copies with each
        # other, which are identical by construction; the resulting input gradients are never
        # compared.
        input = torch.randn(3, 256, 224, 224, dtype=dtype, requires_grad=True)
        conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dtype=dtype)
        output_grad = torch.randn(3, 256, 224, 224, dtype=dtype)
        input_0 = input.to(device="xpu:0")
        conv_0 = copy.deepcopy(conv).to(device="xpu:0")
        output_0 = conv_0(input_0)
        input_1 = input.to(device="xpu:1")
        conv_1 = copy.deepcopy(conv).to(device="xpu:1")
        output_1 = conv_1(input_1)
        self.assertEqual(output_0.cpu(), output_1.cpu())
        output_grad_0 = output_grad.to(device="xpu:0")
        output_0.backward(output_grad_0)
        output_grad_1 = output_grad.to(device="xpu:1")
        output_1.backward(output_grad_1)
        self.assertEqual(output_grad_0.cpu(), output_grad_1.cpu())

    def test_conv_double_backward_strided_with_3D_input_and_weight(self, device):
        # [FUNCTION DOC] Computes a strided 1D convolution through the low-level aten
        # convolution op on a 3D input and weight with bias, then calls the aten double-backward
        # op with random second-order gradients, an output gradient and an all-true output mask,
        # and asserts the returned gradients have the shapes of the output gradient, input and
        # weight respectively.
        input = torch.randn(2, 3, 6, device=device)
        weight = torch.randn(3, 3, 3, device=device)
        bias = torch.randn(3, device=device)
        stride = (2,)
        padding = (1,)
        dilation = (1,)
        transposed = False
        output_padding = (0,)
        groups = 1
        output = torch.ops.aten.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )

        ggI = torch.randn(input.shape, device=device)
        ggW = torch.randn(weight.shape, device=device)
        ggB = torch.randn(bias.shape, device=device)
        gO = torch.randn(output.shape, device=device)
        output_mask = [True, True, True]
        (
            grad_grad_output,
            grad_input,
            grad_weight,
        ) = torch.ops.aten._convolution_double_backward(
            ggI,
            ggW,
            ggB,
            gO,
            weight,
            input,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )

        self.assertEqual(grad_grad_output.shape, gO.shape)
        self.assertEqual(grad_input.shape, input.shape)
        self.assertEqual(grad_weight.shape, weight.shape)

    def test_aten_conv_backward_op(self, device):
        # [FUNCTION DOC] Compares the aten convolution backward op on the device against CPU for
        # all eight combinations of the output mask (input, weight, bias gradients), by calling
        # the nested helper for each mask.
        # test low-level aten.convolution_backward ops
        def test_aten_conv_backward_op_output_mask(output_mask):
            # [FUNCTION DOC] HELPER: Takes a mask of three flags. Runs the aten convolution
            # backward op on CPU with fixed random input (1x1x4x4), weight (1x1x3x3) and output
            # gradient (1x1x2x2), stride 1, no padding, then repeats with copies on the device,
            # and asserts each enabled output (input, weight or bias gradient) matches between
            # CPU and device.
            input_cpu = torch.randn(1, 1, 4, 4, device="cpu")
            weight_cpu = torch.randn(1, 1, 3, 3, device="cpu")
            grad_output_cpu = torch.randn(1, 1, 2, 2, device="cpu")
            grad_input_cpu, grad_weight_cpu, grad_bias_cpu = (
                torch.ops.aten.convolution_backward(
                    grad_output_cpu,
                    input_cpu,
                    weight_cpu,
                    [1],
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    False,
                    (0, 0),
                    1,
                    output_mask,
                )
            )
            input = input_cpu.to(device=device)
            weight = weight_cpu.to(device=device)
            grad_output = grad_output_cpu.to(device=device)
            grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                [1],
                (1, 1),
                (0, 0),
                (1, 1),
                False,
                (0, 0),
                1,
                output_mask,
            )
            if output_mask[0]:
                self.assertEqual(grad_input_cpu, grad_input)
            if output_mask[1]:
                self.assertEqual(grad_weight_cpu, grad_weight)
            if output_mask[2]:
                self.assertEqual(grad_bias_cpu, grad_bias)

        test_aten_conv_backward_op_output_mask([True, True, True])
        test_aten_conv_backward_op_output_mask([True, True, False])
        test_aten_conv_backward_op_output_mask([True, False, True])
        test_aten_conv_backward_op_output_mask([True, False, False])
        test_aten_conv_backward_op_output_mask([False, True, True])
        test_aten_conv_backward_op_output_mask([False, True, False])
        test_aten_conv_backward_op_output_mask([False, False, True])
        test_aten_conv_backward_op_output_mask([False, False, False])

    @onlyXPU
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_channels_last_ouput_stride(self, device, dtype):
        # [FUNCTION DOC] XPU only. For four floating dtypes, convolves a channels-last input
        # (2x3x16x16) with a channels-last weight (512 output channels, 3x3, stride 2, no
        # padding) and asserts the output has size 2x512x7x7 with channels-last strides.
        input = torch.randn(
            (2, 3, 16, 16), device=device, dtype=dtype, requires_grad=True
        )
        weight = torch.randn(
            (512, 3, 3, 3), device=device, dtype=dtype, requires_grad=True
        )
        input = input.to(memory_format=torch.channels_last)
        weight = weight.to(memory_format=torch.channels_last)
        out = torch.conv2d(input, weight, None, (2, 2), (0, 0), (1, 1), 1)

        # input NHWC, output NHWC
        assert_size_stride(out, (2, 512, 7, 7), (25088, 1, 3584, 512))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_conv_misaligned_input(self, device, dtype):
        # [FUNCTION DOC] For four floating dtypes, compares 1D, 2D and 3D convolution (kernel 3,
        # 3 to 4 channels, batch 2) on CPU with the device using input, weight and bias stored
        # at an address deliberately not aligned to 64 bytes, and asserts the results are equal.
        N, C, OUT_CHANNELS = 2, 3, 4

        def make_misaligned_tensor(tensor, shape):
            # [FUNCTION DOC] HELPER: Allocates a device buffer one element larger than needed,
            # takes a view that starts one element in, reshapes it, copies the given tensor into
            # it, asserts the data address is not a multiple of 64 and returns the misaligned
            # tensor.
            numel = math.prod(shape)
            base = torch.empty(numel + 1, device=device, dtype=dtype)
            tensor_device = base[1:].reshape(shape)
            tensor_device.copy_(tensor.to(device))
            self.assertTrue(tensor_device.data_ptr() % 64 != 0)
            return tensor_device

        for conv_fn, spatial in [
            (F.conv1d, (16,)),
            (F.conv2d, (8, 8)),
            (F.conv3d, (4, 4, 4)),
        ]:
            kernel = (3,) * len(spatial)
            input_shape = (N, C) + spatial
            weight_shape = (OUT_CHANNELS, C) + kernel

            input_cpu = torch.randn(input_shape, dtype=dtype)
            weight_cpu = torch.randn(weight_shape, dtype=dtype)
            bias_cpu = torch.randn(OUT_CHANNELS, dtype=dtype)

            input_device = make_misaligned_tensor(input_cpu, input_shape)
            weight_device = make_misaligned_tensor(weight_cpu, weight_shape)
            bias_device = make_misaligned_tensor(bias_cpu, (OUT_CHANNELS,))

            output_cpu = conv_fn(input_cpu, weight_cpu, bias_cpu)
            output_device = conv_fn(input_device, weight_device, bias_device)
            self.assertEqual(output_device.cpu(), output_cpu)

    @onlyXPU
    def test_onednn_allow_tf32_get_set(self):
        # [FUNCTION DOC] XPU only. Uses the mkldnn flags context manager to set the tf32 flag
        # false and then true (leaving enabled and deterministic untouched) and asserts the
        # backend attribute reports each value inside the context.
        with torch.backends.mkldnn.flags(
            enabled=None, deterministic=None, allow_tf32=False
        ):
            self.assertFalse(torch.backends.mkldnn.allow_tf32)
        with torch.backends.mkldnn.flags(
            enabled=None, deterministic=None, allow_tf32=True
        ):
            self.assertTrue(torch.backends.mkldnn.allow_tf32)


instantiate_device_type_tests(
    TestConvolutionNNDeviceType, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
