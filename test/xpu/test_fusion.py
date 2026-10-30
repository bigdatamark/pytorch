# Owner(s): ["module: intel"]
# [MODULE DOC] XPU tests that the oneDNN fused pointwise operators (linear and convolution
# followed by a unary or binary pointwise op) match eager computation. Defines a named tuple
# describing a unary post-op, a lookup table of convolution module classes keyed by spatial
# dimension (2 and 3), and one device-type test class instantiated for the xpu device only (with
# xpu allowed) at import time. Runs the tests when executed as a script.

import itertools
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


CONV_MODULES = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


class PointwisePostOp(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple describing one unary post-op: the attribute name passed to
    # the fused kernel, the reference module applying the same op in eager mode, optional scalar
    # arguments (default empty list, shared across instances), and an algorithm string (default
    # empty) such as the gelu approximation.
    attr: str
    pointwise_module: nn.Module
    scalars: list = []
    algorithm: str = ""


class TestoneDNNFusion(TestCase):
    # [CLASS DOC] Device-type test class comparing fused mkldnn linear and convolution pointwise
    # kernels against the eager module followed by the equivalent pointwise op. Holds no state;
    # helpers supply the tables of unary and binary post-ops. Instantiated only for xpu.
    def _unary_list(self):
        # [FUNCTION DOC] HELPER: Returns a mapping from post-op name to a description with the
        # fused-kernel attribute name and the reference module: relu, sigmoid, tanh, hardswish,
        # swish (SiLU), leaky relu with slope 0.1, hardtanh with bounds minus 0.5 and 4, and
        # gelu with the none and tanh approximations (carrying the algorithm string). Builds new
        # modules on each call.
        unary_list = {
            "relu": PointwisePostOp("relu", nn.ReLU()),
            "sigmoid": PointwisePostOp("sigmoid", nn.Sigmoid()),
            "tanh": PointwisePostOp("tanh", nn.Tanh()),
            "hardswish": PointwisePostOp("hardswish", nn.Hardswish()),
            "swish": PointwisePostOp("swish", nn.SiLU()),
            "leaky_relu": PointwisePostOp(
                "leaky_relu", nn.LeakyReLU(0.1, inplace=False), scalars=[0.1]
            ),
            "hardtanh": PointwisePostOp(
                "hardtanh",
                nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False),
                scalars=[-0.5, 4],
            ),
            "gelu_none": PointwisePostOp(
                "gelu", nn.GELU(approximate="none"), algorithm="none"
            ),
            "gelu_tanh": PointwisePostOp(
                "gelu", nn.GELU(approximate="tanh"), algorithm="tanh"
            ),
        }
        return unary_list

    def _binary_list(self):
        # [FUNCTION DOC] HELPER: Returns a mapping from binary post-op name (add, sub, mul) to
        # the matching torch function, used both as the eager reference and, by name, as the
        # fused-kernel attribute.
        binary_list = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
        }
        return binary_list

    def test_linear_unary_fusion_ops(self, device):
        # [FUNCTION DOC] For every unary post-op, input shape (a three-dimensional and a two-
        # dimensional one with last dim 10) and bias on/off, builds a linear layer plus post-op
        # in eval mode on the given device, runs it eagerly under no-grad, then calls the fused
        # linear pointwise kernel with the layer weight, bias, post-op name, scalars and
        # algorithm, and asserts the two outputs are equal.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module that applies a linear layer then a stored unary
            # function. Constructor takes the unary callable, in and out feature counts, a bias
            # flag and extra keyword arguments forwarded to the linear layer.
            def __init__(self, unary_fn, in_channels, out_channels, bias, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the linear layer from the sizes, bias flag and
                # extra keyword arguments, and stores the unary callable.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.unary = unary_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer, then the stored unary
                # callable, and returns the result.
                x = self.linear(x)
                x = self.unary(x)
                return x

        for pointwise_info in self._unary_list().values():
            options = itertools.product([[2, 3, 10], [2, 10]], [True, False])
            for input_shape, bias in options:
                with torch.no_grad():
                    mod = M(
                        pointwise_info.pointwise_module, input_shape[-1], 10, bias
                    ).eval()
                    mod = mod.to(device)
                    v = torch.randn(input_shape)
                    v = v.to(device)
                    ref = mod(v)
                    attr = pointwise_info.attr
                    scalars = pointwise_info.scalars
                    algorithm = pointwise_info.algorithm
                    fused = torch.ops.mkldnn._linear_pointwise(
                        v,
                        mod.linear.weight,
                        mod.linear.bias,
                        attr,
                        scalars,
                        algorithm,
                    )
                    self.assertEqual(ref, fused)

    def test_linear_binary_fusion_ops(self, device):
        # [FUNCTION DOC] For each binary post-op (add, sub, mul), builds a linear layer from 10
        # to 20 features with bias plus the binary op in eval mode on the given device, draws a
        # random input and a random second operand, computes the eager result under no-grad,
        # then calls the fused linear pointwise kernel with the input, second operand, weight,
        # bias and op name, and asserts both results are equal.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module that applies a linear layer and then combines the
            # result with a second input using a stored binary function. Constructor takes the
            # binary callable, in and out feature counts, a bias flag and extra keyword
            # arguments forwarded to the linear layer.
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the linear layer from the sizes, bias flag and
                # extra keyword arguments, and stores the binary callable.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary = binary_fn

            def forward(self, x, other):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the first input, then the
                # stored binary callable with the second input, and returns the result.
                x = self.linear(x)
                x = self.binary(x, other)
                return x

        out_feature = 20
        in_feature = 10
        for pointwise_name, pointwise_fn in self._binary_list().items():
            with torch.no_grad():
                input = torch.randn(4, in_feature).to(device)
                model = M(pointwise_fn, in_feature, out_feature, True).eval().to(device)
                other = torch.randn(4, out_feature).to(device)
                ref = model(input, other)
                attr = pointwise_name
                fused = torch.ops.mkldnn._linear_pointwise(
                    input,
                    other,
                    model.linear.weight,
                    model.linear.bias,
                    attr,
                )
                self.assertEqual(ref, fused)

    def test_conv_unary_fusion_ops(self):
        # [FUNCTION DOC] Convolution plus unary post-op test, hard-coded to the xpu device.
        # Loops over each unary post-op, 2D and 3D, bias on/off, dilation 1 or 2, groups 1 or 4
        # and contiguous or channels-last memory format, with input of 3 times groups channels
        # and 32 times groups output channels, kernel size 3. Runs the module eagerly and the
        # fused convolution pointwise kernel (weight, bias, padding, stride, dilation, groups,
        # post-op name, scalars, algorithm) under no-grad on xpu, and asserts equal outputs.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module that applies a 2D or 3D convolution, chosen by the
            # dimension argument through the module-level lookup table, followed by a stored
            # unary callable. Extra keyword arguments such as kernel size go to the convolution.
            def __init__(
                self,
                unary_fn,
                dim,
                in_channels,
                out_channels,
                dilation,
                groups,
                bias,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Builds the convolution of the requested dimensionality
                # with the given channels, dilation, groups, bias and extra keyword arguments,
                # and stores the unary callable.
                super().__init__()
                self.conv = CONV_MODULES[dim](
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    **kwargs,
                )
                self.unary = unary_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution, then the stored unary
                # callable, and returns the result.
                x = self.conv(x)
                x = self.unary(x)
                return x

        input_shapes = {2: (112, 112), 3: (55, 55, 55)}
        for pointwise_info in self._unary_list().values():
            for dim in [2, 3]:
                channels_last = (
                    torch.channels_last if dim == 2 else torch.channels_last_3d
                )
                options = itertools.product(
                    [True, False],
                    [1, 2],
                    [1, 4],
                    [torch.contiguous_format, channels_last],
                )
                for bias, dilation, groups, memory_format in options:
                    oC = 32 * groups
                    iC = 3 * groups
                    x_shape = (1, iC) + input_shapes[dim]
                    x = torch.randn(x_shape, dtype=torch.float32).to(
                        memory_format=memory_format
                    )
                    mod = M(
                        pointwise_info.pointwise_module,
                        dim,
                        iC,
                        oC,
                        dilation,
                        groups,
                        bias,
                        kernel_size=3,
                    )
                    mod = mod.to(memory_format=memory_format).eval()
                    with torch.no_grad():
                        x = x.to("xpu")
                        mod = mod.to("xpu")
                        ref = mod(x)
                        attr = pointwise_info.attr
                        scalars = pointwise_info.scalars
                        algorithm = pointwise_info.algorithm
                        fused = torch.ops.mkldnn._convolution_pointwise(
                            x,
                            mod.conv.weight,
                            mod.conv.bias,
                            mod.conv.padding,
                            mod.conv.stride,
                            mod.conv.dilation,
                            mod.conv.groups,
                            attr,
                            scalars,
                            algorithm,
                        )
                    self.assertEqual(ref, fused)

    def test_conv_binary_fusion_ops(self):
        # [FUNCTION DOC] Convolution plus binary post-op test on xpu. For each binary op (add,
        # sub, mul), builds a 2D 3-to-3 channel convolution module with kernel size 3 and a
        # random input of shape 1x3x112x112, with a second operand shaped like the convolution
        # output. Computes the eager reference and the fused kernel result. For add, also calls
        # the in-place fused variant, which writes into the second operand, and asserts it
        # equals the reference, as does the returned tensor. Finally compares the fused result
        # with the reference using tolerances of 5e-4.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module that applies a 2D or 3D convolution, chosen by
            # dimension through the module-level lookup table, then combines the result with a
            # second input using a stored binary callable. Extra keyword arguments go to the
            # convolution.
            def __init__(
                self,
                binary_fn,
                dim,
                in_channels,
                out_channels,
                dilation,
                groups,
                bias,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Builds the convolution of the requested dimensionality
                # with the given channels, dilation, groups, bias and extra keyword arguments,
                # and stores the binary callable.
                super().__init__()
                self.conv = CONV_MODULES[dim](
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    **kwargs,
                )
                self.binary = binary_fn

            def forward(self, x, other):
                # [FUNCTION DOC] HELPER: Applies the convolution to the first input, then the
                # stored binary callable with the second input, and returns the result.
                x = self.conv(x)
                x = self.binary(x, other)
                return x

        for pointwise_name, pointwise_fn in self._binary_list().items():
            x = torch.randn(
                (
                    1,
                    3,
                    112,
                    112,
                )
            ).to("xpu")
            mod = M(pointwise_fn, 2, 3, 3, 1, 1, True, kernel_size=3).to("xpu")
            other = torch.randn_like(mod.conv(x))
            with torch.no_grad():
                ref = mod(x, other)
                unary_attr = None
                attr = pointwise_name
                fused = torch.ops.mkldnn._convolution_pointwise(
                    x,
                    other,
                    mod.conv.weight,
                    mod.conv.bias,
                    mod.conv.padding,
                    mod.conv.stride,
                    mod.conv.dilation,
                    mod.conv.groups,
                    attr,
                    None,
                    unary_attr,
                    [],
                    None,
                )
                if attr == "add":
                    fused_inplace = torch.ops.mkldnn._convolution_pointwise_(
                        other,
                        x,
                        mod.conv.weight,
                        mod.conv.bias,
                        mod.conv.padding,
                        mod.conv.stride,
                        mod.conv.dilation,
                        mod.conv.groups,
                        attr,
                        None,
                        unary_attr,
                        [],
                        None,
                    )
                    self.assertEqual(ref, other)
                    self.assertEqual(ref, fused_inplace)
                self.assertEqual(ref, fused, atol=5e-4, rtol=5e-4)


instantiate_device_type_tests(
    TestoneDNNFusion, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
