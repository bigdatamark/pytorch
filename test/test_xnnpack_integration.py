# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests for the XNNPACK mobile integration. Covers the prepacked linear, conv2d and
# conv2d_transpose clamp ops compared against eager references, TorchScript save/load round
# trips of prepacked modules, the graph rewrite passes (prepack insertion, prepack folding,
# clamp fusion) and the conv1d to conv2d transform. Each test class is skipped when XNNPACK is
# not built in or under TSAN. Import time has no side effects beyond imports and decorators;
# run_tests is invoked under main.

import io
import itertools
import unittest

from hypothesis import assume, given, strategies as st

import torch
import torch.backends.xnnpack
import torch.testing._internal.hypothesis_utils as hu
from torch.nn import functional as F
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    run_tests,
    slowTest,
    TEST_WITH_TSAN,
    TestCase,
)
from torch.utils.mobile_optimizer import optimize_for_mobile


@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN fails with XNNPACK. Does not seem to have a good reason for failures.",
)
class TestXNNPACKOps(TestCase):
    # [CLASS DOC] Hypothesis-driven eager tests of the prepacked XNNPACK ops. Each test builds
    # random inputs and weights, runs the reference functional op and the prepack plus run pair,
    # and checks closeness with loose float tolerances. No scripting is involved.
    @unittest.skip(
        "Fails on some platforms, see https://github.com/pytorch/pytorch/issues/73488"
    )
    @given(
        batch_size=st.integers(0, 3),
        data_shape=hu.array_shapes(1, 3, 2, 64),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    def test_linear(self, batch_size, data_shape, weight_output_dim, use_bias):
        # [FUNCTION DOC] Skipped unconditionally (platform failures, issue 73488). Prepends the
        # batch size to the generated shape, makes a random weight and optional bias, and
        # compares the functional linear result with the linear_clamp_prepack then
        # linear_clamp_run result using rtol 1e-2 and atol 1e-3.
        data_shape = [batch_size] + list(data_shape)
        input_data = torch.rand(data_shape)
        weight = torch.rand((weight_output_dim, data_shape[-1]))
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        ref_result = F.linear(input_data, weight, bias)
        packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(weight, bias)
        output_linearprepacked = torch.ops.prepacked.linear_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

    @given(
        input_size=st.integers(2, 32),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    def test_linear_1d_input(self, input_size, weight_output_dim, use_bias):
        # [FUNCTION DOC] Same comparison as the batched linear test but for a one-dimensional
        # input vector of the generated size. Reference is the functional linear; the candidate
        # is linear_clamp_prepack followed by linear_clamp_run; closeness uses rtol 1e-2 and
        # atol 1e-3.
        input_data = torch.rand(input_size)
        weight = torch.rand((weight_output_dim, input_data.shape[-1]))
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        ref_result = F.linear(input_data, weight, bias)
        packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(weight, bias)
        output_linearprepacked = torch.ops.prepacked.linear_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        use_bias,
        format,
    ):
        # [FUNCTION DOC] Compares functional conv2d with conv2d_clamp_prepack plus
        # conv2d_clamp_run on random data. Channel counts are per-group sizes times groups;
        # hypothesis assume filters out shapes where the padded input is smaller than the
        # dilated kernel. The input may be converted to a chosen memory format (preserve,
        # contiguous or channels last). Checks rtol 1e-2, atol 1e-3.
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)

        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        weight = torch.rand(
            (output_channels, input_channels_per_group, kernel_h, kernel_w)
        )
        bias = None
        if use_bias:
            bias = torch.rand(output_channels)

        ref_result = F.conv2d(
            input_data, weight, bias, strides, paddings, dilations, groups
        )
        packed_weight_bias = torch.ops.prepacked.conv2d_clamp_prepack(
            weight, bias, strides, paddings, dilations, groups
        )
        xnnpack_result = torch.ops.prepacked.conv2d_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

    @given(
        batch_size=st.integers(1, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        output_pad_h=st.integers(0, 2),
        output_pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d_transpose(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        output_pad_h,
        output_pad_w,
        dilation,
        use_bias,
        format,
    ):
        # [FUNCTION DOC] Compares functional conv_transpose2d with
        # conv2d_transpose_clamp_prepack plus conv2d_transpose_clamp_run. Filters out shapes
        # with too small padded input or with output padding not smaller than both stride and
        # dilation. The weight has input channels first. Results are made contiguous before
        # comparison (rtol 1e-2, atol 1e-3).
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        output_paddings = (output_pad_h, output_pad_w)
        dilations = (dilation, dilation)
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)
        assume((output_pad_h < stride_h) and (output_pad_h < dilation))
        assume((output_pad_w < stride_w) and (output_pad_w < dilation))

        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        weight = torch.rand(
            (input_channels, output_channels_per_group, kernel_h, kernel_w)
        )
        bias = None
        if use_bias:
            bias = torch.rand(output_channels)

        # Note that groups/dilation is in reverse order from conv2d
        ref_result = F.conv_transpose2d(
            input_data,
            weight,
            bias,
            strides,
            paddings,
            output_paddings,
            groups,
            dilation,
        )
        packed_weight_bias = torch.ops.prepacked.conv2d_transpose_clamp_prepack(
            weight, bias, strides, paddings, output_paddings, dilations, groups
        )
        xnnpack_result = torch.ops.prepacked.conv2d_transpose_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(
            ref_result.contiguous(), xnnpack_result.contiguous(), rtol=1e-2, atol=1e-3
        )


@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN fails with XNNPACK. Does not seem to have a good reason for failures.",
)
class TestXNNPACKSerDes(TestCase):
    # [CLASS DOC] Tests that eager and prepacked versions of small modules, once scripted with
    # TorchScript, agree before and after being saved to an in-memory buffer and loaded back.
    # Nested module classes inside each test define the plain and prepacked variants.
    @unittest.skip(
        "Fails on some platforms, see https://github.com/pytorch/pytorch/issues/73488"
    )
    @given(
        batch_size=st.integers(0, 3),
        data_shape=hu.array_shapes(1, 3, 2, 64),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    def test_linear(self, batch_size, data_shape, weight_output_dim, use_bias):
        # [FUNCTION DOC] Skipped unconditionally (issue 73488). Scripts a plain linear module
        # and a prepacked linear module, checks they agree on random input, then saves each to a
        # BytesIO buffer, loads it back and checks agreement again with fresh input (rtol 1e-2,
        # atol 1e-3).
        class Linear(torch.nn.Module):
            # [CLASS DOC] HELPER: Plain module holding a weight and optional bias as attributes;
            # forward applies functional linear. Reference model for the serialization test.
            def __init__(self, weight, bias=None):
                # [FUNCTION DOC] HELPER: Stores the given weight and bias tensors as attributes
                # (bias may be None).
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the functional linear of the input with the
                # stored weight and bias.
                return F.linear(x, self.weight, self.bias)

        class LinearPrePacked(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that prepacks the weight and bias once at construction
            # and runs the prepacked linear op in forward.
            def __init__(self, weight, bias=None):
                # [FUNCTION DOC] HELPER: Calls linear_clamp_prepack on the weight and bias and
                # keeps the packed object as an attribute.
                super().__init__()
                self.packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(
                    weight, bias
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs linear_clamp_run on the input with the stored
                # packed weight and bias.
                return torch.ops.prepacked.linear_clamp_run(x, self.packed_weight_bias)

        data_shape = [batch_size] + list(data_shape)
        weight = torch.rand((weight_output_dim, data_shape[-1]))
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        scripted_linear = torch.jit.script(Linear(weight, bias))
        scripted_linear_clamp_prepacked = torch.jit.script(
            LinearPrePacked(weight, bias)
        )
        input_data = torch.rand(data_shape)
        ref_result = scripted_linear(input_data)
        output_linearprepacked = scripted_linear_clamp_prepacked(input_data)
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

        # Serialize the modules and then deserialize
        input_data = torch.rand(data_shape)
        buffer = io.BytesIO()
        torch.jit.save(scripted_linear, buffer)
        buffer.seek(0)
        deserialized_linear = torch.jit.load(buffer)
        buffer = io.BytesIO()
        torch.jit.save(scripted_linear_clamp_prepacked, buffer)
        buffer.seek(0)
        deserialized_linear_clamp_prepacked = torch.jit.load(buffer)
        ref_result = deserialized_linear(input_data)
        output_linearprepacked = deserialized_linear_clamp_prepacked(input_data)
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        use_bias,
        format,
    ):
        # [FUNCTION DOC] Scripts a plain conv2d module and a prepacked conv2d module, compares
        # outputs on random input in an optional memory format, then round trips both through
        # save and load and compares again on new input (rtol 1e-2, atol 1e-3). Shape
        # combinations that violate the dilated kernel size are discarded by assume.
        class Conv2D(torch.nn.Module):
            # [CLASS DOC] HELPER: Plain conv2d module storing weight, bias, strides, paddings,
            # dilations and groups; forward calls functional conv2d.
            def __init__(self, weight, bias, strides, paddings, dilations, groups):
                # [FUNCTION DOC] HELPER: Stores the weight, bias and conv hyperparameters as
                # attributes.
                super().__init__()
                self.weight = weight
                self.bias = bias
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies functional conv2d to the input using the stored
                # parameters.
                return F.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.strides,
                    self.paddings,
                    self.dilations,
                    self.groups,
                )

        class Conv2DPrePacked(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that prepacks conv2d weights at construction and runs
            # conv2d_clamp_run in forward.
            def __init__(self, weight, bias, strides, paddings, dilations, groups):
                # [FUNCTION DOC] HELPER: Calls conv2d_clamp_prepack with weight, bias, strides,
                # paddings, dilations and groups, storing the packed result.
                super().__init__()
                self.packed_weight_bias = torch.ops.prepacked.conv2d_clamp_prepack(
                    weight, bias, strides, paddings, dilations, groups
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs conv2d_clamp_run on the input with the stored
                # packed parameters.
                return torch.ops.prepacked.conv2d_clamp_run(x, self.packed_weight_bias)

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)

        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        weight = torch.rand(
            (output_channels, input_channels_per_group, kernel_h, kernel_w)
        )
        bias = None
        if use_bias:
            bias = torch.rand(output_channels)

        scripted_conv2d = torch.jit.script(
            Conv2D(weight, bias, strides, paddings, dilations, groups)
        )
        scripted_conv2d_clamp_prepacked = torch.jit.script(
            Conv2DPrePacked(weight, bias, strides, paddings, dilations, groups)
        )
        ref_result = scripted_conv2d(input_data)
        xnnpack_result = scripted_conv2d_clamp_prepacked(input_data)
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

        # Serialize the modules and then deserialize
        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        buffer = io.BytesIO()
        torch.jit.save(scripted_conv2d, buffer)
        buffer.seek(0)
        deserialized_conv2d = torch.jit.load(buffer)
        buffer = io.BytesIO()
        torch.jit.save(scripted_conv2d_clamp_prepacked, buffer)
        buffer.seek(0)
        deserialized_conv2d_clamp_prepacked = torch.jit.load(buffer)
        ref_result = deserialized_conv2d(input_data)
        xnnpack_result = deserialized_conv2d_clamp_prepacked(input_data)
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        output_pad_h=st.integers(0, 2),
        output_pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d_transpose(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        output_pad_h,
        output_pad_w,
        dilation,
        use_bias,
        format,
    ):
        # [FUNCTION DOC] Same scripted comparison and save/load round trip as the conv2d test
        # but for transposed convolution. Also discards combinations where output padding is not
        # smaller than stride and dilation. Weight layout puts input channels first.
        class Conv2DT(torch.nn.Module):
            # [CLASS DOC] HELPER: Plain transposed-conv module holding weight, bias and conv
            # hyperparameters including output paddings; forward calls functional
            # conv_transpose2d (groups before dilations).
            def __init__(
                self,
                weight,
                bias,
                strides,
                paddings,
                output_paddings,
                dilations,
                groups,
            ):
                # [FUNCTION DOC] HELPER: Stores weight, bias, strides, paddings, output
                # paddings, dilations and groups as attributes.
                super().__init__()
                self.weight = weight
                self.bias = bias
                self.strides = strides
                self.paddings = paddings
                self.output_paddings = output_paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies functional conv_transpose2d to the input using
                # the stored parameters.
                return F.conv_transpose2d(
                    x,
                    self.weight,
                    self.bias,
                    self.strides,
                    self.paddings,
                    self.output_paddings,
                    self.groups,
                    self.dilations,
                )

        class Conv2DTPrePacked(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that prepacks transposed-conv parameters at
            # construction and runs the prepacked op in forward.
            def __init__(
                self,
                weight,
                bias,
                strides,
                paddings,
                output_paddings,
                dilations,
                groups,
            ):
                # [FUNCTION DOC] HELPER: Calls conv2d_transpose_clamp_prepack with the given
                # parameters and stores the packed object.
                super().__init__()
                self.packed_weight_bias = (
                    torch.ops.prepacked.conv2d_transpose_clamp_prepack(
                        weight,
                        bias,
                        strides,
                        paddings,
                        output_paddings,
                        dilations,
                        groups,
                    )
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs conv2d_transpose_clamp_run on the input with the
                # stored packed parameters.
                return torch.ops.prepacked.conv2d_transpose_clamp_run(
                    x, self.packed_weight_bias
                )

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        output_paddings = (output_pad_h, output_pad_w)
        dilations = (dilation, dilation)
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)
        assume((output_pad_h < stride_h) and (output_pad_h < dilation))
        assume((output_pad_w < stride_w) and (output_pad_w < dilation))

        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        weight = torch.rand(
            (input_channels, output_channels_per_group, kernel_h, kernel_w)
        )
        bias = None
        if use_bias:
            bias = torch.rand(output_channels)

        scripted_conv2d = torch.jit.script(
            Conv2DT(weight, bias, strides, paddings, output_paddings, dilations, groups)
        )
        scripted_conv2d_clamp_prepacked = torch.jit.script(
            Conv2DTPrePacked(
                weight, bias, strides, paddings, output_paddings, dilations, groups
            )
        )
        ref_result = scripted_conv2d(input_data)
        xnnpack_result = scripted_conv2d_clamp_prepacked(input_data)
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

        # Serialize the modules and then deserialize
        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        buffer = io.BytesIO()
        torch.jit.save(scripted_conv2d, buffer)
        buffer.seek(0)
        deserialized_conv2d = torch.jit.load(buffer)
        buffer = io.BytesIO()
        torch.jit.save(scripted_conv2d_clamp_prepacked, buffer)
        buffer.seek(0)
        deserialized_conv2d_clamp_prepacked = torch.jit.load(buffer)
        ref_result = deserialized_conv2d(input_data)
        xnnpack_result = deserialized_conv2d_clamp_prepacked(input_data)
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

    @unittest.skip(
        "Fails on some platforms, see https://github.com/pytorch/pytorch/issues/73488"
    )
    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        linear_weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_combined_model(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        linear_weight_output_dim,
        use_bias,
        format,
    ):
        # [FUNCTION DOC] Skipped unconditionally (issue 73488). Scripts a conv2d, permute,
        # linear, relu model and its prepacked equivalent, compares them on random input, then
        # saves and loads both and compares again using a channels last input regardless of the
        # generated format. Bias is used for both layers only if use_bias is set.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Plain model: conv2d, permute to channels last order, linear,
            # then relu, all through functional ops with stored weights.
            def __init__(
                self,
                conv_weight,
                conv_bias,
                linear_weight,
                linear_bias,
                strides,
                paddings,
                dilations,
                groups,
            ):
                # [FUNCTION DOC] HELPER: Stores conv and linear weights and biases plus conv
                # strides, paddings, dilations and groups.
                super().__init__()
                self.conv_weight = conv_weight
                self.conv_bias = conv_bias
                self.linear_weight = linear_weight
                self.linear_bias = linear_bias
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs functional conv2d, permutes dimensions to batch,
                # height, width, channels, applies functional linear and returns relu of the
                # result.
                o = F.conv2d(
                    x,
                    self.conv_weight,
                    self.conv_bias,
                    self.strides,
                    self.paddings,
                    self.dilations,
                    self.groups,
                )
                o = o.permute([0, 2, 3, 1])
                o = F.linear(o, self.linear_weight, self.linear_bias)
                return F.relu(o)

        class MPrePacked(torch.nn.Module):
            # [CLASS DOC] HELPER: Prepacked counterpart of the plain model: both layers are
            # prepacked at construction and run through the clamp run ops.
            def __init__(
                self,
                conv_weight,
                conv_bias,
                linear_weight,
                linear_bias,
                strides,
                paddings,
                dilations,
                groups,
            ):
                # [FUNCTION DOC] HELPER: Prepacks the conv and linear parameters, keeping each
                # packed object as an attribute.
                super().__init__()
                self.conv2d_clamp_run_weight_bias = (
                    torch.ops.prepacked.conv2d_clamp_prepack(
                        conv_weight, conv_bias, strides, paddings, dilations, groups
                    )
                )
                self.linear_clamp_run_weight_bias = (
                    torch.ops.prepacked.linear_clamp_prepack(linear_weight, linear_bias)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Same data flow as the plain model but through
                # conv2d_clamp_run and linear_clamp_run, ending with relu.
                o = torch.ops.prepacked.conv2d_clamp_run(
                    x, self.conv2d_clamp_run_weight_bias
                )
                o = o.permute([0, 2, 3, 1])
                o = torch.ops.prepacked.linear_clamp_run(
                    o, self.linear_clamp_run_weight_bias
                )
                return F.relu(o)

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)

        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        conv_weight = torch.rand(
            (output_channels, input_channels_per_group, kernel_h, kernel_w)
        )
        conv_bias = None
        if use_bias:
            conv_bias = torch.rand(output_channels)

        linear_weight = torch.rand((linear_weight_output_dim, output_channels))
        linear_bias = None
        if use_bias:
            linear_bias = torch.rand(linear_weight_output_dim)

        scripted_m = torch.jit.script(
            M(
                conv_weight,
                conv_bias,
                linear_weight,
                linear_bias,
                strides,
                paddings,
                dilations,
                groups,
            )
        )
        scripted_m_prepacked = torch.jit.script(
            MPrePacked(
                conv_weight,
                conv_bias,
                linear_weight,
                linear_bias,
                strides,
                paddings,
                dilations,
                groups,
            )
        )
        ref_result = scripted_m(input_data)
        xnnpack_result = scripted_m_prepacked(input_data)
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

        # Serialize the modules and then deserialize
        input_data = torch.rand((batch_size, input_channels, height, width))
        input_data = input_data.contiguous(memory_format=torch.channels_last)
        buffer = io.BytesIO()
        torch.jit.save(scripted_m, buffer)
        buffer.seek(0)
        deserialized_m = torch.jit.load(buffer)
        buffer = io.BytesIO()
        torch.jit.save(scripted_m_prepacked, buffer)
        buffer.seek(0)
        deserialized_m_prepacked = torch.jit.load(buffer)
        ref_result = deserialized_m(input_data)
        xnnpack_result = deserialized_m_prepacked(input_data)
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)


@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN fails with XNNPACK. Does not seem to have a good reason for failures.",
)
class TestXNNPACKRewritePass(TestCase):
    # [CLASS DOC] Tests of the TorchScript rewrite passes that replace functional linear and
    # conv ops with prepacked ops, fold prepacking and fuse clamp-like activations. Graphs are
    # inspected with FileCheck pattern counts and outputs compared with the unmodified model.
    @staticmethod
    def validate_transformed_module(
        # To please flake
        self,
        pattern_count_map,
        data_shape,
        prepack_removal=False,
        fuse_clamping_ops=False,
    ):
        # [FUNCTION DOC] HELPER: Shared checker. For both script and trace, builds the module,
        # records a reference result, runs the insert prepacked ops pass, optionally freezes the
        # module, fuses clamp ops and folds prepacking, then saves and loads it. Each pattern in
        # the map is checked in the graph: zero means present, minus one means absent, other
        # values mean exactly that many occurrences. Finally compares the output with the
        # reference (rtol 1e-2, atol 1e-3). The first parameter named self is the module under
        # test, not a test case.
        input_data = torch.normal(1, 20, size=data_shape)

        for jit_method in ["script", "trace"]:
            module_instance = self
            if jit_method == "script":
                scripted_model = torch.jit.script(module_instance)
            else:
                scripted_model = torch.jit.trace(module_instance, input_data)
            scripted_model.eval()
            ref_result = scripted_model(input_data)
            torch._C._jit_pass_insert_prepacked_ops(scripted_model._c)
            if fuse_clamping_ops or prepack_removal:
                scripted_model._c = torch._C._freeze_module(scripted_model._c)
            if fuse_clamping_ops:
                torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv(scripted_model._c)
            if prepack_removal:
                torch._C._jit_pass_fold_prepacking_ops(scripted_model._c)

            buffer = io.BytesIO()
            torch.jit.save(scripted_model, buffer)
            buffer.seek(0)
            deserialized_scripted_model = torch.jit.load(buffer)
            for pattern, v in pattern_count_map.items():
                if v == 0:
                    FileCheck().check(pattern).run(deserialized_scripted_model.graph)
                elif v == -1:
                    FileCheck().check_not(pattern).run(
                        deserialized_scripted_model.graph
                    )
                else:
                    FileCheck().check_count(pattern, v, exactly=True).run(
                        deserialized_scripted_model.graph
                    )
            xnnpack_result = deserialized_scripted_model(input_data)
            torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

    def test_linear(self):
        # [FUNCTION DOC] Checks the rewrite for linear with and without bias, conv2d,
        # conv_transpose2d, and a conv plus linear model with activations. Verifies prepack/run
        # op counts, prepack removal, and clamp fusion for relu, in-place relu, hardtanh and in-
        # place hardtanh, plus two hardtanh cases that must not fuse.
        data_shape = [2, 3, 32]
        weight_output_dim = 24
        weight_shape = (weight_output_dim, data_shape[-1])

        class Linear(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear module with frozen random weight and bias parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates non-trainable weight and bias parameters from
                # the enclosing shapes.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )
                self.bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional linear of the input with weight and
                # bias.
                return F.linear(x, self.weight, self.bias)

        class LinearNoBias(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear module with a frozen weight and no bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates only the non-trainable weight parameter.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional linear with a None bias.
                return F.linear(x, self.weight, None)

        # Linear with bias pattern.
        pattern_count_map = {
            "Tensor = prim::CallFunction": -1,
            "prepacked::linear_clamp_prepack": 1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            Linear(), pattern_count_map, data_shape
        )
        TestXNNPACKRewritePass.validate_transformed_module(
            LinearNoBias(), pattern_count_map, data_shape
        )

        # Conv params
        batch_size = 2
        input_channels_per_group = 6
        height = 16
        width = 16
        output_channels_per_group = 6
        groups = 4
        kernel_h = kernel_w = 3
        stride_h = stride_w = 1
        pad_h = pad_w = 1
        output_pad_h = output_pad_w = 0
        dilation = 1
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        output_paddings = (output_pad_h, output_pad_w)
        dilations = (dilation, dilation)
        conv_weight_shape = (
            output_channels,
            input_channels_per_group,
            kernel_h,
            kernel_w,
        )
        conv_transpose_weight_shape = (
            input_channels,
            output_channels_per_group,
            kernel_h,
            kernel_w,
        )
        conv_bias_shape = output_channels

        class Conv2D(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv2d module with frozen weight and bias parameters and fixed
            # strides, paddings, dilations and groups from the enclosing test.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameters and stores the conv
                # hyperparameters.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.rand(conv_weight_shape), requires_grad=False
                )
                self.bias = torch.nn.Parameter(
                    torch.rand(conv_bias_shape), requires_grad=False
                )
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies functional conv2d with the stored parameters.
                return F.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.strides,
                    self.paddings,
                    self.dilations,
                    self.groups,
                )

        class Conv2DT(torch.nn.Module):
            # [CLASS DOC] HELPER: Transposed conv module with frozen weight and bias and fixed
            # hyperparameters including output paddings.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameters and stores the conv
                # hyperparameters.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.rand(conv_transpose_weight_shape), requires_grad=False
                )
                self.bias = torch.nn.Parameter(
                    torch.rand(conv_bias_shape), requires_grad=False
                )
                self.strides = strides
                self.paddings = paddings
                self.output_paddings = output_paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies functional conv_transpose2d with the stored
                # parameters.
                return F.conv_transpose2d(
                    x,
                    self.weight,
                    self.bias,
                    self.strides,
                    self.paddings,
                    self.output_paddings,
                    self.groups,
                    self.dilations,
                )

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {
            "Tensor = aten::conv2d": -1,
            "prepacked::conv2d_clamp_prepack": 1,
            "prepacked::conv2d_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            Conv2D(), pattern_count_map, data_shape
        )

        transpose_data_shape = (batch_size, input_channels, height, width)  # noqa: F841
        transpose_pattern_count_map = {
            "Tensor = aten::conv_transpose2d": -1,
            "prepacked::conv2d_transpose_clamp_prepack": 1,
            "prepacked::conv2d_transpose_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            Conv2DT(), transpose_pattern_count_map, data_shape
        )

        input_data = torch.rand((batch_size, input_channels, height, width))
        conv_weight = torch.rand(
            (output_channels, input_channels_per_group, kernel_h, kernel_w)
        )
        conv_bias = torch.rand(output_channels)
        result = F.conv2d(
            input_data, conv_weight, conv_bias, strides, paddings, dilations, groups
        )
        linear_input_shape = result.shape[1]
        linear_weight_shape = (weight_output_dim, linear_input_shape)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, activation, permute, linear, activation model; the
            # activation function is a constructor argument defaulting to relu.
            def __init__(self, activation_fn=F.relu):
                # [FUNCTION DOC] HELPER: Creates frozen conv and linear parameters, stores
                # hyperparameters and the activation callable.
                super().__init__()
                self.conv_weight = torch.nn.Parameter(
                    torch.rand(conv_weight_shape), requires_grad=False
                )
                self.conv_bias = torch.nn.Parameter(
                    torch.rand(conv_bias_shape), requires_grad=False
                )
                self.linear_weight = torch.nn.Parameter(
                    torch.rand(linear_weight_shape), requires_grad=False
                )
                self.linear_bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups
                self.activation_fn = activation_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs conv2d, the activation, a permute to channels last
                # order, linear, and the activation again.
                o = F.conv2d(
                    x,
                    self.conv_weight,
                    self.conv_bias,
                    self.strides,
                    self.paddings,
                    self.dilations,
                    self.groups,
                )
                o = self.activation_fn(o)
                o = o.permute([0, 2, 3, 1])
                o = F.linear(o, self.linear_weight, self.linear_bias)
                return self.activation_fn(o)

        pattern_count_map = {
            "Tensor = aten::conv2d": -1,
            "prepacked::conv2d_clamp_prepack": 1,
            "prepacked::conv2d_clamp_run": 1,
            "prepacked::linear_clamp_prepack": 1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            M(), pattern_count_map, data_shape
        )
        pattern_count_map["prepacked::conv2d_clamp_prepack"] = -1
        pattern_count_map["Tensor = prim::CallFunction"] = -1
        pattern_count_map["prepacked::linear_clamp_prepack"] = -1
        TestXNNPACKRewritePass.validate_transformed_module(
            M(), pattern_count_map, data_shape, prepack_removal=True
        )

        # Not inplace relu fusion test.
        pattern_count_map = {
            "aten::relu": 2,
            "prepacked::conv2d_clamp_prepack": -1,
            "prepacked::conv2d_clamp_run": 1,
            "prepacked::linear_clamp_prepack": -1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            M(), pattern_count_map, data_shape, prepack_removal=True
        )
        pattern_count_map["prepacked::conv2d_clamp_prepack"] = -1
        pattern_count_map["prepacked::linear_clamp_prepack"] = -1
        pattern_count_map["aten::relu"] = -1
        TestXNNPACKRewritePass.validate_transformed_module(
            M(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True,
        )

        # Inplace relu fusion test.
        pattern_count_map = {
            "aten::relu": 2,
            "prepacked::conv2d_clamp_prepack": -1,
            "prepacked::conv2d_clamp_run": 1,
            "prepacked::linear_clamp_prepack": -1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            M(F.relu_), pattern_count_map, data_shape, prepack_removal=True
        )
        pattern_count_map["prepacked::conv2d_clamp_prepack"] = -1
        pattern_count_map["prepacked::linear_clamp_prepack"] = -1
        pattern_count_map["aten::relu"] = -1
        TestXNNPACKRewritePass.validate_transformed_module(
            M(F.relu_),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True,
        )

        # Not inplace hardtanh fusion test.
        pattern_count_map = {
            "aten::hardtanh": 2,
            "prepacked::conv2d_clamp_prepack": -1,
            "prepacked::conv2d_clamp_run": 1,
            "prepacked::linear_clamp_prepack": -1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            M(F.hardtanh), pattern_count_map, data_shape, prepack_removal=True
        )
        pattern_count_map["prepacked::conv2d_clamp_prepack"] = -1
        pattern_count_map["prepacked::linear_clamp_prepack"] = -1
        pattern_count_map["aten::hardtanh"] = -1
        TestXNNPACKRewritePass.validate_transformed_module(
            M(F.hardtanh),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True,
        )

        # Inplace hardtanh fusion test.
        pattern_count_map = {
            "aten::hardtanh_": 2,
            "prepacked::conv2d_clamp_prepack": -1,
            "prepacked::conv2d_clamp_run": 1,
            "prepacked::linear_clamp_prepack": -1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            M(F.hardtanh_), pattern_count_map, data_shape, prepack_removal=True
        )
        pattern_count_map["prepacked::conv2d_clamp_prepack"] = -1
        pattern_count_map["prepacked::linear_clamp_prepack"] = -1
        pattern_count_map["aten::hardtanh_"] = -1
        TestXNNPACKRewritePass.validate_transformed_module(
            M(F.hardtanh_),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True,
        )

        class MFusionAntiPattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear followed by relu and then hardtanh. The relu can be
            # fused into the prepacked linear but the trailing hardtanh cannot.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates frozen linear parameters and stores unused conv
                # hyperparameters.
                super().__init__()
                self.linear_weight = torch.nn.Parameter(
                    torch.rand(linear_weight_shape), requires_grad=False
                )
                self.linear_bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, relu and hardtanh in sequence.
                o = F.linear(x, self.linear_weight, self.linear_bias)
                o = F.relu(o)
                o = F.hardtanh(o)
                return o

        # Unfusable hardtanh.
        pattern_count_map = {
            "aten::hardtanh": 1,  # hardtanh cannot be.
            "aten::relu": -1,  # relu is fused.
            "prepacked::linear_clamp_prepack": -1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            MFusionAntiPattern(),
            pattern_count_map,
            (16, linear_weight_shape[1]),
            prepack_removal=True,
            fuse_clamping_ops=True,
        )

        class MFusionAntiPatternParamMinMax(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear followed by hardtanh whose bounds are runtime tensors,
            # so the hardtanh is not fusable.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates frozen linear parameters and stores unused conv
                # hyperparameters.
                super().__init__()
                self.linear_weight = torch.nn.Parameter(
                    torch.rand(linear_weight_shape), requires_grad=False
                )
                self.linear_bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Takes the first element of the input as the lower bound
                # and that plus ten as the upper bound, applies linear, then hardtanh with those
                # tensor bounds.
                min = x[0, 0]
                max = min + 10
                o = F.linear(x, self.linear_weight, self.linear_bias)
                o = F.hardtanh(o, min, max)
                return o

        # Unfusable hardtanh.
        pattern_count_map = {
            "aten::hardtanh": 1,  # hardtanh cannot be.
            "prepacked::linear_clamp_prepack": -1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            MFusionAntiPatternParamMinMax(),
            pattern_count_map,
            (16, linear_weight_shape[1]),
            prepack_removal=True,
            fuse_clamping_ops=True,
        )

    def test_decomposed_linear(self):
        # [FUNCTION DOC] Checks that decomposed linear forms (transpose plus addmm, matmul plus
        # in-place add, matmul alone) are all rewritten into a single prepacked linear prepack
        # and run pair with no remaining function calls.
        data_shape = [2, 32]
        weight_output_dim = 24
        weight_shape = (weight_output_dim, data_shape[-1])

        class DecomposedLinearAddmm(torch.nn.Module):
            # [CLASS DOC] HELPER: Decomposed linear using a transposed weight and addmm with the
            # bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates frozen weight and bias parameters.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )
                self.bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Transposes the weight and returns addmm of bias, input
                # and transposed weight.
                weight_t = self.weight.t()
                return torch.addmm(self.bias, x, weight_t)

        class DecomposedLinearMatmulAdd(torch.nn.Module):
            # [CLASS DOC] HELPER: Decomposed linear using matmul followed by an in-place add of
            # the bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates frozen weight and bias parameters.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )
                self.bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Matmul with the transposed weight, then adds the bias
                # in place and returns it.
                weight_t = self.weight.t()
                y = torch.matmul(x, weight_t)
                res = y.add_(self.bias)
                return res

        class DecomposedLinearMatmul(torch.nn.Module):
            # [CLASS DOC] HELPER: Decomposed linear using matmul with the transposed weight and
            # no bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates frozen weight and bias parameters (the bias is
                # unused in forward).
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.rand(weight_shape), requires_grad=False
                )
                self.bias = torch.nn.Parameter(
                    torch.rand(weight_output_dim), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns matmul of the input with the transposed weight.
                weight_t = self.weight.t()
                res = torch.matmul(x, weight_t)
                return res

        # Linear with bias pattern.
        pattern_count_map = {
            "Tensor = prim::CallFunction": -1,
            "prepacked::linear_clamp_prepack": 1,
            "prepacked::linear_clamp_run": 1,
        }
        TestXNNPACKRewritePass.validate_transformed_module(
            DecomposedLinearAddmm(), pattern_count_map, data_shape
        )
        TestXNNPACKRewritePass.validate_transformed_module(
            DecomposedLinearMatmulAdd(), pattern_count_map, data_shape
        )
        TestXNNPACKRewritePass.validate_transformed_module(
            DecomposedLinearMatmul(), pattern_count_map, data_shape
        )


@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class TestXNNPACKConv1dTransformPass(TestCase):
    # [CLASS DOC] Tests of the pass that rewrites conv1d to conv2d, followed by mobile
    # optimization that prepacks the resulting conv2d. Skipped when XNNPACK is off or under
    # TSAN.
    @staticmethod
    def validate_transform_conv1d_to_conv2d(
        self, pattern_count_transformed_map, pattern_count_optimized_map, data_shape
    ):
        # [FUNCTION DOC] Shared checker. For both script and trace, runs the conv1d to conv2d
        # pass, also produces an optimize_for_mobile version, saves and loads both, and checks
        # graph patterns (zero present, minus one absent, otherwise exact count) and output
        # closeness to the reference result (rtol 1e-2, atol 1e-3). The first parameter named
        # self is the module under test.
        input_data = torch.normal(1, 20, size=data_shape)

        for jit_method in ["script", "trace"]:
            module_instance = self
            if jit_method == "script":
                scripted_model = torch.jit.script(module_instance)
            else:
                scripted_model = torch.jit.trace(module_instance, input_data)
            scripted_model.eval()
            ref_result = scripted_model(input_data)
            torch._C._jit_pass_transform_conv1d_to_conv2d(scripted_model._c)
            optimized_scripted_model = optimize_for_mobile(scripted_model)

            buffer = io.BytesIO()
            torch.jit.save(scripted_model, buffer)
            buffer.seek(0)
            deserialized_scripted_model = torch.jit.load(buffer)

            for pattern, v in pattern_count_transformed_map.items():
                if v == 0:
                    FileCheck().check(pattern).run(deserialized_scripted_model.graph)
                elif v == -1:
                    FileCheck().check_not(pattern).run(
                        deserialized_scripted_model.graph
                    )
                else:
                    FileCheck().check_count(pattern, v, exactly=True).run(
                        deserialized_scripted_model.graph
                    )
            transformed_result = deserialized_scripted_model(input_data)
            torch.testing.assert_close(
                ref_result, transformed_result, rtol=1e-2, atol=1e-3
            )

            optimized_buffer = io.BytesIO()
            torch.jit.save(optimized_scripted_model, optimized_buffer)
            optimized_buffer.seek(0)
            deserialized_optimized_scripted_model = torch.jit.load(optimized_buffer)

            for pattern, v in pattern_count_optimized_map.items():
                if v == 0:
                    FileCheck().check(pattern).run(
                        deserialized_optimized_scripted_model.graph
                    )
                elif v == -1:
                    FileCheck().check_not(pattern).run(
                        deserialized_optimized_scripted_model.graph
                    )
                else:
                    FileCheck().check_count(pattern, v, exactly=True).run(
                        deserialized_optimized_scripted_model.graph
                    )
            xnnpack_result = deserialized_optimized_scripted_model(input_data)
            torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

    @unittest.skipIf(IS_FBCODE, "T137513244")
    def test_conv1d_basic(self):
        # [FUNCTION DOC] Skipped in FBCODE. Iterates over a product of batch size, channel,
        # width, group, kernel, stride, padding and dilation values, builds a conv1d module for
        # each and checks that conv1d is replaced by conv2d, and after mobile optimization only
        # the prepacked conv2d run remains.
        batch_size_list = range(1, 3)
        input_channels_per_group_list = range(10, 12)
        width_list = range(10, 12)
        output_channels_per_group_list = range(10, 12)
        groups_list = range(1, 3)
        kernel_list = range(1, 4)
        stride_list = range(1, 3)
        padding_list = range(3)
        dilation_list = range(1, 3)

        for hparams in itertools.product(
            batch_size_list,
            input_channels_per_group_list,
            width_list,
            output_channels_per_group_list,
            groups_list,
            kernel_list,
            stride_list,
            padding_list,
            dilation_list,
        ):
            (
                batch_size,
                input_channels_per_group,
                width,
                output_channels_per_group,
                groups,
                kernel,
                stride,
                padding,
                dilation,
            ) = hparams

            input_channels = input_channels_per_group * groups
            output_channels = output_channels_per_group * groups
            conv_weight_shape = (output_channels, input_channels_per_group, kernel)
            conv_bias_shape = output_channels

            class Conv1D(torch.nn.Module):
                # [CLASS DOC] HELPER: Conv1d module with frozen weight and bias parameters and
                # scalar stride, padding, dilation and groups taken from the current loop
                # iteration.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the parameters and stores the
                    # hyperparameters.
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.rand(conv_weight_shape), requires_grad=False
                    )
                    self.bias = torch.nn.Parameter(
                        torch.rand(conv_bias_shape), requires_grad=False
                    )
                    self.stride = stride
                    self.padding = padding
                    self.dilation = dilation
                    self.groups = groups

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies functional conv1d with the stored
                    # parameters.
                    return F.conv1d(
                        x,
                        self.weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )

            data_shape = (batch_size, input_channels, width)
            pattern_count_transformed_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": 1,
            }
            pattern_count_optimized_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": -1,
                "prepacked::conv2d_clamp_prepack": -1,
                "prepacked::conv2d_clamp_run": 1,
            }

            TestXNNPACKConv1dTransformPass.validate_transform_conv1d_to_conv2d(
                Conv1D(),
                pattern_count_transformed_map,
                pattern_count_optimized_map,
                data_shape,
            )

    # See https://github.com/pytorch/pytorch/issues/46066
    @slowTest
    def test_conv1d_with_relu_fc(self):
        # [FUNCTION DOC] Marked slow. Same pattern checks as the basic test but over an extra
        # output feature count, with a model of conv1d, relu, flatten and linear (see issue
        # 46066). Fully connected size derives from the computed conv output width.
        batch_size_list = range(1, 3)
        input_channels_per_group_list = range(10, 12)
        width_list = range(10, 12)
        output_channels_per_group_list = range(10, 12)
        groups_list = range(1, 3)
        kernel_list = range(1, 4)
        stride_list = range(1, 3)
        padding_list = range(3)
        dilation_list = range(1, 3)
        output_features_list = range(1, 3)

        for hparams in itertools.product(
            batch_size_list,
            input_channels_per_group_list,
            width_list,
            output_channels_per_group_list,
            groups_list,
            kernel_list,
            stride_list,
            padding_list,
            dilation_list,
            output_features_list,
        ):
            (
                batch_size,
                input_channels_per_group,
                width,
                output_channels_per_group,
                groups,
                kernel,
                stride,
                padding,
                dilation,
                output_features,
            ) = hparams

            input_channels = input_channels_per_group * groups
            output_channels = output_channels_per_group * groups
            conv_weight_shape = (output_channels, input_channels_per_group, kernel)
            conv_bias_shape = output_channels
            conv_output_width = (
                int((width + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1
            )
            fc_weight_shape = (output_features, output_channels * conv_output_width)
            fc_bias_shape = output_features

            class Net(torch.nn.Module):
                # [CLASS DOC] HELPER: Conv1d, relu, flatten view, linear model with frozen
                # parameters sized from the current loop iteration.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates frozen conv and fully connected parameters
                    # and stores the hyperparameters.
                    super().__init__()
                    self.conv_weight = torch.nn.Parameter(
                        torch.rand(conv_weight_shape), requires_grad=False
                    )
                    self.conv_bias = torch.nn.Parameter(
                        torch.rand(conv_bias_shape), requires_grad=False
                    )
                    self.stride = stride
                    self.padding = padding
                    self.dilation = dilation
                    self.groups = groups

                    self.fc_weight = torch.nn.Parameter(
                        torch.rand(fc_weight_shape), requires_grad=False
                    )
                    self.fc_bias = torch.nn.Parameter(
                        torch.rand(fc_bias_shape), requires_grad=False
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies conv1d and relu, reshapes to batch by
                    # features, then applies linear and returns it.
                    x = F.conv1d(
                        x,
                        self.conv_weight,
                        self.conv_bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                    x = F.relu(x)
                    x = x.view(x.size(0), -1)
                    x = F.linear(x, self.fc_weight, self.fc_bias)
                    return x

            data_shape = (batch_size, input_channels, width)
            pattern_count_transformed_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": 1,
            }
            pattern_count_optimized_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": -1,
                "prepacked::conv2d_clamp_prepack": -1,
                "prepacked::conv2d_clamp_run": 1,
            }
            TestXNNPACKConv1dTransformPass.validate_transform_conv1d_to_conv2d(
                Net(),
                pattern_count_transformed_map,
                pattern_count_optimized_map,
                data_shape,
            )


if __name__ == "__main__":
    run_tests()
