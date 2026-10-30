# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests for the Metal (iOS GPU) TorchScript rewrite passes. A scripted convolution
# module is run through the Metal prepacking, clamp-fusion and prepack-folding passes, and the
# resulting graph is checked for expected operator counts after a save/load round trip. Importing
# has no side effects; running the file executes the test runner.

import torch
from torch.nn import functional as F

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing import FileCheck
import io

class TestMetalRewritePass(TestCase):
    # [CLASS DOC] Test case holding the graph-rewrite checks for Metal prepacked convolution ops.
    # It has one static helper that applies the passes and checks graph patterns, and one test
    # that exercises plain, relu and hardtanh convolution modules.
    @staticmethod
    def validate_transformed_module(
            # To please flake
            self,
            pattern_count_map,
            data_shape,
            prepack_removal=False,
            fuse_clamping_ops=False):
        # [FUNCTION DOC] HELPER: Scripts the given module (passed in the first parameter, which
        # is the module instance, not a test case), puts it in eval mode and runs it once on
        # normal random data of the given shape. Then it inserts Metal prepacked ops;
        # optionally freezes the module, fuses clamping ops into prepacked conv, and folds
        # prepacking ops. The module is saved to and reloaded from an in-memory buffer, and
        # each pattern in the count mapping is checked in the loaded graph with FileCheck: zero
        # means at least one occurrence, minus one means absent, any other value means exactly
        # that many. Raises on a failed check.
        module_instance = self
        scripted_model = torch.jit.script(module_instance)
        scripted_model.eval()
        input_data = torch.normal(1, 20, size=data_shape)
        scripted_model(input_data)
        torch._C._jit_pass_metal_insert_prepacked_ops(scripted_model._c)
        if fuse_clamping_ops or prepack_removal:
            scripted_model._c = torch._C._freeze_module(scripted_model._c)
        if fuse_clamping_ops:
            torch._C._jit_pass_metal_fuse_clamp_w_prepacked_conv(scripted_model._c)
        if prepack_removal:
            torch._C._jit_pass_metal_fold_prepacking_ops(scripted_model._c)

        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        buffer.seek(0)
        deserialized_scripted_model = torch.jit.load(buffer)
        for pattern, v in pattern_count_map.items():
            if (v == 0):
                FileCheck().check(pattern).run(deserialized_scripted_model.graph)
            elif (v == -1):
                FileCheck().check_not(pattern).run(deserialized_scripted_model.graph)
            else:
                FileCheck().check_count(pattern, v, exactly=True).run(deserialized_scripted_model.graph)

    def test_conv(self):
        # [FUNCTION DOC] Builds grouped 2D convolution modules (plain, followed by relu, followed
        # by hardtanh) with constant random parameters and checks the rewritten graphs. Plain
        # conv must lose the aten conv op and gain one prepack and one run op. With relu or
        # hardtanh, prepack removal must drop the prepack op while the activation stays; for
        # relu, adding clamp fusion must also remove the activation op.
        # KNOWN ISSUE: the final hardtanh check scripts the relu module, not the hardtanh
        # module, so the absence of the hardtanh op there is trivially true and does not
        # test hardtanh fusion.
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
        dilation = 1
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        conv_weight_shape = (output_channels, input_channels_per_group, kernel_h, kernel_w)
        conv_bias_shape = (output_channels)

        class Conv2D(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with fixed random frozen weight and bias plus the
            # stride, padding, dilation and group settings captured from the enclosing test;
            # forward is one conv2d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates non-trainable random weight and bias
                # parameters and stores the captured stride, padding, dilation and group
                # values as attributes.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the 2D convolution to the input with the stored
                # parameters and settings and returns the result.
                return F.conv2d(x, self.weight, self.bias,
                                self.strides, self.paddings, self.dilations, self.groups)

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "metal_prepack::conv2d_prepack": 1,
                             "metal_prepack::conv2d_run": 1}
        TestMetalRewritePass.validate_transformed_module(Conv2D(), pattern_count_map, data_shape)

        class Conv2DRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Same convolution module as the plain one but forward applies
            # relu to the conv output; used to check prepack, relu handling and clamp fusion.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates non-trainable random weight and bias
                # parameters and stores the captured stride, padding, dilation and group
                # values as attributes.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution and then relu, returning
                # the activated result.
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "metal_prepack::conv2d_prepack": 1,
                             "metal_prepack::conv2d_run": 1}
        TestMetalRewritePass.validate_transformed_module(
            Conv2DRelu(), pattern_count_map, data_shape)

        pattern_count_map["aten::relu"] = 1
        pattern_count_map["metal_prepack::conv2d_prepack"] = -1
        TestMetalRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::relu"] = -1
        TestMetalRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)


        class Conv2DHardtanh(torch.nn.Module):
            # [CLASS DOC] HELPER: Same convolution module as the plain one but forward applies
            # hardtanh to the conv output; used to check prepack and hardtanh handling.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates non-trainable random weight and bias
                # parameters and stores the captured stride, padding, dilation and group
                # values as attributes.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution and then hardtanh,
                # returning the result.
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.hardtanh(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "metal_prepack::conv2d_prepack": 1,
                             "metal_prepack::conv2d_run": 1}
        TestMetalRewritePass.validate_transformed_module(Conv2DHardtanh(), pattern_count_map, data_shape)
        pattern_count_map["aten::hardtanh"] = 1
        pattern_count_map["metal_prepack::conv2d_prepack"] = -1
        TestMetalRewritePass.validate_transformed_module(
            Conv2DHardtanh(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::hardtanh"] = -1
        TestMetalRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)

if __name__ == "__main__":
    run_tests()
