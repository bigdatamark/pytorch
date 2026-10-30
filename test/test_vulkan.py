# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests for the Vulkan TorchScript rewrite passes that replace convolution with
# prepacked Vulkan ops. Defines one test class, skipped entirely when the Vulkan backend is
# unavailable. Importing has no side effects; running as a script calls run_tests.

import unittest
import torch
from torch.nn import functional as F

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing import FileCheck
import io

@unittest.skipUnless(torch.is_vulkan_available(),
                     "Vulkan backend must be available for these tests.")
class TestVulkanRewritePass(TestCase):
    # [CLASS DOC] Checks that Vulkan graph passes (insert prepacked ops, freeze, fold
    # prepacking, fuse clamp into prepacked convolution) transform scripted convolution modules
    # into the expected operator patterns. Skipped unless the Vulkan backend is available.
    @staticmethod
    def validate_transformed_module(
            # To please flake
            self,
            pattern_count_map,
            data_shape,
            prepack_removal=False,
            fuse_clamping_ops=False):
        # [FUNCTION DOC] HELPER: Static helper. The first parameter is the module under
        # test, not a class instance (the name self is used only to please the linter).
        # Scripts the module, sets eval mode and runs it once on random input, then
        # applies the Vulkan prepack insertion pass. When either option is set the module
        # is frozen; fuse_clamping_ops then fuses clamp activations into the prepacked
        # convolution, and prepack_removal folds the prepacking ops away. The module is
        # saved to memory and reloaded, and each pattern is checked against the reloaded
        # graph: a count of zero means the pattern must appear, minus one means it must
        # be absent, and any other value means it must appear exactly that many times.
        # Failures surface as FileCheck errors.
        module_instance = self
        scripted_model = torch.jit.script(module_instance)
        scripted_model.eval()
        input_data = torch.normal(1, 20, size=data_shape)
        scripted_model(input_data)
        torch._C._jit_pass_vulkan_insert_prepacked_ops(scripted_model._c)
        if fuse_clamping_ops or prepack_removal:
            scripted_model._c = torch._C._freeze_module(scripted_model._c)
        if fuse_clamping_ops:
            torch._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv(scripted_model._c)
        if prepack_removal:
            torch._C._jit_pass_vulkan_fold_prepacking_ops(scripted_model._c)

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
        # [FUNCTION DOC] Build grouped 2D convolution parameters and exercise the rewrite passes
        # on three module variants: plain convolution, convolution followed by relu, and
        # convolution followed by hardtanh. Checks that the plain convolution op is replaced by
        # one prepack and one run op; that the activation remains when only prepack removal is
        # applied (and the prepack op disappears); and that the activation is removed when clamp
        # fusion is also enabled. The final check in the hardtanh section reuses the relu
        # module, so it does not test fused hardtanh on the hardtanh module.
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
            # [CLASS DOC] HELPER: Module with a single frozen 2D convolution (random non-
            # trainable weight and bias, with stride, padding, dilation and groups captured from
            # the enclosing test).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create random weight and bias parameters without
                # gradients and store the convolution settings as attributes.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the 2D convolution to the input using the
                # stored parameters.
                return F.conv2d(x, self.weight, self.bias,
                                self.strides, self.paddings, self.dilations, self.groups)

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2D(), pattern_count_map, data_shape)

        class Conv2DRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Same convolution module followed by a relu activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create random weight and bias parameters without
                # gradients and store the convolution settings as attributes.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the convolution then relu and return the result.
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(), pattern_count_map, data_shape)

        pattern_count_map["aten::relu"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::relu"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)


        class Conv2DHardtanh(torch.nn.Module):
            # [CLASS DOC] HELPER: Same convolution module followed by a hardtanh activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create random weight and bias parameters without
                # gradients and store the convolution settings as attributes.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the convolution then hardtanh and return
                # the result.
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.hardtanh(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2DHardtanh(), pattern_count_map, data_shape)
        pattern_count_map["aten::hardtanh"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DHardtanh(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::hardtanh"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)

if __name__ == "__main__":
    run_tests()
