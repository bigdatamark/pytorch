# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests for torch.utils.mobile_optimizer: optimize_for_mobile (conv/linear
# prepacking through XNNPACK, add plus relu fusion, conv plus batch norm folding, dropout
# removal, preserved methods, mobile_optimized tag, hoisting quantized conv packed params),
# generate_mobile_module_lints and the private module cloning helper. Defines a flag for
# torchvision availability (set by a guarded import at import time) and the FileCheck alias
# used to assert on printed graphs.

import unittest
import torch
import torch.nn as nn
import torch.utils.bundled_inputs
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfNoXNNPACK
from torch.testing._internal.jit_utils import get_forward, get_forward_graph
from torch.utils.mobile_optimizer import (LintCode,
                                          generate_mobile_module_lints,
                                          optimize_for_mobile,
                                          MobileOptimizerType)
from torch.nn import functional as F
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import (
    IS_MACOS,
)

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

FileCheck = torch._C.FileCheck

class TestOptimizer(TestCase):
    # [CLASS DOC] Test case grouping all mobile optimizer tests; the tests build small
    # scripted modules, optimize them and compare graphs and numerical results.

    @skipIfNoXNNPACK
    def test_optimize_for_mobile(self):
        # [FUNCTION DOC] Main end to end check of optimize_for_mobile on scripted modules
        # (needs XNNPACK). A conv, relu, permute, linear, add, relu module with an exported
        # extra method is optimized with that method preserved; graph checks (via FileCheck)
        # require conv and linear to be replaced by a single prepacked run op each and add
        # plus relu fused into one op, for both forward and the extra method, with results
        # close to the originals. Then: blocklisting the prepack pass keeps the plain conv;
        # a conv plus batch norm module has its operator count drop from 11 to 1 when the
        # batch norm folding runs and keeps one batch norm node when that pass is
        # blocklisted; a linear module gets the mobile_optimized tag; unpreserved exported
        # methods are dropped unless listed in preserved_methods; and modules without a
        # forward method but with an exported foo method are optimized (dropout removed, add
        # relu fused, batch norm folded) when foo is preserved.
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

        input_data = torch.rand((batch_size, input_channels, height, width))
        conv_weight = torch.rand((output_channels, input_channels_per_group, kernel_h, kernel_w))
        conv_bias = torch.rand(output_channels)
        result = F.conv2d(input_data, conv_weight, conv_bias, strides, paddings, dilations, groups)
        weight_output_dim = 24
        linear_input_shape = result.shape[1]
        linear_weight_shape = (weight_output_dim, linear_input_shape)

        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with conv weights, linear weights, and conv
            # settings stored as attributes; forward and the exported method foo compute the
            # same pipeline.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates random Parameters for conv
                # weight and bias and linear weight and bias from the enclosing test shapes
                # and stores strides, paddings, dilations and groups.
                super().__init__()
                self.conv_weight = torch.nn.Parameter(torch.rand(conv_weight_shape))
                self.conv_bias = torch.nn.Parameter(torch.rand(conv_bias_shape))
                self.linear_weight = torch.nn.Parameter(torch.rand(linear_weight_shape))
                self.linear_bias = torch.nn.Parameter(torch.rand(weight_output_dim))
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Functional conv2d, relu, permute to channels last
                # order, linear with the stored weights, add of the permuted input, then
                # relu; returns the result.
                o = F.conv2d(x, self.conv_weight, self.conv_bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                x = o.permute([0, 2, 3, 1])
                o = F.linear(x, self.linear_weight, self.linear_bias)
                o = o + x
                return F.relu(o)

            @torch.jit.export
            def foo(self, x):
                # [FUNCTION DOC] HELPER: Exported method with exactly the same computation
                # as forward, used to test method preservation during optimization.
                o = F.conv2d(x, self.conv_weight, self.conv_bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                x = o.permute([0, 2, 3, 1])
                o = F.linear(x, self.linear_weight, self.linear_bias)
                o = o + x
                return F.relu(o)


        class BNTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Conv2d followed by a BatchNorm2d whose eps
            # is changed.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates a 1 to 20 channel 5x5 conv and
                # a 20 feature batch norm with eps set to 0.0023.
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)
                self.bn.eps = 0.0023

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv then the batch norm and returns
                # the result.
                x = self.conv(x)
                x = self.bn(x)
                return x

        data_shape = (batch_size, input_channels, height, width)
        input_data = torch.normal(1, 20, size=data_shape)

        scripted_model = torch.jit.script(MyTestModule())
        scripted_model.eval()
        initial_result = scripted_model(input_data)
        initial_foo_result = scripted_model.foo(input_data)

        optimized_scripted_model = optimize_for_mobile(scripted_model, preserved_methods=['foo'])
        optimized_result = optimized_scripted_model(input_data)
        optimized_foo_result = optimized_scripted_model.foo(input_data)

        FileCheck().check_not("Tensor = aten::conv2d") \
                   .check_not("Tensor = prim::CallFunction") \
                   .check_not("prepacked::conv2d_clamp_prepack") \
                   .check_count("prepacked::conv2d_clamp_run", 1, exactly=True) \
                   .check_not("prepacked::linear_clamp_prepack") \
                   .check_count("prepacked::linear_clamp_run", 1, exactly=True) \
                   .check_not("aten::add(") \
                   .check_not("aten::relu(") \
                   .check_count("aten::_add_relu(", 1, exactly=True) \
                   .run(optimized_scripted_model.graph)
        torch.testing.assert_close(initial_result, optimized_result, rtol=1e-2, atol=1e-3)

        FileCheck().check_not("Tensor = aten::conv2d") \
                   .check_not("Tensor = prim::CallFunction") \
                   .check_not("prepacked::conv2d_clamp_prepack") \
                   .check_count("prepacked::conv2d_clamp_run", 1, exactly=True) \
                   .check_not("prepacked::linear_clamp_prepack") \
                   .check_count("prepacked::linear_clamp_run", 1, exactly=True) \
                   .check_not("aten::add(") \
                   .check_not("aten::relu(") \
                   .check_count("aten::_add_relu(", 1, exactly=True) \
                   .run(optimized_scripted_model.foo.graph)
        torch.testing.assert_close(initial_foo_result, optimized_foo_result, rtol=1e-2, atol=1e-3)


        optimization_blocklist_no_prepack = {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS}
        optimized_scripted_model_no_prepack = optimize_for_mobile(scripted_model, optimization_blocklist_no_prepack)
        optimized_result_no_prepack = optimized_scripted_model_no_prepack(input_data)

        FileCheck().check_count("Tensor = aten::conv2d", 1, exactly=True) \
                   .check_not("prepacked::linear_clamp_run") \
                   .check_not("prepacked::conv2d_clamp_run") \
                   .run(optimized_scripted_model_no_prepack.graph)
        torch.testing.assert_close(initial_result, optimized_result_no_prepack, rtol=1e-2, atol=1e-3)


        bn_test_module = BNTestModule()
        bn_scripted_module = torch.jit.script(bn_test_module)
        bn_scripted_module.eval()

        self.assertEqual(len(torch.jit.export_opnames(bn_scripted_module)), 11)
        FileCheck().check_count('prim::CallMethod[name="forward"]', 2, exactly=True) \
                   .run(str(get_forward(bn_scripted_module._c).graph))

        optimization_blocklist_no_prepack = {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS}
        bn_fold_scripted_module = optimize_for_mobile(bn_scripted_module, optimization_blocklist_no_prepack)
        self.assertEqual(len(torch.jit.export_opnames(bn_fold_scripted_module)), 1)
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_close(bn_scripted_module(bn_input), bn_fold_scripted_module(bn_input), rtol=1e-2, atol=1e-3)

        optimization_blocklist_no_fold_bn = {MobileOptimizerType.CONV_BN_FUSION}
        no_bn_fold_scripted_module = optimize_for_mobile(bn_scripted_module, optimization_blocklist_no_fold_bn)
        FileCheck().check_count("aten::batch_norm", 1, exactly=True) \
                   .run(str(get_forward_graph(no_bn_fold_scripted_module._c)))
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_close(bn_scripted_module(bn_input), no_bn_fold_scripted_module(bn_input), rtol=1e-2, atol=1e-3)

        class MyMobileOptimizedTagTest(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear plus relu module used to check the mobile_optimized
            # tag.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates random Parameters for linear
                # weight and bias.
                super().__init__()
                self.linear_weight = torch.nn.Parameter(torch.rand(linear_weight_shape))
                self.linear_bias = torch.nn.Parameter(torch.rand(weight_output_dim))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Functional linear with the stored parameters
                # followed by relu.
                o = F.linear(x, self.linear_weight, self.linear_bias)
                return F.relu(o)

        mobile_optimized_tag_module = MyMobileOptimizedTagTest()
        m = torch.jit.script(mobile_optimized_tag_module)
        m.eval()
        opt_m = optimize_for_mobile(m)
        tag = getattr(opt_m, "mobile_optimized", None)
        self.assertTrue(tag)

        class MyPreserveMethodsTest(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear plus relu module that also has an exported method
            # preserveThis, used to test the preserved_methods argument.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates random Parameters for linear
                # weight and bias.
                super().__init__()
                self.linear_weight = torch.nn.Parameter(torch.rand(linear_weight_shape))
                self.linear_bias = torch.nn.Parameter(torch.rand(weight_output_dim))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Functional linear with the stored parameters
                # followed by relu.
                o = F.linear(x, self.linear_weight, self.linear_bias)
                return F.relu(o)

            @torch.jit.export
            def preserveThis(self):
                # [FUNCTION DOC] HELPER: Exported method with an empty body; only its
                # presence is tested.
                pass

        preserve_method_module = MyPreserveMethodsTest()
        m = torch.jit.script(preserve_method_module)
        m.eval()
        opt_m = optimize_for_mobile(m)
        no_preserveThis = getattr(opt_m, "preserveThis", None)
        self.assertEqual(no_preserveThis, None)
        opt_m = optimize_for_mobile(m, preserved_methods=["preserveThis"])
        preserveThis = getattr(opt_m, "preserveThis", None)
        self.assertNotEqual(preserveThis, None)

        class OptimizeNoForwardTest(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no forward method; only an exported method
            # foo. Holds two Linear layers and a Dropout with probability 0.2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates Linear layers 10 to 100 and
                # 100 to 1 and a dropout.
                super().__init__()
                self.l = nn.Linear(10, 100)
                self.l2 = nn.Linear(100, 1)
                self.d = nn.Dropout(p=0.2)

            @torch.jit.export
            def foo(self, x):
                # [FUNCTION DOC] HELPER: Exported method: relu of the first linear, dropout,
                # second linear, add a tensor of ones of shape one by 100, then relu;
                # returns the result.
                x = self.d(F.relu(self.l(x)))
                x = self.l2(x)
                x = x + torch.ones(1, 100)
                return F.relu(x)
        input_data = torch.ones(1, 10)
        m = torch.jit.script(OptimizeNoForwardTest())
        m.eval()
        initial_result = m.foo(input_data)

        optimized_scripted_model = optimize_for_mobile(m, preserved_methods=['foo'])
        optimized_result = optimized_scripted_model.foo(input_data)

        FileCheck().check_not("dropout.__") \
            .check_count("aten::_add_relu(", 1, exactly=True) \
            .run(optimized_scripted_model.foo.graph)
        torch.testing.assert_close(initial_result, optimized_result, rtol=1e-2, atol=1e-3)

        class BNTestNoForwardModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no forward method; only an exported method foo
            # that applies a conv and a batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates a 1 to 20 channel 5x5 conv and
                # a 20 feature batch norm with eps set to 0.0023.
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)
                self.bn.eps = 0.0023

            @torch.jit.export
            def foo(self, x):
                # [FUNCTION DOC] HELPER: Exported method applying the conv then the batch
                # norm.
                x = self.conv(x)
                x = self.bn(x)
                return x

        bn_test_no_forward_module = BNTestNoForwardModule()
        bn_no_forward_scripted_module = torch.jit.script(bn_test_no_forward_module)
        bn_no_forward_scripted_module.eval()

        self.assertEqual(len(torch.jit.export_opnames(bn_no_forward_scripted_module)), 11)
        FileCheck().check_count('prim::CallMethod[name="forward"]', 2, exactly=True) \
                   .run(bn_no_forward_scripted_module.foo.graph)

        bn_fold_no_forward_scripted_module = optimize_for_mobile(bn_no_forward_scripted_module, preserved_methods=['foo'])
        self.assertEqual(len(torch.jit.export_opnames(bn_fold_no_forward_scripted_module)), 1)
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_close(
            bn_no_forward_scripted_module.foo(bn_input),
            bn_fold_no_forward_scripted_module.foo(bn_input),
            rtol=1e-2,
            atol=1e-3)

    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/157254")
    @skipIfNoXNNPACK
    def test_quantized_conv_no_asan_failures(self):
        # [FUNCTION DOC] Regression test that optimize_for_mobile on an already quantized
        # (qnnpack) model with a nested child conv does not hit ASAN failures during conv
        # batch norm folding. Returns immediately if qnnpack is not a supported quantized
        # engine. Prepares, calibrates once, converts and scripts the model under the
        # qnnpack engine, then optimizes it. No assertions beyond not crashing. Skipped on
        # macOS and without XNNPACK.
        # There were ASAN failures when fold_conv_bn was run on
        # already quantized conv modules. Verifying that this does
        # not happen again.

        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            return

        class Child(nn.Module):
            # [CLASS DOC] HELPER: Module holding one 1x1 Conv2d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the Conv2d.
                super().__init__()
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv and returns the result.
                x = self.conv2(x)
                return x

        class Parent(nn.Module):
            # [CLASS DOC] HELPER: Module with quant stub, a conv, a child module holding a
            # conv, and a dequant stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the stubs, a 1x1 conv and the
                # child module.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.child = Child()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantize, conv, child, dequantize in that order;
                # returns the result.
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.dequant(x)
                return x

        with override_quantized_engine('qnnpack'):
            model = Parent()
            model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
            torch.ao.quantization.prepare(model, inplace=True)
            model(torch.randn(4, 1, 4, 4))
            torch.ao.quantization.convert(model, inplace=True)
            model = torch.jit.script(model)
            # this line should not have ASAN failures
            optimize_for_mobile(model)

    def test_generate_mobile_module_lints(self):
        # [FUNCTION DOC] Checks generate_mobile_module_lints on scripted modules. A linear
        # plus dropout module yields four lints: one bundled input, one dropout and two
        # requires-grad lints. A batch norm module yields four lints: one bundled input, one
        # batch norm and two requires-grad. A module augmented with bundled inputs yields no
        # lints. Counts are taken by filtering the lint records on the lint type name.
        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear layer followed by a Dropout of
            # probability 0.5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the Linear (4 to 4) and
                # Dropout layers.
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Applies the linear layer then dropout and returns
                # the result.
                out = self.fc(inputs)
                out = self.dropout(out)
                return out

        class MyBNModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one affine BatchNorm2d layer of four features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the BatchNorm2d layer.
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(4, affine=True)

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Applies the batch norm and returns the result.
                bn = self.bn(inputs)
                return bn

        class MyBundledInputModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity module used as the target for adding bundled
            # inputs.
            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return inputs

        def get_lint_count_by_type(lint_type, module_lint_List):
            # [FUNCTION DOC] HELPER: Returns the number of lint records in the list whose
            # name field equals the name of the given lint type.
            return len([lint_dict for lint_dict in module_lint_List if lint_dict['name'] == lint_type.name])

        test_module = torch.jit.script(MyTestModule())
        test_module_lint_list = generate_mobile_module_lints(test_module)
        self.assertEqual(len(test_module_lint_list), 4)
        self.assertEqual(get_lint_count_by_type(LintCode.BUNDLED_INPUT, test_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.DROPOUT, test_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.REQUIRES_GRAD, test_module_lint_list), 2)

        bn_module = torch.jit.script(MyBNModule())
        bn_module_lint_list = generate_mobile_module_lints(bn_module)
        self.assertEqual(len(bn_module_lint_list), 4)
        self.assertEqual(get_lint_count_by_type(LintCode.BUNDLED_INPUT, bn_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.BATCHNORM, bn_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.REQUIRES_GRAD, bn_module_lint_list), 2)

        bi_module = torch.jit.script(MyBundledInputModule())
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            bi_module, [(torch.tensor([1]),)], [])
        bi_module_lint_list = generate_mobile_module_lints(bi_module)
        self.assertEqual(len(bi_module_lint_list), 0)

    @skipIfNoXNNPACK
    def test_preserve_bundled_inputs_methods(self):
        # [FUNCTION DOC] Checks which bundled-input methods survive optimize_for_mobile.
        # With none present, none appear afterward. After augmenting the module with bundled
        # inputs, both the get-all and get-num methods are preserved and the bundled input
        # can be run. With only the get-all method present (incomplete set), it is dropped
        # unless explicitly listed in preserved_methods. Needs XNNPACK.
        class MyBundledInputModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity module used as the target for adding bundled
            # inputs.
            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return inputs

        class MyIncompleteBundledInputModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity module that defines only one of the bundled
            # inputs methods (the get-all method, with an empty body).
            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return inputs

            @torch.jit.export
            def get_all_bundled_inputs(self):
                # [FUNCTION DOC] HELPER: Exported stub with an empty body that returns
                # nothing.
                pass

        bi_module = torch.jit.script(MyBundledInputModule())
        module_optim_bi_not_preserved = optimize_for_mobile(bi_module)

        # Expected to be False since no bundled inputs methods were added
        self.assertFalse(
            hasattr(module_optim_bi_not_preserved, 'get_all_bundled_inputs') or
            hasattr(module_optim_bi_not_preserved, 'get_num_bundled_inputs')
        )

        # Add bundled inputs methods to the module
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            bi_module, [(torch.tensor([1]),)], [])
        # Now they should be preserved
        module_optim_bi_preserved = optimize_for_mobile(bi_module)

        # All of the bundled inputs methods were preserved
        self.assertTrue(
            hasattr(module_optim_bi_preserved, 'get_all_bundled_inputs') and
            hasattr(module_optim_bi_preserved, 'get_num_bundled_inputs')
        )

        bundled_input = module_optim_bi_preserved.get_all_bundled_inputs()[0]
        module_optim_bi_preserved(*bundled_input)

        # If not all 3 bundled inputs methods are present in the module,
        # we will not try to preserve them unless specified by the user.
        incomplete_bi_module = torch.jit.script(MyIncompleteBundledInputModule())
        incomplete_bi_module_optim = optimize_for_mobile(incomplete_bi_module)
        self.assertFalse(hasattr(incomplete_bi_module_optim, 'get_all_bundled_inputs'))

        # Specifically preserve get_all_bundled_inputs even if it's the only one
        # bundled inputs method available.
        incomplete_bi_module_optim = optimize_for_mobile(incomplete_bi_module, preserved_methods=['get_all_bundled_inputs'])
        self.assertTrue(hasattr(incomplete_bi_module_optim, 'get_all_bundled_inputs'))

    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/157247")
    @skipIfNoXNNPACK
    def test_hoist_conv_packed_params(self):
        # [FUNCTION DOC] Checks that optimize_for_mobile hoists quantized conv packed
        # parameters out of submodules. Returns early without the qnnpack engine; skipped on
        # macOS and without XNNPACK. Under the qnnpack engine, a quantized scripted model (a
        # fused standalone model, and a parent with a nested child) is optimized; the
        # optimized graph must contain exactly two packed-params constants, no conv
        # attribute lookups, no remaining conv or child attributes on the module, and
        # outputs close to the unoptimized model.

        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            return

        class Standalone(nn.Module):
            # [CLASS DOC] HELPER: Module with quant stub, two convs, a relu and a dequant
            # stub; supports fusing the second conv with the relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the stubs, two 1x1 convs and a
                # relu.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)
                self.relu = nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantize, conv1, conv2, relu, dequantize in that
                # order; returns the result.
                x = self.quant(x)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

            def fuse_model(self):
                # [FUNCTION DOC] HELPER: Fuses conv2 and relu in place using the
                # quantization module fuser.
                torch.ao.quantization.fuse_modules(self, [['conv2', 'relu']], inplace=True)

        class Child(nn.Module):
            # [CLASS DOC] HELPER: Module holding one 1x1 Conv2d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the Conv2d.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv and returns the result.
                x = self.conv1(x)
                return x

        class Parent(nn.Module):
            # [CLASS DOC] HELPER: Module with quant stub, a conv, a child module holding a
            # conv, and a dequant stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the stubs, a 1x1 conv and the
                # child module.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.child = Child()
                # TODO: test nn.Sequential after #42039 is fixed
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantize, conv1, child, dequantize in that order;
                # returns the result.
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.dequant(x)
                return x

            def fuse_model(self):
                # [FUNCTION DOC] HELPER: Does nothing; there is nothing to fuse in this
                # model.
                pass

        with override_quantized_engine('qnnpack'):
            def _quant_script_and_optimize(model):
                # [FUNCTION DOC] HELPER: Sets the qnnpack qconfig on the model, fuses it,
                # prepares it, runs one random batch for calibration, converts it (all in
                # place), scripts it and optimizes the scripted model for mobile. Returns
                # the scripted model and the optimized model.
                model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
                model.fuse_model()
                torch.ao.quantization.prepare(model, inplace=True)
                model(torch.randn(4, 1, 4, 4))
                torch.ao.quantization.convert(model, inplace=True)
                model = torch.jit.script(model)
                model_optim = optimize_for_mobile(model)
                return model, model_optim

            # basic case

            m, m_optim = _quant_script_and_optimize(Standalone())
            FileCheck().check_not('Conv2d = prim::GetAttr[name="conv1"]') \
                       .check_count("__torch__.torch.classes.quantized.Conv2dPackedParamsBase = prim::Constant", 2, exactly=True) \
                       .run(m_optim.graph)
            self.assertFalse(hasattr(m_optim, "conv1"))
            self.assertFalse(hasattr(m_optim, "conv2"))

            data = torch.randn(4, 1, 4, 4)
            m_res = m(data)
            m_optim_res = m_optim(data)
            torch.testing.assert_close(m_res, m_optim_res, rtol=1e-2, atol=1e-3)

            # generic case

            m, m_optim = _quant_script_and_optimize(Parent())
            FileCheck().check_not('Conv2d = prim::GetAttr[name="conv1"]') \
                       .check_count("__torch__.torch.classes.quantized.Conv2dPackedParamsBase = prim::Constant", 2, exactly=True) \
                       .run(m_optim.graph)
            self.assertFalse(hasattr(m_optim, "conv1"))
            self.assertFalse(hasattr(m_optim, "child"))

            data = torch.randn(4, 1, 4, 4)
            m_res = m(data)
            m_optim_res = m_optim(data)
            torch.testing.assert_close(m_res, m_optim_res, rtol=1e-2, atol=1e-3)

    @skipIfNoXNNPACK
    @unittest.skipUnless(HAS_TORCHVISION, "Needs torchvision")
    def test_mobilenet_optimize_for_mobile(self):
        # [FUNCTION DOC] Scripts torchvision mobilenet_v3_small, optimizes it for mobile and
        # runs it three times on a zero input, asserting 1000 outputs each time (regression
        # guard for a segfault on the third run). Needs XNNPACK and torchvision.
        m = torchvision.models.mobilenet_v3_small()
        m = torch.jit.script(m)
        m = optimize_for_mobile(m)

        # run forward 3 times until segfault, see https://github.com/pytorch/pytorch/issues/52463
        x = torch.zeros(1, 3, 56, 56)
        self.assertEqual(m(x).numel(), 1000)
        self.assertEqual(m(x).numel(), 1000)
        self.assertEqual(m(x).numel(), 1000)

    def test_clone_module_with_class(self):
        # [FUNCTION DOC] Tests the private module cloning helper that copies a scripted
        # module while ignoring named methods and attributes. Case one ignores two methods
        # and cloned module lacks them but keeps the other method and attribute, with a
        # class name under the torch namespace. Case two also ignores a method that
        # references an attribute plus that attribute; both are absent. Cases three and four
        # expect a RuntimeError when a preserved method references an ignored method or an
        # ignored attribute.
        # KNOWN ISSUE: the comment in case three names a method called dummy_method_cloned2,
        # which does not exist; the preserved method that references the ignored one is
        # dummy_method_not_cloned2.
        class MyInnerTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module with a tensor attribute and exported methods,
            # one of which is ignored when cloning.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; stores a three element float tensor as
                # attribute pqr.
                super().__init__()
                self.pqr = torch.Tensor([10., 20., 30.])

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return inputs

            @torch.jit.export
            def dummy_method_not_cloned(self):
                # [FUNCTION DOC] HELPER: Exported method returning a constant integer 20.
                return 20

        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module with an integer attribute, a tensor attribute
            # pqr, an inner module and several exported methods that call each other.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; sets an integer attribute, a three
                # element tensor pqr and the inner module.
                super().__init__()
                self.abc = 23
                self.pqr = torch.Tensor([1., 2., 3.])
                self.inner = MyInnerTestModule()

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Calls the cloned dummy method, the inner module
                # not-cloned method and reads the inner pqr; returns a tuple of the input
                # and those three values.
                x = self.dummy_method_cloned()
                # The call to self.inner.dummy_method_not_cloned should not raise an error
                y = self.inner.dummy_method_not_cloned()
                # The call to self.inner.pqr should not raise an error
                z = self.inner.pqr
                return (inputs, x, y, z)

            @torch.jit.export
            def dummy_method_not_cloned2(self):
                # [FUNCTION DOC] HELPER: Exported method returning a tuple of its own pqr,
                # the result of its not-cloned sibling method, the inner not-cloned method
                # result and the inner pqr.
                # The call to self.inner.dummy_method_not_cloned should not raise an error
                y = self.inner.dummy_method_not_cloned()
                # The call to self.inner.pqr should not raise an error
                z = self.inner.pqr
                return self.pqr, self.dummy_method_not_cloned(), y, z

            @torch.jit.export
            def dummy_method_not_cloned(self):
                # [FUNCTION DOC] HELPER: Exported method returning nothing.
                return None

            @torch.jit.export
            def dummy_method_cloned(self):
                # [FUNCTION DOC] HELPER: Exported method returning nothing; the one method
                # expected to survive cloning.
                return None

            @torch.jit.export
            def dummy_method_ref_attr_pqr(self):
                # [FUNCTION DOC] HELPER: Exported method returning its own pqr and the inner
                # module pqr; it depends on the pqr attribute.
                return self.pqr, self.inner.pqr

        m = torch.jit.script(MyTestModule())

        # Check that the methods exist on the original model.
        self.assertEqual(hasattr(m, "dummy_method_not_cloned"), True)
        self.assertEqual(hasattr(m, "dummy_method_cloned"), True)
        self.assertEqual(hasattr(m, "dummy_method_not_cloned2"), True)
        self.assertEqual(hasattr(m, "pqr"), True)

        # Case-1: Successfully clone, ignoring 2 methods, keeping all attributes.
        cloned = torch._C._hack_do_not_use_clone_module_with_class(
            m._c,
            ["dummy_method_not_cloned", "dummy_method_not_cloned2"],  # ignored_methods
            [],  # ignored_attributes
        )

        # Check that the ignored methods don't exist on the cloned model.
        self.assertEqual(hasattr(cloned, "dummy_method_not_cloned"), False)
        self.assertEqual(hasattr(cloned, "dummy_method_cloned"), True)
        self.assertEqual(hasattr(cloned, "dummy_method_not_cloned2"), False)
        self.assertEqual(hasattr(cloned, "pqr"), True)

        # Check that the cloned class has a classname that starts with __torch__.
        self.assertTrue(
            cloned.qualified_name.startswith('__torch__.'),
            lambda msg: (
                f"{msg}\nExpected the cloned module's name to start with the string "
                f"'__torch__.', but got: {cloned.qualified_name}"
            ),
        )


        # Case-2: Successfully clone the module, ignoring the attribute pqr, and the method that references it.
        cloned = torch._C._hack_do_not_use_clone_module_with_class(
            m._c,
            ["dummy_method_not_cloned", "dummy_method_not_cloned2", "dummy_method_ref_attr_pqr"],
            ["pqr"],
        )

        # Check that the ignored methods don't exist on the cloned model.
        self.assertEqual(hasattr(cloned, "dummy_method_not_cloned"), False)
        self.assertEqual(hasattr(cloned, "dummy_method_cloned"), True)
        self.assertEqual(hasattr(cloned, "dummy_method_not_cloned2"), False)
        self.assertEqual(hasattr(cloned, "dummy_method_ref_attr_pqr"), False)
        self.assertEqual(hasattr(cloned, "pqr"), False)


        # Case-3: The statement below will throw since dummy_method_cloned2 is preserved,
        # and references dummy_method_not_cloned, which is not cloned.
        with self.assertRaises(RuntimeError):
            cloned = torch._C._hack_do_not_use_clone_module_with_class(m._c, ["dummy_method_not_cloned"], [])

        # Case-4: The statement below will throw since dummy_method_ref_attr_pqr
        # is preserved, and references "pqr", which is not cloned.
        with self.assertRaises(RuntimeError):
            cloned = torch._C._hack_do_not_use_clone_module_with_class(
                m._c,
                ["dummy_method_not_cloned", "dummy_method_not_cloned2"],
                ["pqr"],
            )


if __name__ == '__main__':
    run_tests()
