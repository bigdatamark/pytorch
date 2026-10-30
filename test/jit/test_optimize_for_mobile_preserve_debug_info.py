# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests that mobile-optimization graph passes (conv1d to conv2d transform,
# insertion of prepacked linear and conv ops, fusing clamp activations into prepacked ops)
# preserve debug information: each node created by a pass must carry the same source range as
# the original node it replaced. All tests require XNNPACK support and are skipped otherwise.

import torch
import torch._C
import torch.nn.functional as F
from torch.testing._internal.common_utils import raise_on_run_directly, skipIfNoXNNPACK
from torch.testing._internal.jit_utils import JitTestCase


class TestOptimizeForMobilePreserveDebugInfo(JitTestCase):
    # [CLASS DOC] JitTestCase with a shared checker comparing node source ranges before and
    # after a pass, plus tests for each pass.
    def check_replacement(
        self,
        model,
        replacements,
        jit_pass,
    ):
        # [FUNCTION DOC] HELPER: Records, for the node kinds that appear as values of the
        # replacements mapping, the source range of nodes of that kind in the model graph (a
        # later node of the same kind overwrites an earlier one). Runs the pass on the model's
        # underlying compiled module, then for every node in the model graph whose kind is a key
        # of the mapping, asserts its source range equals the recorded range of the original
        # kind it maps to. Args are the scripted or traced model, the mapping from new kinds to
        # original kinds, and the pass callable.
        """
        model: Model which optimization is performed on
        replacements: Dict mapping from nodes' kinds in the optimized model
            to the kinds of nodes they replaced in the original model
        jit_pass: Function to perform optimization
        """

        original_kinds = set(replacements.values())
        original_source_ranges = {
            node.kind(): node.sourceRange()
            for node in model.graph.nodes()
            if node.kind() in original_kinds
        }

        jit_pass(model._c)

        for node in model.graph.nodes():
            if node.kind() in replacements:
                self.assertEqual(
                    node.sourceRange(),
                    original_source_ranges[replacements[node.kind()]],
                )

    @skipIfNoXNNPACK
    def test_replace_conv1d_with_conv2d(self):
        # [FUNCTION DOC] Skipped without XNNPACK. Scripts a small conv1d module and runs the
        # conv1d to conv2d transform; checks the generated list, unsqueeze, conv2d and squeeze
        # nodes take the source range of the original conv1d node.
        class TestConv1d(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the functional one-dimensional convolution
            # with stored weight and bias.
            def __init__(self, weight, bias):
                # [FUNCTION DOC] HELPER: Stores the weight and bias tensors as attributes.
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the functional conv1d to the input with the
                # stored weight and bias.
                return F.conv1d(x, self.weight, self.bias)

        self.check_replacement(
            model=torch.jit.script(
                TestConv1d(
                    weight=torch.rand(3, 3, 3),
                    bias=torch.rand(3),
                ),
            ),
            replacements={
                "prim::ListUnpack": "aten::conv1d",
                "prim::ListConstruct": "aten::conv1d",
                "aten::unsqueeze": "aten::conv1d",
                "aten::conv2d": "aten::conv1d",
                "aten::squeeze": "aten::conv1d",
            },
            jit_pass=torch._C._jit_pass_transform_conv1d_to_conv2d,
        )

    @skipIfNoXNNPACK
    def test_insert_pre_packed_linear_before_inline_and_conv_2d_op(self):
        # [FUNCTION DOC] Skipped without XNNPACK. Scripts a module with a linear, a conv2d and
        # a transposed conv2d call, runs the pass inserting prepacked ops, and checks the
        # prepack and run nodes keep the source range of the original linear, conv2d and conv
        # transpose nodes.
        class TestPrepackedLinearBeforeInlineAndConv2dOp(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a linear layer, a convolution and a transposed
            # convolution using stored float weights and biases.
            def __init__(
                self,
                linear_weight,
                linear_bias,
                conv2d_weight,
                conv2d_bias,
                conv_transpose2d_weight,
                conv_transpose2d_bias,
            ):
                # [FUNCTION DOC] HELPER: Calls the parent initializer and stores each weight and
                # bias converted to float.
                super(
                    TestPrepackedLinearBeforeInlineAndConv2dOp,
                    self,
                ).__init__()
                self.linear_weight = linear_weight.float()
                self.linear_bias = linear_bias.float()
                self.conv2d_weight = conv2d_weight.float()
                self.conv2d_bias = conv2d_bias.float()
                self.conv_transpose2d_weight = conv_transpose2d_weight.float()
                self.conv_transpose2d_bias = conv_transpose2d_bias.float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the functional linear, then an unsqueezed
                # functional conv2d, then a functional transposed conv2d, and returns the
                # last result.
                linear_res = F.linear(
                    x.float(),
                    self.linear_weight,
                    self.linear_bias,
                )
                conv2d_res = F.conv2d(
                    input=linear_res.unsqueeze(dim=0).float(),
                    weight=self.conv2d_weight,
                    bias=self.conv2d_bias,
                )
                return F.conv_transpose2d(
                    input=conv2d_res,
                    weight=self.conv_transpose2d_weight,
                    bias=self.conv_transpose2d_bias,
                )

        in_channels = 6
        iW = 5
        out_channels = 6
        kH = 2
        kW = 3

        self.check_replacement(
            model=torch.jit.script(
                TestPrepackedLinearBeforeInlineAndConv2dOp(
                    linear_weight=torch.rand(iW, 3),
                    linear_bias=torch.rand(iW),
                    conv2d_weight=torch.rand(out_channels, in_channels, kH, kW),
                    conv2d_bias=torch.rand(out_channels),
                    conv_transpose2d_weight=torch.rand(
                        out_channels,
                        in_channels,
                        kH,
                        kW,
                    ),
                    conv_transpose2d_bias=torch.rand(out_channels),
                ),
            ),
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear",
                "prepacked::conv2d_clamp_prepack": "aten::conv2d",
                "prepacked::conv2d_clamp_run": "aten::conv2d",
                "prepacked::conv2d_transpose_clamp_prepack": "aten::conv_transpose2d",
                "prepacked::conv2d_transpose_clamp_run": "aten::conv_transpose2d",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    @skipIfNoXNNPACK
    def test_insert_pre_packed_linear_op(self):
        # [FUNCTION DOC] Skipped without XNNPACK. Traces a linear layer, runs the pass inserting
        # prepacked ops and checks the prepack and run nodes keep the source range of the
        # original linear node.
        self.check_replacement(
            model=torch.jit.trace(torch.nn.Linear(5, 4), torch.rand(3, 2, 5)),
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    def run_test_fuse_activation_with_pack_ops_linear_conv2d(
        self,
        linear_activation,
        linear_activation_kind,
        conv2d_activation,
        conv2d_activation_kind,
    ):
        # [FUNCTION DOC] HELPER: Traces a module doing linear, the given activation, conv2d and
        # the second activation. Runs the pass inserting prepacked ops, then the pass fusing
        # clamp activations into prepacked ops, and checks each prepack node keeps its own
        # source range and each run node has the source range of the original activation node
        # whose kind is given.
        class TestFuseActivationLinearConv2d(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a linear, an activation callable from the
            # enclosing scope, a convolution and a second activation callable.
            def __init__(
                self,
                linear_weight,
                linear_bias,
                conv2d_weight,
                conv2d_bias,
            ):
                # [FUNCTION DOC] HELPER: Stores the linear and convolution weights and biases
                # as attributes.
                super().__init__()
                self.linear_weight = linear_weight
                self.linear_bias = linear_bias
                self.conv2d_weight = conv2d_weight
                self.conv2d_bias = conv2d_bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, the first activation, an unsqueezed
                # conv2d, then returns the second activation of the result.
                x = F.linear(
                    input=x,
                    weight=self.linear_weight,
                    bias=self.linear_bias,
                )
                x = linear_activation(x)
                x = F.conv2d(
                    input=x.unsqueeze(dim=0),
                    weight=self.conv2d_weight,
                    bias=self.conv2d_bias,
                )
                return conv2d_activation(x)

        linear_in_features = 5
        linear_out_features = 4
        conv2d_in_channels = 3
        conv2d_out_channels = 4
        conv2d_kernel = 2
        x_shape = (3, 2, 5)

        model = torch.jit.trace(
            TestFuseActivationLinearConv2d(
                linear_weight=torch.nn.Parameter(
                    data=torch.rand(
                        linear_out_features,
                        linear_in_features,
                    ),
                    requires_grad=False,
                ),
                linear_bias=torch.nn.Parameter(
                    data=torch.rand(linear_out_features),
                    requires_grad=False,
                ),
                conv2d_weight=torch.rand(
                    conv2d_out_channels,
                    conv2d_in_channels,
                    conv2d_kernel,
                    conv2d_kernel,
                ),
                conv2d_bias=torch.rand(conv2d_out_channels),
            ),
            torch.rand(x_shape),
        )

        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            replacements={
                "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                "prepacked::linear_clamp_run": linear_activation_kind,
                "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                "prepacked::conv2d_clamp_run": conv2d_activation_kind,
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_1(self):
        # [FUNCTION DOC] Skipped without XNNPACK. Runs the fusion check with hardtanh for the
        # linear activation and in-place hardtanh for the convolution activation.
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.hardtanh,
            linear_activation_kind="aten::hardtanh",
            conv2d_activation=F.hardtanh_,
            conv2d_activation_kind="aten::hardtanh_",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_2(self):
        # [FUNCTION DOC] Skipped without XNNPACK. Runs the fusion check with in-place hardtanh
        # for the linear activation and hardtanh for the convolution activation.
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.hardtanh_,
            linear_activation_kind="aten::hardtanh_",
            conv2d_activation=F.hardtanh,
            conv2d_activation_kind="aten::hardtanh",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_3(self):
        # [FUNCTION DOC] Skipped without XNNPACK. Runs the fusion check with relu for the linear
        # activation and in-place relu for the convolution activation.
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu,
            linear_activation_kind="aten::relu",
            conv2d_activation=F.relu_,
            conv2d_activation_kind="aten::relu_",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_4(self):
        # [FUNCTION DOC] Skipped without XNNPACK. Runs the fusion check with in-place relu for
        # the linear activation and relu for the convolution activation.
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu_,
            linear_activation_kind="aten::relu_",
            conv2d_activation=F.relu,
            conv2d_activation_kind="aten::relu",
        )


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
