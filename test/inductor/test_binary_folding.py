# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that Inductor folds a trailing binary op (add, sub, mul, div with a scalar or
# per-channel tensor) into a preceding convolution or linear layer, and folds batch norm into conv
# under freezing. A template test class is copied by copy_tests into CPU and GPU test classes and
# then deleted from the module namespace. Import-time side effects: appends the test directory to
# sys.path, imports functorch and filelock, and imports helpers from the inductor freezing and
# torchinductor tests.
import functools
import importlib
import itertools
import os
import sys

import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch.testing._internal.common_cuda import TEST_CUDNN


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_inductor_freezing import (  # @manual=fbcode//caffe2/test/inductor:inductor_freezing-library
    TestCase,
)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
)
from torch.testing._internal.inductor_utils import skipCUDAIf


importlib.import_module("functorch")
importlib.import_module("filelock")

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


aten = torch.ops.aten


class BinaryFoldingTemplate(TestCase):
    # [CLASS DOC] Device-agnostic template (not run directly; deleted at module end) whose tests
    # use self.device. Tests count successful folds via the inductor binary_folding counter.
    @skipCUDAIf(TEST_CUDNN, "CUDNN has accuracy issues for this test")
    def test_conv_binary_folding(self):
        # [FUNCTION DOC] Skipped when cuDNN is enabled (accuracy issues). Defines a local checker
        # and runs it for every combination of bias, Conv1d/2d/3d, add/sub/mul/div and scalar or
        # tensor operand (all must fold), then for Conv2d with a rank-three broadcast operand
        # (must not fold), a one-by-one operand (must fold) and a float64 operand (must not fold,
        # with relaxed tolerances).
        @torch.no_grad()
        def test_conv_fusion(
            use_bias,
            module,
            op,
            scalar,
            add_tensor,
            expect_success,
            rtol=None,
            atol=None,
        ):
            # [FUNCTION DOC] HELPER: Builds a module of one conv plus a binary op (scalar 2.0 or a
            # stored parameter tensor, shaped per output channel unless add_tensor is given),
            # compiles it, and runs a random input (seeded, rank matching the conv type) through
            # eager and compiled versions. Asserts the outputs match within rtol/atol and that the
            # binary_folding counter is 1 if expect_success else 0. Resets dynamo and clears
            # counters first. Runs without gradients.
            class ConvOp(nn.Module):
                # [CLASS DOC] HELPER: Conv module of the given type followed by the chosen binary
                # op. Reads use_bias, module, op, scalar and add_tensor from the enclosing
                # function.
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, device, **kwargs):
                    # [FUNCTION DOC] HELPER: Creates the conv on the device, the scalar flag, the
                    # op, and a Parameter holding add_tensor, or random values shaped to broadcast
                    # along the channel dimension.
                    super().__init__()
                    self.conv = module(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.use_scalar = scalar
                    tensor_size = [1 for _ in range(self.conv.weight.ndim)]
                    tensor_size[1] = self.conv.weight.size(0)
                    self.tensor = torch.nn.Parameter(
                        add_tensor
                        if add_tensor is not None
                        else torch.rand(tensor_size).to(device)
                    )
                    self.op = op

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the conv, then the op with the scalar 2.0 or
                    # the tensor parameter.
                    x = self.conv(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            torch._dynamo.reset()
            counters.clear()
            mod_eager = ConvOp(3, 32, self.device, kernel_size=3, stride=2).eval()
            out_optimized = torch.compile(mod_eager)

            inps = [4, 3, 4]
            if module is nn.Conv2d:
                inps.append(inps[-1])
            if module is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            torch.manual_seed(1234)
            inp = torch.rand(inps).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            self.assertEqual(out_optimized, out_eager, rtol=rtol, atol=atol)
            if expect_success:
                self.assertEqual(counters["inductor"]["binary_folding"], 1)
            else:
                self.assertEqual(counters["inductor"]["binary_folding"], 0)

        conv_bias = [True, False]
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        use_scalar = [True, False]
        ops = [torch.add, torch.sub, torch.mul, torch.div]
        for use_bias, module, pytorch_op, scalar in itertools.product(
            conv_bias, modules, ops, use_scalar
        ):
            test_conv_fusion(
                use_bias,
                module,
                pytorch_op,
                scalar,
                add_tensor=None,
                expect_success=True,
            )

        for use_bias, pytorch_op in itertools.product(conv_bias, ops):
            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                pytorch_op,
                False,
                add_tensor=torch.rand(
                    32,
                    1,
                    32,
                ).to(self.device),
                expect_success=False,
            )

            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                pytorch_op,
                False,
                add_tensor=torch.rand(1, 1).to(self.device),
                expect_success=True,
            )

            # add with different dtype
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                pytorch_op,
                False,
                add_tensor=torch.tensor([2]).to(torch.float64).to(self.device),
                expect_success=False,
                # This test is for float32 conv fusion with different dtype, like float64,
                # which will not be fused. The tolerance of float64 is too tight
                # for float32 conv post fusion with float64 tensor. Will relax the tolerance
                # for this case.
                rtol=1.3e-6,
                atol=1e-5,
            )

    @inductor_config.patch({"freezing": True})
    def test_conv_bn_folding(self):
        # [FUNCTION DOC] Under the freezing config, checks that batch norm after Conv1d/2d/3d
        # (with and without bias) is folded into the conv so the compiled graph contains no add,
        # sub, mul or div aten ops, by running the local checker for every bias and module pair
        # with expected success.
        @torch.no_grad()
        def test_conv_fusion(use_bias, module, expect_success):
            # [FUNCTION DOC] HELPER: Compiles a conv followed by batch norm in eval mode with
            # compile_fx using a custom inner compiler that counts binary aten ops in the graph it
            # receives. Asserts outputs match (atol 2e-4, rtol 1e-5) and the count is zero if
            # expect_success else greater than one.
            class ConvOp(nn.Module):
                # [CLASS DOC] HELPER: Module holding a conv and a batch norm built from the given
                # module pair.
                def __init__(self, in_channels, out_channels, device, **kwargs):
                    # [FUNCTION DOC] HELPER: Creates the conv and batch norm on the device.
                    super().__init__()
                    self.conv = module[0](
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.bn = module[1](out_channels).to(device)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the conv then the batch norm.
                    x = self.conv(x)
                    return self.bn(x)

            from torch._inductor.compile_fx import compile_fx, compile_fx_inner

            aten_binary = [
                aten.add.Tensor,
                aten.sub.Tensor,
                aten.mul.Tensor,
                aten.div.Tensor,
            ]
            n_binary_ops = 0

            def my_inner_compile(gm, example_inputs, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Inner compile wrapper: calls compile_fx_inner, adds the
                # number of binary aten nodes in the graph to the enclosing counter, and returns
                # the compiled result unchanged.
                out = compile_fx_inner(gm, example_inputs, *args, **kwargs)
                nonlocal n_binary_ops
                binarry_ops = [n for n in gm.graph.nodes if n.target in aten_binary]
                n_binary_ops += len(binarry_ops)
                return out

            torch._dynamo.reset()
            mod_eager = ConvOp(3, 32, self.device, kernel_size=3, stride=2).eval()
            out_optimized = torch.compile(
                mod_eager,
                backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
            )

            inps = [4, 3, 4]
            if module[0] is nn.Conv2d:
                inps.append(inps[-1])
            if module[0] is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            self.assertEqual(out_optimized, out_eager, atol=2e-04, rtol=1e-5)
            if expect_success:
                self.assertTrue(n_binary_ops == 0)
            else:
                self.assertTrue(n_binary_ops > 1)

        conv_bias = [True, False]
        modules = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]
        for use_bias, module in itertools.product(conv_bias, modules):
            test_conv_fusion(
                use_bias,
                module,
                expect_success=True,
            )

    @inductor_config.patch({"enable_linear_binary_folding": True})
    def test_linear_binary_folding(self):
        # [FUNCTION DOC] With linear binary folding enabled, runs the local checker over bias, op,
        # scalar and several add-tensor shapes (rank one, and rank two with leading one) with 2-D
        # input, then also rank-three shapes of ones with 3-D input, all expected to fold; operands
        # shaped four by thirty-two or four by one must not fold.
        @torch.no_grad()
        def test_linear_fusion(
            use_bias, op, scalar, add_tensor, expect_success, input_3d=False
        ):
            # [FUNCTION DOC] HELPER: Compiles a linear layer followed by a binary op (scalar 2.0
            # or a tensor parameter), compares against eager on a seeded random 2-D or 3-D input
            # (atol 5e-5, rtol 5e-6) and asserts the binary_folding counter is 1 if expect_success
            # else 0. Resets dynamo and counters first.
            class LinearOp(nn.Module):
                # [CLASS DOC] HELPER: Linear layer followed by the chosen binary op, configured
                # from the enclosing function arguments.
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, device, **kwargs):
                    # [FUNCTION DOC] HELPER: Creates the linear layer on the device, the scalar
                    # flag, the op, and a Parameter from add_tensor or random values of the output
                    # feature size.
                    super().__init__()
                    self.linear = nn.Linear(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.use_scalar = scalar
                    tensor_size = [
                        self.linear.weight.size(0),
                    ]
                    self.tensor = torch.nn.Parameter(
                        add_tensor
                        if add_tensor is not None
                        else torch.rand(tensor_size).to(device)
                    )
                    self.op = op

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the linear layer, then the op with scalar 2.0
                    # or the tensor parameter.
                    x = self.linear(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            torch._dynamo.reset()
            counters.clear()
            mod_eager = LinearOp(3, 32, self.device).eval()
            out_optimized = torch.compile(mod_eager)

            torch.manual_seed(1234)
            if input_3d:
                inp = torch.rand([2, 4, 3]).to(self.device)
            else:
                inp = torch.rand([4, 3]).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            self.assertEqual(out_optimized, out_eager, atol=5e-05, rtol=5e-06)
            if expect_success:
                self.assertEqual(counters["inductor"]["binary_folding"], 1)
            else:
                self.assertEqual(counters["inductor"]["binary_folding"], 0)

        linear_bias = [True, False]
        use_scalar = [True, False]
        ops = [torch.add, torch.sub, torch.mul, torch.div]
        add_tensor_size = [
            [
                32,
            ],
            [1, 32],
            [
                1,
            ],
            [1, 1],
        ]
        for use_bias, pytorch_op, scalar, tensor_size in itertools.product(
            linear_bias, ops, use_scalar, add_tensor_size
        ):
            test_linear_fusion(
                use_bias,
                pytorch_op,
                scalar,
                add_tensor=torch.rand(tensor_size).to(self.device),
                expect_success=True,
            )

        add_tensor_size.extend([[1, 1, 32], [1, 1, 1]])
        for use_bias, pytorch_op, scalar, tensor_size in itertools.product(
            linear_bias, ops, use_scalar, add_tensor_size
        ):
            test_linear_fusion(
                use_bias,
                pytorch_op,
                scalar,
                add_tensor=torch.rand(tensor_size).to(self.device),
                expect_success=True,
                input_3d=True,
            )

        # In the following test, the shape of 'add_tensor' does not satisfy
        # the requirements of binary folding, so it will not be folded.
        for use_bias, pytorch_op in itertools.product(linear_bias, ops):
            test_linear_fusion(
                use_bias,
                pytorch_op,
                False,
                add_tensor=torch.rand(
                    4,
                    32,
                ).to(self.device),
                expect_success=False,
            )

            test_linear_fusion(
                use_bias,
                pytorch_op,
                False,
                add_tensor=torch.rand(
                    4,
                    1,
                ).to(self.device),
                expect_success=False,
            )


if HAS_CPU and not torch.backends.mps.is_available():

    class FreezingCpuTests(TestCase):
    # [CLASS DOC] CPU test class created only when CPU is available and MPS is not. Receives
    # copies of the template tests (device cpu) and has one extra test of its own.
        common = check_model
        device = "cpu"
        autocast = torch.cpu.amp.autocast

        @torch.no_grad()
        def test_conv_binary_folding_rejects_dynamic_bias(self):
        # [FUNCTION DOC] Compiles a conv with a runtime bias input followed by an add of a
        # parameter with full graph capture, checks results equal eager, and checks no binary
        # folding happened since the bias is not a constant.
            class ConvWithDynamicBias(nn.Module):
            # [CLASS DOC] HELPER: Module with conv weight and an add operand parameter; bias is a
            # forward input.
                def __init__(self):
                # [FUNCTION DOC] HELPER: Creates random weight and add operand parameters.
                    super().__init__()
                    self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))
                    self.other = nn.Parameter(torch.randn(1, 4, 1, 1))

                def forward(self, x, bias):
                # [FUNCTION DOC] HELPER: Calls the aten convolution with the given bias and then
                # adds the parameter.
                    x = aten.convolution.default(
                        x,
                        self.weight,
                        bias,
                        [1, 1],
                        [0, 0],
                        [1, 1],
                        False,
                        [0, 0],
                        1,
                    )
                    return torch.add(x, self.other)

            torch._dynamo.reset()
            counters.clear()
            mod = ConvWithDynamicBias().eval()
            x = torch.randn(2, 3, 8, 8)
            bias = torch.randn(4)
            expected = mod(x, bias)
            actual = torch.compile(mod, fullgraph=True)(x, bias)
            self.assertEqual(actual, expected)
            self.assertEqual(counters["inductor"]["binary_folding"], 0)

    copy_tests(BinaryFoldingTemplate, FreezingCpuTests, "cpu")

if HAS_GPU:

    class FreezingGpuTests(TestCase):
    # [CLASS DOC] GPU test class created only when a GPU is available; receives copies of the
    # template tests.
        common = check_model_gpu
        device = GPU_TYPE
        autocast = torch.amp.autocast(device_type=GPU_TYPE)

    copy_tests(BinaryFoldingTemplate, FreezingGpuTests, GPU_TYPE)


del BinaryFoldingTemplate

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
