# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests the Inductor FX pass that fuses a convolution (or linear) with an eval-mode
# batch norm. Defines two test modules (one pair, and several pairs with multiple users), a test
# template with regression and sweep tests, and copies it into cpu and GPU classes. Import-time
# side effects: adds the test directory to the path, imports functorch and filelock, and deletes
# the template so it is not collected itself.
import copy
import importlib
import itertools
import os
import sys

import torch
from torch import nn


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_cuda import tf32_on_and_off
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


importlib.import_module("functorch")
importlib.import_module("filelock")

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    copy_tests,
)


class ConvOp(nn.Module):
    # [CLASS DOC] Single conv-or-linear followed by a batch norm. Attribute
    # expected_optimization_count of 1 is the number of fusions the pass should perform.
    expected_optimization_count = 1

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        # [FUNCTION DOC] Builds the conv module with given in/out channels, bias flag and extra
        # keyword arguments, and the batch norm module over the output channels, both moved to
        # the device.
        super().__init__()
        self.conv = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn = bn_class(out_channels).to(device)

    def forward(self, x):
        # [FUNCTION DOC] Applies the conv then the batch norm.
        x = self.conv(x)
        return self.bn(x)


class MultiUserConvOp(nn.Module):
    # [CLASS DOC] Three conv plus batch norm pairs with differing reuse: one fusable pair, one
    # where the conv is applied twice so not fusable, and one where the batch norm is applied
    # twice. Expected fusion count is 3.
    expected_optimization_count = 3

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        # [FUNCTION DOC] Builds three conv layers (first maps in to out channels, others out to
        # out) and three batch norms, all moved to the device.
        super().__init__()
        self.conv1 = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn1 = bn_class(out_channels).to(device)
        self.conv2 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn2 = bn_class(out_channels).to(device)
        self.conv3 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn3 = bn_class(out_channels).to(device)

    def forward(self, x):
        # [FUNCTION DOC] Runs the three pairs in sequence, with the third batch norm applied a
        # second time and added to its input as a residual; comments mark which pairs are
        # fusable.
        # this conv-bn pair can use efficient_conv_bn_eval
        x = self.bn1(self.conv1(input=x))
        # this conv-bn pair cannot use efficient_conv_bn_eval feature
        # just for the second forward of the `self.conv2`
        x = self.bn2(input=self.conv2(self.conv2(x)))
        # this conv-bn pair can use efficient_conv_bn_eval feature
        # just for the first forward of the `self.bn3`
        # test for multiple users of one computation node
        x = self.bn3(input=self.conv3(input=x))
        x = self.bn3(x) + x
        return x


class EfficientConvBNEvalTemplate(TestCase):
    # [CLASS DOC] Test template (not run directly; copied to device-specific classes, which
    # supply the device attribute, then deleted from the module).
    @tf32_on_and_off(0.003)
    @inductor_config.patch({"efficient_conv_bn_eval_fx_passes": True})
    def test_functional_batch_norm_defaults(self):
        # [FUNCTION DOC] Regression test: compiles an eval model calling functional batch norm
        # with only input, mean and variance, with the fusion pass config on, on the device
        # given by the device attribute (cpu default); only checks it runs. TF32 tolerance
        # wrapper applied.
        class Model(torch.nn.Module):
            # [CLASS DOC] Tiny eval-mode module whose forward calls functional batch norm with
            # default arguments.
            def forward(self, x, mean, var):
                # [FUNCTION DOC] Calls functional batch norm with input and running statistics
                # only.
                return torch.nn.functional.batch_norm(x, mean, var)

        mod = Model().eval()
        x = torch.randn(1, 3, 4, 4)
        mean = torch.randn(3)
        var = torch.abs(torch.randn(3))

        device = getattr(self, "device", "cpu")
        mod.to(device)
        x = x.to(device)
        mean = mean.to(device)
        var = var.to(device)

        opt = torch.compile(mod, backend="inductor")
        opt(x, mean, var)

    @tf32_on_and_off(0.003)
    @inductor_config.patch({"efficient_conv_bn_eval_fx_passes": True})
    def test_fx_graph_batch_norm_defaults(self):
        # [FUNCTION DOC] Regression test for issue 169011: builds an FX graph with a single
        # batch norm call with three positional arguments, compiles with inductor, and compares
        # against the uncompiled graph module. Previously failed an assert about eight node
        # arguments.
        """Regression test for issue #169011.

        Tests that torch.compile handles FX graphs containing F.batch_norm
        with only 3 positional arguments (input, running_mean, running_var).
        The original bug was an AssertionError: assert len(bn_node.args) == 8.
        """
        from torch.fx import Graph, GraphModule

        graph = Graph()

        # Create input placeholders
        inp = graph.placeholder("input")
        mean = graph.placeholder("mean")
        var = graph.placeholder("var")

        # Create F.batch_norm call with only 3 args (the original bug case)
        z = graph.call_function(torch.nn.functional.batch_norm, args=(inp, mean, var))

        # Create output node
        graph.output(z)

        # Wrap in a GraphModule
        gm = GraphModule({}, graph)

        device = getattr(self, "device", "cpu")
        gm.to(device)
        gm_compiled = torch.compile(gm, backend="inductor")

        inp_tensor = torch.randn(4, 4, device=device)
        mean_tensor = torch.randn(4, device=device)
        var_tensor = torch.abs(torch.randn(4, device=device))  # Must be positive

        # This should not raise AssertionError
        out = gm_compiled(inp_tensor, mean_tensor, var_tensor)

        # Verify result matches eager evaluation
        expected = gm(inp_tensor, mean_tensor, var_tensor)
        self.assertEqual(out, expected)

    @tf32_on_and_off(0.003)
    @inductor_config.patch({"efficient_conv_bn_eval_fx_passes": True})
    @functorch_config.patch({"enable_autograd_cache": False})
    def test_basic(self):
        # [FUNCTION DOC] Sweeps all combinations of the two test model classes, bias on or off,
        # linear plus conv and transposed-conv layers of 1 to 3 dimensions with matching batch
        # norms, sync batch norm on or off, and decomposition on or off, calling the per-case
        # helper for each. Autograd cache is disabled.
        def test_conv_bn_eval(
            test_class, use_bias, module, sync_bn, decompose_nn_module
        ):
            # [FUNCTION DOC] HELPER: For one test class, bias flag, module pair, sync-batch-norm
            # flag and decompose flag: builds the eager model in eval mode, a deep copy to
            # compile (optionally converting to sync batch norm, optionally traced with make_fx
            # at pre-dispatch), runs forward, backward and an SGD step on both, compares outputs
            # both before and after the update, and asserts the efficient_conv_bn_eval counter
            # grew by the class's expected count.
            from functorch import make_fx
            from torch._dispatch.python import enable_python_dispatcher

            kwargs = {"kernel_size": 3, "stride": 2} if module[0] != nn.Linear else {}
            mod_eager = test_class(
                module[0],
                module[1],
                use_bias,
                3,
                32,
                self.device,
                **kwargs,
            ).eval()
            # Copy module to test backward
            mod_optimized = copy.deepcopy(mod_eager)
            if sync_bn:
                mod_eager = nn.SyncBatchNorm.convert_sync_batchnorm(mod_eager).eval()
                mod_optimized = nn.SyncBatchNorm.convert_sync_batchnorm(
                    mod_optimized
                ).eval()
            torch._dynamo.reset()

            inps = [4, 3]
            # Conv shape goes from big to small, and ConvTranspose shape goes from small to big
            spatial_d = (
                4 if issubclass(module[0], nn.modules.conv._ConvTransposeNd) else 96
            )
            if module[0] is nn.Conv1d or module[0] is nn.ConvTranspose1d:
                inps += [spatial_d] * 1
            if module[0] is nn.Conv2d or module[0] is nn.ConvTranspose2d:
                inps += [spatial_d] * 2
            if module[0] is nn.Conv3d or module[0] is nn.ConvTranspose3d:
                inps += [spatial_d] * 3
            inp = torch.rand(inps).to(self.device)

            if decompose_nn_module:
                with enable_python_dispatcher():
                    mod_optimized = make_fx(mod_optimized, pre_dispatch=True)(inp)
            mod_optimized = torch.compile(mod_optimized)

            original_value = counters["inductor"]["efficient_conv_bn_eval"]

            optim_eager = torch.optim.SGD(mod_eager.parameters(), lr=1e-3)
            optim_optimized = torch.optim.SGD(mod_optimized.parameters(), lr=1e-3)

            optim_eager.zero_grad()
            optim_optimized.zero_grad()

            # test forward
            out_eager = mod_eager(inp)
            out_optimized = mod_optimized(inp)

            self.assertEqual(out_optimized, out_eager)

            out_eager.mean().backward()
            out_optimized.mean().backward()

            optim_eager.step()
            optim_optimized.step()
            # test forward (by testing forward again after one training iteration)
            inp_bw = torch.rand_like(inp)
            out_eager_bw = mod_eager(inp_bw)
            out_optimized_bw = mod_optimized(inp_bw)

            self.assertEqual(out_eager_bw, out_optimized_bw)
            current_value = counters["inductor"]["efficient_conv_bn_eval"]
            self.assertEqual(
                current_value - original_value, test_class.expected_optimization_count
            )

        conv_bias = [True, False]
        modules = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
            (nn.ConvTranspose1d, nn.BatchNorm1d),
            (nn.ConvTranspose2d, nn.BatchNorm2d),
            (nn.ConvTranspose3d, nn.BatchNorm3d),
        ]
        test_classes = [ConvOp, MultiUserConvOp]
        sync_bns = [False, True]
        decompose_nn_modules = [False, True]
        for (
            test_class,
            use_bias,
            module,
            sync_bn,
            decompose_nn_module,
        ) in itertools.product(
            test_classes,
            conv_bias,
            modules,
            sync_bns,
            decompose_nn_modules,
        ):
            test_conv_bn_eval(
                test_class, use_bias, module, sync_bn, decompose_nn_module
            )


if HAS_CPU and not torch.backends.mps.is_available():

    class EfficientConvBNEvalCpuTests(TestCase):
        # [CLASS DOC] CPU instantiation of the template with device set to cpu; created only if
        # a CPU is available and MPS is not.
        device = "cpu"

    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalCpuTests, "cpu")

if HAS_GPU:

    class EfficientConvBNEvalGpuTests(TestCase):
        # [CLASS DOC] GPU instantiation of the template with device set to the GPU type; created
        # only if a GPU is available.
        device = GPU_TYPE

    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalGpuTests, GPU_TYPE)

del EfficientConvBNEvalTemplate

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
