# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for DTensor convolution support (Conv1d/2d/3d forward and backward) on
# a 2-rank mesh. Compares DTensor-based training/forward/backward against plain-tensor
# references for downsampling, depthwise, batch-sharded and last-spatial-dim-sharded
# convolutions, bias=None regressions, and checks communication counts via CommDebugMode.
# Defines _conv_fn (replicates module parameters), the DistConvolutionOpsTest class and a
# local-tensor variant of it created at import time with several send/recv tests skipped.
import copy

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.nn import functional as F
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_distributed import run_subtests
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


ITER_TIME = 10
LR = 0.001


def _conv_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    # [FUNCTION DOC] HELPER: Partition function for distribute_module. Converts every named
    # parameter of the module into an nn.Parameter wrapping a Replicate DTensor on the mesh
    # and re-registers it on the module under the name with dots replaced by underscores
    # (for the flat conv modules this overwrites the same-named parameter). The name
    # argument is unused because the loop variable shadows it. Mutates the module in place.
    for name, param in module.named_parameters():
        dist_spec = [Replicate()]
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        name = "_".join(name.split("."))
        module.register_parameter(name, dist_param)


class DistConvolutionOpsTest(DTensorTestBase):
    # [CLASS DOC] Multi-process (2 ranks) DTensor convolution tests. Each test builds the
    # device mesh, distributes a conv module with replicated parameters and a (usually
    # sharded) input, and compares forward/backward results or trained weights against a
    # deep-copied non-distributed reference model.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size of 2 ranks for these tests.
        # hard code world size to 2
        return 2

    @with_comms
    def test_downsampling_convolution(self):
        # [FUNCTION DOC] Trains a strided Conv2d (kernel 4, stride 4, weights 1, bias 0) for
        # ITER_TIME SGD steps, once with replicated parameters and an input and gradient
        # sharded on the last dim, and once as a plain model with the same data. Asserts the
        # absolute and relative mean squared difference of the local weight and bias versus
        # the reference are each at most 1e-6.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(3)]

        input_list = torch.rand(ITER_TIME, 7, 3, 512, 1024)
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        model = nn.Conv2d(3, 256, kernel_size=4, stride=4, padding=0).to(
            self.device_type
        )
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        model_gt = copy.deepcopy(model).to(self.device_type)

        # training with dtensor
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            output = model(inp_dtensor)
            grad_output = grad_output_list[i].to(self.device_type)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output.backward(grad_output_dtensor)
            optimizer.step()

        # training with plain tensor
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer_gt.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            output = model_gt(inp)
            grad_output = grad_output_list[i].to(self.device_type)
            output.backward(grad_output)
            optimizer_gt.step()

        weight_diff_abs = model.weight.to_local() - model_gt.weight
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            lambda msg: f"{msg}\nToo large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            lambda msg: f"{msg}\nToo large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            lambda msg: f"{msg}\nToo large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            lambda msg: f"{msg}\nToo large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )

    # TODO: test_depthwise_convolution is broken in CI with gloo backend.
    # Temporarily disable it to unblock CI.
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_depthwise_convolution(self):
        # [FUNCTION DOC] Same train-and-compare procedure as the downsampling test but with
        # a depthwise Conv2d (256 groups, kernel 7, padding 3) on inputs sharded on the last
        # dim; asserts weight and bias MSE (absolute and relative) vs the plain model are
        # at most 1e-6. Requires at least 2 GPUs.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(3)]

        input_list = torch.rand(ITER_TIME, 7, 256, 128, 256)
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        model = nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256).to(
            self.device_type
        )
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        model_gt = copy.deepcopy(model).to(self.device_type)

        # training with dtensor
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            output = model(inp_dtensor)
            grad_output = grad_output_list[i].to(self.device_type)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output.backward(grad_output_dtensor)
            optimizer.step()

        # training with plain tensor
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer_gt.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            output = model_gt(inp)
            grad_output = grad_output_list[i].to(self.device_type)
            output.backward(grad_output)
            optimizer_gt.step()

        weight_diff_abs = model.weight.to_local() - model_gt.weight
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            lambda msg: f"{msg}\nToo large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            lambda msg: f"{msg}\nToo large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            lambda msg: f"{msg}\nToo large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            lambda msg: f"{msg}\nToo large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_conv_backward_none_grad_inp(self):
        # [FUNCTION DOC] Runs conv2d on replicated DTensor input, weight and bias, where only
        # weight and bias require grad, then backward. Asserts weight and bias grads are set
        # and the input grad stays None. Requires at least 2 GPUs.
        device_mesh = self.build_device_mesh()
        conv = nn.Conv2d(64, 64, 3, padding=1).train()
        x = torch.randn(1, 64, 32, 32)
        x_dt = DTensor.from_local(x, device_mesh, [Replicate()])
        w = conv.weight
        w_dt = torch.nn.Parameter(DTensor.from_local(w, device_mesh, [Replicate()]))

        b = conv.bias
        b_dt = torch.nn.Parameter(DTensor.from_local(b, device_mesh, [Replicate()]))

        res = F.conv2d(x_dt, w_dt, b_dt, padding=1)
        dres = torch.rand_like(res)
        res.backward(dres)
        self.assertTrue(w_dt.grad is not None)
        self.assertTrue(b_dt.grad is not None)
        self.assertTrue(x_dt.grad is None)

    def _run_single_arg_fwd(
        self, model, arg, placements=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Runs one forward pass of the model both distributed and
        # locally. Wraps arg as a DTensor with the given placements (default replicated),
        # runs the module distributed with replicated params, and runs a deep copy of the
        # original on the gathered full tensor. Returns (distributed full output, local
        # output).
        """Given model and arg, runs fwd model local and distbuted given device_mesh"""
        device_mesh = self.build_device_mesh()
        model_copy = copy.deepcopy(model).to(device=self.device_type)
        dist_model = distribute_module(model, device_mesh, _conv_fn)
        arg_dt = DTensor.from_local(arg, device_mesh, placements)
        out_dt = dist_model(arg_dt.to(device=self.device_type))
        out = model_copy(arg_dt.full_tensor())
        return (out_dt.full_tensor(), out)

    @with_comms
    def test_conv1d(self):
        # [FUNCTION DOC] Checks a Conv1d forward with a replicated input matches the local
        # result.
        model = nn.Conv1d(64, 64, 3, padding=1)
        x = torch.randn(1, 64, 8, device=self.device_type)
        out_dt, out = self._run_single_arg_fwd(model, x)
        self.assertEqual(out_dt, out)

    @with_comms
    def test_conv3d(self):
        # [FUNCTION DOC] Checks a Conv3d forward with the input built from a local tensor as
        # Shard(0) matches the local result on the gathered full tensor.
        model = nn.Conv3d(64, 64, 3, padding=1)
        x = torch.randn(1, 64, 8, 8, 8, device=self.device_type)
        out_dt, out = self._run_single_arg_fwd(model, x, [Shard(0)])
        self.assertEqual(out_dt, out)

    @with_tf32_off
    @with_comms
    def test_conv2d_no_bias_compile(self):
        # [FUNCTION DOC] Regression test for issue 167091: a torch.compiled conv2d with bias
        # None on replicated DTensors must not crash, must give output shape [1, 8, 5, 5],
        # and must match the eager DTensor result locally. TF32 is disabled by decorator.
        """Test Conv2d with bias=False in compile mode (Issue #167091)

        Regression test: Previously this would fail during torch.compile
        tracing with AssertionError when bias_spec was None.
        """
        device_mesh = self.build_device_mesh()

        def conv_fn(x, w):
            # [FUNCTION DOC] HELPER: conv2d of x with weight w, no bias, padding 1.
            return F.conv2d(x, w, bias=None, padding=1)

        compiled_fn = torch.compile(conv_fn)

        # Create tensors
        x = torch.randn(1, 4, 5, 5, device=self.device_type)
        w = torch.randn(8, 4, 3, 3, device=self.device_type)

        # Distribute tensors
        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        w_dt = distribute_tensor(w, device_mesh, [Replicate()])

        # Test eager mode for comparison
        result_eager = conv_fn(x_dt, w_dt)

        # Test compiled mode - this should not crash
        result_compiled = compiled_fn(x_dt, w_dt)

        # Verify shape is correct (the key regression test)
        self.assertEqual(result_compiled.shape, torch.Size([1, 8, 5, 5]))

        # Verify numerical correctness
        self.assertEqual(result_compiled.to_local(), result_eager.to_local())

    @with_comms
    def test_conv2d_no_bias_backward(self):
        # [FUNCTION DOC] Regression test for issue 167091: eager conv2d with bias None on
        # replicated DTensors, then backward with a random gradient. Asserts the weight
        # parameter receives a gradient of shape [8, 4, 3, 3].
        """Test Conv2d backward pass with bias=False (Issue #167091)

        Regression test: Previously backward pass would fail when
        grad_bias_spec was None.
        """
        device_mesh = self.build_device_mesh()

        # Create tensors with requires_grad
        x = torch.randn(1, 4, 5, 5, device=self.device_type)
        w = torch.randn(8, 4, 3, 3, device=self.device_type, requires_grad=True)

        # Distribute tensors
        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        w_dt = torch.nn.Parameter(distribute_tensor(w, device_mesh, [Replicate()]))

        # Forward pass
        result = F.conv2d(x_dt, w_dt, bias=None, padding=1)

        # Backward pass - this should not crash
        grad_output = torch.randn_like(result)
        result.backward(grad_output)

        # Check weight gradient exists (the key regression test)
        self.assertIsNotNone(w_dt.grad)
        self.assertEqual(w_dt.grad.shape, torch.Size([8, 4, 3, 3]))

    @with_tf32_off
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_conv2d_batch_shard_strided(self):
        # [FUNCTION DOC] Strided (stride 2, kernel 7) Conv2d with the input sharded on the
        # batch dim: forward and backward must perform zero collectives (CommDebugMode total
        # count 0) and the gathered output and input gradient must match a plain reference.
        # Requires at least 2 GPUs.
        """Batch-dim sharding with stride != 1 should not hit _is_supported."""
        device_mesh = self.build_device_mesh()

        model = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3).to(self.device_type)
        model_ref = copy.deepcopy(model).to(self.device_type)
        model = distribute_module(model, device_mesh, _conv_fn)

        x = torch.randn(4, 3, 16, 16, device=self.device_type, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        x_dt = distribute_tensor(x, device_mesh, [Shard(0)])

        comm_mode = CommDebugMode()
        with comm_mode:
            out_dt = model(x_dt)
        self.assertEqual(comm_mode.get_total_counts(), 0)

        out_ref = model_ref(x_ref)
        self.assertEqual(out_dt.full_tensor(), out_ref)

        grad = torch.randn_like(out_ref)
        grad_dt = distribute_tensor(grad, device_mesh, [Shard(0)])
        with comm_mode:
            out_dt.backward(grad_dt)
        self.assertEqual(comm_mode.get_total_counts(), 0)

        out_ref.backward(grad)
        self.assertEqual(x_dt.grad.full_tensor(), x_ref.grad)

    @with_tf32_off
    @with_comms
    def test_convolution_last_dim_shard(self):
        # [FUNCTION DOC] Runs the last-dim-shard check for 1-D, 2-D and 3-D convolutions via
        # run_subtests with the parameter conv_dim in 1, 2, 3.
        run_subtests(
            self,
            {"conv_dim": [1, 2, 3]},
            self._test_convolution_last_dim_shard,
        )

    def _test_convolution_last_dim_shard(self, conv_dim):
        # [FUNCTION DOC] HELPER: For conv_dim in 1..3 builds the matching Conv class with
        # kernel 4, stride 4 and padding 0 on the last spatial dim (size 16), other spatial
        # dims size 8 with kernel 3, stride 1, padding 1. Shards the input on its last dim,
        # and checks forward output and input gradient placements, shapes and gathered values
        # against a plain reference model.
        device_mesh = self.build_device_mesh()
        conv_cls = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[conv_dim - 1]
        spatial_shape = (8,) * (conv_dim - 1) + (16,)
        kernel_size = (3,) * (conv_dim - 1) + (4,)
        stride = (1,) * (conv_dim - 1) + (4,)
        padding = (1,) * (conv_dim - 1) + (0,)

        model = conv_cls(
            3, 8, kernel_size=kernel_size, stride=stride, padding=padding
        ).to(self.device_type)
        model_ref = copy.deepcopy(model).to(self.device_type)
        model = distribute_module(model, device_mesh, _conv_fn)

        x = torch.randn(
            4, 3, *spatial_shape, device=self.device_type, requires_grad=True
        )
        placements = (Shard(x.ndim - 1),)
        x_ref = x.detach().clone().requires_grad_(True)
        x_dt = distribute_tensor(x, device_mesh, placements)

        out_dt = model(x_dt)
        out_ref = model_ref(x_ref)
        self.assertEqual(out_dt.placements, placements)
        self.assertEqual(out_dt.shape, out_ref.shape)
        self.assertEqual(out_dt.full_tensor(), out_ref)

        grad = torch.randn_like(out_ref)
        grad_dt = distribute_tensor(grad, device_mesh, placements)
        out_dt.backward(grad_dt)
        out_ref.backward(grad)

        self.assertEqual(x_dt.grad.placements, placements)
        self.assertEqual(x_dt.grad.shape, x_ref.grad.shape)
        self.assertEqual(x_dt.grad.full_tensor(), x_ref.grad)

    @with_comms
    def test_conv2d_module_no_bias(self):
        # [FUNCTION DOC] Regression test for issue 167091: an nn.Conv2d with bias=False,
        # distributed with replicated params, runs forward on a replicated DTensor input,
        # gives output shape [1, 8, 5, 5], and the module bias remains None.
        """Test nn.Conv2d module with bias=False (Issue #167091)

        Regression test: Ensures nn.Conv2d with bias=False works with DTensor.
        """
        device_mesh = self.build_device_mesh()

        # Create model with bias=False
        model = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False).to(
            self.device_type
        )
        nn.init.ones_(model.weight)

        # Distribute model
        model_dt = distribute_module(model, device_mesh, _conv_fn)

        # Create input
        x = torch.randn(1, 4, 5, 5, device=self.device_type)
        x_dt = distribute_tensor(x, device_mesh, [Replicate()])

        # Forward pass - this should not crash
        output_dt = model_dt(x_dt)

        # Check outputs shape is correct
        self.assertEqual(output_dt.shape, torch.Size([1, 8, 5, 5]))

        # Check that model.bias is None
        self.assertIsNone(model.bias)

    @with_tf32_off
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_conv1d_batch_shard(self):
        # [FUNCTION DOC] Conv1d with the input sharded on the batch dim: forward and backward
        # must issue no collectives, the output and the input gradient stay Shard(0), and
        # gathered values match a plain reference. Requires at least 2 GPUs.
        device_mesh = self.build_device_mesh()

        model = nn.Conv1d(3, 8, kernel_size=3, padding=1).to(self.device_type)
        model_ref = copy.deepcopy(model).to(self.device_type)
        model = distribute_module(model, device_mesh, _conv_fn)

        x = torch.randn(4, 3, 16, device=self.device_type, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        x_dt = distribute_tensor(x, device_mesh, [Shard(0)])

        comm_mode = CommDebugMode()
        with comm_mode:
            out_dt = model(x_dt)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertTrue(out_dt.placements[0].is_shard(0))

        out_ref = model_ref(x_ref)
        self.assertEqual(out_dt.full_tensor(), out_ref)

        grad = torch.randn_like(out_ref)
        grad_dt = distribute_tensor(grad, device_mesh, [Shard(0)])
        with comm_mode:
            out_dt.backward(grad_dt)
        self.assertEqual(comm_mode.get_total_counts(), 0)

        out_ref.backward(grad)
        self.assertTrue(x_dt.grad.placements[0].is_shard(0))
        self.assertEqual(x_dt.grad.full_tensor(), x_ref.grad)

    @with_tf32_off
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_conv3d_batch_shard(self):
        # [FUNCTION DOC] Conv3d counterpart of the batch-shard test: no collectives in forward
        # or backward, output and input gradient stay Shard(0), values match a plain
        # reference. Requires at least 2 GPUs.
        device_mesh = self.build_device_mesh()

        model = nn.Conv3d(3, 8, kernel_size=3, padding=1).to(self.device_type)
        model_ref = copy.deepcopy(model).to(self.device_type)
        model = distribute_module(model, device_mesh, _conv_fn)

        x = torch.randn(4, 3, 8, 8, 8, device=self.device_type, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        x_dt = distribute_tensor(x, device_mesh, [Shard(0)])

        comm_mode = CommDebugMode()
        with comm_mode:
            out_dt = model(x_dt)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertTrue(out_dt.placements[0].is_shard(0))

        out_ref = model_ref(x_ref)
        self.assertEqual(out_dt.full_tensor(), out_ref)

        grad = torch.randn_like(out_ref)
        grad_dt = distribute_tensor(grad, device_mesh, [Shard(0)])
        with comm_mode:
            out_dt.backward(grad_dt)
        self.assertEqual(comm_mode.get_total_counts(), 0)

        out_ref.backward(grad)
        self.assertTrue(x_dt.grad.placements[0].is_shard(0))
        self.assertEqual(x_dt.grad.full_tensor(), x_ref.grad)


DistConvolutionOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistConvolutionOpsTest,
    # Send / recv ops are not supported
    skipped_tests=[
        "test_conv_backward_none_grad_inp",
        "test_depthwise_convolution",
        "test_downsampling_convolution",
        "test_conv2d_batch_shard_strided",
        # New tests for Issue #167091 - use send/recv via tp_convolution
        "test_conv2d_no_bias_compile",
        "test_conv2d_no_bias_backward",
        "test_conv2d_module_no_bias",
        "test_conv1d_batch_shard",
        "test_conv3d_batch_shard",
    ],
)

if __name__ == "__main__":
    run_tests()
