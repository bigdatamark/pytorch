# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for MemTracker (torch.distributed._tools.mem_tracker) on CUDA or XPU (one
# test also runs on the default device): checks that tracked peak memory matches the device
# allocator's active-bytes peak within 10 percent, with and without activation checkpointing, and
# that parameter, gradient and optimizer-state bytes are attributed correctly at each training
# stage.
import gc
import unittest

import torch
import torch.nn as nn
from torch.distributed._tools.mem_tracker import MemTracker
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_CUDA,
    TEST_XPU,
    TestCase,
)
from torch.utils.checkpoint import checkpoint


class TestMemTracker(TestCase):
    # [CLASS DOC] Single-process test case for MemTracker accuracy with small hand-made models
    # and an Adam optimizer.
    def _init_cublas_workspace(self, dev: torch.device):
        # [FUNCTION DOC] HELPER: Runs a forward and backward pass of a 768 by 768 linear layer on
        # the device so that the BLAS workspace is allocated before measurements, then frees the
        # layer and input.
        lin = torch.nn.Linear(768, 768, device=dev)
        inp = torch.randn(1, 768, device=dev)
        lin(inp).sum().backward()
        del lin
        del inp

    def _reset_mem_stats(self, dev: torch.device):
        # [FUNCTION DOC] HELPER: Empties the device allocator cache and resets the accumulated
        # and peak memory statistics of the given device so later peak readings start clean.
        mod = torch.get_device_module(dev)
        mod.empty_cache()
        mod.reset_accumulated_memory_stats(dev)
        mod.reset_peak_memory_stats(dev)

    @unittest.skipIf(
        not TEST_CUDA and not TEST_XPU, "Neither CUDA or XPU is not available"
    )
    def test_accelerator_tracker_equivalence(
        self,
    ):
        # [FUNCTION DOC] Trains a small bfloat16 stack of linear and ReLU layers for two
        # iterations under MemTracker, resetting module stats after the first, then compares the
        # tracker's peak total for the device with the allocator's peak active bytes minus the
        # starting active bytes; the ratio must be 1.0 within a 0.1 tolerance. Skipped without
        # CUDA or XPU.
        """
        Tests that the tracker correctly calculates the peak memory.
        """
        dev = torch.device(torch.accelerator.current_device_index())
        self._init_cublas_workspace(dev)
        gc.collect(1)
        self._reset_mem_stats(dev)
        mod = torch.get_device_module(dev)
        mem_stats = mod.memory_stats(dev)
        pre_acc_active = mem_stats["active_bytes.all.current"]
        bsz, n_layers, dim, dtype = 16, 4, 512, torch.bfloat16

        class DummyModel(nn.Module):
            # [CLASS DOC] HELPER: Model of n_layers pairs of a linear layer and ReLU, in a
            # ModuleList, used to exercise the tracker.
            def __init__(self, n_layers: int, dim: int, dtype: torch.dtype):
                # [FUNCTION DOC] HELPER: Builds the list of alternating linear and ReLU layers of
                # the given width and dtype.
                super().__init__()
                self.linears = nn.ModuleList()
                for _ in range(n_layers):
                    self.linears.append(nn.Linear(dim, dim, dtype=dtype))
                    self.linears.append(nn.ReLU())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies every layer in order and returns the result.
                for layer in self.linears:
                    x = layer(x)
                return x

        with torch.device(dev):
            model = DummyModel(n_layers, dim, dtype=dtype)
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        input_batch = torch.randn(bsz, dim, device=dev, dtype=dtype)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim, input_batch)
        with mem_tracker as mt:
            for iter_idx in range(2):
                model(input_batch).sum().backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()
        # Check for accuracy of peak memory

        tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]
        mem_stats = mod.memory_stats(dev)
        acc_max = mem_stats["active_bytes.all.peak"] - pre_acc_active
        accuracy = tracker_max / acc_max
        self.assertAlmostEqual(accuracy, 1.0, delta=0.1)

    @unittest.skipIf(
        not TEST_CUDA and not TEST_XPU, "Neither CUDA or XPU is not available"
    )
    def test_tracker_with_activation_checkpointing(
        self,
    ):
        # [FUNCTION DOC] Same peak-memory accuracy check as the equivalence test, but with a
        # float16 model of MLP blocks where all blocks after the first are wrapped in
        # non-reentrant activation checkpointing, so recomputation is included in the measured
        # peak. The tracker/allocator ratio must be 1.0 within 0.1. Skipped without CUDA or XPU.
        """
        Tests that the tracker correctly computes the peak memory during activation checkpointing.
        """
        dev = torch.device(torch.accelerator.current_device_index())
        self._init_cublas_workspace(dev)
        gc.collect(1)
        self._reset_mem_stats(dev)
        mod = torch.get_device_module(dev)
        mem_stats = mod.memory_stats(dev)
        pre_acc_active = mem_stats["active_bytes.all.current"]

        bsz, n_layers, dim, dtype = 128, 4, 1024, torch.float16

        class MLPBlock(nn.Module):
            # [CLASS DOC] HELPER: Two-layer feed-forward block (linear up to twice the width,
            # ReLU, linear back) held in a Sequential.
            def __init__(self, dim: int, dtype: torch.dtype):
                # [FUNCTION DOC] HELPER: Builds the up-projection, ReLU and down-projection
                # layers with the given dim and dtype.
                super().__init__()
                self.mlp_block = nn.Sequential(
                    nn.Linear(dim, 2 * dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(2 * dim, dim, dtype=dtype),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the output of the block's Sequential for the
                # input.
                return self.mlp_block(x)

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Model made of n_layers MLPBlock modules; when use_ac is set
            # every block except the first is run through checkpoint with RNG state preserved and
            # non-reentrant mode.
            def __init__(
                self, n_layers: int, dim: int, dtype: torch.dtype, use_ac: bool = False
            ):
                # [FUNCTION DOC] HELPER: Creates the list of MLP blocks and stores the activation
                # checkpointing flag.
                super().__init__()
                self.mlp_blocks = nn.ModuleList()
                self.use_ac = use_ac
                for _ in range(n_layers):
                    self.mlp_blocks.append(MLPBlock(dim, dtype=dtype))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the input through each block, using checkpointing
                # for blocks after the first when enabled, and returns the result.
                for i, block in enumerate(self.mlp_blocks):
                    if i >= 1 and self.use_ac:
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        x = block(x)
                return x

        with torch.device(dev):
            model = MyModule(n_layers, dim, dtype, True)
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim)
        with mem_tracker as mt:
            input_batch = torch.randn(bsz, dim, dim, device=dev, dtype=dtype)
            for iter_idx in range(2):
                model(input_batch).sum().backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()

        # Check for accuracy of peak memory
        tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]
        mem_stats = mod.memory_stats(dev)
        acc_max = mem_stats["active_bytes.all.peak"] - pre_acc_active
        accuracy = tracker_max / acc_max
        self.assertAlmostEqual(accuracy, 1.0, delta=0.1)

    def test_tracker_attribution(self):
        # [FUNCTION DOC] Trains a small float32 MLP with Adam under MemTracker on the default
        # device and, before forward, after backward, after the optimizer step and after
        # zero_grad, checks that the tracker's Parameter, Gradient and Optstate byte totals match
        # the bytes computed directly from the model and optimizer state (exactly 0 when the real
        # value is 0, otherwise within a ratio tolerance of 0.1).
        """
        Tests that the tracker correctly categorizes params, gradients, and optimizer states.
        """
        dev = torch.device(torch.get_default_device())
        gc.collect(1)
        bsz, n_layers, dim, dtype = 16, 3, 128, torch.float32

        def get_param_grad_optstate_actual_bytes(
            model: nn.Module, opt: torch.optim.Optimizer
        ) -> tuple[int, int, int]:
            # [FUNCTION DOC] HELPER: Returns a tuple of parameter bytes, gradient bytes and
            # optimizer state bytes, counting only tensors on the test device, computed from
            # element counts and element sizes of the model parameters, their gradients and the
            # optimizer state tensors.
            param_bytes = 0
            grad_bytes = 0
            opt_state_bytes = 0
            for param in model.parameters():
                if param.device == dev:
                    param_bytes += param.numel() * param.element_size()
                if param.grad is not None and param.grad.device == dev:
                    grad_bytes += param.grad.numel() * param.grad.element_size()

            for state in opt.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor) and v.device == dev:
                        opt_state_bytes += v.numel() * v.element_size()
            return param_bytes, grad_bytes, opt_state_bytes

        def get_param_grad_optstate_bytes_from_tracker(
            tracker: MemTracker,
        ) -> tuple[int, int, int]:
            # [FUNCTION DOC] HELPER: Reads the tracker snapshot for the test device and returns a
            # tuple of its Parameter, Gradient and Optstate byte counts.
            snapshot = tracker.get_tracker_snapshot()
            param_bytes = snapshot[dev]["Parameter"]
            grad_bytes = snapshot[dev]["Gradient"]
            opt_state_bytes = snapshot[dev]["Optstate"]
            return param_bytes, grad_bytes, opt_state_bytes

        def test_attribution_equivalence(
            mt: MemTracker,
            model: nn.Module,
            opt: torch.optim.Optimizer,
        ) -> None:
            # [FUNCTION DOC] HELPER: Compares the actual and tracked tuples element by element: a
            # zero actual value requires a zero tracked value, otherwise the ratio of tracked to
            # actual must be 1.0 within 0.1. Fails with a test assertion otherwise. Named like a
            # test but it is a nested helper.
            actual = get_param_grad_optstate_actual_bytes(model, opt)
            tracker = get_param_grad_optstate_bytes_from_tracker(mt)
            for a, b in zip(actual, tracker):
                if a == 0:
                    self.assertEqual(b, 0)
                else:
                    self.assertAlmostEqual(b / a, 1.0, delta=0.1)

        class DummyModel(nn.Module):
            # [CLASS DOC] HELPER: Model of n_layers repetitions of an expanding linear layer,
            # GELU, a contracting linear layer and GELU, kept in a ModuleList.
            def __init__(self, n_layers: int, dim: int, dtype: torch.dtype):
                # [FUNCTION DOC] HELPER: Builds the list of linear and GELU layers with the given
                # dim and dtype.
                super().__init__()
                self.MLP_layers = nn.ModuleList()
                for _ in range(n_layers):
                    self.MLP_layers.extend(
                        [nn.Linear(dim, 2 * dim, dtype=dtype), nn.GELU()]
                    )
                    self.MLP_layers.extend(
                        [nn.Linear(2 * dim, dim, dtype=dtype), nn.GELU()]
                    )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies every layer in order and returns the result.
                for layer in self.MLP_layers:
                    x = layer(x)
                return x

        with torch.device(dev):
            model = DummyModel(n_layers, dim, dtype=dtype)
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim)
        with mem_tracker as mt:
            input_batch = torch.randn(bsz, dim, device=dev, dtype=dtype)
            # Before forward: Only parameters and input are allocated
            test_attribution_equivalence(mt, model, optim)
            output = model(input_batch)
            output.sum().backward()
            # After backward: Gradients are allocated
            test_attribution_equivalence(mt, model, optim)
            output = None
            optim.step()
            # After step: Optimizer state is allocated
            test_attribution_equivalence(mt, model, optim)
            optim.zero_grad()
            # After zero_grad: Gradients are deallocated
            test_attribution_equivalence(mt, model, optim)


if __name__ == "__main__":
    run_tests()
