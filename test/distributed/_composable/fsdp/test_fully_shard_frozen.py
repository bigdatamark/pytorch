# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests of fully_shard (FSDP2) with frozen parameters
# (requires_grad False). Each test compares training against a DDP-wrapped
# reference model. Covers frozen/non-frozen mixes inside one communication
# group, across groups (with optional unfreezing), and modules that run several
# times per forward. Defines one FSDPTest subclass and a device_type global
# derived from get_devtype; no other import-time side effects.

import copy
import functools
import itertools

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable import checkpoint, replicate
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    RegisterPostBackwardFunction,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    get_devtype,
    MLP,
    patch_reduce_scatter,
    patch_register_post_backward_hook_backward,
    reduce_scatter_with_assert,
)
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


class TestFullyShardFrozen(FSDPTest):
    # [CLASS DOC] Test case for fully_shard on models with frozen parameters.
    # Stateless apart from the FSDPTest process-group setup; the world size is
    # capped at 4 ranks. Each public test runs subtests over a config grid.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of ranks: the smaller of 4 and the
        # number of devices available for the current device type.
        return min(4, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_mixed_requires_grad_per_group(self):
        # [FUNCTION DOC] Runs the per-group mixed-frozen check over a grid of
        # reshard_after_forward (False, True, 2), activation checkpointing on/off,
        # and whether freezing happens before or after fully_shard. Needs 2+ GPUs.
        """
        Tests training parity with DDP when mixing frozen and non-frozen
        parameters in the same FSDP communication group. This checks that
        the reduce-scatters reduce the expected numel and that they are called
        via the custom autograd function backward (i.e. that they are not
        delayed until the end of backward).
        """
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "use_activation_checkpointing": [False, True],
                "freeze_after_init": [False, True],
            },
            self._test_train_mixed_requires_grad_per_group,
        )

    def _test_train_mixed_requires_grad_per_group(
        self,
        reshard_after_forward: bool | int,
        use_activation_checkpointing: bool,
        freeze_after_init: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds three MLPs where only biases train,
        # either frozen before sharding or after it. Wraps a DDP reference and an
        # FSDP model (each MLP plus root, optionally checkpointed), then trains 10
        # iterations with Adam. Patches reduce-scatter to assert its output numel
        # equals the local bias numel, and counts post-backward autograd function
        # calls. Checks sharded parity and equal losses each step, and that at
        # least num_mlps minus one hooks ran in autograd backward. Raises
        # AssertionError if a child is not an MLP or a check fails.
        torch.manual_seed(42)
        num_mlps, lin_dim = (3, 32)
        model = nn.Sequential(
            *[MLP(lin_dim, torch.device("cpu")) for _ in range(num_mlps)]
        )
        # Train biases only (e.g. like BitFit)
        if not freeze_after_init:
            for param_name, param in model.named_parameters():
                if "bias" not in param_name:
                    param.requires_grad_(False)
        ref_model = replicate(
            copy.deepcopy(model).to(device_type),
            device_ids=[self.rank],
            find_unused_parameters=freeze_after_init,
        )
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            if use_activation_checkpointing:
                checkpoint(mlp)
            fully_shard(mlp, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        orig_reduce_scatter = dist.reduce_scatter_single
        if freeze_after_init:
            for param_name, param in itertools.chain(
                model.named_parameters(), ref_model.named_parameters()
            ):
                if "bias" not in param_name:
                    param.requires_grad_(False)
        for mlp in model:
            if not isinstance(mlp, MLP):
                raise AssertionError(
                    "The reduce-scatter numel check assumes the model consists of "
                    f"only the same MLP class but got {type(mlp)}"
                )
        expected_numel = sum(
            p._local_tensor.numel()
            for n, p in model[0].named_parameters()
            if "bias" in n
        )

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor has
            # exactly the expected number of elements (local bias elements only).
            self.assertEqual(output.numel(), expected_numel)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        orig_backward = RegisterPostBackwardFunction.backward
        backward_count = 0

        def backward_with_count(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper for the post-backward function's
            # backward. Increments the enclosing backward_count, then delegates to
            # the original backward and returns its result.
            nonlocal backward_count
            backward_count += 1
            return orig_backward(*args, **kwargs)

        torch.manual_seed(42 + self.rank + 1)
        device = device_type
        with (
            patch_reduce_scatter(reduce_scatter),
            patch_register_post_backward_hook_backward(backward_with_count),
        ):
            for iter_idx in range(10):
                inp = torch.randn((8, lin_dim), device=device)
                losses: list[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                    losses.append(_model(inp).sum())
                    losses[-1].backward()
                    _optim.step()
                check_sharded_parity(self, ref_model, model)
                self.assertEqual(losses[0], losses[1])
                # Check that the post-backward hooks ran through the autograd
                # backward, not the final callback (except possibly that of the
                # first MLP, which does not have an input that requires grad)
                self.assertTrue(backward_count >= num_mlps - 1)

    @skip_if_lt_x_gpu(2)
    def test_train_mixed_requires_grad_across_groups(self):
        # [FUNCTION DOC] Runs the across-groups frozen check over reshard_after_forward
        # (False, True, 2) and whether frozen params are unfrozen on the last
        # iteration. Needs 2+ GPUs.
        """
        Tests training parity with DDP when mixing frozen and non-frozen
        parameters across different FSDP communication groups, including
        possibly unfreezing parameters.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "unfreeze_params": [False, True],
            },
            self._test_train_mixed_requires_grad_across_groups,
        )

    def _test_train_mixed_requires_grad_across_groups(
        self,
        reshard_after_forward: bool | int,
        unfreeze_params: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds six Linear+ReLU blocks, shards each Linear
        # as its own group (no root wrap), and keeps a DDP reference with unused
        # parameter detection. Freezes alternating layers, then runs 3 iterations
        # where iteration 1 is forward-only under no_grad and the last optionally
        # unfreezes the FSDP model's params. Finally asserts equal losses and that
        # at least num_linears minus one post-backward calls came from autograd.
        torch.manual_seed(42)
        num_linears, lin_dim = (6, 32)
        modules: list[nn.Module] = []
        for _ in range(num_linears):
            modules += [nn.Linear(lin_dim, lin_dim), nn.ReLU()]
        model = nn.Sequential(*modules)
        ref_model = replicate(
            copy.deepcopy(model).to(device_type),
            device_ids=[self.rank],
            find_unused_parameters=True,
        )
        for module in model.modules():
            if isinstance(module, nn.Linear):
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        orig_backward = RegisterPostBackwardFunction.backward
        backward_count = 0

        def _set_requires_grad(seq: nn.Module, requires_grad: bool):
            # [FUNCTION DOC] HELPER: Sets requires_grad on the parameters of the
            # even-indexed layers of the sequential. Mutates the parameters.
            # KNOWN ISSUE: the inner index is i mod 2, which is always 0 for even
            # i, so only layer 0 is ever modified, not every other layer as the
            # comment suggests.
            for i in range(num_linears):
                # Interleave frozen -> non-frozen -> ... linears
                if i % 2 == 0:
                    for param in seq[i % 2].parameters():
                        param.requires_grad_(requires_grad)

        def backward_with_count(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Counts calls to the post-backward function's
            # backward via the enclosing backward_count, then delegates to the
            # original backward and returns its result.
            nonlocal backward_count
            backward_count += 1
            return orig_backward(*args, **kwargs)

        _set_requires_grad(model, False)
        _set_requires_grad(ref_model, False)
        num_iters, no_grad_iter_idx = (3, 1)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn((8, lin_dim), device=device_type)
        with patch_register_post_backward_hook_backward(backward_with_count):
            for iter_idx in range(num_iters):
                losses: list[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    # Unfreeze the parameters on the last step to emulate some
                    # kinds of fine-tuning
                    if unfreeze_params and iter_idx == num_iters - 1:
                        _set_requires_grad(model, True)
                    if iter_idx == no_grad_iter_idx:
                        with torch.no_grad():
                            losses.append(_model(inp).sum())
                    else:
                        losses.append(_model(inp).sum())
                        losses[-1].backward()
                        _optim.step()
                        _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            self.assertEqual(losses[0], losses[1])
            # Check that the post-backward hooks ran through the autograd
            # backward, not the final callback (except possibly that of the
            # first linear, which does not have an input that requires grad)
            self.assertTrue(backward_count >= num_linears - 1)

    @skip_if_lt_x_gpu(2)
    def test_multi_forward_mixed_requires_grad(self):
        # [FUNCTION DOC] Runs the multi-forward frozen/trainable check for each
        # reshard_after_forward value (True, False, 2). Needs 2+ GPUs.
        """
        Tests training parity with DDP when having trainable and frozen modules
        that participate multiple times in forward.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_multi_forward_mixed_requires_grad,
        )

    def _test_multi_forward_mixed_requires_grad(
        self,
        reshard_after_forward: bool | int,
    ):
        # [FUNCTION DOC] HELPER: Shards every Linear of a module that calls a
        # frozen and a trainable Linear three times each per forward (also once
        # under no_grad), plus the root. Trains 10 iterations against a DDP
        # reference and asserts equal losses each iteration.
        class MultiForwardModule(nn.Module):
            # [CLASS DOC] HELPER: Test model with three Linear layers; layer_no_grad
            # is frozen. Its forward reuses layer_no_grad and layer_with_grad
            # repeatedly, including calls under no_grad.
            def __init__(self, device: torch.device):
                # [FUNCTION DOC] HELPER: Creates the three 5x5 Linear layers on the
                # given device and freezes layer_no_grad.
                super().__init__()
                self.layer_0 = nn.Linear(5, 5, device=device)
                self.layer_no_grad = nn.Linear(5, 5, device=device)
                self.layer_with_grad = nn.Linear(5, 5, device=device)
                self.layer_no_grad.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies layer_0, then three rounds of
                # relu(layer_with_grad) into layer_no_grad, each followed by an
                # in-place add of relu(layer_with_grad(x)) computed under no_grad.
                # Returns the final tensor.
                x = self.layer_0(x)
                for _ in range(3):
                    x = self.layer_no_grad(F.relu(self.layer_with_grad(x)))
                    # Make sure that calling the same layer multiple times
                    # works regardless whether gradient is enabled
                    with torch.no_grad():
                        x += F.relu(self.layer_with_grad(x))
                return x

        torch.manual_seed(42)
        model = MultiForwardModule(torch.device("cpu"))
        ref_model = replicate(
            copy.deepcopy(model).to(device_type), device_ids=[self.rank]
        )
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        for iter_idx in range(10):
            inp = torch.randn((8, 5), device=device_type)
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
