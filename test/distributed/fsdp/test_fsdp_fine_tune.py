# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests of FSDP fine-tuning scenarios where some parameters are frozen
# (do not require grad): post-backward resharding still happens for frozen flat parameters,
# parameters used multiple times in one forward work with all hooks, mixed frozen/trainable
# layers match DDP numerically, and a frozen layer with an unused input matches an unfrozen
# reference. Also defines small helper modules. Import-time side effects: computes the device
# type; exits with status 0 if distributed is unavailable or under dev-asan; instantiates the
# device-generic test class for cuda/hpu/xpu. Running the file calls run_tests().

import copy
import sys
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _get_device_module
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous, get_devtype
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
)


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class LinearUnusedInput(nn.Linear):
    # [CLASS DOC] HELPER: Linear layer whose forward takes two inputs but only uses the first;
    # the second is accepted and ignored so it never gets a gradient from this layer.
    def forward(self, frozen_input, learnable_input):
        # [FUNCTION DOC] HELPER: Applies the linear transform to frozen_input only and returns
        # it; learnable_input is ignored.
        return super().forward(frozen_input)


class ModelUnusedInput(nn.Module):
    # [CLASS DOC] HELPER: Three LinearUnusedInput layers (4 to 4), the middle one optionally
    # frozen, whose outputs are concatenated with the second input; used to exercise frozen
    # FSDP modules that are resharded without an unshard in backward.
    def __init__(self, freeze: bool):
        # [FUNCTION DOC] HELPER: Builds layer0, layer1_frozen and layer2; if freeze is true,
        # marks all of layer1_frozen's parameters as not requiring grad.
        super().__init__()
        self.layer0 = LinearUnusedInput(4, 4)
        self.layer1_frozen = LinearUnusedInput(4, 4)
        if freeze:
            for param in self.layer1_frozen.parameters():
                param.requires_grad = False
        self.layer2 = LinearUnusedInput(4, 4)

    def forward(self, frozen_input, learnable_input):
        # [FUNCTION DOC] HELPER: Runs all three layers on frozen_input and returns the
        # concatenation of their three outputs followed by learnable_input along dim 0.
        x = self.layer0(frozen_input, learnable_input)
        y = self.layer1_frozen(frozen_input, learnable_input)
        z = self.layer2(frozen_input, learnable_input)
        return torch.concat([x, y, z, learnable_input])


class TestFSDPFineTune(FSDPTestContinuous):
    # [CLASS DOC] Multi-process test case for FSDP with mixed frozen and trainable parameters.
    # NUM_LINEARS (6) is the number of Linear layers used in the sequential test models.
    """Tests fine-tuning cases where some parameters are frozen."""

    NUM_LINEARS = 6

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Number of processes to launch: accelerator device count, capped at 2.
        return min(_get_device_module(self.device_type).device_count(), 2)

    def _init_seq_module(self, device) -> nn.Module:
        # [FUNCTION DOC] HELPER: Seeds RNG with 42 and builds a Sequential of six
        # (Linear 5 to 5, ReLU) pairs on the device, then freezes every other Linear (the
        # even-indexed ones) via the requires-grad helper. Returns the module.
        torch.manual_seed(42)
        modules = []
        for _ in range(self.NUM_LINEARS):
            modules += [nn.Linear(5, 5, device=device), nn.ReLU()]
        seq = nn.Sequential(*modules)
        self._set_seq_module_requires_grad(seq, False)
        return seq

    def _set_seq_module_requires_grad(self, seq: nn.Module, requires_grad: bool):
        # [FUNCTION DOC] HELPER: Sets requires_grad on all parameters of the even-indexed
        # Linear layers of the Sequential (every other one), leaving the rest unchanged.
        # Works before or after FSDP wrapping since it recurses into parameters.
        # Assume that the linears are leaf modules, meaning that we can pass
        # `recurse=True` to have this to work for both pre/post FSDP wrapping
        for i in range(self.NUM_LINEARS):
            # Only set for every other linear to test mixing frozen/non-frozen
            if i % 2 == 0:
                for param in seq[i * 2].parameters(recurse=True):
                    param.requires_grad = requires_grad

    @skip_if_lt_x_gpu(2)
    def test_backward_reshard_hooks(self, device):
        # [FUNCTION DOC] Runs the post-backward reshard check over three sharding strategies,
        # use_orig_params on/off, input requiring grad on/off, and whether parameters are
        # unfrozen on the final step.
        """
        Tests that the post-backward reshard happens even for flat parameters
        that do not require gradients.
        """
        self.run_subtests(
            {
                "device_id": [device],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
                "inp_requires_grad": [False, True],
                "unfreeze_params": [False, True],
            },
            self._test_backward_reshard_hooks,
        )

    def _test_backward_reshard_hooks(
        self,
        device_id,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        inp_requires_grad: bool,
        unfreeze_params: bool,
    ):
        # [FUNCTION DOC] HELPER: Wraps the half-frozen sequential model with one FSDP per
        # Linear and counts calls to the runtime post-backward reshard function (patched with a
        # counting wrapper). Runs three steps; step 1 is under no_grad (no backward, no hooks
        # expected); on the last step parameters are optionally unfrozen. After each backward it
        # asserts all params require grad when unfrozen and that the reshard count equals six,
        # or five when the input does not require grad and params are still frozen (the first
        # frozen layer is then resharded by the catch-all path). The device_id argument is unused.
        seq = self._init_seq_module(device_type)
        policy = ModuleWrapPolicy({nn.Linear})
        fsdp_kwargs = {"device_id": device_type}
        seq = FSDP(
            seq,
            auto_wrap_policy=policy,
            sharding_strategy=sharding_strategy,
            use_orig_params=use_orig_params,
            **fsdp_kwargs,
        )
        orig_post_backward_reshard = (
            torch.distributed.fsdp._runtime_utils._post_backward_reshard
        )
        post_backward_reshard_count = 0

        def _post_backward_reshard_with_count(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Increments the enclosing reshard counter and delegates to
            # the original post-backward reshard function with the same arguments.
            nonlocal post_backward_reshard_count
            post_backward_reshard_count += 1
            return orig_post_backward_reshard(*args, **kwargs)

        def _assert_post_backward_requires_grad(seq):
            # [FUNCTION DOC] HELPER: On the final step of an unfreezing run, asserts every
            # parameter of seq requires grad. Reads the loop variables step_idx and num_steps
            # from the enclosing scope.
            if step_idx == num_steps - 1 and unfreeze_params:
                self.assertTrue(
                    all(p.requires_grad for p in seq.parameters()),
                    msg="Expected all parameters to require grad but some did not!",
                )

        def _assert_post_backward_reshard_count(step_idx, num_steps):
            # [FUNCTION DOC] HELPER: Asserts the counted reshards since the last reset equal
            # the expected value: six on the unfrozen last step or when the input requires
            # grad, otherwise five.
            if step_idx < num_steps - 1 or not unfreeze_params:
                # If the input does not require gradient, then the 0th
                # frozen linear gets resharded in the catch-all reshard
                # since we cannot register an autograd hook on it
                expected_post_backward_reshard_count = (
                    self.NUM_LINEARS if inp_requires_grad else self.NUM_LINEARS - 1
                )
            else:
                # This follows the normal post-backward hook path
                expected_post_backward_reshard_count = self.NUM_LINEARS
            self.assertEqual(
                post_backward_reshard_count, expected_post_backward_reshard_count
            )

        with mock.patch(
            "torch.distributed.fsdp._runtime_utils._post_backward_reshard",
            _post_backward_reshard_with_count,
        ):
            num_steps = 3
            # interleave a `no_grad` step to validate post-backward hooks are not registered in that context
            # and that `requires_grad` is reset appropriately when unfreezing
            nograd_step_idx = 1
            for step_idx in range(num_steps):
                if unfreeze_params and step_idx == num_steps - 1:
                    # Unfreeze the parameters on the last step to emulate some
                    # kinds of fine-tuning
                    self._set_seq_module_requires_grad(seq, True)

                inp = torch.randn(
                    (8, 5), device=device_type, requires_grad=inp_requires_grad
                )
                if step_idx == nograd_step_idx:
                    with torch.no_grad():
                        output = seq(inp)
                else:
                    output = seq(inp)
                if step_idx != nograd_step_idx:
                    output.sum().backward()
                    _assert_post_backward_requires_grad(seq)
                    _assert_post_backward_reshard_count(step_idx, num_steps)
                    post_backward_reshard_count = 0

    def _init_multi_traversal_module(self, device) -> nn.Module:
        # [FUNCTION DOC] HELPER: Seeds RNG with 42 and returns a small module with three
        # Linear layers (one frozen) whose forward reuses two of them ten times each, including
        # inside a no_grad block, so the same parameters are used repeatedly in one forward.
        torch.manual_seed(42)

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with layer_0, a frozen layer_no_grad and a trainable
            # layer_with_grad; the last two are called repeatedly in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three 5 to 5 Linear layers on the device
                # and freezes layer_no_grad.
                super().__init__()
                self.layer_0 = nn.Linear(5, 5, device=device)
                self.layer_no_grad = nn.Linear(5, 5, device=device)
                self.layer_with_grad = nn.Linear(5, 5, device=device)
                self.layer_no_grad.requires_grad_(False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer_0, then ten times applies
                # layer_with_grad followed by layer_no_grad and adds (in place, under no_grad)
                # another application of layer_with_grad. Returns the final tensor.
                # Layer `layer_no_grad` and `layer_with_grad` are called
                # multiple times, IOW, their parameters are used multiple times
                # during forward pass.
                x = self.layer_0(x)
                for _ in range(10):
                    x = self.layer_no_grad(self.layer_with_grad(x))
                    # Make sure calling the same layer multiple times works
                    # regardless whether gradient is enabled.
                    with torch.no_grad():
                        x += self.layer_with_grad(x)
                return x

        return TestModule()

    @skip_if_lt_x_gpu(2)
    def test_hooks_multi_traversal(self):
        # [FUNCTION DOC] Runs the multi-traversal DDP-parity check over three sharding
        # strategies, use_orig_params on/off, input requires-grad on/off and forward prefetch
        # on/off.
        """
        Tests that the hooks do reshard / unshard correctly in the case of same
        parameters being used multiple times during forward pass.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
                "inp_requires_grad": [False, True],
                "forward_prefetch": [False, True],
            },
            self._test_hooks_multi_traversal,
        )

    def _test_hooks_multi_traversal(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        inp_requires_grad: bool,
        forward_prefetch: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds the multi-traversal module, makes an FSDP copy (one
        # instance per Linear) and a DDP copy, trains both with Adam (lr 0.01) on the same
        # per-rank random batch for six steps, and after each step asserts the two losses are
        # close.
        seq = self._init_multi_traversal_module(device_type.type)
        policy = ModuleWrapPolicy({nn.Linear})
        fsdp_kwargs = {"device_id": device_type}
        fsdp_seq = FSDP(
            copy.deepcopy(seq),
            auto_wrap_policy=policy,
            sharding_strategy=sharding_strategy,
            use_orig_params=use_orig_params,
            forward_prefetch=forward_prefetch,
            **fsdp_kwargs,
        )
        ddp_seq = DDP(copy.deepcopy(seq), device_ids=[device_type])
        fsdp_optim = torch.optim.Adam(fsdp_seq.parameters(), lr=1e-2)
        ddp_optim = torch.optim.Adam(ddp_seq.parameters(), lr=1e-2)
        torch.manual_seed(self.rank + 1)
        losses = []
        for _ in range(6):
            inp = torch.randn(
                (8, 5), device=device_type, requires_grad=inp_requires_grad
            )
            for seq, optim in ((fsdp_seq, fsdp_optim), (ddp_seq, ddp_optim)):
                loss = seq(inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()
            torch.testing.assert_close(losses[0], losses[1])
            losses.clear()

    @skip_if_lt_x_gpu(2)
    def test_parity_with_ddp(self):
        # [FUNCTION DOC] Runs the frozen/trainable-mix DDP-parity check over three sharding
        # strategies and use_orig_params on/off.
        """
        Tests parity with DDP when mixing flat parameters that require and do
        not require gradients.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_parity_with_ddp,
        )

    def _test_parity_with_ddp(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        # [FUNCTION DOC] HELPER: Wraps a copy of the half-frozen sequential model in FSDP and
        # another in DDP, trains both with Adam (lr 0.01) for six steps on identical random
        # inputs, and each step asserts the losses are close (default tolerance on CUDA, 1e-3
        # otherwise).
        seq = self._init_seq_module(device_type)
        policy = ModuleWrapPolicy({nn.Linear})
        fsdp_kwargs = {"device_id": device_type}
        fsdp_seq = FSDP(
            copy.deepcopy(seq),
            auto_wrap_policy=policy,
            sharding_strategy=sharding_strategy,
            use_orig_params=use_orig_params,
            **fsdp_kwargs,
        )
        ddp_seq = DDP(copy.deepcopy(seq), device_ids=[device_type])
        fsdp_optim = torch.optim.Adam(fsdp_seq.parameters(), lr=1e-2)
        ddp_optim = torch.optim.Adam(ddp_seq.parameters(), lr=1e-2)
        torch.manual_seed(self.rank + 1)
        losses = []
        for _ in range(6):
            inp = torch.randn((8, 5), device=device_type.type)
            for seq, optim in ((fsdp_seq, fsdp_optim), (ddp_seq, ddp_optim)):
                loss = seq(inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()
            if TEST_CUDA:
                torch.testing.assert_close(losses[0], losses[1])
            else:
                torch.testing.assert_close(losses[0], losses[1], atol=1e-03, rtol=1e-03)
            losses.clear()

    @skip_if_lt_x_gpu(2)
    def test_parity_with_non_frozen_fsdp(self, device):
        # [FUNCTION DOC] Runs the frozen-vs-unfrozen parity check over two sharding strategies,
        # use_orig_params, CPU offload, default and float16 mixed precision, and two backward
        # prefetch modes.
        """
        For frozen modules with unused input, reshard could happen without unshard
        Verify numerical parity between `_post_backward_reshard_only_hook` and
        `_post_backward_hook` path
        """
        self.run_subtests(
            {
                "device_id": [device],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                ],
                "use_orig_params": [True, False],
                "offload_params": [True, False],
                "mixed_precision": [
                    MixedPrecision(),
                    MixedPrecision(
                        param_dtype=torch.float16,
                        buffer_dtype=torch.float16,
                        reduce_dtype=torch.float16,
                    ),
                ],
                "backward_prefetch": [
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                ],
            },
            self._test_parity_with_non_frozen_fsdp,
        )

    def _test_parity_with_non_frozen_fsdp(
        self,
        device_id,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        offload_params: bool,
        mixed_precision: MixedPrecision,
        backward_prefetch: BackwardPrefetch,
    ):
        # [FUNCTION DOC] HELPER: Builds two identically seeded ModelUnusedInput models, one
        # with its middle layer frozen and one fully trainable, wraps both in FSDP with the
        # same options, and trains six steps; the reference optimizer excludes the frozen
        # layer's parameters so both do equivalent updates. Asserts equal losses each step and,
        # after summoning full parameters, equal final parameters. device_id is unused.
        torch.manual_seed(42)
        model = ModelUnusedInput(freeze=True).to(device_type)
        torch.manual_seed(42)
        ref_model = ModelUnusedInput(freeze=False).to(device_type)
        fsdp_kwargs = {
            "device_id": device_type,
            "auto_wrap_policy": ModuleWrapPolicy({LinearUnusedInput}),
            "sharding_strategy": sharding_strategy,
            "use_orig_params": use_orig_params,
            "cpu_offload": CPUOffload(offload_params=offload_params),
            "mixed_precision": mixed_precision,
            "backward_prefetch": backward_prefetch,
        }
        model = FSDP(model, **fsdp_kwargs)
        ref_model = FSDP(ref_model, **fsdp_kwargs)
        model_optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        ref_model_optim = torch.optim.Adam(
            [
                param
                for name, param in ref_model.named_parameters()
                if not name.startswith("_fsdp_wrapped_module.layer1_frozen")
            ],
            lr=1e-2,
        )
        torch.manual_seed(self.rank + 1)
        losses = []
        for _ in range(6):
            frozen_input = torch.randn((4, 4), device=device_type, requires_grad=False)
            for _model, _optim in ((model, model_optim), (ref_model, ref_model_optim)):
                loss = _model(frozen_input, frozen_input).sum()
                losses.append(loss)
                loss.backward()
                _optim.step()
                _optim.zero_grad()
            self.assertEqual(losses[0], losses[1])
            losses.clear()
        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(ref_model):
                for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                    self.assertEqual(param, ref_param)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestFSDPFineTune, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
