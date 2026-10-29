# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests of FSDP with use_orig_params=True. Covers: parity with DDP when
# parameter groups have different hyperparameters or trainability, multiple optimizers,
# torch.compile of an FSDP model, parity with use_orig_params=False under unshard/reshard flows, access to the
# sharded original parameters between forward and backward, writeback of user changes to
# parameters and gradients, named_parameters FQNs, no_sync behavior, non-uniform requires_grad
# initialization, and a multi-tensor-apply regression for zero-sized tensors. Import-time:
# exits the process with status 0 if distributed is unavailable or dev-asan is on; computes
# device_type and instantiates parametrized tests for five of the test classes.

import copy
import functools
import itertools
import os
import sys
import unittest
from typing import Any

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.fsdp._flat_param import (
    _FSDP_SKIP_WRITEBACK_CHECK,
    _FSDP_USE_FULL_PREC_IN_EVAL,
)
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_GPU


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestFSDPUseOrigParamsMultipleParamGroups(FSDPTest):
    # [CLASS DOC] Two-rank tests comparing FSDP (use_orig_params=True) against DDP on a shared
    # parameter transformer when the optimizer has several parameter groups (weights, biases,
    # other) with different hyperparameters, frozen biases, or several optimizers. Holds helpers
    # that build the DDP and FSDP models, optimizers and compare training and parameters.
    """Tests multiple parameter groups."""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of ranks to spawn: always 2.
        return 2

    def _get_param_groups(self, model: nn.Module) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Splits the model's named parameters into three optimizer groups:
        # names containing weight (decay 0.1, lr 0.01), names containing bias (decay 0.01, lr
        # 0.001), and everything else (default options). Returns the list of group dictionaries.
        """
        Constructs separate parameter groups for weights, biases, and other
        parameters.
        """
        param_groups = [
            {"params": [], "weight_decay": 0.1, "lr": 1e-2},
            {"params": [], "weight_decay": 0.01, "lr": 1e-3},
            {"params": []},
        ]
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                param_groups[0]["params"].append(param)
            elif "bias" in param_name:
                param_groups[1]["params"].append(param)
            else:
                param_groups[2]["params"].append(param)
        return param_groups

    def _get_optim(
        self,
        model: nn.Module,
        optim_class: type[torch.optim.Optimizer],
        multi_tensor: bool,
    ) -> torch.optim.Optimizer:
        # [FUNCTION DOC] HELPER: Builds an optimizer of the given class over the three parameter
        # groups from _get_param_groups, with default learning rate 0.005 and the foreach flag
        # set from multi_tensor.
        # KNOWN ISSUE: the docstring says it constructs an Adam optimizer, but the class is
        # a parameter.
        """
        Constructs an Adam optimizer with three parameter groups, one for
        weights, one for biases, and one for everything else, each with
        different weight decay and learning rates.
        """
        param_groups = self._get_param_groups(model)
        return optim_class(param_groups, lr=5e-3, foreach=multi_tensor)

    def _get_ddp_transformer(self, find_unused_params: bool) -> DDP:
        # [FUNCTION DOC] HELPER: Builds a deterministic transformer with shared parameters on the
        # device (no FSDP) and wraps it in DDP for this rank, passing find_unused_params through.
        """Returns a transformer with shared parameters wrapped with DDP."""
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        ddp_model = DDP(
            model,
            device_ids=[self.rank],
            find_unused_parameters=find_unused_params,
        )
        return ddp_model

    def _get_fsdp_transformer_and_optim(
        self,
        device_init_mode: DEVICEInitMode,
        init_optim_before_wrap: bool,
        optim_class: type[torch.optim.Optimizer],
        multi_tensor: bool,
        sharding_strategy: ShardingStrategy,
        backward_prefetch: BackwardPrefetch | None,
        cpu_offload: CPUOffload,
    ) -> tuple[FSDP, torch.optim.Optimizer]:
        # [FUNCTION DOC] HELPER: Builds a deterministic shared-parameter transformer wrapped with
        # FSDP (use_orig_params, encoder and decoder layers wrapped separately) using the given
        # sharding strategy, prefetch and CPU offload, plus an optimizer made either before or
        # after wrapping per init_optim_before_wrap. Moves the model to the device afterward for
        # DEVICE_AFTER unless parameters are offloaded. Returns the (FSDP model, optimizer) pair.
        """
        Returns a transformer with shared parameters wrapped with FSDP and a
        corresponding optimizer.
        """
        # Each transformer layer has multiple linear layers, so this policy, in
        # combination with the parameter group construction, ensures different
        # hyperparameter settings within one `FlatParameter`
        fsdp_kwargs = {
            "auto_wrap_policy": ModuleWrapPolicy(
                {
                    TransformerEncoderLayer,
                    TransformerDecoderLayer,
                }
            ),
            "use_orig_params": True,
            "sharding_strategy": sharding_strategy,
            "backward_prefetch": backward_prefetch,
            "cpu_offload": cpu_offload,
        }
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            device_init_mode,
            deterministic=True,
        )
        if init_optim_before_wrap:
            fsdp_optim = self._get_optim(model, optim_class, multi_tensor)
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
        else:
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
            fsdp_optim = self._get_optim(fsdp_model, optim_class, multi_tensor)
        if (
            device_init_mode == DEVICEInitMode.DEVICE_AFTER
            and not fsdp_model.cpu_offload.offload_params
        ):
            fsdp_model = fsdp_model.to(device=device_type)
        return fsdp_model, fsdp_optim

    def _check_train_parity(
        self,
        ddp_model: DDP,
        ddp_optim: torch.optim.Optimizer,
        fsdp_model: FSDP,
        fsdp_optim: torch.optim.Optimizer,
        set_to_none: bool,
        num_iters: int = 10,
    ):
        # [FUNCTION DOC] HELPER: Trains the DDP and FSDP models in lockstep for num_iters steps,
        # alternating whether gradients are zeroed before the forward or before the backward
        # (with set_to_none passed through). Asserts the two losses are close every iteration
        # (moving the DDP model to CPU for its optimizer step when FSDP offloads parameters), then
        # compares final parameters.
        """Checks training parity between DDP and FSDP."""
        device = torch.device(device_type)
        for i in range(num_iters):
            iter_losses = []
            for model, optim in ((ddp_model, ddp_optim), (fsdp_model, fsdp_optim)):
                module = model.module
                # Test two different `zero_grad()` timings
                if i % 2 == 0:
                    optim.zero_grad(set_to_none=set_to_none)  # pre-forward
                inp = module.get_input(device)
                output = model(*inp)
                loss = module.get_loss(inp, output).to(device)
                iter_losses.append(loss)
                if i % 2 == 1:
                    optim.zero_grad(set_to_none=set_to_none)  # pre-backward
                module.run_backward(loss)
                # Perform the DDP optimizer step on CPU to match FSDP if needed
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(torch.device("cpu"))
                optim.step()
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(device)
            torch.testing.assert_close(iter_losses[0], iter_losses[1])
            iter_losses.clear()
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)

    def _check_ddp_fsdp_param_parity(self, ddp_model: DDP, fsdp_model: FSDP):
        # [FUNCTION DOC] HELPER: Gathers the full FSDP parameters and asserts each matches the
        # corresponding DDP parameter in name (after stripping FSDP wrapper prefixes) and value.
        with FSDP.summon_full_params(fsdp_model):
            for (n1, p1), (n2, p2) in zip(
                ddp_model.module.named_parameters(), fsdp_model.named_parameters()
            ):
                # Allow for FSDP prefixes
                self.assertEqual(n1, clean_tensor_name(n2))
                torch.testing.assert_close(p1, p2)

    def _get_sharding_strategy_from_str(
        self, sharding_strategy_str: str
    ) -> ShardingStrategy:
        # [FUNCTION DOC] HELPER: Maps the string no_shard, shard_grad_op or full_shard to the
        # matching ShardingStrategy member; raises ValueError for any other string.
        if sharding_strategy_str == "no_shard":
            sharding_strategy = ShardingStrategy.NO_SHARD
        elif sharding_strategy_str == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy_str == "full_shard":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            raise ValueError(f"Invalid string: {sharding_strategy_str}")
        return sharding_strategy

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_fsdp_compile(self):
        # [FUNCTION DOC] Runs the compile check as subtests over three sharding strategies and
        # with fsdp guards skipped or not. Needs a GPU with Inductor support and 2 GPUs.
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "skip_fsdp_guards": [True, False],
            },
            self._test_fsdp_compile,
        )

    def _test_fsdp_compile(
        self, sharding_strategy: ShardingStrategy, skip_fsdp_guards: bool
    ):
        # [FUNCTION DOC] HELPER: Sets the dynamo option for skipping FSDP guards, wraps copies of
        # one transformer in FSDP, compiles one of them, and trains both for 10 Adam steps on
        # identical inputs asserting equal losses.
        # KNOWN ISSUE: assigns the global dynamo config option without restoring it afterward,
        # leaking the setting into later tests in the process.
        torch._dynamo.config.skip_fsdp_guards = skip_fsdp_guards
        fsdp_kwargs = {
            "auto_wrap_policy": ModuleWrapPolicy(
                {
                    TransformerEncoderLayer,
                    TransformerDecoderLayer,
                }
            ),
            "use_orig_params": True,
            "sharding_strategy": sharding_strategy,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "cpu_offload": CPUOffload(False),
        }
        base_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        ref_model = FSDP(copy.deepcopy(base_model), self.process_group, **fsdp_kwargs)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        model = FSDP(copy.deepcopy(base_model), self.process_group, **fsdp_kwargs)
        model = torch.compile(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            losses = []
            inp = ref_model.get_input(torch.device(device_type))
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                loss = _model(*inp).sum()
                losses.append(loss)
                loss.backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy_str",
        ["no_shard", "shard_grad_op", "full_shard"],
    )
    def test_diff_hyperparams(self, sharding_strategy_str: str):
        # [FUNCTION DOC] Runs the hyperparameter parity check as subtests over device init mode,
        # optimizer creation order, foreach, zero_grad mode, backward prefetch and writeback
        # check skipping, without CPU offload, for the given sharding strategy name.
        """
        Tests FSDP parity with DDP when using multiple parameter groups with
        different hyperparameter settings.
        """
        sharding_strategy = self._get_sharding_strategy_from_str(sharding_strategy_str)
        self.run_subtests(
            {
                "device_init_mode": [
                    DEVICEInitMode.DEVICE_BEFORE,
                    DEVICEInitMode.DEVICE_AFTER,
                ],
                "init_optim_before_wrap": [False, True],
                "optim_class": [torch.optim.AdamW],
                "multi_tensor": [False, True],
                "set_to_none": [False, True],
                "backward_prefetch": [
                    None,
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                ],
                "skip_writeback_check": [False, True],
            },
            self._test_diff_hyperparams,
            cpu_offload=CPUOffload(offload_params=False),
            sharding_strategy=sharding_strategy,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy_str",
        ["no_shard", "shard_grad_op", "full_shard"],
    )
    def test_diff_hyperparams_cpu_offload(self, sharding_strategy_str: str):
        # [FUNCTION DOC] Runs the hyperparameter parity check with parameter CPU offload for both
        # settings of the writeback check skip, in a plain loop rather than subtests.
        """
        Tests FSDP parity with DDP when using multiple parameter groups with
        different hyperparameter settings with CPU offloading enabled. This is
        separate from :meth:`test_diff_hyperparams` because CPU offloading has
        some issues with subtesting for some specific subtesting configs (e.g.,
        with ``offload_params=False`` followed by ``True`` but not vice versa).
        """
        sharding_strategy = self._get_sharding_strategy_from_str(sharding_strategy_str)
        for skip_writeback_check in (False, True):
            self._test_diff_hyperparams(
                device_init_mode=DEVICEInitMode.DEVICE_BEFORE,
                init_optim_before_wrap=False,
                optim_class=torch.optim.Adam,
                multi_tensor=False,
                set_to_none=False,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=CPUOffload(offload_params=True),
                sharding_strategy=sharding_strategy,
                skip_writeback_check=skip_writeback_check,
            )

    def _test_diff_hyperparams(
        self,
        device_init_mode: DEVICEInitMode,
        init_optim_before_wrap: bool,
        optim_class: type[torch.optim.Optimizer],
        multi_tensor: bool,
        set_to_none: bool,
        backward_prefetch: BackwardPrefetch | None,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy,
        skip_writeback_check: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds DDP and FSDP versions of the transformer with the
        # three-group optimizer and checks training parity. Returns without testing when the model
        # is moved to the device after wrapping with parameter CPU offload. When
        # skip_writeback_check is set, sets the process environment flag that disables the
        # FSDP writeback check.
        # KNOWN ISSUE: the environment flag is set and never cleared, so it persists for the rest
        # of the process.
        """
        Args:
            init_optim_before_wrap (bool): If ``True``, initializes the
                FSDP optimizer before wrapping the model with FSDP; otherwise,
                initializes the FSDP optimizer after wrapping the model with
                FSDP. We permit both forms of initialization to give users
                flexibility.
        """
        if (
            device_init_mode == DEVICEInitMode.DEVICE_AFTER
            and cpu_offload.offload_params
        ):
            return  # not supported
        if skip_writeback_check:
            os.environ[_FSDP_SKIP_WRITEBACK_CHECK] = "1"
        ddp_model = self._get_ddp_transformer(find_unused_params=False)
        ddp_optim = self._get_optim(ddp_model, optim_class, multi_tensor)
        fsdp_model, fsdp_optim = self._get_fsdp_transformer_and_optim(
            device_init_mode=device_init_mode,
            init_optim_before_wrap=init_optim_before_wrap,
            optim_class=optim_class,
            multi_tensor=multi_tensor,
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
        )
        self._check_train_parity(
            ddp_model, ddp_optim, fsdp_model, fsdp_optim, set_to_none
        )

    @skip_if_lt_x_gpu(2)
    def test_diff_trainability(self):
        # [FUNCTION DOC] Runs the trainability parity check as subtests over foreach and three
        # sharding strategies.
        """
        Tests FSDP parity with DDP when using multiple parameter groups and
        freezing the parameters in one parameter group.
        """
        self.run_subtests(
            {
                "multi_tensor": [False, True],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
            },
            self._test_diff_trainability,
        )

    def _test_diff_trainability(
        self,
        multi_tensor: bool,
        sharding_strategy: ShardingStrategy,
    ):
        # [FUNCTION DOC] HELPER: Builds DDP (with unused-parameter detection) and FSDP models, sets
        # requires_grad to False on every bias in both, and checks training parity with
        # gradients set to zero rather than None.
        optim_class = torch.optim.Adam
        ddp_model = self._get_ddp_transformer(find_unused_params=True)
        ddp_optim = self._get_optim(ddp_model, optim_class, multi_tensor)
        fsdp_model, fsdp_optim = self._get_fsdp_transformer_and_optim(
            device_init_mode=DEVICEInitMode.DEVICE_BEFORE,
            init_optim_before_wrap=False,
            optim_class=optim_class,
            multi_tensor=multi_tensor,
            sharding_strategy=sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=None,
        )
        # Freeze all biases (which happen to be in the same parameter group)
        for param_name, param in ddp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        for param_name, param in fsdp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim, False)

    @skip_if_lt_x_gpu(2)
    def test_multiple_optimizers(self):
        # [FUNCTION DOC] Runs the multiple-optimizer check as subtests over full shard and
        # shard-grad-op strategies.
        """
        Tests using two optimizers where only one sets gradients to ``None``.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                ]
            },
            self._test_multiple_optimizers,
        )

    def _test_multiple_optimizers(self, sharding_strategy: ShardingStrategy):
        # [FUNCTION DOC] HELPER: Uses one optimizer (Adam, AdamW) per weight and bias group on both
        # DDP and FSDP models. After one step, zeroes only the weight optimizer's gradients to None
        # and runs a forward, then (for strategies that reshard after forward) checks gradients are
        # None for weights and present for biases with matching values, and that parameters match.
        # Finishes the iteration and runs one more to confirm step-dependent bias correction
        # agrees.
        # Raises AssertionError if the group count is not 3 or if no rank-local flat parameter
        # holds both a weight and a bias.
        ddp_model = self._get_ddp_transformer(find_unused_params=True)
        ddp_param_groups = self._get_param_groups(ddp_model)
        if not (len(ddp_param_groups) == 3):
            raise AssertionError(
                f"Expected 3 param groups, got {len(ddp_param_groups)}"
            )
        (
            fsdp_model,
            _,
        ) = self._get_fsdp_transformer_and_optim(  # ignore returned optimizer
            device_init_mode=DEVICEInitMode.DEVICE_BEFORE,
            init_optim_before_wrap=False,
            optim_class=torch.optim.Adam,  # ignored
            multi_tensor=False,  # ignored
            sharding_strategy=sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=None,
        )
        fsdp_param_groups = self._get_param_groups(fsdp_model)
        if not (len(fsdp_param_groups) == 3):
            raise AssertionError(
                f"Expected 3 param groups, got {len(fsdp_param_groups)}"
            )
        ddp_optims = []
        fsdp_optims = []
        # For the transformer model, every parameter is either a weight or a
        # bias, so we only use the first two parameter groups. Moreover, we use
        # Adam and AdamW in particular since they both use bias correction
        # dependent on the step, which is incremented even if a parameter has a
        # zero gradient but not if the gradient is `None`. This is to test that
        # we are differentiating between a zero and `None` gradient correctly.
        optim_ctors = [
            functools.partial(torch.optim.Adam, lr=5e-3),
            functools.partial(torch.optim.AdamW, lr=1e-2),
        ]

        for optim_ctor, ddp_param_group, fsdp_param_group in zip(
            optim_ctors,
            ddp_param_groups[:2],
            fsdp_param_groups[:2],
        ):
            ddp_optims.append(optim_ctor(ddp_param_group["params"]))
            fsdp_optims.append(optim_ctor(fsdp_param_group["params"]))
        device = torch.device(device_type)

        # Check that there exists a `FlatParameter` that has both a weight and
        # a bias in this rank's shard
        has_both = False
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            handle = fsdp_module._handle
            if not handle:
                continue
            flat_param = handle.flat_param
            if flat_param._params is None:
                raise AssertionError("Expected flat_param._params to not be None")
            has_weight = False
            has_bias = False
            for param, fqn in zip(flat_param._params, flat_param._fqns):
                if "weight" in fqn and param.numel() > 0:
                    has_weight = True
                elif "bias" in fqn and param.numel() > 0:
                    has_bias = True
            has_both |= has_weight and has_bias
        if not has_both:
            raise AssertionError(
                f"Rank {self.rank} does not have a `FlatParameter` with both a "
                "weight and a bias in its shard, meaning that this test is vacuous"
            )

        # Run one iteration to generate gradients
        def run_iter():
            # [FUNCTION DOC] HELPER: One training iteration of the DDP and FSDP models with the
            # per-group optimizers: forward, loss, backward and step; asserts equal losses and
            # matching parameters.
            iter_losses = []
            for model, optims in ((ddp_model, ddp_optims), (fsdp_model, fsdp_optims)):
                module = model.module
                inp = module.get_input(device)
                output = model(*inp)
                loss = module.get_loss(inp, output).to(device)
                iter_losses.append(loss)
                module.run_backward(loss)
                for optim in optims:
                    optim.step()
            torch.testing.assert_close(iter_losses[0], iter_losses[1])
            iter_losses.clear()
            self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)

        run_iter()

        # Only set the weights' gradients to None
        ddp_optims[0].zero_grad(set_to_none=True)
        fsdp_optims[0].zero_grad(set_to_none=True)
        inp = ddp_model.module.get_input(device)
        ddp_output = ddp_model(*inp)
        fsdp_output = fsdp_model(*inp)

        # Check that FSDP correctly exposes gradients even after forward
        # (namely, `None` for weights and non-`None` for biases)
        if sharding_strategy in NO_RESHARD_AFTER_FORWARD_STRATEGIES:
            # Skip the check since we do not expose the gradients after forward
            # for these strategies
            return
        for (ddp_n, ddp_p), (fsdp_n, fsdp_p) in zip(
            ddp_model.module.named_parameters(),
            fsdp_model.named_parameters(),
        ):
            self.assertEqual(ddp_n, clean_tensor_name(fsdp_n))
            if fsdp_p.numel() == 0:
                # Not in this rank's shard
                self.assertTrue(fsdp_p.grad is None)
                continue
            if ddp_p.grad is None:
                self.assertTrue(fsdp_p.grad is None)
            else:
                self.assertEqual(ddp_p.flatten(), fsdp_p.flatten())
                self.assertEqual(ddp_p.grad.flatten(), fsdp_p.grad.flatten())
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)

        # Finish the iteration (backward pass and optimizer step)
        ddp_loss = ddp_model.module.get_loss(inp, ddp_output).to(device)
        fsdp_loss = fsdp_model.module.get_loss(inp, fsdp_output).to(device)
        ddp_model.module.run_backward(ddp_loss)
        fsdp_model.module.run_backward(fsdp_loss)
        for optim in itertools.chain(ddp_optims, fsdp_optims):
            optim.step()
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)

        # Run one more iteration to confirm bias corrections are correct
        run_iter()
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)


class TestFSDPUseOrigParamsUnshardReshard(FSDPTest):
    # [CLASS DOC] Two-rank tests that FSDP with use_orig_params=True matches use_orig_params=False
    # (same model, deterministic init) across unshard and reshard sequences such as several
    # forwards before a backward, or a summon of full parameters between forwards.
    """Tests the unshard/reshard flow."""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of ranks to spawn: always 2.
        return 2

    def _get_fsdp_models_and_optims(
        self,
        sharding_strategy: ShardingStrategy,
        cpu_offload: CPUOffload,
    ) -> tuple[FSDP, torch.optim.Optimizer, FSDP, torch.optim.Optimizer]:
        # [FUNCTION DOC] HELPER: Builds two identical recursively wrapped transformers with the
        # given sharding strategy and CPU offload, one with use_orig_params False and one True,
        # each with its own Adam optimizer (learning rate 0.01, foreach off). Returns the model
        # and optimizer for False, then for True.
        """
        Returns a pair of (FSDP model, optimizer) for ``use_orig_params=False``
        and ``True``, respectively.
        """
        LR = 1e-2
        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
            "cpu_offload": cpu_offload,
            "use_orig_params": False,
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs=fsdp_kwargs,
            deterministic=True,
        )
        optim = torch.optim.Adam(fsdp_model.parameters(), foreach=False, lr=LR)
        fsdp_kwargs["use_orig_params"] = True
        fsdp_model_orig_params = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs=fsdp_kwargs,
            deterministic=True,
        )
        optim_orig_params = torch.optim.Adam(
            fsdp_model_orig_params.parameters(), foreach=False, lr=LR
        )
        return fsdp_model, optim, fsdp_model_orig_params, optim_orig_params

    def _check_fsdp_parameter_parity(self, fsdp1: FSDP, fsdp2: FSDP) -> None:
        # [FUNCTION DOC] HELPER: Summons the full parameters of both FSDP models and asserts names
        # are equal and values are close, pairwise.
        """Checks that two FSDP instances have the same model parameters."""
        with FSDP.summon_full_params(fsdp1), FSDP.summon_full_params(fsdp2):
            for (n1, p1), (n2, p2) in zip(
                fsdp1.named_parameters(),
                fsdp2.named_parameters(),
            ):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(p1, p2)

    def _get_fsdp_parity_subtest_config(self):
        # [FUNCTION DOC] HELPER: Returns the subtest mapping that sweeps the three sharding
        # strategies (no shard, shard-grad-op, full shard).
        return {
            "sharding_strategy": [
                ShardingStrategy.NO_SHARD,
                ShardingStrategy.SHARD_GRAD_OP,
                ShardingStrategy.FULL_SHARD,
            ],
        }

    @skip_if_lt_x_gpu(2)
    @parametrize("offload_params", [False, True])
    def test_multiple_forward(self, offload_params: bool):
        # [FUNCTION DOC] Runs the multiple-forward parity check as subtests over the sharding
        # strategies, with or without parameter CPU offload.
        """
        Tests that ``use_orig_params=True`` has parity with ``False`` when
        running multiple forward passes before a backward pass.
        """
        cpu_offload = CPUOffload(offload_params=offload_params)
        self.run_subtests(
            self._get_fsdp_parity_subtest_config(),
            self._test_multiple_forward,
            cpu_offload=cpu_offload,
        )

    @skip_if_lt_x_gpu(2)
    def _test_multiple_forward(
        self,
        sharding_strategy: ShardingStrategy,
        cpu_offload: CPUOffload,
    ):
        # [FUNCTION DOC] HELPER: For 3 iterations, runs two forwards (different inputs) per model
        # before one backward and optimizer step on both the False and True models; asserts both
        # forward losses and the total loss are equal, then compares final parameters.
        # KNOWN ISSUE: this helper carries a GPU-count skip decorator even though it is not a test.
        (
            fsdp_model,
            optim,
            fsdp_model_orig_params,
            optim_orig_params,
        ) = self._get_fsdp_models_and_optims(sharding_strategy, cpu_offload)
        device = torch.device(device_type)
        for _ in range(3):
            inp1 = fsdp_model.get_input(device)
            _inp2 = fsdp_model.get_input(device)
            inp2 = tuple(
                t + torch.ones_like(t) for t in _inp2
            )  # make different from `inp1`
            # For these loss lists: elem 0 is baseline; elem 1 is test
            losses1 = []
            losses2 = []
            losses = []
            for _model, _optim in (
                (fsdp_model, optim),
                (
                    fsdp_model_orig_params,
                    optim_orig_params,
                ),
            ):
                _optim.zero_grad()
                loss1 = _model(*inp1)
                losses1.append(loss1)
                loss2 = _model(*inp2)
                losses2.append(loss2)
                loss = (loss1 + loss2).sum()
                losses.append(loss)
                _model.run_backward(loss)
                _optim.step()
            self.assertEqual(losses1[0], losses1[1])
            self.assertEqual(losses2[0], losses2[1])
            self.assertEqual(losses[0], losses[1])
        self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("offload_params", [False, True])
    def test_summon_between_two_forwards(self, offload_params: bool):
        # [FUNCTION DOC] Runs the summon-between-forwards parity check as subtests over the
        # sharding strategies, with or without parameter CPU offload.
        """
        Tests that ``use_orig_params=True`` has parity with ``False`` when
        running a forward pass, :meth:`summon_full_params()`, and another
        forward pass before a backward pass.
        """
        cpu_offload = CPUOffload(offload_params=offload_params)
        self.run_subtests(
            self._get_fsdp_parity_subtest_config(),
            self._test_summon_between_two_forwards,
            cpu_offload=cpu_offload,
        )

    def _test_summon_between_two_forwards(
        self,
        sharding_strategy: ShardingStrategy,
        cpu_offload: CPUOffload,
    ):
        # [FUNCTION DOC] HELPER: For 3 iterations, runs a forward, a parameter parity check (which
        # summons full parameters), a second forward, then backward and step on both models;
        # asserts the forward losses match each time and final parameters match.
        (
            fsdp_model,
            optim,
            fsdp_model_orig_params,
            optim_orig_params,
        ) = self._get_fsdp_models_and_optims(sharding_strategy, cpu_offload)
        device = torch.device(device_type)
        for _ in range(3):
            optim.zero_grad()
            optim_orig_params.zero_grad()

            inp1 = fsdp_model.get_input(device)
            loss1 = fsdp_model(*inp1)
            loss_orig_params1 = fsdp_model_orig_params(*inp1)
            self.assertEqual(loss1, loss_orig_params1)

            # Calls into `summon_full_params()`
            self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)

            inp2 = fsdp_model.get_input(device)
            loss2 = fsdp_model(*inp2)
            loss_orig_params2 = fsdp_model_orig_params(*inp2)
            self.assertEqual(loss2, loss_orig_params2)

            loss = (loss1 + loss2).sum()
            loss_orig_params = (loss_orig_params1 + loss_orig_params2).sum()
            fsdp_model.run_backward(loss)
            fsdp_model_orig_params.run_backward(loss_orig_params)
            optim.step()
            optim_orig_params.step()
        self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)


class TestFSDPUseOrigParamsParamAccess(FSDPTest):
    # [CLASS DOC] Two-rank tests that with use_orig_params=True the original (sharded, flattened)
    # parameters can be read after forward and before backward, matching the expected slice of the
    # DDP parameters under each sharding strategy.
    """Tests original parameter access."""

    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the number of ranks to spawn: always 2 (the tests hard-code the
        # expected shard layout).
        # Force a world size of 2 since the tests hard code to the FSDP
        # sharding strategy to check sharded parameter parity
        return 2

    @skip_if_lt_x_gpu(2)
    def test_access_params_after_forward(self):
        # [FUNCTION DOC] Runs the parameter access check as subtests over the three sharding
        # strategies.
        """
        Tests that accessing the original parameters after the forward but
        before the backward. Notably, this is not supported when
        ``use_orig_params=False``. However, for ``True``, FSDP exposes the
        (flattened) sharded original parameters, making it possible.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.NO_SHARD,
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                ],
            },
            self._test_access_params_after_forward,
        )

    def _test_access_params_after_forward(
        self,
        sharding_strategy: ShardingStrategy,
    ):
        # [FUNCTION DOC] HELPER: Compares DDP parameters with the FSDP original parameters after a
        # forward, after backward and step, and after another forward, using a toy two-layer model
        # whose shard layout over 2 ranks is hard-coded. Depends on the current FSDP sharding
        # algorithm.
        # NOTE: This test needs to be changed if the FSDP sharding algorithm
        # changes. It is still valuable until such a change to sanity check the
        # `use_orig_params=True` implementation.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Toy model of a bias-free 5 by 5 Linear, ReLU and a 5 by 7 Linear
            # with bias; supplies the input and loss helpers the tests call.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Seeds the RNG with 42 and creates lin1 and lin2.
                super().__init__()
                torch.manual_seed(42)
                # 5 * 5 = 25 numel -> pad to 26 -> 13 on each rank
                self.lin1 = nn.Linear(5, 5, bias=False)
                # 5 * 7 + (1) + 7 = 43 numel -> pad to 44 -> 22 on each rank,
                # where the (1) is from intra-`FlatParameter` alignment padding
                # 22 of weight on rank 0; 13 of weight, 1 alignment padding,
                # and 7 of bias on rank 1
                self.lin2 = nn.Linear(5, 7)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies lin1, ReLU and lin2 and returns the result.
                z = self.lin1(x)
                z = nn.functional.relu(z)
                z = self.lin2(z)
                return z

            def get_input(self, device: torch.device) -> tuple[torch.Tensor, ...]:
                # [FUNCTION DOC] HELPER: Returns a one-element tuple with a random 2 by 5 tensor
                # on the device.
                return (torch.randn((2, 5)).to(device),)

            def get_loss(self, inp, out):
                # [FUNCTION DOC] HELPER: Returns the sum of the model output as the loss.
                return out.sum()

        def check_parameter_parity(
            ddp_model: DDP, fsdp_model: FSDP, between_fwd_and_bwd: bool
        ):
            # [FUNCTION DOC] HELPER: Compares each DDP parameter with the FSDP one by name. For
            # sharded strategies it slices the DDP value to this rank's expected piece (first or
            # second part of lin1.weight, lin2.weight, or empty/full bias) and asserts closeness;
            # unflattened (no shard) and no-reshard-between-forward-and-backward cases are skipped.
            # Raises AssertionError if the rank is not 0 or 1.
            if self.rank not in (0, 1):
                raise AssertionError(
                    f"Expects world size of 2 but got {self.world_size}"
                )
            for (n1, p1), (n2, p2) in zip(
                ddp_model.module.named_parameters(),
                fsdp_model.named_parameters(),
            ):
                self.assertEqual(n1, clean_tensor_name(n2))
                if sharding_strategy == ShardingStrategy.NO_SHARD:
                    # For `NO_SHARD`, do nothing since the original parameters
                    # are unflattened
                    pass
                elif (
                    between_fwd_and_bwd
                    and sharding_strategy in NO_RESHARD_AFTER_FORWARD_STRATEGIES
                ):
                    # For no reshard after forward strategies, do nothing since
                    # FSDP did not use sharded views after forward
                    pass
                # Otherwise, case on the parameter (see the model definition)
                elif n1 == "lin1.weight":
                    if self.rank == 0:
                        p1 = p1.flatten()[:13]
                    elif self.rank == 1:
                        p1 = p1.flatten()[13:]
                elif n1 == "lin2.weight":
                    if self.rank == 0:
                        p1 = p1.flatten()[:22]
                    elif self.rank == 1:
                        p1 = p1.flatten()[22:]
                elif n1 == "lin2.bias":
                    if self.rank == 0:
                        p1 = torch.empty(0, device=p1.device)
                    elif self.rank == 1:
                        p1 = p1.flatten()
                torch.testing.assert_close(p1, p2)

        ddp_model = DDP(Model().to(device=device_type), device_ids=[self.rank])
        fsdp_model = FSDP(
            Model().to(device=device_type),
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=always_wrap_policy,
            use_orig_params=True,
        )
        LR = 1e-2
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)
        device = torch.device(device_type)

        inp = fsdp_model.get_input(device)
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        check_parameter_parity(ddp_model, fsdp_model, True)

        ddp_loss = ddp_model.module.get_loss(inp, ddp_out)
        fsdp_loss = fsdp_model.get_loss(inp, fsdp_out)
        ddp_loss.backward()
        fsdp_loss.backward()
        ddp_optim.step()
        fsdp_optim.step()
        check_parameter_parity(ddp_model, fsdp_model, False)

        inp = fsdp_model.get_input(device)
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        check_parameter_parity(ddp_model, fsdp_model, True)


class TestFSDPUseOrigParamsWriteback(FSDPTest):
    # [CLASS DOC] Two-rank tests that user modifications of original parameters and gradients
    # (by data or by replacing the variable) are written back into FSDP's flat storage, that shape
    # mismatches raise, and that changing parameters between forward and backward is rejected for
    # strategies that skip resharding. Contains the Model fixture class.
    """Tests parameter and gradient writeback."""

    class Model(nn.Module):
        # [CLASS DOC] HELPER: Toy model of two Linear layers with bias (5 to 5, 5 to 7) with a
        # ReLU in between, built directly on the given device.
        def __init__(self, device: torch.device):
            # [FUNCTION DOC] HELPER: Seeds the RNG with 42 and creates lin1 and lin2 on device.
            super().__init__()
            torch.manual_seed(42)
            self.lin1 = nn.Linear(5, 5, bias=True, device=device)
            self.lin2 = nn.Linear(5, 7, bias=True, device=device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies lin1, ReLU and lin2 and returns the result.
            z = self.lin1(x)
            z = nn.functional.relu(z)
            z = self.lin2(z)
            return z

        def get_input(self, device: torch.device) -> tuple[torch.Tensor, ...]:
            # [FUNCTION DOC] HELPER: Returns a one-element tuple with a random 2 by 5 tensor on
            # the device.
            return (torch.randn((2, 5)).to(device),)

        def get_loss(self, inp, out):
            # [FUNCTION DOC] HELPER: Returns the sum of the model output as the loss.
            return out.sum()

    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the number of ranks to spawn: always 2.
        # Force a world size of 2 since the tests hard code to the FSDP
        # sharding strategy
        return 2

    def _check_param_parity(self, ddp_model: DDP, fsdp_model: FSDP):
        # [FUNCTION DOC] HELPER: Summons the full FSDP parameters (which triggers a writeback of
        # user changes) and asserts names are equal and values are close to the DDP parameters.
        with FSDP.summon_full_params(fsdp_model):
            for (n1, p1), (n2, p2) in zip(
                ddp_model.module.named_parameters(),
                fsdp_model.named_parameters(),
            ):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_param_writeback(self):
        # [FUNCTION DOC] Runs the parameter writeback check as subtests over changing the first or
        # second weight, by setting its data or replacing the parameter.
        """Tests that changes to the original parameters are written back."""
        self.run_subtests(
            {
                "change_first_weight": [True, False],  # first vs. second `weight`
                "change_data": [True, False],  # change `.data` vs. variable itself
            },
            self._test_param_writeback,
        )

    def _test_param_writeback(self, change_first_weight: bool, change_data: bool):
        # [FUNCTION DOC] HELPER: Replaces one weight (data or variable) with a tensor of 2s in both
        # DDP and FSDP models, then checks parameter parity, which triggers FSDP writeback.
        def transform_param(param: nn.Parameter) -> nn.Parameter:
            # [FUNCTION DOC] HELPER: Returns a new parameter filled with 2s shaped like param.
            return nn.Parameter(torch.ones_like(param) * 2)

        # Check that the writeback propagates
        ddp_model = DDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type)),
            device_ids=[self.rank],
        )
        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type)),
            use_orig_params=True,
        )
        ddp = ddp_model.module  # for brevity
        fsdp = fsdp_model.module
        if change_first_weight:
            if change_data:
                ddp.lin1.weight.data = transform_param(ddp.lin1.weight)
                fsdp.lin1.weight.data = transform_param(fsdp.lin1.weight)
            else:
                ddp.lin1.weight = transform_param(ddp.lin1.weight)
                fsdp.lin1.weight = transform_param(fsdp.lin1.weight)
        else:
            if change_data:
                ddp.lin2.weight.data = transform_param(ddp.lin2.weight)
                fsdp.lin2.weight.data = transform_param(fsdp.lin2.weight)
            else:
                ddp.lin2.weight = transform_param(ddp.lin2.weight)
                fsdp.lin2.weight = transform_param(fsdp.lin2.weight)
        self._check_param_parity(ddp_model, fsdp_model)  # triggers a writeback

    @skip_if_lt_x_gpu(2)
    def test_grad_writeback(self):
        # [FUNCTION DOC] Runs the gradient writeback check as subtests over which weight's gradient
        # is changed, data versus variable replacement, and set to None versus a tensor.
        """
        Tests that changes to the original parameters' gradients are written
        back.
        """
        self.run_subtests(
            {
                "change_first_weight_grad": [False, True],
                "change_data": [False, True],  # change `.data` vs. variable itself
                "set_to_none": [False, True],
            },
            self._test_grad_writeback,
        )

    def _test_grad_writeback(
        self,
        change_first_weight_grad: bool,
        change_data: bool,
        set_to_none: bool,
    ):
        # [FUNCTION DOC] HELPER: After one backward, replaces a weight gradient (to 2s or None) in
        # DDP and FSDP models, steps Adam to surface whether the change was written back, checks
        # parameter parity, then runs another iteration without zeroing gradients and checks again.
        # Returns immediately for the ill-defined combination of data change with None.
        if change_data and set_to_none:
            return  # not well-defined

        def transform_grad(param: nn.Parameter) -> nn.Parameter:
            # [FUNCTION DOC] HELPER: Returns None if set_to_none else a gradient of 2s shaped like
            # param.
            return None if set_to_none else torch.ones_like(param) * 2

        ddp_model = DDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type)),
            device_ids=[self.rank],
        )
        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type)),
            use_orig_params=True,
        )
        LR = 1e-2
        # TODO: If we add `summon_full_params(with_grads=True)`, then replace
        # the following. For now, we use the optimizer step as a surrogate for
        # checking that gradients were written back.
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)

        # Generate an initial gradient
        inp = fsdp_model.get_input(torch.device(device_type))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()

        # Change the gradient through the original parameters
        ddp = ddp_model.module  # for brevity
        fsdp = fsdp_model.module
        if change_first_weight_grad:
            if change_data:
                ddp.lin1.weight.grad.data = transform_grad(ddp.lin1.weight)
                if fsdp.lin1.weight.grad is not None:
                    fsdp.lin1.weight.grad.data = transform_grad(fsdp.lin1.weight)
            else:
                ddp.lin1.weight.grad = transform_grad(ddp.lin1.weight)
                fsdp.lin1.weight.grad = transform_grad(fsdp.lin1.weight)
        else:
            if change_data:
                ddp.lin2.weight.grad.data = transform_grad(ddp.lin2.weight)
                if fsdp.lin2.weight.grad is not None:
                    fsdp.lin2.weight.grad.data = transform_grad(fsdp.lin2.weight)
            else:
                ddp.lin2.weight.grad = transform_grad(ddp.lin2.weight)
                fsdp.lin2.weight.grad = transform_grad(fsdp.lin2.weight)
        ddp_optim.step()
        fsdp_optim.step()
        self._check_param_parity(ddp_model, fsdp_model)  # triggers a writeback

        # Intentionally do not zero the gradient to check writeback
        inp = fsdp_model.get_input(torch.device(device_type))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()
        ddp_optim.step()
        fsdp_optim.step()
        self._check_param_parity(ddp_model, fsdp_model)  # triggers a writeback

    @skip_if_lt_x_gpu(2)
    def test_writeback_shape_mismatch(self):
        # [FUNCTION DOC] On rank 0 replaces lin1.weight and its gradient, and on rank 1 lin2.weight
        # and its gradient, with tensors one larger in every dimension, and asserts that summoning
        # full parameters raises a RuntimeError containing the text Cannot writeback.
        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type)),
            use_orig_params=True,
        )
        # Check that writing back with mismatched shape errors
        fsdp = fsdp_model.module  # for brevity
        if self.rank not in (0, 1):
            raise AssertionError(f"Expects world size of 2 but got {self.world_size}")
        with self.assertRaisesRegex(RuntimeError, "Cannot writeback"):
            # Change the gradient to a new one with 1 added to each dimension
            # to force a shape mismatch when writing back
            if self.rank == 0:
                # Change `lin1.weight.grad` since it exists on rank 0
                lin1_weight_shape = list(fsdp.lin1.weight.shape)
                for dim_index in range(len(lin1_weight_shape)):
                    lin1_weight_shape[dim_index] += 1
                fsdp.lin1.weight = nn.Parameter(
                    torch.randn(
                        torch.Size(lin1_weight_shape), device=fsdp.lin1.weight.device
                    )
                )
                fsdp.lin1.weight.grad = torch.randn(
                    torch.Size(lin1_weight_shape), device=fsdp.lin1.weight.device
                )
            elif self.rank == 1:
                # Change `lin2.weight.grad` since it exists (partially) on rank 1
                lin2_weight_shape = list(fsdp.lin2.weight.shape)
                for dim_index in range(len(lin2_weight_shape)):
                    lin2_weight_shape[dim_index] += 1
                fsdp.lin2.weight = nn.Parameter(
                    torch.randn(
                        torch.Size(lin2_weight_shape), device=fsdp.lin2.weight.device
                    )
                )
                fsdp.lin2.weight.grad = torch.randn(
                    torch.Size(lin2_weight_shape), device=fsdp.lin2.weight.device
                )
            with FSDP.summon_full_params(fsdp_model):  # triggers a writeback
                ...

    @skip_if_lt_x_gpu(2)
    def test_writeback_between_fwd_and_bwd_for_no_reshard_raises(self):
        # [FUNCTION DOC] With the shard-grad-op strategy (no reshard after forward), changes a
        # parameter between forward and backward, first by replacing its data with a clone, then by
        # replacing the parameter object, and asserts backward raises an AssertionError saying FSDP
        # does not support changing the parameters between forward and backward.
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
            "use_orig_params": True,
        }
        fsdp_wrapper = functools.partial(FSDP, **fsdp_kwargs)

        # Test changing the parameter storage to no longer be a view into the
        # flat parameter
        fsdp_model = fsdp_wrapper(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type))
        )
        inp = fsdp_model.get_input(torch.device(device_type))
        loss = fsdp_model(*inp).sum()
        fsdp_model.lin1.weight.data = fsdp_model.lin1.weight.clone()
        assert_msg = (
            "FSDP does not support changing the parameters between forward and backward"
        )
        with self.assertRaisesRegex(AssertionError, assert_msg):
            loss.backward()

        # Test changing the parameter variable itself
        fsdp_model = fsdp_wrapper(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type))
        )
        inp = fsdp_model.get_input(torch.device(device_type))
        loss = fsdp_model(*inp).sum()
        fsdp_model.lin1._fsdp_wrapped_module.weight = nn.Parameter(
            fsdp_model.lin1.weight.clone()
        )
        with self.assertRaisesRegex(AssertionError, assert_msg):
            loss.backward()

    @skip_if_lt_x_gpu(2)
    def test_no_reshard_and_mixed_precision(self):
        # [FUNCTION DOC] Runs the no-reshard mixed precision check as subtests with and without the
        # full-precision-in-eval environment flag.
        """
        Tests that writeback does not falsely get triggered for a few
        configurations (exercising the sharded view skipping logic):
        - Train forward -> full-precision unshard -> train forward
        - Train forward -> eval forward
        - Train forward/backward -> eval forward -> model checkpoint
        """
        self.run_subtests(
            {"use_full_prec_in_eval": [False, True]},
            self._test_no_reshard_and_mixed_precision,
        )

    def _test_no_reshard_and_mixed_precision(self, use_full_prec_in_eval: bool):
        # [FUNCTION DOC] HELPER: With fp16 parameter mixed precision and shard-grad-op, runs train
        # forward, a full-parameter summon and another forward; train then eval forward; and
        # train forward/backward, eval forward and a sharded state dict save and load; none of
        # these should falsely trigger writeback errors. Side effect: sets the environment flag for
        # full precision in eval when requested and does not clear it.
        if use_full_prec_in_eval:
            os.environ[_FSDP_USE_FULL_PREC_IN_EVAL] = "1"
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
            "mixed_precision": MixedPrecision(param_dtype=torch.float16),
            "use_orig_params": True,
        }

        # Train forward -> full-precision unshard -> train forward
        fsdp_model = FSDP(
            TestFSDPUseOrigParamsWriteback.Model(torch.device(device_type)),
            **fsdp_kwargs,
        )
        inp = fsdp_model.get_input(torch.device(device_type))
        fsdp_model(*inp)
        with FSDP.summon_full_params(fsdp_model):
            ...
        fsdp_model(*inp).sum()

        # Train forward -> eval forward
        fsdp_model.train()
        fsdp_model(*inp)
        fsdp_model.eval()
        fsdp_model(*inp)

        # Train forward/backward -> eval forward -> model checkpoint
        fsdp_model.train()
        fsdp_model(*inp).sum().backward()
        fsdp_model.eval()
        fsdp_model(*inp)
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
            sd = fsdp_model.state_dict()
            fsdp_model.load_state_dict(sd)
        fsdp_model(*inp).sum().backward()


class TestFSDPUseOrigParamsFQNs(FSDPTest):
    # [CLASS DOC] Tests that parameter names and shapes seen inside a forward pass under FSDP
    # with use_orig_params=True are those of the original, unsharded parameters.
    @skip_if_lt_x_gpu(2)
    def test_named_parameters_in_forward(self):
        # [FUNCTION DOC] Wraps a one-layer model whose forward asserts that named_parameters
        # yields the cleaned names weight and bias of the layer with the original unsharded
        # shapes, then runs one forward. Needs 2 GPUs.
        """
        Tests that calling ``named_parameters()`` during forward returns FQNs
        and ``Tensor`` s corresponding to the original parameters.
        """
        param_shapes = [None, None]
        assert_equal_fn = self.assertEqual

        class Model(nn.Module):
            # [CLASS DOC] HELPER: One 5 by 5 Linear layer whose forward checks parameter names
            # and shapes against the enclosing test's expectations.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Linear layer lin.
                super().__init__()
                self.lin = nn.Linear(5, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Asserts the parameter names (prefixes stripped) are
                # lin.weight and lin.bias and their shapes match the saved unsharded shapes, then
                # returns lin applied to x. Raises AssertionError if shapes were not saved.
                nonlocal param_shapes
                # Allow for FSDP prefixes
                param_names = [
                    clean_tensor_name(tup[0]) for tup in self.named_parameters()
                ]
                params = [tup[1] for tup in self.named_parameters()]
                if not (param_shapes[0] is not None and param_shapes[1] is not None):
                    raise AssertionError("`param_sizes` should be set")
                assert_equal_fn(
                    param_names,
                    [
                        "lin.weight",
                        "lin.bias",
                    ],
                )
                assert_equal_fn(params[0].shape, param_shapes[0])
                assert_equal_fn(params[1].shape, param_shapes[1])
                return self.lin(x)

        model = Model().to(device=device_type)
        # Save the *unsharded* original parameter shapes and check the shapes
        # match in the forward pass
        param_shapes[0] = model.lin.weight.shape
        param_shapes[1] = model.lin.bias.shape
        fsdp_model = FSDP(model, use_orig_params=True)
        inp = torch.randn((2, 5), device=torch.device(device_type))
        fsdp_model(inp)


class TestFSDPUseOrigParamsNoSync(FSDPTest):
    # [CLASS DOC] Two-rank tests of FSDP no_sync with use_orig_params=True: gradient values
    # versus the flat-parameter mode, and gradient dtypes under mixed precision.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of ranks to spawn: always 2.
        return 2

    @skip_if_lt_x_gpu(2)
    def test_no_sync_correctness(self):
        # [FUNCTION DOC] Runs the no_sync correctness check as subtests over three sharding
        # strategies.
        """
        Tests a basic ``no_sync()`` setup by comparing ``use_orig_params=True``
        against ``use_orig_params=False``.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
            },
            self._test_no_sync_correctness,
        )

    def _test_no_sync_correctness(self, sharding_strategy: ShardingStrategy):
        # [FUNCTION DOC] HELPER: Wraps copies of a single-weight Linear layer in FSDP with
        # use_orig_params False and True and compares output and gradients after one forward and
        # backward; then repeats a pass inside no_sync and one outside, checking gradient parity
        # each time and that accumulated gradients equal twice the reference gradients.
        model = nn.Linear(7, 1, bias=False, device=device_type)
        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
        }
        model_use_flat_params = FSDP(
            copy.deepcopy(model), use_orig_params=False, **fsdp_kwargs
        )
        model_use_orig_params = FSDP(model, use_orig_params=True, **fsdp_kwargs)
        optim_use_flat_params = torch.optim.AdamW(
            model_use_flat_params.parameters(), foreach=True
        )
        optim_use_orig_params = torch.optim.AdamW(
            model_use_orig_params.parameters(), foreach=True
        )

        def _check_param_grad_parity(
            _baseline_model: nn.Module,
            _test_model: nn.Module,
        ):
            # [FUNCTION DOC] HELPER: Asserts each model has exactly one parameter, that the
            # baseline (flat) parameter values and gradient, limited to the unpadded element
            # count, equal the other model's original parameter and gradient. The baseline may
            # be longer due to padding.
            """
            This assumes that the model is ``nn.Linear(7, 1, bias=False)``
            (i.e. with a single 1D weight parameter) to be able to directly
            compare the baseline and test models. On rank 1, the baseline
            includes 1 element of padding.
            """
            self.assertEqual(len(list(_baseline_model.parameters())), 1)
            self.assertEqual(len(list(_test_model.parameters())), 1)
            for flat_param, orig_param in zip(
                _baseline_model.parameters(), _test_model.parameters()
            ):
                # Baseline is permitted to have padding
                self.assertGreaterEqual(flat_param.numel(), orig_param.numel())
                unpadded_param_numel = orig_param.numel()
                # For `NO_SHARD`, `use_orig_params=True` presents unflattened
                # parameters, while `False` presents flattened ones
                torch.testing.assert_close(
                    flat_param[:unpadded_param_numel], orig_param.flatten()
                )
                # Gradient numel is different if right after `no_sync()` since
                # the gradient is unsharded, while the parameter is sharded
                unpadded_grad_numel = orig_param.grad.numel()
                # For `use_orig_params=False`, the unsharded gradient is
                # flattened, while for `True`, it is unflattened
                torch.testing.assert_close(
                    flat_param.grad[:unpadded_grad_numel].reshape(
                        orig_param.grad.shape
                    ),
                    orig_param.grad,
                )

        inp = torch.randn((2, 7), device=device_type)
        grad = torch.randn((2, 1), device=device_type)

        # Compute some reference gradients using one forward/backward
        out_use_flat_params = model_use_flat_params(inp)
        out_use_orig_params = model_use_orig_params(inp)
        torch.testing.assert_close(out_use_flat_params, out_use_orig_params)
        out_use_flat_params.backward(grad)
        out_use_orig_params.backward(grad)
        _check_param_grad_parity(model_use_flat_params, model_use_orig_params)
        ref_grads_use_flat_params = [
            param.grad.detach().clone() for param in model_use_flat_params.parameters()
        ]
        ref_grads_use_orig_params = [
            param.grad.detach().clone()
            for param in model_use_orig_params.parameters()
            if param.grad is not None
        ]

        # Run a forward/backward in `no_sync()`
        optim_use_flat_params.zero_grad(set_to_none=True)
        optim_use_orig_params.zero_grad(set_to_none=True)
        for model in (model_use_flat_params, model_use_orig_params):
            with model.no_sync():
                out = model(inp)
                out.backward(grad)
        _check_param_grad_parity(model_use_flat_params, model_use_orig_params)

        # Run a forward/backward outside `no_sync()`
        for model in (model_use_flat_params, model_use_orig_params):
            out = model(inp)
            out.backward(grad)
        _check_param_grad_parity(model_use_flat_params, model_use_orig_params)

        # Check that, since we accumulated gradients across 2 iterations, that
        # the new gradients are 2x the reference gradients
        grads_use_flat_params = [
            param.grad.detach().clone() for param in model_use_flat_params.parameters()
        ]
        grads_use_orig_params = [
            param.grad.detach().clone()
            for param in model_use_orig_params.parameters()
            if param.grad is not None
        ]
        for grad, ref_grad in zip(grads_use_flat_params, ref_grads_use_flat_params):
            torch.testing.assert_close(grad, 2 * ref_grad)
        for grad, ref_grad in zip(grads_use_orig_params, ref_grads_use_orig_params):
            torch.testing.assert_close(grad, 2 * ref_grad)

    @skip_if_lt_x_gpu(2)
    def test_no_sync_mixed_precision(self):
        # [FUNCTION DOC] Runs the no_sync mixed precision check as subtests over three sharding
        # strategies.
        """
        Tests that dtypes are as expected when using ``no_sync()`` with
        ``use_orig_params=True`` and parameter mixed precision.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_no_sync_mixed_precision,
        )

    def _test_no_sync_mixed_precision(self, sharding_strategy: ShardingStrategy):
        # [FUNCTION DOC] HELPER: Wraps a Linear layer with fp16 parameter and fp32 reduce dtypes;
        # asserts gradients are fp16 after two backward passes inside no_sync and fp32 after one
        # outside it.
        model = nn.Linear(3, 3, device=device_type)
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
        )
        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
            "mixed_precision": mixed_precision,
            "use_orig_params": True,
        }
        fsdp_model = FSDP(model, **fsdp_kwargs)
        inp = torch.randn((2, 3), device=device_type)
        with fsdp_model.no_sync():
            # For each of these `no_sync()` backward passes, check that the
            # gradients are in the low precision parameter dtype (FP16)
            fsdp_model(inp).sum().backward()
            for param in fsdp_model.parameters():
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)
            fsdp_model(inp).sum().backward()
            for param in fsdp_model.parameters():
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)
        # For the backward pass outside `no_sync()`, check that the gradients
        # are cast to the full precision in preparation for the optimizer step
        fsdp_model(inp).sum().backward()
        for param in fsdp_model.parameters():
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float32)


class TestFSDPUseOrigParamsInit(FSDPTest):
    # [CLASS DOC] Tests of FSDP initialization with use_orig_params=True.
    @skip_if_lt_x_gpu(2)
    def test_non_uniform_requires_grad(self):
        # [FUNCTION DOC] Freezes the biases of a two-layer model, wraps it with FSDP so weights
        # and biases share one flat parameter, and asserts each original parameter keeps its own
        # requires_grad value. Needs 2 GPUs.
        model = nn.Sequential(
            nn.Linear(3, 3, device=device_type),
            nn.Linear(3, 3, device=device_type),
        )
        # Freeze biases only and flatten both weights and biases into the same
        # `FlatParameter` to exercise non-uniform `requires_grad`
        model[0].bias.requires_grad = False
        model[1].bias.requires_grad = False
        fsdp_model = FSDP(model, use_orig_params=True)
        self.assertTrue(fsdp_model[0].weight.requires_grad)
        self.assertFalse(fsdp_model[0].bias.requires_grad)
        self.assertTrue(fsdp_model[1].weight.requires_grad)
        self.assertFalse(fsdp_model[1].bias.requires_grad)


# Define this to be large enough to trigger stack corruption
NUM_SIZE0_TENSORS = 1000


class TestMultiTensorApply(TestCase):
    # [CLASS DOC] Single-process regression tests that foreach operations on a thousand
    # zero-sized tensors do not crash (stack corruption in multi-tensor apply).
    def test_multi_tensor_apply_size0_tensors_cpu(self):
        # [FUNCTION DOC] Runs a foreach multiply on 1000 empty CPU tensors; passes if no crash.
        size0_tensors = [torch.empty(0, device="cpu") for _ in range(NUM_SIZE0_TENSORS)]
        # Check that this does not segfault
        torch._foreach_mul_(size0_tensors, 0.1)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "no cuda and no xpu")
    def test_multi_tensor_apply_size0_tensors_cuda(self):
        # [FUNCTION DOC] Runs a foreach multiply on 1000 empty tensors on the accelerator
        # device; passes if no crash. Skipped without CUDA or XPU.
        size0_tensors = [
            torch.empty(0, device=device_type) for _ in range(NUM_SIZE0_TENSORS)
        ]
        # Check that this does not segfault
        torch._foreach_mul_(size0_tensors, 0.1)


instantiate_parametrized_tests(TestFSDPUseOrigParamsMultipleParamGroups)
instantiate_parametrized_tests(TestFSDPUseOrigParamsUnshardReshard)
instantiate_parametrized_tests(TestFSDPUseOrigParamsParamAccess)
instantiate_parametrized_tests(TestFSDPUseOrigParamsFQNs)
instantiate_parametrized_tests(TestFSDPUseOrigParamsNoSync)

if __name__ == "__main__":
    run_tests()
