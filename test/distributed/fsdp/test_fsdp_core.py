# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Core multi-GPU FSDP tests: numerical parity with DDP across models (nested,
# always-wrap, transformer, delayed-stream, mixture-of-experts) and a grid of CPU offload,
# sharding strategy, device-init mode, prefetch and use_orig_params settings; parameter
# in-place changes after init; backward-hook registration; eval vs no_grad forward equality;
# and a check that unsharded parameters are used as plain tensor views in forward/backward.
# Import-time side effects: exits with status 0 if torch.distributed is unavailable or under
# dev-asan; builds the module-level config grid (cpu_offload_config, sharding_strategy_config,
# configs), rebinds subtest_name with a name mapping, and instantiates device-generic test
# classes for cuda/hpu/xpu. Running the file calls run_tests().
import contextlib
import functools
import itertools
import sys
import unittest
from collections.abc import Callable
from typing import Any
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp._flat_param import FlatParamHandle
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.utils import _p_assert
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    AlwaysWrapNestedWrappedModule,
    DEVICEInitMode,
    DummyDDP,
    FSDPInitMode,
    FSDPTest,
    get_devtype,
    MixtureOfExperts,
    NestedWrappedModule,
    NestedWrappedModuleWithDelay,
    subtest_name,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    IS_LINUX,
    parametrize,
    run_tests,
    TEST_HPU,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = torch.device(get_devtype())
params = "cpu_offload,sharding_strategy"
cpu_offload_config = [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
sharding_strategy_config = [
    None,
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.NO_SHARD,
]
configs = list(itertools.product(cpu_offload_config, sharding_strategy_config))
test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",
    str(CPUOffload(offload_params=False)): "offload_false",
    str(ShardingStrategy.SHARD_GRAD_OP): "shard_grad_op",
    str(ShardingStrategy.NO_SHARD): "no_shard",
}
subtest_name = functools.partial(subtest_name, test_name_mapping)


class TestParityWithDDP(FSDPTest):
    # [CLASS DOC] Multi-process test case (needs 2+ GPUs) that trains a reference model
    # (DDP, or a custom reducer for mixture-of-experts) and an FSDP model with the same inputs
    # and compares losses and parameters, via the shared _test_fsdp_parity helper, over a grid
    # of subtest settings. Each test method picks the model class and extra init arguments.
    """
    Compare losses and parameter values after several updates when using
    PyTorch DDP vs. FullyShardedDataParallel.
    """

    def _get_device_init_modes(self, cpu_offload: CPUOffload) -> list[DEVICEInitMode]:
        # [FUNCTION DOC] HELPER: Returns the device init modes to test: DEVICE_AFTER and
        # DEVICE_BEFORE always, plus DEVICE_NEVER only when parameters are offloaded to CPU
        # (otherwise the CPU parameters could not be all-gathered over a GPU backend).
        modes = [
            DEVICEInitMode.DEVICE_AFTER,
            DEVICEInitMode.DEVICE_BEFORE,
        ]
        # Note that DEVICEInitMode.DEVICE_NEVER works currently only with CPU
        # offload as we explicitly bring the param back to CUDA device. In
        # general, it will not work since we try to all_gather p.data which is
        # on CPU but NCCL only supports GPU.
        if cpu_offload.offload_params:
            modes.append(DEVICEInitMode.DEVICE_NEVER)
        return modes

    def _get_subtest_config(self, cpu_offload: CPUOffload) -> dict[str, list[Any]]:
        # [FUNCTION DOC] HELPER: Builds the mapping of subtest option names to value lists:
        # device_init_mode (depends on cpu_offload), backward_prefetch (None, PRE, POST),
        # forward_prefetch (False, True) and use_orig_params (False, True). Passed to
        # run_subtests, which runs the cross product.
        """Returns a subtest configuration that subtests CUDA initialization
        modes and prefetching settings together."""
        return {
            "device_init_mode": self._get_device_init_modes(cpu_offload),
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
            "forward_prefetch": [False, True],
            "use_orig_params": [False, True],
        }

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/170373")
    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_nested_wrapped_model(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Runs the FSDP-vs-DDP parity check for NestedWrappedModule with
        # recursive wrapping over the standard subtest grid, for the given CPU offload and
        # sharding strategy.
        # KNOWN ISSUE: the skip decorator skips this test on every Linux machine (issue 170373),
        # so it effectively never runs on Linux.
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_nested_wrapped_model_single_iteration_mixed_precision(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Parity check for NestedWrappedModule with float16 parameter, buffer
        # and gradient-reduce dtypes, but only a single training iteration (to limit
        # accumulated low-precision divergence), over the standard subtest grid.
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            buffer_dtype=torch.float16,
            reduce_dtype=torch.float16,
        )
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            num_iters=1,
            mixed_precision=mixed_precision,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_nested_always_wrap_model(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Parity check for AlwaysWrapNestedWrappedModule (every submodule
        # wrapped in its own FSDP) with recursive wrapping over the standard subtest grid.
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            AlwaysWrapNestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_transformer(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Parity check for the transformer with shared parameters, recursively
        # wrapped, over the standard subtest grid.
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            TransformerWithSharedParams,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_delayed_optim_step(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Parity check using NestedWrappedModuleWithDelay configured with a
        # 250 ms device sleep after the loss, so the optimizer step is late and the next
        # forward all-gathers must wait for it; compares against DDP over the standard grid.
        """Tests the FSDP forward, backward, and optimizer step runtime by
        using a model with a long CUDA delay after the loss computation/before
        the optimizer step to exercise the internal CUDA stream usage in that
        the forward pass all-gathers do not start until after the optimizer
        step completes."""
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            NestedWrappedModuleWithDelay,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            init_kwargs={"delay_after_loss_ms": 250},
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_delayed_reduce_scatter(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Parity check using NestedWrappedModuleWithDelay configured with a
        # 250 ms device sleep before gradient reduction, so the backward pass must wait for the
        # delayed reduce-scatters; compares against DDP over the standard grid.
        """Tests the FSDP forward, backward, and optimizer step runtime by
        using a model with a long CUDA delay before the gradient reduce-scatter
        to exercise the internal CUDA stream usage in that the backward pass
        waits for those reductions to finish."""
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            NestedWrappedModuleWithDelay,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            init_kwargs={"delay_before_reduction_ms": 250},
        )

    def _dummy_ddp_fn(self, model):
        # [FUNCTION DOC] HELPER: Reference-model factory for mixture-of-experts parity: wraps
        # model in DummyDDP (which defers to the model's own custom gradient reduction)
        # instead of real DDP, and returns it.
        # `MixtureOfExperts`` implements custom gradient reduction logic, so
        # the reference behavior should follow that logic instead of DDP
        return DummyDDP(model)

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixture_of_experts(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Parity check for MixtureOfExperts, recursively wrapped with the
        # accelerator type as device_id, using the DummyDDP reference (custom expert gradient
        # reduction) over the standard subtest grid.
        fsdp_kwargs = {"device_id": device_type.type}
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            MixtureOfExperts,
            FSDPInitMode.RECURSIVE,
            ref_init_fn=self._dummy_ddp_fn,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            **fsdp_kwargs,
        )

    @unittest.skipIf(
        TEST_HPU, "HPU doesn't has HW sleep API support (like CUDA), skipping"
    )
    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixture_of_experts_with_delay_before_free(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Same as the mixture-of-experts parity test but the model is told to
        # sleep 250 ms before freeing parameters, stressing stream synchronization around
        # freeing. Skipped on HPU (no hardware sleep API).
        fsdp_kwargs = {"device_id": device_type.type}
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            MixtureOfExperts,
            FSDPInitMode.RECURSIVE,
            ref_init_fn=self._dummy_ddp_fn,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            init_kwargs={"delay_before_free_ms": 250},
            **fsdp_kwargs,
        )


class TestParamInit(FSDPTest):
    # [CLASS DOC] Multi-process test case about parameter initialization semantics of FSDP.
    @skip_if_lt_x_gpu(2)
    @parametrize("mixed_precision", [True, False])
    def test_param_change_after_init(self, mixed_precision):
        # [FUNCTION DOC] Builds two identical deterministic FSDP transformers (optionally with
        # default MixedPrecision), computes a forward output from the first, then overwrites
        # the second's first parameter in place with random normal values after wrapping and
        # asserts its forward output differs, proving post-init in-place edits take effect.
        """
        Tests that changing FSDP model parameter values in-place after FSDP
        initialization persist.
        """
        # Establish reference behavior
        fsdp_kwargs = {"device_id": device_type}
        if mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision()
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs,
            deterministic=True,
        )
        input = fsdp_model.module.get_input(device_type)
        ref_output = fsdp_model(*input)
        # Initialize the same model but change its first parameter value
        # in-place after FSDP initialization
        new_fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs,
            deterministic=True,
        )
        first_param = next(new_fsdp_model.parameters())
        nn.init.normal_(first_param.data)
        new_output = new_fsdp_model(*input)
        self.assertNotEqual(
            ref_output,
            new_output,
            msg="new_output did not reflect change to param after init",
        )


class TestHooks(FSDPTest):
    # [CLASS DOC] Multi-process test case checking that FSDP registers its pre- and
    # post-backward hooks during forward.
    @skip_if_lt_x_gpu(2)
    @parametrize("cuda_first", [False, True])
    def test_pre_backward_hook_registration(self, cuda_first: bool):
        # [FUNCTION DOC] Builds a recursively wrapped transformer (moved to device before or
        # after wrapping per cuda_first) and runs the shared pre-backward-hook registration
        # check on it.
        """Tests that FSDP pre-backward hooks are registered on forward pass
        outputs."""
        fsdp_kwargs = {"device_id": device_type.type}
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE if cuda_first else DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs,
        )
        self._test_pre_backward_hook_registration(fsdp_model)

    @skip_if_lt_x_gpu(2)
    def test_pre_backward_hook_registration_after_state_dict(self):
        # [FUNCTION DOC] Trains a wrapped transformer for two steps, round-trips its state dict
        # through load_state_dict, then runs the shared pre-backward-hook registration check.
        """Tests that FSDP pre-backward hooks are registered on forward pass
        outputs after saving and loading the model from a checkpoint."""
        fsdp_kwargs = {"device_id": device_type.type}
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs,
        )
        self._train_for_several_steps(fsdp_model, num_steps=2, autocast=False)
        state_dict = fsdp_model.state_dict()
        fsdp_model.load_state_dict(state_dict)
        self._test_pre_backward_hook_registration(fsdp_model)

    def _test_pre_backward_hook_registration(self, model):
        # [FUNCTION DOC] HELPER: Runs one forward on model and asserts the output tensor has
        # exactly one backward hook (the pre-backward hook); then runs backward and an SGD step
        # and asserts the hook count remains one, i.e. the hook is not removed.
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optim.zero_grad()
        # Inputs always cuda, as computation happens on CUDA device only
        input = model.module.get_input(device_type)
        output = model(*input)
        # this is pre-bwd hook
        self.assertEqual(len(output._backward_hooks), 1)
        loss = model.module.get_loss(input, output).to(device_type.type)
        loss.backward()
        # It doesn't get removed
        self.assertEqual(len(output._backward_hooks), 1)
        optim.step()
        self.assertEqual(len(output._backward_hooks), 1)

    @skip_if_lt_x_gpu(2)
    @parametrize("cuda_first", [False, True])
    @parametrize("mixed_precision", [True, False])
    def test_register_functions_called(self, cuda_first: bool, mixed_precision: bool):
        # [FUNCTION DOC] Verifies that one forward call invokes the runtime helper that
        # registers pre-backward hooks (counted by a wrapper that delegates to the original)
        # and the one that registers post-backward hooks (a plain mock). Both counts must be
        # zero before the forward and non-zero after. Patches module attributes temporarily.
        """Tests that ``_register_{pre|post}_backward_hooks()`` are called
        during the FSDP forward."""
        fsdp_kwargs = {"device_id": device_type.type}
        if mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision()
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE if cuda_first else DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs,
        )
        input = fsdp_model.module.get_input(device_type)
        # Since `_register_pre_backward_hooks()` modifies the forward output,
        # we cannot directly mock it. We implement our own counter instead.
        orig_register_pre_backward_hooks = (
            torch.distributed.fsdp._runtime_utils._register_pre_backward_hooks
        )
        register_pre_backward_hooks_call_count = 0

        def _register_pre_backward_hooks_with_count(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Increments the enclosing call counter and forwards all
            # arguments to the original pre-backward-hook registration function, returning its
            # result (which is the possibly modified forward output).
            nonlocal register_pre_backward_hooks_call_count
            register_pre_backward_hooks_call_count += 1
            return orig_register_pre_backward_hooks(*args, **kwargs)

        with (
            mock.patch(
                "torch.distributed.fsdp._runtime_utils._register_pre_backward_hooks",
                _register_pre_backward_hooks_with_count,
            ),
            mock.patch(
                "torch.distributed.fsdp._runtime_utils._register_post_backward_hook"
            ) as register_post_bwd_mock,
        ):
            self.assertEqual(register_pre_backward_hooks_call_count, 0)
            self.assertFalse(register_post_bwd_mock.called)
            fsdp_model(*input)
            self.assertTrue(register_pre_backward_hooks_call_count > 0)
            self.assertTrue(register_post_bwd_mock.called)


class TestNoGrad(FSDPTest):
    # [CLASS DOC] Multi-process test case comparing eval-mode and no_grad forwards.
    @skip_if_lt_x_gpu(2)
    @parametrize("mixed_precision", [True, False])
    def test_transformer_no_grad(self, mixed_precision):
        # [FUNCTION DOC] Trains a wrapped transformer one step (with float16 mixed precision
        # if requested), then compares the output of a forward in eval mode with one in
        # no_grad mode on the same input and asserts they are equal.
        """Tests that for an FSDP-wrapped transformer model with shared
        parameters, after training for one iteration, running a forward pass in
        ``eval()`` mode gives the same output as running a forward pass in
        ``torch.no_grad()``."""
        fsdp_kwargs = {"device_id": device_type.type}
        if mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            fsdp_kwargs["mixed_precision"] = None
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs,
        )
        self._train_for_several_steps(
            fsdp_model,
            num_steps=1,
            autocast=False,
            mixed_precision=fsdp_kwargs["mixed_precision"],
        )
        input = fsdp_model.module.get_input(device_type)
        # Run a forward in eval mode
        fsdp_model.eval()
        ref_output = fsdp_model(*input)
        # Run a forward in `no_grad()` and compare
        with torch.no_grad():
            no_grad_output = fsdp_model(*input)
        self.assertEqual(ref_output, no_grad_output)


class TestAutograd(FSDPTest):
    # [CLASS DOC] Multi-process test case about how FSDP exposes unsharded parameters to
    # autograd; patches FlatParamHandle internals temporarily during its subtests.
    @skip_if_lt_x_gpu(2)
    def test_unshard_params_as_tensors(
        self,
    ):
        # [FUNCTION DOC] Runs the unshard-as-tensors check over FULL_SHARD and SHARD_GRAD_OP,
        # use_orig_params on/off, forward prefetch on/off and three backward prefetch settings.
        # NO_SHARD is excluded because it reuses the unsharded-view helper for sharded views.
        """
        Tests that FSDP always unshards the logical parameters as ``Tensor``
        views during forward and backward computation even when forward and/or
        backward prefetching.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    # Skip testing `NO_SHARD` since it doubly uses
                    # `_use_unsharded_views()` for sharded views. Testing
                    # `FULL_SHARD` and `SHARD_GRAD_OP` provides good confidence
                    # that the `as_params` logic is correct.
                ],
                "use_orig_params": [False, True],
                "forward_prefetch": [False, True],
                "backward_prefetch": [
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                    None,
                ],
            },
            self._test_unshard_params_as_tensors,
        )

    def _test_unshard_params_as_tensors(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        forward_prefetch: bool,
        backward_prefetch: BackwardPrefetch | None,
    ):
        # [FUNCTION DOC] HELPER: Wraps a stack of five small Linear layers so each gets its own
        # FSDP instance (asserting six FSDP modules including the root), then for three
        # iterations runs forward and backward while the flat-parameter handle's
        # unsharded-views method is replaced by a checker that fails if parameters (rather than
        # plain tensors) are requested.
        orig_use_unsharded_views = FlatParamHandle._use_unsharded_views

        def _use_unsharded_views_assert_as_tensors(
            self: FlatParamHandle, as_params: bool
        ) -> None:
            # [FUNCTION DOC] HELPER: Replacement for the handle's unsharded-views method:
            # asserts (via _p_assert, which raises on failure) that as_params is false, then
            # delegates to the original implementation and returns None.
            _p_assert(
                not as_params, "Expects to use Tensor views but using parameter views"
            )
            return orig_use_unsharded_views(self, as_params)

        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
            "use_orig_params": use_orig_params,
            "forward_prefetch": forward_prefetch,
            "backward_prefetch": backward_prefetch,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
            "device_id": device_type,
        }
        # Define a model with enough FSDP instances to exercise prefetching
        NUM_LINEARS = 5
        model = nn.Sequential(
            *[nn.Linear(3, 3, device=device_type) for _ in range(NUM_LINEARS)]
        )
        fsdp_model = FSDP(model, **fsdp_kwargs)
        self.assertEqual(len(list(FSDP.fsdp_modules(fsdp_model))), NUM_LINEARS + 1)
        for _ in range(3):
            inp = torch.randn((2, 3), device=device_type)
            with self._patch_use_unsharded_views(
                _use_unsharded_views_assert_as_tensors
            ):
                loss = fsdp_model(inp).sum()
                loss.backward()

    @contextlib.contextmanager
    def _patch_use_unsharded_views(self, new_use_unsharded_views: Callable):
        # [FUNCTION DOC] HELPER: Context manager that replaces the class-level
        # unsharded-views method of FlatParamHandle with the given callable for the duration
        # of the block and restores the original afterward, even on error. Global side effect.
        orig_use_unsharded_views = FlatParamHandle._use_unsharded_views
        FlatParamHandle._use_unsharded_views = new_use_unsharded_views
        try:
            yield
        finally:
            FlatParamHandle._use_unsharded_views = orig_use_unsharded_views


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestHooks, globals(), only_for=devices, allow_xpu=True)
instantiate_device_type_tests(
    TestParityWithDDP, globals(), only_for=devices, allow_xpu=True
)
instantiate_device_type_tests(TestNoGrad, globals(), only_for=devices, allow_xpu=True)
instantiate_device_type_tests(
    TestParamInit, globals(), only_for=devices, allow_xpu=True
)
instantiate_device_type_tests(TestAutograd, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()
