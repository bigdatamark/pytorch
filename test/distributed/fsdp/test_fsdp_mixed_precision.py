# Owner(s): ["oncall: distributed"]
# [MODULE DOC] FSDP mixed precision tests (param, buffer and reduce dtypes, grad scaler,
# eval-time full precision, input casting, batchnorm handling, ignored modules, EMA flow).
# Defines module level MixedPrecision configs (default_mp, mp_only_reduce,
# mp_only_param_and_buf, mp_no_mixed_precision and a bf16 variant when NCCL >= 2.10) and the
# cross product of configs used for parametrized test names.
# Import-time side effects: exits the process with code 0 if torch.distributed is
# unavailable or under dev-asan; tries to import torchvision to define a skip decorator.
# Several tests set the FSDP_USE_FULL_PREC_IN_EVAL environment variable and never reset it.

import contextlib
import itertools
import os
import sys
from functools import partial
from itertools import product
from typing import Any

import torch
import torch.cuda.nccl as nccl
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.swa_utils import AveragedModel
from torch.testing._internal.common_distributed import (
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    get_devtype,
    subtest_name,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


device_type = torch.device(get_devtype())

try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = skip_but_pass_in_sandcastle_if(
    not HAS_TORCHVISION, "no torchvision"
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

# Various mixed precision configs to test under.
default_mp = MixedPrecision(
    param_dtype=torch.float16,
    buffer_dtype=torch.float16,
    reduce_dtype=torch.float16,
)

# Params and buffers are not cast, comm only happens
# in reduced precision.
mp_only_reduce = MixedPrecision(reduce_dtype=torch.float16)

# Only parameters are cast (thus comm should happen in the param_dtype precision)
mp_only_param_and_buf = MixedPrecision(
    param_dtype=torch.float16, buffer_dtype=torch.float16
)

# Nothing is cast (thus param, comm, grad, and buffer should be in the full precision)
mp_no_mixed_precision = MixedPrecision()

nccl_supports_bf16 = dist.is_nccl_available() and nccl.version() >= (2, 10)

mp_configs = [default_mp, mp_only_reduce, mp_only_param_and_buf, mp_no_mixed_precision]
if nccl_supports_bf16:
    mp_diff_buffer_and_reduce = MixedPrecision(
        param_dtype=torch.float16,
        buffer_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    mp_configs.extend([mp_diff_buffer_and_reduce])

# Buffer original dtype, which can differ from model params.
_BUFFER_ORIG_DTYPE = torch.float64

params = "mp_config,cpu_offload,full_precision_param_dtype,enable_sharded_grad_scaler"
cpu_offload_config = [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
full_precision_param_dtype_config = [torch.float32, torch.float64]
enable_sharded_grad_scaler = ["enable_sharded_grad_scaler", None]

configs = list(
    product(
        mp_configs,
        cpu_offload_config,
        full_precision_param_dtype_config,
        enable_sharded_grad_scaler,
    )
)

test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",
    str(CPUOffload(offload_params=False)): "offload_false",
    str(default_mp): "mp_fp16",
    str(mp_only_reduce): "mp_only_reduce",
    str(mp_only_param_and_buf): "mp_only_param_and_buf",
    str(mp_no_mixed_precision): "mp_no_mp",
    str(torch.float32): "fp32",
    str(torch.float64): "fp64",
    "enable_sharded_grad_scaler": "enable_sharded_grad_scaler",
}

if nccl_supports_bf16:
    test_name_mapping.update(
        {
            str(mp_diff_buffer_and_reduce): "mp_diff_buffer_reduce",
        }
    )

subtest_name = partial(subtest_name, test_name_mapping)

_CURRENT_FULL_PRECISION_PARAM_DTYPE = None


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter, full_precision_param_dtype):
    # [FUNCTION DOC] HELPER: Context manager that temporarily replaces the reduce_scatter_single
    # function of the distributed package with new_reduce_scatter and records
    # full_precision_param_dtype in the module global _CURRENT_FULL_PRECISION_PARAM_DTYPE.
    # On exit (even on error) it restores the original function and resets the global to None.
    """
    Patches ``dist.reduce_scatter_single`` with ``new_reduce_scatter`` and
    restores upon exiting. Used for validation of mixed precision.
    """
    orig_reduce_scatter = dist.reduce_scatter_single
    dist.reduce_scatter_single = new_reduce_scatter
    global _CURRENT_FULL_PRECISION_PARAM_DTYPE
    _CURRENT_FULL_PRECISION_PARAM_DTYPE = full_precision_param_dtype
    try:
        yield
    finally:
        dist.reduce_scatter_single = orig_reduce_scatter
        _CURRENT_FULL_PRECISION_PARAM_DTYPE = None


class LinearMixedPrecision(nn.Module):
    # [CLASS DOC] HELPER: Bias-free 10x10 linear module plus one float64 buffer, whose forward
    # optionally asserts that FSDP mixed precision is applied correctly. State: lin, the
    # buffer under a configurable name, the original param and buffer dtypes, run_checks.
    """
    A linear module with extra checks for mixed precision training.
    """

    def __init__(self, param_dtype, buffer_name="buffer", run_checks=True):
        # [FUNCTION DOC] HELPER: Creates the linear layer in param_dtype and registers a (1, 2)
        # random buffer of the module level original buffer dtype (float64) under buffer_name.
        # Stores the original dtypes and the run_checks flag for use in forward.
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False).to(param_dtype)
        # Use a configurable buffer name to avoid all submodules sharing the
        # same buffer name, which may hide prefixed vs. unprefixed name bugs
        self.buffer_name = buffer_name
        self.register_buffer(buffer_name, torch.randn((1, 2), dtype=_BUFFER_ORIG_DTYPE))
        self._orig_param_type = param_dtype
        self._orig_buffer_dtype = _BUFFER_ORIG_DTYPE
        self.run_checks = run_checks

    def forward(self, tup):
        # [FUNCTION DOC] HELPER: Takes a tuple (input, test case, FSDP root, MixedPrecision
        # config, full precision param dtype) and returns the same tuple with the input
        # replaced by the linear output, so it can be chained in a Sequential. When run_checks
        # is set it asserts, via the test case, that the input and buffer have the configured
        # mixed precision dtypes (or originals when unset). It also checks each sharded FSDP
        # unit: if active, params have the low precision dtype and the _mp_shard storage is
        # freed (or absent when param_dtype is unset); if inactive, params are full precision.
        # With world size above one, at least one unit must be active. Raises on mismatch.
        inp, cls, fsdp, mp_config, full_precision_param_dtype = tup
        if self.run_checks:
            # Param and input should be the mixed precision type
            expected_param_type = (
                mp_config.param_dtype
                if mp_config.param_dtype is not None
                else self._orig_param_type
            )
            expected_buffer_type = (
                mp_config.buffer_dtype
                if mp_config.buffer_dtype is not None
                else self._orig_buffer_dtype
            )
            cls.assertEqual(inp.dtype, expected_param_type)
            # Buffer should be in specified precision as well.
            cls.assertEqual(getattr(self, self.buffer_name).dtype, expected_buffer_type)

            # In FSDP, self.params should point to the right type.
            num_active_fsdp = 0
            for fsdp_module in FSDP.fsdp_modules(fsdp):
                fsdp_managed_params = fsdp_module.params
                # Single param assumption
                cls.assertEqual(1, len(fsdp_managed_params))
                for param in fsdp_managed_params:
                    # FSDP unit is currently active if it is not using the param
                    # local shard. This supports both FULL_SHARD and SHARD_GRAD_OP
                    # cases. In FULL_SHARD, we have the additional property that
                    # param._full_param_padded has not been freed.
                    param_is_sharded = (
                        fsdp_module.sharding_strategy != ShardingStrategy.NO_SHARD
                        and fsdp_module.world_size > 1
                    )
                    is_fsdp_unit_active = (
                        param_is_sharded
                        and param.data.data_ptr() != param._local_shard.data_ptr()
                    )
                    if is_fsdp_unit_active:
                        num_active_fsdp += 1
                        # This FSDP unit is active, verify param points to mixed
                        cls.assertEqual(param.dtype, expected_param_type)
                        # _unshard should have also freed the fp16 shard.
                        # Shard is never allocated if param_dtype mixed precision is not
                        # enabled.
                        if mp_config.param_dtype is not None:
                            cls.assertEqual(0, param._mp_shard.untyped_storage().size())
                        else:
                            cls.assertFalse(hasattr(param, "_mp_shard"))
                    elif param_is_sharded:
                        # This FSDP unit is not active as full param has been
                        # freed or not yet allocated. Ensure param points to full
                        # precision param.
                        cls.assertEqual(param.dtype, full_precision_param_dtype)
            # We should have gotten at least one active FSDP unit for sharded
            # (world size > 1) cases. For cases where param is not sharded
            # (ie world_size == 1) it is a bit hard to check if FSDP unit is active
            # as we'd always point to the local shard, so we rely on the forward
            # pass self.lin(inp) working well and inp being reduced precision to
            # implicitly validate that the param is indeed in the reduced precision.
            if cls.world_size > 1:
                cls.assertGreater(num_active_fsdp, 0)

        return (self.lin(inp), cls, fsdp, mp_config, full_precision_param_dtype)


class TestFSDPMixedPrecision(FSDPTest):
    # [CLASS DOC] Base test case with shared helpers for FSDP mixed precision tests: model
    # builders, validators of _mp_shard allocation, a reduce-scatter dtype checker and the
    # end to end train loop. Subclasses define world_size and the concrete tests.
    @property
    def world_size(self):
        # [FUNCTION DOC] Abstract property; always raises ValueError because child classes
        # must override it.
        raise ValueError("To be implemented by child classes")

    def _get_simple_nested_model(
        self, param_dtype, run_checks, *fsdp_args, **fsdp_kwargs
    ):
        # [FUNCTION DOC] HELPER: Builds an FSDP-wrapped Sequential of two LinearMixedPrecision
        # modules (buffers named buffer0 and buffer1) on the test device. The first is wrapped
        # in its own inner FSDP; fsdp_args and fsdp_kwargs go to every FSDP constructor.
        # Returns the root FSDP model.
        model = FSDP(
            nn.Sequential(
                FSDP(
                    LinearMixedPrecision(
                        param_dtype, buffer_name="buffer0", run_checks=run_checks
                    ).to(device_type),
                    *fsdp_args,
                    **fsdp_kwargs,
                ),
                LinearMixedPrecision(
                    param_dtype, buffer_name="buffer1", run_checks=run_checks
                ).to(device_type),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        return model

    def _get_simple_model(self, param_dtype, *fsdp_args, **fsdp_kwargs):
        # [FUNCTION DOC] HELPER: Returns a single LinearMixedPrecision (default buffer name,
        # checks on) moved to the test device and wrapped in one FSDP with the given arguments.
        model = FSDP(
            LinearMixedPrecision(param_dtype).to(device_type), *fsdp_args, **fsdp_kwargs
        )
        return model

    def _validate_no_mp_shard(self, fsdp_model):
        # [FUNCTION DOC] HELPER: Asserts that no flat parameter of any FSDP unit under
        # fsdp_model has an _mp_shard attribute (it is only allocated with param_dtype set).
        """
        Validates that there is no mixed precision _mp_shard allocated
        when it is not expected to be.
        """
        fsdp_units = FSDP.fsdp_modules(fsdp_model)
        for fsdp in fsdp_units:
            for param in fsdp.params:
                self.assertFalse(hasattr(param, "_mp_shard"))

    def _validate_mp_shard_freed(self, fsdp_model):
        # [FUNCTION DOC] HELPER: Asserts that for every flat parameter of every FSDP unit under
        # fsdp_model the _mp_shard untyped storage has size zero (low precision shard freed).
        """
        Ensures that the mixed precision shard is greed for all FSDP units.
        """
        fsdp_units = FSDP.fsdp_modules(fsdp_model)
        for fsdp in fsdp_units:
            for param in fsdp.params:
                self.assertEqual(0, param._mp_shard.untyped_storage().size())

    def _reduce_scatter_validate_mp(
        self, orig_reduce_scatter, mp_config, should_run_low_prec, *args, **kwargs
    ):
        # [FUNCTION DOC] HELPER: Wrapper used in place of reduce_scatter_single. Gathers all
        # tensor arguments, computes the expected dtype (if should_run_low_prec: reduce_dtype,
        # else param_dtype, else the current full precision dtype global; if not
        # should_run_low_prec: the full precision dtype) and asserts every tensor has it.
        # Then calls orig_reduce_scatter with the original arguments and returns its result.
        """
        Runs reduce-scatter but verifies mixed precision settings before. This
        is to test mixed precision is working as expected during backward pass.
        In particular it ensures that the gradients were cast to the right type
        and comm. is going to happen in the right type.
        """
        tensors = []
        for x in args:
            if isinstance(x, torch.Tensor):
                tensors.append(x)
        for x in kwargs.values():
            if isinstance(x, torch.Tensor):
                tensors.append(x)

        # reduce_dtype has higher priority than param_dtype, because mixed_precision
        # supports overriding param_dtype with reduce_dtype to control the
        # reduction precision. In the case where reduce_dtype == param_dtype
        # this tests that gradients are in the expected precision as well.
        # If reduce_dtype is not specified (is None) we comm. in the param_dtype
        # if that is specified, otherwise full precision dtype.
        if should_run_low_prec:
            expected_dtype = (
                mp_config.reduce_dtype
                if mp_config.reduce_dtype is not None
                else (
                    mp_config.param_dtype
                    if mp_config.param_dtype is not None
                    else _CURRENT_FULL_PRECISION_PARAM_DTYPE
                )
            )
        else:
            expected_dtype = _CURRENT_FULL_PRECISION_PARAM_DTYPE

        for t in tensors:
            self.assertEqual(
                expected_dtype,
                t.dtype,
                lambda msg: f"{msg}\nExpected to reduce in {expected_dtype} but got tensors in {t.dtype}",
            )

        return orig_reduce_scatter(*args, **kwargs)

    def _test_grads_reduced_precision(
        self, offload_params: bool, use_orig_params: bool
    ):
        # [FUNCTION DOC] HELPER: Wraps a two-linear model in nested FSDP with fp16 mixed
        # precision and keep_low_precision_grads enabled, using the given cpu offload and
        # use_orig_params settings. Runs 6 forward/backward steps and asserts every present
        # gradient is float16, then calls a barrier.
        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: Two stacked 10x10 linear layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates lin1 and lin2.
                super().__init__()
                self.lin1 = nn.Linear(10, 10)
                self.lin2 = nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies lin1 then lin2.
                return self.lin2(self.lin1(x))

        m = MyModel().to(device_type)
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
            keep_low_precision_grads=True,
        )
        fsdp_kwargs = {
            "mixed_precision": mp,
            "cpu_offload": CPUOffload(offload_params=offload_params),
            "use_orig_params": use_orig_params,
        }
        m.lin1 = FSDP(m.lin1, **fsdp_kwargs)
        m = FSDP(m, **fsdp_kwargs)
        for _ in range(6):
            inp = torch.ones(1, 10)
            m(inp).sum().backward()
            for param in m.parameters():
                if param.grad is not None:
                    self.assertEqual(torch.float16, param.grad.dtype)

        dist.barrier()

    def _run_test_mixed_precision_e2e(
        self,
        mp_config,
        cpu_offload,
        backward_prefetch,
        forward_prefetch,
        full_precision_param_dtype,
        sharding_strategy,
        enable_sharded_grad_scaler,
    ):
        # [FUNCTION DOC] HELPER: Core end to end check. Sets the device to this rank, builds a
        # simple and a nested FSDP model with the given sharding, offload, prefetch and mixed
        # precision settings, and for each model, with reduce-scatter patched to validate
        # dtypes, runs 3 iterations of forward, scaled backward and optimizer step. Asserts the
        # dtypes of buffers, loss, params and grads, that _mp_shard is freed or never
        # allocated, that summon_full_params yields full precision params and that state_dict
        # entries are in original dtypes (buffers float64, params full precision).
        torch.accelerator.set_device_index(self.rank)
        fsdp_models = [
            self._get_simple_model(
                param_dtype=full_precision_param_dtype,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                mixed_precision=mp_config,
                backward_prefetch=backward_prefetch,
                forward_prefetch=forward_prefetch,
            ),
            self._get_simple_nested_model(
                param_dtype=full_precision_param_dtype,
                run_checks=True,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                mixed_precision=mp_config,
                backward_prefetch=backward_prefetch,
                forward_prefetch=forward_prefetch,
            ),
        ]
        for model in fsdp_models:
            if not cpu_offload.offload_params:
                model.to(device_type)

            # Patch reduce_scatter to add validation for mixed precision types.
            orig_reduce_scatter = dist.reduce_scatter_single
            test_reduce_scatter = partial(
                self._reduce_scatter_validate_mp,
                orig_reduce_scatter,
                mp_config,
                True,
            )
            with patch_reduce_scatter(test_reduce_scatter, full_precision_param_dtype):
                scaler = ShardedGradScaler(enabled=enable_sharded_grad_scaler)
                optim = torch.optim.Adam(model.parameters())

                for _ in range(3):
                    inp = torch.randn(
                        3, 10, device=device_type, dtype=full_precision_param_dtype
                    )
                    # Forward pass of LinearMixedPrecision check casting of
                    # inputs, params, buffers.
                    act, *_ = model(
                        (inp, self, model, mp_config, full_precision_param_dtype)
                    )
                    # Buffers should be casted.
                    for buf in model.buffers():
                        if mp_config.buffer_dtype is not None:
                            self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                        else:
                            self.assertEqual(buf.dtype, _BUFFER_ORIG_DTYPE)
                    # p._mp_shard should be freed.
                    if mp_config.param_dtype is not None:
                        self._validate_mp_shard_freed(model)
                    else:
                        # We never should have allocated an _mp_shard.
                        self._validate_no_mp_shard(model)

                    loss = act.sum()
                    loss = scaler.scale(loss)
                    if mp_config.param_dtype is not None:
                        self.assertEqual(loss.dtype, mp_config.param_dtype)
                    else:
                        self.assertEqual(loss.dtype, full_precision_param_dtype)
                    # Will run patched reduce scatter that validates mixed_precision
                    # types in backward.
                    loss.backward()
                    # Buffers stay casted even after backwards.
                    for buf in model.buffers():
                        if mp_config.buffer_dtype is not None:
                            self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                        else:
                            self.assertEqual(buf.dtype, _BUFFER_ORIG_DTYPE)
                    # p._mp_shard should be freed.
                    if mp_config.param_dtype is not None:
                        self._validate_mp_shard_freed(model)
                    else:
                        self._validate_no_mp_shard(model)

                    # Ensure params and grads are in full precision,
                    # as after fwd/backward we maintain full precision shards.
                    for param in model.parameters():
                        self.assertEqual(param.dtype, full_precision_param_dtype)
                        if param.grad is not None:
                            self.assertEqual(
                                param.grad.dtype, full_precision_param_dtype
                            )

                    # Unscale the gradients and step
                    scaler.step(optim)
                    # Update the scale factor
                    scaler.update()

                    # Summon full params should be in full precision
                    with model.summon_full_params(model):
                        # It is not expected for summon_full_params to allocate
                        # a mixed precision shard.
                        if mp_config.param_dtype is not None:
                            self._validate_mp_shard_freed(model)
                        else:
                            self._validate_no_mp_shard(model)
                        params = list(model.parameters())
                        for p in params:
                            self.assertEqual(p.dtype, full_precision_param_dtype)

                        # Note that buffers are cast only once and only restored
                        # to the original buffer dtype in state_dict, so
                        # summon_full_params is not expected to restore buffer
                        # types to their original.
                        named_buffers = dict(model.named_buffers())
                        for v in named_buffers.values():
                            if mp_config.buffer_dtype is not None:
                                self.assertEqual(v.dtype, mp_config.buffer_dtype)
                            else:
                                self.assertEqual(v.dtype, _BUFFER_ORIG_DTYPE)

                    # state_dict should be in full precision
                    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    for name, tensor in state_dict.items():
                        # Parameters and buffers are checkpointed in their
                        # original dtypes, which may be different.
                        if name in named_buffers:
                            self.assertEqual(tensor.dtype, _BUFFER_ORIG_DTYPE)
                        else:
                            self.assertEqual(
                                tensor.dtype,
                                full_precision_param_dtype,
                                lambda msg: f"{msg}\n{name}: {tensor.dtype} vs {full_precision_param_dtype}",
                            )

                    # After state_dict, buffer's dtype should have been restored
                    # to the mixed precision one.
                    for buf in model.buffers():
                        if mp_config.buffer_dtype is not None:
                            self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                        else:
                            self.assertEqual(buf.dtype, _BUFFER_ORIG_DTYPE)


class TestFSDPMixedPrecisionSharded(TestFSDPMixedPrecision):
    # [CLASS DOC] Mixed precision tests run with two ranks (sharded parameters): end to end
    # configs, embedding inputs, resnet, batchnorm, eval behavior and input gradients.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns 2, the number of ranks for this test class.
        return 2

    def _get_subtest_config(self) -> dict[str, list[Any]]:
        # [FUNCTION DOC] HELPER: Returns the subtest mapping for run_subtests: forward_prefetch
        # in False/True and backward_prefetch in None, BACKWARD_PRE, BACKWARD_POST.
        """Returns a subtest configuration that subtests prefetching settings
        together."""
        return {
            "forward_prefetch": [False, True],
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
        }

    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_no_reshard_after_forward(self):
        # [FUNCTION DOC] Needs 2 GPUs. Runs the end to end check once with the bf16 mixed
        # config if NCCL supports it (else the default fp16 config), CPU param offload,
        # float64 full precision params, SHARD_GRAD_OP sharding and the grad scaler disabled.
        # Note that we don't exercise all possible different configs so as to
        # not increase test TTS too much.
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            enable_sharded_grad_scaler=False,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixed_precision_e2e_full_shard(
        self,
        mp_config,
        cpu_offload,
        full_precision_param_dtype,
        enable_sharded_grad_scaler,
    ):
        # [FUNCTION DOC] Needs 2 GPUs; parametrized over the cross product of mixed precision
        # config, CPU offload, full precision param dtype and grad scaler flag. For each
        # prefetch subtest it runs the end to end check with FULL_SHARD sharding.
        self.run_subtests(
            self._get_subtest_config(),
            self._run_test_mixed_precision_e2e,
            mp_config=mp_config,
            cpu_offload=cpu_offload,
            full_precision_param_dtype=full_precision_param_dtype,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            enable_sharded_grad_scaler=enable_sharded_grad_scaler,
        )

    def _test_mixed_precision_embedding_table(self, mp_config):
        # [FUNCTION DOC] HELPER: With reduce-scatter patched to validate dtypes, wraps a
        # TransformerWithSharedParams (which has an embedding taking integer inputs) in one
        # top-level FSDP using mp_config and trains 6 SGD steps. Asserts the loss has the
        # param dtype (float32 if unset), which also fails if integer inputs were wrongly cast.
        # Basic test to ensure int inputs are not casted which would break
        # modules such as embedding tables.
        param_dtype = mp_config.param_dtype or torch.float32
        orig_reduce_scatter = dist.reduce_scatter_single
        test_reduce_scatter = partial(
            self._reduce_scatter_validate_mp,
            orig_reduce_scatter,
            mp_config,
            True,
        )
        with patch_reduce_scatter(test_reduce_scatter, param_dtype):
            # TODO: `test_mp_embedding_reduce()` fails if we do not wrap the
            # entire `TransformerWithSharedParams` with a single top-level FSDP
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                DEVICEInitMode.DEVICE_BEFORE,
                {"mixed_precision": mp_config},
            )
            fsdp_model = FSDP(model, mixed_precision=mp_config)
            optim = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
            for _ in range(6):
                inp = fsdp_model.module.get_input(torch.device(device_type))
                # This would fail if we casted integer module inputs such as for
                # embedding tables.
                output = fsdp_model(*inp)
                loss = fsdp_model.module.get_loss(inp, output).to(device_type)
                self.assertEqual(loss.dtype, param_dtype)
                fsdp_model.module.run_backward(loss)
                optim.step()

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_reduce(self):
        # [FUNCTION DOC] Needs 2 GPUs. Embedding check with only reduce_dtype set to float16.
        self._test_mixed_precision_embedding_table(
            mp_config=MixedPrecision(reduce_dtype=torch.float16)
        )

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_only_params_and_bufs(self):
        # [FUNCTION DOC] Needs 2 GPUs. Embedding check with param and buffer dtype float16
        # and no reduce dtype.
        self._test_mixed_precision_embedding_table(
            mp_config=MixedPrecision(
                param_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        )

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_default(self):
        # [FUNCTION DOC] Needs 2 GPUs. Embedding check with param, buffer and reduce dtypes
        # all float16.
        default_mp_config = MixedPrecision(
            param_dtype=torch.float16,
            buffer_dtype=torch.float16,
            reduce_dtype=torch.float16,
        )
        self._test_mixed_precision_embedding_table(mp_config=default_mp_config)

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_params_and_reduce_diff(self):
        # [FUNCTION DOC] Needs 2 GPUs. Embedding check with float16 param and buffer dtypes
        # and a float32 reduce dtype.
        params_and_reduce_different = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )
        self._test_mixed_precision_embedding_table(
            mp_config=params_and_reduce_different
        )

    @skip_if_lt_x_gpu(2)
    @skipIfNoTorchVision
    def test_mixed_precision_resnet(self):
        # [FUNCTION DOC] Needs 2 GPUs and torchvision. Converts resnet50 to sync batchnorm,
        # wraps it with FSDP using size-based auto wrap and fp16 mixed precision, asserts the
        # number of FSDP units wrapping a batchnorm equals the batchnorm count in the original
        # model, then runs a forward and backward pass to confirm no dtype mismatch.
        """
        End to end test to ensure mixed precision + auto_wrap works
        for ResNet model.
        """
        resnet_model = torchvision.models.resnet50().to(device_type)
        resnet_model = nn.SyncBatchNorm.convert_sync_batchnorm(
            resnet_model, process_group=dist.distributed_c10d._get_default_group()
        )
        n_bn = sum(
            1 if isinstance(x, _BatchNorm) else 0 for x in resnet_model.modules()
        )
        inp = torch.ones(1, 3, 1000, 1000, device=device_type)
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        fsdp = FSDP(
            resnet_model,
            auto_wrap_policy=size_based_auto_wrap_policy,
            mixed_precision=mp_config,
        )
        # Batchnorm units should be wrapped individually. Validate this by
        # ensuring there are equal no. of FSDP units that are BN as BN units
        # in original resnet model.
        fsdp_bn = 0
        for module in fsdp.fsdp_modules(fsdp):
            wrapped_module = module.module
            if isinstance(wrapped_module, _BatchNorm):
                fsdp_bn += 1

        self.assertEqual(fsdp_bn, n_bn)
        # Would throw type mismatch issue without mixed precision autowrapping.
        loss = fsdp(inp).sum()
        loss.backward()

    @skip_if_lt_x_gpu(2)
    def test_grads_reduced_precision(self):
        # [FUNCTION DOC] Needs 2 GPUs. Runs the low precision gradient check over all
        # combinations of offload_params and use_orig_params.
        self.run_subtests(
            {
                "offload_params": [False, True],
                "use_orig_params": [False, True],
            },
            self._test_grads_reduced_precision,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("convert_sync_bn", [True, False])
    def test_mp_batchnorm(self, convert_sync_bn):
        # [FUNCTION DOC] Needs 2 GPUs; parametrized on converting to sync batchnorm. Wraps a
        # net with batchnorm and layernorm using fp16 mixed precision that ignores those module
        # classes and a policy that never wraps. Expects a UserWarning, asserts the norm
        # modules became separate FSDP units with default (no) mixed precision while the other
        # layers stay unwrapped and the root keeps the config, then runs forward and backward.
        class BatchNormNet(nn.Module):
            # [CLASS DOC] HELPER: Small net of linear, batchnorm, linear, layernorm, linear
            # layers ending with a softmax.
            def __init__(self, affine=True):
                # [FUNCTION DOC] HELPER: Creates fc1 (2 to 40), bn (BatchNorm1d, 4 features,
                # affine per argument), fc2 (40 to 4), ln (LayerNorm 4) and fc3 (4 to 4).
                super().__init__()
                self.fc1 = nn.Linear(2, 40, bias=False)
                self.bn = nn.BatchNorm1d(4, affine=affine)
                self.fc2 = nn.Linear(40, 4, bias=False)
                self.ln = nn.LayerNorm(4)
                self.fc3 = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, reshapes to batch of 4 by 10, applies bn,
                # flattens to 40 features, applies fc2, ln and fc3, and returns the softmax
                # over dimension 1.
                x = torch.reshape(self.fc1(x), (-1, 4, 10))
                x = self.bn(x)
                x = torch.reshape(x, (-1, 40))
                x = self.fc2(x)
                x = self.ln(x)
                x = self.fc3(x)
                return F.softmax(x, dim=1)

        def never_wrap_policy(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Auto wrap policy that ignores its arguments and always
            # returns False, so no submodule is wrapped by the policy.
            return False

        net = BatchNormNet().to(device_type)
        if convert_sync_bn:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        # FSDP detects that mixed precision + batchnorm will cause issues
        # and thus wrap batchnorm in a distinct FSDP unit that does not
        # use mixed precision.
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
            _module_classes_to_ignore=[_BatchNorm, nn.LayerNorm],
        )
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="These modules will be wrapped as separate FSDP",
        ):
            model = FSDP(
                net,
                mixed_precision=mp_config,
                auto_wrap_policy=never_wrap_policy,
            )

        no_mp = MixedPrecision()
        for mod in [model.ln, model.bn]:
            self.assertTrue(isinstance(mod, FSDP))
            self.assertEqual(no_mp, mod.mixed_precision)
        # policy should not have wrapped any other submodules
        for mod in [model.fc1, model.fc2, model.fc3]:
            self.assertFalse(isinstance(mod, FSDP))

        # Overall mixed precision is still enabled
        self.assertEqual(mp_config, model.mixed_precision)

        inp = torch.randn((1, 2), device=device_type)
        # Without FSDP BN mixed precision fix, this would result in
        # RuntimeError: Expected counts to have type Half but got Float
        # for syncBN
        model(inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_eval_root_cast_inputs(self):
        # [FUNCTION DOC] Needs 2 GPUs. For FSDP_USE_FULL_PREC_IN_EVAL set to 1 and 0, wraps a
        # model whose root has no parameters of its own (only a wrapped child) in eval mode and
        # runs forward and backward; the model raises if its input is not float32 (flag on) or
        # float16 (flag off). Side effect: sets the environment variable.
        """
        In a case where root module does not manage FSDP parameters,
        ensure that we don't cast forward inputs which could potentially
        cause a dtype mismatch. Check that FSDP_USE_FULL_PREC_IN_EVAL controls
        this.
        """

        low_prec_dtype = torch.float16

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: One linear layer that validates the dtype of its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 5 to 5 linear layer a.
                super().__init__()
                self.a = nn.Linear(5, 5)

            def forward(self, x, expect_use_full_prec_in_eval):
                # [FUNCTION DOC] HELPER: Raises AssertionError if x is not float32 when
                # expect_use_full_prec_in_eval is true, or not the low precision dtype when
                # false; otherwise returns the linear layer output.
                if expect_use_full_prec_in_eval:
                    if x.dtype != torch.float32:
                        raise AssertionError(f"Expected fp32, got {x.dtype}")
                else:
                    if x.dtype != low_prec_dtype:
                        raise AssertionError(
                            f"Expected {low_prec_dtype}, got {x.dtype}"
                        )
                return self.a(x)

        mp_config = MixedPrecision(
            param_dtype=low_prec_dtype,
            reduce_dtype=low_prec_dtype,
            buffer_dtype=low_prec_dtype,
        )

        for use_full_prec_in_eval in [True, False]:
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            m = MyModel().to(device_type)
            m.a = FSDP(m.a, mixed_precision=mp_config)
            model = FSDP(m, mixed_precision=mp_config)
            model.eval()
            inp = torch.randn(5, 5)
            model(inp, use_full_prec_in_eval).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_full_precision_in_eval(self):
        # [FUNCTION DOC] Needs 2 GPUs. For each combination of cast_forward_inputs and
        # FSDP_USE_FULL_PREC_IN_EVAL, builds a recursively wrapped transformer with fp16 mixed
        # precision, checks the training loss is float16 and grads float32, then in eval mode
        # checks the loss is float32 only if the flag is on. Side effect: sets the env variable.
        """
        Tests that eval runs in full precision if FSDP_USE_FULL_PREC_IN_EVAL is set.
        """
        for (
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False]):
            mp_config = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
                cast_forward_inputs=cast_forward_inputs,
            )
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                {"mixed_precision": mp_config},
            )
            inp = model.get_input(torch.device(device_type))
            output = model(*inp)
            loss = model.get_loss(inp, output).to(device_type)
            # Loss should be in fp16
            self.assertEqual(torch.float16, loss.dtype)
            model.run_backward(loss)
            # Grads should be in fp32 as we upcast them
            for p in model.parameters():
                if p.grad is not None:
                    self.assertEqual(torch.float32, p.grad.dtype)

            # Now in eval mode, loss should be fp32 if use_full_prec_in_eval is set.
            model.eval()
            inp = model.get_input(torch.device(device_type))
            output = model(*inp)
            loss = model.get_loss(inp, output).to(device_type)
            expected_dtype = torch.float32 if use_full_prec_in_eval else torch.float16
            self.assertEqual(expected_dtype, loss.dtype)

    @skip_if_lt_x_gpu(2)
    def test_full_precision_in_eval_buffers(self):
        # [FUNCTION DOC] Needs 2 GPUs. For each combination of cast_forward_inputs and
        # FSDP_USE_FULL_PREC_IN_EVAL, runs a nested model forward and checks buffers are fp16,
        # then in eval mode (with forward pre-hooks that check buffer dtypes) checks buffers
        # are the original float64 if the flag is on else fp16, and after train mode and
        # another forward checks they are fp16 again. Side effect: sets the env variable.
        """
        Tests that when model.eval() and FSDP_USE_FULL_PREC_IN_EVAL is set,
        buffers are in the full precision.
        """
        for (
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False]):
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            mp_config = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
                cast_forward_inputs=cast_forward_inputs,
            )
            model_getter = self._get_simple_nested_model
            fsdp_model = model_getter(
                param_dtype=torch.float32,
                run_checks=False,
                mixed_precision=mp_config,
            )

            inp = torch.randn(3, 10, device=device_type)
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            for buf in fsdp_model.buffers():
                self.assertEqual(torch.float16, buf.dtype)

            # model.eval() + forward pass should make the buffers in full prec again
            # Add pre-forward hooks
            def verify_eval_buffer_dtype(module, input):
                # [FUNCTION DOC] HELPER: Forward pre-hook asserting every buffer of module has
                # the original buffer dtype if the flag is on, else float16.
                expected_dtype = (
                    _BUFFER_ORIG_DTYPE if use_full_prec_in_eval else torch.float16
                )
                for buf in module.buffers():
                    self.assertEqual(expected_dtype, buf.dtype)

            def _get_underlying_module(m):
                # [FUNCTION DOC] HELPER: Returns the wrapped module of an FSDP instance, or m
                # itself if it is not an FSDP instance.
                return m.module if isinstance(m, FSDP) else m

            hook_handles = []
            hook_handles.append(
                _get_underlying_module(fsdp_model[0]).register_forward_pre_hook(
                    verify_eval_buffer_dtype
                )
            )
            hook_handles.append(
                _get_underlying_module(fsdp_model[1]).register_forward_pre_hook(
                    verify_eval_buffer_dtype
                )
            )

            fsdp_model.eval()
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            for hook_handle in hook_handles:
                hook_handle.remove()

            expected_dtype = (
                _BUFFER_ORIG_DTYPE if use_full_prec_in_eval else torch.float16
            )
            for buf in fsdp_model.buffers():
                self.assertEqual(expected_dtype, buf.dtype)

            # model.train() + forward again should make buffers in fp16
            fsdp_model.train()
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            for buf in fsdp_model.buffers():
                self.assertEqual(torch.float16, buf.dtype)

    @skip_if_lt_x_gpu(2)
    def test_full_precision_in_eval_comm(self):
        # [FUNCTION DOC] Needs 2 GPUs. For each combination of cast_forward_inputs and
        # FSDP_USE_FULL_PREC_IN_EVAL, runs forward and backward in eval mode on a transformer
        # with float16 reduce dtype and checks via patched reduce-scatter that gradients are
        # communicated in float16 only when the flag is off. Side effect: sets the env variable.
        for (
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False]):
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            mp_config = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32,
                cast_forward_inputs=cast_forward_inputs,
                # cast reduction for batchnorm also just in this test, to make
                # validation easier.
                _module_classes_to_ignore=[],
            )
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                {"mixed_precision": mp_config},
            )
            # Patch reduce_scatter to add validation for mixed precision types.
            orig_reduce_scatter = dist.reduce_scatter_single
            test_reduce_scatter = partial(
                self._reduce_scatter_validate_mp,
                orig_reduce_scatter,
                mp_config,
                not use_full_prec_in_eval,
            )
            model.eval()
            with patch_reduce_scatter(test_reduce_scatter, torch.float32):
                inp = model.get_input(torch.device(device_type))
                output = model(*inp)
                loss = model.get_loss(inp, output).to(device_type)
                model.run_backward(loss)

    @skip_if_lt_x_gpu(2)
    def test_input_grads_with_param_mixed_precision(self):
        # [FUNCTION DOC] Needs 2 GPUs. Runs the input gradient check over FULL_SHARD,
        # SHARD_GRAD_OP and NO_SHARD, each with use_orig_params false and true.
        """
        Tests that input tensors that require gradients do get their gradients
        even after being cast to a low precision (when parameter mixed
        precision is enabled).
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
            self._test_input_grads_with_param_mixed_precision,
        )

    def _test_input_grads_with_param_mixed_precision(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        # [FUNCTION DOC] HELPER: Wraps a bias-free 1024x1024 linear in FSDP with float16 param
        # dtype and float32 reduce and buffer dtypes. Runs forward and backward on a float32
        # input requiring grad and asserts it received a gradient that is still float32.
        model = nn.Linear(1024, 1024, bias=False)
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            device_id=torch.accelerator.current_device_index(),
            use_orig_params=use_orig_params,
        )
        # Use an input with dtype not equal to the mixed precision
        # `param_dtype` so that it gets cast
        x_float = torch.randn(
            (32, 1024),
            device=device_type,
            dtype=torch.float32,
            requires_grad=True,
        )
        fsdp_model(x_float).sum().backward()
        self.assertTrue(x_float.grad is not None)
        # Check that `x_float` preserves its dtype, meaning that the gradient
        # propagated via `ToCopyBackward0`
        self.assertEqual(x_float.grad.dtype, torch.float32)

    @skip_if_lt_x_gpu(2)
    def test_buffer_dtype_no_root_handle(self):
        # [FUNCTION DOC] Needs 2 GPUs. Builds a Sequential of conv layers and fixed-kernel
        # convs (buffers only), wraps it with FSDP (orig params, auto wrap every Sequential,
        # float16 mixed precision) so the root owns no parameters, and checks that forward and
        # backward run without dtype errors.
        class NonLearnableConv(nn.Module):
            # [CLASS DOC] HELPER: Depthwise stride 2 convolution with a fixed, normalized
            # separable kernel stored as a buffer (no learnable parameters).
            def __init__(self, kernel, in_channels: int):
                # [FUNCTION DOC] HELPER: Normalizes the 1D kernel to sum one, takes its outer
                # product, repeats it for each input channel and registers it as buffer kernel.
                # Sets padding to half the kernel length minus one, rounded down.
                super().__init__()
                self.padding = (len(kernel) - 1) // 2
                kernel = torch.tensor(kernel, dtype=torch.float32)
                kernel = kernel / kernel.sum()
                kernel = kernel.outer(kernel)[None, None].repeat(in_channels, 1, 1, 1)
                self.register_buffer("kernel", kernel)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Grouped 2D convolution of x with the kernel buffer,
                # one group per channel, stride 2 and the stored padding.
                return nn.functional.conv2d(
                    x,
                    self.kernel,
                    groups=self.kernel.shape[0],
                    stride=2,
                    padding=self.padding,
                )

        model = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 64, 3, padding=1)),
            nn.Sequential(NonLearnableConv((1, 2, 2, 1), 64)),
            nn.Sequential(nn.Conv2d(64, 3, 3, padding=1)),
            nn.Sequential(NonLearnableConv((1, 2, 2, 1), 3)),
        ).to(device_type)

        dtype = torch.float16
        model = FSDP(
            module=model,
            device_id=self.rank,
            use_orig_params=True,
            limit_all_gathers=True,
            auto_wrap_policy=ModuleWrapPolicy({nn.Sequential}),
            mixed_precision=MixedPrecision(
                param_dtype=dtype,
                buffer_dtype=dtype,
                reduce_dtype=dtype,
            ),
        )

        # Check that we can run forward/backward without dtype errors
        x = torch.randn(2, 3, 128, 128, device=device_type)
        out = model(x)
        out.mean().backward()


class TestFSDPMixedPrecisionUnsharded(TestFSDPMixedPrecision):
    # [CLASS DOC] Mixed precision tests with a single rank, so parameters are not sharded.
    """
    Smaller test suite for unshared param (i.e. world_size == 1) case.
    """

    @property
    def world_size(self):
        # [FUNCTION DOC] Returns 1, the number of ranks for this test class.
        return 1

    @skip_if_lt_x_gpu(1)
    def test_grads_reduced_precision(self):
        # [FUNCTION DOC] Needs 1 GPU. Runs the low precision gradient check over all
        # combinations of offload_params and use_orig_params.
        self.run_subtests(
            {"offload_params": [False, True], "use_orig_params": [False, True]},
            self._test_grads_reduced_precision,
        )

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_no_reshard_after_forward(self):
        # [FUNCTION DOC] Needs 1 GPU. Same single end to end run as the sharded variant:
        # SHARD_GRAD_OP, CPU offload, float64 params, bf16 config if supported else fp16.
        # Note that we don't exercise all possible different configs so as to
        # not increase test TTS too much.
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            enable_sharded_grad_scaler=False,
        )

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_e2e_full_shard(self):
        # [FUNCTION DOC] Needs 1 GPU. Single end to end run with FULL_SHARD, CPU offload,
        # float64 params, grad scaler off, bf16 config if supported else fp16.
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            enable_sharded_grad_scaler=False,
        )


instantiate_parametrized_tests(TestFSDPMixedPrecisionSharded)


class IgnoredModule(nn.Module):
    # [CLASS DOC] HELPER: Wrapper around a single 100x100 linear layer, used as a module to
    # exclude from FSDP.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the linear layer l.
        super().__init__()
        self.l = nn.Linear(100, 100)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to x.
        return self.l(x)


class ModelWithIgnoredModule(nn.Module):
    # [CLASS DOC] HELPER: Three layer chain linear, IgnoredModule, linear (all 100 wide).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates l1, the ignored submodule and l2.
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.ignored = IgnoredModule()
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies l1, then the ignored module, then l2.
        return self.l2(self.ignored(self.l1(x)))


class TestFSDPMixedPrecisionIgnoredModules(FSDPTest):
    # [CLASS DOC] Single rank test case for mixed precision combined with ignored modules.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns 1, the number of ranks for this test class.
        return 1

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_with_ignored_module(self):
        # [FUNCTION DOC] Needs 1 GPU. Wraps a model with float16 param mixed precision while
        # ignoring one submodule, which stays float32, and asserts that forward and backward
        # raise a RuntimeError saying the dtypes must be the same.
        model = ModelWithIgnoredModule().to(device_type)
        float16 = MixedPrecision(param_dtype=torch.float16)
        model = FSDP(
            model,
            ignored_modules=[model.ignored],
            mixed_precision=float16,
        )

        x = torch.ones(2, 100, device=device_type)

        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            model(x).sum().backward()


class TestFSDPDifferentSubmodulePrecision(FSDPTest):
    # [CLASS DOC] Two rank test case checking how forward inputs are cast when different FSDP
    # units use different mixed precision settings, using SaveForwardInputsModel to record the
    # input each submodule saw.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns 2, the number of ranks for this test class.
        return 2

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule(self):
        # [FUNCTION DOC] Needs 2 GPUs. Only child c2 is wrapped with float16 and input casting;
        # the root uses default precision. Asserts the recorded inputs are float32 for the
        # root model and c1 and float16 for c2.
        forward_inputs: dict[str, nn.Module] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)

        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).to(device_type)
        c1, c2 = model.c1, model.c2
        x = torch.zeros(2, 100, device=device_type)

        # float16 on one submodule and float32 on everything else
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model)

        fsdp(x).sum().backward()

        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule_skip_inputs(self):
        # [FUNCTION DOC] Needs 2 GPUs. Child c2 gets float16 mixed precision with
        # cast_forward_inputs off, while the model itself casts inputs. Asserts every recorded
        # input (root, c1, c2) stays float32.
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=False)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=True
        ).to(device_type)
        c1, c2 = model.c1, model.c2
        x = torch.zeros(2, 100, device=device_type)

        # float16 on one submodule and float32 on everything else
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model)

        fsdp(x).sum().backward()

        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float32)

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule_skip_inputs_error(self):
        # [FUNCTION DOC] Needs 2 GPUs. Same as the skip-inputs case but no one casts inputs, so
        # the float16 child receives float32 data. Asserts a RuntimeError about mismatched
        # matrix multiply dtypes is raised in forward or backward.
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=False)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).to(device_type)
        x = torch.zeros(2, 100, device=device_type)

        # float16 on one submodule and float32 on everything else
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model)

        with self.assertRaisesRegex(
            RuntimeError, "mat1 and mat2 must have the same dtype"
        ):
            fsdp(x).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_different_precisions_error(self):
        # [FUNCTION DOC] Needs 2 GPUs. Child c1 (the first module executed) uses float16 under
        # a float32 root; the root casts inputs to float32 up front so c1 gets the wrong dtype.
        # Asserts a RuntimeError about mismatched matrix multiply dtypes is raised.
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        float32 = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).to(device_type)
        x = torch.zeros(2, 100, device=device_type)

        # For submodules with different precisions, right now current design
        # does not support the case when the root FSDP instance wraps a submodule
        # that is not the first one executed. Because for that submodule, its inputs
        # (or previous submodule's outputs) have no way to be casted, instead,
        # the root module's inputs are casted upfront before entering
        # root module's forward
        model.c1 = FSDP(model.c1, mixed_precision=float16)
        fsdp = FSDP(model, mixed_precision=float32)
        with self.assertRaisesRegex(
            RuntimeError, "mat1 and mat2 must have the same dtype"
        ):
            fsdp(x).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_different_precisions(self):
        # [FUNCTION DOC] Needs 2 GPUs. Float32 root with float16 on child c2, both casting
        # inputs. Asserts recorded inputs are float32 for the root and c1 and float16 for c2.
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        float32 = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).to(device_type)
        c1, c2 = model.c1, model.c2
        x = torch.zeros(2, 100, device=device_type)

        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model, mixed_precision=float32)

        fsdp(x).sum().backward()

        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_external_inputs(self):
        # [FUNCTION DOC] Needs 2 GPUs. Both root and submodule l2 use float16 param mixed
        # precision. Asserts the root casts its input and the tensor flowing into l2 to
        # float16 but an extra float32 tensor created inside the root forward and passed to
        # l2 is not cast.
        class ToyModule(nn.Module):
            # [CLASS DOC] HELPER: Linear layer that records both of its forward inputs in a
            # shared dictionary.
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer l and keeps the shared
                # dictionary used to record inputs.
                super().__init__()
                self.l = nn.Linear(100, 100)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Records x and y in the dictionary under l2_input_x
                # and l2_input_y, then returns the linear layer applied to x only.
                self.forward_inputs["l2_input_x"] = x
                self.forward_inputs["l2_input_y"] = y
                return self.l(x)

        class ToyModel(nn.Module):
            # [CLASS DOC] HELPER: Linear layer followed by a ToyModule that receives an extra
            # float32 tensor created inside forward.
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Creates l1 and the ToyModule l2 and keeps the shared
                # dictionary used to record inputs.
                super().__init__()
                self.l1 = nn.Linear(100, 100)
                self.l2 = ToyModule(forward_inputs)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Records x as model_input_x, creates a float32 tensor
                # of ones on the test device and returns l2 applied to the l1 output and it.
                self.forward_inputs["model_input_x"] = x
                y = torch.ones(2, 100, device=device_type, dtype=torch.float32)
                return self.l2(self.l1(x), y)

        forward_inputs: dict[str, torch.Tensor] = {}

        float16 = MixedPrecision(param_dtype=torch.float16)
        model = ToyModel(forward_inputs).to(device_type)
        x = torch.zeros(2, 100, device=device_type, dtype=torch.float32)
        model.l2 = FSDP(model.l2, mixed_precision=float16)
        fsdp = FSDP(model, mixed_precision=float16)

        fsdp(x).sum().backward()

        # Inputs are casted in root module in default, inputs of submodules are not
        # explicitly casted, so the external inputs ``y`` of module ``self.l2`` is
        # not casted.
        self.assertEqual(forward_inputs["model_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_y"].dtype, torch.float32)


class TestFSDPTrainEval(FSDPTest):
    # [CLASS DOC] Two rank test case for switching between training and evaluation with
    # mixed precision and an EMA copy of the model.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns 2, the number of ranks for this test class.
        return 2

    @skip_if_lt_x_gpu(2)
    def test_train_ema_eval_flow(self):
        # [FUNCTION DOC] Needs 2 GPUs. Runs the train, EMA update and eval flow once for each
        # of SHARD_GRAD_OP, FULL_SHARD and NO_SHARD sharding strategies.
        """
        Tests a train -> EMA update -> eval flow with mixed precision enabled.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    # We mainly want to test `SHARD_GRAD_OP` since it surfaced
                    # the original bug of not using the right EMA parameters
                    # for eval, but we also test the others for completeness
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_train_ema_eval_flow,
        )

    def _test_train_ema_eval_flow(self, sharding_strategy: ShardingStrategy):
        # [FUNCTION DOC] HELPER: Wraps a transformer plus an EMA AveragedModel copy in FSDP
        # (transformer layers auto wrapped, float16 params, given sharding). Three times:
        # train step with Adam, update EMA params from the main copy under summon_full_params,
        # then eval forward without gradients. Asserts eval output sums differ across
        # iterations, showing eval uses the updated EMA parameters. Rank 0 prints the model.
        class TransformerWithEMA(nn.Module):
            # [CLASS DOC] HELPER: Holds a main transformer used in training and an EMA
            # averaged copy (with buffers) used in eval.
            def __init__(self, device: torch.device):
                # [FUNCTION DOC] HELPER: Creates the main nn.Transformer and an AveragedModel
                # around a second one using the EMA multi-average function with buffers.
                super().__init__()
                self.module = nn.Transformer(device=device)
                self.ema_module = AveragedModel(
                    nn.Transformer(device=device),
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(),
                    use_buffers=True,
                )

            def forward(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the main transformer when
                # in training mode and to the EMA copy otherwise, returning its output.
                # Use main copy for training and EMA copy for eval
                if self.training:
                    return self.module(*args, **kwargs)
                return self.ema_module(*args, **kwargs)

        device = torch.device(device_type)
        model = TransformerWithEMA(device=device)
        policy = ModuleWrapPolicy(
            {nn.Transformer, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
        )
        mixed_precision = MixedPrecision(param_dtype=torch.float16)
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
        )
        optim = torch.optim.Adam(fsdp_model.module.parameters(), lr=1e-2)
        if self.rank == 0:
            print(fsdp_model)
        torch.manual_seed(1 + self.rank)
        eval_src = torch.randn((8, 1, 512), device=device)
        eval_tgt = torch.randn((16, 1, 512), device=device)
        eval_out_sums: list[torch.Tensor] = []
        # An iteration consists of training forward/backward/optimizer,
        # updating the EMA copy with the main copy, and eval forward
        for _ in range(3):
            fsdp_model.train()
            train_src = torch.randn((8, 4, 512), device=device)
            train_tgt = torch.randn((16, 4, 512), device=device)
            train_out = fsdp_model(train_src, train_tgt)
            train_out.sum().backward()
            optim.step()
            optim.zero_grad()
            with FSDP.summon_full_params(fsdp_model):
                fsdp_model.ema_module.update_parameters(fsdp_model.module)
            fsdp_model.eval()
            with torch.no_grad():
                eval_out = fsdp_model(eval_src, eval_tgt)
            eval_out_sums.append(eval_out.sum())
        # Check that the eval outputs differ from iteration to iteration as a
        # proxy for eval using the correct EMA parameters
        for i in range(len(eval_out_sums) - 1):
            self.assertNotEqual(eval_out_sums[i], eval_out_sums[i + 1])
        self.assertNotEqual(eval_out_sums[0], eval_out_sums[-1])


if __name__ == "__main__":
    run_tests()
