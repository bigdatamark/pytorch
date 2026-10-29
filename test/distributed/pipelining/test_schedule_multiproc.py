# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process (one rank per accelerator) tests of the pipelining schedules.
# Provides helper functions to build models, stages and reference runs, then three test
# classes: ScheduleTest (forward only, eval, return outputs, multi-iteration, kwargs and
# gradients with tracer or manual stages, interleaved and flex attention, weight-update
# schedules, V-shaped schedules, callbacks, zero bubble, None-gradient handling),
# CustomSchedulesTest (non-symmetric stage ids, custom runtime schedules, native zero bubble)
# and PerDirectionScheduleTest (separate process groups per direction). Module constants:
# d_hid 512, batch_size 64, none-grad hidden size 32 and 8 microbatches. Import-time side
# effects: seeds torch with 0, resolves device_type/backend from the current accelerator and
# registers the parametrized tests. Tests need NCCL/XCCL and several accelerators.
import copy
import logging
import os
from dataclasses import dataclass

from model_registry import (
    ConditionalGradStack,
    FlexAttentionTransformer,
    ModelWithKwargs,
    MultiMLP,
    MultiMLPKwargs,
    MultiMLPWithDw,
)
from schedule_registry import (
    ScheduleUnbalanced,
    ScheduleVShaped,
    ScheduleWithReorderedB,
    ScheduleWithW,
)

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
from torch.distributed.pipelining import (
    _ScheduleForwardOnly,
    pipeline,
    PipelineStage,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
    FORWARD,
    OVERLAP_F_B,
)
from torch.distributed.pipelining.stage import _PipelineStageBase  # noqa: TC002
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.modules.loss import MSELoss
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
)


logger = logging.getLogger(__name__)

d_hid = 512
batch_size = 64
none_grad_d_hid = 32
none_grad_microbatches = 8
torch.manual_seed(0)
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)


@dataclass
class PipelineTestConfig:
    # [CLASS DOC] HELPER: Plain data holder passed to helpers: world_size, device (this rank's
    # torch device) and rank.
    world_size: int
    device: torch.device
    rank: int


def setup_models_and_data(
    config: PipelineTestConfig, n_layers=None, model_class=MultiMLP
):
    # [FUNCTION DOC] HELPER: Builds a model of the given class (default MultiMLP, hidden size
    # d_hid, n_layers defaulting to the world size) on the config device, and a deep copy as
    # reference. Creates a random input of shape batch_size by d_hid and a target equal to
    # the reference output plus noise, with a sum-reduced MSE loss. Returns the tuple
    # (model, reference model, x, target, loss function).
    """Setup models, input data, target data, and loss function."""
    if n_layers is None:
        n_layers = config.world_size

    full_mod = model_class(d_hid, n_layers=n_layers)
    full_mod.to(config.device)
    ref_mod = copy.deepcopy(full_mod)

    x = torch.randn(batch_size, d_hid, device=config.device)
    with torch.no_grad():
        y = ref_mod(x)
        target = y + torch.randn(batch_size, d_hid, device=config.device)

    loss_fn = torch.nn.MSELoss(reduction="sum")
    return full_mod, ref_mod, x, target, loss_fn


def create_single_stage_pipeline(
    config: PipelineTestConfig, mod, x, chunks, use_tracer=True
):
    # [FUNCTION DOC] HELPER: Builds one pipeline stage for this rank. With use_tracer, traces
    # the module with pipeline() on the first microbatch chunk of x (using the module's
    # split_spec if any) and builds the stage from the pipe. Otherwise treats layers.<rank>
    # as the stage module and wraps it in a PipelineStage. Returns (stage, stage module,
    # list holding that module).
    """Create a single-stage pipeline using either tracer or manual stage creation."""
    if use_tracer:
        x_mb = x.chunk(chunks)[0]
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(mod, mb_args=(x_mb,), split_spec=split_spec)
        stage = pipe.build_stage(config.rank, config.device)
        stage_module = pipe.get_stage_module(config.rank)
        return stage, stage_module, [stage_module]
    else:
        # Manual stage creation
        submod_name = f"layers.{config.rank}"
        stage_module = mod.get_submodule(submod_name)
        stage = PipelineStage(
            stage_module, config.rank, config.world_size, config.device
        )
        return stage, stage_module, [stage_module]


def create_multi_stage_pipeline(
    config: PipelineTestConfig, mod, stages_per_rank, n_stages, stage_indices=None
):
    # [FUNCTION DOC] HELPER: Builds PipelineStage objects for interleaved schedules. Default
    # stage indices for this rank are rank plus i times world size for i below
    # stages_per_rank. Each stage wraps submodule layers.<index> of mod with n_stages total.
    # Returns (stages, stage modules, submodule names).
    """Create multiple pipeline stages for interleaved schedules."""
    if stage_indices is None:
        stage_indices = [
            config.rank + i * config.world_size for i in range(stages_per_rank)
        ]

    submod_names = [f"layers.{i}" for i in stage_indices]
    stage_modules = [mod.get_submodule(submod_name) for submod_name in submod_names]
    stages = [
        PipelineStage(stage_module, stage_idx, n_stages, config.device)
        for stage_module, stage_idx in zip(stage_modules, stage_indices, strict=True)
    ]
    return stages, stage_modules, submod_names


def run_reference_model(ref_mod, x, target, loss_fn, num_iterations=2, **kwargs):
    # [FUNCTION DOC] HELPER: Runs the reference model for num_iterations: zero gradients,
    # forward (extra keyword arguments passed to the model), loss, backward. Gradients are
    # cleared each iteration, so they reflect only the last iteration. Returns the last
    # output and loss.
    """Run reference model for specified iterations and return final output and loss."""
    ref_out = None
    ref_loss = None

    for _ in range(num_iterations):
        ref_mod.zero_grad()
        ref_out = ref_mod(x, **kwargs)
        ref_loss = loss_fn(ref_out, target)
        ref_loss.backward()

    return ref_out, ref_loss


def check_gradients(
    config: PipelineTestConfig,
    stage_modules,
    ref_mod,
    submod_names=None,
    rtol=1e-5,
    atol=4e-5,
):
    # [FUNCTION DOC] HELPER: Compares parameter gradients of pipeline stage modules with the
    # reference model using assert_close with rtol/atol. With no submod_names, stage_modules
    # is a single module and the code detects whether its parameter names exist on the
    # reference root (tracer case) or are local to layers.<rank> (manual case). With
    # submod_names, each stage module is matched with the reference submodule of that name.
    # Raises AssertionError if exactly one gradient is None or values differ (printing both).
    # KNOWN ISSUE: the tolerance argument of the inner grad_check is never used.
    """Check that gradients match between pipeline stages and reference model using flexible comparison."""

    def grad_check(grad1, grad2, param_name, rtol, atol, tolerance=0.05):
        # [FUNCTION DOC] HELPER: Returns silently if both gradients are None, raises
        # AssertionError if only one is None, otherwise asserts they are close within
        # rtol/atol, printing both gradients before re-raising on mismatch.
        if grad1 is None and grad2 is None:
            return
        if grad1 is None or grad2 is None:
            raise AssertionError(
                f"One gradient is None for {param_name}: {grad1} vs {grad2}"
            )
        try:
            torch.testing.assert_close(grad1, grad2, rtol=rtol, atol=atol)
        except AssertionError:
            print(
                f"Numerical issues detected for {param_name}: param grad {grad1} vs ref grad {grad2}"
            )
            raise

    if submod_names is None:
        # Single stage case - need to detect tracer vs manual pipeline
        stage_modules = [stage_modules]

        # Try to detect if this is a tracer-based pipeline by checking if parameter exists in ref_mod
        sample_param_name = next(iter(stage_modules[0].named_parameters()))[0]
        try:
            # Try to get parameter directly from reference model (tracer-based)
            ref_mod.get_parameter(sample_param_name)
            is_tracer_based = True
        except AttributeError:
            # Parameter doesn't exist at root level, must be manual pipeline
            is_tracer_based = False

        if is_tracer_based:
            # Tracer-based pipeline: parameter names are full paths from root model
            for name, p in stage_modules[0].named_parameters():
                ref_p = ref_mod.get_parameter(name)
                grad_check(p.grad, ref_p.grad, name, rtol, atol)
        else:
            # Manual pipeline: parameter names are local to the submodule
            submod_name = f"layers.{config.rank}"
            ref_submod = ref_mod.get_submodule(submod_name)
            for name, p in stage_modules[0].named_parameters():
                ref_p = ref_submod.get_parameter(name)
                grad_check(p.grad, ref_p.grad, f"{submod_name}.{name}", rtol, atol)
    else:
        # Multi-stage case - always use submodule approach
        for stage_module, submod_name in zip(stage_modules, submod_names):
            ref_submod = ref_mod.get_submodule(submod_name)
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                grad_check(p.grad, ref_p.grad, f"{submod_name}.{name}", rtol, atol)


def zero_gradients(stage_modules):
    # [FUNCTION DOC] HELPER: Calls zero_grad on a stage module or on each module of a list.
    """Zero gradients for all stage modules."""
    if not isinstance(stage_modules, list):
        stage_modules = [stage_modules]
    for stage_module in stage_modules:
        stage_module.zero_grad()


def none_grad_loss(output, target):
    # [FUNCTION DOC] HELPER: Loss for the None-gradient tests: sum-reduced MSE between the
    # first element of the stage output and the target.
    return torch.nn.functional.mse_loss(output[0], target, reduction="sum")


def make_none_grad_flags(pattern: str, device: torch.device) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds a boolean tensor of shape batch_size by 1 made of one
    # block per microbatch (none_grad_microbatches blocks). Default: first block False and
    # the rest True; for the pattern first_true_then_false: first block True and the rest
    # False. Created on the given device.
    values = [False] + [True] * (none_grad_microbatches - 1)
    if pattern == "first_true_then_false":
        values = [True] + [False] * (none_grad_microbatches - 1)
    chunk = batch_size // none_grad_microbatches
    return torch.cat(
        [
            torch.full((chunk, 1), value, dtype=torch.bool, device=device)
            for value in values
        ]
    )


def setup_none_grad_model_and_data(config: PipelineTestConfig, n_layers: int):
    # [FUNCTION DOC] HELPER: Seeds torch with 0 and builds a ConditionalGradStack with n_layers
    # layers on the config device, a deep-copied reference, and random input and target of
    # shape batch_size by none_grad_d_hid. Returns (model, reference, x, target).
    torch.manual_seed(0)
    mod = ConditionalGradStack(none_grad_d_hid, n_layers).to(config.device)
    ref_mod = copy.deepcopy(mod)
    x = torch.randn(batch_size, none_grad_d_hid, device=config.device)
    target = torch.randn(batch_size, none_grad_d_hid, device=config.device)
    return mod, ref_mod, x, target


def run_none_grad_reference(ref_mod, x, flag, target) -> None:
    # [FUNCTION DOC] HELPER: Zeroes reference gradients, then splits x, flag and target into
    # none_grad_microbatches pieces (lengths must match) and accumulates backward of the
    # None-gradient loss for each piece. Leaves gradients on ref_mod; returns nothing.
    ref_mod.zero_grad()
    for x_mb, flag_mb, target_mb in zip(
        torch.tensor_split(x, none_grad_microbatches),
        torch.tensor_split(flag, none_grad_microbatches),
        torch.tensor_split(target, none_grad_microbatches),
        strict=True,
    ):
        none_grad_loss(ref_mod(x_mb, flag_mb), target_mb).backward()


def step_with_optional_pre_split(
    schedule,
    num_microbatches,
    args=(),
    kwargs=None,
    target=None,
    losses=None,
    return_outputs=True,
    pre_split=False,
):
    # [FUNCTION DOC] HELPER: Runs one schedule step either directly or with pre-split inputs.
    # Without pre_split, forwards args, kwargs, target, losses and return_outputs to
    # schedule.step. With pre_split, splits args and kwargs into num_microbatches chunks
    # (passed as arg_mbs / kwarg_mbs only when present) and splits target along the batch
    # into target_mbs. Returns the step result.
    kwargs = kwargs or {}
    if not pre_split:
        return schedule.step(
            *args,
            target=target,
            losses=losses,
            return_outputs=return_outputs,
            **kwargs,
        )

    step_kwargs = {
        "losses": losses,
        "return_outputs": return_outputs,
    }
    if args or kwargs:
        arg_mbs, kwarg_mbs = split_args_kwargs_into_chunks(
            args,
            kwargs,
            num_microbatches,
        )
        if args:
            step_kwargs["arg_mbs"] = arg_mbs
        if kwargs:
            step_kwargs["kwarg_mbs"] = kwarg_mbs
    if target is not None:
        step_kwargs["target_mbs"] = list(torch.tensor_split(target, num_microbatches))

    return schedule.step(**step_kwargs)


def create_packed_document_block_mask(
    positions: torch.Tensor, num_heads: int
) -> BlockMask:
    # [FUNCTION DOC] HELPER: Builds a flex-attention block mask for packed documents. A new
    # document starts wherever positions equals 0 (cumulative count gives document ids). A
    # query may attend to a key only if the key is not later and both share a document id.
    # Returns the BlockMask for the batch size and sequence length of positions with
    # num_heads heads, built on the positions device with dq options enabled.
    doc_ids = torch.cumsum((positions == 0).int(), dim=1) - 1

    def document_causal_mask(b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: Mask predicate: true when q_idx is at or after kv_idx and both
        # belong to the same document in batch element b (h is unused).
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    batch_size, seq_len = positions.shape
    return create_block_mask(
        document_causal_mask,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=positions.device,
        compute_dq_write_order=True,
        dq_kv_order=True,
    )


class ScheduleTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process continuous test class (world size 4) for pipeline schedules
    # against a reference model. Properties config and device describe this rank. The test
    # methods need NCCL/XCCL and enough accelerators.
    world_size = 4

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the module-level distributed backend name.
        # Testing with NCCL backend
        return backend

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns this rank's device (device type indexed by rank).
        return torch.device(device_type, self.rank)

    @property
    def config(self) -> PipelineTestConfig:
        # [FUNCTION DOC] HELPER: Builds a PipelineTestConfig from the world size, device and rank.
        """Lazily create and return the pipeline test configuration."""
        return PipelineTestConfig(
            world_size=self.world_size, device=self.device, rank=self.rank
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [_ScheduleForwardOnly])
    @skip_if_lt_x_gpu(4)
    def test_forward_only(self, ScheduleClass):
        # [FUNCTION DOC] Runs the forward-only schedule over a single-stage pipeline with 8
        # microbatches for 20 iterations. Rank 0 feeds x and receives the last rank's output
        # back into x, the last rank sends its output to rank 0. The last rank then applies
        # the reference model 20 times to a clone of the input and asserts equality with the
        # final pipelined output.
        mod, mod_ref, x, _, _ = setup_models_and_data(self.config)
        x_clone = x.clone()

        num_microbatches = 2 * self.world_size
        stage, _, _ = create_single_stage_pipeline(
            self.config, mod, x, num_microbatches
        )
        schedule = ScheduleClass(stage, num_microbatches, scale_grads=False)

        # Run forward-only schedule
        out = None
        num_iters = 20
        for _ in range(num_iters):
            if self.rank == 0:
                schedule.step(x)
                dist.recv(x, src=self.world_size - 1)
            elif self.rank == self.world_size - 1:
                out = schedule.step()
                dist.send(out, dst=0)
            else:
                schedule.step()

        # Validate pipelined output matches reference model
        if self.rank == self.world_size - 1:
            for _ in range(num_iters):
                x_clone = mod_ref(x_clone)
            torch.testing.assert_close(x_clone, out)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @skip_if_lt_x_gpu(4)
    def test_eval_inference_mode(self, ScheduleClass):
        # [FUNCTION DOC] Builds multi-stage (2 stages per rank) or single-stage pipelines
        # depending on the schedule class, runs the schedule in eval mode (rank 0 under no_grad)
        # and asserts that no parameter gradient was produced, and on the last rank that the
        # losses list is non-empty.
        num_microbatches = 4
        if ScheduleClass in [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ]:
            # Multi-stage schedules
            stages_per_rank = 2
            n_stages = stages_per_rank * self.world_size
            mod, _, x, target, loss_fn = setup_models_and_data(
                self.config, n_layers=n_stages
            )

            # Create multi-stage pipeline
            stages, stage_modules, _ = create_multi_stage_pipeline(
                self.config, mod, stages_per_rank, n_stages
            )
            schedule = ScheduleClass(
                stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )
        else:
            # Single-stage schedules
            mod, _, x, target, loss_fn = setup_models_and_data(self.config)

            # Create single-stage pipeline
            stage, stage_module, _ = create_single_stage_pipeline(
                self.config, mod, x, num_microbatches
            )
            stage_modules = [stage_module]
            schedule = ScheduleClass(
                stage, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )

        # Clear gradients and run eval
        zero_gradients(stage_modules)
        losses = []

        if self.rank == 0:
            # Support with and without no_grad()
            with torch.no_grad():
                schedule.eval(x)
        elif self.rank == self.world_size - 1:
            schedule.eval(target=target, losses=losses)
        else:
            schedule.eval()

        # Check that gradients were NOT computed during eval
        grad_computed_eval = any(
            param.grad is not None
            for stage_module in stage_modules
            for param in stage_module.parameters()
        )

        # Verify that gradients were not computed during eval
        self.assertFalse(
            grad_computed_eval, "Gradients should not be computed during eval()"
        )

        # Verify that losses are still computed during eval
        if self.rank == self.world_size - 1:
            self.assertTrue(len(losses) > 0, "Losses should be computed during eval()")

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @skip_if_lt_x_gpu(4)
    def test_return_output(self, ScheduleClass):
        # [FUNCTION DOC] Builds multi-stage (interleaved, looped BFS, interleaved zero
        # bubble) or single-stage pipelines with 4 microbatches, then steps the schedule with
        # return_outputs disabled (last rank passes target and losses; others pass the
        # input). Asserts on the last rank that the returned output is None.
        num_microbatches = 4
        if ScheduleClass in [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ]:
            # Multi-stage schedules
            stages_per_rank = 2
            n_stages = stages_per_rank * self.world_size
            mod, _, x, target, loss_fn = setup_models_and_data(
                self.config, n_layers=n_stages
            )

            # Create multi-stage pipeline
            stages, stage_modules, _ = create_multi_stage_pipeline(
                self.config, mod, stages_per_rank, n_stages
            )
            schedule = ScheduleClass(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
        else:
            # Single-stage schedules
            mod, _, x, target, loss_fn = setup_models_and_data(self.config)

            # Create single-stage pipeline
            stage, stage_module, _ = create_single_stage_pipeline(
                self.config, mod, x, num_microbatches
            )
            schedule = ScheduleClass(
                stage,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )

        losses = []

        if self.rank == self.world_size - 1:
            output = schedule.step(target=target, losses=losses, return_outputs=False)
        else:
            schedule.step(x)

        # Verify that output is None
        if self.rank == self.world_size - 1:
            self.assertTrue(output is None, "Output should be None")

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @skip_if_lt_x_gpu(4)
    def test_multi_iter(self, ScheduleClass):
        # [FUNCTION DOC] Builds a single-stage pipeline with 4 microbatches and steps the
        # GPipe or 1F1B schedule 20 times (rank 0 feeds x, the last rank passes target and a
        # losses list, the others step with no data), then waits at a barrier. Only checks
        # that it runs to completion.
        mod, _, x, target, loss_fn = setup_models_and_data(self.config)
        chunks = 4
        stage, _, _ = create_single_stage_pipeline(self.config, mod, x, chunks)
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run
        for _ in range(20):
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                losses = []
                schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier(device_ids=[self.rank])

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @parametrize("pre_split", [False, True])
    @skip_if_lt_x_gpu(4)
    def test_kwargs_with_tracer(self, ScheduleClass, pre_split):
        # [FUNCTION DOC] Parametrized over GPipe/1F1B and pre_split. Traces a ModelWithKwargs
        # with a keyword input y, builds this rank's stage and runs the schedule with 4
        # microbatches (rank 0 passes x and y, optionally pre-split into chunks). On the last
        # rank, compares the output (loose tolerances) and the summed microbatch losses with
        # the eager model and loss.
        mod = ModelWithKwargs(d_hid, splits=self.world_size)
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        chunks = 4
        x_mb = x.chunk(chunks)[0]
        y_mb = y.chunk(chunks)[0]

        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            mb_kwargs={"y": y_mb},
        )

        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run
        out = None
        losses = []
        if self.rank == 0:
            step_with_optional_pre_split(
                schedule,
                chunks,
                args=(x,),
                kwargs={"y": y},
                pre_split=pre_split,
            )
        elif self.rank == self.world_size - 1:
            out = step_with_optional_pre_split(
                schedule,
                chunks,
                target=target,
                losses=losses,
                pre_split=pre_split,
            )
        else:
            step_with_optional_pre_split(
                schedule,
                chunks,
                pre_split=pre_split,
            )

        dist.barrier(device_ids=[self.rank])

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x, y=y)
            ref_loss = loss_fn(ref_out, target)
            pipe_loss = sum(losses)
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=5e-3)
            torch.testing.assert_close(pipe_loss, ref_loss)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @parametrize("pre_split", [False, True])
    @skip_if_lt_x_gpu(4)
    def test_grad_with_tracer(self, ScheduleClass, pre_split):
        # [FUNCTION DOC] Parametrized over GPipe/1F1B and pre_split. Computes a reference run,
        # builds a traced single-stage pipeline with 2 times world size microbatches, runs two
        # iterations (zeroing stage gradients before each), then compares output and summed
        # loss on the last rank and per-parameter gradients on every rank via check_gradients.
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(self.config)

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create pipeline and schedule
        chunks = 2 * self.world_size
        stage, stage_module, stage_modules = create_single_stage_pipeline(
            self.config, mod, x, chunks
        )
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_module)
            if self.rank == 0:
                step_with_optional_pre_split(
                    schedule,
                    chunks,
                    args=(x,),
                    pre_split=pre_split,
                )
            elif self.rank == self.world_size - 1:
                out = step_with_optional_pre_split(
                    schedule,
                    chunks,
                    target=target,
                    losses=losses,
                    pre_split=pre_split,
                )
            else:
                step_with_optional_pre_split(
                    schedule,
                    chunks,
                    pre_split=pre_split,
                )

        dist.barrier(device_ids=[self.rank])

        # Last rank checks result
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_module, ref_mod)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @parametrize("shape_inference", [True, False])
    @skip_if_lt_x_gpu(4)
    def test_grad_with_manual(self, ScheduleClass, shape_inference):
        # [FUNCTION DOC] Parametrized over GPipe/1F1B and shape_inference. Builds a manual
        # stage; when shape_inference is false, runs the stage module on one microbatch to
        # create example input and output tensors (detached with requires_grad for non-first
        # ranks) and rebuilds the stage with explicit input_args and output_args. Runs two
        # iterations, compares output and loss on the last rank and gradients on all ranks.
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(self.config)

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create manual pipeline stage
        chunks = 2 * self.world_size
        stage, stage_module, _ = create_single_stage_pipeline(
            self.config, mod, x, chunks, use_tracer=False
        )

        # Handle shape inference
        if not shape_inference:
            input_args = (x.chunk(chunks)[0],)
            if self.rank > 0:
                # Non-first stages receive activations from previous stages,
                # which have requires_grad=True in training mode.
                input_args = tuple(a.detach().requires_grad_(True) for a in input_args)
            output_args = stage_module(*input_args)
            output_args = output_args.detach().requires_grad_(output_args.requires_grad)
            stage = PipelineStage(
                stage_module,
                self.rank,
                self.world_size,
                self.device,
                input_args=input_args,
                output_args=output_args,
            )

        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_module)
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier(device_ids=[self.rank])

        # Last rank checks result
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_module, ref_mod)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @parametrize("pre_split", [False, True])
    @skip_if_lt_x_gpu(4)
    def test_grad_with_manual_interleaved(self, ScheduleClass, pre_split):
        # [FUNCTION DOC] Parametrized over interleaved 1F1B, looped BFS and interleaved zero
        # bubble. Builds 2 stages per rank, uses the class num_microbatches if it has one
        # (else 2 times world size) and runs two iterations inside check_leaked_tensors,
        # asserting no tensors leaked. Compares output and loss on the last rank and gradients
        # on every stage with relaxed tolerances (5e-3).
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, stages_per_rank, n_stages
        )
        print(f"Rank {self.rank} stages: {[stage.stage_index for stage in stages]}")

        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 2 * self.world_size
        )

        # Create schedule
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline with tensor leak checking
        out = None
        losses = []
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(2):
                zero_gradients(stage_modules)
                if self.rank == 0:
                    step_with_optional_pre_split(
                        schedule,
                        num_microbatches,
                        args=(x,),
                        pre_split=pre_split,
                    )
                elif self.rank == self.world_size - 1:
                    out = step_with_optional_pre_split(
                        schedule,
                        num_microbatches,
                        target=target,
                        losses=losses,
                        pre_split=pre_split,
                    )
                else:
                    step_with_optional_pre_split(
                        schedule,
                        num_microbatches,
                        pre_split=pre_split,
                    )

        self.assertEqual(
            len(garbage_tensors),
            0,
            "Found leaked tensors, check logs above for debug info",
        )
        dist.barrier()

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients - use relaxed tolerances for interleaved schedules
        # since gradients are small
        check_gradients(
            self.config, stage_modules, ref_mod, submod_names, rtol=5e-3, atol=5e-3
        )

    @requires_accelerator_dist_backend(["nccl"])
    @skip_but_pass_in_sandcastle_if(
        device_type != "cuda" or not TEST_MULTIACCELERATOR,
        "CUDA/NCCL flex attention test requires 4+ GPUs",
    )
    @skip_if_lt_x_gpu(4)
    def test_interleaved_1f1b_pre_split_flex_attention(self):
        # [FUNCTION DOC] CUDA/NCCL only, 4+ GPUs. Builds a FlexAttentionTransformer split into
        # 2 stages per rank and two deep copies plus a reference, with packed-document block
        # masks. Runs interleaved 1F1B twice under deterministic mode: once passing the block
        # mask as a normal keyword (auto-split) and once with pre-split arg/kwarg/target
        # microbatches. Asserts the two runs agree on outputs, losses, and parameter gradients
        # and that both match the reference output, summed loss and gradients. First/last stage
        # presence must be identical across the two pipelines.
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        num_microbatches = 8
        batch = 8
        seq_len = 64
        model_dim = 32
        num_heads = 2
        ffn_dim = 64

        auto_mod = FlexAttentionTransformer(model_dim, num_heads, ffn_dim, n_stages).to(
            self.device
        )
        pre_split_mod = copy.deepcopy(auto_mod)
        ref_mod = copy.deepcopy(auto_mod)
        x = torch.randn(batch, seq_len, model_dim, device=self.device)
        target = torch.randn_like(x)
        positions = torch.stack(
            [
                torch.arange(seq_len, device=self.device) % (8 * (i % 4 + 1))
                for i in range(batch)
            ]
        )
        block_mask = create_packed_document_block_mask(positions, num_heads)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        auto_stages, auto_stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, auto_mod, stages_per_rank, n_stages
        )
        pre_split_stages, pre_split_stage_modules, pre_split_submod_names = (
            create_multi_stage_pipeline(
                self.config, pre_split_mod, stages_per_rank, n_stages
            )
        )
        self.assertEqual(pre_split_submod_names, submod_names)

        has_first_stage = any(stage.is_first for stage in auto_stages)
        has_last_stage = any(stage.is_last for stage in auto_stages)
        self.assertEqual(
            has_first_stage,
            any(stage.is_first for stage in pre_split_stages),
        )
        self.assertEqual(
            has_last_stage,
            any(stage.is_last for stage in pre_split_stages),
        )

        arg_mbs = [(x_mb,) for x_mb in torch.tensor_split(x, num_microbatches)]
        target_mbs = list(torch.tensor_split(target, num_microbatches))
        position_mbs = torch.tensor_split(positions, num_microbatches)
        kwarg_mbs = [
            {"block_mask": create_packed_document_block_mask(positions_mb, num_heads)}
            for positions_mb in position_mbs
        ]

        auto_schedule = ScheduleInterleaved1F1B(
            auto_stages,
            num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )
        pre_split_schedule = ScheduleInterleaved1F1B(
            pre_split_stages,
            num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )

        auto_out = None
        auto_losses = []
        pre_split_out = None
        pre_split_losses = []
        with DeterministicGuard(True):
            ref_out, ref_loss = run_reference_model(
                ref_mod,
                x,
                target,
                loss_fn,
                num_iterations=1,
                block_mask=block_mask,
            )

            zero_gradients(auto_stage_modules)
            auto_out = auto_schedule.step(
                *((x,) if has_first_stage else ()),
                block_mask=block_mask,
                target=target if has_last_stage else None,
                losses=auto_losses if has_last_stage else None,
            )

            dist.barrier()

            zero_gradients(pre_split_stage_modules)
            pre_split_out = pre_split_schedule.step(
                arg_mbs=arg_mbs if has_first_stage else None,
                kwarg_mbs=kwarg_mbs,
                target_mbs=target_mbs if has_last_stage else None,
                losses=pre_split_losses if has_last_stage else None,
            )

            dist.barrier()

        if has_last_stage:
            self.assertEqual(pre_split_out, auto_out)
            self.assertEqual(
                torch.stack(pre_split_losses),
                torch.stack(auto_losses),
            )
            self.assertEqual(auto_out, ref_out)
            self.assertEqual(sum(auto_losses), ref_loss)

        for auto_stage_module, pre_split_stage_module in zip(
            auto_stage_modules,
            pre_split_stage_modules,
            strict=True,
        ):
            for (auto_name, auto_param), (pre_split_name, pre_split_param) in zip(
                auto_stage_module.named_parameters(),
                pre_split_stage_module.named_parameters(),
                strict=True,
            ):
                self.assertEqual(auto_name, pre_split_name)
                self.assertEqual(pre_split_param.grad, auto_param.grad)

        check_gradients(self.config, auto_stage_modules, ref_mod, submod_names)
        check_gradients(self.config, pre_split_stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleInterleavedZeroBubble])
    @skip_if_lt_x_gpu(4)
    def test_schedule_with_weight_update_mlp_e2e(self, ScheduleClass):
        # [FUNCTION DOC] Uses MultiMLPWithDw models (toggled) with 2 stages per rank and a
        # mean MSE loss; computes a reference run, then rebuilds each stage with a dw_builder
        # from a small counting state class that calls the module's compute_dW. Runs the
        # interleaved zero bubble schedule (2 microbatches, default grad scaling) twice. On the
        # last rank compares output and the mean of the microbatch losses with the reference,
        # then checks gradients on all stages.
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        full_mod, ref_mod, x, target, _ = setup_models_and_data(
            self.config, n_layers=n_stages, model_class=MultiMLPWithDw
        )
        full_mod.toggle()
        loss_fn = MSELoss()

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom dw_builder
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, full_mod, stages_per_rank, n_stages
        )

        class CustomState:
            # [CLASS DOC] HELPER: Per-stage state for the weight-gradient builder: call counter
            # i plus the stage module, stage index and rank (used for logging and compute_dW).
            def __init__(self, stage_module, stage_idx, rank):
                # [FUNCTION DOC] HELPER: Stores the stage module, stage index and rank, and
                # starts the dw call counter i at 0.
                self.i = 0
                self.stage_module = stage_module
                self.stage_idx = stage_idx
                self.rank = rank

            def dw_builder(self):
                # [FUNCTION DOC] HELPER: Returns a dw_runner closure for the stage.
                def dw_runner():
                    # [FUNCTION DOC] HELPER: Increments the call counter, prints rank, count
                    # and stage index, then calls compute_dW on the stage module.
                    self.i += 1
                    print(
                        f"[Rank {self.rank}] dw_count={self.i} stage={self.stage_idx}"
                    )
                    self.stage_module.compute_dW()

                return dw_runner

        # Create custom states and rebuild stages with dw_builder
        cs = {}
        stage_indices = [
            self.rank + i * self.world_size for i in range(stages_per_rank)
        ]
        for stage_module, stage_idx in zip(stage_modules, stage_indices):
            cs[stage_idx] = CustomState(stage_module, stage_idx, self.rank)

        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
                dw_builder=cs[stage_idx].dw_builder,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

        schedule = ScheduleClass(stages, 2, loss_fn=loss_fn)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier(device_ids=[self.rank])

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses) / len(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "schedule_class",
        [ScheduleZBVZeroBubble, ScheduleDualPipeV],
    )
    @skip_if_lt_x_gpu(4)
    def test_v_shape_schedules(self, schedule_class):
        # [FUNCTION DOC] Parametrized over ScheduleZBVZeroBubble and ScheduleDualPipeV. Uses 8
        # stages in a V layout (rank r owns stages r and 7-r) with 8 microbatches. Rank 0 owns
        # first and last stages so it passes input, target and losses, others step with
        # nothing. After two iterations rank 0 compares output and summed loss with the
        # reference and all ranks check gradients.
        n_stages = 8
        rank_stages = {0: [0, 7], 1: [1, 6], 2: [2, 5], 3: [3, 4]}
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom stage indices
        num_microbatches = 8
        stage_indices = rank_stages[self.rank]
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        schedule = schedule_class(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline - special case where first and last stage are on rank 0
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                out = schedule.step(x, target=target, losses=losses)
            else:
                schedule.step()

        # Verify results (rank 0 has both first and last stages)
        if self.rank == 0:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @skip_if_lt_x_gpu(4)
    def test_custom_function_callback(self):
        # [FUNCTION DOC] Builds a DualPipeV schedule on a V layout (8 stages, 8 microbatches),
        # prepares its comms schedule (private method) and registers custom callbacks for the
        # FORWARD and OVERLAP_F_B actions that re-implement the runtime behavior (wait for
        # receives, run forward/backward chunks, compute loss, scale grads on the last
        # backward, hand off locally between adjacent stages on the same rank). Runs two
        # loops, then on rank 0 checks output, loss, that the overlap callback ran, that the
        # forward callback count equals microbatches * 2 * loops, and that all calls were for
        # local stages; finally checks gradients on all ranks.
        """Test the custom function callback functionality with _PipelineScheduleRuntime."""
        n_stages = 8
        rank_stages = {0: [0, 7], 1: [1, 6], 2: [2, 5], 3: [3, 4]}
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom stage indices
        num_microbatches = 8
        stage_indices = rank_stages[self.rank]
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        # Use DualPipeV schedule as the base schedule
        base_schedule = ScheduleDualPipeV(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )
        base_schedule._prepare_schedule_with_comms(base_schedule.pipeline_order)

        # Track both types of callbacks separately
        forward_calls = []
        overlap_calls = []

        def forward_callback(action: _Action, ctx: _PipelineContext):
            # [FUNCTION DOC] HELPER: Callback for FORWARD actions on the runtime schedule.
            # Looks up the action's stage and microbatch, records the pair in forward_calls,
            # waits for the pending receive unless the previous stage is local, runs
            # forward_one_chunk, computes the loss, and passes the output directly to the next
            # stage if it is on this rank. Raises AssertionError if the schedule is not a
            # _PipelineScheduleRuntime, the microbatch index is None, or no receive was queued.
            # KNOWN ISSUE: re-imports _wait_batch_p2p locally though it is already imported
            # at module level.
            """Custom callback for FORWARD computation that mimics the original implementation."""
            schedule = ctx.schedule_ref
            if not isinstance(schedule, _PipelineScheduleRuntime):
                raise AssertionError(
                    f"Expected _PipelineScheduleRuntime, got {type(schedule)}"
                )
            stage_index_to_stage: dict[int, _PipelineStageBase] = {
                stage.stage_index: stage for stage in schedule._stages
            }
            stage = stage_index_to_stage[action.stage_index]
            stage_index = stage.stage_index
            mb_index = action.microbatch_index
            if mb_index is None:
                raise AssertionError("Expected mb_index to not be None")
            fwd_recv_ops = schedule.fwd_recv_ops
            arg_mbs = ctx.arg_mbs
            kwarg_mbs = ctx.kwarg_mbs

            is_next_stage_on_this_rank = stage_index + 1 in stage_index_to_stage
            is_prev_stage_on_this_rank = stage_index - 1 in stage_index_to_stage

            # used in verification at the end
            forward_calls.append((stage_index, mb_index))

            if (
                not stage.is_first
                # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                and not is_prev_stage_on_this_rank
            ):
                if (stage_index, mb_index) not in fwd_recv_ops:
                    raise AssertionError(f"Computing {action=} before receiving input")
                from torch.distributed.pipelining.schedules import _wait_batch_p2p

                _wait_batch_p2p(fwd_recv_ops.pop((stage_index, mb_index)))

            output = stage.forward_one_chunk(
                mb_index,
                arg_mbs[mb_index],  # type: ignore[index]
                kwarg_mbs[mb_index],  # type: ignore[index]
            )
            schedule._maybe_compute_loss(stage, output, ctx.target_mbs, mb_index)

            # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
            # see [Note: V-schedule special case]
            if is_next_stage_on_this_rank:
                stage_index_to_stage[stage_index + 1].set_local_fwd_input(
                    output, mb_index
                )

        def overlap_callback(action: _Action, ctx: _PipelineContext):
            # [FUNCTION DOC] HELPER: Callback for OVERLAP_F_B actions. Runs the forward
            # sub-action through forward_callback and records the (forward stage, mb,
            # backward stage, mb) tuple in overlap_calls. Then for the backward sub-action
            # waits for the pending receive unless the next stage is local, runs
            # backward_one_chunk with the loss (scaling grads on the last backward) and passes
            # the gradient to the previous stage locally if it is on this rank. Raises
            # AssertionError on a wrong schedule type, missing sub_actions, missing microbatch
            # index or a missing queued receive.
            """Custom callback for OVERLAP_F_B computation that mimics the original implementation."""
            schedule = ctx.schedule_ref
            if not isinstance(schedule, _PipelineScheduleRuntime):
                raise AssertionError(
                    f"Expected _PipelineScheduleRuntime, got {type(schedule)}"
                )
            stage_index_to_stage: dict[int, _PipelineStageBase] = {
                stage.stage_index: stage for stage in schedule._stages
            }
            if action.sub_actions is None:
                raise AssertionError("Expected action.sub_actions to not be None")
            fwd_action = action.sub_actions[0]
            bwd_action = action.sub_actions[1]

            # Forward ========================================================
            forward_callback(fwd_action, ctx)
            overlap_calls.append(
                (
                    fwd_action.stage_index,
                    fwd_action.microbatch_index,
                    bwd_action.stage_index,
                    bwd_action.microbatch_index,
                )
            )

            # Backward ========================================================
            backward_stage_index = bwd_action.stage_index
            backward_stage = stage_index_to_stage[backward_stage_index]
            backward_mb_index = bwd_action.microbatch_index
            if backward_mb_index is None:
                raise AssertionError("Expected backward_mb_index to not be None")
            bwd_recv_ops = schedule.bwd_recv_ops
            is_next_stage_on_this_rank = (
                backward_stage.stage_index + 1 in stage_index_to_stage
            )
            is_prev_stage_on_this_rank = (
                backward_stage.stage_index - 1 in stage_index_to_stage
            )
            if (
                not backward_stage.is_last
                # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                and not is_next_stage_on_this_rank
            ):
                if (backward_stage_index, backward_mb_index) not in bwd_recv_ops:
                    raise AssertionError(
                        f"Attempted to run compute {action=} before receiving input"
                    )
                _wait_batch_p2p(
                    bwd_recv_ops.pop((backward_stage_index, backward_mb_index))
                )
            loss = schedule._maybe_get_loss(backward_stage, backward_mb_index)
            schedule.backward_counter[backward_stage_index] += 1
            last_backward = (
                schedule.backward_counter[backward_stage_index]
                == schedule._n_microbatches
            )
            grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1
            backward_stage.backward_one_chunk(
                backward_mb_index,
                loss=loss,
                full_backward=True,
                last_backward=last_backward,
            )
            if last_backward:
                backward_stage.scale_grads(grad_scale_factor)
            # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
            # see [Note: V-schedule special case]
            if is_prev_stage_on_this_rank:
                stage_index_to_stage[backward_stage_index - 1].set_local_bwd_input(
                    backward_stage.get_local_bwd_output(backward_mb_index),
                    backward_mb_index,
                )

        # Add the callback for FORWARD computation type

        base_schedule.register_custom_function(FORWARD, forward_callback)
        base_schedule.register_custom_function(OVERLAP_F_B, overlap_callback)

        # Run pipeline - special case where first and last stage are on rank 0
        out = None
        losses = []
        num_loops = 2
        for _ in range(num_loops):
            zero_gradients(stage_modules)
            if self.rank == 0:
                out = base_schedule.step(x, target=target, losses=losses)
            else:
                base_schedule.step()

        dist.barrier()

        # Verify results (rank 0 has both first and last stages)
        if self.rank == 0:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

            # Verify overlap callbacks were called
            self.assertGreater(
                len(overlap_calls), 0, "OVERLAP_F_B callback should have been called"
            )

            # In a V-schedule with 8 microbatches and 2 stages per rank,
            # rank 0 should have 32 calls (8 microbatches * 2 stages * 2 loops)
            expected_count = num_microbatches * 2 * num_loops
            self.assertEqual(len(forward_calls), expected_count)

            # Verify all callback calls are for stages on this rank
            for stage_idx, _ in forward_calls:
                self.assertIn(
                    stage_idx,
                    stage_indices,
                    lambda msg: f"{msg}\nCallback called for stage {stage_idx} not on rank {self.rank}",
                )

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, "NCCL test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [ScheduleInterleavedZeroBubble, ScheduleInterleaved1F1B],
    )
    @skip_if_lt_x_gpu(4)
    def test_zero_bubble_with_model_kwargs(self, ScheduleClass):
        # [FUNCTION DOC] Parametrized over interleaved zero bubble and interleaved 1F1B with
        # MultiMLPKwargs models whose forward takes an extra keyword that is unused. Rank 0
        # passes a per-microbatch expanded copy of the keyword while the reference uses a
        # single tensor. Compares output and summed loss on the last rank and gradients (rtol
        # 3e-5, atol 5e-3) on all ranks.
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages, model_class=MultiMLPKwargs
        )
        unused_kwarg = torch.tensor([1.0], device=self.device)

        # Run reference with kwargs
        ref_out, ref_loss = run_reference_model(
            ref_mod, x, target, loss_fn, unused_kwarg=unused_kwarg
        )

        # Create multi-stage pipeline
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, stages_per_rank, n_stages
        )

        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 2 * self.world_size
        )
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline with kwargs
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                schedule.step(
                    x,
                    unused_kwarg=unused_kwarg.clone()
                    .unsqueeze(0)
                    .expand(num_microbatches, -1),
                )
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(
            self.config, stage_modules, ref_mod, submod_names, rtol=3e-5, atol=5e-3
        )

    def _none_grad_stage_indices(self, ScheduleClass):
        # [FUNCTION DOC] HELPER: Returns (number of stages, stage indices owned by this rank)
        # for a schedule class: GPipe has world_size stages with one per rank, ZBV uses a V
        # layout (rank and its mirror), and other classes use looped placement (rank and rank
        # plus world_size).
        if ScheduleClass is ScheduleGPipe:
            return self.world_size, [self.rank]
        n_stages = 2 * self.world_size
        if ScheduleClass is ScheduleZBVZeroBubble:
            return n_stages, [self.rank, n_stages - 1 - self.rank]
        return n_stages, [self.rank, self.rank + self.world_size]

    def _run_none_grad_schedule(self, ScheduleClass, pattern):
        # [FUNCTION DOC] HELPER: Runs one None-gradient scenario: builds the ConditionalGradStack
        # model and data, computes the microbatched reference with the chosen flag pattern,
        # builds this rank's stages and the schedule (single stage for GPipe) with the
        # None-grad loss and 8 microbatches. Steps once with arguments depending on whether the
        # rank has the first stage, last stage, both or neither, waits at a barrier and checks
        # gradients with tolerance 1e-5.
        n_stages, stage_indices = self._none_grad_stage_indices(ScheduleClass)
        mod, ref_mod, x, target = setup_none_grad_model_and_data(self.config, n_stages)
        flag = make_none_grad_flags(pattern, self.device)
        run_none_grad_reference(ref_mod, x, flag, target)

        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )
        schedule = (
            ScheduleClass(
                stages[0],
                none_grad_microbatches,
                loss_fn=none_grad_loss,
                scale_grads=False,
            )
            if ScheduleClass is ScheduleGPipe
            else ScheduleClass(
                stages,
                none_grad_microbatches,
                loss_fn=none_grad_loss,
                scale_grads=False,
            )
        )

        losses = []
        has_first = any(stage.is_first for stage in stages)
        has_last = any(stage.is_last for stage in stages)
        if has_first and has_last:
            schedule.step(x, flag, target=target, losses=losses)
        elif has_first:
            schedule.step(x, flag)
        elif has_last:
            schedule.step(target=target, losses=losses)
        else:
            schedule.step()

        dist.barrier()
        check_gradients(
            self.config, stage_modules, ref_mod, submod_names, rtol=1e-5, atol=1e-5
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [ScheduleGPipe, ScheduleInterleaved1F1B, ScheduleZBVZeroBubble],
    )
    @parametrize("pattern", ["first_false_then_true", "first_true_then_false"])
    @skip_if_lt_x_gpu(4)
    def test_NoneGrad_conditional_input_grad(self, ScheduleClass, pattern):
        # [FUNCTION DOC] Parametrized over GPipe, interleaved 1F1B and ZBV zero bubble and two
        # flag patterns; runs the None-gradient scenario helper to check that conditionally
        # missing input gradients are handled and match the reference.
        self._run_none_grad_schedule(ScheduleClass, pattern)


instantiate_parametrized_tests(ScheduleTest)


class CustomSchedulesTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test class (world size 2) for unusual custom schedules from
    # the schedule registry: non-symmetric stage placement, a runtime schedule with reordered
    # backward actions, and a schedule with native weight-gradient actions. Provides device
    # and config properties for the current rank.
    """
    These schedules are from the ScheduleRegistry and require world_size == 2
    The schedules test weird and unconventional schedules for edge cases
    """

    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the module-level distributed backend name.
        # Testing with NCCL backend
        return backend

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns this rank's device (device type indexed by rank).
        return torch.device(device_type, self.rank)

    @property
    def config(self) -> PipelineTestConfig:
        # [FUNCTION DOC] HELPER: Builds a PipelineTestConfig from world size, device and rank.
        """Lazily create and return the pipeline test configuration."""
        return PipelineTestConfig(
            world_size=self.world_size, device=self.device, rank=self.rank
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "schedule_class",
        [ScheduleVShaped, ScheduleUnbalanced],
    )
    @skip_if_lt_x_gpu(4)
    def test_non_symmetric_stage_ids(self, schedule_class):
        # [FUNCTION DOC] Parametrized over ScheduleVShaped and ScheduleUnbalanced. Uses the
        # schedule class's n_stages and rank_stages to assign (non-symmetric) stage indices to
        # ranks, with 1 microbatch. Rank 0 owns first and last stage, passing input, target and
        # losses; others step with nothing. After two iterations rank 0 compares output and
        # summed loss with the reference, and all ranks check gradients.
        n_stages = schedule_class.n_stages
        rank_stages = schedule_class.rank_stages

        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom stage indices
        num_microbatches = 1
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        schedule = schedule_class(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline - special case where first and last stage are on rank 0
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                out = schedule.step(x, target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Verify results (rank 0 has both first and last stages)
        if self.rank == 0:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleWithReorderedB])
    @skip_if_lt_x_gpu(4)
    def test_pipeline_schedule_runtime_custom_sched(self, ScheduleClass):
        # [FUNCTION DOC] Parametrized over ScheduleWithReorderedB. Builds 2 stages (1 per rank),
        # uses the class num_microbatches or 8, and asserts the schedule is a
        # _PipelineScheduleRuntime. Runs two iterations inside check_leaked_tensors and asserts
        # no leaked tensors, then compares output and summed loss on the last rank and
        # gradients on all ranks.
        n_stages = 2
        stages_per_rank = 1
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create pipeline stages
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, stages_per_rank, n_stages
        )
        print(f"Rank {self.rank} stages: {[stage.stage_index for stage in stages]}")

        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 8
        )

        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )
        if not isinstance(schedule, _PipelineScheduleRuntime):
            raise AssertionError(
                f"Expected _PipelineScheduleRuntime, got {type(schedule)}"
            )

        # Run pipeline with tensor leak checking
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(2):
                zero_gradients(stage_modules)
                if self.rank == 0:
                    schedule.step(x)
                elif self.rank == self.world_size - 1:
                    losses = []
                    out = schedule.step(target=target, losses=losses)
                else:
                    schedule.step()

        self.assertEqual(
            len(garbage_tensors),
            0,
            "Found leaked tensors, check logs above for debug info",
        )
        dist.barrier()

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleWithW])
    @skip_if_lt_x_gpu(4)
    def test_schedule_with_native_zero_bubble(self, ScheduleClass):
        # [FUNCTION DOC] Parametrized over ScheduleWithW. Takes stage count, microbatch count
        # and stage placement from the schedule class, builds the stages and schedule, runs the
        # reference model 4 times without zeroing gradients (so gradients accumulate over 4
        # steps), then steps the pipeline 4 times inside check_leaked_tensors (no zeroing
        # either). Asserts no leaked tensors and that gradients match the reference. Output
        # and loss are not compared.
        n_stages = ScheduleClass.n_stages
        num_microbatches = ScheduleClass.num_microbatches
        rank_stages = ScheduleClass.rank_stages

        num_steps = 4
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Create multi-stage pipeline with custom stage indices
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run reference model
        ref_x = x.detach().clone().requires_grad_(x.requires_grad)
        torch.testing.assert_close(x, ref_x)
        for _ in range(num_steps):
            ref_out = ref_mod(ref_x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Run pipeline with tensor leak checking
        losses = []
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(num_steps):
                if self.rank == 0:
                    schedule.step(x)
                elif self.rank == self.world_size - 1:
                    schedule.step(target=target, losses=losses)
                else:
                    schedule.step()

        self.assertEqual(
            len(garbage_tensors),
            0,
            "Found leaked tensors, check logs above for debug info",
        )

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)


instantiate_parametrized_tests(CustomSchedulesTest)


class PerDirectionScheduleTest(MultiProcContinuousTest):
    # [CLASS DOC] Tests per-direction P2P communicators (config flag pipeline_per_direction_p2p)
    # with 4 ranks. It overrides process group initialization to bind a device id so that
    # PipelineStage can split the default group into downstream and upstream subgroups.
    """Per-direction PP communicators (``config.pipeline_per_direction_p2p``).

    A single PP communicator serializes all send/recv in one FIFO; splitting
    downstream (r -> r+1 activations) and upstream (r -> r-1 gradients) onto two
    communicators removes the cross-batch deadlock hazard. PipelineStage builds
    the two subgroups with ``split_group``, which requires a device-bound default
    PG -- so this class binds ``device_id`` in ``_init_pg`` (see below) rather
    than relying on the shared MultiProcContinuousTest path.
    """

    world_size = 4

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the module-level distributed backend name.
        return backend

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Class-level process group initialization for each rank.
        # Raises AssertionError if rdvz_file is None. Sets LOCAL_RANK in the environment,
        # opens a FileStore at rdvz_file, binds a device id (device type, rank) only when
        # enough accelerators exist, initializes the default process group with the class
        # backend, options and timeout, and stores the default group in the class attribute pg.
        # Bind device_id so PipelineStage can split the default PG into
        # downstream/upstream subgroups. We override here instead of changing
        # MultiProcContinuousTest._init_pg, which also serves CPU/Gloo callers
        # where device_id would alter PG initialization semantics.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        store = dist.FileStore(rdvz_file, world_size)
        # _init_pg runs at class spawn, before the per-test skip_if_lt_x_gpu(4).
        # Only bind device_id when every rank maps to a real accelerator;
        # otherwise the tests are skipped anyway and an out-of-range device_id
        # would make init_process_group raise instead of skip.
        device_id = None
        if torch.accelerator.device_count() >= world_size:
            device_id = torch.device(cls.device_type(), rank)
        dist.init_process_group(
            backend=cls.backend_str(),
            world_size=world_size,
            rank=rank,
            store=store,
            pg_options=cls.opts(),
            timeout=cls.timeout,
            device_id=device_id,
        )
        cls.pg = dist.distributed_c10d._get_default_group()

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns this rank's device (device type indexed by rank).
        return torch.device(device_type, self.rank)

    @property
    def config(self) -> PipelineTestConfig:
        # [FUNCTION DOC] HELPER: Builds a PipelineTestConfig from world size, device and rank.
        return PipelineTestConfig(
            world_size=self.world_size, device=self.device, rank=self.rank
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @skip_if_lt_x_gpu(4)
    def test_creates_distinct_direction_groups(self):
        # [FUNCTION DOC] With the per-direction config flag patched on, builds a manual stage
        # and asserts it reports per-direction P2P and that its downstream and upstream groups
        # are neither the WORLD group nor the same group.
        """PipelineStage builds two distinct, non-WORLD direction groups when the
        config flag is set (no constructor arg)."""
        with dist_config.patch(pipeline_per_direction_p2p=True):
            mod, _, x, _, _ = setup_models_and_data(self.config)
            chunks = 2 * self.world_size
            stage, _, _ = create_single_stage_pipeline(
                self.config, mod, x, chunks, use_tracer=False
            )
            self.assertTrue(stage.p2p_per_direction)
            self.assertIsNot(stage._downstream_group, dist.group.WORLD)
            self.assertIsNot(stage._upstream_group, dist.group.WORLD)
            self.assertIsNot(stage._downstream_group, stage._upstream_group)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @skip_if_lt_x_gpu(4)
    def test_grad_with_manual_per_direction(self, ScheduleClass):
        # [FUNCTION DOC] Parametrized over GPipe and 1F1B with the per-direction flag patched
        # on. Builds a manual stage (asserts distinct direction groups), runs two iterations
        # with 2 times world size microbatches, then compares output and summed loss on the
        # last rank and gradients on all ranks with the reference model.
        """Per-direction P2P only changes which communicator carries the bytes,
        not the math: gradients/outputs must still match the reference model."""
        with dist_config.patch(pipeline_per_direction_p2p=True):
            mod, ref_mod, x, target, loss_fn = setup_models_and_data(self.config)

            # Run reference
            ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

            # Create manual pipeline stage; the per-direction groups are built
            # inside PipelineStage from the config flag set above.
            chunks = 2 * self.world_size
            stage, stage_module, _ = create_single_stage_pipeline(
                self.config, mod, x, chunks, use_tracer=False
            )
            self.assertTrue(stage.p2p_per_direction)
            self.assertIsNot(stage._downstream_group, stage._upstream_group)

            schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

            # Run pipeline
            out = None
            losses = []
            for _ in range(2):
                zero_gradients(stage_module)
                if self.rank == 0:
                    schedule.step(x)
                elif self.rank == self.world_size - 1:
                    out = schedule.step(target=target, losses=losses)
                else:
                    schedule.step()

            dist.barrier(device_ids=[self.rank])

            # Last rank checks result
            if self.rank == self.world_size - 1:
                torch.testing.assert_close(out, ref_out)
                pipe_loss = sum(losses)
                torch.testing.assert_close(pipe_loss, ref_loss)

            # Check gradients using helper method
            check_gradients(self.config, stage_module, ref_mod)


instantiate_parametrized_tests(PerDirectionScheduleTest)


if __name__ == "__main__":
    run_tests()
