# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Four-GPU NCCL tests that compose pipeline parallelism with data parallelism
# (DDP or FSDP2 fully_shard, optionally with bfloat16 mixed precision) on a 2D device mesh
# named dp and pp, comparing gradients from pipelined runs to a single-model reference. Also
# tests FSDP unshard/reshard actions via the pipeline schedule runtime. Defines two small MLP
# modules, a scaled cross-entropy loss helper, and the ComposabilityTest case; parametrized
# tests are instantiated at import time.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineScheduleRuntime,
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
)
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
)


device_type = "cuda"


# MLP Layer
class MLPModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Two-layer MLP (linear, relu, linear) with hidden size d_hid for
    # both input and output; submodules net1, relu, net2. Weights get Kaiming uniform init.
    def __init__(self, d_hid: int):
        # [FUNCTION DOC] HELPER: Creates the two linear layers (d_hid to d_hid) and the relu,
        # then initializes the weights.
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)
        self.init_weights()

    def init_weights(self):
        # [FUNCTION DOC] HELPER: Re-initializes both linear weights in place with Kaiming
        # uniform (fan in, relu gain); biases are untouched.
        # ensure a proper init otherwise gradient tests will be more likely to get zero grad values
        torch.nn.init.kaiming_uniform_(
            self.net1.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_uniform_(
            self.net2.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies net1, relu, then net2 and returns the result.
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MLPModuleEven(torch.nn.Module):
    # [CLASS DOC] HELPER: Three-layer MLP with relu after every layer; the last layer widens
    # the output to twice d_hid. Weights get Kaiming uniform init. Not used by tests in
    # this file.
    def __init__(self, d_hid: int):
        # [FUNCTION DOC] HELPER: Creates three linear layers (d_hid, d_hid, then d_hid to
        # twice d_hid) and initializes their weights.
        super().__init__()
        self.net1 = nn.Linear(d_hid, d_hid)
        self.net2 = nn.Linear(d_hid, d_hid)
        self.net3 = nn.Linear(d_hid, d_hid * 2)
        self.init_weights()

    def init_weights(self):
        # [FUNCTION DOC] HELPER: Re-initializes all three linear weights with Kaiming
        # uniform (fan in, relu gain); biases are untouched.
        torch.nn.init.kaiming_uniform_(
            self.net1.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_uniform_(
            self.net2.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_uniform_(
            self.net3.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies each of the three linear layers followed by relu
        # and returns the result (last dimension twice d_hid).
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x


def loss_fn(y, target, scale=1e-4):
    # [FUNCTION DOC] HELPER: Cross entropy between predictions y and target, multiplied by
    # scale (default 1e-4) to mimic a small learning rate and keep gradients from blowing
    # up. Returns the scaled scalar loss.
    # Scale the loss to simulate a small learning rate and avoid exploding grads
    return torch.nn.functional.cross_entropy(y, target) * scale


class ComposabilityTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process NCCL test case (one process per GPU) for pipeline parallel
    # combined with DDP or FSDP2. Helper methods build random microbatches, pipeline stages
    # and schedules; tests compare pipeline gradients to a locally computed reference.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Selects the nccl backend for the test process group.
        # Testing with NCCL backend
        return "nccl"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property giving the cuda device indexed by this rank.
        return torch.device(device_type, self.rank)

    def _rand_microbatches(self, dp_mesh, num_microbatches, dim, dtype=torch.float32):
        # [FUNCTION DOC] HELPER: Draws one random tensor of shape (num_microbatches, dim) per
        # data-parallel rank (dp mesh size) on this device. Returns the list of all of them,
        # the one belonging to this rank's local dp rank, and a list of single-row
        # microbatches, each wrapped in a one-element list, taken from the local tensor.
        full = [
            torch.rand((num_microbatches, dim), device=self.device, dtype=dtype)
            for _ in range(dp_mesh.size())
        ]
        local = full[dp_mesh.get_local_rank()]
        local_mb = [[local[i].reshape((1, dim))] for i in range(num_microbatches)]
        return full, local, local_mb

    # build a pipeline stage
    def _build_pp_stage(
        self, pp_group, full_model, total_layers, apply_dp, stage_idx, num_stages
    ):
        # [FUNCTION DOC] HELPER: Builds one pipeline stage. Slices a contiguous equal share
        # of total_layers from full_model for stage_idx out of num_stages, moves it to this
        # device, wraps it with the apply_dp callable, and creates a PipelineStage on
        # pp_group. Returns the stage and the layer offset of its first layer. Raises
        # AssertionError if layers do not divide evenly by the stage count.
        # divide the model (e.g. 8 layers) by the number of stages
        layers_per_stage = total_layers // num_stages
        if layers_per_stage * num_stages != total_layers:
            raise AssertionError(
                f"layers_per_stage * num_stages != total_layers: "
                f"{layers_per_stage * num_stages} vs {total_layers}"
            )
        # return offset so validation code can match partial layer back to orig model
        offset = stage_idx * layers_per_stage
        partial_model = nn.Sequential(
            *full_model[offset : (stage_idx + 1) * layers_per_stage]
        )
        partial_model.to(self.device)
        dp_model = apply_dp(partial_model)
        stage = PipelineStage(
            dp_model,
            stage_idx,
            num_stages,
            self.device,
            group=pp_group,
        )
        return stage, offset

    def _build_pp_schedule(
        self,
        ScheduleClass,
        num_microbatches,
        pp_group,
        full_model,
        total_layers,
        apply_dp,
        loss_fn,
        scale_grads=True,
    ):
        # [FUNCTION DOC] HELPER: Creates a pipeline schedule of the given class. For
        # single-stage-per-rank schedules builds one stage with index equal to the pp rank.
        # Otherwise builds two virtual stages per rank (stage ids rank and rank plus two
        # times the virtual index, over twice the pp size stages) and passes the list.
        # Returns the schedule, the per-stage data-parallel-wrapped submodules and the
        # layer offsets of each stage.
        # KNOWN ISSUE: the virtual stage index uses the virtual count as the stride, which
        # only matches the usual looped layout when the pp group size equals that count (2).
        if issubclass(ScheduleClass, PipelineScheduleSingle):
            pipeline_stage, offset = self._build_pp_stage(
                pp_group,
                full_model,
                total_layers,
                apply_dp,
                pp_group.rank(),
                pp_group.size(),
            )

            partial_models = [pipeline_stage.submod]
            offsets = [offset]
            pipeline_schedule = ScheduleClass(
                pipeline_stage,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
                scale_grads=scale_grads,
            )
        else:
            n_virtual = 2
            num_stages = pp_group.size() * n_virtual
            stages = []
            offsets = []
            for i in range(n_virtual):
                stage, offset = self._build_pp_stage(
                    pp_group,
                    full_model,
                    total_layers,
                    apply_dp,
                    pp_group.rank() + n_virtual * i,
                    num_stages,
                )
                stages.append(stage)
                offsets.append(offset)
            partial_models = [pipeline_stage.submod for pipeline_stage in stages]
            pipeline_schedule = ScheduleClass(
                stages,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
                scale_grads=scale_grads,
            )
        return pipeline_schedule, partial_models, offsets

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            ScheduleInterleaved1F1B,
            ScheduleInterleavedZeroBubble,
        ],
    )
    def test_pp_ddp(self, ScheduleClass):
        # [FUNCTION DOC] Parametrized over GPipe, interleaved 1F1B and interleaved zero
        # bubble; the zero-bubble case returns immediately (DDP unsupported there, see the
        # linked issue). Builds a (world/2, 2) dp-by-pp mesh, an eight layer MLP model and
        # DDP-wrapped stages, runs one pipeline step (first stage feeds input, others feed
        # targets), computes reference gradients by running the full model on each simulated
        # dp rank's input and averaging, and asserts each stage parameter gradient matches
        # the matching reference one. Needs 4 GPUs and NCCL.
        if ScheduleClass == ScheduleInterleavedZeroBubble:
            # TODO: DDP + InterleavedZeroBubble is not currently supported due to issue with DDP reducer not triggering
            # https://github.com/pytorch/pytorch/issues/144530
            return

        torch.get_device_module(device_type).set_device(self.device)
        mesh_shape = (self.world_size // 2, 2)
        mesh_dim_names = ("dp", "pp")
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
        )
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]

        # create "entire model"
        total_layers = 8
        num_microbatches = 8
        dim = 10
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
        ref_model = nn.Sequential(*copy.deepcopy(full_model))
        ref_model.to(self.device)

        # Prepare inputs
        inputs, input_local, _ = self._rand_microbatches(dp_mesh, num_microbatches, dim)
        targets, target_local, _ = self._rand_microbatches(
            dp_mesh, num_microbatches, dim
        )

        def apply_dp(partial_model):
            # [FUNCTION DOC] HELPER: Wraps the stage module in DDP over the dp mesh group.
            return DDP(partial_model, process_group=dp_mesh.get_group())

        # Build pipeline stages, apply data parallelism and attach to a schedule
        pipeline_schedule, partial_models, offsets = self._build_pp_schedule(
            ScheduleClass,
            num_microbatches,
            pp_group,
            full_model,
            total_layers,
            apply_dp,
            loss_fn,
        )

        # Run the pipeline
        if pp_group.rank() == 0:
            pipeline_schedule.step(input_local)
        else:
            pipeline_schedule.step(target=target_local)

        # Ref model runs on 2 different inputs, accumulating grads across them.
        # this ensures that we detect if the DDP all-reduce becomes a no-op.
        for sim_dp_rank in range(dp_mesh.size()):
            loss_fn(ref_model(inputs[sim_dp_rank]), targets[sim_dp_rank]).backward()
        ref_model.to(torch.float32)
        for p in ref_model.parameters():
            p.grad = p.grad.to(torch.float32)
            p.grad /= dp_mesh.size()

        # Validate that whichever weights we have locally match that part of our local/full ref model
        ref_parameters = dict(ref_model.named_parameters())
        for partial_model, offset in zip(partial_models, offsets):
            for name, p in partial_model.named_parameters():
                parts = name.split(".")[
                    1:
                ]  # remove the DDP module. prefix (FSDP2 doesn't have one)
                parts[0] = str(int(parts[0]) + offset)
                name = ".".join(parts)
                ref_p = ref_parameters[name]
                torch.testing.assert_close(p.grad, ref_p.grad)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize("dp_type", ["FSDP", "FSDP_MP"])
    @parametrize(
        "ScheduleClass",
        [
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    def test_pp_fsdp(self, dp_type, ScheduleClass):
        # [FUNCTION DOC] Parametrized over FSDP with and without bfloat16 mixed precision and
        # four schedule classes; returns immediately on ROCm. Same flow as the DDP test but
        # each stage is sharded with fully_shard (per-layer without reshard after forward,
        # then the whole stage). After the pipeline step each gradient is divided then
        # multiplied by two to expose races between pipelining and FSDP reduce-scatter.
        # Asserts every gradient is a DTensor whose full tensor is close to the reference
        # (absolute 5e-5, relative 2e-2). Raises AssertionError if a gradient is missing.
        if TEST_WITH_ROCM:
            return

        torch.get_device_module(device_type).set_device(self.device)
        mesh_shape = (self.world_size // 2, 2)
        mesh_dim_names = ("dp", "pp")
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
        )
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]

        # fsdp_mixed-precision dtype
        mp_dtype = torch.bfloat16 if dp_type == "FSDP_MP" else torch.float32

        # create "entire model"
        total_layers = 8
        num_microbatches = 8
        dim = 10
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
        ref_model = nn.Sequential(*copy.deepcopy(full_model))
        ref_model.to(self.device)
        if dp_type == "FSDP_MP":
            ref_model.to(dtype=mp_dtype)

        # Prepare inputs
        inputs, input_local, _ = self._rand_microbatches(
            dp_mesh, num_microbatches, dim, dtype=mp_dtype
        )
        targets, target_local, _ = self._rand_microbatches(
            dp_mesh, num_microbatches, dim, dtype=mp_dtype
        )

        # Apply FSDP to stage module
        def apply_dp(partial_model):
            # [FUNCTION DOC] HELPER: Applies fully_shard to each child layer (no reshard
            # after forward) and then to the whole module, using the dp mesh and a mixed
            # precision policy (param dtype mp_dtype, reduce dtype float32). Returns the
            # sharded module.
            mp_policy = MixedPrecisionPolicy(
                param_dtype=mp_dtype,
                reduce_dtype=torch.float32,
            )
            fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
            for layer in partial_model.children():
                fully_shard(
                    layer,
                    **fsdp_config,
                    reshard_after_forward=False,
                )
            return fully_shard(partial_model, **fsdp_config)

        # Build pipeline stages, apply data parallelism and attach to a schedule
        pipeline_schedule, partial_models, offsets = self._build_pp_schedule(
            ScheduleClass,
            num_microbatches,
            pp_group,
            full_model,
            total_layers,
            apply_dp,
            loss_fn,
        )

        # Run the pipeline
        if pp_group.rank() == 0:
            pipeline_schedule.step(input_local)
        else:
            pipeline_schedule.step(target=target_local)
        for m in partial_models:
            for p in m.parameters():
                if p.grad is None:
                    raise AssertionError("Expected p.grad to not be None")
                # introduce a race condition for FSDP's reduce-scatter which could corrupt gradients if pipelining
                # does not properly synchronize with FSDP
                p.grad.div_(2.0)
                p.grad.mul_(2.0)

        # Ref model runs on 2 different inputs, accumulating grads across them.
        # this ensures that we detect if the FSDP reduce becomes a no-op.
        # (in fsdp case, we use one of these inputs on each DP rank)
        for sim_dp_rank in range(dp_mesh.size()):
            loss_fn(ref_model(inputs[sim_dp_rank]), targets[sim_dp_rank]).backward()
        ref_model.to(torch.float32)
        for p in ref_model.parameters():
            p.grad = p.grad.to(torch.float32)
            p.grad /= dp_mesh.size()

        # Validate that whichever weights we have locally match that part of our local/full ref model
        # (we force FSDP's grads to be all-gathered (.full_tensor) to make it simpler)
        ref_parameters = dict(ref_model.named_parameters())
        for partial_model, offset in zip(partial_models, offsets):
            for name, p in partial_model.named_parameters():
                parts = name.split(".")
                parts[0] = str(int(parts[0]) + offset)
                name = ".".join(parts)
                ref_p = ref_parameters[name]
                self.assertTrue(isinstance(p.grad, DTensor))
                torch.testing.assert_close(
                    p.grad.full_tensor(), ref_p.grad, atol=5e-5, rtol=2e-2
                )

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize("dp_type", ["FSDP", "FSDP_MP"])
    def test_pp_fsdp_unshard_reshard_runtime(self, dp_type):
        # [FUNCTION DOC] On a (world, 1) mesh, builds one four-layer FSDP stage and drives it
        # with hand-written schedules in the pipeline schedule runtime (one microbatch, no
        # loss). Checks all FSDP parameters are sharded initially (skips if none exist), runs
        # an unshard, forward, reshard schedule and asserts parameters are sharded again,
        # then an unshard, forward, reduce-grad schedule and asserts they are sharded again.
        # Returns immediately on ROCm.
        """Test FSDP UNSHARD/RESHARD functionality using _PipelineScheduleRuntime with custom schedules."""
        if TEST_WITH_ROCM:
            return

        torch.get_device_module(device_type).set_device(self.device)
        mesh_shape = (self.world_size, 1)
        mesh_dim_names = ("dp", "pp")
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
        )
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]

        # fsdp_mixed-precision dtype
        mp_dtype = torch.bfloat16 if dp_type == "FSDP_MP" else torch.float32
        total_layers = 4
        dim = 10
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])

        def apply_dp(partial_model):
            # [FUNCTION DOC] HELPER: Same sharding as in the FSDP test: fully_shard on each
            # child layer (no reshard after forward) and then on the whole module, with the
            # dp mesh and mixed precision (param dtype mp_dtype, reduce dtype float32).
            # Returns the sharded module.
            mp_policy = MixedPrecisionPolicy(
                param_dtype=mp_dtype,
                reduce_dtype=torch.float32,
            )
            fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
            for layer in partial_model.children():
                fully_shard(
                    layer,
                    **fsdp_config,
                    reshard_after_forward=False,
                )
            return fully_shard(partial_model, **fsdp_config)

        # Build pipeline stages
        num_stages = pp_group.size()
        layers_per_stage = total_layers // num_stages
        stage_idx = pp_group.rank()
        offset = stage_idx * layers_per_stage

        partial_model = nn.Sequential(
            *full_model[offset : (stage_idx + 1) * layers_per_stage]
        )
        partial_model.to(self.device)
        fsdp_model = apply_dp(partial_model)
        distributed_state = fully_shard.state(fsdp_model)
        distributed_state._lazy_init()

        stage = PipelineStage(
            fsdp_model,
            stage_idx,
            num_stages,
            self.device,
            group=pp_group,
        )

        # Helper function to check FSDP sharding state
        def check_fsdp_unsharded_state(module, expected_unsharded=False):
            # [FUNCTION DOC] HELPER: Walks all FSDP states and parameter groups of the module,
            # counting parameters whose sharded state is unsharded. Asserts that all are
            # unsharded if expected_unsharded, else that none are. Returns whether any FSDP
            # parameter was found.
            """Check if FSDP parameters are in expected sharding state."""
            distributed_state = fully_shard.state(module)
            unsharded_count = 0
            total_fsdp_params = 0

            for state in distributed_state._state_ctx.all_states:
                for group in state._fsdp_param_groups:
                    for fsdp_param in group.fsdp_params:
                        total_fsdp_params += 1
                        if fsdp_param.sharded_state == ShardedState.UNSHARDED:
                            unsharded_count += 1

            if expected_unsharded:
                self.assertEqual(
                    unsharded_count,
                    total_fsdp_params,
                    lambda msg: f"{msg}\nExpected all {total_fsdp_params} FSDP parameters to be unsharded, "
                    f"but only {unsharded_count} are unsharded",
                )
            else:
                self.assertEqual(
                    unsharded_count,
                    0,
                    lambda msg: f"{msg}\nExpected all FSDP parameters to be sharded, "
                    f"but {unsharded_count} out of {total_fsdp_params} are unsharded",
                )

            return total_fsdp_params > 0  # Return whether we found any FSDP parameters

        # Test initial state - should be sharded
        has_fsdp = check_fsdp_unsharded_state(stage.submod, expected_unsharded=False)

        if not has_fsdp:
            self.skipTest("No FSDP parameters found in the model")

        def create_schedule(computation_types, microbatch_index=None):
            # [FUNCTION DOC] HELPER: Builds a one-rank schedule mapping rank 0 to a list of
            # actions on stage 0, one per computation type; only forward actions carry the
            # given microbatch index, others get none. Returns the mapping.
            schedule = {
                0: [
                    _Action(
                        stage_index=0,  # stage 0 (the only stage)
                        computation_type=comp_type,
                        microbatch_index=microbatch_index
                        if comp_type == _ComputationType.FORWARD
                        else None,
                    )
                    for comp_type in computation_types
                ]
            }
            return schedule

        unshard_schedule = create_schedule(
            [
                _ComputationType.UNSHARD,
                _ComputationType.FORWARD,
                _ComputationType.REDUCE_GRAD,  # Contains final fsdp post_backward
            ],
            microbatch_index=0,
        )
        unshard_reshard_schedule = create_schedule(
            [
                _ComputationType.UNSHARD,
                _ComputationType.FORWARD,
                _ComputationType.RESHARD,
            ],
            microbatch_index=0,
        )

        # Test 1: Run UNSHARD + RESHARD schedule
        runtime = _PipelineScheduleRuntime(
            [stage], n_microbatches=1, loss_fn=None, scale_grads=False
        )
        runtime.pipeline_order_with_comms = unshard_reshard_schedule
        dummy_input = torch.randn(1, dim, device=self.device, dtype=mp_dtype)
        runtime.step(dummy_input)

        # Verify parameters are now sharded again
        check_fsdp_unsharded_state(stage.submod, expected_unsharded=False)

        # Test 2: Run UNSHARD only schedule
        runtime.pipeline_order_with_comms = unshard_schedule
        runtime.step(dummy_input)

        # Verify parameters are still sharded
        check_fsdp_unsharded_state(stage.submod, expected_unsharded=False)


instantiate_parametrized_tests(ComposabilityTest)
if __name__ == "__main__":
    run_tests()
