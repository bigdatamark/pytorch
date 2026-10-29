# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for FSDP2 (fully_shard) communication behavior. Contents: direct tests of
# the foreach all-gather and foreach reduce-scatter collectives at world size 128 on threads;
# communication-count tests using CommDebugMode (reshard_after_forward variants, manual reshard,
# set_reshard_after_forward); gradient divide factor and reduce-scatter input buffer cap setters;
# forward/backward prefetch ordering tests built on recorded unshard, reshard and post-backward
# events; async unshard; allocation from the process group; symmetric memory; forced sum
# reduction; and reduce op at world size 1. Import-time side effects: defines c10d_ops,
# EventType, device_type and device_module from get_devtype().

import copy
import functools
import itertools
import os
import tempfile
import unittest
from collections.abc import Callable
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._composable import checkpoint, replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _div_if_needed,
    _get_gradient_divide_factors,
    DefaultAllGather,
    DefaultReduceScatter,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo, TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental import implicit_replication
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    DoubleLinear,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_post_backward,
    patch_reshard,
    patch_unshard,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfTorchInductor,
    TEST_WITH_ROCM,
    TEST_XPU,
    xfailIf,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    FeedForward,
    ModelArgs,
    Transformer,
    TransformerBlock,
)
from torch.testing._internal.inductor_utils import skipCUDAIf


c10d_ops = torch.ops.c10d

# For recording FSDP events like unshard or post-backward
EventType = tuple[str, str, TrainingState]

from torch.testing._internal.common_fsdp import get_devtype


device_type = torch.device(get_devtype())
device_module = torch.get_device_module(device_type)


class TestFullyShardCollectiveOps(FSDPTestMultiThread):
    # [CLASS DOC] Single-process multi-thread tests (128 simulated ranks) of the FSDP2 collective
    # helpers: builds an FSDPParamGroup directly over a parameter list and checks the foreach
    # all-gather (with copy-in and copy-out) and foreach reduce-scatter against references.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Fixed simulated world size of 128.
        return 128

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns device index 0 of the test device type.
        return torch.device(device_type.type, 0)

    def _get_param_sizes(self) -> list[torch.Size]:
        # [FUNCTION DOC] HELPER: Returns a fixed list of 8 parameter shapes (mixed 1D and 2D,
        # many not divisible by 128) used for the collective tests; sized for about 0.22 GB
        # at world 128.
        # For world size 128, the fp32 all-gather and reduce-scatter testing
        # requires ~0.22 GB
        return [
            torch.Size([17, 257]),
            torch.Size([17]),
            torch.Size([64, 312]),
            torch.Size([64]),
            torch.Size([64, 64]),
            torch.Size([512, 64]),
            torch.Size([256]),
            torch.Size([64, 297]),
        ]

    def _init_params(self, param_sizes: list[torch.Size]) -> list[nn.Parameter]:
        # [FUNCTION DOC] HELPER: Seeds the RNG with 42, creates a random parameter on the test
        # device for each size and broadcasts each from rank 0 so all (thread) ranks hold
        # identical values. Returns the list of nn.Parameter.
        torch.manual_seed(42)
        orig_params = [
            nn.Parameter(torch.randn(size, device=self.device)) for size in param_sizes
        ]
        # Since seed is per process, not per thread, we broadcast to ensure the
        # same original parameters across ranks
        for orig_param in orig_params:
            dist.broadcast(orig_param, src=0)
        return orig_params

    def _init_fsdp_param_group(
        self, params: list[nn.Parameter], reshard_after_forward: bool | int
    ):
        # [FUNCTION DOC] HELPER: Wraps copies of params in an nn.ParameterList module, builds a
        # one-dim shard mesh info over the default mesh and the post-forward mesh info for
        # reshard_after_forward, and constructs an FSDPParamGroup (default mixed precision, no
        # offload, no placement fn). Calls lazy_init and returns the group.
        module = nn.ParameterList([param.detach().clone() for param in params])
        mesh_info = FSDPMeshInfo(_init_default_fully_shard_mesh(), shard_mesh_dim=0)
        post_forward_mesh_info = _get_post_forward_mesh_info(
            reshard_after_forward, mesh_info
        )
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()),
            (module,),
            mesh_info,
            post_forward_mesh_info,
            self.device,
            None,  # shard_placement_fn
            MixedPrecisionPolicy(),
            OffloadPolicy(),
        )
        fsdp_param_group.lazy_init()
        return fsdp_param_group

    @skip_if_lt_x_gpu(1)
    def test_all_gather_fp32(self):
        # [FUNCTION DOC] Runs the all-gather check over the product of sync/async, default-stream
        # versus two separate streams (copy-in, all-gather), and reshard_after_forward True or 8.
        # The int reshard case is only tested for sync and non-default streams to save time.
        # Needs at least 1 GPU.
        param_sizes = self._get_param_sizes()
        default_stream = device_module.current_stream()
        stream1, stream2 = (
            device_module.Stream(),
            device_module.Stream(),
        )
        for async_op, streams, reshard_after_forward in itertools.product(
            (False, True),
            ((default_stream, default_stream), (stream1, stream2)),
            (True, 8),
        ):
            all_gather_copy_in_stream, all_gather_stream = streams
            # Save test time by only testing reshard after forward as an int
            # for non-async and non-default streams (like in pre-backward)
            if type(reshard_after_forward) is int and (
                async_op or all_gather_stream is default_stream
            ):
                continue
            self._test_all_gather(
                param_sizes,
                reshard_after_forward=reshard_after_forward,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
            )

    def _test_all_gather(
        self,
        param_sizes: list[torch.Size],
        reshard_after_forward: bool | int,
        async_op: bool,
        all_gather_copy_in_stream,
        all_gather_stream,
    ):
        # [FUNCTION DOC] HELPER: Core all-gather check. Builds a param group from random
        # parameters, asserts the sharded params are DTensors whose full tensor equals
        # the originals, runs the foreach all-gather with copy-in, copy-out and the
        # transition to unsharded state, and asserts the unsharded params are
        # nn.Parameters equal to the originals. When reshard_after_forward is an int it
        # also moves to the sharded-post-forward state and repeats the all-gather over
        # the post-forward process group.
        # KNOWN ISSUE: the nested all_gather helper calls the copy-out with the enclosing
        # variable fsdp_params (bound later in this function) rather than the group passed in; it
        # only works because both name the same group's parameters.
        def all_gather(fsdp_param_group: FSDPParamGroup, group: dist.ProcessGroup):
            # [FUNCTION DOC] HELPER: Runs the foreach all-gather for the group's FSDP params over
            # the given process group using the default all-gather, then the copy-out,
            # initializes each unsharded param and moves the group to the unsharded state.
            all_gather_comm = DefaultAllGather()
            all_gather_result = foreach_all_gather(
                fsdp_param_group.fsdp_params,
                group,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
                device=self.device,
                all_gather_comm=all_gather_comm,
            )
            foreach_all_gather_copy_out(all_gather_result, fsdp_params, group)
            # Transition to unsharded state to register unsharded parameters
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.init_unsharded_param()
            fsdp_param_group._to_unsharded()

        def check_all_gathered_params(
            orig_params: list[nn.Parameter], module: nn.Module
        ):
            # [FUNCTION DOC] HELPER: Asserts each parameter of module is an nn.Parameter tensor
            # equal to the corresponding original parameter cast to its dtype.
            for orig_param, param in zip(orig_params, module.parameters()):
                self.assertIsInstance(param, torch.Tensor)
                self.assertIsInstance(param, nn.Parameter)
                self.assertEqual(param, orig_param.to(param.dtype))

        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(
            orig_params, reshard_after_forward
        )
        fsdp_params = fsdp_param_group.fsdp_params
        module = fsdp_param_group.modules[0]

        # Sanity check that the parameter sharding is as expected
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, DTensor))
            self.assertEqual(param.full_tensor(), orig_param)

        # Run the foreach all-gather (including copy-in and copy-out)
        all_gather(fsdp_param_group, fsdp_param_group.mesh_info.shard_process_group)

        # Check all-gather correctness
        check_all_gathered_params(orig_params, module)

        # For reshard after after forward as an int, further test emulating the
        # pre-backward all-gather
        if type(reshard_after_forward) is not int:
            return
        fsdp_param_group._to_sharded_post_forward()
        all_gather(
            fsdp_param_group,
            fsdp_param_group.post_forward_mesh_info.shard_process_group,
        )
        check_all_gathered_params(orig_params, module)

    @skip_if_lt_x_gpu(1)
    def test_reduce_scatter_fp32(self):
        # [FUNCTION DOC] Runs the reduce-scatter check in float32 with both the default stream
        # and a new stream as the reduce-scatter stream. Needs at least 1 GPU.
        param_sizes = self._get_param_sizes()
        default_stream = device_module.current_stream()
        stream = device_module.Stream()
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float32,
            )

    @skip_if_lt_x_gpu(1)
    def test_reduce_scatter_fp16(self):
        # [FUNCTION DOC] Same as the fp32 reduce-scatter test but reduces in float16. Needs at
        # least 1 GPU.
        param_sizes = self._get_param_sizes()
        default_stream = torch.get_device_module(device_type).current_stream()
        stream = device_module.Stream()
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float16,
            )

    def _test_reduce_scatter(
        self,
        param_sizes: list[torch.Size],
        reduce_scatter_stream,
        reduce_scatter_dtype: torch.dtype,
    ):
        # [FUNCTION DOC] HELPER: Core reduce-scatter check. Builds a param group, initializes
        # the comm context, does one unshard, wait and reshard to set up metadata, then runs
        # the foreach reduce-scatter of gradients that equal each rank's index on the given
        # stream in the given dtype (no all-reduce group). After syncing on the post-reduce
        # event it computes a reference by dividing with the default factors, all-reducing
        # and dividing, and asserts each sharded param grad is a DTensor whose full tensor
        # equals the reference.
        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(orig_params, True)
        fsdp_params = fsdp_param_group.fsdp_params
        fsdp_param_group.comm_ctx.lazy_init(self.device)

        # Run one unshard to initialize metadata
        fsdp_param_group.unshard()
        fsdp_param_group.wait_for_unshard()
        fsdp_param_group.reshard()

        # Run the foreach reduce-scatter (including copy-in and view-out)
        torch.manual_seed(42)
        unsharded_grads = [torch.ones_like(param) * self.rank for param in orig_params]
        group = fsdp_param_group.mesh_info.shard_process_group
        self.assertEqual(group.size(), self.world_size)
        all_reduce_stream = device_module.Stream()
        comm = DefaultReduceScatter()
        (
            _,
            _,
            _,
            post_reduce_event,
            _,
            _,
            _,
        ) = foreach_reduce(
            fsdp_params,
            unsharded_grads,
            group,
            reduce_scatter_stream,
            comm,
            orig_dtype=orig_params[0].dtype,
            reduce_dtype=reduce_scatter_dtype,
            device=self.device,
            gradient_divide_factor=None,
            all_reduce_group=None,
            all_reduce_stream=all_reduce_stream,
            all_reduce_hook=None,
            all_reduce_grads=True,
            partial_reduce_output=None,
        )
        torch.get_device_module(device_type).current_stream().wait_event(
            post_reduce_event
        )

        # Check reduce-scatter correctness
        (
            predivide_factor,
            postdivide_factor,
            _,
            all_reduce_op,
        ) = _get_gradient_divide_factors(group, None, reduce_scatter_dtype)
        reduced_grads = [grad.detach().clone() for grad in unsharded_grads]
        for grad in reduced_grads:
            _div_if_needed(grad, predivide_factor)
            dist.all_reduce(
                grad,
                group=group,
                op=all_reduce_op,
            )
            _div_if_needed(grad, postdivide_factor)
        for fsdp_param, reduced_grad in zip(fsdp_params, reduced_grads):
            sharded_grad = fsdp_param.sharded_param.grad
            self.assertIsInstance(sharded_grad, DTensor)
            self.assertEqual(sharded_grad.full_tensor(), reduced_grad)


class TestFullyShardCommunication(FSDPTest):
    # [CLASS DOC] Multi-process tests that count collectives issued by FSDP2 on a transformer
    # under different reshard_after_forward settings and check gradient divide factor and
    # reduce-scatter buffer settings. world_size is min(4, device count).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns min(4, device count for the test device type).
        return min(4, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_communication_count(self):
        # [FUNCTION DOC] Runs the communication count subtest for reshard_after_forward in True,
        # False, 2 and None. Needs at least 2 GPUs.
        """
        Tests that FSDP issues the expected number of all-gathers and
        reduce-scatters during forward and backward.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2, None]},
            self._test_communication_count,
        )

    def _test_communication_count(
        self,
        reshard_after_forward: bool | int | None,
    ):
        # [FUNCTION DOC] HELPER: Shards each TransformerBlock and the root of a Transformer with
        # the given reshard_after_forward, then uses CommDebugMode to assert the forward has only
        # all-gathers (blocks plus one). In backward it asserts all-gathers of blocks plus root
        # when resharding is on, only blocks when None (root not resharded), and none when False,
        # while reduce-scatters always equal blocks plus one.
        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        num_blocks = 0
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
                num_blocks += 1
        fully_shard_fn(model)
        # We construct `num_blocks` plus 1 FSDP states/communication groups

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)
        with CommDebugMode() as fwd_comm_mode:
            loss = model(inp)
        fwd_comm_counts = fwd_comm_mode.get_comm_counts()
        self.assertEqual(len(fwd_comm_counts), 1)
        self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_blocks + 1)
        with CommDebugMode() as bwd_comm_mode:
            loss.sum().backward()
        bwd_comm_counts = bwd_comm_mode.get_comm_counts()
        if reshard_after_forward is None:
            # 2 means two types of collectives (all-gather, reduce-scatter)
            self.assertEqual(len(bwd_comm_counts), 2)
            # do not reshard root model
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_blocks)
        elif reshard_after_forward:
            self.assertEqual(len(bwd_comm_counts), 2)
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_blocks + 1)
        else:
            self.assertEqual(len(bwd_comm_counts), 1)
        self.assertEqual(
            bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_blocks + 1
        )

    @skip_if_lt_x_gpu(2)
    def test_manual_reshard_with_reshard_after_forward_false(self):
        # [FUNCTION DOC] Shards all modules with reshard_after_forward False, checks forward does
        # one all-gather per FSDP module, manually calls reshard on every FSDP module, and then
        # checks backward re-all-gathers once per module and reduce-scatters once per module.
        # Needs at least 2 GPUs.
        """
        Tests that we can manually call ``reshard`` on FSDP modules that were
        initialized with ``reshard_after_forward=False`` and still run unshard.
        """
        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, reshard_after_forward=False)
        model = fully_shard(model, reshard_after_forward=False)
        num_fsdp_modules = sum(
            isinstance(module, FSDPModule) for module in model.modules()
        )

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)
        with CommDebugMode() as fwd_comm_mode:
            loss = model(inp)
        fwd_comm_counts = fwd_comm_mode.get_comm_counts()
        self.assertEqual(len(fwd_comm_counts), 1)
        self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_fsdp_modules)

        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        with CommDebugMode() as bwd_comm_mode:
            loss.sum().backward()
        bwd_comm_counts = bwd_comm_mode.get_comm_counts()
        self.assertEqual(len(bwd_comm_counts), 2)
        self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_fsdp_modules)
        self.assertEqual(
            bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_fsdp_modules
        )

    @skip_if_lt_x_gpu(2)
    @xfailIf(TEST_XPU)  # https://github.com/intel/torch-xpu-ops/issues/1571
    def test_set_reduce_scatter_divide_factor(self):
        # [FUNCTION DOC] Runs the gradient divide factor parity subtests over divide factors
        # (twice the world size and the world size) and mesh shapes (1D, 2D, and shard dim of 1)
        # and then a mixed precision variant. xfail on XPU. Needs at least 2 GPUs.
        self.run_subtests(
            {
                "divide_factor": [self.world_size * 2, self.world_size],
                "mesh_shape": [
                    (self.world_size,),
                    (self.world_size // 2, 2),
                    (self.world_size, 1),
                ],
            },
            self._test_set_reduce_scatter_divide_factor,
        )
        self.run_subtests(
            {"divide_factor": [self.world_size]},
            self._test_set_reduce_scatter_divide_factor_mixed_prevision,
        )

    def _test_set_reduce_scatter_divide_factor(
        self, divide_factor: float, mesh_shape: tuple[int] | tuple[int, int]
    ):
        # [FUNCTION DOC] HELPER: Compares an FSDP transformer using set_gradient_divide_factor on
        # a mesh of the given shape to a reference whose gradients are scaled manually (by the
        # divide factor for non-block params and by world size for block params) and all-reduced.
        # Runs 10 AdamW steps asserting equal losses and sharded parity before zero_grad.
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0, weight_tying=False)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        mesh_dim_names = ("outer",) if len(mesh_shape) == 1 else ("outer", "inner")
        mesh = init_device_mesh(
            device_type.type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, reshard_after_forward=False, mesh=mesh)
        model = fully_shard(model, reshard_after_forward=False, mesh=mesh)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.set_gradient_divide_factor(divide_factor)

        # Get ref_model params which should have the specific division factor applied
        block_params = set()
        for ref_mod in ref_model.modules():
            if isinstance(ref_mod, TransformerBlock):
                block_params.update(ref_mod.parameters())
        non_block_params = set(ref_model.parameters()) - block_params

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)

        for _ in range(10):
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            for param in ref_model.parameters():
                factor = divide_factor if param in non_block_params else self.world_size
                param.grad.mul_(1.0 / factor)
                dist.all_reduce(param.grad)
            loss = model(inp).sum()
            loss.backward()
            ref_optim.step()
            optim.step()
            self.assertEqual(ref_loss, loss)
            # Check parity before calling zero_grad so that grads are also checked
            check_sharded_parity(self, ref_model, model)
            ref_optim.zero_grad()
            optim.zero_grad()

    def _test_set_reduce_scatter_divide_factor_mixed_prevision(
        self, divide_factor: float
    ):
        # [FUNCTION DOC] HELPER: Same divide factor check with bfloat16 param dtype and float32
        # reduce dtype on a three-layer MLP. The reference is a bfloat16 copy whose grads are
        # cast to float32, scaled by the divide factor, all-reduced and moved to a float32
        # reference that steps AdamW and copies the result back. Runs 10 steps asserting equal
        # losses and sharded parity.
        torch.manual_seed(42)
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        model = nn.Sequential(*[MLP(16) for _ in range(3)])
        ref_model = copy.deepcopy(model).to(device_type)
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            fully_shard(mlp, mp_policy=mp_policy)
        model = fully_shard(model, mp_policy=mp_policy)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.set_gradient_divide_factor(divide_factor)

        torch.manual_seed(42 + self.rank)
        inp = torch.randn((4, 16), device=device_type.type, dtype=param_dtype)

        for _ in range(10):
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

            ref_loss = ref_model_bf16(inp.to(param_dtype)).sum()
            ref_loss.backward()
            for param in ref_model_bf16.parameters():
                param.grad.data = param.grad.to(torch.float32)
                param.grad.mul_(1.0 / divide_factor)
                dist.all_reduce(param.grad)
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad
                param_bf16.grad = None
            ref_optim.step()
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)
            ref_optim.zero_grad()

            self.assertEqual(ref_loss, loss)
            check_sharded_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(2)
    def test_set_reduce_scatter_max_input_buffers(self):
        # [FUNCTION DOC] Runs the max-input-buffers parity subtest over 1D and 2D meshes, both
        # reshard_after_forward values and caps of 1 and 2: the cap is a scheduling change only
        # and must not change results. Needs at least 2 GPUs.
        """
        Tests that ``set_reduce_scatter_max_input_buffers`` is a pure
        scheduling change: changing the in-flight reduce-scatter copy-in buffer
        cap (``2``, or an explicit ``1`` that matches the default) produces
        identical parameters and gradients to the default single-buffer
        behavior, across FSDP/HSDP and reshard-after-forward (under bf16 mixed
        precision, where the retained buffer is in the reduce dtype).
        """
        self.run_subtests(
            {
                "mesh_shape": [(self.world_size,), (self.world_size // 2, 2)],
                "reshard_after_forward": [True, False],
                "max_input_buffers": [1, 2],
            },
            self._test_set_reduce_scatter_max_input_buffers,
        )

    def _test_set_reduce_scatter_max_input_buffers(
        self,
        mesh_shape: tuple[int] | tuple[int, int],
        reshard_after_forward: bool,
        max_input_buffers: int,
    ):
        # [FUNCTION DOC] HELPER: Builds a reference and a copy transformer sharded with
        # bfloat16 param and reduce dtypes. Calls set_reduce_scatter_max_input_buffers on the
        # copy only, then runs 3 AdamW steps on both asserting equal losses and sharded
        # parity (grads included).
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0, weight_tying=False)
        ref_model = Transformer(model_args)
        model = copy.deepcopy(ref_model)
        mesh_dim_names = ("outer",) if len(mesh_shape) == 1 else ("outer", "inner")
        mesh = init_device_mesh(
            device_type.type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
        fsdp_kwargs = {
            "reshard_after_forward": reshard_after_forward,
            "mesh": mesh,
            "mp_policy": mp_policy,
        }
        # Reference model keeps the default (copy-in on the compute stream)
        for module in ref_model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, **fsdp_kwargs)
        ref_model = fully_shard(ref_model, **fsdp_kwargs)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        # Test model retains multiple reduce-scatter copy-in buffers in flight
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, **fsdp_kwargs)
        model = fully_shard(model, **fsdp_kwargs)
        model.set_reduce_scatter_max_input_buffers(max_input_buffers)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)
        for _ in range(3):
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            ref_optim.step()
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            self.assertEqual(ref_loss, loss)
            # Check parity before zero_grad so gradients are compared too
            check_sharded_parity(self, ref_model, model)
            ref_optim.zero_grad()
            optim.zero_grad()

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_max_input_buffers_resolves_to_max(self):
        # [FUNCTION DOC] Checks that mixed per-module reduce-scatter input buffer caps resolve to
        # the maximum on the shared comm context. Sets cap 3 on one block (no recursion), runs a
        # forward and backward to trigger lazy init, then asserts the comm context cap is 3, that
        # block's param group keeps 3 and another block's group keeps the default of 1. Needs at
        # least 2 GPUs.
        """
        Per-module reduce-scatter input-buffer caps share one pipeline (a single
        ``comm_ctx.reduce_scatter_states`` list), so mixed caps must resolve to
        the max on the shared comm context: a lower-cap module cannot silently
        undo a higher-cap module's retention. The per-group values stay as set.
        """
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)
        blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks:
            fully_shard(block)
        model = fully_shard(model)
        # Set a non-default cap on a single block only (recurse=False); every
        # other group (including the root) keeps the default of 1.
        expected_max = 3
        blocks[0].set_reduce_scatter_max_input_buffers(expected_max, recurse=False)

        # Resolution runs in lazy init on the first forward.
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)
        model(inp).sum().backward()

        comm_ctx = model._get_fsdp_state()._comm_ctx
        self.assertEqual(comm_ctx.reduce_scatter_max_input_buffers, expected_max)
        # Resolution leaves the per-group input values untouched.
        block0_pg = blocks[0]._get_fsdp_state()._fsdp_param_groups[0]
        block1_pg = blocks[1]._get_fsdp_state()._fsdp_param_groups[0]
        self.assertEqual(block0_pg.reduce_scatter_max_input_buffers, expected_max)
        self.assertEqual(block1_pg.reduce_scatter_max_input_buffers, 1)

    @skip_if_lt_x_gpu(2)
    def test_set_reduce_scatter_max_input_buffers_validation(self):
        # [FUNCTION DOC] Checks validation in set_reduce_scatter_max_input_buffers: 0 raises
        # ValueError, a float and a bool raise TypeError, and a valid positive int is accepted.
        # Needs at least 2 GPUs.
        model = fully_shard(nn.Linear(16, 16))
        # Non-positive -> ValueError
        with self.assertRaises(ValueError):
            model.set_reduce_scatter_max_input_buffers(0)
        # Non-int -> TypeError, even though 2.5 >= 1 numerically
        with self.assertRaises(TypeError):
            model.set_reduce_scatter_max_input_buffers(2.5)
        # bool is an int subclass but is not a valid count -> TypeError
        with self.assertRaises(TypeError):
            model.set_reduce_scatter_max_input_buffers(True)
        # A valid value is accepted
        model.set_reduce_scatter_max_input_buffers(2)

    @skip_if_lt_x_gpu(2)
    def test_set_reshard_after_forward(self):
        # [FUNCTION DOC] Runs the set_reshard_after_forward communication count subtest over
        # None, True and False with recurse on and off. Needs at least 2 GPUs.
        """
        Tests that FSDP issues the expected number of all-gathers and
        reduce-scatters during a train step when setting reshard_after_forward.
        comm_count should perform same as test_fully_shard_communication_count.
        """
        self.run_subtests(
            {
                "set_reshard_after_forward": [True, False, None],
                "recurse": [True, False],
            },
            self._test_set_reshard_after_forward_by_communication_count,
        )

    def _test_set_reshard_after_forward_by_communication_count(
        self,
        set_reshard_after_forward: bool | None,
        recurse: bool,
    ):
        # [FUNCTION DOC] HELPER: Shards a transformer (with the opposite reshard_after_forward of
        # the value to be set, or default when None), then calls set_reshard_after_forward on the
        # root with the recurse flag. Using CommDebugMode asserts the forward has one all-gather
        # per FSDP module and checks the backward all-gather count (blocks, blocks plus one, only
        # the root, or none) for each case, and that reduce-scatters equal blocks plus one.
        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args).to(device_type)
        if set_reshard_after_forward is None:
            fully_shard_fn = fully_shard
        else:
            fully_shard_fn = functools.partial(
                fully_shard, reshard_after_forward=not set_reshard_after_forward
            )

        num_blocks = 0
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
                num_blocks += 1
        fully_shard_fn(model)
        num_fsdp_modules = sum(
            isinstance(module, FSDPModule) for module in model.modules()
        )
        if set_reshard_after_forward is not None:
            model.set_reshard_after_forward(
                reshard_after_forward=set_reshard_after_forward, recurse=recurse
            )

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)
        with CommDebugMode() as fwd_comm_mode:
            loss = model(inp)
        fwd_comm_counts = fwd_comm_mode.get_comm_counts()
        self.assertEqual(len(fwd_comm_counts), 1)
        self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_fsdp_modules)

        with CommDebugMode() as bwd_comm_mode:
            loss.sum().backward()
        bwd_comm_counts = bwd_comm_mode.get_comm_counts()
        # If recurse is False, set_reshard_after_forward only affects the root module
        if set_reshard_after_forward is None:
            self.assertEqual(len(bwd_comm_counts), 2)
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_blocks)
        elif set_reshard_after_forward:
            self.assertEqual(len(bwd_comm_counts), 2)
            self.assertEqual(
                bwd_comm_counts[c10d_ops._allgather_base_],
                num_blocks + 1 if recurse else 1,
            )
        else:
            if recurse:
                self.assertEqual(len(bwd_comm_counts), 1)
            else:
                self.assertEqual(len(bwd_comm_counts), 2)
                self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_blocks)

        self.assertEqual(
            bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_blocks + 1
        )


class TestFullyShardPrefetch(FSDPTest):
    # [CLASS DOC] Multi-process tests of FSDP2 forward and backward prefetching. Wraps unshard,
    # reshard and post-backward to record events and compares against expected event orders.
    # world_size is min(4, device count).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns min(4, device count for the test device type).
        return min(4, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_backward_prefetch(self):
        # [FUNCTION DOC] Runs the backward prefetch subtests (single forward/backward, multiple
        # forwards before one backward) over reshard_after_forward in True, False, 2, None and
        # checkpoint implementation none, utils or composable, then the unused-in-backward case
        # with True. Activation checkpointing must not change the expected FSDP event order.
        # Needs at least 2 GPUs.
        # Activation checkpointing should not affect the expected FSDP events
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2, None],
                "checkpoint_impl": [None, "utils", "composable"],
            },
            self._test_backward_prefetch_forward_backward,
        )
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2, None],
                "checkpoint_impl": [None, "utils", "composable"],
            },
            self._test_backward_prefetch_multi_forward,
        )
        self._test_backward_prefetch_unused_in_backward(True)

    def _test_backward_prefetch_forward_backward(
        self,
        reshard_after_forward: bool | int | None,
        checkpoint_impl: str | None,
    ):
        # [FUNCTION DOC] HELPER: Builds a 3-layer transformer, records unshard and post-backward
        # events via patches, and for 3 iterations asserts the forward events (root then each
        # layer) and the backward events (root unshard unless reshard_after_forward is None,
        # layer unshards moved one module early by explicit prefetch, post-backwards from last
        # layer to root). With reshard_after_forward False it expects no backward unshard events.
        # Steps the optimizer and alternates the zero_grad mode.
        n_layers = 3
        model, optim, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )
        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        # Check the order for normal 1 forward, 1 backward, 1 optimizer step
        with (
            patch_unshard(unshard_with_record),
            patch_post_backward(post_backward_with_record),
        ):
            for iter_idx in range(3):
                loss = model(inp)
                expected_events = [
                    ("unshard", "", TrainingState.FORWARD),  # root
                    ("unshard", "layers.0", TrainingState.FORWARD),
                    ("unshard", "layers.1", TrainingState.FORWARD),
                    ("unshard", "layers.2", TrainingState.FORWARD),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                loss.sum().backward()
                expected_events = []
                # Root does not reshard after forward so there is no
                # unshard event for it in backward
                if reshard_after_forward is not None:
                    expected_events.append(("unshard", "", TrainingState.PRE_BACKWARD))
                expected_events.extend(
                    [
                        ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                        # Explicit backward prefetching moves the unshards early
                        # by one module (note how swapping each unshard down one
                        # event would give the natural event order)
                        ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                        ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                        ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                        ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                        ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                        ("post_backward", "", TrainingState.POST_BACKWARD),
                    ]
                )
                if reshard_after_forward is False:
                    # No reshard after forward means no backward unshards
                    expected_events = [e for e in expected_events if e[0] != "unshard"]
                self.assertEqual(events, expected_events)
                events.clear()
                optim.step()
                optim.zero_grad(set_to_none=(iter_idx % 2 == 0))

    def _test_backward_prefetch_multi_forward(
        self, reshard_after_forward: bool | int, checkpoint_impl: str | None
    ):
        # [FUNCTION DOC] HELPER: Records unshard and post-backward events for two forwards before
        # one backward on a 3-layer transformer. Asserts the forward events (the second forward
        # unshards again unless reshard_after_forward is False), then the backward events for the
        # first backward pass (without the root post-backward, which waits for the final
        # callback) followed by the repeated pattern ending with the root post-backward.
        n_layers = 3
        model, _, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )
        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        # Check the order for multiple forwards before 1 backward
        with (
            patch_unshard(unshard_with_record),
            patch_post_backward(post_backward_with_record),
        ):
            loss1 = model(inp)
            loss2 = model(inp)
            expected_events = [
                ("unshard", "", TrainingState.FORWARD),  # root
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1", TrainingState.FORWARD),
                ("unshard", "layers.2", TrainingState.FORWARD),
            ]
            if reshard_after_forward is not None:
                expected_events.append(("unshard", "", TrainingState.FORWARD))
            expected_events.extend(
                [
                    ("unshard", "layers.0", TrainingState.FORWARD),
                    ("unshard", "layers.1", TrainingState.FORWARD),
                    ("unshard", "layers.2", TrainingState.FORWARD),
                ]
            )
            if reshard_after_forward is False:
                # No reshard after forward means no second set of unshards
                expected_events = expected_events[:-4]
            self.assertEqual(events, expected_events)
            events.clear()
            (loss1 + loss2).sum().backward()
            expected_events = []
            if reshard_after_forward is not None:
                expected_events.append(("unshard", "", TrainingState.PRE_BACKWARD))
            expected_events.extend(
                [
                    # Same as the single forward/backward case except the root's
                    # post-backward does not run until the end of backward in the
                    # final callback (since the input not requiring gradient means
                    # that we do not have a tensor on which to hook for
                    # post-backward)
                    ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                    ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                    ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                    ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                    ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                    ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ]
            )
            if reshard_after_forward is False:
                # No reshard after forward means no backward unshards
                expected_events = [e for e in expected_events if e[0] != "unshard"]
                # However, the post-backward reshards, so the second set of
                # unshards will run as real ops
            expected_events += [
                # Repeat the same pattern except with the root's post-backward
                # at the end since the final callback runs
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ("post_backward", "", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

    def _test_backward_prefetch_unused_in_backward(
        self, reshard_after_forward: bool | int | None
    ):
        # [FUNCTION DOC] HELPER: Runs a model with a shared linear followed by a module splitting
        # into two linears, running backward through one branch with retained graph and then the
        # other with last-backward set. Each linear and the root are sharded with the given
        # reshard_after_forward. Asserts the recorded forward events, then the backward events
        # including a mistargeted prefetch of the unused branch (its post-backward is a no-op)
        # and that its already-unsharded state avoids a second unshard in the next backward.
        """
        Test a model with a linear module then a split into two linear modules,
        where we run backward through one path first before the other, meaning
        that (1) only one linear of the two split is used per backward and (2)
        the initial shared linear is used in both backwards.
        """
        dim = 8
        model = nn.Sequential(nn.Linear(dim, dim), DoubleLinear(dim))
        fully_shard(model[0], reshard_after_forward=reshard_after_forward)
        fully_shard(model[1].lin1, reshard_after_forward=reshard_after_forward)
        fully_shard(model[1].lin2, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        inp = torch.randn((4, dim), device=device_type.type)
        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        with (
            patch_unshard(unshard_with_record),
            patch_post_backward(post_backward_with_record),
        ):
            loss1, loss2 = model(inp)
            expected_events = [
                # Root has no parameters, so it does not have an unshard
                ("unshard", "0", TrainingState.FORWARD),
                ("unshard", "1.lin1", TrainingState.FORWARD),
                ("unshard", "1.lin2", TrainingState.FORWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

            model.set_is_last_backward(False)
            loss2.sum().backward(retain_graph=True)
            expected_events = [
                ("unshard", "1.lin2", TrainingState.PRE_BACKWARD),
                # NOTE: This `1.lin1` unshard is a mistargeted prefetch.
                ("unshard", "1.lin1", TrainingState.PRE_BACKWARD),
                ("post_backward", "1.lin2", TrainingState.POST_BACKWARD),
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
                # `1.lin1` post-backward hook runs but is a no-op
                ("post_backward", "1.lin1", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

            model.set_is_last_backward(True)
            loss1.sum().backward()
            expected_events = [
                # NOTE: `1.lin1` is already unsharded from the mistargeted
                # prefetch in the first backward.
                # Prefetch `0`
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "1.lin1", TrainingState.POST_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
                # `1.lin2` post-backward hook runs but is a no-op
                ("post_backward", "1.lin2", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

    @skip_if_lt_x_gpu(2)
    def test_set_modules_to_forward_prefetch(self):
        # [FUNCTION DOC] Checks set_modules_to_forward_prefetch on a 4-layer checkpointed
        # transformer (reshard_after_forward True). With each layer prefetching the next 1, then
        # next 2 layers, records unshard and reshard events and asserts the expected forward
        # order (prefetched unshards happen early and reshards later), and that backward follows
        # the default backward prefetch order in both cases. Needs at least 2 GPUs.
        n_layers = 4
        reshard_after_forward = True
        checkpoint_impl = "utils"
        model, _, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )

        def set_forward_prefetch(model: Transformer, num_to_prefetch: int) -> None:
            # [FUNCTION DOC] HELPER: For each layer that has at least num_to_prefetch layers
            # after it, sets its forward prefetch list to the following num_to_prefetch layers.
            # Use model-specific knowledge to configure forward prefetching:
            # each transformer block (layer) prefetches for the next few
            for i, layer in enumerate(model.layers):
                if i >= len(model.layers) - num_to_prefetch:
                    break
                layers_to_prefetch = [
                    model.layers[i + j] for j in range(1, num_to_prefetch + 1)
                ]
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)

        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        reshard_with_record = self._get_reshard_with_record(
            FSDPParamGroup.reshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        expected_backward_events = [
            # Default backward prefetching
            ("unshard", "", TrainingState.PRE_BACKWARD),
            ("unshard", "layers.3", TrainingState.PRE_BACKWARD),
            ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
            ("reshard", "layers.3", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.3", TrainingState.POST_BACKWARD),
            ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
            ("reshard", "layers.2", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
            ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
            ("reshard", "layers.1", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
            ("reshard", "layers.0", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
            ("reshard", "", TrainingState.POST_BACKWARD),
            ("post_backward", "", TrainingState.POST_BACKWARD),
        ]
        with (
            patch_unshard(unshard_with_record),
            patch_reshard(reshard_with_record),
            patch_post_backward(post_backward_with_record),
        ):
            set_forward_prefetch(model, num_to_prefetch=1)
            loss = model(inp)
            expected_forward_events = [
                ("unshard", "", TrainingState.FORWARD),
                # `layers.i` prefetches `layers.i+1`
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1", TrainingState.FORWARD),
                ("reshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.1", TrainingState.FORWARD),
                ("unshard", "layers.3", TrainingState.FORWARD),
                ("reshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.3", TrainingState.FORWARD),
                ("reshard", "", TrainingState.FORWARD),
            ]
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            self.assertEqual(events, expected_backward_events)
            events.clear()

            set_forward_prefetch(model, num_to_prefetch=2)
            loss = model(inp)
            expected_forward_events = [
                ("unshard", "", TrainingState.FORWARD),
                # `layers.i` prefetches `layers.i+1` and `layers.i+2`
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1", TrainingState.FORWARD),
                ("unshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.3", TrainingState.FORWARD),
                ("reshard", "layers.1", TrainingState.FORWARD),
                ("reshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.3", TrainingState.FORWARD),
                ("reshard", "", TrainingState.FORWARD),
            ]
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            self.assertEqual(events, expected_backward_events)
            events.clear()

    @skip_if_lt_x_gpu(2)
    def test_set_modules_to_backward_prefetch(self):
        # [FUNCTION DOC] Checks set_modules_to_backward_prefetch on a 4-layer checkpointed
        # transformer (reshard_after_forward True). Asserts the default forward event order, and
        # for each layer prefetching the previous 1 and then previous 2 layers asserts the exact
        # backward sequence of unshard, reshard and post-backward events (the one-layer case
        # equals the default order). Needs at least 2 GPUs.
        n_layers = 4
        reshard_after_forward = True
        checkpoint_impl = "utils"
        model, _, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )

        def set_backward_prefetch(model: Transformer, num_to_prefetch: int) -> None:
            # [FUNCTION DOC] HELPER: For each layer with index at least num_to_prefetch, sets its
            # backward prefetch list to the preceding num_to_prefetch layers, nearest first.
            # Use model-specific knowledge to configure backward prefetching:
            # each transformer block (layer) prefetches for the previous few
            for i, layer in enumerate(model.layers):
                if i < num_to_prefetch:
                    continue
                layers_to_prefetch = [
                    model.layers[i - j] for j in range(1, num_to_prefetch + 1)
                ]
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)

        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        reshard_with_record = self._get_reshard_with_record(
            FSDPParamGroup.reshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        expected_forward_events = [
            # Default forward prefetching
            ("unshard", "", TrainingState.FORWARD),  # root
            ("unshard", "layers.0", TrainingState.FORWARD),
            ("reshard", "layers.0", TrainingState.FORWARD),
            ("unshard", "layers.1", TrainingState.FORWARD),
            ("reshard", "layers.1", TrainingState.FORWARD),
            ("unshard", "layers.2", TrainingState.FORWARD),
            ("reshard", "layers.2", TrainingState.FORWARD),
            ("unshard", "layers.3", TrainingState.FORWARD),
            ("reshard", "layers.3", TrainingState.FORWARD),
            ("reshard", "", TrainingState.FORWARD),
        ]
        with (
            patch_unshard(unshard_with_record),
            patch_reshard(reshard_with_record),
            patch_post_backward(post_backward_with_record),
        ):
            set_backward_prefetch(model, num_to_prefetch=1)
            loss = model(inp)
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            expected_backward_events = [
                ("unshard", "", TrainingState.PRE_BACKWARD),
                # Root prefetches `layers.3` per default
                ("unshard", "layers.3", TrainingState.PRE_BACKWARD),
                # `layers.i` prefetches for `layers.i-1` (same as default)
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.3", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.3", TrainingState.POST_BACKWARD),
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.2", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.1", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                ("reshard", "layers.0", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ("reshard", "", TrainingState.POST_BACKWARD),
                ("post_backward", "", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_backward_events)
            events.clear()

            set_backward_prefetch(model, num_to_prefetch=2)
            loss = model(inp)
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            expected_backward_events = [
                ("unshard", "", TrainingState.PRE_BACKWARD),
                # Root prefetches `layers.3` per default
                ("unshard", "layers.3", TrainingState.PRE_BACKWARD),
                # `layers.i` prefetches for `layers.i-1` and `layers.i-2`
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.3", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.3", TrainingState.POST_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.2", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                ("reshard", "layers.1", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                ("reshard", "layers.0", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ("reshard", "", TrainingState.POST_BACKWARD),
                ("post_backward", "", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_backward_events)
            events.clear()

    @skip_if_lt_x_gpu(2)
    def test_set_modules_to_backward_prefetch_inside_ac(self):
        # [FUNCTION DOC] Checks explicit backward prefetch when activation checkpoint wrappers
        # are used for blocks and feed-forwards, with fully_shard on embeddings, each
        # feed-forward w1, each layer, norm and output together, and the root. Asserts the
        # default backward event order first, then after configuring a DeepSeek-style prefetch
        # (output prefetches the last layer; each layer prefetches the previous layer and that
        # layer's w1; the first layer prefetches the embeddings) asserts the new unshard and
        # reshard order including the checkpoint recompute unshard in FORWARD state. Needs at
        # least 2 GPUs.
        n_layers = 3
        reshard_after_forward = True
        # use checkpoint wrapper instead of torch.utils
        model_args = ModelArgs(
            n_layers=n_layers, checkpoint_activations=False, weight_tying=False
        )
        model = Transformer(model_args)
        apply_activation_checkpointing(
            model, check_fn=lambda m: isinstance(m, TransformerBlock)
        )
        apply_activation_checkpointing(
            model, check_fn=lambda m: isinstance(m, FeedForward)
        )
        fully_shard([model.tok_embeddings, model.pos_embeddings])
        for layer in model.layers:
            # mimic fully_shard(layer.moe.experts)
            fully_shard(
                layer.feed_forward.w1, reshard_after_forward=reshard_after_forward
            )
            fully_shard(layer, reshard_after_forward=reshard_after_forward)
        fully_shard(
            [model.norm, model.output], reshard_after_forward=reshard_after_forward
        )
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        inp = torch.randint(
            0,
            model_args.vocab_size,
            (2, model_args.max_seq_len),
            device=device_type.type,
        )

        def set_backward_prefetch(model: Transformer) -> None:
            # [FUNCTION DOC] HELPER: Asserts (raising AssertionError) that the model, its output
            # and the relevant blocks, feed-forward w1 and embeddings are FSDP modules, then
            # sets: output prefetches the last layer; each layer prefetches the previous layer
            # and its feed-forward w1 (or the token embeddings for the first layer).
            # tell pyre model.set_modules_to_backward_prefetch is available
            if not isinstance(model, FSDPModule):
                raise AssertionError(f"Expected FSDPModule, got {type(model)}")
            if not isinstance(model.output, FSDPModule):
                raise AssertionError(f"Expected FSDPModule, got {type(model.output)}")

            # mimic deepseek MOE
            # prefetch layer - 1 and its feedforward before cpu sync during a2a
            reversed_transformer_blocks = list(reversed(model.layers))
            prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

            if (
                model.norm is not None
                and model.output is not None
                and len(model.layers) > 0
            ):
                if not isinstance(reversed_transformer_blocks[0], FSDPModule):
                    raise AssertionError(
                        f"Expected FSDPModule, got {type(reversed_transformer_blocks[0])}"
                    )
                model.output.set_modules_to_backward_prefetch(
                    [reversed_transformer_blocks[0]]
                )

            for transformer_block, prev_transformer_block in zip(
                reversed_transformer_blocks, prev_transformer_blocks
            ):
                if not isinstance(transformer_block, FSDPModule):
                    raise AssertionError(
                        f"Expected FSDPModule, got {type(transformer_block)}"
                    )
                if prev_transformer_block is not None:
                    if not isinstance(prev_transformer_block, FSDPModule):
                        raise AssertionError(
                            f"Expected FSDPModule, got {type(prev_transformer_block)}"
                        )
                    if not hasattr(prev_transformer_block.feed_forward, "w1"):
                        raise AssertionError(
                            "Expected prev_transformer_block.feed_forward to have 'w1' attribute"
                        )
                    if not isinstance(
                        prev_transformer_block.feed_forward.w1, FSDPModule
                    ):
                        raise AssertionError(
                            f"Expected FSDPModule, got {type(prev_transformer_block.feed_forward.w1)}"
                        )
                    transformer_block.set_modules_to_backward_prefetch(
                        [
                            prev_transformer_block,
                            prev_transformer_block.feed_forward.w1,
                        ]
                    )
                elif model.tok_embeddings is not None:
                    if not isinstance(model.tok_embeddings, FSDPModule):
                        raise AssertionError(
                            f"Expected FSDPModule, got {type(model.tok_embeddings)}"
                        )
                    transformer_block.set_modules_to_backward_prefetch(
                        [model.tok_embeddings]
                    )

        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        reshard_with_record = self._get_reshard_with_record(
            FSDPParamGroup.reshard, events
        )
        with (
            patch_unshard(unshard_with_record),
            patch_reshard(reshard_with_record),
        ):
            loss = model(inp)
            events.clear()
            loss.sum().backward()
            expected_backward_events = [
                ("unshard", "norm, output", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("reshard", "norm, output", TrainingState.POST_BACKWARD),
                # layers.2 prefetch w1
                (
                    "unshard",
                    "layers.2._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.PRE_BACKWARD,
                ),
                # layers.2.w1 prefetch layers.1
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                (
                    "reshard",
                    "layers.2._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.POST_BACKWARD,
                ),
                ("reshard", "layers.2", TrainingState.POST_BACKWARD),
                (
                    "unshard",
                    "layers.1._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.PRE_BACKWARD,
                ),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                (
                    "reshard",
                    "layers.1._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.POST_BACKWARD,
                ),
                ("reshard", "layers.1", TrainingState.POST_BACKWARD),
                (
                    "unshard",
                    "layers.0._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.PRE_BACKWARD,
                ),
                (
                    "unshard",
                    "tok_embeddings, pos_embeddings",
                    TrainingState.PRE_BACKWARD,
                ),
                (
                    "reshard",
                    "layers.0._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.POST_BACKWARD,
                ),
                ("reshard", "layers.0", TrainingState.POST_BACKWARD),
                (
                    "reshard",
                    "tok_embeddings, pos_embeddings",
                    TrainingState.POST_BACKWARD,
                ),
            ]
            self.assertEqual(events, expected_backward_events)
            events.clear()

            set_backward_prefetch(model)
            loss = model(inp)
            events.clear()
            loss.sum().backward()
            expected_backward_events = [
                ("unshard", "norm, output", TrainingState.PRE_BACKWARD),
                # root explicit prefetch layers.2
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("reshard", "norm, output", TrainingState.POST_BACKWARD),
                # layers.2 prefetch layers.1 and feed_forward
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                (
                    "unshard",
                    "layers.1._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.PRE_BACKWARD,
                ),
                # AC recompute_fn
                (
                    "unshard",
                    "layers.2._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.FORWARD,
                ),
                (
                    "reshard",
                    "layers.2._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.POST_BACKWARD,
                ),
                ("reshard", "layers.2", TrainingState.POST_BACKWARD),
                # layers.1 prefetch layers.0
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                (
                    "unshard",
                    "layers.0._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.PRE_BACKWARD,
                ),
                (
                    "reshard",
                    "layers.1._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.POST_BACKWARD,
                ),
                ("reshard", "layers.1", TrainingState.POST_BACKWARD),
                # layers.0 prefetch embeddings
                (
                    "unshard",
                    "tok_embeddings, pos_embeddings",
                    TrainingState.PRE_BACKWARD,
                ),
                (
                    "reshard",
                    "layers.0._checkpoint_wrapped_module.feed_forward._checkpoint_wrapped_module.w1",
                    TrainingState.POST_BACKWARD,
                ),
                ("reshard", "layers.0", TrainingState.POST_BACKWARD),
                (
                    "reshard",
                    "tok_embeddings, pos_embeddings",
                    TrainingState.POST_BACKWARD,
                ),
            ]
            self.assertEqual(events, expected_backward_events)
            events.clear()

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_multi_module_backward_prefetch(self):
        # [FUNCTION DOC] Checks backward prefetch with multi-module FSDP groups on a 5-layer
        # checkpointed transformer (layer 0 alone, layers 1-2 and 3-4 grouped, embeddings
        # grouped, norm plus output grouped without reshard after forward). For 3 iterations
        # asserts the forward unshard order, then runs backward and steps the optimizer.
        # KNOWN ISSUE: the expected backward event list is built but never compared with the
        # recorded events (the assertion is missing), so backward ordering is not actually
        # checked. Needs at least 2 GPUs.
        n_layers = 5
        model_args = ModelArgs(
            n_layers=n_layers, checkpoint_activations=True, weight_tying=False
        )
        model = Transformer(model_args)
        for i in range(n_layers):
            if i == 0:
                fully_shard(model.layers[i])
            elif i % 2 == 1:
                fully_shard([model.layers[i], model.layers[i + 1]])
        fully_shard([model.tok_embeddings, model.pos_embeddings])
        fully_shard([model.norm, model.output], reshard_after_forward=False)
        fully_shard(model)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        inp = torch.randint(
            0,
            model_args.vocab_size,
            (2, model_args.max_seq_len),
            device=device_type.type,
        )
        with (
            patch_unshard(unshard_with_record),
            patch_post_backward(post_backward_with_record),
        ):
            for _ in range(3):
                loss = model(inp)
                expected_events = [
                    (
                        "unshard",
                        "tok_embeddings, pos_embeddings",
                        TrainingState.FORWARD,
                    ),
                    ("unshard", "layers.0", TrainingState.FORWARD),
                    ("unshard", "layers.1, layers.2", TrainingState.FORWARD),
                    ("unshard", "layers.3, layers.4", TrainingState.FORWARD),
                    ("unshard", "norm, output", TrainingState.FORWARD),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                loss.sum().backward()
                expected_events = [
                    # (norm, output) does not reshard after forward, so there is
                    # no unshard to begin backward
                    ("unshard", "layers.3, layers.4", TrainingState.PRE_BACKWARD),
                    ("post_backward", "norm, output", TrainingState.POST_BACKWARD),
                    ("unshard", "layers.1, layers.2", TrainingState.PRE_BACKWARD),
                    (
                        "post_backward",
                        "layers.3, layers.4",
                        TrainingState.POST_BACKWARD,
                    ),
                    ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                    (
                        "post_backward",
                        "layers.1, layers.2",
                        TrainingState.POST_BACKWARD,
                    ),
                    (
                        "unshard",
                        "tok_embeddings, pos_embeddings",
                        TrainingState.PRE_BACKWARD,
                    ),
                    ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                    (
                        "post_backward",
                        "tok_embeddings, pos_embeddings",
                        TrainingState.POST_BACKWARD,
                    ),
                ]
                events.clear()
                optim.step()
                optim.zero_grad()

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_multi_module_unused_module(self):
        # [FUNCTION DOC] Checks FSDP groups that include a module never run in forward. Two
        # modules each have an unused Linear grouped with the used one, followed by a final
        # Linear and root. For 3 iterations asserts the forward unshard events, then asserts the
        # backward events: the incomplete groups are force-completed, their pre-backward
        # unshards happen at backward start, only the last group's post-backward fires via
        # autograd and the root and first group post-backwards fire in the final callback. Needs
        # at least 2 GPUs.
        class ModuleWithUnusedLinear(nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear that forward never uses (unused_lin) and a
            # 16 by 16 Linear (lin); forward applies lin then relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the unused 1 by 1 Linear and the used 16
                # by 16 Linear.
                super().__init__()
                self.unused_lin = nn.Linear(1, 1)
                self.lin = nn.Linear(16, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns relu of the used Linear applied to x;
                # unused_lin is skipped.
                return nn.functional.relu(self.lin(x))

        model = nn.Sequential(
            ModuleWithUnusedLinear(), ModuleWithUnusedLinear(), nn.Linear(16, 16)
        )
        fully_shard([model[0].unused_lin, model[0].lin], reshard_after_forward=True)
        fully_shard([model[1].unused_lin, model[1].lin], reshard_after_forward=True)
        fully_shard(model)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        events: list[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        inp = torch.randn((2, 16), device=device_type.type)
        with (
            patch_unshard(unshard_with_record),
            patch_post_backward(post_backward_with_record),
        ):
            for _ in range(3):
                loss = model(inp)
                expected_events = [
                    ("unshard", "", TrainingState.FORWARD),
                    ("unshard", "0.unused_lin, 0.lin", TrainingState.FORWARD),
                    ("unshard", "1.unused_lin, 1.lin", TrainingState.FORWARD),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                loss.sum().backward()
                # `_force_complete_incomplete_states` completes post-forward
                # for the incomplete groups (both `model[0]` and `model[1]`
                # have `unused_lin` that never ran forward), reshards, and
                # registers pre-backward hooks on the root output; those
                # hooks fire at backward start (unshard events). Only
                # group1's post_backward fires via autograd (model[1]'s
                # input requires grad); root and group0's post_backwards
                # fire in the final callback in all-states pre-order.
                expected_events = [
                    ("unshard", "1.unused_lin, 1.lin", TrainingState.PRE_BACKWARD),
                    ("unshard", "0.unused_lin, 0.lin", TrainingState.PRE_BACKWARD),
                    (
                        "post_backward",
                        "1.unused_lin, 1.lin",
                        TrainingState.POST_BACKWARD,
                    ),
                    ("post_backward", "", TrainingState.POST_BACKWARD),
                    (
                        "post_backward",
                        "0.unused_lin, 0.lin",
                        TrainingState.POST_BACKWARD,
                    ),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                optim.step()
                optim.zero_grad()

    @skip_if_lt_x_gpu(2)
    def test_backward_misprefetch(self):
        # [FUNCTION DOC] Checks that a misprefetch (in_proj prefetching out_proj, which
        # backward no longer needs) does not break correctness. Shards in_proj, out_proj and
        # the root of an MLP, sets that backward prefetch, then for 3 steps compares losses
        # against an unsharded reference whose gradients are averaged with an all-reduce. Needs
        # at least 2 GPUs.
        torch.manual_seed(42)
        model = MLP(dim=16, device=device_type)
        ref_model = copy.deepcopy(model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard(model.in_proj)
        fully_shard(model.out_proj)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Backward should run through `out_proj` -> `in_proj`, so if `in_proj`
        # prefetches for `out_proj`, then this is a misprefetch, as `out_proj`
        # should not be needed anymore for backward.
        model.in_proj.set_modules_to_backward_prefetch([model.out_proj])

        torch.manual_seed(self.rank + 1)
        inp = torch.randn((2, 16), device=device_type.type)
        for _ in range(3):
            ref_optim.zero_grad()
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            ref_optim.step()
            optim.zero_grad()
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            self.assertEqual(ref_loss, loss)

    def _init_transformer(
        self,
        n_layers: int,
        reshard_after_forward: bool | int | None,
        checkpoint_impl: str | None,
    ):
        # [FUNCTION DOC] HELPER: Builds a transformer with n_layers (using utils checkpointing
        # when requested), optionally applies composable checkpoint to each block, and
        # fully_shards each block and the root with the given reshard_after_forward. Returns the
        # model, an Adam optimizer and a random integer token input of batch 2.
        model_args = ModelArgs(
            n_layers=n_layers, checkpoint_activations=(checkpoint_impl == "utils")
        )
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                if checkpoint_impl == "composable":
                    checkpoint(module)
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        inp = torch.randint(
            0,
            model_args.vocab_size,
            (2, model_args.max_seq_len),
            device=device_type.type,
        )
        return model, optim, inp

    def _get_unshard_with_record(
        self, orig_unshard: Callable, events: list[EventType]
    ) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a replacement for FSDPParamGroup.unshard that appends an
        # event of kind unshard, module fqn and training state to events when it is not a no-op
        # (no pending all-gather result and not already unsharded), then calls the original.
        def unshard_with_record(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Recording wrapper for unshard; takes the param group as
            # self. Skips recording for no-op calls and always delegates to the original,
            # returning its result.
            nonlocal events
            if (
                self._all_gather_result is None
                and self._sharded_state != ShardedState.UNSHARDED
            ):  # skip no-ops
                events.append(("unshard", self._module_fqn, self._training_state))
            return orig_unshard(self, *args, **kwargs)

        return unshard_with_record

    def _get_reshard_with_record(
        self, orig_reshard: Callable, events: list[EventType]
    ) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a replacement for FSDPParamGroup.reshard that
        # records reshard events (module fqn and training state) into events and delegates
        # to the original.
        def reshard_with_record(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Recording wrapper for reshard; takes the param group as
            # self. During FORWARD with reshard_after_forward disabled it returns
            # immediately without recording or resharding; otherwise records the event then
            # delegates to the original.
            nonlocal events
            if (
                self._training_state == TrainingState.FORWARD
                and not self._reshard_after_forward
            ):  # skip no-ops
                return
            events.append(("reshard", self._module_fqn, self._training_state))
            return orig_reshard(self, *args, **kwargs)

        return reshard_with_record

    def _get_post_backward_with_record(
        self, orig_post_backward: Callable, events: list[EventType]
    ) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a replacement for FSDPParamGroup.post_backward that runs
        # the original first and then records a post_backward event (module fqn and the training
        # state after the call) into events.
        def post_backward_with_record(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Recording wrapper for post_backward; runs the
            # original, records the resulting state (which should be POST_BACKWARD) and
            # returns the original result.
            nonlocal events
            ret = orig_post_backward(self, *args, **kwargs)
            # Use training state after running post-backward to check that the
            # state is transitioned to `POST_BACKWARD` as expected
            events.append(("post_backward", self._module_fqn, self._training_state))
            return ret

        return post_backward_with_record


class TestFullyShardUnshardMultiProcess(FSDPTest):
    # [CLASS DOC] Multi-process test of the asynchronous unshard API of FSDP modules overlapped
    # with non-differentiable async all-reduces. world_size is min(device count, 2).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns min(device count for the test device type, 2).
        return min(torch.get_device_module(device_type).device_count(), 2)

    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/149349")
    @skip_if_lt_x_gpu(2)
    def test_unshard_async(self):
        # [FUNCTION DOC] Checks that calling unshard with async_op on FSDP modules between
        # launching async all-reduces gives the same training result as a replicated reference.
        # Defines a model with three reduce modules (matmul, relu, async all-reduce on the mesh
        # group) feeding three MLPs that wait on each work object; the FSDP model shards each MLP
        # (no reshard after forward) and the container, then both models train 10 steps with Adam
        # (implicit replication for the FSDP optimizer) and losses are asserted equal. Skipped
        # under torch inductor (issue 149349). Needs at least 2 GPUs.
        class ReduceModule(nn.Module):
            # [CLASS DOC] HELPER: Module with a random square weight whose forward applies relu
            # of the matmul and launches an asynchronous all-reduce of the result on the mesh
            # process group, returning both the tensor and the work handle.
            def __init__(self, dim: int, mesh: DeviceMesh):
                # [FUNCTION DOC] HELPER: Stores the mesh and creates a dim by dim
                # random weight parameter.
                super().__init__()
                self.mesh = mesh
                self.weight = nn.Parameter(torch.randn(dim, dim))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns relu of x times the weight and the handle of
                # an async all-reduce on that result (the all-reduce is not differentiable and
                # only exercises overlap).
                y = F.relu(x @ self.weight)
                # NOTE: This all-reduce is not differentiable and is included
                # to exercise the overlap.
                work = dist.all_reduce(y, group=self.mesh.get_group(), async_op=True)
                return y, work

        class MLPs(nn.Module):
            # [CLASS DOC] HELPER: Holds three MLPs (mlp1 to mlp3); forward waits for the
            # matching work handle before applying each MLP to the matching input and returns
            # the sum of the outputs.
            def __init__(self, dim: int):
                # [FUNCTION DOC] HELPER: Creates three MLPs of the given dim.
                super().__init__()
                self.mlp1 = MLP(dim)
                self.mlp2 = MLP(dim)
                self.mlp3 = MLP(dim)

            def forward(self, ys: list[torch.Tensor], works: list[dist.Work]):
                # [FUNCTION DOC] HELPER: Given three tensors and three work handles, waits for
                # each handle, applies the matching MLP and returns the sum of the three results.
                (y1, y2, y3), (work1, work2, work3) = ys, works
                work1.wait()
                z1 = self.mlp1(y1)
                work2.wait()
                z2 = self.mlp2(y2)
                work3.wait()
                z3 = self.mlp3(y3)
                return z1 + z2 + z3

        class ReduceModel(nn.Module):
            # [CLASS DOC] HELPER: Model of three ReduceModules followed by MLPs; forward
            # interleaves them so each FSDP MLP is asynchronously unshard-ed while later
            # reduce modules run.
            def __init__(self, dim: int, mesh: DeviceMesh):
                # [FUNCTION DOC] HELPER: Creates three reduce modules sharing the mesh and
                # the MLPs container.
                super().__init__()
                self.reduce_module1 = ReduceModule(dim, mesh)
                self.reduce_module2 = ReduceModule(dim, mesh)
                self.reduce_module3 = ReduceModule(dim, mesh)
                self.mlps = MLPs(dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Runs each reduce module in turn; after each one, if
                # the matching MLP is an FSDP module, starts its asynchronous unshard. Finally
                # feeds the three outputs and work handles through the MLPs container and
                # returns the result.
                y1, work1 = self.reduce_module1(x)
                if isinstance(self.mlps.mlp1, FSDPModule):
                    self.mlps.mlp1.unshard(async_op=True)
                y2, work2 = self.reduce_module2(x)
                if isinstance(self.mlps.mlp2, FSDPModule):
                    self.mlps.mlp2.unshard(async_op=True)
                y3, work3 = self.reduce_module3(x)
                if isinstance(self.mlps.mlp3, FSDPModule):
                    self.mlps.mlp3.unshard(async_op=True)
                return self.mlps([y1, y2, y3], [work1, work2, work3])

        mesh = init_device_mesh(device_type.type, (self.world_size,))
        batch_size, dim = 2, 8
        torch.manual_seed(42)
        ref_model = replicate(ReduceModel(dim, mesh).to(device_type))
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        torch.manual_seed(42)
        model = ReduceModel(dim, mesh)
        fully_shard(model.mlps.mlp1, reshard_after_forward=False)
        fully_shard(model.mlps.mlp2, reshard_after_forward=False)
        fully_shard(model.mlps.mlp3, reshard_after_forward=False)
        fully_shard(model.mlps)
        replicate(model.to(device_type))
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((batch_size, dim), device=device_type.type)
        for _ in range(10):
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                losses.append(_model(inp).sum())
                losses[-1].backward()
                with implicit_replication():
                    _optim.step()
                _optim.zero_grad()
            self.assertEqual(losses[0], losses[1])


class TestFullyShardUnshardMultiThread(FSDPTestMultiThread):
    # [CLASS DOC] Multi-thread tests (world size 2) of the unshard API edge cases.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: World size of 2.
        return 2

    def test_unshard_no_param_group(self):
        # [FUNCTION DOC] Checks that unshard with async_op on a root module that manages no
        # parameters of its own (only child FSDP modules) returns a handle that can be waited
        # on without error.
        # Check that we can call `unshard()` on a module with no parameter
        # group / no managed parameters without erroring
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        for lin in model:
            fully_shard(lin)
        fully_shard(model)
        handle = model.unshard(async_op=True)
        handle.wait()

    def test_unshard_without_lazy_init(self):
        # [FUNCTION DOC] Checks that calling unshard before the first forward (so before lazy
        # init) works: builds an MLP with parameters broadcast from rank 0, shards it, calls
        # unshard and asserts the parameters equal an unsharded copy.
        torch.manual_seed(42)
        model = MLP(4)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(model)
        model.unshard()  # no lazy init yet
        for ref_param, param in zip(ref_model.parameters(), model.parameters()):
            self.assertEqual(ref_param, param)


class TestFullyShardAllocFromPG(FSDPTest):
    # [CLASS DOC] Multi-process tests that FSDP2 allocates communication buffers from the NCCL
    # process group when requested, detected through NCCL debug logs written to a temporary
    # directory. MEMORY_REGISTER_RE is the log line pattern for buffer registration, which may
    # change with NCCL versions.
    # The messages might change when we move to a different NCCL version.
    # Please update this test if it starts failing.
    MEMORY_REGISTER_RE = (
        "NCCL INFO register comm 0x[0-9a-f]+ buffer 0x[0-9a-f]+ size [0-9]+"
    )

    @classmethod
    def _run(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Test runner hook executed per spawned process: creates a
        # temporary directory for the NCCL log, sets NCCL_DEBUG, NCCL_DEBUG_SUBSYS and
        # NCCL_DEBUG_FILE environment variables in this process (side effect on os.environ) and
        # runs the parent implementation.
        cls.nccl_log_dir = tempfile.TemporaryDirectory()
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,REG"
        os.environ["NCCL_DEBUG_FILE"] = cls.nccl_log_dir.name + "/nccl_log"
        super()._run(*args, **kwargs)

    @skip_if_lt_x_gpu(2)
    # The NCCL PG refuses to allocate tensors if multicast is unavailable, see
    # https://github.com/pytorch/pytorch/blob/503362d019b3782581492af7767945dbd75ca1c9/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L5634
    def test_fully_shard_alloc_from_pg(self):
        # [FUNCTION DOC] Checks the allocate-memory-from-process-group option. Skips if multicast
        # is not supported. Runs a forward and backward and asserts the NCCL log has no buffer
        # registration line; then enables the option on all blocks and the root, runs again, and
        # asserts a registration line now appears. The multicast check is inside the test to
        # avoid initializing the symmetric memory allocator at import. Needs at least 2 GPUs.
        # Run this check inside test instead of using @requires_multicast_support().
        # The decorator would trigger an initialization of SymmMem allocator
        # when Python statically initializes classes in this file, causing
        # SymmMem to fix the allocate backend to "CUDA". This is unfriendly for
        # other tests in this file that requires NCCL backend
        if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0):
            self.skipTest("multicast support is not available")

        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fully_shard(model)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")

        loss = model(inp)
        loss.sum().backward()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            self.assertNotRegex(f.read(), self.MEMORY_REGISTER_RE)

        for module in model.modules():
            if isinstance(module, TransformerBlock):
                module.set_allocate_memory_from_process_group_for_comm(True)
        model.set_allocate_memory_from_process_group_for_comm(True)

        loss = model(inp)
        loss.sum().backward()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            self.assertRegex(f.read(), self.MEMORY_REGISTER_RE)

    @skip_if_lt_x_gpu(2)
    def test_exception_when_used_together_with_comm_hooks(self):
        # [FUNCTION DOC] Checks that enabling allocation from the process group is accepted, that
        # a custom all-gather can still be set afterwards, and that enabling allocation after a
        # custom comm is set raises AssertionError. Needs at least 2 GPUs.
        model = nn.Linear(16, 16)
        model = fully_shard(model)
        # ok
        model.set_allocate_memory_from_process_group_for_comm(True)

        # setting custom hook after is also ok
        # (overrides set_allocate_memory_from_process_group_for_comm)
        mock_all_gather = MagicMock(spec=AllGather)
        model.set_custom_all_gather(mock_all_gather)

        # setting this after custom comm is used is ko
        with self.assertRaises(AssertionError):
            model.set_allocate_memory_from_process_group_for_comm(True)


@requires_cuda_p2p_access()
@skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Not enough GPUs to run the test")
@unittest.skipIf(
    not PLATFORM_SUPPORTS_SYMM_MEM, "SymmMem is not supported on this platform"
)
@skipCUDAIf(TEST_WITH_ROCM, "requires NVIDIA GPUs")
@skipCUDAIf(not SM90OrLater, "requires sm90+")
class TestFullyShardSymmMem(MultiProcContinuousTest):
    # [CLASS DOC] Continuous multi-process test (NCCL, NVIDIA GPUs with sm90 or later, p2p
    # access, symmetric memory support) of FSDP2 using symmetric memory for communication.
    # Process group options enable the zero-CTA policy for copy-engine collectives.
    @classmethod
    def backend_str(cls) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the backend name, nccl.
        return "nccl"

    @classmethod
    def opts(cls):
        # [FUNCTION DOC] HELPER: Returns None when NCCL is unavailable; otherwise
        # ProcessGroupNCCL options with the CTA policy set to zero so collectives
        # use copy engines.
        if not dist.is_nccl_available():
            return None
        # Enable Zero-CTA policy for CE collectives
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        return opts

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns the CUDA device of this rank.
        return torch.device("cuda", self.rank)

    @parametrize("sum_reduction", [True, False])
    def test_fully_shard_symm_mem(self, sum_reduction: bool):
        # [FUNCTION DOC] Smoke test: builds a transformer of width 4096 on this rank's GPU,
        # enables symmetric memory for communication and the forced-sum reduction flag (from the
        # parameter) on each block and the root, runs one forward and backward, and synchronizes.
        # Passes if no error is raised.
        torch.manual_seed(42 + self.rank)
        device = torch.device("cuda", self.rank)
        torch.cuda.set_device(device)
        seq_len = 64
        model_args = ModelArgs()
        model_args.dim = 4096
        model_args.max_seq_len = seq_len
        model = Transformer(model_args).to(device)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
                module.set_force_sum_reduction_for_comms(sum_reduction)
                module.set_symm_mem_for_comm()
        fully_shard(model)
        model.set_force_sum_reduction_for_comms(sum_reduction)
        model.set_symm_mem_for_comm()

        bs = 4
        inp = torch.randint(0, model_args.vocab_size, (bs, seq_len), device=device)

        def run():
            # [FUNCTION DOC] HELPER: Runs a forward and a backward on the sum of the loss.
            loss = model(inp)
            loss.sum().backward()

        run()
        torch.cuda.synchronize(device)


instantiate_parametrized_tests(TestFullyShardSymmMem)


class TestFullyShardForceSumReduction(FSDPTest):
    # [CLASS DOC] Multi-process tests that set_force_sum_reduction_for_comms makes reduce-scatter
    # and all-reduce use the SUM op instead of AVG, detected by matching NCCL collective debug
    # logs (the regex differs for ROCm with NCCL 2.27 or later). SUM_REDUCTION is 0 and
    # AVG_REDUCTION is 4 per the NCCL enum.
    # The messages might change when we move to a different NCCL version.
    # Please update this test if it starts failing.

    if TEST_WITH_ROCM and torch.cuda.nccl.version()[:2] >= (2, 27):
        COLLECTIVE_RE = (
            r"NCCL INFO {coll}: opCount [0-9a-f]+ sendbuff 0x[0-9a-f]+ recvbuff 0x[0-9a-f]+ acc \(nil\) "
            "count {count} datatype [0-9]+ op {reduce_op} root [0-9]+ comm 0x[0-9a-f]+"
        )
    else:
        COLLECTIVE_RE = (
            "NCCL INFO {coll}: opCount [0-9a-f]+ sendbuff 0x[0-9a-f]+ recvbuff 0x[0-9a-f]+ "
            "count {count} datatype [0-9]+ op {reduce_op} root [0-9]+ comm 0x[0-9a-f]+"
        )
    # See here for the numerical values for each reduction op:
    # https://github.com/NVIDIA/nccl/blob/72d2432094d6ae36abd6e511c3a16a2d052dbf94/src/nccl.h.in#L260-L275
    SUM_REDUCTION = 0
    AVG_REDUCTION = 4

    @classmethod
    def _run(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Test runner hook executed per spawned process: creates a
        # temporary directory for the NCCL log, sets NCCL_DEBUG, NCCL_DEBUG_SUBSYS
        # (collectives) and NCCL_DEBUG_FILE environment variables (side effect on os.environ)
        # and runs the parent implementation.
        cls.nccl_log_dir = tempfile.TemporaryDirectory()
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "COLL"
        os.environ["NCCL_DEBUG_FILE"] = cls.nccl_log_dir.name + "/nccl_log"
        super()._run(*args, **kwargs)

    # Test reduce-scatter only on plain FSDP on 2 GPUs
    # This test verifies NCCL debug logs and is CUDA-specific.
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not TEST_CUDA, "This test verifies NCCL debug logs and is CUDA-specific"
    )
    def test_fully_shard_force_sum_reduce_scatter(self):
        # [FUNCTION DOC] On plain FSDP, checks reduce-scatter initially uses AVG and not SUM in
        # NCCL logs for a message of the first layer's per-rank element count, then after
        # enabling forced sum reduction on all blocks and root the logs also contain SUM. CUDA
        # only. Needs at least 2 GPUs.
        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fully_shard(model)

        # We target a specific count so that we don't pick up the barrier ops
        layer_numel = sum(w.numel() for w in model.layers[0].parameters())
        comms_size = layer_numel // self.world_size
        reduce_scatter_avg_re = self.COLLECTIVE_RE.format(
            coll="ReduceScatter", count=comms_size, reduce_op=self.AVG_REDUCTION
        )
        reduce_scatter_sum_re = self.COLLECTIVE_RE.format(
            coll="ReduceScatter", count=comms_size, reduce_op=self.SUM_REDUCTION
        )

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")

        loss = model(inp)
        loss.sum().backward()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            logs = f.read()
        # At this stage we should have only AVG, no SUM
        self.assertRegex(logs, reduce_scatter_avg_re)
        self.assertNotRegex(logs, reduce_scatter_sum_re)

        for module in model.modules():
            if isinstance(module, TransformerBlock):
                module.set_force_sum_reduction_for_comms(True)
        model.set_force_sum_reduction_for_comms(True)

        loss = model(inp)
        loss.sum().backward()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            logs = f.read()
        # Now we should also have SUM
        self.assertRegex(logs, reduce_scatter_sum_re)

    # Test both reduce-scatter and all-reduce on HSDP (DDP+FSDP) on 4 GPUs
    # This test verifies NCCL debug logs and is CUDA-specific.
    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(
        not TEST_CUDA, "This test verifies NCCL debug logs and is CUDA-specific"
    )
    def test_fully_shard_force_sum_both_reductions(self):
        # [FUNCTION DOC] On a 2D (ddp, fsdp) HSDP mesh, checks reduce-scatter and all-reduce both
        # use AVG and not SUM at first, and after enabling forced sum reduction the logs also
        # contain SUM for both collectives. CUDA only. Needs at least 4 GPUs.
        mesh = init_device_mesh(
            device_type.type, (2, self.world_size // 2), mesh_dim_names=("ddp", "fsdp")
        )

        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)

        # We target a specific count so that we don't pick up the barrier ops
        layer_numel = sum(w.numel() for w in model.layers[0].parameters())
        comms_size = layer_numel // (self.world_size // 2)
        reduce_scatter_avg_re = self.COLLECTIVE_RE.format(
            coll="ReduceScatter", count=comms_size, reduce_op=self.AVG_REDUCTION
        )
        reduce_scatter_sum_re = self.COLLECTIVE_RE.format(
            coll="ReduceScatter", count=comms_size, reduce_op=self.SUM_REDUCTION
        )
        all_reduce_avg_re = self.COLLECTIVE_RE.format(
            coll="AllReduce", count=comms_size, reduce_op=self.AVG_REDUCTION
        )
        all_reduce_sum_re = self.COLLECTIVE_RE.format(
            coll="AllReduce", count=comms_size, reduce_op=self.SUM_REDUCTION
        )

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")

        loss = model(inp)
        loss.sum().backward()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            logs = f.read()
        # At this stage we should have only AVG, no SUM
        self.assertRegex(logs, reduce_scatter_avg_re)
        self.assertRegex(logs, all_reduce_avg_re)
        self.assertNotRegex(logs, reduce_scatter_sum_re)
        self.assertNotRegex(logs, all_reduce_sum_re)

        for module in model.modules():
            if isinstance(module, TransformerBlock):
                module.set_force_sum_reduction_for_comms(True)
        model.set_force_sum_reduction_for_comms(True)

        loss = model(inp)
        loss.sum().backward()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            logs = f.read()
        # Now we should also have SUM
        self.assertRegex(logs, reduce_scatter_sum_re)
        self.assertRegex(logs, all_reduce_sum_re)


class TestFullyShardReduceOpWorldSize1(FSDPTest):
    # [CLASS DOC] Single-rank test that the gradient reduce op is SUM when the shard group
    # has size 1.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: World size of 1.
        return 1

    def test_size1_reduceop(self):
        # [FUNCTION DOC] With world size 1, trains a Linear with fully_shard (mesh of 1 rank, no
        # reshard after forward) against a reference that all-reduces gradients with SUM,
        # asserting equal losses and bias gradients for 3 steps. Then asserts the all-reduce op
        # returned by the gradient divide factor helper for the shard group and float32 is SUM.
        from torch.distributed.distributed_c10d import ReduceOp

        model = nn.Linear(1024, 1025)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters())
        fully_shard(
            model,
            mesh=init_device_mesh(device_type.type, (1,)),
            reshard_after_forward=False,
        )
        optim = torch.optim.Adam(model.parameters())

        inp = torch.randn(1025, 1024, device=device_type.type)
        for _ in range(3):
            ref_optim.zero_grad()
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            ref_optim.step()

            optim.zero_grad()
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            self.assertEqual(loss, ref_loss)
            self.assertEqual(
                model.bias.grad._local_tensor,
                ref_model.bias.grad,
            )

        state = model._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        group = fsdp_param_group.mesh_info.shard_process_group
        (
            _,
            _,
            _,
            all_reduce_op,
        ) = _get_gradient_divide_factors(group, None, torch.float32)
        self.assertEqual(all_reduce_op, ReduceOp.SUM)


if __name__ == "__main__":
    run_tests()
