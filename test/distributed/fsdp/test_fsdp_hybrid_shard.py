# Owner(s): ["oncall: distributed"]

import contextlib
import sys
from collections import Counter
from enum import auto, Enum
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _rank_not_in_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp._init_utils import (
    _init_intra_and_inter_node_groups,
    HYBRID_SHARDING_STRATEGIES,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTestContinuous,
    FSDPTestMultiThread,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
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

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


@contextlib.contextmanager
def patch_allreduce(new_allreduce):
    # [FUNCTION DOC] HELPER: Context manager that replaces the all_reduce function of
    # torch.distributed with new_allreduce for the duration of the block and always restores
    # the original afterwards, even on error. Side effect: mutates global module state.
    """
    Patches dist.all_reduce with a new all_reduce and
    restores upon exiting.
    """
    orig_ar = dist.all_reduce
    dist.all_reduce = new_allreduce
    try:
        yield
    finally:
        dist.all_reduce = orig_ar


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter):
    # [FUNCTION DOC] HELPER: Context manager that replaces the reduce_scatter_single function
    # of torch.distributed with new_reduce_scatter for the duration of the block and always
    # restores the original afterwards. Side effect: mutates global module state.
    # KNOWN ISSUE: the docstring says it patches reduce_scatter_tensor, but the code patches
    # reduce_scatter_single.
    """
    Patches dist.reduce_scatter_single with a new reduce_scatter_tensor and
    restores upon exiting.
    """
    orig_reduce_scatter = dist.reduce_scatter_single
    dist.reduce_scatter_single = new_reduce_scatter
    try:
        yield
    finally:
        dist.reduce_scatter_single = orig_reduce_scatter


class MyModel(nn.Module):
    # [CLASS DOC] HELPER: Small module made of three sequential 10x10 linear layers
    # (lin1, lin2, lin3) with no activations, used as a simple FSDP target.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the three 10x10 linear layers.
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies lin1, lin2 and lin3 in order to x and returns it.
        return self.lin3(self.lin2(self.lin1(x)))


class ShardingStrategyMode(Enum):
    # [CLASS DOC] HELPER: Enum choosing how the test model is wrapped: ALL_HYBRID_SHARD uses
    # the hybrid strategy everywhere; MIXED_HYBRID_FULL_SHARD uses it only for the
    # transformer submodule and FULL_SHARD for the outer wrapper.
    ALL_HYBRID_SHARD = auto()
    MIXED_HYBRID_FULL_SHARD = auto()


class TestFSDPHybridShard(FSDPTestContinuous):
    # [CLASS DOC] Multi-process test case for FSDP hybrid sharding. Uses all available
    # devices (at least 2) as the world. Covers argument validation, process group setup,
    # collective counts, parity with plain FSDP, and state dict / sync_module_states with
    # explicit shard and replicate groups. Key helpers: _init_fsdp_model and _init_hsdp_model.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property returning the device count, but at least 2.
        return max(torch.accelerator.device_count(), 2)

    @property
    def process_group(self):
        # [FUNCTION DOC] HELPER: Property returning the default (world) process group.
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_raises_manual_wrap_hybrid_shard_when_none_policy(self):
        # [FUNCTION DOC] Test: wrapping a model with HYBRID_SHARD or _HYBRID_SHARD_ZERO2 but
        # without a process group tuple or device mesh must raise ValueError saying it
        # requires explicit specification of process group or device_mesh.
        model = MyModel().to(device_type)
        err_ctx = self.assertRaisesRegex(
            ValueError,
            "requires explicit specification of process group or device_mesh.",
        )

        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)

        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_save_load_state_dict(self):
        # [FUNCTION DOC] Test (4+ GPUs): splits ranks into two shard groups (lower and upper
        # half) and strided replicate groups pairing one rank from each half, wraps MyModel
        # with HYBRID_SHARD using that tuple, and takes one AdamW step. Checks the model's
        # process groups equal the supplied ones, takes sharded model and optimizer state
        # dicts, and loads them into a fresh identically wrapped model and optimizer.
        model = MyModel().to(device_type)
        num_node_devices = torch.accelerator.device_count()
        shard_rank_lists = (
            list(range(num_node_devices // 2)),
            list(range(num_node_devices // 2, num_node_devices)),
        )
        shard_groups = (
            dist.new_group(shard_rank_lists[0]),
            dist.new_group(shard_rank_lists[1]),
        )
        my_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )
        my_replicate_group = None
        my_rank = self.rank
        # Create groups like (0, 4), (1, 5), (2, 6) etc and assign appropriately
        shard_factor = len(shard_rank_lists[0])
        for i in range(num_node_devices // 2):
            replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
            replicate_group = dist.new_group(replicate_group_ranks)
            if my_rank in replicate_group_ranks:
                my_replicate_group = replicate_group

        fsdp_ctor = partial(
            FSDP,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            process_group=(my_shard_group, my_replicate_group),
        )
        model = fsdp_ctor(model)
        optim = torch.optim.AdamW(model.parameters())
        # Initialize optimizer states
        model(torch.randn(2, 10)).sum().backward()
        optim.step()
        shard_g = model.process_group
        replicate_g = model._inter_node_pg
        if shard_g != my_shard_group:
            raise AssertionError(
                f"Expected shard_g == my_shard_group, got {shard_g} vs {my_shard_group}"
            )
        if replicate_g != my_replicate_group:
            raise AssertionError(
                f"Expected replicate_g == my_replicate_group, got {replicate_g} vs {my_replicate_group}"
            )
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            msd = model.state_dict()
            osd = FSDP.optim_state_dict(model, optim)

        load_model = fsdp_ctor(MyModel().to(device_type))
        load_optim = torch.optim.AdamW(load_model.parameters())
        with FSDP.state_dict_type(load_model, StateDictType.SHARDED_STATE_DICT):
            load_model.load_state_dict(msd)
            FSDP.optim_state_dict_to_load(load_model, load_optim, osd)
        load_optim.load_state_dict(osd)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_sync_module_state(self):
        # [FUNCTION DOC] Test (4+ GPUs): builds the same shard and replicate groups as the
        # state dict test, sets every linear weight to the local rank value, wraps with
        # HYBRID_SHARD and sync_module_states, and asserts that all weights became 0 (rank
        # 0's values) everywhere, so states were broadcast across both group levels.
        model = MyModel().to(device_type)
        num_node_devices = torch.accelerator.device_count()
        shard_rank_lists = (
            list(range(num_node_devices // 2)),
            list(range(num_node_devices // 2, num_node_devices)),
        )
        shard_groups = (
            dist.new_group(shard_rank_lists[0]),
            dist.new_group(shard_rank_lists[1]),
        )
        my_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )
        my_replicate_group = None
        my_rank = self.rank
        # Create groups like (0, 4), (1, 5), (2, 6) etc and assign appropriately
        shard_factor = len(shard_rank_lists[0])
        for i in range(num_node_devices // 2):
            replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
            replicate_group = dist.new_group(replicate_group_ranks)
            if my_rank in replicate_group_ranks:
                my_replicate_group = replicate_group

        nn.init.constant_(model.lin1.weight, self.rank)
        nn.init.constant_(model.lin2.weight, self.rank)
        nn.init.constant_(model.lin3.weight, self.rank)

        fsdp_ctor = partial(
            FSDP,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            process_group=(my_shard_group, my_replicate_group),
        )
        model = fsdp_ctor(model)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            self.assertTrue((model.lin1.weight == 0).all())
            self.assertTrue((model.lin2.weight == 0).all())
            self.assertTrue((model.lin3.weight == 0).all())

    @skip_if_lt_x_gpu(2)
    def test_invalid_pg_specification_raises(self):
        # [FUNCTION DOC] Test: wrapping with an auto wrap policy, HYBRID_SHARD and a single
        # process group (not a tuple) must raise ValueError saying a pair was expected.
        pol = ModuleWrapPolicy({nn.Linear})
        model = MyModel().to(device_type)
        with self.assertRaisesRegex(
            ValueError, "Expected process_group to be passed in"
        ):
            model = FSDP(
                model,
                auto_wrap_policy=pol,
                process_group=self.process_group,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

    # TODO - add test for ZeRO-2 style sharding ensure params are not
    # resharded after forward.

    @skip_if_lt_x_gpu(2)
    def test_fsdp_hybrid_shard_basic_setup(self):
        # [FUNCTION DOC] Test driver (2+ GPUs): runs _test_fsdp_hybrid_shard_basic_setup over
        # the grid of both hybrid strategies, both wrapping modes, use_orig_params on/off and
        # device mesh vs. default groups.
        """
        Tests basic functionality of HYBRID_SHARD and _HYBRID_SHARD_ZERO2:
            1. Inter and intra-node process groups are correctly setup
            2. Process groups are the same across FSDP wrapped instances
            3. reduce_scatter and allreduce called the expected no. of times
        """
        self.run_subtests(
            {
                "hsdp_sharding_strategy": [
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "sharding_strategy_mode": [
                    ShardingStrategyMode.ALL_HYBRID_SHARD,
                    ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD,
                ],
                "use_orig_params": [False, True],
                "use_device_mesh": [False, True],
            },
            self._test_fsdp_hybrid_shard_basic_setup,
        )

    def _test_fsdp_hybrid_shard_basic_setup(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: ShardingStrategyMode,
        use_orig_params: bool,
        use_device_mesh: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds an HSDP model (optionally via a 1 x world_size device
        # mesh) and checks that every hybrid-strategy FSDP module shares one intra-node group
        # spanning the world and one single-rank inter-node group; non-hybrid modules are only
        # allowed in the mixed mode and must use FULL_SHARD. Then runs one forward/backward
        # with all_reduce and reduce_scatter_single counted via patches and asserts the counts
        # equal the number of flat parameter handles (all-reduce counts only the transformer's
        # handles in mixed mode, reduce-scatter counts all handles).
        if use_device_mesh:
            device_mesh = init_device_mesh(device_type, (1, self.world_size))
        else:
            device_mesh = None
        hsdp_model = self._init_hsdp_model(
            hsdp_sharding_strategy,
            sharding_strategy_mode,
            use_orig_params,
            hsdp_device_mesh=device_mesh,
        )
        # All FSDP modules should have state.process_group as the process group over which to
        # shard (default process group), and state._inter_node_pg (process group containing only
        # this rank)
        intra_node_pgs = set()
        inter_node_pgs = set()
        for fsdp_module in hsdp_model.fsdp_modules(hsdp_model):
            # TODO: This needs to be replaced if we deprecate
            # `FSDP.sharding_strategy` to only use the handle one.
            # https://github.com/pytorch/pytorch/issues/90857
            if fsdp_module.sharding_strategy not in HYBRID_SHARDING_STRATEGIES:
                self.assertEqual(
                    sharding_strategy_mode, ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD
                )
                self.assertEqual(
                    fsdp_module.sharding_strategy, ShardingStrategy.FULL_SHARD
                )
                continue
            # process_group should be across the node, which is just the
            # whole world here.
            self.assertEqual(
                dist.get_world_size(fsdp_module.process_group),
                dist.get_world_size(self.process_group),
            )
            intra_node_pgs.add(fsdp_module.process_group)
            inter_node_pg = fsdp_module._inter_node_pg
            inter_node_pgs.add(inter_node_pg)
            self.assertEqual(1, dist.get_world_size(inter_node_pg))
            self.assertFalse(_rank_not_in_group(inter_node_pg))
            self.assertEqual(hsdp_sharding_strategy, fsdp_module.sharding_strategy)
        # All fsdp modules should share the same process groups
        self.assertEqual(1, len(intra_node_pgs))
        self.assertEqual(1, len(inter_node_pgs))

        orig_ar = dist.all_reduce
        orig_rs = dist.reduce_scatter_single

        def patched_collective(orig_collective, counter, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper that increments the counter entry keyed by the
            # original collective, then calls and returns the original with all arguments.
            counter[orig_collective] += 1
            return orig_collective(*args, **kwargs)

        cntr = Counter()
        patched_allreduce = partial(patched_collective, orig_ar, cntr)
        patched_reduce_scatter = partial(patched_collective, orig_rs, cntr)
        with (
            patch_allreduce(patched_allreduce),
            patch_reduce_scatter(patched_reduce_scatter),
        ):
            inp = hsdp_model.get_input(device=torch.accelerator.current_device_index())
            out = hsdp_model(inp[0], inp[1])
            loss = hsdp_model.get_loss(inp, out)
            loss.backward()

        if sharding_strategy_mode == ShardingStrategyMode.ALL_HYBRID_SHARD:
            num_flat_params = len(list(traversal_utils._get_fsdp_handles(hsdp_model)))
            self.assertEqual(num_flat_params, cntr[orig_ar])
            self.assertEqual(num_flat_params, cntr[orig_rs])
        elif sharding_strategy_mode == ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD:
            num_hsdp_flat_params = len(
                list(traversal_utils._get_fsdp_handles(hsdp_model.transformer))
            )
            num_flat_params = len(list(traversal_utils._get_fsdp_handles(hsdp_model)))
            self.assertEqual(num_hsdp_flat_params, cntr[orig_ar])
            self.assertEqual(num_flat_params, cntr[orig_rs])

    @skip_if_lt_x_gpu(4)
    def test_fsdp_hybrid_shard_parity(self):
        # [FUNCTION DOC] Test driver (4+ GPUs): runs _test_fsdp_hybrid_shard_parity for both
        # hybrid strategies and use_orig_params on/off.
        self.run_subtests(
            {
                "hsdp_sharding_strategy": [
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "use_orig_params": [False, True],
            },
            self._test_fsdp_hybrid_shard_parity,
        )

    def _test_fsdp_hybrid_shard_parity(
        self, hsdp_sharding_strategy: ShardingStrategy, use_orig_params: bool
    ):
        # [FUNCTION DOC] HELPER: Builds a plain FSDP model and an HSDP model (intra/inter-node
        # groups created with shard size 2) and trains both with Adam for 5 steps on the same
        # per-rank input, asserting the losses are equal at every step. Raises AssertionError
        # if the HSDP model has no replication (inter-node group size is 1 or less).
        fsdp_model = self._init_fsdp_model(use_orig_params)
        global_pg = dist.distributed_c10d._get_default_group()
        hsdp_pgs = _init_intra_and_inter_node_groups(global_pg, 2)
        hsdp_model = self._init_hsdp_model(
            hsdp_sharding_strategy,
            ShardingStrategyMode.ALL_HYBRID_SHARD,
            use_orig_params,
            hsdp_process_groups=hsdp_pgs,
        )
        if not (hsdp_model._inter_node_pg.size() > 1):
            raise AssertionError("HSDP model initialized without replication")
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)
        hsdp_optim = torch.optim.Adam(hsdp_model.parameters(), lr=1e-2)
        torch.manual_seed(global_pg.rank() + 1)
        for _ in range(5):
            inp = fsdp_model.module.get_input(torch.device(device_type))
            losses: list[torch.Tensor] = []
            for model, optim in ((fsdp_model, fsdp_optim), (hsdp_model, hsdp_optim)):
                optim.zero_grad()
                loss = model(*inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
            self.assertEqual(losses[0], losses[1])

    def _init_fsdp_model(self, use_orig_params: bool) -> nn.Module:
        # [FUNCTION DOC] HELPER: Returns a deterministic TransformerWithSharedParams wrapped
        # recursively by FSDP (default strategy) with encoder and decoder layers as wrap
        # units, over the default process group.
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        hsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "device_id": torch.accelerator.current_device_index(),
            "use_orig_params": use_orig_params,
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            hsdp_kwargs,
            deterministic=True,
        )
        return fsdp_model

    def _init_hsdp_model(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: str,
        use_orig_params: bool,
        hsdp_process_groups: tuple[dist.ProcessGroup, dist.ProcessGroup] | None = None,
        hsdp_device_mesh: Optional = None,
    ):
        # [FUNCTION DOC] HELPER: Builds a deterministic transformer test model sharded with
        # the given hybrid strategy, using either a pair of process groups or a device mesh
        # (at most one may be given, else AssertionError). ALL_HYBRID_SHARD wraps every
        # encoder/decoder layer and the root hybrid; MIXED_HYBRID_FULL_SHARD wraps only the
        # transformer submodule hybrid and the root with FULL_SHARD. Returns the wrapped model.
        # KNOWN ISSUE: sharding_strategy_mode is annotated as str but an enum is passed; any
        # other value would leave hsdp_model unbound.
        if not (hsdp_process_groups is None or hsdp_device_mesh is None):
            raise AssertionError(
                "Expected hsdp_process_groups or hsdp_device_mesh to be None"
            )
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        hsdp_kwargs = {
            "device_id": torch.accelerator.current_device_index(),
            "auto_wrap_policy": auto_wrap_policy,
            "sharding_strategy": hsdp_sharding_strategy,
            "use_orig_params": use_orig_params,
            "device_mesh": hsdp_device_mesh,
        }
        if sharding_strategy_mode == ShardingStrategyMode.ALL_HYBRID_SHARD:
            hsdp_model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                hsdp_kwargs,
                deterministic=True,
            )
        elif sharding_strategy_mode == ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD:
            model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.NO_FSDP,
                DEVICEInitMode.DEVICE_BEFORE,
                {},
                deterministic=True,
            )
            # Use the HSDP strategy for the transformer module
            model.transformer = FSDP(model.transformer, **hsdp_kwargs)
            # Use `FULL_SHARD` for the embedding and output projection
            hsdp_model = FSDP(
                model,
                device_id=torch.accelerator.current_device_index(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=use_orig_params,
            )
        return hsdp_model


class TestHSDPSyncModuleStates(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank multi-threaded test case checking that HSDP sync_module_states
    # also broadcasts buffers when only rank 0 holds real data and other ranks use meta.
    class _ModelWithBuffer(nn.Module):
        # [CLASS DOC] HELPER: Module with a 10x10 linear layer followed by a 1d batchnorm
        # (which carries running-stat buffers), both created on the given device.
        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Creates the linear and batchnorm layers on device.
            super().__init__()
            self.lin = nn.Linear(10, 10, device=device)
            self.bn = nn.BatchNorm1d(10, device=device)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies the linear layer then batchnorm to x.
            return self.bn(self.lin(x))

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property giving the number of ranks; always 4.
        return 4

    @skip_if_lt_x_gpu(1)
    def test_hsdp_buffer_sync_from_meta_device(self):
        # [FUNCTION DOC] Test: on a 2 x 2 device mesh, rank 0 builds the model on the real
        # device with batchnorm running_mean 42 and running_var 7 while other ranks use meta
        # device. Wraps with HYBRID_SHARD, use_orig_params, sync_module_states and a
        # param_init_fn that materializes meta modules with to_empty. Asserts every rank's
        # buffers end up with rank 0's values (regression test for the FSDP_SYNCED flag not
        # being reset between the inter-node and intra-node broadcasts).
        """Test that HSDP sync_module_states correctly broadcasts buffers
        when only rank 0 has real weights and other ranks use meta device.

        _sync_module_params_and_buffers marks buffers with FSDP_SYNCED=True
        to avoid redundant syncs in nested wrapping. With two-phase broadcast
        (inter-node then intra-node), this flag must be reset between phases,
        otherwise the intra-node broadcast skips buffers. Non-persistent
        buffers (e.g. RoPE inv_freq) that are materialized from meta device
        via to_empty() and not restored by reset_parameters() would remain
        as uninitialized values on local ranks 1..N.

        Parameters are unaffected by broadcast order because they are
        unconditionally included in every sync call. This test specifically
        targets buffers, which are the only tensors gated by FSDP_SYNCED.
        """
        dev = torch.device(device_type)
        mesh_2d = init_device_mesh(device_type, (2, self.world_size // 2))

        if self.rank == 0:
            model = self._ModelWithBuffer(device=dev)
            model.bn.running_mean.fill_(42.0)
            model.bn.running_var.fill_(7.0)
        else:
            model = self._ModelWithBuffer(device="meta")

        def param_init_fn(module):
            # [FUNCTION DOC] HELPER: If the module directly owns any meta parameter or buffer,
            # moves its own tensors (non-recursive) to uninitialized storage on the real
            # device; otherwise does nothing.
            if any(p.is_meta for p in module.parameters(recurse=False)) or any(
                b.is_meta for b in module.buffers(recurse=False)
            ):
                module.to_empty(device=dev, recurse=False)

        model = FSDP(
            model,
            device_mesh=mesh_2d,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            param_init_fn=param_init_fn,
        )

        # Buffers are not sharded by FSDP, so each rank has a local copy.
        # If the FSDP_SYNCED reset is missing, ranks on non-source nodes
        # will have uninitialized buffer values instead of rank 0's values.
        self.assertEqual(model.bn.running_mean, torch.full((10,), 42.0))
        self.assertEqual(model.bn.running_var, torch.full((10,), 7.0))


instantiate_parametrized_tests(TestFSDPHybridShard)

if __name__ == "__main__":
    run_tests()
