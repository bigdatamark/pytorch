# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for fully_shard / replicate_with_fsdp on parameters that are already
# DTensors on an SPMD device mesh, selected via DataParallelMeshDims (dp_mesh_dims). Covers train
# parity with a DDP reference over 1D FSDP, DDP-only, HSDP, multi-dim shard/replicate and FSDP+TP
# meshes, the sharded parameter mesh and placements, reduce-scatter of unused DTensor params, and
# the validation errors for bad dp_mesh_dims. Import-time side effect: defines the module-level
# device_type from get_devtype().

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed._composable.replicate_with_fsdp import (
    replicate as replicate_with_fsdp,
)
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, MLP
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


def _tp_partition_fn(name, module, device_mesh):
    # [FUNCTION DOC] HELPER: distribute_module partition function for tensor parallelism. Does
    # nothing for non-Linear modules. For a Linear, replaces each direct parameter with a DTensor
    # parameter (same requires_grad) over device_mesh: replicated on all but the last mesh dim;
    # on the last dim the weight is Shard(0) if the module name contains in_proj, else Shard(1),
    # and other parameters (bias) are Replicate. Mutates the module via register_parameter.
    """Partition Linear weights across the last mesh dim for TP."""
    if not isinstance(module, nn.Linear):
        return
    num_non_tp_dims = device_mesh.ndim - 1
    replicate_prefix = [Replicate()] * num_non_tp_dims
    for param_name, param in list(module.named_parameters(recurse=False)):
        if param_name == "weight":
            if "in_proj" in name:
                placements = replicate_prefix + [Shard(0)]
            else:
                placements = replicate_prefix + [Shard(1)]
        else:
            placements = replicate_prefix + [Replicate()]
        dist_param = nn.Parameter(
            distribute_tensor(param, device_mesh, placements),
            requires_grad=param.requires_grad,
        )
        module.register_parameter(param_name, dist_param)


def _tp_shard_fn(param):
    # [FUNCTION DOC] HELPER: shard_placement_fn for FSDP+TP. Returns Shard(1) if the param
    # already has a Shard on dim 0 in its placements (the TP dim), else Shard(0), so FSDP avoids
    # the dim TP already shards.
    """FSDP shard placement that avoids the existing TP shard dim."""
    if any(isinstance(p, Shard) and p.dim == 0 for p in param.placements):
        return Shard(1)
    return Shard(0)


class TestFullyShardDTensor(FSDPTest):
    # [CLASS DOC] Multi-process tests (up to 4 GPUs) of FSDP2 with DTensor parameters on SPMD
    # meshes. world_size is min(4, device count), or 0 without an accelerator.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Returns min(4, accelerator device count), or 0 when no
        # accelerator is available.
        if torch.accelerator.is_available():
            return min(4, torch.accelerator.device_count())
        return 0

    def _run_train_parity(
        self, model, ref_model, dp_pg, mesh=None, num_iters=5, mlp_dim=16
    ):
        # [FUNCTION DOC] HELPER: Runs num_iters training steps on model and ref_model with
        # separate Adam optimizers (lr 1e-2, no foreach), using per-rank seeded random inputs of
        # width mlp_dim; zero_grad alternates between set_to_none and not. When mesh is given,
        # the input is wrapped as a fully replicated DTensor for the model. Asserts the losses
        # match each step (full_tensor for DTensor loss), then asserts all final parameters
        # match the reference (gathered with full_tensor when DTensor), with a message naming
        # the mismatched parameters.
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        torch.manual_seed(42 + dp_pg.rank() + 1)
        for i in range(num_iters):
            inp = torch.randn((2, mlp_dim), device=device_type)
            ref_optim.zero_grad(set_to_none=(i % 2 == 0))
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            ref_optim.step()

            optim.zero_grad(set_to_none=(i % 2 == 0))
            if mesh is not None:
                # Use Replicate on all dims: each rank computes on the full
                # input (like DDP). DTensor handles the TP-sharded params.
                inp = DTensor.from_local(
                    inp, mesh, [Replicate()] * mesh.ndim, run_check=False
                )
            loss = model(inp).sum()
            loss.backward()
            optim.step()

            loss_cmp = loss.full_tensor() if isinstance(loss, DTensor) else loss
            self.assertEqual(ref_loss, loss_cmp)

        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), model.named_parameters(), strict=True
        ):
            p2_full = p2.full_tensor() if isinstance(p2, DTensor) else p2
            self.assertEqual(
                p1, p2_full, msg=lambda msg: f"{msg}\nParam mismatch: {n1} vs {n2}"
            )

    @skip_if_lt_x_gpu(2)
    def test_dtensor_train_parity(self):
        # [FUNCTION DOC] Train parity of fully_shard or replicate_with_fsdp with DTensor params
        # against a DDP-replicated reference. Builds the case list (1D FSDP with and without
        # reshard_after_forward, DDP-only, replicate_with_fsdp; with 4 or more ranks also HSDP,
        # multi-shard-dim, FSDP+TP, HSDP+TP 3D and multi-dim replicate). Each case unflattens the
        # world mesh, distributes the MLP (with TP partitioning when requested), applies the
        # wrapper with matching dp_mesh_dims, picks the data-parallel group for the reference
        # replicate, runs the parity loop in a subTest and barriers. Needs at least 2 GPUs.
        """Train parity for FSDP/HSDP/DDP with DTensors on SPMD meshes."""
        ws = self.world_size
        world_mesh = init_device_mesh(
            device_type.type, (ws,), mesh_dim_names=("world",)
        )
        # (sizes, names, dp_dims, use_tp, reshard, dp_pg_source, use_rep_fsdp)
        cases = [
            # 1D: FSDP
            (
                (ws,),
                ("fsdp",),
                DataParallelMeshDims(shard="fsdp"),
                False,
                True,
                None,
                False,
            ),
            # 1D: FSDP with reshard_after_forward=False
            (
                (ws,),
                ("fsdp0",),
                DataParallelMeshDims(shard="fsdp0"),
                False,
                False,
                None,
                False,
            ),
            # 1D: DDP-only
            (
                (ws,),
                ("ddp",),
                DataParallelMeshDims(replicate="ddp"),
                False,
                True,
                None,
                False,
            ),
            # 1D: replicate_with_fsdp
            (
                (ws,),
                ("ddp0",),
                DataParallelMeshDims(replicate="ddp0"),
                False,
                True,
                None,
                True,
            ),
        ]
        if ws >= 4:
            cases.extend(
                [
                    # HSDP 2D
                    (
                        (2, ws // 2),
                        ("rep", "shard"),
                        DataParallelMeshDims(shard="shard", replicate="rep"),
                        False,
                        True,
                        "world",
                        False,
                    ),
                    # Multi-shard FSDP
                    (
                        (2, ws // 2),
                        ("dp0", "dp1"),
                        DataParallelMeshDims(shard=("dp0", "dp1")),
                        False,
                        True,
                        "world",
                        False,
                    ),
                    # FSDP+TP
                    (
                        (2, ws // 2),
                        ("fsdp1", "tp"),
                        DataParallelMeshDims(shard="fsdp1"),
                        True,
                        True,
                        "fsdp1",
                        False,
                    ),
                    # FSDP+TP with reshard_after_forward=False
                    (
                        (2, ws // 2),
                        ("fsdp2", "tp0"),
                        DataParallelMeshDims(shard="fsdp2"),
                        True,
                        False,
                        "fsdp2",
                        False,
                    ),
                    # HSDP+TP 3D
                    (
                        (1, ws // 2, 2),
                        ("rep0", "fsdp3", "tp1"),
                        DataParallelMeshDims(shard="fsdp3", replicate="rep0"),
                        True,
                        True,
                        "fsdp3",
                        False,
                    ),
                    # Multi-dim replicate
                    (
                        (1, ws // 2, 2),
                        ("ddp1", "ddp2", "fsdp4"),
                        DataParallelMeshDims(shard="fsdp4", replicate=("ddp1", "ddp2")),
                        False,
                        True,
                        "world",
                        False,
                    ),
                ]
            )
        mlp_dim = 16
        for sizes, names, dp_dims, use_tp, reshard, dp_pg_src, use_rep in cases:
            with self.subTest(
                names=names, use_tp=use_tp, reshard=reshard, use_rep=use_rep
            ):
                mesh = world_mesh._unflatten(0, sizes, names)

                torch.manual_seed(42)
                model = MLP(mlp_dim, device=device_type)
                ref_model = copy.deepcopy(model)

                partition_fn = _tp_partition_fn if use_tp else None
                distribute_module(model, mesh, partition_fn)

                if use_rep:
                    replicate_with_fsdp(model, mesh=mesh, dp_mesh_dims=dp_dims)
                else:
                    shard_fn = _tp_shard_fn if use_tp else None
                    fully_shard(
                        model,
                        mesh=mesh,
                        reshard_after_forward=reshard,
                        shard_placement_fn=shard_fn,
                        dp_mesh_dims=dp_dims,
                    )

                if dp_pg_src is None:
                    dp_pg = mesh.get_group()
                elif dp_pg_src == "world":
                    dp_pg = dist.group.WORLD
                else:
                    dp_pg = mesh[dp_pg_src].get_group()

                replicate(
                    ref_model,
                    device_ids=[self.rank] if device_type.type != "cpu" else None,
                    process_group=dp_pg,
                )

                self._run_train_parity(
                    model, ref_model, dp_pg, mesh=mesh, mlp_dim=mlp_dim
                )
            dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_sharded_param_correctness_1d(self):
        # [FUNCTION DOC] Shards an MLP over a 1D mesh named fsdp and asserts every parameter is a
        # DTensor on that mesh with a single Shard placement. Needs at least 2 GPUs.
        """Verify sharded param mesh and placements for FSDP on 1D mesh."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )

        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        for param in model.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.device_mesh, mesh)
            self.assertEqual(len(param.placements), 1)
            self.assertIsInstance(param.placements[0], Shard)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_dtensor_sharded_params(self):
        # [FUNCTION DOC] Builds a 2D (fsdp, tp) mesh, applies TP placements to the MLP, then
        # fully_shard with a placement function avoiding the TP dim and checks each parameter's
        # two placements: in_proj weight Shard(1) then Shard(0), out_proj weight Shard(0) then
        # Shard(1), bias Shard(0) then Replicate. Needs at least 4 GPUs.
        """Verify sharded param mesh and placements for FSDP+TP on 2D mesh."""
        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, tp_size),
            mesh_dim_names=("fsdp", "tp"),
        )

        mlp_dim = 16
        model = MLP(mlp_dim, device=device_type)

        def partition_fn(name, module, device_mesh):
            # [FUNCTION DOC] HELPER: Local copy of the TP partition function: for Linear modules
            # turns weights into DTensors replicated on the fsdp dim and sharded on the tp dim
            # (dim 0 for names with in_proj, else dim 1); biases replicated on both.
            if not isinstance(module, nn.Linear):
                return
            for param_name, param in list(module.named_parameters(recurse=False)):
                if param_name == "weight":
                    if "in_proj" in name:
                        placements = [Replicate(), Shard(0)]
                    else:
                        placements = [Replicate(), Shard(1)]
                else:
                    placements = [Replicate(), Replicate()]
                dist_param = nn.Parameter(
                    distribute_tensor(param, device_mesh, placements),
                    requires_grad=param.requires_grad,
                )
                module.register_parameter(param_name, dist_param)

        distribute_module(model, mesh, partition_fn)

        def shard_fn(param):
            # [FUNCTION DOC] HELPER: Local shard placement function: Shard(1) when the param is
            # already sharded on dim 0, otherwise Shard(0).
            if any(isinstance(p, Shard) and p.dim == 0 for p in param.placements):
                return Shard(1)
            return Shard(0)

        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=shard_fn,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        for name, param in model.named_parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.device_mesh, mesh)
            self.assertEqual(len(param.placements), 2)
            if "in_proj.weight" in name:
                # FSDP shards dim 1 (avoiding TP dim 0), TP shards dim 0
                self.assertIsInstance(param.placements[0], Shard)
                self.assertEqual(param.placements[0].dim, 1)
                self.assertIsInstance(param.placements[1], Shard)
                self.assertEqual(param.placements[1].dim, 0)
            elif "out_proj.weight" in name:
                # FSDP shards dim 0 (default), TP shards dim 1
                self.assertIsInstance(param.placements[0], Shard)
                self.assertEqual(param.placements[0].dim, 0)
                self.assertIsInstance(param.placements[1], Shard)
                self.assertEqual(param.placements[1].dim, 1)
            elif "bias" in name:
                # FSDP shards dim 0 (default), TP replicates
                self.assertIsInstance(param.placements[0], Shard)
                self.assertEqual(param.placements[0].dim, 0)
                self.assertIsInstance(param.placements[1], Replicate)

    @skip_if_lt_x_gpu(4)
    def test_reduce_scatter_unused_dtensor_param(self):
        # [FUNCTION DOC] Checks that with set_reduce_scatter_unused_params enabled, a DTensor
        # parameter unused in forward still receives a zero DTensor gradient. Uses a (dp, ep)
        # mesh, a module with used and unused expert params sharded on ep, fully_shard over dp,
        # runs a forward and backward and asserts the unused param's grad is a DTensor of zeros.
        # Needs at least 4 GPUs.
        class TwoExpertParams(nn.Module):
            # [CLASS DOC] HELPER: Module with two DTensor parameters, used and unused, each an
            # 8x4 random tensor sharded on dim 0 across the given expert-parallel mesh.
            def __init__(self, ep_mesh) -> None:
                # [FUNCTION DOC] HELPER: Creates random used and unused tensors, distributes both
                # with Shard(0) over ep_mesh and registers them as parameters.
                super().__init__()
                used = torch.randn(8, 4, device=device_type)
                unused = torch.randn(8, 4, device=device_type)
                self.register_parameter(
                    "used",
                    nn.Parameter(distribute_tensor(used, ep_mesh, [Shard(0)])),
                )
                self.register_parameter(
                    "unused",
                    nn.Parameter(distribute_tensor(unused, ep_mesh, [Shard(0)])),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of the local shard of the used param (as
                # float) plus the input sum times zero, so the input is touched and the unused
                # param gets no grad.
                return self.used.to_local().float().sum() + (x.float().sum() * 0)

        mesh = init_device_mesh(
            device_type.type,
            (self.world_size // 2, 2),
            mesh_dim_names=("dp", "ep"),
        )
        model = TwoExpertParams(mesh["ep"])
        fully_shard(model, mesh=mesh["dp"], reshard_after_forward=True)
        model.set_reduce_scatter_unused_params(True)

        loss = model(torch.ones(1, device=device_type))
        loss.backward()
        torch.get_device_module(device_type).synchronize()

        unused_grad = model.unused.grad
        self.assertIsNotNone(unused_grad)
        self.assertIsInstance(unused_grad, DTensor)
        self.assertEqual(
            unused_grad.to_local(), torch.zeros_like(unused_grad.to_local())
        )

    @skip_if_lt_x_gpu(2)
    def test_validation_non_replicate_dp_placement(self):
        # [FUNCTION DOC] Asserts fully_shard raises ValueError mentioning Expected Replicate when
        # a parameter (weight, Shard(0)) is not Replicate on the dp shard mesh dim. Needs 2 GPUs.
        """Error when a param has non-Replicate placement on the DP shard dim."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = nn.Linear(16, 32, device=device_type)
        # Distribute weight with Shard(0) on the FSDP dim
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.data, mesh, [Shard(0)]),
            requires_grad=True,
        )
        model.bias = nn.Parameter(
            distribute_tensor(model.bias.data, mesh, [Replicate()]),
            requires_grad=True,
        )
        with self.assertRaisesRegex(ValueError, "Expected Replicate"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_invalid_dim_names(self):
        # [FUNCTION DOC] Asserts fully_shard raises ValueError (not found in mesh dim names) when
        # dp_mesh_dims names a mesh dim that does not exist. Needs 2 GPUs.
        """Error when dp_mesh_dims references nonexistent mesh dim names."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        with self.assertRaisesRegex(ValueError, "not found in mesh.mesh_dim_names"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="nonexistent"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_mesh_mismatch(self):
        # [FUNCTION DOC] Asserts fully_shard raises ValueError mentioning same mesh when the
        # parameters' DTensor mesh (mesh1) differs from the mesh passed in (a second, separately
        # created mesh2). Needs 2 GPUs.
        """Error when param DTensor mesh differs from the mesh passed to fully_shard."""
        mesh1 = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        mesh2 = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = nn.Linear(16, 32, device=device_type)
        # Distribute params on mesh1 but pass mesh2 to fully_shard
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.data, mesh1, [Replicate()]),
            requires_grad=True,
        )
        model.bias = nn.Parameter(
            distribute_tensor(model.bias.data, mesh1, [Replicate()]),
            requires_grad=True,
        )
        with self.assertRaisesRegex(ValueError, "same mesh"):
            fully_shard(
                model,
                mesh=mesh2,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )

    def test_validation_at_least_one_required(self):
        # [FUNCTION DOC] Asserts constructing DataParallelMeshDims with neither shard nor
        # replicate raises ValueError. Needs no GPUs.
        """Error when neither shard nor replicate is set."""
        with self.assertRaisesRegex(ValueError, "At least one of shard or replicate"):
            DataParallelMeshDims()

    @skip_if_lt_x_gpu(2)
    def test_validation_spmd_mesh_non_dtensor_params(self):
        # [FUNCTION DOC] Asserts fully_shard raises ValueError (must be DTensors) when
        # dp_mesh_dims is given but the module parameters are plain tensors. Needs 2 GPUs.
        """Error when dp_mesh_dims is provided but params are not DTensors."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        # Do NOT call distribute_module -- params are plain tensors
        with self.assertRaisesRegex(ValueError, "must be DTensors"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_reshard_after_forward_int_spmd(self):
        # [FUNCTION DOC] Asserts fully_shard raises NotImplementedError when
        # reshard_after_forward is an int together with SPMD dp_mesh_dims. Needs 2 GPUs.
        """Error when reshard_after_forward is int with SPMD mesh."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        with self.assertRaisesRegex(
            NotImplementedError, "reshard_after_forward as int"
        ):
            fully_shard(
                model,
                mesh=mesh,
                reshard_after_forward=2,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )


if __name__ == "__main__":
    run_tests()
