# [MODULE DOC] Tests for DTensor initialization and constructor APIs: in-place nn.init ops on a
# DTensor (constant_, eye_) compared with plain tensors, and the factory functions ones, empty,
# full, zeros, linspace and logspace with different placements, uneven sharding, 2-D meshes and
# sub-meshes. Also creates a LocalTensor variant of the constructor tests at import time.
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    linspace,
    logspace,
    Replicate,
    Shard,
    zeros,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    with_comms,
)


class DTensorInitOpsTest(DTensorTestBase):
    # [CLASS DOC] Tests applying torch.nn.init functions to a sharded or replicated DTensor.
    def _run_init_op(self, init_op, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Builds a random (8, 4) local tensor and a DTensor of it sharded
        # on dim 0. Seeds the RNG with the rank, applies init_op with the extra arguments to a
        # clone of the local tensor, reseeds, applies it to the DTensor, and asserts the DTensor's
        # local shard equals the plain result.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        local_tensor_clone = torch.clone(input_tensor)
        torch.manual_seed(self.rank)
        local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
        torch.manual_seed(self.rank)
        dtensor = init_op(dtensor, *args, **kwargs)
        self.assertEqual(local_tensor_clone, dtensor.to_local())

    @with_comms
    def test_init_ops(self):
        # [FUNCTION DOC] Checks nn.init.constant_ with value 2.4 on a sharded DTensor matches the
        # plain tensor result (random init tests live elsewhere).
        # NOTE: random init tests are moved to test_random_ops.py
        self._run_init_op(torch.nn.init.constant_, 2.4)

    @with_comms
    def test_eye_init(self):
        # [FUNCTION DOC] Applies nn.init.eye_ in place to a replicated (8, 8) DTensor and asserts
        # its local tensor equals the identity matrix.
        # Test nn.init.eye_() with DTensor (issue #173357)
        device_mesh = self.build_device_mesh()
        shard_spec = [Replicate()]
        input_size = (8, 8)

        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        torch.nn.init.eye_(dtensor)

        expected = torch.eye(*input_size, device=self.device_type)
        self.assertEqual(expected, dtensor.to_local())


class DTensorConstructorTest(DTensorTestBase):
    # [CLASS DOC] Tests the DTensor factory functions on a 1-D mesh of 4 ranks (and some 2-D and
    # sub-mesh cases), comparing local shards with the corresponding plain torch results.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Returns the fixed world size, 4.
        return 4

    def _run_init_op(self, init_op, dist_init_op, eq_op, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Compares a distributed factory (dist_init_op) with its plain
        # counterpart (init_op) using the comparison callable eq_op, passing extra arguments to
        # both. On a 1-D mesh it tries placements Shard on dims 0, 1, 2 and Replicate. First with
        # evenly divisible size (4, 8, 12): the local shard must match a plain tensor of the
        # reduced size. Then with uneven size (5, 10, 15): for sharded placements each rank's
        # local tensor must match its torch.chunk piece when the rank has one (checked per rank
        # under local tensor mode); replicated must match the full tensor. Finally an empty
        # shape replicated case is compared.
        # 1d mesh test
        device_mesh = self.build_device_mesh()
        placements_list = [[Shard(0)], [Shard(1)], [Shard(2)], [Replicate()]]

        # even sharding
        tensor_size = [4, 8, 12]
        for placements in placements_list:
            local_tensor_size = tensor_size.copy()
            if isinstance(placements[0], Shard):
                shard_dim = placements[0].dim
                local_tensor_size[shard_dim] //= self.world_size

            dist_tensor = dist_init_op(
                tensor_size,
                *args,
                **kwargs,
                device_mesh=device_mesh,
                placements=placements,
            )
            ones_expected = init_op(local_tensor_size, *args, **kwargs)
            eq_op(ones_expected, dist_tensor.to_local())

        # uneven sharding
        tensor_size = [5, 10, 15]
        for placements in placements_list:
            dist_tensor = dist_init_op(
                tensor_size,
                *args,
                **kwargs,
                device_mesh=device_mesh,
                placements=placements,
            )
            if isinstance(placements[0], Shard):
                shard_dim = placements[0].dim
                exp_tensor_list = list(
                    torch.chunk(
                        init_op(tensor_size, *args, **kwargs),
                        self.world_size,
                        dim=shard_dim,
                    )
                )

                @maybe_run_for_local_tensor
                def check_per_rank_chunk(rank, local_tensor):
                    # [FUNCTION DOC] HELPER: If the rank has an expected chunk, applies eq_op
                    # to that chunk and the given local tensor; ranks beyond the chunks are
                    # skipped.
                    if rank < len(exp_tensor_list):
                        eq_op(exp_tensor_list[rank], local_tensor)

                check_per_rank_chunk(self.rank, dist_tensor.to_local())
            else:
                exp_tensor = init_op(tensor_size, *args, **kwargs)
                eq_op(exp_tensor, dist_tensor.to_local())

        # empty shape
        local_tensor = dist_init_op(
            [], *args, **kwargs, device_mesh=device_mesh, placements=[Replicate()]
        ).to_local()
        expected_tensor = init_op([], *args, **kwargs)
        eq_op(expected_tensor, local_tensor)

    @with_comms
    def test_ones(self):
        # [FUNCTION DOC] Runs the shared comparison for ones versus the DTensor ones, requires_grad
        # set, using assertEqual.
        self._run_init_op(
            torch.ones,
            torch.distributed.tensor.ones,
            self.assertEqual,
            requires_grad=True,
        )

    @with_comms
    def test_empty(self):
        # [FUNCTION DOC] Runs the shared comparison for empty versus the DTensor empty; since
        # values are uninitialized, only shape, dtype and layout are compared.
        self._run_init_op(
            torch.empty,
            torch.distributed.tensor.empty,
            lambda x, y: (x.shape == y.shape)
            and (x.dtype == y.dtype)
            and (x.layout == y.layout),
            requires_grad=True,
        )

    @with_comms
    def test_full(self):
        # [FUNCTION DOC] Runs the shared comparison for full with fill value 123.4 versus the
        # DTensor full, requires_grad set, using assertEqual.
        self._run_init_op(
            torch.full,
            torch.distributed.tensor.full,
            self.assertEqual,
            123.4,
            requires_grad=True,
        )

    @with_comms
    def test_linspace(self):
        # [FUNCTION DOC] For replicated and Shard(0) placements, checks the DTensor linspace
        # against torch.linspace via full_tensor for: float64 with 8 steps, default dtype, a
        # single step, zero steps (size 0 only), a descending range, and 0-dim DTensor start and
        # end. Also asserts a RuntimeError about 0-dimensional start and end when start is a
        # one-dimensional plain tensor.
        mesh = self.build_device_mesh()
        steps = 8

        for placements in ([Replicate()], [Shard(0)]):
            dist_tensor = linspace(
                1.0,
                2.0,
                steps,
                dtype=torch.float64,
                device_mesh=mesh,
                placements=placements,
            )
            self.assertEqual(dist_tensor.size(), torch.Size([steps]))
            self.assertEqual(
                dist_tensor.full_tensor(),
                torch.linspace(1.0, 2.0, steps, dtype=torch.float64),
            )

            dist_tensor = linspace(
                0.0, 1.0, steps, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(0.0, 1.0, steps))

            dist_tensor = linspace(1.0, 2.0, 1, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([1]))
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(1.0, 2.0, 1))

            dist_tensor = linspace(1.0, 2.0, 0, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([0]))

            dist_tensor = linspace(
                2.0, -2.0, 5, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(2.0, -2.0, 5))

            start = DTensor.from_local(
                torch.tensor(1.0, device=self.device_type), mesh, [Replicate()]
            )
            end = DTensor.from_local(
                torch.tensor(2.0, device=self.device_type), mesh, [Replicate()]
            )
            dist_tensor = linspace(
                start, end, steps, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(1.0, 2.0, steps))

            with self.assertRaisesRegex(
                RuntimeError,
                "linspace only supports 0-dimensional start and end tensors",
            ):
                linspace(
                    torch.tensor([1.0, 2.0], device=self.device_type),
                    2.0,
                    steps,
                    device_mesh=mesh,
                    placements=placements,
                )

    @with_comms
    def test_logspace(self):
        # [FUNCTION DOC] Same structure as the linspace test but for logspace: float64, a custom
        # base of 2.0, one step, zero steps, descending range, 0-dim DTensor endpoints, and the
        # RuntimeError for non-0-dim start, with both replicated and Shard(0) placements.
        mesh = self.build_device_mesh()
        steps = 8

        for placements in ([Replicate()], [Shard(0)]):
            dist_tensor = logspace(
                1.0,
                2.0,
                steps,
                dtype=torch.float64,
                device_mesh=mesh,
                placements=placements,
            )
            self.assertEqual(dist_tensor.size(), torch.Size([steps]))
            self.assertEqual(
                dist_tensor.full_tensor(),
                torch.logspace(1.0, 2.0, steps, dtype=torch.float64),
            )

            dist_tensor = logspace(
                0.0, 1.0, steps, base=2.0, device_mesh=mesh, placements=placements
            )
            self.assertEqual(
                dist_tensor.full_tensor(), torch.logspace(0.0, 1.0, steps, base=2.0)
            )

            dist_tensor = logspace(1.0, 2.0, 1, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([1]))
            self.assertEqual(dist_tensor.full_tensor(), torch.logspace(1.0, 2.0, 1))

            dist_tensor = logspace(1.0, 2.0, 0, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([0]))

            dist_tensor = logspace(
                2.0, -2.0, 5, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.logspace(2.0, -2.0, 5))

            start = DTensor.from_local(
                torch.tensor(1.0, device=self.device_type), mesh, [Replicate()]
            )
            end = DTensor.from_local(
                torch.tensor(2.0, device=self.device_type), mesh, [Replicate()]
            )
            dist_tensor = logspace(
                start, end, steps, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.logspace(1.0, 2.0, steps))

            with self.assertRaisesRegex(
                RuntimeError,
                "logspace only supports 0-dimensional start and end tensors",
            ):
                logspace(
                    torch.tensor([1.0, 2.0], device=self.device_type),
                    2.0,
                    steps,
                    device_mesh=mesh,
                    placements=placements,
                )

    @with_comms
    def test_zeros(self):
        # [FUNCTION DOC] Runs the shared comparison for zeros versus the DTensor zeros with
        # requires_grad set, using assertEqual.
        self._run_init_op(
            torch.zeros,
            torch.distributed.tensor.zeros,
            self.assertEqual,
            requires_grad=True,
        )

    @with_comms
    def test_zeros_full_mesh(self):
        # [FUNCTION DOC] Checks local shard shapes and contents of zeros over the full 4-rank
        # mesh: 1-D Shard(0) of (32, 3) gives (8, 3); uneven (31, 3) gives 8 rows on ranks up to 2
        # and 7 on rank 3; on a 2x2 mesh Shard/Replicate gives (16, 4), Shard/Shard gives (16, 2),
        # and uneven (31, 3) Shard/Shard gives per-rank shapes (16,2), (16,1), (15,2), (15,1).
        # construct a gpu device 1d mesh
        mesh = self.build_device_mesh()
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([8, 3]))

        local_tensor = torch.zeros(8, 3)
        self.assertEqual(dist_tensor.to_local(), local_tensor)

        self.assertEqual(dist_tensor.device.type, self.device_type)

        # 1d sharded unevenly
        size = [31, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        @maybe_run_for_local_tensor
        def check_per_rank_tensors(rank, local_tensor):
            # [FUNCTION DOC] HELPER: Asserts shape and zero contents of a local tensor: (8, 3) for
            # ranks 0 to 2, otherwise (7, 3).
            if rank <= 2:
                self.assertEqual(local_tensor.size(), torch.Size([8, 3]))
                self.assertEqual(torch.zeros(8, 3), local_tensor)
            else:
                self.assertEqual(local_tensor.size(), torch.Size([7, 3]))
                self.assertEqual(torch.zeros(7, 3), local_tensor)

        check_per_rank_tensors(self.rank, local_tensor)

        # construct a gpu device mesh with 2d: shard, replicate
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        placements = [Shard(0), Replicate()]
        size = [32, 4]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([16, 4]))
        self.assertEqual(local_tensor, torch.zeros([16, 4]))

        # construct a gpu device mesh with 2d: shard, shard
        placements = [Shard(0), Shard(1)]
        size = [32, 4]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([16, 2]))
        self.assertEqual(local_tensor, torch.zeros([16, 2]))

        # 2d sharded unevenly
        placements = [Shard(0), Shard(1)]
        size = [31, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        if self.rank == 0:
            self.assertEqual(local_tensor, torch.zeros([16, 2]))
        elif self.rank == 1:
            self.assertEqual(local_tensor, torch.zeros([16, 1]))
        elif self.rank == 2:
            self.assertEqual(local_tensor, torch.zeros([15, 2]))
        elif self.rank == 3:
            self.assertEqual(local_tensor, torch.zeros([15, 1]))

    @with_comms
    def test_zeros_submesh(self):
        # [FUNCTION DOC] Checks zeros on meshes that cover only some ranks: ranks {0, 3} (16, 3
        # each), uneven ranks {0, 1, 3} (11, 11, 10 rows), and a 2-D mesh of ranks 0 and 3 with
        # two shard dims. Ranks outside the mesh must hold an empty tensor of size 0.
        # default world_size is 4
        # construct a gpu device 1d mesh, with no sub pg initialized
        sub_mesh_list = [0, 3]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in sub_mesh_list:
            self.assertEqual(local_tensor.size(), torch.Size([16, 3]))
            self.assertEqual(local_tensor, torch.zeros([16, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.zeros(0))

        # construct a gpu device 1d mesh: unevenly, with subpg initialized
        sub_mesh_list = [0, 1, 3]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in sub_mesh_list:
            if self.rank != 3:
                self.assertEqual(local_tensor.size(), torch.Size([11, 3]))
                self.assertEqual(local_tensor, torch.zeros([11, 3]))
            else:
                self.assertEqual(local_tensor.size(), torch.Size([10, 3]))
                self.assertEqual(local_tensor, torch.zeros([10, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.tensor([]))

        # construct a gpu device 2d mesh, with no subpg initialized
        sub_mesh_list = [[0], [3]]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0), Shard(1)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in [0, 3]:
            self.assertEqual(local_tensor.size(), torch.Size([16, 3]))
            self.assertEqual(local_tensor, torch.zeros([16, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.tensor([]))


DTensorConstructorTestWithLocalTensor = create_local_tensor_test_class(
    DTensorConstructorTest,
    skipped_tests=[
        # Non-contigous sub-meshes are not supported
        "test_zeros_submesh",
    ],
)

if __name__ == "__main__":
    run_tests()
