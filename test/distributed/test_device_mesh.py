# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for DeviceMesh and init_device_mesh (creation, process group reuse, device
# selection, slicing, flatten/unflatten/concatenate, backend overrides, torchcomms integration)
# plus mesh collectives and the CuTe-style _FlatLayout/_MeshLayout algebra. Distributed cases
# are run as multi-process DTensorTestBase tests; the layout tests are single-process.
# Import-time side effects: computes the module globals device_type and device_count from
# the current accelerator and sets _NCCL_AVAILABLE by trying to import the NCCL binding.
import functools
import os
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._C._distributed_c10d import Backend as C10dBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed import config as dist_config
from torch.distributed._mesh_layout import _FlatLayout, _MeshLayout
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import (
    _get_default_group,
    _TORCHCOMM_AVAILABLE,
    _world,
    get_global_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    new_group,
    ProcessGroup,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._collective_utils import (
    mesh_broadcast,
    mesh_scatter,
    unpad_tensor,
)
from torch.distributed.tensor.placement_types import _Partial, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_HPU, TEST_XPU, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup, FakeStore
from torch.utils._typing_utils import not_none


device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)
device_count = torch.accelerator.device_count()

try:
    import torch._C._distributed_c10d.ProcessGroupNCCL

    _NCCL_AVAILABLE = True
except ImportError:
    _NCCL_AVAILABLE = False


def _set_env_var(addr="localhost", port="25364", world_size=1, rank=0, local_rank=-1):
    # [FUNCTION DOC] HELPER: Sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE and RANK in os.environ
    # from the arguments (strings), and LOCAL_RANK only when local_rank is not -1. Mutates the
    # process environment and does not clean it up.
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["RANK"] = f"{rank}"
    if local_rank != -1:
        os.environ["LOCAL_RANK"] = f"{local_rank}"


def _with_torchcomm_env(func):
    # [FUNCTION DOC] HELPER: Decorator for tests that need torchcomms. The returned wrapper sets
    # TORCHCOMM_RANK, TORCHCOMM_SIZE and TORCHCOMM_STORE_PATH (from the test's rank, world_size
    # and file_name) plus MASTER_ADDR/MASTER_PORT, runs the test, and in a finally block
    # removes the three TORCHCOMM variables. MASTER_ADDR and MASTER_PORT are left set
    # afterwards.
    """Sets TORCHCOMM env vars needed by torchcomms before init_pg runs."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that applies the torchcomms environment around the
        # wrapped test method and returns its result.
        os.environ["TORCHCOMM_RANK"] = str(self.rank)
        os.environ["TORCHCOMM_SIZE"] = str(self.world_size)
        os.environ["TORCHCOMM_STORE_PATH"] = self.file_name
        # torchcomms' StoreManager unconditionally requires MASTER_ADDR /
        # MASTER_PORT (TORCHCOMM_STORE_PATH no longer suffices after the
        # StoreManager refactor in torchcomms #971).
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "25901"
        try:
            return func(self, *args, **kwargs)
        finally:
            os.environ.pop("TORCHCOMM_RANK", None)
            os.environ.pop("TORCHCOMM_SIZE", None)
            os.environ.pop("TORCHCOMM_STORE_PATH", None)

    return wrapper


@unittest.skipIf(TEST_XPU or TEST_HPU, "XPU/HPU does not support gloo backend.")
class DeviceMeshTestGlooBackend(DTensorTestBase):
    # [CLASS DOC] Tests DeviceMesh behavior when the default process group uses the gloo
    # backend. Skipped on XPU and HPU. Runs as a DTensorTestBase multi-process test.
    @property
    def backend(self):
        # [FUNCTION DOC] HELPER: Test property that makes the base class initialize a gloo
        # process group.
        return "gloo"

    @with_comms
    def test_device_mesh_reuse_default_group(self):
        # [FUNCTION DOC] Checks that a 1D mesh over the whole world either reuses the default
        # group (CPU-only case, groups equal) or, when CUDA is available, gets a distinct group
        # of the same size.
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        mesh_group = mesh.get_group()
        default_group = _get_default_group()
        if torch.cuda.is_available():
            self.assertNotEqual(mesh_group, default_group)
            self.assertEqual(get_world_size(mesh_group), get_world_size(default_group))
        else:
            self.assertEqual(mesh_group, default_group)


class DeviceMeshSetDeviceTest(DTensorTestBase):
    # [CLASS DOC] Tests how DeviceMesh construction selects the accelerator device when the
    # process group is not yet initialized. World size 4; each test needs at least 4 GPUs.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Test property giving the world size (4).
        return 4

    @skip_if_lt_x_gpu(4)
    def test_manual_set_device(self):
        # [FUNCTION DOC] Sets the accelerator device to a rank-shifted index before building a
        # DeviceMesh from env vars, and checks the mesh initializes the group but keeps that
        # manually chosen device.
        mesh_tensor = torch.arange(4).reshape(2, 2)
        self.assertTrue(not is_initialized())

        # Set the device on each process before DeviceMesh constructor,
        # and device to be different than the default world rank
        torch.accelerator.set_device_index((self.rank + 2) % self.world_size)
        _set_env_var(world_size=self.world_size, rank=self.rank)
        DeviceMesh(self.device_type, mesh_tensor)
        self.assertTrue(is_initialized())

        # check that the device is set to the correct device
        # and respect the previous set_device calls
        self.assertEqual(
            torch.accelerator.current_device_idx(), (self.rank + 2) % self.world_size
        )
        self.destroy_pg()

    @skip_if_lt_x_gpu(4)
    def test_auto_set_device_from_local_rank(self):
        # [FUNCTION DOC] Sets LOCAL_RANK to a rank-shifted value, builds a DeviceMesh, and
        # checks the group gets initialized and the current device index equals LOCAL_RANK.
        mesh_tensor = torch.arange(4).reshape(2, 2)
        self.assertTrue(not is_initialized())
        # set the local rank to be different than the default world rank,
        # DeviceMesh should respect LOCAL_RANK env var if it's set
        local_rank = (self.rank + 1) % self.world_size

        _set_env_var(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=local_rank,
        )
        DeviceMesh(self.device_type, mesh_tensor)
        self.assertTrue(is_initialized())

        # check that the device is set to the correct device
        # and respect the LOCAL_RANK env var
        self.assertEqual(torch.accelerator.current_device_idx(), local_rank)
        self.destroy_pg()

    @skip_if_lt_x_gpu(4)
    def test_auto_set_device_from_heuristic(self):
        # [FUNCTION DOC] With no device chosen and no LOCAL_RANK, builds a DeviceMesh and checks
        # it warns about the unset default device, initializes the group, and selects the device
        # equal to the rank.
        mesh_tensor = torch.arange(4).reshape(2, 2)
        self.assertTrue(not is_initialized())

        _set_env_var(
            world_size=self.world_size,
            rank=self.rank,
        )
        with self.assertWarnsRegex(
            UserWarning, "It seems like you did not set/select the default device"
        ):
            DeviceMesh(self.device_type, mesh_tensor)
        self.assertTrue(is_initialized())

        # check that the device is set to the correct device
        self.assertEqual(torch.accelerator.current_device_idx(), self.rank)
        self.destroy_pg()


class DeviceMeshTest(DTensorTestBase):
    # [CLASS DOC] Tests for DeviceMesh construction, group access, local ranks, fake process
    # groups, from_group and root-mesh lookup on a 4-rank world.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Test property giving the world size (4).
        return 4

    @skip_if_lt_x_gpu(4)
    def test_init_process_group(self):
        # [FUNCTION DOC] Builds a 2x2 DeviceMesh with only env vars set and no prior init, and
        # checks the default process group is initialized; then destroys the process group.
        mesh_tensor = torch.arange(4).reshape(2, 2)
        self.assertTrue(not is_initialized())
        _set_env_var(world_size=self.world_size, rank=self.rank)
        DeviceMesh(self.device_type, mesh_tensor)
        self.assertTrue(is_initialized())
        self.destroy_pg(self.rank)

    @with_comms()
    def test_2d_mesh_non_eager_init_subgroup(self):
        # [FUNCTION DOC] Checks that with lazy (non-eager) init the subgroups of a 2D mesh have
        # no bound device id.
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(self.device_type, mesh_shape)

        self.assertEqual(mesh_2d.get_group(0).bound_device_id, None)
        self.assertEqual(mesh_2d.get_group(1).bound_device_id, None)

    # TODO: need to refactor the other tests in this file to test both
    # eager_init=True and eager_init=False scenarios.
    @with_comms(eager_init=True)
    def test_2d_mesh_eager_init_subgroup(self):
        # [FUNCTION DOC] Checks that with eager init on nccl the subgroups of a 2D mesh already
        # have a bound device id equal to the current CUDA device; other backends are not
        # checked.
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(self.device_type, mesh_shape)

        # when eager init is used, the subgroup is created from nccl comm split and
        # there would be bound_device_id immediately assigned for the subgroup.
        if self.backend == "nccl":
            curr_device = torch.cuda.current_device()
            self.assertEqual(mesh_2d.get_group(0).bound_device_id.index, curr_device)
            self.assertEqual(mesh_2d.get_group(1).bound_device_id.index, curr_device)

    @with_comms()
    def test_get_group_and_get_all_groups(self):
        # [FUNCTION DOC] Checks that get_group by index equals get_group by name, that sliced 1D
        # meshes return the same groups, and that get_all_groups returns both groups of a named
        # 2D mesh.
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )

        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]

        self.assertEqual(mesh_2d.get_group(0), mesh_2d.get_group("dp"))
        self.assertEqual(mesh_2d.get_group(1), mesh_2d.get_group("tp"))

        self.assertEqual(mesh_2d.get_group("dp"), dp_mesh.get_group())
        self.assertEqual(mesh_2d.get_group("tp"), tp_mesh.get_group())

        groups = mesh_2d.get_all_groups()
        self.assertEqual(len(groups), 2)
        self.assertTrue(tp_mesh.get_group() in groups)
        self.assertTrue(dp_mesh.get_group() in groups)

    @with_comms
    def test_get_local_rank_raises_exception(self):
        # [FUNCTION DOC] Checks that get_local_rank without a mesh_dim on a 2D mesh raises
        # RuntimeError saying mesh_dim must be specified when ndim is above 1.
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
        ):
            mesh_2d.get_local_rank()

    @with_comms
    def test_get_local_rank(self):
        # [FUNCTION DOC] Checks get_local_rank by name equals by index on a 2D mesh, equals the
        # local rank of the sliced 1D meshes, and that a flattened dp/tp mesh reports the global
        # rank.
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )
        self.assertEqual(mesh_2d.get_local_rank("dp"), mesh_2d.get_local_rank(0))
        self.assertEqual(mesh_2d.get_local_rank("tp"), mesh_2d.get_local_rank(1))

        dp_mesh = mesh_2d["dp"]
        tp_mesh = mesh_2d["tp"]
        self.assertEqual(dp_mesh.get_local_rank(), mesh_2d.get_local_rank("dp"))
        self.assertEqual(tp_mesh.get_local_rank(), mesh_2d.get_local_rank("tp"))

        # Verify flattened mesh local rank correctness.
        flattened_mesh = mesh_2d["dp", "tp"]._flatten()
        self.assertEqual(flattened_mesh.get_local_rank(), self.rank)

    @with_comms
    def test_device_mesh_2d(self):
        # [FUNCTION DOC] Builds a 2x2 mesh from a tensor and checks each per-dimension group has
        # size 2 and contains the global ranks expected for the current rank along that
        # dimension.
        mesh_tensor = torch.arange(4).reshape(2, 2)
        # construct a device mesh for self.device_type
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()

        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < 2)
            dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

    @with_comms
    def test_device_mesh_init_backend(self):
        # [FUNCTION DOC] Builds a mesh with the backend initialization skipped and an explicit
        # rank, and checks get_group raises RuntimeError (groups not initialized) while
        # get_coordinate is still set.
        mesh = DeviceMesh(
            self.device_type, torch.arange(10), _init_backend=False, _rank=5
        )

        with self.assertRaisesRegex(RuntimeError, "process groups not initialized!"):
            mesh.get_group()

        # coordinates should always been populated when init_backend is False, as whenever
        # we call init_backend we should make sure the default pg already created
        self.assertEqual(mesh.get_coordinate(), [5])

    def test_fake_pg_device_mesh(self):
        # [FUNCTION DOC] Initializes a fake process group (single rank, fake store), builds a
        # mesh and checks an all-gather through the mesh yields a tensor whose first dim is
        # world_size times larger.
        fake_store = FakeStore()
        init_process_group("fake", store=fake_store, rank=0, world_size=self.world_size)
        device_type = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        mesh = DeviceMesh(device_type, torch.arange(self.world_size))

        local_tensor = torch.randn(2, 8)
        global_tensor = funcol.all_gather_single(
            local_tensor, gather_dim=0, group=(mesh, 0)
        ).wait()
        self.assertEqual(global_tensor.shape, (self.world_size * 2, 8))

    def test_fake_pg_device_mesh_cuda_on_cpu(self):
        # [FUNCTION DOC] Checks init_device_mesh with device type cuda works over a fake process
        # group even on CPU-only machines, with the expected ndim, size, dim names and fake
        # backend object.
        """
        Test that DeviceMesh can be initialized with fake backend using 'cuda'
        device type even on CPU-only machines.
        """
        fake_store = FakeStore()
        init_process_group("fake", store=fake_store, rank=0, world_size=1)

        # This should NOT fail even on CPU-only machines because
        # the fake backend skips device setup
        device_mesh = init_device_mesh(
            "cuda",
            (1,),
            mesh_dim_names=("dp",),
        )

        # Verify mesh is created correctly
        self.assertEqual(device_mesh.ndim, 1)
        self.assertEqual(device_mesh.size(), 1)
        self.assertEqual(device_mesh.mesh_dim_names, ("dp",))
        backend = device_mesh.get_all_groups()[0]._get_backend(torch.device("cuda"))
        self.assertIsInstance(backend, torch._C._distributed_c10d.FakeProcessGroup)

    @with_comms
    def test_from_group_with_global_pg(self):
        # [FUNCTION DOC] Checks DeviceMesh.from_group on a mesh's group, with and without an
        # explicit mesh tensor, reproduces the reference mesh, its group names and its
        # coordinates.
        # Simple test: check `from_group` from a mesh pg vs. directly
        # initializing via `init_device_mesh`
        ref_global_mesh = init_device_mesh(self.device_type, (self.world_size,))
        mesh_pg = ref_global_mesh.get_group()
        global_mesh = DeviceMesh.from_group(mesh_pg, self.device_type)
        self.assertEqual(ref_global_mesh, global_mesh)
        self.assertEqual(ref_global_mesh._dim_group_names, global_mesh._dim_group_names)
        self.assertEqual(
            ref_global_mesh._coordinate_on_dim, global_mesh._coordinate_on_dim
        )
        # Check when `mesh` is passed as well
        global_mesh = DeviceMesh.from_group(
            mesh_pg, self.device_type, mesh=torch.arange(self.world_size)
        )
        self.assertEqual(ref_global_mesh, global_mesh)
        self.assertEqual(ref_global_mesh._dim_group_names, global_mesh._dim_group_names)
        self.assertEqual(
            ref_global_mesh._coordinate_on_dim, global_mesh._coordinate_on_dim
        )

    @with_comms
    def test_from_group_with_invalid_mesh(self):
        # [FUNCTION DOC] Checks from_group raises ValueError for a mesh whose shape does not
        # match the group's ranks, and for a mesh whose ndim differs from the number of groups
        # given.
        # KNOWN ISSUE: the first call passes the module-level device_type, not self.device_type.
        global_pg = _get_default_group()
        global_pg_size = global_pg.size()
        if global_pg_size != 4:
            raise AssertionError("Test assumes global world size of 4")
        invalid_mesh = [[0, 1], [2, 3]]  # 2D mesh when we need 1D
        regex = r"Invalid mesh \[\[0, 1\], \[2, 3\]\] for ProcessGroup with ranks \[0, 1, 2, 3\]"
        with self.assertRaisesRegex(ValueError, regex):
            DeviceMesh.from_group(
                global_pg, device_type, invalid_mesh, mesh_dim_names=("dim0", "dim1")
            )

        device_mesh = init_device_mesh(self.device_type, (2, 2))
        groups = device_mesh.get_all_groups()
        invalid_mesh = (0, 1, 2, 3)  # 1D mesh when we need 2D
        regex = r"Expects mesh with ndim equal to number of ProcessGroups but got mesh \[0, 1, 2, 3\] and 2 ProcessGroups"
        with self.assertRaisesRegex(ValueError, regex):
            DeviceMesh.from_group(
                groups, self.device_type, invalid_mesh, mesh_dim_names=("dim0", "dim1")
            )

    def test_raises_invalid_device_type(self):
        # [FUNCTION DOC] Checks init_device_mesh raises RuntimeError when the device type string
        # includes an index (for example a device type followed by a colon and 0).
        with self.assertRaisesRegex(
            RuntimeError,
            "Device type with index is not supported",
        ):
            # test init_device_mesh with an invalid device type that contains a GPU index
            mesh_shape = (2, self.world_size // 2)
            init_device_mesh(
                f"{device_type}:0", mesh_shape=mesh_shape, mesh_dim_names=("dp", "tp")
            )

    @with_comms
    def test_get_root_mesh_multiple_independent_meshes(self):
        # [FUNCTION DOC] Regression test for issue 163330: with two independent 2D meshes,
        # get_root_mesh of each sliced submesh returns its own parent mesh and not the other
        # mesh.
        # regression test for issue #163330
        # when creating multiple independent device meshes and slicing them,
        # get_root_mesh should return the correct parent mesh for each submesh
        mesh1 = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp", "tp"),
        )
        mesh1_dp = mesh1["dp"]
        mesh1_tp = mesh1["tp"]

        mesh2 = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dim1", "dim2"),
        )
        mesh2_dim1 = mesh2["dim1"]
        mesh2_dim2 = mesh2["dim2"]

        self.assertEqual(_mesh_resources.get_root_mesh(mesh1_dp), mesh1)
        self.assertEqual(_mesh_resources.get_root_mesh(mesh1_tp), mesh1)
        self.assertEqual(_mesh_resources.get_root_mesh(mesh2_dim1), mesh2)
        self.assertEqual(_mesh_resources.get_root_mesh(mesh2_dim2), mesh2)

        self.assertNotEqual(_mesh_resources.get_root_mesh(mesh1_dp), mesh2)
        self.assertNotEqual(_mesh_resources.get_root_mesh(mesh1_tp), mesh2)


class DeviceMeshTestNDim(DTensorTestBase):
    # [CLASS DOC] Tests for DeviceMesh with three dimensions or more on 8 ranks: groups,
    # hashing, local ranks and from_group.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Test property giving the world size (8).
        return 8

    @with_comms
    def test_device_mesh_nd(self):
        # [FUNCTION DOC] Builds a 2x2x2 mesh and checks every per-dimension group has size 2 and
        # holds the global ranks that share the other coordinates.
        # construct a device mesh for self.device_type
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()

        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < mesh_tensor.ndim)
            dim_ranks = mesh_tensor.swapdims(-1, dim).reshape(-1, 2)

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            for ranks in dim_ranks:
                if self.rank in ranks:
                    self.assertEqual(global_ranks, ranks.tolist())

    @with_comms
    def test_device_mesh_hash(self):
        # [FUNCTION DOC] Checks that equal mesh tensors give equal hashes while meshes of
        # different shape differ.
        mesh_tensor_2d = torch.arange(8).reshape(4, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor_2d)
        mesh2 = DeviceMesh(self.device_type, mesh_tensor_2d)
        self.assertEqual(hash(mesh), hash(mesh2))
        mesh_tensor_3d = torch.arange(8).reshape(2, 2, 2)
        mesh3 = DeviceMesh(self.device_type, mesh_tensor_3d)
        self.assertNotEqual(hash(mesh), hash(mesh3))
        self.assertNotEqual(hash(mesh2), hash(mesh3))

    @with_comms
    def test_get_local_rank_3d(self):
        # [FUNCTION DOC] Checks get_local_rank by name on a dp/pp/tp 2x2x2 mesh against the
        # values computed from the global rank.
        """
        If we have a 3D mesh and we want to apply dp, pp, tp to it,
        mesh_dim_names = ["dp", "pp", "tp"], and the mesh tensor would be:
        mesh_3d_tensor = [
            [
                [0, 1],
                [2, 3],
            ],
            [
                [4, 5],
                [6, 7],
            ]

        ]
        """
        mesh_shape = (2, 2, 2)
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "pp", "tp")
        )

        # tp_rank_0: [0, 2, 4, 6], tp_rank_1: [1, 3, 5, 7]
        tp_rank = mesh_3d.get_local_rank("tp")
        expected_tp_rank = self.rank % 2
        self.assertEqual(tp_rank, expected_tp_rank)

        # pp_rank_0: [0, 1, 4, 5], pp_rank_1: [2, 3, 6, 7]
        pp_rank = mesh_3d.get_local_rank("pp")
        expected_pp_rank = 0 if self.rank % 4 <= 1 else 1
        self.assertEqual(pp_rank, expected_pp_rank)

        # dp_rank_0: [0, 1, 2, 3], dp_rank_1: [4, 5, 6, 7]
        dp_rank = mesh_3d.get_local_rank("dp")
        expected_dp_rank = self.rank // 4
        self.assertEqual(dp_rank, expected_dp_rank)

    @with_comms
    def test_device_mesh_parent_child_hash(self):
        # [FUNCTION DOC] Checks equality and hashing: a slice of a 2D mesh differs (names and
        # hash) from a standalone mesh over the same ranks, while two standalone meshes with the
        # same ranks are equal in layout, rank map, names, thread id and hash.
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("DP", "TP")
        )

        mesh_group_1 = torch.arange(0, self.world_size // 2)
        mesh_group_2 = torch.arange(self.world_size // 2, self.world_size)
        ep_mesh_1 = DeviceMesh(self.device_type, mesh_group_1)
        ep_mesh_2 = DeviceMesh(self.device_type, mesh_group_2)
        ep_mesh = ep_mesh_1 if self.rank < self.world_size // 2 else ep_mesh_2
        # ep_mesh is considered different from mesh_2d["TP"]
        self.assertEqual(
            mesh_2d["TP"].mesh.flatten().tolist(), ep_mesh.mesh.flatten().tolist()
        )
        self.assertEqual(mesh_2d["TP"]._layout, ep_mesh._layout)
        self.assertEqual(mesh_2d["TP"].mesh.shape, ep_mesh.mesh.shape)
        self.assertEqual(mesh_2d["TP"].device_type, ep_mesh.device_type)
        self.assertNotEqual(mesh_2d["TP"].mesh_dim_names, ep_mesh.mesh_dim_names)
        self.assertEqual(mesh_2d["TP"]._thread_id, ep_mesh._thread_id)
        self.assertNotEqual(hash(mesh_2d["TP"]), hash(ep_mesh))
        self.assertNotEqual(mesh_2d["TP"], ep_mesh)

        another_mesh_1 = DeviceMesh(self.device_type, mesh_group_1)
        another_mesh_2 = DeviceMesh(self.device_type, mesh_group_2)
        another_mesh = (
            another_mesh_1 if self.rank < self.world_size // 2 else another_mesh_2
        )
        # another_mesh is considered the same as ep_mesh
        self.assertEqual(ep_mesh._flatten_rank_map, another_mesh._flatten_rank_map)
        self.assertEqual(ep_mesh._layout, another_mesh._layout)
        self.assertEqual(ep_mesh.mesh.shape, another_mesh.mesh.shape)
        self.assertEqual(ep_mesh.device_type, another_mesh.device_type)
        self.assertEqual(ep_mesh.mesh_dim_names, another_mesh.mesh_dim_names)
        self.assertEqual(ep_mesh._thread_id, another_mesh._thread_id)
        self.assertEqual(hash(ep_mesh), hash(another_mesh))
        self.assertEqual(ep_mesh, another_mesh)

    @with_comms
    def test_from_group_with_mesh_shape_3d(self):
        # [FUNCTION DOC] Checks from_group with two groups taken from a 3D mesh and a mesh
        # tensor sliced at the local tp rank: group names, mesh tensors and sub-mesh groups
        # match the reference.
        """Tests ``from_group`` when passing ``mesh_shape`` as 3D."""
        # Consider the following 3D scenario and we need to create the 2D HSDP mesh from it.
        # - (2, 2, 2) ("dp_replicate", "dp_shard", "tp") mesh
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
        ref_mesh = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        dp_shard_group = ref_mesh["dp_shard"].get_group()
        dp_replicate_group = ref_mesh["dp_replicate"].get_group()

        dp_mesh = DeviceMesh.from_group(
            [dp_replicate_group, dp_shard_group],
            self.device_type,
            mesh=ref_mesh.mesh[:, :, ref_mesh.get_local_rank(mesh_dim="tp")],
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )

        ref_mesh_dp_dim_group_names = ref_mesh._dim_group_names[:2]
        self.assertEqual(ref_mesh_dp_dim_group_names, dp_mesh._dim_group_names[:2])
        # Cannot check directly for mesh equality since parent meshes are not
        # the same since the ref's parent mesh is 3D
        self.assertEqual(dp_mesh["dp_replicate"].mesh, ref_mesh["dp_replicate"].mesh)
        self.assertEqual(
            dp_mesh["dp_replicate"]._dim_group_names,
            ref_mesh["dp_replicate"]._dim_group_names,
        )
        self.assertEqual(dp_mesh["dp_shard"].mesh, ref_mesh["dp_shard"].mesh)
        self.assertEqual(
            dp_mesh["dp_shard"]._dim_group_names,
            ref_mesh["dp_shard"]._dim_group_names,
        )

    @with_comms()
    def test_from_group_with_mesh_shape_2d(self):
        # [FUNCTION DOC] Creates shard and replicate groups by hand for an 8-rank world, builds
        # a 2D mesh with from_group, and checks groups, mesh and sub-meshes equal those of a
        # reference mesh.
        """Tests ``from_group`` when passing ``mesh_shape`` as 2D."""
        # Consider the following scenario where the process group has been created,
        # but we need to create the 2D HSDP mesh from it later in the program.
        mesh_shape = (2, 4)
        mesh_dim_names = ("dp_replicate", "dp_shard")
        ref_mesh = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Create shard groups (e.g. (0, 1, 2, 3), (4, 5, 6, 7))
        # and assign the correct shard group to each rank
        shard_rank_lists = (
            list(range(self.world_size // 2)),
            list(range(self.world_size // 2, self.world_size)),
        )
        shard_groups = (
            new_group(shard_rank_lists[0]),
            new_group(shard_rank_lists[1]),
        )
        current_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )

        # Create replicate groups (for example, (0, 4), (1, 5), (2, 6), (3, 7))
        # and assign the correct replicate group to each rank
        current_replicate_group = None
        shard_factor = len(shard_rank_lists[0])
        for i in range(self.world_size // 2):
            replicate_group_ranks = list(range(i, self.world_size, shard_factor))
            replicate_group = new_group(replicate_group_ranks)
            if self.rank in replicate_group_ranks:
                current_replicate_group = replicate_group

        dp_mesh = DeviceMesh.from_group(
            [not_none(current_replicate_group), current_shard_group],
            self.device_type,
            mesh=ref_mesh.mesh,
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )

        for mesh_dim_group, ref_mesh_dim_group in zip(
            dp_mesh.get_all_groups(), ref_mesh.get_all_groups()
        ):
            mesh_dim_group_ranks = dist.get_process_group_ranks(mesh_dim_group)
            ref_mesh_dim_group_ranks = dist.get_process_group_ranks(ref_mesh_dim_group)
            self.assertEqual(mesh_dim_group_ranks, ref_mesh_dim_group_ranks)
        # check both the 2d mesh and the submeshes are exactly the same.
        self.assertEqual(dp_mesh, ref_mesh)
        self.assertEqual(dp_mesh["dp_replicate"], ref_mesh["dp_replicate"])
        self.assertEqual(dp_mesh["dp_shard"], ref_mesh["dp_shard"])


class InitDeviceMeshTest(DTensorTestBase):
    # [CLASS DOC] Tests for init_device_mesh argument handling and backend_override on 8 ranks.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Test property giving the world size (8).
        return 8

    @with_comms
    def test_init_device_mesh(self):
        # [FUNCTION DOC] Checks init_device_mesh equals a hand-built DeviceMesh and keeps the
        # dim names.
        mesh_shape = (2, 4)
        mesh_dim_names = ("DP", "TP")
        ref_mesh = DeviceMesh(
            self.device_type,
            torch.arange(8).view(mesh_shape),
            mesh_dim_names=mesh_dim_names,
        )

        # test init_device_mesh with mesh_dim_names
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        self.assertEqual(mesh_2d, ref_mesh)
        self.assertEqual(mesh_2d.mesh_dim_names, mesh_dim_names)

    @with_comms
    def test_raises_duplicate_mesh_dim_names(self):
        # [FUNCTION DOC] Checks init_device_mesh raises RuntimeError when mesh_dim_names
        # contains duplicates.
        with self.assertRaisesRegex(
            RuntimeError,
            "Each mesh_dim_name must be unique.",
        ):
            init_device_mesh(
                self.device_type,
                (2, 4),
                mesh_dim_names=["dp", "dp"],
            )

    @with_comms
    def test_raises_mesh_shape_mesh_dim_names_mismatch(self):
        # [FUNCTION DOC] Checks init_device_mesh raises RuntimeError when the lengths of
        # mesh_shape and mesh_dim_names differ.
        with self.assertRaisesRegex(
            RuntimeError,
            "mesh_shape and mesh_dim_names should have same length!",
        ):
            init_device_mesh(
                self.device_type,
                (8,),
                mesh_dim_names=["dp", "tp"],
            )

    def _test_backend_override_argument_dict_with_idx_and_backend(self):
        # [FUNCTION DOC] HELPER: Shared body for the lazy and eager backend_override tests.
        # Builds a 3D mesh with an index-keyed override (fake, and fake with options) and checks
        # backend names and options per dimension, then checks _flatten honors a
        # backend_override given as a name or as a name with options.
        opts = FakeProcessGroup.Options()
        opts.fake_option = 42

        mesh = init_device_mesh(
            self.device_type,
            (2, 2, 2),
            mesh_dim_names=("dp", "tp", "cp"),
            backend_override={0: "fake", 2: ("fake", opts)},
        )

        def get_opts(mesh: DeviceMesh, dim_idx: int) -> C10dBackend.Options:
            # [FUNCTION DOC] HELPER: Returns the options object of the backend of the group at
            # the given mesh dim.
            return (
                mesh.get_group(dim_idx)
                ._get_backend(torch.device(f"{self.device_type}:{self.rank}"))
                .options
            )

        # Fake pg only have BackendType as BackendType::CUSTOM.
        self.assertEqual(mesh.get_group(0)._get_backend_name(), "custom")
        self.assertNotEqual(mesh.get_group(1)._get_backend_name(), "custom")
        self.assertEqual(mesh.get_group(2)._get_backend_name(), "custom")

        self.assertIsNone(get_opts(mesh, 0))
        self.assertEqual(get_opts(mesh, 2).fake_option, 42)

        dp_tp_mesh = mesh["dp", "tp"]._flatten()
        dp_cp_mesh = mesh["dp", "cp"]._flatten(backend_override="fake")
        tp_cp_mesh = mesh["tp", "cp"]._flatten(backend_override=("fake", opts))

        self.assertNotEqual(dp_tp_mesh.get_group(0)._get_backend_name(), "custom")
        self.assertEqual(dp_cp_mesh.get_group(0)._get_backend_name(), "custom")
        self.assertEqual(tp_cp_mesh.get_group(0)._get_backend_name(), "custom")

        self.assertIsNone(get_opts(dp_cp_mesh, 0))
        self.assertEqual(get_opts(tp_cp_mesh, 0).fake_option, 42)

    @with_comms
    def test_backend_override_argument_dict_with_idx_and_backend_lazy(self):
        # [FUNCTION DOC] Runs the shared backend_override check with lazy group initialization.
        self._test_backend_override_argument_dict_with_idx_and_backend()

    @with_comms(eager_init=True)
    def test_backend_override_argument_dict_with_idx_and_backend_eager(self):
        # [FUNCTION DOC] Runs the shared backend_override check with eager group initialization.
        self._test_backend_override_argument_dict_with_idx_and_backend()

    @with_comms(backend="fake")
    def test_backend_override_argument_dict_with_name_and_options(self):
        # [FUNCTION DOC] Uses the fake default backend and a backend_override keyed by dim name
        # with only an options object; checks the option lands on that dim only, and on
        # flattened meshes made with a backend_override of options.
        opts = FakeProcessGroup.Options()
        opts.fake_option = 42

        mesh = init_device_mesh(
            self.device_type,
            (2, 2, 2),
            mesh_dim_names=("dp", "tp", "cp"),
            backend_override={"tp": opts},
        )

        def get_opts(mesh: DeviceMesh, dim_idx: int) -> C10dBackend.Options:
            # [FUNCTION DOC] HELPER: Returns the options object of the backend of the group at
            # the given mesh dim.
            return (
                mesh.get_group(dim_idx)
                ._get_backend(torch.device(f"{self.device_type}:{self.rank}"))
                .options
            )

        self.assertIsNone(get_opts(mesh, 0))
        self.assertEqual(get_opts(mesh, 1).fake_option, 42)
        self.assertIsNone(get_opts(mesh, 2))

        dp_tp_mesh = mesh["dp", "tp"]._flatten()
        dp_cp_mesh = mesh["dp", "cp"]._flatten(backend_override=opts)

        self.assertIsNone(get_opts(dp_tp_mesh, 0))
        self.assertEqual(get_opts(dp_cp_mesh, 0).fake_option, 42)

    @with_comms
    def test_backend_override_argument_errors(self):
        # [FUNCTION DOC] Checks init_device_mesh raises RuntimeError for a backend_override that
        # has both a name and an index for the same dim, an unknown dim name, or an invalid
        # integer key.
        with self.assertRaisesRegex(
            RuntimeError,
            "Found redundant dim index 0 and name dp in backend_override",
        ):
            init_device_mesh(
                self.device_type,
                (2, 4),
                mesh_dim_names=("dp", "tp"),
                backend_override={"dp": "foo", 0: "bar"},
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Found invalid keys in backend_override: got \['cp'\]",
        ):
            init_device_mesh(
                self.device_type,
                (2, 4),
                mesh_dim_names=("dp", "tp"),
                backend_override={"cp": "foo"},
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Found invalid keys in backend_override: got \[42\]",
        ):
            init_device_mesh(
                self.device_type,
                (2, 4),
                mesh_dim_names=("dp", "tp"),
                backend_override={42: "bar"},
            )


class TestDeviceMeshGetItem(DTensorTestBase):
    # [CLASS DOC] Tests DeviceMesh indexing (slicing by name) and flatten/unflatten/concatenate
    # on 8 ranks.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Test property giving the world size (8).
        return 8

    @with_comms
    def test_raises_no_mesh_dim_found(self):
        # [FUNCTION DOC] Checks slicing a mesh that has no dim names raises RuntimeError.
        with self.assertRaisesRegex(
            RuntimeError, "Cannot slice a DeviceMesh without mesh_dim_names!"
        ):
            mesh = init_device_mesh(self.device_type, (2, 4))
            mesh["DP"]

    @with_comms
    def test_raises_invalid_mesh_dim_name(self):
        # [FUNCTION DOC] Checks slicing with a name that is not a mesh dim name raises KeyError.
        child_mesh_dim_name = ("PP",)
        with self.assertRaisesRegex(KeyError, "Invalid mesh_dim_name"):
            mesh_dim_names = ("DP", "TP")
            mesh = init_device_mesh(
                self.device_type,
                (2, 4),
                mesh_dim_names=mesh_dim_names,
            )
            mesh[child_mesh_dim_name]

    @with_comms
    def test_get_item_2d(self):
        # [FUNCTION DOC] Checks slicing a 2D mesh by name gives the mesh tensor row for the
        # current rank.
        mesh_shape = (2, 4)
        mesh_dim_names = ("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        pg_ranks_by_dim_name = {}
        for mesh_dim_name in mesh_dim_names:
            mesh_dim = mesh_dim_names.index(mesh_dim_name)
            pg_ranks_by_dim_name[mesh_dim_name] = mesh_2d.mesh.swapdims(
                -1, mesh_dim
            ).reshape(-1, mesh_2d.mesh.size(mesh_dim))

        tp_mesh = mesh_2d["TP"]
        tp_group_idx = self.rank // 4
        self.assertEqual(tp_mesh.mesh, pg_ranks_by_dim_name["TP"][tp_group_idx])

        dp_group_idx = self.rank % 4
        self.assertEqual(mesh_2d["DP"].mesh, pg_ranks_by_dim_name["DP"][dp_group_idx])

    @with_comms
    def test_get_item_1d(self):
        # [FUNCTION DOC] Checks slicing a 1D named mesh by its own name returns an equal mesh,
        # and an unknown name raises KeyError.
        mesh = init_device_mesh(self.device_type, (8,), mesh_dim_names=("dp",))
        # Make sure slicing out 1D mesh from a 1D mesh works.
        dp_mesh = mesh["dp"]
        self.assertEqual(dp_mesh, mesh)

        with self.assertRaisesRegex(KeyError, "Invalid mesh_dim_name"):
            dp_mesh = mesh["dim0"]

    @with_comms
    def test_get_item_3d(self):
        # [FUNCTION DOC] Checks single-name and multi-name slices (list and tuple forms) of a 3D
        # mesh, and re- slicing a 2D sub-mesh by one name, give the expected rank groups.
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("Replicate", "Shard", "TP")
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        tp_group = [[0, 1], [2, 3], [4, 5], [6, 7]]
        tp_group_idx = int(self.rank / 2)
        self.assertEqual(mesh_3d["TP"].mesh.tolist(), tp_group[tp_group_idx])

        shard_group = [[0, 2], [1, 3], [4, 6], [5, 7]]
        shard_group_idx = self.rank % 2 + self.rank // 4 * 2
        self.assertEqual(mesh_3d["Shard"].mesh.tolist(), shard_group[shard_group_idx])

        replicate_group = [[0, 4], [1, 5], [2, 6], [3, 7]]
        replicate_group_idx = self.rank % 4
        self.assertEqual(
            mesh_3d["Replicate"].mesh.tolist(), replicate_group[replicate_group_idx]
        )

        # We support both UX for nD slicing.
        # mesh_3d[["Replicate", "Shard"]] or mesh_3d["Replicate", "Shard"]
        hsdp_mesh_1 = mesh_3d[["Replicate", "Shard"]]
        hsdp_mesh_2 = mesh_3d["Replicate", "Shard"]
        hsdp_group = [[[0, 2], [4, 6]], [[1, 3], [5, 7]]]
        hsdp_group_idx = self.rank % 2
        self.assertEqual(hsdp_mesh_1.mesh.tolist(), hsdp_group[hsdp_group_idx])
        self.assertEqual(hsdp_mesh_2.mesh.tolist(), hsdp_group[hsdp_group_idx])
        self.assertEqual(hsdp_mesh_1, hsdp_mesh_2)

        # Test slicing out 1D mesh from a sub-2D mesh.
        shard_mesh = hsdp_mesh_2["Shard"]
        self.assertEqual(shard_mesh.mesh.tolist(), shard_group[shard_group_idx])
        replicate_mesh = hsdp_mesh_2["Replicate"]
        self.assertEqual(
            replicate_mesh.mesh.tolist(), replicate_group[replicate_group_idx]
        )

    @with_comms
    def test_cache_and_reuse_submesh_slice_result(self):
        # [FUNCTION DOC] Checks the process group count in the world is unchanged after slicing
        # the tp dim.
        # KNOWN ISSUE: no dp slice is taken before the first count check, so the first assertion
        # compares the count with itself and the cache-reuse of the dp slice is not tested.
        mesh = init_device_mesh(self.device_type, (2, 4), mesh_dim_names=("dp", "tp"))

        ref_pg_count = _world.group_count

        # When we call the "dp" slice second time, it should not create any new pg.
        # As we are just using the cached result so the pg count should be the same.
        self.assertEqual(ref_pg_count, _world.group_count)

        # When we call the "tp" slice, it should not create a new pg, as the "tp" slice would
        # just reuse the parent mesh pg.
        mesh["tp"]
        self.assertEqual(_world.group_count, ref_pg_count)

    @with_comms
    def test_get_item_3d_noncontiguous_slicing(self):
        # [FUNCTION DOC] Checks slicing dp and cp from a 3D mesh puts the later name innermost
        # with the expected ranks, and that slicing names out of order raises KeyError.
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("dp", "pp", "cp")
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Slice order simply decides which mesh_dim sits on which mesh_dim.
        # For dp_cp_mesh, cp mesh is the innermost dimension.
        dp_cp_mesh = mesh_3d["dp", "cp"]
        expected_mesh_tensor = (
            torch.tensor([[0, 1], [4, 5]], dtype=torch.int)
            if self.rank in (0, 1, 4, 5)
            else torch.tensor([[2, 3], [6, 7]], dtype=torch.int)
        )
        dp_local_rank = dp_cp_mesh.get_local_rank("dp")
        self.assertEqual(dp_cp_mesh.mesh, expected_mesh_tensor)
        cp_mesh = mesh_3d["cp"]
        # Check on the current dp_local_rank, whether the cp mesh tensor is the same.
        self.assertEqual(dp_cp_mesh.mesh[dp_local_rank], cp_mesh.mesh)

        with self.assertRaisesRegex(
            KeyError,
            "Invalid mesh_dim_names",
        ):
            mesh_3d["cp", "dp"]

    @with_comms
    def test_flatten_mesh_1d(self):
        # [FUNCTION DOC] Checks _flatten on a one-dimensional mesh runs without error (no
        # assertions).
        mesh_shape = (4,)
        mesh_dim_names = ("default",)
        mesh_1d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        mesh_1d._flatten()

    @with_comms
    def test_flatten_mesh_3d(self):
        # [FUNCTION DOC] Checks _flatten on a 3D mesh: error for an existing dim name,
        # contiguous and non- contiguous flatten layouts and ranks, no new group on repeated
        # flatten, ordering errors on slicing, custom flatten name, and error when a flatten
        # name is reused.
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("dp", "cp", "tp")
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Test flatten into an existing mesh_dim_name inside the mesh
        with self.assertRaisesRegex(
            ValueError,
            "already exists for submesh of the DeviceMesh",
        ):
            mesh_3d._flatten("dp")

        # Test flatten contiguous dims
        dp_cp_mesh = mesh_3d["dp", "cp"]
        flattened_dp_cp_mesh = dp_cp_mesh._flatten()
        self.assertEqual(dp_cp_mesh.mesh.flatten(), flattened_dp_cp_mesh.mesh)
        self.assertEqual(flattened_dp_cp_mesh.mesh_dim_names[0], "dp_cp")
        self.assertEqual(flattened_dp_cp_mesh.get_group().group_desc, "mesh_dp_cp")
        root_mesh = dp_cp_mesh._get_root_mesh()
        self.assertEqual(root_mesh, mesh_3d)
        flatten_mesh_layout = root_mesh._flatten_mapping["dp_cp"]._layout
        self.assertEqual(flatten_mesh_layout, flattened_dp_cp_mesh._layout)
        self.assertEqual(
            flattened_dp_cp_mesh._layout.collapse().global_ranks(8),
            [[0, 2, 4, 6], [1, 3, 5, 7]],
        )
        # This should not raise as they are well separated
        mesh_3d["dp_cp", "tp"]

        ref_pg_count = _world.group_count
        # Calling flatten again should not create a new pg.
        flattened_dp_cp_mesh_2 = dp_cp_mesh._flatten()
        self.assertEqual(flattened_dp_cp_mesh, flattened_dp_cp_mesh_2)
        self.assertEqual(ref_pg_count, _world.group_count)

        # Test flatten non-contiguous dims
        dp_tp_mesh = mesh_3d["dp", "tp"]
        flattened_dp_tp_mesh = dp_tp_mesh._flatten()
        self.assertEqual(dp_tp_mesh.mesh.flatten(), flattened_dp_tp_mesh.mesh)
        self.assertEqual(flattened_dp_tp_mesh.mesh_dim_names[0], "dp_tp")
        root_mesh = dp_tp_mesh._get_root_mesh()
        self.assertEqual(root_mesh, mesh_3d)
        flatten_mesh_root_layout = root_mesh._flatten_mapping["dp_tp"]._layout
        self.assertEqual(flatten_mesh_root_layout, flattened_dp_tp_mesh._layout)
        self.assertEqual(
            flattened_dp_tp_mesh._layout.collapse().global_ranks(8),
            [[0, 1, 4, 5], [2, 3, 6, 7]],
        )
        with self.assertRaisesRegex(
            KeyError, "Mesh dim indices should be in ascending order"
        ):
            # dp_tp is partly "above" and partly "below" cp
            mesh_3d["dp_tp", "cp"]

        # Test flatten with a flattened mesh_dim_name
        cp_tp_mesh = mesh_3d["cp", "tp"]
        cp_tp_mesh._flatten("dummy")
        self.assertEqual(mesh_3d["dummy"].mesh_dim_names[0], "dummy")

        # Test flatten into an existing mesh_dim_name inside the mesh
        with self.assertRaisesRegex(
            ValueError,
            "dp already exists for submesh of the DeviceMesh",
        ):
            mesh_3d._flatten("dp")
        with self.assertRaisesRegex(
            ValueError,
            "Flatten mesh with mesh_dim_name dp_tp has been created before",
        ):
            mesh_3d["cp", "tp"]._flatten("dp_tp")

    @with_comms(eager_init=True)
    def test_flatten_mesh_4d(self):
        # [FUNCTION DOC] Eager-init test of flattening in a 4D mesh: flatten
        # dp_replicate/dp_shard/cp, flatten further into dp, check shapes, and a degenerate
        # all-ones mesh smaller than the world.
        mesh_shape = (2, 2, 2, 1)
        mesh_dim_names = ("dp_replicate", "dp_shard", "cp", "tp")
        mesh_4d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # flatten HSDP and CP into one mesh
        dp_cp_mesh = mesh_4d[mesh_dim_names[:3]]._flatten("dp_cp")
        # check flattened mesh integrity
        self.assertEqual(mesh_4d["dp_cp"].mesh.flatten(), dp_cp_mesh.mesh)
        # check flattened mesh dim names is correct
        self.assertEqual(dp_cp_mesh.mesh_dim_names, ("dp_cp",))
        # check flattened mesh dependency
        self.assertEqual(dp_cp_mesh._get_root_mesh(), mesh_4d)
        mesh_4d[mesh_dim_names[1:3]]._flatten("dp_shard_cp")
        mesh_4d["dp_replicate", "dp_shard_cp"]._flatten("dp")
        self.assertEqual(mesh_4d["dp", "tp"].mesh.shape, (8, 1))

        # Corner cases when slicing and flattening a mesh which is smaller than the world size.
        mesh_4d = init_device_mesh(
            self.device_type,
            mesh_shape=(1, 1, 1, 1),
            mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"),
        )
        mesh_4d["dp_shard", "cp"]._flatten("dp_cp")
        self.assertEqual(mesh_4d["dp_replicate", "dp_cp", "tp"].mesh.shape, (1, 1, 1))

    @with_comms
    def test_unflatten_mesh_2d(self):
        # [FUNCTION DOC] Checks _unflatten on a 2D mesh splits dim 0 into two named dims, keeps
        # the tp group, and that the new dim names cannot be sliced from the original root mesh
        # (KeyError).
        mesh_shape = (4, 2)
        mesh_dim_names = ("dp", "tp")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        unflatten_mesh = mesh_2d._unflatten(0, (2, 2), ("dp_shard", "dp_replicate"))
        self.assertEqual(
            unflatten_mesh.mesh_dim_names, ["dp_shard", "dp_replicate", "tp"]
        )
        self.assertEqual(mesh_2d["tp"].mesh, unflatten_mesh["tp"].mesh)
        self.assertEqual(mesh_2d["tp"].get_group(), unflatten_mesh["tp"].get_group())

        # Not supporting slicing out unflatten dim name from root mesh.
        with self.assertRaises(KeyError):
            self.assertEqual(mesh_2d["dp_shard"].mesh, unflatten_mesh["dp_shard"].mesh)

    @with_comms
    def test_unflatten_mesh_3d(self):
        # [FUNCTION DOC] Checks _unflatten from a world mesh into several layouts, shares equal
        # meshes between alternative splits, and nested unflatten. If NCCL is available, also
        # checks per-dim backend_override (fake and nccl with timeouts) by timing collectives on
        # the groups.
        # Test unflatten from a dummy world mesh, which is the case we need for Expert Parallelism(EP).
        global_mesh = init_device_mesh(
            self.device_type,
            (8,),
            mesh_dim_names=("world",),
        )
        non_ep_mesh = global_mesh._unflatten(0, (2, 2, 2), ("dp", "cp", "tp"))
        ep_mesh = global_mesh._unflatten(0, (2, 2, 2), ("dp", "ep", "ep_tp"))
        self.assertEqual(non_ep_mesh["cp"].mesh, ep_mesh["ep"].mesh)
        self.assertEqual(non_ep_mesh["tp"].mesh, ep_mesh["ep_tp"].mesh)
        mesh_3d = global_mesh._unflatten(0, (4, 2, 1), ("dp", "cp", "tp"))
        unflatten_mesh = mesh_3d._unflatten(0, (2, 2), ("dp_shard", "dp_replicate"))
        self.assertEqual(
            unflatten_mesh.mesh_dim_names, ["dp_shard", "dp_replicate", "cp", "tp"]
        )
        self.assertEqual(mesh_3d["tp"].mesh, unflatten_mesh["tp"].mesh)
        self.assertEqual(mesh_3d["tp"].get_group(), unflatten_mesh["tp"].get_group())
        self.assertEqual(mesh_3d["cp"].mesh, unflatten_mesh["cp"].mesh)
        self.assertEqual(mesh_3d["cp"].get_group(), unflatten_mesh["cp"].get_group())

        # Test unflatten with backend override set.
        if not _NCCL_AVAILABLE:
            return
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=30)
        mesh_2d = global_mesh._unflatten(
            0,
            (1, 8),
            ("pp", "spmd"),
            backend_override={"pp": "fake", "spmd": ("nccl", opts)},
        )
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=60)
        mesh_4d = mesh_2d._unflatten(
            1,
            (2, 2, 2),
            ("dp", "cp", "tp"),
            backend_override={"dp": "nccl", "cp": "nccl", "tp": ("nccl", opts)},
        )
        self.assertEqual(mesh_4d["pp"].get_group()._get_backend_name(), "custom")
        spmd_pg = mesh_2d["spmd"].get_group()
        self.assertEqual(spmd_pg._get_backend_name(), "nccl")
        w = spmd_pg.allreduce(torch.rand(10).cuda(self.rank))
        self.assertEqual(w.timeout, timedelta(seconds=30))
        w.wait()
        tp_pg = mesh_4d["tp"].get_group()
        self.assertEqual(tp_pg._get_backend_name(), "nccl")
        w = tp_pg.allreduce(torch.rand(10).cuda(self.rank))
        self.assertEqual(w.timeout, timedelta(seconds=60))
        w.wait()

    @with_comms
    def test_concatenate_2d(self):
        # [FUNCTION DOC] Checks DeviceMesh._concatenate of the dp and tp 1D slices equals the 2D
        # mesh and keeps the same groups.
        mesh_shape = (2, 4)
        mesh_dim_names = ("dp", "tp")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        concatenated_mesh = DeviceMesh._concatenate([mesh_2d["dp"], mesh_2d["tp"]])
        self.assertEqual(concatenated_mesh.mesh, mesh_2d.mesh)
        self.assertEqual(concatenated_mesh.get_group("dp"), mesh_2d.get_group("dp"))
        self.assertEqual(concatenated_mesh.get_group("tp"), mesh_2d.get_group("tp"))

    @with_comms
    def test_concatenate_3d(self):
        # [FUNCTION DOC] Checks _concatenate of 1D slices and of a 2D plus a 1D slice rebuilds
        # the same 3D mesh.
        mesh_shape = (2, 2, 2)
        mesh_dim_names = ("pp", "dp", "tp")
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        concatenated_mesh = DeviceMesh._concatenate([mesh_3d["dp"], mesh_3d["tp"]])
        dp_tp_mesh = mesh_3d["dp", "tp"]
        self.assertEqual(concatenated_mesh.mesh, dp_tp_mesh.mesh)
        self.assertEqual(concatenated_mesh.get_group("dp"), dp_tp_mesh.get_group("dp"))
        self.assertEqual(concatenated_mesh.get_group("tp"), dp_tp_mesh.get_group("tp"))
        self.assertEqual(
            mesh_3d, DeviceMesh._concatenate([mesh_3d["pp", "dp"], mesh_3d["tp"]])
        )

    @with_comms
    def test_reconstruct_mesh_with_flatten_dim(self):
        # [FUNCTION DOC] Checks that slicing with a previously flattened dim name (shard_cp or
        # dp_cp) gives the expected mesh tensors and reuses the flattened mesh's group.
        mesh_3d = init_device_mesh(
            self.device_type, (2, 2, 2), mesh_dim_names=("replicate", "shard", "cp")
        )
        shard_cp_mesh = mesh_3d["shard", "cp"]._flatten()
        hsdp_mesh = mesh_3d["replicate", "shard_cp"]
        expected_mesh_tensor = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int
        )
        self.assertEqual(hsdp_mesh.mesh, expected_mesh_tensor)
        self.assertEqual(shard_cp_mesh.get_group(), mesh_3d["shard_cp"].get_group())
        self.assertEqual(
            shard_cp_mesh.get_group(), mesh_3d.get_group(mesh_dim="shard_cp")
        )

        mesh_3d = init_device_mesh(
            self.device_type, (2, 2, 2), mesh_dim_names=("dp", "cp", "tp")
        )
        dp_cp_mesh = mesh_3d["dp", "cp"]._flatten()
        spmd_mesh = mesh_3d["dp_cp", "tp"]
        expected_mesh_tensor = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int
        )
        self.assertEqual(spmd_mesh.mesh, expected_mesh_tensor)
        self.assertEqual(dp_cp_mesh.get_group(), mesh_3d["dp_cp"].get_group())
        self.assertEqual(dp_cp_mesh.get_group(), mesh_3d.get_group(mesh_dim="dp_cp"))


class TestMeshEnv(DTensorTestBase):
    # [CLASS DOC] Tests the root-mesh bookkeeping helpers (_mesh_resources and DeviceMesh
    # private methods) on 8 ranks.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Test property giving the world size (8).
        return 8

    @with_comms
    def test_get_root_mesh(self):
        # [FUNCTION DOC] Checks get_root_mesh (module helper and mesh method) returns the
        # original 3D mesh for every 1D and 2D slice.
        mesh_3d = init_device_mesh(
            self.device_type,
            (2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        dp_cp_mesh = mesh_3d["dp", "cp"]
        dp_tp_mesh = mesh_3d["dp", "tp"]
        cp_tp_mesh = mesh_3d["cp", "tp"]
        dp_mesh = mesh_3d["dp"]
        cp_mesh = mesh_3d["cp"]
        tp_mesh = mesh_3d["tp"]
        # Test BC case is still working
        self.assertEqual(_mesh_resources.get_root_mesh(dp_cp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(dp_tp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(cp_tp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(dp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(cp_mesh), mesh_3d)
        self.assertEqual(_mesh_resources.get_root_mesh(tp_mesh), mesh_3d)
        self.assertEqual(dp_cp_mesh._get_root_mesh(), mesh_3d)
        self.assertEqual(dp_tp_mesh._get_root_mesh(), mesh_3d)
        self.assertEqual(cp_tp_mesh._get_root_mesh(), mesh_3d)
        self.assertEqual(dp_mesh._get_root_mesh(), mesh_3d)
        self.assertEqual(cp_mesh._get_root_mesh(), mesh_3d)
        self.assertEqual(tp_mesh._get_root_mesh(), mesh_3d)

    @with_comms
    def test_get_root_mesh_dim_exist(self):
        # [FUNCTION DOC] Checks _get_root_mesh_dim returns the dim index of a sliced mesh in the
        # root mesh.
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        self.assertEqual(mesh_2d["DP"]._get_root_mesh_dim(), 0)
        self.assertEqual(mesh_2d["TP"]._get_root_mesh_dim(), 1)

    @with_comms
    def test_get_root_mesh_dim_not_exist(self):
        # [FUNCTION DOC] Checks _get_root_mesh_dim returns None for a root mesh.
        mesh_shape = (self.world_size,)
        mesh = init_device_mesh(self.device_type, mesh_shape)

        self.assertEqual(mesh._get_root_mesh_dim(), None)

    @with_comms
    def test_get_mesh_dim_by_name(self):
        # [FUNCTION DOC] Checks _get_mesh_dim_by_name maps names to dim indices.
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        self.assertEqual(mesh_2d._get_mesh_dim_by_name("DP"), 0)
        self.assertEqual(mesh_2d._get_mesh_dim_by_name("TP"), 1)

    @with_comms
    def test_get_all_submeshes(self):
        # [FUNCTION DOC] Checks _get_all_submeshes on the replicate dim gives 4 meshes of 2
        # ranks each.
        mesh_2d = init_device_mesh(
            self.device_type,
            (2, 4),
            mesh_dim_names=("replicate", "shard"),
        )
        all_submeshes = mesh_2d._get_all_submeshes("replicate")
        self.assertEqual(len(all_submeshes), 4)
        self.assertEqual(
            all(submesh.mesh.numel() == 2 for submesh in all_submeshes), True
        )

    @with_comms
    def test_mesh_slice_fake_tensor_mode(self):
        # [FUNCTION DOC] Checks slicing by one name and by two names works under FakeTensorMode (no
        # assertion).
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        with FakeTensorMode():
            mesh_2d["DP"]
            mesh_2d["TP"]
            mesh_2d["DP", "TP"]


class DeviceMeshCollectiveTest(DTensorTestBase):
    # [CLASS DOC] Tests mesh collectives (broadcast, scatter, all-gather, reduce-scatter) and
    # torchcomms process group APIs on 8 ranks.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Test property giving the world size (8).
        return 8

    @with_comms
    def test_broadcast_1d(self):
        # [FUNCTION DOC] Checks mesh_broadcast on a 1D mesh makes every rank hold rank 0's
        # tensor in place.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        mesh_broadcast(local_tensor, mesh, mesh_dim=0)
        self.assertEqual(local_tensor, torch.zeros(3, 3))

    @with_comms
    def test_scatter_1d(self):
        # [FUNCTION DOC] Checks mesh_scatter on a 1D mesh along each tensor dim, with padded
        # contiguous chunks from Shard splitting, delivers each rank its own chunk.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        scatter_tensor_shape = [3, 3, 3]
        for scatter_dim in range(len(scatter_tensor_shape)):
            shard_placement = Shard(scatter_dim)
            scatter_tensor_shape[scatter_dim] *= self.world_size
            # make the random seed same across rank
            torch.manual_seed(0)
            global_tensor = torch.randn(scatter_tensor_shape, device=self.device_type)
            splitted_list, _ = shard_placement._split_tensor(
                global_tensor, mesh.size(), with_padding=True, contiguous=True
            )
            recv_tensor = torch.empty_like(splitted_list[mesh.get_rank()])
            # scatter on dim > 0 would generate non-contiguous tensor, verify that works
            mesh_scatter(recv_tensor, splitted_list, mesh, mesh_dim=0)
            self.assertEqual(recv_tensor, splitted_list[mesh.get_rank()])

    @with_comms
    def test_scatter_uneven(self):
        # [FUNCTION DOC] Checks mesh_scatter with a tensor not divisible by world size: scatters
        # padded chunks, removes padding, and compares with the chunk list (padding with empty
        # tensors).
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.randn(
            device_mesh.size() + 3, device_mesh.size() + 1, device=self.device_type
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)

            tensor_to_scatter = tensor_to_split.clone()
            tensor_splitted_list = list(
                torch.chunk(tensor_to_split, self.world_size, dim=shard_dim)
            )
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            scattered_tensor = torch.empty_like(padded_tensor_list[my_rank])
            mesh_scatter(scattered_tensor, padded_tensor_list, device_mesh, mesh_dim=0)

            if pad_sizes[my_rank] != 0:
                scattered_tensor = unpad_tensor(
                    scattered_tensor, shard_dim, pad_sizes[my_rank]
                )

            if scattered_tensor.numel() == 0:
                # We need to check numel() instead of size if a tensor is ([]) after unpadding,
                # since the size could be ([0, 8]) after unpadding.
                self.assertEqual(
                    scattered_tensor.numel(), tensor_splitted_list[my_rank].numel()
                )
            else:
                self.assertEqual(
                    scattered_tensor.size(), tensor_splitted_list[my_rank].size()
                )
                self.assertEqual(scattered_tensor, tensor_splitted_list[my_rank])

    @with_comms
    def test_all_gather_uneven(self):
        # [FUNCTION DOC] Checks all-gather of padded uneven shards, removing the padding per
        # chunk, rebuilds the original tensor along every dim.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.ones(
            device_mesh.size() + 3,
            device_mesh.size() + 1,
            device=self.device_type,
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_padded_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_split,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )
            local_tensor = tensor_padded_list[my_rank]
            big_tensor = funcol.all_gather_single(
                local_tensor, gather_dim=shard_dim, group=(device_mesh, 0)
            )
            big_tensor_chunks = list(
                torch.chunk(big_tensor, device_mesh.size(), dim=shard_dim)
            )
            unpadded_list = [
                (
                    unpad_tensor(big_tensor, shard_dim, pad_sizes[i])
                    if pad_sizes[i] > 0
                    else big_tensor
                )
                for i, big_tensor in enumerate(big_tensor_chunks)
            ]
            all_gathered_tensor = torch.cat(unpadded_list, dim=shard_dim)

            self.assertEqual(all_gathered_tensor.size(), tensor_to_split.size())
            self.assertEqual(all_gathered_tensor, tensor_to_split)

    @with_comms
    def test_reduce_scatter_contiguous(self):
        # [FUNCTION DOC] Checks a Partial to Shard(0) redistribution (reduce-scatter) gives
        # equal results for a non-contiguous slice and its contiguous copy, with the expected
        # shape and summed values.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()

        # Init the tensor
        step = self.world_size * 2
        total_elem = step**2
        tensor = torch.arange(0, total_elem).view(step, -1).to(device=self.device_type)
        tensor = tensor * (my_rank + 1)

        # Get non-contiguous tensor by slicing
        tensor_to_reduce = tensor[::2, :2]
        tensor_contiguous = tensor_to_reduce.clone().contiguous()

        # Partial to Shard to trigger reduce_scatter
        tensor_to_reduce = DTensor.from_local(
            tensor_to_reduce, device_mesh, [_Partial()]
        )
        tensor_contiguous = DTensor.from_local(
            tensor_contiguous, device_mesh, [_Partial()]
        )
        new_tensor = tensor_to_reduce.redistribute(device_mesh, [Shard(0)])
        new_tensor_contiguous = tensor_contiguous.redistribute(device_mesh, [Shard(0)])

        # The output for contiguous and non-contiguous tensors of the same value
        # should return the same reducescatter value.
        new_tensor_local = new_tensor._local_tensor
        new_tensor_contiguous_local = new_tensor_contiguous._local_tensor
        self.assertEqual(new_tensor_local, new_tensor_contiguous_local)
        self.assertEqual(list(new_tensor_local.size()), [1, 2])

        # Check the reduce numerical value
        sum_base = (1 + self.world_size) * self.world_size / 2
        first_elem = my_rank * sum_base * step * 2
        expected_tensor = torch.tensor(
            [[first_elem, first_elem + sum_base]],
            dtype=new_tensor_local.dtype,
            device=self.device_type,
        )
        self.assertEqual(new_tensor_local, expected_tensor)

    @with_comms
    def test_reduce_scatter_uneven(self):
        # [FUNCTION DOC] Checks reduce_scatter_single of padded uneven chunks, after removing
        # padding, gives the expected sum of ranks and the same shapes as torch.chunk.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = (
            torch.ones(
                device_mesh.size() + 3,
                device_mesh.size() + 1,
                device=self.device_type,
            )
            * self.rank
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_to_scatter = tensor_to_split.clone()

            tensor_splitted_list = list(
                torch.chunk(tensor_to_split, self.world_size, dim=shard_dim)
            )
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            tensor_to_reduce = torch.cat(padded_tensor_list, shard_dim)

            res_num = ((0 + self.world_size - 1) * self.world_size) / 2

            scattered_tensor = funcol.reduce_scatter_single(
                tensor_to_reduce,
                reduceOp="sum",
                scatter_dim=shard_dim,
                group=(device_mesh, 0),
            )

            # unpad scattered_tensor
            if pad_sizes[my_rank] > 0:
                scattered_tensor = unpad_tensor(
                    scattered_tensor, shard_dim, pad_sizes[my_rank]
                )

            if scattered_tensor.numel() == 0:
                # We need to check numel() instead of size if a tensor is ([]) after unpadding,
                # since the size could be ([0, 8]) after unpadding.
                self.assertEqual(
                    scattered_tensor.numel(), tensor_splitted_list[my_rank].numel()
                )
            else:
                self.assertEqual(
                    scattered_tensor.size(), tensor_splitted_list[my_rank].size()
                )
                self.assertEqual(
                    scattered_tensor,
                    torch.ones_like(tensor_splitted_list[my_rank]) * res_num,
                )

    @with_comms
    def test_broadcast_nd(self):
        # [FUNCTION DOC] Checks mesh_broadcast along each dim of a 2x2x2 mesh yields the value
        # of the first rank in that dim's group.
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            cloned_local_tensor = local_tensor.clone()
            mesh_broadcast(cloned_local_tensor, mesh, mesh_dim=dim)
            res_num = global_ranks[0]
            self.assertEqual(cloned_local_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_scatter_nd(self):
        # [FUNCTION DOC] Checks mesh_scatter along each dim of a 2x2x2 mesh delivers to each
        # rank the tensor holding its own rank number.
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]
            received_tensor = torch.empty_like(
                scattered_tensors[mesh.get_coordinate()[dim]]
            )
            mesh_scatter(received_tensor, scattered_tensors, mesh, mesh_dim=dim)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)

    @unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
    @dist_config.patch(use_torchcomms=True)
    @_with_torchcomm_env
    @with_comms(backend="cpu:gloo,cuda:nccl")
    def test_pg_api_w_torchcomms(self) -> None:
        # [FUNCTION DOC] Needs torchcomms (skipped otherwise). Creates a hybrid cpu gloo and
        # cuda nccl group and checks all_reduce for CPU and device tensors, a no-op split, and a
        # real split into two sub-groups, each dispatching by tensor device.
        ranks = list(range(self.world_size))
        pg = new_group(
            backend="cpu:gloo,cuda:nccl",
            ranks=ranks,
            group_desc="new_pg",
        )

        # Test CPU tensor
        cpu_tensor = torch.ones(3, 3)
        dist.all_reduce(cpu_tensor, group=pg)
        expected_cpu_tensor = torch.ones(3, 3) * self.world_size
        self.assertEqual(cpu_tensor, expected_cpu_tensor)

        # Test GPU tensor
        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=pg)
        expected_gpu_tensor = (
            torch.ones(3, 3, device=self.device_type) * self.world_size
        )
        self.assertEqual(gpu_tensor, expected_gpu_tensor)

        # No-op split should preserve both backends and produce identical
        # results to the parent.
        split_group_no_op = dist.split_group(pg, [ranks])
        cpu_tensor = torch.ones(3, 3)
        dist.all_reduce(cpu_tensor, group=split_group_no_op)
        self.assertEqual(cpu_tensor, expected_cpu_tensor)
        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=split_group_no_op)
        self.assertEqual(gpu_tensor, expected_gpu_tensor)

        # Real split with multiple sub-groups; each sub-group must keep both
        # cpu:gloo and cuda:nccl backends and dispatch to the right one based
        # on tensor device.
        pg_ranks_by_dim = torch.arange(self.world_size).view(2, 4)
        split_pg = dist.split_group(pg, pg_ranks_by_dim.tolist())
        expected_split_size = self.world_size // 2

        cpu_tensor = torch.ones(3, 3)
        dist.all_reduce(cpu_tensor, group=split_pg)
        self.assertEqual(cpu_tensor, torch.ones(3, 3) * expected_split_size)

        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=split_pg)
        self.assertEqual(
            gpu_tensor,
            torch.ones(3, 3, device=self.device_type) * expected_split_size,
        )

    @unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
    @dist_config.patch(use_torchcomms=True)
    @_with_torchcomm_env
    @with_comms(backend="nccl")
    def test_pg_api_w_torchcomms_nccl(self) -> None:
        # [FUNCTION DOC] Needs torchcomms. With an nccl group, checks all_reduce, a no-op split,
        # a split of the group by rank lists, and a direct split_group call whose comm is
        # manually appended to the world's comm list for finalization.
        ranks = list(range(self.world_size))
        pg = new_group(
            backend="nccl",
            ranks=ranks,
            group_desc="new_pg",
        )
        # Test GPU tensor
        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=pg)
        expected_gpu_tensor = (
            torch.ones(3, 3, device=self.device_type) * self.world_size
        )
        self.assertEqual(gpu_tensor, expected_gpu_tensor)
        # Double check if we do a no-op split, we get the same result
        split_group_no_op = dist.split_group(pg, [ranks])
        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=split_group_no_op)
        self.assertEqual(gpu_tensor, expected_gpu_tensor)

        # First way to do split, this is also how we do split within device mesh
        pg_ranks_by_dim = torch.arange(self.world_size).view(2, 4)
        split_group = dist.split_group(pg, pg_ranks_by_dim.tolist())
        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=split_group)
        expected_gpu_tensor = (
            torch.ones(3, 3, device=self.device_type) * self.world_size // 2
        )
        self.assertEqual(gpu_tensor, expected_gpu_tensor)

        # Second way to do split when we need to manually add comm into world
        pg_ranks_by_dim = torch.arange(self.world_size).view(2, 4)[self.rank // 4]
        split_group_2 = pg.split_group(
            pg_ranks_by_dim.tolist(), group_name="direct_split_test"
        )
        # This API directly calls the pybind API, so we need to manually track the comm for finalization.
        _world.comms.append(
            split_group_2._get_backend(
                torch.device(f"{self.device_type}:{self.rank}")
            ).get_comm()
        )
        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=split_group_2)
        self.assertEqual(gpu_tensor, expected_gpu_tensor)

    @unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
    @dist_config.patch(use_torchcomms=True)
    @_with_torchcomm_env
    @with_comms(eager_init=True, backend="cpu:gloo,cuda:nccl")
    def test_split_group_backend_filter_w_torchcomms(self) -> None:
        # [FUNCTION DOC] Needs torchcomms and eager init. Splits a hybrid parent keeping only
        # the cuda nccl backend and checks the stored backend config and all_reduce result;
        # checks ValueError for a mismatched backend name and for a device type absent from the
        # parent.
        # Hybrid parent (cpu:gloo + cuda:nccl); request only cuda:nccl in the
        # child via the `backend` arg. eager_init=True binds device_id=cuda so
        # nccl is the parent's default backend, satisfying the C++ check that
        # the filter must include the default backend's device.
        ranks = list(range(self.world_size))
        pg = dist.distributed_c10d._get_default_group()

        pg_ranks_by_dim = torch.arange(self.world_size).view(2, 4)
        ng = dist.split_group(pg, pg_ranks_by_dim.tolist(), backend="cuda:nccl")
        self.assertIsNotNone(ng)
        self.assertEqual(
            dist.distributed_c10d._world.pg_backend_config[ng], "cuda:nccl"
        )
        gpu_tensor = torch.ones(3, 3, device=self.device_type)
        dist.all_reduce(gpu_tensor, group=ng)
        expected_split_size = self.world_size // 2
        self.assertEqual(
            gpu_tensor,
            torch.ones(3, 3, device=self.device_type) * expected_split_size,
        )

        # Backend name mismatch (parent has cuda:nccl, request cuda:gloo).
        with self.assertRaisesRegex(ValueError, "Backend mismatch"):
            dist.split_group(pg, [ranks], backend="cuda:gloo")

        # Device type not present in parent.
        with self.assertRaisesRegex(ValueError, "is not present in the parent"):
            dist.split_group(pg, [ranks], backend="xpu:nccl")

    @unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
    @dist_config.patch(use_torchcomms=True)
    @_with_torchcomm_env
    @with_comms(backend="cpu:gloo,cuda:nccl")
    def test_device_mesh_w_torchcomms(self) -> None:
        # [FUNCTION DOC] Needs torchcomms. Builds a pp/dp/tp mesh, checks the dp local rank, an
        # all_reduce over the pp group and over the flattened mesh group, against expected sums.
        mesh_shape = (2, 2, self.world_size // 4)
        mesh_3d = init_device_mesh(
            self.device_type,
            mesh_shape,
            mesh_dim_names=("pp", "dp", "tp"),
        )
        dp_rank = mesh_3d.get_local_rank("dp")
        expected_dp_rank = 0 if self.rank % 4 <= 1 else 1
        self.assertEqual(dp_rank, expected_dp_rank)

        # test slicing and pg functionality
        pp_pg = mesh_3d["pp"].get_group()
        tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        dist.all_reduce(tensor, group=pp_pg)
        expected_tensor = torch.ones(3, 3) * 2 * ((self.rank % 4) + 2)
        self.assertEqual(tensor, expected_tensor)

        # test flatten functionality
        flatten_mesh = mesh_3d._flatten()
        flatten_pg = flatten_mesh.get_group()
        tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        dist.all_reduce(tensor, group=flatten_pg)
        expected_tensor = torch.ones(3, 3) * 28
        self.assertEqual(tensor, expected_tensor)

    @unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
    @dist_config.patch(use_torchcomms=True)
    @_with_torchcomm_env
    @with_comms(backend="cpu:gloo,cuda:nccl")
    def test_fake_backend_pg_names_w_torchcomms(self) -> None:
        # [FUNCTION DOC] Needs torchcomms. Unflattens a world mesh with one fake-backend dim and
        # checks that the fake dim's group name is hash-based and not an all-digit integer
        # string.
        """Fake-backend PG names must be hash-based when torchcomms is enabled.

        When torchcomms is enabled, split_group produces hash-based PG names
        for real backends. Fake-backend dimensions (from backend_override)
        must also use hash-based names; sequential integer names are not
        resolvable from compiled code via the C++ GroupRegistry.
        """
        world_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("world",)
        )
        # One fake dim and one real dim, mimicking torchtitan's
        # unflatten_mesh for disabled parallelism dimensions.
        mesh = world_mesh._unflatten(
            0,
            (1, self.world_size),
            ("fake_dim", "real_dim"),
            backend_override={"fake_dim": "fake"},
        )
        fake_pg_name = mesh._dim_group_names[0]
        self.assertFalse(
            fake_pg_name.isdigit(),
            lambda msg: f"{msg}\nFake-backend PG name '{fake_pg_name}' is a sequential integer; "
            f"expected a hash-based name for torchcomms compatibility.",
        )


class CuTeLayoutTest(TestCase):
    # [CLASS DOC] Single-process tests of _FlatLayout and _MeshLayout: coalescing, flattening,
    # complement, composition, orthogonality check and remapping onto rank tensors.
    def test_coalesce(self):
        # [FUNCTION DOC] Checks _FlatLayout merges contiguous (size, stride) pairs into one pair.
        # ((3,2),(2,1)) -> (6,1)
        l = _FlatLayout((3, 2), (2, 1))
        self.assertEqual(list(l.sizes_and_strides), [(6, 1)])

        # ((2,12),(3,4),(4,1)) -> (24,1)
        l = _FlatLayout((2, 3, 4), (12, 4, 1))
        self.assertEqual(list(l.sizes_and_strides), [(24, 1)])

    def test_coalesce_non_coalescible(self):
        # [FUNCTION DOC] Checks _FlatLayout keeps pairs separate when the stride does not match.
        # ((3,4),(2,1)) stays as-is (4 ≠ 2*1)
        l = _FlatLayout((3, 2), (4, 1))
        self.assertEqual(list(l.sizes_and_strides), [(3, 4), (2, 1)])

    def test_coalesce_singleton_dims(self):
        # [FUNCTION DOC] Checks size-one dims are dropped when coalescing, up to an empty result.
        # Dimensions with size=1 are removed, even if it's all of them
        l = _FlatLayout((1, 3, 1, 5, 1), (0, 10, 17, 2, 1))
        self.assertEqual(list(l.sizes_and_strides), [(15, 2)])

        l = _FlatLayout((1, 1, 1), (1, 42, 0))
        self.assertEqual(list(l.sizes_and_strides), [])

    def test_flatten(self):
        # [FUNCTION DOC] Checks nested size and stride tuples are flattened and then coalesced
        # when possible.
        l = _FlatLayout(((7, (5, 3)), 2), ((900, (36, 4)), 1))
        self.assertEqual(list(l.sizes_and_strides), [(7, 900), (5, 36), (3, 4), (2, 1)])

        l = _FlatLayout(((7, (5, 3)), 2), ((30, (6, 2)), 1))
        self.assertEqual(list(l.sizes_and_strides), [(7 * 5 * 3 * 2, 1)])

    def test_optional_strides(self):
        # [FUNCTION DOC] Checks omitting strides defaults to contiguous strides, which coalesce
        # into one pair.
        l = _FlatLayout((7, 5, 3, 2))
        # Strides default to suffix_product, which means they're coalescible
        self.assertEqual(list(l.sizes_and_strides), [(7 * 5 * 3 * 2, 1)])

    def test_mismatch_sizes_strides(self):
        # [FUNCTION DOC] Checks _FlatLayout raises ValueError when the sizes and strides
        # structures differ.
        with self.assertRaisesRegex(ValueError, "sizes .* and strides .* don't match"):
            _FlatLayout(42, (1,))

        with self.assertRaisesRegex(ValueError, "sizes .* and strides .* don't match"):
            _FlatLayout((3, 2), 1)

        with self.assertRaisesRegex(ValueError, "sizes .* and strides .* don't match"):
            _FlatLayout((5, 3, 2), (2, 1))

        with self.assertRaisesRegex(ValueError, "sizes .* and strides .* don't match"):
            _FlatLayout((5, (3, 2)), ((5, 3), 2))

    def test_complement_n_group_layout(self):
        # [FUNCTION DOC] Checks complement, all_ranks_from_zero and global_ranks for several
        # layouts and world sizes (8 and 16), including a layout whose strides are not in
        # decreasing order.
        # complement((4,2), 8) = (2,1); together form (8,1)
        pg_layout = _FlatLayout(
            (4,),
            (2,),
        )
        outer = pg_layout.complement(world_size=8)
        self.assertEqual(list(outer.sizes_and_strides), [(2, 1)])
        self.assertEqual(
            pg_layout.all_ranks_from_zero(),
            [0, 2, 4, 6],
        )
        groups = [
            [o + i for i in pg_layout.all_ranks_from_zero()]
            for o in outer.all_ranks_from_zero()
        ]
        self.assertEqual(
            groups,
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
            ],
        )
        self.assertEqual(
            pg_layout.global_ranks(8),
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
            ],
        )
        # complement((4,2), 16) = ((2,8), (2,1)); together form (16,1)
        outer = pg_layout.complement(world_size=16)
        self.assertEqual(list(outer.sizes_and_strides), [(2, 8), (2, 1)])
        self.assertEqual(
            outer.all_ranks_from_zero(),
            [0, 1, 8, 9],
        )
        self.assertEqual(
            pg_layout.global_ranks(16),
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [8, 10, 12, 14],
                [9, 11, 13, 15],
            ],
        )

        # Complement ((2,4), (2,1)) under world_size=16 → complement ((2,8), (2,2))
        pg_layout = _FlatLayout((2, 2), (4, 1))
        self.assertEqual(
            pg_layout.all_ranks_from_zero(),
            [0, 1, 4, 5],
        )
        outer = pg_layout.complement(world_size=16)
        self.assertEqual(list(outer.sizes_and_strides), [(2, 8), (2, 2)])
        self.assertEqual(
            outer.all_ranks_from_zero(),
            [0, 2, 8, 10],
        )
        self.assertEqual(
            pg_layout.global_ranks(16),
            [
                [0, 1, 4, 5],
                [2, 3, 6, 7],
                [8, 9, 12, 13],
                [10, 11, 14, 15],
            ],
        )

        # Test layout_to_global_ranks and layout_to_all_ranks_from_zero
        pg_layout = _FlatLayout((2, 2), (4, 2))
        self.assertEqual(
            pg_layout.all_ranks_from_zero(),
            [0, 2, 4, 6],
        )
        self.assertEqual(
            pg_layout.global_ranks(16),
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [8, 10, 12, 14],
                [9, 11, 13, 15],
            ],
        )
        outer = pg_layout.complement(world_size=16)
        self.assertEqual(list(outer.sizes_and_strides), [(2, 8), (2, 1)])
        # Test when stride is not monotonically decreasing, the complement layout
        # is same as the one sorted its stride.
        pg_layout_r = _FlatLayout((2, 2), (2, 4))
        outer = pg_layout_r.complement(world_size=16)
        self.assertEqual(list(outer.sizes_and_strides), [(2, 8), (2, 1)])
        self.assertEqual(
            pg_layout_r.global_ranks(16),
            [
                [0, 4, 2, 6],
                [1, 5, 3, 7],
                [8, 12, 10, 14],
                [9, 13, 11, 15],
            ],
        )

        # Test just all_ranks_from_zero and global_ranks.
        pg_layout = _FlatLayout((4,), (2,))
        self.assertEqual(
            pg_layout.all_ranks_from_zero(),
            [0, 2, 4, 6],
        )
        self.assertEqual(
            pg_layout.global_ranks(16),
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [8, 10, 12, 14],
                [9, 11, 13, 15],
            ],
        )

    def test_composition(self):
        # [FUNCTION DOC] Checks composition of a layout with another layout, including one that
        # yields two layouts, and that an incompatible right layout raises AssertionError.
        # self = ((4,2), (2,1)), l = (2,1)  → self o l = (2,1)
        orig_l = _FlatLayout((4, 2), (2, 1))
        right_l = _MeshLayout.from_sizes_strides((2,), (1,))
        (composed_layout,) = orig_l.composition(right_l)
        self.assertEqual(list(composed_layout.sizes_and_strides), [(2, 1)])
        self.assertEqual(
            composed_layout.global_ranks(8),
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ],
        )

        # self = (4,2), l = (2,1)  → self o l = (2,2)
        orig_l = _FlatLayout((4,), (2,))
        right_l = _MeshLayout.from_sizes_strides((2,), (1,))
        (composed_layout,) = orig_l.composition(right_l)
        self.assertEqual(list(composed_layout.sizes_and_strides), [(2, 2)])
        self.assertEqual(
            composed_layout.global_ranks(8),
            [
                [0, 2],
                [1, 3],
                [4, 6],
                [5, 7],
            ],
        )

        # self = (4,2), l = ((2,2), (2,1))  → self o l = ((2,4), (2,2))
        # This is to mimic the un-flatten from a 2D mesh to a 1D mesh.
        right_l = _MeshLayout.from_sizes_strides((2, 2), (2, 1))
        composed_layout = orig_l.composition(right_l)
        self.assertEqual(list(composed_layout[0].sizes_and_strides), [(2, 4)])
        self.assertEqual(list(composed_layout[1].sizes_and_strides), [(2, 2)])
        self.assertEqual(
            composed_layout[0].global_ranks(8),
            [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
        )
        self.assertEqual(
            composed_layout[1].global_ranks(8),
            [
                [0, 2],
                [1, 3],
                [4, 6],
                [5, 7],
            ],
        )

        # Error case.
        orig_l = _FlatLayout((4, 2), (4, 1))
        with self.assertRaises(
            AssertionError,
        ):
            right_l = _MeshLayout.from_sizes_strides((2,), (3,))
            orig_l.composition(right_l)

    def test_check_orthogonal(self):
        # [FUNCTION DOC] Checks check_orthogonal returns True for non-overlapping layouts
        # (including nested and reordered ones) and False for overlapping or incomparable
        # strides.
        """Test the check_orthogonal method for various layout configurations."""
        # Test 1: Valid layout - no overlap
        # sizes=(2,3), strides=(6,1) - stride 6 > span 3, so no overlap
        layout1 = _FlatLayout((2, 3), (6, 1))
        self.assertTrue(layout1.check_orthogonal())

        # Test 2: Invalid layout - overlap due to stride < previous span
        # sizes=(2,3), strides=(2,1) - stride 2 < span 3, causes overlap
        layout2 = _FlatLayout((2, 3), (2, 1))
        self.assertFalse(layout2.check_orthogonal())

        # Test 3: Invalid layout - duplicate strides
        # sizes=(2,3), strides=(1,1) - same stride, causes overlap
        layout3 = _FlatLayout((2, 3), (1, 1))
        self.assertFalse(layout3.check_orthogonal())

        # Test 4: Valid layout - single dimension
        layout4 = _FlatLayout((4,), (1,))
        self.assertTrue(layout4.check_orthogonal())

        # Test 5: Valid layout - exact boundary case
        # sizes=(2,3), strides=(3,1) - stride 3 == span 3, valid
        layout5 = _FlatLayout((2, 3), (3, 1))
        self.assertTrue(layout5.check_orthogonal())

        # Test 6: Valid layout - multi-dimensional with proper spacing
        layout6 = _FlatLayout((2, 2, 2), (8, 4, 1))
        self.assertTrue(layout6.check_orthogonal())

        # Test 7: Valid layout - stride not ordered
        layout7 = _FlatLayout((2, 2, 2), (4, 1, 2))
        self.assertTrue(layout7.check_orthogonal())

        # Test 8: Invalid layout - dimensions are not "comparable", neither
        # can be "stacked" above or below the other.
        layout8 = _FlatLayout((3, 2), (2, 3))
        self.assertFalse(layout8.check_orthogonal())

        # Test 9: Valid layout - dimensions can be dropped, shuffled, coalesced
        layout9 = _FlatLayout((2, (11, 3), 7), (1, (210, 2), 30))
        self.assertTrue(layout9.check_orthogonal())

    def test_remap_to_tensor(self):
        # [FUNCTION DOC] Checks remap_to_tensor maps a layout over an original rank tensor to a
        # tensor of actual ranks, for contiguous, shifted, column-major and uneven rank values.
        """Test the remap_to_tensor method for various scenarios."""
        # Test 1: Consecutive ranks, full world - should return logical groups directly
        original_mesh = torch.tensor([0, 1, 2, 3], dtype=torch.int)
        layout1 = _MeshLayout.from_sizes_strides((2, 2), (2, 1))  # row-major 2x2
        result1 = layout1.remap_to_tensor(original_mesh)
        expected1 = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int)
        self.assertEqual(result1, expected1)

        # Test 2: Non-consecutive ranks - should map to actual ranks
        original_mesh = torch.tensor([10, 20, 30, 40], dtype=torch.int)
        layout2 = _MeshLayout.from_sizes_strides((2, 2), (2, 1))
        result2 = layout2.remap_to_tensor(original_mesh)
        expected2 = torch.tensor([[[10, 20], [30, 40]]], dtype=torch.int)
        self.assertEqual(result2, expected2)

        # Test 4: 1D layout with consecutive ranks
        original_mesh = torch.tensor([0, 1, 2, 3], dtype=torch.int)
        layout4 = _MeshLayout.from_sizes_strides((4,), (1,))
        result4 = layout4.remap_to_tensor(original_mesh)
        expected4 = torch.tensor([[0, 1, 2, 3]], dtype=torch.int)
        self.assertEqual(result4, expected4)

        # Test 5: Complex strided layout with non-consecutive ranks
        original_mesh = torch.tensor([5, 10, 15, 20], dtype=torch.int)
        layout5 = _MeshLayout.from_sizes_strides((2, 2), (2, 1))
        result5 = layout5.remap_to_tensor(original_mesh)
        expected5 = torch.tensor([[[5, 10], [15, 20]]], dtype=torch.int)
        self.assertEqual(result5, expected5)

        # Test 6: Tensor Cute representation of a 2D mesh
        original_mesh = torch.tensor([0, 2, 1, 3], dtype=torch.int)
        layout6 = _MeshLayout.from_sizes_strides((2, 2), (1, 2))  # column-major style
        result6 = layout6.remap_to_tensor(original_mesh)
        expected6 = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int)
        self.assertEqual(result6, expected6)

        # Test 7: Layout with different stride pattern
        original_mesh = torch.tensor([0, 2, 1, 4], dtype=torch.int)
        layout7 = _MeshLayout.from_sizes_strides((2, 2), (1, 2))  # column-major style
        result7 = layout7.remap_to_tensor(original_mesh)
        expected7 = torch.tensor([[[0, 1], [2, 4]]], dtype=torch.int)
        self.assertEqual(result7, expected7)


class ProcessGroupOpaqueTypeTest(TestCase):
    # [CLASS DOC] Tests that the ProcessGroup members registered for the opaque type exist on
    # the class.
    """Test that ProcessGroup opaque type members are registered and exist on the class."""

    def test_registered_members_exist_on_process_group(self):
        # [FUNCTION DOC] Checks each registered member name has a member type registered for
        # ProcessGroup and is an attribute of the ProcessGroup class.
        from torch._library.opaque_object import get_member_type

        # Every member registered in _register_process_group_opaque_type()
        # must actually exist on ProcessGroup. This catches renames or
        # removals of C++ attributes that would cause torch.compile
        # (fullgraph=True) to silently register a stale name while the
        # real attribute has moved.
        registered_members = [
            "size",
            "rank",
            "_get_backend_name",
            "group_name",
            "group_desc",
            "__eq__",
        ]
        for member_name in registered_members:
            self.assertIsNotNone(
                get_member_type(ProcessGroup, member_name),
                lambda msg: f"{msg}\n'{member_name}' is not registered as a ProcessGroup opaque "
                f"type member. Add it to _register_process_group_opaque_type() "
                f"in torch/distributed/distributed_c10d.py",
            )
            self.assertTrue(
                hasattr(ProcessGroup, member_name),
                lambda msg: f"{msg}\n'{member_name}' is registered as a ProcessGroup opaque type "
                f"member but does not exist on the ProcessGroup class. "
                f"Was it renamed or removed?",
            )


if __name__ == "__main__":
    run_tests()
